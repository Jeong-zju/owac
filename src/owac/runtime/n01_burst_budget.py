"""Continue budget-limited baselines without changing data or optimization.

The original completed run stays immutable. Only max_steps changes in a child
run; optimizer, scheduler, RNG and stream position resume from the parent. A
settled run is never extended and an already consumed test cannot be retuned.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from owac.evaluation.n01_burst_training import atomic_json, train, training_sources


def continue_budget(parent, output, maximum):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if parent == output:
        raise ValueError("continuation must preserve the parent run")
    result = json.loads((parent / "result.json").read_text())
    if result["stop_reason"] != "budget_limit" or result["test"] is not None:
        raise ValueError("only untested budget-limited runs can be continued")
    if maximum <= result["protocol"]["max_steps"] or maximum >= 100000:
        raise ValueError("continuation must increase only the bounded step budget")
    if result["sources"] != training_sources():
        raise ValueError("parent execution source differs; no silent source migration")
    output.mkdir(parents=True, exist_ok=True)
    protocol = copy.deepcopy(result["protocol"])
    protocol["max_steps"] = maximum
    if not (output / "run.json").exists():
        if any(output.iterdir()):
            raise FileExistsError("new continuation output must be empty")
        run = json.loads((parent / "run.json").read_text())
        run["protocol"] = protocol
        run["budget_continuation"] = {
            "parent": str(parent),
            "parent_result_sha256": hashlib.sha256(
                (parent / "result.json").read_bytes()
            ).hexdigest(),
            "parent_max_steps": result["protocol"]["max_steps"],
            "new_max_steps": maximum,
            "changed_protocol_fields": ["max_steps"],
        }
        for name in ["best.pt", "last.pt"]:
            shutil.copy2(parent / name, output / name)
        atomic_json(output / "run.json", run)
    elif json.loads((output / "run.json").read_text())["protocol"] != protocol:
        raise ValueError("existing continuation has a different protocol")
    seed = protocol["seeds"][0]
    method = protocol["methods"][0]
    return train(protocol, seed, method, output, resume=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", required=True, type=int)
    args = parser.parse_args()
    continue_budget(args.parent, args.output, args.max_steps)


if __name__ == "__main__":
    main()

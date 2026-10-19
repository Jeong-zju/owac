"""Recompute live RAW policy execution, input identity and completed-episode receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def audit(root):
    records, first = [], None
    for path in sorted(root.glob("*-step*/result.json")):
        data = json.loads(path.read_text())
        if not data["results"]:
            continue  # Preserved infrastructure failures are not task failures.
        case = path.parent
        actions = read_rows(case / "actions.jsonl")
        predicted = read_rows(case / "policy/inference.jsonl")
        lookup = {r["step"]: np.array(r["actions"], dtype=np.float32) for r in predicted}
        delta = max(
            float(
                np.max(
                    np.abs(
                        np.array(a["action"], dtype=np.float32)
                        - lookup[a["step"] // 50 * 50][a["step"] % 50]
                    )
                )
            )
            for a in actions
        )
        with np.load(next((case / "policy").glob("*_first_input.npz"))) as f:
            arrays = {key: f[key].copy() for key in f.files}
        if first is None:
            first = arrays
        diffs = {key: float(np.max(np.abs(arrays[key] - first[key]))) for key in arrays}
        load = json.loads((case / "policy/load_audit.json").read_text())
        record = {
            "case_id": data["case_id"],
            "success": data["results"][0]["success"],
            "num_steps": len(actions),
            "prediction_calls": len(predicted),
            "execution_max_abs_difference": delta,
            "history_causal": all(
                max(r["history_steps"]) == r["step"]
                and min(r["history_steps"]) >= max(0, r["step"] - 7)
                for r in predicted
            ),
            "same_initial_input_max_abs_difference": diffs,
            "language": predicted[0]["language"],
            "checkpoint_sha256": load["checkpoint_sha256"],
            "strict_r7_load": not load["legacy_weights_probe"]
            and load["execution_revision"] == load["runtime_execution_revision"],
            "first_raw_hashes": predicted[0]["raw_input_sha256"],
            "files_sha256": {
                f: hashlib.sha256((case / f).read_bytes()).hexdigest()
                for f in [
                    "result.json",
                    "actions.jsonl",
                    "policy/inference.jsonl",
                    "policy/load_audit.json",
                    "episode0.mp4",
                ]
            },
        }
        assert delta == 0 and record["history_causal"] and max(diffs.values()) == 0
        assert record["strict_r7_load"] and all(r["finite"] for r in predicted)
        assert len(actions) == data["results"][0]["num_steps"]
        assert ("Success!" in (case / "run.log").read_text()) == record["success"]
        records.append(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "completed": len(result),
                "success": sum(r["success"] for r in result),
                "actual_steps": sum(r["num_steps"] for r in result),
            }
        )
    )

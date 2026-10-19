"""Run one seed's DBSR baselines under an external process supervisor."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path


def run_seed(config, seed, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(config).read_text())
    with (root / f"seed-{seed}.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another worker already owns this seed") from exc
        for method in protocol["methods"]:
            output = root / f"{seed}-{method}"
            result_path = output / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                expected = dict(protocol, seeds=[seed], methods=[method])
                if result["protocol"] != expected:
                    raise ValueError("existing result belongs to a different protocol")
                continue
            command = [
                sys.executable,
                "-m",
                "owac.evaluation.n01_burst_training",
                "--config",
                str(Path(config).resolve()),
                "--seed",
                str(seed),
                "--method",
                method,
                "--output",
                str(output.resolve()),
            ]
            if (output / "last.pt").exists():
                command.append("--resume")
            print(json.dumps({"seed": seed, "method": method, "action": "start"}), flush=True)
            subprocess.run(command, check=True)
        rows = [
            json.loads((root / f"{seed}-{method}" / "result.json").read_text())
            for method in protocol["methods"]
        ]
        status = {
            "seed": seed,
            "baseline_training_finished": True,
            "all_baselines_operationally_settled": all(r["operationally_settled"] for r in rows),
            "methods": [
                {k: row[k] for k in ["steps", "best_step", "stop_reason"]} | {"method": method}
                for method, row in zip(protocol["methods"], rows, strict=True)
            ],
        }
        (root / f"seed-{seed}-status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run_seed(args.config, args.seed, args.root)


if __name__ == "__main__":
    main()

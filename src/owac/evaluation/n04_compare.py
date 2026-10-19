"""Reproducible trained-control sweep; outputs metrics, never task-success claims."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    from owac.baselines.n04 import BASELINE_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001, 10002])
    args = parser.parse_args(argv)
    records = []
    for name in ["candidate", *BASELINE_NAMES, "frozen"]:
        for seed in args.seeds:
            checkpoint = Path(args.root) / name / "checkpoint.pt"
            output = checkpoint.parent / f"eval-{seed}"
            command = [
                sys.executable,
                "-m",
                "owac.evaluation.n04",
                "--config",
                args.config,
                "--checkpoint",
                str(checkpoint),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
            if name not in ("candidate", "frozen"):
                command += ["--baseline", name]
            command += ["--device", args.device]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            payload = json.loads((output / "mechanism.json").read_text())
            conditions = [
                {k: v for k, v in c.items() if k != "trajectory"}
                for c in payload["report"]["conditions"]
            ]
            record = {
                "name": name,
                "seed": seed,
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "location_mean_px": statistics.fmean(c["location_error"] for c in conditions),
                "contrast": payload["report"]["contrast"],
                "conditions": conditions,
                "counterfactuals": payload["report"]["counterfactuals"],
            }
            records.append(record)
            print(f"{name} seed={seed} location={record['location_mean_px']:.4f}px", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

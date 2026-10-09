"""Fetch only small numerical/provenance artifacts from a cloud baseline run."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from owac.evaluation.baseline_protocol import file_sha256, write_json


def verify_completed(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    training = json.loads((directory / "training.json").read_text())
    summary = json.loads((directory / "evaluation/summary.json").read_text())
    records = [
        json.loads(line)
        for line in (directory / "evaluation/episodes.jsonl").read_text().splitlines()
    ]
    seeds = [item["seed"] for item in records]
    if len(seeds) != 100 or seeds != protocol["test_seeds"]:
        raise ValueError("Archive must contain each of the 100 prescribed test seeds in order")
    if summary["status"] != "complete" or summary["episodes"] != 100:
        raise ValueError("Evaluation summary is incomplete")
    if summary["successes"] != sum(item["success"] for item in records):
        raise ValueError("Evaluation success count does not match per-episode evidence")
    if summary["episodes_sha256"] != file_sha256(directory / "evaluation/episodes.jsonl"):
        raise ValueError("Evaluation checksum mismatch")
    if training["smoke_test"] or not training["convergence_observed"]:
        raise ValueError("Formal converged training evidence is required")
    if training["protocol_sha256"] != file_sha256(directory / "protocol.json"):
        raise ValueError("Training protocol differs from the archived protocol")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="root@212.51.142.37")
    parser.add_argument("--port", type=int, default=16406)
    parser.add_argument("--run-root", default="/workspace/owac/runs/pi05-lw-mug-v1")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -p {args.port} -o BatchMode=yes",
            "--include=*/",
            "--include=*.json",
            "--include=*.jsonl",
            "--include=*.patch",
            "--include=*.txt",
            "--exclude=*",
            f"{args.host}:{shlex.quote(args.run_root + '/results/')}",
            str(args.destination) + "/",
        ],
        check=True,
    )
    status = json.loads((args.destination / "status.json").read_text())
    if status["status"] == "complete":
        verify_completed(args.destination)
    write_json(
        args.destination / "archive.json",
        {
            "archived_unix": time.time(),
            "remote_run_root": args.run_root,
            "status": status["status"],
            "checkpoints_downloaded": False,
            "sha256": {
                str(path.relative_to(args.destination)): file_sha256(path)
                for path in sorted(args.destination.rglob("*"))
                if path.is_file() and path.name != "archive.json"
            },
        },
    )
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()

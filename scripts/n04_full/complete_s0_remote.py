"""Sequential bounded live collection in an immutable snapshot; no network training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    source = args.source.resolve()
    frozen = Path("/workspace/owac/rawvla/source")
    plan = json.loads((source / "configs/n04/full/s0_completion.json").read_text())
    report = dict(source=str(source), started=time.time(), commands=[], plan=plan)
    if args.resume:
        report = json.loads((args.output / "run.json").read_text())
        if report["plan"] != plan:
            raise ValueError("resume plan differs from original frozen plan")
        report.setdefault("resume_sources", []).append(str(source))
    report.setdefault("resolved_plan", copy.deepcopy(plan))
    report["scene_rejection_policy"] = (
        "Only setup UnStableError may resample a fresh scene: seed + 1000*k, k=1..5; "
        "keep failed attempts and same split slot; never filter task success or model metrics"
    )
    for index, original_entry in enumerate(plan["episodes"]):
        entry = dict(original_entry)
        existing = args.output / f"episode-{index:03d}" / "record.json"
        if existing.exists():
            previous = json.loads(existing.read_text())
            if len(previous["frames"]) != plan["frames_per_episode"]:
                raise ValueError("partial episode requires explicit recovery in a new directory")
            continue

        platform = entry["platform"]
        name = f"episode-{index:03d}"
        interpreter = (
            "/workspace/owac/rawvla/envs/"
            + ("starvla" if platform == "libero" else "robotwin")
            + "/bin/python"
        )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            LIBERO_CONFIG_PATH=str(frozen / ".local/libero"),
            PYTHONPATH=":".join(
                [
                    str(source / "src"),
                    str(frozen / "experiments/simulation/libero"),
                    str(frozen / "starVLA"),
                    str(frozen / "third_party/LIBERO"),
                ]
            ),
        )
        command = [
            interpreter,
            "-m",
            "owac.raw.n04_live_collection",
            "--platform",
            platform,
            "--task",
            entry["task"],
            "--seed",
            str(entry["seed"]),
            "--source-id",
            entry["source_trajectory_id"],
            "--split",
            entry["split"],
            "--frames",
            str(plan["frames_per_episode"]),
            "--output",
            str(args.output / name),
        ]
        if entry.get("source"):
            command += ["--source", entry["source"]]
        for attempt in range(6):
            if attempt:
                entry["seed"] = original_entry["seed"] + 1000 * attempt
                entry["source_trajectory_id"] = f"n04ext/robotwin2/{entry['task']}/{entry['seed']}"
                command[command.index("--seed") + 1] = str(entry["seed"])
                command[command.index("--source-id") + 1] = entry["source_trajectory_id"]
            log_path = args.output / f"{name}-resume-{attempt}.log"
            if log_path.exists():
                raise FileExistsError(log_path)
            start = time.time()
            with log_path.open("w") as log:
                run = subprocess.run(
                    command,
                    env=env,
                    cwd=source,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
            report["commands"].append(
                dict(
                    name=name,
                    command=list(command),
                    exit_code=run.returncode,
                    seconds=time.time() - start,
                    attempt=attempt,
                    log=str(log_path),
                )
            )
            if run.returncode == 0:
                report["resolved_plan"]["episodes"][index] = dict(entry)
            (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, platform, attempt, run.returncode, flush=True)
            if run.returncode == 0:
                break
            if (
                platform != "robotwin2"
                or "UnStableError" not in log_path.read_text()
                or (args.output / name).exists()
                or attempt == 5
            ):
                raise SystemExit(run.returncode)
    report["finished"] = time.time()
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

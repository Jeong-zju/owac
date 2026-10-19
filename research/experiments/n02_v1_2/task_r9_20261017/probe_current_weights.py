"""Repeat the same fixed sensor controls on the actual new frontend and selected control."""

import subprocess
import sys
from pathlib import Path

root = Path("/workspace/owac/worktrees/n02-measurement-completion")
probe = root / "research/experiments/n02_v1_2/task_r9_20261017/noise_probe.py"
for arm, step in [("task", 48), ("control", 32)]:
    subprocess.run(
        [
            sys.executable,
            str(probe),
            "--config",
            f"configs/encoders/n02_v12/raw_task_r9_{arm}_robotwin.json",
            "--checkpoint",
            f"outputs/n02-v12/P2/raw-task-r9-{arm}-s1103-a1/checkpoint_{step:07d}.pt",
            "--captures",
            "outputs/n02-v12/task-r9-noise-capture-a1",
            "--output",
            f"outputs/n02-v12/task-r9-noise-probes/r9-{arm}-{step}-a1",
        ],
        cwd=root,
        check=True,
    )

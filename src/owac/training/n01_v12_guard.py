"""Persistent user/audit holds are checked by every training entry point."""

from pathlib import Path


def assert_training_allowed(repo_root, run_root):
    paths = [
        Path(repo_root) / "configs/encoders/n01_full/v1_2/TRAINING_HOLD.json",
        Path(run_root) / "P3_AUDIT_HOLD.json",
    ]
    for path in paths:
        if path.is_file():
            raise RuntimeError(f"training is disabled by persistent hold: {path}")

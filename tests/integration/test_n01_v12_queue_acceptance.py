"""Execute the production shell queue with deterministic, GPU-free child programs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fail_first", [False, True])
def test_handover_reaches_second_platform_and_reports_failure(tmp_path, fail_first):
    fake_repo = tmp_path / "repo"
    config = fake_repo / "configs/encoders/n01_full/v1_2/complete"
    config.mkdir(parents=True)
    for platform in ("libero", "robotwin2"):
        (config / f"{platform}-raw_n01-seed101.json").write_text("{}")
    child = tmp_path / "python"
    child.write_text("""#!/bin/bash
if [[ "$1" == "-c" ]]; then exit 0; fi
printf '%s\\n' "$*" >> "$QUEUE_TEST_LOG"
if [[ "$FAIL_FIRST" == "1" && "$*" == *"libero-raw_n01"* ]]; then exit 3; fi
""")
    child.chmod(0o700)
    wrapper = tmp_path / "wrap"
    wrapper.write_text('#!/bin/bash\nshift\nexec "$@"\n')
    wrapper.chmod(0o700)
    log = tmp_path / "calls"
    env = dict(
        os.environ,
        ROLE="raw",
        N01_REPO=str(fake_repo),
        N01_RUN_ROOT=str(tmp_path),
        N01_PYTHON=str(child),
        N01_GPU_WRAPPER=str(wrapper),
        QUEUE_TEST_LOG=str(log),
        FAIL_FIRST=str(int(fail_first)),
    )
    script = ROOT / "configs/backends/n01-v12-runtime/p3-queue.sh"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == int(fail_first), result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 2
    assert "libero-raw_n01" in calls[0]
    assert "robotwin2-raw_n01" in calls[1]
    assert "--batch-size 2 --precision bf16" in calls[0]
    assert "--flow-batch-size 8 --prefetch 2" in calls[0]
    # A persisted audit hold forbids every child launch, including preflight.
    log.unlink()
    (tmp_path / "P3_AUDIT_HOLD.json").write_text('{"reason":"independent audit"}')
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 42
    assert not log.exists()


def test_reboot_skips_only_matching_completed_receipt(tmp_path):
    import json
    import sys

    fake_repo = tmp_path / "repo"
    configs = fake_repo / "configs/encoders/n01_full/v1_2/complete"
    configs.mkdir(parents=True)
    config = {"training": {"total_steps": 50000}}
    for platform in ("libero", "robotwin2"):
        (configs / f"{platform}-raw_n01-seed101.json").write_text(json.dumps(config))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "train-libero-raw_n01-seed101.json").write_text(
        json.dumps({"config": config, "optimizer_steps": 50000})
    )
    (receipts / "train-robotwin2-raw_n01-seed101.json").write_text(
        json.dumps({"config": config, "optimizer_steps": 49999})
    )
    child = tmp_path / "python"
    child.write_text(
        '#!/bin/bash\nif [[ "$1" == "-c" ]]; then\n'
        '  if [[ "$2" == *optimizer_steps* ]]; then exec "$REAL_PY" "$@"; fi\n'
        '  exit 0\nfi\nprintf "%s\\n" "$*" >> "$QUEUE_TEST_LOG"\n'
        'test "$N01_SOURCE" = "$N01_REPO"\n'
    )
    child.chmod(0o700)
    wrapper = tmp_path / "wrap"
    wrapper.write_text('#!/bin/bash\nshift\nexec "$@"\n')
    wrapper.chmod(0o700)
    log = tmp_path / "calls"
    result = subprocess.run(
        ["bash", str(ROOT / "configs/backends/n01-v12-runtime/p3-queue.sh")],
        env=dict(
            os.environ,
            ROLE="raw",
            N01_REPO=str(fake_repo),
            N01_RUN_ROOT=str(tmp_path),
            N01_PYTHON=str(child),
            N01_GPU_WRAPPER=str(wrapper),
            QUEUE_TEST_LOG=str(log),
            REAL_PY=sys.executable,
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 1 and "robotwin2-raw_n01" in calls[0]

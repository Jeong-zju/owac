"""A failed probe must release its GPU and resume only the trainer it paused."""

import json
import os
from types import SimpleNamespace

import pytest

from owac.evaluation.n01_v12_timeline import video_outcome
from owac.runtime import n01_v12_paired_probe as probe


def test_missing_terminal_signal_cannot_be_reported_as_a_failed_video(tmp_path):
    key = ("libero_10", 6, 2)
    assert "unavailable" in video_outcome(key)
    ledger = tmp_path / "episodes.jsonl"
    ledger.write_text(
        json.dumps({"task_suite": key[0], "task_id": 6, "episode_idx": 2, "success": True}) + "\n"
    )
    assert "SUCCESS" in video_outcome(key, ledger)
    assert "unavailable" in video_outcome(("libero_10", 3, 1), ledger)


@pytest.mark.parametrize("trainer_alive", [True, False])
def test_loading_failure_restores_only_owned_paused_training(tmp_path, monkeypatch, trainer_alive):
    root, train, deploy = (tmp_path / name for name in ("probe", "train", "deploy"))
    root.mkdir()
    worker = train / "monitor/workers/libero-raw_n01-seed101.json"
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps({"pid": os.getpid(), "step": 100, "checkpoint_step": 100}))
    (root / "captured-checkpoints.json").write_text(
        json.dumps([{"arm": "raw_n01", "step": 20800, "checkpoint": "captured.pt"}])
    )
    commands, children = [], []
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(probe, "stop_group", lambda child: children.append(child))
    process = SimpleNamespace(pid=123456, poll=lambda: 1)
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **kw: process)
    if not trainer_alive:

        def missing(*args):
            raise ProcessLookupError("trainer no longer exists")

        monkeypatch.setattr(probe.os, "kill", missing)
    with pytest.raises(RuntimeError):
        probe.main(
            [
                "--root",
                str(root),
                "--training-root",
                str(train),
                "--deploy-root",
                str(deploy),
                "--role",
                "raw",
            ]
        )
    state = json.loads((root / "raw_n01/progress.json").read_text())
    assert state["state"] == "failed" and not state["paused_training"]
    if trainer_alive:
        assert commands == [
            ["supervisorctl", "stop", "owac-n01-v12-complete-raw"],
            ["supervisorctl", "start", "owac-n01-v12-complete-raw"],
        ]
        assert process in children
        assert state["training_resumed"]
    else:
        assert not commands

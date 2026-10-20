"""Bound runtime CLI tests; no GPU/model/training execution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import owac.runtime.rawvla_teacher as runtime


def _authority(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "owac.raw-teacher.authorization.v1",
                "authorization": {
                    "teacher_optimizer_updates": True,
                    "student_training": False,
                    "physical_actions": False,
                },
                "limits": {
                    "local_gpu_hours": 12,
                    "additional_disk_bytes": 50 * 1024**3,
                    "demonstrations_replayed_max": 50,
                },
                "training": {
                    "max_cumulative_updates": 2000,
                    "max_anchor_presentations": 16000,
                    "effective_batch": 8,
                    "microbatch": 1,
                    "gradient_accumulation": 8,
                },
                "run_root": str(path.parent),
            }
        )
    )
    return path


def _args(tmp_path: Path):
    auth = _authority(tmp_path / "authority.json")
    catalog = tmp_path / "validation-catalog.json"
    catalog.write_text("{}")
    return runtime.build_parser().parse_args(
        [
            "train",
            "--authority",
            str(auth),
            "--source-manifest",
            str(tmp_path / "sources.json"),
            "--catalog",
            str(catalog),
            "--author-root",
            str(tmp_path / "author"),
            "--starvla-root",
            str(tmp_path / "starvla"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--statistics",
            str(tmp_path / "stats"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--run-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--activation-memory-mode",
            "save_on_cpu",
        ]
    )


def test_train_has_no_generic_callback_or_bypass_flags_and_requires_memory_mode(tmp_path):
    parser = runtime.build_parser()
    base = [
        "train",
        "--authority",
        str(tmp_path / "a"),
        "--source-manifest",
        str(tmp_path / "s"),
        "--catalog",
        str(tmp_path / "c"),
        "--author-root",
        str(tmp_path / "ar"),
        "--starvla-root",
        str(tmp_path / "st"),
        "--checkpoint",
        str(tmp_path / "ck"),
        "--tokenizer",
        str(tmp_path / "tok"),
        "--statistics",
        str(tmp_path / "stats"),
        "--config",
        str(tmp_path / "cfg"),
        "--run-root",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args(
            base + ["--activation-memory-mode", "none", "--validation-callback", "x:y"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--activation-memory-mode", "none", "--budget-integration-ready"])


def test_training_cannot_restart_phase_budget_in_another_root(tmp_path):
    args = _args(tmp_path)
    args.run_root = str(tmp_path / "other-root")
    with pytest.raises(runtime.TeacherTrainingBlocked, match="explicit phase authority"):
        runtime.cmd_train(args)
    assert not (tmp_path / "other-root").exists()


def test_cmd_train_leases_before_model_and_uses_bound_validator(tmp_path, monkeypatch):
    args = _args(tmp_path)
    events = []

    class FakeTeacher:
        pass

    class FakeTrainer:
        def __init__(
            self, *, teacher, data, authority, run_root, stats, budget, activation_memory_mode
        ):
            events.append(("trainer", Path(run_root).name, activation_memory_mode))
            self.cumulative_updates = 0
            self.anchor_presentations = 0
            self.budget = budget

        def load_checkpoint(self, path):
            events.append(("resume", str(path)))

        def train(self, *, max_updates, validation):
            events.append(("train", int(max_updates)))
            value = validation()
            events.append(("validator", value["first5_mae7"]))
            return {"status": "fixture", "target": int(max_updates), "validation": value}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            events.append(("evaluator", "bound"))

        def evaluate(self, *, split, step):
            return {"first5_mae7": 0.1, "full10_mae7": 0.2, "predictions_data": []}

    def fake_build(assets, device):
        lease = tmp_path / "budget" / "lease.json"
        assert lease.is_file() and "offline" not in lease.read_text()
        events.append(("build", device))
        return FakeTeacher()

    monkeypatch.setattr(runtime, "build_author_teacher", fake_build)
    monkeypatch.setattr(runtime, "_teacher_data", lambda args: SimpleNamespace())
    monkeypatch.setattr(runtime, "_teacher_stats", lambda args: SimpleNamespace())
    monkeypatch.setattr(runtime, "TeacherTrainer", FakeTrainer)
    import owac.evaluation.rawvla_teacher as evaluator_module

    monkeypatch.setattr(evaluator_module, "TeacherOfflineEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        runtime,
        "save_evaluation_result",
        lambda result, output: {
            "first5_mae7": result["first5_mae7"],
            "full10_mae7": result["full10_mae7"],
        },
    )
    result = runtime.cmd_train(args)
    assert result == 0
    assert events[0][0] == "build"
    assert any(event[0] == "trainer" and event[2] == "save_on_cpu" for event in events)
    assert any(event[0] == "validator" for event in events)

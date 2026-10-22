"""Independent regressions for checkpoint identity and selected-version ownership."""

from types import SimpleNamespace

import pytest
import torch

from owac.runtime import raw_temporal_distillation as cli
from owac.training.raw_temporal_distillation import (
    TemporalDistillationTrainer,
    TemporalTrainingConfig,
)


class TinyStudent(torch.nn.Module):
    _scope = "stem_only"

    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.ones(1))

    @property
    def model_identity(self):
        return str(float(self.value.detach()[0]))


def test_full_recipe_mismatch_rejected_without_optimizer(tmp_path):
    first = TemporalDistillationTrainer(
        student=TinyStudent(), bridge=None, config=TemporalTrainingConfig()
    )
    path = first.save_checkpoint(tmp_path / "zero.pt")
    other = TemporalDistillationTrainer(
        student=TinyStudent(), bridge=None, config=TemporalTrainingConfig(learning_rate=0.2)
    )
    with pytest.raises(RuntimeError, match="full training recipe"):
        other.load_checkpoint(path)


def test_selection_sidecar_identifies_best_weights_not_current_weights(tmp_path):
    trainer = TemporalDistillationTrainer(
        student=TinyStudent(), bridge=None, config=TemporalTrainingConfig()
    )
    checkpoint = trainer.save_checkpoint(tmp_path / "best-zero.pt")
    with torch.no_grad():
        trainer.student.value.fill_(2)
    trainer.cumulative_updates = 50
    value = cli._write_selection_sidecar(
        tmp_path,
        trainer,
        best_value=0.3,
        best_step=0,
        best_path=str(checkpoint),
        best_metrics={"first5_mae7": 0.3},
    )
    assert value["model_identity"] == "1.0"
    assert value["cumulative_updates"] == 0
    assert value["run_cumulative_updates"] == 50


def test_export_uses_same_teacher_audit_and_run_identity(tmp_path, monkeypatch, capsys):
    import owac.raw.temporal_distillation as raw
    import owac.training.raw_temporal_distillation as training

    config = {"training": {}, "assets": {"output_root": str(tmp_path)}}
    identity = {
        key: "a" * 64
        for key in [
            "artifact_sha256",
            "backend_sha256",
            "tokenizer_sha256",
            "stats_sha256",
            "sources_sha256",
        ]
    }
    teacher = SimpleNamespace(
        policy=torch.nn.Linear(1, 1), frontend=SimpleNamespace(model=torch.nn.Linear(1, 1))
    )
    bundle = SimpleNamespace(identity=identity, teacher=teacher)
    monkeypatch.setattr(cli, "_config", lambda path: config)
    monkeypatch.setattr(cli, "_asset_set", lambda *a, **kw: object())
    monkeypatch.setattr(
        cli, "_cache_for", lambda *a, **kw: (SimpleNamespace(identity_key="cache-key"), None)
    )
    monkeypatch.setattr(raw, "build_teacher_bundle", lambda *a, **kw: bundle)
    monkeypatch.setattr(
        raw, "build_temporal_student", lambda *a, **kw: (TinyStudent(), object(), {})
    )
    calls = []

    def inspect_constructor(**kw):
        calls.append(kw)
        raise ValueError("constructor captured before any load/update")

    monkeypatch.setattr(training, "TemporalDistillationTrainer", inspect_constructor)
    args = SimpleNamespace(config="unused", device="cpu", arm="C", step=0, checkpoint="zero.pt")
    assert cli.cmd_export(args) == 2
    assert calls[0]["teacher"][0] is teacher.policy
    assert calls[0]["teacher"][1] is teacher.frontend.model
    assert calls[0]["run_identity"] == cli._training_run_identity(config, bundle, "cache-key")
    assert "constructor captured" in capsys.readouterr().out


def test_missing_resource_budget_rejects_before_touching_real_assets(monkeypatch):
    from owac.training.raw_temporal_distillation import TemporalTrainingAuthority

    monkeypatch.setattr(cli, "_config", lambda path: {"authorization": {"optimizer_updates": True}})
    monkeypatch.setattr(
        cli,
        "_authority",
        lambda *a: TemporalTrainingAuthority(authorized=True, max_updates=1, max_anchors=16),
    )

    def forbidden_assets(*a, **kw):
        raise AssertionError("budget rejection must precede any asset/cache read")

    monkeypatch.setattr(cli, "_asset_set", forbidden_assets)
    args = SimpleNamespace(config="fixture", authority="fixture", max_updates=1)
    assert cli.cmd_train(args) == 2

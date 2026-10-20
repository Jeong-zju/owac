"""Import-safe CLI tests: current config must be unavailable, never a success stub."""

from __future__ import annotations

import json

import pytest

from owac.runtime.raw_distillation import main


def test_help_is_import_safe(capsys):
    try:
        code = main(["--help"])
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 0
    output = capsys.readouterr().out
    assert "prepare-pairs" in output
    assert "export-student" in output


def test_assets_current_config_is_structured_unavailable(capsys):
    code = main(["assets"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["status"] == "unavailable"
    assert report["missing"]["learned_teacher"] is True
    assert "Default ISP" in report["note"]


def test_train_default_is_blocked_and_never_updates(capsys):
    code = main(["train"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["status"] == "blocked"
    assert report["updates"] == 0
    assert "authority" in report["reason"]


def test_prepare_parity_and_evaluate_are_unavailable(capsys):
    for argv in (
        ["prepare-pairs", "--output", "/tmp/not-written"],
        ["parity"],
        ["preflight"],
        ["evaluate"],
        ["export-student", "--output", "/tmp/not-written"],
    ):
        code = main(argv)
        report = json.loads(capsys.readouterr().out)
        assert code == 2
        assert report["status"] == "unavailable"


def _fixture_context(tmp_path):
    import torch

    from owac.backends.raw_distillation_context import ExperimentContext
    from owac.baselines.raw_stem_siglip import HistoryContract
    from owac.training.raw_distillation import (
        PreparedTeacherBatch,
        TeacherPreparationReceipt,
        TrainingAuthority,
    )
    from tests.test_raw_distillation_training import _student_and_bridge

    student, bridge = _student_and_bridge()
    history = HistoryContract()

    class Teacher:
        def prepare(self, raw, history=None):
            receipt = TeacherPreparationReceipt(
                adapter_name="fixture",
                checkpoint_sha256="0" * 64,
                config_sha256="1" * 64,
                code_revision="fixture",
                evidence_level="fixture_contract",
                real_ready=False,
                raw_sha256="2" * 64,
                teacher_tokens_sha256="3" * 64,
                prepared_rgb_sha256="4" * 64,
                state_before_sha256="5" * 64,
                state_after_sha256="6" * 64,
                causal_prefix_sha256="7" * 64,
                history=history,
                geometry={},
                preprocessing={},
                limitation="fixture_contract only",
            )
            return PreparedTeacherBatch(
                raw=raw,
                prepared_rgb=torch.zeros(raw.shape[0], 3, 8, 8),
                tokens=torch.zeros(raw.shape[0], 4, 6),
                receipt=receipt,
            )

    context = ExperimentContext(
        config_path=tmp_path / "config.json",
        config={},
        evidence_level="fixture_contract",
        student=student,
        bridge=bridge,
        history=history,
        teacher_adapter=Teacher(),
        backend_identity={"sha256": "a" * 64},
        tokenizer_identity={"sha256": "b" * 64},
        stats_identity={"sha256": "c" * 64},
        source_split={"a": "train", "b": "validation", "c": "test"},
    )
    return context, TrainingAuthority(
        path="fixture-authority",
        sha256="d" * 64,
        authorized=True,
        max_updates=0,
        max_anchors=0,
    )


def test_fixture_context_positive_prepare_preflight_export(tmp_path, capsys):
    import numpy as np

    context, _ = _fixture_context(tmp_path)
    loader = lambda args: context  # noqa: E731
    raw = tmp_path / "raw.npy"
    np.save(raw, np.zeros((1, 2, 4, 8, 8), dtype=np.float32))
    code = main(
        ["prepare-pairs", "--raw", str(raw), "--output", str(tmp_path / "prepared")],
        context_loader=loader,
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["status"] == "prepared"
    assert report["real_ready"] is False
    assert (tmp_path / "prepared" / "receipt.json").is_file()
    code = main(["preflight"], context_loader=loader)
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["status"] == "preflight_ready"
    assert report["actual_updates"] == 0
    code = main(
        ["export-student", "--output", str(tmp_path / "student")],
        context_loader=loader,
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["status"] == "exported"
    assert (tmp_path / "student" / "state.pt").is_file()


def test_context_real_missing_assets_and_fixture_camera_anchors(tmp_path):
    from dataclasses import replace

    from owac.backends.raw_distillation_context import (
        ContextUnavailable,
        camera_anchors,
        load_experiment_context,
    )
    from tests.test_distillation_pairs import _load, _make_fixture

    with pytest.raises(ContextUnavailable, match=r"config\.assets\.paired_manifest is absent"):
        load_experiment_context("configs/experiments/raw_stem_siglip_distillation_v1.json")
    path, manifest, _ = _make_fixture(tmp_path)
    loaded = _load(path)
    anchors = camera_anchors(loaded)
    assert anchors and all(anchor.camera_order == ("base_0_rgb",) for anchor in anchors)
    source = loaded.sources[0]
    changed = replace(source, camera_ids=("base_0_rgb", "wrist_0_rgb"))
    bad_manifest = replace(loaded, sources=(changed, *loaded.sources[1:]))
    with pytest.raises(ContextUnavailable, match="camera anchor"):
        camera_anchors(bad_manifest)


def test_fixture_context_positive_evaluate_full_horizon(tmp_path, capsys, monkeypatch):
    from owac.backends.raw_distillation_context import ExperimentContext
    from owac.backends.rawvla_features_n02 import build_stub_bridge
    from owac.baselines.raw_stem_siglip import HistoryContract
    from owac.raw.distillation_pairs import (
        NormalisationStats,
        PairedDistillationDataset,
        load_paired_manifest,
    )
    from tests.test_distillation_pairs import _make_fixture
    from tests.test_raw_distillation_training import _student_and_bridge

    path, _, stats_payload = _make_fixture(tmp_path)
    student, _ = _student_and_bridge()
    built = build_stub_bridge(
        width=6,
        token_dim=6,
        tokens_per_camera=4,
        channels=3,
        frame_pixels=2,
        action_horizon=10,
        cameras=("base_0_rgb",),
        with_frame=False,
    )
    bridge = built["bridge"]

    class Tokenizer:
        def tokenize(self, text, state=None):
            return [2, 3, 0], [True, True, False]

    bridge.base.tokenizer = Tokenizer()
    stats = NormalisationStats.from_json(tmp_path / "norm_stats.json", mode="quantile")
    dataset = PairedDistillationDataset(
        load_paired_manifest(path, allow_fixture=True, fixture_source_count=6)
    )
    context = ExperimentContext(
        config_path=tmp_path / "config.json",
        config={},
        evidence_level="fixture_contract",
        student=student,
        bridge=bridge,
        dataset=dataset,
        tokenizer=bridge.base.tokenizer,
        stats=stats,
        history=HistoryContract(),
        backend_identity={"sha256": "a" * 64},
        tokenizer_identity={"sha256": "b" * 64},
        stats_identity={"sha256": "c" * 64},
        source_split=dataset.manifest.source_splits,
    )
    code = main(["evaluate"], context_loader=lambda args: context)
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["status"] == "evaluated"
    assert report["actual_rollouts"] == 0
    assert report["records"]
    assert report["selection"] is None or report["selection"]["split"] == "validation"
    assert report["records"][0]["split"] in {"validation", "test"}
    import owac.backends.raw_distillation_context as context_module

    monkeypatch.setattr(context_module, "build_pure_context", lambda *args, **kwargs: context)
    code = main(
        ["evaluate", "--student-artifact", str(tmp_path / "student")],
        context_loader=lambda args: context,
    )
    artifact_report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert artifact_report["student_artifact"].endswith("student")
    assert artifact_report["normalized"] and artifact_report["physical"]
    assert artifact_report["checkpoint_step"] == 0


def test_fixture_context_positive_parity_handler(capsys):
    from types import SimpleNamespace

    ctx, _ = _fixture_context(__import__("pathlib").Path("/tmp"))
    ctx.parity_runner = SimpleNamespace(
        run=lambda: SimpleNamespace(as_dict=lambda: {"passed": True, "first_mismatch": None})
    )
    code = main(["parity"], context_loader=lambda args: ctx)
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert "passed" in report
    assert report["passed"] is True
    assert report["actual_updates"] == 0


def test_fixture_context_train_cli_positive_with_spy_optimizer(tmp_path, capsys):
    import hashlib
    from types import SimpleNamespace

    from owac.runtime.raw_distillation import _config
    from tests.test_distillation_pairs import _make_fixture

    ctx, _ = _fixture_context(tmp_path)
    path, _, _ = _make_fixture(tmp_path)
    from owac.raw.distillation_pairs import PairedDistillationDataset, load_paired_manifest

    dataset = PairedDistillationDataset(
        load_paired_manifest(path, allow_fixture=True, fixture_source_count=6)
    )

    class Tok:
        def tokenize(self, text, state=None):
            return [1, 2], [True, True]

    ctx.dataset = dataset
    ctx.tokenizer = Tok()
    ctx.bridge.base.tokenizer = Tok()
    ctx.bridge.base.max_state_dim = 32
    ctx.checkpoint = SimpleNamespace(sha256="a" * 64)
    source_sha = hashlib.sha256(
        __import__("json")
        .dumps({s.source_id: s.source_sha256 for s in dataset.manifest.sources}, sort_keys=True)
        .encode()
    ).hexdigest()
    stage0 = tmp_path / "stage0.json"
    stage0.write_text(
        __import__("json").dumps(
            {
                "passed": True,
                "checkpoint_sha256": "a" * 64,
                "source_sha256": source_sha,
                "tolerance_digest": "t" * 64,
                "native_digest": "n" * 64,
            }
        )
    )
    config = _config(
        __import__("pathlib").Path("configs/experiments/raw_stem_siglip_distillation_v1.json")
    )
    config["authorization"]["optimizer_updates"] = True
    config["assets"]["stage0_parity"] = str(stage0)
    config["resource_limits"]["gpu_hours"] = 1.0
    config["resource_limits"]["training_disk_bytes"] = 100000000
    config["training"]["effective_batch"] = 1
    config["training"]["checkpoint_updates"] = [0, 1]
    cfg = tmp_path / "config.json"
    cfg.write_text(__import__("json").dumps(config))
    stage0.write_text(
        __import__("json").dumps(
            {
                "passed": True,
                "checkpoint_sha256": "a" * 64,
                "source_sha256": source_sha,
                "config_sha256": __import__("hashlib")
                .sha256(__import__("json").dumps(config, sort_keys=True, default=str).encode())
                .hexdigest(),
                "tolerance_digest": "t" * 64,
                "native_digest": "n" * 64,
            }
        )
    )
    authority = tmp_path / "authority.json"
    authority.write_text(
        __import__("json").dumps({"authorized": True, "max_updates": 1, "max_anchors": 16})
    )

    class Spy:
        steps = 0

        def zero_grad(self, *args, **kwargs):
            return None

        def step(self):
            self.steps += 1

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None

    spy = Spy()
    ctx.optimizer_factory = lambda trainer: spy
    ctx.scheduler_factory = lambda trainer: spy
    ctx.validation_callback = lambda: {
        "first5_mae7": 0.1,
        "full10_mae7": 0.2,
        "valid_first5": 1,
        "valid_full10": 1,
    }
    code = main(
        [
            "--config",
            str(cfg),
            "--fixture-context",
            "fixture",
            "train",
            "--authority",
            str(authority),
            "--arm",
            "C",
            "--checkpoint-dir",
            str(tmp_path / "ck"),
            "--max-updates",
            "1",
        ],
        context_loader=lambda args: ctx,
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0, report
    assert report["actual_updates"] == 1
    assert report["cumulative_updates"] == 1
    assert (tmp_path / "ck" / "checkpoint_0.pt").is_file()
    assert (tmp_path / "ck" / "checkpoint_1.pt").is_file()


def test_export_from_training_checkpoint_preserves_changed_stem_weights(tmp_path, capsys):
    import hashlib

    import torch

    from owac.baselines.raw_stem_siglip import load_student_artifact
    from owac.raw.distillation_pairs import PairedDistillationDataset, load_paired_manifest
    from owac.training.raw_distillation import _identity_hash
    from tests.test_distillation_pairs import _make_fixture

    ctx, _ = _fixture_context(tmp_path)
    path, _, _ = _make_fixture(tmp_path)
    dataset = PairedDistillationDataset(
        load_paired_manifest(path, allow_fixture=True, fixture_source_count=6)
    )
    ctx.dataset = dataset
    config = {"student": {"stem_hidden": 64}, "training": {"effective_batch": 1}}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    state = {name: tensor.clone() for name, tensor in ctx.student.state_dict().items()}
    stem_name = next(name for name in state if "patch_embedding" in name and "weight" in name)
    state[stem_name] = state[stem_name] + 0.25
    payload = {
        "schema": "owac.raw-distillation.checkpoint.v1",
        "student": state,
        "arm": "C",
        "cumulative_updates": 500,
        "config_sha256": _identity_hash(config),
        "data_sha256": dataset.manifest.manifest_sha256,
        "source_sha256": _identity_hash(
            {s.source_id: s.source_sha256 for s in dataset.manifest.sources}
        ),
    }
    checkpoint = tmp_path / "checkpoint_500.pt"
    torch.save(payload, checkpoint)
    code = main(
        [
            "--config",
            str(cfg),
            "--fixture-context",
            "fixture",
            "export-student",
            "--output",
            str(tmp_path / "artifact"),
            "--training-checkpoint",
            str(checkpoint),
            "--step",
            "500",
        ],
        context_loader=lambda args: ctx,
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0, report
    assert report["checkpoint_step"] == 500
    artifact_meta = json.loads((tmp_path / "artifact" / "frontend.json").read_text())
    assert artifact_meta["step"] == 500
    assert (
        artifact_meta["extra"]["training_checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    # Recreate the fixture architecture and compare the exact trained stem tensor.
    from tests.test_raw_stem_siglip import make_policy

    def factory(geometry, input_mode):
        policy = make_policy(seed=999)
        return (
            policy.vlm.model.model.vision_tower,
            policy.vlm.model.model.multi_modal_projector,
        )

    deployment = load_student_artifact(tmp_path / "artifact", frontend_factory=factory)
    actual = next(
        tensor
        for name, tensor in deployment.state_dict().items()
        if "patch_embedding" in name and "weight" in name
    )
    assert torch.equal(actual, state[stem_name])


def test_conversion_and_serialized_reload_counts_are_distinct():
    from owac.backends.raw_distillation_context import (
        ContextUnavailable,
        _validate_load_counts,
    )

    receipt = {
        "loaded_keys": 812,
        "model_keys": 813,
        "inactive_keys": ["action_head.gemma_expert.lm_head.weight"],
        "active_missing": [],
        "unexpected": [],
        "shape_mismatch": [],
    }
    actual = {
        "loaded_keys": 813,
        "model_keys": 813,
        "inactive_keys": [],
        "active_missing": [],
        "unexpected": [],
        "shape_mismatch": [],
    }
    _validate_load_counts(receipt, actual)
    bad = dict(actual)
    bad["loaded_keys"] = 812
    with pytest.raises(ContextUnavailable):
        _validate_load_counts(receipt, bad)

"""The real artifact loader must receive the actual frozen-backend identity."""

import hashlib

import pytest

from owac.evaluation.raw_temporal_distillation import TemporalPureRuntime


def test_pure_factory_passes_expected_checkpoint_hash_before_model_build(tmp_path, monkeypatch):
    import owac.evaluation.raw_temporal_distillation as evaluation

    checkpoint = tmp_path / "backend.safetensors"
    checkpoint.write_bytes(b"a fixed policy identity")
    calls = []

    class IdentityChecked(Exception):
        pass

    def inspect_loader(path, **kwargs):
        calls.append(kwargs["expected_checkpoint_sha256"])
        raise IdentityChecked

    monkeypatch.setattr(evaluation, "load_temporal_student_artifact", inspect_loader)
    with pytest.raises(IdentityChecked):
        TemporalPureRuntime.from_artifact(
            tmp_path / "artifact",
            checkpoint=checkpoint,
            tokenizer=tmp_path / "tokenizer",
            statistics=tmp_path / "stats",
        )
    assert calls == [hashlib.sha256(checkpoint.read_bytes()).hexdigest()]


def test_only_validated_temporal_wrapper_enables_component_teacher_history(tmp_path, monkeypatch):
    import inspect

    import owac.baselines.raw_temporal_siglip as temporal
    from owac.baselines.raw_stem_siglip import load_student_artifact

    assert (
        inspect.signature(load_student_artifact).parameters["allow_temporal_teacher"].default
        is False
    )
    calls = []

    class ComponentChecked(Exception):
        pass

    monkeypatch.setattr(temporal, "_read_temporal_payload", lambda path: ({}, {}))

    def inspect_component(path, **kwargs):
        calls.append(kwargs)
        raise ComponentChecked

    monkeypatch.setattr(temporal, "load_student_artifact", inspect_component)
    with pytest.raises(ComponentChecked):
        temporal.load_temporal_student_artifact(tmp_path, expected_checkpoint_sha256="a" * 64)
    assert calls[0]["allow_temporal_teacher"] is True
    assert calls[0]["expected_checkpoint_sha256"] == "a" * 64


def test_pure_factory_rejects_rgb_diagnostic_artifact_before_backend_build(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import owac.evaluation.raw_temporal_distillation as pure

    monkeypatch.setattr(pure, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(
        pure,
        "load_temporal_student_artifact",
        lambda *a, **kw: SimpleNamespace(input_mode="rgb3_control"),
    )
    with pytest.raises(pure.TemporalPureUnavailable, match="not a pure RAW"):
        pure.TemporalPureRuntime.from_artifact(
            tmp_path, checkpoint=tmp_path, tokenizer=tmp_path, statistics=tmp_path
        )

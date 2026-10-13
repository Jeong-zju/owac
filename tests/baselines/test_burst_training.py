"""Causal batching equivalence, declared stopping and exact resumption."""

import copy
import json
from pathlib import Path

import pytest
import torch

from owac.evaluation.n01_burst import make_models, run_sequence
from owac.evaluation.n01_burst_training import plateau, sequence, train
from owac.evaluation.n01_content import make_protocol_batch


def protocol():
    p = json.loads(
        (
            Path(__file__).parents[2] / "configs/encoders/n01/burst_features_convergence.json"
        ).read_text()
    )
    p.update(device="cpu", min_steps=2, max_steps=4, validation_every=2, evaluation_batch_size=1)
    p["encoder"].update(
        stem_width=8, residual_blocks=2, feature_width=16, projection_width=8, offset_width=8
    )
    return p


@pytest.mark.parametrize(
    "condition", ["static_weak", "translation_oracle_alignment", "local_change_stale_history"]
)
def test_batched_causal_references_match_streaming_outputs_loss_and_gradients(condition):
    p = protocol()
    a = make_models(p, 61)["dbsr_reference"]
    b = copy.deepcopy(a)
    batch = make_protocol_batch(p, 61, "train", 0, condition)
    first, first_metrics = run_sequence(a, batch, p, training=True)
    second, _ = sequence(b, batch, p, training=True)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    first.backward()
    second.backward()
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
    with torch.no_grad():
        _, second_metrics = sequence(b, batch, p)
    assert second_metrics["error_per_frame"] == pytest.approx(
        first_metrics["error_per_frame"], abs=1e-5
    )


def test_plateau_requires_minimum_training_and_reduced_lr_confirmation():
    p = protocol()
    p.update(min_steps=2048, plateau_checks=6)
    rows = [
        dict(step=256 * (i + 1), learning_rate=1e-5, validation_early_error_px=1.0)
        for i in range(8)
    ]
    assert plateau(rows, p)
    assert not plateau(rows[:7], p)
    rows[-2]["learning_rate"] = 1e-4
    assert not plateau(rows, p)
    rows[-2]["learning_rate"] = 1e-5
    rows[-1]["validation_early_error_px"] = 0.8
    assert not plateau(rows, p)


def test_resume_restores_optimizer_scheduler_stream_and_matches_uninterrupted(
    tmp_path, monkeypatch
):
    import owac.evaluation.n01_burst_training as runner

    p = protocol()
    reference = train(p, 61, "dbsr_reference", tmp_path / "reference")
    assert reference["stop_reason"] == "budget_limit" and reference["test"] is None
    original = runner.atomic_json

    def interrupted(path, payload):
        original(path, payload)
        if path.name == "progress.json" and payload["step"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "atomic_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train(p, 61, "dbsr_reference", tmp_path / "resumed")
    monkeypatch.setattr(runner, "atomic_json", original)
    resumed = train(p, 61, "dbsr_reference", tmp_path / "resumed", resume=True)
    for key in ["curves", "best_parameter_digest", "train_input_sha256", "validation"]:
        assert resumed[key] == reference[key]
    with pytest.raises(FileExistsError):
        train(p, 61, "dbsr_reference", tmp_path / "resumed", resume=True)


def test_budget_extension_preserves_parent_and_continues_the_same_learning_stream(tmp_path):
    from owac.runtime.n01_burst_budget import continue_budget

    p = protocol()
    parent = train(p, 61, "dbsr_reference", tmp_path / "parent")
    parent_bytes = (tmp_path / "parent/result.json").read_bytes()
    extended = continue_budget(tmp_path / "parent", tmp_path / "child", 6)
    reference = train(dict(p, max_steps=6), 61, "dbsr_reference", tmp_path / "reference")
    for key in ["curves", "best_parameter_digest", "train_input_sha256"]:
        assert extended[key] == reference[key]
    assert (tmp_path / "parent/result.json").read_bytes() == parent_bytes
    assert parent["steps"] == 4 and extended["steps"] == 6
    assert extended["budget_continuation"]["changed_protocol_fields"] == ["max_steps"]
    with pytest.raises(ValueError, match="increase"):
        continue_budget(tmp_path / "parent", tmp_path / "invalid", 4)

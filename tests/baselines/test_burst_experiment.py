"""Fair comparisons, target boundary and real checkpoint training behavior."""

import copy
import json
from pathlib import Path

import torch

from owac.baselines.burst_features import BurstFeatureConfig
from owac.evaluation.n01_burst import (
    BurstProbe,
    evaluate,
    make_models,
    oracle_offsets,
    parameter_digest,
    run_experiment,
)
from owac.evaluation.n01_content import make_protocol_batch


def protocol():
    p = json.loads(
        (Path(__file__).parents[2] / "configs/encoders/n01/burst_features_pilot.json").read_text()
    )
    return p


def test_full_capacity_controls_and_shared_initialization():
    p = protocol()
    models = make_models(p, 61)
    ref = models["dbsr_reference"]
    for method in ["current", "mean", "dbsr_consensus"]:
        assert parameter_digest(models[method].encoder.backbone) == parameter_digest(
            ref.encoder.backbone
        )
        assert parameter_digest(models[method].head) == parameter_digest(ref.head)
    assert parameter_digest(models["dbsr_consensus"].encoder.fusion) == parameter_digest(
        ref.encoder.fusion
    )
    capacity = sum(p.numel() for p in models["mean_capacity"].parameters())
    target = sum(p.numel() for p in ref.parameters())
    assert abs(capacity - target) / target < 0.02
    assert models["mean_capacity"].encoder.config.residual_blocks == 16


def test_alignment_fixture_uses_only_arrived_translations_and_not_target_labels():
    p = protocol()
    batch = make_protocol_batch(p, 61, "validation", 1, "translation_oracle_alignment")
    original = oracle_offsets(batch, 8, 1, device="cpu")
    other = copy.deepcopy(batch)
    other.target_xy[:] = 1e6
    other.translations[:, 9:] = 1e6
    assert torch.equal(original, oracle_offsets(other, 8, 1, device="cpu"))
    assert original[:, 0].count_nonzero() == 0
    ordinary = make_protocol_batch(p, 61, "validation", 0, "static_weak")
    assert oracle_offsets(ordinary, 8, 1, device="cpu") is None


def test_training_updates_backbone_and_fusion_and_checkpoint_reproduces_evaluation(tmp_path):
    p = protocol()
    # Small dimensions are unit-fixture only; shipped experiment defaults stay full size.
    p.update(
        steps=2,
        validation_every=2,
        batch_size=1,
        evaluation_batch_size=1,
        methods=["dbsr_reference"],
        seeds=[61],
    )
    p["encoder"].update(
        stem_width=8, residual_blocks=2, feature_width=16, projection_width=8, offset_width=8
    )
    result = run_experiment(p, tmp_path / "run")
    run = result["runs"][0]
    # Integer oracle translations have zero offset modulo 1: offset input weights
    # need not receive nonzero gradients. Check actual backbone/fusion updates.
    assert 0 < run["active_parameters"] <= run["parameters"]
    assert run["initial_parameter_hashes"]["full"] != run["best_parameter_hash"]
    checkpoint = torch.load(tmp_path / "run/61-dbsr_reference-best.pt", weights_only=True)
    restored = BurstProbe(BurstFeatureConfig(**checkpoint["encoder_config"]))
    restored.load_state_dict(checkpoint["model"], strict=True)
    initial = make_models(p, 61)["dbsr_reference"]
    assert parameter_digest(restored.encoder.backbone) != parameter_digest(initial.encoder.backbone)
    assert parameter_digest(restored.encoder.fusion) != parameter_digest(initial.encoder.fusion)
    assert parameter_digest(restored) == run["best_parameter_hash"]
    assert evaluate(restored, p, 61, "validation") == run["evaluation"]
    assert run["evaluation_split"] == "validation"
    assert run["evaluation"]["oracle_excluded_from_main_aggregate"]
    assert set(run["evaluation"]["cases"]) == {
        "static_weak",
        "translation_oracle_alignment",
        "occlusion_release",
        "local_change_stale_history",
    }

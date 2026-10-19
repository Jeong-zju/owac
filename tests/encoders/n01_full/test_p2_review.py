"""Independent checks of P2 stop conditions and metric denominators."""

from pathlib import Path

import torch
from torch import nn

from owac.backends.raw_correspondence import CorrespondenceEstimator, RawSensorFrame
from owac.evaluation.n01_motion import _Accumulator, _static_false_rejection
from owac.training.n01_motion import decide_platform


class ZeroFlow(nn.Module):
    def forward(self, image1, image2, **kwargs):
        return {"final": torch.zeros(1, 2, *image1.shape[-2:])}


def frame(frame_id, validity):
    return RawSensorFrame(
        dataset_id="d",
        episode_id="e",
        camera_id="c",
        frame_id=frame_id,
        values=torch.ones(3, 4, 4),
        cfa_pattern="rgb3",
        channel_phase=(),
        sensor_origin_xy=(0, 0),
        validity=validity,
    )


def test_invalid_reference_sample_has_no_correspondence_taps():
    valid = torch.ones(3, 4, 4, dtype=torch.bool)
    valid[0, 1, 1] = False
    result = CorrespondenceEstimator(mode="rgb3", model=ZeroFlow()).estimate(
        frame(1, valid), frame(0, torch.ones_like(valid))
    )
    assert not result.valid[0, 1, 1]
    assert result.taps_for_query(0, 1, 1) == ()
    assert result.valid[1, 1, 1]


def test_unspent_budget_cannot_make_the_last_best_a_plateau():
    history = [
        {"update": (i + 1) * 1000, "val_epe_sensor_px": v}
        for i, v in enumerate([2.0, 1.0] + [1.0] * 8 + [0.999])
    ]
    decision = decide_platform(
        val_history=history,
        budget=300000,
        interval=1000,
        monitor="val_epe_sensor_px",
        delta_val=0.002,
        plateau_checks=8,
        lr_drops=2,
        final_phase=True,
        diagnostics_passed=True,
    )
    assert decision.status == "budget_exhausted"
    assert not decision.details["conditions"]["best_not_at_budget_end"]


def test_occluded_queries_remain_in_detection_population():
    acc = _Accumulator()
    # EPE exists only at the first two cells. The last two are occluded and
    # have perfect detector scores. Filtering by EPE validity would erase them.
    acc.add(
        torch.zeros(4),
        torch.zeros(4),
        mask=torch.tensor([1, 1, 0, 0]).bool(),
        risk=torch.tensor([0.0, 0.0, 2.0, 2.0]),
        coverage=None,
        invalid_label=torch.tensor([0, 0, 1, 1]).bool(),
        supervision_valid=torch.tensor([1, 1, 0, 0]).bool(),
        predicted_valid=torch.ones(4, dtype=torch.bool),
    )
    report = acc.summary()
    assert report["occlusion_auc"] == 1.0
    assert report["invalid_label_fraction"] == 0.5
    assert report["occlusion_precision_recall"]["recall"] == 1.0


def test_detection_report_survives_an_entirely_occluded_pair():
    acc = _Accumulator()
    acc.add(
        torch.zeros(2),
        torch.zeros(2),
        mask=torch.zeros(2).bool(),
        risk=torch.ones(2) * 2,
        coverage=None,
        invalid_label=torch.ones(2).bool(),
        supervision_valid=torch.zeros(2).bool(),
        predicted_valid=torch.ones(2).bool(),
    )
    report = acc.summary()
    assert report["valid_cells"] == 0
    assert report["occlusion_precision_recall"]["recall"] == 1.0


def test_static_false_rejection_counts_rejected_valid_sensor_cells():
    from types import SimpleNamespace

    from owac.evaluation.n01_motion import MotionEvaluator
    from owac.training.n01_motion import load_motion_training_config

    root = Path(__file__).resolve().parents[3]
    config = load_motion_training_config(root / "configs/encoders/n01_full/p2_motion_train.json")
    evaluator = MotionEvaluator(config)
    specs = evaluator.specs("validation", limit=8)
    calls = []

    class RejectAll:
        def estimate(self, reference, source, **kwargs):
            calls.append((reference, source))
            return SimpleNamespace(packed_flow=torch.zeros(2, *reference.packed_shape), valid=False)

    evaluator.estimators = {"sea_raft": RejectAll()}
    report = _static_false_rejection(evaluator, "validation", specs)
    assert report["pairs"] > 0
    assert report["false_rejection_rate"] == 1.0
    reference, source = calls[0]
    assert reference.frame_id != source.frame_id
    assert not torch.equal(reference.values, source.values)
    # Both are sampled from exactly the same scene geometry, with independent
    # frame noise. No source identity or noise realization is duplicated.
    assert reference.sample_key(0, 0, 0) != source.sample_key(0, 0, 0)


def test_prior_seed_101_verdict_requires_more_observations():
    import json

    root = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (
            root / "research/experiments/results/n01_full/P2/remote-runner-seed-101-receipt.json"
        ).read_text()
    )
    decision = decide_platform(
        val_history=payload["val_history"],
        budget=payload["budget"],
        interval=1000,
        monitor="val_epe_sensor_px",
        delta_val=0.002,
        plateau_checks=8,
        lr_drops=payload["lr_drops"],
        final_phase=True,
        diagnostics_passed=True,
    )
    assert decision.status == "budget_exhausted"


def test_calibration_fit_and_lookup_use_identical_bins():
    from owac.evaluation.n01_motion import ErrorCoverageTable

    proxy = torch.arange(100, dtype=torch.float64)
    table = ErrorCoverageTable.fit(
        [
            {
                "flow_risk": proxy,
                "visibility": proxy,
                "error": proxy,
                "valid": torch.ones(100, dtype=torch.bool),
            }
        ],
        split="calibration",
        bins=4,
        nominal_coverage=0.9,
    )
    assert len(table.risk_edges) == 3
    radius, _ = table.map_proxy(flow_risk=proxy, visibility=proxy)
    # Each populated diagonal bin contains 25 observations; highest values
    # cannot alias a different bin or be silently omitted during fitting.
    for start in (0, 25, 50, 75):
        expected = torch.quantile(proxy[start : start + 25], 0.9)
        torch.testing.assert_close(radius[start : start + 25], expected.expand(25))


def test_generated_sequences_do_not_alias_raw_sample_identities():
    from owac.raw.n01_motion_data import MotionDataGenerator

    root = Path(__file__).resolve().parents[3]
    generator = MotionDataGenerator.from_config_path(
        root / "configs/encoders/n01_full/p2_motion_noise.json"
    )
    pairs = [
        generator.sample_pair(split="train", case="rotation", sequence=sequence)
        for sequence in ("t0", "t1")
    ]
    first, second = [pair.visible.reference for pair in pairs]
    assert first.sample_key(0, 0, 0) != second.sample_key(0, 0, 0)
    pseudo = generator.sample_pair(
        split="train", case="rotation", sequence="t0", input_mode="rawvla_rgb3"
    )
    assert first.sample_key(0, 0, 0) != pseudo.visible.reference.sample_key(0, 0, 0)
    assert (
        pairs[0].visible.metadata["noise_block_id"] != pairs[1].visible.metadata["noise_block_id"]
    )


def test_resume_refuses_a_changed_calibration_file(tmp_path):
    from types import SimpleNamespace

    import pytest

    from owac.training.n01_motion import CHECKPOINT_SCHEMA, MotionTrainer

    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"changed": true}')
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "core_fingerprint": "fixed-config",
            "seed": 101,
            "iters": 4,
            "base_lr": 0.001,
            "hashes": {"data_config": {"sha256": "old-digest"}},
        },
        checkpoint,
    )
    holder = SimpleNamespace(
        config=SimpleNamespace(
            core_fingerprint="fixed-config", data=SimpleNamespace(config_path=calibration)
        ),
        seed=101,
        iters=4,
        base_lr=0.001,
    )
    with pytest.raises(ValueError, match="data/calibration file changed"):
        MotionTrainer.load_checkpoint(holder, checkpoint)


def test_all_rejected_arm_still_has_a_report():
    from owac.evaluation.n01_motion import _aggregate_arm

    report = _aggregate_arm(
        [
            {
                "epe_sensor_px": None,
                "valid_cells": 0,
                "cells": 100,
                "error_quantiles_sensor_px": {},
            }
        ]
    )
    assert report["epe_sensor_px"] is None
    assert report["valid_cells"] == 0

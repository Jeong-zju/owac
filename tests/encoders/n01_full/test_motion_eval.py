"""Behavioural properties of the P2 motion evaluation and its risk calibration.

The oracles here are independent: the ROC AUC is recomputed by brute force over
pairs, the calibrated risk table is checked against an exactly known monotone
error law, the EPE accumulator against a constant one-packed-cell offset, the
simplified alignment against a shift it has to recover (sign included), and the
oracle arm is driven with estimators that fail the test if they are ever called.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.backends.raw_correspondence import RiskCalibrationTable
from owac.evaluation.n01_motion import (
    BUCKET_LABELS,
    FITTING_SPLIT,
    GROUP_ORDER,
    ErrorCoverageTable,
    EvaluationConfigError,
    GroupMap,
    MotionEvaluator,
    _align_to,
    _bucket_report,
    _cfa_phase_consistency,
    _collect_calibration_samples,
    _latency_report,
    _load_checkpoint,
    _pair_error,
    _risk_radius_coverage,
    _simplified_global_shift,
    _static_false_rejection,
    _validate_buckets,
    error_quantiles,
    roc_auc,
    run_evaluation,
)
from owac.raw.n01_motion_data import MotionDataGenerator
from owac.training.n01_motion import (
    MotionTrainingConfig,
    flow_target_from_supervision,
    load_motion_training_config,
)

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/encoders/n01_full/p2_motion_train.json"
DATA_CONFIG = REPO_ROOT / "configs/encoders/n01_full/p2_motion_noise.json"
CASES = (
    "rotation",
    "layered_occlusion",
    "local_nonrigid",
    "repeated_texture",
    "exposure_gain",
    "motion_blur",
    "combined_hard",
)


def _region(prefix: str) -> dict:
    cases: dict = {"translation_subpixel": ["x+0.25", "x-0.25"]}
    for case in CASES:
        cases[case] = {"prefix": prefix, "count": 1}
    return {"cases": cases, "history_offsets": [1]}


def tiny_config() -> MotionTrainingConfig:
    """The shipped configuration reduced to a fast but real CPU evaluation."""
    payload = json.loads(CONFIG_PATH.read_text())
    payload["model"].update({"depth": "resnet18", "iters": 0, "pretrained_checkpoint": None})
    payload["data"].update(
        {
            "config_path": str(DATA_CONFIG),
            "geometry": None,
            "pseudo_raw_geometry": None,
            "recipe_scale": 1.0,
            "batch_size": 1,
            "cache_entries": 8,
            "train": _region("tr"),
            "calibration": _region("ca"),
            "validation": _region("va"),
            "test": _region("te"),
        }
    )
    for key in ("max_calibration_pairs", "max_validation_pairs", "max_test_pairs"):
        payload["evaluation"][key] = 9
    payload["evaluation"]["latency_warmup"] = 1
    payload["evaluation"]["latency_repeats"] = 2
    return MotionTrainingConfig.from_dict(payload)


@pytest.fixture(scope="module")
def generator() -> MotionDataGenerator:
    return MotionDataGenerator.from_config_path(DATA_CONFIG)


def _sample(risk: torch.Tensor, visibility: torch.Tensor, error: torch.Tensor) -> dict:
    return {
        "flow_risk": risk,
        "visibility": visibility,
        "error": error,
        "valid": torch.ones_like(risk, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# risk calibration table
# ---------------------------------------------------------------------------


def test_error_coverage_table_fits_on_calibration_only_and_is_a_protocol():
    risk = torch.tensor([[0.1, 0.2, 0.3, 0.4]] * 4, dtype=torch.float64)
    visibility = torch.full((4, 4), 0.5, dtype=torch.float64)
    error = risk * 2.0
    table = ErrorCoverageTable.fit(
        [_sample(risk, visibility, error)],
        split=FITTING_SPLIT,
        bins=4,
        nominal_coverage=0.9,
    )
    assert isinstance(table, RiskCalibrationTable)
    assert table.fitted_split == FITTING_SPLIT
    assert table.cells == 16

    # the error law is exactly known: each bin has a constant error, so the
    # calibrated radius is that constant and the empirical coverage is one
    mapped, coverage = table.map_proxy(
        flow_risk=torch.tensor([[0.15, 0.35]]), visibility=torch.tensor([[0.5, 0.5]])
    )
    assert mapped.tolist() == [[pytest.approx(0.4), pytest.approx(0.8)]]
    assert coverage.tolist() == [[1.0, 1.0]]
    assert mapped.shape == coverage.shape == (1, 2)

    for forbidden in ("validation", "test", "train"):
        with pytest.raises(EvaluationConfigError, match="calibration"):
            ErrorCoverageTable.fit([_sample(risk, visibility, error)], split=forbidden)


def test_map_proxy_keeps_the_proxy_shape_and_a_bounded_coverage():
    rng = np.random.default_rng(4)
    risk = torch.tensor(np.abs(rng.normal(size=(8, 6))) + 0.01)
    visibility = torch.tensor(rng.random((8, 6)))
    error = risk * torch.tensor(np.abs(rng.normal(size=(8, 6))) + 0.5)
    table = ErrorCoverageTable.fit(
        [_sample(risk.double(), visibility.double(), error.double())],
        split=FITTING_SPLIT,
        bins=4,
        nominal_coverage=0.9,
    )
    mapped, coverage = table.map_proxy(flow_risk=risk, visibility=visibility)
    assert mapped.shape == risk.shape and coverage.shape == risk.shape
    assert bool(((coverage >= 0.0) & (coverage <= 1.0)).all())
    assert bool(torch.isfinite(mapped).all())
    serialized = table.to_dict()
    assert serialized["fitted_split"] == FITTING_SPLIT
    assert len(serialized["risk"]) == len(serialized["coverage"])


def test_roc_auc_matches_a_brute_force_pairwise_oracle():
    rng = np.random.default_rng(11)
    scores = rng.normal(size=120)
    labels = rng.random(120) < 0.4
    positive = scores[labels]
    negative = scores[~labels]
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    expected = (wins + 0.5 * ties) / (positive.size * negative.size)
    assert roc_auc(scores, labels) == pytest.approx(float(expected), rel=1e-12)

    assert roc_auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert roc_auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    assert roc_auc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert math.isnan(roc_auc([1.0, 2.0], [True, True]))
    with pytest.raises(ValueError, match="aligned"):
        roc_auc([1.0, 2.0], [True])


def test_error_quantiles_are_empty_safe():
    quantiles = error_quantiles([1.0, 2.0, 3.0, 4.0])
    assert quantiles["p50"] == pytest.approx(2.5)
    assert quantiles["p95"] >= quantiles["p90"] >= quantiles["p50"]
    assert math.isnan(error_quantiles([])["p50"])


# ---------------------------------------------------------------------------
# metrics use only the legal mask
# ---------------------------------------------------------------------------


def test_pair_error_and_accumulator_report_the_legal_mask_only(generator):
    geometry = generator.geometry()
    pair = generator.sample_pair(split="train", case="layered_occlusion", sequence="s0")
    target = flow_target_from_supervision(pair.supervision, geometry, mode="cfa4")
    # exactly one packed cell off along the row axis in every channel, so the EPE is
    # one packed cell == two sensor pixels; the mean flow keeps the shared-field
    # branch consistent with the per-channel branch
    channel_flow = target.channel_flow.clone()
    channel_flow[:, 0] += 1.0
    predicted = channel_flow.mean(dim=0)
    group = GroupMap(flow=predicted, valid=torch.ones_like(target.valid), channel_flow=channel_flow)
    error_sensor, error_packed, mask, invalid = _pair_error(
        predicted, target, predicted_valid=group.valid, group_map=group
    )
    assert torch.equal(mask, target.channel_valid)
    assert int((mask & ~target.channel_valid).sum()) == 0
    assert int((mask & invalid).sum()) == 0  # an invalid cell is never in the mask

    from owac.evaluation.n01_motion import _Accumulator

    accumulator = _Accumulator().with_threshold(1.0)
    accumulator.add(
        error_sensor.mean(dim=0),
        error_packed.mean(dim=0),
        mask=mask.any(dim=0),
        risk=None,
        coverage=None,
        invalid_label=invalid.any(dim=0),
        supervision_valid=target.valid,
        predicted_valid=group.valid,
    )
    summary = accumulator.summary()
    assert summary["valid_cells"] == int(target.valid.sum())
    assert summary["epe_packed"] == pytest.approx(1.0, abs=1e-6)
    assert summary["epe_sensor_px"] == pytest.approx(2.0, abs=1e-6)
    # occlusion identification needs a risk score; without one it is reported as absent
    assert "occlusion_auc" not in summary


# ---------------------------------------------------------------------------
# the four arms
# ---------------------------------------------------------------------------


def test_oracle_arm_cannot_touch_an_estimator(generator):
    class _Exploding:
        def estimate(self, *args, **kwargs):
            raise AssertionError("the oracle arm must never call an estimator")

    geometry = generator.geometry()
    pair = generator.sample_pair(split="train", case="translation_subpixel", sequence="x+0.25")
    target = flow_target_from_supervision(pair.supervision, geometry, mode="cfa4")
    dummy = SimpleNamespace(
        estimators={"sea_raft": _Exploding(), "no_alignment": _Exploding()},
        sensor_per_packed=2,
    )
    group = MotionEvaluator.arm_map(dummy, "oracle", None, None, target, table=None)
    assert group.oracle is True
    assert group.support_label == "oracle_support_diagnostic"
    assert torch.equal(group.channel_flow, target.channel_flow)
    assert torch.equal(group.valid, target.valid)


def _pull_shifted(reference: torch.Tensor, pull_y: int, pull_x: int) -> torch.Tensor:
    """``source`` with ``reference[r] == source[r + pull]`` exactly."""
    rows, cols = reference.shape[1:]
    source = torch.zeros_like(reference)
    source[:, max(0, pull_y) : rows + min(0, pull_y), max(0, pull_x) : cols + min(0, pull_x)] = (
        reference[
            :, max(0, -pull_y) : rows + min(0, -pull_y), max(0, -pull_x) : cols + min(0, -pull_x)
        ]
    )
    return source


def test_simplified_alignment_recovers_a_known_pull_shift_including_its_sign():
    rng = torch.Generator().manual_seed(17)
    reference = torch.rand(4, 24, 24, generator=rng)
    for pull_y, pull_x in ((3, -4), (-2, 5), (0, 0)):
        source = _pull_shifted(reference, pull_y, pull_x)
        assert _simplified_global_shift(reference, source, search_cells=6) == (pull_y, pull_x)

    assert _simplified_global_shift(reference, reference, search_cells=3) == (0, 0)
    with pytest.raises(ValueError, match="equal-shape"):
        _simplified_global_shift(reference, reference[:, :10], search_cells=1)


# ---------------------------------------------------------------------------
# buckets, latency, static pairs, the full receipt
# ---------------------------------------------------------------------------


def _record(bucket: str, case: str, arm: str) -> dict:
    return {
        "region": "test",
        "case": case,
        "sequence": "s0",
        "history_offset": 1,
        "bucket": bucket,
        "arm": arm,
        "epe_sensor_px": 1.0,
        "epe_packed": 0.5,
        "valid_cells": 10,
        "cells": 12,
        "error_quantiles_sensor_px": {"p50": 1.0, "p90": 1.0, "p95": 1.0},
    }


def test_bucket_report_refuses_to_drop_a_hard_bucket():
    config = tiny_config()
    _validate_buckets(config, list(config.data.train.cases))
    records = [
        _record(bucket, case, arm)
        for case, bucket in config.evaluation.buckets.items()
        for arm in GROUP_ORDER
    ]
    report = _bucket_report(records, config, 2)
    assert set(report) == set(BUCKET_LABELS)
    assert all(entry["cases"] for entry in report.values())

    without_anomaly = [record for record in records if record["bucket"] != "current_frame_anomaly"]
    with pytest.raises(EvaluationConfigError, match="must not be dropped"):
        _bucket_report(without_anomaly, config, 2)


def test_bucket_map_must_cover_every_case_and_use_all_four_labels():
    payload = json.loads(CONFIG_PATH.read_text())
    # dropping a case that shares its bucket with others keeps four labels but
    # breaks the case coverage the evaluation refuses to guess
    del payload["evaluation"]["buckets"]["motion_blur"]
    config = MotionTrainingConfig.from_dict(payload)
    with pytest.raises(KeyError):
        config.evaluation.bucket_of("motion_blur")
    with pytest.raises(
        EvaluationConfigError, match="every declared case needs an evaluation bucket"
    ):
        _validate_buckets(config, ["motion_blur", "rotation"])

    payload = json.loads(CONFIG_PATH.read_text())
    payload["evaluation"]["buckets"]["combined_hard"] = "fast_alignable"
    with pytest.raises(ValueError, match="four declared buckets"):
        MotionTrainingConfig.from_dict(payload)


def test_region_cap_keeps_every_case_so_no_bucket_disappears():
    config = tiny_config()
    evaluator = MotionEvaluator(config, device="cpu")
    selected = evaluator.specs("test", limit=3)
    assert {spec.case for spec in selected} == set(config.data.test.cases)
    assert all(spec.split == "test" for spec in selected)
    full = evaluator.specs("test", limit=10_000)
    assert len(full) == len(config.data.test.specs("test"))
    with pytest.raises(EvaluationConfigError, match="unknown region"):
        evaluator.specs("fitting", limit=1)


def test_align_to_moves_only_the_device():
    reference = torch.zeros(2, 2)
    other = torch.zeros(2, 2, dtype=torch.float64)
    moved = _align_to(other, reference)
    assert moved.device == reference.device
    assert moved.dtype == torch.float64  # dtype is the caller's business
    assert _align_to(reference, reference) is reference


def test_calibration_sampling_aligns_to_the_model_output_device(monkeypatch):
    """The reported CUDA defect: a device-only ``.to(dtype)`` kept the supervision on CPU.

    ``meta`` is the only second device a CPU-only machine has; a meta model output
    with a CPU supervision reproduces exactly the cross-device failure class seen
    with ``cuda:0`` and fails on the unfixed sampling path.
    """
    evaluator = MotionEvaluator(tiny_config(), device="cpu")
    evaluator._build_estimators()
    spec = evaluator.specs("calibration", limit=1)[0]
    original = evaluator.estimator.estimate

    def estimate_on_meta(reference, source, **kwargs):
        result = original(reference, source, **kwargs)
        return dataclasses.replace(
            result,
            packed_flow=result.packed_flow.to("meta"),
            valid=result.valid.to("meta"),
            flow_risk=result.flow_risk.to("meta"),
            visibility=result.visibility.to("meta"),
        )

    monkeypatch.setattr(evaluator.estimator, "estimate", estimate_on_meta)
    samples = _collect_calibration_samples(evaluator, [spec])
    assert samples
    devices = {tensor.device.type for sample in samples for tensor in sample.values()}
    assert devices == {"meta"}
    # every value the table will consume is on one device (the fit itself moves this
    # device to the host with .cpu()); meta cannot be copied to the host by design,
    # so the table fit is exercised on the real CPU path in the test above


def test_metric_paths_align_foreign_tensors_to_the_supervision(monkeypatch):
    """Every metric combination must ask for alignment onto the supervision device."""
    import owac.evaluation.n01_motion as motion_eval

    geometry = MotionDataGenerator.from_config_path(DATA_CONFIG).geometry()
    pair = MotionDataGenerator.from_config_path(DATA_CONFIG).sample_pair(
        split="train", case="layered_occlusion", sequence="s0"
    )
    target = flow_target_from_supervision(pair.supervision, geometry, mode="cfa4")
    prediction = target.flow + 1.0
    mask = torch.ones_like(target.valid)

    calls: list[tuple[torch.device, torch.device]] = []
    real = motion_eval._align_to

    def spy(tensor, reference):
        calls.append((tensor.device, reference.device))
        return real(tensor, reference)

    monkeypatch.setattr(motion_eval, "_align_to", spy)
    error_sensor, error_packed, _, _ = _pair_error(
        prediction,
        target,
        predicted_valid=mask,
        group_map=GroupMap(flow=prediction, valid=mask),
    )
    assert len(calls) == 2  # the shared flow and the validity mask
    assert all(reference == target.channel_flow.device for _tensor, reference in calls)
    assert error_packed.device == target.channel_flow.device
    assert error_sensor.device == target.channel_flow.device

    calls.clear()
    _cfa_phase_consistency(prediction, target, sensor_per_packed=2)
    assert len(calls) == 1
    assert calls[0][1] == target.channel_flow.device

    calls.clear()
    per_cell = error_sensor.mean(dim=0)
    _risk_radius_coverage(
        per_cell,
        torch.ones_like(per_cell),
        mask,
        fitted_coverage=torch.ones_like(per_cell),
    )
    assert len(calls) == 2  # the calibrated risk and the fitted coverage map
    assert all(reference == per_cell.device for _tensor, reference in calls)


def test_map_proxy_builds_bucket_boundaries_on_the_input_device(monkeypatch):
    risk = torch.tensor([[0.1, 0.2, 0.3, 0.4]] * 4, dtype=torch.float64)
    visibility = torch.full((4, 4), 0.5, dtype=torch.float64)
    table = ErrorCoverageTable.fit(
        [
            {
                "flow_risk": risk,
                "visibility": visibility,
                "error": risk * 2,
                "valid": torch.ones_like(risk, dtype=torch.bool),
            }
        ],
        split=FITTING_SPLIT,
        bins=4,
        nominal_coverage=0.9,
    )
    seen: list[tuple[torch.device, torch.device]] = []
    real_bucketize = torch.bucketize

    def spy(input, boundaries, **kwargs):
        seen.append((input.device, boundaries.device))
        return real_bucketize(input, boundaries, **kwargs)

    monkeypatch.setattr(torch, "bucketize", spy)
    mapped, coverage = table.map_proxy(flow_risk=risk, visibility=visibility)
    assert len(seen) == 2
    assert all(input_device == boundary_device for input_device, boundary_device in seen)
    assert mapped.device == risk.device and coverage.device == risk.device


@pytest.mark.gpu
def test_cuda_evaluation_runs_end_to_end(tmp_path):
    """The real evaluation path on CUDA: the defect fixed here surfaced here first."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is not available")
    config = load_motion_training_config(CONFIG_PATH)
    receipt = run_evaluation(config, output=tmp_path / "eval-cuda", device="cuda", max_pairs=8)
    assert receipt["latency"]["measured"] is True
    assert receipt["latency"]["device"] == "cuda"
    assert receipt["latency"]["peak_device_bytes"] > 0
    assert set(receipt["regions_report"]["test"]["arms"]) == set(GROUP_ORDER)
    assert receipt["regions_report"]["test"]["arms"]["oracle"]["epe_sensor_px"] == 0.0
    assert receipt["risk_calibration"]["fitted_split"] == "calibration"
    assert set(receipt["buckets"]) == set(BUCKET_LABELS)
    assert receipt["static_false_rejection"]["test"]["pairs"] >= 1


def test_static_pairs_are_bit_identical_and_the_zero_flow_arm_rejects_nothing():
    config = tiny_config()
    evaluator = MotionEvaluator(config, device="cpu")
    evaluator._build_estimators()
    specs = evaluator.specs("validation", limit=2)
    spec = specs[0]
    frames = evaluator.generator.frames(
        split=spec.split,
        case=spec.case,
        sequence=spec.sequence,
        length=2,
        input_mode=evaluator.input_mode,
    )
    scene = frames[-1]
    first = evaluator.generator.render(
        scene,
        split=spec.split,
        case=spec.case,
        sequence=spec.sequence,
        input_mode=evaluator.input_mode,
    )
    second = evaluator.generator.render(
        scene,
        split=spec.split,
        case=spec.case,
        sequence=spec.sequence,
        input_mode=evaluator.input_mode,
    )
    assert first.frame_id == second.frame_id
    assert torch.equal(first.values, second.values)

    evaluator.estimators["sea_raft"] = evaluator.estimators["no_alignment"]
    report = _static_false_rejection(evaluator, "validation", specs)
    assert report["pairs"] == 2
    assert report["false_rejection_rate"] == 0.0
    assert report["mean_predicted_displacement_sensor_px"] == pytest.approx(0.0)


def test_latency_report_is_end_to_end_and_ordered():
    config = tiny_config()
    evaluator = MotionEvaluator(config, device="cpu")
    evaluator._build_estimators()
    report = _latency_report(evaluator, evaluator.specs("test", limit=1))
    assert report["measured"] is True
    assert report["repeats"] == config.evaluation.latency_repeats
    assert report["p50_ms"] <= report["p95_ms"]
    assert report["parameter_count"] > 0
    assert report["peak_device_bytes"] is None
    assert report["device"] == "cpu"


def test_evaluation_checkpoint_must_belong_to_the_run_definition(tmp_path):
    config = tiny_config()
    with pytest.raises(FileNotFoundError):
        _load_checkpoint(
            config,
            config.data.resolved_data_config(),
            tmp_path / "missing.pt",
            device="cpu",
            normalization={},
        )
    path = tmp_path / "foreign.pt"
    torch.save(
        {
            "schema": "owac.n01_full.p2_motion_checkpoint.v1",
            "model": {},
            "core_fingerprint": "0" * 64,
        },
        path,
    )
    with pytest.raises(EvaluationConfigError, match="different run definition"):
        _load_checkpoint(
            config, config.data.resolved_data_config(), path, device="cpu", normalization={}
        )


def test_full_evaluation_keeps_the_four_way_split_and_lists_the_oracle_separately(tmp_path):
    config = tiny_config()
    output = tmp_path / "eval"
    receipt = run_evaluation(config, output=output, device="cpu")
    assert receipt["regions"]["fitted_on"] == "calibration"
    assert receipt["regions"]["validated_on"] == "validation"
    assert receipt["regions"]["reported_on"] == "test"
    assert receipt["regions"]["test_used_only_for_reporting"] is True
    assert receipt["risk_calibration"]["fitted_split"] == "calibration"
    assert receipt["oracle"] == {
        "role": "upper bound only",
        "listed_separately": True,
        "used_as_inference_input": False,
    }
    assert set(receipt["buckets"]) == set(BUCKET_LABELS)
    assert all(entry["cases"] for entry in receipt["buckets"].values())
    assert set(receipt["regions_report"]["test"]["arms"]) == set(GROUP_ORDER)
    assert receipt["regions_report"]["test"]["arms"]["oracle"]["epe_sensor_px"] == 0.0
    assert receipt["checkpoint"] is None
    assert receipt["latency"]["measured"] is True
    assert receipt["static_false_rejection"]["test"]["pairs"] >= 1
    coverage = receipt["regions_report"]["validation"]["calibration_coverage_on_validation"]
    assert coverage["by_arm"]["sea_raft"]["coverage_observed_mean"] is not None

    rows = [
        json.loads(line)
        for line in (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    assert all(row["oracle_input"] is False for row in rows)
    assert all(
        row["support_label"] in ("fixed_support", "oracle_support_diagnostic") for row in rows
    )
    assert (output / "risk_calibration.json").is_file()
    table = json.loads((output / "risk_calibration.json").read_text(encoding="utf-8"))
    assert table["fitted_split"] == "calibration"

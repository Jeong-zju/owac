"""Independent adversarial acceptance checks for the P2 RAW motion/noise hand-off.

This module is written by the independent verifier, not by the P2 implementer.  Every
assertion is meant to fail a plausible wrong implementation; the design follows three
rules:

* two implementations that should agree are cross-checked against each other instead
  of one being restated (the generator's analytic pull truth vs. the estimator's
  emitted ``source_xy``; the training units vs. the generator's packed targets);
* where only one implementation exists, the oracle is a NumPy FP64 formula, a closed
  form, or a hand-derived constant built here from the contract text;
* masks, resume and retraction are checked with gradients, bit equality and explicit
  positive controls, so a vacuous or content-blind path cannot pass.

The file only adds tests: no implementation module is touched.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from owac.backends.raw_correspondence import (
    SENSOR_PER_PACKED,
    CorrespondenceEstimator,
    RawSensorFrame,
)
from owac.backends.rawvla_features import ACTION_PROTOCOLS, RawVLAFeaturePolicy
from owac.core.encoder_contract import ComputeBudget, MeasurementFrame, VisualTokens
from owac.evaluation.n01_motion import ErrorCoverageTable, EvaluationConfigError
from owac.frameworks.n01_full import replay as replay_module
from owac.frameworks.n01_full.adapters import DbsrAdapterConfig, DbsrFeatureAdapter
from owac.frameworks.n01_full.config import ReplayConfig
from owac.frameworks.n01_full.contracts import SourceKey
from owac.frameworks.n01_full.domains import make_domain
from owac.frameworks.n01_full.ledger import MISSING, ContentReplacement
from owac.frameworks.n01_full.motion_builders import (
    LedgerCorrespondenceBuilder,
    LedgerNoiseModelBuilder,
)
from owac.frameworks.n01_full.statistics import (
    InterpolationSpec,
    align_interpolation_sources,
)
from owac.raw.n01_motion_data import (
    VISIBILITY_KINDS,
    MotionDataGenerator,
    parse_subpixel_token,
)
from owac.raw.n01_noise import (
    NoiseCalibration,
    NoiseProvenance,
    SharedBlackLevel,
    ShotReadParameters,
    UnsupportedNoiseStructureError,
    add_sensor_noise,
    apply_exposure_gain,
    build_noise_model,
    draw_block_error,
    interpolated_variance,
    shot_read_variance,
)
from owac.runtime import n01_full_policy_preflight as preflight
from owac.training.n01_motion import (
    MotionTrainer,
    MotionTrainingConfig,
    apply_selection,
    decide_platform,
    flow_target_from_supervision,
    packed_to_sensor,
    sensor_to_packed,
    sequence_loss,
)

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIG = REPO_ROOT / "configs/encoders/n01_full/p2_motion_noise.json"
TRAIN_CONFIG = REPO_ROOT / "configs/encoders/n01_full/p2_motion_train.json"
PROTOCOL = REPO_ROOT / "configs/encoders/n01_full/protocol_v1.json"

CFA_MODE = "synthetic_cfa"
RGB3_MODE = "rawvla_rgb3"
CFA_STEM = "cfa4"
RGB3_STEM = "rgb3"


# ---------------------------------------------------------------------------------------
# shared fixtures: injected packed-flow models and small RAW frames
# ---------------------------------------------------------------------------------------


class _PackedFlow(nn.Module):
    """A declared packed pull field; the estimator's documented model injection point."""

    def __init__(self, flow, *, info=None):
        super().__init__()
        self.register_buffer("flow", torch.as_tensor(flow, dtype=torch.float32))
        self.register_buffer("info", None if info is None else torch.as_tensor(info).float())
        self.calls = 0

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        self.calls += 1
        batch = int(image1.shape[0])
        output = {"final": self.flow.unsqueeze(0).expand(batch, -1, -1, -1).clone()}
        if self.info is not None:
            output["info"] = self.info.expand(batch, *self.info.shape[1:]).clone()
        return output


def _estimator(flow, *, mode=RGB3_STEM, info=None, **kwargs):
    return CorrespondenceEstimator(mode=mode, model=_PackedFlow(flow, info=info), **kwargs)


def _rgb3_frame(values, *, frame_id=0, valid=None):
    return RawSensorFrame(
        dataset_id="d",
        episode_id="e0",
        camera_id="c0",
        frame_id=frame_id,
        values=torch.as_tensor(values, dtype=torch.float32),
        channel_phase=(),
        cfa_pattern="rgb3",
        sensor_origin_xy=(0, 0),
        validity=valid,
    )


def _cfa_frame(values, *, phase=(0, 1, 2, 3), frame_id=0, valid=None):
    return RawSensorFrame(
        dataset_id="d",
        episode_id="e0",
        camera_id="c0",
        frame_id=frame_id,
        values=torch.as_tensor(values, dtype=torch.float32),
        channel_phase=tuple(phase),
        cfa_pattern="RGGB",
        sensor_origin_xy=(0, 0),
        validity=valid,
    )


def _generator() -> MotionDataGenerator:
    return MotionDataGenerator.from_config_path(DATA_CONFIG)


def _subpixel_pair(generator, token, *, mode=CFA_MODE):
    return generator.sample_pair(
        split="train",
        case="translation_subpixel",
        sequence=token,
        add_noise=False,
        input_mode=mode,
    )


def _grid_indices(geometry, channel):
    rows, cols = geometry.packed_shape
    if geometry.is_cfa:
        return geometry.grid_indices_yx(channel)
    return (
        torch.arange(rows, dtype=torch.int64).reshape(rows, 1),
        torch.arange(cols, dtype=torch.int64).reshape(1, cols),
    )


def _pull_flow_from_truth(supervision):
    """The packed pull field of channel 0: target packed coordinate minus the index."""
    packed = supervision.packed_xy_source
    rows, cols = packed.shape[1], packed.shape[2]
    index_y = torch.arange(rows, dtype=torch.float64).reshape(rows, 1)
    index_x = torch.arange(cols, dtype=torch.float64).reshape(1, cols)
    return torch.stack([packed[0, :, :, 1] - index_y, packed[0, :, :, 0] - index_x], dim=0)


# ---------------------------------------------------------------------------------------
# face 1: coordinates, units and the pull direction
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [CFA_MODE, RGB3_MODE])
@pytest.mark.parametrize("token", ["x+0.25", "x-0.25", "y+0.75", "y-0.75"])
def test_generator_truth_and_estimator_agree_on_the_pull_for_both_signs(token, mode):
    """Two independent implementations: analytic scene truth vs. estimator geometry."""
    generator = _generator()
    pair = _subpixel_pair(generator, token, mode=mode)
    supervision = pair.supervision
    reference, source = pair.visible.reference, pair.visible.source
    geometry = generator.geometry(mode)

    # the injected model is the generator's own truth, so any estimator coordinate flaw
    # (pull vs push, packed vs sensor, phase offset) shows up as a mismatch
    flow = _pull_flow_from_truth(supervision)
    estimator = _estimator(flow.to(torch.float32), mode=reference.mode)
    result = estimator.estimate(reference, source)

    dx_sensor, dy_sensor = parse_subpixel_token(token)
    checked = 0
    for channel in range(reference.channels):
        rows_index, cols_index = _grid_indices(geometry, channel)
        truth = supervision.sensor_xy_source[rows_index.reshape(-1)][:, cols_index.reshape(-1)]
        truth_y = truth[..., 1].reshape(*reference.packed_shape)
        truth_x = truth[..., 0].reshape(*reference.packed_shape)
        generator_valid = supervision.valid[rows_index.reshape(-1)][:, cols_index.reshape(-1)]
        generator_valid = generator_valid.reshape(*reference.packed_shape)
        valid = result.valid[channel]
        take = valid & generator_valid
        assert int(take.sum()) > 0, "no common cell to compare"
        got_y = result.source_xy[channel, :, :, 0]
        got_x = result.source_xy[channel, :, :, 1]
        assert torch.allclose(got_y[take], truth_y[take], atol=1e-9)
        assert torch.allclose(got_x[take], truth_x[take], atol=1e-9)
        checked += int(take.sum())

        # sign + unit oracle: the pull displacement off the reference sample must be the
        # token's sensor displacement for every channel (the phases share one flow here)
        origin_y, origin_x = reference.sensor_origin_xy
        offset_y, offset_x = reference.phase_offset(channel)
        rows = torch.arange(reference.packed_shape[0], dtype=torch.float64).reshape(-1, 1)
        cols = torch.arange(reference.packed_shape[1], dtype=torch.float64).reshape(1, -1)
        step = reference.sensor_per_packed
        displacement_y = got_y - (origin_y + step * rows + offset_y)
        displacement_x = got_x - (origin_x + step * cols + offset_x)
        count = int(valid.sum())
        assert torch.allclose(
            displacement_y[valid], torch.full((count,), dy_sensor, dtype=torch.float64), atol=1e-9
        )
        assert torch.allclose(
            displacement_x[valid], torch.full((count,), dx_sensor, dtype=torch.float64), atol=1e-9
        )
    assert checked > 0


def test_packed_and_sensor_units_differ_by_the_declared_stem_factor():
    generator = _generator()
    for mode, factor in ((CFA_MODE, 2), (RGB3_MODE, 1)):
        pair = _subpixel_pair(generator, "x+0.25", mode=mode)
        supervision = pair.supervision
        geometry = generator.geometry(mode)
        reference = pair.visible.reference
        assert reference.sensor_per_packed == factor == SENSOR_PER_PACKED[reference.mode]
        rows, cols = reference.packed_shape
        index_x = torch.arange(cols, dtype=torch.float64).reshape(1, cols)
        for channel in range(reference.channels):
            packed_displacement_x = supervision.packed_xy_source[channel][:, :, 0] - index_x
            rows_index, cols_index = _grid_indices(geometry, channel)
            sensor_x = supervision.sensor_xy_source[rows_index.reshape(-1)][
                :, cols_index.reshape(-1)
            ][..., 0].reshape(rows, cols)
            reference_x = torch.tensor(
                [
                    reference.sensor_xy_of(channel, row, col)[1]
                    for row in range(rows)
                    for col in range(cols)
                ],
                dtype=torch.float64,
            ).reshape(rows, cols)
            interior = supervision.valid[rows_index.reshape(-1)][:, cols_index.reshape(-1)]
            interior = interior.reshape(rows, cols) & (reference_x > 1.0)
            interior &= reference_x < geometry.width - 2.0
            assert int(interior.sum()) > 0
            assert torch.allclose(
                (sensor_x - reference_x)[interior],
                factor * packed_displacement_x[interior],
                atol=1e-9,
            )
            # the requested token is exactly 0.25 sensor px == 0.25/factor packed cells
            assert torch.allclose(
                packed_displacement_x[interior],
                torch.full((int(interior.sum()),), 0.25 / factor, dtype=torch.float64),
                atol=1e-9,
            )
    # training side of the same declared conversion
    assert float(packed_to_sensor(torch.tensor(0.125), 2)) == pytest.approx(0.25)
    assert float(sensor_to_packed(torch.tensor(0.25), 2)) == pytest.approx(0.125)


def _no_wrap_oracle(flow_y: float, flow_x: float, height: int, width: int) -> np.ndarray:
    """Independent NumPy rule: contributing taps must stay inside the source frame."""
    rows = np.arange(height, dtype=np.float64).reshape(-1, 1) + flow_y
    cols = np.arange(width, dtype=np.float64).reshape(1, -1) + flow_x
    i0, j0 = np.floor(rows), np.floor(cols)
    fy, fx = rows - i0, cols - j0
    weights = ((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx)
    tap_rows = (i0, i0, i0 + 1, i0 + 1)
    tap_cols = (j0, j0 + 1, j0, j0 + 1)
    ok = np.ones((height, width), dtype=bool)
    for weight, tap_row, tap_col in zip(weights, tap_rows, tap_cols, strict=True):
        contributes = weight > 0.0
        inside = (tap_row >= 0) & (tap_row < height) & (tap_col >= 0) & (tap_col < width)
        ok &= inside | ~contributes
    return ok


@pytest.mark.parametrize(
    ("flow_y", "flow_x"),
    [(0.0, -0.25), (0.0, 0.25), (0.25, 0.0), (-0.25, -0.25), (0.0, -1.0)],
)
def test_border_queries_follow_an_independent_no_wrap_oracle(flow_y, flow_x):
    side = 6
    values = torch.arange(3 * side * side, dtype=torch.float32).reshape(3, side, side)
    frame = _rgb3_frame(values)
    flow = torch.tensor([flow_y, flow_x]).reshape(2, 1, 1).expand(2, side, side).clone()
    result = _estimator(flow).estimate(frame, frame)
    expected = torch.as_tensor(_no_wrap_oracle(flow_y, flow_x, side, side))
    assert torch.equal(result.valid[0], expected)
    assert torch.equal(result.valid, expected.unsqueeze(0).expand(3, side, side))
    # every emitted tap is a real in-frame sample of the source: never wrapped, never invented
    for channel in range(3):
        for row in range(side):
            for col in range(side):
                if not bool(result.valid[channel, row, col]):
                    continue
                taps = result.taps_for_query(channel, row, col)
                assert taps
                for key, weight in taps:
                    assert weight > 0.0
                    assert 0 <= key.sensor_y < side and 0 <= key.sensor_x < side
                    assert (key.sensor_y, key.sensor_x) == frame.sensor_xy_of(
                        channel, key.sensor_y, key.sensor_x
                    )


def test_out_of_frame_flow_is_invalid_without_a_fabricated_sample():
    side = 4
    frame = _rgb3_frame(torch.ones(3, side, side))
    far = torch.full((2, side, side), -1e6)
    result = _estimator(far).estimate(frame, frame)
    # whole-map rejection: the contract has no placeholder sample for this
    assert result.valid is False
    assert result.tap_source_ids == ()
    assert result.tap_weights == ()
    assert result.source_identity is None
    # the coordinates are not wrapped back into the frame either
    assert bool((result.source_xy[..., 0] < 0).all())


def test_four_phases_keep_distinct_sensor_offsets_and_per_phase_validity():
    phase = (0, 3, 1, 2)  # channel 0 stays the origin anchor; the rest are permuted
    side = 4
    values = torch.arange(4 * side * side, dtype=torch.float32).reshape(4, side, side)
    reference = _cfa_frame(values, phase=phase)
    source = _cfa_frame(values, phase=phase)
    flow = torch.zeros(2, side, side)
    result = _estimator(flow, mode=CFA_STEM).estimate(reference, source)

    expected_offsets = ((0, 0), (1, 1), (0, 1), (1, 0))
    base = result.source_xy[0]
    for channel, (dy, dx) in enumerate(expected_offsets):
        assert source.phase_offset(channel) == (dy, dx)
        for row, col in ((0, 0), (1, 2), (3, 1)):
            key = source.sample_key(channel, row, col)
            assert key.channel_or_phase == phase[channel]
            assert (key.sensor_y, key.sensor_x) == (2 * row + dy, 2 * col + dx)
            # the sensor parity of the coordinate must be the declared phase
            assert 2 * (key.sensor_y % 2) + (key.sensor_x % 2) == phase[channel]
        assert torch.allclose(result.source_xy[channel] - base, torch.tensor([[dy, dx]]).double())
    assert result.phase_of_channel(2) == 1

    # per-phase validity: masking one source phase must invalidate only that phase's queries
    validity = torch.ones(4, side, side, dtype=torch.bool)
    validity[2] = False
    masked = _cfa_frame(values, phase=phase, valid=validity)
    masked_result = _estimator(flow, mode=CFA_STEM).estimate(reference, masked)
    assert not bool(masked_result.valid[2].any())
    assert bool(masked_result.valid[0].any()) and bool(masked_result.valid[1].any())
    for row in range(side):
        for col in range(side):
            for key, _weight in masked_result.taps_for_query(0, row, col):
                assert key.channel_or_phase == 0


# ---------------------------------------------------------------------------------------
# face 2: interpolation variance
# ---------------------------------------------------------------------------------------


def test_interpolation_variance_is_a_quadratic_form_not_a_variance_map():
    # Anticorrelated taps:  a.T Sigma a = (0.25 - 0.75)^2 * 9 = 2.25,
    # while interpolating the variance map gives 0.25*9 + 0.75*9 = 9.
    sigma_sq = 9.0
    covariance = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * sigma_sq
    weights = torch.tensor([[0.25, 0.75]])
    got = float(interpolated_variance(weights, covariance))
    assert got == pytest.approx(2.25, abs=1e-12)
    variance_map = float(weights @ torch.tensor([sigma_sq, sigma_sq]))
    assert variance_map == pytest.approx(sigma_sq)
    assert got != pytest.approx(variance_map)
    diagonal_only = float(weights.pow(2) @ torch.diagonal(covariance))
    assert got != pytest.approx(diagonal_only)


def test_fully_correlated_taps_reproduce_the_contract_bound_but_keep_the_exact_form():
    # rho = 1 with different per-tap sigmas: the exact quadratic form equals the P1
    # conservative bound (sum |a_i| sigma_i)^2 and both differ from the interpolated map.
    sigmas = torch.tensor([1.0, 2.0])
    covariance = torch.outer(sigmas, sigmas)
    weights = torch.tensor([[0.5, 0.5]])
    exact = float(interpolated_variance(weights, covariance))
    bound = float((weights.abs() * sigmas).sum().pow(2))
    interpolated_map = float(weights @ sigmas.pow(2))
    assert exact == pytest.approx(2.25)
    assert exact == pytest.approx(bound)
    assert exact != pytest.approx(interpolated_map)  # 2.5
    # a known negative correlation is smaller than the envelope: the exact form must not
    # be replaced by the conservative unknown-correlation bound
    negative = torch.tensor([[1.0, -2.0], [-2.0, 4.0]])
    small = float(interpolated_variance(torch.tensor([[0.5, 0.5]]), negative))
    assert small == pytest.approx(0.25)
    assert small != pytest.approx(bound)


def test_interpolation_variance_matches_a_numpy_double_loop_on_a_four_tap_block():
    generator = torch.Generator().manual_seed(11)
    matrix = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    covariance = matrix @ matrix.T + torch.eye(4, dtype=torch.float64)
    weights = torch.rand(7, 4, generator=generator, dtype=torch.float64)
    got = interpolated_variance(weights, covariance)
    dense = covariance.numpy()
    expected = np.array(
        [
            sum(w[i] * dense[i, j] * w[j] for i in range(4) for j in range(4))
            for w in weights.numpy()
        ]
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="has 3 taps but the covariance covers 4"):
        interpolated_variance(weights[:, :3], covariance)


# ---------------------------------------------------------------------------------------
# face 3: risk semantics (a proxy is not a posterior)
# ---------------------------------------------------------------------------------------


class _StubRiskTable:
    """A declared calibration table with a distinctive, verifiable mapping."""

    def __init__(self, *, version="p2-verify-table-v1", risk=7.0, coverage=0.5, shape=None):
        self.version = version
        self.risk = risk
        self.coverage = coverage
        self.shape = shape
        self.calls = 0

    def map_proxy(self, *, flow_risk, visibility):
        self.calls += 1
        shape = self.shape or tuple(flow_risk.shape)
        return (
            torch.full(shape, self.risk, dtype=flow_risk.dtype),
            torch.full(shape, self.coverage, dtype=visibility.dtype),
        )


def _info_tensor(shape, *, weight=(0.0, 0.0), log_b=(0.0, 0.0)):
    info = torch.zeros(*shape, dtype=torch.float32)
    info[:, 0] = weight[0]
    info[:, 1] = weight[1]
    info[:, 2] = log_b[0]
    info[:, 3] = log_b[1]
    return info


def test_proxy_risk_is_never_presented_as_a_calibrated_posterior():
    side = 4
    frame = _rgb3_frame(torch.ones(3, side, side))
    info = _info_tensor(
        (1, 4, side, side),
        weight=(math.log(3.0), 0.0),
        log_b=(math.log(2.0), math.log(0.25)),
    )
    result = _estimator(torch.zeros(2, side, side), info=info, var_min=-5.0).estimate(frame, frame)
    assert result.risk_kind == "proxy"
    assert result.calibrated_risk is None and result.coverage is None
    # independent oracle of the declared mixture mean (sensor pixels):
    # p = softmax(log 3, 0) = (0.75, 0.25), scales exp(clamp(log 2)) = 2 and exp(-log 4) = 0.25
    scale = 0.75 * 2.0 + 0.25 * 0.25
    assert torch.allclose(result.flow_risk, torch.full_like(result.flow_risk, scale), atol=1e-6)
    assert torch.allclose(result.visibility, torch.full_like(result.visibility, 0.25), atol=1e-6)
    # the declared default var_min = 0.0 clamps the small log-scale into [0, 0], i.e. scale 1:
    # the proxy must follow that declared clamp instead of reading the raw log_b
    default = _estimator(torch.zeros(2, side, side), info=info).estimate(frame, frame)
    assert torch.allclose(
        default.flow_risk, torch.full_like(default.flow_risk, 0.75 * 2.0 + 0.25 * 1.0), atol=1e-6
    )

    # without any proxy scores the risk is explicitly unavailable, not a default number
    silent = _estimator(torch.zeros(2, side, side)).estimate(frame, frame)
    assert silent.risk_kind == "unavailable"
    assert silent.flow_risk is None and silent.calibrated_risk is None


def test_a_risk_table_really_switches_the_version_and_the_fields():
    side = 4
    frame = _rgb3_frame(torch.ones(3, side, side))
    info = _info_tensor((1, 4, side, side))
    table = _StubRiskTable(version="p2-verify-table-v9", risk=3.5, coverage=0.8)
    result = _estimator(torch.zeros(2, side, side), info=info, risk_calibration=table).estimate(
        frame, frame
    )
    assert table.calls == 1
    assert result.risk_kind == "calibrated:p2-verify-table-v9"
    assert torch.equal(result.calibrated_risk, torch.full((side, side), 3.5))
    assert torch.equal(result.coverage, torch.full((side, side), 0.8))
    # the proxy fields stay the raw proxy, not silently relabelled as calibrated
    assert result.flow_risk is not None
    assert not torch.equal(result.flow_risk, result.calibrated_risk)


def test_wrong_shaped_risk_fields_are_refused():
    side = 4
    frame = _rgb3_frame(torch.ones(3, side, side))
    info = _info_tensor((1, 4, side, side))
    bad = _StubRiskTable(shape=(side + 1, side))
    with pytest.raises(ValueError, match="packed grid"):
        _estimator(torch.zeros(2, side, side), info=info, risk_calibration=bad).estimate(
            frame, frame
        )
    with pytest.raises(TypeError, match="RiskCalibrationTable"):
        CorrespondenceEstimator(
            mode=RGB3_STEM,
            model=_PackedFlow(torch.zeros(2, side, side)),
            risk_calibration=object(),
        )
    # a table that cannot map (no proxy scores) must be refused, not silently skipped
    with pytest.raises(ValueError, match="no proxy scores"):
        _estimator(torch.zeros(2, side, side), risk_calibration=_StubRiskTable()).estimate(
            frame, frame
        )


def test_evaluation_error_coverage_table_fits_on_calibration_only():
    with pytest.raises(EvaluationConfigError, match="calibration"):
        ErrorCoverageTable.fit((), split="test")
    with pytest.raises(EvaluationConfigError, match="calibration"):
        ErrorCoverageTable.fit((), split="validation")
    table = ErrorCoverageTable.fit(
        (
            {
                "flow_risk": torch.linspace(0.0, 1.0, 8),
                "visibility": torch.linspace(1.0, 0.0, 8),
                "error": torch.linspace(0.0, 2.0, 8),
                "valid": torch.ones(8, dtype=torch.bool),
            },
        ),
        split="calibration",
    )
    assert table.fitted_split == "calibration"


# ---------------------------------------------------------------------------------------
# face 4: noise law, shared black level, exposure and non-finite inputs
# ---------------------------------------------------------------------------------------


def _noise_calibration(*, block_variance=0.25, audited=True, upper_bound=None):
    provenance = NoiseProvenance(
        kind="explicit_synthesis", split="synthesis", source="p2-verifier", method="declared"
    )
    return NoiseCalibration(
        calibration_version="p2-verify-v1",
        units="dn",
        shot_read=ShotReadParameters(shot=0.5, read_var=2.0, units="dn", provenance=provenance),
        black_level=SharedBlackLevel(
            kind="per_frame",
            id_template="{dataset_id}:{episode_id}:{camera_id}:{frame_id}",
            variance=block_variance if audited else None,
            upper_bound=upper_bound,
            provenance=provenance,
        ),
        black_level_dn=100.0,
    )


def _source(frame_id, phase=0, y=0, x=0):
    return SourceKey("d", "e", "c", frame_id, phase, y, x)


def test_shot_read_law_matches_an_independent_numpy_formula_and_keeps_non_finite():
    means = np.array([-5.0, 99.0, 100.0, 101.0, 1e6, np.nan])
    shot, read_var, black = 0.5, 2.0, 100.0
    got = shot_read_variance(means, shot=shot, read_var=read_var, black_level=black).numpy()
    expected = shot * np.clip(means - black, 0.0, None) + read_var
    np.testing.assert_allclose(got[:5], expected[:5], rtol=0.0, atol=1e-12)
    assert got[0] == pytest.approx(2.0)  # below black: read-only
    assert got[3] == pytest.approx(2.5)
    assert math.isnan(float(got[5]))  # a NaN mean is a NaN variance, not a substituted one
    assert not (float(got[5]) == read_var)
    # an overflow is reported as such, never clipped to a plausible finite number
    assert math.isinf(float(shot_read_variance([1e308], shot=2.0, read_var=2.0)[0]))
    with pytest.raises(ValueError, match="non-negative"):
        shot_read_variance(means, shot=-1.0, read_var=read_var)


def test_shared_black_level_block_really_correlates_same_block_samples():
    calibration = _noise_calibration(block_variance=0.25)
    keys = [_source(0, 0), _source(0, 1), _source(1, 0), _source(1, 1)]
    means = [110.0, 120.0, 130.0, 140.0]
    model = build_noise_model(keys, means, calibration)
    assert model.kind == "dense"
    variance = 0.5 * np.clip(np.array(means) - 100.0, 0.0, None) + 2.0
    expected = np.diag(variance + 0.25)
    for left, right in ((0, 1), (1, 0), (2, 3), (3, 2)):
        expected[left, right] = 0.25
    np.testing.assert_allclose(
        model.covariance_for(keys, dtype=torch.float64).numpy(), expected, rtol=0.0, atol=1e-12
    )
    # no correlation across blocks
    assert float(model.matrix[0, 2]) == 0.0 and float(model.matrix[1, 3]) == 0.0
    assert float(model.variance_of(keys[0])) == pytest.approx(variance[0] + 0.25)

    # a single sample per block degenerates to the diagonal model (off-diagonals are zero);
    # the marginal of that sample is checked separately, with the exact declared number
    diagonal = build_noise_model([_source(0, 0), _source(1, 0)], [110.0, 130.0], calibration)
    assert diagonal.kind == "diagonal"


def test_a_single_sample_block_still_carries_its_black_level_variance():
    """The declared marginal of one sample never depends on its fit membership.

    One marginal rule for every branch: a sample in a ``noise_block_id`` carries the
    audited ``sigma_b**2`` on top of its own shot/read variance.  The exact number is
    checked against the declared calibration and against the dense branch for the same
    sample set, so an implementation that drops the block term in one branch only, or
    that adds it in the wrong unit, fails.
    """
    calibration = _noise_calibration(block_variance=0.25)
    key = _source(0, 0)
    alone = build_noise_model([key], [110.0], calibration)
    assert alone.kind == "diagonal"
    # 0.5 * (110 - 100) + 2.0 shot/read + 0.25 audited block = 7.25 exactly
    assert float(alone.variance_of(key)) == pytest.approx(7.25)
    assert float(alone.variance_of(key)) == pytest.approx(float(calibration.variance([110.0])[0]))

    # the same sample with a block sibling must report the same marginal
    sibling = build_noise_model([key, _source(0, 1)], [110.0, 120.0], calibration)
    assert sibling.kind == "dense"
    assert float(sibling.variance_of(key)) == pytest.approx(float(alone.variance_of(key)))


def test_unaudited_shared_block_is_explicitly_refused_or_reported_unknown():
    calibration = _noise_calibration(audited=False, upper_bound=0.5)
    keys = [_source(0, 0), _source(0, 1)]
    with pytest.raises(UnsupportedNoiseStructureError, match="unknown"):
        build_noise_model(keys, [110.0, 120.0], calibration)
    with pytest.raises(UnsupportedNoiseStructureError):
        draw_block_error("d:e:c:0", calibration, torch.Generator().manual_seed(0))
    unknown = build_noise_model(keys, [110.0, 120.0], calibration, correlation="unknown")
    assert unknown.kind == "unknown_correlation"
    # the conservative record adds the declared bound to the marginal; it is not a zero
    base = 0.5 * np.array([10.0, 20.0]) + 2.0
    np.testing.assert_allclose(np.array(unknown.variances), base + 0.5, rtol=0.0, atol=1e-12)


def test_exposure_gain_moves_value_and_variance_with_the_declared_power():
    values = torch.tensor([100.0, 200.0, 1000.0], dtype=torch.float64)
    variance = torch.tensor([4.0, 9.0, 16.0], dtype=torch.float64)
    gain, black = 3.0, 50.0
    scaled_values, scaled_variance = apply_exposure_gain(
        values, variance, gain=gain, black_level=black
    )
    assert torch.allclose(scaled_values, black + gain * (values - black))
    assert torch.allclose(scaled_variance, gain * gain * variance)
    with pytest.raises(ValueError, match="non-negative"):
        apply_exposure_gain(values, variance, gain=-1.0)
    with pytest.raises(ValueError, match="finite"):
        apply_exposure_gain(values, variance, gain=float("nan"))
    # the parameter rescaling and the value rescaling describe the same gain
    calibration = _noise_calibration()
    means = torch.tensor([10.0, 20.0], dtype=torch.float64)
    scaled = calibration.shot_read.scaled(gain)
    assert torch.allclose(
        scaled.variance(gain * means), gain * gain * calibration.shot_read.variance(means)
    )


def test_noise_realisation_is_reproducible_and_keeps_the_shared_offset():
    calibration = _noise_calibration()
    clean = torch.tensor([10.0, 20.0], dtype=torch.float64)
    baseline = add_sensor_noise(
        clean,
        calibration=calibration,
        generator=torch.Generator().manual_seed(3),
        block_error=0.0,
    )
    shifted = add_sensor_noise(
        clean,
        calibration=calibration,
        generator=torch.Generator().manual_seed(3),
        block_error=0.75,
    )
    again = add_sensor_noise(
        clean,
        calibration=calibration,
        generator=torch.Generator().manual_seed(3),
        block_error=0.75,
    )
    assert torch.equal(shifted, again)
    # the block error is shared by the whole block: it is one constant offset here
    assert torch.allclose(shifted - baseline, torch.full((2,), 0.75, dtype=torch.float64))
    nan_drawn = add_sensor_noise(
        torch.tensor([float("nan")]),
        calibration=calibration,
        generator=torch.Generator().manual_seed(3),
    )
    assert bool(torch.isnan(nan_drawn).all())


# ---------------------------------------------------------------------------------------
# face 5: data generation, truth isolation and the split discipline
# ---------------------------------------------------------------------------------------


def _walk(value, path="$", seen=None):
    """Walk a real object graph: yields ``(path, object)`` for every reachable node."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, f"{path}[{key!r}]", seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]", seen)
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            yield from _walk(getattr(value, name), f"{path}.{name}", seen)


def test_the_model_visible_half_shares_no_tensor_and_no_label_with_the_truth_half():
    generator = _generator()
    pair = generator.sample_pair(
        split="train", case="layered_occlusion", sequence="s0", add_noise=False
    )
    truth_fields = {
        "sensor_xy_source",
        "sensor_xy_reference",
        "packed_xy_source",
        "visible_layer_reference",
        "visible_layer_source",
        "layer_ids",
        "visibility",
        "supervision",
        "ambiguity_period_px",
    }
    truth_tensor_ids = set()
    for _path, node in _walk(pair.supervision):
        if isinstance(node, torch.Tensor):
            truth_tensor_ids.add(id(node))
    assert truth_tensor_ids, "the truth walker must see the analytic tensors"

    leaked_names = []
    shared_tensors = []
    seen_truth_object = False
    for path, node in _walk(pair.visible):
        if isinstance(node, Mapping):
            leaked_names.extend(key for key in node if key in truth_fields)
        elif hasattr(node, "__dataclass_fields__"):
            leaked_names.extend(name for name in node.__dataclass_fields__ if name in truth_fields)
        if node is pair.supervision:
            seen_truth_object = True
        if isinstance(node, torch.Tensor) and id(node) in truth_tensor_ids:
            shared_tensors.append(path)
    assert leaked_names == []
    assert shared_tensors == [], f"a truth tensor is reachable from visible: {shared_tensors}"
    assert seen_truth_object is False
    # the visible half is still real data, not an empty shell
    assert pair.visible.reference.values.numel() > 0
    assert pair.visible.metadata["case"] == "layered_occlusion"


def test_longer_sequences_do_not_change_earlier_frames_and_same_seed_is_bit_identical():
    generator = _generator()
    long_run = generator.frames(split="train", case="combined_hard", sequence="s0", length=5)
    short_run = generator.frames(split="train", case="combined_hard", sequence="s0", length=2)
    interleaved = generator.frames(split="test", case="rotation", sequence="other", length=3)
    assert interleaved  # a different draw between two calls must not change the result
    repeat = generator.frames(split="train", case="combined_hard", sequence="s0", length=2)
    for scene in (long_run[0], long_run[1]):
        shorter = short_run[scene.frame_id]
        assert scene.frame_id == shorter.frame_id and scene.time == shorter.time
        assert [layer.layer_id for layer in scene.layers] == [
            layer.layer_id for layer in shorter.layers
        ]
        long_render = generator.render(
            scene, split="train", case="combined_hard", sequence="s0", add_noise=False
        )
        short_render = generator.render(
            shorter, split="train", case="combined_hard", sequence="s0", add_noise=False
        )
        assert torch.equal(long_render.values, short_render.values)
    for first, second in zip(short_run, repeat, strict=True):
        rendered = generator.render(
            first, split="train", case="combined_hard", sequence="s0", add_noise=False
        )
        again = generator.render(
            second, split="train", case="combined_hard", sequence="s0", add_noise=False
        )
        assert torch.equal(rendered.values, again.values)


def test_seed_namespace_separates_data_and_model_streams_and_is_stable():
    generator = _generator()
    rule = generator.config.seed_rule
    assert set(rule.model_seeds).isdisjoint({rule.data_seed_base})
    namespace = generator.namespace("train", "rotation", "s0")
    assert namespace.startswith("n01-full-v1/train/rotation/s0")
    # independent oracle of the declared hash rule: base | namespace -> sha256 -> modulus
    payload = f"{rule.data_seed_base}|{namespace}".encode()
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert generator.seed("train", "rotation", "s0") == digest % rule.modulus
    assert generator.seed("test", "rotation", "s0") != generator.seed("train", "rotation", "s0")
    assert generator.seed("train", "motion_blur", "s0") != generator.seed("train", "rotation", "s0")
    pair = generator.sample_pair(split="train", case="rotation", sequence="s0", add_noise=False)
    again = _generator().sample_pair(split="train", case="rotation", sequence="s0", add_noise=False)
    assert torch.equal(pair.visible.reference.values, again.visible.reference.values)
    assert torch.equal(pair.supervision.sensor_xy_source, again.supervision.sensor_xy_source)


def _region(prefix: str) -> dict:
    cases: dict = {"translation_subpixel": ["x+0.25", "x-0.25"]}
    for case in (
        "rotation",
        "layered_occlusion",
        "local_nonrigid",
        "repeated_texture",
        "exposure_gain",
        "motion_blur",
        "combined_hard",
    ):
        cases[case] = {"prefix": prefix, "count": 1}
    return {"cases": cases, "history_offsets": [1]}


def _training_config(**overrides) -> MotionTrainingConfig:
    payload = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    payload["model"].update({"depth": "resnet18", "iters": 0, "pretrained_checkpoint": None})
    payload["data"].update(
        {
            "config_path": str(DATA_CONFIG),
            "geometry": None,
            "recipe_scale": 1.0,
            "batch_size": 1,
            "cache_entries": 4,
            "train": _region("tr"),
            "calibration": _region("ca"),
            "validation": _region("va"),
            "test": _region("te"),
        }
    )
    payload["selection"] = {"lrs": [0.00025, 5e-05], "iters": [0], "updates": 1}
    payload["budget"] = {
        "updates": 8,
        "max_updates": 16,
        "validation_interval": 2,
        "plateau_checks": 8,
        "delta_val": 0.002,
        "monitor": "val_epe_sensor_px",
    }
    payload["schedule"] = {"drops": [[2, 0.1], [4, 0.01]], "min_drop_ratio": 0.1}
    payload["diagnostics"] = {
        "fit_updates": 1,
        "check_updates": 1,
        "fit_batch": 1,
        "fit_max_ratio": 0.999999,
        "fit_target": 1e9,
        "input_limit": 8.0,
    }
    payload["checkpoint"] = {"every": 2, "keep_last": 1}
    for key in ("max_calibration_pairs", "max_validation_pairs", "max_test_pairs"):
        payload["evaluation"][key] = 4
    payload["evaluation"]["latency_warmup"] = 1
    payload["evaluation"]["latency_repeats"] = 1
    payload.update(overrides)
    return MotionTrainingConfig.from_dict(payload)


def test_test_region_cannot_be_used_for_fitting_or_selection(tmp_path):
    config = _training_config()
    policy = config.data.resolved_data_config().policy
    assert policy.fitting_splits == ("train", "calibration")
    assert policy.selection_split == "validation"
    assert policy.evaluation_split == "test"
    assert all(spec.split == "train" for spec in config.data.train.specs("train"))
    assert all(spec.split == "test" for spec in config.data.test.specs("test"))
    with pytest.raises(EvaluationConfigError):
        ErrorCoverageTable.fit((), split="test")

    receipt = {
        "winner": {"lr": 0.00025, "iters": 0},
        "selection_split": "test",
        "test_split_used": True,
        "config_fingerprint": config.core_fingerprint,
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        apply_selection(config, path)
    receipt["selection_split"] = "validation"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="test"):
        apply_selection(config, path)
    receipt["test_split_used"] = False
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert apply_selection(config, path) == {"lr": 0.00025, "iters": 0}


# ---------------------------------------------------------------------------------------
# face 6: training semantics (masks, units, plateau, resume)
# ---------------------------------------------------------------------------------------


def test_invalid_cells_contribute_no_gradient_through_the_author_loss():
    prediction = torch.zeros(1, 2, 5, 5, dtype=torch.float64, requires_grad=True)
    flow_gt = torch.full((1, 2, 5, 5), 0.4, dtype=torch.float64)
    valid = torch.zeros(1, 5, 5, dtype=torch.bool)
    valid[:, 2:, :] = True

    def author_nll():
        return [((prediction - flow_gt) ** 2).sum(dim=1)]

    loss = sequence_loss({"flow": [None], "nf": author_nll()}, flow_gt, valid)
    assert torch.isfinite(loss)
    loss.backward()
    gradient = prediction.grad
    assert float(gradient[0, :, :2, :].abs().max()) == 0.0, "an invalid cell still trains"
    assert float(gradient[0, :, 2:, :].abs().sum()) > 0.0, "valid cells must carry gradient"

    # an empty mask is a zero loss, not a NaN and not a fabricated unit loss
    prediction.grad = None
    empty_loss = sequence_loss(
        {"flow": [None], "nf": author_nll()}, flow_gt, torch.zeros_like(valid)
    )
    assert float(empty_loss.detach()) == 0.0 and torch.isfinite(empty_loss)
    empty_loss.backward()
    assert float(prediction.grad.abs().sum()) == 0.0

    # a cell pushed beyond max_flow leaves the loss exactly, including its gradient
    excluded = flow_gt.clone()
    excluded[0, :, 4, 4] = 1e6
    prediction.grad = None
    sequence_loss({"flow": [None], "nf": author_nll()}, excluded, valid).backward()
    assert float(prediction.grad[0, :, 4, 4].abs().sum()) == 0.0


def test_occluded_and_out_of_bounds_cells_are_excluded_from_the_target_mask():
    generator = _generator()
    for case in ("layered_occlusion", "combined_hard"):
        pair = generator.sample_pair(split="train", case=case, sequence="s0", add_noise=False)
        supervision = pair.supervision
        geometry = generator.geometry()
        target = flow_target_from_supervision(supervision, geometry, mode="cfa4")
        expected = torch.zeros_like(target.valid)
        for channel in range(4):
            rows_index, cols_index = _grid_indices(geometry, channel)
            channel_valid = supervision.valid[rows_index.reshape(-1)][:, cols_index.reshape(-1)]
            channel_valid = channel_valid.reshape(*target.valid.shape)
            assert torch.equal(target.channel_valid[channel], channel_valid)
            expected |= channel_valid
            occluded = supervision.visibility != VISIBILITY_KINDS.index("valid")
            occluded_channel = occluded[rows_index.reshape(-1)][:, cols_index.reshape(-1)]
            occluded_channel = occluded_channel.reshape(*target.valid.shape)
            assert not bool((target.channel_valid[channel] & occluded_channel).any())
        assert torch.equal(target.valid, expected)
        assert int((supervision.visibility != VISIBILITY_KINDS.index("valid")).sum()) > 0
        # the excluded cells never silently become a zero target that trains
        assert bool((target.flow[:, ~target.valid] == 0).all())


def test_the_loss_target_uses_packed_cells_not_sensor_pixels():
    generator = _generator()
    cfa = flow_target_from_supervision(
        _subpixel_pair(generator, "x+0.25", mode=CFA_MODE).supervision,
        generator.geometry(CFA_MODE),
        mode="cfa4",
    )
    valid = cfa.valid
    assert bool(valid.any())
    count = int(valid.sum())
    assert torch.allclose(cfa.flow[1][valid], torch.full((count,), 0.125), atol=1e-6)
    assert torch.allclose(cfa.flow[0][valid], torch.zeros(count), atol=1e-6)
    assert torch.allclose(cfa.flow_sensor_px()[1][valid], torch.full((count,), 0.25), atol=1e-6)
    # the wrong unit (sensor pixels used as packed cells) is excluded by the check above
    assert not torch.allclose(cfa.flow[1][valid], torch.full((count,), 0.25), atol=1e-6)
    rgb3 = flow_target_from_supervision(
        _subpixel_pair(generator, "x+0.25", mode=RGB3_MODE).supervision,
        generator.geometry(RGB3_MODE),
        mode="rgb3",
    )
    assert torch.allclose(
        rgb3.flow[1][rgb3.valid], torch.full((int(rgb3.valid.sum()),), 0.25), atol=1e-6
    )


def _history(values, *, interval=1_000, start=1_000):
    return [
        {"update": start + index * interval, "val_epe_sensor_px": float(value)}
        for index, value in enumerate(values)
    ]


def _decide(history, **overrides):
    kwargs = {
        "budget": 100_000,
        "interval": 1_000,
        "monitor": "val_epe_sensor_px",
        "delta_val": 0.01,
        "plateau_checks": 8,
        "lr_drops": 2,
        "final_phase": True,
        "diagnostics_passed": True,
    }
    kwargs.update(overrides)
    return decide_platform(val_history=history, **kwargs)


def test_plateau_needs_a_real_improvement_and_a_flat_tail():
    values = [10.0 - index for index in range(10)] + [1.0] * 10
    decision = _decide(_history(values))
    assert decision.status == "validation_plateau"
    assert all(decision.details["conditions"].values())
    creeping = _decide(_history([10.0 - 1.0 * index for index in range(20)]))
    assert creeping.status == "budget_exhausted"


def test_a_degrading_validation_curve_is_not_a_plateau():
    """A curve that only gets worse is not convergence, whatever its tail looks like.

    Both decisive conditions must be named, so an implementation that only special-cases
    ``best_index > 0`` without checking the tail drift still fails this test.
    """
    decision = _decide(_history([1.0 + index for index in range(12)]))
    assert decision.status == "budget_exhausted"
    assert decision.details["conditions"]["real_improvement_over_run"] is False
    assert decision.details["conditions"]["tail_within_delta_val_of_best"] is False
    assert any("real_improvement_over_run" in reason for reason in decision.reasons)
    assert any("tail_within_delta_val_of_best" in reason for reason in decision.reasons)


def test_a_constant_validation_curve_is_not_a_plateau():
    """No validation ever improved, so the flat tail is not evidence of a plateau."""
    decision = _decide(_history([5.0] * 12))
    assert decision.status == "budget_exhausted"
    # the tail *does* look flat; the run-level improvement is the condition that refuses it
    assert decision.details["conditions"]["flat_tail_below_delta_val"] is True
    assert decision.details["conditions"]["real_improvement_over_run"] is False
    assert decision.details["run_improvement"] == pytest.approx(0.0)
    assert any("real_improvement_over_run" in reason for reason in decision.reasons)


def test_a_flat_tail_that_drifts_away_from_its_best_is_not_converged():
    """A plateau is flat *at* its best, not a tail that slid back down from it."""
    values = [10.0 - index for index in range(10)]
    values += [1.0 + 0.02 * step for step in range(1, 11)]
    decision = _decide(_history(values))
    assert decision.status == "budget_exhausted"
    # the run improved, and its tail is flat in the delta_val sense: only the drift is left
    assert decision.details["conditions"]["real_improvement_over_run"] is True
    assert decision.details["conditions"]["flat_tail_below_delta_val"] is True
    assert decision.details["conditions"]["tail_within_delta_val_of_best"] is False
    assert decision.details["tail_drift"] == pytest.approx(0.2)
    assert any("tail_within_delta_val_of_best" in reason for reason in decision.reasons)


def _nested_state_equal(left, right, path="state"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert set(left) == set(right), f"{path}: key mismatch"
        for key in left:
            _nested_state_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _nested_state_equal(a, b, f"{path}[{index}]")
        return
    assert left == right, f"{path}: {left!r} != {right!r}"


def _rng_triple():
    numpy_state = np.random.get_state()
    return (random.getstate(), numpy_state, torch.get_rng_state().clone())


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[1][0] == right[1][0] and left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_resume_replays_the_next_batches_losses_optimizer_and_rng(tmp_path):
    config = _training_config()
    checkpoint = tmp_path / "motion.pt"
    continuous = MotionTrainer(config, device="cpu")
    continuous.diagnostics = {"passed": True}
    for update in range(2):
        continuous.advance(update=update)
    continuous.save_checkpoint(checkpoint, update=2)
    optimizer_at_split = copy.deepcopy(continuous.optimizer.state_dict())
    rng_at_split = _rng_triple()
    sampler_at_split = continuous.stream.receipt()
    consumed_at_split = continuous.consumed_samples
    for update in range(2, 4):
        continuous.advance(update=update)

    resumed = MotionTrainer(config, device="cpu")
    resumed.diagnostics = {"passed": True}
    # perturb every global stream first: a resume that forgets to restore the RNG would
    # otherwise still look identical whenever the forward pass consumes no global entropy
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    assert resumed.load_checkpoint(checkpoint) == 2
    assert resumed.consumed_samples == consumed_at_split
    _assert_rng_equal(rng_at_split, _rng_triple())
    assert resumed.stream.receipt() == sampler_at_split
    _nested_state_equal(optimizer_at_split, resumed.optimizer.state_dict(), "optimizer")
    for update in range(2, 4):
        resumed.advance(update=update)

    assert resumed.batch_log == continuous.batch_log[2:]
    assert resumed.train_losses == continuous.train_losses[2:]
    assert resumed.lr_history == continuous.lr_history[2:]
    _nested_state_equal(continuous.model.state_dict(), resumed.model.state_dict(), "model")


# ---------------------------------------------------------------------------------------
# face 7: DBSR adapter pooling, parameter accounting, arrivals and missing cameras
# ---------------------------------------------------------------------------------------


def _adapter_config(**overrides) -> DbsrAdapterConfig:
    payload = {
        "input_mode": "rawvla_rgb3",
        "stem_width": 4,
        "residual_blocks": 1,
        "feature_width": 8,
        "projection_width": 4,
        "offset_width": 8,
        "max_history": 4,
        "pool_grid": 2,
        "tokens_per_camera": 4,
        "readout_width": 20,
    }
    payload.update(overrides)
    return DbsrAdapterConfig(**payload)


def _adapter_budget(adapter):
    return ComputeBudget(max_updates=1, max_history=adapter.config.max_history)


def _frame(values, *, camera="a", frame_ids=None, valid=None, episodes=None):
    batch, channels, height, width = values.shape
    if valid is None:
        valid = torch.ones(batch, channels, height, width, dtype=torch.bool)
    return MeasurementFrame(
        values=values,
        valid_mask=valid,
        frame_ids=None if frame_ids is None else torch.as_tensor(frame_ids),
        episode_ids=episodes,
        camera_id=camera,
        timestamp=None,
        new_measurement=None,
    )


def test_pooling_marks_a_fully_invalid_cell_false_and_ignores_its_values():
    adapter = DbsrFeatureAdapter(_adapter_config())
    adapter.eval()
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    valid = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    valid[:, :, 4:, 0:2] = True  # only the bottom-left 2x4 strip is measured
    clean = torch.randn(1, 3, 8, 8)
    clean[~valid] = 0.0
    with torch.no_grad():
        output = adapter.step(
            _frame(clean, frame_ids=[0], valid=valid), None, state, _adapter_budget(adapter)
        )
        tokens = adapter.readout(output)
        garbage = clean.clone()
        garbage[~valid] = 1e4
        other = adapter.step(
            _frame(garbage, frame_ids=[0], valid=valid), None, state, _adapter_budget(adapter)
        )
        other_tokens = adapter.readout(other)
    assert tokens.mask[0].tolist() == [False, False, True, False]
    assert float(tokens.values[0, 2].abs().sum()) > 0.0
    for index, present in enumerate(tokens.mask[0].tolist()):
        if not present:
            assert float(tokens.values[0, index].abs().sum()) == 0.0
    assert torch.equal(tokens.values, other_tokens.values), "invalid pixels entered a token"
    assert torch.equal(tokens.mask, other_tokens.mask)


def test_no_hidden_square_readout_projection_is_allocated():
    adapter = DbsrFeatureAdapter(_adapter_config())
    post = adapter.build_postprocessor(("a", "b"))
    assert post.in_proj.in_features == adapter.config.feature_width
    assert post.in_proj.out_features == adapter.config.readout_width
    linears = [module for module in adapter.modules() if isinstance(module, nn.Linear)]
    linears += [module for module in post.modules() if isinstance(module, nn.Linear)]
    width = adapter.config.readout_width
    hidden = [module for module in linears if module.in_features == module.out_features == width]
    assert hidden == [], f"a hidden {width}x{width} projection is allocated: {hidden}"
    assert post.in_proj.weight.numel() == adapter.config.feature_width * width
    # the reported total must cover every real parameter: no hidden allocation
    report = post.parameter_report()
    assert int(report["total"]) == sum(parameter.numel() for parameter in post.parameters())
    assert sum(parameter.numel() for parameter in adapter.parameters()) > 0


class _CountingVLM(nn.Module):
    """Stub VLM surface: the image entry point counts and raises when used."""

    def __init__(self, width, vocab=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.image_entries = 0

    def embed_language_tokens(self, tokens):
        return self.embed(tokens)

    def embed_image(self, image):
        self.image_entries += 1
        raise AssertionError("the image tower must never run on the raw adapter path")

    def forward_prefix(
        self, *, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache=True
    ):
        return SimpleNamespace(past_key_values=SimpleNamespace(prefix_embs=inputs_embeds))


class _StubActionHead(nn.Module):
    """Interface replica of the pinned action head used by the feature bridge."""

    def __init__(self, width, action_dim=4, horizon=3):
        super().__init__()
        self.action_in_proj = nn.Linear(action_dim, width)
        self.hidden = nn.Linear(width, width)
        self.out = nn.Linear(width, action_dim)
        self.action_dim = action_dim
        self.action_horizon = horizon

    def sample_action_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def prepare_flow_targets(self, actions, noise=None, time=None):
        time = torch.full((actions.shape[0],), 0.5) if time is None else time
        noise = torch.zeros_like(actions) if noise is None else noise
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        return x_t, noise - actions, noise, time

    def embed_action_suffix(self, state, x_t, timestep):
        embs = self.action_in_proj(x_t)
        pad = torch.ones(embs.shape[:2], dtype=torch.bool)
        return embs, pad, torch.zeros_like(pad), None

    def decode_action_velocity(self, hidden):
        return self.out(hidden)


class _StubPolicy(nn.Module):
    """Interface replica of the PI0/PI05 surface the bridge programs against."""

    def __init__(self, width, action_dim=4, horizon=3):
        super().__init__()
        self.vlm = _CountingVLM(width)
        self.action_head = _StubActionHead(width, action_dim, horizon)
        self.action_dim = action_dim
        self.action_horizon = horizon
        self.max_state_dim = 4
        self.num_inference_steps = 10
        self.state_pad_value = 0.0
        self.discrete_state_input = False

    def forward_with_action_expert(
        self, *, prefix_embs, suffix_embs, attention_mask, position_ids, adarms_cond
    ):
        return self.action_head.hidden(suffix_embs) + prefix_embs.mean(1, keepdim=True)

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        context = past_key_values.prefix_embs.mean(1, keepdim=True)
        hidden = self.action_head.hidden(self.action_head.action_in_proj(x_t))
        return self.action_head.out(hidden + context)


def test_the_image_tower_never_runs_on_the_raw_adapter_path():
    torch.manual_seed(2)
    adapter = DbsrFeatureAdapter(_adapter_config())
    post = adapter.build_postprocessor(("a", "b"))
    policy = _StubPolicy(width=adapter.config.readout_width)
    bridge = RawVLAFeaturePolicy(
        policy, encoder=adapter, postprocessor=post, action_protocol=ACTION_PROTOCOLS["libero"]
    )
    bridge.freeze_backend_parameters()
    state = adapter.init_state(1, {"camera_ids": ["a", "b"], "shape": (8, 8)}, "cpu")
    output = None
    for step in range(2):
        for camera in ("a", "b"):
            frame = _frame(
                torch.randn(1, 3, 8, 8), camera=camera, frame_ids=[step], episodes=["e0"]
            )
            output = adapter.step(frame, None, state, _adapter_budget(adapter))
            state = output.state
    tokens = adapter.readout(output)
    frozen = VisualTokens(
        values=tokens.values.detach(),
        mask=tokens.mask.detach(),
        camera_ids=list(tokens.camera_ids),
        positions=None if tokens.positions is None else tokens.positions.detach(),
    )
    observation = SimpleNamespace(
        state=torch.zeros(1, 1, 4),
        tokenized_prompt=torch.randint(1, 16, (1, 4)),
        tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    actions = torch.randn(1, 3, 4)
    adapter.train()
    post.train()
    result = bridge.forward_from_observation_with_features(
        observation, tokens, actions, return_debug=True
    )
    result["action_loss"].backward()
    assert policy.vlm.image_entries == 0
    assert result["prefix_source"] == "visual_tokens"
    per_camera = adapter.config.tokens_per_camera
    assert torch.equal(result["pad_masks"][:, : per_camera * 2], tokens.mask)
    assert (
        sum(
            float(parameter.grad.norm())
            for parameter in adapter.parameters()
            if parameter.grad is not None
        )
        > 0.0
    )
    assert all(parameter.grad is None for parameter in policy.parameters())

    calls = bridge.stats["encoder_calls"]
    steps_before = bridge.stats["denoise_steps"]
    sampled = bridge.sample_actions_with_features(
        observation, frozen, noise=torch.zeros(1, 3, 4), num_steps=10
    )
    assert bridge.stats["encoder_calls"] == calls
    assert bridge.stats["denoise_steps"] - steps_before == 10
    assert policy.vlm.image_entries == 0
    assert sampled.shape == (1, 3, 4) and bool(torch.isfinite(sampled).all())


def test_one_step_observes_exactly_one_arrival_event():
    adapter = DbsrFeatureAdapter(_adapter_config())
    adapter.eval()
    state = adapter.init_state(2, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    values = torch.randn(2, 3, 8, 8)
    first = adapter.step(_frame(values, frame_ids=[0, 0]), None, state, _adapter_budget(adapter))
    assert int(first.diagnostics["window_frames"]) == 1
    assert first.diagnostics["unique_frames_this_step"] == [True, True]
    assert int(state.step_index) == 0
    second = adapter.step(
        _frame(values, frame_ids=[0, 0]), None, first.state, _adapter_budget(adapter)
    )
    assert second.diagnostics["unique_frames_this_step"] == [False, False]
    assert second.diagnostics["repeated_this_step"] == [True, True]
    assert int(first.state.cameras["a"].filled.max()) == 1
    assert int(second.state.cameras["a"].filled.max()) == 1
    assert int(second.state.arrivals["a"].max()) == int(first.state.arrivals["a"].max()) + 1
    assert torch.equal(second.state.unique_arrivals["a"], first.state.unique_arrivals["a"])
    assert int(second.state.step_index) == int(first.state.step_index) + 1


def test_a_silent_camera_never_pollutes_another_camera():
    adapter = DbsrFeatureAdapter(_adapter_config())
    adapter.eval()
    state = adapter.init_state(1, {"camera_ids": ["a", "b"], "shape": (8, 8)}, "cpu")
    output = None
    for step in range(3):
        current = output.state if output is not None else state
        output = adapter.step(
            _frame(torch.randn(1, 3, 8, 8), camera="a", frame_ids=[step]),
            None,
            current,
            _adapter_budget(adapter),
        )
    tokens_a = adapter.readout(output)
    assert tokens_a.camera_ids == ["a"]
    snapshot = output.state.tokens["a"].values.clone()
    after_b = adapter.step(
        _frame(torch.randn(1, 3, 8, 8) + 5.0, camera="b", frame_ids=[0]),
        None,
        output.state,
        _adapter_budget(adapter),
    )
    assert torch.equal(after_b.state.tokens["a"].values, snapshot)
    assert after_b.state.tokens["a"].window_frames == 3
    assert after_b.state.tokens["b"].window_frames == 1
    both = adapter.readout(after_b)
    assert both.camera_ids == ["a", "b"]
    assert both.values.shape[1] == 2 * adapter.config.tokens_per_camera


# ---------------------------------------------------------------------------------------
# face 8: ledger recomputation, retraction and the exact window
# ---------------------------------------------------------------------------------------


LEDGER_DATASET = "p2-verify"
LEDGER_EPISODE = "e0"
LEDGER_CAMERA = "c0"
LEDGER_GRID = 3


class _ContentFlow(nn.Module):
    """A packed-flow model whose output depends on the two frames' content."""

    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = float(scale)
        self.calls = 0

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        self.calls += 1
        batch = int(image1.shape[0])
        delta = (image1 - image2).reshape(batch, -1).mean(dim=1)
        flow = torch.zeros(batch, 2, *image1.shape[-2:], dtype=image1.dtype, device=image1.device)
        flow[:, 0] = 0.125
        flow[:, 1] = self.scale * torch.tanh(10.0 * delta)[:, None, None]
        return {"final": flow}


def _ledger_keys(frames):
    return [
        SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, frame, channel, y, x)
        for frame in range(frames)
        for channel in range(3)
        for y in range(LEDGER_GRID)
        for x in range(LEDGER_GRID)
    ]


def _observe(frame_id, *, spike=1.0):
    generator = torch.Generator().manual_seed(1000 + frame_id)
    values = torch.randn(3, LEDGER_GRID, LEDGER_GRID, generator=generator)
    values[0, 0, 0] = spike
    return replay_module.ObserveEvent(
        MeasurementFrame(
            values=values[None],
            frame_ids=[frame_id],
            episode_ids=[LEDGER_EPISODE],
            camera_id=LEDGER_CAMERA,
            timestamp=torch.tensor([float(frame_id)]),
            new_measurement=torch.tensor([True]),
        )
    )


def _ledger_stream(frames):
    return replay_module.EventStream(
        LEDGER_DATASET, tuple(_observe(frame) for frame in range(frames))
    )


def _ledger_domain_builder(keys):
    def builder(ledger):
        evidence = [
            key for key in keys if ledger.has_source(key) and ledger.validity_of(key) == "valid"
        ]
        return [
            make_domain(
                "D",
                reference_chart="ref",
                basis="phase_local_constant",
                phase=None,
                evidence_source_ids=evidence,
                decision_version="dv1",
                mapping_version="mv1",
            )
        ]

    return builder


def _ledger_builder(keys):
    return LedgerCorrespondenceBuilder(
        CorrespondenceEstimator(mode=RGB3_STEM, model=_ContentFlow(), iters=1),
        evidence_ids=keys,
        cfa_pattern="rgb3",
        camera_id=LEDGER_CAMERA,
    )


def test_retracted_content_cannot_reach_a_content_sensitive_counterfactual_rebuild():
    keys = _ledger_keys(3)
    builder = _ledger_builder(keys)
    retracted = SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, 0, 0, 0, 0)
    retained = SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, 2, 0, 1, 1)
    ledger, _ = replay_module.consume(_ledger_stream(3))
    view = ledger.without([retracted])
    assert view.has_source(retracted) is False
    base = builder(view)

    # positive control: the builder path is content-sensitive, so the invariance below
    # cannot hold for a vacuous reason
    changed = view.with_content(ContentReplacement("positive-control", {retained: 50.0}))
    assert not torch.equal(builder(changed).weights, base.weights)

    # the retracted content is erased before the decision callback: replacing it with
    # anything at all cannot change one bit of the rebuild
    for trial, value in enumerate((1e3, -1e3, 0.0)):
        hidden = view.with_content(
            ContentReplacement(f"retracted-content-{trial}", {retracted: value})
        )
        assert hidden.has_source(retracted) is False
        rebuilt = builder(hidden)
        assert torch.equal(rebuilt.weights, base.weights)
        assert rebuilt.source_ids == base.source_ids
        assert rebuilt.valid_rows == base.valid_rows


def test_builder_columns_match_has_source_and_the_same_view_order():
    keys = _ledger_keys(3)
    builder = _ledger_builder(keys)
    ledger, _ = replay_module.consume(_ledger_stream(3))
    retracted = SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, 1, 1, 2, 0)
    view = ledger.without([retracted])
    rebuilt = builder(view)
    usable = builder.usable_columns(view)
    assert rebuilt.source_ids == tuple(usable)
    assert all(view.has_source(key) for key in rebuilt.source_ids)
    assert all(view.validity_of(key) == "valid" for key in rebuilt.source_ids)
    assert retracted not in rebuilt.source_ids
    frame = view.get_frame(LEDGER_EPISODE, LEDGER_CAMERA, 1)
    assert frame is not None and int(frame.validity[1, 2, 0]) == MISSING
    sums = rebuilt.weights.sum(dim=1)
    assert float(sums.min()) > 1.0 - 1e-12 and float(sums.max()) < 1.0 + 1e-12
    nonzero = rebuilt.weights.abs() > 0.0
    assert int(nonzero.sum()) > 0
    for column in range(rebuilt.weights.shape[1]):
        if bool(nonzero[:, column].any()):
            assert rebuilt.source_ids[column].identity() != retracted.identity()


def test_interpolation_columns_are_re_indexed_by_identity_not_by_position():
    keys = _ledger_keys(3)
    builder = _ledger_builder(keys)
    ledger, _ = replay_module.consume(_ledger_stream(3))
    rebuilt = builder(ledger)
    used = builder.usable_columns(ledger)
    aligned = align_interpolation_sources(rebuilt, used)
    assert aligned.source_ids == tuple(used)

    order = list(range(len(rebuilt.source_ids)))
    random.Random(7).shuffle(order)
    permuted = InterpolationSpec(
        row_positions=rebuilt.row_positions,
        weights=rebuilt.weights[:, order],
        version=rebuilt.version,
        source_ids=tuple(rebuilt.source_ids[index] for index in order),
    )
    aligned_permuted = align_interpolation_sources(permuted, used)
    assert torch.equal(aligned.weights, aligned_permuted.weights)
    assert aligned_permuted.source_ids == tuple(used)

    # a non-zero weight on a column the domain does not carry is refused, not dropped
    missing = keys[0]
    position = rebuilt.source_ids.index(missing)
    used_without = [key for key in used if key.identity() != missing.identity()]
    broken = torch.zeros_like(rebuilt.weights)
    broken[:, position] = 1e-3
    dropped = InterpolationSpec(
        row_positions=rebuilt.row_positions,
        weights=broken,
        version=rebuilt.version,
        source_ids=rebuilt.source_ids,
    )
    with pytest.raises(ValueError, match="outside the domain's usable evidence set"):
        align_interpolation_sources(dropped, used_without)
    # a zero-weight column is dropped without changing the remaining rows
    zeroed = torch.zeros_like(rebuilt.weights)
    empty_column = InterpolationSpec(
        row_positions=rebuilt.row_positions,
        weights=zeroed,
        version=rebuilt.version,
        source_ids=rebuilt.source_ids,
    )
    assert torch.equal(
        align_interpolation_sources(empty_column, used_without).weights,
        torch.zeros(len(rebuilt.row_positions), len(used_without), dtype=torch.float64),
    )


def test_sources_outside_the_exact_replay_window_never_reach_the_builder():
    replay_config = ReplayConfig()
    assert replay_config.exact_window_unique_frames == 32
    total_frames = replay_config.exact_window_unique_frames + 2
    keys = _ledger_keys(total_frames)
    builder = _ledger_builder(keys)
    seen = []

    def probe(view):
        seen.append(view)
        return builder(view)

    inside = SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, 5, 0, 0, 0)
    result = replay_module.counterfactual_replay(
        _ledger_stream(total_frames),
        domain_builder=_ledger_domain_builder(keys),
        retracted_ids=[inside],
        readout_grid=replay_module.ReadoutGrid(()),
        correspondence_builder=probe,
        replay_config=replay_config,
    )
    assert len(seen) == 1
    view = seen[0]
    retained = view.unique_frames(episode_id=LEDGER_EPISODE, camera_id=LEDGER_CAMERA)
    assert len(retained) == replay_config.exact_window_unique_frames
    retained_ids = sorted(int(identity.frame_id) for identity in retained)
    assert retained_ids == list(range(2, total_frames))
    for outside in (0, 1):
        assert not view.has_source(
            SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, outside, 0, 0, 0)
        )
    assert view.has_source(
        SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, total_frames - 1, 0, 0, 0)
    )
    rebuilt = builder(view)
    assert all(int(frame_id) >= 2 for frame_id in rebuilt.source_frame_ids)
    # the retraction targets an in-window source, so the claim stays the strong one
    assert result.claim == "exact_independence_in_event_stream"
    loss_frames = {int(item.frame.frame_id) for item in result.history_loss}
    assert loss_frames == {0, 1}


def test_noise_builder_recomputes_from_the_view_and_uses_the_declared_law():
    keys = _ledger_keys(2)
    calibration = _noise_calibration()
    noise_builder = LedgerNoiseModelBuilder(calibration, tuple(keys))
    retracted = SourceKey(LEDGER_DATASET, LEDGER_EPISODE, LEDGER_CAMERA, 0, 2, 1, 1)
    ledger, _ = replay_module.consume(_ledger_stream(2))
    view = ledger.without([retracted])
    model = noise_builder(view)
    assert model.version == calibration.noise_version
    identities = {key.identity() for key in model.source_order}
    assert retracted.identity() not in identities
    assert len(model.source_order) == len(keys) - 1
    assert model.kind == "dense"  # the per-frame shared block correlates the two phases
    means = [float(view.get_source(key).raw_value) for key in model.source_order]
    base = 0.5 * np.clip(np.array(means) - 100.0, 0.0, None) + 2.0 + 0.25
    np.testing.assert_allclose(np.array(model.variances), base, rtol=0.0, atol=1e-12)
    # a content replacement of a retained source changes the recomputed means
    other = [key for key in keys if key.identity() != retracted.identity()][-1]
    replaced = view.with_content(ContentReplacement("noise-control", {other: 900.0}))
    assert not np.allclose(np.array(noise_builder(replaced).variances), base)


# ---------------------------------------------------------------------------------------
# face 9: preflight receipt honesty
# ---------------------------------------------------------------------------------------


def test_raw_adapter_smoke_receipt_never_claims_acceptance():
    report = preflight.raw_adapter_smoke(SimpleNamespace(seed=3))
    assert report["mode"] == "raw_adapter"
    assert report["acceptance"] is False
    assert "stub" in report["backend"]
    assert "synthetic" in report["data"]
    assert report["failures"] == []
    assert "acceptance" in report["note"]
    checks = report["checks"]
    assert checks["vision_entry_calls"] == 0
    assert checks["p3_support_enabled"] is False
    assert checks["aux_raw_loss_enabled"] is False


def test_missing_frozen_assets_fail_hard_without_writing_a_receipt(tmp_path):
    output = tmp_path / "preflight-out"
    with pytest.raises(FileNotFoundError, match="absent"):
        preflight.main(
            [
                "--mode",
                "raw_adapter",
                "--protocol",
                str(PROTOCOL),
                "--platform",
                "libero",
                "--output",
                str(output),
            ]
        )
    assert not (output / "raw-adapter-preflight.json").exists()
    assert not (output / "raw-adapter-smoke.json").exists()

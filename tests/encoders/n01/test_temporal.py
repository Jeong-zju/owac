"""Measurement refutation, noise calibration and real motion/start/stop behavior."""

from types import SimpleNamespace

import pytest
import torch

from owac.evaluation.n01_temporal import validate_motion
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.model import N01Encoder
from owac.frameworks.n01.motion import CausalCorrespondence, MotionField
from owac.frameworks.n01.temporal import temporal_innovation_cost


def state_for(values, *, variance=0.001, observed=None, valid=None):
    slots, batch = values.shape[:2]
    return SimpleNamespace(
        values=values,
        variance=torch.full_like(values, variance),
        valid=torch.ones_like(values, dtype=torch.bool) if valid is None else valid,
        observed=torch.ones(slots, batch, dtype=torch.bool) if observed is None else observed,
    )


def cost(state, depths=(1, 2, 4), *, planes=1, radius=2, aligned=None):
    if aligned is None:
        aligned = state.values, state.variance, state.valid
    return temporal_innovation_cost(state, aligned, depths, planes=planes, radius=radius)


def test_coherent_change_is_detected_below_the_single_pixel_threshold():
    # Each pixel is only two standard deviations from its previous reading.
    values = torch.zeros(2, 1, 1, 9, 9)
    values[1, :, :, :, 4] = 0.04
    state = state_for(values, variance=0.0002)
    assert bool((cost(state, radius=0) == 0).all())
    result = cost(state, depths=(1, 2))
    assert float(result[1, 0, 0, 4, 4]) == pytest.approx(11, rel=1e-5)
    assert bool((result[0] == 0).all())
    assert torch.equal(result, cost(state_for(-values, variance=0.0002), depths=(1, 2)))


def test_cost_is_monotonic_only_over_observed_history_and_keeps_cfa_planes_separate():
    values = torch.zeros(4, 2, 4, 7, 7)
    values[1:, :, 2] = 0.4
    observed = torch.tensor([[True, True], [True, False], [False, False], [True, False]])
    result = cost(state_for(values, observed=observed), planes=4)
    assert bool((result[:, :, [0, 1, 3]] == 0).all())
    assert bool((result[:, 1] == 0).all())
    assert bool((result[1] >= result[0]).all() & (result[2] >= result[1]).all())
    assert float(result[2, 0, 2, 3, 3]) == pytest.approx(2 * (0.4**2 * 5 / 0.0015 - 9), rel=1e-5)


def test_copied_channels_do_not_create_more_independent_evidence():
    values = torch.zeros(4, 1, 1, 7, 7)
    values[1:, :, :, :, 3] = 0.08
    single = cost(state_for(values))
    copied = cost(state_for(values.expand(-1, -1, 3, -1, -1).clone()))
    torch.testing.assert_close(single, copied, atol=1e-5, rtol=1e-5)


def test_bad_channels_and_nan_values_cannot_hide_or_create_refutation():
    values = torch.zeros(2, 1, 3, 7, 7)
    values[1, :, 0] = 0.2
    values[1, :, 1:] = float("nan")
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[1, :, 1:] = False
    result = cost(state_for(values, valid=valid))
    assert bool(torch.isfinite(result).all())
    assert float(result[1].mean()) > 0
    valid.zero_()
    assert bool((cost(state_for(values, valid=valid)) == 0).all())


def test_exact_alignment_preserves_compatible_moving_measurements():
    config = N01Config.tiny(alignment="fixed", fractional_alignment=False)
    estimator = CausalCorrespondence(config, neural=False)
    current = torch.zeros(1, 3, 9, 9)
    current[:, :, :, 4] = 0.3
    source = torch.roll(current, shifts=1, dims=-1)
    state = state_for(torch.stack([current, source]))
    field = MotionField(
        torch.tensor([[0.0, 1.0]]), torch.zeros(1), torch.ones(1), torch.ones(1, dtype=torch.bool)
    )
    aligned = estimator.align(source, state.variance[1], state.valid[1], field)
    result = cost(
        state,
        aligned=tuple(
            torch.stack([a[0], b])
            for a, b in zip((state.values, state.variance, state.valid), aligned, strict=True)
        ),
    )
    assert bool((result == 0).all())
    assert float(cost(state)[1].max()) > 0


def test_box_variance_bound_includes_shared_fractional_interpolation_sources():
    # A half-pixel warp plus a 3-pixel box has source coefficients .5,1,1,.5.
    # True historical variance is 2.5*v, bounded by the linear sum 3*v.
    values = torch.zeros(2, 1, 1, 7, 7)
    values[1] = 0.1
    state = state_for(values, variance=0.001)
    current_variance, source_bound = 3 * 0.001, 3 * 0.001
    actual_source_variance = sum(c * c * 0.001 for c in (0.5, 1, 1, 0.5))
    assert actual_source_variance <= source_bound
    expected = 0.3**2 / (current_variance + source_bound) - 9
    assert float(cost(state, radius=1)[1, 0, 0, 3, 3]) == pytest.approx(expected, rel=1e-5)


def test_noise_excess_units_never_fit_a_near_zero_motion_scale_and_old_weights_reject():
    encoder = N01Encoder(N01Config.tiny(width=8))
    encoder.calibrating = True
    payload = {
        key: torch.zeros(2, 1, 1, 4, 4)
        for key in ("variance_proxy", "age", "support_count", "unique_frames", "motion_error", "mu")
    }
    scale = encoder._statistics_scale(payload, torch.arange(2))
    assert float(scale["motion_error"]) == 1
    old = encoder.state_dict()
    del old["temporal_cost_version"]
    with pytest.raises(RuntimeError, match="temporal_cost_version"):
        encoder.load_state_dict(old)


def test_actual_history_shortens_on_motion_and_recovers_after_stopping():
    result = validate_motion(batch_size=4, seed=2100)
    assert result["still_history"] > 7.5
    assert result["paired_positive_members"] == 4
    assert result["fast_history"] < result["still_history"] - 0.7
    assert result["fast_peak_history"] < result["still_peak_history"] - 2
    assert result["fast_stop_history"] > result["fast_history"] + 0.7
    # A quarter-pixel step is a weaker signal; it must respond in the same
    # direction, without prescribing the reduction of the one-pixel motion.
    assert result["subpixel_history"] < result["still_history"]

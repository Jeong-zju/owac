"""Hand arithmetic, causal identities and explicit failure boundaries of source risk."""

from dataclasses import replace

import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.model import N01Encoder
from owac.frameworks.n01.motion import MotionField
from owac.frameworks.n01.reliability import _reference_on_source, innovation_weights
from owac.frameworks.n01.sequence1d import N01Sequence1DEncoder
from owac.frameworks.n01.state import init_state, push, reset
from owac.frameworks.n01.support import SupportDomainIntegrator


def config(**overrides):
    args = dict(
        channels=4,
        planes=4,
        input_mode="bayer_cfa",
        width=8,
        num_blocks=1,
        spatial_radii=(0,),
        neighbourhood_radius=0,
        time_depths=(8,),
        learnable_gate=False,
        aux_measurement_prediction=False,
        content_mode="integrated",
        source_reliability="innovation_risk",
    )
    return N01Config.tiny(**(args | overrides))


def frame(value, index, variance=0.01, valid=None):
    x = torch.as_tensor(value).float()
    if x.ndim == 0:
        x = x.expand(1, 4, 5, 5).clone()
    return MeasurementFrame(
        values=x,
        valid_mask=torch.ones_like(x, dtype=torch.bool) if valid is None else valid,
        camera_id="camera",
        frame_ids=torch.full((x.shape[0],), index),
        timestamp=torch.full((x.shape[0],), float(index)),
        time_unit="frame",
        metadata={"variance": torch.full_like(x, variance)},
    )


def state_for(cfg, values, variances=None):
    state = init_state(cfg, 1, {"camera_ids": ["camera"], "shape": (5, 5)})
    for i, value in enumerate(values):
        state = push(state, frame(value, i, 0.01 if variances is None else variances[i]))
    return state


def statistics(cfg, values, variances=None):
    state = state_for(cfg, values, variances)
    integrator = SupportDomainIntegrator(cfg)
    stats, _ = integrator.domain_statistics(state, torch.ones(1, 4, 2, 5, 5))
    return stats, integrator


def test_heteroscedastic_hand_integral_retained_without_excess_innovation():
    stats, integrator = statistics(config(), [0.1, 0.3], [0.04, 0.01])
    torch.testing.assert_close(stats.mu, torch.full_like(stats.mu, 0.26), atol=3e-6, rtol=0)
    torch.testing.assert_close(
        stats.variance_proxy, torch.full_like(stats.mu, 0.008), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(stats.age, torch.full_like(stats.mu, 0.2), atol=2e-5, rtol=0)
    assert torch.all(integrator.last_source_reliability[:2] == 1)


def test_eight_compatible_sources_keep_noise_reduction():
    stats, integ = statistics(config(), [0.3] * 8)
    torch.testing.assert_close(stats.mu, torch.full_like(stats.mu, 0.3))
    torch.testing.assert_close(stats.variance_proxy, torch.full_like(stats.mu, 0.01 / 8))
    torch.testing.assert_close(stats.age, torch.full_like(stats.mu, 3.5))
    torch.testing.assert_close(integ.last_history_mass, torch.full_like(stats.mu, 7 / 8))
    assert torch.all(stats.unique_frames == 8)


@pytest.mark.parametrize("old_count", [1, 3, 7])
def test_common_stale_bias_cannot_be_averaged_away(old_count):
    stats, integ = statistics(config(), [0.0] * old_count + [1.0])
    torch.testing.assert_close(stats.mu, torch.full_like(stats.mu, 0.99), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        integ.last_history_mass, torch.full_like(stats.mu, 0.01), atol=1e-5, rtol=0
    )
    # Soft suppression is not removal: count alone would conceal the response.
    assert torch.all(stats.unique_frames == old_count + 1)


def test_hard_rejection_and_channel_separation():
    old = torch.tensor([3.0, 0.2, 0.3, 0.4])[None, :, None, None].expand(1, 4, 5, 5)
    new = old.clone()
    new[:, 0] = 0
    stats, integ = statistics(config(), [old, new])
    assert torch.all(integ.last_source_reliability[1, :, 0] == 0)
    assert torch.all(integ.last_source_reliability[1, :, 1:] == 1)
    assert torch.all(stats.unique_frames[:, :, 0] == 1)
    assert torch.all(stats.unique_frames[:, :, 1:] == 2)
    torch.testing.assert_close(stats.mu[0], new)


def test_mixed_good_and_wrong_sources_not_whole_window_drop():
    stats, integ = statistics(config(), [0.3, 1.0, 0.3])
    assert 0.306 < float(stats.mu.mean()) < 0.309
    assert torch.all(integ.last_source_reliability[2] == 1)
    assert torch.all(integ.last_source_reliability[1] < 0.03)


def test_missing_current_preserves_old_evidence_and_age():
    cfg = config()
    state = state_for(cfg, [0.3, 0.3])
    state = push(state, frame(0, 2, valid=torch.zeros(1, 4, 5, 5, dtype=torch.bool)))
    stats, _ = SupportDomainIntegrator(cfg).domain_statistics(state, torch.ones(1, 4, 2, 5, 5))
    torch.testing.assert_close(stats.mu, torch.full_like(stats.mu, 0.3))
    torch.testing.assert_close(stats.age, torch.full_like(stats.mu, 1.5))
    assert torch.all(stats.unique_frames == 2)


def test_empty_state_produces_no_evidence():
    cfg = config()
    state = state_for(cfg, [])
    stats, _ = SupportDomainIntegrator(cfg).domain_statistics(state, torch.ones(1, 4, 2, 5, 5))
    for name in ["mu", "age", "variance_proxy", "support_count", "unique_frames", "unique_samples"]:
        assert not getattr(stats, name).any()


def test_inverse_fractional_reference_squares_noise_coefficients():
    x = torch.arange(5).float()[None, None, None].expand(1, 4, 5, 5)
    ref, variance, valid = _reference_on_source(
        x,
        torch.full_like(x, 0.04),
        torch.ones_like(x, dtype=torch.bool),
        torch.tensor([[0.0, 0.5]]),
        True,
    )
    torch.testing.assert_close(ref[..., 1:], x[..., 1:] - 0.5)
    torch.testing.assert_close(variance[..., 1:], torch.full_like(variance[..., 1:], 0.02))
    assert not valid[..., 0].any()
    assert valid[..., 1:].all()


@pytest.mark.parametrize("shift", [1.0, 0.5])
def test_correct_alignment_keeps_linear_measurement_history(shift):
    cfg = config(alignment="fixed", fractional_alignment=True, time_depths=(2,))
    current = torch.arange(5).float()[None, None, None].expand(1, 4, 5, 5) / 5
    state = state_for(cfg, [current - shift / 5, current])
    fields = [
        MotionField(
            torch.tensor([[0.0, 0.0 if i == 0 else shift]]),
            torch.zeros(1),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            fractional=torch.zeros(1, 2),
        )
        for i in range(8)
    ]
    integ = SupportDomainIntegrator(cfg)
    stats, _ = integ.domain_statistics(state, torch.ones(1, 4, 2, 5, 5), motion=fields)
    torch.testing.assert_close(stats.mu[0, ..., 1:3], current[..., 1:3])
    assert torch.all(integ.last_source_reliability[1, ..., 1:4] == 1)


def test_fractional_shared_sources_are_collated_before_squared_moments():
    cfg = config(
        alignment="fixed",
        fractional_alignment=True,
        spatial_radii=(1,),
        neighbourhood_radius=1,
        time_depths=(2,),
    )
    state = state_for(cfg, [0.3, 0.3])
    fields = [
        MotionField(
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            fractional=torch.tensor([[0.0, 0.0 if i == 0 else 0.5]]),
        )
        for i in range(8)
    ]
    integ = SupportDomainIntegrator(cfg)
    edges = torch.ones(1, 4, 2, 5, 5)
    stats, _ = integ.domain_statistics(state, edges, motion=fields)
    # 9 current unique sources have coefficient 1; historical columns carry
    # [0.5, 1, 1, 0.5], repeated over 3 rows. 12 old identities, not 18 paths.
    coeffs = torch.tensor([1.0] * 9 + [0.5, 1.0, 1.0, 0.5] * 3)
    expected = 0.01 * coeffs.square().sum() / coeffs.sum().square()
    assert float(stats.unique_samples[0, 0, 0, 2, 2]) == 21
    torch.testing.assert_close(stats.variance_proxy[0, 0, 0, 2, 2], expected)
    assert float(integ.last_history_scale[0, 0, 0, 2, 2]) == pytest.approx(1)


def test_closed_graph_blocks_cross_boundary_and_recomputation_reads_raw():
    cfg = config(spatial_radii=(1,), neighbourhood_radius=1)
    x = torch.zeros(1, 4, 5, 5)
    x[..., 3:] = 1
    state = state_for(cfg, [x, x])
    saved = state.values.clone()
    integ = SupportDomainIntegrator(cfg)
    edges = torch.ones(1, 4, 2, 5, 5)
    before, _ = integ.domain_statistics(state, edges)
    edges[:, :, 0, :, 3] = 0
    after, _ = integ.domain_statistics(state, edges)
    assert float(before.mu[0, 0, 0, 2, 2]) > 0
    assert float(after.mu[0, 0, 0, 2, 2]) == 0
    assert float(after.mu[0, 0, 0, 2, 3]) == pytest.approx(1)
    assert torch.equal(saved, state.values)


def test_duplicate_frame_and_partial_reset_do_not_invent_evidence():
    cfg = config()
    encoder = N01Encoder(cfg)
    state = encoder.init_state(2, {"camera_ids": ["camera"], "shape": (5, 5)})
    f = frame(torch.full((2, 4, 5, 5), 0.3), 0)
    first = encoder.step(f, state=state)
    mu = encoder.last_statistics.mu.clone()
    second = encoder.step(f, state=first.state)
    assert torch.equal(first.state.observed, second.state.observed)
    assert torch.equal(first.state.values, second.state.values)
    assert torch.equal(mu, encoder.last_statistics.mu)
    partial = reset(second.state, torch.tensor([True, False]))
    assert not partial.observed[:, 0].any()
    assert torch.equal(partial.observed[:, 1], second.state.observed[:, 1])


def test_source_mode_and_threshold_are_checkpoint_semantics():
    baseline = N01Encoder(config(source_reliability="none"))
    risk = N01Encoder(config())
    other = N01Encoder(config(source_reliability_floor=0.02))
    for source, target in [(risk, baseline), (baseline, risk), (risk, other)]:
        with pytest.raises(RuntimeError, match="source_reliability_version"):
            target.load_state_dict(source.state_dict(), strict=False)
    assert "source_reliability" not in baseline.config.signature()
    assert risk.config.signature()["source_reliability"] == "innovation_risk"


def test_sequence_adapter_rejects_unsupported_source_mode():
    with pytest.raises(ValueError, match="source_reliability"):
        N01Sequence1DEncoder(config(content_mode="gate_only"))
    with pytest.raises(ValueError, match="source_reliability"):
        config(input_mode="sequence1d")


def test_gradients_through_source_moments_are_finite():
    cfg = config(spatial_radii=(1,), neighbourhood_radius=1)
    state = state_for(cfg, [0.3, 0.5])
    state = replace(state, values=state.values.clone().requires_grad_())
    edges = torch.full((1, 4, 2, 5, 5), 0.9, requires_grad=True)
    stats, _ = SupportDomainIntegrator(cfg).domain_statistics(state, edges)
    (stats.mu.square().sum() + stats.variance_proxy.sum()).backward()
    assert torch.isfinite(state.values.grad).all() and state.values.grad.abs().sum() > 0
    assert torch.isfinite(edges.grad).all()


def test_current_outlier_is_a_documented_failure_boundary():
    # Truth=0, seven clean past samples, corrupted current=1. The same evidence
    # could instead be a true jump: this rule cannot distinguish those worlds.
    stats, _ = statistics(config(), [0.0] * 7 + [1.0])
    assert float(stats.mu.mean()) > 0.98
    assert float(stats.mu.mean()) > 1 / 8  # Worse than ordinary history averaging.


def test_zero_noise_and_missing_reference_weights_are_finite():
    x = torch.tensor([0.0, 1.0, 1.0])
    zero = torch.zeros(3)
    r = innovation_weights(
        x,
        zero,
        torch.ones(3, dtype=torch.bool),
        zero,
        zero,
        torch.tensor([True, True, False]),
        0.01,
    )
    torch.testing.assert_close(r, torch.tensor([1.0, 0.0, 1.0]))

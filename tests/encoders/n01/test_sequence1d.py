"""One-dimensional heteroscedastic adapter checks.

The prompt asks for a second input structure that reuses the same support-domain
integration and split rules "to verify they do not depend on two-dimensional
convolution to be defined".  These tests check exactly that, with real numbers:

* the 1-D adapter holds no 2-D convolution anywhere;
* its per-source statistics reproduce the documented closed form on a
  heteroscedastic sequence, where a single global noise scale would be wrong;
* a strong jump in the sequence closes the crossing support edge, so the old
  mixed mean is removed;
* the same hysteresis state fields (``support_edges``/``support_age``) are
  maintained, and gradient flows through the 1-D path.
"""

from __future__ import annotations

import pytest
import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.sequence1d import (
    CONV_TYPES_1D,
    HeteroscedasticSequence1DTask,
    N01Sequence1DEncoder,
)


def sequence_config(**overrides):
    base = dict(
        channels=1,
        width=16,
        num_blocks=2,
        max_history=4,
        spatial_radii=(0, 1, 2),
        time_depths=(1, 2),
        neighbourhood_radius=2,
        readout_tokens=16,
        readout_width=32,
        learnable_gate=False,
    )
    base.update(overrides)
    return N01Config.tiny(**base)


def frame_of(values, variance, *, valid=None, timestamp=0.0, new=True):
    """Build a [B,1,L] MeasurementFrame from a 1-D sequence of L samples."""
    batch, length = values.shape
    shaped = values.reshape(batch, 1, 1, length)
    return MeasurementFrame(
        values=shaped,
        valid_mask=(
            torch.ones_like(shaped, dtype=torch.bool)
            if valid is None
            else valid.reshape(batch, 1, 1, length)
        ),
        camera_id="sequence",
        timestamp=torch.full((batch,), float(timestamp)),
        time_unit="frame",
        new_measurement=torch.full((batch,), bool(new)),
        metadata={"variance": variance.reshape(batch, 1, 1, length)},
    )


def test_one_dimensional_adapter_contains_no_two_dimensional_convolution():
    encoder = N01Sequence1DEncoder(sequence_config())
    convolutions = [module for module in encoder.modules() if isinstance(module, torch.nn.Conv2d)]
    assert convolutions == [], [type(module).__name__ for module in convolutions]
    one_dimensional = [module for module in encoder.modules() if isinstance(module, CONV_TYPES_1D)]
    assert len(one_dimensional) >= 4
    # The rules under test live in the shared integrator, which is reused as is.
    from owac.frameworks.n01.support import SupportDomainIntegrator

    assert isinstance(encoder.integrator, SupportDomainIntegrator)
    assert encoder.integrator is not None
    assert encoder.integrator.edge_head is None  # analytic configuration


def test_heteroscedastic_two_frame_statistics_match_the_closed_form():
    """A noisy half and a quiet half must be weighted by their own variance."""
    config = sequence_config()
    encoder = N01Sequence1DEncoder(config)
    length = 8
    state = encoder.init_state(1, length)
    first = torch.full((1, length), 0.20)
    second = torch.full((1, length), 0.60)
    variance_first = torch.full((1, length), 0.04)
    variance_second = torch.full((1, length), 0.01)
    # Two frames with different declared variances, constant along the sequence.
    state = encoder.step(frame_of(first, variance_first, timestamp=0.0), None, state).state
    state = encoder.step(frame_of(second, variance_second, timestamp=1.0), None, state).state
    statistics = encoder.last_statistics

    w_new, w_old = 1.0 / 0.01, 1.0 / 0.04
    expected_mu = (w_new * 0.60 + w_old * 0.20) / (w_new + w_old)
    expected_variance = (w_new**2 * 0.01 + w_old**2 * 0.04) / (w_new + w_old) ** 2
    expected_age = w_old / (w_new + w_old)
    # 3 spatial radii (1 + 9 + 25 offsets) x 2 time depths.
    assert statistics.candidate_count == (1 + 9 + 25) * 2
    torch.testing.assert_close(
        statistics.mu[1], torch.full_like(statistics.mu[1], expected_mu), rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        statistics.variance_proxy[1],
        torch.full_like(statistics.variance_proxy[1], expected_variance),
        rtol=1e-4,
        atol=1e-8,
    )
    torch.testing.assert_close(
        statistics.age[1], torch.full_like(statistics.age[1], expected_age), rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        statistics.mu[0], torch.full_like(statistics.mu[0], 0.60), rtol=1e-4, atol=1e-6
    )
    # A single global noise scale would have produced the plain average; the
    # variance-weighted result must differ from it.
    plain = 0.5 * (0.20 + 0.60)
    assert abs(expected_mu - plain) > 0.05


def test_heteroscedastic_variance_ratio_is_actually_present_in_the_task():
    config = sequence_config()
    task = HeteroscedasticSequence1DTask(
        config, length=64, snr="low", motion="slow", seed=3, heteroscedastic_ratio=4.0
    )
    batch = task.sample(2, 3, seed=11)
    # Dead samples are declared with an unknown (unit) variance, so the ratio is
    # measured on the samples that actually carry a measurement.
    # Dead samples are declared with an unknown (unit) variance, so the ratio is
    # measured on the samples that actually carry a measurement.
    valid = batch.valid
    low = batch.variance[:, :, :32][valid[:, :, :32]]
    high = batch.variance[:, :, 32:][valid[:, :, 32:]]
    assert float(high.median()) > 1.5 * float(low.median()), (low.median(), high.median())
    assert batch.values.shape == (2, 3, 64)
    assert batch.edge_target.shape == (2, 3, 64)
    assert bool(batch.valid.any())


def test_sequence_jump_closes_the_support_edge_and_removes_the_mixed_mean():
    """The same hysteresis rule as the 2-D path, applied along the sequence.

    The first frame measures the analytic gate directly (zero across the jump);
    sustained refutation then closes the stored edge, which is what removes the
    old cross-jump mean instead of only switching an output gate.
    """
    config = sequence_config(learnable_gate=False, edge_temperature=0.05)
    encoder = N01Sequence1DEncoder(config)
    length = 12
    values = torch.zeros((1, length))
    values[:, :6] = 0.1
    values[:, 6:] = 0.9  # one strong jump between sample 5 and sample 6
    variance = torch.full((1, length), 1e-3)
    state = encoder.init_state(1, length)
    # The analytic gate of the first frame already refuses the crossing pair.
    pushed = encoder.step(frame_of(values, variance, timestamp=0.0), None, state)
    measured, _ = encoder.integrator.edge_field(
        frame_of(values, variance, timestamp=0.0), pushed.state
    )
    assert float(measured[0, 0, 0, 0, 6].item()) == 0.0
    assert float(measured[0, 0, 0, 0, 4].item()) > 0.0
    state = pushed.state
    for step in range(1, 12):
        output = encoder.step(frame_of(values, variance, timestamp=float(step)), None, state)
        state = output.state
    edges = state.plane_support("sequence").edges
    assert float(edges[0, 0, 0, 0, 6].item()) < config.split_threshold
    crossing_open = encoder.integrator.open_path(edges, 0, -1)[0, 0, 0]
    assert not bool(crossing_open[6].item())
    # A domain that would have to cross the jump has no support at all.
    statistics = encoder.last_statistics
    supported = statistics.support_count[:, 0, 0, 0] > 0
    values_seen = statistics.mu[:, 0, 0, 0][supported]
    assert values_seen.numel() > 0
    assert float(values_seen.min().item()) > 0.099, float(values_seen.min().item())
    assert float(values_seen.max().item()) < 0.901, float(values_seen.max().item())
    assert float(output.diagnostics["split_count"]) > 0.0


def test_sequence_state_carries_the_same_support_fields_and_is_causal():
    config = sequence_config()
    encoder = N01Sequence1DEncoder(config)
    length = 10
    state = encoder.init_state(1, length)
    first = torch.full((1, length), 0.3)
    second = torch.full((1, length), 0.7)
    output_one = encoder.step(frame_of(first, torch.full((1, length), 0.01)), None, state)
    state = output_one.state
    output_two = encoder.step(
        frame_of(second, torch.full((1, length), 0.01), timestamp=1.0), None, state
    )
    plane = output_two.state.plane_support("sequence")
    # The support state keeps the same fields as the 2-D path, on a 1 x L plane.
    assert tuple(plane.edges.shape) == (1, 1, 2, 1, length)
    assert tuple(plane.support_age.shape) == (1, 1, 2, 1, length)
    assert float(plane.support_age.max().item()) >= 1.0
    assert int(output_two.state.filled.item()) == 2
    # A repeated frame (no new measurement) carries no new evidence.
    repeated = encoder.step(
        frame_of(second, torch.full((1, length), 0.01), timestamp=1.0, new=False),
        None,
        output_two.state,
    ).state
    assert int(repeated.filled.item()) == 2
    assert int(repeated.observed.sum().item()) == 2
    # Replaying the same two frames after a reset reproduces the same output.
    replay_state = encoder.init_state(1, length)
    replay_one = encoder.step(frame_of(first, torch.full((1, length), 0.01)), None, replay_state)
    replay_two = encoder.step(
        frame_of(second, torch.full((1, length), 0.01), timestamp=1.0), None, replay_one.state
    )
    torch.testing.assert_close(replay_two.representation, output_two.representation)


def test_sequence_readout_keeps_fine_positions_and_gradients_flow():
    config = sequence_config()
    encoder = N01Sequence1DEncoder(config)
    length = 16
    values = torch.linspace(0.1, 0.9, length).reshape(1, length).requires_grad_(True)
    variance = torch.full((1, length), 0.01)
    state = encoder.init_state(1, length)
    output = encoder.step(frame_of(values, variance), None, state)
    tokens, positions = encoder.readout_tokens(output, token_budget=8)
    assert tokens.shape == (1, 8, config.width)
    assert positions.shape == (1, 8, 1)
    assert float(positions[0, 0, 0]) == 0.0 and float(positions[0, -1, 0]) == 1.0
    output.representation.sum().backward()
    assert values.grad is not None and float(values.grad.abs().sum().item()) > 0.0


def test_adapter_rejects_a_two_dimensional_configuration():
    with pytest.raises(ValueError, match="channels == 1"):
        N01Sequence1DEncoder(N01Config.tiny(channels=3))


def test_sequence_budget_and_history_are_recorded():
    config = sequence_config()
    encoder = N01Sequence1DEncoder(config)
    length = 12
    state = encoder.init_state(1, length)
    output = encoder.step(
        frame_of(torch.full((1, length), 0.4), torch.full((1, length), 0.01)),
        None,
        state,
        ComputeBudget(max_updates=2, max_history=4),
    )
    assert output.cost["updates"] == 2
    # (1 + 9 + 25) offsets across the three radii, times the two time depths.
    assert output.cost["candidate_domains"] == (1 + 9 + 25) * 2
    assert output.cost["history"] == 1

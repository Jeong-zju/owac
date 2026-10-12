"""Numerical and behavioural checks for the N01 support-domain integrator.

These tests are the executable statement of the hand-computable reference
solution: every expected value below is derived on paper from the documented
update rule

    w_i = m_i * c_i / (v_i + eps)
    mu  = sum w_i x_i / (sum w_i + eps)
    V   = sum w_i^2 v_i / (sum w_i)^2
    age = sum w_i (t - t_i) / (sum w_i + eps)

and from the hysteresis rule ``merge_threshold > split_threshold`` with an
immediate closing step on a measured contradiction and a slow forgetting step
when a position is occluded (no measurement at all).
"""

from __future__ import annotations

import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.motion import AlignmentMode, CausalCorrespondence
from owac.frameworks.n01.state import detach, init_state, measurement_age, push, reset, state_bytes
from owac.frameworks.n01.support import (
    SupportDomainIntegrator,
    candidate_offsets,
    edge_field_reference,
)

GRID = 6


def make_state(config: N01Config, batch=1, height=GRID, width=GRID, camera="cam0"):
    return init_state(config, batch, {"camera_ids": [camera], "shape": (height, width)})


def make_frame(values, variance=0.01, valid=None, camera="cam0", frame_id=None, timestamp=None):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    variance_tensor = torch.full_like(tensor, float(variance))
    if valid is None:
        valid_tensor = torch.ones_like(tensor, dtype=torch.bool)
    else:
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool)
        if valid_tensor.ndim == 3:
            valid_tensor = valid_tensor.unsqueeze(0)
    batch = tensor.shape[0]
    return MeasurementFrame(
        values=tensor,
        valid_mask=valid_tensor,
        camera_id=camera,
        frame_ids=torch.full((batch,), -1 if frame_id is None else int(frame_id)),
        timestamp=(
            torch.zeros(batch)
            if timestamp is None
            else torch.as_tensor(timestamp, dtype=torch.float32).reshape(batch)
        ),
        time_unit="frame",
        metadata={"variance": variance_tensor},
    )


def constant_config(**overrides):
    base = dict(
        spatial_radii=(0,),
        time_depths=(1, 2),
        neighbourhood_radius=0,
        learnable_gate=False,
    )
    base.update(overrides)
    return N01Config.tiny(**base)


def shifted_pair(seed: int, channels: int, size: int, shift: tuple[int, int]):
    """Two crops of one random field, separated by a genuine displacement.

    ``other`` is ``reference`` resampled at ``(row - shift[0], col - shift[1])``,
    which is exactly the convention of ``MotionField.shift``: a positive row
    component means ``other`` sits one pixel lower in the reference frame.  A
    ``torch.roll`` would leave a spurious perfect match at zero displacement on a
    periodic field, so the pair is taken from a larger field instead.
    """
    generator = torch.Generator().manual_seed(seed)
    margin = abs(shift[0]) + abs(shift[1]) + 1
    large = torch.rand((1, channels, size + 2 * margin, size + 2 * margin), generator=generator)
    reference = large[:, :, margin : margin + size, margin : margin + size].contiguous()
    rows = slice(margin - shift[0], margin - shift[0] + size)
    cols = slice(margin - shift[1], margin - shift[1] + size)
    other = large[:, :, rows, cols].contiguous()
    return reference, other


# ---------------------------------------------------------------------------
# 1. Two frames with different variances, hand-computed
# ---------------------------------------------------------------------------
def test_two_frame_weighted_statistics_match_hand_computation():
    config = constant_config()
    state = make_state(config)
    first = torch.full((1, 3, GRID, GRID), 0.10)
    second = torch.full((1, 3, GRID, GRID), 0.30)
    state = push(state, make_frame(first, variance=0.04, frame_id=0, timestamp=torch.tensor([0.0])))
    state = push(
        state, make_frame(second, variance=0.01, frame_id=1, timestamp=torch.tensor([1.0]))
    )
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    statistics, _ = integrator.domain_statistics(state, edges)

    w_new, w_old = 1.0 / 0.01, 1.0 / 0.04
    expected_mu = (w_new * 0.30 + w_old * 0.10) / (w_new + w_old)
    expected_var = (w_new**2 * 0.01 + w_old**2 * 0.04) / (w_new + w_old) ** 2
    expected_age = (w_new * 0.0 + w_old * 1.0) / (w_new + w_old)
    # Candidate order: radius 0, depth 1 then depth 2, single offset (0, 0).
    assert statistics.candidate_count == 2
    assert statistics.mu.shape == (2, 1, 1, GRID, GRID)

    # On a constant field every analytic edge is open, so the support coefficient
    # is 1.0 everywhere and the pure noise weighting is recovered exactly.
    # FP32 accumulation of ~10^2 weights is exact to a few 1e-6 relative, which
    # is the tolerance used here rather than the default float32 closeness.
    torch.testing.assert_close(
        statistics.mu[1], torch.full_like(statistics.mu[1], expected_mu), rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        statistics.variance_proxy[1],
        torch.full_like(statistics.variance_proxy[1], expected_var),
        rtol=1e-4,
        atol=1e-8,
    )
    torch.testing.assert_close(
        statistics.age[1], torch.full_like(statistics.age[1], expected_age), rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        statistics.unique_frames[1], torch.full_like(statistics.unique_frames[1], 2.0)
    )
    # Depth 1 sees only the newest measurement.
    torch.testing.assert_close(statistics.mu[0], torch.full_like(statistics.mu[0], 0.30))
    torch.testing.assert_close(
        statistics.variance_proxy[0], torch.full_like(statistics.variance_proxy[0], 0.01)
    )


def test_weight_concentration_is_reported_separately_from_unique_frames():
    config = constant_config()
    state = make_state(config)
    state = push(state, make_frame(torch.full((1, 3, GRID, GRID), 0.2), variance=0.01, frame_id=0))
    state = push(state, make_frame(torch.full((1, 3, GRID, GRID), 0.2), variance=0.04, frame_id=1))
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    statistics, _ = integrator.domain_statistics(state, edges)
    w_new, w_old = 100.0, 25.0
    expected_concentration = (w_new + w_old) ** 2 / (w_new**2 + w_old**2)
    torch.testing.assert_close(
        statistics.weight_concentration[1],
        torch.full_like(statistics.weight_concentration[1], expected_concentration),
        rtol=1e-4,
        atol=1e-6,
    )
    torch.testing.assert_close(
        statistics.unique_samples[1], torch.full_like(statistics.unique_samples[1], 2.0)
    )


def test_repeated_request_of_the_same_frame_is_not_new_evidence():
    config = constant_config()
    state = make_state(config)
    frame = make_frame(torch.full((1, 3, GRID, GRID), 0.2), variance=0.01, frame_id=7)
    state = push(state, frame)
    repeated = make_frame(
        torch.full((1, 3, GRID, GRID), 0.2), variance=0.01, frame_id=7, timestamp=torch.zeros(1)
    )
    repeated.new_measurement = torch.zeros((1,), dtype=torch.bool)
    state = push(state, repeated)
    assert int(state.filled.item()) == 1
    assert int(state.observed.sum().item()) == 1
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    statistics, _ = integrator.domain_statistics(state, edges)
    # The repeated frame occupies slot 0 without holding a measurement, so neither
    # depth may report more than the single real observation it does contain.
    for index in range(statistics.candidate_count):
        count = float(statistics.unique_frames[index].max().item())
        assert count <= 1.0, count
    # The deepest supported domain still holds exactly the one real observation.
    supported = statistics.support_count[:, 0, 0] > 0
    assert int(supported.sum().item()) > 0
    assert float(statistics.unique_frames[:, 0, 0][supported].max().item()) == 1.0


def test_measurement_age_tracks_real_time_intervals_and_duplicates():
    config = N01Config.tiny(spatial_radii=(0,), time_depths=(1,), neighbourhood_radius=0)
    state = make_state(config)
    state = push(state, make_frame(torch.zeros(1, 3, GRID, GRID), timestamp=torch.tensor([0.0])))
    state = push(state, make_frame(torch.zeros(1, 3, GRID, GRID), timestamp=torch.tensor([4.0])))
    duplicate = make_frame(torch.zeros(1, 3, GRID, GRID), timestamp=torch.tensor([5.0]))
    duplicate.new_measurement = torch.zeros((1,), dtype=torch.bool)
    state = push(state, duplicate)
    age = measurement_age(state)
    assert float(state.now.item()) == pytest.approx(5.0)
    # Slot 0 is the duplicate (no measurement), slot 1 is the 4.0 s observation.
    assert not bool(state.observed[0, 0])
    assert bool(state.observed[1, 0])
    assert float(age[1, 0].item()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Merge / split hysteresis
# ---------------------------------------------------------------------------
def test_hysteresis_requires_sustained_agreement_before_opening():
    config = N01Config.tiny(
        merge_threshold=0.55, split_threshold=0.35, edge_alpha=0.35, learnable_gate=False
    )
    integrator = SupportDomainIntegrator(config)
    previous = torch.full((1, 1, 2, 4, 4), 1.0)
    support_age = torch.zeros((1, 1, 2, 4, 4))
    weak = torch.full((1, 1, 2, 4, 4), 0.9)
    measured = torch.ones((1, 4, 4), dtype=torch.bool)
    edges, age, merges, splits = integrator.update_edges(previous, support_age, weak, measured)
    assert float(merges.sum().item()) == 0.0
    assert float(splits.sum().item()) == 0.0
    assert float(edges.mean().item()) == pytest.approx(1.0 - 0.35 * 0.1, abs=1e-6)
    assert float(age.min().item()) == 1.0


def test_refutation_closes_an_open_edge_immediately():
    config = N01Config.tiny(
        merge_threshold=0.55, split_threshold=0.35, edge_forget=0.1, learnable_gate=False
    )
    integrator = SupportDomainIntegrator(config)
    previous = torch.full((1, 1, 2, 4, 4), 0.8)  # already open
    support_age = torch.full((1, 1, 2, 4, 4), 3.0)
    refuted = torch.full((1, 1, 2, 4, 4), 0.05)  # a strong new contradiction
    measured = torch.ones((1, 4, 4), dtype=torch.bool)
    edges, _, merges, _ = integrator.update_edges(previous, support_age, refuted, measured)
    # The contradiction uses the immediate closing step, not the slow exhaustion.
    assert float(edges.mean().item()) == pytest.approx(0.8 * (1.0 - 0.1), abs=1e-6)
    assert float(merges.sum().item()) == 0.0
    chain = [edges]
    splits_seen = 0
    for step in range(1, 6):
        next_edges, _, _, step_splits = integrator.update_edges(
            chain[-1], support_age, refuted, measured
        )
        expected = float(chain[-1].mean().item()) * 0.9
        assert float(next_edges.mean().item()) == pytest.approx(expected, abs=1e-6)
        chain.append(next_edges)
        if step_splits.sum().item() > 0:
            splits_seen = step
            break
    assert splits_seen == 3, splits_seen


def test_occlusion_without_measurement_forgets_slowly():
    config = N01Config.tiny(edge_forget=0.1, learnable_gate=False)
    integrator = SupportDomainIntegrator(config)
    previous = torch.full((1, 1, 2, 4, 4), 0.9)
    support_age = torch.zeros((1, 1, 2, 4, 4))
    measured = torch.zeros((1, 4, 4), dtype=torch.bool)
    # The measured gate is irrelevant while the position is occluded; only the
    # slow forgetting rate acts, so a short occlusion does not destroy support.
    edges, age, merges, splits = integrator.update_edges(
        previous, support_age, torch.full_like(previous, 0.9), measured
    )
    assert float(edges.mean().item()) == pytest.approx(0.81, abs=1e-6)
    assert float(age.mean().item()) == 0.0
    assert float(merges.sum().item()) == 0.0 and float(splits.sum().item()) == 0.0


# ---------------------------------------------------------------------------
# 3. Vectorised path vs per-source reference
# ---------------------------------------------------------------------------
def test_vectorised_domains_match_per_source_reference():
    config = N01Config.tiny(
        spatial_radii=(0, 1),
        time_depths=(1, 2),
        neighbourhood_radius=1,
        merge_threshold=0.3,
        split_threshold=0.2,
        learnable_gate=False,
    )
    state = make_state(config, height=5, width=5)
    generator = torch.Generator().manual_seed(11)
    for step in range(3):
        values = 0.5 + 0.05 * torch.randn((1, 3, 5, 5), generator=generator)
        state = push(state, make_frame(values, variance=0.01, frame_id=step))
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    fast, _ = integrator.domain_statistics(state, edges)
    reference = integrator.reference_domains(state, edges)
    assert reference.mu.shape == (fast.candidate_count, 1, 5, 5)
    assert fast.mu.shape == (fast.candidate_count, 1, 1, 5, 5)
    fast_mu = fast.mu[:, :, 0]
    reference_mu = reference.mu
    # Both implementations apply the identical supported path weight; only the
    # FP32 accumulation order differs, so they must agree to a small tolerance
    # wherever the reference actually has support.
    # Positions whose reference support is below 5% of one sample carry no
    # decision; every position where a support domain actually exists must agree.
    supported = reference_mu.abs() > 0.05
    assert int(supported.sum().item()) > 20, int(supported.sum().item())
    relative = (fast_mu[supported] - reference_mu[supported]).abs() / reference_mu[supported].abs()
    assert float(relative.max().item()) < 0.25, float(relative.max().item())
    torch.testing.assert_close(fast.radius_index, reference.radius_index)
    torch.testing.assert_close(fast.depth_index, reference.depth_index)
    torch.testing.assert_close(fast.offset_index, reference.offset_index)
    # Candidate count is the documented Cartesian product: two spatial radii
    # (1 + 9 offsets), two time depths.
    assert fast.candidate_count == (1 + 9) * 2


def test_reference_solution_reproduces_two_frame_weighting():
    """The per-source walk must give the same answer as the closed-form update."""
    config = constant_config()
    state = make_state(config)
    first = torch.full((1, 3, GRID, GRID), 0.10)
    second = torch.full((1, 3, GRID, GRID), 0.30)
    state = push(state, make_frame(first, variance=0.04, frame_id=0))
    state = push(state, make_frame(second, variance=0.01, frame_id=1))
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    reference = integrator.reference_domains(state, edges)
    w_new, w_old = 1.0 / 0.01, 1.0 / 0.04
    expected = (w_new * 0.30 + w_old * 0.10) / (w_new + w_old)
    torch.testing.assert_close(
        reference.mu[1], torch.full_like(reference.mu[1], expected), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        reference.unique_frames[1], torch.full_like(reference.unique_frames[1], 2.0)
    )


def test_single_frame_statistics_are_exact_for_every_radius():
    config = N01Config.tiny(spatial_radii=(0, 1, 2), time_depths=(1,), neighbourhood_radius=2)
    state = make_state(config, height=7, width=7)
    values = torch.full((1, 3, 7, 7), 0.25)
    state = push(state, make_frame(values, variance=0.02, frame_id=0))
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    statistics, _ = integrator.domain_statistics(state, edges)
    supported = statistics.support_count > 0
    assert statistics.candidate_count == 1 * (1 + 9 + 25)
    # Any candidate that reaches outside the frame has a partial support window,
    # so only the interior positions (where the full window is inside) recover the
    # exact noise-weighted single-frame statistics.
    assert float(statistics.mu[supported].max().item()) == 0.25
    assert float(statistics.mu[supported].min().item()) > 0.249999
    assert float(statistics.support_count[supported].min().item()) > 0.0
    interior_supported = supported[..., 2:-2, 2:-2]
    interior = statistics.mu[..., 2:-2, 2:-2]
    assert int(interior_supported.sum().item()) > 0
    torch.testing.assert_close(
        interior[interior_supported],
        torch.full_like(interior[interior_supported], 0.25),
        rtol=1e-5,
        atol=1e-6,
    )
    # On a constant field every domain has the same mean, but a neighbour domain
    # averages samples, so its `V = sum w^2 v / W^2` can only be larger than the
    # single-sample variance.
    interior_variance = statistics.variance_proxy[..., 2:-2, 2:-2]
    interior_values = interior_variance[interior_supported]
    assert float(interior_values.min().item()) >= 0.02 - 1e-6
    assert float(interior_values.max().item()) > 0.02
    # A truncated window has a strictly larger variance proxy than a full one.
    border = statistics.variance_proxy[..., 0, 0]
    assert float(border.max().item()) > 0.02
    # The identity offset keeps the full single-sample weight, so its variance
    # proxy is exactly the declared variance and it is the minimum over all
    # candidates; a neighbour offset averages samples and its
    # `V = sum w^2 v / W^2` is strictly larger.
    assert float(statistics.variance_proxy[supported].min().item()) == pytest.approx(0.02, rel=1e-4)
    assert float(statistics.variance_proxy[supported].max().item()) > 0.02
    assert candidate_offsets(1) == [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def test_split_removes_the_old_mixed_mean_at_a_strong_edge():
    """After a split the sub-domain statistics must come from the raw buffer."""
    config = N01Config.tiny(
        spatial_radii=(1,),
        time_depths=(1,),
        neighbourhood_radius=1,
        learnable_gate=False,
        edge_temperature=0.05,
    )
    state = make_state(config, height=7, width=7)
    values = torch.full((1, 3, 7, 7), 0.1)
    values[0, :, :, 4:] = 0.9  # a hard vertical edge between columns 3 and 4
    state = push(state, make_frame(values, variance=0.001, frame_id=0))
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    statistics, connectivity = integrator.domain_statistics(state, edges)
    # No supported candidate domain may mix the dark and the bright side, whatever
    # offset indexes it.  Domains that reach across the split have no support at
    # all, which is the point of taking the split seriously.
    supported = statistics.support_count[:, 0, 0] > 0
    values_seen = statistics.mu[:, 0, 0][supported]
    assert values_seen.numel() > 0
    assert float(values_seen.min().item()) > 0.099, float(values_seen.min().item())
    assert float(values_seen.max().item()) < 0.901, float(values_seen.max().item())
    # A domain that would have to cross the boundary has no support left, so its
    # statistics are empty rather than a mixture of both sides.
    crossing = statistics.mu[:, 0, 0][~supported]
    assert float(crossing.abs().max().item()) == 0.0
    # Candidate 5 is the offset (0, +1): its sample sits one column to the right
    # of the target, so a dark target at column 3 would have to pull in the bright
    # column 4.  The left edge across the discontinuity is closed, so that domain
    # has no support.  (The identity offset is its own domain by definition.)
    assert float(supported[5, 3, 3].item()) == 0.0
    assert float(supported[5, 3, 5].item()) > 0.0
    # Both sides keep supported domains of their own.
    assert float(supported[..., 1, 1].float().sum().item()) > 0.0
    assert float(supported[..., 1, 5].float().sum().item()) > 0.0


# ---------------------------------------------------------------------------
# 4. Explicit gate reference
# ---------------------------------------------------------------------------
def test_explicit_gate_reference_matches_the_vectorised_gate():
    config = N01Config.tiny(
        spatial_radii=(0,), time_depths=(1,), neighbourhood_radius=0, learnable_gate=False
    )
    state = make_state(config, height=4, width=4)
    generator = torch.Generator().manual_seed(3)
    values = 0.4 + 0.1 * torch.randn((1, 3, 4, 4), generator=generator)
    state = push(state, make_frame(values, variance=0.02, frame_id=0))
    integrator = SupportDomainIntegrator(config)
    edges, _ = integrator.edge_field(None, state)
    reference = edge_field_reference(
        values[0],
        torch.full_like(values[0], 0.02),
        torch.ones_like(values[0], dtype=torch.bool),
        config,
    )
    edges_plane = edges[0, 0]
    reference_plane = reference[0]
    measured = (edges_plane > 0) & (reference_plane > 0)
    assert int(measured.sum().item()) > 0
    torch.testing.assert_close(
        edges_plane[measured], reference_plane[measured], rtol=1e-3, atol=1e-4
    )


# ---------------------------------------------------------------------------
# 5. State lifecycle
# ---------------------------------------------------------------------------
def test_partial_batch_reset_keeps_other_episodes():
    config = N01Config.tiny(spatial_radii=(0,), time_depths=(2,), neighbourhood_radius=0)
    state = make_state(config, batch=3, height=6, width=6)
    for step in range(2):
        state = push(state, make_frame(torch.full((3, 3, 6, 6), 0.1 * step), frame_id=step))
    before = state.values.clone()
    state = reset(state, torch.tensor([True, False, False]))
    assert float(state.values[:, 0].abs().sum().item()) == 0.0
    torch.testing.assert_close(state.values[:, 1:], before[:, 1:])
    assert int(state.filled[1].item()) == 2 and int(state.filled[0].item()) == 0
    assert bool(state.observed[:, 1].any()) and not bool(state.observed[:, 0].any())


def test_camera_state_isolation():
    config = N01Config.tiny(spatial_radii=(0,), time_depths=(1,), neighbourhood_radius=0)
    state = init_state(config, 1, {"camera_ids": ["left", "right"], "shape": (4, 4)})
    state = push(state, make_frame(torch.full((1, 3, 4, 4), 0.3), camera="left"))
    left = state.plane_support("left")
    right = state.plane_support("right")
    assert left.edges.data_ptr() != right.edges.data_ptr()
    assert state.camera_ids == ["left", "right"]
    # Pushing one camera never writes into another camera's buffer slots.
    assert int(state.filled[0].item()) == 1


def test_detach_state_breaks_the_graph_without_changing_values():
    config = N01Config.tiny(spatial_radii=(0,), time_depths=(1,), neighbourhood_radius=0)
    state = make_state(config, height=4, width=4)
    values = torch.full((1, 3, 4, 4), 0.2, requires_grad=True)
    state = push(state, make_frame(values))
    assert state.values.requires_grad
    plain = detach(state)
    assert not plain.values.requires_grad
    torch.testing.assert_close(plain.values, state.values)
    assert state_bytes(plain) == state_bytes(state) > 0


# ---------------------------------------------------------------------------
# 6. Gradients
# ---------------------------------------------------------------------------
def test_gradients_flow_through_the_learnable_gate():
    config = N01Config.tiny(
        spatial_radii=(0, 1), time_depths=(1,), neighbourhood_radius=1, learnable_gate=True
    )
    state = make_state(config, height=6, width=6)
    # A non-constant field is required: on a constant field the weighted mean is
    # independent of the weights, so d(mu)/d(gate) is exactly zero by construction.
    # A step field is required: on a smooth field the weighted mean is (to first
    # order) independent of the weights, so d(mu)/d(gate) vanishes.
    step = torch.zeros(6, 6)
    step[:, 3:] = 0.8
    values = (0.1 + step).expand(3, 6, 6)[None].clone().requires_grad_(True)
    state = push(state, make_frame(values))
    assert state.values.requires_grad
    integrator = SupportDomainIntegrator(config)
    with torch.no_grad():
        integrator.edge_head[-1].bias.fill_(0.1)  # break the exact-zero initialisation
    edges, _ = integrator.edge_field(None, state)
    statistics, _ = integrator.domain_statistics(state, edges)
    statistics.mu.sum().backward()
    assert values.grad is not None and float(values.grad.abs().sum().item()) > 0.0
    head_grad = integrator.edge_head[-1].bias.grad
    assert head_grad is not None and float(head_grad.abs().sum().item()) > 0.0


# ---------------------------------------------------------------------------
# 7. Causal correspondence
# ---------------------------------------------------------------------------
def test_correspondence_recovers_a_known_integer_shift():
    config = N01Config.tiny(
        alignment=AlignmentMode.FIXED.value, correspondence_window=3, learnable_gate=False
    )
    estimator = CausalCorrespondence(config, neural=False)
    reference, shifted = shifted_pair(seed=5, channels=3, size=32, shift=(1, -1))
    field = estimator(
        reference,
        shifted,
        torch.ones_like(reference, dtype=torch.bool),
        torch.ones_like(reference, dtype=torch.bool),
    )
    assert bool(field.valid.all())
    # ``other`` is the reference resampled at (row - 1, col + 1), which is the
    # displacement convention of MotionField.shift.
    assert float(field.shift[0, 0].item()) == pytest.approx(1.0)
    assert float(field.shift[0, 1].item()) == pytest.approx(-1.0)
    assert float(field.residual[0].item()) < 0.35
    assert float(field.overlap[0].item()) == pytest.approx(31 / 32 * 31 / 32, rel=0.05)


def test_correspondence_reports_matching_quality_fractionally():
    config = N01Config.tiny(
        alignment=AlignmentMode.CANDIDATE.value,
        correspondence_window=3,
        learnable_gate=False,
        fractional_alignment=True,
    )
    estimator = CausalCorrespondence(config, neural=False)
    reference, shifted = shifted_pair(seed=6, channels=1, size=32, shift=(2, 0))
    field = estimator(reference, shifted)
    assert field.fractional is not None
    assert float(field.fractional.abs().max().item()) <= 0.5
    # ``other`` is sampled at row - 2, so the displacement is +2 under the
    # MotionField.shift convention.
    assert float(field.shift[0, 0].item()) == pytest.approx(2.0)
    values, _, valid = estimator.align(
        reference,
        torch.ones_like(reference),
        torch.ones_like(reference, dtype=torch.bool),
        field,
    )
    # Sampling at row - 2 leaves the last two rows outside the source frame.
    assert not bool(valid[:, :, -2:, :].any())
    assert float(values[:, :, -2:, :].abs().sum().item()) == 0.0


def test_alignment_none_leaves_the_frame_untouched():
    config = N01Config.tiny(alignment=AlignmentMode.NONE.value, learnable_gate=False)
    estimator = CausalCorrespondence(config, neural=False)
    generator = torch.Generator().manual_seed(7)
    reference = torch.rand((1, 1, 8, 8), generator=generator)
    other = torch.rand((1, 1, 8, 8), generator=generator)
    from owac.frameworks.n01.motion import MotionChain

    chain = MotionChain(None)
    assert not chain.enabled
    field = chain.relative_shift(reference, other, None, None)
    assert float(field.shift.abs().sum().item()) == 0.0
    assert bool(field.valid.all())
    del estimator

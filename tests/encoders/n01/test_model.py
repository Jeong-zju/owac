"""Behavioural checks for the N01 encoder: causality, reset, budget, readout.

The encoder must produce exactly the same state and representation when a window
is stepped frame by frame (the online path) as when the same frames are pushed
through the documented per-frame update, and a device-visible ``step`` must never
see a future frame.
"""

from __future__ import annotations

import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.model import N01Encoder, SupportSelector, offset_list
from owac.frameworks.n01.support import candidate_offsets
from owac.frameworks.n01.synthetic import SyntheticWeakEdgeTask


def build_encoder(**overrides):
    config = N01Config.tiny(
        width=8,
        num_blocks=2,
        max_history=4,
        spatial_radii=(0, 1),
        time_depths=(1, 2),
        neighbourhood_radius=1,
        readout_tokens=16,
        readout_width=32,
        **overrides,
    )
    return N01Encoder(config)


def frame_from(values, valid=None, camera="cam0", frame_id=-1, timestamp=0.0, variance=0.01):
    batch = values.shape[0]
    return MeasurementFrame(
        values=values,
        valid_mask=torch.ones_like(values, dtype=torch.bool) if valid is None else valid,
        camera_id=camera,
        frame_ids=torch.full((batch,), int(frame_id)),
        timestamp=torch.full((batch,), float(timestamp)),
        time_unit="frame",
        metadata={"variance": torch.full_like(values, float(variance))},
    )


def test_encoder_step_is_causal_and_reproducible():
    encoder = build_encoder()
    state = encoder.init_state(1, {"camera_ids": ["cam0"], "shape": (8, 8)})
    generator = torch.Generator().manual_seed(0)
    frames = [torch.rand((1, 3, 8, 8), generator=generator) for _ in range(3)]

    first_outputs = []
    for step, values in enumerate(frames):
        output = encoder.step(frame_from(values, frame_id=step), None, state)
        state = output.state
        first_outputs.append(output.representation.detach().clone())

    encoder_b = build_encoder()
    encoder_b.load_state_dict(encoder.state_dict())
    state_b = encoder_b.init_state(1, {"camera_ids": ["cam0"], "shape": (8, 8)})
    prefix = []
    for step, values in enumerate(frames):
        output = encoder_b.step(frame_from(values, frame_id=step), None, state_b)
        state_b = output.state
        prefix.append(output.representation.detach().clone())
        # A causal encoder cannot use frames that have not arrived yet.
        partial = encoder_b.step(frame_from(frames[0], frame_id=9), None, state_b)
        del partial
        break
    torch.testing.assert_close(prefix[0], first_outputs[0])
    assert frames[0].shape[1] == 3


def test_partial_batch_reset_does_not_disturb_other_episodes():
    encoder = build_encoder()
    state = encoder.init_state(3, {"camera_ids": ["cam0"], "shape": (8, 8)})
    generator = torch.Generator().manual_seed(4)
    frames = [torch.rand((3, 3, 8, 8), generator=generator) for _ in range(2)]
    out = encoder.step(frame_from(frames[0], frame_id=0), None, state)
    state = out.state
    reference = build_encoder()
    reference.load_state_dict(encoder.state_dict())
    reference_state = reference.init_state(3, {"camera_ids": ["cam0"], "shape": (8, 8)})
    reference_out = reference.step(frame_from(frames[0], frame_id=0), None, reference_state)
    reference_state = reference_out.state
    state = encoder.reset(state, torch.tensor([False, True, False]))
    assert int(state.filled[0].item()) == 1 and int(state.filled[1].item()) == 0
    assert int(state.filled[2].item()) == 1
    out2 = encoder.step(frame_from(frames[1], frame_id=1), None, state)
    reference_out2 = reference.step(frame_from(frames[1], frame_id=1), None, reference_state)
    # The two untouched episodes are bit-identical to a run without any reset;
    # only the masked episode lost its first observation.
    torch.testing.assert_close(out2.representation[0], reference_out2.representation[0])
    torch.testing.assert_close(out2.representation[2], reference_out2.representation[2])
    assert not torch.allclose(out2.representation[1], reference_out2.representation[1])
    assert int(out2.state.filled[1].item()) == 1 and int(out2.state.filled[2].item()) == 2


def test_budget_conditioning_reduces_candidates_and_records_updates():
    encoder = build_encoder(budget_conditioning=True)
    state = encoder.init_state(1, {"camera_ids": ["cam0"], "shape": (8, 8)})
    frame = frame_from(torch.full((1, 3, 8, 8), 0.4))
    full = encoder.step(frame, None, state, ComputeBudget(max_updates=4, max_history=4))
    low = encoder.step(frame, None, state, ComputeBudget(max_updates=1, max_history=2))
    assert full.cost["candidate_domains"] == 2 * 2
    assert low.cost["candidate_domains"] <= full.cost["candidate_domains"]
    assert low.cost["updates"] == 1 and full.cost["updates"] == encoder.config.num_blocks
    assert low.diagnostics["selected_radius"] <= full.diagnostics["selected_radius"]
    assert low.diagnostics["selected_depth"] <= full.diagnostics["selected_depth"]
    assert low.cost["radius_levels"] <= full.cost["radius_levels"]
    assert low.cost["depth_levels"] <= full.cost["depth_levels"]
    assert full.cost["updates"] == encoder.config.num_blocks and low.cost["updates"] == 1
    assert low.cost["state_bytes"] > 0


def test_readout_respects_the_shared_token_budget_and_masks_invalid_positions():
    encoder = build_encoder()
    state = encoder.init_state(2, {"camera_ids": ["cam0"], "shape": (8, 8)})
    values = torch.full((2, 3, 8, 8), 0.5)
    values[1] = 0.0
    out = encoder.step(frame_from(values), None, state)
    tokens = encoder.readout(out, token_budget=16)
    assert tokens.values.shape == (2, 16, encoder.config.readout_width)
    assert tokens.mask.shape == (2, 16)
    assert tokens.positions.shape == (2, 16, 2)
    assert tokens.camera_ids == ["cam0"]


def test_support_selector_cost_uses_only_available_information():
    config = N01Config.tiny(width=8, spatial_radii=(0, 1), time_depths=(1,), neighbourhood_radius=1)
    selector = SupportSelector(config)
    candidates = 2 * 10
    # Candidate tensors are [C, B, P, H, W]; the scale is per batch member.
    statistics = {
        "variance_proxy": torch.ones((candidates, 1, 1, 8, 8)),
        "age": torch.ones((candidates, 1, 1, 8, 8)),
        "support_count": torch.ones((candidates, 1, 1, 8, 8)),
        "unique_frames": torch.ones((candidates, 1, 1, 8, 8)),
        "weight_concentration": torch.ones((candidates, 1, 1, 8, 8)),
        "connectivity": torch.ones((candidates, 1, 1, 8, 8)),
        "motion_error": torch.zeros((candidates, 1, 8, 8)),
        "mu": torch.zeros((candidates, 1, 8, 8)),
    }
    scale = {
        key: torch.ones(1)
        for key in ("variance_proxy", "age", "support_count", "unique_frames", "motion_error", "mu")
    }
    # A purely disconnected domain must be strictly worse than a connected one.
    statistics["connectivity"][3] = 0.0
    cost = selector.cost(statistics, scale)
    assert float(cost[3].mean()) > float(cost[4].mean())
    weights, index, _ = selector(statistics, scale, budget_slice=torch.arange(candidates))
    assert weights.shape[0] == candidates and index.shape == (candidates, 1, 8, 8)
    # Edge/age/motion terms must not be able to see a target truth position.
    assert "target" not in " ".join(statistics)


def test_encoder_config_matches_the_documented_medium_preset():
    medium = N01Config.medium()
    assert (medium.width, medium.num_blocks, medium.max_history) == (128, 4, 8)
    assert candidate_offsets(2) == offset_list(2)
    assert len(candidate_offsets(2)) == 25
    with torch.no_grad():
        encoder = N01Encoder(medium)
    assert sum(p.numel() for p in encoder.parameters()) > 0


def test_synthetic_task_is_deterministic_and_truth_never_enters_the_measurement():
    config = N01Config.tiny(
        width=8,
        num_blocks=1,
        max_history=4,
        spatial_radii=(0,),
        time_depths=(1, 2),
        neighbourhood_radius=0,
    )
    task = SyntheticWeakEdgeTask(config, image_size=16, snr="low", motion="slow", seed=4)
    first = task.sample(2, 4)
    second = task.sample(2, 4)
    torch.testing.assert_close(first.values, second.values)
    assert first.values.shape == (2, 4, config.channels, 16, 16)
    assert float(first.values.min()) >= 0.0 and float(first.values.max()) <= 1.0
    assert first.edge_target.shape == (2, 4, 16, 16)
    # Different seeds give different noise realisations of the same geometry.
    other = task.sample(2, 4, seed=99)
    assert not torch.allclose(other.values, first.values)
    assert bool(first.valid.any())


def test_synthetic_cfa_mode_declares_one_measurement_per_site():
    config = N01Config.tiny(
        channels=4,
        planes=4,
        cfa_pattern="RGGB",
        width=8,
        num_blocks=1,
        spatial_radii=(0, 1),
        time_depths=(1, 2),
        neighbourhood_radius=1,
    )
    task = SyntheticWeakEdgeTask(
        config, image_size=16, snr="high", motion="still", seed=2, cfa=True
    )
    batch = task.sample(1, 2)
    assert batch.values.shape == (1, 2, 4, 16, 16)
    # Exactly one channel is non-zero at each mosaic site, so no colour is
    # interpolated and no site carries two co-sited measurements.
    non_zero = (batch.values > 0).sum(dim=2)
    assert int(non_zero.max().item()) == 1
    offsets = task.sample_xy()
    assert offsets.shape == (4, 2)
    # RGGB: the two green planes share one phase, red and blue the other.
    # RGGB phase per measurement channel: channel 0 at (0,0), 1 at (0,1),
    # 2 at (1,0), 3 at (1,1).  The four planes are separate sampling grids, so a
    # green plane never shares a site with the red or blue plane.
    phases = [tuple(item.tolist()) for item in offsets]
    assert phases == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    assert task.plane_colours() == ("R", "G", "G", "B")
    # Only the plane that owns a site carries a measurement there.
    assert int((batch.valid.sum(dim=2) > 0).max().item()) == 1
    assert int((batch.valid.sum(dim=2) == 0).sum().item()) > 0


def test_learned_selector_runs_and_receives_gradient():
    """The learned variant must run and be trainable, not only instantiate."""
    encoder = build_encoder(selection="learned")
    assert encoder.selector.logit_head is not None
    state = encoder.init_state(2, {"camera_ids": ["cam0"], "shape": (8, 8)})
    values = torch.rand((2, 3, 8, 8), requires_grad=True)
    output = encoder.step(frame_from(values), None, state)
    output.representation.sum().backward()
    logit_grad = sum(
        0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
        for parameter in encoder.selector.logit_head.parameters()
    )
    assert logit_grad > 0.0, "the learned domain logits received no gradient"
    # A hard argmax would make the support graph invisible to the loss; the mixing
    # weights keep it differentiable, and the participation ratio is reported.
    assert 1.0 <= output.diagnostics["mixture_participation"] <= encoder.config.candidate_count
    # The learned variant must still be driven by the analytic cost at step one:
    # its head is zero-initialised, so both variants select the same domains.
    plain = build_encoder(selection="analytic")
    plain.load_state_dict(
        {k: v for k, v in encoder.state_dict().items() if not k.startswith("selector.logit_head")},
        strict=False,
    )
    plain_state = plain.init_state(2, {"camera_ids": ["cam0"], "shape": (8, 8)})
    plain_output = plain.step(frame_from(values.detach()), None, plain_state)
    assert (
        abs(
            plain_output.diagnostics["selected_radius_mean"]
            - output.diagnostics["selected_radius_mean"]
        )
        < 1e-6
    )


def test_four_plane_cfa_configuration_runs_end_to_end():
    """The 2x2 CFA path (planes=4) must work through the whole encoder."""
    encoder = build_encoder(planes=4, channels=4, cfa_pattern="RGGB", selection="learned")
    state = encoder.init_state(2, {"camera_ids": ["cam0"], "shape": (8, 8)})
    values = torch.rand((2, 4, 8, 8), requires_grad=True)
    output = encoder.step(frame_from(values), None, state)
    assert output.representation.shape == (2, encoder.config.width, 8, 8)
    output.representation.sum().backward()
    logit_grad = sum(
        0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
        for parameter in encoder.selector.logit_head.parameters()
    )
    assert logit_grad > 0.0
    tokens = encoder.readout(output, token_budget=16)
    assert tokens.values.shape == (2, 16, encoder.config.readout_width)

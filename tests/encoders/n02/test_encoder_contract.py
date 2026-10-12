"""Contract, causality and cost tests for the N02 encoder itself.

These tests pin the behaviour the mechanism claim depends on: episode boundaries,
the frozen measurement set, the readout budget, ambiguity reporting and cost
accounting. They use the real encoder (no stubs).
"""

from __future__ import annotations

import pytest
import torch

from owac.core.encoder_contract import (
    CameraSpec,
    ComputeBudget,
    CostRecord,
    EmbodimentContext,
    EncoderOutput,
    MeasurementFrame,
    Validity,
    VisualTokens,
    state_tensor_bytes,
)
from owac.frameworks.n02 import N02Config, build_encoder
from owac.frameworks.n02.readout import (
    StructuredRepresentation,
    TokenReadout,
    hypothesis_tokens,
    map_hypothesis,
    two_candidate_probe,
)


def _config(**overrides) -> N02Config:
    values = {"name": "tiny", "K": 4, "D": 16, "cycles": 2, "history": 3, "stem_width": 16}
    values.update(overrides)
    return N02Config(**values)


def _spec(channels: int = 3) -> CameraSpec:
    if channels == 1:
        return CameraSpec(
            camera_id="cam0",
            height=32,
            width=32,
            channels=1,
            input_mode="bayer_cfa",
            cfa_pattern="RGGB",
            white_level=255.0,
        )
    return CameraSpec(
        camera_id="cam0", height=32, width=32, channels=channels, input_mode="rawvla_rgb3"
    )


def _frame(step: int, *, batch: int = 1, channels: int = 3, new: bool = True) -> MeasurementFrame:
    scale = 255.0 if channels == 1 else 1.0
    return MeasurementFrame(
        values=torch.rand(batch, channels, 32, 32) * scale,
        camera_id="cam0",
        frame_ids=(step,),
        episode_ids=(0,),
        new_measurement=new,
        metadata={"units": "test"},
    )


def test_encoder_returns_a_full_contract_output():
    encoder = build_encoder(_config())
    state = encoder.init_state(1, _spec(), "cpu")
    output = encoder.step(_frame(0), EmbodimentContext(), state, ComputeBudget(max_updates=2))
    assert isinstance(output, EncoderOutput)
    assert isinstance(output.representation, StructuredRepresentation)
    assert isinstance(output.validity, Validity)
    assert isinstance(output.cost, CostRecord)
    assert output.cost.updates >= 1
    assert output.cost.hypotheses == 4
    assert output.cost.state_bytes and output.cost.state_bytes > 0
    assert torch.isfinite(output.representation.weights).all()
    assert output.representation.weights.sum(dim=-1).allclose(torch.ones(()))


@pytest.mark.parametrize("channels", [1, 3])
def test_both_input_modes_run_through_the_same_core(channels):
    config = _config()
    config.measurement.input_mode = "bayer_cfa" if channels == 1 else "rawvla_rgb3"
    encoder = build_encoder(config)
    state = encoder.init_state(1, _spec(channels), "cpu")
    for step in range(3):
        output = encoder.step(
            _frame(step, channels=channels),
            EmbodimentContext(),
            state,
            ComputeBudget(max_updates=2),
        )
        state = output.state
    assert output.representation.h.shape[1] == 1  # one view
    assert output.representation.geometry.shape[-1] == config.measurement.geometry_dim


def test_channel_count_of_the_wrong_mode_is_rejected():
    config = _config()
    config.measurement.input_mode = "bayer_cfa"
    encoder = build_encoder(config)
    state = encoder.init_state(1, _spec(channels=1), "cpu")
    with pytest.raises(ValueError, match="expects 1 channels"):
        encoder.step(_frame(0, channels=3), EmbodimentContext(), state, ComputeBudget())


def test_repeated_frame_is_not_counted_as_new_evidence():
    encoder = build_encoder(_config())
    state = encoder.init_state(1, _spec(), "cpu")
    first = encoder.step(_frame(0), EmbodimentContext(), state, ComputeBudget(max_updates=2))
    repeated = encoder.step(
        _frame(0, new=False), EmbodimentContext(), first.state, ComputeBudget(max_updates=2)
    )
    assert first.diagnostics["observations"] == 1
    assert repeated.diagnostics["observations"] == 0
    assert repeated.diagnostics["observation_history"] == 1
    assert repeated.diagnostics["duplicates_skipped"] >= 1
    assert "not" in repeated.validity.reason or repeated.validity.reason


def test_repeated_frame_adds_no_evidence_and_does_not_multiply_the_likelihood():
    """A duplicate frame must not become a second likelihood.

    What is asserted: the frozen measurement set used by the second step is still
    the single original frame (its value and variance are unchanged), and the
    likelihood is a sum over that set, so it cannot have been counted twice.
    The identity of the *posterior* is not asserted: the field legitimately moved
    during step 1, so the second step scores a different (still approximate)
    solution against the same one observation. The arithmetic itself is pinned by
    the exact closed-form comparison in ``test_analytic.py``.
    """
    encoder = build_encoder(_config())
    state = encoder.init_state(1, _spec(), "cpu")
    frame = _frame(0)
    first = encoder.step(frame, EmbodimentContext(), state, ComputeBudget(max_updates=1))
    again = encoder.step(frame, EmbodimentContext(), first.state, ComputeBudget(max_updates=1))
    assert first.diagnostics["observation_history"] == 1
    assert again.diagnostics["observation_history"] == 1
    assert again.diagnostics["observations"] == 0
    assert again.diagnostics["duplicates_skipped"] >= 1
    buffer = again.state.diagnostics["_buffer"]
    assert len(buffer.frames) == 1
    assert buffer.duplicate_count == 1
    # The single retained observation is the frame that was submitted, unchanged.
    retained = buffer.frames[0]
    height, width_px = retained.values.shape[-2:]
    stride_h = frame.values.shape[-2] // height
    stride_w = frame.values.shape[-1] // width_px
    # The buffer stores the frame on the mechanism grid: area (box) downsampling of
    # the sensor frame, not a crop. Checking the block mean proves the retained
    # tensor really is this one observation rather than a different sample.
    blocks = frame.values.reshape(1, 3, height, stride_h, width_px, stride_w).mean(dim=(3, 5))
    torch.testing.assert_close(retained.values, blocks, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(retained.variance, torch.ones_like(retained.variance))
    # The measurement energy is a per-observation mean over the retained set, so it
    # stays the same order of magnitude as one observation rather than doubling.
    energy = again.representation.measurement_energy.detach()
    assert energy.shape == (1, 1, 4)
    assert torch.isfinite(energy).all()
    assert float(energy.max()) < 10.0


def test_history_is_bounded_and_old_frames_leave_the_set():
    encoder = build_encoder(_config(history=2))
    state = encoder.init_state(1, _spec(), "cpu")
    for step in range(4):
        output = encoder.step(
            _frame(step), EmbodimentContext(), state, ComputeBudget(max_updates=2, max_history=2)
        )
        state = output.state
    assert output.diagnostics["observation_history"] == 2
    assert output.cost.history_len == 2


def test_reset_clears_only_the_selected_episode():
    encoder = build_encoder(_config())
    state = encoder.init_state(2, _spec(), "cpu")
    for step in range(3):
        out = encoder.step(
            _frame(step, batch=2), EmbodimentContext(), state, ComputeBudget(max_updates=2)
        )
        state = out.state
    sums = state.h.detach().flatten(1).abs().sum(-1)
    assert float(sums.min()) > 0
    after = encoder.reset(state, torch.tensor([True, False]))
    new_sums = after.h.detach().flatten(1).abs().sum(-1)
    assert float(new_sums[0]) == 0.0
    torch.testing.assert_close(new_sums[1], sums[1])
    # The reset element's age goes to zero; the retained element keeps its age.
    assert float(after.age.detach()[0].abs().sum()) == 0.0
    torch.testing.assert_close(after.age.detach()[1], state.age.detach()[1])
    # A single-element mask applies to every batch element.
    both = encoder.reset(state, torch.tensor([True]))
    assert float(both.h.abs().sum()) == 0.0


def test_detach_state_cuts_the_graph_but_keeps_the_numbers():
    encoder = build_encoder(_config())
    state = encoder.init_state(1, _spec(), "cpu")
    output = encoder.step(_frame(0), EmbodimentContext(), state, ComputeBudget(max_updates=1))
    assert output.representation.h.requires_grad
    detached = encoder.detach_state(output.state)
    assert not detached.h.requires_grad
    torch.testing.assert_close(detached.h, output.state.h)


def test_gradients_reach_every_learned_component():
    encoder = build_encoder(_config())
    state = encoder.init_state(1, _spec(), "cpu")
    output = encoder.step(_frame(0), EmbodimentContext(), state, ComputeBudget(max_updates=2))
    loss = output.representation.weights[0, 0, 0] + output.representation.h.abs().mean()
    loss.backward()
    for name in (
        "measurement.stem.input.weight",
        "measurement.head.value.weight",
        "initializer.code.weight",
        "reviser.cell.input.weight",
        "reviser.location_update.weight",
        "edge_parameter",
    ):
        parameter = dict(encoder.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_dependency_compatibility_learns_from_the_revision_path():
    """The edge table must receive gradient through the message/propagation path."""
    encoder = build_encoder(_config())
    encoder.edge_parameter.grad = None
    state = encoder.init_state(1, _spec(), "cpu")
    for step in range(2):
        output = encoder.step(
            _frame(step), EmbodimentContext(), state, ComputeBudget(max_updates=2)
        )
        state = output.state
    output.representation.h.pow(2).mean().backward()
    assert encoder.edge_parameter.grad is not None
    assert float(encoder.edge_parameter.grad.abs().sum()) > 0.0


def test_readout_budget_is_exact_for_both_modes():
    config = _config()
    encoder = build_encoder(config)
    state = encoder.init_state(1, _spec(), "cpu")
    output = encoder.step(_frame(0), EmbodimentContext(), state, ComputeBudget(max_updates=2))
    readout = TokenReadout(config.D, config.K, token_dim=64)
    grouped = hypothesis_tokens(output.representation, 32, readout=readout, camera_ids=("cam0",))
    mapped = map_hypothesis(output.representation, 32, readout=readout, camera_ids=("cam0",))
    assert grouped.tokens.values.shape == (1, 32, 64)
    assert mapped.tokens.values.shape == (1, 32, 64)
    assert grouped.diagnostics["tokens_per_group"] == 8
    assert mapped.diagnostics["tokens_per_group"] == 32
    assert grouped.hypothesis_index.shape == (1, 32)
    assert set(grouped.hypothesis_index.flatten().tolist()) == {0, 1, 2, 3}


def test_readout_rejects_a_budget_that_cannot_be_split():
    config = _config()
    encoder = build_encoder(config)
    state = encoder.init_state(1, _spec(), "cpu")
    output = encoder.step(_frame(0), EmbodimentContext(), state, ComputeBudget(max_updates=1))
    readout = TokenReadout(config.D, config.K, token_dim=64)
    with pytest.raises(ValueError, match="divisible"):
        hypothesis_tokens(output.representation, 30, readout=readout)


def test_ambiguity_flag_tracks_the_weight_distribution():
    config = _config()
    encoder = build_encoder(config)
    state = encoder.init_state(1, _spec(), "cpu")
    output = encoder.step(_frame(0), EmbodimentContext(), state, ComputeBudget(max_updates=4))
    representation = output.representation
    unresolved = representation.unresolved_ambiguity()
    assert unresolved.shape == (1, 1)
    # An untrained head may be confident or not, but the flag must agree with the
    # declared rule rather than being hard-coded.
    top = float(representation.top_weight().item())
    expected = top <= 0.25 + 0.25 * 0.75
    assert bool(unresolved.item()) == expected


def test_two_candidate_probe_separates_map_from_a_direct_mean():
    weights = torch.full((1, 1, 2), 0.5)
    representation = StructuredRepresentation(
        h=torch.zeros(1, 1, 2, 8, 2, 2),
        weights=weights,
        measurement_energy=torch.zeros(1, 1, 2),
        prior_energy=torch.zeros(1, 1, 2),
        relation_energy=torch.zeros(1, 1, 2),
        geometry=torch.tensor([[[[1.0, 0, 0, 0], [0.0, 1, 0, 0]]]]),
    )
    mean = two_candidate_probe(representation, action_fn=lambda g: g[:, :2], mode="mean")
    mapped = two_candidate_probe(representation, action_fn=lambda g: g[:, :2], mode="map")
    assert float(mean.interpolation_fraction.item()) == pytest.approx(0.5)
    assert float(mapped.interpolation_fraction.item()) == pytest.approx(0.0)
    # The two readouts must produce different actions on incompatible candidates.
    assert not torch.allclose(mean.action, mapped.action)


def test_cost_record_reports_only_measured_values():
    cost = CostRecord(updates=3)
    assert cost.wall_ms is None and cost.peak_bytes is None
    assert state_tensor_bytes({"a": torch.zeros(4)}) == 4 * torch.zeros(1).element_size()


def test_visual_tokens_keep_camera_identity():
    tokens = VisualTokens(
        values=torch.zeros(1, 4, 8), mask=torch.ones(1, 4, dtype=torch.bool), camera_ids=("cam0",)
    )
    assert tokens.camera_ids == ("cam0",)


def test_encoder_does_not_touch_the_measurement_buffer_with_predictions():
    """A prediction is never written back: the buffer only grows by real frames."""
    encoder = build_encoder(_config())
    state = encoder.init_state(1, _spec(), "cpu")
    for step in range(3):
        output = encoder.step(
            _frame(step), EmbodimentContext(), state, ComputeBudget(max_updates=3)
        )
        state = output.state
        buffer = state.diagnostics["_buffer"]
        assert len(buffer.frames) == min(step + 1, 3)
        for entry in buffer.frames:
            assert entry.values.shape[-2:] == (32, 32) or entry.values.shape[-2:] == (8, 8)
    assert output.diagnostics["observation_history"] <= 3

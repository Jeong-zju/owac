"""Counterexamples at the measurement/support boundary."""

from __future__ import annotations

import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01_full.model import N01V12Config, N01V12Model
from owac.frameworks.n01_full.support import SpatialSupport, SupportConfig


def _model():
    return N01V12Model(
        N01V12Config(
            feature_width=8,
            stem_width=4,
            residual_blocks=1,
            window_frames=8,
            support=SupportConfig(region_grid=2, receptive_field_radius=0),
        )
    )


def _push(model, state, frame_id, values):
    return model.step(
        MeasurementFrame(values, frame_ids=torch.tensor([frame_id]), camera_id="agentview"),
        state=state,
        budget=ComputeBudget(max_updates=8, max_history=8),
    ).state


def test_duplicate_arrival_does_not_advance_any_support_evidence():
    model = _model()
    state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
    values = torch.rand(1, 3, 16, 16)
    for frame_id in range(4):
        state = _push(model, state, frame_id, values + frame_id * 0.001)
    entry = state.support["agentview"]
    before = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in entry["detector"].state_dict().items()
    }
    age = entry["age"].clone()
    for _ in range(3):
        state = _push(model, state, 3, values + 0.003)
    after = state.support["agentview"]
    for key, value in before.items():
        actual = after["detector"].state_dict()[key]
        assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual, key
    assert torch.equal(age, after["age"])


def test_episode_reset_has_one_prior_hypothesis():
    support = SpatialSupport(SupportConfig(region_grid=2))
    state = support.initial_state(1, 7, 16, 16, "cpu")
    for _ in range(10):
        state["detector"].update(torch.ones(1, 4))
    reset = support.reset(state, torch.tensor([True]))
    normalizer = torch.logsumexp(reset["detector"].log_message, -1)
    assert torch.allclose(normalizer, torch.zeros_like(normalizer))
    fresh = support.initial_state(1, 7, 16, 16, "cpu")
    for _ in range(6):
        reset["detector"].update(torch.ones(1, 4))
        fresh["detector"].update(torch.ones(1, 4))
    assert torch.allclose(
        reset["detector"].run_length_log_probability(),
        fresh["detector"].run_length_log_probability(),
    )


def test_invalid_reference_cannot_establish_valid_support_residual():
    support = SpatialSupport(SupportConfig(region_grid=2))
    values = torch.rand(1, 3, 3, 16, 16)
    mask = torch.ones(1, 3, 1, 16, 16, dtype=torch.bool)
    mask[:, 0] = False
    _, valid = support.region_residual(values, mask)
    assert not valid.any()


def test_partial_reset_keeps_other_member_and_one_prior_mass():
    support = SpatialSupport(SupportConfig(region_grid=2))
    state = support.initial_state(2, 7, 16, 16, "cpu")
    for _ in range(8):
        state["detector"].update(torch.ones(2, 4))
    retained = state["detector"].log_message[1].clone()
    reset = support.reset(state, torch.tensor([True, False]))
    posterior = reset["detector"].log_message
    assert torch.equal(posterior[1], retained)
    assert torch.allclose(torch.logsumexp(posterior[0], -1), torch.zeros(4))
    assert torch.isfinite(posterior[0]).sum(-1).tolist() == [1] * 4


def test_detector_and_source_shift_only_advance_selected_members():
    support = SpatialSupport(SupportConfig(region_grid=2))
    state = support.initial_state(2, 7, 16, 16, "cpu")
    for _ in range(4):
        state["detector"].update(torch.ones(2, 4))
    before = state["detector"].run_length_log_probability()[1].clone()
    for _ in range(4):
        state["detector"].update(torch.ones(2, 4), active=torch.tensor([[True] * 4, [False] * 4]))
    after = state["detector"].run_length_log_probability()[1]
    assert torch.equal(after[:, : before.shape[-1]], before)
    assert not torch.isfinite(after[:, before.shape[-1] :]).any()
    state["age"][:] = torch.arange(7)[None, :, None]
    shifted = support.shift(state, selected=torch.tensor([True, False]))
    assert torch.equal(shifted["age"][1], state["age"][1])
    assert shifted["age"][0, 0].sum() == 0
    assert torch.equal(shifted["age"][0, 1:], state["age"][0, :-1])

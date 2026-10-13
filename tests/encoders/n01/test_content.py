"""Content-path source, information-preservation, and compatibility checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01.config import N01Config, config_fingerprint
from owac.frameworks.n01.content import ContentFields, ContentUpdate, content_fields
from owac.frameworks.n01.model import N01Encoder
from owac.frameworks.n01.motion import MotionField
from owac.frameworks.n01.sequence1d import N01Sequence1DEncoder
from owac.frameworks.n01.state import push
from owac.frameworks.n01.support import SupportDomainIntegrator


def config_for(mode="integrated", **kwargs):
    base = dict(
        content_mode=mode,
        width=8,
        num_blocks=1,
        readout_tokens=16,
        readout_width=16,
        spatial_radii=(0, 1),
        neighbourhood_radius=1,
        max_history=4,
        time_depths=(1, 4),
        learnable_gate=False,
        aux_measurement_prediction=False,
    )
    base.update(kwargs)
    return N01Config.tiny(**base)


def frame(values, index, variance=0.02, valid=None, sample_xy=None):
    return MeasurementFrame(
        values=values,
        valid_mask=torch.ones_like(values, dtype=torch.bool) if valid is None else valid,
        camera_id="camera",
        frame_ids=torch.full((values.shape[0],), index),
        timestamp=torch.full((values.shape[0],), float(index)),
        time_unit="frame",
        sample_xy=sample_xy,
        metadata={"variance": torch.as_tensor(variance).expand_as(values)},
    )


def init(encoder, batch=1, size=8):
    return encoder.init_state(batch, {"camera_ids": ["camera"], "shape": (size, size)})


def selection_for(statistics):
    available = statistics.support_count > 0
    weights = available.float() / available.sum(0, keepdim=True).clamp_min(1)
    return {"weights": weights, "slice": torch.arange(statistics.candidate_count)}


def test_shared_grid_content_preserves_channel_specific_hand_integrals():
    encoder = N01Encoder(config_for(spatial_radii=(0,), time_depths=(2,)))
    state = init(encoder)
    first = torch.tensor([0.10, 0.20, 0.70])[None, :, None, None].expand(1, 3, 8, 8)
    second = torch.tensor([0.30, 0.50, 0.10])[None, :, None, None].expand_as(first)
    var_first = torch.tensor([0.04, 0.09, 0.16])[None, :, None, None].expand_as(first)
    var_second = torch.tensor([0.01, 0.04, 0.25])[None, :, None, None].expand_as(first)
    state = encoder.step(frame(first, 0, var_first), state=state).state
    output = encoder.step(frame(second, 1, var_second), state=state)
    fields = encoder.last_content_fields
    old_weight, new_weight = 1 / (var_first + 1e-6), 1 / (var_second + 1e-6)
    total = old_weight + new_weight
    expected = (old_weight * first + new_weight * second) / total
    expected_var = (old_weight.square() * var_first + new_weight.square() * var_second) / (
        total.square() + 1e-6
    )
    torch.testing.assert_close(fields.mean, expected)
    torch.testing.assert_close(fields.variance_proxy, expected_var)
    torch.testing.assert_close(fields.age, old_weight / total)
    assert fields.mean.shape[1] == 3
    assert output.cost["content_candidate_statistics_computed"] == 3


def test_fractional_content_reuses_original_source_coefficient_statistics():
    config = config_for(channels=3, alignment="fixed", fractional_alignment=True, time_depths=(2,))
    encoder = N01Encoder(config)
    state = init(encoder, size=5)
    values = torch.arange(75, dtype=torch.float32).reshape(1, 3, 5, 5) / 80
    state = push(state, frame(values, 0))
    state = push(state, frame(values.flip(-1), 1))
    edges = torch.ones(1, 1, 2, 5, 5)
    motion = [
        MotionField(
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            fractional=torch.tensor([[0.0, 0.0 if slot == 0 else 0.5]]),
        )
        for slot in range(state.num_slots)
    ]
    statistics, _ = encoder.integrator.domain_statistics(state, edges, motion=motion)
    selection = selection_for(statistics)
    fields = content_fields(
        config,
        state,
        statistics,
        selection,
        integrator=encoder.content_integrator,
        edges=edges,
        motion=motion,
    )
    primitive = SupportDomainIntegrator(config.replace(channels=1, learnable_gate=False))
    for channel in range(3):
        view = replace(
            state,
            values=state.values[:, :, channel : channel + 1],
            variance=state.variance[:, :, channel : channel + 1],
            valid=state.valid[:, :, channel : channel + 1],
        )
        reference, _ = primitive.fractional_statistics(view, edges, motion, config.time_depths)
        weights = selection["weights"]
        torch.testing.assert_close(
            fields.mean[:, channel : channel + 1], (weights * reference.mu).sum(0)
        )
        # Overlapping candidates are mixed conservatively, never as independent.
        torch.testing.assert_close(
            fields.variance_proxy[:, channel : channel + 1],
            (weights * reference.variance_proxy).sum(0),
        )


def test_cfa_history_permutation_survives_interior_feature_update():
    def encode_pair(mode):
        config = config_for(
            mode,
            input_mode="bayer_cfa",
            channels=4,
            planes=4,
            max_history=8,
            time_depths=(8,),
        )
        torch.manual_seed(5)
        encoder = N01Encoder(config).eval()
        twin = deepcopy(encoder)
        outputs = []
        for network, order in ((encoder, [0, 1, 2, 3]), (twin, [3, 1, 2, 0])):
            state = init(network, size=12)
            for step in range(8):
                values = (
                    torch.tensor([0.375, 0.125, 0.125, 0.125])[order]
                    if step < 7
                    else torch.full((4,), 0.1875)
                )
                values = values[None, :, None, None].expand(1, 4, 12, 12)
                output = network.step(
                    frame(values, step, 1 / 256 if step < 7 else 1 / 16), state=state
                )
                state = output.state
            outputs.append(output.representation)
        return outputs

    with torch.no_grad():
        legacy_a, legacy_b = encode_pair("gate_only")
        new_a, new_b = encode_pair("integrated")
    torch.testing.assert_close(legacy_a, legacy_b, atol=1e-6, rtol=0)
    # Exclude padding effects: measured DC differences must reach interior features.
    assert float((new_a[..., 3:-3, 3:-3] - new_b[..., 3:-3, 3:-3]).abs().mean()) > 1e-4


def test_current_control_content_has_no_historical_input():
    config = config_for("current_control")
    encoder = N01Encoder(config)
    state = init(encoder)
    values = torch.rand(1, 3, 8, 8)
    state = push(state, frame(values * 0.1, 0))
    state = push(state, frame(values, 1))
    first = content_fields(config, state, None, None)
    altered = deepcopy(state)
    altered.values[1:] = 1e4
    altered.variance[1:] = 1e-9
    altered.valid[1:] = False
    second = content_fields(config, altered, None, None)
    for name in ("mean", "variance_proxy", "age", "valid", "coordinates"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert torch.equal(first.mean, values)


def test_content_gradients_reach_history_values_projection_reliability_and_coordinates():
    torch.manual_seed(41)
    encoder = N01Encoder(config_for(time_depths=(4,)))
    state = init(encoder)
    old = torch.rand(1, 3, 8, 8, requires_grad=True)
    current = torch.rand(1, 3, 8, 8)
    xy = torch.rand(3, 2, requires_grad=True)
    state = encoder.step(frame(old, 0), state=state).state
    output = encoder.step(frame(current, 1, sample_xy=xy), state=state)
    output.representation.square().mean().backward()
    assert old.grad is not None and float(old.grad.abs().sum()) > 0
    assert xy.grad is not None and float(xy.grad.abs().sum()) > 0
    branch = encoder.blocks[0].content_update
    for module in (branch.projection, branch.reliability):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0


def test_invalid_content_and_phase_coordinates_cannot_inject_measured_features():
    config = config_for("current_control")
    encoder = N01Encoder(config)
    state = init(encoder)
    values = torch.full((1, 3, 8, 8), float("nan"))
    state = push(state, frame(values, 0, valid=torch.zeros_like(values, dtype=torch.bool)))
    fields = content_fields(config, state, None, None, torch.full((3, 2), 100.0))
    branch = encoder.blocks[0].content_update
    actual = branch(fields, torch.ones(6))
    assert bool(torch.isfinite(actual).all())
    assert torch.count_nonzero(actual) == 0
    assert torch.count_nonzero(fields.coordinates) == 0
    assert torch.count_nonzero(fields.mean) == 0


def test_partial_missing_channel_cannot_change_content_projection():
    config = config_for("current_control")
    encoder = N01Encoder(config)
    state = init(encoder)
    values = torch.rand(1, 3, 8, 8)
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[:, 1] = False
    state = push(state, frame(values, 0, valid=valid))
    fields = content_fields(config, state, None, None)
    modified = replace(
        fields,
        mean=fields.mean.clone(),
        variance_proxy=fields.variance_proxy.clone(),
        age=fields.age.clone(),
        coordinates=fields.coordinates.clone(),
    )
    modified.mean[:, 1] = float("nan")
    modified.variance_proxy[:, 1] = float("inf")
    modified.age[:, 1] = float("nan")
    modified.coordinates[:, 1] = 1e6
    branch = encoder.blocks[0].content_update
    torch.testing.assert_close(branch(fields, torch.ones(6)), branch(modified, torch.ones(6)))


def test_sample_coordinates_remain_content_specific_and_quality_independent():
    config = config_for("integrated", input_mode="bayer_cfa", channels=4, planes=4)
    torch.manual_seed(7)
    branch = ContentUpdate(config)
    valid = torch.ones(1, 4, 8, 8, dtype=torch.bool)
    fields = ContentFields(
        mean=torch.ones(1, 4, 8, 8) * 0.2,
        variance_proxy=torch.ones(1, 4, 8, 8) * 0.02,
        age=torch.ones(1, 4, 8, 8),
        valid=valid,
        coordinates=torch.zeros(1, 4, 2, 8, 8),
    )
    shifted = replace(fields, coordinates=fields.coordinates + 0.5)
    gates = []
    handle = branch.reliability.register_forward_hook(
        lambda _m, _a, out: gates.append(out.detach())
    )
    first, second = branch(fields, torch.ones(6)), branch(shifted, torch.ones(6))
    handle.remove()
    assert torch.equal(gates[0], gates[1])
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("mode", ["integrated", "current_control"])
def test_added_capacity_preserves_legacy_initialization_and_zero_branch_output(mode):
    torch.manual_seed(43)
    legacy = N01Encoder(config_for("gate_only"))
    torch.manual_seed(43)
    extended = N01Encoder(config_for(mode))
    for name, parameter in legacy.named_parameters():
        assert torch.equal(parameter, dict(extended.named_parameters())[name]), name
    # With extra content zeroed, the feature update is the exact legacy computation.
    for block in extended.blocks:
        for parameter in block.content_update.parameters():
            parameter.data.zero_()
    first, second = init(legacy), init(extended)
    for index in range(3):
        observed = frame(torch.rand(1, 3, 8, 8), index)
        a, b = legacy.step(observed, state=first), extended.step(observed, state=second)
        assert torch.equal(a.representation, b.representation)
        first, second = a.state, b.state
        assert torch.equal(first.values, second.values)
        assert torch.equal(first.frame_ids, second.frame_ids)
        for name in ("edges", "open_edges", "merge_count", "split_count"):
            assert torch.equal(
                getattr(first.support["camera"], name), getattr(second.support["camera"], name)
            )


def test_duplicate_frame_reuses_representation_and_does_not_integrate_again():
    encoder = N01Encoder(config_for())
    observed = frame(torch.rand(1, 3, 8, 8), 0)
    first = encoder.step(observed, state=init(encoder))
    previous_fields = encoder.last_content_fields
    duplicate = encoder.step(observed, state=first.state)
    assert duplicate.diagnostics["reused_representation"]
    assert duplicate.representation is first.representation
    assert encoder.last_content_fields is previous_fields
    assert torch.equal(duplicate.state.values, first.state.values)
    assert torch.equal(duplicate.state.observed, first.state.observed)


def test_integrated_and_current_control_have_identical_trainable_capacity():
    torch.manual_seed(9)
    integrated = N01Encoder(config_for("integrated"))
    torch.manual_seed(9)
    control = N01Encoder(config_for("current_control"))
    params = dict(integrated.named_parameters())
    assert params.keys() == dict(control.named_parameters()).keys()
    for name, parameter in control.named_parameters():
        assert torch.equal(parameter, params[name])
    assert config_fingerprint(integrated.config) != config_fingerprint(control.config)


@pytest.mark.parametrize("strict", [True, False])
def test_mode_mismatch_and_wrong_version_are_rejected_even_on_partial_load(strict):
    models = {
        mode: N01Encoder(config_for(mode))
        for mode in ("gate_only", "integrated", "current_control")
    }
    for source_mode, source in models.items():
        for destination_mode, destination in models.items():
            if source_mode == destination_mode:
                destination.load_state_dict(source.state_dict(), strict=strict)
            else:
                with pytest.raises(RuntimeError, match="content_mode_version mismatch"):
                    destination.load_state_dict(source.state_dict(), strict=strict)
    invalid = deepcopy(models["integrated"].state_dict())
    invalid["content_mode_version"] = torch.tensor(9)
    with pytest.raises(RuntimeError, match="content_mode_version mismatch"):
        models["integrated"].load_state_dict(invalid, strict=strict)
    # Gate-only preserves pre-change state keys and does not require a new marker.
    assert "content_mode_version" not in models["gate_only"].state_dict()


def test_unsupported_sequence_content_mode_is_explicit():
    with pytest.raises(ValueError, match="sequence1d currently supports only"):
        config_for(input_mode="sequence1d")
    # Existing callers can construct the 1-D adapter with the default input_mode.
    # The concrete adapter must reject unsupported modes independently of config.
    for mode in ("integrated", "current_control"):
        with pytest.raises(ValueError, match="1-D adapter currently supports only"):
            N01Sequence1DEncoder(config_for(mode, channels=1))


def test_content_rebuilds_after_support_split_without_cross_boundary_measurements():
    config = config_for(
        input_mode="bayer_cfa", planes=4, channels=4, spatial_radii=(1,), time_depths=(4,)
    )
    encoder = N01Encoder(config)
    state = init(encoder)
    values = torch.full((1, 4, 8, 8), 0.1)
    values[:, 0, :, 4:] = 0.9
    for index in range(2):
        state = push(state, frame(values, index))
    preserved = state.values.clone()
    edges = torch.ones(1, 4, 2, 8, 8)

    def measured_content(edges):
        statistics, _ = encoder.integrator.domain_statistics(state, edges)
        return content_fields(config, state, statistics, selection_for(statistics))

    merged = measured_content(edges)
    edges[:, :, 0, :, 4] = 0
    split = measured_content(edges)
    assert float(merged.mean[0, 0, 3, 3]) > 0.3
    assert float(split.mean[0, 0, 3, 3]) == pytest.approx(0.1)
    assert float(split.mean[0, 0, 3, 4]) == pytest.approx(0.9)
    assert torch.equal(state.values, preserved)


def test_all_missing_integrated_frame_remains_finite_with_zero_content():
    encoder = N01Encoder(config_for())
    values = torch.rand(1, 3, 8, 8)
    output = encoder.step(
        frame(values, 0, valid=torch.zeros_like(values, dtype=torch.bool)), state=init(encoder)
    )
    assert bool(torch.isfinite(output.representation).all())
    assert torch.count_nonzero(encoder.last_content_fields.mean) == 0
    assert not bool(encoder.last_content_fields.valid.any())
    branch = encoder.blocks[0].content_update
    assert torch.count_nonzero(branch(encoder.last_content_fields, torch.ones(6))) == 0

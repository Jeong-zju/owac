"""Behavioral regression and training audits for the six source-review findings."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig, N04TrainConfig
from owac.frameworks.n04.geometry import (
    GridSpec,
    backward_grid_from_translation,
    warp_feature_field,
)
from owac.frameworks.n04.innovation import InnovationDecomposer, estimate_low_frequency_gain
from owac.frameworks.n04.model import N04Encoder, linear_measurement_planes, warp_linear_measurement
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import _change_loss, build_context, decomposition_loss, train


def config():
    return N04Config(
        measurement_size=(16, 16),
        geometry=N04GeometryConfig(feature_size=(16, 16), pixel_stride=1),
        model=N04ModelConfig(width=4, hidden=8, blocks=1, scales=1, readout_tokens=4),
    )


def test_real_training_updates_confidence_stability_and_distinguishes_causes(tmp_path):
    recipe = N04TrainConfig(
        steps=3, batch_size=1, sequence_length=5, burn_in=1, tbptt=3, threads=2, log_every=0
    )
    train(config(), recipe, output_dir=tmp_path)
    payload = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    state = payload["encoder"]
    assert state["propagator.head.weight"][2].abs().sum() > 0
    assert state["decomposer.head.weight"][4].abs().sum() > 0
    assert not torch.equal(state["decomposer.head.weight"][0], state["decomposer.head.weight"][3])


def test_cause_and_stability_heads_have_different_update_effects():
    dec = InnovationDecomposer(feature_channels=4, hidden=8)
    inputs = dict(
        current=torch.ones(1, 4, 8, 8),
        predicted=torch.zeros(1, 4, 8, 8),
        validity=torch.ones(1, 1, 8, 8),
        uncertainty=torch.full((1, 1, 8, 8), 0.2),
    )
    gates = []
    for cause in (0, 3):
        with torch.no_grad():
            dec.head.bias.zero_()
            dec.head.bias[cause] = 20
        gates.append(dec(**inputs).g_update.mean())
    assert gates[0] > gates[1]
    with torch.no_grad():
        dec.head.bias[4] = -20
    low = dec(**inputs).g_update.mean()
    with torch.no_grad():
        dec.head.bias[4] = 20
    assert dec(**inputs).g_update.mean() > low


def test_uniform_moving_interior_is_not_given_impossible_cell_amplitude_target():
    zero = torch.zeros(1, 4, 16, 16, requires_grad=True)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., 4:12, 4:12] = 1
    frame = SimpleNamespace(
        measured=zero,
        object_mask=mask,
        true_object_translation=torch.tensor([[1.0, 0.0]]),
        condition="object_only",
        metadata={"object_moving": [True]},
        valid_mask=mask,
        saturated_mask=torch.zeros_like(mask),
    )
    loss = _change_loss(zero, frame, residual=torch.zeros_like(zero))
    assert loss.item() == 0
    # Observable nonzero innovations still carry nonzero supervision.
    assert _change_loss(zero, frame, residual=torch.ones_like(zero)) > 0.9


@pytest.mark.parametrize("dx", [-1, 1])
def test_old_change_and_stable_use_the_same_motion_coordinates(dx):
    cfg = config()
    encoder = N04Encoder(cfg)
    state = encoder.init_state(1)["head"]
    state.stable[..., 8, 8] = 1
    state.change[..., 8, 8] = 1
    state.warp_validity = torch.ones(1, 1, 16, 16)
    grid = backward_grid_from_translation(encoder.spec, dx, 0, batch=1)
    warp = warp_feature_field(state.stable, grid, encoder.spec)
    predicted = warp.values
    corrected, _ = encoder.correct_after_observe(
        state,
        predicted,
        predicted,
        warp.validity[:, None],
        torch.zeros(1, 1, 16, 16),
        backward_grid=grid,
    )
    assert corrected.change[0, 0].argmax() == predicted[0, 0].argmax()
    assert corrected.change[0, 0, 8, 8] == 0


@pytest.mark.parametrize("pattern", [None, "RGGB", "BGGR", "GRBG", "GBRG"])
def test_actual_stem_illumination_is_estimated_in_linear_measurement_domain(pattern):
    cfg = config()
    if pattern:
        cfg = replace(cfg, input_mode="bayer_cfa", channels=1, cfa_pattern=pattern)
    encoder = N04Encoder(cfg)
    scene = SceneConfig(
        height=16,
        width=16,
        input_mode=cfg.input_mode,
        cfa_pattern=cfg.cfa_pattern,
        object_colors=((0.4, 0.2, 0.1),),
        noise_sigma=0,
    )
    frame = SyntheticSequenceGenerator(scene).sequence(length=2, condition="illumination")[0]
    values = frame.measured * 0.5
    state = encoder.init_state(1)["head"]
    first = encoder.step(values, build_context(frame, None), state)
    second = encoder.step(values * 1.5, build_context(frame, frame), first.state)
    assert second.innovation.illumination_gain.median().item() == pytest.approx(1.5, abs=1e-3)
    assert torch.allclose(second.state.measured, encoder.encode_measurement(values * 1.5))


def test_unknown_linear_domain_does_not_fabricate_gain_from_normalized_features():
    dec = InnovationDecomposer(feature_channels=4, hidden=8)
    result = dec(
        current=torch.ones(1, 4, 8, 8) * 2,
        predicted=torch.ones(1, 4, 8, 8),
        validity=torch.ones(1, 1, 8, 8),
        uncertainty=torch.ones(1, 1, 8, 8),
    )
    assert not result.diagnostics["illumination_available"]
    assert torch.all(result.illumination_gain == 1)


def test_low_frequency_gain_excludes_invalid_content_and_empty_channels():
    previous = torch.ones(1, 3, 12, 12)
    current = previous * 1.5
    valid = torch.ones_like(previous)
    valid[..., 6:] = 0
    valid[:, :2] = 0  # only one measured channel: empty planes cannot vote gain=0.2
    before, _ = estimate_low_frequency_gain(current, previous, valid)
    current[..., 6:] = 100
    after, _ = estimate_low_frequency_gain(current, previous, valid)
    assert torch.allclose(before * valid, after * valid)
    assert after[:, 2, :, :6].median() == pytest.approx(1.5, abs=1e-4)
    empty, _ = estimate_low_frequency_gain(current, previous, torch.zeros_like(valid))
    assert torch.all(empty == 1)


@pytest.mark.parametrize("dx", [0, 1, 2])
def test_cfa_linear_warp_respects_sampling_offsets(dx):
    spec = GridSpec(16, 16)
    ys, xs = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    values = (0.2 + xs * 0.01 + ys * 0.005)[None, None]
    planes, valid = linear_measurement_planes(values, torch.ones_like(values), "bayer_cfa")
    grid = backward_grid_from_translation(spec, dx, 0, batch=1)
    warped, support = warp_linear_measurement(planes, valid, grid, "bayer_cfa")
    assert torch.allclose(warped[..., 2:-2, 2:-2], planes[..., 2:-2, 2:-2] - 0.01 * dx, atol=1e-6)
    assert (support[..., 2:-2, 2:-2] > 0.99).all()


def test_linear_history_lifecycle_preserves_other_batch_and_detaches_graph():
    encoder = N04Encoder(config())
    state = encoder.init_state(2)["head"]
    state.linear_measurement = torch.ones(2, 3, 16, 16, requires_grad=True)
    state.linear_validity = torch.ones(2, 1, 16, 16)
    size = state.state_bytes()
    selected = state.select(slice(1, 2))
    assert selected.linear_measurement.shape[0] == 1
    state.detach().masked_reset(torch.tensor([True, False]))
    assert not state.linear_measurement.requires_grad
    assert state.linear_measurement[0].sum() == 0
    assert state.linear_validity[1].sum() == 256
    assert state.state_bytes() == size


@pytest.mark.parametrize("mode", ["rawvla_rgb3", "bayer_cfa"])
def test_linear_grid_resizing_preserves_identity_even_at_sensor_borders(mode):
    values = torch.rand(1, 1 if mode == "bayer_cfa" else 3, 16, 16)
    planes, valid = linear_measurement_planes(values, torch.ones_like(values[:, :1]) * 0.5, mode)
    grid = GridSpec(4, 4).identity_grid(1, values.device, values.dtype)
    warped, support = warp_linear_measurement(planes, valid, grid, mode)
    assert torch.allclose(warped, planes, atol=1e-6)
    assert torch.allclose(support, valid, atol=1e-6)


def test_stability_supervision_uses_the_same_calibrated_scale_as_its_output():
    cfg = config()
    valid = torch.ones(1, 1, 16, 16)
    frame = SimpleNamespace(
        measured=valid,
        object_mask=valid,
        true_object_translation=torch.zeros(1, 2),
        condition="camera_only",
        metadata={"object_moving": [False]},
        valid_mask=valid,
        saturated_mask=torch.zeros_like(valid),
        oracle_camera_flow=torch.zeros(1, 16, 16, 2),
        true_illumination=torch.ones(1),
    )
    innovation = SimpleNamespace(
        mix=torch.full((1, 4, 16, 16), 0.25),
        propagation_validity=valid,
        gate_change=torch.ones_like(valid) * 0.25,
        stability=valid * torch.exp(torch.tensor(-cfg.model.sigma_floor)),
    )
    loss = decomposition_loss(innovation, frame.oracle_camera_flow, frame, frame, cfg)
    assert loss.item() == 0

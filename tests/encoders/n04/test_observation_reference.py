"""Innovation compares sensor observations independently of task-state corrections."""

from dataclasses import replace

import pytest
import torch

from owac.frameworks.n04.model import (
    N04Encoder,
    linear_measurement_planes,
    measurement_from_linear_planes,
)
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import build_context, load_checkpoint

from .test_six_fixes import config


def cfg(reference="sensor_history", **changes):
    c = config().with_model(predictor="identity", innovation_reference=reference, **changes)
    return c


def test_static_sensor_innovation_ignores_nonzero_task_block_corrections():
    old = N04Encoder(cfg("task_state"))
    new = N04Encoder(cfg())
    with torch.no_grad():
        old.blocks[0].mix[-1].bias.fill_(2)
    new.load_state_dict(old.state_dict())
    scene = SceneConfig(height=16, width=16, noise_sigma=0, object_colors=((0.4, 0.2, 0.1),))
    seq = SyntheticSequenceGenerator(scene).sequence(
        length=6, batch=2, condition="illumination", seed=99
    )
    states = [x.init_state(2)["head"] for x in (old, new)]
    previous = None
    for frame in seq:
        outputs = []
        for i, encoder in enumerate((old, new)):
            out = encoder.step(
                frame.measured,
                build_context(frame, previous),
                states[i],
                saturation=frame.saturated_mask,
                measurement_valid=frame.valid_mask,
            )
            states[i] = out.state
            outputs.append(out)
        previous = frame
    assert outputs[0].innovation.residual.abs().mean() > 0.08
    assert outputs[1].innovation.residual.abs().max() < 2e-5
    assert (states[1].stable - new.encode_measurement(frame.measured)).abs().mean() > 0.08


def test_first_observation_has_no_fabricated_external_history():
    encoder = N04Encoder(cfg())
    state = encoder.init_state(1)["head"]
    scene = SceneConfig(height=16, width=16, noise_sigma=0, object_colors=((0.4, 0.2, 0.1),))
    frame = SyntheticSequenceGenerator(scene).sequence(length=1, condition="object_only")[0]
    out = encoder.step(frame.measured, build_context(frame, None), state)
    assert out.innovation.propagation_validity.sum() == 0
    assert out.innovation.external_evidence.sum() == 0


@pytest.mark.parametrize("pattern", ("RGGB", "BGGR", "GRBG", "GBRG"))
def test_sensor_prediction_preserves_cfa_sampling_phase(pattern):
    c = replace(cfg(), input_mode="bayer_cfa", channels=1, cfa_pattern=pattern)
    encoder = N04Encoder(c)
    samples = torch.rand(1, 1, 16, 16)
    planes, _ = linear_measurement_planes(samples, torch.ones_like(samples), "bayer_cfa")
    torch.testing.assert_close(measurement_from_linear_planes(planes, "bayer_cfa"), samples)
    torch.testing.assert_close(
        encoder.predict_observation(planes), encoder.encode_measurement(samples)
    )


def test_observation_prediction_has_gradient_to_encoder_and_propagated_history():
    encoder = N04Encoder(cfg())
    past = torch.rand(1, 3, 16, 16, requires_grad=True)
    prediction = encoder.predict_observation(past)
    prediction.square().sum().backward()
    assert past.grad.abs().sum() > 0
    assert encoder.stem.conv[0].weight.grad.abs().sum() > 0


def test_checkpoint_reference_change_requires_explicit_diagnostic_override(tmp_path):
    old = N04Encoder(cfg("task_state"))
    new = N04Encoder(cfg())
    path = tmp_path / "cp.pt"
    torch.save(
        dict(
            encoder=old.state_dict(),
            model_config=old.config.model.as_dict(),
            mechanism_revision="n04-independent-event-scores-v1",
        ),
        path,
    )
    with pytest.raises(ValueError, match="observation reference"):
        load_checkpoint(path, encoder=new)
    load_checkpoint(path, encoder=new, allow_design_override=True)

"""Gain changes are explained in linear measurement space; motion remains visible."""

import pytest
import torch

from owac.frameworks.n04.model import N04Encoder, linear_measurement_planes
from owac.training.n04 import load_checkpoint

from .test_observation_reference import cfg


def result(encoder, previous, current, support):
    feature = encoder.encode_measurement(current * support)
    state = encoder.init_state(current.shape[0])["head"]
    return encoder.correct_after_observe(
        state,
        feature,
        feature,
        support,
        torch.zeros_like(support),
        linear_current=current,
        linear_predicted=previous,
        linear_validity=support,
        current_linear_support=support,
    )[1]


def test_gain_explains_prediction_without_modifying_measurement():
    torch.manual_seed(4)
    encoder = N04Encoder(cfg(illumination_explanation=True, match_observation_support=True))
    raw = torch.rand(2, 3, 16, 16) * 0.3 + 0.1
    mask = torch.ones(2, 1, 16, 16)
    current = raw * 1.6
    original = current.clone()
    explained = result(encoder, raw, current, mask)
    assert explained.residual.abs().max() < 2e-4
    torch.testing.assert_close(current, original)
    old = N04Encoder(cfg(match_observation_support=True))
    old.load_state_dict(encoder.state_dict())
    assert result(old, raw, current, mask).residual.abs().mean() > 1e-3
    moved = torch.roll(current, 3, -1)
    assert result(encoder, raw, moved, mask).residual.abs().mean() > 0.05


def test_unit_gain_agrees_with_unexplained_observation():
    encoder = N04Encoder(cfg(illumination_explanation=True, match_observation_support=True))
    old = N04Encoder(cfg(match_observation_support=True))
    old.load_state_dict(encoder.state_dict())
    raw = torch.rand(1, 3, 16, 16) * 0.4 + 0.1
    mask = torch.ones(1, 1, 16, 16)
    a, b = result(encoder, raw, raw, mask), result(old, raw, raw, mask)
    torch.testing.assert_close(a.g_update, b.g_update, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_sensor_gain_preserves_cfa_phases(pattern):
    from dataclasses import replace

    c = replace(
        cfg(illumination_explanation=True, match_observation_support=True),
        input_mode="bayer_cfa",
        channels=1,
        cfa_pattern=pattern,
    )
    encoder = N04Encoder(c)
    samples = torch.rand(1, 1, 16, 16) * 0.3 + 0.1
    planes, _ = linear_measurement_planes(samples, torch.ones_like(samples), "bayer_cfa")
    gain = encoder.decomposer.estimate_sensor_gain(
        planes * 1.4,
        planes,
        torch.ones(1, 1, 16, 16),
        torch.zeros(1, 1, 16, 16),
        torch.zeros(1, 1, 16, 16),
    )
    torch.testing.assert_close(
        encoder.predict_observation(planes * gain),
        encoder.encode_measurement(samples * 1.4),
        atol=2e-4,
        rtol=1e-4,
    )


def test_imaging_design_is_checkpoint_guarded(tmp_path):
    encoder = N04Encoder(cfg(match_observation_support=True))
    path = tmp_path / "old.pt"
    torch.save(
        dict(
            encoder=encoder.state_dict(),
            model_config=encoder.config.model.as_dict(),
            mechanism_revision="n04-independent-event-scores-v1",
        ),
        path,
    )
    new = N04Encoder(cfg(match_observation_support=True, illumination_explanation=True))
    with pytest.raises(ValueError, match="illumination_explanation"):
        load_checkpoint(path, encoder=new)

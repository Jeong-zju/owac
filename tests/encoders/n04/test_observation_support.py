"""Current and predicted measurements use the same observable sensor support."""

import pytest
import torch

from owac.frameworks.n04.model import N04Encoder

from .test_observation_reference import cfg


@pytest.mark.parametrize("missing", ["saturation", "occlusion"])
def test_supported_static_prediction_has_no_mask_induced_innovation(missing):
    torch.manual_seed(11)
    encoder = N04Encoder(cfg(match_observation_support=True))
    raw = torch.rand(2, 3, 16, 16) * 0.6 + 0.1
    mask = torch.ones(2, 1, 16, 16)
    mask[..., 5:11, 4:9] = 0
    measured = encoder.encode_measurement(raw * mask)
    state = encoder.init_state(2)["head"]
    _, innovation = encoder.correct_after_observe(
        state,
        torch.zeros_like(measured),
        measured,
        torch.ones_like(mask),
        torch.ones_like(mask) * 0.05,
        saturation=1 - mask if missing == "saturation" else None,
        measurement_valid=mask if missing == "occlusion" else None,
        linear_current=raw,
        linear_predicted=raw,
        linear_validity=mask,
        current_linear_support=mask,
    )
    assert innovation.residual.abs().max() < 1e-6
    # The input measurement has not been rescaled or overwritten.
    torch.testing.assert_close(encoder.encode_measurement(raw * mask), measured)


def test_missing_current_support_is_not_silently_assumed():
    encoder = N04Encoder(cfg(match_observation_support=True))
    state = encoder.init_state(1)["head"]
    raw = torch.rand(1, 3, 16, 16)
    feature = encoder.encode_measurement(raw)
    with pytest.raises(ValueError, match="current sensor support"):
        encoder.correct_after_observe(
            state,
            feature,
            feature,
            torch.ones(1, 1, 16, 16),
            torch.zeros(1, 1, 16, 16),
            linear_predicted=raw,
        )

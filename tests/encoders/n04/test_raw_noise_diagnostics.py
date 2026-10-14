"""Sensor variance units and paired diagnostic input integrity."""

import torch

from owac.evaluation.n04_raw_diagnostics import bilinear_variance, frame_pair, residual_score
from owac.frameworks.n04.geometry import GridSpec, backward_grid_from_translation


def test_bilinear_noise_uses_squared_weights():
    spec = GridSpec(8, 8)
    grid = backward_grid_from_translation(spec, 0.5, 0.5)
    variance = torch.ones(1, 3, 8, 8)
    propagated = bilinear_variance(variance, grid)
    assert torch.allclose(propagated[:, :, 1:, 1:], torch.full_like(propagated[:, :, 1:, 1:], 0.25))
    assert torch.equal(
        bilinear_variance(variance, spec.identity_grid(1, torch.device("cpu"), torch.float32)),
        variance,
    )


def test_residual_noise_scale_is_invariant_to_common_gain():
    current, prediction = torch.tensor([[[[0.4]]]]), torch.tensor([[[[0.2]]]])
    v = torch.full_like(current, 0.01)
    score = residual_score(current, prediction, current_variance=v, prediction_variance=v)
    scaled = residual_score(
        3 * current, 3 * prediction, current_variance=9 * v, prediction_variance=9 * v
    )
    assert torch.allclose(score, scaled)


def test_pair_measurements_repeat_and_camera_context_excludes_object_motion():
    first = frame_pair(951001, object_on=True)
    second = frame_pair(951001, object_on=True)
    assert all(torch.equal(a.measured, b.measured) for a, b in zip(first, second, strict=True))
    assert all(not f.proprio.any() for f in first)
    assert first[1].true_object_translation.abs().sum() > 0
    assert not torch.equal(first[0].oracle_clean, first[1].oracle_clean)

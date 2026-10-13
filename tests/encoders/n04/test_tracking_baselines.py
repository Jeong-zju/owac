"""Causality and missing-observation behavior of the matched tracking experiment."""

import torch

from owac.baselines.n04_tracking import RKNPositionTracker, hold_last
from owac.evaluation.n04_tracking import InstantaneousMeasurement, frame_sequence
from owac.frameworks.n04.model import MeasurementStem


def positions():
    t = torch.arange(16, dtype=torch.float32)
    z = torch.stack((20 + 0.5 * t, 30 - 0.2 * t), -1)[:, None]
    available = torch.ones(16, 1, dtype=torch.bool)
    available[8:12] = False
    return z, available


def test_all_trackers_ignore_poisoned_missing_measurements():
    z, valid = positions()
    poisoned = z.clone()
    poisoned[~valid] = float("nan")
    variance = torch.tensor([0.05, 0.05])
    model = RKNPositionTracker(variance).eval()
    methods = (
        hold_last,
        model,
    )
    for method in methods:
        assert torch.equal(method(z, valid).position, method(poisoned, valid).position)


def test_rkn_outputs_current_posterior_and_is_causal():
    z, valid = positions()
    model = RKNPositionTracker(torch.tensor([0.05, 0.05])).eval()
    with torch.no_grad():
        output = model(z, valid)
        assert torch.equal(output.position[:12], model(z[:12], valid[:12]).position)
        assert torch.allclose(output.position[0], z[0], atol=1e-6)
        assert torch.equal(output.position[8:12], output.prior[8:12])
    # Missing labels are only a loss target; hidden-frame gradients reach dynamics.
    loss = (model(z, valid).position[8:12] - z[8:12]).square().mean()
    loss.backward()
    assert model.cell._tm_12_basis.grad.abs().sum() > 0


def test_instantaneous_head_has_no_cross_frame_or_cross_sample_state():
    torch.manual_seed(19)
    stem = MeasurementStem(channels=3, width=4)
    measurement = InstantaneousMeasurement(stem).eval()
    x = torch.rand(3, 3, 16, 16)
    one = measurement(x[:1])
    measurement(torch.rand_like(x))
    assert torch.allclose(one, measurement(x)[:1], atol=1e-6)
    assert not any(p.requires_grad for p in measurement.stem.parameters())


def test_new_scene_paths_have_full_target_support_and_no_pose_object_leak():
    seq = list(frame_sequence(910000, "mixed_cv", batch=2))
    assert all((f.object_mask.sum((1, 2, 3)) > 100).all() for f in seq)
    assert all(f.proprio.shape[-1] == 3 for f in seq)
    assert all(torch.equal(f.proprio[:, :2], f.true_camera_translation) for f in seq)

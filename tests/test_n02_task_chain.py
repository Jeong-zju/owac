"""Check causal inputs, full-horizon masks, and frozen/joint gradient ownership."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owac.frameworks.n02_full.task_chain import (  # noqa: E402
    ResidualTokenAdapter,
    SmallActionHead,
    action_metrics,
    masked_action_loss,
    split_groups,
    visual_tokens,
)


def test_adapter_identity_keeps_spatial_order_and_learns():
    torch.manual_seed(1)
    adapter = ResidualTokenAdapter(width=8, bottleneck=4)
    x = torch.randn(2, 12, 8, requires_grad=True)
    torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
    adapter(x).square().mean().backward()
    assert adapter.up.weight.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0


def test_padding_cannot_change_loss_or_metrics():
    predicted = torch.zeros(2, 50, 32, requires_grad=True)
    target = torch.ones_like(predicted)
    valid = torch.zeros(2, 50, dtype=torch.bool)
    valid[:, :10] = True
    target[:, 10:] = float("nan")
    loss = masked_action_loss(predicted, target, valid)
    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[:, 10:].abs().sum() == 0
    p, t = predicted.detach().numpy()[..., :14], target.numpy()[..., :14]
    m = action_metrics(p, t, valid.numpy(), np.ones(14))
    assert m["valid_action_steps"] == 20
    assert m["normalized_control_mse"] == 1
    assert m["joint_rms_deg"] == pytest.approx(180 / np.pi)


def test_all_padding_rejected_and_full_horizon_retained():
    head = SmallActionHead(width=8, tokens=12)
    out = head(torch.randn(2, 12, 8), torch.randn(2, 14))
    assert out.shape == (2, 50, 32)
    with pytest.raises(ValueError, match="nonempty"):
        masked_action_loss(out, out.detach(), torch.zeros(2, 50, dtype=torch.bool))


def test_group_holdout_rejects_adjacent_frame_leakage():
    rows = [dict(group=s, split=s) for s in ("train", "val", "test")]
    split_groups(rows)
    rows.append(dict(group="train", split="test"))
    with pytest.raises(ValueError, match="leaks"):
        split_groups(rows)


def test_small_head_joint_gradient_is_an_explicit_choice():
    producer = torch.nn.Linear(4, 8)
    head = SmallActionHead(width=8, tokens=2)
    x = torch.randn(3, 2, 4)
    state = torch.randn(3, 14)
    producer.requires_grad_(False)
    head(producer(x), state).square().mean().backward()
    assert producer.weight.grad is None
    producer.requires_grad_(True)
    head(producer(x), state).square().mean().backward()
    assert producer.weight.grad.abs().sum() > 0


def test_camera_and_pixel_coordinates_stay_explicit():
    x = torch.randn(2, 3 * 256, 8)
    out = visual_tokens(x, ("head", "left", "right"))
    assert out.values is x
    assert out.camera_ids == ("head", "left", "right")
    torch.testing.assert_close(out.positions[0, 0], torch.tensor([0.5 / 16, 0.5 / 16]))
    torch.testing.assert_close(out.positions[0, 255], torch.tensor([15.5 / 16, 15.5 / 16]))
    torch.testing.assert_close(out.positions[:, :256], out.positions[:, 256:512])

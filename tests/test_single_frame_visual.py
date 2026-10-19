"""Spatial order, measurement sensitivity and learning in the simple replacement."""

import hashlib

import pytest
import torch
from torch import nn

from owac.baselines.single_frame_visual import SingleFrameVisualEncoder, make_convnext_base


class SpatialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = nn.Conv2d(3, 4, 1)

    def forward_features(self, frames):
        return [self.channels(frames)]


def encoder():
    torch.manual_seed(7)
    return SingleFrameVisualEncoder(
        SpatialBackbone(), camera_ids=("external", "wrist"), feature_dim=4, token_dim=8, grid=3
    )


def test_camera_order_and_independent_measurements():
    model = encoder().eval()
    frames = torch.randn(2, 2, 3, 6, 6)
    output = model(frames)
    assert output.values.shape == (2, 18, 8)
    assert output.camera_ids == ("external", "wrist")
    assert output.mask.all()
    assert output.positions.shape == (2, 18, 2)
    assert torch.equal(output.positions[:, :9], output.positions[:, 9:])
    changed = frames.clone()
    changed[:, 1] += torch.randn_like(changed[:, 1])
    other = model(changed).values
    torch.testing.assert_close(output.values[:, :9], other[:, :9], rtol=0, atol=0)
    assert not torch.equal(output.values[:, 9:], other[:, 9:])
    swapped = model(frames.flip(1)).values.reshape(2, 2, 9, 8).flip(1).reshape(2, 18, 8)
    torch.testing.assert_close(output.values, swapped, rtol=0, atol=0)


def test_gradients_and_reload_preserve_real_output():
    model = encoder()
    frames = torch.randn(2, 2, 3, 6, 6, requires_grad=True)
    initial = model(frames).values.detach()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(frames).values.square().mean().backward()
    assert frames.grad.abs().sum() > 0
    assert model.backbone.channels.weight.grad.abs().sum() > 0
    assert model.project[1].weight.grad.abs().sum() > 0
    optimizer.step()
    updated = model(frames).values.detach()
    assert not torch.equal(initial, updated)
    restored = encoder()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(frames).values, updated, rtol=0, atol=0)


def test_simple_tokens_reach_shared_bridge_without_an_extra_projection():
    from owac.backends.rawvla_features_n02 import build_stub_bridge

    model = encoder()
    bridge = build_stub_bridge(
        width=8,
        token_dim=8,
        tokens_per_camera=9,
        frame_pixels=3,
        cameras=model.camera_ids,
        with_frame=False,
    )["bridge"]
    tokens = model(torch.randn(2, 2, 3, 6, 6))
    language = torch.ones(2, 4, dtype=torch.long)
    prefix = bridge.embed_prefix_from_visual_tokens(tokens, language, language.bool())
    torch.testing.assert_close(prefix.embeds[:, :18], tokens.values, rtol=0, atol=0)
    assert prefix.pad_masks.all()
    assert not prefix.att_masks.any()
    prefix.embeds[:, :18].square().mean().backward()
    assert model.backbone.channels.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["history", "camera_count", "nan", "integer", "empty"])
def test_reject_implicit_history_or_invalid_planes(kind):
    frames = torch.ones(1, 2, 3, 6, 6)
    if kind == "history":
        frames = frames.unsqueeze(1)
    elif kind == "camera_count":
        frames = frames[:, :1]
    elif kind == "nan":
        frames[0, 0, 0, 0, 0] = float("nan")
    elif kind == "integer":
        frames = frames.byte()
    else:
        frames = frames[:0]
    with pytest.raises(ValueError):
        encoder()(frames)


def test_pretrained_hash_mismatch_cannot_be_random_initialization(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not pretrained weights")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        make_convnext_base(checkpoint, sha256="0" * 64, camera_ids=("external",))

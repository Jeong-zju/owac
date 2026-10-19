"""Single-frame spatial visual replacement, independent of N01/N02/N04 mechanisms.

Inputs are explicitly normalized RGB or RAW planes, never a history window.
The shared backbone emits one spatial map; fixed bilinear sampling, LayerNorm
and a channel projection produce ordered camera tokens for the N02 bridge.
This module neither constructs a policy nor starts training or downloads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from owac.core.encoder_contract_n02 import VisualTokens


class SingleFrameVisualEncoder(nn.Module):
    """A shared backbone and one channel projection, with no temporal state."""

    def __init__(self, backbone, *, camera_ids, feature_dim=1024, token_dim=2048, grid=16):
        super().__init__()
        cameras = tuple(camera_ids)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("camera_ids must be nonempty and unique")
        if min(feature_dim, token_dim, grid) < 1:
            raise ValueError("positive feature, token and grid dimensions required")
        self.backbone = backbone
        self.camera_ids = cameras
        self.feature_dim, self.grid = feature_dim, grid
        self.project = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, token_dim))

    def forward(self, frames):
        """Encode [B,camera,channel,H,W]; callers declare preprocessing separately."""
        if frames.ndim != 5 or frames.shape[1] != len(self.camera_ids):
            raise ValueError("expected one frame per camera: [B,camera,channel,H,W]")
        if frames.shape[0] < 1 or not frames.is_floating_point():
            raise ValueError("a nonempty batch of normalized floating-point planes is required")
        if not torch.isfinite(frames).all():
            raise ValueError("nonfinite measurement planes are not supported by this baseline")
        batch, cameras = frames.shape[:2]
        maps = self.backbone.forward_features(frames.flatten(0, 1))
        if len(maps) != 1:
            raise ValueError("backbone must return exactly one spatial feature map")
        spatial = maps[0]
        if spatial.ndim != 4 or spatial.shape[:2] != (batch * cameras, self.feature_dim):
            raise ValueError("backbone feature layout mismatch")
        # Resampling does not add spatial information or learn camera mixing.
        spatial = F.interpolate(
            spatial, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        values = self.project(spatial.flatten(2).transpose(1, 2))
        values = values.reshape(batch, cameras * self.grid**2, -1)
        axis = (
            torch.arange(self.grid, device=values.device, dtype=torch.float32) + 0.5
        ) / self.grid
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((yy, xx), -1).reshape(1, self.grid**2, 2).repeat(1, cameras, 1)
        return VisualTokens(
            values=values,
            mask=torch.ones(values.shape[:2], dtype=torch.bool, device=values.device),
            camera_ids=self.camera_ids,
            positions=positions.expand(batch, -1, -1),
        )


def make_convnext_base(checkpoint: Path, *, sha256: str, camera_ids):
    """Require pinned public weights; never silently fall back to random weights.

    Both three-channel arms load the same backbone. RGB pretraining is disclosed
    initialization, not RGB teacher supervision or an RGB image in the RAW path.
    Native four-plane CFA adaptation is outside this initial pseudo-RAW study.
    """
    from owac.frameworks.n02_full.components.convnextv2 import RawConvNeXtV2

    checkpoint = Path(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if len(sha256) != 64 or digest.hexdigest() != sha256:
        raise ValueError("pretrained backbone SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = dict(payload.get("model", payload))
    # Public ImageNet classifiers are unused; all backbone keys must still match.
    state.pop("head.weight", None)
    state.pop("head.bias", None)
    backbone = RawConvNeXtV2(out_indices=(3,), init_weights=False)
    backbone.load_state_dict(state, strict=True)
    # The pooled classification norm is not used by forward_features.
    backbone.norm.requires_grad_(False)
    return SingleFrameVisualEncoder(backbone, camera_ids=camera_ids)

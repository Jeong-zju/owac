"""Causal RAW feature baselines reusing the published DBSR backbone and fusion.

No RGB alignment, decoder, pretrained weights, or N01 support decisions are used.
Default dimensions match the upstream feature encoder (64 / 9 / 512). Only the
validity/causality adapter is new. Feature attention is not a noise posterior.
The mask-aware forward below adapts Huawei's CC-BY-NC-SA-4.0 WeightedSum; its
source, attribution, license and unchanged implementation live in _vendor/dbsr.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from owac.core.encoder_contract import EncoderOutput
from owac.frameworks.n01 import state as measurement_state
from owac.frameworks.n01.config import N01Config

from ._vendor.dbsr.encoders import ResEncoderWarpAlignnet
from ._vendor.dbsr.merging import WeightedSum

CFA_PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
FUSIONS = ("current", "mean", "dbsr_reference", "dbsr_consensus")


@dataclass(frozen=True)
class BurstFeatureConfig:
    fusion: str = "dbsr_reference"
    stem_width: int = 64
    residual_blocks: int = 9
    feature_width: int = 512
    projection_width: int = 32
    offset_width: int = 64
    max_history: int = 8

    def __post_init__(self):
        if self.fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {self.fusion!r}")
        for name, value in asdict(self).items():
            if name != "fusion" and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return asdict(self)


class RawBurstBackbone(nn.Module):
    """Use the upstream encoder's complete feature branch, omitting RGB/PWC.

    Packed input has canonical R,G1,G2,B channel order and the recorded RGGB
    phase coordinates. Repacking other CFA patterns is the caller's job; these
    four measurements are never averaged into a three-channel image.
    """

    def __init__(self, config):
        super().__init__()
        self.upstream = ResEncoderWarpAlignnet(
            config.stem_width,
            config.residual_blocks,
            config.feature_width,
            alignment_net=None,
            train_alignmentnet=False,
        )

    def forward(self, values):
        # These are the exact three feature operations in upstream.forward.
        return self.upstream.out_layer(self.upstream.res_layers(self.upstream.init_layer(values)))


class MaskedDbsrFusion(WeightedSum):
    """Preserve upstream modules; exclude invalid sources before base and softmax.

    All-valid results and gradients match WeightedSum with offset_modulo=1,
    ref_offset_noise=0, softmax=True. If the reference is missing locally, use
    the available-frame consensus. The all-invalid result and weights are zero.
    Convolutional context is not a claim of N01 graph-constrained dependence.
    """

    def __init__(self, config):
        super().__init__(
            config.feature_width,
            config.projection_width,
            config.offset_width,
            offset_modulo=1.0,
            use_base_frame=config.fusion == "dbsr_reference",
        )

    def forward(self, features, valid, offsets_xy):
        b, t, c, h, w = features.shape
        features = features.masked_fill(~valid, 0)
        projected = self.feat_project_layer(features.reshape(b * t, c, h, w))
        projected = projected.reshape(b, t, -1, h, w).masked_fill(~valid, 0)
        consensus = projected.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        base = torch.where(valid[:, :1], projected[:, :1], consensus)
        if not self.use_base_frame:
            base = consensus
        difference = (projected - base).masked_fill(~valid, 0)
        base = base.expand(-1, t, -1, -1, -1)
        offset_features = self.offset_feat_extractor(
            offsets_xy.remainder(self.offset_modulo).reshape(b * t, 2, h, w)
        )
        weight_input = torch.cat(
            [base.reshape(b * t, -1, h, w), difference.reshape(b * t, -1, h, w), offset_features],
            dim=1,
        )
        logits = self.weight_predictor(weight_input).reshape(b, t, c, h, w)
        available = valid.any(1, keepdim=True)
        logits = logits.masked_fill(~valid, -torch.inf)
        logits = torch.where(available, logits, torch.zeros_like(logits))
        weights = logits.softmax(1).masked_fill(~valid, 0)
        return (features * weights).sum(1), weights


def align_features(features, valid, offsets_xy):
    """Pull from (column+dx,row+dy) in packed-grid cells; never wrap boundaries.

    offsets_xy is [B,T,2,H,W], reference/newest slot is zero. Every positive
    bilinear tap must be valid. Explicit align_corners=False matches pixel
    centers and avoids the legacy upstream warp's implicit version default.
    """
    b, t, c, h, w = features.shape
    if offsets_xy is None:
        offsets_xy = features.new_zeros(b, t, 2, h, w)
        return features.masked_fill(~valid, 0), valid, offsets_xy
    if offsets_xy.shape != (b, t, 2, h, w) or not torch.isfinite(offsets_xy).all():
        raise ValueError("finite offsets_xy [B,T,2,H,W] required")
    if bool((offsets_xy[:, 0] != 0).any()):
        raise ValueError("newest/reference source must have zero offset")
    rows = torch.arange(h, device=features.device, dtype=features.dtype)[None, None, :, None]
    cols = torch.arange(w, device=features.device, dtype=features.dtype)[None, None, None, :]
    x, y = cols + offsets_xy[:, :, 0], rows + offsets_xy[:, :, 1]
    grid = torch.stack((2 * (x + 0.5) / w - 1, 2 * (y + 0.5) / h - 1), dim=-1)
    grid = grid.reshape(b * t, h, w, 2)
    aligned = F.grid_sample(
        features.masked_fill(~valid, 0).reshape(b * t, c, h, w), grid, align_corners=False
    ).reshape(b, t, c, h, w)
    coverage = F.grid_sample(
        valid.float().reshape(b * t, 1, h, w), grid, align_corners=False
    ).reshape(b, t, 1, h, w)
    inside = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)).unsqueeze(2)
    usable = (coverage >= 1 - 1e-6) & inside
    return aligned.masked_fill(~usable, 0), usable, offsets_xy


class BurstFeatureEncoder(nn.Module):
    """Independent baseline using N01's unchanged unique-measurement buffer.

    `step` recomputes from raw sources and never caches trainable features across
    optimizer updates. `fuse_encoded` permits a shared per-sequence feature bank
    within ONE optimizer step. No support graph is updated or claimed here.
    Call reset_state explicitly at episode boundaries; cameras have separate states.
    """

    def __init__(self, config: BurstFeatureConfig | None = None):
        super().__init__()
        self.config = config or BurstFeatureConfig()
        self.backbone = RawBurstBackbone(self.config)
        self.fusion = (
            MaskedDbsrFusion(self.config) if self.config.fusion.startswith("dbsr_") else None
        )
        self.buffer_config = N01Config.tiny(
            input_mode="bayer_cfa",
            channels=4,
            planes=4,
            spatial_radii=(0,),
            time_depths=(self.config.max_history,),
            max_history=self.config.max_history,
            aux_measurement_prediction=False,
        )

    def encode_frames(self, values, valid):
        if values.ndim != 5 or values.shape[2] != 4 or values.shape[1] < 1:
            raise ValueError("values must be [B,T,4,H,W] packed RGGB RAW")
        if valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean mask matching the RAW measurements")
        valid = valid & torch.isfinite(values)
        # Conservative center availability: all four CFA phases must be present.
        centers = valid.all(2, keepdim=True)
        values = values.masked_fill(~valid, 0).float()
        b, t, c, h, w = values.shape
        encoded = self.backbone(values.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w)
        return encoded.masked_fill(~centers, 0), centers

    def fuse_encoded(self, features, valid, offsets_xy=None):
        if features.ndim != 5 or features.shape[2] != self.config.feature_width:
            raise ValueError("wrong encoded feature shape")
        if valid.shape != (*features.shape[:2], 1, *features.shape[-2:]):
            raise ValueError("feature validity must be [B,T,1,H,W]")
        if features.shape[1] > self.config.max_history:
            raise ValueError("history exceeds configured unique-source capacity")
        aligned, valid, offsets_xy = align_features(features, valid, offsets_xy)
        if self.config.fusion == "current":
            weights = torch.zeros_like(valid, dtype=features.dtype)
            weights[:, 0] = valid[:, 0]
            result = aligned[:, 0]
        elif self.config.fusion == "mean":
            weights = valid.to(features.dtype) / valid.sum(1, keepdim=True).clamp_min(1)
            result = (aligned * weights).sum(1)
        else:
            result, weights = self.fusion(aligned, valid, offsets_xy)
        available = valid[:, 0] if self.config.fusion == "current" else valid.any(1)
        return result.masked_fill(~available, 0), available, weights

    def forward(self, values, valid, offsets_xy=None):
        """Encode a caller-provided causal window in NEWEST-FIRST order."""
        features, centers = self.encode_frames(values, valid)
        return self.fuse_encoded(features, centers, offsets_xy)

    def init_state(self, batch_size, camera_spec, device="cpu"):
        return measurement_state.init_state(self.buffer_config, batch_size, camera_spec, device)

    @staticmethod
    def reset_state(state, batch_mask):
        return measurement_state.reset(state, batch_mask)

    @staticmethod
    def detach_state(state):
        return measurement_state.detach(state)

    def step(self, frame, *, state, offsets_xy=None):
        if frame.episode_ids is not None:
            raise ValueError(
                "reset_state explicitly at episode boundaries; episode_ids unsupported"
            )
        if frame.sample_xy is None:
            raise ValueError("explicit canonical RGGB sample_xy phases required")
        phases = torch.as_tensor(frame.sample_xy, device=frame.values.device)
        expected = torch.tensor(CFA_PHASES, device=frame.values.device)
        if phases.shape != expected.shape or not torch.equal(phases, expected):
            raise ValueError("explicit canonical RGGB sample_xy phases required")
        updated = measurement_state.push(state, frame)
        count = max(1, int(updated.filled.max()))
        values = updated.values[:count].transpose(0, 1)
        valid = updated.valid[:count].transpose(0, 1)
        valid = valid & updated.observed[:count].T[:, :, None, None, None]
        valid = valid & torch.isfinite(updated.variance[:count].transpose(0, 1))
        representation, available, weights = self(values, valid, offsets_xy)
        updated.last_representation = representation
        return EncoderOutput(
            representation=representation,
            state=updated,
            validity=available,
            diagnostics={
                "source_ids": updated.frame_ids[:count].T.detach(),
                "source_timestamps": updated.timestamps[:count].T.detach(),
                "fusion_weights": weights.detach(),
                "noise_variance_used_as_fusion_weight": False,
                "support_graph_applied": False,
            },
            cost={"state_bytes": measurement_state.state_bytes(updated)},
        )

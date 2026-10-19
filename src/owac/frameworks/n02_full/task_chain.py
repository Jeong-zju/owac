"""Isolated spatial-only action diagnostic; no measurement decoder or hidden labels.

Both readouts receive ordered camera/16x16 spatial tokens and current measured
joint state. These modules do not change the existing N02 production policy.
"""

import hashlib

import numpy as np
import torch
from torch import nn

from owac.core.encoder_contract_n02 import VisualTokens


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


class SpatialTaskEncoder(nn.Module):
    """Reuse the actual RAW encoder and multiscale fusion, with no context offset."""

    def __init__(self, encoder, readout):
        super().__init__()
        self.encoder = encoder
        self.readout = readout
        self.cameras = tuple(encoder.camera_ids)

    def forward(self, observation):
        spatial = self.encoder(observation)
        if tuple(spatial.camera_ids) != self.cameras:
            raise ValueError("camera order changed")
        return self.readout.spatial_output_tokens(spatial)

    def set_joint_training(self, enabled, blocks=6):
        self.requires_grad_(False)
        if enabled:
            if not 1 <= blocks <= len(self.encoder.backbone.stages[2]):
                raise ValueError("invalid joint encoder scope")
            self.encoder.backbone.stages[2][-blocks:].requires_grad_(True)
            self.encoder.proj.requires_grad_(True)
            # Only parameters actually used by spatial_output_tokens participate.
            self.readout.spatial_skip.requires_grad_(True)
            for name in ("scale_logits", "local_gain", "global_gain"):
                value = getattr(self.readout, name, None)
                if value is not None:
                    value.requires_grad_(True)
        self.eval()  # Fixed stochastic/drop-path behaviour in both phases.


class ResidualTokenAdapter(nn.Module):
    """Per-position trainable channel adapter, identity at initialization."""

    def __init__(self, width=2048, bottleneck=128):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        return tokens + self.up(torch.nn.functional.gelu(self.down(self.norm(tokens))))


class SmallActionHead(nn.Module):
    """Small full-horizon head; flattening preserves camera and grid position."""

    def __init__(self, *, width=2048, tokens=768, horizon=50, action_dim=32, state_dim=14):
        super().__init__()
        self.tokens, self.width = tokens, width
        self.horizon, self.action_dim = horizon, action_dim
        self.norm = nn.LayerNorm(width)
        self.channels = nn.Linear(width, 8)
        self.predict = nn.Sequential(
            nn.Linear(tokens * 8 + state_dim, 256),
            nn.GELU(),
            nn.Linear(256, horizon * action_dim),
        )

    def forward(self, tokens, state):
        if tuple(tokens.shape[1:]) != (self.tokens, self.width):
            raise ValueError("small head token layout mismatch")
        spatial = self.channels(self.norm(tokens.float())).flatten(1)
        return self.predict(torch.cat((spatial, state.float()), -1)).reshape(
            -1, self.horizon, self.action_dim
        )


def visual_tokens(values, cameras):
    count = len(cameras)
    if values.ndim != 3 or values.shape[1] != count * 256:
        raise ValueError("expected 256 ordered spatial tokens per camera")
    grid = torch.arange(16, device=values.device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(grid / 16, grid / 16, indexing="ij")
    positions = torch.stack((yy, xx), -1).reshape(1, 256, 2).repeat(1, count, 1)
    return VisualTokens(
        values=values,
        mask=torch.ones(values.shape[:2], dtype=torch.bool, device=values.device),
        camera_ids=tuple(cameras),
        positions=positions.expand(values.shape[0], -1, -1),
    )


def masked_action_loss(predicted, target, valid):
    valid = torch.as_tensor(valid, device=predicted.device, dtype=torch.bool)
    if predicted.shape != target.shape or valid.shape != predicted.shape[:2] or not valid.any():
        raise ValueError("nonempty matched action target/mask required")
    # Mask before arithmetic; padded NaNs must not poison a valid loss.
    difference = (predicted - target).masked_fill(~valid.unsqueeze(-1), 0)
    return difference.square().sum() / (valid.sum() * predicted.shape[-1])


def action_metrics(prediction, target, valid, action_std):
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or prediction.shape[-1] != 14:
        raise ValueError("expected matching B,50,14 control actions")
    if valid.shape != prediction.shape[:2] or not valid.any():
        raise ValueError("empty or mismatched control mask")
    error = (prediction - target)[valid]
    physical = error * np.asarray(action_std)
    joints = [*range(6), *range(7, 13)]
    output = {
        "normalized_control_mse": float(np.square(error).mean()),
        "joint_rms_deg": float(np.square(physical[:, joints]).mean() ** 0.5 * 180 / np.pi),
        "gripper_mae": float(np.abs(physical[:, [6, 13]]).mean()),
        "gripper_max_error": float(np.abs(physical[:, [6, 13]]).max()),
        "valid_action_steps": int(valid.sum()),
        "windows": len(valid),
    }
    for label, cutoff in (("first", 1), ("first10", 10)):
        mask = valid.copy()
        mask[:, cutoff:] = False
        e = ((prediction - target) * np.asarray(action_std))[mask]
        output[label + "_joint_rms_deg"] = float(
            np.square(e[:, joints]).mean() ** 0.5 * 180 / np.pi
        )
        output[label + "_joint_max_deg"] = float(np.abs(e[:, joints]).max() * 180 / np.pi)
        output[label + "_gripper_mae"] = float(np.abs(e[:, [6, 13]]).mean())
    return output


def split_groups(records):
    groups = {}
    result = {name: [] for name in ("train", "val", "test")}
    for i, row in enumerate(records):
        group, split = row["group"], row["split"]
        if split not in result or groups.setdefault(group, split) != split:
            raise ValueError("group leaks across partitions or split is unknown")
        result[split].append(i)
    if any(not indices for indices in result.values()):
        raise ValueError("train, val and test must all be nonempty")
    return result

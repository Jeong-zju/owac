"""Causal measurement-only adapters for the same retrained official embedding.

These are declared preprocessing controls, not N02 mechanism changes. Alignment
uses the first observed frame, never a hidden scene center. Temporal innovation
subtracts that same measured reference, including its noise. No predicted/RGB
image, cue mask, target label or unseen observation enters either operation.
"""

from __future__ import annotations

import torch

from owac.baselines.n02_measurement import RetrainedMeasurementScorer

MODES = ("center", "innovation", "center_innovation")


def adapt_raw(raw, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if raw.ndim != 5 or raw.shape[2:] != (3, 64, 64):
        raise ValueError("measurement adapter expects [B,T,3,64,64]")
    value = raw
    if "center" in mode:
        # Public threshold 5 * calibrated noise sigma=.01; the visible arc, not
        # the later weak cue, dominates this first-frame measurement statistic.
        weight = (raw[:, 0].mean(1) - 0.05).clamp_min(0)
        total = weight.sum((1, 2))
        y = torch.arange(64, device=raw.device, dtype=raw.dtype)[None, :, None]
        x = torch.arange(64, device=raw.device, dtype=raw.dtype)[None, None, :]
        safe = total.clamp_min(1e-12)
        cy, cx = (weight * y).sum((1, 2)) / safe, (weight * x).sum((1, 2)) / safe
        cy = torch.where(total > 0, cy, torch.full_like(cy, 32))
        cx = torch.where(total > 0, cx, torch.full_like(cx, 32))
        offsets = (32 - cy.round().long(), 32 - cx.round().long())
        # Integer circular permutation preserves every measured value; no crop,
        # interpolation or hidden scene geometry. This control is pseudo-RAW only.
        value = torch.stack(
            [
                torch.roll(sequence, (int(dy), int(dx)), dims=(-2, -1))
                for sequence, dy, dx in zip(raw, *offsets, strict=True)
            ]
        )
    if "innovation" in mode:
        value = value - value[:, :1]
    return value


class AdaptedMeasurementScorer(RetrainedMeasurementScorer):
    def __init__(self, mean, std, mode):
        super().__init__(mean, std)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode

    def forward(self, raw):
        return super().forward(adapt_raw(raw, self.mode))

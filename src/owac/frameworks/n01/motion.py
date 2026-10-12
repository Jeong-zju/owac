"""N01 causal correspondence and alignment.

``CausalCorrespondence`` estimates the displacement between two *already
arrived* low-resolution RAW feature maps plus a match residual and the valid
overlap region.  It is a small local correlation window with an optional small
convolutional head, following the RAFT correlation / recurrent-update idea; a
full RGB-pretrained RAFT is only ever an independent diagnostic reference and is
not a required teacher for the main RAW path (see
``third_party``/reference notes in the experiment record).

The motion estimate here serves integration only.  It does not decompose ego
motion from external change (that is N04): the module reports a displacement, a
matching residual and an overlap mask, nothing more.

Three profiles share this one module and the same training data:
``none`` disables alignment, ``fixed`` uses the plain correlation estimate, and
``candidate`` may additionally use the sub-pixel residual so that the
support-domain restructuring can consume the per-frame matching quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N01Config

__all__ = ["AlignmentMode", "MotionField", "CausalCorrespondence", "MotionChain"]


class AlignmentMode(StrEnum):
    NONE = "none"
    FIXED = "fixed"
    CANDIDATE = "candidate"


@dataclass
class MotionField:
    """Displacement of ``other`` relative to ``reference`` in pixels (dy, dx)."""

    shift: torch.Tensor
    residual: torch.Tensor
    overlap: torch.Tensor
    valid: torch.Tensor
    fractional: torch.Tensor | None = None
    scores: torch.Tensor | None = None

    def select(self, mask: torch.Tensor) -> MotionField:
        """Return a copy where batch members outside ``mask`` carry no motion."""
        keep = mask.to(self.shift.device)
        shift = torch.where(keep[:, None], self.shift, torch.zeros_like(self.shift))
        residual = torch.where(keep, self.residual, torch.ones_like(self.residual))
        overlap = torch.where(keep, self.overlap, torch.zeros_like(self.overlap))
        valid = self.valid & keep
        fractional = None
        if self.fractional is not None:
            fractional = torch.where(
                keep[:, None], self.fractional, torch.zeros_like(self.fractional)
            )
        return MotionField(
            shift=shift,
            residual=residual,
            overlap=overlap,
            valid=valid,
            fractional=fractional,
            scores=self.scores,
        )

    def detach(self) -> MotionField:
        return MotionField(
            shift=self.shift.detach(),
            residual=self.residual.detach(),
            overlap=self.overlap.detach(),
            valid=self.valid.detach(),
            fractional=None if self.fractional is None else self.fractional.detach(),
            scores=None if self.scores is None else self.scores.detach(),
        )


def _gather_flat(values: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    """Gather with an int64 flat spatial index of shape [B,H,W].

    The spatial grid is flattened first, because ``torch.gather`` indexes one
    dimension at a time and a row*W+col index is only valid on the flattened
    H*W axis.
    """
    batch, channels, height, width = values.shape
    flat_values = values.reshape(batch, channels, height * width)
    index = flat_index.reshape(batch, 1, height * width).expand(batch, channels, height * width)
    return torch.gather(flat_values, 2, index).reshape(batch, channels, height, width)


class CausalCorrespondence(nn.Module):
    """Displacement estimator over a small local search window."""

    def __init__(self, config: N01Config, *, neural: bool | None = None):
        super().__init__()
        self.config = config
        self.window = int(config.correspondence_window)
        self.width = int(config.correspondence_width)
        if neural is None:
            # "fixed" is the plain correlation estimate; the learnable residual
            # belongs to the candidate profile, and the head is zero-initialised so
            # that even the candidate starts from the analytic estimate.
            neural = config.alignment == AlignmentMode.CANDIDATE.value
        self.neural = bool(neural)
        # A differentiable soft-argmax keeps the correlation head trainable; the
        # integer peak is still reported for the discrete reference path.
        self.soft_argmax = bool(config.soft_argmax)
        self.temperature = float(config.soft_argmax_temperature)
        self.head: nn.Module | None = None
        if self.neural:
            side = 2 * self.window + 1
            self.head = nn.Sequential(
                nn.Conv2d(4, self.width, 3, padding=1),
                nn.GroupNorm(1, self.width),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.width, 1, 1),
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            self._side = side

    # -- low resolution RAW features ---------------------------------------
    @staticmethod
    def low_resolution_features(frames: torch.Tensor) -> torch.Tensor:
        """Mean / local deviation / gradient features of a RAW measurement."""
        if frames.ndim != 4:
            raise ValueError("frames must be [B,C,H,W]")
        pooled = F.avg_pool2d(frames, 2, ceil_mode=True)
        local_mean = F.avg_pool2d(pooled, 3, stride=1, padding=1)
        deviation = pooled - local_mean
        grad_y = F.pad(pooled[:, :, 1:] - pooled[:, :, :-1], (0, 0, 0, 1))
        grad_x = F.pad(pooled[:, :, :, 1:] - pooled[:, :, :, :-1], (0, 1, 0, 0))
        return torch.cat([pooled, deviation, grad_y, grad_x], dim=1)

    @staticmethod
    def _normalised_patch(patch: torch.Tensor) -> torch.Tensor:
        flat = patch.reshape(patch.shape[0], -1)
        centred = flat - flat.mean(dim=1, keepdim=True)
        norm = centred.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return centred / norm

    def correlate(self, reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Normalised correlation for every integer shift in the search window."""
        window = self.window
        batch, _, height, width = reference.shape
        ref_centred = self._normalised_patch(reference)
        padded = F.pad(other, (window, window, window, window))
        side = 2 * window + 1
        scores = reference.new_zeros((batch, side, side))
        for index_y in range(side):
            rows = padded[:, :, index_y : index_y + height, :]
            for index_x in range(side):
                patch = rows[:, :, :, index_x : index_x + width]
                scores[:, index_y, index_x] = (ref_centred * self._normalised_patch(patch)).sum(
                    dim=1
                )
        return scores

    def _correlate_window(
        self, reference: torch.Tensor, other: torch.Tensor, window: int
    ) -> torch.Tensor:
        """Same normalised correlation with a reduced (but still valid) window."""
        batch, _, height, width = reference.shape
        ref_centred = self._normalised_patch(reference)
        padded = F.pad(other, (window, window, window, window))
        side = 2 * window + 1
        scores = reference.new_zeros((batch, side, side))
        for index_y in range(side):
            rows = padded[:, :, index_y : index_y + height, :]
            for index_x in range(side):
                patch = rows[:, :, :, index_x : index_x + width]
                scores[:, index_y, index_x] = (ref_centred * self._normalised_patch(patch)).sum(
                    dim=1
                )
        return scores

    # -- estimation ---------------------------------------------------------
    def effective_window(self, height: int, width: int) -> int:
        """Largest search window that still fits inside the measurement grid."""
        limit = max((min(height, width) - 1) // 2, 0)
        if limit < 1:
            raise ValueError(
                f"correspondence needs a measurement grid of at least 3x3, got {height}x{width}"
            )
        return min(self.window, limit)

    def forward(
        self,
        reference: torch.Tensor,
        other: torch.Tensor,
        reference_valid: torch.Tensor | None = None,
        other_valid: torch.Tensor | None = None,
    ) -> MotionField:
        if reference.shape != other.shape:
            raise ValueError("correspondence expects two measurements of the same shape")
        batch = reference.shape[0]
        device = reference.device
        height, width = reference.shape[-2:]
        window = self.effective_window(height, width)
        if window != self.window:
            scores = self._correlate_window(reference, other, window)
        else:
            scores = self.correlate(reference, other)
        if self.head is not None:
            best = scores.amax(dim=(1, 2)).reshape(scores.shape[0], 1, 1, 1)
            patch_mean = F.avg_pool2d(scores.unsqueeze(1), 3, stride=1, padding=1)
            # ``expand`` aligns from the right, so the batch and channel axes are
            # spelled out and only the spatial axes are broadcast.
            best_map = best.expand(-1, 1, scores.shape[-2], scores.shape[-1])
            head_input = torch.cat(
                [
                    scores.unsqueeze(1),
                    scores.unsqueeze(1) - patch_mean,
                    best_map,
                    patch_mean,
                ],
                dim=1,
            )
            scores = scores + self.head(head_input).squeeze(1)

        side = 2 * window + 1
        flat = scores.reshape(batch, -1)
        best_index = flat.argmax(dim=1)
        shift_y = torch.div(best_index, side, rounding_mode="floor").to(torch.float32) - window
        shift_x = (best_index % side).to(torch.float32) - window
        shift = torch.stack([shift_y, shift_x], dim=1)
        best_score = flat.gather(1, best_index[:, None]).squeeze(1)
        residual = (1.0 - best_score).clamp(min=0.0)

        overlap = ((width - shift_x.abs()).clamp(min=0.0) / width) * (
            (height - shift_y.abs()).clamp(min=0.0) / height
        )
        valid = torch.ones((batch,), dtype=torch.bool, device=device)
        for mask in (reference_valid, other_valid):
            if mask is not None:
                usable = (mask > 0.5).float().mean(dim=(1, 2, 3))
                valid = valid & (usable > 0.25)
        valid = valid & (overlap > 0.0)

        # Two readings of the same correlation surface:
        #  * ``shift`` is the integer peak, used by the discrete index translation
        #    (exactly comparable with the per-source reference solution);
        #  * ``fractional`` is the offset from a differentiable soft-argmax
        #    expectation, which is what lets the small correlation head be trained
        #    at all -- a hard argmax has zero gradient with respect to the scores.
        if self.soft_argmax:
            probability = torch.softmax(
                (scores / max(self.temperature, 1e-6)).reshape(batch, -1), dim=-1
            )
            offsets = self._offset_grid(window, scores.device, scores.dtype)
            expected = (probability[:, :, None] * offsets[None]).sum(dim=1)
            fine = expected - shift
        elif self.config.fractional_alignment:
            fine = self._subpixel(scores, best_index, window)
        else:
            fine = None
        return MotionField(
            shift=shift,
            residual=residual,
            overlap=overlap,
            valid=valid,
            fractional=fine,
            scores=scores,
        )

    @staticmethod
    def _offset_grid(window: int, device, dtype) -> torch.Tensor:
        """[K, 2] integer displacement of every search-window entry."""
        side = 2 * window + 1
        rows = torch.arange(side, device=device, dtype=dtype) - window
        cols = torch.arange(side, device=device, dtype=dtype) - window
        grid = torch.stack(
            [rows[:, None].expand(side, side), cols[None, :].expand(side, side)], dim=-1
        )
        return grid.reshape(side * side, 2)

    def _subpixel(
        self, scores: torch.Tensor, best_index: torch.Tensor, window: int | None = None
    ) -> torch.Tensor:
        """Parabola fit around the integer peak, bounded to half a pixel."""
        window = self.window if window is None else window
        side = 2 * window + 1
        grid = scores.reshape(-1, side, side)
        batch = grid.shape[0]
        index = torch.arange(batch, device=grid.device)
        y = torch.div(best_index, side, rounding_mode="floor").reshape(-1)
        x = (best_index % side).reshape(-1)
        low_y, high_y = (y - 1).clamp(min=0), (y + 1).clamp(max=side - 1)
        low_x, high_x = (x - 1).clamp(min=0), (x + 1).clamp(max=side - 1)
        centre = grid[index, y, x].reshape(-1)

        def offset(low_value: torch.Tensor, high_value: torch.Tensor) -> torch.Tensor:
            low_value = low_value.reshape(-1)
            high_value = high_value.reshape(-1)
            denominator = low_value - 2.0 * centre + high_value
            return torch.where(
                denominator.abs() < 1e-6,
                torch.zeros_like(denominator),
                (0.5 * (low_value - high_value) / denominator).clamp(-0.5, 0.5),
            )

        vertical = offset(grid[index, low_y, x], grid[index, high_y, x])
        horizontal = offset(grid[index, y, low_x], grid[index, y, high_x])
        return torch.stack([vertical, horizontal], dim=1)

    # -- alignment ----------------------------------------------------------
    def align(
        self,
        source_values: torch.Tensor,
        source_variance: torch.Tensor,
        source_valid: torch.Tensor,
        field: MotionField,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a source frame into the reference frame's pixel grid.

        Integer shifts use index translation so that the vectorised result can be
        compared with the per-source reference solution; sub-pixel shifts use
        bilinear sampling.  Regions outside the source frame become invalid.
        """
        batch, channels, height, width = source_values.shape
        device = source_values.device
        if field.fractional is not None and self.config.fractional_alignment:
            fine = field.shift + field.fractional
            rows = torch.arange(height, device=device, dtype=torch.float32)[None, :, None]
            cols = torch.arange(width, device=device, dtype=torch.float32)[None, None, :]
            grid_y = rows + fine[:, 0][:, None, None]
            grid_x = cols + fine[:, 1][:, None, None]
            norm_y = (2.0 * grid_y / max(height - 1, 1) - 1.0).expand(batch, height, width)
            norm_x = (2.0 * grid_x / max(width - 1, 1) - 1.0).expand(batch, height, width)
            grid = torch.stack([norm_x, norm_y], dim=-1).contiguous()
            flag = torch.ones((batch, 1, height, width), device=device)
            packed = torch.cat(
                [
                    source_values * source_valid.float(),
                    source_variance,
                    source_valid.float(),
                    flag,
                ],
                dim=1,
            )
            sampled = F.grid_sample(
                packed, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            out_values = sampled[:, :channels]
            out_variance = sampled[:, channels : 2 * channels].clamp(min=1e-12)
            weight = sampled[:, 2 * channels : 3 * channels]
            inside = sampled[:, 3 * channels : 4 * channels]
            out_valid = (weight > 0.999) & (inside > 0.999)
            out_valid = out_valid & field.valid.reshape(-1, 1, 1, 1)
            out_values = torch.where(out_valid, out_values, torch.zeros_like(out_values))
            return out_values, out_variance, out_valid

        shift = torch.round(field.shift).to(torch.int64)
        rows = torch.arange(height, device=device)[None, :, None] + shift[:, 0][:, None, None]
        cols = torch.arange(width, device=device)[None, None, :] + shift[:, 1][:, None, None]
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        flat = (rows.clamp(0, height - 1) * width + cols.clamp(0, width - 1)).expand(
            batch, height, width
        )
        out_values = _gather_flat(source_values, flat)
        out_variance = _gather_flat(source_variance, flat).clamp(min=1e-12)
        # ``_gather_flat`` returns [B,C,H,W] while ``inside`` is [B,H,W] and
        # ``field.valid`` is [B]; the spatial and batch axes are spelled out so a
        # channel axis can never be silently paired with a spatial one.
        gathered_valid = _gather_flat(source_valid.float(), flat) > 0.5
        channel_ok = gathered_valid.all(dim=1, keepdim=True)
        spatial_ok = inside.unsqueeze(1) & field.valid.reshape(-1, 1, 1, 1)
        out_valid = (channel_ok & spatial_ok).expand_as(gathered_valid)
        out_values = torch.where(out_valid, out_values, torch.zeros_like(out_values))
        return out_values, out_variance, out_valid


class MotionChain:
    """Cumulative causal displacements between the current frame and history.

    Only already-arrived frames are compared.  A frame's displacement relative to
    the next-newer observed frame is estimated once; the shift used for
    alignment is the cumulative sum along that chain, so a long history never
    needs a direct and stale estimate against the reference frame.  The cached
    per-step shift is stored on the N01 state and never re-estimated for a frame
    that has already arrived.
    """

    def __init__(self, estimator: CausalCorrespondence | None):
        self.estimator = estimator

    @property
    def enabled(self) -> bool:
        return self.estimator is not None and self.estimator.config.alignment != AlignmentMode.NONE

    def relative_shift(
        self,
        newer: torch.Tensor,
        older: torch.Tensor,
        newer_valid: torch.Tensor,
        older_valid: torch.Tensor,
    ) -> MotionField:
        batch = newer.shape[0]
        device = newer.device
        if not self.enabled:
            return MotionField(
                shift=torch.zeros((batch, 2), device=device),
                residual=torch.zeros((batch,), device=device),
                overlap=torch.ones((batch,), device=device),
                valid=torch.ones((batch,), dtype=torch.bool, device=device),
            )
        return self.estimator(newer, older, newer_valid, older_valid)

    @staticmethod
    def cumulative(
        relative: list[MotionField], newer_observed: list[torch.Tensor]
    ) -> list[MotionField]:
        """Turn per-step shifts into shifts relative to the current frame.

        ``relative[i]`` is the displacement of slot ``i`` measured against the
        next-newer observed slot; ``newer_observed[i]`` is ``[B]`` bool telling
        whether that next-newer slot actually holds a measurement.
        """
        out: list[MotionField | None] = [None] * len(relative)
        if not relative:
            return []
        batch = relative[0].shift.shape[0]
        device = relative[0].shift.device
        running = torch.zeros((batch, 2), device=device)
        residual = torch.zeros((batch,), device=device)
        overlap = torch.ones((batch,), device=device)
        valid = torch.ones((batch,), dtype=torch.bool, device=device)
        for index in range(len(relative)):
            field = relative[index]
            informative = newer_observed[index]
            delta = field.select(informative)
            running = running + delta.shift
            residual = residual + delta.residual
            overlap = overlap * delta.overlap
            valid = valid & delta.valid
            out[index] = MotionField(
                shift=running.clone(),
                residual=residual.clone(),
                overlap=overlap.clone(),
                valid=valid.clone(),
                fractional=None if delta.fractional is None else delta.fractional.clone(),
            )
        return out  # type: ignore[return-value]

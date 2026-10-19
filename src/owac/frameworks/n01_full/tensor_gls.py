"""Batched local constant/affine GLS on aligned innovations, in FP32.

Moments are sums over finite rectangular candidates, never connectivity closures.
For overlapping interpolation taps/unknown temporal correlation, coefficient
variance uses the Cauchy bound n * diag((A'WA)^-1), rather than claiming independent
noise. Scores and BOCPD posteriors are empirical/model quantities, not p-values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LocalFit:
    score: torch.Tensor
    valid: torch.Tensor
    radius: torch.Tensor
    coefficients: torch.Tensor
    rank: torch.Tensor
    residual_rms: torch.Tensor


def _sums(field, index, regions):
    flat = field.flatten(-2)
    out = flat.new_zeros(*flat.shape[:-1], regions)
    return out.scatter_add(-1, index.expand_as(flat), flat)


def _neighbour_sum(field, rows, cols, radius):
    shape = field.shape
    field = field.reshape(-1, 1, rows, cols)
    if radius:
        field = F.avg_pool2d(field, radius * 2 + 1, stride=1, padding=radius, divisor_override=1)
    return field.reshape(shape)


def fit_aligned_innovations(
    difference,
    valid,
    variance,
    *,
    rows,
    cols,
    radii=(0, 1, 2),
    affine=True,
    minimum_cells=4,
    compatible_score=3.0,
    unknown_correlation=True,
) -> LocalFit:
    """Fit each source/channel's innovation field on each finite local candidate.

    Inputs are B,S,C,H,W and B,S,1,H,W validity. Coordinates are current-query
    coordinates, whose values were pulled from their actual historical sample
    locations. The zero-innovation hypothesis keeps constant shifts detectable;
    fitting a brightness transform and subtracting it would hide those changes.
    """
    difference, variance = difference.float(), variance.float()
    if difference.ndim != 5 or variance.shape != difference.shape:
        raise ValueError("GLS innovation/variance must have the same B,S,C,H,W shape")
    batch, sources, channels, height, width = difference.shape
    if valid.shape != (batch, sources, 1, height, width):
        raise ValueError("GLS validity must be B,S,1,H,W")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=difference.device),
        torch.arange(width, device=difference.device),
        indexing="ij",
    )
    region_y = (yy * rows // height).clamp_max(rows - 1)
    region_x = (xx * cols // width).clamp_max(cols - 1)
    index = (region_y * cols + region_x).flatten()
    x, y = (xx.float() + 0.5) * (2 / width) - 1, (yy.float() + 0.5) * (2 / height) - 1
    basis = (
        torch.stack((torch.ones_like(x), x, y))
        if affine
        else torch.ones(1, height, width, device=x.device)
    )
    terms = basis.shape[0]
    legal = (
        valid
        & torch.isfinite(difference).all(2, keepdim=True)
        & torch.isfinite(variance).all(2, keepdim=True)
        & (variance > 0).all(2, keepdim=True)
    )
    safe_difference = torch.where(legal, difference, torch.zeros_like(difference))
    weight = torch.where(legal, variance.clamp_min(1e-12).reciprocal(), torch.zeros_like(variance))
    # The small matrices are [B,S,C,R,P,P]; no per-pixel Python objects or dense n*n covariance.
    normal = _sums(
        weight[..., None, None, :, :] * basis[:, None] * basis[None, :], index, rows * cols
    )
    rhs = _sums((weight * safe_difference)[..., None, :, :] * basis, index, rows * cols)
    energy = _sums(weight * safe_difference.square(), index, rows * cols)
    counts = _sums(legal.float(), index, rows * cols).expand(-1, -1, channels, -1)
    candidates = []
    for radius in sorted(set(radii)):
        normal_r = _neighbour_sum(normal, rows, cols, radius).movedim(-1, -3)
        rhs_r = _neighbour_sum(rhs, rows, cols, radius).movedim(-1, -2)
        count = _neighbour_sum(counts, rows, cols, radius)
        energy_r = _neighbour_sum(energy, rows, cols, radius)
        if affine:
            # Re-express global moments in the candidate's local chart for numerical rank.
            cy, cx = torch.meshgrid(
                torch.arange(rows, device=x.device),
                torch.arange(cols, device=x.device),
                indexing="ij",
            )
            cx, cy = (
                (cx.flatten().float() + 0.5) * 2 / cols - 1,
                (cy.flatten().float() + 0.5) * 2 / rows - 1,
            )
            chart = torch.eye(3, device=x.device).expand(rows * cols, 3, 3).clone()
            sx, sy = (radius * 2 + 1) / cols, (radius * 2 + 1) / rows
            chart[:, 1, 0], chart[:, 1, 1] = -cx / sx, 1 / sx
            chart[:, 2, 0], chart[:, 2, 2] = -cy / sy, 1 / sy
            normal_r = chart @ normal_r @ chart.transpose(-1, -2)
            rhs_r = (chart @ rhs_r[..., None]).squeeze(-1)
        eigen = torch.linalg.eigvalsh(normal_r)
        rank = (eigen > eigen[..., -1:].clamp_min(1e-12) * 1e-5).sum(-1)
        inverse = torch.linalg.pinv(normal_r, hermitian=True, rtol=1e-5)
        coefficients = (inverse @ rhs_r[..., None]).squeeze(-1)
        coefficient_variance = inverse.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        if unknown_correlation:
            coefficient_variance = coefficient_variance * count[..., None].clamp_min(1)
        fitted_energy = (coefficients * rhs_r).sum(-1)
        residual = ((energy_r - fitted_energy).clamp_min(0) / (count - terms).clamp_min(1)).sqrt()
        coefficient_score = (coefficients.square() / coefficient_variance).mean(-1).sqrt()
        channel_score = torch.maximum(coefficient_score, residual)
        accepted = (rank == terms) & (count >= max(minimum_cells, terms))
        score = channel_score.amax(2)
        accepted = accepted.all(2)
        candidates.append((score, accepted, coefficients, rank, residual, radius))
    if not candidates:
        raise ValueError("at least one finite spatial candidate required")
    # Smallest compatible candidate; otherwise the valid candidate with least lack-of-fit.
    scores = torch.stack([x[0].masked_fill(~x[1], float("inf")) for x in candidates])
    compatible = scores <= compatible_score
    first = compatible.to(torch.int64).argmax(0)
    best = scores.argmin(0)
    choice = torch.where(compatible.any(0), first, best)
    selected_score = scores.gather(0, choice[None]).squeeze(0)
    valid_fit = torch.isfinite(selected_score)
    selected_score = torch.where(valid_fit, selected_score, torch.zeros_like(selected_score))
    radii_tensor = torch.tensor([x[-1] for x in candidates], device=x.device)
    # Preserve all candidate coefficients for exact inspection, with chosen radius explicit.
    return LocalFit(
        selected_score,
        valid_fit,
        radii_tensor[choice],
        torch.stack([x[2] for x in candidates]),
        torch.stack([x[3] for x in candidates]),
        torch.stack([x[4] for x in candidates]),
    )


def conservative_source_mask(reference_mask, source_mask, offsets):
    """Remove every source sample reached by a withdrawn reference cell.

    Reverse scatter covers all positive bilinear taps, not just a rounded flow
    coordinate. Input masking before the CNN prevents withdrawn pixel values from
    leaking through its receptive field and the learned weight predictor.
    """
    if offsets is None:
        return source_mask & reference_mask
    batch, length, _, height, width = source_mask.shape
    rows = torch.arange(height, device=offsets.device)[None, None, :, None]
    cols = torch.arange(width, device=offsets.device)[None, None, None, :]
    x, y = cols + offsets[:, :, 0].detach(), rows + offsets[:, :, 1].detach()
    left, top = x.floor(), y.floor()
    fx, fy = x - left, y - top
    forbidden = (~reference_mask[:, :, 0]).to(torch.int64)
    marks = torch.zeros(batch, length, height * width, dtype=torch.int64, device=x.device)
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        xx, yy = left + dx, top + dy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height) & (weight > 0)
        indices = (yy.clamp(0, height - 1).long() * width + xx.clamp(0, width - 1).long()).flatten(
            2
        )
        marks.scatter_add_(2, indices, (forbidden * inside).flatten(2))
    return source_mask & (marks.reshape(batch, length, 1, height, width) == 0)

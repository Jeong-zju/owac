"""Measurement-only temporal refutation for the existing N01 selector.

Single-pixel three-sigma tests miss coherent subpixel changes at low SNR. Compare
current measurements with a precision-weighted historical prediction, then test
signed innovations over small directional neighbourhoods. A box sum uses each
current sample once; linearly warped variances bound shared interpolation noise.
Channels may be correlated or literal copies: their standard deviations add
before squaring. Only distinct source frames/pixels are assumed independent.
No feature or geometry label enters here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def temporal_innovation_cost(state, aligned, depths, *, planes: int, radius: int):
    """Return local three-sigma prediction excess per candidate [K,B,P,H,W].

    The centre, vertical and horizontal tests cover changes of either sign without
    averaging the opposite sides of a narrow edge together. Costs are dimensionless
    and detached. Historical source coefficients are combined before squaring;
    repeated tests against the same current observation are not independent
    evidence. Invalid channels are excluded, and CFA planes remain separate.
    """
    values, variance, valid = aligned
    depths = torch.as_tensor(depths, device=values.device, dtype=torch.long)
    slots, batch, channels, height, width = values.shape
    overlap = (valid > 0.5) & state.valid[0][None]
    overlap &= state.observed[:, :, None, None, None]
    # Slot zero is the same observation, rather than independent evidence.
    overlap = overlap.clone()
    overlap[0] = False
    group = channels // planes
    side = 2 * radius + 1
    kernels = ((1, 1),) if radius == 0 else ((1, 1), (side, 1), (1, side))
    rank = state.observed.long().cumsum(0)
    cache = {}
    shape = (batch, planes, group, height, width)
    for depth in depths.unique().tolist():
        use = overlap & (rank <= depth)[:, :, None, None, None]
        precision = torch.where(use, 1 / variance.clamp_min(1e-12), 0)
        coefficients = precision / precision.sum(0, keepdim=True).clamp_min(1e-12)
        prediction = (coefficients * torch.where(use, values, 0)).sum(0)
        historical_noise = (coefficients.square() * torch.where(use, variance, 0)).sum(0)
        usable = use.any(0)
        difference = torch.where(usable, prediction - state.values[0], 0).reshape(shape).sum(2)
        # Cauchy bound covers all within-pixel channel correlations, including
        # the synthetic three-channel copies. It never invents three readings.
        channel_noise = torch.where(usable, historical_noise + state.variance[0], 0)
        noise = channel_noise.sqrt().reshape(shape).sum(2).square()
        supported = usable.reshape(shape).any(2)
        tests = []
        for kernel in kernels:
            padding = (kernel[0] // 2, kernel[1] // 2)
            area = kernel[0] * kernel[1]
            delta = F.avg_pool2d(difference, kernel, stride=1, padding=padding) * area
            bound = F.avg_pool2d(noise, kernel, stride=1, padding=padding) * area
            excess = (delta.square() / bound.clamp_min(1e-12) - 9.0).clamp_min(0.0)
            tests.append(torch.where(supported, excess, torch.zeros_like(excess)))
        # The hypothesis test uses the current sample once. The selector's
        # cumulative risk charges each carried unique historical frame; this
        # multiplier is a cost, never an independent-evidence count for the test.
        carried = use.reshape(slots, batch, planes, group, height, width).any(3).sum(0)
        cache[depth] = torch.stack(tests).amax(0) * carried
    return torch.stack([cache[int(depth)] for depth in depths])

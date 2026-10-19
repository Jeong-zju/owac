"""Experimental source-wise innovation weights and a common-history risk cap.

The newest unique measurement anchors each older source at its original pixel.
Weights depend on noisy data: the returned variance is a diagnostic proxy, not
a calibrated posterior. A corrupted current anchor can reject correct history.
Raw buffers, source identities and the support graph are never modified here.
"""

from __future__ import annotations

import torch

from .state import measurement_age
from .support import DomainStatistics, candidate_offsets

_EPS = 1e-6  # Same precision-weight convention as the legacy integrator.
_TINY = 1e-12


def innovation_weights(x, variance, valid, reference, reference_variance, reference_valid, floor):
    """Noise-subtracted squared innovation, once per original source sample.

    Missing current evidence does not refute valid old evidence. Zero estimated
    excess bias gives weight one, including a zero declared noise variance.
    """
    bias = ((x - reference).square() - variance - reference_variance).clamp_min(0)
    ratio = variance / (variance + bias).clamp_min(_TINY)
    ratio = torch.where(bias == 0, torch.ones_like(ratio), ratio)
    ratio = torch.where(reference_valid, ratio, torch.ones_like(ratio))
    return torch.where(valid & (ratio >= floor), ratio, torch.zeros_like(ratio))


def history_scale(current_w, current_s, current_q, history_w, history_s, history_q):
    """Downscale history so a common stale bias cannot average itself away.

    Minimize (1-a)^2 V0 + a^2 (Vh + bias^2), constrained to no more history
    than the source weights already allowed. The variance/bias terms are plug-in
    proxies; shared anchor noise and data-dependent selection remain correlated.
    """
    current = current_s / current_w.clamp_min(_TINY)
    history = history_s / history_w.clamp_min(_TINY)
    v0 = current_q / current_w.square().clamp_min(_TINY)
    vh = history_q / history_w.square().clamp_min(_TINY)
    bias = ((history - current).square() - v0 - vh).clamp_min(0)
    original = history_w / (current_w + history_w).clamp_min(_TINY)
    risk = v0 + vh + bias
    optimum = torch.where(risk > 0, v0 / risk.clamp_min(_TINY), original)
    alpha = torch.minimum(original, optimum)
    scale = alpha * current_w / ((1 - alpha) * history_w).clamp_min(_TINY)
    scale = scale.clamp(0, 1)
    return torch.where((current_w > 0) & (history_w > 0), scale, torch.ones_like(scale))


def _gather(tensor, shift, dy=0, dx=0):
    """Index translation without wrapping, independently for each member."""
    batch, planes, height, width = tensor.shape
    rows = torch.arange(height, device=tensor.device)[None, :, None] + shift[:, 0, None, None] + dy
    cols = torch.arange(width, device=tensor.device)[None, None, :] + shift[:, 1, None, None] + dx
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    flat = (rows.clamp(0, height - 1) * width + cols.clamp(0, width - 1)).expand(
        batch, height, width
    )
    index = flat.reshape(batch, 1, -1).expand(batch, planes, -1)
    value = tensor.reshape(batch, planes, -1).gather(2, index).reshape_as(tensor)
    return value, inside[:, None]


def _parts(fine, fractional):
    if not fractional:
        return fine.round().long(), [(0, 0, torch.ones_like(fine[:, 0]))]
    base = fine.floor().long()
    fraction = fine - base
    corners = []
    for dy in (0, 1):
        for dx in (0, 1):
            a = fraction[:, 0] if dy else 1 - fraction[:, 0]
            b = fraction[:, 1] if dx else 1 - fraction[:, 1]
            if bool(((a * b) > 0).any()):
                corners.append((dy, dx, a * b))
    return base, corners


def _reference_on_source(current, variance, valid, fine, fractional):
    """Inverse warp the current anchor, retaining squared interpolation noise.

    One reference for each original source, reused by every integration path.
    All positive interpolation taps must be valid; absent reference is unknown.
    """
    base, corners = _parts(-fine, fractional)
    value, noise = torch.zeros_like(current), torch.zeros_like(variance)
    usable = torch.ones_like(valid)
    for dy, dx, coefficient in corners:
        c = coefficient[:, None, None, None]
        x, inside = _gather(current, base, dy, dx)
        v, _ = _gather(variance, base, dy, dx)
        m, _ = _gather(valid, base, dy, dx)
        value = value + c * x
        noise = noise + c.square() * v
        usable &= (c == 0) | (m & inside)
    return value, noise, usable


def reliable_statistics(integrator, state, edges, *, motion=None, depths=None):
    """Enumerate coefficients by original (slot, plane, coordinate) identity.

    The small experimental path favors an auditable source walk over speed.
    Interpolation paths reaching the SAME source are added before reliability,
    inverse variance weighting, and squared-weight moments. Connected paths and
    temporal prefixes are unchanged; no coefficient crosses a closed graph edge.
    """
    config = integrator.config
    values = state.values
    slots, batch, _, height, width = values.shape
    x = integrator._split_planes(values, slots, batch, (height, width))
    v = integrator._split_planes(state.variance, slots, batch, (height, width))
    m = integrator._split_valid(state.valid, slots, batch, (height, width))
    m = m & state.observed[:, :, None, None, None]
    depths = tuple(config.time_depths if depths is None else depths)
    ages = torch.nan_to_num(measurement_age(state), nan=0.0, posinf=0.0)
    rank = state.observed.long().cumsum(0)
    fractional = config.fractional_alignment
    shifts, motion_valid, reliabilities = [], [], []
    for slot in range(slots):
        fine = torch.zeros(batch, 2, device=values.device)
        ok = torch.ones(batch, dtype=torch.bool, device=values.device)
        if motion is not None and config.alignment != "none":
            field = motion[slot]
            fine = field.shift
            if fractional and field.fractional is not None:
                fine = fine + field.fractional
            ok = field.valid
        shifts.append(fine)
        motion_valid.append(ok)
        if slot == 0:
            reliability = m[slot].float()
        else:
            reference, reference_v, reference_m = _reference_on_source(
                x[0], v[0], m[0], fine, fractional
            )
            reliability = innovation_weights(
                x[slot],
                v[slot],
                m[slot],
                reference,
                reference_v,
                reference_m,
                config.source_reliability_floor,
            )
        reliabilities.append(reliability)
    reliability = torch.stack(reliabilities)
    # Diagnostic snapshots only; these tensors are not recurrent evidence.
    integrator.last_source_reliability = reliability.detach()
    keys = (
        "mu",
        "variance_proxy",
        "age",
        "support_count",
        "unique_frames",
        "unique_samples",
        "weight_concentration",
    )
    chips = {key: [] for key in keys}
    connections, radii, depth_ids, masses, scales = [], [], [], [], []
    for radius in config.spatial_radii:
        offsets = candidate_offsets(radius)
        path = {
            offset: integrator._edge_coefficient(edges, *offset, radius)[0] for offset in offsets
        }
        # All source paths are built once, then filtered into each time prefix.
        weights, source_x, source_v, source_slot = [], [], [], []
        for slot in range(slots):
            if not bool(state.observed[slot].any()):
                continue
            base, corners = _parts(shifts[slot], fractional)
            coefficients = {}
            for (dy, dx), connected in path.items():
                for cy, cx, alpha in corners:
                    key = dy + cy, dx + cx
                    c = connected * alpha[:, None, None, None]
                    coefficients[key] = coefficients.get(key, 0.0) + c
            for (dy, dx), coefficient in coefficients.items():
                xx, inside = _gather(x[slot], base, dy, dx)
                vv, _ = _gather(v[slot], base, dy, dx)
                mm, _ = _gather(m[slot], base, dy, dx)
                rr, _ = _gather(reliability[slot], base, dy, dx)
                c = coefficient * rr * (mm & inside & motion_valid[slot][:, None, None, None])
                weights.append(c / (vv + _EPS))
                source_x.append(xx)
                source_v.append(vv)
                source_slot.append(slot)
        if not weights:
            # init_state has no observations; preserve correctly shaped zeros.
            weights = [torch.zeros_like(x[0])]
            source_x, source_v, source_slot = [x[0]], [v[0]], [0]
        w, xx, vv = map(torch.stack, (weights, source_x, source_v))
        slot_ids = torch.tensor(source_slot, device=values.device)
        current = (slot_ids == 0)[:, None, None, None, None]
        age = ages[slot_ids, :, None, None, None]
        for depth in depths:
            allowed = (rank[slot_ids] <= int(depth)) & state.observed[slot_ids]
            dw = w * allowed[:, :, None, None, None]
            cw, hw = dw * current, dw * ~current
            cw_sum, hw_sum = cw.sum(0), hw.sum(0)
            beta = history_scale(
                cw_sum,
                (cw * xx).sum(0),
                (cw.square() * vv).sum(0),
                hw_sum,
                (hw * xx).sum(0),
                (hw.square() * vv).sum(0),
            )
            final_w = cw + beta[None] * hw
            total = final_w.sum(0)
            denominator = total.clamp_min(_EPS)
            actual = final_w > 0
            frames = (
                torch.stack([actual[slot_ids == i].any(0) for i in range(slots)]).sum(0).float()
            )
            for key, value in (
                ("mu", (final_w * xx).sum(0) / denominator),
                ("variance_proxy", (final_w.square() * vv).sum(0) / (denominator.square() + _EPS)),
                ("age", (final_w * age).sum(0) / denominator),
                ("support_count", total),
                ("unique_frames", frames),
                ("unique_samples", actual.sum(0).float()),
                ("weight_concentration", total.square() / (final_w.square().sum(0) + _EPS)),
            ):
                chips[key].append(value)
            masses.append((beta * hw_sum / denominator).detach())
            scales.append(beta.detach())
            connections.append(torch.stack([c > 0 for c in path.values()]).float().mean(0))
            radii.append(radius)
            depth_ids.append(int(depth))
    integrator.last_history_mass = torch.stack(masses)
    integrator.last_history_scale = torch.stack(scales)
    return DomainStatistics(
        **{key: torch.stack(value) for key, value in chips.items()},
        radius_index=torch.tensor(radii, device=values.device),
        depth_index=torch.tensor(depth_ids, device=values.device),
        offset_index=torch.zeros(len(radii), dtype=torch.long, device=values.device),
        offsets=[(0, 0)] * len(radii),
        depths=depths,
        correlation_note=(
            "Data-dependent innovation weighting and shared current anchor noise: "
            "variance_proxy is not a calibrated posterior."
        ),
    ), torch.stack(connections)

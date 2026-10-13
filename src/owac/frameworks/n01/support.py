"""N01 support graph, merge/split and per-source statistics.

This module is the definition site of the hand-computable reference solution.

Definitions (all causal; only the current and past frames are used)
------------------------------------------------------------------
* A **support domain** for target position ``p`` is the Cartesian product of a
  spatial radius ``r`` in ``spatial_radii``, a temporal length ``D`` in
  ``time_depths``. All connected offsets inside the radius contribute samples
  to that one domain. Tiny has nine domains, not 105 single-offset candidates.
* The same-plane one-hop support graph uses the (left, up) neighbour pairs at
  their true sampling positions.  ``merge_threshold > split_threshold`` is the
  hysteresis band: an edge opens only after sustained compatibility, and a new
  significant measured refutation closes it at once. Missing evidence decays
  slowly; the explicit open bit remembers the hysteresis band.
* Per-sample coefficient ``w_i = m_i * c_i / (v_i + eps)`` with
  ``S = sum w_i x_i``, ``W = sum w_i``, ``mu = S/(W+eps)``,
  ``V_diag = sum w_i^2 v_i / (W^2+eps)`` and ``age = sum w_i (t - t_i)/(W+eps)``.
* ``connected_power`` is the supported path weight: the product of the open edge
  weights along the local path inside the window, evaluated by separability from
  the target position.  It decays with the number of hops and stays bounded by
  the window, so no arbitrary transitive connection is pulled in.
  ``reference_domains`` independently walks the same original sources with
  those coefficients; both use the identical statistic formulas and
  are compared numerically in ``tests/encoders/n01/test_support.py``.

``V_diag`` is a variance estimate only under the corresponding independent-noise
assumption.  It is reported as ``variance_proxy`` with a correlation note, never
as an exact posterior.  Feature updates never write back into the measurement
buffer: the statistics always come from real measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N01Config
from .state import measurement_age

__all__ = [
    "DomainStatistics",
    "SupportDomainIntegrator",
    "candidate_offsets",
    "plane_index",
    "edge_field_reference",
]

_EPS = 1e-6


def candidate_offsets(radius: int) -> list[tuple[int, int]]:
    """Deterministic (row, col) offsets of the radius window, row-major."""
    return [(row, col) for row in range(-radius, radius + 1) for col in range(-radius, radius + 1)]


def plane_index(config: N01Config, channel: int) -> int:
    """CFA plane of a measurement channel.

    ``planes == 1`` keeps the measurement channels as one grid (the RawVLA
    three-channel pseudo-RAW / HDR-derived case, which is not called Bayer).
    ``planes == 4`` maps the four raw channels of the 2x2 mosaic onto four
    independent CFA planes; they are separately sampled measurements at their
    own phase, never four co-sited colours.
    """
    if config.planes == 1:
        return 0
    if not 0 <= channel < config.planes:
        raise ValueError(f"channel {channel} outside the {config.planes} CFA planes")
    return channel


@dataclass
class DomainStatistics:
    """Stacked statistics of every candidate domain.

    Entries are ``[C,B,P,H,W]`` with ``C`` the flattened
    (radius, time-depth, offset) candidate index, ``B`` the batch, ``P`` the CFA
    plane and ``H, W`` the plane grid.  ``mu`` is blocked here by the candidate
    index order; the per-chip fields are also returned separately.
    """

    mu: torch.Tensor
    variance_proxy: torch.Tensor
    age: torch.Tensor
    support_count: torch.Tensor
    unique_frames: torch.Tensor
    unique_samples: torch.Tensor
    weight_concentration: torch.Tensor
    radius_index: torch.Tensor
    depth_index: torch.Tensor
    offset_index: torch.Tensor
    offsets: list[tuple[int, int]] = field(default_factory=list)
    depths: tuple[int, ...] = ()
    correlation_note: str = (
        "variance_proxy assumes independent per-sample noise; motion interpolation and "
        "correlated sensor noise are not represented, so it is not an exact posterior"
    )

    @property
    def candidate_count(self) -> int:
        return int(self.mu.shape[0])


class SupportDomainIntegrator(nn.Module):
    """Reorganisable multi-timescale measurement state field."""

    def __init__(self, config: N01Config):
        super().__init__()
        self.config = config
        # The gather window is always odd (2r+1) so that unfolding uses a
        # symmetric padding of exactly (window-1)/2.
        self.radius = int(max(config.spatial_radii))
        self.window = 2 * self.radius + 1
        if config.neighbourhood_radius < self.radius:
            raise ValueError("neighbourhood_radius must cover every spatial radius")
        self.offsets = candidate_offsets(self.radius)
        self.edge_head: nn.Module | None = None
        self.gate_scale = None
        if config.learnable_gate:
            channels = 8
            self.edge_head = nn.Sequential(
                nn.Conv2d(channels, config.gate_hidden, 3, padding=1),
                nn.GroupNorm(1, config.gate_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.gate_hidden, 2, 1),
            )
            # The gate multiplies the analytic value by ``1 + s * delta`` with
            # ``delta = tanh(head(...))`` and a strictly positive ``s``.  Zero-
            # initialising the head therefore makes the analytic gate the exact
            # initialisation (the published control condition) while the gradient
            # still reaches the head through ``s`` instead of being blocked by a
            # multiplicative zero.
            nn.init.zeros_(self.edge_head[-1].weight)
            nn.init.zeros_(self.edge_head[-1].bias)
            self.gate_scale = nn.Parameter(torch.full((1,), 0.01))

    # -- planes -------------------------------------------------------------
    def _split_planes(self, tensor: torch.Tensor, slots: int, batch: int, hw: tuple[int, int]):
        """[N,B,C,H,W] -> [N,B,P,H,W] with a per-plane channel reduction."""
        channels = tensor.shape[2]
        height, width = hw
        planes = self.config.planes
        if planes == 1:
            return tensor.reshape(slots, batch, 1, channels, height, width).mean(dim=3)
        return tensor.reshape(slots, batch, planes, channels // planes, height, width).mean(dim=3)

    def _split_valid(self, tensor: torch.Tensor, slots: int, batch: int, hw: tuple[int, int]):
        channels = tensor.shape[2]
        height, width = hw
        planes = self.config.planes
        if planes == 1:
            return (tensor > 0.5).reshape(slots, batch, 1, channels, height, width).all(dim=3)
        return (
            (tensor > 0.5)
            .reshape(slots, batch, planes, channels // planes, height, width)
            .all(dim=3)
        )

    # -- alignment ----------------------------------------------------------
    def aligned_slots(self, state, motion, estimator=None):
        """Per-slot frames sampled into the current frame's pixel grid."""
        config = self.config
        values = state.values
        variance = state.variance
        valid = state.valid.float()
        if estimator is None or config.alignment == "none" or motion is None:
            return values, variance, valid
        out_values, out_variance, out_valid = [], [], []
        for slot in range(values.shape[0]):
            shifted_values, shifted_variance, shifted_valid = estimator.align(
                values[slot], variance[slot], valid[slot] > 0.5, motion[slot]
            )
            out_values.append(shifted_values)
            out_variance.append(shifted_variance)
            out_valid.append(shifted_valid.float())
        return (
            torch.stack(out_values, dim=0),
            torch.stack(out_variance, dim=0),
            torch.stack(out_valid, dim=0),
        )

    # -- support graph ------------------------------------------------------
    def edge_field(self, frame, state, motion=None, aligned=None, estimator=None):
        """Analytic one-hop support edges plus a zero-initialised learned residual.

        Returns ``edges`` ``[B,P,2,H,W]`` in (0, 1] and a diagnostics dict.  The
        motion term uses only correspondence that was actually estimated; when no
        motion information is available the term is 1 and the unresolved fraction
        is reported instead of inventing a value.
        """
        if aligned is None:
            aligned = self.aligned_slots(state, motion, estimator)
        values, variance, valid = aligned
        slots, batch = values.shape[0], values.shape[1]
        hw = (values.shape[3], values.shape[4])
        plane_values = self._split_planes(values, slots, batch, hw)
        plane_variance = self._split_planes(variance, slots, batch, hw)
        plane_valid = self._split_valid(valid, slots, batch, hw)
        height, width = hw

        current = plane_values[0]
        current_var = plane_variance[0]
        current_valid = plane_valid[0]
        left_values = F.pad(current[..., :, :-1], (1, 0))
        left_var = F.pad(current_var[..., :, :-1], (1, 0))
        left_valid = F.pad(current_valid[..., :, :-1], (1, 0))
        up_values = F.pad(current[..., :-1, :], (0, 0, 1, 0))
        up_var = F.pad(current_var[..., :-1, :], (0, 0, 1, 0))
        up_valid = F.pad(current_valid[..., :-1, :], (0, 0, 1, 0))

        left_gate = self._pair_gate(
            current, left_values, current_var, left_var, current_valid & left_valid
        )
        up_gate = self._pair_gate(current, up_values, current_var, up_var, current_valid & up_valid)
        gates = torch.stack([left_gate, up_gate], dim=2)
        distance = self._distance_gate(current)
        gates = gates * distance
        motion_gate, unresolved = self._motion_gate(motion, batch, current)
        gates = (gates * motion_gate[:, :, None]).clamp(0.0, 1.0)
        if self.edge_head is not None:
            gates = self._residual_gate(gates, current, height, width)
        edges = gates * (current_valid | left_valid | up_valid)[:, :, None].float()
        diagnostics = {
            "motion_unresolved_fraction": unresolved,
            "edge_mean": float(edges.mean().detach()),
        }
        return edges, diagnostics

    def _pair_gate(self, centre, neighbour, centre_var, neighbour_var, pair_valid):
        config = self.config
        sigma = torch.sqrt(centre_var + neighbour_var + _EPS)
        innovation = (centre - neighbour).abs() / sigma
        data_gate = torch.exp(-(innovation**2) / (config.edge_temperature * 2.0))
        gradient = (centre - neighbour).abs()
        edge_effect = gradient / (gradient + sigma + _EPS)
        edge_gate = torch.exp(-edge_effect / max(config.edge_beta, _EPS))
        gate = data_gate * edge_gate
        return torch.where(pair_valid, gate, torch.zeros_like(gate))

    def _distance_gate(self, current: torch.Tensor) -> torch.Tensor:
        """Distance term per (left, up) pair, shaped [B,P,2,H,W].

        Two same-plane samples of a 2x2 CFA sit two mosaic pixels apart; the
        separation is normalised by that plane spacing.  The term is a scalar
        weight per direction, so it is broadcast over the spatial grid.
        """
        config = self.config
        batch, planes, height, width = current.shape
        # The one-hop pair is one grid spacing apart: two mosaic pixels on a 2x2
        # CFA, one pixel on a shared measurement grid.  The decay length is the
        # largest candidate radius, and a radius-0 configuration has no distance
        # term at all.
        spacing = 2.0 if planes > 1 else 1.0
        decay = max(config.spatial_radii)
        if decay <= 0:
            gate = 1.0
        else:
            separation = spacing
            physical_radius = spacing * float(decay)
            gate = math.exp(-(separation**2) / (2.0 * physical_radius**2))
        gate = max(min(gate, 1.0), 0.0)
        return torch.full(
            (batch, planes, 2, height, width), gate, device=current.device, dtype=current.dtype
        )

    def _motion_gate(self, motion, batch: int, current: torch.Tensor):
        config = self.config
        dtype, device = current.dtype, current.device
        if motion is None or (isinstance(motion, (list, tuple)) and len(motion) < 2):
            return torch.ones((batch, 1, 1, 1), device=device, dtype=dtype), 1.0
        # Slot zero is the current frame's identity warp. Spatial support uses
        # the nearest historical correspondence; identity carries no evidence
        # of temporal compatibility. Invalid history remains unresolved.
        head = motion[1] if isinstance(motion, (list, tuple)) else motion
        error = head.residual.detach().to(dtype)
        valid = head.valid.detach()
        gate = torch.exp(-config.edge_alpha * error)[:, None, None, None]
        gate = torch.where(valid[:, None, None, None], gate, torch.ones_like(gate))
        unresolved = float((~valid).float().mean().detach())
        return gate.expand(-1, self.config.planes, -1, -1), unresolved

    def _residual_gate(self, gates, current, height, width):
        batch, planes, _, _, _ = gates.shape
        flat = gates.reshape(batch * planes, 2, height, width)
        centre = current.reshape(batch * planes, 1, height, width)
        left = F.pad(centre[..., :, :-1], (1, 0))
        up = F.pad(centre[..., :-1, :], (0, 0, 1, 0))
        gradient = F.pad((centre[..., 1:, :] - centre[..., :-1, :]).abs(), (0, 0, 1, 0))
        head_input = torch.cat(
            [flat, flat - flat.mean(dim=1, keepdim=True), centre, left, up, gradient], dim=1
        )
        residual = torch.tanh(self.edge_head(head_input))
        scale = 0.01 + torch.nn.functional.softplus(self.gate_scale).reshape(1, 1, 1, 1)
        refined = (flat * (1.0 + scale * residual)).clamp(0.0, 1.0)
        return refined.reshape(batch, planes, 2, height, width)

    def update_edges(self, previous_edges, support_age, new_edges, valid, previous_open=None):
        """Hysteresis update; returns edges, per-pair age and merge/split counts.

        ``merge_threshold`` is the barrier an edge must cross to become open, and
        once open it only closes below ``split_threshold``.  Three regimes:

        * compatible evidence (``new_edges >= merge_threshold``): the edge rises
          towards the measured compatibility by ``edge_alpha``, so it needs
          sustained agreement before it opens;
        * measured refutation (``new_edges < split_threshold``): replace the
          strength by the new incompatibility and close immediately;
        * occlusion (no measurement for the pair): the edge only decays by
          ``edge_forget``, so a short occlusion does not destroy the support.

        ``valid`` is ``[B,H,W]`` (the measurement mask, shared by both directions
        and all planes), ``[B,P,H,W]`` per plane, or ``[B,P,2,H,W]`` per pair.
        """
        config = self.config
        if previous_edges.ndim != 5:
            raise ValueError(
                "previous_edges must be [B,P,2,H,W]; the per-camera support state is "
                f"created with that rank, got {tuple(previous_edges.shape)}"
            )
        if support_age.ndim != 5:
            raise ValueError(
                "support_age must be [B,P,2,H,W] like the edge field, got "
                f"{tuple(support_age.shape)}"
            )
        measured = valid.float()
        if measured.ndim == 3:  # [B,H,W]
            measured = measured[:, None, None]
        elif measured.ndim == 4:  # [B,P,H,W]
            measured = measured[:, :, None]
        elif measured.ndim != 5:
            raise ValueError(
                f"valid must be [B,H,W], [B,P,H,W] or [B,P,2,H,W], got {tuple(valid.shape)}"
            )
        measured = measured.expand_as(previous_edges)
        seen = measured > 0.5
        compatible = seen & (new_edges >= config.merge_threshold)
        refuted = seen & (new_edges < config.split_threshold)
        rising = previous_edges.abs() + (new_edges - previous_edges.abs()) * config.edge_alpha
        falling = previous_edges.abs() * (1.0 - config.edge_forget)
        if previous_open is None:
            previous_open = previous_edges >= config.merge_threshold
        candidate = torch.where(compatible, rising, torch.where(refuted, new_edges, falling))
        next_open = torch.where(
            previous_open, candidate >= config.split_threshold, candidate >= config.merge_threshold
        )
        edges = candidate
        edges = edges.clamp(0.0, 1.0)
        next_open = next_open & (edges >= config.split_threshold)
        self.last_open_edges = next_open
        age = torch.where(seen, support_age + 1.0, torch.zeros_like(support_age))
        merges = (next_open & ~previous_open).sum(dim=(1, 2, 3, 4))
        splits = (~next_open & previous_open).sum(dim=(1, 2, 3, 4))
        return edges, age, merges, splits

    # -- domains ------------------------------------------------------------
    def _coverage(self, state, slot: int) -> torch.Tensor:
        """[B,P,H,W] mask: the slot still holds a usable measurement."""
        batch = state.observed.shape[1]
        valid = state.valid[slot]
        height, width = state.shape
        usable = self._split_valid(valid.unsqueeze(0), 1, batch, (height, width))[0]
        observed = state.observed[slot][:, None, None, None]
        return usable & observed

    def _edge_coefficient(self, edges: torch.Tensor, row: int, col: int, radius: int):
        """Supported path weight of one offset, evaluated for every target.

        The weight is the product of the open edge weights along the local path
        from the candidate sample to the target, so it decays with the number of
        hops and can never pull in a connection outside the radius window.  The
        physical scaling ``1 / sqrt(2 pi r^2)`` of a two-dimensional Gaussian
        support is applied separately from the structural connectivity decision,
        which keeps a hard boundary visible instead of fading with the radius.
        """
        connected = self.open_path(edges, row, col)
        coefficient = self.path_weight(edges, row, col)
        # ``path_weight`` answers "is this sample connected"; the transmitted
        # weight still decays with the number of hops, so the same value can be
        # used for both without arithmetically multiplying by the indicator.
        support = coefficient.clamp(min=0.0, max=1.0) ** self.config.support_power
        support = torch.where(
            connected,
            support.clamp_min(self.config.support_floor),
            torch.zeros_like(support),
        )
        return support, connected

    def open_path(self, edges: torch.Tensor, row: int, col: int) -> torch.Tensor:
        """Whether every hop from the offset sample to the target is an open edge.

        The decision uses the open/closed state of the edges, not their strength,
        so a blocked edge separates the two sides completely.
        """
        open_edges = (edges >= self.config.split_threshold).to(edges.dtype)
        if row == 0 and col == 0:
            return torch.ones(
                (edges.shape[0], edges.shape[1], edges.shape[-2], edges.shape[-1]),
                dtype=torch.bool,
                device=edges.device,
            )
        return self.path_weight(open_edges, row, col) > 0.5

    def path_weight(self, edges: torch.Tensor, row: int, col: int) -> torch.Tensor:
        """Product of the local edge weights from the offset sample to the target.

        Offsets are limited to the (2r+1) window, so at most ``|row| + |col|``
        hops are multiplied; the result is exactly zero outside the grid.
        """
        batch, planes, _, height, width = edges.shape
        left = edges[:, :, 0]
        up = edges[:, :, 1]

        def shifted(field, delta_row, delta_col):
            """Field sampled at ``target + (delta_row, delta_col)``.

            The padding must cover the whole span of the path inside the window,
            otherwise a sample just outside the grid would read its in-grid
            neighbour and a genuine boundary would be lost.
            """
            pad_row = abs(delta_row) + abs(row) + 1
            pad_col = abs(delta_col) + abs(col) + 1
            padded = F.pad(field, (pad_col, pad_col, pad_row, pad_row), value=0.0)
            rows = torch.arange(height, device=field.device) + pad_row + delta_row
            cols = torch.arange(width, device=field.device) + pad_col + delta_col
            if int(rows.min()) < 0 or int(rows.max()) >= padded.shape[-2]:
                raise RuntimeError("support path left the padded field vertically")
            if int(cols.min()) < 0 or int(cols.max()) >= padded.shape[-1]:
                raise RuntimeError("support path left the padded field horizontally")
            return padded[:, :, rows[:, None], cols[None, :]]

        # The sample sits at ``target + (row, col)``.  Walking back to the target
        # uses the ``left`` edge of the positions to the right of the target
        # (deltas col .. 1 for col > 0) and the ``left`` edge of the positions at
        # or to the right of the target when the sample is on the left
        # (deltas col+1 .. 0 for col < 0).
        coefficient = torch.ones_like(left)
        if col > 0:
            for delta in range(col, 0, -1):
                coefficient = coefficient * shifted(left, row, delta)
        elif col < 0:
            for delta in range(col + 1, 1):
                coefficient = coefficient * shifted(left, row, delta)
        if row > 0:
            for delta in range(row, 0, -1):
                coefficient = coefficient * shifted(up, delta, 0)
        elif row < 0:
            for delta in range(row + 1, 1):
                coefficient = coefficient * shifted(up, delta, 0)
        return coefficient

    def _unfold_time(self, tensor: torch.Tensor, window: int) -> torch.Tensor:
        """[K,B,P,H,W] -> [K, B*P, H*W, window^2] gathered in offset-major order."""
        steps, batch, planes, height, width = tensor.shape
        flat = tensor.reshape(steps * batch * planes, 1, height, width)
        unfolded = F.unfold(flat, kernel_size=window, padding=window // 2)
        return unfolded.reshape(steps, batch * planes, height * width, -1)

    def _cumulative_time(self, tensor: torch.Tensor, window: int) -> torch.Tensor:
        return self._unfold_time(tensor, window).cumsum(dim=0)

    def domain_statistics(self, state, edges, motion=None, estimator=None, depths=None):
        """Integrate each bounded radius x time domain, once per source sample.

        Offsets are samples within a domain, not separate domains. Coefficients
        enter W/S linearly and variance/squared-weight sums quadratically.
        Counts are unweighted integers and exclude invalid or disconnected data.
        """
        if motion is not None and self.config.fractional_alignment:
            return self.fractional_statistics(
                state, edges, motion, tuple(self.config.time_depths if depths is None else depths)
            )
        values, variance, valid = self.aligned_slots(state, motion, estimator)
        n, batch, _, height, width = values.shape
        planes = self.config.planes
        x = self._split_planes(values, n, batch, (height, width))
        v = self._split_planes(variance, n, batch, (height, width))
        m = self._split_valid(valid, n, batch, (height, width))
        m = m & state.observed[:, :, None, None, None]
        ages = torch.nan_to_num(measurement_age(state), nan=0.0, posinf=0.0)
        depths = tuple(self.config.time_depths if depths is None else depths)
        chips = {
            key: []
            for key in (
                "mu",
                "variance_proxy",
                "age",
                "support_count",
                "unique_frames",
                "unique_samples",
                "weight_concentration",
            )
        }
        connections, radii, depth_ids = [], [], []
        window = self.window

        # Unfold preserves the original source identity (slot, plane, coordinate).
        def patches(tensor):
            flat = tensor.reshape(n * batch * planes, 1, height, width).float()
            return F.unfold(flat, window, padding=self.radius).reshape(
                n, batch, planes, window * window, height, width
            )

        xp, vp, mp = patches(x), patches(v), patches(m)
        for radius in self.config.spatial_radii:
            offsets = candidate_offsets(radius)
            indices = [(dy + self.radius) * window + dx + self.radius for dy, dx in offsets]
            coefficients = torch.stack(
                [self._edge_coefficient(edges, dy, dx, radius)[0] for dy, dx in offsets], dim=2
            )
            for depth in depths:
                allowed = (state.observed.long().cumsum(0) <= int(depth)) & state.observed
                usable = (mp[:, :, :, indices] > 0.5) & allowed[:, :, None, None, None, None]
                c = coefficients[None] * usable
                w = c / (vp[:, :, :, indices] + _EPS)
                total = w.sum(dim=(0, 3))
                denom = total.clamp_min(_EPS)
                chips["mu"].append((w * xp[:, :, :, indices]).sum(dim=(0, 3)) / denom)
                chips["variance_proxy"].append(
                    (w.square() * vp[:, :, :, indices]).sum(dim=(0, 3)) / (denom.square() + _EPS)
                )
                chips["age"].append(
                    (w * ages[:, :, None, None, None, None]).sum(dim=(0, 3)) / denom
                )
                actual = c > 0
                chips["support_count"].append(total)
                chips["unique_samples"].append(actual.sum(dim=(0, 3)).float())
                chips["unique_frames"].append(actual.any(dim=3).sum(0).float())
                chips["weight_concentration"].append(
                    total.square() / (w.square().sum(dim=(0, 3)) + _EPS)
                )
                connections.append((coefficients > 0).float().mean(dim=2))
                radii.append(radius)
                depth_ids.append(int(depth))
        device = values.device
        return DomainStatistics(
            **{key: torch.stack(value) for key, value in chips.items()},
            radius_index=torch.tensor(radii, device=device),
            depth_index=torch.tensor(depth_ids, device=device),
            offset_index=torch.zeros(len(radii), dtype=torch.long, device=device),
            offsets=[(0, 0)] * len(radii),
            depths=depths,
        ), torch.stack(connections)

    # -- forward ------------------------------------------------------------
    def forward(self, frame, state, motion=None, estimator=None, edges=None):
        """Return candidate statistics, connectivity and diagnostics."""
        if edges is None:
            edges, edge_info = self.edge_field(frame, state, motion=motion, estimator=estimator)
        else:
            edge_info = {"motion_unresolved_fraction": 1.0, "edge_mean": float(edges.mean())}
        statistics, connectivity = self.domain_statistics(
            state, edges, motion=motion, estimator=estimator
        )
        diagnostics = {
            **edge_info,
            "candidate_count": statistics.candidate_count,
            "mean_age": float(statistics.age.mean().detach()),
            "mean_support_count": float(statistics.support_count.mean().detach()),
            "mean_support_frames": float(statistics.unique_frames.mean().detach()),
        }
        return statistics, connectivity, edges, diagnostics

    # -- reference implementation ------------------------------------------
    def fractional_statistics(self, state, edges, motion, depths):
        """Combine interpolation paths by ORIGINAL source ID before squaring.

        Adjacent interpolated samples share source pixels. Squaring each path
        separately would invent independent evidence and understate variance.
        """
        values, variance, valid = state.values, state.variance, state.valid
        n, batch, _, h, w = values.shape
        planes = self.config.planes
        x = self._split_planes(values, n, batch, (h, w))
        v = self._split_planes(variance, n, batch, (h, w))
        m = self._split_valid(valid, n, batch, (h, w))
        ages = torch.nan_to_num(measurement_age(state), nan=0.0, posinf=0.0)
        chips = {
            key: []
            for key in (
                "mu",
                "variance_proxy",
                "age",
                "support_count",
                "unique_frames",
                "unique_samples",
                "weight_concentration",
            )
        }
        connections, radii, depth_ids = [], [], []
        rows = torch.arange(h, device=values.device)[None, :, None]
        cols = torch.arange(w, device=values.device)[None, None, :]
        for radius in self.config.spatial_radii:
            offsets = candidate_offsets(radius)
            path = {offset: self._edge_coefficient(edges, *offset, radius)[0] for offset in offsets}
            for depth in depths:
                template = torch.zeros(batch, planes, h, w, device=values.device)
                W, S, Q, A, W2, samples, frames = [template.clone() for _ in range(7)]
                rank = state.observed.long().cumsum(0)
                for slot in range(n):
                    if not bool(((rank[slot] <= int(depth)) & state.observed[slot]).any()):
                        continue
                    field = motion[slot]
                    fine = field.shift + (field.fractional if field.fractional is not None else 0.0)
                    base = fine.floor().long()
                    frac = fine - base
                    # The key is the offset from the member's floored warp. Two
                    # paths with the same key point at the same original sample.
                    coefficients = {}
                    for dy, dx in offsets:
                        for cy in (0, 1):
                            for cx in (0, 1):
                                alpha = frac[:, 0] if cy else 1 - frac[:, 0]
                                beta = frac[:, 1] if cx else 1 - frac[:, 1]
                                key = (dy + cy, dx + cx)
                                c = path[(dy, dx)] * (alpha * beta)[:, None, None, None]
                                coefficients[key] = coefficients.get(key, 0.0) + c
                    used_frame = torch.zeros_like(template, dtype=torch.bool)
                    for (dy, dx), c in coefficients.items():
                        rr, cc = (
                            rows + base[:, 0, None, None] + dy,
                            cols + base[:, 1, None, None] + dx,
                        )
                        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                        flat = (rr.clamp(0, h - 1) * w + cc.clamp(0, w - 1)).expand(batch, h, w)
                        index = flat.reshape(batch, 1, -1).expand(batch, planes, -1)

                        def gather(tensor, slot=slot, index=index):
                            return (
                                tensor[slot]
                                .reshape(batch, planes, -1)
                                .gather(2, index)
                                .reshape(batch, planes, h, w)
                            )

                        usable = gather(m) & inside[:, None] & field.valid[:, None, None, None]
                        usable &= ((rank[slot] <= int(depth)) & state.observed[slot])[
                            :, None, None, None
                        ]
                        c = c * usable
                        vv = gather(v)
                        weight = c / (vv + _EPS)
                        actual = c > 0
                        W = W + weight
                        S = S + weight * gather(x)
                        Q = Q + weight.square() * vv
                        W2 = W2 + weight.square()
                        A = A + weight * ages[slot, :, None, None, None]
                        samples = samples + actual
                        used_frame |= actual
                    frames = frames + used_frame
                denominator = W.clamp_min(_EPS)
                for key, value in (
                    ("mu", S / denominator),
                    ("variance_proxy", Q / (denominator.square() + _EPS)),
                    ("age", A / denominator),
                    ("support_count", W),
                    ("unique_frames", frames),
                    ("unique_samples", samples),
                    ("weight_concentration", W.square() / (W2 + _EPS)),
                ):
                    chips[key].append(value)
                connections.append(torch.stack([c > 0 for c in path.values()]).float().mean(0))
                radii.append(radius)
                depth_ids.append(int(depth))
        return DomainStatistics(
            **{key: torch.stack(value) for key, value in chips.items()},
            radius_index=torch.tensor(radii, device=values.device),
            depth_index=torch.tensor(depth_ids, device=values.device),
            offset_index=torch.zeros(len(radii), dtype=torch.long, device=values.device),
            offsets=[(0, 0)] * len(radii),
            depths=tuple(depths),
        ), torch.stack(connections)

    def reference_domains(self, state, edges, depths=None, motion=None, estimator=None):
        """Independent scalar source walk, including integer-warped observations."""
        values, variance, valid = self.aligned_slots(state, motion, estimator)
        n, batch, _, height, width = values.shape
        planes = self.config.planes
        x = self._split_planes(values, n, batch, (height, width))
        v = self._split_planes(variance, n, batch, (height, width))
        m = self._split_valid(valid, n, batch, (height, width))
        ages = measurement_age(state)
        depths = tuple(self.config.time_depths if depths is None else depths)
        chips = {
            key: []
            for key in (
                "mu",
                "variance_proxy",
                "age",
                "support_count",
                "unique_frames",
                "unique_samples",
                "weight_concentration",
            )
        }
        radii, depth_ids = [], []
        for radius in self.config.spatial_radii:
            offsets = candidate_offsets(radius)
            coefficients = {
                offset: self._edge_coefficient(edges, *offset, radius)[0] for offset in offsets
            }
            for depth in depths:
                chip = {
                    key: torch.zeros(batch, planes, height, width, device=values.device)
                    for key in chips
                }
                for b in range(batch):
                    slots = state.observed[:, b].nonzero().flatten().tolist()[: int(depth)]
                    for plane in range(planes):
                        for row in range(height):
                            for col in range(width):
                                samples = []
                                used_frames = set()
                                for slot in slots:
                                    for dy, dx in offsets:
                                        rr, cc = row + dy, col + dx
                                        if not (0 <= rr < height and 0 <= cc < width):
                                            continue
                                        c = float(coefficients[(dy, dx)][b, plane, row, col])
                                        if c <= 0 or not bool(m[slot, b, plane, rr, cc]):
                                            continue
                                        vv = float(v[slot, b, plane, rr, cc])
                                        w = c / (vv + _EPS)
                                        samples.append(
                                            (
                                                w,
                                                float(x[slot, b, plane, rr, cc]),
                                                vv,
                                                float(ages[slot, b]),
                                            )
                                        )
                                        used_frames.add(slot)
                                W = sum(a[0] for a in samples)
                                if not W:
                                    continue
                                index = (b, plane, row, col)
                                chip["mu"][index] = sum(w * xx for w, xx, vv, age in samples) / W
                                chip["variance_proxy"][index] = sum(
                                    w * w * vv for w, xx, vv, age in samples
                                ) / (W * W + _EPS)
                                chip["age"][index] = sum(w * age for w, xx, vv, age in samples) / W
                                chip["support_count"][index] = W
                                chip["unique_samples"][index] = len(samples)
                                chip["unique_frames"][index] = len(used_frames)
                                chip["weight_concentration"][index] = (
                                    W * W / (sum(w * w for w, xx, vv, age in samples) + _EPS)
                                )
                for key in chips:
                    chips[key].append(chip[key])
                radii.append(radius)
                depth_ids.append(int(depth))
        return DomainStatistics(
            **{key: torch.stack(value) for key, value in chips.items()},
            radius_index=torch.tensor(radii, device=values.device),
            depth_index=torch.tensor(depth_ids, device=values.device),
            offset_index=torch.zeros(len(radii), dtype=torch.long, device=values.device),
            offsets=[(0, 0)] * len(radii),
            depths=depths,
        )


def _downsample_edges(edges: torch.Tensor, planes: int) -> torch.Tensor:
    """[B,P,2,H,W] -> [P,B,2,H,W] for the reference walk."""
    return edges.permute(1, 0, 2, 3, 4).contiguous()


def _components(edges: torch.Tensor, threshold: float) -> torch.Tensor:
    """Label one-hop connected components on the (left, up) edge graph.

    ``edges`` is ``[P,B,2,H,W]`` with channel 0 the left neighbour and channel 1
    the up neighbour.  Only existing one-hop adjacencies are used, so a component
    never acquires an arbitrary transitive connection; the radius window is
    applied separately when the source samples are read.
    """
    planes, batch, _, height, width = edges.shape
    labels = torch.full((planes, batch, height, width), -1, dtype=torch.int64)
    open_edge = edges >= threshold
    for plane in range(planes):
        for sample in range(batch):
            next_label = 0
            for start_row in range(height):
                for start_col in range(width):
                    if labels[plane, sample, start_row, start_col] >= 0:
                        continue
                    labels[plane, sample, start_row, start_col] = next_label
                    stack = [(start_row, start_col)]
                    while stack:
                        row, col = stack.pop()
                        if col + 1 < width and bool(open_edge[plane, sample, 0, row, col + 1]):
                            if labels[plane, sample, row, col + 1] < 0:
                                labels[plane, sample, row, col + 1] = next_label
                                stack.append((row, col + 1))
                        if col - 1 >= 0 and bool(open_edge[plane, sample, 0, row, col]):
                            if labels[plane, sample, row, col - 1] < 0:
                                labels[plane, sample, row, col - 1] = next_label
                                stack.append((row, col - 1))
                        if row + 1 < height and bool(open_edge[plane, sample, 1, row + 1, col]):
                            if labels[plane, sample, row + 1, col] < 0:
                                labels[plane, sample, row + 1, col] = next_label
                                stack.append((row + 1, col))
                        if row - 1 >= 0 and bool(open_edge[plane, sample, 1, row, col]):
                            if labels[plane, sample, row - 1, col] < 0:
                                labels[plane, sample, row - 1, col] = next_label
                                stack.append((row - 1, col))
                    next_label += 1
    return labels


def _window_members(labels, plane, sample, label, inner_row, inner_col, radius):
    """Members of one component inside the bounded radius window around a sample."""
    _, _, height, width = labels.shape
    members = []
    for row in range(max(0, inner_row - radius), min(height, inner_row + radius + 1)):
        for col in range(max(0, inner_col - radius), min(width, inner_col + radius + 1)):
            if int(labels[plane, sample, row, col]) == int(label):
                members.append((row, col))
    return members


def edge_field_reference(values, variance, valid, config: N01Config) -> torch.Tensor:
    """Explicit per-pixel analytic gate for one frame; written for hand-checking.

    ``values`` is ``[C,H,W]``.  It reproduces the pair gate of
    ``SupportDomainIntegrator.edge_field`` by direct indexing, without unfolding.
    """
    channels, height, width = values.shape
    planes = config.planes
    if planes == 1:
        plane_values = values.mean(dim=0, keepdim=True)
        plane_variance = variance.mean(dim=0, keepdim=True)
        plane_valid = (valid > 0.5).all(dim=0, keepdim=True)
    else:
        plane_values = values
        plane_variance = variance
        plane_valid = valid > 0.5
    edges = torch.zeros((planes, 2, height, width), dtype=values.dtype, device=values.device)
    for plane in range(planes):
        for row in range(height):
            for col in range(width):
                for direction, (offset_row, offset_col) in enumerate(((0, -1), (-1, 0))):
                    inner_row = row + offset_row
                    inner_col = col + offset_col
                    if not (0 <= inner_row < height and 0 <= inner_col < width):
                        continue
                    if not (
                        plane_valid[plane, row, col] and plane_valid[plane, inner_row, inner_col]
                    ):
                        continue
                    centre = float(plane_values[plane, row, col])
                    neighbour = float(plane_values[plane, inner_row, inner_col])
                    sigma = (
                        float(plane_variance[plane, row, col])
                        + float(plane_variance[plane, inner_row, inner_col])
                        + _EPS
                    ) ** 0.5
                    innovation = abs(centre - neighbour) / sigma
                    gradient = abs(centre - neighbour)
                    edge_effect = gradient / (gradient + sigma + _EPS)
                    edges[plane, direction, row, col] = float(
                        torch.exp(
                            torch.tensor(
                                -(innovation**2) / (config.edge_temperature * 2.0)
                                - edge_effect / max(config.edge_beta, _EPS)
                            )
                        )
                    )
    radius = max(config.spatial_radii)
    return edges * (math.exp(-1 / (2 * radius**2)) if radius else 1.0)

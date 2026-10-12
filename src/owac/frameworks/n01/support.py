"""N01 support graph, merge/split and per-source statistics.

This module is the definition site of the hand-computable reference solution.

Definitions (all causal; only the current and past frames are used)
------------------------------------------------------------------
* A **support domain** for target position ``p`` is the Cartesian product of a
  spatial radius ``r`` in ``spatial_radii``, a temporal length ``D`` in
  ``time_depths`` and one spatial offset inside the ``(2r+1) x (2r+1)`` window.
  Candidates are enumerated row-major, radius outer, depth middle, offset inner.
* The same-plane one-hop support graph uses the (left, up) neighbour pairs at
  their true sampling positions.  ``merge_threshold > split_threshold`` is the
  hysteresis band: an edge opens only after sustained compatibility, and a new
  significant refutation, motion inconsistency or occlusion closes it at once.
* Per-sample coefficient ``w_i = m_i * c_i / (v_i + eps)`` with
  ``S = sum w_i x_i``, ``W = sum w_i``, ``mu = S/(W+eps)``,
  ``V_diag = sum w_i^2 v_i / (W^2+eps)`` and ``age = sum w_i (t - t_i)/(W+eps)``.
* ``connected_power`` is the supported path weight: the product of the open edge
  weights along the local path inside the window, evaluated by separability from
  the target position.  It decays with the number of hops and stays bounded by
  the window, so no arbitrary transitive connection is pulled in.
  ``reference_domains`` replaces it with the hard one-hop connectivity
  indicator of the same component; both use the identical statistic formulas and
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
            separation = spacing * math.sqrt(2.0)
            gate = math.exp(-(separation**2) / (2.0 * float(decay) ** 2))
        gate = max(min(gate, 1.0), 0.0)
        return torch.full(
            (batch, planes, 2, height, width), gate, device=current.device, dtype=current.dtype
        )

    def _motion_gate(self, motion, batch: int, current: torch.Tensor):
        config = self.config
        dtype, device = current.dtype, current.device
        if motion is None:
            return torch.ones((batch, 1, 1, 1), device=device, dtype=dtype), 1.0
        head = motion[0] if isinstance(motion, (list, tuple)) else motion
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

    def update_edges(self, previous_edges, support_age, new_edges, valid):
        """Hysteresis update; returns edges, per-pair age and merge/split counts.

        ``merge_threshold`` is the barrier an edge must cross to become open, and
        once open it only closes below ``split_threshold``.  Three regimes:

        * compatible evidence (``new_edges >= merge_threshold``): the edge rises
          towards the measured compatibility by ``edge_alpha``, so it needs
          sustained agreement before it opens;
        * measured refutation (``new_edges < split_threshold``): the edge falls
          immediately by ``edge_forget`` instead of waiting for the slow path;
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
        rising = previous_edges + (new_edges - previous_edges) * config.edge_alpha
        falling = previous_edges * (1.0 - config.edge_forget)
        edges = torch.where(compatible, rising, torch.where(refuted, falling, falling))
        edges = edges.clamp(0.0, 1.0)
        previous_open = previous_edges >= config.merge_threshold
        next_open = edges >= config.merge_threshold
        age = torch.where(seen, support_age + 1.0, support_age)
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
                coefficient = coefficient * shifted(up, delta, col)
        elif row < 0:
            for delta in range(row + 1, 1):
                coefficient = coefficient * shifted(up, delta, col)
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
        """Vectorised statistics of every candidate domain (the fast path).

        ``x`` is the plane-collapsed calibrated measurement, ``v`` the noise
        variance, ``c`` the support coefficient and ``score = 1/(v+eps)``; the
        documented update ``w = m * c * score`` is applied once per candidate
        domain.  Depth ``D`` covers the last ``D`` slots that actually hold a
        unique measurement, so a repeated frame shrinks the window instead of
        silently counting a duplicate as evidence.
        """
        config = self.config
        values, variance, valid = self.aligned_slots(state, motion, estimator)
        steps, batch = values.shape[0], values.shape[1]
        height, width = state.shape
        planes = config.planes
        plane_values = self._split_planes(values, steps, batch, (height, width))
        plane_variance = self._split_planes(variance, steps, batch, (height, width))
        plane_valid = self._split_valid(valid, steps, batch, (height, width)).float()

        depths = tuple(int(item) for item in (config.time_depths if depths is None else depths))
        # Every slot is walked: a slot without a measurement contributes nothing
        # (its coverage mask is zero), and walking all of them keeps the depth
        # arithmetic independent of where the repeated frames happen to sit.  The
        # buffer holds at most ``max_history`` slots, so this stays bounded.
        observed = state.observed
        # The rank table below is built over these slots; a depth beyond the deepest
        # available observation is clamped when the depth is resolved.
        max_depth = min(int(max(depths)), steps)
        window = self.window
        area = height * width
        kernel = window * window

        coverage = torch.stack(
            [self._coverage(state, slot).float() for slot in range(max_depth)], dim=0
        )
        age = measurement_age(state)[:max_depth]
        age = torch.nan_to_num(age, nan=0.0, posinf=0.0)
        age = age[:, :, None, None, None].expand(-1, -1, planes, -1, -1)

        active = plane_valid[:max_depth] * coverage
        score = active / (plane_variance[:max_depth] + _EPS)
        x = plane_values[:max_depth]

        # The cumulative tensors carry a leading zero, so index D is the prefix
        # over the first D walked slots.
        leading = torch.zeros_like(score[:1])
        num = self._cumulative_time(torch.cat([leading, score * x], dim=0), window)
        square_variance = self._cumulative_time(
            torch.cat([leading, (score**2) * plane_variance[:max_depth]], dim=0), window
        )
        age_mass = self._cumulative_time(torch.cat([leading, score * age], dim=0), window)
        square = self._cumulative_time(torch.cat([leading, score**2], dim=0), window)
        mass = self._cumulative_time(torch.cat([leading, score], dim=0), window)
        samples = self._cumulative_time(torch.cat([leading, (score > 0).float()], dim=0), window)
        frames = self._cumulative_time(torch.cat([leading, active], dim=0), window)

        # Depth D maps onto the D-th slot that actually holds a unique
        # measurement, whether or not slot 0 holds one.  ``rank_to_slot[r]`` is the
        # slot index of the r-th observation, so a depth is resolved by the rank of
        # the last observation it covers.
        observed_prefix = observed[:max_depth].cpu().numpy().astype("int64")
        rank_source = observed_prefix.cumsum(axis=0)
        rank = torch.as_tensor(rank_source - 1, device=values.device, dtype=torch.int64)
        rank = torch.where(
            torch.as_tensor(observed_prefix > 0, device=values.device, dtype=torch.bool),
            rank,
            torch.zeros_like(rank),
        )
        ranks = rank.max(dim=1).values  # walked slots per batch member
        observed_mask = observed[:max_depth]
        rank_to_slot = []
        for member_index in range(batch):
            column = observed_mask[:, member_index]
            slots = torch.nonzero(column, as_tuple=False).squeeze(1)
            rank_to_slot.append(slots)
        max_rank = int(ranks.max().item()) if ranks.numel() else 0

        statistics = {
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
        connectivity, radius_index, depth_index, offset_index = [], [], [], []
        plane_count = batch * planes
        pixel = torch.arange(area, device=values.device)

        slot_count = max_depth + 1  # the leading zero makes the first axis D+1

        def patch(cumulative, rank_index, flat_index, coefficient):
            """Prefix through the ``rank_index``-th walked slot for one offset."""
            # Resolve the depth rank to the slot of the last observation covered.
            wanted = min(int(rank_index), max_rank)
            slot = torch.zeros((batch,), dtype=torch.int64, device=values.device)
            for member_index, slots in enumerate(rank_to_slot):
                if slots.numel():
                    slot[member_index] = slots[min(wanted, slots.numel() - 1)]
            # The unfold output folds batch and plane into one axis (batch-major),
            # so the per-batch slot index is repeated once per CFA plane.
            if planes > 1:
                slot = slot[:, None].expand(batch, planes).reshape(plane_count)
            # cumsum carries a leading zero, so the prefix through slot s is entry
            # s + 1, clamped to the deepest walked slot.
            slot = (slot + 1).clamp(max=slot_count - 1)
            # ``cumulative`` is the ordinary unfold output [T, B*P, H*W*K] with a
            # leading zero, so entry ``slot`` already is the prefix through walked
            # slot ``slot - 1`` for every batch-plane member.
            flat = cumulative.reshape(cumulative.shape[0], plane_count, area * kernel)
            slot_index = slot.reshape(1, plane_count, 1).expand(1, plane_count, area * kernel)
            selected = torch.gather(flat, 0, slot_index)[0]
            base = flat_index * area
            offset_index = (base + pixel)[None, :].expand(plane_count, area)
            return torch.gather(selected, 1, offset_index) * coefficient

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, planes, height, width)

        for radius in config.spatial_radii:
            offsets = candidate_offsets(radius)
            for depth in depths:
                clipped = min(int(depth), steps)
                for local_offset, (row, col) in enumerate(offsets):
                    flat_index = (row + self.radius) * window + (col + self.radius)
                    support, connected = self._edge_coefficient(edges, row, col, radius)
                    coefficient = support.reshape(plane_count, area)
                    count = patch(mass, clipped - 1, flat_index, coefficient)
                    denominator_mass = count.clamp_min(_EPS)
                    mu = patch(num, clipped - 1, flat_index, coefficient) / denominator_mass
                    variance_proxy = patch(
                        square_variance, clipped - 1, flat_index, coefficient
                    ) / (denominator_mass**2 + _EPS)
                    mean_age = (
                        patch(age_mass, clipped - 1, flat_index, coefficient) / denominator_mass
                    )
                    concentration = count**2 / (
                        patch(square, clipped - 1, flat_index, coefficient) + _EPS
                    )
                    statistics["mu"].append(reshape(mu))
                    statistics["variance_proxy"].append(reshape(variance_proxy))
                    statistics["age"].append(reshape(mean_age))
                    statistics["support_count"].append(reshape(count))
                    statistics["unique_frames"].append(
                        reshape(patch(frames, clipped - 1, flat_index, coefficient))
                    )
                    statistics["unique_samples"].append(
                        reshape(patch(samples, clipped - 1, flat_index, coefficient))
                    )
                    statistics["weight_concentration"].append(reshape(concentration))
                    connectivity.append(connected.reshape(batch, planes, height, width).float())
                    radius_index.append(radius)
                    depth_index.append(int(depth))
                    offset_index.append(local_offset)
        device = values.device
        return (
            DomainStatistics(
                mu=torch.stack(statistics["mu"], dim=0),
                variance_proxy=torch.stack(statistics["variance_proxy"], dim=0),
                age=torch.stack(statistics["age"], dim=0),
                support_count=torch.stack(statistics["support_count"], dim=0),
                unique_frames=torch.stack(statistics["unique_frames"], dim=0),
                unique_samples=torch.stack(statistics["unique_samples"], dim=0),
                weight_concentration=torch.stack(statistics["weight_concentration"], dim=0),
                radius_index=torch.tensor(radius_index, device=device, dtype=torch.int64),
                depth_index=torch.tensor(depth_index, device=device, dtype=torch.int64),
                offset_index=torch.tensor(offset_index, device=device, dtype=torch.int64),
                offsets=[
                    item for radius in config.spatial_radii for item in candidate_offsets(radius)
                ],
                depths=depths,
            ),
            torch.stack(connectivity, dim=0),
        )

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
    def reference_domains(self, state, edges, depths=None, motion=None, estimator=None):
        """Per-source Python reference solution for the same candidate domains.

        Deliberately explicit and slow: each candidate domain is rebuilt from the
        unique observation buffer by walking its source samples with the identical
        ``w = m * c / (v + eps)`` formula and the identical supported path weight
        of ``domain_statistics``.  The two implementations are compared
        numerically in ``tests/encoders/n01/test_support.py``; this one is the
        definition of record for small, hand-checkable inputs.
        """
        config = self.config
        if motion is not None and estimator is not None and config.alignment != "none":
            raise NotImplementedError("the reference solution integrates unaligned frames only")
        values = state.values
        variance = state.variance
        valid = state.valid
        slots, batch = values.shape[0], values.shape[1]
        height, width = state.shape
        planes = config.planes
        age_source = torch.nan_to_num(measurement_age(state), nan=0.0, posinf=0.0)
        observed = state.observed
        steps = max(1, min(slots, int(state.filled.max().item()) or 1))
        depths = tuple(int(item) for item in (config.time_depths if depths is None else depths))

        def sample(slot, index, plane, row, col):
            if planes == 1:
                return (
                    float(values[slot, index, :, row, col].mean()),
                    float(variance[slot, index, :, row, col].mean()),
                    bool((valid[slot, index, :, row, col] > 0.5).all()),
                )
            return (
                float(values[slot, index, plane, row, col]),
                float(variance[slot, index, plane, row, col]),
                bool(valid[slot, index, plane, row, col] > 0.5),
            )

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
        radius_index, depth_index, offset_index = [], [], []
        for plane in range(planes):
            for radius in config.spatial_radii:
                offsets = candidate_offsets(radius)
                for depth in depths:
                    clipped = min(int(depth), steps)
                    for offset_index_local, (row, col) in enumerate(offsets):
                        chip = {
                            key: torch.zeros((batch, height, width), dtype=values.dtype)
                            for key in chips
                        }
                        for index in range(batch):
                            for inner_row in range(height):
                                for inner_col in range(width):
                                    target_row = inner_row + row
                                    target_col = inner_col + col
                                    if not (0 <= target_row < height and 0 <= target_col < width):
                                        continue
                                    # The support relation is indexed by the
                                    # *target* position: it is the weight of the
                                    # path from the offset sample to that target.
                                    coefficient = float(
                                        self.path_weight(edges, row, col)[
                                            index, plane, target_row, target_col
                                        ]
                                    )
                                    if coefficient <= 0.0:
                                        continue
                                    if not bool(
                                        self.open_path(edges, row, col)[
                                            index, plane, target_row, target_col
                                        ]
                                    ):
                                        continue
                                    coefficient = max(
                                        coefficient**config.support_power, config.support_floor
                                    )
                                    numerator = 0.0
                                    denominator = 0.0
                                    square_mass = 0.0
                                    variance_mass = 0.0
                                    age_mass = 0.0
                                    unique_frames = 0
                                    unique_samples = 0
                                    # Depth D covers the last D slots that hold a
                                    # unique measurement, so a repeated frame
                                    # shrinks the window instead of adding evidence.
                                    used = 0
                                    for slot in range(slots):
                                        if used >= clipped:
                                            break
                                        if not bool(observed[slot, index]):
                                            continue
                                        used += 1
                                        value, variance_value, usable = sample(
                                            slot, index, plane, inner_row, inner_col
                                        )
                                        if not usable:
                                            continue
                                        weight = coefficient / (variance_value + _EPS)
                                        numerator += weight * value
                                        denominator += weight
                                        variance_mass += (weight**2) * variance_value
                                        square_mass += weight**2
                                        age_mass += weight * float(age_source[slot, index])
                                        unique_samples += 1
                                        unique_frames += 1
                                    if denominator <= 0.0:
                                        continue
                                    chip["mu"][index, target_row, target_col] = (
                                        numerator / denominator
                                    )
                                    chip["variance_proxy"][index, target_row, target_col] = (
                                        variance_mass / (denominator**2 + _EPS)
                                    )
                                    chip["age"][index, target_row, target_col] = (
                                        age_mass / denominator
                                    )
                                    chip["support_count"][index, target_row, target_col] = (
                                        denominator
                                    )
                                    chip["weight_concentration"][index, target_row, target_col] = (
                                        denominator** 2 / (square_mass + _EPS)
                                    )
                                    chip["unique_frames"][index, target_row, target_col] = float(
                                        unique_frames
                                    )
                                    chip["unique_samples"][index, target_row, target_col] = float(
                                        unique_samples
                                    )
                        for key in chips:
                            chips[key].append(chip[key])
                        radius_index.append(radius)
                        depth_index.append(int(depth))
                        offset_index.append(offset_index_local)
        device = values.device
        return DomainStatistics(
            mu=torch.stack(chips["mu"], dim=0),
            variance_proxy=torch.stack(chips["variance_proxy"], dim=0),
            age=torch.stack(chips["age"], dim=0),
            support_count=torch.stack(chips["support_count"], dim=0),
            unique_frames=torch.stack(chips["unique_frames"], dim=0),
            unique_samples=torch.stack(chips["unique_samples"], dim=0),
            weight_concentration=torch.stack(chips["weight_concentration"], dim=0),
            radius_index=torch.tensor(radius_index, device=device, dtype=torch.int64),
            depth_index=torch.tensor(depth_index, device=device, dtype=torch.int64),
            offset_index=torch.tensor(offset_index, device=device, dtype=torch.int64),
            offsets=[item for radius in config.spatial_radii for item in candidate_offsets(radius)],
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
        plane_values = values.mean(dim=0)
        plane_variance = variance.mean(dim=0)
        plane_valid = (valid > 0.5).all(dim=0)
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
                    if not (plane_valid[row, col] and plane_valid[inner_row, inner_col]):
                        continue
                    centre = float(plane_values[row, col])
                    neighbour = float(plane_values[inner_row, inner_col])
                    sigma = (
                        float(plane_variance[row, col])
                        + float(plane_variance[inner_row, inner_col])
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
    return edges

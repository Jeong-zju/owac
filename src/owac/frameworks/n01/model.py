"""N01 encoder: measurement stem, support-domain integrator, evidence blocks, readout.

Order for the first reference implementation::

    MeasurementStem -> SupportDomainIntegrator -> EvidenceBlock x num_blocks -> SpatialReadout

The stem uses 3x3 small convolutions with GroupNorm and consumes RAW values,
sampling coordinates and metadata indicators; BatchNorm is not used because its
statistics would depend on the cross-time batch composition.  An EvidenceBlock
performs local exchange over the support graph, retain/decay gating and a feature
update; explicit statistics and learned features are kept in separate channels so
that a feature magnitude is never read as a confidence.  The readout turns the
field into backend tokens at the shared budget (256 tokens x 2048 dims by
default, distinct smaller CPU test configurations are named separately).

The selector chooses a support domain per position.  Its analytic cost is::

    J(domain) = lambda_noise * uncertainty_proxy
              + lambda_edge  * boundary_mixing_proxy
              + lambda_age   * mean_measurement_age
              + lambda_motion * correspondence_error

Only information available at the current time enters the cost.  Task precision
or a reaction deadline participate only when they are genuinely provided as a
public task requirement; the target truth position is never an input.  Cost
weights are normalised with training-set statistics and are chosen on validation
only.  The learned variant initialises from the analytic choice and learns domain
logits from localisation/action losses; the analytic variant is always kept.

Budget conditioning reduces the candidate radius, the time depth, or the number
of update blocks; the last valid representation is returned and the number of
performed updates is recorded.  The first round computes every candidate in
order to choose, so the selection cost is counted in the measured cost rather
than being called a saving.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.encoder_contract import ComputeBudget, EncoderOutput, MeasurementFrame, VisualTokens
from .config import N01Config
from .motion import AlignmentMode, CausalCorrespondence, MotionChain, MotionField
from .state import N01State, state_bytes
from .state import detach as detach_state_fn
from .state import init_state as init_state_fn
from .state import push as push_fn
from .state import reset as reset_fn
from .support import SupportDomainIntegrator, candidate_offsets

__all__ = ["N01Encoder", "EvidenceBlock", "SupportSelector", "MeasurementStem", "SpatialReadout"]

STATISTIC_NAMES = (
    "mu",
    "variance_proxy",
    "age",
    "support_count",
    "unique_frames",
    "weight_concentration",
)


def statistics_channels(config: N01Config) -> int:
    """Channels of the explicit statistics block fed to every block and head.

    Six explicit statistics are kept per position and stay separate from the
    learned features, so a feature magnitude is never read as a confidence.  On a
    2x2 CFA the four planes are reduced here for the exchange block only; the
    integrator and its support graph still treat each plane as its own sampling
    grid (``SupportDomainIntegrator.resolution``), and the measurement channels
    still enter the stem plane by plane.
    """
    del config
    return len(STATISTIC_NAMES)


class MeasurementStem(nn.Module):
    """Local RAW stem: measurement channels plus sampling/metadata indicators."""

    def __init__(self, config: N01Config):
        super().__init__()
        self.config = config
        # Measurement channels plus two explicit measurement-statistics
        # channels (log1p variance and validity), never a synthetic image.
        metadata_channels = 2 if config.planes == 1 else 0
        self.in_channels = config.channels + metadata_channels
        self.body = nn.Sequential(
            nn.Conv2d(self.in_channels, config.width, 3, padding=1),
            nn.GroupNorm(_groups(config.width), config.width),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.width, config.width, 3, padding=1),
            nn.GroupNorm(_groups(config.width), config.width),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor, variance: torch.Tensor):
        if values.shape[1] != self.config.channels:
            raise ValueError(
                f"measurement stem expects {self.config.channels} channels, got {values.shape[1]}"
            )
        inputs = [values]
        if self.config.planes == 1:
            # The measurement grid shares one plane, so the noise level and the
            # validity are summarised per spatial position instead of being
            # repeated for every measurement channel.
            inputs.append(torch.log1p(variance.mean(dim=1, keepdim=True)))
            inputs.append(valid.float().mean(dim=1, keepdim=True))
        stacked = torch.cat(inputs, dim=1)
        if stacked.shape[1] != self.in_channels:
            raise ValueError(
                f"measurement stem received {stacked.shape[1]} channels, "
                f"expected {self.in_channels}"
            )
        return self.body(stacked)


class EvidenceBlock(nn.Module):
    """Local exchange over the support graph, retain/decay and feature update."""

    def __init__(self, config: N01Config):
        super().__init__()
        self.config = config
        width = config.width
        self.exchange = nn.Conv2d(width, width, 3, padding=1, groups=_groups(width))
        self.gate = nn.Sequential(
            nn.Conv2d(width + statistics_channels(config), width, 1),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.GroupNorm(_groups(width), width)
        self.graph = SupportDomainIntegrator(config) if False else None

    def forward(self, features, statistics, connectivity, update_mask=None):
        """``features`` [B,W,H,W]; ``statistics`` [B,S,H,W]; ``connectivity`` [B,K,H,W]."""
        graph = connectivity
        if graph is not None and graph.numel():
            weight = graph.mean(dim=1, keepdim=True)
            exchanged = self.exchange(features) * weight
            mixed = features + exchanged
        else:
            mixed = features
        gate_input = torch.cat([mixed, statistics], dim=1)
        retain = self.gate(gate_input)
        proposal = self.update(mixed)
        updated = self.norm(retain * proposal + (1.0 - retain) * mixed)
        activated = F.relu(mixed + updated)
        if update_mask is not None:
            activated = torch.where(update_mask[:, None, None, None], activated, mixed)
        return activated


class SupportSelector(nn.Module):
    """Analytic cost-based domain selection with an optional learned refinement."""

    def __init__(self, config: N01Config):
        super().__init__()
        self.config = config
        self.learned = config.selection == "learned" or config.selection_use_learned
        self.logit_head: nn.Module | None = None
        if self.learned:
            self.logit_head = nn.Sequential(
                nn.Conv2d(statistics_channels(config), config.width, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.width, 1, 1),
            )
            nn.init.zeros_(self.logit_head[-1].weight)
            nn.init.zeros_(self.logit_head[-1].bias)

    def cost(self, statistics: dict, scale: dict, precision: float | None = None):
        """Normalised analytic cost per candidate domain."""
        weights = self.config.cost_weights

        def broadcast(key: str) -> torch.Tensor:
            """Align a per-batch scale with dimension 1 of a candidate tensor."""
            tensor = statistics[key]
            shape = [1] * tensor.ndim
            shape[1] = scale[key].numel()
            return scale[key].reshape(shape)

        def divide(key: str) -> torch.Tensor:
            return statistics[key] / broadcast(key)

        uncertainty = divide("variance_proxy")
        boundary = 1.0 - statistics["connectivity"]
        age = divide("age")
        motion = divide("motion_error")
        cost = (
            weights["noise"] * uncertainty
            + weights["edge"] * boundary
            + weights["age"] * age
            + weights["motion"] * motion
        )
        if precision is not None:
            # Only a genuinely published task precision enters here (the reaction
            # deadline / localisation tolerance of the task).  The target truth
            # position is never an input.
            cost = cost + precision * boundary
        return cost

    def normalised_features(self, statistics: dict, scale: dict):
        """Logit features, normalised with the candidate statistics at this step.

        A training-set calibration can be substituted on validation through
        ``scale``; nothing here can see a target truth position.
        """

        def unit(key: str) -> torch.Tensor:
            tensor = statistics[key]
            shape = [1] * tensor.ndim
            shape[1] = scale[key].numel()
            return tensor / scale[key].reshape(shape)

        return torch.cat(
            [
                unit("variance_proxy"),
                unit("age"),
                unit("support_count"),
                unit("unique_frames"),
                unit("motion_error"),
                statistics["weight_concentration"],
                unit("mu"),
            ],
            dim=1,
        )

    def forward(self, statistics, scale, budget_slice=None):
        """Return per-position selection weights and the chosen candidate index."""
        if budget_slice is None:
            raise ValueError("SupportSelector requires the candidate slice under the budget")
        cost = self.cost(statistics, scale)[budget_slice]
        logits = -cost
        if self.logit_head is not None:
            features = self.normalised_features(statistics, scale)[budget_slice]
            logits = logits + self.logit_head(features).squeeze(1)
        # A sharp softmax keeps the mixture close to the analytic choice while
        # staying differentiable; the participation ratio is reported so the
        # effective number of contributing domains is never implicit.
        weights = torch.softmax(logits / self.config.selector_temperature, dim=0)
        index = logits.argmax(dim=0)
        return weights, index, cost


class SpatialReadout(nn.Module):
    """Turn the representation field into backend tokens at a shared budget."""

    def __init__(self, config: N01Config):
        super().__init__()
        self.config = config
        self.project = nn.Conv2d(config.width, config.readout_width, 1)
        self.position = nn.Conv2d(2, config.readout_width, 1)
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)

    def forward(self, features, valid, token_budget=None):
        budget = int(self.config.readout_tokens if token_budget is None else token_budget)
        batch, _, height, width = features.shape
        grid = min(budget, height * width)
        tokens = self.project(features)
        rows = torch.linspace(-1.0, 1.0, height, device=features.device)
        cols = torch.linspace(-1.0, 1.0, width, device=features.device)
        coords = torch.stack(
            [cols[None, :].expand(height, width), rows[:, None].expand(height, width)], dim=0
        )[None].expand(batch, -1, -1, -1)
        tokens = tokens + self.position(coords)
        flat = tokens.reshape(batch, self.config.readout_width, height * width).transpose(1, 2)
        mask_source = valid.reshape(batch, -1)
        if grid < height * width:
            positions = (
                torch.linspace(0, height * width - 1, grid, device=features.device)
                .round()
                .to(torch.int64)
            )
            flat = flat[:, positions]
            mask_source = mask_source[:, positions]
        positions_xy = coords.reshape(batch, 2, height * width).transpose(1, 2)
        if grid < height * width:
            flat_positions = positions_xy[
                :,
                torch.linspace(0, height * width - 1, grid, device=features.device)
                .round()
                .to(torch.int64),
            ]
        else:
            flat_positions = positions_xy
        return VisualTokens(
            values=flat,
            mask=mask_source,
            camera_ids=[],
            positions=flat_positions,
        )


def _groups(width: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if width % candidate == 0:
            return candidate
    return 1


@dataclass
class _CandidateScale:
    variance_proxy: float = 1.0
    age: float = 1.0
    motion_error: float = 1.0


class N01Encoder(nn.Module):
    """Support-domain-reorganisable multi-timescale measurement state field."""

    def __init__(self, config: N01Config):
        super().__init__()
        config.validate()
        self.config = config
        self.stem = MeasurementStem(config)
        self.integrator = SupportDomainIntegrator(config)
        self.blocks = nn.ModuleList([EvidenceBlock(config) for _ in range(config.num_blocks)])
        self.selector = SupportSelector(config)
        self.readout_layer = SpatialReadout(config)
        self.measurement_head: nn.Module | None = None
        if config.aux_measurement_prediction:
            self.measurement_head = nn.Sequential(
                nn.Conv2d(config.width, config.width, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.width, config.channels, 1),
            )
        self.correspondence: CausalCorrespondence | None = None
        if config.alignment != AlignmentMode.NONE.value:
            self.correspondence = CausalCorrespondence(config)
        self.chain = MotionChain(self.correspondence)
        self.register_buffer("_cost_scale", torch.tensor([1.0, 1.0, 1.0]), persistent=False)
        self.last_diagnostics: dict = {}
        # The candidate statistics of the most recent step, kept for diagnostics
        # and for the trajectory explorer.  Never part of the recurrent state.
        self.last_statistics: Any = None
        self.last_connectivity: Any = None
        self.calls = 0

    # -- state --------------------------------------------------------------
    def init_state(self, batch_size: int, camera_spec, device: str = "cpu") -> N01State:
        return init_state_fn(self.config, batch_size, camera_spec, device)

    def reset(self, state: N01State, batch_mask) -> N01State:
        return reset_fn(state, batch_mask)

    def detach_state(self, state: N01State) -> N01State:
        return detach_state_fn(state)

    def to(self, *args, **kwargs):  # pragma: no cover - device movement helper
        super().to(*args, **kwargs)
        return self

    # -- one causal step ----------------------------------------------------
    def step(
        self,
        frame: MeasurementFrame,
        context=None,
        state: N01State | None = None,
        budget: ComputeBudget | None = None,
    ) -> EncoderOutput:
        started = time.perf_counter()
        if state is None:
            raise ValueError("N01Encoder.step requires a state from init_state")
        if frame.camera_id and frame.camera_id not in state.camera_ids:
            raise ValueError(f"unknown camera {frame.camera_id!r}; state has {state.camera_ids}")
        budget = budget or ComputeBudget(max_history=self.config.max_history)
        if budget.max_history > state.num_slots:
            raise ValueError("budget.max_history exceeds the state capacity")
        plan = self._budget_plan(budget)

        # Reuse the measured variance declared by the frame in the frame metadata
        # paths; the caller owns calibration, this module never re-derives units.
        metadata = dict(frame.metadata or {})
        if frame.saturated_mask is not None:
            metadata.setdefault("saturated_fraction", None)
        previous = state
        shifted = push_fn(previous, frame)

        camera = frame.camera_id or shifted.camera_ids[0]
        motion = self._estimate_motion(previous, frame)
        measured_edges, edge_info = self.integrator.edge_field(
            frame, shifted, motion=motion, estimator=self.correspondence
        )
        plane = shifted.plane_support(camera)
        measured = shifted.valid[0].any(dim=1)
        # Hysteresis: an edge opens after sustained compatibility and closes at
        # once on refutation, motion inconsistency or occlusion.  The support age
        # counts consecutive measured updates of that edge.
        edges, support_age, merges, splits = self.integrator.update_edges(
            plane.edges, plane.support_age, measured_edges, measured
        )
        seen_now = plane.seen | measured[:, None, None, None]
        now = shifted.now[:, None, None].expand_as(plane.last_seen)
        last_seen = torch.where(measured[:, None, None], now, plane.last_seen)
        shifted.support[camera] = replace(
            plane,
            edges=edges,
            seen=seen_now,
            support_age=support_age,
            last_seen=last_seen,
            merge_count=plane.merge_count + merges,
            split_count=plane.split_count + splits,
        )

        statistics, connectivity = self.integrator.domain_statistics(
            shifted, edges, motion=motion, estimator=self.correspondence
        )
        selection = self._select(statistics, connectivity, motion, plan)
        features = self._encode(frame, shifted, statistics, connectivity, selection, plan)

        cost = {
            "updates": int(plan["updates"]),
            "candidate_domains": int(statistics.candidate_count),
            "radius_levels": int(plan["radii"]),
            "depth_levels": int(plan["depths"]),
            "history": int(shifted.filled.max().item()),
            "state_bytes": int(state_bytes(shifted)),
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "candidate_statistics_computed": int(statistics.candidate_count),
        }
        diagnostics = {
            **edge_info,
            **self.last_diagnostics,
            "selected_radius": plan["radii"],
            "selected_depth": plan["depths"],
            "budget_max_updates": int(budget.max_updates),
            "budget_max_history": int(budget.max_history),
            "merge_count": float(
                (plane.merge_count.float().mean() + merges.float().mean()).detach()
            ),
            "split_count": float(
                (plane.split_count.float().mean() + splits.float().mean()).detach()
            ),
            "support_age_mean": float(support_age.mean().detach()),
            "mean_age": float(statistics.age.mean().detach()),
            "mean_variance_proxy": float(statistics.variance_proxy.mean().detach()),
            "mean_support_width": float(statistics.support_count.mean().detach()),
        }
        self.last_diagnostics = diagnostics
        self.last_statistics = statistics
        self.last_connectivity = connectivity
        self.calls += 1
        return EncoderOutput(
            representation=features,
            state=shifted,
            validity=shifted.valid[0].any(dim=1),
            diagnostics=diagnostics,
            cost=cost,
        )

    # -- internals ----------------------------------------------------------
    def _budget_plan(self, budget: ComputeBudget) -> dict:
        config = self.config
        plan = {
            "updates": max(int(budget.max_updates), 1),
            "radii": len(config.spatial_radii),
            "depths": len(config.time_depths),
        }
        if not config.budget_conditioning:
            return plan
        # Low budget: smaller candidate radius, shorter time depth, fewer blocks.
        if plan["updates"] <= 1:
            plan["radii"] = min(1, plan["radii"])
            plan["depths"] = min(2, plan["depths"])
        plan["blocks"] = min(plan["updates"], config.num_blocks)
        return plan

    def _estimate_motion(self, previous: N01State, frame: MeasurementFrame):
        """Cumulative causal displacement of every stored unique frame.

        Only already-arrived frames are compared, each against its next-newer
        observed neighbour; the shift used for alignment is the cumulative sum
        along that chain.  Slots without a measurement carry an invalid field so
        that they cannot contribute an invented displacement.
        """
        if self.correspondence is None:
            return None
        device = previous.values.device
        values = torch.as_tensor(frame.values, device=device, dtype=torch.float32)
        valid = frame.valid_mask
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid, device=device)
            if valid.dtype != torch.bool:
                valid = valid > 0.5
        observed_slots = [
            slot for slot in range(previous.values.shape[0]) if bool(previous.observed[slot].any())
        ]
        if not observed_slots:
            return None
        slots = [0, *observed_slots]
        relative: list[MotionField] = []
        informative: list[torch.Tensor] = []
        reference_values, reference_valid = values, valid
        for slot in slots[1:]:
            field = self.chain.relative_shift(
                reference_values,
                previous.values[slot],
                reference_valid,
                previous.valid[slot] > 0.5,
            )
            observed = previous.observed[slot]
            relative.append(field.select(observed))
            informative.append(observed)
            reference_values = previous.values[slot]
            reference_valid = previous.valid[slot] > 0.5
        cumulative = self.chain.cumulative(relative, informative)
        fields: list[MotionField] = []
        cursor = 0
        for slot in range(previous.values.shape[0]):
            if slot in observed_slots:
                fields.append(cumulative[cursor])
                cursor += 1
            else:
                batch = values.shape[0]
                fields.append(
                    MotionField(
                        shift=torch.zeros((batch, 2), device=device),
                        residual=torch.ones((batch,), device=device),
                        overlap=torch.zeros((batch,), device=device),
                        valid=torch.zeros((batch,), dtype=torch.bool, device=device),
                    )
                )
        if self.config.alignment == AlignmentMode.NONE.value:
            return None
        return fields

    def _select(self, statistics, connectivity, motion, plan):
        """Choose a support domain per position; records the real cost of doing so."""
        config = self.config
        device = statistics.mu.device
        payload = self._statistics_payload(statistics, connectivity, motion)
        order = torch.argsort(statistics.radius_index * 1000 + statistics.depth_index)
        del order
        depth_limit = int(config.time_depths[min(plan["depths"], len(config.time_depths)) - 1])
        keep = statistics.radius_index < int(plan["radii"])
        if config.budget_conditioning:
            keep = keep & (statistics.depth_index <= depth_limit)
        budget_slice = keep.nonzero(as_tuple=False).squeeze(1)
        if budget_slice.numel() == 0:
            budget_slice = torch.arange(statistics.candidate_count, device=device)
        scale = self._statistics_scale(payload, budget_slice)
        weights, index, cost = self.selector(payload, scale, budget_slice=budget_slice)
        selected = budget_slice[index]
        self.last_diagnostics = {
            "selection_entropy": float(
                -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log())
                .sum(dim=0)
                .mean()
                .detach()
            ),
            # 1 / sum(w^2): the effective number of domains contributing to the
            # mixed statistics field.  Reported so a mixture is never read as one
            # independent observation.
            "mixture_participation": float(
                (1.0 / (weights**2).sum(dim=0).clamp_min(1e-9)).mean().detach()
            ),
            "selected_radius_mean": float(
                statistics.radius_index[selected].float().mean().detach()
            ),
            "selected_depth_mean": float(statistics.depth_index[selected].float().mean().detach()),
            "cost_mean": float(cost.mean().detach()),
            "candidates_considered": int(budget_slice.numel()),
        }
        return {"index": selected, "weights": weights, "slice": budget_slice, "statistics": payload}

    def _statistics_payload(self, statistics, connectivity, motion):
        if motion is None:
            motion_error = torch.zeros_like(statistics.variance_proxy)
        else:
            # Accumulated correspondence error over the frames a candidate
            # integrates: [N, B] residuals become one value per candidate, depth
            # and batch member.  A deeper window then pays for carrying more
            # motion uncertainty, which is what lets a fast edge prefer a shorter
            # history while a still scene can still afford a long one.
            stacked = torch.stack([field.residual.detach() for field in motion], dim=0)
            slots, batch = stacked.shape
            prefix = torch.cat(
                [
                    torch.zeros((1, batch), device=stacked.device, dtype=stacked.dtype),
                    stacked.cumsum(0),
                ],
                dim=0,
            )
            depths = statistics.depth_index.clamp(max=slots).reshape(-1)
            motion_error = prefix[depths].reshape(-1, batch, 1, 1, 1)
            motion_error = motion_error.expand_as(statistics.variance_proxy)
        return {
            "mu": statistics.mu,
            "variance_proxy": statistics.variance_proxy.clamp_min(0.0),
            "age": statistics.age,
            "support_count": statistics.support_count,
            "unique_frames": statistics.unique_frames,
            "weight_concentration": statistics.weight_concentration,
            "connectivity": connectivity,
            "motion_error": motion_error,
        }

    def _statistics_scale(self, payload, budget_slice):
        """Normalisation scales taken from the candidate statistics at this step.

        They are reported in the diagnostics so a training-set calibration can be
        substituted on validation without changing the mechanism.
        """
        # The scale is reduced per batch member (never across the batch), so a
        # reset or a differently sized episode cannot change another episode's
        # selection.  Spatial positions inside one member are pooled.
        subset = {key: value[budget_slice] for key, value in payload.items()}

        def reduce(key: str, absolute: bool = False) -> torch.Tensor:
            tensor = subset[key]
            if absolute:
                tensor = tensor.abs()
            return tensor.mean(dim=(0, 2, 3, 4)).detach().clamp_min(1e-6)

        return {
            "variance_proxy": reduce("variance_proxy"),
            "age": reduce("age"),
            "support_count": reduce("support_count"),
            "unique_frames": reduce("unique_frames"),
            "motion_error": reduce("motion_error"),
            "mu": reduce("mu", absolute=True),
        }

    def _encode(self, frame, state, statistics, connectivity, selection, plan):
        config = self.config
        values = state.values[0]
        valid = state.valid[0]
        variance = state.variance[0]
        features = self.stem(values, valid, variance)
        # The selector returns [K, B, P, H, W] weights over the budgeted candidates.
        weights = selection["weights"]
        candidate_slice = selection["slice"]

        def select(stack: torch.Tensor) -> torch.Tensor:
            """Mix the candidate domains per position: [C,B,P,H,W] -> [B,P,H,W].

            The selector outputs **mixing weights** (the proposal allows either a
            chosen domain or a mixing weight), so the statistics field is the
            weighted mixture of the candidate domains.  A hard argmax would make
            the whole support graph invisible to the loss, so the gate could never
            be trained; the argmax choice is still reported separately in the
            diagnostics and the mixture's participation ratio is recorded, because
            overlapping domains are not independent observations.
            """
            subset = stack[candidate_slice]
            return (weights * subset).sum(dim=0)

        plane_mu = select(statistics.mu)
        plane_variance = select(statistics.variance_proxy)
        plane_age = select(statistics.age)
        plane_count = select(statistics.support_count)
        plane_frames = select(statistics.unique_frames)
        plane_concentration = select(statistics.weight_concentration)

        # Six explicit statistics channels: mu, variance_proxy, age, support
        # count, unique frame count and weight concentration.  They are kept
        # separate from the learned features so a feature magnitude is never read
        # as a confidence.  On the shared measurement grid the per-plane
        # statistics are reduced to one channel per statistic.
        def as_channel(stack: torch.Tensor, name: str) -> torch.Tensor:
            if stack.ndim != 4:
                raise ValueError(f"selected statistic {name} has rank {stack.ndim}, expected 4")
            if stack.shape[1] != 1:
                stack = stack.mean(dim=1, keepdim=True)
            return stack

        statistic_list = [
            as_channel(plane_mu, "mu"),
            as_channel(plane_variance, "variance_proxy"),
            as_channel(plane_age, "age"),
            as_channel(plane_count, "support_count"),
            as_channel(plane_frames, "unique_frames"),
            as_channel(plane_concentration, "weight_concentration"),
        ]
        statistics_tensor = torch.cat(statistic_list, dim=1)
        if statistics_tensor.shape[1] != statistics_channels(config):
            raise ValueError(
                f"statistics block has {statistics_tensor.shape[1]} channels, the blocks "
                f"were built for {statistics_channels(config)}"
            )
        graph = _graph_channels(select(connectivity), 1)
        blocks = int(plan.get("blocks", config.num_blocks))
        update_mask = torch.ones((features.shape[0],), device=features.device, dtype=torch.bool)
        for block in range(config.num_blocks):
            if block >= blocks:
                break
            features = self.blocks[block](features, statistics_tensor, graph, update_mask)
        return features

    # -- readout ------------------------------------------------------------
    def readout(self, output: EncoderOutput, token_budget: int | None = None) -> VisualTokens:
        representation = output.representation
        validity = output.validity
        tokens = self.readout_layer(representation, validity, token_budget)
        tokens.camera_ids = list(output.state.camera_ids)
        return tokens

    def predict_measurement(self, output: EncoderOutput) -> torch.Tensor:
        """Auxiliary RAW-domain prediction head (never fed back as a measurement)."""
        if self.measurement_head is None:
            raise RuntimeError("measurement prediction is disabled in this configuration")
        return self.measurement_head(output.representation)


def _is_contiguous(index: torch.Tensor) -> bool:
    return bool(torch.equal(index, torch.arange(index.numel(), device=index.device)))


def _graph_channels(connectivity: torch.Tensor, channels: int) -> torch.Tensor:
    """[B,K,H,W] support-graph mask -> [B, channels, H, W] for the local exchange."""
    mean = connectivity.mean(dim=1, keepdim=True)
    return mean.expand(-1, channels, -1, -1)


def domain_cost_reference(statistics: dict, scale: dict, weights: dict) -> torch.Tensor:
    """Explicit analytic cost, written for hand-checking the selector."""
    return (
        weights["noise"] * statistics["variance_proxy"] / max(scale["variance_proxy"], 1e-6)
        + weights["edge"] * (1.0 - statistics["connectivity"])
        + weights["age"] * statistics["age"] / max(scale["age"], 1e-6)
        + weights["motion"] * statistics["motion_error"] / max(scale["motion_error"], 1e-6)
    )


def offset_list(radius: int) -> list[tuple[int, int]]:
    """Expose the candidate offset enumeration for tests and diagnostics."""
    return candidate_offsets(radius)

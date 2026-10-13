"""N02Encoder: the measurement-constrained local multi-hypothesis recurrent field.

One ``step`` consumes one new frame (already adapted into the shared contract)
and runs:

1. **frozen measurement set.** The new frame enters the buffer only if it is new
   evidence. The set used by this step's cycles is then frozen: repeated cycles
   recompute the full score on the *same* set instead of accumulating one
   likelihood per iteration.
2. **field initialisation / prior.** The stem consumes measurements only. The
   causal prior uses the previous step's field (or the declared prior condition
   for the analytic case); it never re-uses the posterior produced by the same
   observation as if it were independent evidence.
3. **cycles.** Energy, contradiction, dependency-propagated selection, local
   update, proposal for refuted cells, and a full energy recomputation. The loop
   stops at ``budget.max_updates`` or when both the residual and the field change
   stay under tolerance.
4. **outputs.** A :class:`StructuredRepresentation`, the new state, the
   per-cycle audit trail and a cost record. The readout is a separate layer
   (``owac.frameworks.n02.readout``) so mechanism effects and readout effects can
   be measured apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from owac.core.encoder_contract import (
    CameraSpec,
    ComputeBudget,
    CostRecord,
    EmbodimentContext,
    EncoderOutput,
    MeasurementFrame,
    Validity,
    state_tensor_bytes,
)
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.hypotheses import (
    HypothesisInitializer,
    diversity_penalty,
    neighbour_validity,
)
from owac.frameworks.n02.measurement import (
    EnergyResult,
    MeasurementModel,
    compute_measurement_energy,
)
from owac.frameworks.n02.readout import StructuredRepresentation
from owac.frameworks.n02.revision import RevisionController
from owac.frameworks.n02.state import (
    HypothesisState,
    MeasurementBuffer,
    estimate_variance,
    measurement_entry_from_frame,
)


def _as_bool_mask(mask: Any, like: torch.Tensor, channels: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(like, dtype=torch.bool)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, device=like.device)
    mask = mask.to(like.device)
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None] if mask.shape[0] == like.shape[0] else mask[None]
    if mask.ndim != 4:
        raise ValueError("measurement masks must be [H,W], [B,H,W] or [B,C,H,W]")
    return mask.expand_as(like)


def _resample_mask(mask: torch.Tensor | None, grid: tuple[int, int]) -> torch.Tensor | None:
    """Bring a boolean mask to the mechanism grid, preserving its channel axis."""
    if mask is None:
        return None
    if mask.shape[-2:] == grid:
        return mask
    return F.interpolate(mask.float(), size=grid, mode="nearest") > 0.5


def _masked_grid_measurement(values, observed, variance, grid):
    """Average available samples only; propagate independent variance of the mean.

    The exact adaptive-pooling bin size accounts for partial boundary bins. CFA
    planes are pooled independently, without mixing filter responses.
    """
    safe = torch.where(observed, values, 0.0)
    if values.shape[-2:] == grid:
        return safe, observed, variance
    height, width = values.shape[-2:]
    rows = torch.arange(grid[0], device=values.device, dtype=torch.float32)
    cols = torch.arange(grid[1], device=values.device, dtype=torch.float32)
    row_size = torch.ceil((rows + 1) * height / grid[0]) - torch.floor(rows * height / grid[0])
    col_size = torch.ceil((cols + 1) * width / grid[1]) - torch.floor(cols * width / grid[1])
    area = row_size[:, None] * col_size[None, :]
    fraction = F.adaptive_avg_pool2d(observed.float(), grid)
    pooled = F.adaptive_avg_pool2d(safe, grid) / fraction.clamp_min(1e-12)
    pooled_variance = None
    if variance is not None:
        pooled_variance = F.adaptive_avg_pool2d(torch.where(observed, variance, 0.0), grid)
        pooled_variance = pooled_variance / (area * fraction.square()).clamp_min(1e-12)
    return pooled, fraction > 0, pooled_variance


@dataclass(slots=True)
class StepDiagnostics:
    """Everything a run log needs from one step, kept out of the state."""

    cycles_run: int = 0
    stopped_early: bool = False
    energy_trace: list[list[float]] = field(default_factory=list)
    change_trace: list[float] = field(default_factory=list)
    contradiction_trace: list[float] = field(default_factory=list)
    current_top_weight: float = 0.0
    unresolved: bool = False
    revision_trace: list[dict] = field(default_factory=list)
    observations: int = 0
    observation_history: int = 0
    duplicates_skipped: int = 0
    energy_delta: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "cycles_run": self.cycles_run,
            "stopped_early": self.stopped_early,
            "current_top_weight": self.current_top_weight,
            "unresolved": self.unresolved,
            "observations": self.observations,
            "observation_history": self.observation_history,
            "duplicates_skipped": self.duplicates_skipped,
            "energy_delta": self.energy_delta,
        }


class N02Encoder(nn.Module):
    """Measurement-constrained local multi-hypothesis recurrent encoder."""

    def __init__(self, config: N02Config) -> None:
        super().__init__()
        self.config = config
        self.measurement = MeasurementModel(config)
        self.initializer = HypothesisInitializer(
            config.D,
            config.K,
            context_dim=config.K + 1,
            code_scale=0.5,
        )
        self.reviser = RevisionController(config, width=config.D)
        self.edge_parameter = nn.Parameter(torch.zeros(4, config.K, config.K))
        with torch.no_grad():
            identity = torch.eye(config.K).unsqueeze(0).repeat(4, 1, 1)
            self.edge_parameter.copy_(identity)
        self._position_cache: dict[tuple[int, int, int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------ state
    def init_state(self, batch_size: int, camera_spec: CameraSpec, device: Any) -> HypothesisState:
        """Initial state with zero hypotheses; the first ``step`` initialises them.

        ``camera_spec`` fixes the grid and the number of views so a later frame
        cannot silently change the field's shape.
        """
        device = torch.device(device)
        height = max(1, camera_spec.height // self._grid_stride(camera_spec.height))
        width = max(1, camera_spec.width // self._grid_stride(camera_spec.width))
        h = torch.zeros(batch_size, 1, self.config.K, self.config.D, height, width, device=device)
        zeros = torch.zeros(batch_size, 1, self.config.K, device=device)
        geometry = torch.zeros(
            batch_size, 1, self.config.K, self.measurement.geometry_dim, device=device
        )
        return HypothesisState(
            h=h,
            measurement_energy=zeros.clone(),
            weights=torch.full_like(zeros, 1.0 / self.config.K),
            prior_energy=zeros.clone(),
            relation_energy=zeros.clone(),
            total_energy=zeros.clone(),
            logits=zeros.clone(),
            geometry=geometry,
            age=torch.zeros_like(zeros),
            times_reset=torch.zeros_like(zeros),
        )

    @staticmethod
    def _grid_stride(size: int) -> int:
        """Keep the mechanism grid small enough for CPU prototype runs."""
        for stride in (4, 2, 1):
            if size // stride >= 8:
                return stride
        return 1

    def reset(self, state: HypothesisState, batch_mask: Any) -> HypothesisState:
        """Reset the selected batch elements; episode boundaries use this.

        ``batch_mask[b] == True`` means "this episode ended, clear it". The mask is
        broadcast only over the *leading* axis of each tensor, so a tensor whose
        second axis happens to have the same size as the batch can never be
        broadcast into a wrong shape.
        """
        if batch_mask is None:
            batch_mask = torch.ones(state.h.shape[0], device=state.h.device, dtype=torch.bool)
        if not torch.is_tensor(batch_mask):
            batch_mask = torch.as_tensor(batch_mask, device=state.h.device)
        batch_mask = batch_mask.to(state.h.device, torch.bool).view(-1)
        if batch_mask.numel() == 1:
            batch_mask = batch_mask.expand(state.h.shape[0])
        batch = state.h.shape[0]
        if batch_mask.numel() != batch:
            raise ValueError(
                f"reset mask has {batch_mask.numel()} entries but the state batch is {batch}"
            )

        def keep(tensor: torch.Tensor | None, default: torch.Tensor) -> torch.Tensor:
            if tensor is None:
                return default
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"state tensor with leading size {tensor.shape[0]} does not match batch {batch}"
                )
            view = (~batch_mask).reshape(batch, *([1] * (tensor.dim() - 1)))
            return torch.where(view, tensor, default)

        zeros = torch.zeros_like(state.measurement_energy)
        result = HypothesisState(
            h=keep(state.h, torch.zeros_like(state.h)),
            measurement_energy=keep(state.measurement_energy, zeros),
            weights=keep(state.weights, torch.full_like(state.weights, 1.0 / self.config.K)),
            prior_energy=keep(state.prior_energy, zeros),
            relation_energy=keep(state.relation_energy, zeros),
            total_energy=keep(state.total_energy, zeros),
            logits=keep(state.logits, zeros),
            geometry=keep(state.geometry, torch.zeros_like(state.geometry)),
            age=keep(state.age, zeros),
            times_reset=keep(state.times_reset, zeros),
            revision_energy=keep(state.revision_energy, zeros),
            support=None
            if state.support is None
            else keep(state.support, torch.zeros_like(state.support)),
            dependency_edge=state.dependency_edge,
            dependency_valid=state.dependency_valid,
            step_index=0 if bool(batch_mask.all()) else state.step_index,
        )
        result.diagnostics = dict(state.diagnostics)
        if "_buffer" in result.diagnostics:
            result.diagnostics["_buffer"] = result.diagnostics["_buffer"].clear_batch(batch_mask)
        for key in ("_initial_prior",):
            if key in result.diagnostics:
                value = result.diagnostics[key]
                result.diagnostics[key] = torch.where(
                    batch_mask[:, None, None], torch.zeros_like(value), value
                )
        result.revision = replace(state.revision)
        for name in (
            "residual_ema",
            "contradiction",
            "contradiction_prev",
            "propagated",
            "priority",
            "last_reset",
        ):
            value = getattr(result.revision, name)
            if value is not None:
                setattr(result.revision, name, keep(value, torch.zeros_like(value)))
        return result

    def detach_state(self, state: HypothesisState) -> HypothesisState:
        return state.detach()

    # ------------------------------------------------------------------ steps
    def step(
        self,
        frame: MeasurementFrame,
        context: EmbodimentContext,
        state: HypothesisState | None,
        budget: ComputeBudget,
    ) -> EncoderOutput:
        """Consume one frame and return the revised field."""
        if state is None:
            source = torch.as_tensor(frame.values)
            state = self.init_state(
                source.shape[0] if source.ndim == 4 else 1,
                CameraSpec(
                    camera_id=frame.camera_id,
                    height=source.shape[-2],
                    width=source.shape[-1],
                    channels=source.shape[-3] if source.ndim >= 3 else 1,
                    input_mode=self.config.measurement.input_mode,
                    cfa_pattern=self.config.measurement.cfa_pattern,
                ),
                next(self.parameters()).device,
            )
        state = replace(
            state,
            diagnostics=dict(state.diagnostics),
            revision=replace(state.revision),
            dependency_edge=None,
        )
        values, valid, bad, saturated = self._frame_tensors(frame, state)
        # The mechanism operates on the grid fixed by the camera spec at
        # ``init_state``, not on the sensor resolution: the field, the dependency
        # graph and the position codes all live on that grid, and the buffer
        # stores the same grid so every later energy pass sees one resolution.
        grid = state.h.shape[-2:]
        supplied = None
        if frame.metadata_available.get("variance", False) and "variance" in frame.metadata:
            supplied = torch.as_tensor(
                frame.metadata["variance"], device=values.device, dtype=torch.float32
            )
            if supplied.numel() == values.shape[0]:
                supplied = supplied.reshape(-1, 1, 1, 1)
            supplied = supplied.expand_as(values)
            if not bool((torch.isfinite(supplied) & (supplied > 0)).all()):
                raise ValueError("declared sensor variance must be finite and positive")
        observed = valid & ~bad
        if not self.config.measurement.saturation_is_observation:
            observed = observed & ~saturated
        values, valid, supplied = _masked_grid_measurement(values, observed, supplied, grid)
        # A partially observed bin remains valid. Bad/saturated samples have
        # already been excluded from its mean and cannot invalidate its support.
        bad = ~valid & (F.adaptive_max_pool2d(bad.float(), grid) > 0)
        saturated = ~valid & (F.adaptive_max_pool2d(saturated.float(), grid) > 0)
        variance, calibrated = estimate_variance(
            values,
            shot_scale=self.config.measurement.shot_scale,
            read_scale=self.config.measurement.read_scale,
            floor=self.config.measurement.variance_floor,
        )
        sample_xy = frame.sample_xy if torch.is_tensor(frame.sample_xy) else None
        entry = measurement_entry_from_frame(
            frame,
            values=values,
            variance=variance.to(values.dtype),
            sample_xy=sample_xy,
            camera_id=frame.camera_id or "camera_0",
            frame_id=self._frame_id(frame),
            episode_id=self._episode_id(frame),
            timestamp=None if frame.timestamp is None else float(frame.timestamp),
        )
        entry.valid_mask, entry.bad_mask, entry.saturated_mask = valid, bad, saturated
        entry.batch_identities = tuple(
            (
                frame.camera_id,
                frame.frame_ids[min(index, len(frame.frame_ids) - 1)] if frame.frame_ids else -1,
                frame.episode_ids[min(index, len(frame.episode_ids) - 1)]
                if frame.episode_ids
                else None,
            )
            for index in range(values.shape[0])
        )
        if supplied is not None:
            entry.variance = supplied.clamp_min(self.config.measurement.variance_floor)
            entry.metadata["variance_is_estimate"] = bool(
                frame.metadata.get("variance_is_estimate", False)
            )
            calibrated = not entry.metadata["variance_is_estimate"]
        buffer = self._buffer(state, budget, values.device)
        entry.new_mask = (
            torch.ones(values.shape[0], device=values.device, dtype=torch.bool)
            if frame.new_measurement is None
            else torch.as_tensor(
                frame.new_measurement, device=values.device, dtype=torch.bool
            ).expand(values.shape[0])
        )
        is_new = bool(entry.new_mask.any())
        added = buffer.add(entry) if is_new else False
        if not is_new:
            buffer.rejected_count += 1

        # Dynamic v1 uses only the current likelihood; retained history is an
        # audit buffer, not a second copy of evidence already carried by the prior.
        active = buffer.copy()
        if active.frames:
            active._frames = active.frames[-1:]
        else:
            # Window padding can precede the first fresh frame. It contributes
            # no likelihood, but still supplies a shape for the solver.
            entry.valid_mask = torch.zeros_like(values, dtype=torch.bool)
            active._frames = [entry]
        result = self._solve(
            active,
            state,
            context,
            values=values,
            valid=valid,
            bad=bad,
            saturated=saturated,
            budget=budget,
            calibrated=calibrated,
            added=added,
            is_new=is_new,
            sampling_metadata=self._sampling_metadata(frame, values),
        )
        result.state.diagnostics["_buffer"] = buffer
        result.representation.diagnostics["camera_id"] = frame.camera_id
        result.diagnostics["observation_history"] = len(buffer)
        result.diagnostics["likelihood_observations"] = int(
            bool(active.frames) and bool(active.frames[-1].valid_mask.any())
        )
        result.cost.history_len = len(buffer)
        result.cost.state_bytes = result.state.state_bytes() + state_tensor_bytes(buffer.frames)
        return result

    def _buffer(
        self, state: HypothesisState, budget: ComputeBudget, device: torch.device
    ) -> MeasurementBuffer:
        buffer = state.diagnostics.get("_buffer")
        history = min(int(budget.max_history), self.config.history)
        if buffer is None or buffer.max_history != history:
            buffer = MeasurementBuffer(history, mask_history=True, device=device)
            state.diagnostics["_buffer"] = buffer
        return buffer.copy()

    def _frame_tensors(
        self, frame: MeasurementFrame, state: HypothesisState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = frame.values
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        values = values.to(state.h.device, torch.float32)
        if values.dim() == 2:
            values = values[None, None]
        elif values.dim() == 3:
            values = values[None]
        channels = values.shape[1]
        expected = self.measurement.out_channels
        if channels != expected:
            raise ValueError(
                f"input_mode={self.config.measurement.input_mode!r} expects {expected} channels, "
                f"got {channels}; the CFA and three-channel inputs must use separate stems"
            )
        valid = _as_bool_mask(frame.valid_mask, values, channels)
        if frame.valid_mask is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        bad = _as_bool_mask(frame.bad_mask, values, channels)
        saturated = _as_bool_mask(frame.saturated_mask, values, channels)
        finite = torch.isfinite(values)
        valid = valid & finite
        bad = bad | ~finite
        return values, valid, bad, saturated

    @staticmethod
    def _frame_id(frame: MeasurementFrame) -> int:
        if frame.frame_ids:
            return frame.frame_ids[0]
        if "frame_id" in frame.metadata:
            return int(frame.metadata["frame_id"])
        return -1

    @staticmethod
    def _episode_id(frame: MeasurementFrame) -> int | None:
        if frame.episode_ids:
            return frame.episode_ids[0]
        return None

    # ----------------------------------------------------------------- solver
    def _solve(
        self,
        buffer: MeasurementBuffer,
        state: HypothesisState,
        context: EmbodimentContext,
        *,
        values: torch.Tensor,
        valid: torch.Tensor,
        bad: torch.Tensor,
        saturated: torch.Tensor,
        budget: ComputeBudget,
        calibrated: bool,
        added: bool,
        is_new: bool,
        sampling_metadata: torch.Tensor | None = None,
    ) -> EncoderOutput:
        config = self.config
        diagnostics = StepDiagnostics()
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, device=state.h.device)
        batch = values.shape[0]
        grid = state.h.shape[-2:]
        frozen = buffer.stack_values()  # [B, T, C, H, W]
        latest = frozen[:, -1]
        mask_latest = torch.ones_like(latest, dtype=torch.bool)
        if buffer.frames:
            last_mask = buffer.frames[-1].valid_mask
            if last_mask is not None and torch.is_tensor(last_mask):
                mask_latest = _as_bool_mask(last_mask, latest, latest.shape[1])
                mask_latest = _resample_mask(mask_latest, latest.shape[-2:])
        bad_latest, sat_latest = bad, saturated
        features = self.measurement(latest, mask_latest, bad_latest, sat_latest, sampling_metadata)
        basis = features.reshape(batch, 1, *features.shape[1:])
        if state.h.shape[0] != batch:
            raise ValueError(
                f"state batch {state.h.shape[0]} does not match frame batch {batch}; "
                "call reset for finished episodes instead of resizing"
            )
        prior_mass, available = self._declared_prior(context, batch, values.device)
        prior_context = torch.cat([prior_mass, available[:, None].to(values.dtype)], dim=-1)
        if config.prior_mode == "fixed_odds":
            prior_context = torch.cat(
                [
                    torch.full_like(prior_mass, 1.0 / config.K),
                    torch.zeros_like(available[:, None], dtype=values.dtype),
                ],
                dim=-1,
            )
        field = self._ensure_field(state, basis, prior_context)
        # A single causal transition incorporates the new measurement features.
        # Internal cycles and repeated IDs do not execute this transition again.
        if added and config.revision.update_scale != 0:
            fresh = self.initializer(basis[:, 0], context=prior_context).unsqueeze(1)
            cold = (state.age.sum(dim=-1) == 0).reshape(batch, 1, 1, 1, 1, 1)
            accepted = (
                buffer.last_added
                if buffer.last_added is not None
                else torch.ones(batch, device=values.device, dtype=torch.bool)
            )
            transitioned = torch.where(cold, fresh, 0.5 * field + 0.5 * fresh)
            field = torch.where(accepted.reshape(batch, 1, 1, 1, 1, 1), transitioned, field)
        # The field is pinned to the grid declared at ``init_state``. A candidate
        # whose shape drifts may not silently resize the dependency graph or the
        # measurement comparison, so it is brought back explicitly.
        field = self._pin_grid(field, grid)
        dependency_valid = self._ensure_dependency(state, field)
        cold_start = state.age.sum(dim=(1, 2)) == 0
        beta_prior = self._bounded_beta_prior()
        first = state.age.sum(dim=-1, keepdim=True) == 0
        declared_energy = -config.temperature * prior_mass.clamp_min(1e-12).log().unsqueeze(1)
        if added:
            previous = -config.temperature * state.weights.clamp_min(1e-12).log() / beta_prior
            frozen_prior = torch.where(first, declared_energy, previous)
            if buffer.last_added is not None:
                frozen_prior = torch.where(buffer.last_added[:, None, None], frozen_prior, previous)
        else:
            frozen_prior = state.prior_energy.detach()
        state = replace(state, prior_energy=frozen_prior)
        frozen_revision = (
            torch.zeros_like(frozen_prior)
            if state.revision_energy is None
            else state.revision_energy.detach()
        )
        max_cycles = max(1, min(int(budget.max_updates), int(config.cycles)))
        previous_energy = None
        stop_reason = "max_iterations"
        cycles_run = 0
        energy_result: EnergyResult | None = None
        weights = state.weights
        for iteration in range(max_cycles):
            cycles_run = iteration + 1
            mean, log_variance, geometry = self.measurement.predict(field)
            prediction = _Prediction(mean, torch.exp(log_variance), geometry)
            energy_result = compute_measurement_energy(
                prediction, buffer, config.measurement, view_index=0
            )
            measurement_scalar = self._as_view_scalar(energy_result.energy, field.shape[1])
            if iteration == 0 and added:
                # Compute withdrawal once from the pre-revision predictions and
                # actual measurement support. New evidence alone refreshes it.
                observed_withdrawal = self.reviser.withdrawal_energy(
                    energy_result.normalized_residual.unsqueeze(1),
                    self._support_map(energy_result, field),
                    temperature=config.temperature,
                ).detach()
                accepted = (~cold_start) & (
                    buffer.last_added
                    if buffer.last_added is not None
                    else torch.ones_like(cold_start)
                )
                # Rejected slots have zero current likelihood. Their prior is
                # already the old posterior, including its old withdrawal term.
                frozen_revision = torch.where(
                    accepted[:, None, None],
                    observed_withdrawal,
                    torch.zeros_like(frozen_revision),
                )
            prior_energy = frozen_prior
            relation_energy = self._relation_energy(field)
            beta_prior = self._bounded_beta_prior()
            total = (
                measurement_scalar
                + beta_prior * prior_energy
                + config.beta_relation * relation_energy
                + frozen_revision
            )
            scores = -total / config.temperature
            weights = torch.softmax(scores, dim=-1)
            with torch.no_grad():
                # One scalar per hypothesis, averaged over the batch and views.
                per_hypothesis = total.detach().mean(dim=0).mean(dim=0)  # [K]
                diagnostics.energy_trace.append(
                    [float(value) for value in per_hypothesis.flatten().tolist()]
                )
            revision = self.reviser.step(
                _with_h(state, field),
                energy_result.normalized_residual.unsqueeze(1),
                residual_feature=self._residual_feature(energy_result, field),
                prior_feature=self._prior_feature(state, field),
                location_code=self._position_code(field),
                dependency_valid=dependency_valid,
                iteration=iteration,
                cold_start=cold_start if iteration == 0 else torch.zeros_like(cold_start),
                measurement_feature=basis,
            )
            diagnostics.contradiction_trace.append(revision.contradicted_fraction)
            diagnostics.revision_trace.append(dict(revision.trace))
            new_field = revision.h
            change = float((new_field - field).abs().mean().item())
            diagnostics.change_trace.append(change)
            field = new_field
            state = _with_h(state, field)
            # The revision bookkeeping now lives on the state the encoder threads
            # through the cycles; the controller itself does not mutate its input.
            if revision.revision is not None:
                state.revision = revision.revision
            tolerance = config.revision.tolerance
            residual_now = float(total.detach().mean().item())
            residual_delta = (
                abs(residual_now - previous_energy) if previous_energy is not None else float("inf")
            )
            previous_energy = residual_now
            if (
                iteration + 1 >= max(1, config.revision.min_iterations)
                and change <= tolerance
                and residual_delta <= tolerance
            ):
                stop_reason = "converged"
                diagnostics.stopped_early = True
                break
        mean, log_variance, geometry = self.measurement.predict(field)
        prediction = _Prediction(mean, torch.exp(log_variance), geometry)
        energy_result = compute_measurement_energy(
            prediction, buffer, config.measurement, view_index=0
        )
        prior_energy = frozen_prior
        relation_energy = self._relation_energy(field)
        beta_prior = self._bounded_beta_prior()
        measurement_scalar = self._as_view_scalar(energy_result.energy, field.shape[1])
        total = (
            measurement_scalar
            + beta_prior * prior_energy
            + config.beta_relation * relation_energy
            + frozen_revision
        )
        logits = -total / config.temperature
        weights = torch.softmax(logits, dim=-1)
        if previous_energy is not None:
            diagnostics.energy_delta = abs(float(total.detach().mean().item()) - previous_energy)
        support = self._support_map(energy_result, field)  # [B,1,K,h,w]
        revision = state.revision
        new_state = HypothesisState(
            h=field,
            measurement_energy=measurement_scalar.detach(),
            weights=weights.detach(),
            prior_energy=prior_energy.detach(),
            relation_energy=relation_energy.detach(),
            total_energy=total.detach(),
            logits=logits.detach(),
            geometry=geometry.detach(),
            age=(
                state.age
                + (
                    buffer.last_added
                    if added and buffer.last_added is not None
                    else torch.zeros(batch, device=values.device)
                ).reshape(batch, 1, 1)
            ).detach(),
            times_reset=(
                state.times_reset + revision.last_reset.to(torch.float32).sum(dim=(-1, -2))
                if revision.last_reset is not None
                else state.times_reset
            ).detach(),
            revision_energy=frozen_revision.detach(),
            support=support.detach(),
            dependency_edge=state.dependency_edge,
            dependency_valid=dependency_valid,
            revision=revision,
            diagnostics={"step_index": state.step_index + 1},
            step_index=state.step_index + 1,
        )
        new_state.diagnostics["_buffer"] = buffer
        unresolved = bool(
            (weights.max(dim=-1).values <= (1.0 / config.K) + 0.25 * (1.0 - 1.0 / config.K))
            .any()
            .item()
        )
        diagnostics.unresolved = unresolved
        diagnostics.cycles_run = cycles_run
        diagnostics.current_top_weight = float(weights.max().item())
        diagnostics.observations = int(added)
        diagnostics.observation_history = len(buffer)
        diagnostics.duplicates_skipped = buffer.duplicate_count + buffer.rejected_count
        diversity, collapsed = diversity_penalty(field)
        validity_flag = (~valid).all(dim=1).all(dim=1).all(dim=1)
        representation = StructuredRepresentation(
            h=field,
            weights=weights,
            measurement_energy=measurement_scalar,
            prior_energy=prior_energy,
            relation_energy=relation_energy,
            geometry=geometry,
            revision_energy=frozen_revision,
            posterior_logits=logits,
            valid=~validity_flag.view(batch, 1),
            support=support,
            step_index=new_state.step_index,
            diagnostics={
                "stop_reason": stop_reason,
                "beta_prior": beta_prior,
                "calibrated_noise": calibrated,
                "new_evidence": added,
                "diversity": float(diversity.item()),
                "collapsed_fraction": float(collapsed.item()),
                "deterministic_stop": stop_reason,
                "history_scheme": "current_likelihood_recursive_prior",
            },
        )
        state_bytes = new_state.state_bytes() + state_tensor_bytes(buffer.frames)
        cost = CostRecord(
            updates=cycles_run,
            history_len=len(buffer),
            hypotheses=config.K,
            updated_fraction=None
            if not diagnostics.revision_trace
            else float(diagnostics.revision_trace[-1]["updated_fraction"]),
            state_bytes=state_bytes,
            notes={
                "views": 1,
                "grid": [int(field.shape[-2]), int(field.shape[-1])],
                "stop_reason": stop_reason,
            },
        )
        missing = ~valid
        validity = Validity(
            valid_mask=valid,
            missing_mask=missing,
            saturated_mask=saturated,
            unresolved_ambiguity=new_state.weights.max(dim=-1).values
            <= (1.0 / config.K + 0.25 * (1.0 - 1.0 / config.K)),
            solved=not unresolved,
            reason=stop_reason,
        )
        return EncoderOutput(
            representation=representation,
            state=new_state,
            validity=validity,
            diagnostics=diagnostics.summary(),
            cost=cost,
        )

    # -------------------------------------------------------------- internals
    def _sampling_metadata(self, frame: MeasurementFrame, values: torch.Tensor):
        if not self.measurement.cfa_planes:
            return None
        if frame.sample_xy is None:
            raise ValueError("CFA planes require sample_xy; four channels are not co-sited colours")
        xy = torch.as_tensor(frame.sample_xy, device=values.device, dtype=torch.float32)
        if xy.shape in {(4, 2, 2), (4, 3, 2)}:  # origin, step, optional reference
            if not bool(torch.all(xy[:, 1] == 2)):
                raise ValueError("CFA v0 expects declared plane sampling step (2,2)")
            xy = xy[:, 0].unsqueeze(0)
        elif xy.shape == (4, 2):
            xy = xy.unsqueeze(0)
        if xy.shape[-2:] != (4, 2):
            raise ValueError("CFA sample_xy must provide four plane origins")
        # Both adapters reorder to (R,G1,G2,B); phase lives in actual origins.
        response = torch.tensor([0.0, 1.0, 1.0, 2.0], device=values.device)
        metadata = torch.zeros(values.shape[0], 4, 4, device=values.device)
        metadata[..., :2] = xy.expand(values.shape[0], -1, -1)
        metadata[..., 2] = response
        metadata[..., 3] = 1
        return metadata

    def _declared_prior(self, context: EmbodimentContext, batch: int, device: Any):
        uniform = torch.full((batch, self.config.K), 1.0 / self.config.K, device=device)
        if context.hypothesis_prior is None:
            return uniform, torch.zeros(batch, device=device, dtype=torch.bool)
        prior = torch.as_tensor(context.hypothesis_prior, device=device, dtype=torch.float32)
        if (
            prior.shape != uniform.shape
            or not bool(torch.isfinite(prior).all())
            or bool((prior < 0).any())
            or bool((prior.sum(-1) <= 0).any())
        ):
            raise ValueError("declared hypothesis_prior must be finite nonnegative [B,K] mass")
        available = (
            torch.ones(batch, device=device, dtype=torch.bool)
            if context.hypothesis_prior_available is None
            else torch.as_tensor(
                context.hypothesis_prior_available, device=device, dtype=torch.bool
            ).expand(batch)
        )
        prior = prior / prior.sum(-1, keepdim=True)
        return torch.where(available[:, None], prior, uniform), available

    def _ensure_field(
        self, state: HypothesisState, basis: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if state.h.shape[0] != basis.shape[0]:
            raise ValueError("field batch mismatch")
        flat = basis.reshape(basis.shape[0] * basis.shape[1], *basis.shape[2:])
        initialised = self.initializer(flat, context=context)
        initialised = initialised.reshape(basis.shape[0], basis.shape[1], *initialised.shape[1:])
        cold = (state.age.sum(dim=-1) == 0).reshape(basis.shape[0], 1, 1, 1, 1, 1)
        return torch.where(cold, initialised, state.h)

    @staticmethod
    def _pin_grid(field: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Resize a field to the declared grid if it drifted, keeping gradient flow."""
        if tuple(field.shape[-2:]) == tuple(grid):
            return field
        batch, views, K, width = field.shape[:4]
        flat = field.reshape(batch * views * K, width, *field.shape[-2:])
        flat = F.interpolate(flat, size=tuple(grid), mode="bilinear", align_corners=False)
        return flat.reshape(batch, views, K, width, *grid)

    def _ensure_dependency(self, state: HypothesisState, field: torch.Tensor) -> torch.Tensor:
        """Record the 4-neighbour dependency graph and its boundary validity.

        The edge tensor states "candidate k at u depends on candidate l at the
        neighbour v_d(u)". It is a *recorded dependency*, not a physical causal
        claim, and it is learnable through ``edge_parameter``.

        The graph is rebuilt whenever the stored one does not match the current
        field, so a resolution or view-count change can never leave a stale graph
        silently in place; the rebuild is reported through the returned mask.
        """
        batch, views = field.shape[0], field.shape[1]
        height, width = field.shape[-2:]
        # ``neighbour_validity`` returns [4, h, w] with the neighbour axis FIRST.
        # A bare ``view(..., 4)`` would reorder the data instead of the axes and
        # would zero valid interior neighbours while keeping out-of-grid ones,
        # which silently disabled propagation along whole directions.
        valid = (
            neighbour_validity(height, width, field.device)
            .permute(1, 2, 0)  # [h, w, 4]
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, views, -1, -1, -1)
        )
        stored_edge = state.dependency_edge
        needs_rebuild = (
            stored_edge is None
            or stored_edge.shape[0] != batch
            or stored_edge.shape[1] != views
            or stored_edge.shape[2:4] != (height, width)
        )
        if needs_rebuild:
            state.dependency_edge = self.dependency_edge(batch, views, height, width, field.device)
        stored_valid = state.dependency_valid
        if (
            stored_valid is None
            or stored_valid.shape[0] != batch
            or stored_valid.shape[1] != views
            or stored_valid.shape[2:4] != (height, width)
        ):
            state.dependency_valid = valid
        return state.dependency_valid

    def dependency_edge(self, batch: int, views: int, height: int, width: int, device: Any):
        """``[B,V,h,w,4,K,K]`` dependency graph broadcast from ``edge_parameter``.

        ``edge_parameter`` is a single learnable ``[4,K,K]`` table shared by every
        location: gradients from all locations accumulate into it, which is what
        makes the compatibility *learned* rather than fixed.
        """
        edge = self.edge_parameter.view(1, 1, 1, 1, 4, self.config.K, self.config.K)
        edge = edge.expand(batch, views, height, width, -1, -1, -1)
        return edge if edge.device == torch.device(device) else edge.to(device)

    @staticmethod
    def _as_view_scalar(energy: torch.Tensor, views: int) -> torch.Tensor:
        """``[B,K]`` (this module evaluates view 0) -> ``[B,V,K]``.

        The single-view evaluation is made explicit here rather than relying on
        broadcasting, which previously produced a ``[B,2V,K]`` score tensor when
        the batch size matched the view count.
        """
        if energy.dim() == 3:
            if energy.shape[1] != views:
                raise ValueError(
                    f"energy carried {energy.shape[1]} views but the field has {views}"
                )
            return energy
        if energy.dim() != 2:
            raise ValueError(f"unexpected energy shape {tuple(energy.shape)}")
        return energy.unsqueeze(1).expand(-1, views, -1)

    def _bounded_beta_prior(self) -> float:
        low, high = self.config.beta_prior_bounds
        return float(min(max(self.config.beta_prior, low), high))

    def _prior_energy(self, state: HypothesisState, *, cold_start: bool) -> torch.Tensor:
        """``[B,V,K]`` prior term with a fixed source inside one step's cycles.

        Before any measurement has been processed the prior is uniform (zero
        energy). Afterwards it is the previous step's total energy, which is *not*
        a function of the current observation, so re-optimising inside this step
        cannot re-count the same likelihood as independent prior evidence.

        The stored tensor is matched to the field's ``(B, V, K)`` explicitly; a
        mismatch raises instead of being reshaped into a plausible-looking value.
        """
        batch, views = state.h.shape[0], state.h.shape[1]
        if cold_start or state.total_energy is None or state.total_energy.numel() == 0:
            return state.h.new_zeros(batch, views, self.config.K)
        prior = state.total_energy.detach()
        if tuple(prior.shape) != (batch, views, self.config.K):
            squeezed = prior
            while squeezed.dim() > 3 and squeezed.shape[1] == 1:
                squeezed = squeezed.squeeze(1)
            squeezed = squeezed.reshape(batch, views, self.config.K)
            prior = squeezed
        return prior

    def _relation_energy(self, field: torch.Tensor) -> torch.Tensor:
        """Neighbour-agreement energy; zero when ``beta_relation`` is zero."""
        if self.config.beta_relation == 0.0:
            return torch.zeros(
                field.shape[0], 1, self.config.K, device=field.device, dtype=field.dtype
            )
        from owac.frameworks.n02.hypotheses import neighbour_stack

        neighbours = neighbour_stack(field)  # [B,V,K,h,w,4,W] after move
        mean_neighbour = neighbours.mean(dim=-1)
        difference = (field - mean_neighbour).pow(2).mean(dim=(-1, -2, -3))
        return difference

    def _residual_feature(self, energy_result: EnergyResult, field: torch.Tensor) -> torch.Tensor:
        """``[B,V,K,h,w]`` normalised residual evidence for the revision cell.

        The residual is mapped onto a *reference cell* and broadcast to every
        hypothesis: it is the observation-to-prediction gap of the best current
        explanation, not a per-candidate residual (which would already encode the
        answer by construction).
        """
        residual = energy_result.normalized_residual  # [B, K or 1, h, w]
        if residual.dim() == 3:
            residual = residual.unsqueeze(1)
        batch = field.shape[0]
        views = field.shape[1]
        if residual.shape[0] != batch:
            raise ValueError("residual batch does not match the field batch")
        if views != 1:
            raise ValueError(
                "residual evidence is defined for a single view; per-view residual maps "
                "must be supplied by a multi-camera extension instead of being broadcast"
            )
        if residual.shape[1] == 1:
            residual = residual.expand(-1, self.config.K, -1, -1)
        elif residual.shape[1] != self.config.K:
            raise ValueError(f"unexpected residual channels {residual.shape[1]}")
        shaped = residual.reshape(batch, 1, self.config.K, *residual.shape[-2:])
        if shaped.shape[-2:] != field.shape[-2:]:
            shaped = F.interpolate(
                shaped.reshape(batch * views * self.config.K, 1, *shaped.shape[-2:]),
                size=field.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, views, self.config.K, *field.shape[-2:])
        return shaped.to(field.dtype)

    def _prior_feature(self, state: HypothesisState, field: torch.Tensor) -> torch.Tensor:
        """``[B,V,K,h,w]`` prior-context map supplied to the revision cell.

        The stored ``prior_energy`` is ``[B,V,K]`` (a scalar level per candidate);
        it is expanded to the field grid so it enters the cell as a separate
        channel group rather than being folded into the measurement residual.
        """
        batch, views = field.shape[0], field.shape[1]
        prior = state.prior_energy
        if prior is None or prior.numel() == 0:
            prior = field.new_zeros(batch, views, self.config.K)
        prior = prior.reshape(batch, -1)  # any extra singleton axes are not evidence
        if prior.shape[1] != self.config.K:
            raise ValueError(
                f"prior energy carried {prior.shape[1]} groups, expected K={self.config.K}"
            )
        prior = prior.view(batch, 1, self.config.K).expand(batch, views, self.config.K)
        return (
            prior.detach()
            .view(batch, views, self.config.K, 1, 1)
            .expand(-1, -1, -1, *field.shape[-2:])
        )

    def _position_code(self, field: torch.Tensor) -> torch.Tensor:
        from owac.frameworks.n02.hypotheses import coordinate_features

        height, width = field.shape[-2:]
        key = (height, width, 4, str(field.device))
        cached = self._position_cache.get(key)
        if cached is None:
            cached = coordinate_features(height, width, 4, field.device).view(
                1, 1, 4, height, width
            )
            self._position_cache[key] = cached
        return cached.to(field.dtype).expand(field.shape[0], 1, -1, -1, -1)

    @staticmethod
    def _support_map(energy_result: EnergyResult, field: torch.Tensor) -> torch.Tensor:
        if energy_result.support_mask is not None:
            return energy_result.support_mask.unsqueeze(1).to(field.dtype)
        residual = energy_result.normalized_residual
        if residual.dim() == 3:
            residual = residual.unsqueeze(1)
        support = residual.detach()
        threshold = residual.mean() + residual.std()
        supported = (support <= threshold).to(field.dtype)
        # Match the field's view axis so the readout can pool it per view.
        if supported.dim() == 4:
            supported = supported.unsqueeze(1)
        return supported


@dataclass(slots=True)
class _Prediction:
    mean: torch.Tensor
    variance: torch.Tensor
    geometry: torch.Tensor


def _with_h(state: HypothesisState, field: torch.Tensor) -> HypothesisState:
    """Return the same bookkeeping state with a new field tensor attached."""
    return replace(state, h=field)


def build_encoder(config: N02Config) -> N02Encoder:
    """Factory used by the training/evaluation CLIs."""
    return N02Encoder(config)

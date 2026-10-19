"""Closest baselines and ablations for the N02 measurement-constrained field.

Every arm in this module is driven by the same
:class:`~owac.frameworks.n02.config.N02Config` family and, critically, by the
*same* measurement head and the *same* energy function
(:mod:`owac.frameworks.n02.measurement`). A baseline that scored itself with a
different likelihood or a second head would not be comparable with the candidate,
so no baseline here defines its own energy: the stem, the head, the measurement
buffer and :func:`~owac.frameworks.n02.measurement.compute_measurement_energy`
are reused verbatim.

Energy convention (quoted from ``measurement.py``, identical for every arm):

* per buffered observation the *mean over valid pixels* of
  ``gaussian_nll = 0.5 * r^2 / variance + 0.5 * log(variance)`` (Huber capped when
  ``robust='huber'``);
* the per-hypothesis energy is the sum over the *frozen unique* measurement set,
  so re-running a cycle never counts one frame twice;
* ``weights = softmax(-energy / temperature)``.

Arms
----
============== ==========================================================
``single_hypothesis``      N02's rule with ``K=1`` (one explanation).
``plain_recurrent``        ordinary single-field ConvGRU, same budget.
``dense_global_recurrent`` N02 with the update applied to the whole field.
``particle_filter``        differentiable particle filter (DPF rewrite).
``no_dependency_propagation`` / ``no_contradiction_reset`` / ``no_proposal``
                           N02 with one revision switch off.
``fixed_correct_prior`` / ``fixed_misleading_prior``
                           N02 with a *declared* prior level injected.
============== ==========================================================

Nothing in this module is a verified result. The per-arm share/differs record is
:func:`honest_notes`; it states what each arm keeps from N02, what it drops, and
what it therefore cannot claim, and it is what the experiment record should quote.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from owac.core.encoder_contract_n02 import (
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
from owac.frameworks.n02.hypotheses import HypothesisInitializer, coordinate_features
from owac.frameworks.n02.measurement import (
    EnergyResult,
    MeasurementModel,
    compute_measurement_energy,
)
from owac.frameworks.n02.model import N02Encoder, _masked_grid_measurement
from owac.frameworks.n02.readout import StructuredRepresentation
from owac.frameworks.n02.revision import RevisionController
from owac.frameworks.n02.state import (
    HypothesisState,
    MeasurementBuffer,
    estimate_variance,
    measurement_entry_from_frame,
)

__all__ = [
    "BASELINE_REGISTRY",
    "BaselineNote",
    "DenseGlobalRecurrentEncoder",
    "FixedPriorEncoder",
    "ParticleFilterEncoder",
    "PlainRecurrentEncoder",
    "PriorCondition",
    "SingleHypothesisEncoder",
    "ablation_config",
    "available_baselines",
    "build_baseline",
    "declared_prior_energy",
    "dense_global_recurrent",
    "fixed_correct_prior",
    "fixed_misleading_prior",
    "format_honest_notes",
    "honest_notes",
    "no_contradiction_reset",
    "no_dependency_propagation",
    "no_proposal",
    "particle_filter",
    "plain_recurrent",
    "single_hypothesis",
]


# --------------------------------------------------------------------------------------
# declared prior conditions
# --------------------------------------------------------------------------------------


class PriorCondition(StrEnum):
    """Condition of a *declared* prior handed to an encoder as explicit input.

    ``correct``/``misleading``/``uninformative``/``missing``/``stale`` mirror the
    synthetic generator's declared conditions, so the encoder-side switch and the
    data-side label use one vocabulary. The condition is context, never evidence:
    it is written into the prior term and the state's ``prior_energy`` only, and
    never into the measurement buffer.
    """

    CORRECT = "correct"
    MISLEADING = "misleading"
    UNINFORMATIVE = "uninformative"
    MISSING = "missing"
    STALE = "stale"

    @property
    def available(self) -> bool:
        """``False`` only for ``missing``: no prior context was supplied at all."""
        return self is not PriorCondition.MISSING

    @property
    def is_declaration(self) -> bool:
        """Whether the condition names a preferred candidate index."""
        return self in {PriorCondition.CORRECT, PriorCondition.STALE, PriorCondition.MISLEADING}


def declared_prior_energy(
    condition: PriorCondition | str,
    level: float,
    K: int,
    *,
    favoured_index: int = 0,
) -> torch.Tensor:
    """``[K]`` energy offsets implementing a declared prior condition.

    ``level >= 0`` is a strength in the *same energy units* as the measurement
    term; the offset is ``-level`` at the favoured candidate because lower energy
    means a better explanation. ``correct``/``stale`` favour ``favoured_index``,
    ``misleading`` favours the next index (a different candidate, i.e. a prior
    that contradicts the designated explanation), ``uninformative``/``missing``
    are all-zero.

    With the framework's score ``score_k = -(E_k + beta_prior * prior_k) / T`` the
    effective log-odds added to candidate ``j`` are
    ``beta_prior * level / T`` (see :meth:`FixedPriorEncoder.effective_log_odds`).
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if level < 0.0:
        raise ValueError("prior level is a strength and must be >= 0")
    condition = PriorCondition(condition)
    index = int(favoured_index) % K
    energy = torch.zeros(K, dtype=torch.float32)
    if condition in {PriorCondition.CORRECT, PriorCondition.STALE}:
        energy[index] = -float(level)
    elif condition is PriorCondition.MISLEADING:
        energy[(index + 1) % K] = -float(level)
    return energy


def prior_condition_is_degenerate(condition: PriorCondition | str, K: int) -> bool:
    """``True`` when the condition cannot be expressed with this many candidates.

    With ``K == 1`` the ``misleading`` condition has no other candidate to prefer,
    so it coincides with ``correct``; callers must report that rather than claim a
    misleading-prior intervention.
    """
    return PriorCondition(condition) is PriorCondition.MISLEADING and int(K) == 1


# --------------------------------------------------------------------------------------
# shared input/state plumbing for the standalone baselines
# --------------------------------------------------------------------------------------


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(slots=True)
class _Prediction:
    """Structural stand-in for ``MeasurementPrediction`` (only three fields are read)."""

    mean: torch.Tensor
    variance: torch.Tensor
    geometry: torch.Tensor


@dataclass(slots=True)
class _Prepared:
    """Normalised step inputs plus the frozen measurement buffer."""

    features: torch.Tensor  # stem features of the latest retained observation
    valid: torch.Tensor  # [B,C,H,W] bool
    saturated: torch.Tensor  # [B,C,H,W] bool
    buffer: MeasurementBuffer
    added: bool
    calibrated: bool


def _bool_mask(mask: Any, like: torch.Tensor, channels: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(like, dtype=torch.bool)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, device=like.device)
    mask = mask.to(like.device)
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.shape[1] != channels:
        mask = mask.expand(-1, channels, -1, -1)
    return mask


def _frame_tensors(
    frame: MeasurementFrame, *, device: torch.device, out_channels: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Promote a contract frame to ``[B,C,H,W]`` tensors (same rule as N02Encoder).

    The baselines must accept exactly the input the candidate accepts; this is the
    shared adapter, not part of any mechanism. Bad pixels are not needed here: they
    are carried by the buffered entry (``measurement_entry_from_frame`` stores the
    frame's own mask) and are excluded inside ``compute_measurement_energy``.
    """
    values = frame.values
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    values = values.to(device, torch.float32)
    if values.dim() == 2:
        values = values[None, None]
    elif values.dim() == 3:
        values = values[None]
    if values.dim() != 4:
        raise ValueError(f"frame values must be [B,C,H,W] or [C,H,W] or [H,W], got {values.dim()}D")
    channels = values.shape[1]
    if channels != out_channels:
        raise ValueError(
            f"input_mode expects {out_channels} channels, got {channels}; the CFA and "
            "three-channel inputs must use separate stems"
        )
    valid = _bool_mask(frame.valid_mask, values, channels)
    if frame.valid_mask is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    saturated = _bool_mask(frame.saturated_mask, values, channels)
    return values, valid, saturated


def _field_grid(camera_spec: CameraSpec) -> tuple[int, int]:
    """Mechanism grid declared for a camera (the rule is N02's, reused not guessed)."""
    height = max(1, camera_spec.height // N02Encoder._grid_stride(camera_spec.height))
    width = max(1, camera_spec.width // N02Encoder._grid_stride(camera_spec.width))
    return height, width


def _fresh_state(
    batch_size: int,
    K: int,
    width: int,
    grid: tuple[int, int],
    geometry_dim: int,
    device: torch.device,
) -> HypothesisState:
    height, width_px = grid
    h = torch.zeros(batch_size, 1, K, width, height, width_px, device=device)
    zeros = torch.zeros(batch_size, 1, K, device=device)
    return HypothesisState(
        h=h,
        measurement_energy=zeros.clone(),
        weights=torch.full_like(zeros, 1.0 / K),
        prior_energy=zeros.clone(),
        relation_energy=zeros.clone(),
        total_energy=zeros.clone(),
        logits=zeros.clone(),
        geometry=torch.zeros(batch_size, 1, K, geometry_dim, device=device),
        age=torch.zeros_like(zeros),
        times_reset=torch.zeros_like(zeros),
    )


def _reset_state(state: HypothesisState, batch_mask: Any, K: int) -> HypothesisState:
    """Reset the selected batch elements (same semantics as ``N02Encoder.reset``)."""
    if not torch.is_tensor(batch_mask):
        batch_mask = torch.as_tensor(batch_mask, device=state.h.device)
    batch_mask = batch_mask.to(state.h.device, torch.bool).view(-1)
    if batch_mask.numel() == 1:
        batch_mask = batch_mask.expand(state.h.shape[0])
    keep = (~batch_mask).view(-1, 1, 1, 1, 1, 1)
    view = (~batch_mask).view(-1, 1, 1)
    zeros = torch.zeros_like(state.measurement_energy)
    return HypothesisState(
        h=torch.where(keep, state.h, torch.zeros_like(state.h)),
        measurement_energy=torch.where(view, state.measurement_energy, zeros),
        weights=torch.where(view, state.weights, torch.full_like(state.weights, 1.0 / K)),
        prior_energy=torch.where(view, state.prior_energy, zeros),
        relation_energy=torch.where(view, state.relation_energy, zeros),
        total_energy=torch.where(view, state.total_energy, zeros),
        logits=torch.where(view, state.logits, zeros),
        geometry=torch.where(view.unsqueeze(-1), state.geometry, torch.zeros_like(state.geometry)),
        age=torch.where(view, state.age, zeros),
        times_reset=torch.where(view, state.times_reset, zeros),
        support=None,
        dependency_edge=state.dependency_edge,
        dependency_valid=state.dependency_valid,
        step_index=0,
    )


def _as_views(energy: torch.Tensor, views: int) -> torch.Tensor:
    """``[B,K]`` -> ``[B,V,K]`` (an explicit broadcast, never shape inference)."""
    if energy.dim() == 3:
        if energy.shape[1] != views:
            raise ValueError(f"energy carried {energy.shape[1]} views but the field has {views}")
        return energy
    if energy.dim() != 2:
        raise ValueError(f"unexpected energy shape {tuple(energy.shape)}")
    return energy.unsqueeze(1).expand(-1, views, -1)


def _residual_feature(residual: torch.Tensor, field: torch.Tensor, K: int) -> torch.Tensor:
    """``[B,K,h,w]`` pooled normalised residual -> ``[B,V,K,h,w]`` on the field grid."""
    if residual.dim() == 3:
        residual = residual.unsqueeze(1)
    if residual.shape[1] == 1 and K > 1:
        residual = residual.expand(-1, K, -1, -1)
    elif residual.shape[1] != K:
        raise ValueError(f"unexpected residual channels {residual.shape[1]}")
    shaped = residual.unsqueeze(1).to(field.dtype)  # [B,1,K,h,w]
    if tuple(shaped.shape[-2:]) != tuple(field.shape[-2:]):
        batch, _, channels, height, width_px = shaped.shape
        shaped = F.interpolate(
            shaped.reshape(batch * channels, 1, height, width_px),
            size=tuple(field.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, 1, channels, *field.shape[-2:])
    return shaped.expand(-1, field.shape[1], -1, -1, -1)


def _support_map(residual: torch.Tensor, views: int) -> torch.Tensor:
    """Same support rule as ``N02Encoder``: cells within mean + 1 std of the residual."""
    if residual.dim() == 3:
        residual = residual.unsqueeze(1)
    support = residual.detach()
    threshold = support.mean() + support.std()
    supported = (support <= threshold).to(residual.dtype)
    if supported.dim() == 4:
        supported = supported.unsqueeze(1)
    return supported.expand(-1, views, -1, -1, -1)


class _StandaloneBaseline(nn.Module):
    """Plumbing shared by the baselines that are not ``N02Encoder`` variants.

    Only input normalisation, the measurement buffer, the scoring pass and the
    output assembly live here; each subclass owns its own update rule.
    """

    def __init__(self, config: N02Config, *, width: int, head_K: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.field_width = int(width)
        # The head's grouped convolution is parameterised by the hypothesis count,
        # so an arm with a different K instantiates the *same head architecture* for
        # its own K; nothing else about the head or the energy changes.
        head_config = (
            config if head_K in (None, config.K) else dataclasses.replace(config, K=head_K)
        )
        head_config = dataclasses.replace(head_config, D=self.field_width)
        self.measurement = MeasurementModel(head_config)
        self._position_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------ contract
    def init_state(self, batch_size: int, camera_spec: CameraSpec, device: Any) -> HypothesisState:
        state = _fresh_state(
            int(batch_size),
            self.representation_K,
            self.field_width,
            _field_grid(camera_spec),
            self.config.measurement.geometry_dim,
            torch.device(device),
        )
        return self._decorate_state(state)

    def reset(self, state: HypothesisState, batch_mask: Any) -> HypothesisState:
        return self._decorate_state(_reset_state(state, batch_mask, self.representation_K))

    def detach_state(self, state: HypothesisState) -> HypothesisState:
        return state.detach()

    @property
    def representation_K(self) -> int:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def _decorate_state(self, state: HypothesisState) -> HypothesisState:
        """Hook for subclasses that need extra per-state bookkeeping."""
        return state

    # ------------------------------------------------------------------ internals
    def _buffer_for(
        self, state: HypothesisState, budget: ComputeBudget, device: Any
    ) -> MeasurementBuffer:
        buffer = state.diagnostics.get("_buffer")
        history = min(int(budget.max_history), int(self.config.history))
        if buffer is None or buffer.max_history != history:
            buffer = MeasurementBuffer(history, mask_history=True, device=device)
            state.diagnostics["_buffer"] = buffer
        return buffer.copy()

    def _prepare(
        self, frame: MeasurementFrame, state: HypothesisState, budget: ComputeBudget
    ) -> _Prepared:
        values, valid, bad, saturated = N02Encoder._frame_tensors(self, frame, state)
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
        saturated = ~valid & (F.adaptive_max_pool2d(saturated.float(), grid) > 0)
        variance, calibrated = estimate_variance(
            values,
            shot_scale=self.config.measurement.shot_scale,
            read_scale=self.config.measurement.read_scale,
            floor=self.config.measurement.variance_floor,
        )
        entry = measurement_entry_from_frame(
            frame,
            values=values,
            variance=variance.to(values.dtype),
            sample_xy=frame.sample_xy if torch.is_tensor(frame.sample_xy) else None,
            camera_id=frame.camera_id or "camera_0",
            frame_id=self._frame_id(frame),
            episode_id=self._episode_id(frame),
            timestamp=None if frame.timestamp is None else float(frame.timestamp),
        )
        entry.valid_mask = valid
        entry.saturated_mask = saturated
        entry.bad_mask = torch.zeros_like(valid)
        if supplied is not None:
            entry.variance = supplied.clamp_min(self.config.measurement.variance_floor)
            entry.metadata["variance_is_estimate"] = bool(
                frame.metadata.get("variance_is_estimate", False)
            )
            calibrated = not entry.metadata["variance_is_estimate"]
        buffer = self._buffer_for(state, budget, values.device)
        if frame.is_new_measurement:
            added = buffer.add(entry)
        else:
            buffer.rejected_count += 1
            added = False
        # The stem consumes the latest *retained* observation under the same mask
        # contract N02 uses, so both share one measurement encoding path.
        latest = buffer.stack_values()[:, -1]
        mask_latest = torch.ones_like(latest, dtype=torch.bool)
        if buffer.frames:
            last_mask = buffer.frames[-1].valid_mask
            if last_mask is not None and torch.is_tensor(last_mask):
                mask_latest = _bool_mask(last_mask, latest, latest.shape[1])
        bad_latest = torch.zeros_like(latest, dtype=torch.bool)
        sat_latest = torch.zeros_like(latest, dtype=torch.bool)
        features = self.measurement(
            latest,
            mask_latest,
            bad_latest,
            sat_latest,
            N02Encoder._sampling_metadata(self, frame, latest),
        )
        return _Prepared(
            features=features,
            valid=valid,
            saturated=saturated,
            buffer=buffer,
            added=added,
            calibrated=calibrated,
        )

    def _score(
        self, field: torch.Tensor, buffer: MeasurementBuffer
    ) -> tuple[EnergyResult, torch.Tensor]:
        """Score the field with the shared head and the shared energy function."""
        mean, log_variance, geometry = self.measurement.predict(field)
        prediction = _Prediction(mean, torch.exp(log_variance), geometry)
        buffer = buffer.copy()
        buffer._frames = buffer.frames[-1:]
        result = compute_measurement_energy(
            prediction, buffer, self.config.measurement, view_index=0
        )
        return result, geometry

    def _cycles(self, budget: ComputeBudget) -> int:
        return max(1, min(int(budget.max_updates), int(self.config.cycles)))

    def _positions(self, field: torch.Tensor) -> torch.Tensor:
        height, width = field.shape[-2:]
        key = (height, width, str(field.device))
        cached = self._position_cache.get(key)
        if cached is None:
            cached = coordinate_features(height, width, 4, field.device).view(
                1, 1, 4, height, width
            )
            self._position_cache[key] = cached
        return cached.to(field.dtype)

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

    def _output(
        self,
        *,
        state: HypothesisState,
        field: torch.Tensor,
        weights: torch.Tensor,
        energy: torch.Tensor,
        geometry: torch.Tensor,
        support: torch.Tensor,
        valid: torch.Tensor,
        saturated: torch.Tensor,
        result: EnergyResult,
        buffer: MeasurementBuffer,
        prepared: _Prepared,
        cycles_run: int,
        stop_reason: str,
        updated_fraction: float | None,
        diagnostics: dict[str, Any],
        times_reset: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Assemble the contract output shared by the standalone baselines."""
        config = self.config
        batch, views, K = field.shape[0], field.shape[1], field.shape[2]
        zeros = torch.zeros_like(energy)
        validity_flag = (~valid).all(dim=1).all(dim=1).all(dim=1)
        representation_diagnostics = {
            "baseline": diagnostics.get("baseline", type(self).__name__),
            "stop_reason": stop_reason,
            "calibrated_noise": prepared.calibrated,
            "new_evidence": prepared.added,
            "ambiguity_axis": "not_applicable_single_explanation" if K == 1 else "K_candidates",
        }
        representation = StructuredRepresentation(
            h=field,
            weights=weights,
            measurement_energy=energy,
            prior_energy=zeros,
            relation_energy=zeros,
            geometry=geometry,
            valid=~validity_flag.view(batch, 1),
            support=support,
            step_index=state.step_index + 1,
            diagnostics=representation_diagnostics,
        )
        new_state = HypothesisState(
            h=field,
            measurement_energy=energy.detach(),
            weights=weights.detach(),
            prior_energy=zeros.detach(),
            relation_energy=zeros.detach(),
            total_energy=energy.detach(),
            logits=(-energy / config.temperature).detach(),
            geometry=geometry.detach(),
            age=(state.age + 1.0).detach(),
            times_reset=(state.times_reset if times_reset is None else times_reset).detach(),
            support=support.detach(),
            dependency_edge=None,
            dependency_valid=None,
            step_index=state.step_index + 1,
            diagnostics={"step_index": state.step_index + 1, "_buffer": buffer},
        )
        cost = CostRecord(
            updates=cycles_run,
            history_len=len(buffer),
            hypotheses=K,
            updated_fraction=updated_fraction,
            state_bytes=new_state.state_bytes() + state_tensor_bytes(buffer.frames),
            notes={
                "views": int(views),
                "grid": [int(field.shape[-2]), int(field.shape[-1])],
                "stop_reason": stop_reason,
                "baseline": diagnostics.get("baseline", type(self).__name__),
            },
        )
        validity = Validity(
            valid_mask=valid,
            missing_mask=~valid,
            saturated_mask=saturated,
            unresolved_ambiguity=None,
            solved=True,
            reason=stop_reason,
        )
        return EncoderOutput(
            representation=representation,
            state=new_state,
            validity=validity,
            diagnostics={
                "cycles_run": cycles_run,
                "stop_reason": stop_reason,
                "observations": int(prepared.added),
                "observation_history": len(buffer),
                "duplicates_skipped": buffer.duplicate_count + buffer.rejected_count,
                "last_energy_mean": float(result.energy.detach().mean().item()),
                **diagnostics,
            },
            cost=cost,
        )


class _PlainConvGRUCell(nn.Module):
    """Ordinary ConvGRU gate, written from the reference notes.

    ``z = sigmoid(convz([h, x]))``, ``r = sigmoid(convr([h, x]))``,
    ``q = tanh(convq([r * h, x]))``, ``h' = (1 - z) * h + z * q``.

    The ``(1 - z) * h + z * q`` convention is the one recorded in
    ``third_party/reference-inspect/NOTES.md`` for the RAFT cell; the opposite
    sign convention would be a different dynamical system under the same name.
    """

    def __init__(self, width: int, evidence_channels: int, kernel: int = 3) -> None:
        super().__init__()
        padding = kernel // 2
        self.width = int(width)
        self.evidence_channels = int(evidence_channels)
        combined = self.width + self.evidence_channels
        self.gates = nn.Conv2d(combined, 2 * self.width, kernel, padding=padding)
        self.gates_norm = nn.GroupNorm(_group_count(2 * self.width), 2 * self.width)
        self.candidate = nn.Conv2d(combined, self.width, kernel, padding=padding)
        self.candidate_norm = nn.GroupNorm(_group_count(self.width), self.width)

    def forward(self, state: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        if state.dim() != 6 or state.shape[2] != 1:
            raise ValueError(
                f"plain recurrence state must be [B,V,1,D,h,w], got {tuple(state.shape)}"
            )
        batch, views, _, width, height, width_px = state.shape
        if evidence.shape[:2] != (batch, views) or evidence.shape[2] != self.evidence_channels:
            raise ValueError(
                f"evidence must be [B,V,{self.evidence_channels},h,w], got {tuple(evidence.shape)}"
            )
        flat = state.reshape(batch * views, width, height, width_px)
        evidence_flat = evidence.reshape(batch * views, self.evidence_channels, height, width_px)
        combined = torch.cat([flat, evidence_flat], dim=1)
        reset, update = self.gates_norm(self.gates(combined)).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        hidden_input = torch.cat([flat * reset, evidence_flat], dim=1)
        candidate = torch.tanh(self.candidate_norm(self.candidate(hidden_input)))
        updated = (1.0 - update) * flat + update * candidate
        return updated.reshape(batch, views, 1, width, height, width_px)


class PlainRecurrentEncoder(_StandaloneBaseline):
    """Ordinary single-field ConvGRU recurrence at the same information budget.

    This is the key control: it sees the same frames, the same history window, the
    same number of cycles, the same stem/head architecture and the same energy
    function, and it also receives the current prediction residual as input. It has
    no hypothesis axis (``K=1`` honestly reported), no contradiction map, no
    dependency edges, no proposal and no revision scheduling: every cycle updates
    the whole field through one shared ConvGRU.

    The stem remains unchanged. The recurrent field is widened to match the
    candidate's useful trainable capacity as closely as integer widths allow;
    actual parameter counts are reported, including its declared-prior input.
    """

    def __init__(self, config: N02Config, *, width: int | None = None) -> None:
        if width is None:
            # Match useful trainable capacity by widening the actual recurrent
            # field, rather than attaching unused parameters to the control.
            with torch.random.fork_rng(devices=[]):
                reference = N02Encoder(config)
                sample = torch.empty(1, 3 * config.D + 6, 1, 1)
                reference.reviser.cell.input.initialize_parameters(sample)
                reference.reviser.cell.hidden.initialize_parameters(sample)
                target = sum(p.numel() for p in reference.parameters())
                candidates = []
                for trial in range(max(8, config.stem_width // 2), 3 * config.D + 1):
                    control = PlainRecurrentEncoder(config, width=trial)
                    count = sum(p.numel() for p in control.parameters())
                    candidates.append((abs(count - target), trial))
                width = min(candidates)[1]
            self.parameter_match_target = target
        width = int(width)
        super().__init__(config, width=width, head_K=1)
        self._representation_K = 1
        self.initializer = HypothesisInitializer(width, 1, context_dim=2, code_scale=0.5)
        self.evidence_channels = width + 1 + 4 + 2  # features, residual, position, declared prior
        self.cell = _PlainConvGRUCell(width, self.evidence_channels, kernel=config.cell_kernel)

    @property
    def representation_K(self) -> int:
        return self._representation_K

    def step(
        self,
        frame: MeasurementFrame,
        context: EmbodimentContext,
        state: HypothesisState,
        budget: ComputeBudget,
    ) -> EncoderOutput:
        config = self.config
        prepared = self._prepare(frame, state, budget)
        prior = prepared.features.new_zeros(prepared.features.shape[0], 2)
        if context.hypothesis_prior is not None:
            mass, available = N02Encoder._declared_prior(
                self, context, prior.shape[0], prior.device
            )
            parity_mass = torch.stack([mass[:, ::2].sum(-1), mass[:, 1::2].sum(-1)], -1)
            prior[:, 0] = (
                parity_mass[:, 0].clamp_min(1e-12).log() - parity_mass[:, 1].clamp_min(1e-12).log()
            ) * available
            prior[:, 1] = available
        field = self._ensure_field(state, prepared.features, prior)
        views = field.shape[1]
        position = self._positions(field).expand(field.shape[0], views, -1, -1, -1)
        evidence_base = torch.cat(
            [
                prepared.features.unsqueeze(1).expand(-1, views, -1, -1, -1),
                position,
                prior[:, None, :, None, None].expand(-1, views, -1, *field.shape[-2:]),
            ],
            dim=2,
        )
        max_cycles = self._cycles(budget)
        tolerance = config.revision.tolerance
        previous_energy: float | None = None
        cycles_run = 0
        stop_reason = "max_iterations"
        for iteration in range(max_cycles):
            cycles_run = iteration + 1
            result, _ = self._score(field, prepared.buffer)
            residual = _residual_feature(result.normalized_residual, field, 1)
            updated = self.cell(field, torch.cat([evidence_base, residual], dim=2))
            change = float((updated - field).abs().mean().item())
            field = updated
            current = float(_as_views(result.energy, views).detach().mean().item())
            delta = abs(current - previous_energy) if previous_energy is not None else float("inf")
            previous_energy = current
            if (
                cycles_run >= max(1, config.revision.min_iterations)
                and change <= tolerance
                and delta <= tolerance
            ):
                stop_reason = "converged"
                break
        result, geometry = self._score(field, prepared.buffer)
        energy = _as_views(result.energy, views)
        weights = torch.ones_like(energy)
        return self._output(
            state=state,
            field=field,
            weights=weights,
            energy=energy,
            geometry=geometry,
            support=_support_map(result.normalized_residual, views),
            valid=prepared.valid,
            saturated=prepared.saturated,
            result=result,
            buffer=prepared.buffer,
            prepared=prepared,
            cycles_run=cycles_run,
            stop_reason=stop_reason,
            updated_fraction=1.0,
            diagnostics={
                "baseline": "plain_recurrent",
                "cycles_budget": max_cycles,
                "hypotheses": 1,
                "explanation": "single_field_global_convgru_no_revision",
            },
        )

    def _ensure_field(
        self, state: HypothesisState, features: torch.Tensor, prior=None
    ) -> torch.Tensor:
        cold = state.h.abs().flatten(1).sum(-1) == 0
        if not bool(cold.any()):
            return state.h
        initialised = self.initializer(features, context=prior)  # [B,1,width,h,w]
        return torch.where(cold[:, None, None, None, None, None], initialised.unsqueeze(1), state.h)


class ParticleFilterEncoder(_StandaloneBaseline):
    """Differentiable particle / multi-hypothesis filter, DPF rewritten minimally.

    Each of the ``K`` particles is one candidate *field*, scored by the shared
    measurement head and the shared current-frame energy function. Kept
    from ``third_party/reference-inspect/dpf.py``: the multi-hypothesis particle
    set, the proposal branch with an iteration-decayed ``propose_ratio``, the
    systematic resampler (``linspace`` markers plus a uniform offset compared
    against the cumulative weights), weight normalisation, count preservation at
    ``K``, and stop-gradient on the resampled branch.

    Dropped, and stated here rather than implied: the 3-D pose state and the
    motion/odometry model (this encoder is given no executed action, so carried
    particles play the motion step and the dynamics are identity), the action-noise
    generator, the sigmoid ``obs_like_estimator`` with its ``min_obs_likelihood``
    clamp (the shared Gaussian NLL replaces it so the likelihood is identical to
    N02's), the DPF training curriculum, and the TensorFlow/sonnet graph.

    Weights use the current frame only in each cycle; retained particle fields
    carry history through the transition. Re-scoring the whole buffer as fresh
    evidence would count retained frames repeatedly. Declared prior context is
    available to both initialization and proposals, as in the other arms.
    """

    def __init__(
        self,
        config: N02Config,
        *,
        propose_ratio: float = 0.5,
        stop_resample_gradient: bool = True,
        resample_jitter: bool = True,
    ) -> None:
        width = int(config.stem_width)
        super().__init__(config, width=width)
        if not 0.0 <= propose_ratio <= 1.0:
            raise ValueError("propose_ratio must be in [0, 1]")
        self._representation_K = int(config.K)
        self.propose_ratio = float(propose_ratio)
        self.stop_resample_gradient = bool(stop_resample_gradient)
        self.resample_jitter = bool(resample_jitter)
        self.initializer = HypothesisInitializer(
            width, config.K, context_dim=config.K + 1, code_scale=0.5
        )
        self.proposer = HypothesisInitializer(
            width, config.K, context_dim=config.K + 1, code_scale=0.5
        )
        self._generator = torch.Generator(device="cpu").manual_seed(int(config.seed))

    @property
    def representation_K(self) -> int:
        return self._representation_K

    def step(
        self,
        frame: MeasurementFrame,
        context: EmbodimentContext,
        state: HypothesisState,
        budget: ComputeBudget,
    ) -> EncoderOutput:
        config = self.config
        K = self._representation_K
        prepared = self._prepare(frame, state, budget)
        features = prepared.features
        basis = features.reshape(features.shape[0], 1, *features.shape[1:])
        mass, available = N02Encoder._declared_prior(self, context, basis.shape[0], basis.device)
        prior = torch.cat([mass, available[:, None].to(mass.dtype)], -1)
        field = self._ensure_particles(state, basis, prior)
        views = field.shape[1]
        max_cycles = self._cycles(budget)
        proposal_counts: list[int] = []
        replaced = torch.zeros(field.shape[0], views, K, device=field.device)
        for iteration in range(max_cycles):
            result, _ = self._score(field, prepared.buffer)
            weights = torch.softmax(-_as_views(result.energy, views) / config.temperature, dim=-1)
            # DPF decays the proposal ratio with the loop counter, which starts at
            # 1 there (the first particle set is already proposed): cycle ``i``
            # proposes ``propose_ratio ** (i + 1) * K`` particles.
            n_proposed = int(round(self.propose_ratio ** (iteration + 1) * K))
            n_proposed = max(0, min(K, n_proposed))
            proposal_counts.append(n_proposed)
            pieces: list[torch.Tensor] = []
            if K - n_proposed > 0:
                indices = self._systematic_resample(weights, K - n_proposed)
                gather = indices.view(field.shape[0], views, -1, 1, 1, 1).expand(
                    -1, -1, -1, self.field_width, *field.shape[-2:]
                )
                resampled = torch.gather(field, 2, gather)
                pieces.append(resampled.detach() if self.stop_resample_gradient else resampled)
            if n_proposed > 0:
                proposed = self._propose(basis, field.shape[0], views, prior)
                pieces.append(proposed[:, :, :n_proposed])
            field = torch.cat(pieces, dim=2) if len(pieces) > 1 else pieces[0]
            if n_proposed:
                replaced[:, :, K - n_proposed :] = replaced[:, :, K - n_proposed :] + 1.0
        result, geometry = self._score(field, prepared.buffer)
        energy = _as_views(result.energy, views)
        weights = torch.softmax(-energy / config.temperature, dim=-1)
        times_reset = state.times_reset + replaced
        return self._output(
            state=state,
            field=field,
            weights=weights,
            energy=energy,
            geometry=geometry,
            support=_support_map(result.normalized_residual, views),
            valid=prepared.valid,
            saturated=prepared.saturated,
            result=result,
            buffer=prepared.buffer,
            prepared=prepared,
            cycles_run=max_cycles,
            stop_reason="max_iterations",
            updated_fraction=float(sum(proposal_counts) / (max_cycles * K)),
            times_reset=times_reset,
            diagnostics={
                "baseline": "particle_filter",
                "particles": K,
                "proposal_counts": proposal_counts,
                "proposed_last": proposal_counts[-1] if proposal_counts else 0,
                "resampled_last": K - (proposal_counts[-1] if proposal_counts else 0),
                "propose_ratio": self.propose_ratio,
                "stop_resample_gradient": self.stop_resample_gradient,
                "resample_jitter": self.resample_jitter,
            },
        )

    # ------------------------------------------------------------------ particles
    def _ensure_particles(
        self, state: HypothesisState, basis: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        cold = state.h.abs().flatten(1).sum(-1) == 0
        if not bool(cold.any()):
            return state.h
        batch, views = basis.shape[:2]
        flat = basis.reshape(batch * views, *basis.shape[2:])
        particles = self.initializer(flat, context=context)
        initialised = particles.reshape(
            batch, views, self._representation_K, self.field_width, *basis.shape[-2:]
        )
        return torch.where(cold[:, None, None, None, None, None], initialised, state.h)

    def _propose(
        self, basis: torch.Tensor, batch: int, views: int, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        flat = basis.reshape(batch * views, *basis.shape[2:])
        proposed = self.proposer(flat, context=context)
        return proposed.reshape(
            batch, views, self._representation_K, self.field_width, *basis.shape[-2:]
        )

    def _systematic_resample(self, weights: torch.Tensor, count: int) -> torch.Tensor:
        """Systematic resampling: ``linspace`` markers plus a uniform offset.

        ``weights`` is ``[B,V,K]``; the returned indices are ``[B,V,count]`` and
        index the particle axis. The uniform offset is drawn from a generator seeded
        with ``config.seed`` so a run is reproducible (DPF draws it fresh each step).
        """
        batch, views, K = weights.shape
        if count < 1 or count > K:
            raise ValueError(f"cannot resample {count} particles out of {K}")
        flat = weights.reshape(batch * views, K)
        cumulative = flat.cumsum(dim=-1)
        cumulative = cumulative / cumulative[..., -1:].clamp_min(1.0e-12)
        if self.resample_jitter:
            offset = torch.rand(batch * views, 1, generator=self._generator) / count
        else:
            offset = torch.full((batch * views, 1), 0.5 / count)
        markers = offset + torch.linspace(0.0, (count - 1) / count, count).view(1, -1)
        matches = markers.unsqueeze(-1) < cumulative.unsqueeze(1)  # [N,count,K]
        indices = matches.to(torch.int64).argmax(dim=-1)
        return indices.reshape(batch, views, count)


class SingleHypothesisEncoder(N02Encoder):
    """N02's rule with exactly one explanation (``K=1``).

    Same stem, head, energy, cycles, history and revision code path; the only
    change is the hypothesis count. This is the honest "does having ``K>1`` help at
    all" control, and the representation reports ``K=1`` rather than padding slots.
    """

    def __init__(self, config: N02Config) -> None:
        self.requested_K = int(config.K)
        super().__init__(dataclasses.replace(config, K=1))


class _GlobalSelectionRevision(RevisionController):
    """N02's revision controller with the update selection forced to the whole field.

    The location gate is still constructed, so this arm has exactly N02's
    parameterisation, but its output is ignored: ``location_mask`` returns an
    all-ones selection, so every cell receives its candidate update.
    Contradiction localisation, dependency propagation, reset and the proposal
    branch are otherwise untouched. A float all-ones tensor is compatible with
    the controller's hard forward selection and straight-through gradient.
    """

    def location_mask(
        self,
        contradiction: torch.Tensor,
        residual_feature: torch.Tensor,
        prior_feature: torch.Tensor,
        location_code: torch.Tensor,
        *,
        cold_start: bool,
    ) -> torch.Tensor:
        del contradiction, prior_feature, location_code, cold_start
        return torch.ones_like(residual_feature, dtype=residual_feature.dtype).expand(
            -1, -1, self.config.K, -1, -1
        )


class DenseGlobalRecurrentEncoder(N02Encoder):
    """N02 with every cycle updating the whole field instead of a selected region.

    Same ``K``, same stem/head/energy, same contradiction map, propagation, reset,
    proposal and cycle budget. The only change is that the shared update cell is
    applied everywhere, which isolates the value of *locality*.
    """

    def __init__(self, config: N02Config) -> None:
        super().__init__(config)
        self.reviser = _GlobalSelectionRevision(config, width=config.D)


class FixedPriorEncoder(N02Encoder):
    """N02 with a declared prior level injected into the prior term.

    The condition and level are constructor inputs and are applied through
    :meth:`_prior_energy`, so they land in the state's ``prior_energy`` and in the
    representation's prior term. They are never written into the measurement
    buffer, so a declared prior can never be mistaken for evidence. The effective
    log-odds of the declaration are ``beta_prior * level / temperature`` at the
    favoured candidate.
    """

    def __init__(
        self,
        config: N02Config,
        condition: PriorCondition | str = PriorCondition.CORRECT,
        *,
        level: float = 1.0,
        favoured_index: int = 0,
    ) -> None:
        super().__init__(dataclasses.replace(config, prior_mode="fixed_odds"))
        self.prior_condition = PriorCondition(condition)
        self.prior_level = float(level)
        self.favoured_index = int(favoured_index) % self.config.K
        self.prior_degenerate = prior_condition_is_degenerate(self.prior_condition, self.config.K)
        self.register_buffer(
            "prior_offset",
            declared_prior_energy(
                self.prior_condition, self.prior_level, self.config.K, favoured_index=favoured_index
            ),
        )

    @property
    def effective_log_odds(self) -> torch.Tensor:
        """``[K]`` log-odds the declaration adds to each candidate's score."""
        scale = self._bounded_beta_prior() / self.config.temperature
        return (-self.prior_offset) * scale

    def _prior_energy(self, state: HypothesisState, *, cold_start: bool) -> torch.Tensor:
        base = super()._prior_energy(state, cold_start=cold_start)
        return base + self.prior_offset.to(base.dtype).view(1, 1, -1)

    def init_state(self, batch_size: int, camera_spec: CameraSpec, device: Any) -> HypothesisState:
        state = super().init_state(batch_size, camera_spec, device)
        state.prior_energy = state.prior_energy + self.prior_offset.to(
            state.prior_energy.dtype
        ).view(1, 1, -1)
        return state

    def reset(self, state: HypothesisState, batch_mask: Any) -> HypothesisState:
        out = super().reset(state, batch_mask)
        mask = batch_mask if torch.is_tensor(batch_mask) else torch.as_tensor(batch_mask)
        mask = mask.to(out.h.device, torch.bool).view(-1)
        if mask.numel() == 1:
            mask = mask.expand(out.h.shape[0])
        offset = (
            self.prior_offset.to(out.prior_energy.dtype).view(1, 1, -1).expand_as(out.prior_energy)
        )
        out.prior_energy = torch.where(mask.view(-1, 1, 1), out.prior_energy, offset)
        return out

    def step(
        self,
        frame: MeasurementFrame,
        context: EmbodimentContext,
        state: HypothesisState,
        budget: ComputeBudget,
    ) -> EncoderOutput:
        context = dataclasses.replace(
            context,
            hypothesis_prior=torch.softmax(-self.prior_offset, dim=-1).expand(state.h.shape[0], -1),
            hypothesis_prior_available=True,
        )
        output = super().step(frame, context, state, budget)
        representation = output.representation
        if isinstance(representation, StructuredRepresentation):
            representation.diagnostics["prior_condition"] = self.prior_condition.value
            representation.diagnostics["prior_level"] = self.prior_level
            representation.diagnostics["prior_favoured_index"] = self.favoured_index
            representation.diagnostics["prior_available"] = self.prior_condition.available
            representation.diagnostics["prior_degenerate"] = self.prior_degenerate
        return output


# --------------------------------------------------------------------------------------
# ablation configuration helpers and factories
# --------------------------------------------------------------------------------------


def ablation_config(config: N02Config, **revision_overrides: Any) -> N02Config:
    """Return ``config`` with ``RevisionConfig`` fields overridden (validated by the dataclass).

    Every ablation in this module is built this way. The candidate code path is not
    copied: switching a flag off is the whole intervention.
    """
    revision = dataclasses.replace(config.revision, **revision_overrides)
    return dataclasses.replace(config, revision=revision)


def single_hypothesis(config: N02Config) -> nn.Module:
    """``K=1`` N02 (see :class:`SingleHypothesisEncoder`)."""
    return SingleHypothesisEncoder(config)


def plain_recurrent(config: N02Config) -> nn.Module:
    """Single-field ConvGRU recurrence at the same budget (see :class:`PlainRecurrentEncoder`)."""
    return PlainRecurrentEncoder(config)


def dense_global_recurrent(config: N02Config) -> nn.Module:
    """N02 with global updates (see :class:`DenseGlobalRecurrentEncoder`)."""
    return DenseGlobalRecurrentEncoder(config)


def particle_filter(
    config: N02Config,
    *,
    propose_ratio: float = 0.5,
    stop_resample_gradient: bool = True,
    resample_jitter: bool = True,
) -> nn.Module:
    """Differentiable particle filter (see :class:`ParticleFilterEncoder`)."""
    return ParticleFilterEncoder(
        config,
        propose_ratio=propose_ratio,
        stop_resample_gradient=stop_resample_gradient,
        resample_jitter=resample_jitter,
    )


def no_dependency_propagation(config: N02Config) -> nn.Module:
    """N02 with ``propagation_enabled=False``: the selection is the contradiction map."""
    return N02Encoder(ablation_config(config, propagation_enabled=False))


def no_contradiction_reset(config: N02Config) -> nn.Module:
    """N02 with ``reset_contradicted=False``: nothing is ever rebuilt by the proposal."""
    return N02Encoder(ablation_config(config, reset_contradicted=False))


def no_proposal(config: N02Config) -> nn.Module:
    """N02 with ``proposal_enabled=False``: refuted cells keep the shared cell's output."""
    return N02Encoder(ablation_config(config, proposal_enabled=False))


def fixed_correct_prior(
    config: N02Config, *, level: float = 1.0, favoured_index: int = 0
) -> nn.Module:
    """N02 with a declared prior favouring the designated candidate index."""
    return FixedPriorEncoder(
        config, PriorCondition.CORRECT, level=level, favoured_index=favoured_index
    )


def fixed_misleading_prior(
    config: N02Config, *, level: float = 1.0, favoured_index: int = 0
) -> nn.Module:
    """N02 with a declared prior favouring a *different* candidate index."""
    return FixedPriorEncoder(
        config, PriorCondition.MISLEADING, level=level, favoured_index=favoured_index
    )


BASELINE_REGISTRY: dict[str, Callable[[N02Config], nn.Module]] = {
    "single_hypothesis": single_hypothesis,
    "plain_recurrent": plain_recurrent,
    "dense_global_recurrent": dense_global_recurrent,
    "particle_filter": particle_filter,
    "no_dependency_propagation": no_dependency_propagation,
    "no_contradiction_reset": no_contradiction_reset,
    "no_proposal": no_proposal,
    "fixed_correct_prior": fixed_correct_prior,
    "fixed_misleading_prior": fixed_misleading_prior,
}


def available_baselines() -> tuple[str, ...]:
    """Registry names, in declaration order."""
    return tuple(BASELINE_REGISTRY)


def build_baseline(name: str, config: N02Config, **kwargs: Any) -> nn.Module:
    """Build one registry arm. Extra keyword arguments go to its factory."""
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](config, **kwargs)


# --------------------------------------------------------------------------------------
# honest share/differs record
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineNote:
    """What one arm shares with the candidate and what it therefore cannot claim."""

    name: str
    shares: tuple[str, ...]
    differs: tuple[str, ...]
    reference: str = ""


def honest_notes() -> dict[str, BaselineNote]:
    """Per-arm share/differs record; the experiment record should quote this verbatim."""
    common = (
        "same N02Config family (K/D/cycles/history/temperature/beta_prior/noise model)",
        "same measurement stem architecture, same measurement head, same robust Gaussian "
        "NLL energy over the frozen unique measurement set",
        "same measurement buffer rules (unique identity, bounded history, padding rejected)",
        "same Encoder contract and the same StructuredRepresentation/readouts",
    )
    return {
        "single_hypothesis": BaselineNote(
            name="single_hypothesis",
            shares=common + ("same revision rule code path (contradiction/propagation/proposal)",),
            differs=(
                "K=1: one explanation, no competition, weights are identically 1.0",
                "the contradiction test compares the single candidate with itself, so no cell "
                "can be refuted relative to a rival; the framework's ambiguity threshold "
                "(uniform = 1/K = 1) then reports the field as unresolved by construction",
            ),
            reference="control: does having K>1 help at all",
        ),
        "plain_recurrent": BaselineNote(
            name="plain_recurrent",
            shares=(
                "same N02Config family and same frames/history window/cycle budget",
                "same measurement stem and head, same energy function, same buffer rules",
                "same Encoder contract, StructuredRepresentation and readouts",
                "global update every cycle, and the current normalised residual is supplied as "
                "input (so this arm is not information-starved relative to N02's cell)",
            ),
            differs=(
                "one field, no hypothesis axis: K=1 is reported honestly, weights are not a "
                "posterior over explanations",
                "no contradiction map, no dependency edges or messages, no proposal, no revision "
                "scheduling, no prior/relation energy terms",
                "its own ConvGRU weights (an ordinary recurrence cannot share the candidate's "
                "revision cell and stay ordinary)",
                "field width follows the head (`stem_width`), not the candidate's per-hypothesis "
                "code width D",
            ),
            reference="key baseline: ordinary recurrence with the same budget",
        ),
        "dense_global_recurrent": BaselineNote(
            name="dense_global_recurrent",
            shares=(
                "N02Encoder verbatim: same parameterisation (the location gate is still built), "
                "same K, same head/energy, same contradiction map, propagation, reset, proposal, "
                "cycles and history",
            ),
            differs=(
                "the update selection is the whole field: `location_mask` returns every cell and "
                "the gate's output is ignored, so `updated_fraction` is always 1.0",
                "the gate parameters exist but receive no gradient in this arm",
            ),
            reference="isolates the value of locality",
        ),
        "particle_filter": BaselineNote(
            name="particle_filter",
            shares=(
                "same K particles/candidates, same stem and head, same robust Gaussian NLL "
                "likelihood over the frozen buffer, same history and cycle budget",
                "same Encoder contract, StructuredRepresentation and readouts; weights are a "
                "normalised distribution over the K candidates",
                "DPF kept: multi-hypothesis particle set, proposer branch with iteration-decayed "
                "propose_ratio, systematic resampling (linspace markers + uniform offset against "
                "the cumulative weights), weight normalisation, count preservation at K, and "
                "stop-gradient on the resampled branch (switchable)",
            ),
            differs=(
                "DPF dropped: the 3-D pose state and motion/odometry model (no executed action is "
                "available in this contract, so carried particles are the motion step and the "
                "dynamics are identity), the action-noise generator, the sigmoid "
                "obs_like_estimator with its min_obs_likelihood clamp, the per-module pretraining "
                "curriculum and the TensorFlow/sonnet graph",
                "deliberate deviation: weights are the softmax of the energy of the frozen buffer "
                "each cycle, not a running product of per-frame likelihoods (a running product "
                "would re-count a retained frame once per later step)",
                "particles are fields on the measurement grid, not 3-D poses; resampling uses a "
                "seeded generator so a run is reproducible",
            ),
            reference="multi-hypothesis filter control (arXiv:1805.11122, rewritten minimally)",
        ),
        "no_dependency_propagation": BaselineNote(
            name="no_dependency_propagation",
            shares=common + ("identical code path and parameters to N02",),
            differs=(
                "`propagation_enabled=False`: `RevisionController.propagate` returns the direct "
                "contradiction map unchanged, so no neighbour ever inherits revision priority",
                "measured caveat: with the edge table at its identity initialisation the decayed "
                "neighbour vote stays below `propagation_decay` for K>=2, so at initialisation "
                "this ablation records the same selection as the candidate and can only differ "
                "once training grows the learnable edge scale",
            ),
            reference="ablation: dependency-propagated selection",
        ),
        "no_contradiction_reset": BaselineNote(
            name="no_contradiction_reset",
            shares=common + ("identical code path and parameters to N02",),
            differs=(
                "`reset_contradicted=False`: `last_reset` is always all-False and `times_reset` "
                "never grows, so a refuted cell is only moved by the shared update cell and never "
                "rebuilt from the current residual",
            ),
            reference="ablation: contradiction-driven reset",
        ),
        "no_proposal": BaselineNote(
            name="no_proposal",
            shares=common + ("identical code path and parameters to N02",),
            differs=(
                "`proposal_enabled=False`: cells selected for reset keep the shared cell's output "
                "instead of the proposal head's rebuild; the proposal parameters still exist but "
                "receive no gradient in this arm",
            ),
            reference="ablation: proposal rebuild after a refutation",
        ),
        "fixed_correct_prior": BaselineNote(
            name="fixed_correct_prior",
            shares=(
                "the whole candidate: N02Encoder with the same switches, parameters and energy",
            ),
            differs=(
                "a declared prior level is added to the prior term and written into the state's "
                "prior_energy (`prior_mode='fixed_odds'`); no prior is learned or inferred",
                "the declaration is explicit constructor input and never enters the measurement "
                "buffer, so it cannot be mistaken for evidence",
                "hand-set correct/misleading priors are a labelled mechanism intervention, not a "
                "deployable prior source",
            ),
            reference="intervention: prior condition `correct`",
        ),
        "fixed_misleading_prior": BaselineNote(
            name="fixed_misleading_prior",
            shares=(
                "the whole candidate: N02Encoder with the same switches, parameters and energy",
            ),
            differs=(
                "the declared prior favours a different candidate index than "
                "`fixed_correct_prior`, with the same level",
                "with K=1 the condition is degenerate (no other candidate to prefer) and the "
                "representation reports `prior_degenerate=True`",
                "as above, the declaration is explicit input and never a measurement",
            ),
            reference="intervention: a confident but wrong prior must be correctable by evidence",
        ),
    }


def format_honest_notes(notes: dict[str, BaselineNote] | None = None) -> str:
    """Markdown rendering of :func:`honest_notes` for the experiment record."""
    notes = honest_notes() if notes is None else notes
    lines = [
        "| arm | shares with N02 | differs from N02 | reference |",
        "| --- | --- | --- | --- |",
    ]
    for name, note in notes.items():
        shares = "<br>".join(note.shares)
        differs = "<br>".join(note.differs)
        lines.append(f"| `{name}` | {shares} | {differs} | {note.reference} |")
    return "\n".join(lines)

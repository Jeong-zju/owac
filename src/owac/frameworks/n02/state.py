"""N02 storage objects: the measurement buffer and the hypothesis state field.

Two facts are kept strictly apart.

* :class:`MeasurementBuffer` stores only real observations: their values, the
  sampling condition that gives them meaning, unique identities and masks. A
  model prediction never enters this buffer.
* :class:`HypothesisState` stores the ``h[B,V,K,D,h,w]`` field, hypothesis
  scores, local revision state, dependency edges and ages.

The buffer's identity rule is the guard against double-counted evidence: a frame
whose ``new_measurement`` flag is false (padding, or the same frame re-submitted
inside one policy step) is recorded as a duplicate and contributes no energy.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import torch

from owac.core.encoder_contract import MeasurementFrame


@dataclass(slots=True)
class MeasurementEntry:
    """One unique observation retained in the buffer."""

    values: torch.Tensor  # [B, C, H, W] in raw numeric units
    valid_mask: torch.Tensor | None  # [B, 1, H, W] or per-plane, bool
    bad_mask: torch.Tensor | None
    saturated_mask: torch.Tensor | None
    variance: torch.Tensor  # [B, C, H, W] measurement variance, same units squared
    sample_xy: torch.Tensor | None  # [B, C, 2] sampling coordinate/phase
    camera_id: str
    frame_id: int
    episode_id: int | None
    timestamp: float | None
    time_unit: str
    metadata: dict = field(default_factory=dict)
    batch_identities: tuple = ()
    new_mask: torch.Tensor | None = None

    @property
    def identity(self) -> tuple[str, int, int | None]:
        return (self.camera_id, self.frame_id, self.episode_id)

    @property
    def observed_count(self) -> int:
        """Number of individually observed scalar measurements that count."""
        mask = self.valid_mask
        if mask is None:
            return int(self.values.numel())
        return int(mask.sum().item())

    def select_batch(self, index) -> MeasurementEntry:
        def take(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                if value.shape[0] > index:
                    return value[index : index + 1]
                return value
            return value

        return MeasurementEntry(
            values=self.values[index : index + 1],
            valid_mask=take(self.valid_mask),
            bad_mask=take(self.bad_mask),
            saturated_mask=take(self.saturated_mask),
            variance=self.variance[index : index + 1],
            sample_xy=take(self.sample_xy),
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            episode_id=self.episode_id,
            timestamp=self.timestamp,
            time_unit=self.time_unit,
            metadata=dict(self.metadata),
        )


class MeasurementBuffer:
    """Bounded causal history of unique observations, per batch element.

    Entries are appended newest-last. ``max_history`` bounds the number of
    retained frames; ``mask_history=False`` keeps every frame this episode (the
    exact static reference example uses it to accumulate the whole buffer).
    """

    def __init__(
        self,
        max_history: int,
        *,
        mask_history: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.max_history = int(max_history)
        self.mask_history = bool(mask_history)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._frames: list[MeasurementEntry] = []
        self._seen: set[tuple[str, int, int | None]] = set()
        self.duplicate_count = 0
        self.rejected_count = 0
        self.last_added: torch.Tensor | None = None
        self._latest: dict = {}

    def copy(self) -> MeasurementBuffer:
        result = MeasurementBuffer(
            self.max_history, mask_history=self.mask_history, device=self.device
        )
        result._frames = list(self._frames)
        result._seen = set(self._seen)
        result.duplicate_count = self.duplicate_count
        result.rejected_count = self.rejected_count
        result.last_added = self.last_added
        result._latest = dict(self._latest)
        return result

    def clear_batch(self, mask: torch.Tensor) -> MeasurementBuffer:
        result = self.copy()
        if bool(mask.all()):
            result.reset()
            result.last_added = torch.zeros_like(mask)
            return result
        result._seen = {key for key in result._seen if not bool(mask[key[0]])}
        result._latest = {
            key: value for key, value in result._latest.items() if not bool(mask[key])
        }
        result._frames = [
            dataclasses.replace(
                entry,
                valid_mask=(
                    torch.ones_like(entry.values, dtype=torch.bool)
                    if entry.valid_mask is None
                    else entry.valid_mask
                )
                & ~mask.reshape(-1, 1, 1, 1),
            )
            for entry in result._frames
        ]
        return result

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> list[MeasurementEntry]:
        return list(self._frames)

    @property
    def unique_evidence_count(self) -> int:
        """Number of unique observations currently supporting the field."""
        return len(self._frames)

    @property
    def observed_measurement_count(self) -> int:
        return sum(frame.observed_count for frame in self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._seen.clear()
        self.duplicate_count = 0
        self.rejected_count = 0
        self._latest.clear()
        self.last_added = None

    def add(self, entry: MeasurementEntry) -> bool:
        """Append one observation. Returns ``True`` when it is new evidence.

        A repeated identity is a duplicate: it is counted in diagnostics and
        contributes no energy, so re-running inference on one frame cannot
        inflate confidence.
        """
        identities = entry.batch_identities or (entry.identity,) * entry.values.shape[0]
        keys = [(index, *identity) for index, identity in enumerate(identities)]
        accepted = torch.tensor(
            [
                key not in self._seen
                and not (
                    isinstance(identity[1], Integral)
                    and index in self._latest
                    and self._latest[index][0] == identity[0]
                    and self._latest[index][2] == identity[2]
                    and identity[1] <= self._latest[index][1]
                )
                for index, (key, identity) in enumerate(zip(keys, identities, strict=False))
            ],
            device=entry.values.device,
        )
        if entry.new_mask is not None:
            accepted = accepted & entry.new_mask
        self.last_added = accepted
        self.duplicate_count += int((~accepted).sum())
        if not bool(accepted.any()):
            return False
        self._seen.update(key for key, keep in zip(keys, accepted.tolist(), strict=False) if keep)
        self._latest.update(
            {
                index: identity
                for index, (identity, keep) in enumerate(
                    zip(identities, accepted.tolist(), strict=False)
                )
                if keep and isinstance(identity[1], Integral)
            }
        )
        valid = (
            torch.ones_like(entry.values, dtype=torch.bool)
            if entry.valid_mask is None
            else entry.valid_mask
        )
        self._frames.append(
            dataclasses.replace(entry, valid_mask=valid & accepted.reshape(-1, 1, 1, 1))
        )
        if self.mask_history and len(self._frames) > self.max_history:
            dropped = self._frames.pop(0)
            dropped_ids = dropped.batch_identities or (dropped.identity,) * dropped.values.shape[0]
            self._seen.difference_update(
                (index, *identity)
                for index, identity in enumerate(dropped_ids)
                if isinstance(identity[1], Integral)
            )
            # Integer frame indices use a high watermark. Opaque identities
            # have no ordering, so retain their small ID ledger until reset;
            # never reject a fresh UUID because of lexicographic order.
        return True

    def mask_for(
        self, entry: MeasurementEntry, *, expected_frames: int | None = None
    ) -> torch.Tensor:
        """Per-frame history mask (1.0) for summing measurement energy.

        Bounded-history mode keeps every retained frame at equal weight; the
        mask exists so an unbounded or decayed scheme can be added without
        changing the energy contract. Normalising the posterior is *not* a
        substitute for declaring a history scheme.
        """
        del expected_frames
        return torch.ones((), device=entry.values.device)

    def stack_values(self) -> torch.Tensor:
        """[B, T, C, H, W] view of the retained buffer, oldest first."""
        if not self._frames:
            raise ValueError("empty measurement buffer")
        return torch.stack([frame.values for frame in self._frames], dim=1)

    def state_dict(self) -> dict[str, Any]:
        """Small serialisable summary; tensors are not persisted here."""
        return {
            "max_history": self.max_history,
            "mask_history": self.mask_history,
            "frames": [
                {
                    "camera_id": frame.camera_id,
                    "frame_id": frame.frame_id,
                    "episode_id": frame.episode_id,
                    "timestamp": frame.timestamp,
                    "time_unit": frame.time_unit,
                }
                for frame in self._frames
            ],
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
        }


def estimate_variance(
    values: torch.Tensor,
    *,
    shot_scale: float | None,
    read_scale: float | None,
    floor: float = 1.0e-6,
    sample_variance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool]:
    """Measurement variance in the same numeric units as ``values``.

    When ``sample_variance`` is provided (the sensor reported a per-pixel
    variance, e.g. a synthetic CFA generator) it is used directly. Otherwise a
    Poisson-Gaussian model ``variance = shot * max(x, 0) + read`` is applied in
    the raw units; with neither scale available the variance is a fitted scale
    placeholder and the second return value is ``False`` so callers can report
    it as an estimate rather than a calibrated model.
    """
    if sample_variance is not None:
        return sample_variance.clamp_min(floor), True
    if shot_scale is None and read_scale is None:
        return torch.full_like(values, max(floor, 1.0)), False
    shot = 0.0 if shot_scale is None else float(shot_scale)
    read = 0.0 if read_scale is None else float(read_scale)
    variance = shot * values.clamp_min(0.0) + read
    return variance.clamp_min(floor), True


def measurement_entry_from_frame(
    frame: MeasurementFrame,
    *,
    values: torch.Tensor,
    variance: torch.Tensor,
    sample_xy: torch.Tensor | None,
    camera_id: str,
    frame_id: int,
    episode_id: int | None = None,
    timestamp: float | None = None,
) -> MeasurementEntry:
    """Build a :class:`MeasurementEntry` from a contract frame plus tensors."""

    def as_mask(mask: Any) -> torch.Tensor | None:
        if mask is None:
            return None
        if torch.is_tensor(mask):
            return mask
        return torch.as_tensor(mask)

    return MeasurementEntry(
        values=values,
        valid_mask=as_mask(frame.valid_mask),
        bad_mask=as_mask(frame.bad_mask),
        saturated_mask=as_mask(frame.saturated_mask),
        variance=variance,
        sample_xy=sample_xy,
        camera_id=camera_id,
        frame_id=frame_id,
        episode_id=episode_id,
        timestamp=timestamp,
        time_unit=frame.time_unit,
        metadata=dict(frame.metadata),
    )


@dataclass(slots=True)
class RevisionState:
    """Per-location revision bookkeeping used by :class:`RevisionController`."""

    residual_ema: torch.Tensor | None = None  # [B,1,h,w] running normalised residual
    contradiction: torch.Tensor | None = None  # [B,K,h,w] bool, this cycle
    contradiction_prev: torch.Tensor | None = None  # [B,K,h,w] bool, before propagation
    propagated: torch.Tensor | None = None  # [B,K,h,w] bool, after propagation
    priority: torch.Tensor | None = None  # [B,K,h,w] float propagation priority
    last_reset: torch.Tensor | None = None  # [B,K,h,w] bool, reset this cycle

    def clone(self) -> RevisionState:
        def clone_tensor(value):
            return None if value is None else value.clone()

        return RevisionState(
            residual_ema=clone_tensor(self.residual_ema),
            contradiction=clone_tensor(self.contradiction),
            contradiction_prev=clone_tensor(self.contradiction_prev),
            propagated=clone_tensor(self.propagated),
            priority=clone_tensor(self.priority),
            last_reset=clone_tensor(self.last_reset),
        )

    def detach(self) -> RevisionState:
        return RevisionState(
            residual_ema=None if self.residual_ema is None else self.residual_ema.detach(),
            contradiction=None if self.contradiction is None else self.contradiction,
            contradiction_prev=None if self.contradiction_prev is None else self.contradiction_prev,
            propagated=None if self.propagated is None else self.propagated,
            priority=None if self.priority is None else self.priority.detach(),
            last_reset=None if self.last_reset is None else self.last_reset,
        )


@dataclass(slots=True)
class HypothesisState:
    """The measurement-constrained hypothesis field.

    ``h`` is ``[B, V, K, D, h, w]``: ``V`` is the number of views (cameras),
    ``K`` the number of candidate explanations for the *same* latent structure
    (not extra observations), ``D`` the per-cell width. ``measurement_energy``
    is ``[B, V, K]`` and holds the mean per-observation energy over the frozen
    measurement set, so it is additive over the unique evidence only.
    """

    h: torch.Tensor
    measurement_energy: torch.Tensor
    weights: torch.Tensor  # [B, V, K] softmax over K
    prior_energy: torch.Tensor  # [B, V, K]
    relation_energy: torch.Tensor  # [B, V, K]
    total_energy: torch.Tensor  # [B, V, K]
    logits: torch.Tensor  # [B, V, K] raw scores before softmax (score = -E/T)
    geometry: torch.Tensor  # [B, V, K, geometry_dim] declared latent geometry
    age: torch.Tensor  # [B, V, K]
    times_reset: torch.Tensor  # [B, V, K]
    support: torch.Tensor | None = None  # [B, V, K, h, w] measurement support
    dependency_edge: torch.Tensor | None = None  # [B, V, h, w, 4, K, K] u depends on v
    dependency_valid: torch.Tensor | None = None  # [B, V, h, w, 4] boundary validity
    revision: RevisionState = field(default_factory=RevisionState)
    diagnostics: dict = field(default_factory=dict)
    step_index: int = 0
    revision_energy: torch.Tensor | None = None  # [B,V,K], once per unique frame

    @property
    def K(self) -> int:
        return self.h.shape[2]

    @property
    def V(self) -> int:
        return self.h.shape[1]

    @property
    def batch_size(self) -> int:
        return self.h.shape[0]

    def select_batch(self, index: int) -> HypothesisState:
        def take(value, dim=0):
            if value is None:
                return None
            if torch.is_tensor(value) and value.shape[0] > index:
                return value[index : index + 1]
            return value

        return HypothesisState(
            h=self.h[index : index + 1],
            measurement_energy=take(self.measurement_energy),
            weights=take(self.weights),
            prior_energy=take(self.prior_energy),
            relation_energy=take(self.relation_energy),
            total_energy=take(self.total_energy),
            logits=take(self.logits),
            geometry=take(self.geometry),
            age=take(self.age),
            times_reset=take(self.times_reset),
            revision_energy=take(self.revision_energy),
            support=take(self.support),
            dependency_edge=take(self.dependency_edge),
            dependency_valid=take(self.dependency_valid),
            revision=self.revision,
            diagnostics=dict(self.diagnostics),
            step_index=self.step_index,
        )

    def masked_update(self, new_h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Merge ``new_h`` into ``h`` only where ``mask`` selects (bool, broadcastable)."""
        return torch.where(mask, new_h, self.h)

    def detach(self) -> HypothesisState:
        def det(value):
            return None if value is None else value.detach()

        return dataclasses.replace(
            self,
            h=det(self.h),
            measurement_energy=det(self.measurement_energy),
            weights=det(self.weights),
            prior_energy=det(self.prior_energy),
            relation_energy=det(self.relation_energy),
            total_energy=det(self.total_energy),
            logits=det(self.logits),
            geometry=det(self.geometry),
            age=det(self.age),
            times_reset=det(self.times_reset),
            revision_energy=det(self.revision_energy),
            support=det(self.support),
            dependency_edge=det(self.dependency_edge),
            dependency_valid=det(self.dependency_valid),
            revision=self.revision.detach(),
        )

    def state_bytes(self) -> int:
        total = 0
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            if torch.is_tensor(value):
                total += value.element_size() * value.numel()
        revision = self.revision
        for item in dataclasses.fields(revision):
            value = getattr(revision, item.name)
            if torch.is_tensor(value):
                total += value.element_size() * value.numel()
        return total

    def summary(self) -> dict[str, Any]:
        with torch.no_grad():
            weights = self.weights.detach()
            return {
                "step_index": self.step_index,
                "K": self.K,
                "V": self.V,
                "weights": weights.mean(dim=0).tolist(),
                "max_weight_mean": float(weights.max(dim=-1).values.mean().item()),
                "measurement_energy_mean": float(self.measurement_energy.detach().mean().item()),
                "age_mean": float(self.age.detach().mean().item()),
                "resets": float(self.times_reset.detach().sum().item()),
                "state_bytes": self.state_bytes(),
            }

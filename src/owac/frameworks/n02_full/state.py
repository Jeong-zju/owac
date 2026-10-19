"""Versioned ledger, identity and state containers for the N02 full (v1.1) line.

The semantics fixed by ``docs/prompts/n02_v1_1/00_SHARED_CONTRACT.md`` sections 3
and 4 are implemented here and nowhere else:

* A measurement is identified by ``(episode_id, camera_id, source_frame_id,
  acquisition_id)``. Re-registering the same acquisition adds no statistical
  evidence; the same source frame re-acquired with a new acquisition ID is a
  distinct acquisition and does count once more.
* A calibration change replaces the derived factor version of an existing
  measurement; it never copies the observation or adds evidence.
* Reset is per episode: the other rows keep their values bit for bit.
* An :class:`ObjectiveKey` fixes the measurement set, model version, structure
  version, calibration versions and initial prior version; it is order
  independent over the calibration version tuple.
* Stage verdicts only use ``pass`` / ``fail`` / ``not_run`` / ``incomplete``.

No third-party import happens at module scope: tensor-shaped payloads are typed
``Any`` and the batch-mask helpers duck-type torch/numpy/list payloads, so this
module loads in the light ``owac`` import path. Reset of a row payload that
cannot be zeroed without its own API raises instead of silently keeping stale
values.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame

__all__ = [
    "EntityState",
    "FeatureLevel",
    "InferenceInput",
    "MeasurementKey",
    "MeasurementLedger",
    "MeasurementRecord",
    "ObjectiveKey",
    "ObservationBundle",
    "STAGE_VERDICTS",
    "SequenceState",
    "StageResult",
    "TrainingTargets",
    "WorldHypothesis",
    "measurement_set_hash",
    "values_equal",
]

#: Legal values of every :class:`StageResult` verdict field.
STAGE_VERDICTS = ("pass", "fail", "not_run", "incomplete")

#: Declared kinds of a world score. A normalized density is a model posterior; an
#: empirical energy has no calibrated posterior claim; an importance weight is a
#: sampling correction. They are never interchangeable.
SCORE_KINDS = ("normalized_density", "empirical_energy", "importance_weight")


def canonical_json(payload: Any) -> str:
    """Canonical JSON text (sorted keys, tight separators) used for every hash."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_tuple(value: Any, name: str) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if value is None:
        return ()
    raise TypeError(f"{name} must be a tuple or list, got {type(value).__name__}")


def _as_str_tuple(value: Any, name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_tuple(value, name))


def _copy_payload(value: Any) -> Any:
    """Copy tensor-like payloads without importing their library."""
    if value is None:
        return None
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copier = getattr(value, "copy", None)
    if callable(copier):
        return copier()
    return copy.deepcopy(value)


def _zero_payload(value: Any) -> Any:
    """Zeroed copy of one reset row; raises for payloads we cannot zero safely."""
    if value is None:
        return None
    zero_ = getattr(value, "zero_", None)
    if callable(zero_):  # torch.Tensor
        return _copy_payload(value).zero_()
    fill = getattr(value, "fill", None)
    if callable(fill):  # numpy.ndarray
        result = _copy_payload(value)
        result.fill(0)
        return result
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0
    if isinstance(value, list):
        return [_zero_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_zero_payload(item) for item in value)
    raise TypeError(f"cannot zero a reset row payload of type {type(value).__name__}")


def values_equal(left: Any, right: Any) -> bool:
    """Element-wise equality that works for torch tensors, numpy arrays, tuples."""
    if left is right:
        return True
    equals = getattr(left, "equals", None)
    if callable(equals):  # torch.Tensor
        try:
            return bool(equals(right))
        except Exception:  # noqa: BLE001 - a failed comparison is "not equal"
            return False
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        if len(left) != len(right):
            return False
        return all(values_equal(a, b) for a, b in zip(left, right, strict=True))
    if hasattr(left, "shape") and hasattr(right, "shape"):
        try:
            return bool((left == right).all())
        except Exception:  # noqa: BLE001 - shape mismatch or dtype issue is "not equal"
            return False
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True, slots=True)
class MeasurementKey:
    """Persistent identity of one acquisition of one source frame of one camera.

    The batch index is never part of this identity. A retransmission of the same
    frame keeps its ``acquisition_id`` and is one measurement; a truly new
    acquisition of the same source frame (for example after a re-exposure)
    carries a new ``acquisition_id`` and is a separate measurement.
    """

    episode_id: str
    camera_id: str
    source_frame_id: str
    acquisition_id: str

    def __post_init__(self) -> None:
        for name in ("episode_id", "camera_id", "source_frame_id", "acquisition_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                object.__setattr__(self, name, str(value))
        if not self.acquisition_id:
            raise ValueError("acquisition_id must be a non-empty string")

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "source_frame_id": self.source_frame_id,
            "acquisition_id": self.acquisition_id,
        }

    def key_hash(self) -> str:
        """sha256 of the canonical identity JSON."""
        return _sha256_text(canonical_json(self.as_dict()))


def measurement_set_hash(keys: Iterable[MeasurementKey]) -> str:
    """Hash a measurement *set*: identical for any iteration order.

    It hashes the sorted unique per-key hashes, so passing the same set twice in
    different orders gives one value and passing a duplicate key does not change
    it (a set has no multiplicity).
    """
    digests = sorted({key.key_hash() for key in keys})
    return _sha256_text(canonical_json(digests))


@dataclass(slots=True)
class MeasurementRecord:
    """One registered measurement: raw reference plus its sampling condition."""

    key: MeasurementKey
    raw_ref: str = ""
    calibration_version: str = ""
    masks: dict = field(default_factory=dict)
    noise_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key, MeasurementKey):
            raise TypeError(f"key must be a MeasurementKey, got {type(self.key).__name__}")
        self.raw_ref = str(self.raw_ref)
        self.calibration_version = str(self.calibration_version)
        if not isinstance(self.masks, dict):
            raise TypeError(f"masks must be a dict, got {type(self.masks).__name__}")
        if not isinstance(self.noise_metadata, dict):
            raise TypeError(
                f"noise_metadata must be a dict, got {type(self.noise_metadata).__name__}"
            )
        self.masks = dict(self.masks)
        self.noise_metadata = copy.deepcopy(self.noise_metadata)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.as_dict(),
            "raw_ref": self.raw_ref,
            "calibration_version": self.calibration_version,
            "masks": copy.deepcopy(self.masks),
            "noise_metadata": copy.deepcopy(self.noise_metadata),
        }


class MeasurementLedger:
    """The unique measurement ledger: one entry per acquisition ID.

    ``state_version`` increments by one for every successful mutation
    (``register`` of a new acquisition, ``replace_calibration`` that changes a
    version, and ``reset`` that removes at least one record). A rejected
    duplicate and a reset with no matching episode change nothing.
    """

    __slots__ = ("_records", "_state_version")

    def __init__(
        self,
        records: Iterable[MeasurementRecord] | None = None,
        state_version: int = 0,
    ) -> None:
        self._records: dict[str, MeasurementRecord] = {}
        self._state_version = int(state_version)
        for record in records or ():
            if not isinstance(record, MeasurementRecord):
                raise TypeError(
                    f"records must contain MeasurementRecord, got {type(record).__name__}"
                )
            self._records[record.key.acquisition_id] = record

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def evidence_count(self) -> int:
        """Number of independent measurements currently held."""
        return len(self._records)

    def register(self, record: MeasurementRecord) -> bool:
        """Register one acquisition; ``False`` if that acquisition already exists."""
        if not isinstance(record, MeasurementRecord):
            raise TypeError(f"record must be a MeasurementRecord, got {type(record).__name__}")
        acquisition_id = record.key.acquisition_id
        if acquisition_id in self._records:
            return False
        self._records[acquisition_id] = record
        self._state_version += 1
        return True

    def replace_calibration(self, key: MeasurementKey, calibration_version: str) -> bool:
        """Replace only the derived calibration version of an existing measurement.

        Raises :class:`KeyError` when the key is not registered. Returns ``True``
        when the version actually changed; the evidence count never changes and a
        no-op replacement does not bump ``state_version``.
        """
        record = self._lookup(key)
        version = str(calibration_version)
        if record.calibration_version == version:
            return False
        self._records[key.acquisition_id] = MeasurementRecord(
            key=record.key,
            raw_ref=record.raw_ref,
            calibration_version=version,
            masks=record.masks,
            noise_metadata=record.noise_metadata,
        )
        self._state_version += 1
        return True

    def _lookup(self, key: MeasurementKey) -> MeasurementRecord:
        if not isinstance(key, MeasurementKey):
            raise TypeError(f"key must be a MeasurementKey, got {type(key).__name__}")
        record = self._records.get(key.acquisition_id)
        if record is None or record.key != key:
            raise KeyError(key)
        return record

    def get(self, key: MeasurementKey, default: Any = None) -> MeasurementRecord | None:
        """The record for ``key``, or ``default`` when it is not registered."""
        if not isinstance(key, MeasurementKey):
            raise TypeError(f"key must be a MeasurementKey, got {type(key).__name__}")
        record = self._records.get(key.acquisition_id)
        if record is None or record.key != key:
            return default
        return record

    def has(self, key: MeasurementKey) -> bool:
        return self.get(key) is not None

    def keys(self) -> tuple[MeasurementKey, ...]:
        """Registered keys in insertion order."""
        return tuple(record.key for record in self._records.values())

    def records(self) -> tuple[MeasurementRecord, ...]:
        """Registered records in insertion order."""
        return tuple(self._records.values())

    def reset(self, episode_ids: str | Iterable[str]) -> int:
        """Drop every record of the given episode(s); return how many were removed.

        Other episodes keep their records bit for bit. ``state_version``
        increments only when at least one record matched.
        """
        if isinstance(episode_ids, str):
            targets = {episode_ids}
        else:
            targets = {str(item) for item in episode_ids}
        removed = [
            acquisition_id
            for acquisition_id, record in self._records.items()
            if record.key.episode_id in targets
        ]
        if not removed:
            return 0
        for acquisition_id in removed:
            del self._records[acquisition_id]
        self._state_version += 1
        return len(removed)

    def __len__(self) -> int:
        return len(self._records)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeasurementLedger):
            return NotImplemented
        return self._state_version == other._state_version and self._records == other._records

    def __repr__(self) -> str:
        return (
            f"MeasurementLedger(evidence_count={self.evidence_count}, "
            f"state_version={self._state_version})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_version": self._state_version,
            "records": [record.as_dict() for record in self._records.values()],
        }

    def to_json(self) -> str:
        """Serialized ledger; ``from_json(to_json())`` compares equal to self."""
        return json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, payload: str | bytes | dict) -> MeasurementLedger:
        """Rebuild a ledger from :meth:`to_json` output (text or decoded dict)."""
        if isinstance(payload, (str, bytes)):
            decoded = json.loads(payload)
        else:
            decoded = payload
        if not isinstance(decoded, dict):
            raise TypeError(f"payload must be a JSON object, got {type(decoded).__name__}")
        entries = decoded.get("records", [])
        if not isinstance(entries, list):
            raise TypeError("payload['records'] must be a list")
        records = []
        for entry in entries:
            records.append(
                MeasurementRecord(
                    key=MeasurementKey(**entry["key"]),
                    raw_ref=entry.get("raw_ref", ""),
                    calibration_version=entry.get("calibration_version", ""),
                    masks=entry.get("masks", {}),
                    noise_metadata=entry.get("noise_metadata", {}),
                )
            )
        return cls(records=records, state_version=int(decoded.get("state_version", 0)))


@dataclass(frozen=True, slots=True)
class ObjectiveKey:
    """Identity of one frozen inference problem.

    Two runs share an objective only when the measurement set, model version,
    structure version, calibration versions and initial prior version all match.
    ``calibration_versions`` is normalized to a sorted tuple so argument order
    cannot change the identity.
    """

    measurement_set_hash: str
    model_version: str
    structure_version: str
    calibration_versions: tuple[str, ...] = ()
    initial_prior_version: str = ""

    def __post_init__(self) -> None:
        for name in ("measurement_set_hash", "model_version", "structure_version"):
            value = getattr(self, name)
            if not isinstance(value, str):
                object.__setattr__(self, name, str(value))
        versions = self.calibration_versions
        if versions is None:
            versions = ()
        if isinstance(versions, str):
            versions = (versions,)
        if not isinstance(versions, (list, tuple, set, frozenset)):
            raise TypeError(f"calibration_versions must be strings, got {type(versions).__name__}")
        normalized = tuple(sorted(str(item) for item in versions))
        object.__setattr__(self, "calibration_versions", normalized)
        if not isinstance(self.initial_prior_version, str):
            object.__setattr__(self, "initial_prior_version", str(self.initial_prior_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "measurement_set_hash": self.measurement_set_hash,
            "model_version": self.model_version,
            "structure_version": self.structure_version,
            "calibration_versions": list(self.calibration_versions),
            "initial_prior_version": self.initial_prior_version,
        }

    def key_hash(self) -> str:
        return _sha256_text(canonical_json(self.as_dict()))


@dataclass(slots=True)
class ObservationBundle:
    """All cameras available at one decision time; one world candidate is chosen.

    ``frames`` maps camera ID to that camera's :class:`MeasurementFrame`. Cameras
    may differ in resolution. ``new_time_mask`` marks which batch rows advanced
    to a new time step; only those rows execute the prediction step once.
    """

    frames: dict[str, MeasurementFrame]
    context: EmbodimentContext
    episode_ids: tuple = ()
    anchor_ids: tuple = ()
    new_time_mask: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.frames, dict):
            raise TypeError(f"frames must be a dict, got {type(self.frames).__name__}")
        for camera_id in self.frames:
            if not isinstance(camera_id, str):
                raise TypeError("frames keys must be camera ID strings")
        self.episode_ids = _as_tuple(self.episode_ids, "episode_ids")
        self.anchor_ids = _as_tuple(self.anchor_ids, "anchor_ids")

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self.frames)


@dataclass(slots=True)
class InferenceInput:
    """Online input: measurements plus declared priors, never hidden truth."""

    bundle: ObservationBundle
    declared_prior: Any = None


@dataclass(slots=True)
class TrainingTargets:
    """Supervision for one training call; never visible to inference."""

    actions: Any = None
    action_mask: Any = None
    optional_geometry: Any = None
    optional_relations: Any = None


@dataclass(slots=True)
class FeatureLevel:
    """One multi-scale feature level with its sampling coordinates and support."""

    values: Any
    valid: Any
    sensor_coordinates: Any
    support_ids: tuple = ()
    level_id: str = ""

    def __post_init__(self) -> None:
        self.support_ids = _as_tuple(self.support_ids, "support_ids")
        if not isinstance(self.level_id, str):
            self.level_id = str(self.level_id)


@dataclass(slots=True)
class EntityState:
    """Recurrent entity slots. IDs are not permanently bound to slot indices."""

    entity_ids: tuple = ()
    slot_hidden: Any = None
    existence: Any = None
    geometry: Any = None
    geometry_frame: str = "view_2d"
    version: int = 0

    def __post_init__(self) -> None:
        self.entity_ids = _as_tuple(self.entity_ids, "entity_ids")
        self.geometry_frame = str(self.geometry_frame)


@dataclass(slots=True)
class WorldHypothesis:
    """One self-consistent world candidate: entities, associations, relations.

    ``score`` is only interpretable together with ``score_kind``; a
    normalized density supports posterior claims, an empirical energy does not,
    and an importance weight is a sampling correction.
    """

    world_id: str
    entities: tuple = ()
    associations: Any = None
    relations: Any = None
    nuisance: Any = None
    score: float = 0.0
    score_kind: str = "normalized_density"
    ancestor_id: str | None = None
    version: int = 0

    def __post_init__(self) -> None:
        self.world_id = str(self.world_id)
        self.entities = _as_tuple(self.entities, "entities")
        if self.score_kind not in SCORE_KINDS:
            raise ValueError(
                f"score_kind must be one of {', '.join(SCORE_KINDS)}, got {self.score_kind!r}"
            )
        if self.ancestor_id is not None:
            self.ancestor_id = str(self.ancestor_id)


@dataclass(slots=True, eq=False)
class SequenceState:
    """Per-episode recurrent state; ``reset_batch`` clears selected rows only.

    Row ``i`` corresponds to batch element ``i``. Reset zeroes the payload rows
    of the selected batch elements and clears their episode IDs; unselected rows
    keep their values bit for bit. Version increments by one per reset that
    selects at least one row, and does not change for an all-``False`` mask.
    """

    episode_ids: tuple = ()
    slot_hidden: Any = None
    existence: Any = None
    geometry: Any = None
    version: int = 0

    def __post_init__(self) -> None:
        self.episode_ids = _as_tuple(self.episode_ids, "episode_ids")

    @property
    def batch_size(self) -> int | None:
        """Batch size inferred from episode IDs or the first shaped payload."""
        if self.episode_ids:
            return len(self.episode_ids)
        for value in (self.slot_hidden, self.existence, self.geometry):
            shape = getattr(value, "shape", None)
            if shape is not None:
                try:
                    return int(shape[0])
                except (IndexError, TypeError, ValueError):
                    continue
            if isinstance(value, (list, tuple)):
                return len(value)
        return None

    def reset_batch(self, mask: Any) -> SequenceState:
        """Return a new state with the rows selected by ``mask`` reset."""
        flags = _mask_to_bools(mask)
        batch_size = self.batch_size
        if batch_size is not None and len(flags) != batch_size:
            raise ValueError(
                f"reset mask length {len(flags)} does not match batch size {batch_size}"
            )
        selected = [index for index, keep in enumerate(flags) if keep]
        if not selected:
            return SequenceState(
                episode_ids=self.episode_ids,
                slot_hidden=_copy_payload(self.slot_hidden),
                existence=_copy_payload(self.existence),
                geometry=_copy_payload(self.geometry),
                version=self.version,
            )
        episode_ids = list(self.episode_ids)
        for index in selected:
            if index < len(episode_ids):
                episode_ids[index] = ""
        return SequenceState(
            episode_ids=tuple(episode_ids),
            slot_hidden=_reset_rows(self.slot_hidden, selected),
            existence=_reset_rows(self.existence, selected),
            geometry=_reset_rows(self.geometry, selected),
            version=self.version + 1,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceState):
            return NotImplemented
        return self.version == other.version and values_equal(
            (self.episode_ids, self.slot_hidden, self.existence, self.geometry),
            (other.episode_ids, other.slot_hidden, other.existence, other.geometry),
        )

    def __repr__(self) -> str:
        return f"SequenceState(batch_size={self.batch_size}, version={self.version})"


def _reset_rows(value: Any, selected: list[int]) -> Any:
    if value is None:
        return None
    result = _copy_payload(value)
    for index in selected:
        try:
            result[index] = _zero_payload(value[index])
        except IndexError as exc:
            raise ValueError(f"reset row {index} is outside the payload batch") from exc
    return result


def _mask_to_bools(mask: Any) -> list[bool]:
    """Normalize a bool sequence / torch bool tensor into a list of Python bools."""
    if mask is None:
        raise TypeError("reset mask must be a bool sequence or bool tensor")
    tolist = getattr(mask, "tolist", None)
    if callable(tolist):
        values = tolist()
    elif isinstance(mask, (list, tuple)):
        values = list(mask)
    else:
        raise TypeError(f"unsupported reset mask type {type(mask).__name__}")
    if not isinstance(values, (list, tuple)):
        raise TypeError("reset mask must be one-dimensional")
    return [bool(item) for item in values]


@dataclass(slots=True)
class StageResult:
    """Per-stage verdict; the four axes are never collapsed into one.

    ``pass`` means the corresponding evidence was produced and checked. Anything
    not actually verified stays ``not_run``.
    """

    implementation: str
    numerical: str
    behavior: str
    closed_loop: str
    evidence_paths: tuple = ()
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("implementation", "numerical", "behavior", "closed_loop"):
            value = getattr(self, name)
            if value not in STAGE_VERDICTS:
                raise ValueError(
                    f"{name} must be one of {', '.join(STAGE_VERDICTS)}, got {value!r}"
                )
        self.evidence_paths = _as_str_tuple(self.evidence_paths, "evidence_paths")
        if not isinstance(self.notes, dict):
            raise TypeError(f"notes must be a dict, got {type(self.notes).__name__}")
        self.notes = copy.deepcopy(self.notes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "numerical": self.numerical,
            "behavior": self.behavior,
            "closed_loop": self.closed_loop,
            "evidence_paths": list(self.evidence_paths),
            "notes": copy.deepcopy(self.notes),
        }

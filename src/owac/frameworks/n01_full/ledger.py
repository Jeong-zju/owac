"""The measurement ledger: raw samples indexed by ``SourceKey``.

Design of ``docs/prompts/n01_v1_0/P1_ANALYTIC_REFERENCE.md`` section 1: raw data
is stored per frame as one tensor plus the frame's masks and sampling offsets, and
``SourceKey`` addresses a sample inside that tensor.  A 256x256x4 frame therefore
costs one tensor and one dict entry, not 262 144 Python objects; ``get_sources``
materialises :class:`~owac.frameworks.n01_full.contracts.MeasurementRecord`
instances only for the identities a caller actually asks about, and the bulk
:meth:`MeasurementLedger.values_for` path never creates them at all.

Two invariants the contract fixes are enforced here, not documented away:

* duplicates are decided by ``SourceKey.identity()``.  Re-reading one packet does
  not create a second measurement and does not advance any count or budget, while
  a new exposure stays a new sample even when every integer is identical (the
  frame identity differs).  A frame whose ``new_measurement`` flag is false and
  that was never registered is *skipped*, never invented;
* raw values are immutable.  Recalibration and the content replacement used by
  ``fixed_decision_replay`` are separate derived objects
  (:class:`ContentReplacement`), so a retracted sample never rewrites its record
  and "was retracted" is never confused with "the sensor changed".

``MeasurementLedger.without`` returns a derived *view* for the counterfactual
replay: the exposure still happened (it keeps its frame identity and arrival
order) but the listed samples are absent from every query.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full.contracts import VALIDITY_KINDS, MeasurementRecord, SourceKey

__all__ = [
    "VALID",
    "MISSING",
    "BAD",
    "SATURATED",
    "VALIDITY_CODES",
    "FrameIdentity",
    "LedgerFrame",
    "ObserveResult",
    "ContentReplacement",
    "MeasurementLedger",
    "SourceNotFound",
]

VALID = 0
MISSING = 1
BAD = 2
SATURATED = 3

#: int8 storage codes; index order matches the four-state vocabulary.
VALIDITY_CODES: tuple[str, ...] = ("valid", "missing", "bad", "saturated")


class SourceNotFound(KeyError):
    """Raised when an identity is not in the ledger (or was removed by a view)."""


class FrameIdentity(NamedTuple):
    """Identity of one exposure: dataset, episode, camera and frame id."""

    dataset_id: str
    episode_id: str
    camera_id: str
    frame_id: int | str


@dataclass(frozen=True)
class LedgerFrame:
    """One exposure stored as a tensor plus its masks and sampling coordinates."""

    identity: FrameIdentity
    values: torch.Tensor  # [C, H, W], raw unit
    validity: torch.Tensor  # [C, H, W] int8 codes
    timestamp: float
    time_unit: str
    value_units: str
    calibration_version: str
    arrival_index: int
    checksum: str
    noise_block_id: str | None = None
    sample_offset: torch.Tensor | None = None  # [C, H, W, 2] real sampling (x, y) order
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self.values.shape)

    def sample_coordinates(self, channel: int, y: int, x: int) -> tuple[float, float]:
        """Real sampling coordinates as ``(x, y)``: declared offsets, else indices."""
        if self.sample_offset is not None:
            offset = self.sample_offset[channel, y, x]
            return float(offset[0]), float(offset[1])
        return float(x), float(y)

    def validity_of(self, channel: int, y: int, x: int) -> str:
        return VALIDITY_CODES[int(self.validity[channel, y, x])]


def _to_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(np.asarray(value))
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a tensor or array-like") from exc


def _scalar(value: Any, index: int, name: str) -> Any:
    """One identity/flag scalar, normalised to a Python value.

    Loader tensors are indexed with ``value[:, step]`` and yield **0-dim tensors**
    rather than integers.  Those must not become dictionary keys: two distinct
    0-dim tensors with the same value compare equal but hash by identity, so a
    re-sent frame would be registered again instead of recognised as a duplicate.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.reshape(-1)[index].item()
    if isinstance(value, str) or isinstance(value, (int, float, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:  # pragma: no cover - array-like without a scalar item
            pass
    try:
        return value[index]
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a tensor, sequence or scalar") from exc


def _content_checksum(values: torch.Tensor, validity: torch.Tensor) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values.detach().to("cpu").numpy())
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    codes = np.ascontiguousarray(validity.detach().to("cpu").numpy().astype(np.int8))
    digest.update(codes.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ObserveResult:
    """Outcome of one or more frame arrivals in a single :meth:`observe` call."""

    registered_frames: tuple[FrameIdentity, ...] = ()
    duplicate_frames: tuple[FrameIdentity, ...] = ()
    skipped_frames: tuple[FrameIdentity, ...] = ()
    new_samples: int = 0
    duplicate_samples: int = 0
    skipped_samples: int = 0

    @property
    def advanced(self) -> bool:
        """Whether this arrival changed the ledger (a duplicate or skip did not)."""
        return bool(self.registered_frames)


@dataclass(frozen=True)
class ContentReplacement:
    """A versioned derived overlay used by ``fixed_decision_replay``.

    ``values`` replaces the *effective* content a content path reads for the
    listed identities (``None`` means the sample carries no measurement).  The
    ledger's raw records are untouched: the overlay is a declared, named
    derivation, exactly like a recalibration.
    """

    replacement_id: str
    values: Mapping[SourceKey, float | None]
    version: str = "content-replacement-v1"

    def __post_init__(self) -> None:
        if not self.replacement_id:
            raise ValueError("ContentReplacement.replacement_id must be declared")
        if not isinstance(self.values, Mapping):
            raise TypeError("ContentReplacement.values must be a mapping of SourceKey")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        for key in self.values:
            if not isinstance(key, SourceKey):
                raise TypeError("ContentReplacement keys must be SourceKey instances")

    def value_of(self, key: SourceKey) -> float | None:
        return self.values.get(key)


class MeasurementLedger:
    """Raw measurements by identity, with duplicate-safe arrival semantics."""

    def __init__(
        self,
        dataset_id: str,
        *,
        value_units: str = "unknown",
        time_unit: str = "frame",
        calibration_version: str = "unknown",
        removed: Iterable[SourceKey] = (),
    ) -> None:
        if not dataset_id:
            raise ValueError("MeasurementLedger.dataset_id must name the dataset")
        self.dataset_id = dataset_id
        self.default_value_units = value_units
        self.default_time_unit = time_unit
        self.default_calibration_version = calibration_version
        self._frames: dict[FrameIdentity, LedgerFrame] = {}
        self._arrival: list[FrameIdentity] = []
        self._counters = {"arrivals": 0, "duplicate_arrivals": 0, "skipped_arrivals": 0}
        self._removed: frozenset[SourceKey] = frozenset(removed)
        self._overlay: ContentReplacement | None = None

    # -- observation --------------------------------------------------------
    def observe(self, frame: MeasurementFrame) -> ObserveResult:
        """Register one camera's frame arrival (one or more batch members)."""
        values = _to_tensor(frame.values, name="values")
        if values.ndim == 3:
            values = values[None]
        if values.ndim != 4:
            raise ValueError(
                "MeasurementFrame.values must be [C,H,W] or [B,C,H,W]; a partial grid is "
                "not a measurement frame"
            )
        batch = int(values.shape[0])
        valid = _mask(frame.valid_mask, values.shape, batch, "valid_mask")
        bad = _mask(frame.bad_mask, values.shape, batch, "bad_mask")
        saturated = _mask(frame.saturated_mask, values.shape, batch, "saturated_mask")
        offsets = None
        if frame.sample_xy is not None:
            offsets = _to_tensor(frame.sample_xy, name="sample_xy")
        if offsets is not None and tuple(offsets.shape) != (*values.shape, 2):
            raise ValueError(
                f"sample_xy must be [B,C,H,W,2] to keep real sampling coordinates, got "
                f"{tuple(offsets.shape)}"
            )
        if not frame.camera_id:
            raise ValueError(
                "MeasurementFrame.camera_id must name the camera; an empty camera id is "
                "not an identity"
            )

        registered: list[FrameIdentity] = []
        duplicates: list[FrameIdentity] = []
        skipped: list[FrameIdentity] = []
        samples = int(values[0].numel())
        for index in range(batch):
            episode_id = _scalar(frame.episode_ids, index, "episode_ids")
            frame_id = _scalar(frame.frame_ids, index, "frame_ids")
            new_measurement = _scalar(frame.new_measurement, index, "new_measurement")
            if episode_id is None or frame_id is None:
                raise ValueError(
                    "every arrival needs an episode_id and a frame_id; batch position is "
                    "not an identity"
                )
            identity = FrameIdentity(self.dataset_id, str(episode_id), frame.camera_id, frame_id)
            self._counters["arrivals"] += 1
            if identity in self._frames:
                self._counters["duplicate_arrivals"] += 1
                duplicates.append(identity)
                continue
            if new_measurement is not None and not bool(new_measurement):
                self._counters["skipped_arrivals"] += 1
                skipped.append(identity)
                continue
            payload = values[index].detach().clone()
            codes = _validity_codes(valid, bad, saturated, index, values.shape[1:])
            offset = None if offsets is None else offsets[index].detach().clone()
            self._frames[identity] = LedgerFrame(
                identity=identity,
                values=payload,
                validity=codes,
                timestamp=float(_scalar(frame.timestamp, index, "timestamp") or 0.0),
                time_unit=frame.time_unit or self.default_time_unit,
                value_units=str(frame.metadata.get("value_units", self.default_value_units)),
                calibration_version=str(
                    frame.metadata.get("calibration_version", self.default_calibration_version)
                ),
                noise_block_id=frame.metadata.get("noise_block_id"),
                sample_offset=offset,
                arrival_index=len(self._arrival),
                checksum=_content_checksum(payload, codes),
                metadata=dict(frame.metadata),
            )
            self._arrival.append(identity)
            registered.append(identity)
        return ObserveResult(
            registered_frames=tuple(registered),
            duplicate_frames=tuple(duplicates),
            skipped_frames=tuple(skipped),
            new_samples=len(registered) * samples,
            duplicate_samples=len(duplicates) * samples,
            skipped_samples=len(skipped) * samples,
        )

    # -- queries ------------------------------------------------------------
    def get_frame(self, episode_id: str, camera_id: str, frame_id: int | str) -> LedgerFrame | None:
        frame = self._frames.get(FrameIdentity(self.dataset_id, episode_id, camera_id, frame_id))
        return None if frame is None else self._public_frame(frame)

    def has_source(self, key: SourceKey) -> bool:
        if key in self._removed:
            return False
        if (
            self._overlay is not None
            and key in self._overlay.values
            and self._overlay.values[key] is None
        ):
            return False
        frame = self._frame_for(key)
        if frame is None:
            return False
        channel, y, x = int(key.channel_or_phase), key.sensor_y, key.sensor_x
        return 0 <= channel < frame.shape[0] and 0 <= y < frame.shape[1] and 0 <= x < frame.shape[2]

    def get_source(self, key: SourceKey) -> MeasurementRecord:
        """Materialise one :class:`MeasurementRecord` on demand."""
        if not isinstance(key, SourceKey):
            raise TypeError("MeasurementLedger.get_source expects a SourceKey")
        if key in self._removed:
            raise SourceNotFound(
                f"source {key.identity()} is absent in this ledger view; the counterfactual "
                "replay removes it from every query"
            )
        frame = self._frame_for(key)
        if frame is None:
            raise SourceNotFound(f"source {key.identity()} was never observed")
        channel, y, x = int(key.channel_or_phase), key.sensor_y, key.sensor_x
        if not (
            0 <= channel < frame.shape[0] and 0 <= y < frame.shape[1] and 0 <= x < frame.shape[2]
        ):
            raise SourceNotFound(f"source {key.identity()} is outside the stored frame grid")
        validity_kind = frame.validity_of(channel, y, x)
        raw_value: float | int | None
        if validity_kind == "missing":
            raw_value = None
        else:
            raw_value = frame.values[channel, y, x].item()
        value_units = frame.value_units
        if self._overlay is not None and key in self._overlay.values:
            # A derived, versioned content view: the raw record on disk is never
            # rewritten (see ContentReplacement).
            replaced = self._overlay.value_of(key)
            if replaced is None:
                raise SourceNotFound(
                    f"{key.identity()} carries no measurement in content view "
                    f"{self._overlay.replacement_id!r}; a replaced-away sample cannot be evidence"
                )
            raw_value = float(replaced)
            value_units = f"{frame.value_units}+{self._overlay.version}"
        return MeasurementRecord(
            key=key,
            raw_value=raw_value,
            value_units=value_units,
            timestamp=frame.timestamp,
            calibration_version=frame.calibration_version,
            time_unit=frame.time_unit,
            validity_kind=validity_kind,
            noise_block_id=frame.noise_block_id,
        )

    def get_sources(self, ids: Iterable[SourceKey]) -> tuple[MeasurementRecord, ...]:
        """Materialise the requested records, in the order requested."""
        return tuple(self.get_source(key) for key in ids)

    def values_for(
        self,
        ids: Sequence[SourceKey],
        *,
        overlay: ContentReplacement | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Bulk effective values: NaN for missing/removed/absent, no per-sample objects."""
        out = torch.full((len(ids),), float("nan"), dtype=dtype)
        if overlay is None:
            overlay = self._overlay
        for index, key in enumerate(ids):
            if not self.has_source(key):
                continue
            if overlay is not None and key in overlay.values:
                value = overlay.value_of(key)
                if value is not None:
                    out[index] = float(value)
                continue
            if key in self._removed:
                continue
            frame = self._frame_for(key)
            if frame is None:
                continue
            channel, y, x = int(key.channel_or_phase), key.sensor_y, key.sensor_x
            if not (
                0 <= channel < frame.shape[0]
                and 0 <= y < frame.shape[1]
                and 0 <= x < frame.shape[2]
            ):
                continue
            if frame.validity_of(channel, y, x) == "missing":
                continue
            out[index] = frame.values[channel, y, x].to(dtype)
        return out

    def sample_coordinates(
        self, key: SourceKey, *, overlay: ContentReplacement | None = None
    ) -> tuple[float, float]:
        """Real sampling coordinates of one raw sample (declared offsets preserved)."""
        del overlay
        frame = self._frame_for(key)
        if frame is None or key in self._removed:
            raise SourceNotFound(f"source {key.identity()} is absent in this ledger view")
        return frame.sample_coordinates(int(key.channel_or_phase), key.sensor_y, key.sensor_x)

    def validity_of(self, key: SourceKey) -> str:
        if key in self._removed:
            raise SourceNotFound(f"source {key.identity()} is absent in this ledger view")
        return self.get_source(key).validity_kind

    # -- lifecycle ----------------------------------------------------------
    def frames(self) -> tuple[LedgerFrame, ...]:
        return tuple(self._public_frame(self._frames[identity]) for identity in self._arrival)

    def _public_frame(self, frame: LedgerFrame) -> LedgerFrame:
        """Detached snapshots prevent callers from mutating the immutable raw store."""
        values = frame.values.clone()
        validity = frame.validity.clone()
        if self._overlay is not None:
            for key, value in self._overlay.values.items():
                if self._frame_for(key) is frame and key not in self._removed:
                    index = (int(key.channel_or_phase), key.sensor_y, key.sensor_x)
                    values[index] = 0 if value is None else value
                    if value is None:
                        validity[index] = MISSING
        return replace(
            frame,
            values=values,
            validity=validity,
            checksum=_content_checksum(values, validity),
            sample_offset=None if frame.sample_offset is None else frame.sample_offset.clone(),
            metadata=deepcopy(dict(frame.metadata)),
        )

    def retain_frames(self, identities: Iterable[FrameIdentity]) -> MeasurementLedger:
        """A bounded replay view; discarded RAW stays in the caller's event archive."""
        selected = set(identities)
        view = self.copy()
        view._arrival = [identity for identity in view._arrival if identity in selected]
        view._frames = {identity: view._frames[identity] for identity in view._arrival}
        return view

    def frame_count(self) -> int:
        return len(self._arrival)

    def unique_frames(
        self, *, episode_id: str | None = None, camera_id: str | None = None
    ) -> tuple[FrameIdentity, ...]:
        """Registered exposures in arrival order, filtered by episode/camera."""
        return tuple(
            identity
            for identity in self._arrival
            if (episode_id is None or identity.episode_id == episode_id)
            and (camera_id is None or identity.camera_id == camera_id)
        )

    def unique_frame_count(
        self, *, episode_id: str | None = None, camera_id: str | None = None
    ) -> int:
        return len(self.unique_frames(episode_id=episode_id, camera_id=camera_id))

    def reset(
        self, episode_ids: Iterable[str] | None = None, *, camera_id: str | None = None
    ) -> dict[str, Any]:
        """Partial episode reset: drop matching exposures, never touch other members."""
        selected = None if episode_ids is None else {str(item) for item in episode_ids}
        removed = [
            identity
            for identity in self._arrival
            if (selected is None or identity.episode_id in selected)
            and (camera_id is None or identity.camera_id == camera_id)
        ]
        for identity in removed:
            self._frames.pop(identity, None)
        if removed:
            dropped = set(removed)
            self._arrival = [identity for identity in self._arrival if identity not in dropped]
            # Arrival order stays dense after a reset: the sequence of remaining
            # unique arrivals is what later windows are ordered by.
            reindexed: dict[FrameIdentity, LedgerFrame] = {}
            for position, identity in enumerate(self._arrival):
                frame = self._frames[identity]
                reindexed[identity] = LedgerFrame(
                    identity=frame.identity,
                    values=frame.values,
                    validity=frame.validity,
                    timestamp=frame.timestamp,
                    time_unit=frame.time_unit,
                    value_units=frame.value_units,
                    calibration_version=frame.calibration_version,
                    arrival_index=position,
                    checksum=frame.checksum,
                    noise_block_id=frame.noise_block_id,
                    sample_offset=frame.sample_offset,
                    metadata=frame.metadata,
                )
            self._frames = reindexed
        return {
            "reset_episodes": sorted(selected) if selected is not None else "all",
            "camera_id": camera_id,
            "removed_frames": [tuple(identity) for identity in removed],
            "remaining_frames": len(self._arrival),
        }

    def without(self, ids: Iterable[SourceKey]) -> MeasurementLedger:
        """Derived counterfactual view with the listed samples absent."""
        view = MeasurementLedger(
            self.dataset_id,
            value_units=self.default_value_units,
            time_unit=self.default_time_unit,
            calibration_version=self.default_calibration_version,
            removed=(),
        )
        view._frames = dict(self._frames)
        view._arrival = list(self._arrival)
        view._counters = dict(self._counters)
        removed = frozenset(ids)
        view._removed = self._removed | removed
        view._overlay = self._overlay
        # Physically redact deleted content before ANY downstream decision callback.
        # A has_source-only filter leaves frames()/get_frame() as a leakage channel.
        grouped: dict[FrameIdentity, list[SourceKey]] = {}
        for key in removed:
            frame = self._frame_for(key)
            if frame is not None and self.has_source(key):
                grouped.setdefault(frame.identity, []).append(key)
        for identity, keys in grouped.items():
            frame = view._frames[identity]
            values, validity = frame.values.clone(), frame.validity.clone()
            for key in keys:
                index = (int(key.channel_or_phase), key.sensor_y, key.sensor_x)
                values[index] = 0
                validity[index] = MISSING
            view._frames[identity] = replace(
                frame,
                values=values,
                validity=validity,
                checksum=_content_checksum(values, validity),
            )
        return view

    def with_content(self, replacement: ContentReplacement) -> MeasurementLedger:
        """Derived view where the listed identities carry replaced content.

        Used by ``fixed_decision_replay``: the replacement is a named, versioned
        derivation, exactly like a recalibration, and never edits a raw record.
        """
        view = self.copy()
        view._overlay = replacement
        return view

    def copy(self) -> MeasurementLedger:
        """A shallow copy: frames are immutable, so sharing them is safe."""
        return self.without(())

    def ledger_stats(self) -> dict[str, Any]:
        """Small receipt summary: counts and content hashes, never pixel dumps."""
        validity_counts = {name: 0 for name in VALIDITY_KINDS}
        samples = 0
        for frame in self.frames():
            samples += int(frame.values.numel())
            codes = frame.validity.reshape(-1)
            counts = torch.bincount(codes.to(torch.int64), minlength=len(VALIDITY_CODES))
            for code, name in enumerate(VALIDITY_CODES):
                validity_counts[name] += int(counts[code])
        return {
            "dataset_id": self.dataset_id,
            "frames": len(self._arrival),
            "samples": samples,
            "validity_counts": validity_counts,
            "removed_sources": len(self._removed),
            "arrivals": self._counters["arrivals"],
            "duplicate_arrivals": self._counters["duplicate_arrivals"],
            "skipped_arrivals": self._counters["skipped_arrivals"],
            "frame_checksums": {
                "/".join(str(part) for part in identity): self._frames[identity].checksum
                for identity in self._arrival
            },
        }

    # -- internals ----------------------------------------------------------
    def _frame_for(self, key: SourceKey) -> LedgerFrame | None:
        if key.dataset_id != self.dataset_id:
            return None
        return self._frames.get(
            FrameIdentity(self.dataset_id, key.episode_id, key.camera_id, key.frame_id)
        )


def _mask(mask: Any, shape: Sequence[int], batch: int, name: str) -> torch.Tensor | None:
    if mask is None:
        return None
    tensor = _to_tensor(mask, name=name)
    if tuple(tensor.shape) == tuple(shape[1:]):
        tensor = tensor[None]
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match values {tuple(shape)}")
    if batch < 1:  # pragma: no cover - defensive
        raise ValueError("a measurement frame needs at least one batch member")
    return tensor.to(device="cpu", dtype=torch.bool)


def _validity_codes(
    valid: torch.Tensor | None,
    bad: torch.Tensor | None,
    saturated: torch.Tensor | None,
    index: int,
    grid_shape: Sequence[int],
) -> torch.Tensor:
    """Combine the supplied bitmaps with an explicit, documented precedence.

    ``bad`` wins over ``saturated`` (a dead pixel is not merely clipped), and both
    win over ``valid``: a sample that is explicitly unusable is never silently
    demoted to a missing measurement.  ``missing`` is the state when there is no
    sample at all -- ``valid_mask`` false with no bad/saturated assertion.
    """
    shape = tuple(int(size) for size in grid_shape)
    # No bitmap at all means "no validity assertion was made": the samples are
    # valid.  A *false* valid_mask is an explicit missing assertion (see observe).
    codes = torch.full(shape, VALID, dtype=torch.int8)
    if valid is not None:
        codes = torch.where(~valid[index], torch.full(shape, MISSING, dtype=torch.int8), codes)
    if saturated is not None:
        codes = torch.where(saturated[index], torch.full(shape, SATURATED, dtype=torch.int8), codes)
    if bad is not None:
        codes = torch.where(bad[index], torch.full(shape, BAD, dtype=torch.int8), codes)
    return codes

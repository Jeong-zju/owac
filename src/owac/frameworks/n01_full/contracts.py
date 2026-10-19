"""N01 v1.0 evidence-integration contracts (shared vocabulary for P1-P8).

The prompt of record is ``docs/prompts/n01_v1_0/00_SHARED_CONTRACT.md``
section 3; ``docs/N01_full_development_plan_2026-10-14_v1.0.md`` section 4
fixes the objects these types describe.  Later phases read these docstrings as
the implementation contract, so field order and meaning follow that table.

The module imports neither torch nor numpy and initializes no device:
``import owac.frameworks.n01_full`` must stay as cheap as
``import owac.core.encoder_contract``.  That core module keeps the external
adapter boundary (``MeasurementFrame``, ``ComputeBudget``, ``EncoderOutput``,
``VisualTokens``); these types are the internal vocabulary of the route, so no
core type is redefined here.

Two invariants are easy to get plausibly wrong, so they are enforced in code:

* ``SourceKey`` carries identity only.  Two exposures with identical integers
  are two samples, one packet re-read twice is one sample, and a value, a
  batch slot or an interpolated coordinate never becomes an ID.
* Rejection is a state of the support relation, not of the sensor.  A
  ``MeasurementRecord`` has no "rejected" validity kind: a domain stops
  listing the source as evidence or context, and the sensor record is never
  rewritten to say it was rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "VALIDITY_KINDS",
    "VERSION_FIELDS",
    "SUPPORT_ROLES",
    "UNCERTAINTY_KINDS",
    "SourceKey",
    "CalibratedMeasurement",
    "MeasurementRecord",
    "CorrespondenceMap",
    "SupportDomain",
    "EvidenceEstimate",
    "DependencyNode",
    "ReplayRange",
    "ReadoutSnapshot",
]

#: Sensor-state vocabulary.  "rejected"/"invalid" are deliberately absent: a
#: rejected source is a support-relation state (see ``SupportDomain``).
VALIDITY_KINDS = ("valid", "missing", "bad", "saturated")

#: The four version fields every derived node and every committed readout
#: records.  A model/parameter update retires derived caches that carry an
#: older ``parameter_version``.
VERSION_FIELDS = (
    "parameter_version",
    "support_version",
    "mapping_version",
    "calibration_version",
)

#: Roles a source can play inside one support domain.  Evidence supports the
#: estimated structure; context is neighbourhood material kept for later
#: inference and never counted as structural evidence.
SUPPORT_ROLES = ("evidence", "context")

#: How an estimate expresses what it does not know.  A rank-deficient design
#: must say so (``identifiable_mask``) instead of hiding a small ridge.
UNCERTAINTY_KINDS = ("covariance", "proxy", "none")


@dataclass(frozen=True)
class SourceKey:
    """Identity of one original sensor sample.

    ``frame_id`` is the identity of the exposure/frame, so two exposures with
    identical integer values are distinct samples while a repeated transmission
    of one frame carries one key.  Values, batch slots and interpolation results
    are not identity: nothing here is derived from ``raw_value`` and there is no
    field that could hold a resampled or batched coordinate.

    All fields are required, and the string identity fields must be non-empty:
    a partial or empty identity cannot silently merge unrelated samples.  A
    single-camera dataset declares an explicit camera name instead of ``""``.
    """

    dataset_id: str
    episode_id: str
    camera_id: str
    frame_id: int | str
    channel_or_phase: int | str
    sensor_y: int
    sensor_x: int

    def __post_init__(self) -> None:
        for name in ("dataset_id", "episode_id", "camera_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"SourceKey.{name} must be a non-empty identity string; an empty "
                    "field is not an identity (declare an explicit name for "
                    "single-camera or metadata-free data)"
                )
        if not isinstance(self.frame_id, int | str) or isinstance(self.frame_id, bool):
            raise TypeError("SourceKey.frame_id must be an int or str frame identity")
        if not isinstance(self.channel_or_phase, int | str) or isinstance(
            self.channel_or_phase, bool
        ):
            raise TypeError(
                "SourceKey.channel_or_phase must be a channel index or a CFA phase name"
            )
        if isinstance(self.channel_or_phase, str) and not self.channel_or_phase:
            raise ValueError("SourceKey.channel_or_phase must not be an empty name")
        for name in ("sensor_y", "sensor_x"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"SourceKey.{name} must be a non-negative sensor index")

    def identity(self) -> tuple[Any, ...]:
        """The complete identity tuple used for deduplication; no value enters it."""
        return (
            self.dataset_id,
            self.episode_id,
            self.camera_id,
            self.frame_id,
            self.channel_or_phase,
            self.sensor_y,
            self.sensor_x,
        )


@dataclass(frozen=True)
class MeasurementRecord:
    """One raw (or explicitly missing) sensor sample in the measurement ledger.

    ``raw_value`` is immutable: calibration produces a new versioned
    :class:`CalibratedMeasurement` instead of editing the record.  ``missing``
    samples carry ``raw_value=None`` so a gap can never be read as a zero
    observation; ``bad`` and ``saturated`` samples keep their measured integer
    and are excluded by the support relation, not deleted from the ledger.

    ``calibration_version`` names the calibration this record's derived values
    belong to.  ``noise_block_id`` groups samples that share a noise block.
    """

    key: SourceKey
    raw_value: float | int | None
    value_units: str
    timestamp: float | int
    calibration_version: str
    time_unit: str = "frame"
    validity_kind: str = "valid"
    noise_block_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, SourceKey):
            raise TypeError("MeasurementRecord.key must be a SourceKey")
        if self.validity_kind not in VALIDITY_KINDS:
            raise ValueError(
                f"Unknown validity_kind {self.validity_kind!r}; expected one of "
                f"{VALIDITY_KINDS}.  A rejected source is a state of the support "
                "relation (SupportDomain), never of the sensor record."
            )
        if self.validity_kind == "missing":
            if self.raw_value is not None:
                raise ValueError(
                    "a missing measurement must not carry a value; use raw_value=None so "
                    "a gap is never read as a zero observation"
                )
        elif self.raw_value is None or isinstance(self.raw_value, bool):
            raise ValueError(f"validity_kind {self.validity_kind!r} requires a measured raw_value")
        if not isinstance(self.value_units, str) or not self.value_units:
            raise ValueError("MeasurementRecord.value_units must name the raw unit")
        if not isinstance(self.time_unit, str) or not self.time_unit:
            raise ValueError("MeasurementRecord.time_unit must name the time unit")
        if not isinstance(self.calibration_version, str) or not self.calibration_version:
            raise ValueError(
                "MeasurementRecord.calibration_version must name the calibration "
                "version these values belong to; use 'unknown' when it is not audited"
            )


@dataclass(frozen=True)
class CalibratedMeasurement:
    """A versioned derived view of one :class:`MeasurementRecord`.

    It keeps the parent ``raw_value`` and the ``calibration_version`` it was
    produced under, so applying a new calibration yields a new object and never
    rewrites the raw history.
    """

    key: SourceKey
    value: float
    value_units: str
    calibration_version: str
    raw_value: float | int

    @classmethod
    def from_record(
        cls,
        record: MeasurementRecord,
        value: float,
        *,
        calibration_version: str | None = None,
        value_units: str | None = None,
    ) -> CalibratedMeasurement:
        if record.validity_kind == "missing":
            raise ValueError("a missing measurement has no calibrated value")
        version = record.calibration_version if calibration_version is None else calibration_version
        if not version:
            raise ValueError("calibration_version must name the applied calibration")
        return cls(
            key=record.key,
            value=float(value),
            value_units=record.value_units if value_units is None else value_units,
            calibration_version=version,
            raw_value=record.raw_value,
        )


@dataclass(frozen=True)
class CorrespondenceMap:
    """Mapping from the current reference frame to one historical source frame.

    ``source_xy`` holds **pull** coordinates: ``source_xy[i]`` is the historical
    source coordinate to sample when the reference wants position ``i``.  A push
    map (the reference position a source lands on) is not this type.

    ``tap_source_ids`` / ``tap_weights`` are the explicit sampling coefficients
    the map generates; ``valid`` marks whether the mapping was accepted at all,
    while ``visibility`` and ``flow_risk`` keep the per-tap evidence that
    rejection was or was not justified ("no conflict found" is not "common
    support established").
    """

    source_frame_id: int | str
    reference_frame_id: int | str
    source_xy: Any
    tap_source_ids: tuple[SourceKey, ...]
    tap_weights: tuple[float, ...]
    valid: Any
    mapping_version: str
    visibility: Any | None = None
    flow_risk: Any | None = None

    def __post_init__(self) -> None:
        if not self.mapping_version:
            raise ValueError("CorrespondenceMap.mapping_version must be declared")
        if len(self.tap_source_ids) != len(self.tap_weights):
            raise ValueError(
                "every tap must name exactly one source sample and one weight "
                f"({len(self.tap_source_ids)} ids vs {len(self.tap_weights)} weights)"
            )
        for tap in self.tap_source_ids:
            if not isinstance(tap, SourceKey):
                raise TypeError("CorrespondenceMap taps must be SourceKey instances")
            if tap.frame_id != self.source_frame_id:
                raise ValueError(
                    f"tap source {tap.frame_id!r} is not from the mapped source frame "
                    f"{self.source_frame_id!r}"
                )
        if self.valid is False and self.tap_source_ids:
            raise ValueError(
                "an invalid correspondence carries no sampling taps; keep the rejected "
                "mapping visible through valid/visibility/flow_risk instead of taps"
            )

    @property
    def tap_count(self) -> int:
        """Number of explicit source coefficients in this mapping."""
        return len(self.tap_source_ids)


@dataclass(frozen=True)
class SupportDomain:
    """One structure domain: which measurements support which local structure.

    ``reference_chart`` names the coordinate chart the structure coefficients
    are expressed in; ``basis`` names the structural basis (the same vocabulary
    as ``config.BasisConfig.kind``).  Evidence and context are separate
    columns: evidence measures the domain's structure, context is retained
    neighbourhood material and never counts as structural evidence.  A source
    may appear in both lists only across different domains, not within one.

    Rejection lives here: a rejected source simply leaves the two columns, so
    the sensor record keeps its own validity and its raw value.
    """

    domain_id: str
    reference_chart: str
    basis: str
    phase: int | str | None
    evidence_source_ids: tuple[SourceKey, ...]
    temporal_segment_id: str | int
    decision_version: str
    mapping_version: str
    context_source_ids: tuple[SourceKey, ...] = ()

    def __post_init__(self) -> None:
        for name in ("domain_id", "reference_chart", "basis"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"SupportDomain.{name} must be a non-empty name")
        if not self.temporal_segment_id and not isinstance(self.temporal_segment_id, int):
            raise ValueError("SupportDomain.temporal_segment_id must identify the time segment")
        for name in ("decision_version", "mapping_version"):
            if not getattr(self, name):
                raise ValueError(f"SupportDomain.{name} must be declared")
        for name in ("evidence_source_ids", "context_source_ids"):
            members = getattr(self, name)
            if not isinstance(members, tuple) or not all(
                isinstance(item, SourceKey) for item in members
            ):
                raise TypeError(f"SupportDomain.{name} must be a tuple of SourceKey")
            if len(set(members)) != len(members):
                raise ValueError(
                    f"SupportDomain.{name} contains a duplicate source; one RAW sample "
                    "is registered once per domain"
                )
        overlap = set(self.evidence_source_ids) & set(self.context_source_ids)
        if overlap:
            raise ValueError(
                f"{len(overlap)} source(s) are listed as both evidence and context in "
                f"domain {self.domain_id!r}; the roles are distinct and must not be mixed"
            )

    def role_of(self, source_id: SourceKey) -> str | None:
        """Return 'evidence', 'context' or None for one source in this domain."""
        if source_id in self.evidence_source_ids:
            return "evidence"
        if source_id in self.context_source_ids:
            return "context"
        return None

    @property
    def evidence_count(self) -> int:
        """Number of distinct structural evidence sources in this domain."""
        return len(self.evidence_source_ids)

    @property
    def context_count(self) -> int:
        """Number of distinct retained context sources in this domain."""
        return len(self.context_source_ids)


@dataclass(frozen=True)
class EvidenceEstimate:
    """Structure coefficients and what an estimate knows about them.

    ``prior_id`` names a prior contribution.  A prior is not a measurement: it
    is never added to :attr:`independent_observation_count`, and a shared prior
    is counted once across merged domains (it lives outside the source list).

    ``identifiable_mask`` declares which directions the observations actually
    determine.  ``uncertainty_kind`` says what ``covariance`` is: an exact
    covariance, a declared conservative proxy, or unavailable.
    """

    theta: Any
    uncertainty_kind: str
    identifiable_mask: Any
    evidence_source_ids: tuple[SourceKey, ...]
    support_version: str | int
    covariance: Any | None = None
    prior_id: str | None = None

    def __post_init__(self) -> None:
        if self.uncertainty_kind not in UNCERTAINTY_KINDS:
            raise ValueError(
                f"Unknown uncertainty_kind {self.uncertainty_kind!r}; expected one of "
                f"{UNCERTAINTY_KINDS}"
            )
        if self.uncertainty_kind == "covariance" and self.covariance is None:
            raise ValueError("uncertainty_kind 'covariance' requires a covariance")
        if self.uncertainty_kind == "none" and self.covariance is not None:
            raise ValueError(
                "uncertainty_kind 'none' must not carry a covariance; a rank-deficient "
                "estimate reports the identifiable mask instead of fake precision"
            )
        if self.identifiable_mask is None:
            raise ValueError(
                "identifiable_mask is required: rank deficiency must be reported, not "
                "hidden behind a small regularizer"
            )
        if not isinstance(self.evidence_source_ids, tuple) or not all(
            isinstance(item, SourceKey) for item in self.evidence_source_ids
        ):
            raise TypeError("evidence_source_ids must be a tuple of SourceKey")
        if not self.support_version and not isinstance(self.support_version, int):
            raise ValueError("EvidenceEstimate.support_version must be declared")

    @property
    def independent_observation_count(self) -> int:
        """Distinct raw sources behind this estimate; the prior is not one of them."""
        return len(set(self.evidence_source_ids))

    @property
    def has_prior(self) -> bool:
        """Whether a prior contribution is declared for this estimate."""
        return self.prior_id is not None


@dataclass(frozen=True)
class DependencyNode:
    """A versioned node of the retraction dependency graph.

    Derived content (features, fusion weights, recursive caches, readouts)
    records the four version fields it was derived from.  A model/parameter
    update retires nodes whose ``parameter_version`` no longer matches, so
    :meth:`is_stale_against` answers whether a cached node may still be reused.
    """

    node_id: str
    kind: str
    parent_ids: tuple[str, ...]
    parameter_version: str | int
    support_version: str | int
    mapping_version: str | int
    calibration_version: str | int

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("DependencyNode.node_id must be non-empty")
        if not self.kind:
            raise ValueError("DependencyNode.kind must name the derived object")
        if not isinstance(self.parent_ids, tuple):
            raise TypeError("DependencyNode.parent_ids must be a tuple of node ids")

    def versions(self) -> dict[str, str | int]:
        """The four version fields this node was derived from."""
        return {name: getattr(self, name) for name in VERSION_FIELDS}

    def is_stale_against(self, current: Mapping[str, str | int]) -> bool:
        """True when any version this node was derived from has moved on.

        ``current`` may carry any subset of :data:`VERSION_FIELDS`; an unknown
        version name is an error rather than a silently ignored check.
        """
        unknown = sorted(set(current) - set(VERSION_FIELDS))
        if unknown:
            raise ValueError(f"Unknown version fields {unknown}; expected {VERSION_FIELDS}")
        return any(current[name] != getattr(self, name) for name in current)


@dataclass(frozen=True)
class ReplayRange:
    """The window of unique frames a result can be exactly replayed from."""

    oldest_frame_id: int | str
    newest_frame_id: int | str
    unique_frames: int

    def __post_init__(self) -> None:
        if self.unique_frames < 1:
            raise ValueError("ReplayRange.unique_frames must be at least one unique frame")

    def covers(self, unique_frames: int) -> bool:
        """Whether a request for ``unique_frames`` still lies inside this range."""
        return unique_frames <= self.unique_frames


@dataclass(frozen=True)
class ReadoutSnapshot:
    """One committed readout: every field belongs to the same committed result.

    ``versions`` must declare all four :data:`VERSION_FIELDS`, so a caller can
    never read half-old/half-new state: a snapshot that cannot name its
    parameter/support/mapping/calibration versions is not a commit.
    ``exact_replay_range`` and ``cost`` are part of the result, not optional
    telemetry.
    """

    representation: Any
    validity: Any
    source_time: Any
    commit_time: Any
    versions: Mapping[str, str | int]
    exact_replay_range: ReplayRange
    cost: Mapping[str, Any]

    def __post_init__(self) -> None:
        missing = [name for name in VERSION_FIELDS if name not in self.versions]
        if missing:
            raise ValueError(
                f"a committed readout must declare every dependency version; missing {missing}"
            )
        if not isinstance(self.exact_replay_range, ReplayRange):
            raise TypeError("ReadoutSnapshot.exact_replay_range must be a ReplayRange")
        if not isinstance(self.cost, Mapping):
            raise TypeError("ReadoutSnapshot.cost must be a mapping of measured costs")
        object.__setattr__(self, "versions", MappingProxyType(dict(self.versions)))
        object.__setattr__(self, "cost", MappingProxyType(dict(self.cost)))

    def version_of(self, name: str) -> str | int:
        """One declared dependency version of this commit."""
        if name not in VERSION_FIELDS:
            raise ValueError(f"Unknown version field {name!r}; expected {VERSION_FIELDS}")
        return self.versions[name]

"""N02 P0 data layer: explicit partition manifests, anchor tables, dataset wrapper and the
real-NPZ audit mechanism.

This module is the P0 foundation for the next-generation N02 data path. It never copies a
numeric conversion out of :mod:`owac.raw.rawvla_sequence`; every window is built by the frozen
loader itself, and the only thing this layer changes is *which* episodes and *which* anchor
frames are read.

The four declared pieces are:

* :func:`build_partition_manifest` / :class:`PartitionManifest` -- source identities, groups
  (LIBERO lighting variants of one source trajectory stay together; RoboTwin groups by task +
  source episode; groups are merged by connected components when a scene / acquisition-session
  id is actually declared), an ~80/10/10 train/dev/calibration allocation over *groups*, and
  feasible cross-scene / cross-object held-out slices. The manifest is written and read back
  verbatim: reading it never re-partitions anything.
* :func:`build_anchor_table` / :class:`AnchorTable` -- the declared list of
  ``(episode, current frame)`` anchors with their history indices, action window bounds and
  action-mask hash, plus deterministic ``shard``/``iter_indices`` for multi-worker and DDP
  consumers.
* :class:`ManifestWindowDataset` -- a thin wrapper over
  :class:`owac.raw.rawvla_sequence.RawVLAWindowDataset` that reads exactly the manifest's
  episodes at exactly the anchor table's frames. ``compute_split`` is unreachable on this path
  by construction.
* :func:`audit_dataset_root` -- the real-NPZ audit *mechanism*: per-file field names, dtypes,
  shapes, ranges, resolutions, camera order, timestamp sources, noise availability, language
  and proprio keys, action units and an action-statistics hash, plus an explicit comparison
  against the P0 two-platform protocol table. A missing or empty root returns
  ``status="not_run"`` with a reason; nothing is invented.

Honesty rules enforced here:

* ``UNKNOWN`` (the string ``"unknown"``) is the only sentinel for a source field that cannot be
  identified. It is never replaced by an empty string, a zero or a guessed value.
* Measurements, hashes and verdicts reported by :func:`audit_dataset_root` come from actually
  reading the files it was pointed at. The caller declares ``evidence_kind``; a synthetic
  fixture root must be declared as such and then reports ``real_cache_evidence=False``.
* The real RawVLA NPZ cache is not on this workstation (2026-10-14). The audit reports that as
  ``not_run`` and the tests use small contract fixtures written into ``tmp_path``; they are not
  evidence about the real cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from owac.raw.rawvla_sequence import (
    FORMAT_SPECS,
    EpisodeRef,
    RawVLASample,
    RawVLAWindowDataset,
    SplitManifest,
    WindowOptions,
    _episode_identity,
    _sha256_file,
    build_measurement_frame,
    discover_npz_files,
)

__all__ = [
    "ANCHOR_TABLE_VERSION",
    "DEFAULT_PARTITION_FRACTIONS",
    "PARTITION_MANIFEST_VERSION",
    "PARTITION_NAMES",
    "P0_PROTOCOL",
    "REAL_CACHE_EVIDENCE_KIND",
    "UNKNOWN",
    "AnchorRecord",
    "AnchorTable",
    "ManifestWindowDataset",
    "PartitionManifest",
    "SourceIdentity",
    "audit_dataset_root",
    "build_anchor_table",
    "build_partition_manifest",
    "sample_context",
    "sample_frames",
    "sample_to_inference_and_targets",
    "sample_to_inference_input",
    "sample_to_training_targets",
    "verify_partitions",
]

#: Sentinel for a source field the cache does not identify. Never an empty string, never a zero.
UNKNOWN = "unknown"

PARTITION_NAMES: tuple[str, ...] = ("train", "dev", "calibration")
DEFAULT_PARTITION_FRACTIONS: dict[str, float] = {"train": 0.8, "dev": 0.1, "calibration": 0.1}
PARTITION_MANIFEST_VERSION = "n02-full-partition-v2"
ANCHOR_TABLE_VERSION = "n02-full-anchor-v1"

#: ``evidence_kind`` value that means "the caller is pointing the audit at the deployed RawVLA
#: cache". Any other value keeps ``real_cache_evidence`` at ``False``.
REAL_CACHE_EVIDENCE_KIND = "real_cache"

#: P0 two-platform protocol table (docs/prompts/n02_v1_1/P0_FOUNDATION.md section 3). The audit
#: compares the frozen loader defaults and the observed files against these declarations.
P0_PROTOCOL: dict[str, dict[str, Any]] = {
    "libero": {
        "raw_keys": ("agentview_raw_uint8", "wrist_raw_uint8"),
        "conversion": "float32 / 255.0, no per-frame rescaling",
        "expected_dtypes": ("uint8",),
        "state_source": {
            "key": "robot_states",
            "width": 8,
            "expression": "robot_states[t]",
            "note": (
                "the 8-dim proprio read at the same step as the observation; the cached 92-dim "
                "`states` array is simulator state and is never used as a policy input"
            ),
        },
        "action_supervision": {"offset": 0, "horizon": 10, "expression": "actions[t:t+10]"},
        "formal_statistics": "quantile and the author gripper rule",
        "expected_loader_default": "quantile",
        "loader_mode_is_formal": True,
        "executed_action_window": 5,
        "executed_action_note": (
            "the NPZ records no executed-action trace, so the number of actions the frozen "
            "policy executed per inference is a protocol constant, not an audited field"
        ),
    },
    "robotwin": {
        "raw_keys": (
            "head_camera_raw_float32",
            "left_camera_raw_float32",
            "right_camera_raw_float32",
        ),
        "conversion": "float32 kept in the original HDR-derived float scale, no /255 and no "
        "per-frame max normalisation",
        "expected_dtypes": ("float32",),
        "state_source": {
            "key": "actions",
            "width": 14,
            "expression": "actions[t]",
            "note": (
                "the author's current 14-dim joint proxy; the cached 16-dim `robot_states` is an "
                "end-effector representation and is not the joint state"
            ),
        },
        "action_supervision": {"offset": 1, "horizon": 50, "expression": "actions[t+1:t+51]"},
        "formal_statistics": "explicit checkpoint MEAN_STD / zscore with epsilon",
        "expected_loader_default": "starvla_minmax",
        "loader_mode_is_formal": False,
        "loader_default_role": (
            "the frozen loader's starvla_minmax default is the author demo recipe, not the "
            "formal pi0.5 checkpoint statistics; the formal run must pass the checkpoint "
            "MEAN_STD/zscore statistics explicitly"
        ),
        "executed_action_window": 50,
        "executed_action_note": (
            "the NPZ records no executed-action trace, so the number of actions the frozen "
            "policy executed per inference is a protocol constant, not an audited field"
        ),
    },
}


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(payload: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in payload:
        raise ValueError(f"serialized {where} is missing key {key!r}")
    return payload[key]


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


def _as_path_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


def _source_trajectory_of(episode_group_id: str, relative_path: str) -> str:
    """Source-trajectory identity derived from the loader's own episode identity.

    ``_episode_identity`` yields ``<suite>/episode_000018/variant_00`` for a LIBERO variant and
    ``<task>/episode_000018`` for RoboTwin. Only the declared ``variant_NN`` filename shape is
    treated as a lighting/derived variant; anything else keeps its own identity, so an
    unrecognised naming scheme cannot silently merge two source trajectories.
    """
    identity = episode_group_id or _episode_identity(relative_path)
    head, _, tail = identity.rpartition("/")
    if head and tail.startswith("variant_"):
        return head
    return identity


def _split_task_suite(dataset_format: str, acquisition_source: str) -> tuple[str, str]:
    """``(task_id, suite_id)`` from the loader's acquisition source.

    LIBERO's suite directory names the task family, so it is both task and suite. RoboTwin has
    no suite level in the cache, so the suite stays UNKNOWN instead of repeating the task.
    """
    if acquisition_source in ("", "."):
        return UNKNOWN, UNKNOWN
    if dataset_format == "libero":
        return acquisition_source, acquisition_source
    return acquisition_source, UNKNOWN


def _variant_label(relative_path: str) -> str:
    stem = Path(relative_path).stem
    return stem if stem.startswith("variant_") else UNKNOWN


# --------------------------------------------------------------------------------------
# source identity
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Provenance of one episode file.

    ``episode_id`` identifies the file including its variant; ``source_trajectory_id`` strips
    only the declared ``variant_NN`` suffix and is what groups are derived from. A field the
    cache does not declare is ``UNKNOWN`` (for ``object_ids``: a single ``UNKNOWN`` element,
    which means "no object identity is declared", never "the empty object set").
    """

    relative_path: str
    source_trajectory_id: str
    task_id: str
    suite_id: str
    scene_id: str
    object_ids: tuple[str, ...]
    acquisition_session_id: str
    illumination_group: str
    camera_ids: tuple[str, ...]
    num_frames: int
    sha256: str
    dataset_format: str
    episode_id: str = ""
    acquisition_source: str = ""
    available_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "source_trajectory_id": self.source_trajectory_id,
            "task_id": self.task_id,
            "suite_id": self.suite_id,
            "scene_id": self.scene_id,
            "object_ids": list(self.object_ids),
            "acquisition_session_id": self.acquisition_session_id,
            "illumination_group": self.illumination_group,
            "camera_ids": list(self.camera_ids),
            "num_frames": int(self.num_frames),
            "sha256": self.sha256,
            "dataset_format": self.dataset_format,
            "episode_id": self.episode_id,
            "acquisition_source": self.acquisition_source,
            "available_fields": list(self.available_fields),
            "missing_fields": list(self.missing_fields),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceIdentity:
        return cls(
            relative_path=str(_require(payload, "relative_path", "SourceIdentity")),
            source_trajectory_id=str(_require(payload, "source_trajectory_id", "SourceIdentity")),
            task_id=str(_require(payload, "task_id", "SourceIdentity")),
            suite_id=str(_require(payload, "suite_id", "SourceIdentity")),
            scene_id=str(_require(payload, "scene_id", "SourceIdentity")),
            object_ids=_as_str_tuple(_require(payload, "object_ids", "SourceIdentity")),
            acquisition_session_id=str(
                _require(payload, "acquisition_session_id", "SourceIdentity")
            ),
            illumination_group=str(_require(payload, "illumination_group", "SourceIdentity")),
            camera_ids=_as_str_tuple(_require(payload, "camera_ids", "SourceIdentity")),
            num_frames=int(_require(payload, "num_frames", "SourceIdentity")),
            sha256=str(_require(payload, "sha256", "SourceIdentity")),
            dataset_format=str(_require(payload, "dataset_format", "SourceIdentity")),
            episode_id=str(payload.get("episode_id", "")),
            acquisition_source=str(payload.get("acquisition_source", "")),
            available_fields=_as_str_tuple(payload.get("available_fields", ())),
            missing_fields=_as_str_tuple(payload.get("missing_fields", ())),
        )


#: Optional provenance keys probed in the NPZ arrays and in ``metadata_json``. Nothing is
#: guessed: an absent or ambiguous field stays UNKNOWN.
_CANDIDATE_FIELDS: dict[str, tuple[str, ...]] = {
    "scene_id": ("scene_id", "scene", "scene_name"),
    "acquisition_session_id": ("acquisition_session_id", "session_id", "acquisition_id"),
    "object_ids": ("object_ids", "object_id", "objects", "object_names"),
    "illumination_group": ("lighting_domain", "illumination", "lighting"),
}

#: A candidate provenance array larger than this is not treated as a declared identity field.
_MAX_DECLARED_ENTRIES = 1024


def _read_metadata_json(archive: Any) -> dict[str, Any]:
    """The loader's own metadata convention: one JSON object under ``metadata_json``."""
    if "metadata_json" not in archive.files:
        return {}
    try:
        payload = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    except (ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalise_declared_value(value: Any) -> str | tuple[str, ...] | None:
    array = np.asarray(value)
    if array.size == 0 or array.size > _MAX_DECLARED_ENTRIES:
        return None
    if array.dtype.kind == "S":
        items = [item.decode("utf-8", "replace").strip() for item in array.reshape(-1).tolist()]
    elif array.dtype.kind in {"U", "O"}:
        items = [str(item).strip() for item in array.reshape(-1).tolist()]
    elif np.issubdtype(array.dtype, np.number):
        items = [str(item) for item in array.reshape(-1).tolist()]
    else:
        return None
    items = [item for item in items if item and item.lower() not in {"none", "nan", "null"}]
    if not items:
        return None
    return items[0] if len(items) == 1 else tuple(items)


def _declared_field_value(
    archive: Any,
    keys: Sequence[str],
    metadata: Mapping[str, Any],
) -> str | tuple[str, ...] | None:
    for name in keys:
        if name in metadata:
            normalised = _normalise_declared_value(metadata[name])
            if normalised is not None:
                return normalised
    for name in keys:
        if name in archive.files:
            normalised = _normalise_declared_value(np.asarray(archive[name]))
            if normalised is not None:
                return normalised
    return None


def _probe_episode(path: str | Path, dataset_format: str) -> dict[str, Any]:
    """Read only what the cache actually declares: file keys and optional identity fields."""
    spec = FORMAT_SPECS[dataset_format]
    expected = (*spec.measurement_keys, spec.state_source_key, "actions", "metadata_json")
    with np.load(path, allow_pickle=False) as archive:
        available = tuple(sorted(archive.files))
        metadata = _read_metadata_json(archive)
        probed: dict[str, Any] = {}
        for field_name, candidates in _CANDIDATE_FIELDS.items():
            value = _declared_field_value(archive, candidates, metadata)
            if field_name == "object_ids":
                if value is None:
                    probed[field_name] = (UNKNOWN,)
                elif isinstance(value, tuple):
                    probed[field_name] = value
                else:
                    probed[field_name] = (value,)
            else:
                # A field that resolves to several values is ambiguous, so it cannot be used
                # to merge groups: it stays UNKNOWN.
                probed[field_name] = UNKNOWN if value is None or isinstance(value, tuple) else value
    probed["source_metadata"] = metadata
    probed["available_fields"] = available
    probed["missing_fields"] = tuple(key for key in expected if key not in set(available))
    return probed


# --------------------------------------------------------------------------------------
# partition manifest
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    """Explicit episode partition, written and read back verbatim.

    ``groups`` maps a group id to its relative paths; ``partitions`` maps a partition name to
    its relative paths. A group never spans two partitions. ``to_json``/``from_json`` round-trip
    every field, so a reader cannot silently re-partition the data.
    """

    version: str
    dataset_format: str
    cache_root: str
    seed: int
    fractions: dict[str, float]
    entries: tuple[SourceIdentity, ...]
    groups: dict[str, tuple[str, ...]]
    partitions: dict[str, tuple[str, ...]]
    holdout_slices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    notes: str = ""

    # -- lookup ------------------------------------------------------------------------

    def entry_of(self, relative_path: str) -> SourceIdentity:
        for entry in self.entries:
            if entry.relative_path == relative_path:
                return entry
        raise KeyError(f"relative path {relative_path!r} is not declared in this manifest")

    def partition_of(self, relative_path: str) -> str:
        for name, paths in self.partitions.items():
            if relative_path in paths:
                return name
        raise KeyError(f"relative path {relative_path!r} is in no partition of this manifest")

    def group_of(self, relative_path: str) -> str:
        for group_id, paths in self.groups.items():
            if relative_path in paths:
                return group_id
        raise KeyError(f"relative path {relative_path!r} is in no group of this manifest")

    # -- serialisation -----------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_format": self.dataset_format,
            "cache_root": self.cache_root,
            "seed": int(self.seed),
            "fractions": {name: float(value) for name, value in self.fractions.items()},
            "entries": [entry.as_dict() for entry in self.entries],
            "groups": {name: list(paths) for name, paths in self.groups.items()},
            "partitions": {name: list(paths) for name, paths in self.partitions.items()},
            "holdout_slices": {name: list(paths) for name, paths in self.holdout_slices.items()},
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return _canonical_json(self.as_dict()) + "\n"

    @classmethod
    def from_json(cls, payload: str | Mapping[str, Any]) -> PartitionManifest:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
        return cls(
            version=str(_require(data, "version", "PartitionManifest")),
            dataset_format=str(_require(data, "dataset_format", "PartitionManifest")),
            cache_root=str(_require(data, "cache_root", "PartitionManifest")),
            seed=int(_require(data, "seed", "PartitionManifest")),
            fractions={
                str(name): float(value)
                for name, value in _require(data, "fractions", "PartitionManifest").items()
            },
            entries=tuple(
                SourceIdentity.from_dict(entry)
                for entry in _require(data, "entries", "PartitionManifest")
            ),
            groups={
                str(name): _as_path_tuple(paths)
                for name, paths in _require(data, "groups", "PartitionManifest").items()
            },
            partitions={
                str(name): _as_path_tuple(paths)
                for name, paths in _require(data, "partitions", "PartitionManifest").items()
            },
            holdout_slices={
                str(name): _as_path_tuple(paths)
                for name, paths in data.get("holdout_slices", {}).items()
            },
            notes=str(data.get("notes", "")),
        )

    def content_hash(self) -> str:
        """sha256 of the canonical serialisation: identical before and after a read-back."""
        return _sha256_text(self.to_json())


def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not fractions:
        raise ValueError("fractions must declare at least one partition")
    cleaned: dict[str, float] = {}
    for name, value in fractions.items():
        fraction = float(value)
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction for {name!r} must lie strictly between 0 and 1")
        cleaned[str(name)] = fraction
    return cleaned


def _build_source_identities(
    episodes: Sequence[EpisodeRef],
    *,
    dataset_format: str,
    cache_root: str | Path,
) -> list[SourceIdentity]:
    spec = FORMAT_SPECS[dataset_format]
    root = Path(cache_root)
    identities: list[SourceIdentity] = []
    seen: set[str] = set()
    for episode in episodes:
        relative = episode.relative_path or Path(episode.path).name
        if relative in seen:
            raise ValueError(f"episode {relative!r} is declared twice")
        seen.add(relative)
        path = Path(episode.path) if episode.path else root / relative
        if not path.is_file():
            raise FileNotFoundError(f"episode file {path} does not exist")
        num_frames = int(episode.num_frames)
        if num_frames <= 0:
            raise ValueError(f"episode {relative!r} declares {num_frames} frames")
        probed = _probe_episode(path, dataset_format)
        episode_id = episode.episode_group_id or _episode_identity(relative)
        acquisition_source = episode.acquisition_source or UNKNOWN
        task_id, suite_id = _split_task_suite(dataset_format, acquisition_source)
        metadata = probed["source_metadata"]
        # Dataset directory names identify suites, not LIBERO tasks. Prefer the
        # declared source demonstration so renamed rerenders stay in one group.
        task_name = metadata.get("task_name")
        if isinstance(task_name, str) and task_name:
            task_id = task_name
        suite_id = str(metadata.get("suite") or suite_id)
        source_trajectory = _source_trajectory_of(episode_id, relative)
        if dataset_format == "libero" and metadata.get("hdf5_path") and metadata.get("demo_key"):
            source_trajectory = (
                f"{suite_id}/{Path(metadata['hdf5_path']).name}/{metadata['demo_key']}"
            )
        elif dataset_format == "robotwin" and metadata.get("source_trajectory"):
            source_trajectory = f"{task_id}/{Path(metadata['source_trajectory']).name}"
        illumination = probed["illumination_group"]
        if illumination == UNKNOWN:
            illumination = _variant_label(relative)
        identities.append(
            SourceIdentity(
                relative_path=relative,
                source_trajectory_id=source_trajectory,
                task_id=task_id,
                suite_id=suite_id,
                scene_id=probed["scene_id"],
                object_ids=probed["object_ids"],
                acquisition_session_id=probed["acquisition_session_id"],
                illumination_group=illumination,
                camera_ids=tuple(spec.camera_ids),
                num_frames=num_frames,
                sha256=episode.sha256 or _sha256_file(path),
                dataset_format=dataset_format,
                episode_id=episode_id,
                acquisition_source=acquisition_source,
                available_fields=probed["available_fields"],
                missing_fields=tuple(
                    sorted(
                        set(probed["missing_fields"])
                        | {
                            name
                            for name in ("scene_id", "acquisition_session_id", "object_ids")
                            if probed[name] in (UNKNOWN, (UNKNOWN,))
                        }
                    )
                ),
            )
        )
    identities.sort(key=lambda item: item.relative_path)
    return identities


def _base_group_id(identity: SourceIdentity) -> str:
    return f"{identity.dataset_format}/{identity.source_trajectory_id}"


def _group_identities(
    identities: Sequence[SourceIdentity],
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """Source-trajectory groups merged by connected components over declared shared fields."""
    base_ids = sorted({_base_group_id(identity) for identity in identities})
    parent = {group_id: group_id for group_id in base_ids}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    merged_by: list[str] = []
    for identity in identities:
        group_id = _base_group_id(identity)
        for field_name in ("scene_id", "acquisition_session_id"):
            value = getattr(identity, field_name)
            if value != UNKNOWN and value:
                union(group_id, f"{field_name}::{value}")
                merged_by.append(field_name)
    grouped: dict[str, list[str]] = {}
    for identity in identities:
        root = find(_base_group_id(identity))
        grouped.setdefault(root, []).append(identity.relative_path)
    groups = {group_id: tuple(sorted(paths)) for group_id, paths in sorted(grouped.items())}
    notes: list[str] = []
    if merged_by:
        declared = sorted(set(merged_by))
        notes.append(
            "groups were enlarged by connected components over declared "
            f"{', '.join(declared)} values"
        )
    else:
        notes.append(
            "no scene or acquisition-session id is declared, so groups are exactly the source "
            "trajectories and an object-disjoint split cannot be claimed"
        )
    if any(identity.object_ids == (UNKNOWN,) for identity in identities):
        notes.append(
            "object ids are UNKNOWN for at least one episode; object-disjoint cannot be "
            "declared for this manifest"
        )
    return groups, notes


def _assign_partitions(
    groups: Mapping[str, tuple[str, ...]],
    frames_of: Mapping[str, int],
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> dict[str, tuple[str, ...]]:
    """Frames-weighted ~80/10/10 allocation over whole groups.

    Groups are visited in a deterministic hash order and each one joins the partition furthest
    below its proportional share. The result is a pure function of (groups, seed, fractions),
    and with at least three groups every partition receives one because all deficits start at
    zero and ties fall back to the declared partition order.
    """
    order = {name: index for index, name in enumerate(fractions)}
    ordered_groups = sorted(
        groups, key=lambda group_id: (_sha256_text(f"{seed}:{group_id}"), group_id)
    )
    assigned: dict[str, list[str]] = {name: [] for name in fractions}
    frames: dict[str, int] = {name: 0 for name in fractions}
    for group_id in ordered_groups:
        group_frames = sum(int(frames_of[path]) for path in groups[group_id])
        target = min(
            fractions,
            key=lambda name: (frames[name] / fractions[name], order[name]),
        )
        assigned[target].extend(groups[group_id])
        frames[target] += group_frames
    return {name: tuple(sorted(paths)) for name, paths in assigned.items()}


def _holdout_slices(
    groups: Mapping[str, tuple[str, ...]],
    identity_of: Mapping[str, SourceIdentity],
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """Feasible cross-scene / cross-object held-out slices (empty when not feasible).

    A slice is group-complete: an evaluation slice may not split a group, because two files of
    one source trajectory must not end up in different sides.
    """
    slices: dict[str, tuple[str, ...]] = {}
    notes: list[str] = []

    scenes_by_group = {
        group_id: {identity_of[path].scene_id for path in paths}
        for group_id, paths in groups.items()
    }
    frames_of_group = {
        group_id: sum(int(identity_of[path].num_frames) for path in paths)
        for group_id, paths in groups.items()
    }
    if any(UNKNOWN in values for values in scenes_by_group.values()):
        notes.append("cross_scene slice not available: at least one group declares no scene id")
    elif any(len(values) > 1 for values in scenes_by_group.values()):
        notes.append("cross_scene slice not available: a merged group spans more than one scene id")
    else:
        scene_frames: dict[str, int] = {}
        for group_id, values in scenes_by_group.items():
            for scene in values:
                scene_frames[scene] = scene_frames.get(scene, 0) + frames_of_group[group_id]
        if len(scene_frames) < 2:
            notes.append("cross_scene slice not available: only one scene id is declared")
        else:
            chosen = min(scene_frames, key=lambda name: (scene_frames[name], name))
            held = [group_id for group_id, values in scenes_by_group.items() if values == {chosen}]
            if not held or len(held) == len(groups):
                notes.append(
                    "cross_scene slice not available: the smallest scene is not a proper subset "
                    "of the groups"
                )
            else:
                slices["cross_scene"] = tuple(
                    sorted(path for group_id in held for path in groups[group_id])
                )
                notes.append(
                    f"cross_scene slice holds out scene {chosen!r} "
                    f"({len(held)} of {len(groups)} groups)"
                )

    objects_by_group = {
        group_id: {
            object_id
            for path in paths
            for object_id in identity_of[path].object_ids
            if object_id != UNKNOWN
        }
        for group_id, paths in groups.items()
    }
    if any(not values for values in objects_by_group.values()):
        notes.append("cross_object slice not available: at least one group declares no object id")
    else:
        object_frames: dict[str, int] = {}
        for group_id, values in objects_by_group.items():
            for object_id in values:
                object_frames[object_id] = (
                    object_frames.get(object_id, 0) + frames_of_group[group_id]
                )
        chosen_object = min(object_frames, key=lambda name: (object_frames[name], name))
        held = [
            group_id for group_id, values in objects_by_group.items() if chosen_object in values
        ]
        if not held or len(held) == len(groups):
            notes.append(
                "cross_object slice not available: the selected object is not carried by a "
                "proper subset of the groups"
            )
        else:
            slices["cross_object"] = tuple(
                sorted(path for group_id in held for path in groups[group_id])
            )
            notes.append(
                f"cross_object slice holds out object {chosen_object!r} "
                f"({len(held)} of {len(groups)} groups)"
            )
    return slices, notes


def build_partition_manifest(
    episodes: Sequence[EpisodeRef],
    *,
    dataset_format: str,
    cache_root: str | Path,
    seed: int,
    fractions: Mapping[str, float] | None = None,
    version: str = PARTITION_MANIFEST_VERSION,
) -> PartitionManifest:
    """Build the explicit four-way manifest from the loader's own episode references.

    Rules: LIBERO variant files of one source trajectory share a group; RoboTwin groups by task
    + source episode; declared scene / acquisition-session ids merge groups through connected
    components; partitions are allocated over whole groups at ~80/10/10 by frames; cross-scene
    and cross-object slices are emitted only when they are actually feasible.
    """
    if dataset_format not in FORMAT_SPECS:
        raise ValueError(f"unknown dataset_format: {dataset_format!r}")
    if not episodes:
        raise ValueError("cannot build a partition manifest from zero episodes")
    declared_fractions = _validate_fractions(
        DEFAULT_PARTITION_FRACTIONS if fractions is None else fractions
    )
    identities = _build_source_identities(
        episodes, dataset_format=dataset_format, cache_root=cache_root
    )
    groups, group_notes = _group_identities(identities)
    frames_of = {identity.relative_path: identity.num_frames for identity in identities}
    identity_of = {identity.relative_path: identity for identity in identities}
    # A shared scene/session component can span tasks: keep that component
    # intact and stratify by its task set rather than splitting its dependency.
    strata: dict[tuple[str, ...], dict[str, tuple[str, ...]]] = {}
    for group_id, paths in groups.items():
        tasks = tuple(sorted({identity_of[path].task_id for path in paths}))
        strata.setdefault(tasks, {})[group_id] = paths
    assigned: dict[str, list[str]] = {name: [] for name in declared_fractions}
    for tasks, task_groups in sorted(strata.items()):
        allocation = _assign_partitions(
            task_groups, frames_of, seed=seed, fractions=declared_fractions
        )
        for name, paths in allocation.items():
            assigned[name].extend(paths)
        if len(task_groups) < len(declared_fractions):
            group_notes.append(
                f"task stratum {tasks!r} has only {len(task_groups)} groups; "
                "all partitions cannot be populated without source leakage"
            )
    partitions = {name: tuple(sorted(paths)) for name, paths in assigned.items()}
    holdout, holdout_notes = _holdout_slices(groups, identity_of)
    allocation = ", ".join(f"{name}={fraction}" for name, fraction in declared_fractions.items())
    notes = [
        f"explicit partition manifest for {dataset_format}; group allocation fractions: "
        f"{allocation}; stratified within task or connected task set",
        *group_notes,
        *holdout_notes,
    ]
    return PartitionManifest(
        version=version,
        dataset_format=dataset_format,
        cache_root=str(cache_root),
        seed=int(seed),
        fractions=declared_fractions,
        entries=tuple(identities),
        groups=groups,
        partitions=partitions,
        holdout_slices=holdout,
        notes="; ".join(notes),
    )


def verify_partitions(manifest: PartitionManifest) -> dict[str, Any]:
    """Structural verification of a manifest: disjointness, group discipline, coverage."""
    violations: list[str] = []
    counts: dict[str, Any] = {"partitions": {}, "groups": len(manifest.groups)}
    declared_entries = {entry.relative_path for entry in manifest.entries}

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for name, paths in manifest.partitions.items():
        duplicate_in_partition = sorted({path for path in paths if paths.count(path) > 1})
        if duplicate_in_partition:
            violations.append(f"partition {name!r} repeats paths: {duplicate_in_partition[:5]}")
        for path in paths:
            if path in seen:
                duplicates.append(path)
            else:
                seen[path] = name
    if duplicates:
        violations.append(
            f"{len(duplicates)} paths appear in more than one partition: {sorted(duplicates)[:5]}"
        )

    unknown_in_partitions = sorted(set(seen) - declared_entries)
    if unknown_in_partitions:
        violations.append(
            f"partition paths missing from the entry table: {unknown_in_partitions[:5]}"
        )
    missing_from_partitions = sorted(declared_entries - set(seen))
    if missing_from_partitions:
        violations.append(f"entries in no partition: {missing_from_partitions[:5]}")

    group_partitions: dict[str, set[str]] = {group_id: set() for group_id in manifest.groups}
    for name, paths in manifest.partitions.items():
        for path in paths:
            for group_id, group_paths in manifest.groups.items():
                if path in group_paths:
                    group_partitions[group_id].add(name)
    for group_id, names in sorted(group_partitions.items()):
        if len(names) > 1:
            violations.append(f"group {group_id!r} spans partitions {sorted(names)}")
        if not names:
            violations.append(f"group {group_id!r} is in no partition")

    group_membership = set()
    for paths in manifest.groups.values():
        for path in paths:
            group_membership.add(path)
    if group_membership != declared_entries:
        violations.append("groups and entry table disagree on the covered files")
    for name, paths in manifest.partitions.items():
        if not paths:
            violations.append(f"partition {name!r} is empty")

    for name, paths in manifest.partitions.items():
        frames = sum(manifest.entry_of(path).num_frames for path in paths)
        partition_paths = set(paths)
        groups_here = sum(
            1 for group_paths in manifest.groups.values() if set(group_paths) & partition_paths
        )
        counts["partitions"][name] = {
            "files": len(paths),
            "frames": frames,
            "groups": groups_here,
        }
    counts["files"] = len(declared_entries)
    counts["frames"] = sum(entry.num_frames for entry in manifest.entries)

    return {
        "disjoint": not duplicates,
        "groups_single_partition": not any(len(names) > 1 for names in group_partitions.values()),
        "violations": violations,
        "counts": counts,
    }


# --------------------------------------------------------------------------------------
# anchor table
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    """One declared sampling anchor.

    ``t`` is the anchor's position on the episode's strided anchor grid and
    ``current_frame_index`` is the source frame it reads (``t * stride``). ``history_indices``
    holds the outer observation frame indices of the window (oldest first, clipped to the
    episode); burst positions are ``history_indices[k] - j`` clipped, for ``j`` in
    ``0 .. burst_frames-1``. ``action_start``/``action_stop`` are the half-open supervision
    bounds in source frames and ``action_mask_hash`` covers the resulting padding mask.
    """

    episode_id: str
    relative_path: str
    t: int
    current_frame_index: int
    history_indices: tuple[int, ...]
    action_start: int
    action_stop: int
    action_mask_hash: str
    partition: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "relative_path": self.relative_path,
            "t": int(self.t),
            "current_frame_index": int(self.current_frame_index),
            "history_indices": [int(value) for value in self.history_indices],
            "action_start": int(self.action_start),
            "action_stop": int(self.action_stop),
            "action_mask_hash": self.action_mask_hash,
            "partition": self.partition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnchorRecord:
        return cls(
            episode_id=str(_require(payload, "episode_id", "AnchorRecord")),
            relative_path=str(_require(payload, "relative_path", "AnchorRecord")),
            t=int(_require(payload, "t", "AnchorRecord")),
            current_frame_index=int(_require(payload, "current_frame_index", "AnchorRecord")),
            history_indices=tuple(
                int(value) for value in _require(payload, "history_indices", "AnchorRecord")
            ),
            action_start=int(_require(payload, "action_start", "AnchorRecord")),
            action_stop=int(_require(payload, "action_stop", "AnchorRecord")),
            action_mask_hash=str(_require(payload, "action_mask_hash", "AnchorRecord")),
            partition=str(_require(payload, "partition", "AnchorRecord")),
        )


def _action_mask_hash(
    relative_path: str,
    *,
    action_start: int,
    action_stop: int,
    num_frames: int,
    mask: np.ndarray,
) -> str:
    payload = {
        "relative_path": relative_path,
        "action_start": int(action_start),
        "action_stop": int(action_stop),
        "num_frames": int(num_frames),
        "mask": [bool(value) for value in mask.tolist()],
    }
    return _sha256_text(_canonical_json(payload))


@dataclass(frozen=True, slots=True)
class AnchorTable:
    """Declared anchors of one partition (see :class:`AnchorRecord`)."""

    version: str
    partition: str
    dataset_format: str
    cache_root: str
    stride: int
    rnn_frames: int
    burst_frames: int
    action_horizon: int
    action_offset: int
    window: dict[str, Any]
    entries: tuple[AnchorRecord, ...]
    positions: tuple[int, ...] = ()
    world_size: int = 1
    rank: int = 0
    notes: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[AnchorRecord]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> AnchorRecord:
        return self.entries[index]

    def iter_indices(self) -> Iterator[int]:
        """Global anchor positions this table serves (needed for shard union/disjointness)."""
        if self.positions:
            return iter(self.positions)
        return iter(range(len(self.entries)))

    def shard(self, world_size: int, rank: int) -> AnchorTable:
        """Strided shard over global anchor positions: disjoint shards, complete union."""
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must lie in [0, {world_size}), got {rank}")
        positions = tuple(
            position for position in self.iter_indices() if position % world_size == rank
        )
        members = {
            position: entry
            for position, entry in zip(self.iter_indices(), self.entries, strict=True)
        }
        return dataclasses.replace(
            self,
            entries=tuple(members[position] for position in positions),
            positions=positions,
            world_size=world_size,
            rank=rank,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "partition": self.partition,
            "dataset_format": self.dataset_format,
            "cache_root": self.cache_root,
            "stride": int(self.stride),
            "rnn_frames": int(self.rnn_frames),
            "burst_frames": int(self.burst_frames),
            "action_horizon": int(self.action_horizon),
            "action_offset": int(self.action_offset),
            "window": dict(self.window),
            "entries": [entry.as_dict() for entry in self.entries],
            "positions": [int(value) for value in self.iter_indices()],
            "world_size": int(self.world_size),
            "rank": int(self.rank),
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return _canonical_json(self.as_dict()) + "\n"

    @classmethod
    def from_json(cls, payload: str | Mapping[str, Any]) -> AnchorTable:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
        return cls(
            version=str(_require(data, "version", "AnchorTable")),
            partition=str(_require(data, "partition", "AnchorTable")),
            dataset_format=str(_require(data, "dataset_format", "AnchorTable")),
            cache_root=str(_require(data, "cache_root", "AnchorTable")),
            stride=int(_require(data, "stride", "AnchorTable")),
            rnn_frames=int(_require(data, "rnn_frames", "AnchorTable")),
            burst_frames=int(_require(data, "burst_frames", "AnchorTable")),
            action_horizon=int(_require(data, "action_horizon", "AnchorTable")),
            action_offset=int(_require(data, "action_offset", "AnchorTable")),
            window=dict(_require(data, "window", "AnchorTable")),
            entries=tuple(
                AnchorRecord.from_dict(entry) for entry in _require(data, "entries", "AnchorTable")
            ),
            positions=tuple(int(value) for value in data.get("positions", ())),
            world_size=int(data.get("world_size", 1)),
            rank=int(data.get("rank", 0)),
            notes=str(data.get("notes", "")),
        )

    def sha256(self) -> str:
        """sha256 of the canonical serialisation; the same parameters give the same digest."""
        return _sha256_text(self.to_json())


def build_anchor_table(
    manifest: PartitionManifest,
    partition: str,
    *,
    window: WindowOptions | None = None,
    stride: int = 1,
    rnn_frames: int = 1,
    burst_frames: int = 1,
    action_horizon: int | None = None,
    action_offset: int | None = None,
    max_anchors: int | None = None,
) -> AnchorTable:
    """Enumerate deterministic anchors over one partition of a partition manifest.

    ``stride`` is the anchor grid step; the window's own ``observation_stride`` (or ``stride``
    when no window is given) sets the spacing of the history indices. When ``window`` is given,
    its resolved values must agree with the explicit arguments, because the anchors describe
    exactly the windows the loader will build.
    """
    if partition not in manifest.partitions:
        raise KeyError(
            f"partition {partition!r} is not declared in the manifest; "
            f"have {sorted(manifest.partitions)}"
        )
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if rnn_frames < 1 or burst_frames < 1:
        raise ValueError("rnn_frames and burst_frames must be >= 1")
    spec = FORMAT_SPECS[manifest.dataset_format]
    resolved: WindowOptions | None = None
    if window is None:
        observation_stride = stride
        horizon = int(spec.default_action_horizon if action_horizon is None else action_horizon)
        offset = int(spec.default_action_offset if action_offset is None else action_offset)
    else:
        resolved = window.resolved_for(spec)
        observation_stride = int(resolved.observation_stride)
        if resolved.rnn_frames != rnn_frames or resolved.burst_frames != burst_frames:
            raise ValueError(
                "window and anchor arguments disagree on the window shape: window declares "
                f"rnn_frames={resolved.rnn_frames}, burst_frames={resolved.burst_frames}"
            )
        if action_horizon is not None and int(action_horizon) != resolved.action_horizon:
            raise ValueError(
                f"window declares action_horizon={resolved.action_horizon}, "
                f"argument is {action_horizon}"
            )
        if action_offset is not None and int(action_offset) != resolved.action_offset:
            raise ValueError(
                f"window declares action_offset={resolved.action_offset}, "
                f"argument is {action_offset}"
            )
        horizon = int(resolved.action_horizon)
        offset = int(resolved.action_offset)
    if horizon < 1:
        raise ValueError("action_horizon must be >= 1")
    if offset < 0:
        raise ValueError("action_offset must be >= 0")
    if max_anchors is not None and max_anchors < 1:
        raise ValueError("max_anchors must be >= 1 when set")

    records: list[AnchorRecord] = []
    truncated = False
    for relative_path in manifest.partitions[partition]:
        identity = manifest.entry_of(relative_path)
        num_frames = int(identity.num_frames)
        if num_frames <= 0:
            raise ValueError(f"episode {relative_path!r} declares {num_frames} frames")
        for grid_index, current in enumerate(range(0, num_frames, stride)):
            history = np.clip(
                current - observation_stride * np.arange(rnn_frames - 1, -1, -1),
                0,
                num_frames - 1,
            )
            action_start = current + offset
            action_stop = action_start + horizon
            raw_indices = np.arange(action_start, action_stop)
            mask = (raw_indices >= 0) & (raw_indices <= num_frames - 1)
            records.append(
                AnchorRecord(
                    episode_id=identity.episode_id,
                    relative_path=relative_path,
                    t=grid_index,
                    current_frame_index=int(current),
                    history_indices=tuple(int(value) for value in history.tolist()),
                    action_start=int(action_start),
                    action_stop=int(action_stop),
                    action_mask_hash=_action_mask_hash(
                        relative_path,
                        action_start=action_start,
                        action_stop=action_stop,
                        num_frames=num_frames,
                        mask=mask,
                    ),
                    partition=partition,
                )
            )
            if max_anchors is not None and len(records) >= max_anchors:
                truncated = True
                break
        if truncated:
            break
    if resolved is None:
        resolved = WindowOptions(
            observation_stride=observation_stride,
            rnn_frames=rnn_frames,
            burst_frames=burst_frames,
            action_horizon=horizon,
            action_offset=offset,
        ).resolved_for(spec)
    window_payload = dataclasses.asdict(resolved)
    notes = (
        f"anchor grid stride={stride}; history spacing={observation_stride}; "
        f"action window t+{offset} .. t+{offset}+{horizon}"
    )
    if truncated:
        notes += f"; truncated to {len(records)} anchors by max_anchors={max_anchors}"
    return AnchorTable(
        version=ANCHOR_TABLE_VERSION,
        partition=partition,
        dataset_format=manifest.dataset_format,
        cache_root=manifest.cache_root,
        stride=int(stride),
        rnn_frames=int(rnn_frames),
        burst_frames=int(burst_frames),
        action_horizon=int(horizon),
        action_offset=int(offset),
        window=window_payload,
        entries=tuple(records),
        positions=tuple(range(len(records))),
        notes=notes,
    )


# --------------------------------------------------------------------------------------
# manifest-driven dataset
# --------------------------------------------------------------------------------------


class _FixedFrameDraw:
    """Stand-in for the loader's seeded RNG: it always returns the declared anchor frame."""

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def integers(self, low: int, high: int | None = None) -> np.int64:
        return np.int64(self.value)


def _explicit_split_manifest(
    manifest: PartitionManifest,
    partition: str,
    relative_paths: Sequence[str],
) -> SplitManifest:
    """A ``SplitManifest`` that states the *declared* partition instead of drawing one.

    Pre-seeding the loader's split cache with this object is what keeps ``compute_split``
    unreachable on the manifest path while leaving every numeric conversion untouched.
    """
    frames = sum(manifest.entry_of(path).num_frames for path in relative_paths)
    sources = sorted({manifest.entry_of(path).acquisition_source for path in relative_paths})
    return SplitManifest(
        version=f"{manifest.version}+explicit-{partition}",
        seed=int(manifest.seed),
        val_fraction=0.0,
        dataset_format=manifest.dataset_format,
        cache_root=manifest.cache_root,
        train_sources=tuple(sources),
        val_sources=(),
        train_episodes=tuple(sorted(relative_paths)),
        val_episodes=(),
        episodes=(),
        num_frames={"train": frames, "validation": 0},
        notes=(
            f"explicit {partition!r} partition from PartitionManifest {manifest.version}; "
            "no random split is computed for this loader"
        ),
    )


class _PartitionLoader(RawVLAWindowDataset):
    """Frozen loader restricted to one explicit partition and one declared anchor frame.

    It subclasses the frozen loader instead of copying it, so all unit conversion, mask and
    normalisation code stays in one place. Two things differ: the episode list comes from the
    manifest, and the current frame comes from the anchor that the wrapper injects rather than
    from the seeded RNG.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        manifest: PartitionManifest,
        partition: str,
        window: WindowOptions | None,
        stats: Any,
        format_spec: Any = None,
        noise: Any = None,
        cfa_pattern: str = "",
        cfa_calibration: Any = None,
    ) -> None:
        relative_paths = tuple(manifest.partitions[partition])
        paths = [str(Path(cache_root) / relative) for relative in relative_paths]
        super().__init__(
            cache_root,
            dataset_format=manifest.dataset_format,
            window=window,
            stats=stats,
            noise=noise,
            cfa_pattern=cfa_pattern,
            cfa_calibration=cfa_calibration,
            format_spec=format_spec,
            samples_per_episode=1,
            split="train",
            episode_paths=paths,
        )
        self.declared_manifest = manifest
        self.declared_partition = partition
        self._anchor_slot: tuple[int, int] | None = None
        # Pre-seed the cached split with the declared partition: `split_manifest()` returns this
        # object and never reaches `compute_split`.
        self._manifest = _explicit_split_manifest(manifest, partition, relative_paths)

    def set_anchor(self, slot: int, current_frame_index: int) -> int:
        self._anchor_slot = (int(slot), int(current_frame_index))
        return int(slot)

    def clear_anchor(self) -> None:
        self._anchor_slot = None

    def indices_for_index(self, index: int) -> tuple[int, int, Any]:
        if self._anchor_slot is None:
            raise RuntimeError(
                "this loader only serves anchors declared by ManifestWindowDataset; it never "
                "draws a random frame"
            )
        slot, current = self._anchor_slot
        return slot, current, _FixedFrameDraw(current)


class ManifestWindowDataset:
    """Sample RawVLA windows at the manifest's episodes and the anchor table's frames.

    ``__getitem__`` is a pure function of the dataset index: the episode comes from the anchor's
    ``relative_path`` and the frame from ``current_frame_index``, so workers and DDP ranks do
    not depend on initialisation order. The returned :class:`RawVLASample` carries the
    partition and anchor metadata in ``provenance`` and its ``split`` field is the partition
    name.
    """

    def __init__(
        self,
        cache_root: str | Path | None,
        manifest: PartitionManifest,
        partition: str,
        anchor_table: AnchorTable,
        *,
        window_options: WindowOptions | None = None,
        normalisation: Any = None,
        format_spec: Any = None,
        noise: Any = None,
        cfa_pattern: str = "",
        cfa_calibration: Any = None,
    ) -> None:
        if partition not in manifest.partitions:
            raise KeyError(
                f"partition {partition!r} is not declared in the manifest; "
                f"have {sorted(manifest.partitions)}"
            )
        if anchor_table.partition != partition:
            raise ValueError(
                f"anchor table is for partition {anchor_table.partition!r}, "
                f"dataset asks for {partition!r}"
            )
        if anchor_table.dataset_format != manifest.dataset_format:
            raise ValueError(
                f"anchor table format {anchor_table.dataset_format!r} does not match the "
                f"manifest format {manifest.dataset_format!r}"
            )
        root = Path(manifest.cache_root if cache_root is None else cache_root)
        declared = set(manifest.partitions[partition])
        stray = sorted(
            {
                anchor.relative_path
                for anchor in anchor_table
                if anchor.relative_path not in declared
            }
        )
        if stray:
            raise ValueError(
                f"{len(stray)} anchors read outside the {partition!r} partition, e.g. {stray[0]!r}"
            )
        self.cache_root = root
        self.manifest = manifest
        self.partition = partition
        self.anchor_table = anchor_table
        spec = format_spec or FORMAT_SPECS[manifest.dataset_format]
        declared_window = (
            WindowOptions(**anchor_table.window).resolved_for(spec)
            if anchor_table.window
            else WindowOptions(
                observation_stride=anchor_table.stride,
                rnn_frames=anchor_table.rnn_frames,
                burst_frames=anchor_table.burst_frames,
                action_horizon=anchor_table.action_horizon,
                action_offset=anchor_table.action_offset,
            ).resolved_for(spec)
        )
        if window_options is not None and window_options.resolved_for(spec) != declared_window:
            raise ValueError("window_options disagree with the frozen anchor table")
        window_options = declared_window
        self.window_options = window_options
        self.normalisation = normalisation
        self._manifest_hash = manifest.content_hash()
        self._anchor_hash = anchor_table.sha256()
        self._loader = _PartitionLoader(
            root,
            manifest=manifest,
            partition=partition,
            window=window_options,
            stats=normalisation,
            format_spec=format_spec,
            noise=noise,
            cfa_pattern=cfa_pattern,
            cfa_calibration=cfa_calibration,
        )
        self._slots = {
            episode.relative_path: index for index, episode in enumerate(self._loader.episodes())
        }
        missing = sorted(declared - set(self._slots))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} {partition!r} episodes are missing under {root}, "
                f"e.g. {missing[0]!r}"
            )

    # -- introspection -----------------------------------------------------------------

    @property
    def episode_paths(self) -> tuple[str, ...]:
        return tuple(self._loader.paths)

    def anchor(self, index: int) -> AnchorRecord:
        return self.anchor_table[index]

    def __len__(self) -> int:
        return len(self.anchor_table)

    # -- sampling ----------------------------------------------------------------------

    def __getitem__(self, index: int) -> RawVLASample:
        anchor = self.anchor_table[index]
        slot = self._slots[anchor.relative_path]
        self._loader.set_anchor(slot, anchor.current_frame_index)
        try:
            sample = self._loader[slot]
        finally:
            self._loader.clear_anchor()
        provenance = dict(sample.provenance)
        provenance.update(
            {
                "partition": self.partition,
                "partition_manifest_version": self.manifest.version,
                "partition_manifest_hash": self._manifest_hash,
                "anchor_table_sha256": self._anchor_hash,
                "dataset_index": int(index),
                "anchor": anchor.as_dict(),
                "split_source": "explicit_partition_manifest",
            }
        )
        return dataclasses.replace(sample, split=self.partition, provenance=provenance)


# --------------------------------------------------------------------------------------
# inference input / training targets split
# --------------------------------------------------------------------------------------


def _state_module() -> Any:
    """Import the frozen P0 state module at call time (n02-core owns that file)."""
    try:
        from owac.frameworks.n02_full import state as state_module
    except ImportError as error:  # pragma: no cover - integration boundary
        raise ImportError(
            "owac.frameworks.n02_full.state is not available in this checkout; the frozen "
            "P0 state contract is owned by the n02-core task"
        ) from error
    return state_module


def sample_frames(sample: RawVLASample) -> dict[str, Any]:
    """One shared-contract ``MeasurementFrame`` per camera, built by the frozen loader."""
    return {
        camera_id: build_measurement_frame(sample, camera_id) for camera_id in sample.camera_ids
    }


def sample_context(sample: RawVLASample) -> Any:
    """The current-step context: proprio, language, and *no* fabricated executed action.

    The frozen loader marks ``executed_action`` unavailable for these caches, and RoboTwin's
    ``actions[t]`` reaches the model only as the declared current-state proxy inside
    ``proprio``; it is not promoted to executed history.
    """
    from owac.core.encoder_contract import EmbodimentContext

    description = str(sample.provenance.get("task_description", ""))
    executed = sample.executed_action
    executed_available = bool(executed is not None and sample.executed_action_available)
    return EmbodimentContext(
        proprio=sample.proprio,
        proprio_available=bool(sample.proprio is not None and sample.proprio_available),
        executed_action=executed,
        executed_action_available=executed_available,
        camera_transform=None,
        camera_transform_available=False,
        task_tokens=description or None,
        task_mask=bool(description),
        hypothesis_prior=None,
        hypothesis_prior_available=False,
    )


def sample_to_inference_input(sample: RawVLASample, *, declared_prior: Any = None) -> Any:
    """Build the no-hidden-truth inference input from one loaded window.

    Only RAW measurements, the declared proprio proxy, language and (when a loader genuinely
    records it) executed actions enter this object. Supervision actions and everything derived
    from the future action block stay out; that separation is what a perturbation test checks.
    """
    state = _state_module()
    frames = sample_frames(sample)
    steps = int(sample.observation_indices.size)
    bundle = state.ObservationBundle(
        frames=frames,
        context=sample_context(sample),
        episode_ids=tuple(sample.episode.episode_group_id for _ in range(steps)),
        anchor_ids=tuple(row[-1] for row in sample.frame_ids[0]),
        new_time_mask=sample.new_measurement.copy(),
    )
    return state.InferenceInput(bundle=bundle, declared_prior=declared_prior)


def sample_to_training_targets(
    sample: RawVLASample, *, geometry: Any = None, relations: Any = None
) -> Any:
    """The supervised targets of one window: the masked action block, nothing else."""
    state = _state_module()
    return state.TrainingTargets(
        actions=sample.supervision_actions,
        action_mask=sample.supervision_mask,
        optional_geometry=geometry,
        optional_relations=relations,
    )


def sample_to_inference_and_targets(
    sample: RawVLASample,
    *,
    declared_prior: Any = None,
    geometry: Any = None,
    relations: Any = None,
) -> tuple[Any, Any]:
    """Convenience pair: (InferenceInput, TrainingTargets)."""
    return (
        sample_to_inference_input(sample, declared_prior=declared_prior),
        sample_to_training_targets(sample, geometry=geometry, relations=relations),
    )


# --------------------------------------------------------------------------------------
# real-NPZ audit
# --------------------------------------------------------------------------------------

_QUANTILE_LIMIT = 1 << 20
_QUANTILE_LEVELS = (0.01, 0.5, 0.99)


def _required_keys(dataset_format: str) -> tuple[str, ...]:
    spec = FORMAT_SPECS[dataset_format]
    return (
        *spec.measurement_keys,
        spec.state_source_key,
        "actions",
        "metadata_json",
    )


def _expected_fields(dataset_format: str) -> dict[str, Any]:
    spec = FORMAT_SPECS[dataset_format]
    protocol = P0_PROTOCOL[dataset_format]
    return {
        "measurement_keys": list(spec.measurement_keys),
        "camera_ids": list(spec.camera_ids),
        "conversion": protocol["conversion"],
        "state_source_key": spec.state_source_key,
        "state_source_width": protocol["state_source"]["width"],
        "state_source_expression": protocol["state_source"]["expression"],
        "action_supervision_expression": protocol["action_supervision"]["expression"],
        "loader_default_action_normalisation": spec.default_action_normalisation,
        "formal_statistics": protocol["formal_statistics"],
        "executed_action_window": protocol["executed_action_window"],
        "required_keys": list(_required_keys(dataset_format)),
    }


def _array_summary(array: np.ndarray, *, quantile_limit: int = _QUANTILE_LIMIT) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": [int(value) for value in array.shape],
        "size": int(array.size),
    }
    if array.ndim == 4:
        summary["frames"] = int(array.shape[0])
        summary["resolution_hw"] = [int(array.shape[1]), int(array.shape[2])]
        summary["channels"] = int(array.shape[3])
    elif array.ndim == 3:
        summary["frames"] = int(array.shape[0])
        summary["resolution_hw"] = [int(array.shape[1]), int(array.shape[2])]
        summary["channels"] = 1
    if array.size == 0:
        summary.update(
            {"min": None, "max": None, "finite": True, "quantiles": None, "quantile_kind": "empty"}
        )
        return summary
    if not np.issubdtype(array.dtype, np.number) and array.dtype.kind != "b":
        summary["non_numeric"] = True
        if array.size <= 4:
            summary["values"] = [str(value) for value in array.reshape(-1).tolist()]
        return summary
    summary["min"] = float(array.min())
    summary["max"] = float(array.max())
    summary["finite"] = bool(np.isfinite(array).all())
    flat = np.asarray(array).reshape(-1)
    stride = max(1, int(np.ceil(flat.size / quantile_limit)))
    sampled = flat[::stride].astype(np.float64, copy=False)
    quantiles = np.quantile(sampled, _QUANTILE_LEVELS)
    summary["quantiles"] = {
        "q01": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q99": float(quantiles[2]),
    }
    summary["quantile_kind"] = "exact" if stride == 1 else f"strided_subsample(stride={stride})"
    return summary


def _action_statistics(array: np.ndarray) -> dict[str, Any] | None:
    """Empirical statistics of the file's own action array (never the author's constants)."""
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    values = array.astype(np.float64, copy=False)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.size > 4_000_000:
        stride = max(1, int(np.ceil(values.shape[0] / (4_000_000 / max(1, values.shape[1])))))
        values = values[::stride]
    quantiles = np.quantile(values, _QUANTILE_LEVELS, axis=0)
    return {
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "min": [float(value) for value in values.min(axis=0).tolist()],
        "max": [float(value) for value in values.max(axis=0).tolist()],
        "mean": [float(value) for value in values.mean(axis=0).tolist()],
        "std": [float(value) for value in values.std(axis=0).tolist()],
        "q01": [float(value) for value in quantiles[0].tolist()],
        "q50": [float(value) for value in quantiles[1].tolist()],
        "q99": [float(value) for value in quantiles[2].tolist()],
    }


def _looks_like_timestamp(name: str) -> bool:
    lowered = name.lower()
    return "time" in lowered or "stamp" in lowered


def _declared_metadata_value(metadata: Mapping[str, Any], names: Iterable[str]) -> str | None:
    for name in names:
        if name in metadata:
            value = metadata[name]
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _audit_file(
    path: Path,
    relative: str,
    dataset_format: str,
    *,
    quantile_limit: int = _QUANTILE_LIMIT,
) -> dict[str, Any]:
    spec = FORMAT_SPECS[dataset_format]
    record: dict[str, Any] = {
        "relative_path": relative,
        "sha256": _sha256_file(path),
        "file_bytes": int(path.stat().st_size),
    }
    required = _required_keys(dataset_format)
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        metadata = _read_metadata_json(archive)
        record["keys"] = keys
        record["num_frames"] = (
            int(np.asarray(archive["actions"]).shape[0]) if "actions" in keys else None
        )
        record["missing_required_keys"] = [key for key in required if key not in keys]
        measurements: dict[str, Any] = {}
        for key in spec.measurement_keys:
            if key in keys:
                measurements[key] = {
                    "present": True,
                    **_array_summary(np.asarray(archive[key]), quantile_limit=quantile_limit),
                }
            else:
                measurements[key] = {"present": False}
        record["measurements"] = measurements
        observed_order = [key for key in keys if key in set(spec.measurement_keys)]
        record["file_measurement_key_order"] = observed_order
        record["camera_order_matches_declared"] = observed_order == list(spec.measurement_keys)
        timestamp_fields = sorted(key for key in metadata if _looks_like_timestamp(str(key)))
        record["timestamp"] = {
            "array_keys_found": [key for key in keys if _looks_like_timestamp(key)],
            "metadata_fields": timestamp_fields,
            "declared_unit": _declared_metadata_value(
                metadata, ("timestamp_unit", "time_unit", "dt_unit")
            )
            or UNKNOWN,
        }
        noise_fields = sorted(
            str(key)
            for key in metadata
            if "noise" in str(key).lower() or "variance" in str(key).lower()
        )
        noise_arrays = [key for key in keys if "noise" in key.lower()]
        record["noise"] = {
            "array_keys_found": noise_arrays,
            "metadata_fields": noise_fields,
            "declared": bool(noise_arrays or noise_fields),
        }
        record["language"] = {
            "task_text": _declared_metadata_value(
                metadata, ("task_description", "task_name", "instruction", "language", "task")
            )
            or "",
            "array_keys_found": [
                key
                for key in keys
                if key.lower() in {"instruction", "language", "task", "prompt", "task_description"}
            ],
        }
        proprio: dict[str, Any] = {}
        for key in (spec.state_source_key, "robot_states", "states"):
            if key in keys:
                proprio[key] = _array_summary(
                    np.asarray(archive[key]), quantile_limit=quantile_limit
                )
        record["proprio"] = proprio
        record["state_source_key"] = spec.state_source_key
        record["state_source_present"] = spec.state_source_key in keys
        record["state_source_width"] = (
            int(np.asarray(archive[spec.state_source_key]).shape[-1])
            if spec.state_source_key in keys
            else None
        )
        if "actions" in keys:
            actions = np.asarray(archive["actions"])
            statistics = _action_statistics(actions)
            record["actions"] = {
                "present": True,
                **_array_summary(actions, quantile_limit=quantile_limit),
                "rows": int(actions.shape[0]),
                "columns": int(actions.shape[1]) if actions.ndim > 1 else 1,
                "units": _declared_metadata_value(
                    metadata, ("action_unit", "action_units", "action_scale")
                )
                or UNKNOWN,
                "normalisation_stats": statistics,
                "normalisation_stats_sha256": None
                if statistics is None
                else _sha256_text(_canonical_json(statistics)),
                "normalisation_stats_source": (
                    "empirical statistics of this file's action array (min/max/mean/std/"
                    "q01/q50/q99 per column); this is not the author's normalisation constants"
                ),
            }
        else:
            record["actions"] = {"present": False}
        record["metadata_json_keys"] = sorted(str(key) for key in metadata)
    return record


def _counts(values: Iterable[bool]) -> int:
    return sum(1 for value in values if value)


def _protocol_checks(
    records: Sequence[Mapping[str, Any]], dataset_format: str
) -> list[dict[str, Any]]:
    spec = FORMAT_SPECS[dataset_format]
    protocol = P0_PROTOCOL[dataset_format]
    files = len(records)
    checks: list[dict[str, Any]] = []

    present_counts = {
        key: _counts(key in record["keys"] for record in records) for key in spec.measurement_keys
    }
    all_present = all(count == files for count in present_counts.values())
    any_present = any(count > 0 for count in present_counts.values())
    checks.append(
        {
            "id": "raw_measurement_keys",
            "item": "RAW key names and camera order",
            "expected": {
                "keys": list(protocol["raw_keys"]),
                "camera_ids": list(spec.camera_ids),
                "order_matters": True,
            },
            "observed": {
                "files": files,
                "present_files": present_counts,
                "file_measurement_key_order": sorted(
                    {tuple(record["file_measurement_key_order"]) for record in records}
                )
                if files <= 8
                else None,
            },
            "verdict": "match" if all_present else ("absent" if not any_present else "mismatch"),
            "note": (
                "the frozen format spec declares exactly these keys; a present-but-reordered "
                "cache is reported as a mismatch because camera order is part of the protocol"
            ),
        }
    )

    observed_dtypes = {
        key: sorted(
            {
                str(record["measurements"][key]["dtype"])
                for record in records
                if record["measurements"][key].get("present")
            }
        )
        for key in spec.measurement_keys
    }
    dtype_ok = all(
        values and set(values) <= set(protocol["expected_dtypes"])
        for values in observed_dtypes.values()
    )
    checks.append(
        {
            "id": "measurement_dtype_conversion",
            "item": "storage dtype and the declared conversion rule",
            "expected": {
                "dtypes": list(protocol["expected_dtypes"]),
                "rule": protocol["conversion"],
            },
            "observed": {"dtypes": observed_dtypes},
            "verdict": "match"
            if dtype_ok and all_present
            else ("absent" if not any_present else "mismatch"),
            "note": (
                "the frozen loader keys the conversion off the stored dtype: uint8 -> float32/255 "
                "and floating -> unchanged (rawvla_sequence.RawVLAWindowDataset._calibrate)"
            ),
        }
    )

    widths = sorted(
        {
            int(record["state_source_width"])
            for record in records
            if record["state_source_width"] is not None
        }
    )
    state_present = _counts(bool(record["state_source_present"]) for record in records)
    expected_width = int(protocol["state_source"]["width"])
    checks.append(
        {
            "id": "current_proprio_source",
            "item": "current proprio source and width",
            "expected": {
                "key": protocol["state_source"]["key"],
                "expression": protocol["state_source"]["expression"],
                "width": expected_width,
                "note": protocol["state_source"]["note"],
            },
            "observed": {
                "files_with_key": state_present,
                "files": files,
                "widths": widths,
                "companion_keys": sorted(
                    {
                        key
                        for record in records
                        for key in record["proprio"]
                        if key != protocol["state_source"]["key"]
                    }
                ),
            },
            "verdict": "match"
            if state_present == files and widths == [expected_width]
            else ("absent" if state_present == 0 else "mismatch"),
            "note": (
                "a companion 16-dim robot_states array may exist on RoboTwin; it is recorded "
                "but is not the joint state"
            ),
        }
    )

    rows = [record["num_frames"] for record in records]
    horizon = int(protocol["action_supervision"]["horizon"])
    offset = int(protocol["action_supervision"]["offset"])
    enough_rows = all(row is not None and row >= offset + horizon for row in rows)
    loader_default = {
        "action_offset": int(spec.default_action_offset),
        "action_horizon": int(spec.default_action_horizon),
    }
    loader_matches = loader_default == {"action_offset": offset, "action_horizon": horizon}
    actions_present = _counts("actions" in record["keys"] for record in records)
    checks.append(
        {
            "id": "action_supervision_window",
            "item": "action supervision window",
            "expected": {
                "expression": protocol["action_supervision"]["expression"],
                "offset": offset,
                "horizon": horizon,
            },
            "observed": {
                "loader_default": loader_default,
                "files_with_actions": actions_present,
                "files": files,
                "num_frames": rows if files <= 8 else None,
                "files_with_horizon_plus_offset_rows": _counts(
                    row is not None and row >= offset + horizon for row in rows
                ),
            },
            "verdict": "match"
            if actions_present == files and loader_matches and enough_rows
            else ("absent" if actions_present == 0 else "mismatch"),
            "note": (
                "the window rule is a declared protocol; the audit can only confirm that the "
                "loader default agrees and that each file has enough action rows"
            ),
        }
    )

    loader_mode = spec.default_action_normalisation
    mode_expected = loader_mode == protocol["expected_loader_default"]
    formal_ok = bool(protocol["loader_mode_is_formal"]) and mode_expected
    checks.append(
        {
            "id": "formal_normalisation_statistics",
            "item": "formal pi0.5 normalisation statistics",
            "expected": {
                "formal_statistics": protocol["formal_statistics"],
                "expected_loader_default": protocol["expected_loader_default"],
            },
            "observed": {"loader_default": loader_mode},
            "verdict": "match" if formal_ok else "mismatch",
            "note": protocol.get(
                "loader_default_role",
                "the loader default is the declared formal statistics for this platform",
            ),
        }
    )

    checks.append(
        {
            "id": "executed_action_window",
            "item": "actually executed actions per inference",
            "expected": {"actions_executed": int(protocol["executed_action_window"])},
            "observed": {"npz_field": None},
            "verdict": "declared_only",
            "note": protocol["executed_action_note"],
        }
    )

    checks.append(
        {
            "id": "camera_order",
            "item": "camera key order in each file",
            "expected": {"order": list(spec.measurement_keys)},
            "observed": {
                "files_matching": _counts(
                    bool(record["camera_order_matches_declared"]) for record in records
                ),
                "files": files,
            },
            "verdict": "match"
            if all_present and all(record["camera_order_matches_declared"] for record in records)
            else ("absent" if not any_present else "mismatch"),
            "note": "streams are synchronised; the order identifies which camera is which",
        }
    )

    timestamp_files = _counts(bool(record["timestamp"]["array_keys_found"]) for record in records)
    checks.append(
        {
            "id": "measurement_timestamp",
            "item": "timestamp key and its unit",
            "expected": {"timestamp_key": True, "unit_declared": True},
            "observed": {
                "files_with_timestamp_key": timestamp_files,
                "files": files,
                "declared_units": sorted(
                    {str(record["timestamp"]["declared_unit"]) for record in records}
                ),
            },
            "verdict": "match" if timestamp_files == files else "not_declared",
            "note": (
                "a frame-index-only cache cannot support exposure-time reasoning; the loader "
                "reports time_unit='frame' and timestamp=None in that case"
            ),
        }
    )

    noise_files = _counts(bool(record["noise"]["declared"]) for record in records)
    checks.append(
        {
            "id": "noise_availability",
            "item": "declared noise / variance source",
            "expected": {"noise_arrays_or_metadata": True},
            "observed": {
                "files_with_noise_declaration": noise_files,
                "files": files,
                "arrays": sorted(
                    {key for record in records for key in record["noise"]["array_keys_found"]}
                ),
            },
            "verdict": "match" if noise_files == files else "not_declared",
            "note": (
                "without a declared noise model no variance may be fabricated; an estimated "
                "scale stays an estimate and any energy built from it is not a posterior"
            ),
        }
    )

    language_files = _counts(bool(record["language"]["task_text"]) for record in records)
    checks.append(
        {
            "id": "language_keys",
            "item": "task instruction / language text",
            "expected": {"task_text": True},
            "observed": {"files_with_task_text": language_files, "files": files},
            "verdict": "match" if language_files == files else "not_declared",
            "note": "language is declared context, never a hidden-truth channel",
        }
    )
    checks.append(
        {
            "id": "action_units",
            "item": "action units and normalisation statistics fingerprint",
            "expected": {"units": "declared cache units or UNKNOWN", "stats_hash": True},
            "observed": {
                "units": sorted(
                    {str(record["actions"].get("units", UNKNOWN)) for record in records}
                ),
                "stats_hashes": sorted(
                    {
                        str(record["actions"]["normalisation_stats_sha256"])
                        for record in records
                        if record["actions"].get("normalisation_stats_sha256")
                    }
                ),
            },
            "verdict": "match"
            if all(record["actions"].get("present") for record in records)
            else "absent",
            "note": (
                "the hash covers the empirical per-column statistics of the file's own action "
                "array, not the author's normalisation constants"
            ),
        }
    )
    return checks


def load_formal_normalisation(dataset_format: str, path: str | Path):
    """Load explicit checkpoint constants; never infer them from audited NPZ values."""
    from owac.raw.rawvla_sequence import NormalisationStats

    if dataset_format not in ("libero", "robotwin"):
        raise ValueError("formal pi0.5 statistics require libero or robotwin")
    stats = NormalisationStats.from_json(
        path,
        mode="quantile" if dataset_format == "libero" else "zscore",
        epsilon=1e-8,
        epsilon_mode="add",
    )
    fields = ("q01", "q99") if dataset_format == "libero" else ("mean", "std")
    for kind, width in (
        ("action", 7 if dataset_format == "libero" else 14),
        ("state", 8 if dataset_format == "libero" else 14),
    ):
        values = getattr(stats, kind)
        if values is None:
            raise ValueError(f"checkpoint has no {kind} statistics")
        for stat_field in fields:
            value = np.asarray(values[stat_field])
            if value.size < width or not np.isfinite(value).all():
                raise ValueError(f"invalid checkpoint {kind}.{stat_field}")
    return stats


def audit_dataset_root(
    cache_root: str | Path,
    dataset_format: str,
    *,
    max_files: int | None = None,
    evidence_kind: str = "unspecified_root",
    stats_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the NPZ files under ``cache_root`` against the P0 two-platform protocol.

    A missing or empty root returns ``status="not_run"`` with a reason, the checked path and the
    expected fields; no numbers are invented. When files are found, every reported dtype, shape,
    range, quantile, resolution, key order, timestamp/noise/language/proprio finding and the
    action-statistics hash comes from reading those files.

    ``evidence_kind`` is the caller's declaration of what the root is. Only
    ``evidence_kind="real_cache"`` sets ``real_cache_evidence=True``; a fixture root must be
    declared otherwise and then never counts as evidence about the deployed RawVLA cache.
    """
    if dataset_format not in FORMAT_SPECS:
        raise ValueError(f"unknown dataset_format: {dataset_format!r}")
    spec = FORMAT_SPECS[dataset_format]
    root = Path(cache_root)
    report: dict[str, Any] = {
        "dataset_format": dataset_format,
        "checked_path": str(root),
        "evidence_kind": str(evidence_kind),
        "real_cache_evidence": str(evidence_kind) == REAL_CACHE_EVIDENCE_KIND,
        "expected_fields": _expected_fields(dataset_format),
        "search_pattern": None,
    }
    if not root.exists():
        report.update(
            {
                "status": "not_run",
                "reason": f"cache_root_missing: {root} does not exist",
                "files_checked": 0,
                "files_total": 0,
                "limits": [],
            }
        )
        return report
    paths, pattern = discover_npz_files(root, dataset_format)
    report["search_pattern"] = pattern
    if not paths:
        report.update(
            {
                "status": "not_run",
                "reason": (
                    f"no_npz_found: no *.npz matched {root} for dataset_format={dataset_format!r}"
                ),
                "files_checked": 0,
                "files_total": 0,
                "limits": [],
            }
        )
        return report
    selected = list(paths)
    truncated = False
    if max_files is not None:
        if max_files < 1:
            raise ValueError("max_files must be >= 1 when set")
        truncated = len(selected) > max_files
        selected = selected[:max_files]
    records = []
    for path in selected:
        relative = str(Path(path).relative_to(root))
        records.append(_audit_file(Path(path), relative, dataset_format))
    checks = _protocol_checks(records, dataset_format)
    stats = load_formal_normalisation(dataset_format, stats_path) if stats_path else None
    for check in checks:
        if check["id"] == "formal_normalisation_statistics":
            check["verdict"] = "match" if stats is not None else "not_run"
            check["observed"] = {
                "loader_default": spec.default_action_normalisation,
                "checkpoint_mode": None if stats is None else stats.mode,
                "stats_sha256": None if stats is None else stats.sha256,
                "epsilon": None if stats is None else stats.epsilon,
                "epsilon_mode": None if stats is None else stats.epsilon_mode,
            }
            check["note"] = (
                "explicit checkpoint statistics verified"
                if stats
                else "checkpoint statistics absent; loader default is not verification"
            )

    verdict_counts: dict[str, int] = {}
    for check in checks:
        verdict_counts[check["verdict"]] = verdict_counts.get(check["verdict"], 0) + 1
    problems: list[str] = []
    for record in records:
        if record["missing_required_keys"]:
            problems.append(
                f"{record['relative_path']}: missing keys {record['missing_required_keys']}"
            )
        for key, summary in record["measurements"].items():
            if summary.get("present") and not summary.get("finite", True):
                problems.append(f"{record['relative_path']}: {key!r} has non-finite values")
    limits = [
        "each file is read fully to measure min/max; quantiles use an evenly strided "
        f"subsample of at most {_QUANTILE_LIMIT} elements and report quantile_kind",
        "the executed-action window and the author's normalisation constants are not stored in "
        "the NPZ, so those checks are declared-only",
        f"max_files={max_files}; {len(selected)} of {len(paths)} discovered files were read"
        if truncated
        else f"all {len(paths)} discovered files were read",
    ]
    return {
        **report,
        "status": "fail"
        if problems or verdict_counts.get("mismatch", 0)
        else ("incomplete" if stats is None else "ok"),
        "files_total": len(paths),
        "files_checked": len(records),
        "truncated": truncated,
        "files": records,
        "protocol_checks": checks,
        "checks_summary": verdict_counts,
        "problems": problems,
        "limits": limits,
        "notes": [
            f"evidence_kind={evidence_kind!r} was declared by the caller; this audit reports "
            "measured facts and never infers that a root is the deployed cache",
            f"FORMAT_SPECS[{dataset_format!r}] default action normalisation is "
            f"{spec.default_action_normalisation!r}",
        ],
    }

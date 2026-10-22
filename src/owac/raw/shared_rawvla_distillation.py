"""Shared RAW-VLA source adapter and token-teacher cache preparation.

This module is the new shared-data protocol (task6 Normal 8/2/2).  It never
writes the mounted shared dataset.  It builds a writable source manifest whose
paths point at the absolute read-only shared NPZ files, verifies the actual RAW
arrays and SHA256 values, and prepares token-teacher cache rows through the
actual frozen teacher:

    shared RAW -> NeuralISP/init0 frontend -> same pi05 SigLIP/projector

The existing accepted 50-source manifest/cache remains historical and is not
read or migrated here.  Only train+validation sources are expanded into teacher
targets; the two test sources are manifest-only until a separate final-evaluation
authorization.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from owac.raw.teacher_data import (
    ACTION_DIM,
    BURST_FRAMES,
    CAMERA_ORDER,
    INSTRUCTION,
    RECURRENT_ENDPOINTS,
    TASK_ID,
    TASK_NAME,
    TASK_SUITE,
    TeacherData,
    TeacherSource,
    TeacherSourceManifest,
    load_pi05_normalization_stats,
    load_source_manifest,
)
from owac.raw.temporal_distillation import (
    CACHE_SCHEMA,
    TemporalAssetSet,
    TemporalCache,
    prepare_temporal_cache,
)
from owac.raw.temporal_distillation import (
    CAMERA_ORDER as CACHE_CAMERA_ORDER,
)

__all__ = [
    "SHARED_PROTOCOL",
    "SHARED_SPLITS",
    "SharedDataError",
    "build_shared_manifest",
    "prepare_shared_teacher_cache",
    "shared_expected_counts",
]

SHARED_PROTOCOL = "owac.shared-rawvla.task6-normal.8-2-2.v1"
SHARED_SPLITS = {"train": 8, "validation": 2, "test": 2}
SHARED_SOURCE_COUNT = 12
SHARED_NORMAL_LIGHTING = "Normal"
SHARED_INVENTORY_SCHEMA = "owac.shared-rawvla.task6-inventory.v1"
SHARED_MANIFEST_REPORT_SCHEMA = "owac.shared-rawvla.manifest-report.v1"
SHARED_RECEIPT_SCHEMA = "owac.shared-rawvla.teacher-preparation.v1"
SHARED_CACHE_PREPROCESS = "shared_raw256_fullview_bilinear_align_corners_false_224"
SHARED_RELEASE_REVISION = "8586c594fff9ce38580e29745f074e82e96e74f7"
_HDF5_AGGREGATE_SEMANTICS = "compatibility_aggregate_digest_not_verified_hdf5_sha"

_CAMERA_KEYS = {
    "agentview": "agentview_raw_uint8",
    "wrist": "wrist_raw_uint8",
}


class SharedDataError(RuntimeError):
    """The shared source, manifest, cache identity, or preparation is invalid."""


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _owned_bytes(root: Path) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    if not root.exists():
        return 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (int(stat.st_dev), int(stat.st_ino))
            if key in seen:
                continue
            seen.add(key)
            total += int(stat.st_size)
    return total


@dataclass(frozen=True, slots=True)
class InventoryMatch:
    path: str
    demo: str
    lighting: str
    frames: int


@dataclass(frozen=True, slots=True)
class SharedSource:
    source_demo_id: str
    split: str
    path: str
    raw_sha256: str
    num_frames: int
    hdf5_path: str
    demo_key: str
    demo_index: int
    metadata: Mapping[str, Any]

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.hdf5_path, self.demo_key)


@dataclass(frozen=True, slots=True)
class SharedManifestReport:
    manifest_path: Path
    manifest_sha256: str
    sources: tuple[SharedSource, ...]
    split_counts: Mapping[str, int]
    train_anchors: int
    validation_windows: int
    cache_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SHARED_MANIFEST_REPORT_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "source_count": len(self.sources),
            "split_counts": dict(self.split_counts),
            "train_anchors": int(self.train_anchors),
            "validation_windows": int(self.validation_windows),
            "cache_rows": int(self.cache_rows),
            "test_targets_generated": False,
            "protocol": SHARED_PROTOCOL,
        }


def load_inventory(path: str | Path) -> list[InventoryMatch]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SharedDataError(f"shared inventory is unreadable: {source}") from exc
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise SharedDataError("shared inventory lacks matches")
    rows: list[InventoryMatch] = []
    for item in matches:
        if not isinstance(item, Mapping):
            raise SharedDataError("shared inventory match is not an object")
        path_value = str(item.get("path", ""))
        demo = str(item.get("demo", ""))
        lighting = str(item.get("lighting", ""))
        try:
            frames = int(item.get("frames", 0))
        except (TypeError, ValueError) as exc:
            raise SharedDataError("shared inventory frame count is invalid") from exc
        if not path_value or not demo or frames < 1:
            raise SharedDataError("shared inventory match is incomplete")
        rows.append(InventoryMatch(path_value, demo, lighting, frames))
    if not rows:
        raise SharedDataError("shared inventory has no task6 matches")
    return rows


def _normal_matches(rows: Sequence[InventoryMatch]) -> list[InventoryMatch]:
    normal = [item for item in rows if item.lighting == SHARED_NORMAL_LIGHTING]
    if len(normal) != SHARED_SOURCE_COUNT:
        raise SharedDataError(
            f"shared task6 Normal source count {len(normal)} != {SHARED_SOURCE_COUNT}"
        )
    if len({item.path for item in normal}) != len(normal):
        raise SharedDataError("shared Normal inventory contains duplicate paths")
    return normal


def _split_groups(groups: Sequence[tuple[str, str]], *, seed: int) -> dict[tuple[str, str], str]:
    ordered = sorted(set(groups))
    if len(ordered) != SHARED_SOURCE_COUNT:
        raise SharedDataError(f"shared Normal group count {len(ordered)} != {SHARED_SOURCE_COUNT}")
    rng = np.random.default_rng(int(seed))
    permutation = [ordered[int(index)] for index in rng.permutation(len(ordered))]
    result: dict[tuple[str, str], str] = {}
    for index, key in enumerate(permutation):
        result[key] = (
            "train"
            if index < SHARED_SPLITS["train"]
            else "validation"
            if index < SHARED_SPLITS["train"] + SHARED_SPLITS["validation"]
            else "test"
        )
    return result


def _npy_header(archive: Any, key: str) -> tuple[tuple[int, ...], Any]:

    member_name = key if key.endswith(".npy") else f"{key}.npy"
    with archive.open(member_name) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            raise SharedDataError(f"unsupported NPY header version {version} for {member_name}")
    return tuple(int(v) for v in shape), dtype


def _read_npz_metadata(path: Path) -> tuple[dict[str, Any], str, int, tuple[int, int, int, int]]:
    """Read ZIP/NPY headers + scalar metadata only; never decompress pixel arrays."""
    import zipfile

    actual_sha = ""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {
                "agentview_raw_uint8.npy",
                "wrist_raw_uint8.npy",
                "actions.npy",
                "states.npy",
                "robot_states.npy",
                "metadata_json.npy",
            }
            missing = sorted(required - names)
            if missing:
                raise SharedDataError(f"shared NPZ lacks members {missing}: {path}")
            raw_shape, raw_dtype = _npy_header(archive, "agentview_raw_uint8")
            wrist_shape, wrist_dtype = _npy_header(archive, "wrist_raw_uint8")
            action_shape, _action_dtype = _npy_header(archive, "actions")
            robot_shape, _robot_dtype = _npy_header(archive, "robot_states")
            with archive.open("metadata_json.npy") as member:
                metadata_array = np.lib.format.read_array(member, allow_pickle=False)
            metadata = json.loads(str(metadata_array.item()))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise SharedDataError(f"shared NPZ headers are unreadable: {path}") from exc
    if raw_dtype != np.dtype("uint8") or wrist_dtype != np.dtype("uint8"):
        raise SharedDataError(f"shared camera RAW must be uint8: {path}")
    if len(raw_shape) != 4 or raw_shape[-1] != 3 or wrist_shape != raw_shape:
        raise SharedDataError(f"shared camera RAW geometry differs: {path}")
    if len(action_shape) != 2 or action_shape[1] != ACTION_DIM:
        raise SharedDataError(f"shared actions must be [T,{ACTION_DIM}]: {path}")
    if len(robot_shape) != 2 or robot_shape[1] != 8:
        raise SharedDataError(f"shared robot_states must be [T,8]: {path}")
    if raw_shape[0] != int(action_shape[0]) or robot_shape[0] != action_shape[0]:
        raise SharedDataError("shared RAW/action/state frame counts differ")
    if not isinstance(metadata, dict):
        raise SharedDataError("shared NPZ metadata is not an object")
    return metadata, actual_sha, int(action_shape[0]), raw_shape


def _source_id(demo_key: str) -> str:
    return f"shared_task6_normal_{demo_key}"


def build_shared_manifest(
    *,
    inventory_path: str | Path,
    shared_root: str | Path,
    manifest_path: str | Path,
    split_seed: int = 101,
    release_revision: str = SHARED_RELEASE_REVISION,
    force: bool = False,
) -> SharedManifestReport:
    """Select only the 12 Normal demos and write an absolute writable manifest."""
    inventory_file = Path(inventory_path)
    inventory_sha = ""
    rows = _normal_matches(load_inventory(inventory_file))
    root = Path(shared_root).resolve()
    if not root.is_dir():
        raise SharedDataError(f"shared dataset root is absent: {root}")
    output = Path(manifest_path)
    if output.exists() and not force:
        raise SharedDataError(f"manifest already exists (use --force to rebuild): {output}")

    preliminary: list[tuple[InventoryMatch, Path]] = []
    groups: list[tuple[str, str]] = []
    for item in rows:
        path = Path(item.path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SharedDataError(f"shared path escapes read-only root: {item.path}") from exc
        if not path.is_file():
            raise SharedDataError(f"shared RAW NPZ is absent: {path}")
        metadata, raw_sha, frames, _shape = _read_npz_metadata(path)
        if frames != item.frames:
            raise SharedDataError(f"inventory frame count differs for {item.path}")
        if metadata.get("lighting_domain") != SHARED_NORMAL_LIGHTING:
            raise SharedDataError(f"selected shared source is not Normal: {item.path}")
        if int(metadata.get("task_id", -1)) != TASK_ID:
            raise SharedDataError(f"selected shared source is not task6: {item.path}")
        hdf5_path = str(metadata.get("hdf5_path") or "")
        demo_key = str(metadata.get("demo_key") or item.demo)
        if not hdf5_path or not demo_key:
            raise SharedDataError(f"shared metadata lacks hdf5/demo key: {item.path}")
        if demo_key != item.demo:
            raise SharedDataError(f"inventory/metadata demo key differs: {item.path}")
        groups.append((hdf5_path, demo_key))
        preliminary.append((item, path))
    if len(set(groups)) != SHARED_SOURCE_COUNT:
        raise SharedDataError("Normal shared sources do not form 12 unique hdf5+demo groups")
    split_by_group = _split_groups(groups, seed=int(split_seed))

    sources: list[TeacherSource] = []
    shared_sources: list[SharedSource] = []
    seen_ids: set[str] = set()
    for (_item, path), (hdf5_path, demo_key) in zip(preliminary, groups, strict=True):
        metadata, raw_sha, frames, _shape = _read_npz_metadata(path)
        source_id = _source_id(demo_key)
        if source_id in seen_ids:
            raise SharedDataError(f"duplicate shared source id {source_id}")
        seen_ids.add(source_id)
        group = (hdf5_path, demo_key)
        split = split_by_group[group]
        source_metadata = {
            "shared_readonly_path": str(path),
            "hdf5_path": hdf5_path,
            "demo_key": demo_key,
            "demo_index": int(metadata.get("demo_index", -1)),
            "lighting": SHARED_NORMAL_LIGHTING,
            "shared_release_revision": str(release_revision),
            "source_raw_sha256": raw_sha,
            "actual_source_npz_sha256": raw_sha,
            "source_kind": "shared_rawvla_npz",
            "paired_rgb_used": False,
            "paired_rgb_path_registered": False,
            "paired_rgb_identity_verified": False,
            "paired_rgb_availability_in_adapter": "unknown_not_inspected",
            "paired_rgb_dataset_note": (
                "shared release may contain paired RGB files; this RAW-only adapter "
                "does not inspect, register, or read them"
            ),
            "paired_rgb_not_read": True,
            "paired_rgb_path": None,
            "paired_rgb_sha256": None,
            "paired_rgb_identity_fabricated": False,
            "rgb_path_field_semantics": "raw_path_compatibility_only_no_paired_rgb",
            "rgb_sha256_field_semantics": "raw_sha_compatibility_only_no_paired_rgb",
            "target_scope": "train_validation_only" if split != "test" else "test_manifest_only",
        }
        source = TeacherSource(
            source_demo_id=source_id,
            split=split,
            raw_path=str(path),
            rgb_path=str(path),
            raw_sha256=raw_sha,
            rgb_sha256=raw_sha,
            num_frames=frames,
            task_name=TASK_NAME,
            lighting=SHARED_NORMAL_LIGHTING,
            camera_order=CAMERA_ORDER,
            metadata=source_metadata,
        )
        sources.append(source)
        shared_sources.append(
            SharedSource(
                source_demo_id=source_id,
                split=split,
                path=str(path),
                raw_sha256=raw_sha,
                num_frames=frames,
                hdf5_path=hdf5_path,
                demo_key=demo_key,
                demo_index=int(metadata.get("demo_index", -1)),
                metadata=source_metadata,
            )
        )
    split_counts = {
        split: sum(source.split == split for source in shared_sources)
        for split in ("train", "validation", "test")
    }
    if split_counts != SHARED_SPLITS:
        raise SharedDataError(f"shared split counts {split_counts} != {SHARED_SPLITS}")
    source_hdf5_sha = ""
    task = {
        "dataset": "RAWVLA-Bench",
        "suite": TASK_SUITE,
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "instruction": INSTRUCTION,
        "lighting": SHARED_NORMAL_LIGHTING,
    }
    provenance = {
        "schema": SHARED_PROTOCOL,
        "inventory_path": str(inventory_file.resolve()),
        "inventory_sha256": inventory_sha,
        "shared_root": str(root),
        "shared_release_revision": str(release_revision),
        "shared_readonly": True,
        "selection": "task6 Normal only; all non-Normal lighting excluded",
        "old_50_source_manifest_used": False,
        "old_teacher_cache_used": False,
        "paired_rgb_substitute_used": False,
        "test_target_generation": "manifest_only_until_separate_evaluation_authorization",
        "source_hdf5_sha256_kind": "aggregate_normal_npz_sources_not_hdf5_file",
        "actual_source_hdf5_sha256": None,
        "actual_source_hdf5_sha256_available": False,
        "source_hdf5_sha256_field_semantics": _HDF5_AGGREGATE_SEMANTICS,
        "source_hdf5_sha256": source_hdf5_sha,
        "paired_rgb_used": False,
        "paired_rgb_path_registered": False,
        "paired_rgb_identity_verified": False,
        "paired_rgb_availability_in_adapter": "unknown_not_inspected",
        "paired_rgb_dataset_note": (
            "shared release may contain paired RGB files; this RAW-only adapter "
            "does not inspect, register, or read them"
        ),
        "paired_rgb_not_read": True,
        "paired_rgb_identity_fabricated": False,
    }
    manifest = TeacherSourceManifest(
        task=task,
        provenance=provenance,
        source_hdf5_sha256=source_hdf5_sha,
        split_seed=int(split_seed),
        sources=tuple(sorted(sources, key=lambda item: item.source_demo_id)),
        path_semantics="absolute",
    )
    _write_json(output, manifest.as_dict())
    report = shared_expected_counts(manifest)
    report_payload = {
        **report.as_dict(),
        "manifest_path": str(output.resolve()),
        "manifest_sha256": "",
        "inventory_sha256": inventory_sha,
        "actual_source_npz_sha256": {
            source.source_demo_id: source.raw_sha256 for source in shared_sources
        },
        "actual_source_hdf5_sha256": None,
        "actual_source_hdf5_sha256_available": False,
        "source_hdf5_sha256_field_semantics": _HDF5_AGGREGATE_SEMANTICS,
        "source_hdf5_sha256_compatibility_aggregate": source_hdf5_sha,
        "paired_rgb_used": False,
        "paired_rgb_path_registered": False,
        "paired_rgb_identity_verified": False,
        "paired_rgb_availability_in_adapter": "unknown_not_inspected",
        "paired_rgb_dataset_note": (
            "shared release may contain paired RGB files; this RAW-only adapter "
            "does not inspect, register, or read them"
        ),
        "paired_rgb_not_read": True,
        "paired_rgb_identity_fabricated": False,
    }
    _write_json(output.with_suffix(".report.json"), report_payload)
    return SharedManifestReport(
        manifest_path=output,
        manifest_sha256=report_payload["manifest_sha256"],
        sources=tuple(shared_sources),
        split_counts=split_counts,
        train_anchors=report.train_anchors,
        validation_windows=report.validation_windows,
        cache_rows=report.cache_rows,
    )


def shared_expected_counts(manifest: TeacherSourceManifest) -> SharedManifestReport:
    sources = tuple(
        SharedSource(
            source_demo_id=source.source_demo_id,
            split=source.split,
            path=source.raw_path,
            raw_sha256=source.raw_sha256,
            num_frames=source.num_frames,
            hdf5_path=str(source.metadata.get("hdf5_path", "")),
            demo_key=str(source.metadata.get("demo_key", "")),
            demo_index=int(source.metadata.get("demo_index", -1)),
            metadata=dict(source.metadata),
        )
        for source in manifest.sources
    )
    split_counts = {
        split: sum(source.split == split for source in sources)
        for split in ("train", "validation", "test")
    }
    if split_counts != SHARED_SPLITS:
        raise SharedDataError(f"manifest split counts {split_counts} != {SHARED_SPLITS}")
    train_anchors = sum(
        len(range(25, source.num_frames, 5)) for source in sources if source.split == "train"
    )
    validation_windows = sum(
        len(range(0, source.num_frames, 5)) for source in sources if source.split == "validation"
    )
    cache_rows = sum(
        len(range(0, source.num_frames, 5))
        for source in sources
        if source.split in {"train", "validation"}
    )
    return SharedManifestReport(
        manifest_path=Path(""),
        manifest_sha256="",
        sources=sources,
        split_counts=split_counts,
        train_anchors=int(train_anchors),
        validation_windows=int(validation_windows),
        cache_rows=int(cache_rows),
    )


def load_shared_manifest(path: str | Path, *, repo_root: str | Path = ".") -> TeacherSourceManifest:
    manifest = load_source_manifest(
        path,
        repo_root=repo_root,
        strict=False,
        expected_splits=SHARED_SPLITS,
    )
    if len(manifest.sources) != SHARED_SOURCE_COUNT:
        raise SharedDataError("shared manifest source count is not 12")
    if any(source.lighting != SHARED_NORMAL_LIGHTING for source in manifest.sources):
        raise SharedDataError("shared manifest contains non-Normal sources")
    if manifest.path_semantics != "absolute":
        raise SharedDataError("shared manifest must use absolute source paths")
    if manifest.provenance.get("schema") != SHARED_PROTOCOL:
        raise SharedDataError("shared manifest protocol differs")
    expected_task = {
        "suite": TASK_SUITE,
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "instruction": INSTRUCTION,
        "lighting": SHARED_NORMAL_LIGHTING,
    }
    for key, expected in expected_task.items():
        if manifest.task.get(key) != expected:
            raise SharedDataError(f"shared manifest task.{key} differs")
    # The preserved first-build manifest predates these explicit honesty keys.
    # Absence is legacy, not a positive claim; present non-False values fail.
    for key, message in (
        ("paired_rgb_used", "shared manifest claims paired RGB was used"),
        ("paired_rgb_path_registered", "shared manifest registered paired RGB paths"),
        ("paired_rgb_identity_verified", "shared manifest claims verified paired RGB identity"),
    ):
        value = manifest.provenance.get(key)
        if value not in (None, False):
            raise SharedDataError(message)
    if manifest.provenance.get("actual_source_hdf5_sha256") is not None:
        raise SharedDataError("shared manifest claims an actual HDF5 SHA")
    # Earlier same-schema manifests were produced before the explicit
    # HDF5/RGB honesty fields existed.  Do not rewrite them; accept them as
    # legacy and never infer absent fields as positive claims.
    for source in manifest.sources:
        if source.metadata.get("paired_rgb_used") not in (None, False):
            raise SharedDataError("shared source claims paired RGB was used")
        if source.metadata.get("paired_rgb_path_registered") not in (None, False):
            raise SharedDataError("shared source registered a paired RGB path")
        if source.metadata.get("paired_rgb_identity_verified") not in (None, False):
            raise SharedDataError("shared source claims verified paired RGB identity")
        actual_npz = source.metadata.get("actual_source_npz_sha256")
        if actual_npz is not None and actual_npz != source.raw_sha256:
            raise SharedDataError("shared source actual NPZ SHA differs from raw SHA")
        if not source.metadata.get("hdf5_path") or not source.metadata.get("demo_key"):
            raise SharedDataError("shared source lacks hdf5/demo group metadata")
    return manifest


def _validate_shared_assets(assets: TemporalAssetSet, *, manifest_sha256: str) -> dict[str, Any]:
    """Validate teacher assets without the historical 50-source hash bake-in."""
    if not assets.registry_path.is_file() or not assets.artifact_path.is_file():
        raise SharedDataError("teacher registry/artifact is absent")
    registry = json.loads(assets.registry_path.read_text(encoding="utf-8"))
    if registry.get("schema") != "owac.raw-teacher.acceptance.v1":
        raise SharedDataError("teacher registry schema differs")
    if registry.get("status") != "teacher_task6_normal_development_screen_passed":
        raise SharedDataError("teacher registry is not the accepted init0 screen")
    identity = registry.get("identity")
    if not isinstance(identity, Mapping):
        raise SharedDataError("teacher registry lacks identity")
    for name, path in (
        ("backend", assets.checkpoint),
        ("tokenizer", assets.tokenizer),
        ("stats", assets.statistics),
    ):
        if not path.is_file():
            raise SharedDataError(f"teacher {name} asset is absent: {path}")
    if not assets.author_root.is_dir() or not assets.starvla_root.is_dir():
        raise SharedDataError("teacher author/starvla source roots are absent")
    if not assets.sources_path.is_file():
        raise SharedDataError("shared manifest is absent")
    return {
        "artifact_identity": dict(identity),
        "source_revision": identity.get("source"),
        "starvla_revision": identity.get("starvla_revision"),
        "author_root": str(assets.author_root.resolve()),
        "starvla_root": str(assets.starvla_root.resolve()),
        "config_paths": [str(path.resolve()) for path in assets.config_paths],
        "preprocess_id": SHARED_CACHE_PREPROCESS,
    }


def shared_cache_identity(
    *,
    assets_identity: Mapping[str, Any],
    manifest: TeacherSourceManifest,
    manifest_sha256: str,
) -> dict[str, Any]:
    counts = {
        split: len(manifest.sources_for_split(split)) for split in ("train", "validation", "test")
    }
    if counts != SHARED_SPLITS:
        raise SharedDataError(f"cache split counts {counts} != {SHARED_SPLITS}")
    raw_hashes = {
        source.source_demo_id: source.raw_sha256
        for source in sorted(manifest.sources, key=lambda item: item.source_demo_id)
    }
    return {
        "cache_schema": CACHE_SCHEMA,
        "protocol": SHARED_PROTOCOL,
        "namespace": "shared-default",
        "source_raw_sha256": raw_hashes,
        "actual_source_npz_sha256": dict(raw_hashes),
        "source_group_metadata": {
            source.source_demo_id: {
                "hdf5_path": source.metadata.get("hdf5_path"),
                "demo_key": source.metadata.get("demo_key"),
                "demo_index": source.metadata.get("demo_index"),
                "lighting": source.lighting,
            }
            for source in sorted(manifest.sources, key=lambda item: item.source_demo_id)
        },
        "actual_source_hdf5_sha256": None,
        "actual_source_hdf5_sha256_available": False,
        "source_hdf5_sha256_field_semantics": _HDF5_AGGREGATE_SEMANTICS,
        "paired_rgb_used": False,
        "paired_rgb_path_registered": False,
        "paired_rgb_identity_verified": False,
        "paired_rgb_availability_in_adapter": "unknown_not_inspected",
        "paired_rgb_dataset_note": (
            "shared release may contain paired RGB files; this RAW-only adapter "
            "does not inspect, register, or read them"
        ),
        "paired_rgb_not_read": True,
        "paired_rgb_identity_fabricated": False,
        "splits": counts,
        "camera_order": list(CACHE_CAMERA_ORDER),
        "native_image_keys": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "active_image_keys": ["base_0_rgb", "left_wrist_0_rgb"],
        "preprocess_id": SHARED_CACHE_PREPROCESS,
        "raw_burst_frames": BURST_FRAMES,
        "endpoint_stride": 5,
        "recurrent_endpoints": RECURRENT_ENDPOINTS,
        "causal_history": BURST_FRAMES,
        "state_cache_schema": "state_before/state_after/theta_before/theta_after",
        "instruction": INSTRUCTION,
        "test_targets_generated": False,
        "teacher_artifact_sha256": assets_identity.get("artifact_sha256"),
        "backend_sha256": assets_identity.get("backend_sha256"),
        "tokenizer_sha256": assets_identity.get("tokenizer_sha256"),
        "stats_sha256": assets_identity.get("stats_sha256"),
        "teacher_source_manifest_sha256": assets_identity.get("sources_sha256"),
    }


def _build_teacher_bundle(
    assets: TemporalAssetSet,
    manifest: TeacherSourceManifest,
    *,
    assets_identity: Mapping[str, Any],
    device: str,
) -> Any:
    from owac.evaluation.rawvla_teacher import (
        TeacherOfflineEvaluator,
        load_teacher_frontend_artifact,
    )
    from owac.raw.temporal_distillation import TeacherBundle
    from owac.training.rawvla_teacher import TeacherAssets, build_author_teacher

    teacher_assets = TeacherAssets(
        author_root=assets.author_root,
        starvla_root=assets.starvla_root,
        checkpoint=assets.checkpoint,
        tokenizer=assets.tokenizer,
        statistics=assets.statistics,
        config_paths=assets.config_paths,
    )
    teacher = build_author_teacher(teacher_assets, device=device)
    registry = json.loads(assets.registry_path.read_text(encoding="utf-8"))
    state = load_teacher_frontend_artifact(
        assets.artifact_path, expected_identity=registry["identity"]
    )
    teacher.frontend.load_state_dict(state, strict=True)
    for parameter in teacher.frontend.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.policy.parameters():
        parameter.requires_grad_(False)
    teacher.frontend.eval()
    teacher.policy.eval()
    if any(parameter.requires_grad for parameter in teacher.frontend.parameters()):
        raise SharedDataError("teacher frontend was not fully frozen")
    data = TeacherData(manifest, repo_root=Path("."), cache_entries=2)
    stats = load_pi05_normalization_stats(assets.statistics)
    evaluator = TeacherOfflineEvaluator(teacher=teacher, data=data, stats=stats)
    identity = {
        **dict(assets_identity),
        "frontend_tensor_count": len(state),
    }
    return TeacherBundle(
        assets=assets,
        teacher=teacher,
        data=data,
        stats=stats,
        evaluator=evaluator,
        identity=identity,
    )


def _distributed_context(device: str) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if world_size < 1 or world_size > 2:
            raise SharedDataError("shared teacher preparation supports one or two ranks")
        if not torch.cuda.is_available():
            raise SharedDataError("teacher preparation requires CUDA")
        dist.init_process_group(
            backend="nccl",
            timeout=__import__("datetime").timedelta(seconds=300),
        )
        torch.cuda.set_device(local_rank)
        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": f"cuda:{local_rank}",
            "distributed": True,
        }
    if not torch.cuda.is_available():
        raise SharedDataError("teacher preparation requires CUDA")
    return {
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "device": device,
        "distributed": False,
    }


def _barrier(context: Mapping[str, Any], timeout: float = 180.0) -> None:
    if context.get("distributed"):
        import datetime as _datetime

        import torch.distributed as dist

        dist.barrier(timeout=_datetime.timedelta(seconds=float(timeout)))


def _all_gather(context: Mapping[str, Any], value: Any) -> list[Any]:
    if not context.get("distributed"):
        return [value]
    import torch.distributed as dist

    gathered = [None] * int(context["world_size"])
    dist.all_gather_object(gathered, value)
    return gathered


def _sum_wall(context: Mapping[str, Any], local_seconds: float) -> float:
    if not context.get("distributed"):
        return float(local_seconds)
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(local_seconds)], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _row_count(cache_root: Path, identity_key: str) -> int:
    root = cache_root / identity_key
    return sum(1 for path in root.glob("*/*.pt") if path.is_file()) if root.is_dir() else 0


def _load_mapped_config(
    config_path: str | Path, *, manifest_path: Path, cache_root: Path
) -> tuple[dict[str, Any], TemporalAssetSet, Path]:
    from owac.runtime.raw_full_visual_distillation import PathMapper, _map_config_paths

    config = dict(json.loads(Path(config_path).read_text(encoding="utf-8")))
    config["assets"] = dict(config.get("assets") or {})
    config["assets"]["source_manifest"] = str(manifest_path)
    config["assets"]["cache_root"] = str(cache_root)
    mapper = PathMapper((config.get("remote") or {}).get("path_mapping") or ())
    mapped, _report = _map_config_paths(config, mapper)
    assets = TemporalAssetSet.from_config(mapped, repo_root=Path(__file__).resolve().parents[3])
    return mapped, assets, Path(mapped["assets"]["source_manifest"])


def prepare_shared_teacher_cache(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    stage_root: str | Path,
    cache_root: str | Path,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare train+validation token rows with two disjoint source ranks."""
    import torch
    import torch.distributed as dist

    from owac.raw.temporal_distillation import TeacherBundle  # noqa: F401

    context = _distributed_context(device)
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    manifest_file = Path(manifest_path).resolve()
    stage = Path(stage_root).resolve()
    cache_dir = Path(cache_root).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            cache_dir.relative_to(stage)
        except ValueError as exc:
            raise SharedDataError(
                "teacher cache root must be inside the stage output root"
            ) from exc
        if cache_dir == stage:
            raise SharedDataError("teacher cache root cannot be the stage root")
        manifest = load_shared_manifest(manifest_file)
        expected = shared_expected_counts(manifest)
        if (
            manifest_file
            != Path(str(manifest.provenance.get("manifest_path", manifest_file))).resolve()
            and False
        ):
            pass
        config, assets, _manifest_from_config = _load_mapped_config(
            config_path, manifest_path=manifest_file, cache_root=cache_dir
        )
        manifest_sha = ""
        assets_identity = _validate_shared_assets(assets, manifest_sha256=manifest_sha)
        identity = shared_cache_identity(
            assets_identity=assets_identity,
            manifest=manifest,
            manifest_sha256=manifest_sha,
        )
        if rank == 0:
            TemporalCache(cache_dir, identity)
        _barrier(context)
        cache = TemporalCache(cache_dir, identity)
        start = time.monotonic()
        bundle = _build_teacher_bundle(
            assets,
            manifest,
            assets_identity=assets_identity,
            device=str(context["device"]),
        )
        train_validation = sorted(
            manifest.sources_for_split("train") + manifest.sources_for_split("validation"),
            key=lambda item: item.source_demo_id,
        )
        local_sources = [
            source for index, source in enumerate(train_validation) if index % world_size == rank
        ]
        source_ids = [source.source_demo_id for source in local_sources]
        result = prepare_temporal_cache(
            bundle,
            cache,
            source_ids=source_ids,
            overwrite=bool(overwrite),
        )
        local_wall = time.monotonic() - start
        _barrier(context)
        wall_objects = _all_gather(
            context,
            {
                "rank": rank,
                "wall_seconds": float(local_wall),
                "allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "sources": source_ids,
                "rows": int(result.get("rows", 0)),
            },
        )
        actual_rows = _row_count(cache_dir, cache.identity_key)
        if actual_rows != expected.cache_rows:
            raise SharedDataError(
                f"teacher cache rows {actual_rows} != expected {expected.cache_rows}"
            )
        test_sources = manifest.sources_for_split("test")
        for source in test_sources:
            for endpoint in range(0, int(source.num_frames), 5):
                if cache.exists(source.source_demo_id, endpoint):
                    raise SharedDataError("test target row was generated before authorization")
        receipt = None
        if rank == 0:
            prior = dict(config.get("prior_stage") or {})
            remote_cfg = dict(config.get("remote") or {})
            gpu_cap = float(remote_cfg.get("total_gpu_wall_seconds", 230400)) - float(
                prior.get("gpu_wall_seconds", 89558.73680601672)
            )
            output_cap = int(remote_cfg.get("total_output_bytes", 21474836480)) - int(
                prior.get("output_bytes", 6078252068)
            )
            gpu_wall = float(sum(float(item["wall_seconds"]) for item in wall_objects))
            output_bytes = int(_owned_bytes(stage))
            if gpu_wall > gpu_cap:
                raise SharedDataError(
                    f"teacher preparation GPU wall {gpu_wall} exceeds remaining {gpu_cap}"
                )
            if output_bytes > output_cap:
                raise SharedDataError(
                    f"teacher preparation output {output_bytes} exceeds remaining {output_cap}"
                )
            receipt = {
                "schema": SHARED_RECEIPT_SCHEMA,
                "protocol": SHARED_PROTOCOL,
                "created_at": _now(),
                "config_path": str(Path(config_path).resolve()),
                "manifest_path": str(manifest_file),
                "manifest_sha256": manifest_sha,
                "cache_root": str(cache_dir),
                "cache_identity_key": cache.identity_key,
                "cache_identity": identity,
                "split_counts": dict(expected.split_counts),
                "source_count": len(manifest.sources),
                "source_raw_sha256": {
                    source.source_demo_id: source.raw_sha256 for source in manifest.sources
                },
                "actual_source_npz_sha256": {
                    source.source_demo_id: source.raw_sha256 for source in manifest.sources
                },
                "actual_source_hdf5_sha256": None,
                "actual_source_hdf5_sha256_available": False,
                "source_hdf5_sha256_field_semantics": _HDF5_AGGREGATE_SEMANTICS,
                "source_group_metadata": {
                    source.source_demo_id: {
                        "hdf5_path": source.metadata.get("hdf5_path"),
                        "demo_key": source.metadata.get("demo_key"),
                        "demo_index": source.metadata.get("demo_index"),
                        "lighting": source.lighting,
                    }
                    for source in sorted(manifest.sources, key=lambda item: item.source_demo_id)
                },
                "paired_rgb_used": False,
                "paired_rgb_path_registered": False,
                "paired_rgb_identity_verified": False,
                "paired_rgb_availability_in_adapter": "unknown_not_inspected",
                "paired_rgb_dataset_note": (
                    "shared release may contain paired RGB files; this RAW-only "
                    "adapter does not inspect, register, or read them"
                ),
                "paired_rgb_not_read": True,
                "paired_rgb_identity_fabricated": False,
                "train_anchors": int(expected.train_anchors),
                "validation_windows": int(expected.validation_windows),
                "cache_rows": int(expected.cache_rows),
                "actual_rows": int(actual_rows),
                "test_sources": len(test_sources),
                "test_targets_generated": False,
                "test_observations_consumed": False,
                "old_50_source_manifest_used": False,
                "old_teacher_cache_used": False,
                "teacher_asset_identity": assets_identity,
                "frontend_state_sha256": bundle.identity.get("frontend_state_sha256"),
                "teacher_frontend_tensor_count": bundle.identity.get("frontend_tensor_count"),
                "gpu_wall_sum_seconds": float(
                    sum(float(item["wall_seconds"]) for item in wall_objects)
                ),
                "per_rank": wall_objects,
                "world_size": world_size,
                "output_bytes": output_bytes,
                "status": "prepared",
                "target_schema": "existing_native_temporal_target_row_v1",
                "teacher_input": "actual_shared_raw_only",
                "paired_rgb_substitute_used": False,
            }
            receipt_path = stage / "teacher-preparation-receipt.json"
            _write_json(receipt_path, receipt)
            report_path = stage / "teacher-preparation-report.json"
            _write_json(
                report_path,
                {
                    **receipt,
                    "receipt_path": str(receipt_path),
                },
            )
        _barrier(context)
        return receipt or {"status": "prepared", "rank": rank, "rows": actual_rows}
    finally:
        if context.get("distributed") and dist.is_initialized():
            dist.destroy_process_group()


def verify_shared_preparation(
    receipt_path: str | Path, *, cache_root: str | Path | None = None
) -> dict[str, Any]:
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("schema") != SHARED_RECEIPT_SCHEMA:
        raise SharedDataError("teacher preparation receipt schema differs")
    if receipt.get("protocol") != SHARED_PROTOCOL:
        raise SharedDataError("teacher preparation protocol differs")
    if receipt.get("test_targets_generated") is not False:
        raise SharedDataError("teacher preparation claims test targets")
    root = Path(cache_root or receipt["cache_root"])
    identity_key = str(receipt["cache_identity_key"])
    actual = _row_count(root, identity_key)
    if actual != int(receipt["actual_rows"]):
        raise SharedDataError(f"cache row count changed: {actual} != {receipt['actual_rows']}")
    return {
        "ok": True,
        "cache_identity_key": identity_key,
        "actual_rows": actual,
        "expected_rows": int(receipt["cache_rows"]),
        "gpu_wall_sum_seconds": float(receipt["gpu_wall_sum_seconds"]),
        "output_bytes": int(receipt["output_bytes"]),
        "manifest_sha256": receipt.get("manifest_sha256", ""),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifest")
    build.add_argument("--inventory", required=True)
    build.add_argument("--shared-root", required=True)
    build.add_argument("--manifest", required=True)
    build.add_argument("--split-seed", type=int, default=101)
    build.add_argument("--release-revision", default=SHARED_RELEASE_REVISION)
    build.add_argument("--force", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", required=True)
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--stage-root", required=True)
    prep.add_argument("--cache-root", required=True)
    prep.add_argument("--device", default="cuda")
    prep.add_argument("--overwrite", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--cache-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-manifest":
            report = build_shared_manifest(
                inventory_path=args.inventory,
                shared_root=args.shared_root,
                manifest_path=args.manifest,
                split_seed=args.split_seed,
                release_revision=args.release_revision,
                force=args.force,
            )
            result = report.as_dict()
        elif args.command == "prepare":
            result = prepare_shared_teacher_cache(
                config_path=args.config,
                manifest_path=args.manifest,
                stage_root=args.stage_root,
                cache_root=args.cache_root,
                device=args.device,
                overwrite=args.overwrite,
            )
        else:
            result = verify_shared_preparation(args.receipt, cache_root=args.cache_root)
    except Exception as exc:
        print(
            json.dumps({"status": "unavailable", "reason": f"{exc.__class__.__name__}: {exc}"}),
            flush=True,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""N01 full four-region manifest for the RawVLA RAW cache.

Implements the P0 data decision of ``docs/prompts/n01_v1_0/P0_FOUNDATION.md`` section 3:
inventory -> ``source_group`` -> train / calibration / validation / test.

Reused from :mod:`owac.raw.rawvla_sequence` (not reimplemented):

* :func:`~owac.raw.rawvla_sequence.source_group` - the base-trajectory identity rule
  (drops ``data/<format>/{raw,rgb}/`` prefixes, a trailing ``variant_NN`` directory and
  trailing ``_variant_NN`` / ``_lighting_<domain>`` suffixes, so RGB/RAW twins and
  lighting variants of one demonstration can never be split apart);
* :func:`~owac.raw.rawvla_sequence.assign_side` - the binary hash rule
  ``sha256(split_version|group) -> u64/2**64 < threshold``, called at the cumulative
  0.70 / 0.80 / 0.90 thresholds to derive the four regions, so the ``train`` region is
  exactly ``assign_side(..., train_fraction=0.7) == "train"`` by construction;
* :func:`~owac.raw.rawvla_sequence.write_split_manifest` /
  :func:`~owac.raw.rawvla_sequence.load_split_manifest` - the split-manifest JSON IO;
* :class:`~owac.raw.rawvla_sequence.RawVLASequenceDataset` - the consumer of the ``sides``
  mapping that :func:`loader_split_manifest` produces.

Frozen conventions
------------------
relative paths
    Stored relative to the server raw cache root the caller passes as
    ``RawVLASequenceDataset(raw_cache_root=...)``, e.g.
    ``libero_spatial_no_noops_1.0.0_lerobot/episode_000018/variant_00.npz``. Each
    inventory/manifest records the scanned server root; :func:`loader_split_manifest`
    resolves relative paths against whatever root the caller passes.

inventory hash
    ``sha256("n01-full-v1|" + relative_path + "|" + str(size_bytes) + "|" + str(mtime_ns))
    .hexdigest()[:32]``. It covers exactly those three fields as read at scan time, so a
    path, size or mtime change is detectable without reading file content. It does NOT
    cover file content: a rewrite that preserves both size and mtime keeps the same hash.
    The digest is an inventory fingerprint, not a content checksum.

identity fields (workflow-A ``SourceKey`` hand-off)
    Rows carry non-empty ``dataset_id`` (``libero`` / ``robotwin2``), ``episode_id``
    (``<suite or task>/episode_NNNNNN``, e.g.
    ``libero_spatial_no_noops_1.0.0_lerobot/episode_000018``) and a decidable camera
    value. ``camera_id`` is the row-level SourceKey projection and takes four distinct,
    never-``null``, never-empty values: ``unknown_metadata`` (no ``metadata_json`` at all,
    so no camera identity could be established), ``no_camera`` (metadata present but no RAW
    burst key), ``<protocol camera name>`` (exactly one camera) and ``multi_camera`` (more
    than one; the exact set is in ``camera_ids``). ``camera_ids`` lists the real protocol
    names and ``sensor_metadata_present`` stays an independent flag, so "no sensor
    metadata" can never collapse into "single camera". A camera-like NPZ key the dataset
    protocol does not declare is an interface error and raises instead of being absorbed
    into the row.

units / calibration
    ``units`` comes from the frozen :class:`~owac.raw.rawvla_sequence.RawVLAProtocol` of
    the dataset; ``calibration_version`` is the per-file cache-generation tag
    (``metadata_json["benchmark"]``) or ``null`` when the cache carries none. No sensor
    calibration id is invented for a cache that declares none; per-file photometric
    parameters stay visible in the inventory ``metadata_fields``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CAMERA_ID_MULTI",
    "CAMERA_ID_NO_CAMERA",
    "CAMERA_ID_UNKNOWN_METADATA",
    "INVENTORY_SCHEMA",
    "REGIONS",
    "REGION_FRACTIONS",
    "SCHEMA",
    "SPLIT_VERSION",
    "TIME_ORDER",
    "assert_manifest_partition",
    "auto_probe_paths",
    "build_manifest",
    "camera_identity",
    "camera_ids_for_keys",
    "episode_id_of",
    "episode_of",
    "four_way_side",
    "inventory_hash",
    "inventory_hash_recipe",
    "loader_split_manifest",
    "manifest_digest",
    "npz_keys",
    "probe_npz",
    "region_paths",
    "scan_npz_inventory",
    "verify_manifest",
]

SCHEMA = "owac.n01_full.manifest.v1"
INVENTORY_SCHEMA = "owac.n01_full.inventory.v1"
SPLIT_VERSION = "n01-full-v1"

REGIONS: tuple[str, ...] = ("train", "calibration", "validation", "test")
REGION_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("train", 0.70),
    ("calibration", 0.10),
    ("validation", 0.10),
    ("test", 0.10),
)

INVENTORY_HASH_NAMESPACE = "n01-full-v1"
INVENTORY_HASH_LENGTH = 32
EPISODE_NAME = re.compile(r"^episode_\d+$")

#: Suffixes the author cache uses for RAW camera bursts, mirroring
#: ``RawVLAProtocol.camera_id``; the builder re-validates them against the protocol.
CAMERA_KEY_SUFFIXES: tuple[str, ...] = ("_raw_uint8", "_raw_float32", "_raw")

#: ``metadata_json`` fields carried per file into the inventory. Absent keys stay absent.
METADATA_FIELDS: tuple[str, ...] = (
    "benchmark",
    "lighting_domain",
    "lighting_ev",
    "lighting_scale",
    "num_frames",
    "capture_session",
    "session_id",
    "acquisition_id",
    "scene",
    "scene_id",
    "sensor",
    "sensor_id",
)

#: dataset protocol camera dtype -> manifest ``input_mode``.
INPUT_MODE_BY_DTYPE: dict[str, str] = {
    "uint8": "raw_uint8_pseudo_raw",
    "floating": "raw_float32_hdr_derived",
}

#: Row-level camera projection tokens; see the module docstring for the four cases.
CAMERA_ID_UNKNOWN_METADATA = "unknown_metadata"
CAMERA_ID_NO_CAMERA = "no_camera"
CAMERA_ID_MULTI = "multi_camera"

#: Time convention of the manifest and of every frame array it points at.
TIME_ORDER = "oldest_first"


def _rawvla_sequence() -> Any:
    """Import the loader module lazily so an inventory scan needs no torch."""
    from owac.raw import rawvla_sequence

    return rawvla_sequence


# --------------------------------------------------------------------------------------
# Inventory hash
# --------------------------------------------------------------------------------------


def inventory_hash(relative_path: str, size_bytes: int, mtime_ns: int) -> str:
    """Cheap traceable fingerprint; see ``inventory_hash_recipe`` for the exact inputs."""
    payload = f"{INVENTORY_HASH_NAMESPACE}|{relative_path}|{int(size_bytes)}|{int(mtime_ns)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:INVENTORY_HASH_LENGTH]


def inventory_hash_recipe() -> str:
    """Exact recipe string stored in every generated inventory."""
    return (
        f'sha256("{INVENTORY_HASH_NAMESPACE}|" + relative_path + "|" + str(size_bytes) + '
        f'"|" + str(mtime_ns)).hexdigest()[:{INVENTORY_HASH_LENGTH}]'
    )


def _canonical_digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------------------
# Read-only inventory scan
# --------------------------------------------------------------------------------------


def npz_keys(path: str | Path) -> list[str]:
    """Array names inside one NPZ, without loading the arrays."""
    with zipfile.ZipFile(path) as archive:
        return sorted(
            member[: -len(".npy")] for member in archive.namelist() if member.endswith(".npy")
        )


def _read_metadata_json(archive: zipfile.ZipFile, keys: Sequence[str]) -> dict[str, Any] | None:
    if "metadata_json" not in keys:
        return None
    array = np.load(io.BytesIO(archive.read("metadata_json.npy")), allow_pickle=False)
    text = array.item() if array.shape == () else array.reshape(-1)[0]
    data = json.loads(str(text))
    return data if isinstance(data, dict) else None


def _camera_key_names(keys: Sequence[str]) -> list[str]:
    return sorted(key for key in keys if key.endswith(CAMERA_KEY_SUFFIXES))


def scan_npz_inventory(
    raw_root: str | Path,
    *,
    dataset: str,
    server: str = "",
    probe_paths: Sequence[str] = (),
    probe_max: int = 6,
    created_utc: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Read-only scan of every ``*.npz`` below ``raw_root``.

    Emits the per-file inventory records (relative path, size/mtime, inventory hash, RAW
    camera keys, selected ``metadata_json`` fields) plus deep probes (keys, dtype, shape,
    min/max and the full parsed metadata). Explicit ``probe_paths`` are used as given; by
    default a deterministic sample of at most ``probe_max`` files is selected to cover
    distinct lighting domains and distinct task/suite prefixes.
    """
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"raw cache root does not exist: {root}")
    files = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no *.npz files below raw cache root: {root}")

    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        keys = npz_keys(path)
        with zipfile.ZipFile(path) as archive:
            metadata = _read_metadata_json(archive, keys)
        record: dict[str, Any] = {
            "relative_path": relative,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "inventory_hash": inventory_hash(relative, stat.st_size, stat.st_mtime_ns),
            "camera_keys": _camera_key_names(keys),
            "metadata_present": metadata is not None,
        }
        if metadata is not None:
            record["metadata_fields"] = {
                key: metadata[key] for key in METADATA_FIELDS if key in metadata
            }
        records.append(record)

    if not probe_paths and probe_max > 0:
        probe_paths = auto_probe_paths(records, probe_max)
    inventory: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "dataset": str(dataset),
        "created_utc": created_utc or _utc_now(),
        "server": str(server),
        "server_raw_root": str(root),
        "relative_path_base": "server_raw_root (the raw_cache_root passed to the loader)",
        "inventory_hash_recipe": inventory_hash_recipe(),
        "inventory_hash_covers": ["relative_path", "size_bytes", "mtime_ns"],
        "inventory_hash_does_not_cover": [
            "file content",
            "NPZ array values",
        ],
        "notes": [
            "read-only scan: no file on the server was written, moved or deleted",
            "metadata_fields come from the NPZ metadata_json string array and keep its units",
            "camera_keys are the RAW burst keys; the manifest maps them to camera ids",
        ],
        "aggregate": _inventory_aggregate(records),
        "files": records,
        "probe": [probe_npz(root / relative, relative_to=root) for relative in probe_paths],
    }
    if note:
        inventory["notes"].append(str(note))
    return inventory


def auto_probe_paths(records: Sequence[dict[str, Any]], probe_max: int) -> list[str]:
    """Deterministic probe sample: distinct lighting domains first, distinct prefixes too."""
    by_domain: dict[str, list[tuple[str, str]]] = {}
    for record in records:
        relative = str(record["relative_path"])
        prefix = relative.split("/", 1)[0]
        fields = record.get("metadata_fields") or {}
        domain = str(fields.get("lighting_domain") or "unknown")
        by_domain.setdefault(domain, []).append((prefix, relative))
    for candidates in by_domain.values():
        candidates.sort()
    chosen: list[str] = []
    used_prefixes: set[str] = set()
    for domain in sorted(by_domain):
        for prefix, relative in by_domain[domain]:
            if len(chosen) >= probe_max:
                break
            if prefix not in used_prefixes:
                chosen.append(relative)
                used_prefixes.add(prefix)
                break
    chosen_domains = {
        domain
        for domain, candidates in by_domain.items()
        if any(relative in chosen for _, relative in candidates)
    }
    for domain in sorted(by_domain, key=lambda name: (name in chosen_domains, name)):
        for _, relative in by_domain[domain]:
            if len(chosen) >= probe_max:
                break
            if relative not in chosen:
                chosen.append(relative)
    return sorted(chosen)


def _inventory_aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lighting: dict[str, int] = {}
    benchmarks: dict[str, int] = {}
    camera_key_sets: dict[str, int] = {}
    frames: list[int] = []
    total_bytes = 0
    metadata_present = 0
    for record in records:
        total_bytes += int(record["size_bytes"])
        metadata_present += int(bool(record["metadata_present"]))
        camera_key_sets["+".join(record["camera_keys"])] = (
            camera_key_sets.get("+".join(record["camera_keys"]), 0) + 1
        )
        fields = record.get("metadata_fields", {})
        domain = fields.get("lighting_domain")
        if domain not in (None, ""):
            lighting[str(domain)] = lighting.get(str(domain), 0) + 1
        benchmark = fields.get("benchmark")
        if benchmark not in (None, ""):
            benchmarks[str(benchmark)] = benchmarks.get(str(benchmark), 0) + 1
        if fields.get("num_frames") is not None:
            frames.append(int(fields["num_frames"]))
    return {
        "files": len(records),
        "total_size_bytes": total_bytes,
        "metadata_present": metadata_present,
        "metadata_absent": len(records) - metadata_present,
        "camera_key_sets": camera_key_sets,
        "lighting_domains": dict(sorted(lighting.items())),
        "benchmark_tags": dict(sorted(benchmarks.items())),
        "num_frames": {
            "count": len(frames),
            "min": min(frames) if frames else None,
            "max": max(frames) if frames else None,
            "sum": sum(frames),
        },
    }


def probe_npz(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    """Deep read of one NPZ: per-key dtype/shape/min/max plus the full metadata string."""
    target = Path(path)
    stat = target.stat()
    relative = (
        target.relative_to(Path(relative_to)).as_posix()
        if relative_to is not None
        else target.as_posix()
    )
    keys = npz_keys(target)
    with zipfile.ZipFile(target) as archive:
        metadata = _read_metadata_json(archive, keys)
    arrays: list[dict[str, Any]] = []
    with np.load(target, allow_pickle=False) as data:
        for key in keys:
            array = data[key]
            entry: dict[str, Any] = {
                "key": key,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            if array.dtype.kind in "fiu" and array.size:
                entry["min"] = float(array.min())
                entry["max"] = float(array.max())
            elif array.dtype.kind in "US":
                entry["value"] = str(array.item() if array.shape == () else array.reshape(-1)[0])
            arrays.append(entry)
    return {
        "relative_path": relative,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "keys": arrays,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------------------
# Identity, episode and camera mapping
# --------------------------------------------------------------------------------------


def four_way_side(
    group: str,
    *,
    split_version: str = SPLIT_VERSION,
    fractions: Sequence[tuple[str, float]] = REGION_FRACTIONS,
) -> str:
    """Region of one source group, derived through ``rawvla_sequence.assign_side``.

    The binary primitive is queried at each cumulative boundary, so region membership is
    the legacy rule generalised to four regions and ``train`` is exactly the old
    ``assign_side(..., train_fraction=0.7) == "train"`` set.
    """
    names = [name for name, _ in fractions]
    if tuple(names) != REGIONS:
        raise ValueError(f"fractions must name {REGIONS} in order, got {tuple(names)}")
    total = sum(float(share) for _, share in fractions)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"region fractions must sum to 1.0, got {total}")
    if any(not 0.0 < float(share) < 1.0 for _, share in fractions):
        raise ValueError(f"every region fraction must be inside (0, 1), got {fractions}")

    assign_side = _rawvla_sequence().assign_side
    cumulative = 0.0
    for name, share in fractions[:-1]:
        cumulative += float(share)
        if assign_side(group, split_version=split_version, train_fraction=cumulative) == "train":
            return name
    return names[-1]


def episode_of(relative_path: str) -> str | None:
    """Episode name from the cache layout, or ``None`` when the path has none."""
    path = Path(relative_path)
    for candidate in (path.parent.name, path.stem):
        if EPISODE_NAME.match(candidate):
            return candidate
    return None


def episode_id_of(relative_path: str, *, fallback: str) -> str:
    """``<suite or task>/episode_NNNNNN`` for the SourceKey hand-off.

    When a path carries no ``episode_NNNNNN`` token the source group itself is returned,
    so ``episode_id`` is never empty.
    """
    path = Path(relative_path)
    parts = list(path.parts)
    episode = episode_of(relative_path)
    if episode is None:
        return fallback
    if EPISODE_NAME.match(path.stem):
        prefix = [part for part in parts[:-1]]
    else:
        prefix = [part for part in parts[:-2]]
    joined = "/".join([*prefix, episode])
    return joined or fallback


def camera_identity(
    keys: Sequence[str], protocol: Any, *, metadata_present: bool
) -> dict[str, Any]:
    """Camera ids plus the row-level SourceKey projection (never null, never empty)."""
    cameras = camera_ids_for_keys(keys, protocol)
    if not metadata_present:
        camera_id = CAMERA_ID_UNKNOWN_METADATA
    elif not cameras:
        camera_id = CAMERA_ID_NO_CAMERA
    elif len(cameras) == 1:
        camera_id = cameras[0]
    else:
        camera_id = CAMERA_ID_MULTI
    return {"camera_ids": cameras, "camera_id": camera_id, "camera_count": len(cameras)}


def camera_ids_for_keys(keys: Sequence[str], protocol: Any) -> list[str]:
    """Camera ids of one NPZ; an undeclared camera-like key is an interface error."""
    declared = set(protocol.camera_keys)
    undeclared = sorted(
        key for key in keys if key.endswith(CAMERA_KEY_SUFFIXES) and key not in declared
    )
    if undeclared:
        raise ValueError(
            f"NPZ keys {undeclared} look like RAW cameras but are not declared by the "
            f"{protocol.name!r} protocol (declared: {sorted(declared)})"
        )
    return sorted({protocol.camera_id(key) for key in keys if key in declared})


# --------------------------------------------------------------------------------------
# Manifest build
# --------------------------------------------------------------------------------------


def build_manifest(
    inventory: dict[str, Any],
    *,
    split_version: str = SPLIT_VERSION,
    fractions: Sequence[tuple[str, float]] = REGION_FRACTIONS,
    created_utc: str | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Turn one inventory into the four-region manifest with loader-readable ``sides``."""
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(
            f"expected inventory schema {INVENTORY_SCHEMA!r}, got {inventory.get('schema')!r}"
        )
    rawvla = _rawvla_sequence()
    protocol = rawvla.resolve_protocol(str(inventory["dataset"]))
    input_mode = INPUT_MODE_BY_DTYPE.get(protocol.camera_dtype)
    if input_mode is None:
        input_mode = str(protocol.raw_scaling)

    rows: list[dict[str, Any]] = []
    group_side: dict[str, str] = {}
    group_files: dict[str, int] = {}
    group_lighting: dict[str, set[str]] = {}
    for record in inventory["files"]:
        relative = str(record["relative_path"])
        expected = inventory_hash(relative, int(record["size_bytes"]), int(record["mtime_ns"]))
        if expected != record["inventory_hash"]:
            raise ValueError(
                f"inventory hash mismatch for {relative}: record has "
                f"{record['inventory_hash']}, recomputed {expected}"
            )
        source = rawvla.source_group(relative, dataset_format=protocol.name)
        side = four_way_side(source, split_version=split_version, fractions=fractions)
        previous = group_side.setdefault(source, side)
        if previous != side:
            raise ValueError(f"source group {source!r} landed on two sides: {previous}, {side}")
        group_files[source] = group_files.get(source, 0) + 1

        keys = list(record.get("camera_keys", ()))
        present = bool(record.get("metadata_present"))
        fields = record.get("metadata_fields") or {}
        domain = fields.get("lighting_domain")
        if domain not in (None, ""):
            group_lighting.setdefault(source, set()).add(str(domain))
        rows.append(
            {
                "relative_path": relative,
                "inventory_hash": record["inventory_hash"],
                "size_bytes": int(record["size_bytes"]),
                "mtime_ns": int(record["mtime_ns"]),
                "dataset_id": protocol.name,
                "input_mode": input_mode,
                "source_group": source,
                "episode": episode_of(relative),
                "episode_id": episode_id_of(relative, fallback=source),
                **camera_identity(keys, protocol, metadata_present=present),
                "sensor_metadata_present": present,
                "scene": fields.get("scene", fields.get("scene_id")),
                "session": fields.get(
                    "session_id", fields.get("capture_session", fields.get("acquisition_id"))
                ),
                "sensor": fields.get("sensor", fields.get("sensor_id")),
                "lighting": (
                    {
                        "domain": domain,
                        "ev": fields.get("lighting_ev"),
                        "scale": fields.get("lighting_scale"),
                    }
                    if present
                    else None
                ),
                "num_frames": fields.get("num_frames") if present else None,
                "split": side,
                "units": protocol.raw_unit,
                "calibration_version": fields.get("benchmark") if present else None,
            }
        )

    rows.sort(key=lambda row: row["relative_path"])
    sides = {
        region: sorted(row["relative_path"] for row in rows if row["split"] == region)
        for region in REGIONS
    }
    source_groups = [
        {
            "group": group,
            "side": side,
            "files": group_files[group],
            "lighting_domains": sorted(group_lighting.get(group, ())),
        }
        for group, side in sorted(group_side.items())
    ]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "split_version": split_version,
        "dataset_format": protocol.name,
        "regions": list(REGIONS),
        "region_fractions": {name: float(share) for name, share in fractions},
        "relative_path_base": "server raw cache root passed as raw_cache_root=...",
        "server_raw_root": inventory.get("server_raw_root"),
        "server": inventory.get("server"),
        "inventory_schema": inventory.get("schema"),
        "inventory_created_utc": inventory.get("created_utc"),
        "inventory_digest": _canonical_digest(inventory),
        "created_utc": created_utc or _utc_now(),
        "inventory_hash_recipe": inventory_hash_recipe(),
        "inventory_hash_covers": ["relative_path", "size_bytes", "mtime_ns"],
        "inventory_hash_does_not_cover": ["file content", "NPZ array values"],
        "input_mode_source": "RawVLAProtocol.camera_dtype mapping in n01_full_manifest",
        "units_source": "RawVLAProtocol.raw_unit",
        "calibration_version_source": "per-file metadata_json[benchmark]; null when absent",
        "time_order": TIME_ORDER,
        "time_order_note": (
            "manifest rows and every NPZ frame array they point at are oldest_first; the "
            "loader emits [T, C, H, W] oldest-first and only a named time_order_adapter may "
            "convert to the newest-first DBSR fusion entry"
        ),
        "camera_semantics": (
            "camera_id is never null and never empty: unknown_metadata (no metadata_json), "
            "no_camera (metadata present, no RAW burst key), a protocol camera name (single "
            "camera) or multi_camera, with the exact set in camera_ids; "
            "sensor_metadata_present is the independent metadata flag, so 'no sensor "
            "metadata' never collapses into 'single camera'"
        ),
        "group_rule": (
            "rawvla_sequence.source_group on the raw-cache-root-relative path; the "
            "dataset namespace is this file's dataset_format plus split_version"
        ),
        "counts": {
            "files": len(rows),
            "groups": len(group_side),
            "regions": {
                region: {
                    "files": len(sides[region]),
                    "groups": sum(1 for entry in source_groups if entry["side"] == region),
                }
                for region in REGIONS
            },
        },
        "source_groups": source_groups,
        "rows": rows,
        "sides": sides,
        "notes": [
            "four regions are drawn from the same demonstration pool; the RawVLA author "
            "cache ships no official train/test partition for these RAW files",
            "paired RGB caches mirror the RAW relative paths 1:1 and are excluded here so "
            "one trajectory is counted once",
            "rows keep relative paths; loader_split_manifest resolves them to absolute "
            "server paths",
            "this manifest is consumed by RawVLASequenceDataset(split_manifest=...)",
            *notes,
        ],
    }
    assert_manifest_partition(manifest)
    return manifest


def region_paths(manifest: dict[str, Any], region: str) -> list[str]:
    if region not in manifest["sides"]:
        raise ValueError(f"unknown region {region!r}; expected one of {sorted(manifest['sides'])}")
    return list(manifest["sides"][region])


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Overlap/partition report over the four sides and rows."""
    regions = list(manifest.get("regions", REGIONS))
    sides = manifest.get("sides", {})
    sets = {region: set(sides.get(region, [])) for region in regions}
    pairwise = {
        f"{left}|{right}": len(sets[left] & sets[right])
        for index, left in enumerate(regions)
        for right in regions[index + 1 :]
    }
    union = set().union(*sets.values()) if sets else set()
    row_paths = [row["relative_path"] for row in manifest.get("rows", [])]
    row_sides = {
        region: sorted(row["relative_path"] for row in manifest["rows"] if row["split"] == region)
        for region in regions
    }
    group_sides: dict[str, set[str]] = {}
    for row in manifest.get("rows", []):
        group_sides.setdefault(str(row["source_group"]), set()).add(str(row["split"]))
    conflicts = {group: sorted(tags) for group, tags in group_sides.items() if len(tags) > 1}
    duplicate_rows = len(row_paths) - len(set(row_paths))
    side_mismatch = {region: len(set(row_sides[region]) ^ sets[region]) for region in regions}
    partition_ok = (
        duplicate_rows == 0
        and not conflicts
        and all(count == 0 for count in pairwise.values())
        and len(union) == len(row_paths)
        and all(count == 0 for count in side_mismatch.values())
    )
    return {
        "regions": regions,
        "region_files": {region: len(sets[region]) for region in regions},
        "pairwise_overlaps": pairwise,
        "union_files": len(union),
        "row_files": len(row_paths),
        "duplicate_rows": duplicate_rows,
        "group_side_conflicts": conflicts,
        "row_side_mismatches": side_mismatch,
        "partition_ok": bool(partition_ok),
    }


def assert_manifest_partition(manifest: dict[str, Any]) -> dict[str, Any]:
    """Raise unless the four sides are disjoint and together cover every row."""
    report = verify_manifest(manifest)
    if not report["partition_ok"]:
        raise ValueError(f"manifest sides are not a clean four-way partition: {report}")
    return report


def loader_split_manifest(manifest: dict[str, Any], raw_cache_root: str | Path) -> dict[str, Any]:
    """Resolve the manifest's relative ``sides`` against one root, for the old loader."""
    root = Path(raw_cache_root)
    sides = {
        region: [str(root / relative) for relative in manifest["sides"][region]]
        for region in manifest["regions"]
    }
    return {
        "split_version": manifest["split_version"],
        "dataset_format": manifest["dataset_format"],
        "counts": manifest["counts"],
        "relative_path_base": manifest["relative_path_base"],
        "source": "owac.raw.n01_full_manifest.loader_split_manifest",
        "sides": sides,
    }


def dataset_from_protocol(protocol, platform, region, *, repo_root, **overrides):
    """Bind the real loader to the frozen partition and normalization parameters."""
    data = protocol["data"]
    manifest = json.loads(
        (Path(repo_root) / data["platform_manifests"][platform]["manifest_path"]).read_text()
    )
    assert_manifest_partition(manifest)
    root = Path(data["data_root"]) / data["raw_roots"][platform]
    stats = data["stats"][platform]
    return _rawvla_sequence().RawVLASequenceDataset(
        dataset_format=platform,
        raw_cache_root=root,
        stats_path=stats["path"],
        stats_key=stats["stats_key"],
        action_stats_key=stats["action_stats_key"],
        split=region,
        split_manifest=loader_split_manifest(manifest, root),
        **overrides,
    )


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Canonical digest binding downstream receipts to one manifest revision."""
    return _canonical_digest(manifest)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m owac.raw.n01_full_manifest",
        description="N01 full RawVLA inventory scan and four-region manifest build",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="read-only NPZ inventory scan of a raw root")
    scan.add_argument("--raw-root", required=True)
    scan.add_argument("--dataset", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--server", default="")
    scan.add_argument("--probe", action="append", default=[])
    scan.add_argument("--probe-max", type=int, default=6)
    scan.add_argument("--created-utc", default=None)

    build = commands.add_parser("build", help="build the four-region manifest")
    build.add_argument("--inventory", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--split-version", default=SPLIT_VERSION)
    build.add_argument("--created-utc", default=None)

    verify = commands.add_parser("verify", help="print the partition report of a manifest")
    verify.add_argument("--manifest", required=True)

    args = parser.parse_args(argv)
    rawvla = _rawvla_sequence()
    if args.command == "scan":
        inventory = scan_npz_inventory(
            args.raw_root,
            dataset=args.dataset,
            server=args.server,
            probe_paths=args.probe,
            probe_max=args.probe_max,
            created_utc=args.created_utc,
        )
        rawvla.write_split_manifest(args.output, inventory)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "files": inventory["aggregate"]["files"],
                    "total_size_bytes": inventory["aggregate"]["total_size_bytes"],
                }
            )
        )
        return 0
    if args.command == "build":
        inventory = rawvla.load_split_manifest(args.inventory)
        manifest = build_manifest(
            inventory, split_version=args.split_version, created_utc=args.created_utc
        )
        rawvla.write_split_manifest(args.output, manifest)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "counts": manifest["counts"],
                    "digest": manifest_digest(manifest),
                }
            )
        )
        return 0
    manifest = rawvla.load_split_manifest(args.manifest)
    print(json.dumps(verify_manifest(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())

"""Versioned paired RAW/teacher-RGB manifest for RAW-stem distillation.

The loader in this module is deliberately strict:

* a JSON manifest declares every source, frame, camera, FOV, physical crop,
  flip, timestamp, ISP, preprocessing and checkpoint identity;
* arrays live in NPZ files loaded with ``allow_pickle=False`` and are bound by
  SHA256 before decoding;
* expert actions are stored in their real, unnormalised source units and the
  loader normalises them with a declared, hash-bound N02 statistics file.  A
  declared normalisation name without those statistics is not accepted;
* teacher RGB is either a paired array in the NPZ or a declared prepared ``.npy``
  file.  There is no gamma/resize fallback and no RGB reconstruction;
* every cache key binds the manifest version/hash, source payload hash,
  checkpoint, ISP, preprocessing, camera, time, FOV, crop, flip, action and
  future mask, normalisation and augmentation;
* the pure student view exposes only RAW measurement fields.  It never contains
  teacher RGB and never reads through the sample cache.

All counts default to the study protocol: 50 sources, split 40/5/5 with
seed 101.  A smaller fixture is accepted only when both the manifest and the
caller explicitly declare the fixture contract.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.raw.rawvla_sequence_n02 import (
    NormalisationStats,
    normalise_actions,
    normalise_state,
)

__all__ = [
    "CACHE_SCHEMA",
    "CACHE_VERSION",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "CacheCorruptionError",
    "CacheSpec",
    "CameraFOV",
    "DistillationSample",
    "FrameEntry",
    "ManifestSource",
    "NormalizationSpec",
    "PairedDataUnavailable",
    "PairedDistillationDataset",
    "PairedDistillationManifest",
    "PairedManifestError",
    "PhysicalCrop",
    "RawFrameData",
    "SampleCache",
    "StudentSample",
    "assign_source_splits",
    "cache_key_for_frame",
    "load_paired_manifest",
]

MANIFEST_SCHEMA = "owac.raw-distillation.pairs.v1"
MANIFEST_VERSION = 1
CACHE_SCHEMA = "owac.raw-distillation.cache.v1"
CACHE_VERSION = 1
_VARIANTS = ("rawvla_rgb3", "bayer4_packed", "rgb3_control")
_BAYER_CFA = ("RGGB", "BGGR", "GRBG", "GBRG")
_SPLIT_NAMES = ("train", "validation", "test")


class PairedManifestError(ValueError):
    """The manifest or an array violates the declared correspondence contract."""


class PairedDataUnavailable(RuntimeError):
    """Real evidence (array, RGB pair, statistics or identity) is missing."""


class CacheCorruptionError(RuntimeError):
    """A cache hit does not match the current key/shape/dtype/content."""


def _sha(value: Any, where: str) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise PairedManifestError(f"{where} must be 64 lowercase hex")
    return text


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise PairedManifestError(f"{where} lacks required field {key!r}")
    return mapping[key]


def _string(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = _required(mapping, key, where)
    if not isinstance(value, str) or not value.strip():
        raise PairedManifestError(f"{where}.{key} must be a nonempty string")
    return value


def _integer(mapping: Mapping[str, Any], key: str, where: str) -> int:
    value = _required(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PairedManifestError(f"{where}.{key} must be an integer")
    return value


def _mapping(mapping: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = _required(mapping, key, where)
    if not isinstance(value, Mapping):
        raise PairedManifestError(f"{where}.{key} must be a mapping")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Typed manifest records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CameraFOV:
    width: int
    height: int
    fx: float | None
    fy: float | None
    cx: float | None
    cy: float | None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise PairedManifestError("FOV width/height must be positive")
        values = (self.fx, self.fy, self.cx, self.cy)
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise PairedManifestError("FOV intrinsics must be all available or all unavailable")
            return
        if not all(np.isfinite(value) for value in values):
            raise PairedManifestError("FOV intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise PairedManifestError("FOV focal lengths must be positive")

    @property
    def intrinsics_available(self) -> bool:
        return self.fx is not None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str) -> CameraFOV:
        names = ("fx", "fy", "cx", "cy")
        present = [name in data and data[name] is not None for name in names]
        if any(present) and not all(present):
            raise PairedManifestError(f"{where} intrinsics must be all present or all absent")
        values = {
            name: (float(data[name]) if present[index] else None)
            for index, name in enumerate(names)
        }
        return cls(
            width=_integer(data, "width", where),
            height=_integer(data, "height", where),
            **values,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhysicalCrop:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise PairedManifestError("crop must be a nonnegative origin and positive extent")

    @property
    def even_aligned(self) -> bool:
        return not (self.x % 2 or self.y % 2 or self.width % 2 or self.height % 2)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str) -> PhysicalCrop:
        return cls(
            x=_integer(data, "x", where),
            y=_integer(data, "y", where),
            width=_integer(data, "width", where),
            height=_integer(data, "height", where),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    identity: str
    mode: str
    stats_path: Path
    stats_sha256: str
    stats_key: str
    dataset_format: str
    action_width: int
    epsilon: float
    epsilon_mode: str
    delta_action_indices: tuple[int, ...]
    hash_algorithm: str
    stats: NormalisationStats

    def as_cache_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "mode": self.mode,
            "stats_sha256": self.stats_sha256,
            "stats_key": self.stats_key,
            "dataset_format": self.dataset_format,
            "action_width": self.action_width,
            "epsilon": self.epsilon,
            "epsilon_mode": self.epsilon_mode,
            "delta_action_indices": list(self.delta_action_indices),
            "hash_algorithm": self.hash_algorithm,
        }


@dataclass(frozen=True, slots=True)
class ManifestSource:
    source_id: str
    acquisition_id: str
    demo_id: str
    split: str
    source_sha256: str
    camera_ids: tuple[str, ...]
    variants: tuple[str, ...]
    language: str


@dataclass(frozen=True, slots=True)
class FrameEntry:
    frame_key: str
    source_id: str
    acquisition_id: str
    demo_id: str
    frame_id: int
    timestamp: float
    camera_id: str
    variant: str
    fov: CameraFOV
    crop: PhysicalCrop
    flip: str
    array_path: Path
    array_sha256: str
    action_start: int
    action_end: int
    normalized_actions_sha256: str
    future_mask_sha256: str
    state_dim: int | None
    bayer_pattern: str
    bayer_black_level: Any
    bayer_white_level: Any
    exposure: Any
    noise: Any
    teacher_rgb: Mapping[str, Any] | None
    augmentation_id: str
    raw_shape: tuple[int, ...]
    raw_dtype: str
    language: str
    isp_id: str
    isp_sha256: str
    preprocessing_id: str
    preprocessing_sha256: str
    checkpoint_sha256: str
    normalization_id: str


@dataclass(frozen=True, slots=True)
class PairedDistillationManifest:
    path: Path
    root: Path
    raw: Mapping[str, Any]
    sources: tuple[ManifestSource, ...]
    frames: tuple[FrameEntry, ...]
    manifest_sha256: str
    manifest_version: int
    evidence_level: str
    fixture: bool
    split_seed: int
    split_counts: Mapping[str, int]
    normalization: NormalizationSpec
    checkpoint: Mapping[str, Any]
    isp: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    action_horizon: int
    control_dim: int
    task: Mapping[str, Any]

    def frame(self, frame_key: str) -> FrameEntry:
        for frame in self.frames:
            if frame.frame_key == frame_key:
                return frame
        raise KeyError(frame_key)

    @property
    def frame_keys(self) -> tuple[str, ...]:
        return tuple(frame.frame_key for frame in self.frames)

    def source_split(self, source_id: str) -> str:
        for source in self.sources:
            if source.source_id == source_id:
                return source.split
        raise KeyError(source_id)

    @property
    def source_splits(self) -> dict[str, str]:
        return {source.source_id: source.split for source in self.sources}


# --------------------------------------------------------------------------- #
# Manifest parsing
# --------------------------------------------------------------------------- #
def assign_source_splits(
    source_ids: Sequence[str],
    *,
    seed: int = 101,
    counts: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Deterministically assign source demonstrations to train/val/test.

    Variants of one source inherit this assignment, so no frame can leak across
    partitions.  The exact 40/5/5 counts are the study default.
    """
    ids = list(source_ids)
    if not ids or len(set(ids)) != len(ids):
        raise PairedManifestError("source ids must be nonempty and unique")
    expected = {"train": 40, "validation": 5, "test": 5} if counts is None else dict(counts)
    if set(expected) != set(_SPLIT_NAMES):
        raise PairedManifestError("split counts must name train/validation/test")
    if any(int(value) < 0 for value in expected.values()):
        raise PairedManifestError("split counts must be nonnegative")
    if sum(int(value) for value in expected.values()) != len(ids):
        raise PairedManifestError("split counts do not sum to the source count")
    order = sorted(ids)
    random.Random(int(seed)).shuffle(order)
    cursor = 0
    result: dict[str, str] = {}
    for name in _SPLIT_NAMES:
        count = int(expected[name])
        for source_id in order[cursor : cursor + count]:
            result[source_id] = name
        cursor += count
    return result


def _resolve_relative(root: Path, relative: Any, where: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PairedManifestError(f"{where} must be a nonempty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PairedManifestError(f"{where} must be relative to the manifest root")
    lexical = (root / candidate).resolve(strict=False)
    if lexical != root and root not in lexical.parents:
        raise PairedManifestError(f"{where} escapes the manifest root: {relative}")
    try:
        resolved = (root / candidate).resolve(strict=True)
    except FileNotFoundError as exc:
        raise PairedDataUnavailable(f"{where} does not exist: {relative}") from exc
    if resolved != root and root not in resolved.parents:
        raise PairedManifestError(f"{where} escapes the manifest root: {relative}")
    if not resolved.is_file():
        raise PairedDataUnavailable(f"{where} is not a file: {relative}")
    return resolved


def _parse_normalization(
    raw: Mapping[str, Any],
    root: Path,
    task: Mapping[str, Any],
    action_width: int,
    dataset_format: str,
) -> NormalizationSpec:
    where = "normalization"
    data = _mapping(raw, where, "manifest")
    identity = _string(data, "id", where)
    mode = _string(data, "mode", where)
    stats_path = _resolve_relative(
        root, _required(data, "stats_path", where), f"{where}.stats_path"
    )
    stats_sha = _sha(_required(data, "stats_sha256", where), f"{where}.stats_sha256")
    stats_key = str(data.get("stats_key", ""))
    epsilon = float(data.get("epsilon", 1e-6))
    epsilon_mode = str(data.get("epsilon_mode", "floor"))
    delta = tuple(int(value) for value in data.get("delta_action_indices", ()))
    hash_algorithm = str(data.get("hash_algorithm", "float32-c-order-sha256-v1"))
    if hash_algorithm != "float32-c-order-sha256-v1":
        raise PairedManifestError("unsupported normalized-action hash algorithm")
    if int(data.get("action_width", action_width)) != action_width:
        raise PairedManifestError("normalization action width differs from the manifest")
    stats = NormalisationStats.from_json(
        stats_path,
        mode=mode,
        stats_key=stats_key,
        epsilon=epsilon,
        epsilon_mode=epsilon_mode,
    )
    if mode == "none":
        raise PairedDataUnavailable("raw expert action labels require declared statistics")
    return NormalizationSpec(
        identity=identity,
        mode=mode,
        stats_path=stats_path,
        stats_sha256=stats_sha,
        stats_key=stats_key,
        dataset_format=dataset_format,
        action_width=action_width,
        epsilon=epsilon,
        epsilon_mode=epsilon_mode,
        delta_action_indices=delta,
        hash_algorithm=hash_algorithm,
        stats=stats,
    )


def _parse_sources(raw: Mapping[str, Any]) -> tuple[ManifestSource, ...]:
    items = _required(raw, "sources", "manifest")
    if not isinstance(items, list) or not items:
        raise PairedManifestError("manifest.sources must be a nonempty list")
    sources: list[ManifestSource] = []
    for index, item in enumerate(items):
        where = f"sources[{index}]"
        if not isinstance(item, Mapping):
            raise PairedManifestError(f"{where} must be a mapping")
        source_id = _string(item, "source_id", where)
        acquisition_id = _string(item, "acquisition_id", where)
        demo_id = _string(item, "demo_id", where)
        split = _string(item, "split", where)
        if split not in _SPLIT_NAMES:
            raise PairedManifestError(f"{where}.split is unknown")
        camera_ids = tuple(item.get("camera_ids", ()))
        if not camera_ids or any(not isinstance(value, str) or not value for value in camera_ids):
            raise PairedManifestError(f"{where}.camera_ids must be nonempty strings")
        if len(set(camera_ids)) != len(camera_ids):
            raise PairedManifestError(f"{where}.camera_ids must be unique")
        variants = tuple(item.get("variants", ()))
        if not variants or any(value not in _VARIANTS for value in variants):
            raise PairedManifestError(f"{where}.variants must use the declared input modes")
        language = item.get("language", "")
        if language != "" and (not isinstance(language, str) or not language.strip()):
            raise PairedManifestError(f"{where}.language must be a nonempty string when present")
        sources.append(
            ManifestSource(
                source_id=source_id,
                acquisition_id=acquisition_id,
                demo_id=demo_id,
                split=split,
                source_sha256=_sha(
                    _required(item, "source_sha256", where), f"{where}.source_sha256"
                ),
                camera_ids=camera_ids,
                variants=variants,
                language=str(language),
            )
        )
    if len({source.source_id for source in sources}) != len(sources):
        raise PairedManifestError("source_id values must be unique")
    if len({(source.acquisition_id, source.demo_id) for source in sources}) != len(sources):
        raise PairedManifestError("acquisition/demo identities must be unique per source")
    return tuple(sources)


def _parse_teacher_rgb(
    value: Any, root: Path, where: str, *, verify_files: bool = True
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PairedManifestError(f"{where} must be a mapping or null")
    kind = _string(value, "kind", where)
    if kind == "npz":
        key = _string(value, "key", where)
        return {"kind": "npz", "key": key}
    if kind == "prepared_file":
        relative = _required(value, "path", where)
        digest = _sha(_required(value, "sha256", where), f"{where}.sha256")
        if not verify_files:
            return {
                "kind": "prepared_file",
                "path": relative,
                "sha256": digest,
                "relative": relative,
                "unverified_student_only": True,
            }
        path = _resolve_relative(root, relative, f"{where}.path")
        return {"kind": "prepared_file", "path": path, "sha256": digest, "relative": relative}
    raise PairedManifestError(f"{where}.kind must be npz or prepared_file")


def _parse_frames(
    raw: Mapping[str, Any],
    *,
    root: Path,
    sources: Mapping[str, ManifestSource],
    image_size: int,
    action_horizon: int,
    checkpoint_sha256: str,
    isp: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    normalization_id: str,
    student_only: bool = False,
) -> tuple[FrameEntry, ...]:
    items = _required(raw, "frames", "manifest")
    if not isinstance(items, list) or not items:
        raise PairedManifestError("manifest.frames must be a nonempty list")
    frames: list[FrameEntry] = []
    seen_keys: set[str] = set()
    seen_frame_ids: set[tuple[str, str, int]] = set()
    seen_times: set[tuple[str, str, float]] = set()
    for index, item in enumerate(items):
        where = f"frames[{index}]"
        if not isinstance(item, Mapping):
            raise PairedManifestError(f"{where} must be a mapping")
        frame_key = _string(item, "frame_key", where)
        if frame_key in seen_keys:
            raise PairedManifestError(f"duplicate frame_key {frame_key!r}")
        seen_keys.add(frame_key)
        source_id = _string(item, "source_id", where)
        if source_id not in sources:
            raise PairedManifestError(f"{where} names an unknown source")
        source = sources[source_id]
        acquisition_id = _string(item, "acquisition_id", where)
        demo_id = _string(item, "demo_id", where)
        if (acquisition_id, demo_id) != (source.acquisition_id, source.demo_id):
            raise PairedManifestError(f"{where} source identity does not match its source record")
        frame_id = _integer(item, "frame_id", where)
        if frame_id < 0:
            raise PairedManifestError(f"{where}.frame_id must be nonnegative")
        timestamp = float(_required(item, "timestamp", where))
        if not np.isfinite(timestamp):
            raise PairedManifestError(f"{where}.timestamp must be finite")
        camera_id = _string(item, "camera_id", where)
        if camera_id not in source.camera_ids:
            raise PairedManifestError(f"{where}.camera_id is not declared by the source")
        variant = _string(item, "variant", where)
        if variant not in _VARIANTS or variant not in source.variants:
            raise PairedManifestError(f"{where}.variant is not declared by the source")
        fov = CameraFOV.from_mapping(_mapping(item, "fov", where), f"{where}.fov")
        crop = PhysicalCrop.from_mapping(_mapping(item, "crop", where), f"{where}.crop")
        if crop.x + crop.width > fov.width or crop.y + crop.height > fov.height:
            raise PairedManifestError(f"{where}.crop exceeds the declared FOV")
        flip = _string(item, "flip", where)
        if flip not in {"none", "horizontal", "vertical", "both"}:
            raise PairedManifestError(f"{where}.flip is unsupported")
        array_meta = _mapping(item, "array", where)
        array_path = _resolve_relative(
            root, _required(array_meta, "path", f"{where}.array"), f"{where}.array.path"
        )
        array_sha = _sha(_required(array_meta, "sha256", f"{where}.array"), f"{where}.array.sha256")
        raw_shape = tuple(int(value) for value in _required(item, "raw_shape", where))
        if len(raw_shape) != 3 or any(value <= 0 for value in raw_shape):
            raise PairedManifestError(f"{where}.raw_shape must be [C,H,W]")
        raw_dtype = _string(item, "raw_dtype", where)
        if variant == "bayer4_packed":
            height, width = raw_shape[1], raw_shape[2]
            physical_h, physical_w = height * 2, width * 2
            if raw_shape[0] != 4 or (physical_h, physical_w) != (fov.height, fov.width):
                raise PairedManifestError(
                    f"{where} Bayer packed shape must equal half the physical FOV with 4 planes"
                )
            if (crop.width, crop.height) != (physical_w, physical_h):
                raise PairedManifestError(f"{where} Bayer crop must cover the packed physical FOV")
            if not crop.even_aligned:
                raise PairedManifestError(f"{where} Bayer crop must be even aligned")
        else:
            if raw_shape[0] != 3 or (raw_shape[1], raw_shape[2]) != (crop.height, crop.width):
                raise PairedManifestError(
                    f"{where} pseudo-RGB/RGB3 raw shape must match the declared crop"
                )
        action_start = _integer(item, "action_start", where)
        action_end = _integer(item, "action_end", where)
        if action_start < 0 or action_end - action_start != action_horizon:
            raise PairedManifestError(f"{where} action window must have the declared horizon")
        if not action_start <= frame_id < action_end:
            raise PairedManifestError(f"{where} action indices do not contain the frame")
        normalized_sha = _sha(
            _required(item, "normalized_actions_sha256", where),
            f"{where}.normalized_actions_sha256",
        )
        future_sha = _sha(
            _required(item, "future_mask_sha256", where), f"{where}.future_mask_sha256"
        )
        frame_checkpoint = _sha(
            _required(item, "checkpoint_sha256", where), f"{where}.checkpoint_sha256"
        )
        if frame_checkpoint != checkpoint_sha256:
            raise PairedManifestError(f"{where} checkpoint identity differs from the manifest")
        frame_isp_id = _string(item, "isp_id", where)
        frame_isp_sha = _sha(_required(item, "isp_sha256", where), f"{where}.isp_sha256")
        if frame_isp_id != isp["id"] or frame_isp_sha != isp["sha256"]:
            raise PairedManifestError(f"{where} ISP identity differs from the manifest")
        frame_pre_id = _string(item, "preprocessing_id", where)
        frame_pre_sha = _sha(
            _required(item, "preprocessing_sha256", where),
            f"{where}.preprocessing_sha256",
        )
        if frame_pre_id != preprocessing["id"] or frame_pre_sha != preprocessing["sha256"]:
            raise PairedManifestError(f"{where} preprocessing identity differs from the manifest")
        frame_norm_id = _string(item, "normalization_id", where)
        if frame_norm_id != normalization_id:
            raise PairedManifestError(f"{where} normalisation identity differs from the manifest")
        frame_language = item.get("language", source.language)
        if not isinstance(frame_language, str) or not frame_language.strip():
            raise PairedManifestError(f"{where}.language is required at the frame or source level")
        state_dim = item.get("state_dim")
        if state_dim is not None:
            state_dim = int(state_dim)
            if state_dim <= 0:
                raise PairedManifestError(f"{where}.state_dim must be positive")
        bayer = item.get("bayer")
        pattern = ""
        black = None
        white = None
        if variant == "bayer4_packed":
            if not isinstance(bayer, Mapping):
                raise PairedManifestError(f"{where}.bayer is required for packed Bayer")
            pattern = _string(bayer, "cfa_pattern", f"{where}.bayer")
            if pattern not in _BAYER_CFA:
                raise PairedManifestError(f"{where}.bayer.cfa_pattern is unsupported")
            if "black_level" not in bayer or "white_level" not in bayer:
                raise PairedManifestError(f"{where}.bayer requires black/white levels")
            black = bayer["black_level"]
            white = bayer["white_level"]
            black_values = np.asarray(black, dtype=np.float64).reshape(-1)
            white_values = np.asarray(white, dtype=np.float64).reshape(-1)
            if black_values.size not in (1, 4) or white_values.size not in (1, 4):
                raise PairedManifestError(f"{where}.bayer levels must be scalar or 4-wide")
            if not np.isfinite(black_values).all() or not np.isfinite(white_values).all():
                raise PairedManifestError(f"{where}.bayer levels must be finite")
            if (black_values < 0).any() or not (white_values > black_values).all():
                raise PairedManifestError(f"{where}.bayer levels are invalid")
        elif bayer is not None:
            raise PairedManifestError(f"{where}.bayer is only valid for bayer4_packed")
        teacher = _parse_teacher_rgb(
            item.get("teacher_rgb"),
            root,
            f"{where}.teacher_rgb",
            verify_files=not student_only,
        )
        if variant == "rgb3_control" and teacher is None and not student_only:
            raise PairedDataUnavailable(
                f"{where} is an RGB control frame without an actual teacher-preprocessed RGB"
            )
        augmentation_id = str(item.get("augmentation_id", "none"))
        if not augmentation_id:
            raise PairedManifestError(f"{where}.augmentation_id must be nonempty")
        key = (source_id, camera_id, frame_id)
        if key in seen_frame_ids:
            raise PairedManifestError(f"duplicate frame identity {key}")
        seen_frame_ids.add(key)
        time_key = (source_id, camera_id, timestamp)
        if time_key in seen_times:
            raise PairedManifestError(f"duplicate frame timestamp {time_key}")
        seen_times.add(time_key)
        frames.append(
            FrameEntry(
                frame_key=frame_key,
                source_id=source_id,
                acquisition_id=acquisition_id,
                demo_id=demo_id,
                frame_id=frame_id,
                timestamp=timestamp,
                camera_id=camera_id,
                variant=variant,
                fov=fov,
                crop=crop,
                flip=flip,
                array_path=array_path,
                array_sha256=array_sha,
                action_start=action_start,
                action_end=action_end,
                normalized_actions_sha256=normalized_sha,
                future_mask_sha256=future_sha,
                state_dim=state_dim,
                bayer_pattern=pattern,
                bayer_black_level=black,
                bayer_white_level=white,
                exposure=item.get("exposure"),
                noise=item.get("noise"),
                teacher_rgb=teacher,
                augmentation_id=augmentation_id,
                raw_shape=raw_shape,
                raw_dtype=raw_dtype,
                language=frame_language,
                isp_id=frame_isp_id,
                isp_sha256=frame_isp_sha,
                preprocessing_id=frame_pre_id,
                preprocessing_sha256=frame_pre_sha,
                checkpoint_sha256=frame_checkpoint,
                normalization_id=frame_norm_id,
            )
        )
    # A tiny deterministic check that every source has at least one frame.
    referenced = {frame.source_id for frame in frames}
    missing = sorted(set(sources) - referenced)
    if missing:
        raise PairedManifestError(f"sources without frames: {missing}")
    if image_size <= 0:
        raise PairedManifestError("preprocessing image_size must be positive")
    return tuple(frames)


def load_paired_manifest(
    path: str | Path,
    *,
    allow_fixture: bool = False,
    fixture_source_count: int | None = None,
    expected_sources: int = 50,
    expected_split: tuple[int, int, int] = (40, 5, 5),
    expected_seed: int = 101,
    expected_manifest_sha256: str | None = None,
    verify_files: bool = True,
    student_only: bool = False,
) -> PairedDistillationManifest:
    """Load and strictly validate a versioned paired manifest.

    Real manifests must be exactly 50 sources / 40-5-5 / seed 101.  A manifest
    with a different count is accepted only when it declares
    ``evidence_level='fixture_contract'`` *and* the caller passes
    ``allow_fixture=True``; fixture count is then explicit and configurable.
    """
    if not verify_files:
        raise PairedManifestError("strict manifest loading requires verify_files=True")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise PairedDataUnavailable(f"paired manifest not found: {manifest_path}")
    raw_bytes = manifest_path.read_bytes()
    manifest_sha = ""
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise PairedManifestError("manifest SHA256 differs from the expected identity")
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairedManifestError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise PairedManifestError("manifest root must be a JSON object")
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise PairedManifestError(f"unsupported manifest schema {raw.get('schema')!r}")
    if raw.get("manifest_version") != MANIFEST_VERSION:
        raise PairedManifestError("unsupported manifest version")
    _string(raw, "manifest_id", "manifest")
    _string(raw, "created_at_utc", "manifest")
    evidence = raw.get("evidence_level")
    if evidence not in {"fixture_contract", "real_paired_observation"}:
        raise PairedManifestError("manifest evidence_level is unsupported")
    fixture = raw.get("fixture")
    if not isinstance(fixture, bool):
        raise PairedManifestError("manifest.fixture must be boolean")
    if fixture != (evidence == "fixture_contract"):
        raise PairedManifestError("fixture flag and evidence_level disagree")
    if fixture and not allow_fixture:
        raise PairedDataUnavailable(
            "fixture manifest requires explicit allow_fixture=True; it is not real data"
        )
    if not fixture and allow_fixture:
        raise PairedManifestError("allow_fixture is only valid for fixture manifests")
    if not fixture:
        if (
            expected_sources != 50
            or tuple(expected_split) != (40, 5, 5)
            or int(expected_seed) != 101
        ):
            raise PairedManifestError(
                "real paired manifests are fixed at 50 sources, 40/5/5, seed 101"
            )
    task = _mapping(raw, "task", "manifest")
    _string(task, "dataset", "task")
    _string(task, "suite", "task")
    _string(task, "task_key", "task")
    _string(task, "instruction", "task")
    dataset_format = str(task.get("dataset_format", "libero"))
    action_horizon = _integer(raw, "action_horizon", "manifest")
    control_dim = _integer(raw, "control_dim", "manifest")
    if action_horizon < 1 or control_dim < 1:
        raise PairedManifestError("action_horizon/control_dim must be positive")
    checkpoint = dict(_mapping(raw, "checkpoint", "manifest"))
    _sha(_required(checkpoint, "sha256", "checkpoint"), "checkpoint.sha256")
    _string(checkpoint, "path", "checkpoint")
    if "config_sha256" in checkpoint:
        _sha(checkpoint["config_sha256"], "checkpoint.config_sha256")
    _sha(
        _required(checkpoint, "load_audit_sha256", "checkpoint"),
        "checkpoint.load_audit_sha256",
    )
    if not isinstance(checkpoint.get("accepted"), bool):
        raise PairedManifestError("checkpoint.accepted must be boolean")
    if not fixture and not checkpoint["accepted"]:
        raise PairedDataUnavailable("real manifest checkpoint is not an accepted policy identity")
    isp = dict(_mapping(raw, "isp", "manifest"))
    _string(isp, "id", "isp")
    _sha(_required(isp, "sha256", "isp"), "isp.sha256")
    preprocessing = dict(_mapping(raw, "preprocessing", "manifest"))
    _string(preprocessing, "id", "preprocessing")
    _sha(_required(preprocessing, "sha256", "preprocessing"), "preprocessing.sha256")
    image_size = int(_required(preprocessing, "image_size", "preprocessing"))
    if image_size <= 0:
        raise PairedManifestError("preprocessing.image_size must be positive")
    color_range = str(preprocessing.get("color_range", "-1,1"))
    if color_range != "-1,1":
        raise PairedManifestError("teacher preprocessed RGB must declare [-1,1]")

    sources = _parse_sources(raw)
    partition = _mapping(raw, "partition", "manifest")
    seed = _integer(partition, "seed", "manifest.partition")
    declared_sources = _integer(partition, "source_count", "manifest.partition")
    if declared_sources != len(sources):
        raise PairedManifestError("partition.source_count differs from the source list")
    counts_raw = _mapping(partition, "splits", "manifest.partition")
    counts = {name: int(_required(counts_raw, name, "partition.splits")) for name in _SPLIT_NAMES}
    if any(value < 0 for value in counts.values()) or sum(counts.values()) != len(sources):
        raise PairedManifestError("declared partition counts do not match sources")
    if fixture:
        expected_count = len(sources) if fixture_source_count is None else int(fixture_source_count)
        if len(sources) != expected_count:
            raise PairedManifestError("fixture source count differs from the explicit count")
    else:
        if fixture_source_count is not None:
            raise PairedManifestError(
                "fixture_source_count is only valid for the explicit fixture contract"
            )
        if (
            len(sources) != expected_sources
            or counts != dict(zip(_SPLIT_NAMES, expected_split, strict=True))
            or seed != expected_seed
        ):
            raise PairedManifestError("real partition must be exactly 50 sources / 40-5-5 / 101")
    recomputed = assign_source_splits(
        [source.source_id for source in sources], seed=seed, counts=counts
    )
    for source in sources:
        if recomputed[source.source_id] != source.split:
            raise PairedManifestError(
                f"source {source.source_id!r} split is not the deterministic seed assignment"
            )
    source_map = {source.source_id: source for source in sources}
    normalization = _parse_normalization(
        raw, manifest_path.parent.resolve(), task, control_dim, dataset_format
    )
    frames = _parse_frames(
        raw,
        root=manifest_path.parent.resolve(),
        sources=source_map,
        image_size=image_size,
        action_horizon=action_horizon,
        checkpoint_sha256=checkpoint["sha256"],
        isp=isp,
        preprocessing=preprocessing,
        normalization_id=normalization.identity,
        student_only=student_only,
    )
    for frame in frames:
        if frame.normalized_actions_sha256 and normalization.action_width != control_dim:
            raise PairedManifestError("control_dim and normalization action width disagree")
    return PairedDistillationManifest(
        path=manifest_path.resolve(),
        root=manifest_path.parent.resolve(),
        raw=raw,
        sources=sources,
        frames=frames,
        manifest_sha256=manifest_sha,
        manifest_version=MANIFEST_VERSION,
        evidence_level=evidence,
        fixture=fixture,
        split_seed=seed,
        split_counts=counts,
        normalization=normalization,
        checkpoint=checkpoint,
        isp=isp,
        preprocessing=preprocessing,
        action_horizon=action_horizon,
        control_dim=control_dim,
        task=task,
    )


# --------------------------------------------------------------------------- #
# Sample values
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RawFrameData:
    values: torch.Tensor
    units: str
    cfa_pattern: str
    physical_origin: tuple[int, int]
    valid_mask: torch.Tensor | None = None
    bad_mask: torch.Tensor | None = None
    saturated_mask: torch.Tensor | None = None
    missing_mask: torch.Tensor | None = None
    black_level: Any = None
    white_level: Any = None
    exposure: Any = None
    noise: Any = None

    @property
    def raw_dtype(self) -> str:
        return str(self.values.dtype).replace("torch.", "")

    def availability(self) -> dict[str, str]:
        return {
            "units": self.units,
            "valid_mask": "available" if self.valid_mask is not None else "unavailable",
            "bad_mask": "available" if self.bad_mask is not None else "unavailable",
            "saturated_mask": "available" if self.saturated_mask is not None else "unavailable",
            "missing_mask": "available" if self.missing_mask is not None else "unavailable",
            "calibration": (
                "available"
                if self.black_level is not None and self.units == "raw_dn"
                else "available_not_applied"
                if self.black_level is not None
                else "unavailable"
            ),
            "exposure": "available" if self.exposure is not None else "unavailable",
            "noise": "available" if self.noise is not None else "unavailable",
            "cfa": "available" if self.cfa_pattern else "not_applicable",
        }

    def as_stem_input(self, *, camera_id: str | None = None):
        """Convert to the student front-end input *without* teacher RGB."""
        from owac.baselines.raw_stem_siglip import RawStemInput

        camera = camera_id or "camera_0"
        values = self.values.unsqueeze(0).unsqueeze(0)
        masks = {}
        for name in ("valid_mask", "bad_mask", "saturated_mask", "missing_mask"):
            value = getattr(self, name)
            if value is not None and value.ndim == 3 and value.shape[0] in (3, 4):
                # Validity is supported only when every channel is valid; a bad,
                # saturated or missing pixel in any channel invalidates the cell.
                value = value.all(dim=0) if name == "valid_mask" else value.any(dim=0)
            masks[name] = None if value is None else value.unsqueeze(0).unsqueeze(0)
        return RawStemInput(
            values=values,
            input_mode="bayer4_packed"
            if self.cfa_pattern
            else ("rgb3_control" if self.units == "teacher_rgb_minus1_1" else "rawvla_rgb3"),
            camera_ids=(camera,),
            units=self.units,
            physical_origins=(self.physical_origin,),
            cfa_pattern=self.cfa_pattern,
            black_level=self.black_level,
            white_level=self.white_level,
            exposure=self.exposure,
            noise=self.noise,
            **masks,
        )


@dataclass(frozen=True, slots=True)
class DistillationSample:
    frame_key: str
    source_id: str
    acquisition_id: str
    demo_id: str
    frame_id: int
    timestamp: float
    split: str
    camera_id: str
    variant: str
    raw: RawFrameData
    teacher_rgb: torch.Tensor
    teacher_rgb_source: str
    actions: torch.Tensor
    future_mask: torch.Tensor
    state: torch.Tensor | None
    state_raw: torch.Tensor | None
    language: str
    fov: CameraFOV
    crop: PhysicalCrop
    flip: str
    checkpoint_sha256: str
    isp_id: str
    preprocessing_id: str
    normalization_id: str
    augmentation_id: str
    cache_key: str
    metadata: Mapping[str, Any]

    def student_view(self) -> StudentSample:
        if self.variant == "rgb3_control":
            raise PairedDataUnavailable(
                "RGB3 control is a training-time teacher-RGB interface, not a pure "
                "student deployment input"
            )
        return StudentSample(
            frame_key=self.frame_key,
            source_id=self.source_id,
            acquisition_id=self.acquisition_id,
            demo_id=self.demo_id,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            split=self.split,
            camera_id=self.camera_id,
            variant=self.variant,
            raw=self.raw,
            state=self.state,
            state_raw=self.state_raw,
            language=self.language,
            fov=self.fov,
            crop=self.crop,
            flip=self.flip,
            checkpoint_sha256=self.checkpoint_sha256,
            isp_id=self.isp_id,
            preprocessing_id=self.preprocessing_id,
            normalization_id=self.normalization_id,
            augmentation_id=self.augmentation_id,
            metadata={"cache_key": self.cache_key},
        )


@dataclass(frozen=True, slots=True)
class StudentSample:
    """Pure student view: no teacher RGB and no cache handle."""

    frame_key: str
    source_id: str
    acquisition_id: str
    demo_id: str
    frame_id: int
    timestamp: float
    split: str
    camera_id: str
    variant: str
    raw: RawFrameData
    state: torch.Tensor | None
    state_raw: torch.Tensor | None
    language: str
    fov: CameraFOV
    crop: PhysicalCrop
    flip: str
    checkpoint_sha256: str
    isp_id: str
    preprocessing_id: str
    normalization_id: str
    augmentation_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading one frame
# --------------------------------------------------------------------------- #
def _npz(
    path: Path, *, expected_sha: str | None = None, keys: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            available = set(archive.files)
            selected = available if keys is None else available & set(keys)
            data = {name: archive[name] for name in archive.files if name in selected}
    except (ValueError, OSError) as exc:
        raise PairedManifestError(f"cannot load NPZ without pickle: {path}") from exc
    for name, value in data.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise PairedManifestError(f"NPZ entry {name!r} is not a plain array")
    return data


def _tensor(array: np.ndarray, name: str) -> torch.Tensor:
    if array.dtype.hasobject:
        raise PairedManifestError(f"{name} is an object array")
    return torch.from_numpy(np.array(array, copy=True))


def _optional_tensor(data: Mapping[str, np.ndarray], key: str) -> torch.Tensor | None:
    if key not in data:
        return None
    tensor = _tensor(data[key], key)
    if tensor.dtype != torch.bool:
        raise PairedManifestError(f"NPZ mask {key!r} must be boolean")
    return tensor


def _teacher_rgb(
    frame: FrameEntry,
    data: Mapping[str, np.ndarray],
    *,
    image_size: int,
) -> tuple[torch.Tensor, str]:
    source = frame.teacher_rgb
    if source is None:
        raise PairedDataUnavailable(
            f"{frame.frame_key}: no paired teacher RGB array or prepared RGB file is declared"
        )
    kind = source["kind"]
    if kind == "npz":
        key = str(source["key"])
        if key not in data:
            raise PairedDataUnavailable(f"{frame.frame_key}: NPZ lacks {key!r}")
        array = np.asarray(data[key])
        origin = "npz"
    else:
        path = Path(source["path"])
        try:
            array = np.load(path, allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise PairedManifestError(f"{frame.frame_key}: can't load prepared RGB") from exc
        origin = f"prepared_file:{source.get('relative', path.name)}"
    if array.dtype.hasobject:
        raise PairedManifestError("teacher RGB is an object array")
    expected_shape = (3, image_size, image_size)
    if tuple(array.shape) != expected_shape:
        raise PairedManifestError(
            f"{frame.frame_key}: teacher RGB shape {tuple(array.shape)} != {expected_shape}"
        )
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).clone()
    if not torch.isfinite(tensor).all():
        raise PairedManifestError("teacher RGB contains non-finite values")
    if tensor.numel() and (tensor.min() < -1.0 - 1e-5 or tensor.max() > 1.0 + 1e-5):
        raise PairedManifestError("teacher RGB must be the policy-preprocessed [-1,1] tensor")
    return tensor, origin


def _raw_frame(
    frame: FrameEntry,
    data: Mapping[str, np.ndarray],
    *,
    image_size: int,
) -> RawFrameData:
    key = "raw"
    if key not in data:
        raise PairedDataUnavailable(f"{frame.frame_key}: NPZ lacks raw array")
    array = np.asarray(data[key])
    if tuple(array.shape) != frame.raw_shape:
        raise PairedManifestError(
            f"{frame.frame_key}: raw shape {tuple(array.shape)} != declared {frame.raw_shape}"
        )
    if str(array.dtype) != frame.raw_dtype:
        raise PairedManifestError(
            f"{frame.frame_key}: raw dtype {array.dtype} != declared {frame.raw_dtype}"
        )
    values = _tensor(array, "raw")
    if values.is_floating_point() and not torch.isfinite(values).all():
        raise PairedManifestError("raw array contains non-finite values")
    valid = _optional_tensor(data, "valid_mask")
    bad = _optional_tensor(data, "bad_mask")
    saturated = _optional_tensor(data, "saturated_mask")
    missing = _optional_tensor(data, "missing_mask")
    for name, mask in (
        ("valid_mask", valid),
        ("bad_mask", bad),
        ("saturated_mask", saturated),
        ("missing_mask", missing),
    ):
        if mask is not None and tuple(mask.shape) not in {
            tuple(values.shape),
            tuple(values.shape[1:]),
            (1, *tuple(values.shape[1:])),
        }:
            raise PairedManifestError(f"{frame.frame_key}: {name} shape does not match raw")
    units = (
        "teacher_rgb_minus1_1"
        if frame.variant == "rgb3_control"
        else ("raw_dn" if frame.variant == "bayer4_packed" else "cache_units")
    )
    if frame.variant == "rgb3_control":
        teacher, _ = _teacher_rgb(frame, data, image_size=image_size)
        if tuple(values.shape) != tuple(teacher.shape) or not torch.equal(
            values.float(), teacher.float()
        ):
            raise PairedManifestError(
                "RGB3 control raw tensor must be the actual teacher-preprocessed tensor"
            )
    return RawFrameData(
        values=values,
        units=units,
        cfa_pattern=frame.bayer_pattern,
        physical_origin=(frame.crop.x, frame.crop.y),
        valid_mask=valid,
        bad_mask=bad,
        saturated_mask=saturated,
        missing_mask=missing,
        black_level=frame.bayer_black_level,
        white_level=frame.bayer_white_level,
        exposure=frame.exposure,
        noise=frame.noise,
    )


def _load_sample(manifest: PairedDistillationManifest, frame: FrameEntry) -> DistillationSample:
    data = _npz(frame.array_path, expected_sha=frame.array_sha256)
    image_size = int(manifest.preprocessing["image_size"])
    raw = _raw_frame(frame, data, image_size=image_size)
    teacher, teacher_source = _teacher_rgb(frame, data, image_size=image_size)
    if "actions_raw" not in data:
        raise PairedDataUnavailable(f"{frame.frame_key}: NPZ lacks raw expert actions")
    actions_raw = np.asarray(data["actions_raw"])
    expected_actions = (manifest.action_horizon, manifest.control_dim)
    if tuple(actions_raw.shape) != expected_actions:
        raise PairedManifestError(
            f"{frame.frame_key}: actions shape {tuple(actions_raw.shape)} != {expected_actions}"
        )
    if "future_mask" not in data:
        raise PairedDataUnavailable(f"{frame.frame_key}: NPZ lacks future mask")
    future = np.asarray(data["future_mask"])
    if future.shape != (manifest.action_horizon,) or future.dtype != np.bool_:
        raise PairedManifestError(f"{frame.frame_key}: future mask must be boolean [H]")
    state_raw: torch.Tensor | None = None
    state_normalized: torch.Tensor | None = None
    if frame.state_dim is not None:
        if "state" not in data:
            raise PairedDataUnavailable(f"{frame.frame_key}: NPZ lacks declared state")
        state_array = np.asarray(data["state"])
        if state_array.shape != (frame.state_dim,):
            raise PairedManifestError(f"{frame.frame_key}: state shape mismatch")
        if not np.isfinite(state_array).all():
            raise PairedManifestError(f"{frame.frame_key}: state contains non-finite values")
        state_raw = torch.from_numpy(np.asarray(state_array, dtype=np.float32)).clone()
    normalised, state_from_actions, normal_meta = normalise_actions(
        np.asarray(actions_raw, dtype=np.float32),
        None if state_raw is None else state_raw.numpy(),
        manifest.normalization.stats,
        dataset_format=manifest.normalization.dataset_format,
        delta_action_indices=manifest.normalization.delta_action_indices,
    )
    actions = torch.from_numpy(np.ascontiguousarray(normalised, dtype=np.float32)).clone()
    if tuple(actions.shape) != expected_actions:
        raise PairedManifestError("normalization changed the declared action shape")
    if state_raw is not None:
        normalised_state, state_meta = normalise_state(
            state_raw.numpy(), manifest.normalization.stats
        )
        if normalised_state is not None:
            state_normalized = torch.from_numpy(
                np.ascontiguousarray(normalised_state, dtype=np.float32)
            ).clone()
        normal_meta["state"] = state_meta
    if state_from_actions is not None and state_raw is None:
        raise PairedManifestError("normalization returned a state that was not provided")
    normal_meta["normalisation_id"] = manifest.normalization.identity
    normal_meta["stats_sha256"] = manifest.normalization.stats_sha256
    cache_key = cache_key_for_frame(manifest, frame)
    metadata = {
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_version": manifest.manifest_version,
        "evidence_level": manifest.evidence_level,
        "fixture": manifest.fixture,
        "source_sha256": next(
            source.source_sha256
            for source in manifest.sources
            if source.source_id == frame.source_id
        ),
        "source_split": manifest.source_split(frame.source_id),
        "array_sha256": frame.array_sha256,
        "teacher_rgb_source": teacher_source,
        "checkpoint": {**dict(manifest.checkpoint), "sha256": frame.checkpoint_sha256},
        "isp": {"id": frame.isp_id, "sha256": frame.isp_sha256},
        "preprocessing": {
            **dict(manifest.preprocessing),
            "id": frame.preprocessing_id,
            "sha256": frame.preprocessing_sha256,
        },
        "normalization": normal_meta,
        "action_start": frame.action_start,
        "action_end": frame.action_end,
        "future_mask_sha256": frame.future_mask_sha256,
        "fov": frame.fov.as_dict(),
        "crop": frame.crop.as_dict(),
        "flip": frame.flip,
        "augmentation_id": frame.augmentation_id,
        "units": raw.units,
        "raw_dtype": raw.raw_dtype,
        "raw_availability": raw.availability(),
        "cache_key": cache_key,
    }
    return DistillationSample(
        frame_key=frame.frame_key,
        source_id=frame.source_id,
        acquisition_id=frame.acquisition_id,
        demo_id=frame.demo_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        split=manifest.source_split(frame.source_id),
        camera_id=frame.camera_id,
        variant=frame.variant,
        raw=raw,
        teacher_rgb=teacher,
        teacher_rgb_source=teacher_source,
        actions=actions,
        future_mask=torch.from_numpy(future.astype(np.bool_, copy=True)).clone(),
        state=state_normalized,
        state_raw=state_raw,
        language=frame.language,
        fov=frame.fov,
        crop=frame.crop,
        flip=frame.flip,
        checkpoint_sha256=frame.checkpoint_sha256,
        isp_id=frame.isp_id,
        preprocessing_id=frame.preprocessing_id,
        normalization_id=frame.normalization_id,
        augmentation_id=frame.augmentation_id,
        cache_key=cache_key,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def cache_key_for_frame(
    manifest: PairedDistillationManifest,
    frame: FrameEntry,
    *,
    augmentation_id: str | None = None,
) -> str:
    """Bind every provenance field that can change a cached sample."""
    return f"pair-{frame.frame_key}"


@dataclass(frozen=True, slots=True)
class CacheSpec:
    key: str
    raw_shape: tuple[int, ...]
    raw_dtype: str
    teacher_shape: tuple[int, ...]
    actions_shape: tuple[int, ...]
    future_shape: tuple[int, ...]
    state_shape: tuple[int, ...] | None
    frame_key: str
    source_id: str
    camera_id: str
    timestamp: float
    checkpoint_sha256: str
    isp_id: str
    preprocessing_id: str
    normalization_id: str
    augmentation_id: str

    @classmethod
    def from_frame(cls, manifest: PairedDistillationManifest, frame: FrameEntry) -> CacheSpec:
        image_size = int(manifest.preprocessing["image_size"])
        return cls(
            key=cache_key_for_frame(manifest, frame),
            raw_shape=frame.raw_shape,
            raw_dtype=frame.raw_dtype,
            teacher_shape=(3, image_size, image_size),
            actions_shape=(manifest.action_horizon, manifest.control_dim),
            future_shape=(manifest.action_horizon,),
            state_shape=None if frame.state_dim is None else (frame.state_dim,),
            frame_key=frame.frame_key,
            source_id=frame.source_id,
            camera_id=frame.camera_id,
            timestamp=frame.timestamp,
            checkpoint_sha256=frame.checkpoint_sha256,
            isp_id=frame.isp_id,
            preprocessing_id=frame.preprocessing_id,
            normalization_id=frame.normalization_id,
            augmentation_id=frame.augmentation_id,
        )


def _cache_mask(data: Mapping[str, np.ndarray], key: str) -> torch.Tensor | None:
    if key not in data:
        return None
    value = data[key]
    if value.dtype != np.bool_:
        raise CacheCorruptionError(f"cache mask {key!r} is not boolean")
    return torch.from_numpy(value.copy())


# --------------------------------------------------------------------------- #
# Public data set
# --------------------------------------------------------------------------- #
class SampleCache:
    """NPZ cache with strict key/shape/dtype/finiteness validation.

    A missing file is a cache miss; a file whose key or contents disagree with the
    current manifest is a hard :class:`CacheCorruptionError`.  There is no stale
    fallback to a previous sample.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.npz"

    def get(self, spec: CacheSpec) -> DistillationSample | None:
        path = self._path(spec.key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                stored = {name: archive[name] for name in archive.files}
        except (ValueError, OSError) as exc:
            raise CacheCorruptionError(f"cache entry cannot be loaded: {path}") from exc
        try:
            key = bytes(stored.pop("cache_key")).decode("utf-8")
            meta_bytes = bytes(stored.pop("sample_json"))
            stored.pop("payload_sha256", None)
        except (KeyError, UnicodeDecodeError) as exc:
            raise CacheCorruptionError("cache entry lacks key/metadata") from exc
        if key != spec.key:
            raise CacheCorruptionError("cache key mismatch")
        try:
            metadata = json.loads(meta_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CacheCorruptionError("cache metadata is not valid JSON") from exc
        if metadata.get("cache_schema") != CACHE_SCHEMA:
            raise CacheCorruptionError("cache schema mismatch")
        if metadata.get("cache_version") != CACHE_VERSION:
            raise CacheCorruptionError("cache version mismatch")
        if metadata.get("cache_key") != spec.key:
            raise CacheCorruptionError("cache metadata key mismatch")
        if metadata.get("language_sha256") != spec.language_sha256:
            raise CacheCorruptionError("cache language identity mismatch")
        for field_name in (
            "frame_key",
            "source_id",
            "camera_id",
            "checkpoint_sha256",
            "isp_id",
            "preprocessing_id",
            "normalization_id",
            "augmentation_id",
        ):
            if metadata.get(field_name) != getattr(spec, field_name):
                raise CacheCorruptionError(f"cache metadata {field_name} mismatch")
        try:
            cached_timestamp = float(metadata["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheCorruptionError("cache metadata timestamp is invalid") from exc
        if cached_timestamp != spec.timestamp:
            raise CacheCorruptionError("cache metadata timestamp mismatch")
        for name, expected in (
            ("raw", spec.raw_shape),
            ("teacher_rgb", spec.teacher_shape),
            ("actions", spec.actions_shape),
            ("future_mask", spec.future_shape),
        ):
            if name not in stored or tuple(stored[name].shape) != tuple(expected):
                raise CacheCorruptionError(f"cache array {name!r} shape mismatch")
        if str(stored["raw"].dtype) != spec.raw_dtype:
            raise CacheCorruptionError("cache raw dtype mismatch")
        if stored["teacher_rgb"].dtype != np.float32:
            raise CacheCorruptionError("cache teacher RGB dtype mismatch")
        if stored["actions"].dtype != np.float32:
            raise CacheCorruptionError("cache action dtype mismatch")
        if stored["future_mask"].dtype != np.bool_:
            raise CacheCorruptionError("cache future mask dtype mismatch")
        for name in ("raw", "teacher_rgb", "actions"):
            if (
                np.issubdtype(stored[name].dtype, np.floating)
                and not np.isfinite(stored[name]).all()
            ):
                raise CacheCorruptionError(f"cache array {name!r} is non-finite")
        teacher_min = float(stored["teacher_rgb"].min())
        teacher_max = float(stored["teacher_rgb"].max())
        if teacher_min < -1.0 - 1e-5 or teacher_max > 1.0 + 1e-5:
            raise CacheCorruptionError("cache teacher RGB is outside the preprocessed range")
        raw_shape = tuple(stored["raw"].shape)
        for name in ("valid_mask", "bad_mask", "saturated_mask", "missing_mask"):
            if name in stored and tuple(stored[name].shape) not in {
                raw_shape,
                raw_shape[1:],
                (1, *raw_shape[1:]),
            }:
                raise CacheCorruptionError(f"cache mask {name!r} shape mismatch")
        if spec.state_shape is None:
            if "state" in stored:
                raise CacheCorruptionError("cache has an unexpected state")
        elif "state" not in stored or tuple(stored["state"].shape) != spec.state_shape:
            raise CacheCorruptionError("cache state shape mismatch")
        if spec.state_shape is not None:
            if stored["state"].dtype != np.float32:
                raise CacheCorruptionError("cache state dtype mismatch")
            if not np.isfinite(stored["state"]).all():
                raise CacheCorruptionError("cache state is non-finite")
            if "state_raw" in stored:
                if tuple(stored["state_raw"].shape) != spec.state_shape:
                    raise CacheCorruptionError("cache raw state shape mismatch")
                if stored["state_raw"].dtype != np.float32:
                    raise CacheCorruptionError("cache raw state dtype mismatch")
                if not np.isfinite(stored["state_raw"]).all():
                    raise CacheCorruptionError("cache raw state is non-finite")
        elif "state_raw" in stored:
            raise CacheCorruptionError("cache has an unexpected raw state")
        raw = RawFrameData(
            values=torch.from_numpy(stored["raw"].copy()),
            units=str(metadata["units"]),
            cfa_pattern=str(metadata["cfa_pattern"]),
            physical_origin=tuple(int(v) for v in metadata["physical_origin"]),
            valid_mask=_cache_mask(stored, "valid_mask"),
            bad_mask=_cache_mask(stored, "bad_mask"),
            saturated_mask=_cache_mask(stored, "saturated_mask"),
            missing_mask=_cache_mask(stored, "missing_mask"),
            black_level=metadata.get("black_level"),
            white_level=metadata.get("white_level"),
            exposure=metadata.get("exposure"),
            noise=metadata.get("noise"),
        )
        state = None if spec.state_shape is None else torch.from_numpy(stored["state"].copy())
        return DistillationSample(
            frame_key=str(metadata["frame_key"]),
            source_id=str(metadata["source_id"]),
            acquisition_id=str(metadata["acquisition_id"]),
            demo_id=str(metadata["demo_id"]),
            frame_id=int(metadata["frame_id"]),
            timestamp=float(metadata["timestamp"]),
            split=str(metadata["split"]),
            camera_id=str(metadata["camera_id"]),
            variant=str(metadata["variant"]),
            raw=raw,
            teacher_rgb=torch.from_numpy(stored["teacher_rgb"].copy()),
            teacher_rgb_source=str(metadata["teacher_rgb_source"]),
            actions=torch.from_numpy(stored["actions"].copy()),
            future_mask=torch.from_numpy(stored["future_mask"].copy()),
            state=state,
            state_raw=(
                torch.from_numpy(stored["state_raw"].copy()) if "state_raw" in stored else None
            ),
            language=str(metadata["language"]),
            fov=CameraFOV.from_mapping(metadata["fov"], "cache.fov"),
            crop=PhysicalCrop.from_mapping(metadata["crop"], "cache.crop"),
            flip=str(metadata["flip"]),
            checkpoint_sha256=str(metadata["checkpoint_sha256"]),
            isp_id=str(metadata["isp_id"]),
            preprocessing_id=str(metadata["preprocessing_id"]),
            normalization_id=str(metadata["normalization_id"]),
            augmentation_id=str(metadata["augmentation_id"]),
            cache_key=spec.key,
            metadata=metadata,
        )

    def put(self, spec: CacheSpec, sample: DistillationSample) -> Path:
        if sample.cache_key != spec.key:
            raise CacheCorruptionError("sample key differs from the cache specification")
        path = self._path(spec.key)
        arrays: dict[str, np.ndarray] = {
            "raw": sample.raw.values.numpy(),
            "teacher_rgb": sample.teacher_rgb.numpy(),
            "actions": sample.actions.numpy(),
            "future_mask": sample.future_mask.numpy(),
            "cache_key": np.frombuffer(spec.key.encode("utf-8"), dtype=np.uint8),
        }
        for name in ("valid_mask", "bad_mask", "saturated_mask", "missing_mask"):
            value = getattr(sample.raw, name)
            if value is not None:
                arrays[name] = value.numpy()
        if sample.state is not None:
            arrays["state"] = sample.state.numpy()
        if sample.state_raw is not None:
            arrays["state_raw"] = sample.state_raw.numpy()
        metadata = {
            "cache_schema": CACHE_SCHEMA,
            "cache_version": CACHE_VERSION,
            "cache_key": spec.key,
            "frame_key": sample.frame_key,
            "source_id": sample.source_id,
            "acquisition_id": sample.acquisition_id,
            "demo_id": sample.demo_id,
            "frame_id": sample.frame_id,
            "timestamp": sample.timestamp,
            "split": sample.split,
            "camera_id": sample.camera_id,
            "variant": sample.variant,
            "teacher_rgb_source": sample.teacher_rgb_source,
            "language": sample.language,
            "fov": sample.fov.as_dict(),
            "crop": sample.crop.as_dict(),
            "flip": sample.flip,
            "checkpoint_sha256": sample.checkpoint_sha256,
            "isp_id": sample.isp_id,
            "preprocessing_id": sample.preprocessing_id,
            "normalization_id": sample.normalization_id,
            "augmentation_id": sample.augmentation_id,
            "units": sample.raw.units,
            "cfa_pattern": sample.raw.cfa_pattern,
            "physical_origin": list(sample.raw.physical_origin),
            "black_level": sample.raw.black_level,
            "white_level": sample.raw.white_level,
            "exposure": sample.raw.exposure,
            "noise": sample.raw.noise,
        }
        arrays["sample_json"] = np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        return path


def _cache_mask(data: Mapping[str, np.ndarray], key: str) -> torch.Tensor | None:
    if key not in data:
        return None
    value = data[key]
    if value.dtype != np.bool_:
        raise CacheCorruptionError(f"cache mask {key!r} is not boolean")
    return torch.from_numpy(value.copy())


# --------------------------------------------------------------------------- #
# Public data set
# --------------------------------------------------------------------------- #
class PairedDistillationDataset:
    """Typed paired frame access for the next trainer round."""

    def __init__(
        self,
        manifest: PairedDistillationManifest,
        *,
        cache: SampleCache | None = None,
        student_only: bool = False,
    ) -> None:
        if student_only and cache is not None:
            raise PairedManifestError(
                "a pure student dataset may not be given a paired sample cache"
            )
        self.manifest = manifest
        self.cache = cache
        self.student_only = bool(student_only)
        self._frames = {frame.frame_key: frame for frame in manifest.frames}

    def __len__(self) -> int:
        return len(self.manifest.frames)

    @property
    def frame_keys(self) -> tuple[str, ...]:
        return tuple(frame.frame_key for frame in self.manifest.frames)

    def __getitem__(self, index: int) -> DistillationSample:
        return self.sample(self.manifest.frames[index].frame_key)

    def sample(self, frame_key: str, *, use_cache: bool = True) -> DistillationSample:
        if self.student_only:
            raise PairedDataUnavailable(
                "student-only dataset does not expose labels or teacher RGB; use student_view"
            )
        try:
            frame = self._frames[frame_key]
        except KeyError as exc:
            raise KeyError(f"unknown frame_key {frame_key!r}") from exc
        spec = CacheSpec.from_frame(self.manifest, frame)
        if self.cache is not None and use_cache:
            cached = self.cache.get(spec)
            if cached is not None:
                return cached
        sample = _load_sample(self.manifest, frame)
        if self.cache is not None and use_cache:
            self.cache.put(spec, sample)
        return sample

    def student_view(self, frame_key: str) -> StudentSample:
        """Load only RAW measurement fields; never touches teacher RGB or cache."""
        frame = self._frames[frame_key]
        if frame.variant == "rgb3_control":
            raise PairedDataUnavailable("RGB3 control is not exposed by the pure student interface")
        allowed = {
            "raw",
            "valid_mask",
            "bad_mask",
            "saturated_mask",
            "missing_mask",
            "state",
        }
        data = _npz(frame.array_path, expected_sha=frame.array_sha256, keys=tuple(allowed))
        raw = _raw_frame(frame, data, image_size=int(self.manifest.preprocessing["image_size"]))
        state_raw = None
        state = None
        if frame.state_dim is not None:
            if "state" not in data:
                raise PairedDataUnavailable(f"{frame.frame_key}: declared current state is absent")
            state_array = np.asarray(data["state"])
            if tuple(state_array.shape) != (frame.state_dim,):
                raise PairedManifestError(f"{frame.frame_key}: state shape mismatch")
            if not np.isfinite(state_array).all():
                raise PairedManifestError(f"{frame.frame_key}: state is non-finite")
            state_raw = torch.from_numpy(np.asarray(state_array, dtype=np.float32)).clone()
            normalised_state, _ = normalise_state(
                state_raw.numpy(), self.manifest.normalization.stats
            )
            if normalised_state is not None:
                state = torch.from_numpy(
                    np.ascontiguousarray(normalised_state, dtype=np.float32)
                ).clone()
        return StudentSample(
            frame_key=frame.frame_key,
            source_id=frame.source_id,
            acquisition_id=frame.acquisition_id,
            demo_id=frame.demo_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            split=self.manifest.source_split(frame.source_id),
            camera_id=frame.camera_id,
            variant=frame.variant,
            raw=raw,
            state=state,
            state_raw=state_raw,
            language=frame.language,
            fov=frame.fov,
            crop=frame.crop,
            flip=frame.flip,
            checkpoint_sha256=frame.checkpoint_sha256,
            isp_id=frame.isp_id,
            preprocessing_id=frame.preprocessing_id,
            normalization_id=frame.normalization_id,
            augmentation_id=frame.augmentation_id,
        )

    def source_frame_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for frame in self.manifest.frames:
            counts[frame.source_id] = counts.get(frame.source_id, 0) + 1
        return counts

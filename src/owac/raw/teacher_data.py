"""Strict source-demonstration pairing and causal windowing for the RAW teacher.

The author replay writes two optional NPZ trees (RAW and paired default-lighting
RGB).  This module validates one explicit source manifest before any frame is
expanded into a teacher window.  It never invokes a simulator or trainer.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "owac.raw-teacher.sources.v1"
TASK_SUITE = "libero_goal"
TASK_ID = 6
TASK_NAME = "put_the_cream_cheese_in_the_bowl"
INSTRUCTION = "put the cream cheese in the bowl"
LIGHTING = "Normal"
CAMERA_ORDER = ("agentview", "wrist")
RAW_KEYS = {"agentview": "agentview_raw_uint8", "wrist": "wrist_raw_uint8"}
RGB_KEYS = {"agentview": "agentview_rgb_uint8", "wrist": "wrist_rgb_uint8"}
ACTION_DIM = 7
PADDED_ACTION_DIM = 32
ACTION_HORIZON = 10
BURST_FRAMES = 6
RECURRENT_ENDPOINTS = 6
ENDPOINT_STRIDE = 5


class TeacherDataError(ValueError):
    """The source manifest, NPZ pair, or causal window violates the contract."""


PI05_STATS_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"


def load_pi05_normalization_stats(path: str | Path):
    """Strict loader for the actual LIBERO pi05 norm_stats JSON.

    The root is ``{"norm_stats": {"state":..., "actions":...}}``.  Action
    statistics must cover native 7D and state statistics native 8D with q01/q99.
    """
    from owac.raw.rawvla_sequence_n02 import NormalisationStats

    source = Path(path)
    if not source.is_file():
        raise TeacherDataError(f"pi05 statistics absent: {source}")
    actual_sha = sha256_file(source)
    if actual_sha != PI05_STATS_SHA256:
        raise TeacherDataError(f"pi05 statistics SHA differs: {actual_sha}")
    stats = NormalisationStats.from_json(source, mode="pi05_quantile")
    if stats.action is None or stats.state is None:
        raise TeacherDataError("pi05 statistics lacks action/state sections")
    for name, section, width in (
        ("action", stats.action, ACTION_DIM),
        ("state", stats.state, 8),
    ):
        for key in ("q01", "q99"):
            values = np.asarray(section.get(key), dtype=np.float32).reshape(-1)
            if values.size < width or not np.isfinite(values[:width]).all():
                raise TeacherDataError(f"pi05 {name}.{key} is missing/nonfinite for width {width}")
    return stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_policy_state(
    ee_pos: np.ndarray,
    ee_ori_axisangle: np.ndarray,
    gripper: np.ndarray,
) -> np.ndarray:
    """Canonical policy state8: eef_xyz, axis-angle xyz, gripper qpos2."""
    position = np.asarray(ee_pos, dtype=np.float32).reshape(-1, 3)
    orientation = np.asarray(ee_ori_axisangle, dtype=np.float32).reshape(-1, 3)
    gripper_values = np.asarray(gripper, dtype=np.float32).reshape(-1, 2)
    if not (position.shape[0] == orientation.shape[0] == gripper_values.shape[0]):
        raise TeacherDataError("canonical state source arrays have different lengths")
    state = np.concatenate([position, orientation, gripper_values], axis=1)
    if not np.isfinite(state).all():
        raise TeacherDataError("canonical state contains non-finite values")
    return state.astype(np.float32, copy=False)


def quaternion_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert source x,y,z,w quaternion rows to axis-angle."""
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(quat).all():
        raise TeacherDataError("quaternion contains non-finite values")
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    xyz = quat[:, :3]
    w = np.clip(quat[:, 3], -1.0, 1.0)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    angle = 2.0 * np.arccos(w)
    result = np.zeros((quat.shape[0], 3), dtype=np.float32)
    normal = sin_half > 1e-8
    result[normal] = (xyz[normal] * angle[normal, None] / sin_half[normal, None]).astype(np.float32)
    return result


@dataclass(frozen=True, slots=True)
class TeacherSource:
    source_demo_id: str
    split: str
    raw_path: str
    rgb_path: str
    raw_sha256: str
    rgb_sha256: str
    num_frames: int
    task_name: str
    lighting: str
    camera_order: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_demo_id": self.source_demo_id,
            "split": self.split,
            "raw_path": self.raw_path,
            "rgb_path": self.rgb_path,
            "raw_sha256": self.raw_sha256,
            "rgb_sha256": self.rgb_sha256,
            "num_frames": int(self.num_frames),
            "task_name": self.task_name,
            "lighting": self.lighting,
            "camera_order": list(self.camera_order),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TeacherSourceManifest:
    task: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_hdf5_sha256: str
    split_seed: int
    sources: tuple[TeacherSource, ...]
    path_semantics: str = "repo_relative"
    schema: str = SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task": dict(self.task),
            "provenance": dict(self.provenance),
            "source_hdf5_sha256": self.source_hdf5_sha256,
            "split_seed": int(self.split_seed),
            "path_semantics": self.path_semantics,
            "sources": [item.as_dict() for item in self.sources],
        }

    def sources_for_split(self, split: str) -> tuple[TeacherSource, ...]:
        if split not in {"train", "validation", "test"}:
            raise TeacherDataError(f"unknown split {split!r}")
        selected = tuple(item for item in self.sources if item.split == split)
        if any(item.split != split for item in selected):
            raise TeacherDataError("source split leakage")
        return selected

    def source(self, source_demo_id: str) -> TeacherSource:
        for item in self.sources:
            if item.source_demo_id == source_demo_id:
                return item
        raise TeacherDataError(f"unknown source_demo_id {source_demo_id!r}")


def _manifest_path(path: str, *, semantics: str, repo_root: Path) -> Path:
    value = Path(path)
    if semantics == "repo_relative":
        if value.is_absolute():
            raise TeacherDataError("repo_relative manifest contains an absolute path")
        return (repo_root / value).resolve()
    if semantics == "absolute":
        if not value.is_absolute():
            raise TeacherDataError("absolute manifest contains a relative path")
        return value.resolve()
    raise TeacherDataError(f"unknown path_semantics {semantics!r}")


def load_source_manifest(
    path: str | Path,
    *,
    repo_root: str | Path,
    strict: bool = True,
    expected_splits: Mapping[str, int] | None = None,
) -> TeacherSourceManifest:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise TeacherDataError(f"source manifest schema must be {SCHEMA}")
    task = payload.get("task")
    if not isinstance(task, Mapping):
        raise TeacherDataError("source manifest lacks task mapping")
    if strict:
        expected = {
            "suite": TASK_SUITE,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "instruction": INSTRUCTION,
            "lighting": LIGHTING,
        }
        for key, value in expected.items():
            if task.get(key) != value:
                raise TeacherDataError(f"task.{key} differs from the authorized task")
    if payload.get("split_seed") != 101 and strict:
        raise TeacherDataError("authorized split_seed is 101")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TeacherDataError("source manifest lacks provenance")
    hdf5_sha = str(payload.get("source_hdf5_sha256", ""))
    if len(hdf5_sha) != 64 or any(char not in "0123456789abcdef" for char in hdf5_sha):
        raise TeacherDataError("source_hdf5_sha256 must be 64 lowercase hex")
    semantics = str(payload.get("path_semantics", "repo_relative"))
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise TeacherDataError("source manifest has no sources")
    seen: set[str] = set()
    sources: list[TeacherSource] = []
    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            raise TeacherDataError("source entry is not an object")
        source_id = str(entry.get("source_demo_id", ""))
        if not source_id or source_id in seen:
            raise TeacherDataError(f"duplicate or empty source_demo_id {source_id!r}")
        seen.add(source_id)
        split = str(entry.get("split", ""))
        if split not in {"train", "validation", "test"}:
            raise TeacherDataError(f"invalid split {split!r}")
        raw_path = str(entry.get("raw_path", ""))
        rgb_path = str(entry.get("rgb_path", ""))
        _manifest_path(raw_path, semantics=semantics, repo_root=Path(repo_root))
        _manifest_path(rgb_path, semantics=semantics, repo_root=Path(repo_root))
        hashes = []
        for key in ("raw_sha256", "rgb_sha256"):
            value = str(entry.get(key, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise TeacherDataError(f"{source_id}.{key} must be 64 lowercase hex")
            hashes.append(value)
        num_frames = int(entry.get("num_frames", 0))
        if num_frames < 1:
            raise TeacherDataError(f"{source_id}.num_frames must be positive")
        camera_order = tuple(str(item) for item in entry.get("camera_order", ()))
        if camera_order != CAMERA_ORDER:
            raise TeacherDataError(f"{source_id}.camera_order must be {CAMERA_ORDER}")
        if str(entry.get("task_name")) != TASK_NAME:
            raise TeacherDataError(f"{source_id}.task_name differs from task6")
        if str(entry.get("lighting")) != LIGHTING:
            raise TeacherDataError(f"{source_id}.lighting must be Normal")
        sources.append(
            TeacherSource(
                source_demo_id=source_id,
                split=split,
                raw_path=raw_path,
                rgb_path=rgb_path,
                raw_sha256=hashes[0],
                rgb_sha256=hashes[1],
                num_frames=num_frames,
                task_name=TASK_NAME,
                lighting=LIGHTING,
                camera_order=camera_order,
                metadata=dict(entry.get("metadata", {}) or {}),
            )
        )
    counts = {
        name: sum(item.split == name for item in sources)
        for name in ("train", "validation", "test")
    }
    expected = (
        dict(expected_splits)
        if expected_splits is not None
        else ({"train": 40, "validation": 5, "test": 5} if strict else counts)
    )
    if counts != expected:
        raise TeacherDataError(f"split counts are {counts}, expected {expected}")
    if strict and len(sources) != 50:
        raise TeacherDataError("authorized source count is exactly 50; no silent shrink")
    return TeacherSourceManifest(
        task=dict(task),
        provenance=dict(provenance),
        source_hdf5_sha256=hdf5_sha,
        split_seed=int(payload.get("split_seed", 0)),
        sources=tuple(sources),
        path_semantics=semantics,
    )


def assign_source_splits(source_ids: Sequence[str], *, seed: int = 101) -> dict[str, str]:
    """Assign 40/5/5 by source id before any frame expansion."""
    ids = sorted(str(item) for item in source_ids)
    if len(ids) != len(set(ids)):
        raise TeacherDataError("duplicate source ids")
    if len(ids) < 50:
        raise TeacherDataError("source split needs at least 50 sources for the authorized protocol")
    rng = np.random.default_rng(int(seed))
    order = [ids[index] for index in rng.permutation(len(ids))]
    result = {}
    for index, source_id in enumerate(order):
        result[source_id] = "train" if index < 40 else "validation" if index < 45 else "test"
    return result


def _npz_file(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise TeacherDataError(f"episode NPZ is absent: {path}")
    if sha256_file(path) != expected_sha:
        raise TeacherDataError(f"episode NPZ hash differs: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise TeacherDataError(f"episode NPZ cannot be read without pickle: {path}") from exc


def _require_uint8_hwc(name: str, array: np.ndarray, frames: int) -> tuple[int, int]:
    if array.dtype != np.uint8:
        raise TeacherDataError(f"{name} must be uint8 pseudo-RAW/RGB, got {array.dtype}")
    if array.ndim != 4 or array.shape[0] != frames or array.shape[-1] != 3:
        raise TeacherDataError(f"{name} must be [T,H,W,3], got {array.shape}")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
        raise TeacherDataError(f"{name} values must be 0..255")
    return int(array.shape[1]), int(array.shape[2])


@dataclass(frozen=True, slots=True)
class TeacherEpisode:
    source: TeacherSource
    raw: Mapping[str, np.ndarray]
    rgb: Mapping[str, np.ndarray]
    actions: np.ndarray
    states: np.ndarray
    robot_states: np.ndarray
    dones: np.ndarray
    timestamps: np.ndarray | None
    source_frame_indices: np.ndarray
    source_frame_indices_exact: bool
    native_anchor_mask: np.ndarray
    control_freq_hz: float | None
    metadata: Mapping[str, Any]
    height: int
    width: int

    @property
    def num_frames(self) -> int:
        return int(self.actions.shape[0])


class EpisodeCache:
    """Small byte-bounded LRU; cache only validated episode arrays."""

    def __init__(self, max_entries: int = 2) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._items: OrderedDict[str, TeacherEpisode] = OrderedDict()

    def get(self, key: str) -> TeacherEpisode | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: str, value: TeacherEpisode) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


def load_teacher_episode(
    source: TeacherSource,
    *,
    repo_root: str | Path,
    path_semantics: str = "repo_relative",
    cache: EpisodeCache | None = None,
    include_rgb: bool = True,
) -> TeacherEpisode:
    key = f"{source.source_demo_id}:{source.raw_sha256}:{source.rgb_sha256}:{int(include_rgb)}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    raw_path = _manifest_path(source.raw_path, semantics=path_semantics, repo_root=Path(repo_root))
    rgb_path = _manifest_path(source.rgb_path, semantics=path_semantics, repo_root=Path(repo_root))
    raw_data = _npz_file(raw_path, source.raw_sha256)
    rgb_data = (
        (raw_data if rgb_path == raw_path else _npz_file(rgb_path, source.rgb_sha256))
        if include_rgb
        else None
    )
    frames = int(source.num_frames)
    raw = {}
    rgb = {}
    height = width = None
    for camera in CAMERA_ORDER:
        if RAW_KEYS[camera] not in raw_data:
            raise TeacherDataError(f"{source.source_demo_id} lacks {RAW_KEYS[camera]}")
        raw[camera] = np.asarray(raw_data[RAW_KEYS[camera]])
        h, w = _require_uint8_hwc(RAW_KEYS[camera], raw[camera], frames)
        if include_rgb:
            assert rgb_data is not None
            if RGB_KEYS[camera] not in rgb_data:
                raise TeacherDataError(f"{source.source_demo_id} lacks {RGB_KEYS[camera]}")
            rgb[camera] = np.asarray(rgb_data[RGB_KEYS[camera]])
            h2, w2 = _require_uint8_hwc(RGB_KEYS[camera], rgb[camera], frames)
            if (h, w) != (h2, w2):
                raise TeacherDataError(f"{source.source_demo_id} RAW/RGB geometry differs")
        height = h if height is None else height
        width = w if width is None else width
        if (height, width) != (h, w):
            raise TeacherDataError(f"{source.source_demo_id} camera geometry differs")
    if "actions" not in raw_data:
        raise TeacherDataError(f"{source.source_demo_id} lacks actions")
    actions = np.asarray(raw_data["actions"])
    if actions.shape != (frames, ACTION_DIM) or actions.dtype.kind not in "fiu":
        raise TeacherDataError(f"{source.source_demo_id} actions must be [T,7] numeric")
    actions = actions.astype(np.float32, copy=False)
    if not np.isfinite(actions).all():
        raise TeacherDataError(f"{source.source_demo_id} actions are non-finite")
    for name in ("states", "robot_states"):
        if name not in raw_data:
            raise TeacherDataError(f"{source.source_demo_id} lacks {name}")
    states = np.asarray(raw_data["states"])
    robot = np.asarray(raw_data["robot_states"])
    if states.ndim != 2 or states.shape[0] != frames:
        raise TeacherDataError(f"{source.source_demo_id} states have wrong length")
    if robot.shape != (frames, 8):
        raise TeacherDataError(
            f"{source.source_demo_id} robot_states must be canonical [T,8], got {robot.shape}"
        )
    if "source_robot_states9" in raw_data:
        source_robot = np.asarray(raw_data["source_robot_states9"])
        if source_robot.shape != (frames, 9):
            raise TeacherDataError(f"{source.source_demo_id} source_robot_states9 must be [T,9]")
    if not np.isfinite(states).all() or not np.isfinite(robot).all():
        raise TeacherDataError(f"{source.source_demo_id} states are non-finite")
    dones = np.asarray(raw_data.get("dones", np.zeros(frames, dtype=np.uint8)))
    if dones.shape != (frames,) or dones.dtype.kind not in "biu":
        raise TeacherDataError(f"{source.source_demo_id} dones must be [T]")
    timestamps = None
    if "timestamps" in raw_data:
        timestamps = np.asarray(raw_data["timestamps"], dtype=np.float64)
        if timestamps.shape != (frames,) or not np.isfinite(timestamps).all():
            raise TeacherDataError(f"{source.source_demo_id} timestamps must be finite [T]")
        if np.any(np.diff(timestamps) < 0):
            raise TeacherDataError(f"{source.source_demo_id} timestamps must be nondecreasing")
    metadata = dict(source.metadata)
    if "metadata_json" in raw_data:
        try:
            metadata = {**json.loads(str(raw_data["metadata_json"].item())), **metadata}
        except (ValueError, TypeError) as exc:
            raise TeacherDataError(f"{source.source_demo_id} metadata_json is invalid") from exc
    if "source_frame_indices" in raw_data:
        source_indices = np.asarray(raw_data["source_frame_indices"], dtype=np.int64)
        exact = True
    else:
        source_indices = np.arange(frames, dtype=np.int64)
        exact = False
    if source_indices.shape != (frames,) or np.any(np.diff(source_indices) <= 0):
        raise TeacherDataError(f"{source.source_demo_id} source frame indices must increase")
    if "native_anchor_mask" in raw_data:
        native_mask = np.asarray(raw_data["native_anchor_mask"], dtype=np.bool_)
        if native_mask.shape != (frames,):
            raise TeacherDataError(f"{source.source_demo_id} native_anchor_mask must be [T]")
    else:
        native_mask = np.ones(frames, dtype=np.bool_)
    control_freq = metadata.get("control_freq_hz")
    episode = TeacherEpisode(
        source=source,
        raw=raw,
        rgb=rgb,
        actions=actions,
        states=states,
        robot_states=robot,
        dones=dones,
        timestamps=timestamps,
        source_frame_indices=source_indices,
        source_frame_indices_exact=exact,
        native_anchor_mask=native_mask,
        control_freq_hz=None if control_freq is None else float(control_freq),
        metadata=metadata,
        height=int(height),
        width=int(width),
    )
    if cache is not None:
        cache.put(key, episode)
    return episode


@dataclass(frozen=True, slots=True)
class TeacherWindow:
    source_demo_id: str
    split: str
    current_frame: int
    camera_order: tuple[str, ...]
    raw_uint8: np.ndarray  # [T_rnn, K_burst, C, H, W, 3]
    rgb_uint8: np.ndarray
    state: np.ndarray
    actions: np.ndarray  # [H,7] real expert actions
    padded_actions: np.ndarray  # [H,32]
    action_mask: np.ndarray  # [H] bool
    padded_state: np.ndarray  # [32]
    source_frame_indices: np.ndarray  # [T_rnn,K_burst]
    timestamps: np.ndarray | None
    anchor_eligible: bool
    nominal_dt: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self, stats: Any, *, state_pad_value: float = 0.0) -> dict[str, np.ndarray]:
        """Quantile normalization using the recovered pi05 statistics.

        Training may defer this call, but the mask and padded shapes stay fixed.
        """
        action_stats = getattr(stats, "action", None)
        if action_stats is None:
            raise TeacherDataError("pi05 action statistics are unavailable")
        q01 = np.asarray(action_stats["q01"], dtype=np.float32).reshape(-1)[:ACTION_DIM]
        q99 = np.asarray(action_stats["q99"], dtype=np.float32).reshape(-1)[:ACTION_DIM]
        if q01.size < ACTION_DIM or q99.size < ACTION_DIM:
            raise TeacherDataError("pi05 action statistics are narrower than 7D")
        valid = self.action_mask
        actions = 2.0 * (self.actions - q01) / np.maximum(q99 - q01, 1.0e-6) - 1.0
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        padded = np.zeros((ACTION_HORIZON, PADDED_ACTION_DIM), dtype=np.float32)
        padded[:, :ACTION_DIM] = actions
        state_stats = getattr(stats, "state", None)
        state = np.asarray(self.state, dtype=np.float32).reshape(-1)
        if state_stats is not None:
            sq01 = np.asarray(state_stats["q01"], dtype=np.float32).reshape(-1)
            sq99 = np.asarray(state_stats["q99"], dtype=np.float32).reshape(-1)
            width = min(state.size, sq01.size, sq99.size)
            state = state.copy()
            state[:width] = np.clip(
                2.0 * (state[:width] - sq01[:width]) / np.maximum(sq99[:width] - sq01[:width], 1e-6)
                - 1.0,
                -1.0,
                1.0,
            )
        # Diagnostic only. Actual LIBERO PI05 config has
        # discrete_state_input=false, so native padding is 0.0; a future true
        # branch must pass its configured pad explicitly. Model paths consume
        # ``state`` (8D), not this vector.
        padded_state = np.full((PADDED_ACTION_DIM,), float(state_pad_value), dtype=np.float32)
        padded_state[: min(state.size, PADDED_ACTION_DIM)] = state[:PADDED_ACTION_DIM]
        return {
            "actions": actions,
            "action_mask": valid.copy(),
            "padded_actions": padded,
            "state": state,
            "padded_state": padded_state,
        }


def make_window(
    episode: TeacherEpisode,
    current: int,
    *,
    burst_frames: int = BURST_FRAMES,
    recurrent_endpoints: int = RECURRENT_ENDPOINTS,
    endpoint_stride: int = ENDPOINT_STRIDE,
    action_horizon: int = ACTION_HORIZON,
    state_pad_value: float = 0.0,
) -> TeacherWindow:
    if min(burst_frames, recurrent_endpoints, endpoint_stride, action_horizon) < 1:
        raise TeacherDataError("temporal dimensions must be positive")
    frames = episode.num_frames
    if not 0 <= current < frames:
        raise TeacherDataError("current frame is outside the episode")
    observations = current - endpoint_stride * np.arange(
        recurrent_endpoints - 1, -1, -1, dtype=np.int64
    )
    observations = np.clip(observations, 0, current)
    bursts = observations[:, None] - np.arange(burst_frames - 1, -1, -1, dtype=np.int64)[None, :]
    bursts = np.clip(bursts, 0, current)
    if np.any(bursts > current) or np.any(bursts < 0):
        raise TeacherDataError("causal window contains a future or negative frame")
    raw = np.stack(
        [
            np.stack([episode.raw[camera][indices] for camera in CAMERA_ORDER], axis=1)
            for indices in bursts
        ],
        axis=0,
    )
    rgb = (
        np.stack(
            [
                np.stack([episode.rgb[camera][indices] for camera in CAMERA_ORDER], axis=1)
                for indices in bursts
            ],
            axis=0,
        )
        if episode.rgb
        else np.empty((0,), dtype=np.uint8)
    )
    # RGB is absent in the RAW-only deployment evaluation view.
    actions = np.zeros((action_horizon, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((action_horizon,), dtype=bool)
    for offset in range(action_horizon):
        index = current + offset
        if index >= frames:
            break
        if offset > 0 and bool(episode.dones[index - 1]):
            break
        # Dense source actions are retained; no-op eligibility is recorded by
        # anchor_eligible and native_anchor_mask, never by rewriting actions.
        actions[offset] = episode.actions[index]
        mask[offset] = True
    padded = np.zeros((action_horizon, PADDED_ACTION_DIM), dtype=np.float32)
    padded[:, :ACTION_DIM] = actions
    # `states` is MuJoCo's privileged physical state79 for rerendering only.
    # The deployed policy consumes the explicitly converted robot observation8.
    state = np.asarray(episode.robot_states[current], dtype=np.float32).reshape(-1)
    if state.shape != (8,):
        raise TeacherDataError("policy window requires canonical robot observation8")
    # Diagnostic only; actual LIBERO PI05 false-state config pads with 0.0.
    # Model paths consume canonical state8 and let the native policy pad it.
    padded_state = np.full((PADDED_ACTION_DIM,), float(state_pad_value), dtype=np.float32)
    padded_state[: min(state.size, PADDED_ACTION_DIM)] = state[:PADDED_ACTION_DIM]
    return TeacherWindow(
        source_demo_id=episode.source.source_demo_id,
        split=episode.source.split,
        current_frame=int(current),
        camera_order=CAMERA_ORDER,
        raw_uint8=raw.astype(np.uint8, copy=False),
        rgb_uint8=rgb.astype(np.uint8, copy=False),
        state=state,
        actions=actions,
        padded_actions=padded,
        action_mask=mask,
        padded_state=padded_state,
        source_frame_indices=episode.source_frame_indices[bursts],
        timestamps=(None if episode.timestamps is None else episode.timestamps[observations]),
        anchor_eligible=bool(episode.native_anchor_mask[current]),
        nominal_dt=(
            None if episode.control_freq_hz in (None, 0) else 1.0 / float(episode.control_freq_hz)
        ),
        metadata={
            "source_frame_indices_exact": episode.source_frame_indices_exact,
            "native_anchor_mask": bool(episode.native_anchor_mask[current]),
            "dense_observation_history": True,
        },
    )


@dataclass(frozen=True, slots=True)
class TeacherAnchor:
    source_demo_id: str
    split: str
    current_frame: int


class TeacherWindowSampler:
    """Deterministic source-block sampler; never materializes pixel windows."""

    def __init__(
        self,
        data: TeacherData,
        *,
        split: str,
        seed: int = 101,
        anchors_per_source: int = 32,
    ) -> None:
        self.data = data
        self.split = split
        self.seed = int(seed)
        self.anchors_per_source = int(anchors_per_source)
        self.epoch = 0
        self.block_cursor = 0
        self.source_order = self._shuffled_sources()
        self.anchor_cursor = 0
        self.current_anchors: list[int] = []

    def _shuffled_sources(self) -> list[TeacherSource]:
        sources = list(self.data.manifest.sources_for_split(self.split))
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(sources)).tolist()
        return [sources[index] for index in order]

    def _source_anchors(self, source: TeacherSource) -> list[int]:
        anchors = self.data.anchor_indices(source.source_demo_id)
        if not anchors:
            raise TeacherDataError(f"source {source.source_demo_id} has no native anchors")
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + self.block_cursor * 7919)
        if len(anchors) >= self.anchors_per_source:
            selected = rng.choice(anchors, size=self.anchors_per_source, replace=False)
        else:
            selected = rng.choice(anchors, size=self.anchors_per_source, replace=True)
        return [int(value) for value in selected]

    def next_anchor(self) -> TeacherAnchor:
        if not self.source_order:
            raise TeacherDataError("no sources for sampler")
        if not self.current_anchors or self.anchor_cursor >= len(self.current_anchors):
            if self.block_cursor >= len(self.source_order):
                self.epoch += 1
                self.block_cursor = 0
                self.source_order = self._shuffled_sources()
            source = self.source_order[self.block_cursor]
            self.current_anchors = self._source_anchors(source)
            self.anchor_cursor = 0
            self.block_cursor += 1
            self._current_source = source.source_demo_id
        current = self.current_anchors[self.anchor_cursor]
        self.anchor_cursor += 1
        return TeacherAnchor(self._current_source, self.split, int(current))

    def next(self) -> TeacherWindow:
        return self.data.window(self.next_anchor())

    def __iter__(self):
        while True:
            yield self.next()

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "split": self.split,
            "epoch": self.epoch,
            "block_cursor": self.block_cursor,
            "anchor_cursor": self.anchor_cursor,
            "current_source": getattr(self, "_current_source", None),
            "current_anchors": list(self.current_anchors),
            "anchor_per_source": self.anchors_per_source,
            "source_order": [item.source_demo_id for item in self.source_order],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("split") != self.split or int(state.get("seed", -1)) != self.seed:
            raise TeacherDataError("sampler resume identity differs")
        self.epoch = int(state["epoch"])
        self.block_cursor = int(state["block_cursor"])
        self.anchor_cursor = int(state.get("anchor_cursor", 0))
        self.current_anchors = [int(value) for value in state.get("current_anchors", ())]
        self._current_source = state.get("current_source")
        by_id = {
            item.source_demo_id: item for item in self.data.manifest.sources_for_split(self.split)
        }
        self.source_order = [by_id[item] for item in state["source_order"]]
        if self.block_cursor < 0 or self.block_cursor > len(self.source_order):
            raise TeacherDataError("sampler cursor is invalid")
        if self.anchor_cursor < 0 or self.anchor_cursor > len(self.current_anchors):
            raise TeacherDataError("sampler anchor cursor is invalid")


class TeacherData:
    """Manifest-bound dataset; split filter happens before frame expansion."""

    def __init__(
        self,
        manifest: TeacherSourceManifest,
        *,
        repo_root: str | Path,
        cache_entries: int = 2,
    ) -> None:
        self.manifest = manifest
        self.repo_root = Path(repo_root)
        self.cache = EpisodeCache(cache_entries)

    def episode(self, source_demo_id: str) -> TeacherEpisode:
        source = self.manifest.source(source_demo_id)
        return load_teacher_episode(
            source,
            repo_root=self.repo_root,
            path_semantics=self.manifest.path_semantics,
            cache=self.cache,
        )

    def episode_raw(self, source_demo_id: str) -> TeacherEpisode:
        """Load only RAW/action/state arrays; never open rgb_path."""
        source = self.manifest.source(source_demo_id)
        return load_teacher_episode(
            source,
            repo_root=self.repo_root,
            path_semantics=self.manifest.path_semantics,
            cache=self.cache,
            include_rgb=False,
        )

    def anchor_indices(self, source_demo_id: str) -> tuple[int, ...]:
        """Load only the small anchor mask/actions, never pixel arrays."""
        source = self.manifest.source(source_demo_id)
        raw_path = _manifest_path(
            source.raw_path,
            semantics=self.manifest.path_semantics,
            repo_root=self.repo_root,
        )
        if sha256_file(raw_path) != source.raw_sha256:
            raise TeacherDataError(f"episode NPZ hash differs: {raw_path}")
        try:
            with np.load(raw_path, allow_pickle=False) as archive:
                if "native_anchor_mask" not in archive.files:
                    return tuple(range(int(source.num_frames)))
                mask = np.asarray(archive["native_anchor_mask"], dtype=np.bool_)
        except (OSError, ValueError) as exc:
            raise TeacherDataError(f"cannot read anchor mask: {raw_path}") from exc
        if mask.shape != (int(source.num_frames),):
            raise TeacherDataError(f"anchor mask shape differs for {source_demo_id}")
        return tuple(int(index) for index in np.flatnonzero(mask))

    def iter_anchors(self, split: str) -> Iterator[TeacherAnchor]:
        for source in self.manifest.sources_for_split(split):
            for current in self.anchor_indices(source.source_demo_id):
                yield TeacherAnchor(source.source_demo_id, split, current)

    def window(self, anchor: TeacherAnchor) -> TeacherWindow:
        if anchor.split not in {"train", "validation", "test"}:
            raise TeacherDataError("invalid anchor split")
        episode = self.episode(anchor.source_demo_id)
        return make_window(episode, anchor.current_frame)

    def iter_windows(self, split: str) -> Iterator[TeacherWindow]:
        for anchor in self.iter_anchors(split):
            yield self.window(anchor)

    def sampler(self, *, split: str, seed: int = 101) -> TeacherWindowSampler:
        return TeacherWindowSampler(self, split=split, seed=seed)


__all__ = [
    "ACTION_DIM",
    "BURST_FRAMES",
    "CAMERA_ORDER",
    "ENDPOINT_STRIDE",
    "EpisodeCache",
    "INSTRUCTION",
    "LIGHTING",
    "PADDED_ACTION_DIM",
    "RECURRENT_ENDPOINTS",
    "SCHEMA",
    "TASK_ID",
    "TASK_NAME",
    "TASK_SUITE",
    "TeacherAnchor",
    "TeacherData",
    "TeacherWindowSampler",
    "TeacherDataError",
    "TeacherEpisode",
    "TeacherSource",
    "TeacherSourceManifest",
    "TeacherWindow",
    "assign_source_splits",
    "canonical_policy_state",
    "quaternion_to_axisangle",
    "load_pi05_normalization_stats",
    "load_source_manifest",
    "load_teacher_episode",
    "make_window",
]

"""N04 S0 数据清单：逐 NPZ 审计、源轨迹/变体分组与冻结拆分。

对应 ``docs/prompts/n04_v1_0/S0_data_contracts.md``「文件与具体开发顺序」第 2 项与
「拆分与预注册」一节。设计约束：

* **不全量解压**：RawVLA-Bench 的 NPZ 是 ZIP stored（``compress_type == 0``），因此
  逐文件只读 zip 中央目录、``metadata_json`` 与各数组的 ``.npy`` header；大数组按帧
  采样读取（seek 到帧起点，读该帧字节），小数组整读。每次读取都计入 I/O 统计。
* **源身份只从真实 metadata 推断**：LIBERO 用 ``dataset_name/task_name/demo_key``，
  RoboTwin 用 ``task_name/source_trajectory``。注意现有
  ``rawvla_sequence.source_identity`` 对 RoboTwin 返回
  ``robotwin:<task_name>:<dataset_seed>``，会把同一任务的全部 episode 折叠成一个源；
  本模块**不**复用该分支，并在 ``splits.json`` 中记录该差异。
* **拆分按 ``source_trajectory_id`` 分组**、平台内分层（再按任务分层以保留小任务分配），
  固定 ``split_seed``；同一输入两次运行的内容哈希必须一致（``content_sha256``）。
* **不编造字段**：所有覆盖状态、provenance 与「不可用条件」都从本模块实测的 key 计数、
  shape/dtype、数值范围与 metadata 取值聚合得到。

字段定义（写入 ``inventory.json.definitions``）：

``effective_frames``
    该文件中相机、本体、动作数组都能对齐到的最小帧数。
``missing_rate``
    ``1 - effective_frames / max(declared_frames, 各数组最大帧数)``；全对齐时为 0。
``sampled_frames``
    大数组按帧采样时实际读取的帧索引；小数组（<= ``small_array_max_bytes``）整读。
``content_sha256``
    去掉易变段（``run``、``generated_utc``、``read_seconds`` 等）后的规范化 JSON 哈希。
"""

from __future__ import annotations

import argparse
import hashlib
import io as io_module
import json
import re
import resource
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "n04-s0-inventory-v1"
COMPANION_FILES = (
    "splits.json",
    "coverage.json",
    "telemetry-provenance.json",
    "inventory-hashes.json",
)
DEFAULT_SMALL_ARRAY_MAX_BYTES = 8 * 1024 * 1024
METADATA_MEMBER = "metadata_json.npy"
#: ws-telemetry 落盘的完整来源表；本模块只引用路径与哈希，不复制内容。
TELEMETRY_SOURCE_AUDIT = (
    "research/experiments/results/2026-10-14_n04_v1_s0/telemetry-source-audit.json"
)
VALUE_COUNTER_TOP = 12
VOLATILE_KEYS = frozenset(
    {"run", "generated_utc", "read_seconds", "wall_seconds", "peak_rss_kb", "elapsed_seconds"}
)


# ---------------------------------------------------------------------------
# 平台规格
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformSpec:
    """一个平台的数组布局与 provenance 字段清单（全部来自实测 key）。"""

    key: str
    cameras: dict[str, str]
    proprio: str
    action: str
    extra_arrays: tuple[str, ...]
    provenance_metadata: tuple[str, ...]
    identity_metadata: tuple[str, ...]
    variant_metadata: tuple[str, ...]

    @property
    def trajectory_arrays(self) -> tuple[str, ...]:
        return (*self.cameras.values(), self.proprio, self.action)


LIBERO_SPEC = PlatformSpec(
    key="libero",
    cameras={"agentview": "agentview_raw_uint8", "wrist": "wrist_raw_uint8"},
    proprio="robot_states",
    action="actions",
    extra_arrays=("states", "dones", "rewards"),
    provenance_metadata=(
        "benchmark",
        "dataset_name",
        "suite",
        "task_id",
        "task_name",
        "demo_index",
        "demo_key",
        "trajectory_id",
        "noops_filtered",
        "hdf5_path",
        "variant_index",
        "representation",
        "render_order",
        "replay_mode",
        "paired_rgb_render_order",
        "lighting_domain",
        "lighting_ev",
        "lighting_rig",
        "lighting_scale",
        "lighting_sample_unit",
        "noise_seed",
        "sensor_saturation_ev",
        "agentview_table_flood",
        "agentview_stats",
        "wrist_stats",
        "raw_shape",
        "source_shape",
        "num_frames",
        "npz_relpath",
        "rgb_npz_relpath",
    ),
    identity_metadata=("dataset_name", "task_name", "demo_key"),
    variant_metadata=("variant_index", "lighting_domain", "noise_seed"),
)

ROBOTWIN_SPEC = PlatformSpec(
    key="robotwin",
    cameras={
        "head": "head_camera_raw_float32",
        "left": "left_camera_raw_float32",
        "right": "right_camera_raw_float32",
    },
    proprio="robot_states",
    action="actions",
    extra_arrays=("left_endpose", "right_endpose", "dones", "rewards"),
    provenance_metadata=(
        "benchmark",
        "task_name",
        "episode",
        "entry_index",
        "dataset_seed",
        "source_hdf5",
        "source_trajectory",
        "clean_protocol",
        "clean_replay_record",
        "raw_protocol",
        "raw_npz",
        "rgb_npz",
        "views",
        "representation",
        "lighting_domain",
        "lighting_ev",
        "lighting_strategy",
        "lighting_scale",
        "lighting_sample_unit",
        "lighting_applied",
        "noise_seed",
        "raw_sensor_noise_enabled",
        "raw_sensor_read_noise",
        "raw_sensor_shot_noise",
        "raw_white_level",
        "num_frames",
    ),
    identity_metadata=("task_name", "source_trajectory"),
    variant_metadata=("episode", "lighting_domain", "noise_seed", "entry_index"),
)

SPECS: dict[str, PlatformSpec] = {"libero": LIBERO_SPEC, "robotwin": ROBOTWIN_SPEC}
#: 相机数组通常是最大的成员：只有它们按帧采样；其余小数组整读。
CAMERA_STEMS: frozenset[str] = frozenset(
    key for spec in SPECS.values() for key in spec.cameras.values()
)
SCENE_TOKEN_RE = re.compile(r"^([A-Z][A-Z0-9_]*_SCENE\d+)_")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def strip_volatile(payload: Any) -> Any:
    """去掉运行期易变字段，用于内容哈希与两次运行一致性比较。"""

    if isinstance(payload, dict):
        return {
            key: strip_volatile(value) for key, value in payload.items() if key not in VOLATILE_KEYS
        }
    if isinstance(payload, list):
        return [strip_volatile(item) for item in payload]
    if isinstance(payload, tuple):
        return [strip_volatile(item) for item in payload]
    return payload


def stable_order(items: Sequence[str], seed: int) -> list[str]:
    """按 ``sha256(seed:item)`` 排序：与输入顺序无关，两次运行结果一致。"""

    return sorted(
        items, key=lambda item: (hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(), item)
    )


def frame_indices(total: int, count: int) -> list[int]:
    """在 ``[0, total)`` 上取 ``count`` 个均匀分布且含首尾的帧索引。"""

    if total <= 0:
        return []
    if count <= 1 or total == 1:
        return [0]
    indices = {round(i * (total - 1) / (count - 1)) for i in range(count)}
    return sorted(indices)


@dataclass
class IoCounter:
    bytes_read: int = 0
    files: int = 0
    seconds: float = 0.0

    def read(self, count: int) -> None:
        self.bytes_read += max(0, int(count))

    def as_dict(self) -> dict[str, Any]:
        return {
            "bytes_read": self.bytes_read,
            "files": self.files,
            "seconds": round(self.seconds, 6),
        }


def _read_exact(stream, count: int, io: IoCounter) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(min(remaining, 4 << 20))
        if not chunk:
            raise EOFError(f"读取 {count} 字节时提前结束（剩余 {remaining}）")
        io.read(len(chunk))
        remaining -= len(chunk)
        chunks.append(chunk)
    if not chunks:
        return b""
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _top_counter(counter: Counter, top: int = VALUE_COUNTER_TOP) -> dict[str, int]:
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    head = {str(key): int(value) for key, value in ordered[:top]}
    if len(ordered) > top:
        head["__other__"] = int(sum(value for _, value in ordered[top:]))
    return head


# ---------------------------------------------------------------------------
# NPZ 读取
# ---------------------------------------------------------------------------


def read_metadata(zf: zipfile.ZipFile, io: IoCounter) -> dict[str, Any]:
    if METADATA_MEMBER not in zf.namelist():
        return {}
    with zf.open(METADATA_MEMBER) as stream:
        payload = stream.read()
    io.read(len(payload))
    text = str(np.load(io_module.BytesIO(payload), allow_pickle=False).item())
    return json.loads(text)


def read_header(zf: zipfile.ZipFile, member: str, io: IoCounter) -> dict[str, Any]:
    with zf.open(member) as stream:
        np.lib.format.read_magic(stream)
        io.read(6)
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        io.read(64)
    return {"shape": [int(dim) for dim in shape], "dtype": str(dtype)}


def read_array(zf: zipfile.ZipFile, member: str, io: IoCounter) -> np.ndarray:
    with zf.open(member) as stream:
        payload = stream.read()
    io.read(len(payload))
    return np.load(io_module.BytesIO(payload), allow_pickle=False)


def read_frames(
    zf: zipfile.ZipFile,
    member: str,
    shape: Sequence[int],
    dtype: np.dtype,
    indices: Sequence[int],
    io: IoCounter,
) -> np.ndarray:
    """按帧读取：stored 成员用 seek 只读目标帧；压缩成员按顺序流式读取。"""

    frame_shape = tuple(int(dim) for dim in shape[1:])
    items = int(np.prod(frame_shape)) if frame_shape else 1
    frame_bytes = items * int(dtype.itemsize)
    out = np.empty((len(indices),) + frame_shape, dtype=dtype)
    stored = zf.getinfo(member).compress_type == zipfile.ZIP_STORED
    with zf.open(member) as stream:
        if stored:
            for slot, index in enumerate(indices):
                stream.seek(int(index) * frame_bytes)
                buffer = _read_exact(stream, frame_bytes, io)
                out[slot] = np.frombuffer(buffer, dtype=dtype).reshape(frame_shape)
        else:
            wanted = {int(index): slot for slot, index in enumerate(indices)}
            for index in range(int(shape[0])):
                buffer = _read_exact(stream, frame_bytes, io)
                if index in wanted:
                    out[wanted[index]] = np.frombuffer(buffer, dtype=dtype).reshape(frame_shape)
    return out


def array_stats(values: np.ndarray, *, white_level: float | None = None) -> dict[str, Any]:
    """有限值统计；非有限值单独计数，不静默丢弃。"""

    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return {
            "elements": 0,
            "nonfinite": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "zero_fraction": None,
            "saturation_fraction": None,
            "negative_fraction": None,
        }
    is_float = flat.dtype.kind == "f"
    nonfinite = 0
    if is_float:
        finite = np.isfinite(flat)
        nonfinite = int(flat.size - int(finite.sum()))
        good = flat[finite]
    else:
        good = flat
    stats: dict[str, Any] = {
        "elements": int(flat.size),
        "nonfinite": nonfinite,
        "min": float(good.min()) if good.size else None,
        "max": float(good.max()) if good.size else None,
        "mean": float(good.mean(dtype=np.float64)) if good.size else None,
        "std": float(good.std(dtype=np.float64)) if good.size else None,
        "zero_fraction": float(np.count_nonzero(flat == 0) / flat.size),
        "negative_fraction": float(np.count_nonzero(flat < 0) / flat.size) if is_float else 0.0,
    }
    if flat.dtype.kind in ("u", "i"):
        info = np.iinfo(flat.dtype)
        stats["saturation_fraction"] = float(np.count_nonzero(flat == info.max) / flat.size)
    elif white_level is not None:
        stats["saturation_fraction"] = float(np.count_nonzero(flat >= white_level) / flat.size)
    else:
        stats["saturation_fraction"] = None
    return stats


# ---------------------------------------------------------------------------
# 记录构建
# ---------------------------------------------------------------------------


def platform_roots(config: dict[str, Any], overrides: dict[str, str]) -> dict[str, Path]:
    data = config["remote"]["data"]
    roots = {"libero": Path(data["libero_raw"]), "robotwin": Path(data["robotwin_raw"])}
    for key, value in overrides.items():
        roots[key] = Path(value)
    return roots


def list_npz(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def source_trajectory_id(
    spec: PlatformSpec, metadata: dict[str, Any], relpath: str
) -> tuple[str, str]:
    """返回 ``(source_trajectory_id, identity_source)``。"""

    if spec.key == "libero":
        values = [metadata.get(key) for key in spec.identity_metadata]
        if all(value for value in values):
            dataset, task, demo = (str(value) for value in values)
            return f"libero:{dataset}:{task}:{demo}", "metadata"
    else:
        task = metadata.get("task_name")
        source = metadata.get("source_trajectory")
        if task and source:
            return f"robotwin:{task}:{source}", "metadata"
    return f"{spec.key}:path:{relpath}", "path"


def variant_label(spec: PlatformSpec, metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    if spec.key == "libero":
        parts.append(f"variant_{int(metadata.get('variant_index', 0)):02d}")
    for key in ("lighting_domain", "lighting_strategy"):
        if metadata.get(key) is not None:
            parts.append(f"{key}={metadata[key]}")
    if metadata.get("noise_seed") is not None:
        parts.append(f"noise_seed={metadata['noise_seed']}")
    if spec.key == "robotwin" and metadata.get("entry_index") is not None:
        parts.append(f"entry_index={metadata['entry_index']}")
    return "|".join(parts) or "variant_00"


def task_key(spec: PlatformSpec, metadata: dict[str, Any], relpath: str) -> str:
    if spec.key == "libero":
        dataset = metadata.get("dataset_name") or relpath.split("/")[0]
        task = metadata.get("task_name") or "unknown_task"
        return f"{dataset}:{task}"
    return str(metadata.get("task_name") or relpath.split("/")[0])


def suite_key(spec: PlatformSpec, metadata: dict[str, Any], relpath: str) -> str:
    if spec.key == "libero":
        return str(metadata.get("suite") or relpath.split("/")[0])
    return str(metadata.get("task_name") or relpath.split("/")[0])


def scene_token(metadata: dict[str, Any]) -> str | None:
    match = SCENE_TOKEN_RE.match(str(metadata.get("task_name") or ""))
    return match.group(1) if match else None


def rgb_sibling(
    spec: PlatformSpec, metadata: dict[str, Any], rgb_root: Path | None
) -> dict[str, Any]:
    if rgb_root is None:
        return {"relpath": None, "exists": None, "reason": "rgb 根目录不存在"}
    if spec.key == "libero":
        relpath = metadata.get("rgb_npz_relpath")
    else:
        task = metadata.get("task_name")
        name = metadata.get("rgb_npz")
        relpath = f"{task}/{name}" if task and name else None
    if not relpath:
        return {"relpath": None, "exists": None, "reason": "metadata 未给出 RGB 配对路径"}
    return {"relpath": str(relpath), "exists": (rgb_root / str(relpath)).is_file(), "reason": None}


def build_record(
    path: Path,
    *,
    spec: PlatformSpec,
    root: Path,
    rgb_root: Path | None,
    sample_frames: int,
    small_array_max_bytes: int,
    io: IoCounter,
) -> dict[str, Any]:
    """一个 NPZ 的只读记录：metadata、header、小数组统计与采样帧范围。"""

    started = time.perf_counter()
    file_io = IoCounter(files=1)
    relpath = str(path.relative_to(root))
    record: dict[str, Any] = {"platform": spec.key, "relpath": relpath, "path": str(path)}
    record["size_bytes"] = int(path.stat().st_size)
    record["mtime_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
    with zipfile.ZipFile(path) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        record["keys"] = sorted(name[: -len(".npy")] for name in infos if name.endswith(".npy"))
        record["members"] = {
            name: {"raw_bytes": int(info.file_size), "compress_type": int(info.compress_type)}
            for name, info in sorted(infos.items())
        }
        metadata = read_metadata(zf, file_io)
        record["metadata_keys"] = sorted(metadata)
        record["metadata"] = {
            key: metadata[key] for key in spec.provenance_metadata if key in metadata
        }
        record["metadata"] = _jsonable(record["metadata"])

        headers: dict[str, Any] = {}
        for member in sorted(infos):
            if not member.endswith(".npy") or member == METADATA_MEMBER:
                continue
            headers[member[: -len(".npy")]] = read_header(zf, member, file_io)
        record["headers"] = headers

        frames_by_key: dict[str, int] = {}
        array_reports: dict[str, Any] = {}
        white_level = _float_or_none(metadata.get("raw_white_level"))
        for key, header in headers.items():
            shape = header["shape"]
            if not shape:
                frames_by_key[key] = 0
                continue
            frames_by_key[key] = int(shape[0])
            if key in CAMERA_STEMS:
                indices = frame_indices(int(shape[0]), sample_frames)
                dtype = np.dtype(header["dtype"])
                values = read_frames(zf, f"{key}.npy", shape, dtype, indices, file_io)
                report = {
                    "read_mode": "sampled_frames",
                    "frames_read": indices,
                    "shape": shape,
                    "dtype": header["dtype"],
                }
                report.update(array_stats(values, white_level=white_level))
            else:
                member_bytes = int(infos[f"{key}.npy"].file_size)
                if member_bytes <= small_array_max_bytes:
                    values = read_array(zf, f"{key}.npy", file_io)
                    report = {
                        "read_mode": "full",
                        "frames_read": list(range(int(values.shape[0]))) if values.ndim else [],
                        "shape": shape,
                        "dtype": header["dtype"],
                    }
                    report.update(array_stats(values, white_level=white_level))
                else:
                    indices = frame_indices(int(shape[0]), sample_frames)
                    values = read_frames(
                        zf, f"{key}.npy", shape, np.dtype(header["dtype"]), indices, file_io
                    )
                    report = {
                        "read_mode": "sampled_frames",
                        "frames_read": indices,
                        "shape": shape,
                        "dtype": header["dtype"],
                    }
                    report.update(array_stats(values, white_level=white_level))
            array_reports[key] = report

    declared = metadata.get("num_frames")
    declared_int = int(declared) if isinstance(declared, (int, float)) else None
    camera_counts = [frames_by_key[key] for key in spec.cameras.values() if key in frames_by_key]
    proprio_count = frames_by_key.get(spec.proprio)
    action_count = frames_by_key.get(spec.action)
    counts = [count for count in (*camera_counts, proprio_count, action_count) if count is not None]
    effective = min(counts) if counts else 0
    widest = max([count for count in counts] + ([declared_int] if declared_int else []), default=0)
    record["frames"] = {
        "declared": declared_int,
        "by_key": frames_by_key,
        "camera_counts": {
            key: frames_by_key[key] for key in spec.cameras.values() if key in frames_by_key
        },
        "effective": int(effective),
        "widest": int(widest),
        "aligned": bool(len(set(counts)) <= 1 and (declared_int in (None, effective))),
        "missing_rate": float(1.0 - effective / widest) if widest else None,
    }
    record["arrays"] = array_reports

    identity, identity_source = source_trajectory_id(spec, metadata, relpath)
    label = variant_label(spec, metadata)
    record.update(
        {
            "source_trajectory_id": identity,
            "identity_source": identity_source,
            "variant_label": label,
            "variant_id": f"{identity}::{label}",
            "variant_index": metadata.get("variant_index"),
            "task_key": task_key(spec, metadata, relpath),
            "suite": suite_key(spec, metadata, relpath),
            "scene_token": scene_token(metadata),
            "rgb_sibling": rgb_sibling(spec, metadata, rgb_root),
        }
    )
    record["io"] = {
        "bytes_read": file_io.bytes_read,
        "read_seconds": round(time.perf_counter() - started, 6),
    }
    io.bytes_read += file_io.bytes_read
    io.files += 1
    io.seconds += record["io"]["read_seconds"]
    return record


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


# ---------------------------------------------------------------------------
# 深读抽样
# ---------------------------------------------------------------------------


def select_deep_sample(
    records: Sequence[dict[str, Any]], per_platform: int, seed: int
) -> list[str]:
    """按任务分层轮转抽样，返回 ``relpath`` 列表；与输入顺序无关。"""

    chosen: list[str] = []
    for platform in sorted({record["platform"] for record in records}):
        by_task: dict[str, list[str]] = defaultdict(list)
        for record in records:
            if record["platform"] == platform:
                by_task[record["task_key"]].append(record["relpath"])
        for values in by_task.values():
            values.sort(
                key=lambda item: (
                    hashlib.sha256(f"{seed}:{platform}:{item}".encode()).hexdigest(),
                    item,
                )
            )
        ordered_tasks = stable_order(sorted(by_task), seed)
        picked: list[str] = []
        cursor = 0
        while len(picked) < per_platform and any(by_task[task] for task in ordered_tasks):
            task = ordered_tasks[cursor % len(ordered_tasks)]
            if by_task[task]:
                picked.append(by_task[task].pop(0))
            cursor += 1
        chosen.extend(picked)
    return chosen


def deep_sample_record(
    record: dict[str, Any],
    *,
    spec: PlatformSpec,
    sample_frames: int,
    small_array_max_bytes: int,
) -> dict[str, Any]:
    """对抽样文件做有限值、对齐、饱和、重复帧与布局一致性检查。"""

    path = Path(record["path"])
    io = IoCounter(files=1)
    checks: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        metadata = read_metadata(zf, io)
        white_level = _float_or_none(metadata.get("raw_white_level"))
        camera_values: dict[str, np.ndarray] = {}
        per_frame: dict[str, Any] = {}
        for name, key in spec.cameras.items():
            if f"{key}.npy" not in infos:
                continue
            header = read_header(zf, f"{key}.npy", io)
            indices = frame_indices(int(header["shape"][0]), sample_frames)
            values = read_frames(
                zf, f"{key}.npy", header["shape"], np.dtype(header["dtype"]), indices, io
            )
            camera_values[key] = values
            per_frame[name] = {
                "frames_read": indices,
                "per_frame_min": [float(frame.min()) for frame in values],
                "per_frame_max": [float(frame.max()) for frame in values],
                "per_frame_mean": [float(frame.mean(dtype=np.float64)) for frame in values],
                "per_frame_nonfinite": [
                    int(np.count_nonzero(~np.isfinite(frame))) if frame.dtype.kind == "f" else 0
                    for frame in values
                ],
                "per_frame_bytes": [int(frame.nbytes) for frame in values],
            }
        camera_stats = array_stats(
            np.concatenate([values.reshape(len(values), -1) for values in camera_values.values()])
            if camera_values
            else np.array([], dtype=np.float32),
            white_level=white_level,
        )
        checks.append(
            {
                "name": "finite_values",
                "status": "pass" if camera_stats["nonfinite"] == 0 else "fail",
                "detail": (
                    f"采样像素 {camera_stats['elements']}，非有限 {camera_stats['nonfinite']}"
                ),
            }
        )
        checks.append(
            {
                "name": "camera_value_range",
                "status": "pass",
                "detail": (
                    f"min={camera_stats['min']} max={camera_stats['max']} "
                    f"饱和比例={camera_stats['saturation_fraction']} "
                    f"负值比例={camera_stats['negative_fraction']}"
                ),
            }
        )
        duplicate_pairs: list[list[int]] = []
        if camera_values:
            first = next(iter(camera_values.values()))
            for left in range(len(first)):
                for right in range(left + 1, len(first)):
                    if np.array_equal(first[left], first[right]):
                        duplicate_pairs.append([left, right])
        checks.append(
            {
                "name": "sampled_frame_duplicates",
                "status": "pass" if not duplicate_pairs else "fail",
                "detail": f"重复采样帧对 {duplicate_pairs}",
            }
        )
        small: dict[str, Any] = {}
        for key in (spec.proprio, spec.action, *spec.extra_arrays):
            member = f"{key}.npy"
            if member not in infos:
                continue
            if int(infos[member].file_size) <= small_array_max_bytes:
                values = read_array(zf, member, io)
                small[key] = {
                    "shape": [int(dim) for dim in values.shape],
                    "dtype": str(values.dtype),
                    **array_stats(values, white_level=white_level),
                }
        alignment = _alignment_checks(record, small)
        checks.extend(alignment)
        for key in sorted(small):
            shape = small[key]["shape"]
            if len(shape) == 2 and shape[0] >= 3:
                checks.append(_time_column_check(zf, io, key))
        if spec.key == "robotwin" and {"robot_states", "left_endpose", "right_endpose"} <= set(
            small
        ):
            checks.append(_layout_check(zf, io))
    return {
        "relpath": record["relpath"],
        "platform": record["platform"],
        "source_trajectory_id": record["source_trajectory_id"],
        "task_key": record["task_key"],
        "sampled_camera_frames": per_frame,
        "small_arrays": small,
        "checks": checks,
        "io": {"bytes_read": io.bytes_read},
    }


def _alignment_checks(record: dict[str, Any], small: dict[str, Any]) -> list[dict[str, Any]]:
    frames = record["frames"]
    counts = sorted(frames["by_key"].items())
    declared = frames["declared"]
    status = "pass" if frames["aligned"] else "fail"
    detail = f"declared={declared} 各数组帧数={counts} effective={frames['effective']}"
    checks = [
        {"name": "stream_alignment", "status": status, "detail": detail},
        {
            "name": "missing_rate",
            "status": "pass" if (frames["missing_rate"] or 0.0) == 0.0 else "fail",
            "detail": f"missing_rate={frames['missing_rate']}",
        },
    ]
    if "actions" in small and "robot_states" in small:
        checks.append(
            {
                "name": "proprio_action_frames",
                "status": "pass"
                if small["actions"]["shape"][0] == small["robot_states"]["shape"][0]
                else "fail",
                "detail": (
                    f"actions={small['actions']['shape']} "
                    f"robot_states={small['robot_states']['shape']}"
                ),
            }
        )
    return checks


def _time_column_check(zf: zipfile.ZipFile, io: IoCounter, key: str) -> dict[str, Any]:
    """小数组首列是否等间隔严格单调（时间列的唯一可测特征）。"""

    name = f"time_column_monotonic:{key}"
    values = read_array(zf, f"{key}.npy", io)
    if values.ndim != 2 or values.shape[1] < 1:
        return {"name": name, "status": "not_applicable", "detail": f"{key} shape={values.shape}"}
    column = np.asarray(values[:, 0], dtype=np.float64)
    diffs = np.diff(column)
    steps = sorted({round(float(diff), 9) for diff in diffs.tolist()})
    increasing = bool(diffs.size and np.all(diffs > 0))
    equal_step = increasing and len(steps) == 1
    if equal_step:
        status = "pass"
    elif increasing:
        status = "partial"
    else:
        status = "not_applicable"
    return {
        "name": name,
        "status": status,
        "detail": (
            f"{key}[:,0] 严格单调={increasing} 等间隔={equal_step} 步长集合="
            f"{steps[:6]}{'...' if len(steps) > 6 else ''} "
            f"首={column[0] if column.size else None} 末={column[-1] if column.size else None}；"
            "列语义未在 metadata 声明，不据此认定时间戳可用"
        ),
    }


def _layout_check(zf: zipfile.ZipFile, io: IoCounter) -> dict[str, Any]:
    """RoboTwin 布局：``robot_states`` 是否为左末端/左夹爪/右末端/右夹爪。"""

    proprio = read_array(zf, "robot_states.npy", io)
    left = read_array(zf, "left_endpose.npy", io)
    right = read_array(zf, "right_endpose.npy", io)
    left_delta = float(np.max(np.abs(proprio[:, 0:7] - left)))
    right_delta = float(np.max(np.abs(proprio[:, 8:15] - right)))
    quat_left = np.linalg.norm(left[:, 3:7], axis=1)
    quat_right = np.linalg.norm(right[:, 3:7], axis=1)
    quat_error = float(max(np.max(np.abs(quat_left - 1.0)), np.max(np.abs(quat_right - 1.0))))
    return {
        "name": "endpose_layout",
        "status": "pass"
        if (left_delta < 1e-5 and right_delta < 1e-5 and quat_error < 1e-5)
        else "fail",
        "detail": (
            "robot_states[:,0:7]==left_endpose 最大偏差="
            f"{left_delta:.3e}；[:,8:15]==right_endpose 最大偏差={right_delta:.3e}；"
            f"左右末端四元数模长最大偏差={quat_error:.3e}（wxyz，模长 1）"
        ),
    }


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def aggregate_records(
    records: Sequence[dict[str, Any]], specs: dict[str, PlatformSpec]
) -> dict[str, Any]:
    platforms: dict[str, Any] = {}
    for platform, spec in specs.items():
        subset = [record for record in records if record["platform"] == platform]
        if not subset:
            continue
        member_presence: Counter = Counter()
        metadata_presence: Counter = Counter()
        dtypes: dict[str, Counter] = defaultdict(Counter)
        frame_span: dict[str, dict[str, int]] = {}
        rgb_present = 0
        rgb_absent = 0
        rgb_unknown = 0
        nonfinite_total = 0
        read_modes: Counter = Counter()
        for record in subset:
            member_presence.update(record["keys"])
            metadata_presence.update(record["metadata_keys"])
            for key, header in record["headers"].items():
                dtypes[key][header["dtype"]] += 1
                span = frame_span.setdefault(
                    key, {"min": header["shape"][0], "max": header["shape"][0]}
                )
                if header["shape"]:
                    span["min"] = min(span["min"], header["shape"][0])
                    span["max"] = max(span["max"], header["shape"][0])
            for report in record["arrays"].values():
                nonfinite_total += int(report.get("nonfinite") or 0)
                read_modes[report["read_mode"]] += 1
            sibling = record["rgb_sibling"]
            if sibling["exists"] is True:
                rgb_present += 1
            elif sibling["exists"] is False:
                rgb_absent += 1
            else:
                rgb_unknown += 1
        value_counts = {}
        for key in spec.provenance_metadata:
            counter: Counter = Counter()
            for record in subset:
                if key in record["metadata"]:
                    counter[
                        json.dumps(record["metadata"][key], sort_keys=True, ensure_ascii=False)
                    ] += 1
            if counter:
                value_counts[key] = _top_counter(counter)
        sizes = [record["size_bytes"] for record in subset]
        frames = {
            "aligned": sum(1 for record in subset if record["frames"]["aligned"]),
            "declared_mismatch": sum(
                1
                for record in subset
                if record["frames"]["declared"] is not None
                and record["frames"]["declared"] != record["frames"]["effective"]
            ),
            "max_missing_rate": max((record["frames"]["missing_rate"] or 0.0) for record in subset),
            "effective_total": sum(record["frames"]["effective"] for record in subset),
            "effective_min": min(record["frames"]["effective"] for record in subset),
            "effective_max": max(record["frames"]["effective"] for record in subset),
        }
        identities = Counter(record["source_trajectory_id"] for record in subset)
        variant_ids = Counter(record["variant_id"] for record in subset)
        platforms[platform] = {
            "n_files": len(subset),
            "n_sources": len(identities),
            "n_variants": len(variant_ids),
            "max_variants_per_source": max(identities.values()),
            "n_tasks": len({record["task_key"] for record in subset}),
            "n_suites": len({record["suite"] for record in subset}),
            "sources_from_path_fallback": sum(
                1 for record in subset if record["identity_source"] == "path"
            ),
            "total_bytes": int(sum(sizes)),
            "size_bytes": {
                "min": int(min(sizes)),
                "max": int(max(sizes)),
                "p50": int(sorted(sizes)[len(sizes) // 2]),
            },
            "member_presence": dict(sorted(member_presence.items())),
            "metadata_presence": dict(sorted(metadata_presence.items())),
            "dtypes": {key: dict(counter) for key, counter in sorted(dtypes.items())},
            "frame_span": frame_span,
            "frames": frames,
            "nonfinite_total": nonfinite_total,
            "read_modes": dict(read_modes),
            "rgb_sibling": {"present": rgb_present, "absent": rgb_absent, "unknown": rgb_unknown},
            "value_counts": value_counts,
            "tasks": sorted({record["task_key"] for record in subset}),
            "suites": sorted({record["suite"] for record in subset}),
            "scene_tokens": _top_counter(
                Counter(record["scene_token"] for record in subset if record["scene_token"])
            ),
        }
    totals = {
        "files": sum(item["n_files"] for item in platforms.values()),
        "sources": sum(item["n_sources"] for item in platforms.values()),
        "tasks": sum(item["n_tasks"] for item in platforms.values()),
        "bytes": sum(item["total_bytes"] for item in platforms.values()),
    }
    return {"platforms": platforms, "totals": totals}


# ---------------------------------------------------------------------------
# 拆分
# ---------------------------------------------------------------------------

SPLIT_ORDER = ("train", "dev", "final")


def allocate_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """最大余数法分配：小样本任务的分配必须显式列出，不用相邻帧补数量。"""

    raw = {name: float(ratios.get(name, 0.0)) * n for name in SPLIT_ORDER}
    base = {name: int(np.floor(raw[name])) for name in SPLIT_ORDER}
    remainder = n - sum(base.values())
    fraction = sorted(
        SPLIT_ORDER, key=lambda name: (-(raw[name] - base[name]), SPLIT_ORDER.index(name))
    )
    for name in fraction[:remainder]:
        base[name] += 1
    return base


def build_split(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    ratios: dict[str, float],
    name: str,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """按 ``source_trajectory_id`` 分组、平台与任务内分层分配。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("identity_source") == "path":
            raise ValueError("unverified source identity: file paths cannot prove variant grouping")
        grouped[record["source_trajectory_id"]].append(record)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source, members in grouped.items():
        if source in exclude:
            continue
        strata[(members[0]["platform"], members[0]["task_key"])].append(source)
    assignments: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    small_groups: list[dict[str, Any]] = []
    for (platform, task), sources in sorted(strata.items()):
        ordered = stable_order(sources, seed)
        counts = allocate_counts(len(ordered), ratios)
        cursor = 0
        buckets: dict[str, list[str]] = {}
        for split in SPLIT_ORDER:
            buckets[split] = ordered[cursor : cursor + counts[split]]
            cursor += counts[split]
        for split in SPLIT_ORDER:
            for source in buckets[split]:
                assignments.append(
                    {
                        "source_trajectory_id": source,
                        "split": split,
                        "platform": platform,
                        "task_key": task,
                        "n_files": len(grouped[source]),
                        "relpaths": sorted(member["relpath"] for member in grouped[source]),
                    }
                )
        per_task.append(
            {
                "platform": platform,
                "task_key": task,
                "n_sources": len(ordered),
                "counts": counts,
                "variants_per_source": sorted({len(grouped[source]) for source in ordered}),
            }
        )
        if len(ordered) < 10:
            small_groups.append(
                {
                    "platform": platform,
                    "task_key": task,
                    "n_sources": len(ordered),
                    "members": buckets,
                }
            )
    assignments.sort(
        key=lambda item: (
            item["platform"],
            item["task_key"],
            item["split"],
            item["source_trajectory_id"],
        )
    )
    per_platform: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in assignments:
        per_platform[item["platform"]][item["split"]] += 1
    member_table = [
        {"source_trajectory_id": item["source_trajectory_id"], "split": item["split"]}
        for item in assignments
    ]
    sources_by_split: dict[str, set[str]] = defaultdict(set)
    for item in assignments:
        sources_by_split[item["split"]].add(item["source_trajectory_id"])
    overlap = {
        "train_dev": sorted(sources_by_split["train"] & sources_by_split["dev"]),
        "train_final": sorted(sources_by_split["train"] & sources_by_split["final"]),
        "dev_final": sorted(sources_by_split["dev"] & sources_by_split["final"]),
    }
    excluded_members = sorted(
        record["relpath"] for record in records if record["source_trajectory_id"] in exclude
    )
    return {
        "name": name,
        "seed": seed,
        "ratios": {key: float(value) for key, value in ratios.items()},
        "group_key": "source_trajectory_id",
        "stratify_by": ["platform", "task_key"],
        "n_sources": len(assignments),
        "n_files": sum(item["n_files"] for item in assignments),
        "counts_by_platform": {key: dict(value) for key, value in sorted(per_platform.items())},
        "counts_by_split": {split: len(sources_by_split[split]) for split in SPLIT_ORDER},
        "per_task": per_task,
        "small_group_allocations": small_groups,
        "assignments": assignments,
        "excluded_sources": sorted(exclude),
        "excluded_files": excluded_members,
        "source_overlap": overlap,
        "member_table_sha256": sha256_bytes(canonical_json(member_table)),
    }


def source_multiplicity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["source_trajectory_id"]].append(record)
    multiplicities = Counter(len(members) for members in grouped.values())
    multi = {
        source: sorted(member["relpath"] for member in members)
        for source, members in grouped.items()
        if len(members) > 1
    }
    return {
        "files_per_source": {str(key): int(value) for key, value in sorted(multiplicities.items())},
        "sources_with_multiple_files": _top_counter(Counter({key: 1 for key in multi}), top=20),
        "variants_per_source_max": max(multiplicities) if multiplicities else 0,
    }


def build_holdouts(
    records: Sequence[dict[str, Any]], *, seed: int, ratios: dict[str, float], names: Sequence[str]
) -> dict[str, Any]:
    holdouts: dict[str, Any] = {}
    for name in names:
        if name == "object":
            holdouts[name] = _object_holdout(records, seed=seed, ratios=ratios)
        elif name == "scene":
            holdouts[name] = _scene_holdout(records, seed=seed, ratios=ratios)
        elif name == "camera":
            holdouts[name] = _camera_holdout(records)
        elif name == "unseen_factor_combination":
            holdouts[name] = _factor_holdout(records, seed=seed, ratios=ratios)
        else:
            holdouts[name] = {"status": "unsupported", "reason": f"未知 holdout 名称 {name!r}"}
    return holdouts


def _holdout_split(
    records: Sequence[dict[str, Any]],
    *,
    name: str,
    seed: int,
    ratios: dict[str, float],
    held_out: Sequence[str],
    rule: str,
    platform: str | None,
    checks: dict[str, Any],
) -> dict[str, Any]:
    excluded = frozenset(held_out)
    subset = [record for record in records if platform is None or record["platform"] == platform]
    split = build_split(subset, seed=seed, ratios=ratios, name=name, exclude=excluded)
    return {
        "status": "built" if excluded else "unavailable",
        "level": "source_group",
        "rule": rule,
        "platform": platform,
        "excluded_from_train_dev": True,
        "n_held_out_sources": len(excluded),
        "n_held_out_files": len(split["excluded_files"]),
        "held_out_source_ids": sorted(excluded),
        "train_dev_split": split,
        "checks": checks,
    }


def _object_holdout(
    records: Sequence[dict[str, Any]], *, seed: int, ratios: dict[str, float]
) -> dict[str, Any]:
    libero = [record for record in records if record["platform"] == "libero"]
    object_tasks = sorted(
        {record["task_key"] for record in libero if record["suite"].startswith("libero_object")}
    )
    if not object_tasks:
        return {
            "status": "unavailable",
            "reason": "LIBERO 中未找到 libero_object 套件任务；RoboTwin metadata 无物体身份字段",
        }
    picked = stable_order(object_tasks, seed)[: max(1, min(2, len(object_tasks) // 5))]
    held_out = sorted(
        {record["source_trajectory_id"] for record in libero if record["task_key"] in set(picked)}
    )
    train_tasks = sorted({record["task_key"] for record in libero} - set(picked))
    checks = {
        "held_out_tasks": picked,
        "held_out_tasks_present_in_train": sorted(set(picked) & set(train_tasks)),
        "evidence": "LIBERO-Object 每个任务对应一个被操作物体；任务级 holdout 即物体级 holdout",
        "robotwin": "metadata 只有 task_name/episode，无物体身份字段 → 未构建",
    }
    return _holdout_split(
        libero,
        name="object",
        seed=seed,
        ratios=ratios,
        held_out=held_out,
        rule="hold out every source trajectory of selected libero_object tasks as whole groups",
        platform="libero",
        checks=checks,
    )


def _scene_holdout(
    records: Sequence[dict[str, Any]], *, seed: int, ratios: dict[str, float]
) -> dict[str, Any]:
    libero = [record for record in records if record["platform"] == "libero"]
    scenes = sorted({record["scene_token"] for record in libero if record["scene_token"]})
    if not scenes:
        return {
            "status": "unavailable",
            "reason": "task_name 中未解析出 <SCENE><n> token；RoboTwin 无场景字段",
        }
    picked = stable_order(scenes, seed)[:1]
    held_out = sorted(
        {
            record["source_trajectory_id"]
            for record in libero
            if record["scene_token"] in set(picked)
        }
    )
    train_scenes = sorted(set(scenes) - set(picked))
    checks = {
        "held_out_scenes": picked,
        "train_scenes": train_scenes,
        "held_out_scenes_present_in_train": sorted(set(picked) & set(train_scenes)),
        "evidence": "task_name 前缀 <SCENE><n> 由正则 ^([A-Z][A-Z0-9_]*_SCENE\\d+)_ 解析",
        "robotwin": "metadata 无场景字段 → 未构建",
    }
    return _holdout_split(
        libero,
        name="scene",
        seed=seed,
        ratios=ratios,
        held_out=held_out,
        rule="hold out every source trajectory of the selected scene token",
        platform="libero",
        checks=checks,
    )


def _camera_holdout(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for platform, spec in SPECS.items():
        present = sorted(
            {
                name
                for record in records
                if record["platform"] == platform
                for name, key in spec.cameras.items()
                if key in record["headers"]
            }
        )
        if len(present) < 2:
            views[platform] = {
                "status": "unavailable",
                "reason": f"平台 {platform} 只有 {len(present)} 路相机，无法留出一路",
            }
            continue
        held_out = present[-1]
        views[platform] = {
            "status": "built",
            "level": "view",
            "held_out_view": held_out,
            "train_views": [name for name in present if name != held_out],
            "excluded_source_ids": [],
            "shares_sources_with_train": True,
            "reason": (
                "view 级 holdout：同一源轨迹的其他视角仍可用于训练/适配，"
                "留出视角只做零样本与适配分报；因此不排除任何 source_trajectory_id"
            ),
        }
    return {"status": "built", "level": "view", "views": views}


def _factor_holdout(
    records: Sequence[dict[str, Any]], *, seed: int, ratios: dict[str, float]
) -> dict[str, Any]:
    per_platform: dict[str, Any] = {}
    for platform in sorted({record["platform"] for record in records}):
        subset = [record for record in records if record["platform"] == platform]
        counter = Counter(
            str(record["metadata"].get("lighting_domain"))
            for record in subset
            if record["metadata"].get("lighting_domain") is not None
        )
        if not counter:
            per_platform[platform] = {
                "status": "unavailable",
                "reason": "metadata 无 lighting_domain 字段",
            }
            continue
        ordered = stable_order(sorted(counter), seed)
        least = min(ordered, key=lambda domain: (counter[domain], ordered.index(domain)))
        held_out = sorted(
            {
                record["source_trajectory_id"]
                for record in subset
                if str(record["metadata"].get("lighting_domain")) == least
            }
        )
        held_tasks = {
            record["task_key"]
            for record in subset
            if record["source_trajectory_id"] in set(held_out)
        }
        train_tasks = {
            record["task_key"]
            for record in subset
            if record["source_trajectory_id"] not in set(held_out)
        }
        split = build_split(
            subset,
            seed=seed,
            ratios=ratios,
            name="unseen_factor_combination:train_dev",
            exclude=frozenset(held_out),
        )
        per_platform[platform] = {
            "status": "built",
            "level": "source_group",
            "rule": (
                "留出该平台样本数最少的 lighting_domain 的全部源轨迹；"
                "被留出源覆盖的任务仍出现在 train/dev，因此是新的成像因素组合而不是新任务"
            ),
            "held_out_factor": {"lighting_domain": least, "n_files": len(held_out)},
            "n_held_out_sources": len(held_out),
            "held_out_source_ids": held_out,
            "held_out_tasks_absent_from_train": sorted(held_tasks - train_tasks),
            "train_dev_split": split,
        }
    return {"status": "built", "level": "source_group", "platforms": per_platform}


# ---------------------------------------------------------------------------
# 覆盖表与 telemetry provenance
# ---------------------------------------------------------------------------


def _platform_evidence(aggregates: dict[str, Any], platform: str) -> dict[str, Any]:
    return aggregates["platforms"].get(platform, {})


def _keys_matching(presence: dict[str, int], pattern: str) -> dict[str, int]:
    regex = re.compile(pattern, re.IGNORECASE)
    return {key: value for key, value in presence.items() if regex.search(key)}


def build_coverage(
    aggregates: dict[str, Any],
    deep_summary: dict[str, Any],
    *,
    reference_root: Path | None,
) -> dict[str, Any]:
    """字段覆盖表 + 不可用条件清单；每行带证据、证据等级与是否已核对。"""

    fields: list[dict[str, Any]] = []
    for spec in (LIBERO_SPEC, ROBOTWIN_SPEC):
        platform = spec.key
        evidence = _platform_evidence(aggregates, platform)
        present_keys = evidence.get("member_presence", {})
        meta = evidence.get("metadata_presence", {})
        values = evidence.get("value_counts", {})
        n_files = evidence.get("n_files", 0)
        dtypes = evidence.get("dtypes", {})
        exec_keys = _keys_matching(present_keys, r"executed|command|control|ctrl|vcmd|action_exec")
        exec_meta = _keys_matching(meta, r"executed|command|control|ctrl|timing|action_log")
        time_keys = _keys_matching(meta, r"time|stamp|clock|fps|latency|arrival")
        exposure_keys = _keys_matching(
            meta,
            r"exposure|lighting|shutter|sensor_saturation|_gain$|^gain$|_iso$|^iso$",
        )
        calib_keys = _keys_matching(meta, r"intrinsic|extrinsic|calib|distortion|k_matrix|focal")
        calib_members = _keys_matching(present_keys, r"intrin|calib|distort|extrin|^k$|_k$")
        pose_keys = _keys_matching(meta, r"pose|eef|endpose|kinematic")
        noise_keys = _keys_matching(meta, r"noise|shot|read_noise|white_level")
        render_modes = {
            key: values.get(key)
            for key in ("render_order", "raw_protocol", "replay_mode", "clean_protocol")
            if values.get(key)
        }
        executed = _executed_action_verdict(
            platform, render_modes, present_keys.get(spec.action, 0), n_files
        )
        has_time_column, time_detail = _timestamp_evidence(
            platform, time_keys, present_keys, deep_summary
        )
        layout_passes = _deep_check_passes(deep_summary, platform, "endpose_layout")
        camera_dtypes = sorted(
            {tuple(sorted(dtypes.get(key, {}))) for key in spec.cameras.values()}
        )
        frames = evidence.get("frames", {})
        fields.append(
            {
                "platform": platform,
                "n_files": n_files,
                "rows": [
                    _row(
                        "camera_measurement",
                        "相机测量数组",
                        "available",
                        f"成员 {sorted(spec.cameras.values())} 在 {n_files} 个文件中全部存在；"
                        f"相机 dtype={camera_dtypes}；"
                        "形状与帧数见 aggregates.frame_span",
                        [],
                    ),
                    _row(
                        "alignment",
                        "帧对齐/有效帧",
                        "available" if frames.get("aligned") == n_files else "partial",
                        f"全对齐文件 {frames.get('aligned')}/{n_files}；"
                        "declared 与实测不一致 "
                        f"{frames.get('declared_mismatch')} 个；最大 missing_rate="
                        f"{frames.get('max_missing_rate')}；"
                        f"有效帧合计 {frames.get('effective_total')}",
                        [],
                    ),
                    _row(
                        "executed_action",
                        "实际执行动作",
                        executed["status"],
                        (
                            f"匹配 executed/command/control 的数组键={exec_keys or '无'}；"
                            f"metadata 键={exec_meta or '无'}；{executed['evidence']}"
                        ),
                        executed["unavailable_conditions"],
                        evidence_level=executed["evidence_level"],
                        verified=executed["verified"],
                        provisional=executed["provisional"],
                    ),
                    _row(
                        "timestamp",
                        "时间戳",
                        "unavailable",
                        (
                            "Lead 裁决：2403 个 NPZ 的键并集中无时间/帧率键，metadata 无 fps → "
                            "只能 time_unit=frame。本项目实测：metadata 时间类键="
                            f"{time_keys or '无'}；{time_detail}"
                        ),
                        [
                            "无绝对时钟、无 fps、无曝光开始/中点/结束、无帧到达时间、无 clock_id",
                            "帧号不等于秒：禁止由帧序换算秒或构造事件时刻",
                            (
                                "存在等间隔单调首列，但列语义未声明；只能登记为未决观测，"
                                "不得当作可用时间戳"
                                if has_time_column
                                else "没有任何可用的时间列"
                            ),
                        ],
                        evidence_level="negative_scan",
                        verified=True,
                    ),
                    _row(
                        "exposure",
                        "曝光/照明",
                        "partial" if exposure_keys else "unavailable",
                        f"metadata 键={sorted(exposure_keys) or '无'}；"
                        f"lighting_domain 取值={values.get('lighting_domain')}",
                        [
                            "有照明域与 EV/scale 标量，无曝光时间、积分区间、快门或增益标定",
                            "EV/scale 是渲染生成参数，不是实测曝光量",
                        ],
                        evidence_level="metadata_only",
                    ),
                    _row(
                        "joint_state",
                        "关节状态",
                        "partial",
                        (
                            f"本体键={spec.proprio}（dtype={dtypes.get(spec.proprio)}）；"
                            f"动作键={spec.action}（dtype={dtypes.get(spec.action)}）；"
                            + (
                                "LIBERO states 含未标注语义的机器人状态块，"
                                "robot_states 为 8 维末端+夹爪"
                                if platform == "libero"
                                else f"robot_states 为双臂末端+夹爪（深读 {layout_passes} 个样本 "
                                "layout 检查通过）；actions 为 14 维向量，其关节量解释来自"
                                "作者缓存协议，metadata 未声明"
                            )
                        ),
                        [
                            "NPZ 内没有关节名、单位、来源声明，也没有独立的当前关节测量键",
                            "RoboTwin 需要作者缓存协议才能把 actions[t] 解释为关节量",
                        ],
                        evidence_level="measured" if platform == "robotwin" else "metadata_only",
                        verified=platform == "robotwin",
                        provisional=platform != "robotwin",
                    ),
                    _row(
                        "camera_calibration",
                        "相机标定",
                        "available" if (calib_keys or calib_members) else "unavailable",
                        (
                            f"metadata 标定类键={calib_keys or '无'}；"
                            f"成员键={calib_members or '无'}；Lead 裁决：RoboTwin 的 "
                            "camera.get_intrinsic_matrix() 只在运行时算出、不写入缓存；"
                            "LIBERO 由 MuJoCo 运行时定义 → 缓存不含 K/畸变/外参/时间偏移"
                        ),
                        [
                            "没有内参 K、畸变模型、外参/基线、坐标约定、标定版本"
                            "或相机-本体时间偏移",
                            "只能外挂名义标定做诊断，并显式标注为假设",
                        ],
                        evidence_level="producer_code_order",
                        verified=True,
                    ),
                    _row(
                        "pose",
                        "位姿",
                        "partial" if (spec.proprio in present_keys) else "unavailable",
                        (
                            f"本体末端键={spec.proprio} 在 {present_keys.get(spec.proprio, 0)}/"
                            f"{n_files} 文件中存在；metadata 位姿类键={pose_keys or '无'}；"
                            + (
                                "LIBERO 的 8 维布局（3 位置 + 3 旋转 + 2 夹爪）依数值范围推断，"
                                "无生产者核对"
                                if platform == "libero"
                                else f"RoboTwin 布局经深读 {layout_passes} 个样本核对："
                                "robot_states=[left_endpose(7), left_gripper, right_endpose(7), "
                                "right_gripper]"
                            )
                        ),
                        [
                            "无相机外参、无基座/世界系声明、无物体位姿",
                            "四元数约定（wxyz）来自数值核对，不是 metadata 声明",
                        ],
                        evidence_level="measured",
                        verified=platform == "robotwin",
                        provisional=platform != "robotwin",
                    ),
                    _row(
                        "noise_model",
                        "噪声模型",
                        "available"
                        if {"raw_sensor_shot_noise", "raw_sensor_read_noise"} <= set(meta)
                        else "partial",
                        f"metadata 噪声键={sorted(noise_keys) or '无'}；noise_seed 在 "
                        f"{meta.get('noise_seed', 0)}/{n_files} 文件中存在",
                        []
                        if {"raw_sensor_shot_noise", "raw_sensor_read_noise"} <= set(meta)
                        else [
                            "只有 render_order 字符串提到固定噪声与每文件的 noise_seed",
                            "没有 shot/read 噪声参数、白电平或每帧噪声场",
                        ],
                        evidence_level="metadata_only",
                    ),
                    _row(
                        "rgb_pair",
                        "RGB 配对",
                        "available"
                        if evidence.get("rgb_sibling", {}).get("present")
                        else "unavailable",
                        f"配对文件存在 {evidence.get('rgb_sibling')}",
                        ["配对 RGB 不是原生 Bayer；只作独立基线与预览"],
                        evidence_level="measured",
                    ),
                    _row(
                        "source_identity",
                        "源身份/变体",
                        "available"
                        if evidence.get("sources_from_path_fallback", 0) == 0
                        else "partial",
                        f"源轨迹 {evidence.get('n_sources')} 个（路径回退 "
                        f"{evidence.get('sources_from_path_fallback', 0)} 个），"
                        f"每个源的变体数上限 {evidence.get('max_variants_per_source')}",
                        ["本份数据每个源只有 1 个变体文件，无法用真实多照明变体验证分组不泄漏"],
                        evidence_level="measured",
                    ),
                ],
            }
        )
    return {
        "schema": "n04-s0-coverage-v1",
        "status_levels": ["available", "partial", "unavailable"],
        "fields": fields,
        "lead_rulings": _lead_rulings(),
        "references": {
            "telemetry_source_audit": _reference_status(TELEMETRY_SOURCE_AUDIT, reference_root)
        },
        "unavailable_conditions": _unavailable_conditions(aggregates),
    }


def _executed_action_verdict(
    platform: str, render_modes: dict[str, Any], action_members: int, n_files: int
) -> dict[str, Any]:
    """「实际执行动作」可用性：实测协议字符串 + Lead 裁决的生产脚本证据。"""

    flat = json.dumps(render_modes, ensure_ascii=False)
    if platform == "robotwin":
        return {
            "status": "unavailable",
            "evidence_level": "producer_code_order",
            "verified": False,
            "provisional": False,
            "evidence": (
                "metadata 渲染协议取值=" + flat + "；Lead 裁决：metadata 引用的 "
                "clean_replay_record / source_trajectory 不在交付数据里，无法核对对应"
            ),
            "unavailable_conditions": [
                "RAW 渲染明确不执行动作（copy state then render without action）",
                "引用的干净重放记录与源轨迹文件未随数据交付",
                "没有逐帧下发的控制值、命令 ID、执行时间区间、限幅记录或执行状态",
                "只有帧索引（frame 单位），无法给出 action→曝光时刻的映射",
            ],
        }
    if "without_action" in flat:
        return {
            "status": "unavailable",
            "evidence_level": "measured",
            "verified": False,
            "provisional": False,
            "evidence": "渲染协议字符串=" + flat + "：该渲染不执行动作",
            "unavailable_conditions": ["渲染协议说明未执行动作"],
        }
    if action_members and action_members == n_files:
        return {
            "status": "partial",
            "evidence_level": "producer_code_order",
            "verified": False,
            "provisional": True,
            "evidence": (
                f"动作数组成员={action_members}/{n_files}；渲染协议取值={flat}；"
                "Lead 裁决（provisional）：生产脚本执行顺序为 get_state() → 渲染 frame t → 记录 "
                "actions[t] → 才 step(action)，因此 actions[t] 是可用的已执行动作候选，"
                "但未做同版本重放逐帧核对"
            ),
            "unavailable_conditions": [
                "未做同版本重放的逐帧核对（verified=false）",
                "没有下发/限幅/执行区间与命令 ID，无法确认实际下发值",
                "只有帧索引（frame 单位），无法给出 action→曝光时刻的映射",
            ],
        }
    return {
        "status": "unavailable",
        "evidence_level": "negative_scan",
        "verified": False,
        "provisional": False,
        "evidence": f"动作数组成员={action_members}/{n_files}：没有动作记录",
        "unavailable_conditions": ["没有动作记录：无法构造已执行控制"],
    }


def _timestamp_evidence(
    platform: str,
    time_keys: dict[str, int],
    present_keys: dict[str, int],
    deep_summary: dict[str, Any],
) -> tuple[bool, str]:
    """返回 ``(是否存在单调列, 证据文本)``；列存在也不等于时间戳可用。"""

    prefix = f"{platform}.time_column_monotonic:"
    passed_arrays: list[str] = []
    uneven_arrays: list[str] = []
    n_arrays = 0
    for key, counts in deep_summary.get("check_counts", {}).items():
        if not key.startswith(prefix):
            continue
        n_arrays += sum(int(value) for value in counts.values())
        if counts.get("pass"):
            passed_arrays.append(key[len(prefix) :])
        if counts.get("partial"):
            uneven_arrays.append(key[len(prefix) :])
    if passed_arrays:
        return (
            True,
            f"深读样本检查了 {n_arrays} 个小数组前导列，"
            f"{sorted(set(passed_arrays))} 等间隔严格单调；列语义未在 metadata 声明，"
            "只登记为未决观测",
        )
    return (
        False,
        f"深读样本检查了 {n_arrays} 个小数组前导列，没有任何等间隔严格单调列"
        + (f"（{sorted(set(uneven_arrays))} 单调但步长不固定）" if uneven_arrays else ""),
    )


def _deep_check_passes(deep_summary: dict[str, Any], platform: str, name: str) -> int:
    return int(deep_summary.get("check_counts", {}).get(f"{platform}.{name}", {}).get("pass", 0))


def _reference_status(path_text: str, root: Path | None) -> dict[str, Any]:
    """记录被引用产物的存在性与哈希；缺失就写缺失，不假装存在。"""

    candidates = []
    if root is not None:
        candidates.append(root / path_text)
    candidates.append(Path(path_text))
    for candidate in candidates:
        if candidate.is_file():
            return {
                "path": path_text,
                "resolved_path": str(candidate),
                "present_at_run": True,
                "sha256": sha256_file(candidate),
            }
    return {"path": path_text, "resolved_path": None, "present_at_run": False, "sha256": None}


def _row(
    field: str,
    label: str,
    status: str,
    evidence: str,
    conditions: list[str],
    *,
    evidence_level: str = "measured",
    verified: bool = True,
    provisional: bool = False,
) -> dict[str, Any]:
    """一行覆盖结论；``provisional`` 为真时禁止当作 available 使用。"""

    assert status in ("available", "partial", "unavailable"), status
    if provisional:
        assert status != "available", "provisional 结论不得写成 available"
    return {
        "field": field,
        "label": label,
        "status": status,
        "evidence_level": evidence_level,
        "verified": verified,
        "provisional": provisional,
        "evidence": evidence,
        "unavailable_conditions": conditions,
    }


def _lead_rulings() -> dict[str, Any]:
    """Lead 在 S0 对本模块覆盖结论的裁决（原样登记，避免自行改判）。"""

    return {
        "source": "Lead 裁决 2026-10-14（本会话消息）",
        "rulings": [
            {
                "field": "timestamp",
                "verdict": "两平台均不可用",
                "reason": (
                    "2403 个 NPZ 的键并集中无时间/帧率键、metadata 无 fps → 只能 time_unit=frame"
                ),
            },
            {
                "field": "camera_calibration",
                "verdict": "两平台均不可用",
                "reason": (
                    "RoboTwin 的 camera.get_intrinsic_matrix() 只在运行时算出、不写入缓存；"
                    "LIBERO 由 MuJoCo 运行时定义"
                ),
            },
            {
                "field": "executed_action",
                "verdict": "LIBERO=partial/provisional（未验证）；RoboTwin=不可用",
                "reason": (
                    "LIBERO 证据是生产脚本执行顺序 get_state()→渲染 frame t→记录 actions[t]→"
                    "才 step(action)，evidence_level=producer_code_order、verified=false；"
                    "RoboTwin 的 clean_replay_record/source_trajectory 不在交付数据里"
                ),
            },
            {
                "field": "partial",
                "verdict": "不得把 partial 写成 available",
                "reason": "不可用条件清单必须显式给出",
            },
        ],
        "telemetry_source_audit": TELEMETRY_SOURCE_AUDIT,
    }


def _unavailable_conditions(aggregates: dict[str, Any]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for platform in aggregates["platforms"]:
        conditions.append(
            {
                "condition": f"{platform}: 使用 executed_action 作为因果输入",
                "verdict": "禁止",
                "reason": (
                    "缓存没有逐帧执行记录；RoboTwin 的 raw_protocol 明确是"
                    "「复制干净物理状态后逐帧渲染、不执行动作」，LIBERO 只有 replay_mode=actions"
                ),
            }
        )
        conditions.append(
            {
                "condition": f"{platform}: 把帧号当秒或把 EV 当曝光时间",
                "verdict": "禁止",
                "reason": "无绝对时间字段；EV/scale 是渲染生成参数",
            }
        )
        conditions.append(
            {
                "condition": f"{platform}: 使用相机标定做几何传播",
                "verdict": "禁止",
                "reason": "无 K/畸变/外参，必须借用名义标定并显式标注为假设",
            }
        )
    conditions.append(
        {
            "condition": "把三通道 RAW 当原生 Bayer CFA",
            "verdict": "禁止",
            "reason": (
                "representation 分别为 raw（uint8）与 three_channel_linear_float32，均为三通道渲染"
            ),
        }
    )
    conditions.append(
        {
            "condition": "训练/开发集与 final 共享源轨迹",
            "verdict": "禁止",
            "reason": (
                "split 按 source_trajectory_id 分组，splits.json 的 source_overlap 三项必须为空"
            ),
        }
    )
    return conditions


def build_telemetry_provenance(
    aggregates: dict[str, Any], *, reference_root: Path | None
) -> dict[str, Any]:
    """逐项 telemetry provenance：来源键、实测取值分布、语义与缺口。"""
    items: list[dict[str, Any]] = []
    for spec in (LIBERO_SPEC, ROBOTWIN_SPEC):
        platform = spec.key
        evidence = _platform_evidence(aggregates, platform)
        values = evidence.get("value_counts", {})
        meta = evidence.get("metadata_presence", {})
        present_keys = evidence.get("member_presence", {})
        n_files = evidence.get("n_files", 0)
        for item, keys, note, missing in (
            (
                "frame_index",
                [],
                "NPZ 没有帧号数组：帧序即数组第 0 维索引，time_unit=frame",
                ["无帧号/时间戳数组，无法检测丢帧或重复帧的真实采集时间"],
            ),
            (
                "episode_time",
                ["lighting_ev"],
                "不可用：metadata 无时间/帧率键（Lead 裁决）；time_unit=frame",
                [
                    "2403 个 NPZ 的键并集中无时间/帧率键、metadata 无 fps",
                    "LIBERO states 首列等间隔单调，但列语义未声明，只登记为未决观测",
                    "无绝对时钟、无曝光时刻、无到达时间",
                ],
            ),
            (
                "exposure",
                ["lighting_ev", "lighting_scale", "lighting_sample_unit", "sensor_saturation_ev"],
                "照明域与 EV/scale；按文件的照明随机量",
                ["无曝光时间/增益/快门标定"],
            ),
            (
                "noise",
                ["noise_seed", "raw_sensor_shot_noise", "raw_sensor_read_noise", "raw_white_level"],
                "噪声种子与（RoboTwin）噪声常数；LIBERO 只有 render_order 描述",
                ["LIBERO 无噪声参数；两者都无每帧噪声场或实测方差"],
            ),
            (
                "proprio",
                [spec.proprio],
                "每帧本体/末端向量随数组存储",
                ["无采样时刻、无关节名与单位"],
            ),
            (
                "action",
                [spec.action],
                (
                    "已执行动作候选（provisional，未验证）：生产脚本顺序 "
                    "get_state()→渲染 frame t→记录 actions[t]→才 step(action)"
                    if platform == "libero"
                    else "不可用：raw_protocol 说明渲染不执行动作，且引用的 "
                    "clean_replay_record/source_trajectory 不在交付数据里"
                ),
                [
                    "无下发/限幅/执行区间/命令 ID，无法确认实际下发值",
                    "无同版本重放逐帧核对（verified=false）",
                ],
            ),
            (
                "calibration",
                [],
                "不可用（Lead 裁决）：RoboTwin 的 camera.get_intrinsic_matrix() 只在运行时算出、"
                "不写入缓存；LIBERO 由 MuJoCo 运行时定义",
                ["无 K、畸变、外参、时间偏移"],
            ),
            (
                "source_identity",
                list(spec.identity_metadata),
                "源身份键；变体标签另见 variant_id",
                ["缺少数据集级版本/生成代码提交"],
            ),
        ):
            item_values = {key: values.get(key) for key in keys if key in values}
            item_values = {key: value for key, value in item_values.items() if value}
            present = {key: int(meta.get(key, present_keys.get(key, 0))) for key in keys}
            items.append(
                {
                    "item": f"{platform}.{item}",
                    "platform": platform,
                    "source_keys": [key for key in keys],
                    "presence": present,
                    "n_files": n_files,
                    "measured_values": item_values,
                    "semantics": note,
                    "gaps": missing,
                }
            )
    items.append(
        {
            "item": "global.sidecar",
            "platform": "both",
            "source_keys": [],
            "presence": {},
            "n_files": aggregates["totals"]["files"],
            "measured_values": {},
            "semantics": "S0 未挂接 sidecar：没有逐帧对应证据，不把重放结果挂在原 NPZ 名下",
            "gaps": ["执行控制日志、绝对时间、标定与曝光时刻全部缺失"],
        }
    )
    return {
        "schema": "n04-s0-telemetry-provenance-v1",
        "source": "RawVLA-Bench NPZ metadata_json 与数组成员（只读）",
        "lead_rulings": _lead_rulings(),
        "references": {
            "telemetry_source_audit": _reference_status(TELEMETRY_SOURCE_AUDIT, reference_root)
        },
        "items": items,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build_inventory(
    config: dict[str, Any],
    *,
    config_path: Path,
    roots: dict[str, Path],
    platforms: Sequence[str],
    limit: int | None,
    deep_per_platform: int | None,
    split_seed: int | None,
    reference_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_cfg = config.get("manifest", {})
    pattern = str(manifest_cfg.get("npz_glob", "**/*.npz"))
    deep_cfg = dict(manifest_cfg.get("deep_sample", {}))
    sample_frames = int(deep_cfg.get("max_frames_read", 4))
    deep_per_platform = int(
        deep_cfg.get("per_platform", 24) if deep_per_platform is None else deep_per_platform
    )
    deep_seed = int(deep_cfg.get("seed", config.get("split_seed", 20261014)))
    split_seed = int(config.get("split_seed", 20261014) if split_seed is None else split_seed)
    ratios = {key: float(value) for key, value in config.get("split_ratio", {}).items()}
    holdout_names = list(manifest_cfg.get("splits", {}).get("holdouts", []))

    io = IoCounter()
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    per_platform_files: dict[str, int] = {}
    for platform in platforms:
        spec = SPECS[platform]
        root = roots[platform]
        rgb_root = root.parent / "rgb"
        paths = list_npz(root, pattern)
        if limit is not None:
            paths = paths[:limit]
        per_platform_files[platform] = len(paths)
        for path in paths:
            records.append(
                build_record(
                    path,
                    spec=spec,
                    root=root,
                    rgb_root=rgb_root if rgb_root.is_dir() else None,
                    sample_frames=sample_frames,
                    small_array_max_bytes=DEFAULT_SMALL_ARRAY_MAX_BYTES,
                    io=io,
                )
            )
    records.sort(key=lambda item: (item["platform"], item["relpath"]))
    scan_seconds = time.perf_counter() - started

    deep_started = time.perf_counter()
    chosen = select_deep_sample(records, deep_per_platform, deep_seed)
    by_relpath = {(record["platform"], record["relpath"]): record for record in records}
    deep_records = []
    for relpath in chosen:
        for platform in platforms:
            record = by_relpath.get((platform, relpath))
            if record is None:
                continue
            deep_records.append(
                deep_sample_record(
                    record,
                    spec=SPECS[platform],
                    sample_frames=sample_frames,
                    small_array_max_bytes=DEFAULT_SMALL_ARRAY_MAX_BYTES,
                )
            )
    deep_seconds = time.perf_counter() - deep_started

    aggregates = aggregate_records(records, SPECS)
    splits = {
        "schema": "n04-s0-splits-v1",
        "seed": split_seed,
        "ratios": ratios,
        "variants_stay_together": True,
        "group_key": "source_trajectory_id",
        "identity_note": (
            "LIBERO=dataset_name:task_name:demo_key；RoboTwin=task_name:source_trajectory。"
            "rawvla_sequence.source_identity 的 robotwin 分支返回 robotwin:<task>:<dataset_seed>，"
            "会把同一任务全部 episode 折叠成一个源；本清单不使用该分支"
        ),
        "source_multiplicity": source_multiplicity(records),
        "main": build_split(records, seed=split_seed, ratios=ratios, name="main"),
        "holdouts": build_holdouts(records, seed=split_seed, ratios=ratios, names=holdout_names),
    }
    deep_summary = summarize_deep(deep_records)
    if reference_root is None:
        local_root = config.get("local", {}).get("repo_root")
        reference_root = Path(local_root) if local_root else None
    coverage = build_coverage(aggregates, deep_summary, reference_root=reference_root)
    provenance = build_telemetry_provenance(aggregates, reference_root=reference_root)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    inventory = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "roots": {key: str(value) for key, value in roots.items()},
        "platforms_scanned": list(platforms),
        "limit_per_platform": limit,
        "definitions": {
            "effective_frames": "相机/本体/动作数组都能对齐到的最小帧数",
            "missing_rate": (
                "1 - effective_frames / max(declared_frames, 各数组最大帧数)；全对齐为 0"
            ),
            "aligned": "所有轨迹数组帧数一致且等于 metadata num_frames",
            "sampled_frames": f"相机大数组按帧采样（最多 {sample_frames} 帧，含首尾）；"
            "小数组整读，read_mode 标明实际读取方式",
            "content_sha256": "去掉 run/generated_utc/read_seconds 等易变段后的规范化 JSON 哈希",
        },
        "counts": {
            "files": len(records),
            "files_by_platform": per_platform_files,
            "sources_by_platform": {
                platform: evidence["n_sources"]
                for platform, evidence in aggregates["platforms"].items()
            },
            "bytes_by_platform": {
                platform: evidence["total_bytes"]
                for platform, evidence in aggregates["platforms"].items()
            },
        },
        "aggregates": aggregates,
        "records": records,
        "deep_sample": {
            "per_platform": deep_per_platform,
            "seed": deep_seed,
            "max_frames_read": sample_frames,
            "selected": [item["relpath"] for item in deep_records],
            "summary": deep_summary,
            "records": deep_records,
        },
        "run": {
            "wall_seconds": round(time.perf_counter() - started, 3),
            "scan_seconds": round(scan_seconds, 3),
            "deep_seconds": round(deep_seconds, 3),
            "peak_rss_kb": peak_rss,
            "io": io.as_dict(),
            "io_mib_per_second_scan": round(io.bytes_read / max(scan_seconds, 1e-9) / (1 << 20), 3),
            "measured_on": {
                "python": __import__("platform").python_version(),
                "numpy": np.__version__,
            },
        },
    }
    extras = {"splits": splits, "coverage": coverage, "provenance": provenance}
    return inventory, extras


def summarize_deep(deep_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    check_counts: dict[str, Counter] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []
    for item in deep_records:
        for check in item["checks"]:
            check_counts[f"{item['platform']}.{check['name']}"][check["status"]] += 1
            if check["status"] == "fail":
                failures.append(
                    {"relpath": item["relpath"], "check": check["name"], "detail": check["detail"]}
                )
    return {
        "n_records": len(deep_records),
        "check_counts": {
            key: {status: int(count) for status, count in sorted(counter.items())}
            for key, counter in sorted(check_counts.items())
        },
        "failures": failures,
        "bytes_read": sum(item["io"]["bytes_read"] for item in deep_records),
    }


def build_summary(inventory: dict[str, Any], hashes: dict[str, Any]) -> dict[str, Any]:
    """本地小型结果：去掉逐文件记录，且不重复伴随文件里的 splits/coverage/provenance。"""

    platforms = {}
    for platform, evidence in inventory["aggregates"]["platforms"].items():
        platforms[platform] = {
            key: value for key, value in evidence.items() if key not in ("value_counts",)
        }
        platforms[platform]["value_counts"] = evidence.get("value_counts", {})
    return {
        "schema": inventory["schema"],
        "generated_utc": inventory["generated_utc"],
        "config": inventory["config"],
        "roots": inventory["roots"],
        "counts": inventory["counts"],
        "definitions": inventory["definitions"],
        "aggregates": {"platforms": platforms, "totals": inventory["aggregates"]["totals"]},
        "deep_sample": {
            "per_platform": inventory["deep_sample"]["per_platform"],
            "seed": inventory["deep_sample"]["seed"],
            "max_frames_read": inventory["deep_sample"]["max_frames_read"],
            "selected": inventory["deep_sample"]["selected"],
            "summary": inventory["deep_sample"]["summary"],
            "records": inventory["deep_sample"]["records"],
        },
        "run": inventory["run"],
        "hashes": hashes,
        "companions": {
            name: hashes.get(name)
            for name in ("main_split_sha256", "coverage_sha256", "telemetry_provenance_sha256")
        },
    }


def build_hashes(
    inventory: dict[str, Any],
    extras: dict[str, Any],
    *,
    code_path: Path | None,
) -> dict[str, Any]:
    stripped = strip_volatile(inventory)
    split_main = extras["splits"]["main"]
    return {
        "schema": "n04-s0-inventory-hashes-v1",
        "content_sha256": sha256_bytes(canonical_json(stripped)),
        "records_sha256": sha256_bytes(canonical_json(stripped["records"])),
        "aggregates_sha256": sha256_bytes(canonical_json(stripped["aggregates"])),
        "deep_sample_sha256": sha256_bytes(canonical_json(stripped["deep_sample"])),
        "main_split_sha256": sha256_bytes(canonical_json(extras["splits"]["main"])),
        "main_member_table_sha256": split_main["member_table_sha256"],
        "coverage_sha256": sha256_bytes(canonical_json(extras["coverage"])),
        "telemetry_provenance_sha256": sha256_bytes(canonical_json(extras["provenance"])),
        "config_sha256": inventory["config"]["sha256"],
        "code_sha256": sha256_file(code_path) if code_path and code_path.is_file() else None,
        "counts": inventory["counts"],
    }


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
    path.write_text(text + "\n", encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m owac.raw.n04_manifest",
        description="N04 S0 RawVLA-Bench 逐 NPZ 清单、源轨迹分组与冻结拆分（只读）。",
    )
    parser.add_argument("--config", required=True, type=Path, help="s0_data.json 路径（只读）")
    parser.add_argument("--output", required=True, type=Path, help="inventory.json 输出路径")
    parser.add_argument(
        "--platform",
        action="append",
        choices=sorted(SPECS),
        default=None,
        help="只扫描指定平台（可重复）；默认两个平台",
    )
    parser.add_argument(
        "--platform-root",
        action="append",
        default=[],
        metavar="PLATFORM=DIR",
        help="覆盖配置里的数据根（fixture/迁移用）",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="每个平台最多处理多少个 NPZ（smoke 用）"
    )
    parser.add_argument(
        "--deep-per-platform", type=int, default=None, help="覆盖 deep_sample.per_platform"
    )
    parser.add_argument("--split-seed", type=int, default=None, help="覆盖 split_seed")
    parser.add_argument("--summary-out", type=Path, default=None, help="额外写出小型汇总 JSON")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="解析被引用产物（如 ws-telemetry 来源表）的根目录；默认用配置 local.repo_root",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, str] = {}
    for item in args.platform_root:
        if "=" not in item:
            parser.error(f"--platform-root 需要 PLATFORM=DIR，收到 {item!r}")
        key, value = item.split("=", 1)
        if key not in SPECS:
            parser.error(f"未知平台 {key!r}")
        overrides[key] = value
    config = json.loads(args.config.read_text(encoding="utf-8"))
    platforms = args.platform or sorted(SPECS)
    roots = platform_roots(config, overrides)
    for platform in platforms:
        if not roots[platform].is_dir():
            raise SystemExit(f"数据根不存在：{platform} -> {roots[platform]}")

    inventory, extras = build_inventory(
        config,
        config_path=args.config,
        roots=roots,
        platforms=platforms,
        limit=args.limit,
        deep_per_platform=args.deep_per_platform,
        split_seed=args.split_seed,
        reference_root=args.reference_root,
    )
    hashes = build_hashes(inventory, extras, code_path=Path(__file__))
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, inventory)
    write_json(output.parent / "splits.json", extras["splits"])
    write_json(output.parent / "coverage.json", extras["coverage"])
    write_json(output.parent / "telemetry-provenance.json", extras["provenance"])
    write_json(output.parent / "inventory-hashes.json", hashes)
    if args.summary_out is not None:
        summary = build_summary(inventory, hashes)
        write_json(args.summary_out, summary)
    report = {
        "output": str(output),
        "counts": inventory["counts"],
        "content_sha256": hashes["content_sha256"],
        "main_split": {
            "counts_by_split": extras["splits"]["main"]["counts_by_split"],
            "member_table_sha256": extras["splits"]["main"]["member_table_sha256"],
        },
        "deep_sample_failures": len(inventory["deep_sample"]["summary"]["failures"]),
        "run": inventory["run"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

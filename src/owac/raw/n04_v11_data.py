"""N04 v1.1 官方 RawVLA-Bench 数据接入：源分组、窗口取样与本体上下文。

只使用官方 RawVLA-Bench 训练池（LIBERO 1,771 + RoboTwin 632 条），字段与目标相位按
实读字节固定（不做 RGB 重建、不补采、不改量纲）：

LIBERO
    ``agentview_raw_uint8`` / ``wrist_raw_uint8`` 三通道 uint8 pseudo-RAW；
    本体 ``robot_states[t]``（8 维）；目标 ``actions[t:t+10]``（7 维）。

RoboTwin
    ``head`` / ``left`` / ``right`` 的 ``*_raw_float32`` 三通道 float32 HDR-derived RAW；
    ``actions[t]`` 是**当前 14 维关节状态**（本体），目标从 ``actions[t+1:t+51]`` 起；
    ``robot_states`` 的 16 维末端表示不能当 14 维关节。

拆分复用 S0 已建立的官方源划分（同源光照／渲染变体必须同侧）。没有 telemetry sidecar 时，
本体上下文只由 ``≤ t`` 的记录构造（见 :func:`proprio_history_tensor`）；未来监督动作只由
``WindowBatch.targets`` 取用，绝不进入本体上下文。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.core.encoder_contract_n04 import MeasurementFrame
from owac.raw.rawvla_sequence_n04 import (
    RawVLASequence,
    RawVLASequenceLoader,
    npz_member_shape,
    v11_supervision_offsets,
)

__all__ = [
    "DOCUMENTED_DATA_ROOT",
    "PLATFORM_SPECS",
    "ActionNormalizer",
    "PlatformSpec",
    "SequenceEntry",
    "V11DataIndex",
    "WindowBatch",
    "load_window_batch",
    "fixed_isp_rgb",
    "proprio_history_tensor",
    "sample_windows",
    "saturation_mask",
    "scaled_raw",
    "window_candidates",
]

#: 文档记录的官方数据部署位置；本机不可达时如实报缺失，不拿合成数据冒充。
DOCUMENTED_DATA_ROOT = "/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data"


@dataclass(frozen=True)
class PlatformSpec:
    """平台级事实：字段名、本体语义、目标相位与固定缩放。"""

    name: str
    raw_dir: str
    camera_ids: tuple[str, ...]
    camera_keys: tuple[str, ...]
    proprio_key: str
    proprio_expr: str
    proprio_dim: int
    action_dim: int
    action_horizon: int
    target_phase: str
    raw_dtype: str
    #: 平台原始单位域（仅记录；缩放由编码器 InputScale 统一施加）
    value_scale: str
    note: str


PLATFORM_SPECS: dict[str, PlatformSpec] = {
    "libero": PlatformSpec(
        name="libero",
        raw_dir="libero/raw",
        camera_ids=("agentview", "wrist"),
        camera_keys=("agentview_raw_uint8", "wrist_raw_uint8"),
        proprio_key="robot_states",
        proprio_expr="robot_states[t]",
        proprio_dim=8,
        action_dim=7,
        action_horizon=10,
        target_phase="actions[t:t+10]",
        raw_dtype="uint8",
        value_scale="uint8_0_255",
        note="三通道 uint8 pseudo-RAW（RGB 逆处理），不是 Bayer CFA",
    ),
    "robotwin": PlatformSpec(
        name="robotwin",
        raw_dir="robotwin2/raw",
        camera_ids=("head", "left", "right"),
        camera_keys=(
            "head_camera_raw_float32",
            "left_camera_raw_float32",
            "right_camera_raw_float32",
        ),
        proprio_key="actions",
        proprio_expr="actions[t]（当前 14 维关节状态）",
        proprio_dim=14,
        action_dim=14,
        action_horizon=50,
        target_phase="actions[t+1:t+51]",
        raw_dtype="float32",
        value_scale="float32_linear",
        note="三通道 float32 HDR-derived RAW；范围依作者元数据，不无条件除以 255",
    ),
}


def scaled_raw(values: torch.Tensor, platform: str) -> torch.Tensor:
    """**平台原始单位，不在这里缩放**：LIBERO 保留 uint8 值域 ``[0,255]``（转 FP32），
    RoboTwin 保留 FP32 HDR-derived 原值。

    平台固定缩放只允许在**一处**施加：编码器侧的 ``InputScale``
    （``contracts.input_scale_for(model.scale_name)``，由 ``preprocess_frames`` 乘进骨干输入）。
    这里再除一次 255 就等于缩了两次——LIBERO 会被压到 ``std ≤ 0.0026``，RAFT 的 census／
    photometric 同时卡在 robust floor（WS-A 2026-10-15 实测），饱和标记也永远为空。
    因此本函数只做 dtype 归一（→ FP32），数值单位保持平台原样。
    """
    PLATFORM_SPECS[platform]  # 早失败：未知平台
    return values.to(torch.float32)


def fixed_isp_rgb(values: torch.Tensor, *, scale_factor: float) -> torch.Tensor:
    """A/B 的固定同源 ISP：固定曝光、[0,1] 裁剪、sRGB 传递函数。

    输入和输出都用原存储单位（LIBERO 0..255，RoboTwin 线性域乘 1）。编码器
    后续仍只施加一次 InputScale。无逐帧统计、自动曝光、RGB 教师或独立 RGB 数据。
    这是项目固定对照，不声称复现作者 ISP；HDR 裁剪只发生于 A/B RGB 对照。
    """
    if scale_factor <= 0:
        raise ValueError("ISP scale_factor 必须为正")
    linear = values.to(torch.float32) * float(scale_factor)
    if not bool(torch.isfinite(linear).all()):
        raise ValueError("ISP 输入含 NaN/Inf")
    linear = linear.clamp(0.0, 1.0)
    rgb = torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055,
    )
    return rgb / float(scale_factor)


def saturation_mask(
    values: torch.Tensor, platform: str, *, white_level: float | None = None
) -> torch.Tensor | None:
    """**单帧** ``[C,H,W]`` 的饱和标记 → ``[1,H,W]``（对 channel 取 any）。

    四个调用点（窗口构造、训练器 ``_saturation``、策略服务、可视化）都传单帧，所以归约维必须
    是 channel（``dim=0``）。此前写成 ``dim=1``（[B,C,H,W] 约定），对 [C,H,W] 会产出
    ``[C,1,W]``，再被 ``_merge_saturation`` 折成 ``[1,W]`` 横条 OR 进整幅平面——饱和通道
    因此一直标记错误（2026-10-15 定位并修复，训练随之重启）。
    """
    if platform == "libero":
        # 原始单位是 uint8 域 [0,255]；满量程即饱和（缩放只发生在编码器 InputScale）
        return (values >= 255.0).any(dim=0, keepdim=True)
    if white_level is None:
        return None
    return (values >= float(white_level)).any(dim=0, keepdim=True)


# --------------------------------------------------------------------------------------
# 源分组与拆分
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceEntry:
    """官方池里的一条序列。``split`` 来自既有官方源划分，同源变体同侧。"""

    platform: str
    relpath: str
    path: Path
    source_id: str
    split: str
    suite: str = ""
    task_key: str = ""

    def identity(self) -> str:
        return self.source_id


def load_split_table(
    path: str | Path,
) -> tuple[dict[str, tuple[str, str, str, str]], dict[str, Any]]:
    """读既有官方源划分表：``relpath -> (source_id, split, platform, task_key)``。

    表由 S0 清单流程生成（``research/experiments/results/2026-10-14_n04_manifest/splits.json``），
    覆盖面就是官方 1,771 + 632 条；这里只做读取与"同源变体不跨拆分"的核查。
    """
    payload = json.loads(Path(path).read_text())
    assignments = payload["main"]["assignments"]
    table: dict[str, tuple[str, str, str, str]] = {}
    per_source: dict[str, set[str]] = {}
    for item in assignments:
        source = str(item["source_trajectory_id"])
        split = str(item["split"])
        platform = str(item["platform"])
        task_key = str(item.get("task_key", ""))
        per_source.setdefault(source, set()).add(split)
        for relpath in item["relpaths"]:
            table[str(relpath)] = (source, split, platform, task_key)
    crossed = {source: sorted(splits) for source, splits in per_source.items() if len(splits) > 1}
    if crossed:
        raise ValueError(f"同源变体跨拆分，拒绝使用：{crossed}")
    report = {
        "source": str(path),
        "schema": payload.get("schema"),
        "seed": payload.get("seed"),
        "counts_by_split": payload["main"].get("counts_by_split"),
        "n_files": len(table),
        "n_sources": len(per_source),
    }
    return table, report


class V11DataIndex:
    """官方数据根目录下的训练／开发源索引（只读 npz 头，不读测量数据本身）。"""

    def __init__(
        self,
        *,
        root: str | Path,
        split_file: str | Path,
        platform_dirs: dict[str, str] | None = None,
        splits: Sequence[str] = ("train", "dev"),
    ) -> None:
        self.root = Path(root)
        self.platform_dirs = dict(
            platform_dirs or {name: spec.raw_dir for name, spec in PLATFORM_SPECS.items()}
        )
        self.splits = tuple(splits)
        self.table, self.split_report = load_split_table(split_file)
        self._entries: dict[tuple[str, str], list[SequenceEntry]] = {}
        self._unmapped: list[Path] = []
        self._scan()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> V11DataIndex:
        data = config["data"]
        return cls(
            root=data["root"],
            split_file=data["split_file"],
            platform_dirs=data.get("platform_dirs"),
            splits=tuple(data.get("splits", ("train", "dev"))),
        )

    def _scan(self) -> None:
        for platform, relative in self.platform_dirs.items():
            platform_root = self.root / relative
            if not platform_root.exists():
                raise FileNotFoundError(
                    f"找不到 {platform} 官方数据目录 {platform_root}；文档记录的位置是 "
                    f"{DOCUMENTED_DATA_ROOT}/{{libero,robotwin2}}/raw；本机不可达时如实报告缺失"
                )
            for path in sorted(platform_root.rglob("*.npz")):
                relpath = str(path.relative_to(platform_root))
                record = self.table.get(relpath)
                if record is None:
                    self._unmapped.append(path)
                    continue
                source, split, table_platform, task_key = record
                if table_platform != platform:
                    raise ValueError(
                        f"{relpath} 在拆分表里属于 {table_platform}，实际在 {platform}"
                    )
                if split not in self.splits:
                    continue
                suite = relpath.split("/", 1)[0] if "/" in relpath else ""
                self._entries.setdefault((platform, split), []).append(
                    SequenceEntry(
                        platform=platform,
                        relpath=relpath,
                        path=path,
                        source_id=source,
                        split=split,
                        suite=suite,
                        task_key=task_key,
                    )
                )

    # -- 查询 ----------------------------------------------------------------------

    def platforms(self) -> tuple[str, ...]:
        return tuple(self.platform_dirs)

    def entries(self, platform: str, split: str) -> tuple[SequenceEntry, ...]:
        return tuple(self._entries.get((platform, split), ()))

    def sequence_frames(self, entry: SequenceEntry) -> int:
        """只解析 npy 头得到帧数，计划窗口时不必解压整段测量。"""
        return int(npz_member_shape(entry.path, PLATFORM_SPECS[entry.platform].camera_keys[0])[0])

    def report(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "platform_dirs": dict(self.platform_dirs),
            "splits": list(self.splits),
            "split_table": self.split_report,
            "entries": {
                f"{platform}/{split}": len(self._entries.get((platform, split), ()))
                for platform in self.platforms()
                for split in self.splits
            },
            "unmapped_files": len(self._unmapped),
            "unmapped_examples": [str(path) for path in self._unmapped[:5]],
        }


# --------------------------------------------------------------------------------------
# 窗口取样
# --------------------------------------------------------------------------------------


def window_candidates(
    index: V11DataIndex,
    *,
    platform: str,
    split: str,
    burn_in: int,
    gradient: int,
    stride: int,
) -> list[tuple[SequenceEntry, int]]:
    """``(entry, window start)`` 平面表；由索引与窗口参数唯一确定。"""
    out: list[tuple[SequenceEntry, int]] = []
    length = burn_in + gradient
    for entry in index.entries(platform, split):
        total = index.sequence_frames(entry)
        if total < length:
            continue
        for start in range(0, total - length + 1, int(stride) or length):
            out.append((entry, start))
    return out


def sample_windows(
    candidates: Sequence[tuple[SequenceEntry, int]],
    *,
    seed: int,
    step: int,
    batch: int,
) -> list[tuple[SequenceEntry, int]]:
    """``(seed, step)`` 的纯函数采样：resume 时同一步得到同一 plan，不持有采样器状态。"""
    if batch < 1:
        raise ValueError("batch 至少为 1")
    if not candidates:
        raise ValueError("训练窗口为空：检查数据根目录、拆分与窗口长度")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step)]))
    return [candidates[int(pick)] for pick in rng.integers(0, len(candidates), size=int(batch))]


# --------------------------------------------------------------------------------------
# 窗口读取
# --------------------------------------------------------------------------------------


@dataclass
class WindowBatch:
    """一条 ``burn_in + gradient`` 帧窗口的全部训练输入（CPU、未按统计归一化）。"""

    platform: str
    episode_id: str
    start: int
    camera_ids: tuple[str, ...]
    frame_ids: tuple[int, ...]
    raw: dict[str, torch.Tensor]  # camera -> [T,C,H,W] float32，平台固定缩放
    proprio: torch.Tensor  # [T, H, D] float32，只用 ≤ t 的记录
    proprio_padded: tuple[int, ...]  # 每帧因窗口开头不足而重复的真实帧数
    targets: torch.Tensor  # [T, horizon, A]
    target_mask: torch.Tensor  # [T, horizon]
    supervision_index: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def supervision_points(self) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """有效动作监督点：``mask`` 全零（尾部 padding）的点不参与损失。"""
        out: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for position in self.supervision_index:
            mask = self.target_mask[position]
            if float(mask.sum()) <= 0.0:
                continue
            out.append((int(position), self.targets[position], mask))
        return out

    def stacked_raw(self, position: int) -> torch.Tensor:
        """一帧的多相机堆叠：``[C_cam, C, H, W]``，相机顺序 = ``camera_ids``。"""
        return torch.stack([self.raw[camera][position] for camera in self.camera_ids], dim=0)

    def measurement_frame(self, position: int) -> MeasurementFrame:
        """该帧的核心契约表示：多相机堆叠 ``values [C_cam,C,H,W]``（训练器再拼 batch）。"""
        values = self.stacked_raw(position)
        white_level = self.metadata.get("raw_white_level")
        return MeasurementFrame(
            values=values,
            saturated_mask=saturation_mask(
                values,
                self.platform,
                white_level=None if white_level is None else float(white_level),
            ),
            camera_id="/".join(self.camera_ids),
            frame_ids=(int(self.frame_ids[position]),),
            episode_ids=(self.episode_id,),
            new_measurement=True,
            time_unit="frame",
            metadata={
                "platform": self.platform,
                "camera_order": list(self.camera_ids),
                "value_scale": "encoder_input_scale",
                "source": self.metadata.get("source"),
                "target_phase": self.metadata.get("target_phase"),
                "time_unit": "frame",
            },
            metadata_available={"target_phase": True, "value_scale": True},
        )


def proprio_history_tensor(
    sequence: RawVLASequence, *, history: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """``[T, history+1, D]`` 的本体历史，只用 ``≤ t`` 的记录。

    窗口开头不足 ``history`` 帧时重复该 episode 已知的最早真实帧，并逐帧返回重复帧数
    （``padded``）；不借用未来值、不写零占位、不编造时间。
    """
    if not sequence.availability.proprio:
        raise ValueError("该序列没有本体记录，不能构造本体上下文")
    total = len(sequence)
    width = max(1, int(history) + 1)
    dim = int(sequence.proprio.shape[-1])
    rows = torch.empty((total, width, dim), dtype=torch.float32)
    padded: list[int] = []
    for frame in range(total):
        first = max(0, frame - int(history))
        index = list(range(first, frame + 1))
        pad = width - len(index)
        index = [index[0]] * pad + index
        rows[frame] = sequence.proprio[torch.as_tensor(index, dtype=torch.long)]
        padded.append(pad)
    return rows, tuple(padded)


def load_window_batch(
    entry: SequenceEntry,
    *,
    start: int,
    burn_in: int,
    gradient: int,
    supervision_points: int,
    proprio_history: int,
    loader: RawVLASequenceLoader | None = None,
) -> WindowBatch:
    """读一条窗口并整理成训练输入；动作保持原始数值，归一化由训练器按冻结统计做。"""
    spec = PLATFORM_SPECS[entry.platform]
    length = burn_in + gradient
    loader = loader or RawVLASequenceLoader(suite=entry.platform)
    sequence = loader.load_window(entry.path, length=length, start=start)
    if len(sequence) < length:
        raise ValueError(f"{entry.relpath} 在 [{start}, {start + length}) 上不足窗口长度")
    supervision_index = tuple(
        int(burn_in + offset)
        for offset in v11_supervision_offsets(int(gradient), int(supervision_points))
    )
    raw = {
        camera: scaled_raw(sequence.measurements[camera], entry.platform)
        for camera in sequence.camera_ids
    }
    proprio, padded = proprio_history_tensor(sequence, history=proprio_history)
    metadata = dict(sequence.metadata)
    metadata.update(
        {
            "platform": entry.platform,
            "relpath": entry.relpath,
            "source_id": entry.source_id,
            "split": entry.split,
            "camera_ids": list(sequence.camera_ids),
            "value_range": {key: list(value) for key, value in sequence.value_range.items()},
            "target_phase": spec.target_phase,
            "proprio_expr": spec.proprio_expr,
            "window": {"start": int(start), "burn_in": int(burn_in), "gradient": int(gradient)},
            "time_unit": sequence.time_unit,
        }
    )
    white_level = metadata.get("raw_white_level")
    if white_level is not None:
        metadata["raw_white_level"] = float(white_level)
    return WindowBatch(
        platform=entry.platform,
        episode_id=sequence.episode_id,
        start=int(start),
        camera_ids=tuple(sequence.camera_ids),
        frame_ids=tuple(int(value) for value in sequence.frame_ids),
        raw=raw,
        proprio=proprio,
        proprio_padded=padded,
        targets=sequence.supervised_action.clone(),
        target_mask=sequence.supervised_action_mask.clone(),
        supervision_index=supervision_index,
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------
# 冻结动作／本体统计
# --------------------------------------------------------------------------------------


#: 真统计文件里的字段名（``action`` 单数与 ``actions`` 复数两种都出现过）。
STATS_ENTRY_KEYS = ("state", "action", "actions")


def _unwrap_stats_entry(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """解掉单具名条目包装：``{名称: {state/action(s)/...}}``（真 LIBERO 文件即此形状）。

    只在顶层没有 state/action(s)、且恰好一个具名条目的内层含这些字段时解包；否则原样返回，
    让缺键错误照实抛出（不猜、不补零）。
    """
    if not isinstance(payload, dict) or any(key in payload for key in STATS_ENTRY_KEYS):
        return payload, None
    if len(payload) == 1:
        name, inner = next(iter(payload.items()))
        if isinstance(inner, dict) and any(key in inner for key in STATS_ENTRY_KEYS):
            return inner, str(name)
    return payload, None


class ActionNormalizer:
    """冻结协议的动作／本体归一化：LIBERO q01/q99 → [-1,1]，RoboTwin MEAN_STD。

    真实文件形状（2026-10-15 远端实测）：

    * LIBERO ``starvla_pi05_libero/dataset_statistics.json`` =
      ``{"openpi_pi05_libero_franka": {"state": {...}, "action": {...}}}``
      （单具名条目，``action`` 单数）；
    * RoboTwin ``pi05_robotwin/policy_norm_stats.json`` =
      ``{"norm_stats": {"state": {...}, "actions": {...}}}``（``actions`` 复数）。

    键缺失时明确报错，不用单位缩放冒充，也不静默补零。
    """

    def __init__(self, platform: str, stats: dict[str, Any]) -> None:
        self.platform = platform
        payload = (
            stats["norm_stats"] if isinstance(stats, dict) and "norm_stats" in stats else stats
        )
        payload, self.entry_name = _unwrap_stats_entry(payload)
        if not isinstance(payload, dict):
            raise ValueError(f"{platform} 统计不是字典：{type(payload).__name__}")
        self.stats = payload
        self.action_key = self._action_key()

    def _action_key(self) -> str:
        for key in ("action", "actions"):
            if key in self.stats:
                return key
        raise KeyError(
            f"{self.platform} 统计缺少 'action'/'actions'；实际键 {sorted(self.stats)}"
            + (f"（具名条目 {self.entry_name}）" if self.entry_name else "")
        )

    @classmethod
    def from_file(cls, platform: str, path: str | Path) -> ActionNormalizer:
        return cls(platform, json.loads(Path(path).read_text()))

    def normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if key not in self.stats:
            raise KeyError(
                f"{self.platform} 统计缺少 {key!r}；实际键 {sorted(self.stats)}"
                + (f"（具名条目 {self.entry_name}）" if self.entry_name else "")
            )
        item = self.stats[key]
        value = value.to(torch.float32)
        width = value.shape[-1]
        if self.platform == "libero":
            for field in ("q01", "q99"):
                if field not in item:
                    raise KeyError(f"LIBERO {key!r} 统计缺少 {field!r}；实际字段 {sorted(item)}")
            low = torch.as_tensor(item["q01"], dtype=torch.float32, device=value.device)
            high = torch.as_tensor(item["q99"], dtype=torch.float32, device=value.device)
            if low.numel() < width or high.numel() < width:
                raise ValueError(
                    f"LIBERO {key!r} 统计只有 {low.numel()} 维，装不下 {width} 维输入；"
                    "不能用错位统计继续算"
                )
            return 2 * (value - low[:width]) / (high[:width] - low[:width] + 1e-6) - 1
        for field in ("mean", "std"):
            if field not in item:
                raise KeyError(f"RoboTwin {key!r} 统计缺少 {field!r}；实际字段 {sorted(item)}")
        mean = torch.as_tensor(item["mean"], dtype=torch.float32, device=value.device)
        std = torch.as_tensor(item["std"], dtype=torch.float32, device=value.device)
        return (value - mean[:width]) / (std[:width] + 1e-8)

    def normalize_actions(self, value: torch.Tensor) -> torch.Tensor:
        """LIBERO 真文件用 ``action``（单数）、RoboTwin 用 ``actions``（复数）。"""
        return self.normalize(self.action_key, value)

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return self.normalize("state", value)

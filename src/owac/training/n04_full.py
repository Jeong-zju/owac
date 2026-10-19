"""S1 正式 RAW 测量的训练、profiling、缓存导出与开发集选择。

设计要点：

* **数据**：只用 S0 验收过的有标签新扩展（成对扩展 + 受控渲染因素套件）。训练项是
  “同一时刻的两个相机视图”，两帧各自独立编码，第二帧只通过对应目标进入损失。
* **因果**：输入只取 ``online_*``（当前帧）与当前内参；``gt_*`` 只作监督，其中
  ``gt_sim_*`` / ``trace_*`` / ``gt_body_*`` 不进入本阶段。
* **可复现**：采样由 ``(seed, step)`` 纯函数决定（`sample_plan`），不持有随机状态；
  resume 只需恢复 step 与优化器状态。checkpoint 保存 model / optimizer / scheduler /
  AMP scaler / RNG 状态 / 数据身份 / 全局步数。
* **训练配方**：AdamW、warmup 5% + cosine、grad clip 1、BF16 只用于骨干前向，
  几何、方差与损失按 FP32 计算（S1 提示词冻结）。

CLI（与 S1 提示词一致）::

    python -m owac.training.n04_full --config <cfg> --mode profile --run-dir <dir>
    python -m owac.training.n04_full --config <cfg> --mode train --run-dir <dir> \
        [--resume <ckpt>]
    python -m owac.training.n04_full --config <cfg> --mode lr-grid --run-dir <dir>
    python -m owac.training.n04_full --config <cfg> --mode export-measurements \
        --checkpoint <ckpt> --run-dir <dir>
    python -m owac.training.n04_full --config <cfg> --mode data-stats --run-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..frameworks.n04.contracts import (
    DenseMeasurement,
    MeasurementFrame,
    SampleGrid,
    build_measurement_id,
    input_scale_for,
)
from ..frameworks.n04.full_config import N04FullConfig, load_config
from ..frameworks.n04.measurement import N04FullMeasurement, build_measurement
from ..raw.n04_extension_loader import LiveExtensionSequence

__all__ = [
    "SceneRef",
    "SequenceStore",
    "build_index",
    "build_scene_views",
    "correspondence_loss",
    "geometry_loss",
    "main",
    "sample_plan",
    "train_main",
]

#: 平台 -> 输入尺度名（S0 冻结的单位换算）。
PLATFORM_SCALE = {
    "libero": "libero_uint8",
    "robotwin2": "robotwin_float_linear",
    "rendered_factor": "rendered_factor_linear",
}
PAIRED_REQUIRED_KEYS = (
    "online_rawvla_rgb3",
    "online_K",
    "gt_depth_m",
    "gt_visible_support",
    "gt_T_W_from_C",
)
FACTOR_REQUIRED_KEYS = ("online_rawvla_rgb3", "input_K", "gt_depth_m", "gt_T_W_from_C")
#: 监督支持里必须排除的字段（透明表面没有唯一深度）。
GEOMETRY_SUPPORT_KEY = "gt_geometry_support"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def savez_deterministic(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """写出字节确定的 ``.npz``：固定 zip 时间戳、排序成员、固定压缩级别。

    ``numpy.savez_compressed`` 会把当前时间写进 zip 头，同一内容两次导出的文件哈希不同；
    缓存身份要求“同一 (源, 帧, 相机) 重复导出一致”，因此这里显式构造 zip。
    """
    import io
    import zipfile

    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# 数据索引与惰性读取
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneRef:
    """一个训练 / 评测项：同一时刻的两个相机视图。"""

    kind: str
    path: str
    frame_index: int
    camera_a: str
    camera_b: str
    camera_index_a: int
    camera_index_b: int
    source_trajectory_id: str
    platform: str
    split: str
    shape_hw: tuple[int, int]
    scale_name: str
    #: 同一源下的变体名（受控渲染的 factor）：只进入 measurement ID 与缓存命名，
    #: 不改变 S0 的拆分身份（`source_trajectory_id`）。
    variant: str = ""

    @property
    def identity_prefix(self) -> str:
        return (
            f"{self.source_trajectory_id}/{self.variant}"
            if self.variant
            else self.source_trajectory_id
        )

    @property
    def shape_key(self) -> tuple[str, tuple[int, int]]:
        return (self.platform, self.shape_hw)

    @property
    def measurement_key(self) -> tuple[str, int, str, str]:
        """(序列文件, 帧号, 相机 A, 相机 B)：采样计划的可比较身份。"""
        return (self.path, self.frame_index, self.camera_a, self.camera_b)

    @property
    def group_label(self) -> str:
        return f"{self.platform}/{self.camera_a}"


class SequenceStore:
    """按序列惰性载入、只缓存测量阶段需要的窄键集合。

    与 `LiveExtensionSequence` / `rendered_training_frame` 使用同一份文件与 schema；
    这里只读取必要子集，避免把 ``gt_segmentation`` 等大数组复制进内存。
    """

    def __init__(self, *, max_frames_per_sequence: int = 0) -> None:
        self._paired_records: dict[str, dict[str, Any]] = {}
        self._paired_frames: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._factor: dict[str, dict[str, np.ndarray]] = {}
        self._max_frames = int(max_frames_per_sequence)

    def paired_record(self, path: str) -> dict[str, Any]:
        if path not in self._paired_records:
            loader = LiveExtensionSequence(path)  # schema 与像素约定校验
            self._paired_records[path] = loader.record
        return self._paired_records[path]

    def paired_frame(self, path: str, index: int) -> dict[str, np.ndarray]:
        key = (path, int(index))
        if key not in self._paired_frames:
            record = self.paired_record(path)
            entry = record["frames"][int(index)]
            with np.load(Path(path).parent / entry["file"], allow_pickle=False) as arrays:
                keep = set(PAIRED_REQUIRED_KEYS) | {GEOMETRY_SUPPORT_KEY}
                self._paired_frames[key] = {
                    name: arrays[name] for name in arrays.files if name in keep
                }
        return self._paired_frames[key]

    def factor_arrays(self, path: str) -> dict[str, np.ndarray]:
        if path not in self._factor:
            with np.load(path, allow_pickle=False) as arrays:
                keep = set(FACTOR_REQUIRED_KEYS) | {"input_frame_index"}
                self._factor[path] = {name: arrays[name] for name in arrays.files if name in keep}
        return self._factor[path]


def _paired_refs(config: N04FullConfig, split: str, store: SequenceStore) -> list[SceneRef]:
    refs: list[SceneRef] = []
    root = Path(config.data.paired_root)
    for record_path in sorted(root.glob("episode-*/record.json")):
        record = store.paired_record(str(record_path))
        if record.get("split") != split:
            continue
        platform = record["metadata"]["platform"]
        camera_names = list(record["metadata"]["camera_names"])
        pairs = config.data.correspondence_pairs[platform]
        scale_name = config.data.scale_names[platform]
        n_frames = len(record["frames"])
        if config.data.max_frames_per_sequence:
            n_frames = min(n_frames, config.data.max_frames_per_sequence)
        with np.load(record_path.parent / record["frames"][0]["file"], allow_pickle=False) as probe:
            raw = probe["online_rawvla_rgb3"]
            shape_hw = (int(raw.shape[-3]), int(raw.shape[-2]))
        for index in range(n_frames):
            for camera_a, camera_b in pairs:
                if camera_a not in camera_names or camera_b not in camera_names:
                    raise ValueError(
                        f"{record_path}: 配对相机 {camera_a}/{camera_b} 不在 {camera_names}"
                    )
                refs.append(
                    SceneRef(
                        kind="paired",
                        path=str(record_path),
                        frame_index=index,
                        camera_a=camera_a,
                        camera_b=camera_b,
                        camera_index_a=camera_names.index(camera_a),
                        camera_index_b=camera_names.index(camera_b),
                        source_trajectory_id=record["source_trajectory_id"],
                        platform=platform,
                        split=split,
                        shape_hw=shape_hw,
                        scale_name=scale_name,
                    )
                )
    return refs


def _factor_refs(config: N04FullConfig, split: str, store: SequenceStore) -> list[SceneRef]:
    refs: list[SceneRef] = []
    manifest_path = Path(config.data.factor_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pairs = config.data.correspondence_pairs["rendered_factor"]
    if len(pairs) != 1:
        raise ValueError("受控渲染因素套件只有一对相机")
    camera_a, camera_b = pairs[0]
    for entry in manifest["sequences"]:
        if entry["split"] != split:
            continue
        path = str(Path(config.data.factor_root) / entry["file"])
        arrays = store.factor_arrays(path)
        n_frames = int(arrays["online_rawvla_rgb3"].shape[0])
        if config.data.max_frames_per_sequence:
            n_frames = min(n_frames, config.data.max_frames_per_sequence)
        shape_hw = (
            int(arrays["online_rawvla_rgb3"].shape[-3]),
            int(arrays["online_rawvla_rgb3"].shape[-2]),
        )
        for index in range(n_frames):
            refs.append(
                SceneRef(
                    kind="factor",
                    path=path,
                    frame_index=index,
                    camera_a=camera_a,
                    camera_b=camera_b,
                    camera_index_a=0,
                    camera_index_b=1,
                    source_trajectory_id=entry["source_trajectory_id"],
                    platform="rendered_factor",
                    split=split,
                    shape_hw=shape_hw,
                    scale_name="rendered_factor_linear",
                    variant=str(entry["factor"]),
                )
            )
    return refs


def build_index(
    config: N04FullConfig, splits: Sequence[str], store: SequenceStore
) -> dict[tuple[str, tuple[int, int]], list[SceneRef]]:
    """按 (平台, 形状) 分组；同一 batch 内的视图必须同形。"""
    groups: dict[tuple[str, tuple[int, int]], list[SceneRef]] = {}
    for split in splits:
        for ref in _paired_refs(config, split, store) + _factor_refs(config, split, store):
            groups.setdefault(ref.shape_key, []).append(ref)
    for key, refs in groups.items():
        if not refs:
            raise ValueError(f"分组 {key} 为空")
    return groups


def sources_per_group(groups: dict[Any, list[SceneRef]]) -> dict[Any, int]:
    return {key: len({ref.source_trajectory_id for ref in refs}) for key, refs in groups.items()}


def sample_plan(
    groups: dict[Any, list[SceneRef]],
    *,
    seed: int,
    step: int,
    batch_frames: int,
    micro_batch_frames: int,
) -> list[list[SceneRef]]:
    """``(seed, step)`` 的纯函数采样，返回按微批分组的场景列表。

    每个微批只从一个 (平台, 形状) 组里取，保证同一批的视图同形且能被同一次前向处理；
    微批之间的组由 softmax 前的加权（概率 ∝ 源轨迹数）抽取，避免受控渲染套件 152 条
    变体占满 batch。每步独立采样：resume 时同一 step 得到同一 plan，不需要保存采样器
    状态。
    """
    if batch_frames % micro_batch_frames or micro_batch_frames % 2:
        raise ValueError("micro_batch_frames 必须为偶数且整除 batch_frames")
    weights = sources_per_group(groups)
    keys = sorted(groups, key=lambda key: (key[0], key[1]))
    total_weight = float(sum(weights[key] for key in keys))
    probabilities = [weights[key] / total_weight for key in keys]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step)]))
    scenes_per_micro = micro_batch_frames // 2
    plan: list[list[SceneRef]] = []
    for _ in range(batch_frames // micro_batch_frames):
        key = keys[int(rng.choice(len(keys), p=probabilities))]
        refs = groups[key]
        picks = [refs[int(index)] for index in rng.integers(0, len(refs), size=scenes_per_micro)]
        plan.append(picks)
    return plan


# --------------------------------------------------------------------------------------
# 视图构造（当前帧输入 + 监督目标）
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewTargets:
    depth: torch.Tensor  # [H,W] 米
    support: torch.Tensor  # [H,W] bool（可见 & 几何支持 & 有限正值）
    K: torch.Tensor  # [3,3] 原图像素
    T_W_from_C: torch.Tensor  # [4,4]
    camera_id: str
    measurement_id: str

    def move_to(self, device: torch.device) -> ViewTargets:
        return ViewTargets(
            depth=self.depth.to(device),
            support=self.support.to(device),
            K=self.K.to(device),
            T_W_from_C=self.T_W_from_C.to(device),
            camera_id=self.camera_id,
            measurement_id=self.measurement_id,
        )


def _raw_bhwc_to_chw(raw: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(raw))
    if tensor.dim() != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"rawvla_rgb3 期望 [H,W,3]，实际 {tuple(tensor.shape)}")
    return tensor.permute(2, 0, 1)


def _depth_support(
    depth: np.ndarray, visible: np.ndarray | None, geometry: np.ndarray | None
) -> tuple[torch.Tensor, torch.Tensor]:
    depth_t = torch.from_numpy(np.asarray(depth, dtype=np.float32))
    support = torch.isfinite(depth_t) & (depth_t > 0)
    if visible is not None:
        support = support & torch.from_numpy(np.asarray(visible, dtype=bool))
    if geometry is not None:
        support = support & torch.from_numpy(np.asarray(geometry, dtype=bool))
    return depth_t, support


def build_scene_views(
    ref: SceneRef, store: SequenceStore
) -> tuple[MeasurementFrame, MeasurementFrame, ViewTargets, ViewTargets]:
    """按 S0 冻结的 schema 读取两个视图；``gt_*`` 只进 `ViewTargets`。"""
    if ref.kind == "paired":
        record = store.paired_record(ref.path)
        arrays = store.paired_frame(ref.path, ref.frame_index)
        sim_time = float(record["frames"][ref.frame_index].get("sim_time_s", "nan"))
        visible = arrays.get("gt_visible_support")
        geometry = arrays.get(GEOMETRY_SUPPORT_KEY)
        pose = arrays["gt_T_W_from_C"]
        frames = []
        targets = []
        for camera_index, camera_id in (
            (ref.camera_index_a, ref.camera_a),
            (ref.camera_index_b, ref.camera_b),
        ):
            measurement_id = build_measurement_id(
                ref.identity_prefix, ref.frame_index, camera_id, "rawvla_rgb3"
            )
            K = torch.from_numpy(np.asarray(arrays["online_K"][camera_index], dtype=np.float32))
            depth, support = _depth_support(
                arrays["gt_depth_m"][camera_index],
                None if visible is None else visible[camera_index],
                None if geometry is None else geometry[camera_index],
            )
            frames.append(
                MeasurementFrame(
                    raw=_raw_bhwc_to_chw(arrays["online_rawvla_rgb3"][camera_index]),
                    camera_id=camera_id,
                    measurement_id=measurement_id,
                    input_mode="rawvla_rgb3",
                    scale=input_scale_for(ref.scale_name),
                    size_hw=ref.shape_hw,
                    K=K,
                    time_s=None if math.isnan(sim_time) else sim_time,
                    time_unit="sim_s",
                    provenance={
                        "platform": ref.platform,
                        "source_trajectory_id": ref.source_trajectory_id,
                        "frame_index": ref.frame_index,
                    },
                )
            )
            targets.append(
                ViewTargets(
                    depth=depth,
                    support=support,
                    K=K,
                    T_W_from_C=torch.from_numpy(
                        np.asarray(pose[camera_index], dtype=np.float64)
                    ).to(torch.float32),
                    camera_id=camera_id,
                    measurement_id=measurement_id,
                )
            )
        return frames[0], frames[1], targets[0], targets[1]
    if ref.kind == "factor":
        arrays = store.factor_arrays(ref.path)
        observed = int(arrays["input_frame_index"][ref.frame_index])
        if not 0 <= observed <= ref.frame_index:
            raise ValueError("非因果的受控渲染帧映射")
        frames = []
        targets = []
        for camera_index, camera_id in (
            (ref.camera_index_a, ref.camera_a),
            (ref.camera_index_b, ref.camera_b),
        ):
            measurement_id = build_measurement_id(
                ref.identity_prefix, ref.frame_index, camera_id, "rawvla_rgb3"
            )
            K = torch.from_numpy(
                np.asarray(arrays["input_K"][observed][camera_index], dtype=np.float32)
            )
            depth, support = _depth_support(
                arrays["gt_depth_m"][ref.frame_index][camera_index], None, None
            )
            frames.append(
                MeasurementFrame(
                    raw=_raw_bhwc_to_chw(arrays["online_rawvla_rgb3"][observed][camera_index]),
                    camera_id=camera_id,
                    measurement_id=measurement_id,
                    input_mode="rawvla_rgb3",
                    scale=input_scale_for(ref.scale_name),
                    size_hw=ref.shape_hw,
                    K=K,
                    time_s=None,
                    time_unit="frame",
                    provenance={
                        "platform": ref.platform,
                        "source_trajectory_id": ref.source_trajectory_id,
                        "frame_index": ref.frame_index,
                    },
                )
            )
            targets.append(
                ViewTargets(
                    depth=depth,
                    support=support,
                    K=K,
                    T_W_from_C=torch.from_numpy(
                        np.asarray(
                            arrays["gt_T_W_from_C"][ref.frame_index][camera_index], dtype=np.float64
                        )
                    ).to(torch.float32),
                    camera_id=camera_id,
                    measurement_id=measurement_id,
                )
            )
        return frames[0], frames[1], targets[0], targets[1]
    raise ValueError(f"未知数据来源类型 {ref.kind!r}")


# --------------------------------------------------------------------------------------
# 目标采样与损失
# --------------------------------------------------------------------------------------


def sample_at_grid(values: torch.Tensor, grid: SampleGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """在融合网格中心取传感器像素值（最近邻），返回 ``(值 [h,w], 域内 [h,w])``。"""
    if values.dim() != 2:
        raise ValueError("values 需要 [H,W]")
    centres = grid.feature_centres().xy.to(values.device)
    ui = centres[..., 0].round()
    vi = centres[..., 1].round()
    height, width = values.shape
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui_c = ui.clamp(0, width - 1).long()
    vi_c = vi.clamp(0, height - 1).long()
    out = values[vi_c, ui_c]
    return torch.where(inside, out, torch.zeros_like(out)), inside


def geometry_loss(
    *,
    depth_pred: torch.Tensor,
    variance_pred: torch.Tensor,
    targets: Sequence[ViewTargets],
    grids: Sequence[SampleGrid],
    config: N04FullConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """有效支持上的归一化深度 SmoothL1 与同单位 Gaussian NLL（含 logdet）。

    向量化实现：全部目标一次搬到计算设备后按融合网格中心 gather；逐场景小算子循环会在
    每步产生上百次同步 H2D 复制，S1 首轮 smoke 实测那是主要瓶颈。
    """
    device = depth_pred.device
    scale = float(config.data.depth_scale_m)
    zero = depth_pred.sum() * 0.0
    empty = {
        "depth_points": torch.zeros((), device=device),
        "smooth": zero.detach(),
        "nll": zero.detach(),
    }
    if not targets:
        return zero, empty
    depths = torch.stack([target.depth.to(torch.float32) for target in targets]).to(device)
    supports = torch.stack([target.support.to(torch.float32) for target in targets]).to(device)
    centres = torch.stack([grid.feature_centres().xy for grid in grids]).to(device)
    scene_count, height, width = depths.shape
    ui = centres[..., 0].round()
    vi = centres[..., 1].round()
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    flat = (vi.clamp(0, height - 1).long() * width + ui.clamp(0, width - 1).long()).view(
        scene_count, -1
    )
    gt_depth = torch.gather(depths.view(scene_count, -1), 1, flat).view_as(inside)
    support = torch.gather(supports.view(scene_count, -1), 1, flat).view_as(inside)
    mask = inside & (support > 0.5) & torch.isfinite(gt_depth) & (gt_depth > 0)
    if not bool(mask.any()):
        return zero, empty
    residual = (depth_pred[:, 0] - gt_depth) / scale
    sigma2 = (variance_pred[:, 0] / scale**2).clamp_min(
        float(config.training.nll_variance_floor_norm)
    )
    ratio = torch.clamp(residual**2 / sigma2, max=float(config.training.nll_ratio_clamp))
    total = mask.sum().to(depth_pred.dtype)
    smooth = (
        F.smooth_l1_loss(
            residual[mask],
            torch.zeros_like(residual[mask]),
            beta=config.training.huber_beta_norm,
            reduction="sum",
        )
        / total
    )
    nll = (0.5 * (ratio + torch.log(sigma2)))[mask].sum() / total
    loss = config.training.w_geometry * smooth + config.training.w_geometry_nll * nll
    return loss, {"depth_points": total, "smooth": smooth.detach(), "nll": nll.detach()}


def correspondence_loss(
    *,
    out_a: DenseMeasurement,
    out_b: DenseMeasurement,
    targets_a: Sequence[ViewTargets],
    targets_b: Sequence[ViewTargets],
    config: N04FullConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """用 GT 深度 / 位姿构造已知投影点，对描述子做带负例的 InfoNCE。

    两帧各自独立编码；第二帧只通过本损失的目标参与，不进入第一帧的输入。
    """
    device = out_a.latent_mean_map.device
    temperature = float(config.training.correspondence_temperature)
    tol = float(config.training.correspondence_depth_tol_m)
    total = int(out_a.batch_size)
    desc_a = out_a.candidates.descriptor.to(device)
    anchors_sensor = out_a.candidates.xy_sensor.to(device)
    valid_a = out_a.candidates.valid.to(device)
    if desc_a.shape[1] != out_b.candidates.count:
        raise ValueError("两个视图的候选数必须相同")
    height_b, width_b = targets_b[0].depth.shape
    matches = []
    for index in range(total):
        grid_b = out_b.sample_grid
        target_a = targets_a[index]
        target_b = targets_b[index]
        sensor = anchors_sensor[index]
        ui = sensor[:, 0].round()
        vi = sensor[:, 1].round()
        height_a, width_a = target_a.depth.shape
        inside_a = (ui >= 0) & (ui < width_a) & (vi >= 0) & (vi < height_a)
        ui_a = ui.clamp(0, width_a - 1).long()
        vi_a = vi.clamp(0, height_a - 1).long()
        depth_a = target_a.depth.to(device)[vi_a, ui_a]
        ok = valid_a[index] & inside_a & target_a.support.to(device)[vi_a, ui_a] & (depth_a > 0)
        rays = _pixel_rays(sensor, target_a.K.to(device))  # [N,3]
        points_a = rays * depth_a.unsqueeze(-1)
        K_b = target_b.K.to(device)
        T_ba = _relative_pose(target_b.T_W_from_C.to(device), target_a.T_W_from_C.to(device))
        points_b = points_a @ T_ba[:3, :3].T + T_ba[:3, 3]
        projected = points_b @ K_b.T
        safe_z = torch.where(points_b[:, 2] > 1e-6, points_b[:, 2], torch.ones_like(points_b[:, 2]))
        uv = projected[:, :2] / safe_z.unsqueeze(-1)
        ok = ok & (points_b[:, 2] > 1e-6)
        # 目标视图的深度一致性（遮挡 / 匹配错误的负例筛除）
        ui_b = uv[:, 0].round()
        vi_b = uv[:, 1].round()
        inside_b = (ui_b >= 0) & (ui_b < width_b) & (vi_b >= 0) & (vi_b < height_b)
        ui_bc = ui_b.clamp(0, width_b - 1).long()
        vi_bc = vi_b.clamp(0, height_b - 1).long()
        depth_b = target_b.depth.to(device)[vi_bc, ui_bc]
        ok = ok & inside_b & target_b.support.to(device)[vi_bc, ui_bc]
        ok = ok & (torch.abs(depth_b - points_b[:, 2]) <= tol)
        matches.append((index, uv, ok))
    if not matches:
        zero = out_a.latent_mean_map.sum() * 0.0
        return zero, {"pairs": torch.zeros((), device=device)}

    desc_b_list = []
    keep_pairs = 0
    src_indices = []
    for index, uv, ok in matches:
        if not bool(ok.any()):
            continue
        grid_b = out_b.sample_grid.with_device(device)
        feature_xy = grid_b.sensor_to_feature.apply(_pixels(uv[ok].to(device), device)).xy
        height_f, width_f = out_b.sample_grid.feature_size_hw
        normalised = torch.stack(
            (
                2.0 * (feature_xy[:, 0] + 0.5) / width_f - 1.0,
                2.0 * (feature_xy[:, 1] + 0.5) / height_f - 1.0,
            ),
            dim=-1,
        )
        sampled = F.grid_sample(
            out_b.latent_mean_map[index : index + 1],
            normalised.view(1, 1, -1, 2),
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        desc_b_list.append(sampled[0, :, 0].T)
        src_indices.append(torch.full((int(ok.sum()),), index, dtype=torch.long, device=device))
        keep_pairs += int(ok.sum())
    if keep_pairs < 2:
        zero = out_a.latent_mean_map.sum() * 0.0
        return zero, {"pairs": torch.tensor(keep_pairs, device=device)}
    src = torch.cat(src_indices)
    desc_b = torch.cat(desc_b_list, dim=0)
    desc_a_kept = torch.cat(
        [desc_a[index][ok] for index, uv, ok in matches if bool(ok.any())], dim=0
    )
    norm_a = F.normalize(desc_a_kept.float(), dim=-1)
    norm_b = F.normalize(desc_b.float(), dim=-1)
    logits = norm_a @ norm_b.T / temperature
    labels = torch.arange(logits.shape[0], device=device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        positive = (norm_a * norm_b).sum(dim=-1).mean()
        same_scene = src[:, None] == src[None, :]
        off_diagonal = ~torch.eye(logits.shape[0], dtype=torch.bool, device=device)
        negatives = (norm_a @ norm_b.T)[off_diagonal & ~same_scene]
        stats = {
            "pairs": torch.tensor(keep_pairs, device=device),
            "positive_cosine": positive.detach(),
            "negative_cosine": (
                negatives.mean() if negatives.numel() else torch.zeros((), device=device)
            ),
            "top1": (logits.argmax(dim=1) == labels).float().mean().detach(),
        }
    return config.training.w_correspondence * loss, stats


def _pixels(xy: torch.Tensor, device: torch.device):
    from ..frameworks.n04.measurement_geometry import PixelPoints

    return PixelPoints(xy=xy.to(device=device, dtype=torch.float32), unit="original_px")


def _pixel_rays(sensor_xy: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(sensor_xy[..., :1])
    homogeneous = torch.cat((sensor_xy, ones), dim=-1)
    K_inv = torch.linalg.inv(K.to(torch.float32))
    return torch.einsum("ij,...j->...i", K_inv, homogeneous)


def _relative_pose(T_W_from_B: torch.Tensor, T_W_from_A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(T_W_from_B) @ T_W_from_A


# --------------------------------------------------------------------------------------
# 训练器
# --------------------------------------------------------------------------------------


@dataclass
class StepBatch:
    refs: list[SceneRef]
    frames_a: list[MeasurementFrame]
    frames_b: list[MeasurementFrame]
    targets_a: list[ViewTargets]
    targets_b: list[ViewTargets]


def build_step_batch(
    refs: Sequence[SceneRef], store: SequenceStore, *, device: torch.device | None = None
) -> StepBatch:
    frames_a, frames_b, targets_a, targets_b = [], [], [], []
    for ref in refs:
        frame_a, frame_b, target_a, target_b = build_scene_views(ref, store)
        if device is not None:
            # 模型参数在 device 上，测量必须用同一设备；否则 autocast 会按 CPU 设备启用
            frame_a = frame_a.move_to(device=device)
            frame_b = frame_b.move_to(device=device)
            target_a = target_a.move_to(device)
            target_b = target_b.move_to(device)
        frames_a.append(frame_a)
        frames_b.append(frame_b)
        targets_a.append(target_a)
        targets_b.append(target_b)
    return StepBatch(list(refs), frames_a, frames_b, targets_a, targets_b)


class Trainer:
    """S1 训练循环：profiling、正式训练、resume、开发集评测与缓存导出共用同一份前向。"""

    def __init__(
        self,
        config: N04FullConfig,
        *,
        seed: int,
        run_dir: str | Path,
        lr: float | None = None,
        device: str | None = None,
        schedule_steps: int | None = None,
    ) -> None:
        self.config = config
        # 开发网格这类短预算必须按自己的步数建调度器；否则 100k 的 warmup（5000 步）
        # 会让整个网格都停在预热段，LR 之间不可比。
        self.schedule_steps = int(schedule_steps or config.training.steps)
        self.seed = int(seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or config.training.device)
        self.store = SequenceStore(max_frames_per_sequence=config.data.max_frames_per_sequence)
        self.train_groups: dict[Any, list[SceneRef]] = {}
        self.model = build_measurement(config).to(self.device)
        self._set_seed(self.seed)
        self.optimizer = self._build_optimizer(lr)
        self.scheduler = self._build_scheduler()
        # BF16 的指数范围与 FP32 相同，不需要梯度缩放；只有 FP16 才启用 GradScaler。
        use_scaler = (
            self._amp_enabled()
            and self.device.type == "cuda"
            and self.config.training.amp_dtype == "float16"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.step = 0
        self.history: list[dict[str, Any]] = []

    # -- 基础工具 -----------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _amp_enabled(self) -> bool:
        return bool(self.config.training.amp) and self.device.type == "cuda"

    def autocast_dtype(self) -> torch.dtype | None:
        """骨干前向使用的 AMP dtype；几何与损失仍在 FP32。"""
        if not self._amp_enabled():
            return None
        return torch.bfloat16 if self.config.training.amp_dtype == "bfloat16" else torch.float16

    def _build_optimizer(self, lr: float | None) -> torch.optim.Optimizer:
        learning_rate = float(lr if lr is not None else self.config.training.lr)
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith(".bias") or "gamma" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        if not decay and not no_decay:
            raise ValueError("模型没有可训练参数")
        groups = [
            {"params": decay, "weight_decay": self.config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.param_groups_report = {
            "decay_tensors": len(decay),
            "no_decay_tensors": len(no_decay),
            "decay_params": int(sum(p.numel() for p in decay)),
            "no_decay_params": int(sum(p.numel() for p in no_decay)),
        }
        return torch.optim.AdamW(groups, lr=learning_rate, betas=self.config.training.betas)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        total = max(1, int(self.schedule_steps))
        warmup = max(1, int(total * self.config.training.warmup_fraction))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)

    # -- 一步训练 -----------------------------------------------------------------
    def train_step(self, step: int) -> dict[str, Any]:
        config = self.config
        micro = int(config.training.micro_batch_frames or config.training.batch_frames)
        if config.training.batch_frames % micro:
            raise ValueError("micro_batch_frames 必须整除 batch_frames")
        plan = sample_plan(
            self.train_groups,
            seed=self.seed,
            step=step,
            batch_frames=config.training.batch_frames,
            micro_batch_frames=micro,
        )
        accumulation = config.training.batch_frames // micro
        self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = defaultdict(float)
        for chunk in plan:
            batch = build_step_batch(chunk, self.store, device=self.device)
            out_a = self.model.measure(batch.frames_a, autocast_dtype=self.autocast_dtype())
            out_b = self.model.measure(batch.frames_b, autocast_dtype=self.autocast_dtype())
            loss, stats = self.compute_losses(out_a, out_b, batch)
            if self.scaler.is_enabled():
                self.scaler.scale(loss / accumulation).backward()
            else:
                (loss / accumulation).backward()
            totals["count"] += 1
            for key, value in stats.items():
                totals[key] += float(value.detach())
            totals["total"] += float(loss.detach()) / accumulation
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), config.training.grad_clip
        )
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.step = step + 1
        step_count = max(1, int(totals.pop("count", 1)))
        totals["grad_norm"] = float(grad_norm)
        totals["lr"] = float(self.scheduler.get_last_lr()[0])
        return {
            key: value / step_count if key not in ("grad_norm", "lr") else value
            for key, value in totals.items()
        }

    def compute_losses(
        self, out_a: DenseMeasurement, out_b: DenseMeasurement, batch: StepBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 每个场景给出两个视图的目标与网格：测量函数的网格是逐帧构造的，不能只给两个
        scenes = len(batch.targets_a)
        geometry, geometry_stats = geometry_loss(
            depth_pred=torch.cat([out_a.depth_mean, out_b.depth_mean], dim=0),
            variance_pred=torch.cat([out_a.depth_variance, out_b.depth_variance], dim=0),
            targets=[*batch.targets_a, *batch.targets_b],
            grids=[out_a.sample_grid] * scenes + [out_b.sample_grid] * scenes,
            config=self.config,
        )
        correspondence, corr_stats = correspondence_loss(
            out_a=out_a,
            out_b=out_b,
            targets_a=batch.targets_a,
            targets_b=batch.targets_b,
            config=self.config,
        )
        total = geometry + correspondence
        stats = {"geometry": geometry.detach(), "correspondence": correspondence.detach()}
        stats.update({f"geometry_{k}": v for k, v in geometry_stats.items()})
        stats.update({f"correspondence_{k}": v for k, v in corr_stats.items()})
        return total, stats

    # -- checkpoint ---------------------------------------------------------------
    def save_checkpoint(self, path: str | Path, *, changes: dict[str, str] | None = None) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.config.checkpoint_schema,
            "model_kind": self.config.model_kind,
            "config_sha256": self.config.sha256(),
            "seed": self.seed,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "data_identity": self.data_identity(),
            "history": self.history,
            "changes": changes or {},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return _sha256_file(path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != self.config.checkpoint_schema:
            raise ValueError(
                f"checkpoint schema {payload.get('schema')!r} 与 "
                f"{self.config.checkpoint_schema!r} 不匹配"
            )
        if payload.get("config_sha256") != self.config.sha256():
            raise ValueError("checkpoint 的配置哈希与当前配置不一致")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step = int(payload["step"])
        rng = payload.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        return payload

    def warm_cache(self, splits: Sequence[str]) -> dict[str, Any]:
        """把用到的序列一次性读入内存（窄键子集）。

        数据是 S0 新扩展的小集合（数十条序列），整段驻留 RAM 后每步不再解压 npz；
        实测这一步把每步的 1.5 s I/O 降到接近 0，是长跑的主要吞吐来源。
        """
        start = time.time()
        frames = 0
        for split in splits:
            for ref in _paired_refs(self.config, split, self.store) + _factor_refs(
                self.config, split, self.store
            ):
                if ref.kind == "paired":
                    self.store.paired_frame(ref.path, ref.frame_index)
                else:
                    self.store.factor_arrays(ref.path)
                frames += 1
        return {
            "sequences": len(
                {ref.path for ref in _paired_refs(self.config, splits[0], self.store)}
            ),
            "scene_refs": frames,
            "seconds": time.time() - start,
        }

    def data_identity(self) -> dict[str, Any]:
        return {
            "paired_root": self.config.data.paired_root,
            "factor_root": self.config.data.factor_root,
            "input_mode": self.config.input_mode,
            "splits": list(self.config.data.splits),
            "depth_scale_m": self.config.data.depth_scale_m,
            "scale_names": dict(self.config.data.scale_names),
        }

    # -- 开发集评测 ---------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, split: str, *, batch_frames: int | None = None) -> dict[str, Any]:
        self.model.eval()
        groups = build_index(self.config, (split,), self.store)
        batch_frames = int(batch_frames or self.config.training.eval_batch_frames or 64)
        metrics: dict[str, list[float]] = defaultdict(list)
        per_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for key in sorted(groups, key=lambda item: (item[0], item[1])):
            refs = groups[key]
            for start in range(0, len(refs), batch_frames // 2):
                chunk = refs[start : start + batch_frames // 2]
                batch = build_step_batch(chunk, self.store, device=self.device)
                out_a = self.model.measure(batch.frames_a, autocast_dtype=self.autocast_dtype())
                out_b = self.model.measure(batch.frames_b, autocast_dtype=self.autocast_dtype())
                for out, targets in (
                    (out_a, batch.targets_a),
                    (out_b, batch.targets_b),
                ):
                    self._accumulate_metrics(out, targets, metrics, per_group, chunk)
                    self._accumulate_correspondence(out, targets, metrics)
        self.model.train()
        summary: dict[str, Any] = {
            "split": split,
            "depth_norm_mae": _mean(metrics["depth_norm_abs"]),
            "depth_mae_m": _mean(metrics["depth_abs_m"]),
            "depth_median_m": float(np.median(metrics["depth_abs_m"]))
            if metrics["depth_abs_m"]
            else None,
            "depth_rmse_m": float(np.sqrt(np.mean(np.square(metrics["depth_abs_m"]))))
            if metrics["depth_abs_m"]
            else None,
            "depth_nll": _mean(metrics["nll"]),
            "variance_std_m": _mean(metrics["sigma_m"]),
            "coverage_1sigma": _mean(metrics["cover1"]),
            "coverage_2sigma": _mean(metrics["cover2"]),
            "position_error_m": _mean(metrics["position_error_m"]),
            "correspondence_positive_cosine": _mean(metrics["corr_pos"]),
            "correspondence_negative_cosine": _mean(metrics["corr_neg"]),
            "pixels": int(sum(metrics["count"])),
        }
        summary["groups"] = {
            label: {
                "depth_mae_m": _mean(values["depth_abs_m"]),
                "depth_norm_mae": _mean(values["depth_norm_abs"]),
                "coverage_1sigma": _mean(values["cover1"]),
                "pixels": int(sum(values["count"])),
            }
            for label, values in sorted(per_group.items())
        }
        return summary

    def _accumulate_metrics(
        self,
        out: DenseMeasurement,
        targets: Sequence[ViewTargets],
        metrics: dict[str, list[float]],
        per_group: dict[str, dict[str, list[float]]],
        refs: Sequence[SceneRef],
    ) -> None:
        scale = float(self.config.data.depth_scale_m)
        for index, target in enumerate(targets):
            grid = out.sample_grid
            pred = out.depth_mean[index, 0]
            variance = out.depth_variance[index, 0]
            gt_depth, inside = sample_at_grid(target.depth.to(pred.device), grid)
            support, _ = sample_at_grid(target.support.to(pred.device).to(torch.float32), grid)
            mask = inside & (support > 0.5) & torch.isfinite(gt_depth) & (gt_depth > 0)
            if not bool(mask.any()):
                continue
            residual = pred[mask] - gt_depth[mask]
            sigma = torch.sqrt(variance[mask])
            metrics["depth_abs_m"].extend(residual.abs().cpu().tolist())
            metrics["depth_norm_abs"].extend((residual.abs() / scale).cpu().tolist())
            metrics["sigma_m"].extend(sigma.cpu().tolist())
            metrics["nll"].extend(
                (0.5 * (residual**2 / variance[mask] + torch.log(variance[mask]))).cpu().tolist()
            )
            metrics["cover1"].extend((residual.abs() <= sigma).float().cpu().tolist())
            metrics["cover2"].extend((residual.abs() <= 2 * sigma).float().cpu().tolist())
            metrics["count"].append(float(residual.numel()))
            label = f"{refs[index].platform}/{refs[index].camera_a}"
            per_group[label]["depth_abs_m"].extend(residual.abs().cpu().tolist())
            per_group[label]["depth_norm_abs"].extend((residual.abs() / scale).cpu().tolist())
            per_group[label]["cover1"].extend((residual.abs() <= sigma).float().cpu().tolist())
            per_group[label]["count"].append(float(residual.numel()))
            # 同一条射线上横向误差为零；三维位置误差是深度误差乘射线范数
            ray_norm = _ray_norm(grid, target.K)
            metrics["position_error_m"].extend((residual.abs() * ray_norm[mask]).cpu().tolist())

    def _accumulate_correspondence(
        self, out: DenseMeasurement, targets: Sequence[ViewTargets], metrics: dict[str, list[float]]
    ) -> None:
        with torch.no_grad():
            descriptor = out.candidates.descriptor
            if descriptor.shape[1] > 1:
                normed = F.normalize(descriptor.float(), dim=-1)
                similarity = normed @ normed.transpose(1, 2)
                off = ~torch.eye(similarity.shape[1], dtype=torch.bool, device=similarity.device)
                metrics["corr_pos"].append(float(similarity.mean()))
                metrics["corr_neg"].append(float(similarity[:, off].mean()))

    # -- 主循环 -------------------------------------------------------------------
    def run(
        self,
        *,
        steps: int,
        resume: str | None = None,
        eval_split: str | None = "dev",
        stop_on_stability: bool = False,
    ) -> dict[str, Any]:
        config = self.config
        if resume:
            payload = self.load_checkpoint(resume)
            self.history = list(payload.get("history", []))
        self.train_groups = build_index(config, config.data.splits, self.store)
        cache_info = self.warm_cache(config.data.splits)
        print(f"data cache warmed: {cache_info}", flush=True)
        report: dict[str, Any] = {
            "stage": config.stage,
            "mode": "train",
            "seed": self.seed,
            "config_sha256": config.sha256(),
            "data_identity": self.data_identity(),
            "param_groups": self.param_groups_report,
            "train_groups": {
                f"{key[0]}|{key[1][0]}x{key[1][1]}": len(refs)
                for key, refs in sorted(self.train_groups.items(), key=lambda item: str(item[0]))
            },
            "steps_requested": steps,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.model.train()
        start = time.time()
        for step in range(self.step, steps):
            info = self.train_step(step)
            if step % config.training.log_every == 0:
                print(
                    f"step {step:7d} total={info.get('total', float('nan')):.4f} "
                    f"geom={info.get('geometry', float('nan')):.4f} "
                    f"corr={info.get('correspondence', float('nan')):.4f} "
                    f"lr={info['lr']:.3e} grad={info['grad_norm']:.3f}",
                    flush=True,
                )
            if eval_split and (step + 1) % config.training.eval_every == 0:
                metrics = self.evaluate(eval_split)
                metrics["step"] = step + 1
                metrics["train"] = {k: v for k, v in info.items()}
                self.history.append(metrics)
                _write_json(self.run_dir / "metrics.json", self.history)
                summary = {
                    key: metrics.get(key) for key in ("depth_mae_m", "coverage_1sigma", "depth_nll")
                }
                print(
                    "eval step "
                    + str(step + 1)
                    + ": "
                    + ", ".join(
                        f"{key}={value:.5f}" if isinstance(value, float) else f"{key}=nan"
                        for key, value in summary.items()
                    ),
                    flush=True,
                )
                if stop_on_stability and detect_stability(self.history, config)["converged"]:
                    report["stopped_on_stability"] = step + 1
                    break
            if (step + 1) % config.training.checkpoint_every == 0:
                path = self.run_dir / f"checkpoint-step{step + 1:07d}.pt"
                digest = self.save_checkpoint(path)
                report.setdefault("checkpoints", []).append(
                    {"path": str(path), "sha256": digest, "step": step + 1}
                )
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["steps_completed"] = self.step
        report["elapsed_s"] = time.time() - start
        report["history"] = self.history
        report["stability"] = detect_stability(self.history, config)
        final_path = self.run_dir / "checkpoint-final.pt"
        report["final_checkpoint"] = {
            "path": str(final_path),
            "sha256": self.save_checkpoint(
                final_path, changes={"history": "see train-report.json"}
            ),
        }
        _write_json(self.run_dir / "train-report.json", report)
        return report


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _ray_norm(grid: SampleGrid, K: torch.Tensor) -> torch.Tensor:
    centres = grid.feature_centres().xy
    K_inv = torch.linalg.inv(K.to(device=centres.device, dtype=torch.float32))
    ones = torch.ones_like(centres[..., :1])
    rays = torch.einsum("ij,...j->...i", K_inv, torch.cat((centres, ones), dim=-1))
    return rays.norm(dim=-1)


def detect_stability(history: Sequence[dict[str, Any]], config: N04FullConfig) -> dict[str, Any]:
    """**最近** ``stability_window`` 次逐步相对改善全部 < 阈值即记趋稳。

    只检查最近一段：扫全历史会把"早期大幅改善"永久当成未趋稳（初版实现即犯此错，S1
    的 seed 17 因此拿到过 always-still_improving 的报告字段）。指标变差记为负改善，
    同样满足 < 阈值——判据问的是"是否还在改善"，不是"是否在变好"。
    """
    window = int(config.training.stability_window)
    threshold = float(config.training.stability_relative_threshold)
    values = [entry.get("depth_norm_mae") for entry in history]
    values = [value for value in values if value is not None]
    if len(values) < window + 1:
        return {"converged": False, "reason": "insufficient_evals", "evals": len(values)}
    recent = values[-(window + 1) :]
    improvements = [
        (recent[index] - recent[index + 1]) / abs(recent[index]) if recent[index] else 0.0
        for index in range(window)
    ]
    converged = all(improvement < threshold for improvement in improvements)
    return {
        "converged": converged,
        "reason": "recent_window_below_threshold" if converged else "still_improving",
        "evals": len(values),
        "threshold": threshold,
        "last_improvements": [round(value, 5) for value in improvements],
    }


# --------------------------------------------------------------------------------------
# 模式：profile / lr-grid / export / data-stats
# --------------------------------------------------------------------------------------


def profile_main(
    config: N04FullConfig, run_dir: str | Path, *, steps: int | None = None
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    trainer = Trainer(config, seed=config.training.seeds[0], run_dir=run_dir)
    trainer.train_groups = build_index(config, config.data.splits, trainer.store)
    trainer.warm_cache(config.data.splits)
    steps = int(steps or config.training.profile_steps)
    on_cuda = torch.cuda.is_available() and trainer.device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(trainer.device)
    trainer.model.train()
    start = time.time()
    first: dict[str, Any] = {}
    for step in range(steps):
        info = trainer.train_step(step)
        if step == 0:
            first = info
    elapsed = time.time() - start
    grads = gradient_audit(trainer.model)
    report = {
        "mode": "profile",
        "config_sha256": config.sha256(),
        "seed": trainer.seed,
        "steps": steps,
        "elapsed_s": elapsed,
        "s_per_step": elapsed / steps,
        "frames_per_s": steps * config.training.batch_frames / elapsed,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(trainer.device)) if on_cuda else 0
        ),
        "params": int(sum(p.numel() for p in trainer.model.parameters())),
        "param_groups": trainer.param_groups_report,
        "first_step": {k: float(v) for k, v in first.items()},
        "gradient_audit": grads,
        "estimated_gpu_hours_100k": (elapsed / steps) * config.training.steps / 3600.0,
        "estimated_gpu_hours_300k": (elapsed / steps) * config.training.max_steps / 3600.0,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(trainer.device),
            "device_name": torch.cuda.get_device_name(trainer.device) if on_cuda else "cpu",
        },
    }
    _write_json(run_dir / "profile.json", report)
    return report


def gradient_audit(model: N04FullMeasurement) -> dict[str, Any]:
    """每个 stage 是否收到非零有限梯度；顺带报告未参与损失、没有梯度的参数。"""
    missing, zero, nonfinite = [], [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        grad = parameter.grad
        if not bool(torch.isfinite(grad).all()):
            nonfinite.append(name)
        elif float(grad.abs().max()) == 0.0:
            zero.append(name)
    return {
        "params_without_grad": missing,
        "params_zero_grad": zero,
        "params_nonfinite_grad": nonfinite,
        "stage_names": sorted({name.split(".")[0] for name, _ in model.named_parameters()}),
    }


def lr_grid_main(
    config: N04FullConfig, run_dir: str | Path, *, steps: int | None = None
) -> dict[str, Any]:
    steps = int(steps or config.training.lr_grid_steps)
    results = []
    base = Path(run_dir)
    for lr in config.training.lr_grid:
        trainer = Trainer(
            config,
            seed=config.training.seeds[0],
            run_dir=base / f"lr{lr:g}",
            schedule_steps=steps,
        )
        trainer.train_groups = build_index(config, config.data.splits, trainer.store)
        trainer.warm_cache(config.data.splits)
        trainer.model.train()
        start = time.time()
        losses = []
        for step in range(steps):
            info = trainer.train_step(step)
            if step % 100 == 0:
                losses.append(float(info.get("total", float("nan"))))
        metrics = trainer.evaluate("dev")
        results.append(
            {
                "lr": lr,
                "steps": steps,
                "elapsed_s": time.time() - start,
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "dev": metrics,
            }
        )
        print(f"lr {lr:g}: dev depth_norm_mae={metrics['depth_norm_mae']}", flush=True)
    report = {"mode": "lr-grid", "config_sha256": config.sha256(), "results": results}
    _write_json(base / "lr-grid.json", report)
    return report


def data_stats_main(config: N04FullConfig, run_dir: str | Path) -> dict[str, Any]:
    """训练集深度尺度与支持统计（用于冻结 ``depth_scale_m``），不训练网络。"""
    store = SequenceStore()
    stats: dict[str, Any] = {"mode": "data-stats", "splits": {}}
    for split in ("train",):
        depths: list[np.ndarray] = []
        supports = 0
        total = 0
        per_platform: dict[str, list[np.ndarray]] = defaultdict(list)
        refs = _paired_refs(config, split, store) + _factor_refs(config, split, store)
        seen_a: set[tuple[str, int, str]] = set()
        for ref in refs:
            key = (ref.path, ref.frame_index, ref.camera_a)
            if key in seen_a:
                continue
            seen_a.add(key)
            _, _, target_a, _ = build_scene_views(ref, store)
            depth = target_a.depth.numpy()
            support = target_a.support.numpy()
            depths.append(depth[support])
            per_platform[ref.platform].append(depth[support])
            supports += int(support.sum())
            total += int(support.size)
        stacked = np.concatenate(depths) if depths else np.zeros(1)
        stats["splits"][split] = {
            "scenes": len(seen_a),
            "support_pixels": supports,
            "support_fraction": supports / max(1, total),
            "depth_median_m": float(np.median(stacked)),
            "depth_p05_m": float(np.percentile(stacked, 5)),
            "depth_p95_m": float(np.percentile(stacked, 95)),
            "per_platform": {
                platform: {
                    "depth_median_m": float(np.median(np.concatenate(arrays))),
                    "pixels": int(sum(len(array) for array in arrays)),
                }
                for platform, arrays in sorted(per_platform.items())
                if arrays
            },
        }
    _write_json(Path(run_dir) / "data-stats.json", stats)
    return stats


def export_main(
    config: N04FullConfig,
    run_dir: str | Path,
    checkpoint: str,
    *,
    splits: Sequence[str] = ("dev",),
    include_pyramid: bool = False,
    verify_samples: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """导出逐相机测量缓存（测量与 GT 分开文件），并抽样在线复核。"""
    run_dir = Path(run_dir)
    frames_dir = run_dir / "frames"
    gt_dir = run_dir / "gt"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config, seed=config.training.seeds[0], run_dir=run_dir)
    payload = trainer.load_checkpoint(checkpoint)
    trainer.model.eval()
    store = trainer.store
    weights_sha = _sha256_file(checkpoint)
    entries = []
    for split in splits:
        unique: dict[tuple[str, int, str], dict[str, Any]] = {}
        for ref in _paired_refs(config, split, store) + _factor_refs(config, split, store):
            for camera_id, camera_index in (
                (ref.camera_a, ref.camera_index_a),
                (ref.camera_b, ref.camera_index_b),
            ):
                unique.setdefault(
                    (ref.path, ref.frame_index, camera_id),
                    {
                        "kind": ref.kind,
                        "source_path": ref.path,
                        "frame_index": ref.frame_index,
                        "camera_id": camera_id,
                        "camera_index": camera_index,
                        "source_trajectory_id": ref.source_trajectory_id,
                        "platform": ref.platform,
                        "split": ref.split,
                        "shape_hw": ref.shape_hw,
                        "scale_name": ref.scale_name,
                        "variant": ref.variant,
                    },
                )
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                item["platform"],
                item["source_path"],
                item["frame_index"],
                item["camera_id"],
            ),
        )
        if limit:
            ordered = ordered[:limit]
        for item in ordered:
            ref = _ref_from_export_item(item)
            frame, _, target, _ = build_scene_views(ref, store)
            frame = frame.move_to(device=trainer.device)
            with torch.no_grad():
                measurement = trainer.model.measure(
                    [frame], autocast_dtype=trainer.autocast_dtype()
                )
            measurement = _detach_measurement(measurement)
            slug = (
                build_measurement_id(
                    ref.identity_prefix, ref.frame_index, ref.camera_a, config.input_mode
                )
                .replace("/", "__")
                .replace("@", "_at_")
            )
            measurement_path = frames_dir / f"{slug}.npz"
            savez_deterministic(
                measurement_path, measurement.to_cache(include_pyramid=include_pyramid)
            )
            gt_path = gt_dir / f"{slug}.npz"
            savez_deterministic(
                gt_path,
                {
                    "depth": target.depth.numpy().astype(np.float32),
                    "support": target.support.numpy().astype(np.uint8),
                    "K": target.K.numpy().astype(np.float64),
                    "T_W_from_C": target.T_W_from_C.numpy().astype(np.float64),
                },
            )
            entries.append(
                {
                    "measurement_id": measurement.measurement_ids[0],
                    "kind": item["kind"],
                    "source_path": item["source_path"],
                    "camera_index": item["camera_index"],
                    "scale_name": item["scale_name"],
                    "split": item["split"],
                    "platform": item["platform"],
                    "camera_id": item["camera_id"],
                    "source_trajectory_id": item["source_trajectory_id"],
                    "variant": item["variant"],
                    "frame_index": item["frame_index"],
                    "shape_hw": list(item["shape_hw"]),
                    "measurement_path": str(measurement_path.relative_to(run_dir)),
                    "gt_path": str(gt_path.relative_to(run_dir)),
                    "measurement_sha256": _sha256_file(measurement_path),
                    "gt_sha256": _sha256_file(gt_path),
                }
            )
    index = {
        "schema": "n04-measurement-cache-v1",
        "config_sha256": config.sha256(),
        "weights_sha256": weights_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "input_mode": config.input_mode,
        "splits": list(splits),
        "entries": entries,
        "include_pyramid": bool(include_pyramid),
    }
    _write_json(run_dir / "index.json", index)
    verification = verify_export(config, run_dir, index, samples=verify_samples, trainer=trainer)
    _write_json(run_dir / "verify.json", verification)
    return {"mode": "export-measurements", "entries": len(entries), "verification": verification}


def _ref_from_export_item(item: dict[str, Any]) -> SceneRef:
    return SceneRef(
        kind=item["kind"],
        path=item["source_path"],
        frame_index=int(item["frame_index"]),
        camera_a=item["camera_id"],
        camera_b=item["camera_id"],
        camera_index_a=int(item["camera_index"]),
        camera_index_b=int(item["camera_index"]),
        source_trajectory_id=item["source_trajectory_id"],
        platform=item["platform"],
        split=item["split"],
        shape_hw=tuple(int(v) for v in item["shape_hw"]),
        scale_name=item["scale_name"],
        variant=item.get("variant", ""),
    )


def _detach_measurement(measurement: DenseMeasurement) -> DenseMeasurement:
    """缓存导出前切断计算图（只改张量属性，不改数值）。"""
    return DenseMeasurement(
        feature_pyramid=tuple(level.detach() for level in measurement.feature_pyramid),
        latent_mean_map=measurement.latent_mean_map.detach(),
        latent_variance_map=measurement.latent_variance_map.detach(),
        depth_mean=measurement.depth_mean.detach(),
        depth_variance=measurement.depth_variance.detach(),
        valid=measurement.valid,
        support=measurement.support,
        saturated=measurement.saturated,
        sample_grid=measurement.sample_grid,
        candidates=measurement.candidates,
        measurement_ids=measurement.measurement_ids,
        units=measurement.units,
        metadata=measurement.metadata,
    )


def verify_export(
    config: N04FullConfig,
    run_dir: Path,
    index: dict[str, Any],
    *,
    samples: int,
    trainer: Trainer,
) -> dict[str, Any]:
    """抽样重新在线计算，与缓存逐张量比较（容忍离线写盘精度）。"""
    import random as _random

    rng = _random.Random(20261014)
    picked = (
        rng.sample(index["entries"], min(samples, len(index["entries"])))
        if index["entries"]
        else []
    )
    results = []
    for entry in picked:
        ref = _ref_from_export_item(entry)
        frame, _, _, _ = build_scene_views(ref, trainer.store)
        frame = frame.move_to(device=trainer.device)
        with torch.no_grad():
            online = _detach_measurement(
                trainer.model.measure([frame], autocast_dtype=trainer.autocast_dtype())
            )
        cached = np.load(run_dir / entry["measurement_path"], allow_pickle=False)
        max_diffs = {
            "latent_mean_map": float(
                np.abs(
                    cached["latent_mean_map"] - online.latent_mean_map.detach().cpu().numpy()
                ).max()
            ),
            "depth_mean": float(
                np.abs(cached["depth_mean"] - online.depth_mean.detach().cpu().numpy()).max()
            ),
            "depth_variance": float(
                np.abs(
                    cached["depth_variance"] - online.depth_variance.detach().cpu().numpy()
                ).max()
            ),
        }
        results.append(
            {
                "measurement_id": entry["measurement_id"],
                "entries_match": bool(
                    json.loads(str(cached["measurement_ids_json"])) == [entry["measurement_id"]]
                ),
                "max_abs_diff": max_diffs,
            }
        )
    return {"samples": len(results), "results": results}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def train_main(
    config: N04FullConfig, run_dir: str | Path, *, resume: str | None, steps: int | None
) -> dict[str, Any]:
    trainer = Trainer(config, seed=config.training.seeds[0], run_dir=run_dir)
    return trainer.run(steps=int(steps or config.training.steps), resume=resume)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("profile", "train", "lr-grid", "export-measurements", "data-stats"),
    )
    parser.add_argument(
        "--run-dir",
        "--output",  # supervisor 模板使用 --output；两者等价
        dest="run_dir",
        required=True,
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--include-pyramid", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.seed is not None:
        config = _with_seed(config, args.seed)
    if args.mode == "profile":
        report = profile_main(config, args.run_dir, steps=args.steps)
    elif args.mode == "train":
        report = train_main(config, args.run_dir, resume=args.resume, steps=args.steps)
    elif args.mode == "lr-grid":
        report = lr_grid_main(config, args.run_dir, steps=args.steps)
    elif args.mode == "data-stats":
        report = data_stats_main(config, args.run_dir)
    else:
        if not args.checkpoint:
            raise SystemExit("--mode export-measurements 需要 --checkpoint")
        report = export_main(
            config,
            args.run_dir,
            args.checkpoint,
            splits=tuple(args.splits.split(",")),
            include_pyramid=args.include_pyramid,
            limit=args.limit,
        )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("history", "results")},
            indent=2,
            default=str,
        )
    )
    return 0


def _with_seed(config: N04FullConfig, seed: int) -> N04FullConfig:
    from dataclasses import replace

    return replace(config, training=replace(config.training, seeds=(int(seed),)))


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())

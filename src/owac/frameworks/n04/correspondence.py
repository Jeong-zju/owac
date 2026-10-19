"""N04 v1.1 S2 完整空间对应：作者完整 RAFT + RAW 自监督对应损失。

坐标契约（docs/n04_v11_s2_interfaces.md §2）
--------------------------------------------
* 全项目只有本模块定义像素坐标与 ``grid_sample`` 归一化坐标的换算：``px_to_grid`` /
  ``grid_to_px``；所有 ``F.grid_sample`` 一律 ``align_corners=False``，整数值即像素中心，
  ``x ∈ [0, W-1]``、``y ∈ [0, H-1]``。
* flow 语义与作者 RAFT 相同（previous→current）：``flow_px`` **定义在上一帧坐标上**，
  上一帧点 ``xy_prev`` 在当前帧的对应位置是 ``xy_prev + flow_px[xy_prev]``；契约 §2 的
  ``src_xy = dst_xy + flow_px`` 中 ``dst`` 指 flow 所在帧（上一帧），§4 的
  ``xy_t = xy_{t-1} + flow(xy_{t-1})`` 同义。单位原始像素。

RAW 尺度适配
------------
作者 ``RAFT.forward`` 假设输入是 RGB uint8 经 ``2 * (x / 255) - 1`` 归一化。本项目两条平台
输入已由 S0/S1 冻结为固定尺度（``contracts.INPUT_SCALES``）：LIBERO 是 RGB8 逆处理
pseudo-RAW（uint8 × 1/255），RoboTwin 是 HDR-derived float32 线性辐射（white level 1.0，×1）。
``RawScale`` 因此只做**平台固定**的仿射 ``normalized = (raw * factor - center) / half_range``，
默认 ``center=0.5, half_range=0.5``（测量单位 [0,1] → 作者期望的 [-1,1]；LIBERO uint8 路径与
作者 ``2*(x/255)-1`` 完全一致）。本模块不做逐帧 min-max、不把 float32 转 uint8、不裁剪高于
white level 的 HDR 值、不做 ``nan_to_num``；非有限 RAW 直接抛错暴露。

损失来源
--------
``RaftCorrespondenceLoss`` 按 SMURF（google-research/smurf, commit 3ef52111）与 UFlow 的自监督
对应做法迁移到 PyTorch。已实现项与未迁移项在类 docstring 中逐条列出，不声称复现论文结果。
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ._vendor.raft import RAFT
from .contracts import InputScale

__all__ = [
    "CorrespondenceLosses",
    "FlowOutput",
    "RaftCorrespondence",
    "RaftCorrespondenceLoss",
    "RaftLossConfig",
    "RawScale",
    "flow_at_points",
    "grid_to_px",
    "points_inside_frame",
    "px_to_grid",
    "sample_flow_at_grid",
    "bidirectional_correspondence",
    "self_supervision_loss",
    "source_visibility",
]


# --------------------------------------------------------------------------------------
# 坐标换算（契约 §2；全项目只在本题定义）
# --------------------------------------------------------------------------------------


def px_to_grid(xy_px: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """原始像素坐标 -> ``grid_sample`` 归一化坐标（``align_corners=False``）。

    ``xy_px`` 最后一维为 ``(x, y)``；``size_hw`` 为该像素空间的 ``(H, W)``。
    契约定义：``(xy + 0.5) * 2 / size_hw - 1``，因此整数值恰好落在像素中心。
    """
    height, width = int(size_hw[0]), int(size_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size_hw 必须为正，实际 {(height, width)}")
    size = torch.tensor([width, height], dtype=xy_px.dtype, device=xy_px.device)
    return (xy_px + 0.5) * 2.0 / size - 1.0


def grid_to_px(grid: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """``grid_sample`` 归一化坐标 -> 原始像素坐标（``px_to_grid`` 的逆）。"""
    height, width = int(size_hw[0]), int(size_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size_hw 必须为正，实际 {(height, width)}")
    size = torch.tensor([width, height], dtype=grid.dtype, device=grid.device)
    return (grid + 1.0) * size / 2.0 - 0.5


def points_inside_frame(xy_px: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """``[..., 2]`` 像素坐标是否落在帧内（闭区间 ``[0, size-1]``，与 SMURF ``mask_invalid`` 同义）。

    越界点仍可被采样（``flow_at_points`` 用边界像素值），是否可用由本掩码交给调用方决定。
    """
    height, width = int(size_hw[0]), int(size_hw[1])
    xy = xy_px
    return (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= width - 1)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= height - 1)
    )


def flow_at_points(flow_px: torch.Tensor, xy_px: torch.Tensor) -> torch.Tensor:
    """在 flow 所在帧（上一帧）的任意原始像素坐标处采样 flow：``[B,2,H,W]`` +
    ``[B,N,2]``（或 ``[N,2]``）-> ``[B,N,2]``。

    双线性、``align_corners=False``，越界点取边界像素值（``padding_mode="border"``）；
    有效性用 ``points_inside_frame`` 单独判断，本函数不返回掩码。
    """
    if flow_px.dim() != 4 or flow_px.shape[1] != 2:
        raise ValueError(f"flow_px 需要 [B,2,H,W]，实际 {tuple(flow_px.shape)}")
    batch = flow_px.shape[0]
    xy = xy_px
    if xy.dim() == 2:
        xy = xy.unsqueeze(0).expand(batch, -1, -1)
    if xy.dim() != 3 or xy.shape[-1] != 2:
        raise ValueError(f"xy_px 需要 [B,N,2] 或 [N,2]，实际 {tuple(xy_px.shape)}")
    if xy.shape[0] != batch:
        raise ValueError(f"xy_px batch {xy.shape[0]} 与 flow_px batch {batch} 不一致")
    grid = px_to_grid(xy.to(flow_px.dtype), flow_px.shape[-2:]).unsqueeze(1)  # [B,1,N,2]
    sampled = F.grid_sample(
        flow_px, grid, mode="bilinear", align_corners=False, padding_mode="border"
    )
    return sampled[:, :, 0, :].permute(0, 2, 1).contiguous()


def sample_flow_at_grid(flow_px: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """把像素单位的 flow 重采样到另一分辨率的规则网格（如 measurement 特征网格 ``h=H/stride``）。

    值保持原始像素单位：网格变粗/变细只改变场的采样密度，不改变位移长度。采样格点即
    ``size_hw`` 各像素中心（等价于对 ``[0,H-1]×[0,W-1]`` 做 ``align_corners=False`` 插值），
    与 ``px_to_grid`` 约定一致。特征网格的像素↔网格映射仍需 measurement 的 pad/size 元数据。
    """
    return F.interpolate(
        flow_px, size=(int(size_hw[0]), int(size_hw[1])), mode="bilinear", align_corners=False
    )


# --------------------------------------------------------------------------------------
# RAW 尺度
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawScale:
    """平台固定的 RAW 换算：存储值 -> 测量单位 -> 网络输入；不逐帧统计。

    ``factor`` 与 ``contracts.INPUT_SCALES`` 一致（LIBERO 1/255、RoboTwin 1.0）；
    ``center`` / ``half_range`` 是固定仿射，默认把测量单位 [0,1] 映射到作者期望的 [-1,1]。
    ``saturation_level`` 是测量单位的白电平（LIBERO 1.0，RoboTwin 配置 3.5），饱和判定为
    ``>= saturation_fraction * saturation_level``（与 ``measurement.preprocess_frames`` 同阈值）。
    噪声权重用简单的读出/散粒噪声模型 ``var = read_sigma**2 + shot_sigma * I``，
    权重 ``1 / (1 + var)``；它是可配置的 RAW 加权，不是已标定的传感器噪声模型。
    """

    name: str
    factor: float = 1.0
    center: float = 0.5
    half_range: float = 0.5
    saturation_level: float = 1.0
    saturation_fraction: float = 0.99
    read_sigma: float = 0.005
    shot_sigma: float = 0.01

    @classmethod
    def from_input_scale(cls, scale: InputScale, **overrides: float) -> RawScale:
        """用 ``contracts.INPUT_SCALES`` 的冻结尺度构造（``factor`` 直接沿用）。"""
        return cls(name=scale.name, factor=float(scale.factor), **overrides)

    def measurement_units(self, raw: torch.Tensor) -> torch.Tensor:
        """存储值 -> 本阶段统一测量单位（FP32，不做任何裁剪）。"""
        return raw.to(torch.float32) * self.factor

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """测量单位 -> 作者 RAFT 输入区间（固定仿射，无逐帧统计、无 clip）。"""
        return (self.measurement_units(raw) - self.center) / self.half_range

    def saturated(self, raw: torch.Tensor) -> torch.Tensor:
        """``[B,1,H,W]`` bool：达到白电平的像素（可能被裁剪，亮度不可信，结构仍可用）。"""
        level = self.saturation_fraction * self.saturation_level
        return (self.measurement_units(raw) >= level).any(dim=1, keepdim=True)

    def confidence(self, raw: torch.Tensor) -> torch.Tensor:
        """``[B,1,H,W]`` float：RAW 噪声权重 ``1/(1+var)``，随亮度单调下降。"""
        variance = self.read_sigma**2 + self.shot_sigma * self.measurement_units(raw).clamp_min(0.0)
        return (1.0 / (1.0 + variance)).mean(dim=1, keepdim=True)


# --------------------------------------------------------------------------------------
# 前向：完整 RAFT + RAW 适配
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowOutput:
    """一帧对的对应输出。

    ``flow_px`` 是定义在上一帧坐标上的 previous→current 位移场（原始像素单位）；``valid``
    表示"上一帧点 + flow"得到的当前帧采样点落在帧内（且调用方给的 ``valid`` 成立），
    ``feature_scale`` 是 RAFT 计算网格相对
    原始像素的尺度：1.0 表示在原生分辨率上计算，s>1 表示在 1/s 分辨率上估计后放大 s 倍
    （对应训练侧传入的固定 resize）。measurement 特征网格换算不读本字段，用其 pad/size 元数据。
    """

    flow_px: torch.Tensor
    valid: torch.Tensor
    feature_scale: float
    backward_flow_px: torch.Tensor | None = None
    visibility: torch.Tensor | None = None  # [B,H,W], defined in previous/source coordinates


class RaftCorrespondence(nn.Module):
    """作者完整 RAFT（``small=False``，默认 12 次迭代）包装成 RAW 帧对 -> flow。

    * 输入 ``[B,C,H,W]`` 的 RAW（C 由平台决定，默认 3）按 ``RawScale`` 做固定仿射；
    * 内部 pad 到 8 的倍数并保证每个相关金字塔层级 ≥1 网格，避免作者
      ``bilinear_sampler`` 对 ``W-1`` 除零（见 ``_vendor/raft/provenance.json``）；
    * ``resize_hw`` 给训练侧在显存不足时固定降分辨率：flow 在上采样回原生分辨率时按
      每个轴的像素比缩放（``ratio = native/compute``），``feature_scale`` 记录该比例；
      等比重采样（保留纵横比）才接受；
    * 非有限 RAW 或非有限 flow 一律抛错，不用数值保护掩盖。
    """

    def __init__(
        self,
        *,
        scale: RawScale,
        in_channels: int = 3,
        iters: int = 12,
        resize_hw: tuple[int, int] | None = None,
        mixed_precision: bool = False,
        alternate_corr: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(iters) < 1:
            raise ValueError(f"iters 必须为正，实际 {iters}")
        if int(in_channels) < 1:
            raise ValueError(f"in_channels 必须为正，实际 {in_channels}")
        self.scale = scale
        self.in_channels = int(in_channels)
        self.iters = int(iters)
        self.resize_hw = None if resize_hw is None else (int(resize_hw[0]), int(resize_hw[1]))
        if self.resize_hw is not None and min(self.resize_hw) < 1:
            raise ValueError(f"resize_hw 必须为正，实际 {resize_hw}")
        args = Namespace(
            small=False,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
            dropout=dropout,
        )
        self.raft = RAFT(args, in_channels=self.in_channels)
        self.corr_levels = int(args.corr_levels)
        self.corr_radius = int(args.corr_radius)

    # -- 内部：尺寸与运行 ------------------------------------------------------------

    def _compute_hw(self, native_hw: tuple[int, int]) -> tuple[int, int]:
        return native_hw if self.resize_hw is None else self.resize_hw

    def _padded_hw(self, size_hw: tuple[int, int]) -> tuple[int, int]:
        """pad 到 8 的倍数，且每轴至少 ``8 * 2**(corr_levels-1)``（最粗金字塔层级 ≥1）。"""
        min_side = 8 * 2 ** (self.corr_levels - 1)
        out = []
        for side in size_hw:
            value = max(int(side), min_side)
            out.append(value + (-value) % 8)
        return out[0], out[1]

    def _flow_at_compute_size(self, image1: torch.Tensor, image2: torch.Tensor, iters: int):
        height, width = image1.shape[-2:]
        pad_h, pad_w = self._padded_hw((height, width))
        if (pad_h, pad_w) != (height, width):
            pad = (0, pad_w - width, 0, pad_h - height)
            image1 = F.pad(image1, pad, mode="replicate")
            image2 = F.pad(image2, pad, mode="replicate")
        predictions = self.raft(image1, image2, iters=iters, normalized=True)
        flow = predictions[-1][..., :height, :width]
        if not torch.isfinite(flow).all():
            raise FloatingPointError(
                "RAFT 输出含 NaN/Inf：检查 RAW 是否近似常数（InstanceNorm 方差退化）或数值范围"
            )
        return flow

    # -- 公开接口 --------------------------------------------------------------------

    def forward(
        self,
        previous_raw: torch.Tensor,
        current_raw: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        iters: int | None = None,
        resize_hw: tuple[int, int] | None = None,
    ) -> FlowOutput:
        """``[B,C,H,W]`` RAW 帧对 -> `FlowOutput`（flow 在原始分辨率，previous→current）。

        ``valid [B,H,W]``（可选）标记两帧都可信的像素（坏点等）；非有限 RAW 直接抛错。
        ``iters`` / ``resize_hw`` 可覆盖构造时的默认值。
        """
        if previous_raw.shape != current_raw.shape:
            raise ValueError(
                f"两帧形状必须一致，实际 {tuple(previous_raw.shape)} vs {tuple(current_raw.shape)}"
            )
        if previous_raw.dim() != 4:
            raise ValueError(f"RAW 需要 [B,C,H,W]，实际 {tuple(previous_raw.shape)}")
        if previous_raw.shape[1] != self.in_channels:
            raise ValueError(
                f"该对应网络为 {self.in_channels} 通道构建，收到 {previous_raw.shape[1]} 通道"
            )
        native_hw = (int(previous_raw.shape[-2]), int(previous_raw.shape[-1]))
        if valid is not None and tuple(valid.shape) != (previous_raw.shape[0], *native_hw):
            raise ValueError(
                f"valid 需要 [B,H,W] = {(previous_raw.shape[0], *native_hw)}，"
                f"实际 {tuple(valid.shape)}"
            )
        for name, raw in (("previous_raw", previous_raw), ("current_raw", current_raw)):
            if not torch.isfinite(raw).all():
                raise ValueError(
                    f"{name} 含 NaN/Inf：对应损失不得用 nan_to_num 掩盖；"
                    "按 measurement.preprocess_frames 的约定在数据层先清理并标记"
                )

        target_hw = self.resize_hw if resize_hw is None else (int(resize_hw[0]), int(resize_hw[1]))
        compute_hw = native_hw if target_hw is None else target_hw
        if compute_hw != native_hw:
            ratio_h = native_hw[0] / compute_hw[0]
            ratio_w = native_hw[1] / compute_hw[1]
            if abs(ratio_h - ratio_w) > 1e-6:
                raise ValueError(
                    f"resize 必须是等比：native {native_hw} -> compute {compute_hw} "
                    f"得到 {(ratio_h, ratio_w)}"
                )
        feature_scale = native_hw[0] / compute_hw[0]

        image1 = self.scale.normalize(previous_raw)
        image2 = self.scale.normalize(current_raw)
        if compute_hw != native_hw:
            image1 = F.interpolate(image1, size=compute_hw, mode="bilinear", align_corners=False)
            image2 = F.interpolate(image2, size=compute_hw, mode="bilinear", align_corners=False)
        flow = self._flow_at_compute_size(
            image1, image2, self.iters if iters is None else int(iters)
        )
        if compute_hw != native_hw:
            flow = sample_flow_at_grid(flow, native_hw)
            factor = torch.tensor(
                [native_hw[1] / compute_hw[1], native_hw[0] / compute_hw[0]],
                dtype=flow.dtype,
                device=flow.device,
            ).view(1, 2, 1, 1)
            flow = flow * factor

        height, width = native_hw
        yy, xx = torch.meshgrid(
            torch.arange(height, device=flow.device, dtype=flow.dtype),
            torch.arange(width, device=flow.device, dtype=flow.dtype),
            indexing="ij",
        )
        # 契约 §2：上一帧格点 + flow 得到当前帧的采样坐标
        sampled_xy = torch.stack([xx, yy], dim=0)[None] + flow
        inside = points_inside_frame(sampled_xy.permute(0, 2, 3, 1), native_hw)
        if valid is not None:
            inside = inside & valid.to(torch.bool)
        return FlowOutput(flow_px=flow, valid=inside, feature_scale=feature_scale)


# --------------------------------------------------------------------------------------
# 自监督对应损失（SMURF / UFlow 迁移）
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RaftLossConfig:
    """损失项权重与非迁移无关的常数。

    默认值对齐 SMURF flags：``weight_census=1.0``、``weight_smooth1=0.0``、
    ``weight_smooth2=2.0``、``smoothness_edge_weighting='exponential'``、
    ``smoothness_edge_constant=150``、``smoothness_at_level=2``、``occlusion_estimation='wang'``、
    ``fb_sigma_student=0.03``。``photometric``（RAW 亮度 robust 项）是本项目为 RAW 域新增，
    SMURF 默认配置只开 census。``fb_support`` 默认 0：SMURF 只在 teacher/student 自监督分支
    使用该 gaussian 一致性掩码，这里提供同一形式但默认关闭。

    2026-10-16 塌缩修复（固定集诊断后授权的最小改动）
    -------------------------------------------------
    * ``static_baseline=True``：matching 项（census / photometric）除以**同一 pair 同一支持掩码
      下的零-flow（静态）残差**，即只保留"相对不动的增益"。依据：固定集上 step100 的 flow 塌成
      常数大位移（LIBERO fy≈25.6px、RoboTwin fy≈27–34px），输运残差是同位置残差的 9.8–10.3 倍，
      而 census soft-hamming 与 robust photometric 在大残差处梯度趋零、``smooth2`` 对常数场为 0
      → "整体平移"曾是免费解。归一化后零 flow 的比值恰为 1，任何无用位移都 >1，越动越亏；同时
      该项天然按 pair 尺度归一，缓解亮域（Normal/Over/ExtremeOver）损失绝对值大而主导均值的问题。
      选它而不是 ``mean|flow|`` 正则：L1 幅度罚会连合法的大运动一起罚，且需要无依据的新超参。
    * ``smoothness_delay_steps`` / ``smoothness_delay_pairs``：平滑项延后生效。注意 SMURF flag
      默认其实是 step0 就开（``smoothness_after_num_steps=-1``，run.sh 未覆盖），所以这不是照搬
      SMURF，而是本项目依据上述塌缩证据做的保守选择：在项目 warmup（~2k 步）内不给"常数场"这个
      唯一免费下降方向。``forward`` 用内部 pair 调用计数（``smoothness_delay_pairs``）；
      trainer 调 ``set_step(step)`` 后改按优化步（``smoothness_delay_steps``）计，避免 resume 后
      计数清零。门槛语义与 SMURF 一致（``index > delay`` 才启用，硬开关不做 ramp）。
    """

    census: float = 1.0
    photometric: float = 1.0
    smooth1: float = 0.0
    smooth2: float = 2.0
    fb_support: float = 0.0
    occlusion: str = "wang"
    edge_weighting: str = "exponential"
    edge_constant: float = 150.0
    smoothness_level: int = 2
    census_patch: int = 7
    census_threshold: float = 0.1
    robust_eps: float = 0.01
    robust_q: float = 0.4
    fb_sigma: float = 0.03
    #: matching 项除以同 pair 的零-flow 残差（见类 docstring 的塌缩修复说明）。
    static_baseline: bool = True
    #: 未接入 ``set_step`` 时的 fallback：前 N 次 pair 调用不启用平滑项。
    smoothness_delay_pairs: int = 1_000_000
    #: 调过 ``set_step(step)`` 后按优化步计：前 N 步不启用平滑项（对齐项目 warmup ~2k）。
    smoothness_delay_steps: int = 2_000
    selfsup: float = 0.1
    selfsup_warmup_steps: int = 2_000
    # A geometric training constraint, not evidence of a valid match. Unlike
    # photometric masks it retains a gradient when all endpoints leave the image.
    boundary: float = 0.0


@dataclass(frozen=True)
class CorrespondenceLosses:
    """逐项损失（都已按有效样本均值归一化）与可见支持比例。"""

    total: torch.Tensor
    census: torch.Tensor
    photometric: torch.Tensor
    smooth1: torch.Tensor
    smooth2: torch.Tensor
    support: float
    boundary: torch.Tensor | None = None
    source_support: float = 0.0


class RaftCorrespondenceLoss(nn.Module):
    """RAW 帧对的 SMURF / UFlow 自监督对应损失（PyTorch 迁移）。

    已实现（逐项对应 ``smurf/smurf_utils.py`` @ 3ef52111）
    ------------------------------------------------------
    * ``census``：``census_loss`` = ``census_transform``（7×7 邻域差经
      ``d/sqrt(0.81+d²)``）+ ``soft_hamming`` + ``zero_mask_border`` + DDFlow
      ``abs_robust_loss``，按有效支持加权求和（``stop_gradient`` 掩码）。RAW 版本把作者的
      ``rgb_to_grayscale(image) * 255`` 换成固定通道权重的亮度平面：测量单位已经等价于作者的
      ``x/255`` 尺度（且 RoboTwin 是 float32 HDR，不能也不该乘 255）。
    * ``photometric``：同族 robust 亮度重投影（``abs_robust_loss`` 作用于
      ``|ref - warp(other)|``），是本项目在 RAW 亮度域的补充项（SMURF 默认只开 census）。
    * ``wang`` 遮挡：UFlow rangemap —— 用后向 flow 对参考帧坐标做双线性 splat 计数，
      ``1 - clip(range_map)``；等价 ``compute_occlusions_wang(threshold=False)``（factor=1）。
    * 边界有效掩码：``mask_invalid``（来源坐标 ``>=0`` 且 ``<= size-1``）。
    * ``fb_support``（默认 0）：``self_supervision_loss`` 的 gaussian 前向后向一致性
      ``exp(-|f + f_rev(warp)|² / (σ²(H²+W²)))``，这里作为重投影支持权重。
    * ``smooth1`` / ``smooth2``：``first_order_smoothness_loss`` /
      ``second_order_smoothness_loss``，edge weighting 支持 gaussian / exponential，
      ``robust_l1`` 作用于 flow 前向差分。
    * RAW 噪声/饱和权重（项目新增）：``RawScale.confidence`` 的读出+散粒噪声权重进入支持；
      饱和像素不计入亮度项（被裁剪的亮度不可信），但保留在 census 结构项里。
    * 静态基线归一化（项目新增，见 `RaftLossConfig` 的塌缩修复说明）：matching 项除以同 pair
      同掩码的零-flow 残差，零 flow 时比值恰为 1，无用位移 >1。
    * 平滑项延后（项目新增）：``smoothness_delay_steps`` / ``smoothness_delay_pairs`` 之前
      ``smooth1``/``smooth2`` 权重为 0；``set_step`` 接入后按优化步计。

    迁移范围 / 删改（诚实清单）
    ------------------------
    * SMURF Gaussian self-supervision：同网原图双向预测作 stop-gradient teacher，
      中心裁剪分支为 student，保留原像素位移单位并按 optimizer step ramp-up。
      未复现完整 SMURF 的随机光度增强、EMA 或整套训练配方。
    * ``compute_occlusions_brox``、``full_size_images`` 的 crop/pad 全图 warp 路径、
      rangemap 的 ``downsampling_factor > 1`` 与 ``reduce_downsampling_bias`` 未实现。
    * ``unsupervised_sequence_loss``（多帧序列一致性）未迁移；本模块只处理相邻帧对。
    * trainer 每次 optimizer update 调 set_step；平滑延迟与 selfsup ramp-up 共用该步数，
      不按相机／帧对次数加速，也不会因 resume 清零。
    * smoothness 的 flow 取自最终全分辨率预测再按 ``smoothness_level`` 下采样
      （SMURF 直接取网络多分辨率金字塔对应层），是等比近似。
    """

    def __init__(
        self,
        *,
        scale: RawScale,
        config: RaftLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.config = RaftLossConfig() if config is None else config
        if self.config.occlusion not in ("wang", "none"):
            raise ValueError(f"未知遮挡估计 {self.config.occlusion!r}；允许 'wang' / 'none'")
        if self.config.edge_weighting not in ("gaussian", "exponential"):
            raise ValueError(f"未知边缘权重 {self.config.edge_weighting!r}")
        if self.config.census_patch < 1 or self.config.census_patch % 2 != 1:
            raise ValueError(f"census_patch 必须为正奇数，实际 {self.config.census_patch}")
        if self.config.smoothness_level < 0:
            raise ValueError("smoothness_level 不能为负")
        if self.config.smoothness_delay_pairs < 0 or self.config.smoothness_delay_steps < 0:
            raise ValueError("平滑项延后步数/次数不能为负")
        #: pair 调用计数（未接入 ``set_step`` 时的 fallback；见 ``_smoothness_enabled``）。
        self._pairs_seen = 0
        self._step: int | None = None

    def set_step(self, step: int) -> None:
        """接入优化步：之后平滑项门槛按 ``smoothness_delay_steps`` 判定（resume 安全）。

        不接也不会报错——``forward`` 退回内部 pair 计数（``smoothness_delay_pairs``），
        但 resume 会让计数清零，精确排程应由 trainer 每步调用本方法。
        """
        if int(step) < 0:
            raise ValueError(f"step 不能为负，实际 {step}")
        self._step = int(step)

    def _smoothness_enabled(self) -> bool:
        """平滑项门槛（SMURF ``current_step > smoothness_after_num_steps`` 同义）。"""
        if self._step is not None:
            return self._step > self.config.smoothness_delay_steps
        return self._pairs_seen > self.config.smoothness_delay_pairs

    def self_supervised(
        self,
        model: nn.Module,
        previous: torch.Tensor,
        current: torch.Tensor,
        forward: torch.Tensor,
        backward: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SMURF-style cropped student, same RAW network's original flow as stop-grad teacher.

        Fixed central crop removes context without inventing RGB or external flow labels.
        Teacher reliability is measured before the crop; student unreliability after it.
        This ports the Gaussian self-supervision term, not SMURF's complete training recipe.
        """
        height, width = previous.shape[-2:]
        # Equal 1/8 margins preserve the actual platforms' aspect ratios (256² / 240x320).
        margin_h, margin_w = height // 8, width // 8
        if min(margin_h, margin_w) < 1:
            raise ValueError("self-supervision needs spatial dimensions >= 8")
        roi = (..., slice(margin_h, height - margin_h), slice(margin_w, width - margin_w))

        # Keep the configured compute grid if its aspect ratio matches the crop.
        def predict(a, b):
            if isinstance(model, RaftCorrespondence):
                configured = model.resize_hw
                h, w = a.shape[-2:]
                size = (
                    configured if configured and configured[0] * w == configured[1] * h else (h, w)
                )
                return model(a, b, resize_hw=size).flow_px
            return model(a, b).flow_px

        student = predict(previous[roi], current[roi])
        with torch.no_grad():  # This direction is used exclusively by a stopped mask.
            student_backward = predict(current[roi], previous[roi])
        value = self_supervision_loss(
            forward,
            backward,
            student,
            student_backward,
            crop=(margin_h, margin_w, height - margin_h, width - margin_w),
            sigma=self.config.fb_sigma,
            valid=valid,
        )
        ramp = min(1.0, ((self._step or 0) + 1) / max(1, self.config.selfsup_warmup_steps))
        return value * self.config.selfsup * ramp

    # -- 公开接口 --------------------------------------------------------------------

    def forward(
        self,
        previous_raw: torch.Tensor,
        current_raw: torch.Tensor,
        flow_px: torch.Tensor,
        *,
        backward_flow_px: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> CorrespondenceLosses:
        """``previous_raw``/``current_raw`` ``[B,C,H,W]`` + previous→current ``flow_px``。

        ``backward_flow_px``（current→previous）给出时：两个方向的重投影项都计算并平均
        （对应 SMURF ``only_forward=False``），同时启用 rangemap 遮挡与 FB 一致性；
        缺省时只算 previous 参考方向，遮挡项退化为无（记录在支持的 ``support`` 里）。
        ``valid [B,H,W]`` 是两帧共用的可信像素掩码（坏点等），参与全部支持计算。
        """
        self._pairs_seen += 1
        shape = tuple(previous_raw.shape)
        if tuple(current_raw.shape) != shape:
            raise ValueError(f"两帧形状必须一致，实际 {shape} vs {tuple(current_raw.shape)}")
        if (
            flow_px.dim() != 4
            or flow_px.shape[0] != shape[0]
            or tuple(flow_px.shape[-2:]) != shape[-2:]
        ):
            raise ValueError(f"flow_px 需要 [B,2,H,W] 匹配 {shape}，实际 {tuple(flow_px.shape)}")
        if flow_px.shape[1] != 2 or not bool(torch.isfinite(flow_px).all()):
            raise ValueError("flow_px must have two finite channels")
        if backward_flow_px is not None and tuple(backward_flow_px.shape) != tuple(flow_px.shape):
            raise ValueError(
                f"backward_flow_px 形状 {tuple(backward_flow_px.shape)} 与 flow_px 不一致"
            )
        if backward_flow_px is not None and not bool(torch.isfinite(backward_flow_px).all()):
            raise ValueError("backward flow contains NaN/Inf")
        height, width = shape[-2:]
        if valid is not None and tuple(valid.shape) != (shape[0], height, width):
            raise ValueError(f"valid 需要 [B,H,W]，实际 {tuple(valid.shape)}")
        for name, raw in (("previous_raw", previous_raw), ("current_raw", current_raw)):
            if not torch.isfinite(raw).all():
                raise ValueError(f"{name} 含 NaN/Inf：不得用 nan_to_num 掩盖")

        previous = self.scale.measurement_units(previous_raw)
        current = self.scale.measurement_units(current_raw)
        user_valid = None if valid is None else valid.to(torch.bool)

        directions = [
            _LossDirection(
                reference=previous,
                other=current,
                flow=flow_px,
                reverse=backward_flow_px,
                reference_conf=self.scale.confidence(previous_raw),
                other_conf=self.scale.confidence(current_raw),
                reference_sat=self.scale.saturated(previous_raw),
                other_sat=self.scale.saturated(current_raw),
                user_valid=user_valid,
            )
        ]
        if backward_flow_px is not None:
            directions.append(
                _LossDirection(
                    reference=current,
                    other=previous,
                    flow=backward_flow_px,
                    reverse=flow_px,
                    reference_conf=self.scale.confidence(current_raw),
                    other_conf=self.scale.confidence(previous_raw),
                    reference_sat=self.scale.saturated(current_raw),
                    other_sat=self.scale.saturated(previous_raw),
                    user_valid=user_valid,
                )
            )

        census = previous.new_zeros(())
        photometric = previous.new_zeros(())
        smooth1 = previous.new_zeros(())
        smooth2 = previous.new_zeros(())
        support = 0.0
        source_support = 0.0
        boundary_loss = previous.new_zeros(())
        config = self.config
        smooth_enabled = self._smoothness_enabled()
        smooth_weights = (
            config.smooth1 if smooth_enabled else 0.0,
            config.smooth2 if smooth_enabled else 0.0,
        )
        for direction in directions:
            parts = self._direction_losses(direction, smooth_weights)
            census = census + parts.census
            photometric = photometric + parts.photometric
            smooth1 = smooth1 + parts.smooth1
            smooth2 = smooth2 + parts.smooth2
            support += parts.support
            source_mask = torch.ones_like(direction.reference_conf)
            if direction.user_valid is not None:
                source_mask = source_mask * direction.user_valid[:, None]
            # Distances are in fractions of the real sensor dimensions; the
            # original RAW, returned flow and its validity are never clamped.
            coords = _flow_to_warp(direction.flow)
            # Our public and internal flow coordinates are (x,y).
            limits = direction.flow.new_tensor([width - 1, height - 1]).view(1, 2, 1, 1)
            normalizer = direction.flow.new_tensor([width, height]).view(1, 2, 1, 1)
            overflow = (F.relu(-coords) + F.relu(coords - limits)) / normalizer
            boundary_loss = boundary_loss + (
                overflow.square().sum(1, keepdim=True) * source_mask
            ).sum() / source_mask.sum().clamp_min(1)
            source_support += float(source_mask.float().mean())
        count = float(len(directions))
        census, photometric, smooth1, smooth2 = (
            census / count,
            photometric / count,
            smooth1 / count,
            smooth2 / count,
        )
        total = (
            config.census * census
            + config.photometric * photometric
            + smooth_weights[0] * smooth1
            + smooth_weights[1] * smooth2
            + config.boundary * boundary_loss / count
        )
        return CorrespondenceLosses(
            total=total,
            census=census,
            photometric=photometric,
            smooth1=smooth1,
            smooth2=smooth2,
            support=support / count,
            boundary=boundary_loss / count,
            source_support=source_support / count,
        )

    # -- 内部：单方向项 --------------------------------------------------------------

    def _direction_losses(
        self, direction: _LossDirection, smooth_weights: tuple[float, float]
    ) -> _DirectionLosses:
        config = self.config
        values = direction.reference
        height, width = values.shape[-2:]
        coords = _flow_to_warp(direction.flow)
        boundary = _mask_invalid(coords).squeeze(1).unsqueeze(1)  # [B,1,H,W] float

        support = boundary * direction.reference_conf
        support = support * _resample(direction.other_conf, coords)
        if direction.user_valid is not None:
            support = support * direction.user_valid.to(support.dtype).unsqueeze(1)
            support = support * _resample(
                direction.user_valid.to(support.dtype).unsqueeze(1), coords
            )
        if config.occlusion == "wang" and direction.reverse is not None:
            # UFlow rangemap：后向 flow 在参考帧坐标上的可见计数，clamp 后即可见权重
            # （SMURF compute_occlusions(occlusions_are_zeros=True) 返回的就是 1-遮挡）。
            visibility = _range_map(direction.reverse).clamp(0.0, 1.0)
            support = support * visibility
        if config.fb_support > 0.0 and direction.reverse is not None:
            consistency = _fb_consistency(direction.flow, direction.reverse, config.fb_sigma)
            support = support * (1.0 - config.fb_support + config.fb_support * consistency)
        support = support.detach()

        warped = _resample(direction.other, coords)
        sat_ref = direction.reference_sat.to(support.dtype)
        sat_warped = _resample(direction.other_sat.to(support.dtype), coords)
        census_support = support
        photometric_support = (support * (1.0 - sat_ref) * (1.0 - sat_warped)).detach()

        luminance_ref = _luminance(direction.reference)
        luminance_warped = _luminance(warped)
        census = _census_loss(luminance_ref, luminance_warped, census_support, config)
        photometric = _photometric_loss(
            luminance_ref, luminance_warped, photometric_support, config
        )
        if config.static_baseline:
            # 零-flow（同位置）基线：同一支持掩码下的未 warp 残差，分母 stop-gradient。
            # 零 flow 时比值恰为 1；任何无用位移 >1（`abs_robust_loss` 的下界保证分母 ≥ 0.01**q）。
            luminance_static = _luminance(direction.other)
            census_static = _census_loss(luminance_ref, luminance_static, census_support, config)
            photometric_static = _photometric_loss(
                luminance_ref, luminance_static, photometric_support, config
            )
            census = census / census_static.detach().clamp_min(1e-6)
            photometric = photometric / photometric_static.detach().clamp_min(1e-6)

        smooth1_weight, smooth2_weight = smooth_weights
        if smooth1_weight > 0.0 or smooth2_weight > 0.0:
            image_level, flow_level = _smoothness_levels(
                _luminance(values), direction.flow, config.smoothness_level
            )
            edge_fn = _edge_weighting_fn(config)
            if smooth1_weight > 0.0:
                smooth1 = _first_order_smoothness(image_level, flow_level, edge_fn)
            else:
                smooth1 = values.new_zeros(())
            if smooth2_weight > 0.0:
                smooth2 = _second_order_smoothness(image_level, flow_level, edge_fn)
            else:
                smooth2 = values.new_zeros(())
        else:
            smooth1 = values.new_zeros(())
            smooth2 = values.new_zeros(())
        return _DirectionLosses(
            census=census,
            photometric=photometric,
            smooth1=smooth1,
            smooth2=smooth2,
            support=float(support.mean()),
        )


@dataclass(frozen=True)
class _LossDirection:
    """一个参考方向的重投影所需张量。"""

    reference: torch.Tensor
    other: torch.Tensor
    flow: torch.Tensor
    reverse: torch.Tensor | None
    reference_conf: torch.Tensor
    other_conf: torch.Tensor
    reference_sat: torch.Tensor
    other_sat: torch.Tensor
    user_valid: torch.Tensor | None


@dataclass(frozen=True)
class _DirectionLosses:
    census: torch.Tensor
    photometric: torch.Tensor
    smooth1: torch.Tensor
    smooth2: torch.Tensor
    support: float


# --------------------------------------------------------------------------------------
# 损失内部工具（SMURF / UFlow 迁移）
# --------------------------------------------------------------------------------------


def _luminance_weights(channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """固定亮度通道权重：3 通道用 SMURF/作者灰度权重，单通道直接用，其它等权。"""
    if channels == 3:
        weights = (0.299, 0.587, 0.114)
    elif channels == 1:
        weights = (1.0,)
    else:
        weights = tuple(1.0 / channels for _ in range(channels))
    return torch.tensor(weights, dtype=dtype, device=device).view(1, channels, 1, 1)


def _luminance(values: torch.Tensor) -> torch.Tensor:
    """``[B,C,H,W]`` 测量单位 -> ``[B,1,H,W]`` 亮度平面（不乘 255，见类 docstring）。"""
    weights = _luminance_weights(values.shape[1], values.dtype, values.device)
    return (values * weights).sum(dim=1, keepdim=True)


def _flow_to_warp(flow_px: torch.Tensor) -> torch.Tensor:
    """``[B,2,H,W]`` flow -> 在当前帧上采样的像素坐标 ``[B,2,H,W]``（上一帧格点 + flow）。"""
    batch, _, height, width = flow_px.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow_px.device, dtype=flow_px.dtype),
        torch.arange(width, device=flow_px.device, dtype=flow_px.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0)[None].expand(batch, 2, height, width)
    return base + flow_px


def _mask_invalid(coords_px: torch.Tensor) -> torch.Tensor:
    """SMURF ``mask_invalid``：来源坐标在帧内为 1，越界为 0。``[B,2,H,W]`` -> ``[B,1,H,W]``。"""
    batch, _, height, width = coords_px.shape
    inside = points_inside_frame(coords_px.permute(0, 2, 3, 1), (height, width))
    return inside.to(coords_px.dtype).unsqueeze(1)


def _resample(values: torch.Tensor, coords_px: torch.Tensor) -> torch.Tensor:
    """在来源像素坐标处双线性采样；越界按零背景（与 SMURF ``resample`` 同义）。"""
    height, width = values.shape[-2:]
    grid = px_to_grid(coords_px.permute(0, 2, 3, 1), (height, width))
    return F.grid_sample(values, grid, mode="bilinear", align_corners=False, padding_mode="zeros")


def _range_map(flow_px: torch.Tensor) -> torch.Tensor:
    """UFlow rangemap：``flow`` 定义的帧上每个坐标被采样的次数（双线性 splat）。

    ``[B,2,H,W]`` -> ``[B,1,H,W]``；零 flow 输入得到全 1。仅实现
    ``downsampling_factor=1``（SMURF 调用 rangemap 遮挡时的取值）。
    """
    batch, _, height, width = flow_px.shape
    coords = _flow_to_warp(flow_px)
    coords_floor = torch.floor(coords)
    offset = coords - coords_floor
    index_floor = coords_floor.to(torch.int64)
    batch_offset = torch.arange(batch, device=flow_px.device).view(batch, 1, 1) * height * width
    counts = flow_px.new_zeros(batch * height * width)
    for di in (0, 1):
        for dj in (0, 1):
            idx_i = index_floor[:, 1] + di
            idx_j = index_floor[:, 0] + dj
            weight_i = (1.0 - di) - ((-1.0) ** di) * offset[:, 1]
            weight_j = (1.0 - dj) - ((-1.0) ** dj) * offset[:, 0]
            weights = weight_i * weight_j
            inside = (idx_i >= 0) & (idx_i < height) & (idx_j >= 0) & (idx_j < width)
            flat = batch_offset + idx_i * width + idx_j
            counts = counts.index_add(0, flat[inside], weights[inside])
    return counts.reshape(batch, 1, height, width)


def _fb_consistency(flow_px: torch.Tensor, reverse_px: torch.Tensor, sigma: float) -> torch.Tensor:
    """SMURF ``self_supervision_loss`` 的 gaussian 前后向一致性 ``[B,1,H,W] ∈ (0,1]``。"""
    coords = _flow_to_warp(flow_px)
    reverse_at_source = _resample(reverse_px, coords)
    height, width = flow_px.shape[-2:]
    squared = (flow_px + reverse_at_source).pow(2).sum(dim=1, keepdim=True)
    return torch.exp(-squared / (sigma**2 * (height**2 + width**2)))


def source_visibility(forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
    """Wang range-map visibility in forward/source coordinates, including endpoints."""
    with torch.no_grad():
        return (
            _range_map(backward.detach()).clamp(0, 1)
            * _mask_invalid(_flow_to_warp(forward.detach()))
        ).squeeze(1)


def bidirectional_correspondence(model: nn.Module, previous, current) -> FlowOutput:
    """Two directions share one RAW network; neither direction is an external teacher."""
    forward = model(previous, current)
    backward = forward.backward_flow_px
    if backward is None:
        # Matching is symmetric: the reverse prediction also carries live gradients.
        backward = model(current, previous).flow_px
    visibility = source_visibility(forward.flow_px, backward)
    if forward.valid is not None:
        visibility = visibility * forward.valid.to(visibility)
    return FlowOutput(
        flow_px=forward.flow_px,
        valid=visibility > 0.5,
        feature_scale=forward.feature_scale,
        backward_flow_px=backward,
        visibility=visibility,
    )


def self_supervision_loss(
    teacher, teacher_backward, student, student_backward, *, crop, sigma=0.03, valid=None
):
    """Port of SMURF Gaussian self-supervision: transformed teacher mask times student failure.

    The crop translates both image origins equally, so flow vectors keep their pixel units.
    Both target and masks are detached; only the cropped student receives gradients.
    """
    top, left, bottom, right = crop
    if teacher.shape != teacher_backward.shape or student.shape != student_backward.shape:
        raise ValueError("both flow directions must have matching shapes")
    if teacher.ndim != 4 or teacher.shape[1] != 2 or student.shape[:2] != teacher.shape[:2]:
        raise ValueError("flows must be [B,2,H,W]")
    if not (
        sigma > 0
        and 0 <= top < bottom <= teacher.shape[-2]
        and 0 <= left < right <= teacher.shape[-1]
    ):
        raise ValueError("invalid crop or consistency sigma")
    if tuple(student.shape[-2:]) != (bottom - top, right - left):
        raise ValueError("student and crop size mismatch")
    with torch.no_grad():
        reliable = _fb_consistency(teacher.detach(), teacher_backward.detach(), sigma)
        coordinates = _flow_to_warp(teacher.detach())
        reliable = reliable * _mask_invalid(coordinates)
        if valid is not None:
            if valid.shape != teacher.shape[:1] + teacher.shape[-2:]:
                raise ValueError("valid mask must be [B,H,W]")
            supported = valid[:, None].to(reliable)
            reliable = reliable * supported * _resample(supported, coordinates)
        reliable = reliable[..., top:bottom, left:right]
        unreliable = 1 - (
            _fb_consistency(student.detach(), student_backward.detach(), sigma)
            * _mask_invalid(_flow_to_warp(student.detach()))
        )
        mask = reliable * unreliable
        target = teacher.detach()[..., top:bottom, left:right]
    return (mask * robust_l1(target - student)).mean()


def robust_l1(values: torch.Tensor) -> torch.Tensor:
    """SMURF ``robust_l1``：``sqrt(x² + 0.001²)``。"""
    return (values**2 + 0.001**2).sqrt()


def abs_robust_loss(values: torch.Tensor, eps: float, q: float) -> torch.Tensor:
    """DDFlow 的 robust loss（SMURF ``abs_robust_loss``）：``(|x| + eps)**q``。"""
    return (values.abs() + eps).pow(q)


def _zero_mask_border(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """SMURF ``zero_mask_border``：把掩码边界 ``patch//2`` 圈清零（忽略 census 边界效应）。"""
    pad = patch_size // 2
    if pad == 0:
        return mask
    if min(mask.shape[-2:]) <= 2 * pad:
        return torch.zeros_like(mask)
    inner = mask[..., pad:-pad, pad:-pad]
    return F.pad(inner, (pad, pad, pad, pad))


def _census_transform(luminance: torch.Tensor, patch_size: int) -> torch.Tensor:
    """SMURF ``census_transform``：邻域差 ``d / sqrt(0.81 + d²)``，返回 ``[B,K,H,W]``。"""
    kernel = torch.eye(patch_size * patch_size, dtype=luminance.dtype, device=luminance.device)
    kernel = kernel.reshape(patch_size * patch_size, 1, patch_size, patch_size)
    neighbors = F.conv2d(luminance, kernel, padding=patch_size // 2)
    diff = neighbors - luminance
    return diff / torch.sqrt(torch.tensor(0.81, dtype=diff.dtype, device=diff.device) + diff**2)


def _soft_hamming(a: torch.Tensor, b: torch.Tensor, threshold: float) -> torch.Tensor:
    """SMURF ``soft_hamming``：对邻域维求和的软汉明距离 ``[B,1,H,W]``。"""
    squared = (a - b) ** 2
    return (squared / (threshold + squared)).sum(dim=1, keepdim=True)


def _census_loss(
    reference: torch.Tensor,
    warped: torch.Tensor,
    support: torch.Tensor,
    config: RaftLossConfig,
) -> torch.Tensor:
    """SMURF ``census_loss``：按支持加权的 robust census 差（求和 / 支持求和）。"""
    census_reference = _census_transform(reference, config.census_patch)
    census_warped = _census_transform(warped, config.census_patch)
    hamming = _soft_hamming(census_reference, census_warped, config.census_threshold)
    padded = _zero_mask_border(support, config.census_patch)
    diff = abs_robust_loss(hamming, config.robust_eps, config.robust_q) * padded
    return diff.sum() / (padded.detach().sum() + 1e-6)


def _photometric_loss(
    reference: torch.Tensor,
    warped: torch.Tensor,
    support: torch.Tensor,
    config: RaftLossConfig,
) -> torch.Tensor:
    """RAW 亮度 robust 重投影项：``abs_robust_loss(|I_ref - I_warp|)`` 按支持加权平均。"""
    diff = abs_robust_loss(reference - warped, config.robust_eps, config.robust_q) * support
    return diff.sum() / (support.detach().sum() + 1e-6)


def _image_grads(values: torch.Tensor, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """前向差分 ``(沿 H, 沿 W)``（SMURF ``image_grads`` 的 PyTorch 版）。"""
    grad_h = values[..., stride:, :] - values[..., :-stride, :]
    grad_w = values[..., :, stride:] - values[..., :, :-stride]
    return grad_h, grad_w


def _edge_weighting_fn(config: RaftLossConfig):
    """SMURF 的 gaussian / exponential 边缘权重（通道维取均值）。"""
    constant = config.edge_constant

    def weighting(gradients: torch.Tensor) -> torch.Tensor:
        if config.edge_weighting == "gaussian":
            return torch.exp(-((constant * gradients) ** 2).mean(dim=1, keepdim=True))
        return torch.exp(-(constant * gradients).abs().mean(dim=1, keepdim=True))

    return weighting


def _smoothness_levels(
    luminance: torch.Tensor, flow_px: torch.Tensor, level: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 SMURF ``smoothness_at_level`` 把亮度与 flow 降到 1/2**level（flow 同步缩到该层单位）。"""
    for _ in range(level):
        luminance = F.interpolate(luminance, scale_factor=0.5, mode="bilinear", align_corners=False)
        flow_px = (
            F.interpolate(flow_px, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        )
    return luminance, flow_px


def _first_order_smoothness(image, flow_px, edge_weighting) -> torch.Tensor:
    """SMURF ``first_order_smoothness_loss``：权重取自图像梯度，损失取 flow 梯度。"""
    image_grad_h, image_grad_w = _image_grads(image)
    flow_grad_h, flow_grad_w = _image_grads(flow_px)
    weights_h = edge_weighting(image_grad_h)
    weights_w = edge_weighting(image_grad_w)
    return (
        (weights_h * robust_l1(flow_grad_h)).mean() + (weights_w * robust_l1(flow_grad_w)).mean()
    ) / 2.0


def _second_order_smoothness(image, flow_px, edge_weighting) -> torch.Tensor:
    """SMURF ``second_order_smoothness_loss``（权重用 stride=2 的图像梯度，只取非混合二阶导）。"""
    image_grad_h, image_grad_w = _image_grads(image, stride=2)
    flow_grad_h, flow_grad_w = _image_grads(flow_px)
    flow_grad_hh, _ = _image_grads(flow_grad_h)
    _, flow_grad_ww = _image_grads(flow_grad_w)
    weights_hh = edge_weighting(image_grad_h)
    weights_ww = edge_weighting(image_grad_w)
    return (
        (weights_hh * robust_l1(flow_grad_hh)).mean()
        + (weights_ww * robust_l1(flow_grad_ww)).mean()
    ) / 2.0

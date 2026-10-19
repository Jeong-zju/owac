"""S1 正式 RAW 测量：ConvNeXt-B 骨干 + 多尺度融合 + 几何 / 方差头 + 单帧候选。

网络结构（S1 提示词冻结）：

* 骨干是作者 ConvNeXt-B 的完整四级 stage（depths ``[3,3,27,3]``、dims
  ``[128,256,512,1024]``），只替换输入 stem 的通道数并移除分类头 / 全局池化；
* 四级空间场各经 1×1 lateral 投影到 256 通道，双线性上采样到 stage-1 网格相加，
  再由 3×3/256 输出层形成测量场（``align_corners=False``）；
* 几何头输出同单位的深度均值与方差（正值头 + softplus + config floor），潜在头输出
  ``z`` 与固定初值的 ``R_z``（S1 不做创新校准，S2 才在冻结 ``z`` 之后标定）；
* 候选锚点按 8×16 分区选每区最大几何置信度点，置信度只来自模型输出，不读 GT。

``measure`` 只接受 `MeasurementFrame` 列表：没有历史状态参数，也不读取 frame 的
provenance 字段；同一帧在不同轨迹 / episode 下必须给出相同输出。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ._vendor.convnext import convnext_base
from .contracts import (
    ANCHOR_CANDIDATES_MAX,
    DENSE_LATENT_CHANNELS,
    AnchorCandidates,
    DenseMeasurement,
    MeasurementFrame,
    SampleGrid,
    build_sample_grid,
)
from .full_config import N04FullConfig

__all__ = [
    "MemorylessCarry",
    "N04FullMeasurement",
    "PreprocessedBatch",
    "build_measurement",
    "extract_cfa_planes",
    "position_with_covariance",
    "preprocess_frames",
]


# --------------------------------------------------------------------------------------
# 预处理：RAW 单位、CFA 相位、pad 与支持
# --------------------------------------------------------------------------------------


def extract_cfa_planes(raw: torch.Tensor, pattern: str) -> torch.Tensor:
    """单平面 CFA -> 四相位平面 ``[4,H/2,W/2]``，按 ``CFA_PLANE_OFFSETS`` 顺序。

    ``raw`` 形状 ``[C,H,W]``（C 必须为 1）；平面 ``(i,j)`` 取原图 ``(2i+δy, 2j+δx)``。
    不做任何插值 / demosaic，也不把四个相位混到同一中心。
    """
    from .measurement_geometry import cfa_plane_offsets

    if raw.dim() != 3 or raw.shape[0] != 1:
        raise ValueError(f"CFA 需要 [1,H,W]，实际 {tuple(raw.shape)}")
    height, width = raw.shape[1], raw.shape[2]
    if height % 2 or width % 2:
        raise ValueError(f"CFA 图像尺寸必须为偶数，实际 {height}x{width}")
    planes = []
    for _, dy, dx in cfa_plane_offsets(pattern):
        planes.append(raw[0, dy::2, dx::2])
    return torch.stack(planes, dim=0)


@dataclass
class PreprocessedBatch:
    """一批同尺寸帧的骨干输入与支持掩码（供损失、缓存与测试共用）。"""

    values: torch.Tensor  # [B,C,Hp,Wp] FP32
    support: torch.Tensor  # [B,1,Hp,Wp] bool，骨干输入平面上“该像素有效”
    saturated: torch.Tensor  # [B,1,Hp,Wp] bool
    pad_hw: tuple[int, int]
    pad_xy: tuple[float, float]
    frames: tuple[MeasurementFrame, ...]


def _pad_to_multiple(
    tensor: torch.Tensor, multiple: int, mode: str = "constant"
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = tensor.shape[-2:]
    pad_h = -(-height // multiple) * multiple
    pad_w = -(-width // multiple) * multiple
    if (pad_h, pad_w) == (height, width):
        return tensor, (pad_h, pad_w)
    padded = F.pad(tensor, (0, pad_w - width, 0, pad_h - height), mode=mode)
    return padded, (pad_h, pad_w)


def _block_all_valid(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """``[1,H,W]`` 有效性 -> 每 ``factor×factor`` 块**全部有效**才算有效。"""
    if factor == 1:
        return mask.to(torch.bool)
    pooled = F.avg_pool2d(mask.to(torch.float32), kernel_size=factor, stride=factor)
    return pooled >= 1.0 - 1e-6


def _block_any(mask: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return mask.to(torch.bool)
    pooled = F.max_pool2d(mask.to(torch.float32), kernel_size=factor, stride=factor)
    return pooled > 0.5


def preprocess_frames(
    frames: Sequence[MeasurementFrame], config: N04FullConfig
) -> PreprocessedBatch:
    """把当前帧变成骨干输入：显式单位缩放、CFA 相位拆分、NaN 清理、pad 与支持。

    无效值一律先 ``where(finite, value, 0)`` 清理再进网络（``NaN * 0`` 与掩码分支的
    梯度污染都由此避免）。支持掩码按“整块全部有效”的保守规则定义：CFA 先把传感器
    有效掩码按 2×2 块归到相位平面，再与骨干输入一起 pad；融合网格上的归约由
    `_feature_support` 完成。饱和是**标记**而不是支持：饱和像素仍携带几何信息，
    不能在测量层清零。
    """
    if not frames:
        raise ValueError("measure 至少需要一帧")
    multiple = config.measurement.pad_multiple
    mode = frames[0].input_mode
    sizes = {tuple(frame.size_hw) for frame in frames}
    if len(sizes) != 1:
        raise ValueError(f"同一批必须是相同尺寸（按尺寸分组后再批）：{sorted(sizes)}")

    values_list: list[torch.Tensor] = []
    support_list: list[torch.Tensor] = []
    saturated_list: list[torch.Tensor] = []
    pad_hw: tuple[int, int] = (0, 0)
    for frame in frames:
        if frame.input_mode != mode:
            raise ValueError("同一批不允许混用输入模式")
        raw = frame.raw.to(torch.float32) * float(frame.scale.factor)
        finite = torch.isfinite(raw)
        invalid = ~finite
        if frame.invalid_mask is not None:
            mask = frame.invalid_mask.to(torch.bool)
            if mask.shape != frame.size_hw:
                raise ValueError(f"invalid_mask 尺寸 {tuple(mask.shape)} 与 {frame.size_hw} 不一致")
            invalid = invalid | mask.unsqueeze(0).expand_as(raw)
        raw = torch.where(finite, raw, torch.zeros_like(raw))
        level = frame.saturation_level if frame.saturation_level is not None else 1.0
        saturated = raw >= (0.99 * level)
        support_sensor = ~invalid
        if mode == "bayer_cfa":
            if frame.cfa_pattern is None:
                raise ValueError("bayer_cfa 帧缺少 cfa_pattern")
            planes = extract_cfa_planes(raw, frame.cfa_pattern)  # [4,H/2,W/2]
            support_plane = _block_all_valid(support_sensor.to(torch.float32), 2)
            sat_plane = _block_any(saturated.to(torch.float32), 2)
            values, pad_hw = _pad_to_multiple(planes, multiple, config.measurement.padding_mode)
            support = F.pad(
                support_plane,
                (0, pad_hw[1] - support_plane.shape[-1], 0, pad_hw[0] - support_plane.shape[-2]),
            )
            sat = F.pad(
                sat_plane, (0, pad_hw[1] - sat_plane.shape[-1], 0, pad_hw[0] - sat_plane.shape[-2])
            )
        else:
            values, pad_hw = _pad_to_multiple(raw, multiple, config.measurement.padding_mode)
            support_sensor = support_sensor.all(dim=0, keepdim=True)
            sat_sensor = saturated.any(dim=0, keepdim=True)
            support = F.pad(
                support_sensor,
                (0, pad_hw[1] - support_sensor.shape[-1], 0, pad_hw[0] - support_sensor.shape[-2]),
            )
            sat = F.pad(
                sat_sensor,
                (0, pad_hw[1] - sat_sensor.shape[-1], 0, pad_hw[0] - sat_sensor.shape[-2]),
            )
        values_list.append(values)
        support_list.append(support.to(torch.bool))
        saturated_list.append(sat.to(torch.bool))
    return PreprocessedBatch(
        values=torch.stack(values_list, dim=0),
        support=torch.stack(support_list, dim=0),
        saturated=torch.stack(saturated_list, dim=0),
        pad_hw=pad_hw,
        pad_xy=(0.0, 0.0),
        frames=tuple(frames),
    )


def _feature_support(support: torch.Tensor, stride: int) -> torch.Tensor:
    """骨干输入平面的支持 -> 融合网格支持：整格全部有效才算有效。"""
    return _block_all_valid(support, stride)


# --------------------------------------------------------------------------------------
# 网络
# --------------------------------------------------------------------------------------


class N04FullMeasurement(nn.Module):
    """S1 正式测量网络：``n04_full_v1`` 的测量部分（无状态、无历史）。"""

    def __init__(self, config: N04FullConfig) -> None:
        super().__init__()
        self.config = config
        channels = 3 if config.input_mode == "rawvla_rgb3" else 4
        backbone = convnext_base(
            pretrained=False,
            in_chans=channels,
            drop_path_rate=float(config.backbone.drop_path_rate),
            layer_scale_init_value=float(config.backbone.layer_scale_init_value),
        )
        # 分类头与最终池化 LayerNorm 不属于测量网络；构造后移除，参数集合只剩测量路径。
        backbone.head = None
        backbone.norm = None
        self.backbone = backbone

        dims = tuple(int(v) for v in config.backbone.dims)
        self.laterals = nn.ModuleList(
            [nn.Conv2d(dim, config.measurement.fusion_channels, kernel_size=1) for dim in dims]
        )
        self.fusion = nn.Conv2d(
            config.measurement.fusion_channels,
            config.measurement.fusion_channels,
            kernel_size=3,
            padding=1,
        )
        hidden = config.measurement.head_hidden
        self.depth_head = nn.Sequential(
            nn.Conv2d(DENSE_LATENT_CHANNELS, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.variance_head = nn.Sequential(
            nn.Conv2d(DENSE_LATENT_CHANNELS, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # 方差头的偏置初始化到配置的初始 σ：softplus(b) = σ² - floor。
        target = max(
            float(config.measurement.variance_init_sigma_m) ** 2
            - float(config.measurement.variance_floor),
            1e-6,
        )
        self.variance_head[-1].bias.data.fill_(math.log(math.expm1(target)))

    # -- 前向各层显式拆分，便于缓存 / 消融共用同一份计算 -------------------------------
    def backbone_levels(self, values: torch.Tensor) -> list[torch.Tensor]:
        """四级空间场（stride 4/8/16/32，相对骨干输入平面）。"""
        levels = []
        x = values
        for index in range(4):
            x = self.backbone.downsample_layers[index](x)
            x = self.backbone.stages[index](x)
            levels.append(x)
        return levels

    def fuse(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        """1×1 lateral -> 上采样到 stage-1 网格相加 -> 3×3/256 测量场。"""
        target = levels[0].shape[-2:]
        fused = None
        for lateral, level in zip(self.laterals, levels, strict=True):
            projected = lateral(level)
            if projected.shape[-2:] != target:
                projected = F.interpolate(
                    projected, size=target, mode="bilinear", align_corners=False
                )
            fused = projected if fused is None else fused + projected
        assert fused is not None
        return self.fusion(fused)

    def heads(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(depth_mean [B,1,h,w] 米, depth_variance [B,1,h,w] 米²)。"""
        floor = float(self.config.measurement.depth_floor_m)
        scale = float(self.config.data.depth_scale_m)
        depth = floor + scale * F.softplus(self.depth_head(latent))
        variance = float(self.config.measurement.variance_floor) + F.softplus(
            self.variance_head(latent)
        )
        return depth, variance

    def forward_levels(self, values: torch.Tensor) -> list[torch.Tensor]:
        """兼容入口：直接返回四级空间场。"""
        return self.backbone_levels(values)

    # -- 单帧测量接口 ---------------------------------------------------------------
    def measure(
        self,
        frames: Sequence[MeasurementFrame],
        *,
        autocast_dtype: torch.dtype | None = None,
    ) -> DenseMeasurement:
        """对一批同尺寸当前帧输出 `DenseMeasurement`；不读取任何历史或 provenance。"""
        for frame in frames:
            if frame.input_mode != self.config.input_mode:
                raise ValueError(
                    f"该测量网络是为 {self.config.input_mode!r} 构建的，"
                    f"收到 {frame.input_mode!r} 帧；两种输入模式各自建网、分别计分"
                )
        batch = preprocess_frames(frames, self.config)
        device_type = batch.values.device.type
        use_amp = autocast_dtype is not None and device_type in ("cuda", "cpu")
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype or torch.bfloat16,
            enabled=bool(use_amp),
        ):
            levels = self.backbone_levels(batch.values)
        levels = [level.float() for level in levels]
        latent = self.fuse(levels)
        depth, variance = self.heads(latent)
        support = _feature_support(batch.support, self.config.measurement.fusion_stride)
        saturated = _feature_support(batch.saturated, self.config.measurement.fusion_stride)
        valid = support & torch.isfinite(depth) & (depth > 0)
        latent_variance = torch.full_like(
            latent, float(self.config.measurement.latent_variance_init)
        )

        grids = tuple(
            build_sample_grid(
                input_mode=frame.input_mode,
                sensor_size_hw=frame.size_hw,
                pad_xy=batch.pad_xy,
                pad_hw=batch.pad_hw,
                stride=self.config.measurement.fusion_stride,
                cfa_pattern=frame.cfa_pattern,
            ).with_device(latent.device)
            for frame in frames
        )
        candidates = self._select_candidates(
            frames=frames,
            latent=latent,
            depth=depth,
            variance=variance,
            valid=valid,
            grids=grids,
        )
        metadata = {
            "model_kind": self.config.model_kind,
            "checkpoint_schema": self.config.checkpoint_schema,
            "config_sha256": self.config.sha256(),
            "input_mode": self.config.input_mode,
            "latent_variance_calibration": "fixed_init_uncalibrated_s1",
            "fusion_stride_backbone_input_px": self.config.measurement.fusion_stride,
        }
        return DenseMeasurement(
            feature_pyramid=tuple(level for level in levels),
            latent_mean_map=latent,
            latent_variance_map=latent_variance,
            depth_mean=depth,
            depth_variance=variance,
            valid=valid,
            support=support,
            saturated=saturated,
            sample_grid=grids[0],
            candidates=candidates,
            measurement_ids=tuple(frame.measurement_id for frame in frames),
            units={
                "latent_mean_map": "unnormalized_feature",
                "latent_variance_map": "feature_unit_squared_uncalibrated",
                "depth_mean": "meter",
                "depth_variance": "meter_squared",
                "position_camera": "meter",
                "position_covariance": "meter_squared",
                "pixel": "original_px",
            },
            metadata=metadata,
        )

    def _select_candidates(
        self,
        *,
        frames: Sequence[MeasurementFrame],
        latent: torch.Tensor,
        depth: torch.Tensor,
        variance: torch.Tensor,
        valid: torch.Tensor,
        grids: Sequence[SampleGrid],
    ) -> AnchorCandidates:
        # 分区数按合同固定为 8×16；融合网格比分区数还小时按网格收缩（小分辨率 / CFA
        # 四相位平面），否则会出现空分区、argmax 无定义。
        rows = min(self.config.measurement.candidate_rows, int(latent.shape[-2]))
        cols = min(self.config.measurement.candidate_cols, int(latent.shape[-1]))
        batch, _, height, width = latent.shape
        device = latent.device
        if rows * cols > ANCHOR_CANDIDATES_MAX:
            raise ValueError("候选分区数超过 128")
        if height % rows or width % cols:
            row_edges = torch.linspace(0, height, rows + 1, device=device).round().long()
            col_edges = torch.linspace(0, width, cols + 1, device=device).round().long()
            if int(row_edges.diff().min()) < 1 or int(col_edges.diff().min()) < 1:
                raise ValueError(f"融合网格 {height}x{width} 无法分成 {rows}x{cols} 个非空分区")
        else:
            row_edges = torch.arange(0, height + 1, height // rows, device=device)
            col_edges = torch.arange(0, width + 1, width // cols, device=device)

        confidence = 1.0 / torch.sqrt(variance + float(self.config.measurement.variance_floor))
        confidence = confidence[:, 0]  # [B,h,w]
        valid_map = valid[:, 0] & torch.isfinite(confidence)
        confidence = torch.where(valid_map, confidence, torch.full_like(confidence, float("-inf")))

        xy_feature, cand_valid, cand_conf = [], [], []
        partition_row, partition_col = [], []
        for row in range(rows):
            for col in range(cols):
                r0, r1 = int(row_edges[row]), int(row_edges[row + 1])
                c0, c1 = int(col_edges[col]), int(col_edges[col + 1])
                window = confidence[:, r0:r1, c0:c1]
                flat = window.reshape(batch, -1)
                best = flat.argmax(dim=1)
                has = torch.isfinite(flat.gather(1, best.unsqueeze(1))).squeeze(1)
                local_r = (best // (c1 - c0)).to(torch.float32)
                local_c = (best % (c1 - c0)).to(torch.float32)
                global_r = local_r + (r0 + r1 - 1) / 2.0
                global_c = local_c + (c0 + c1 - 1) / 2.0
                xy_feature.append(torch.stack((global_c, global_r), dim=-1))
                cand_valid.append(has)
                cand_conf.append(
                    torch.where(
                        has,
                        flat.gather(1, best.unsqueeze(1)).squeeze(1),
                        torch.zeros_like(has, dtype=confidence.dtype),
                    )
                )
                partition_row.append(torch.full_like(best, row))
                partition_col.append(torch.full_like(best, col))
        xy_feature_t = torch.stack(xy_feature, dim=1)  # [B,N,2] (x,y) feature px
        cand_valid_t = torch.stack(cand_valid, dim=1)
        cand_conf_t = torch.stack(cand_conf, dim=1)

        descriptor = F.grid_sample(
            latent,
            _normalised_grid(xy_feature_t, (height, width)),
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )  # [B,256,N]
        descriptor = descriptor.squeeze(2).permute(0, 2, 1)

        xy_sensor_t = torch.empty_like(xy_feature_t)
        position = torch.full((batch, xy_feature_t.shape[1], 3), float("nan"), device=device)
        covariance = torch.full((batch, xy_feature_t.shape[1], 3, 3), float("nan"), device=device)
        position_valid = torch.zeros(batch, xy_feature_t.shape[1], dtype=torch.bool, device=device)
        depth_at = torch.zeros(batch, xy_feature_t.shape[1], device=device)
        var_at = torch.zeros(batch, xy_feature_t.shape[1], device=device)
        for index, grid in enumerate(grids):
            sensor = grid.feature_to_sensor.apply(
                _pixel_points(xy_feature_t[index], "feature_px", device)
            )
            xy_sensor_t[index] = sensor.xy
            depth_index = F.grid_sample(
                depth[index : index + 1],
                _normalised_grid(xy_feature_t[index : index + 1], (height, width)),
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )[0, 0, 0]
            var_index = F.grid_sample(
                variance[index : index + 1],
                _normalised_grid(xy_feature_t[index : index + 1], (height, width)),
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )[0, 0, 0]
            depth_at[index] = depth_index
            var_at[index] = var_index
            if frames[index].K is None:
                continue
            K = frames[index].K.to(device=device, dtype=torch.float32)
            pix_valid = cand_valid_t[index] & torch.isfinite(depth_index) & (depth_index > 0)
            pos, cov = position_with_covariance(
                sensor.xy,
                depth_index,
                var_index,
                K,
                sigma_pixel=float(self.config.measurement.position_pixel_sigma_px),
                valid=pix_valid,
            )
            position[index] = pos
            covariance[index] = cov
            position_valid[index] = pix_valid
        return AnchorCandidates(
            xy_feature=xy_feature_t,
            xy_sensor=xy_sensor_t,
            valid=cand_valid_t,
            confidence=cand_conf_t,
            partition_row=torch.stack(partition_row, dim=1),
            partition_col=torch.stack(partition_col, dim=1),
            descriptor=descriptor,
            position_camera=position,
            position_covariance=covariance,
            position_valid=position_valid,
            anchors_per_row=rows,
            anchors_per_col=cols,
        )


def _normalised_grid(xy_feature: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    height, width = size_hw
    nx = 2.0 * (xy_feature[..., 0] + 0.5) / width - 1.0
    ny = 2.0 * (xy_feature[..., 1] + 0.5) / height - 1.0
    return torch.stack((nx, ny), dim=-1).unsqueeze(1)


def _pixel_points(xy: torch.Tensor, unit: str, device: torch.device):
    from .measurement_geometry import PixelPoints

    return PixelPoints(xy=xy.to(device=device, dtype=torch.float32), unit=unit)


def position_with_covariance(
    sensor_xy: torch.Tensor,
    depth: torch.Tensor,
    depth_variance: torch.Tensor,
    K: torch.Tensor,
    *,
    sigma_pixel: float,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由 ``p = d · K⁻¹ [u,v,1]`` 的解析 Jacobian 传播像素 / 深度协方差。

    ``Σ = J diag(σu², σv², σd²) Jᵀ``，``J = [∂p/∂u, ∂p/∂v, ∂p/∂d]``；无效点返回 NaN。
    """
    if K.shape != (3, 3):
        raise ValueError("K 需要 [3,3]")
    dtype = torch.promote_types(
        torch.promote_types(K.dtype, sensor_xy.dtype),
        torch.promote_types(depth.dtype, depth_variance.dtype),
    )
    K = K.to(dtype)
    sensor_xy = sensor_xy.to(dtype)
    depth = depth.to(dtype)
    depth_variance = depth_variance.to(dtype)
    K_inv = torch.linalg.inv(K)
    ones = torch.ones_like(sensor_xy[..., :1])
    homogeneous = torch.cat((sensor_xy, ones), dim=-1)
    rays = torch.einsum("ij,...j->...i", K_inv, homogeneous)  # [N,3]
    d = depth.unsqueeze(-1)
    position = rays * d
    jac = torch.stack(
        (d * K_inv[:, 0], d * K_inv[:, 1], rays),
        dim=-1,
    )  # [N,3,3]
    scales = torch.stack(
        (
            torch.full_like(depth, float(sigma_pixel) ** 2),
            torch.full_like(depth, float(sigma_pixel) ** 2),
            depth_variance,
        ),
        dim=-1,
    )  # [N,3]
    covariance = torch.einsum("...ij,...j,...kj->...ik", jac, scales, jac)
    mask = valid.unsqueeze(-1)
    position = torch.where(mask, position, torch.full_like(position, float("nan")))
    mask3 = valid.unsqueeze(-1).unsqueeze(-1)
    covariance = torch.where(mask3, covariance, torch.full_like(covariance, float("nan")))
    return position, covariance


class MemorylessCarry:
    """无记忆基线：可见时直接用当前测量，缺失时保留最后有效测量并记录 age。

    S2 的状态模块必须与它比较；这里只实现“无记忆 + 显式 age”这一固定逻辑，
    保证后续阶段不会悄悄换成更弱的测量。
    """

    def __init__(self, latent: torch.Tensor, measurement_ids: Sequence[str]) -> None:
        if latent.dim() != 4:
            raise ValueError("latent 需要 [B,C,h,w]")
        if len(measurement_ids) != latent.shape[0]:
            raise ValueError("measurement_ids 与 batch 不一致")
        self._latent = latent.detach().clone()
        self._ids = list(measurement_ids)
        self._age = torch.zeros(latent.shape[0], dtype=torch.long, device=latent.device)
        self._fresh = torch.ones(latent.shape[0], dtype=torch.bool, device=latent.device)

    def update(self, latent: torch.Tensor, measurement_ids: Sequence[str], available: torch.Tensor):
        """``available`` 为 [B] bool；不可用样本沿用上一份测量并把 age 加一。"""
        if latent.shape != self._latent.shape:
            raise ValueError("carry 只接受与初始形状一致的测量")
        if len(measurement_ids) != latent.shape[0] or available.shape != (latent.shape[0],):
            raise ValueError("measurement_ids / available 与 batch 不一致")
        keep = available.to(latent.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self._latent = torch.where(keep, latent.detach(), self._latent)
        self._ids = [
            mid if bool(fresh) else old
            for mid, old, fresh in zip(measurement_ids, self._ids, available, strict=True)
        ]
        self._age = torch.where(
            available.to(self._age.device), torch.zeros_like(self._age), self._age + 1
        )
        self._fresh = available.to(self._fresh.device).clone()

    @property
    def latent(self) -> torch.Tensor:
        return self._latent

    @property
    def measurement_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def age_frames(self) -> torch.Tensor:
        return self._age

    @property
    def fresh(self) -> torch.Tensor:
        return self._fresh


def build_measurement(config: N04FullConfig) -> N04FullMeasurement:
    """按配置构造测量网络（不加载权重、不连接设备）。"""
    return N04FullMeasurement(config)

"""N04 v1.1 质量头：当前 RAW 测量 -> 每个支持点的 256 维潜在测量方差 R。

契约（docs/n04_v11_s2_interfaces.md §3）：

* 输入只有当前帧：预处理后的 RAW ``values [B,C,Hp,Wp]``、``support/saturated
  [B,1,Hp,Wp]``、当前融合特征 ``latent [B,256,h,w]``、支持点 ``xy_px [B,N,2]``
  （原始像素）；
* 输出 ``R_diag [B,N,256]``（FP32，``>= 1e-6``），与潜在测量语义一致；
* 不读历史 innovation residual、不读 S1 深度方差、不读当前 flow。

RAW 统计在 ``values`` 上按融合 stride 做局部归约（绝对强度、能量、梯度、饱和与
支持分数），因此保留的是传感器尺度上的量，不是被 LayerNorm 抹掉的中间激活；
支持点处用与测量层相同的坐标链重采样。整个前向是单帧纯函数：同一帧在不同轨迹 /
episode 下给出相同 R。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import DENSE_LATENT_CHANNELS, SampleGrid, build_sample_grid
from .measurement_geometry import PixelPoints, normalized_from_xy

__all__ = [
    "RAW_STAT_CHANNELS",
    "QualityHead",
    "feature_grid_from_sizes",
    "sample_feature_map",
]

# 统计通道顺序固定，测试与调试按此顺序取值。
RAW_STAT_CHANNELS = ("mean", "std", "log_mean", "edge", "support", "saturated")
_STAT_EPS = 1e-6
_MIN_MEASUREMENT_VARIANCE = 1e-6


def sample_feature_map(feature_map: torch.Tensor, xy_px: torch.Tensor, grid: SampleGrid):
    """在原始像素 ``xy_px [B,N,2]`` 处双线性采样 ``feature_map [B,C,h,w]`` -> ``[B,N,C]``。

    ``grid`` 提供 ``original_px -> feature_px`` 的仿射（``SampleGrid.sensor_to_feature``），
    归一化用 ``normalized_from_xy(..., (h, w))``，即 ``align_corners=False`` 的像素中心
    约定。质量头与状态估计器共用这一份采样约定，禁止在别处再写一遍 ``xy / stride``。
    """
    if feature_map.dim() != 4:
        raise ValueError(f"feature_map 需要 [B,C,h,w]，实际 {tuple(feature_map.shape)}")
    if xy_px.shape[-1] != 2:
        raise ValueError(f"xy_px 需要 [...,2]，实际 {tuple(xy_px.shape)}")
    if xy_px.shape[0] != feature_map.shape[0]:
        raise ValueError("xy_px 与 feature_map 的 batch 不一致")
    height, width = feature_map.shape[-2:]
    points = PixelPoints(xy=xy_px.to(dtype=torch.float32), unit="original_px")
    xy_feature = grid.with_device(xy_px.device).sensor_to_feature.apply(points)
    normalized = normalized_from_xy(xy_feature, (height, width)).unsqueeze(1)  # [B,1,N,2]
    sampled = F.grid_sample(
        feature_map.to(torch.float32),
        normalized,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return sampled.squeeze(2).permute(0, 2, 1).contiguous()


def feature_grid_from_sizes(
    values_size_hw: tuple[int, int], feature_size_hw: tuple[int, int]
) -> SampleGrid:
    """由骨干输入平面与融合网格尺寸反推坐标链（假设无 CFA 折叠、无左上方 pad）。

    只用于调用方没有显式传 `SampleGrid` 的情形（RAW3 / RGB3 输入、pad 只加在右下方）。
    CFA 与 resize 情形必须由调用方传入真实 `SampleGrid`。
    """
    height, width = (int(values_size_hw[0]), int(values_size_hw[1]))
    feature_h, feature_w = (int(feature_size_hw[0]), int(feature_size_hw[1]))
    if feature_h <= 0 or feature_w <= 0 or height % feature_h or width % feature_w:
        raise ValueError(
            f"{height}x{width} 无法被融合网格 {feature_h}x{feature_w} 整除；必须显式传 grid"
        )
    stride_h, stride_w = height // feature_h, width // feature_w
    if stride_h != stride_w:
        raise ValueError(f"非各向同性 stride {stride_h}x{stride_w} 不能反推坐标链")
    return build_sample_grid(
        input_mode="rawvla_rgb3",
        sensor_size_hw=(height, width),
        pad_xy=(0.0, 0.0),
        pad_hw=(height, width),
        stride=stride_h,
    )


def _as_plane_mask(mask: torch.Tensor, values: torch.Tensor, name: str) -> torch.Tensor:
    if mask.dim() == values.dim() - 1:
        mask = mask.unsqueeze(1)
    if mask.dim() != values.dim() or mask.shape[1] != 1:
        raise ValueError(f"{name} 需要 [B,1,H,W] 或 [B,H,W]，实际 {tuple(mask.shape)}")
    if mask.shape[-2:] != values.shape[-2:]:
        raise ValueError(
            f"{name} 与 values 尺寸不一致：{tuple(mask.shape)} / {tuple(values.shape)}"
        )
    return mask.to(torch.float32)


def _raw_statistics(
    values: torch.Tensor, support: torch.Tensor, saturated: torch.Tensor, stride: int
) -> torch.Tensor:
    """``[B,S,h,w]``：在当前 RAW 上按 stride 归约的局部统计（顺序见 `RAW_STAT_CHANNELS`）。"""
    height, width = values.shape[-2:]
    if height % stride or width % stride:
        raise ValueError(f"骨干输入 {height}x{width} 不是 stride {stride} 的整数倍")
    intensity = values.mean(dim=1, keepdim=True)  # [B,1,H,W]
    energy = values.square().mean(dim=1, keepdim=True)
    dx = (intensity[..., 1:] - intensity[..., :-1]).abs()
    dy = (intensity[..., 1:, :] - intensity[..., :-1, :]).abs()
    edge = 0.5 * (F.pad(dx, (0, 1)) + F.pad(dy, (0, 0, 0, 1)))
    mean = F.avg_pool2d(intensity, stride)
    variance = (F.avg_pool2d(energy, stride) - mean.square()).clamp_min(0.0)
    return torch.cat(
        (
            mean,
            variance.sqrt(),
            torch.log(mean.clamp_min(_STAT_EPS)),
            F.avg_pool2d(edge, stride),
            F.avg_pool2d(support, stride),
            F.avg_pool2d(saturated, stride),
        ),
        dim=1,
    )


class QualityHead(nn.Module):
    """当前帧质量头：``(values, support, saturated, latent, xy_px) -> R_diag``。

    统计路径与特征路径各经过一个小 MLP，拼起来输出 256 维 log 方差，
    ``R = floor + softplus(log_var)``。输出下界由 ``floor`` 与 ``clamp_min(1e-6)``
    双重保证；``log_var`` 只做上限截断，避免 FP32 里 softplus 溢出成 inf。
    除这条正值参数化外不做任何 ``nan_to_num`` 之类的掩盖。
    """

    def __init__(
        self,
        *,
        obs_dim: int = DENSE_LATENT_CHANNELS,
        hidden: int = 64,
        initial_variance: float = 1.0,
        floor: float = _MIN_MEASUREMENT_VARIANCE,
        log_var_max: float = 30.0,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or hidden <= 0:
            raise ValueError("obs_dim / hidden 必须为正")
        if floor <= 0 or initial_variance <= floor:
            raise ValueError("需要 0 < floor < initial_variance")
        self.obs_dim = int(obs_dim)
        self.floor = float(floor)
        self.log_var_max = float(log_var_max)
        stat_dim = len(RAW_STAT_CHANNELS)
        self.stat_encoder = nn.Sequential(
            nn.Linear(stat_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.output = nn.Linear(2 * hidden, self.obs_dim)
        # 偏置初始化到配置里的初始潜在方差：softplus(b) = initial_variance - floor。
        target = max(float(initial_variance) - self.floor, _MIN_MEASUREMENT_VARIANCE)
        self.output.bias.data.fill_(math.log(math.expm1(target)))

    def forward(
        self,
        values: torch.Tensor,
        support: torch.Tensor,
        saturated: torch.Tensor,
        latent: torch.Tensor,
        xy_px: torch.Tensor,
        *,
        grid: SampleGrid | None = None,
    ) -> torch.Tensor:
        """返回 ``R_diag [B,N,obs_dim]``，FP32 且逐元素 ``>= 1e-6``。"""
        values = values.to(torch.float32)
        latent = latent.to(torch.float32)
        support = _as_plane_mask(support, values, "support")
        saturated = _as_plane_mask(saturated, values, "saturated")
        if xy_px.shape[0] != values.shape[0]:
            raise ValueError("xy_px 与当前帧的 batch 不一致")
        if grid is None:
            grid = feature_grid_from_sizes(
                (values.shape[-2], values.shape[-1]), (latent.shape[-2], latent.shape[-1])
            )
        stride = int(values.shape[-1]) // int(latent.shape[-1])
        statistics = _raw_statistics(values, support, saturated, stride)
        stat_features = sample_feature_map(statistics, xy_px, grid)  # [B,N,S]
        features = sample_feature_map(latent, xy_px, grid)  # [B,N,obs_dim]
        hidden = torch.cat(
            (self.stat_encoder(stat_features), self.latent_encoder(features)), dim=-1
        )
        log_var = self.output(hidden).clamp(max=self.log_var_max)
        variance = self.floor + F.softplus(log_var)
        return variance.clamp_min(_MIN_MEASUREMENT_VARIANCE).to(torch.float32)

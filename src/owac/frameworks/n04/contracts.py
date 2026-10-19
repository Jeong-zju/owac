"""S1 正式测量的数据合同：单帧输入、稠密测量输出、候选锚点与缓存序列化。

本模块只定义形状、单位、坐标与可序列化结构，不含网络层。所有合同都围绕同一条
因果要求：``measure(frame)`` 只读取**当前帧**及其传感器元数据，不接触历史状态、
轨迹 ID 之外的任何上下文；轨迹 / episode / 时间戳只进入 provenance 与 measurement ID，
不得影响数值输出。

单位约定（S1 冻结）：

* 图像张量 ``[C,H,W]``：``rawvla_rgb3`` 为 3 通道；``bayer_cfa`` 为单平面原始采样。
* 归一化线性测量单位 ``normalized_linear``：``raw * InputScale.factor``，方差乘
  ``factor**2``。LIBERO uint8 显式 /255；RoboTwin 与受控渲染保持其线性 float 单位。
* 像素坐标：仓库存放 ``integer pixel centres``，特征网格用 ``align_corners=False`` 中心
  对齐；原图像素与特征像素是不同单位，必须经 `SampleGrid` 的仿射换算。
* 深度：米；位置：米（相机系）；协方差：对应单位的平方。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch

from .measurement_geometry import (
    PixelAffine,
    PixelPoints,
    PixelUnit,
    cfa_plane_offsets,
    compose_pixel_affine,
    crop_pixel_affine,
    identity_pixel_affine,
    invert_pixel_affine,
    stride_pixel_affine,
)

__all__ = [
    "ANCHOR_CANDIDATES_MAX",
    "BACKBONE_STRIDE",
    "CFA_PATTERNS",
    "DENSE_LATENT_CHANNELS",
    "FUSION_STRIDE",
    "INPUT_MODES",
    "INPUT_SCALES",
    "AnchorCandidates",
    "DenseMeasurement",
    "InputScale",
    "MeasurementFrame",
    "SampleGrid",
    "build_measurement_id",
    "input_mode_channels",
    "measurement_digest",
]

#: S1 固定输入模式；两种模式的 stem 通道数与坐标链分别实现、分别计分。
INPUT_MODES: tuple[str, ...] = ("rawvla_rgb3", "bayer_cfa")
CFA_PATTERNS: tuple[str, ...] = ("RGGB", "BGGR", "GRBG", "GBRG")

#: 测量场（融合网格）通道数与下采样倍率。
DENSE_LATENT_CHANNELS = 256
FUSION_STRIDE = 4
#: ConvNeXt-B 的四级 stage 步长（stem 4× + 三次 2× 下采样），输入必须 pad 到它的倍数。
BACKBONE_STRIDE = 32
#: 单帧候选锚点上限（S1 提示词冻结：单相机 ≤128，按 8×16 分区选点）。
ANCHOR_CANDIDATES_MAX = 128


def input_mode_channels(input_mode: str) -> int:
    """该输入模式送入骨干的通道数（``bayer_cfa`` 为四相位平面）。"""
    if input_mode == "rawvla_rgb3":
        return 3
    if input_mode == "bayer_cfa":
        return 4
    raise ValueError(f"未知输入模式 {input_mode!r}；允许 {INPUT_MODES}")


def sensor_channels(input_mode: str) -> int:
    """``MeasurementFrame.raw`` 的通道数：CFA 是单平面原始采样，不是四相位。"""
    if input_mode == "rawvla_rgb3":
        return 3
    if input_mode == "bayer_cfa":
        return 1
    raise ValueError(f"未知输入模式 {input_mode!r}；允许 {INPUT_MODES}")


@dataclass(frozen=True)
class InputScale:
    """一种数据来源的测量单位换算与 provenance。

    ``factor`` 把原始存储值乘到本阶段统一的归一化线性测量单位；方差按 ``factor**2``
    缩放（方差是平方量）。``train_set_scale`` 由训练集统计冻结，不允许逐帧归一化。
    """

    name: str
    factor: float
    unit: str
    provenance: str

    @property
    def variance_factor(self) -> float:
        return float(self.factor) ** 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "factor": self.factor,
            "variance_factor": self.variance_factor,
            "unit": self.unit,
            "provenance": self.provenance,
        }


#: 训练集尺度冻结表。``libero_uint8`` 按 S1 提示词显式 /255；RoboTwin 的 ``raw_linear``
#: 在 S0 采集时以 white level 1.0 记录（``hdr_raw_white_level=1.0``），受控渲染是
#: 平方根解码后的线性辐射代理，两者都已是 [0,1] 线性单位，因此 factor 为 1.0。
INPUT_SCALES: dict[str, InputScale] = {
    "libero_uint8": InputScale(
        name="libero_uint8",
        factor=1.0 / 255.0,
        unit="normalized_linear",
        provenance=(
            "S0 paired extension: online_rawvla_rgb3 uint8 (frozen author producer); "
            "explicit /255, no per-frame normalization"
        ),
    ),
    "robotwin_float_linear": InputScale(
        name="robotwin_float_linear",
        factor=1.0,
        unit="normalized_linear",
        provenance=(
            "S0 paired extension: renderer HDR-derived raw_linear recorded at "
            "hdr_raw_white_level=1.0 (n04_live_collection.py); float32 kept as measurement units"
        ),
    ),
    "rendered_factor_linear": InputScale(
        name="rendered_factor_linear",
        factor=1.0,
        unit="normalized_linear",
        provenance=(
            "S0 controlled rig: (rgb/255)**2.2 radiance proxy in [0,1] "
            "(n04_rendered_factors.py); explicitly synthetic pseudo-RAW"
        ),
    ),
}


def input_scale_for(name: str) -> InputScale:
    if name not in INPUT_SCALES:
        raise ValueError(f"未知输入尺度 {name!r}；允许 {tuple(INPUT_SCALES)}")
    return INPUT_SCALES[name]


def build_measurement_id(
    source_trajectory_id: str, frame_index: int, camera_id: str, input_mode: str
) -> str:
    """稳定 measurement ID：源轨迹 + 帧号 + 相机 + 输入模式。

    只由身份字段构成，与轨迹内容、模型输出无关；同一 (源, 帧, 相机, 模式) 重复导出时
    必须得到同一字符串，便于 S2 用它做“重复 measurement ID 不再次更新”的去重。
    """
    if not source_trajectory_id:
        raise ValueError("source_trajectory_id 不能为空")
    if "#" in source_trajectory_id:
        raise ValueError("source_trajectory_id 不得包含 '#'（候选 ID 分隔符）")
    if frame_index < 0:
        raise ValueError("frame_index 必须非负")
    if not camera_id or any(ch in camera_id for ch in "#/@"):
        raise ValueError("camera_id 不能为空且不含 '#', '/', '@'")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"未知输入模式 {input_mode!r}")
    return f"{source_trajectory_id}/{frame_index:06d}/{camera_id}@{input_mode}"


def measurement_digest(
    *, weights_sha256: str, config_sha256: str, input_mode: str, scale: str
) -> str:
    """测量缓存指纹：权重 / 配置 / 输入模式 / 单位尺度任一不同即为不同缓存。"""
    payload = json.dumps(
        {
            "weights_sha256": weights_sha256,
            "config_sha256": config_sha256,
            "input_mode": input_mode,
            "scale": scale,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MeasurementFrame:
    """一次当前帧测量的**全部**输入：图像、可选内参与传感器元数据。

    ``provenance`` 里的 episode / 时间戳 / 轨迹字段只用于记账与缓存命名，`measure` 的
    数值路径不得读取它们；测试会用“替换全部 provenance 后输出逐位不变”来验收这一点。
    """

    raw: torch.Tensor
    camera_id: str
    measurement_id: str
    input_mode: str
    scale: InputScale
    size_hw: tuple[int, int]
    K: torch.Tensor | None = None
    cfa_pattern: str | None = None
    time_s: float | None = None
    time_unit: str = "frame"
    saturation_level: float | None = None
    #: 坏点 / 缺失传感器像素：``[H,W]`` bool，True 表示该原图像素无效。物体被遮挡者不清零。
    invalid_mask: torch.Tensor | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"未知输入模式 {self.input_mode!r}；允许 {INPUT_MODES}")
        if self.raw.dim() != 3:
            raise ValueError(f"raw 需要 [C,H,W]，实际 {tuple(self.raw.shape)}")
        channels = sensor_channels(self.input_mode)
        if self.raw.shape[0] != channels:
            raise ValueError(
                f"{self.input_mode} 的传感器平面需要 {channels} 通道，实际 {self.raw.shape[0]}"
            )
        height, width = self.size_hw
        if (self.raw.shape[1], self.raw.shape[2]) != (height, width):
            raise ValueError(
                f"raw 空间尺寸 {tuple(self.raw.shape[1:])} 与 size_hw {self.size_hw} 不一致"
            )
        if min(height, width) <= 0:
            raise ValueError("size_hw 必须为正")
        if self.raw.dtype not in (torch.uint8, torch.float32, torch.float64):
            raise TypeError(f"raw dtype 只允许 uint8/float32/float64，实际 {self.raw.dtype}")
        if self.input_mode == "bayer_cfa":
            if self.cfa_pattern not in CFA_PATTERNS:
                raise ValueError(f"bayer_cfa 需要 cfa_pattern ∈ {CFA_PATTERNS}")
        elif self.cfa_pattern is not None:
            raise ValueError("rawvla_rgb3 不允许携带 cfa_pattern")
        if self.K is not None:
            if self.K.shape != (3, 3):
                raise ValueError(f"K 需要 [3,3]，实际 {tuple(self.K.shape)}")
            if not self.K.is_floating_point():
                raise TypeError("K 必须是浮点张量")

    @property
    def unit(self) -> PixelUnit:
        return "original_px"

    def move_to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """返回新 frame，只改张量设备/精度，不改身份字段。"""
        raw = self.raw
        if device is not None:
            raw = raw.to(device)
        if dtype is not None and raw.is_floating_point():
            raw = raw.to(dtype)
        K = None if self.K is None else (self.K.to(device=device) if device is not None else self.K)
        invalid_mask = self.invalid_mask
        if invalid_mask is not None and device is not None:
            invalid_mask = invalid_mask.to(device)
        return MeasurementFrame(
            raw=raw,
            camera_id=self.camera_id,
            measurement_id=self.measurement_id,
            input_mode=self.input_mode,
            scale=self.scale,
            size_hw=self.size_hw,
            K=K,
            cfa_pattern=self.cfa_pattern,
            time_s=self.time_s,
            time_unit=self.time_unit,
            saturation_level=self.saturation_level,
            invalid_mask=invalid_mask,
            provenance=dict(self.provenance),
        )


@dataclass(frozen=True)
class SampleGrid:
    """从原图传感器像素到测量场（融合网格）的完整坐标链。

    链：``sensor -> (CFA 相位, 可选) -> backbone 输入平面 -> pad 裁剪 -> stride 融合``。
    所有仿射都以 ``original_px`` 为单位表达；CFA 相位仿射把平面索引写成“传感器像素
    计数”，因此矩阵是 ``diag(2,2)`` 而不是 1。任何坐标换算都必须走本结构，禁止写死
    ``xy/stride``。

    CFA 的坐标帧约定：四相位是四个**不同**的采样子网格，不存在共享中心。融合网格的
    坐标以相位 0（δ=(0,0)）的子网格帧表达；其余相位的真实偏移由 `cfa_plane_names`
    与 `measurement_geometry.CFA_PLANE_OFFSETS` 给出，下游不得把它们当作同一中心。
    """

    sensor_size_hw: tuple[int, int]
    backbone_input_size_hw: tuple[int, int]
    feature_size_hw: tuple[int, int]
    stride: int
    pad_xy: tuple[float, float]
    sensor_to_backbone_input: PixelAffine
    backbone_input_to_feature: PixelAffine
    mode: str
    cfa_pattern: str | None = None
    cfa_plane_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride 必须为正")
        for size in (self.sensor_size_hw, self.backbone_input_size_hw, self.feature_size_hw):
            if min(size) <= 0:
                raise ValueError(f"网格尺寸必须为正：{size}")
        if self.sensor_to_backbone_input.from_unit != "original_px":
            raise ValueError("sensor_to_backbone_input 的源单位必须是 original_px")
        if self.backbone_input_to_feature.from_unit != "original_px":
            raise ValueError("backbone_input_to_feature 的源单位必须是 original_px")
        if self.backbone_input_to_feature.to_unit != "feature_px":
            raise ValueError("backbone_input_to_feature 的目标单位必须是 feature_px")

    @property
    def sensor_to_feature(self) -> PixelAffine:
        return compose_pixel_affine(self.backbone_input_to_feature, self.sensor_to_backbone_input)

    @property
    def feature_to_sensor(self) -> PixelAffine:
        return invert_pixel_affine(self.sensor_to_feature)

    def with_device(self, device: torch.device, dtype: torch.dtype = torch.float32) -> SampleGrid:
        """把坐标链的仿射放到指定设备 / 精度（网格是纯几何，可自由搬运）。

        测量网络在 GPU 上运行时，候选点的坐标换算也必须发生在同一设备；CPU 仿射与
        CUDA 点混用会被 `PixelAffine.apply` 直接拒绝。
        """

        def move(affine: PixelAffine) -> PixelAffine:
            return PixelAffine(
                matrix=affine.matrix.to(device=device, dtype=dtype),
                translation=affine.translation.to(device=device, dtype=dtype),
                from_unit=affine.from_unit,
                to_unit=affine.to_unit,
            )

        return SampleGrid(
            sensor_size_hw=self.sensor_size_hw,
            backbone_input_size_hw=self.backbone_input_size_hw,
            feature_size_hw=self.feature_size_hw,
            stride=self.stride,
            pad_xy=self.pad_xy,
            sensor_to_backbone_input=move(self.sensor_to_backbone_input),
            backbone_input_to_feature=move(self.backbone_input_to_feature),
            mode=self.mode,
            cfa_pattern=self.cfa_pattern,
            cfa_plane_names=self.cfa_plane_names,
        )

    def feature_centres(self) -> PixelPoints:
        """融合网格整数坐标 -> 原图传感器像素坐标 ``[Hf,Wf,2]``（与仿射同设备）。"""
        height, width = self.feature_size_hw
        reference = self.backbone_input_to_feature.matrix
        ys = torch.arange(height, dtype=reference.dtype, device=reference.device)
        xs = torch.arange(width, dtype=reference.dtype, device=reference.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points = PixelPoints(xy=torch.stack((grid_x, grid_y), dim=-1), unit="feature_px")
        return self.feature_to_sensor.apply(points)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensor_size_hw": list(self.sensor_size_hw),
            "backbone_input_size_hw": list(self.backbone_input_size_hw),
            "feature_size_hw": list(self.feature_size_hw),
            "stride": self.stride,
            "pad_xy": [float(v) for v in self.pad_xy],
            "sensor_to_backbone_input": self.sensor_to_backbone_input.as_dict(),
            "backbone_input_to_feature": self.backbone_input_to_feature.as_dict(),
            "mode": self.mode,
            "cfa_pattern": self.cfa_pattern,
            "cfa_plane_names": list(self.cfa_plane_names) if self.cfa_plane_names else None,
        }


def build_sample_grid(
    *,
    input_mode: str,
    sensor_size_hw: tuple[int, int],
    pad_xy: tuple[float, float],
    pad_hw: tuple[int, int],
    stride: int = FUSION_STRIDE,
    cfa_pattern: str | None = None,
) -> SampleGrid:
    """按实际 pad 与（可选）CFA 相位构造坐标链。

    ``pad_xy`` / ``pad_hw`` 是 **backbone 输入平面** 左侧/上方与总尺寸的 padding；
    CFA 模式下 backbone 平面是半分辨率的四相位平面，相位仿射负责原图 ↔ 平面换算：
    平面像素 ``(i,j)`` 的原图格点是 ``(2i+δy, 2j+δx)``，故其逆为
    ``p_plane = (p_sensor - δ) / 2``。
    """
    pad_left, pad_top = float(pad_xy[0]), float(pad_xy[1])
    pad_h, pad_w = int(pad_hw[0]), int(pad_hw[1])
    if stride <= 0:
        raise ValueError("stride 必须为正")
    if pad_h % stride or pad_w % stride:
        raise ValueError(f"backbone 输入尺寸 {pad_h}x{pad_w} 不是 stride {stride} 的倍数")
    if input_mode == "rawvla_rgb3":
        sensor_to_backbone = identity_pixel_affine()
        plane_names = None
    elif input_mode == "bayer_cfa":
        if cfa_pattern is None:
            raise ValueError("bayer_cfa 需要 cfa_pattern")
        offsets = cfa_plane_offsets(cfa_pattern)
        if len(offsets) != 4:
            raise AssertionError("CFA 必须有四个相位")
        dy, dx = int(offsets[0][1]), int(offsets[0][2])
        sensor_to_backbone = PixelAffine(
            matrix=torch.diag(torch.tensor([0.5, 0.5])),
            translation=torch.tensor([-0.5 * dx, -0.5 * dy]),
            from_unit="original_px",
            to_unit="original_px",
        )
        plane_names = tuple(name for name, _, _ in offsets)
    else:
        raise ValueError(f"未知输入模式 {input_mode!r}")

    backbone_input_to_feature = compose_pixel_affine(
        stride_pixel_affine(stride, from_unit="original_px", to_unit="feature_px"),
        crop_pixel_affine((pad_left, pad_top), unit="original_px"),
    )
    feature_h = pad_h // stride
    feature_w = pad_w // stride
    return SampleGrid(
        sensor_size_hw=(int(sensor_size_hw[0]), int(sensor_size_hw[1])),
        backbone_input_size_hw=(pad_h, pad_w),
        feature_size_hw=(feature_h, feature_w),
        stride=stride,
        pad_xy=(pad_left, pad_top),
        sensor_to_backbone_input=sensor_to_backbone,
        backbone_input_to_feature=backbone_input_to_feature,
        mode=input_mode,
        cfa_pattern=cfa_pattern,
        cfa_plane_names=plane_names,
    )


@dataclass(frozen=True)
class AnchorCandidates:
    """单帧候选锚点（≤128）。持久 ID 由 S2 关联层分配，本层只给本帧位置。"""

    xy_feature: torch.Tensor  # [B,N,2] 融合网格坐标
    xy_sensor: torch.Tensor  # [B,N,2] 原图传感器坐标（同帧位置，不跨帧复用）
    valid: torch.Tensor  # [B,N] bool
    confidence: torch.Tensor  # [B,N]
    partition_row: torch.Tensor  # [B,N] long
    partition_col: torch.Tensor  # [B,N] long
    descriptor: torch.Tensor  # [B,N,256]
    position_camera: torch.Tensor  # [B,N,3] 米；K 缺失时全 NaN 且 position_valid=False
    position_covariance: torch.Tensor  # [B,N,3,3] m^2；同上
    position_valid: torch.Tensor  # [B,N] bool
    anchors_per_row: int
    anchors_per_col: int

    @property
    def count(self) -> int:
        return int(self.xy_feature.shape[1])

    def candidate_measurement_ids(self, measurement_ids: tuple[str, ...]) -> list[list[str]]:
        if len(measurement_ids) != self.xy_feature.shape[0]:
            raise ValueError("measurement_ids 与 batch 不一致")
        return [
            [f"{mid}#cand{index:03d}" for index in range(self.count)] for mid in measurement_ids
        ]


@dataclass(frozen=True)
class DenseMeasurement:
    """单帧、单相机的完整测量输出（无历史）。

    ``feature_pyramid`` 是四级融合特征（每级 256 通道，供后续读出）；``latent_mean_map``
    是融合测量场；``depth_*`` 与候选协方差是几何量。``latent_variance_map`` 在 S1 是固定
    初值（未做创新校准），其状态写入 ``metadata["latent_variance_calibration"]``。
    """

    feature_pyramid: tuple[torch.Tensor, ...]
    latent_mean_map: torch.Tensor
    latent_variance_map: torch.Tensor
    depth_mean: torch.Tensor
    depth_variance: torch.Tensor
    valid: torch.Tensor
    support: torch.Tensor
    saturated: torch.Tensor
    sample_grid: SampleGrid
    candidates: AnchorCandidates
    measurement_ids: tuple[str, ...]
    units: dict[str, str]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        batch = self.latent_mean_map.shape[0]
        if len(self.measurement_ids) != batch:
            raise ValueError("measurement_ids 与 batch 不一致")
        for name in ("latent_variance_map", "depth_mean", "depth_variance", "valid", "support"):
            tensor = getattr(self, name)
            if tensor.shape[0] != batch:
                raise ValueError(f"{name} 的 batch 与 latent_mean_map 不一致")
        if self.latent_mean_map.shape[1] != DENSE_LATENT_CHANNELS:
            raise ValueError(
                f"latent_mean_map 通道数必须是 {DENSE_LATENT_CHANNELS}，"
                f"实际 {self.latent_mean_map.shape[1]}"
            )
        if self.candidates.count > ANCHOR_CANDIDATES_MAX:
            raise ValueError(f"候选锚点数 {self.candidates.count} 超过上限 {ANCHOR_CANDIDATES_MAX}")

    @property
    def batch_size(self) -> int:
        return int(self.latent_mean_map.shape[0])

    def camera_position_from_depth(self, K: torch.Tensor) -> torch.Tensor:
        """用预测深度把测量场每个格子反投影到相机系 ``[B,3,Hf,Wf]``（米）。"""
        if K.shape[-2:] != (3, 3):
            raise ValueError("K 需要 [...,3,3]")
        height, width = self.sample_grid.feature_size_hw
        centres = self.sample_grid.feature_centres().xy  # [Hf,Wf,2] sensor px
        xy = centres.to(self.depth_mean).view(1, height, width, 2)
        depth = self.depth_mean[:, 0]
        ones = torch.ones_like(xy[..., :1])
        homogeneous = torch.cat((xy, ones), dim=-1)  # [1,Hf,Wf,3]
        K_inv = torch.linalg.inv(K)
        rays = torch.einsum("...ij,...j->...i", K_inv, homogeneous)  # [B,Hf,Wf,3]
        return (rays * depth.unsqueeze(-1)).permute(0, 3, 1, 2)

    def to_cache(self, *, include_pyramid: bool = False) -> dict[str, Any]:
        """序列化为 numpy 缓存（GT 不在这里；缓存只含测量与其身份）。"""
        payload: dict[str, Any] = {
            "latent_mean_map": self.latent_mean_map.detach().float().cpu().numpy(),
            "latent_variance_map": self.latent_variance_map.detach().float().cpu().numpy(),
            "depth_mean": self.depth_mean.detach().float().cpu().numpy(),
            "depth_variance": self.depth_variance.detach().float().cpu().numpy(),
            "valid": self.valid.detach().cpu().numpy().astype("uint8"),
            "support": self.support.detach().cpu().numpy().astype("uint8"),
            "saturated": self.saturated.detach().cpu().numpy().astype("uint8"),
            "candidate_xy_sensor": self.candidates.xy_sensor.detach().float().cpu().numpy(),
            "candidate_valid": self.candidates.valid.detach().cpu().numpy().astype("uint8"),
            "candidate_confidence": self.candidates.confidence.detach().float().cpu().numpy(),
            "candidate_descriptor": self.candidates.descriptor.detach().float().cpu().numpy(),
            "candidate_position_camera": self.candidates.position_camera.detach()
            .float()
            .cpu()
            .numpy(),
            "candidate_position_covariance": self.candidates.position_covariance.detach()
            .float()
            .cpu()
            .numpy(),
            "candidate_position_valid": self.candidates.position_valid.detach()
            .cpu()
            .numpy()
            .astype("uint8"),
            "measurement_ids_json": json.dumps(list(self.measurement_ids)),
            "metadata_json": json.dumps(
                {"units": dict(self.units), **self.metadata}, sort_keys=True, default=str
            ),
            "sample_grid_json": json.dumps(self.sample_grid.as_dict(), sort_keys=True),
        }
        if include_pyramid:
            for index, level in enumerate(self.feature_pyramid):
                payload[f"feature_pyramid_{index}"] = level.detach().float().cpu().numpy()
        return payload


def sample_grid_from_dict(payload: dict[str, Any]) -> SampleGrid:
    def affine(entry: dict[str, Any]) -> PixelAffine:
        return PixelAffine(
            matrix=torch.tensor(entry["matrix"], dtype=torch.float32),
            translation=torch.tensor(entry["translation"], dtype=torch.float32),
            from_unit=entry["from_unit"],
            to_unit=entry["to_unit"],
        )

    return SampleGrid(
        sensor_size_hw=tuple(int(v) for v in payload["sensor_size_hw"]),
        backbone_input_size_hw=tuple(int(v) for v in payload["backbone_input_size_hw"]),
        feature_size_hw=tuple(int(v) for v in payload["feature_size_hw"]),
        stride=int(payload["stride"]),
        pad_xy=tuple(float(v) for v in payload["pad_xy"]),
        sensor_to_backbone_input=affine(payload["sensor_to_backbone_input"]),
        backbone_input_to_feature=affine(payload["backbone_input_to_feature"]),
        mode=payload["mode"],
        cfa_pattern=payload.get("cfa_pattern"),
        cfa_plane_names=(
            tuple(payload["cfa_plane_names"]) if payload.get("cfa_plane_names") else None
        ),
    )

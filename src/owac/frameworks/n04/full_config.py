"""S1 正式测量的配置对象与 JSON 装载。

配置是实验身份的一部分：``sha256`` 由装载后的规范 JSON 计算，训练、评测与缓存导出
都记录它。允许 CLI 覆盖的只有路径与运行身份字段（run-dir 等），结构与默认值不通过
命令行修改，避免同一份报告对应多个网络。

默认值来自[首版实现合同](../../../../../docs/prompts/n04_v1_0/implementation_contract.md)
与 S1 提示词：AdamW、默认 3e-4、weight decay 0.05、grad clip 1、warmup 5%、cosine；
S1 有效 batch 128 帧；正式 100k 步、上限 300k；种子 ``[17, 29, 43]``。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BackboneConfig",
    "DataConfig",
    "MeasurementConfig",
    "N04FullConfig",
    "TrainingConfig",
    "load_config",
]

MODEL_KIND = "n04_full_v1"
CHECKPOINT_SCHEMA = "n04-full-v1"
CONFIG_SCHEMA = "n04-full-s1-v1"


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "convnext_base"
    depths: tuple[int, ...] = (3, 3, 27, 3)
    dims: tuple[int, ...] = (128, 256, 512, 1024)
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6
    #: "random" 为主线；"imagenet1k" 必须作为独立初始化实验单独建账。
    init: str = "random"
    upstream_commit: str = "048efcea897d999aed302f2639b6270aedf8d4c8"
    upstream_sha256: str = "e081a71bcad826035cdac2c314bc26de93b0dcb95a70ce8b7668f3e61490b7d2"

    def __post_init__(self) -> None:
        if self.name != "convnext_base":
            raise ValueError("S1 冻结 convnext_base；其他规模必须另开实验")
        if tuple(self.depths) != (3, 3, 27, 3) or tuple(self.dims) != (128, 256, 512, 1024):
            raise ValueError("S1 冻结作者 Base 配置 depths=[3,3,27,3] dims=[128,256,512,1024]")
        if self.init not in ("random", "imagenet1k"):
            raise ValueError("backbone.init 只允许 random / imagenet1k")


@dataclass(frozen=True)
class MeasurementConfig:
    fusion_channels: int = 256
    fusion_stride: int = 4
    head_hidden: int = 64
    variance_floor: float = 1e-6
    depth_floor_m: float = 1e-3
    depth_ceiling_m: float = 20.0
    #: S1 未做创新校准：潜在方差是固定初值，状态写入缓存元数据。
    latent_variance_init: float = 1.0
    train_latent_variance: bool = False
    #: 方差头初始化（米）：初始预测 σ 取该值，避免 NLL 在未收敛方差上爆炸。
    variance_init_sigma_m: float = 0.25
    candidate_rows: int = 8
    candidate_cols: int = 16
    #: 位置协方差里“像素定位不确定性”的固定 1σ（原图像素），S1 不做估计。
    position_pixel_sigma_px: float = 0.5
    pad_multiple: int = 32
    #: Numerical boundary extension only; padded pixels always have support=False.
    padding_mode: str = "constant"

    def __post_init__(self) -> None:
        if self.padding_mode not in ("constant", "replicate"):
            raise ValueError("padding_mode must be constant or replicate")
        if self.fusion_channels != 256:
            raise ValueError("S1 冻结 256 通道测量场")
        if self.candidate_rows * self.candidate_cols > 128:
            raise ValueError("候选分区数不得超过 128")
        if self.variance_floor <= 0 or self.depth_floor_m <= 0:
            raise ValueError("variance_floor / depth_floor_m 必须为正")
        if self.train_latent_variance:
            raise ValueError("S1 不训练潜在方差；S2 在冻结 z 之后另行校准")


@dataclass(frozen=True)
class DataConfig:
    paired_root: str
    factor_root: str
    input_mode: str = "rawvla_rgb3"
    splits: tuple[str, ...] = ("train",)
    #: 训练集冻结的深度尺度（米）：用于把物理量标准化后求 loss。
    depth_scale_m: float = 0.8
    #: 对应集来源的输入尺度名，逐平台在装载时选择。
    scale_names: dict[str, str] = field(
        default_factory=lambda: {
            "libero": "libero_uint8",
            "robotwin2": "robotwin_float_linear",
            "rendered_factor": "rendered_factor_linear",
        }
    )
    #: 每源最多使用的帧数（0 = 全部）；S1 用满 24 / 12 帧。
    max_frames_per_sequence: int = 0
    correspondence_pairs: dict[str, tuple[tuple[str, str], ...]] = field(
        default_factory=lambda: {
            "libero": (("agentview", "robot0_eye_in_hand"),),
            "robotwin2": (("front_camera", "left_camera"), ("head_camera", "right_camera")),
            "rendered_factor": (("camera_0", "camera_1"),),
        }
    )

    def __post_init__(self) -> None:
        if self.input_mode not in ("rawvla_rgb3", "bayer_cfa"):
            raise ValueError(f"未知输入模式 {self.input_mode!r}")
        if self.depth_scale_m <= 0:
            raise ValueError("depth_scale_m 必须为正")


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    batch_frames: int = 128
    #: 0 表示不切分；否则为梯度累积的微批帧数（必须整除 batch_frames）。
    micro_batch_frames: int = 0
    eval_batch_frames: int = 64
    steps: int = 100_000
    max_steps: int = 300_000
    lr: float = 3e-4
    lr_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    lr_grid_steps: int = 2_000
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: str = "bfloat16"
    eval_every: int = 5_000
    log_every: int = 50
    checkpoint_every: int = 5_000
    profile_steps: int = 500
    #: 损失项（S1 提示词冻结权重）。
    w_geometry: float = 1.0
    w_geometry_nll: float = 0.1
    w_correspondence: float = 0.1
    huber_beta_norm: float = 0.05
    nll_variance_floor_norm: float = 1e-4
    #: 单像素 r²/σ² 的上限：数值防护，防止未收敛方差主导梯度（不影响覆盖率指标）。
    nll_ratio_clamp: float = 100.0
    correspondence_temperature: float = 0.1
    correspondence_samples: int = 128
    correspondence_depth_tol_m: float = 0.05
    #: 趋稳判据：连续 eval 次数中主指标相对改善 < 1%。
    stability_window: int = 5
    stability_relative_threshold: float = 0.01
    num_workers: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.batch_frames < 1:
            raise ValueError("batch_frames 必须为正")
        if self.micro_batch_frames:
            if self.micro_batch_frames % 2 or self.batch_frames % self.micro_batch_frames:
                raise ValueError("micro_batch_frames 必须为偶数且整除 batch_frames")
        if self.eval_batch_frames % 2:
            raise ValueError("eval_batch_frames 必须为偶数（成对视图）")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction 必须在 [0,1)")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds 不得重复")
        for value in self.lr_grid:
            if value <= 0:
                raise ValueError("lr_grid 必须为正")


@dataclass(frozen=True)
class N04FullConfig:
    schema: str
    stage: str
    model_kind: str
    checkpoint_schema: str
    backbone: BackboneConfig
    measurement: MeasurementConfig
    data: DataConfig
    training: TrainingConfig
    name: str = "s1_measurement"

    @property
    def input_mode(self) -> str:
        return self.data.input_mode

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _build(section: type, payload: dict[str, Any], name: str):
    known = {f for f in section.__dataclass_fields__}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{name} 配置含未知字段 {sorted(unknown)}")
    converted: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ("depths", "dims"):
            value = tuple(int(v) for v in value)
        elif key in ("seeds", "splits", "lr_grid"):
            value = tuple(value)
        elif key == "betas":
            value = tuple(float(v) for v in value)
        elif key == "correspondence_pairs":
            value = {name: tuple(tuple(pair) for pair in pairs) for name, pairs in value.items()}
        converted[key] = value
    return section(**converted)


def load_config(path: str | Path) -> N04FullConfig:
    """从 JSON 装载并校验；未知字段直接报错，避免拼写错误被静默忽略。"""
    payload = json.loads(Path(path).read_text())
    known = {f for f in N04FullConfig.__dataclass_fields__}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"配置文件含未知顶层字段 {sorted(unknown)}")
    backbone = _build(BackboneConfig, payload.get("backbone", {}), "backbone")
    measurement = _build(MeasurementConfig, payload.get("measurement", {}), "measurement")
    data = _build(DataConfig, payload["data"], "data")
    training = _build(TrainingConfig, payload.get("training", {}), "training")
    config = N04FullConfig(
        schema=payload.get("schema", CONFIG_SCHEMA),
        stage=payload.get("stage", "S1"),
        model_kind=payload.get("model_kind", MODEL_KIND),
        checkpoint_schema=payload.get("checkpoint_schema", CHECKPOINT_SCHEMA),
        backbone=backbone,
        measurement=measurement,
        data=data,
        training=training,
        name=payload.get("name", "s1_measurement"),
    )
    if config.model_kind != MODEL_KIND or config.checkpoint_schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            "S1 只允许 model_kind=n04_full_v1 / checkpoint_schema=n04-full-v1；"
            "旧原型 checkpoint 不得静默装入"
        )
    return config

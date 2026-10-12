"""N04 配置：运行模式、传播算子选择、状态规模与损失权重。

本模块刻意只依赖标准库，使 `import owac.frameworks.n04.config` 不引入 torch，
核心契约可以在无深度学习依赖的环境中检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

InputMode = Literal["rawvla_rgb3", "bayer_cfa"]
PredictorMode = Literal["identity", "proprio_only", "learned_proprio", "calibrated_geometry"]
#: `learned_proprio` 是默认可部署路线：RawVLA NPZ 提供本体/动作但没有相机位姿与深度。
#: `calibrated_geometry` 只在相机内参、外参、深度/平面假设实际可得时启用；
#: 合成 oracle 用同一开关但单独报告，只作上限诊断，不与无深度模型混报。
DEFAULT_PREDICTOR: PredictorMode = "learned_proprio"

CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


@dataclass(frozen=True)
class N04GeometryConfig:
    """坐标与采样约定。所有约定在此集中固定，任何模块不得自行改用别的约定。"""

    #: 特征网格尺寸 (H, W)；`grid_sample` 的采样坐标定义在该网格上。
    feature_size: tuple[int, int] = (32, 32)
    #: `torch.nn.functional.grid_sample` 的 align_corners。默认 False：
    #: 归一化坐标 -1 对应像素中心 0，+1 对应像素中心 W-1。
    align_corners: bool = False
    #: `grid_sample` 的 padding_mode。越界处必须为 "zeros"：用 border 复制会向前一
    #: 特征场边界外“借”未观测值，等于伪造历史证据。validity 另由 mask 显式给出。
    padding_mode: str = "zeros"
    #: 特征网格相对原始测量的降采样倍率。原始测量像素 (u, v) 对应特征坐标
    #: (v / stride, u / stride)。stride=1 时两者同一坐标。
    pixel_stride: float = 4.0
    #: 平移/仿射参数的单位。`feature` 表示直接作用在特征网格像素坐标上；
    #: `pixel` 表示作用在原始测量坐标上，由调用方在 warp 前换算。
    flow_unit: Literal["feature", "pixel"] = "feature"
    #: 位移幅度限制（特征像素）。用于零位移初始化后不让首步跳变。
    max_flow: float = 8.0

    def __post_init__(self) -> None:
        if self.align_corners:
            raise ValueError("N04 固定 align_corners=False，见 geometry.py 的坐标约定")
        if self.padding_mode != "zeros":
            raise ValueError("N04 禁止 border padding：越界必须产生 validity=0")
        if self.pixel_stride <= 0:
            raise ValueError("pixel_stride 必须为正")
        if len(self.feature_size) != 2 or min(self.feature_size) <= 1:
            raise ValueError("feature_size 必须是 >=2 的 (H, W)")


@dataclass(frozen=True)
class N04ModelConfig:
    """编码器结构与损失权重。"""

    name: str = "n04-tiny"
    #: 特征场宽度（每层通道数）。
    width: int = 32
    #: 共享的传播/修正 block 数；核心规则可组合，不写成固定层数的单体。
    blocks: int = 2
    #: 多尺度层数。1 表示只用最高分辨率尺度。
    scales: int = 2
    #: 每个尺度的相对降采样倍率（相对 feature_size）。
    scale_strides: tuple[int, ...] = (1, 2)
    #: coarse backward flow 的分辨率（以特征网格为单位再降采样）。
    coarse_stride: int = 4
    #: 变化/更新分支的隐藏宽度。
    hidden: int = 64
    camera_embedding: int = 8
    #: 传播可靠性 logits 的温度下限，避免除零。
    sigma_floor: float = 0.05
    #: 是否启用照明增益分支（全局/低频增益，只在可靠区域估计）。
    illumination_branch: bool = True
    #: 是否启用独立 change cell（外部变化特征）。
    change_branch: bool = True
    #: 读出 token 数上限（每相机）。
    readout_tokens: int = 256
    #: 读出 token 宽度。真实 π0.5 前缀为 2048；小尺寸配置单独命名，不冒充真实后端。
    readout_dim: int = 128
    #: 传播算子。
    predictor: PredictorMode = DEFAULT_PREDICTOR
    #: learned_proprio 是否读取低分辨率上一特征。False = 纯本体对照（proprio_only）。
    use_previous_features: bool = True
    #: 运动上下文向量维度（本体 delta + 已执行动作 + dt）。``None`` 表示由第一次
    #: ``step`` 的实际上下文字段推断，避免把平台维度写死在配置里；一旦确定，
    #: 后续维度不一致会显式报错。
    motion_dim: int | None = None

    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "location": 1.0,
            "future_measurement": 0.5,
            "change": 0.1,
            "ego_warp": 0.1,
        }
    )

    def scale_strides_for(self, scales: int | None = None) -> tuple[int, ...]:
        n = self.scales if scales is None else scales
        if n <= 0:
            raise ValueError("scales 必须为正")
        #: 几何级数 1,2,4,... 不预设固定层数。
        return tuple(2**i for i in range(n))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "blocks": self.blocks,
            "scales": self.scales,
            "scale_strides": list(self.scale_strides),
            "coarse_stride": self.coarse_stride,
            "hidden": self.hidden,
            "camera_embedding": self.camera_embedding,
            "sigma_floor": self.sigma_floor,
            "illumination_branch": self.illumination_branch,
            "change_branch": self.change_branch,
            "readout_tokens": self.readout_tokens,
            "readout_dim": self.readout_dim,
            "predictor": self.predictor,
            "use_previous_features": self.use_previous_features,
            "motion_dim": self.motion_dim,
            "loss_weights": dict(self.loss_weights),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> N04ModelConfig:
        known = {f for f in cls.__dataclass_fields__}
        extra = set(data) - known
        if extra:
            raise ValueError(f"未知的模型配置字段: {sorted(extra)}")
        payload = dict(data)
        if "scale_strides" in payload and payload["scale_strides"] is not None:
            payload["scale_strides"] = tuple(int(s) for s in payload["scale_strides"])
        obj = cls(**payload)
        obj.scale_strides_for()
        return obj

    def without(self, **changes: Any) -> N04ModelConfig:
        """派生消融配置（无 ego、无变化分支、无照明分支等）。"""
        return replace(self, **changes)


@dataclass(frozen=True)
class N04Config:
    """一次运行的完整配置。"""

    input_mode: InputMode = "rawvla_rgb3"
    #: 原始测量网格尺寸 (H, W)。
    measurement_size: tuple[int, int] = (128, 128)
    #: 通道数：rawvla_rgb3 为 3；bayer_cfa 为 1（单测量面）。
    channels: int = 3
    cfa_pattern: str = "RGGB"
    #: CFA 采样相位（行、列偏移），用于记录每平面实际采样坐标。
    cfa_phase: tuple[int, int] = (0, 0)
    camera_ids: tuple[str, ...] = ("head",)
    geometry: N04GeometryConfig = field(default_factory=N04GeometryConfig)
    model: N04ModelConfig = field(default_factory=N04ModelConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cfa_pattern not in CFA_PATTERNS:
            raise ValueError(f"未知 CFA 排列 {self.cfa_pattern!r}")
        h, w = self.measurement_size
        if min(h, w) <= 0:
            raise ValueError("measurement_size 必须为正")
        if self.input_mode == "bayer_cfa":
            if self.channels != 1:
                raise ValueError("bayer_cfa 输入为单测量面，channels 必须为 1")
            fh, fw = self.geometry.feature_size
            if (h, w) != (fh, fw):
                raise ValueError("bayer_cfa 首版要求测量网格与特征网格同尺寸")
        elif self.channels != 3:
            raise ValueError("rawvla_rgb3 输入为三通道")
        if not self.camera_ids:
            raise ValueError("至少需要声明一个相机")

    def with_model(self, **changes: Any) -> N04Config:
        return replace(self, model=self.model.without(**changes))


@dataclass(frozen=True)
class N04TrainConfig:
    """训练与评测的起跑参数。这些是起跑值，不是已调优协议。"""

    sequence_length: int = 12
    tbptt: int = 8
    batch_size: int = 8
    steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    burn_in: int = 2
    #: 传播模块是否冻结（只训 decomposition/update），用于判断分解是否只是依赖更好的对应。
    freeze_propagation: bool = False
    #: 未来测量监督是否开启（只在 loss 处读取，绝不进入 predict 输入）。
    future_measurement_loss: bool = True
    cfa: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    log_every: int = 20
    checkpoint_every: int = 100
    seed: int = 0
    synthetic: dict[str, Any] = field(default_factory=dict)
    output: str = "outputs/n04"

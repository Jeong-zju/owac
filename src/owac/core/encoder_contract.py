"""编码器互操作契约 v0（无第三方依赖）。

这是 N01/N02/N04 三条独立实现共享的**最小类型约定**，不规定候选内部状态：内部表示可以
是状态场、假设集合或几何状态。字段语义优先于具体 dataclass 写法。

设计约束：

* 本模块**不导入 torch/numpy**，也不初始化模型或设备。核心 ``import owac`` 在任何环境
  下都保持轻量；需要张量的实现模块自行定义张量类型。
* ``MeasurementFrame`` 描述**单相机**一帧的测量。三通道 RawVLA 数据不得改名为 Bayer；
  ``bayer_cfa`` 是每像素一个滤色测量的独立模式。
* 缺失信息用显式 availability/mask 表达，不用零值冒充。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CameraSpec",
    "ComputeBudget",
    "EmbodimentContext",
    "EncoderOutput",
    "MeasurementFrame",
    "StateT",
    "VisualTokens",
]

StateT = Any
TensorLike = Any


@dataclass
class MeasurementFrame:
    """单相机、单帧的测量与其元数据。

    ``values`` 形状 ``[B, C, H, W]``，数值单位由 ``metadata["value_range"]`` 记录：
    LIBERO uint8 与 RoboTwin float32 的单位不同，禁止无条件按 255 归一化或按帧最大值
    归一化。``sample_xy`` 用于 CFA：记录每个通道的采样坐标/偏移，不同分辨率不混放。
    """

    values: TensorLike
    #: 有效测量掩码（缺失/未观测）。
    valid_mask: TensorLike | None = None
    #: 坏点掩码。
    bad_mask: TensorLike | None = None
    #: 饱和掩码。饱和是高读数，不等于高置信度。
    saturated_mask: TensorLike | None = None
    #: CFA 每通道的采样坐标/偏移；非 CFA 输入为 None。
    sample_xy: TensorLike | None = None
    #: 身份与时间。
    episode_ids: Sequence[Any] = ()
    frame_ids: Sequence[Any] = ()
    camera_id: str = ""
    #: 真实时间戳；缺失时 ``time_unit="frame"``，不编造秒数。
    timestamp: TensorLike | None = None
    time_unit: str = "frame"
    #: 这一帧是否是新测量。开头 padding 的重复帧标 False，避免同一测量被重复计分。
    new_measurement: bool | Sequence[bool] = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: 哪些元数据字段真实可得。
    metadata_available: Mapping[str, bool] = field(default_factory=dict)

    def availability(self) -> dict[str, bool]:
        return {
            "valid_mask": self.valid_mask is not None,
            "bad_mask": self.bad_mask is not None,
            "saturated_mask": self.saturated_mask is not None,
            "sample_xy": self.sample_xy is not None,
            "timestamp": self.timestamp is not None and self.time_unit != "frame",
            **dict(self.metadata_available),
        }


@dataclass
class EmbodimentContext:
    """截至当前时刻**已经实际发生**的本体与运动上下文。

    缺失信息用 ``*_available=False`` 表达；不得用未来专家动作或隐藏真值补齐。
    """

    proprio: TensorLike | None = None
    proprio_available: bool = False
    #: **已执行**的动作，不是未来监督动作。
    executed_action: TensorLike | None = None
    executed_action_available: bool = False
    #: 相机变换（例如 ``T_world_from_camera``）。
    camera_transform: TensorLike | None = None
    camera_transform_available: bool = False
    task_tokens: TensorLike | None = None
    task_mask: TensorLike | None = None
    #: 时间语义。
    time_unit: str = "frame"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ComputeBudget:
    """确定性预算。``deadline_ms`` 是可选实测软预算，不是硬实时保证。"""

    max_updates: int = 1
    max_history: int = 1
    deadline_ms: float | None = None


@dataclass
class EncoderOutput:
    """一个编码 step 的输出。

    ``representation`` 可以是状态场、假设集合或几何状态；``validity`` 表达哪些位置有
    测量/历史支持；``diagnostics`` 用于记录分解统计等可检查量；``cost`` 记录本步成本。
    """

    representation: Any
    state: StateT
    validity: TensorLike | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class VisualTokens:
    """π0.5 视觉前缀形状的 token。

    ``values`` 形状 ``[B, L, D]``（首个受控配置 ``D=2048``、每相机 ``L=256``）；
    ``mask`` 形状 ``[B, L]``，False 表示该位置没有支持，必须进入 prefix attention mask。
    """

    values: TensorLike
    mask: TensorLike
    camera_ids: Sequence[str] = ()
    positions: TensorLike | None = None

    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)


@dataclass
class CameraSpec:
    """一个相机的测量规格。"""

    camera_id: str
    height: int
    width: int
    channels: int = 3
    input_mode: str = "rawvla_rgb3"
    cfa_pattern: str | None = None
    pixel_stride: float = 1.0

    def __post_init__(self) -> None:
        if self.input_mode == "bayer_cfa":
            if self.channels != 1:
                raise ValueError("bayer_cfa 每像素一个滤色测量，channels 必须为 1")
            if not self.cfa_pattern:
                raise ValueError("bayer_cfa 必须声明 CFA 排列")
        elif self.input_mode == "rawvla_rgb3":
            if self.channels != 3:
                raise ValueError("rawvla_rgb3 为三通道测量")
        else:
            raise ValueError(f"未知 input_mode {self.input_mode!r}")


@runtime_checkable
class StreamingEncoder(Protocol):
    """逐帧编码器契约。训练与在线必须调用**同一个** ``step``。"""

    def init_state(self, batch_size: int, camera_spec: Any, device: Any) -> StateT: ...

    def step(
        self,
        frame: MeasurementFrame,
        context: EmbodimentContext,
        state: StateT,
        budget: ComputeBudget,
    ) -> EncoderOutput: ...

    def reset(self, state: StateT, batch_mask: Any) -> StateT: ...

    def detach_state(self, state: StateT) -> StateT: ...

    def readout(self, output: EncoderOutput, token_budget: int) -> VisualTokens: ...

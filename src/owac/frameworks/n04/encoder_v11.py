"""N04 v1.1 总调度编码器：当前 RAW 测量 -> 质量 R -> 支持关联 -> 条件预测 -> 双模式更新 -> 读出。

因果与调用契约（``docs/n04_v11_s2_interfaces.md`` §1/§4/§5/§6）
------------------------------------------------------------------------

* ``reset(batch_or_episode)``；``step(current_raw, proprio_context, new_measurement=True)``
  返回 `StepOutput`（``prefix_tokens`` + ``debug`` + 携带状态）。第三个位置参数是**携带状态**
  （`EncoderState`，首次 ``None``）：既有特征桥 ``RawVLAFeaturePolicy.encode_observation`` 就是按
  ``step(values, context, state, **kwargs)`` 调用的，因此这样同时满足契约与桥。
* 一次新观测只融合一次：同一 ``frame_key`` 的重复 ``step`` 会累加
  ``duplicate_frame_count`` 并默认抛 `RepeatedFrameError`（``allow_duplicate_step=True`` 时返回
  上一帧输出而不重复融合）；首帧由 ``StateEstimator.reset`` + 一次 ``evidence_update.update``
  吸收，不会对同一帧再跑 predict。
* 内部 ``StateEstimator.predict`` 只吃上一 posterior 与截至当前时刻的本体；当前 RAW／当前特征／
  当前 flow 只进关联、质量和读出。``reset`` 后不跨 episode 继承。
* 四臂（§6）：``D`` = RAW + 完整 N04；``C`` = RAW 单帧（同骨干／同读出，无状态／无对应）；
  ``A`` = 同源固定 ISP RGB 单帧；``B`` = 同源 RGB 的完整 N04。A/B 在公共输入边界
  将同源 RAW 转为 RGB，按同一三通道容器传入（质量头因此只读 RGB）。``scale_name`` 必须是
  ``contracts.INPUT_SCALES`` 里的键。配置里的 ``input_mode`` 接受契约写法 ``rawvla_raw3``／
  ``rawvla_rgb3``，统一归一化为 S1 的三通道容器名 ``rawvla_rgb3``；RAW 与 RGB 的区别由
  ``ARM_SPECS[arm].input_domain`` 表达，不靠容器名。
* 批量：``step`` 是单帧在线接口，batch 维固定为 1（与 10 次去噪复用同一 prefix 的执行方式
  一致）；一次调用可传多个相机（一个 ``MeasurementFrame``／张量列表，要求同尺寸），
  ``prefix_tokens`` 按相机顺序拼接。

损失与下游接口
----------------

* ``StepOutput.losses["correspondence"]``：本步上一帧/当前帧 RAW 与刚算出的 flow 交给 WS-A
  ``RaftCorrespondenceLoss`` 得到的标量（真实对应自监督，梯度回 RAFT 的 fnet／cnet／
  update_block）。没有帧对（首帧）或该臂不用对应（A/C）时**该键缺席**，不用 0 冒充。
* ``StepOutput.forecast_inputs[camera]``：t 时刻的支持先验（``StateEstimator.step`` 之后、
  本帧测量校正之前）——``mode_weights [B,N,2]``、``mode_mean [B,N,2,2*obs]``、``mode_cov``
  (upper/lower/side 各 ``[B,N,2,obs]``) 与 ``xy [B,N,2]``、``ids [B,N]``、``fresh [B,N]``；
  WS-D 用未来帧 stop-grad 特征构造 target 后调用 ``evidence_update.forecast_loss``。
  A/C 臂没有状态，该 dict 为空。

模块分工（契约 §0）
------------------------

WS-A ``correspondence.py``（``RaftCorrespondence``／``RawScale`` 与坐标、flow 采样）与 WS-B
``quality.py``／``state_estimator.py``／``evidence_update.py``（``QualityHead``、
``StateEstimator``、``evidence_update.update``）都是**本模块直接调用的真实实现**，本模块只负责
按顺序编排。唯一保留的测试替身是 `StandinCorrespondence`（零／固定 flow，用于单测注入已知位移
并省掉 RAFT 前向），它返回与 WS-A 相同的 `FlowOutput`。
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import (
    DENSE_LATENT_CHANNELS,
    FUSION_STRIDE,
    MeasurementFrame,
    SampleGrid,
    build_measurement_id,
    build_sample_grid,
    input_scale_for,
)
from .correspondence import (
    FlowOutput,
    RaftCorrespondence,
    RaftCorrespondenceLoss,
    RaftLossConfig,
    RawScale,
    bidirectional_correspondence,
    flow_at_points,
    grid_to_px,
    px_to_grid,
)
from .evidence_update import update as evidence_update
from .full_config import (
    BackboneConfig,
    DataConfig,
    MeasurementConfig,
    N04FullConfig,
    TrainingConfig,
)
from .measurement import N04FullMeasurement, build_measurement, preprocess_frames
from .measurement_geometry import PixelPoints, in_bounds_xy
from .quality import QualityHead
from .readout import N04V11Readout, ReadoutConfig
from .state_estimator import (
    StateEstimator,
    SupportState,
    covering_grid,
    covering_radius_px,
)

__all__ = [
    "ARM_SPECS",
    "EncoderState",
    "EncoderV11Config",
    "N04V11Encoder",
    "RepeatedFrameError",
    "StandinCorrespondence",
    "StepOutput",
    "TokenReadout",
    "build_encoder",
    "platform_defaults",
    "flow_at_points",
    "grid_to_px",
    "px_to_grid",
]


@contextlib.contextmanager
def _fp32_region(device: torch.device | None = None):
    """坐标与统计的 FP32 区：不依赖调用方的 autocast 设置（cuda 与 cpu 都关掉）。

    ``PixelAffine`` 的 compose／invert 是 matmul，autocast 会把它们变成 BF16；随后与 FP32 的
    ``points.xy`` 相遇，``_require_same_dtype_device`` 直接报错（远端 LIBERO smoke 的现场）。
    契约 §6 要求「网络 BF16，RAW 统计／协方差／NLL FP32」，因此 flow 采样、支持 xy 推进、
    特征采样、质量头与双模式更新都在这个区域里算；测量骨干与读出仍走调用方的 autocast。
    """
    kinds: list[str] = []
    if (device is None or device.type == "cuda") and torch.cuda.is_available():
        kinds.append("cuda")
    if device is None or device.type == "cpu":
        kinds.append("cpu")
    with contextlib.ExitStack() as stack:
        for kind in kinds:
            stack.enter_context(torch.autocast(device_type=kind, enabled=False))
        yield


# --------------------------------------------------------------------------------------
# 四臂与配置
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """一个消融臂的输入域与是否启用状态／对应。"""

    input_domain: str
    use_state: bool
    use_correspondence: bool


#: 所有臂接收同源 RAW；A/B 在 _as_frames 内统一经过固定 ISP，含训练/在线/forecast。
ARM_SPECS: dict[str, ArmSpec] = {
    "D": ArmSpec("raw3", use_state=True, use_correspondence=True),
    "C": ArmSpec("raw3", use_state=False, use_correspondence=False),
    "B": ArmSpec("rgb3", use_state=True, use_correspondence=True),
    "A": ArmSpec("rgb3", use_state=False, use_correspondence=False),
}


def default_measurement_config() -> N04FullConfig:
    """默认测量配置（结构与 S1 相同；真实训练由配置入口显式构造）。"""
    return N04FullConfig(
        schema="n04-v1.1-encoder-v11",
        stage="s2",
        model_kind="n04_full_v1",
        checkpoint_schema="n04-full-v1",
        backbone=BackboneConfig(),
        measurement=MeasurementConfig(),
        data=DataConfig(paired_root=".", factor_root="."),
        training=TrainingConfig(),
        name="s2_encoder_v11",
    )


#: v1.1 契约 §6 的输入域写法 -> S1 三通道容器名（``MeasurementFrame.input_mode``）。
#: 两个域都是 3 通道，RAW／RGB 的区别由 ``arm`` 的 ``input_domain`` 表达，映射无损；
#: 训练 JSON 里写的是 ``rawvla_raw3``（D/C）或 ``rawvla_rgb3``（A/B）。
_INPUT_MODE_ALIASES: dict[str, str] = {
    "rawvla_raw3": "rawvla_rgb3",
    "raw3": "rawvla_rgb3",
}
_INPUT_CONTAINERS: tuple[str, ...] = ("rawvla_rgb3", "bayer_cfa")

#: 训练 JSON（``configs/n04/v1_1/*.json``）里的字段名 -> 本 dataclass 字段名。名字不同但语义
#: 唯一，映射后仍按同一份配置生效；冲突时直接报错，不静默二选一。
_JSON_FIELD_ALIASES: dict[str, str] = {"raft_iterations": "raft_iters"}
_JSON_READOUT_ALIASES: dict[str, str] = {
    "readout_blocks": "depth",
    "tokens_per_camera": "num_latents",
    "token_dim": "prefix_dim",
}
#: 必须与契约常量一致的 JSON 键：给了就要对上，避免"配置写了却不生效"。
_JSON_CONTRACT_CHECKS: dict[str, Any] = {
    "observation_dim": DENSE_LATENT_CHANNELS,
    "state_dim": 2 * DENSE_LATENT_CHANNELS,
    "fusion_channels": DENSE_LATENT_CHANNELS,
    "fusion_stride": FUSION_STRIDE,
    "raft_small": False,
}

#: 平台 -> 契约尺度名（``contracts.INPUT_SCALES``）。相机集合与缩放语义的唯一来源是 WS-D 的
#: ``raw/n04_v11_data.py:PLATFORM_SPECS``；这里只做平台事实到尺度名的映射与一致性校验。
_PLATFORM_SCALE_NAMES: dict[str, str] = {
    "libero": "libero_uint8",
    "robotwin": "robotwin_float_linear",
}


def platform_defaults(platform: str) -> tuple[tuple[str, ...], str]:
    """平台事实：``(相机集合, 尺度名)``。

    取自 WS-D 的 ``raw/n04_v11_data.py:PLATFORM_SPECS``，并校验**输入原始域**（``value_scale``：
    ``uint8_0_255`` → factor 1/255、``float32_linear`` → factor 1.0）与契约 ``INPUT_SCALES`` 的
    尺度名一致；不一致或域未登记都直接报错（平台固定缩放只在编码器 ``InputScale`` 一处施加）。
    """
    from ...raw.n04_v11_data import PLATFORM_SPECS

    if platform not in PLATFORM_SPECS:
        raise ValueError(f"未知平台 {platform!r}；支持 {sorted(PLATFORM_SPECS)}")
    spec = PLATFORM_SPECS[platform]
    if platform not in _PLATFORM_SCALE_NAMES:
        raise ValueError(f"平台 {platform!r} 没有登记的契约尺度名")
    # value_scale 现在记录**输入原始域**（WS-D 2026-10-15 改）：uint8_0_255 / float32_linear。
    # 平台固定缩放只在编码器 InputScale 一处施加（契约 §3），故这里校验"输入域 ↔ 契约尺度"一致。
    domain_expectations = {"uint8_0_255": 1.0 / 255.0, "float32_linear": 1.0}
    if spec.value_scale not in domain_expectations:
        raise ValueError(f"平台 {platform!r} 的 value_scale {spec.value_scale!r} 未登记")
    scale_name = _PLATFORM_SCALE_NAMES[platform]
    expected = domain_expectations[spec.value_scale]
    factor = float(input_scale_for(scale_name).factor)
    if abs(factor - expected) > 1e-12:
        raise ValueError(
            f"平台 {platform!r} 的输入域 {spec.value_scale} 与契约尺度 {scale_name!r}"
            f"（factor={factor}）不一致"
        )
    return tuple(spec.camera_ids), scale_name


@dataclass
class EncoderV11Config:
    """总调度编码器的配置（数值口径与结构由 WS-A/WS-B 模块自己持有）。"""

    arm: str = "D"
    cameras: tuple[str, ...] = ("camera0", "camera1")
    #: S1 ``MeasurementFrame`` 的输入模式（三通道 RAW 与同源 RGB 都用这个容器）。
    input_mode: str = "rawvla_rgb3"
    #: 平台固定缩放名，必须在 ``contracts.INPUT_SCALES`` 内（A/B 臂的选择属数据入口）。
    scale_name: str = "libero_uint8"
    #: 饱和电平；``None`` 时由 ``preprocess_frames`` 用默认 1.0。
    #: 饱和电平（测量单位）；``None`` 时由 preprocess_frames 用默认 1.0。
    #: 同时透传对应损失 RawScale；A/B 固定 ISP 后在 RGB 的 1.0 白电平判断饱和。
    saturation_level: float | None = None
    in_channels: int = 3
    supports_per_camera: int = 128
    obs_dim: int = DENSE_LATENT_CHANNELS
    proprio_dim: int = 16
    control_hidden: int = 128
    trajectory_len: int = 8
    collect_readout_attention: bool = True
    #: 重复观测的默认行为；True 时返回上一帧输出而不重复融合（仍然计数）。
    allow_duplicate_step: bool = False
    #: 出生方差 / 修订模式额外过程噪声初值（透传给 WS-B 估计器）。
    state_var_init: float = 1.0
    transition_init: str = "author"
    measurement_padding_mode: str | None = None
    observation_normalization: str = "none"
    transition_norm_limit: float | None = None
    correspondence_boundary_weight: float = 0.0
    revision_variance_init: float = 0.1
    #: RAFT 迭代与可选固定降分辨率（显存不足时才用，flow 会按比例放大回原生分辨率）。
    raft_iters: int = 12
    raft_resize_hw: tuple[int, int] | None = None
    readout: ReadoutConfig | None = None

    def __post_init__(self) -> None:
        # 契约把输入域写成 rawvla_raw3 / rawvla_rgb3；S1 的帧容器只有 3 通道与 CFA 两种，
        # 这里显式归一化，避免 WS-D 的 JSON 名称在构造时被静默丢弃或后置报错。
        self.input_mode = _INPUT_MODE_ALIASES.get(self.input_mode, self.input_mode)
        if self.input_mode not in _INPUT_CONTAINERS:
            raise ValueError(
                f"未知输入容器 {self.input_mode!r}；允许 {_INPUT_CONTAINERS} 或别名 "
                f"{tuple(_INPUT_MODE_ALIASES)}"
            )
        if self.arm not in ARM_SPECS:
            raise ValueError(f"未知消融臂 {self.arm!r}；允许 {tuple(ARM_SPECS)}")
        if not self.cameras:
            raise ValueError("cameras 不能为空")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras 不得重复（状态按相机身份索引）")
        if self.supports_per_camera <= 0:
            raise ValueError("supports_per_camera 必须为正")
        if self.obs_dim != DENSE_LATENT_CHANNELS:
            raise ValueError("S1 冻结 256 通道测量场；obs_dim 必须是 256")
        if self.raft_iters < 1:
            raise ValueError("raft_iters 必须为正")
        if self.observation_normalization not in ("none", "rms"):
            raise ValueError("observation_normalization must be none or rms")
        if (
            not math.isfinite(self.correspondence_boundary_weight)
            or self.correspondence_boundary_weight < 0
        ):
            raise ValueError("correspondence_boundary_weight must be finite and nonnegative")
        input_scale_for(self.scale_name)

    @property
    def spec(self) -> ArmSpec:
        return ARM_SPECS[self.arm]

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def state_dim(self) -> int:
        """作者 RKN 状态维：可观测半 + 隐藏半，与 ``obs_dim`` 对应。"""
        return 2 * self.obs_dim

    def resolve_readout(self) -> ReadoutConfig:
        base = self.readout or ReadoutConfig()
        return replace(
            base,
            num_cameras=self.num_cameras,
            latent_channels=self.obs_dim,
            state_dim=self.state_dim,
            collect_attention=self.collect_readout_attention,
        )


# --------------------------------------------------------------------------------------
# 输出结构
# --------------------------------------------------------------------------------------


class TokenReadout(NamedTuple):
    """与 S1 ``model.TokenReadout`` 同域（``values`` / ``mask``），供特征桥读取。"""

    values: torch.Tensor
    mask: torch.Tensor


class RepeatedFrameError(RuntimeError):
    """同一 ``frame_key`` 被 step 了两次：显式报错，不静默重复融合。"""


@dataclass
class EncoderState:
    """携带状态：每个相机一份 WS-B `SupportState`（契约 §4 的 128 支持与双模式分量）。"""

    per_camera: dict[str, SupportState]

    def cameras(self) -> tuple[str, ...]:
        return tuple(self.per_camera)

    def detach(self) -> EncoderState:
        """TBPTT 边界：切断图，保留值。"""
        return EncoderState(
            per_camera={
                camera: replace(
                    state,
                    mode_mean=state.mode_mean.detach(),
                    mode_cov=tuple(block.detach() for block in state.mode_cov),
                    mode_weights=state.mode_weights.detach(),
                )
                for camera, state in self.per_camera.items()
            }
        )


@dataclass
class StepOutput:
    """一次 ``step`` 的输出。``state`` 由调用方携带并在下一次调用传回。

    ``losses`` 目前只有 ``correspondence``（上一帧/当前帧 RAW 的真实对应自监督，梯度回到
    RAFT）：没有上一真实帧（首帧）或该臂不用对应时该键缺席——缺席表示"未产生"，不用 0 冒充。
    ``forecast_inputs`` 是 t 时刻各支持的模式先验（``StateEstimator.step`` 之后、本帧测量
    校正之前）与 ``xy``/``ids``/``fresh``，供 WS-D 用未来帧 stop-grad 特征构造 target 后调用
    ``evidence_update.forecast_loss``；不接收状态的臂（A/C）为空 dict。
    """

    prefix_tokens: torch.Tensor
    state: EncoderState | None
    debug: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    forecast_inputs: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    updated: bool = True
    frame_key: Any = None
    cache_hit: bool = False
    target_features: dict[str, dict[str, Any]] = field(default_factory=dict)


class StandinCorrespondence(nn.Module):
    """单测替身：flow 恒为 0 或注入固定 flow／可见掩码，省掉 RAFT 前向（真实实现是 WS-A）。"""

    def __init__(
        self,
        flow_px: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("_fixed_flow", None if flow_px is None else flow_px.detach())
        #: 可选的逐像素可见掩码（True = 可信）；None 表示全可见。
        fixed_valid = None if valid is None else torch.as_tensor(valid).to(torch.bool).detach()
        self.register_buffer("_fixed_valid", fixed_valid)

    def forward(self, previous_raw: torch.Tensor, current_raw: torch.Tensor) -> FlowOutput:
        batch, _, height, width = current_raw.shape
        if self._fixed_flow is None:
            flow = torch.zeros(
                batch, 2, height, width, dtype=torch.float32, device=current_raw.device
            )
        else:
            flow = self._fixed_flow.to(current_raw.device, torch.float32)
            if tuple(flow.shape[-2:]) != (height, width):
                raise ValueError("替身 flow 与当前帧尺寸不一致")
            flow = flow.expand(batch, -1, -1, -1)
        if self._fixed_valid is None:
            valid = torch.ones(batch, height, width, dtype=torch.bool, device=current_raw.device)
        else:
            mask = self._fixed_valid.to(current_raw.device)
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask[:, 0]
            if tuple(mask.shape[-2:]) != (height, width):
                raise ValueError("替身可见掩码与当前帧尺寸不一致")
            valid = mask.reshape(1, height, width).expand(batch, -1, -1)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=flow.device),
            torch.arange(width, device=flow.device),
            indexing="ij",
        )
        target_xy = torch.stack((xx, yy), -1).float().reshape(1, -1, 2).expand(batch, -1, -1)
        inverse = -flow.permute(0, 2, 3, 1).reshape(batch, -1, 2)
        for _ in range(16):
            inverse = -flow_at_points(flow, target_xy + inverse)
        backward = inverse.reshape(batch, height, width, 2).permute(0, 3, 1, 2)
        return FlowOutput(flow_px=flow, valid=valid, feature_scale=1.0, backward_flow_px=backward)


# --------------------------------------------------------------------------------------
# 总调度编码器
# --------------------------------------------------------------------------------------


class N04V11Encoder(nn.Module):
    """N04 v1.1 单帧在线编码器（每个相机独立状态，读出后按相机顺序拼接）。"""

    def __init__(
        self,
        measurement: N04FullMeasurement | None = None,
        config: EncoderV11Config | None = None,
        *,
        measurement_config: N04FullConfig | None = None,
        quality_head: nn.Module | None = None,
        correspondence: nn.Module | None = None,
        correspondence_loss: nn.Module | None = None,
        state_stack: nn.Module | None = None,
        readout: N04V11Readout | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EncoderV11Config()
        self.spec = self.config.spec
        if measurement is None:
            measurement = build_measurement(measurement_config or default_measurement_config())
        if measurement.config.input_mode != self.config.input_mode:
            raise ValueError(
                f"测量网络输入模式 {measurement.config.input_mode!r} 与编码器 "
                f"{self.config.input_mode!r} 不一致"
            )
        self.measurement = measurement
        if self.config.measurement_padding_mode is not None:
            measurement.config = replace(
                measurement.config,
                measurement=replace(
                    measurement.config.measurement,
                    padding_mode=self.config.measurement_padding_mode,
                ),
            )
        self.quality_head = quality_head or QualityHead(
            obs_dim=self.config.obs_dim, initial_variance=self.config.state_var_init
        )
        self.raw_scale = RawScale.from_input_scale(
            input_scale_for(self.config.scale_name),
            saturation_level=(
                1.0 if self.config.arm in ("A", "B") else (self.config.saturation_level or 1.0)
            ),
        )
        self.correspondence = correspondence or RaftCorrespondence(
            scale=self.raw_scale,
            in_channels=self.config.in_channels,
            iters=self.config.raft_iters,
            resize_hw=self.config.raft_resize_hw,
        )
        self.correspondence_loss = correspondence_loss or RaftCorrespondenceLoss(
            scale=self.raw_scale,
            config=RaftLossConfig(boundary=self.config.correspondence_boundary_weight),
        )
        self.state_stack = state_stack or StateEstimator(
            num_supports=self.config.supports_per_camera,
            obs_dim=self.config.obs_dim,
            proprio_dim=self.config.proprio_dim,
            control_hidden=self.config.control_hidden,
            revision_variance_init=self.config.revision_variance_init,
            initial_variance=self.config.state_var_init,
            transition_init=self.config.transition_init,
            transition_norm_limit=self.config.transition_norm_limit,
        )
        self.readout = readout or N04V11Readout(self.config.resolve_readout())
        # 诊断计数（reset 不清计数；携带状态在 reset 时清空）。
        self.step_index = 0
        self.measurement_count = 0
        self.state_update_count = 0
        self.duplicate_frame_count = 0
        self._episode_id = "unset"
        self._prev_frames: dict[str, MeasurementFrame | None] = {}
        self._trajectory: dict[str, deque[torch.Tensor]] = {}
        self._last_key: Any = None
        self._last_output: StepOutput | None = None

    # -- 生命周期 ------------------------------------------------------------------

    def reset(self, batch_or_episode: Any = None, batch_mask: torch.Tensor | None = None) -> None:
        """episode／窗口边界：清空携带状态与历史帧，不跨 episode 继承。"""
        if batch_mask is not None and not bool(torch.as_tensor(batch_mask).all()):
            raise ValueError("v1.1 step 是单帧在线接口；不支持部分 batch 的 reset")
        if batch_or_episode is not None:
            self._episode_id = str(batch_or_episode)
        self.step_index = 0
        self._prev_frames = {}
        self._trajectory = {}
        self._last_key = None
        self._last_output = None

    # -- 只读状态查询（测试与可视化用；不参与数值路径） ------------------------------

    @property
    def prev_frame_cameras(self) -> tuple[str, ...]:
        """已保存上一真实帧的相机（``reset`` 后为空）。"""
        return tuple(sorted(self._prev_frames))

    @property
    def support_trajectories(self) -> dict[str, list[torch.Tensor]]:
        """每相机的支持点历史（最多 ``trajectory_len`` 步，已 detach）。"""
        return {camera: list(history) for camera, history in self._trajectory.items()}

    # -- 输入适配 ------------------------------------------------------------------

    def _as_frames(
        self,
        current_raw: Any,
        camera_id: Any,
        measurement_valid: Any,
        saturation: Any,
    ) -> tuple[list[MeasurementFrame], tuple[str, ...]]:
        items = list(current_raw) if isinstance(current_raw, Sequence) else [current_raw]
        if not items:
            raise ValueError("step 至少需要一个相机帧")
        ids = (
            list(camera_id)
            if isinstance(camera_id, Sequence) and not isinstance(camera_id, str)
            else [camera_id] * len(items)
        )
        if len(ids) != len(items):
            raise ValueError("camera_id 数量与帧数量不一致")
        valid_list = (
            list(measurement_valid)
            if isinstance(measurement_valid, Sequence)
            and not isinstance(measurement_valid, str)
            and len(measurement_valid) == len(items)
            else [measurement_valid] * len(items)
        )
        frames = []
        for index, item in enumerate(items):
            cam = ids[index] or f"camera{index}"
            if isinstance(item, MeasurementFrame):
                if ids[index] is not None and item.camera_id != ids[index]:
                    raise ValueError("MeasurementFrame.camera_id 与 camera_id 参数不一致")
                frames.append(item)
                continue
            # 注意：torch.Tensor 自带 ``.values()`` 方法，必须先判类型再取 ``.values`` 字段。
            source = item if torch.is_tensor(item) else getattr(item, "values", item)
            values = torch.as_tensor(source)
            if values.dim() != 3:
                raise ValueError(
                    "step 是单帧在线接口：需要 [C,H,W]（批量请在调用方循环），"
                    f"实际 {tuple(values.shape)}"
                )
            if values.is_floating_point() and values.dtype not in (torch.float32, torch.float64):
                values = values.to(torch.float32)
            height, width = int(values.shape[-2]), int(values.shape[-1])
            raw_valid = valid_list[index]
            if raw_valid is None:
                bad = getattr(item, "bad_mask", None)
                if bad is not None:
                    raw_valid = ~torch.as_tensor(bad).to(torch.bool)
            invalid = None
            if raw_valid is not None:
                valid = torch.as_tensor(raw_valid).to(torch.bool)
                invalid = (~valid).reshape(height, width) if valid.dim() > 2 else ~valid
            frames.append(
                MeasurementFrame(
                    raw=values,
                    camera_id=cam,
                    measurement_id=build_measurement_id(
                        self._episode_id, self.step_index, cam, self.config.input_mode
                    ),
                    input_mode=self.config.input_mode,
                    scale=input_scale_for(self.config.scale_name),
                    size_hw=(height, width),
                    K=None,
                    cfa_pattern=None,
                    time_s=None,
                    time_unit="frame",
                    saturation_level=self.config.saturation_level,
                    invalid_mask=invalid,
                    provenance={"source": "N04V11Encoder.step", "arm": self.config.arm},
                )
            )
        if self.config.arm in ("A", "B"):
            from ...raw.n04_v11_data import fixed_isp_rgb

            frames = [
                replace(
                    frame,
                    raw=fixed_isp_rgb(frame.raw, scale_factor=float(frame.scale.factor)),
                    saturation_level=1.0,
                    provenance={**frame.provenance, "isp": "fixed_clip_srgb_v1"},
                )
                for frame in frames
            ]
        return frames, tuple(frame.camera_id for frame in frames)

    def _merge_saturation(
        self, batch: Any, saturation: Any, frames: Sequence[MeasurementFrame]
    ) -> torch.Tensor:
        """把调用方的逐像素饱和标记并到骨干输入平面的 ``batch.saturated``。"""
        if saturation is None:
            return batch.saturated
        masks = (
            list(saturation)
            if isinstance(saturation, Sequence) and not isinstance(saturation, str)
            else [saturation] * len(frames)
        )
        planes = []
        for index, frame in enumerate(frames):
            plane = batch.saturated[index]
            mask = masks[index] if index < len(masks) else None
            if mask is None:
                planes.append(plane)
                continue
            if frame.input_mode != "rawvla_rgb3":
                raise ValueError("CFA 相位平面不支持逐像素饱和掩码注入")
            value = torch.as_tensor(mask).to(torch.bool)
            while value.dim() > 2:
                value = value.any(dim=0)
            pad_h = batch.saturated.shape[-2] - value.shape[-2]
            pad_w = batch.saturated.shape[-1] - value.shape[-1]
            if pad_h < 0 or pad_w < 0:
                raise ValueError("饱和掩码比骨干输入平面还大")
            value = F.pad(value.unsqueeze(0), (0, pad_w, 0, pad_h)).to(plane.device)
            planes.append(plane | value)
        return torch.stack(planes, dim=0)

    def _proprio_vector(
        self, proprio_context: Any, batch: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate body context and return availability; absent inputs disable learned control."""
        value: Any = None
        if proprio_context is not None:
            if torch.is_tensor(proprio_context):
                value = proprio_context
            elif isinstance(proprio_context, Mapping):
                for key in ("proprio_past", "proprio", "vector", "current"):
                    if proprio_context.get(key) is not None:
                        value = proprio_context[key]
                        break
            else:
                for name in ("proprio_past", "proprio", "vector"):
                    candidate = getattr(proprio_context, name, None)
                    if candidate is not None:
                        value = candidate
                        break
        if value is None:
            return (
                torch.zeros(batch, self.config.proprio_dim, device=device, dtype=dtype),
                torch.zeros(batch, dtype=torch.bool, device=device),
            )
        vector = torch.as_tensor(value, device=device, dtype=dtype)
        vector = vector.reshape(batch, -1)
        if vector.shape[-1] != self.config.proprio_dim or not bool(torch.isfinite(vector).all()):
            raise ValueError(
                f"proprio must be finite [B,{self.config.proprio_dim}]; got {tuple(vector.shape)}"
            )
        if vector.shape[0] != batch:
            vector = vector.expand(batch, -1)
        return vector, torch.ones(batch, dtype=torch.bool, device=device)

    # -- 一步 ---------------------------------------------------------------------

    def step(
        self,
        current_raw: Any,
        proprio_context: Any = None,
        state: EncoderState | None = None,
        *,
        new_measurement: bool = True,
        camera_id: Any = None,
        frame_key: Any = None,
        measurement_valid: Any = None,
        saturation: Any = None,
        collect_attention: bool | None = None,
        compute_auxiliary: bool | None = None,
    ) -> StepOutput:
        """逐帧调用：当前 RAW（或帧对象）-> 前缀 token + debug + 携带状态。"""
        frames, camera_ids = self._as_frames(current_raw, camera_id, measurement_valid, saturation)
        compute_auxiliary = (
            self.training and torch.is_grad_enabled()
            if compute_auxiliary is None
            else bool(compute_auxiliary)
        )
        # RGB 臂的质量/对应只读 ISP 后的帧和由它产生的饱和标记，不能旁路读取 RAW mask。
        if self.config.arm in ("A", "B"):
            saturation = None
        key = (camera_ids, frame_key) if frame_key is not None else None
        if key is not None and self._last_key is not None and key == self._last_key:
            self.duplicate_frame_count += 1
            if not self.config.allow_duplicate_step:
                raise RepeatedFrameError(
                    f"frame_key {frame_key!r} 在一帧内被 step 了两次；一次新观测只融合一次，"
                    "重复读取请复用该输出，或显式设置 allow_duplicate_step=True"
                )
            assert self._last_output is not None
            self._last_output.cache_hit = True
            return self._last_output
        if state is not None and tuple(state.cameras()) != tuple(camera_ids):
            raise ValueError(
                f"携带状态的相机 {tuple(state.cameras())} 与本次调用 {tuple(camera_ids)} 不一致；"
                "换相机集合时必须 reset"
            )

        # 1. 当前帧测量：完全复用 S1 的预处理 / 四级骨干 / 多尺度融合。
        batch = preprocess_frames(frames, self.measurement.config)
        saturated = self._merge_saturation(batch, saturation, frames)
        levels = [level.float() for level in self.measurement.backbone_levels(batch.values)]
        latent = self.normalize_observation(self.measurement.fuse(levels))
        self.measurement_count += 1
        batch_size, device = 1, latent.device
        with _fp32_region(device):  # 坐标链（含 affine compose）恒 FP32
            grids = [
                build_sample_grid(
                    input_mode=frame.input_mode,
                    sensor_size_hw=frame.size_hw,
                    pad_xy=batch.pad_xy,
                    pad_hw=batch.pad_hw,
                    stride=self.measurement.config.measurement.fusion_stride,
                    cfa_pattern=frame.cfa_pattern,
                ).with_device(device)
                for frame in frames
            ]
        use_state = bool(self.spec.use_state)
        carried = dict(state.per_camera) if state is not None else {}

        debug: dict[str, Any] = {
            "arm": self.config.arm,
            "step_index": self.step_index,
            "camera_ids": list(camera_ids),
            "new_measurement": bool(new_measurement),
            "flow": {},
            "quality": {},
            "evidence": {},
            "support": {},
            "mode_weights": {},
            "readout_attention": {},
            "flow_valid": {},
            "correspondence_terms": {},
        }
        tokens: list[torch.Tensor] = []
        pair_losses: list[torch.Tensor] = []
        forecast_inputs: dict[str, dict[str, torch.Tensor]] = {}
        birth_counts: list[int] = []
        id_maxima: list[int] = []
        live_ages: list[torch.Tensor] = []
        target_features: dict[str, dict[str, Any]] = {}
        for index, frame in enumerate(frames):
            cam = frame.camera_id
            slice_latent = latent[index : index + 1]
            grid = grids[index]
            # 本体条件是 FP32 预测／统计区的输入，不用网络 dtype。
            proprio, proprio_valid = self._proprio_vector(
                proprio_context, batch_size, device, torch.float32
            )
            # 2. 对应：previous real RAW -> current RAW（flow 是总图像运动，无相机/物体标签）。
            previous = self._prev_frames.get(cam)
            if self.spec.use_correspondence and previous is not None:
                flow_output = bidirectional_correspondence(
                    self.correspondence, previous.raw[None], frame.raw[None]
                )
                flow_px = flow_output.flow_px.to(device=device, dtype=torch.float32)
                occlusion = None if flow_output.valid is None else ~flow_output.valid.to(torch.bool)
            else:
                height, width = frame.size_hw
                flow_px = torch.zeros(
                    batch_size, 2, height, width, device=device, dtype=torch.float32
                )
                occlusion = None

            # 3. 支持位置：首帧覆盖网格出生；之后 predict + flow 推进 + 新区域补种。
            prior_weights: torch.Tensor | None = None
            prior_state: SupportState | None = None
            current = carried.get(cam)
            spawned = current is None
            births = self.config.supports_per_camera if spawned else 0
            if use_state:
                if spawned or current is None:
                    current = self.state_stack.reset(
                        batch_size=batch_size, sensor_size_hw=frame.size_hw, device=device
                    )
                else:
                    with _fp32_region(device):
                        ids_before_birth = current.ids.clone()
                        current = self.state_stack.step(
                            current,
                            proprio,
                            flow_px,
                            occlusion=occlusion,
                            proprio_valid=proprio_valid,
                        )
                        current = self._seed_new_area(current, frame, flow_px)
                        births = int((current.ids != ids_before_birth).sum())
                    prior_weights = current.mode_weights
                prior_state = current
            xy = current.xy if current is not None else self._spawn_positions(frame, device)
            # 2b. 真实对应自监督：上一帧/当前帧 RAW + 刚算出的 flow -> WS-A 损失（梯度回 RAFT）。
            if previous is not None and self.spec.use_correspondence and compute_auxiliary:
                with _fp32_region(device):
                    pair_valid = None
                    if previous.invalid_mask is not None or frame.invalid_mask is not None:
                        pair_valid = torch.ones(1, *frame.size_hw, dtype=torch.bool, device=device)
                        for image in (previous, frame):
                            if image.invalid_mask is not None:
                                pair_valid = pair_valid & ~image.invalid_mask[None].to(device)
                    pair = self.correspondence_loss(
                        previous.raw[None],
                        frame.raw[None],
                        flow_px,
                        backward_flow_px=flow_output.backward_flow_px,
                        valid=pair_valid,
                    )
                selfsup = pair.total.new_zeros(())
                if pair.support > 0 and isinstance(self.correspondence, RaftCorrespondence):
                    selfsup = self.correspondence_loss.self_supervised(
                        self.correspondence,
                        previous.raw[None],
                        frame.raw[None],
                        flow_px,
                        flow_output.backward_flow_px,
                        valid=pair_valid,
                    )
                if pair.support > 0 or (
                    self.config.correspondence_boundary_weight > 0 and pair.source_support > 0
                ):
                    pair_losses.append(pair.total + selfsup)
                debug["correspondence_terms"][cam] = {
                    "matching": pair.total.detach(),
                    "selfsup": selfsup.detach(),
                    "visible_fraction": pair.support,
                    "boundary": pair.boundary.detach() if pair.boundary is not None else None,
                }

            # 4. 在关联位置采样当前特征与质量 R（R 不读历史 residual）。
            with _fp32_region(device):
                features = self._sample_features(slice_latent, xy, grid)
                r_diag = self.quality_head(
                    values=batch.values[index : index + 1],
                    support=batch.support[index : index + 1],
                    saturated=saturated[index : index + 1],
                    latent=slice_latent,
                    xy_px=xy,
                    grid=grid,
                )
            expected = (batch_size, self.config.supports_per_camera, self.config.obs_dim)
            if tuple(r_diag.shape) != expected:
                raise ValueError(
                    f"质量头的 R_diag 形状必须是 {expected}，实际 {tuple(r_diag.shape)}"
                )
            if not bool(torch.isfinite(r_diag).all()) or bool((r_diag <= 0).any()):
                raise ValueError("质量头返回了非正或非有限的方差")

            # 5. 双模式更新：当前测量只进入状态一次（fresh 支持由 WS-B 用 y 初始化）。
            evidence = None
            if use_state and current is not None and new_measurement:
                with _fp32_region(device):
                    support_xy = grid.sensor_to_backbone_input.apply(
                        PixelPoints(xy=xy.float(), unit="original_px")
                    ).xy
                    sampled_support = (
                        F.grid_sample(
                            batch.support[index : index + 1].float(),
                            px_to_grid(support_xy, batch.support.shape[-2:]).unsqueeze(1),
                            align_corners=False,
                            padding_mode="zeros",
                        )
                        .squeeze(2)
                        .squeeze(1)
                    )
                    valid_features = sampled_support > 0.999
                    evidence = evidence_update(
                        self.state_stack, current, features, r_diag, valid=valid_features
                    )
                current = evidence.state
                self.state_update_count += 1
            posterior_weights = None if current is None else current.mode_weights

            # 6. 读出：当前稠密特征直通 + 两个记忆分量/权重 + 位置 + 年龄。
            readout_kwargs: dict[str, Any] = {}
            if current is not None:
                readout_kwargs = dict(
                    support_state=current.mode_mean,
                    mode_weights=current.mode_weights,
                    xy_px=current.xy,
                    age=current.age,
                    size_hw=frame.size_hw,
                    support_valid=current.valid & ~current.fresh,
                )
            token, attention, counts = self.readout(
                latent_map=slice_latent,
                camera_index=index,
                dense_valid=F.adaptive_avg_pool2d(
                    batch.support[index : index + 1].float(), slice_latent.shape[-2:]
                )[:, 0]
                > 0.999,
                return_attention=collect_attention,
                **readout_kwargs,
            )
            tokens.append(token)

            debug["flow"][cam] = flow_px.detach()
            if previous is not None and self.spec.use_correspondence:
                debug["flow_valid"][cam] = flow_output.valid.detach()
            debug["quality"][cam] = {
                "R_diag": r_diag.detach(),
                "min": float(r_diag.min().detach()),
                "mean": float(r_diag.mean().detach()),
                "max": float(r_diag.max().detach()),
            }
            if evidence is not None:
                debug["evidence"][cam] = {
                    "residual": evidence.residual.detach(),
                    "innovation_variance": evidence.innovation_variance.detach(),
                    "log_likelihood": evidence.log_likelihood.detach(),
                    "log_weight": evidence.log_weight.detach(),
                    "obs_valid": evidence.obs_valid.detach(),
                }
            debug["support"][cam] = {
                "xy_px": xy.detach(),
                "valid": None if current is None else current.valid.detach(),
                "fresh": None if current is None else current.fresh.detach(),
                "ids": None if current is None else current.ids.detach(),
                "age": (
                    torch.zeros(batch_size, device=device)
                    if current is None
                    else current.age.detach()
                ),
                "feature_size_hw": tuple(int(v) for v in slice_latent.shape[-2:]),
                "media_tokens": counts,
                "trajectory": [item.detach() for item in self._trajectory.get(cam, [])]
                + [xy.detach()],
            }
            if prior_state is not None:
                forecast_inputs[cam] = {
                    "mode_weights": prior_state.mode_weights,
                    "mode_mean": prior_state.mode_mean,
                    "mode_cov": prior_state.mode_cov,
                    "xy": prior_state.xy,
                    "ids": prior_state.ids,
                    "fresh": prior_state.fresh,
                    "R_diag": r_diag,
                    "proprio_valid": proprio_valid,
                }
            target_features[cam] = {
                "latent": slice_latent.detach(),
                "values": batch.values[index : index + 1].detach(),
                "support": batch.support[index : index + 1].detach(),
                "saturated": saturated[index : index + 1].detach(),
                "pad_xy": tuple(float(v) for v in batch.pad_xy),
                "pad_hw": tuple(batch.values.shape[-2:]),
                "size_hw": tuple(frame.size_hw),
            }
            if use_state:
                debug["mode_weights"][cam] = {
                    "posterior": (
                        None if posterior_weights is None else posterior_weights.detach()
                    ),
                    "prior": None if prior_weights is None else prior_weights.detach(),
                    "updated": bool(evidence is not None),
                }
            if attention is not None:
                debug["readout_attention"][cam] = attention.detach()
            self._remember_frame(cam, frame)
            self._remember_trajectory(cam, xy.detach())
            if current is not None:
                carried[cam] = current
                birth_counts.append(births)  # 本帧该相机出生的支持数
                id_maxima.append(int(current.ids.max()))
                live_ages.append(current.age[current.valid])

        if birth_counts:
            # 覆盖判据的可观测指标（契约 §4）：median_age 用**可见支持**的年龄中位数，
            # 短寿命支持才会把它压低；若真实 dev 数据上 median_age < 5 帧，按裁定改 (c)。
            debug["births_per_frame"] = int(sum(birth_counts))
            debug["ids_max"] = int(max(id_maxima))
            stacked = [age for age in live_ages if age.numel() > 0]
            if stacked:
                debug["median_age"] = float(torch.median(torch.cat(stacked).to(torch.float32)))
            else:
                debug["median_age"] = float(
                    torch.median(
                        torch.cat([state.age.flatten() for state in carried.values()]).to(
                            torch.float32
                        )
                    )
                )
        else:  # A/C 臂没有状态：未产生，不用 0 冒充
            debug["births_per_frame"] = None
            debug["ids_max"] = None
            debug["median_age"] = None

        prefix = tokens[0] if len(tokens) == 1 else torch.cat(tokens, dim=1)
        debug["prefix_shape"] = tuple(int(v) for v in prefix.shape)
        debug["num_visual_tokens"] = int(prefix.shape[1])
        losses: dict[str, torch.Tensor] = {}
        if pair_losses:
            losses["correspondence"] = torch.stack(pair_losses).mean()
        output = StepOutput(
            prefix_tokens=prefix,
            state=EncoderState(per_camera=carried) if use_state else None,
            debug=debug,
            losses=losses,
            forecast_inputs=forecast_inputs,
            updated=bool(new_measurement),
            frame_key=key,
            target_features=target_features,
        )
        self.step_index += 1
        self._last_key = key
        self._last_output = output
        return output

    def normalize_observation(self, latent: torch.Tensor) -> torch.Tensor:
        """Fix the latent scale shared by measurements, state and stopped targets.

        Channel RMS=1 is a non-affine parameterization of the latent observation,
        not an image transform or a bound imposed on the forecast residual/loss.
        """
        if self.config.observation_normalization == "none":
            return latent
        if self.config.observation_normalization != "rms":
            raise ValueError("observation_normalization must be none or rms")
        values = latent.float()
        return F.normalize(values, dim=1, eps=1e-4) * math.sqrt(values.shape[1])

    # -- 读出接口（特征桥调用） ------------------------------------------------------

    def readout_tokens(self, output: StepOutput, token_budget: int | None = None) -> TokenReadout:
        """把前缀交给特征桥；``token_budget`` 是**每相机**预算（桥传 256）。"""
        values = output.prefix_tokens
        if token_budget is not None:
            per_camera_total = token_budget * self.config.num_cameras
            if values.shape[1] != per_camera_total and token_budget < values.shape[1]:
                values = values[:, :token_budget]
            elif values.shape[1] % token_budget:
                raise ValueError("前缀 token 数不是每相机预算的整数倍")
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        return TokenReadout(values=values, mask=mask)

    # -- 内部工具 ------------------------------------------------------------------

    def _spawn_positions(self, frame: MeasurementFrame, device: torch.device) -> torch.Tensor:
        """无状态臂（C/A）的位置：同一套覆盖网格（读出只用当前稠密特征直通）。"""
        grid = covering_grid(self.config.supports_per_camera, frame.size_hw)
        return grid.unsqueeze(0).to(device=device, dtype=torch.float32)

    def _seed_new_area(
        self,
        state: SupportState,
        frame: MeasurementFrame,
        flow_px: torch.Tensor,
    ) -> SupportState:
        """新区域出生：候选=覆盖网格里「本帧可见、且下一帧仍留在画幅内」的空位。

        WS-B ``StateEstimator.birth`` 只填**越界**空槽（``vacant = ~in_bounds``，遮挡不产生空槽），
        并在候选里挑与所有存活支持距离 > ``radius_px`` 的位置。候选规则是本模块的职责：若把整张
        覆盖网格无脑递上去，刚被推出画幅的支持会在**同一位置**立刻被重新生下、下一帧又被推出，
        形成 ID 抖动（本地实测：常量 flow 6px/帧时每帧 ~25 次出生、其中 ~90% 是上一帧同一点）。
        这里用当前 flow 采样一次（契约 §2 源点约定）要求候选下一步仍在画幅内，出生点因此落在
        **新进入画幅的区域**，而不是刚刚流出的位置。这只是**防抖动守卫**（下一帧即越界的候选不再
        重生同一像点），不改变 §4 的出生语义；真实比例（flow ≪ 网格间距）下它不拒绝任何候选
        （`test_new_area_births_survive_the_current_motion` 里的 realism 断言）。
        """
        candidates = covering_grid(self.config.supports_per_camera, frame.size_hw)
        candidate_xy = candidates.unsqueeze(0).to(device=state.xy.device, dtype=state.xy.dtype)
        next_xy = candidate_xy + flow_at_points(flow_px.to(candidate_xy), candidate_xy)
        candidate_valid = in_bounds_xy(PixelPoints(xy=next_xy, unit="original_px"), frame.size_hw)
        radius = covering_radius_px(self.config.supports_per_camera, frame.size_hw)
        return self.state_stack.birth(state, candidate_xy, candidate_valid, radius_px=radius)

    @staticmethod
    def _sample_features(
        latent: torch.Tensor, xy_px: torch.Tensor, sample_grid: SampleGrid
    ) -> torch.Tensor:
        """在像素位置采样融合特征：走 SampleGrid 的 pad/stride 元数据，不写死 xy/stride。"""
        height, width = latent.shape[-2:]
        # 坐标与统计恒 FP32：样本喂给协方差／NLL；不依赖调用方的 autocast。
        grid = sample_grid.with_device(latent.device, torch.float32)
        feature_xy = grid.sensor_to_feature.apply(
            PixelPoints(xy=xy_px.to(latent.device, torch.float32), unit="original_px")
        ).xy
        sampled = F.grid_sample(
            latent.to(torch.float32),
            px_to_grid(feature_xy, (height, width)).unsqueeze(1),
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        )
        return sampled.squeeze(2).permute(0, 2, 1).contiguous()

    def _remember_frame(self, camera: str, frame: MeasurementFrame) -> None:
        """保存上一真实帧（切断图），供下一次 flow 使用；不做任何数值融合。"""
        self._prev_frames[camera] = MeasurementFrame(
            raw=frame.raw.detach(),
            camera_id=frame.camera_id,
            measurement_id=frame.measurement_id,
            input_mode=frame.input_mode,
            scale=frame.scale,
            size_hw=frame.size_hw,
            K=None,
            cfa_pattern=frame.cfa_pattern,
            time_s=frame.time_s,
            time_unit=frame.time_unit,
            saturation_level=frame.saturation_level,
            invalid_mask=None if frame.invalid_mask is None else frame.invalid_mask.detach(),
            provenance=dict(frame.provenance),
        )

    def _remember_trajectory(self, camera: str, xy: torch.Tensor) -> None:
        history = self._trajectory.setdefault(camera, deque(maxlen=self.config.trajectory_len))
        history.append(xy)


def _normalise_encoder_fields(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """把 JSON ``model`` 段收敛到 `EncoderV11Config` 字段。

    训练 JSON 的键名与本 dataclass 不同（见 `_JSON_FIELD_ALIASES` /
    `_JSON_READOUT_ALIASES`）：能唯一对应的做映射，冲突时报错；与契约常量冲突的键
    （`_JSON_CONTRACT_CHECKS`）报错；其余未识别键记录在返回值里，不静默丢弃。
    """
    payload = dict(payload)
    for name, expected in _JSON_CONTRACT_CHECKS.items():
        value = payload.pop(name, None)
        if value is not None and value != expected:
            raise ValueError(
                f"配置 {name}={value!r} 与契约要求 {expected!r} 不一致；不要用它覆盖契约常量"
            )
    for alias, target in _JSON_FIELD_ALIASES.items():
        value = payload.pop(alias, None)
        if value is None:
            continue
        if payload.get(target) not in (None, value):
            raise ValueError(f"配置同时给了 {alias} 与 {target} 且取值不同")
        payload[target] = value
    readout_overrides = {
        target: payload.pop(alias)
        for alias, target in _JSON_READOUT_ALIASES.items()
        if payload.get(alias) is not None
    }

    known = {entry.name for entry in dataclasses.fields(EncoderV11Config)}
    fields = {name: value for name, value in payload.items() if name in known and value is not None}
    if "cameras" in fields and not isinstance(fields["cameras"], str):
        fields["cameras"] = tuple(str(value) for value in fields["cameras"])
    if fields.get("raft_resize_hw") is not None:
        fields["raft_resize_hw"] = tuple(int(value) for value in fields["raft_resize_hw"])
    raw_readout = fields.get("readout")
    if isinstance(raw_readout, ReadoutConfig):
        readout = dataclasses.asdict(raw_readout)
    elif isinstance(raw_readout, Mapping):
        readout = dict(raw_readout)
    elif raw_readout is None:
        readout = {}
    else:
        raise TypeError(f"readout 需要映射 / ReadoutConfig，实际 {type(raw_readout).__name__}")
    for name, value in readout_overrides.items():
        if name in readout and readout[name] != value:
            raise ValueError(f"readout.{name} 与 JSON 顶层别名取值不同")
        readout[name] = value
    if raw_readout is not None or readout_overrides:
        fields["readout"] = ReadoutConfig(**readout)
    return fields, sorted(set(payload) - known)


def build_encoder(
    config: EncoderV11Config | Mapping[str, Any] | None = None,
    *,
    platform: str | None = None,
    arm: str | None = None,
    **encoder_kwargs: Any,
) -> N04V11Encoder:
    """按配置构造编码器（不连接设备、不启动训练）。

    训练入口的约定是 ``build_encoder(model_cfg, platform=<平台>, arm=<A|B|C|D>)``：
    ``platform`` 决定相机集合与平台尺度（取自 WS-D 的 ``PLATFORM_SPECS``，见
    `platform_defaults`），``arm`` 决定四臂开关；显式给出的平台／臂与配置冲突时报错，不静默
    丢弃。``config`` 也可以直接是 `EncoderV11Config`（WS-D 的本地 wrapper 走这条路径），此时
    平台／臂若与配置不一致同样报错。未识别的配置键记录在 ``encoder.config_ignored_keys``。
    其余关键字必须是 `N04V11Encoder.__init__` 自己的参数（measurement／correspondence／…）。
    """
    payload: dict[str, Any] = {}
    if isinstance(config, EncoderV11Config):
        payload = dict(dataclasses.asdict(config))
    elif isinstance(config, Mapping):
        payload = dict(config)
    elif config is not None:
        raise TypeError(f"config 需要 EncoderV11Config / 映射 / None，实际 {type(config).__name__}")
    if arm is not None:
        if arm not in ARM_SPECS:
            raise ValueError(f"未知消融臂 {arm!r}；允许 {tuple(ARM_SPECS)}")
        if payload.get("arm") not in (None, arm):
            raise ValueError(f"arm={arm!r} 与配置里的 {payload['arm']!r} 冲突")
        payload["arm"] = arm
    if platform is not None:
        cameras, scale_name = platform_defaults(platform)
        for name, value in (("cameras", cameras), ("scale_name", scale_name)):
            if payload.get(name) not in (None, value):
                raise ValueError(
                    f"platform={platform!r} 要求 {name}={value!r}，与配置里的 "
                    f"{payload[name]!r} 冲突；要覆盖平台事实就去掉 platform 参数"
                )
            payload[name] = value
    fields, ignored = _normalise_encoder_fields(payload)
    encoder = N04V11Encoder(config=EncoderV11Config(**fields), **encoder_kwargs)
    encoder.config_ignored_keys = ignored
    return encoder

"""N04 的最近机制基线（获得与候选**相同**的信息）。

全部基线接收同样的测量序列、同样的运动上下文（本体 delta、已执行动作、dt、相机身份）
与同样的历史长度，只是内部机制不同：

``single_frame``
    单帧 CNN：每帧独立编码，不带时序状态。
``convgru``
    同上下文 ConvGRU：同样的历史与上下文输入，用卷积门控循环替换 N04 的传播/分解。
``motion_compensation``
    普通运动补偿 + EMA：按上报运动做一次 warp，再与当前测量做指数滑动平均。
``predict_update_filter``
    标准 predict-update filter：状态转移 + 测量修正，创新残差直接参与更新，
    不区分“传播误差 / 外部变化 / 成像不可靠”。
``action_conditioned_recurrent``
    动作条件化循环状态模型：把已执行动作拼到每帧输入上，再交给 GRU。

消融（对候选本体）：

``n04_no_ego``            关闭传播，只用测量（无 ego 分支）
``n04_no_change_branch``  关闭独立变化分支
``n04_all_residual_noise`` 把全部残差当作传播噪声（不做外部变化分解）
``n04_time_shifted_context`` 上下文时间错位（用 t-1 的上下文配 t 的测量）

所有基线都实现与 `N04Encoder` 相同的 ``step`` 接口（``init_state`` / ``step`` /
``readout_tokens``），因此评测代码不需要分支。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..frameworks.n04.config import N04Config
from ..frameworks.n04.ego import EgoMotionContext
from ..frameworks.n04.geometry import GridSpec, backward_grid_from_translation
from ..frameworks.n04.model import N04Encoder, N04Readout, StepOutput, TokenReadout
from ..frameworks.n04.state import new_state

__all__ = ["BASELINE_NAMES", "N04Ablation", "build_baseline"]

BASELINE_NAMES = (
    "single_frame",
    "convgru",
    "motion_compensation",
    "predict_update_filter",
    "action_conditioned_recurrent",
    "n04_no_ego",
    "n04_no_change_branch",
    "n04_all_residual_noise",
    "n04_time_shifted_context",
)


class _BaselineBase(nn.Module):
    """所有基线共享的样板：状态字典、读出与 step 去重。"""

    def __init__(self, cfg: N04Config, *, name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.width = cfg.model.width
        stride = max(1, int(round(cfg.geometry.pixel_stride)))
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.channels, self.width, 3, stride=stride, padding=1),
            nn.GELU(),
            nn.Conv2d(self.width, self.width, 3, padding=1),
        )
        self.norm = nn.GroupNorm(1, self.width)
        self.readout = N04Readout(
            in_channels=self.width,
            tokens=cfg.model.readout_tokens,
            dim=cfg.model.readout_dim,
            cameras=max(1, len(cfg.camera_ids)),
        )
        self._updates: dict[int, int] = {}

    def init_state(self, batch_size: int, camera_ids=None, device="cpu", dtype=torch.float32):
        """与 `N04Encoder.init_state` 相同：每个相机一份独立状态。"""
        cameras = tuple(camera_ids) if camera_ids else self.cfg.camera_ids
        h, w = self.cfg.geometry.feature_size
        return {
            cam: new_state(batch_size, self.width, h, w, camera_id=cam, device=device)
            for cam in cameras
        }

    def reset(self, state, batch_mask):
        return {k: v.masked_reset(batch_mask) for k, v in state.items()}

    @staticmethod
    def detach_state(state):
        return {k: v.detach() for k, v in state.items()}

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """把测量编码成特征场。名字与 `N04Encoder.encode_measurement` 对齐，
        使评测代码对所有候选/基线使用同一条路径。"""
        return self.norm(self.encoder(values))

    def encode_measurement(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode(values)

    def readout_tokens(self, output: StepOutput, token_budget: int | None = None):
        values, mask = self.readout(output.representation, output.validity)
        if token_budget is not None and token_budget < values.shape[1]:
            values = values[:, :token_budget]
            mask = mask[:, :token_budget]
        return TokenReadout(values=values, mask=mask)

    @staticmethod
    def saturation(values: torch.Tensor) -> torch.Tensor:
        return (values >= 0.99).to(values.dtype).amax(dim=1, keepdim=True)


class SingleFrameCNN(_BaselineBase):
    """单帧 CNN：无时序状态，每帧独立编码。"""

    def __init__(self, cfg: N04Config) -> None:
        super().__init__(cfg, name="single_frame")

    def step(
        self,
        values,
        context,
        state,
        *,
        camera_id=None,
        frame_key=None,
        new_measurement=True,
        **kwargs,
    ):
        measured = self.encode(values)
        validity = torch.ones(
            measured.shape[0], 1, *measured.shape[-2:], device=measured.device, dtype=measured.dtype
        )
        fresh = new_state(
            measured.shape[0], self.width, *measured.shape[-2:], camera_id=camera_id or ""
        )
        fresh.stable = measured
        fresh.measured = measured
        fresh.history_valid = torch.ones(measured.shape[0], dtype=torch.bool)
        return StepOutput(
            representation=measured,
            state=fresh,
            validity=validity,
            innovation=None,
            diagnostics={"baseline": self.name},
        )


class ConvGRUCell(nn.Module):
    """卷积门控循环单元（同上下文循环基线）。"""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.conv_z = nn.Conv2d(channels + hidden, hidden, 3, padding=1)
        self.conv_r = nn.Conv2d(channels + hidden, hidden, 3, padding=1)
        self.conv_h = nn.Conv2d(channels + hidden, hidden, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((x, h), dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        candidate = torch.tanh(self.conv_h(torch.cat((x, r * h), dim=1)))
        return (1 - z) * h + z * candidate


class ConvGRUBaseline(_BaselineBase):
    """同上下文 ConvGRU：同样的历史与运动上下文，用卷积 GRU 替换传播/分解。"""

    def __init__(self, cfg: N04Config) -> None:
        super().__init__(cfg, name="convgru")
        self.cell = ConvGRUCell(self.width + 4, self.width)

    def step(
        self,
        values,
        context,
        state,
        *,
        camera_id=None,
        frame_key=None,
        new_measurement=True,
        **kwargs,
    ):
        measured = self.encode(values)
        b, _, h, w = measured.shape
        hidden = state.stable
        if not bool(torch.is_tensor(state.history_valid) and state.history_valid.all()):
            hidden = torch.zeros_like(measured)
        ctx = _context_image(context, b, h, w, measured.device, measured.dtype)
        hidden = self.cell(torch.cat((measured, ctx), dim=1), hidden)
        validity = torch.ones(b, 1, h, w, device=measured.device, dtype=measured.dtype)
        fresh = new_state(b, self.width, h, w, camera_id=camera_id or "")
        fresh.stable = hidden
        fresh.measured = measured
        fresh.history_valid = torch.ones(b, dtype=torch.bool)
        return StepOutput(
            representation=hidden,
            state=fresh,
            validity=validity,
            diagnostics={"baseline": self.name},
        )


class MotionCompensationEMA(_BaselineBase):
    """普通运动补偿 + EMA：按上报运动 warp 上一状态，再与当前测量做指数平均。"""

    def __init__(self, cfg: N04Config, *, alpha: float = 0.4) -> None:
        super().__init__(cfg, name="motion_compensation")
        self.alpha = alpha
        self.spec = GridSpec(
            cfg.geometry.feature_size[0],
            cfg.geometry.feature_size[1],
            pixel_stride=cfg.geometry.pixel_stride,
        )

    def step(
        self,
        values,
        context,
        state,
        *,
        camera_id=None,
        frame_key=None,
        new_measurement=True,
        **kwargs,
    ):
        measured = self.encode(values)
        b, _, h, w = measured.shape
        motion = _reported_motion(context, b, measured.device, measured.dtype)
        grid = backward_grid_from_translation(
            self.spec,
            motion[:, 0],
            motion[:, 1],
            batch=b,
            device=measured.device,
            dtype=measured.dtype,
        )
        warped = F.grid_sample(
            state.stable, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        combined = (1 - self.alpha) * warped + self.alpha * measured
        validity = torch.ones(b, 1, h, w, device=measured.device, dtype=measured.dtype)
        fresh = new_state(b, self.width, h, w, camera_id=camera_id or "")
        fresh.stable = combined
        fresh.measured = measured
        fresh.history_valid = torch.ones(b, dtype=torch.bool)
        return StepOutput(
            representation=combined,
            state=fresh,
            validity=validity,
            diagnostics={"baseline": self.name, "alpha": self.alpha},
        )


class PredictUpdateFilter(_BaselineBase):
    """标准 predict-update filter：创新残差直接参与更新，不区分来源。"""

    def __init__(self, cfg: N04Config, *, gain: float = 0.5) -> None:
        super().__init__(cfg, name="predict_update_filter")
        self.gain = nn.Parameter(torch.tensor(gain))
        self.spec = GridSpec(
            cfg.geometry.feature_size[0],
            cfg.geometry.feature_size[1],
            pixel_stride=cfg.geometry.pixel_stride,
        )

    def step(
        self,
        values,
        context,
        state,
        *,
        camera_id=None,
        frame_key=None,
        new_measurement=True,
        **kwargs,
    ):
        measured = self.encode(values)
        b, _, h, w = measured.shape
        motion = _reported_motion(context, b, measured.device, measured.dtype)
        grid = backward_grid_from_translation(
            self.spec,
            motion[:, 0],
            motion[:, 1],
            batch=b,
            device=measured.device,
            dtype=measured.dtype,
        )
        predicted = F.grid_sample(
            state.stable, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        if not bool(torch.is_tensor(state.history_valid) and state.history_valid.all()):
            predicted = torch.zeros_like(measured)
        #: 普通 predict-update：更新只取决于创新残差，没有来源分解。
        innovation = measured - predicted
        gain = torch.sigmoid(self.gain)
        combined = predicted + gain * innovation
        validity = torch.ones(b, 1, h, w, device=measured.device, dtype=measured.dtype)
        fresh = new_state(b, self.width, h, w, camera_id=camera_id or "")
        fresh.stable = combined
        fresh.prediction = predicted
        fresh.measured = measured
        fresh.history_valid = torch.ones(b, dtype=torch.bool)
        fresh.decomposition = {"innovation": innovation, "gain": gain.expand_as(innovation)}
        return StepOutput(
            representation=combined,
            state=fresh,
            validity=validity,
            diagnostics={"baseline": self.name},
        )


class ActionConditionedRecurrent(_BaselineBase):
    """动作条件化循环状态模型：把已执行动作/本体直接拼进每帧输入。"""

    def __init__(self, cfg: N04Config) -> None:
        super().__init__(cfg, name="action_conditioned_recurrent")
        #: 每帧输入 = 测量特征(width) + 投影后的动作/本体嵌入(embed)；两者拼接后送入 GRU。
        self.action_embed = 8
        self.cell = ConvGRUCell(self.width + self.action_embed, self.width)
        #: 动作/本体的投影宽度在首次调用时按实际上下文维度确定（平台维度不写死）。
        self.motion_proj: nn.Linear | None = None

    def step(
        self,
        values,
        context,
        state,
        *,
        camera_id=None,
        frame_key=None,
        new_measurement=True,
        **kwargs,
    ):
        measured = self.encode(values)
        b, _, h, w = measured.shape
        hidden = state.stable
        if not bool(torch.is_tensor(state.history_valid) and state.history_valid.all()):
            hidden = torch.zeros_like(measured)
        vector, _ = context.motion_vector()
        if self.motion_proj is None:
            #: 首次调用时按实际上下文维度确定投影宽度，之后维度不一致直接报错。
            self.motion_proj = nn.Linear(vector.shape[-1], self.action_embed).to(
                vector.device, vector.dtype
            )
        elif vector.shape[-1] != self.motion_proj.in_features:
            raise ValueError(
                "运动上下文维度不一致："
                f"期望 {self.motion_proj.in_features}，实际 {vector.shape[-1]}"
            )
        ctx = self.motion_proj(vector)[:, :, None, None].expand(b, self.action_embed, h, w)
        hidden = self.cell(torch.cat((measured, ctx), dim=1), hidden)
        validity = torch.ones(b, 1, h, w, device=measured.device, dtype=measured.dtype)
        fresh = new_state(b, self.width, h, w, camera_id=camera_id or "")
        fresh.stable = hidden
        fresh.measured = measured
        fresh.history_valid = torch.ones(b, dtype=torch.bool)
        return StepOutput(
            representation=hidden,
            state=fresh,
            validity=validity,
            diagnostics={"baseline": self.name},
        )


class N04Ablation(nn.Module):
    """N04 本体的消融：直接改写配置或上下文，其余路径完全一致。"""

    def __init__(self, cfg: N04Config, *, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode == "n04_no_ego":
            cfg = cfg.with_model(predictor="identity")
        elif mode == "n04_no_change_branch":
            cfg = cfg.with_model(change_branch=False)
        elif mode == "n04_all_residual_noise":
            cfg = cfg.with_model(change_branch=False, illumination_branch=False)
        elif mode != "n04_time_shifted_context":
            raise ValueError(f"未知消融模式 {mode!r}")
        self.encoder = N04Encoder(cfg)
        self._previous: EgoMotionContext | None = None

    def init_state(self, *a, **kw):
        return self.encoder.init_state(*a, **kw)

    def reset(self, *a, **kw):
        return self.encoder.reset(*a, **kw)

    def detach_state(self, *a, **kw):
        return self.encoder.detach_state(*a, **kw)

    def readout_tokens(self, *a, **kw):
        return self.encoder.readout_tokens(*a, **kw)

    def encode_measurement(self, *a, **kw):
        #: 评测与训练会对所有候选/基线调用 ``encode_measurement``；消融必须实现同一接口，
        #: 否则评测直接 AttributeError，而不是给出可比较的数字。
        return self.encoder.encode_measurement(*a, **kw)

    def predict_before_observe(self, *a, **kw):
        return self.encoder.predict_before_observe(*a, **kw)

    def correct_after_observe(self, *a, **kw):
        return self.encoder.correct_after_observe(*a, **kw)

    def step(self, values, context, state, **kwargs):
        if self.mode == "n04_time_shifted_context":
            #: 时间错位：用**上一帧**的上下文配当前帧的测量。
            used = self._previous if self._previous is not None else context
            self._previous = context
            context = used
        return self.encoder.step(values, context, state, **kwargs)


def _context_image(
    context: EgoMotionContext, b: int, h: int, w: int, device, dtype
) -> torch.Tensor:
    """把运动上下文铺成与特征场同形状的 4 通道图（位置编码由网络自行学习）。"""
    vector, _ = context.motion_vector()
    if vector.shape[0] != b:
        vector = vector.expand(b, -1)
    reduced = vector[:, :4] if vector.shape[-1] >= 4 else F.pad(vector, (0, 4 - vector.shape[-1]))
    return reduced[:, :, None, None].expand(b, 4, h, w).to(device=device, dtype=dtype)


def _reported_motion(context: EgoMotionContext, b: int, device, dtype) -> torch.Tensor:
    """取上报的相机二维平移（没有本体信息时为 0，并记录在调用方的诊断里）。"""
    if context.proprio_delta is None:
        return torch.zeros(b, 2, device=device, dtype=dtype)
    delta = context.proprio_delta.to(device=device, dtype=dtype)
    if delta.shape[-1] >= 2:
        return delta[:, :2]
    return F.pad(delta, (0, 2 - delta.shape[-1]))


def build_baseline(name: str, cfg: N04Config) -> nn.Module:
    if name == "single_frame":
        return SingleFrameCNN(cfg)
    if name == "convgru":
        return ConvGRUBaseline(cfg)
    if name == "motion_compensation":
        return MotionCompensationEMA(cfg)
    if name == "predict_update_filter":
        return PredictUpdateFilter(cfg)
    if name == "action_conditioned_recurrent":
        return ActionConditionedRecurrent(cfg)
    if name.startswith("n04_"):
        return N04Ablation(cfg, mode=name)
    raise ValueError(f"未知基线 {name!r}；可选：{BASELINE_NAMES}")

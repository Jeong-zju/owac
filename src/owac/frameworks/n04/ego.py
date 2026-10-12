"""N04 自运动条件化传播（predict 阶段）。

本模块只实现 ``predict_before_observe`` 一侧：它**只**使用上一状态和截至当前已实际
发生的运动上下文（本体、已执行动作、dt、相机身份、可选标定）产生传播结果，**绝不读
当前图像**。当前图像进入 `innovation`/`model` 的 correction 分支。

输入信息分级（见提示词第一步）：

``learned_proprio``
    默认可部署路线。``MLP(proprio_delta, executed_action, dt, camera_id)`` 结合低分辨率
    上一特征预测每相机的 coarse backward flow，再上采样到特征网格。RawVLA NPZ 不提供
    相机六自由度位姿、深度或完整标定，因此这是当前信息条件下的默认配置。
``proprio_only``
    只用本体/动作、不读上一视觉的简化对照，用来检验“读上一视觉”是否必要。
``calibrated_geometry``
    显式几何。约定 ``T_world_from_camera``，可由深度做 forward projection 后转 backward
    采样，或由纯旋转/显式平面同构直接给出 backward 映射。仅在标定实际可得时启用；
    合成 oracle 用同一开关但单独报告，只作上限诊断。
``identity``
    零位移传播。固定相机必须允许学到接近 identity 的传播，这是可复现的起点。

角度语义：旋转分量不能盲目相减。`ProprioLayout` 明确声明状态向量的分段语义，
四元数旋转段用相对四元数计算，欧拉/轴角按其声明方式处理，无法判定时使用差值并在
``rotation_semantics`` 中显式记为 ``raw_diff``（不假装它是正确的角增量）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N04GeometryConfig
from .geometry import (
    GridSpec,
    WarpResult,
    forward_project_points,
    warp_feature_field,
)

__all__ = [
    "CalibratedGeometryPropagator",
    "CameraCalibration",
    "EgoMotionContext",
    "IdentityPropagator",
    "LearnedProprioPropagator",
    "ProprioLayout",
    "ProprioOnlyPropagator",
    "PropagationResult",
    "build_propagator",
    "relative_rotation",
    "state_delta",
]

RotationSemantics = Literal["quaternion", "euler", "axis_angle", "raw_diff", "none"]


# --------------------------------------------------------------------------------------
# 本体状态语义
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProprioLayout:
    """本体状态向量的分段语义。

    不假设所有平台的状态向量同构：LIBERO 的 8 维状态与 RoboTwin 的 14 维关节状态
    含义不同，旋转段的位置和表示也不同。未声明的平台使用 ``rotation_semantics="none"``
    并把整个向量按普通差值处理（明确记为 raw diff，不假装它是正确的角增量）。
    """

    #: 平移分量索引。
    position_index: tuple[int, ...] = ()
    #: 旋转分量索引。
    rotation_index: tuple[int, ...] = ()
    rotation_semantics: RotationSemantics = "none"
    #: 夹爪/末端开合索引（作为普通标量差值处理）。
    gripper_index: tuple[int, ...] = ()
    #: 关节角索引（作为普通标量差值，仅在有明确关节定义时声明）。
    joint_index: tuple[int, ...] = ()
    name: str = "generic"

    @classmethod
    def libero_ee8(cls) -> ProprioLayout:
        """LIBERO 8 维状态的**声明式**占位布局。

        RawVLA 部署记录只说明 ``robot_states`` 为 8 维本体、来源为作者数据定义，
        并未在本仓库中逐位固定切分。这里按“位置 3 + 旋转 3 + 夹爪 2”声明，并在
        ``name`` 中标记为待核对；调用方若拿到作者定义应显式传入自己的布局，
        而不是依赖这个默认值。
        """
        return cls(
            position_index=(0, 1, 2),
            rotation_index=(3, 4, 5),
            rotation_semantics="euler",
            gripper_index=(6, 7),
            name="libero_ee8_unverified",
        )

    @classmethod
    def robotwin_joint14(cls) -> ProprioLayout:
        """RoboTwin 14 维关节状态：左右臂各 6 关节 + 夹爪，明确为关节角。"""
        return cls(
            joint_index=tuple(range(12)),
            gripper_index=(12, 13),
            rotation_semantics="none",
            name="robotwin_joint14",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position_index": list(self.position_index),
            "rotation_index": list(self.rotation_index),
            "rotation_semantics": self.rotation_semantics,
            "gripper_index": list(self.gripper_index),
            "joint_index": list(self.joint_index),
        }


def relative_rotation(
    q_prev: torch.Tensor, q_curr: torch.Tensor, semantics: RotationSemantics
) -> torch.Tensor:
    """按声明的语义计算相对旋转。

    ``quaternion``：``q_prev^{-1} ⊗ q_curr``（wxyz，单位化）。``euler``/``axis_angle``
    与 ``raw_diff`` 使用差值，但语义标签保留，调用方据此知道差值不等于角增量。
    """
    if semantics == "quaternion" and q_prev.shape[-1] == 4:
        p = F.normalize(q_prev, dim=-1)
        c = F.normalize(q_curr, dim=-1)
        pw, px, py, pz = p.unbind(-1)
        cw, cx, cy, cz = c.unbind(-1)
        # q_prev^{-1} = conj(q_prev)（单位四元数）
        out_w = pw * cw + px * cx + py * cy + pz * cz
        out_x = pw * cx - px * cw - py * cz + pz * cy
        out_y = pw * cy + px * cz - py * cw - pz * cx
        out_z = pw * cz - px * cy + py * cx - pz * cw
        return torch.stack((out_w, out_x, out_y, out_z), dim=-1)
    return q_curr - q_prev


def state_delta(
    proprio_prev: torch.Tensor,
    proprio_curr: torch.Tensor,
    layout: ProprioLayout,
    *,
    dt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``q_t - q_{t-1}`` 的语义正确版本。

    平移/夹爪/关节走差值；旋转段按 ``layout.rotation_semantics`` 处理。返回
    ``(delta, info)``，``info`` 记录实际使用的语义，便于报告里写清“这里是 raw diff”。
    """
    if proprio_prev.shape != proprio_curr.shape:
        raise ValueError("proprio_prev / proprio_curr 形状必须一致")
    delta = proprio_curr - proprio_prev
    info: dict[str, Any] = {"rotation_semantics": layout.rotation_semantics, "layout": layout.name}
    if layout.rotation_index and layout.rotation_semantics != "none":
        idx = list(layout.rotation_index)
        rot = relative_rotation(
            proprio_prev[..., idx], proprio_curr[..., idx], layout.rotation_semantics
        )
        delta = delta.clone()
        delta[..., idx] = rot
        info["rotation_applied"] = True
    else:
        info["rotation_applied"] = False
    if dt is not None:
        info["dt_available"] = True
    return delta, info


# --------------------------------------------------------------------------------------
# 运动上下文与传播结果
# --------------------------------------------------------------------------------------


@dataclass
class EgoMotionContext:
    """截至当前时刻**已经实际发生**的运动上下文。

    全部字段都必须是当前时刻可得的量。缺失信息用 ``*_available=False`` 表达，
    不能用未来动作标签或仿真隐藏状态补齐。
    """

    #: ``q_t - q_{t-1}``（语义正确版本），``[B, D]``。
    proprio_delta: torch.Tensor | None = None
    proprio_delta_available: torch.Tensor | bool = False
    #: **已执行**的动作（不是未来监督），``[B, A]``。
    executed_action: torch.Tensor | None = None
    executed_action_available: torch.Tensor | bool = False
    #: 时间间隔。``dt_available=False`` 时不应把帧索引当作秒。
    dt: torch.Tensor | None = None
    dt_available: bool = False
    time_unit: str = "frame"
    #: 相机身份，``[B]`` 的 long。
    camera_index: torch.Tensor | None = None
    camera_id: str = ""
    #: 可选标定：上一时刻的 ``T_world_from_camera``，``[B,4,4]``（相机坐标 -> 世界坐标）。
    T_world_from_camera: torch.Tensor | None = None
    #: 当前时刻的 ``T_world_from_camera``。显式几何据此得到
    #: ``T_current_from_prev = inverse(T_world_from_current) @ T_world_from_prev``。
    T_world_from_camera_current: torch.Tensor | None = None
    T_world_from_camera_available: bool = False
    camera_transform_available: bool = False
    #: 可选内参 ``[B,3,3]``、上一帧深度 ``[B,H,W]``、平面假设 ``(n, d)``。
    intrinsics: torch.Tensor | None = None
    depth_prev: torch.Tensor | None = None
    plane_normal_prev: torch.Tensor | None = None
    plane_distance: torch.Tensor | float | None = None
    #: 语义/可用性记录。
    info: dict[str, Any] = field(default_factory=dict)

    def motion_vector(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """把可用运动信息拼成 `learned_proprio` 的输入向量与掩码。"""
        parts: list[torch.Tensor] = []
        mask: list[float] = []
        names: list[str] = []

        def add(value: torch.Tensor | None, available: bool, name: str) -> None:
            if value is None:
                return
            v = value if value.dim() >= 2 else value.unsqueeze(0)
            parts.append(v)
            mask.extend([1.0 if available else 0.0] * v.shape[-1])
            names.extend([name] * v.shape[-1])

        add(self.proprio_delta, bool(self._bool_any(self.proprio_delta_available)), "proprio_delta")
        add(
            self.executed_action,
            bool(self._bool_any(self.executed_action_available)),
            "executed_action",
        )
        if self.dt is not None:
            dt = self.dt
            if dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            parts.append(dt)
            mask.append(1.0 if self.dt_available else 0.0)
            names.append("dt")
        if not parts:
            raise ValueError("没有任何可用的运动信息；不应在无运动上下文时调用 learned_proprio")
        vector = torch.cat(parts, dim=-1)
        return vector, {"motion_mask": mask, "motion_names": names, "time_unit": self.time_unit}

    @staticmethod
    def _bool_any(value: torch.Tensor | bool) -> bool:
        if torch.is_tensor(value):
            return bool(value.any())
        return bool(value)


@dataclass
class CameraCalibration:
    """显式几何所需的标定。缺失任一必需项即不可用。"""

    intrinsics: torch.Tensor | None = None
    T_world_from_camera: torch.Tensor | None = None
    depth_prev: torch.Tensor | None = None
    plane_normal_prev: torch.Tensor | None = None
    plane_distance: torch.Tensor | float | None = None
    camera_id: str = ""

    def mode(self) -> Literal["depth", "plane", "unavailable"]:
        if self.intrinsics is None or self.T_world_from_camera is None:
            return "unavailable"
        if self.depth_prev is not None:
            return "depth"
        if self.plane_normal_prev is not None and self.plane_distance is not None:
            return "plane"
        return "unavailable"


@dataclass
class PropagationResult:
    """传播输出：ego warp/transition、可靠性、必要的低维成像参数。"""

    #: 预测的上一特征场（已 warp 到当前坐标）。
    predicted: torch.Tensor | list[torch.Tensor]
    #: 传播有效性 ``[B,1,H,W]``。
    validity: torch.Tensor
    #: 传播不确定性 ``[B,1,H,W]``（越大越不可靠）。
    uncertainty: torch.Tensor
    #: forward flow（特征像素单位），``[B,H,W,2]``；用于诊断和 ego 监督。
    forward_flow: torch.Tensor | None = None
    #: backward 采样网格 ``[B,H,W,2]``（归一化）。
    backward_grid: torch.Tensor | None = None
    #: 相机身份与运动可用性。
    camera_id: str = ""
    motion_available: bool = False
    geometry_mode: str = "none"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Propagator(Protocol):
    """传播算子契约：只依赖上一状态与运动上下文。"""

    def __call__(
        self,
        prev_features: torch.Tensor | list[torch.Tensor],
        spec: GridSpec,
        context: EgoMotionContext,
    ) -> PropagationResult: ...


# --------------------------------------------------------------------------------------
# identity / proprio-only
# --------------------------------------------------------------------------------------


class IdentityPropagator(nn.Module):
    """零位移传播：预测恒等于上一特征，validity 全 1。"""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prev_features: torch.Tensor | list[torch.Tensor],
        spec: GridSpec,
        context: EgoMotionContext,
    ) -> PropagationResult:
        fields = prev_features if isinstance(prev_features, list) else [prev_features]
        b, _, h, w = fields[0].shape
        validity = torch.ones(b, 1, h, w, device=fields[0].device, dtype=fields[0].dtype)
        uncertainty = torch.full_like(validity, 1e-3)
        flow = torch.zeros(b, h, w, 2, device=fields[0].device, dtype=fields[0].dtype)
        grid = spec.identity_grid(b, fields[0].device, fields[0].dtype)
        return PropagationResult(
            predicted=prev_features,
            validity=validity,
            uncertainty=uncertainty,
            forward_flow=flow,
            backward_grid=grid,
            camera_id=context.camera_id,
            motion_available=False,
            geometry_mode="identity",
            diagnostics={"source": "identity"},
        )


class _BaseLearnedPropagator(nn.Module):
    """``learned_proprio`` / ``proprio_only`` 的共享实现。

    ``use_previous_features=False`` 时退化为 ``proprio_only``：低分辨率上一特征被置零，
    网络只能从本体/动作/dt/相机身份预测位移。这是检验“读上一视觉是否必要”的对照。
    """

    def __init__(
        self,
        *,
        motion_dim: int,
        width: int,
        coarse_stride: int,
        max_flow: float,
        camera_embedding: int = 8,
        hidden: int = 64,
        use_previous_features: bool = True,
        encode_feature: bool = True,
    ) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        self.width = width
        self.coarse_stride = max(1, int(coarse_stride))
        self.max_flow = float(max_flow)
        self.use_previous_features = bool(use_previous_features)
        self.camera_embedding = int(camera_embedding)
        self.hidden = int(hidden)
        self.motion_encoder = self._make_motion_encoder(motion_dim) if motion_dim else None
        feat_ch = width if self.use_previous_features else 0
        self.feature_encoder = (
            nn.Sequential(
                nn.Conv2d(feat_ch, hidden, 3, padding=1),
                nn.GELU(),
            )
            if feat_ch > 0
            else None
        )
        self.head = nn.Conv2d(hidden, 3, 3, padding=1)  # (dx, dy, log_sigma)
        #: 零位移初始化：最后一层权重与偏置清零 ⇒ 初始 forward flow 恒为 0。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.camera_embed = nn.Embedding(64, self.camera_embedding)
        nn.init.normal_(self.camera_embed.weight, std=0.02)
        self.encode_feature = bool(encode_feature)
        self.last_flow: torch.Tensor | None = None

    def _coarse_previous(self, prev: torch.Tensor, spec: GridSpec) -> torch.Tensor:
        """把上一特征场降到 coarse 分辨率（feature 网格的 1/coarse_stride）。"""
        s = self.coarse_stride
        h, w = prev.shape[-2:]
        ch = max(1, h // s)
        cw = max(1, w // s)
        return F.adaptive_avg_pool2d(prev, (ch, cw))

    def _make_motion_encoder(self, motion_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(motion_dim + 1 + self.camera_embedding, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )

    def _motion_encoding(
        self, coarse: torch.Tensor, context: EgoMotionContext, b: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        vector, info = context.motion_vector()
        if self.motion_encoder is None:
            #: 运动向量维度由数据侧声明（本体/动作/dt 的平台差异很大），首次调用时确定。
            #: 若后续出现不同维度则显式报错，而不是静默截断或补零。
            self.motion_dim = int(vector.shape[-1])
            self.motion_encoder = self._make_motion_encoder(self.motion_dim).to(
                vector.device, vector.dtype
            )
            info["motion_encoder_initialized_at_runtime"] = True
        elif int(vector.shape[-1]) != int(self.motion_dim):
            raise ValueError(
                f"运动上下文维度不一致：期望 {self.motion_dim}，实际 {vector.shape[-1]}"
            )
        if vector.shape[0] != b:
            vector = vector.expand(b, -1)
        cam_idx = context.camera_index
        if cam_idx is None:
            cam_idx = torch.zeros(b, dtype=torch.long, device=vector.device)
        elif cam_idx.dim() == 0:
            cam_idx = cam_idx.unsqueeze(0).expand(b)
        cam_emb = self.camera_embed(cam_idx.clamp(0, 63))
        dt_present = torch.full(
            (b, 1),
            0.0 if not context.dt_available else 1.0,
            device=vector.device,
            dtype=vector.dtype,
        )
        assert self.motion_encoder is not None
        encoded = self.motion_encoder(torch.cat((vector, dt_present, cam_emb), dim=-1))
        info["camera_embedding_used"] = True
        return encoded, info

    def forward(
        self,
        prev_features: torch.Tensor | list[torch.Tensor],
        spec: GridSpec,
        context: EgoMotionContext,
    ) -> PropagationResult:
        fields = prev_features if isinstance(prev_features, list) else [prev_features]
        prev = fields[0]
        b, _, h, w = prev.shape
        coarse = self._coarse_previous(prev, spec)
        encoded, info = self._motion_encoding(coarse, context, b)
        if not self.use_previous_features or self.feature_encoder is None:
            feat_field = encoded[:, :, None, None].expand(b, encoded.shape[-1], *coarse.shape[-2:])
            info["previous_features_used"] = False
            #: proprio_only 对照：显式记录“低分辨率上一特征被置零”，保证同上下文可比。
            info["previous_feature_replaced_by_zeros"] = True
        else:
            feat_field = self.feature_encoder(coarse)
            feat_field = feat_field + encoded[:, :, None, None]
            info["previous_features_used"] = True
        raw = self.head(feat_field)
        dx, dy, log_sigma = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3]
        #: 位移幅度限制，避免零初始化后首步跳变。
        scale = self.max_flow / float(max(coarse.shape[-2], coarse.shape[-1]))
        dx = torch.tanh(dx / scale) * scale
        dy = torch.tanh(dy / scale) * scale

        #: coarse 网格中心 -> 特征网格坐标：coarse 第 i 个中心对应特征坐标
        #: (i + 0.5) * H / ch - 0.5。上采样用 align_corners=False 的 bilinear。
        def upsample(tensor: torch.Tensor) -> torch.Tensor:
            """coarse 分辨率 -> 特征分辨率（bilinear, align_corners=False）。"""
            return F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)

        #: coarse 上采样到特征分辨率；去掉通道维后组成 [B,H,W,2] 的位移场。
        dx_full, dy_full = upsample(dx).squeeze(1), upsample(dy).squeeze(1)
        #: 低分辨率 log_sigma 上采样后取正值，作为传播不确定性的下界。
        sigma = F.softplus(upsample(log_sigma)) + 1e-3  # [B,1,H,W]
        forward_flow = torch.stack((dx_full, dy_full), dim=-1)  # [B,H,W,2] feature px
        #: backward 采样：p_prev = p_cur - forward_flow（绝不能直接用 forward_flow 当网格）。
        prev_xy = spec.identity_xy(b, prev.device, prev.dtype) - forward_flow
        grid = spec.normalized_from_xy(prev_xy)
        warped: list[torch.Tensor] = []
        validity = None
        for i, scale_field in enumerate(fields):
            if i == 0:
                res = warp_feature_field(scale_field, grid, spec)
                validity = res.validity.unsqueeze(1)
                warped.append(res.values)
            else:
                #: 低分辨率尺度用同一 flow，按尺度换算到该尺度的像素坐标。
                scale_h = scale_field.shape[-2] / h
                scale_w = scale_field.shape[-1] / w
                sub_spec = GridSpec(
                    scale_field.shape[-2],
                    scale_field.shape[-1],
                    pixel_stride=spec.pixel_stride / scale_h,
                )
                sub_flow = forward_flow.clone()
                sub_flow[..., 0] = sub_flow[..., 0] * scale_w
                sub_flow[..., 1] = sub_flow[..., 1] * scale_h
                sub_grid = sub_spec.normalized_from_xy(
                    sub_spec.identity_xy(b, scale_field.device, scale_field.dtype) - sub_flow
                )
                sub_res = warp_feature_field(scale_field, sub_grid, sub_spec)
                warped.append(sub_res.values)
        assert validity is not None
        self.last_flow = forward_flow
        return PropagationResult(
            predicted=warped if isinstance(prev_features, list) else warped[0],
            validity=validity,
            uncertainty=sigma * validity,
            forward_flow=forward_flow,
            backward_grid=grid,
            camera_id=context.camera_id,
            motion_available=True,
            geometry_mode="learned_proprio" if self.use_previous_features else "proprio_only",
            diagnostics={**info, "coarse_stride": self.coarse_stride, "max_flow": self.max_flow},
        )


class LearnedProprioPropagator(_BaseLearnedPropagator):
    """默认路线：``MLP(proprio_delta, executed_action, dt, camera_id) + 低分辨率上一特征``。"""


class ProprioOnlyPropagator(_BaseLearnedPropagator):
    """只用本体、不读上一视觉的传播对照。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["use_previous_features"] = False
        super().__init__(**kwargs)


# --------------------------------------------------------------------------------------
# 显式几何
# --------------------------------------------------------------------------------------


class CalibratedGeometryPropagator(nn.Module):
    """显式几何传播（``calibrated_geometry``）。

    两条路径：

    * ``depth``：用上一帧深度做 forward projection（``forward_project_points``），再由
      投影结果与上一像素坐标求解析 backward 采样网格，并用 forward splat 竞争关系给出
      遮挡 validity。前向投影**不**直接塞进 backward grid。
    * ``plane``：纯旋转或显式平面同构直接给出 backward 映射
      （``H = K_cur (R - t n^T/d) K_prev^{-1}``），平面路径不存在遮挡。

    标定缺失时返回 ``geometry_mode="unavailable"``，不猜测几何。
    """

    def __init__(self, *, max_flow: float = 1e6, check_occlusion: bool = True) -> None:
        super().__init__()
        self.max_flow = max_flow
        self.check_occlusion = check_occlusion

    def _warp_scales(
        self,
        fields: list[torch.Tensor],
        spec: GridSpec,
        grid_main: torch.Tensor,
        forward_flow: torch.Tensor,
    ) -> list[torch.Tensor]:
        """把同一 flow 应用到多尺度场（低分辨率尺度按尺度换算）。"""
        b, _, h, w = fields[0].shape
        out = []
        for i, scale_field in enumerate(fields):
            if i == 0:
                out.append(warp_feature_field(scale_field, grid_main, spec).values)
                continue
            scale_h = scale_field.shape[-2] / h
            scale_w = scale_field.shape[-1] / w
            sub_spec = GridSpec(
                scale_field.shape[-2],
                scale_field.shape[-1],
                pixel_stride=spec.pixel_stride / scale_h,
            )
            sub_flow = forward_flow.clone()
            sub_flow[..., 0] = sub_flow[..., 0] * scale_w
            sub_flow[..., 1] = sub_flow[..., 1] * scale_h
            sub_xy = sub_spec.identity_xy(b, scale_field.device, scale_field.dtype) - sub_flow
            sub_grid = sub_spec.normalized_from_xy(sub_xy)
            out.append(warp_feature_field(scale_field, sub_grid, sub_spec).values)
        return out

    def _unavailable(
        self, prev_features, fields, b, h, w, device, dtype, reason, context
    ) -> PropagationResult:
        validity = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
        predicted = (
            [torch.zeros_like(f) for f in fields]
            if isinstance(prev_features, list)
            else torch.zeros_like(fields[0])
        )
        return PropagationResult(
            predicted=predicted,
            validity=validity,
            uncertainty=torch.ones_like(validity),
            forward_flow=None,
            backward_grid=None,
            camera_id=context.camera_id,
            motion_available=False,
            geometry_mode="unavailable",
            diagnostics={"reason": reason},
        )

    def forward(
        self,
        prev_features: torch.Tensor | list[torch.Tensor],
        spec: GridSpec,
        context: EgoMotionContext,
    ) -> PropagationResult:
        fields = prev_features if isinstance(prev_features, list) else [prev_features]
        prev = fields[0]
        b, _, h, w = prev.shape
        device, dtype = prev.device, prev.dtype
        K = context.intrinsics
        Tprev = context.T_world_from_camera
        if K is None or Tprev is None:
            return self._unavailable(
                prev_features,
                fields,
                b,
                h,
                w,
                device,
                dtype,
                "missing intrinsics or T_world_from_camera",
                context,
            )
        if Tprev.dim() == 2:
            Tprev = Tprev.unsqueeze(0).expand(b, -1, -1)
        Tcurr = context.T_world_from_camera_current
        pose_current_given = Tcurr is not None
        if Tcurr is None:
            #: 没有当前位姿时退化为“位姿未变”，并在诊断里明确记录，不假装有运动。
            Tcurr = Tprev
        if Tcurr.dim() == 2:
            Tcurr = Tcurr.unsqueeze(0).expand(b, -1, -1)
        xy = spec.identity_xy(b, device, dtype)
        hard_validity: torch.Tensor | None = None
        if context.depth_prev is not None:
            depth = context.depth_prev
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            uv_prev = spec.feature_xy_to_pixel_xy(xy)
            uv_curr, in_front = forward_project_points(
                uv_prev, depth.squeeze(1), K, K, Tprev, Tcurr
            )
            curr_xy = spec.pixel_xy_to_feature_xy(uv_curr)
            forward_flow = curr_xy - xy
            #: ``forward_flow = p_proj(p_prev) - p_prev``；backward 采样取 negate：
            #: ``p_prev = p_cur - forward_flow``。不是把 forward flow 直接当网格。
            backward_xy = xy - forward_flow
            grid = spec.normalized_from_xy(backward_xy)
            res: WarpResult = warp_feature_field(
                prev, grid, spec, previous_xy=curr_xy, check_occlusion=self.check_occlusion
            )
            hard_validity = res.validity * in_front.to(dtype)
            mode = "depth"
        elif context.plane_normal_prev is not None and context.plane_distance is not None:
            from .geometry import homography_from_transform

            H = homography_from_transform(
                K, K, Tprev, Tcurr, context.plane_normal_prev, context.plane_distance
            )
            uv_prev = spec.feature_xy_to_pixel_xy(xy)
            ones = torch.ones(b, h, w, 1, device=device, dtype=dtype)
            mapped = torch.einsum("bij,bhwj->bhwi", H, torch.cat((uv_prev, ones), dim=-1))
            z = mapped[..., 2:3]
            eps = torch.finfo(dtype).eps
            safe = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
            curr_xy = spec.pixel_xy_to_feature_xy(mapped[..., :2] / safe)
            forward_flow = curr_xy - xy
            grid = spec.normalized_from_xy(xy - forward_flow)
            res = warp_feature_field(prev, grid, spec)
            hard_validity = res.validity * (z.squeeze(-1) > eps).to(dtype)
            mode = "plane"
        else:
            return self._unavailable(
                prev_features,
                fields,
                b,
                h,
                w,
                device,
                dtype,
                "no depth and no explicit plane assumption",
                context,
            )
        if self.max_flow < float("inf"):
            mag = forward_flow.norm(dim=-1, keepdim=True)
            forward_flow = forward_flow * (self.max_flow / mag.clamp_min(1e-6)).clamp(max=1.0)
        warped = self._warp_scales(fields, spec, grid, forward_flow)
        warped[0] = res.values * hard_validity.unsqueeze(1)
        validity = hard_validity.unsqueeze(1)
        return PropagationResult(
            predicted=warped if isinstance(prev_features, list) else warped[0],
            validity=validity,
            uncertainty=(1.0 - validity) + 1e-3,
            forward_flow=forward_flow,
            backward_grid=grid,
            camera_id=context.camera_id,
            motion_available=True,
            geometry_mode=mode,
            diagnostics={
                "occlusion_checked": res.occlusion_checked,
                "valid_fraction": float(hard_validity.mean()),
                "out_of_bounds_fraction": float(res.out_of_bounds.mean()),
                "pose_current_provided": pose_current_given,
            },
        )


# --------------------------------------------------------------------------------------


def build_propagator(
    mode: str,
    *,
    model_config,
    geometry: N04GeometryConfig,
    motion_dim: int | None = None,
) -> nn.Module:
    """按配置构造传播算子。"""
    if mode == "identity":
        return IdentityPropagator()
    if mode == "calibrated_geometry":
        return CalibratedGeometryPropagator(max_flow=geometry.max_flow)
    if mode == "proprio_only":
        return ProprioOnlyPropagator(
            motion_dim=motion_dim,
            width=model_config.width,
            coarse_stride=model_config.coarse_stride,
            max_flow=geometry.max_flow,
            camera_embedding=model_config.camera_embedding,
            hidden=model_config.hidden,
        )
    if mode == "learned_proprio":
        return LearnedProprioPropagator(
            motion_dim=motion_dim,
            width=model_config.width,
            coarse_stride=model_config.coarse_stride,
            max_flow=geometry.max_flow,
            camera_embedding=model_config.camera_embedding,
            hidden=model_config.hidden,
            use_previous_features=model_config.use_previous_features,
        )
    raise ValueError(f"未知传播模式 {mode!r}")

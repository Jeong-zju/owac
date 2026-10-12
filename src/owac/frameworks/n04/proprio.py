"""本体状态的语义化相对变换与可用性。

N04 传播算子的运动输入必须是**语义正确**的相对量。直接把状态向量逐元素相减在旋转
通道上是错的：两个轴角之差不是相对旋转，也没有短弧归一化。本模块按仓库已记录的字段
定义把状态切成通道组，线性组做差，旋转组用四元数求 ``R_t R_{t-1}^{T}`` 的旋转向量。

数据定义（与 `src/owac/raw/rawvla_sequence.py` 保持一致）：

LIBERO ``robot_states[t]``（8 维）
    ``[0:3]`` 末端位置、``[3:6]`` 末端轴角（弧度）、``[6:8]`` 夹爪。旋转**不能**盲目相减。
RoboTwin 缓存 ``actions[t]``（14 维关节状态）
    ``[0:6]`` 左臂关节、``[6:7]`` 左夹爪、``[7:13]`` 右臂关节、``[13:14]`` 右夹爪；
    关节角是线性量，可逐元素相减，但左右臂/夹爪必须分开标记。

本模块只做语义正确的差分与可用性标记：不假设绝对标定，不推断物理尺度，缺失信息一律
用 ``availability=False`` 表达，而不是用零值冒充“测到的零运动”。RoboTwin 的
``robot_states``（16 维末端表示）不是关节状态，不在这里解释。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = [
    "LIBERO_STATE_LAYOUT",
    "ROBOTWIN_JOINT_LAYOUT",
    "ChannelGroup",
    "ProprioDelta",
    "layout_for_suite",
    "proprio_delta",
    "quaternion_to_rotvec",
    "relative_rotation_vector",
    "rotvec_to_quaternion",
]

#: 通道组的语义类型。``linear`` 逐元素相减；``rotation_vector`` 走四元数相对旋转。
GroupKind = Literal["linear", "rotation_vector"]


@dataclass(frozen=True)
class ChannelGroup:
    """状态向量中一段连续通道的语义。"""

    name: str
    start: int
    stop: int
    kind: GroupKind
    description: str

    @property
    def width(self) -> int:
        return self.stop - self.start

    def slice(self) -> slice:
        return slice(self.start, self.stop)


LIBERO_STATE_LAYOUT: tuple[ChannelGroup, ...] = (
    ChannelGroup("ee_position", 0, 3, "linear", "末端位置（绝对标定未记录，只做差分）"),
    ChannelGroup("ee_rotation", 3, 6, "rotation_vector", "末端轴角（弧度），相对旋转用四元数"),
    ChannelGroup("gripper", 6, 8, "linear", "夹爪开度"),
)

ROBOTWIN_JOINT_LAYOUT: tuple[ChannelGroup, ...] = (
    ChannelGroup("left_arm", 0, 6, "linear", "左臂 6 关节角"),
    ChannelGroup("left_gripper", 6, 7, "linear", "左夹爪"),
    ChannelGroup("right_arm", 7, 13, "linear", "右臂 6 关节角"),
    ChannelGroup("right_gripper", 13, 14, "linear", "右夹爪"),
)


def layout_for_suite(suite: str) -> tuple[ChannelGroup, ...]:
    """按套件返回状态通道布局。未知套件必须显式传布局，不能猜。"""
    if suite == "libero":
        return LIBERO_STATE_LAYOUT
    if suite == "robotwin":
        return ROBOTWIN_JOINT_LAYOUT
    raise ValueError(f"未知套件 {suite!r}：无法确定状态通道语义，请显式传入 layout")


def _normalise_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    #: 零四元数（退化旋转）无法归一化：退回单位旋转，并由调用方按可用性掩码处理。
    safe = torch.where(norm < 1e-8, torch.ones_like(norm), norm)
    return quaternion / safe


def rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """轴角 ``[..., 3]`` -> 四元数 ``[..., 4]``（w, x, y, z）。"""
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * theta
    #: sin(half)/theta 在 theta -> 0 处取极限 0.5，用级数展开避免 0/0。
    small = theta < 1e-6
    scale = torch.where(small, 0.5 - theta**2 / 48.0, torch.sin(half) / theta.clamp_min(1e-12))
    xyz = rotvec * scale
    w = torch.where(small, 1.0 - theta**2 / 8.0, torch.cos(half))
    return torch.cat([w, xyz], dim=-1)


def quaternion_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    """四元数 ``[..., 4]``（w, x, y, z）-> 轴角 ``[..., 3]``，取短弧（角度 <= pi）。"""
    q = _normalise_quaternion(quaternion)
    #: q 与 -q 表示同一旋转：统一到 w >= 0 的半球即可得到最短弧。
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    q = q * sign
    w = q[..., 0].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w.unsqueeze(-1))
    small = sin_half < 1e-6
    #: angle/sin_half -> 2 在 sin_half -> 0 处；小角度用级数避免 0/0。
    scale = torch.where(small, 2.0 + angle**2 / 6.0, angle / sin_half.clamp_min(1e-12))
    return xyz * scale


def relative_rotation_vector(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """``current``/``previous`` 为轴角 ``[..., 3]``，返回 ``R_current R_previous^T`` 的轴角。

    这是**体坐标系**下的相对旋转（先施加 previous，再施加 current 的增量），与
    ``q_t - q_{t-1}`` 的逐元素差完全不同。返回角度被限制在 ``[0, pi]``。
    """
    q_current = rotvec_to_quaternion(current)
    q_previous = rotvec_to_quaternion(previous)
    #: 四元数共轭即逆旋转（单位四元数）。
    conjugate = torch.cat([q_previous[..., :1], -q_previous[..., 1:]], dim=-1)
    w1, x1, y1, z1 = q_current.unbind(dim=-1)
    w2, x2, y2, z2 = conjugate.unbind(dim=-1)
    relative = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return quaternion_to_rotvec(relative)


@dataclass
class ProprioDelta:
    """相对运动及其可用性。

    ``delta`` 形状 ``[B, D]``，与传入状态同宽；``availability`` 形状 ``[B, D]``（float，
    1=该通道的运动增量可用，0=不可用）。按组不可用时整组置零并标 0，不用零值冒充测量。
    """

    delta: torch.Tensor
    availability: torch.Tensor
    time_unit: str = "frame"
    dt: torch.Tensor | None = None
    info: dict[str, Any] | None = None

    def group_available(self, layout: tuple[ChannelGroup, ...], name: str) -> bool:
        for group in layout:
            if group.name == name:
                return bool(self.availability[:, group.slice()].min() > 0.5)
        raise KeyError(f"布局里没有通道组 {name!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta_shape": list(self.delta.shape),
            "time_unit": self.time_unit,
            "dt_available": self.dt is not None,
            "groups": None if self.info is None else self.info.get("groups"),
        }


def proprio_delta(
    proprio: torch.Tensor,
    previous: torch.Tensor | None,
    *,
    layout: tuple[ChannelGroup, ...],
    available: torch.Tensor | bool = True,
    dt: torch.Tensor | None = None,
    time_unit: str = "frame",
) -> ProprioDelta:
    """按通道语义计算 ``proprio_t`` 相对 ``previous`` 的增量。

    * ``previous is None``：没有前置状态，全部标为不可用（缺失 != 零运动）。
    * ``available`` 可以是整体布尔或 ``[B]``/``[B, 1]`` 张量，用于“本体整体缺失”的情况。
    * ``dt`` 只接受**真实单位**的时间间隔；只有帧索引时传 ``None``，``time_unit="frame"``。
    """
    if proprio.dim() != 2:
        raise ValueError(f"proprio 期望 [B, D]，实际 {tuple(proprio.shape)}")
    width = proprio.shape[-1]
    for group in layout:
        if group.stop > width:
            raise ValueError(f"通道组 {group.name} 需要 {group.stop} 维，实际状态只有 {width} 维")
    batch = proprio.shape[0]
    device, dtype = proprio.device, proprio.dtype
    if previous is None:
        return ProprioDelta(
            delta=torch.zeros_like(proprio),
            availability=torch.zeros(batch, width, device=device, dtype=dtype),
            time_unit=time_unit,
            dt=dt,
            info={"groups": {g.name: False for g in layout}, "reason": "no_previous_state"},
        )

    delta = torch.zeros_like(proprio)
    availability = torch.zeros(batch, width, device=device, dtype=dtype)
    groups: dict[str, bool] = {}
    if isinstance(available, bool):
        base = torch.full((batch, 1), 1.0 if available else 0.0, device=device, dtype=dtype)
    else:
        base = available.to(device=device, dtype=dtype)
        if base.dim() == 1:
            base = base.reshape(-1, 1)
        if base.shape[0] != batch:
            raise ValueError(f"available 的 batch 维 {base.shape[0]} 与 proprio 的 {batch} 不一致")
    for group in layout:
        index = group.slice()
        if group.kind == "linear":
            value = proprio[:, index] - previous[:, index]
        else:
            value = relative_rotation_vector(proprio[:, index], previous[:, index])
        delta[:, index] = value * base
        availability[:, index] = base
        groups[group.name] = bool(base.min() > 0.5)
    return ProprioDelta(
        delta=delta,
        availability=availability,
        time_unit=time_unit,
        dt=dt,
        info={"groups": groups},
    )

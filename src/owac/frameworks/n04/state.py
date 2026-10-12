"""N04 持续状态与时间/相机语义。

`N04State` 是 N04 的完整内部状态。它**不是**一张物理地图，而是每个相机独立维护的
多尺度特征场 + 有效性 + 分解统计 + 时间/身份元数据。核心字段：

``stable``
    上一稳定特征。传播后作为 ``pred_t`` 参与更新，代表“被自运动解释掉的部分”。
``change``
    独立的变化特征 cell：持续空间连贯、无法由自运动解释的外部变化。它和 ``stable``
    分开存放，是为了不让稳定分支把移动目标直接抹平。
``prediction``
    本步传播后的预测特征（``stable`` 经 warp）。保留它是为了让 correction 分支能看到
    “预测”与“上一稳定”之间的差别，也便于外部检查 predict/correct 两个阶段。
``warp_validity``
    传播有效性：该位置的历史证据是否有真实来源。除 warp 越界/被遮挡外，还有
    ``available``（该相机本步是否真的有测量）与 ``history_valid``（是否有可用历史）。
``uncertainty``
    传播可靠性（越大越不可靠）。首版由 learned_proprio 的 log-sigma 与观测可靠性合成。
``decomposition``
    最近一步的三个分解分支与 unknown 残留通道（更新权重语义，未校准前不写成概率）。
``illumination``
    只在可靠区域上估计的全局/低频增益，供残差解释；不消除测量特征。
``measured``
    最近一次有效测量特征，供 action readout 使用；不因照明分支被归一化掉。
``timestamp`` / ``time_unit``
    时间语义。只有帧索引时 ``time_unit="frame"``，不编造秒数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["N04State", "new_state", "select_state", "state_bytes"]


@dataclass
class N04State:
    """单个 batch 的 N04 状态。所有特征场的形状为 ``[B, C, H, W]``。"""

    #: 稳定特征（多尺度时为 list，``[0]`` 为最高分辨率）。
    stable: torch.Tensor | list[torch.Tensor]
    #: 独立变化特征 cell，与 ``stable`` 同结构。
    change: torch.Tensor | list[torch.Tensor] | None = None
    #: 本步传播后的预测特征。第一帧为空（没有历史）。
    prediction: torch.Tensor | list[torch.Tensor] | None = None
    #: 传播有效性 ``[B,1,H,W]``（多尺度时同样成 list，或以最高分辨率为准）。
    warp_validity: torch.Tensor | None = None
    #: 是否存在可用历史。
    history_valid: torch.Tensor | bool = False
    #: 该相机本步是否真的有测量（``new_measurement``）。
    available: torch.Tensor | bool = True
    #: 传播不确定性。
    uncertainty: torch.Tensor | None = None
    #: 分解输出（更新权重 + unknown 残留通道）。
    decomposition: dict[str, torch.Tensor] = field(default_factory=dict)
    #: 低频照明增益估计 ``[B,1,1,1]`` 或 ``[B,1,h,w]``。
    illumination: torch.Tensor | None = None
    #: 最近一次有效测量特征，供 readout。
    measured: torch.Tensor | None = None
    #: 最近一步的 forward flow（特征像素），形状 ``[B,H,W,2]``（诊断/监督用）。
    forward_flow: torch.Tensor | None = None
    #: 相机身份与时间语义。
    camera_id: str = ""
    frame_id: torch.Tensor | None = None
    step_index: int = -1
    timestamp: torch.Tensor | None = None
    time_unit: str = "frame"
    #: 每个 episode 的年龄（从 reset 起累计的有效测量数），用于 burn-in 与诊断。
    age: torch.Tensor | None = None
    #: 累积诊断计数（更新次数等），供“同一帧多次去噪只更新一次”的检查。
    updates: int = 0
    #: 记录本帧是否已经跑过 update，防止同一帧重复累计观测。
    observed_frame: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- 结构工具 ---------------------------------------------------------------

    def fields(self) -> list[torch.Tensor]:
        """返回多尺度特征 list（兼容单张量表示）。"""
        if isinstance(self.stable, list):
            return self.stable
        return [self.stable]

    def change_fields(self) -> list[torch.Tensor] | None:
        if self.change is None:
            return None
        if isinstance(self.change, list):
            return self.change
        return [self.change]

    def prediction_fields(self) -> list[torch.Tensor] | None:
        if self.prediction is None:
            return None
        if isinstance(self.prediction, list):
            return self.prediction
        return [self.prediction]

    @property
    def batch_size(self) -> int:
        return self.fields()[0].shape[0]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> N04State:
        def move(x):
            if x is None or isinstance(x, (str, int, float, bool)):
                return x
            if isinstance(x, list):
                return [move(v) for v in x]
            if torch.is_tensor(x):
                if dtype is not None and x.is_floating_point():
                    return x.to(device=device, dtype=dtype)
                return x.to(device=device)
            return x

        self.stable = move(self.stable)
        self.change = move(self.change)
        self.prediction = move(self.prediction)
        self.warp_validity = move(self.warp_validity)
        self.uncertainty = move(self.uncertainty)
        self.illumination = move(self.illumination)
        self.measured = move(self.measured)
        self.forward_flow = move(self.forward_flow)
        self.frame_id = move(self.frame_id)
        self.timestamp = move(self.timestamp)
        self.age = move(self.age)
        self.decomposition = {k: move(v) for k, v in self.decomposition.items()}
        return self

    def detach(self) -> N04State:
        """TBPTT 边界使用：切断图但保留状态值。"""

        def d(x):
            if isinstance(x, list):
                return [d(v) for v in x]
            if torch.is_tensor(x):
                return x.detach()
            return x

        self.stable = d(self.stable)
        self.change = d(self.change)
        self.prediction = d(self.prediction)
        self.warp_validity = d(self.warp_validity)
        self.uncertainty = d(self.uncertainty)
        self.illumination = d(self.illumination)
        self.measured = d(self.measured)
        self.forward_flow = d(self.forward_flow)
        self.decomposition = {k: d(v) for k, v in self.decomposition.items()}
        return self

    def select(self, index: torch.Tensor | slice) -> N04State:
        """按 batch 索引取子状态（部分 batch reset / episode 结束时使用）。"""

        def s(x):
            if isinstance(x, list):
                return [s(v) for v in x]
            if torch.is_tensor(x) and x.dim() >= 1 and x.shape[0] == self.batch_size:
                return x[index]
            return x

        return N04State(
            stable=s(self.stable),
            change=s(self.change),
            prediction=s(self.prediction),
            warp_validity=s(self.warp_validity),
            history_valid=self.history_valid,
            available=self.available,
            uncertainty=s(self.uncertainty),
            decomposition={k: s(v) for k, v in self.decomposition.items()},
            illumination=s(self.illumination),
            measured=s(self.measured),
            forward_flow=s(self.forward_flow),
            camera_id=self.camera_id,
            frame_id=s(self.frame_id),
            step_index=self.step_index,
            timestamp=s(self.timestamp),
            time_unit=self.time_unit,
            age=s(self.age),
            updates=self.updates,
            observed_frame=self.observed_frame,
            extra=dict(self.extra),
        )

    def masked_reset(self, batch_mask: torch.Tensor) -> N04State:
        """把 ``batch_mask`` 为 True 的样本重置为“无历史”空状态。

        ``batch_mask`` 形状 ``[B]``（bool）。重置的样本 ``history_valid=False``，
        下一步会用有效当前测量直接初始化，而不是把上一次 episode 的状态带过来。
        """
        if batch_mask.dim() != 1 or batch_mask.shape[0] != self.batch_size:
            raise ValueError("batch_mask 必须是 [B] 的 bool 张量")
        keep = (~batch_mask).to(self.measured.dtype if self.measured is not None else torch.float32)
        view = keep.reshape(-1, *([1] * (self.fields()[0].dim() - 1)))

        def blank(x):
            if isinstance(x, list):
                return [blank(v) for v in x]
            if torch.is_tensor(x):
                if x.dim() >= 1 and x.shape[0] == self.batch_size:
                    return x * view.to(x.dtype)
                return x
            return x

        self.stable = blank(self.stable)
        self.change = blank(self.change)
        self.prediction = blank(self.prediction)
        self.warp_validity = blank(self.warp_validity)
        self.uncertainty = blank(self.uncertainty)
        self.illumination = blank(self.illumination)
        self.measured = blank(self.measured)
        self.forward_flow = blank(self.forward_flow)
        self.decomposition = {k: blank(v) for k, v in self.decomposition.items()}
        self.history_valid = (
            torch.zeros_like(batch_mask) if torch.is_tensor(self.history_valid) else False
        )
        self.available = torch.ones_like(batch_mask)
        if self.age is not None:
            self.age = self.age * keep.to(self.age.dtype)
        self.observed_frame = None
        return self

    def state_bytes(self) -> int:
        """状态占用的字节数（float32 计），用于报告状态成本。"""
        total = 0

        def acc(x):
            nonlocal total
            if isinstance(x, list):
                for v in x:
                    acc(v)
            elif torch.is_tensor(x):
                total += x.numel() * x.element_size()

        for value in (
            self.stable,
            self.change,
            self.prediction,
            self.warp_validity,
            self.uncertainty,
            self.illumination,
            self.measured,
            self.forward_flow,
            self.frame_id,
            self.timestamp,
            self.age,
        ):
            acc(value)
        for v in self.decomposition.values():
            acc(v)
        return total


def new_state(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    *,
    camera_id: str = "",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    scales: int = 1,
) -> N04State:
    """构造空状态：``history_valid=False``，下一步用当前测量初始化。"""
    stable = []
    for s in range(max(1, scales)):
        h = max(1, height // (2**s))
        w = max(1, width // (2**s))
        stable.append(torch.zeros(batch_size, channels, h, w, device=device, dtype=dtype))
    return N04State(
        stable=stable if scales > 1 else stable[0],
        change=[torch.zeros_like(t) for t in stable] if scales > 1 else torch.zeros_like(stable[0]),
        prediction=None,
        warp_validity=None,
        history_valid=False,
        available=torch.ones(batch_size, device=device, dtype=torch.bool),
        uncertainty=None,
        measured=None,
        camera_id=camera_id,
        age=torch.zeros(batch_size, device=device, dtype=torch.long),
        extra={"initialized": False},
    )


def select_state(state: N04State, index: torch.Tensor | slice) -> N04State:
    return state.select(index)


def state_bytes(state: N04State) -> int:
    return state.state_bytes()

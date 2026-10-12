"""N04 创新分解：ego_error / external_change / imaging_unreliability + unknown 残留通道。

概念顺序（与提示词一致）::

    f_t = MeasurementStem(raw_t, metadata_t)
    pred_t, warp_valid, uncertainty = Propagate(state_{t-1}, context_t)
    innovation = f_t - pred_t
    ego_error, external_change, imaging_unreliability = InnovationDecomposer(
        innovation, f_t, pred_t, warp_valid, uncertainty, metadata_t)
    stable_t, change_t = Update(pred_t, f_t, decomposition)

**未校准前不写成概率。** 三个分支输出的是用于状态更新的学习权重 ``w_*``（0..1 的
连续门），加一个 ``g_unknown`` 残留通道处理“信息不足/不可识别”。三者与 unknown 通过
softmax 归一化，故 ``w_ego + w_ext + w_img + w_unknown = 1``，但这不是事件概率，
只是分配规则。报告里必须这样描述。

照明分支只在**可靠区域**上估计全局/低频增益，用来解释残差，**不**把测量特征除掉：
未消除的测量特征仍供 action readout。增益开启/关闭可配置；颜色决定目标选择的反例
见 `tests/encoders/n04/test_innovation.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Innovation", "InnovationDecomposer", "estimate_low_frequency_gain"]


@dataclass
class Innovation:
    """一次创新分解的全部输出。"""

    #: 更新权重（未校准前不写成概率）。
    gate_ego: torch.Tensor
    gate_change: torch.Tensor
    gate_uncertain: torch.Tensor
    #: unknown / ambiguous 残留通道。
    unknown: torch.Tensor
    #: 归一化后的三分支占比（和为 1，仅供分析）。
    mix: torch.Tensor
    #: 稳定性门（用于稳定分支更新）：观测可靠、传播可靠时更大。
    stability: torch.Tensor
    #: 最终观测门 ``g_update``：状态更新对当前测量的接受程度。
    g_update: torch.Tensor
    #: 照明：只在可靠区域估计的全局/低频增益与其偏差。
    illumination_gain: torch.Tensor | None = None
    illumination_delta: torch.Tensor | None = None
    #: 中间量，便于测试与诊断。
    residual: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def estimate_low_frequency_gain(
    current: torch.Tensor,
    predicted: torch.Tensor,
    weight: torch.Tensor,
    *,
    low_frequency: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在可靠区域上估计全局/低频增益 ``g ≈ <current>/<predicted>``。

    返回 ``(gain [B,1,H,W] 或 [B,1,1,1], delta = gain - 1)``。只在 ``weight`` 大的
    区域参与统计；对有限增益做裁剪，避免除零放大。
    """
    w = weight.clamp_min(0.0)
    num = (current.detach() * predicted.detach() * w).sum(dim=(-2, -1), keepdim=True)
    den = (predicted.detach().square() * w).sum(dim=(-2, -1), keepdim=True)
    gain_scalar = (num / (den + eps)).clamp(0.2, 5.0)
    if not low_frequency:
        return gain_scalar, gain_scalar - 1.0
    #: 低频：把标量增益与预测场的局部平均能量结合，得到空间平滑的逐位置增益。
    local = F.avg_pool2d(predicted.detach().abs(), kernel_size=5, stride=1, padding=2)
    local_cur = F.avg_pool2d(current.detach().abs(), kernel_size=5, stride=1, padding=2)
    smooth = (local_cur + eps) / (local + eps)
    gain_field = (0.5 * gain_scalar + 0.5 * smooth).clamp(0.2, 5.0)
    return gain_field, gain_field - 1.0


class InnovationDecomposer(nn.Module):
    """三分支分解 + unknown 残留通道。

    ``use_illumination_branch=False`` 时，照明增益恒为 1（``illumination_delta=0``），
    其余路径不变，便于做开启/关闭对比与反例检验。
    """

    def __init__(
        self,
        *,
        feature_channels: int,
        hidden: int = 64,
        sigma_floor: float = 0.05,
        use_illumination_branch: bool = True,
        use_change_branch: bool = True,
        exposure_gate: float = 0.25,
    ) -> None:
        super().__init__()
        self.sigma_floor = float(sigma_floor)
        self.use_illumination_branch = bool(use_illumination_branch)
        self.use_change_branch = bool(use_change_branch)
        self.exposure_gate = float(exposure_gate)
        self.feature_channels = int(feature_channels)
        self.hidden = int(hidden)
        #: 输入：innovation(C) + current(C) + predicted(C) + validity(1) + uncertainty(1)
        #:        + saturation(1) + illumination_delta(1)
        self.trunk: nn.Sequential | None = None
        self.head: nn.Conv2d | None = None

    def _build(self, in_ch: int, device: torch.device, dtype: torch.dtype) -> None:
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, self.hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1),
            nn.GELU(),
        ).to(device, dtype)
        #: 三个分支 + unknown + stability，共 5 个 logit 通道；零初始化使首步门控为
        #: “三分支各 1/3、unknown 1/3、stability=0.5”，训练自行调整。
        self.head = nn.Conv2d(self.hidden, 5, 1).to(device, dtype)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        *,
        current: torch.Tensor,
        predicted: torch.Tensor,
        validity: torch.Tensor,
        uncertainty: torch.Tensor,
        saturation: torch.Tensor | None = None,
    ) -> Innovation:
        if current.shape != predicted.shape:
            raise ValueError("current 与 predicted 形状必须一致")
        b, c, h, w = current.shape
        innovation = current - predicted
        #: 传播可靠性 sigmoid(-uncertainty)；``sigma_floor`` 避免在极低不确定度时饱和。
        prop_reliability = torch.sigmoid(-(uncertainty.clamp_min(self.sigma_floor)))
        if saturation is None:
            saturation = torch.zeros(b, 1, h, w, device=current.device, dtype=current.dtype)
        else:
            saturation = saturation.reshape(b, 1, h, w)
        if self.use_illumination_branch:
            gain_weight = validity.clamp(0, 1) * prop_reliability
            gain, gain_delta = estimate_low_frequency_gain(current, predicted, gain_weight)
        else:
            gain = torch.ones(b, 1, h, w, device=current.device, dtype=current.dtype)
            gain_delta = torch.zeros_like(gain)
        #: 照明分支不修改 current：只把增益偏差作为“成像不可靠”的线索送入门控。
        features = torch.cat(
            (innovation, current, predicted, validity, prop_reliability, saturation, gain_delta),
            dim=1,
        )
        if self.trunk is None or self.head is None:
            self._build(features.shape[1], features.device, features.dtype)
        assert self.trunk is not None and self.head is not None
        logits = self.head(self.trunk(features))
        mix = torch.softmax(logits, dim=1)
        w_ego = mix[:, 0:1]
        w_change = mix[:, 1:2]
        w_uncertain = mix[:, 2:3]
        unknown = mix[:, 3:4]
        stability = torch.sigmoid(logits[:, 4:5])
        if not self.use_change_branch:
            #: 消融“无外部变化分支”：变化权重并入 unknown，不再单独承载外部变化。
            w_ego = w_ego + w_change
            w_change = torch.zeros_like(w_change)
            unknown = (unknown + mix[:, 1:2]).clamp(max=1.0)
        w_ego = w_ego * validity
        #: ``g_update``：观测可靠 + 传播误差小 ⇒ 更接受当前测量；
        #: 明显传播失配或饱和 ⇒ 降低不可靠一方的权重。
        mismatch = (innovation.abs().mean(dim=1, keepdim=True) * prop_reliability).clamp(0, 10)
        mismatch_score = torch.exp(-mismatch)
        base = 0.35 + 0.4 * (1.0 - prop_reliability) + 0.25 * (1.0 - mismatch_score)
        exposure_reliability = 1.0 - saturation
        gate = (base * (0.5 + 0.5 * w_change)).clamp(0.0, 1.0)
        gate = gate * (1.0 - self.exposure_gate * saturation)
        g_update = (gate * exposure_reliability + (1.0 - exposure_reliability) * 0.05).clamp(
            0.0, 1.0
        )
        #: 稳定分支只在传播可靠且有效的位置保持历史。
        return Innovation(
            gate_ego=w_ego,
            gate_change=w_change,
            gate_uncertain=w_uncertain,
            unknown=unknown,
            mix=mix,
            stability=(stability * validity * prop_reliability).clamp(0.0, 1.0),
            g_update=g_update,
            illumination_gain=gain,
            illumination_delta=gain_delta,
            residual=innovation,
            diagnostics={
                "prop_reliability_mean": float(prop_reliability.mean()),
                "gate_mean": float(g_update.mean()),
                "change_gate_mean": float(w_change.mean()),
                "unknown_mean": float(unknown.mean()),
                "illumination_branch": self.use_illumination_branch,
                "change_branch": self.use_change_branch,
            },
        )

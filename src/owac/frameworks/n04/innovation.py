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
    propagation_validity: torch.Tensor | None = None
    #: Instantaneous signed external contribution, before memory decay/write scaling.
    external_evidence: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def estimate_low_frequency_gain(
    current: torch.Tensor,
    predicted: torch.Tensor,
    weight: torch.Tensor,
    *,
    low_frequency: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在**可靠且曝光良好**的区域上估计全局/低频增益 ``g``：``current ≈ g · predicted``。

    输入必须是校准到共同单位的线性测量，不能是经过 GELU/GroupNorm 的特征。
    CFA 调用方保持各滤色采样平面分离。实现要点：

    * 只用 ``weight>0`` 且 ``|predicted|`` 足够大的像素参与最小二乘
      ``g = Σ w·c·p / Σ w·p²``。全图幅值比会被大片暗背景（光照变化在那里不可观测）
      拉偏，越暗越不可识别，因此暗像素必须排除而不是放大。
    * 逐通道估计后取中位数并裁剪到 ``[0.2, 5.0]``，避免单个通道的病态解。
    * ``low_frequency=True`` 时与预测场的局部均值比做平均，得到空间平滑的增益场；
      ``False`` 时返回标量增益（逐像素广播）。

    返回 ``(gain, delta)``，``delta = gain - 1``。它只用于把“亮度变化”从几何残差里
    解释出来，**不修改测量特征本身**。
    """
    w = weight.detach().clamp_min(0.0)
    cur = current.detach()
    pred = predicted.detach()
    if cur.shape != pred.shape:
        raise ValueError("current 与 predicted 形状必须一致")
    b = cur.shape[0]
    #: 只在预测幅值足够大的位置估计增益：暗区里的比例没有信息。
    expose = (pred.abs() > 1e-3).to(cur.dtype)
    ww = w * expose
    num = (ww * cur * pred).sum(dim=(-2, -1))  # [B,C]
    den = (ww * pred * pred).sum(dim=(-2, -1))
    per_channel = num / (den + eps)
    valid = (den > eps).to(torch.bool)
    per_channel = per_channel.clamp(0.2, 5.0).masked_fill(~valid, float("nan"))
    gain_scalar = per_channel.nanmedian(dim=1).values.clamp(0.2, 5.0)  # [B]
    gain_scalar = torch.where((den > eps).any(dim=1), gain_scalar, torch.ones_like(gain_scalar))
    scalar = gain_scalar.reshape(b, 1, 1, 1)
    if not low_frequency:
        return scalar.expand_as(cur), (scalar - 1.0).expand_as(cur)
    #: 与标量估计使用同一可靠支持；无效内容不能经局部池化污染邻近有效区域。
    local_num = F.avg_pool2d(ww * cur * pred, kernel_size=5, stride=1, padding=2)
    local_den = F.avg_pool2d(ww * pred.square(), kernel_size=5, stride=1, padding=2)
    smooth = torch.where(local_den > eps, local_num / local_den.clamp_min(eps), scalar)
    gain_field = (0.5 * scalar + 0.5 * smooth).clamp(0.2, 5.0)
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
        # The feature width is known. Register these parameters before any optimizer.
        self._build(4 * feature_channels + 3, torch.device("cpu"), torch.float32)

    def _build(self, in_ch: int, device: torch.device, dtype: torch.dtype) -> None:
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, self.hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1),
            nn.GELU(),
        ).to(device, dtype)
        #: 三个分支 + unknown + stability，共 5 个 logit 通道；零初始化使首步门控为
        #: “三分支各 1/4、unknown 1/4、stability=0.5”，训练自行调整。
        self.head = nn.Conv2d(self.hidden, 5, 1).to(device, dtype)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def ensure_in_channels(self, in_channels: int) -> None:
        """在加载 checkpoint 前按已知输入宽度建层。

        输入宽度固定为 4*C+3，构造时已注册；恢复时检查旧配置的宽度，拒绝静默丢参数。
        """
        in_channels = int(in_channels)
        if self.trunk is None:
            weight = next(iter(self.parameters()), None)
            device = weight.device if weight is not None else torch.device("cpu")
            dtype = weight.dtype if weight is not None else torch.float32
            self._build(in_channels, device, dtype)
        else:
            first = self.trunk[0]
            assert isinstance(first, nn.Conv2d)
            if int(first.in_channels) != in_channels:
                raise ValueError(
                    f"checkpoint 的分解器输入宽度 {in_channels} 与当前配置 "
                    f"{int(first.in_channels)} 不一致"
                )

    def forward(
        self,
        *,
        current: torch.Tensor,
        predicted: torch.Tensor,
        validity: torch.Tensor,
        uncertainty: torch.Tensor,
        saturation: torch.Tensor | None = None,
        linear_current: torch.Tensor | None = None,
        linear_predicted: torch.Tensor | None = None,
        linear_validity: torch.Tensor | None = None,
    ) -> Innovation:
        if current.shape != predicted.shape:
            raise ValueError("current 与 predicted 形状必须一致")
        b, c, h, w = current.shape
        innovation = current - predicted
        #: 传播可靠性 = exp(-uncertainty)，映射到 (0,1]：uncertainty 为 0 表示完全可靠。
        #: ``sigma_floor`` 给出可靠性的下界，避免极小不确定度把可靠性推到精确 1。
        prop_reliability = torch.exp(-uncertainty.clamp_min(self.sigma_floor)).clamp(0.0, 1.0)
        if saturation is None:
            saturation = torch.zeros(b, 1, h, w, device=current.device, dtype=current.dtype)
        else:
            saturation = saturation.reshape(b, 1, h, w)
        gain_available = linear_current is not None and linear_predicted is not None
        if self.use_illumination_branch and gain_available:
            support = validity * prop_reliability * (1 - saturation)
            support = F.interpolate(
                support, size=linear_current.shape[-2:], mode="bilinear", align_corners=False
            )
            if linear_validity is not None:
                support = support * linear_validity
            linear_gain, _ = estimate_low_frequency_gain(linear_current, linear_predicted, support)
            gain = F.interpolate(
                linear_gain.median(1, keepdim=True).values,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).expand_as(current)
            gain_delta = gain - 1
        else:
            gain = torch.ones_like(current)
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
        mix = torch.softmax(logits[:, :4], dim=1)
        w_ego = mix[:, 0:1]
        w_change = mix[:, 1:2]
        w_uncertain = mix[:, 2:3]
        unknown = mix[:, 3:4]
        stability = torch.sigmoid(logits[:, 4:5])
        if not self.use_change_branch:
            #: 消融“无外部变化分支”：变化权重并入 unknown，不再单独承载外部变化。
            w_change = torch.zeros_like(w_change)
            unknown = (unknown + mix[:, 1:2]).clamp(max=1.0)
        w_ego = w_ego * validity
        #: ``g_update`` 首版是一个**可检查的解析先验**，训练后由分解头继续调整：
        #:
        #: * 传播越可靠（``prop_reliability`` 越大）越接受当前测量；
        #: * 传播误差越大 → 残差更可能来自外部真实变化 → 更及时更新，
        #:   而不是因为 residual 大就一律拒绝新测量；
        #: * 饱和/曝光失效 → 降低当前测量的权重，等待下一帧修正，不把缺失测量
        #:   强行分类成物体运动。
        mismatch = (innovation.abs().mean(dim=1, keepdim=True) * prop_reliability).clamp(0, 10)
        change_score = 1.0 - torch.exp(-mismatch)
        exposure_reliability = 1.0 - saturation
        base = (0.10 + 0.40 * prop_reliability + 0.25 * change_score).clamp(0.0, 1.0)
        # Ego mismatch accepts corrective evidence; unknown limits commitment.
        # Their distinct effects break the former identical-gradient symmetry.
        gate = (
            base
            * (0.5 + 0.5 * w_change + 0.5 * w_ego)
            * (1.0 - 0.5 * w_uncertain)
            * (0.75 + 0.5 * stability)
            * (1.0 - 0.25 * unknown)
        ).clamp(0.0, 1.0)
        gate = gate * (1.0 - self.exposure_gate * saturation)
        g_update = (gate * exposure_reliability + (1.0 - exposure_reliability) * 0.05).clamp(
            0.0, 1.0
        )
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
            propagation_validity=validity,
            diagnostics={
                #: 诊断量只用于记录/评测：显式 detach，避免把带梯度的张量转成标量。
                "prop_reliability_mean": float(prop_reliability.detach().mean()),
                "gate_mean": float(g_update.detach().mean()),
                "change_gate_mean": float(w_change.detach().mean()),
                "unknown_mean": float(unknown.detach().mean()),
                "illumination_branch": self.use_illumination_branch,
                "illumination_available": gain_available,
                "change_branch": self.use_change_branch,
            },
        )

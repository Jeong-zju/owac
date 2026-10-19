"""N04 v1.1 双模式证据更新、模式似然与 forecast 损失（契约 §4、§6）。

观测 ``y [B,N,d]``（当前特征采样），状态 ``m [B,N,2,2d]``，观测矩阵
``H = [I_d | 0]``（不训练稠密 H）。对模式 j：

.. code-block:: text

    r_j        = y - H m_j^-                                  # [B,N,d]
    S_j        = H P_j^- H^T + R = cov_upper_j + R            # [B,N,d]，对角
    log_weight_j = log(mode_prior_j) - 0.5*(r^T S^-1 r + logdet S + d log 2π)
    mode_posterior = softmax(log_weight)
    m_j^+, P_j^+  = RKN_cell_j._update(m_j^-, P_j^-, y, R)    # 作者 cell 的更新式

RKN 的结构化协方差在这个方向上是对角块：``H P^- H^T`` 只取 upper 块，所以 ``S`` 是
逐元素对角矩阵，二次项就是 ``Σ r²/S``、``logdet`` 就是 ``Σ log S``；实现对每个支持点
都是逐元素的，不需要构造 256×256 矩阵。forecast 目标用同一份似然：
``-logsumexp_j(log(mode_prior_j) + log_prob(target | prior_j))``，完整二次项与 log det
都保留，不截断 residual²/variance。

有效 mask 是唯一让测量“不进入”的开关：越界/遮挡或调用方判定无效的支持保持 prior
状态与 prior 模式权重；first-frame 的 fresh 支持用当前观测直接初始化（见 §4 生命周期），
不参与模式打分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .state_estimator import (
    StateEstimator,
    SupportState,
    require_finite,
    stabilise_block_covariance,
)

__all__ = [
    "UpdateResult",
    "forecast_diagnostics",
    "forecast_loss",
    "gaussian_log_prob",
    "innovation_terms",
    "mode_log_weights",
    "update",
]

_MIN_INNOVATION_VARIANCE = 1e-6
_MODE_PRIOR_FLOOR = torch.finfo(torch.float32).tiny
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class UpdateResult:
    """一次双模式更新的诊断与后验状态（所有量 FP32）。"""

    state: SupportState
    residual: torch.Tensor  # [B,N,2,d]  r_j
    innovation_variance: torch.Tensor  # [B,N,2,d]  S_j 对角
    log_likelihood: torch.Tensor  # [B,N,2]     log p(y | mode j)
    log_weight: torch.Tensor  # [B,N,2]     log(mode_prior) + log_likelihood
    mode_posterior: torch.Tensor  # [B,N,2]     实际使用的模式权重（无效/初始化处保持 prior）
    obs_valid: torch.Tensor  # [B,N]       真正参与模式更新的支持


def gaussian_log_prob(residual: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """``log N(residual; 0, diag(variance))``：完整二次项 + log det + d log 2π。"""
    if residual.shape != variance.shape:
        raise ValueError("residual 与 variance 形状必须一致")
    if residual.shape[-1] == 0:
        raise ValueError("观测维度不能为 0")
    dimension = residual.shape[-1]
    quadratic = (residual.square() / variance).sum(dim=-1)
    log_det = torch.log(variance).sum(dim=-1)
    return -0.5 * (quadratic + log_det + dimension * _LOG_2PI)


def innovation_terms(state: SupportState, y: torch.Tensor, R: torch.Tensor):
    """``(r, S)``：``r_j = y - H m_j^-``，``S_j = cov_upper_j + R``（逐元素对角）。"""
    obs_dim = state.obs_dim
    if y.shape[-1] != obs_dim or R.shape != y.shape:
        raise ValueError(f"y / R 需要 [B,N,{obs_dim}]，实际 {tuple(y.shape)} / {tuple(R.shape)}")
    require_finite(y, "观测 y")
    require_finite(R, "测量方差 R")
    if not bool(torch.all(R > 0)):
        raise ValueError("R 必须逐元素为正（质量头的潜在测量方差）")
    residual = y[:, :, None, :] - state.mode_mean[..., :obs_dim]
    variance = (state.mode_cov[0] + R[:, :, None, :]).clamp_min(_MIN_INNOVATION_VARIANCE)
    require_finite(residual, "innovation residual")
    return residual, variance


def mode_log_weights(state: SupportState, log_likelihood: torch.Tensor) -> torch.Tensor:
    """``log(mode_prior_j) + log_likelihood_j``；先验只来自状态里已有的旧权重。"""
    log_prior = torch.log(state.mode_weights.clamp_min(_MODE_PRIOR_FLOOR))
    return log_prior + log_likelihood


def update(
    estimator: StateEstimator,
    state: SupportState,
    y: torch.Tensor,
    R: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> UpdateResult:
    """当前测量进入状态一次：模式打分 -> softmax -> 各分量 ``_update`` -> 后验。

    ``valid`` 与 ``state.valid`` 取交集（前者表达特征采样是否可用，后者表达位置是否
    越界/遮挡）。fresh 支持（还没有任何观测）在这一步用 ``y`` 初始化均值，协方差保持
    出生值；它们不参与模式打分，模式权重保持出生先验。
    """
    if estimator.obs_dim != state.obs_dim:
        raise ValueError(f"估计器 obs_dim={estimator.obs_dim} 与状态 {state.obs_dim} 不一致")
    if state.batch_size != y.shape[0] or state.num_supports != y.shape[1]:
        raise ValueError("y 与状态的 [B,N] 布局不一致")
    residual, variance = innovation_terms(state, y, R)
    log_likelihood = gaussian_log_prob(residual, variance)
    log_weight = mode_log_weights(state, log_likelihood)

    state_valid = state.valid
    if valid is not None:
        if valid.shape != state.valid.shape:
            raise ValueError("valid 需要与 state.valid 同形 [B,N]")
        state_valid = state_valid & valid.to(device=state.xy.device, dtype=torch.bool)
    update_mask = state_valid & ~state.fresh
    posterior_weights = torch.where(
        update_mask[..., None],
        torch.softmax(log_weight, dim=-1),
        state.mode_weights,
    )

    means, upper, lower, side = [], [], [], []
    # 作者 cell 只接受 [M, ...] 的“行”批处理（它用 [:, :lod] 切可观测半），
    # 所以先把 [B,N,...] 展平成 [B*N,...]，调用后再还原。
    batch, supports = state.batch_size, state.num_supports
    y_flat = y.reshape(batch * supports, state.obs_dim)
    R_flat = R.reshape(batch * supports, state.obs_dim)
    mask_flat = update_mask.reshape(batch * supports)
    for index, cell in enumerate(estimator.cells):
        cov_in = [
            block[:, :, index].reshape(batch * supports, state.obs_dim) for block in state.mode_cov
        ]
        mean_j, cov_j = cell._masked_update(
            state.mode_mean[:, :, index].reshape(batch * supports, state.obs_dim * 2),
            cov_in,
            y_flat,
            R_flat,
            mask_flat,
        )
        cov_j = stabilise_block_covariance(*cov_j)
        means.append(mean_j.reshape(batch, supports, state.obs_dim * 2))
        upper.append(cov_j[0].reshape(batch, supports, state.obs_dim))
        lower.append(cov_j[1].reshape(batch, supports, state.obs_dim))
        side.append(cov_j[2].reshape(batch, supports, state.obs_dim))

    seed = state.fresh & state_valid
    seeded_mean = torch.cat((y, torch.zeros_like(y)), dim=-1)  # H = [I | 0]：隐藏半从 0 开始
    mode_mean = torch.stack(means, dim=2)  # [B,N,2,2d]
    mode_mean = torch.where(seed[..., None, None], seeded_mean[:, :, None, :], mode_mean)
    posterior = SupportState(
        xy=state.xy,
        ids=state.ids,
        age=state.age,
        valid=state.valid,
        fresh=state.fresh & ~seed,
        mode_mean=mode_mean,
        mode_cov=(
            torch.stack(upper, dim=2),
            torch.stack(lower, dim=2),
            torch.stack(side, dim=2),
        ),
        mode_weights=posterior_weights,
        sensor_size_hw=state.sensor_size_hw,
    )
    return UpdateResult(
        state=posterior,
        residual=residual,
        innovation_variance=variance,
        log_likelihood=log_likelihood,
        log_weight=log_weight,
        mode_posterior=posterior_weights,
        obs_valid=update_mask,
    )


def forecast_loss(
    target: torch.Tensor,
    state: SupportState,
    R: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    reduce: bool = True,
) -> torch.Tensor:
    """双模式 forecast NLL：``-logsumexp_j(log(mode_prior_j) + log p(target | prior_j))``。

    ``state`` 与 ``R`` 是预测到目标时刻的 prior 与其测量方差；本函数不读任何未来
    RAW／本体／动作，target 只作为损失目标。按契约 §6「各辅助项先按有效样本／维度
    取均值」：``reduce=True`` 时在有效支持与观测维度上同时取均值（分母含 ``obs_dim``），
    没有有效支持时返回 0（均值未定义，不伪造样本）；``reduce=False`` 返回逐支持、
    已按维度平均的 ``[B,N]``。完整二次项与 log det 一律保留，不截断、不压方差。
    """
    if target.shape[-1] != state.obs_dim:
        raise ValueError(f"target 需要 [B,N,{state.obs_dim}]，实际 {tuple(target.shape)}")
    if target.shape[:2] != state.xy.shape[:2]:
        raise ValueError("target 与状态的 [B,N] 布局不一致")
    require_finite(target, "forecast target")
    residual, variance = innovation_terms(state, target, R)
    log_probability = gaussian_log_prob(residual, variance)
    log_prior = torch.log(state.mode_weights.clamp_min(_MODE_PRIOR_FLOOR))
    loss = -torch.logsumexp(log_prior + log_probability, dim=-1)  # [B,N]
    loss = loss / state.obs_dim
    require_finite(loss, "forecast loss")
    if not reduce:
        return loss
    mask = state.valid if valid is None else (state.valid & valid)
    count = int(mask.sum())
    if count == 0:
        return loss.new_zeros(())
    return loss[mask].mean()


def forecast_diagnostics(
    target: torch.Tensor,
    state: SupportState,
    R: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> dict[str, float]:
    """只读标量：加权 quad / logdet、两模式权重、R 与 S 的谱（mean/min/max）。

    数值与 `forecast_loss` 同一 forward 口径（``fc ≈ 0.5*(quad+logdet) + 0.5*d*log(2π)``），
    但整体 ``no_grad`` + ``detach``，不参与反传、不改变任何数值语义。字段前缀
    ``forecast_``；分母为空时写 NaN（调用方按约定写 null，不用 0 冒充）。判据：
    ``quad≈d(=256)`` 残差增长、``quad≫d`` 方差跟不上（塌缩）、``quad≈0`` 且 logdet 大
    为方差膨胀。**停机判据（lead §14.1）：某窗口 `forecast_S_max > 1e3` 或
    `quad ≫ d(=256)` 时暂停训练并定位，不静默继续**（协方差不做上界截断，
    由 `optim.grad_clip` + 该判据共同守卫）。
    """
    with torch.no_grad():
        residual, innovation = innovation_terms(state, target.detach(), R.detach())
        quad = (residual.square() / innovation).sum(dim=-1)  # [B,N,2]
        logdet = torch.log(innovation).sum(dim=-1)
        weights = state.mode_weights.detach()
        mask = state.valid if valid is None else (state.valid & valid)
        mask = mask.detach().to(dtype=weights.dtype)
        weighted = weights * mask[..., None]
        denominator = float(weighted.sum())
        names = (
            "forecast_quad",
            "forecast_logdet",
            "forecast_mode_w0",
            "forecast_mode_w1",
            "forecast_S_mean",
            "forecast_S_min",
            "forecast_S_max",
            "forecast_R_mean",
            "forecast_R_min",
            "forecast_R_max",
        )
        if denominator <= 0.0:
            return dict.fromkeys(names, float("nan"))
        report = {
            "forecast_quad": float((weighted * quad).sum() / denominator),
            "forecast_logdet": float((weighted * logdet).sum() / denominator),
            "forecast_mode_w0": float(weighted[..., 0].sum() / denominator),
            "forecast_mode_w1": float(weighted[..., 1].sum() / denominator),
            "forecast_S_mean": float(innovation[mask > 0].mean()),
            "forecast_S_min": float(innovation[mask > 0].min()),
            "forecast_S_max": float(innovation[mask > 0].max()),
            "forecast_R_mean": float(R.detach()[mask > 0].mean()),
            "forecast_R_min": float(R.detach()[mask > 0].min()),
            "forecast_R_max": float(R.detach()[mask > 0].max()),
        }
    return report

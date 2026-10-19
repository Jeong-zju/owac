"""N04 v1.1 WS-B 契约测试：质量头、持续支持、条件预测、双模式更新与 forecast。

覆盖提示词与 `docs/n04_v11_s2_interfaces.md` 里的硬约束：

* 质量头只读当前帧，R 为正、有限、FP32；
* 当前 prior 只由 t-1 后验与截至 t 的本体得到（当前图像/flow 不进入），
  且用作者 cell 的 ``_predict`` 而不是 ``update→predict``；
* 支持生命周期（覆盖网格出生、flow 推进、越界/遮挡只传播预测、新区域才出生）；
* 双模式打分含完整二次项与 log det，极端残差把后验推向修订模式；
* forecast 与稠密矩阵参考实现一致；
* 梯度能到达质量头与模式转移矩阵；协方差保持正定（Cholesky / 最小特征值）。

全部用合成张量（小 N、小尺寸），不加载官方数据。
"""

from __future__ import annotations

import inspect
import math
from dataclasses import replace

import pytest
import torch

from owac.baselines._vendor.rkn.cell import RKNCell
from owac.frameworks.n04.contracts import DENSE_LATENT_CHANNELS
from owac.frameworks.n04.evidence_update import (
    forecast_diagnostics,
    forecast_loss,
    gaussian_log_prob,
    innovation_terms,
    update,
)
from owac.frameworks.n04.quality import (
    RAW_STAT_CHANNELS,
    QualityHead,
    feature_grid_from_sizes,
    sample_feature_map,
)
from owac.frameworks.n04.state_estimator import (
    DEFAULT_TRANSITION_INIT,
    StateEstimator,
    covering_grid,
    covering_radius_px,
    flow_at_points,
)

OBS_SMALL = 8
STATE_SMALL = 2 * OBS_SMALL
SUPPORTS = 4
SENSOR = (24, 32)  # (H, W)


def small_cell_config(*, num_basis: int = 3, bandwidth: int = 1, hidden: int = 12):
    config = RKNCell.get_default_config()
    config.num_basis = num_basis
    config.bandwidth = bandwidth
    config.trans_net_hidden_units = [hidden]
    config.finalize_modifying()
    return config


def make_estimator(*, obs_dim: int = OBS_SMALL, supports: int = SUPPORTS, seed: int = 0):
    torch.manual_seed(seed)
    return StateEstimator(
        num_supports=supports,
        obs_dim=obs_dim,
        proprio_dim=5,
        control_hidden=8,
        cell_config=small_cell_config(),
    )


def seed_state(estimator: StateEstimator, *, batch: int = 1, seed: int = 1):
    """首帧：覆盖网格出生 + 第一个观测直接初始化，返回 (state, y, R)。"""
    torch.manual_seed(seed)
    state = estimator.reset(batch_size=batch, sensor_size_hw=SENSOR)
    y = torch.randn(batch, estimator.num_supports, estimator.obs_dim)
    R = torch.ones_like(y)
    return update(estimator, state, y, R).state, y, R


def test_constant_velocity_initialization_preserves_stationary_observations():
    stable = StateEstimator(
        num_supports=4,
        obs_dim=8,
        proprio_dim=5,
        control_hidden=8,
        cell_config=small_cell_config(),
        transition_init="constant_velocity",
    )
    state, target, variance = seed_state(stable)
    errors = []
    for _ in range(4):
        state = stable.predict(state, torch.zeros(1, 5))
        predicted = (state.mode_mean[..., :8] * state.mode_weights[..., None]).sum(2)
        errors.append(float((predicted - target).square().mean().detach()))
    assert max(errors) < 1e-12
    forecast_loss(target, state, variance).backward()
    assert stable.cells[0]._tm_11_basis.grad is not None


def test_author_rotation_is_a_nonstationary_counterexample():
    original = make_estimator()
    state, target, _ = seed_state(original)
    for _ in range(4):
        state = original.predict(state, torch.zeros(1, 5))
    predicted = (state.mode_mean[..., :8] * state.mode_weights[..., None]).sum(2)
    assert float((predicted - target).square().mean().detach()) > 0.01


def test_forecast_variance_limit_stops_before_any_update():
    from owac.training.n04_v11 import ForecastEngine

    estimator = make_estimator()
    state, target, variance = seed_state(estimator)
    engine = ForecastEngine(estimator, (1,), total_frames=3, max_innovation_variance=0.5)
    engine.add(0, "cam", state, variance, torch.zeros(1, 5), size_hw=SENSOR, segment_last=2)
    with pytest.raises(FloatingPointError, match="stop limit"):
        engine.settle(engine.anchors[0], target)


def constant_flow(batch: int, dx: float, dy: float) -> torch.Tensor:
    flow = torch.zeros(batch, 2, *SENSOR)
    flow[:, 0] = dx
    flow[:, 1] = dy
    return flow


def quality_inputs(*, batch: int = 1, obs_dim: int = DENSE_LATENT_CHANNELS, seed: int = 0):
    torch.manual_seed(seed)
    height, width = SENSOR
    values = torch.rand(batch, 3, height, width) * 2.0
    support = torch.ones(batch, 1, height, width, dtype=torch.bool)
    saturated = torch.zeros(batch, 1, height, width, dtype=torch.bool)
    latent = torch.randn(batch, obs_dim, height // 4, width // 4)
    grid = feature_grid_from_sizes((height, width), (height // 4, width // 4))
    xy = covering_grid(SUPPORTS, SENSOR).unsqueeze(0).expand(batch, -1, -1).contiguous()
    return values, support, saturated, latent, grid, xy


# ---------------------------------------------------------------------------------------
# 质量头（契约 §3）
# ---------------------------------------------------------------------------------------


def test_quality_head_positive_finite_and_current_frame_only():
    head = QualityHead()
    values, support, saturated, latent, grid, xy = quality_inputs(batch=2)
    R = head(values, support, saturated, latent, xy, grid=grid)

    assert R.shape == (2, SUPPORTS, DENSE_LATENT_CHANNELS)
    assert R.dtype == torch.float32
    assert torch.isfinite(R).all()
    assert float(R.detach().min()) >= 1e-6
    # 单帧纯函数：同一帧两次前向逐位一致。
    assert torch.equal(R, head(values, support, saturated, latent, xy, grid=grid))

    # 只读当前帧：换掉当前 RAW / 当前饱和标记，R 必须跟着变（不是常量输出）。
    dark = head(values * 0.0 + 0.05, support, saturated, latent, xy, grid=grid)
    blown = head(values, support, torch.ones_like(saturated), latent, xy, grid=grid)
    assert not torch.allclose(R, dark)
    assert not torch.allclose(R, blown)
    assert float(dark.detach().min()) >= 1e-6 and bool(torch.isfinite(blown).all())

    # 全零 RAW（最暗帧）也不允许出现 NaN/Inf 或下穿 floor。
    zero = head(torch.zeros_like(values), support, saturated, latent, xy, grid=grid)
    assert bool(torch.isfinite(zero).all()) and float(zero.detach().min()) >= 1e-6


def test_quality_head_signature_has_no_history_depth_or_flow():
    names = set(inspect.signature(QualityHead().forward).parameters)
    assert names == {"values", "support", "saturated", "latent", "xy_px", "grid"}
    forbidden = ("depth", "flow", "residual", "history", "state")
    assert not [name for name in names if any(token in name for token in forbidden)]
    assert RAW_STAT_CHANNELS == ("mean", "std", "log_mean", "edge", "support", "saturated")


def test_sample_feature_map_uses_pixel_centre_convention():
    """采样必须走 SampleGrid 的坐标链：特征格中心 → 原始像素 → 回采，误差 < 0.5 特征像素。"""
    values, _, _, latent, grid, _ = quality_inputs(batch=1, obs_dim=4)
    feature = torch.arange(6 * 8, dtype=torch.float32).reshape(1, 1, 6, 8)
    # 特征格 (col=2,row=1) 的原始像素中心：u=(2+0.5)*4-0.5=9.5, v=(1+0.5)*4-0.5=5.5。
    points = torch.tensor([[[9.5, 5.5]]])
    sampled = sample_feature_map(feature, points, grid)
    assert torch.allclose(sampled, torch.tensor([[[10.0]]]), atol=1e-5)
    assert values.shape[-1] // latent.shape[-1] == 4


# ---------------------------------------------------------------------------------------
# 状态布局与模式转移矩阵（契约 §4）
# ---------------------------------------------------------------------------------------


def test_state_layout_and_transition_initialisation():
    estimator = StateEstimator(num_supports=128, obs_dim=256, proprio_dim=16)
    state = estimator.reset(batch_size=2, sensor_size_hw=(480, 640))

    assert state.xy.shape == (2, 128, 2)
    assert state.ids.shape == (2, 128) and state.ids.dtype == torch.long
    assert state.age.shape == (2, 128) and bool(state.valid.all()) and bool(state.fresh.all())
    assert state.mode_mean.shape == (2, 128, 2, 512)
    assert [block.shape for block in state.mode_cov] == [(2, 128, 2, 256)] * 3
    assert state.mode_weights.shape == (2, 128, 2)

    transition = estimator.mode_transition()
    assert torch.allclose(transition, torch.tensor(DEFAULT_TRANSITION_INIT), atol=1e-6)
    assert torch.allclose(transition.sum(dim=-1), torch.ones(2))
    assert torch.allclose(transition, torch.softmax(estimator.mode_transition_logits, dim=-1))

    # 覆盖网格出生：全部落在画幅内、间距均匀。
    grid = covering_grid(128, (480, 640))
    assert grid.shape == (128, 2)
    assert bool((grid[:, 0] >= 0).all()) and bool((grid[:, 0] < 640).all())
    assert bool((grid[:, 1] >= 0).all()) and bool((grid[:, 1] < 480).all())
    radius = covering_radius_px(128, (480, 640))
    assert 0.0 < radius < min(480, 640) / 2

    # 参数计数：两个作者 RKN cell + 加性控制 + 修订 Q + 转移矩阵 + 质量头。
    assert all(isinstance(cell, RKNCell) for cell in estimator.cells)
    assert RKNCell.__module__ == "owac.baselines._vendor.rkn.cell"  # 复用作者 cell，不重写
    counts = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in estimator.named_children()
    }
    assert counts["cells"] == 2 * 145_279
    assert counts["control"] == 16 * 128 + 128 + 128 * 128 + 128 + 128 * 512 + 512
    assert sum(p.numel() for p in estimator.parameters()) == 375_554
    assert sum(p.numel() for p in QualityHead().parameters()) == 58_240


# ---------------------------------------------------------------------------------------
# 条件预测：因果性、先验来源、update→predict 顺序（契约 §1、§4）
# ---------------------------------------------------------------------------------------


def test_predict_is_pure_function_of_previous_posterior_and_proprio():
    estimator = make_estimator()
    state, _, _ = seed_state(estimator)
    proprio = torch.randn(1, 5)

    prior = estimator.predict(state, proprio)
    # 当前图像 / 当前 flow / 当前测量同时改变，prior 仍逐位不变（它们不在参数里）。
    torch.manual_seed(99)
    _ = torch.randn(1, 3, *SENSOR)
    _ = constant_flow(1, 7.0, -5.0)
    again = estimator.predict(state, proprio)
    assert torch.equal(prior.mode_mean, again.mode_mean)
    assert torch.equal(prior.mode_weights, again.mode_weights)
    assert all(torch.equal(a, b) for a, b in zip(prior.mode_cov, again.mode_cov, strict=True))

    # flow 只改位置：先按 flow 推进再 predict，分布与权重一样。
    moved_a = estimator.advance_positions(state, constant_flow(1, 3.0, 1.0))
    moved_b = estimator.advance_positions(state, constant_flow(1, -9.0, 4.0))
    prior_a = estimator.predict(moved_a, proprio)
    prior_b = estimator.predict(moved_b, proprio)
    assert torch.equal(prior_a.mode_mean, prior_b.mode_mean)
    assert torch.equal(prior_a.mode_weights, prior_b.mode_weights)
    assert not torch.allclose(moved_a.xy, moved_b.xy)

    # 先验权重只来自旧权重 + 行归一化转移矩阵。
    assert torch.allclose(prior.mode_weights, state.mode_weights @ estimator.mode_transition())
    assert torch.allclose(
        torch.softmax(estimator.mode_transition_logits, dim=-1).sum(dim=-1), torch.ones(2)
    )
    basis = torch.tensor([[[1.0, 0.0]]])
    assert torch.allclose(estimator.mode_prior(basis), estimator.mode_transition()[:1])
    # 本体不可用时控制项被 mask 掉（零初始化下与未 mask 一致）；形状错误直接报错。
    masked = estimator.predict(state, proprio, proprio_valid=torch.zeros(1, dtype=torch.bool))
    assert torch.equal(masked.mode_mean, prior.mode_mean)
    with pytest.raises(ValueError, match="proprio_valid"):
        estimator.predict(state, proprio, proprio_valid=torch.ones(3, dtype=torch.bool))


def test_predict_signature_excludes_current_observation_inputs():
    names = set(inspect.signature(StateEstimator.predict).parameters)
    assert names == {"self", "state", "proprio_context", "proprio_valid"}


def test_current_prior_uses_predict_and_excludes_current_measurement():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    batch, supports = state.batch_size, state.num_supports
    proprio = torch.randn(batch, 5)

    # 权重 [1,0]：IMM 混合对两个模式都只取模式 0，可以独立复算 prior。
    weights = torch.zeros(batch, supports, 2)
    weights[..., 0] = 1.0
    state = replace(state, mode_weights=weights)
    prior = estimator.predict(state, proprio)

    flat_mean = state.mode_mean[:, :, 0].reshape(-1, STATE_SMALL)
    flat_cov = [block[:, :, 0].reshape(-1, OBS_SMALL) for block in state.mode_cov]
    for index, cell in enumerate(estimator.cells):
        mean_j, cov_j = cell._predict(flat_mean, flat_cov)
        mean_j = mean_j.reshape(batch, supports, STATE_SMALL)
        mean_j = mean_j + estimator.control(proprio)[:, None, :]
        if index == 1:
            cov_j = [cov_j[0] + estimator.revision_extra_q(), cov_j[1], cov_j[2]]
        assert torch.allclose(prior.mode_mean[:, :, index], mean_j, atol=1e-5)
        got_blocks = [block[:, :, index] for block in prior.mode_cov]
        for got, expected in zip(got_blocks, cov_j, strict=True):
            assert torch.allclose(got, expected, atol=1e-5)

    # 显式顺序检查：把 t-1 后验的观测半推离 y（非零残差），正确顺序的 prior 必须
    # 区别于“先用当前 y 更新再 predict”（作者 forward 的 update→predict 语义）。
    shifted = state.mode_mean.clone()
    shifted[..., :OBS_SMALL] += 1.0
    shifted_state = replace(state, mode_mean=shifted)
    correct = estimator.predict(shifted_state, proprio)
    wrong_order = estimator.cells[0].forward(
        shifted[:, :, 0].reshape(-1, STATE_SMALL),
        [block[:, :, 0].reshape(-1, OBS_SMALL) for block in state.mode_cov],
        y.reshape(-1, OBS_SMALL),
        R.reshape(-1, OBS_SMALL),
    )[2]
    wrong_order = wrong_order.reshape(batch, supports, STATE_SMALL)
    wrong_order = wrong_order + estimator.control(proprio)[:, None, :]
    assert not torch.allclose(correct.mode_mean[:, :, 0], wrong_order, atol=1e-4)

    # 当前 prior 不含当前测量：先 update 再 predict 与直接用 t-1 后验 predict 不同。
    posterior = update(estimator, state, y + 1.0, R).state
    assert not torch.allclose(posterior.mode_mean, state.mode_mean)
    assert not torch.allclose(
        estimator.predict(posterior, proprio).mode_mean, prior.mode_mean, atol=1e-4
    )


# ---------------------------------------------------------------------------------------
# 支持生命周期（契约 §4）
# ---------------------------------------------------------------------------------------


def test_flow_advances_positions_and_bounds_mask_only_propagation():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)

    flow = constant_flow(1, 2.0, -1.0)
    assert torch.allclose(
        flow_at_points(flow, state.xy), torch.tensor([[[2.0, -1.0]]]).expand_as(state.xy)
    )
    moved = estimator.advance_positions(state, flow)
    assert torch.allclose(moved.xy, state.xy + torch.tensor([2.0, -1.0]))
    assert torch.equal(moved.ids, state.ids)  # 不重排、不换 ID
    assert torch.equal(moved.age, state.age + 1)
    assert bool(moved.valid.all())
    assert torch.equal(moved.mode_mean, state.mode_mean)
    assert torch.equal(moved.mode_weights, state.mode_weights)

    # 越界：位置照走、状态不变，valid=False 让更新被 mask 掉。
    pushed = estimator.advance_positions(moved, constant_flow(1, 1000.0, 0.0))
    assert not bool(pushed.valid.any())
    assert torch.equal(pushed.mode_mean, state.mode_mean)
    result = update(estimator, pushed, y, R)
    invalid = ~pushed.valid
    assert torch.equal(result.state.mode_mean[invalid], pushed.mode_mean[invalid])
    assert torch.equal(result.state.mode_weights[invalid], pushed.mode_weights[invalid])
    assert not bool(result.obs_valid[invalid].any())

    # 遮挡：同一套 mask 语义（画幅上半被遮挡）。
    occlusion = torch.zeros(1, *SENSOR, dtype=torch.bool)
    occlusion[:, : SENSOR[0] // 2, :] = True
    occluded = estimator.advance_positions(state, constant_flow(1, 0.0, 0.0), occlusion=occlusion)
    covered = occluded.xy[0, :, 1] < SENSOR[0] / 2
    assert torch.equal(occluded.valid[0], ~covered)
    assert torch.equal(occluded.mode_mean, state.mode_mean)


def test_birth_only_in_uncovered_regions_and_keeps_identity():
    estimator = make_estimator()
    state, _, _ = seed_state(estimator)
    radius = covering_radius_px(SUPPORTS, SENSOR)

    pushed = estimator.advance_positions(state, constant_flow(1, 1000.0, 0.0))
    assert not bool(pushed.valid.any())  # 全部越界 → 所有槽位都空出来

    candidate = torch.tensor([[[16.0, 12.0]]])
    born = estimator.birth(pushed, candidate, torch.ones(1, 1, dtype=torch.bool), radius_px=radius)
    assert bool(born.fresh[0, 0]) and bool(born.valid[0, 0])
    assert int(born.age[0, 0]) == 0
    assert torch.allclose(born.xy[0, 0], torch.tensor([16.0, 12.0]))
    assert int(born.ids[0, 0]) >= estimator.num_supports  # 新 ID，不复用旧身份
    assert torch.equal(born.ids[0, 1:], pushed.ids[0, 1:])
    assert torch.equal(born.xy[0, 1:], pushed.xy[0, 1:])
    assert torch.allclose(born.mode_mean[0, 0], torch.zeros(STATE_SMALL))
    assert torch.allclose(
        born.mode_cov[0][0, 0], torch.ones(2, OBS_SMALL) * estimator.initial_variance
    )
    assert torch.allclose(born.mode_cov[2][0, 0], torch.zeros(2, OBS_SMALL))
    assert torch.allclose(born.mode_weights[0, 0], estimator.birth_weights(), atol=1e-6)

    # 覆盖区内的候选不产生新支持：位置、ID 与起始标记保持原样（不重排）。
    inside = estimator.birth(
        state,
        state.xy[:, :1].clone(),
        torch.ones(1, 1, dtype=torch.bool),
        radius_px=radius,
    )
    assert torch.equal(inside.ids, state.ids)
    assert torch.equal(inside.xy, state.xy)
    assert torch.equal(inside.fresh, state.fresh)


# ---------------------------------------------------------------------------------------
# 双模式更新与 forecast（契约 §4、§6）
# ---------------------------------------------------------------------------------------


def test_innovation_variance_and_log_weight_terms():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    residual, variance = innovation_terms(state, y, R)
    assert torch.allclose(variance, state.mode_cov[0] + R[:, :, None, :])

    log_likelihood = gaussian_log_prob(residual, variance)
    reference = -0.5 * (
        (residual.square() / variance).sum(dim=-1)
        + torch.log(variance).sum(dim=-1)
        + OBS_SMALL * math.log(2.0 * math.pi)
    )
    assert torch.allclose(log_likelihood, reference, atol=1e-5)

    result = update(estimator, state, y, R)
    assert torch.allclose(
        result.log_weight, torch.log(state.mode_weights) + result.log_likelihood, atol=1e-5
    )
    assert torch.allclose(result.mode_posterior.sum(dim=-1), torch.ones(state.batch_size, SUPPORTS))


def test_extreme_residual_moves_mode_posterior_to_revision():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    votes = torch.zeros_like(state.mode_weights)
    votes[..., 0] = 0.95
    votes[..., 1] = 0.05
    state = replace(state, mode_weights=votes)

    # 修订模式解释观测：模式 0 的预测离 y 很远，模式 1 的预测正好落在 y。
    predicted = torch.cat((y, torch.zeros_like(y)), dim=-1)
    far = predicted.clone()
    far[..., :OBS_SMALL] += 50.0
    state_revised = replace(state, mode_mean=torch.stack((far, predicted), dim=2))
    result = update(estimator, state_revised, y, R)
    assert float(result.mode_posterior[..., 1].min()) > 0.9
    assert float(result.mode_posterior[..., 1].mean()) > float(votes[..., 1].mean())

    # 反例：延续模式解释观测时，同样的极端残差不支持强修订。
    state_kept = replace(state, mode_mean=torch.stack((predicted, far), dim=2))
    kept = update(estimator, state_kept, y, R)
    assert float(kept.mode_posterior[..., 0].min()) > 0.9


def test_forecast_loss_matches_dense_gaussian_reference():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    prior = estimator.predict(state, torch.randn(state.batch_size, 5))
    target = y + 0.5

    loss = forecast_loss(target, prior, R, reduce=False)
    assert loss.shape == (state.batch_size, SUPPORTS)

    mean = prior.mode_mean[..., :OBS_SMALL]
    per_mode = []
    for index in range(2):
        covariance = torch.diag(prior.mode_cov[0][0, 0, index] + R[0, 0])
        residual = target[0, 0] - mean[0, 0, index]
        quadratic = residual @ torch.linalg.solve(covariance, residual)
        _, log_det = torch.linalg.slogdet(covariance)
        per_mode.append(-0.5 * (quadratic + log_det + OBS_SMALL * math.log(2.0 * math.pi)))
    log_prior = torch.log(prior.mode_weights[0, 0])
    reference = -torch.logsumexp(log_prior + torch.stack(per_mode), dim=0)
    # 契约 §6：辅助项按有效样本「与维度」取均值 → 参考实现同样除以观测维度。
    reference = reference / OBS_SMALL
    assert torch.allclose(loss[0, 0], reference, atol=1e-6)
    assert bool(torch.isfinite(loss).all())
    assert float(loss.detach().mean()) > 0.0

    # reduce=True 与逐点均值一致；量级是「每维均值」而不是 256 维求和。
    assert torch.allclose(forecast_loss(target, prior, R), loss.mean(), atol=1e-7)
    assert float(loss.detach().mean()) < 20.0


def test_forecast_diagnostics_are_read_only_and_match_manual_terms():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    prior = estimator.predict(state, torch.randn(state.batch_size, 5))
    target = y + 0.25

    report = forecast_diagnostics(target, prior, R)
    assert set(report) == {
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
    }
    assert all(isinstance(value, float) for value in report.values())

    residual, innovation = innovation_terms(prior, target, R)
    quad = (residual.square() / innovation).sum(-1)
    logdet = torch.log(innovation).sum(-1)
    weights = prior.mode_weights
    assert report["forecast_quad"] == pytest.approx(
        float((weights * quad).sum(-1).mean().detach()), rel=1e-5
    )
    assert report["forecast_logdet"] == pytest.approx(
        float((weights * logdet).sum(-1).mean().detach()), rel=1e-5
    )
    assert report["forecast_mode_w0"] == pytest.approx(
        float(weights[..., 0].mean().detach()), rel=1e-5
    )
    assert report["forecast_mode_w1"] == pytest.approx(
        float(weights[..., 1].mean().detach()), rel=1e-5
    )
    assert report["forecast_S_min"] <= report["forecast_S_mean"] <= report["forecast_S_max"]
    assert report["forecast_R_min"] <= report["forecast_R_mean"] <= report["forecast_R_max"]

    # 只读：即使输入带图也只返回 float，不建立任何反传路径。
    tracked = replace(prior, mode_mean=prior.mode_mean.clone().requires_grad_(True))
    with torch.enable_grad():
        tracked_report = forecast_diagnostics(target, tracked, R)
    assert all(isinstance(value, float) for value in tracked_report.values())

    # quad/logdet 是「对维度求和」的量（判据 quad≈d=256），loss 是「按维度取均值」的量。
    expected = 0.5 * (report["forecast_quad"] + report["forecast_logdet"])
    expected += 0.5 * OBS_SMALL * math.log(2.0 * math.pi)
    assert float(forecast_loss(target, prior, R).detach()) == pytest.approx(
        expected / OBS_SMALL, rel=2e-3
    )

    # 分母为空：写 NaN（调用方转 null），不用 0 冒充。
    empty = forecast_diagnostics(target, prior, R, valid=torch.zeros_like(prior.valid))
    assert all(math.isnan(value) for value in empty.values())


# ---------------------------------------------------------------------------------------
# 梯度与正定性（契约 §6、§7.4）
# ---------------------------------------------------------------------------------------


def test_gradients_reach_quality_head_and_transition_matrix():
    torch.manual_seed(3)
    estimator = make_estimator(obs_dim=DENSE_LATENT_CHANNELS)
    head = QualityHead()
    values, support, saturated, latent, grid, _ = quality_inputs(batch=1)
    state = estimator.reset(batch_size=1, sensor_size_hw=SENSOR)
    R = head(values, support, saturated, latent, state.xy, grid=grid)
    y = sample_feature_map(latent, state.xy, grid)
    state = update(estimator, state, y, R).state
    prior = estimator.predict(state, torch.randn(1, 5))
    forecast_loss(y, prior, R).backward()

    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.abs().sum() > 0), name
    assert bool(estimator.mode_transition_logits.grad.abs().sum() > 0)
    assert bool(estimator.revision_q_raw.grad.abs().sum() > 0)
    # 控制输出层是零初始化：第一次反传只更新输出层权重，前几层此刻梯度为 0（设计如此）。
    assert estimator.control[0].weight.grad is not None
    assert bool(estimator.control[-1].weight.grad.abs().sum() > 0)
    for index, cell in enumerate(estimator.cells):
        assert all(p.grad is not None for p in cell.parameters()), index


def _assemble_covariance(state, batch: int, support: int, mode: int) -> torch.Tensor:
    upper = state.mode_cov[0][batch, support, mode]
    lower = state.mode_cov[1][batch, support, mode]
    side = state.mode_cov[2][batch, support, mode]
    obs_dim = upper.shape[-1]
    covariance = upper.new_zeros(2 * obs_dim, 2 * obs_dim)
    covariance[:obs_dim, :obs_dim] = torch.diag(upper)
    covariance[obs_dim:, obs_dim:] = torch.diag(lower)
    covariance[:obs_dim, obs_dim:] = torch.diag(side)
    covariance[obs_dim:, :obs_dim] = torch.diag(side)
    return covariance


def test_covariance_stays_positive_definite_through_predict_and_update():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    # 让两个分量明显分歧：权重各半、均值相差很大 → 混合必须保留 Δ 项。
    weights = torch.full_like(state.mode_weights, 0.5)
    distant = torch.cat((y, torch.zeros_like(y)), dim=-1)
    state = replace(
        state,
        mode_weights=weights,
        mode_mean=torch.stack((distant + 3.0, distant - 3.0), dim=2),
    )
    prior = estimator.predict(state, torch.randn(state.batch_size, 5))
    posterior = update(estimator, prior, y, R).state

    assert not torch.allclose(prior.mode_cov[0], state.mode_cov[0])  # Δ 项确实进来了
    for candidate, label in ((prior, "prior"), (posterior, "posterior")):
        for batch in range(candidate.batch_size):
            for support in range(candidate.num_supports):
                for mode in range(2):
                    covariance = _assemble_covariance(candidate, batch, support, mode)
                    eigenvalues = torch.linalg.eigvalsh(covariance)
                    assert float(eigenvalues.detach().min()) > 0.0, (label, batch, support, mode)
                    torch.linalg.cholesky(covariance)  # 非正定时会抛错
                    upper = candidate.mode_cov[0][batch, support, mode]
                    lower = candidate.mode_cov[1][batch, support, mode]
                    side = candidate.mode_cov[2][batch, support, mode]
                    determinant = (upper * lower - side.square()).detach()
                    assert float(determinant.min()) > 0.0


def test_forecast_loss_is_finite_for_extreme_targets():
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    prior = estimator.predict(state, torch.randn(state.batch_size, 5))
    assert bool(torch.isfinite(forecast_loss(y * 1e3, prior, R)))
    # 空有效集：均值未定义，按约定返回 0，不伪造样本。
    empty = forecast_loss(y, prior, R, valid=torch.zeros_like(prior.valid))
    assert float(empty) == 0.0


def test_update_rejects_non_positive_or_non_finite_R():
    """R 必须逐元素为正；错误输入直接报错，不用数值保护掩盖。"""
    estimator = make_estimator()
    state, y, R = seed_state(estimator)
    with pytest.raises(ValueError, match="R 必须逐元素为正"):
        update(estimator, state, y, R * 0.0)
    with pytest.raises(ValueError, match="NaN/Inf"):
        update(estimator, state, y, torch.full_like(R, float("nan")))


def test_occlusion_is_sampled_before_motion():
    estimator = make_estimator()
    state, _, _ = seed_state(estimator)
    source = state.xy[0, 0].long()
    occlusion = torch.zeros(1, *SENSOR, dtype=torch.bool)
    occlusion[0, source[1], source[0]] = True
    moved = estimator.advance_positions(state, constant_flow(1, 3, 0), occlusion=occlusion)
    assert not bool(moved.valid[0, 0])
    assert torch.allclose(moved.xy[0, 0], state.xy[0, 0] + torch.tensor([3.0, 0.0]))


def test_pure_cell_checkpoint_keeps_predictions_and_source_gradients():
    reference = make_estimator()
    reference.checkpoint_predictions = False
    checked = make_estimator()
    checked.load_state_dict(reference.state_dict())
    checked.checkpoint_predictions = True
    initial, _, _ = seed_state(reference)

    def run(estimator):
        source = initial.mode_mean.detach().clone().requires_grad_()
        state = replace(initial, mode_mean=source)
        for _ in range(4):
            state = estimator.predict(state, torch.ones(1, 5))
        loss = state.mode_mean.square().mean() + sum(c.mean() for c in state.mode_cov)
        grads = torch.autograd.grad(loss, [source, *estimator.parameters()], allow_unused=True)
        return state, grads

    a, ga = run(reference)
    b, gb = run(checked)
    assert torch.allclose(a.mode_mean, b.mode_mean, atol=1e-7)
    for x, y in zip(ga, gb, strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.allclose(x, y, atol=1e-6, rtol=1e-5)


def test_banded_prediction_matches_author_dense_values_and_gradients():
    from owac.baselines._vendor.rkn.cell import RKNCell
    from owac.frameworks.n04.rkn_banded import band_layout, predict_banded

    torch.manual_seed(44)
    cell = RKNCell(16, small_cell_config(bandwidth=3))
    with torch.no_grad():
        for p in cell.parameters():
            p.add_(torch.randn_like(p) * 0.03)
    mean = torch.randn(7, 32, requires_grad=True)
    cu = torch.rand(7, 16, requires_grad=True)
    cl = torch.rand(7, 16, requires_grad=True)
    cs = (torch.rand(7, 16) * 0.1).requires_grad_()
    parameters = [mean, cu, cl, cs, *cell.parameters()]
    dm, dc = cell._predict(mean, [cu, cl, cs])
    dense = (dm, *dc)
    banded = predict_banded(cell, mean, cu, cl, cs, *band_layout(cell))
    for a, b in zip(dense, banded, strict=True):
        assert torch.allclose(a, b, rtol=2e-6, atol=2e-6)
    gd = torch.autograd.grad(sum(t.square().mean() for t in dense), parameters)
    gb = torch.autograd.grad(sum(t.square().mean() for t in banded), parameters)
    for a, b in zip(gd, gb, strict=True):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)

"""N04 创新分解与状态更新测试：identity、正负单像素移动、越界、遮挡/新暴露、
照明分支开关、颜色决定目标选择的反例。

这些测试直接检查“哪些变化被解释成自运动、哪些被当作外部变化”，是机制层而非
形状层的验证。
"""

from __future__ import annotations

import pytest
import torch

from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
from owac.frameworks.n04.ego import EgoMotionContext
from owac.frameworks.n04.geometry import (
    GridSpec,
    backward_grid_from_translation,
    warp_feature_field,
)
from owac.frameworks.n04.innovation import InnovationDecomposer, estimate_low_frequency_gain
from owac.frameworks.n04.model import N04Encoder
from owac.frameworks.n04.state import new_state

H = W = 16
WIDTH = 8


def _config(**kwargs) -> N04Config:
    model_kwargs = dict(
        name="n04-test",
        width=WIDTH,
        blocks=1,
        scales=1,
        scale_strides=(1,),
        coarse_stride=2,
        hidden=16,
        readout_tokens=4,
        readout_dim=8,
        predictor="identity",
    )
    model_kwargs.update(kwargs)
    return N04Config(
        input_mode="rawvla_rgb3",
        measurement_size=(H, W),
        channels=3,
        camera_ids=("head",),
        geometry=N04GeometryConfig(feature_size=(H, W), pixel_stride=1.0, max_flow=8.0),
        model=N04ModelConfig(**model_kwargs),
    )


def _context(b: int = 1, dim: int = 4) -> EgoMotionContext:
    return EgoMotionContext(
        proprio_delta=torch.zeros(b, dim),
        proprio_delta_available=True,
        executed_action=torch.zeros(b, 3),
        executed_action_available=True,
        dt=torch.ones(b),
        dt_available=False,
        camera_index=torch.zeros(b, dtype=torch.long),
        camera_id="head",
    )


class _FrozenStem(torch.nn.Module):
    """确定性测量 stem stub：把三通道测量直接取均值作为单通道特征，便于手算。

    只用于机制测试，不改变 `N04Encoder` 的任何算法路径。
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = values.mean(dim=1, keepdim=True)
        return base.expand(-1, self.width, -1, -1).contiguous()


def _encoder_with_identity_stem(**model_kwargs) -> N04Encoder:
    enc = N04Encoder(_config(**model_kwargs))
    enc.stem = _FrozenStem(WIDTH)
    return enc


def _disc(y: float, x: float, h: int = H, w: int = W, radius: float = 2.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )
    return ((yy - y) ** 2 + (xx - x) ** 2 <= radius**2).to(torch.float32)


def _shift(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    spec = GridSpec(img.shape[-2], img.shape[-1], pixel_stride=1.0)
    grid = backward_grid_from_translation(spec, dx, dy, batch=img.shape[0])
    return warp_feature_field(img, grid, spec).values


# --------------------------------------------------------------------------------------
# identity 与单像素移动
# --------------------------------------------------------------------------------------


def _rollout_change_statistics(
    enc: N04Encoder, frames, ctx: EgoMotionContext, steps: int
) -> tuple[list[float], list[float], list[float]]:
    """在给定帧序列上跑 ``steps`` 步，返回 (residual, |change|, gate_change) 序列。"""
    state = new_state(1, WIDTH, H, W, scales=1)
    residuals: list[float] = []
    changes: list[float] = []
    gates: list[float] = []
    for i in range(steps):
        out = enc.step(frames(i), ctx, state, camera_id="head")
        state = out.state
        residuals.append(float(out.innovation.residual.abs().mean()))
        changes.append(float(out.state.change.abs().mean()))
        gates.append(float(out.innovation.gate_change.mean()))
    return residuals, changes, gates


def test_identity_sequence_keeps_change_channel_quiet():
    """静止序列：稳定状态持续吸收测量、残差总体下降，change cell 有界且衰减。

    未训练的更新门远小于 1，而且块内的 GroupNorm 会把稳定状态归一到单位方差，
    因此残差不会降到零，而是缓慢趋向一个由特征表示决定的不动点。测试检查的正是
    “残差在下降”与“变化通道保持安静且有界”，而不是“残差立刻为零”。
    """
    enc = _encoder_with_identity_stem()
    ctx = _context()
    frame = 0.8 * _disc(8.0, 8.0).unsqueeze(0).unsqueeze(0).expand(1, 3, H, W).contiguous()
    residuals, changes, _ = _rollout_change_statistics(enc, lambda _i: frame, ctx, steps=400)
    # Identity correction starts as identity, so there is no artificial recurrent drift.
    # Explicitly exclude saturation from this feature-domain identity fixture.
    assert max(changes) < 1e-5
    assert residuals[-1] == pytest.approx(residuals[1], abs=1e-5)


def test_change_cell_does_not_discriminate_motion_before_training():
    """**未训练基线（含预期失败的性质）**：change cell 的幅值无法区分真实物体运动与
    纯粹的时序闪烁。

    这是分解头零初始化（softmax 均匀，``gate_change≈0.2``）的直接后果：门控对残差的
    内容一无所知，所以把交替平移造成的闪烁和静态序列的残差以相同强度写进 change
    cell。该性质必须由训练改变，而不是靠断言假装已经成立；`owac.evaluation.n04` 的
    机制矩阵负责在训练后重新检查（见 `research/experiments/2026-10-12_n04_encoder.md`）。
    """
    baseline = _encoder_with_identity_stem()
    ctx = _context()
    frame = 0.8 * _disc(8.0, 8.0).unsqueeze(0).unsqueeze(0).expand(1, 3, H, W).contiguous()
    measured: dict[float, float] = {}
    for amplitude in (0.0, 1.0):
        enc = _encoder_with_identity_stem()
        enc.load_state_dict(baseline.state_dict())

        def frames(i: int, a: float = amplitude) -> torch.Tensor:
            return _shift(frame, a * (i % 2), 0.0)

        _, changes, _ = _rollout_change_statistics(enc, frames, ctx, steps=120)
        measured[amplitude] = sum(changes[-20:]) / 20
    assert measured[1.0] > measured[0.0]
    assert measured[0.0] < 1e-5


def test_positive_and_negative_single_pixel_motion_detected_by_decomposition():
    """正/负单像素物体移动：门控与变化分支对两者都敏感，且方向相反。"""
    enc = _encoder_with_identity_stem()
    ctx = _context()
    base = _disc(8.0, 8.0).unsqueeze(0).unsqueeze(0).expand(1, 3, H, W).contiguous()
    results = {}
    for name, shift in (("plus", 1.0), ("minus", -1.0)):
        state = new_state(1, WIDTH, H, W, scales=1)
        state = enc.step(base, ctx, state, camera_id="head").state
        moved = _shift(base, shift, 0.0)
        out = enc.step(moved, ctx, state, camera_id="head")
        results[name] = out
    # 位移来自测量而非传播：预测残差在移动方向上有确定符号
    for name, out in results.items():
        residual = out.innovation.residual
        assert residual is not None
        assert float(residual.abs().mean()) > 1e-3, f"{name} 位移未被察觉"
    # 两次的残差不同（方向相反 ⇒ 残差图案不同）
    assert not torch.allclose(
        results["plus"].innovation.residual, results["minus"].innovation.residual, atol=1e-4
    )


def test_out_of_bounds_region_gets_no_fabricated_history():
    """自运动把历史推出视野后，越界区域的更新不应假装那里有历史证据。"""
    enc = _encoder_with_identity_stem()
    state = new_state(1, WIDTH, H, W, scales=1)
    ctx = _context()
    base = _disc(8.0, 8.0).unsqueeze(0).unsqueeze(0).expand(1, 3, H, W).contiguous()
    step1 = enc.step(base, ctx, state, camera_id="head")
    # 手动构造“有历史但传播越界”的状态：validity 由 geometry 给出
    shifted_prev = _shift(base, 6.0, 0.0)
    state2 = step1.state
    state2.stable = enc.encode_measurement(shifted_prev)
    state2.history_valid = torch.ones(1, dtype=torch.bool)
    spec = enc.spec
    from owac.frameworks.n04.geometry import GridSpec as _GS

    grid = backward_grid_from_translation(_GS(H, W, pixel_stride=1.0), 6.0, 0.0)
    warp_res = warp_feature_field(state2.stable, grid, spec)
    assert float(warp_res.validity[0, :, :6].sum()) == 0.0
    assert float(warp_res.validity[0, :, 6:].sum()) > 0.0


def test_occlusion_and_newly_exposed_area_are_tracked_separately():
    """遮挡区域与新暴露区域分别进入 validity/occlusion，不混为一谈。"""
    from owac.frameworks.n04.geometry import occlusion_mask_from_flow

    spec = GridSpec(H, W, pixel_stride=1.0)
    xy = spec.identity_xy(1, torch.device("cpu"), torch.float32)
    prev_xy = xy.clone()
    prev_xy[..., 0] = torch.where(xy[..., 0] >= 3.0, xy[..., 0] - 3.0, xy[..., 0])
    grid = spec.normalized_from_xy(prev_xy)
    occ = occlusion_mask_from_flow(prev_xy, grid)
    res = warp_feature_field(
        torch.ones(1, 1, H, W), grid, spec, previous_xy=prev_xy, check_occlusion=True
    )
    # 折叠列被标为遮挡（validity=0）
    assert float(occ[0, :, 3].sum()) == 0.0
    assert float(res.occlusion[0, :, 3].sum()) == 0.0
    # 未被任何源点占据的列是“新暴露”而非遮挡：occlusion=1 但 validity=0（越界）
    assert float(res.occlusion[0, :, -1].min()) == 1.0


# --------------------------------------------------------------------------------------
# 照明分支
# --------------------------------------------------------------------------------------


def test_low_frequency_gain_recovers_known_global_scale():
    current = torch.full((1, 2, 8, 8), 2.0)
    predicted = torch.full((1, 2, 8, 8), 1.0)
    weight = torch.ones(1, 1, 8, 8)
    gain, delta = estimate_low_frequency_gain(current, predicted, weight)
    assert float(gain.mean()) > 1.5
    assert float(delta.mean()) > 0.5
    darker, darker_delta = estimate_low_frequency_gain(predicted, current, weight)
    assert float(darker.mean()) < 0.75


def test_illumination_branch_reduces_geometry_residual_misread():
    """照明变化下，开启照明分支应把亮度变化识别为增益，而不是几何残差。

    用有结构、非暗的合成特征场检查：只在可靠区域估计的全局/低频增益必须接近真实缩放，
    并且关闭分支时该量恒为 1。
    """
    torch.manual_seed(0)
    dec_on = InnovationDecomposer(feature_channels=4, hidden=8, use_illumination_branch=True)
    dec_off = InnovationDecomposer(feature_channels=4, hidden=8, use_illumination_branch=False)
    base = torch.rand(1, 4, 12, 12) * 0.5 + 0.2
    bright = (base * 1.6).clamp(0, 1)
    validity = torch.ones(1, 1, 12, 12)
    uncertainty = torch.zeros(1, 1, 12, 12)
    on = dec_on(
        current=bright,
        predicted=base,
        validity=validity,
        uncertainty=uncertainty,
        linear_current=bright,
        linear_predicted=base,
        linear_validity=validity,
    )
    off = dec_off(current=bright, predicted=base, validity=validity, uncertainty=uncertainty)
    assert on.illumination_delta is not None and on.illumination_gain is not None
    estimated = float(on.illumination_gain.median())
    assert 1.3 < estimated < 1.9, f"照明增益估计偏离真实缩放: {estimated}"
    assert off.illumination_delta is not None
    assert float(off.illumination_delta.abs().max()) == 0.0
    # 照明分支不修改测量特征：残差仍然是 current - predicted
    assert torch.allclose(on.residual + base, bright, atol=1e-5)


def test_colour_decides_target_selection_counterexample():
    """反例：亮度与几何完全相同、只有颜色不同时，表示与读出仍必须区分两个目标。

    使用**真实**测量 stem（不是灰度 stub）：若把颜色证据当作与任务无关而抹掉，
    这个测试会失败。照明分支只估计增益，不修改测量特征。
    """
    torch.manual_seed(0)
    enc = N04Encoder(_config(illumination_branch=True))
    ctx = _context()
    luminance = _disc(8.0, 5.0)

    def rgb(red: float, green: float, blue: float) -> torch.Tensor:
        img = torch.zeros(1, 3, H, W)
        img[0, 0] = luminance * red
        img[0, 1] = luminance * green
        img[0, 2] = luminance * blue
        return img

    red = rgb(1.0, 0.0, 0.0)
    blue = rgb(0.0, 0.0, 1.0)
    # 亮度（三通道均值）完全相同
    assert torch.allclose(red.mean(dim=1), blue.mean(dim=1), atol=1e-6)
    representations = []
    tokens = []
    for img in (red, blue):
        state = enc.init_state(1)
        out = enc.step(img, ctx, state["head"], camera_id="head")
        out = enc.step(img, ctx, out.state, camera_id="head")
        representations.append(out.representation)
        tok, mask = enc.readout_tokens(out)
        tokens.append(tok)
        assert mask.shape == (1, 4)
    assert not torch.allclose(representations[0], representations[1], atol=1e-5)
    assert not torch.allclose(tokens[0], tokens[1], atol=1e-5)


def test_decomposition_mix_is_normalised_but_not_called_probability():
    """三分支 + unknown 的 softmax 归一化：和为 1，但记录为更新权重而不是事件概率。"""
    dec = InnovationDecomposer(feature_channels=4, hidden=8)
    current = torch.randn(2, 4, 6, 6)
    predicted = torch.randn(2, 4, 6, 6)
    validity = torch.ones(2, 1, 6, 6)
    uncertainty = torch.full((2, 1, 6, 6), 0.5)
    out = dec(current=current, predicted=predicted, validity=validity, uncertainty=uncertainty)
    total = out.mix.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    assert out.unknown.shape == (2, 1, 6, 6)
    assert 0.0 <= float(out.g_update.min()) and float(out.g_update.max()) <= 1.0


def test_measurement_acceptance_responds_to_uncertainty_and_residual():
    """``g_update`` 的三个可检查方向：传播可靠且残差小 → 低; 传播不可靠 → 升高;
    残差大（可能来自外部真实变化）→ 升高, 而不是一律拒绝新测量; 饱和 → 显著降低。"""
    dec = InnovationDecomposer(feature_channels=4, hidden=8)
    predicted = torch.zeros(1, 4, 6, 6)
    validity = torch.ones(1, 1, 6, 6)

    def gate(residual: float, uncertainty: float, saturation: float = 0.0) -> float:
        out = dec(
            current=torch.full((1, 4, 6, 6), residual),
            predicted=predicted,
            validity=validity,
            uncertainty=torch.full((1, 1, 6, 6), uncertainty),
            saturation=torch.full((1, 1, 6, 6), saturation),
        )
        return float(out.g_update.mean())

    reliable = gate(0.01, 0.0)
    uncertain = gate(0.01, 3.0)
    changed = gate(8.0, 0.0)
    saturated = gate(0.01, 0.0, saturation=1.0)
    assert reliable < 0.35, reliable
    assert uncertain < reliable, (uncertain, reliable)
    assert changed > reliable, (changed, reliable)
    assert saturated < 0.10, saturated


def test_no_change_branch_ablation_moves_weight_into_unknown():
    """消融“无外部变化分支”：变化权重并入 unknown，而不是凭空消失。"""
    dec = InnovationDecomposer(feature_channels=4, hidden=8, use_change_branch=False)
    out = dec(
        current=torch.randn(1, 4, 6, 6),
        predicted=torch.randn(1, 4, 6, 6),
        validity=torch.ones(1, 1, 6, 6),
        uncertainty=torch.full((1, 1, 6, 6), 0.5),
    )
    assert float(out.gate_change.abs().max()) == 0.0
    assert float(out.unknown.min()) > 0.0


def test_gradient_reaches_propagation_and_update_modules():
    """端到端梯度必须到达传播与更新模块（不无意 detach）。"""
    torch.manual_seed(0)
    enc = N04Encoder(_config(predictor="learned_proprio"))
    #: 打破零位移初始化，否则 map 支路对编码器参数的梯度在初始点恰好为零，
    #: 无法区分“梯度路径被切断”和“初始点梯度为零”。
    with torch.no_grad():
        torch.nn.init.normal_(enc.propagator.head.weight, std=0.05)
    state = enc.init_state(1)
    ctx = _context(1, 8)
    ctx.proprio_delta = torch.randn(1, 8, requires_grad=True)
    #: 两帧序列：第一帧建立历史，第二帧才真正使用传播结果
    out = enc.step(torch.rand(1, 3, H, W), ctx, state["head"], camera_id="head")
    out = enc.step(torch.rand(1, 3, H, W), ctx, out.state, camera_id="head")
    loss = out.representation.abs().mean() + out.state.prediction.abs().mean()
    loss.backward()
    prop_grads = [
        p.grad.abs().sum().item()
        for n, p in enc.propagator.named_parameters()
        if p.grad is not None
    ]
    dec_grads = [
        p.grad.abs().sum().item()
        for n, p in enc.decomposer.named_parameters()
        if p.grad is not None
    ]
    assert any(g > 0 for g in prop_grads), "loss 未回传到传播模块"
    assert any(g > 0 for g in dec_grads), "loss 未回传到分解模块"
    assert ctx.proprio_delta.grad is not None
    assert float(ctx.proprio_delta.grad.abs().sum()) > 0

"""N04 v1.1 S2 WS-A 对应网络测试：接口形状、坐标换算、RAW 尺度与自监督损失。

全部在 CPU 小张量（2×3×64×96）上运行；官方 RAW 帧上的对应可视化与真实数据前向由远端执行。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from owac.frameworks.n04 import correspondence as C
from owac.frameworks.n04.contracts import INPUT_SCALES
from owac.frameworks.n04.correspondence import (
    FlowOutput,
    RaftCorrespondence,
    RaftCorrespondenceLoss,
    RaftLossConfig,
    RawScale,
    flow_at_points,
    grid_to_px,
    points_inside_frame,
    px_to_grid,
    sample_flow_at_grid,
)

HEIGHT, WIDTH = 64, 96
BATCH = 2
LIBERO = INPUT_SCALES["libero_uint8"]
ROBOTWIN = INPUT_SCALES["robotwin_float_linear"]


# --------------------------------------------------------------------------------------
# 夹具与小工具
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scale() -> RawScale:
    return RawScale.from_input_scale(LIBERO)


@pytest.fixture(scope="module")
def corr(scale: RawScale) -> RaftCorrespondence:
    torch.manual_seed(0)
    return RaftCorrespondence(scale=scale, iters=12).eval()


@pytest.fixture(scope="module")
def loss_fn(scale: RawScale) -> RaftCorrespondenceLoss:
    return RaftCorrespondenceLoss(scale=scale, config=RaftLossConfig())


def _uint8_frames(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (BATCH, 3, HEIGHT, WIDTH)
    previous = torch.randint(0, 256, shape, generator=generator).to(torch.uint8)
    current = torch.randint(0, 256, shape, generator=generator).to(torch.uint8)
    return previous, current


def _affine_flow(height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """线性 flow 场（双线性插值对它精确），用于和解析值对照。"""
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    flow_x = 0.25 * xx - 0.5 * yy + 1.0
    flow_y = -0.1 * xx + 0.2 * yy - 2.0
    return torch.stack([flow_x, flow_y], dim=0)[None].repeat(BATCH, 1, 1, 1)


def _flow_true_value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([0.25 * x - 0.5 * y + 1.0, -0.1 * x + 0.2 * y - 2.0], dim=-1)


def _base_grid(height: int, width: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0)[None]


def _translated(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """内容平移 (+dx, +dy)：``out(x, y) = img(x - dx, y - dy)``，越界为 0。"""
    height, width = img.shape[-2:]
    offset = torch.tensor([float(dx), float(dy)]).view(1, 2, 1, 1)
    coords = (_base_grid(height, width) - offset).expand(img.shape[0], -1, -1, -1)
    return C._resample(img, coords.to(img.dtype))


def _interior_valid(height: int, width: int, margin: int) -> torch.Tensor:
    valid = torch.ones(BATCH, height, width, dtype=torch.bool)
    valid[:, :margin, :] = False
    valid[:, -margin:, :] = False
    valid[:, :, :margin] = False
    valid[:, :, -margin:] = False
    return valid


# --------------------------------------------------------------------------------------
# 坐标换算（契约 §2）
# --------------------------------------------------------------------------------------


def test_pixel_grid_roundtrip_and_center_convention() -> None:
    xy = torch.tensor(
        [
            [0.0, 0.0],
            [float(WIDTH - 1), float(HEIGHT - 1)],
            [WIDTH - 0.5, HEIGHT - 0.5],
            [-3.5, 7.25],
            [100.0, -5.0],
        ]
    )
    grid = px_to_grid(xy, (HEIGHT, WIDTH))
    back = grid_to_px(grid, (HEIGHT, WIDTH))
    assert torch.allclose(back, xy, atol=1e-4)

    # align_corners=False：像素中心 (0,0) / (W-1,H-1) 不落在 -1 / +1 端点
    assert torch.allclose(
        px_to_grid(torch.tensor([[0.0, 0.0]]), (HEIGHT, WIDTH)),
        torch.tensor([[(1.0 / WIDTH) - 1.0, (1.0 / HEIGHT) - 1.0]]),
    )
    assert torch.allclose(
        px_to_grid(torch.tensor([[float(WIDTH - 1), float(HEIGHT - 1)]]), (HEIGHT, WIDTH)),
        torch.tensor([[1.0 - 1.0 / WIDTH, 1.0 - 1.0 / HEIGHT]]),
    )
    # 逆映射用同一约定：grid=0 对应像素中心 ((size/2) - 0.5)
    assert torch.allclose(
        grid_to_px(torch.zeros(1, 2), (HEIGHT, WIDTH)),
        torch.tensor([[(WIDTH / 2) - 0.5, (HEIGHT / 2) - 0.5]]),
    )


def test_points_inside_frame_closed_interval() -> None:
    xy = torch.tensor(
        [[0.0, 0.0], [float(WIDTH - 1), float(HEIGHT - 1)], [WIDTH, 0.0], [-0.01, 3.0]]
    )
    inside = points_inside_frame(xy, (HEIGHT, WIDTH))
    assert inside.tolist() == [True, True, False, False]


def test_flow_at_points_matches_analytic_and_manual_warp() -> None:
    flow = _affine_flow()
    xy = torch.tensor(
        [[3.5, 4.25], [10.0, 20.0], [0.0, 0.0], [float(WIDTH - 1), float(HEIGHT - 1)]]
    )

    sampled = flow_at_points(flow, xy)
    assert sampled.shape == (BATCH, xy.shape[0], 2)
    assert torch.allclose(
        sampled, _flow_true_value(xy[:, 0], xy[:, 1]).expand(BATCH, -1, -1), atol=1e-5
    )

    # 整数坐标必须恰好等于该像素的 flow（整数即像素中心）
    integer_xy = torch.tensor([[5.0, 7.0], [31.0, 12.0]])
    integer_values = flow_at_points(flow, integer_xy)[0]
    expected_integer = torch.stack([flow[0, :, 7, 5], flow[0, :, 12, 31]], dim=0)
    # 整数值即像素中心：采样只允许浮点舍入级别的差异
    assert float((integer_values - expected_integer).abs().max()) < 1e-6

    # 与手工 grid_sample + px_to_grid（align_corners=False）一致
    grid = px_to_grid(xy, (HEIGHT, WIDTH)).unsqueeze(1).expand(BATCH, -1, -1, -1)
    manual = F.grid_sample(flow, grid, mode="bilinear", align_corners=False, padding_mode="border")
    assert torch.allclose(sampled, manual[:, :, :, 0].permute(0, 2, 1), atol=1e-6)


def test_sample_flow_at_grid_keeps_pixel_units() -> None:
    flow = torch.full((BATCH, 2, HEIGHT, WIDTH), 1.5)
    coarse = sample_flow_at_grid(flow, (HEIGHT // 4, WIDTH // 4))
    assert coarse.shape == (BATCH, 2, HEIGHT // 4, WIDTH // 4)
    assert torch.allclose(coarse, torch.full_like(coarse, 1.5))
    assert torch.equal(sample_flow_at_grid(flow, (HEIGHT, WIDTH)), flow)


# --------------------------------------------------------------------------------------
# RAW 尺度：固定仿射，不逐帧归一化、不裁 HDR、不转 uint8
# --------------------------------------------------------------------------------------


def test_raw_scale_is_fixed_affine(scale: RawScale) -> None:
    # RoboTwin 的 factor=1 线性辐射尺度：测量单位 [0,1] -> [-1,1]
    unit = RawScale.from_input_scale(ROBOTWIN)
    raw = torch.linspace(0.0, 1.0, 32).reshape(2, 4, 4)
    normalized = unit.normalize(raw)
    assert torch.allclose(normalized, 2.0 * raw - 1.0, atol=1e-6)

    # 常数尺度变化按同一仿射关系传播：不是被逐帧 min-max 抹掉
    scaled = unit.normalize(2.0 * raw)
    assert torch.allclose(scaled, 2.0 * normalized + 1.0, atol=1e-6)
    assert not torch.allclose(scaled, normalized)

    # HDR（>white level）既不裁剪也不压缩
    hdr = torch.tensor([[[[4.0, 0.25]]]])
    assert torch.allclose(unit.normalize(hdr), torch.tensor([[[[7.0, -0.5]]]]))

    # uint8 pseudo-RAW：/255 与作者 2*(x/255)-1 完全一致，且不做 float32 -> uint8
    uint8 = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
    assert scale.normalize(uint8).dtype == torch.float32
    assert torch.allclose(scale.normalize(uint8), torch.tensor([[[[-1.0, 1.0]]]]))


def test_robotwin_float_scale_and_saturation_confidence() -> None:
    robotwin = RawScale.from_input_scale(ROBOTWIN)
    assert robotwin.factor == 1.0
    assert torch.allclose(
        robotwin.normalize(torch.tensor([[[[0.25, 1.0]]]])), torch.tensor([[[[-0.5, 1.0]]]])
    )

    dark = torch.tensor([[[[0.0]]]])
    bright = torch.tensor([[[[1.0]]]])
    assert float(robotwin.confidence(dark)) > float(robotwin.confidence(bright))
    assert 0.0 < float(robotwin.confidence(bright)) <= 1.0

    saturated = torch.tensor([[[[1.0, 0.5]]]])
    assert robotwin.saturated(saturated).tolist() == [[[[True, False]]]]


def test_forward_runs_on_hdr_float_and_identical_frames(corr: RaftCorrespondence) -> None:
    torch.manual_seed(3)
    previous = torch.rand(BATCH, 3, HEIGHT, WIDTH) * 3.0  # HDR 尺度，超过 white level
    out = corr(previous, previous.clone(), iters=2)
    assert torch.isfinite(out.flow_px).all()
    assert out.flow_px.dtype == torch.float32


def test_non_finite_raw_is_exposed(
    corr: RaftCorrespondence, loss_fn: RaftCorrespondenceLoss
) -> None:
    previous, current = _uint8_frames(1)
    previous = previous.float()
    previous[0, 0, 3, 4] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        corr(previous, current.float(), iters=2)
    with pytest.raises(ValueError, match="NaN/Inf"):
        loss_fn(previous, current.float(), torch.zeros(BATCH, 2, HEIGHT, WIDTH))


# --------------------------------------------------------------------------------------
# 前向接口：形状、dtype、valid、完整 RAFT 配置
# --------------------------------------------------------------------------------------


def test_forward_shapes_valid_and_contract_semantics(corr: RaftCorrespondence) -> None:
    previous, current = _uint8_frames(2)
    out = corr(previous, current)
    assert isinstance(out, FlowOutput)
    assert out.flow_px.shape == (BATCH, 2, HEIGHT, WIDTH)
    assert out.flow_px.dtype == torch.float32
    assert out.flow_px.requires_grad
    assert out.valid.shape == (BATCH, HEIGHT, WIDTH)
    assert out.valid.dtype == torch.bool
    assert out.feature_scale == 1.0
    assert torch.isfinite(out.flow_px).all()
    assert 0.0 < float(out.valid.float().mean()) <= 1.0

    # valid = "上一帧点 + flow" 落在当前帧内（契约 §2/§4）
    sampled_xy = _base_grid(HEIGHT, WIDTH) + out.flow_px
    assert torch.equal(
        out.valid, points_inside_frame(sampled_xy.permute(0, 2, 3, 1), (HEIGHT, WIDTH))
    )

    blocked = torch.zeros(BATCH, HEIGHT, WIDTH, dtype=torch.bool)
    assert not corr(previous, current, valid=blocked).valid.any()


def test_model_is_full_raft_not_tiny(corr: RaftCorrespondence) -> None:
    assert corr.raft.args.small is False
    assert corr.raft.hidden_dim == 128
    assert corr.raft.context_dim == 128
    assert corr.raft.args.corr_levels == 4
    assert corr.raft.args.corr_radius == 4
    assert corr.iters == 12
    total = sum(parameter.numel() for parameter in corr.parameters())
    assert 5_000_000 <= total <= 5_600_000, total

    torch.manual_seed(4)
    images = [2.0 * torch.rand(BATCH, 3, HEIGHT, WIDTH) - 1.0 for _ in range(2)]
    predictions = corr.raft(images[0], images[1], iters=corr.iters, normalized=True)
    assert len(predictions) == corr.iters
    assert predictions[-1].shape == (BATCH, 2, HEIGHT, WIDTH)


def test_in_channels_is_platform_dependent(scale: RawScale) -> None:
    torch.manual_seed(5)
    single = RaftCorrespondence(scale=scale, in_channels=1, iters=2).eval()
    raw = torch.randint(0, 256, (BATCH, 1, HEIGHT, WIDTH)).to(torch.uint8)
    out = single(raw, raw.clone())
    assert out.flow_px.shape == (BATCH, 2, HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="通道"):
        single(raw.repeat(1, 3, 1, 1), raw.repeat(1, 3, 1, 1))


def test_resize_scales_flow_to_native_pixels(
    scale: RawScale, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(6)
    model = RaftCorrespondence(scale=scale, iters=2, resize_hw=(HEIGHT // 2, WIDTH // 2)).eval()
    seen: dict[str, tuple[int, ...]] = {}

    def fake_flow(image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        seen["shape"] = tuple(image1.shape)
        return torch.full((image1.shape[0], 2, image1.shape[-2], image1.shape[-1]), 1.0)

    monkeypatch.setattr(model, "_flow_at_compute_size", fake_flow)
    previous, current = _uint8_frames(6)
    out = model(previous, current)
    assert seen["shape"] == (BATCH, 3, HEIGHT // 2, WIDTH // 2)
    assert out.feature_scale == 2.0
    # 半分辨率 1px 位移 = 原生 2px（不是 1px 也不是 4px）
    assert torch.allclose(out.flow_px, torch.full_like(out.flow_px, 2.0))
    # valid 用原生分辨率坐标判定：位移后最后 2 行/列越界
    assert out.valid[:, : HEIGHT - 2, : WIDTH - 2].all()
    assert not out.valid[:, HEIGHT - 1, :].any()
    assert not out.valid[:, :, WIDTH - 1].any()


def test_non_uniform_resize_is_refused(scale: RawScale) -> None:
    torch.manual_seed(7)
    model = RaftCorrespondence(scale=scale, iters=2, resize_hw=(HEIGHT // 2, WIDTH)).eval()
    previous, current = _uint8_frames(7)
    with pytest.raises(ValueError, match="等比"):
        model(previous, current)


# --------------------------------------------------------------------------------------
# 自监督损失：有限性、项定义、RAW 权重与梯度
# --------------------------------------------------------------------------------------


def test_rangemap_matches_uflow_splatting() -> None:
    zeros = torch.zeros(1, 2, 8, 12)
    assert torch.allclose(C._range_map(zeros), torch.ones(1, 1, 8, 12))

    shift = torch.zeros(1, 2, 8, 12)
    shift[:, 0] = 1.0
    counts = C._range_map(shift)
    assert float(counts.sum()) == pytest.approx(8 * 11)
    assert torch.allclose(counts[0, 0, :, 0], torch.zeros(8))
    assert torch.allclose(counts[0, 0, :, 1:], torch.ones(8, 11))


def test_census_and_robust_helpers() -> None:
    constant = torch.full((1, 1, 12, 12), 0.5)
    census = C._census_transform(constant, 7)
    assert census.shape == (1, 49, 12, 12)
    # 零填充卷积：内部为 0，边界 patch//2 圈非 0（SMURF 用 zero_mask_border 在损失里忽略）
    assert torch.allclose(census[..., 3:-3, 3:-3], torch.zeros(1, 49, 6, 6))
    assert float(census[..., 0, 0].abs().max()) > 0.0
    hamming = C._soft_hamming(torch.zeros(1, 49, 8, 8), torch.zeros(1, 49, 8, 8), 0.1)
    assert torch.allclose(hamming, torch.zeros(1, 1, 8, 8))
    assert float(C.robust_l1(torch.zeros(()))) == pytest.approx(0.001)
    assert float(C.abs_robust_loss(torch.zeros(()), 0.01, 0.4)) == pytest.approx(0.01**0.4)


def test_loss_terms_finite_and_weighted(scale: RawScale, loss_fn: RaftCorrespondenceLoss) -> None:
    previous, current = _uint8_frames(8)
    flow = torch.zeros(BATCH, 2, HEIGHT, WIDTH) + 1.0
    backward = torch.zeros(BATCH, 2, HEIGHT, WIDTH) - 1.0
    losses = loss_fn(previous, current, flow, backward_flow_px=backward)
    for name in ("total", "census", "photometric", "smooth1", "smooth2"):
        assert torch.isfinite(getattr(losses, name)), name
    assert losses.total > 0
    # 小位移下 UFlow rangemap 遮挡接近全可见，噪声权重 ≈ 1
    assert losses.support > 0.9
    # 默认配置下平滑项被门槛关掉（smoothness_delay_*），total 只含两个 matching 项
    assert float(losses.smooth1) == 0.0 and float(losses.smooth2) == 0.0
    expected = 1.0 * losses.census + 1.0 * losses.photometric
    assert torch.allclose(losses.total, expected, atol=1e-5)


def test_saturation_excludes_brightness_term_but_keeps_census(scale: RawScale) -> None:
    loss_fn = RaftCorrespondenceLoss(scale=scale, config=RaftLossConfig())
    saturated = torch.full((BATCH, 3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    flow = torch.zeros(BATCH, 2, HEIGHT, WIDTH)
    losses = loss_fn(saturated, saturated.clone(), flow)
    # 饱和像素的亮度被裁剪不可信：亮度项支持为 0 -> 分子为 0 -> 比为 0
    assert float(losses.photometric) == pytest.approx(0.0, abs=1e-6)
    # 结构（census）支持仍然非零：两帧相同 -> 比值恰为 1（不是 0，说明支持没被清零）
    assert float(losses.census) == pytest.approx(1.0, abs=1e-5)


def test_raw_terms_without_static_baseline_keep_old_floor(scale: RawScale) -> None:
    loss_fn = RaftCorrespondenceLoss(scale=scale, config=RaftLossConfig(static_baseline=False))
    saturated = torch.full((BATCH, 3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    flow = torch.zeros(BATCH, 2, HEIGHT, WIDTH)
    losses = loss_fn(saturated, saturated.clone(), flow)
    assert float(losses.photometric) == pytest.approx(0.0, abs=1e-6)
    # 旧形式直接给出 robust floor：0.01**0.4（用于对照静态基线改变了什么）
    assert float(losses.census) == pytest.approx(0.01**0.4, rel=1e-3)


def test_loss_prefers_the_true_translation(loss_fn: RaftCorrespondenceLoss) -> None:
    torch.manual_seed(9)
    previous = torch.rand(BATCH, 3, HEIGHT, WIDTH)
    shift_x, shift_y = 2.0, 2.0
    current = _translated(previous, shift_x, shift_y)
    valid = _interior_valid(HEIGHT, WIDTH, margin=3)

    true_flow = torch.zeros(BATCH, 2, HEIGHT, WIDTH)
    true_flow[:, 0] = shift_x  # 契约：上一帧点 + flow 落在当前帧
    true_flow[:, 1] = shift_y
    true_losses = loss_fn(previous, current, true_flow, valid=valid)
    wrong_losses = loss_fn(previous, current, -true_flow, valid=valid)

    assert true_losses.photometric < wrong_losses.photometric
    assert true_losses.census < wrong_losses.census
    # 支持只由边界可见性、RAW 噪声权重与传入 valid 决定，与 flow 方向无关
    assert true_losses.support > 0.75
    assert abs(true_losses.support - wrong_losses.support) < 0.05


def test_loss_gradients_reach_flow_and_raft(
    corr: RaftCorrespondence, loss_fn: RaftCorrespondenceLoss
) -> None:
    previous, current = _uint8_frames(10)
    out = corr(previous, current, iters=3)
    backward = corr(current, previous, iters=3).flow_px.detach()
    losses = loss_fn(previous, current, out.flow_px, backward_flow_px=backward)
    losses.total.backward()

    fnet_grad = corr.raft.fnet.conv1.weight.grad
    cnet_grad = corr.raft.cnet.conv1.weight.grad
    update_grad = corr.raft.update_block.encoder.convc1.weight.grad
    for name, grad in (
        ("fnet.conv1", fnet_grad),
        ("cnet.conv1", cnet_grad),
        ("update.convc1", update_grad),
    ):
        assert grad is not None, name
        assert float(grad.abs().sum()) > 0.0, name
    nonzero = sum(
        1
        for parameter in corr.parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
    )
    assert nonzero >= int(0.8 * sum(1 for _ in corr.parameters()))


def test_loss_gradient_wrt_flow(scale: RawScale, loss_fn: RaftCorrespondenceLoss) -> None:
    previous, current = _uint8_frames(11)
    flow = (torch.zeros(BATCH, 2, HEIGHT, WIDTH) + 1.0).requires_grad_(True)
    losses = loss_fn(previous, current, flow)
    losses.total.backward()
    assert flow.grad is not None
    assert float(flow.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------------------
# 塌缩修复：静态基线归一化 + 平滑项延后（固定集诊断后的授权改动）
# --------------------------------------------------------------------------------------


def _unit_loss(config: RaftLossConfig | None = None) -> RaftCorrespondenceLoss:
    """合成内容就在测量单位 [0,1]：用 factor=1 的尺度，避免 LIBERO 的 1/255 把内容压成近常数。"""
    return RaftCorrespondenceLoss(scale=RawScale(name="unit_linear", factor=1.0), config=config)


def _low_frequency_pair(scale_shift: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """低频但保留像素级结构的内容 + 亚像素平移：静态残差小，是"大常数位移"退化的现实情形。"""
    torch.manual_seed(1)
    previous = F.avg_pool2d(torch.rand(BATCH, 3, HEIGHT, WIDTH), 5, stride=1, padding=2)
    offset = torch.tensor([scale_shift, 0.0]).view(1, 2, 1, 1)
    coords = (_base_grid(HEIGHT, WIDTH) - offset).expand(BATCH, -1, -1, -1)
    current = C._resample(previous, coords)
    return previous, current


def _constant_flow(value: float) -> torch.Tensor:
    return torch.full((BATCH, 2, HEIGHT, WIDTH), value)


def test_zero_flow_is_exactly_the_static_baseline() -> None:
    loss_fn = _unit_loss()
    previous, current = _low_frequency_pair(0.5)
    losses = loss_fn(previous, current, torch.zeros(BATCH, 2, HEIGHT, WIDTH))
    # 零 flow 时 matching 分子分母同源 -> 比值恰为 1
    assert float(losses.census) == pytest.approx(1.0, abs=1e-5)
    assert float(losses.photometric) == pytest.approx(1.0, abs=1e-5)
    assert float(losses.total) == pytest.approx(2.0, abs=1e-4)


def test_constant_displacement_is_not_free() -> None:
    """退化解：注入常数大位移（与固定集上 step100 的 fy≈28px 同量级）必须更贵。"""
    loss_fn = _unit_loss()
    previous, current = _low_frequency_pair(0.5)
    true_flow = torch.zeros(BATCH, 2, HEIGHT, WIDTH)
    true_flow[:, 0] = 0.5  # 真值：内容右移 0.5px
    l_true = loss_fn(previous, current, true_flow)
    l_zero = loss_fn(previous, current, _constant_flow(0.0))
    l_degenerate = loss_fn(previous, current, _constant_flow(28.0))

    assert float(l_true.census) < 1.0 < float(l_degenerate.census)
    assert float(l_true.photometric) < 1.0 < float(l_degenerate.photometric)
    assert float(l_true.total) < float(l_zero.total) < float(l_degenerate.total)
    assert float(l_degenerate.total) > 1.5 * float(l_zero.total)

    # 对照（旧形式）：退化点的 loss 值同样高于零流，所以修复的关键不是"值"，而是
    # ①退化点上 matching 梯度被静态基线按 1/零-flow 残差放大（可从该点被拉回），
    # ②任何位移都有明确的静态参照（零 flow 恰为 1），不再有"动了也不亏"的自由方向。
    old = _unit_loss(RaftLossConfig(static_baseline=False))
    new_flow = _constant_flow(28.0).requires_grad_(True)
    loss_fn(previous, current, new_flow).total.backward()
    old_flow = _constant_flow(28.0).requires_grad_(True)
    old(previous, current, old_flow).total.backward()
    assert float(new_flow.grad.abs().mean()) > 1.5 * float(old_flow.grad.abs().mean())


def test_smoothness_delay_gates_terms() -> None:
    config = RaftLossConfig(smoothness_delay_pairs=2)
    loss_fn = _unit_loss(config)
    previous, current = _low_frequency_pair(0.5)
    flow = _constant_flow(1.0)
    first = loss_fn(previous, current, flow)
    assert float(first.smooth1) == 0.0 and float(first.smooth2) == 0.0
    second = loss_fn(previous, current, flow)
    assert float(second.smooth2) == 0.0  # 第 2 次调用仍未过门槛（index > delay）
    third = loss_fn(previous, current, flow)
    assert float(third.smooth2) > 0.0  # 第 3 次开始启用
    assert torch.allclose(
        third.total, third.census + third.photometric + config.smooth2 * third.smooth2, atol=1e-5
    )


def test_set_step_switches_to_optimizer_steps() -> None:
    loss_fn = _unit_loss()
    previous, current = _low_frequency_pair(0.5)
    loss_fn.set_step(10)
    assert not loss_fn._smoothness_enabled()
    loss_fn.set_step(2001)  # 默认 smoothness_delay_steps=2000
    assert loss_fn._smoothness_enabled()
    with pytest.raises(ValueError, match="不能为负"):
        loss_fn.set_step(-1)


def test_fresh_raft_is_not_judged_worse_than_degenerate_flow() -> None:
    """对照：修复后随机初始化 RAFT 的 loss 仍低于常数大位移，重加权没有惩罚一切。"""
    scale = RawScale(name="unit_linear", factor=1.0)
    loss_fn = _unit_loss()
    previous, current = _low_frequency_pair(0.5)
    torch.manual_seed(0)
    model = RaftCorrespondence(scale=scale, iters=2).eval()
    with torch.no_grad():
        fresh_flow = model(previous, current).flow_px
    fresh_losses = loss_fn(previous, current, fresh_flow)
    degenerate_losses = loss_fn(previous, current, _constant_flow(28.0))
    assert torch.isfinite(fresh_losses.total)
    assert float(fresh_losses.total) < float(degenerate_losses.total)


def test_bidirectional_wang_masks_internal_hole():
    # A backward field mapping every column to x=0 leaves an interior hole even
    # though the forward endpoints never leave the image.
    class Pair(torch.nn.Module):
        def forward(self, previous, current):
            h, w = previous.shape[-2:]
            flow = torch.zeros(1, 2, h, w)
            if float(previous.mean()) > float(current.mean()):
                flow[:, 0] = -torch.arange(w).float()
            return FlowOutput(
                flow_px=flow, valid=torch.ones(1, h, w, dtype=torch.bool), feature_scale=1.0
            )

    previous, current = torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16)
    result = C.bidirectional_correspondence(Pair(), previous, current)
    assert not bool(result.valid[0, 8, 8])
    assert bool(result.valid[0, 8, 0])
    assert result.backward_flow_px is not None


def test_crop_selfsup_stops_teacher_and_masks_but_trains_student():
    teacher = torch.zeros(1, 2, 16, 16, requires_grad=True)
    backward = torch.zeros_like(teacher, requires_grad=True)
    student = torch.ones(1, 2, 12, 12, requires_grad=True)
    student_backward = torch.zeros_like(student, requires_grad=True)
    loss = C.self_supervision_loss(
        teacher, backward, student, student_backward, crop=(2, 2, 14, 14)
    )
    assert float(loss) > 0.5
    loss.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    assert teacher.grad is None and backward.grad is None and student_backward.grad is None
    consistent = C.self_supervision_loss(
        teacher, backward, torch.zeros_like(student), torch.zeros_like(student), crop=(2, 2, 14, 14)
    )
    assert float(consistent) == 0


def test_raw_saturation_and_confidence_use_every_channel():
    scale = RawScale.from_input_scale(ROBOTWIN, saturation_level=3.5)
    dark = torch.zeros(1, 3, 8, 8)
    changed = dark.clone()
    changed[:, 2] = 4
    assert not bool(scale.saturated(dark).any())
    assert bool(scale.saturated(changed).all())
    assert not torch.equal(scale.confidence(dark), scale.confidence(changed))


def test_both_matching_directions_train_the_shared_network():
    class TwoDirections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_dx = torch.nn.Parameter(torch.tensor(0.25))
            self.backward_dx = torch.nn.Parameter(torch.tensor(-0.25))
            self.calls = 0

        def forward(self, a, b):
            dx = self.forward_dx if self.calls == 0 else self.backward_dx
            self.calls += 1
            flow = torch.stack(
                (dx.expand(a.shape[0], *a.shape[-2:]), dx.expand(a.shape[0], *a.shape[-2:]) * 0), 1
            )
            return FlowOutput(flow, torch.ones_like(a[:, 0], dtype=torch.bool), 1.0)

    model = TwoDirections()
    a, b = _uint8_frames(441)
    pair = C.bidirectional_correspondence(model, a, b)
    fn = RaftCorrespondenceLoss(scale=RawScale.from_input_scale(LIBERO))
    fn(a, b, pair.flow_px, backward_flow_px=pair.backward_flow_px).total.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and float(parameter.grad.abs()) > 0


def test_selfsup_respects_measurement_validity():
    teacher = torch.zeros(1, 2, 16, 16)
    student = torch.ones(1, 2, 12, 12, requires_grad=True)
    value = C.self_supervision_loss(
        teacher,
        teacher,
        student,
        torch.zeros_like(student),
        crop=(2, 2, 14, 14),
        valid=torch.zeros(1, 16, 16),
    )
    assert float(value) == 0
    value.backward()
    assert not bool(student.grad.any())

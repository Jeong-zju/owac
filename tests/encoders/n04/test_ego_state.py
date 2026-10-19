"""N04 状态、传播与因果性测试。

重点回答三件事：

1. ``predict_before_observe`` 是否真的与当前图像无关（因果性）。
2. ``learned_proprio`` 是否零位移初始化、固定相机能否学到接近 identity 的传播。
3. 状态语义：相机独立、部分 batch reset、同一帧多次调用只更新一次、变换组合/逆。
"""

from __future__ import annotations

import math

import torch

from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
from owac.frameworks.n04.ego import (
    CalibratedGeometryPropagator,
    EgoMotionContext,
    IdentityPropagator,
    LearnedProprioPropagator,
    ProprioLayout,
    ProprioOnlyPropagator,
    build_propagator,
    relative_rotation,
    state_delta,
)
from owac.frameworks.n04.geometry import GridSpec
from owac.frameworks.n04.model import N04Encoder
from owac.frameworks.n04.state import new_state

#: 测试用的运动上下文维度：本体 delta(4) + 已执行动作(3) + dt(1)。
MOTION_DIM = 8


def _spec(h: int = 8, w: int = 8, stride: float = 1.0) -> GridSpec:
    return GridSpec(h, w, pixel_stride=stride)


def _context(b: int = 2, motion_dim: int = 4, **kwargs) -> EgoMotionContext:
    ctx = EgoMotionContext(
        proprio_delta=torch.zeros(b, motion_dim),
        proprio_delta_available=True,
        executed_action=torch.zeros(b, 3),
        executed_action_available=True,
        dt=torch.full((b,), 1.0),
        dt_available=False,
        camera_index=torch.zeros(b, dtype=torch.long),
        camera_id="head",
    )
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


# --------------------------------------------------------------------------------------
# 因果性：纯 predict 不读当前图像
# --------------------------------------------------------------------------------------


def test_pure_predict_signature_has_no_measurement_argument():
    """接口层面固定：`predict_before_observe` 的签名里没有图像/测量参数。"""
    import inspect

    params = list(inspect.signature(N04Encoder.predict_before_observe).parameters)
    assert params == ["self", "state", "context"]


def test_current_image_cannot_influence_pure_predict():
    """同一 state + context，喂入完全不同的当前图像，纯 predict 输出必须逐位相同。"""
    torch.manual_seed(0)
    cfg = _tiny_config(predictor="learned_proprio")
    enc = N04Encoder(cfg)
    spec = enc.spec
    state = new_state(2, cfg.model.width, 8, 8, camera_id="head", scales=1)
    # 给状态一个非零历史与历史有效性
    state.stable = torch.randn(2, cfg.model.width, 8, 8)
    state.history_valid = torch.ones(2, dtype=torch.bool)
    ctx = _context(2, MOTION_DIM)
    ctx.proprio_delta = torch.randn(2, MOTION_DIM)
    pred_a, valid_a, unc_a, _ = enc.predict_before_observe(state, ctx)
    pred_b, valid_b, unc_b, _ = enc.predict_before_observe(state, ctx)
    assert torch.equal(pred_a, pred_b)
    assert torch.equal(valid_a, valid_b)
    assert torch.equal(unc_a, unc_b)
    # 显式证明：即使存在一个“当前图像”，纯 predict 也不接受它
    img = torch.randn(2, cfg.channels, *cfg.measurement_size)
    step_out = enc.step(img, ctx, state, camera_id="head")
    img2 = torch.randn(2, cfg.channels, *cfg.measurement_size)
    state2 = new_state(2, cfg.model.width, 8, 8, camera_id="head", scales=1)
    state2.stable = state.stable.clone()
    state2.history_valid = torch.ones(2, dtype=torch.bool)
    step_out2 = enc.step(img2, ctx, state2, camera_id="head")
    # step 里的 predict 阶段（state.prediction）只取决于 prev 状态与 context
    assert torch.equal(step_out.state.prediction, step_out2.state.prediction)
    # 而 correction 阶段确实随图像改变
    assert not torch.equal(step_out.state.stable, step_out2.state.stable)
    assert spec.height == 8


def test_unavailable_history_yields_zero_validity():
    """没有历史时 predict 的 validity 必须为 0，不能用 pred=0 冒充历史证据。"""
    torch.manual_seed(0)
    cfg = _tiny_config(predictor="learned_proprio")
    enc = N04Encoder(cfg)
    state = new_state(1, cfg.model.width, 8, 8, scales=1)
    ctx = _context(1, MOTION_DIM)
    _, validity, _, _ = enc.predict_before_observe(state, ctx)
    assert float(validity.abs().sum()) == 0.0


# --------------------------------------------------------------------------------------
# identity / learned_proprio 初始化
# --------------------------------------------------------------------------------------


def test_identity_propagator_is_exact_identity():
    spec = _spec()
    prop = IdentityPropagator()
    prev = torch.randn(2, 3, 8, 8)
    res = prop(prev, spec, _context(2))
    assert torch.allclose(res.predicted, prev, atol=0)
    assert torch.equal(res.validity, torch.ones_like(res.validity))
    assert torch.equal(res.forward_flow, torch.zeros_like(res.forward_flow))


def test_learned_proprio_starts_at_zero_displacement():
    """零位移初始化：未训练时 learned_proprio 的 forward flow 必须恒为 0。"""
    torch.manual_seed(0)
    prop = LearnedProprioPropagator(
        motion_dim=None, width=8, coarse_stride=4, max_flow=8.0, hidden=16
    )
    spec = _spec()
    prev = torch.randn(2, 8, 8, 8)
    ctx = _context(2, 6)
    ctx.proprio_delta = torch.randn(2, 6)
    res = prop(prev, spec, ctx)
    assert torch.allclose(res.forward_flow, torch.zeros_like(res.forward_flow), atol=1e-7)
    assert torch.allclose(res.predicted, prev, atol=1e-6)
    assert float(res.validity.min()) == 1.0


def test_learned_proprio_can_learn_identity_for_fixed_camera():
    """固定相机（action/proprio 恒零）时，恒等传播必须是可学的：恒等解 loss 最低。"""
    torch.manual_seed(0)
    prop = LearnedProprioPropagator(
        motion_dim=None, width=8, coarse_stride=4, max_flow=8.0, hidden=16
    )
    spec = _spec()
    prev = torch.randn(1, 8, 8, 8)
    ctx = _context(1, 4)  # 本体/动作/dt 全零 = 固定相机
    opt = torch.optim.Adam(prop.parameters(), lr=3e-2)
    for _ in range(60):
        opt.zero_grad()
        res = prop(prev, spec, ctx)
        loss = (res.predicted - prev).abs().mean()
        loss.backward()
        opt.step()
    res = prop(prev, spec, ctx)
    identity_err = float((res.predicted - prev).abs().mean())
    assert identity_err < 1e-3, f"固定相机下 identity 传播未被学到: {identity_err}"
    assert float(res.forward_flow.abs().max()) < 1e-3


def test_proprio_only_ignores_previous_features_by_construction():
    """proprio_only 对照：上一特征被显式置零，同运动上下文下输出与视觉内容无关。"""
    torch.manual_seed(0)
    prop = ProprioOnlyPropagator(motion_dim=None, width=8, coarse_stride=4, max_flow=8.0, hidden=16)
    # 打破对称：让 head 非零，否则零初始化会把差异全部掩盖
    torch.nn.init.normal_(prop.head.weight, std=0.05)
    spec = _spec()
    ctx = _context(1, 4)
    ctx.proprio_delta = torch.randn(1, 4)
    out_a = prop(torch.zeros(1, 8, 8, 8), spec, ctx).forward_flow
    out_b = prop(torch.randn(1, 8, 8, 8) * 10, spec, ctx).forward_flow
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert float(out_a.abs().max()) > 0


def test_proprio_only_and_learned_share_the_same_context_contract():
    """两个对照接收同样的运动上下文向量布局，只有“是否读上一视觉”不同。"""
    torch.manual_seed(0)
    ctx = _context(1, 4)
    ctx.proprio_delta = torch.randn(1, 4)
    vec_a, info_a = ctx.motion_vector()
    vec_b, info_b = ctx.motion_vector()
    assert torch.equal(vec_a, vec_b)
    assert info_a["motion_names"] == info_b["motion_names"]
    assert info_a["time_unit"] == "frame"


# --------------------------------------------------------------------------------------
# 本体语义：相对旋转
# --------------------------------------------------------------------------------------


def test_quaternion_relative_rotation_is_not_naive_difference():
    q_prev = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_curr = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # 绕 x 轴 180°
    rel = relative_rotation(q_prev, q_curr, "quaternion")
    assert torch.allclose(rel, q_curr, atol=1e-6)
    naive = q_curr - q_prev
    assert not torch.allclose(rel, naive, atol=1e-3)


def test_euler_layout_is_labelled_raw_diff_not_angle_increment():
    layout = ProprioLayout(
        position_index=(0, 1, 2), rotation_index=(3, 4, 5), rotation_semantics="euler"
    )
    prev = torch.zeros(1, 8)
    curr = torch.zeros(1, 8)
    curr[0, 3] = 0.5
    delta, info = state_delta(prev, curr, layout)
    assert info["rotation_semantics"] == "euler"
    assert info["rotation_applied"] is True
    assert abs(float(delta[0, 3]) - 0.5) < 1e-6


def test_libero_and_robotwin_layouts_are_distinct():
    a = ProprioLayout.libero_ee8()
    b = ProprioLayout.robotwin_joint14()
    assert a.rotation_index == (3, 4, 5) and b.rotation_index == ()
    assert b.joint_index == tuple(range(12))
    assert b.gripper_index == (12, 13)
    assert a.name != b.name  # 明确标注：libero 的切分在本仓库尚未逐位核对


# --------------------------------------------------------------------------------------
# 显式几何传播
# --------------------------------------------------------------------------------------


def test_calibrated_geometry_reports_unavailable_without_calibration():
    prop = CalibratedGeometryPropagator()
    spec = _spec()
    res = prop(torch.randn(1, 2, 8, 8), spec, _context(1))
    assert res.geometry_mode == "unavailable"
    assert float(res.validity.sum()) == 0.0
    assert res.forward_flow is None


def test_calibrated_geometry_identity_pose_gives_identity_warp():
    prop = CalibratedGeometryPropagator()
    spec = _spec()
    K = torch.tensor([[[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]]])
    T = torch.eye(4).unsqueeze(0)
    ctx = _context(1)
    ctx.intrinsics = K
    ctx.T_world_from_camera = T
    ctx.T_world_from_camera_current = T
    ctx.depth_prev = torch.full((1, 8, 8), 5.0)
    prev = torch.randn(1, 2, 8, 8)
    res = prop(prev, spec, ctx)
    assert res.geometry_mode == "depth"
    assert torch.allclose(res.forward_flow, torch.zeros_like(res.forward_flow), atol=1e-5)
    assert float(res.validity.min()) == 1.0
    assert torch.allclose(res.predicted, prev, atol=1e-4)


def test_calibrated_geometry_depth_path_direction_and_validity():
    """相机 +x 平移时，静止平面上的点在图像中左移，新暴露区域 validity=0。"""
    prop = CalibratedGeometryPropagator()
    spec = GridSpec(8, 8, pixel_stride=1.0)
    K = torch.tensor([[[12.0, 0.0, 3.5], [0.0, 12.0, 3.5], [0.0, 0.0, 1.0]]])
    b = 1
    xy = spec.identity_xy(b, torch.device("cpu"), torch.float32)
    xy_prev = spec.pixel_xy_to_feature_xy(spec.feature_xy_to_pixel_xy(xy))
    x_prev_alt = spec.feature_xy_to_pixel_xy(xy_prev)[..., 0]
    from owac.frameworks.n04.geometry import forward_project_points

    depth = torch.full((b, 8, 8), 5.0)
    T_prev = torch.eye(4).unsqueeze(0)
    T_curr = torch.eye(4).unsqueeze(0)
    T_curr[0, 0, 3] = 0.5
    uv_curr, _ = forward_project_points(
        spec.feature_xy_to_pixel_xy(xy), depth, K, K, T_prev, T_curr
    )
    assert (uv_curr[..., 0] < x_prev_alt - 0.1).all()
    ctx = _context(b)
    ctx.intrinsics = K
    ctx.T_world_from_camera = T_prev
    ctx.T_world_from_camera_current = T_curr
    ctx.depth_prev = depth
    res = prop(torch.randn(b, 2, 8, 8), spec, ctx)
    assert res.geometry_mode == "depth"
    assert (res.forward_flow[..., 0] < -0.1).all()
    assert res.diagnostics["occlusion_checked"] is True
    # 左移后右侧条带没有历史来源
    assert float(res.validity[0, :, 0, -1].max()) == 0.0


def test_calibrated_geometry_plane_path_matches_depth_path_for_pure_rotation():
    """纯旋转 + 常数深度时，平面路径与深度路径应给出相同映射。"""
    spec = GridSpec(8, 8, pixel_stride=1.0)
    K = torch.tensor([[[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]])
    theta = 0.15
    R = torch.tensor(
        [
            [
                [math.cos(theta), 0.0, math.sin(theta)],
                [0.0, 1.0, 0.0],
                [-math.sin(theta), 0.0, math.cos(theta)],
            ]
        ]
    )
    T_prev = torch.eye(4).unsqueeze(0)
    T_curr = torch.eye(4).unsqueeze(0)
    T_curr[0, :3, :3] = R
    depth = torch.full((1, 8, 8), 4.0)
    ctx_depth = _context(1)
    ctx_depth.intrinsics = K
    ctx_depth.T_world_from_camera = T_prev
    ctx_depth.T_world_from_camera_current = T_curr
    ctx_depth.depth_prev = depth
    res_depth = CalibratedGeometryPropagator()(torch.randn(1, 2, 8, 8), spec, ctx_depth)
    ctx_plane = _context(1)
    ctx_plane.intrinsics = K
    ctx_plane.T_world_from_camera = T_prev
    ctx_plane.T_world_from_camera_current = T_curr
    ctx_plane.plane_normal_prev = torch.tensor([[0.0, 0.0, 1.0]])
    ctx_plane.plane_distance = 4.0
    res_plane = CalibratedGeometryPropagator()(torch.randn(1, 2, 8, 8), spec, ctx_plane)
    assert res_depth.geometry_mode == "depth" and res_plane.geometry_mode == "plane"
    assert torch.allclose(res_depth.forward_flow, res_plane.forward_flow, atol=1e-3)


def test_build_propagator_selects_configured_mode():
    geom = N04GeometryConfig(feature_size=(8, 8), pixel_stride=1.0)
    model = N04ModelConfig(width=8, hidden=16, motion_dim=4)
    assert isinstance(
        build_propagator("identity", model_config=model, geometry=geom, motion_dim=4),
        IdentityPropagator,
    )
    assert isinstance(
        build_propagator("learned_proprio", model_config=model, geometry=geom, motion_dim=4),
        LearnedProprioPropagator,
    )
    assert isinstance(
        build_propagator("proprio_only", model_config=model, geometry=geom, motion_dim=4),
        ProprioOnlyPropagator,
    )
    assert isinstance(
        build_propagator("calibrated_geometry", model_config=model, geometry=geom, motion_dim=4),
        CalibratedGeometryPropagator,
    )


# --------------------------------------------------------------------------------------
# 状态语义
# --------------------------------------------------------------------------------------


def test_each_camera_has_independent_state():
    cfg = _tiny_config()
    cfg = N04Config(
        input_mode=cfg.input_mode,
        measurement_size=cfg.measurement_size,
        channels=cfg.channels,
        camera_ids=("head", "wrist"),
        geometry=cfg.geometry,
        model=cfg.model,
    )
    enc = N04Encoder(cfg)
    states = enc.init_state(2)
    assert set(states) == {"head", "wrist"}
    assert states["head"] is not states["wrist"]
    ctx = _context(2, MOTION_DIM)
    ctx.camera_id = "head"
    out_head = enc.step(torch.rand(2, 3, 8, 8), ctx, states["head"], camera_id="head")
    ctx_w = _context(2, MOTION_DIM)
    ctx_w.camera_id = "wrist"
    ctx_w.proprio_delta = torch.ones(2, MOTION_DIM)
    out_wrist = enc.step(torch.rand(2, 3, 8, 8), ctx_w, states["wrist"], camera_id="wrist")
    assert out_head.state.camera_id == "head"
    assert out_wrist.state.camera_id == "wrist"
    assert not torch.equal(out_head.state.stable, out_wrist.state.stable)


def test_partial_batch_reset_clears_only_marked_samples():
    cfg = _tiny_config()
    enc = N04Encoder(cfg)
    states = enc.init_state(3)
    ctx = _context(3, MOTION_DIM)
    ctx.proprio_delta = torch.randn(3, MOTION_DIM)
    out = enc.step(torch.rand(3, 3, 8, 8), ctx, states["head"], camera_id="head")
    assert bool(out.state.history_valid.all())
    mask = torch.tensor([True, False, False])
    reset = enc.reset({"head": out.state}, mask)
    assert reset["head"].history_valid.tolist() == [False, True, True]
    assert float(reset["head"].stable[0].abs().sum()) == 0.0
    assert float(reset["head"].stable[1].abs().sum()) > 0.0
    assert float(reset["head"].stable[2].abs().sum()) > 0.0


def test_same_frame_multiple_calls_update_state_once():
    """同一帧被读多次（例如 10 次去噪迭代）时状态只更新一次。"""
    cfg = _tiny_config()
    enc = N04Encoder(cfg)
    states = enc.init_state(1)
    ctx = _context(1, MOTION_DIM)
    img = torch.rand(1, 3, 8, 8)
    first = enc.step(img, ctx, states["head"], camera_id="head", frame_key="t0")
    assert first.updated and first.state.updates == 1
    for _ in range(9):
        again = enc.step(img, ctx, first.state, camera_id="head", frame_key="t0")
        assert again.updated is False
        assert again.cache_key == "t0"
    assert first.state.updates == 1
    second = enc.step(torch.rand(1, 3, 8, 8), ctx, first.state, camera_id="head", frame_key="t1")
    assert second.updated and second.state.updates == 2


def test_new_measurement_false_propagates_without_updating():
    cfg = _tiny_config()
    enc = N04Encoder(cfg)
    states = enc.init_state(1)
    ctx = _context(1, MOTION_DIM)
    out = enc.step(torch.rand(1, 3, 8, 8), ctx, states["head"], camera_id="head")
    blank = torch.rand(1, 3, 8, 8)
    out2 = enc.step(blank, ctx, out.state, camera_id="head", new_measurement=False)
    # 没有新测量时不得把这一帧的内容写进稳定状态（"重复帧"不是新证据）
    stem_blank = enc.encode_measurement(blank)
    assert out2.state.decomposition["g_update"].abs().max().item() == 0.0
    assert not torch.allclose(out2.state.stable, stem_blank, atol=1e-3)
    # 无新测量时，读出也不能绕过 correction 偷读任意当前图像。
    other = enc.step(blank + 10, ctx, out.state, camera_id="head", new_measurement=False)
    assert torch.equal(out2.representation, other.representation)
    assert not out2.updated and not out2.state.available.any()
    assert out2.state.updates == out.state.updates
    assert torch.equal(out2.state.age, out.state.age)


def test_detach_state_breaks_graph_but_keeps_values():
    cfg = _tiny_config()
    enc = N04Encoder(cfg)
    states = enc.init_state(1)
    ctx = _context(1, MOTION_DIM)
    out = enc.step(
        torch.rand(1, 3, 8, 8, requires_grad=False), ctx, states["head"], camera_id="head"
    )
    detached = enc.detach_state({"head": out.state})["head"]
    assert detached.stable.requires_grad is False
    assert torch.allclose(detached.stable, out.state.stable)


def test_state_bytes_reported():
    """状态字节数按实际缓冲区计算：stable + change + age（float32/int64 混合）。"""
    cfg = _tiny_config()
    state = new_state(4, cfg.model.width, 8, 8, scales=1)
    per_feature = 4 * cfg.model.width * 8 * 8 * 4  # [B,C,H,W] float32
    assert state.state_bytes() == per_feature * 2 + 4 * 8  # stable + change + age(int64)
    state.measured = torch.zeros(4, cfg.model.width, 8, 8)
    assert state.state_bytes() == per_feature * 3 + 4 * 8


def _tiny_config(predictor: str = "learned_proprio") -> N04Config:
    return N04Config(
        input_mode="rawvla_rgb3",
        measurement_size=(8, 8),
        channels=3,
        camera_ids=("head",),
        geometry=N04GeometryConfig(feature_size=(8, 8), pixel_stride=1.0),
        model=N04ModelConfig(
            name="n04-test",
            width=8,
            blocks=1,
            scales=1,
            scale_strides=(1,),
            coarse_stride=2,
            hidden=16,
            readout_tokens=4,
            readout_dim=8,
            predictor=predictor,
        ),
    )

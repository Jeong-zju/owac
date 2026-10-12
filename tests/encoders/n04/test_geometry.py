"""N04 解析几何测试：坐标约定、warp 方向、组合/逆、validity 与显式投影。

全部为可手算的小尺寸例子（6x6 / 8x8），不依赖训练或数据。
"""

from __future__ import annotations

import math

import pytest
import torch

from owac.frameworks.n04.geometry import (
    AffineFlow,
    GridSpec,
    backward_grid_from_affine,
    backward_grid_from_translation,
    compose_affine,
    forward_project_points,
    forward_splat,
    homography_from_rt,
    invert_affine,
    invert_rigid_2d,
    project_points,
    unproject_pixels,
    warp_feature_field,
)


def _impulse(h: int = 6, w: int = 6, y: int = 2, x: int = 3) -> torch.Tensor:
    src = torch.zeros(1, 1, h, w)
    src[0, 0, y, x] = 1.0
    return src


# --------------------------------------------------------------------------------------
# 坐标约定
# --------------------------------------------------------------------------------------


def test_pixel_center_normalized_coordinates():
    """align_corners=False 下像素中心 0 与 W-1 的归一化坐标必须是 (2(i+0.5)/W - 1)。"""
    spec = GridSpec(4, 4)
    xy = spec.identity_xy(1, torch.device("cpu"), torch.float32)
    grid = spec.normalized_from_xy(xy)
    assert grid[0, 0, 0].tolist() == pytest.approx([2 * 0.5 / 4 - 1, 2 * 0.5 / 4 - 1])
    assert grid[0, 3, 3].tolist() == pytest.approx([2 * 3.5 / 4 - 1, 2 * 3.5 / 4 - 1])
    # 往返一致
    back = spec.xy_from_normalized(grid)
    assert torch.allclose(back, xy, atol=1e-6)


def test_identity_grid_roundtrip_is_exact():
    """identity 网格必须精确取回原值（整数中心落在采样点上）。"""
    spec = GridSpec(5, 7)
    src = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
    res = warp_feature_field(src, spec.identity_grid(1, torch.device("cpu"), torch.float32), spec)
    assert torch.allclose(res.values, src, atol=1e-5)
    assert torch.all(res.validity == 1)


def test_pixel_stride_conversion():
    spec = GridSpec(8, 8, pixel_stride=4.0)
    xy = torch.tensor([[[[2.0, 3.0]]]])
    assert spec.feature_xy_to_pixel_xy(xy).tolist() == [[[[8.0, 12.0]]]]
    assert spec.pixel_xy_to_feature_xy(spec.feature_xy_to_pixel_xy(xy)).tolist() == xy.tolist()


def test_align_corners_and_padding_are_locked():
    with pytest.raises(ValueError):
        GridSpec(4, 4, align_corners=True)
    with pytest.raises(ValueError):
        GridSpec(4, 4, padding_mode="border")


# --------------------------------------------------------------------------------------
# 平移与仿射
# --------------------------------------------------------------------------------------


def test_positive_and_negative_single_pixel_translation():
    """forward dx=+1 表示内容右移一格：上一帧 x=3 的点在当前帧出现在 x=4。"""
    spec = GridSpec(6, 6)
    src = _impulse(y=2, x=3)
    res = warp_feature_field(src, backward_grid_from_translation(spec, 1.0, 0.0), spec)
    assert res.values.argmax().item() == 2 * 6 + 4  # 内容右移 -> (y=2, x=4)
    res = warp_feature_field(src, backward_grid_from_translation(spec, -1.0, 0.0), spec)
    assert res.values.argmax().item() == 2 * 6 + 2  # 内容左移 -> (y=2, x=2)


def test_vertical_translation_direction():
    spec = GridSpec(6, 6)
    src = _impulse(y=1, x=3)
    res = warp_feature_field(src, backward_grid_from_translation(spec, 0.0, 2.0), spec)
    assert res.values.argmax().item() == 3 * 6 + 3  # 内容下移 2 -> 现在在 y=3


def test_forward_flow_must_not_be_used_as_backward_grid():
    """把 forward flow 直接塞进 backward grid 会得到反向对应；本测试固定这一事实。"""
    spec = GridSpec(6, 6)
    src = _impulse(y=2, x=3)
    correct = warp_feature_field(src, backward_grid_from_translation(spec, 1.0, 0.0), spec)
    xy = spec.identity_xy(1, torch.device("cpu"), torch.float32)
    wrong_grid = spec.normalized_from_xy(xy + torch.tensor([1.0, 0.0]))  # 错把 forward flow 当网格
    wrong = warp_feature_field(src, wrong_grid, spec)
    assert correct.values.argmax().item() == 2 * 6 + 4
    assert wrong.values.argmax().item() == 2 * 6 + 2
    assert correct.values.argmax().item() != wrong.values.argmax().item()


def test_affine_compose_and_inverse():
    a = AffineFlow(torch.eye(2).unsqueeze(0), torch.tensor([[2.0, -1.0]]))
    theta = 0.3
    r = torch.tensor([[[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]])
    b = AffineFlow(r, torch.tensor([[0.5, 0.25]]))
    composed = compose_affine(b, a)
    xy = torch.tensor([[[[1.0, 1.0], [3.0, 4.0]]]])  # [1,1,2,2]
    direct = composed.forward_xy(xy)
    manual = b.forward_xy(a.forward_xy(xy))
    assert torch.allclose(direct, manual, atol=1e-5)
    back = invert_affine(composed).forward_xy(direct)
    assert torch.allclose(back, xy, atol=1e-4)


def test_rigid_inverse_matches_matrix_inverse():
    theta, tx, ty = 0.42, 1.3, -0.7
    inv_theta, inv_tx, inv_ty = invert_rigid_2d(theta, tx, ty)
    m = torch.tensor([[[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]])
    flow = AffineFlow(m, torch.tensor([[tx, ty]]))
    analytic = AffineFlow(
        torch.tensor(
            [
                [
                    [math.cos(inv_theta), -math.sin(inv_theta)],
                    [math.sin(inv_theta), math.cos(inv_theta)],
                ]
            ]
        ),
        torch.tensor([[inv_tx, inv_ty]]),
    )
    assert torch.allclose(invert_affine(flow).matrix, analytic.matrix, atol=1e-6)
    assert torch.allclose(invert_affine(flow).translation, analytic.translation, atol=1e-6)


def test_rotation_about_center_keeps_center_fixed():
    spec = GridSpec(9, 9)
    src = _impulse(h=9, w=9, y=4, x=4)
    theta = 0.5
    cx = cy = 4.0
    m = torch.tensor([[[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]])
    t = torch.tensor(
        [
            [
                cx - (math.cos(theta) * cx - math.sin(theta) * cy),
                cy - (math.sin(theta) * cx + math.cos(theta) * cy),
            ]
        ]
    )
    res = warp_feature_field(src, backward_grid_from_affine(spec, AffineFlow(m, t)), spec)
    assert res.values.argmax().item() == 4 * 9 + 4
    assert res.values.max().item() == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------------------
# validity / out-of-bounds / 遮挡
# --------------------------------------------------------------------------------------


def test_out_of_bounds_has_zero_validity_and_no_padding_evidence():
    """越界处必须 validity=0 且值为 0；不得把它当作历史证据。"""
    spec = GridSpec(4, 4)
    src = torch.ones(1, 1, 4, 4)
    res = warp_feature_field(src, backward_grid_from_translation(spec, 3.0, 0.0), spec)
    # forward dx=+3：内容右移三格，只有上一帧最左一列 (x=0) 落入当前视野的最右一列。
    # 其余列对应的上一坐标 x-3 < 0 完全越界，取不到任何真实值。
    assert res.validity.shape == (1, 4, 4)
    assert res.validity[0, :, 3].tolist() == [1.0] * 4
    assert res.validity[0, :, :3].sum().item() == 0.0
    assert res.out_of_bounds[0, :, :3].sum().item() == 12.0
    assert res.values[0, 0, :, :3].abs().sum().item() == 0.0
    assert res.values[0, 0, :, 3].sum().item() == 4.0
    # validity 必须与 grid_sample 的真实取值域逐像素一致（不是另一套自定义边界）
    sampled = torch.nn.functional.grid_sample(
        src, res.grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    assert torch.equal((sampled.abs() > 0).to(res.validity.dtype).squeeze(1), res.validity)


def test_source_validity_propagates_into_warp():
    spec = GridSpec(4, 4)
    src = torch.ones(1, 1, 4, 4)
    src_valid = torch.ones(1, 4, 4)
    src_valid[:, 0, 0] = 0.0
    res = warp_feature_field(
        src, backward_grid_from_translation(spec, 0.0, 0.0), spec, source_validity=src_valid
    )
    assert res.validity[0, 0, 0].item() == 0.0
    assert res.values[0, 0, 0, 0].item() == 0.0
    assert res.validity.sum().item() == 15.0


def test_translation_creates_newly_exposed_region_without_support():
    """平移后新暴露的条带没有历史来源：validity=0。"""
    spec = GridSpec(6, 6)
    src = torch.ones(1, 1, 6, 6)
    res = warp_feature_field(src, backward_grid_from_translation(spec, 2.0, 0.0), spec)
    # forward dx=+2：内容右移两格，当前帧最右两列对应的上一坐标 x-2 落在
    # [-1, 1) 之间，grid_sample 取不到真实像素（越界），因此没有历史支持。
    xy = res.grid  # 复用同一采样网格，用 GridSpec 的边界判定来自查
    spec6 = GridSpec(6, 6)
    in_bounds = spec6.in_bounds_xy(spec6.xy_from_normalized(res.grid))
    assert in_bounds[0, :, :2].sum().item() == 0
    assert in_bounds[0, :, 2:].all()
    assert res.validity[0, :, :2].sum().item() == 0.0
    assert res.validity[0, :, 2:].sum().item() == 24.0
    assert res.values[0, 0, :, :2].abs().sum().item() == 0.0
    assert xy.shape == (1, 6, 6, 2)


def test_occlusion_detection_flags_folded_region():
    """折叠（多个源点落到同一目标格）必须被标为遮挡。"""
    spec = GridSpec(4, 4)
    src = torch.ones(1, 1, 4, 4)
    # 构造非单射的坐标映射：把 x=0 与 x=1 都映射到 x=0。
    xy = spec.identity_xy(1, torch.device("cpu"), torch.float32)
    prev_xy = xy.clone()
    prev_xy[..., 0] = torch.where(xy[..., 0] >= 1.0, xy[..., 0] - 1.0, xy[..., 0])
    grid = spec.normalized_from_xy(prev_xy)
    res = warp_feature_field(src, grid, spec, previous_xy=prev_xy, check_occlusion=True)
    assert res.occlusion_checked
    assert res.occlusion.min().item() == 0.0
    assert res.validity[0, 0].sum().item() < 16.0


def test_forward_splat_keeps_nearest_and_reports_occlusion():
    """z-buffer：同格多候选时深度最小者胜出，落败者不再贡献，空出的格子无支持。"""
    spec = GridSpec(3, 3)
    src = torch.zeros(1, 1, 3, 3)
    src[0, 0, 1, 0] = 5.0  # 平铺序号 3，深度 3（同格候选中最小的深度）
    src[0, 0, 1, 1] = 1.0  # 平铺序号 4，深度 4
    src[0, 0, 2, 2] = 7.0  # 平铺序号 8，深度 8
    xy = spec.identity_xy(1, torch.device("cpu"), torch.float32)
    dst = xy.clone()
    # 三个源点都 forward 投到同一目标格 (2,2)
    dst[0, 1, 0] = torch.tensor([2.0, 2.0])
    dst[0, 1, 1] = torch.tensor([2.0, 2.0])
    values, valid, occ = forward_splat(src, dst, spec=spec)
    # 深度最小的源点（序号 3）在目标格 (2,2) 胜出：该格有支持、未被遮挡
    assert occ[0, 2, 2].item() == 1.0
    assert valid[0, 2, 2].item() == 1.0
    assert values[0, 0, 2, 2].item() == 5.0
    # 落败的两个源点不再贡献任何值，它们自己的目标格也没有支持
    assert valid[0, 1, 1].item() == 0.0
    assert values[0, 0, 1, 1].item() == 0.0
    assert valid[0, 2, 2].item() == 1.0
    # 源点 (1,0) 从目标格 (0,1) 移走后该格空出：共 7 个目标格被占据
    assert valid.sum().item() == 7.0
    assert values[0, 0, 0, 1].item() == 0.0


def test_forward_splat_winner_is_stable_under_candidate_order():
    """胜者是“深度最小”而不是“最后写入”：交换候选顺序结果不变。"""
    spec = GridSpec(3, 3)
    xy = spec.identity_xy(1, torch.device("cpu"), torch.float32)
    dst_a = xy.clone()
    dst_a[0, 1, 0] = torch.tensor([2.0, 2.0])
    dst_a[0, 1, 1] = torch.tensor([2.0, 2.0])
    src_a = torch.zeros(1, 1, 3, 3)
    src_a[0, 0, 1, 0] = 5.0
    src_a[0, 0, 1, 1] = 9.0
    # 交换两个候选的深度：给序号 4 一个更小的深度
    depth_a = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    depth_b = depth_a.clone()
    depth_b[0, 1, 0] = 100.0
    v_a, _, _ = forward_splat(src_a, dst_a, source_depth=depth_a, spec=spec)
    v_b, _, _ = forward_splat(src_a, dst_a, source_depth=depth_b, spec=spec)
    assert v_a[0, 0, 2, 2].item() == 5.0
    assert v_b[0, 0, 2, 2].item() == 9.0


# --------------------------------------------------------------------------------------
# 显式投影
# --------------------------------------------------------------------------------------


def _K(fx=8.0, fy=8.0, cx=3.5, cy=3.5) -> torch.Tensor:
    return torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]])


def test_unproject_project_roundtrip():
    K = _K()
    xy = torch.tensor([[[[0.0, 0.0], [7.0, 7.0]]]])
    depth = torch.tensor([[[2.0, 3.0]]])
    pts = unproject_pixels(xy, depth, K)
    uv, in_front = project_points(pts, K)
    assert torch.allclose(uv, xy, atol=1e-4)
    assert bool(in_front.all())


def test_forward_projection_pure_translation_direction():
    """相机沿 x 右移 t，静止平面上的点在图像中向左移动。"""
    K = _K()
    b, h, w = 1, 4, 4
    spec = GridSpec(h, w, pixel_stride=1.0)
    xy = spec.identity_xy(b, torch.device("cpu"), torch.float32)
    depth = torch.full((b, h, w), 5.0)
    T_prev = torch.eye(4).unsqueeze(0)
    T_curr = torch.eye(4).unsqueeze(0)
    T_curr[0, 0, 3] = 0.5  # 相机 +x 平移 0.5
    uv_curr, in_front = forward_project_points(xy, depth, K, K, T_prev, T_curr)
    # 点向左移动（u 减小）
    assert (uv_curr[..., 0] < xy[..., 0] - 0.1).all()
    assert bool(in_front.all())


def test_forward_projection_pure_rotation_of_stationary_pixel():
    """纯旋转：深度为常数时，投影结果等于平面同构给出的结果。"""
    K = _K()
    b, h, w = 1, 4, 4
    spec = GridSpec(h, w, pixel_stride=1.0)
    xy = spec.identity_xy(b, torch.device("cpu"), torch.float32)
    depth = torch.full((b, h, w), 3.0)
    theta = 0.2
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
    uv_curr, _ = forward_project_points(xy, depth, K, K, T_prev, T_curr)
    # 平面同构（n = (0,0,1) 指向相机前方，d = 3）应给出相同映射
    t = torch.zeros(1, 3)
    H = homography_from_rt(K, K, R.transpose(-1, -2), t, torch.tensor([[0.0, 0.0, 1.0]]), 3.0)
    hom = torch.cat((xy, torch.ones(b, h, w, 1)), dim=-1)
    mapped = torch.einsum("bij,bhwj->bhwi", H, hom)
    uv_h = mapped[..., :2] / mapped[..., 2:3]
    assert torch.allclose(uv_curr, uv_h, atol=1e-3)


def test_forward_projection_out_of_view_flagged_not_filled():
    K = _K()
    b, h, w = 1, 4, 4
    spec = GridSpec(h, w, pixel_stride=1.0)
    xy = spec.identity_xy(b, torch.device("cpu"), torch.float32)
    depth = torch.full((b, h, w), 5.0)
    T_prev = torch.eye(4).unsqueeze(0)
    T_curr = torch.eye(4).unsqueeze(0)
    T_curr[0, 2, 3] = 10.0  # 相机前移越过平面，所有点到相机后方
    uv, in_front = forward_project_points(xy, depth, K, K, T_prev, T_curr)
    assert not bool(in_front.any())

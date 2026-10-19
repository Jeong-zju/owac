"""N04 S0 测量几何合同测试：SE(3)、像素仿射、内参、CFA 相位、投影与支持掩码。

每个数值断言都有手算值或独立的 FP64 / numpy 参照（测试内自带 Rodrigues 实现，
不复用被测代码），不靠形状计数充当验证。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from owac.frameworks.n04.config import CFA_PATTERNS
from owac.frameworks.n04.measurement_geometry import (
    CFA_PLANE_OFFSETS,
    CameraIntrinsics,
    FrameMismatchError,
    PixelPoints,
    PixelUnitError,
    align_corners_false_from_true_xy,
    align_corners_true_from_false_xy,
    apply_se3,
    cfa_plane_coordinates,
    cfa_plane_names,
    cfa_plane_offsets,
    cfa_plane_origin,
    compose_pixel_affine,
    compose_se3,
    crop_pixel_affine,
    finite_difference_jacobian,
    in_bounds_xy,
    invert_pixel_affine,
    invert_se3,
    normalized_from_xy,
    project,
    project_with_support,
    resize_pixel_affine,
    se3_from_matrix,
    se3_from_rt,
    se3_identity,
    stride_pixel_affine,
    unproject,
    xy_from_normalized,
    z_buffer_visibility,
)
from owac.frameworks.n04.synthetic import cfa_plane_offsets as synthetic_cfa_plane_offsets

FP64 = torch.float64


def _rot_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=FP64)


def _rot_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=FP64)


def _camera_intrinsics(
    fx: float = 320.0,
    fy: float = 320.0,
    cx: float = 128.0,
    cy: float = 128.0,
    unit: str = "original_px",
) -> CameraIntrinsics:
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=FP64)
    return CameraIntrinsics(K=K, unit=unit)


def _rodrigues_np(rotvec: np.ndarray) -> np.ndarray:
    """测试内独立实现的 Rodrigues 公式，用于 numpy 参照。"""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def _points_xy(tensor: torch.Tensor, unit: str = "original_px") -> PixelPoints:
    return PixelPoints(xy=tensor.to(FP64), unit=unit)


# --------------------------------------------------------------------------------------
# SE(3)
# --------------------------------------------------------------------------------------


def test_se3_named_composition_applies_first_then_second() -> None:
    T_C_from_W = se3_from_rt(
        torch.eye(3, dtype=FP64),
        torch.tensor([1.0, 2.0, 3.0], dtype=FP64),
        to_frame="C",
        from_frame="W",
    )
    T_W_from_B = se3_from_rt(
        _rot_z(math.pi / 2), torch.zeros(3, dtype=FP64), to_frame="W", from_frame="B"
    )
    composed = compose_se3(T_C_from_W, T_W_from_B)
    assert composed.name == "T_C_from_B"
    points_B = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=FP64)
    expected = apply_se3(T_C_from_W, apply_se3(T_W_from_B, points_B))
    assert torch.allclose(composed.apply(points_B), expected, atol=1e-12)


def test_se3_frame_mismatch_raises_instead_of_guessing() -> None:
    T_C_from_W = se3_identity("C", dtype=FP64)
    T_W_from_B = se3_from_rt(
        torch.eye(3, dtype=FP64), torch.ones(3, dtype=FP64), to_frame="W", from_frame="B"
    )
    with pytest.raises(FrameMismatchError, match="from_frame"):
        compose_se3(T_W_from_B, T_C_from_W)
    with pytest.raises(FrameMismatchError, match="points_frame"):
        T_W_from_B.apply(torch.zeros(1, 3, dtype=FP64), points_frame="C")


def test_se3_inverse_round_trip_and_matrix() -> None:
    T = se3_from_rt(
        _rot_y(0.4) @ _rot_z(-0.7),
        torch.tensor([0.1, -0.2, 0.3], dtype=FP64),
        to_frame="A",
        from_frame="B",
    )
    inverse = invert_se3(T)
    assert inverse.name == "T_B_from_A"
    points = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.25, 0.75]], dtype=FP64)
    assert torch.allclose(inverse.apply(T.apply(points)), points, atol=1e-12)
    assert torch.allclose(inverse.matrix, torch.linalg.inv(T.matrix), atol=1e-12)


def test_se3_90_degree_rotation_hand_computed() -> None:
    T = se3_from_rt(_rot_z(math.pi / 2), torch.zeros(3, dtype=FP64), to_frame="C", from_frame="W")
    out = T.apply(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=FP64))
    expected = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=FP64)
    assert torch.allclose(out, expected, atol=1e-12)


def test_se3_composition_is_not_commutative_hand_computed() -> None:
    rotate = se3_from_rt(
        _rot_z(math.pi / 2), torch.zeros(3, dtype=FP64), to_frame="W", from_frame="B"
    )
    #: 在 B 系里先平移，再叠加世界系旋转：链是 T_W_from_B ∘ T_B_from_B'。
    shift_in_B = se3_from_rt(
        torch.eye(3, dtype=FP64),
        torch.tensor([1.0, 0.0, 0.0], dtype=FP64),
        to_frame="B",
        from_frame="Bp",
    )
    shift_in_W = se3_from_rt(
        torch.eye(3, dtype=FP64),
        torch.tensor([1.0, 0.0, 0.0], dtype=FP64),
        to_frame="W",
        from_frame="W",
    )
    origin = torch.zeros(1, 3, dtype=FP64)
    #: 先平移再旋转：(0,0,0) -> (1,0,0) -> (0,1,0)；先旋转再平移：(0,0,0) -> (0,0,0) -> (1,0,0)。
    assert compose_se3(rotate, shift_in_B).name == "T_W_from_Bp"
    assert torch.allclose(
        compose_se3(rotate, shift_in_B).apply(origin),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=FP64),
        atol=1e-12,
    )
    assert torch.allclose(
        compose_se3(shift_in_W, rotate).apply(origin),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=FP64),
        atol=1e-12,
    )


def test_se3_constructors_reject_non_rigid_inputs() -> None:
    with pytest.raises(ValueError, match="正交"):
        se3_from_rt(
            torch.eye(3, dtype=FP64) * 2.0,
            torch.zeros(3, dtype=FP64),
            to_frame="A",
            from_frame="B",
        )
    with pytest.raises(ValueError, match="行列式"):
        se3_from_rt(
            torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=FP64)),
            torch.zeros(3, dtype=FP64),
            to_frame="A",
            from_frame="B",
        )
    malformed = torch.eye(4, dtype=FP64)
    malformed[3, 0] = 1.0
    with pytest.raises(ValueError, match="末行"):
        se3_from_matrix(malformed, to_frame="A", from_frame="B")


# --------------------------------------------------------------------------------------
# 投影 / 逆投影
# --------------------------------------------------------------------------------------


def test_project_matches_independent_numpy_implementation() -> None:
    """FP64 下与测试内 numpy 手算（Rodrigues + K @ X / z）逐点一致。"""
    rng = np.random.default_rng(20261014)
    R = _rodrigues_np(np.array([0.35, -0.6, 0.25]))
    T_np = np.eye(4)
    T_np[:3, :3] = R
    T_np[:3, 3] = np.array([0.05, -0.12, 0.2])
    K_np = np.array([[320.0, 0.0, 128.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    points_np = rng.normal(size=(6, 3))
    points_np[:, 2] = np.abs(points_np[:, 2]) + 0.4
    camera_np = points_np @ R.T + T_np[:3, 3]
    projected_np = camera_np @ K_np.T
    expected_uv = projected_np[:, :2] / projected_np[:, 2:3]
    in_front = camera_np[:, 2] > 0.0
    #: 种子固定时至少有一个点落在相机后方，用来验证占位与掩码语义。
    assert int((~in_front).sum()) >= 1

    T = se3_from_matrix(torch.tensor(T_np), to_frame="C", from_frame="W")
    pixels = project(
        CameraIntrinsics(K=torch.tensor(K_np), unit="original_px"),
        T,
        torch.tensor(points_np),
        points_frame="W",
    )
    assert np.allclose(pixels.xy.numpy()[in_front], expected_uv[in_front], atol=1e-11)
    #: 非正深度不做除零：按合同返回 0 占位（不是 numpy 里那个被负深度“翻转”的像素）。
    assert np.allclose(pixels.xy.numpy()[~in_front], 0.0, atol=0.0)


def test_project_hand_computed_90_degree_camera_rotation() -> None:
    """K=(320, 320, 128, 128)，相机绕 C 的 z 轴转 90°：p_W=(1,0,2) -> p_C=(0,1,2)。"""
    T = se3_from_rt(_rot_z(math.pi / 2), torch.zeros(3, dtype=FP64), to_frame="C", from_frame="W")
    pixels = project(_camera_intrinsics(), T, torch.tensor([[1.0, 0.0, 2.0]], dtype=FP64))
    assert torch.allclose(pixels.xy, torch.tensor([[128.0, 288.0]], dtype=FP64), atol=1e-9)


def test_negative_and_zero_depth_are_masked_not_aliased_into_pixels() -> None:
    T = se3_identity("C", dtype=FP64)
    K = _camera_intrinsics()
    points = torch.tensor([[0.1, 0.0, -0.6], [0.1, 0.0, 0.6], [0.0, 0.0, 0.0]], dtype=FP64)
    support = project_with_support(K, T, points, grid_size_hw=(256, 256))
    assert support.in_front.tolist() == [False, True, False]
    assert support.support.tolist() == [0.0, 1.0, 0.0]
    #: 负深度不能给出“镜像像素”：占位为 0，且与同 (x,y) 正深度点的像素不同。
    assert torch.allclose(support.pixels.xy[0], torch.zeros(2, dtype=FP64))
    assert float(support.pixels.xy[1, 0]) == pytest.approx(320.0 * 0.1 / 0.6 + 128.0, abs=1e-9)
    assert torch.isfinite(support.pixels.xy).all()


def test_out_of_bounds_support_matches_explicit_domain() -> None:
    K = _camera_intrinsics()
    T = se3_identity("C", dtype=FP64)
    #: u = 320 * 0.5 / 0.6 + 128 = 394.67（越界）；v = 320 * (-0.5) / 0.6 + 128 = -138.67（越界）。
    points = torch.tensor([[0.2, 0.0, 0.6], [0.5, 0.0, 0.6], [0.0, -0.5, 0.6]], dtype=FP64)
    support = project_with_support(K, T, points, grid_size_hw=(256, 256))
    assert support.in_bounds.tolist() == [True, False, False]
    assert support.support.tolist() == [1.0, 0.0, 0.0]
    explicit = in_bounds_xy(support.pixels, (256, 256))
    assert torch.equal(explicit, support.in_bounds)
    assert float(support.pixels.xy[0, 0]) == pytest.approx(320.0 * 0.2 / 0.6 + 128.0, abs=1e-9)


def test_project_with_support_honours_external_valid_mask() -> None:
    T = se3_identity("C", dtype=FP64)
    points = torch.tensor([[0.0, 0.0, 0.6], [0.1, 0.0, 0.6]], dtype=FP64)
    valid = torch.tensor([True, False])
    support = project_with_support(
        _camera_intrinsics(), T, points, grid_size_hw=(256, 256), valid=valid
    )
    assert support.support.tolist() == [1.0, 0.0]


def test_project_unproject_round_trip_fp64() -> None:
    K = _camera_intrinsics()
    T = se3_from_rt(
        _rot_y(0.3) @ _rot_z(-0.2),
        torch.tensor([0.02, -0.03, 0.1], dtype=FP64),
        to_frame="C",
        from_frame="W",
    )
    points_W = torch.tensor([[0.2, -0.1, 0.5], [-0.3, 0.15, 0.9], [0.0, 0.0, 1.2]], dtype=FP64)
    camera = apply_se3(T, points_W)
    pixels = project(K, T, points_W)
    restored = unproject(pixels, camera[..., 2], K)
    assert torch.allclose(restored, camera, atol=1e-11)


def test_unproject_matches_independent_numpy_formula() -> None:
    K_np = np.array([[320.0, 0.0, 128.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    xy_np = np.array([[10.5, 20.5], [200.0, 30.0]])
    depth_np = np.array([0.6, 1.3])
    homogeneous = np.concatenate([xy_np, np.ones((2, 1))], axis=1)
    expected = (homogeneous @ np.linalg.inv(K_np).T) * depth_np[:, None]
    pixels = PixelPoints(xy=torch.tensor(xy_np, dtype=FP64), unit="original_px")
    restored = unproject(
        pixels,
        torch.tensor(depth_np, dtype=FP64),
        CameraIntrinsics(torch.tensor(K_np), "original_px"),
    )
    assert np.allclose(restored.numpy(), expected, atol=1e-12)


def test_z_buffer_many_to_one_keeps_nearest_positive_depth() -> None:
    pixels = _points_xy(torch.tensor([[[10.5, 20.5], [10.5, 20.5], [10.5, 20.5], [300.0, 5.0]]]))
    depth = torch.tensor([[1.0, 0.5, -0.25, 0.5]], dtype=FP64)
    result = z_buffer_visibility(pixels, depth, grid_size_hw=(256, 256))
    assert result.visible.tolist() == [[False, True, False, False]]
    assert bool(result.occupied[0, 20, 10])
    assert int(result.occupied[0].sum()) == 1
    assert float(result.depth_at_cell[0, 20, 10]) == 0.5
    assert int(result.cell_index[0, 1]) == 20 * 256 + 10
    assert int(result.cell_index[0, 2]) == -1


def test_z_buffer_tie_keeps_lowest_index() -> None:
    pixels = _points_xy(torch.tensor([[[10.5, 20.5], [10.5, 20.5]]]))
    depth = torch.tensor([[0.5, 0.5]], dtype=FP64)
    result = z_buffer_visibility(pixels, depth, grid_size_hw=(256, 256))
    assert result.visible.tolist() == [[True, False]]


# --------------------------------------------------------------------------------------
# 像素中心、仿射与内参
# --------------------------------------------------------------------------------------


def test_pixel_center_convention_and_normalized_round_trip() -> None:
    width = height = 256
    #: 像素中心 0 -> n = 2*0.5/W - 1；归一化 +1 对应 x = W - 0.5（末像素右边缘），
    #: 不是末像素中心 W-1；x = W-1 的归一化坐标略小于 1。
    xy = torch.tensor(
        [[0.0, 0.0], [width - 0.5, height - 0.5], [width - 1.0, height - 1.0]],
        dtype=FP64,
    )
    grid = normalized_from_xy(_points_xy(xy), (height, width))
    expected = torch.tensor(
        [
            [2 * 0.5 / width - 1, 2 * 0.5 / height - 1],
            [1.0, 1.0],
            [2 * (width - 0.5) / width - 1, 2 * (height - 0.5) / height - 1],
        ],
        dtype=FP64,
    )
    assert torch.allclose(grid, expected, atol=1e-12)
    assert float(grid[2, 0]) == pytest.approx(0.99609375, abs=1e-12)
    restored = xy_from_normalized(grid, (height, width), unit="original_px")
    assert torch.allclose(restored.xy, xy, atol=1e-12)


def test_in_bounds_domain_is_half_pixel_edges() -> None:
    inside = _points_xy(torch.tensor([[-0.5, -0.5], [255.5, 255.5], [0.0, 0.0]]))
    assert in_bounds_xy(inside, (256, 256)).tolist() == [True, True, True]
    outside = _points_xy(torch.tensor([[-0.51, 0.0], [255.51, 0.0]]))
    assert in_bounds_xy(outside, (256, 256)).tolist() == [False, False]


def test_align_corners_conversion_hand_computed() -> None:
    size = (4, 4)
    false_xy = _points_xy(torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]))
    true_xy = align_corners_true_from_false_xy(false_xy, size)
    #: x_true = (x_false + 0.5) * 3 / 4。
    expected = torch.tensor([[0.375, 0.375], [1.125, 1.125], [2.625, 2.625]], dtype=FP64)
    assert torch.allclose(true_xy.xy, expected, atol=1e-12)
    assert true_xy.unit == "original_px"
    round_trip = align_corners_false_from_true_xy(true_xy, size)
    assert torch.allclose(round_trip.xy, false_xy.xy, atol=1e-12)
    #: x_false = x_true * 4 / 3 - 0.5：AC=True 的 0/3 对应 False 约定的 -0.5 / 3.5。
    edges = align_corners_false_from_true_xy(
        _points_xy(torch.tensor([[0.0, 0.0], [3.0, 3.0]])), size
    )
    assert torch.allclose(
        edges.xy, torch.tensor([[-0.5, -0.5], [3.5, 3.5]], dtype=FP64), atol=1e-12
    )


def test_resize_pixel_affine_center_mapping_hand_computed() -> None:
    affine = resize_pixel_affine((4, 4), (2, 2), dtype=FP64)
    out = affine.apply(_points_xy(torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])))
    #: x_dst = 0.5 x_src - 0.25。
    expected = torch.tensor([[-0.25, -0.25], [0.25, 0.25], [1.25, 1.25]], dtype=FP64)
    assert torch.allclose(out.xy, expected, atol=1e-12)
    upsample = resize_pixel_affine((2, 2), (4, 4), dtype=FP64)
    assert torch.allclose(
        upsample.apply(_points_xy(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))).xy,
        torch.tensor([[0.5, 0.5], [2.5, 2.5]], dtype=FP64),
        atol=1e-12,
    )


def test_crop_pixel_affine_shift_and_composition() -> None:
    crop = crop_pixel_affine((1.0, 2.0), dtype=FP64)
    out = crop.apply(_points_xy(torch.tensor([[1.0, 2.0], [3.0, 5.0]])))
    assert torch.allclose(out.xy, torch.tensor([[0.0, 0.0], [2.0, 3.0]], dtype=FP64), atol=1e-12)
    #: 先裁剪再缩放：源 (1,1) -> (0,0) -> (-0.25,-0.25)。
    combined = compose_pixel_affine(
        resize_pixel_affine((4, 4), (2, 2), dtype=FP64), crop_pixel_affine((1.0, 1.0), dtype=FP64)
    )
    assert torch.allclose(
        combined.apply(_points_xy(torch.tensor([[1.0, 1.0], [3.0, 3.0]]))).xy,
        torch.tensor([[-0.25, -0.25], [0.75, 0.75]], dtype=FP64),
        atol=1e-12,
    )
    inverse = invert_pixel_affine(combined)
    assert torch.allclose(
        inverse.apply(combined.apply(_points_xy(torch.tensor([[2.0, 3.0]])))).xy,
        torch.tensor([[2.0, 3.0]], dtype=FP64),
        atol=1e-12,
    )
    assert inverse.from_unit == "original_px" and inverse.to_unit == "original_px"


def test_stride_center_offset_hand_computed() -> None:
    stride = 32.0
    center = stride_pixel_affine(stride, dtype=FP64)
    corner = stride_pixel_affine(stride, center_aligned=False, dtype=FP64)
    source = _points_xy(torch.tensor([[0.0, 0.0], [16.0, 16.0], [32.0, 32.0]]))
    center_xy = center.apply(source).xy
    corner_xy = corner.apply(source).xy
    #: 中心对齐 f=(u+0.5)/32-0.5；角点对齐（旧 GridSpec）f=u/32。
    assert torch.allclose(
        center_xy,
        torch.tensor(
            [[-0.484375, -0.484375], [0.015625, 0.015625], [0.515625, 0.515625]], dtype=FP64
        ),
        atol=1e-12,
    )
    assert torch.allclose(
        corner_xy, torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=FP64), atol=1e-12
    )
    offset = (stride - 1.0) / (2.0 * stride)
    assert offset == pytest.approx(0.484375)
    assert torch.allclose(corner_xy - center_xy, torch.full_like(center_xy, offset), atol=1e-12)
    assert center.from_unit == "original_px" and center.to_unit == "feature_px"
    back = invert_pixel_affine(center).apply(center.apply(source))
    assert torch.allclose(back.xy, source.xy, atol=1e-12)


def test_resize_intrinsics_updates_principal_point_not_only_focal() -> None:
    K = _camera_intrinsics()
    resized = K.resized((256, 256), (128, 128))
    assert float(resized.K[0, 0]) == pytest.approx(160.0)
    assert float(resized.K[0, 2]) == pytest.approx(63.75)  # 0.5 * (128 + 0.5) - 0.5
    assert resized.unit == "original_px"
    #: 与像素仿射一致：同一归一化光线 u -> u' 的两条路径必须给出同一个数。
    nx = 0.2
    u_sensor = 320.0 * nx + 128.0
    u_dst = (
        resize_pixel_affine((256, 256), (128, 128), dtype=FP64)
        .apply(_points_xy(torch.tensor([[u_sensor, 0.0]])))
        .xy[0, 0]
    )
    assert float(u_dst) == pytest.approx(
        float(resized.K[0, 0]) * nx + float(resized.K[0, 2]), abs=1e-9
    )
    with pytest.raises(ValueError, match="align_corners"):
        K.resized((256, 256), (128, 128), align_corners=True)


def test_stride_intrinsics_center_and_corner() -> None:
    K = _camera_intrinsics()
    centered = K.on_feature_grid(32.0)
    assert centered.unit == "feature_px"
    assert float(centered.K[0, 0]) == pytest.approx(10.0)
    assert float(centered.K[0, 2]) == pytest.approx(3.515625)  # (128 + 0.5) / 32 - 0.5
    corner = K.on_feature_grid(32.0, center_aligned=False)
    assert float(corner.K[0, 2]) == pytest.approx(4.0)
    #: 与 stride 仿射一致：u=200 的传感器光线在特征网格上落在 f=5.765625。
    u_sensor = 200.0
    f = (
        stride_pixel_affine(32.0, dtype=FP64)
        .apply(_points_xy(torch.tensor([[u_sensor, 0.0]])))
        .xy[0, 0]
    )
    nx = (u_sensor - 128.0) / 320.0
    assert float(f) == pytest.approx(
        float(centered.K[0, 0]) * nx + float(centered.K[0, 2]), abs=1e-9
    )
    assert float(f) == pytest.approx(5.765625, abs=1e-12)
    with pytest.raises(PixelUnitError, match="original_px"):
        centered.on_feature_grid(2.0)


def test_pixel_unit_mixing_raises_instead_of_converting() -> None:
    feature_points = PixelPoints(xy=torch.tensor([[10.0, 10.0]], dtype=FP64), unit="feature_px")
    with pytest.raises(PixelUnitError, match="feature_px"):
        unproject(feature_points, torch.tensor([0.6], dtype=FP64), _camera_intrinsics())
    with pytest.raises(PixelUnitError, match="feature_px"):
        resize_pixel_affine((256, 256), (128, 128), dtype=FP64).apply(feature_points)
    with pytest.raises(PixelUnitError, match="compose"):
        compose_pixel_affine(
            stride_pixel_affine(2.0, dtype=FP64), stride_pixel_affine(2.0, dtype=FP64)
        )
    feature_K = _camera_intrinsics(unit="feature_px")
    assert feature_K.unit == "feature_px"
    #: 单位一致的投影路径必须能正常工作。
    pixels = project(
        feature_K,
        se3_identity("C", dtype=FP64),
        torch.tensor([[0.0, 0.0, 0.6]], dtype=FP64),
    )
    assert pixels.unit == "feature_px"


def test_dtype_mismatch_and_non_float_points_raise_clear_errors() -> None:
    affine = resize_pixel_affine((4, 4), (2, 2))  # 默认 float32
    with pytest.raises(ValueError, match="dtype"):
        affine.apply(_points_xy(torch.tensor([[1.0, 1.0]])))
    with pytest.raises(TypeError):
        PixelPoints(xy=torch.tensor([[1, 2]]), unit="original_px")


# --------------------------------------------------------------------------------------
# CFA 四相位
# --------------------------------------------------------------------------------------


def test_cfa_plane_coordinates_use_phase_origins() -> None:
    #: (i, j) -> (2i + δy, 2j + δx)；GRBG 的 δ 为 Gr(0,0), R(0,1), B(1,0), Gb(1,1)。
    assert cfa_plane_coordinates("GRBG", "R", (2, 3)).tolist() == [4, 7]
    assert cfa_plane_coordinates("GRBG", "B", (2, 3)).tolist() == [5, 6]
    assert cfa_plane_coordinates("GRBG", "Gr", (2, 3)).tolist() == [4, 6]
    assert cfa_plane_coordinates("GRBG", "Gb", (2, 3)).tolist() == [5, 7]
    #: R/G1/G2/B 不得共享忽略相位的坐标：同一平面索引给出四个不同原图格点。
    coords = {
        name: tuple(cfa_plane_coordinates("GRBG", name, (2, 3)).tolist())
        for name in ("R", "Gr", "Gb", "B")
    }
    assert len(set(coords.values())) == 4


def test_cfa_plane_origins_are_distinct_for_every_pattern() -> None:
    expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for pattern in CFA_PATTERNS:
        names = cfa_plane_names(pattern)
        assert set(names) == {"R", "Gr", "Gb", "B"}
        origins = {cfa_plane_origin(pattern, name) for name in names}
        assert origins == expected
    assert cfa_plane_origin("RGGB", "R") == (0, 0)
    assert cfa_plane_origin("RGGB", "Gr") == (0, 1)
    assert cfa_plane_origin("RGGB", "Gb") == (1, 0)
    assert cfa_plane_origin("RGGB", "B") == (1, 1)


def test_cfa_g1_g2_alias_is_explicit_raster_order() -> None:
    assert cfa_plane_origin("RGGB", "G1") == cfa_plane_origin("RGGB", "Gr") == (0, 1)
    assert cfa_plane_origin("RGGB", "G2") == cfa_plane_origin("RGGB", "Gb") == (1, 0)
    assert cfa_plane_origin("GRBG", "G1") == (0, 0)
    assert cfa_plane_origin("GBRG", "G2") == (1, 1)


def test_cfa_ambiguous_or_unknown_names_raise() -> None:
    with pytest.raises(ValueError, match="不唯一"):
        cfa_plane_origin("GRBG", "G")
    with pytest.raises(ValueError, match="未知 CFA 平面"):
        cfa_plane_origin("GRBG", "X")
    with pytest.raises(ValueError, match="未知 CFA 排列"):
        cfa_plane_offsets("RGBG")
    with pytest.raises(TypeError):
        cfa_plane_coordinates("GRBG", "R", (2.0, 3.0))
    with pytest.raises(ValueError, match="index_ij"):
        cfa_plane_coordinates("GRBG", "R", (2, 3, 4))


def test_cfa_table_matches_synthetic_generator() -> None:
    for pattern in CFA_PATTERNS:
        assert tuple(CFA_PLANE_OFFSETS[pattern]) == tuple(synthetic_cfa_plane_offsets(pattern))


# --------------------------------------------------------------------------------------
# 有限差分
# --------------------------------------------------------------------------------------


def test_finite_difference_jacobian_matches_hand_derivative() -> None:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[..., 0] ** 2, x[..., 0] * x[..., 1], torch.sin(x[..., 1])], dim=-1)

    x = torch.tensor([[2.0, 3.0], [0.5, -1.0]], dtype=FP64)
    jacobian = finite_difference_jacobian(fn, x)
    for index, (x0, x1) in enumerate(((2.0, 3.0), (0.5, -1.0))):
        expected = torch.tensor([[2 * x0, 0.0], [x1, x0], [0.0, math.cos(x1)]], dtype=FP64)
        assert torch.allclose(jacobian[index], expected, atol=1e-8)
    unbatched = finite_difference_jacobian(fn, torch.tensor([2.0, 3.0], dtype=FP64))
    assert torch.allclose(unbatched, jacobian[0], atol=1e-9)


def test_finite_difference_jacobian_matches_projection_analytic() -> None:
    K = _camera_intrinsics()
    T = se3_identity("C", dtype=FP64)
    point = torch.tensor([[0.2, -0.1, 0.6]], dtype=FP64)

    def project_xy(p: torch.Tensor) -> torch.Tensor:
        return project(K, T, p).xy

    numeric = finite_difference_jacobian(project_xy, point, eps=1e-6)
    x, y, z = (float(v) for v in point[0])
    fx = fy = 320.0
    analytic = torch.tensor(
        [[[fx / z, 0.0, -fx * x / z**2], [0.0, fy / z, -fy * y / z**2]]], dtype=FP64
    )
    assert torch.allclose(numeric, analytic, atol=1e-7)

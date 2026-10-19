"""N04 S0 测量几何合同：命名 SE(3)、像素/CFA/特征坐标、投影与支持掩码。

本模块是 ``geometry.py`` 之外新增的**正式测量几何接口**（方案 §6.1、§12），把仓库约定
写成可执行检查，而不是再立一套平行约定：

1. **坐标系命名 ``T_A_from_B``**：A←B 表示把 B 系坐标变到 A 系，
   ``p_A = T_A_from_B @ p_B``。`Se3` 同时携带 ``to_frame`` / ``from_frame``；复合要求
   相邻坐标系一致，作用可以要求输入点声明坐标系，不一致直接抛 `FrameMismatchError`，
   不靠变量名猜方向。
2. **像素中心与 ``align_corners=False``**：像素坐标是连续量，整数落在像素中心；
   ``grid_sample`` 归一化坐标 ``n = 2 (x + 0.5) / W - 1``，有效采样域为
   ``x ∈ [-0.5, W-0.5]``。仓库统一 ``padding_mode="zeros"``：越界位置取值 0 且支持掩码
   置 0，绝不用 border/reflection 把没有来源的像素搬成证据。外部使用
   ``align_corners=True`` 的坐标要先经 `align_corners_false_from_true_xy` 换算到本约定；
   两种约定的同一个采样点坐标不是同一个数，不能直接拼接。
3. **单位显式**：像素坐标带 `PixelPoints.unit`（``original_px`` = 原图/测量像素，
   ``feature_px`` = stride 缩减后的特征像素），内参带 `CameraIntrinsics.unit`。跨单位
   换算必须经过显式 `PixelAffine`；单位不一致抛 `PixelUnitError`，不做隐式换算。
   深度参数一律写作 ``depth_m``（米）。完整单位表见 ``research/theory/n04_v1.md``。
4. **前向投影与反向采样分离**：`project` 回答“世界点落到哪个像素”，不产生
   ``grid_sample`` 采样网格；可见性、多对一遮挡、越界和负深度都由 `ProjectionSupport`
   与 `z_buffer_visibility` 的掩码表达，而不是靠取整后的坐标猜。
5. **CFA 四相位**：平面像素 ``(i, j)`` 对应原始格点 ``(2i + δy, 2j + δx)``；R/G1/G2/B
   各有自己的原点，模糊的 ``"G"`` 平面名直接报错，不同相位不允许共享一个忽略相位的
   坐标。``G1``/``G2`` 在本模块定义为**表中光栅序**在前/在后的绿色相位（显式约定，
   不是通用记号）。

本模块不插值 CFA 数值，也不生成 ``grid_sample`` 网格：数值重采样仍归
``geometry.py`` / ``model.py`` 的特征场路径。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = [
    "ALIGN_CORNERS",
    "CFA_PLANE_OFFSETS",
    "PADDING_MODE",
    "CameraIntrinsics",
    "FrameMismatchError",
    "PixelAffine",
    "PixelPoints",
    "PixelUnitError",
    "ProjectionSupport",
    "Se3",
    "VisibilityResult",
    "align_corners_false_from_true_xy",
    "align_corners_true_from_false_xy",
    "apply_pixel_affine",
    "apply_se3",
    "cfa_plane_coordinates",
    "cfa_plane_names",
    "cfa_plane_offsets",
    "cfa_plane_origin",
    "compose_pixel_affine",
    "compose_se3",
    "crop_pixel_affine",
    "finite_difference_jacobian",
    "identity_pixel_affine",
    "in_bounds_xy",
    "invert_pixel_affine",
    "invert_se3",
    "normalized_from_xy",
    "project",
    "project_with_support",
    "resize_pixel_affine",
    "se3_from_matrix",
    "se3_from_rt",
    "se3_identity",
    "stride_pixel_affine",
    "unproject",
    "xy_from_normalized",
    "z_buffer_visibility",
]

#: 仓库统一约定：``align_corners=False`` 与 ``padding_mode="zeros"``。外部约定先换算。
ALIGN_CORNERS = False
PADDING_MODE = "zeros"

#: 像素单位：原始测量像素 / 特征网格像素。两者不得隐式互换。
PixelUnit = Literal["original_px", "feature_px"]
_PIXEL_UNITS: tuple[PixelUnit, ...] = ("original_px", "feature_px")


class FrameMismatchError(ValueError):
    """坐标系不一致（复合或作用时相邻 frame 不匹配）。"""


class PixelUnitError(ValueError):
    """像素单位不一致（例如把 feature_px 当 original_px 使用）。"""


def _check_pixel_unit(unit: Any, what: str) -> PixelUnit:
    if unit not in _PIXEL_UNITS:
        raise PixelUnitError(f"{what}: 未知像素单位 {unit!r}，只允许 {_PIXEL_UNITS}")
    return unit


def _require_pixel_unit(actual: PixelUnit, expected: PixelUnit, what: str) -> None:
    if actual != expected:
        raise PixelUnitError(
            f"{what}: 像素单位不一致（{actual!r} != {expected!r}）；"
            "请先经 PixelAffine / 内参换算显式转换，不要直接混用"
        )


def _require_same_dtype_device(
    first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str
) -> None:
    """张量运算显式要求 dtype/device 一致：einsum 不做类型提升，混用会给出难懂的报错。"""
    if first.dtype != second.dtype or first.device != second.device:
        raise ValueError(
            f"{first_name} 与 {second_name} 的 dtype/device 不一致："
            f"{first.dtype}@{first.device} vs {second.dtype}@{second.device}；"
            "请在构造时传入相同的 dtype/device"
        )


# --------------------------------------------------------------------------------------
# SE(3)：命名 T_A_from_B
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Se3:
    """刚体变换 ``T_A_from_B``：``p_A = matrix @ p_B``。

    ``to_frame`` / ``from_frame`` 是坐标系标签（例如 ``"C"``、``"W"``、``"B"``）。
    复合、逆和作用都检查标签，防止把 ``T_W_from_C`` 当 ``T_C_from_W`` 用。
    """

    matrix: torch.Tensor
    to_frame: str
    from_frame: str

    def __post_init__(self) -> None:
        if self.matrix.shape[-2:] != (4, 4):
            raise ValueError(f"T_{self.to_frame}_from_{self.from_frame} 需要 [...,4,4] 矩阵")
        bottom = self.matrix[..., 3, :]
        target = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=bottom.dtype, device=bottom.device)
        if not torch.allclose(bottom, target, atol=1e-6):
            raise ValueError(
                f"T_{self.to_frame}_from_{self.from_frame} 的末行必须是 [0,0,0,1]，"
                f"实际 {bottom.detach().cpu().tolist()}"
            )

    @property
    def name(self) -> str:
        return f"T_{self.to_frame}_from_{self.from_frame}"

    def compose(self, inner: Se3) -> Se3:
        """``self ∘ inner``：先 ``inner`` 再 ``self``。"""
        return compose_se3(self, inner)

    def inverse(self) -> Se3:
        return invert_se3(self)

    def apply(self, points: torch.Tensor, *, points_frame: str | None = None) -> torch.Tensor:
        return apply_se3(self, points, points_frame=points_frame)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matrix": self.matrix.detach().cpu().tolist(),
        }


def se3_from_matrix(matrix: torch.Tensor, *, to_frame: str, from_frame: str) -> Se3:
    """直接包装 ``[...,4,4]`` 刚体矩阵，并记录坐标系标签。"""
    return Se3(matrix=matrix, to_frame=to_frame, from_frame=from_frame)


def se3_from_rt(
    R: torch.Tensor,
    t: torch.Tensor,
    *,
    to_frame: str,
    from_frame: str,
    tol: float = 1e-4,
) -> Se3:
    """由旋转 ``R [...,3,3]`` 与平移 ``t [...,3]`` 构造 ``T_A_from_B``。

    ``R`` 必须是（近似）正交且行列式为 +1 的旋转；显式报错而不是接受缩放/反射矩阵。
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R 需要 [...,3,3]，实际 {tuple(R.shape)}")
    if t.shape[-1] != 3:
        raise ValueError(f"t 需要 [...,3]，实际 {tuple(t.shape)}")
    eye = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
    if not torch.allclose(R @ R.transpose(-1, -2), eye, atol=tol):
        raise ValueError(f"R 不是正交矩阵（atol={tol}）")
    det = torch.linalg.det(R)
    if not torch.allclose(det, torch.ones_like(det), atol=tol):
        raise ValueError(f"R 的行列式必须为 +1（实际 {det.detach().cpu().tolist()}）")
    batch = torch.broadcast_shapes(R.shape[:-2], t.shape[:-1])
    matrix = torch.zeros(*batch, 4, 4, dtype=R.dtype, device=R.device)
    matrix[..., :3, :3] = R
    matrix[..., :3, 3] = t
    matrix[..., 3, 3] = 1.0
    return Se3(matrix=matrix, to_frame=to_frame, from_frame=from_frame)


def se3_identity(
    frame: str, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None
) -> Se3:
    """单位变换 ``T_frame_from_frame``。"""
    return Se3(matrix=torch.eye(4, dtype=dtype, device=device), to_frame=frame, from_frame=frame)


def compose_se3(second: Se3, first: Se3) -> Se3:
    """复合 ``second ∘ first``：先 ``first`` 再 ``second``，要求 ``first.to == second.from``。"""
    if second.from_frame != first.to_frame:
        raise FrameMismatchError(
            f"无法复合 {second.name} ∘ {first.name}："
            f"{second.name} 的 from_frame={second.from_frame!r} 与 "
            f"{first.name} 的 to_frame={first.to_frame!r} 不一致"
        )
    return Se3(
        matrix=second.matrix @ first.matrix,
        to_frame=second.to_frame,
        from_frame=first.from_frame,
    )


def invert_se3(T: Se3) -> Se3:
    """SE(3) 逆：``T_A_from_B`` 的逆是 ``T_B_from_A``，用 ``R^T`` 闭式解。"""
    R = T.matrix[..., :3, :3]
    t = T.matrix[..., :3, 3]
    Rt = R.transpose(-1, -2)
    inv = torch.zeros_like(T.matrix)
    inv[..., :3, :3] = Rt
    inv[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rt, t)
    inv[..., 3, 3] = 1.0
    return Se3(matrix=inv, to_frame=T.from_frame, from_frame=T.to_frame)


def apply_se3(T: Se3, points: torch.Tensor, *, points_frame: str | None = None) -> torch.Tensor:
    """``T`` 作用在 ``[...,3]`` 点上；给出 ``points_frame`` 时必须与 ``T.from_frame`` 一致。"""
    if points.shape[-1] != 3:
        raise ValueError(f"points 需要 [...,3]，实际 {tuple(points.shape)}")
    if points_frame is not None and points_frame != T.from_frame:
        raise FrameMismatchError(
            f"{T.name} 期望输入坐标系 {T.from_frame!r}，实际传入 points_frame={points_frame!r}"
        )
    _require_same_dtype_device(T.matrix, points, f"{T.name}.matrix", "points")
    R = T.matrix[..., :3, :3]
    t = T.matrix[..., :3, 3]
    return torch.einsum("...ij,...j->...i", R, points) + t


# --------------------------------------------------------------------------------------
# 像素网格与仿射
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelPoints:
    """带单位的像素坐标 ``[..., 2]``，最后一维为 ``(x, y)``（列、行）。"""

    xy: torch.Tensor
    unit: PixelUnit

    def __post_init__(self) -> None:
        _check_pixel_unit(self.unit, "PixelPoints")
        if self.xy.shape[-1] != 2:
            raise ValueError(f"PixelPoints 需要 [...,2]，实际 {tuple(self.xy.shape)}")
        if not self.xy.is_floating_point():
            raise TypeError("像素坐标是连续量，必须是浮点张量（整数索引不是像素中心）")

    def as_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "shape": list(self.xy.shape)}


@dataclass(frozen=True)
class PixelAffine:
    """像素坐标仿射 ``p_dst = matrix @ p_src + translation``，带源/目标单位。

    跨单位（``original_px`` -> ``feature_px``）的换算只能通过本类完成；``apply`` 检查
    输入单位，``compose`` 检查中间单位，单位不一致抛 `PixelUnitError`。
    """

    matrix: torch.Tensor
    translation: torch.Tensor
    from_unit: PixelUnit
    to_unit: PixelUnit

    def __post_init__(self) -> None:
        _check_pixel_unit(self.from_unit, "PixelAffine.from_unit")
        _check_pixel_unit(self.to_unit, "PixelAffine.to_unit")
        if self.matrix.shape[-2:] != (2, 2):
            raise ValueError(f"PixelAffine.matrix 需要 [...,2,2]，实际 {tuple(self.matrix.shape)}")
        if self.translation.shape[-1] != 2:
            raise ValueError(
                f"PixelAffine.translation 需要 [...,2]，实际 {tuple(self.translation.shape)}"
            )

    def apply(self, points: PixelPoints) -> PixelPoints:
        _require_pixel_unit(points.unit, self.from_unit, "PixelAffine.apply")
        _require_same_dtype_device(self.matrix, points.xy, "PixelAffine.matrix", "points.xy")
        xy = torch.einsum("...ij,...j->...i", self.matrix, points.xy) + self.translation
        return PixelPoints(xy=xy, unit=self.to_unit)

    def inverse(self) -> PixelAffine:
        return invert_pixel_affine(self)

    def compose(self, inner: PixelAffine) -> PixelAffine:
        """``self ∘ inner``：先 ``inner`` 再 ``self``。"""
        return compose_pixel_affine(self, inner)

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_unit": self.from_unit,
            "to_unit": self.to_unit,
            "matrix": self.matrix.detach().cpu().tolist(),
            "translation": self.translation.detach().cpu().tolist(),
        }


def identity_pixel_affine(
    unit: PixelUnit = "original_px",
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> PixelAffine:
    """同单位恒等仿射。"""
    return PixelAffine(
        matrix=torch.eye(2, dtype=dtype, device=device),
        translation=torch.zeros(2, dtype=dtype, device=device),
        from_unit=unit,
        to_unit=unit,
    )


def apply_pixel_affine(affine: PixelAffine, points: PixelPoints) -> PixelPoints:
    return affine.apply(points)


def invert_pixel_affine(affine: PixelAffine) -> PixelAffine:
    """``p_src = A^{-1} (p_dst - t)``，同时交换源/目标单位。"""
    inv = torch.linalg.inv(affine.matrix)
    inv_t = -torch.einsum("...ij,...j->...i", inv, affine.translation)
    return PixelAffine(
        matrix=inv,
        translation=inv_t,
        from_unit=affine.to_unit,
        to_unit=affine.from_unit,
    )


def compose_pixel_affine(second: PixelAffine, first: PixelAffine) -> PixelAffine:
    """``second ∘ first``：先 ``first`` 再 ``second``，检查中间单位一致。"""
    _require_pixel_unit(
        first.to_unit, second.from_unit, f"compose({second.from_unit}->{second.to_unit})"
    )
    matrix = second.matrix @ first.matrix
    translation = (
        torch.einsum("...ij,...j->...i", second.matrix, first.translation) + second.translation
    )
    return PixelAffine(
        matrix=matrix,
        translation=translation,
        from_unit=first.from_unit,
        to_unit=second.to_unit,
    )


def crop_pixel_affine(
    offset_xy: tuple[float, float],
    *,
    unit: PixelUnit = "original_px",
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> PixelAffine:
    """裁剪仿射：新原点在源坐标 ``offset_xy``，即 ``p_new = p_src - offset``。"""
    offset = torch.tensor([float(offset_xy[0]), float(offset_xy[1])], dtype=dtype, device=device)
    return PixelAffine(
        matrix=torch.eye(2, dtype=dtype, device=device),
        translation=-offset,
        from_unit=unit,
        to_unit=unit,
    )


def resize_pixel_affine(
    src_size_hw: tuple[int, int],
    dst_size_hw: tuple[int, int],
    *,
    unit: PixelUnit = "original_px",
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> PixelAffine:
    """缩放仿射（``align_corners=False`` 中心对齐）：``x_dst = s x_src + (s - 1) / 2``。

    ``s = dst / src`` 逐轴取值。该式来自“同一个归一化采样点”：
    ``2 (x_dst + 0.5) / W_dst - 1 = 2 (x_src + 0.5) / W_src - 1``。
    外部按 ``align_corners=True`` 缩放的坐标必须先换算，不能直接把这里的 ``s`` 搬过去。
    """
    src_h, src_w = src_size_hw
    dst_h, dst_w = dst_size_hw
    if min(src_h, src_w, dst_h, dst_w) <= 0:
        raise ValueError("resize 尺寸必须为正")
    sx = dst_w / src_w
    sy = dst_h / src_h
    matrix = torch.diag(torch.tensor([sx, sy], dtype=dtype, device=device))
    translation = torch.tensor([(sx - 1.0) / 2.0, (sy - 1.0) / 2.0], dtype=dtype, device=device)
    return PixelAffine(matrix=matrix, translation=translation, from_unit=unit, to_unit=unit)


def stride_pixel_affine(
    stride: float,
    *,
    center_aligned: bool = True,
    from_unit: PixelUnit = "original_px",
    to_unit: PixelUnit = "feature_px",
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> PixelAffine:
    """stride 缩减网格的原图 -> 特征映射。

    * ``center_aligned=True``（默认，``align_corners=False`` 一致）：
      ``f = (u + 0.5) / s - 0.5``，特征中心与 ``s`` 个原图像素块的中心对齐。
    * ``center_aligned=False``：``f = u / s``，即仓库旧 ``GridSpec.pixel_stride`` 的
      **角点对齐**约定，仅供旧接口对照；两者在 ``u`` 处的差为 ``(s - 1) / (2 s)``
      特征像素。

    两种约定都显式给出；调用方必须声明自己在哪一条上，不允许把结果互相顶替。
    """
    if stride <= 0:
        raise ValueError("stride 必须为正")
    _require_pixel_unit(to_unit, "feature_px", "stride_pixel_affine 的目标单位")
    _require_pixel_unit(from_unit, "original_px", "stride_pixel_affine 的源单位")
    if center_aligned:
        scale = 1.0 / stride
        offset = (scale - 1.0) / 2.0
    else:
        scale = 1.0 / stride
        offset = 0.0
    matrix = torch.eye(2, dtype=dtype, device=device) * scale
    translation = torch.full((2,), offset, dtype=dtype, device=device)
    return PixelAffine(matrix=matrix, translation=translation, from_unit=from_unit, to_unit=to_unit)


def normalized_from_xy(points: PixelPoints, size_hw: tuple[int, int]) -> torch.Tensor:
    """像素坐标 -> ``grid_sample`` 归一化坐标（``align_corners=False``）。"""
    height, width = size_hw
    if min(height, width) <= 0:
        raise ValueError("size_hw 必须为正")
    x = points.xy[..., 0]
    y = points.xy[..., 1]
    nx = 2.0 * (x + 0.5) / width - 1.0
    ny = 2.0 * (y + 0.5) / height - 1.0
    return torch.stack((nx, ny), dim=-1)


def xy_from_normalized(
    grid: torch.Tensor, size_hw: tuple[int, int], *, unit: PixelUnit = "original_px"
) -> PixelPoints:
    """``grid_sample`` 归一化坐标 -> 像素坐标（``align_corners=False``）。"""
    _check_pixel_unit(unit, "xy_from_normalized")
    if grid.shape[-1] != 2:
        raise ValueError(f"grid 需要 [...,2]，实际 {tuple(grid.shape)}")
    height, width = size_hw
    if min(height, width) <= 0:
        raise ValueError("size_hw 必须为正")
    nx = grid[..., 0]
    ny = grid[..., 1]
    x = (nx + 1.0) * width / 2.0 - 0.5
    y = (ny + 1.0) * height / 2.0 - 0.5
    return PixelPoints(xy=torch.stack((x, y), dim=-1), unit=unit)


def in_bounds_xy(
    points: PixelPoints, size_hw: tuple[int, int], *, tol: float = 1e-6
) -> torch.Tensor:
    """坐标是否落在 ``align_corners=False`` + ``zeros`` 的有效采样域内。

    有效域为 ``x ∈ [-0.5, W - 0.5]``：两端是首末像素中心向外半个像素的边缘。
    这与 ``geometry.GridSpec.in_bounds_xy`` 的归一化判定一致，但这里以像素坐标表达，
    便于和 `project` / `unproject` 的像素单位直接对齐。
    """
    height, width = size_hw
    x = points.xy[..., 0]
    y = points.xy[..., 1]
    return (
        (x >= -0.5 - tol) & (x <= width - 0.5 + tol) & (y >= -0.5 - tol) & (y <= height - 0.5 + tol)
    )


def align_corners_false_from_true_xy(points: PixelPoints, size_hw: tuple[int, int]) -> PixelPoints:
    """把外部 ``align_corners=True`` 的像素坐标换算到仓库的 ``False`` 约定。

    两种约定的同一个物理采样点：``2 x_true / (W - 1) = 2 (x_false + 0.5) / W``，
    因此 ``x_false = x_true W / (W - 1) - 0.5``（``W == 1`` 无定义，直接报错）。
    单位不变，只换坐标约定。
    """
    height, width = size_hw
    if width < 2 or height < 2:
        raise ValueError("align_corners 换算要求宽高 >= 2")
    x = points.xy[..., 0] * width / (width - 1) - 0.5
    y = points.xy[..., 1] * height / (height - 1) - 0.5
    return PixelPoints(xy=torch.stack((x, y), dim=-1), unit=points.unit)


def align_corners_true_from_false_xy(points: PixelPoints, size_hw: tuple[int, int]) -> PixelPoints:
    """``align_corners_false_from_true_xy`` 的逆：``x_true = (x_false + 0.5)(W - 1) / W``。"""
    height, width = size_hw
    if width < 2 or height < 2:
        raise ValueError("align_corners 换算要求宽高 >= 2")
    x = (points.xy[..., 0] + 0.5) * (width - 1) / width
    y = (points.xy[..., 1] + 0.5) * (height - 1) / height
    return PixelPoints(xy=torch.stack((x, y), dim=-1), unit=points.unit)


# --------------------------------------------------------------------------------------
# 内参
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔内参 ``K [...,3,3]`` 及其像素单位。

    ``K`` 的行列约定为 ``u = fx X/Z + cx``、``v = fy Y/Z + cy``。任何把 ``K`` 从原图
    换算到特征网格的操作都必须显式经过 `CameraIntrinsics.on_feature_grid` 或
    `CameraIntrinsics.resized`，并且这些方法按 ``align_corners=False`` 的中心公式
    同时更新主点，而不是只乘焦距。
    """

    K: torch.Tensor
    unit: PixelUnit

    def __post_init__(self) -> None:
        _check_pixel_unit(self.unit, "CameraIntrinsics")
        if self.K.shape[-2:] != (3, 3):
            raise ValueError(f"K 需要 [...,3,3]，实际 {tuple(self.K.shape)}")
        one = self.K[..., 2, 2]
        if not torch.allclose(one, torch.ones_like(one), atol=1e-6):
            raise ValueError("K[...,2,2] 必须为 1")
        fx = self.K[..., 0, 0]
        fy = self.K[..., 1, 1]
        eps = torch.finfo(self.K.dtype).eps
        if bool((fx.abs() < eps).any()) or bool((fy.abs() < eps).any()):
            raise ValueError("fx / fy 必须非零")

    def scaled(self, factor: float, *, to_unit: PixelUnit | None = None) -> CameraIntrinsics:
        """纯像素缩放 ``K' = diag(factor, factor, 1) K``（不引入中心偏移）。

        用于整幅图像在同一像素单位族内放大/缩小时的坐标缩放。stride/外部分辨率变换
        请用 `on_feature_grid` / `resized`，它们带正确的中心偏移。
        """
        if factor == 0:
            raise ValueError("factor 不能为 0")
        scale = torch.diag(
            torch.tensor(
                [float(factor), float(factor), 1.0], dtype=self.K.dtype, device=self.K.device
            )
        )
        return CameraIntrinsics(K=scale @ self.K, unit=to_unit or self.unit)

    def resized(
        self,
        src_size_hw: tuple[int, int],
        dst_size_hw: tuple[int, int],
        *,
        align_corners: bool = False,
        to_unit: PixelUnit | None = None,
    ) -> CameraIntrinsics:
        """按 ``align_corners=False`` 中心公式更新内参。

        ``sx = W_dst / W_src``，``cx' = sx (cx + 0.5) - 0.5``（``cy`` 同理，逐轴），
        ``fx' = sx fx``。只做 ``sx fx`` / ``sx cx`` 会漏掉 ``(sx - 1) / 2`` 的中心偏移。
        外部 ``align_corners=True`` 的缩放坐标先经
        `align_corners_false_from_true_xy` 换算后再进入本函数。
        """
        if align_corners:
            raise ValueError(
                "仓库固定 align_corners=False；外部 True 约定请先用 "
                "align_corners_false_from_true_xy 换算坐标，再调用本函数"
            )
        src_h, src_w = src_size_hw
        dst_h, dst_w = dst_size_hw
        if min(src_h, src_w, dst_h, dst_w) <= 0:
            raise ValueError("resize 尺寸必须为正")
        sx = dst_w / src_w
        sy = dst_h / src_h
        K = self.K.clone()
        K[..., 0, 0] = K[..., 0, 0] * sx
        K[..., 1, 1] = K[..., 1, 1] * sy
        K[..., 0, 2] = sx * (K[..., 0, 2] + 0.5) - 0.5
        K[..., 1, 2] = sy * (K[..., 1, 2] + 0.5) - 0.5
        return CameraIntrinsics(K=K, unit=to_unit or self.unit)

    def on_feature_grid(
        self, stride: float, *, center_aligned: bool = True, to_unit: PixelUnit = "feature_px"
    ) -> CameraIntrinsics:
        """原图内参 -> 特征网格内参（``stride`` = 每个特征像素覆盖的原图像素数）。

        ``center_aligned=True``：``f = (u + 0.5) / s - 0.5``，故
        ``fx' = fx / s``、``cx' = (cx + 0.5) / s - 0.5``。
        ``center_aligned=False``：旧 ``GridSpec`` 角点约定，``cx' = cx / s``。
        """
        _require_pixel_unit(self.unit, "original_px", "on_feature_grid 的输入单位")
        _require_pixel_unit(to_unit, "feature_px", "on_feature_grid 的输出单位")
        if stride <= 0:
            raise ValueError("stride 必须为正")
        K = self.K.clone()
        K[..., 0, 0] = K[..., 0, 0] / stride
        K[..., 1, 1] = K[..., 1, 1] / stride
        if center_aligned:
            K[..., 0, 2] = (K[..., 0, 2] + 0.5) / stride - 0.5
            K[..., 1, 2] = (K[..., 1, 2] + 0.5) / stride - 0.5
        else:
            K[..., 0, 2] = K[..., 0, 2] / stride
            K[..., 1, 2] = K[..., 1, 2] / stride
        return CameraIntrinsics(K=K, unit=to_unit)

    def as_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "K": self.K.detach().cpu().tolist()}


# --------------------------------------------------------------------------------------
# CFA 四相位
# --------------------------------------------------------------------------------------

#: 2x2 排列的四个子平面：``(滤色片, δy, δx)``，顺序固定为 (0,0), (0,1), (1,0), (1,1)。
CFA_PLANE_OFFSETS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "RGGB": (("R", 0, 0), ("Gr", 0, 1), ("Gb", 1, 0), ("B", 1, 1)),
    "BGGR": (("B", 0, 0), ("Gb", 0, 1), ("Gr", 1, 0), ("R", 1, 1)),
    "GRBG": (("Gr", 0, 0), ("R", 0, 1), ("B", 1, 0), ("Gb", 1, 1)),
    "GBRG": (("Gb", 0, 0), ("B", 0, 1), ("R", 1, 0), ("Gr", 1, 1)),
}


def cfa_plane_offsets(pattern: str) -> tuple[tuple[str, int, int], ...]:
    """返回 ``(滤色片, δy, δx)`` 四元组；与 ``synthetic.cfa_plane_offsets`` 必须一致。"""
    if pattern not in CFA_PLANE_OFFSETS:
        raise ValueError(f"未知 CFA 排列 {pattern!r}；允许 {tuple(CFA_PLANE_OFFSETS)}")
    return CFA_PLANE_OFFSETS[pattern]


def cfa_plane_names(pattern: str) -> tuple[str, ...]:
    """该排列下四个子平面的规范名（``R`` / ``Gr`` / ``Gb`` / ``B``）。"""
    return tuple(name for name, _, _ in cfa_plane_offsets(pattern))


def _resolve_cfa_plane(pattern: str, plane: str) -> str:
    names = cfa_plane_names(pattern)
    if plane in names:
        return plane
    if plane in ("G1", "G2"):
        greens = [name for name in names if name in ("Gr", "Gb")]
        return greens[0] if plane == "G1" else greens[1]
    if plane == "G":
        raise ValueError(
            f"CFA 排列 {pattern!r} 的平面名 'G' 不唯一：R 与 B 之间有两个绿色相位，"
            "请显式使用 'G1'/'G2' 或 'Gr'/'Gb'"
        )
    raise ValueError(f"未知 CFA 平面 {plane!r}；排列 {pattern!r} 的规范名为 {names}")


def cfa_plane_origin(pattern: str, plane: str) -> tuple[int, int]:
    """子平面原点 ``(δy, δx)``：平面像素 ``(i, j)`` 对应原图格点 ``(2i+δy, 2j+δx)``。

    ``G1``/``G2`` 在本模块定义为表中光栅序在前的第一个/第二个绿色相位。
    """
    resolved = _resolve_cfa_plane(pattern, plane)
    for name, dy, dx in cfa_plane_offsets(pattern):
        if name == resolved:
            return (int(dy), int(dx))
    raise AssertionError(f"平面 {resolved!r} 不在 {pattern!r} 表中")


def cfa_plane_coordinates(
    pattern: str, plane: str, index_ij: torch.Tensor | tuple[int, int]
) -> torch.Tensor:
    """平面索引 ``(i, j)`` -> 原图格点坐标 ``(row, col) = (2i + δy, 2j + δx)``。

    返回整数张量（``row, col`` 顺序，与原图 ``(y, x)`` 网格一致）。
    """
    dy, dx = cfa_plane_origin(pattern, plane)
    index = torch.as_tensor(index_ij)
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"平面索引必须是整数，实际 dtype={index.dtype}")
    if index.shape[-1] != 2:
        raise ValueError(f"index_ij 需要 [...,2]（i=行, j=列），实际 {tuple(index.shape)}")
    row = 2 * index[..., 0] + dy
    col = 2 * index[..., 1] + dx
    return torch.stack((row, col), dim=-1)


# --------------------------------------------------------------------------------------
# 投影、逆投影与支持掩码
# --------------------------------------------------------------------------------------


def _project_camera_points(camera_points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """相机坐标 -> 像素；非正深度用 0 占位并由调用方用掩码判定。

    ``K`` 为 ``[...,3,3]``：``[3,3]`` 与任意批形状的点一起广播；批内逐样本的 ``K``
    需要与点的前导维一致。非正深度不做除零：坐标置 0，可见性由掩码决定。
    """
    _require_same_dtype_device(K, camera_points, "K", "camera_points")
    proj = torch.einsum("...ij,...j->...i", K, camera_points)
    z = camera_points[..., 2]
    eps = torch.finfo(camera_points.dtype).eps
    in_front = z > eps
    safe_z = torch.where(in_front, z, torch.ones_like(z))
    uv = proj[..., :2] / safe_z.unsqueeze(-1)
    return torch.where(in_front.unsqueeze(-1), uv, torch.zeros_like(uv))


def project(
    K: CameraIntrinsics,
    T_C_from_W: Se3,
    points: torch.Tensor,
    *,
    points_frame: str | None = None,
    camera_frame: str | None = None,
) -> PixelPoints:
    """前向投影 ``pixel = project(K @ (T_C_from_W @ p_W))``。

    返回带 ``K.unit`` 单位的 `PixelPoints`。负深度/零深度**不会**被当成正像素使用：
    坐标用 0 占位（有限值），可见性必须由 `project_with_support` 或
    `project_with_support(...).in_front` 判定。本函数不生成 ``grid_sample`` 网格。
    """
    if camera_frame is not None and T_C_from_W.to_frame != camera_frame:
        raise FrameMismatchError(
            f"{T_C_from_W.name} 的目标坐标系是 {T_C_from_W.to_frame!r}，"
            f"与 camera_frame={camera_frame!r} 不一致"
        )
    camera_points = apply_se3(T_C_from_W, points, points_frame=points_frame)
    return PixelPoints(xy=_project_camera_points(camera_points, K.K), unit=K.unit)


def unproject(points: PixelPoints, depth_m: torch.Tensor, K: CameraIntrinsics) -> torch.Tensor:
    """逆投影 ``X_C = depth_m * K^{-1} @ [x, y, 1]``（``depth_m`` 单位：米）。

    ``points`` 的单位必须与 ``K.unit`` 一致；不一致直接抛 `PixelUnitError`，
    防止把特征像素当成原图像素喂给原图内参。
    """
    _require_pixel_unit(points.unit, K.unit, "unproject")
    _require_same_dtype_device(K.K, points.xy, "K", "points.xy")
    if tuple(depth_m.shape) != tuple(points.xy.shape[:-1]):
        raise ValueError(
            f"depth_m 形状 {tuple(depth_m.shape)} 与像素前导形状 "
            f"{tuple(points.xy.shape[:-1])} 不一致"
        )
    ones = torch.ones_like(points.xy[..., :1])
    homogeneous = torch.cat((points.xy, ones), dim=-1)
    K_inv = torch.linalg.inv(K.K)
    rays = torch.einsum("...ij,...j->...i", K_inv, homogeneous)
    return rays * depth_m.unsqueeze(-1)


@dataclass(frozen=True)
class ProjectionSupport:
    """一次投影的坐标与全部支持掩码。

    ``support`` 是 ``in_front & in_bounds & valid`` 的浮点掩码（1/0）；``pixels`` 里
    越界或负深度的坐标只是占位，**必须**结合掩码使用。多对一遮挡由
    `z_buffer_visibility` 单独计算，不由本结构猜测。
    """

    pixels: PixelPoints
    depth_m: torch.Tensor
    in_front: torch.Tensor
    in_bounds: torch.Tensor
    support: torch.Tensor
    grid_size_hw: tuple[int, int]
    valid: torch.Tensor | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.pixels.unit,
            "grid_size_hw": list(self.grid_size_hw),
            "support_fraction": float(self.support.mean()),
            "valid_provided": self.valid is not None,
        }


def project_with_support(
    K: CameraIntrinsics,
    T_C_from_W: Se3,
    points: torch.Tensor,
    *,
    grid_size_hw: tuple[int, int],
    points_frame: str | None = None,
    camera_frame: str | None = None,
    valid: torch.Tensor | None = None,
    tol: float = 1e-6,
) -> ProjectionSupport:
    """投影并同时给出 ``in_front`` / ``in_bounds`` / 组合支持掩码。

    * 负深度（``z <= eps``）：``in_front=0``，支持掩码 0；
    * 越界（``align_corners=False`` 有效域外）：``in_bounds=0``，支持掩码 0；
    * ``valid`` 提供测量维度的额外有效性（例如超出标定深度范围），逐点相与。
    """
    if camera_frame is not None and T_C_from_W.to_frame != camera_frame:
        raise FrameMismatchError(
            f"{T_C_from_W.name} 的目标坐标系是 {T_C_from_W.to_frame!r}，"
            f"与 camera_frame={camera_frame!r} 不一致"
        )
    camera_points = apply_se3(T_C_from_W, points, points_frame=points_frame)
    depth = camera_points[..., 2]
    eps = torch.finfo(camera_points.dtype).eps
    in_front = depth > eps
    pixels = PixelPoints(
        xy=_project_camera_points(camera_points, K.K),
        unit=K.unit,
    )
    in_bounds = in_bounds_xy(pixels, grid_size_hw, tol=tol)
    support = in_front & in_bounds
    if valid is not None:
        if valid.shape != support.shape:
            raise ValueError(
                f"valid 形状 {tuple(valid.shape)} 与投影点形状 {tuple(support.shape)} 不一致"
            )
        support = support & valid.bool()
    return ProjectionSupport(
        pixels=pixels,
        depth_m=depth,
        in_front=in_front,
        in_bounds=in_bounds,
        support=support.to(depth.dtype),
        grid_size_hw=grid_size_hw,
        valid=valid,
    )


@dataclass(frozen=True)
class VisibilityResult:
    """多对一 z-buffer 可见性结果。

    ``visible``：``[B, N]``，该点是否在其目标格上胜出（最小正深度，平局保留最小索引）。
    ``occupied``：``[B, H, W]``，目标格是否被任何有效点占据。
    ``depth_at_cell``：``[B, H, W]``，胜出点深度，空格为 ``inf``。
    ``cell_index``：``[B, N]`` 平铺目标格索引，无效点为 ``-1``。
    """

    visible: torch.Tensor
    occupied: torch.Tensor
    depth_at_cell: torch.Tensor
    cell_index: torch.Tensor


def z_buffer_visibility(
    points: PixelPoints,
    depth_m: torch.Tensor,
    *,
    grid_size_hw: tuple[int, int],
    valid: torch.Tensor | None = None,
) -> VisibilityResult:
    """多对一投影的可见性：同一目标格只保留最小**正**深度。

    ``points.xy`` 为 ``[B, N, 2]``，``depth_m`` 为 ``[B, N]``。``round`` 取最近整数格
    （半整数平局由 ``torch.round`` 的银行家舍入决定，属于测度零的约定）。实现显式逐点
    循环：本仓库实测 torch 2.14.1 的 scatter 系算子在重复索引下会静默丢写
    （见 ``geometry.forward_splat`` 的同款处理），锚点规模下不做这个风险交换。
    """
    if points.xy.dim() != 3 or points.xy.shape[-1] != 2:
        raise ValueError(f"points.xy 需要 [B,N,2]，实际 {tuple(points.xy.shape)}")
    if depth_m.dim() != 2 or depth_m.shape != points.xy.shape[:2]:
        raise ValueError(
            f"depth_m 需要 [B,N] 且与像素一致，实际 {tuple(depth_m.shape)} "
            f"vs {tuple(points.xy.shape[:2])}"
        )
    height, width = grid_size_hw
    if min(height, width) <= 0:
        raise ValueError("grid_size_hw 必须为正")
    batch, count, _ = points.xy.shape
    dtype = depth_m.dtype
    eps = torch.finfo(dtype).eps
    cells = points.xy.round().long()
    cx, cy = cells[..., 0], cells[..., 1]
    in_bounds = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
    in_front = torch.isfinite(depth_m) & (depth_m > eps)
    if valid is not None:
        if valid.shape != depth_m.shape:
            raise ValueError(f"valid 形状 {tuple(valid.shape)} 与 depth 不一致")
        in_front = in_front & valid.bool()
    point_ok = (in_bounds & in_front).detach().cpu()
    depth_cpu = depth_m.detach().cpu()
    cx_cpu, cy_cpu = cx.detach().cpu(), cy.detach().cpu()

    visible = torch.zeros(batch, count, dtype=torch.bool)
    occupied = torch.zeros(batch, height, width, dtype=torch.bool)
    depth_at_cell = torch.full((batch, height, width), float("inf"), dtype=dtype)
    cell_index = torch.full((batch, count), -1, dtype=torch.long)
    for bi in range(batch):
        chosen: dict[int, int] = {}
        for ni in range(count):
            if not bool(point_ok[bi, ni]):
                continue
            cell = int(cy_cpu[bi, ni]) * width + int(cx_cpu[bi, ni])
            previous = chosen.get(cell)
            if previous is None or float(depth_cpu[bi, ni]) < float(depth_cpu[bi, previous]):
                chosen[cell] = ni
        for cell, ni in chosen.items():
            row, col = divmod(cell, width)
            visible[bi, ni] = True
            occupied[bi, row, col] = True
            depth_at_cell[bi, row, col] = depth_cpu[bi, ni]
            cell_index[bi, ni] = cell
    return VisibilityResult(
        visible=visible,
        occupied=occupied,
        depth_at_cell=depth_at_cell,
        cell_index=cell_index,
    )


# --------------------------------------------------------------------------------------
# 有限差分辅助
# --------------------------------------------------------------------------------------


def finite_difference_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    eps: float | None = None,
) -> torch.Tensor:
    """中心差分 Jacobian：``fn`` 把 ``[..., n]`` 映射到 ``[..., m]``，返回 ``[..., m, n]``。

    ``eps=None`` 时逐元素步长取 ``eps_machine^(1/3) * max(1, |x|)``（中心差分的标准
    折中）；建议在 **FP64** 下做数值对照。不建立计算图（``no_grad``），输入按值处理。
    """
    with torch.no_grad():
        x = x.detach()
        if x.dim() == 0:
            raise ValueError("finite_difference_jacobian 需要至少 1 维输入")
        if not x.is_floating_point():
            raise TypeError("有限差分需要浮点输入")
        dim = x.shape[-1]
        if eps is None:
            base = torch.finfo(x.dtype).eps ** (1.0 / 3.0)
            step = base * (1.0 + x.abs())
        else:
            if eps <= 0:
                raise ValueError("eps 必须为正")
            step = torch.full_like(x, float(eps))
        columns = []
        for index in range(dim):
            direction = torch.zeros_like(x)
            direction[..., index] = step[..., index]
            plus = fn(x + direction)
            minus = fn(x - direction)
            columns.append((plus - minus) / (2.0 * step[..., index]).unsqueeze(-1))
        return torch.stack(columns, dim=-1)

"""N04 坐标、投影、warp/splat 与 validity。

本模块是整个 N04 方向的几何地基，集中固定以下约定，其他模块不得另立一套：

1. **`align_corners=False`**。归一化坐标 ``-1`` 对应像素**中心** ``0``，``+1`` 对应像素中心
   ``W-1``；像素中心的归一化坐标是 ``(2 * (i + 0.5) / W - 1)``。`N04GeometryConfig`
   直接拒绝 ``align_corners=True``。
2. **`padding_mode="zeros"`**。越界位置取零值，并且必须由 `WarpResult.validity` 标为
   ``0``。禁止 border/reflection padding：那会把上一状态边界之外、从未被观测的像素值
   搬进“历史证据”。
3. **forward 与 backward 方向分离**。`forward_*` 回答“上一帧的点在新一帧跑到哪里”，
   `backward_grid_from_*` 回答“新一帧的每个采样点应该去上一帧的哪里取值”。两者不可
   互相顶替：把 forward flow 直接塞进 backward grid 会得到方向相反的错误对应。
4. **flow 的符号约定**：``forward_flow`` 定义为“上一帧位置 ``p_prev`` 的点在当前帧出现在
   ``p_curr = p_prev + F``”，即 ``F = p_curr - p_prev``（与相机/内容运动方向同号，符合
   光流惯例）。backward 采样必须取反：``p_prev = p_curr - F``。
   ``backward_grid_from_*`` 的入参就是 ``forward_flow``，内部取负，调用方不需要自己翻符号，
   但**绝不能**把 ``forward_flow`` 直接当采样网格用。
5. **meas / feature 坐标分离**。``GridSpec.pixel_stride`` 给出原始测量像素与特征像素的
   换算；`GridSpec.pixel_xy_to_feature_xy` / `feature_xy_to_pixel_xy` 是唯一的换算入口。

CFA 数值不在本模块内插值：带采样坐标的测量先编码成特征场，再在特征坐标下传播
（见 `model.MeasurementStem`）。本模块只处理已经定义在某个规则网格上的特征场。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "AffineFlow",
    "FlowToAffineFit",
    "GridSpec",
    "WarpResult",
    "backward_grid_from_affine",
    "backward_grid_from_translation",
    "compose_affine",
    "compose_transforms",
    "forward_project_points",
    "forward_splat",
    "homography_from_rt",
    "homography_from_transform",
    "inference_pose_from_homography",
    "invert_affine",
    "invert_rigid_2d",
    "invert_transform",
    "occlusion_mask_from_flow",
    "pixel_to_normalized",
    "project_points",
    "rigid_2d_from_translation",
    "segment_min",
    "unproject_pixels",
    "warp_feature_field",
]

# --------------------------------------------------------------------------------------
# 坐标与网格
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    """一个规则采样网格的坐标约定。

    ``pixel_stride`` 是原始测量像素与特征像素之间的倍率：原始测量像素 ``u`` 对应特征
    坐标 ``u / pixel_stride``。``pixel_stride=1`` 表示特征网格与测量网格共坐标。
    """

    height: int
    width: int
    align_corners: bool = False
    padding_mode: str = "zeros"
    pixel_stride: float = 1.0

    def __post_init__(self) -> None:
        if self.align_corners:
            raise ValueError("N04 固定 align_corners=False")
        if self.padding_mode != "zeros":
            raise ValueError("N04 禁止 border/reflection padding")
        if self.height < 2 or self.width < 2:
            raise ValueError("网格至少 2x2")
        if self.pixel_stride <= 0:
            raise ValueError("pixel_stride 必须为正")

    # -- 坐标换算 ----------------------------------------------------------------

    def feature_xy_to_pixel_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """特征坐标 -> 原始测量坐标（``xy`` 最后一维为 (x, y)）。"""
        return xy * self.pixel_stride

    def pixel_xy_to_feature_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """原始测量坐标 -> 特征坐标。"""
        return xy / self.pixel_stride

    def pixel_flow_to_feature_flow(self, flow: torch.Tensor) -> torch.Tensor:
        return flow / self.pixel_stride

    def feature_flow_to_pixel_flow(self, flow: torch.Tensor) -> torch.Tensor:
        return flow * self.pixel_stride

    def normalized_from_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """特征像素坐标 -> ``grid_sample`` 归一化坐标（``align_corners=False``）。"""
        x = xy[..., 0]
        y = xy[..., 1]
        nx = 2.0 * (x + 0.5) / self.width - 1.0
        ny = 2.0 * (y + 0.5) / self.height - 1.0
        return torch.stack((nx, ny), dim=-1)

    def xy_from_normalized(self, grid: torch.Tensor) -> torch.Tensor:
        """``grid_sample`` 归一化坐标 -> 特征像素坐标。"""
        nx = grid[..., 0]
        ny = grid[..., 1]
        x = (nx + 1.0) * self.width / 2.0 - 0.5
        y = (ny + 1.0) * self.height / 2.0 - 0.5
        return torch.stack((x, y), dim=-1)

    def identity_xy(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``[B,H,W,2]`` 的特征像素坐标网格，``xy[...,0]=x``（列）、``xy[...,1]=y``（行）。"""
        ys = torch.arange(self.height, device=device, dtype=dtype)
        xs = torch.arange(self.width, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack((gx, gy), dim=-1)
        return xy.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()

    def identity_grid(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``[B,H,W,2]`` 的 ``grid_sample`` 网格，最后一维为 (nx, ny)。"""
        return self.normalized_from_xy(self.identity_xy(batch, device, dtype))

    def in_bounds_xy(self, xy: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
        """坐标是否落在 ``grid_sample`` 的**有效采样域**内。

        判定与 ``padding_mode="zeros"`` 的边界完全对齐：在 ``align_corners=False`` 下，
        归一化坐标 ``[-1, 1]`` 之间的位置落在首末像素中心之间，才可能取到真实值。
        因此这里先把像素坐标换成归一化坐标再比较，而不是直接和 ``[0, W-1]`` 比较：
        后者会被浮点舍入（``2*(0.5)/W - 1`` 往返后得到 ``-3e-8``）误判为越界，从而把
        合法的中心像素错误地标为无支持。
        """
        grid = self.normalized_from_xy(xy)
        limit = 1.0 + tol
        return (grid.abs() <= limit).all(dim=-1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "align_corners": self.align_corners,
            "padding_mode": self.padding_mode,
            "pixel_stride": self.pixel_stride,
        }


# --------------------------------------------------------------------------------------
# 平移与仿射传播
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslationFlow:
    """整场常量 forward flow（特征像素单位）：``p_curr = p_prev + (dx, dy)``。"""

    dx: float = 0.0
    dy: float = 0.0

    def forward_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """上一坐标 -> 当前坐标。"""
        return xy + torch.as_tensor([self.dx, self.dy], dtype=xy.dtype, device=xy.device)

    def backward_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """当前坐标 -> 应当采样的上一坐标（取负，不是同一个张量）。"""
        return xy - torch.as_tensor([self.dx, self.dy], dtype=xy.dtype, device=xy.device)

    def as_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([self.dx, self.dy], device=device, dtype=dtype)


@dataclass(frozen=True)
class AffineFlow:
    """二维仿射传播 ``p_current = A @ p_prev + t``（特征像素单位）。

    ``A`` 是 ``[B,2,2]`` 或 ``[2,2]``，``t`` 是 ``[B,2]`` 或 ``[2]``，作用在 (x, y) 上。
    """

    matrix: torch.Tensor
    translation: torch.Tensor

    def _broadcast(self, batch: int) -> AffineFlow:
        a = self.matrix
        t = self.translation
        if a.dim() == 2:
            a = a.unsqueeze(0).expand(batch, -1, -1)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(batch, -1)
        if a.shape[0] == 1 and batch > 1:
            a = a.expand(batch, -1, -1)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch, -1)
        return AffineFlow(a, t)

    def forward_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """``[B,H,W,2]`` 上一坐标 -> 当前坐标。"""
        b, h, w, _ = xy.shape
        flow = self._broadcast(b)
        flat = xy.reshape(b, -1, 2)
        out = torch.einsum("bij,bnj->bni", flow.matrix, flat) + flow.translation[:, None, :]
        return out.reshape(b, h, w, 2)

    def backward_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """``[B,H,W,2]`` 当前坐标 -> 应当采样的上一坐标。"""
        inv = invert_affine(self)
        return inv.forward_xy(xy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.detach().cpu().tolist(),
            "translation": self.translation.detach().cpu().tolist(),
        }


def invert_affine(flow: AffineFlow) -> AffineFlow:
    """仿射逆变换 ``p_prev = A^{-1} (p_current - t)``。"""
    inv = torch.linalg.inv(flow.matrix)
    inv_t = -torch.einsum("bij,bj->bi", inv, flow.translation)
    return AffineFlow(inv, inv_t)


def compose_affine(second: AffineFlow, first: AffineFlow) -> AffineFlow:
    """复合 ``second ∘ first``：先 ``first`` 再 ``second``。"""
    batch = max(second.matrix.shape[0], first.matrix.shape[0])
    a = second._broadcast(batch)
    b = first._broadcast(batch)
    matrix = torch.einsum("bij,bjk->bik", a.matrix, b.matrix)
    translation = torch.einsum("bij,bj->bi", a.matrix, b.translation) + a.translation
    return AffineFlow(matrix, translation)


def rigid_2d_from_translation(
    dx: float, dy: float, *, device=None, dtype=torch.float32
) -> AffineFlow:
    """纯二维平移的仿射表示。"""
    m = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
    t = torch.tensor([[dx, dy]], device=device, dtype=dtype)
    return AffineFlow(m, t)


def invert_rigid_2d(theta: float, tx: float, ty: float) -> tuple[float, float, float]:
    """平面刚体逆：``(R(θ), t)`` 的逆为 ``(R(-θ), -R(-θ) t)``。"""
    c, s = math.cos(theta), math.sin(theta)
    return -theta, -(c * tx + s * ty), -(-s * tx + c * ty)


def _translation_shift(
    dx: float | torch.Tensor,
    dy: float | torch.Tensor,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把 ``(dx, dy)`` 规范成可与 ``[B,H,W,2]`` 广播的位移张量。

    支持标量（全体共享同一平移）与逐样本张量 ``[B]``/``[B,1]``（每个 batch 项各自的
    平移，例如基线里由本体状态换算出来的位移）。混用两种形状会显式报错，不静默广播。
    """
    dx_tensor = torch.as_tensor(dx, device=device, dtype=dtype)
    dy_tensor = torch.as_tensor(dy, device=device, dtype=dtype)
    if dx_tensor.dim() == 0 and dy_tensor.dim() == 0:
        return torch.stack([dx_tensor, dy_tensor]).reshape(1, 1, 1, 2)
    dx_flat = dx_tensor.reshape(-1)
    dy_flat = dy_tensor.reshape(-1)
    if dx_flat.shape[0] != dy_flat.shape[0]:
        raise ValueError(f"dx/dy 的样本数不一致：{dx_flat.shape[0]} vs {dy_flat.shape[0]}")
    if dx_flat.shape[0] == 1:
        dx_flat = dx_flat.expand(batch)
        dy_flat = dy_flat.expand(batch)
    if dx_flat.shape[0] != batch:
        raise ValueError(f"逐样本位移的样本数 {dx_flat.shape[0]} 与 batch={batch} 不一致")
    return torch.stack([dx_flat, dy_flat], dim=-1).reshape(batch, 1, 1, 2)


def backward_grid_from_translation(
    spec: GridSpec,
    dx: float | torch.Tensor,
    dy: float | torch.Tensor,
    *,
    batch: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """常量 **forward** 平移 ``(dx, dy)`` 对应的 ``grid_sample`` 采样网格。

    内容在上一帧位于 ``p_prev``、当前帧位于 ``p_prev + (dx, dy)``；要在当前帧取值，采样
    位置必须是 ``p_prev = p_cur - (dx, dy)``。函数内部完成这一次取负，返回 ``[B,H,W,2]``
    的 (nx, ny)。调用方**不得**先取负再传入，也不得把返回的网格当作 flow。

    ``dx``/``dy`` 可以是标量，也可以是逐样本张量 ``[B]``。
    """
    xy = spec.identity_xy(batch, device or torch.device("cpu"), dtype)
    prev_xy = xy - _translation_shift(dx, dy, xy.shape[0], xy.device, xy.dtype)
    return spec.normalized_from_xy(prev_xy)


def backward_grid_from_affine(spec: GridSpec, flow: AffineFlow) -> torch.Tensor:
    """仿射 **forward** flow 对应的 ``grid_sample`` 网格（内部取逆）。"""
    xy = spec.identity_xy(
        flow.matrix.shape[0] if flow.matrix.dim() == 3 else 1,
        flow.matrix.device,
        flow.matrix.dtype,
    )
    prev_xy = flow.backward_xy(xy)
    return spec.normalized_from_xy(prev_xy)


# --------------------------------------------------------------------------------------
# validity 与遮挡
# --------------------------------------------------------------------------------------


def segment_min(keys: torch.Tensor, values: torch.Tensor, num_cells: int) -> torch.Tensor:
    """按 ``keys`` 对 ``values`` 求分段最小值。

    等价于 ``out[b, k] = min(values[b, i] for keys[b, i] == k)``，未被任何 key 命中的
    单元返回 ``inf``。用 1-D ``index_reduce_`` 实现：本仓库实测 torch 2.14.1 的
    ``scatter_reduce`` 会在含哨兵值的张量上静默丢结果（dim-0 的普通 ``scatter`` 也会
    不写入），因此几何层不依赖 ``scatter*`` 系列算子。
    """
    if keys.shape != values.shape:
        raise ValueError("keys 与 values 形状必须一致")
    b, n = keys.shape
    offsets = torch.arange(b, device=keys.device).unsqueeze(1) * num_cells
    flat_keys = (keys + offsets).reshape(-1)
    out = torch.full((b * num_cells,), float("inf"), device=values.device, dtype=values.dtype)
    #: ``include_self=False``：未被写入的单元保持初始化的 inf，而不是被自身值参与归约。
    out.index_reduce_(0, flat_keys, values.reshape(-1), reduce="amin", include_self=False)
    return out.reshape(b, num_cells)


def _sample_mask(mask: torch.Tensor, grid: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    """在源网格上采样 validity mask，与特征值使用**同一**双线性采样关系与 padding。

    度量取 ``bilinear`` 而不是 ``nearest``：nearest 在归一化坐标略小于 -1 的边界处会
    跳过零填充（round(-1-f) 可能落在 0），从而把越界位置误判为有支持。双线性与
    ``grid_sample`` 的值采样完全一致，保证 validity 与取到的值同源。
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    out = F.grid_sample(
        mask,
        grid,
        mode="bilinear",
        padding_mode=spec.padding_mode,
        align_corners=spec.align_corners,
    )
    return out.squeeze(1)


def occlusion_mask_from_flow(prev_xy: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """forward splat 的可见性：多个源点落到同一目标格时保留深度最小者。

    ``prev_xy`` 为 ``[B,H,W,2]`` 的源坐标（特征像素），``grid`` 为它们的整数化目标索引。
    返回 ``[B,H,W]``，``0`` 表示该源点被别的源点遮挡。
    """
    b, h, w = grid.shape[0], grid.shape[1], grid.shape[2]
    gx = grid[..., 0].round().long()
    gy = grid[..., 1].round().long()
    # 越界源点不参与竞争
    valid = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    key = (gy * w + gx).clamp(0, h * w - 1)
    #: 深度用平铺序号近似：约定序号越小越近（调用方传入按深度排序的坐标即可换成真实深度）。
    order = torch.arange(h * w, device=grid.device, dtype=prev_xy.dtype).reshape(1, h * w)
    order = order.expand(b, -1).contiguous()
    best = segment_min(key.reshape(b, h * w), order, h * w)
    winner = best.gather(1, key.reshape(b, -1)) == order
    return (winner.reshape(b, h, w)).to(prev_xy.dtype) * valid.to(prev_xy.dtype)


@dataclass
class WarpResult:
    """一次 warp 的全部输出。"""

    values: torch.Tensor
    validity: torch.Tensor
    grid: torch.Tensor
    out_of_bounds: torch.Tensor
    occlusion: torch.Tensor
    occlusion_checked: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_fraction(self) -> torch.Tensor:
        return self.validity.mean(dim=(-2, -1))


def warp_feature_field(
    source: torch.Tensor,
    grid: torch.Tensor,
    spec: GridSpec,
    *,
    source_validity: torch.Tensor | None = None,
    previous_xy: torch.Tensor | None = None,
    mode: str = "bilinear",
    check_occlusion: bool = False,
) -> WarpResult:
    """把上一特征场按 backward 采样关系搬到当前坐标。

    ``grid`` 是 ``grid_sample`` 的 ``[B,H,W,2]`` 归一化采样网格；它必须由
    ``backward_grid_from_*`` 或 ``AffineFlow.backward_xy`` 产生，不能填入 forward flow。

    ``validity`` 只表达“这份历史证据是否有来源”，绝不通过 padding 伪造：
    越界处 ``padding_mode="zeros"`` 产生零值，validity 同步置 0。
    """
    if grid.dim() != 4 or grid.shape[-1] != 2:
        raise ValueError("grid 必须是 [B,H,W,2]")
    if grid.shape[0] != source.shape[0]:
        raise ValueError("grid 与 source 的 batch 不一致")
    sampled = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode=spec.padding_mode,
        align_corners=spec.align_corners,
    )
    xy = spec.xy_from_normalized(grid)
    in_bounds = spec.in_bounds_xy(xy)
    validity = in_bounds.to(source.dtype)
    if source_validity is not None:
        validity = validity * _sample_mask(source_validity, grid, spec)
    occlusion = torch.ones_like(validity)
    occlusion_checked = False
    if check_occlusion:
        if previous_xy is None:
            raise ValueError("check_occlusion=True 需要 previous_xy（源侧 forward 坐标）")
        occlusion = occlusion_mask_from_flow(previous_xy, xy)
        occlusion_checked = True
        validity = validity * occlusion
    return WarpResult(
        values=sampled * validity.unsqueeze(1),
        validity=validity,
        grid=grid,
        out_of_bounds=(~in_bounds).to(source.dtype),
        occlusion=occlusion,
        occlusion_checked=occlusion_checked,
        info={"mode": mode, "align_corners": spec.align_corners, "padding_mode": spec.padding_mode},
    )


def forward_splat(
    source: torch.Tensor,
    source_xy: torch.Tensor,
    source_validity: torch.Tensor | None = None,
    *,
    source_depth: torch.Tensor | None = None,
    spec: GridSpec | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """带可见性处理的 forward splat。

    返回 ``(values, validity, occlusion)``，形状 ``[B,C,H,W]`` / ``[B,H,W]`` /
    ``[B,H,W]``。同格多候选时保留 ``source_depth`` 最小者（无深度时用平铺序号替代，
    并在文档中明确这是无深度信息的近似）；落败候选 ``validity=0``，未被任何源点占据的
    目标格 ``occlusion=0``。用于显式几何的可见性处理与诊断；主链路 backward warp 走
    `warp_feature_field`。

    实现故意写成显式的候选循环：本仓库实测 torch 2.14.1 的 ``scatter``/高级索引赋值
    在重复索引、部分形状下会**静默丢写**，因此这里不依赖任何 scatter 系算子。
    """
    if spec is None:
        spec = GridSpec(source.shape[-2], source.shape[-1])
    b, c, h, w = source.shape
    n = h * w
    dst = source_xy.reshape(b, n, 2)
    src_flat = source.reshape(b, c, n)
    if source_depth is None:
        depth = torch.arange(n, device=source.device, dtype=source.dtype).reshape(1, n).expand(b, n)
    else:
        depth = source_depth.reshape(b, n)
    values = torch.zeros(b, c, n, device=source.device, dtype=source.dtype)
    valid = torch.zeros(b, n, device=source.device, dtype=source.dtype)
    occ = torch.zeros(b, n, device=source.device, dtype=source.dtype)
    for bi in range(b):
        slots: dict[int, int] = {}
        for si in range(n):
            x = int(dst[bi, si, 0].round().item())
            y = int(dst[bi, si, 1].round().item())
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            cell = y * w + x
            prev = slots.get(cell)
            if prev is None or float(depth[bi, si]) < float(depth[bi, prev]):
                slots[cell] = si
        for cell, si in slots.items():
            values[bi, :, cell] = src_flat[bi, :, si]
            occ[bi, cell] = 1.0
            valid[bi, cell] = 1.0
    if source_validity is not None:
        sv = source_validity.reshape(b, n)
        #: 源侧无效的测量不能通过 splat 变成有效证据，逐格重新判定。
        for bi in range(b):
            for cell in range(n):
                if valid[bi, cell] > 0:
                    winner = None
                    for si in range(n):
                        x = int(dst[bi, si, 0].round().item())
                        y = int(dst[bi, si, 1].round().item())
                        if x < 0 or x >= w or y < 0 or y >= h or y * w + x != cell:
                            continue
                        if winner is None or float(depth[bi, si]) < float(depth[bi, winner]):
                            winner = si
                    if winner is None or float(sv[bi, winner]) == 0.0:
                        valid[bi, cell] = 0.0
                        values[bi, :, cell] = 0.0
    return values.reshape(b, c, h, w), valid.reshape(b, h, w), occ.reshape(b, h, w)


# --------------------------------------------------------------------------------------
# 显式几何：投影与同构
# --------------------------------------------------------------------------------------


def pixel_to_normalized(u: torch.Tensor, v: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """像素坐标 -> 归一化相机坐标（不含深度）。``K`` 为 ``[B,3,3]`` 或 ``[3,3]``。

    返回 ``[...,3]``，最后一维为 ``(x/z, y/z, 1)`` 的齐次形式。
    """
    if K.dim() == 2:
        K = K.unsqueeze(0)
    b = K.shape[0]
    ones = torch.ones_like(u)
    pix = torch.stack((u, v, ones), dim=-1)
    if pix.dim() == 3:
        pix = pix.unsqueeze(0).expand(b, -1, -1, -1)
    K_inv = torch.linalg.inv(K)
    flat = pix.reshape(b, -1, 3)
    rays = torch.einsum("bij,bnj->bni", K_inv, flat)
    return rays.reshape(*pix.shape[:-1], 3)


def unproject_pixels(pixels_xy: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """像素 + 深度 -> 相机坐标 ``X_prev = d * K^{-1} [u,v,1]``。

    ``pixels_xy`` 为 ``[B,H,W,2]``（或 ``[H,W,2]``），``depth`` 为 ``[B,H,W]``（或
    ``[H,W]``），``K`` 为 ``[B,3,3]``（或 ``[3,3]``）。返回 ``[B,H,W,3]``。
    """
    squeeze_b = pixels_xy.dim() == 3
    if squeeze_b:
        pixels_xy = pixels_xy.unsqueeze(0)
        depth = depth.unsqueeze(0)
    b = pixels_xy.shape[0]
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(b, -1, -1)
    rays = pixel_to_normalized(pixels_xy[..., 0], pixels_xy[..., 1], K)
    x = rays * depth.unsqueeze(-1)
    return x.squeeze(0) if squeeze_b else x


def project_points(points_cam: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """相机坐标 -> ``p = project(K @ X)``。

    返回 ``(pixels_xy [B,H,W,2], in_front [B,H,W])``；``in_front`` 表示深度为正。
    """
    squeeze_b = points_cam.dim() == 3
    if squeeze_b:
        points_cam = points_cam.unsqueeze(0)
    b = points_cam.shape[0]
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(b, -1, -1)
    flat = points_cam.reshape(b, -1, 3)
    proj = torch.einsum("bij,bnj->bni", K, flat)
    z = proj[..., 2]
    eps = torch.finfo(proj.dtype).eps
    safe_z = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
    uv = proj[..., :2] / safe_z.unsqueeze(-1)
    uv = uv.reshape(*points_cam.shape[:-1], 2)
    in_front = (z > eps).reshape(points_cam.shape[:-1])
    if squeeze_b:
        return uv.squeeze(0), in_front.squeeze(0)
    return uv, in_front


def invert_transform(T: torch.Tensor) -> torch.Tensor:
    """SE(3) 逆，``T`` 为 ``[...,4,4]``。"""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def compose_transforms(second: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
    """复合 ``second @ first``。"""
    return torch.matmul(second, first)


def forward_project_points(
    pixels_prev: torch.Tensor,
    depth_prev: torch.Tensor,
    K_prev: torch.Tensor,
    K_current: torch.Tensor,
    T_world_from_prev: torch.Tensor,
    T_world_from_current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """显式几何的 forward projection。

    约定 ``T_world_from_camera`` 把相机坐标送到世界坐标：:

        X_prev = d * inverse(K_prev) @ [u, v, 1]
        T_current_from_prev = inverse(T_world_from_current) @ T_world_from_prev
        X_current = T_current_from_prev @ X_prev
        p_current = project(K_current @ X_current)

    返回 ``(pixels_current [B,H,W,2], in_front [B,H,W])``。
    """
    X_prev = unproject_pixels(pixels_prev, depth_prev, K_prev)
    T_current_from_prev = compose_transforms(
        invert_transform(T_world_from_current), T_world_from_prev
    )
    R = T_current_from_prev[..., :3, :3]
    t = T_current_from_prev[..., :3, 3]
    X_current = torch.einsum("...ij,...j->...i", R, X_prev) + t.unsqueeze(-2)
    return project_points(X_current, K_current)


def homography_from_rt(
    K_prev: torch.Tensor,
    K_current: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    plane_normal_prev: torch.Tensor,
    plane_distance: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """平面诱导同构 ``H = K_cur (R - t n^T / d) K_prev^{-1}``。

    ``plane_normal_prev`` 是上一相机坐标系下的平面法向（``[B,3]`` 或 ``[3]``），
    ``plane_distance`` 是平面到上一相机中心的有符号距离。返回 ``[B,3,3]``，作用于
    上一像素齐次坐标得到当前像素齐次坐标（未归一化）。
    """
    if K_prev.dim() == 2:
        K_prev = K_prev.unsqueeze(0)
    if K_current.dim() == 2:
        K_current = K_current.unsqueeze(0)
    b = K_current.shape[0]
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if plane_normal_prev.dim() == 1:
        plane_normal_prev = plane_normal_prev.unsqueeze(0).expand(b, -1)
    if not torch.is_tensor(plane_distance):
        plane_distance = torch.full(
            (b,), float(plane_distance), device=K_prev.device, dtype=K_prev.dtype
        )
    d = plane_distance.reshape(b, 1, 1)
    # t n^T / d 为外积；H = K_cur (R - t n^T / d) K_prev^{-1}
    outer = t.unsqueeze(-1) * plane_normal_prev.unsqueeze(-2)
    inner = R - outer / d
    K_prev_inv = torch.linalg.inv(K_prev)
    return torch.einsum("bij,bjk,bkl->bil", K_current, inner, K_prev_inv)


def homography_from_transform(
    K_prev: torch.Tensor,
    K_current: torch.Tensor,
    T_world_from_prev: torch.Tensor,
    T_world_from_current: torch.Tensor,
    plane_normal_prev: torch.Tensor,
    plane_distance: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """从两个 ``T_world_from_camera`` 与一个显式场景平面得到同构。"""
    T_cur_from_prev = compose_transforms(invert_transform(T_world_from_current), T_world_from_prev)
    return homography_from_rt(
        K_prev,
        K_current,
        T_cur_from_prev[..., :3, :3],
        T_cur_from_prev[..., :3, 3],
        plane_normal_prev,
        plane_distance,
    )


def homography_to_affine_on_plane(
    H: torch.Tensor,
    pixels_xy: torch.Tensor,
    spec: GridSpec,
) -> AffineFlow:
    """把作用于齐次像素的同构局部线性化成特征坐标下的仿射。

    在参考点集合 ``pixels_xy``（测量像素坐标）上做最小二乘，得到特征坐标下的 ``(A, t)``。
    这只在参考点覆盖的邻域内有效，用于把平面同构接到 `warp_feature_field`；
    全图同构仍应直接用逐点投影。
    """
    b, h, w, _ = pixels_xy.shape
    ones = torch.ones(b, h, w, 1, device=pixels_xy.device, dtype=pixels_xy.dtype)
    hom = torch.cat((pixels_xy, ones), dim=-1)
    mapped = torch.einsum("bij,bhwj->bhwi", H, hom)
    z = mapped[..., 2:3]
    eps = torch.finfo(mapped.dtype).eps
    z = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
    dst = mapped[..., :2] / z
    x = pixels_xy.reshape(b, -1, 2)
    y = dst.reshape(b, -1, 2)
    xm = x.mean(dim=1, keepdim=True)
    ym = y.mean(dim=1, keepdim=True)
    xc = x - xm
    yc = y - ym
    A = torch.einsum("bni,bnj->bij", yc, xc) @ torch.linalg.inv(
        torch.einsum("bni,bnj->bij", xc, xc) + 1e-6 * torch.eye(2, device=x.device, dtype=x.dtype)
    )
    t = (ym - torch.einsum("bij,bj->bi", A, xm.squeeze(1))).squeeze(1)
    # 像素坐标 -> 特征坐标
    s = spec.pixel_stride
    return AffineFlow(A.unsqueeze(0).expand(b, -1, -1) if A.dim() == 2 else A, t / s)


@dataclass
class FlowToAffineFit:
    """把逐点 forward flow 拟合成仿射，并报告残差。"""

    flow: AffineFlow
    residual: torch.Tensor
    valid_fraction: torch.Tensor


def inference_pose_from_homography(
    H: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """从同构与内参解出 ``(R, t/d)``；仅供显式几何诊断，不是主链路输入。"""
    if K.dim() == 2:
        K = K.unsqueeze(0)
    b = K.shape[0]
    K_inv = torch.linalg.inv(K)
    if H.dim() == 2:
        H = H.unsqueeze(0).expand(b, -1, -1)
    M = torch.einsum("bij,bjk->bik", K_inv, H)
    scale = M.norm(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(M.dtype).eps)
    M = M / scale
    r1 = M[..., 0]
    r2 = M[..., 1]
    r3 = torch.cross(r1, r2, dim=-1)
    R = torch.stack((r1, r2, r3), dim=-1)
    u, _, vh = torch.linalg.svd(R)
    R = torch.einsum("bij,bjk->bik", u, vh)
    return R, M[..., 2]

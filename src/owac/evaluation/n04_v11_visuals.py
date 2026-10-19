"""N04 v1.1 S2 训练／推理可视化：官方 dev 固定片段的**同帧**叠图。

只读模型与数据：不写训练状态、不改训练／模型代码。真实 checkpoint + 官方 dev 帧才出图；
没有 checkpoint 时明确报“尚未生成”，不生成示例图。产出写到 ``--output``（run 的 ``artifacts/``）::

    <output>/<platform>-<arm>-seed<seed>/index.json
    <output>/<platform>-<arm>-seed<seed>/clip_01/frame_0000012.png

一帧一张图，七块面板固定同坐标（原始像素：x∈[0,W-1]，y∈[0,H-1]，整数值即像素中心）：

1. RAW 固定显示映射（平台固定，**不逐帧自动拉伸**）
2. 同源 RGB（官方 ``rgb/`` 树同名相对路径；缺失时标“未提供”，不用 RAW 冒充）
3. flow 对应：HSV 图 + 均匀格点箭头（flow 定义在上一帧坐标，箭头 1:1 像素位移）
4. 当前质量 R：每个支撑点的 ``log10(mean R)`` 叠在原图上（固定色标）
5. 两模式权重：修订模式后验叠在原图上（固定色标）
6. 支持轨迹：同 ID 连线，越界／遮挡标注
7. readout attention：模型读取分布叠在原图上（相对均匀读取，非像素真值）

固定显示映射（与 ``owac.simulation.raw_preview.raw_display`` 同一 log 压缩形式）：

* ``display(v) = log1p(G * clip(v / white_level, 0, 1)) / log1p(G)``
* ``libero``：``v`` = RAW uint8 原始计数（模型输入 = ``v / 255``），``white_level = 255``，
  ``G = 15``
* ``robotwin``：``v`` = RAW float32 原值（identity 缩放），``white_level = 1.0``、``G = 1000``
* 质量 R 用 ``log10(mean_j R_j)``，固定 ``[-3, 3]``；模式权重固定 ``[0, 1]``；
  attention 用“相对均匀读取倍数” ``a * M``，固定 ``[0, 3]``（1.0 = 均匀）。

图例约定：叠加图上的箭头／热图／点都是**模型估计**；attention 是模型读取分布，不是像素真值；
潜在方差 R 不是毫米；模式权重不是外部运动概率。
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ATTENTION_VMAX",
    "FIGURE_SIZE_IN",
    "FLOW_HALF_VALUE_PX",
    "QUALITY_LOG10_RANGE",
    "RAW_DISPLAY",
    "FrameVisual",
    "RawDisplay",
    "apply_thread_limit",
    "as_hwc",
    "artifact_tag",
    "attention_to_map",
    "build_parser",
    "compose_frame",
    "display_columns",
    "display_scale_for",
    "expected_png_size",
    "fixed_display",
    "flow_arrows",
    "flow_hsv",
    "grid_cell_pixels",
    "grid_to_pixel",
    "info_lines",
    "main",
    "overlay_scalar",
    "points_hw2",
    "parse_clip_override",
    "raw_display_rgb",
    "rgb_member_candidates",
    "rgb_panel",
    "run",
    "select_clips",
    "short_relpath",
    "format_direction",
    "format_summary_table",
    "support_scatter",
    "summarize_frames",
    "write_png",
]

# --------------------------------------------------------------------------------------
# 固定显示映射
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawDisplay:
    """平台固定的 RAW 显示参数；一次冻结后不随帧变化。"""

    white_level: float
    log_gain: float
    unit: str
    unit_en: str
    measured_on: str

    def formula(self) -> str:
        return (
            f"display(v) = log1p({self.log_gain:g} * clip(v / {self.white_level:g}, 0, 1)) "
            f"/ log1p({self.log_gain:g})"
        )


#: LIBERO RAW 是 uint8 pseudo-RAW（RGB8 逆处理）：模型输入 = 计数 / 255。
#: ``white_level=255`` 取 uint8 上限；``G=15`` 沿用 ``simulation.raw_preview.raw_display``
#: 的固定 log 增益。
#: RoboTwin RAW 是 HDR-derived float32（identity 缩放），官方数据实测取值 ∈ [0, 1]，逐 episode
#: 中位数跨越约 3 个数量级（深色 episode p50≈1.6e-3），所以取 ``G=1000`` 的固定 log 压缩：
#: 既不做逐帧 min-max，也不逐帧取分位数。
RAW_DISPLAY: dict[str, RawDisplay] = {
    "libero": RawDisplay(
        white_level=255.0,
        log_gain=15.0,
        unit="uint8 原始计数（缩放由编码器侧统一施加）",
        unit_en="uint8 counts (encoder-side InputScale)",
        measured_on="官方 libero/raw 抽样：per-file max ∈ [7, 255]（2026-10-15 远端实测）",
    ),
    "robotwin": RawDisplay(
        white_level=1.0,
        log_gain=1000.0,
        unit="float32 原值（identity 缩放）",
        unit_en="float32 raw values (identity scale)",
        measured_on="官方 robotwin2/raw 抽样：per-file max ∈ [0.003, 1.0]（2026-10-15 远端实测）",
    ),
}

#: 质量 R 的固定显示范围：``log10(mean_j R_j)``。
QUALITY_LOG10_RANGE: tuple[float, float] = (-3.0, 3.0)
#: readout attention 的固定显示上限：相对均匀读取的倍数（1.0 = 把稠密网格当作均匀分布读取）。
ATTENTION_VMAX = 3.0
#: HSV flow 图的固定半值半径：``value = r / (r + FLOW_HALF_VALUE_PX)``（r 单位像素）。
FLOW_HALF_VALUE_PX = 8.0
#: 箭头采样步长（原始像素）。
FLOW_ARROW_STRIDE = 32
#: 图窗尺寸（英寸）；PNG 像素尺寸 = round(figsize * dpi)。
FIGURE_SIZE_IN = (20.0, 10.5)


def fixed_display(values: np.ndarray, *, white_level: float, log_gain: float) -> np.ndarray:
    """固定黑白位 + log 压缩：``log1p(G * clip(v / white_level, 0, 1)) / log1p(G)``。

    与 ``owac.simulation.raw_preview.raw_display`` 同一形式；任何帧都走同一组常量，
    不做 per-frame min-max／分位数拉伸。
    """
    if white_level <= 0 or log_gain <= 0:
        raise ValueError("white_level 与 log_gain 必须为正")
    array = np.asarray(values, dtype=np.float32)
    linear = np.clip(array / float(white_level), 0.0, 1.0)
    return np.log1p(float(log_gain) * linear) / np.log1p(float(log_gain))


def display_scale_for(platform: str) -> float:
    """模型输入的 RAW 数值 -> 该平台**原始单位**的换算系数（显示映射只在原始单位上定义）。

    数值来源是数据模块的 ``PLATFORM_SPECS[platform].value_scale``，可视化不另立约定：
    ``uint8_0_255``（LIBERO 现状）与 ``float32_linear``（RoboTwin）已是原始单位 → 1.0；
    旧的 ``uint8_div255`` 输入已除过 255 → 255.0。未知量纲直接报错，不猜。
    """
    from ..raw.n04_v11_data import PLATFORM_SPECS

    scale = PLATFORM_SPECS[platform].value_scale
    if scale in ("uint8_0_255", "float32_linear", "identity"):
        return 1.0
    if scale == "uint8_div255":
        return 255.0
    raise ValueError(f"未知 RAW 量纲 {scale!r}（{platform}）；不确定就不要猜显示映射")


def raw_display_rgb(values: Any, platform: str) -> np.ndarray:
    """RAW -> 固定映射 uint8 RGB 图（``[H,W,3]``），输入是模型实际读到的 RAW 数值。

    先按 :func:`display_scale_for` 换回平台原始单位（LIBERO 计数、RoboTwin 原值），
    再套固定的 log 压缩；``white_level``／``log_gain`` 是平台常量，不随帧变化。
    """
    if platform not in RAW_DISPLAY:
        raise KeyError(f"未知平台 {platform!r}；已有 {sorted(RAW_DISPLAY)}")
    spec = RAW_DISPLAY[platform]
    hwc = as_hwc(values)
    if hwc.shape[-1] == 1:
        hwc = np.repeat(hwc, 3, axis=-1)
    if hwc.shape[-1] > 3:
        hwc = hwc[..., :3]
    mapped = fixed_display(
        hwc * display_scale_for(platform), white_level=spec.white_level, log_gain=spec.log_gain
    )
    return np.rint(mapped * 255.0).astype(np.uint8)


def as_hwc(values: Any) -> np.ndarray:
    """``[H,W,C]`` 或 ``[C,H,W]`` 张量／数组 -> ``[H,W,C]`` float32（不缩放、不裁剪）。"""
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array[..., None]
    if array.ndim != 3:
        raise ValueError(f"需要 [H,W,C] 或 [C,H,W]，实际 {array.shape}")
    if array.shape[0] in (1, 2, 3, 4) and array.shape[-1] not in (1, 2, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    return array


def _as_hw2(flow_px: Any) -> np.ndarray:
    """flow -> ``[H,W,2]`` float32；接受 ``[H,W,2]``、``[2,H,W]``、``[1,2,H,W]``。"""
    array = flow_px.detach().cpu().numpy() if hasattr(flow_px, "detach") else np.asarray(flow_px)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 2 and array.shape[-1] != 2:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"flow 需要 [H,W,2]（或 [2,H,W]），实际 {array.shape}")
    return array


def rgb_panel(rgb: Any, camera: str) -> tuple[np.ndarray | None, str]:
    """同源 RGB 面板：缺失时返回 ``(None, "…未提供…")``，不用 RAW 或任何重建冒充。"""
    if rgb is None:
        return None, f"同源 RGB（{camera}）：未提供"
    hwc = as_hwc(rgb)
    if hwc.shape[-1] == 1:
        hwc = np.repeat(hwc, 3, axis=-1)
    if hwc.shape[-1] < 3:
        return None, f"同源 RGB（{camera}）：未提供（通道数 {hwc.shape[-1]}）"
    return np.clip(hwc[..., :3], 0.0, 255.0).astype(np.uint8), f"同源 RGB（{camera}）"


# --------------------------------------------------------------------------------------
# flow / 支撑 / attention 的坐标与编码
# --------------------------------------------------------------------------------------


def flow_hsv(flow_px: Any) -> np.ndarray:
    """flow ``[H,W,2]``（像素，上一帧坐标）-> 固定 HSV 编码 ``[H,W,3]`` uint8。

    色相 = ``atan2(v, u) / 2π mod 1``（0 = +x 向右，0.25 = +y 向下，0.5 = -x，0.75 = -y）；
    明度 = ``r / (r + FLOW_HALF_VALUE_PX)``（r 是位移模长，单位像素），饱和固定 1。无逐帧归一化。
    """
    flow = _as_hw2(flow_px)
    u = flow[..., 0]
    v = flow[..., 1]
    radius = np.hypot(u, v)
    hue = (np.arctan2(v, u) / (2.0 * math.pi)) % 1.0
    value = radius / (radius + FLOW_HALF_VALUE_PX)
    rgb = np.empty((*hue.shape, 3), dtype=np.float32)
    flat_hue = hue.reshape(-1)
    flat_value = value.reshape(-1)
    flat_rgb = rgb.reshape(-1, 3)
    for index in range(flat_hue.size):
        flat_rgb[index] = colorsys.hsv_to_rgb(float(flat_hue[index]), 1.0, float(flat_value[index]))
    return np.rint(rgb * 255.0).astype(np.uint8)


def flow_arrows(flow_px: Any, *, stride: int = FLOW_ARROW_STRIDE) -> tuple[np.ndarray, ...]:
    """均匀格点采样：返回 ``(x, y, u, v)``；箭头尾在 ``(x,y)``、头在 ``(x+u, y+v)``（像素）。"""
    flow = _as_hw2(flow_px)
    if stride < 1:
        raise ValueError("stride 至少为 1")
    height, width = flow.shape[:2]
    ys = np.arange(stride // 2, height, stride, dtype=np.float32)
    xs = np.arange(stride // 2, width, stride, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    iy = np.clip(grid_y.astype(int), 0, height - 1)
    ix = np.clip(grid_x.astype(int), 0, width - 1)
    return grid_x, grid_y, flow[iy, ix, 0], flow[iy, ix, 1]


def grid_to_pixel(
    iy: int, ix: int, size_hw: tuple[int, int], grid_hw: tuple[int, int]
) -> tuple[float, float]:
    """特征网格格点 ``(iy, ix)`` 的中心 -> 原始像素坐标 ``(x, y)``（与 ``px_to_grid`` 互逆）。"""
    height, width = int(size_hw[0]), int(size_hw[1])
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    if not (0 <= iy < grid_h and 0 <= ix < grid_w):
        raise IndexError(f"格点 ({iy}, {ix}) 超出 {grid_hw}")
    return (ix + 0.5) * width / grid_w - 0.5, (iy + 0.5) * height / grid_h - 0.5


def grid_cell_pixels(
    iy: int, ix: int, size_hw: tuple[int, int], grid_hw: tuple[int, int]
) -> tuple[int, int, int, int]:
    """特征网格格点覆盖的像素矩形 ``(x0, y0, x1, y1)``（闭区间）。"""
    height, width = int(size_hw[0]), int(size_hw[1])
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    x0 = int(math.floor(ix * width / grid_w))
    x1 = max(x0, int(math.ceil((ix + 1) * width / grid_w)) - 1)
    y0 = int(math.floor(iy * height / grid_h))
    y1 = max(y0, int(math.ceil((iy + 1) * height / grid_h)) - 1)
    return x0, y0, x1, y1


def resize_map(values: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """把特征网格标量图双线性放大到原始像素 ``size_hw``（只用于叠加显示，不改数值语义）。"""
    from PIL import Image

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"需要二维标量图，实际 {array.shape}")
    height, width = int(size_hw[0]), int(size_hw[1])
    if array.shape == (height, width):
        return array.copy()
    image = Image.fromarray(array, mode="F")
    return np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.float32)


def attention_to_map(
    attention: Any, *, dense_tokens: int, latent_hw: tuple[int, int]
) -> np.ndarray:
    """末块 readout attention -> ``[h,w]`` 相对均匀读取倍数。

    ``attention`` 是读出末块对全部媒体 token 的注意力（跨头平均），形状 ``[queries, media]``
    或 ``[1, queries, media]``；前 ``dense_tokens`` 列与当前帧稠密网格 ``latent_hw`` 按行主序对齐。
    返回 ``a * media``：1.0 表示与该网格上的均匀读取同强度（固定标度，不做逐帧归一化）。
    """
    array = (
        attention.detach().cpu().numpy() if hasattr(attention, "detach") else np.asarray(attention)
    )
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"attention 需要单序列，实际 {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"attention 需要 [queries, media]，实际 {array.shape}")
    _, media = array.shape
    grid_h, grid_w = int(latent_hw[0]), int(latent_hw[1])
    if int(dense_tokens) != grid_h * grid_w:
        raise ValueError(f"dense_tokens={dense_tokens} 与 latent_hw={latent_hw} 不一致")
    if media < int(dense_tokens):
        raise ValueError(f"attention 媒体列数 {media} 少于稠密 token 数 {dense_tokens}")
    dense = array[:, : int(dense_tokens)].mean(axis=0)
    return (dense * float(media)).reshape(grid_h, grid_w)


def support_scatter(
    xy_px: Any,
    size_hw: tuple[int, int],
    *,
    valid: Any = None,
    subset: int | None = None,
) -> dict[str, np.ndarray]:
    """把支撑点分成三类：帧内有效／帧内失效（遮挡）／越界；坐标保持原始像素。"""
    points = np.asarray(
        xy_px.detach().cpu().numpy() if hasattr(xy_px, "detach") else xy_px, dtype=np.float32
    ).reshape(-1, 2)
    ok = None if valid is None else np.asarray(valid).reshape(-1).astype(bool)
    if subset:
        points = points[:: int(subset)]
        if ok is not None:
            ok = ok[:: int(subset)]
    height, width = int(size_hw[0]), int(size_hw[1])
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )
    if ok is None:
        ok = np.ones(points.shape[0], dtype=bool)
    if ok.size != points.shape[0]:
        raise ValueError(f"valid 长度 {ok.size} 与点数 {points.shape[0]} 不一致")
    clamped = np.stack(
        [np.clip(points[:, 0], 0, width - 1), np.clip(points[:, 1], 0, height - 1)], axis=-1
    )
    return {
        "xy": points,
        "clamped": clamped,
        "inside": inside,
        "valid": ok & inside,
        "occluded": (~ok) & inside,
        "outside": ~inside,
    }


def overlay_scalar(
    base: np.ndarray,
    values: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str = "inferno",
    alpha: float = 0.55,
    mask_floor: float = 0.01,
) -> np.ndarray:
    """固定 ``[vmin, vmax]`` 的标量图叠到 uint8 底图上（值域外只做裁剪，不重新拉伸）。

    ``mask_floor`` 是固定阈值（色标宽度的 1%）：低于它的像素不上色，底图保持原样。
    这样 colormap 的近零颜色不会把整帧染成紫色，也不需要任何逐帧归一化。
    """
    import matplotlib

    base_rgb = as_hwc(base)
    if base_rgb.shape[-1] == 1:
        base_rgb = np.repeat(base_rgb, 3, axis=-1)
    height, width = base_rgb.shape[:2]
    if vmax <= vmin:
        raise ValueError("vmax 必须大于 vmin")
    scalar = np.asarray(values, dtype=np.float32)
    if scalar.shape[:2] != (height, width):
        scalar = resize_map(scalar, (height, width))
    normalized = np.clip((scalar - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    color = matplotlib.colormaps[cmap](normalized)[..., :3] * 255.0
    mask = normalized > float(mask_floor)
    blend = base_rgb[..., :3].astype(np.float32).copy()
    blend[mask] = (1.0 - float(alpha)) * blend[mask] + float(alpha) * color[mask]
    return np.clip(blend, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 一帧的图
# --------------------------------------------------------------------------------------


@dataclass
class FrameVisual:
    """一帧的全部可视化输入（来自真实帧与真实 ``StepOutput.debug``）。"""

    platform: str
    camera: str
    clip_id: str
    frame_index: int
    raw_hwc: np.ndarray
    rgb_hwc: np.ndarray | None = None
    rgb_source: str | None = None
    rgb_file: str | None = None
    flow_px: np.ndarray | None = None
    flow_valid: bool = False
    quality_xy: np.ndarray | None = None
    quality_scalar: np.ndarray | None = None
    mode_revision: np.ndarray | None = None
    support_xy: np.ndarray | None = None
    support_valid: np.ndarray | None = None
    support_ids: np.ndarray | None = None
    support_age: np.ndarray | None = None
    trajectory: np.ndarray | None = None
    attention_map: np.ndarray | None = None
    attention_dense_mass: float | None = None
    tag: str | None = None
    notes: list[str] = field(default_factory=list)


_CJK_KEYWORDS = (
    "notosanscjk",
    "notoserifcjk",
    "notosanssc",
    "wenquanyimicrohei",
    "wenquanyizenhei",
    "wqy",
    "uming",
    "ukai",
    "droidsansfallback",
    "arplu",
)


@lru_cache(maxsize=1)
def _cjk_font_family() -> str | None:
    """找一个可用的中文字体族名；远端常常没有，没有就退回英文标签，不画豆腐块。"""
    try:
        from matplotlib import font_manager
    except Exception:  # pragma: no cover - matplotlib 缺失时按无中文字体处理
        return None
    roots = ("/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts"))
    paths: list[str] = []
    for root in roots:
        base = Path(root)
        if base.is_dir():
            paths.extend(str(path) for path in base.rglob("*.[to]tf"))
            paths.extend(str(path) for path in base.rglob("*.ttc"))
    paths.extend(font_manager.findSystemFonts(fontext="ttf"))
    for path in paths:
        name = Path(path).name.lower().replace("-", "").replace("_", "").replace(" ", "")
        if any(keyword in name for keyword in _CJK_KEYWORDS):
            try:
                return str(font_manager.FontProperties(fname=path).get_name())
            except Exception:  # pragma: no cover - 单个字体损坏时继续找
                continue
    return None


def _labels(zh: bool) -> dict[str, str]:
    if zh:
        return {
            "raw": "① RAW 固定显示映射（{camera}，非 RGB）",
            "rgb": "② 同源 RGB（{camera}）",
            "rgb_missing": "② 同源 RGB（{camera}）：未提供",
            "flow": "③ flow 对应（上一帧→当前帧；箭头 1:1 像素）",
            "quality": "④ 当前质量 R（模型估计；log10 均值方差，非毫米）",
            "mode": "⑤ 两模式权重：修订模式后验（模型估计）",
            "support": "⑥ 支持轨迹（同 ID 连线；越界／遮挡标注）",
            "attention": "⑦ readout attention（模型读取分布，非像素真值）",
            "info": "帧索引与图例",
            "inside": "帧内有效",
            "occluded": "遮挡／失效",
            "outside": "越界",
            "uniform": "1.0 = 全部媒体 token 上均匀",
            "no_state": "该臂不携带支持状态",
            "quality_footer": "log10(mean_j R_j)，固定色标（模型估计；R 不是毫米）",
            "quality_missing": "质量 R 未提供",
            "mode_footer": "P(修订)，固定 [0,1]（模型估计；不是外部运动概率）",
            "support_missing": "无支持点",
            "attention_missing": "未收集 attention",
            "attention_footer": "{uniform}；稠密总质量 = {mass}（模型读取分布，非像素真值）",
            "flow_missing": "flow 未提供",
            "raw_note": "三通道按 R,G,B 顺序直接映射；偏色是显示映射结果，不是传感器标定",
            "flow_first": "（首帧无上一帧，flow=0）",
        }
    return {
        "raw": "1 RAW fixed display ({camera}, not RGB)",
        "rgb": "2 same-source RGB ({camera})",
        "rgb_missing": "2 same-source RGB ({camera}): NOT PROVIDED",
        "flow": "3 flow prev->current (arrows are 1:1 pixel)",
        "quality": "4 quality R (model estimate; log10 mean var, not mm)",
        "mode": "5 mode weights: revision posterior (model estimate)",
        "support": "6 support tracks (same id connected; outside/occluded)",
        "attention": "7 readout attention (model reading, not pixel truth)",
        "info": "frame index and legend",
        "inside": "inside & valid",
        "occluded": "occluded/invalid",
        "outside": "outside frame",
        "uniform": "1.0 = uniform over all media tokens",
        "no_state": "this arm carries no support state",
        "quality_footer": "log10(mean_j R_j), fixed color scale (model estimate; R is not mm)",
        "quality_missing": "quality R not available",
        "mode_footer": "P(revision), fixed [0,1] (model estimate; not a motion probability)",
        "support_missing": "no support points",
        "attention_missing": "attention not collected",
        "attention_footer": "{uniform}; dense mass = {mass} (reading, not truth)",
        "flow_missing": "flow not available",
        "raw_note": (
            "channels mapped in R,G,B order; any color cast is the fixed display map, "
            "not sensor calibration"
        ),
        "flow_first": " (first frame: no previous frame, flow=0)",
    }


def _extent(size_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = int(size_hw[0]), int(size_hw[1])
    return (-0.5, width - 0.5, height - 0.5, -0.5)


def _image_axis(ax: Any, size_hw: tuple[int, int], title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, int(size_hw[1]) - 0.5)
    ax.set_ylim(int(size_hw[0]) - 0.5, -0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (px)", fontsize=8)
    ax.set_ylabel("y (px)", fontsize=8)


def points_hw2(xy: Any) -> np.ndarray:
    """支撑点坐标 -> ``[N,2]``（debug 里的张量是 ``[1,N,2]``，这里统一摊平）。"""
    return np.asarray(xy, dtype=np.float32).reshape(-1, 2)


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    rgb = as_hwc(image)
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    return np.repeat(rgb[..., :3].mean(axis=-1, keepdims=True).astype(np.uint8), 3, axis=-1)


#: 信息面板每行允许的显示列数（monospace 7.6pt，面板宽度约 4.4in）。
INFO_LINE_WIDTH = 62


def display_columns(text: str) -> int:
    """等宽字体下的显示宽度：CJK 等全角字符按 2 列计。"""
    return sum(2 if ord(char) > 0x2E80 else 1 for char in text)


def short_relpath(relpath: str, limit: int = 28) -> str:
    """长相对路径裁成 ``…/episode/variant.npz`` 形式（完整路径仍在 index.json 里）。"""
    if len(relpath) <= limit:
        return relpath
    tail = "/".join(relpath.split("/")[-2:])
    if len(tail) <= limit - 1:
        return "…/" + tail
    return "…" + tail[-(limit - 1) :]


def apply_thread_limit(threads: int) -> int:
    """把渲染的 CPU 线程压到 ``threads``（训练在同机时避免抢 CPU）；``<=0`` 表示不改。

    返回实际生效的线程数。只影响本进程的 torch 算子并行度，不改数值结果。
    """
    if int(threads) <= 0:
        return 0
    import torch

    torch.set_num_threads(int(threads))
    return int(torch.get_num_threads())


def info_lines(visual: FrameVisual, size_hw: tuple[int, int], *, zh: bool) -> list[str]:
    """信息面板的固定文字行（与输入帧同一帧索引；每行不超过 :data:`INFO_LINE_WIDTH` 列）。"""
    spec = RAW_DISPLAY[visual.platform]
    return [
        f"clip         : {visual.clip_id}",
        f"platform     : {visual.platform}",
        f"camera       : {visual.camera}",
        f"frame index  : {visual.frame_index}  (record step; = input frame)",
        f"frame size   : {int(size_hw[1])}x{int(size_hw[0])} px",
        "",
        f"RAW display  : fixed log, G={spec.log_gain:g}, white={spec.white_level:g}",
        f"RAW unit     : {spec.unit if zh else spec.unit_en}",
        "RGB          : " + ("same source" if visual.rgb_hwc is not None else "NOT PROVIDED"),
        f"RGB file     : {visual.rgb_file or '-'}",
        f"RGB member   : {visual.rgb_source or '-'}",
        "",
        f"quality R    : log10(mean R), fixed {QUALITY_LOG10_RANGE}",
        "mode weights : P(revision), fixed [0, 1]",
        f"attention    : a*M, fixed [0, {ATTENTION_VMAX:g}] (1.0 = uniform)",
        "",
        "coordinates  : raw pixels x in [0,W-1], y in [0,H-1];",
        "               every panel is the same input frame",
        "legend       : overlays are MODEL ESTIMATES;",
        "               attention is a model reading, not pixel truth;",
        "               R is not mm; mode weights are not motion.",
    ] + list(visual.notes)


def compose_frame(
    visual: FrameVisual,
    *,
    dpi: int = 110,
    figsize: tuple[float, float] = FIGURE_SIZE_IN,
    cjk: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """把一帧画成 2x4 图组；返回 ``(figure, panels_meta)``，不写文件。"""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    family = _cjk_font_family()
    zh = (family is not None) if cjk is None else bool(cjk)
    if zh and family:
        matplotlib.rcParams["font.sans-serif"] = [family, *matplotlib.rcParams["font.sans-serif"]]
        matplotlib.rcParams["axes.unicode_minus"] = False
    text = _labels(zh)
    raw_image = raw_display_rgb(visual.raw_hwc, visual.platform)
    size_hw = (int(raw_image.shape[0]), int(raw_image.shape[1]))
    base_gray = _gray_uint8(raw_image)

    figure, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi, facecolor="#0f1216")
    axes = np.asarray(axes).reshape(2, 4)
    for row in axes:
        for ax in row:
            ax.set_facecolor("#0f1216")

    def footer(ax: Any, content: str) -> None:
        ax.text(0.01, -0.16, content, transform=ax.transAxes, fontsize=7, color="#9fb3c8")

    meta: dict[str, Any] = {
        "platform": visual.platform,
        "camera": visual.camera,
        "clip_id": visual.clip_id,
        "frame_index": int(visual.frame_index),
        "size_hw": list(size_hw),
        "cjk_font": family,
        "tag": visual.tag,
        "same_coordinates": "原始像素 x∈[0,W-1], y∈[0,H-1]；所有面板与输入帧同帧同坐标",
    }

    # ① RAW 固定显示
    ax = axes[0, 0]
    ax.imshow(raw_image, extent=_extent(size_hw), interpolation="nearest")
    _image_axis(ax, size_hw, text["raw"].format(camera=visual.camera))
    footer(ax, RAW_DISPLAY[visual.platform].formula())
    spec = RAW_DISPLAY[visual.platform]
    meta["raw"] = {
        "available": True,
        "mapping": spec.formula(),
        "white_level": spec.white_level,
        "log_gain": spec.log_gain,
        "unit": spec.unit,
    }

    # ② 同源 RGB（缺失即“未提供”）
    ax = axes[0, 1]
    rgb_image, rgb_label = rgb_panel(visual.rgb_hwc, visual.camera)
    if rgb_image is None:
        ax.axis("off")
        ax.set_title(text["rgb_missing"].format(camera=visual.camera), fontsize=11, color="#ffb859")
        ax.text(
            0.5,
            0.5,
            rgb_label,
            ha="center",
            va="center",
            fontsize=13,
            color="#ffb859",
            transform=ax.transAxes,
        )
    else:
        ax.imshow(rgb_image, extent=_extent(rgb_image.shape[:2]), interpolation="nearest")
        _image_axis(ax, rgb_image.shape[:2], text["rgb"].format(camera=visual.camera))
        footer(ax, f"{text['rgb'].format(camera=visual.camera)} · {visual.rgb_source or 'rgb/'}")
    meta["rgb"] = {
        "available": rgb_image is not None,
        "label": rgb_label,
        "source": visual.rgb_source,
    }

    # ③ flow：HSV + 箭头
    ax = axes[0, 2]
    if visual.flow_px is None:
        ax.axis("off")
        ax.set_title(text["flow"], fontsize=11, color="#ffb859")
        ax.text(
            0.5,
            0.5,
            text["flow_missing"],
            ha="center",
            va="center",
            fontsize=13,
            color="#ffb859",
            transform=ax.transAxes,
        )
        meta["flow"] = {"available": False}
    else:
        flow = _as_hw2(visual.flow_px)
        ax.imshow(flow_hsv(flow), extent=_extent(size_hw), interpolation="nearest")
        _image_axis(ax, size_hw, text["flow"])
        grid_x, grid_y, u, v = flow_arrows(flow)
        ax.quiver(
            grid_x,
            grid_y,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            headwidth=3.5,
            headlength=4.0,
            color="#f2f6fa",
            alpha=0.9,
        )
        magnitude = np.hypot(flow[..., 0], flow[..., 1])
        footer(
            ax,
            f"median |flow| = {float(np.median(magnitude)):.2f} px, p95 = "
            f"{float(np.percentile(magnitude, 95)):.2f} px"
            + ("" if visual.flow_valid else text["flow_first"]),
        )
        meta["flow"] = {
            "available": True,
            "valid": bool(visual.flow_valid),
            "median_magnitude_px": float(np.median(magnitude)),
            "p95_magnitude_px": float(np.percentile(magnitude, 95)),
            "arrow_stride": FLOW_ARROW_STRIDE,
            "encoding": "hue=atan2(v,u)/2pi, value=r/(r+8)",
        }

    # ④ 质量 R（固定 log10 范围）
    ax = axes[0, 3]
    ax.imshow(base_gray, extent=_extent(size_hw), interpolation="nearest")
    _image_axis(ax, size_hw, text["quality"])
    if visual.quality_xy is not None and visual.quality_scalar is not None:
        scatter = ax.scatter(
            points_hw2(visual.quality_xy)[:, 0],
            points_hw2(visual.quality_xy)[:, 1],
            c=np.asarray(visual.quality_scalar),
            cmap="inferno",
            vmin=QUALITY_LOG10_RANGE[0],
            vmax=QUALITY_LOG10_RANGE[1],
            s=22,
            marker="o",
            linewidths=0,
        )
        figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        values = np.asarray(visual.quality_scalar)
        footer(ax, text["quality_footer"])
        meta["quality"] = {
            "available": True,
            "n": int(values.size),
            "log10_range": list(QUALITY_LOG10_RANGE),
            "observed_log10_min": float(np.min(values)),
            "observed_log10_max": float(np.max(values)),
            "clipped": bool(
                np.any(values < QUALITY_LOG10_RANGE[0]) or np.any(values > QUALITY_LOG10_RANGE[1])
            ),
            "note": "log10(mean_j R_j)，R 是潜在测量方差（模型估计，不是毫米）",
        }
    else:
        footer(ax, text["quality_missing"])
        meta["quality"] = {"available": False}

    # ⑤ 模式权重（固定 [0,1]）
    ax = axes[1, 0]
    ax.imshow(base_gray, extent=_extent(size_hw), interpolation="nearest")
    _image_axis(ax, size_hw, text["mode"])
    if visual.mode_revision is not None and visual.quality_xy is not None:
        weights = np.asarray(visual.mode_revision)
        scatter = ax.scatter(
            points_hw2(visual.quality_xy)[:, 0],
            points_hw2(visual.quality_xy)[:, 1],
            c=weights,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=22,
            marker="o",
            linewidths=0,
        )
        figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        footer(ax, text["mode_footer"])
        meta["mode"] = {
            "available": True,
            "n": int(weights.size),
            "range": [0.0, 1.0],
            "mean_revision": float(np.mean(weights)),
            "frac_above_half": float(np.mean(weights > 0.5)),
            "note": "修订模式后验（模型估计；不是外部运动概率）",
        }
    else:
        footer(ax, text["no_state"])
        meta["mode"] = {"available": False}

    # ⑥ 支持轨迹
    ax = axes[1, 1]
    ax.imshow(base_gray, extent=_extent(size_hw), interpolation="nearest")
    _image_axis(ax, size_hw, text["support"])
    if visual.support_xy is not None:
        groups = support_scatter(visual.support_xy, size_hw, valid=visual.support_valid)
        if visual.trajectory is not None:
            track = np.asarray(visual.trajectory, dtype=np.float32)
            if track.ndim == 3:
                for index in range(0, track.shape[1], 8):
                    ax.plot(
                        track[:, index, 0],
                        track[:, index, 1],
                        color="#7fd1ff",
                        linewidth=0.7,
                        alpha=0.55,
                    )
                ids = (
                    None
                    if visual.support_ids is None
                    else np.asarray(visual.support_ids).reshape(-1)
                )
                if ids is not None and ids.size == track.shape[1]:
                    for index in range(0, track.shape[1], 32):
                        ax.text(
                            track[0, index, 0],
                            track[0, index, 1],
                            str(int(ids[index])),
                            fontsize=6,
                            color="#ffd166",
                        )
        if visual.support_age is not None:
            age = np.asarray(visual.support_age).reshape(-1)
            if age.size == groups["xy"].shape[0]:
                current = ax.scatter(
                    groups["clamped"][groups["inside"], 0],
                    groups["clamped"][groups["inside"], 1],
                    c=age[groups["inside"]],
                    cmap="viridis",
                    s=14,
                    marker="o",
                    linewidths=0,
                )
                figure.colorbar(current, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(
                    labelsize=7
                )
            else:
                ax.scatter(
                    groups["clamped"][groups["inside"], 0],
                    groups["clamped"][groups["inside"], 1],
                    s=14,
                    c="#7fd1ff",
                )
        else:
            ax.scatter(
                groups["clamped"][groups["inside"], 0],
                groups["clamped"][groups["inside"], 1],
                s=14,
                c="#7fd1ff",
            )
        if groups["occluded"].any():
            ax.scatter(
                groups["clamped"][groups["occluded"], 0],
                groups["clamped"][groups["occluded"], 1],
                s=34,
                facecolors="none",
                edgecolors="#d9dee3",
                linewidths=0.8,
            )
        if groups["outside"].any():
            ax.scatter(
                groups["clamped"][groups["outside"], 0],
                groups["clamped"][groups["outside"], 1],
                s=40,
                marker="x",
                c="#ff4d4d",
                linewidths=1.1,
            )
        counts = {
            "n": int(groups["xy"].shape[0]),
            "inside_valid": int(groups["valid"].sum()),
            "occluded": int(groups["occluded"].sum()),
            "outside": int(groups["outside"].sum()),
        }
        suffix = "（越界画红 x）" if zh else " (red x)"
        footer(
            ax,
            f"{text['inside']} {counts['inside_valid']} · {text['occluded']} "
            f"{counts['occluded']} · {text['outside']} {counts['outside']}{suffix}",
        )
        ids = None if visual.support_ids is None else np.asarray(visual.support_ids).reshape(-1)
        ages = None if visual.support_age is None else np.asarray(visual.support_age).reshape(-1)
        meta["support"] = {
            "available": True,
            **counts,
            "id_min": None if ids is None else int(np.min(ids)),
            "id_max": None if ids is None else int(np.max(ids)),
            "age_median": None if ages is None or not ages.size else float(np.median(ages)),
            "age_max": None if ages is None or not ages.size else float(np.max(ages)),
            # 本帧新生支持数：出生帧 age 记 0（state_estimator 的 age 语义）
            "births": None if ages is None else int(np.sum(ages == 0)),
        }
    else:
        footer(ax, text["support_missing"])
        meta["support"] = {"available": False}

    # ⑦ readout attention
    ax = axes[1, 2]
    ax.imshow(base_gray, extent=_extent(size_hw), interpolation="nearest")
    _image_axis(ax, size_hw, text["attention"])
    if visual.attention_map is None:
        ax.text(
            0.5,
            0.5,
            text["attention_missing"],
            ha="center",
            va="center",
            fontsize=12,
            color="#ffb859",
            transform=ax.transAxes,
        )
        meta["attention"] = {"available": False}
    else:
        attention = resize_map(np.asarray(visual.attention_map, dtype=np.float32), size_hw)
        blended = overlay_scalar(
            base_gray, attention, vmin=0.0, vmax=ATTENTION_VMAX, cmap="magma", alpha=0.6
        )
        ax.imshow(blended, extent=_extent(size_hw), interpolation="nearest")
        scalar = figure.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=0.0, vmax=ATTENTION_VMAX), cmap="magma"
            ),
            ax=ax,
            fraction=0.046,
            pad=0.02,
        )
        scalar.ax.tick_params(labelsize=7)
        mass = (
            "nan" if visual.attention_dense_mass is None else f"{visual.attention_dense_mass:.3f}"
        )
        footer(ax, text["attention_footer"].format(uniform=text["uniform"], mass=mass))
        meta["attention"] = {
            "available": True,
            "range": [0.0, ATTENTION_VMAX],
            "relative_to_uniform_dense": True,
            "dense_mass": visual.attention_dense_mass,
            "note": (
                "查询平均的末块注意力；前 h*w 列 = 当前帧稠密网格，"
                "dense_mass 是每 query 落在该网格上的总质量（模型读取分布，非像素真值）"
            ),
        }

    # ⑧ 文字面板（固定英文，避免远端无中文字体时出现豆腐块）
    ax = axes[1, 3]
    ax.axis("off")
    ax.set_title(text["info"], fontsize=11)
    lines = info_lines(visual, size_hw, zh=zh)
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        color="#dbe6f0",
        family="monospace",
    )

    figure.subplots_adjust(
        left=0.03, right=0.985, top=0.895, bottom=0.085, wspace=0.24, hspace=0.34
    )
    return figure, meta


def write_png(figure: Any, path: str | Path, *, dpi: int = 110) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
    return path


def expected_png_size(
    dpi: int, *, figsize: tuple[float, float] = FIGURE_SIZE_IN
) -> tuple[int, int]:
    """``write_png`` 写出的 PNG 像素尺寸（测试断言用）。"""
    return int(round(figsize[0] * dpi)), int(round(figsize[1] * dpi))


# --------------------------------------------------------------------------------------
# 多步统计汇总（给 lead／报告用的 100/500/1000 对比表）
# --------------------------------------------------------------------------------------


def _range(values: Sequence[float]) -> list[float] | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return [min(numbers), max(numbers)]


def artifact_tag(dir_name: str) -> str | None:
    """从输出目录名取 ``--tag`` 后缀（``<platform>-<arm>-seed<seed>[-<tag>]``）。"""
    match = re.match(r".+-seed\d+(?:-(?P<tag>.+))?$", str(dir_name))
    return None if match is None or not match.group("tag") else match.group("tag")


def summarize_frames(output_root: str | Path) -> dict[str, Any]:
    """汇总 artifacts 下的帧 meta，**按输出目录分组**（同 step 的 pre-fix/post-fix 各自成列）。

    每条统计量给出跨帧的 ``[min, max]``；``births_per_frame`` 给每帧均值；flow 只用有上一帧的
    帧；缺失字段跳过，不补零。
    """
    root = Path(output_root)
    buckets: dict[str, dict[str, list[Any]]] = {}
    info: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("**/clip_*/frame_*.json")):
        try:
            meta = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        group = path.parent.parent.name  # <platform>-<arm>-seed<seed>[-tag]
        bucket = buckets.setdefault(group, {})
        entry = info.setdefault(group, {"frames": 0, "clips": set(), "steps": set()})
        entry["frames"] += 1
        entry["clips"].add(str(meta.get("clip_id")))
        entry["steps"].add(meta.get("checkpoint_step"))
        if meta.get("checkpoint_path"):
            entry["checkpoint_path"] = meta["checkpoint_path"]
        flow = meta.get("flow") or {}
        if flow.get("valid"):
            bucket.setdefault("flow_median_px", []).append(flow.get("median_magnitude_px"))
            bucket.setdefault("flow_p95_px", []).append(flow.get("p95_magnitude_px"))
        quality = meta.get("quality") or {}
        bucket.setdefault("log10R_min", []).append(quality.get("observed_log10_min"))
        bucket.setdefault("log10R_max", []).append(quality.get("observed_log10_max"))
        mode = meta.get("mode") or {}
        bucket.setdefault("mean_P_revision", []).append(mode.get("mean_revision"))
        bucket.setdefault("frac_P_revision_above_half", []).append(mode.get("frac_above_half"))
        attention = meta.get("attention") or {}
        bucket.setdefault("attention_dense_mass", []).append(attention.get("dense_mass"))
        support = meta.get("support") or {}
        for key, name in (
            ("inside_valid", "support_valid"),
            ("occluded", "support_occluded"),
            ("outside", "support_outside"),
            ("id_max", "support_id_max"),
            ("age_median", "median_age"),
            ("births", "births_per_frame"),
        ):
            bucket.setdefault(name, []).append(support.get(key))

    groups: list[dict[str, Any]] = []
    for group in sorted(buckets, key=lambda name: (min(info[name]["steps"] or [-1]), name)):
        entry = info[group]
        record: dict[str, Any] = {
            "dir": group,
            "step": max((value for value in entry["steps"] if value is not None), default=None),
            "steps": sorted(value for value in entry["steps"] if value is not None),
            "tag": artifact_tag(group),
            "frames": entry["frames"],
            "clips": sorted(entry["clips"]),
            "checkpoint_path": entry.get("checkpoint_path"),
        }
        for key, values in buckets[group].items():
            record[key] = _range(values)
        record["frames_with_flow"] = len(buckets[group].get("flow_median_px", []))
        birth_values = [
            float(value)
            for value in buckets[group].get("births_per_frame", [])
            if value is not None
        ]
        record["births_per_frame_mean"] = (
            sum(birth_values) / len(birth_values) if birth_values else None
        )
        groups.append(record)
    return {
        "schema": "n04-v1.1-s2-visuals-summary-v2",
        "output_root": str(root),
        "steps": [record["step"] for record in groups if record["step"] is not None],
        "groups": groups,
    }


def _representative(value: Any) -> float | None:
    """统计量的代表值：``[min,max]`` 取中点（只用于相邻 step 的方向比较，不改变原始区间）。"""
    if value is None:
        return None
    if isinstance(value, list) and value:
        return (float(value[0]) + float(value[-1])) / 2.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: 相对变化小于该比例记作持平（方向判断只用这一个阈值，不改数值口径）。
DIRECTION_EPS = 0.10


def format_direction(summary: Mapping[str, Any]) -> str:
    """相邻**列**的**事实性**方向（↑/↓/≈），不代替好坏判断。

    每个统计量取 ``[min,max]`` 中点做比较，相对变化 < :data:`DIRECTION_EPS` 记 ≈。
    """
    columns = list(summary.get("groups", []))
    steps = [f"{record['dir']}" for record in columns]
    if len(steps) < 2:
        return "（只有一列，无法比较方向；后续 step 出来后再看）"
    rows = [
        ("flow median px", "flow_median_px"),
        ("flow p95 px", "flow_p95_px"),
        ("log10(R) 区间宽度", "_log10R_width"),
        ("attention dense_mass", "attention_dense_mass"),
        ("mean P(revision)", "mean_P_revision"),
        ("support valid", "support_valid"),
        ("support occluded", "support_occluded"),
        ("support id_max", "support_id_max"),
        ("median_age", "median_age"),
        ("births_per_frame", "births_per_frame_mean"),
    ]
    lines = ["", "相邻列变化（↑增大 / ↓减小 / ≈相对变化<10%，取区间中点比较）："]
    for title, key in rows:
        cells: list[str] = []
        for left, right in zip(steps, steps[1:], strict=False):

            def value_of(step: str, stat_key: str = key) -> float | None:
                entry = next((record for record in columns if record["dir"] == step), {})
                if stat_key == "_log10R_width":
                    low = _representative(entry.get("log10R_min"))
                    high = _representative(entry.get("log10R_max"))
                    return None if low is None or high is None else high - low
                return _representative(entry.get(stat_key))

            before, after = value_of(left), value_of(right)
            if before is None or after is None:
                cells.append(f"{left}→{right}: —")
                continue
            scale = max(abs(before), abs(after), 1e-12)
            delta = (after - before) / scale
            mark = "≈" if abs(delta) < DIRECTION_EPS else ("↑" if delta > 0 else "↓")
            cells.append(f"{left}→{right} {mark} ({before:.4g}→{after:.4g})")
        lines.append(f"  {title}: " + "; ".join(cells))
    return "\n".join(lines)


def format_summary_table(summary: Mapping[str, Any]) -> str:
    """把 :func:`summarize_frames` 的结果排成 step 并排的表格（报告直接粘贴）。

    列头写明每个 step 的帧数与来源（该 step 自己的帧 meta，``--tag`` 分目录互不覆盖）；
    下方给出 ``id_max`` 口径说明与相邻 step 的**事实性**方向，不代替好坏结论。
    """
    columns = list(summary.get("groups", []))
    steps = [record["dir"] for record in columns]
    if not steps:
        return "（没有可汇总的帧 meta）"
    rows = [
        ("frames（帧数）", "frames"),
        ("clips（片段数）", "clips"),
        ("flow median px", "flow_median_px"),
        ("flow p95 px", "flow_p95_px"),
        ("log10(R) low", "log10R_min"),
        ("log10(R) high", "log10R_max"),
        ("attention dense_mass", "attention_dense_mass"),
        ("mean P(revision)", "mean_P_revision"),
        ("max frac P(rev)>0.5", "frac_P_revision_above_half"),
        ("support valid", "support_valid"),
        ("support occluded", "support_occluded"),
        ("support outside", "support_outside"),
        ("support id_max", "support_id_max"),
        ("median_age", "median_age"),
        ("births_per_frame（均值）", "births_per_frame_mean"),
    ]

    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, list):
            return " / ".join(f"{float(item):.4g}" for item in value)
        return str(value)

    def column(dir_name: str) -> dict[str, Any]:
        return next((record for record in columns if record["dir"] == dir_name), {})

    headers = []
    for dir_name in steps:
        entry = column(dir_name)
        step = entry.get("step")
        ckpt = entry.get("checkpoint_path") or (
            "--" if step is None else f"checkpoint-step{int(step):07d}.pt"
        )
        headers.append(
            f"{dir_name}（step {step}，{entry.get('frames', 0)} 帧，ckpt={Path(ckpt).name}）"
        )
    lines = ["| 统计量 | " + " | ".join(headers) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in steps) + " |")
    for title, key in rows:
        values = []
        for step in steps:
            entry = column(step)
            value = entry.get(key)
            if key == "clips":
                value = len(entry.get("clips", []))
            values.append(cell(value))
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    flow_counts = ", ".join(f"{step}: {column(step).get('frames_with_flow', 0)}" for step in steps)
    lines.append("")
    lines.append(
        "口径：一列 = 一个 artifacts 输出目录"
        "（`--tag` 分目录，同 step 的 pre-fix/post-fix 也分开）；"
        f"flow 只用有上一帧的帧（{flow_counts}）。"
    )
    lines.append(
        "id_max / median_age / births_per_frame：本版起才写入帧 meta，老帧没有这些字段所以显示 —；"
        "补测值只在报告里注明。births = 该帧 age==0 的支持数（出生帧语义）。"
    )
    lines.append(
        "方向只表示相邻 step 数值增大/减小/持平，不代表方法变好；单 checkpoint 观察不构成结论。"
    )
    lines.append(format_direction(summary))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 片段选择与同源 RGB
# --------------------------------------------------------------------------------------


def select_clips(
    entries: Sequence[Any],
    *,
    frames: int,
    clips: int,
    seed: int,
    frame_count: Any,
) -> list[tuple[Any, int]]:
    """固定挑选：按 relpath 排序的 dev 条目 + 固定种子选出 ``clips`` 个连续片段。

    规则（可复现，不扫描整盘）：只保留 ``frame_count >= frames`` 的条目，用
    ``SeedSequence([seed, 1101])`` 无放回抽条目，再用同一 rng 抽起点；结果按条目序排列。
    """
    if clips < 1 or frames < 1:
        raise ValueError("clips 与 frames 都必须 >= 1")
    pool = [
        entry
        for entry in sorted(entries, key=lambda item: item.relpath)
        if int(frame_count(entry)) >= int(frames)
    ]
    if not pool:
        raise ValueError(f"dev 里没有 >= {frames} 帧的条目，无法出图")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1101]))
    picked = sorted(
        int(index)
        for index in rng.choice(len(pool), size=min(int(clips), len(pool)), replace=False)
    )
    chosen: list[tuple[Any, int]] = []
    for index in picked:
        entry = pool[index]
        total = int(frame_count(entry))
        start = int(rng.integers(0, total - int(frames) + 1))
        chosen.append((entry, start))
    return chosen


def parse_clip_override(value: str) -> tuple[str, int]:
    """``--clip relpath:start``（也接受 ``relpath:start:frames`` 里的前两段）。"""
    parts = value.rsplit(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "--clip 需要 relpath:start，例如 suite/episode/variant_00.npz:12"
        )
    try:
        return parts[0], int(parts[1])
    except ValueError as error:  # pragma: no cover - argparse 已挡住大多数写法
        raise argparse.ArgumentTypeError(f"--clip 的 start 不是整数：{value!r}") from error


def rgb_root_for(data_root: str | Path, platform: str, override: str | None = None) -> Path | None:
    """同源 RGB 根目录：默认是 RAW 目录的兄弟 ``rgb/``（``libero/raw`` -> ``libero/rgb``）。"""
    if override:
        return Path(override)
    from ..raw.n04_v11_data import PLATFORM_SPECS

    raw_dir = Path(PLATFORM_SPECS[platform].raw_dir)
    if raw_dir.name != "raw":
        return None
    return Path(data_root) / raw_dir.parent / "rgb"


def rgb_member_candidates(raw_key: str) -> list[str]:
    """RAW 成员名 -> 同源 RGB 成员名候选（实测 ``*_raw_uint8`` → ``*_rgb_uint8``）。"""
    base = raw_key.split("_raw_")[0]
    candidates = [f"{base}_rgb_uint8", raw_key.replace("_raw_", "_rgb_"), f"{base}_rgb_float32"]
    unique: list[str] = []
    for name in candidates:
        if name not in unique:
            unique.append(name)
    return unique


def load_rgb_clip(
    rgb_root: Path | None,
    entry: Any,
    *,
    camera_keys: Sequence[str],
    start: int,
    frames: int,
) -> dict[str, Any]:
    """读一个片段的同源 RGB；文件／成员缺失时如实返回 ``None``，不造数据。"""
    report: dict[str, Any] = {
        "root": None if rgb_root is None else str(rgb_root),
        "members": {},
        "images": {},
    }
    if rgb_root is None:
        return report
    path = Path(rgb_root) / entry.relpath
    if not path.exists():
        report["missing_file"] = str(path)
        return report
    report["path"] = str(path)
    with np.load(path) as payload:
        names = set(payload.files)
        for key in camera_keys:
            for candidate in rgb_member_candidates(key):
                if candidate not in names:
                    continue
                member = payload[candidate]
                if member.ndim != 4:
                    report.setdefault("skipped_members", []).append(
                        f"{candidate} 形状 {tuple(member.shape)} 不是 [T,H,W,C]"
                    )
                    continue
                end = min(int(member.shape[0]), int(start) + int(frames))
                report["members"][key] = candidate
                report["images"][key] = np.asarray(member[int(start) : end])
                break
    return report


# --------------------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------------------


def _resolve_split_file(split_file: str | Path, config_path: str | Path) -> Path:
    """``split_file`` 常写成仓库相对路径：依次试 CWD、配置文件所在目录及其祖先。"""
    candidate = Path(split_file)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    config_dir = Path(config_path).resolve().parent
    for root in (Path.cwd(), config_dir, *config_dir.parents):
        probe = root / candidate
        if probe.exists():
            return probe
    raise FileNotFoundError(f"找不到拆分表 {split_file}（CWD／配置目录祖先都试过）")


def _numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return np.asarray(array)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "N04 v1.1 S2 固定 dev 片段可视化：RAW／RGB／flow／质量／模式／支持轨迹／attention"
        ),
    )
    parser.add_argument("--config", required=True, help="run 配置（JSON，支持 include）")
    parser.add_argument("--data-root", default=None, help="官方数据根；缺省用配置 data.root")
    parser.add_argument("--output", required=True, help="输出目录，通常是 <run>/artifacts")
    parser.add_argument(
        "--checkpoint", default=None, help="训练 checkpoint（checkpoint-stepNNNNNNN.pt）"
    )
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0；默认 cpu，不占训练 GPU")
    parser.add_argument("--platform", default=None, choices=("libero", "robotwin"))
    parser.add_argument("--arm", default=None)
    parser.add_argument("--clips", type=int, default=3, help="固定片段数（3—5）")
    parser.add_argument("--frames", type=int, default=8, help="每个片段的连续帧数（>=8）")
    parser.add_argument("--split", default="dev", help="只用官方 dev 拆分")
    parser.add_argument("--seed", type=int, default=None, help="片段挑选种子；缺省用配置 seed")
    parser.add_argument("--camera", default=None, help="要画的主相机；缺省用平台第一个相机")
    parser.add_argument("--rgb-root", default=None, help="同源 RGB 根目录；缺省取 RAW 的兄弟 rgb/")
    parser.add_argument(
        "--clip",
        action="append",
        type=parse_clip_override,
        default=None,
        help="显式固定片段 relpath:start（可重复）",
    )
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--threads", type=int, default=0, help="CPU 线程上限（<=0 表示不改）；训练同机时建议 2—4"
    )
    parser.add_argument("--list-clips", action="store_true", help="只打印固定挑选结果，不加载模型")
    parser.add_argument(
        "--tag",
        default=None,
        help="输出目录后缀（例如 step0500）：不同 checkpoint 的图分开放，不覆盖上一批",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="不渲染：汇总 --output 下已有帧 meta，按 checkpoint step 生成 summary.json + 表格",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "summary_only", False):
        return summary_run(args)
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import torch

    from ..core.encoder_contract_n04 import EmbodimentContext
    from ..raw.n04_v11_data import (
        PLATFORM_SPECS,
        RawVLASequenceLoader,
        V11DataIndex,
        load_window_batch,
        saturation_mask,
    )
    from ..training.n04_v11 import build_v11_encoder, load_config

    thread_cap = apply_thread_limit(int(getattr(args, "threads", 0) or 0))
    config = load_config(args.config, data_root=args.data_root)
    platform = str(args.platform or config.get("platform") or "")
    if platform not in PLATFORM_SPECS:
        raise SystemExit(f"未知平台 {platform!r}；可选 {sorted(PLATFORM_SPECS)}")
    arm = str(args.arm or config.get("arm", "D"))
    seed = int(args.seed if args.seed is not None else config.get("seed", 101))
    split = str(args.split)
    spec = PLATFORM_SPECS[platform]
    camera = str(args.camera or spec.camera_ids[0])
    if camera not in spec.camera_ids:
        raise SystemExit(f"{platform} 没有相机 {camera!r}；可选 {list(spec.camera_ids)}")
    camera_index = list(spec.camera_ids).index(camera)

    data_root = Path(config["data"]["root"])
    split_file = _resolve_split_file(config["data"]["split_file"], args.config)
    index = V11DataIndex(
        root=data_root,
        split_file=split_file,
        platform_dirs={platform: spec.raw_dir},
        splits=(split,),
    )
    entries = index.entries(platform, split)
    if not entries:
        raise SystemExit(f"{platform} 的 {split} 拆分为空：检查 {data_root} 与 {split_file}")

    def frame_count(entry: Any) -> int:
        return int(index.sequence_frames(entry))

    if args.clip:
        table = {entry.relpath: entry for entry in entries}
        chosen: list[tuple[Any, int]] = []
        for relpath, start in args.clip:
            if relpath not in table:
                raise SystemExit(f"--clip 的 {relpath!r} 不在 {platform} 的 {split} 拆分里")
            chosen.append((table[relpath], int(start)))
    else:
        chosen = select_clips(
            entries,
            frames=int(args.frames),
            clips=int(args.clips),
            seed=seed,
            frame_count=frame_count,
        )

    plan = [
        {
            "clip_id": f"clip_{position + 1:02d}",
            "platform": platform,
            "relpath": entry.relpath,
            "source_id": entry.source_id,
            "task_key": entry.task_key,
            "start": int(start),
            "frames": int(args.frames),
            "total_frames": frame_count(entry),
        }
        for position, (entry, start) in enumerate(chosen)
    ]
    if args.list_clips:
        print(
            json.dumps(
                {"platform": platform, "split": split, "clips": plan}, indent=2, ensure_ascii=False
            )
        )
        return {"clips": plan, "generated": 0, "manifest": None}

    if not args.checkpoint:
        raise SystemExit(
            "需要 --checkpoint：真实 checkpoint + 官方 dev 帧才出图，没有就报“尚未生成”"
        )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"尚未生成：找不到 checkpoint {checkpoint}")

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "encoder" not in payload:
        raise SystemExit(f"{checkpoint} 不是本项目的训练 checkpoint（缺 encoder 段）")
    from ..frameworks.n04.compatibility import require_current_implementation

    require_current_implementation(payload)
    if payload.get("platform") not in (None, platform):
        raise SystemExit(
            f"checkpoint 平台是 {payload.get('platform')!r}，与 --platform {platform!r} 不一致"
        )
    if payload.get("arm") not in (None, arm):
        raise SystemExit(f"checkpoint 臂是 {payload.get('arm')!r}，与 --arm {arm!r} 不一致")
    device = torch.device(args.device)
    # 用 checkpoint 自己带的 model 段建网络：训练时的结构就是它，配置文件后来改了也不会
    # 造成 state_dict 对不上（两边不一致时明确提示，不静默取其一）。
    checkpoint_model = (payload.get("config") or {}).get("model")
    model_cfg = config["model"]
    if isinstance(checkpoint_model, dict) and checkpoint_model != model_cfg:
        print(
            "[visuals] note: checkpoint 的 model 段与当前配置文件不一致，"
            "按 checkpoint 建网络（可视化必须画训练时的那个结构）",
            file=sys.stderr,
        )
        model_cfg = checkpoint_model
    encoder = build_v11_encoder(model_cfg, platform=platform, arm=arm)
    missing, unexpected = encoder.load_state_dict(payload["encoder"], strict=False)
    if missing or unexpected:
        raise SystemExit(
            "checkpoint 与当前模型结构不一致："
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
        )
    encoder.eval().to(device)
    step = int(payload.get("step") or 0)
    named = re.search(r"checkpoint-step(\d+)\.pt$", checkpoint.name)
    if named and int(named.group(1)) != step:
        print(
            f"[visuals] warn: 文件名标 step {int(named.group(1))}，payload 里是 {step}；"
            "按 payload 记录（不拿旧 checkpoint 冒充新 step）",
            file=sys.stderr,
        )
    checkpoint_info = {
        "path": str(checkpoint),
        "step": step,
        "platform": payload.get("platform"),
        "arm": payload.get("arm"),
        "seed": payload.get("seed"),
        "dev_loss": payload.get("dev_loss"),
        "trained": step > 0,
        "config_fingerprint": payload.get("config_fingerprint"),
    }

    rgb_root = rgb_root_for(data_root, platform, args.rgb_root)
    loader = RawVLASequenceLoader(suite=platform)
    proprio_history = int(config["data"].get("proprio_history", 2))
    tag = getattr(args, "tag", None)
    suffix = "" if not tag else f"-{tag}"
    output_root = Path(args.output) / f"{platform}-{arm}-seed{seed}{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)

    produced: list[str] = []
    clip_reports: list[dict[str, Any]] = []
    for position, (entry, start) in enumerate(chosen):
        clip_id = plan[position]["clip_id"]
        clip_dir = output_root / clip_id
        window = load_window_batch(
            entry,
            start=int(start),
            burn_in=0,
            gradient=int(args.frames),
            supervision_points=1,
            proprio_history=proprio_history,
            loader=loader,
        )
        white_level = window.metadata.get("raw_white_level")
        rgb = load_rgb_clip(
            rgb_root,
            entry,
            camera_keys=spec.camera_keys,
            start=int(start),
            frames=int(args.frames),
        )
        rgb_images = rgb["images"].get(spec.camera_keys[camera_index])
        if rgb_images is not None and int(rgb_images.shape[0]) < int(args.frames):
            print(
                f"[warn] {entry.relpath} 的 RGB 只读到 {int(rgb_images.shape[0])} 帧"
                f"（RAW {int(args.frames)} 帧）",
                file=sys.stderr,
            )
        encoder.reset(f"{platform}:{window.episode_id}:{int(start)}")
        state = None
        frame_records: list[str] = []
        notes = [
            f"checkpoint   : step {step}" + ("" if step > 0 else " (structural init, untrained)"),
        ]
        for offset in range(int(args.frames)):
            values = [window.raw[name][offset].to(device) for name in window.camera_ids]
            masks = [
                saturation_mask(window.raw[name][offset], platform, white_level=white_level)
                for name in window.camera_ids
            ]
            saturation = (
                None if any(mask is None for mask in masks) else [mask.to(device) for mask in masks]
            )
            context = EmbodimentContext(
                proprio=window.proprio[offset].flip(0).reshape(-1).to(device),
                proprio_available=True,
                time_unit="frame",
                metadata={
                    "platform": platform,
                    "proprio_expr": spec.proprio_expr,
                    "ordering": "newest_first",
                    "padded_frames": int(window.proprio_padded[offset]),
                },
            )
            with torch.no_grad():
                output = encoder.step(
                    values,
                    context,
                    state,
                    camera_id=list(window.camera_ids),
                    frame_key=(window.episode_id, int(window.frame_ids[offset])),
                    measurement_valid=None,
                    saturation=saturation,
                    new_measurement=True,
                    collect_attention=True,
                )
            state = output.state
            debug = output.debug
            support = debug.get("support", {}).get(camera, {})
            quality = debug.get("quality", {}).get(camera, {})
            modes = debug.get("mode_weights", {}).get(camera, {})
            counts = support.get("media_tokens") or {}
            dense_tokens = int(counts.get("dense_tokens") or 0)
            latent_hw = tuple(int(value) for value in support.get("feature_size_hw", (0, 0)))
            attention = None
            dense_mass = None
            raw_attention = debug.get("readout_attention", {}).get(camera)
            if raw_attention is not None and dense_tokens and all(latent_hw):
                full = _numpy(raw_attention)
                if full is not None:
                    media = int(full.shape[-1])
                    dense_weight = float(full.reshape(-1, media)[:, :dense_tokens].mean())
                    dense_mass = dense_weight * float(dense_tokens)  # 每个 query 的稠密总质量
                    attention = attention_to_map(
                        raw_attention, dense_tokens=dense_tokens, latent_hw=latent_hw
                    )

            quality_scalar = None
            r_diag = quality.get("R_diag")
            if r_diag is not None:
                r_array = _numpy(r_diag)
                if r_array is not None:
                    r_flat = r_array.reshape(-1, r_array.shape[-1])
                    quality_scalar = np.log10(np.maximum(r_flat.mean(axis=-1), 1e-12)).astype(
                        np.float32
                    )

            trajectory = None
            track = support.get("trajectory")
            if track:
                stack = [
                    np.asarray(
                        item.detach().cpu().numpy() if hasattr(item, "detach") else item,
                        dtype=np.float32,
                    ).reshape(-1, 2)
                    for item in track
                ]
                if stack and all(item.shape == stack[0].shape for item in stack):
                    trajectory = np.stack(stack, axis=0)

            posterior = modes.get("posterior")
            visual = FrameVisual(
                platform=platform,
                camera=camera,
                clip_id=clip_id,
                frame_index=int(window.frame_ids[offset]),
                raw_hwc=np.asarray(
                    window.raw[spec.camera_ids[camera_index]][offset].detach().cpu().numpy(),
                    dtype=np.float32,
                ),
                rgb_hwc=None if rgb_images is None else rgb_images[offset],
                rgb_source=rgb.get("members", {}).get(spec.camera_keys[camera_index]),
                rgb_file=short_relpath(entry.relpath),
                flow_px=_numpy(debug.get("flow", {}).get(camera)),
                flow_valid=offset > 0,
                quality_xy=_numpy(support.get("xy_px")),
                quality_scalar=quality_scalar,
                mode_revision=None if posterior is None else _numpy(posterior)[..., 1].reshape(-1),
                support_xy=_numpy(support.get("xy_px")),
                support_valid=_numpy(support.get("valid")),
                support_ids=_numpy(support.get("ids")),
                support_age=_numpy(support.get("age")),
                trajectory=trajectory,
                attention_map=attention,
                attention_dense_mass=dense_mass,
                tag=tag,
                notes=notes,
            )
            figure, meta = compose_frame(visual, dpi=int(args.dpi))
            path = clip_dir / f"frame_{int(window.frame_ids[offset]):07d}.png"
            write_png(figure, path, dpi=int(args.dpi))
            plt.close(figure)
            relpath_png = str(path.relative_to(Path(args.output)))
            produced.append(relpath_png)
            frame_records.append(relpath_png)
            meta.update(
                {
                    "artifact": relpath_png,
                    "checkpoint_step": step,
                    "checkpoint_path": str(checkpoint),
                }
            )
            clip_dir.joinpath(f"frame_{int(window.frame_ids[offset]):07d}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=str)
            )
            del output, debug
            print(f"[visuals] {relpath_png}", flush=True)

        clip_reports.append(
            {
                **plan[position],
                "artifacts": frame_records,
                "rgb": {key: value for key, value in rgb.items() if key != "images"},
            }
        )

    manifest = {
        "schema": "n04-v1.1-s2-visuals-v1",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform,
        "arm": arm,
        "seed": seed,
        "split": split,
        "device": str(device),
        "cpu_threads": thread_cap if thread_cap else None,
        "camera": camera,
        "tag": tag,
        "checkpoint": checkpoint_info,
        "data": {"root": str(data_root), "split_file": str(split_file), "entries": len(entries)},
        "frames_per_clip": int(args.frames),
        "display_mappings": {
            "raw": {
                "formula": RAW_DISPLAY[platform].formula(),
                "white_level": RAW_DISPLAY[platform].white_level,
                "log_gain": RAW_DISPLAY[platform].log_gain,
                "unit": RAW_DISPLAY[platform].unit,
                "measured_on": RAW_DISPLAY[platform].measured_on,
                "note": "平台固定；同一 run 内不逐帧自动拉伸",
            },
            "rgb": "官方 rgb/ 兄弟树同名相对路径；缺失即标“未提供”，不用 RAW 冒充",
            "quality_log10_range": list(QUALITY_LOG10_RANGE),
            "mode_weights_range": [0.0, 1.0],
            "attention": {
                "formula": f"a * M，固定 [0, {ATTENTION_VMAX:g}]",
                "note": "1.0 = 均匀读取；模型读取分布，不是像素真值",
            },
            "flow": {
                "encoding": "hue=atan2(v,u)/2pi, value=r/(r+8)",
                "note": "flow 定义在上一帧坐标；箭头 1:1 像素位移",
            },
        },
        "coordinates": "原始像素 x∈[0,W-1], y∈[0,H-1]；所有面板与输入帧同帧同坐标",
        "clips": clip_reports,
        "artifacts": produced,
        "notes": [
            "图例：叠加内容是模型估计",
            "attention 是模型读取分布，不是像素真值",
            "潜在方差 R 不是毫米",
            "模式权重不是外部运动概率",
            "RGB 缺失时显示“未提供”",
        ],
    }
    manifest_path = output_root / "index.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    print(f"[visuals] manifest {manifest_path}（{len(produced)} 张图）")
    return {"manifest": str(manifest_path), "clips": clip_reports, "generated": len(produced)}


def summary_run(args: argparse.Namespace) -> dict[str, Any]:
    """``--summary-only``：只做统计汇总，不加载模型、不写训练状态。"""
    from ..training.n04_v11 import load_config

    config = load_config(args.config, data_root=args.data_root)
    platform = str(args.platform or config.get("platform") or "")
    arm = str(args.arm or config.get("arm", "D"))
    seed = int(args.seed if args.seed is not None else config.get("seed", 101))
    root = Path(args.output)
    if not root.is_dir():
        raise SystemExit(f"尚未生成：{root} 不存在（先出图再汇总）")
    summary = summarize_frames(root)
    summary.update({"platform": platform, "arm": arm, "seed": seed})
    path = root / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    table = format_summary_table(summary)
    (root / "summary_table.md").write_text(table + "\n")
    print(table)
    print(f"[visuals] summary {path}")
    return {"summary": str(path), "table": table, "generated": 0}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    if args.list_clips or args.summary_only:
        return 0
    return 0 if report.get("generated") else 1


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())

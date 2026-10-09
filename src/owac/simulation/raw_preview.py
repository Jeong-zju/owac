"""Explicit display transforms for a RAW/RGB comparison, never sensor preprocessing."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _font(size):
    from PIL import ImageFont

    return ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size, index=2
    )


def raw_display(cfa, *, white_level: int = 16777215, log_gain: float = 15.0):
    """Fixed black/white mapping plus log compression; no frame-dependent rescaling."""
    import numpy as np

    if cfa.dtype != np.uint32 or cfa.ndim != 2:
        raise ValueError("Expected a uint32 CFA mosaic")
    if white_level <= 0 or log_gain <= 0:
        raise ValueError("Display white level and log gain must be positive")
    linear = np.clip(cfa.astype(np.float32) / white_level, 0, 1)
    mapped = np.log1p(log_gain * linear) / np.log1p(log_gain)
    return np.rint(mapped * 255).astype(np.uint8)


def cfa_colored_display(cfa):
    """Sparse GRBG phase colors identify measurement sites without interpolation."""
    import numpy as np

    gray = raw_display(cfa)
    colors = np.zeros((*gray.shape, 3), dtype=np.uint8)
    colors[0::2, 0::2, 1] = gray[0::2, 0::2]
    colors[0::2, 1::2, 0] = gray[0::2, 1::2]
    colors[1::2, 0::2, 2] = gray[1::2, 0::2]
    colors[1::2, 1::2, 1] = gray[1::2, 1::2]
    return colors


def comparison_frame(rgb, cfa, *, time_seconds, phase, success, roi_center=None):
    """Create a labelled 1280x768 display frame with synchronized Bayer zooms."""
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = cfa.shape
    if (height, width) != (480, 640) or rgb.shape[:2] != cfa.shape:
        raise ValueError("Comparison requires matched 640x480 sensor buffers")
    frame = Image.new("RGB", (1280, 768), "#101820")
    draw = ImageDraw.Draw(frame)
    font = _font(24)
    small = _font(18)
    draw.text((18, 8), "RGB · 常规渲染预览", font=font, fill="#eef4f7")
    draw.text((658, 8), "RAW · 原生 CFA 灰度显示", font=font, fill="#eef4f7")
    draw.text((18, 40), f"{time_seconds:05.2f} s    {phase}", font=small, fill="#a4d8e8")
    draw.text(
        (658, 40), "固定 log 映射 · 无去马赛克 · 原始 uint32 另存", font=small, fill="#a4d8e8"
    )
    rgb_image = Image.fromarray(np.asarray(rgb)[..., :3])
    gray_image = Image.fromarray(raw_display(cfa)).convert("RGB")
    frame.paste(rgb_image, (0, 72))
    frame.paste(gray_image, (640, 72))
    cx, cy = roi_center or (320, 240)
    # Even crop origin keeps the displayed GRBG phase explicit.
    x = min(max(int(cx) - 24, 0), width - 48) // 2 * 2
    y = min(max(int(cy) - 20, 0), height - 40) // 2 * 2
    crop = (x, y, x + 48, y + 40)
    rgb_zoom = rgb_image.crop(crop).resize((192, 160), Image.Resampling.NEAREST)
    mosaic_zoom = Image.fromarray(cfa_colored_display(cfa[y : y + 40, x : x + 48])).resize(
        (192, 160), Image.Resampling.NEAREST
    )
    frame.paste(rgb_zoom, (16, 592))
    frame.paste(mosaic_zoom, (656, 592))
    draw.text((16, 561), "杯体局部 · RGB", font=small, fill="#eef4f7")
    draw.text((656, 561), "同一区域 · Bayer 格点 ×4", font=small, fill="#eef4f7")
    for offset in (0, 640):
        draw.rectangle((x + offset, y + 72, x + 48 + offset, y + 112), outline="#ffb859", width=1)
    draw.text((225, 594), "脚本控制 · 连续物理仿真", font=small, fill="#eef4f7")
    draw.text((225, 628), "接近 → 抓取 → 抬升 → 放置 → 松手", font=small, fill="#a4d8e8")
    draw.text(
        (225, 663), f"原任务判据：{'通过' if success else '尚未通过'}", font=small, fill="#ffb859"
    )
    draw.text((225, 698), "合成传感器 RAW · 同一相机 / 同一时刻", font=small, fill="#a4d8e8")
    draw.text((868, 594), "GRBG：  G  R", font=small, fill="#eef4f7")
    draw.text((868, 626), "               B  G", font=small, fill="#eef4f7")
    draw.text((868, 663), "格点颜色仅标记采样通道", font=small, fill="#a4d8e8")
    draw.text((868, 698), "显示映射固定，不逐帧自动调亮", font=small, fill="#a4d8e8")
    return np.asarray(frame)

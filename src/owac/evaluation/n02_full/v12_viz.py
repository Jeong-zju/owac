"""N02 v1.2 diagnostic figures on fixed dev official trajectories (P2 deliverable).

Five real figures per fixed dev sample, written into the run's ``preview/`` directory:

1. RAW fixed display mapping + **same-source RGB** (derived from that very RAW through a fixed,
   declared ISP - never the cache's paired normal-light RGB) + a local zoom.
2. attention over the original image, from the **real** tensor the readout returns
   (``encoder_attention``: Perceiver encoder cross-attention, head-averaged, mean over latents);
   no synthetic blob is drawn.
3. candidate RAW prediction and the noise-scale residual of the supported explanation.
4. per-round energy/support change of the real inner loop.
5. the action curve: the real pi0.5 sample over this sample's tokens against the loader's
   reference chunk (written as "prediction unavailable" when the backend is absent).

Every figure carries ``episode / camera / frame / iteration / model step``. Output is plain SVG
with embedded PNG (a ~40-line stdlib PNG writer), so ``matplotlib`` is not a dependency, and the
fixed display constants are frozen once per run and reused for every frame instead of being
recomputed per frame.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["render_run", "main", "png_bytes", "heat_rgb", "gray_rgb", "rasterise"]

FONT = "font-family='DejaVu Sans, Helvetica, Arial, sans-serif'"


# ---------------------------------------------------------------------------------------
# tiny PNG writer / colour maps (no image library)
# ---------------------------------------------------------------------------------------


def png_bytes(image: np.ndarray) -> bytes:
    """Encode ``(H, W)`` or ``(H, W, 3)`` uint8 as a PNG (color type 0 or 2)."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        color_type, height, width = 0, *array.shape
        raw = array.reshape(height, width, 1)
    elif array.ndim == 3 and array.shape[2] == 3:
        color_type, height, width = 2, *array.shape[:2]
        raw = array
    else:
        raise ValueError(f"png_bytes expects (H,W) or (H,W,3), got {tuple(array.shape)}")
    payload = b"".join(b"\x00" + raw[row].tobytes() for row in range(height))

    def chunk(name: bytes, data: bytes) -> bytes:
        body = name + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(payload, 6))
        + chunk(b"IEND", b"")
    )


def _ramp(stops: Sequence[tuple[float, tuple[int, int, int]]], values: np.ndarray) -> np.ndarray:
    """Piecewise-linear colour ramp over ``values`` in [0, 1] (declared display mapping)."""
    grid = np.asarray(values, dtype=np.float32)
    out = np.zeros(grid.shape + (3,), dtype=np.float32)
    for index in range(len(stops) - 1):
        low, low_colour = stops[index]
        high, high_colour = stops[index + 1]
        weight = np.clip((grid - low) / max(high - low, 1e-6), 0.0, 1.0)
        inside = ((grid >= low) & (grid <= high)).astype(np.float32)
        for channel in range(3):
            segment = (1.0 - weight) * low_colour[channel] + weight * high_colour[channel]
            out[..., channel] = np.where(inside > 0, segment, out[..., channel])
    out = np.where(grid[..., None] < stops[0][0], np.asarray(stops[0][1], np.float32), out)
    out = np.where(grid[..., None] > stops[-1][0], np.asarray(stops[-1][1], np.float32), out)
    return np.clip(out, 0, 255).astype(np.uint8)


HEAT_STOPS = (
    (0.0, (0, 0, 0)),
    (0.25, (0, 0, 180)),
    (0.5, (0, 200, 120)),
    (0.75, (255, 220, 0)),
    (1.0, (255, 20, 0)),
)


def heat_rgb(values: np.ndarray, *, maximum: float | None = None) -> np.ndarray:
    """Declared heat ramp; normalised by the frame maximum unless one is given."""
    grid = np.asarray(values, dtype=np.float32)
    scale = float(maximum if maximum is not None else max(grid.max(), 1e-6))
    return _ramp(HEAT_STOPS, np.clip(grid / max(scale, 1e-6), 0.0, 1.0))


def gray_rgb(values: np.ndarray, *, white_level: float, gamma: float) -> np.ndarray:
    """Fixed display mapping: ``clip(x / white_level, 0, 1) ** (1 / gamma)``."""
    grid = np.asarray(values, dtype=np.float32) / max(float(white_level), 1e-6)
    display = np.clip(grid, 0.0, 1.0) ** (1.0 / max(float(gamma), 1e-6))
    return np.repeat((display * 255.0).astype(np.uint8)[..., None], 3, axis=2)


def rgb_display(values: np.ndarray, *, white_level: float, gamma: float, gains) -> np.ndarray:
    """Same-source RGB: the same RAW through the fixed declared ISP (gains + gamma)."""
    grid = np.asarray(values, dtype=np.float32)
    if grid.ndim == 2:
        grid = grid[..., None]
    gain = np.asarray(gains, dtype=np.float32).reshape(1, 1, -1)
    scaled = np.clip(grid * gain / max(float(white_level), 1e-6), 0.0, 1.0)
    display = scaled ** (1.0 / max(float(gamma), 1e-6))
    if display.shape[2] == 1:
        display = np.repeat(display, 3, axis=2)
    elif display.shape[2] == 2:
        display = np.concatenate([display, display[..., :1]], axis=2)
    return (display[..., :3] * 255.0).astype(np.uint8)


def _resize(image: np.ndarray, size: tuple[int, int], *, mode: str = "area") -> np.ndarray:
    """Resize display pixels or floating scientific maps without quantising the latter."""
    tensor = torch.as_tensor(np.asarray(image, dtype=np.float32))
    squeeze = tensor.ndim == 2
    if squeeze:
        tensor = tensor[..., None]
    tensor = tensor.permute(2, 0, 1)[None]
    out = F.interpolate(tensor, size=size, mode=mode)
    out = out[0].permute(1, 2, 0).numpy()
    if squeeze:
        out = out[..., 0]
    if np.asarray(image).dtype == np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def rasterise(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    height: int,
    width: int,
    max_side: int = 160,
) -> np.ndarray:
    """Scatter sampled sensor cells onto their own grid and fill the gaps by dilation.

    ``coords`` is ``(Q, 2)`` in ``(row, column)`` pixel centres. Unsampled cells take the nearest
    sampled cell's value (declared nearest fill); cells still unreached take the sampled median.
    """
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    grid = np.full((height, width), np.nan, dtype=np.float64)
    for index in range(points.shape[0]):
        row = int(min(max(np.floor(points[index, 0]), 0), height - 1))
        column = int(min(max(np.floor(points[index, 1]), 0), width - 1))
        grid[row, column] = data[index]
    filled = np.isfinite(grid)
    if not filled.any():
        return np.zeros((max_side, max_side), dtype=np.float32)
    for _ in range(64):
        if filled.all():
            break
        source = np.where(filled, grid, 0.0)
        counts = filled.astype(np.float64)
        neighbours = np.zeros_like(source)
        neighbour_counts = np.zeros_like(counts)
        for shift, axis in (
            ((1, 0), 0),
            ((-1, 0), 0),
            ((0, 1), 1),
            ((0, -1), 1),
        ):
            neighbours += np.roll(source, shift, axis=axis)
            neighbour_counts += np.roll(counts, shift, axis=axis)
        candidate = (neighbour_counts > 0) & ~filled
        grid = np.where(candidate, neighbours / np.maximum(neighbour_counts, 1.0), grid)
        filled = np.isfinite(grid)
    grid = np.where(np.isfinite(grid), grid, float(np.nanmedian(data)))
    step = max(1, int(math.ceil(max(height, width) / float(max_side))))
    if step > 1:
        grid = grid[: (height // step) * step, : (width // step) * step]
        grid = grid.reshape(grid.shape[0] // step, step, grid.shape[1] // step, step).mean((1, 3))
    return grid


# ---------------------------------------------------------------------------------------
# SVG scaffolding
# ---------------------------------------------------------------------------------------


def _escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&apos;")
    )


def _text(
    x: float, y: float, text: Any, *, size: int = 12, colour: str = "#111", anchor: str = "start"
) -> str:
    return (
        f"<text x='{x:.0f}' y='{y:.0f}' font-size='{size}' fill='{colour}' "
        f"text-anchor='{anchor}' {FONT}>{_escape(text)}</text>"
    )


def _image(
    png: bytes | np.ndarray, x: float, y: float, width: float, height: float, label: str
) -> str:
    if isinstance(png, np.ndarray):
        png = png_bytes(png)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("SVG image payload must be an encoded PNG, not raw pixel bytes")
    payload = base64.b64encode(png).decode("ascii")
    return (
        f"<image x='{x:.0f}' y='{y:.0f}' width='{width:.0f}' height='{height:.0f}' "
        f"xlink:href='data:image/png;base64,{payload}'/>"
        + _text(x, y - 6, label, size=12)
        + f"<rect x='{x:.0f}' y='{y:.0f}' width='{width:.0f}' height='{height:.0f}' "
        "fill='none' stroke='#999'/>"
    )


def _svg(width: int, height: int, body: list[str], *, title: str, labels: str) -> str:
    header = [
        f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' "
        f"width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        f"<title>{_escape(title)}</title>",
        f"<rect width='{width}' height='{height}' fill='#fff'/>",
        _text(14, 22, title, size=16, colour="#003"),
        _text(14, 40, labels, size=12, colour="#333"),
    ]
    return "\n".join([*header, *body, "</svg>"]) + "\n"


def _polyline(points: Sequence[tuple[float, float]], colour: str, *, width: float = 1.6) -> str:
    if not points:
        return ""
    body = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"<polyline fill='none' stroke='{colour}' stroke-width='{width}' points='{body}'/>"


def _line_chart(
    series: Sequence[dict],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    x_labels: Sequence[str],
) -> list[str]:
    """A minimal line chart in SVG; one dict per series with ``values``/``colour``/``name``."""
    finite = [float(item) for group in series for item in group["values"] if math.isfinite(item)]
    if not finite:
        return [_text(x, y + height / 2, f"{title}: no finite values", size=12)]
    low, high = min(finite), max(finite)
    span = max(high - low, 1e-9)
    body = [_text(x, y - 8, title, size=13, colour="#003")]
    body.append(_text(x, y + height + 26, y_label, size=11, colour="#666"))
    body.append(
        f"<rect x='{x:.0f}' y='{y:.0f}' width='{width:.0f}' height='{height:.0f}' "
        "fill='#fafafa' stroke='#ddd'/>"
    )
    body.append(_text(x - 6, y + 12, f"{high:.4g}", size=10, colour="#666", anchor="end"))
    body.append(_text(x - 6, y + height, f"{low:.4g}", size=10, colour="#666", anchor="end"))
    count = max(1, max(len(item["values"]) for item in series) - 1)
    for position, group in enumerate(series):
        values = [float(item) for item in group["values"]]
        points = [
            (
                x + width * (index / count if count else 0.0),
                y + height * (1.0 - (value - low) / span) if math.isfinite(value) else y + height,
            )
            for index, value in enumerate(values)
        ]
        body.append(_polyline(points, group.get("colour", "#06c")))
        if group.get("markers"):
            for index, value in enumerate(values):
                if math.isfinite(value) and group["markers"][index]:
                    px = x + width * (index / count if count else 0.0)
                    py = y + height * (1.0 - (value - low) / span)
                    colour = group.get("colour", "#06c")
                    body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3.2' fill='{colour}'/>")
        if values:
            body.append(
                _text(
                    x + width + 6,
                    y + 12 + 14 * position,
                    group["name"],
                    size=10,
                    colour=group.get("colour", "#06c"),
                )
            )
    for index, label in enumerate(x_labels):
        px = x + width * (index / max(1, len(x_labels) - 1))
        body.append(_text(px, y + height + 14, label, size=10, colour="#666", anchor="middle"))
    return body


# ---------------------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------------------


def _np(value: Any) -> np.ndarray:
    """Host numpy copy of a tensor that may live on the GPU."""
    if torch.is_tensor(value):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def _anchor_capture(
    model: torch.nn.Module, batch: Any, *, device: str, replay: Mapping[str, Any]
) -> dict:
    """Everything the figures need about one fixed dev sample, from the real forward pass."""
    from owac.evaluation.n02_full import v12 as v12_eval

    rows = int(np.asarray(batch.frame_index).reshape(-1).size)
    anchor_index = rows - 1
    observation = v12_eval.row_observation(batch.observation, anchor_index)
    inp = v12_eval.inference_input(batch, observation=observation, rows=[anchor_index])
    inference = model.inference
    with torch.no_grad():
        queries = inference.build_queries(inp, batch=1)
        spatial = inference.encoder(inp.observation)
    output = replay["rows"][-1]
    aux = dict(output.diagnostics.get("readout_aux") or {})
    return {
        "rows": rows,
        "anchor_index": anchor_index,
        "instruction": inp,
        "queries": queries,
        "spatial": spatial,
        "output": output,
        "aux": aux,
        "frame_index": int(np.asarray(batch.frame_index).reshape(-1)[anchor_index]),
    }


def _camera_grid(frame: Any) -> tuple[int, int, np.ndarray]:
    values = np.asarray(frame.values)
    if values.ndim == 4:
        values = values[0]
    if values.ndim == 2:
        values = values[None]
    height, width = int(values.shape[-2]), int(values.shape[-1])
    return height, width, values.astype(np.float32)


def _token_sensor_coords(spatial: Any, readout: Any, camera_id: str, multi: bool) -> np.ndarray:
    """Sensor ``(column, row)`` of every spatial token, in the readout's own token order."""
    blocks = []
    for level_id in readout.level_ids:
        key = f"{camera_id}::{level_id}" if multi else level_id
        coords = _np(spatial.coords[key])[0].reshape(-1, 2)
        blocks.append(coords)
    return np.concatenate(blocks, axis=0)


def _attention_maps(capture: Mapping[str, Any], settings: Any) -> dict:
    """Per-camera attention mass of the real readout tensor, averaged over latents."""
    aux = capture["aux"]
    tensor = aux.get("encoder_attention")
    if not torch.is_tensor(tensor):
        return {"available": False, "reason": "readout aux carries no encoder_attention tensor"}
    mass = _np(tensor[0].mean(dim=0))  # (T,) mean over latents
    camera_index = aux.get("token_camera_index")
    if not torch.is_tensor(camera_index):
        return {"available": False, "reason": "readout aux carries no token_camera_index"}
    camera_index = _np(camera_index).reshape(-1)
    n_spatial = int(aux.get("n_spatial_tokens", len(mass)))
    readout = capture["model"].inference.readout
    spatial = capture["spatial"]
    multi = len(spatial.camera_ids) > 1
    maps: dict[str, Any] = {
        "available": True,
        "layer": "readout_v12.N02ReadoutV12 encoder cross-attention (softmax over T tokens)",
        "reduction": "head-averaged by the module, then mean over the 128 latent queries",
        "n_spatial_tokens": int(n_spatial),
        "normalisation": "divided by the frame maximum for display",
    }
    cursor = 0
    for slot, camera_id in enumerate(spatial.camera_ids):
        coords = _token_sensor_coords(spatial, readout, camera_id, multi)
        count = int(coords.shape[0])
        selected = camera_index[cursor : cursor + count]
        values = mass[cursor : cursor + count]
        valid = selected == slot
        maps[str(camera_id)] = {"coords": coords[valid], "attention": values[valid]}
        cursor += count
    maps["state_tokens"] = float(np.sum(mass[n_spatial:]))
    return maps


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------


def figure_raw_display(
    frame: Any,
    *,
    camera_id: str,
    labels: str,
    title: str,
    white_level: float,
    gamma: float,
    gains: Sequence[float],
    zoom_box: tuple[int, int, int, int],
    max_side: int = 200,
) -> bytes:
    height, width, values = _camera_grid(frame)
    gray = values.mean(axis=0) if values.shape[0] > 1 else values[0]
    raw_panel = _resize(gray_rgb(gray, white_level=white_level, gamma=1.0), (max_side, max_side))
    rgb_panel = _resize(
        rgb_display(values.transpose(1, 2, 0), white_level=white_level, gamma=gamma, gains=gains),
        (max_side, max_side),
    )
    row_start, row_stop, col_start, col_stop = zoom_box
    crop = np.transpose(values, (1, 2, 0))[row_start:row_stop, col_start:col_stop]
    zoom = _resize(
        rgb_display(crop, white_level=white_level, gamma=gamma, gains=gains),
        (max_side, max_side),
        mode="nearest",
    )
    caption = (
        f"declared display: RAW = clip(x/white_level,0,1) linear (white_level={white_level:g}); "
        f"same-source RGB = clip(x*gains/white_level,0,1)**(1/{gamma:g}) with gains="
        f"{[round(float(g), 3) for g in gains]}; both panels come from this very RAW "
        f"({values.shape[0]} channels), never from the cache's paired RGB; zoom = sensor rows "
        f"{row_start}:{row_stop} cols {col_start}:{col_stop}, nearest-neighbour upscale"
    )
    body = [
        _image(raw_panel, 20, 80, max_side, max_side, f"RAW fixed display ({camera_id})"),
        _image(
            rgb_panel, 20 + max_side + 30, 80, max_side, max_side, "same-source RGB (declared ISP)"
        ),
        _image(zoom, 20 + 2 * (max_side + 30), 80, max_side, max_side, "local zoom (ISP)"),
    ]
    body.append(_text(20, 80 + max_side + 28, caption, size=11, colour="#444"))
    body.append(
        _text(
            20,
            80 + max_side + 46,
            f"raw min={float(values.min()):.4g} max={float(values.max()):.4g}",
            size=11,
        )
    )
    return _svg(
        40 + 3 * max_side + 60,
        160 + max_side,
        body,
        title=title,
        labels=labels,
    ).encode("utf-8")


def figure_attention(
    frame: Any,
    attention: Mapping[str, Any],
    *,
    camera_id: str,
    labels: str,
    title: str,
    white_level: float,
    max_side: int = 200,
) -> bytes:
    height, width, values = _camera_grid(frame)
    gray = values.mean(axis=0) if values.shape[0] > 1 else values[0]
    base = _resize(gray_rgb(gray, white_level=white_level, gamma=1.0), (max_side, max_side)).astype(
        np.float32
    )
    entry = attention.get(str(camera_id)) if attention.get("available") else None
    if entry is None or len(entry["attention"]) == 0:
        body = [
            _text(
                20, 100, "attention tensor unavailable in the readout aux", size=13, colour="#900"
            )
        ]
        return _svg(520, 200, body, title=title, labels=labels).encode("utf-8")
    grid = rasterise(
        entry["coords"], entry["attention"], height=height, width=width, max_side=max_side
    )
    grid = _resize(grid - float(np.nanmin(grid)), (max_side, max_side), mode="nearest")
    heat = heat_rgb(grid.astype(np.float32))
    overlay = np.clip(0.55 * heat.astype(np.float32) + 0.45 * base, 0, 255).astype(np.uint8)
    body = [
        _image(overlay, 20, 80, max_side, max_side, f"attention over RAW ({camera_id})"),
        _image(heat, 20 + max_side + 30, 80, max_side, max_side, "attention heat (real tensor)"),
        _text(
            20,
            80 + max_side + 28,
            "layer: "
            f"{attention.get('layer')}; {attention.get('reduction')}; "
            f"{attention.get('normalisation')}; state tokens take "
            f"{float(attention.get('state_tokens', float('nan'))):.3g} "
            "of the mass (128 latents x T tokens)",
            size=11,
            colour="#444",
        ),
        _text(
            20,
            80 + max_side + 46,
            "token -> sensor pixel mapping uses SpatialV12.coords of the same forward; unsampled "
            "cells take the nearest sampled cell (declared fill)",
            size=11,
            colour="#444",
        ),
    ]
    return _svg(40 + 2 * max_side + 30, 160 + max_side, body, title=title, labels=labels).encode(
        "utf-8"
    )


def figure_prediction(
    capture: Mapping[str, Any],
    frame: Any,
    *,
    camera_id: str,
    camera_slot: int,
    labels: str,
    title: str,
    max_side: int = 170,
) -> bytes:
    height, width, _ = _camera_grid(frame)
    queries = capture["queries"]
    output = capture["output"]
    chosen = int(torch.argmax(output.q[0]).item())
    support = float(output.q[0, chosen])
    coords = _np(queries.coords[0])
    camera = _np(queries.camera_index).reshape(-1)
    target = _np(queries.target[0]).astype(np.float32).reshape(-1)
    mask = _np(queries.masks["valid"][0]).reshape(-1)
    mean = _np(output.mean[0, chosen]).astype(np.float32).reshape(-1)
    variance = _np(output.variance[0, chosen]).astype(np.float32).reshape(-1)
    residual = np.abs(target - mean) / np.sqrt(np.maximum(variance, 1e-12))
    selected = (camera == camera_slot) & mask
    if not selected.any():
        body = [_text(20, 100, f"no valid query for {camera_id}", size=13, colour="#900")]
        return _svg(520, 200, body, title=title, labels=labels).encode("utf-8")
    observed = rasterise(
        coords[selected], target[selected], height=height, width=width, max_side=max_side
    )
    predicted = rasterise(
        coords[selected], mean[selected], height=height, width=width, max_side=max_side
    )
    residual_grid = rasterise(
        coords[selected], residual[selected], height=height, width=width, max_side=max_side
    )
    noise_grid = rasterise(
        coords[selected],
        np.sqrt(np.maximum(variance[selected], 1e-12)),
        height=height,
        width=width,
        max_side=max_side,
    )
    span = float(np.nanmax(observed) - np.nanmin(observed)) or 1.0
    to_rgb = lambda grid: np.repeat(  # noqa: E731 - local display helper
        np.clip((grid - float(np.nanmin(observed))) / span, 0, 1)[..., None] * 255.0, 3, axis=2
    ).astype(np.uint8)
    body = [
        _image(to_rgb(observed), 20, 80, max_side, max_side, "observed RAW (sampled cells)"),
        _image(
            to_rgb(predicted),
            20 + max_side + 20,
            80,
            max_side,
            max_side,
            "candidate RAW prediction",
        ),
        _image(
            heat_rgb(residual_grid),
            20 + 2 * (max_side + 20),
            80,
            max_side,
            max_side,
            "|standardised residual|",
        ),
        _image(
            heat_rgb(noise_grid),
            20 + 3 * (max_side + 20),
            80,
            max_side,
            max_side,
            "noise scale sqrt(variance)",
        ),
        _text(
            20,
            80 + max_side + 28,
            f"supported candidate {chosen} of {int(output.q.shape[1])} (support q={support:.4f}); "
            f"cells sampled at stride {capture.get('stride')} in the declared measurement space; "
            "unsampled cells take the nearest sampled cell (declared)",
            size=11,
            colour="#444",
        ),
        _text(
            20,
            80 + max_side + 46,
            f"|standardised residual| mean={float(np.nanmean(residual[selected])):.4f}, "
            f"max={float(np.nanmax(residual[selected])):.4f}; sqrt(variance) mean="
            f"{float(np.nanmean(np.sqrt(np.maximum(variance[selected], 1e-12)))):.4f}",
            size=11,
        ),
    ]
    return _svg(30 + 4 * (max_side + 20), 160 + max_side, body, title=title, labels=labels).encode(
        "utf-8"
    )


def figure_rounds(output: Any, *, labels: str, title: str, rounds: int) -> bytes:
    series = []
    colours = ("#06c", "#c30", "#093", "#a0a", "#888")
    energies = [_np(item["energies_before"][0]) for item in output.rounds]
    energies_after = [_np(item["energies_after"][0]) for item in output.rounds]
    accepted = [bool(item["accepted"][0].item()) for item in output.rounds]
    candidates = int(energies[0].shape[0]) if energies else 0
    markers: list[bool] = []
    for round_index in range(len(energies)):
        markers.extend([False, accepted[round_index]])
    for candidate in range(candidates):
        values = [float(group[candidate]) for group in energies]
        after = [float(group[candidate]) for group in energies_after]
        series.append(
            {
                "name": f"candidate {candidate}",
                "colour": colours[candidate % len(colours)],
                "values": [value for pair in zip(values, after, strict=True) for value in pair],
                "markers": list(markers),
            }
        )
    support = [float(item) for item in _np(output.q[0])]
    support_series = [
        {"name": "final support q", "colour": "#06c", "values": support},
    ]
    body = _line_chart(
        series,
        x=70,
        y=90,
        width=460,
        height=170,
        title="complete objective per candidate, before -> after each accepted round",
        y_label="E(q) (sensor-domain energy, same measurement set)",
        x_labels=[f"r{index}{suffix}" for index in range(len(energies)) for suffix in ("b", "a")],
    )
    body += _line_chart(
        support_series,
        x=70,
        y=330,
        width=460,
        height=110,
        title="final support weights q = softmax(log p0 - E/tau) (not a calibrated posterior)",
        y_label="q",
        x_labels=[f"k{index}" for index in range(len(support))],
    )
    body.append(
        _text(
            70,
            490,
            f"inner rounds={rounds}; accepted rounds={sum(accepted)}; "
            f"accepted steps never raise the complete objective (see diagnostics)",
            size=11,
            colour="#444",
        )
    )
    return _svg(600, 520, body, title=title, labels=labels).encode("utf-8")


def figure_actions(
    predicted: np.ndarray | None,
    reference: np.ndarray | None,
    *,
    labels: str,
    title: str,
    control_dim: int,
    reason: str = "",
) -> bytes:
    body: list[str] = []
    if reference is None:
        body.append(
            _text(
                24,
                100,
                "no reference action block available for this sample",
                size=13,
                colour="#900",
            )
        )
        return _svg(700, 200, body, title=title, labels=labels).encode("utf-8")
    reference = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    horizon = int(reference.shape[0])
    dims = min(control_dim, int(reference.shape[1]))
    series = []
    for dim in range(dims):
        series.append(
            {
                "name": f"reference dim {dim}",
                "colour": "#06c",
                "values": [float(item) for item in reference[:, dim]],
            }
        )
    if predicted is not None:
        predicted = np.asarray(predicted, dtype=np.float64)
        for dim in range(min(dims, int(predicted.shape[1]))):
            series.append(
                {
                    "name": f"predicted dim {dim}",
                    "colour": "#c30",
                    "values": [float(item) for item in predicted[:horizon, dim]],
                }
            )
    body += _line_chart(
        series,
        x=70,
        y=90,
        width=560,
        height=220,
        title="pi0.5 action chunk: predicted (red) vs loader reference (blue), normalised units",
        y_label="normalised action",
        x_labels=["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
        if horizon == 10
        else [],
    )
    note = (
        f"control dims plotted={dims}/{control_dim}, horizon={horizon}; "
        "the values are the loader's normalised units "
        "(inverse normalisation happens at take_action time, not here)"
    )
    if reason:
        note = f"prediction unavailable: {reason}; " + note
    body.append(_text(70, 350, note, size=11, colour="#444"))
    return _svg(700, 380, body, title=title, labels=labels).encode("utf-8")


# ---------------------------------------------------------------------------------------
# run rendering
# ---------------------------------------------------------------------------------------


def _declared_display(
    values: np.ndarray, frame: Any, dataset_format: str
) -> tuple[float, float, list[float]]:
    """The fixed display constants of this run (frozen once, then reused for every frame)."""
    metadata = dict(getattr(frame, "metadata", {}) or {})
    white = metadata.get("white_level")
    if isinstance(white, (int, float)) and float(white) > 0:
        white_level = float(white)
    elif dataset_format == "libero":
        white_level = 1.0
    else:
        white_level = (
            float(np.percentile(values[np.isfinite(values)], 99.9)) if values.size else 1.0
        )
        white_level = white_level if white_level > 0 else 1.0
    gains = metadata.get("white_balance") or [1.0, 1.0, 1.0]
    gains = [float(item) for item in np.asarray(gains, dtype=float).reshape(-1)][:3]
    while len(gains) < 3:
        gains.append(1.0)
    return white_level, 2.2, gains


def _labels(
    settings: Any,
    identity: Mapping[str, Any],
    camera_id: str,
    frame_index: int,
    round_index: int,
    step: Any,
) -> str:
    episode = (
        identity.get("episode_group_id")
        or identity.get("episode")
        or identity.get("episode_id")
        or identity.get("relative_path")
        or "unidentified"
    )
    return (
        f"episode={episode} | camera={camera_id} | frame={frame_index} | "
        f"iteration=inner round {round_index} (final accepted) | "
        f"model step={step if step is not None else 'unknown'}"
    )


def render_run(
    model: torch.nn.Module,
    settings: Any,
    *,
    device: str,
    backend: Mapping[str, Any] | None = None,
    model_step: Any = None,
) -> dict:
    """Write the five figures of every fixed dev sample into ``settings.preview_dir``."""
    from owac.evaluation.n02_full import v12 as v12_eval

    backend = dict(backend or {})
    preview = Path(settings.preview_dir)
    preview.mkdir(parents=True, exist_ok=True)
    source = v12_eval.build_source(settings, "dev", max_windows=None)
    sample_receipt = Path(settings.output) / "dev_samples.json"
    identities = []
    if sample_receipt.is_file():
        try:
            identities = (
                json.loads(sample_receipt.read_text(encoding="utf-8")).get("identities") or []
            )
        except ValueError:
            identities = []
    indices = [int(item["anchor_index"]) for item in identities] or v12_eval.anchor_indices(
        source, settings.dev_samples
    )
    files: list[str] = []
    page_entries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    display_receipt: dict[str, Any] = {}
    for index in indices:
        batch = source.window(index, index + 1)
        rows = int(np.asarray(batch.frame_index).reshape(-1).size)
        # the causal rows are replayed with their real frame indices (the check-labelled step is
        # only defined for the three-state check, not for the figures)
        replay = v12_eval.replay_window(model, batch, device=device, check_label=0, check=False)
        outputs = replay["rows"]
        capture = _anchor_capture(model, batch, device=device, replay=replay)
        capture["model"] = model
        capture["stride"] = settings.query_stride
        identity = v12_eval.sample_identity(source, index)
        anchor_frame = int(np.asarray(batch.frame_index).reshape(-1)[-1])
        attention = _attention_maps(capture, settings)
        record: dict[str, Any] = {
            "anchor_index": int(index),
            "identity": identity,
            "frame_index": anchor_frame,
            "figures": [],
        }
        for camera_slot, camera_id in enumerate(capture["spatial"].camera_ids):
            frame = v12_eval.row_observation(batch.observation, rows - 1).frames[str(camera_id)]
            height, width, values = _camera_grid(frame)
            white_level, gamma, gains = _declared_display(values, frame, settings.dataset_format)
            display_receipt.setdefault(
                "declared",
                {
                    "raw_display": "clip(x/white_level,0,1) linear, grayscale of the channel mean",
                    "same_source_rgb": "clip(x*gains/white_level,0,1)**(1/gamma)",
                    "white_level": white_level,
                    "gamma": gamma,
                    "gains": gains,
                    "note": (
                        "frozen from the first rendered frame and reused for every "
                        "frame/sample; never recomputed per frame; the cache's paired "
                        "normal-light RGB is not used"
                    ),
                },
            )
            labels = _labels(
                settings, identity, camera_id, anchor_frame, len(outputs[-1].rounds) - 1, model_step
            )
            zoom = (height // 3, 2 * height // 3, width // 3, 2 * width // 3)
            name = f"{index:03d}_{camera_id}"
            body = figure_raw_display(
                frame,
                camera_id=str(camera_id),
                labels=labels,
                title=(
                    f"N02 v1.2 RAW / same-source RGB / zoom - {identity.get('relative_path', '')}"
                ),
                white_level=white_level,
                gamma=gamma,
                gains=gains,
                zoom_box=zoom,
            )
            path = preview / f"01_raw_rgb_zoom_{name}.svg"
            path.write_bytes(body)
            files.append(path.name)
            page_entry = {
                "camera": str(camera_id),
                "frame": anchor_frame,
                "loop": len(outputs[-1].rounds) - 1,
                "raw": f"preview/{path.name}",
                "rgb": f"preview/{path.name}",
            }
            body = figure_attention(
                frame,
                attention,
                camera_id=str(camera_id),
                labels=labels,
                title=(
                    "N02 v1.2 real readout attention over RAW - "
                    f"{identity.get('relative_path', '')}"
                ),
                white_level=white_level,
            )
            path = preview / f"02_attention_{name}.svg"
            path.write_bytes(body)
            files.append(path.name)
            page_entry["attention"] = f"preview/{path.name}"
            body = figure_prediction(
                capture,
                frame,
                camera_id=str(camera_id),
                camera_slot=camera_slot,
                labels=labels,
                title=(
                    f"N02 v1.2 candidate RAW prediction / residual / noise scale - "
                    f"{identity.get('relative_path', '')}"
                ),
            )
            path = preview / f"03_prediction_residual_{name}.svg"
            path.write_bytes(body)
            files.append(path.name)
            page_entry["residual"] = f"preview/{path.name}"
            page_entries.append(page_entry)
            record["figures"].extend(files[-3:])
        labels = _labels(
            settings, identity, "all", anchor_frame, len(outputs[-1].rounds) - 1, model_step
        )
        body = figure_rounds(
            outputs[-1],
            labels=labels,
            title=f"N02 v1.2 inner-loop rounds and support - {identity.get('relative_path', '')}",
            rounds=int(model.inference.rounds),
        )
        path = preview / f"04_rounds_support_{index:03d}.svg"
        path.write_bytes(body)
        files.append(path.name)
        record["figures"].append(path.name)
        page_entry = {
            "frame": anchor_frame,
            "loop": len(outputs[-1].rounds) - 1,
            "rounds": f"preview/{path.name}",
        }

        predicted, reference, reason = None, None, ""
        targets = source.targets(index, index + 1)
        # the loader's supervision block is the per-sample (horizon, control_dim) array
        reference = np.asarray(targets["actions"], dtype=np.float64)
        if reference.ndim == 3:
            reference = reference[0]
        reference = np.atleast_2d(reference)
        if backend.get("status") == "ready":
            try:
                predicted, reason = _predict_chunk(
                    model, capture, identity, backend, settings, batch, device
                )
            except Exception as exc:  # noqa: BLE001 - an unavailable figure says why
                predicted, reason = None, f"{type(exc).__name__}: {exc}"
        else:
            reason = str(backend.get("reason", "backend unavailable"))
        body = figure_actions(
            predicted,
            reference,
            labels=labels,
            title=f"N02 v1.2 action curve - {identity.get('relative_path', '')}",
            control_dim=int(getattr(backend.get("protocol"), "control_dim", targets["action_dim"])),
            reason=reason,
        )
        path = preview / f"05_action_curve_{index:03d}.svg"
        path.write_bytes(body)
        files.append(path.name)
        record["figures"].append(path.name)
        page_entry["action"] = f"preview/{path.name}"
        page_entries.append(page_entry)
        record["prediction"] = "ran" if predicted is not None else f"not_run: {reason}"
        records.append(record)

    receipt = {
        "status": "ran",
        "preview_dir": str(preview),
        "files": files,
        "page_preview_files": page_entries,
        "count": len(files),
        "samples": records,
        "display": display_receipt,
        "attention": {
            "layer": "readout_v12.N02ReadoutV12 encoder cross-attention weights (real tensor)",
            "reduction": (
                "head-averaged by the module, mean over latents, max-normalised for display"
            ),
        },
        "labels": "every figure carries episode / camera / frame / iteration / model step",
        "model_step": model_step,
    }
    (preview / "figures.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8"
    )
    return receipt


def _predict_chunk(
    model: torch.nn.Module,
    capture: Mapping[str, Any],
    identity: Mapping[str, Any],
    backend: Mapping[str, Any],
    settings: Any,
    batch: Any,
    device: str,
):
    from owac.core.encoder_contract import VisualTokens
    from owac.evaluation.n02_full import v12 as v12_eval

    policy = backend["policy"]
    tokens = capture["output"].tokens[-1:].detach().clone()
    visual = VisualTokens(
        values=tokens,
        mask=torch.ones(tokens.shape[:2], dtype=torch.bool),
        camera_ids=tuple(backend["camera_ids"]),
        positions=torch.zeros(tokens.shape[:2] + (2,)),
    )
    proprio = v12_eval._anchor_proprio(batch, policy)
    texts = v12_eval.official_prompts(settings, [identity])
    lang_tokens, lang_masks, _ = v12_eval._language_batch(
        policy, texts, v12_eval._anchor_proprio(batch, policy, pad=False), settings
    )
    prediction = policy.predict_action_from_visual_tokens(
        visual,
        lang_tokens.to(device),
        lang_masks.to(device),
        torch.as_tensor(proprio, dtype=torch.float32, device=device),
        num_steps=int(backend["protocol"].num_denoise_steps),
    )
    return np.asarray(prediction["control_actions"], dtype=np.float64)[0], ""


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def build_parser() -> Any:
    parser = argparse.ArgumentParser(
        prog="python -m owac.evaluation.n02_full.v12_viz",
        description=(
            "Render the N02 v1.2 diagnostic figures (RAW/same-source RGB/zoom, real readout "
            "attention over RAW, candidate prediction and noise-scale residual, inner-loop "
            "support, action curve) of the fixed dev samples into <run>/preview/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/encoders/n02_v12/raw_n02_robotwin.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/n02-v12/P2/eval")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--dataset-format", default=None, choices=["robotwin", "libero"])
    parser.add_argument("--camera-ids", default=None, help="comma-separated camera ids")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--anchors-dir", default=None)
    parser.add_argument("--dev-samples", type=int, default=4)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--query-stride", dest="query_stride", type=int, default=None)
    parser.add_argument("--n-queries", dest="n_queries", type=int, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--backend-checkpoint", default=None)
    parser.add_argument("--action-check", default="auto", choices=["auto", "skip"])
    parser.add_argument("--lang-max-len", type=int, default=48)
    parser.add_argument(
        "--model-step",
        type=int,
        default=None,
        help=(
            "model step to label the figures with when the checkpoint file records none; a step "
            "recorded in the file wins"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render the figures of one run directory; the numbers stay in the eval receipts."""
    from owac.evaluation.n02_full import v12 as v12_eval

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings = v12_eval.resolve_settings(args)
    model = v12_eval.build_model(settings, device=settings.device)
    checkpoint = v12_eval.load_checkpoint_into(model, settings.checkpoint)
    backend = v12_eval.load_backend(settings, device=settings.device)
    receipt = render_run(
        model,
        settings,
        device=settings.device,
        backend=backend,
        model_step=v12_eval.resolve_model_step(
            checkpoint.get("training_step_recorded_in_file"), args.model_step
        ),
    )
    receipt["checkpoint"] = checkpoint
    receipt["backend"] = {key: value for key, value in backend.items() if key != "policy"}
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "preview_dir": receipt["preview_dir"],
                "count": receipt["count"],
                "files": receipt["files"][:8],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

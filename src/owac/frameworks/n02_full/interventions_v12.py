"""Real RAW interventions on a served window batch (N02 v1.2 P2).

P1's "cue swap" only permuted ``correct_index`` in a dictionary and the exposure / noise /
shift groups only relabelled the same generated sequence, so none of them changed a
measurement.  Every function here changes the actual RAW pixels of the batch - including the
causal history rows, which is where a real cue swap has to happen too - and returns a record
that names the affected ``(episode, frame, camera)`` rows, the pixel ranges, the applied
strength and the measured difference ``raw_diff_norm`` between the new and the served values.

An intervention that genuinely changed nothing reports ``no_effect = True`` with
``raw_diff_norm == 0.0``; it is never quietly reported as applied.  Interventions never touch
``measurement_ids``, ``frame_index`` or the partition: the *identity* of the measurement stays
the same while its *content* changes, and the modification is recorded in the batch
provenance.

Positions are given in the camera's own pixel grid as ``(row_start, row_stop, col_start,
col_stop)`` boxes, so a report can draw them on the real frame without guessing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from owac.frameworks.n02_full.data_v12 import V12WindowBatch
from owac.raw.rawvla_sequence import fitted_variance_estimate

__all__ = [
    "cue_swap",
    "exposure_shift",
    "no_information",
    "noise_shift",
    "translation_shift",
]

Box = tuple[int, int, int, int]
Transform = Callable[[torch.Tensor, "torch.Tensor | None", Any, str, int], "_Applied"]


# --------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


def _restore(tensor: torch.Tensor, like: Any) -> Any:
    """Return the modified values in the type the batch was served with."""
    if isinstance(like, np.ndarray):
        return tensor.detach().cpu().numpy().astype(like.dtype, copy=False)
    if isinstance(like, torch.Tensor):
        return tensor.to(like.dtype)
    return tensor


def _restore_mask(tensor: torch.Tensor, values: Any) -> Any:
    if isinstance(values, np.ndarray):
        return tensor.detach().cpu().numpy().astype(bool, copy=False)
    return tensor.to(dtype=torch.bool)


def _check_batch(batch: Any) -> V12WindowBatch:
    if not isinstance(batch, V12WindowBatch):
        raise TypeError(
            f"interventions operate on a real served V12WindowBatch, got {type(batch).__name__}"
        )
    if not getattr(batch.observation, "frames", None):
        raise ValueError("the batch carries no measurement frames")
    return batch


def _box_shape(box: Box) -> tuple[int, int]:
    return int(box[1]) - int(box[0]), int(box[3]) - int(box[2])


def _validate_box(box: Any, shape: Sequence[int], where: str) -> Box:
    values = tuple(box)
    if len(values) != 4:
        raise ValueError(f"{where} must be (row_start, row_stop, col_start, col_stop), got {box}")
    row_start, row_stop, col_start, col_stop = (int(value) for value in values)
    if not 0 <= row_start < row_stop <= shape[-2]:
        raise ValueError(f"{where} rows {row_start}:{row_stop} leave the {shape[-2]}-row frame")
    if not 0 <= col_start < col_stop <= shape[-1]:
        raise ValueError(f"{where} cols {col_start}:{col_stop} leave the {shape[-1]}-col frame")
    return (row_start, row_stop, col_start, col_stop)


def _cue_boxes(positions: Any, shape: Sequence[int]) -> tuple[Box, Box]:
    """Two equal-sized boxes: a mapping with two boxes, or a 2-sequence of boxes."""
    if isinstance(positions, Mapping):
        if len(positions) != 2:
            raise ValueError(
                f"cue positions needs exactly two named boxes, got {sorted(positions)}"
            )
        first, second = positions.values()
    elif isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
        if len(positions) != 2:
            raise ValueError(f"cue positions needs exactly two boxes, got {len(positions)}")
        first, second = positions
    else:
        raise ValueError("cue positions must be a mapping or a 2-sequence of boxes")
    box_a = _validate_box(first, shape, "cue box A")
    box_b = _validate_box(second, shape, "cue box B")
    if _box_shape(box_a) != _box_shape(box_b):
        raise ValueError(f"cue boxes must have equal shape: {box_a} vs {box_b}")
    return box_a, box_b


def _row_layout(batch: V12WindowBatch) -> tuple[list[int], list[int]]:
    """``(history rows, current rows)`` of the batch's causal rows."""
    rows = int(np.asarray(batch.frame_index).size)
    stride = max(1, int(batch.history))
    history = [row for row in range(rows) if row % stride != stride - 1]
    current = [row for row in range(rows) if row % stride == stride - 1]
    return history, current


@dataclass(slots=True)
class _Applied:
    values: torch.Tensor
    valid: torch.Tensor | None
    bad: torch.Tensor | None
    boxes: dict[str, list[list[int]]]
    detail: dict[str, Any]
    changed_valid: bool = False


def _apply(
    batch: Any,
    *,
    kind: str,
    strength: Any,
    strength_detail: dict[str, Any],
    transform: Transform,
) -> tuple[V12WindowBatch, dict[str, Any]]:
    served = _check_batch(batch)
    frames: dict[str, Any] = {}
    per_camera: dict[str, float] = {}
    boxes: dict[str, list[list[int]]] = {}
    details: dict[str, dict[str, Any]] = {}
    total_squared = 0.0
    pixels_changed = 0
    valid_removed = 0
    for camera_slot, (camera_id, frame) in enumerate(served.observation.frames.items()):
        values = _as_tensor(frame.values)
        if values.ndim != 4:
            raise ValueError(f"measurement values must be (B, C, H, W), got {tuple(values.shape)}")
        valid = None if frame.valid_mask is None else _as_tensor(frame.valid_mask, dtype=torch.bool)
        applied = transform(values, valid, frame, str(camera_id), camera_slot)
        difference = applied.values.to(torch.float64) - values.to(torch.float64)
        norm = float(difference.pow(2).sum().sqrt())
        total_squared += norm * norm
        per_camera[str(camera_id)] = norm
        pixels_changed += int((difference != 0).sum())
        boxes[str(camera_id)] = applied.boxes[str(camera_id)]
        details[str(camera_id)] = applied.detail
        if applied.changed_valid and applied.valid is not None and valid is not None:
            valid_removed += int((valid & ~applied.valid).sum())
        payload: dict[str, Any] = {"values": _restore(applied.values, frame.values)}
        if applied.valid is not None:
            payload["valid_mask"] = _restore_mask(applied.valid, frame.values)
        if applied.bad is not None:
            payload["bad_mask"] = _restore_mask(applied.bad, frame.values)
        frames[str(camera_id)] = replace(frame, **payload)
    history_rows, current_rows = _row_layout(served)
    record: dict[str, Any] = {
        "kind": str(kind),
        "strength": strength,
        "strength_detail": dict(strength_detail),
        "positions": {
            "pixel_boxes": boxes,
            "rows_touched": list(range(int(np.asarray(served.frame_index).size))),
            "history_rows": history_rows,
            "current_rows": current_rows,
            "measurement_ids": [list(item) for item in served.measurement_ids],
        },
        "raw_diff_norm": math.sqrt(total_squared),
        "per_camera_diff_norm": per_camera,
        "pixels_changed": int(pixels_changed),
        "valid_cells_removed": int(valid_removed),
        "applied_to_history": bool(history_rows),
        "applied_to_current": bool(current_rows),
        "partition": served.partition,
        "no_effect": bool(math.sqrt(total_squared) == 0.0),
        "per_camera_detail": details,
    }
    provenance = dict(served.provenance)
    provenance["interventions"] = [*provenance.get("interventions", []), record]
    new_batch = replace(
        served,
        observation=replace(served.observation, frames=frames),
        provenance=provenance,
    )
    return new_batch, record


# --------------------------------------------------------------------------------------
# real interventions
# --------------------------------------------------------------------------------------


def cue_swap(batch: Any, *, eps: float, positions: Any) -> tuple[V12WindowBatch, dict[str, Any]]:
    """Exchange the real pixel content of two cue regions on every causal row.

    ``positions`` names the two cue boxes in the camera grid (a 2-sequence of
    ``(row_start, row_stop, col_start, col_stop)`` or a mapping with two boxes).  ``eps`` in
    ``(0, 1]`` is the exchange fraction: the result is
    ``(1 - eps) * served + eps * exchanged``, so ``eps=1`` is a pure swap and the recorded
    ``raw_diff_norm`` grows with the declared strength.  History rows are swapped with the
    same boxes, so a prefix replay sees the exchanged measurement too.
    """
    eps = float(eps)
    if not 0.0 < eps <= 1.0:
        raise ValueError(f"eps must lie in (0, 1], got {eps}")

    def transform(
        values: torch.Tensor, valid: torch.Tensor | None, frame: Any, camera_id: str, slot: int
    ) -> _Applied:
        box_a, box_b = _cue_boxes(positions, values.shape)
        rows_a, rows_b = slice(box_a[0], box_a[1]), slice(box_b[0], box_b[1])
        cols_a, cols_b = slice(box_a[2], box_a[3]), slice(box_b[2], box_b[3])
        exchanged = values.clone()
        exchanged[:, :, rows_b, cols_b] = values[:, :, rows_a, cols_a]
        exchanged[:, :, rows_a, cols_a] = values[:, :, rows_b, cols_b]
        new_values = (1.0 - eps) * values + eps * exchanged
        return _Applied(
            values=new_values,
            valid=valid,
            bad=None,
            boxes={camera_id: [list(box_a), list(box_b)]},
            detail={
                "box_a": list(box_a),
                "box_b": list(box_b),
                "box_shape": list(_box_shape(box_a)),
                "eps": eps,
                "exchange_expression": "(1-eps)*served + eps*exchanged",
            },
        )

    return _apply(
        batch,
        kind="cue_swap",
        strength=eps,
        strength_detail={"eps": eps, "boxes": _json_boxes(positions)},
        transform=transform,
    )


def no_information(
    batch: Any, *, seed: int, region: str = "full", mode: str = "matched_noise"
) -> tuple[V12WindowBatch, dict[str, Any]]:
    """Replace the measured content with a genuinely uninformative measurement.

    ``mode='matched_noise'`` draws i.i.d. Gaussian values that keep each row's own per-channel
    mean and standard deviation, so the replacement has the same first two moments but no
    spatial or scene structure; ``mode='uniform_field'`` replaces the region with that same
    per-row mean (a flat field).  Both destroy the identifiable cue while leaving the
    measurement units untouched.  This is the constructed no-information measurement the P1
    run never produced (P1 scored the first frame of an ordinary generated sequence).
    """
    if mode not in ("matched_noise", "uniform_field"):
        raise ValueError(f"unknown no-information mode: {mode!r}")
    if region not in ("full", "cues"):
        raise ValueError(f"unknown no-information region: {region!r}")
    generator = torch.Generator().manual_seed(int(seed))

    def transform(
        values: torch.Tensor, valid: torch.Tensor | None, frame: Any, camera_id: str, slot: int
    ) -> _Applied:
        height, width = int(values.shape[-2]), int(values.shape[-1])
        if region == "full":
            box = [0, height, 0, width]
        else:
            box = [height // 4, 3 * height // 4, width // 4, 3 * width // 4]
        rows = slice(box[0], box[1])
        cols = slice(box[2], box[3])
        window = values[:, :, rows, cols]
        mean = window.mean(dim=(-2, -1), keepdim=True)
        std = window.std(dim=(-2, -1), unbiased=False, keepdim=True)
        if mode == "uniform_field":
            replacement = mean.expand_as(window)
            expression = "per-row-channel mean (flat field)"
        else:
            noise = torch.randn(
                window.shape, generator=generator, dtype=window.dtype, device=window.device
            )
            replacement = mean + std * noise
            expression = "per-row-channel mean + std * N(0,1) (same first two moments)"
        new_values = values.clone()
        new_values[:, :, rows, cols] = replacement
        return _Applied(
            values=new_values,
            valid=valid,
            bad=None,
            boxes={camera_id: [list(box)]},
            detail={
                "mode": mode,
                "region": region,
                "box": list(box),
                "seed": int(seed),
                "expression": expression,
                "row_mean": [float(item) for item in mean.reshape(-1)[:8].detach().tolist()],
                "row_std": [float(item) for item in std.reshape(-1)[:8].detach().tolist()],
                "replacement_fraction": 1.0 if region == "full" else 0.25,
            },
        )

    return _apply(
        batch,
        kind="no_information",
        strength=1.0,
        strength_detail={"mode": mode, "region": region, "seed": int(seed)},
        transform=transform,
    )


def exposure_shift(batch: Any, *, gain: float) -> tuple[V12WindowBatch, dict[str, Any]]:
    """Scale the real measurement by ``gain`` (a real exposure change, not a label).

    No mask is fabricated: a gain that pushes values past a white level is a real clipping
    event whose level is not declared in these caches, so ``saturation_reevaluated`` stays
    ``False`` and the actual before/after means and maxima are recorded instead.
    """
    gain = float(gain)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"gain must be finite and positive, got {gain}")

    def transform(
        values: torch.Tensor, valid: torch.Tensor | None, frame: Any, camera_id: str, slot: int
    ) -> _Applied:
        after = values * gain
        return _Applied(
            values=after,
            valid=valid,
            bad=None,
            boxes={camera_id: [[0, int(values.shape[-2]), 0, int(values.shape[-1])]]},
            detail={
                "gain": gain,
                "mean_before": float(values.detach().to(torch.float64).mean()),
                "mean_after": float(after.detach().to(torch.float64).mean()),
                "min_before": float(values.detach().min()),
                "max_before": float(values.detach().max()),
                "min_after": float(after.detach().min()),
                "max_after": float(after.detach().max()),
                "saturation_reevaluated": False,
                "saturation_note": (
                    "the caches declare no white level, so no saturation mask is fabricated; "
                    "the measured maximum is recorded instead"
                ),
            },
        )

    return _apply(
        batch,
        kind="exposure_shift",
        strength=gain,
        strength_detail={"gain": gain, "saturation_reevaluated": False},
        transform=transform,
    )


def noise_shift(
    batch: Any, *, sigma_scale: float, seed: int = 0
) -> tuple[V12WindowBatch, dict[str, Any]]:
    """Add real signal-dependent noise to the measurement.

    With a declared sensor variance in the frame metadata the noise is
    ``sigma_scale * sqrt(declared_variance)``, i.e. the declared shot/read model itself.  With
    no declared model the scale is *estimated* from the frame's own neighbour differences
    (``fitted_variance_estimate``) and made signal-dependent by the Poisson-like factor
    ``max(x, 0) / mean(x)``; every constant of that estimate is written into the record, and
    the estimate is never called a calibrated sensor variance.
    """
    sigma_scale = float(sigma_scale)
    if not math.isfinite(sigma_scale) or sigma_scale <= 0.0:
        raise ValueError(f"sigma_scale must be finite and positive, got {sigma_scale}")

    def transform(
        values: torch.Tensor, valid: torch.Tensor | None, frame: Any, camera_id: str, slot: int
    ) -> _Applied:
        generator = torch.Generator().manual_seed(int(seed) * 1000003 + slot * 104729 + 17)
        declared = _declared_variance(_metadata_variance(frame, values))
        if declared is not None:
            sigma = torch.sqrt(declared.clamp(min=0.0).to(values.dtype)) * sigma_scale
            detail: dict[str, Any] = {
                "variance_model": "declared",
                "variance_expression": "sigma_scale * sqrt(declared_variance)",
                "variance_is_estimate": False,
            }
        else:
            estimate, info = _fitted_scale(values, valid)
            mean = values.detach().to(torch.float64).mean(dim=(-2, -1), keepdim=True)
            ratio = (values.detach().to(torch.float64).clamp(min=0.0) / mean.clamp(min=1e-6)).clamp(
                max=4.0
            )
            sigma = (sigma_scale * math.sqrt(max(estimate, 0.0)) * torch.sqrt(ratio)).to(
                values.dtype
            )
            detail = {
                "variance_model": "estimated",
                "variance_expression": (
                    "sigma_scale^2 * fitted_scale * clamp(max(x,0)/mean(x), 0, 4)"
                ),
                "variance_is_estimate": True,
                "fitted_scale": float(estimate),
                "fitted_scale_detail": info,
                "ratio_clamp": 4.0,
                "ratio_mean": float(ratio.mean()),
                "calibrated_probability": False,
            }
        noise = (
            torch.randn(values.shape, generator=generator, dtype=values.dtype, device=values.device)
            * sigma
        )
        detail.update(
            {
                "sigma_scale": sigma_scale,
                "seed": int(seed),
                "sigma_mean": float(sigma.detach().to(torch.float64).mean()),
                "sigma_max": float(sigma.detach().max()),
                "draws": int(values.numel()),
            }
        )
        return _Applied(
            values=values + noise,
            valid=valid,
            bad=None,
            boxes={camera_id: [[0, int(values.shape[-2]), 0, int(values.shape[-1])]]},
            detail=detail,
        )

    return _apply(
        batch,
        kind="noise_shift",
        strength=sigma_scale,
        strength_detail={"sigma_scale": sigma_scale, "seed": int(seed)},
        transform=transform,
    )


def translation_shift(batch: Any, *, pixels: Any) -> tuple[V12WindowBatch, dict[str, Any]]:
    """Translate the real image content by whole pixels and mark the vacated band unmeasured.

    ``pixels`` is an int (horizontal shift) or ``(dy, dx)``.  The content is moved without
    wrap-around and the cells that received no measurement are marked invalid/bad, because
    those sensor positions were genuinely not observed after the shift.
    """
    dy, dx = _pixel_offset(pixels)

    def transform(
        values: torch.Tensor, valid: torch.Tensor | None, frame: Any, camera_id: str, slot: int
    ) -> _Applied:
        height, width = int(values.shape[-2]), int(values.shape[-1])
        shifted = torch.zeros_like(values)
        source_rows = slice(max(0, -dy), height - max(0, dy))
        destination_rows = slice(max(0, dy), height - max(0, -dy))
        source_cols = slice(max(0, -dx), width - max(0, dx))
        destination_cols = slice(max(0, dx), width - max(0, -dx))
        shifted[:, :, destination_rows, destination_cols] = values[:, :, source_rows, source_cols]
        # the frozen loader stores validity per spatial cell (no channel axis), so the mask
        # layout is detected from its rank instead of assumed
        if valid is None:
            new_valid = torch.ones(
                (values.shape[0], values.shape[-2], values.shape[-1]),
                dtype=torch.bool,
                device=values.device,
            )
        else:
            new_valid = valid.clone()
        if dy > 0:
            _spatial_mask(new_valid, values.ndim, slice(0, dy), None)[...] = False
        elif dy < 0:
            _spatial_mask(new_valid, values.ndim, slice(height + dy, height), None)[...] = False
        if dx > 0:
            _spatial_mask(new_valid, values.ndim, None, slice(0, dx))[...] = False
        elif dx < 0:
            _spatial_mask(new_valid, values.ndim, None, slice(width + dx, width))[...] = False
        band = [
            0 if dy >= 0 else height + dy,
            dy if dy >= 0 else height,
            0 if dx >= 0 else width + dx,
            dx if dx >= 0 else width,
        ]
        return _Applied(
            values=shifted,
            valid=new_valid,
            bad=~new_valid,
            boxes={camera_id: [[0, height, 0, width], list(band)]},
            detail={
                "pixels": [dy, dx],
                "wrap_around": False,
                "valid_mask_layout": "per_spatial_cell" if new_valid.ndim == 3 else "per_channel",
                "vacated_band_rows_cols": band,
                "fraction_valid_after": float(new_valid.to(torch.float64).mean()),
                "frame_shape": [height, width],
            },
            changed_valid=True,
        )

    return _apply(
        batch,
        kind="translation_shift",
        strength=[dy, dx],
        strength_detail={"pixels": [dy, dx], "wrap_around": False},
        transform=transform,
    )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _spatial_mask(
    mask: torch.Tensor, values_ndim: int, rows: slice | None, cols: slice | None
) -> torch.Tensor:
    """A view of ``mask`` at spatial ``rows``/``cols``, honouring the mask's own layout."""
    if mask.ndim == 3:  # (B, H, W): the frozen loader's per-cell layout
        return mask[:, rows, cols] if rows is not None else mask[:, :, cols]
    if mask.ndim == values_ndim:  # (B, C, H, W): per-channel validity
        return mask[:, :, rows, cols] if rows is not None else mask[:, :, :, cols]
    raise ValueError(
        f"validity mask rank {mask.ndim} is neither per-cell ({values_ndim - 1}) nor "
        f"per-channel ({values_ndim})"
    )


def _pixel_offset(pixels: Any) -> tuple[int, int]:
    if isinstance(pixels, (int, np.integer)):
        return 0, int(pixels)
    values = tuple(int(item) for item in pixels)
    if len(values) != 2:
        raise ValueError(f"pixels must be an int or (dy, dx), got {pixels!r}")
    if values == (0, 0):
        raise ValueError("pixels must move the content; got (0, 0)")
    return values


def _json_boxes(positions: Any) -> Any:
    if isinstance(positions, Mapping):
        return {str(name): [int(item) for item in box] for name, box in positions.items()}
    return [[int(item) for item in box] for box in positions]


def _metadata_variance(frame: Any, values: torch.Tensor) -> Any:
    metadata = getattr(frame, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    variance = metadata.get("variance")
    if variance is None:
        return None
    tensor = _as_tensor(variance)
    if tuple(tensor.shape) != tuple(values.shape):
        return None
    return tensor


def _declared_variance(variance: Any) -> torch.Tensor | None:
    if variance is None:
        return None
    tensor = _as_tensor(variance)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    if not bool(torch.isfinite(tensor).any()):
        return None
    return tensor


def _fitted_scale(values: torch.Tensor, valid: torch.Tensor | None) -> tuple[float, dict[str, Any]]:
    array = values.detach().to(torch.float64).cpu().numpy()
    residual = (array[..., 1:, :] - array[..., :-1, :]) / math.sqrt(2.0)
    if valid is not None:
        mask = valid.detach().cpu().numpy()[..., 1:, :]
        if mask.shape == residual.shape:
            residual = residual[mask]
    estimate, info = fitted_variance_estimate(residual)
    if not math.isfinite(estimate):
        raise ValueError("the frame has no finite neighbour-difference scale for noise injection")
    return float(estimate), info

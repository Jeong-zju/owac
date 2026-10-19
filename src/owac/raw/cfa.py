"""CFA plane geometry, black/white calibration and noise for RAW measurement planes.

This is the native-CFA boundary of ``docs/prompts/DeepSeek_N01_implementation.md``:
a 2x2 colour-filter array takes one measurement per pixel, so a mosaic is
rearranged into four measurement planes that keep their own sampling offsets and
colour names (RGGB/BGGR/GRBG/GBRG).  Planes are sampled separately and mixed only
later, as features; they are never four co-located colour channels of one pixel.
The module owns plane geometry, calibration ``x=(raw-black)/(white-black)``, the
separate missing/bad/saturated masks, and the noise model
``variance=shot*max(x,0)+read_var`` in the unit of ``x``.  It never builds an
RGB/sRGB image and never re-mosaics a rendered image into "RAW".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "CFA_PATTERNS",
    "PLANE_OFFSETS",
    "CfaGeometry",
    "CfaMosaic",
    "CfaPlaneStack",
    "NoiseModel",
    "crop_planes",
    "estimate_noise_model",
    "shift_planes",
    "shot_read_variance",
]

#: Colour of the four positions of one 2x2 cell, indexed by ``2*row + col``.
CFA_PATTERNS: dict[str, tuple[str, str, str, str]] = {
    "RGGB": ("R", "G", "G", "B"),
    "BGGR": ("B", "G", "G", "R"),
    "GRBG": ("G", "R", "B", "G"),
    "GBRG": ("G", "B", "R", "G"),
}

#: Plane ``p`` samples the local offset ``PLANE_OFFSETS[p]`` of every 2x2 cell.
PLANE_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


def _pattern_colors(pattern: str) -> tuple[str, str, str, str]:
    key = str(pattern).upper()
    if key not in CFA_PATTERNS:
        raise ValueError(f"unknown CFA pattern {pattern!r}; expected one of {sorted(CFA_PATTERNS)}")
    return CFA_PATTERNS[key]


def _color_at(pattern: str, row: int, col: int) -> str:
    colors = _pattern_colors(pattern)
    return colors[(int(row) % 2) * 2 + (int(col) % 2)]


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.is_complex():
        raise TypeError(f"{name} must be real-valued, got dtype {tensor.dtype}")
    return tensor.to(torch.float32)


def _as_bool_mask(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(torch.bool)
    return torch.as_tensor(value).to(torch.bool)


@dataclass(frozen=True)
class CfaGeometry:
    """Sampling geometry of a CFA frame that sits inside a larger sensor grid.

    ``origin`` is the ``(row, col)`` of this frame's top-left pixel **in the
    original sensor grid** and ``shape`` is the frame size.  Cropping or shifting
    a frame changes ``origin`` and therefore the colour each of the four local
    planes actually samples; plane order itself stays stable at
    :data:`PLANE_OFFSETS`.
    """

    pattern: str = "RGGB"
    origin: tuple[int, int] = (0, 0)
    shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _pattern_colors(self.pattern)
        row, col = int(self.origin[0]), int(self.origin[1])
        if row < 0 or col < 0:
            raise ValueError(
                f"CFA frame origin must be non-negative in the sensor grid, got {(row, col)}"
            )
        object.__setattr__(self, "origin", (row, col))
        if self.shape is not None:
            height, width = int(self.shape[0]), int(self.shape[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"CFA frame shape must be positive, got {self.shape}")
            object.__setattr__(self, "shape", (height, width))

    @property
    def canonical_pattern(self) -> str:
        return str(self.pattern).upper()

    def local_offsets(self) -> tuple[tuple[int, int], ...]:
        """Plane-local ``(row, col)`` offsets, identical for every frame."""
        return PLANE_OFFSETS

    def global_offsets(self) -> tuple[tuple[int, int], ...]:
        """Plane offsets in the original sensor grid (``origin + local``)."""
        row, col = self.origin
        return tuple((row + dr, col + dc) for dr, dc in PLANE_OFFSETS)

    def colors(self) -> tuple[str, ...]:
        """Colour sampled by each plane under the current phase."""
        return tuple(_color_at(self.pattern, row, col) for row, col in self.global_offsets())

    def cropped(self, top: int, left: int, height: int, width: int) -> CfaGeometry:
        """Geometry of the sub-frame at sensor offset ``(top, left)``."""
        if min(int(top), int(left)) < 0 or height <= 0 or width <= 0:
            raise ValueError(
                f"invalid crop top={top} left={left} height={height} width={width}; sizes "
                "must be positive and offsets non-negative"
            )
        row, col = self.origin
        return CfaGeometry(
            self.pattern, (row + int(top), col + int(left)), (int(height), int(width))
        )

    def shifted(self, dy: int, dx: int) -> CfaGeometry:
        """Geometry after moving the frame origin by ``(dy, dx)`` sensor pixels."""
        row, col = self.origin
        return CfaGeometry(self.pattern, (row + int(dy), col + int(dx)), self.shape)


@dataclass(frozen=True)
class CfaMosaic:
    """One mosaic frame in digital numbers plus its sampling provenance.

    Masks stay separate and optional: ``missing`` means no measurement was
    recorded, ``bad`` a known defective pixel, ``saturated`` a pixel clipped at
    the full-well level.  ``None`` means "not supplied" and is preserved across
    geometry operations instead of being silently turned into all-False.
    """

    values: torch.Tensor
    pattern: str = "RGGB"
    origin: tuple[int, int] = (0, 0)
    missing: torch.Tensor | None = None
    bad: torch.Tensor | None = None
    saturated: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _as_float_tensor(self.values, "mosaic values")
        if values.ndim != 2:
            raise ValueError(
                "a CFA mosaic holds exactly one measurement per pixel; expected an [H, W] "
                f"frame, got shape {tuple(values.shape)}"
            )
        object.__setattr__(self, "values", values)
        CfaGeometry(self.pattern, self.origin, (values.shape[0], values.shape[1]))
        for name in ("missing", "bad", "saturated"):
            mask = getattr(self, name)
            if mask is None:
                continue
            mask = _as_bool_mask(mask)
            if tuple(mask.shape) != tuple(values.shape):
                raise ValueError(
                    f"{name} mask shape {tuple(mask.shape)} does not match mosaic shape "
                    f"{tuple(values.shape)}"
                )
            object.__setattr__(self, name, mask)

    @property
    def height(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    def geometry(self) -> CfaGeometry:
        return CfaGeometry(self.pattern, self.origin, (self.height, self.width))

    def crops(
        self,
        *,
        top: int = 0,
        left: int = 0,
        height: int | None = None,
        width: int | None = None,
        fill: float = math.nan,
    ) -> CfaMosaic:
        """Crop in **sensor** pixels; every parity is exact because ``origin`` moves."""
        top, left = int(top), int(left)
        height = self.height - top if height is None else int(height)
        width = self.width - left if width is None else int(width)
        if top < 0 or left < 0 or height <= 0 or width <= 0:
            raise ValueError(f"invalid crop top={top} left={left} height={height} width={width}")
        if top + height > self.height or left + width > self.width:
            raise ValueError(
                f"crop top={top} left={left} height={height} width={width} leaves the "
                f"{self.height}x{self.width} mosaic"
            )
        window = (slice(top, top + height), slice(left, left + width))
        values = self.values[window].clone()
        missing = None if self.missing is None else self.missing[window].clone()
        if missing is not None and not math.isnan(fill):
            values = torch.where(missing, torch.full_like(values, float(fill)), values)
        metadata = dict(self.metadata)
        metadata["crop"] = {"top": top, "left": left, "height": height, "width": width}
        return CfaMosaic(
            values=values,
            pattern=self.pattern,
            origin=self.geometry().cropped(top, left, height, width).origin,
            missing=missing,
            bad=None if self.bad is None else self.bad[window].clone(),
            saturated=None if self.saturated is None else self.saturated[window].clone(),
            metadata=metadata,
        )

    def shifted(self, *, dy: int, dx: int, fill: float = math.nan) -> CfaMosaic:
        """Shift the sampled window by ``(dy, dx)`` sensor pixels.

        The window moves by ``(dy, dx)`` sensor pixels and keeps its size; the
        new origin is ``origin + (dy, dx)``.  Samples that enter the window
        without having been measured before are filled and marked missing: no
        measurement exists there, so they must not be read as new evidence.  The
        new origin is only valid while it stays inside the sensor grid.
        """
        dy, dx = int(dy), int(dx)
        height, width = self.height, self.width
        values = torch.full((height, width), float(fill), dtype=self.values.dtype)
        missing = torch.ones((height, width), dtype=torch.bool)
        bad = None if self.bad is None else torch.zeros((height, width), dtype=torch.bool)
        saturated = (
            None if self.saturated is None else torch.zeros((height, width), dtype=torch.bool)
        )

        src_r0, src_r1 = max(0, dy), min(height, height + dy)
        src_c0, src_c1 = max(0, dx), min(width, width + dx)
        if src_r0 < src_r1 and src_c0 < src_c1:
            dst = (slice(src_r0 - dy, src_r1 - dy), slice(src_c0 - dx, src_c1 - dx))
            src = (slice(src_r0, src_r1), slice(src_c0, src_c1))
            values[dst] = self.values[src]
            if self.missing is None:
                missing[dst] = False
            else:
                missing[dst] = self.missing[src]
            if bad is not None and self.bad is not None:
                bad[dst] = self.bad[src]
            if saturated is not None and self.saturated is not None:
                saturated[dst] = self.saturated[src]
        metadata = dict(self.metadata)
        metadata["shift"] = {"dy": dy, "dx": dx, "fill": float(fill)}
        return CfaMosaic(
            values=values,
            pattern=self.pattern,
            origin=self.geometry().shifted(dy, dx).origin,
            missing=missing,
            bad=bad,
            saturated=saturated,
            metadata=metadata,
        )

    def split(
        self,
        *,
        black_level: Any,
        white_level: Any,
        saturation_level: Any | None = None,
        fill_value: float = 0.0,
    ) -> CfaPlaneStack:
        """Rearrange the mosaic into four planes and calibrate ``x=(raw-black)/(white-black)``.

        ``black_level``/``white_level`` are scalars or per-plane values in plane
        order :data:`PLANE_OFFSETS`, in digital numbers; both must be finite.
        ``saturation_level`` defaults to ``white_level``; pass ``math.inf`` when
        the full-well level is unknown, which keeps every ``saturated`` flag
        False and records ``saturation_level_available=False`` instead of
        claiming that no pixel is saturated.
        """
        height, width = self.height, self.width
        if height % 2 or width % 2:
            raise ValueError(
                "a 2x2 CFA rearrangement needs an even-sized frame; got "
                f"{height}x{width}. Crop the mosaic to an even size first."
            )
        zeros = torch.zeros_like(self.values)
        raw_planes = torch.stack([self.values[dr::2, dc::2] for dr, dc in PLANE_OFFSETS])
        missing_planes = torch.stack(
            [
                (self.missing if self.missing is not None else zeros)[dr::2, dc::2]
                for dr, dc in PLANE_OFFSETS
            ]
        ).to(torch.bool)
        bad_planes = torch.stack(
            [
                (self.bad if self.bad is not None else zeros)[dr::2, dc::2]
                for dr, dc in PLANE_OFFSETS
            ]
        ).to(torch.bool)

        black = self._per_plane(black_level, "black_level")
        white = self._per_plane(white_level, "white_level")
        if not bool(torch.isfinite(black).all()) or not bool(torch.isfinite(white).all()):
            raise ValueError(
                f"black_level and white_level must be finite digital numbers; got "
                f"black={black.tolist()} white={white.tolist()}. An unknown full-well level "
                "belongs in saturation_level=math.inf, not in white_level."
            )
        span = white - black
        if bool((span.abs() < 1e-12).any()):
            raise ValueError(
                f"white_level must differ from black_level per plane; got black="
                f"{black.tolist()} white={white.tolist()}"
            )
        values = (raw_planes - black[:, None, None]) / span[:, None, None]

        if saturation_level is None:
            saturated = raw_planes >= white[:, None, None]
            saturation_rule = "raw >= white_level"
            saturation_available = bool(torch.isfinite(white).all())
        else:
            level = self._per_plane(saturation_level, "saturation_level")
            saturated = raw_planes >= level[:, None, None]
            saturation_rule = "raw >= declared saturation_level"
            saturation_available = bool(torch.isfinite(level).all())
        if self.saturated is not None:
            saturated = saturated | torch.stack(
                [self.saturated[dr::2, dc::2] for dr, dc in PLANE_OFFSETS]
            ).to(torch.bool)
            saturation_rule = f"{saturation_rule} | declared saturated mask"
            saturation_available = True
        saturated = saturated.to(torch.bool)

        drop = missing_planes | bad_planes
        values = torch.where(drop, torch.full_like(values, float(fill_value)), values)
        valid = ~(drop | saturated)
        geometry = self.geometry()
        metadata = dict(self.metadata)
        metadata.update(
            {
                "pattern": geometry.canonical_pattern,
                "plane_colors": list(geometry.colors()),
                "plane_local_offsets": [list(offset) for offset in geometry.local_offsets()],
                "plane_global_offsets": [list(offset) for offset in geometry.global_offsets()],
                "frame_origin": list(geometry.origin),
                "co_located": False,
                "co_located_note": (
                    "the four planes are four sub-sampled measurement grids, not four "
                    "co-located colour channels of one pixel"
                ),
                "calibration": "x=(raw-black)/(white-black)",
                "saturation_rule": saturation_rule,
                "saturation_level_available": saturation_available,
                "fill_value": float(fill_value),
                "masked_fraction": float(drop.float().mean().item()),
                "saturated_fraction": float(saturated.float().mean().item()),
            }
        )
        return CfaPlaneStack(
            values=values,
            valid=valid,
            bad=bad_planes,
            saturated=saturated,
            missing=missing_planes,
            geometry=geometry,
            black_level=black,
            white_level=white,
            metadata=metadata,
        )

    def _per_plane(self, level: Any, name: str) -> torch.Tensor:
        tensor = _as_float_tensor(level, name).reshape(-1)
        if tensor.numel() == 1:
            return tensor.expand(4).clone()
        if tensor.numel() != 4:
            raise ValueError(
                f"{name} must be a scalar or one value per CFA plane (4), got {tensor.numel()}"
            )
        return tensor.clone()


@dataclass(frozen=True)
class CfaPlaneStack:
    """Four calibrated CFA measurement planes with their real sampling offsets."""

    values: torch.Tensor
    valid: torch.Tensor
    bad: torch.Tensor
    saturated: torch.Tensor
    missing: torch.Tensor
    geometry: CfaGeometry
    black_level: torch.Tensor
    white_level: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or int(self.values.shape[0]) != 4:
            raise ValueError(
                "a 2x2 CFA plane stack has exactly four [4, h, w] planes; got shape "
                f"{tuple(self.values.shape)}"
            )
        for name in ("valid", "bad", "saturated", "missing"):
            mask = getattr(self, name)
            if tuple(mask.shape) != tuple(self.values.shape):
                raise ValueError(
                    f"{name} shape {tuple(mask.shape)} does not match plane values "
                    f"{tuple(self.values.shape)}"
                )

    @property
    def plane_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def colors(self) -> tuple[str, ...]:
        return self.geometry.colors()

    @property
    def sample_xy(self) -> torch.Tensor:
        """``[4, 2]`` int64 ``(row, col)`` offsets inside the current frame."""
        return torch.tensor(self.geometry.local_offsets(), dtype=torch.int64)

    @property
    def global_sample_xy(self) -> torch.Tensor:
        """``[4, 2]`` int64 offsets in the original sensor grid."""
        return torch.tensor(self.geometry.global_offsets(), dtype=torch.int64)

    @property
    def co_located(self) -> bool:
        """Always ``False``: every plane samples a different pixel."""
        return False

    def plane_shape(self) -> tuple[int, int]:
        return int(self.values.shape[1]), int(self.values.shape[2])


def crop_planes(
    stack: CfaPlaneStack,
    *,
    top: int,
    left: int,
    height: int | None = None,
    width: int | None = None,
    fill_value: float | None = None,
) -> CfaPlaneStack:
    """Crop in **plane** units (one plane pixel spans two sensor pixels).

    All coordinate records move together: ``origin`` shifts by ``2*(top, left)``
    sensor pixels, plane-local offsets stay fixed and the plane colours are
    recomputed from the new phase.  Passing ``fill_value`` replaces samples that
    are not valid (they stay masked out either way).
    """
    plane_height, plane_width = stack.plane_shape()
    top, left = int(top), int(left)
    if top < 0 or left < 0:
        raise ValueError(f"crop top/left must be non-negative, got top={top} left={left}")
    height = plane_height - top if height is None else int(height)
    width = plane_width - left if width is None else int(width)
    if height <= 0 or width <= 0 or top + height > plane_height or left + width > plane_width:
        raise ValueError(
            f"crop top={top} left={left} height={height} width={width} leaves the "
            f"{plane_height}x{plane_width} plane grid"
        )
    rows = slice(top, top + height)
    cols = slice(left, left + width)
    values = stack.values[:, rows, cols].clone()
    valid = stack.valid[:, rows, cols].clone()
    if fill_value is not None:
        values = torch.where(valid, values, torch.full_like(values, float(fill_value)))
    geometry = CfaGeometry(
        stack.geometry.pattern,
        (stack.geometry.origin[0] + 2 * top, stack.geometry.origin[1] + 2 * left),
        (2 * height, 2 * width),
    )
    metadata = dict(stack.metadata)
    metadata["plane_crop"] = {
        "top": top,
        "left": left,
        "height": height,
        "width": width,
        "units": "plane pixels",
    }
    geometry_metadata = dict(metadata)
    geometry_metadata["plane_colors"] = list(geometry.colors())
    geometry_metadata["plane_global_offsets"] = [list(o) for o in geometry.global_offsets()]
    geometry_metadata["frame_origin"] = list(geometry.origin)
    return CfaPlaneStack(
        values=values,
        valid=valid,
        bad=stack.bad[:, rows, cols].clone(),
        saturated=stack.saturated[:, rows, cols].clone(),
        missing=stack.missing[:, rows, cols].clone(),
        geometry=geometry,
        black_level=stack.black_level.clone(),
        white_level=stack.white_level.clone(),
        metadata=geometry_metadata,
    )


def shift_planes(
    stack: CfaPlaneStack,
    *,
    dy: int,
    dx: int,
    fill_value: float = 0.0,
) -> CfaPlaneStack:
    """Shift the plane window by ``(dy, dx)`` **plane** pixels (even sensor steps).

    The frame keeps its plane size, the origin moves by ``2*(dy, dx)`` sensor
    pixels and colours follow the new phase: content previously at plane
    ``(r, c)`` sits at plane ``(r - dy, c - dx)`` afterwards, so the plane that
    becomes the frame's top-left really samples the sensor pixel the new origin
    names.  Samples entering the window without an earlier measurement are
    marked missing and filled with ``fill_value``; they are not measurements and
    must stay masked in every downstream statistic.
    """
    dy, dx = int(dy), int(dx)
    plane_height, plane_width = stack.plane_shape()
    values = torch.full_like(stack.values, float(fill_value))
    valid = torch.zeros_like(stack.valid)
    missing = torch.ones_like(stack.missing)
    bad = torch.zeros_like(stack.bad)
    saturated = torch.zeros_like(stack.saturated)

    src_r0, src_r1 = max(0, dy), min(plane_height, plane_height + dy)
    src_c0, src_c1 = max(0, dx), min(plane_width, plane_width + dx)
    if src_r0 < src_r1 and src_c0 < src_c1:
        dst = (slice(src_r0 - dy, src_r1 - dy), slice(src_c0 - dx, src_c1 - dx))
        src = (slice(src_r0, src_r1), slice(src_c0, src_c1))
        values[:, dst[0], dst[1]] = stack.values[:, src[0], src[1]]
        valid[:, dst[0], dst[1]] = stack.valid[:, src[0], src[1]]
        missing[:, dst[0], dst[1]] = stack.missing[:, src[0], src[1]]
        bad[:, dst[0], dst[1]] = stack.bad[:, src[0], src[1]]
        saturated[:, dst[0], dst[1]] = stack.saturated[:, src[0], src[1]]
    geometry = CfaGeometry(
        stack.geometry.pattern,
        (stack.geometry.origin[0] + 2 * dy, stack.geometry.origin[1] + 2 * dx),
        (2 * plane_height, 2 * plane_width),
    )
    metadata = dict(stack.metadata)
    metadata["plane_shift"] = {
        "dy": dy,
        "dx": dx,
        "fill_value": float(fill_value),
        "units": "plane pixels",
    }
    metadata["plane_colors"] = list(geometry.colors())
    metadata["plane_global_offsets"] = [list(o) for o in geometry.global_offsets()]
    metadata["frame_origin"] = list(geometry.origin)
    return CfaPlaneStack(
        values=values,
        valid=valid,
        bad=bad,
        saturated=saturated,
        missing=missing,
        geometry=geometry,
        black_level=stack.black_level.clone(),
        white_level=stack.white_level.clone(),
        metadata=metadata,
    )


def shot_read_variance(x: Any, *, shot: float, read_var: float) -> torch.Tensor:
    """``variance = shot*max(x,0) + read_var`` in the numeric unit of ``x``.

    The formula is only meaningful when ``shot``/``read_var`` were measured or
    fitted in the **same** unit as ``x``; this function never rescales them.
    """
    shot = float(shot)
    read_var = float(read_var)
    if shot < 0.0 or read_var < 0.0:
        raise ValueError(f"shot and read_var must be non-negative, got {shot} and {read_var}")
    values = _as_float_tensor(x, "x")
    return shot * torch.clamp(values, min=0.0) + read_var


@dataclass(frozen=True)
class NoiseModel:
    """Shot/read variance model in one declared numeric unit.

    ``fitted=True`` or ``estimator=True`` marks a scale model obtained from the
    data itself: it is an estimate under an independent-noise, locally-smooth
    signal assumption, not a calibrated sensor measurement.
    """

    shot: float
    read_var: float
    units: str = "calibrated_digital_number"
    fitted: bool = False
    estimator: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def variance(self, x: Any) -> torch.Tensor:
        return shot_read_variance(x, shot=self.shot, read_var=self.read_var)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shot": float(self.shot),
            "read_var": float(self.read_var),
            "units": self.units,
            "fitted": bool(self.fitted),
            "estimator": bool(self.estimator),
            "evidence": dict(self.evidence),
        }


def estimate_noise_model(
    stack: CfaPlaneStack,
    *,
    bins: int = 4,
    min_samples: int = 64,
) -> NoiseModel:
    """Fit ``variance = shot*x + read_var`` from same-plane neighbour differences.

    Neighbouring pixels of one plane satisfy ``E[(x_i - x_j)^2] = 2*variance``
    when the noise is independent and the signal is locally smooth.  Pairs are
    binned by intensity, a variance is estimated per bin and a two-parameter
    least-squares fit is returned **as an estimator** (``estimator=True``) with
    its bin evidence.  Missing, bad, saturated and filled samples are excluded.
    """
    if int(bins) < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    values = stack.values
    valid = stack.valid
    pair_ok = valid[:, :, :-1] & valid[:, :, 1:]
    pair_count = int(pair_ok.sum().item())
    if pair_count < int(min_samples):
        raise ValueError(
            "not enough valid same-plane neighbour pairs to fit a noise model: "
            f"{pair_count} < min_samples={int(min_samples)}"
        )
    left = values[:, :, :-1][pair_ok]
    right = values[:, :, 1:][pair_ok]
    diff = (right - left).to(torch.float64)
    mid = (0.5 * (right + left)).to(torch.float64)
    squared = diff * diff

    order = torch.argsort(mid)
    chunk = max(1, int(math.ceil(order.numel() / int(bins))))
    rows: list[dict[str, float]] = []
    for start in range(0, order.numel(), chunk):
        index = order[start : start + chunk]
        if index.numel() < 4:
            continue
        rows.append(
            {
                "x": float(mid[index].mean().item()),
                "variance": float(0.5 * squared[index].mean().item()),
                "pairs": float(index.numel()),
            }
        )
    if len(rows) < 2:
        raise ValueError("noise fit needs at least two usable intensity bins")

    sum_x = sum(row["x"] for row in rows)
    sum_v = sum(row["variance"] for row in rows)
    sum_xx = sum(row["x"] * row["x"] for row in rows)
    sum_xv = sum(row["x"] * row["variance"] for row in rows)
    count = float(len(rows))
    determinant = count * sum_xx - sum_x * sum_x
    if abs(determinant) < 1e-18:
        shot = 0.0
        read_var = max(sum_v / count, 0.0)
    else:
        shot = (count * sum_xv - sum_x * sum_v) / determinant
        read_var = (sum_v - shot * sum_x) / count
    shot = max(float(shot), 0.0)
    read_var = max(float(read_var), 0.0)
    return NoiseModel(
        shot=shot,
        read_var=read_var,
        fitted=True,
        estimator=True,
        evidence={
            "method": "same-plane neighbour differences, binned by intensity",
            "assumption": "independent noise and a locally smooth signal inside each bin",
            "cfa_pattern": stack.geometry.canonical_pattern,
            "plane_colors": list(stack.colors),
            "bins": rows,
            "min_samples": int(min_samples),
            "pairs_used": pair_count,
        },
    )

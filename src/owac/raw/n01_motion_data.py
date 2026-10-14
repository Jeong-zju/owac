"""Synthetic RAW motion pairs sampled from an analytic layered radiance field.

Every pair is produced by *sampling a known continuous radiance field*, so the
geometric truth is analytic and the model-visible input can be separated from it
completely:

* :class:`MotionDataGenerator` returns a :class:`SupervisedPair` whose
  ``visible`` half holds only the two :class:`RawSensorFrame` exposures plus
  metadata a real pipeline has (identities, timestamps, CFA pattern, packed-grid
  convention, exposure value *or* an explicit unknown, noise block id,
  calibration version).  The ``supervision`` half holds the analytic flow,
  validity and occlusion labels and is never reachable from ``visible``.
* ``RawSensorFrame`` is the single shared type from
  ``owac.backends.raw_correspondence`` (motion-core); this module defines no
  second frame type.

Conventions
-----------

* ``values`` of a CFA frame are ``[4, h, w]`` packed planes with ``h = H/2``,
  ``w = W/2`` of the full sensor grid; channel ``k`` samples phase
  ``channel_phase[k]`` and its packed ``(i, j)`` maps to the full sensor site
  ``(y, x) = (y0 + 2i + dy_k, x0 + 2j + dx_k)`` with the *relative* offset
  ``d_k = off(phase_k) - off(phase_0)`` and ``off(p) = (p // 2, p % 2)``.  The
  frame origin parity must match channel 0's phase.  One packed step therefore
  spans two sensor pixels.
* ``sensor_origin_xy`` follows motion-core's landing contract: it carries
  ``(y0, x0)``, the full sensor coordinate of ``values[0, 0, 0]``, despite the
  field name.
* ``rawvla_rgb3`` frames are ``[3, H, W]`` collocated HDR-derived pseudo-RAW
  with ``channel_phase = ()`` and ``cfa_pattern = "rgb3"``: no fourth
  measurement and no CFA phase is fabricated, and the material class is
  recorded separately from a real/synthetic CFA sensor.
* The supervision flow is the **pull** mapping required by the contract: a
  reference-frame position (the current frame) to the historical source frame,
  in full sensor ``(x, y)`` pixels.  :func:`packed_pull_targets` converts it to
  each channel's packed coordinates in the source frame.

Determinism
-----------

Namespaces are ``n01-full-v1/<split>/<case>/<sequence>``; the data seed of a
namespace is a SHA-256 digest of ``<data_seed_base>|<namespace>`` reduced modulo
2**63, and every per-frame sub-seed appends ``/scene/<frame_id>`` or
``/noise/<frame_id>``.  Frame ``k`` is a pure function of its own sub-seed, so
truncating a sequence never changes an earlier frame (no future-frame leakage)
and the same namespace reproduces bit-identical values in a fresh process.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from owac.backends.raw_correspondence import RawSensorFrame
from owac.frameworks.n01_full.config import CFA_PATTERNS, INPUT_MODES
from owac.raw.cfa import CFA_PATTERNS as CFA_PATTERN_COLORS
from owac.raw.cfa import PLANE_OFFSETS
from owac.raw.n01_noise import (
    FITTING_SPLITS,
    NoiseCalibration,
    add_sensor_noise,
    draw_block_error,
)

__all__ = [
    "CASE_KINDS",
    "CFA_MATERIAL_CLASS",
    "FIELD_KINDS",
    "MATERIAL_CLASSES",
    "MOTION_SCHEMA",
    "NAMESPACE_ROOT",
    "PSEUDO_RAW_MATERIAL_CLASS",
    "REGIONS",
    "SUBPIXEL_AXES",
    "SUBPIXEL_TIERS",
    "VISIBILITY_KINDS",
    "BlobField",
    "FrameGeometry",
    "Layer",
    "LayerPair",
    "LocalWarp",
    "MotionDataGenerator",
    "MotionSupervision",
    "P2MotionNoiseConfig",
    "ProvenancePolicy",
    "PairSolve",
    "RadianceField",
    "SceneFrame",
    "SceneRecipe",
    "SeedRule",
    "SimilarityPose",
    "SinusoidField",
    "StepField",
    "SupervisedPair",
    "VisiblePair",
    "load_p2_motion_noise_config",
    "packed_pull_targets",
    "sensor_xy_grid",
    "solve_pull",
    "subpixel_tokens",
]

MOTION_SCHEMA = "owac.n01_full.p2_motion_noise.v1"
NAMESPACE_ROOT = "n01-full-v1"
REGIONS = ("train", "calibration", "validation", "test")

#: Correspondence labels of one reference sample.  ``valid`` is a real
#: correspondence; ``occluded_in_source`` means the pulled source position is
#: covered by a *different, nearer* layer, so the reference content is not
#: visible there (the reference region is newly exposed and has no flow target);
#: ``out_of_bounds`` means the pull left the source frame's sampled extent.
#:
#: The mirror case -- the reference sample is covered by a nearer layer whose
#: pull is *uncovered* in the source -- cannot occur while the source layer is
#: rigid: the pull is defined as the source coordinate whose chart equals the
#: reference chart, so the source's cover predicate takes exactly the same value
#: as the reference's.  :func:`solve_pull` asserts that invariant instead of
#: inventing a label for an impossible event.
VISIBILITY_KINDS = ("valid", "occluded_in_source", "out_of_bounds")

CASE_KINDS = (
    "translation_subpixel",
    "rotation",
    "layered_occlusion",
    "local_nonrigid",
    "repeated_texture",
    "exposure_gain",
    "motion_blur",
    "combined_hard",
)

#: Sub-pixel displacement tiers (both signs, both axes) that must exist in truth.
SUBPIXEL_TIERS = (0.25, 0.5, 0.75)
SUBPIXEL_AXES = ("x", "y")
FIELD_KINDS = ("sinusoid", "blobs", "steps")
LAYER_SHAPES = ("fullscreen", "disk")
RGB3_PATTERN = "rgb3"

#: Material classes stay distinguishable: a synthetic CFA sensor is not the
#: HDR-derived three-channel pseudo-RAW, and neither is an inverted RGB capture.
CFA_MATERIAL_CLASS = "synthetic_cfa"
PSEUDO_RAW_MATERIAL_CLASS = "rawvla_rgb3_hdr_derived"

#: Every material class the config must describe, so none is silently merged into another.
MATERIAL_CLASSES = (
    CFA_MATERIAL_CLASS,
    PSEUDO_RAW_MATERIAL_CLASS,
    "inverted_rgb_capture",
)


def _reject_unknown_keys(data: Mapping[str, Any], allowed: Sequence[str], what: str) -> None:
    """A declared JSON vocabulary is closed: an unknown field is an error, not a comment."""
    extra = sorted(set(data) - set(allowed))
    if extra:
        raise ValueError(
            f"{what} carries unknown fields {extra}; the declared vocabulary is {sorted(allowed)}"
        )


def subpixel_tokens() -> tuple[str, ...]:
    """Sequence tokens of the sub-pixel truth grid, e.g. ``"x+0.25"``, ``"y-0.75"``."""
    return tuple(
        f"{axis}{sign * tier:+.2f}"
        for axis in SUBPIXEL_AXES
        for tier in SUBPIXEL_TIERS
        for sign in (1, -1)
    )


def parse_subpixel_token(token: str) -> tuple[float, float]:
    """Displacement ``(dx, dy)`` in *sensor* pixels encoded by a sub-pixel token."""
    token = str(token)
    if len(token) < 3 or token[0] not in SUBPIXEL_AXES:
        raise ValueError(f"not a sub-pixel displacement token: {token!r}")
    tier = float(token[1:])
    if abs(tier) not in SUBPIXEL_TIERS:
        raise ValueError(f"tier {tier!r} is not one of {SUBPIXEL_TIERS} (up to sign)")
    return (tier, 0.0) if token[0] == "x" else (0.0, tier)


@dataclass(frozen=True)
class FrameGeometry:
    """The sampling grid of one frame family: full sensor size, CFA phase and origin.

    ``height``/``width`` are the **full sensor** grid; CFA planes are packed by
    two in each axis.  ``channel_phase[k]`` is the CFA phase of channel ``k``
    and must be a permutation of ``(0, 1, 2, 3)``; channel 0 defines the origin
    parity, so the relative offset of channel ``k`` is
    ``off(phase_k) - off(phase_0)``.
    """

    height: int
    width: int
    input_mode: str = "synthetic_cfa"
    cfa_pattern: str = "RGGB"
    origin_yx: tuple[int, int] = (0, 0)
    channel_phase: tuple[int, ...] = (0, 1, 2, 3)
    value_units: str = "raw_dn_12bit"

    def __post_init__(self) -> None:
        for name in ("height", "width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"FrameGeometry.{name} must be a positive integer, got {value!r}")
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input_mode {self.input_mode!r}; expected {INPUT_MODES}")
        if not isinstance(self.value_units, str) or not self.value_units:
            raise ValueError("FrameGeometry.value_units must name the RAW unit")
        row, col = self.origin_yx
        if isinstance(row, bool) or isinstance(col, bool) or int(row) < 0 or int(col) < 0:
            raise ValueError(
                f"FrameGeometry.origin_yx must be non-negative, got {self.origin_yx!r}"
            )
        object.__setattr__(self, "origin_yx", (int(row), int(col)))
        if self.is_cfa:
            if self.height % 2 or self.width % 2:
                raise ValueError(
                    "a CFA sensor grid must be even in both axes so the packed planes tile it; "
                    f"got {(self.height, self.width)}"
                )
            if self.cfa_pattern not in CFA_PATTERNS:
                raise ValueError(
                    f"unknown CFA pattern {self.cfa_pattern!r}; expected {tuple(CFA_PATTERNS)}"
                )
            phases = tuple(self.channel_phase)
            if len(phases) != 4 or sorted(phases) != [0, 1, 2, 3]:
                raise ValueError(
                    "channel_phase must be a permutation of (0, 1, 2, 3), got "
                    f"{self.channel_phase!r}"
                )
            object.__setattr__(self, "channel_phase", phases)
            first = PLANE_OFFSETS[phases[0]]
            if (self.origin_yx[0] % 2, self.origin_yx[1] % 2) != first:
                raise ValueError(
                    f"origin {self.origin_yx!r} has parity "
                    f"{(self.origin_yx[0] % 2, self.origin_yx[1] % 2)!r} but channel 0 phase "
                    f"{phases[0]} samples offset {first!r}; the packed grid would not cover "
                    "the declared origin"
                )
        else:
            if self.cfa_pattern != RGB3_PATTERN:
                raise ValueError(
                    "rawvla_rgb3 carries no CFA phase; use cfa_pattern='rgb3', got "
                    f"{self.cfa_pattern!r}"
                )
            if tuple(self.channel_phase) != ():
                raise ValueError(
                    "rawvla_rgb3 has no CFA phase; channel_phase must be empty, got "
                    f"{self.channel_phase!r}.  Do not fabricate a fourth measurement."
                )

    @property
    def is_cfa(self) -> bool:
        return self.input_mode in ("bayer_cfa", "synthetic_cfa")

    @property
    def channels(self) -> int:
        return 4 if self.is_cfa else 3

    @property
    def packed_shape(self) -> tuple[int, int]:
        """``[C, h, w]`` plane shape: packed by two for CFA, full resolution for rgb3."""
        return (self.height // 2, self.width // 2) if self.is_cfa else (self.height, self.width)

    @property
    def sensor_shape(self) -> tuple[int, int]:
        """Full sensor grid covered by all channels together."""
        return (self.height, self.width)

    @property
    def channel_offsets(self) -> tuple[tuple[int, int], ...]:
        """Relative sensor offset ``d_k`` of each channel against channel 0."""
        if not self.is_cfa:
            return tuple((0, 0) for _ in range(self.channels))
        first = PLANE_OFFSETS[self.channel_phase[0]]
        return tuple(
            (
                PLANE_OFFSETS[phase][0] - first[0],
                PLANE_OFFSETS[phase][1] - first[1],
            )
            for phase in self.channel_phase
        )

    @property
    def grid_origin_yx(self) -> tuple[int, int]:
        """Sensor coordinate of the supervision grid's ``(0, 0)`` cell."""
        offsets = self.channel_offsets
        return (
            self.origin_yx[0] + min(item[0] for item in offsets),
            self.origin_yx[1] + min(item[1] for item in offsets),
        )

    def sample_sensor_yx(self, channel: int, row: int, col: int) -> tuple[int, int]:
        """Full sensor ``(y, x)`` of packed ``(row, col)`` in ``channel``."""
        if not 0 <= int(channel) < self.channels:
            raise ValueError(f"channel {channel!r} outside 0..{self.channels - 1}")
        rows, cols = self.packed_shape
        if not 0 <= int(row) < rows or not 0 <= int(col) < cols:
            raise ValueError(f"packed index ({row}, {col}) outside {self.packed_shape}")
        offset = self.channel_offsets[int(channel)]
        if self.is_cfa:
            return (
                self.origin_yx[0] + 2 * int(row) + offset[0],
                self.origin_yx[1] + 2 * int(col) + offset[1],
            )
        return (self.origin_yx[0] + int(row), self.origin_yx[1] + int(col))

    def grid_indices_yx(self, channel: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rows, cols)`` of one channel's packed samples inside the supervision grid."""
        rows, cols = self.packed_shape
        origin = self.grid_origin_yx
        offset = self.channel_offsets[int(channel)]
        row_index = torch.arange(rows, dtype=torch.int64).reshape(rows, 1) * 2 + (
            self.origin_yx[0] + offset[0] - origin[0]
        )
        col_index = torch.arange(cols, dtype=torch.int64).reshape(1, cols) * 2 + (
            self.origin_yx[1] + offset[1] - origin[1]
        )
        if not self.is_cfa:
            row_index = torch.arange(rows, dtype=torch.int64).reshape(rows, 1)
            col_index = torch.arange(cols, dtype=torch.int64).reshape(1, cols)
        return row_index, col_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "input_mode": self.input_mode,
            "cfa_pattern": self.cfa_pattern,
            "origin_yx": list(self.origin_yx),
            "channel_phase": list(self.channel_phase),
            "value_units": self.value_units,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameGeometry:
        _reject_unknown_keys(
            data,
            (
                "height",
                "width",
                "input_mode",
                "cfa_pattern",
                "origin_yx",
                "channel_phase",
                "value_units",
            ),
            "FrameGeometry",
        )
        return cls(
            height=int(data["height"]),
            width=int(data["width"]),
            input_mode=str(data.get("input_mode", "synthetic_cfa")),
            cfa_pattern=str(data.get("cfa_pattern", "RGGB")),
            origin_yx=tuple(int(item) for item in data.get("origin_yx", (0, 0))),
            channel_phase=tuple(int(item) for item in data.get("channel_phase", (0, 1, 2, 3))),
            value_units=str(data.get("value_units", "raw_dn_12bit")),
        )


def sensor_xy_grid(geometry: FrameGeometry, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``[H, W, 2]`` full sensor ``(x, y)`` coordinates of the supervision grid."""
    height, width = geometry.sensor_shape
    origin_y, origin_x = geometry.grid_origin_yx
    rows = torch.arange(height, dtype=dtype).reshape(height, 1) + origin_y
    cols = torch.arange(width, dtype=dtype).reshape(1, width) + origin_x
    x = cols.expand(height, width)
    y = rows.expand(height, width)
    return torch.stack([x, y], dim=-1)


@dataclass(frozen=True)
class SimilarityPose:
    """Sensor pixels to scene units: ``p = scale * R(theta) (x - c) + c + t``.

    ``x`` is the ``(x, y)`` sensor pixel of a sample and ``p`` the scene
    coordinate its radiance is read from.  The inverse is closed form, which is
    what makes the pull mapping of a rigid layer analytic.
    """

    scale: float = 1.0
    theta_deg: float = 0.0
    center_xy: tuple[float, float] = (0.0, 0.0)
    translation_xy: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError(
                f"SimilarityPose.scale must be finite and positive, got {self.scale!r}"
            )
        for name in ("theta_deg",):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"SimilarityPose.{name} must be finite")
        for name in ("center_xy", "translation_xy"):
            item = tuple(float(value) for value in getattr(self, name))
            if len(item) != 2 or not all(math.isfinite(value) for value in item):
                raise ValueError(f"SimilarityPose.{name} must be two finite numbers")
            object.__setattr__(self, name, item)

    @staticmethod
    def _rotate(xy: torch.Tensor, theta_deg: float) -> torch.Tensor:
        theta = math.radians(float(theta_deg))
        cos, sin = math.cos(theta), math.sin(theta)
        x, y = xy[..., 0], xy[..., 1]
        return torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)

    def apply(self, xy: torch.Tensor) -> torch.Tensor:
        xy = torch.as_tensor(xy, dtype=torch.float64)
        center = torch.tensor(self.center_xy, dtype=torch.float64)
        translation = torch.tensor(self.translation_xy, dtype=torch.float64)
        return float(self.scale) * self._rotate(xy - center, self.theta_deg) + center + translation

    def inverse(self, point: torch.Tensor) -> torch.Tensor:
        point = torch.as_tensor(point, dtype=torch.float64)
        center = torch.tensor(self.center_xy, dtype=torch.float64)
        translation = torch.tensor(self.translation_xy, dtype=torch.float64)
        return (
            self._rotate((point - center - translation) / float(self.scale), -self.theta_deg)
            + center
        )


@dataclass(frozen=True)
class LocalWarp:
    """Local non-rigid displacement, analytic and localised in scene space.

    ``w(p) = amplitude * exp(-|p - center|^2 / (2 sigma^2)) * direction``.  It is
    applied to the **reference** (current) frame only, so the source pull stays a
    closed-form inverse of a rigid pose while the current frame is genuinely
    deformed relative to the past.
    """

    amplitude: float
    sigma: float
    center_xy: tuple[float, float] = (0.0, 0.0)
    direction_xy: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.amplitude)):
            raise ValueError("LocalWarp.amplitude must be finite")
        if not math.isfinite(float(self.sigma)) or float(self.sigma) <= 0.0:
            raise ValueError(f"LocalWarp.sigma must be finite and positive, got {self.sigma!r}")
        for name in ("center_xy", "direction_xy"):
            item = tuple(float(value) for value in getattr(self, name))
            if len(item) != 2 or not all(math.isfinite(value) for value in item):
                raise ValueError(f"LocalWarp.{name} must be two finite numbers")
            object.__setattr__(self, name, item)

    def displacement(self, point: torch.Tensor) -> torch.Tensor:
        point = torch.as_tensor(point, dtype=torch.float64)
        center = torch.tensor(self.center_xy, dtype=torch.float64)
        direction = torch.tensor(self.direction_xy, dtype=torch.float64)
        offset = point - center
        weight = torch.exp(-(offset * offset).sum(dim=-1) / (2.0 * float(self.sigma) ** 2))
        return weight.unsqueeze(-1) * direction * float(self.amplitude)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LocalWarp:
        _reject_unknown_keys(data, ("amplitude", "sigma", "center_xy", "direction_xy"), "LocalWarp")
        return cls(
            amplitude=float(data["amplitude"]),
            sigma=float(data["sigma"]),
            center_xy=tuple(float(item) for item in data.get("center_xy", (0.0, 0.0))),
            direction_xy=tuple(float(item) for item in data.get("direction_xy", (1.0, 0.0))),
        )


class RadianceField:
    """Continuous radiance ``[..., 3]`` in linear scene units at scene ``(x, y)``."""

    def sample(self, xy: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class SinusoidField(RadianceField):
    """Periodic smooth field; its period is the repeated-texture ambiguity scale."""

    period_xy: tuple[float, float] = (8.0, 8.0)
    phase_rgb: tuple[float, float, float] = (0.0, 0.4, 0.8)

    def __post_init__(self) -> None:
        for value in self.period_xy:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"SinusoidField.period_xy must be positive, got {self.period_xy!r}"
                )

    def sample(self, xy: torch.Tensor) -> torch.Tensor:
        xy = torch.as_tensor(xy, dtype=torch.float64)
        period = torch.tensor(self.period_xy, dtype=torch.float64)
        angle = 2.0 * math.pi * xy / period
        base = torch.sin(angle[..., 0]) * torch.sin(angle[..., 1])
        phases = torch.tensor(self.phase_rgb, dtype=torch.float64)
        return 0.5 + 0.5 * torch.sin(base.unsqueeze(-1) * math.pi + phases)

    @property
    def ambiguity_period_xy(self) -> tuple[float, float]:
        return (float(self.period_xy[0]), float(self.period_xy[1]))


@dataclass(frozen=True)
class BlobField(RadianceField):
    """Smooth non-periodic structure: a sum of Gaussian blobs of different colours."""

    centers_xy: tuple[tuple[float, float], ...]
    sigma: float = 4.0
    colors_rgb: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.centers_xy:
            raise ValueError("BlobField needs at least one blob centre")
        if not math.isfinite(float(self.sigma)) or float(self.sigma) <= 0.0:
            raise ValueError(f"BlobField.sigma must be positive, got {self.sigma!r}")
        colors = self.colors_rgb or tuple(
            (0.35 + 0.2 * index, 0.3, 0.8 - 0.15 * index) for index in range(len(self.centers_xy))
        )
        if len(colors) != len(self.centers_xy):
            raise ValueError("BlobField needs one colour per centre")
        object.__setattr__(self, "colors_rgb", tuple(tuple(float(v) for v in c) for c in colors))

    def sample(self, xy: torch.Tensor) -> torch.Tensor:
        xy = torch.as_tensor(xy, dtype=torch.float64)
        out = torch.zeros(xy.shape[:-1] + (3,), dtype=torch.float64)
        for center, color in zip(self.centers_xy, self.colors_rgb, strict=True):
            delta = xy - torch.tensor(center, dtype=torch.float64)
            weight = torch.exp(-(delta * delta).sum(dim=-1) / (2.0 * float(self.sigma) ** 2))
            out = out + weight.unsqueeze(-1) * torch.tensor(color, dtype=torch.float64)
        return out


@dataclass(frozen=True)
class StepField(RadianceField):
    """Sharp checkerboard: strong edges, where linearised flow propagation breaks."""

    period_xy: tuple[float, float] = (10.0, 10.0)
    levels_rgb: tuple[float, float, float] = (0.15, 0.55, 0.95)

    def __post_init__(self) -> None:
        for value in self.period_xy:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"StepField.period_xy must be positive, got {self.period_xy!r}")

    def sample(self, xy: torch.Tensor) -> torch.Tensor:
        xy = torch.as_tensor(xy, dtype=torch.float64)
        period = torch.tensor(self.period_xy, dtype=torch.float64)
        angle = 2.0 * math.pi * xy / period
        cell = torch.sign(torch.sin(angle[..., 0])) * torch.sign(torch.sin(angle[..., 1]))
        levels = torch.tensor(self.levels_rgb, dtype=torch.float64)
        return torch.where(cell.unsqueeze(-1) > 0, levels, 1.0 - levels)


@dataclass(frozen=True)
class Layer:
    """One scene layer of one frame: a field plus the pose of *this* frame.

    ``shape='disk'`` builds an object that occludes whatever is below it;
    ``occluder`` marks it as the occluding layer so the pair solver can label
    which side of an occlusion boundary a mismatch came from.
    """

    layer_id: str
    field: RadianceField
    pose: SimilarityPose
    shape: str = "fullscreen"
    radius: float = 1.0
    occluder: bool = False
    warp: LocalWarp | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise ValueError("Layer.layer_id must be a non-empty identity")
        if self.shape not in LAYER_SHAPES:
            raise ValueError(f"unknown layer shape {self.shape!r}; expected {LAYER_SHAPES}")
        if self.shape == "disk" and not (
            math.isfinite(float(self.radius)) and float(self.radius) > 0
        ):
            raise ValueError(f"a disk layer needs a positive radius, got {self.radius!r}")
        if self.occluder and self.shape != "disk":
            raise ValueError("only a disk layer can be the occluding layer in this version")

    def chart_reference(self, xy_sensor: torch.Tensor) -> torch.Tensor:
        """Scene/object coordinates of the reference frame's samples."""
        point = self.pose.apply(xy_sensor)
        if self.warp is not None:
            point = point + self.warp.displacement(point)
        return point

    def chart_source(self, xy_sensor: torch.Tensor) -> torch.Tensor:
        """Scene/object coordinates of the source frame's samples (rigid)."""
        return self.pose.apply(xy_sensor)

    def covers(self, chart_xy: torch.Tensor) -> torch.Tensor:
        if self.shape == "fullscreen":
            return torch.ones(chart_xy.shape[:-1], dtype=torch.bool)
        return (chart_xy * chart_xy).sum(dim=-1) <= float(self.radius) ** 2


@dataclass(frozen=True)
class SceneFrame:
    """One generated exposure of the layered scene, before sampling onto the grid."""

    frame_id: int
    time: float
    layers: tuple[Layer, ...]
    exposure_gain: float = 1.0
    blur_scene_displacement: tuple[float, float] = (0.0, 0.0)
    blur_samples: int = 1
    exposure_known: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise TypeError("SceneFrame.frame_id must be an integer exposure identity")
        if not math.isfinite(float(self.time)):
            raise ValueError("SceneFrame.time must be finite")
        if not self.layers:
            raise ValueError("SceneFrame needs at least one layer")
        if self.layers[0].shape != "fullscreen":
            raise ValueError(
                "the first (bottom-most painted) layer must be fullscreen so every sample has "
                "a radiance source; later layers paint over it"
            )
        if not math.isfinite(float(self.exposure_gain)) or float(self.exposure_gain) <= 0.0:
            raise ValueError(
                f"exposure_gain must be finite and positive, got {self.exposure_gain!r}"
            )
        if int(self.blur_samples) < 1:
            raise ValueError(f"blur_samples must be at least one, got {self.blur_samples!r}")
        displacement = tuple(float(item) for item in self.blur_scene_displacement)
        if len(displacement) != 2 or not all(math.isfinite(item) for item in displacement):
            raise ValueError("blur_scene_displacement must be two finite numbers")
        object.__setattr__(self, "blur_scene_displacement", displacement)
        if int(self.blur_samples) > 1 and displacement == (0.0, 0.0):
            raise ValueError(
                "a multi-sample exposure needs a non-zero blur displacement; otherwise the "
                "declared blur is a no-op that would only slow the draw"
            )

    def layer(self, layer_id: str) -> Layer:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        raise KeyError(f"no layer {layer_id!r} in frame {self.frame_id}")

    def blur_length_px(self) -> float:
        """Sensor-pixel length of the exposure blur under the fullscreen layer's scale."""
        scale = next(layer.pose.scale for layer in self.layers if layer.shape == "fullscreen")
        return math.hypot(*self.blur_scene_displacement) / float(scale)


@dataclass(frozen=True)
class LayerPair:
    """One layer seen in the reference frame and in the historical source frame."""

    layer_id: str
    field: RadianceField
    reference_pose: SimilarityPose
    source_pose: SimilarityPose
    shape: str = "fullscreen"
    radius: float = 1.0
    occluder: bool = False
    reference_warp: LocalWarp | None = None

    def reference_chart(self, xy_sensor: torch.Tensor) -> torch.Tensor:
        point = self.reference_pose.apply(xy_sensor)
        if self.reference_warp is not None:
            point = point + self.reference_warp.displacement(point)
        return point

    def source_chart(self, xy_sensor: torch.Tensor) -> torch.Tensor:
        return self.source_pose.apply(xy_sensor)

    def pull_source_xy(self, xy_reference: torch.Tensor) -> torch.Tensor:
        """Closed-form pull: rigid source pose inverted at the reference point's chart."""
        return self.source_pose.inverse(self.reference_chart(xy_reference))

    def covers(self, chart_xy: torch.Tensor) -> torch.Tensor:
        if self.shape == "fullscreen":
            return torch.ones(chart_xy.shape[:-1], dtype=torch.bool)
        return (chart_xy * chart_xy).sum(dim=-1) <= float(self.radius) ** 2


def pair_layers(reference: SceneFrame, source: SceneFrame) -> tuple[LayerPair, ...]:
    """Pair the layers of two frames by identity; the source layer must be the same object."""
    if len(reference.layers) != len(source.layers):
        raise ValueError(
            f"frame {reference.frame_id} has {len(reference.layers)} layers but frame "
            f"{source.frame_id} has {len(source.layers)}"
        )
    pairs = []
    for ref_layer, src_layer in zip(reference.layers, source.layers, strict=True):
        if ref_layer.layer_id != src_layer.layer_id:
            raise ValueError(
                f"layer order differs between frames: {ref_layer.layer_id!r} vs "
                f"{src_layer.layer_id!r}"
            )
        if ref_layer.field != src_layer.field:
            raise ValueError(
                f"layer {ref_layer.layer_id!r} holds a different field in the two frames; a "
                "correspondence between two different objects is not a correspondence"
            )
        if (ref_layer.shape, ref_layer.radius) != (src_layer.shape, src_layer.radius):
            raise ValueError(f"layer {ref_layer.layer_id!r} changed shape between frames")
        pairs.append(
            LayerPair(
                layer_id=ref_layer.layer_id,
                field=ref_layer.field,
                reference_pose=ref_layer.pose,
                source_pose=src_layer.pose,
                shape=ref_layer.shape,
                radius=ref_layer.radius,
                occluder=ref_layer.occluder,
                reference_warp=ref_layer.warp,
            )
        )
    return tuple(pairs)


@dataclass(frozen=True)
class PairSolve:
    """Analytic pull geometry of one frame pair on the reference sensor grid."""

    sensor_xy_source: torch.Tensor
    sensor_xy_reference: torch.Tensor
    valid: torch.Tensor
    visibility: torch.Tensor
    visible_layer_reference: torch.Tensor
    visible_layer_source: torch.Tensor
    layer_ids: tuple[str, ...]

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.to(torch.float64).mean())

    def visibility_counts(self) -> dict[str, int]:
        return {
            kind: int((self.visibility == index).sum())
            for index, kind in enumerate(VISIBILITY_KINDS)
        }


def _gather_layer(field: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather ``[L, ...]`` along the layer axis at a ``[...]`` per-position index.

    ``index`` addresses the first ``index.ndim`` axes of ``field`` (after the layer
    axis); the remaining trailing axes are kept whole and broadcast.
    """
    expected = tuple(field.shape[1 : 1 + index.ndim])
    if tuple(index.shape) != expected:
        raise ValueError(
            f"layer index shape {tuple(index.shape)} does not address field axes {expected}"
        )
    trailing = tuple(field.shape[1 + index.ndim :])
    expanded = index.reshape(index.shape + (1,) * len(trailing)).expand(
        tuple(index.shape) + trailing
    )
    flat_field = field.reshape(field.shape[0], -1)
    flat_index = expanded.reshape(-1)
    positions = torch.arange(flat_index.numel(), dtype=torch.int64)
    out = flat_field[flat_index, positions]
    return out.reshape(tuple(index.shape) + trailing)


def _visible_index(covers: torch.Tensor) -> torch.Tensor:
    """Index of the last covering layer per pixel, ``-1`` when none covers it."""
    layers = covers.shape[0]
    index = torch.full(covers.shape[1:], -1, dtype=torch.int64)
    for position in range(layers):
        index = torch.where(covers[position], torch.full_like(index, position), index)
    return index


def _visible_index_groups(covers: torch.Tensor) -> torch.Tensor:
    """Index of the last covering layer for every ``[Q, L, ...]`` coverage group."""
    groups, layers = covers.shape[0], covers.shape[1]
    index = torch.full((groups,) + tuple(covers.shape[2:]), -1, dtype=torch.int64)
    for position in range(layers):
        index = torch.where(covers[:, position], torch.full_like(index, position), index)
    return index


def solve_pull(
    layers: Sequence[LayerPair],
    geometry_reference: FrameGeometry,
    geometry_source: FrameGeometry,
) -> PairSolve:
    """Solve the analytic pull for every reference sensor position.

    The reference sample's visible layer decides which content it shows; the
    pulled source coordinate is that layer's closed-form inverse.  A sample is
    ``valid`` only when the source shows the *same* layer at the pulled
    coordinate; the two occlusion kinds say which side moved, and a pull outside
    the source frame's sampled sensor extent is ``out_of_bounds`` (no wrap, no
    filled value).
    """
    pairs = tuple(layers)
    if not pairs:
        raise ValueError("solve_pull needs at least one paired layer")
    reference_xy = sensor_xy_grid(geometry_reference)
    height, width = reference_xy.shape[:2]
    count = len(pairs)

    covers_reference = torch.zeros((count, height, width), dtype=torch.bool)
    for position, layer in enumerate(pairs):
        covers_reference[position] = layer.covers(layer.reference_chart(reference_xy))
    visible_reference = _visible_index(covers_reference)
    if bool((visible_reference < 0).any()):
        raise ValueError(
            "some reference samples are covered by no layer; the scene must declare a "
            "fullscreen background so every sample has a radiance source"
        )

    pulls = torch.stack([layer.pull_source_xy(reference_xy) for layer in pairs], dim=0)
    covers_source = torch.zeros((count, count, height, width), dtype=torch.bool)
    for query in range(count):
        for position, layer in enumerate(pairs):
            covers_source[query, position] = layer.covers(layer.source_chart(pulls[query]))
    visible_source = _visible_index_groups(covers_source)

    chosen_pull = _gather_layer(pulls, visible_reference)
    chosen_source_layer = _gather_layer(visible_source, visible_reference)

    origin_y, origin_x = geometry_source.grid_origin_yx
    source_height, source_width = geometry_source.sensor_shape
    y_max = float(origin_y + source_height - 1)
    x_max = float(origin_x + source_width - 1)
    tolerance = 1e-9
    in_bounds = (
        (chosen_pull[..., 0] >= origin_x - tolerance)
        & (chosen_pull[..., 0] <= x_max + tolerance)
        & (chosen_pull[..., 1] >= origin_y - tolerance)
        & (chosen_pull[..., 1] <= y_max + tolerance)
    )

    visibility = torch.zeros((height, width), dtype=torch.int64)
    mismatch = in_bounds & (chosen_source_layer != visible_reference)
    visibility[~in_bounds] = VISIBILITY_KINDS.index("out_of_bounds")
    visibility[mismatch] = VISIBILITY_KINDS.index("occluded_in_source")
    nearer_in_reference = (
        mismatch & (chosen_source_layer >= 0) & (chosen_source_layer < visible_reference)
    )
    if bool(nearer_in_reference.any()):
        raise RuntimeError(
            "a reference sample is covered by a nearer layer whose pull is uncovered in the "
            "source; the pull is chart-consistent with a rigid source layer, so this cannot "
            "happen and the scene construction changed underneath the solver"
        )
    return PairSolve(
        sensor_xy_source=chosen_pull,
        sensor_xy_reference=reference_xy,
        valid=visibility == VISIBILITY_KINDS.index("valid"),
        visibility=visibility,
        visible_layer_reference=visible_reference,
        visible_layer_source=chosen_source_layer,
        layer_ids=tuple(layer.layer_id for layer in pairs),
    )


def packed_pull_targets(
    solve: PairSolve,
    geometry_reference: FrameGeometry,
    geometry_source: FrameGeometry,
) -> torch.Tensor:
    """``[C, h, w, 2]`` pull coordinates in the **source** frame's packed grid.

    Channel ``k`` of the reference query grid is mapped through the source
    frame's own per-channel offset, so a different origin or phase permutation
    between the two frames is carried in the target instead of being assumed
    away.  For ``rawvla_rgb3`` the three channels are collocated and the result
    stays in sensor pixels.
    """
    if tuple(geometry_reference.channel_phase) != tuple(geometry_source.channel_phase):
        raise ValueError(
            "packed_pull_targets pairs channel k with channel k; the two frames must declare "
            "the same channel_phase"
        )
    sensor = solve.sensor_xy_source
    origin_y, origin_x = geometry_source.grid_origin_yx
    depths = []
    for channel in range(geometry_reference.channels):
        rows, cols = geometry_reference.grid_indices_yx(channel)
        point = sensor[rows.reshape(-1)][:, cols.reshape(-1)]
        offset = geometry_source.channel_offsets[channel]
        if geometry_source.is_cfa:
            packed_x = (point[..., 0] - (origin_x + offset[1])) / 2.0
            packed_y = (point[..., 1] - (origin_y + offset[0])) / 2.0
        else:
            packed_x = point[..., 0] - origin_x
            packed_y = point[..., 1] - origin_y
        depths.append(torch.stack([packed_x, packed_y], dim=-1))
    return torch.stack(depths, dim=0)


@dataclass(frozen=True)
class VisiblePair:
    """Everything a model may consume: two exposures plus audited metadata.

    This object (and the two frames inside it) holds no geometric truth by
    construction; ``SupervisedPair`` is the only place the analytic labels live.
    """

    reference: RawSensorFrame
    source: RawSensorFrame
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class MotionSupervision:
    """Analytic geometry truth of one pair, kept out of the model-visible half."""

    case: str
    split: str
    sequence: str
    namespace: str
    seed: int
    reference_frame_id: int
    source_frame_id: int
    reference_time: float
    source_time: float
    sensor_xy_source: torch.Tensor
    sensor_xy_reference: torch.Tensor
    valid: torch.Tensor
    visibility: torch.Tensor
    visible_layer_reference: torch.Tensor
    visible_layer_source: torch.Tensor
    layer_ids: tuple[str, ...]
    packed_xy_source: torch.Tensor
    channel_phase: tuple[int, ...]
    channel_offsets: tuple[tuple[int, int], ...]
    grid_origin_yx: tuple[int, int]
    ambiguity_period_px: float | None
    blur_length_px: float
    blur_samples: int
    exposure_gain_reference: float
    exposure_gain_source: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.to(torch.float64).mean())

    def visibility_counts(self) -> dict[str, int]:
        return {
            kind: int((self.visibility == index).sum())
            for index, kind in enumerate(VISIBILITY_KINDS)
        }


@dataclass(frozen=True)
class SupervisedPair:
    """One supervised sample: the model-visible pair and its separate truth."""

    visible: VisiblePair
    supervision: MotionSupervision


@dataclass(frozen=True)
class SeedRule:
    """The declared data-seed rule: hash namespace, separate from the model seeds."""

    namespace_template: str = f"{NAMESPACE_ROOT}/{{split}}/{{case}}/{{sequence}}"
    digest: str = "sha256"
    modulus: int = 1 << 63
    data_seed_base: int = 1009
    model_seeds: tuple[int, ...] = (101, 211, 307)
    statement: str = ""

    def __post_init__(self) -> None:
        for token in ("{split}", "{case}", "{sequence}"):
            if token not in self.namespace_template:
                raise ValueError(f"namespace_template must contain {token}")
        if self.digest != "sha256":
            raise ValueError(f"the declared digest is sha256, got {self.digest!r}")
        if int(self.modulus) < 2:
            raise ValueError("modulus must be at least two")
        if int(self.data_seed_base) in set(self.model_seeds):
            raise ValueError(
                "the data seed base must not be a model seed; data and model streams are separate"
            )

    def namespace(self, split: str, case: str, sequence: str) -> str:
        if split not in REGIONS:
            raise ValueError(f"unknown region {split!r}; expected {REGIONS}")
        for name, token in (("case", case), ("sequence", sequence)):
            if not isinstance(token, str) or not token or "/" in token:
                raise ValueError(
                    f"the {name} token must be a non-empty string without '/', got {token!r}; "
                    "it is part of the hash namespace"
                )
        return self.namespace_template.format(split=split, case=case, sequence=sequence)

    def seed(self, namespace: str) -> int:
        payload = f"{int(self.data_seed_base)}|{namespace}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % int(self.modulus)
        if value in set(self.model_seeds):
            raise ValueError(
                f"data seed {value} collides with a declared model seed; the two streams must "
                "stay separate"
            )
        return value

    def frame_seed(self, namespace: str, purpose: str, frame_id: int) -> int:
        """Sub-seed of one frame's scene or noise draw; independent per frame id."""
        return self.seed(f"{namespace}/{purpose}/{int(frame_id)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_template": self.namespace_template,
            "digest": self.digest,
            "modulus": int(self.modulus),
            "data_seed_base": int(self.data_seed_base),
            "model_seeds": list(self.model_seeds),
            "statement": self.statement,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeedRule:
        _reject_unknown_keys(
            data,
            (
                "namespace_template",
                "digest",
                "modulus",
                "data_seed_base",
                "model_seeds",
                "statement",
            ),
            "SeedRule",
        )
        return cls(
            namespace_template=str(data["namespace_template"]),
            digest=str(data.get("digest", "sha256")),
            modulus=int(data.get("modulus", 1 << 63)),
            data_seed_base=int(data["data_seed_base"]),
            model_seeds=tuple(int(item) for item in data.get("model_seeds", (101, 211, 307))),
            statement=str(data.get("statement", "")),
        )


@dataclass(frozen=True)
class SceneRecipe:
    """Per-case knobs of the scene; every value is a declared synthetic choice."""

    background: str = "sinusoid"
    object_field: str = "blobs"
    occluder: bool = False
    occluder_radius: float = 6.0
    occluder_scale_reference: float = 1.0
    background_step_px: tuple[float, float] = (0.5, 0.0)
    object_step_px: tuple[float, float] = (2.0, 0.0)
    rotation_deg: float = 0.0
    warp_amplitude_px: float = 0.0
    warp_sigma_px: float = 4.0
    warp_ramp_frames: int = 1
    gain_reference: float = 1.0
    gain_source: float = 1.0
    exposure_known: bool = True
    blur_length_px: float = 0.0
    blur_samples: int = 1
    texture_period_px: float = 8.0
    jitter_px: float = 0.25
    sequence_convention: str = ""

    def __post_init__(self) -> None:
        if self.background not in FIELD_KINDS:
            raise ValueError(
                f"unknown background field {self.background!r}; expected {FIELD_KINDS}"
            )
        if self.object_field not in FIELD_KINDS:
            raise ValueError(f"unknown object field {self.object_field!r}; expected {FIELD_KINDS}")
        if float(self.occluder_radius) <= 0.0 and self.occluder:
            raise ValueError("an occluder needs a positive radius")
        if (
            not math.isfinite(float(self.occluder_scale_reference))
            or float(self.occluder_scale_reference) <= 0.0
        ):
            raise ValueError("occluder_scale_reference must be finite and positive")
        for name in ("background_step_px", "object_step_px"):
            item = tuple(float(value) for value in getattr(self, name))
            if len(item) != 2 or not all(math.isfinite(value) for value in item):
                raise ValueError(f"SceneRecipe.{name} must be two finite numbers")
            object.__setattr__(self, name, item)
        if int(self.warp_ramp_frames) < 1:
            raise ValueError("warp_ramp_frames must be at least one")
        if float(self.gain_reference) <= 0.0 or float(self.gain_source) <= 0.0:
            raise ValueError("both exposure gains must be positive")
        if int(self.blur_samples) < 1:
            raise ValueError("blur_samples must be at least one")
        if int(self.blur_samples) > 1 and float(self.blur_length_px) <= 0.0:
            raise ValueError("a multi-sample exposure needs a positive blur length")
        if not math.isfinite(float(self.jitter_px)) or float(self.jitter_px) < 0.0:
            raise ValueError(f"jitter_px must be finite and non-negative, got {self.jitter_px!r}")
        if float(self.blur_length_px) > 0.0 and int(self.blur_samples) < 2:
            raise ValueError(
                "a positive blur length needs at least two exposure samples; otherwise the "
                "declared blur is a no-op"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "background": self.background,
            "object_field": self.object_field,
            "occluder": bool(self.occluder),
            "occluder_radius": float(self.occluder_radius),
            "occluder_scale_reference": float(self.occluder_scale_reference),
            "background_step_px": list(self.background_step_px),
            "object_step_px": list(self.object_step_px),
            "rotation_deg": float(self.rotation_deg),
            "warp_amplitude_px": float(self.warp_amplitude_px),
            "warp_sigma_px": float(self.warp_sigma_px),
            "warp_ramp_frames": int(self.warp_ramp_frames),
            "gain_reference": float(self.gain_reference),
            "gain_source": float(self.gain_source),
            "exposure_known": bool(self.exposure_known),
            "blur_length_px": float(self.blur_length_px),
            "blur_samples": int(self.blur_samples),
            "texture_period_px": float(self.texture_period_px),
            "jitter_px": float(self.jitter_px),
            "sequence_convention": self.sequence_convention,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneRecipe:
        _reject_unknown_keys(
            data,
            (
                "background",
                "object_field",
                "occluder",
                "occluder_radius",
                "occluder_scale_reference",
                "background_step_px",
                "object_step_px",
                "rotation_deg",
                "warp_amplitude_px",
                "warp_sigma_px",
                "warp_ramp_frames",
                "gain_reference",
                "gain_source",
                "exposure_known",
                "blur_length_px",
                "blur_samples",
                "texture_period_px",
                "jitter_px",
                "sequence_convention",
            ),
            "SceneRecipe",
        )
        defaults = cls()
        return cls(
            background=str(data.get("background", defaults.background)),
            object_field=str(data.get("object_field", defaults.object_field)),
            occluder=bool(data.get("occluder", defaults.occluder)),
            occluder_radius=float(data.get("occluder_radius", defaults.occluder_radius)),
            occluder_scale_reference=float(
                data.get("occluder_scale_reference", defaults.occluder_scale_reference)
            ),
            background_step_px=tuple(
                float(item) for item in data.get("background_step_px", defaults.background_step_px)
            ),
            object_step_px=tuple(
                float(item) for item in data.get("object_step_px", defaults.object_step_px)
            ),
            rotation_deg=float(data.get("rotation_deg", defaults.rotation_deg)),
            warp_amplitude_px=float(data.get("warp_amplitude_px", defaults.warp_amplitude_px)),
            warp_sigma_px=float(data.get("warp_sigma_px", defaults.warp_sigma_px)),
            warp_ramp_frames=int(data.get("warp_ramp_frames", defaults.warp_ramp_frames)),
            gain_reference=float(data.get("gain_reference", defaults.gain_reference)),
            gain_source=float(data.get("gain_source", defaults.gain_source)),
            exposure_known=bool(data.get("exposure_known", defaults.exposure_known)),
            blur_length_px=float(data.get("blur_length_px", defaults.blur_length_px)),
            blur_samples=int(data.get("blur_samples", defaults.blur_samples)),
            texture_period_px=float(data.get("texture_period_px", defaults.texture_period_px)),
            jitter_px=float(data.get("jitter_px", defaults.jitter_px)),
            sequence_convention=str(data.get("sequence_convention", defaults.sequence_convention)),
        )


@dataclass(frozen=True)
class ProvenancePolicy:
    """Which region may fit, select and evaluate; enforced against the noise module."""

    fitting_splits: tuple[str, ...] = ("train", "calibration")
    selection_split: str = "validation"
    evaluation_split: str = "test"
    statement: str = ""

    def __post_init__(self) -> None:
        if tuple(self.fitting_splits) != tuple(FITTING_SPLITS):
            raise ValueError(
                f"a noise parameter is fitted on {FITTING_SPLITS} only, got "
                f"{tuple(self.fitting_splits)}"
            )
        if self.selection_split != "validation":
            raise ValueError("checkpoints are selected on the validation region")
        if self.evaluation_split != "test":
            raise ValueError("the test region is the evaluation region")
        if set(self.fitting_splits) & {self.selection_split, self.evaluation_split}:
            raise ValueError("the fitting, selection and evaluation roles must stay disjoint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitting_splits": list(self.fitting_splits),
            "selection_split": self.selection_split,
            "evaluation_split": self.evaluation_split,
            "statement": self.statement,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProvenancePolicy:
        _reject_unknown_keys(
            data,
            ("fitting_splits", "selection_split", "evaluation_split", "statement"),
            "ProvenancePolicy",
        )
        return cls(
            fitting_splits=tuple(str(item) for item in data.get("fitting_splits", FITTING_SPLITS)),
            selection_split=str(data.get("selection_split", "validation")),
            evaluation_split=str(data.get("evaluation_split", "test")),
            statement=str(data.get("statement", "")),
        )


@dataclass(frozen=True)
class P2MotionNoiseConfig:
    """Loadable, validated P2 configuration: seeds, regions, geometry, cases, noise."""

    schema: str
    config_version: str
    seed_rule: SeedRule
    policy: ProvenancePolicy
    regions: Mapping[str, str]
    geometry: FrameGeometry
    pseudo_raw_geometry: FrameGeometry
    cases: Mapping[str, SceneRecipe]
    noise: NoiseCalibration
    materials: Mapping[str, str] = field(default_factory=dict)
    stage: str = "P2"
    created_on: str = ""
    source_documents: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema != MOTION_SCHEMA:
            raise ValueError(f"unknown schema {self.schema!r}; expected {MOTION_SCHEMA!r}")
        if not self.config_version:
            raise ValueError("P2MotionNoiseConfig.config_version must be non-empty")
        regions = dict(self.regions)
        if tuple(regions) != REGIONS:
            raise ValueError(f"regions must declare {REGIONS} in order, got {tuple(regions)}")
        if any(not isinstance(value, str) or not value for value in regions.values()):
            raise ValueError("every region needs a non-empty usage statement")
        object.__setattr__(self, "regions", MappingProxyType(regions))
        cases = dict(self.cases)
        if tuple(sorted(cases)) != tuple(sorted(CASE_KINDS)):
            raise ValueError(
                f"cases must cover exactly {sorted(CASE_KINDS)}, got {sorted(cases)}; a case "
                "without a declared recipe could not be reproduced"
            )
        object.__setattr__(self, "cases", MappingProxyType(cases))
        materials = dict(self.materials)
        if tuple(materials) != MATERIAL_CLASSES:
            raise ValueError(
                f"materials must declare {MATERIAL_CLASSES} in order, got {tuple(materials)}; a "
                "real CFA capture, a HDR-derived pseudo-RAW entry and an inverted RGB capture "
                "are different material classes and stay distinguishable"
            )
        if any(not isinstance(value, str) or not value for value in materials.values()):
            raise ValueError("every material class needs a non-empty statement")
        object.__setattr__(self, "materials", MappingProxyType(materials))
        object.__setattr__(self, "source_documents", tuple(self.source_documents))
        if not isinstance(self.policy, ProvenancePolicy):
            raise TypeError("P2MotionNoiseConfig.policy must be a ProvenancePolicy")
        if not self.geometry.is_cfa:
            raise ValueError("the primary geometry must be a CFA sensor grid")
        if self.pseudo_raw_geometry.is_cfa:
            raise ValueError("pseudo_raw_geometry must be the three-channel rawvla_rgb3 entry")

    def recipe(self, case: str) -> SceneRecipe:
        if case not in self.cases:
            raise KeyError(f"unknown case {case!r}; declared cases are {sorted(self.cases)}")
        return self.cases[case]

    def geometry_for(self, input_mode: str) -> FrameGeometry:
        if input_mode == self.geometry.input_mode:
            return self.geometry
        if input_mode == self.pseudo_raw_geometry.input_mode:
            return self.pseudo_raw_geometry
        raise ValueError(f"no declared geometry for input_mode {input_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config_version": self.config_version,
            "seed_rule": self.seed_rule.to_dict(),
            "policy": self.policy.to_dict(),
            "regions": dict(self.regions),
            "geometry": self.geometry.to_dict(),
            "pseudo_raw_geometry": self.pseudo_raw_geometry.to_dict(),
            "cases": {name: recipe.to_dict() for name, recipe in self.cases.items()},
            "noise": self.noise.to_dict(),
            "materials": dict(self.materials),
            "stage": self.stage,
            "created_on": self.created_on,
            "source_documents": list(self.source_documents),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> P2MotionNoiseConfig:
        _reject_unknown_keys(
            data,
            (
                "schema",
                "config_version",
                "seed_rule",
                "policy",
                "regions",
                "geometry",
                "pseudo_raw_geometry",
                "cases",
                "noise",
                "materials",
                "stage",
                "created_on",
                "source_documents",
                "notes",
            ),
            "P2MotionNoiseConfig",
        )
        for required in (
            "schema",
            "config_version",
            "seed_rule",
            "policy",
            "regions",
            "geometry",
            "pseudo_raw_geometry",
            "cases",
            "noise",
            "materials",
        ):
            if required not in data:
                raise ValueError(f"P2 motion/noise JSON is missing the required field {required!r}")
        return cls(
            schema=str(data["schema"]),
            config_version=str(data["config_version"]),
            seed_rule=SeedRule.from_dict(data["seed_rule"]),
            policy=ProvenancePolicy.from_dict(data["policy"]),
            regions={str(key): str(value) for key, value in data["regions"].items()},
            geometry=FrameGeometry.from_dict(data["geometry"]),
            pseudo_raw_geometry=FrameGeometry.from_dict(data["pseudo_raw_geometry"]),
            cases={
                str(name): SceneRecipe.from_dict(recipe) for name, recipe in data["cases"].items()
            },
            noise=NoiseCalibration.from_dict(data["noise"]),
            materials={str(key): str(value) for key, value in data["materials"].items()},
            stage=str(data.get("stage", "P2")),
            created_on=str(data.get("created_on", "")),
            source_documents=tuple(str(item) for item in data.get("source_documents", ())),
            notes=str(data.get("notes", "")),
        )


def load_p2_motion_noise_config(path: str | Path) -> P2MotionNoiseConfig:
    """Load and validate ``configs/encoders/n01_full/p2_motion_noise.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return P2MotionNoiseConfig.from_dict(payload)


def _field_from_kind(kind: str, *, period_px: float, seed: int) -> RadianceField:
    generator = torch.Generator().manual_seed(int(seed) % (1 << 63))
    if kind == "sinusoid":
        return SinusoidField(
            period_xy=(float(period_px), float(period_px)),
            phase_rgb=(0.0, 0.4, 0.8),
        )
    if kind == "steps":
        return StepField(period_xy=(float(period_px), float(period_px)))
    centers = tuple(
        (
            float(torch.rand((), generator=generator) * 24.0 - 12.0),
            float(torch.rand((), generator=generator) * 24.0 - 12.0),
        )
        for _ in range(6)
    )
    colors = tuple(
        tuple(float(value) for value in torch.rand(3, generator=generator) * 0.5 + 0.2)
        for _ in range(len(centers))
    )
    return BlobField(centers_xy=centers, sigma=float(period_px) / 2.0, colors_rgb=colors)


def _layer_radiance(scene: SceneFrame, layer: Layer, chart_xy: torch.Tensor) -> torch.Tensor:
    """Radiance of one layer at its own chart coordinates, integrated over the exposure."""
    if scene.blur_samples <= 1:
        return layer.field.sample(chart_xy)
    blur = torch.tensor(scene.blur_scene_displacement, dtype=torch.float64)
    offsets = torch.linspace(-0.5, 0.5, steps=int(scene.blur_samples), dtype=torch.float64)
    total = torch.zeros(chart_xy.shape[:-1] + (3,), dtype=torch.float64)
    for offset in offsets:
        total = total + layer.field.sample(chart_xy - blur * offset)
    return total / float(int(scene.blur_samples))


def scene_radiance(scene: SceneFrame, xy_sensor: torch.Tensor) -> torch.Tensor:
    """``[..., 2]`` sensor ``(x, y)`` to ``[..., 3]`` linear radiance of the visible layer.

    Later layers paint over earlier ones.  The silhouette is evaluated at the
    exposure-centre chart, which is exactly where the pair truth is defined; the
    blur only integrates radiance inside the silhouette.
    """
    xy_sensor = torch.as_tensor(xy_sensor, dtype=torch.float64)
    out = torch.zeros(xy_sensor.shape[:-1] + (3,), dtype=torch.float64)
    for layer in scene.layers:
        chart = layer.pose.apply(xy_sensor)
        if layer.warp is not None:
            chart = chart + layer.warp.displacement(chart)
        value = _layer_radiance(scene, layer, chart)
        out = torch.where(layer.covers(chart).unsqueeze(-1), value, out)
    return out


def _channel_sensor_grid(geometry: FrameGeometry) -> torch.Tensor:
    """``[C, h, w, 2]`` full sensor ``(x, y)`` coordinates of every packed sample."""
    rows, cols = geometry.packed_shape
    grids = []
    for channel in range(geometry.channels):
        offset = geometry.channel_offsets[channel]
        if geometry.is_cfa:
            y = geometry.origin_yx[0] + 2 * torch.arange(rows).reshape(rows, 1) + offset[0]
            x = geometry.origin_yx[1] + 2 * torch.arange(cols).reshape(1, cols) + offset[1]
        else:
            y = geometry.origin_yx[0] + torch.arange(rows).reshape(rows, 1)
            x = geometry.origin_yx[1] + torch.arange(cols).reshape(1, cols)
        y = y.expand(rows, cols).to(torch.float64)
        x = x.expand(rows, cols).to(torch.float64)
        grids.append(torch.stack([x, y], dim=-1))
    return torch.stack(grids, dim=0)


_COLOR_INDEX = {"R": 0, "G": 1, "B": 2}


def _measured_signal(
    radiance: torch.Tensor, geometry: FrameGeometry, grid: torch.Tensor
) -> torch.Tensor:
    """Select what each sample actually measures: one CFA colour, or the collocated triple."""
    rows, cols = geometry.packed_shape
    out = torch.zeros((geometry.channels, rows, cols), dtype=torch.float64)
    for channel in range(geometry.channels):
        if not geometry.is_cfa:
            out[channel] = radiance[channel, :, :, channel]
            continue
        ys = grid[channel, :, :, 1].to(torch.int64)
        xs = grid[channel, :, :, 0].to(torch.int64)
        colors = CFA_PATTERN_COLORS[geometry.cfa_pattern]
        index = ((ys % 2) * 2 + (xs % 2)).reshape(-1)
        table = torch.tensor([_COLOR_INDEX[color] for color in colors], dtype=torch.int64)
        selected = table[index].reshape(rows, cols)
        out[channel] = torch.gather(radiance[channel], 2, selected.unsqueeze(-1)).squeeze(-1)
    return out


def render_raw_frame(
    scene: SceneFrame,
    geometry: FrameGeometry,
    calibration: NoiseCalibration,
    *,
    dataset_id: str,
    episode_id: str,
    camera_id: str,
    add_noise: bool = True,
    noise_seed: int | None = None,
) -> RawSensorFrame:
    """Sample the scene onto the sensor grid and apply the declared noise realisation.

    ``add_noise=False`` is a diagnostic switch for oracle comparisons against the
    analytic field; it bypasses the declared calibration and must not be used for
    training data.
    """
    grid = _channel_sensor_grid(geometry)
    radiance = scene_radiance(scene, grid)
    signal = _measured_signal(radiance, geometry, grid)
    black = float(calibration.black_level_dn)
    clean = black + float(scene.exposure_gain) * signal

    block_id = calibration.black_level.id_template.format(
        dataset_id=dataset_id, episode_id=episode_id, camera_id=camera_id, frame_id=scene.frame_id
    )
    if add_noise:
        if noise_seed is None:
            raise ValueError("rendering a noisy frame needs an explicit noise_seed")
        generator = torch.Generator().manual_seed(int(noise_seed) % (1 << 63))
        block_error = draw_block_error(block_id, calibration, generator)
        noisy = add_sensor_noise(
            clean, calibration=calibration, generator=generator, block_error=block_error
        )
    else:
        noisy = clean

    white = calibration.white_level_dn
    if white is None:
        clipped = noisy.clamp(min=0.0)
        validity = noisy >= 0.0
    else:
        clipped = noisy.clamp(min=0.0, max=float(white))
        validity = (noisy >= 0.0) & (noisy <= float(white))
    values = clipped.to(torch.float32)
    return RawSensorFrame(
        dataset_id=dataset_id,
        episode_id=episode_id,
        camera_id=camera_id,
        frame_id=int(scene.frame_id),
        values=values,
        channel_phase=tuple(geometry.channel_phase),
        cfa_pattern=geometry.cfa_pattern,
        sensor_origin_xy=tuple(geometry.origin_yx),
        validity=validity,
        time=float(scene.time),
        calibration_version=calibration.calibration_version,
    )


def build_frames(
    recipe: SceneRecipe,
    *,
    geometry: FrameGeometry,
    namespace: str,
    seed_rule: SeedRule,
    length: int,
    displacement_xy: tuple[float, float] | None = None,
) -> tuple[SceneFrame, ...]:
    """Deterministic per-frame scene states; frame ``k`` never reads frame ``k + 1``.

    Odd frame ids carry the reference-side appearance (``gain_reference`` and the
    exposure blur), even ids the source-side one, so consecutive pairs always
    have a declared exposure difference in both directions.  Every per-frame
    decision comes from that frame's own sub-seed or from the recipe, which is
    what makes truncating a sequence leave every earlier frame bit-identical.
    """
    if length < 2:
        raise ValueError(f"a pair needs at least two frames, got length={length}")
    motion = torch.Generator().manual_seed(seed_rule.frame_seed(namespace, "motion", 0) % (1 << 63))
    center = (float(geometry.width - 1) / 2.0, float(geometry.height - 1) / 2.0)
    jitter_amplitude = 0.0 if displacement_xy is not None else float(recipe.jitter_px)
    jitter = (
        (torch.rand(2, generator=motion, dtype=torch.float64) - 0.5) * 2.0 * jitter_amplitude
        if jitter_amplitude > 0.0
        else torch.zeros(2, dtype=torch.float64)
    )
    background = _field_from_kind(
        recipe.background,
        period_px=recipe.texture_period_px,
        seed=seed_rule.frame_seed(namespace, "background", 0),
    )
    object_field = _field_from_kind(
        recipe.object_field,
        period_px=max(recipe.texture_period_px / 2.0, 2.0),
        seed=seed_rule.frame_seed(namespace, "object", 0),
    )
    base_step = torch.tensor(
        displacement_xy if displacement_xy is not None else recipe.background_step_px,
        dtype=torch.float64,
    )
    object_step = torch.tensor(recipe.object_step_px, dtype=torch.float64)
    if displacement_xy is not None:
        object_step = torch.zeros(2, dtype=torch.float64)

    frames = []
    for frame_id in range(int(length)):
        index = torch.tensor(float(frame_id), dtype=torch.float64)
        theta = float(recipe.rotation_deg) * float(frame_id)
        background_translation = index * (base_step + jitter)
        # The object chart is centred on the image centre, so motion is measured in
        # scene/sensor pixels around that centre and the disk sits in the middle of the frame.
        object_translation = index * object_step - torch.tensor(center, dtype=torch.float64)
        object_scale = float(recipe.occluder_scale_reference) if frame_id % 2 == 1 else 1.0
        layers: list[Layer] = []
        if recipe.occluder:
            layers.append(
                Layer(
                    layer_id="background",
                    field=background,
                    pose=SimilarityPose(
                        scale=1.0,
                        theta_deg=theta,
                        center_xy=center,
                        translation_xy=tuple(background_translation.tolist()),
                    ),
                    shape="fullscreen",
                )
            )
            layers.append(
                Layer(
                    layer_id="object",
                    field=object_field,
                    pose=SimilarityPose(
                        scale=object_scale,
                        theta_deg=theta,
                        center_xy=(0.0, 0.0),
                        translation_xy=tuple(object_translation.tolist()),
                    ),
                    shape="disk",
                    radius=float(recipe.occluder_radius),
                    occluder=True,
                    warp=_warp_for(recipe, frame_id, center),
                )
            )
        else:
            layers.append(
                Layer(
                    layer_id="background",
                    field=background,
                    pose=SimilarityPose(
                        scale=1.0,
                        theta_deg=theta,
                        center_xy=center,
                        translation_xy=tuple(background_translation.tolist()),
                    ),
                    shape="fullscreen",
                    warp=_warp_for(recipe, frame_id, center),
                )
            )
        reference_like = frame_id % 2 == 1
        gain = float(recipe.gain_reference) if reference_like else float(recipe.gain_source)
        blur = 0.0
        blur_samples = 1
        if reference_like and float(recipe.blur_length_px) > 0.0:
            blur = float(recipe.blur_length_px)
            blur_samples = int(recipe.blur_samples)
        frames.append(
            SceneFrame(
                frame_id=frame_id,
                time=float(frame_id),
                layers=tuple(layers),
                exposure_gain=gain,
                blur_scene_displacement=(blur, 0.0),
                blur_samples=blur_samples,
                exposure_known=bool(recipe.exposure_known),
            )
        )
    return tuple(frames)


def _warp_for(
    recipe: SceneRecipe, frame_id: int, center_xy: tuple[float, float]
) -> LocalWarp | None:
    """Local non-rigid deformation of the current frame; frame 0 stays undeformed.

    The deformation is centred on the sensor centre, where an un-rotated,
    un-scaled pose keeps its chart coordinates; the corners stay undeformed.
    """
    if float(recipe.warp_amplitude_px) == 0.0:
        return None
    weight = min(1.0, float(frame_id) / float(int(recipe.warp_ramp_frames)))
    if weight <= 0.0:
        return None
    return LocalWarp(
        amplitude=float(recipe.warp_amplitude_px) * weight,
        sigma=float(recipe.warp_sigma_px),
        center_xy=tuple(float(item) for item in center_xy),
        direction_xy=(1.0, 0.0),
    )


@dataclass(frozen=True)
class MotionDataGenerator:
    """Deterministic generator of supervised RAW motion pairs for one P2 config.

    ``visible`` holds only what a model may consume; ``supervision`` holds the
    analytic flow, validity and occlusion truth.  Nothing in ``visible`` is
    derived from ``supervision``.
    """

    config: P2MotionNoiseConfig
    dataset_id: str = "n01_synth_motion"
    camera_id: str = "cam0"

    @classmethod
    def from_config_path(cls, path: str | Path) -> MotionDataGenerator:
        return cls(load_p2_motion_noise_config(path))

    @property
    def calibration(self) -> NoiseCalibration:
        return self.config.noise

    def namespace(self, split: str, case: str, sequence: str) -> str:
        return self.config.seed_rule.namespace(split, case, sequence)

    def seed(self, split: str, case: str, sequence: str) -> int:
        return self.config.seed_rule.seed(self.namespace(split, case, sequence))

    def geometry(self, input_mode: str | None = None) -> FrameGeometry:
        return self.config.geometry_for(input_mode or self.config.geometry.input_mode)

    def frames(
        self,
        *,
        split: str,
        case: str,
        sequence: str,
        length: int = 2,
        input_mode: str | None = None,
    ) -> tuple[SceneFrame, ...]:
        geometry = self.geometry(input_mode)
        displacement = parse_subpixel_token(sequence) if case == "translation_subpixel" else None
        return build_frames(
            self.config.recipe(case),
            geometry=geometry,
            namespace=self.namespace(split, case, sequence),
            seed_rule=self.config.seed_rule,
            length=length,
            displacement_xy=displacement,
        )

    def _render(
        self,
        scene: SceneFrame,
        geometry: FrameGeometry,
        *,
        split: str,
        case: str,
        sequence: str,
        add_noise: bool,
    ) -> RawSensorFrame:
        namespace = self.namespace(split, case, sequence)
        return render_raw_frame(
            scene,
            geometry,
            self.calibration,
            dataset_id=self.dataset_id,
            episode_id=f"{split}/{case}",
            camera_id=self.camera_id,
            add_noise=add_noise,
            noise_seed=self.config.seed_rule.frame_seed(namespace, "noise", scene.frame_id),
        )

    def render(
        self,
        scene: SceneFrame,
        *,
        split: str,
        case: str,
        sequence: str,
        input_mode: str | None = None,
        add_noise: bool = True,
    ) -> RawSensorFrame:
        """Render one scene state; the noise seed follows the frame, not the call order."""
        return self._render(
            scene,
            self.geometry(input_mode),
            split=split,
            case=case,
            sequence=sequence,
            add_noise=add_noise,
        )

    def _truth(
        self,
        *,
        case: str,
        split: str,
        sequence: str,
        namespace: str,
        seed: int,
        reference_scene: SceneFrame,
        source_scene: SceneFrame,
        geometry_reference: FrameGeometry,
        geometry_source: FrameGeometry,
    ) -> MotionSupervision:
        layers = pair_layers(reference_scene, source_scene)
        solve = solve_pull(layers, geometry_reference, geometry_source)
        recipe = self.config.recipe(case)
        ambiguity = None
        if case == "repeated_texture" and isinstance(layers[-1].field, SinusoidField):
            ambiguity = float(layers[-1].field.ambiguity_period_xy[0])
        occluders = [index for index, layer in enumerate(layers) if layer.occluder]
        occluder_fraction = (
            float(sum(int((solve.visible_layer_reference == index).sum()) for index in occluders))
            / float(solve.visible_layer_reference.numel())
            if occluders
            else 0.0
        )
        return MotionSupervision(
            case=case,
            split=split,
            sequence=sequence,
            namespace=namespace,
            seed=seed,
            reference_frame_id=int(reference_scene.frame_id),
            source_frame_id=int(source_scene.frame_id),
            reference_time=float(reference_scene.time),
            source_time=float(source_scene.time),
            sensor_xy_source=solve.sensor_xy_source,
            sensor_xy_reference=solve.sensor_xy_reference,
            valid=solve.valid,
            visibility=solve.visibility,
            visible_layer_reference=solve.visible_layer_reference,
            visible_layer_source=solve.visible_layer_source,
            layer_ids=solve.layer_ids,
            packed_xy_source=packed_pull_targets(solve, geometry_reference, geometry_source),
            channel_phase=tuple(geometry_reference.channel_phase),
            channel_offsets=tuple(geometry_reference.channel_offsets),
            grid_origin_yx=tuple(geometry_reference.grid_origin_yx),
            ambiguity_period_px=ambiguity,
            blur_length_px=float(reference_scene.blur_length_px()),
            blur_samples=int(reference_scene.blur_samples),
            exposure_gain_reference=float(reference_scene.exposure_gain),
            exposure_gain_source=float(source_scene.exposure_gain),
            diagnostics={
                "recipe": recipe.to_dict(),
                "requested_reference_frame": int(reference_scene.frame_id),
                "requested_source_frame": int(source_scene.frame_id),
                "valid_fraction": solve.valid_fraction,
                "visibility_counts": solve.visibility_counts(),
                "occluder_visible_fraction": occluder_fraction,
            },
        )

    def _metadata(
        self,
        *,
        split: str,
        case: str,
        sequence: str,
        geometry: FrameGeometry,
        reference_scene: SceneFrame,
        source_scene: SceneFrame,
        reference_frame: RawSensorFrame,
        source_frame: RawSensorFrame,
    ) -> Mapping[str, Any]:
        material = CFA_MATERIAL_CLASS if geometry.is_cfa else PSEUDO_RAW_MATERIAL_CLASS
        template = self.calibration.black_level.id_template
        episode_id = f"{split}/{case}"
        exposure_known = bool(reference_scene.exposure_known and source_scene.exposure_known)
        packed_step = 2 if geometry.is_cfa else 1
        return {
            "split": split,
            "case": case,
            "sequence": sequence,
            "material_class": material,
            "input_mode": geometry.input_mode,
            "is_real_cfa": False,
            "value_units": geometry.value_units,
            "calibration_version": self.calibration.calibration_version,
            "packed_grid": {
                "packed_step_sensor_px": packed_step,
                "channel_phase": list(geometry.channel_phase),
                "channel_offsets_yx": [list(item) for item in geometry.channel_offsets],
                "grid_origin_yx": list(geometry.grid_origin_yx),
                "sensor_origin_xy": list(geometry.origin_yx),
                "note": (
                    "packed (i, j) of channel k is sensor "
                    "(y0 + 2i + dy_k, x0 + 2j + dx_k); one packed step spans two sensor pixels"
                    if geometry.is_cfa
                    else "rgb3 channels are collocated at full sensor resolution; no CFA phase"
                ),
            },
            "exposure": {
                "known": exposure_known,
                "reference_gain": (
                    float(reference_scene.exposure_gain) if exposure_known else None
                ),
                "source_gain": float(source_scene.exposure_gain) if exposure_known else None,
                "black_level_dn": float(self.calibration.black_level_dn),
                "white_level_dn": self.calibration.white_level_dn,
            },
            "noise_block_id": {
                "reference": template.format(
                    dataset_id=self.dataset_id,
                    episode_id=episode_id,
                    camera_id=self.camera_id,
                    frame_id=reference_frame.frame_id,
                ),
                "source": template.format(
                    dataset_id=self.dataset_id,
                    episode_id=episode_id,
                    camera_id=self.camera_id,
                    frame_id=source_frame.frame_id,
                ),
            },
            "timestamps": {
                "reference": float(reference_frame.time),
                "source": float(source_frame.time),
            },
        }

    def _pair_from_frames(
        self,
        frames: Sequence[SceneFrame],
        *,
        split: str,
        case: str,
        sequence: str,
        reference_index: int,
        source_index: int,
        geometry: FrameGeometry,
        geometry_source: FrameGeometry | None,
        add_noise: bool,
    ) -> SupervisedPair:
        reference_scene = frames[reference_index]
        source_scene = frames[source_index]
        if int(source_scene.time) >= int(reference_scene.time):
            raise ValueError(
                "a pair is always (current reference, strictly older source): "
                f"got reference {reference_scene.frame_id} and source {source_scene.frame_id}"
            )
        reference_source_geometry = geometry_source or geometry
        reference_frame = self._render(
            reference_scene,
            geometry,
            split=split,
            case=case,
            sequence=sequence,
            add_noise=add_noise,
        )
        source_frame = self._render(
            source_scene,
            reference_source_geometry,
            split=split,
            case=case,
            sequence=sequence,
            add_noise=add_noise,
        )
        namespace = self.namespace(split, case, sequence)
        supervision = self._truth(
            case=case,
            split=split,
            sequence=sequence,
            namespace=namespace,
            seed=self.seed(split, case, sequence),
            reference_scene=reference_scene,
            source_scene=source_scene,
            geometry_reference=geometry,
            geometry_source=reference_source_geometry,
        )
        visible = VisiblePair(
            reference=reference_frame,
            source=source_frame,
            metadata=self._metadata(
                split=split,
                case=case,
                sequence=sequence,
                geometry=geometry,
                reference_scene=reference_scene,
                source_scene=source_scene,
                reference_frame=reference_frame,
                source_frame=source_frame,
            ),
        )
        return SupervisedPair(visible=visible, supervision=supervision)

    def sample_pair(
        self,
        *,
        split: str,
        case: str,
        sequence: str,
        history_offset: int = 1,
        length: int | None = None,
        input_mode: str | None = None,
        source_geometry: FrameGeometry | None = None,
        add_noise: bool = True,
    ) -> SupervisedPair:
        """One supervised pair: current reference frame and a strictly older source."""
        if int(history_offset) < 1:
            raise ValueError("history_offset must be at least one frame")
        if case not in self.config.cases:
            raise ValueError(
                f"unknown case {case!r}; declared cases are {sorted(self.config.cases)}"
            )
        resolved_length = int(length) if length is not None else int(history_offset) + 1
        if resolved_length < int(history_offset) + 1:
            raise ValueError(
                f"length={resolved_length} cannot hold history_offset={history_offset}"
            )
        geometry = self.geometry(input_mode)
        frames = self.frames(
            split=split,
            case=case,
            sequence=sequence,
            length=resolved_length,
            input_mode=input_mode,
        )
        reference_index = resolved_length - 1
        return self._pair_from_frames(
            frames,
            split=split,
            case=case,
            sequence=sequence,
            reference_index=reference_index,
            source_index=reference_index - int(history_offset),
            geometry=geometry,
            geometry_source=source_geometry,
            add_noise=add_noise,
        )

    def sample_sequence(
        self,
        *,
        split: str,
        case: str,
        sequence: str,
        length: int = 3,
        input_mode: str | None = None,
        add_noise: bool = True,
    ) -> tuple[SupervisedPair, ...]:
        """Consecutive pairs ``(k, k - 1)`` of one sequence, oldest first."""
        if int(length) < 2:
            raise ValueError(f"a sequence needs at least two frames, got {length}")
        geometry = self.geometry(input_mode)
        frames = self.frames(
            split=split, case=case, sequence=sequence, length=int(length), input_mode=input_mode
        )
        return tuple(
            self._pair_from_frames(
                frames,
                split=split,
                case=case,
                sequence=sequence,
                reference_index=index,
                source_index=index - 1,
                geometry=geometry,
                geometry_source=None,
                add_noise=add_noise,
            )
            for index in range(1, int(length))
        )

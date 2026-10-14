"""N04 mechanism data: D0 analytic 3-D rendering diagnostics and D1 collection feasibility.

Scope
-----
This module is the S0 "mechanism data" deliverable of N04 v1.0 (``docs/plans`` §9.1 D0/D1):

* **D0** builds analytic pinhole + z-buffer diagnostics that need no extra simulator.
  Every pixel, depth value, visibility mask and event label of a frame comes from the same
  per-sample world state (camera pose and rigid-body poses).  :func:`audit_d0` re-derives
  labels through an independent point-projection path and rejects cross-sample contamination,
  including the historical ``[B,1,2] + [B,2] -> [B,B,2]`` broadcast defect that once made
  ``frameworks/n04/synthetic.py`` merge the object positions of every batch sample into one
  label map.
* **D1** is *partial* in this round: a runnable collector plus an item-by-item feasibility
  report.  No large-scale collection is started here.  The collector also defines the
  *executed-trace* schema (:func:`d1_trace_schema`): every step stores its pre-state, the
  command actually issued to the simulator and the post-state, together with an explicit
  solver warm-start mode and controller-buffer policy.  With those fields, plus the ordered
  replay recipe and a passing verification replay, "executed action <-> state transition" is
  bit-exactly recomputable; missing any of them leaves only phase-level evidence.

Conventions (the same text is written into ``d0/manifest.json``)
---------------------------------------------------------------
* Units: metres for geometry, seconds for time, digital numbers (DN) for CFA counts.
* Camera frame: OpenCV layout, ``x`` right, ``y`` down, ``z`` forward.  ``depth_m`` is the
  camera-space ``z`` (not the Euclidean ray length).
* Pinhole: ``K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.  Pixel ``(u, v)`` uses integer
  column/row indices whose origin is the top-left pixel centre, and its ray is
  ``r = ((u - cx) / fx, (v - cy) / fy, 1)``.
* World frame: right-handed.  A body's local ``+z`` is its face normal and local ``x``/``y``
  span its rectangular face.  A camera-facing body uses ``rpy = (180, 0, 0)`` degrees.
* A pixel with no surface has ``body_index = -1`` and ``depth_m = inf``.
* Ground truth (``gt_*``) never enters the online bundle (``online_*``); the audit enforces
  the separation.

CFA declaration
---------------
The CFA branch samples the declared irradiance channel **and phase** directly from the linear
scene radiance: for ``GRBG`` the phase ``(0, 0)`` keeps green, ``(0, 1)`` red, ``(1, 0)`` blue
and ``(1, 1)`` green.  The output is a synthetic mosaic of a declared sensor model
(``synthetic_cfa``) with explicit black/white levels; it is never a re-mosaicked RGB image
presented as original RAW.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "BodyTrack",
    "D0Config",
    "PinholeCamera",
    "PlaneBody",
    "RenderResult",
    "WorldBatch",
    "audit_d0",
    "build_feasibility",
    "d1_collect_libero_trace",
    "d1_main",
    "d1_recomputability_report",
    "d1_trace_schema",
    "d1_write_trace",
    "generate_d0",
    "main",
    "render_world",
    "run_cli",
]

SCHEMA_D0 = "n04-d0-mechanism-data-v1"
SCHEMA_FACTORS = "n04-d0-factors-v1"
SCHEMA_AUDIT = "n04-d0-label-audit-v1"
SCHEMA_FEASIBILITY = "n04-d1-feasibility-v1"

#: Single-factor dimensions required by ``configs/n04/full/s0_data.json``.
DEFAULT_FACTORS: tuple[str, ...] = (
    "static",
    "camera_only",
    "object_only",
    "both",
    "stop_reverse",
    "occlusion_recovery",
    "robot_self_occlusion",
    "gain",
    "noise",
    "time_shift",
    "wrong_association",
    "unidentifiable_full_view_plane",
)

#: Declared combinations, applied after the single factors ("single factor first, then
#: combinations").  Components resolve in listed order; the first component that sets a
#: field wins.
DEFAULT_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("camera_only", "gain"),
    ("both", "occlusion_recovery"),
    ("object_only", "noise"),
    ("camera_only", "noise", "gain", "time_shift"),
)

#: CFA phase -> spectral channel per 2x2 cell, channel order (R, G, B).
#: Row phase 0/1 selects the tuple, column phase 0/1 the entry.
CFA_PHASE_CHANNELS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "RGGB": ((0, 1), (1, 2)),
    "BGGR": ((2, 1), (1, 0)),
    "GRBG": ((1, 0), (2, 1)),
    "GBRG": ((1, 2), (0, 1)),
}

CHANNEL_NAMES: tuple[str, ...] = ("R", "G", "B")

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def rotation_rpy_deg(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` in degrees, batched over the leading dims."""
    r, p, y = torch.deg2rad(roll), torch.deg2rad(pitch), torch.deg2rad(yaw)
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    shape = torch.broadcast_shapes(r.shape, p.shape, y.shape)
    out = torch.zeros(shape + (3, 3), dtype=torch.float64)
    cr, sr = torch.broadcast_tensors(cr, sr)
    cp, sp = torch.broadcast_tensors(cp, sp)
    cy, sy = torch.broadcast_tensors(cy, sy)
    out[..., 0, 0] = cy * cp
    out[..., 0, 1] = cy * sp * sr - sy * cr
    out[..., 0, 2] = cy * sp * cr + sy * sr
    out[..., 1, 0] = sy * cp
    out[..., 1, 1] = sy * sp * sr + cy * cr
    out[..., 1, 2] = sy * sp * cr - cy * sr
    out[..., 2, 0] = -sp
    out[..., 2, 1] = cp * sr
    out[..., 2, 2] = cp * cr
    return out


def transform_from_rt(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Build ``[..., 4, 4]`` transforms from ``[..., 3, 3]`` rotations and ``[..., 3]`` offsets."""
    shape = rotation.shape[:-2]
    out = torch.zeros(shape + (4, 4), dtype=torch.float64)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = translation
    out[..., 3, 3] = 1.0
    return out


def invert_transform(matrix: torch.Tensor) -> torch.Tensor:
    """Invert rigid ``[..., 4, 4]`` transforms."""
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    rotation_t = rotation.transpose(-1, -2)
    out = torch.zeros_like(matrix)
    out[..., :3, :3] = rotation_t
    out[..., :3, 3] = -(rotation_t @ translation[..., None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def project_points(
    camera: PinholeCamera, points_cam: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-space points; returns ``(u_px, v_px)``. Division by zero stays inf/NaN."""
    z = points_cam[..., 2]
    u = camera.fx * points_cam[..., 0] / z + camera.cx
    v = camera.fy * points_cam[..., 1] / z + camera.cy
    return u, v


# --------------------------------------------------------------------------------------
# scene description
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PinholeCamera:
    """Declared diagnostic pinhole camera.  ``fx``/``fy``/``cx``/``cy`` are in pixels."""

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def matrix(self) -> torch.Tensor:
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

    def ray_directions(self) -> torch.Tensor:
        """``[H, W, 3]`` unit-``z`` ray directions, pixel centres at integer coordinates."""
        rows = torch.arange(self.height, dtype=torch.float64)
        cols = torch.arange(self.width, dtype=torch.float64)
        grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")
        x = (grid_u - self.cx) / self.fx
        y = (grid_v - self.cy) / self.fy
        return torch.stack((x, y, torch.ones_like(x)), dim=-1)

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fx_px": self.fx,
            "fy_px": self.fy,
            "cx_px": self.cx,
            "cy_px": self.cy,
            "width": self.width,
            "height": self.height,
            "model": "pinhole_no_distortion",
        }


@dataclass(frozen=True)
class PlaneBody:
    """A rigid rectangular face: local ``x``/``y`` half extents in metres, local ``+z`` normal."""

    name: str
    half_extents: tuple[float, float]
    albedo: tuple[float, float, float]
    texture: str = "checker"
    tex_scale: float = 40.0
    kind: str = "object"


@dataclass
class BodyTrack:
    body: PlaneBody
    #: ``[B, 4, 4]`` world-from-body pose per batch sample.
    T_world_from_body: torch.Tensor
    #: ``[B]``; a body that is not present is physically absent (birth / removal).
    present: torch.Tensor


@dataclass
class WorldBatch:
    """One frame of a batch of independent world states plus the shared camera rig."""

    cameras: tuple[PinholeCamera, ...]
    T_world_from_cam: dict[str, torch.Tensor]
    bodies: tuple[BodyTrack, ...]
    factor: str = ""
    frame: int = 0
    #: Online-reported (possibly delayed or wrong) camera context, ``[B, 3]`` each.
    reported_translation: torch.Tensor | None = None
    reported_rpy_deg: torch.Tensor | None = None
    gain: torch.Tensor | None = None
    noise_sigma: float = 0.0

    @property
    def batch(self) -> int:
        return next(iter(self.T_world_from_cam.values())).shape[0]

    @property
    def body_names(self) -> tuple[str, ...]:
        return tuple(track.body.name for track in self.bodies)

    def with_reports(
        self,
        *,
        reported_translation: torch.Tensor,
        reported_rpy_deg: torch.Tensor,
        gain: torch.Tensor,
        noise_sigma: float,
    ) -> WorldBatch:
        self.reported_translation = reported_translation
        self.reported_rpy_deg = reported_rpy_deg
        self.gain = gain
        self.noise_sigma = noise_sigma
        return self


@dataclass
class RenderResult:
    """Renderer output.  All spatial maps share ``[B, V, H, W]`` layouts."""

    radiance: torch.Tensor  # [B, V, 3, H, W] float32 linear scene radiance (clean)
    depth_m: torch.Tensor  # [B, V, H, W] float32 camera-space z, inf = no surface
    body_index: torch.Tensor  # [B, V, H, W] int16, -1 = no surface
    silhouette: torch.Tensor  # [B, V, N, H, W] bool, pre-occlusion coverage
    camera_names: tuple[str, ...]
    body_names: tuple[str, ...]

    def visibility(self, body_id: int) -> torch.Tensor:
        return self.body_index == body_id


# --------------------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------------------


def _texture_value(body: PlaneBody, local_xy: torch.Tensor) -> torch.Tensor:
    """Procedural texture in ``[0, 1]`` from local face coordinates ``[..., 2]``."""
    x = local_xy[..., 0]
    y = local_xy[..., 1]
    if body.texture == "checker":
        cells = torch.floor(x * body.tex_scale) + torch.floor(y * body.tex_scale)
        return 0.25 + 0.75 * torch.remainder(cells, 2.0)
    if body.texture == "gradient":
        hx, hy = body.half_extents
        return 0.15 + 0.85 * ((x / (2.0 * hx)) + 0.5).clamp(0.0, 1.0) * (
            0.4 + 0.6 * ((y / (2.0 * hy)) + 0.5).clamp(0.0, 1.0)
        )
    if body.texture == "flat":
        return torch.ones_like(x)
    raise ValueError(f"unknown texture {body.texture!r}")


def render_world(world: WorldBatch, *, dtype: torch.dtype = torch.float32) -> RenderResult:
    """Exact per-pixel plane rasterisation with a z-buffer.

    For every body and pixel the ray is intersected with the body's plane analytically,
    the rectangle test uses the local face coordinates, and the nearest positive depth wins.
    Image radiance, depth, instance index and silhouette therefore all come from the same
    world state; visibility is ``body_index == body_id`` plus the z-buffer.
    """
    cameras = world.cameras
    batch = world.batch
    height, width = cameras[0].height, cameras[0].width
    n_bodies = len(world.bodies)
    radiance = torch.zeros(batch, len(cameras), 3, height, width, dtype=torch.float64)
    depth = torch.full((batch, len(cameras), height, width), math.inf, dtype=torch.float64)
    index = torch.full((batch, len(cameras), height, width), -1, dtype=torch.int64)
    silhouette = torch.zeros(batch, len(cameras), n_bodies, height, width, dtype=torch.bool)
    for view, camera in enumerate(cameras):
        rays = camera.ray_directions().reshape(1, height, width, 3)
        T_cam_from_world = invert_transform(world.T_world_from_cam[camera.name])
        for body_id, track in enumerate(world.bodies):
            body = track.body
            T_cam_from_body = T_cam_from_world @ track.T_world_from_body
            rotation = T_cam_from_body[..., :3, :3]
            offset = T_cam_from_body[..., :3, 3]
            normal = rotation[..., 2]
            plane_d = (normal * offset).sum(-1)
            denominator = (rays * normal[:, None, None, :]).sum(-1)
            z = plane_d[:, None, None] / denominator
            points_cam = z[..., None] * rays
            local = torch.einsum("bhwj,bjk->bhwk", points_cam - offset[:, None, None, :], rotation)
            inside = (
                track.present[:, None, None]
                & (z > 1e-6)
                & (denominator.abs() > 1e-12)
                & (local[..., 0].abs() <= body.half_extents[0])
                & (local[..., 1].abs() <= body.half_extents[1])
            )
            silhouette[:, view, body_id] = inside
            closer = inside & (z < depth[:, view])
            depth[:, view] = torch.where(closer, z, depth[:, view])
            index[:, view] = torch.where(
                closer, torch.full_like(index[:, view], body_id), index[:, view]
            )
            shade = _texture_value(body, local[..., :2])
            albedo = torch.tensor(body.albedo, dtype=torch.float64)
            rgb = (shade[..., None] * albedo).permute(0, 3, 1, 2)
            radiance[:, view] = torch.where(closer[:, None], rgb, radiance[:, view])
    return RenderResult(
        radiance=radiance.to(dtype),
        depth_m=depth.to(dtype),
        body_index=index.to(torch.int16),
        silhouette=silhouette,
        camera_names=tuple(camera.name for camera in cameras),
        body_names=world.body_names,
    )


# --------------------------------------------------------------------------------------
# sensor model
# --------------------------------------------------------------------------------------


def cfa_phase_channel_map(pattern: str, height: int, width: int) -> torch.Tensor:
    """``[H, W]`` channel index per pixel for a declared CFA pattern."""
    if pattern not in CFA_PHASE_CHANNELS:
        raise ValueError(f"unknown CFA pattern {pattern!r}")
    phase = CFA_PHASE_CHANNELS[pattern]
    channels = torch.zeros(height, width, dtype=torch.int64)
    for row_phase, row in enumerate(phase):
        for col_phase, channel in enumerate(row):
            channels[row_phase::2, col_phase::2] = channel
    return channels


def apply_cfa(
    radiance: torch.Tensor,
    *,
    pattern: str,
    bit_depth: int,
    black_level_dn: int,
    white_level_dn: int,
) -> torch.Tensor:
    """Sample the declared spectral channel and phase, quantise to DN.

    ``radiance`` is ``[B, V, 3, H, W]`` linear; the result is ``[B, V, H, W]`` uint16.
    """
    if radiance.shape[-3] != 3:
        raise ValueError("CFA sampling expects a three-channel linear radiance stack")
    max_dn = int(2**bit_depth - 1)
    if not 0 <= black_level_dn < white_level_dn <= max_dn:
        raise ValueError("CFA levels must satisfy 0 <= black < white <= 2**bit_depth - 1")
    batch, views, _, height, width = radiance.shape
    channels = cfa_phase_channel_map(pattern, height, width)
    gathered = radiance.gather(
        2, channels.reshape(1, 1, 1, height, width).expand(batch, views, 1, height, width)
    )
    value = gathered.clamp(0.0, 1.0)
    dn = black_level_dn + value * float(white_level_dn - black_level_dn)
    return torch.round(dn).to(torch.uint16).squeeze(2)


@dataclass
class Measurement:
    rgb3: torch.Tensor  # [B, V, 3, H, W] float32, linear
    cfa_dn: torch.Tensor  # [B, V, H, W] uint16
    clip_mask: torch.Tensor  # [B, V, H, W] bool, sensor clipped at/above white
    gain: torch.Tensor  # [B]


def form_measurement(
    radiance: torch.Tensor,
    *,
    gain: torch.Tensor,
    noise_sigma: float,
    generator: torch.Generator | None = None,
    cfa_pattern: str = "GRBG",
    cfa_bit_depth: int = 12,
    cfa_black_level_dn: int = 64,
    cfa_white_level_dn: int = 4095,
) -> Measurement:
    """Apply exposure gain, read noise, clipping and the declared CFA sampling."""
    batch = radiance.shape[0]
    gained = radiance * gain.reshape(batch, 1, 1, 1, 1)
    if noise_sigma > 0.0:
        noise = torch.randn(gained.shape, generator=generator, dtype=gained.dtype) * noise_sigma
        gained = gained + noise
    clip = gained >= 1.0
    clipped = gained.clamp(0.0, 1.0)
    cfa = apply_cfa(
        clipped,
        pattern=cfa_pattern,
        bit_depth=cfa_bit_depth,
        black_level_dn=cfa_black_level_dn,
        white_level_dn=cfa_white_level_dn,
    )
    return Measurement(
        rgb3=clipped.to(torch.float32),
        cfa_dn=cfa,
        clip_mask=clip.any(dim=2) if clip.dim() == 5 else clip,
        gain=gain,
    )


# --------------------------------------------------------------------------------------
# factor plans
# --------------------------------------------------------------------------------------


@dataclass
class FactorPlan:
    """Resolved factor settings for one generated sequence."""

    name: str
    components: tuple[str, ...]
    camera_motion: str = "none"  # none | translate | full
    object_motion: str = "none"  # none | sweep | stop_reverse
    occluder_sweep: bool = False
    robot_visible: bool = False
    birth_death: bool = False
    gain: bool = False
    noise: bool = False
    time_shift: int = 0
    wrong_association: bool = False
    full_view_plane: bool = False

    @property
    def is_combination(self) -> bool:
        return len(self.components) > 1


#: Per-component settings; a combination merges components in listed order and the first
#: component that sets a field wins (``setdefault`` semantics).
COMPONENT_PLANS: dict[str, dict[str, Any]] = {
    "static": {},
    "camera_only": {"camera_motion": "full"},
    "object_only": {"object_motion": "sweep", "birth_death": True},
    "both": {"camera_motion": "full", "object_motion": "sweep"},
    "stop_reverse": {"object_motion": "stop_reverse"},
    "occlusion_recovery": {"occluder_sweep": True},
    "robot_self_occlusion": {"robot_visible": True, "camera_motion": "translate"},
    "gain": {"gain": True},
    "noise": {"noise": True},
    "time_shift": {"time_shift": 3, "camera_motion": "full"},
    "wrong_association": {
        "wrong_association": True,
        "object_motion": "sweep",
        "camera_motion": "translate",
    },
    "unidentifiable_full_view_plane": {"full_view_plane": True, "object_motion": "sweep"},
}


def resolve_plan(factor: str) -> FactorPlan:
    """Resolve a factor name (single component or ``a+b+c`` combination) into a plan."""
    components = tuple(factor.split("+"))
    for component in components:
        if component not in COMPONENT_PLANS:
            raise ValueError(f"unknown D0 factor component {component!r}")
    merged: dict[str, Any] = {}
    for component in components:
        for key, value in COMPONENT_PLANS[component].items():
            merged.setdefault(key, value)
    return FactorPlan(name=factor, components=components, **merged)


def default_factor_names() -> tuple[str, ...]:
    combinations = tuple("+".join(parts) for parts in DEFAULT_COMBINATIONS)
    return DEFAULT_FACTORS + combinations


# --------------------------------------------------------------------------------------
# world construction
# --------------------------------------------------------------------------------------


def default_cameras(width: int, height: int, names: Sequence[str]) -> tuple[PinholeCamera, ...]:
    """Declared diagnostic pinhole rig (not a calibration).

    Focal lengths scale with the resolution so the field of view, and therefore every world
    coordinate in metres, is independent of the chosen image size.
    """
    scale_x = width / 128.0
    scale_y = height / 96.0
    presets = (
        PinholeCamera(
            "cam0",
            110.0 * scale_x,
            110.0 * scale_y,
            (width - 1) / 2.0,
            (height - 1) / 2.0,
            width,
            height,
        ),
        PinholeCamera(
            "cam1",
            95.0 * scale_x,
            100.0 * scale_y,
            width / 2.0,
            height / 2.0,
            width,
            height,
        ),
    )
    cameras = []
    for index, name in enumerate(names):
        preset = presets[index % len(presets)]
        cameras.append(replace(preset, name=name))
    return tuple(cameras)


@dataclass(frozen=True)
class D0Config:
    """D0 generation settings resolved from ``mechanism_data.d0`` of the S0 config."""

    height: int = 96
    width: int = 128
    depths_m: tuple[float, ...] = (0.35, 0.6, 1.2)
    objects: int = 3
    frames: int = 24
    cameras: tuple[str, ...] = ("cam0", "cam1")
    factors: tuple[str, ...] = DEFAULT_FACTORS
    cfa_pattern: str = "GRBG"
    cfa_bit_depth: int = 12
    cfa_black_level_dn: int = 64
    cfa_white_level_dn: int = 4095
    seed: int = 20261014
    batch: int = 2

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> D0Config:
        mechanism = config.get("mechanism_data", {}).get("d0", {})
        cfa = mechanism.get("cfa", {})
        resolution = mechanism.get("resolution", [96, 128])
        values: dict[str, Any] = {
            "height": int(resolution[0]),
            "width": int(resolution[1]),
            "depths_m": tuple(float(v) for v in mechanism.get("depths_m", (0.35, 0.6, 1.2))),
            "objects": int(mechanism.get("objects", 3)),
            "frames": int(mechanism.get("frames", 24)),
            "cameras": tuple(str(v) for v in mechanism.get("cameras", ("cam0", "cam1"))),
            "factors": tuple(str(v) for v in mechanism.get("factors", DEFAULT_FACTORS)),
            "cfa_pattern": str(cfa.get("pattern", "GRBG")),
            "cfa_bit_depth": int(cfa.get("bit_depth", 12)),
            "cfa_black_level_dn": int(cfa.get("black_level_dn", 64)),
            "cfa_white_level_dn": int(cfa.get("white_level_dn", 4095)),
            "seed": int(config.get("split_seed", 20261014)),
        }
        values.update(overrides)
        return cls(**values)

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["depths_m"] = list(self.depths_m)
        record["cameras"] = list(self.cameras)
        record["factors"] = list(self.factors)
        record["combinations"] = ["+".join(parts) for parts in DEFAULT_COMBINATIONS]
        return record


def _object_half_extents(index: int) -> tuple[float, float]:
    return (0.055 + 0.005 * index, 0.045 + 0.004 * index)


def _object_albedo(index: int) -> tuple[float, float, float]:
    palette = ((1.0, 0.30, 0.22), (0.25, 0.90, 0.35), (0.30, 0.45, 1.0), (0.95, 0.85, 0.20))
    return palette[index % len(palette)]


BACKGROUND_BODY = PlaneBody(
    name="background",
    half_extents=(1.6, 1.2),
    albedo=(0.72, 0.70, 0.68),
    texture="checker",
    tex_scale=4.0,
    kind="background",
)
ROBOT_BODY = PlaneBody(
    name="robot",
    half_extents=(0.12, 0.08),
    albedo=(0.55, 0.56, 0.62),
    texture="gradient",
    tex_scale=12.0,
    kind="robot",
)
PLANE_BODY = PlaneBody(
    name="plane",
    half_extents=(1.5, 1.2),
    albedo=(0.80, 0.62, 0.45),
    texture="checker",
    tex_scale=6.0,
    kind="plane",
)


def _stop_reverse_progress(steps: torch.Tensor, hold_start: int, hold_end: int) -> torch.Tensor:
    """Advance, hold, then reverse along the same axis."""
    forward = steps.clamp(max=hold_start)
    reversed_tail = hold_start - (steps - hold_end + 1)
    return torch.where(steps < hold_end, forward, reversed_tail)


def build_factor_worlds(
    plan: FactorPlan,
    config: D0Config,
    *,
    batch: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> tuple[list[WorldBatch], dict[str, Any]]:
    """Generate the world batch of every frame of one factor plus its declared metadata."""
    batch = config.batch if batch is None else batch
    frames = config.frames if frames is None else frames
    seed = config.seed if seed is None else seed
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cameras = default_cameras(config.width, config.height, config.cameras)

    # Per-sample randomisation: direction/speed magnitudes and object placement jitter.
    cam_direction = torch.randn(batch, 3, generator=generator, dtype=torch.float64)
    cam_direction[:, 2] = cam_direction[:, 2] * 0.15
    cam_direction = cam_direction / cam_direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cam_speed = 0.008 + 0.006 * torch.rand(batch, generator=generator, dtype=torch.float64)
    rot_rate = (2.0 * torch.rand(batch, 3, generator=generator, dtype=torch.float64) - 1.0) * 0.45
    object_direction = torch.randn(batch, 2, generator=generator, dtype=torch.float64)
    object_direction = object_direction / object_direction.norm(dim=-1, keepdim=True).clamp_min(
        1e-9
    )
    object_speed = 0.012 + 0.010 * torch.rand(batch, generator=generator, dtype=torch.float64)
    jitter = (
        torch.rand(batch, config.objects, 2, generator=generator, dtype=torch.float64) - 0.5
    ) * 0.10
    gain_base = 1.0 + 0.10 * torch.rand(batch, generator=generator, dtype=torch.float64)

    n_objects = config.objects
    spaced = -0.22 + torch.arange(n_objects, dtype=torch.float64) * (0.44 / max(1, n_objects - 1))
    base_xy = torch.zeros(n_objects, 2, dtype=torch.float64)
    base_xy[:, 0] = spaced
    base_xy[:, 1] = torch.tensor([0.02, -0.04, 0.03, -0.02][:n_objects], dtype=torch.float64)
    depths = torch.tensor(config.depths_m[:n_objects], dtype=torch.float64)

    object_bodies = []
    for index in range(n_objects):
        # The occlusion factor needs an occluder that fully covers the far object at every
        # probed depth, so object_0 is enlarged and vertically aligned with the far objects.
        half_extents = (
            (0.09, 0.075) if plan.occluder_sweep and index == 0 else _object_half_extents(index)
        )
        object_bodies.append(
            PlaneBody(
                name=f"object_{index}",
                half_extents=half_extents,
                albedo=_object_albedo(index),
                texture="checker",
                tex_scale=45.0 if half_extents[0] < 0.08 else 28.0,
                kind="object",
            )
        )
    bodies = [BACKGROUND_BODY] + object_bodies
    if plan.robot_visible:
        bodies.append(ROBOT_BODY)
    if plan.full_view_plane:
        bodies = [PLANE_BODY]

    worlds: list[WorldBatch] = []
    events: list[dict[str, Any]] = []
    for frame in range(frames):
        steps = torch.full((batch,), float(frame), dtype=torch.float64)
        if plan.camera_motion == "none":
            cam_translation = torch.zeros(batch, 3, dtype=torch.float64)
            cam_rpy = torch.zeros(batch, 3, dtype=torch.float64)
        elif plan.camera_motion == "translate":
            cam_translation = cam_direction * (cam_speed * steps).reshape(batch, 1)
            cam_rpy = torch.zeros(batch, 3, dtype=torch.float64)
        else:
            cam_translation = cam_direction * (cam_speed * steps).reshape(batch, 1)
            cam_rpy = rot_rate * steps.reshape(batch, 1)
        # Keep the moving camera and the scene inside each other's field of view.
        cam_rpy = cam_rpy.clamp(-4.0, 4.0)

        if plan.object_motion == "none":
            object_progress = torch.zeros(batch, dtype=torch.float64)
        elif plan.object_motion == "stop_reverse":
            object_progress = _stop_reverse_progress(
                steps, hold_start=frames // 3, hold_end=2 * frames // 3
            )
        else:
            object_progress = steps

        tracks: list[BodyTrack] = []
        for body_id, body in enumerate(bodies):
            if body.kind == "background" or body.kind == "plane":
                if plan.full_view_plane:
                    centre = torch.zeros(batch, 3, dtype=torch.float64)
                    centre[:, 0] = 0.010 * object_progress
                    centre[:, 2] = 0.6
                else:
                    centre = torch.zeros(batch, 3, dtype=torch.float64)
                    centre[:, 2] = 1.4
                present = torch.ones(batch, dtype=torch.bool)
            elif body.kind == "robot":
                centre = torch.zeros(batch, 3, dtype=torch.float64)
                centre[:, 0] = -0.06 + 0.004 * steps
                centre[:, 1] = 0.10
                centre[:, 2] = 0.22
                present = torch.ones(batch, dtype=torch.bool)
            else:
                object_index = body_id - 1
                centre = torch.zeros(batch, 3, dtype=torch.float64)
                xy = base_xy[object_index][None, :] + jitter[:, object_index, :]
                centre[:, :2] = xy
                centre[:, 2] = depths[object_index]
                if plan.occluder_sweep and object_index == 0:
                    centre[:, 0] = -0.45 + 0.038 * steps
                    centre[:, 1] = 0.02
                    centre[:, 2] = 0.35
                elif plan.object_motion in {"sweep", "stop_reverse"} and object_index == 0:
                    centre[:, :2] = centre[:, :2] + object_direction * (
                        object_speed * object_progress
                    ).reshape(batch, 1)
                present = torch.ones(batch, dtype=torch.bool)
                if plan.birth_death and object_index == 1 and frames > 6:
                    present = present & (steps >= 6.0)
                if plan.birth_death and object_index == 2 and frames > 6:
                    present = present & (steps < frames - 6.0)
            rotation = rotation_rpy_deg(
                torch.full((batch,), 180.0, dtype=torch.float64),
                torch.zeros(batch, dtype=torch.float64),
                torch.zeros(batch, dtype=torch.float64),
            ).clone()
            tracks.append(
                BodyTrack(
                    body=body,
                    T_world_from_body=transform_from_rt(rotation, centre),
                    present=present,
                )
            )

        T_world_from_cam = {}
        for camera_index, camera in enumerate(cameras):
            offset = torch.zeros(batch, 3, dtype=torch.float64)
            if camera_index > 0:
                offset = (
                    torch.tensor([0.05, 0.0, -0.03], dtype=torch.float64).expand(batch, -1).clone()
                )
            rotation = rotation_rpy_deg(
                cam_rpy[:, 0] + (2.0 if camera_index else 0.0),
                cam_rpy[:, 1],
                cam_rpy[:, 2],
            )
            T_world_from_cam[camera.name] = transform_from_rt(rotation, cam_translation + offset)

        if plan.gain:
            gain = gain_base * torch.where(
                steps >= frames / 2.0,
                torch.full_like(steps, 1.6),
                torch.ones_like(steps),
            )
        else:
            gain = gain_base * 1.0

        delay = float(plan.time_shift)
        delayed = (steps - delay).clamp(min=0.0)
        if plan.camera_motion == "none":
            reported_translation = torch.zeros(batch, 3, dtype=torch.float64)
            reported_rpy = torch.zeros(batch, 3, dtype=torch.float64)
        else:
            if plan.camera_motion == "translate":
                reported_translation = cam_direction * (cam_speed * delayed).reshape(batch, 1)
                reported_rpy = torch.zeros(batch, 3, dtype=torch.float64)
            else:
                reported_translation = cam_direction * (cam_speed * delayed).reshape(batch, 1)
                reported_rpy = (rot_rate * delayed.reshape(batch, 1)).clamp(-4.0, 4.0)

        noise_sigma = 0.02 if plan.noise else 0.0
        world = WorldBatch(
            cameras=cameras,
            T_world_from_cam=T_world_from_cam,
            bodies=tuple(tracks),
            factor=plan.name,
            frame=frame,
        ).with_reports(
            reported_translation=reported_translation,
            reported_rpy_deg=reported_rpy,
            gain=gain,
            noise_sigma=noise_sigma,
        )
        worlds.append(world)
        events.append({"frame": frame, "gain": gain.detach().tolist()})

    metadata = {
        "factor": plan.name,
        "components": list(plan.components),
        "is_combination": plan.is_combination,
        "camera_motion": plan.camera_motion,
        "object_motion": plan.object_motion,
        "occluder_sweep": plan.occluder_sweep,
        "robot_visible": plan.robot_visible,
        "birth_death": plan.birth_death,
        "gain_step": plan.gain,
        "noise_sigma": 0.02 if plan.noise else 0.0,
        "time_shift_frames": plan.time_shift,
        "wrong_association": plan.wrong_association,
        "full_view_plane": plan.full_view_plane,
    }
    return worlds, metadata


# --------------------------------------------------------------------------------------
# labels derived from the same world state
# --------------------------------------------------------------------------------------


def frame_labels(world: WorldBatch, render: RenderResult) -> dict[str, Any]:
    """Per-sample labels of one frame, all derived from the rendered z-buffer and state."""
    batch = world.batch
    labels: dict[str, Any] = {
        "camera_translation": {},
        "camera_rpy_deg": {},
        "bodies": {},
    }
    for camera in world.cameras:
        T_world_from_cam = world.T_world_from_cam[camera.name]
        labels["camera_translation"][camera.name] = T_world_from_cam[:, :3, 3].clone()
        rotation = T_world_from_cam[:, :3, :3]
        labels["camera_rpy_deg"][camera.name] = _rotation_to_rpy_deg(rotation)
    for body_id, track in enumerate(world.bodies):
        name = track.body.name
        entry: dict[str, Any] = {
            "world_position": track.T_world_from_body[:, :3, 3].clone(),
            "present": track.present.clone(),
        }
        visible = torch.zeros(batch, len(world.cameras), dtype=torch.long)
        covered = torch.zeros(batch, len(world.cameras), dtype=torch.long)
        centre_px = torch.full((batch, len(world.cameras), 2), float("nan"), dtype=torch.float64)
        centre_depth = torch.full((batch, len(world.cameras)), float("nan"), dtype=torch.float64)
        for view, camera in enumerate(world.cameras):
            index_map = render.body_index[:, view]
            visible[:, view] = (index_map == body_id).flatten(1).sum(dim=1)
            covered[:, view] = render.silhouette[:, view, body_id].flatten(1).sum(dim=1)
            world_point = track.T_world_from_body[:, :3, 3]
            T_cam_from_world = invert_transform(world.T_world_from_cam[camera.name])
            point_cam = (
                T_cam_from_world
                @ torch.cat((world_point, torch.ones(batch, 1, dtype=torch.float64)), dim=1)[
                    ..., None
                ]
            )[..., 0][:, :3]
            u, v = project_points(camera, point_cam)
            centre_px[:, view, 0] = u
            centre_px[:, view, 1] = v
            centre_depth[:, view] = point_cam[:, 2]
        entry["visible_pixels"] = visible
        entry["covered_pixels"] = covered
        entry["centre_px"] = centre_px
        entry["centre_depth_m"] = centre_depth
        labels["bodies"][name] = entry
    return labels


def _rotation_to_rpy_deg(rotation: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rotation_rpy_deg` for the small diagnostic angles used here."""
    sy = -rotation[..., 2, 0]
    pitch = torch.asin(sy.clamp(-1.0, 1.0))
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    return torch.stack((torch.rad2deg(roll), torch.rad2deg(pitch), torch.rad2deg(yaw)), dim=-1)


def motion_phase(positions: torch.Tensor) -> torch.Tensor:
    """``[T, B, 3]`` positions -> ``[T, B]``: -1 reverse, 0 stopped/unknown, 1 forward."""
    if positions.shape[0] < 3:
        return torch.zeros(positions.shape[:2], dtype=torch.int8)
    velocity = positions[1:] - positions[:-1]  # [T-1, B, 3]
    frames, batch = positions.shape[:2]
    phases = torch.zeros(frames, batch, dtype=torch.int8)
    # Each moving frame is compared with the previous *moving* frame, so a stop between two
    # opposite segments does not hide the reversal.
    for sample in range(batch):
        previous: torch.Tensor | None = None
        for t in range(frames - 1):
            step = velocity[t, sample]
            if float(step.norm()) <= 1e-9:
                phases[t + 1, sample] = 0
                continue
            if previous is not None and float((step * previous).sum()) < 0.0:
                phases[t + 1, sample] = -1
            else:
                phases[t + 1, sample] = 1
            previous = step
    return phases


# --------------------------------------------------------------------------------------
# independent reference projection (audit path)
# --------------------------------------------------------------------------------------


def reference_visibility(
    world: WorldBatch,
    camera: PinholeCamera,
    *,
    grid: int | None = None,
    max_grid: int = 257,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Independent point-splat label path.

    Samples each body's rectangular face on a dense local grid, projects the points and keeps
    the nearest depth per pixel.  Returns ``(index [B, H, W], depth [B, H, W], count [B, H, W],
    info)``.  ``grid=None`` picks a per-body grid whose image-space spacing is at most half a
    pixel (capped by ``max_grid``), so the independent path can both confirm and reject every
    rendered silhouette pixel.  This path shares no rasterisation code with :func:`render_world`.
    """
    batch = world.batch
    height, width = camera.height, camera.width
    best_depth = torch.full((batch, height, width), math.inf, dtype=torch.float64)
    best_index = torch.full((batch, height, width), -1, dtype=torch.int64)
    counts = torch.zeros(batch, height, width, dtype=torch.int64)
    coverage = torch.zeros(batch, len(world.bodies), height, width, dtype=torch.bool)
    T_cam_from_world = invert_transform(world.T_world_from_cam[camera.name])
    grid_steps: dict[str, int] = {}
    for body_id, track in enumerate(world.bodies):
        half = torch.tensor(track.body.half_extents, dtype=torch.float64)
        steps = (
            grid if grid is not None else _grid_for_body(track, camera, T_cam_from_world, max_grid)
        )
        grid_steps[track.body.name] = steps
        axis = torch.linspace(-1.0, 1.0, steps, dtype=torch.float64)
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
        flat = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)  # [K, 2]
        local = flat * half  # [K, 2]
        zeros = torch.zeros(local.shape[0], 1, dtype=torch.float64)
        ones = torch.ones(local.shape[0], 1, dtype=torch.float64)
        local_h = torch.cat((local, zeros, ones), dim=1)  # [K, 4]
        points_cam = torch.einsum(
            "bij,kj->bki", T_cam_from_world @ track.T_world_from_body, local_h
        )
        u, v = project_points(camera, points_cam)
        valid = (
            track.present[:, None]
            & torch.isfinite(u)
            & torch.isfinite(v)
            & (points_cam[..., 2] > 1e-6)
            & (u >= -0.5)
            & (u <= width - 0.5)
            & (v >= -0.5)
            & (v <= height - 0.5)
        )
        pixel_u = u.round().clamp(0, width - 1).to(torch.int64)
        pixel_v = v.round().clamp(0, height - 1).to(torch.int64)
        flat_pixel = (pixel_v * width + pixel_u).reshape(batch, -1)
        depth_flat = points_cam[..., 2].reshape(batch, -1)
        valid_flat = valid.reshape(batch, -1)
        depth_flat = torch.where(valid_flat, depth_flat, torch.full_like(depth_flat, math.inf))
        candidate = torch.full((batch, height * width), math.inf, dtype=torch.float64)
        candidate.scatter_reduce_(1, flat_pixel, depth_flat, reduce="amin", include_self=True)
        coverage[:, body_id] = (candidate < math.inf).reshape(batch, height, width)
        hit = torch.isfinite(candidate)
        closer = hit & (candidate < best_depth.reshape(batch, -1))
        best_depth = torch.where(closer, candidate, best_depth.reshape(batch, -1)).reshape(
            batch, height, width
        )
        best_index = torch.where(
            closer,
            torch.full_like(best_index.reshape(batch, -1), body_id),
            best_index.reshape(batch, -1),
        ).reshape(batch, height, width)
        counts += (candidate.reshape(batch, height, width) < math.inf).to(torch.int64)
    return (
        best_index,
        best_depth,
        counts,
        coverage,
        {"grid_steps_per_body": grid_steps, "max_grid": max_grid},
    )


def _grid_for_body(
    track: BodyTrack,
    camera: PinholeCamera,
    T_cam_from_world: torch.Tensor,
    max_grid: int,
) -> int:
    """Grid steps so adjacent samples are at most half an image pixel apart."""
    half_x, half_y = track.body.half_extents
    corners = torch.tensor(
        [
            [-half_x, -half_y, 0.0, 1.0],
            [half_x, -half_y, 0.0, 1.0],
            [-half_x, half_y, 0.0, 1.0],
            [half_x, half_y, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    points_cam = torch.einsum("bij,kj->bki", T_cam_from_world @ track.T_world_from_body, corners)
    u, v = project_points(camera, points_cam)
    finite = torch.isfinite(u) & torch.isfinite(v) & (points_cam[..., 2] > 1e-6)
    if not bool(finite.any()):
        return 17
    u_values = torch.where(finite, u, torch.full_like(u, float("nan")))
    v_values = torch.where(finite, v, torch.full_like(v, float("nan")))
    span_u = float(
        torch.nan_to_num(u_values, nan=float("-inf")).max()
        - torch.nan_to_num(u_values, nan=float("inf")).min()
    )
    span_v = float(
        torch.nan_to_num(v_values, nan=float("-inf")).max()
        - torch.nan_to_num(v_values, nan=float("inf")).min()
    )
    extent = max(span_u, span_v)
    extent = min(extent, 4.0 * max(camera.width, camera.height))
    return int(min(max_grid, max(17, math.ceil(2.0 * extent) + 8)))


def _dilate(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    grown = F.max_pool2d(mask.float().unsqueeze(1), kernel, stride=1, padding=radius)
    return grown.squeeze(1) > 0.0


def compare_label_maps(
    rendered_index: torch.Tensor,
    reference_index: torch.Tensor,
    *,
    coverage: torch.Tensor | None = None,
    radius: int = 1,
) -> dict[str, torch.Tensor]:
    """Per-(sample, body) agreement between a label map and the independent path.

    ``unexplained`` counts label pixels the independent path never projects the claimed body
    to (cross-sample contamination, or a label wider than its own world state); it uses the
    per-body projection *coverage* so that one-pixel silhouette-boundary rounding cannot
    create false positives.  ``missed`` counts independent winner pixels the label omits.
    ``sym_diff`` and ``perimeter_ref`` bound the mask-size difference: point splatting marks
    every pixel cell that contains a surface point, which is up to one pixel wider than the
    centre-sampled renderer at a boundary, so the allowed difference scales with the boundary
    length instead of the area.  ``iou`` is reported for reference only.
    """
    body_ids = sorted(
        set(rendered_index.flatten().tolist()) | set(reference_index.flatten().tolist())
    )
    body_ids = [int(v) for v in body_ids if int(v) >= 0]
    stats: dict[str, list[torch.Tensor]] = {
        "unexplained": [],
        "missed": [],
        "iou": [],
        "sym_diff": [],
        "perimeter_ref": [],
    }
    for body_id in body_ids:
        rendered = rendered_index == body_id
        reference = reference_index == body_id
        nearby = reference if coverage is None else coverage[:, body_id]
        unexplained = rendered & ~_dilate(nearby, radius)
        missed = reference & ~_dilate(rendered, radius)
        union = (rendered | reference).sum(dim=(-1, -2)).clamp_min(1)
        intersection = (rendered & reference).sum(dim=(-1, -2))
        sym_diff = (rendered ^ reference).sum(dim=(-1, -2))
        perimeter = (_dilate(reference, 1) & ~reference).sum(dim=(-1, -2))
        stats["unexplained"].append(unexplained.sum(dim=(-1, -2)))
        stats["missed"].append(missed.sum(dim=(-1, -2)))
        stats["iou"].append(intersection / union)
        stats["sym_diff"].append(sym_diff)
        stats["perimeter_ref"].append(perimeter)
    if not body_ids:
        zeros = torch.zeros(rendered_index.shape[0], dtype=torch.int64)
        ones = torch.ones(rendered_index.shape[0], dtype=torch.float64)
        return {
            "unexplained": zeros,
            "missed": zeros,
            "iou": ones,
            "sym_diff": zeros,
            "perimeter_ref": zeros,
        }
    return {name: torch.stack(value, dim=-1) for name, value in stats.items()}


def _near_texture_boundary(body: PlaneBody, local_xy: torch.Tensor, eps: float) -> torch.Tensor:
    """Pixels within ``eps`` metres of a checker cell edge, where the texture is discontinuous."""
    if body.texture != "checker":
        return torch.zeros(local_xy.shape[:-1], dtype=torch.bool)
    x = local_xy[..., 0] * body.tex_scale
    y = local_xy[..., 1] * body.tex_scale
    distance_x = (x - x.round()).abs() / body.tex_scale
    distance_y = (y - y.round()).abs() / body.tex_scale
    return torch.minimum(distance_x, distance_y) < eps


def image_label_consistency(
    world: WorldBatch,
    camera: PinholeCamera,
    render: RenderResult,
    *,
    stride: int = 2,
    tolerance: float = 1e-5,
    inside_tolerance_m: float = 1e-6,
    texture_boundary_eps_m: float = 1e-5,
) -> dict[str, Any]:
    """Recompute radiance/depth from the world state for the body each pixel claims.

    This validates image <-> label <-> depth <-> world state on the same frame.  Pixels within
    ``texture_boundary_eps_m`` of a checker cell edge are counted separately instead of being
    compared exactly, because the reconstructed local coordinate of a boundary pixel can cross
    a discontinuity of the procedural texture.
    """
    view = render.camera_names.index(camera.name)
    rays = camera.ray_directions()
    depth = render.depth_m[:, view]
    index = render.body_index[:, view]
    radiance = render.radiance[:, view]
    worst_depth = 0.0
    worst_radiance = 0.0
    checked = 0
    worst_outside_margin = 0.0
    skipped_texture_boundary = 0
    for body_id, track in enumerate(world.bodies):
        T_cam_from_body = (
            invert_transform(world.T_world_from_cam[camera.name]) @ track.T_world_from_body
        )
        rotation = T_cam_from_body[:, :3, :3]
        offset = T_cam_from_body[:, :3, 3]
        for sample in range(world.batch):
            mask = (index[sample] == body_id)[::stride, ::stride]
            if not bool(mask.any()):
                continue
            z = depth[sample][::stride, ::stride][mask]
            ray = rays[::stride, ::stride][mask]
            points_cam = z[:, None] * ray
            local = (points_cam - offset[sample]) @ rotation[sample]
            margin = torch.maximum(
                local[:, 0].abs() - track.body.half_extents[0],
                local[:, 1].abs() - track.body.half_extents[1],
            )
            worst_outside_margin = max(worst_outside_margin, float(margin.max()))
            if not bool((margin <= inside_tolerance_m).all()):
                raise AssertionError(
                    f"label claims body {track.body.name} for pixels outside its face "
                    f"(margin {float(margin.max()):.3e} m)"
                )
            shade = _texture_value(track.body, local[:, :2])
            albedo = torch.tensor(track.body.albedo, dtype=torch.float64)
            expected = (shade[:, None] * albedo).T
            observed = radiance[sample][:, ::stride, ::stride][:, mask]
            near_boundary = _near_texture_boundary(track.body, local[:, :2], texture_boundary_eps_m)
            skipped_texture_boundary += int(near_boundary.sum())
            comparable = ~near_boundary
            if bool(comparable.any()):
                worst_radiance = max(
                    worst_radiance,
                    float(
                        (expected[:, comparable] - observed[:, comparable].to(torch.float64))
                        .abs()
                        .max()
                    ),
                )
            normal = rotation[sample][:, 2]
            plane_d = float((normal * offset[sample]).sum())
            denominator = (ray * normal).sum(-1)
            z_expected = plane_d / denominator
            worst_depth = max(worst_depth, float((z_expected - z).abs().max()))
            checked += int(comparable.sum())
    return {
        "pixels_checked": checked,
        "pixels_skipped_texture_boundary": skipped_texture_boundary,
        "max_radiance_error": worst_radiance,
        "max_depth_error": worst_depth,
        "max_outside_face_margin_m": worst_outside_margin,
        "tolerance": tolerance,
        "inside_tolerance_m": inside_tolerance_m,
    }


# --------------------------------------------------------------------------------------
# per-sample audit
# --------------------------------------------------------------------------------------


#: Audit objects are smaller than the factor objects so per-sample placements stay disjoint
#: at every rendered depth (a contamination measurement needs disjoint references).
AUDIT_OBJECT_HALF_EXTENTS: tuple[float, float] = (0.035, 0.030)


def _audit_world(
    batches: int,
    objects: int,
    config: D0Config,
    *,
    seed: int,
    cameras: Sequence[PinholeCamera],
    perturbation_index: int | None = None,
) -> WorldBatch:
    """Audit scene: disjoint per-sample placements so contamination is measurable."""
    bodies = [BACKGROUND_BODY]
    for index in range(objects):
        bodies.append(
            PlaneBody(
                name=f"object_{index}",
                half_extents=AUDIT_OBJECT_HALF_EXTENTS,
                albedo=_object_albedo(index),
                texture="checker",
                tex_scale=70.0,
                kind="object",
            )
        )
    if batches > 1:
        band = -0.21 + 0.42 * torch.arange(batches, dtype=torch.float64) / (batches - 1)
    else:
        band = torch.zeros(1, dtype=torch.float64)
    tracks: list[BodyTrack] = []
    for body_id, body in enumerate(bodies):
        rotation = rotation_rpy_deg(
            torch.full((batches,), 180.0, dtype=torch.float64),
            torch.zeros(batches, dtype=torch.float64),
            torch.zeros(batches, dtype=torch.float64),
        )
        centre = torch.zeros(batches, 3, dtype=torch.float64)
        if body.kind == "background":
            centre[:, 2] = 1.4
        else:
            object_index = body_id - 1
            centre[:, 0] = band
            centre[:, 1] = 0.10 - 0.10 * object_index
            centre[:, 2] = 0.6 + 0.25 * object_index
            if perturbation_index is not None:
                centre = centre.clone()
                centre[perturbation_index, 0] += 0.05
        tracks.append(
            BodyTrack(
                body=body,
                T_world_from_body=transform_from_rt(rotation, centre),
                present=torch.ones(batches, dtype=torch.bool),
            )
        )
    T_world_from_cam = {}
    for view, camera in enumerate(cameras):
        offset = torch.zeros(batches, 3, dtype=torch.float64)
        if view > 0:
            offset[:, 0] = 0.05
            offset[:, 2] = -0.03
        rotation = rotation_rpy_deg(
            torch.zeros(batches, dtype=torch.float64),
            torch.zeros(batches, dtype=torch.float64),
            torch.full((batches,), 3.0 * view, dtype=torch.float64),
        )
        T_world_from_cam[camera.name] = transform_from_rt(rotation, offset)
    return WorldBatch(
        cameras=tuple(cameras),
        T_world_from_cam=T_world_from_cam,
        bodies=tuple(tracks),
        factor="audit",
    )


def _buggy_merged_labels(
    world: WorldBatch,
    camera: PinholeCamera,
    *,
    grid: int = 129,
) -> torch.Tensor:
    """Reproduce the historical ``[B,1,2] + [B,2] -> [B,B,2]`` label defect.

    The defect added a ``[B, 2]`` translation directly to ``[B, 1, 2]`` base centres, so the
    label of sample ``b`` unioned the object positions of **every** sample in the batch.
    Here sample ``b``'s label is rebuilt as the union of the projected faces of sample
    ``j`` for every ``j``, which is exactly what the broadcast produced.
    """
    batch = world.batch
    height, width = camera.height, camera.width
    merged = torch.full((batch, height, width), -1, dtype=torch.int64)
    axis = torch.linspace(-1.0, 1.0, grid, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    flat = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    T_cam_from_world = invert_transform(world.T_world_from_cam[camera.name])
    for body_id, track in enumerate(world.bodies):
        if track.body.kind != "object":
            continue
        half = torch.tensor(track.body.half_extents, dtype=torch.float64)
        local = flat * half
        local_h = torch.cat(
            (
                local,
                torch.zeros(local.shape[0], 1, dtype=torch.float64),
                torch.ones(local.shape[0], 1, dtype=torch.float64),
            ),
            dim=1,
        )
        for source in range(batch):
            T_cam_from_body = (T_cam_from_world @ track.T_world_from_body)[source]
            points_cam = (T_cam_from_body @ local_h.T).T[:, :3].reshape(1, -1, 3)
            u, v = project_points(camera, points_cam)
            valid = (
                (points_cam[..., 2] > 1e-6)
                & (u >= -0.5)
                & (u <= width - 0.5)
                & (v >= -0.5)
                & (v <= height - 0.5)
            )
            pixel_u = u.round().clamp(0, width - 1).to(torch.int64)
            pixel_v = v.round().clamp(0, height - 1).to(torch.int64)
            hit = (pixel_v * width + pixel_u)[0][valid[0]]
            if hit.numel():
                for target in range(batch):
                    merged[target].view(-1)[hit] = body_id
    return merged


def audit_d0(
    config: D0Config,
    *,
    batch_sizes: Sequence[int] = (1, 2, 4),
    grid: int | None = None,
) -> dict[str, Any]:
    """Per-sample label audit: independent projection, isolation, permutation, counterexample.

    ``grid=None`` uses a per-body adaptive reference grid (at most half a pixel of spacing);
    an integer forces a fixed grid, which is only used by focused tests.
    """
    cameras = default_cameras(config.width, config.height, config.cameras)
    report: dict[str, Any] = {
        "schema": SCHEMA_AUDIT,
        "grid": "adaptive" if grid is None else grid,
        "cameras": [camera.as_record() for camera in cameras],
        "batch_sizes": list(batch_sizes),
        "checks": {},
        "failures": [],
    }
    checks: dict[str, Any] = {}
    for batches in batch_sizes:
        world = _audit_world(
            batches, config.objects, config, seed=config.seed + batches, cameras=cameras
        )
        render = render_world(world)
        per_batch: dict[str, Any] = {
            "per_camera": {},
            "max_unexplained": 0,
            "max_missed": 0,
            "max_symmetric_difference_excess": 0.0,
        }
        for camera in cameras:
            view = render.camera_names.index(camera.name)
            ref_index, _ref_depth, ref_count, ref_coverage, ref_info = reference_visibility(
                world, camera, grid=grid
            )
            stats = compare_label_maps(render.body_index[:, view], ref_index, coverage=ref_coverage)
            consistency = image_label_consistency(world, camera, render)
            contamination = _cross_sample_contamination(render.body_index[:, view], ref_index)
            reference_areas = torch.tensor(
                [int((ref_index == body_id).sum()) for body_id in range(len(world.bodies))]
            )
            # Point splatting marks a boundary pixel cell up to one pixel outside the
            # centre-sampled silhouette, so the mask-size difference is bounded by the boundary
            # length (1.5 px per boundary pixel plus a constant), not by the silhouette area.
            allowed_sym_diff = 1.5 * stats["perimeter_ref"] + 4.0
            sym_diff_excess = (stats["sym_diff"].to(torch.float64) - allowed_sym_diff).clamp_min(
                0.0
            )
            max_unexplained = int(stats["unexplained"].max())
            max_missed = int(stats["missed"].max())
            max_sym_diff_excess = float(sym_diff_excess.max())
            min_iou = float(stats["iou"].min())
            per_camera = {
                "unexplained_pixels": stats["unexplained"].tolist(),
                "missed_pixels": stats["missed"].tolist(),
                "iou_per_body": stats["iou"].tolist(),
                "symmetric_difference_per_body": stats["sym_diff"].tolist(),
                "perimeter_per_body": stats["perimeter_ref"].tolist(),
                "allowed_symmetric_difference_per_body": allowed_sym_diff.tolist(),
                "reference_area_per_body": reference_areas.tolist(),
                "max_unexplained": max_unexplained,
                "max_missed": max_missed,
                "max_symmetric_difference_excess": max_sym_diff_excess,
                "min_iou": min_iou,
                "contamination": contamination,
                "image_label_consistency": consistency,
                "reference_grid_steps_per_body": ref_info["grid_steps_per_body"],
                "reference_points_per_pixel": {
                    "min_covered": int(ref_count.clamp(max=1).min()),
                    "max_covered": int(ref_count.max()),
                },
            }
            per_batch["per_camera"][camera.name] = per_camera
            per_batch["max_unexplained"] = max(per_batch["max_unexplained"], max_unexplained)
            per_batch["max_missed"] = max(per_batch["max_missed"], max_missed)
            per_batch["max_symmetric_difference_excess"] = max(
                per_batch["max_symmetric_difference_excess"], max_sym_diff_excess
            )
            if max_unexplained > 0 or max_missed > 0 or max_sym_diff_excess > 0.0:
                report["failures"].append(
                    f"B={batches} {camera.name}: independent projection mismatch "
                    f"(unexplained={max_unexplained}, missed={max_missed}, "
                    f"sym_diff_excess={max_sym_diff_excess:.1f})"
                )
            if max(contamination.values(), default=0.0) > 0.01:
                report["failures"].append(
                    f"B={batches} {camera.name}: cross-sample contamination {contamination}"
                )
            if consistency["pixels_checked"] <= 0:
                report["failures"].append(f"B={batches} {camera.name}: no pixels checked")
        per_batch["reference_depth_max_error"] = _reference_depth_error(
            world, cameras[0], render, grid=grid
        )
        checks[f"B={batches}"] = per_batch

    # Batch permutation must only permute samples.
    order = torch.tensor([2, 0, 3, 1])[:4]
    batches = int(order.numel())
    base_world = _audit_world(batches, 3, config, seed=config.seed + 11, cameras=cameras)
    permuted_world = _permute_world(base_world, order)
    base_render = render_world(base_world)
    permuted_render = render_world(permuted_world)
    permutation_ok = bool(
        torch.equal(permuted_render.body_index, base_render.body_index[order])
        and torch.equal(permuted_render.depth_m, base_render.depth_m[order])
        and torch.equal(permuted_render.radiance, base_render.radiance[order])
    )
    checks["batch_permutation"] = {
        "samples": batches,
        "order": order.tolist(),
        "exact": permutation_ok,
    }
    if not permutation_ok:
        report["failures"].append("batch permutation is not an exact per-sample permutation")

    # Perturbing one sample must not touch the others.
    perturbed_world = _audit_world(
        4, 3, config, seed=config.seed + 11, cameras=cameras, perturbation_index=1
    )
    perturbed_render = render_world(perturbed_world)
    keep = torch.tensor([0, 2, 3])
    others_equal = bool(
        torch.equal(perturbed_render.radiance[keep], base_render.radiance[keep])
        and torch.equal(perturbed_render.body_index[keep], base_render.body_index[keep])
    )
    moved = bool(not torch.equal(perturbed_render.radiance[1], base_render.radiance[1]))
    checks["single_sample_perturbation"] = {
        "others_unchanged": others_equal,
        "perturbed_sample_changed": moved,
    }
    if not (others_equal and moved):
        report["failures"].append("single-sample perturbation leaked across samples")

    # Reproducible counterexample for the historical broadcast defect.
    counterexample = _historical_counterexample(
        base_world, cameras[0], render=base_render, grid=grid
    )
    checks["historical_broadcast_counterexample"] = counterexample
    if not counterexample["detected"]:
        report["failures"].append("historical [B,1,2]+[B,2] label defect was not detected")

    # Static geometry and exposure must give a numerically identical frame.
    static_plan = resolve_plan("static")
    static_worlds, _ = build_factor_worlds(
        static_plan, config, batch=2, frames=2, seed=config.seed + 21
    )
    static_a = render_world(static_worlds[0])
    static_b = render_world(static_worlds[1])
    residual = float((static_a.radiance - static_b.radiance).abs().max())
    checks["static_zero_residual"] = {"max_abs_residual": residual, "tolerance": 1e-7}
    if residual > 1e-6:
        report["failures"].append(f"static residual {residual} exceeds numerical precision")

    # Unidentifiable single moving plane: two distinct world states give the same image.
    ambiguity = _unidentifiable_equivalence(config)
    checks["unidentifiable_full_view_plane"] = ambiguity
    if not ambiguity["identical"]:
        report["failures"].append(
            "full-view single plane did not reproduce the translation ambiguity"
        )

    # CFA sampling must follow the declared channel/phase exactly.
    cfa_check = _cfa_phase_check(config)
    checks["cfa_channel_phase"] = cfa_check
    if not cfa_check["exact"]:
        report["failures"].append("CFA sampling does not match the declared phase/channel table")

    report["checks"] = checks
    report["passed"] = not report["failures"]
    return report


def _cross_sample_contamination(
    rendered_index: torch.Tensor,
    reference_index: torch.Tensor,
) -> dict[str, float]:
    """Fraction of sample ``source``'s independent object pixels that ``target`` claims.

    Both index maps are ``[B, H, W]``.  For correctly computed labels this is 0 whenever the
    per-sample placements are disjoint, so it is a direct cross-sample contamination test.
    """
    batch = rendered_index.shape[0]
    result: dict[str, float] = {}
    if batch < 2:
        return result
    for target in range(batch):
        for source in range(batch):
            if source == target:
                continue
            reference_objects = reference_index[source] >= 1
            if not bool(reference_objects.any()):
                result[f"claim[{target}]<-ref[{source}]"] = 0.0
                continue
            claimed = (rendered_index[target][reference_objects] >= 1).to(torch.float64).mean()
            result[f"claim[{target}]<-ref[{source}]"] = float(claimed)
    return result


def _reference_depth_error(
    world: WorldBatch,
    camera: PinholeCamera,
    render: RenderResult,
    *,
    grid: int,
) -> float:
    """Worst depth disagreement between the renderer and the independent splat path.

    Only pixels where both paths agree on the visible body are comparable; index disagreement
    is already covered by the unexplained/missed pixel counts.
    """
    ref_index, ref_depth, _, _, _ = reference_visibility(world, camera, grid=grid)
    view = render.camera_names.index(camera.name)
    mask = (ref_index >= 0) & (ref_index == render.body_index[:, view])
    if not bool(mask.any()):
        return 0.0
    diff = (ref_depth[mask] - render.depth_m[:, view][mask].to(torch.float64)).abs()
    return float(diff.max())


def _permute_world(world: WorldBatch, order: torch.Tensor) -> WorldBatch:
    return WorldBatch(
        cameras=world.cameras,
        T_world_from_cam={name: matrix[order] for name, matrix in world.T_world_from_cam.items()},
        bodies=tuple(
            BodyTrack(
                body=track.body,
                T_world_from_body=track.T_world_from_body[order],
                present=track.present[order],
            )
            for track in world.bodies
        ),
        factor=world.factor,
        frame=world.frame,
    )


def _historical_counterexample(
    world: WorldBatch,
    camera: PinholeCamera,
    *,
    render: RenderResult,
    grid: int | None = None,
) -> dict[str, Any]:
    """Feed the historical merged labels through the audit and record what it catches."""
    buggy = _buggy_merged_labels(world, camera, grid=grid or 129)
    view = render.camera_names.index(camera.name)
    reference_index, _, _, reference_coverage, _ = reference_visibility(world, camera, grid=None)
    stats = compare_label_maps(buggy, reference_index, coverage=reference_coverage)
    image_index = render.body_index[:, view]
    # Pixels claimed by the merged label that the actual rendered image shows as background
    # (or a different body) are the direct image/label misalignment.
    image_disagreement = int(((buggy >= 0) & (buggy != image_index)).sum())
    contamination: dict[str, float] = {}
    for target in range(world.batch):
        for source in range(world.batch):
            if target == source:
                continue
            ref_source = (reference_index[source] >= 0) & (reference_index[source] >= 1)
            if bool(ref_source.any()):
                claimed = (buggy[target][ref_source] >= 0).float().mean()
                contamination[f"claim[{target}]<-ref[{source}]"] = float(claimed)
    detected = bool(
        int(stats["unexplained"].max()) > 0
        and image_disagreement > 0
        and max(contamination.values(), default=0.0) > 0.5
    )
    return {
        "defect": "[B,1,2] + [B,2] broadcast to [B,B,2] merges object positions across samples",
        "source": "docs/status.md 2026-10-13 (synthetic.py label merge)",
        "detected": detected,
        "unexplained_pixels": stats["unexplained"].tolist(),
        "image_label_disagreement_pixels": image_disagreement,
        "contamination": contamination,
    }


def _unidentifiable_equivalence(config: D0Config) -> dict[str, Any]:
    """A plane filling the whole view makes camera and object translation inseparable."""
    camera = default_cameras(config.width, config.height, ("cam0",))[0]
    delta = 0.02
    body = PlaneBody(
        name="plane",
        half_extents=(1.5, 1.2),
        albedo=(0.8, 0.62, 0.45),
        texture="checker",
        tex_scale=6.0,
        kind="plane",
    )

    def make(cam_x: float, plane_x: float) -> RenderResult:
        rotation = rotation_rpy_deg(
            torch.full((1,), 180.0, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )
        cam_rotation = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3)
        plane_centre = torch.tensor([[plane_x, 0.0, 0.6]], dtype=torch.float64)
        cam_centre = torch.tensor([[cam_x, 0.0, 0.0]], dtype=torch.float64)
        world = WorldBatch(
            cameras=(camera,),
            T_world_from_cam={"cam0": transform_from_rt(cam_rotation, cam_centre)},
            bodies=(
                BodyTrack(
                    body=body,
                    T_world_from_body=transform_from_rt(rotation, plane_centre),
                    present=torch.ones(1, dtype=torch.bool),
                ),
            ),
            factor="unidentifiable_full_view_plane",
        )
        return render_world(world)

    camera_moves = make(delta, 0.0)
    plane_moves = make(0.0, -delta)
    identical = bool(
        torch.allclose(camera_moves.radiance, plane_moves.radiance, atol=1e-6, rtol=0.0)
        and (camera_moves.body_index >= 0).all()
        and (plane_moves.body_index >= 0).all()
    )
    return {
        "delta_m": delta,
        "identical": identical,
        "max_radiance_difference": float(
            (camera_moves.radiance - plane_moves.radiance).abs().max()
        ),
        "camera_only_covers_full_view": bool((camera_moves.body_index >= 0).all()),
        "object_only_covers_full_view": bool((plane_moves.body_index >= 0).all()),
        "meaning": "measurement alone cannot separate camera translation from plane translation",
    }


def _cfa_phase_check(config: D0Config) -> dict[str, Any]:
    """CFA DN must equal the declared channel of the quantised radiance, phase by phase."""
    pattern = config.cfa_pattern
    radiance = torch.rand(1, 1, 3, 8, 8, generator=torch.Generator().manual_seed(config.seed)).to(
        torch.float64
    )
    dn = apply_cfa(
        radiance,
        pattern=pattern,
        bit_depth=config.cfa_bit_depth,
        black_level_dn=config.cfa_black_level_dn,
        white_level_dn=config.cfa_white_level_dn,
    )
    channels = cfa_phase_channel_map(pattern, 8, 8)
    expected = torch.round(
        config.cfa_black_level_dn
        + radiance[0, 0].gather(0, channels[None])[0].clamp(0.0, 1.0)
        * float(config.cfa_white_level_dn - config.cfa_black_level_dn)
    ).to(torch.uint16)
    max_error = int((dn[0, 0].to(torch.int64) - expected.to(torch.int64)).abs().max())
    hand_table = {name: [list(row) for row in table] for name, table in CFA_PHASE_CHANNELS.items()}
    return {
        "pattern": pattern,
        "bit_depth": config.cfa_bit_depth,
        "black_level_dn": config.cfa_black_level_dn,
        "white_level_dn": config.cfa_white_level_dn,
        "phase_channel_table": hand_table,
        "max_dn_error": max_error,
        "exact": max_error == 0,
    }


# --------------------------------------------------------------------------------------
# D0 generation
# --------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_d0(
    config: D0Config,
    output_dir: Path,
    *,
    audit_batches: Sequence[int] = (1, 2, 4),
    write: bool = True,
) -> dict[str, Any]:
    """Render every factor, write ``d0/*.npz`` plus the manifest and label audit."""
    output_dir = Path(output_dir)
    d0_dir = output_dir / "d0"
    d0_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_d0(config, batch_sizes=audit_batches)
    factor_records: list[dict[str, Any]] = []
    for factor in default_factor_names():
        plan = resolve_plan(factor)
        worlds, factor_meta = build_factor_worlds(plan, config)
        factor_records.append(
            _render_factor(plan, factor_meta, config, worlds, d0_dir if write else None)
        )
    manifest = {
        "schema": SCHEMA_D0,
        "config": config.as_record(),
        "units": {
            "geometry": "metre",
            "time": "frame index (synthetic, unit = one frame)",
            "depth_m": "camera-space z in metres, inf = no surface",
            "cfa": "digital numbers (DN)",
            "angle": "degree for rpy, radian for rotation matrices",
        },
        "frames_and_conventions": {
            "camera": "OpenCV, x right, y down, z forward",
            "pixel": "u=column, v=row, origin at the top-left pixel centre",
            "projection": "u = fx*x/z + cx, v = fy*y/z + cy",
            "body": "local +z is the face normal, local x/y span the rectangular face",
            "camera_facing_rpy_deg": [180.0, 0.0, 0.0],
            "world": "right-handed; not a calibrated robot base frame",
            "intrinsics_source": "declared diagnostic pinhole, not a real calibration",
        },
        "online_vs_ground_truth": {
            "online_keys": [
                "online_measurement_rgb3",
                "online_measurement_cfa",
                "online_clip_mask",
                "online_camera_translation",
                "online_camera_rpy_deg",
                "online_action_delta",
                "online_body_association",
            ],
            "gt_keys": [
                "gt_radiance",
                "gt_depth_m",
                "gt_body_index",
                "gt_silhouette",
                "gt_camera_translation",
                "gt_camera_rpy_deg",
                "gt_body_position",
                "gt_body_present",
                "gt_visible_pixels",
                "gt_covered_pixels",
                "gt_in_view",
                "gt_motion_phase",
                "gt_gain",
                "gt_noise_sigma",
            ],
            "rule": "gt_* keys are supervision/diagnostics only and never enter the online bundle",
        },
        "cfa": {
            "representation": "synthetic_cfa_mosaic",
            "sampling": "declared spectral channel and phase sampled directly from linear radiance",
            "not": "no re-mosaicked RGB is presented as original RAW",
            "pattern": config.cfa_pattern,
            "bit_depth": config.cfa_bit_depth,
            "black_level_dn": config.cfa_black_level_dn,
            "white_level_dn": config.cfa_white_level_dn,
            "phase_channel_table": {
                name: [list(row) for row in table] for name, table in CFA_PHASE_CHANNELS.items()
            },
            "channel_order": list(CHANNEL_NAMES),
            "rgb3_representation": "three-channel linear float32 (declared synthetic)",
        },
        "factors": factor_records,
        "audit": audit,
    }
    if write:
        audit_path = d0_dir / "audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["audit_path"] = str(audit_path.relative_to(output_dir))
        manifest_path = d0_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _render_factor(
    plan: FactorPlan,
    factor_meta: dict[str, Any],
    config: D0Config,
    worlds: list[WorldBatch],
    out_dir: Path | None,
) -> dict[str, Any]:
    """Render one factor sequence and (optionally) write its compressed npz."""
    frames = len(worlds)
    batch = worlds[0].batch
    views = len(worlds[0].cameras)
    bodies = worlds[0].body_names
    n_bodies = len(bodies)
    radiance = torch.zeros(
        frames, batch, views, 3, config.height, config.width, dtype=torch.float32
    )
    depth = torch.zeros(frames, batch, views, config.height, config.width, dtype=torch.float32)
    index = torch.full((frames, batch, views, config.height, config.width), -1, dtype=torch.int16)
    silhouette = torch.zeros(
        frames, batch, views, n_bodies, config.height, config.width, dtype=torch.uint8
    )
    rgb3 = torch.zeros_like(radiance)
    cfa_dn = torch.zeros(frames, batch, views, config.height, config.width, dtype=torch.uint16)
    clip = torch.zeros(frames, batch, views, config.height, config.width, dtype=torch.uint8)
    reported_t = torch.zeros(frames, batch, 3, dtype=torch.float32)
    reported_rpy = torch.zeros(frames, batch, 3, dtype=torch.float32)
    action_delta = torch.zeros(frames, batch, 6, dtype=torch.float32)
    gt_t = torch.zeros(frames, batch, 3, dtype=torch.float32)
    gt_rpy = torch.zeros(frames, batch, 3, dtype=torch.float32)
    gt_gain = torch.zeros(frames, batch, dtype=torch.float32)
    visible_pixels = torch.zeros(frames, batch, views, n_bodies, dtype=torch.int32)
    in_view = torch.zeros(frames, batch, views, n_bodies, dtype=torch.uint8)
    covered_pixels = torch.zeros(frames, batch, views, n_bodies, dtype=torch.int32)
    body_position = torch.zeros(frames, batch, n_bodies, 3, dtype=torch.float32)
    body_present = torch.zeros(frames, batch, n_bodies, dtype=torch.uint8)
    association = torch.full(
        (frames, batch, views, config.height, config.width), -1, dtype=torch.int8
    )
    event_log: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    generator = torch.Generator().manual_seed(config.seed + 977 * len(plan.name))
    for frame, world in enumerate(worlds):
        render = render_world(world)
        radiance[frame] = render.radiance
        depth[frame] = torch.nan_to_num(
            render.depth_m, nan=math.inf, posinf=math.inf, neginf=math.inf
        )
        index[frame] = render.body_index
        silhouette[frame] = render.silhouette.to(torch.uint8)
        measurement = form_measurement(
            render.radiance.to(torch.float64),
            gain=world.gain.to(torch.float64),
            noise_sigma=world.noise_sigma,
            generator=generator,
            cfa_pattern=config.cfa_pattern,
            cfa_bit_depth=config.cfa_bit_depth,
            cfa_black_level_dn=config.cfa_black_level_dn,
            cfa_white_level_dn=config.cfa_white_level_dn,
        )
        rgb3[frame] = measurement.rgb3
        cfa_dn[frame] = measurement.cfa_dn
        clip[frame] = measurement.clip_mask.to(torch.uint8)
        gt_gain[frame] = world.gain.to(torch.float32)
        labels = frame_labels(world, render)
        rig_camera = world.cameras[0].name
        gt_t[frame] = labels["camera_translation"][rig_camera].to(torch.float32)
        gt_rpy[frame] = labels["camera_rpy_deg"][rig_camera].to(torch.float32)
        reported_t[frame] = world.reported_translation.to(torch.float32)
        reported_rpy[frame] = world.reported_rpy_deg.to(torch.float32)
        if frame > 0:
            action_delta[frame, :, :3] = reported_t[frame] - reported_t[frame - 1]
            action_delta[frame, :, 3:] = reported_rpy[frame] - reported_rpy[frame - 1]
        for body_id, name in enumerate(bodies):
            entry = labels["bodies"][name]
            visible_pixels[frame, :, :, body_id] = entry["visible_pixels"].to(torch.int32)
            covered_pixels[frame, :, :, body_id] = entry["covered_pixels"].to(torch.int32)
            for view, camera in enumerate(world.cameras):
                centre = entry["centre_px"][:, view, :]
                inside = (
                    (centre[:, 0] >= 0)
                    & (centre[:, 0] <= camera.width - 1)
                    & (centre[:, 1] >= 0)
                    & (centre[:, 1] <= camera.height - 1)
                    & (entry["centre_depth_m"][:, view] > 0)
                    & entry["present"]
                )
                in_view[frame, :, view, body_id] = inside.to(torch.uint8)
            body_position[frame, :, body_id] = entry["world_position"].to(torch.float32)
            body_present[frame, :, body_id] = entry["present"].to(torch.uint8)
        for view in range(views):
            association[frame, :, view] = render.body_index[:, view].to(torch.int8)
        if plan.wrong_association:
            # Declared online corruption: left half of the image swaps object 0 and object 2.
            half = config.width // 2
            left = association[frame, :, :, :, :half]
            first = left == 1
            third = left == 3
            left[first] = 3
            left[third] = 1
        events = _frame_events(labels, previous, worlds[frame])
        if events:
            event_log.append({"frame": frame, "events": events})
        previous = labels
    motion = motion_phase(_object_positions(worlds))
    record: dict[str, Any] = {
        "factor": plan.name,
        "components": list(plan.components),
        "frames": frames,
        "batch": batch,
        "views": views,
        "bodies": list(bodies),
        "metadata": factor_meta,
        "events": event_log,
        "arrays": {},
        "npz": None,
    }
    if out_dir is not None:
        path = out_dir / f"{plan.name}.npz"
        arrays = {
            "online_measurement_rgb3": rgb3.numpy(),
            "online_measurement_cfa": cfa_dn.numpy(),
            "online_clip_mask": clip.numpy(),
            "online_camera_translation": reported_t.numpy(),
            "online_camera_rpy_deg": reported_rpy.numpy(),
            "online_action_delta": action_delta.numpy(),
            "online_body_association": association.numpy(),
            "gt_radiance": radiance.numpy(),
            "gt_depth_m": depth.numpy(),
            "gt_body_index": index.numpy(),
            "gt_silhouette": silhouette.numpy(),
            "gt_camera_translation": gt_t.numpy(),
            "gt_camera_rpy_deg": gt_rpy.numpy(),
            "gt_body_position": body_position.numpy(),
            "gt_body_present": body_present.numpy(),
            "gt_visible_pixels": visible_pixels.numpy(),
            "gt_covered_pixels": covered_pixels.numpy(),
            "gt_in_view": in_view.numpy(),
            "gt_motion_phase": motion.numpy(),
            "gt_gain": gt_gain.numpy(),
            "gt_noise_sigma": np.full(
                (frames, batch), np.float32(worlds[0].noise_sigma), dtype=np.float32
            ),
        }
        np.savez_compressed(str(path), **arrays)
        record["npz"] = path.name
        record["npz_sha256"] = _sha256(path)
        record["arrays"] = {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        }
        record["bytes"] = path.stat().st_size
    record["coverage"] = {
        "visible_pixels_total": int(visible_pixels.sum()),
        "occluded_events": int(
            sum(1 for item in event_log for event in item["events"] if "occlusion_start" in event)
        ),
        "birth_events": int(
            sum(1 for item in event_log for event in item["events"] if "born" in event)
        ),
        "left_view_events": int(
            sum(1 for item in event_log for event in item["events"] if "left_view" in event)
        ),
        "reverse_frames": int((motion == -1).sum()),
        "stopped_frames": int((motion == 0).sum()),
    }
    return record


def _object_positions(worlds: list[WorldBatch]) -> torch.Tensor:
    frames = len(worlds)
    batch = worlds[0].batch
    positions = torch.zeros(frames, batch, 3, dtype=torch.float64)
    for frame, world in enumerate(worlds):
        for track in world.bodies:
            if track.body.kind == "object":
                positions[frame] = track.T_world_from_body[:, :3, 3]
                break
    return positions


def _frame_events(
    labels: dict[str, Any],
    previous: dict[str, Any] | None,
    world: WorldBatch,
) -> dict[str, Any]:
    """Frame events derived from the same labels (birth, exit, occlusion transitions)."""
    events: dict[str, Any] = {}
    if previous is None:
        return events
    for track in world.bodies:
        name = track.body.name
        entry = labels["bodies"][name]
        before = previous["bodies"][name]
        for view in range(len(world.cameras)):
            centre = entry["centre_px"][:, view, :]
            inside = (
                (centre[:, 0] >= 0)
                & (centre[:, 0] <= world.cameras[view].width - 1)
                & (centre[:, 1] >= 0)
                & (centre[:, 1] <= world.cameras[view].height - 1)
                & entry["present"]
                & (entry["centre_depth_m"][:, view] > 0)
            )
            previous_inside = (
                (before["centre_px"][:, view, 0] >= 0)
                & (before["centre_px"][:, view, 0] <= world.cameras[view].width - 1)
                & (before["centre_px"][:, view, 1] >= 0)
                & (before["centre_px"][:, view, 1] <= world.cameras[view].height - 1)
                & before["present"]
                & (before["centre_depth_m"][:, view] > 0)
            )
            born = inside & ~previous_inside
            left = ~inside & previous_inside
            occluded_now = (
                inside
                & (entry["visible_pixels"][:, view] == 0)
                & (entry["covered_pixels"][:, view] > 0)
            )
            occluded_before = (
                previous_inside
                & (before["visible_pixels"][:, view] == 0)
                & (before["covered_pixels"][:, view] > 0)
            )
            if bool(born.any()):
                events.setdefault("born", []).append(
                    {
                        "body": name,
                        "camera": world.cameras[view].name,
                        "samples": born.nonzero().flatten().tolist(),
                    }
                )
            if bool(left.any()):
                events.setdefault("left_view", []).append(
                    {
                        "body": name,
                        "camera": world.cameras[view].name,
                        "samples": left.nonzero().flatten().tolist(),
                    }
                )
            if bool((occluded_now & ~occluded_before).any()):
                events.setdefault("occlusion_start", []).append(
                    {"body": name, "camera": world.cameras[view].name}
                )
            if bool((occluded_before & ~occluded_now).any()) and bool(occluded_before.any()):
                events.setdefault("occlusion_end", []).append(
                    {"body": name, "camera": world.cameras[view].name}
                )
    return events


# --------------------------------------------------------------------------------------
# D1: collector and feasibility
# --------------------------------------------------------------------------------------

D1_REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "libero_sim_assets",
        "need": "deployed LIBERO simulator, assets and bddl files for same-version replay",
        "evidence_key": "libero_sim",
        "checks": ("import_ok", "env_created", "bddl_files_present"),
        "verification_level": "executed",
        "plan": (
            "reuse the deployed env with MUJOCO_GL=egl and LIBERO_CONFIG_PATH set; "
            "no new assets needed"
        ),
    },
    {
        "id": "robotwin_sim_assets",
        "need": "deployed RoboTwin simulator, assets and task configs",
        "evidence_key": "robotwin_sim",
        "checks": ("import_ok", "assets_present", "task_config_present"),
        "verification_level": "api_and_assets_only",
        "plan": "before collection, create one task env in the robotwin interpreter to verify the "
        "Vulkan/rendering path; assets and task configs already exist",
    },
    {
        "id": "controller_logs",
        "need": "per-step controller / rollout logs aligned with recorded frames",
        "evidence_key": "controller_logs",
        "checks": ("files_found", "actions_present"),
        "verification_level": "files_inspected",
        "plan": (
            "LIBERO benchmark rollouts already log one action per env step; training "
            "replays must write their own n04ext_ control log because the D3 replay path has none"
        ),
    },
    {
        "id": "same_version_replay",
        "need": "same-version replay: pinned sources plus recorded state or source trajectory",
        "evidence_key": "replay",
        "checks": ("libero_pinned_head", "robotwin_pinned_head", "libero_state_replay_exact"),
        "verification_level": "executed",
        "plan": "LIBERO replays exactly from the recorded `states`; the original RoboTwin clean "
        "trajectory pkl/hdf5 is absent, so RoboTwin can only be replayed from its npz record or "
        "re-collected as a new named extension set",
    },
    {
        "id": "executed_action",
        "need": "actual executed action per frame, distinguishable from unexecuted predictions",
        "evidence_key": "npz_fields",
        "checks": ("libero_actions_per_frame", "robotwin_actions_per_frame"),
        "verification_level": "fields_inspected",
        "plan": (
            "LIBERO `actions` are the values passed to env.step; RoboTwin `actions` are "
            "14-D joint states, so a collector must record the commanded joint path itself"
        ),
    },
    {
        "id": "camera_intrinsics_k",
        "need": "camera intrinsics K per view",
        "evidence_key": "geometry_fields",
        "checks": ("libero_K", "robotwin_K", "robotwin_K_executed"),
        "verification_level": "libero_executed_robotwin_api",
        "plan": "LIBERO: robosuite get_camera_intrinsic_matrix(sim, name, h, w); RoboTwin: "
        "SAPIEN RenderCameraComponent.get_intrinsic_matrix via Camera.get_config()",
    },
    {
        "id": "camera_pose",
        "need": "camera and robot pose per frame",
        "evidence_key": "geometry_fields",
        "checks": ("libero_state_len_gt_0", "robotwin_robot_states"),
        "verification_level": "fields_inspected",
        "plan": "LIBERO: replay the 79-D recorded state then read sim poses/extrinsics; RoboTwin: "
        "robot_states/endpose plus entity poses, with camera model matrix from the camera wrapper",
    },
    {
        "id": "depth",
        "need": "per-frame depth aligned with the measurement",
        "evidence_key": "geometry_fields",
        "checks": ("libero_depth_render", "robotwin_depth_api", "robotwin_depth_executed"),
        "verification_level": "libero_executed_robotwin_api",
        "plan": (
            "LIBERO: env.sim.render(depth=True) at the replayed state; RoboTwin: "
            "cameras.get_depth() (millimetres, alpha-masked) after copying the physics state"
        ),
    },
    {
        "id": "visibility",
        "need": "per-object visibility / instance support masks",
        "evidence_key": "geometry_fields",
        "checks": (
            "libero_segmentation_render",
            "robotwin_segmentation_api",
            "robotwin_segmentation_executed",
        ),
        "verification_level": "libero_executed_robotwin_api",
        "plan": "LIBERO: env.sim.render(segmentation=True) plus depth for occlusion; RoboTwin: "
        "cameras.get_segmentation(level='actor'|'mesh')",
    },
    {
        "id": "contact_state",
        "need": "contact state between gripper and objects",
        "evidence_key": "geometry_fields",
        "checks": ("libero_contacts", "robotwin_contacts_api", "robotwin_contacts_executed"),
        "verification_level": "libero_executed_robotwin_api",
        "plan": (
            "LIBERO: sim.data.contact (geom pairs, distance, position) at the replayed state; "
            "RoboTwin: scene.get_contacts() / check_actors_contact / "
            "get_gripper_actor_contact_position"
        ),
    },
    {
        "id": "cfa_representation",
        "need": "declared CFA status of the available measurements: native or synthetic",
        "evidence_key": "npz_fields",
        "checks": ("libero_three_channel_only", "robotwin_three_channel_only", "d0_synthetic_cfa"),
        "verification_level": "fields_and_d0_output_inspected",
        "plan": (
            "both platforms ship three-channel pseudo-RAW, not native CFA; D0 supplies "
            "the declared synthetic CFA. A real CFA requirement means a new independently "
            "named capture, never renaming the existing npz files"
        ),
    },
)


def npz_member_headers(npz_path: str | Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Shape and dtype of every member without decompressing the arrays.

    The D3/RAW recordings reach hundreds of megabytes per episode; a collector must not
    decompress a whole trajectory just to inventory its fields.
    """
    import zipfile

    headers: dict[str, tuple[tuple[int, ...], str]] = {}
    with zipfile.ZipFile(npz_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:  # pragma: no cover - numpy never writes other versions today
                    raise ValueError(f"unsupported npy version {version} in {member}")
            headers[member[: -len(".npy")]] = (tuple(int(v) for v in shape), str(dtype))
    return headers


def d1_npz_field_probe(
    npz_path: str | Path,
    *,
    deep: bool = True,
    sha256: bool = True,
) -> dict[str, Any]:
    """NPZ field availability for the D1 fields of interest.

    ``deep=True`` additionally reads the small bookkeeping arrays (``actions``, ``states``,
    ``robot_states``, ``metadata_json``); image arrays are only ever read through their
    headers.
    """
    path = Path(npz_path)
    probe: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return probe
    headers = npz_member_headers(path)
    probe["keys"] = list(headers)
    probe["shapes"] = {key: list(shape) for key, (shape, _) in headers.items()}
    probe["dtypes"] = {key: dtype for key, (_, dtype) in headers.items()}
    probe["bytes"] = int(path.stat().st_size)
    if sha256:
        probe["sha256"] = _sha256(path)
    if deep:
        with np.load(path, allow_pickle=True) as data:
            if "metadata_json" in data.files:
                try:
                    metadata = json.loads(str(data["metadata_json"].item()))
                except (ValueError, TypeError):
                    metadata = {}
                probe["metadata"] = metadata
            if "actions" in data.files:
                actions = np.asarray(data["actions"])
                probe["actions_per_frame"] = int(actions.shape[1]) if actions.ndim == 2 else 0
                probe["actions_finite"] = bool(
                    np.isfinite(actions[: min(4, actions.shape[0])]).all()
                )
            if "states" in data.files:
                probe["state_dim"] = int(np.asarray(data["states"]).shape[-1])
            if "robot_states" in data.files:
                probe["robot_state_dim"] = int(np.asarray(data["robot_states"]).shape[-1])
    image_keys = [
        key
        for key, (_shape, dtype) in headers.items()
        if dtype in {"uint8", "uint16", "float16", "float32"} and len(headers[key][0]) >= 3
    ]
    probe["image_keys"] = image_keys
    probe["three_channel_image_keys"] = [
        key for key in image_keys if len(headers[key][0]) == 4 and headers[key][0][-1] == 3
    ]
    probe["cfa_like_keys"] = [
        key for key in image_keys if len(headers[key][0]) == 3 or headers[key][0][-1] == 1
    ]
    lowered = {key.lower() for key in headers}
    probe["has_depth"] = any("depth" in key for key in lowered)
    probe["has_intrinsics"] = any("intrinsic" in key or key == "k" for key in lowered)
    probe["has_pose"] = any("pose" in key for key in lowered)
    probe["has_contact"] = any("contact" in key for key in lowered)
    probe["has_visibility"] = any("segmentation" in key or "visibility" in key for key in lowered)
    return probe


def build_feasibility(evidence: dict[str, Any], *, checked_at: str | None = None) -> dict[str, Any]:
    """Map observed remote evidence to per-item verdicts (missing evidence is not fabricated)."""
    items: list[dict[str, Any]] = []
    for requirement in D1_REQUIREMENTS:
        key = str(requirement["evidence_key"])
        entry = _dig(evidence, key)
        checks = [str(name) for name in requirement["checks"]]
        detail = {name: _dig(entry, name) for name in checks} if isinstance(entry, dict) else {}
        passed = [name for name, value in detail.items() if bool(value)]
        if entry is None:
            verdict = "not_probed"
        elif len(passed) == len(checks):
            verdict = "available"
        elif passed:
            verdict = "partial"
        else:
            verdict = "not_available"
        observed = (
            {name: value for name, value in entry.items() if name not in checks}
            if isinstance(entry, dict)
            else None
        )
        record = {
            "id": requirement["id"],
            "need": requirement["need"],
            "verdict": verdict,
            "verification_level": str(requirement.get("verification_level", "not_probed")),
            "checks": {name: detail.get(name) for name in checks},
            "evidence_key": key,
            "evidence_pointer": str(requirement.get("evidence_pointer", key)),
            "missing": [name for name in checks if not detail.get(name)],
            "plan": requirement.get("plan", ""),
            "observed": observed,
        }
        items.append(record)
    return {
        "schema": SCHEMA_FEASIBILITY,
        "scope": "partial: runnable collector plus per-item feasibility; no large-scale collection",
        "checked_at_utc": checked_at,
        "policy": {
            "sidecar": "only attached when frame-by-frame correspondence is verified",
            "otherwise": (
                "a separately named extension set (n04ext_*) is created; "
                "original NPZ names are not reused"
            ),
            "no_large_scale_collection": True,
        },
        "items": items,
        "verdict_counts": {
            verdict: sum(1 for item in items if item["verdict"] == verdict)
            for verdict in ("available", "partial", "not_available", "not_probed")
        },
    }


def _dig(mapping: Any, dotted: str) -> Any:
    node = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# --------------------------------------------------------------------------------------
# D1 executed-trace schema: make "bit-exact recomputable" an executable contract
# --------------------------------------------------------------------------------------

SCHEMA_D1_TRACE = "n04-d1-executed-trace-v1"

#: Per-step fields of an executed-trace record.  ``issued_command`` is the value the collector
#: actually handed to the simulator at that step (recorded at the call site), not a renamed
#: copy of the source ``actions`` array.
D1_TRACE_STEP_FIELDS: tuple[str, ...] = (
    "step_index",
    "pre_state",
    "issued_command",
    "post_state",
)

#: How the solver warm-start is made reproducible.
D1_TRACE_WARMSTART_MODES: tuple[str, ...] = (
    "disabled_by_flag",
    "zeroed_each_step",
    "saved_before_step",
)

#: How the controller / observable buffers are handled; the choice must be explicit.
D1_TRACE_CONTROLLER_BUFFER_POLICIES: tuple[str, ...] = (
    "recorded_initial_conditions",
    "declared_not_recomputable",
)

D1_TRACE_FIELD_SEMANTICS: dict[str, str] = {
    "step_index": "0-based index of the collected step inside this trace",
    "pre_state": "flattened simulator state before the step, [time, qpos, qvel] (float64)",
    "issued_command": "command actually passed to the simulator at this step (float32)",
    "post_state": "flattened simulator state after the step (float64)",
    "qacc_warmstart_before": (
        "solver warm-start used by this step when warmstart_mode=saved_before_step"
    ),
    "qacc_warmstart_after": (
        "solver warm-start left behind by this step (chain continuity, optional)"
    ),
}

D1_TRACE_RECOMPUTABILITY_RULE = (
    "bit-exact recomputation of 'executed action -> state transition' requires all of: "
    "(1) pre_state, post_state and issued_command on every step; "
    "(2) a controlled solver warm-start: warmstart_mode in {disabled_by_flag, zeroed_each_step} "
    "or warmstart_mode=saved_before_step together with the recorded qacc_warmstart_before array; "
    "(3) an explicit controller/observable buffer policy: recorded_initial_conditions (the recipe "
    "fixes them from a fresh, reset environment) or declared_not_recomputable; "
    "(4) the ordered replay recipe; and (5) a passing verification replay "
    "(verification.bit_exact == true). Missing any of (1)-(4) leaves only phase-level evidence."
)


def d1_trace_schema() -> dict[str, Any]:
    """Declared schema, semantics and recomputability rule of a collected executed trace."""
    return {
        "schema": SCHEMA_D1_TRACE,
        "artifacts": {
            "arrays": "<name>.npz holding the per-step arrays listed under step_fields",
            "record": "<name>.trace.json holding this schema, the recipe, the policies and the "
            "recomputability report",
        },
        "step_fields": {name: D1_TRACE_FIELD_SEMANTICS[name] for name in D1_TRACE_STEP_FIELDS},
        "optional_arrays": {
            name: D1_TRACE_FIELD_SEMANTICS[name]
            for name in ("qacc_warmstart_before", "qacc_warmstart_after")
        },
        "warmstart_modes": {
            "disabled_by_flag": "set mjDSBL_WARMSTART (bit 256) on the model before stepping; "
            "the solver ignores the warm-start, so no warm-start array is needed",
            "zeroed_each_step": "write zeros into qacc_warmstart before every step (cold start)",
            "saved_before_step": "record qacc_warmstart immediately before the step and restore "
            "exactly that array when recomputing",
        },
        "controller_buffer_policies": {
            "recorded_initial_conditions": "the trace was collected from a fresh environment whose "
            "construction, seed, reset and ordered step recipe are recorded, so the controller and "
            "observable buffers are fixed by that recipe",
            "declared_not_recomputable": "the source recording does not expose the buffers of the "
            "original run; the trace is explicit that only phase-level evidence is available",
        },
        "recomputability_rule": D1_TRACE_RECOMPUTABILITY_RULE,
        "evidence_levels": {
            "bit_exact_recomputable": "all requirements met and the verification replay passed",
            "phase_and_state_recorded": "states and issued commands recorded, but at least one "
            "requirement is missing; only phase-level evidence",
            "insufficient": "pre/post state or issued command missing",
        },
        "source_fidelity": "separate from recomputability: replaying an external recording whose "
        "controller buffers are unknown stays phase-level even when the collected trace itself is "
        "bit-exactly recomputable",
    }


def d1_recomputability_report(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply :data:`D1_TRACE_RECOMPUTABILITY_RULE` to a trace summary."""
    missing: list[str] = []
    fields_present = [str(name) for name in summary.get("fields_present", [])]
    for field_name in D1_TRACE_STEP_FIELDS:
        if field_name not in fields_present:
            missing.append(field_name)
    warmstart_mode = summary.get("warmstart_mode")
    if warmstart_mode not in D1_TRACE_WARMSTART_MODES:
        missing.append("warmstart_mode")
    elif warmstart_mode == "saved_before_step" and not summary.get("warmstart_array_present"):
        missing.append("qacc_warmstart_before")
    buffer_policy = summary.get("controller_buffer_policy")
    if buffer_policy not in D1_TRACE_CONTROLLER_BUFFER_POLICIES:
        missing.append("controller_buffer_policy")
    if not summary.get("recipe_recorded"):
        missing.append("replay_recipe")
    verification = summary.get("verification") or {}
    error = verification.get("max_abs_diff")
    tolerance = verification.get("tolerance", 1e-12)
    verified = (
        verification.get("bit_exact") is True
        and isinstance(error, (int, float))
        and np.isfinite(error)
        and isinstance(tolerance, (int, float))
        and np.isfinite(tolerance)
        and 0 <= error <= tolerance <= 1e-12
        and int(verification.get("steps", 0)) > 0
    )
    bit_exact = not missing and buffer_policy == "recorded_initial_conditions" and verified
    if missing:
        level = "insufficient"
    elif bit_exact:
        level = "bit_exact_recomputable"
    else:
        level = "phase_and_state_recorded"
    notes: list[str] = []
    if buffer_policy == "declared_not_recomputable":
        notes.append(
            "controller/observable buffers are declared not recomputable for this source; "
            "the transition is reproducible only at phase level"
        )
    if not verified and not missing:
        notes.append(
            "no passing verification replay is recorded; the recipe is complete but the "
            "bit-exact claim is unverified"
        )
    return {
        "rule": D1_TRACE_RECOMPUTABILITY_RULE,
        "bit_exact_recomputable": bit_exact,
        "evidence_level": level,
        "missing": missing,
        "verification": verification or None,
        "controller_buffer_policy": buffer_policy,
        "warmstart_mode": warmstart_mode,
        "notes": notes,
    }


def d1_write_trace(
    output_dir: str | Path,
    name: str,
    *,
    platform: str,
    source_npz: str | Path,
    steps: Sequence[dict[str, Any]],
    recipe: dict[str, Any],
    warmstart_mode: str,
    controller_buffer_policy: str,
    issued_command_source: str = "collector_step_call_argument",
    controller_buffer_notes: Sequence[str] = (),
    source_fidelity: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``<name>.npz`` plus ``<name>.trace.json`` and report recomputability.

    ``steps`` must provide ``pre_state``, ``issued_command`` and ``post_state`` per step; a
    ``qacc_warmstart_before`` value is required when ``warmstart_mode="saved_before_step"``.
    """
    if warmstart_mode not in D1_TRACE_WARMSTART_MODES:
        raise ValueError(f"unknown warmstart mode {warmstart_mode!r}")
    if controller_buffer_policy not in D1_TRACE_CONTROLLER_BUFFER_POLICIES:
        raise ValueError(f"unknown controller buffer policy {controller_buffer_policy!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not steps:
        raise ValueError("an executed trace needs at least one step")
    if warmstart_mode == "saved_before_step":
        unsaved = [
            int(step.get("step_index", index))
            for index, step in enumerate(steps)
            if "qacc_warmstart_before" not in step
        ]
        if unsaved:
            raise ValueError(
                "warmstart_mode='saved_before_step' needs a recorded qacc_warmstart_before for "
                f"every step; missing at steps {unsaved}"
            )
    arrays: dict[str, Any] = {
        "step_index": np.asarray([int(s["step_index"]) for s in steps], dtype=np.int32),
        "pre_state": np.stack([np.asarray(s["pre_state"], dtype=np.float64) for s in steps]),
        "issued_command": np.stack(
            [np.asarray(s["issued_command"], dtype=np.float32) for s in steps]
        ),
        "post_state": np.stack([np.asarray(s["post_state"], dtype=np.float64) for s in steps]),
    }
    if warmstart_mode == "saved_before_step":
        arrays["qacc_warmstart_before"] = np.stack(
            [np.asarray(s["qacc_warmstart_before"], dtype=np.float64) for s in steps]
        )
        if all("qacc_warmstart_after" in s for s in steps):
            arrays["qacc_warmstart_after"] = np.stack(
                [np.asarray(s["qacc_warmstart_after"], dtype=np.float64) for s in steps]
            )
    npz_path = output_dir / f"{name}.npz"
    np.savez_compressed(npz_path, **arrays)
    report = d1_recomputability_report(
        {
            "fields_present": [f for f in D1_TRACE_STEP_FIELDS if f in arrays],
            "warmstart_mode": warmstart_mode,
            "warmstart_array_present": "qacc_warmstart_before" in arrays,
            "controller_buffer_policy": controller_buffer_policy,
            "recipe_recorded": bool(recipe),
            "verification": verification,
        }
    )
    record: dict[str, Any] = {
        "schema": SCHEMA_D1_TRACE,
        "name": name,
        "platform": platform,
        "source_npz": str(source_npz),
        "steps": len(steps),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "npz": npz_path.name,
        "npz_sha256": _sha256(npz_path),
        "npz_bytes": npz_path.stat().st_size,
        "step_field_semantics": {
            name_: D1_TRACE_FIELD_SEMANTICS[name_] for name_ in D1_TRACE_STEP_FIELDS
        },
        "warmstart_mode": warmstart_mode,
        "warmstart_semantics": d1_trace_schema()["warmstart_modes"][warmstart_mode],
        "controller_buffer_policy": controller_buffer_policy,
        "controller_buffer_notes": list(controller_buffer_notes),
        "issued_command_source": issued_command_source,
        "recipe": recipe,
        "source_fidelity": source_fidelity,
        "recomputability": report,
    }
    json_path = output_dir / f"{name}.trace.json"
    json_path.write_text(
        json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    record["trace_record"] = json_path.name
    record["trace_record_sha256"] = _sha256(json_path)
    return record


COLLECTOR_TEMPLATE = '''#!/usr/bin/env python3
"""N04 D1 mechanism-data collector (emitted by owac.raw.n04_mechanism_data).

Verifies per-frame correspondence before writing anything, then either attaches a sidecar to
the original recording or creates a separately named extension set.  With ``--trace-steps`` it
also collects an executed trace (pre-state, issued command, post-state plus the declared
warm-start and controller-buffer policies) so that "executed action <-> state transition" is
bit-exactly recomputable whenever the recorded requirements are met.  Run inside the frozen
simulator environment with the OWAC source snapshot on PYTHONPATH.
"""

from __future__ import annotations

import sys

from owac.raw.n04_mechanism_data import d1_main

if __name__ == "__main__":
    raise SystemExit(d1_main(sys.argv[1:]))
'''


def emit_collector(output_dir: Path) -> Path:
    """Write the runnable D1 collector entry point."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "collect_mechanism_data.py"
    path.write_text(COLLECTOR_TEMPLATE, encoding="utf-8")
    return path


def d1_plan(
    *,
    platform: str,
    npz_path: str | Path,
    output_dir: str | Path,
    correspondence_verified: bool,
    correspondence_notes: list[str] | None = None,
    limit: int = 4,
) -> dict[str, Any]:
    """Decide sidecar versus extension set from the correspondence check."""
    stem = Path(npz_path).stem
    if correspondence_verified:
        target = {
            "mode": "sidecar",
            "name": f"{stem}.n04-sidecar.npz",
            "reason": "frame-by-frame correspondence verified in this environment",
        }
    else:
        target = {
            "mode": "extension_set",
            "name": f"n04ext_{platform}_{stem}.npz",
            "reason": "correspondence not confirmed; original NPZ name is not reused",
        }
    return {
        "platform": platform,
        "source_npz": str(npz_path),
        "output_dir": str(output_dir),
        "limit_frames": limit,
        "correspondence_verified": correspondence_verified,
        "correspondence_notes": list(correspondence_notes or []),
        "target": target,
        "fields_to_collect": [
            "executed_action",
            "camera_intrinsics_k",
            "camera_pose",
            "robot_pose",
            "depth_m",
            "visibility_mask",
            "contact_state",
        ],
        "trace_schema": SCHEMA_D1_TRACE,
        "trace_step_fields": list(D1_TRACE_STEP_FIELDS),
        "recomputability_rule": D1_TRACE_RECOMPUTABILITY_RULE,
    }


def d1_main(argv: Sequence[str] | None = None) -> int:
    """Collector CLI; refuses sidecars without a verified correspondence."""
    parser = argparse.ArgumentParser(prog="collect_mechanism_data", description=__doc__)
    parser.add_argument("--platform", choices=("libero", "robotwin2"), required=True)
    parser.add_argument("--npz", required=True, help="recorded episode npz")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=4, help="frames to probe")
    parser.add_argument("--sim-probe", action="store_true", help="attempt the simulator checks")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write only the collection plan plus any explicitly requested --trace-steps trace",
    )
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=0,
        help="collect an executed trace of this many steps (0 = plan only, max 8)",
    )
    parser.add_argument(
        "--capture-measurements",
        action="store_true",
        help="capture paired measurements and geometry; needs --trace-steps",
    )
    parser.add_argument(
        "--warmstart-mode",
        choices=D1_TRACE_WARMSTART_MODES,
        default="saved_before_step",
        help="how the solver warm-start is made reproducible in the trace",
    )
    parser.add_argument(
        "--controller-buffer-policy",
        choices=D1_TRACE_CONTROLLER_BUFFER_POLICIES,
        default="recorded_initial_conditions",
        help="explicit statement about the controller/observable buffers of the trace",
    )
    args = parser.parse_args(argv)
    if args.capture_measurements and args.trace_steps <= 0:
        parser.error("--capture-measurements requires --trace-steps > 0")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = d1_npz_field_probe(args.npz)
    notes: list[str] = []
    verified = False
    if args.platform == "libero":
        states = probe.get("state_dim")
        actions = probe.get("actions_per_frame")
        if states and actions:
            notes.append("recorded `states` and per-frame `actions` present in the npz")
            if args.sim_probe:
                verified, sim_notes = _probe_libero_correspondence(args.npz, limit=args.limit)
                notes.extend(sim_notes)
            else:
                notes.append("simulator check not requested (--sim-probe); sidecar refused")
        else:
            notes.append("missing `states` or per-frame `actions`; sidecar refused")
    else:
        if probe.get("robot_state_dim") and probe.get("actions_per_frame"):
            notes.append("recorded `robot_states` and per-frame `actions` present in the npz")
        if args.sim_probe:
            verified, sim_notes = _probe_robotwin_correspondence(args.npz, limit=args.limit)
            notes.extend(sim_notes)
        else:
            notes.append("simulator check not requested (--sim-probe); sidecar refused")
    plan = d1_plan(
        platform=args.platform,
        npz_path=args.npz,
        output_dir=output_dir,
        correspondence_verified=verified,
        correspondence_notes=notes,
        limit=args.limit,
    )
    plan["npz_probe"] = probe
    if args.trace_steps > 0:
        if args.platform != "libero":
            print(
                "trace collection needs a live simulator; robotwin2 is not available this round",
                file=sys.stderr,
            )
            return 3
        trace_name = Path(plan["target"]["name"]).with_suffix("").name
        try:
            trace = d1_collect_libero_trace(
                args.npz,
                output_dir=output_dir,
                name=trace_name,
                steps=args.trace_steps,
                capture_measurements=args.capture_measurements,
                warmstart_mode=args.warmstart_mode,
                controller_buffer_policy=args.controller_buffer_policy,
                source_fidelity={
                    "level": (
                        "phase_only"
                        if plan["target"]["mode"] == "sidecar"
                        else "not_applicable_new_capture"
                    ),
                    "basis": (
                        "S0 action-phase verification: k=+1 median 1.6e-4..7.1e-4 and 70-300x "
                        "better than the wrong lags, but bit-exact replay of the external "
                        "recording is not achievable because it does not store qacc_warmstart "
                        "or the controller buffers"
                    ),
                },
            )
        except Exception as error:  # pragma: no cover - environment dependent
            print(f"trace collection failed: {error!r}", file=sys.stderr)
            return 4
        plan["measurement_capture"] = trace.get("measurement_capture")
        plan["trace"] = {
            "record": trace["trace_record"],
            "npz": trace["npz"],
            "npz_sha256": trace["npz_sha256"],
            "steps": trace["steps"],
            "warmstart_mode": trace["warmstart_mode"],
            "recomputability": trace["recomputability"],
        }
    path = output_dir / f"collection-plan-{Path(args.npz).stem}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "plan": str(path),
                "mode": plan["target"]["mode"],
                "trace": (plan.get("trace") or {}).get("recomputability"),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if args.trace_steps > 0:
        trace_ok = plan["trace"]["recomputability"]["bit_exact_recomputable"]
        capture = plan.get("measurement_capture")
        return 0 if trace_ok and (capture is None or capture["replay_verified"]) else 1
    if not verified:
        print(
            "correspondence not verified: no sidecar written; "
            f"extension set name would be {plan['target']['name']}",
            file=sys.stderr,
        )
        return 2
    print("plan only: no measurement dataset collected", file=sys.stderr)
    return 2


def _libero_env_from_npz(
    npz_path: str | Path, *, env_seed: int = 0
) -> tuple[Any, dict[str, Any], Path]:
    """Build the LIBERO environment declared by an episode npz (frozen simulator env only)."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    with np.load(npz_path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    suite = benchmark.get_benchmark_dict()[str(metadata.get("suite", "libero_goal"))]()
    task = suite.get_task(int(metadata.get("task_id", 0)))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(int(env_seed))
    return env, metadata, bddl


def _apply_warmstart_policy(env: Any, mode: str, *, saved: Any = None) -> None:
    """Apply the declared solver warm-start policy to a live MuJoCo-backed environment."""
    import mujoco

    warmstart = env.sim.data._data.qacc_warmstart
    if mode == "disabled_by_flag":
        model = env.sim.model._model
        model.opt.disableflags = int(model.opt.disableflags) | int(
            mujoco.mjtDisableBit.mjDSBL_WARMSTART
        )
    elif mode == "zeroed_each_step":
        warmstart[:] = 0.0
    elif mode == "saved_before_step":
        if saved is None:
            raise ValueError("saved_before_step needs the recorded qacc_warmstart_before array")
        warmstart[:] = np.asarray(saved, dtype=np.float64)
    else:  # pragma: no cover - guarded by the CLI choices
        raise ValueError(f"unknown warmstart mode {mode!r}")


def d1_collect_libero_trace(
    npz_path: str | Path,
    *,
    output_dir: str | Path,
    name: str,
    steps: int = 3,
    warmstart_mode: str = "saved_before_step",
    controller_buffer_policy: str = "recorded_initial_conditions",
    env_seed: int = 0,
    verify: bool = True,
    source_fidelity: dict[str, Any] | None = None,
    capture_measurements: bool = False,
) -> dict[str, Any]:
    """Collect a bounded executed trace from a recorded LIBERO episode.

    Each step records the pre-state, the command actually handed to ``env.step``, the
    post-state, and the declared warm-start and controller-buffer policies.  With
    ``verify=True`` the whole recipe is re-run in a second fresh environment and compared,
    which is what turns "bit-exact recomputable" into a measured claim rather than a promise.
    """
    npz_path = Path(npz_path)
    for suffix in (".npz", ".trace.json", ".measurements.npz", ".measurements.json"):
        if (Path(output_dir) / f"{name}{suffix}").exists():
            raise FileExistsError(Path(output_dir) / f"{name}{suffix}")
    if not 1 <= int(steps) <= 8:
        raise ValueError("bounded trace collection: steps must be within [1, 8]")
    with np.load(npz_path, allow_pickle=True) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float32)
    if states.shape[0] < int(steps) + 1:
        raise ValueError("not enough recorded frames for the requested trace length")

    def apply_policy(target: Any) -> None:
        if warmstart_mode == "disabled_by_flag":
            _apply_warmstart_policy(target, warmstart_mode)

    env, metadata, bddl = _libero_env_from_npz(npz_path, env_seed=env_seed)
    records: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []
    measurement_errors: dict[str, float] = {}
    if capture_measurements:
        from owac.raw.n04_d1_capture import capture_frame, configure_renderer, write_capture
    try:
        apply_policy(env)
        env.reset()
        if capture_measurements:
            configure_renderer(env)
        env.set_init_state(states[0])
        for index in range(int(steps)):
            if capture_measurements:
                captured.append(capture_frame(env, frame_index=index, noise_seed=env_seed))
            if warmstart_mode == "zeroed_each_step":
                _apply_warmstart_policy(env, warmstart_mode)
            step: dict[str, Any] = {
                "step_index": index,
                "pre_state": np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
                "issued_command": np.clip(
                    np.array(actions[index], dtype=np.float32, copy=True), -1.0, 1.0
                ),
            }
            if warmstart_mode == "saved_before_step":
                step["qacc_warmstart_before"] = np.array(
                    env.sim.data._data.qacc_warmstart, dtype=np.float64, copy=True
                )
            env.step(step["issued_command"].tolist())
            step["post_state"] = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            if warmstart_mode == "saved_before_step":
                step["qacc_warmstart_after"] = np.array(
                    env.sim.data._data.qacc_warmstart, dtype=np.float64, copy=True
                )
            records.append(step)
    finally:
        env.close()

    verification = None
    if verify:
        env_second, _, _ = _libero_env_from_npz(npz_path, env_seed=env_seed)
        try:
            apply_policy(env_second)
            env_second.reset()
            if capture_measurements:
                configure_renderer(env_second)
            env_second.set_init_state(records[0]["pre_state"])
            differences: list[float] = []
            for index, step in enumerate(records):
                if capture_measurements:
                    replay_frame = capture_frame(env_second, frame_index=index, noise_seed=env_seed)
                    for key, value in captured[index].items():
                        other = replay_frame[key]
                        error = (
                            float(
                                np.max(
                                    np.abs(value.astype(float) - other.astype(float)), initial=0.0
                                )
                            )
                            if value.shape == other.shape
                            else float("inf")
                        )
                        measurement_errors[key] = max(measurement_errors.get(key, 0.0), error)
                _apply_warmstart_policy(
                    env_second, warmstart_mode, saved=step.get("qacc_warmstart_before")
                )
                env_second.step(np.asarray(step["issued_command"], dtype=np.float32).tolist())
                replayed = np.asarray(env_second.sim.get_state().flatten(), dtype=np.float64)
                differences.append(float(np.max(np.abs(replayed - step["post_state"]))))
            verification = {
                "method": "re-run the recorded recipe in a fresh environment and re-apply the "
                "recorded warm-start policy, then compare with the collected post_state",
                "steps": len(differences),
                "max_abs_diff": max(differences),
                "tolerance": 1e-12,
                "bit_exact": max(differences) <= 1e-12,
            }
        finally:
            env_second.close()

    recipe = {
        "env": {
            "bddl_file": str(bddl),
            "task_name": metadata.get("task_name"),
            "suite": metadata.get("suite"),
            "task_id": metadata.get("task_id"),
            "camera_heights": 256,
            "camera_widths": 256,
            "env_seed": int(env_seed),
        },
        "ordered_operations": [
            "build env and call env.seed(env_seed)",
            "apply warmstart policy at build time when warmstart_mode=disabled_by_flag",
            "env.reset()",
            "env.set_init_state(pre_state[0]) once; thereafter continuous execution",
            "per step t: apply warmstart policy; env.step(issued_command[t]); read post_state[t]",
        ],
        "source_npz": str(npz_path),
        "source_frames_used": [step["step_index"] for step in records],
        "warmstart_mode": warmstart_mode,
        "issued_command_source": "collector_step_call_argument",
        "measurement_renderer": "offsamples=0; pixel-index principal point=127.5"
        if capture_measurements
        else None,
    }
    result = d1_write_trace(
        output_dir,
        name,
        platform="libero",
        source_npz=npz_path,
        steps=records,
        recipe=recipe,
        warmstart_mode=warmstart_mode,
        controller_buffer_policy=controller_buffer_policy,
        issued_command_source="collector_step_call_argument",
        controller_buffer_notes=[
            "the trace is collected from a fresh, reset environment whose construction, seed and "
            "ordered operations are recorded in `recipe`",
            "the controller/observable buffers therefore start from the post-reset state and are "
            "reproduced by re-running the same recipe",
            "this does not transfer to the original external recording: its own buffers are not "
            "stored in the npz (see source_fidelity)",
        ],
        source_fidelity=source_fidelity,
        verification=verification,
    )
    if capture_measurements:
        result["measurement_capture"] = write_capture(
            Path(output_dir),
            name,
            captured,
            replay_max_errors=measurement_errors,
            trace=result,
            noise_seed=env_seed,
        )
    return result


def _probe_libero_correspondence(npz_path: str | Path, *, limit: int) -> tuple[bool, list[str]]:
    """Same-version check: replay recorded states and compare the simulator state."""
    notes: list[str] = []
    try:
        env, _metadata, _bddl = _libero_env_from_npz(npz_path)
        states = np.asarray(np.load(npz_path, allow_pickle=True)["states"])
    except Exception as error:  # pragma: no cover - environment dependent
        return False, [f"LIBERO simulator unavailable: {error!r}"]
    try:
        for index in range(min(limit, states.shape[0])):
            env.set_init_state(states[index])
            now = np.asarray(env.sim.get_state().flatten())
            if now.shape != states[index].shape:
                notes.append(f"state dim mismatch at frame {index}")
                return False, notes
            difference = float(np.max(np.abs(now - states[index])))
            notes.append(f"frame {index}: max |state_replay - recorded| = {difference:.3e}")
            if difference > 1e-9:
                return False, notes
    finally:
        env.close()
    notes.append(
        "state roundtrip is exact within 1e-9 for the probed frames; this does NOT "
        "verify measurement/action correspondence, lighting or controller history"
    )
    return False, notes


def _probe_robotwin_correspondence(npz_path: str | Path, *, limit: int) -> tuple[bool, list[str]]:
    """RoboTwin has no recorded full physics state; only declared joint paths can be replayed."""
    notes = [
        "RoboTwin npz records actions and robot_states but no full physics state; "
        "exact frame correspondence cannot be confirmed from this file alone",
    ]
    with np.load(npz_path, allow_pickle=True) as data:
        if "robot_states" not in data.files:
            return False, notes + ["`robot_states` missing"]
        states = np.asarray(data["robot_states"])
        notes.append(f"robot_states shape {list(states.shape)}")
    _ = limit
    return False, notes


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m owac.raw.n04_mechanism_data",
        description="N04 S0 D0 analytic 3-D mechanism data and D1 collection feasibility",
    )
    parser.add_argument("--config", required=True, help="configs/n04/full/s0_data.json")
    parser.add_argument(
        "--output",
        required=True,
        help="output directory, for example <run_dir>/mechanism-data",
    )
    parser.add_argument("--frames", type=int, default=None, help="override D0 frame count")
    parser.add_argument("--batch", type=int, default=None, help="override D0 batch size")
    parser.add_argument("--audit-batches", default="1,2,4", help="comma separated batch sizes")
    parser.add_argument(
        "--remote-evidence",
        action="append",
        default=[],
        help="JSON file with observed remote evidence for the D1 feasibility report",
    )
    parser.add_argument("--skip-d0", action="store_true", help="only build D1 outputs")
    parser.add_argument("--no-write", action="store_true", help="render without writing npz files")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides: dict[str, Any] = {}
    if args.frames is not None:
        overrides["frames"] = int(args.frames)
    if args.batch is not None:
        overrides["batch"] = int(args.batch)
    d0_config = D0Config.from_config(config, **overrides)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] | None = None
    if not args.skip_d0:
        batches = [int(value) for value in args.audit_batches.split(",") if value]
        manifest = generate_d0(
            d0_config, output_dir, audit_batches=batches, write=not args.no_write
        )

    factors = {
        "schema": SCHEMA_FACTORS,
        "order": "single factors first, then declared combinations",
        "single_factors": list(DEFAULT_FACTORS),
        "combinations": ["+".join(parts) for parts in DEFAULT_COMBINATIONS],
        "table": [
            {
                "factor": name,
                **{
                    key: value
                    for key, value in asdict(resolve_plan(name)).items()
                    if key not in {"name"}
                },
            }
            for name in default_factor_names()
        ],
        "config_factors": list(d0_config.factors),
        "coverage_complete": set(d0_config.factors).issubset(set(DEFAULT_FACTORS)),
    }
    (output_dir / "factors.json").write_text(
        json.dumps(factors, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    evidence: dict[str, Any] = {}
    for path in args.remote_evidence:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        evidence.update(payload)
    feasibility = build_feasibility(evidence, checked_at=_utc_now())
    d1_dir = output_dir / "d1"
    emit_collector(d1_dir)
    (d1_dir / "feasibility.json").write_text(
        json.dumps(feasibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_record = {
        "schema": "n04-mechanism-data-run-v1",
        "generated_at_utc": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "output_dir": str(output_dir),
        "git_head": _git_output(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_output(["status", "--porcelain"])),
        "d0_factors": len(default_factor_names()),
        "d1_evidence_files": list(args.remote_evidence),
        "d0_manifest": manifest is not None,
        "audit_passed": None if manifest is None else bool(manifest["audit"]["passed"]),
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "factors": len(default_factor_names()),
                "audit_passed": run_record["audit_passed"],
                "feasibility": feasibility["verdict_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if run_record["audit_passed"] in (True, None) else 1


def _utc_now() -> str:
    # datetime.UTC is Python 3.11+; the D1 collector runs in the frozen 3.10 simulator envs.
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def _git_output(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

"""N02 full (v1.1) P0 controlled synthetic measurement generator.

Scope
-----
This module builds the *new mechanism data* the v1.1 plan's P0 section asks for:
a reproducible layered 2D/2.5D scene program, an event program whose per-class
times are independent, and two declared sensor measurement models. It is a
controlled mechanism set for numeric and inference work, not embodied evidence,
and it does not claim that the P1/P2 data targets are built.

Boundaries kept explicit
------------------------
* ``bayer_cfa`` sensors are rendered as a mosaic: every pixel measures exactly
  one colour channel through that plane's filter response, at the pixel-centre
  convention ``(row + 0.5, col + 0.5)``. No sRGB image is produced anywhere and
  no demosaicing is inverted to pretend a measurement. Cropping a mosaic
  re-declares the *window's* CFA phase and moves the sample coordinates; the
  plane/coordinate convention itself is delegated to
  :mod:`owac.raw.rawvla_sequence_n02` (``cfa_plane_values``/``cfa_plane_positions``).
* ``rawvla_rgb3`` sensors are a *different* observation model: three co-sited
  pseudo-RAW channels stored as float32 DN with their own declarations. The two
  models are never mixed inside one sequence and are calibrated separately.
* Inference payloads carry RAW measurements, the declared calibration, available
  embodiment context, language tokens, executed-action history and a declared
  prior. Evaluation truth and future actions live in separate objects and never
  alias the inference payloads; ``generate_sequence`` copies everything that
  enters an :class:`~owac.frameworks.n02_full.state.InferenceInput`.
* Missing samples and known-bad pixels are excluded from the measurement terms
  (``contributing_mask``); saturated samples contribute with an interval/tail
  probability instead of being treated as exact observations.

Seeds
-----
Scene seeds are derived from ``partition`` + ``scene_id`` only, so they are
independent of the model seeds ``(1103, 2207, 3301)`` the plan fixes for
training. Every random stream (scene layout, per-event-class times, per-frame
sensor noise, action policy) is derived from its own text part, so changing one
stream cannot move another.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
import tracemalloc
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from owac.core.encoder_contract_n02 import CameraSpec, EmbodimentContext, MeasurementFrame
from owac.raw.rawvla_sequence_n02 import (
    CFA_PATTERNS,
    PIXEL_CENTRE_CONVENTION,
    CfaCalibration,
    apply_cfa_calibration,
    cfa_plane_positions,
    cfa_plane_values,
)

from .state import InferenceInput, ObservationBundle, TrainingTargets

__all__ = [
    "EVENT_TYPES",
    "MODEL_SEEDS",
    "P1_TRAIN_FRAME_TARGET",
    "P2_SEQUENCE_FRAMES",
    "P2_SEQUENCE_TARGET",
    "REFERENCE_EXPOSURE_MS",
    "SYNTHETIC_VERSION",
    "AppearanceEvent",
    "CameraShiftEvent",
    "DatasetCounters",
    "EvaluationTruth",
    "EventSchedule",
    "ExistenceEvent",
    "ExposureEvent",
    "IdentitySwapEvent",
    "InteractionPlan",
    "LightingPlan",
    "MeasurementModel",
    "MotionEvent",
    "NoiseEvent",
    "OcclusionEvent",
    "OcclusionPlan",
    "PartitionVocabulary",
    "RelationEvent",
    "RenderShape",
    "RenderState",
    "SceneSpec",
    "SensorMeasurement",
    "SensorSpec",
    "ShapeSpec",
    "SyntheticDataset",
    "SyntheticSequence",
    "build_event_schedule",
    "cfa_plane_colour_channels",
    "cfa_plane_tile_indices",
    "cfa_pixel_plane_index",
    "crop_sensor_measurement",
    "derive_seed",
    "generate_dataset",
    "generate_scene",
    "generate_sequence",
    "measure_cost",
    "measurement_hash",
    "quantisation_interval_probability",
    "rotate_cfa_pattern",
    "saturation_tail_probability",
    "scene_hash",
    "scene_seed",
    "schedule_hash",
    "schedule_seed",
    "sequences_for_frame_target",
    "shape_footprint_radius",
    "view_seed",
]

#: Version of the generator contract. Any change to the emitted measurement
#: semantics must change this string, because hashes are built on top of it.
SYNTHETIC_VERSION = "n02-full-synthetic-v1"

#: Model seeds fixed by the v1.1 plan. Scene seeds are derived elsewhere and are
#: never drawn from this set.
MODEL_SEEDS = (1103, 2207, 3301)

#: Declared shape vocabulary. Simple geometry is the numeric mechanism set.
SHAPE_KINDS = ("disc", "square", "triangle", "ring")

#: Declared sensor input modes. One sequence uses one mode for all its cameras.
SENSOR_MODES = ("bayer_cfa", "rawvla_rgb3")

#: Measurement-model kinds; ``cfa`` pairs with ``bayer_cfa`` and ``raw3`` with
#: ``rawvla_rgb3``. They are declared as two different observation models.
MODEL_KINDS = ("cfa", "raw3")

#: Declared partitions. ``test`` is the sealed mechanism test set.
PARTITIONS = ("train", "dev", "calibration", "test")

#: Every event class the schedule can carry.
EVENT_TYPES = (
    "exposure_change",
    "noise_change",
    "camera_shift",
    "object_motion",
    "occlusion",
    "enter_leave",
    "false_merge",
    "false_split",
    "identity_swap",
    "relation_change",
)

#: Event class name -> :class:`EventSchedule` field name.
EVENT_FIELDS: dict[str, str] = {
    "exposure_change": "exposure",
    "noise_change": "noise",
    "camera_shift": "camera_shift",
    "object_motion": "object_motion",
    "occlusion": "occlusion",
    "enter_leave": "existence",
    "false_merge": "false_merge",
    "false_split": "false_split",
    "identity_swap": "identity_swap",
    "relation_change": "relation",
}

#: Exposure scale reference: ``exposure_ms / REFERENCE_EXPOSURE_MS`` multiplies
#: the declared scene radiance; with 1.0 ms and gain 1.0 a fully reflective
#: surface sits at the full-well signal 1.0.
REFERENCE_EXPOSURE_MS = 1.0

#: Signal unit of the declared sensor model: 1.0 == saturation_level default.
SIGNAL_UNIT = "normalised_full_well_signal"

#: Declared build targets of the plan (P1 >= 1e6 training frames, P2 onwards
#: >= 1e5 sequences of 16..64 frames). These are *targets*, never measurements;
#: P0 only claims that small samples run.
P1_TRAIN_FRAME_TARGET = 1_000_000
P2_SEQUENCE_TARGET = 100_000
P2_SEQUENCE_FRAMES = (16, 64)

#: Pixel-centre convention string reused from the RawVLA CFA tools.
PIXEL_CENTRE = PIXEL_CENTRE_CONVENTION


# --------------------------------------------------------------------------------------
# seeds and hashing
# --------------------------------------------------------------------------------------


def derive_seed(*parts: Any) -> int:
    """Deterministic 64-bit seed from declared text parts (never from floats)."""
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def scene_seed(partition: str, scene_id: str) -> int:
    """Scene seed derived from ``partition`` + ``scene_id`` only."""
    return derive_seed("n02-full-scene", partition, scene_id)


def view_seed(partition: str, scene_id: str, view_index: int) -> int:
    """Seed of one view of one scene; independent of the scene layout stream."""
    return derive_seed("n02-full-view", partition, scene_id, view_index)


def schedule_seed(partition: str, scene_id: str) -> int:
    """Seed of the event schedule of one scene."""
    return derive_seed("n02-full-schedule", partition, scene_id)


def _rng(*parts: Any) -> np.random.Generator:
    """Independent generator for one declared stream."""
    return np.random.default_rng(derive_seed(*parts))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples into JSON values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            spec.name: _jsonable(getattr(value, spec.name)) for spec in dataclasses.fields(value)
        }
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    raise TypeError(f"cannot canonicalise {type(value).__name__}")


def _hash_payload(payload: Any) -> str:
    """Hash canonical JSON text: sorted keys, tight separators, no ASCII escapes.

    The canonicalisation matches ``owac.frameworks.n02_full.state.canonical_json``
    by construction, so a generator hash and a ledger hash never disagree because
    of formatting.
    """
    text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_text(text)


def _hash_array(digest: Any, array: Any) -> None:
    """Feed dtype, shape and exact bytes of one array into a growing hash."""
    values = np.ascontiguousarray(array)
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())


# --------------------------------------------------------------------------------------
# declared scene program
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """Static declaration of one synthetic camera.

    ``black_level``/``white_level`` are in DN of the declared ``bit_depth`` scale
    (for example 64/4095 for a 12-bit sensor). ``read_noise_e`` is in electrons at
    the sensor and ``gain`` converts electrons to DN; see
    :class:`MeasurementModel` for the full noise declaration.
    """

    camera_id: str
    height: int
    width: int
    input_mode: str
    cfa_pattern: str = ""
    black_level: float = 0.0
    white_level: float = 1.0
    bit_depth: int = 12
    read_noise_e: float = 2.0
    gain: float = 1.0
    exposure_ms: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.camera_id):
            raise ValueError("camera_id must be a non-empty string")
        if self.input_mode not in SENSOR_MODES:
            raise ValueError(f"unknown input_mode: {self.input_mode!r}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("sensor height and width must be positive")
        if self.input_mode == "bayer_cfa":
            if self.cfa_pattern not in CFA_PATTERNS:
                raise ValueError(f"bayer_cfa requires a Bayer pattern, got {self.cfa_pattern!r}")
            if self.height % 2 or self.width % 2:
                raise ValueError("a mosaic sensor needs even height and width")
        elif self.cfa_pattern:
            raise ValueError("rawvla_rgb3 has no CFA pattern; leave it empty")
        if not 1 <= int(self.bit_depth) <= 16:
            raise ValueError("bit_depth must be in 1..16 for the declared DN scale")
        if self.white_level <= self.black_level:
            raise ValueError("white_level must exceed black_level")
        if self.read_noise_e < 0.0 or not math.isfinite(self.read_noise_e):
            raise ValueError("read_noise_e must be finite and non-negative")
        if self.gain <= 0.0 or self.exposure_ms <= 0.0:
            raise ValueError("gain and exposure_ms must be positive")

    @property
    def channels(self) -> int:
        """1 for a mosaic, 3 for the co-sited pseudo-RAW model."""
        return 1 if self.input_mode == "bayer_cfa" else 3

    @property
    def is_mosaic(self) -> bool:
        return self.input_mode == "bayer_cfa"

    @property
    def dn_scale(self) -> float:
        """Full-scale DN span of this sensor."""
        return float(self.white_level - self.black_level)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def camera_spec(self) -> CameraSpec:
        """Bridge to the shared contract's :class:`CameraSpec`."""
        return CameraSpec(
            camera_id=self.camera_id,
            height=self.height,
            width=self.width,
            channels=self.channels,
            input_mode=self.input_mode,
            cfa_pattern=self.cfa_pattern,
            black_level=float(self.black_level),
            white_level=float(self.white_level),
        )


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    """One controlled object: simple geometry, an initial pose and a colour.

    ``position_px`` is the shape centre at frame 0 in declared scene coordinates
    ``(y, x)`` whose origin is the canvas corner (pixel centres at ``+0.5``).
    ``colour_response`` is the per-channel reflectance of the surface in
    ``(R, G, B)``; it is not an sRGB colour and never becomes one here.
    """

    shape_id: str
    kind: str
    scale_px: float
    position_px: tuple[float, float]
    velocity_px_per_step: tuple[float, float]
    colour_response: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.kind not in SHAPE_KINDS:
            raise ValueError(f"unknown shape kind: {self.kind!r}")
        if self.scale_px <= 0.0:
            raise ValueError("scale_px must be positive")
        if len(tuple(self.velocity_px_per_step)) != 2:
            raise ValueError("velocity_px_per_step must be (dy, dx)")
        if len(tuple(self.colour_response)) != 3:
            raise ValueError("colour_response must be the three (R, G, B) coefficients")
        if any(not 0.0 <= float(item) <= 1.0 for item in self.colour_response):
            raise ValueError("colour_response coefficients must lie in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class OcclusionPlan:
    """Declared scripted occlusion: the occluder sweeps across the occluded shape.

    The sweep only steers the simulated positions; whether the occluded object is
    actually hidden is computed geometrically per camera and recorded in the
    truth, never declared by this plan.
    """

    occluder_id: str
    occluded_id: str
    start_step: int
    end_step: int
    mode: str = "sweep_x"

    def __post_init__(self) -> None:
        if self.mode not in {"sweep_x", "sweep_y"}:
            raise ValueError(f"unknown occlusion sweep mode: {self.mode!r}")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("occlusion window must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class LightingPlan:
    """Declared illumination program: piecewise-constant multipliers over frames."""

    base_illumination: float
    steps: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if self.base_illumination <= 0.0:
            raise ValueError("base_illumination must be positive")
        for step, value in self.steps:
            if int(step) < 0 or float(value) <= 0.0:
                raise ValueError("lighting steps need step >= 0 and a positive multiplier")

    def scale_at(self, frame_index: int) -> float:
        """Illumination multiplier in force at ``frame_index``."""
        value = float(self.base_illumination)
        for step, multiplier in self.steps:
            if int(step) <= frame_index:
                value *= float(multiplier)
        return value


@dataclass(frozen=True, slots=True)
class InteractionPlan:
    """Declared relation that holds between two objects during a frame window."""

    predicate: str
    subject_id: str
    object_id: str
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("interaction window must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """A reproducible layered scene program.

    ``shapes`` are listed back-to-front: a later entry is painted over an earlier
    one, so the layer order that decides real occlusion is part of the scene.
    ``occlusion_plan``/``lighting_plan``/``interaction_plan`` declare the scene
    program; :class:`EventSchedule` carries the realised discrete times.
    """

    scene_id: str
    seed: int
    partition: str
    shapes: tuple[ShapeSpec, ...]
    background_level: float
    camera_motion_px: tuple[float, float]
    occlusion_plan: tuple[OcclusionPlan, ...] = ()
    lighting_plan: LightingPlan = LightingPlan(base_illumination=1.0)
    interaction_plan: tuple[InteractionPlan, ...] = ()

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {self.partition!r}")
        if not 1 <= len(self.shapes) <= 12:
            raise ValueError("a scene needs between 1 and 12 shapes")
        ids = [shape.shape_id for shape in self.shapes]
        if len(set(ids)) != len(ids):
            raise ValueError("shape ids must be unique inside a scene")
        if not 0.0 <= float(self.background_level) <= 1.0:
            raise ValueError("background_level must lie in [0, 1]")
        if len(tuple(self.camera_motion_px)) != 2:
            raise ValueError("camera_motion_px must be (dy, dx)")
        for plan in (*self.occlusion_plan, *self.interaction_plan):
            for name in ("occluder_id", "occluded_id", "subject_id", "object_id"):
                value = getattr(plan, name, None)
                if value is not None and value not in ids:
                    raise ValueError(f"plan refers to unknown shape id {value!r}")

    @property
    def shape_ids(self) -> tuple[str, ...]:
        return tuple(shape.shape_id for shape in self.shapes)

    @property
    def shape_kinds(self) -> tuple[str, ...]:
        return tuple(shape.kind for shape in self.shapes)

    def colour_map(self) -> dict[str, tuple[float, float, float]]:
        return {shape.shape_id: tuple(shape.colour_response) for shape in self.shapes}

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


# --------------------------------------------------------------------------------------
# CFA phase tools (aligned with owac.raw.rawvla_sequence_n02)
# --------------------------------------------------------------------------------------


def cfa_plane_tile_indices(pattern: str) -> tuple[int, int, int, int]:
    """Tile index (row-major, 0..3) of the ``(R, G1, G2, B)`` planes.

    This mirrors the rule the RawVLA tools use: ``R`` and ``B`` take their only
    occurrences and the two greens keep first/second identity, so the plane order
    is stable across all four Bayer phases.
    """
    if pattern not in CFA_PATTERNS:
        raise ValueError(f"unknown CFA pattern: {pattern!r}")
    tiles = CFA_PATTERNS[pattern]
    return (
        tiles.index("R"),
        tiles.index("G"),
        tiles.index("G", tiles.index("G") + 1),
        tiles.index("B"),
    )


def cfa_plane_colour_channels() -> tuple[int, int, int, int]:
    """Scene colour channel ``(R, G, B)`` of each ``(R, G1, G2, B)`` plane."""
    return (0, 1, 1, 2)


def cfa_pixel_plane_index(pattern: str, height: int, width: int) -> np.ndarray:
    """``[H, W]`` plane index (0..3, ``R/G1/G2/B``) measured at each pixel."""
    plane_of_tile = [0, 0, 0, 0]
    for plane, tile in enumerate(cfa_plane_tile_indices(pattern)):
        plane_of_tile[tile] = plane
    rows = np.arange(height, dtype=np.int64) % 2
    cols = np.arange(width, dtype=np.int64) % 2
    return np.asarray(plane_of_tile, dtype=np.int8)[rows[:, None] * 2 + cols[None, :]]


def rotate_cfa_pattern(pattern: str, phase_y: int, phase_x: int) -> str:
    """Window pattern of a window whose ``(0, 0)`` sits at ``(phase_y, phase_x)``.

    A crop at an odd row or column samples a different tile position, so the
    window has its own Bayer phase. The returned name is the pattern whose own
    phase equals the cropped window, which is exactly what
    ``cfa_plane_values`` requires.
    """
    if pattern not in CFA_PATTERNS:
        raise ValueError(f"unknown CFA pattern: {pattern!r}")
    flat = CFA_PATTERNS[pattern]
    rotated = tuple(
        flat[2 * ((tile // 2 + phase_y) % 2) + ((tile % 2 + phase_x) % 2)] for tile in range(4)
    )
    for name, tiles in CFA_PATTERNS.items():
        if tiles == rotated:
            return name
    raise AssertionError(f"rotated pattern {rotated!r} is not a Bayer pattern")


def _phase_of_offset(row0: int, col0: int) -> tuple[int, int]:
    """Parity of a crop origin; the crop phase that drives the window pattern."""
    return int(row0) % 2, int(col0) % 2


# --------------------------------------------------------------------------------------
# declared sensor measurements
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorMeasurement:
    """One camera's declared measurement for one frame.

    ``values`` keeps the raw sampling: ``uint16`` DN mosaic for a CFA sensor and
    ``float32`` DN co-sited pseudo-RAW for the RGB3 model. ``unit``/``dtype_name``
    declare what the numbers are; ``saturated_mask`` marks clipped samples and
    ``saturation_tail_probability`` carries P(noisy signal >= saturation_level)
    under the declared normal approximation, so a saturation is an interval/tail
    statement rather than a silently exact value. Missing samples and known-bad
    pixels never contribute a measurement term.
    """

    camera_id: str
    input_mode: str
    values: np.ndarray
    unit: str
    dtype_name: str
    valid_mask: np.ndarray
    bad_mask: np.ndarray
    missing_mask: np.ndarray
    saturated_mask: np.ndarray
    saturation_tail_probability: np.ndarray
    frame_index: int
    crop_offset: tuple[int, int] = (0, 0)
    cfa_pattern: str = ""
    plane_positions: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.values.shape)

    def contributing_mask(self) -> np.ndarray:
        """Samples that contribute a measurement term (missing/bad excluded)."""
        return ~(np.asarray(self.missing_mask) | np.asarray(self.bad_mask))

    def contributing_count(self) -> int:
        return int(self.contributing_mask().sum())

    def calibrated(self, dtype: Any = np.float32) -> tuple[np.ndarray, dict]:
        """Black/white normalisation with the RawVLA reference conventions.

        Delegates to :func:`owac.raw.rawvla_sequence_n02.apply_cfa_calibration`: missing
        and bad samples become NaN because an absent sample is not a measurement,
        saturated samples keep their clipped value but stay flagged.
        """
        calibration = CfaCalibration(
            black_level=float(self.metadata["black_level"]),
            white_level=float(self.metadata["white_level"]),
            missing_mask=np.asarray(self.missing_mask),
            bad_mask=np.asarray(self.bad_mask),
            saturated_mask=np.asarray(self.saturated_mask),
            analog_gain=float(self.metadata["gain"]),
            exposure_s=float(self.metadata["exposure_ms"]) / 1000.0,
        )
        return apply_cfa_calibration(self.values, calibration, dtype=dtype)

    def plane_set(self) -> Any:
        """Four Bayer planes with true coordinates (CFA measurements only)."""
        if self.input_mode != "bayer_cfa":
            raise ValueError("plane_set() is only defined for bayer_cfa measurements")
        return cfa_plane_values(self.values, self.cfa_pattern, self.crop_offset)

    def summary(self) -> dict[str, Any]:
        """JSON-safe counts and declarations; never claims an unmeasured value."""
        return {
            "camera_id": self.camera_id,
            "input_mode": self.input_mode,
            "frame_index": self.frame_index,
            "values_shape": list(self.shape),
            "dtype": self.dtype_name,
            "unit": self.unit,
            "crop_offset": list(self.crop_offset),
            "cfa_pattern": self.cfa_pattern,
            "missing_count": int(np.asarray(self.missing_mask).sum()),
            "bad_count": int(np.asarray(self.bad_mask).sum()),
            "saturated_count": int(np.asarray(self.saturated_mask).sum()),
            "contributing_count": self.contributing_count(),
            "saturation_tail_probability_max": float(
                np.nanmax(np.asarray(self.saturation_tail_probability, dtype=np.float32))
            ),
        }

    def as_measurement_frame(
        self,
        *,
        episode_id: str,
        frame_id: str,
        anchor_id: str,
    ) -> MeasurementFrame:
        """Shared-contract view of this measurement (no truth, no future)."""
        return MeasurementFrame(
            values=self.values,
            valid_mask=self.valid_mask,
            bad_mask=self.bad_mask,
            saturated_mask=self.saturated_mask,
            sample_xy=None if self.plane_positions is None else np.array(self.plane_positions),
            episode_ids=(episode_id,),
            frame_ids=(frame_id,),
            camera_id=self.camera_id,
            timestamp=None,
            time_unit="frame",
            new_measurement=True,
            metadata={
                "synthetic_version": SYNTHETIC_VERSION,
                "unit": self.unit,
                "dtype": self.dtype_name,
                "calibration": {
                    "black_level": self.metadata["black_level"],
                    "white_level": self.metadata["white_level"],
                    "bit_depth": self.metadata["bit_depth"],
                },
                "camera": {
                    "cfa_pattern": self.cfa_pattern,
                    "crop_offset": list(self.crop_offset),
                    "pixel_centre_convention": PIXEL_CENTRE,
                },
                "noise": {
                    "expression": self.metadata["noise_expression"],
                    "shot_noise_scale": self.metadata["shot_noise_scale"],
                    "read_noise_e": self.metadata["read_noise_e"],
                },
                "saturation": {
                    "level": self.metadata["saturation_level"],
                    "semantics": self.metadata["saturation_semantics"],
                },
                "anchor_id": anchor_id,
            },
            metadata_available={
                "calibration": True,
                "gain": True,
                "exposure": True,
                "camera_transform": False,
                "saturation_tail_probability": True,
            },
        )


def crop_sensor_measurement(
    measurement: SensorMeasurement,
    *,
    row0: int,
    col0: int,
    height: int,
    width: int,
) -> SensorMeasurement:
    """Crop one measurement, moving phase and coordinates with the crop.

    For a mosaic the window keeps its own Bayer phase: the pattern is rotated by
    the crop parity and ``crop_offset`` records where the window sits in the full
    sensor, so ``plane_set()`` still returns the true plane coordinates. Mask
    counts and the tail-probability array are recomputed/sliced for the window.
    """
    if row0 < 0 or col0 < 0 or height <= 0 or width <= 0:
        raise ValueError("crop arguments must be non-negative with a positive size")
    values = np.asarray(measurement.values)
    frame_height, frame_width = int(values.shape[-2]), int(values.shape[-1])
    if row0 + height > frame_height or col0 + width > frame_width:
        raise ValueError("crop exceeds the measured frame")
    rows = slice(row0, row0 + height)
    cols = slice(col0, col0 + width)
    base_row = int(measurement.crop_offset[0]) + row0
    base_col = int(measurement.crop_offset[1]) + col0

    if measurement.input_mode == "bayer_cfa":
        if height % 2 or width % 2:
            raise ValueError("a mosaic crop needs an even height and width")
        phase_y, phase_x = _phase_of_offset(row0, col0)
        pattern = rotate_cfa_pattern(measurement.cfa_pattern, phase_y, phase_x)
    else:
        pattern = ""

    def _slice(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        return np.array(array[..., rows, cols])

    values_crop = _slice(values)
    missing = _slice(measurement.missing_mask)
    bad = _slice(measurement.bad_mask)
    saturated = _slice(measurement.saturated_mask)
    tail = _slice(measurement.saturation_tail_probability)
    if measurement.input_mode == "bayer_cfa":
        # An impossible sample keeps its declared storage value, not a NaN cast.
        values_crop = np.where(missing | bad, values_crop.dtype.type(0), values_crop)
    metadata = dict(measurement.metadata)
    metadata["crop_offset"] = [base_row, base_col]
    metadata["missing_count"] = int(missing.sum())
    metadata["bad_count"] = int(bad.sum())
    metadata["saturated_count"] = int(saturated.sum())
    metadata["contributing_count"] = int((~(missing | bad)).sum())
    return dataclasses.replace(
        measurement,
        values=values_crop,
        valid_mask=~missing,
        bad_mask=bad,
        missing_mask=missing,
        saturated_mask=saturated,
        saturation_tail_probability=tail,
        crop_offset=(base_row, base_col),
        cfa_pattern=pattern,
        plane_positions=(
            None if not pattern else cfa_plane_positions(pattern, (base_row, base_col))
        ),
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------
# the two declared measurement models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RenderShape:
    """One paint instruction: a real object or a declared imaging artefact."""

    shape_id: str
    kind: str
    position_px: tuple[float, float]
    scale_px: float
    colour_response: tuple[float, float, float]
    depth: int
    visible: bool = True
    is_artefact: bool = False


@dataclass(frozen=True, slots=True)
class RenderState:
    """Everything the measurement model may read for one frame.

    ``exposure_scale`` is the declared exposure multiplier in force (exposure
    change events), ``illumination`` the scene lighting plan value, and
    ``read_noise_e``/``read_noise_e_source`` the already-resolved read noise so
    the metadata can name where the number came from.
    """

    frame_index: int
    shapes: tuple[RenderShape, ...]
    sensors: tuple[SensorSpec, ...]
    background_level: float
    illumination: float
    exposure_scale: float
    shot_noise_scale: float
    read_noise_e: float
    read_noise_e_source: str
    camera_shift_px: dict[str, tuple[float, float]]
    noise_seed: int


#: Geometry fields a candidate hypothesis may override, using the same names and
#: semantics as the declared world geometry of the public candidate dictionary
#: (``world_px`` is the renderer's ``RenderShape.position_px``).
HYPOTHESIS_GEOMETRY_FIELDS = ("world_px", "scale_px", "depth")


def _state_with_geometry_hypothesis(
    state: RenderState, hypothesis: Mapping[str, Any]
) -> RenderState:
    """Return ``state`` with the hypothesis geometry applied to its declared shapes.

    Only per-entity geometry changes (world position, scale, depth); every appearance,
    visibility, occlusion, exposure, illumination and noise parameter stays exactly as
    rendered.  Unknown entity ids or unknown field names raise instead of being invented.
    """
    if not isinstance(hypothesis, Mapping):
        raise TypeError(f"hypothesis must be a mapping, got {type(hypothesis).__name__}")
    shapes_by_id = {shape.shape_id: shape for shape in state.shapes}
    replacements: dict[str, dict[str, Any]] = {}
    for shape_id, declared in hypothesis.items():
        key = str(shape_id)
        if key not in shapes_by_id:
            raise ValueError(f"hypothesis names an entity not present in the frame state: {key!r}")
        if not isinstance(declared, Mapping):
            raise TypeError(f"hypothesis[{key!r}] must be a mapping of geometry fields")
        update: dict[str, Any] = {}
        for field_name, value in declared.items():
            name = str(field_name)
            if name in ("world_px", "position_px"):
                try:
                    position = tuple(float(item) for item in value)
                except TypeError as error:
                    raise ValueError(f"hypothesis[{key!r}].{name} must be two numbers") from error
                if len(position) != 2:
                    raise ValueError(f"hypothesis[{key!r}].{name} must have two entries")
                update["position_px"] = position
            elif name == "scale_px":
                scale = float(value)
                if not scale > 0.0:
                    raise ValueError(f"hypothesis[{key!r}].scale_px must be positive")
                update["scale_px"] = scale
            elif name == "depth":
                update["depth"] = int(value)
            else:
                raise ValueError(
                    f"hypothesis[{key!r}] field {name!r} is not one of the declared entity "
                    f"geometry fields {HYPOTHESIS_GEOMETRY_FIELDS}"
                )
        if update:
            replacements[key] = update
    if not replacements:
        return state
    shapes = tuple(
        dataclasses.replace(shape, **replacements[shape.shape_id])
        if shape.shape_id in replacements
        else shape
        for shape in state.shapes
    )
    return dataclasses.replace(state, shapes=shapes)


@dataclass(frozen=True, slots=True)
class MeasurementModel:
    """One declared observation model for the controlled mechanism set.

    Units are declared, not implied:

    * scene radiance is unitless in ``[0, 1]`` per channel and is multiplied by
      ``illumination``, ``exposure_ms / REFERENCE_EXPOSURE_MS`` and ``gain``;
    * ``shot_noise_scale`` is electrons per unit of normalised signal, so the mean
      photoelectron count is ``shot_noise_scale * mean_signal`` and Poisson shot
      noise adds signal variance ``mean_signal / shot_noise_scale``;
    * ``read_noise_e`` is a Gaussian read-noise standard deviation in electrons and
      becomes ``read_noise_e / shot_noise_scale`` in signal units; a sensor that
      declares ``read_noise_e > 0`` overrides this default and a noise event
      overrides both, with the used value and its source recorded per frame;
    * ``saturation_level`` is the normalised signal at which a sample clips;
    * ``response`` is the filter response: ``(R, G1, G2, B)`` for the CFA model and
      ``(R, G, B)`` for the co-sited three-channel model.

    ``bad_pixel_fraction``/``missing_sample_fraction`` declare a static defect
    pattern per camera (derived from the camera id and these fractions), which is
    what makes defects a property of the sensor instead of a per-frame accident.
    """

    kind: str
    shot_noise_scale: float = 1000.0
    read_noise_e: float = 2.0
    saturation_level: float = 1.0
    response: tuple[float, ...] = (0.95, 1.0, 1.0, 0.75)
    bad_pixel_fraction: float = 0.0
    missing_sample_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"unknown measurement model kind: {self.kind!r}")
        expected = 4 if self.kind == "cfa" else 3
        if len(tuple(self.response)) != expected:
            raise ValueError(f"{self.kind} response needs {expected} coefficients")
        if any(not 0.0 < float(item) <= 1.0 for item in self.response):
            raise ValueError("filter responses must lie in (0, 1]")
        if self.shot_noise_scale <= 0.0:
            raise ValueError("shot_noise_scale must be positive")
        if self.read_noise_e < 0.0:
            raise ValueError("read_noise_e must be non-negative")
        if self.saturation_level <= 0.0:
            raise ValueError("saturation_level must be positive")
        for name in ("bad_pixel_fraction", "missing_sample_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")

    @property
    def version(self) -> str:
        """Identifier of this declared model, part of every measurement hash."""
        payload = _jsonable(self)
        return f"{SYNTHETIC_VERSION}/{self.kind}/{_hash_payload(payload)[:12]}"

    @property
    def input_mode(self) -> str:
        return "bayer_cfa" if self.kind == "cfa" else "rawvla_rgb3"

    def unit(self) -> str:
        return "raw_dn_uint16_mosaic" if self.kind == "cfa" else "pseudo_raw_dn_float32"

    def storage_dtype(self) -> Any:
        return np.uint16 if self.kind == "cfa" else np.float32

    def defect_masks(self, sensor: SensorSpec) -> tuple[np.ndarray, np.ndarray]:
        """Static ``(missing, bad)`` pixel masks for one camera, in pixel space."""
        rng = _rng(
            "n02-full-defects",
            self.kind,
            sensor.camera_id,
            sensor.height,
            sensor.width,
            self.bad_pixel_fraction,
            self.missing_sample_fraction,
        )
        shape = (int(sensor.height), int(sensor.width))
        bad = rng.random(shape) < float(self.bad_pixel_fraction)
        missing = rng.random(shape) < float(self.missing_sample_fraction)
        return missing, bad

    def render(self, scene_state: RenderState) -> dict[str, SensorMeasurement]:
        """Render every declared sensor of ``scene_state`` for one frame."""
        if not isinstance(scene_state, RenderState):
            raise TypeError("render() expects a RenderState")
        return {
            sensor.camera_id: self._render_sensor(sensor, scene_state)
            for sensor in scene_state.sensors
        }

    def _expected_signal(
        self, sensor: SensorSpec, state: RenderState
    ) -> tuple[np.ndarray, np.ndarray]:
        """Noise-free normalised-signal mean and declared shot+read variance.

        This is the single projection/response/exposure/noise-parameter code path shared
        with :meth:`_render_sensor`; it draws **no** random numbers and never reads or
        advances ``state.noise_seed``, so calling it cannot change a seeded render.
        """
        if sensor.input_mode != self.input_mode:
            raise ValueError(
                f"model kind {self.kind!r} cannot render a {sensor.input_mode!r} sensor "
                f"({sensor.camera_id})"
            )
        height, width = int(sensor.height), int(sensor.width)
        shift = state.camera_shift_px.get(sensor.camera_id, (0.0, 0.0))
        albedo = _paint_albedo(
            state.shapes, height, width, shift, background=state.background_level
        )
        albedo = np.clip(albedo, 0.0, 1.0)
        exposure = (
            float(state.exposure_scale)
            * float(state.illumination)
            * float(sensor.exposure_ms)
            / REFERENCE_EXPOSURE_MS
            * float(sensor.gain)
        )
        read_noise_e = float(state.read_noise_e)

        if self.kind == "cfa":
            plane_index = cfa_pixel_plane_index(sensor.cfa_pattern, height, width)
            response = np.asarray(self.response, dtype=np.float32)
            channels = np.asarray(cfa_plane_colour_channels(), dtype=np.int64)
            channel_index = channels[plane_index]
            mean_signal = (
                exposure
                * response[plane_index]
                * albedo[channel_index, np.arange(height)[:, None], np.arange(width)[None, :]]
            )
        else:
            response = np.asarray(self.response, dtype=np.float32)[:, None, None]
            mean_signal = exposure * response * albedo

        mean_signal = np.clip(mean_signal.astype(np.float32), 0.0, None)
        variance = (
            mean_signal / float(self.shot_noise_scale)
            + (read_noise_e / float(self.shot_noise_scale)) ** 2
        )
        return mean_signal, variance

    def mean_variance(
        self,
        sensor: SensorSpec,
        state: RenderState,
        *,
        hypothesis: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Noise-free expectation of one camera's measurement, in the declared unit.

        Returns ``(mean, variance)`` as float32 arrays whose unit string is
        ``self.unit()`` - the same declared domain as the ``SensorMeasurement`` produced
        by :meth:`render` for this sensor (DN for both declared kinds):

        * ``mean = black_level + mean_signal * dn_scale``;
        * ``variance = (mean_signal / shot_noise_scale
          + (read_noise_e / shot_noise_scale) ** 2) * dn_scale ** 2``,

        with ``mean_signal`` produced by the exact projection/response/exposure path of
        :meth:`render`.  The value is the *linear* expectation: the ADC rounding and the
        saturation clip applied by :meth:`render` are measurement effects and are not part
        of it.  No random number is drawn here and no generator state is advanced, so the
        hook can be called any number of times before, between or after seeded renders
        without changing any rendered value.

        ``hypothesis`` overrides the **geometry** of declared entities for this
        evaluation only, using the same field names and semantics as the declared world
        geometry of the public candidate dictionary (``world_px``/``scale_px``/``depth``)::

            hypothesis = {"shape_a": {"world_px": (12.0, 20.0), "scale_px": 6.0}}

        ``world_px`` is the entity's declared world position - the same quantity the truth
        geometry carries - and is mapped to the renderer's ``RenderShape.position_px``;
        ``position_px`` is accepted as an alias.  Entities not named keep the frame
        state's geometry, and kind, colour, visibility, artefact flag, occlusion,
        exposure, illumination and noise parameters always come from ``state``.  Unknown
        entity ids or field names raise ``ValueError`` instead of being invented.
        ``hypothesis=None`` (and an empty mapping) use the unchanged path and are
        bit-identical to the plain call.
        """
        if not isinstance(sensor, SensorSpec):
            raise TypeError(f"mean_variance() expects a SensorSpec, got {type(sensor).__name__}")
        if not isinstance(state, RenderState):
            raise TypeError(f"mean_variance() expects a RenderState, got {type(state).__name__}")
        if hypothesis is not None:
            state = _state_with_geometry_hypothesis(state, hypothesis)
        mean_signal, variance = self._expected_signal(sensor, state)
        dn_scale = float(sensor.dn_scale)
        mean_dn = float(sensor.black_level) + mean_signal * dn_scale
        variance_dn = variance * dn_scale * dn_scale
        return np.asarray(mean_dn, dtype=np.float32), np.asarray(variance_dn, dtype=np.float32)

    def _render_sensor(self, sensor: SensorSpec, state: RenderState) -> SensorMeasurement:
        mean_signal, variance = self._expected_signal(sensor, state)
        shift = state.camera_shift_px.get(sensor.camera_id, (0.0, 0.0))
        shape: tuple[int, ...] = tuple(mean_signal.shape)
        read_noise_e = float(state.read_noise_e)
        rng = np.random.default_rng(state.noise_seed)
        electrons = rng.poisson(lam=mean_signal * float(self.shot_noise_scale)).astype(np.float32)
        if read_noise_e > 0.0:
            electrons = electrons + rng.normal(0.0, read_noise_e, size=shape).astype(np.float32)
        signal = electrons / float(self.shot_noise_scale)

        saturation_dn = float(sensor.black_level) + float(self.saturation_level) * sensor.dn_scale
        saturated = signal >= float(self.saturation_level)
        dn = float(sensor.black_level) + signal * sensor.dn_scale
        dn = np.clip(dn, 0.0, saturation_dn)

        missing, bad = self.defect_masks(sensor)
        if self.kind == "cfa":
            values = np.rint(dn).astype(np.uint16)
            missing_mask = missing
            bad_mask = bad
            values = np.where(missing | bad, np.uint16(0), values)
            tail = saturation_tail_probability(mean_signal, variance, float(self.saturation_level))
            plane_positions = cfa_plane_positions(sensor.cfa_pattern, (0, 0))
        else:
            values = dn.astype(np.float32)
            missing_mask = np.broadcast_to(missing, shape).copy()
            bad_mask = np.broadcast_to(bad, shape).copy()
            values = np.where(missing_mask | bad_mask, np.float32("nan"), values)
            tail = saturation_tail_probability(mean_signal, variance, float(self.saturation_level))
            plane_positions = None

        metadata = {
            "synthetic_version": SYNTHETIC_VERSION,
            "model_version": self.version,
            "model_kind": self.kind,
            "unit": self.unit(),
            "dtype": str(self.storage_dtype().__name__),
            "input_mode": sensor.input_mode,
            "black_level": float(sensor.black_level),
            "white_level": float(sensor.white_level),
            "bit_depth": int(sensor.bit_depth),
            "saturation_level": float(self.saturation_level),
            "saturation_dn": saturation_dn,
            "saturation_semantics": (
                "clipped at saturation_dn and flagged; the pre-clip value carries an "
                "interval/tail probability, it is not an exact observation"
            ),
            "shot_noise_scale": float(self.shot_noise_scale),
            "read_noise_e": float(read_noise_e),
            "read_noise_e_source": str(state.read_noise_e_source),
            "noise_expression": (
                "electrons = Poisson(shot_noise_scale * mean_signal) + Normal(0, read_noise_e); "
                "signal = electrons / shot_noise_scale"
            ),
            "unit_scale_bits": int(sensor.bit_depth),
            "quantisation": (
                "round_to_integer_dn" if self.kind == "cfa" else "none_float32_storage"
            ),
            "gain": float(sensor.gain),
            "exposure_ms": float(sensor.exposure_ms),
            "response": list(float(item) for item in self.response),
            "illumination": float(state.illumination),
            "exposure_scale": float(state.exposure_scale),
            "frame_index": int(state.frame_index),
            "camera_shift_px": [float(shift[0]), float(shift[1])],
            "pixel_centre_convention": PIXEL_CENTRE,
            "rasterisation": "pixel-centre indicator, no anti-aliasing",
            "crop_offset": [0, 0],
            "missing_value_semantics": (
                "stored 0 DN and excluded by valid_mask; never a measurement"
                if self.kind == "cfa"
                else "stored NaN; never a measurement"
            ),
            "bad_value_semantics": (
                "stored 0 DN and excluded by valid_mask; never a measurement"
                if self.kind == "cfa"
                else "stored NaN; never a measurement"
            ),
            "missing_count": int(missing_mask.sum()),
            "bad_count": int(bad_mask.sum()),
            "saturated_count": int(saturated.sum()),
            "contributing_count": int((~(missing_mask | bad_mask)).sum()),
            "truth_used_for_rendering": [
                "shape_ids",
                "geometry",
                "colour_response",
                "existence",
                "occlusion_steering",
            ],
        }
        return SensorMeasurement(
            camera_id=sensor.camera_id,
            input_mode=sensor.input_mode,
            values=values,
            unit=self.unit(),
            dtype_name=str(self.storage_dtype().__name__),
            valid_mask=~(missing_mask | bad_mask),
            bad_mask=bad_mask,
            missing_mask=missing_mask,
            saturated_mask=saturated,
            saturation_tail_probability=tail,
            frame_index=int(state.frame_index),
            crop_offset=(0, 0),
            cfa_pattern=sensor.cfa_pattern if self.kind == "cfa" else "",
            plane_positions=plane_positions,
            metadata=metadata,
        )


def _shape_mask(kind: str, scale: float, y0: float, x0: float, height: int, width: int) -> Any:
    """Hard pixel-centre footprint of one shape in image coordinates."""
    yy = (np.arange(height, dtype=np.float32) + 0.5)[:, None]
    xx = (np.arange(width, dtype=np.float32) + 0.5)[None, :]
    dy = yy - float(y0)
    dx = xx - float(x0)
    if kind == "disc":
        return dy * dy + dx * dx <= float(scale) ** 2
    if kind == "square":
        return (np.abs(dy) <= float(scale)) & (np.abs(dx) <= float(scale))
    if kind == "ring":
        radius2 = dy * dy + dx * dx
        inner = (0.55 * float(scale)) ** 2
        return (radius2 <= float(scale) ** 2) & (radius2 >= inner)
    if kind == "triangle":
        half = float(scale) * math.sin(math.pi / 3.0)
        vertices = np.asarray(
            [
                [float(y0) - float(scale), float(x0)],
                [float(y0) + float(scale) / 2.0, float(x0) - half],
                [float(y0) + float(scale) / 2.0, float(x0) + half],
            ],
            dtype=np.float32,
        )
        return _point_in_polygon(vertices, yy, xx)
    raise ValueError(f"unknown shape kind: {kind!r}")


def _point_in_polygon(vertices: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Convex polygon test via consistent edge-cross signs."""
    count = int(vertices.shape[0])
    area = 0.0
    for index in range(count):
        y_a, x_a = vertices[index]
        y_b, x_b = vertices[(index + 1) % count]
        area += float(y_a) * float(x_b) - float(y_b) * float(x_a)
    inside = np.ones(np.broadcast(yy, xx).shape, dtype=bool)
    orientation = 1.0 if area >= 0.0 else -1.0
    for index in range(count):
        y_a, x_a = vertices[index]
        y_b, x_b = vertices[(index + 1) % count]
        cross = (float(x_b) - float(x_a)) * (yy - float(y_a)) - (float(y_b) - float(y_a)) * (
            xx - float(x_a)
        )
        inside &= cross * orientation >= -1e-4
    return inside


def _paint_albedo(
    shapes: tuple[RenderShape, ...],
    height: int,
    width: int,
    shift: tuple[float, float],
    *,
    background: float | None = None,
) -> Any:
    """Paint the scene's per-pixel ``[3, H, W]`` reflectance in layer order."""
    level = 0.0 if background is None else float(background)
    albedo = np.full((3, height, width), level, dtype=np.float32)
    for shape in sorted(shapes, key=lambda item: int(item.depth)):
        if not shape.visible:
            continue
        mask = _shape_mask(
            shape.kind,
            float(shape.scale_px),
            float(shape.position_px[0]) - float(shift[0]),
            float(shape.position_px[1]) - float(shift[1]),
            height,
            width,
        )
        if not mask.any():
            continue
        colour = np.asarray(shape.colour_response, dtype=np.float32)[:, None, None]
        albedo = np.where(mask[None, :, :], colour, albedo)
    return albedo


def _render_layers(
    shapes: tuple[RenderShape, ...],
    height: int,
    width: int,
    shift: tuple[float, float],
) -> tuple[np.ndarray, tuple[str, ...], tuple[np.ndarray, ...]]:
    """Owner map and footprints for the occlusion truth of one camera.

    Only real, visible objects take part in the layering: a declared artefact such
    as a false-split ghost is painted but never counts as an occluder.
    """
    owner = np.full((height, width), -1, dtype=np.int64)
    ids: list[str] = []
    footprints: list[np.ndarray] = []
    for shape in sorted(shapes, key=lambda item: int(item.depth)):
        if shape.is_artefact or not shape.visible:
            continue
        mask = _shape_mask(
            shape.kind,
            float(shape.scale_px),
            float(shape.position_px[0]) - float(shift[0]),
            float(shape.position_px[1]) - float(shift[1]),
            height,
            width,
        )
        footprints.append(mask)
        ids.append(shape.shape_id)
        owner = np.where(mask, len(footprints) - 1, owner)
    return owner, tuple(ids), tuple(footprints)


def shape_footprint_radius(scale_px: float) -> float:
    """Declared bounding radius of every shape kind (used for relation labels)."""
    return float(scale_px)


def _normal_sf(z: Any) -> np.ndarray:
    """P(Z >= z) for a standard normal, Abramowitz-Stegun 26.2.17 (|err| < 1e-7).

    The closed form is used instead of a per-element ``math.erfc`` so a frame of
    tail probabilities stays a vector operation; the declared accuracy is checked
    against ``math.erfc`` in the tests.
    """
    values = np.asarray(z, dtype=np.float64)
    magnitude = np.abs(values)
    t = 1.0 / (1.0 + 0.2316419 * magnitude)
    poly = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    density = np.exp(-0.5 * magnitude * magnitude) / math.sqrt(2.0 * math.pi)
    upper = density * poly
    upper = np.clip(upper, 0.0, 1.0)
    return np.where(values >= 0.0, upper, 1.0 - upper)


def saturation_tail_probability(mean: Any, variance: Any, threshold: float) -> np.ndarray:
    """Declared P(noisy sample >= threshold) under the Gaussian approximation.

    With variance <= 0 the statement degenerates to the deterministic comparison.
    """
    mean_array = np.asarray(mean, dtype=np.float64)
    variance_array = np.asarray(variance, dtype=np.float64)
    sigma = np.sqrt(np.clip(variance_array, 0.0, None))
    safe = np.where(sigma > 0.0, sigma, 1.0)
    z = (float(threshold) - mean_array) / safe
    probability = _normal_sf(z)
    return np.where(sigma > 0.0, probability, (mean_array >= float(threshold)).astype(np.float64))


def quantisation_interval_probability(
    value: Any,
    mean: Any,
    variance: Any,
    *,
    step: float = 1.0,
) -> np.ndarray:
    """Declared P(value - step/2 <= noisy sample < value + step/2)."""
    if step <= 0.0:
        raise ValueError("step must be positive")
    half = 0.5 * float(step)
    centre = np.asarray(value, dtype=np.float64) - np.asarray(mean)
    sigma = _safe_sigma(variance)
    low = _normal_sf((centre - half) / sigma)
    high = _normal_sf((centre + half) / sigma)
    return np.clip(low - high, 0.0, 1.0)


def _safe_sigma(variance: Any) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.asarray(variance, dtype=np.float64), 0.0, None))
    return np.where(sigma > 0.0, sigma, 1.0)


# --------------------------------------------------------------------------------------
# event program
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExposureEvent:
    """Multiplies the sensor-side exposure scale from ``step`` onwards."""

    step: int
    scale: float


@dataclass(frozen=True, slots=True)
class NoiseEvent:
    """Replaces the shot/read noise declaration from ``step`` onwards."""

    step: int
    shot_noise_scale: float
    read_noise_e: float


@dataclass(frozen=True, slots=True)
class CameraShiftEvent:
    """Uncommanded camera displacement of 1..4 px, persistent from ``step`` on."""

    step: int
    camera_id: str
    dy: int
    dx: int

    def __post_init__(self) -> None:
        for name in ("dy", "dx"):
            value = int(getattr(self, name))
            if abs(value) > 4:
                raise ValueError(f"{name} must stay within +-4 px for this event class")


@dataclass(frozen=True, slots=True)
class MotionEvent:
    """Replaces one object's velocity from ``step`` onwards."""

    step: int
    shape_id: str
    velocity_px_per_step: tuple[float, float]


@dataclass(frozen=True, slots=True)
class OcclusionEvent:
    """Starts (``active``) or stops the scripted occlusion of one pair."""

    step: int
    occluder_id: str
    occluded_id: str
    active: bool


@dataclass(frozen=True, slots=True)
class ExistenceEvent:
    """An object enters (``exists``) or leaves the scene."""

    step: int
    shape_id: str
    exists: bool


@dataclass(frozen=True, slots=True)
class AppearanceEvent:
    """A declared measurement-only artefact: false merge or false split."""

    step: int
    shape_id: str
    partner_id: str
    offset_px: tuple[float, float] = (0.0, 0.0)
    duration_steps: int = 4

    def __post_init__(self) -> None:
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")

    @property
    def end_step(self) -> int:
        return int(self.step) + int(self.duration_steps)


@dataclass(frozen=True, slots=True)
class IdentitySwapEvent:
    """Two objects exchange their appearance (colour response), not their ids."""

    step: int
    shape_id_a: str
    shape_id_b: str


@dataclass(frozen=True, slots=True)
class RelationEvent:
    """A declared relation between two objects becomes active or inactive."""

    step: int
    subject_id: str
    predicate: str
    object_id: str
    active: bool


@dataclass(frozen=True, slots=True)
class EventSchedule:
    """Discrete event times and parameters for one sequence horizon.

    Every class draws its times from its own generator stream, so the classes are
    independent: changing one class's event count never moves another class.
    """

    horizon: int
    seed: int
    exposure: tuple[ExposureEvent, ...] = ()
    noise: tuple[NoiseEvent, ...] = ()
    camera_shift: tuple[CameraShiftEvent, ...] = ()
    object_motion: tuple[MotionEvent, ...] = ()
    occlusion: tuple[OcclusionEvent, ...] = ()
    existence: tuple[ExistenceEvent, ...] = ()
    false_merge: tuple[AppearanceEvent, ...] = ()
    false_split: tuple[AppearanceEvent, ...] = ()
    identity_swap: tuple[IdentitySwapEvent, ...] = ()
    relation: tuple[RelationEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        for name in EVENT_TYPES:
            for event in getattr(self, EVENT_FIELDS[name]):
                if not 0 <= int(event.step) < int(self.horizon):
                    raise ValueError(
                        f"{name} event at step {event.step} is outside the horizon {self.horizon}"
                    )

    def __getitem__(self, name: str) -> tuple[Any, ...]:
        if name not in EVENT_FIELDS:
            raise KeyError(name)
        return tuple(getattr(self, EVENT_FIELDS[name]))

    def event_types(self) -> tuple[str, ...]:
        """Class names that carry at least one event, in declared order."""
        return tuple(name for name in EVENT_TYPES if len(self[name]) > 0)

    def counts(self) -> dict[str, int]:
        return {name: len(self[name]) for name in EVENT_TYPES}

    def all_events(self) -> tuple[Any, ...]:
        """Every event ordered by ``(step, class)``."""
        events: list[tuple[int, int, Any]] = []
        for order, name in enumerate(EVENT_TYPES):
            for event in self[name]:
                events.append((int(event.step), order, event))
        events.sort(key=lambda item: (item[0], item[1]))
        return tuple(event for _, _, event in events)

    def steps(self, name: str) -> tuple[int, ...]:
        return tuple(int(event.step) for event in self[name])

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def schedule_hash(self) -> str:
        return schedule_hash(self)


def schedule_hash(schedule: EventSchedule) -> str:
    """Deterministic hash of the realised event program."""
    return _hash_payload({"version": SYNTHETIC_VERSION, "schedule": _jsonable(schedule)})


def build_event_schedule(
    scene: SceneSpec,
    *,
    frames: int,
    seed: int,
    exposure_events: int = 2,
    noise_events: int = 2,
    camera_shift_events: int = 2,
    motion_events: int = 2,
    occlusion_events: int = 2,
    existence_events: int = 2,
    false_merge_events: int = 1,
    false_split_events: int = 1,
    identity_swap_events: int = 1,
    relation_events: int = 2,
    camera_id: str = "",
) -> EventSchedule:
    """Build the event program of one scene with independent per-class times.

    ``*_events`` are per-class counts; each class samples distinct steps in
    ``[1, frames - 1]`` from its own stream. Object-referencing classes pick their
    objects from the scene's own id list, so the schedule is always consistent
    with the scene it belongs to.
    """
    if frames < 2:
        raise ValueError("a sequence needs at least two frames")
    horizon = int(frames)
    shape_ids = list(scene.shape_ids)
    camera = camera_id or "camera-0"

    def _steps(count: int, name: str) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        upper = horizon - 1
        stream = _rng("n02-full-event", name, seed)
        levels = np.arange(1, upper + 1)
        size = min(int(count), int(levels.size))
        drawn = stream.choice(levels, size=size, replace=False)
        return np.sort(drawn.astype(np.int64))

    exposure: list[ExposureEvent] = []
    scale_stream = _rng("n02-full-event-params", "exposure", seed)
    for step in _steps(exposure_events, "exposure_change"):
        exposure.append(
            ExposureEvent(step=int(step), scale=float(scale_stream.choice([0.7, 0.85, 1.2])))
        )

    noise: list[NoiseEvent] = []
    noise_stream = _rng("n02-full-event-params", "noise", seed)
    for step in _steps(noise_events, "noise_change"):
        noise.append(
            NoiseEvent(
                step=int(step),
                shot_noise_scale=float(noise_stream.choice([200.0, 600.0, 2000.0])),
                read_noise_e=float(noise_stream.choice([0.5, 2.0, 6.0])),
            )
        )

    shift: list[CameraShiftEvent] = []
    shift_stream = _rng("n02-full-event-params", "camera_shift", seed)
    for step in _steps(camera_shift_events, "camera_shift"):
        magnitude = int(shift_stream.integers(1, 5))
        sign_y = int(shift_stream.choice([-1, 1]))
        sign_x = int(shift_stream.choice([-1, 1]))
        dy = magnitude * sign_y if shift_stream.random() < 0.7 else 0
        dx = 0 if dy != 0 else magnitude * sign_x
        shift.append(CameraShiftEvent(step=int(step), camera_id=camera, dy=int(dy), dx=int(dx)))

    motion: list[MotionEvent] = []
    motion_stream = _rng("n02-full-event-params", "object_motion", seed)
    for index, step in enumerate(_steps(motion_events, "object_motion")):
        shape_id = shape_ids[index % len(shape_ids)]
        speed = float(motion_stream.uniform(0.2, 1.2))
        angle = float(motion_stream.uniform(0.0, 2.0 * math.pi))
        motion.append(
            MotionEvent(
                step=int(step),
                shape_id=shape_id,
                velocity_px_per_step=(speed * math.sin(angle), speed * math.cos(angle)),
            )
        )

    occlusion: list[OcclusionEvent] = []
    occlusion_stream = _rng("n02-full-event-params", "occlusion", seed)
    for index, step in enumerate(_steps(occlusion_events, "occlusion")):
        if len(shape_ids) < 2:
            break
        first, second = shape_ids[index % len(shape_ids)], shape_ids[(index + 1) % len(shape_ids)]
        occluder, occluded = (first, second) if index % 2 == 0 else (second, first)
        duration = int(occlusion_stream.integers(2, max(3, min(8, horizon - 1))))
        occlusion.append(
            OcclusionEvent(step=int(step), occluder_id=occluder, occluded_id=occluded, active=True)
        )
        stop = min(horizon - 1, int(step) + duration)
        if stop > int(step):
            occlusion.append(
                OcclusionEvent(
                    step=int(stop), occluder_id=occluder, occluded_id=occluded, active=False
                )
            )

    existence: list[ExistenceEvent] = []
    existence_stream = _rng("n02-full-event-params", "enter_leave", seed)
    for index, step in enumerate(_steps(existence_events, "enter_leave")):
        shape_id = shape_ids[index % len(shape_ids)]
        appears = bool(existence_stream.random() < 0.5)
        existence.append(ExistenceEvent(step=int(step), shape_id=shape_id, exists=bool(appears)))

    merges: list[AppearanceEvent] = []
    merge_stream = _rng("n02-full-event-params", "false_merge", seed)
    for index, step in enumerate(_steps(false_merge_events, "false_merge")):
        if len(shape_ids) < 2:
            break
        merges.append(
            AppearanceEvent(
                step=int(step),
                shape_id=shape_ids[index % len(shape_ids)],
                partner_id=shape_ids[(index + 1) % len(shape_ids)],
                offset_px=(0.0, 0.0),
                duration_steps=int(merge_stream.integers(2, max(3, min(8, horizon - 1)))),
            )
        )

    splits: list[AppearanceEvent] = []
    split_stream = _rng("n02-full-event-params", "false_split", seed)
    for index, step in enumerate(_steps(false_split_events, "false_split")):
        offset = float(split_stream.uniform(1.5, 3.5))
        splits.append(
            AppearanceEvent(
                step=int(step),
                shape_id=shape_ids[index % len(shape_ids)],
                partner_id="",
                offset_px=(offset, offset),
                duration_steps=int(split_stream.integers(2, max(3, min(8, horizon - 1)))),
            )
        )

    swaps: list[IdentitySwapEvent] = []
    swap_stream = _rng("n02-full-event-params", "identity_swap", seed)
    for step in _steps(identity_swap_events, "identity_swap"):
        if len(shape_ids) < 2:
            break
        picks = swap_stream.choice(len(shape_ids), size=2, replace=False)
        swaps.append(
            IdentitySwapEvent(
                step=int(step),
                shape_id_a=shape_ids[int(picks[0])],
                shape_id_b=shape_ids[int(picks[1])],
            )
        )

    relations: list[RelationEvent] = []
    relation_stream = _rng("n02-full-event-params", "relation_change", seed)
    predicates = ("left_of", "above", "approaching")
    for index, step in enumerate(_steps(relation_events, "relation_change")):
        if len(shape_ids) < 2:
            break
        subject = shape_ids[index % len(shape_ids)]
        partner = shape_ids[(index + 1) % len(shape_ids)]
        predicate = predicates[index % len(predicates)]
        relations.append(
            RelationEvent(
                step=int(step),
                subject_id=subject,
                predicate=predicate,
                object_id=partner,
                active=True,
            )
        )
        duration = int(relation_stream.integers(2, max(3, min(8, horizon - 1))))
        stop = min(horizon - 1, int(step) + duration)
        if stop > int(step):
            relations.append(
                RelationEvent(
                    step=int(stop),
                    subject_id=subject,
                    predicate=predicate,
                    object_id=partner,
                    active=False,
                )
            )

    return EventSchedule(
        horizon=horizon,
        seed=int(seed),
        exposure=tuple(exposure),
        noise=tuple(noise),
        camera_shift=tuple(shift),
        object_motion=tuple(motion),
        occlusion=tuple(occlusion),
        existence=tuple(existence),
        false_merge=tuple(merges),
        false_split=tuple(splits),
        identity_swap=tuple(swaps),
        relation=tuple(relations),
    )


# --------------------------------------------------------------------------------------
# scene generation with partition vocabularies
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartitionVocabulary:
    """Declared design space of one partition family.

    Disjointness of ``train`` and the sealed ``test`` family is checked on all
    four axes: shape kinds, object counts, occlusion modes and interaction
    structures.
    """

    name: str
    shape_kinds: tuple[str, ...]
    shape_combinations: tuple[tuple[str, ...], ...]
    object_counts: tuple[int, ...]
    occlusion_modes: tuple[str, ...]
    interaction_structures: tuple[str, ...]
    order_margin_px: float = 8.0


_VOCAB_A = PartitionVocabulary(
    name="family-A",
    shape_kinds=("disc", "square"),
    shape_combinations=(("disc", "disc"), ("disc", "square"), ("square", "disc", "square")),
    object_counts=(2, 3),
    occlusion_modes=("none", "single_sweep"),
    interaction_structures=("approach_pair", "no_interaction"),
)

_VOCAB_B = PartitionVocabulary(
    name="family-B",
    shape_kinds=("triangle", "ring"),
    shape_combinations=(
        ("triangle", "ring", "triangle", "ring"),
        ("ring", "ring", "triangle", "triangle", "ring"),
    ),
    object_counts=(4, 5),
    occlusion_modes=("chain", "mutual"),
    interaction_structures=("cycle_three", "swap_pair"),
)

#: Partition -> vocabulary. ``dev``/``calibration`` share the train family's
#: design space (they are selection splits), while the sealed ``test`` family is
#: disjoint in every declared axis.
VOCABULARIES: dict[str, PartitionVocabulary] = {
    "train": _VOCAB_A,
    "dev": _VOCAB_A,
    "calibration": _VOCAB_A,
    "test": _VOCAB_B,
}


def generate_scene(
    partition: str,
    scene_id: str,
    *,
    seed: int | None = None,
    canvas: tuple[int, int] = (64, 64),
    horizon: int = 32,
) -> SceneSpec:
    """Build one reproducible scene from its partition vocabulary.

    The scene seed is derived from ``partition`` + ``scene_id`` (or from that pair
    plus an explicit ``seed``), so it can never collide with the model seeds.
    """
    if partition not in VOCABULARIES:
        raise ValueError(f"unknown partition: {partition!r}")
    vocabulary = VOCABULARIES[partition]
    base = (
        scene_seed(partition, scene_id)
        if seed is None
        else derive_seed("n02-full-scene-explicit", partition, scene_id, int(seed))
    )
    rng = np.random.default_rng(base)
    height, width = int(canvas[0]), int(canvas[1])
    margin = float(vocabulary.order_margin_px)

    combination = vocabulary.shape_combinations[
        int(rng.integers(0, len(vocabulary.shape_combinations)))
    ]
    kinds = list(combination)
    count_choices = vocabulary.object_counts
    target_count = int(count_choices[int(rng.integers(0, len(count_choices)))])
    while len(kinds) < target_count:
        kinds.append(vocabulary.shape_kinds[int(rng.integers(0, len(vocabulary.shape_kinds)))])

    shapes: list[ShapeSpec] = []
    short_side = float(min(height, width))
    for index, kind in enumerate(kinds):
        scale = float(rng.uniform(0.10, 0.20)) * short_side
        low_y, high_y = margin + scale, height - margin - scale
        low_x, high_x = margin + scale, width - margin - scale
        y = float(rng.uniform(low_y, high_y)) if high_y > low_y else height / 2.0
        x = float(rng.uniform(low_x, high_x)) if high_x > low_x else width / 2.0
        speed = float(rng.uniform(0.1, 0.8))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        shapes.append(
            ShapeSpec(
                shape_id=f"{scene_id}-o{index}",
                kind=kind,
                scale_px=scale,
                position_px=(y, x),
                velocity_px_per_step=(speed * math.sin(angle), speed * math.cos(angle)),
                colour_response=(
                    float(rng.uniform(0.15, 1.0)),
                    float(rng.uniform(0.15, 1.0)),
                    float(rng.uniform(0.15, 1.0)),
                ),
            )
        )

    occlusion_plan: list[OcclusionPlan] = []
    mode = vocabulary.occlusion_modes[int(rng.integers(0, len(vocabulary.occlusion_modes)))]
    if mode != "none" and len(shapes) >= 2 and horizon >= 5:
        pairs = 1 if mode == "single_sweep" else min(2, len(shapes) - 1)
        for index in range(pairs):
            start = int(rng.integers(1, max(2, horizon // 2)))
            duration = int(rng.integers(3, max(4, min(9, horizon - start))))
            end = min(horizon - 1, start + duration)
            if end <= start:
                continue
            occlusion_plan.append(
                OcclusionPlan(
                    occluder_id=shapes[index].shape_id,
                    occluded_id=shapes[(index + 1) % len(shapes)].shape_id,
                    start_step=int(start),
                    end_step=int(end),
                    mode="sweep_x" if index % 2 == 0 else "sweep_y",
                )
            )

    interaction_plan: list[InteractionPlan] = []
    structure = vocabulary.interaction_structures[
        int(rng.integers(0, len(vocabulary.interaction_structures)))
    ]
    if structure == "no_interaction":
        interaction_plan = []
    elif structure in {"approach_pair", "swap_pair"}:
        interaction_plan.append(
            InteractionPlan(
                predicate="approaching",
                subject_id=shapes[0].shape_id,
                object_id=shapes[1].shape_id,
                start_step=1,
                end_step=max(2, horizon - 1),
            )
        )
    else:  # cycle_three / chain
        for index in range(min(3, len(shapes))):
            interaction_plan.append(
                InteractionPlan(
                    predicate="linked",
                    subject_id=shapes[index].shape_id,
                    object_id=shapes[(index + 1) % len(shapes)].shape_id,
                    start_step=1,
                    end_step=max(2, horizon - 1),
                )
            )

    lighting_steps: tuple[tuple[int, float], ...] = ()
    if horizon >= 8:
        lighting_steps = (
            (int(rng.integers(3, max(4, horizon - 3))), float(rng.uniform(0.6, 1.3))),
        )

    return SceneSpec(
        scene_id=str(scene_id),
        seed=int(base),
        partition=partition,
        shapes=tuple(shapes),
        background_level=float(rng.uniform(0.05, 0.6)),
        camera_motion_px=(float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3))),
        occlusion_plan=tuple(occlusion_plan),
        lighting_plan=LightingPlan(base_illumination=1.0, steps=lighting_steps),
        interaction_plan=tuple(interaction_plan),
    )


def scene_hash(scene: SceneSpec) -> str:
    """Deterministic hash of a scene program (layout and declared plans)."""
    return _hash_payload({"version": SYNTHETIC_VERSION, "scene": scene.as_dict()})


# --------------------------------------------------------------------------------------
# evaluation truth (never enters an inference payload)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationTruth:
    """Evaluator-side truth for one frame; never a policy input.

    ``geometry`` holds ``world_px`` ``[N, 2]`` plus, per camera,
    ``image_px`` ``[N, 2]``, ``scale_px`` ``[N]`` and ``depth`` ``[N]``.
    ``occlusion_mask`` and ``visible_fraction`` are per camera; ``relations`` are
    ``(subject, predicate, object)`` triples; ``identity_links`` pair an appearance
    token with the object currently carrying it (an identity swap changes the link
    while the object ids and geometry stay); ``event_labels`` name the effects in
    force at this frame.
    """

    object_ids: tuple[str, ...]
    geometry: dict[str, np.ndarray]
    existence: np.ndarray
    occlusion_mask: dict[str, np.ndarray]
    relations: tuple[tuple[str, str, str], ...]
    identity_links: tuple[tuple[str, str], ...]
    event_labels: tuple[str, ...]
    visible_fraction: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "object_ids": list(self.object_ids),
            "existence": [bool(item) for item in np.asarray(self.existence)],
            "relations": [list(item) for item in self.relations],
            "identity_links": [list(item) for item in self.identity_links],
            "event_labels": list(self.event_labels),
        }


# --------------------------------------------------------------------------------------
# sequence generation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticSequence:
    """One generated sequence: measurements, inference inputs, targets, truth.

    ``frames`` keeps the rich declared measurements and ``inference`` the
    shared-contract payloads built from them; the two share the pixel arrays but
    nothing from ``truth``/``targets``.  ``render_states`` keeps, per frame, the exact
    :class:`RenderState` object that was rendered (no reconstruction), so a consumer can
    ask a measurement model for the noise-free expectation of that frame or derive a
    truth-geometry hypothesis.  It defaults to empty so sequences built before this field
    existed keep constructing unchanged.
    """

    sequence_id: str
    scene_id: str
    seed: int
    partition: str
    cameras: tuple[SensorSpec, ...]
    frames: tuple[dict[str, SensorMeasurement], ...]
    inference: tuple[InferenceInput, ...]
    targets: tuple[TrainingTargets, ...]
    truth: tuple[EvaluationTruth, ...]
    manifest: dict
    render_states: tuple[RenderState, ...] = ()

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sensor.camera_id for sensor in self.cameras)


def measurement_hash(sequence: SyntheticSequence) -> str:
    """Deterministic hash of every declared measurement of a sequence."""
    digest = hashlib.sha256()
    digest.update(SYNTHETIC_VERSION.encode("ascii"))
    digest.update(str(sequence.seed).encode("ascii"))
    for frame_index, frame in enumerate(sequence.frames):
        digest.update(f"frame={frame_index}".encode("ascii"))
        for camera_id in sorted(frame):
            measurement = frame[camera_id]
            digest.update(f"camera={camera_id}".encode("ascii"))
            digest.update(str(measurement.input_mode).encode("ascii"))
            digest.update(str(measurement.unit).encode("ascii"))
            digest.update(str(measurement.cfa_pattern).encode("ascii"))
            digest.update(str(tuple(measurement.crop_offset)).encode("ascii"))
            _hash_array(digest, measurement.values)
            for mask in (
                measurement.valid_mask,
                measurement.bad_mask,
                measurement.missing_mask,
                measurement.saturated_mask,
            ):
                _hash_array(digest, np.asarray(mask, dtype=bool))
            digest.update(
                str(
                    [
                        measurement.metadata["black_level"],
                        measurement.metadata["white_level"],
                        measurement.metadata["bit_depth"],
                        measurement.metadata["saturation_level"],
                        measurement.metadata["shot_noise_scale"],
                        measurement.metadata["read_noise_e"],
                        measurement.metadata["exposure_ms"],
                        measurement.metadata["gain"],
                        measurement.metadata["camera_shift_px"],
                        measurement.metadata["illumination"],
                    ]
                ).encode("ascii")
            )
    return digest.hexdigest()


@dataclass(slots=True)
class _ShapeTrack:
    shape_id: str
    kind: str
    scale_px: float
    position: np.ndarray
    velocity: np.ndarray
    colour: np.ndarray
    appearance_id: str
    depth: int
    exists: bool = True


def _apply_events(
    tracks: dict[str, _ShapeTrack],
    schedule: EventSchedule,
    step: int,
    state: dict[str, Any],
) -> None:
    """Apply every event scheduled exactly at ``step`` to the running state."""
    for event in schedule.exposure:
        if event.step == step:
            state["exposure_scale"] = float(event.scale)
            state["labels"].append(f"exposure_change@{step}")
    for event in schedule.noise:
        if event.step == step:
            state["shot_noise_scale"] = float(event.shot_noise_scale)
            state["read_noise_e"] = float(event.read_noise_e)
            state["read_noise_e_source"] = "noise_event"
            state["noise_event_active"] = True
            state["labels"].append(f"noise_change@{step}")
    for event in schedule.camera_shift:
        if event.step == step:
            current = state["event_shift"][event.camera_id]
            state["event_shift"][event.camera_id] = (
                current[0] + float(event.dy),
                current[1] + float(event.dx),
            )
            state["labels"].append(f"camera_shift@{step}:{event.dy},{event.dx}")
    for event in schedule.object_motion:
        if event.step == step and event.shape_id in tracks:
            tracks[event.shape_id].velocity = np.asarray(
                event.velocity_px_per_step, dtype=np.float64
            )
            state["labels"].append(f"object_motion@{step}:{event.shape_id}")
    for event in schedule.occlusion:
        if event.step == step:
            state["occlusion"][(event.occluder_id, event.occluded_id)] = bool(event.active)
            state["labels"].append(
                f"occlusion@{step}:{event.occluder_id}>{event.occluded_id}:"
                f"{'on' if event.active else 'off'}"
            )
    for event in schedule.existence:
        if event.step == step and event.shape_id in tracks:
            tracks[event.shape_id].exists = bool(event.exists)
            state["labels"].append(
                f"{'enter' if event.exists else 'leave'}@{step}:{event.shape_id}"
            )
    for event in schedule.false_merge:
        if event.step == step:
            state["labels"].append(f"false_merge@{step}:{event.shape_id}+{event.partner_id}")
    for event in schedule.false_split:
        if event.step == step:
            state["labels"].append(f"false_split@{step}:{event.shape_id}")
    for event in schedule.identity_swap:
        if event.step == step and event.shape_id_a in tracks and event.shape_id_b in tracks:
            first, second = tracks[event.shape_id_a], tracks[event.shape_id_b]
            first.colour, second.colour = second.colour.copy(), first.colour.copy()
            first.appearance_id, second.appearance_id = (
                second.appearance_id,
                first.appearance_id,
            )
            state["labels"].append(f"identity_swap@{step}:{event.shape_id_a}<->{event.shape_id_b}")
    for event in schedule.relation:
        if event.step == step:
            state["relations"][(event.subject_id, event.predicate, event.object_id)] = bool(
                event.active
            )
            state["labels"].append(
                f"relation_change@{step}:{event.predicate}:{event.subject_id}>"
                f"{event.object_id}:{'on' if event.active else 'off'}"
            )


def _steer(
    tracks: dict[str, _ShapeTrack],
    scene: SceneSpec,
    step: int,
    state: dict[str, Any],
) -> None:
    """Apply scripted occlusion sweeps so the declared occlusion really happens."""
    for plan in scene.occlusion_plan:
        active = state["occlusion"].get((plan.occluder_id, plan.occluded_id), False)
        if not active:
            continue
        occluder = tracks.get(plan.occluder_id)
        occluded = tracks.get(plan.occluded_id)
        if occluder is None or occluded is None:
            continue
        span = max(1, int(plan.end_step) - int(plan.start_step))
        phase = (step - int(plan.start_step)) / float(span)
        if not 0.0 <= phase <= 1.0:
            continue
        sweep = (2.0 * phase - 1.0) * 2.0 * float(occluded.scale_px)
        if plan.mode == "sweep_x":
            occluder.position = occluded.position + np.asarray([0.0, sweep])
        else:
            occluder.position = occluded.position + np.asarray([sweep, 0.0])
        state["labels"].append(f"occlusion_sweep@{step}:{plan.occluder_id}")


def _frame_shapes(
    tracks: dict[str, _ShapeTrack],
    schedule: EventSchedule,
    state: dict[str, Any],
    step: int,
) -> tuple[RenderShape, ...]:
    """Render instructions for one frame: real objects plus declared artefacts.

    A false merge places the partner at the primary object's image position for the
    declared window, and a false split adds a dim ghost copy; both are declared
    measurement-only artefacts and never change the truth geometry.
    """
    shapes: list[RenderShape] = []
    merge_partner_of: dict[str, str] = {}
    split_of: dict[str, AppearanceEvent] = {}
    for merge in schedule.false_merge:
        if merge.step <= step < merge.end_step and merge.partner_id:
            merge_partner_of[merge.partner_id] = merge.shape_id
    for split in schedule.false_split:
        if split.step <= step < split.end_step:
            split_of[split.shape_id] = split
    for shape_id in state["order"]:
        track = tracks[shape_id]
        position = track.position
        primary = merge_partner_of.get(shape_id)
        if primary is not None and primary in tracks:
            position = tracks[primary].position
        shapes.append(
            RenderShape(
                shape_id=shape_id,
                kind=track.kind,
                position_px=(float(position[0]), float(position[1])),
                scale_px=float(track.scale_px),
                colour_response=(
                    float(track.colour[0]),
                    float(track.colour[1]),
                    float(track.colour[2]),
                ),
                depth=int(track.depth),
                visible=bool(track.exists),
            )
        )
        split = split_of.get(shape_id)
        if split is not None:
            shapes.append(
                RenderShape(
                    shape_id=f"{shape_id}-ghost",
                    kind=track.kind,
                    position_px=(
                        float(position[0]) + float(split.offset_px[0]),
                        float(position[1]) + float(split.offset_px[1]),
                    ),
                    scale_px=float(track.scale_px),
                    colour_response=(
                        float(track.colour[0]) * 0.5,
                        float(track.colour[1]) * 0.5,
                        float(track.colour[2]) * 0.5,
                    ),
                    depth=int(track.depth) * 2 - 1,
                    visible=True,
                    is_artefact=True,
                )
            )
    return tuple(shapes)


def _camera_shifts(
    state: dict[str, Any], sensors: tuple[SensorSpec, ...], step: int
) -> dict[str, tuple[float, float]]:
    commanded = state["commanded_pan"]
    base = state["base_motion"]
    shifts: dict[str, tuple[float, float]] = {}
    for sensor in sensors:
        event_shift = state["event_shift"].get(sensor.camera_id, (0.0, 0.0))
        shifts[sensor.camera_id] = (
            float(commanded[0]) + float(event_shift[0]) + float(base[0]) * step,
            float(commanded[1]) + float(event_shift[1]) + float(base[1]) * step,
        )
    return shifts


def _relations_for_frame(
    tracks: dict[str, _ShapeTrack], state: dict[str, Any], step: int
) -> tuple[tuple[str, str, str], ...]:
    triples: set[tuple[str, str, str]] = set()
    ids = list(state["order"])
    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            one, two = tracks[first], tracks[second]
            if not (one.exists and two.exists):
                continue
            if one.position[1] < two.position[1]:
                triples.add((first, "left_of", second))
            else:
                triples.add((second, "left_of", first))
            if one.position[0] < two.position[0]:
                triples.add((first, "above", second))
            else:
                triples.add((second, "above", first))
            distance = float(np.linalg.norm(one.position - two.position))
            if distance < shape_footprint_radius(one.scale_px) + shape_footprint_radius(
                two.scale_px
            ):
                triples.add((first, "overlaps", second))
    for (subject, predicate, obj), active in state["relations"].items():
        if active:
            triples.add((subject, predicate, obj))
    return tuple(sorted(triples))


def generate_sequence(
    scene: SceneSpec,
    sensors: tuple[SensorSpec, ...],
    measurement_model: MeasurementModel,
    events: EventSchedule,
    *,
    frames: int,
    seed: int,
    actions: Any = None,
    action_horizon: int = 10,
    view_kind: str = "independent",
    episode_id: str = "",
) -> SyntheticSequence:
    """Generate one sequence: measurements, inference inputs, targets and truth.

    ``actions`` overrides the declared action policy (used by causal tests); the
    action stream is drawn from its own generator, so an override cannot move any
    other random stream. The executed action at frame ``t`` is ``a[t - 1]`` and it
    moves the camera between ``t - 1`` and ``t``; the future chunk ``a[t:]`` is
    supervision only.
    """
    if not sensors:
        raise ValueError("a sequence needs at least one sensor")
    camera_ids = [sensor.camera_id for sensor in sensors]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("camera ids must be unique")
    for sensor in sensors:
        if sensor.input_mode != measurement_model.input_mode:
            raise ValueError(
                f"model kind {measurement_model.kind!r} cannot render sensor "
                f"{sensor.camera_id!r} ({sensor.input_mode!r})"
            )
    if int(frames) < 2:
        raise ValueError("a sequence needs at least two frames")
    if int(frames) > int(events.horizon):
        raise ValueError(f"requested {frames} frames but the schedule horizon is {events.horizon}")
    for name in EVENT_TYPES:
        for event in events[name]:
            for attribute in ("shape_id", "occluder_id", "occluded_id", "subject_id", "object_id"):
                value = getattr(event, attribute, None)
                if value is not None and value not in scene.shape_ids:
                    raise ValueError(f"{name} event refers to unknown shape {value!r}")
    for event in events.camera_shift:
        if event.camera_id not in camera_ids:
            raise ValueError(
                f"camera_shift event refers to unknown camera {event.camera_id!r}; "
                f"declared cameras are {camera_ids}"
            )

    episode = episode_id or f"{scene.scene_id}:{seed}"
    policy = _rng("n02-full-actions", seed, scene.scene_id)
    if actions is None:
        # Declared synthetic action space: commanded camera pan in px per step.
        action_array = np.stack(
            [
                policy.integers(-2, 3, size=int(frames) + int(action_horizon)).astype(np.int8),
                policy.integers(-2, 3, size=int(frames) + int(action_horizon)).astype(np.int8),
            ],
            axis=1,
        )
    else:
        action_array = np.asarray(actions, dtype=np.int8)
        if action_array.ndim != 2 or action_array.shape[1] != 2:
            raise ValueError("actions must have shape [T, 2] in the synthetic pan space")
        if action_array.shape[0] < int(frames):
            raise ValueError("an action override needs at least one action per frame")

    tracks: dict[str, _ShapeTrack] = {}
    for index, shape in enumerate(scene.shapes):
        tracks[shape.shape_id] = _ShapeTrack(
            shape_id=shape.shape_id,
            kind=shape.kind,
            scale_px=float(shape.scale_px),
            position=np.asarray(shape.position_px, dtype=np.float64),
            velocity=np.asarray(shape.velocity_px_per_step, dtype=np.float64),
            colour=np.asarray(shape.colour_response, dtype=np.float64),
            appearance_id=shape.shape_id,
            depth=index,
        )
    state: dict[str, Any] = {
        "order": list(scene.shape_ids),
        "exposure_scale": 1.0,
        "shot_noise_scale": float(measurement_model.shot_noise_scale),
        "read_noise_e": float(measurement_model.read_noise_e),
        "event_shift": {camera_id: (0.0, 0.0) for camera_id in camera_ids},
        "commanded_pan": np.zeros(2, dtype=np.float64),
        "base_motion": np.asarray(scene.camera_motion_px, dtype=np.float64),
        "occlusion": {},
        "relations": {},
        "labels": [],
        "noise_event_active": False,
    }

    declared_read_noise = {
        float(sensor.read_noise_e) for sensor in sensors if float(sensor.read_noise_e) > 0.0
    }
    if len(declared_read_noise) == 1:
        initial_read_noise = float(next(iter(declared_read_noise)))
        initial_read_noise_source = "sensor"
    else:
        initial_read_noise = float(measurement_model.read_noise_e)
        initial_read_noise_source = "model_default"
    state["read_noise_e"] = initial_read_noise
    state["read_noise_e_source"] = initial_read_noise_source

    measurements: list[dict[str, SensorMeasurement]] = []
    inference: list[InferenceInput] = []
    targets: list[TrainingTargets] = []
    truth: list[EvaluationTruth] = []
    render_states: list[RenderState] = []

    for step in range(int(frames)):
        state["labels"] = []
        _apply_events(tracks, events, step, state)
        if step > 0:
            for track in tracks.values():
                track.position = track.position + track.velocity
            state["commanded_pan"] = state["commanded_pan"] + np.asarray(
                action_array[step - 1], dtype=np.float64
            )
        _steer(tracks, scene, step, state)

        illumination = scene.lighting_plan.scale_at(step)
        shifts = _camera_shifts(state, sensors, step)
        render_state = RenderState(
            frame_index=step,
            shapes=_frame_shapes(tracks, events, state, step),
            sensors=tuple(sensors),
            background_level=float(scene.background_level),
            illumination=float(illumination),
            exposure_scale=float(state["exposure_scale"]),
            shot_noise_scale=float(state["shot_noise_scale"]),
            read_noise_e=float(state["read_noise_e"]),
            read_noise_e_source=str(state["read_noise_e_source"]),
            camera_shift_px=shifts,
            noise_seed=derive_seed("n02-full-noise", seed, scene.scene_id, step),
        )
        model = measurement_model
        if state["shot_noise_scale"] != float(measurement_model.shot_noise_scale) or float(
            state["read_noise_e"]
        ) != float(measurement_model.read_noise_e):
            model = dataclasses.replace(
                measurement_model,
                shot_noise_scale=float(state["shot_noise_scale"]),
                read_noise_e=float(state["read_noise_e"]),
            )
        frame_measurements = model.render(render_state)
        measurements.append(frame_measurements)
        # keep the very state that was rendered (never a reconstruction) so the
        # noise-free expectation and truth-geometry hypotheses can be recovered
        render_states.append(render_state)

        world_px = np.asarray(
            [tracks[shape_id].position for shape_id in scene.shape_ids], dtype=np.float64
        )
        existence = np.asarray(
            [bool(tracks[shape_id].exists) for shape_id in scene.shape_ids], dtype=bool
        )
        geometry: dict[str, np.ndarray] = {"world_px": world_px.copy()}
        occlusion_mask: dict[str, np.ndarray] = {}
        visible_fraction: dict[str, np.ndarray] = {}
        for sensor in sensors:
            image_px = world_px - np.asarray(shifts[sensor.camera_id], dtype=np.float64)
            owner, layer_ids, footprints = _render_layers(
                render_state.shapes,
                int(sensor.height),
                int(sensor.width),
                shifts[sensor.camera_id],
            )
            slot_of = {shape_id: slot for slot, shape_id in enumerate(layer_ids)}
            fractions: list[float] = []
            hidden: list[bool] = []
            for shape_id in scene.shape_ids:
                slot = slot_of.get(shape_id)
                if slot is None or not tracks[shape_id].exists:
                    fractions.append(0.0)
                    hidden.append(False)
                    continue
                footprint = footprints[slot]
                total = int(footprint.sum())
                visible = int(((owner == slot) & footprint).sum())
                fractions.append(0.0 if total == 0 else visible / float(total))
                hidden.append(bool(total > 0 and visible == 0))
            geometry[f"{sensor.camera_id}:image_px"] = image_px
            occlusion_mask[sensor.camera_id] = np.asarray(hidden, dtype=bool)
            visible_fraction[sensor.camera_id] = np.asarray(fractions, dtype=np.float32)
        geometry["scale_px"] = np.asarray(
            [tracks[shape_id].scale_px for shape_id in scene.shape_ids], dtype=np.float32
        )
        geometry["depth"] = np.asarray(
            [tracks[shape_id].depth for shape_id in scene.shape_ids], dtype=np.int16
        )
        for sensor in sensors:
            geometry[f"{sensor.camera_id}:camera_shift_px"] = np.asarray(
                shifts[sensor.camera_id], dtype=np.float64
            )
        labels = list(dict.fromkeys(state["labels"]))
        for (occluder, occluded), active in state["occlusion"].items():
            if active:
                labels.append(f"occlusion_active:{occluder}>{occluded}")
        for merge in events.false_merge:
            if merge.step <= step < merge.end_step:
                labels.append(f"false_merge_active:{merge.shape_id}+{merge.partner_id}")
        for split in events.false_split:
            if split.step <= step < split.end_step:
                labels.append(f"false_split_active:{split.shape_id}")
        for (subject, predicate, obj), active in state["relations"].items():
            if active:
                labels.append(f"relation_active:{predicate}:{subject}>{obj}")
        truth.append(
            EvaluationTruth(
                object_ids=tuple(scene.shape_ids),
                geometry=geometry,
                existence=existence,
                occlusion_mask=occlusion_mask,
                relations=_relations_for_frame(tracks, state, step),
                identity_links=tuple(
                    sorted(
                        (tracks[shape_id].appearance_id, shape_id) for shape_id in scene.shape_ids
                    )
                ),
                event_labels=tuple(sorted(dict.fromkeys(labels))),
                visible_fraction=visible_fraction,
            )
        )

        frame_id = f"{episode}-f{step:04d}"
        anchor_id = f"{episode}#anchor-{step:04d}"
        bundle = ObservationBundle(
            frames={
                sensor.camera_id: frame_measurements[sensor.camera_id].as_measurement_frame(
                    episode_id=episode, frame_id=frame_id, anchor_id=anchor_id
                )
                for sensor in sensors
            },
            context=EmbodimentContext(
                proprio=np.array(state["commanded_pan"], dtype=np.float32),
                proprio_available=True,
                executed_action=(
                    None if step == 0 else np.array(action_array[step - 1], dtype=np.float32)
                ),
                executed_action_available=bool(step > 0),
                camera_transform=None,
                camera_transform_available=False,
                task_tokens=np.asarray(
                    _task_tokens(scene, sensor_count=len(sensors)), dtype=np.int64
                ),
                task_mask=np.ones(_TASK_TOKEN_COUNT, dtype=bool),
                hypothesis_prior=np.full(8, 1.0 / 8.0, dtype=np.float32),
                hypothesis_prior_available=True,
            ),
            episode_ids=(episode,),
            anchor_ids=(anchor_id,),
            new_time_mask=np.asarray([True]),
        )
        inference.append(
            InferenceInput(
                bundle=bundle,
                declared_prior={
                    "version": f"{SYNTHETIC_VERSION}/uniform-K8",
                    "kind": "declared_uniform",
                    "candidates": 8,
                    "mass": [0.125] * 8,
                },
            )
        )

        future = np.zeros((int(action_horizon), 2), dtype=np.float32)
        mask = np.zeros((int(action_horizon),), dtype=bool)
        for offset in range(int(action_horizon)):
            index = step + offset
            if index < int(frames):
                future[offset] = np.asarray(action_array[index], dtype=np.float32)
                mask[offset] = True
        targets.append(
            TrainingTargets(
                actions=future,
                action_mask=mask,
                optional_geometry=geometry["world_px"].copy(),
                optional_relations=_overlap_adjacency(tracks, scene.shape_ids),
            )
        )

    manifest = _sequence_manifest(
        scene=scene,
        sensors=sensors,
        model=measurement_model,
        events=events,
        frames=int(frames),
        seed=int(seed),
        episode=episode,
        view_kind=view_kind,
        action_horizon=int(action_horizon),
        action_array=action_array,
    )
    return SyntheticSequence(
        sequence_id=f"{scene.scene_id}:{int(seed)}",
        scene_id=scene.scene_id,
        seed=int(seed),
        partition=scene.partition,
        cameras=tuple(sensors),
        frames=tuple(measurements),
        inference=tuple(inference),
        targets=tuple(targets),
        truth=tuple(truth),
        manifest=manifest,
        render_states=tuple(render_states),
    )


#: Declared synthetic language vocabulary size of the task instruction.
_TASK_TOKEN_COUNT = 4


def _task_tokens(scene: SceneSpec, *, sensor_count: int) -> list[int]:
    """Declared language token ids of the synthetic instruction."""
    text = (
        f"track {len(scene.shapes)} objects with {sensor_count} cameras "
        f"in {scene.partition} partition"
    )
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [int(digest[index]) for index in range(_TASK_TOKEN_COUNT)]


def _overlap_adjacency(tracks: dict[str, _ShapeTrack], shape_ids: tuple[str, ...]) -> np.ndarray:
    count = len(shape_ids)
    adjacency = np.zeros((count, count), dtype=np.float32)
    for first in range(count):
        for second in range(first + 1, count):
            one, two = tracks[shape_ids[first]], tracks[shape_ids[second]]
            if not (one.exists and two.exists):
                continue
            distance = float(np.linalg.norm(one.position - two.position))
            radius = shape_footprint_radius(one.scale_px) + shape_footprint_radius(two.scale_px)
            if distance < radius:
                adjacency[first, second] = 1.0
                adjacency[second, first] = 1.0
    return adjacency


def _sequence_manifest(
    *,
    scene: SceneSpec,
    sensors: tuple[SensorSpec, ...],
    model: MeasurementModel,
    events: EventSchedule,
    frames: int,
    seed: int,
    episode: str,
    view_kind: str,
    action_horizon: int,
    action_array: Any,
) -> dict[str, Any]:
    """JSON-safe declarations of one sequence; measured counts only."""
    return {
        "synthetic_version": SYNTHETIC_VERSION,
        "scene_id": scene.scene_id,
        "scene_hash": scene_hash(scene),
        "schedule_hash": schedule_hash(events),
        "partition": scene.partition,
        "scene_seed": int(scene.seed),
        "sequence_seed": int(seed),
        "episode_id": episode,
        "view_kind": view_kind,
        "frames": int(frames),
        "action_horizon": int(action_horizon),
        "declared_action_space": {
            "dims": 2,
            "unit": "px",
            "kind": "commanded_camera_pan",
            "range": [-2, 2],
            "note": (
                "synthetic declared control space for the mechanism set; not the "
                "LIBERO/RoboTwin action protocol"
            ),
        },
        "declared_action_stats": {
            "steps": int(action_array.shape[0]),
            "min": int(np.asarray(action_array).min()),
            "max": int(np.asarray(action_array).max()),
        },
        "shape_ids": list(scene.shape_ids),
        "shape_kinds": list(scene.shape_kinds),
        "object_count": len(scene.shapes),
        "occlusion_plan_modes": [plan.mode for plan in scene.occlusion_plan],
        "interaction_predicates": sorted({plan.predicate for plan in scene.interaction_plan}),
        "background_level": float(scene.background_level),
        "lighting_base": float(scene.lighting_plan.base_illumination),
        "event_types": list(events.event_types()),
        "event_counts": events.counts(),
        "camera_shift_cameras": sorted({event.camera_id for event in events.camera_shift}),
        "cameras": [sensor.as_dict() for sensor in sensors],
        "measurement_model": {
            "kind": model.kind,
            "version": model.version,
            "unit": model.unit(),
            "storage_dtype": str(model.storage_dtype().__name__),
            "shot_noise_scale": float(model.shot_noise_scale),
            "read_noise_e": float(model.read_noise_e),
            "saturation_level": float(model.saturation_level),
            "response": [float(item) for item in model.response],
            "bad_pixel_fraction": float(model.bad_pixel_fraction),
            "missing_sample_fraction": float(model.missing_sample_fraction),
            "signal_unit": SIGNAL_UNIT,
            "pixel_centre_convention": PIXEL_CENTRE,
        },
        "truth_isolation": (
            "InferenceInput carries RAW measurements, declared calibration, embodiment "
            "context, language tokens, executed-action history and a declared prior only; "
            "EvaluationTruth and TrainingTargets are separate objects"
        ),
        "declared_targets": {
            "P1_train_frames": P1_TRAIN_FRAME_TARGET,
            "P2_sequences": P2_SEQUENCE_TARGET,
            "P2_sequence_frames": list(P2_SEQUENCE_FRAMES),
            "status": "targets_declared_not_built",
        },
        "p0_scale_claim": "small_sample_only",
    }


# --------------------------------------------------------------------------------------
# dataset iterator with separated counters
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DatasetCounters:
    """Counts reported per dataset pass; ``independent_scenes`` is never inflated
    by augmented views of the same scene."""

    independent_scenes: int = 0
    sequences: int = 0
    frames: int = 0
    augmented_views: int = 0
    measurements: int = 0
    events: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "independent_scenes": self.independent_scenes,
            "sequences": self.sequences,
            "frames": self.frames,
            "augmented_views": self.augmented_views,
            "measurements": self.measurements,
            "events": self.events,
        }


class SyntheticDataset(Iterator[SyntheticSequence]):
    """Streaming dataset over partitioned scenes.

    Each scene is generated once and re-rendered for ``views_per_scene`` views.
    View 0 is the independent rendering; later views reuse the same scene and the
    same event program but add a declared exposure augmentation and a new noise
    seed, so they are counted as augmented views, never as new scenes. Camera-shift
    events target the first declared camera, which the sequence manifest records.
    """

    #: Declared exposure multipliers of augmented views (never 1.0).
    AUGMENTATION_EXPOSURE_SCALES = (0.85, 1.15, 0.7)

    def __init__(
        self,
        *,
        sensors: tuple[SensorSpec, ...],
        measurement_model: MeasurementModel,
        partitions: tuple[str, ...] = ("train",),
        scenes: int = 8,
        frames: int = 32,
        views_per_scene: int = 1,
        seed: int = 0,
        scene_id_prefix: str = "",
        canvas: tuple[int, int] = (64, 64),
        action_horizon: int = 10,
        event_counts: dict[str, int] | None = None,
    ) -> None:
        if not partitions:
            raise ValueError("at least one partition is required")
        for partition in partitions:
            if partition not in VOCABULARIES:
                raise ValueError(f"unknown partition: {partition!r}")
        if scenes <= 0 or frames < 2 or views_per_scene <= 0:
            raise ValueError("scenes must be positive, frames >= 2, views_per_scene positive")
        self.sensors = tuple(sensors)
        self.measurement_model = measurement_model
        self.partitions = tuple(partitions)
        self.scenes = int(scenes)
        self.frames = int(frames)
        self.views_per_scene = int(views_per_scene)
        self.seed = int(seed)
        self.scene_id_prefix = str(scene_id_prefix)
        self.canvas = (int(canvas[0]), int(canvas[1]))
        self.action_horizon = int(action_horizon)
        self.event_counts = dict(event_counts or {})
        self.counters = DatasetCounters()
        self._iteration = 0

    def _scene_id(self, partition: str, index: int) -> str:
        return f"{self.scene_id_prefix}{partition}-{index:04d}"

    def __iter__(self) -> SyntheticDataset:
        self.counters = DatasetCounters()
        self._iteration += 1
        return self

    def __next__(self) -> SyntheticSequence:
        total = len(self.partitions) * self.scenes * self.views_per_scene
        if self.counters.sequences >= total:
            raise StopIteration
        sequence_index = self.counters.sequences
        partition = self.partitions[sequence_index // (self.scenes * self.views_per_scene)]
        remainder = sequence_index % (self.scenes * self.views_per_scene)
        scene_index = remainder // self.views_per_scene
        view_index = remainder % self.views_per_scene
        scene_id = self._scene_id(partition, scene_index)
        scene = generate_scene(
            partition,
            scene_id,
            canvas=self.canvas,
            horizon=self.frames,
        )
        schedule = build_event_schedule(
            scene,
            frames=self.frames,
            seed=derive_seed("n02-full-schedule", partition, scene_id, self.seed),
            camera_id=self.sensors[0].camera_id if self.sensors else "camera-0",
            **self.event_counts,
        )
        sequence_seed = derive_seed("n02-full-view", partition, scene_id, view_index, self.seed)
        view_kind = "independent" if view_index == 0 else "augmented_exposure"
        if view_index > 0:
            multiplier = self.AUGMENTATION_EXPOSURE_SCALES[
                (view_index - 1) % len(self.AUGMENTATION_EXPOSURE_SCALES)
            ]
            schedule = dataclasses.replace(
                schedule,
                exposure=(ExposureEvent(step=0, scale=float(multiplier)), *schedule.exposure),
            )
        sequence = generate_sequence(
            scene,
            self.sensors,
            self.measurement_model,
            schedule,
            frames=self.frames,
            seed=sequence_seed,
            action_horizon=self.action_horizon,
            view_kind=view_kind,
            episode_id=f"{scene_id}-v{view_index}",
        )
        self.counters.sequences += 1
        self.counters.frames += sequence.num_frames
        self.counters.measurements += sequence.num_frames * len(self.sensors)
        self.counters.events += sum(schedule.counts().values())
        if view_index == 0:
            self.counters.independent_scenes += 1
        else:
            self.counters.augmented_views += 1
        return sequence

    def manifest(self) -> dict[str, Any]:
        """Declared plan plus the counters of the most recent pass."""
        return {
            "synthetic_version": SYNTHETIC_VERSION,
            "partitions": list(self.partitions),
            "scenes_per_partition": self.scenes,
            "frames_per_sequence": self.frames,
            "views_per_scene": self.views_per_scene,
            "augmentation": {
                "kind": "declared_exposure_augmentation",
                "exposure_scales": list(self.AUGMENTATION_EXPOSURE_SCALES),
                "counted_as": "augmented_views",
                "independent_scene_claim": False,
            },
            "cameras": [sensor.as_dict() for sensor in self.sensors],
            "measurement_model_kind": self.measurement_model.kind,
            "counters": self.counters.as_dict(),
            "declared_targets": {
                "P1_train_frames": P1_TRAIN_FRAME_TARGET,
                "P2_sequences": P2_SEQUENCE_TARGET,
                "P2_sequence_frames": list(P2_SEQUENCE_FRAMES),
                "status": "targets_declared_not_built",
            },
            "p0_scale_claim": "small_sample_only",
        }


def generate_dataset(**kwargs: Any) -> SyntheticDataset:
    """Build a streaming dataset; the object itself is the iterator."""
    return SyntheticDataset(**kwargs)


def sequences_for_frame_target(target_frames: int, *, frames_per_sequence: int) -> int:
    """Sequences needed to reach a declared frame target (ceiling division).

    A planning helper only: the returned number is the parameter to hand to
    :func:`generate_dataset`, never a claim that those frames were generated.
    """
    if target_frames <= 0 or frames_per_sequence <= 0:
        raise ValueError("targets and sequence lengths must be positive")
    return -(-int(target_frames) // int(frames_per_sequence))


# --------------------------------------------------------------------------------------
# cost profile
# --------------------------------------------------------------------------------------


def _array_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Exact number of array bytes reachable from ``value`` (no duplicates)."""
    if seen is None:
        seen = set()
    if isinstance(value, np.ndarray):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return int(value.nbytes)
    total = 0
    if isinstance(value, dict):
        for item in value.values():
            total += _array_bytes(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            total += _array_bytes(item, seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        for spec in dataclasses.fields(value):
            total += _array_bytes(getattr(value, spec.name), seen)
    return total


def measure_cost(
    *,
    frames: int = 32,
    seed: int = 0,
    model_kind: str = "cfa",
    cameras: int = 2,
    height: int = 64,
    width: int = 64,
    partition: str = "train",
    scene_id: str = "cost-probe",
    sequences: int = 1,
    canvas: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Measure real generation cost of one declared configuration.

    Wall time is ``time.perf_counter`` around the whole probe and memory is the
    ``tracemalloc`` peak of tracked Python/numpy allocations plus the exact sum of
    array bytes of the produced sequence. Nothing here is an estimate.
    """
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"unknown measurement model kind: {model_kind!r}")
    sensors: list[SensorSpec] = []
    for index in range(int(cameras)):
        if model_kind == "cfa":
            pattern = ("RGGB", "BGGR", "GRBG", "GBRG")[index % 4]
            sensors.append(
                SensorSpec(
                    camera_id=f"cam{index}",
                    height=height,
                    width=width,
                    input_mode="bayer_cfa",
                    cfa_pattern=pattern,
                    black_level=64.0,
                    white_level=4095.0,
                    bit_depth=12,
                    read_noise_e=2.0,
                    gain=1.0,
                    exposure_ms=1.0,
                )
            )
        else:
            sensors.append(
                SensorSpec(
                    camera_id=f"cam{index}",
                    height=height,
                    width=width,
                    input_mode="rawvla_rgb3",
                    black_level=0.0,
                    white_level=4095.0,
                    bit_depth=16,
                    read_noise_e=0.0,
                    gain=1.0,
                    exposure_ms=1.0,
                )
            )
    model = (
        MeasurementModel(kind=model_kind, bad_pixel_fraction=0.0005, missing_sample_fraction=0.0002)
        if model_kind == "cfa"
        else MeasurementModel(
            kind="raw3",
            response=(0.95, 1.0, 0.75),
            bad_pixel_fraction=0.0005,
            missing_sample_fraction=0.0002,
        )
    )
    sensor_tuple = tuple(sensors)
    scene = generate_scene(
        partition,
        scene_id,
        canvas=canvas if canvas is not None else (int(height), int(width)),
        horizon=int(frames),
    )
    schedule = build_event_schedule(
        scene, frames=int(frames), seed=seed + scene.seed, camera_id=sensor_tuple[0].camera_id
    )

    tracemalloc.start()
    try:
        start = time.perf_counter()
        produced = [
            generate_sequence(
                scene,
                sensor_tuple,
                model,
                schedule,
                frames=int(frames),
                seed=seed + 1000 * index,
            )
            for index in range(int(sequences))
        ]
        wall_seconds = time.perf_counter() - start
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    array_bytes = sum(_array_bytes(sequence) for sequence in produced)
    total_frames = int(frames) * int(sequences)
    total_measurements = total_frames * int(cameras)
    return {
        "model_kind": model_kind,
        "cameras": [sensor.as_dict() for sensor in sensor_tuple],
        "frames_per_sequence": int(frames),
        "sequences": int(sequences),
        "measurements": total_measurements,
        "wall_seconds_total": float(wall_seconds),
        "wall_seconds_per_sequence": float(wall_seconds) / max(1, int(sequences)),
        "wall_ms_per_frame": float(wall_seconds) * 1000.0 / max(1, total_frames),
        "wall_ms_per_measurement": float(wall_seconds) * 1000.0 / max(1, total_measurements),
        "tracemalloc_peak_bytes": int(peak_bytes),
        "tracemalloc_current_bytes": int(current_bytes),
        "sequence_array_bytes": int(array_bytes),
        "array_bytes_per_frame": float(array_bytes) / max(1, total_frames),
        "measured_frames": total_frames,
        "environment": {
            "numpy": np.__version__,
            "python_flavor": "CPython",
            "threads": 1,
        },
        "notes": [
            "wall time is a single-process perf_counter measurement",
            "tracemalloc peak covers tracked Python/numpy allocations, not process RSS",
            "sequence_array_bytes counts distinct arrays reachable from the sequences",
            "P0 scale claim is small_sample_only; the P1/P2 targets are not built",
        ],
        "p0_scale_claim": "small_sample_only",
    }

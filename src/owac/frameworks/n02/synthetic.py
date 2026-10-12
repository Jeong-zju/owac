"""Synthetic measurement-domain generator for the N02 mechanism experiments.

The generator produces *measurements*, not images. A latent geometry is mapped by
a declared forward operator onto a sensor sampling grid; nothing in the pipeline
builds an RGB picture, and no RGB teacher or reconstruction target is emitted.
Hidden geometry appears only in supervision fields, so an encoder can never read
the answer out of its input.

Three tasks, one two-hypothesis structure
-----------------------------------------
``opening_direction``
    A handle arc spanning ``w`` degrees of a ``[r_in, r_out]`` radial band has its
    opening gap on one side. ``left`` and ``right``. Declared widths are 90-150
    degrees, so the ambiguous arc is a plainly visible segment and not a sliver.
``handle_orientation``
    The arc's major axis is ``horizontal`` or ``vertical``.
``contour_connected``
    Two contour segments are joined (``connected``) or separated (``separate``).

One uniform parameterisation covers all three: the latent value fixes a *cue
angle* ``theta_v``, and the forward operator places a cue marker at that angle.
The same parameterisation makes the two-hypothesis case hand-checkable in
:class:`ExplicitObservationModel`.

The forward operator
--------------------
For a sample coordinate ``p`` (continuous pixel-centre coordinates), with arc
centre ``c``, radii ``(r_in, r_out)`` and the latent cue angle ``theta``::

    d        = ||p - c||
    phi      = atan2(p_y - c_y, p_x - c_x)                  # in [0, 2*pi)
    g        = clip(min(d - r_in, r_out - d) / edge, 0, 1)
    arc_m(p) = g * sigma_cos(phi, arc_angle, w)              # w = angular width
    cue_m(p) = exp(-||p - m||^2 / (2 * s^2))
    clean_m  = arc_m + strength * cue_m

``m = c + r_orbit * (cos theta, sin theta)`` is the cue marker at the latent
value's cue angle ``theta``. The arc is *genuinely ambiguous*: its angular centre
``arc_angle`` is a fixed scene fact that does **not** depend on the latent value,
so both hypotheses render the identical arc and only the marker's position
carries the answer. The marker rides a separate orbit
``r_orbit = r_out + 2 * edge + 3 * s`` whose *inner* tail is exactly tangent to
the arc band's outer edge, so the two signals are radially disjoint and no
hypothesis can be read off the arc.

Evidence is never a fixed frame index. The arrival step is sampled per episode
from a Beta distribution stretched over the sequence and clipped away from both
ends, and the per-frame strength is an independent random ramp. Episodes built
with ``identifiable=False`` keep the cue at amplitude zero for every frame, so
their measurements genuinely contain no cue.

The declared measurement law is additive Gaussian with a known variance
(``noise_sigma**2``), which is what makes the explicit mode's likelihood exact::

    frames = transmission * clean + noise,  noise ~ N(0, noise_sigma^2 * I)

``transmission`` is the occluder attenuation in ``[max_occluder_transmission, 1]``
applied after the signal is formed, so a later (and therefore more occluded)
frame cannot make the cue *cleaner*. Both terms of the equation are recorded, so
a test can assert the equality exactly rather than approximately.

What the generator refuses
--------------------------
* No RGB/sRGB/XYZ plane, no re-mosaicking of a three-channel image.
* No prior condition inside the measurement. ``prior_condition`` is an explicit
  episode field; changing it changes the *declared context*, never the frames.
* No answer in the input. ``frames`` and ``metadata`` depend on the latent
  geometry, the evidence schedule and the noise only; the latent value, the
  per-step stage truth and the cue angle live in supervision fields.

Modes
-----
``rawvla_rgb3``
    Three co-sited channels sampled at identical pixel centres, one value per
    pixel repeated across channels. This exists so mechanism code can run against
    the author's pseudo-RAW input shape; it is labelled as *not* a CFA.
``bayer_cfa``
    One filter per photosite. Four planes ``(R, G1, G2, B)`` are sampled
    separately at their own coordinates with the ``+0.5`` pixel-centre
    convention, and each plane sees the signal through its own declared spectral
    response, so the planes are never treated as four co-sited colours. Plane
    ``p`` sample ``(i, j)`` reads the sensor at ``positions[p] + 2 * (i, j)``.

Verification hooks
------------------
* :meth:`SyntheticGenerator.clean_measurements` recomputes the forward operator,
  so a test can assert ``frames == transmission * clean + noise`` exactly.
* :meth:`SyntheticGenerator.infer_cue_angle` recovers the cue's angular position
  from the measurements alone (radial-band intensity centroid), independently of
  the renderer's own bookkeeping.
* :class:`ExplicitObservationModel` gives a one-dimensional two-hypothesis model
  whose Bayes update is exact and hand-checkable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "CFA_PATTERNS",
    "CFA_PHASES",
    "MISSING_PRIOR_LOG_ODDS",
    "OCCLUDER_TRANSPARENT_LEVEL",
    "PLANE_COLOURS",
    "PLANE_TILE_OFFSETS",
    "PRIOR_CONDITIONS",
    "SAMPLE_XY_ROWS",
    "SPECTRAL_RESPONSE",
    "TASK_NAMES",
    "ExplicitObservationModel",
    "LatentScene",
    "MeasurementContext",
    "PriorSpec",
    "SyntheticEpisode",
    "SyntheticGenerator",
    "SyntheticOptions",
    "TaskSpec",
    "cfa_plane_positions",
    "cfa_sample",
    "cue_angle_for_value",
    "default_tasks",
    "gaussian_logpdf",
    "log_likelihood_ratio",
    "marker_xy",
    "measurement_field",
    "render_plane",
    "sample_coordinates",
    "sigma_cos",
    "truth_value",
    "values_for_task",
    "wrapped_angle_difference",
]

#: CFA colour order for a 2x2 Bayer tile, stored as ``((0,0), (0,1), (1,0), (1,1))``.
CFA_PATTERNS: dict[str, tuple[str, str, str, str]] = {
    "RGGB": ("R", "G", "G", "B"),
    "BGGR": ("B", "G", "G", "R"),
    "GRBG": ("G", "R", "B", "G"),
    "GBRG": ("G", "B", "R", "G"),
}

#: Alias kept for the keyword used by :func:`sample_coordinates`.
CFA_PHASES: tuple[str, ...] = tuple(CFA_PATTERNS)

#: Plane order of the 2x2 rearrangement.
PLANE_COLOURS: tuple[str, str, str, str] = ("R", "G1", "G2", "B")

#: ``(dy, dx)`` of each plane inside the 2x2 tile.
PLANE_TILE_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

#: Declared filter response per plane. These are *declared* numbers, not measured
#: sensor data: the handle reads warm and the occluder reads cool.
SPECTRAL_RESPONSE: dict[str, dict[str, float]] = {
    "handle": {"R": 1.00, "G1": 0.86, "G2": 0.86, "B": 0.58},
    "cue": {"R": 1.00, "G1": 0.92, "G2": 0.92, "B": 0.82},
}

PRIOR_CONDITIONS: tuple[str, ...] = (
    "correct",
    "misleading",
    "uninformative",
    "missing",
    "stale",
)

TASK_NAMES: tuple[str, ...] = ("opening_direction", "handle_orientation", "contour_connected")

#: Entries in one ``sample_xy`` descriptor row: first sample, step, origin.
SAMPLE_XY_ROWS = 3

#: Half-width of the soft angular window used by :func:`sigma_cos`.
SIGMA_COS_HALF_WIDTH = 0.35

#: Log-odds assigned to a declared-missing prior.
MISSING_PRIOR_LOG_ODDS = -math.inf

#: Attenuation floor below which an occluder is declared optically transparent.
#: A Gaussian tail never reaches exactly zero, so the occlusion contract is stated
#: against this declared level instead of against an unattainable exact zero.
OCCLUDER_TRANSPARENT_LEVEL = 1.0e-3


# --------------------------------------------------------------------------------------
# tasks and priors
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One two-way task: the latent answer to cue-angle mapping.

    ``cue_base`` is the cue angle both values share (the ambiguous part of the
    geometry) and ``cue_separation`` the angular step from value 0 to value 1.
    """

    name: str
    values: tuple[str, str]
    cue_base: float = 0.0
    cue_separation: float = math.pi

    def cue_angle(self, value: str) -> float:
        if value not in self.values:
            raise ValueError(f"unknown value {value!r} for task {self.name!r}")
        return float(self.cue_base + self.values.index(value) * self.cue_separation)


def default_tasks() -> dict[str, TaskSpec]:
    """The three declared tasks, all sharing one two-hypothesis structure."""
    return {
        "opening_direction": TaskSpec(
            name="opening_direction",
            values=("left", "right"),
            cue_base=0.0,
            cue_separation=math.pi,
        ),
        "handle_orientation": TaskSpec(
            name="handle_orientation",
            values=("horizontal", "vertical"),
            cue_base=math.pi / 2.0,
            cue_separation=math.pi / 2.0,
        ),
        "contour_connected": TaskSpec(
            name="contour_connected",
            values=("connected", "separate"),
            cue_base=0.0,
            cue_separation=math.pi,
        ),
    }


def values_for_task(task: str) -> tuple[str, str]:
    return default_tasks()[task].values


def cue_angle_for_value(task: str, value: str) -> float:
    """Declared cue angle of ``value`` for ``task``."""
    return default_tasks()[task].cue_angle(value)


def truth_value(task: str, prior_value: str, prior_condition: str, changed: bool) -> str:
    """Resolve the true latent value for a prior condition.

    ``stale`` is the only condition where the answer and the prior disagree for a
    reason other than being wrong: the prior was true when it was formed and the
    scene changed afterwards.
    """
    values = values_for_task(task)
    if prior_condition == "stale" and changed:
        return values[1 - values.index(prior_value)]
    return prior_value


@dataclass(frozen=True, slots=True)
class PriorSpec:
    """Declared prior and its condition. Never encoded into the measurement."""

    condition: str
    value: str
    log_odds: float

    def __post_init__(self) -> None:
        if self.condition not in PRIOR_CONDITIONS:
            raise ValueError(f"unknown prior condition: {self.condition!r}")

    @property
    def available(self) -> bool:
        return self.condition != "missing"

    def context(self, task: str) -> dict[str, Any]:
        return {
            "prior_condition": self.condition,
            "prior_value": None if not self.available else self.value,
            "prior_log_odds": None if not self.available else float(self.log_odds),
            "prior_is_uninformative": self.condition == "uninformative",
            "prior_may_be_stale": self.condition == "stale",
            "task_value_count": len(values_for_task(task)),
        }


# --------------------------------------------------------------------------------------
# geometry primitives (the declared forward operator)
# --------------------------------------------------------------------------------------


def wrapped_angle_difference(a: Any, b: Any) -> Any:
    """``a - b`` mapped into ``(-pi, pi]``."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi


def sigma_cos(angles: Any, center: float, half_width: float = SIGMA_COS_HALF_WIDTH) -> np.ndarray:
    """Soft angular window ``exp(-z^2 / (2 * half_width^2))`` with ``z`` wrapped."""
    z = wrapped_angle_difference(angles, center)
    return np.exp(-(z**2) / (2.0 * half_width**2))


def sample_coordinates(
    height: int,
    width: int,
    *,
    cfa_pattern: str = "",
    point_sampling: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sensor sampling coordinates as ``(positions [n, 2], grid [H, W, ...])``.

    ``rawvla_rgb3`` samples every pixel centre, so ``grid`` is the regular
    ``(row + 0.5, col + 0.5)`` lattice and ``positions`` its ``[H*W, 2]``
    flattening. ``bayer_cfa`` samples one filter per photosite; ``positions`` is
    still the full lattice (flattened in row-major order) and the plane members
    are selected by :func:`cfa_sample`, which is what keeps each plane's true
    coordinate rather than pretending the planes are co-sited.

    With ``point_sampling=False`` each pixel integrates a 2x2 box of four
    sub-samples (``grid`` keeps a trailing axis of 4); the per-site coordinates
    returned in ``positions`` are unchanged.
    """
    if cfa_pattern and cfa_pattern not in CFA_PATTERNS:
        raise ValueError(f"unknown CFA pattern: {cfa_pattern!r}")
    if height % 2 or width % 2:
        raise ValueError("height and width must be even so a 2x2 tile is complete")
    rows = np.arange(height, dtype=np.float64) + 0.5
    cols = np.arange(width, dtype=np.float64) + 0.5
    lattice = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1)
    if point_sampling:
        return lattice.reshape(-1, 2).copy(), lattice
    offsets = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    sub = lattice[:, :, None, :] + offsets[None, None, :, :] - 0.5
    return lattice.reshape(-1, 2).copy(), sub


def _arc_value(
    coordinates: np.ndarray,
    *,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    angular_center: float,
    angular_width: float,
    edge_softness: float,
) -> np.ndarray:
    """Radial band mask times the soft angular window, evaluated at ``coordinates``.

    ``coordinates`` is ``(y, x)`` like every other coordinate in this module, and
    ``phi`` is the standard math angle from the ``+x`` axis in that ``(y, x)``
    plane, so ``phi = 0`` points along ``+x`` (increasing column) and ``phi =
    pi/2`` along ``+y`` (increasing row).
    """
    dy = coordinates[:, 0] - center[0]
    dx = coordinates[:, 1] - center[1]
    distance = np.hypot(dy, dx)
    phi = np.arctan2(dy, dx) % (2.0 * np.pi)
    radial = np.minimum(distance - inner_radius, outer_radius - distance)
    radial = np.clip(radial / max(edge_softness, 1.0e-6), 0.0, 1.0)
    return radial * sigma_cos(phi, angular_center, angular_width)


def cue_orbit_radius(outer_radius: float, edge_softness: float, cue_sigma: float) -> float:
    """Radius of the cue orbit, measured from the arc band's outer edge."""
    return outer_radius + 2.0 * edge_softness + 3.0 * cue_sigma


def cue_clearance(outer_radius: float, edge_softness: float, cue_sigma: float) -> float:
    """Minimum allowed distance between the arc centre and each frame edge.

    The cue marker of the *other* hypothesis sits on the cue orbit. The occluder
    wedge only exists beyond ``required = orbit + 5 * sigma``, so requiring that
    marker's centre to be at least ``required + 3 * sigma`` from the edge puts its
    whole 3-sigma disk beyond ``required``. The bound is derived, not tuned, and
    :meth:`SyntheticGenerator.validate_scene` checks it numerically instead of
    trusting the derivation.
    """
    orbit = cue_orbit_radius(outer_radius, edge_softness, cue_sigma)
    # The farther marker sits at ``orbit`` from the centre, the occluder starts at
    # ``orbit + 3 * sigma`` beyond it, and the nearer marker must stay inside the
    # frame: distance from the frame edge >= orbit + (orbit + 3 * sigma) - orbit.
    return 2.0 * orbit + 4.0 * cue_sigma


def marker_xy(scene: LatentScene, cue_angle: float | None = None) -> np.ndarray:
    """Declared ``(y, x)`` centre of the cue marker for a scene.

    The marker rides :attr:`LatentScene.cue_orbit_radius`, which sits outside the
    arc band so the two signals are radially disjoint.
    """
    theta = cue_angle_for_value(scene.task, scene.value) if cue_angle is None else float(cue_angle)
    return np.array(
        [
            scene.arc_center_xy[0] + scene.cue_orbit_radius * math.cos(theta),
            scene.arc_center_xy[1] + scene.cue_orbit_radius * math.sin(theta),
        ],
        dtype=np.float64,
    )


def measurement_field(
    coordinates: np.ndarray,
    scene: LatentScene,
    *,
    cue_strength: float,
    cue_angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward operator: latent geometry -> ``(arc, cue, clean)`` at ``coordinates``.

    ``arc`` is the ambiguous handle signal, ``cue`` the normalised distinguishing
    marker and ``clean = arc + cue_strength * cue``. The marker sits at the *true*
    latent value's cue angle, so its position - not a frame index - carries the
    answer.
    """
    theta = cue_angle_for_value(scene.task, scene.value) if cue_angle is None else float(cue_angle)
    arc = _arc_value(
        coordinates,
        center=scene.arc_center_xy,
        inner_radius=scene.inner_radius,
        outer_radius=scene.outer_radius,
        angular_center=scene.arc_angle,
        angular_width=scene.angular_width,
        edge_softness=scene.edge_softness,
    )
    return _compose_measurement(coordinates, scene, arc, theta, cue_strength)


def _compose_measurement(
    coordinates: np.ndarray,
    scene: LatentScene,
    arc: np.ndarray,
    theta: float,
    cue_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine a precomputed arc field with the cue marker at ``theta``."""
    marker = marker_xy(scene, theta)
    squared = (coordinates[:, 0] - marker[0]) ** 2 + (coordinates[:, 1] - marker[1]) ** 2
    cue = np.exp(-squared / (2.0 * scene.cue_sigma**2))
    return arc, cue, arc + float(cue_strength) * cue


def render_plane(
    coordinates: np.ndarray,
    scene: LatentScene,
    *,
    cue_strength: float,
    base_value: float,
    plane: str,
    cue_angle: float | None = None,
) -> np.ndarray:
    """One CFA plane's *measurement* values from the same physical signal.

    The plane sees the arc and the cue through its own declared filter response,
    so plane values differ by more than a resampling of one co-sited colour.
    """
    arc, cue, _ = measurement_field(
        coordinates, scene, cue_strength=cue_strength, cue_angle=cue_angle
    )
    if plane not in SPECTRAL_RESPONSE["handle"]:
        raise ValueError(f"unknown plane {plane!r}")
    return (
        base_value
        + SPECTRAL_RESPONSE["handle"][plane] * arc
        + cue_strength * SPECTRAL_RESPONSE["cue"][plane] * cue
    )


def _tile_of_plane_indices(pattern: str) -> tuple[int, int, int, int]:
    """Tile index (row-major, 0..3) of each ``(R, G1, G2, B)`` plane."""
    if pattern not in CFA_PATTERNS:
        raise ValueError(f"unknown CFA pattern: {pattern!r}")
    tiles = CFA_PATTERNS[pattern]
    for colour in ("R", "B"):
        if tiles.count(colour) != 1:
            raise ValueError(f"pattern {pattern!r} must place exactly one {colour}")
    if tiles.count("G") != 2:
        raise ValueError(f"pattern {pattern!r} must place exactly two G samples")
    return (
        tiles.index("R"),
        tiles.index("G"),
        tiles.index("G", tiles.index("G") + 1),
        tiles.index("B"),
    )


def cfa_plane_positions(
    pattern: str,
    grid_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """``[4, 2]`` ``(y, x)`` coordinate of each ``(R, G1, G2, B)`` plane's first sample.

    Uses the ``+0.5`` pixel-centre convention, so ``RGGB`` puts ``R`` at
    ``(0.5, 0.5)`` and ``G1`` at ``(0.5, 1.5)``: the planes are at least one pixel
    apart in one axis and therefore never co-sited. Plane ``p`` sample ``(i, j)``
    reads the sensor at ``positions[p] + 2 * (i, j)``.
    """
    positions = np.empty((4, 2), dtype=np.float64)
    for plane_index, tile_index in enumerate(_tile_of_plane_indices(pattern)):
        ty, tx = divmod(int(tile_index), 2)
        positions[plane_index] = (
            grid_offset[0] + ty + 0.5,
            grid_offset[1] + tx + 0.5,
        )
    return positions


def cfa_sample(
    field: np.ndarray,
    pattern: str,
    *,
    grid_offset: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a full-resolution measurement plane on a CFA mosaic.

    Returns ``(values [4, H/2, W/2], positions [4, 2])``. Each plane is sampled at
    its own photosites; treating the four planes as four co-sited colours would
    invent three samples per pixel.
    """
    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"field must be 2-D, got shape {array.shape}")
    if array.shape[0] % 2 or array.shape[1] % 2:
        raise ValueError(f"field needs even H and W, got {array.shape}")
    if grid_offset[0] % 2 or grid_offset[1] % 2:
        raise ValueError("grid_offset must be even to preserve the CFA phase")
    positions = cfa_plane_positions(pattern, grid_offset)
    values = np.empty((4, array.shape[0] // 2, array.shape[1] // 2), dtype=np.float64)
    for plane_index, tile_index in enumerate(_tile_of_plane_indices(pattern)):
        dy, dx = divmod(int(tile_index), 2)
        values[plane_index] = array[dy::2, dx::2]
    return values, positions


# --------------------------------------------------------------------------------------
# scenes, priors, options
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatentScene:
    """The hidden scene. Present in supervision only, never in the measurement."""

    task: str
    value: str
    arc_center_xy: tuple[float, float]
    inner_radius: float
    outer_radius: float
    angular_width: float
    edge_softness: float
    cue_radius: float
    cue_sigma: float
    arc_angle: float
    shape_id: str
    position_id: str
    occlusion_template: str

    @property
    def cue_orbit_radius(self) -> float:
        """Radius of the cue orbit, tangent to the arc band's outer edge.

        ``r_out + 2 * edge_softness + 3 * cue_sigma`` puts the marker's inner
        tail exactly where the arc field has decayed to zero, so the arc and
        the cue never overlap and the answer cannot be read off the arc.
        """
        return cue_orbit_radius(self.outer_radius, self.edge_softness, self.cue_sigma)

    def with_value(self, value: str) -> LatentScene:
        """Same geometry with the other latent value (used for the `stale` flip)."""
        return dataclasses.replace(self, value=value)

    def as_supervision(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "value": self.value,
            "arc_center_xy": list(self.arc_center_xy),
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "angular_width": self.angular_width,
            "edge_softness": self.edge_softness,
            "cue_radius": self.cue_radius,
            "cue_sigma": self.cue_sigma,
            "arc_angle": self.arc_angle,
            "shape_id": self.shape_id,
            "position_id": self.position_id,
            "occlusion_template": self.occlusion_template,
        }

    @classmethod
    def from_supervision(cls, payload: dict[str, Any]) -> LatentScene:
        return cls(
            task=str(payload["task"]),
            value=str(payload["value"]),
            arc_center_xy=(float(payload["arc_center_xy"][0]), float(payload["arc_center_xy"][1])),
            inner_radius=float(payload["inner_radius"]),
            outer_radius=float(payload["outer_radius"]),
            angular_width=float(payload["angular_width"]),
            edge_softness=float(payload["edge_softness"]),
            cue_radius=float(payload["cue_radius"]),
            cue_sigma=float(payload["cue_sigma"]),
            arc_angle=float(payload["arc_angle"]),
            shape_id=str(payload["shape_id"]),
            position_id=str(payload["position_id"]),
            occlusion_template=str(payload["occlusion_template"]),
        )


@dataclass(frozen=True, slots=True)
class SyntheticOptions:
    """Declared generator geometry and sampling conditions."""

    height: int = 80
    width: int = 80
    channels: int = 3
    input_mode: str = "rawvla_rgb3"
    cfa_pattern: str = "RGGB"
    base_value: float = 0.25
    white_level: float = 1.0
    noise_sigma: float = 0.01
    occlusion_schedule: str = "linear"
    max_occluder_transmission: float = 0.35
    identifiability: float = 0.8
    arrival_shape_a: float = 1.4
    arrival_shape_b: float = 1.1
    arrival_lo: float = 0.20
    arrival_hi: float = 0.92
    staleness: float = 0.5
    point_sampling: bool = True
    #: Angle grid step (radians) used by the measurement-only cue estimator.
    cue_angle_grid_step: float = math.radians(2.0)

    def __post_init__(self) -> None:
        if self.input_mode not in {"rawvla_rgb3", "bayer_cfa"}:
            raise ValueError(f"unknown input_mode: {self.input_mode!r}")
        if self.input_mode == "bayer_cfa" and self.cfa_pattern not in CFA_PATTERNS:
            raise ValueError(f"bayer_cfa requires a CFA pattern, got {self.cfa_pattern!r}")
        if self.height % 2 or self.width % 2:
            raise ValueError("height and width must be even so a 2x2 CFA tile is complete")
        if self.channels not in {1, 3}:
            raise ValueError(f"rawvla_rgb3 channels must be 1 or 3, got {self.channels}")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        if not 0.0 <= self.max_occluder_transmission <= 1.0:
            raise ValueError("max_occluder_transmission must lie in [0, 1]")
        if not 0.0 <= self.identifiability <= 1.0:
            raise ValueError("identifiability must lie in [0, 1]")
        if not 0.0 <= self.staleness <= 1.0:
            raise ValueError("staleness must lie in [0, 1]")
        if not 0.0 < self.arrival_lo < self.arrival_hi < 1.0:
            raise ValueError("arrival_lo/arrival_hi must satisfy 0 < lo < hi < 1")
        if self.occlusion_schedule not in {"none", "linear", "quadratic", "step"}:
            raise ValueError(f"unknown occlusion_schedule: {self.occlusion_schedule!r}")
        if not 0.0 < self.cue_angle_grid_step < math.pi:
            raise ValueError("cue_angle_grid_step must lie in (0, pi)")


# --------------------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class MeasurementContext:
    """Declared context handed to the encoder; carries no geometry truth."""

    prior: PriorSpec
    task: str
    input_mode: str
    channel_names: tuple[str, ...]
    camera_id: str = "synthetic_measurement"
    frame_index_unit: str = "frame"
    timestamps_available: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "input_mode": self.input_mode,
            "channel_names": list(self.channel_names),
            "camera_id": self.camera_id,
            "frame_index_unit": self.frame_index_unit,
            "timestamps_available": self.timestamps_available,
        }
        payload.update(self.prior.context(self.task))
        return payload


@dataclass(slots=True)
class SyntheticEpisode:
    """One generated event sequence, with supervision kept separate from input.

    ``frames`` is ``[T, C, H, W]`` float32 in measurement units. ``metadata``
    holds sampling conditions and the declared context only; latent geometry,
    per-step stage truth and the cue schedule live in ``supervision``.
    ``prior_condition`` is an explicit field.

    Exact measurement identity for every step::

        frames[t] == attenuation_t * clean_t + noise_t

    where ``clean_t`` is rebuilt by :meth:`SyntheticGenerator.clean_measurements`.
    ``noise`` stores the *applied* term, occlusion included, so the identity holds
    term by term rather than only in expectation.
    """

    episode_id: str
    split: str
    frames: np.ndarray
    new_measurement: np.ndarray
    cue_strength: np.ndarray
    cue_angle: np.ndarray
    evidence_arrival_step: int
    identifiable: bool
    task: str
    value: str
    prior: PriorSpec
    context: MeasurementContext
    geometry_before: LatentScene
    geometry_after: LatentScene
    value_over_time: tuple[str, ...]
    geometry_changed: bool
    geometry_change_step: int
    frame_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_xy: np.ndarray
    occluder_mask: np.ndarray | None
    transmission: np.ndarray
    noise: np.ndarray
    supervision: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prior_condition(self) -> str:
        return self.prior.condition

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def latent(self) -> LatentScene:
        """Scene in force at the end of the sequence (the current truth)."""
        return self.geometry_after

    @property
    def unique_measurement_count(self) -> int:
        return int(self.new_measurement.sum())

    def scene_at(self, step: int) -> LatentScene:
        if self.geometry_changed and step >= self.geometry_change_step:
            return self.geometry_after
        return self.geometry_before

    def as_measurement_payload(self) -> dict[str, Any]:
        """Encoder-visible payload: measurements plus declared context only."""
        return {
            "values": self.frames,
            "new_measurement": self.new_measurement.copy(),
            "frame_ids": self.frame_ids,
            "sample_ids": self.sample_ids,
            "camera_id": self.context.camera_id,
            "time_unit": self.context.frame_index_unit,
            "timestamp": None,
            "episode_ids": tuple(self.episode_id for _ in range(self.num_frames)),
            "sample_xy": self.sample_xy,
            "context": self.context.as_dict(),
            "metadata": dict(self.metadata),
            "metadata_available": {
                "timestamp": False,
                "timestamps": False,
                "executed_action": False,
                "camera_transform": False,
                "proprio": False,
                "prior_value": self.prior.available,
                "prior_log_odds": self.prior.available,
                "cfa_pattern": self.context.input_mode == "bayer_cfa",
                "noise_model": True,
                "variance": True,
            },
            "variance": np.full(
                self.frames.shape[0], self.metadata["noise_variance"], dtype=np.float32
            ),
            "variance_is_estimate": False,
        }

    def frame_payload(self, step: int) -> dict[str, Any]:
        """One frame's encoder-visible payload (batch axis of one)."""
        payload = self.as_measurement_payload()
        payload["values"] = self.frames[step]
        payload["new_measurement"] = np.asarray([bool(self.new_measurement[step])])
        payload["frame_ids"] = (self.frame_ids[step],)
        payload["sample_ids"] = (self.sample_ids[step],)
        payload["episode_ids"] = (self.episode_id,)
        payload["variance"] = np.asarray([payload["variance"][step]])
        return payload


# --------------------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------------------


class SyntheticGenerator:
    """Deterministic measurement-domain episode generator.

    ``split`` selects disjoint shape, position and occlusion *values* for train
    and validation; :meth:`check_split_disjointness` asserts the separation.
    """

    def __init__(
        self,
        options: SyntheticOptions | None = None,
        *,
        seed: int = 0,
        split: str = "train",
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown split: {split!r}")
        self.options = options or SyntheticOptions()
        self.tasks = default_tasks()
        self.seed = int(seed)
        self.split = split

    # -- split geometry -----------------------------------------------------------------

    def centre_bounds(self, outer_radius: float) -> tuple[float, float]:
        """Inclusive ``(low, high)`` pixel range for the arc centre.

        Derived from the frame, not tuned: the arc centre must be at least
        ``orbit + 3 * cue_sigma + margin`` from every edge, otherwise the *other*
        hypothesis's marker would fall outside the sensor and the two hypotheses
        would stop being symmetric.
        """
        margin = cue_clearance(outer_radius, 1.0, 0.6)
        low = margin
        high = min(self.options.height, self.options.width) - margin
        if high <= low:
            raise ValueError(
                f"frame {self.options.height}x{self.options.width} is too small for "
                f"outer_radius={outer_radius}: need more than {2 * margin:.1f} pixels"
            )
        return float(low), float(high)

    def split_values(self, split: str | None = None) -> dict[str, tuple[Any, ...]]:
        """Disjoint shape/position/occlusion values for one split."""
        side = 0 if (split or self.split) == "train" else 1
        shapes = (
            ("handle_r2_w120", "handle_r3_w090"),
            ("handle_r3_w150", "handle_r4_w120"),
        )[side]
        positions = (
            (("center", 38.0, 40.0), ("offset", 32.0, 44.0)),
            (("leftedge", 30.0, 34.0), ("rightedge", 44.0, 48.0)),
        )[side]
        templates = (("left_half", "top_strip"), ("right_half", "shuffle"))[side]
        return {"shape_id": shapes, "position_id": positions, "occlusion_template": templates}

    def check_split_disjointness(self) -> dict[str, Any]:
        """Verify the declared split really separates every geometry value."""
        train = self.split_values("train")
        validation = self.split_values("validation")
        report: dict[str, Any] = {}
        for key in train:
            overlap = set(train[key]) & set(validation[key])
            report[key] = {"overlap": sorted(map(str, overlap)), "disjoint": not overlap}
        report["disjoint"] = all(
            item["disjoint"] for item in report.values() if isinstance(item, dict)
        )
        return report

    # -- latent sampling ----------------------------------------------------------------

    def _hash_unit(self, *parts: Any) -> float:
        digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def _mix(self, seed: int | None, *parts: Any) -> int:
        base = self.seed if seed is None else int(seed)
        digest = hashlib.sha256(":".join([str(base), *map(str, parts)]).encode()).digest()
        return int.from_bytes(digest[:4], "big")

    def sample_latent(
        self,
        task: str,
        value: str,
        split: str | None = None,
        *,
        seed: int | None = None,
        episode_index: int = 0,
    ) -> LatentScene:
        """Sample a latent scene from the split's own geometry values."""
        values = self.split_values(split)
        rng = np.random.default_rng(
            self._mix(seed, task, value, split or self.split, episode_index)
        )
        shape = str(values["shape_id"][int(rng.integers(0, len(values["shape_id"])))])
        inner_str, width_str = shape.replace("handle_r", "").split("_w")
        inner_radius = float(inner_str)
        outer_radius = inner_radius + 2.0
        # Width is authored in whole degrees so the identifier stays unambiguous
        # ("w12" would read as 12 degrees, not 120).
        angular_width = math.radians(float(width_str))
        if not math.radians(30.0) <= angular_width <= math.radians(300.0):
            raise ValueError(f"declared angular width out of range for {shape!r}")
        position_id, base_y, base_x = values["position_id"][
            int(rng.integers(0, len(values["position_id"])))
        ]
        low, high = self.centre_bounds(outer_radius)
        center = (
            float(np.clip(base_y + rng.uniform(-0.06, 0.06), low, high)),
            float(np.clip(base_x + rng.uniform(-0.06, 0.06), low, high)),
        )
        template = str(
            values["occlusion_template"][int(rng.integers(0, len(values["occlusion_template"])))]
        )
        return LatentScene(
            task=task,
            value=value,
            arc_center_xy=center,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            angular_width=angular_width,
            edge_softness=1.0,
            cue_radius=0.5 * (inner_radius + outer_radius),
            cue_sigma=0.6,
            # A fixed absolute arc angle: it is a scene fact, identical for both
            # hypotheses, which is what makes the arc genuinely ambiguous.
            arc_angle=math.pi * 0.75,
            shape_id=shape,
            position_id=str(position_id),
            occlusion_template=template,
        )

    # -- evidence and occlusion schedules ------------------------------------------------

    def _arrival_step(self, rng: np.random.Generator, num_frames: int) -> int:
        """Sample the evidence arrival step from a stretched Beta distribution."""
        options = self.options
        unit = float(rng.beta(options.arrival_shape_a, options.arrival_shape_b))
        scaled = options.arrival_lo + (options.arrival_hi - options.arrival_lo) * unit
        return int(np.clip(round(scaled * (num_frames - 1)), 1, max(num_frames - 2, 1)))

    def _cue_ramp(self, rng: np.random.Generator, num_frames: int, arrival: int) -> np.ndarray:
        """Per-frame cue strength, sampled independently of :meth:`_arrival_step`."""
        plateau = float(rng.uniform(0.55, 1.0))
        growth = float(rng.uniform(1.4, 3.2))
        ramp = np.zeros(num_frames, dtype=np.float64)
        for step in range(arrival, num_frames):
            ramp[step] = plateau * (1.0 - math.exp(-growth * (step - arrival + 1) / 3.0))
        return ramp

    def _occlusion_schedule(self, num_frames: int) -> np.ndarray:
        """Occluder attenuation per frame, decreasing as the sequence advances."""
        options = self.options
        if options.occlusion_schedule == "none":
            return np.ones(num_frames, dtype=np.float64)
        if options.occlusion_schedule == "linear":
            fraction = np.linspace(1.0, 0.0, num_frames)
        elif options.occlusion_schedule == "quadratic":
            fraction = np.linspace(1.0, 0.0, num_frames) ** 2
        else:  # "step"
            fraction = np.ones(num_frames)
            fraction[num_frames // 2 :] = 0.0
        return 1.0 - (1.0 - options.max_occluder_transmission) * (1.0 - fraction)

    def _occluder_mask(
        self,
        template: str,
        shape: tuple[int, int],
        rng: np.random.Generator,
        scene: LatentScene,
        extra_markers: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Occluder mask in ``[0, 1]``; 1.0 means a surface covers the photosite.

        Every declared template is *azimuthally smooth* around the arc centre and
        is placed away from the cue ring, so occlusion changes the overall
        luminance and texture but not the azimuthal cue signature. That is a
        property of the measurement operator, not a convenience: the ambiguous
        arc and the distinguishing cue must stay separable under occlusion, and
        :meth:`occlusion_excludes_cue_ring` asserts it numerically.

        ``extra_markers`` lets a caller protect additional marker positions, which
        a ``stale`` episode needs: the geometry change moves a marker, and the
        occluder is fixed for the whole sequence.

        ``"shuffle"`` is the validation-only template family, so the two families
        stay disjoint across the split.
        """
        rows = np.arange(shape[0], dtype=np.float64)[:, None]
        cols = np.arange(shape[1], dtype=np.float64)[None, :]
        center = np.asarray(scene.arc_center_xy, dtype=np.float64)
        distance = np.hypot(rows - center[0], cols - center[1])
        phi = np.arctan2(rows - center[0], cols - center[1])
        markers = dict(extra_markers) if extra_markers else {}
        for value in values_for_task(scene.task):
            markers.setdefault(value, marker_xy(scene, cue_angle_for_value(scene.task, value)))
        if template == "shuffle":
            # A soft fade perpendicular to the cue axis. It has no hard wedge edge,
            # and its radial floor keeps it off the cue orbit; the hard clearance
            # term below then removes anything that still comes near a marker.
            body = np.clip((cols - center[1]) / max(shape[1] * 0.5, 1.0e-6), 0.0, 1.0)
            mask = np.where(distance > scene.cue_orbit_radius, body, 0.0)
        else:
            directions = {
                "left_half": math.pi,
                "right_half": 0.0,
                "top_strip": 1.5 * math.pi,
                "bottom_strip": 0.5 * math.pi,
            }
            if template not in directions:
                raise ValueError(f"unknown occlusion template: {template!r}")
            # The wedge points away from both cue directions (0 and pi), which sit
            # 180 degrees apart, so it can never aim at a marker.
            guard = math.pi / 3.0
            on_cue_side = np.minimum(
                np.abs(wrapped_angle_difference(phi, 0.0)),
                np.abs(wrapped_angle_difference(phi, math.pi)),
            )
            body = np.clip((on_cue_side - guard) / 0.05, 0.0, 1.0)
            # Start beyond the farther marker's own distance plus 4 sigma; the
            # guard band then keeps the wedge's angular distance large enough that
            # the nearest occluder pixel is a full 4 sigma from that marker.
            safe = (
                max(float(np.hypot(*(point - center))) for point in markers.values())
                + 4.0 * scene.cue_sigma
            )
            mask = np.where(distance > safe, body, 0.0)
        # Hard guarantee, independent of the geometry above: no occluder pixel may
        # come within a marker's 3-sigma disk, so the distinguishing cue always
        # survives occlusion. Without this a later occluder could silently erase
        # the very signal the episode exists to deliver.
        for point in markers.values():
            # Strictly outside: a zero-distance sample would otherwise sit exactly
            # on the boundary and survive a `>=` comparison.
            outside = np.hypot(rows - point[0], cols - point[1]) > 3.0 * scene.cue_sigma
            mask = np.where(outside, mask, 0.0)
        return mask

    def markers_inside_frame(self, scene: LatentScene, margin: float = 0.0) -> dict[str, bool]:
        """Whether each hypothesis's marker fits inside the sensor (plus ``margin``)."""
        report: dict[str, bool] = {}
        for value in values_for_task(scene.task):
            marker = marker_xy(scene, cue_angle_for_value(scene.task, value))
            report[value] = bool(
                margin <= marker[0] <= self.options.height - 1 - margin
                and margin <= marker[1] <= self.options.width - 1 - margin
            )
        return report

    def validate_scene(
        self,
        scene: LatentScene,
        margin: float = 3.0,
        extra_markers: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Numeric checks that make the scene usable, run before any rendering.

        Asserts the two structural facts the mechanism experiment depends on:
        both hypotheses' markers fit inside the sensor, and every declared
        occluder template leaves the cue disk at or below the declared
        transparent level.
        """
        inside = self.markers_inside_frame(scene, margin)
        occluders = {
            template: self.occlusion_excludes_cue_ring(template, scene, extra_markers)
            for template in ("left_half", "right_half", "top_strip", "bottom_strip", "shuffle")
        }
        report = {
            "markers_inside_frame": inside,
            "occluders_clear_of_cue": occluders,
            "valid": all(inside.values()) and all(occluders.values()),
        }
        if not report["valid"]:
            raise ValueError(f"scene failed validation: {report}")
        return report

    def occlusion_excludes_cue_ring(
        self,
        template: str,
        scene: LatentScene,
        extra_markers: dict[str, np.ndarray] | None = None,
    ) -> bool:
        """Whether ``template`` leaves every marker's 3-sigma disk clear of occluders.

        The check is per marker disk, at each marker's own coordinates - not a ring
        around the arc centre - because a marker sits on the cue orbit at the cue
        angle, and the disk that must survive is the one around *that* point.
        """
        rng = np.random.default_rng(self._mix(None, scene.task, scene.value, template))
        mask = self._occluder_mask(
            template, (self.options.height, self.options.width), rng, scene, extra_markers
        )
        markers = dict(extra_markers) if extra_markers else {}
        for value in values_for_task(scene.task):
            markers.setdefault(value, marker_xy(scene, cue_angle_for_value(scene.task, value)))
        rows = np.arange(self.options.height, dtype=np.float64)[:, None]
        cols = np.arange(self.options.width, dtype=np.float64)[None, :]
        for marker in markers.values():
            disk = (
                np.hypot(
                    np.broadcast_to(rows, mask.shape) - marker[0],
                    np.broadcast_to(cols, mask.shape) - marker[1],
                )
                <= 3.0 * scene.cue_sigma
            )
            if float(mask[disk].max(initial=0.0)) > OCCLUDER_TRANSPARENT_LEVEL:
                return False
        return True

    # -- rendering ----------------------------------------------------------------------

    def _channel_frames_for_step(
        self,
        scene: LatentScene,
        step: int,
        cue_strength: float,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """One step's clean channel tensor ``[C, H, W]`` (no occlusion, no noise).

        In ``bayer_cfa`` mode each channel is one CFA plane, so ``H``/``W`` are
        ``height // 2`` / ``width // 2``: four planes of one sample per photosite,
        not four co-sited colours at full resolution.
        """
        options = self.options
        shape = (options.height, options.width)
        if options.input_mode == "rawvla_rgb3":
            _, _, clean = measurement_field(coordinates, scene, cue_strength=cue_strength)
            rendered = np.clip(clean, 0.0, options.white_level).reshape(shape)
            return np.repeat(rendered[None], options.channels, axis=0)
        plane_shape = (options.height // 2, options.width // 2)
        planes = np.empty((len(PLANE_COLOURS),) + plane_shape, dtype=np.float64)
        for plane_index, name in enumerate(PLANE_COLOURS):
            field = render_plane(
                coordinates,
                scene,
                cue_strength=cue_strength,
                base_value=options.base_value,
                plane=name,
            ).reshape(shape)
            values, _ = cfa_sample(field, options.cfa_pattern)
            planes[plane_index] = values[plane_index]
        del step
        return planes

    # -- episode generation --------------------------------------------------------------

    def generate(
        self,
        *,
        task: str = "opening_direction",
        prior_condition: str = "correct",
        num_frames: int = 8,
        episode_index: int = 0,
        split: str | None = None,
        identifiable: bool | None = None,
        value: str | None = None,
        prior_value: str | None = None,
        seed: int | None = None,
    ) -> SyntheticEpisode:
        """Generate one event sequence with its supervision."""
        if num_frames < 4:
            raise ValueError("num_frames must be at least 4 for a non-degenerate schedule")
        side = split or self.split
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}; have {list(self.tasks)}")
        if prior_condition not in PRIOR_CONDITIONS:
            raise ValueError(f"unknown prior condition: {prior_condition!r}")
        task_spec = self.tasks[task]
        # The latent value and the prior value are drawn from their own streams so
        # a caller asking for one value never gets a different prior than a caller
        # asking for the other. Everything else keeps the shared stream.
        latent_rng = np.random.default_rng(
            self._mix(seed, task, prior_condition, episode_index, side, num_frames)
        )
        rng = np.random.default_rng(
            self._mix(seed, "schedule", task, prior_condition, episode_index, side, num_frames)
        )
        if identifiable is None:
            identifiable = bool(latent_rng.random() < self.options.identifiability)
        if value is None:
            value = task_spec.values[int(latent_rng.integers(0, 2))]
        if prior_value is None:
            prior_value = value
        changed = bool(latent_rng.random() < self.options.staleness)
        # A stale prior describes the scene BEFORE the change, not its inverse.
        true_value = value
        geometry_before = self.sample_latent(
            task, true_value, side, seed=seed, episode_index=episode_index
        )
        arrival = self._arrival_step(rng, num_frames)
        geometry_change_step = arrival if (prior_condition == "stale" and changed) else -1
        geometry_after = (
            geometry_before.with_value(task_spec.values[1 - task_spec.values.index(true_value)])
            if geometry_change_step >= 0
            else geometry_before
        )
        prior_log_odds = {
            "correct": float(rng.uniform(0.6, 1.6)),
            "misleading": -float(rng.uniform(0.6, 1.6)),
            "uninformative": 0.0,
            "missing": MISSING_PRIOR_LOG_ODDS,
            "stale": float(rng.uniform(0.6, 1.6)),
        }[prior_condition]
        prior = PriorSpec(condition=prior_condition, value=prior_value, log_odds=prior_log_odds)

        cue_strength = self._cue_ramp(rng, num_frames, arrival)
        if not identifiable:
            cue_strength = np.zeros(num_frames, dtype=np.float64)
        cue_angle = np.array(
            [cue_angle_for_value(task, geometry_before.value)] * num_frames, dtype=np.float64
        )
        transmission = self._occlusion_schedule(num_frames)
        template = geometry_before.occlusion_template
        extra_markers: dict[str, np.ndarray] = {}
        if geometry_change_step >= 0:
            # A `stale` flip moves a marker, and the occluder is fixed for the whole
            # sequence, so both the pre- and post-change markers must be protected.
            for value in values_for_task(task_spec.name):
                extra_markers[f"after:{value}"] = marker_xy(
                    geometry_after, cue_angle_for_value(task_spec.name, value)
                )
        occluder_mask = (
            None
            if self.options.occlusion_schedule == "none"
            else self._occluder_mask(
                template,
                (self.options.height, self.options.width),
                rng,
                geometry_before,
                extra_markers or None,
            )
        )
        if self.options.input_mode == "bayer_cfa":
            sensor_shape = (
                self._num_channels(),
                self.options.height // 2,
                self.options.width // 2,
            )
        else:
            sensor_shape = (
                self._num_channels(),
                self.options.height,
                self.options.width,
            )
        unit_noise = rng.standard_normal((num_frames,) + sensor_shape)

        coordinates, _ = sample_coordinates(
            self.options.height,
            self.options.width,
            cfa_pattern=(
                self.options.cfa_pattern if self.options.input_mode == "bayer_cfa" else ""
            ),
            point_sampling=self.options.point_sampling,
        )
        frames = np.zeros((num_frames,) + sensor_shape, dtype=np.float64)
        applied_noise = np.zeros_like(frames)
        for step in range(num_frames):
            changed_over = geometry_change_step >= 0 and step >= geometry_change_step
            scene = geometry_after if changed_over else geometry_before
            clean = self._channel_frames_for_step(
                scene, step, float(cue_strength[step]), coordinates
            )
            attenuation = (
                np.full((self.options.height, self.options.width), float(transmission[step]))
                if occluder_mask is None
                else occluder_mask * transmission[step]
            )
            # ``attenuation`` is [H, W] and ``clean`` is [C, H, W]; the explicit
            # leading axis is required, because numpy would otherwise treat
            # [H, W] * [C, H, W] as a contraction over the channel axis. In CFA
            # mode each channel is one sub-sampled plane, so the occluder is read
            # at that plane's own photosites rather than at mosaic resolution.
            if clean.shape[1:] == attenuation.shape:
                per_channel = attenuation[None, :, :]
            else:
                # CFA mode: read the occluder at each plane's own photosites.
                per_channel = cfa_sample(attenuation, self.options.cfa_pattern)[0]
            broadcast = per_channel.astype(np.float64)
            applied_noise[step] = broadcast * (self.options.noise_sigma * unit_noise[step])
            frames[step] = broadcast * clean + applied_noise[step]
        return self._assemble(
            task=task,
            task_spec=task_spec,
            prior=prior,
            split=side,
            episode_index=episode_index,
            num_frames=num_frames,
            arrival=arrival,
            cue_strength=cue_strength,
            cue_angle=cue_angle,
            transmission=transmission,
            occluder_mask=occluder_mask,
            frames=frames,
            noise=applied_noise,
            prior_value=prior_value,
            changed=changed,
            geometry_change_step=geometry_change_step,
            geometry_before=geometry_before,
            geometry_after=geometry_after,
            identifiable=bool(identifiable),
        )

    def _num_channels(self) -> int:
        if self.options.input_mode == "bayer_cfa":
            return len(PLANE_COLOURS)
        return self.options.channels

    def _assemble(
        self,
        *,
        task: str,
        task_spec: TaskSpec,
        prior: PriorSpec,
        split: str,
        episode_index: int,
        num_frames: int,
        arrival: int,
        cue_strength: np.ndarray,
        cue_angle: np.ndarray,
        transmission: np.ndarray,
        occluder_mask: np.ndarray | None,
        frames: np.ndarray,
        noise: np.ndarray,
        prior_value: str,
        changed: bool,
        geometry_change_step: int,
        geometry_before: LatentScene,
        geometry_after: LatentScene,
        identifiable: bool,
    ) -> SyntheticEpisode:
        options = self.options
        cfa = options.input_mode == "bayer_cfa"
        episode_id = (
            f"{split}-{geometry_before.shape_id}-{geometry_before.position_id}-"
            f"{geometry_before.occlusion_template}-{task_spec.name}-{prior.condition}-"
            f"{episode_index:04d}"
        )
        channel_names = (
            PLANE_COLOURS if cfa else tuple(f"ch{index}" for index in range(options.channels))
        )
        context = MeasurementContext(
            prior=prior,
            task=task_spec.name,
            input_mode=options.input_mode,
            channel_names=channel_names,
        )
        if cfa:
            positions = cfa_plane_positions(options.cfa_pattern)
            sample_xy = np.zeros((4, SAMPLE_XY_ROWS, 2), dtype=np.float32)
            sample_xy[:, 0, :] = positions.astype(np.float32)
            sample_xy[:, 1, :] = np.float32(2.0)
        else:
            sample_xy = np.zeros((options.channels, SAMPLE_XY_ROWS, 2), dtype=np.float32)
            sample_xy[:, 0, :] = np.float32(0.5)
            sample_xy[:, 1, :] = np.float32(1.0)
        frame_ids = tuple(f"{episode_id}#frame{step:03d}" for step in range(num_frames))
        sample_ids = tuple(
            f"{episode_id}#ch{channel}@frame{step:03d}"
            for step in range(num_frames)
            for channel in range(frames.shape[1])
        )
        change = geometry_change_step if (prior.condition == "stale" and changed) else -1
        value_over_time = tuple(
            geometry_before.value if (change < 0 or step < change) else geometry_after.value
            for step in range(num_frames)
        )
        supervision = {
            "task": task_spec.name,
            "value": geometry_after.value if change < 0 else geometry_before.value,
            "value_over_time": list(value_over_time),
            "geometry_before": geometry_before.as_supervision(),
            "geometry_after": geometry_after.as_supervision(),
            "geometry_changed": change >= 0,
            "geometry_change_step": change,
            "prior_value": prior_value,
            "prior_is_correct": (
                None
                if not prior.available or prior.log_odds == 0
                else (
                    prior_value
                    if prior.log_odds > 0
                    else values_for_task(task_spec.name)[
                        1 - values_for_task(task_spec.name).index(prior_value)
                    ]
                )
                == value_over_time[-1]
            ),
            "prior_log_odds": prior.log_odds,
            "cue_angle_per_step": cue_angle.tolist(),
            "cue_angle_true": cue_angle_for_value(task_spec.name, geometry_before.value),
            "evidence_arrival_step": arrival,
            "identifiable": identifiable,
            "task_values": list(task_spec.values),
        }
        metadata = {
            "generator": "owac.frameworks.n02.synthetic",
            "seed": self.seed,
            "split": split,
            "input_mode": options.input_mode,
            "cfa_pattern": options.cfa_pattern if cfa else "",
            "channel_names": list(channel_names),
            "height": options.height,
            "width": options.width,
            "base_value": options.base_value,
            "white_level": options.white_level,
            "noise_sigma": options.noise_sigma,
            "noise_variance": float(options.noise_sigma**2),
            "noise_model_declared": True,
            "variance_expression": "variance = noise_sigma**2 (additive Gaussian, same units)",
            "measurement_law": "frames = attenuation * clean + noise",
            "occluder_mask_available": occluder_mask is not None,
            "point_sampling": options.point_sampling,
            "occlusion_schedule": options.occlusion_schedule,
            "occlusion_template": geometry_before.occlusion_template,
            "max_occluder_transmission": options.max_occluder_transmission,
            "pixel_centre_convention": (
                "centre of sensor pixel (row, col) is at (row + 0.5, col + 0.5)"
            ),
            "planes_are_not_co_sited": bool(cfa),
            "rgb_image_output": False,
            "rgb_teacher": False,
            "rgb_reconstruction_target": False,
            "measurement_domain_only": True,
            "timestamps_available": False,
            "executed_action_available": False,
        }
        return SyntheticEpisode(
            episode_id=episode_id,
            split=split,
            frames=frames.astype(np.float32),
            new_measurement=np.ones(num_frames, dtype=bool),
            cue_strength=cue_strength,
            cue_angle=cue_angle,
            evidence_arrival_step=arrival,
            identifiable=identifiable,
            task=task_spec.name,
            value=supervision["value"],
            prior=prior,
            context=context,
            geometry_before=geometry_before,
            geometry_after=geometry_after,
            value_over_time=value_over_time,
            geometry_changed=change >= 0,
            geometry_change_step=change,
            frame_ids=frame_ids,
            sample_ids=sample_ids,
            sample_xy=sample_xy,
            occluder_mask=occluder_mask,
            transmission=transmission,
            noise=noise,
            supervision=supervision,
            metadata=metadata,
        )

    # -- verification helpers ------------------------------------------------------------

    def clean_measurements(self, episode: SyntheticEpisode) -> np.ndarray:
        """Recompute the forward operator for ``episode`` (no occlusion, no noise).

        The returned ``[T, C, H, W]`` array is the *signal before attenuation*;
        :attr:`SyntheticEpisode.occluder_mask` (absent when the schedule is
        ``"none"``), :attr:`~SyntheticEpisode.transmission` and
        :attr:`~SyntheticEpisode.noise` complete the measurement law, so a test can
        assert ``frames == attenuation * clean + noise`` exactly.
        """
        options = self.options
        coordinates, _ = sample_coordinates(
            options.height,
            options.width,
            cfa_pattern=options.cfa_pattern if options.input_mode == "bayer_cfa" else "",
            point_sampling=options.point_sampling,
        )
        steps = []
        for step in range(episode.num_frames):
            steps.append(
                self._channel_frames_for_step(
                    episode.scene_at(step),
                    step,
                    float(episode.cue_strength[step]),
                    coordinates,
                )
            )
        return np.stack(steps, axis=0)

    def hypothesis_fit(
        self, episode: SyntheticEpisode, step: int | None = None
    ) -> dict[str, dict[str, float]]:
        """Fit each hypothesis's declared measurement template to the frame.

        The template is the *complete* forward operator for that hypothesis - the
        ambiguous arc plus the cue marker at that hypothesis's declared cue angle,
        at the frame's own cue strength, through the frame's own occluder - so the
        reported residual is a real prediction error and not a proxy.

        Returns per hypothesis the mean squared residual, the correlation and the
        decisive residual restricted to the other hypothesis's marker disk. The
        decisive residual is the readable one: an incorrect hypothesis predicts a
        marker where the frame is empty, so it pays for it there and nowhere else.
        The latent value is never consulted.
        """
        scene = episode.geometry_before
        step = int(np.argmax(episode.cue_strength)) if step is None else int(step)
        strength = float(episode.cue_strength[step])
        if strength <= 0.0:
            raise ValueError("selected frame carries no cue; the episode is unidentifiable")
        frame = np.asarray(episode.frames[step], dtype=np.float64).mean(axis=0)
        attenuation = (
            episode.transmission[step]
            if episode.occluder_mask is None
            else episode.occluder_mask * episode.transmission[step]
        )
        coordinates, grid = sample_coordinates(
            self.options.height,
            self.options.width,
            cfa_pattern=self.options.cfa_pattern if self.options.input_mode == "bayer_cfa" else "",
            point_sampling=self.options.point_sampling,
        )
        values = values_for_task(scene.task)
        flat_frame = frame.ravel()
        reports: dict[str, dict[str, float]] = {}
        for value in values:
            angle = cue_angle_for_value(scene.task, value)
            _, _, clean = measurement_field(
                coordinates, scene, cue_strength=strength, cue_angle=angle
            )
            prediction = (attenuation * clean.reshape(grid.shape[:-1])).ravel()
            residual = flat_frame - prediction
            marker = marker_xy(scene, angle)
            disk = np.hypot(coordinates[:, 0] - marker[0], coordinates[:, 1] - marker[1]) <= 3.0 * (
                scene.cue_sigma
            )
            other = marker_xy(
                scene, cue_angle_for_value(scene.task, values[1 - values.index(value)])
            )
            other_disk = (
                np.hypot(coordinates[:, 0] - other[0], coordinates[:, 1] - other[1])
                <= 3.0 * scene.cue_sigma
            )
            reports[value] = {
                "mean_squared_residual": float(np.mean(residual**2)),
                "decisive_residual": float(np.mean(residual[other_disk] ** 2))
                if other_disk.any()
                else float("nan"),
                "own_marker_residual": float(np.mean(residual[disk] ** 2))
                if disk.any()
                else float("nan"),
                "cue_angle": angle,
                "marker_yx": [float(marker[0]), float(marker[1])],
            }
        return reports

    def infer_hypothesis(self, episode: SyntheticEpisode, step: int | None = None) -> str:
        """Latent value selected by the measurement-only template residual."""
        fits = self.hypothesis_fit(episode, step)
        return min(fits, key=lambda key: fits[key]["mean_squared_residual"])

    def infer_cue_angle(self, episode: SyntheticEpisode, step: int | None = None) -> float:
        """Cue angle recovered from the measurement alone, in ``[0, 2*pi)``.

        Returns the declared cue angle of the best-fitting hypothesis, so agreement
        with :func:`cue_angle_for_value` is a statement about the renderer: only a
        frame whose signal really sits at that hypothesis's marker can win.
        """
        return float(
            self.hypothesis_fit(episode, step)[self.infer_hypothesis(episode, step)]["cue_angle"]
            % (2.0 * math.pi)
        )

    def verify_measurement(
        self, episode: SyntheticEpisode, step: int | None = None
    ) -> dict[str, Any]:
        """Independent check that the frame sits where the declared operator says.

        Measures the frame excess inside each hypothesis's marker disk (frame minus
        the arc-only prediction, averaged over the disk). The true hypothesis must
        show the larger excess, and for a noiseless frame the false one must show
        none. This uses only measurements and declared geometry facts.
        """
        scene = episode.geometry_before
        step = int(np.argmax(episode.cue_strength)) if step is None else int(step)
        strength = float(episode.cue_strength[step])
        if strength <= 0.0:
            raise ValueError("selected frame carries no cue; the episode is unidentifiable")
        frame = np.asarray(episode.frames[step], dtype=np.float64).mean(axis=0)
        attenuation = (
            episode.transmission[step]
            if episode.occluder_mask is None
            else episode.occluder_mask * episode.transmission[step]
        )
        coordinates, grid = sample_coordinates(
            self.options.height,
            self.options.width,
            cfa_pattern=self.options.cfa_pattern if self.options.input_mode == "bayer_cfa" else "",
            point_sampling=self.options.point_sampling,
        )
        arc_only = measurement_field(coordinates, scene, cue_strength=0.0)[0].reshape(
            grid.shape[:-1]
        )
        excess = (frame - attenuation * arc_only).ravel()
        report: dict[str, Any] = {"strength": strength, "excess": {}}
        for value in values_for_task(scene.task):
            marker = marker_xy(scene, cue_angle_for_value(scene.task, value))
            disk = np.hypot(coordinates[:, 0] - marker[0], coordinates[:, 1] - marker[1]) <= (
                3.0 * scene.cue_sigma
            )
            report["excess"][value] = float(np.mean(excess[disk])) if disk.any() else float("nan")
        report["true_value"] = scene.value
        report["argmax_excess"] = max(report["excess"], key=lambda key: report["excess"][key])
        report["agrees"] = report["argmax_excess"] == scene.value
        return report

    def cue_angle_from_measurements(
        self, episode: SyntheticEpisode, step: int | None = None
    ) -> float:
        """Alias of :meth:`infer_cue_angle` for the noiseless-verification use case."""
        return self.infer_cue_angle(episode, step)

    def _sample_coordinates_for_estimation(self) -> np.ndarray:
        coordinates, _ = sample_coordinates(
            self.options.height,
            self.options.width,
            cfa_pattern=self.options.cfa_pattern if self.options.input_mode == "bayer_cfa" else "",
            point_sampling=self.options.point_sampling,
        )
        if self.options.input_mode == "bayer_cfa":
            # The CFA frames are plane-stacked, so the estimator uses the plane
            # lattice each plane actually sampled rather than the mosaic lattice.
            positions = cfa_plane_positions(self.options.cfa_pattern)
            plane_rows = np.arange(self.options.height // 2) * 2.0
            plane_cols = np.arange(self.options.width // 2) * 2.0
            rows = positions[:, 0, None, None] + plane_rows[None, :, None]
            cols = positions[:, 1, None, None] + plane_cols[None, None, :]
            stacked = np.stack(np.broadcast_arrays(rows, cols), axis=-1)
            return stacked.reshape(-1, 2)
        return coordinates


# --------------------------------------------------------------------------------------
# exact two-hypothesis mode
# --------------------------------------------------------------------------------------


class ExplicitObservationModel:
    """One-dimensional two-hypothesis model with an exact, hand-checkable update.

    The observation is a cue angle ``y`` in radians::

        H0: y = theta0 + N(0, sigma^2 / amplitude)
        H1: y = theta1 + N(0, sigma^2 / amplitude),   theta1 = theta0 + separation

    ``amplitude`` is the per-observation evidence strength in ``[0, 1]``; at
    ``amplitude == 0`` the observation is infinitely noisy and carries no
    information. With the wrapped residual ``r = wrap(y - theta0)`` the
    log-likelihood ratio is::

        LLR(y) = amplitude * separation * (2*r - separation) / (2 * sigma^2)

    so :meth:`accumulate` is exact up to float64 rounding and can be compared with
    :meth:`exact_products`, which multiplies the raw Gaussian likelihoods.
    """

    def __init__(
        self,
        *,
        theta0: float = 0.0,
        separation: float = math.pi,
        noise_sigma: float = 0.20,
        amplitude: float = 1.0,
        prior_log_odds: float = 0.0,
    ) -> None:
        if noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive")
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError("amplitude must lie in [0, 1]")
        if not 0.0 < separation < math.pi:
            raise ValueError(
                "separation must lie strictly in (0, pi): at separation == pi the two "
                "means are antipodal, so a caller who intended circular semantics would "
                "get an ill-posed real-line model (y and y - 2*pi would be the same "
                "observation). Use a circular (von Mises) formulation for antipodal "
                "hypotheses."
            )
        self.theta0 = float(theta0)
        self.separation = float(separation)
        self.noise_sigma = float(noise_sigma)
        self.amplitude = float(amplitude)
        self.prior_log_odds = float(prior_log_odds)

    @property
    def variance(self) -> float:
        """Effective observation variance ``sigma^2 / amplitude`` (``inf`` at zero)."""
        if self.amplitude == 0.0:
            return math.inf
        return self.noise_sigma**2 / self.amplitude

    def means(self) -> tuple[float, float]:
        return self.theta0, self.theta0 + self.separation

    def mean_for(self, hypothesis: int) -> float:
        if hypothesis not in (0, 1):
            raise ValueError(f"hypothesis must be 0 or 1, got {hypothesis!r}")
        return self.means()[hypothesis]

    def log_likelihood(self, y: float, hypothesis: int) -> float:
        """``log N(y; mean_h, sigma^2 / amplitude)`` in the unwrapped window."""
        variance = self.variance
        if math.isinf(variance):
            return 0.0
        return gaussian_logpdf(float(y) - self.mean_for(hypothesis), 0.0, variance)

    def log_likelihood_ratio(self, y: float) -> float:
        variance = self.variance
        if math.isinf(variance):
            return 0.0
        residual = float(y) - self.theta0
        # NOTE: no extra amplitude factor. The effective variance is already
        # sigma^2 / amplitude, so the Gaussian difference reduces exactly to
        # separation * (2 * residual - separation) / (2 * variance).
        return self.separation * (2.0 * residual - self.separation) / (2.0 * variance)

    def sample(self, rng: np.random.Generator, hypothesis: int) -> float:
        """One observation from ``hypothesis`` (deterministic midpoint at zero amplitude)."""
        if self.amplitude == 0.0:
            return float(self.theta0 + 0.5 * self.separation)
        # Plain scalar Gaussian: the value stays on the real line, as the declared
        # model says. No wrapping, so the sample mean of the log-odds converges to
        # the exact `steps * separation**2 / (2 * variance)`.
        return float(
            rng.normal(self.mean_for(hypothesis), self.noise_sigma / math.sqrt(self.amplitude))
        )

    def accumulate(self, observations: Any) -> dict[str, Any]:
        """Sequential Bayes update; returns posterior, trajectory and per-step LLRs."""
        log_odds = self.prior_log_odds
        trajectory = [log_odds]
        ratios = []
        for y in np.asarray(observations, dtype=np.float64).ravel():
            ratio = self.log_likelihood_ratio(float(y))
            ratios.append(ratio)
            log_odds += ratio
            trajectory.append(log_odds)
        return {
            "prior_log_odds": self.prior_log_odds,
            "log_odds": log_odds,
            "posterior": _sigmoid(log_odds),
            "log_odds_trajectory": trajectory,
            "log_likelihood_ratios": ratios,
            "variance": self.variance,
            "amplitude": self.amplitude,
        }

    def exact_products(self, observations: Any) -> dict[str, Any]:
        """The same posterior from the raw per-observation likelihood products."""
        log_odds = self.prior_log_odds
        for y in np.asarray(observations, dtype=np.float64).ravel():
            log_odds += self.log_likelihood(float(y), 1) - self.log_likelihood(float(y), 0)
        return {"log_odds": log_odds, "posterior": _sigmoid(log_odds)}

    def expected_information(self) -> float:
        """Expected LLR magnitude per observation (Fisher information for the mean)."""
        if math.isinf(self.variance):
            return 0.0
        return self.separation**2 / (2.0 * self.variance)

    def exact_update(
        self,
        rng: np.random.Generator,
        hypothesis: int,
        steps: int,
    ) -> dict[str, Any]:
        """Sample from ``hypothesis`` and run the exact update, returning both views."""
        observations = np.array(
            [self.sample(rng, hypothesis) for _ in range(int(steps))], dtype=np.float64
        )
        accumulated = self.accumulate(observations)
        reference = self.exact_products(observations)
        accumulated.update(
            {
                "hypothesis": hypothesis,
                "observations": observations,
                "reference_log_odds": reference["log_odds"],
                "reference_posterior": reference["posterior"],
                "absolute_difference": abs(accumulated["log_odds"] - reference["log_odds"]),
            }
        )
        return accumulated


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def gaussian_logpdf(value: float, mean: float, variance: float) -> float:
    """``log N(value; mean, variance)`` including the normalising constant."""
    if variance <= 0:
        raise ValueError("variance must be positive")
    return -0.5 * ((value - mean) ** 2 / variance + math.log(2.0 * math.pi * variance))


def log_likelihood_ratio(model: ExplicitObservationModel, y: float) -> float:
    """Module-level wrapper around :meth:`ExplicitObservationModel.log_likelihood_ratio`."""
    return model.log_likelihood_ratio(y)

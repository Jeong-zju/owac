"""RawVLA NPZ window loader producing *measurements* instead of RGB images.

Purpose
-------
The N02 (and sibling) encoder prototypes need real RawVLA replay windows as
measurements with explicit sampling conditions, not as PIL images with an RGB
chroma target. This module re-implements the window/action indexing and the
normalisation semantics of the author's pinned loader

    starVLA/dataloader/rawvla_npz_datasets.py
    commit 57d508cb180d0510ac9060358009c0f6ffaac420
    sha256 9d5d3b2034e46b6c27609b7719ce53f802058735e2d01a27fb93ee257dcc1be8

(local copy: ``third_party/reference-inspect/rawvla_npz_datasets.py``; that hash
was verified locally against ``docs/prompts/reference_sources.json``) while
deliberately *not* inheriting:

* ``PIL.Image.fromarray`` / the ``image`` layout output,
* the paired RGB episode read and the ``rawvla_chroma_target_float32`` target,
* the torch ``Dataset`` base class.

No RGB image, RGB teacher or RGB reconstruction is produced anywhere in this
module. Measurement tensors are float32 (or the original integer dtype before
conversion) and their units are declared per sample.

Source data availability (checked 2026-10-12 - do not soften this claim)
-----------------------------------------------------------------------
The real RawVLA-Bench NPZ caches are **not** present on this workstation. The
checks that were actually run:

* ``ls -la /workspace`` -> ``No such file or directory`` (the deployed server's
  ``/workspace`` is a container overlay on *that* host, not this one),
* a bounded ``find /home/jeong -maxdepth 6 -name '*.npz'`` returned only
  unrelated NPZ files from other projects; nothing under this checkout's
  ``data/`` (it holds only ``README.md`` and ``manifests/README.md``).

The data lives on the deployed server recorded in
``research/experiments/2026-10-10_rawvla_pi05_deployment.md``: server working
directory ``/workspace/owac/rawvla`` (field ``root`` of
``research/experiments/results/2026-10-10_rawvla_pi05_deployment/service-check.json``),
holding the pinned HuggingFace revision
``8586c594fff9ce38580e29745f074e82e96e74f7`` of ``ToferFish/RawVLA-Bench``.
Relative to that root, the files actually recorded in
``.../results/2026-10-10_rawvla_pi05_deployment/dataset-check.json`` are::

    data/libero/raw/<suite>_lerobot/episode_%06d/variant_%02d.npz   # 1771 files
    data/libero/rgb/<suite>_lerobot/episode_%06d/variant_%02d.npz   # paired RGB
    data/robotwin2/raw/<task>/episode_%06d.npz                      # 632 files
    data/robotwin2/rgb/<task>/episode_%06d.npz                      # paired RGB

So on that server the loader is driven with
``--cache-root /workspace/owac/rawvla/data/libero/raw`` (LIBERO) or
``--cache-root /workspace/owac/rawvla/data/robotwin2/raw`` (RoboTwin). A
*file-level* cache root matters because the HuggingFace snapshot also contains
the paired RGB trees, which this loader must not read; discovery still accepts
the snapshot root itself if that is all a caller has.

Dtype/shape/range evidence (server-side read-back recorded in
``dataset-check.json``, *not* a local re-measurement here). LIBERO
``agentview_raw_uint8`` / ``wrist_raw_uint8`` are ``[63, 256, 256, 3]`` uint8
with min 0 / max 8, ``robot_states`` ``[63, 8]`` float32, ``actions`` ``[63, 7]``
float32 in ``[-1, 1]``. RoboTwin ``head/left/right_camera_raw_float32`` are
``[88, 240, 320, 3]`` float32 with maxima 0.4332 / 0.3744 / 0.3738,
``actions`` ``[88, 14]`` float32 in ``[-2.2486, 2.2652]``, ``robot_states``
``[88, 16]`` float32 (an end-effector representation, *not* the 14-dim joint
state). Two consequences are enforced below: the LIBERO max of 8 in uint8 is not
scaled to 255, and the RoboTwin float32 maxima near 0.43 are neither divided by
255 nor per-frame max-normalised.

What this module refuses to invent
----------------------------------
* No timestamps. The caches carry no time axis, so ``time_unit="frame"``,
  ``timestamp=None`` and ``metadata_available["timestamp"] is False``.
* No executed actions. A replay cache stores the expert command stream for
  supervision; it is not a record of what the robot actually executed. Every
  sample reports ``executed_action_available=False`` and the action block is
  exposed only as *supervision*, never as executed-action context.
* No Bayer CFA for the shipped data (``input_mode="rawvla_rgb3"``). The CFA path
  exists for a genuinely mosaic input and declares its pattern; re-mosaicking
  the three-channel pseudo-RAW is never done here.
* No fabricated statistics: normalisation statistics are read from a declared
  stats file whose sha256 is recorded, otherwise the sample is returned
  unnormalised with ``normalisation == "none"``.
* No fabricated variance: without a declared noise model the payload carries
  ``variance = None`` plus a *fitted* scale explicitly marked as an estimate.

Missing, bad and saturated are three separate masks and stay separate.

Usage
-----
::

    uv run --locked --extra encoder python -m owac.raw.rawvla_sequence \\
        --cache-root /workspace/owac/rawvla/data/libero/raw --format libero \\
        --report outputs/n02-data-report.json

The CLI writes a small JSON report (episode/frame counts, dtype and range
evidence, split sizes, missing assets). With no cache present it reports
``"status": "no_data_found"`` and invents nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CFA_PATTERNS",
    "FORMAT_SPECS",
    "AcquisitionSource",
    "CfaCalibration",
    "CfaPlanes",
    "EpisodeRef",
    "FormatSpec",
    "NoiseModel",
    "NormalisationStats",
    "RawVLASample",
    "RawVLAWindowDataset",
    "SplitManifest",
    "WindowOptions",
    "apply_cfa_calibration",
    "build_measurement_frame",
    "build_report",
    "cfa_plane_positions",
    "cfa_plane_values",
    "compute_split",
    "describe_frame_sampling",
    "discover_npz_files",
    "fitted_variance_estimate",
    "inspect_cache",
    "load_split_manifest",
    "main",
    "normalise_actions",
    "normalise_state",
    "sensor_variance",
    "write_split_manifest",
]

#: CFA colour order for a 2x2 Bayer tile, stored as ``((0,0), (0,1), (1,0), (1,1))``.
CFA_PATTERNS: dict[str, tuple[str, str, str, str]] = {
    "RGGB": ("R", "G", "G", "B"),
    "BGGR": ("B", "G", "G", "R"),
    "GRBG": ("G", "R", "B", "G"),
    "GBRG": ("G", "B", "R", "G"),
}

#: Plane order used by the 2x2 rearrangement.
PLANE_COLOURS: tuple[str, str, str, str] = ("R", "G1", "G2", "B")

#: ``(dy, dx)`` of each plane inside the 2x2 tile.
PLANE_TILE_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

#: Pixel-centre convention, quoted in every CFA payload.
PIXEL_CENTRE_CONVENTION = "centre of sensor pixel (row, col) is at (row + 0.5, col + 0.5)"


# --------------------------------------------------------------------------------------
# format specifications
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Declared layout of one RawVLA NPZ dataset format.

    Every field mirrors one decision of the author loader; nothing is inferred at
    runtime, so a configuration can be diffed against the reference.
    """

    name: str
    measurement_keys: tuple[str, ...]
    camera_ids: tuple[str, ...]
    input_mode: str
    state_source_key: str
    state_source_is_current: bool
    state_is_joint_vector: bool
    state_notes: str
    default_observation_stride: int
    default_action_horizon: int
    default_action_offset: int
    default_action_normalisation: str
    default_state_normalisation: str
    #: Search prefixes tried in order; the first non-empty match wins.
    markers: tuple[str, ...]
    #: Declared channel count of each cached measurement array. Three means the
    #: author's co-sited pseudo-RAW / HDR-derived RAW; one means a real mosaic.
    measurement_channels: int = 3
    rgb_keys: tuple[str, ...] = ()


FORMAT_SPECS: dict[str, FormatSpec] = {
    "libero": FormatSpec(
        name="libero",
        measurement_keys=("agentview_raw_uint8", "wrist_raw_uint8"),
        camera_ids=("libero_agentview", "libero_wrist"),
        input_mode="rawvla_rgb3",
        state_source_key="robot_states",
        state_source_is_current=True,
        state_is_joint_vector=True,
        state_notes=(
            "LIBERO robot_states[t] is the 8-dim proprio read at the same step as the "
            "observation. The cached 92-dim `states` array is simulator state and is never "
            "used as a policy input."
        ),
        default_observation_stride=1,
        default_action_horizon=10,
        default_action_offset=0,
        default_action_normalisation="quantile",
        default_state_normalisation="quantile",
        markers=("data/libero/raw", "libero/raw", "raw"),
        rgb_keys=("agentview_rgb_uint8", "wrist_rgb_uint8"),
    ),
    "robotwin": FormatSpec(
        name="robotwin",
        measurement_keys=(
            "head_camera_raw_float32",
            "left_camera_raw_float32",
            "right_camera_raw_float32",
        ),
        camera_ids=("robotwin_head", "robotwin_left", "robotwin_right"),
        input_mode="rawvla_rgb3",
        state_source_key="actions",
        state_source_is_current=True,
        state_is_joint_vector=True,
        state_notes=(
            "RoboTwin replay caches have no separate current-state array. The author's "
            "protocol (and this loader) therefore treats actions[t] as the current 14-dim "
            "joint vector and supervises from actions[t+1 : t+1+horizon]. The cached 16-dim "
            "`robot_states` is an end-effector representation, not the joint state."
        ),
        default_observation_stride=1,
        default_action_horizon=50,
        default_action_offset=1,
        default_action_normalisation="starvla_minmax",
        default_state_normalisation="none",
        markers=("data/robotwin2/raw", "robotwin2/raw", "raw"),
        rgb_keys=(
            "head_camera_rgb_uint8",
            "left_camera_rgb_uint8",
            "right_camera_rgb_uint8",
        ),
    ),
}


# --------------------------------------------------------------------------------------
# CFA utilities
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CfaPlanes:
    """One 2x2 CFA rearrangement result with true sampling coordinates.

    ``values`` is ``[4, H//2, W//2]`` float32, plane order ``(R, G1, G2, B)``.
    ``positions`` is ``[4, 2]`` and holds the ``(y, x)`` coordinate of the plane's
    *first* sample in the sensor's pixel-centre convention; ``step`` is ``(2, 2)``
    for every plane. Plane ``p`` sample ``(i, j)`` therefore reads the sensor at::

        (y, x) = positions[p] + 2 * (i, j)

    so the four planes are never co-sited: two of them are offset by one pixel in
    ``y`` or ``x``, and the diagonal pair by one pixel in both.
    """

    values: np.ndarray
    colours: tuple[str, str, str, str]
    positions: np.ndarray
    grid_offset: tuple[float, float]
    pattern: str
    step: tuple[float, float] = (2.0, 2.0)

    @property
    def shapes(self) -> tuple[int, int]:
        return int(self.values.shape[1]), int(self.values.shape[2])

    def coordinates(self) -> np.ndarray:
        """``[4, H//2, W//2, 2]`` absolute ``(y, x)`` coordinates of every plane sample."""
        _, h, w = self.values.shape
        rows = np.arange(h, dtype=np.float64)[None, :, None] * self.step[0]
        cols = np.arange(w, dtype=np.float64)[None, None, :] * self.step[1]
        y = self.positions[:, 0, None, None] + rows
        x = self.positions[:, 1, None, None] + cols
        return np.stack(np.broadcast_arrays(y, x), axis=-1)


def cfa_plane_positions(
    pattern: str,
    grid_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """``[4, 2]`` ``(y, x)`` coordinate of each plane's first sample.

    Coordinates use the ``+0.5`` pixel-centre convention, so ``GRBG`` sample
    ``G1`` (green at tile position ``(0, 0)``) sits at ``(0.5, 0.5)`` while
    ``R`` (tile position ``(0, 1)``) sits at ``(0.5, 1.5)``. ``grid_offset``
    shifts the whole mosaic inside the sensor, so a crop propagates into the
    recorded coordinates instead of silently re-phasing the pattern.
    """
    tile_of_plane = _cfa_tile_of_plane(pattern)
    positions = np.empty((4, 2), dtype=np.float64)
    for plane_index, tile_index in enumerate(tile_of_plane):
        ty, tx = divmod(int(tile_index), 2)
        positions[plane_index] = (
            grid_offset[0] + ty + 0.5,
            grid_offset[1] + tx + 0.5,
        )
    return positions


def _cfa_tile_of_plane(pattern: str) -> tuple[int, int, int, int]:
    """Tile index (0..3, row-major) of each ``(R, G1, G2, B)`` plane."""
    if pattern not in CFA_PATTERNS:
        raise ValueError(f"unknown CFA pattern: {pattern!r}")
    tiles = CFA_PATTERNS[pattern]
    for colour in ("R", "B"):
        if tiles.count(colour) != 1:
            raise ValueError(f"pattern {pattern!r} must place exactly one {colour}")
    if tiles.count("G") != 2:
        raise ValueError(f"pattern {pattern!r} must place exactly two G samples")
    # ``tiles.index`` in order returns the first then second green occurrence,
    # so G1/G2 keep a stable identity across the four Bayer phases.
    return (
        tiles.index("R"),
        tiles.index("G"),
        tiles.index("G", tiles.index("G") + 1),
        tiles.index("B"),
    )


def cfa_plane_values(
    raw: np.ndarray,
    pattern: str,
    grid_offset: tuple[int, int] = (0, 0),
) -> CfaPlanes:
    """Rearrange a 2x2 CFA mosaic into four planes, keeping the true offsets.

    The mosaic must already be in the pattern's own phase: element ``(i, j)`` has
    the colour ``pattern`` declares for ``(i mod 2, j mod 2)``. ``grid_offset`` is
    the ``(row, col)`` index of that mosaic inside the full sensor.
    """
    if pattern not in CFA_PATTERNS:
        raise ValueError(f"unknown CFA pattern: {pattern!r}")
    array = np.asarray(raw)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"CFA mosaic must be 2-D (or [1,H,W]), got shape {np.asarray(raw).shape}")
    if array.shape[0] % 2 or array.shape[1] % 2:
        raise ValueError(f"CFA mosaic needs even H and W, got {array.shape}")
    values = np.empty((4, array.shape[0] // 2, array.shape[1] // 2), dtype=np.float32)
    for plane_index, tile_index in enumerate(_cfa_tile_of_plane(pattern)):
        dy, dx = divmod(int(tile_index), 2)
        values[plane_index] = array[dy::2, dx::2].astype(np.float32)
    return CfaPlanes(
        values=values,
        colours=PLANE_COLOURS,
        positions=cfa_plane_positions(pattern, grid_offset),
        grid_offset=(float(grid_offset[0]), float(grid_offset[1])),
        pattern=pattern,
    )


@dataclass(frozen=True, slots=True)
class CfaCalibration:
    """Black/white level calibration plus the three independent quality masks.

    ``missing_mask`` (never sampled), ``bad_mask`` (known defective pixel) and
    ``saturated_mask`` (clipped at the white level) are kept separate; a sample
    may carry several. Gain and exposure are optional and default to ``None``
    because the RawVLA caches do not provide them - unavailable stays unavailable.
    """

    black_level: float | np.ndarray = 0.0
    white_level: float | np.ndarray = 1.0
    missing_mask: np.ndarray | None = None
    bad_mask: np.ndarray | None = None
    saturated_mask: np.ndarray | None = None
    analog_gain: float | None = None
    digital_gain: float | None = None
    exposure_s: float | None = None

    def __post_init__(self) -> None:
        white = np.asarray(self.white_level, dtype=np.float64)
        black = np.asarray(self.black_level, dtype=np.float64)
        if np.any(white - black <= 0.0):
            raise ValueError("white_level must exceed black_level everywhere")


def apply_cfa_calibration(
    raw: np.ndarray,
    calibration: CfaCalibration,
    *,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``x = (raw - black) / (white - black)`` in FP32 with the masks kept apart.

    Missing and bad samples become NaN because an absent or defective sample is
    not a measurement; the caller keeps the masks to tell them from a real value.
    Saturated samples keep their clipped value but stay flagged, so downstream
    likelihood code must not treat them as ordinary exact observations.
    """
    array = np.asarray(raw)
    black = np.asarray(calibration.black_level, dtype=np.float32)
    span = np.asarray(calibration.white_level, dtype=np.float32) - black
    normalised = (array.astype(np.float32) - black) / span
    missing = _mask_or_false(calibration.missing_mask, array.shape)
    bad = _mask_or_false(calibration.bad_mask, array.shape)
    saturated = _mask_or_false(calibration.saturated_mask, array.shape)
    invalid = missing | bad
    if invalid.any():
        normalised = np.where(invalid, np.float32("nan"), normalised)
    metadata = {
        "black_level": _scalar_or_list(calibration.black_level),
        "white_level": _scalar_or_list(calibration.white_level),
        "normalisation": "x = (raw - black) / (white - black)",
        "missing_mask_semantics": "true where no sample was acquired; value stored as NaN",
        "bad_mask_semantics": "true where the pixel is known defective; value stored as NaN",
        "saturated_mask_semantics": (
            "true where the sample clipped at the white level; value kept"
        ),
        "missing_count": int(missing.sum()),
        "bad_count": int(bad.sum()),
        "saturated_count": int(saturated.sum()),
        "analog_gain": calibration.analog_gain,
        "digital_gain": calibration.digital_gain,
        "exposure_s": calibration.exposure_s,
    }
    return normalised.astype(dtype, copy=False), metadata


def _mask_or_false(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _scalar_or_list(value: Any) -> Any:
    array = np.asarray(value)
    return float(array) if array.ndim == 0 else array.astype(float).tolist()


# --------------------------------------------------------------------------------------
# noise model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """Declared sensor noise, in the *same numeric units* as the values it scores.

    ``variance = shot * max(x, 0) + read_var``, with ``read_var`` a variance (not
    a standard deviation). The default scaling follows the RawVLA conversion
    itself: LIBERO uint8 is divided by ``255``, so DN coefficients must be divided
    by ``255`` (shot) and ``255**2`` (read); RoboTwin float32 is used as stored and
    the coefficients apply directly. This is a declared *scaling convention*, not
    a measurement of the RawVLA sensors - their true noise is not published in the
    caches - so :func:`fitted_variance_estimate` exists for the undeclared case.
    """

    shot: float = 0.0
    read_var: float = 0.0
    declared: bool = False
    source: str = "none"

    def scaled_for(self, unit_scale: float, unit_scale_is_divide: bool) -> NoiseModel:
        """Rescale the coefficients when values were divided by ``unit_scale``."""
        if not unit_scale_is_divide or unit_scale == 1.0:
            return self
        return dataclasses.replace(
            self,
            shot=self.shot / unit_scale,
            read_var=self.read_var / (unit_scale * unit_scale),
        )


def sensor_variance(
    values: np.ndarray,
    model: NoiseModel,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``variance = shot * max(x, 0) + read_var`` plus provenance metadata."""
    array = np.asarray(values, dtype=np.float32)
    variance = (model.shot * np.maximum(array, 0.0) + model.read_var).astype(np.float32)
    if valid_mask is not None:
        variance = np.where(np.asarray(valid_mask, dtype=bool), variance, np.float32("nan"))
    metadata = {
        "variance_expression": "variance = shot * max(x, 0) + read_var",
        "noise_model_declared": bool(model.declared),
        "noise_source": model.source,
        "shot": float(model.shot),
        "read_var": float(model.read_var),
        "variance_is_estimate": not bool(model.declared),
    }
    return variance, metadata


def fitted_variance_estimate(
    residuals: np.ndarray,
    *,
    min_samples: int = 8,
) -> tuple[float, dict[str, Any]]:
    """Scale-only variance fitted from residuals, explicitly marked as an estimate.

    With ``sigma2 = mean(r^2)`` the result is a *fitted scale*, not a calibrated
    sensor variance. Energies built from it are relative explanation weights and
    must not be reported as posterior probabilities.
    """
    array = np.asarray(residuals, dtype=np.float64).ravel()
    finite = np.isfinite(array)
    count = int(finite.sum())
    if count < min_samples:
        return float("nan"), {
            "fitted_variance": None,
            "estimate_available": False,
            "samples": count,
            "reason": f"fewer than {min_samples} finite residuals",
        }
    variance = float(np.mean(array[finite] ** 2))
    return variance, {
        "fitted_variance": variance,
        "estimate_available": True,
        "variance_is_estimate": True,
        "estimator": "mean(residual**2)",
        "samples": count,
        "calibrated_probability": False,
    }


# --------------------------------------------------------------------------------------
# sample-coordinate description
# --------------------------------------------------------------------------------------


def describe_frame_sampling(
    channels: int,
    *,
    cfa_pattern: str = "",
    pixel_offset: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """``[C, n, 2]`` float32 ``(y, x)`` preview of a channel's sampling grid.

    Returned layout, uniform for both input modes:

    * ``[c, 0]`` - coordinate of the first sample of channel ``c``,
    * ``[c, 1]`` - coordinate step between consecutive samples of channel ``c``,
    * ``[c, 2]`` - ``(0, 0)`` reference marker, present so the array length is
      fixed at ``n = 3`` for every mode (the contract requires a rectangular
      array, and the full per-pixel coordinates of the CFA path are carried by
      :meth:`RawVLASample.measurement_payload` in ``planes[*]["sample_xy"]``).

    ``rawvla_rgb3`` is co-sited: every channel starts at the frame's first pixel
    centre (``pixel_offset``) with step ``(1, 1)``. ``bayer_cfa`` uses the four
    planes' true positions with step ``(2, 2)``, so the channels are *not*
    interchangeable colours on one grid.
    """
    if cfa_pattern:
        positions = cfa_plane_positions(cfa_pattern)
        array = np.zeros((4, 3, 2), dtype=np.float32)
        array[:, 0, :] = positions.astype(np.float32)
        array[:, 1, :] = np.float32(2.0)
        return array
    array = np.zeros((int(channels), 3, 2), dtype=np.float32)
    array[:, 0, 0] = np.float32(pixel_offset[0])
    array[:, 0, 1] = np.float32(pixel_offset[1])
    array[:, 1, :] = np.float32(1.0)
    return array


# --------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalisationStats:
    """Statistics for the author's normalisation modes, loaded from a declared JSON.

    Two layouts occur in the frozen RawVLA assets and both are accepted:
    ``{"norm_stats": {"state": ..., "actions": ...}}`` (openpi / LIBERO) and a
    bare ``{"action": ..., "state": ...}`` mapping (StarVLA / RoboTwin).
    """

    mode: str
    state: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    source_path: str = ""
    sha256: str = ""
    stats_key: str = ""
    epsilon: float = 1e-6
    epsilon_mode: str = "floor"

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        mode: str,
        stats_key: str = "",
        action_stats_key: str = "actions",
        state_stats_key: str = "state",
        epsilon: float = 1e-6,
        epsilon_mode: str = "floor",
    ) -> NormalisationStats:
        stats_path = Path(path)
        if not stats_path.is_file():
            raise FileNotFoundError(f"Normalisation statistics not found: {stats_path}")
        raw_bytes = stats_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if stats_key:
            node: Any = payload
            for component in stats_key.split("."):
                node = node[component]
        else:
            node = payload.get("norm_stats", payload)
        if not isinstance(node, dict):
            raise ValueError(f"statistics at {stats_path} do not resolve to a mapping")
        action_stats = node.get(action_stats_key, node.get("action"))
        state_stats = node.get(state_stats_key, node.get("state"))
        return cls(
            mode=mode,
            state=None if state_stats is None else dict(state_stats),
            action=None if action_stats is None else dict(action_stats),
            source_path=str(stats_path),
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            stats_key=stats_key,
            epsilon=epsilon,
            epsilon_mode=epsilon_mode,
        )


def _std_denominator(std: np.ndarray, stats: NormalisationStats) -> np.ndarray:
    if not np.isfinite(stats.epsilon) or stats.epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if np.any(std < 0) or not np.isfinite(std).all():
        raise ValueError("standard deviations must be finite and nonnegative")
    if stats.epsilon_mode == "add":
        return std + np.float32(stats.epsilon)
    if stats.epsilon_mode == "floor":
        return np.maximum(std, np.float32(stats.epsilon))
    raise ValueError(f"unknown epsilon mode {stats.epsilon_mode!r}")


def _as_stat_array(values: Any, dim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).ravel()
    if array.size < dim:
        raise ValueError(f"statistics field {name!r} has {array.size} entries, need {dim}")
    return array[:dim]


#: Epsilon used by the author's ``_minmax`` denominator floor. Reproduced exactly;
#: a different epsilon (``1e-6`` == ``1.0e-6`` but a different literal path) would
#: silently change results for degenerate statistics.
MINMAX_EPSILON = 1.0e-6


def _minmax(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Author's ``RAWVLALightNPZDataset._minmax``, reproduced exactly."""
    return np.clip(2.0 * (value - low) / np.maximum(high - low, MINMAX_EPSILON) - 1.0, -1.0, 1.0)


def normalise_actions(
    actions: np.ndarray,
    state: np.ndarray | None,
    stats: NormalisationStats | None,
    *,
    dataset_format: str,
    delta_action_indices: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Apply one declared normalisation to an action block.

    ``starvla_minmax`` reproduces the author's RoboTwin path exactly, including
    the ``[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]`` joint reordering, the
    12-dim min-max and the ``> 0.49`` gripper binarisation. ``quantile``
    reproduces the author's LIBERO path (6-dim q01/q99 min-max, then
    ``(a + 1) / 2`` for the gripper). ``zscore``/``pi0_zscore`` reproduce the
    declared delta conversion. Unknown modes raise instead of guessing.
    """
    mode = "none" if stats is None else stats.mode
    block = np.asarray(actions, dtype=np.float32).copy()
    metadata: dict[str, Any] = {
        "normalisation": mode,
        "action_stats_source": None if stats is None else stats.source_path,
        "action_stats_sha256": None if stats is None else stats.sha256,
        "delta_action_indices": list(delta_action_indices),
    }
    if mode == "none":
        metadata["reason"] = "no declared statistics; action values stay in cache units"
        return block, None, metadata

    if mode == "starvla_minmax":
        if dataset_format != "robotwin":
            raise ValueError("starvla_minmax is the author's RoboTwin action protocol")
        order = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])
        if block.shape[-1] != order.size:
            raise ValueError(f"robotwin actions must be 14-dim, got {block.shape[-1]}")
        block = block[:, order]
        if stats is None or stats.action is None:
            raise ValueError("starvla_minmax needs declared action statistics")
        low = _as_stat_array(stats.action["min"], 12, "action.min")
        high = _as_stat_array(stats.action["max"], 12, "action.max")
        block[:, :12] = _minmax(block[:, :12], low, high)
        block[:, 12:] = (block[:, 12:] > 0.49).astype(np.float32)
        metadata["joint_order"] = order.tolist()
        metadata["gripper_rule"] = "(value > 0.49) -> {0, 1}"
        metadata["note"] = "cache-unit input; no delta conversion in this mode"
        return block.astype(np.float32), None, metadata

    if mode in {"zscore", "pi0_zscore"}:
        indices = delta_action_indices
        if mode == "pi0_zscore" and not indices:
            width = block.shape[-1] if state is None else min(6, block.shape[-1], state.shape[-1])
            indices = tuple(range(width))
        if indices:
            if state is None:
                raise ValueError("delta conversion requires the current state array")
            selected = np.asarray(indices, dtype=np.int64)
            block[:, selected] -= np.asarray(state, dtype=np.float32)[None, selected]
        if stats is None or stats.action is None:
            raise ValueError(f"{mode} needs declared action statistics")
        mean = _as_stat_array(stats.action["mean"], block.shape[-1], "action.mean")
        std = _as_stat_array(stats.action["std"], block.shape[-1], "action.std")
        block = (block - mean) / _std_denominator(std, stats)
        normalised_state = None
        if state is not None and stats.state is not None:
            state_mean = _as_stat_array(stats.state["mean"], state.shape[-1], "state.mean")
            state_std = _as_stat_array(stats.state["std"], state.shape[-1], "state.std")
            normalised_state = ((state - state_mean) / _std_denominator(state_std, stats)).astype(
                np.float32
            )
        metadata["epsilon"] = stats.epsilon
        metadata["epsilon_mode"] = stats.epsilon_mode
        metadata["delta_action_indices"] = list(indices)
        return block.astype(np.float32), normalised_state, metadata

    if mode == "quantile":
        if stats is None or stats.action is None:
            raise ValueError("quantile normalisation needs declared action statistics")
        width = block.shape[-1]
        # The author's LIBERO recipe min-maxes the first six control dims and maps
        # index 6 with (a + 1) / 2. Keep that layout explicit instead of deriving it
        # from the array width, so a widened block cannot silently change meaning.
        gripper_index = 6
        if width <= gripper_index:
            raise ValueError(
                f"quantile mode needs at least {gripper_index + 1} action dims, got {width}"
            )
        low = _as_stat_array(stats.action["q01"], gripper_index, "action.q01")
        high = _as_stat_array(stats.action["q99"], gripper_index, "action.q99")
        block[:, :gripper_index] = _minmax(block[:, :gripper_index], low, high)
        block[:, gripper_index] = (block[:, gripper_index] + 1.0) * 0.5
        metadata["gripper_rule"] = "(action + 1) * 0.5"
        metadata["gripper_index"] = gripper_index
        return block.astype(np.float32), None, metadata

    raise ValueError(f"Unknown action normalisation mode: {mode}")


def normalise_state(
    state: np.ndarray,
    stats: NormalisationStats | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Normalise the proprio vector, or return ``None`` when nothing is declared.

    The author's LIBERO/RoboTwin recipes return the state only for the
    z-score/quantile modes; everywhere else this returns ``None`` rather than
    passing cache units off as normalised.
    """
    mode = "none" if stats is None else stats.mode
    metadata: dict[str, Any] = {
        "state_normalisation": "unavailable",
        "state_stats_source": None if stats is None else stats.source_path,
        "reason": (
            "the author protocol does not normalise the state in this mode; raw cache "
            "units are not silently passed off as normalised"
        ),
    }
    if mode in {"none", "starvla_minmax"} or stats is None or stats.state is None:
        return None, metadata
    array = np.asarray(state, dtype=np.float32)
    if mode in {"zscore", "pi0_zscore"}:
        mean = _as_stat_array(stats.state["mean"], array.size, "state.mean")
        std = _as_stat_array(stats.state["std"], array.size, "state.std")
        return (
            ((array - mean) / _std_denominator(std, stats)).astype(np.float32),
            {"state_normalisation": mode, "state_stats_source": stats.source_path},
        )
    if mode == "quantile":
        low = _as_stat_array(stats.state["q01"], array.size, "state.q01")
        high = _as_stat_array(stats.state["q99"], array.size, "state.q99")
        return _minmax(array, low, high), {
            "state_normalisation": mode,
            "state_stats_source": stats.source_path,
        }
    raise ValueError(f"Unknown state normalisation mode: {mode}")


# --------------------------------------------------------------------------------------
# dataset windows
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowOptions:
    """Declared window geometry, mirroring the author loader field for field."""

    #: ``None`` means "take the format's declared default". ``0`` is a real value
    #: (LIBERO supervises from the current step), so the two must stay distinct.
    observation_stride: int | None = None
    burst_frames: int = 1
    rnn_frames: int = 1
    action_horizon: int | None = None
    action_offset: int | None = None
    state_source_key: str = ""
    #: ``False`` until a cache is shown to contain a real executed-action record.
    executed_action_available: bool = False

    def resolved_for(self, spec: FormatSpec) -> WindowOptions:
        """Fill the format's declared defaults for every field left as ``None``."""
        return dataclasses.replace(
            self,
            observation_stride=(
                spec.default_observation_stride
                if self.observation_stride is None
                else self.observation_stride
            ),
            action_horizon=(
                spec.default_action_horizon if self.action_horizon is None else self.action_horizon
            ),
            action_offset=(
                spec.default_action_offset if self.action_offset is None else self.action_offset
            ),
            state_source_key=self.state_source_key or spec.state_source_key,
        )

    def __post_init__(self) -> None:
        for name in ("burst_frames", "rnn_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("observation_stride", "action_horizon"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when set, got {value}")
        # `action_offset` is a shift, not a count: 0 means "supervise from the
        # current step" and is LIBERO's declared default.
        if self.action_offset is not None and self.action_offset < 0:
            raise ValueError("action_offset must be non-negative when set")


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    """One NPZ episode file with its provenance fields."""

    path: str
    relative_path: str
    acquisition_source: str
    episode_group_id: str
    num_frames: int
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class AcquisitionSource:
    """A group of episodes that must stay on one side of the split."""

    source_id: str
    episodes: tuple[str, ...]
    num_frames: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "episodes": list(self.episodes),
            "num_frames": self.num_frames,
        }


@dataclass(slots=True)
class RawVLASample:
    """One window of measurements plus independent supervision.

    Shapes, with ``R = rnn_frames``, ``T = burst_frames``, ``K = action_horizon``
    and ``C`` cameras:

    * ``burst_values[c]``       ``[R, T, C, H, W]`` float32 measurements,
    * ``burst_new_measurement`` ``[R, T]`` bool; ``False`` marks a frame whose
      source index already appeared earlier in the window (padding repetition),
    * ``new_measurement``       ``[R]`` bool for the outer recurrent axis; it
      follows the newest burst frame of each step, i.e. the index the author's
      outer axis addresses,
    * ``observation_indices``   ``[R]`` clipped source frame indices,
    * ``burst_indices[c]``      ``[R, T]`` clipped source frame indices,
    * ``supervision_actions``   ``[K, A]`` normalised target actions,
    * ``supervision_mask``      ``[K]`` bool; ``False`` marks edge padding, and
      those entries must be dropped so padding is never counted twice,
    * ``padding_contribution``  fraction of the action block that is padding,
    * ``executed_action``       always ``None`` here - see the module docstring.
    """

    dataset_format: str
    split: str
    episode: EpisodeRef
    camera_ids: tuple[str, ...]
    observation_indices: np.ndarray
    burst_indices: np.ndarray
    new_measurement: np.ndarray
    burst_new_measurement: np.ndarray
    unique_measurement_count: int
    burst_values: tuple[np.ndarray, ...]
    pixel_valid_mask: tuple[np.ndarray, ...]
    bad_mask: tuple[np.ndarray, ...]
    saturated_mask: tuple[np.ndarray, ...]
    frame_ids: tuple[tuple[tuple[str, ...], ...], ...]
    supervision_actions: np.ndarray
    supervision_indices: np.ndarray
    supervision_mask: np.ndarray
    padding_contribution: float
    proprio: np.ndarray | None
    proprio_available: bool
    executed_action: None
    executed_action_available: bool
    variance: tuple[np.ndarray | None, ...]
    range_evidence: tuple[dict[str, Any], ...]
    normalisation: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- payload ----------------------------------------------------------------------

    def measurement_payload(self, camera_id: str) -> dict[str, Any]:
        """Camera payload in the shared-contract field names plus agreed extras.

        Keys: ``values``, ``valid_mask``, ``bad_mask``, ``saturated_mask``,
        ``sample_xy``, ``episode_ids``, ``frame_ids``, ``camera_id``,
        ``timestamp``, ``time_unit``, ``new_measurement``, ``metadata``,
        ``metadata_available``, plus ``variance`` (``None`` when no noise model is
        declared), ``variance_is_estimate``, ``range_evidence`` and ``planes``
        (CFA path only). Nothing is fabricated: an absent field is ``None`` with
        ``metadata_available[field] is False``, never a plausible zero.
        """
        slot = self._camera_slot(camera_id)
        # `burst_indices` is shared by every camera: the streams are synchronised,
        # so the source frame index of an observation step does not depend on which
        # camera is being read.
        indices = self.burst_indices[:, -1]
        planes = self.provenance["cameras"][slot].get("planes")
        noise = self.provenance.get("noise", {})
        declared = bool(noise.get("noise_model_declared", False))
        variance = None if not declared else self.variance[slot]
        metadata = {
            "dataset_format": self.dataset_format,
            "split": self.split,
            "episode_path": self.episode.path,
            "episode_group_id": self.episode.episode_group_id,
            "acquisition_source": self.episode.acquisition_source,
            "num_frames": self.episode.num_frames,
            "frame_index_unit": "frame",
            "observation_indices": indices.tolist(),
            "burst_indices": self.burst_indices.tolist(),
            "unique_measurement_count": self.unique_measurement_count,
            "input_mode": self.provenance["cameras"][slot]["metadata"].get(
                "input_mode", "rawvla_rgb3"
            ),
            "unit_metadata": self.provenance["cameras"][slot]["metadata"],
            "noise": noise,
            "window": self.provenance.get("window", {}),
            "task_description": self.provenance.get("task_description", ""),
        }
        return {
            "values": self.burst_values[slot][:, -1],
            "valid_mask": self.pixel_valid_mask[slot][:, -1],
            "bad_mask": self.bad_mask[slot][:, -1],
            "saturated_mask": self.saturated_mask[slot][:, -1],
            "sample_xy": self.provenance["cameras"][slot].get("sample_xy"),
            "episode_ids": tuple(self.episode.episode_group_id for _ in range(indices.size)),
            # frame_ids is [camera][rnn step][burst position]; take the newest burst
            # frame of each observation step for this camera.
            "frame_ids": tuple(row[-1] for row in self.frame_ids[slot]),
            "camera_id": camera_id,
            "timestamp": None,
            "time_unit": "frame",
            "new_measurement": self.new_measurement.copy(),
            "variance": variance,
            "variance_is_estimate": not declared,
            "range_evidence": self.range_evidence[slot],
            "planes": planes,
            "metadata": metadata,
            "metadata_available": {
                "timestamp": False,
                "timestamps": False,
                "exposure": False,
                "analog_gain": False,
                "digital_gain": False,
                "cfa_pattern": bool(planes),
                "camera_intrinsics": False,
                "camera_transform": False,
                "executed_action": False,
                "noise_model": declared,
                "variance": declared,
                "task_description": bool(metadata["task_description"]),
            },
        }

    def _camera_slot(self, camera_id: str) -> int:
        try:
            return self.camera_ids.index(camera_id)
        except ValueError as error:
            have = list(self.camera_ids)
            raise KeyError(f"unknown camera_id {camera_id!r}; have {have}") from error

    def all_measurement_payloads(self) -> dict[str, dict[str, Any]]:
        return {camera_id: self.measurement_payload(camera_id) for camera_id in self.camera_ids}


def build_measurement_frame(sample: RawVLASample, camera_id: str) -> Any:
    """Build one shared-contract ``MeasurementFrame`` from a sample payload.

    The contract object carries the values, the three masks, the sample
    coordinates, the identities, ``new_measurement=False`` for repeated padding
    frames and a truthful ``metadata_available`` map, so a caller can tell "this
    field is absent" from "this field is zero". The extras that do not fit the
    frozen dataclass (``variance``, ``planes``, ``range_evidence``) stay reachable
    through :meth:`RawVLASample.measurement_payload`.
    """
    from owac.core.encoder_contract import MeasurementFrame

    payload = sample.measurement_payload(camera_id)
    return MeasurementFrame(
        values=payload["values"],
        valid_mask=payload["valid_mask"],
        bad_mask=payload["bad_mask"],
        saturated_mask=payload["saturated_mask"],
        sample_xy=payload["sample_xy"],
        episode_ids=payload["episode_ids"],
        frame_ids=payload["frame_ids"],
        camera_id=camera_id,
        timestamp=None,
        time_unit="frame",
        new_measurement=payload["new_measurement"],
        metadata={
            "payload": payload["metadata"],
            "variance": payload["variance"],
            "variance_is_estimate": payload["variance_is_estimate"],
            "planes": payload["planes"],
            "range_evidence": payload["range_evidence"],
        },
        metadata_available=payload["metadata_available"],
    )


# --------------------------------------------------------------------------------------
# split manifest
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Deterministic episode-level train/validation split.

    The split unit is the *acquisition source* (a LIBERO suite directory, a
    RoboTwin task directory). Frames of one episode are never separated, and every
    derived variant of a source (``variant_00``, ``variant_01``, ...) lands on the
    same side because the source is the unit.
    """

    version: str
    seed: int
    val_fraction: float
    dataset_format: str
    cache_root: str
    train_sources: tuple[str, ...]
    val_sources: tuple[str, ...]
    train_episodes: tuple[str, ...]
    val_episodes: tuple[str, ...]
    episodes: tuple[dict[str, Any], ...]
    num_frames: dict[str, int]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "split_unit": "acquisition_source",
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "dataset_format": self.dataset_format,
            "cache_root": self.cache_root,
            "train_sources": list(self.train_sources),
            "val_sources": list(self.val_sources),
            "train_episodes": list(self.train_episodes),
            "val_episodes": list(self.val_episodes),
            "episodes": [dict(item) for item in self.episodes],
            "num_frames": dict(self.num_frames),
            "notes": self.notes,
        }

    def episodes_for(self, split: str) -> tuple[str, ...]:
        if split == "train":
            return self.train_episodes
        if split == "validation":
            return self.val_episodes
        raise ValueError(f"unknown split: {split!r}")

    def check(self) -> list[str]:
        """Consistency problems (empty means self-consistent)."""
        problems: list[str] = []
        overlap = set(self.train_episodes) & set(self.val_episodes)
        if overlap:
            problems.append(f"episodes on both sides: {sorted(overlap)[:5]}")
        source_overlap = set(self.train_sources) & set(self.val_sources)
        if source_overlap:
            problems.append(f"sources on both sides: {sorted(source_overlap)[:5]}")
        declared = {item["relative_path"]: item["acquisition_source"] for item in self.episodes}
        for relative in self.train_episodes + self.val_episodes:
            if relative not in declared:
                problems.append(f"episode {relative!r} missing from the episode table")
        for relative in self.val_episodes:
            if declared.get(relative) in set(self.train_sources):
                problems.append(f"validation episode {relative!r} comes from a train source")
        if self.num_frames.get("train", 0) <= 0 or self.num_frames.get("validation", 0) <= 0:
            problems.append("a split side has no frames")
        return problems


def discover_npz_files(cache_root: str | Path, dataset_format: str) -> tuple[list[str], str]:
    """Find episode NPZ files and report which search pattern matched.

    Tried in order: each declared prefix (``data/robotwin2/raw``, then
    ``robotwin2/raw``, then ``raw``), then a recursive ``**/*.npz``. The first
    non-empty result wins, so a caller may point at the HuggingFace snapshot root,
    at the ``data/`` subtree, or straight at a per-format leaf directory. Results
    are sorted for determinism.
    """
    if dataset_format not in FORMAT_SPECS:
        raise ValueError(f"unknown dataset_format: {dataset_format!r}")
    root = Path(cache_root)
    if not root.exists():
        return [], "cache_root_missing"
    spec = FORMAT_SPECS[dataset_format]
    for pattern in (*spec.markers, "**"):
        if pattern == "**":
            found = sorted(glob.glob(str(root / "**" / "*.npz"), recursive=True))
        else:
            found = sorted(glob.glob(str(root / pattern / "**" / "*.npz"), recursive=True))
        if found:
            return list(dict.fromkeys(found)), f"{pattern}/**/*.npz"
    return [], "no_npz_found"


def _relative_episode_path(path: str, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return Path(path).name


def _acquisition_source(relative_path: str) -> str:
    """Group id that must stay on one side of the split.

    LIBERO ``data/libero/raw/<suite>/episode_XXXX/variant_YY.npz`` -> ``<suite>``;
    RoboTwin ``data/robotwin2/raw/<task>/episode_XXXX.npz`` -> ``<task>``. A
    fallback keeps a flat directory usable by naming the parent directory.
    """
    parts = Path(relative_path).parts
    for index, part in enumerate(parts):
        if part.startswith("episode_"):
            return parts[index - 1] if index >= 1 else "."
    if len(parts) >= 2:
        return parts[-2]
    return "."


def _episode_identity(relative_path: str) -> str:
    """Concise episode identity including the variant, so variants stay distinct.

    ``data/libero/raw/<suite>/episode_000018/variant_00.npz`` becomes
    ``<suite>/episode_000018/variant_00`` and
    ``data/robotwin2/raw/<task>/episode_000018.npz`` becomes
    ``<task>/episode_000018``. The acquisition source is still the outer key of the
    split; this identity is what frame ids and episode-level bookkeeping carry.
    """
    path = Path(relative_path).with_suffix("")
    parts = path.parts
    for index, part in enumerate(parts):
        if part.startswith("episode_"):
            return "/".join(parts[index - 1 :]) if index >= 1 else "/".join(parts[index:])
    return path.as_posix()


def _split_side(source_id: str, seed: int, val_fraction: float) -> str:
    """Deterministic hash split over source ids (no RNG state, stable across runs)."""
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "validation" if unit < val_fraction else "train"


def compute_split(
    episodes: list[EpisodeRef],
    *,
    seed: int = 0,
    val_fraction: float = 0.25,
    dataset_format: str,
    cache_root: str,
    version: str = "n02-rawvla-split-v1",
) -> SplitManifest:
    """Episode-level deterministic split, implemented over acquisition sources."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie strictly between 0 and 1")
    if not episodes:
        raise ValueError("cannot build a split from zero episodes")
    grouped: dict[str, list[EpisodeRef]] = {}
    for episode in episodes:
        grouped.setdefault(episode.acquisition_source, []).append(episode)
    # Deterministic hash split. With very few acquisition sources a single seed can
    # easily put every source on one side; re-salt deterministically until both
    # sides are non-empty rather than emitting a useless split. The result is still
    # a pure function of (sources, seed, val_fraction).
    for attempt in range(64):
        salt = "" if attempt == 0 else f"#{attempt}"
        train_sources: list[str] = []
        val_sources: list[str] = []
        for source_id in sorted(grouped):
            side = _split_side(f"{source_id}{salt}", seed, val_fraction)
            (val_sources if side == "validation" else train_sources).append(source_id)
        if train_sources and val_sources:
            break
    else:  # pragma: no cover - only reachable with a single source
        raise ValueError(
            "cannot build a non-empty two-sided split from one acquisition source; "
            "add episodes from another source"
        )
    train_episodes = tuple(
        sorted(ep.relative_path for source in train_sources for ep in grouped[source])
    )
    val_episodes = tuple(
        sorted(ep.relative_path for source in val_sources for ep in grouped[source])
    )
    frames = {
        "train": sum(ep.num_frames for source in train_sources for ep in grouped[source]),
        "validation": sum(ep.num_frames for source in val_sources for ep in grouped[source]),
    }
    return SplitManifest(
        version=version,
        seed=seed,
        val_fraction=val_fraction,
        dataset_format=dataset_format,
        cache_root=cache_root,
        train_sources=tuple(train_sources),
        val_sources=tuple(val_sources),
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        episodes=tuple(ep.as_dict() for ep in episodes),
        num_frames=frames,
        notes=(
            "Split unit is the acquisition source (LIBERO suite directory / RoboTwin task "
            "directory). Adjacent frames are never separated and every derived variant of an "
            "episode stays on the same side. Formal closed-loop manifests are tracked "
            "separately and are not produced here."
        ),
    )


def write_split_manifest(manifest: SplitManifest, path: str | Path) -> str:
    """Write the split JSON and return its sha256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.as_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_split_manifest(path: str | Path) -> SplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitManifest(
        version=payload["version"],
        seed=int(payload["seed"]),
        val_fraction=float(payload["val_fraction"]),
        dataset_format=payload["dataset_format"],
        cache_root=payload["cache_root"],
        train_sources=tuple(payload["train_sources"]),
        val_sources=tuple(payload["val_sources"]),
        train_episodes=tuple(payload["train_episodes"]),
        val_episodes=tuple(payload["val_episodes"]),
        episodes=tuple(payload["episodes"]),
        num_frames={key: int(value) for key, value in payload["num_frames"].items()},
        notes=payload.get("notes", ""),
    )


def _sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_cache(cache_root: str | Path, dataset_format: str) -> dict[str, Any]:
    """Structural evidence for a cache: keys, shapes, dtypes, ranges, hashes."""
    paths, pattern = discover_npz_files(cache_root, dataset_format)
    spec = FORMAT_SPECS[dataset_format]
    evidence: dict[str, Any] = {
        "cache_root": str(cache_root),
        "dataset_format": dataset_format,
        "search_pattern": pattern,
        "npz_files": len(paths),
        "measurement_keys": list(spec.measurement_keys),
        "state_source_key": spec.state_source_key,
        "episodes": [],
        "key_availability": {},
        "range_check_method": "measured on the loaded array (min/max over all elements)",
        "problems": [],
    }
    if not paths:
        evidence["status"] = "no_data_found"
        evidence["note"] = (
            "No NPZ files matched; nothing is inferred. On the deployed server the file "
            "cache root is /workspace/owac/rawvla/data/libero/raw (LIBERO) or "
            "/workspace/owac/rawvla/data/robotwin2/raw (RoboTwin)."
        )
        return evidence
    key_counts: dict[str, int] = {}
    for path in paths:
        relative = _relative_episode_path(path, Path(cache_root))
        record: dict[str, Any] = {"path": str(path), "relative_path": relative}
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = list(archive.files)
                record["keys"] = keys
                for key in keys:
                    key_counts[key] = key_counts.get(key, 0) + 1
                record["num_frames"] = (
                    int(np.asarray(archive["actions"]).shape[0]) if "actions" in keys else None
                )
                for key in (*spec.measurement_keys, spec.state_source_key):
                    if key not in keys:
                        evidence["problems"].append(f"{relative}: missing key {key!r}")
                        continue
                    array = np.asarray(archive[key])
                    record[key] = {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "min": float(array.min()),
                        "max": float(array.max()),
                        "finite": bool(np.isfinite(array).all()),
                    }
        except (OSError, ValueError, KeyError) as error:  # pragma: no cover - corrupt file path
            evidence["problems"].append(f"{relative}: unreadable ({error})")
            record["error"] = str(error)
        record["sha256"] = _sha256_file(path)
        evidence["episodes"].append(record)
    evidence["key_availability"] = {
        key: {"files": count, "fraction": count / len(paths)}
        for key, count in sorted(key_counts.items())
    }
    evidence["status"] = "ok"
    return evidence


# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------


class RawVLAWindowDataset:
    """Causal measurement windows over RawVLA replay NPZ caches.

    numpy only: no torch import, no PIL, no RGB image output, no RGB target.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        dataset_format: str = "libero",
        window: WindowOptions | None = None,
        stats: NormalisationStats | None = None,
        noise: NoiseModel | None = None,
        cfa_pattern: str = "",
        cfa_calibration: CfaCalibration | None = None,
        format_spec: FormatSpec | None = None,
        split_seed: int = 0,
        val_fraction: float = 0.25,
        samples_per_episode: int = 32,
        seed: int = 0,
        split: str = "train",
        max_episodes: int | None = None,
        hash_episodes: bool = False,
        episode_paths: list[str] | None = None,
    ) -> None:
        if dataset_format not in FORMAT_SPECS:
            raise ValueError(f"unknown dataset_format: {dataset_format!r}")
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown split: {split!r}")
        self.cache_root = Path(cache_root)
        self.dataset_format = dataset_format
        # An explicit spec lets a caller load a genuinely mosaic cache without
        # inventing an RGB->CFA conversion; every other consumer uses the frozen
        # per-format spec.
        self.spec = format_spec or FORMAT_SPECS[dataset_format]
        self.window = (window or WindowOptions()).resolved_for(self.spec)
        self.stats = stats
        self.noise = noise or NoiseModel()
        self.cfa_pattern = cfa_pattern
        # The CFA path applies *no implicit* scaling: without an explicit
        # calibration the planes keep the cache's own integer units, which is the
        # only honest reading of a mosaic. Declaring `cfa_calibration` is what turns
        # them into normalised floats via `x = (raw - black) / (white - black)`.
        self.cfa_calibration = cfa_calibration
        self.split_seed = split_seed
        self.val_fraction = val_fraction
        self.samples_per_episode = samples_per_episode
        self.seed = seed
        self.split = split
        self.hash_episodes = hash_episodes
        if cfa_pattern and cfa_pattern not in CFA_PATTERNS:
            raise ValueError(f"unknown CFA pattern: {cfa_pattern!r}")
        if samples_per_episode < 1:
            raise ValueError("samples_per_episode must be positive")
        paths, self.search_pattern = discover_npz_files(self.cache_root, dataset_format)
        if episode_paths is not None:
            if not set(episode_paths).issubset(paths):
                raise ValueError("episode_paths must be real files discovered inside cache_root")
            paths = sorted(set(episode_paths))
        if max_episodes is not None:
            paths = paths[:max_episodes]
        self._paths = paths
        self._episodes: list[EpisodeRef] | None = None
        self._manifest: SplitManifest | None = None
        self._cached_path: str | None = None
        self._cached: dict[str, np.ndarray] | None = None

    # -- episode bookkeeping -----------------------------------------------------------

    @property
    def paths(self) -> list[str]:
        return list(self._paths)

    def _read_num_frames(self, path: str) -> int:
        with np.load(path, allow_pickle=False) as archive:
            if "actions" not in archive.files:
                raise KeyError(f"episode {path} has no 'actions' array")
            return int(np.asarray(archive["actions"]).shape[0])

    def episodes(self) -> list[EpisodeRef]:
        if self._episodes is None:
            records: list[EpisodeRef] = []
            for path in self._paths:
                relative = _relative_episode_path(path, self.cache_root)
                records.append(
                    EpisodeRef(
                        path=path,
                        relative_path=relative,
                        acquisition_source=_acquisition_source(relative),
                        episode_group_id=_episode_identity(relative),
                        num_frames=self._read_num_frames(path),
                        size_bytes=Path(path).stat().st_size,
                        sha256=_sha256_file(path) if self.hash_episodes else "",
                    )
                )
            self._episodes = records
        return list(self._episodes)

    def split_manifest(self) -> SplitManifest:
        """Build (and cache) the deterministic split for this cache root."""
        if self._manifest is None:
            self._manifest = compute_split(
                self.episodes(),
                seed=self.split_seed,
                val_fraction=self.val_fraction,
                dataset_format=self.dataset_format,
                cache_root=str(self.cache_root),
            )
        return self._manifest

    def split_episodes(self) -> list[EpisodeRef]:
        wanted = set(self.split_manifest().episodes_for(self.split))
        selected = [episode for episode in self.episodes() if episode.relative_path in wanted]
        if not selected:
            raise ValueError(f"split {self.split!r} selected no episodes from {self.cache_root}")
        return selected

    def __len__(self) -> int:
        return self.samples_per_episode * len(self.split_episodes())

    # -- loading -----------------------------------------------------------------------

    def _load_episode(self, path: str) -> dict[str, np.ndarray]:
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as archive:
                self._cached = {key: np.asarray(archive[key]) for key in archive.files}
            self._cached_path = path
        assert self._cached is not None
        return self._cached

    def _calibrate(
        self,
        key: str,
        values: np.ndarray,
        source_dtype: np.dtype,
        source_array: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """Convert one camera stream to declared numeric units, recording the rule.

        Returns ``(values, unit_metadata, range_evidence)``. The conversion itself
        is the only place where units change, and the rule applied is recorded so
        a reader never has to guess whether a ``/255`` happened.
        """
        array = np.asarray(values)
        range_evidence = {
            "dtype": str(source_dtype),
            "min": float(source_array.min()),
            "max": float(source_array.max()),
            "units": "cache units (as stored in the NPZ)",
            "checked_on": "the loaded episode array of this sample",
            "method": "min/max over all elements of the loaded array",
        }
        metadata: dict[str, Any] = {"camera_key": key, "source_dtype": str(source_dtype)}
        if self.cfa_pattern:
            planes = cfa_plane_values(array, self.cfa_pattern)
            values = planes.values
            metadata.update(
                {
                    "input_mode": "bayer_cfa",
                    "cfa_pattern": self.cfa_pattern,
                    "plane_colours": list(planes.colours),
                    "plane_positions_yx": planes.positions.tolist(),
                    "plane_step_yx": list(planes.step),
                    "pixel_centre_convention": PIXEL_CENTRE_CONVENTION,
                    "planes_are_not_co_sited": True,
                }
            )
            if self.cfa_calibration is None:
                metadata.update(
                    {
                        "value_conversion": "none",
                        "unit_divisor": 1.0,
                        "unit_scale_is_divide": False,
                        "raw_units": f"{source_dtype}_cfa_planes_uncorrected",
                        "note": (
                            "No CFA calibration was declared, so the planes keep the "
                            "cache's own units. Declare cfa_calibration to apply "
                            "x = (raw - black) / (white - black)."
                        ),
                    }
                )
                return values, metadata, range_evidence
            calibrated, calibration_metadata = apply_cfa_calibration(values, self.cfa_calibration)
            metadata.update(calibration_metadata)
            metadata.update(
                {
                    "value_conversion": "x = (raw - black) / (white - black)",
                    "unit_divisor": None,
                    "unit_scale_is_divide": False,
                    "raw_units": "cfa_planes_calibrated",
                }
            )
            return calibrated, metadata, range_evidence
        if source_dtype == np.uint8:
            metadata.update(
                {
                    "input_mode": "rawvla_rgb3",
                    "value_conversion": "float32 / 255.0",
                    "unit_divisor": 255.0,
                    "unit_scale_is_divide": True,
                    "raw_units": "uint8_pseudo_raw",
                    "values_are_author_pseudo_raw": True,
                    "note": (
                        "The author's three-channel pseudo-RAW: uint8 divided by 255.0. This "
                        "is not a native Bayer CFA measurement and must not be renamed as one."
                    ),
                }
            )
            return array.astype(np.float32) / 255.0, metadata, range_evidence
        if np.issubdtype(source_dtype, np.floating):
            metadata.update(
                {
                    "input_mode": "rawvla_rgb3",
                    "value_conversion": "none",
                    "unit_divisor": 1.0,
                    "unit_scale_is_divide": False,
                    "raw_units": "float32_hdr_derived_raw",
                    "range_check": (
                        "measured on the loaded array; no /255 and no per-frame max "
                        "normalisation is applied"
                    ),
                    "values_are_author_hdr_derived_raw": True,
                }
            )
            return array.astype(np.float32), metadata, range_evidence
        raise ValueError(f"unsupported measurement dtype for {key!r}: {source_dtype}")

    def _camera_cfa_metadata(self, key: str) -> dict[str, Any]:
        return {
            "camera_key": key,
            "input_mode": "bayer_cfa",
            "cfa_pattern": self.cfa_pattern,
            "plane_colours": list(PLANE_COLOURS),
            "plane_positions_yx": cfa_plane_positions(self.cfa_pattern).tolist(),
            "plane_step_yx": [2.0, 2.0],
            "pixel_centre_convention": PIXEL_CENTRE_CONVENTION,
            "planes_are_not_co_sited": True,
        }

    # -- item construction -------------------------------------------------------------

    def indices_for_index(self, index: int) -> tuple[int, int, np.random.Generator]:
        """Deterministic ``(episode_slot, item_seed, rng)`` for a dataset index."""
        episodes = self.split_episodes()
        episode_slot = (int(index) // self.samples_per_episode) % len(episodes)
        item_seed = self.seed + int(index) * 104729
        return episode_slot, item_seed, np.random.default_rng(item_seed)

    def __getitem__(self, index: int) -> RawVLASample:
        episodes = self.split_episodes()
        episode_slot, _, rng = self.indices_for_index(index)
        episode = episodes[episode_slot]
        arrays = self._load_episode(episode.path)
        num_frames = episode.num_frames
        window = self.window
        rnn_frames = window.rnn_frames
        burst_frames = window.burst_frames

        current = int(rng.integers(0, num_frames))
        observation_indices = current - window.observation_stride * np.arange(
            rnn_frames - 1, -1, -1
        )
        clipped_observations = np.clip(observation_indices, 0, num_frames - 1)
        burst_offsets = np.arange(burst_frames - 1, -1, -1)
        burst_indices = np.clip(
            clipped_observations[:, None] - burst_offsets[None, :], 0, num_frames - 1
        )

        # A frame is new evidence only if its source index has not already been used.
        seen: set[int] = set()
        burst_new = np.zeros(burst_indices.shape, dtype=bool)
        for row in range(rnn_frames):
            for column in range(burst_frames):
                value = int(burst_indices[row, column])
                burst_new[row, column] = value not in seen
                seen.add(value)
        # The outer-axis flag follows the newest burst frame of each observation,
        # which is exactly the index the author's outer axis addresses.
        new_measurement = burst_new[:, burst_frames - 1].copy()
        unique_measurement_count = int(burst_new.sum())

        action_start = current + window.action_offset
        raw_action_indices = np.arange(action_start, action_start + window.action_horizon)
        supervision_indices = np.clip(raw_action_indices, 0, num_frames - 1)
        supervision_mask = (raw_action_indices >= 0) & (raw_action_indices <= num_frames - 1)

        state_key = window.state_source_key
        if state_key not in arrays:
            raise KeyError(f"state_source_key={state_key!r} missing from {episode.path}")
        state_read_index = int(np.clip(current, 0, num_frames - 1))
        raw_state = np.asarray(arrays[state_key][state_read_index], dtype=np.float32)
        action_block = np.asarray(arrays["actions"][supervision_indices], dtype=np.float32)

        actions, _, normalisation = normalise_actions(
            action_block,
            raw_state,
            self.stats,
            dataset_format=self.dataset_format,
        )
        state_vector, state_metadata = normalise_state(raw_state, self.stats)
        normalisation.update(state_metadata)

        burst_values: list[np.ndarray] = []
        pixel_valid: list[np.ndarray] = []
        bad_masks: list[np.ndarray] = []
        saturated_masks: list[np.ndarray] = []
        frame_ids: list[tuple[tuple[str, ...], ...]] = []
        range_evidence: list[dict[str, Any]] = []
        variances: list[np.ndarray | None] = []
        camera_units: list[dict[str, Any]] = []
        for camera_slot, (key, camera_id) in enumerate(
            zip(self.spec.measurement_keys, self.spec.camera_ids, strict=True)
        ):
            if key not in arrays:
                raise KeyError(f"measurement key {key!r} missing from {episode.path}")
            source = arrays[key]
            channels = self.spec.measurement_channels
            if channels == 1:
                # A genuine mosaic: one photosite per element, no channel axis.
                if source.ndim != 3:
                    raise ValueError(
                        f"{key!r} is declared as a 1-channel mosaic but has shape "
                        f"{source.shape}; expected [T,H,W]"
                    )
                source = source[..., None]
            if source.ndim != 4:
                raise ValueError(f"{key!r} must be [T,H,W,C], got {source.shape}")
            frames = np.moveaxis(np.asarray(source[burst_indices]), -1, -3)  # [R,T,C,H,W]
            if self.cfa_pattern:
                # The 2x2 rearrangement halves both spatial axes, so the output is
                # four planes of H/2 x W/2 - never H x W.
                converted = np.empty(
                    (rnn_frames, burst_frames, 4, frames.shape[-2] // 2, frames.shape[-1] // 2),
                    dtype=np.float32,
                )
                # Seed with the declared CFA facts, then let `_calibrate` add the
                # actual conversion record; dropping its metadata would hide whether
                # any calibration was applied at all.
                unit_metadata: dict[str, Any] = self._camera_cfa_metadata(key)
                evidence: dict[str, Any] = {}
                for row in range(rnn_frames):
                    for column in range(burst_frames):
                        converted[row, column], per_frame, evidence = self._calibrate(
                            key, frames[row, column], source.dtype, source
                        )
                        unit_metadata.update(per_frame)
                values = converted
            else:
                unit_metadata = {"camera_key": key}
                values, unit_metadata, evidence = self._calibrate(key, frames, source.dtype, source)
            values = np.ascontiguousarray(values, dtype=np.float32)
            # `values` is [R, T, C, H, W]: a sample is valid only when every
            # channel is finite, so the reduction is over the channel axis.
            valid = np.isfinite(values).all(axis=2)
            saturated = (
                (np.asarray(frames) >= 255).any(axis=2)
                if source.dtype == np.uint8
                else np.zeros(valid.shape, dtype=bool)
            )
            variance = self._camera_variance(values, valid)
            burst_values.append(values)
            pixel_valid.append(valid)
            bad_masks.append(~valid)
            saturated_masks.append(saturated)
            variances.append(variance)
            range_evidence.append(evidence)
            # [R][T]: one frame identity per (observation step, burst position).
            frame_ids.append(
                tuple(
                    tuple(
                        f"{episode.episode_group_id}#frame{int(burst_indices[row, column]):06d}"
                        for column in range(burst_frames)
                    )
                    for row in range(rnn_frames)
                )
            )
            camera_units.append(
                {
                    "camera_id": camera_id,
                    "camera_slot": camera_slot,
                    "metadata": unit_metadata,
                    "sample_xy": describe_frame_sampling(
                        values.shape[2], cfa_pattern=self.cfa_pattern
                    ),
                    "planes": self._plane_payload(key, values, camera_id),
                    "range_evidence": evidence,
                }
            )

        padding = float(1.0 - supervision_mask.mean()) if supervision_mask.size else 0.0
        provenance: dict[str, Any] = {
            "episode_path": episode.path,
            "episode_group_id": episode.episode_group_id,
            "acquisition_source": episode.acquisition_source,
            "num_frames": num_frames,
            "frame_index_unit": "frame",
            "timestamps_available": False,
            "executed_action_available": bool(window.executed_action_available),
            "cfa_available": bool(self.cfa_pattern),
            "rgb_image_output": False,
            "rgb_reconstruction_target": False,
            "window": dataclasses.asdict(window),
            "cameras": camera_units,
            "noise": self._noise_provenance(),
            "task_description": self._task_description(arrays),
        }
        return RawVLASample(
            dataset_format=self.dataset_format,
            split=self.split,
            episode=episode,
            camera_ids=tuple(self.spec.camera_ids),
            observation_indices=clipped_observations.astype(np.int64),
            burst_indices=burst_indices.astype(np.int64),
            new_measurement=new_measurement,
            burst_new_measurement=burst_new,
            unique_measurement_count=unique_measurement_count,
            burst_values=tuple(burst_values),
            pixel_valid_mask=tuple(pixel_valid),
            bad_mask=tuple(bad_masks),
            saturated_mask=tuple(saturated_masks),
            frame_ids=tuple(frame_ids),
            supervision_actions=actions,
            supervision_indices=supervision_indices.astype(np.int64),
            supervision_mask=supervision_mask,
            padding_contribution=padding,
            proprio=state_vector,
            proprio_available=state_vector is not None,
            executed_action=None,
            executed_action_available=bool(window.executed_action_available),
            variance=tuple(variances),
            range_evidence=tuple(range_evidence),
            normalisation=normalisation,
            provenance=provenance,
        )

    def _plane_payload(
        self, key: str, values: np.ndarray, camera_id: str
    ) -> list[dict[str, Any]] | None:
        """CFA path only: per-plane dicts with true sampling offsets and semantics."""
        if not self.cfa_pattern:
            return None
        positions = cfa_plane_positions(self.cfa_pattern)
        semantics = {
            "RGGB": "R at (0,0), G at (0,1) and (1,0), B at (1,1)",
            "BGGR": "B at (0,0), G at (0,1) and (1,0), R at (1,1)",
            "GRBG": "G at (0,0), R at (0,1), B at (1,0), G at (1,1)",
            "GBRG": "G at (0,0), B at (0,1), R at (1,0), G at (1,1)",
        }[self.cfa_pattern]
        planes = []
        for plane_index, name in enumerate(PLANE_COLOURS):
            planes.append(
                {
                    "name": name,
                    "values": values[:, :, plane_index],
                    "sample_xy": _plane_sample_xy(positions[plane_index]),
                    "cfa_semantics": semantics,
                    "camera_id": camera_id,
                    "camera_key": key,
                    "step_yx": [2.0, 2.0],
                    "pixel_centre_convention": PIXEL_CENTRE_CONVENTION,
                }
            )
        return planes

    def _load_cfa_frame(self, mosaic: np.ndarray, frame_index: int) -> np.ndarray:
        """Rearrange one mosaic frame into ``[4, H//2, W//2]`` planes.

        Used by the CFA path and by tests that need to rebuild the planes from the
        cached array without going through a full window sample.
        """
        if not self.cfa_pattern:
            raise ValueError("cfa_pattern is not set; the CFA plane path is inactive")
        planes = cfa_plane_values(np.asarray(mosaic)[frame_index], self.cfa_pattern)
        return planes.values

    def _camera_variance(self, values: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
        """Declared variance in the same units, or ``None`` when nothing is declared."""
        if not self.noise.declared:
            return None
        model = self.noise
        if self.dataset_format == "libero" and not self.cfa_pattern:
            model = model.scaled_for(255.0, True)
        # `values` is [R, T, C, H, W] but the validity mask is per sample
        # [R, T, H, W]; add the channel axis so the mask broadcasts instead of
        # being read as extra spatial extent.
        variance, _ = sensor_variance(values, model, valid_mask=valid[:, :, None])
        return variance

    def _noise_provenance(self) -> dict[str, Any]:
        declared = self.noise.declared
        report: dict[str, Any] = {
            "noise_model_declared": declared,
            "noise_source": self.noise.source,
            "variance_expression": "variance = shot * max(x, 0) + read_var",
            "variance_is_estimate": not declared,
            "calibrated_probability": False,
        }
        if declared:
            model = self.noise
            if self.dataset_format == "libero" and not self.cfa_pattern:
                model = model.scaled_for(255.0, True)
            report.update(
                {
                    "shot": float(model.shot),
                    "read_var": float(model.read_var),
                    "units": "normalised_value_units",
                    "scaling_note": (
                        "LIBERO uint8 was divided by 255, so DN shot/read coefficients are "
                        "divided by 255 / 255**2; RoboTwin float32 coefficients apply as given."
                    ),
                }
            )
            return report
        report.update(
            {
                "shot": None,
                "read_var": None,
                "estimate_scale": None,
                "estimator": "variance of valid samples in normalised units",
                "note": (
                    "No noise model was declared, so no variance array is fabricated. The "
                    "fitted scale is computed on demand by fitted_variance_estimate() and "
                    "marked as an estimate; energies built from it are relative explanation "
                    "weights, not posterior probabilities."
                ),
            }
        )
        return report

    def _task_description(self, arrays: dict[str, np.ndarray]) -> str:
        payload = arrays.get("metadata_json")
        if payload is None:
            return ""
        try:
            metadata = json.loads(str(np.asarray(payload).item()))
        except (ValueError, TypeError, json.JSONDecodeError):
            return ""
        if not isinstance(metadata, dict):
            return ""
        if metadata.get("task_description"):
            return str(metadata["task_description"])
        if self.dataset_format == "robotwin" and metadata.get("task_name"):
            return str(metadata["task_name"]).replace("_", " ")
        return ""

    def sample_summary(self, sample: RawVLASample) -> dict[str, Any]:
        """Small JSON-ready summary of one window (no arrays)."""
        plane_names = None
        if sample.provenance["cameras"][0].get("planes"):
            plane_names = [plane["name"] for plane in sample.provenance["cameras"][0]["planes"]]
        return {
            "dataset_format": sample.dataset_format,
            "split": sample.split,
            "episode_group_id": sample.episode.episode_group_id,
            "acquisition_source": sample.episode.acquisition_source,
            "num_frames": sample.episode.num_frames,
            "observation_indices": sample.observation_indices.tolist(),
            "burst_indices": sample.burst_indices.tolist(),
            "new_measurement": sample.new_measurement.tolist(),
            "unique_measurement_count": sample.unique_measurement_count,
            "burst_shape": [list(value.shape) for value in sample.burst_values],
            "supervision_indices": sample.supervision_indices.tolist(),
            "supervision_mask": sample.supervision_mask.tolist(),
            "padding_contribution": sample.padding_contribution,
            "proprio_available": sample.proprio_available,
            "executed_action_available": sample.executed_action_available,
            "variance_available": sample.variance[0] is not None,
            "range_evidence": list(sample.range_evidence),
            "cfa_plane_names": plane_names,
            "normalisation": {
                key: value
                for key, value in sample.normalisation.items()
                if isinstance(value, (str, int, float, bool, type(None), list))
            },
            "value_ranges": [
                [float(value.min()), float(value.max())] for value in sample.burst_values
            ],
        }


def _plane_sample_xy(position: np.ndarray) -> np.ndarray:
    """``[3, 2]`` ``(y, x)`` preview for one CFA plane (first sample, step, origin)."""
    array = np.zeros((3, 2), dtype=np.float32)
    array[0] = position.astype(np.float32)
    array[1] = np.float32(2.0)
    return array


# --------------------------------------------------------------------------------------
# report + CLI
# --------------------------------------------------------------------------------------


def build_report(
    cache_root: str | Path,
    dataset_format: str,
    *,
    split_seed: int = 0,
    val_fraction: float = 0.25,
    stats_path: str | Path | None = None,
    stats_key: str = "",
    window: WindowOptions | None = None,
    max_episodes: int | None = None,
    noise: NoiseModel | None = None,
    probe_sample: bool = True,
) -> dict[str, Any]:
    """Assemble the data report. Reports only what was actually inspected."""
    resolved_window = (window or WindowOptions()).resolved_for(FORMAT_SPECS[dataset_format])
    report: dict[str, Any] = {
        "generator": "owac.raw.rawvla_sequence",
        "dataset_format": dataset_format,
        "cache_root": str(cache_root),
        "window": dataclasses.asdict(resolved_window),
        "missing_assets": [],
    }
    evidence = inspect_cache(cache_root, dataset_format)
    report["inspection"] = evidence
    if evidence["status"] != "ok":
        report["status"] = "no_data_found"
        report["missing_assets"].append(
            {
                "asset": "RawVLA replay NPZ cache",
                "expected": str(cache_root),
                "server_path": (
                    "/workspace/owac/rawvla/data/libero/raw"
                    if dataset_format == "libero"
                    else "/workspace/owac/rawvla/data/robotwin2/raw"
                ),
                "evidence": (
                    "research/experiments/results/2026-10-10_rawvla_pi05_deployment/"
                    "dataset-check.json"
                ),
            }
        )
        report["split"] = None
        return report

    stats: NormalisationStats | None = None
    if stats_path is not None:
        try:
            stats = NormalisationStats.from_json(
                stats_path,
                mode=FORMAT_SPECS[dataset_format].default_action_normalisation,
                stats_key=stats_key,
            )
            report["normalisation"] = {
                "mode": stats.mode,
                "source": stats.source_path,
                "sha256": stats.sha256,
            }
        except (OSError, ValueError, KeyError) as error:
            report["missing_assets"].append(
                {
                    "asset": "normalisation statistics",
                    "expected": str(stats_path),
                    "error": str(error),
                }
            )
    dataset = RawVLAWindowDataset(
        cache_root,
        dataset_format=dataset_format,
        window=resolved_window,
        stats=stats,
        noise=noise,
        split_seed=split_seed,
        val_fraction=val_fraction,
        max_episodes=max_episodes,
    )
    episodes = dataset.episodes()
    manifest = dataset.split_manifest()
    report["episodes"] = {
        "count": len(episodes),
        "total_frames": sum(episode.num_frames for episode in episodes),
        "sources": sorted({episode.acquisition_source for episode in episodes}),
    }
    report["split"] = {
        "version": manifest.version,
        "seed": manifest.seed,
        "val_fraction": manifest.val_fraction,
        "split_unit": "acquisition_source",
        "train": {
            "sources": list(manifest.train_sources),
            "episodes": len(manifest.train_episodes),
            "frames": manifest.num_frames.get("train", 0),
        },
        "validation": {
            "sources": list(manifest.val_sources),
            "episodes": len(manifest.val_episodes),
            "frames": manifest.num_frames.get("validation", 0),
        },
        "problems": manifest.check(),
    }
    if probe_sample:
        probe: dict[str, Any] = {}
        for side in ("train", "validation"):
            try:
                dataset.split = side
                probe[side] = dataset.sample_summary(dataset[0])
            except (KeyError, ValueError, IndexError) as error:
                probe[side] = {"error": str(error)}
        dataset.split = "train"
        report["probe_sample"] = probe
    report["status"] = "ok"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m owac.raw.rawvla_sequence",
        description=(
            "Inspect a RawVLA replay NPZ cache and write a small JSON data report. Never "
            "invents numbers: with no cache present it reports no_data_found."
        ),
    )
    parser.add_argument("--cache-root", required=True, help="directory holding episode .npz files")
    parser.add_argument(
        "--format", dest="dataset_format", choices=sorted(FORMAT_SPECS), required=True
    )
    parser.add_argument("--report", default="", help="JSON report path; empty prints to stdout")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split-manifest", default="", help="also write the split manifest JSON")
    parser.add_argument("--stats-path", default="", help="normalisation statistics JSON")
    parser.add_argument("--stats-key", default="", help="dotted key inside the statistics JSON")
    parser.add_argument("--observation-stride", type=int, default=0)
    parser.add_argument("--rnn-frames", type=int, default=1)
    parser.add_argument("--burst-frames", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=0)
    parser.add_argument("--action-offset", type=int, default=0)
    parser.add_argument("--state-source-key", default="")
    parser.add_argument("--cfa-pattern", default="", choices=["", *sorted(CFA_PATTERNS)])
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all")
    parser.add_argument("--shot", type=float, default=None, help="declared shot coefficient")
    parser.add_argument("--read-var", type=float, default=None, help="declared read variance")
    args = parser.parse_args(argv)

    window = WindowOptions(
        observation_stride=args.observation_stride or 1,
        rnn_frames=args.rnn_frames,
        burst_frames=args.burst_frames,
        action_horizon=args.action_horizon or 1,
        action_offset=args.action_offset,
        state_source_key=args.state_source_key,
    )
    noise = None
    if args.shot is not None or args.read_var is not None:
        noise = NoiseModel(
            shot=float(args.shot or 0.0),
            read_var=float(args.read_var or 0.0),
            declared=True,
            source="cli_declared",
        )
    report = build_report(
        args.cache_root,
        args.dataset_format,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        stats_path=args.stats_path or None,
        stats_key=args.stats_key,
        window=window,
        max_episodes=args.max_episodes or None,
        noise=noise,
    )
    if args.cfa_pattern:
        report["cfa_pattern_requested"] = args.cfa_pattern
    if args.split_manifest and report.get("split") is not None:
        dataset = RawVLAWindowDataset(
            args.cache_root,
            dataset_format=args.dataset_format,
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
            max_episodes=args.max_episodes or None,
        )
        digest = write_split_manifest(dataset.split_manifest(), args.split_manifest)
        report["split_manifest"] = {"path": args.split_manifest, "sha256": digest}
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {target} (status={report['status']})")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

"""RAW observation encoding and masked RAW pretraining for the N02 full (v1.1) line.

This module is the P1 observation layer (``docs/prompts/n02_v1_1/P1_OBSERVATION.md``
section 2, interface frozen by ``docs/prompts/n02_v1_1/p1_interface_freeze.md`` section
3.1):

* :class:`RawObservationEncoder` runs the P0-parity ConvNeXt V2 RAW backbone and projects
  the stride 8/16 stages to ``projection_dim`` (512) wide levels. Every level carries its
  own validity map and the sensor pixel coordinates of its cells.
* :class:`RawFCMAEPretrainer` implements masked RAW reconstruction with the P1 starting
  hyper-parameters (``mask_ratio=0.6``, patch 32, decoder width 512 / depth 1,
  ``norm_pix_loss=False``). Its reconstruction exists for the training loss only and is
  never part of an encoder or readout path.

Declared conventions
--------------------

**Input modes.** ``raw3`` consumes three co-sited pseudo-RAW channels; ``cfa4`` consumes
either a single-plane Bayer mosaic (packed here into four planes) or four already packed
CFA planes. The mode is an explicit constructor argument: no code path converts one mode
into the other, no RGB/sRGB image is produced anywhere, and no RGB weight is reused. A
one-channel frame in ``raw3`` mode and a three-channel frame in ``cfa4`` mode are both
errors rather than implicit conversions.

**Packed CFA grid.** ``cfa4`` keeps the four Bayer planes (R, G1, G2, B) at half
resolution: a 512x512 sensor crop becomes a 256x256 *packed* grid. Coordinates are always
reported in **sensor** pixels, never in packed cells. Packed cell ``p`` holds the sensor
samples at ``2p`` and ``2p+1``; its centre is sensor coordinate ``2*(p+0.5)`` under the
``+0.5`` pixel-centre convention shared with ``owac.raw.rawvla_sequence_n02``.

**Batch axis.** ``MeasurementFrame.values`` is read batch-first: a batched frame is
``(B, C, H, W)``, so a batch of CFA4 mosaics is ``(B, 1, H, W)`` and a batch of packed CFA4
frames is ``(B, 4, H, W)``. The single-sample spellings ``(C, H, W)`` and ``(H, W)`` stay
accepted, but a three-dimensional leading axis is read as the batch whenever it is not a
legal channel count for the mode or the frame declares exactly that many ``episode_ids``;
a B>=2 ``(B, H, W)`` mosaic batch is never misread as ``(C, H, W)``.

**Coordinates.** ``sensor_coordinates[b, i, j] = (y, x)`` is the centre of the sensor
window that level cell ``(i, j)`` covers: a cell at stride ``s`` covers frame cells
``[s*i, s*(i+1))`` and reports ``crop + packed_scale * (pad + s*(i+0.5))``, where
``packed_scale`` is 1 for ``raw3`` and 2 for ``cfa4``. The frame's declared crop origin
(``metadata["camera"]["crop_offset"]``) shifts the whole mapping, so a cropped window keeps
true sensor coordinates.

**Padding and crops.** Rectangular frames are padded on the bottom/right to the backbone's
total stride (32) so that every stage grid is exact; the applied padding, the declared crop
origin and the sensor resolution are recorded in a :class:`FrameTransform` and attached to
the returned :class:`FeaturePyramid`. The encoder never crops and never stretches a frame;
the 256x256 square crop stays a pre-registered training augmentation owned by the data
layer, and generic resizing is never applied.

**Validity.** ``FeatureLevel.valid[b, i, j]`` is true iff the cell's stride-aligned input
window contains at least one contributing measurement (declared valid and not bad). It is a
support statement about that window, not a claim about the network's full receptive field.
Feature values at invalid cells are exactly zero.

**FCMAE causality.** The pretraining mask depends only on the current frame's own identity
(``episode_id``, ``camera_id``, ``source_frame_id``), the current frame's own support and
the configured mask ratio. Pixel content, later frames and any truth/evaluation field
cannot change it: pixels never enter the mask, and later frames are never read. Missing,
bad and saturated samples never enter the reconstruction loss; saturated samples are
counted separately so the density model can handle their tail probability.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from owac.core.encoder_contract_n02 import MeasurementFrame
from owac.frameworks.n02_full.components.convnextv2 import (
    DEFAULT_DEPTHS,
    DEFAULT_DIMS,
    STAGE_STRIDES,
    RawConvNeXtV2,
)
from owac.frameworks.n02_full.components.fcmae import (
    RawFCMAE,
    dilate_support,
    gen_random_mask,
    unpatchify,
    upsample_patch_mask,
)
from owac.frameworks.n02_full.state import FeatureLevel, ObservationBundle
from owac.frameworks.n02_full.synthetic import cfa_plane_tile_indices

__all__ = [
    "CAMERA_LEVEL_SEPARATOR",
    "CFA4",
    "FCMAE_MASK_RATIO",
    "FCMAE_PATCH_SIZE",
    "INPUT_MODES",
    "LEVEL_STRIDES",
    "PROJECTION_DIM",
    "RAW3",
    "STRIDE_ALIGNMENT",
    "FCMAEOutput",
    "FeaturePyramid",
    "FrameTransform",
    "ObservationTokens",
    "RawFCMAEPretrainer",
    "RawObservationEncoder",
    "frame_transform",
    "mask_seed",
    "packed_to_sensor_coordinates",
    "sensor_to_packed_coordinates",
]

#: Three co-sited pseudo-RAW channels (RawVLA triplet), no CFA phase.
RAW3 = "raw3"
#: Four Bayer planes (R, G1, G2, B) on the packed half-resolution grid.
CFA4 = "cfa4"

#: The two declared input modes. They are never converted into each other.
INPUT_MODES = (RAW3, CFA4)

#: Feature level ID -> stride in frame cells. ``raw3`` cells are sensor pixels; ``cfa4``
#: cells are packed 2x2 cells, so the sensor stride is twice as large.
LEVEL_STRIDES = {"stride8": 8, "stride16": 16}

#: Stage index of every returned level in the four-stage ConvNeXt V2 layout.
LEVEL_STAGES = {"stride8": 1, "stride16": 2}

#: Width of every projected feature level (contract section 5).
PROJECTION_DIM = 512

#: Total backbone stride; frames are padded on the bottom/right to a multiple of it.
STRIDE_ALIGNMENT = 32

#: FCMAE starting hyper-parameters fixed by P1 section 2.
FCMAE_PATCH_SIZE = 32
FCMAE_MASK_RATIO = 0.6

#: Separator used when one pyramid carries more than one camera. Camera IDs must not
#: contain it; a single-camera pyramid keeps the plain ``stride8``/``stride16`` keys.
CAMERA_LEVEL_SEPARATOR = "::"

_LEVEL_IDS = ("stride8", "stride16")


def _check_mode(mode: str) -> str:
    if mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {mode!r}")
    return str(mode)


def _packed_scale(mode: str) -> int:
    """Sensor pixels per frame cell: 1 for ``raw3``, 2 for the packed ``cfa4`` grid."""
    return 2 if _check_mode(mode) == CFA4 else 1


def _channels_for_mode(mode: str, in_chans: int | None) -> int:
    default = 4 if mode == CFA4 else 3
    if in_chans is None:
        return default
    channels = int(in_chans)
    allowed = (1, 4) if mode == CFA4 else (3,)
    if channels not in allowed:
        raise ValueError(
            f"{mode} mode accepts input channels {allowed}, got {channels}; the two input "
            "modes are configured explicitly and are never converted into each other"
        )
    return channels


def _shape_of(values: Any) -> tuple[int, ...]:
    if hasattr(values, "shape"):
        return tuple(int(item) for item in values.shape)
    return tuple(int(item) for item in np.shape(values))


def _as_coord_tensor(coords: Any) -> torch.Tensor:
    tensor = coords if isinstance(coords, torch.Tensor) else torch.as_tensor(coords)
    tensor = tensor.to(torch.float64)
    if tensor.ndim < 1 or tensor.shape[-1] != 2:
        raise ValueError(
            f"coordinates must have a trailing (y, x) axis of size 2, got {tuple(tensor.shape)}"
        )
    return tensor


def packed_to_sensor_coordinates(
    coords: Any,
    *,
    mode: str,
    pad_before: tuple[int, int] = (0, 0),
    crop_offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Frame-cell coordinates -> sensor pixel coordinates (``+0.5`` convention, float64).

    ``coords`` is any shape with a trailing ``(y, x)`` axis whose values are frame-cell
    indices (either packed CFA cells or sensor pixels in ``raw3``). ``pad_before`` is the
    padding applied before the frame (in frame cells) and ``crop_offset`` is the origin of
    the frame window in the full sensor (in sensor pixels).
    """
    scale = _packed_scale(mode)
    cells = _as_coord_tensor(coords)
    out = cells.clone()
    out[..., 0] = float(crop_offset[0]) + scale * (float(pad_before[0]) + cells[..., 0] + 0.5)
    out[..., 1] = float(crop_offset[1]) + scale * (float(pad_before[1]) + cells[..., 1] + 0.5)
    return out


def sensor_to_packed_coordinates(
    coords: Any,
    *,
    mode: str,
    pad_before: tuple[int, int] = (0, 0),
    crop_offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Inverse of :func:`packed_to_sensor_coordinates` (float64 frame-cell indices)."""
    scale = _packed_scale(mode)
    cells = _as_coord_tensor(coords)
    out = cells.clone()
    out[..., 0] = (cells[..., 0] - float(crop_offset[0])) / scale - float(pad_before[0]) - 0.5
    out[..., 1] = (cells[..., 1] - float(crop_offset[1])) / scale - float(pad_before[1]) - 0.5
    return out


def mask_seed(episode_id: str, camera_id: str, source_frame_id: str) -> int:
    """Seed of an FCMAE mask, derived from the frame's own identity only.

    The three identity strings are the only inputs. Pixel values, other frames, the
    frame's masks and any truth/evaluation field cannot influence the result, so a mask
    is reproducible from identity alone and cannot read the future.
    """
    payload = f"n02-fcmae-mask-v1|{episode_id}|{camera_id}|{source_frame_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


# --------------------------------------------------------------------------------------
# frame geometry
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class FrameTransform:
    """Record of how one frame was mapped into the encoder grid (all in frame cells).

    ``frame_shape`` is the frame's own grid (packed cells for ``cfa4``), ``padded_shape``
    the grid fed to the backbone, and ``pad_before``/``pad_after`` the applied padding.
    ``crop_offset`` is the declared origin of the frame window in the full sensor, in
    sensor pixels, and ``sensor_shape`` the sensor resolution the samples come from.
    """

    camera_id: str
    input_mode: str
    sensor_shape: tuple[int, int]
    frame_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    pad_before: tuple[int, int]
    pad_after: tuple[int, int]
    crop_offset: tuple[int, int]
    packed_scale: int
    alignment: int
    cfa_pattern: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "input_mode": self.input_mode,
            "sensor_shape": list(self.sensor_shape),
            "frame_shape": list(self.frame_shape),
            "padded_shape": list(self.padded_shape),
            "pad_before": list(self.pad_before),
            "pad_after": list(self.pad_after),
            "crop_offset": list(self.crop_offset),
            "packed_scale": self.packed_scale,
            "alignment": self.alignment,
            "cfa_pattern": self.cfa_pattern,
        }


def _metadata_camera(frame: MeasurementFrame) -> Mapping:
    metadata = frame.metadata if isinstance(frame.metadata, Mapping) else {}
    camera = metadata.get("camera")
    return camera if isinstance(camera, Mapping) else {}


def _metadata_lookup(frame: MeasurementFrame, key: str) -> Any:
    """Read ``metadata[key]`` with the per-camera block taking precedence."""
    metadata = frame.metadata if isinstance(frame.metadata, Mapping) else {}
    camera = _metadata_camera(frame)
    if key in camera:
        return camera[key]
    return metadata.get(key)


def _cfa_pattern(frame: MeasurementFrame) -> str:
    pattern = _metadata_lookup(frame, "cfa_pattern")
    return "" if pattern is None else str(pattern)


def _crop_offset(frame: MeasurementFrame) -> tuple[int, int]:
    offset = _metadata_lookup(frame, "crop_offset")
    if offset is None:
        return (0, 0)
    values = tuple(int(item) for item in offset)
    if len(values) != 2:
        raise ValueError(f"crop_offset must be a (row, col) pair, got {offset!r}")
    return values


def _declared_sensor_shape(frame: MeasurementFrame) -> tuple[int, int] | None:
    declared = _metadata_lookup(frame, "sensor_shape")
    if declared is None:
        return None
    values = tuple(int(item) for item in declared)
    if len(values) != 2:
        raise ValueError(f"sensor_shape must be a (height, width) pair, got {declared!r}")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"sensor_shape must be positive, got {declared!r}")
    return values


@dataclass(slots=True)
class _Geometry:
    batch: int
    channels: int
    frame_shape: tuple[int, int]
    packed_scale: int
    sensor_shape: tuple[int, int]
    crop_offset: tuple[int, int]
    cfa_pattern: str


def _frame_layout(frame: MeasurementFrame, mode: str) -> tuple[int, int, int, int]:
    """Interpret ``frame.values`` as ``(B, C, H, W)`` with the batch axis always first.

    The contract (freeze section 3.9) fixes a batched frame as ``(B, C, H, W)``: a CFA4
    mosaic row carries ``C=1`` and a packed CFA4 row carries ``C=4``, and the leading axis
    is the batch. The legacy single-sample spellings ``(C, H, W)`` and ``(H, W)`` stay
    accepted, so a three-dimensional leading axis is read as batch whenever it cannot be a
    legal channel count for the mode -- or whenever the frame declares exactly that many
    batch rows (``len(episode_ids) == leading > 1``). A B>=2 ``(B, H, W)`` mosaic batch is
    therefore never misread as ``(C, H, W)``.
    """
    shape = _shape_of(frame.values)
    legal_channels = (1, 4) if _check_mode(mode) == CFA4 else (3,)
    if len(shape) == 4:
        batch, channels, height, width = shape
    elif len(shape) == 3:
        leading, height, width = shape
        declared_rows = len(frame.episode_ids)
        if leading in legal_channels and not (declared_rows == leading and leading > 1):
            batch, channels = 1, leading
        else:
            batch, channels = leading, 1
    elif len(shape) == 2:
        batch, channels, height, width = 1, 1, shape[0], shape[1]
    else:
        raise ValueError(f"frame values must be 2-D/3-D/4-D, got shape {shape}")
    if batch < 1 or channels < 1 or height < 1 or width < 1:
        raise ValueError(f"frame values must be non-empty, got shape {shape}")
    return int(batch), int(channels), int(height), int(width)


def _frame_geometry(frame: MeasurementFrame, mode: str) -> _Geometry:
    """Classify a frame's raw layout for the declared mode; never converts between modes."""
    batch, channels, height, width = _frame_layout(frame, mode)

    crop = _crop_offset(frame)
    pattern = _cfa_pattern(frame)
    if mode == CFA4:
        packed_scale = 2
        if channels == 1:
            if height % 2 or width % 2:
                raise ValueError(
                    f"a CFA mosaic needs an even height and width, got {height}x{width}"
                )
            if not pattern:
                raise ValueError(
                    "cfa4 mode needs a Bayer pattern in metadata['camera']['cfa_pattern'] to "
                    "pack a single-plane mosaic; raw3 and cfa4 are never converted implicitly"
                )
            frame_shape = (height // 2, width // 2)
        elif channels == 4:
            frame_shape = (height, width)
        else:
            raise ValueError(
                f"cfa4 mode accepts 1 mosaic channel or 4 packed CFA planes, got {channels}; "
                "raw3 and cfa4 are configured separately and are never converted into each other"
            )
    elif mode == RAW3:
        packed_scale = 1
        if channels != 3:
            raise ValueError(
                f"raw3 mode needs 3 co-sited channels, got {channels}; a mosaic belongs to the "
                "cfa4 mode and the two modes are never converted into each other"
            )
        frame_shape = (height, width)
    else:  # pragma: no cover - _check_mode rejects anything else first
        raise ValueError(f"unknown input mode {mode!r}")

    sensor_shape = _declared_sensor_shape(frame)
    if sensor_shape is None:
        sensor_shape = (
            crop[0] + packed_scale * frame_shape[0],
            crop[1] + packed_scale * frame_shape[1],
        )
    return _Geometry(
        batch=batch,
        channels=channels,
        frame_shape=frame_shape,
        packed_scale=packed_scale,
        sensor_shape=sensor_shape,
        crop_offset=crop,
        cfa_pattern=pattern,
    )


def _pad_plan(
    frame_shape: tuple[int, int], alignment: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    pad_after = ((-frame_shape[0]) % alignment, (-frame_shape[1]) % alignment)
    return (0, 0), pad_after


def frame_transform(
    frame: MeasurementFrame, *, input_mode: str, alignment: int = STRIDE_ALIGNMENT
) -> FrameTransform:
    """Pure record of the padding/crop mapping of one frame (no tensor work)."""
    mode = _check_mode(input_mode)
    geometry = _frame_geometry(frame, mode)
    pad_before, pad_after = _pad_plan(geometry.frame_shape, int(alignment))
    return FrameTransform(
        camera_id=str(frame.camera_id),
        input_mode=mode,
        sensor_shape=geometry.sensor_shape,
        frame_shape=geometry.frame_shape,
        padded_shape=(
            geometry.frame_shape[0] + pad_before[0] + pad_after[0],
            geometry.frame_shape[1] + pad_before[1] + pad_after[1],
        ),
        pad_before=pad_before,
        pad_after=pad_after,
        crop_offset=geometry.crop_offset,
        packed_scale=geometry.packed_scale,
        alignment=int(alignment),
        cfa_pattern=geometry.cfa_pattern,
    )


def _level_sensor_coordinates(
    grid: tuple[int, int], stride: int, transform: FrameTransform
) -> torch.Tensor:
    """``(H_l, W_l, 2)`` float32 sensor coordinates of every level cell centre."""
    height, width = int(grid[0]), int(grid[1])
    rows = (torch.arange(height, dtype=torch.float64) + 0.5) * float(stride) - 0.5
    cols = (torch.arange(width, dtype=torch.float64) + 0.5) * float(stride) - 0.5
    # the +0.5 turns a frame-cell index into the +0.5 pixel-centre coordinate
    rows = rows + float(transform.pad_before[0]) + 0.5
    cols = cols + float(transform.pad_before[1]) + 0.5
    grid_y = float(transform.crop_offset[0]) + transform.packed_scale * rows
    grid_x = float(transform.crop_offset[1]) + transform.packed_scale * cols
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    return torch.stack((mesh_y, mesh_x), dim=-1).to(torch.float32)


# --------------------------------------------------------------------------------------
# frame preparation (shared by the encoder and the pretrainer)
# --------------------------------------------------------------------------------------


def _frame_values(frame: MeasurementFrame, mode: str) -> torch.Tensor:
    """``(B, C, H, W)`` float32 view of a frame, using the batch-first layout rule."""
    values = frame.values
    if values is None:
        raise TypeError("frame.values must not be None")
    tensor = (
        values.detach() if isinstance(values, torch.Tensor) else torch.as_tensor(np.asarray(values))
    )
    batch, channels, height, width = _frame_layout(frame, mode)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0) if batch == 1 else tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f"frame values must be 2-D/3-D/4-D, got shape {tuple(tensor.shape)}")
    if tuple(tensor.shape) != (batch, channels, height, width):
        raise ValueError(
            f"frame values shape {tuple(tensor.shape)} does not match the batch-first layout "
            f"{(batch, channels, height, width)}"
        )
    return tensor.to(torch.float32)


def _mask_tensor(value: Any, batch: int, height: int, width: int, *, default: bool) -> torch.Tensor:
    """Normalise an optional ``(B,H,W)``/``(B,1,H,W)`` mask to bool ``(B,H,W)``."""
    if value is None:
        return torch.full((batch, height, width), bool(default), dtype=torch.bool)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"mask must be (B,H,W), got shape {tuple(tensor.shape)}")
    if tensor.shape[0] == 1 and batch > 1:
        tensor = tensor.expand(batch, -1, -1)
    if tuple(tensor.shape) != (batch, height, width):
        raise ValueError(
            f"mask shape {tuple(tensor.shape)} does not match the frame grid "
            f"{(batch, height, width)}"
        )
    return tensor.bool()


def _calibration(frame: MeasurementFrame) -> tuple[float, float] | None:
    metadata = frame.metadata if isinstance(frame.metadata, Mapping) else {}
    calibration = metadata.get("calibration")
    if not isinstance(calibration, Mapping):
        return None
    try:
        black = float(calibration["black_level"])
        white = float(calibration["white_level"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(black) or not np.isfinite(white) or white <= black:
        return None
    return black, white


def _pack_planes(volume: torch.Tensor, pattern: str) -> torch.Tensor:
    """``(B,1,H,W)`` mosaic -> ``(B,4,H/2,W/2)`` planes in R/G1/G2/B order."""
    tiles = cfa_plane_tile_indices(pattern)
    planes = []
    for tile in tiles:
        dy, dx = divmod(int(tile), 2)
        planes.append(volume[:, :, dy::2, dx::2])
    return torch.cat(planes, dim=1)


@dataclass(slots=True)
class _PreparedFrame:
    values: torch.Tensor
    contributing: torch.Tensor
    saturated: torch.Tensor
    valid: torch.Tensor
    missing: torch.Tensor
    bad: torch.Tensor
    transform: FrameTransform


def _prepare_frame(
    frame: MeasurementFrame,
    mode: str,
    *,
    alignment: int = STRIDE_ALIGNMENT,
    device: torch.device | None = None,
) -> _PreparedFrame:
    """Normalise, mask and stride-align one frame for the declared mode."""
    mode = _check_mode(mode)
    geometry = _frame_geometry(frame, mode)
    values = _frame_values(frame, mode)
    if values.shape[1] != geometry.channels:
        raise ValueError(
            f"frame declares {values.shape[1]} channels but the geometry expects "
            f"{geometry.channels}"
        )
    batch, channels, height, width = values.shape
    if device is not None:
        values = values.to(device)

    valid = _mask_tensor(frame.valid_mask, batch, height, width, default=True).to(values.device)
    bad = _mask_tensor(frame.bad_mask, batch, height, width, default=False).to(values.device)
    saturated = _mask_tensor(frame.saturated_mask, batch, height, width, default=False).to(
        values.device
    )
    missing = ~valid

    calibration = _calibration(frame)
    if calibration is not None:
        black, white = calibration
        values = (values - black) / (white - black)

    finite = torch.isfinite(values).all(dim=1)
    contributing = valid & ~bad & finite
    values = torch.where(contributing.unsqueeze(1), values, torch.zeros_like(values))

    if mode == CFA4 and channels == 1:
        values = _pack_planes(values, geometry.cfa_pattern)

        def _pack_mask(mask: torch.Tensor) -> torch.Tensor:
            packed = _pack_planes(mask.unsqueeze(1).to(values.dtype), geometry.cfa_pattern)
            return packed > 0.5

        valid = _pack_mask(valid)
        missing = _pack_mask(missing)
        bad = _pack_mask(bad)
        saturated = _pack_mask(saturated)
        contributing = _pack_mask(contributing)
    else:
        valid = valid.unsqueeze(1).expand(-1, channels, -1, -1).clone()
        missing = missing.unsqueeze(1).expand(-1, channels, -1, -1).clone()
        bad = bad.unsqueeze(1).expand(-1, channels, -1, -1).clone()
        saturated = saturated.unsqueeze(1).expand(-1, channels, -1, -1).clone()
        contributing = contributing.unsqueeze(1).expand(-1, channels, -1, -1).clone()

    transform = frame_transform(frame, input_mode=mode, alignment=alignment)
    pad_after = transform.pad_after
    if pad_after[0] or pad_after[1]:
        pad = (0, pad_after[1], 0, pad_after[0])
        values = F.pad(values, pad, value=0.0)
        valid = F.pad(valid.to(torch.uint8), pad, value=0).bool()
        missing = F.pad(missing.to(torch.uint8), pad, value=1).bool()
        bad = F.pad(bad.to(torch.uint8), pad, value=0).bool()
        saturated = F.pad(saturated.to(torch.uint8), pad, value=0).bool()
        contributing = F.pad(contributing.to(torch.uint8), pad, value=0).bool()

    return _PreparedFrame(
        values=values,
        contributing=contributing,
        saturated=saturated,
        valid=valid,
        missing=missing,
        bad=bad,
        transform=transform,
    )


def _row_identities(
    frame: MeasurementFrame, batch: int, *, required: bool
) -> list[tuple[str, str]]:
    """Per-row ``(episode_id, source_frame_id)``; one ID per batch row when required."""
    episodes = tuple(str(item) for item in frame.episode_ids)
    sources = tuple(str(item) for item in frame.frame_ids)

    def resolve(values: tuple[str, ...], name: str) -> tuple[str, ...]:
        if len(values) == batch:
            return values
        if not values and not required:
            return ("",) * batch
        if len(values) == 1 and batch == 1:
            return values
        if required:
            raise ValueError(
                f"{name} must carry one ID per batch row ({batch}), got {len(values)}; the FCMAE "
                "mask seed is derived from the frame identity and a missing identity is an error"
            )
        return ("",) * batch

    return list(zip(resolve(episodes, "episode_ids"), resolve(sources, "frame_ids"), strict=True))


def _support_digest(contributing: torch.Tensor) -> str:
    """Stable sha256 of a ``(B,H,W)`` support map (pixel content never enters)."""
    array = contributing.detach().to("cpu", torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ObservationTokens:
    """Flattened token view of a :class:`FeaturePyramid` (cameras on the token axis)."""

    values: torch.Tensor
    valid: torch.Tensor
    sensor_coordinates: torch.Tensor
    camera_index: torch.Tensor
    level_index: torch.Tensor
    camera_ids: tuple[str, ...]
    level_ids: tuple[str, ...]


@dataclass(slots=True)
class FeaturePyramid:
    """Multi-scale RAW observation for one decision time.

    ``levels`` is keyed by ``stride8``/``stride16``. When the bundle carries more than one
    camera the level keys are ``"<camera_id>::<level_id>"``: cameras are never folded into
    the batch axis, each camera keeps its own resolution, coordinates and validity. Use
    :meth:`per_camera` to read the pyramid without depending on the key format.
    """

    levels: dict[str, FeatureLevel]
    camera_ids: tuple[str, ...]
    input_mode: str
    sensor_shape: dict[str, tuple[int, int]]
    support_hash: str
    transforms: dict[str, FrameTransform] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.camera_ids = tuple(str(item) for item in self.camera_ids)
        self.input_mode = _check_mode(self.input_mode)
        self.sensor_shape = {
            str(key): (int(value[0]), int(value[1]))
            for key, value in dict(self.sensor_shape).items()
        }
        self.transforms = dict(self.transforms)
        self.support_hash = str(self.support_hash)
        if not isinstance(self.levels, dict) or not self.levels:
            raise ValueError("FeaturePyramid.levels must be a non-empty dict")
        if len(self.camera_ids) > 1:
            for key in self.levels:
                camera, separator, level_id = str(key).partition(CAMERA_LEVEL_SEPARATOR)
                if not separator or camera not in self.camera_ids or level_id not in LEVEL_STRIDES:
                    raise ValueError(
                        f"multi-camera level key {key!r} must be '<camera_id>"
                        f"{CAMERA_LEVEL_SEPARATOR}<level_id>'"
                    )

    @property
    def level_ids(self) -> tuple[str, ...]:
        return _LEVEL_IDS

    def per_camera(self) -> dict[str, dict[str, FeatureLevel]]:
        """``{camera_id: {level_id: FeatureLevel}}`` regardless of the key format."""
        grouped: dict[str, dict[str, FeatureLevel]] = {camera: {} for camera in self.camera_ids}
        single = len(self.camera_ids) == 1
        for key, level in self.levels.items():
            if single:
                camera, level_id = self.camera_ids[0], str(key)
            else:
                camera, _, level_id = str(key).partition(CAMERA_LEVEL_SEPARATOR)
            grouped.setdefault(camera, {})[level_id] = level
        return grouped

    def tokens(self) -> ObservationTokens:
        """Concatenate every camera's levels along the token axis, keeping camera identity.

        The token axis is ``(camera, level, row, column)``; per token the camera index into
        :attr:`camera_ids`, the level index into :attr:`level_ids` and the sensor
        coordinates are returned alongside the values and validity.
        """
        values: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        coordinates: list[torch.Tensor] = []
        camera_index: list[torch.Tensor] = []
        level_index: list[torch.Tensor] = []
        grouped = self.per_camera()
        for camera_position, camera in enumerate(self.camera_ids):
            for level_position, level_id in enumerate(_LEVEL_IDS):
                level = grouped[camera][level_id]
                if not isinstance(level, FeatureLevel):  # pragma: no cover - defensive
                    raise TypeError(f"level {level_id} of camera {camera} is not a FeatureLevel")
                batch, channels, height, width = level.values.shape
                values.append(
                    level.values.reshape(batch, channels, height * width).permute(0, 2, 1)
                )
                valid.append(level.valid.reshape(batch, height * width))
                coordinates.append(level.sensor_coordinates.reshape(batch, height * width, 2))
                camera_index.append(
                    torch.full((batch, height * width), camera_position, dtype=torch.long)
                )
                level_index.append(
                    torch.full((batch, height * width), level_position, dtype=torch.long)
                )
        first = values[0]
        if any(item.shape[0] != first.shape[0] for item in values):
            raise ValueError("every camera and level of one pyramid must share the episode batch")
        device = first.device
        return ObservationTokens(
            values=torch.cat(values, dim=1),
            valid=torch.cat(valid, dim=1),
            sensor_coordinates=torch.cat(coordinates, dim=1),
            camera_index=torch.cat(camera_index, dim=1).to(device),
            level_index=torch.cat(level_index, dim=1).to(device),
            camera_ids=self.camera_ids,
            level_ids=_LEVEL_IDS,
        )


def _projection_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


@dataclass(slots=True)
class _CameraEncoding:
    levels: dict[str, FeatureLevel]
    transform: FrameTransform
    support_digest: str
    identities: list[tuple[str, str]]


class RawObservationEncoder(nn.Module):
    """ConvNeXt V2 RAW backbone with stride 8/16 projection to ``projection_dim``.

    The backbone is the P0-parity :class:`RawConvNeXtV2` (official depths ``[3,3,27,3]``
    and dims ``[128,256,512,1024]`` by default, stride 8/16 returned). Frames are padded on
    the bottom/right to the backbone stride, validity is tracked by coordinate dilation and
    every level reports sensor pixel coordinates. No reconstruction head and no RGB path
    exist on this module.
    """

    def __init__(
        self,
        *,
        input_mode: str = RAW3,
        in_chans: int | None = None,
        depths: Sequence[int] = DEFAULT_DEPTHS,
        dims: Sequence[int] = DEFAULT_DIMS,
        projection_dim: int = PROJECTION_DIM,
        alignment: int = STRIDE_ALIGNMENT,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.input_mode = _check_mode(input_mode)
        self.in_chans = _channels_for_mode(self.input_mode, in_chans)
        self.alignment = int(alignment)
        if self.alignment <= 0 or self.alignment % STRIDE_ALIGNMENT:
            raise ValueError(
                f"alignment must be a positive multiple of the {STRIDE_ALIGNMENT}-pixel backbone "
                f"stride, got {alignment}"
            )
        depths = tuple(int(item) for item in depths)
        dims = tuple(int(item) for item in dims)
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("the observation backbone keeps the four-stage ConvNeXt V2 layout")
        self.depths = depths
        self.dims = dims
        if backbone is None:
            backbone = RawConvNeXtV2(
                in_chans=self.in_chans,
                depths=depths,
                dims=dims,
                num_classes=0,
                out_indices=(1, 2),
            )
        elif tuple(int(item) for item in getattr(backbone, "dims", dims)) != dims:
            raise ValueError("the supplied backbone dims do not match the declared dims")
        self.backbone = backbone
        self.projection_dim = int(projection_dim)
        self.proj = nn.ModuleDict(
            {
                "stride8": nn.Conv2d(dims[1], self.projection_dim, kernel_size=1),
                "stride16": nn.Conv2d(dims[2], self.projection_dim, kernel_size=1),
            }
        )
        self.proj.apply(_projection_init)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode_camera(self, frame: MeasurementFrame, camera_id: str) -> _CameraEncoding:
        prepared = _prepare_frame(
            frame, self.input_mode, alignment=self.alignment, device=self.device
        )
        features = self.backbone.forward_multiscale(prepared.values)
        support = prepared.contributing.any(dim=1, keepdim=True).to(prepared.values.dtype)
        supports = {STAGE_STRIDES[0]: dilate_support(support, kernel=4, stride=4)}
        for stride in STAGE_STRIDES[1:]:
            supports[stride] = dilate_support(supports[stride // 2], kernel=2, stride=2)

        transform = prepared.transform
        identities = _row_identities(frame, prepared.values.shape[0], required=False)
        levels: dict[str, FeatureLevel] = {}
        for level_id, stride in LEVEL_STRIDES.items():
            stage = LEVEL_STAGES[level_id]
            feature = features[f"stage{stage}_stride{stride}"]
            cell_support = supports[stride]
            values = self.proj[level_id](feature) * cell_support
            coordinates = _level_sensor_coordinates(values.shape[-2:], stride, transform)
            coordinates = (
                coordinates.unsqueeze(0)
                .expand(values.shape[0], -1, -1, -1)
                .to(values.device)
                .contiguous()
            )
            levels[level_id] = FeatureLevel(
                values=values,
                valid=cell_support.squeeze(1).bool(),
                sensor_coordinates=coordinates,
                support_ids=tuple(
                    f"{episode}|{camera_id}|{source}" for episode, source in identities
                ),
                level_id=level_id,
            )
        return _CameraEncoding(
            levels=levels,
            transform=transform,
            support_digest=_support_digest(prepared.contributing.any(dim=1)),
            identities=identities,
        )

    def encode_frame(self, frame: MeasurementFrame, *, camera_id: str) -> dict[str, FeatureLevel]:
        """Encode one camera's frame into ``{level_id: FeatureLevel}``."""
        if not isinstance(frame, MeasurementFrame):
            raise TypeError(f"encode_frame expects a MeasurementFrame, got {type(frame).__name__}")
        return self._encode_camera(frame, str(camera_id)).levels

    def encode(self, bundle: ObservationBundle) -> FeaturePyramid:
        """Encode every camera of one decision time; cameras are never folded into batch."""
        if not isinstance(bundle, ObservationBundle):
            raise TypeError(f"encode expects an ObservationBundle, got {type(bundle).__name__}")
        camera_ids = tuple(str(item) for item in bundle.camera_ids)
        if not camera_ids:
            raise ValueError("the bundle carries no camera frames")
        multi = len(camera_ids) > 1
        levels: dict[str, FeatureLevel] = {}
        transforms: dict[str, FrameTransform] = {}
        sensor_shape: dict[str, tuple[int, int]] = {}
        entries: list[dict[str, Any]] = []
        for camera_id in camera_ids:
            encoding = self._encode_camera(bundle.frames[camera_id], camera_id)
            transforms[camera_id] = encoding.transform
            sensor_shape[camera_id] = encoding.transform.sensor_shape
            entries.append(
                {
                    "camera_id": camera_id,
                    "input_mode": self.input_mode,
                    "identity": [list(item) for item in encoding.identities],
                    "sensor_shape": list(encoding.transform.sensor_shape),
                    "frame_shape": list(encoding.transform.frame_shape),
                    "pad_after": list(encoding.transform.pad_after),
                    "crop_offset": list(encoding.transform.crop_offset),
                    "support_sha256": encoding.support_digest,
                }
            )
            for level_id, level in encoding.levels.items():
                key = f"{camera_id}{CAMERA_LEVEL_SEPARATOR}{level_id}" if multi else level_id
                levels[key] = level
        support_hash = hashlib.sha256(
            _canonical(sorted(entries, key=lambda item: item["camera_id"])).encode("utf-8")
        ).hexdigest()
        return FeaturePyramid(
            levels=levels,
            camera_ids=camera_ids,
            input_mode=self.input_mode,
            sensor_shape=sensor_shape,
            support_hash=support_hash,
            transforms=transforms,
        )


# --------------------------------------------------------------------------------------
# FCMAE masked pretraining
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class FCMAEOutput:
    """Masked RAW reconstruction result.

    ``reconstruction`` and ``mask`` carry a leading camera axis in ``camera_ids`` order
    (``squeeze(0)`` for a single camera). ``loss`` is the mean squared error over the
    masked, contributing, non-saturated samples only; ``valid_target_count`` is that number
    of scalar samples and ``saturated_count`` the saturated target samples reported
    separately. The reconstruction is consumed by the training loss and nowhere else.
    """

    loss: torch.Tensor
    reconstruction: torch.Tensor
    mask: torch.Tensor
    valid_target_count: torch.Tensor
    saturated_count: torch.Tensor


class RawFCMAEPretrainer(nn.Module):
    """FCMAE masked-RAW reconstruction with the P1 starting hyper-parameters.

    ``mask_ratio=0.6``, ``patch_size=32``, ``decoder_embed_dim=512``, ``decoder_depth=1``
    and ``norm_pix_loss=False`` are the frozen starting values. The mask of a frame is
    generated from the frame's own identity and support only (see :func:`mask_seed`), so the
    same acquisition always yields the same mask and no future frame, truth or evaluation
    field can change it.
    """

    def __init__(
        self,
        *,
        input_mode: str = RAW3,
        in_chans: int | None = None,
        depths: Sequence[int] = DEFAULT_DEPTHS,
        dims: Sequence[int] = DEFAULT_DIMS,
        patch_size: int = FCMAE_PATCH_SIZE,
        mask_ratio: float = FCMAE_MASK_RATIO,
        decoder_embed_dim: int = PROJECTION_DIM,
        decoder_depth: int = 1,
        norm_pix_loss: bool = False,
        img_size: tuple[int, int] = (256, 256),
        alignment: int = STRIDE_ALIGNMENT,
        encoder: nn.Module | None = None,
        model: RawFCMAE | None = None,
    ):
        super().__init__()
        self.input_mode = _check_mode(input_mode)
        self.in_chans = _channels_for_mode(self.input_mode, in_chans)
        self.alignment = int(alignment)
        if self.alignment <= 0 or self.alignment % STRIDE_ALIGNMENT:
            raise ValueError(
                f"alignment must be a positive multiple of the {STRIDE_ALIGNMENT}-pixel backbone "
                f"stride, got {alignment}"
            )
        if int(patch_size) != STRIDE_ALIGNMENT:
            raise ValueError(
                f"the four-stage backbone has stride {STRIDE_ALIGNMENT}; patch_size must match it, "
                f"got {patch_size}"
            )
        if model is None:
            model = RawFCMAE(
                img_size=img_size,
                in_chans=self.in_chans,
                depths=depths,
                dims=dims,
                decoder_depth=decoder_depth,
                decoder_embed_dim=decoder_embed_dim,
                patch_size=int(patch_size),
                mask_ratio=float(mask_ratio),
                norm_pix_loss=bool(norm_pix_loss),
                encoder=encoder,
            )
        self.fcmae = model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def patch_size(self) -> int:
        return int(self.fcmae.patch_size)

    @property
    def mask_ratio(self) -> float:
        return float(self.fcmae.mask_ratio)

    def _build_mask(
        self, frame: MeasurementFrame, prepared: _PreparedFrame, camera_id: str
    ) -> torch.Tensor:
        patch = self.patch_size
        batch, _, height, width = prepared.values.shape
        grid = (height // patch, width // patch)
        if grid[0] < 1 or grid[1] < 1:
            raise ValueError(
                f"padded frame {height}x{width} is smaller than one {patch}-pixel patch"
            )
        length = grid[0] * grid[1]
        identities = _row_identities(frame, batch, required=True)
        rows = [
            gen_random_mask(
                grid,
                1,
                self.mask_ratio,
                generator=torch.Generator(device="cpu").manual_seed(
                    mask_seed(episode_id, camera_id, source_frame_id)
                ),
            )
            for episode_id, source_frame_id in identities
        ]
        # The identity-seeded random draw is generated on the CPU on purpose: a CPU
        # generator is available on every device and makes the mask identical on CPU and
        # CUDA, so an acquisition keeps one mask across devices. Only after the draw is it
        # moved next to the frame's support, where the two are combined.
        mask = torch.cat(rows, dim=0).to(device=prepared.values.device, dtype=prepared.values.dtype)
        support = prepared.contributing.any(dim=1, keepdim=True).to(prepared.values.dtype)
        patch_support = dilate_support(support, kernel=patch, stride=patch) > 0
        unsupported = (~patch_support).reshape(batch, length).to(mask.dtype)
        return torch.maximum(mask, unsupported)

    def generate_mask(self, frame: MeasurementFrame, *, camera_id: str) -> torch.Tensor:
        """The ``(B, L)`` mask of one frame: 1 removes (target) and 0 keeps the patch."""
        prepared = _prepare_frame(
            frame, self.input_mode, alignment=self.alignment, device=self.device
        )
        return self._build_mask(frame, prepared, str(camera_id))

    def forward(
        self, frames: Mapping[str, MeasurementFrame], *, camera_ids: tuple[str, ...]
    ) -> FCMAEOutput:
        if not isinstance(frames, Mapping):
            raise TypeError(
                f"frames must be a mapping of camera ID to frame, got {type(frames).__name__}"
            )
        ids = tuple(str(item) for item in camera_ids)
        if not ids:
            raise ValueError("camera_ids must name at least one camera")
        unknown = [camera for camera in ids if camera not in frames]
        if unknown:
            raise KeyError(f"frames has no entry for camera(s) {unknown}")

        prepared_frames = [
            _prepare_frame(
                frames[camera], self.input_mode, alignment=self.alignment, device=self.device
            )
            for camera in ids
        ]
        layouts = {(item.values.shape[1], item.transform.padded_shape) for item in prepared_frames}
        if len(layouts) > 1:
            raise ValueError(
                "every camera of one pretraining call must share the channel count and the padded "
                "grid; call forward once per camera group instead"
            )

        patch = self.patch_size
        losses: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        reconstructions: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        valid_target_count = 0
        saturated_count = 0
        for camera_id, prepared in zip(ids, prepared_frames, strict=True):
            mask = self._build_mask(frames[camera_id], prepared, camera_id)
            batch, channels, height, width = prepared.values.shape
            grid = (height // patch, width // patch)
            features, _, _ = self.fcmae.encoder(prepared.values, mask)
            prediction = self.fcmae.forward_decoder(features, mask)
            patches = prediction.reshape(batch, patch * patch * channels, grid[0] * grid[1])
            reconstruction = unpatchify(patches.permute(0, 2, 1), patch, channels, grid)
            target_mask = upsample_patch_mask(mask, patch, grid).unsqueeze(1)
            weight = target_mask * (prepared.contributing & ~prepared.saturated).to(
                prepared.values.dtype
            )
            error = (reconstruction - prepared.values) ** 2
            losses.append((error * weight).sum())
            denominators.append(weight.sum())
            valid_target_count += int(weight.detach().sum().item())
            saturated_count += int(
                (target_mask * prepared.saturated.to(target_mask.dtype)).sum().item()
            )
            reconstructions.append(reconstruction)
            masks.append(mask)
        total = torch.stack(losses).sum()
        denominator = torch.stack(denominators).sum()
        loss = total / denominator.clamp_min(1.0)
        return FCMAEOutput(
            loss=loss,
            reconstruction=torch.stack(reconstructions),
            mask=torch.stack(masks),
            valid_target_count=torch.tensor(
                valid_target_count, dtype=torch.long, device=loss.device
            ),
            saturated_count=torch.tensor(saturated_count, dtype=torch.long, device=loss.device),
        )

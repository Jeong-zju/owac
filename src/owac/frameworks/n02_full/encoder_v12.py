"""N02 v1.2 direct spatial branch: RAW encoder with stride 4/8/16 levels.

This module is the v1.2 front end of the main chain
``RAW3 -> ConvNeXt V2-Base multi-scale features -> candidates -> readout``.  It is a thin
extension of the P0/P1 observation layer, not a new backbone:

* the backbone is the P0-parity ``RawConvNeXtV2`` from
  :mod:`owac.frameworks.n02_full.components.convnextv2`
  with the official Base depths ``[3, 3, 27, 3]`` and dims ``[128, 256, 512, 1024]``;
* frame packing (RAW3 triplet / CFA4 planes), rectangular padding, crop origin, coordinate
  dilation and validity are the P1 helpers of
  :mod:`owac.frameworks.n02_full.observation` (``_prepare_frame``,
  ``_level_sensor_coordinates``, ``_row_identities``, ``_support_digest``,
  ``CAMERA_LEVEL_SEPARATOR``), reused verbatim so the two encoders cannot drift apart.

Differences against :class:`owac.frameworks.n02_full.observation.RawObservationEncoder`
(recorded because P0/P1's encoder is accepted and must not be changed in place):

1. **stride 4 is kept.**  P1 returns only ``stride8``/``stride16`` (``out_indices=(1, 2)``);
   the official ConvNeXt V2 stage 0 is available at stride 4 (``dims[0] = 128``) and is
   retained here as ``stride4``.  The plan configures stride 4/8/16 for the v1.2 direct
   branch, so the finest level is a new, separately stated extension.
2. **Every level is projected to one width** ``d_model`` (default 512) by a 1x1 convolution
   so that the readout and the decoder see a uniform channel count.  P1 projects to
   ``projection_dim`` for stride 8/16 only.
3. **Spatial outputs remain separate by camera** and every level keeps its own resolution,
   sensor coordinates and validity. Optional runtime grouping batches equal-shaped cameras
   only inside the shared backbone, then restores their identities and layout.
4. ``forward`` returns :class:`SpatialV12` instead of ``FeaturePyramid``; the stride-32 stage
   of the backbone is never executed because nothing consumes it (P1's
   ``forward_multiscale`` runs all four stages).

No RGB/sRGB/XYZ image is produced anywhere on this path and no RGB teacher or RGB weight is
used.  ``RawConvNeXtV2`` is initialised from scratch (``trunc_normal_(std=0.02)`` inside the
P0 component); the new 1x1 projection follows the same P0 projection initialisation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n02_full.components.convnextv2 import (
    DEFAULT_DEPTHS,
    DEFAULT_DIMS,
    STAGE_STRIDES,
    RawConvNeXtV2,
    trunc_normal_,
)
from owac.frameworks.n02_full.components.fcmae import dilate_support
from owac.frameworks.n02_full.observation import (
    CAMERA_LEVEL_SEPARATOR,
    RAW3,
    STRIDE_ALIGNMENT,
    FrameTransform,
    _canonical,
    _level_sensor_coordinates,
    _prepare_frame,
    _row_identities,
    _support_digest,
)
from owac.frameworks.n02_full.state import FeatureLevel, ObservationBundle

__all__ = [
    "LEVEL_IDS_V12",
    "LEVEL_STAGES_V12",
    "LEVEL_STRIDES_V12",
    "RawEncoderV12",
    "SpatialV12",
]

#: Level ID -> stride of that level in frame cells (RAW3) / packed cells (CFA4).
LEVEL_STRIDES_V12 = {"stride4": 4, "stride8": 8, "stride16": 16}

#: Level ID -> ConvNeXt V2 stage index (stage 0 is the 4x4/stride-4 stem output).
LEVEL_STAGES_V12 = {"stride4": 0, "stride8": 1, "stride16": 2}

#: The three direct spatial levels of the v1.2 front end, finest first.
LEVEL_IDS_V12 = ("stride4", "stride8", "stride16")

_INPUT_MODE_ALIASES = {"RAW3": RAW3, "raw3": RAW3, "CFA4": "cfa4", "cfa4": "cfa4"}


def _level_key(camera_id: str, level_id: str, *, multi: bool) -> str:
    """``"<camera>::<level>"`` for multi-camera pyramids, the plain level ID otherwise."""
    return f"{camera_id}{CAMERA_LEVEL_SEPARATOR}{level_id}" if multi else level_id


@dataclass(slots=True)
class SpatialV12:
    """Multi-scale RAW spatial features for one decision time.

    Frozen field names and order (v1.2 cross-module interface): ``levels``, ``coords``,
    ``valid``, ``camera_ids``, ``level_ids``, ``level_dims``, ``support_hash``.  Keys are
    the plain level ID for a single camera and ``"<camera_id>::<level_id>"`` when the bundle
    carries more than one camera; cameras are never folded into the batch axis.

    ``coords[key]`` is ``(B, H, W, 2)`` float32 in **sensor pixel** coordinates
    (``(row, column)``, the ``+0.5`` pixel-centre convention of
    ``owac.raw.rawvla_sequence``), ``valid[key]`` is ``(B, H, W)`` bool and is true iff the
    cell's stride-aligned input window contains at least one contributing measurement.
    Feature values at invalid cells are exactly zero.

    Trailing fields with defaults are additive v1.2 metadata (``transforms``,
    ``support_ids``, ``input_mode``); they do not change the frozen names or order.
    """

    levels: dict[str, torch.Tensor]
    coords: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]
    camera_ids: tuple[str, ...]
    level_ids: tuple[str, ...]
    level_dims: dict[str, int]
    support_hash: str
    transforms: dict[str, Any] = field(default_factory=dict)
    support_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    input_mode: str = RAW3

    def __post_init__(self) -> None:
        if not isinstance(self.levels, dict) or not self.levels:
            raise ValueError("SpatialV12.levels must be a non-empty dict")
        self.camera_ids = tuple(str(item) for item in self.camera_ids)
        self.level_ids = tuple(str(item) for item in self.level_ids)
        self.level_dims = {str(key): int(value) for key, value in dict(self.level_dims).items()}
        self.support_hash = str(self.support_hash)
        for name in ("levels", "coords", "valid"):
            payload = getattr(self, name)
            if set(payload) != set(self.levels):
                raise ValueError(f"SpatialV12.{name} keys must match SpatialV12.levels keys")
        for key, values in self.levels.items():
            if values.ndim != 4:
                raise ValueError(f"level {key!r} must be (B, C, H, W), got {tuple(values.shape)}")
            batch, channels, height, width = values.shape
            if self.level_dims[key] != channels:
                raise ValueError(
                    f"level {key!r} has {channels} channels but level_dims declares "
                    f"{self.level_dims[key]}"
                )
            if tuple(self.coords[key].shape) != (batch, height, width, 2):
                raise ValueError(f"coords[{key!r}] must be (B, H, W, 2)")
            if tuple(self.valid[key].shape) != (batch, height, width):
                raise ValueError(f"valid[{key!r}] must be (B, H, W)")

    @property
    def batch_size(self) -> int:
        return int(next(iter(self.levels.values())).shape[0])

    def per_camera(self) -> dict[str, dict[str, FeatureLevel]]:
        """``{camera_id: {level_id: FeatureLevel}}`` regardless of the key format."""
        grouped: dict[str, dict[str, FeatureLevel]] = {camera: {} for camera in self.camera_ids}
        multi = len(self.camera_ids) > 1
        for camera in self.camera_ids:
            for level_id in self.level_ids:
                key = _level_key(camera, level_id, multi=multi)
                if key not in self.levels:
                    continue
                grouped[camera][level_id] = FeatureLevel(
                    values=self.levels[key],
                    valid=self.valid[key],
                    sensor_coordinates=self.coords[key],
                    support_ids=self.support_ids.get(key, ()),
                    level_id=level_id,
                )
        return grouped


class RawEncoderV12(nn.Module):
    """ConvNeXt V2-Base RAW encoder with stride 4/8/16 levels projected to ``d_model``.

    Args:
        camera_ids: camera IDs this encoder is configured for.  A forward pass may carry a
            non-empty subset (for example a single-camera smoke test) but never an unknown
            camera.
        dims/depths: the four-stage ConvNeXt V2 layout; the official Base sizes are the
            defaults ``(128, 256, 512, 1024)`` / ``(3, 3, 27, 3)``.
        in_chans: sensor planes; 3 for RAW3.
        input_mode: ``"RAW3"`` (three co-sited pseudo-RAW channels) or ``"CFA4"``; the two
            modes are never converted into each other.
        d_model: width every level is projected to.
        alignment: frame padding alignment in frame cells; a multiple of the backbone's
            total stride 32.
    """

    LEVEL_IDS = LEVEL_IDS_V12

    def __init__(
        self,
        *,
        camera_ids: Sequence[str],
        dims: Sequence[int] = DEFAULT_DIMS,
        depths: Sequence[int] = DEFAULT_DEPTHS,
        in_chans: int = 3,
        input_mode: str = "RAW3",
        d_model: int = 512,
        alignment: int = STRIDE_ALIGNMENT,
    ):
        super().__init__()
        camera_ids = tuple(str(item) for item in camera_ids)
        if not camera_ids:
            raise ValueError("camera_ids must name at least one camera")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError(f"camera_ids must be unique, got {camera_ids}")
        if any(CAMERA_LEVEL_SEPARATOR in camera for camera in camera_ids):
            raise ValueError(
                f"camera IDs must not contain the level separator {CAMERA_LEVEL_SEPARATOR!r}"
            )
        self.camera_ids = camera_ids
        if input_mode not in _INPUT_MODE_ALIASES:
            raise ValueError(f"unknown input_mode {input_mode!r}")
        self.input_mode = _INPUT_MODE_ALIASES[input_mode]
        self.in_chans = int(in_chans)
        if self.input_mode == RAW3 and self.in_chans != 3:
            raise ValueError("RAW3 requires 3 co-sited channels")
        self.alignment = int(alignment)
        if self.alignment <= 0:
            raise ValueError("alignment must be positive")
        dims = tuple(int(item) for item in dims)
        depths = tuple(int(item) for item in depths)
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("only the four-stage ConvNeXt V2 layout is supported")
        self.dims = dims
        self.depths = depths
        self._stages = tuple(LEVEL_STAGES_V12[level] for level in LEVEL_IDS_V12)
        self.backbone = RawConvNeXtV2(
            in_chans=self.in_chans,
            depths=depths,
            dims=dims,
            num_classes=0,
            out_indices=self._stages,
        )
        self.d_model = int(d_model)
        projections: dict[str, nn.Module] = {}
        for level, stage in zip(LEVEL_IDS_V12, self._stages, strict=True):
            projections[level] = nn.Conv2d(dims[stage], self.d_model, kernel_size=1)
        self.proj = nn.ModuleDict(projections)
        # Same projection initialisation as the P0/P1 observation encoder.
        for module in self.proj.modules():
            if isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    # -- reporting ----------------------------------------------------------------------

    @property
    def spatial_dims(self) -> dict[str, int]:
        """``level_id -> channel count`` of the returned level tensors."""
        return {level: self.d_model for level in LEVEL_IDS_V12}

    @property
    def backbone_dims(self) -> dict[str, int]:
        """Native ConvNeXt V2 stage widths before the 1x1 ``d_model`` projection."""
        return {
            level: self.dims[stage]
            for level, stage in zip(LEVEL_IDS_V12, self._stages, strict=True)
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- forward ------------------------------------------------------------------------

    def _backbone_levels(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Stages 0/1/2 of the backbone; identical to ``forward_features((0, 1, 2))``.

        The stride-32 stage is never executed.  ``RawConvNeXtV2`` exposes its stage modules,
        so this is the upstream loop with the unused stage dropped rather than a new
        network; a test compares it against ``forward_multiscale`` on the same weights.
        """
        features: dict[str, torch.Tensor] = {}
        x = values
        for stage in range(3):
            x = self.backbone.downsample_layers[stage](x)
            x = self.backbone.stages[stage](x)
            features[f"stage{stage}_stride{STAGE_STRIDES[stage]}"] = x
        return features

    def _encode_camera(
        self, frame: MeasurementFrame, camera_id: str, *, prepared=None, raw_levels=None
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, int],
        FrameTransform,
        tuple[tuple[str, str], ...],
        torch.Tensor,
    ]:
        if prepared is None:
            prepared = _prepare_frame(
                frame, self.input_mode, alignment=self.alignment, device=self.device
            )
        if raw_levels is None:
            raw_levels = self._backbone_levels(prepared.values)
        support = prepared.contributing.any(dim=1, keepdim=True).to(prepared.values.dtype)
        supports = {4: dilate_support(support, kernel=4, stride=4)}
        supports[8] = dilate_support(supports[4], kernel=2, stride=2)
        supports[16] = dilate_support(supports[8], kernel=2, stride=2)

        transform = prepared.transform
        identities = _row_identities(frame, prepared.values.shape[0], required=False)
        levels: dict[str, torch.Tensor] = {}
        coords: dict[str, torch.Tensor] = {}
        valid: dict[str, torch.Tensor] = {}
        dims: dict[str, int] = {}
        for level_id in LEVEL_IDS_V12:
            stride = LEVEL_STRIDES_V12[level_id]
            feature = raw_levels[f"stage{LEVEL_STAGES_V12[level_id]}_stride{stride}"]
            cell_support = supports[stride]
            values = self.proj[level_id](feature) * cell_support
            coordinates = _level_sensor_coordinates(values.shape[-2:], stride, transform)
            coordinates = (
                coordinates.unsqueeze(0)
                .expand(values.shape[0], -1, -1, -1)
                .to(values.device)
                .contiguous()
            )
            levels[level_id] = values
            coords[level_id] = coordinates
            valid[level_id] = cell_support.squeeze(1).bool()
            dims[level_id] = int(values.shape[1])
        return (
            levels,
            coords,
            valid,
            dims,
            transform,
            tuple(identities),
            prepared.contributing.any(dim=1),
        )

    def forward(self, observation: Any) -> SpatialV12:
        """Encode every camera of one decision time into :class:`SpatialV12`."""
        bundle = getattr(observation, "bundle", observation)
        if not isinstance(bundle, ObservationBundle):
            raise TypeError(
                "RawEncoderV12.forward expects an ObservationBundle (or InferenceInput), got "
                f"{type(observation).__name__}"
            )
        camera_ids = tuple(str(item) for item in bundle.camera_ids)
        if not camera_ids:
            raise ValueError("the bundle carries no camera frames")
        unknown = [camera for camera in camera_ids if camera not in self.camera_ids]
        if unknown:
            raise ValueError(
                f"bundle carries cameras {unknown} that this encoder was not configured for "
                f"({self.camera_ids})"
            )
        multi = len(camera_ids) > 1
        levels: dict[str, torch.Tensor] = {}
        coords: dict[str, torch.Tensor] = {}
        valid: dict[str, torch.Tensor] = {}
        dims: dict[str, int] = {}
        support_ids: dict[str, tuple[str, ...]] = {}
        transforms: dict[str, Any] = {}
        entries: list[dict[str, Any]] = []
        prepared_cameras, camera_levels_batched = {}, {}
        if getattr(self, "batch_cameras", False):
            groups = {}
            for camera_id in camera_ids:
                prepared = _prepare_frame(
                    bundle.frames[camera_id],
                    self.input_mode,
                    alignment=self.alignment,
                    device=self.device,
                )
                prepared_cameras[camera_id] = prepared
                groups.setdefault(tuple(prepared.values.shape[1:]), []).append(camera_id)
            for group in groups.values():
                joined = torch.cat([prepared_cameras[c].values for c in group])
                joined_levels = self._backbone_levels(joined)
                offset = 0
                for camera_id in group:
                    count = prepared_cameras[camera_id].values.shape[0]
                    camera_levels_batched[camera_id] = {
                        key: value[offset : offset + count] for key, value in joined_levels.items()
                    }
                    offset += count
        for camera_id in camera_ids:
            (
                camera_levels,
                camera_coords,
                camera_valid,
                camera_dims,
                transform,
                identities,
                contributing,
            ) = self._encode_camera(
                bundle.frames[camera_id],
                camera_id,
                prepared=prepared_cameras.get(camera_id),
                raw_levels=camera_levels_batched.get(camera_id),
            )
            transforms[camera_id] = transform
            entries.append(
                {
                    "camera_id": camera_id,
                    "input_mode": self.input_mode,
                    "identity": [list(item) for item in identities],
                    "sensor_shape": list(transform.sensor_shape),
                    "frame_shape": list(transform.frame_shape),
                    "pad_after": list(transform.pad_after),
                    "crop_offset": list(transform.crop_offset),
                    "support_sha256": _support_digest(contributing),
                }
            )
            for level_id in LEVEL_IDS_V12:
                key = _level_key(camera_id, level_id, multi=multi)
                levels[key] = camera_levels[level_id]
                coords[key] = camera_coords[level_id]
                valid[key] = camera_valid[level_id]
                dims[key] = camera_dims[level_id]
                support_ids[key] = tuple(
                    f"{episode}|{camera_id}|{source}" for episode, source in identities
                )
        support_hash = hashlib.sha256(
            _canonical(sorted(entries, key=lambda item: item["camera_id"])).encode("utf-8")
        ).hexdigest()
        return SpatialV12(
            levels=levels,
            coords=coords,
            valid=valid,
            camera_ids=camera_ids,
            level_ids=LEVEL_IDS_V12,
            level_dims=dims,
            support_hash=support_hash,
            transforms=transforms,
            support_ids=support_ids,
            input_mode=self.input_mode,
        )

    # -- small helpers used by the readout/decoder and the report ------------------------

    def level_key(self, camera_id: str, level_id: str) -> str:
        """Key of one level in the returned :class:`SpatialV12`."""
        return _level_key(str(camera_id), str(level_id), multi=len(self.camera_ids) > 1)

    def config_summary(self) -> dict[str, Any]:
        """Compact, human-readable configuration and size record."""
        return {
            "camera_ids": list(self.camera_ids),
            "input_mode": self.input_mode,
            "in_chans": self.in_chans,
            "dims": list(self.dims),
            "depths": list(self.depths),
            "d_model": self.d_model,
            "alignment": self.alignment,
            "level_ids": list(LEVEL_IDS_V12),
            "parameters": self.parameter_count(),
            "backbone_parameters": sum(p.numel() for p in self.backbone.parameters()),
        }

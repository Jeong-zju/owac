"""Teacher-free RAW temporal packets, labels and geometry.

This module is intentionally independent of the teacher registry, teacher
artifact and target cache.  Pure student evaluation imports it directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from owac.baselines.raw_stem_siglip import RawStemInput
from owac.baselines.raw_temporal_siglip import (
    TemporalPacket,
    TemporalRawStemStudent,
    causal_frame_ids,
    causal_padding_mask,
)
from owac.raw.teacher_data import CAMERA_ORDER, TeacherData, make_window

__all__ = [
    "ACTIVE_IMAGE_KEYS",
    "CAMERA_ALIASES",
    "INACTIVE_IMAGE_KEY",
    "RawTemporalDataset",
    "RawTemporalSample",
    "native_camera_ids",
    "physical_positions_224_to_256",
    "positions_for_geometry",
    "resize_raw_256_to_224",
]

ACTIVE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
INACTIVE_IMAGE_KEY = "right_wrist_0_rgb"
CAMERA_ALIASES = {"agentview": "base_0_rgb", "wrist": "left_wrist_0_rgb"}


def native_camera_ids() -> tuple[str, ...]:
    return tuple(CAMERA_ALIASES[camera] for camera in CAMERA_ORDER)


def resize_raw_256_to_224(frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Explicit dtype-safe full-view 256→224 RAW conversion.

    uint8 is always interpreted as DN in [0,255] and divided by 255, including
    frames whose observed min is 0 or 1.  Floating input must already be finite
    in [0,1].  uint16 and other ambiguous integer/float DN ranges are rejected.
    """
    if isinstance(frames, np.ndarray):
        array = np.ascontiguousarray(frames)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError("RAW frames must be [T,H,W,3]")
        if array.shape[1] != array.shape[2] or array.shape[1] not in (224, 256):
            raise ValueError("RAW spatial layout must be square 224 or 256")
        if array.dtype == np.uint8:
            tensor = torch.from_numpy(array).float() / 255.0
        elif np.issubdtype(array.dtype, np.floating):
            tensor = torch.from_numpy(array).float()
            if not torch.isfinite(tensor).all():
                raise ValueError("floating RAW contains non-finite values")
            if float(tensor.min()) < -1e-6 or float(tensor.max()) > 1.0 + 1e-6:
                raise ValueError("floating RAW must already be in [0,1]")
            tensor = tensor.clamp(0.0, 1.0)
        else:
            raise ValueError(
                f"RAW dtype {array.dtype} is ambiguous; uint8 DN or float [0,1] required"
            )
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        tensor = frames.detach()
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("RAW tensor must be [T,3,H,W]")
        if tensor.shape[2] != tensor.shape[3] or tensor.shape[2] not in (224, 256):
            raise ValueError("RAW spatial layout must be square 224 or 256")
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif tensor.dtype == torch.uint16:
            raise ValueError("uint16 RAW is ambiguous; use uint8 DN or float [0,1]")
        elif tensor.is_floating_point():
            tensor = tensor.float()
            if not torch.isfinite(tensor).all():
                raise ValueError("floating RAW contains non-finite values")
            if float(tensor.min()) < -1e-6 or float(tensor.max()) > 1.0 + 1e-6:
                raise ValueError("floating RAW must already be in [0,1]")
            tensor = tensor.clamp(0.0, 1.0)
        else:
            raise ValueError(f"RAW dtype {tensor.dtype} is ambiguous")
    if tuple(tensor.shape[-2:]) == (224, 224):
        return tensor.contiguous()
    return F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False).contiguous()


def physical_positions_224_to_256(positions: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(positions) or positions.shape[-1] != 3:
        raise ValueError("positions must be [...,3]")
    result = positions.detach().clone()
    result[..., :3] *= 256.0 / 224.0
    return result


def positions_for_geometry(
    geometry: Any,
    *,
    cameras: Sequence[str] = ACTIVE_IMAGE_KEYS,
    physical_origins: Sequence[tuple[int, int]] | None = None,
    backmap_256: bool = True,
) -> torch.Tensor:
    """Create physical patch-center positions [1,Ncam*N,3] for a geometry."""
    grid_y = torch.arange(geometry.grid_height, dtype=torch.float32)
    grid_x = torch.arange(geometry.grid_width, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    base_y = ((yy + 0.5) * geometry.patch_size).reshape(-1, 1)
    base_x = ((xx + 0.5) * geometry.patch_size).reshape(-1, 1)
    origins = list(physical_origins or [(0, 0)] * len(cameras))
    groups = []
    for origin_y, origin_x in origins:
        physical_y = base_y + float(origin_y)
        physical_x = base_x + float(origin_x)
        patch = torch.full_like(physical_y, float(geometry.patch_size))
        groups.append(torch.cat((physical_y, physical_x, patch), dim=-1))
    positions = torch.stack(groups, dim=0).reshape(1, -1, 3)
    if backmap_256 and int(geometry.image_size) == 224:
        positions = physical_positions_224_to_256(positions)
    return positions


@dataclass(frozen=True, slots=True)
class RawTemporalSample:
    source_id: str
    split: str
    endpoint: int
    packet: TemporalPacket
    state8_raw: torch.Tensor
    actions32: torch.Tensor
    actions7: torch.Tensor
    future_mask: torch.Tensor
    language: str
    frame_ids: tuple[int, ...]


class RawTemporalDataset:
    """RAW-only packet/label iterator; no teacher, cache or target row."""

    def __init__(
        self,
        data: TeacherData,
        student: TemporalRawStemStudent,
        *,
        stats: Any,
        arm: str = "C",
    ) -> None:
        if arm not in {"C", "D"}:
            raise ValueError("pure RAW dataset supports C/D only")
        self.data = data
        self.student = student
        self.stats = stats
        self.arm = arm

    def _device(self) -> torch.device:
        return next(self.student.parameters()).device

    def packet_for_endpoint(self, source_id: str, endpoint: int) -> TemporalPacket:
        episode = self.data.episode_raw(source_id)
        frame_ids = causal_frame_ids(endpoint)
        timestamps = None
        if episode.timestamps is not None:
            timestamps = tuple(float(episode.timestamps[index]) for index in frame_ids)
        frames = []
        for frame_id in frame_ids:
            raw = np.stack([episode.raw[camera][frame_id] for camera in CAMERA_ORDER], axis=0)
            resized = resize_raw_256_to_224(raw).to(self._device())
            frames.append(
                RawStemInput(
                    values=resized.unsqueeze(0),
                    input_mode="rawvla_rgb3",
                    camera_ids=native_camera_ids(),
                    units="normalized_0_1",
                )
            )
        return TemporalPacket(
            frames=tuple(frames),
            source_demo_id=source_id,
            endpoint=endpoint,
            frame_ids=frame_ids,
            padding_mask=causal_padding_mask(endpoint),
            reset=endpoint == 0,
            timestamps=timestamps,
            position_source_size=256,
            position_target_size=224,
            position_backmap=True,
            preprocess_id="raw256_fullview_bilinear_align_corners_false_224",
        )

    def sample(self, source_id: str, endpoint: int) -> RawTemporalSample:
        if endpoint < 0 or endpoint % 5:
            raise ValueError("endpoint must be a non-negative multiple of 5")
        source = self.data.manifest.source(source_id)
        episode = self.data.episode_raw(source_id)
        window = make_window(episode, endpoint)
        normalized = window.normalized(self.stats)
        return RawTemporalSample(
            source_id=source_id,
            split=source.split,
            endpoint=endpoint,
            packet=self.packet_for_endpoint(source_id, endpoint),
            state8_raw=torch.from_numpy(np.asarray(window.state, dtype=np.float32).copy()),
            actions32=torch.from_numpy(normalized["padded_actions"]).float(),
            actions7=torch.from_numpy(window.actions).float(),
            future_mask=torch.from_numpy(window.action_mask).bool(),
            language=str(
                episode.source.metadata.get("language", "put the cream cheese in the bowl")
            ),
            frame_ids=causal_frame_ids(endpoint),
        )

    def source_samples(self, source_id: str) -> list[RawTemporalSample]:
        episode = self.data.episode_raw(source_id)
        return [self.sample(source_id, endpoint) for endpoint in range(0, episode.num_frames, 5)]

"""RAW-stem student with an inherited, frozen SigLIP visual tower.

This module is the front end for the RAW_STEM_SIGLIP candidate.  It does *not*
construct a policy, read a checkpoint, or start training.  A caller supplies an
already loaded and audited OpenPI/PaliGemma policy; the factory deep-copies the
exact SigLIP position embedding, encoder, post-layernorm and original
multi-modal projector, then replaces **only** the original patch embedding with
a small trainable ``conv64`` + patch-convolution stem.

Three front ends are kept separate:

``rawvla_rgb3``
    Three-channel pseudo-RAW / HDR-derived view, the current v1 regime.
``bayer4_packed``
    Four-channel packed Bayer view at half physical resolution.  The patch
    stride is half the checkpoint patch size, the physical crop must be even and
    no token interpolation is performed.
``rgb3_control``
    An RGB control that consumes the teacher's *already preprocessed* tensor in
    ``[-1, 1]``; this module never applies resize, padding or gamma.

The teacher wrapper (:class:`SiglipTeacher`) is deliberately lazy and separate.
It calls the original ``OpenPIPaliGemma.embed_image`` under ``torch.no_grad()``
and returns detached clones (never ``inference_mode`` tensors).  It rejects a
temporal/burst teacher declaration.

The deployment path (:class:`RawStemSiglipDeployment`,
:func:`export_student_artifact`, :func:`load_student_artifact`) contains only the
exported front-end weights.  It accepts an architecture-only factory and never
instantiates a teacher, an ISP or a cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from owac.core.encoder_contract_n02 import Validity, VisualTokens

__all__ = [
    "BAYER_CFA_PATTERNS",
    "INPUT_MODES",
    "OPENPI_PALIGEMMA_MODULE",
    "STUDENT_ARTIFACT_SCHEMA",
    "STUDENT_ARTIFACT_VERSION",
    "BayerPacked",
    "CheckpointIdentity",
    "HistoryContract",
    "InitializationAudit",
    "LocalConvStem",
    "PolicyLoadEvidence",
    "RawStemInput",
    "RawStemSiglipDeployment",
    "RawStemSiglipStudent",
    "RawStemOutput",
    "SiglipTeacher",
    "TeacherOutput",
    "TensorFingerprint",
    "VisionGeometry",
    "assert_backend_frozen",
    "assert_bridge_has_no_projection",
    "build_raw_stem_student",
    "build_artifact_frontend_from_metadata",
    "export_student_artifact",
    "load_student_artifact",
    "pack_bayer4",
    "pack_bayer_input",
    "sha256_file",
    "sha256_tensor",
]

INPUT_MODES = ("rawvla_rgb3", "bayer4_packed", "rgb3_control")
BAYER_CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
OPENPI_PALIGEMMA_MODULE = "starVLA.model.modules.vlm.OpenPIPaliGemma"
SIGLIP_MODEL_MODULE = "starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip"
STUDENT_ARTIFACT_SCHEMA = "owac.raw-stem-student.v1"
STUDENT_ARTIFACT_VERSION = 1

_CFA_COLORS = {
    "RGGB": ("R", "G", "G", "B"),
    "BGGR": ("B", "G", "G", "R"),
    "GRBG": ("G", "R", "B", "G"),
    "GBRG": ("G", "B", "R", "G"),
}
_CFA_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))


class RawStemContractError(ValueError):
    """The input, policy evidence, geometry or artifact violates the contract."""


class RawStemUnavailableError(RuntimeError):
    """Required real evidence (checkpoint, audit or policy source) is absent."""


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash the exact bytes of a tensor (dtype, shape and values)."""
    value = tensor.detach().cpu().contiguous().reshape(-1)
    if value.numel() == 0:
        payload = b""
    else:
        payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_key(name: str) -> str:
    """Normalize policy/student parameter names to an inheritance key.

    ``PI05.vlm.model.model.vision_tower...``, a bare ``vision_tower...`` and the
    student's local ``vision_tower...`` all map to the same canonical name.  The
    original projector is named ``multi_modal_projector`` in the policy and
    ``projector`` in this front end; both map to the policy name.
    """
    key = name.replace("paligemma_with_expert.paligemma.", "")
    for marker in ("multi_modal_projector.", "vision_tower."):
        index = key.find(marker)
        if index >= 0:
            return key[index:]
    if key.startswith("projector."):
        return "multi_modal_projector." + key[len("projector.") :]
    return key


def _as_tuple(value: Any, name: str) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{name} must be a tuple or list, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class HistoryContract:
    """Explicit causal history, reset and camera-order declaration.

    The contract is data, not an architecture.  A single-frame call is
    ``HistoryContract()``; temporal callers must provide ordered frame ids,
    timestamps and stride.  No future frame may be implied.
    """

    frame_ids: tuple[int, ...] = (0,)
    timestamps: tuple[float, ...] = (0.0,)
    stride: int = 1
    camera_order: tuple[str, ...] = ()
    burst_frames: int = 1
    rnn_frames: int = 1
    reset: bool = False
    max_history: int | None = None
    require_causal: bool = True
    padding_mask: tuple[bool, ...] = ()
    new_measurement: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        frame_ids = tuple(int(value) for value in self.frame_ids)
        timestamps = tuple(float(value) for value in self.timestamps)
        camera_order = tuple(self.camera_order)
        padding = tuple(bool(value) for value in self.padding_mask)
        new_measurement = tuple(bool(value) for value in self.new_measurement)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "camera_order", camera_order)
        if not frame_ids or len(frame_ids) != len(timestamps):
            raise RawStemContractError(
                "history ids and timestamps must be nonempty and equal length"
            )
        if not padding:
            padding = tuple(False for _ in frame_ids)
        if not new_measurement:
            new_measurement = tuple(not value for value in padding)
        object.__setattr__(self, "padding_mask", padding)
        object.__setattr__(self, "new_measurement", new_measurement)
        if len(padding) != len(frame_ids) or len(new_measurement) != len(frame_ids):
            raise RawStemContractError("history padding/new_measurement lengths must match")
        if int(self.stride) < 1:
            raise RawStemContractError("history stride must be positive")
        if any(b < a for a, b in zip(frame_ids, frame_ids[1:], strict=False)):
            raise RawStemContractError("history frame ids must be nondecreasing")
        if not all(math.isfinite(value) for value in timestamps):
            raise RawStemContractError("history timestamps must be finite")
        if any(b < a for a, b in zip(timestamps, timestamps[1:], strict=False)):
            raise RawStemContractError("history timestamps must be nondecreasing")
        for index in range(1, len(frame_ids)):
            repeated = frame_ids[index] == frame_ids[index - 1]
            same_time = timestamps[index] == timestamps[index - 1]
            if repeated or same_time:
                if repeated != same_time:
                    raise RawStemContractError("repeated physical frame id/time must both repeat")
                # Left padding ends with the one actual acquisition; right
                # padding starts with it. Both describe the same physical burst.
                if not (padding[index] or padding[index - 1]):
                    raise RawStemContractError(
                        "repeated physical ids/times are allowed only as declared padding"
                    )
            else:
                if frame_ids[index] - frame_ids[index - 1] != int(self.stride):
                    raise RawStemContractError(
                        "history frame ids must advance by the declared stride"
                    )
                if padding[index]:
                    raise RawStemContractError("padding cannot advance a physical frame id")
        acquisitions: set[tuple[int, float]] = set()
        for frame_id, timestamp, padded, new in zip(
            frame_ids, timestamps, padding, new_measurement, strict=True
        ):
            if padded and new:
                raise RawStemContractError("padding cannot be a new measurement")
            if not padded:
                identity = (frame_id, timestamp)
                if identity in acquisitions:
                    raise RawStemContractError("one physical acquisition is declared twice")
                acquisitions.add(identity)
        if int(self.burst_frames) < 1 or int(self.rnn_frames) < 1:
            raise RawStemContractError("burst/rnn frame counts must be positive")
        if len(set(camera_order)) != len(camera_order):
            raise RawStemContractError("camera_order must be unique")
        if self.max_history is not None and self.max_history < len(frame_ids):
            raise RawStemContractError("max_history is smaller than the declared history")
        if not isinstance(self.reset, bool):
            raise RawStemContractError("history reset must be boolean")

    def compatible_with(self, other: HistoryContract) -> bool:
        """Structural compatibility; later windows may change ids/time/reset."""
        return (
            self.stride == other.stride
            and self.camera_order == other.camera_order
            and self.burst_frames == other.burst_frames
            and self.rnn_frames == other.rnn_frames
            and self.max_history == other.max_history
            and self.require_causal == other.require_causal
        )

    @property
    def is_temporal(self) -> bool:
        return len(self.frame_ids) > 1 or self.burst_frames > 1 or self.rnn_frames > 1

    @property
    def temporal_frames(self) -> int:
        return len(self.frame_ids)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_frames(
        cls,
        frame_ids: Sequence[int],
        timestamps: Sequence[float],
        *,
        stride: int = 1,
        camera_order: Sequence[str] = (),
        burst_frames: int | None = None,
        rnn_frames: int | None = None,
        reset: bool = False,
        max_history: int | None = None,
        padding_mask: Sequence[bool] = (),
        new_measurement: Sequence[bool] = (),
    ) -> HistoryContract:
        ids = tuple(frame_ids)
        times = tuple(timestamps)
        count = len(ids)
        return cls(
            frame_ids=ids,
            timestamps=times,
            stride=stride,
            camera_order=tuple(camera_order),
            burst_frames=count if burst_frames is None else burst_frames,
            rnn_frames=count if rnn_frames is None else rnn_frames,
            reset=reset,
            max_history=count if max_history is None else max_history,
            padding_mask=tuple(padding_mask),
            new_measurement=tuple(new_measurement),
        )


def _check_dtype_and_finite(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise RawStemContractError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(tensor).all():
        raise RawStemContractError(f"{name} contains non-finite values")


# --------------------------------------------------------------------------- #
# Checkpoint / policy evidence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TensorFingerprint:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Identity of the actual policy checkpoint that supplied the visual weights.

    ``tensor_sha256`` is keyed by a canonical visual name (a suffix beginning at
    ``vision_tower.`` or ``multi_modal_projector.``).  Real-mode construction
    requires every inherited tensor to appear there and validates the full file
    SHA256 before copying anything.
    """

    path: str
    sha256: str
    verified: bool = False
    config_sha256: str = ""
    tensor_sha256: Mapping[str, str] = field(default_factory=dict)
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RawStemContractError("checkpoint sha256 must be 64 lowercase hex characters")
        normalized = {
            _canonical_key(str(key)): str(value).lower()
            for key, value in self.tensor_sha256.items()
        }
        for key, value in normalized.items():
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RawStemContractError(f"invalid tensor sha256 for {key}")
        object.__setattr__(self, "tensor_sha256", normalized)
        if self.config_sha256:
            value = str(self.config_sha256).lower()
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RawStemContractError("config_sha256 must be 64 lowercase hex characters")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolicyLoadEvidence:
    """The audited full-policy load, not merely a visual backbone source audit.

    ``acceptance`` must be ``accepted_real_policy`` for real construction.  A
    source-archive or backbone-only audit is explicitly rejected by
    :func:`build_raw_stem_student`.
    """

    checkpoint_sha256: str
    policy_module: str
    loaded_keys: int
    model_keys: int
    verified: bool = False
    acceptance: str = "unverified"
    visual_weights_source: str = ""
    active_missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    shape_mismatch: tuple[str, ...] = ()
    inactive_keys: tuple[str, ...] = ()
    vision_keys_loaded: tuple[str, ...] = ()
    temporal_teacher: bool = False
    audit_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("active_missing", "unexpected", "shape_mismatch", "inactive_keys"):
            object.__setattr__(self, name, _as_tuple(getattr(self, name), name))
        object.__setattr__(
            self, "vision_keys_loaded", _as_tuple(self.vision_keys_loaded, "vision_keys_loaded")
        )
        if self.loaded_keys < 0 or self.model_keys < 0:
            raise RawStemContractError("load audit key counts must be nonnegative")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> PolicyLoadEvidence:
        """Build evidence from a full ``load_audited`` result plus identity fields.

        The caller must add ``checkpoint_sha256``, ``policy_module``,
        ``acceptance`` and ``vision_keys_loaded``; a raw ``load_audited`` dict
        alone is intentionally insufficient.
        """
        if not isinstance(mapping, Mapping):
            raise RawStemContractError("load audit must be a mapping")
        data = dict(mapping)
        required = ("checkpoint_sha256", "policy_module", "loaded_keys", "model_keys")
        missing = [key for key in required if key not in data]
        if missing:
            raise RawStemUnavailableError(f"load audit lacks identity fields: {missing}")
        audit = cls(
            checkpoint_sha256=str(data["checkpoint_sha256"]),
            policy_module=str(data["policy_module"]),
            loaded_keys=int(data["loaded_keys"]),
            model_keys=int(data["model_keys"]),
            verified=bool(data.get("verified", False)),
            acceptance=str(data.get("acceptance", "unverified")),
            visual_weights_source=str(data.get("visual_weights_source", "")),
            active_missing=tuple(data.get("active_missing", ())),
            unexpected=tuple(data.get("unexpected", ())),
            shape_mismatch=tuple(
                str(item) if not isinstance(item, str) else item
                for item in data.get("shape_mismatch", ())
            ),
            inactive_keys=tuple(data.get("inactive_keys", ())),
            vision_keys_loaded=tuple(data.get("vision_keys_loaded", ())),
            temporal_teacher=bool(data.get("temporal_teacher", False)),
            audit_sha256=str(data.get("audit_sha256", "")),
        )
        if not audit.audit_sha256:
            object.__setattr__(audit, "audit_sha256", _json_sha256(audit.as_dict()))
        return audit

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class VisionGeometry:
    """Geometry read from the loaded visual modules, never from defaults."""

    hidden_size: int
    grid_height: int
    grid_width: int
    patch_size: int
    image_size: int
    num_positions: int
    encoder_layers: int
    projector_in: int
    projector_out: int
    dtype: str
    post_layernorm_eps: float

    def __post_init__(self) -> None:
        positive = (
            self.hidden_size,
            self.grid_height,
            self.grid_width,
            self.patch_size,
            self.image_size,
            self.num_positions,
            self.encoder_layers,
            self.projector_in,
            self.projector_out,
        )
        if any(int(value) <= 0 for value in positive):
            raise RawStemContractError("all geometry dimensions must be positive")
        if self.num_positions != self.grid_height * self.grid_width:
            raise RawStemContractError("position count does not match the grid")
        if self.image_size != self.grid_height * self.patch_size:
            raise RawStemContractError("image_size, grid and patch_size disagree")
        if self.image_size != self.grid_width * self.patch_size:
            raise RawStemContractError("square SigLIP geometry required by this front end")
        if self.projector_in != self.hidden_size:
            raise RawStemContractError("projector input width differs from the vision width")

    @property
    def num_patches(self) -> int:
        return self.num_positions

    def physical_size(self) -> tuple[int, int]:
        return self.image_size, self.image_size

    def input_channels(self, input_mode: str) -> int:
        return 4 if input_mode == "bayer4_packed" else 3

    def packed_patch_stride(self, input_mode: str) -> int:
        if input_mode != "bayer4_packed":
            return self.patch_size
        if self.patch_size % 2:
            raise RawStemContractError(
                "Bayer packed assets require an even physical patch size; "
                f"checkpoint patch_size={self.patch_size} is unsupported"
            )
        return self.patch_size // 2

    def expected_spatial_shape(self, input_mode: str) -> tuple[int, int]:
        if input_mode not in INPUT_MODES:
            raise RawStemContractError(f"unknown input mode {input_mode!r}")
        if input_mode != "bayer4_packed":
            return self.image_size, self.image_size
        if self.image_size % 2:
            raise RawStemContractError("Bayer physical field of view must be even")
        return self.image_size // 2, self.image_size // 2

    @classmethod
    def from_modules(cls, vision_tower: nn.Module, projector: nn.Module) -> VisionGeometry:
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        embeddings = getattr(vision_model, "embeddings", None)
        encoder = getattr(vision_model, "encoder", None)
        post = getattr(vision_model, "post_layernorm", None)
        position = getattr(getattr(embeddings, "position_embedding", None), "weight", None)
        linear = getattr(projector, "linear", None)
        if not isinstance(position, torch.Tensor):
            raise RawStemContractError("vision tower lacks embeddings.position_embedding")
        if encoder is None or post is None or not isinstance(linear, nn.Linear):
            raise RawStemContractError("vision tower/projector layout does not match OpenPI")
        if position.ndim != 2:
            raise RawStemContractError("position embedding must be [positions, hidden]")
        count, hidden = int(position.shape[0]), int(position.shape[1])
        side = math.isqrt(count)
        if side * side != count:
            raise RawStemContractError("SigLIP position count is not a square grid")
        config = getattr(vision_model, "config", None)
        patch_size = getattr(config, "patch_size", None)
        image_size = getattr(config, "image_size", None)
        if patch_size is None or image_size is None:
            raise RawStemContractError("vision config lacks patch_size/image_size")
        layers = getattr(encoder, "layers", None)
        if layers is None:
            raise RawStemContractError("vision encoder lacks layers")
        return cls(
            hidden_size=hidden,
            grid_height=side,
            grid_width=side,
            patch_size=int(patch_size),
            image_size=int(image_size),
            num_positions=count,
            encoder_layers=len(layers),
            projector_in=int(linear.in_features),
            projector_out=int(linear.out_features),
            dtype=str(position.dtype).replace("torch.", ""),
            post_layernorm_eps=float(getattr(post, "eps", 1e-5)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VisionGeometry:
        required = set(cls.__dataclass_fields__)
        missing = required - set(data)
        if missing:
            raise RawStemContractError(f"geometry artifact missing fields: {sorted(missing)}")
        return cls(**{key: data[key] for key in required})


# --------------------------------------------------------------------------- #
# Inputs and packing
# --------------------------------------------------------------------------- #
def _mask_to_bchw(mask: Any, batch: int, cameras: int, height: int, width: int, name: str):
    tensor = torch.as_tensor(mask, dtype=torch.bool)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0).expand(batch, cameras, height, width)
    elif tensor.ndim == 3:
        if tensor.shape != (batch, height, width):
            raise RawStemContractError(
                f"{name} [B,H,W] must be {(batch, height, width)}, got {tuple(tensor.shape)}"
            )
        tensor = tensor.unsqueeze(1).expand(batch, cameras, height, width)
    elif tensor.ndim == 4:
        if tensor.shape == (batch, height, width):
            tensor = tensor.unsqueeze(1).expand(batch, cameras, height, width)
        elif tensor.shape[1] == 1:
            tensor = tensor.expand(batch, cameras, height, width)
        elif tensor.shape != (batch, cameras, height, width):
            raise RawStemContractError(
                f"{name} expected {(batch, cameras, height, width)}, got {tuple(tensor.shape)}"
            )
    else:
        raise RawStemContractError(f"{name} must be [B,C,H,W], [B,H,W], [H,W] or [B,1,H,W]")
    return tensor.contiguous()


def _camera_origins(value: Any, cameras: int) -> tuple[tuple[int, int], ...]:
    if value is None:
        return tuple((0, 0) for _ in range(cameras))
    origins = tuple(tuple(int(v) for v in item) for item in value)
    if len(origins) == 1 and cameras > 1:
        origins = origins * cameras
    if len(origins) != cameras or any(len(item) != 2 for item in origins):
        raise RawStemContractError("physical_origins must contain one (y, x) pair per camera")
    return origins


@dataclass(slots=True)
class RawStemInput:
    """One batch of camera frames with explicit measurement metadata.

    ``values`` is ``[B, C, channels, H, W]``.  For Bayer the ``H, W`` are the
    packed half-resolution shape and channels are four row-major CFA planes.
    Missing metadata stays ``None`` and is reported as ``unavailable``; this
    class never invents a calibration, exposure, noise or mask.
    """

    values: torch.Tensor
    input_mode: str
    camera_ids: tuple[str, ...]
    units: str
    valid_mask: Any = None
    bad_mask: Any = None
    saturated_mask: Any = None
    missing_mask: Any = None
    black_level: Any = None
    white_level: Any = None
    exposure: Any = None
    noise: Any = None
    cfa_pattern: str = ""
    physical_origins: Any = None
    camera_present: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise RawStemContractError("values must be a torch.Tensor")
        if self.values.ndim != 5:
            raise RawStemContractError("values must be [B,C,channels,H,W]")
        if self.values.shape[0] < 1 or self.values.shape[1] < 1:
            raise RawStemContractError("values must contain at least one batch and camera")
        if self.input_mode not in INPUT_MODES:
            raise RawStemContractError(f"unknown input mode {self.input_mode!r}")
        cameras = tuple(self.camera_ids)
        if not cameras or len(set(cameras)) != len(cameras):
            raise RawStemContractError("camera_ids must be nonempty and unique")
        if len(cameras) != self.values.shape[1]:
            raise RawStemContractError("camera_ids length must match the camera axis")
        object.__setattr__(self, "camera_ids", cameras)
        allowed_units = {"raw_dn", "normalized_0_1", "cache_units", "teacher_rgb_minus1_1"}
        if self.units not in allowed_units:
            raise RawStemContractError(f"unknown units {self.units!r}")
        if self.input_mode == "rgb3_control" and self.units != "teacher_rgb_minus1_1":
            raise RawStemContractError(
                "RGB control must declare teacher_rgb_minus1_1 and use the actual "
                "teacher-preprocessed tensor"
            )
        if self.input_mode != "rgb3_control" and self.units == "teacher_rgb_minus1_1":
            raise RawStemContractError("teacher_rgb_minus1_1 is reserved for RGB control")
        if self.input_mode == "bayer4_packed":
            if self.cfa_pattern not in BAYER_CFA_PATTERNS:
                raise RawStemContractError("Bayer input requires a declared CFA pattern")
        elif self.cfa_pattern:
            raise RawStemContractError("CFA pattern is only valid for bayer4_packed")
        if (self.black_level is None) != (self.white_level is None):
            raise RawStemContractError("black_level and white_level must be provided together")
        for name in ("exposure", "noise"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                tensor = torch.as_tensor(value)
            except (TypeError, ValueError) as exc:
                raise RawStemContractError(f"{name} is not numeric") from exc
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise RawStemContractError(f"{name} contains non-finite values")
        object.__setattr__(
            self, "physical_origins", _camera_origins(self.physical_origins, len(cameras))
        )

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def channels(self) -> int:
        return int(self.values.shape[2])

    def availability(self) -> dict[str, str]:
        bayer = self.input_mode == "bayer4_packed"
        levels = self.black_level is not None and self.white_level is not None
        return {
            "units": self.units,
            "valid_mask": "available" if self.valid_mask is not None else "unavailable",
            "bad_mask": "available" if self.bad_mask is not None else "unavailable",
            "saturated_mask": "available" if self.saturated_mask is not None else "unavailable",
            "missing_mask": "available" if self.missing_mask is not None else "unavailable",
            "calibration": (
                "available"
                if levels and self.units == "raw_dn"
                else "available_not_applied"
                if levels
                else "unavailable"
            ),
            "exposure": "available" if self.exposure is not None else "unavailable",
            "noise": "available" if self.noise is not None else "unavailable",
            "cfa": "available" if bayer else "not_applicable",
        }


@dataclass(frozen=True, slots=True)
class BayerPacked:
    values: torch.Tensor
    valid_mask: torch.Tensor
    bad_mask: torch.Tensor | None
    saturated_mask: torch.Tensor | None
    cfa_pattern: str
    plane_order: tuple[str, ...]
    physical_offsets: tuple[tuple[int, int], ...]

    def to_input(
        self,
        *,
        camera_ids: Sequence[str] = ("camera_0",),
        units: str = "raw_dn",
        black_level: Any = None,
        white_level: Any = None,
        exposure: Any = None,
        noise: Any = None,
        physical_origins: Any = None,
    ) -> RawStemInput:
        if len(camera_ids) != 1:
            raise RawStemContractError("pack_bayer4 supports one camera per packed input")
        if self.values.ndim == 4:
            values = self.values.unsqueeze(1)
            valid = self.valid_mask.unsqueeze(1) if self.valid_mask.ndim == 4 else self.valid_mask
        else:
            values = self.values
            valid = self.valid_mask
        if values.ndim != 5 or valid.ndim != 5:
            raise RawStemContractError("packed Bayer tensors must include a camera axis")
        valid = valid.all(dim=2) if valid.ndim == 5 else valid
        bad = self.bad_mask
        saturated = self.saturated_mask
        if bad is not None and bad.ndim == 4:
            bad = bad.unsqueeze(1).any(dim=2)
        if saturated is not None and saturated.ndim == 4:
            saturated = saturated.unsqueeze(1).any(dim=2)
        return RawStemInput(
            values=values,
            input_mode="bayer4_packed",
            camera_ids=tuple(camera_ids),
            units=units,
            valid_mask=valid,
            bad_mask=bad,
            saturated_mask=saturated,
            black_level=black_level,
            white_level=white_level,
            exposure=exposure,
            noise=noise,
            cfa_pattern=self.cfa_pattern,
            physical_origins=physical_origins,
        )


def pack_bayer4(
    raw: torch.Tensor,
    cfa_pattern: str,
    *,
    valid_mask: Any = None,
    bad_mask: Any = None,
    saturated_mask: Any = None,
) -> BayerPacked:
    """Pack a mosaic ``[B,1,H,W]`` (or ``[B,H,W]``) into four CFA planes.

    Plane order is row-major ``(0,0), (0,1), (1,0), (1,1)``; ``plane_order``
    records the colour at each plane.  This is a real spatial packing, not a
    resize; every physical pixel appears exactly once.
    """
    if cfa_pattern not in BAYER_CFA_PATTERNS:
        raise RawStemContractError(f"unknown CFA pattern {cfa_pattern!r}")
    value = torch.as_tensor(raw)
    if value.ndim == 4:
        if value.shape[1] != 1:
            raise RawStemContractError("mosaic raw must have one channel")
        value = value[:, 0]
    if value.ndim != 3:
        raise RawStemContractError("mosaic raw must be [B,1,H,W] or [B,H,W]")
    batch, height, width = value.shape
    if height % 2 or width % 2:
        raise RawStemContractError("Bayer physical height/width must both be even")
    cells = []
    cell_masks = []
    for offset_y, offset_x in _CFA_OFFSETS:
        cells.append(value[:, offset_y::2, offset_x::2])
        if valid_mask is None:
            cell_masks.append(torch.ones_like(cells[-1], dtype=torch.bool))
        else:
            mask = _mask_to_bchw(valid_mask, batch, 1, height, width, "valid_mask")[:, 0]
            cell_masks.append(mask[:, offset_y::2, offset_x::2])
    planes = torch.stack(cells, dim=1).contiguous()
    valid = torch.stack(cell_masks, dim=1).contiguous()
    bad = None
    if bad_mask is not None:
        mask = _mask_to_bchw(bad_mask, batch, 1, height, width, "bad_mask")[:, 0]
        bad = torch.stack([mask[:, oy::2, ox::2] for oy, ox in _CFA_OFFSETS], dim=1)
    saturated = None
    if saturated_mask is not None:
        mask = _mask_to_bchw(saturated_mask, batch, 1, height, width, "saturated_mask")[:, 0]
        saturated = torch.stack([mask[:, oy::2, ox::2] for oy, ox in _CFA_OFFSETS], dim=1)
    return BayerPacked(
        values=planes,
        valid_mask=valid,
        bad_mask=bad,
        saturated_mask=saturated,
        cfa_pattern=cfa_pattern,
        plane_order=_CFA_COLORS[cfa_pattern],
        physical_offsets=_CFA_OFFSETS,
    )


def pack_bayer_input(
    raw: torch.Tensor,
    cfa_pattern: str,
    *,
    camera_id: str = "camera_0",
    units: str = "raw_dn",
    black_level: Any = None,
    white_level: Any = None,
    exposure: Any = None,
    noise: Any = None,
    valid_mask: Any = None,
    bad_mask: Any = None,
    saturated_mask: Any = None,
    physical_origin: tuple[int, int] = (0, 0),
) -> RawStemInput:
    packed = pack_bayer4(
        raw,
        cfa_pattern,
        valid_mask=valid_mask,
        bad_mask=bad_mask,
        saturated_mask=saturated_mask,
    )
    return packed.to_input(
        camera_ids=(camera_id,),
        units=units,
        black_level=black_level,
        white_level=white_level,
        exposure=exposure,
        noise=noise,
        physical_origins=(physical_origin,),
    )


# --------------------------------------------------------------------------- #
# Stem and shared front-end computation
# --------------------------------------------------------------------------- #
class LocalConvStem(nn.Module):
    """Small local ``conv64`` plus a patch convolution producing patch features.

    The patch convolution is the only module that replaces the checkpoint's
    original ``patch_embedding``.  Its output is already ``[B, hidden, gh, gw]``;
    the inherited SigLIP embeddings then flatten, transpose and add positions
    exactly as in the inspected source.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_stride: int,
        *,
        conv_channels: int = 64,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if min(in_channels, hidden_size, patch_stride, conv_channels) < 1:
            raise RawStemContractError("stem dimensions must be positive")
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.patch_stride = int(patch_stride)
        self.conv_channels = int(conv_channels)
        self.conv64 = nn.Conv2d(self.in_channels, self.conv_channels, 3, padding=1)
        self.activation = nn.GELU()
        self.patch_embedding = nn.Conv2d(
            self.conv_channels, self.hidden_size, self.patch_stride, self.patch_stride
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            nn.init.kaiming_uniform_(self.conv64.weight, a=math.sqrt(5))
            nn.init.zeros_(self.conv64.bias)
            nn.init.kaiming_uniform_(self.patch_embedding.weight, a=math.sqrt(5))
            nn.init.zeros_(self.patch_embedding.bias)
        self.to(device=device, dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Needed by the original embedding's dtype lookup."""
        return self.patch_embedding.weight

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(self.activation(self.conv64(values)))


@dataclass(slots=True)
class RawStemOutput:
    """Projected tokens plus the unimplemented pre-projector features."""

    tokens: VisualTokens
    hidden_tokens: torch.Tensor
    projected_tokens: torch.Tensor
    support_fraction: torch.Tensor
    availability: tuple[dict[str, str], ...]
    measurement_metadata: dict[str, Any]
    geometry: VisionGeometry
    input_mode: str

    def visual_tokens(self, bridge: Any = None) -> VisualTokens:
        assert_bridge_has_no_projection(bridge)
        return self.tokens

    # Convenience pass-throughs keep the result usable where a bare N02
    # ``VisualTokens`` is expected while retaining diagnostics.
    @property
    def values(self) -> torch.Tensor:
        return self.tokens.values

    @property
    def mask(self) -> torch.Tensor:
        return self.tokens.mask

    @property
    def camera_ids(self) -> tuple:
        return self.tokens.camera_ids

    @property
    def positions(self) -> torch.Tensor:
        return self.tokens.positions

    @property
    def validity(self):
        return self.tokens.validity


def _level_tensor(
    value: Any,
    *,
    batch: int,
    cameras: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1, 1).expand(batch, cameras, channels)
    elif tensor.ndim == 1:
        if tensor.numel() not in (1, channels):
            raise RawStemContractError(f"{name} must have one or {channels} entries")
        tensor = tensor.reshape(1, 1, -1).expand(batch, cameras, channels)
    elif tensor.ndim == 2:
        if tensor.shape != (cameras, channels):
            raise RawStemContractError(f"{name} per-camera shape must be {(cameras, channels)}")
        tensor = tensor.unsqueeze(0).expand(batch, cameras, channels)
    elif tensor.ndim == 3:
        if tensor.shape != (batch, cameras, channels):
            raise RawStemContractError(
                f"{name} expected {(batch, cameras, channels)}, got {tuple(tensor.shape)}"
            )
    else:
        raise RawStemContractError(f"{name} has unsupported shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RawStemContractError(f"{name} contains non-finite values")
    return tensor


def _prepare_measurement(input: RawStemInput, geometry: VisionGeometry) -> dict[str, Any]:
    values = input.values
    batch, cameras, channels, height, width = values.shape
    expected_channels = geometry.input_channels(input.input_mode)
    if channels != expected_channels:
        raise RawStemContractError(
            f"{input.input_mode} requires {expected_channels} channels, got {channels}"
        )
    expected_h, expected_w = geometry.expected_spatial_shape(input.input_mode)
    if (height, width) != (expected_h, expected_w):
        raise RawStemContractError(
            f"{input.input_mode} requires packed spatial shape {(expected_h, expected_w)} "
            f"for the loaded {geometry.grid_height}x{geometry.grid_width} grid and patch "
            f"{geometry.patch_size}; got {(height, width)}. Token interpolation is not used."
        )
    origins = input.physical_origins
    assert origins is not None
    for origin_y, origin_x in origins:
        if input.input_mode == "bayer4_packed" and (origin_y % 2 or origin_x % 2):
            raise RawStemContractError("Bayer physical crop origins must be even aligned")
    if input.input_mode == "bayer4_packed" and geometry.patch_size % 2:
        raise RawStemContractError("Bayer physical patch size must be even")

    valid = (
        torch.ones((batch, cameras, height, width), dtype=torch.bool, device=values.device)
        if input.valid_mask is None
        else _mask_to_bchw(input.valid_mask, batch, cameras, height, width, "valid_mask")
    )
    bad = (
        torch.zeros_like(valid)
        if input.bad_mask is None
        else _mask_to_bchw(input.bad_mask, batch, cameras, height, width, "bad_mask")
    )
    saturated = (
        torch.zeros_like(valid)
        if input.saturated_mask is None
        else _mask_to_bchw(input.saturated_mask, batch, cameras, height, width, "saturated_mask")
    )
    missing = (
        torch.zeros_like(valid)
        if input.missing_mask is None
        else _mask_to_bchw(input.missing_mask, batch, cameras, height, width, "missing_mask")
    )
    present = (
        torch.ones((batch, cameras), dtype=torch.bool, device=values.device)
        if input.camera_present is None
        else torch.as_tensor(input.camera_present, dtype=torch.bool, device=values.device)
    )
    if present.shape != (batch, cameras):
        raise RawStemContractError("camera_present must be [B,C]")
    valid = valid.to(values.device)
    bad = bad.to(values.device)
    saturated = saturated.to(values.device)
    missing = missing.to(values.device)
    usable = valid & ~bad & ~saturated & ~missing & present[:, :, None, None]
    floating = values.to(torch.float32)
    supported_pixels = usable.unsqueeze(2).expand_as(floating)
    if (supported_pixels & ~torch.isfinite(floating)).any():
        raise RawStemContractError("a supported pixel contains a non-finite value")

    levels = None
    if input.black_level is not None:
        black = _level_tensor(
            input.black_level,
            batch=batch,
            cameras=cameras,
            channels=channels,
            device=values.device,
            dtype=torch.float32,
            name="black_level",
        )
        white = _level_tensor(
            input.white_level,
            batch=batch,
            cameras=cameras,
            channels=channels,
            device=values.device,
            dtype=torch.float32,
            name="white_level",
        )
        if (black < 0).any():
            raise RawStemContractError("black_level must be nonnegative")
        if not (white > black).all():
            raise RawStemContractError("white_level must exceed black_level")
        levels = (black, white)

    if input.input_mode == "rgb3_control":
        supported = supported_pixels
        if supported.any():
            low = floating[supported].min()
            high = floating[supported].max()
            if low < -1.0 - 1e-5 or high > 1.0 + 1e-5:
                raise RawStemContractError(
                    "RGB control values must be the teacher-preprocessed [-1,1] tensor; "
                    "no resize/pad/gamma is applied here"
                )
    elif input.units == "normalized_0_1":
        supported = supported_pixels
        if supported.any():
            if floating[supported].min() < -1e-5 or floating[supported].max() > 1.0 + 1e-5:
                raise RawStemContractError("normalized_0_1 values are outside [0,1]")

    normalized = floating
    if levels is not None and input.units == "raw_dn":
        black, white = levels
        normalized = (floating - black[..., None, None]) / (
            white[..., None, None] - black[..., None, None]
        )
    normalized = torch.where(usable[:, :, None, :, :], normalized, torch.zeros_like(normalized))
    return {
        "values": normalized,
        "usable": usable,
        "bad": bad,
        "saturated": saturated,
        "missing": missing,
        "present": present,
        "channels": channels,
        "height": height,
        "width": width,
        "origins": origins,
    }


def _patch_pool(values: torch.Tensor, stride: int, mode: str) -> torch.Tensor:
    if mode == "max":
        return F.max_pool2d(values, kernel_size=stride, stride=stride)
    return F.avg_pool2d(values, kernel_size=stride, stride=stride)


def _forward_frontend(
    vision_tower: nn.Module,
    projector: nn.Module,
    geometry: VisionGeometry,
    input: RawStemInput,
) -> RawStemOutput:
    """Run the shared exact SigLIP semantics after the replacement stem."""
    prepared = _prepare_measurement(input, geometry)
    values = prepared["values"]
    batch, cameras = values.shape[:2]
    flat = values.flatten(0, 1)
    patch_stride = geometry.packed_patch_stride(input.input_mode)
    stem = vision_tower.vision_model.embeddings.patch_embedding
    if not isinstance(stem, LocalConvStem):
        raise RawStemContractError("vision patch_embedding must be the RAW stem")
    flat = flat.to(stem.conv64.weight.dtype)
    expected = (geometry.grid_height, geometry.grid_width)
    patch_features = stem(flat)
    if tuple(patch_features.shape[-2:]) != expected:
        raise RawStemContractError(
            f"stem grid {tuple(patch_features.shape[-2:])} does not equal "
            f"checkpoint grid {expected}"
        )
    patch_tokens = patch_features.flatten(2).transpose(1, 2)
    embeddings = vision_tower.vision_model.embeddings
    target_dtype = embeddings.position_embedding.weight.dtype
    patch_tokens = patch_tokens.to(target_dtype)
    position = embeddings.position_embedding(embeddings.position_ids).to(target_dtype)
    hidden = patch_tokens + position
    encoder = vision_tower.vision_model.encoder
    first_attention = getattr(encoder.layers[0], "self_attn", None) if encoder.layers else None
    if first_attention is not None and first_attention.q_proj.weight.dtype == torch.bfloat16:
        hidden = hidden.to(torch.bfloat16)
    encoder_outputs = encoder(inputs_embeds=hidden)
    hidden = encoder_outputs.last_hidden_state
    hidden = vision_tower.vision_model.post_layernorm(hidden)
    projected = projector(hidden)

    usable = prepared["usable"].flatten(0, 1).to(torch.float32).unsqueeze(1)
    support = _patch_pool(usable, patch_stride, "avg")[:, 0]
    support = support.reshape(batch, cameras, -1)
    token_mask = (support >= 1.0 - 1e-6) & prepared["present"][:, :, None]
    token_mask = token_mask.reshape(batch, -1)

    per_token_bad = (
        _patch_pool(prepared["bad"].flatten(0, 1).float().unsqueeze(1), patch_stride, "max")[
            :, 0
        ].reshape(batch, -1)
        > 0.5
    )
    per_token_sat = (
        _patch_pool(prepared["saturated"].flatten(0, 1).float().unsqueeze(1), patch_stride, "max")[
            :, 0
        ].reshape(batch, -1)
        > 0.5
    )
    per_token_missing = (
        _patch_pool(prepared["missing"].flatten(0, 1).float().unsqueeze(1), patch_stride, "max")[
            :, 0
        ].reshape(batch, -1)
        > 0.5
    )
    missing_camera = (~prepared["present"])[:, :, None].expand(batch, cameras, geometry.num_patches)
    per_token_missing = per_token_missing | missing_camera.reshape(batch, -1)

    hidden = hidden.reshape(batch, cameras * geometry.num_patches, -1)
    projected = projected.reshape(batch, cameras * geometry.num_patches, -1)
    hidden = torch.where(token_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))
    projected = torch.where(token_mask.unsqueeze(-1), projected, torch.zeros_like(projected))

    grid_y = torch.arange(geometry.grid_height, device=values.device, dtype=torch.float32)
    grid_x = torch.arange(geometry.grid_width, device=values.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    base_y = ((yy + 0.5) * geometry.patch_size).reshape(1, -1, 1)
    base_x = ((xx + 0.5) * geometry.patch_size).reshape(1, -1, 1)
    origins = torch.as_tensor(prepared["origins"], device=values.device, dtype=torch.float32)
    physical_y = base_y + origins[:, 0].reshape(-1, 1, 1)
    physical_x = base_x + origins[:, 1].reshape(-1, 1, 1)
    patch_size = torch.full_like(physical_y, float(geometry.patch_size))
    positions = torch.cat((physical_y, physical_x, patch_size), dim=-1)
    positions = positions.unsqueeze(0).expand(batch, -1, -1, -1)
    positions = positions.reshape(batch, cameras * geometry.num_patches, 3)

    values_out = projected.reshape(batch, cameras * geometry.num_patches, -1)
    missing_out = per_token_missing.reshape(batch, -1)
    validity = Validity(
        valid_mask=token_mask,
        missing_mask=missing_out,
        saturated_mask=per_token_sat.reshape(batch, -1),
        reason="token valid only when every physical pixel/CFA cell is supported",
    )
    tokens = VisualTokens(
        values=values_out,
        mask=token_mask,
        camera_ids=input.camera_ids,
        positions=positions,
        scale=float(geometry.patch_size),
        validity=validity,
    )
    availability = tuple(input.availability() for _ in range(batch))
    metadata = {
        "input_mode": input.input_mode,
        "units": input.units,
        "cfa_pattern": input.cfa_pattern or None,
        "black_level": input.black_level,
        "white_level": input.white_level,
        "exposure": input.exposure,
        "noise": input.noise,
        "physical_origins": input.physical_origins,
        "token_bad_mask": per_token_bad.reshape(batch, -1),
        "geometry": geometry,
        "coordinate_frame": "physical_pixel_center_yx_patch_size",
    }
    return RawStemOutput(
        tokens=tokens,
        hidden_tokens=hidden.reshape(batch, cameras * geometry.num_patches, -1),
        projected_tokens=values_out,
        support_fraction=support.reshape(batch, -1),
        availability=availability,
        measurement_metadata=metadata,
        geometry=geometry,
        input_mode=input.input_mode,
    )


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #
def _resolve_openpi_modules(policy: Any) -> tuple[nn.Module, nn.Module, nn.Module]:
    vlm = getattr(policy, "vlm", policy)
    if not isinstance(vlm, nn.Module):
        raise RawStemContractError("policy/vlm is not an nn.Module")
    core = getattr(getattr(vlm, "model", None), "model", None)
    if core is not None and hasattr(core, "vision_tower"):
        vision_tower = core.vision_tower
        projector = getattr(core, "multi_modal_projector", None)
    else:
        vision_tower = getattr(vlm, "vision_tower", None)
        projector = getattr(vlm, "multi_modal_projector", None)
    if not isinstance(vision_tower, nn.Module) or not isinstance(projector, nn.Module):
        raise RawStemContractError(
            "expected OpenPIPaliGemma.model.model.vision_tower and multi_modal_projector"
        )
    if not callable(getattr(vlm, "embed_image", None)):
        raise RawStemContractError("policy vlm lacks the original embed_image")
    return vlm, vision_tower, projector


def _collect_visual_fingerprints(
    vision_tower: nn.Module, projector: nn.Module
) -> dict[str, TensorFingerprint]:
    fingerprints: dict[str, TensorFingerprint] = {}
    for name, tensor in vision_tower.named_parameters():
        if "patch_embedding" in name:
            continue
        key = _canonical_key("vision_tower." + name)
        fingerprints[key] = TensorFingerprint(
            name=key,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            sha256=sha256_tensor(tensor),
        )
    for name, tensor in projector.named_parameters():
        key = _canonical_key("multi_modal_projector." + name)
        fingerprints[key] = TensorFingerprint(
            name=key,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            sha256=sha256_tensor(tensor),
        )
    return fingerprints


@dataclass(frozen=True, slots=True)
class InitializationAudit:
    source_tensors: tuple[TensorFingerprint, ...]
    student_tensors: tuple[TensorFingerprint, ...]
    stem_tensors: tuple[str, ...]
    checkpoint_sha256: str
    load_audit_sha256: str
    evidence_level: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_tensors": [item.as_dict() for item in self.source_tensors],
            "student_tensors": [item.as_dict() for item in self.student_tensors],
            "stem_tensors": list(self.stem_tensors),
            "checkpoint_sha256": self.checkpoint_sha256,
            "load_audit_sha256": self.load_audit_sha256,
            "evidence_level": self.evidence_level,
        }


class RawStemSiglipStudent(nn.Module):
    """Inherited SigLIP tower + local stem, with frozen-vision semantics."""

    def __init__(
        self,
        *,
        vision_tower: nn.Module,
        projector: nn.Module,
        input_mode: str,
        evidence_level: str,
        checkpoint: CheckpointIdentity,
        load_audit: PolicyLoadEvidence | None,
        initialization_audit: InitializationAudit,
        unfreeze_input_blocks: int = 0,
        allow_input_block_unfreeze: bool = False,
    ) -> None:
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise RawStemContractError(f"unknown input mode {input_mode!r}")
        self.vision_tower = vision_tower
        self.projector = projector
        self.input_mode = input_mode
        self.evidence_level = evidence_level
        self.checkpoint_identity = checkpoint
        self.load_evidence = load_audit
        self.initialization_audit = initialization_audit
        self.geometry = VisionGeometry.from_modules(vision_tower, projector)
        self.bridge_projection = None
        self.backend_trainable = False
        self.history_supported = False
        self.input_domain = "raw_or_teacher_rgb_per_frame"
        self._unfreeze_input_blocks = 0
        self._scope = "unconfigured"
        if unfreeze_input_blocks:
            self.configure_trainable_scope(
                "stem_plus_input_blocks",
                input_blocks=unfreeze_input_blocks,
                explicit=allow_input_block_unfreeze,
            )
        else:
            self.configure_trainable_scope("stem_only")
        self.train(False)

    @property
    def stem(self) -> LocalConvStem:
        return self.vision_tower.vision_model.embeddings.patch_embedding

    @property
    def position_embedding(self) -> nn.Embedding:
        return self.vision_tower.vision_model.embeddings.position_embedding

    @property
    def post_layernorm(self) -> nn.Module:
        return self.vision_tower.vision_model.post_layernorm

    def train(self, mode: bool = True) -> RawStemSiglipStudent:
        """Keep every frozen inherited block in eval mode.

        Optional stage-2 input-near blocks follow ``mode``; all later encoder
        layers, the position embedding, post-layernorm and projector stay in
        eval so dropout cannot be enabled there by a caller.
        """
        super().train(bool(mode))
        self.vision_tower.eval()
        self.projector.eval()
        self.stem.train(bool(mode))
        layers = self.vision_tower.vision_model.encoder.layers
        for index, layer in enumerate(layers):
            layer.train(bool(mode) and index < self._unfreeze_input_blocks)
        return self

    def configure_trainable_scope(
        self,
        scope: str,
        *,
        input_blocks: int = 0,
        explicit: bool = False,
    ) -> dict[str, Any]:
        """Set an explicit stage scope; the backend is never part of this module."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._unfreeze_input_blocks = 0
        if scope == "all_frozen":
            pass
        elif scope == "stem_only":
            for parameter in self.stem.parameters():
                parameter.requires_grad_(True)
        elif scope == "stem_plus_input_blocks":
            if not explicit:
                raise RawStemContractError(
                    "unfreezing input blocks requires explicit=True; stage 2 is "
                    "not enabled automatically"
                )
            layers = self.vision_tower.vision_model.encoder.layers
            if not 1 <= int(input_blocks) <= len(layers):
                raise RawStemContractError("input_blocks must be within the encoder depth")
            self._unfreeze_input_blocks = int(input_blocks)
            for parameter in self.stem.parameters():
                parameter.requires_grad_(True)
            for layer in layers[: self._unfreeze_input_blocks]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        else:
            raise RawStemContractError(f"unknown trainable scope {scope!r}")
        self._scope = scope
        return self.trainable_report()

    def trainable_report(self) -> dict[str, Any]:
        named = [(name, parameter) for name, parameter in self.named_parameters()]
        trainable = [
            {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
            for name, parameter in named
            if parameter.requires_grad
        ]
        total = sum(parameter.numel() for _, parameter in named)
        trainable_total = sum(item["numel"] for item in trainable)
        return {
            "scope": self._scope,
            "input_blocks_unfrozen": self._unfreeze_input_blocks,
            "trainable_parameters": trainable_total,
            "frozen_parameters": total - trainable_total,
            "total_parameters": total,
            "trainable_names": [item["name"] for item in trainable],
            "trainable": trainable,
            "backend_trainable": False,
            "bridge_projection": None,
        }

    def trainable_parameters(self) -> dict[str, Any]:
        """Alias for the named trainable-parameter report."""
        return self.trainable_report()

    def gradient_report(self) -> dict[str, Any]:
        entries = []
        missing = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            entry = {
                "name": name,
                "shape": list(parameter.shape),
                "grad_present": grad is not None,
                "grad_norm": None if grad is None else float(grad.detach().float().norm()),
            }
            entries.append(entry)
            if grad is None:
                missing.append(name)
        return {
            "scope": self._scope,
            "entries": entries,
            "missing_gradients": missing,
            "all_trainable_have_gradients": not missing,
        }

    def forward(self, input: RawStemInput) -> RawStemOutput:
        if not isinstance(input, RawStemInput):
            raise RawStemContractError("forward requires a RawStemInput")
        if input.input_mode != self.input_mode:
            raise RawStemContractError(
                f"student front end is {self.input_mode!r}, input declares {input.input_mode!r}"
            )
        return _forward_frontend(self.vision_tower, self.projector, self.geometry, input)


def _build_frontend(
    policy: Any,
    *,
    input_mode: str,
    evidence_level: str,
    checkpoint: CheckpointIdentity,
    load_audit: PolicyLoadEvidence | None,
    stem_hidden: int,
    stem_seed: int,
    unfreeze_input_blocks: int,
    allow_input_block_unfreeze: bool,
) -> RawStemSiglipStudent:
    if input_mode not in INPUT_MODES:
        raise RawStemContractError(f"unknown input mode {input_mode!r}")
    vlm, vision_tower, projector = _resolve_openpi_modules(policy)
    geometry = VisionGeometry.from_modules(vision_tower, projector)
    if stem_hidden != 64:
        # The study recipe fixes conv64.  Keep the check explicit so a caller
        # cannot silently instantiate the wrong recipe.
        raise RawStemContractError("the v1 RAW stem requires stem_hidden=64")
    patch_stride = geometry.packed_patch_stride(input_mode)
    source_fingerprints = _collect_visual_fingerprints(vision_tower, projector)
    source_map = {key: item.sha256 for key, item in source_fingerprints.items()}
    if evidence_level == "real_policy":
        if type(vision_tower.vision_model).__module__ != SIGLIP_MODEL_MODULE:
            raise RawStemContractError(
                "real mode requires the pinned SigLIP modeling module, got "
                f"{type(vision_tower.vision_model).__module__}"
            )
        if type(projector).__module__ != OPENPI_PALIGEMMA_MODULE:
            raise RawStemContractError(
                "real mode requires the original OpenPIPaliGemma projector module, got "
                f"{type(projector).__module__}"
            )
        _validate_real_evidence(
            vlm=vlm,
            checkpoint=checkpoint,
            audit=load_audit,
            source_hashes=source_map,
        )
    else:
        if checkpoint.tensor_sha256:
            for key, digest in checkpoint.tensor_sha256.items():
                if key in source_map and source_map[key] != digest:
                    raise RawStemContractError(f"fixture tensor hash mismatch for {key}")

    # Copy before any release / replacement.  The original policy is never
    # mutated here, so the teacher and backend remain stable.
    tower_copy = copy.deepcopy(vision_tower)
    projector_copy = copy.deepcopy(projector)
    copied_map = {
        key: item.sha256
        for key, item in _collect_visual_fingerprints(tower_copy, projector_copy).items()
    }
    if copied_map != source_map:
        raise RawStemContractError("deep-copied visual tensors differ from the source policy")
    embeddings = tower_copy.vision_model.embeddings
    old_patch = embeddings.patch_embedding
    if not isinstance(old_patch, nn.Conv2d) or old_patch.out_channels != geometry.hidden_size:
        raise RawStemContractError("original patch_embedding does not match hidden width")
    stem = LocalConvStem(
        geometry.input_channels(input_mode),
        geometry.hidden_size,
        patch_stride,
        conv_channels=64,
        seed=stem_seed,
        dtype=embeddings.position_embedding.weight.dtype,
        device=embeddings.position_embedding.weight.device,
    )
    embeddings.patch_embedding = stem
    # Eager attention is the inspected source semantics for embed_image.
    config = getattr(tower_copy.vision_model, "config", None)
    if config is not None and hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"

    student = RawStemSiglipStudent(
        vision_tower=tower_copy,
        projector=projector_copy,
        input_mode=input_mode,
        evidence_level=evidence_level,
        checkpoint=checkpoint,
        load_audit=load_audit,
        initialization_audit=InitializationAudit(
            source_tensors=tuple(source_fingerprints[key] for key in sorted(source_fingerprints)),
            student_tensors=(),
            stem_tensors=(),
            checkpoint_sha256=checkpoint.sha256,
            load_audit_sha256="" if load_audit is None else load_audit.audit_sha256,
            evidence_level=evidence_level,
        ),
        unfreeze_input_blocks=unfreeze_input_blocks,
        allow_input_block_unfreeze=allow_input_block_unfreeze,
    )
    student_fingerprints = []
    stem_names = []
    for name, tensor in student.named_parameters():
        key = _canonical_key(name)
        if "patch_embedding" in key:
            stem_names.append(key)
        student_fingerprints.append(
            TensorFingerprint(
                name=key,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).replace("torch.", ""),
                sha256=sha256_tensor(tensor),
            )
        )
    student.initialization_audit = InitializationAudit(
        source_tensors=tuple(source_fingerprints[key] for key in sorted(source_fingerprints)),
        student_tensors=tuple(student_fingerprints),
        stem_tensors=tuple(sorted(stem_names)),
        checkpoint_sha256=checkpoint.sha256,
        load_audit_sha256="" if load_audit is None else load_audit.audit_sha256,
        evidence_level=evidence_level,
    )
    student.train(False)
    return student


def _validate_real_evidence(
    *,
    vlm: nn.Module,
    checkpoint: CheckpointIdentity,
    audit: PolicyLoadEvidence | None,
    source_hashes: Mapping[str, str],
) -> None:
    if audit is None:
        raise RawStemUnavailableError("real policy construction requires a full load audit")
    if not checkpoint.verified:
        raise RawStemUnavailableError("checkpoint identity is not verified")
    if not audit.verified:
        raise RawStemUnavailableError("policy load audit is not verified")
    if audit.acceptance != "accepted_real_policy":
        raise RawStemUnavailableError(
            "backbone/source audit is not real-policy acceptance; "
            f"got acceptance={audit.acceptance!r}"
        )
    if audit.temporal_teacher:
        raise RawStemUnavailableError("temporal/burst teacher declaration is not supported")
    if audit.checkpoint_sha256 != checkpoint.sha256:
        raise RawStemContractError("load audit checkpoint hash differs from checkpoint identity")
    if type(vlm).__module__ != OPENPI_PALIGEMMA_MODULE or type(vlm).__name__ != "OpenPIPaliGemma":
        raise RawStemContractError(
            "real mode requires the pinned OpenPIPaliGemma module path "
            f"{OPENPI_PALIGEMMA_MODULE}, got {type(vlm).__module__}.{type(vlm).__name__}"
        )
    if audit.policy_module != type(vlm).__module__:
        raise RawStemContractError("load audit policy module differs from the loaded policy")
    if audit.active_missing or audit.unexpected or audit.shape_mismatch:
        raise RawStemContractError(
            "load audit reports missing, unexpected or shape-mismatched keys"
        )
    if audit.loaded_keys <= 0 or audit.model_keys <= 0:
        raise RawStemContractError("load audit key counts are not usable")
    source = audit.visual_weights_source.lower()
    if not source or "random" in source or "generic" in source or "backbone" in source:
        raise RawStemUnavailableError(
            "visual weights must come from an audited actual policy checkpoint"
        )
    if not checkpoint.tensor_sha256:
        raise RawStemUnavailableError(
            "real mode requires per-tensor checkpoint hashes for inherited visual tensors"
        )
    missing = sorted(set(source_hashes) - set(checkpoint.tensor_sha256))
    if missing:
        raise RawStemContractError(f"checkpoint lacks hashes for inherited tensors: {missing}")
    for key, digest in source_hashes.items():
        if checkpoint.tensor_sha256[key] != digest:
            raise RawStemContractError(f"loaded policy tensor does not match checkpoint: {key}")
    loaded_visual = {_canonical_key(name) for name in audit.vision_keys_loaded}
    missing = sorted(set(source_hashes) - loaded_visual)
    if missing:
        raise RawStemContractError(f"load audit did not load visual keys: {missing}")
    checkpoint_path = Path(checkpoint.path)
    if not checkpoint_path.is_file():
        raise RawStemUnavailableError(f"checkpoint file not found: {checkpoint_path}")
    actual = sha256_file(checkpoint_path)
    if actual != checkpoint.sha256:
        raise RawStemContractError("checkpoint file SHA256 differs from the verified identity")


def build_raw_stem_student(
    policy: Any,
    *,
    input_mode: str,
    checkpoint: CheckpointIdentity,
    load_audit: PolicyLoadEvidence | Mapping[str, Any],
    evidence_level: str = "real_policy",
    stem_seed: int = 0,
    unfreeze_input_blocks: int = 0,
    allow_input_block_unfreeze: bool = False,
) -> RawStemSiglipStudent:
    """Build a real student from an already fully load-audited policy.

    The checkpoint file is hashed on this path, the full-policy audit must be
    ``accepted_real_policy`` and every inherited tensor must match the checkpoint
    tensor hashes.  Use :meth:`RawStemSiglipStudent.from_fixture` for the
    explicitly labelled fixture contract; there is no generic/random fallback.
    """
    if evidence_level != "real_policy":
        raise RawStemContractError(
            "build_raw_stem_student only accepts evidence_level='real_policy'; "
            "fixture construction is the explicit from_fixture path"
        )
    if checkpoint is None:
        raise RawStemUnavailableError("real construction requires a checkpoint identity")
    if load_audit is None:
        raise RawStemUnavailableError("real construction requires a full policy load audit")
    evidence = (
        load_audit
        if isinstance(load_audit, PolicyLoadEvidence)
        else PolicyLoadEvidence.from_mapping(load_audit)
    )
    return _build_frontend(
        policy,
        input_mode=input_mode,
        evidence_level=evidence_level,
        checkpoint=checkpoint,
        load_audit=evidence,
        stem_hidden=64,
        stem_seed=stem_seed,
        unfreeze_input_blocks=unfreeze_input_blocks,
        allow_input_block_unfreeze=allow_input_block_unfreeze,
    )


def _build_fixture_student(
    policy: Any,
    *,
    input_mode: str,
    stem_seed: int,
    source_hashes: Mapping[str, str] | None,
    unfreeze_input_blocks: int,
    allow_input_block_unfreeze: bool,
    fixture_id: str,
) -> RawStemSiglipStudent:
    if input_mode not in INPUT_MODES:
        raise RawStemContractError(f"unknown input mode {input_mode!r}")
    _, vision_tower, projector = _resolve_openpi_modules(policy)
    source_map = {
        key: item.sha256
        for key, item in _collect_visual_fingerprints(vision_tower, projector).items()
    }
    if source_hashes is None:
        source_hashes = source_map
    checkpoint = CheckpointIdentity(
        path=f"fixture://{fixture_id}",
        sha256="0" * 64,
        verified=False,
        tensor_sha256=dict(source_hashes),
    )
    return _build_frontend(
        policy,
        input_mode=input_mode,
        evidence_level="fixture_contract",
        checkpoint=checkpoint,
        load_audit=None,
        stem_hidden=64,
        stem_seed=stem_seed,
        unfreeze_input_blocks=unfreeze_input_blocks,
        allow_input_block_unfreeze=allow_input_block_unfreeze,
    )


# --------------------------------------------------------------------------- #
# Teacher wrapper
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TeacherOutput:
    projected: torch.Tensor
    sha256: str
    temporal: bool = False
    history: HistoryContract | None = None

    @property
    def last_hidden_state(self) -> None:
        """The original ``embed_image`` output is post-projector only."""
        return None


class SiglipTeacher:
    """Frozen eval wrapper around the original ``embed_image`` path."""

    def __init__(
        self,
        policy: Any,
        *,
        evidence_level: str,
        checkpoint: CheckpointIdentity | None = None,
        load_audit: PolicyLoadEvidence | None = None,
        temporal_frames: int = 1,
        burst_frames: int = 1,
        history_contract: HistoryContract | None = None,
        sequence_embed: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if history_contract is None:
            if int(temporal_frames) != 1 or int(burst_frames) != 1:
                raise RawStemUnavailableError(
                    "temporal teacher requires an explicit HistoryContract with frame ids, "
                    "timestamps, stride and reset; the legacy temporal_frames shortcut is "
                    "not a history declaration"
                )
            history_contract = HistoryContract()
        self.history = history_contract
        self.sequence_embed = sequence_embed
        declared_temporal = any(
            bool(getattr(policy, attribute, False))
            for attribute in ("temporal_teacher", "uses_temporal", "burst_teacher")
        )
        policy_burst = getattr(policy, "burst_frames", 1)
        self.policy = policy
        self.vlm, self.vision_tower, self.projector = _resolve_openpi_modules(policy)
        declared_temporal = declared_temporal or any(
            bool(getattr(self.vlm, attribute, False))
            for attribute in ("temporal_teacher", "uses_temporal", "burst_teacher")
        )
        declared_burst = getattr(self.vlm, "burst_frames", policy_burst)
        if declared_temporal or declared_burst not in (None, 1):
            if not history_contract.is_temporal:
                raise RawStemUnavailableError(
                    "policy declares temporal/burst behavior but no temporal HistoryContract "
                    "was supplied"
                )
        if history_contract.is_temporal and sequence_embed is None:
            raise RawStemUnavailableError(
                "temporal teacher adapter is not registered; provide a learned sequence_embed "
                "callback or use the raw-teacher adapter in owac.training.raw_distillation"
            )
        self.geometry = VisionGeometry.from_modules(self.vision_tower, self.projector)
        self.evidence_level = evidence_level
        self.checkpoint = checkpoint
        self.load_evidence = load_audit
        self.input_domain = "teacher_preprocessed_rgb3"
        self.raw_input_supported = False
        self.temporal_supported = sequence_embed is not None
        if evidence_level == "real_policy":
            if checkpoint is None or load_audit is None:
                raise RawStemUnavailableError("real teacher requires checkpoint and load audit")
            _validate_teacher_evidence_type(self.vlm)
            if type(self.vision_tower.vision_model).__module__ != SIGLIP_MODEL_MODULE:
                raise RawStemContractError(
                    "real teacher requires the pinned SigLIP modeling module"
                )
            if not checkpoint.verified or not load_audit.verified:
                raise RawStemUnavailableError("real teacher evidence is not verified")
            if load_audit.checkpoint_sha256 != checkpoint.sha256:
                raise RawStemContractError("teacher checkpoint hash differs from its load audit")
            if load_audit.acceptance != "accepted_real_policy":
                raise RawStemUnavailableError("teacher needs accepted real-policy evidence")
            source = load_audit.visual_weights_source.lower()
            if not source or any(token in source for token in ("random", "generic", "backbone")):
                raise RawStemUnavailableError("teacher visual weights are not an audited policy")
            if not Path(checkpoint.path).is_file():
                raise RawStemUnavailableError(
                    f"teacher checkpoint file not found: {checkpoint.path}"
                )
            if sha256_file(checkpoint.path) != checkpoint.sha256:
                raise RawStemContractError("teacher checkpoint file SHA256 mismatch")
            required = {
                _canonical_key("vision_tower." + name)
                for name, _ in self.vision_tower.named_parameters()
            }
            required |= {
                _canonical_key("multi_modal_projector." + name)
                for name, _ in self.projector.named_parameters()
            }
            loaded = {_canonical_key(name) for name in load_audit.vision_keys_loaded}
            missing = sorted(required - loaded)
            if missing:
                raise RawStemContractError(
                    f"teacher load audit does not cover visual tensors: {missing[:4]}"
                )
        elif evidence_level != "fixture_contract":
            raise RawStemContractError(
                "teacher evidence_level must be real_policy or fixture_contract"
            )
        self.vlm.eval()
        for parameter in self.vlm.parameters():
            parameter.requires_grad_(False)
        self.initial_state_hashes = {
            name: sha256_tensor(parameter)
            for name, parameter in self.vlm.named_parameters()
            if "vision_tower" in name or "multi_modal_projector" in name
        }

    @property
    def image_size(self) -> int:
        return self.geometry.image_size

    def eval(self) -> SiglipTeacher:
        self.vlm.eval()
        return self

    def __call__(self, preprocessed_rgb: torch.Tensor) -> TeacherOutput:
        if not isinstance(preprocessed_rgb, torch.Tensor):
            raise RawStemContractError("teacher input must be a tensor")
        if self.history.is_temporal:
            if self.sequence_embed is None:
                raise RawStemUnavailableError("temporal teacher adapter is not registered")
            if preprocessed_rgb.ndim != 5 or preprocessed_rgb.shape[2] != 3:
                raise RawStemContractError("temporal teacher input must be [B,T,3,H,W]")
            if int(preprocessed_rgb.shape[1]) != self.history.temporal_frames:
                raise RawStemContractError(
                    "temporal teacher input length differs from the HistoryContract"
                )
            expected = (self.geometry.image_size, self.geometry.image_size)
            if tuple(preprocessed_rgb.shape[-2:]) != expected:
                raise RawStemContractError("temporal teacher spatial shape mismatch")
            _check_dtype_and_finite(preprocessed_rgb, "teacher temporal RGB")
            self.vlm.eval()
            with torch.no_grad():
                output = self.sequence_embed(preprocessed_rgb)
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RawStemContractError("sequence_embed must return [B,T,N,D] teacher tokens")
            if tuple(output.shape[:2]) != (
                preprocessed_rgb.shape[0],
                self.history.temporal_frames,
            ):
                raise RawStemContractError("sequence_embed output history mismatch")
            detached = output.detach().clone()
            return TeacherOutput(
                projected=detached,
                sha256=sha256_tensor(detached),
                temporal=True,
                history=self.history,
            )
        if preprocessed_rgb.ndim != 4 or preprocessed_rgb.shape[1] != 3:
            raise RawStemContractError("teacher input must be [B,3,H,W] without history")
        expected = (self.geometry.image_size, self.geometry.image_size)
        if tuple(preprocessed_rgb.shape[-2:]) != expected:
            raise RawStemContractError(
                f"teacher input must be actual preprocessed {expected} RGB, "
                f"got {tuple(preprocessed_rgb.shape[-2:])}"
            )
        if not preprocessed_rgb.is_floating_point():
            raise RawStemContractError("teacher input must be floating point")
        _check_dtype_and_finite(preprocessed_rgb, "teacher preprocessed RGB")
        if preprocessed_rgb.numel():
            if preprocessed_rgb.min() < -1.0 - 1e-5 or preprocessed_rgb.max() > 1.0 + 1e-5:
                raise RawStemContractError(
                    "teacher input must be the policy-preprocessed [-1,1] tensor"
                )
        self.vlm.eval()
        with torch.no_grad():
            output = self.vlm.embed_image(preprocessed_rgb)
        if not isinstance(output, torch.Tensor):
            raise RawStemContractError("embed_image must return a tensor")
        expected_shape = (
            preprocessed_rgb.shape[0],
            self.geometry.num_positions,
            self.geometry.projector_out,
        )
        if tuple(output.shape) != expected_shape:
            raise RawStemContractError(
                f"embed_image shape {tuple(output.shape)} != expected {expected_shape}"
            )
        detached = output.detach().clone()
        return TeacherOutput(projected=detached, sha256=sha256_tensor(detached))

    def forward(self, preprocessed_rgb: torch.Tensor) -> TeacherOutput:
        return self.__call__(preprocessed_rgb)

    def current_visual_hashes(self) -> dict[str, str]:
        return {
            name: sha256_tensor(parameter)
            for name, parameter in self.vlm.named_parameters()
            if "vision_tower" in name or "multi_modal_projector" in name
        }


def _validate_teacher_evidence_type(vlm: nn.Module) -> None:
    if type(vlm).__module__ != OPENPI_PALIGEMMA_MODULE or type(vlm).__name__ != "OpenPIPaliGemma":
        raise RawStemContractError("real teacher requires the pinned OpenPIPaliGemma module")


# Attach the explicit fixture constructor without weakening the public factory.
def _from_fixture(
    cls,
    policy: Any,
    *,
    input_mode: str,
    fixture_id: str,
    stem_seed: int = 0,
    source_hashes: Mapping[str, str] | None = None,
    unfreeze_input_blocks: int = 0,
    allow_input_block_unfreeze: bool = False,
) -> RawStemSiglipStudent:
    return _build_fixture_student(
        policy,
        input_mode=input_mode,
        stem_seed=stem_seed,
        source_hashes=source_hashes,
        unfreeze_input_blocks=unfreeze_input_blocks,
        allow_input_block_unfreeze=allow_input_block_unfreeze,
        fixture_id=fixture_id,
    )


RawStemSiglipStudent.from_fixture = classmethod(_from_fixture)  # type: ignore[attr-defined]


def _teacher_from_fixture(
    cls,
    policy: Any,
    *,
    fixture_id: str = "fixture",
    temporal_frames: int = 1,
    burst_frames: int = 1,
    history_contract: HistoryContract | None = None,
    sequence_embed: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> SiglipTeacher:
    checkpoint = CheckpointIdentity(
        path=f"fixture://{fixture_id}",
        sha256="0" * 64,
        verified=False,
    )
    return cls(
        policy,
        evidence_level="fixture_contract",
        checkpoint=checkpoint,
        load_audit=None,
        temporal_frames=temporal_frames,
        burst_frames=burst_frames,
        history_contract=history_contract,
        sequence_embed=sequence_embed,
    )


SiglipTeacher.from_fixture = classmethod(_teacher_from_fixture)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Freeze / bridge guards
# --------------------------------------------------------------------------- #
def assert_backend_frozen(backend: nn.Module) -> dict[str, Any]:
    """Fail if any backend parameter would be updated; never unfreezes it."""
    trainable = [name for name, parameter in backend.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RawStemContractError(f"backend is not frozen: {trainable[:8]}")
    return {
        "backend_trainable": False,
        "trainable_names": [],
        "parameters": sum(parameter.numel() for parameter in backend.parameters()),
    }


def assert_bridge_has_no_projection(bridge: Any = None) -> bool:
    """Reject an extra learned projection at the visual-token bridge boundary."""
    if bridge is None:
        return True
    for attribute in ("projection", "hidden_projection", "projector"):
        module = getattr(bridge, attribute, None)
        if module is not None:
            raise RawStemContractError(
                f"bridge projection {attribute} must be None; the student inherits the "
                "only learned projector from the policy"
            )
    return True


# --------------------------------------------------------------------------- #
# Deployment export/loader
# --------------------------------------------------------------------------- #
class RawStemSiglipDeployment(nn.Module):
    """Exported front end only; no teacher, ISP or cache object is reachable."""

    def __init__(
        self,
        *,
        vision_tower: nn.Module,
        projector: nn.Module,
        geometry: VisionGeometry,
        input_mode: str,
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector
        self.geometry = geometry
        self.input_mode = input_mode
        self.metadata = dict(metadata)
        self.bridge_projection = None
        self.history_supported = False
        self.input_domain = f"raw_{input_mode}"
        self.vision_tower.eval()
        self.projector.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def stem(self) -> LocalConvStem:
        return self.vision_tower.vision_model.embeddings.patch_embedding

    def train(self, mode: bool = True) -> RawStemSiglipDeployment:
        """Deployment is always eval; its exported weights are frozen."""
        super().train(False)
        return self

    def forward(self, input: RawStemInput) -> RawStemOutput:
        if input.input_mode != self.input_mode:
            raise RawStemContractError("deployment input mode mismatch")
        return _forward_frontend(self.vision_tower, self.projector, self.geometry, input)

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        frontend_factory: Callable[[VisionGeometry, str], tuple[nn.Module, nn.Module]]
        | None = None,
        expected_checkpoint_sha256: str | None = None,
        expected_state_dict_sha256: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> RawStemSiglipDeployment:
        return load_student_artifact(
            path,
            frontend_factory=frontend_factory,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_state_dict_sha256=expected_state_dict_sha256,
            map_location=map_location,
        )


def _architecture_metadata(student: Any) -> dict[str, Any]:
    vision = student.vision_tower.vision_model
    config = getattr(vision, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        raw = config.to_dict()
    elif config is not None and hasattr(config, "__dict__"):
        raw = vars(config)
    else:
        raw = {}

    def clean(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        return str(value)

    projector = getattr(student, "projector", None)
    linear = getattr(projector, "linear", None)
    source_files = {}
    for module in (
        type(vision).__module__,
        type(student.vision_tower).__module__,
        type(projector).__module__ if projector is not None else "",
    ):
        try:
            source = Path(__import__("sys").modules[module].__file__)
        except Exception:
            continue
        if source.is_file():
            source_files[str(source)] = sha256_file(source)
    return {
        "attention_implementation": getattr(config, "_attn_implementation", "eager"),
        "vision_config_class": (
            None if config is None else f"{type(config).__module__}.{type(config).__name__}"
        ),
        "source_files": source_files,
        "vision_model_class": f"{type(vision).__module__}.{type(vision).__name__}",
        "vision_tower_class": (
            f"{type(student.vision_tower).__module__}.{type(student.vision_tower).__name__}"
        ),
        "projector_class": (
            None
            if projector is None
            else f"{type(projector).__module__}.{type(projector).__name__}"
        ),
        "projector_linear": (
            None
            if linear is None
            else {
                "in_features": int(linear.in_features),
                "out_features": int(linear.out_features),
                "bias": bool(linear.bias is not None),
            }
        ),
        "config": clean(raw),
    }


def export_student_artifact(
    student: RawStemSiglipStudent | RawStemSiglipDeployment,
    path: str | Path,
    *,
    backend_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any] | None = None,
    normalizer_identity: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    checkpoint_step: int = 0,
) -> dict[str, Any]:
    """Write ``frontend.json`` and ``state.pt`` as one strict deployment artifact."""
    if getattr(student, "evidence_level", "") == "real_policy":
        if not backend_identity or tokenizer_identity is None or normalizer_identity is None:
            raise RawStemUnavailableError(
                "real student export requires backend, tokenizer and normalizer identities"
            )
    target = Path(path)
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"artifact path is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    state = student.state_dict()
    if any("teacher" in name.lower() or "isp" in name.lower() for name in state):
        raise RawStemContractError("frontend state must not contain teacher/ISP tensors")
    state_path = target / "state.pt"
    torch.save({"format_version": STUDENT_ARTIFACT_VERSION, "state_dict": state}, state_path)
    state_sha = sha256_file(state_path)
    metadata = {
        "schema": STUDENT_ARTIFACT_SCHEMA,
        "format_version": STUDENT_ARTIFACT_VERSION,
        "evidence_level": getattr(student, "evidence_level", "deployment"),
        "input_mode": student.input_mode,
        "geometry": student.geometry.as_dict(),
        "vision_architecture": _architecture_metadata(student),
        "stem": {
            "kind": "local_conv64_patch_embedding",
            "conv_channels": student.stem.conv_channels,
            "in_channels": student.stem.in_channels,
            "hidden_size": student.stem.hidden_size,
            "patch_stride": student.stem.patch_stride,
        },
        "bridge_projection": None,
        "state_dict_sha256": state_sha,
        "state_dict_keys": sorted(state),
        "state_schema": {
            name: {
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
            }
            for name, tensor in state.items()
        },
        "step": int(checkpoint_step),
        "checkpoint": (
            student.checkpoint_identity.as_dict()
            if hasattr(student, "checkpoint_identity")
            else None
        ),
        "load_audit_sha256": (
            student.initialization_audit.load_audit_sha256
            if hasattr(student, "initialization_audit")
            else None
        ),
        "policy_acceptance": (
            student.load_evidence.acceptance
            if getattr(student, "load_evidence", None) is not None
            else None
        ),
        "temporal_teacher": (
            student.load_evidence.temporal_teacher
            if getattr(student, "load_evidence", None) is not None
            else False
        ),
        "initialization_audit": (
            student.initialization_audit.as_dict()
            if hasattr(student, "initialization_audit")
            else None
        ),
        "backend": dict(backend_identity),
        "tokenizer": None if tokenizer_identity is None else dict(tokenizer_identity),
        "normalizer": None if normalizer_identity is None else dict(normalizer_identity),
        "extra": {} if extra_metadata is None else dict(extra_metadata),
    }
    (target / "frontend.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_student_artifact(
    path: str | Path,
    *,
    frontend_factory: Callable[[VisionGeometry, str], tuple[nn.Module, nn.Module]] | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_state_dict_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> RawStemSiglipDeployment:
    """Load only exported front-end weights.

    ``frontend_factory(geometry, input_mode)`` must construct a *blank*
    architecture (vision tower with the original patch-embedding slot and the
    projector).  It must not load a teacher, ISP or cache.  Every exported
    tensor, shape and key is then loaded strictly from ``state.pt``.
    """
    target = Path(path)
    frontend_path = target / "frontend.json"
    state_path = target / "state.pt"
    if not frontend_path.is_file() or not state_path.is_file():
        raise RawStemUnavailableError(f"student artifact is incomplete: {target}")
    metadata = json.loads(frontend_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != STUDENT_ARTIFACT_SCHEMA:
        raise RawStemContractError("unknown student artifact schema")
    if metadata.get("format_version") != STUDENT_ARTIFACT_VERSION:
        raise RawStemContractError("unsupported student artifact version")
    if int(metadata.get("step", 0)) < 0:
        raise RawStemContractError("artifact checkpoint step must be nonnegative")
    if metadata.get("bridge_projection", "missing") is not None:
        raise RawStemContractError("artifact declares an extra bridge projection")
    recorded = str(metadata.get("state_dict_sha256", ""))
    actual = sha256_file(state_path)
    if recorded != actual:
        raise RawStemContractError("student state.pt hash mismatch")
    if expected_state_dict_sha256 is not None and expected_state_dict_sha256 != recorded:
        raise RawStemContractError("student state dictionary identity mismatch")
    checkpoint = metadata.get("checkpoint") or {}
    if metadata.get("evidence_level") == "real_policy":
        if expected_checkpoint_sha256 is None:
            raise RawStemUnavailableError(
                "real deployment load requires expected_checkpoint_sha256"
            )
        if checkpoint.get("sha256") != expected_checkpoint_sha256:
            raise RawStemContractError("student checkpoint identity mismatch")
        if not metadata.get("load_audit_sha256"):
            raise RawStemUnavailableError("real deployment artifact lacks load audit identity")
        if not checkpoint.get("verified", False):
            raise RawStemUnavailableError("real deployment checkpoint identity is not verified")
        if metadata.get("policy_acceptance") != "accepted_real_policy":
            raise RawStemUnavailableError(
                "real deployment artifact lacks accepted real-policy evidence"
            )
        for key in ("backend", "tokenizer", "normalizer"):
            if not metadata.get(key):
                raise RawStemUnavailableError(f"real deployment artifact lacks {key} identity")
        if metadata.get("temporal_teacher"):
            raise RawStemContractError("this deployment contract is single-frame only")
    elif expected_checkpoint_sha256 is not None:
        if checkpoint.get("sha256") != expected_checkpoint_sha256:
            raise RawStemContractError("student checkpoint identity mismatch")
    geometry = VisionGeometry.from_dict(metadata["geometry"])
    input_mode = str(metadata["input_mode"])
    if input_mode not in INPUT_MODES:
        raise RawStemContractError("artifact declares an unknown input mode")
    payload = torch.load(state_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != STUDENT_ARTIFACT_VERSION:
        raise RawStemContractError("student state payload has an unsupported format")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RawStemContractError("student state payload lacks a state_dict")
    recorded_keys = metadata.get("state_dict_keys")
    if not isinstance(recorded_keys, list) or sorted(state) != sorted(recorded_keys):
        raise RawStemContractError("student state keys differ from the artifact metadata")
    schema = metadata.get("state_schema")
    if schema is not None:
        if not isinstance(schema, Mapping) or sorted(schema) != sorted(state):
            raise RawStemContractError("state schema keys differ from state_dict")
        for name, tensor in state.items():
            expected = schema[name]
            if not isinstance(expected, Mapping):
                raise RawStemContractError("state schema entry is invalid")
            if str(expected.get("dtype")) != str(tensor.dtype).replace("torch.", ""):
                raise RawStemContractError(f"state schema dtype mismatch for {name!r}")
            if list(expected.get("shape", ())) != list(tensor.shape):
                raise RawStemContractError(f"state schema shape mismatch for {name!r}")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise RawStemContractError(f"state entry {name!r} is not a tensor")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RawStemContractError(f"state tensor {name!r} contains non-finite values")
    vision_tower, projector = (
        build_artifact_frontend_from_metadata(metadata)
        if frontend_factory is None
        else frontend_factory(geometry, input_mode)
    )
    factory_geometry = VisionGeometry.from_modules(vision_tower, projector)
    if factory_geometry != geometry:
        raise RawStemContractError("architecture factory geometry differs from artifact")
    stem_meta = metadata.get("stem", {})
    if stem_meta.get("kind") != "local_conv64_patch_embedding":
        raise RawStemContractError("artifact does not declare the local conv64 stem")
    if int(stem_meta.get("conv_channels", 0)) != 64:
        raise RawStemContractError("artifact does not declare the required conv64 stem")
    expected_channels = geometry.input_channels(input_mode)
    expected_stride = geometry.packed_patch_stride(input_mode)
    if int(stem_meta.get("in_channels", -1)) != expected_channels:
        raise RawStemContractError("artifact stem channels differ from the input mode")
    if int(stem_meta.get("hidden_size", -1)) != geometry.hidden_size:
        raise RawStemContractError("artifact stem hidden width differs from the checkpoint")
    if int(stem_meta.get("patch_stride", -1)) != expected_stride:
        raise RawStemContractError("artifact stem stride differs from the physical geometry")
    stem = LocalConvStem(
        int(stem_meta["in_channels"]),
        int(stem_meta["hidden_size"]),
        int(stem_meta["patch_stride"]),
        conv_channels=int(stem_meta["conv_channels"]),
        seed=0,
        dtype=geometry_parameter_dtype(vision_tower),
        device=geometry_parameter_device(vision_tower),
    )
    vision_tower.vision_model.embeddings.patch_embedding = stem
    deployment = RawStemSiglipDeployment(
        vision_tower=vision_tower,
        projector=projector,
        geometry=geometry,
        input_mode=input_mode,
        metadata=metadata,
    )
    parameter_map = dict(deployment.named_parameters())
    buffer_map = dict(deployment.named_buffers())
    for name, tensor in state.items():
        target = parameter_map.get(name, buffer_map.get(name))
        if target is None:
            continue
        if target.dtype != tensor.dtype:
            target.data = target.data.to(dtype=tensor.dtype)
    current = deployment.state_dict()
    if set(state) != set(current):
        raise RawStemContractError("student state keys differ from the target architecture")
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(current[name].shape):
            raise RawStemContractError(f"student state shape mismatch for {name!r}")
        if tensor.dtype != current[name].dtype:
            raise RawStemContractError(f"student state dtype mismatch for {name!r}")
    result = deployment.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RawStemContractError("student state has missing or unexpected keys")
    deployment.eval()
    for parameter in deployment.parameters():
        parameter.requires_grad_(False)
    return deployment


def geometry_parameter_device(vision_tower: nn.Module) -> torch.device:
    for parameter in vision_tower.parameters():
        if parameter.is_floating_point():
            return parameter.device
    return torch.device("cpu")


def geometry_parameter_dtype(vision_tower: nn.Module) -> torch.dtype:
    for parameter in vision_tower.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def build_artifact_frontend_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[nn.Module, nn.Module]:
    """Construct the allowlisted OpenPI SigLIP/projector from artifact metadata.

    The exported internal vision class is ``SiglipVisionTransformer``; the outer
    class is ``SiglipVisionModel``.  No policy/teacher/ISP/cache is constructed
    or copied.
    """
    from types import SimpleNamespace

    architecture = metadata.get("vision_architecture")
    if not isinstance(architecture, Mapping):
        raise RawStemContractError("artifact lacks serialized vision architecture")
    vision_class = str(architecture.get("vision_model_class", ""))
    tower_class = str(architecture.get("vision_tower_class", ""))
    projector_class = str(architecture.get("projector_class", ""))
    expected_internal = (
        "starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip."
        "SiglipVisionTransformer"
    )
    expected_outer = (
        "starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip.SiglipVisionModel"
    )
    expected_projector = "starVLA.model.modules.vlm.OpenPIPaliGemma.OpenPIMultiModalProjector"
    if vision_class != expected_internal:
        raise RawStemContractError(
            f"artifact internal vision class is not pinned SiglipVisionTransformer: {vision_class}"
        )
    if tower_class != expected_outer:
        raise RawStemContractError(
            f"artifact outer vision class is not pinned SiglipVisionModel: {tower_class}"
        )
    if projector_class != expected_projector:
        raise RawStemContractError("artifact projector is not the pinned OpenPI projector")
    config_dict = architecture.get("config")
    if not isinstance(config_dict, Mapping) or not config_dict:
        raise RawStemContractError("artifact vision config is absent")
    try:
        from starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip import (
            SiglipVisionModel,
        )
        from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIMultiModalProjector
        from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    except Exception as exc:
        raise RawStemUnavailableError(
            f"pinned StarVLA architecture modules are unavailable: {exc}"
        ) from exc
    config_kwargs = dict(config_dict)
    # Preserve full config fields, including projection_dim which may arrive via
    # **kwargs rather than an explicit signature parameter.
    try:
        vision_config = SiglipVisionConfig(**config_kwargs)
    except TypeError:
        import inspect

        signature = inspect.signature(SiglipVisionConfig.__init__)
        accepted = {
            key: value for key, value in config_kwargs.items() if key in signature.parameters
        }
        vision_config = SiglipVisionConfig(**accepted)
        for key, value in config_kwargs.items():
            if not hasattr(vision_config, key):
                setattr(vision_config, key, value)
    if not hasattr(vision_config, "projection_dim"):
        linear_meta = architecture.get("projector_linear") or {}
        if linear_meta.get("out_features") is None:
            raise RawStemContractError("vision config lacks projection_dim")
        vision_config.projection_dim = int(linear_meta["out_features"])
    if int(vision_config.hidden_size) <= 0 or int(vision_config.projection_dim) <= 0:
        raise RawStemContractError("vision config lacks hidden/projection dimensions")
    # to_dict omits this execution field; automatic SDPA selection changes the
    # BF16 numerical path of an otherwise identical reloaded student.
    if architecture.get("attention_implementation", "eager") != "eager":
        raise RawStemContractError("artifact attention must match the audited eager path")
    vision_config._attn_implementation = "eager"
    vision_model = SiglipVisionModel(vision_config)
    projector = OpenPIMultiModalProjector(SimpleNamespace(vision_config=vision_config))
    linear_meta = architecture.get("projector_linear")
    if linear_meta is not None:
        if (
            int(linear_meta["in_features"]) != int(projector.linear.in_features)
            or int(linear_meta["out_features"]) != int(projector.linear.out_features)
            or bool(linear_meta["bias"]) != bool(projector.linear.bias is not None)
        ):
            raise RawStemContractError(
                "projector linear metadata differs from constructed projector"
            )
    source_files = architecture.get("source_files")
    if isinstance(source_files, Mapping):
        for path_value, digest in source_files.items():
            source = Path(str(path_value))
            if not source.is_file():
                raise RawStemUnavailableError(f"pinned architecture source is absent: {source}")
            if sha256_file(source) != str(digest):
                raise RawStemContractError(f"pinned architecture source hash mismatch: {source}")
    return vision_model, projector


def default_artifact_frontend_factory(
    geometry: VisionGeometry, input_mode: str
) -> tuple[nn.Module, nn.Module]:
    raise RawStemContractError(
        "default artifact factory requires metadata; use load_student_artifact without "
        "frontend_factory"
    )

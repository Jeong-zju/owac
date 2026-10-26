"""Persistent causal temporal wrapper around the per-frame RAW-stem student.

This module is additive: the existing per-frame ``RawStemSiglipStudent`` and
``RawStemSiglipDeployment`` remain the single source of the inherited SigLIP
tower, position embedding, post-layernorm and original projector.  The temporal
wrapper runs the existing replacement stem on each frame of an explicit causal
six-frame RAW burst (or one actual teacher RGB per endpoint for the diagnostic
B arm), aggregates valid patch supports, applies a channelwise gated recurrent
state in patch space, and then runs the inherited tower exactly once.

It never generates RGB, never imports a teacher/ISP/cache module, and never
creates a second projection.  A raw-RAW packet has no teacher privilege; an
``rgb3_control`` packet is explicitly labelled a diagnostic teacher-RGB
privilege.

The state is explicit input/output.  There is no module-global hidden state.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from owac.baselines.raw_stem_siglip import (
    RawStemContractError,
    RawStemInput,
    RawStemSiglipDeployment,
    RawStemSiglipStudent,
    VisionGeometry,
    _forward_frontend,
    _patch_pool,
    _prepare_measurement,
    _stem_patch_tokens,
    export_student_artifact,
    load_student_artifact,
)

__all__ = [
    "TEMPORAL_HISTORY_LENGTH",
    "TEMPORAL_ENDPOINT_STRIDE",
    "TEMPORAL_MANIFEST_SCHEMA",
    "TEMPORAL_STATE_SCHEMA",
    "TemporalPacket",
    "TemporalRawStemStudent",
    "TemporalState",
    "causal_frame_ids",
    "causal_padding_mask",
    "export_temporal_student_artifact",
    "load_temporal_student_artifact",
]

TEMPORAL_HISTORY_LENGTH = 6
TEMPORAL_ENDPOINT_STRIDE = 5
TEMPORAL_MANIFEST_SCHEMA = "owac.raw-stem-temporal.manifest.v1"
TEMPORAL_STATE_SCHEMA = "owac.raw-stem-temporal.state.v1"
_TEMPORAL_CONFIG_SCHEMA = "owac.raw-stem-temporal.config.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def causal_frame_ids(endpoint: int) -> tuple[int, ...]:
    """Return the six causal frame ids ending at ``endpoint``.

    The first five positions are left-padded with frame 0 when the episode is
    shorter than six frames.  The returned ids are deliberately fixed; callers
    cannot silently reorder or supply a future frame.
    """
    endpoint = int(endpoint)
    if endpoint < 0:
        raise RawStemContractError("endpoint must be nonnegative")
    return tuple(max(0, endpoint - TEMPORAL_ENDPOINT_STRIDE + index) for index in range(6))


def causal_padding_mask(endpoint: int) -> tuple[bool, ...]:
    """True for duplicated frame-0 positions, false for actual acquisitions."""
    endpoint = int(endpoint)
    if endpoint < 0:
        raise RawStemContractError("endpoint must be nonnegative")
    actual = min(endpoint + 1, TEMPORAL_HISTORY_LENGTH)
    return tuple(index < TEMPORAL_HISTORY_LENGTH - actual for index in range(6))


@dataclass(frozen=True, slots=True)
class TemporalPacket:
    """One explicit causal burst and its identity.

    RAW packets contain exactly six ``RawStemInput`` values.  The last frame is
    the current endpoint; the first five may be left-padding duplicates and are
    excluded by ``padding_mask`` during aggregation.  ``rgb3_control`` packets
    contain exactly one actual teacher-preprocessed RGB at the endpoint and must
    not replicate it into six fake measurements.
    """

    frames: tuple[RawStemInput, ...]
    source_demo_id: str
    endpoint: int
    frame_ids: tuple[int, ...]
    padding_mask: tuple[bool, ...]
    reset: bool = False
    timestamps: tuple[float, ...] | None = None
    position_source_size: int | None = None
    position_target_size: int | None = None
    position_backmap: bool = False
    preprocess_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        object.__setattr__(self, "frames", frames)
        if not frames or not isinstance(frames[0], RawStemInput):
            raise RawStemContractError("temporal packet requires at least one RawStemInput")
        first = frames[0]
        control = first.input_mode == "rgb3_control"
        expected_length = 1 if control else TEMPORAL_HISTORY_LENGTH
        if len(frames) != expected_length:
            raise RawStemContractError(
                f"{'rgb3_control' if control else 'RAW'} temporal packet requires exactly "
                f"{expected_length} frame(s)"
            )
        if not isinstance(self.source_demo_id, str) or not self.source_demo_id:
            raise RawStemContractError("source_demo_id must be a nonempty string")
        endpoint = int(self.endpoint)
        if endpoint < 0:
            raise RawStemContractError("endpoint must be nonnegative")
        object.__setattr__(self, "endpoint", endpoint)
        if control:
            # One actual teacher-preprocessed RGB per endpoint.  It is already a
            # burst/state summary and must never be replicated into six fake
            # measurements.
            expected_ids = (endpoint,)
            expected_padding = (False,)
        else:
            expected_ids = causal_frame_ids(endpoint)
            expected_padding = causal_padding_mask(endpoint)
        if tuple(self.frame_ids) != expected_ids:
            raise RawStemContractError(
                f"frame_ids must be {expected_ids}, got {tuple(self.frame_ids)}"
            )
        if tuple(self.padding_mask) != expected_padding:
            raise RawStemContractError(
                f"padding_mask must be {expected_padding}, got {tuple(self.padding_mask)}"
            )
        if bool(self.padding_mask[-1]):
            raise RawStemContractError("the current endpoint frame cannot be padding")
        if not isinstance(self.reset, bool):
            raise RawStemContractError("reset must be boolean")
        if self.timestamps is not None:
            timestamps = tuple(float(value) for value in self.timestamps)
            if len(timestamps) != expected_length or not all(
                math.isfinite(value) for value in timestamps
            ):
                raise RawStemContractError(f"timestamps must be {expected_length} finite number(s)")
            for index in range(1, len(timestamps)):
                if timestamps[index] < timestamps[index - 1]:
                    raise RawStemContractError("timestamps must be nondecreasing")
                same_frame = expected_ids[index] == expected_ids[index - 1]
                same_time = timestamps[index] == timestamps[index - 1]
                if same_frame != same_time:
                    raise RawStemContractError(
                        "timestamps must repeat exactly for a repeated padded frame and "
                        "advance for each new frame id"
                    )
            object.__setattr__(self, "timestamps", timestamps)
        if not isinstance(self.metadata, Mapping):
            raise RawStemContractError("metadata must be a mapping")
        if self.position_backmap:
            if self.position_source_size != 256 or self.position_target_size != 224:
                raise RawStemContractError(
                    "position backmap requires declared 256 source and 224 target"
                )
            if self.input_mode == "bayer4_packed":
                raise RawStemContractError(
                    "packed Bayer geometry has no implicit 224-from-256 backmap"
                )
        for index, frame in enumerate(frames):
            if not isinstance(frame, RawStemInput):
                raise RawStemContractError(f"frame {index} must be a RawStemInput")
            if frame.batch_size != 1:
                raise RawStemContractError("temporal packets require batch size 1")
            if frame.input_mode != first.input_mode:
                raise RawStemContractError("all packet frames must share input_mode")
            if frame.camera_ids != first.camera_ids:
                raise RawStemContractError("all packet frames must share camera order")
            if frame.values.shape != first.values.shape:
                raise RawStemContractError("all packet frames must share values shape")
            if frame.units != first.units:
                raise RawStemContractError("all packet frames must share units")
            if frame.cfa_pattern != first.cfa_pattern:
                raise RawStemContractError("all packet frames must share CFA pattern")
        # The current endpoint's availability is the conservative output mask.
        # A frame may carry camera_present; absence is validated again by the
        # base measurement preparation, so no camera is silently invented.

    @property
    def camera_order(self) -> tuple[str, ...]:
        return self.frames[0].camera_ids

    @property
    def input_mode(self) -> str:
        return self.frames[0].input_mode


@dataclass(slots=True)
class TemporalState:
    """Explicit per-camera recurrent patch state.

    ``hidden`` has shape ``[B, Ncam, num_patches, hidden]``.  The wrapper
    currently accepts ``B=1`` only; the shape is retained so a later batch
    implementation cannot silently mix sources.
    """

    hidden: torch.Tensor
    source_demo_id: str
    camera_order: tuple[str, ...]
    last_endpoint: int
    model_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.hidden, torch.Tensor):
            raise RawStemContractError("TemporalState.hidden must be a tensor")
        if not self.hidden.is_floating_point():
            raise RawStemContractError("TemporalState.hidden must be floating point")
        if self.hidden.ndim != 4 or self.hidden.shape[0] != 1:
            raise RawStemContractError("TemporalState.hidden must be [1,Ncam,num_patches,hidden]")
        if not isinstance(self.source_demo_id, str) or not self.source_demo_id:
            raise RawStemContractError("TemporalState.source_demo_id must be nonempty")
        camera_order = tuple(self.camera_order)
        if not camera_order or len(set(camera_order)) != len(camera_order):
            raise RawStemContractError("TemporalState.camera_order must be unique and nonempty")
        object.__setattr__(self, "camera_order", camera_order)
        if self.hidden.shape[1] != len(camera_order):
            raise RawStemContractError("TemporalState camera axis differs from camera_order")
        endpoint = int(self.last_endpoint)
        if endpoint < 0:
            raise RawStemContractError("TemporalState.last_endpoint must be nonnegative")
        object.__setattr__(self, "last_endpoint", endpoint)
        if not isinstance(self.model_identity, str) or not self.model_identity:
            raise RawStemContractError(
                "TemporalState.model_identity must be a nonempty in-process generation string"
            )

    def detach(self) -> TemporalState:
        return TemporalState(
            hidden=self.hidden.detach(),
            source_demo_id=self.source_demo_id,
            camera_order=self.camera_order,
            last_endpoint=self.last_endpoint,
            model_identity=self.model_identity,
        )

    def to(self, device: str | torch.device) -> TemporalState:
        return TemporalState(
            hidden=self.hidden.to(device),
            source_demo_id=self.source_demo_id,
            camera_order=self.camera_order,
            last_endpoint=self.last_endpoint,
            model_identity=self.model_identity,
        )

    def snapshot(self) -> dict[str, Any]:
        hidden = self.hidden.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        return {
            "schema": TEMPORAL_STATE_SCHEMA,
            "format_version": 1,
            "hidden": hidden,
            "source_demo_id": self.source_demo_id,
            "camera_order": list(self.camera_order),
            "last_endpoint": self.last_endpoint,
            "model_identity": self.model_identity,
        }

    @classmethod
    def from_snapshot(
        cls, payload: Mapping[str, Any], *, device: str | torch.device = "cpu"
    ) -> TemporalState:
        required = {
            "schema",
            "format_version",
            "hidden",
            "source_demo_id",
            "camera_order",
            "last_endpoint",
            "model_identity",
        }
        if not isinstance(payload, Mapping) or not required <= set(payload):
            raise RawStemContractError("temporal state snapshot has unexpected/missing keys")
        if payload["schema"] != TEMPORAL_STATE_SCHEMA or int(payload["format_version"]) != 1:
            raise RawStemContractError("temporal state snapshot schema/version differs")
        hidden = payload["hidden"]
        if not isinstance(hidden, torch.Tensor):
            raise RawStemContractError("temporal state snapshot hidden is not a tensor")
        if hidden.dtype != torch.float32 or hidden.device.type != "cpu":
            raise RawStemContractError("temporal snapshot hidden must be CPU float32")
        state = cls(
            hidden=hidden.contiguous().clone(),
            source_demo_id=str(payload["source_demo_id"]),
            camera_order=tuple(str(value) for value in payload["camera_order"]),
            last_endpoint=int(payload["last_endpoint"]),
            model_identity=str(payload["model_identity"]),
        )
        return state.to(device)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.snapshot(), target)
        return target

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> TemporalState:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise RawStemContractError("temporal state file is not a mapping")
        return cls.from_snapshot(payload, device=device)


class TemporalRawStemStudent(nn.Module):
    """Causal six-frame patch recurrence feeding the inherited per-frame tower.

    Stage-1 trainable scope is the existing stem plus one channelwise gate.  The
    inherited vision tower, position embedding, post-layernorm and projector are
    frozen and stay in eval mode; gradients still flow through them to the stem
    and gate because no ``no_grad`` is used.
    """

    def __init__(self, base_student: Any, *, gate_init: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base_student, (RawStemSiglipStudent, RawStemSiglipDeployment)):
            raise RawStemContractError(
                "TemporalRawStemStudent requires a RawStemSiglipStudent or deployment"
            )
        self.base = base_student
        hidden = int(self.base.geometry.hidden_size)
        if hidden <= 0:
            raise RawStemContractError("base geometry has no hidden size")
        stem_device = next(self.base.stem.parameters()).device
        # Gate stays FP32 but lives on the base stem device; a CPU gate must not
        # be created under a CUDA base.
        self.gate_logit = nn.Parameter(
            torch.full((hidden,), float(gate_init), dtype=torch.float32, device=stem_device)
        )
        self.history_length = TEMPORAL_HISTORY_LENGTH
        self.endpoint_stride = TEMPORAL_ENDPOINT_STRIDE
        self.history_supported = True
        self.backend_trainable = False
        self.bridge_projection = None
        self.privileged_rgb_control = self.base.input_mode == "rgb3_control"
        self.input_domain = (
            "teacher_rgb_control_temporal_diagnostic"
            if self.privileged_rgb_control
            else "raw_temporal_causal"
        )
        self._scope = "stem_gate"
        self._generation_counter: int = 0
        self._generation_signature: tuple[Any, ...] | None = None
        self._generation_token: str = f"model-{id(self):x}/generation-0"
        if hasattr(self.base, "configure_trainable_scope"):
            self.configure_trainable_scope("stem_only")
        else:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self._scope = "all_frozen"

    @property
    def input_mode(self) -> str:
        return self.base.input_mode

    @property
    def geometry(self) -> VisionGeometry:
        return self.base.geometry

    @property
    def vision_tower(self) -> nn.Module:
        return self.base.vision_tower

    @property
    def projector(self) -> nn.Module:
        return self.base.projector

    @property
    def stem(self) -> nn.Module:
        return self.base.stem

    @property
    def position_embedding(self) -> nn.Module:
        return self.base.vision_tower.vision_model.embeddings.position_embedding

    @property
    def post_layernorm(self) -> nn.Module:
        return self.base.vision_tower.vision_model.post_layernorm

    @property
    def evidence_level(self) -> str:
        return str(
            getattr(
                self.base,
                "evidence_level",
                getattr(self.base, "metadata", {}).get("evidence_level", "deployment"),
            )
        )

    @property
    def checkpoint_identity(self) -> Any:
        return getattr(self.base, "checkpoint_identity", None)

    @property
    def initialization_audit(self) -> Any:
        return getattr(self.base, "initialization_audit", None)

    @property
    def load_evidence(self) -> Any:
        return getattr(self.base, "load_evidence", None)

    @property
    def gate_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def _checkpoint_payload(self) -> Mapping[str, Any]:
        identity = getattr(self.base, "checkpoint_identity", None)
        if identity is not None and hasattr(identity, "as_dict"):
            return dict(identity.as_dict())
        metadata = getattr(self.base, "metadata", {})
        payload = metadata.get("checkpoint") if isinstance(metadata, Mapping) else None
        if isinstance(payload, Mapping):
            return dict(payload)
        return {"evidence_level": self.evidence_level, "sha256": None, "path": None}

    def temporal_config(self) -> dict[str, Any]:
        return {
            "schema": _TEMPORAL_CONFIG_SCHEMA,
            "raw_burst_frames": TEMPORAL_HISTORY_LENGTH,
            "rgb_control_frames_per_endpoint": 1,
            "endpoint_stride": TEMPORAL_ENDPOINT_STRIDE,
            "aggregation": "mask_aware_mean_patch",
            "recurrence": "h=(1-alpha)*mean+alpha*h_prev",
            "gate": "channelwise_sigmoid",
            "hidden_size": int(self.geometry.hidden_size),
            "input_mode": self.input_mode,
            "privileged_rgb_control": self.privileged_rgb_control,
        }

    def _identity_tensor_signature(self) -> tuple[Any, ...]:
        # Recurrent state depends only on the stem and gate.  Scanning all 428M
        # inherited/frontend tensors at every endpoint was pure repeated work.
        named = tuple(self.stem.named_parameters(remove_duplicate=False)) + (
            ("gate_logit", self.gate_logit),
        )
        tensors = tuple(
            (
                name,
                id(tensor),
                int(tensor.data_ptr()),
                str(tensor.dtype),
                tuple(int(value) for value in tensor.shape),
                int(tensor._version),
            )
            for name, tensor in sorted(named, key=lambda item: item[0])
        )
        return (
            tensors,
            _canonical_json(self.temporal_config()),
            _canonical_json(self.geometry.as_dict()),
            self.input_mode,
        )

    @property
    def model_identity(self) -> str:
        """Lightweight in-process parameter/version generation (no hashing).

        The token changes whenever a named parameter/buffer is replaced or its
        PyTorch version advances, or when temporal/geometry/input configuration
        changes.  It is intentionally not a content hash and does not read tensor
        values.  Temporal states use it only for freshness; episode/camera/reset
        and shape checks remain explicit.
        """
        signature = self._identity_tensor_signature()
        if signature != self._generation_signature:
            self._generation_counter += 1
            self._generation_signature = signature
            self._generation_token = f"model-{id(self):x}/generation-{self._generation_counter}"
        return self._generation_token

    def configure_trainable_scope(
        self,
        scope: str,
        *,
        input_blocks: int = 0,
        explicit: bool = False,
    ) -> dict[str, Any]:
        if not hasattr(self.base, "configure_trainable_scope"):
            if scope != "all_frozen":
                raise RawStemContractError("deployment base is inference-only")
            self.gate_logit.requires_grad_(False)
            self._scope = "all_frozen"
            return self.trainable_report()
        self.base.configure_trainable_scope(scope, input_blocks=input_blocks, explicit=explicit)
        self.gate_logit.requires_grad_(scope != "all_frozen")
        self._scope = scope
        return self.trainable_report()

    def train(self, mode: bool = True) -> TemporalRawStemStudent:
        super().train(bool(mode))
        self.base.train(bool(mode))
        return self

    def trainable_report(self) -> dict[str, Any]:
        named = list(self.named_parameters())
        trainable = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
            }
            for name, parameter in named
            if parameter.requires_grad
        ]
        total = sum(parameter.numel() for _, parameter in named)
        trainable_total = sum(item["numel"] for item in trainable)
        return {
            "scope": self._scope,
            "trainable_parameters": trainable_total,
            "frozen_parameters": total - trainable_total,
            "total_parameters": total,
            "trainable_names": [item["name"] for item in trainable],
            "trainable": trainable,
            "backend_trainable": False,
            "bridge_projection": None,
            "gate_parameters": int(self.gate_logit.numel()),
        }

    def trainable_parameters(self) -> dict[str, Any]:
        return self.trainable_report()

    def gradient_report(self) -> dict[str, Any]:
        entries = []
        missing = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            entries.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "grad_present": grad is not None,
                    "grad_norm": None if grad is None else float(grad.detach().float().norm()),
                }
            )
            if grad is None:
                missing.append(name)
        return {
            "scope": self._scope,
            "entries": entries,
            "missing_gradients": missing,
            "all_trainable_have_gradients": not missing,
        }

    def dependency_audit(self) -> dict[str, Any]:
        return {
            "teacher": False,
            "isp": False,
            "cache": False,
            "rgb_generation": False,
            "privileged_rgb_control": self.privileged_rgb_control,
        }

    def _validate_packet(self, packet: TemporalPacket) -> None:
        if not isinstance(packet, TemporalPacket):
            raise RawStemContractError("forward requires a TemporalPacket")
        if packet.input_mode != self.input_mode:
            raise RawStemContractError(
                f"temporal student is {self.input_mode!r}, packet declares {packet.input_mode!r}"
            )

    def _validate_state(self, packet: TemporalPacket, state: TemporalState | None) -> None:
        camera_order = packet.camera_order
        expected_shape = (
            1,
            len(camera_order),
            int(self.geometry.num_patches),
            int(self.geometry.hidden_size),
        )
        if state is None:
            if not packet.reset:
                raise RawStemContractError("the first temporal packet requires reset=True")
            return
        if not isinstance(state, TemporalState):
            raise RawStemContractError("state must be a TemporalState")
        if state.model_identity != self.model_identity:
            raise RawStemContractError("temporal state model identity differs")
        if tuple(state.hidden.shape) != expected_shape:
            raise RawStemContractError(
                f"temporal state shape {tuple(state.hidden.shape)} != {expected_shape}"
            )
        if state.hidden.dtype != torch.float32:
            raise RawStemContractError("temporal state hidden must be float32")
        if packet.reset:
            return
        if packet.endpoint != state.last_endpoint + self.endpoint_stride:
            raise RawStemContractError(
                f"temporal carry requires endpoint {state.last_endpoint + self.endpoint_stride}, "
                f"got {packet.endpoint}"
            )
        if packet.source_demo_id != state.source_demo_id:
            raise RawStemContractError("temporal source changed without reset")
        if camera_order != state.camera_order:
            raise RawStemContractError("temporal camera order changed without reset")

    def _patch_burst(
        self, packet: TemporalPacket
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        patch_stride = self.geometry.packed_patch_stride(self.input_mode)
        tokens = []
        masks = []
        current_prepared: Mapping[str, Any] | None = None
        for index, frame in enumerate(packet.frames):
            prepared = _prepare_measurement(frame, self.geometry)
            patch_tokens, prepared = _stem_patch_tokens(
                self.base.vision_tower, self.geometry, frame, prepared=prepared
            )
            patch_tokens = patch_tokens.reshape(
                1,
                len(frame.camera_ids),
                int(self.geometry.num_patches),
                int(self.geometry.hidden_size),
            )
            usable = prepared["usable"].flatten(0, 1).to(torch.float32).unsqueeze(1)
            support = _patch_pool(usable, patch_stride, "avg")[:, 0]
            support = support.reshape(1, len(frame.camera_ids), self.geometry.num_patches)
            token_mask = (support >= 1.0 - 1e-6) & prepared["present"][:, :, None]
            if packet.padding_mask[index]:
                token_mask = torch.zeros_like(token_mask)
            tokens.append(patch_tokens)
            masks.append(token_mask)
            current_prepared = prepared
        if current_prepared is None:
            raise RawStemContractError("temporal packet has no current frame")
        stacked = torch.stack(tokens, dim=1)
        mask = torch.stack(masks, dim=1)
        count = mask.sum(dim=1)
        weighted = (stacked * mask.unsqueeze(-1)).sum(dim=1)
        mean = weighted / count.clamp(min=1).unsqueeze(-1)
        has_support = count > 0
        mean = torch.where(has_support.unsqueeze(-1), mean, torch.zeros_like(mean))
        return mean, has_support, current_prepared

    def _advance(
        self,
        packet: TemporalPacket,
        mean: torch.Tensor,
        has_support: torch.Tensor,
        current_prepared: Mapping[str, Any],
        state: TemporalState | None,
    ) -> TemporalState:
        present = current_prepared["present"][:, :, None, None]
        previous = None if packet.reset else state
        if previous is None:
            hidden = torch.where(has_support.unsqueeze(-1), mean, torch.zeros_like(mean))
        else:
            alpha = self.gate_alpha.reshape(1, 1, 1, -1)
            advanced = (1.0 - alpha) * mean + alpha * previous.hidden
            hidden = torch.where(has_support.unsqueeze(-1), advanced, previous.hidden)
            hidden = torch.where(present, hidden, previous.hidden)
        return TemporalState(
            hidden=hidden,
            source_demo_id=packet.source_demo_id,
            camera_order=packet.camera_order,
            last_endpoint=packet.endpoint,
            model_identity=self.model_identity,
        )

    def _run_burst(
        self, packet: TemporalPacket, state: TemporalState | None
    ) -> tuple[TemporalState, torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        self._validate_packet(packet)
        self._validate_state(packet, state)
        if state is not None and state.hidden.device != packet.frames[0].values.device:
            raise RawStemContractError("temporal state device differs from packet device")
        mean, has_support, current_prepared = self._patch_burst(packet)
        new_state = self._advance(packet, mean, has_support, current_prepared, state)
        return new_state, mean, has_support, current_prepared

    def update_state(self, packet: TemporalPacket, state: TemporalState | None) -> TemporalState:
        """Advance the recurrent patch state without running inherited SigLIP."""
        new_state, _, _, _ = self._run_burst(packet, state)
        return new_state

    def burn_in_state(self, packet: TemporalPacket, state: TemporalState | None) -> TemporalState:
        """Alias for full-prefix burn-in; caller may wrap this in ``no_grad``."""
        return self.update_state(packet, state)

    def detach_state(self, state: TemporalState) -> TemporalState:
        if not isinstance(state, TemporalState) or state.model_identity != self.model_identity:
            raise RawStemContractError("cannot detach a foreign temporal state")
        return state.detach()

    def forward_batched(self, samples: Any) -> Any:
        from owac.training.raw_full_visual_batched import batched_anchor_forward

        return batched_anchor_forward(self, samples)

    def forward(self, packet: TemporalPacket, state: TemporalState | None = None) -> Any:
        from owac.training.raw_full_visual_batched import BatchedTemporalRequest

        if isinstance(packet, BatchedTemporalRequest):
            return self.forward_batched(packet.samples)
        new_state, _, has_support, _ = self._run_burst(packet, state)
        flat_hidden = new_state.hidden.reshape(1, -1, int(self.geometry.hidden_size))
        flat_hidden = torch.where(
            has_support.reshape(1, -1, 1), flat_hidden, torch.zeros_like(flat_hidden)
        )
        output = _forward_frontend(
            self.base.vision_tower,
            self.base.projector,
            self.geometry,
            packet.frames[-1],
            patch_tokens=flat_hidden,
        )
        # The 224 model grid is not the original sensor grid.  Both RAW and the
        # single-RGB diagnostic B packet are 224 derivations of the 256 full
        # view, so expose physical 256 coordinates and the 16-pixel patch scale
        # in the actual output, not only in a test helper.
        if int(self.geometry.image_size) == 224 and packet.position_backmap:
            positions = output.tokens.positions.detach().clone()
            positions[..., :3] *= 256.0 / 224.0
            output.tokens.positions = positions
            output.tokens.scale = 16.0
            output.measurement_metadata["physical_position_contract"] = {
                "model_image_size": 224,
                "model_patch_size": int(self.geometry.patch_size),
                "physical_patch_size": 16,
                "backmap_scale": 256.0 / 224.0,
                "coordinate_frame": "original_256_sensor_patch_center_yx",
                "source_image_size": packet.position_source_size,
                "target_image_size": packet.position_target_size,
                "preprocess_id": packet.preprocess_id,
            }
        output.measurement_metadata["temporal"] = {
            "schema": TEMPORAL_MANIFEST_SCHEMA,
            "endpoint": packet.endpoint,
            "frame_ids": list(packet.frame_ids),
            "padding_mask": list(packet.padding_mask),
            "history_length": len(packet.frames),
            "endpoint_stride": TEMPORAL_ENDPOINT_STRIDE,
            "model_identity": self.model_identity,
            "privileged_rgb_control": self.privileged_rgb_control,
            "gate_alpha_mean": float(self.gate_alpha.detach().float().mean()),
        }
        return output, new_state

    def snapshot_state(self, state: TemporalState) -> dict[str, Any]:
        if not isinstance(state, TemporalState) or state.model_identity != self.model_identity:
            raise RawStemContractError("cannot snapshot a foreign temporal state")
        return state.snapshot()

    def state_from_snapshot(
        self, payload: Mapping[str, Any], *, device: str | torch.device = "cpu"
    ) -> TemporalState:
        state = TemporalState.from_snapshot(payload, device=device)
        if state.model_identity != self.model_identity:
            raise RawStemContractError("snapshot model identity differs from current student")
        return state

    def export_artifact(
        self,
        path: str | Path,
        *,
        backend_identity: Mapping[str, Any],
        tokenizer_identity: Mapping[str, Any] | None = None,
        normalizer_identity: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        checkpoint_step: int = 0,
    ) -> dict[str, Any]:
        """Export the existing per-frame artifact plus a temporal manifest."""
        target = Path(path)
        base_metadata = export_student_artifact(
            self.base,
            target,
            backend_identity=backend_identity,
            tokenizer_identity=tokenizer_identity,
            normalizer_identity=normalizer_identity,
            extra_metadata=extra_metadata,
            checkpoint_step=checkpoint_step,
        )
        config = self.temporal_config()
        payload = {
            "schema": TEMPORAL_MANIFEST_SCHEMA,
            "format_version": 1,
            "gate_logit": self.gate_logit.detach().to(device="cpu", dtype=torch.float32),
            "temporal_config": config,
            "model_identity": self.model_identity,
            "checkpoint_step": int(checkpoint_step),
        }
        torch.save(payload, target / "temporal.pt")
        manifest = {
            "schema": TEMPORAL_MANIFEST_SCHEMA,
            "format_version": 1,
            "base_artifact": base_metadata,
            "gate": {
                "name": "gate_logit",
                "shape": [int(self.gate_logit.numel())],
                "dtype": "float32",
            },
            "temporal_config": config,
            "model_identity": self.model_identity,
            "checkpoint_step": int(checkpoint_step),
            "temporal_state_keys": ["gate_logit"],
        }
        (target / "temporal.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest


def _read_temporal_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = path / "temporal.json"
    payload_path = path / "temporal.pt"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise RawStemContractError("temporal manifest/state is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != TEMPORAL_MANIFEST_SCHEMA:
        raise RawStemContractError("temporal manifest schema differs")
    if int(manifest.get("format_version", -1)) != 1:
        raise RawStemContractError("temporal manifest format version differs")
    frontend_path = path / "frontend.json"
    if not frontend_path.is_file():
        raise RawStemContractError("temporal base frontend.json is absent")
    base_artifact = json.loads(frontend_path.read_text(encoding="utf-8"))
    if not isinstance(base_artifact, dict) or manifest.get("base_artifact") != base_artifact:
        raise RawStemContractError("temporal manifest base artifact differs")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != TEMPORAL_MANIFEST_SCHEMA:
        raise RawStemContractError("temporal payload schema differs")
    if int(payload.get("format_version", -1)) != 1:
        raise RawStemContractError("temporal payload format version differs")
    required = {
        "schema",
        "format_version",
        "gate_logit",
        "temporal_config",
        "model_identity",
        "checkpoint_step",
    }
    if not required <= set(payload):
        raise RawStemContractError("temporal payload has unexpected/missing keys")
    gate = payload["gate_logit"]
    if not isinstance(gate, torch.Tensor) or gate.dtype != torch.float32 or gate.ndim != 1:
        raise RawStemContractError("temporal gate payload must be a rank-1 float32 tensor")
    return manifest, payload


def export_temporal_student_artifact(
    student: TemporalRawStemStudent,
    path: str | Path,
    *,
    backend_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any] | None = None,
    normalizer_identity: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    checkpoint_step: int = 0,
) -> dict[str, Any]:
    """Module-level convenience wrapper for ``TemporalRawStemStudent.export_artifact``."""
    if not isinstance(student, TemporalRawStemStudent):
        raise RawStemContractError("export requires a TemporalRawStemStudent")
    return student.export_artifact(
        path,
        backend_identity=backend_identity,
        tokenizer_identity=tokenizer_identity,
        normalizer_identity=normalizer_identity,
        extra_metadata=extra_metadata,
        checkpoint_step=checkpoint_step,
    )


def load_temporal_student_artifact(
    path: str | Path,
    *,
    frontend_factory: Any = None,
    expected_checkpoint_sha256: str | None = None,
    expected_state_dict_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> TemporalRawStemStudent:
    """Reload a per-frame artifact plus its temporal gate without teacher imports."""
    target = Path(path)
    manifest, payload = _read_temporal_payload(target)
    base = load_student_artifact(
        target,
        frontend_factory=frontend_factory,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_state_dict_sha256=expected_state_dict_sha256,
        map_location=map_location,
        allow_temporal_teacher=True,
    )
    config = payload["temporal_config"]
    if not isinstance(config, Mapping) or config.get("schema") != _TEMPORAL_CONFIG_SCHEMA:
        raise RawStemContractError("temporal config schema differs")
    gate = payload["gate_logit"]
    if int(config.get("hidden_size", -1)) != int(base.geometry.hidden_size):
        raise RawStemContractError("temporal gate width differs from base geometry")
    if tuple(gate.shape) != (int(base.geometry.hidden_size),):
        raise RawStemContractError("temporal gate shape differs from base geometry")
    student = TemporalRawStemStudent(base)
    if manifest.get("gate", {}).get("shape") != [int(base.geometry.hidden_size)]:
        raise RawStemContractError("temporal manifest gate shape differs")
    with torch.no_grad():
        student.gate_logit.copy_(gate.to(device=student.gate_logit.device))
    return student

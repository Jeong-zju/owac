"""RAW-stem distillation integration: losses, native action step, teacher prep, resume.

Diagnostics perform zero optimizer updates. The separately guarded training
engine requires explicit authority and validated assets before it can step.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from owac.baselines.raw_stem_siglip import (
    HistoryContract,
    RawStemInput,
    assert_backend_frozen,
    sha256_file,
    sha256_tensor,
)

__all__ = [
    "DistillationBatch",
    "FrozenStabilityReport",
    "NativeActionResult",
    "RawDistillationTrainer",
    "RawTeacherCheckpoint",
    "RawVLATeacherAdapter",
    "ResumeState",
    "TeacherPreparationReceipt",
    "TeacherPreparationUnavailable",
    "TrainingAuthority",
    "TrainingUnavailable",
    "available_raw_teacher_adapters",
    "build_token_loss",
    "native_action_loss",
    "register_raw_teacher_adapter",
    "token_distillation_loss",
    "write_prepared_teacher_bundle",
]

_CONFIG_KEYS = {
    "seed",
    "effective_batch",
    "source_demonstrations",
    "split",
    "max_cumulative_updates_per_arm",
    "max_anchor_presentations_per_arm",
    "checkpoint_updates",
    "optimizer",
    "weight_decay",
    "gradient_clip",
    "warmup_updates",
    "scheduler",
    "stage1",
    "stage2",
    "backend_trainable",
    "loss",
    "phase_gate",
}


class TrainingUnavailable(RuntimeError):
    """A required real training asset or gate is absent."""


class TeacherPreparationUnavailable(RuntimeError):
    """The learned RAW teacher checkpoint/code is absent or not accepted."""


@dataclass(frozen=True, slots=True)
class TrainingAuthority:
    path: str
    sha256: str
    authorized: bool
    max_updates: int
    max_anchors: int
    scope: str = "raw-stem-distillation"

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError("authority receipt must carry a SHA256")
        if self.max_updates < 0 or self.max_anchors < 0:
            raise ValueError("authority budgets must be nonnegative")

    @classmethod
    def from_file(cls, path: str | Path) -> TrainingAuthority:
        source = Path(path)
        if not source.is_file():
            raise TrainingUnavailable(f"training authority receipt not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("authorized") is not True:
            raise TrainingUnavailable("training authority receipt is not authorized")
        return cls(
            path=str(source),
            sha256=sha256_file(source),
            authorized=True,
            max_updates=int(payload.get("max_updates", 0)),
            max_anchors=int(payload.get("max_anchors", 0)),
            scope=str(payload.get("scope", "raw-stem-distillation")),
        )


@dataclass(frozen=True, slots=True)
class RawTeacherCheckpoint:
    """Identity of a *learned RAW-input* author RawVLA frontend."""

    frontend_checkpoint_sha256: str
    frontend_config_sha256: str
    code_revision: str
    adapter_name: str
    source: str
    verified: bool
    state_path: str | None = None
    state_dict_prefix: str = ""
    default_isp: bool = False
    official_pi05_only: bool = False

    def __post_init__(self) -> None:
        for name in ("frontend_checkpoint_sha256", "frontend_config_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be 64 lowercase hex characters")

    @property
    def learned_raw_ready(self) -> bool:
        source = self.source.lower()
        return (
            self.verified
            and not self.default_isp
            and not self.official_pi05_only
            and self.adapter_name in _RAW_ADAPTERS
            and "rawvla" in source
            and "default" not in source
        )


_RAW_ADAPTERS: dict[str, Callable[..., Any]] = {}


def register_raw_teacher_adapter(name: str, factory: Callable[..., Any]) -> None:
    if not name or not callable(factory):
        raise ValueError("adapter name and callable factory required")
    _RAW_ADAPTERS[name] = factory


def available_raw_teacher_adapters() -> tuple[str, ...]:
    return tuple(sorted(_RAW_ADAPTERS))


@dataclass(frozen=True, slots=True)
class TeacherPreparationReceipt:
    adapter_name: str
    checkpoint_sha256: str
    config_sha256: str
    code_revision: str
    evidence_level: str
    real_ready: bool
    raw_sha256: str
    teacher_tokens_sha256: str
    prepared_rgb_sha256: str
    state_before_sha256: str
    state_after_sha256: str
    causal_prefix_sha256: str
    history: HistoryContract
    geometry: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    limitation: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["history"] = self.history.as_dict()
        return data

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        return target


@dataclass(slots=True)
class PreparedTeacherBatch:
    raw: torch.Tensor
    prepared_rgb: torch.Tensor
    tokens: torch.Tensor
    receipt: TeacherPreparationReceipt


class RawVLATeacherAdapter:
    """Registered learned-RAW adapter: frontend -> original preprocessing -> VLA.

    The adapter is generic so a real author adapter can be registered later.  It
    never substitutes the deterministic Default ISP: ``default_isp`` and
    ``official_pi05_only`` identities are rejected as real readiness.  Fixture
    adapters are explicitly labelled ``fixture_contract``.
    """

    def __init__(
        self,
        *,
        policy: Any,
        frontend: nn.Module,
        checkpoint: RawTeacherCheckpoint,
        history: HistoryContract,
        evidence_level: str = "real_policy",
        checkpoint_state: Mapping[str, torch.Tensor] | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        if evidence_level not in {"real_policy", "fixture_contract"}:
            raise ValueError("teacher evidence must be real_policy or fixture_contract")
        if evidence_level == "real_policy" and not checkpoint.learned_raw_ready:
            raise TeacherPreparationUnavailable(
                "learned RAW teacher is not ready: missing/verified author frontend "
                "checkpoint or a Default-ISP/pi05-only identity was offered"
            )
        self.policy = policy
        self.frontend = frontend
        self.checkpoint = checkpoint
        self.history = history
        self.evidence_level = evidence_level
        self.state_path = None if state_path is None else Path(state_path)
        if state_path is not None:
            path = Path(state_path)
            if not path.is_file():
                raise TeacherPreparationUnavailable(f"teacher state not found: {path}")
            if checkpoint.verified and sha256_file(path) != checkpoint.frontend_checkpoint_sha256:
                raise TeacherPreparationUnavailable("teacher state SHA256 mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise TeacherPreparationUnavailable("teacher state file is not a dict")
            if "state_dict" in state:
                state = state["state_dict"]
            self._load_frontend_state(state)
        elif checkpoint_state is not None:
            digest = _mapping_tensor_digest(checkpoint_state)
            if checkpoint.verified and digest != checkpoint.frontend_checkpoint_sha256:
                raise TeacherPreparationUnavailable("teacher state tensor digest mismatch")
            self._load_frontend_state(checkpoint_state)
        elif evidence_level == "real_policy":
            raise TeacherPreparationUnavailable("real learned teacher requires a state file")
        self.frontend.eval()
        self.policy.eval()
        for parameter in list(self.frontend.parameters()) + list(self.policy.parameters()):
            parameter.requires_grad_(False)
        self._state = None
        self._theta = None
        self._prefix_digest = ""

    def _load_frontend_state(self, state: Mapping[str, torch.Tensor]) -> None:
        prefix = self.checkpoint.state_dict_prefix
        selected = {
            (key[len(prefix) :] if prefix and key.startswith(prefix) else key): value
            for key, value in state.items()
        }
        try:
            self.frontend.load_state_dict(selected, strict=True)
        except (RuntimeError, ValueError) as exc:
            raise TeacherPreparationUnavailable("teacher state does not match adapter") from exc

    def reset(self) -> None:
        reset = getattr(self.frontend, "reset_state", None)
        if callable(reset):
            reset()
        self._state = None
        self._theta = None
        self._prefix_digest = ""

    def prepare(
        self,
        raw_burst: torch.Tensor,
        *,
        history: HistoryContract | None = None,
    ) -> PreparedTeacherBatch:
        history = history or self.history
        if not self.history.compatible_with(history):
            raise ValueError("teacher history structure differs from the registered contract")
        if not isinstance(raw_burst, torch.Tensor) or raw_burst.ndim != 5:
            raise ValueError("raw_burst must be [B,T,C,H,W]")
        if int(raw_burst.shape[1]) != history.temporal_frames:
            raise ValueError("raw burst length differs from the HistoryContract")
        if history.reset:
            self.reset()
        state_before = _state_hash(self._state, self._theta)
        state_before_payload = self._state, self._theta
        with torch.no_grad():
            kwargs = {}
            try:
                import inspect

                parameters = inspect.signature(self.frontend.forward).parameters
                if "state" in parameters:
                    kwargs["state"] = self._state
                if "theta_prev" in parameters:
                    kwargs["theta_prev"] = self._theta
            except (TypeError, ValueError):
                pass
            output = self.frontend(raw_burst, **kwargs)
            author_output = not torch.is_tensor(output)
            if torch.is_tensor(output):
                if output.ndim == 5:
                    images = output
                elif output.ndim == 4:
                    images = output.unsqueeze(1)
                else:
                    raise ValueError("RAW frontend tensor output must be [B,3,H,W] or [B,T,3,H,W]")
            else:
                rgb = getattr(output, "rgb", None)
                if not isinstance(rgb, torch.Tensor) or rgb.ndim != 4:
                    raise ValueError("RAWVLAOutput.rgb must be [B,3,H,W]")
                images = rgb.unsqueeze(1)
                returned_state = getattr(output, "state", None)
                returned_theta = getattr(output, "theta", None)
                if returned_theta is None:
                    returned_theta = getattr(output, "theta_prev", None)
                self._state = _clone_state(returned_state)
                self._theta = _clone_state(returned_theta)
            if images.ndim != 5 or int(images.shape[2]) != 3 or images.dtype != torch.float32:
                raise ValueError("RAW frontend RGB must be float32 [B,T,3,H,W]")
            if images.min() < 0 or images.max() > 1:
                raise ValueError("RAW frontend RGB must be [0,1] before original preprocessing")
            preprocess = getattr(self.policy, "_image_to_tensor", None)
            if not callable(preprocess):
                raise TeacherPreparationUnavailable(
                    "original policy._image_to_tensor is unavailable"
                )
            prepared = []
            geometry = []
            for index in range(int(images.shape[0])):
                row = []
                for frame in range(int(images.shape[1])):
                    source = images[index, frame]
                    row.append(preprocess(source))
                    source_h, source_w = (int(v) for v in source.shape[-2:])
                    model_h, model_w = (int(v) for v in row[-1].shape[-2:])
                    scale = min(model_h / source_h, model_w / source_w)
                    resized_h = max(1, round(source_h * scale))
                    resized_w = max(1, round(source_w * scale))
                    geometry.append(
                        {
                            "source_shape": [source_h, source_w],
                            "model_shape": [model_h, model_w],
                            "scale": scale,
                            "resized_shape": [resized_h, resized_w],
                            "pad": [model_h - resized_h, model_w - resized_w],
                            "coordinate_note": "source physical ROI -> letterboxed model pixels",
                        }
                    )
                prepared.append(torch.stack(row, dim=0))
            prepared_rgb = torch.stack(prepared, dim=0)
            if author_output and prepared_rgb.shape[1] == 1:
                prepared_rgb = prepared_rgb[:, 0]
            embed = getattr(getattr(self.policy, "vlm", self.policy), "embed_image", None)
            if callable(embed):
                flat = prepared_rgb.reshape(-1, *prepared_rgb.shape[-3:])
                embedded = embed(flat)
                if embedded.ndim != 3:
                    raise TeacherPreparationUnavailable(
                        "embed_image must return [B,N,D] teacher tokens"
                    )
                if author_output:
                    tokens = embedded
                elif prepared_rgb.shape[1] == 1:
                    tokens = embedded.unsqueeze(1)
                else:
                    tokens = embedded.reshape(
                        prepared_rgb.shape[0],
                        prepared_rgb.shape[1],
                        *embedded.shape[-2:],
                    )
            elif self.evidence_level == "fixture_contract":
                tokens = prepared_rgb
            else:
                raise TeacherPreparationUnavailable("original embed_image is unavailable")
        after_state = _state_hash(self._state, self._theta)
        prefix_digest = hashlib.sha256()
        prefix_digest.update(self._prefix_digest.encode())
        prefix_digest.update(self.checkpoint.frontend_checkpoint_sha256.encode())
        prefix_digest.update(self.checkpoint.frontend_config_sha256.encode())
        prefix_digest.update(_causal_prefix_hash(raw_burst, state_before_payload).encode())
        prefix_digest.update(json.dumps(history.as_dict(), sort_keys=True).encode())
        self._prefix_digest = prefix_digest.hexdigest()
        receipt = TeacherPreparationReceipt(
            adapter_name=self.checkpoint.adapter_name,
            checkpoint_sha256=self.checkpoint.frontend_checkpoint_sha256,
            config_sha256=self.checkpoint.frontend_config_sha256,
            code_revision=self.checkpoint.code_revision,
            evidence_level=self.evidence_level,
            real_ready=bool(
                self.evidence_level == "real_policy" and self.checkpoint.learned_raw_ready
            ),
            raw_sha256=sha256_tensor(raw_burst),
            teacher_tokens_sha256=sha256_tensor(tokens),
            prepared_rgb_sha256=sha256_tensor(prepared_rgb),
            state_before_sha256=state_before,
            state_after_sha256=after_state,
            causal_prefix_sha256=self._prefix_digest,
            history=history,
            geometry={"frames": geometry, "model_image_size": list(prepared_rgb.shape[-2:])},
            preprocessing={
                "callable": "policy._image_to_tensor",
                "embedding": "policy.vlm.embed_image",
                "color_range": "[-1,1]",
            },
            limitation=(
                "fixture_contract only; not a trained RawVLA teacher"
                if self.evidence_level == "fixture_contract"
                else ""
            ),
        )
        return PreparedTeacherBatch(
            raw=raw_burst.detach().clone(),
            prepared_rgb=prepared_rgb.detach().clone(),
            tokens=tokens.detach().clone(),
            receipt=receipt,
        )


def _clone_state(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{field.name: _clone_state(getattr(value, field.name)) for field in fields(value)}
        )
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_state(item) for item in value)
    return value


def _state_hash(state: Any, theta: Any) -> str:
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if value is None:
            digest.update(b"none;")
        elif torch.is_tensor(value):
            digest.update(sha256_tensor(value).encode())
        elif is_dataclass(value) and not isinstance(value, type):
            digest.update(type(value).__qualname__.encode())
            for field in fields(value):
                digest.update(field.name.encode())
                visit(getattr(value, field.name))
        elif isinstance(value, Mapping):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                digest.update(str(index).encode())
                visit(item)
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())

    visit({"state": state, "theta": theta})
    return digest.hexdigest()


def _causal_prefix_hash(raw: torch.Tensor, state_before: Any) -> str:
    state, theta = state_before
    digest = hashlib.sha256()
    digest.update(sha256_tensor(raw).encode())
    digest.update(_state_hash(state, theta).encode())
    return digest.hexdigest()


def _mapping_tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_prepared_teacher_bundle(
    directory: str | Path, prepared: PreparedTeacherBatch
) -> dict[str, str]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "raw": prepared.raw.detach().cpu().numpy(),
        "prepared_rgb": prepared.prepared_rgb.detach().cpu().float().numpy(),
        "teacher_tokens": prepared.tokens.detach().cpu().float().numpy(),
    }
    hashes = {}
    for name, array in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, array)
        hashes[name] = sha256_file(path)
    receipt_path = root / "receipt.json"
    prepared.receipt.write(receipt_path)
    hashes["receipt"] = sha256_file(receipt_path)
    return hashes


@dataclass(frozen=True, slots=True)
class TokenLoss:
    mse: torch.Tensor
    cosine: torch.Tensor
    total: torch.Tensor
    valid_tokens: int
    width: int


def token_distillation_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    cosine_weight: float = 0.1,
    mse_weight: float = 1.0,
) -> TokenLoss:
    """Exact FP32 token MSE/cosine reduction; no magnitude renormalisation."""
    if student_tokens.shape != teacher_tokens.shape or student_tokens.ndim != 3:
        raise ValueError("student/teacher tokens must share [B,N,D]")
    if not student_tokens.is_floating_point():
        student_tokens = student_tokens.float()
    if not teacher_tokens.is_floating_point():
        teacher_tokens = teacher_tokens.float()
    student = student_tokens.float()
    teacher = teacher_tokens.float().to(student.device)
    bsize, length, width = student.shape
    if mask is None:
        mask = torch.ones((bsize, length), dtype=torch.bool, device=student.device)
    else:
        mask = torch.as_tensor(mask, device=student.device, dtype=torch.bool)
        if tuple(mask.shape) != (bsize, length):
            raise ValueError("token mask must be [B,N]")
    count = int(mask.sum())
    if count <= 0:
        raise ValueError("token loss needs at least one valid token")
    diff = student - teacher
    mse = (diff.square() * mask.unsqueeze(-1)).sum() / (count * width)
    cos = F.cosine_similarity(student, teacher, dim=-1)
    cosine = ((1.0 - cos) * mask).sum() / count
    total = mse_weight * mse + cosine_weight * cosine
    return TokenLoss(mse=mse, cosine=cosine, total=total, valid_tokens=count, width=width)


build_token_loss = token_distillation_loss


@dataclass(slots=True)
class NativeActionResult:
    action_loss: torch.Tensor
    normalized_actions: torch.Tensor
    output: Mapping[str, Any]


def native_action_loss(
    bridge: Any,
    tokens: Any,
    *,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
) -> NativeActionResult:
    """Run the pinned bridge native flow objective without an optimizer step."""
    out = bridge.forward_from_visual_tokens(
        tokens,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=noise,
        time=time,
        action_mask=action_mask,
    )
    if "action_loss" not in out:
        raise ValueError("native bridge did not return action_loss")
    return NativeActionResult(
        action_loss=out["action_loss"],
        normalized_actions=out.get("v_t", out["action_loss"].new_zeros(actions.shape)),
        output=out,
    )


@dataclass(frozen=True, slots=True)
class DistillationBatch:
    raw: RawStemInput
    teacher_tokens: torch.Tensor
    lang_tokens: torch.Tensor
    lang_masks: torch.Tensor
    state: torch.Tensor
    actions: torch.Tensor
    future_mask: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    time: torch.Tensor | None = None
    teacher_rgb: torch.Tensor | None = None
    arm: str = "C"


@dataclass(slots=True)
class FrozenStabilityReport:
    before: Mapping[str, str]
    after: Mapping[str, str]

    @property
    def stable(self) -> bool:
        return dict(self.before) == dict(self.after)


class RawDistillationTrainer:
    """Zero-update diagnostics and an explicitly authorized training engine."""

    def __init__(
        self,
        *,
        student: nn.Module,
        bridge: Any,
        config: Mapping[str, Any],
        history: HistoryContract,
        authority: TrainingAuthority | None = None,
        teacher_ready: bool = False,
        stage0_parity_ready: bool = False,
        source_split_ready: bool = False,
        resource_budget_ready: bool = False,
        phase1_diagnosis: Mapping[str, Any] | None = None,
        matched_cd_declaration: Mapping[str, Any] | None = None,
        teacher: Any = None,
    ) -> None:
        self.student = student
        self.bridge = bridge
        self.config = dict(config)
        self.history = history
        self.authority = authority
        self.teacher_ready = bool(teacher_ready)
        self.stage0_parity_ready = bool(stage0_parity_ready)
        self.source_split_ready = bool(source_split_ready)
        self.resource_budget_ready = bool(resource_budget_ready)
        self.phase1_diagnosis = dict(phase1_diagnosis or {})
        self.matched_cd_declaration = dict(matched_cd_declaration or {})
        self.cumulative_updates = int(config.get("cumulative_updates", 0))
        self.anchor_presentations = int(config.get("anchor_presentations", 0))
        self.last_frozen_report: FrozenStabilityReport | None = None
        self.teacher = teacher

    def _training(self) -> Mapping[str, Any]:
        training = self.config.get("training", self.config)
        return training if isinstance(training, Mapping) else self.config

    def _authorized_for_update(self) -> tuple[bool, str]:
        if self.authority is None or not self.authority.authorized:
            return False, "explicit actual training authority receipt is required"
        authorization = self.config.get("authorization", {})
        if authorization.get("optimizer_updates") is not True:
            return False, "config.authorization.optimizer_updates is false"
        for name, value in (
            ("teacher_ready", self.teacher_ready),
            ("stage0_parity_ready", self.stage0_parity_ready),
            ("source_split_ready", self.source_split_ready),
            ("resource_budget_ready", self.resource_budget_ready),
        ):
            if not value:
                return False, f"{name} is not satisfied"
        if not _backend_frozen(self.bridge):
            return False, "backend is not frozen"
        if self.history.is_temporal:
            return False, "temporal student history is not implemented"
        return True, "authorized"

    def preflight(self) -> dict[str, Any]:
        checks = {
            "teacher_ready": self.teacher_ready,
            "stage0_parity_ready": self.stage0_parity_ready,
            "source_split_ready": self.source_split_ready,
            "resource_budget_ready": self.resource_budget_ready,
            "update_authority": self.authority is not None,
            "backend_frozen": _backend_frozen(self.bridge),
            "history_supported": not self.history.is_temporal,
        }
        status = "ready" if all(checks.values()) else "unavailable"
        if not self.teacher_ready:
            status = "unavailable"
        return {
            "status": status,
            "checks": checks,
            "default_zero_updates": True,
            "trainable": _trainable_report(self.student),
            "max_updates": (None if self.authority is None else self.authority.max_updates),
            "cumulative_updates": self.cumulative_updates,
            "anchor_presentations": self.anchor_presentations,
        }

    def diagnostic_forward(
        self,
        batch: DistillationBatch,
        *,
        include_action: bool = False,
        backward: bool = True,
    ) -> dict[str, Any]:
        if self.history.is_temporal:
            raise TrainingUnavailable(
                "the current per-frame student has no registered temporal history encoder"
            )
        before = _frozen_hashes(self.bridge, self.teacher)
        output = self.student(batch.raw)
        feature = token_distillation_loss(
            output.projected_tokens,
            batch.teacher_tokens,
            output.tokens.mask,
        )
        action_value = None
        if include_action:
            if not (
                self.phase1_diagnosis.get("passed") is True
                and self.matched_cd_declaration.get("declared") is True
            ):
                raise TrainingUnavailable(
                    "D native action objective requires recorded phase-1 diagnosis and "
                    "a matched C/D declaration"
                )
            action = native_action_loss(
                self.bridge,
                output.tokens,
                lang_tokens=batch.lang_tokens,
                lang_masks=batch.lang_masks,
                state=batch.state,
                actions=batch.actions,
                noise=batch.noise,
                time=batch.time,
                action_mask=batch.future_mask,
            )
            action_value = action.action_loss
        total = feature.total
        if action_value is not None:
            total = total + action_value
        if backward:
            total.backward()
        after = _frozen_hashes(self.bridge, self.teacher)
        self.last_frozen_report = FrozenStabilityReport(before=before, after=after)
        return {
            "total": total.detach(),
            "feature_mse": feature.mse.detach(),
            "feature_cosine": feature.cosine.detach(),
            "feature_total": feature.total.detach(),
            "action_loss": None if action_value is None else action_value.detach(),
            "valid_tokens": feature.valid_tokens,
            "token_width": feature.width,
            "frozen_stable": self.last_frozen_report.stable,
            "gradient_report": _gradient_report(self.student),
            "optimizer_updates": 0,
        }

    def configure_stage2(self, *, enabled: bool = False, explicit: bool = False) -> dict[str, Any]:
        """Optional first-2 visual blocks require an explicit study decision."""
        if not enabled:
            return {"enabled": False, "scope": getattr(self.student, "_scope", None)}
        if not explicit:
            raise TrainingUnavailable(
                "stage 2 is disabled unless an explicit bottleneck diagnosis selects it"
            )
        stage2 = self.config.get("stage2", {})
        if int(stage2.get("block_count", 2)) != 2:
            raise TrainingUnavailable("the configured stage-2 block_count is not the v1 recipe")
        self.student.configure_trainable_scope(
            "stem_plus_input_blocks", input_blocks=2, explicit=True
        )
        self.config["stage2_lr"] = 1e-6
        return {
            "enabled": True,
            "scope": "stem_plus_input_blocks",
            "input_blocks": 2,
            "lr": 1e-6,
        }

    def run_accumulated_update(
        self,
        microbatches: Sequence[DistillationBatch],
        *,
        optimizer: Any,
        scheduler: Any,
        include_action: bool = False,
        allow_partial: bool = False,
        arm: str = "C",
    ) -> dict[str, Any]:
        """Actual guarded optimizer branch; never called by the engineering tests.

        The method is implemented for a future authorized execution.  This round
        deliberately calls it only with non-updating fakes, never a torch
        optimizer.
        """
        allowed, reason = self._authorized_for_update()
        if not allowed:
            raise TrainingUnavailable(reason)
        training = self._training()
        effective = int(training.get("effective_batch", 16))
        if effective < 1:
            raise TrainingUnavailable("effective_batch must be positive")
        if not microbatches:
            raise TrainingUnavailable("at least one microbatch is required")
        total_anchors = sum(int(batch.raw.values.shape[0]) for batch in microbatches)
        if total_anchors != effective and not allow_partial:
            raise TrainingUnavailable(
                f"accumulation requires {effective} anchors; got {total_anchors}"
            )
        if total_anchors > effective:
            raise TrainingUnavailable("accumulation exceeds effective_batch")
        partial = total_anchors != effective
        max_updates = min(
            int(self.authority.max_updates),
            int(training.get("max_cumulative_updates_per_arm", 5000)),
            5000,
        )
        max_anchors = min(
            int(self.authority.max_anchors),
            int(training.get("max_anchor_presentations_per_arm", 80000)),
            80000,
        )
        if self.cumulative_updates < 0 or self.anchor_presentations < 0:
            raise TrainingUnavailable("negative cumulative counters")
        if self.cumulative_updates >= max_updates:
            raise TrainingUnavailable("cumulative update budget exhausted")
        proposed_anchors = total_anchors
        if self.anchor_presentations + proposed_anchors > max_anchors:
            raise TrainingUnavailable("anchor budget would be exceeded")
        # Masks are parameter-independent; count them without retaining sixteen
        # vision graphs or changing the exact effective-batch loss denominator.
        from owac.baselines.raw_stem_siglip import _patch_pool, _prepare_measurement

        token_counts = []
        action_counts = []
        for batch in microbatches:
            prepared = _prepare_measurement(batch.raw, self.student.geometry)
            usable = prepared["usable"].flatten(0, 1).float().unsqueeze(1)
            stride = self.student.geometry.packed_patch_stride(batch.raw.input_mode)
            support = _patch_pool(usable, stride, "avg")[:, 0]
            support = support.reshape(*batch.raw.values.shape[:2], -1)
            valid = (support >= 1.0 - 1e-6) & prepared["present"][:, :, None]
            token_counts.append(int(valid.sum()))
            action_counts.append(
                int(batch.actions.shape[0] * batch.actions.shape[1])
                if batch.future_mask is None
                else int(batch.future_mask.sum())
            )
        if min(token_counts) <= 0 or min(action_counts) <= 0:
            raise TrainingUnavailable("each anchor microbatch needs valid tokens and actions")
        interval = max(1, int(training.get("frozen_audit_interval", 500)))
        reference = getattr(self, "_frozen_reference", None)
        audit_due = reference is None or (self.cumulative_updates + 1) % interval == 0
        before = _frozen_hashes(self.bridge, self.teacher) if audit_due else None
        if audit_due:
            if reference is not None and before != reference:
                raise TrainingUnavailable("frozen backend/teacher differs from initial audit")
            if reference is None:
                self._frozen_reference = before
                reference = before
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, batch in enumerate(microbatches):
            if self.history.is_temporal:
                raise TrainingUnavailable("temporal student history is not implemented")
            output = self.student(batch.raw)
            feature = token_distillation_loss(
                output.projected_tokens,
                batch.teacher_tokens,
                output.tokens.mask,
                mse_weight=float(training.get("loss", {}).get("token_mse_weight", 1.0)),
                cosine_weight=float(training.get("loss", {}).get("token_cosine_weight", 0.1)),
            )
            if feature.valid_tokens != token_counts[index]:
                raise TrainingUnavailable("student token mask differs from denominator audit")
            loss_weights = self._training().get("loss", {})
            lambda_f = float(loss_weights.get("lambda_f", 1.0))
            loss = lambda_f * feature.total * token_counts[index] / sum(token_counts)
            action_enabled = arm == "D" or include_action
            if action_enabled:
                if arm not in {"D"}:
                    raise TrainingUnavailable("native action loss is D-only")
                if not (
                    self.phase1_diagnosis.get("passed") is True
                    and self.matched_cd_declaration.get("declared") is True
                ):
                    raise TrainingUnavailable(
                        "D native action objective requires phase-1 diagnosis and matched C/D"
                    )
                action = native_action_loss(
                    self.bridge,
                    output.tokens,
                    lang_tokens=batch.lang_tokens,
                    lang_masks=batch.lang_masks,
                    state=batch.state,
                    actions=batch.actions,
                    noise=batch.noise,
                    time=batch.time,
                    action_mask=batch.future_mask,
                )
                lambda_a = float(loss_weights.get("lambda_a_D", 1.0))
                loss = loss + lambda_a * action.action_loss * action_counts[index] / sum(
                    action_counts
                )
            if not bool(torch.isfinite(loss)):
                raise TrainingUnavailable("nonfinite loss; optimizer was not called")
            loss.backward()
            losses.append(float(loss.detach()))
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in self.student.parameters()
            if p.requires_grad
        ):
            raise TrainingUnavailable("nonfinite gradient; optimizer was not called")
        clip = float(training.get("gradient_clip", 1.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.student.parameters() if p.requires_grad], clip
            )
        # This is the future-authorized branch.  Engineering tests never pass a
        # torch optimizer here; the strict test runner would fail immediately.
        optimizer.step()
        scheduler.step()
        after = _frozen_hashes(self.bridge, self.teacher) if audit_due else None
        stable = None if not audit_due else reference == after
        if stable is False:
            raise TrainingUnavailable("frozen backend/teacher changed during update")
        self.cumulative_updates += 1
        self.anchor_presentations += proposed_anchors
        report = {
            "status": "updated",
            "actual_updates": 1,
            "cumulative_updates": self.cumulative_updates,
            "anchor_presentations": self.anchor_presentations,
            "effective_batch": effective,
            "microbatches": len(microbatches),
            "partial": partial,
            "losses": losses,
            "frozen_stable": stable,
            "frozen_audit_performed": audit_due,
            "valid_token_denominator": sum(token_counts),
            "valid_action_step_denominator": sum(action_counts),
        }
        if audit_due:
            self.last_frozen_report = FrozenStabilityReport(before=before, after=after)
        self.last_update_report = report
        return report

    def dry_run_accumulation(
        self,
        microbatches: Sequence[DistillationBatch],
        *,
        include_action: bool = False,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Forward/backward proposals with counters/RNG restored; actual_updates=0."""
        training = self._training()
        effective = int(training.get("effective_batch", 16))
        anchors = sum(int(batch.raw.values.shape[0]) for batch in microbatches)
        if not allow_partial and anchors != effective:
            raise TrainingUnavailable(
                f"dry-run requires {effective} anchors or explicit allow_partial"
            )
        saved_updates = self.cumulative_updates
        saved_anchors = self.anchor_presentations
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        proposals = []
        try:
            for batch in microbatches:
                proposals.append(
                    self.diagnostic_forward(batch, include_action=include_action, backward=True)
                )
        finally:
            self.cumulative_updates = saved_updates
            self.anchor_presentations = saved_anchors
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(list(cuda_state))
        return {
            "status": "dry_run",
            "actual_updates": 0,
            "proposed_updates": len(proposals),
            "proposed_anchors": sum(int(batch.raw.values.shape[0]) for batch in microbatches),
            "cumulative_updates": self.cumulative_updates,
            "anchor_presentations": self.anchor_presentations,
            "effective_batch": effective,
            "microbatches": len(microbatches),
            "partial": len(microbatches) != effective,
            "proposals": proposals,
        }

    def optimizer_and_scheduler(self) -> tuple[Any, Any]:
        """Create state objects for resume tests; never call ``step`` here."""
        student_cfg = self.config.get("student", {})
        training = self._training()
        stem_lr = float(student_cfg.get("stem_lr", 1e-4))
        stage2_cfg = student_cfg.get("stage2", {})
        stage2_lr = float(stage2_cfg.get("lr", 1e-6))
        weight_decay = float(training.get("weight_decay", 0.01))
        warmup = int(training.get("warmup_updates", 100))
        groups: dict[str, dict[str, Any]] = {}
        for name, parameter in self.student.named_parameters():
            if not parameter.requires_grad:
                continue
            is_stem = "patch_embedding" in name
            key = "stem" if is_stem else "input_blocks"
            groups.setdefault(
                key,
                {"params": [], "lr": stem_lr if is_stem else stage2_lr, "name": key},
            )["params"].append(parameter)
        if not groups:
            raise TrainingUnavailable("no trainable student parameters")
        optimizer = torch.optim.AdamW(list(groups.values()), lr=stem_lr, weight_decay=weight_decay)

        def factor(step: int) -> float:
            return min(1.0, float(step + 1) / max(1, warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return optimizer, scheduler

    def forbid_update(self) -> None:
        raise TrainingUnavailable(
            "optimizer.step is forbidden in this engineering round; training needs a new "
            "explicit authority and a separate execution session"
        )


def _trainable_report(module: nn.Module) -> dict[str, Any]:
    names = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    return {
        "trainable_names": names,
        "trainable_parameters": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
        "backend_trainable": False,
        "bridge_projection": getattr(module, "bridge_projection", None),
    }


def _gradient_report(module: nn.Module) -> dict[str, Any]:
    entries = []
    missing = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        present = parameter.grad is not None
        entries.append({"name": name, "grad_present": present})
        if not present:
            missing.append(name)
    return {"entries": entries, "missing_gradients": missing}


def _backend_frozen(bridge: Any) -> bool:
    base = getattr(bridge, "base", None)
    if base is None:
        return True
    try:
        assert_backend_frozen(base)
    except Exception:
        return False
    return True


def _frozen_hashes(bridge: Any, teacher: Any = None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    base = getattr(bridge, "base", bridge)
    for name, parameter in base.named_parameters():
        hashes[f"backend.{name}"] = sha256_tensor(parameter)
    if teacher is not None:
        source = getattr(teacher, "policy", None)
        if isinstance(source, nn.Module):
            for name, parameter in source.named_parameters():
                hashes[f"teacher.policy.{name}"] = sha256_tensor(parameter)
        frontend = getattr(teacher, "frontend", None)
        if isinstance(frontend, nn.Module):
            for name, parameter in frontend.named_parameters():
                hashes[f"teacher.frontend.{name}"] = sha256_tensor(parameter)
    return hashes


# Backward-compatible private alias; it now hashes every frozen parameter.
_visual_hashes = _frozen_hashes


# --------------------------------------------------------------------------- #
# Resume / sampler
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ResumeState:
    optimizer_state: Mapping[str, Any]
    scheduler_state: Mapping[str, Any]
    python_rng: tuple
    numpy_rng: tuple
    torch_rng: torch.Tensor
    cuda_rng: tuple | None
    sampler_permutation: tuple[int, ...]
    sampler_cursor: int
    cumulative_updates: int
    anchor_presentations: int
    config_sha256: str
    data_sha256: str
    source_sha256: str

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "optimizer_state": self.optimizer_state,
                "scheduler_state": self.scheduler_state,
                "python_rng": self.python_rng,
                "numpy_rng": self.numpy_rng,
                "torch_rng": self.torch_rng,
                "cuda_rng": self.cuda_rng,
                "sampler_permutation": list(self.sampler_permutation),
                "sampler_cursor": self.sampler_cursor,
                "cumulative_updates": self.cumulative_updates,
                "anchor_presentations": self.anchor_presentations,
                "config_sha256": self.config_sha256,
                "data_sha256": self.data_sha256,
                "source_sha256": self.source_sha256,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ResumeState:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            optimizer_state=payload["optimizer_state"],
            scheduler_state=payload["scheduler_state"],
            python_rng=tuple(payload["python_rng"]),
            numpy_rng=tuple(payload["numpy_rng"]),
            torch_rng=payload["torch_rng"],
            cuda_rng=payload["cuda_rng"],
            sampler_permutation=tuple(int(v) for v in payload["sampler_permutation"]),
            sampler_cursor=int(payload["sampler_cursor"]),
            cumulative_updates=int(payload["cumulative_updates"]),
            anchor_presentations=int(payload["anchor_presentations"]),
            config_sha256=str(payload["config_sha256"]),
            data_sha256=str(payload["data_sha256"]),
            source_sha256=str(payload["source_sha256"]),
        )


def capture_resume_state(
    *,
    optimizer: Any,
    scheduler: Any,
    sampler_permutation: Sequence[int],
    sampler_cursor: int,
    cumulative_updates: int,
    anchor_presentations: int,
    config_sha256: str,
    data_sha256: str,
    source_sha256: str,
) -> ResumeState:
    numpy_state = np.random.get_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return ResumeState(
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        scheduler_state=copy.deepcopy(scheduler.state_dict()),
        python_rng=random.getstate(),
        numpy_rng=(
            numpy_state[0],
            tuple(int(v) for v in numpy_state[1].tolist()),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=None if cuda_state is None else tuple(t.clone() for t in cuda_state),
        sampler_permutation=tuple(int(v) for v in sampler_permutation),
        sampler_cursor=int(sampler_cursor),
        cumulative_updates=int(cumulative_updates),
        anchor_presentations=int(anchor_presentations),
        config_sha256=str(config_sha256),
        data_sha256=str(data_sha256),
        source_sha256=str(source_sha256),
    )


def restore_resume_state(state: ResumeState, *, optimizer: Any, scheduler: Any) -> None:
    optimizer.load_state_dict(copy.deepcopy(state.optimizer_state))
    scheduler.load_state_dict(copy.deepcopy(state.scheduler_state))
    random.setstate(state.python_rng)
    numpy_state = (
        state.numpy_rng[0],
        np.asarray(state.numpy_rng[1], dtype=np.uint32),
        state.numpy_rng[2],
        state.numpy_rng[3],
        state.numpy_rng[4],
    )
    np.random.set_state(numpy_state)
    torch.set_rng_state(state.torch_rng)
    if state.cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state.cuda_rng))


class AnchorBatchIterator:
    """Deterministic all-camera paired-anchor iterator.

    Each anchor contributes one control window containing every manifest camera
    in declared order.  Teacher targets are prepared before student permutation;
    adapter recurrence is reset between cameras so no camera inherits another
    camera's state/theta.  A per-frame student baseline is explicit; temporal
    student history remains unavailable.
    """

    def __init__(
        self,
        context: Any,
        *,
        split: str,
        arm: str = "C",
        seed: int = 101,
        epoch: int = 0,
    ) -> None:
        if arm not in {"B", "C", "D"}:
            raise TrainingUnavailable("arm must be B, C or D")
        self.context = context
        self.split = split
        self.arm = arm
        self.seed = int(seed)
        self.epoch = int(epoch)
        self._all_anchors = tuple(
            anchor
            for anchor in context.camera_anchors()
            if context.dataset.manifest.source_split(anchor.source_id) == split
            and anchor.variant == "rawvla_rgb3"
        )
        if not self._all_anchors:
            raise TrainingUnavailable(f"no paired anchors for split {split!r}")
        self.anchors = self._permuted()
        self.cursor = 0

    def _permuted(self):
        anchors = list(self._all_anchors)
        random.Random(self.seed + self.epoch).shuffle(anchors)
        return tuple(anchors)

    def reset_epoch(self) -> None:
        self.epoch += 1
        self.anchors = self._permuted()
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.anchors)

    def _raw_anchor(self, samples, *, mode: str):
        from owac.baselines.raw_stem_siglip import RawStemInput

        prepared = [sample.raw.as_stem_input(camera_id=sample.camera_id) for sample in samples]
        base = prepared[0]
        values = torch.cat([item.values for item in prepared], dim=1)
        masks = {}
        for name in ("valid_mask", "bad_mask", "saturated_mask", "missing_mask"):
            values_list = [getattr(item, name) for item in prepared]
            masks[name] = (
                None
                if any(value is None for value in values_list)
                else torch.cat(values_list, dim=1)
            )
        return RawStemInput(
            values=values,
            input_mode=base.input_mode,
            camera_ids=tuple(sample.camera_id for sample in samples),
            units=base.units,
            valid_mask=masks["valid_mask"],
            bad_mask=masks["bad_mask"],
            saturated_mask=masks["saturated_mask"],
            missing_mask=masks["missing_mask"],
            black_level=base.black_level,
            white_level=base.white_level,
            exposure=base.exposure,
            noise=base.noise,
            cfa_pattern=base.cfa_pattern,
            physical_origins=tuple(sample.raw.physical_origin for sample in samples),
        )

    def __iter__(self):
        from owac.backends.rawvla_language import prepare_language_state

        while self.cursor < len(self.anchors):
            anchor = self.anchors[self.cursor]
            self.cursor += 1
            if not anchor.frame_keys:
                raise TrainingUnavailable("camera anchor has no frames")
            samples = [self.context.dataset.sample(frame_key) for frame_key in anchor.frame_keys]
            if any(sample.variant != "rawvla_rgb3" for sample in samples):
                raise TrainingUnavailable("only rawvla_rgb3 anchors use this baseline")
            if self.context.history.is_temporal:
                raise TrainingUnavailable(
                    "per-frame student baseline cannot consume temporal teacher history"
                )
            first = samples[0]
            for sample in samples[1:]:
                if (
                    sample.state is None
                    or first.state is None
                    or not torch.equal(sample.state, first.state)
                ):
                    raise TrainingUnavailable("camera anchor states differ")
                if not torch.equal(sample.actions, first.actions):
                    raise TrainingUnavailable("camera anchor action labels differ")
                if not torch.equal(sample.future_mask, first.future_mask):
                    raise TrainingUnavailable("camera anchor future masks differ")
            if first.state is None:
                raise TrainingUnavailable("current state is unavailable")
            raw_input = self._raw_anchor(samples, mode="raw")
            teacher_tokens = []
            teacher_rgbs = []
            for frame_key in anchor.frame_keys:
                sample = self.context.dataset.sample(frame_key)
                raw_camera = sample.raw.as_stem_input(camera_id=sample.camera_id)
                # Independent per-camera recurrent state/theta.
                reset = getattr(self.context.teacher_adapter, "reset", None)
                if callable(reset):
                    reset()
                prepared = self.context.teacher_adapter.prepare(
                    raw_camera.values.unsqueeze(1), history=self.context.history
                )
                tokens = prepared.tokens
                if tokens.ndim == 4 and tokens.shape[1] == 1:
                    tokens = tokens[:, 0]
                teacher_tokens.append(tokens)
                teacher_rgbs.append(prepared.prepared_rgb)
            teacher = torch.cat(teacher_tokens, dim=1)
            camera_rgb = []
            for rgb in teacher_rgbs:
                if rgb.ndim == 5 and rgb.shape[1] == 1:
                    rgb = rgb[:, 0]
                if rgb.ndim != 4 or rgb.shape[1] != 3:
                    raise TrainingUnavailable("teacher current RGB must be [B,3,H,W]")
                camera_rgb.append(rgb)
            teacher_rgb = torch.stack(camera_rgb, dim=1)
            lang_tokens, lang_masks, state = prepare_language_state(
                self.context.bridge,
                [first.language],
                first.state.detach().cpu().numpy().reshape(1, -1),
                device=raw_input.values.device,
            )
            actions = torch.zeros(1, 10, 32, device=raw_input.values.device)
            actions[:, :, :7] = first.actions[:10, :7].unsqueeze(0).to(raw_input.values.device)
            if self.arm == "B":
                student_raw = self._raw_anchor(samples, mode="teacher")
                # Teacher RGB is already original-preprocessed [-1,1].
                student_raw = type(student_raw)(
                    values=teacher_rgb.to(raw_input.values.device),
                    input_mode="rgb3_control",
                    camera_ids=tuple(sample.camera_id for sample in samples),
                    units="teacher_rgb_minus1_1",
                    valid_mask=None,
                    bad_mask=None,
                    saturated_mask=None,
                    missing_mask=None,
                    physical_origins=tuple(sample.raw.physical_origin for sample in samples),
                )
            else:
                student_raw = raw_input
            yield DistillationBatch(
                raw=student_raw,
                teacher_tokens=teacher.to(raw_input.values.device),
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                actions=actions,
                future_mask=first.future_mask.unsqueeze(0).to(raw_input.values.device),
                noise=None,
                time=None,
                teacher_rgb=teacher_rgb.to(raw_input.values.device),
                arm=self.arm,
            )

    def state_dict(self) -> dict[str, Any]:
        manifest = self.context.dataset.manifest
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "arm": self.arm,
            "split": self.split,
            "permutation": [anchor.frame_keys for anchor in self.anchors],
            "record_keys": [
                frame_key for anchor in self.anchors for frame_key in anchor.frame_keys
            ],
            "manifest_sha256": manifest.manifest_sha256,
            "source_sha256": {
                source.source_id: source.source_sha256 for source in manifest.sources
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Reconstruct the saved epoch permutation before comparing it. Comparing
        # an epoch-3 checkpoint to a freshly constructed epoch-0 order is invalid.
        self.epoch = int(state["epoch"])
        self.anchors = self._permuted()
        expected = self.state_dict()
        for key in (
            "seed",
            "arm",
            "split",
            "permutation",
            "record_keys",
            "manifest_sha256",
            "source_sha256",
        ):
            if state.get(key) != expected[key]:
                raise TrainingUnavailable(f"sampler resume identity mismatch: {key}")
        self.cursor = int(state["cursor"])
        if not 0 <= self.cursor <= len(self.anchors):
            raise TrainingUnavailable("sampler cursor out of range")


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _estimated_checkpoint_bytes(trainer: RawDistillationTrainer) -> int:
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in trainer.student.state_dict().values()
    )
    return int(tensor_bytes * 1.25) + 1024 * 1024


def _identity_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _save_training_checkpoint(
    path: Path,
    *,
    trainer: RawDistillationTrainer,
    optimizer: Any,
    scheduler: Any,
    iterator: AnchorBatchIterator,
    arm: str,
    config: Mapping[str, Any],
    validation: Mapping[str, Any] | None = None,
    best_validation: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema": "owac.raw-distillation.checkpoint.v1",
        "student": trainer.student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state()[1].tolist()
        + [
            int(np.random.get_state()[2]),
            int(np.random.get_state()[3]),
            float(np.random.get_state()[4]),
        ],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "phase": {
            "scope": getattr(trainer.student, "_scope", None),
            "trainable_names": [
                n for n, p in trainer.student.named_parameters() if p.requires_grad
            ],
            "phase1_diagnosis": trainer.phase1_diagnosis,
            "matched_cd_declaration": trainer.matched_cd_declaration,
        },
        "sampler": iterator.state_dict(),
        "cumulative_updates": trainer.cumulative_updates,
        "anchor_presentations": trainer.anchor_presentations,
        "arm": arm,
        "config_sha256": _identity_hash(config),
        "data_sha256": iterator.state_dict()["manifest_sha256"],
        "source_sha256": _identity_hash(iterator.state_dict()["source_sha256"]),
        "validation": None if validation is None else dict(validation),
        "best_validation": None if best_validation is None else dict(best_validation),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("schema", "arm", "config_sha256", "data_sha256", "source_sha256"):
            if previous.get(key) != payload.get(key):
                raise TrainingUnavailable(f"refusing to overwrite unrelated checkpoint: {path}")
        if path.name != "checkpoint_best.pt":
            if previous.get("cumulative_updates") != payload["cumulative_updates"] or any(
                key not in previous["student"]
                or not torch.equal(value.cpu(), previous["student"][key])
                for key, value in payload["student"].items()
            ):
                raise TrainingUnavailable(
                    f"checkpoint already exists with different weights: {path}"
                )
            return path
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix="checkpoint-tmp-", suffix=".pt", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_training_checkpoint(
    path: str | Path,
    *,
    trainer: RawDistillationTrainer,
    optimizer: Any,
    scheduler: Any,
    iterator: AnchorBatchIterator,
    arm: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Plain weights_only-safe resume; restores weights/optimizer/RNG/sampler."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "owac.raw-distillation.checkpoint.v1":
        raise TrainingUnavailable("unsupported checkpoint schema")
    if payload.get("arm") != arm:
        raise TrainingUnavailable("checkpoint arm differs from requested arm")
    if payload.get("config_sha256") != _identity_hash(config):
        raise TrainingUnavailable("checkpoint config identity differs")
    if payload.get("data_sha256") != iterator.state_dict()["manifest_sha256"]:
        raise TrainingUnavailable("checkpoint data identity differs")
    if payload.get("source_sha256") != _identity_hash(iterator.state_dict()["source_sha256"]):
        raise TrainingUnavailable("checkpoint source identity differs")
    phase = payload.get("phase", {})
    names = [n for n, p in trainer.student.named_parameters() if p.requires_grad]
    if phase.get("trainable_names") != names or phase.get("scope") != getattr(
        trainer.student, "_scope", None
    ):
        raise TrainingUnavailable("checkpoint trainable scope differs")
    trainer.student.load_state_dict(payload["student"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(tuple(payload["python_rng"]))
    numpy_state = payload["numpy_rng"]
    np.random.set_state(
        (
            "MT19937",
            np.asarray(numpy_state[:-3], dtype=np.uint32),
            int(numpy_state[-3]),
            int(numpy_state[-2]),
            float(numpy_state[-1]),
        )
    )
    torch.set_rng_state(payload["torch_rng"])
    if payload.get("cuda_rng") is not None:
        if not torch.cuda.is_available():
            raise TrainingUnavailable("CUDA RNG checkpoint requires a CUDA runtime")
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    trainer.phase1_diagnosis = dict(phase.get("phase1_diagnosis", {}))
    trainer.matched_cd_declaration = dict(phase.get("matched_cd_declaration", {}))
    trainer._resume_best_validation = payload.get("best_validation")
    iterator.load_state_dict(payload["sampler"])
    trainer.cumulative_updates = int(payload["cumulative_updates"])
    trainer.anchor_presentations = int(payload["anchor_presentations"])
    return {
        "resumed": True,
        "cumulative_updates": trainer.cumulative_updates,
        "anchor_presentations": trainer.anchor_presentations,
        "sampler_cursor": iterator.cursor,
    }


def run_distillation_loop(
    *,
    trainer: RawDistillationTrainer,
    iterator: AnchorBatchIterator,
    optimizer: Any,
    scheduler: Any,
    checkpoint_dir: str | Path,
    arm: str = "C",
    max_updates: int | None = None,
    checkpoint_updates: Sequence[int] = (0, 500, 2000, 5000),
    validation: Callable[[], Mapping[str, Any]] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Future-authorized loop; tests invoke only no-op optimizer spies."""
    allowed, reason = trainer._authorized_for_update()
    if not allowed:
        raise TrainingUnavailable(reason)
    if arm not in {"B", "C", "D"}:
        raise TrainingUnavailable("arm must be B, C or D")
    if arm == "D" and not (
        trainer.phase1_diagnosis.get("passed") is True
        and trainer.matched_cd_declaration.get("declared") is True
    ):
        raise TrainingUnavailable("D requires phase-1 diagnosis and matched C/D declaration")
    training = trainer._training()
    effective = int(training.get("effective_batch", 16))
    if effective < 1:
        raise TrainingUnavailable("effective_batch must be positive")
    authority_cap = int(trainer.authority.max_updates) if trainer.authority else 0
    config_cap = int(training.get("max_cumulative_updates_per_arm", 5000))
    cap = min(authority_cap, config_cap, 5000)
    requested = cap if max_updates is None else min(int(max_updates), cap)
    start = trainer.cumulative_updates
    if start < 0 or trainer.anchor_presentations < 0:
        raise TrainingUnavailable("negative cumulative counters")
    if start > cap:
        raise TrainingUnavailable("checkpoint already exceeds update cap")
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved = []
    resource_limits = trainer.config.get("resource_limits", {})
    gpu_hour_cap = float(resource_limits.get("gpu_hours") or 0.0)
    disk_cap = int(resource_limits.get("training_disk_bytes") or 0)
    if not math.isfinite(gpu_hour_cap) or gpu_hour_cap <= 0:
        raise TrainingUnavailable("resource_limits.gpu_hours must be positive finite")
    if disk_cap <= 0:
        raise TrainingUnavailable("resource_limits.training_disk_bytes must be positive")
    started = time.monotonic()
    start_disk = _directory_bytes(root)
    estimated_checkpoint = _estimated_checkpoint_bytes(trainer)
    validation_cache: dict[int, dict[str, Any]] = {}

    def validation_at(step: int) -> dict[str, Any] | None:
        if validation is None:
            return None
        if step not in validation_cache:
            validation_cache[step] = dict(validation())
            reported_step = validation_cache[step].setdefault("step", step)
            if int(reported_step) != step:
                raise TrainingUnavailable("validation callback returned another checkpoint step")
        return validation_cache[step]

    initial_validation = validation_at(start)
    if start_disk + estimated_checkpoint > disk_cap:
        raise TrainingUnavailable("checkpoint reserve exceeds training disk cap")
    # Checkpoint zero is always considered before any iteration.
    if 0 in set(int(v) for v in checkpoint_updates) or start == 0:
        saved.append(
            str(
                _save_training_checkpoint(
                    root / f"checkpoint_{start}.pt",
                    trainer=trainer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    iterator=iterator,
                    arm=arm,
                    config=trainer.config,
                    validation=initial_validation,
                )
            )
        )
    if requested <= start:
        return {
            "status": "at_cap" if start >= requested else "completed",
            "actual_updates": 0,
            "cumulative_updates": start,
            "anchor_presentations": trainer.anchor_presentations,
            "checkpoints": saved,
            "stop_reason": "max_updates_zero_or_reached",
        }
    max_anchors = min(
        int(trainer.authority.max_anchors) if trainer.authority else 80000,
        int(training.get("max_anchor_presentations_per_arm", 80000)),
        80000,
    )
    if trainer.anchor_presentations + (requested - start) * effective > max_anchors:
        raise TrainingUnavailable("anchor budget would be exceeded before iteration")
    history = []
    best = None
    previous_best = getattr(trainer, "_resume_best_validation", None)
    if previous_best is not None and previous_best.get("first5_mae7") is not None:
        previous_score = float(previous_best["first5_mae7"])
        previous_step = int(previous_best["step"])
        if math.isfinite(previous_score) and previous_step <= start:
            best = (
                (previous_score, previous_step),
                {
                    "cumulative_update": previous_step,
                    "validation": previous_best,
                },
            )
    if initial_validation is not None and initial_validation.get("first5_mae7") is not None:
        initial_score = float(initial_validation["first5_mae7"])
        if math.isfinite(initial_score) and (best is None or (initial_score, start) < best[0]):
            best = (
                (initial_score, start),
                {"cumulative_update": start, "validation": initial_validation},
            )
            _save_training_checkpoint(
                root / "checkpoint_best.pt",
                trainer=trainer,
                optimizer=optimizer,
                scheduler=scheduler,
                iterator=iterator,
                arm=arm,
                config=trainer.config,
                validation=initial_validation,
                best_validation=initial_validation,
            )
    micro: list[DistillationBatch] = []
    anchors_in_batch = 0
    stop_reason = "completed"
    while trainer.cumulative_updates < requested:
        elapsed_hours = (time.monotonic() - started) / 3600.0
        if elapsed_hours >= gpu_hour_cap:
            stop_reason = "gpu_hour_cap"
            break
        if _directory_bytes(root) + estimated_checkpoint > disk_cap:
            stop_reason = "training_disk_cap"
            break
        if stop_requested is not None and stop_requested():
            stop_reason = "owned_stop_request"
            break
        try:
            batch = next(iter(iterator))
        except StopIteration:
            iterator.reset_epoch()
            continue
        micro.append(batch)
        anchors_in_batch += int(batch.raw.values.shape[0])
        if anchors_in_batch < effective:
            continue
        if anchors_in_batch != effective:
            raise TrainingUnavailable("final partial anchor batch is not enabled")
        report = trainer.run_accumulated_update(
            micro,
            optimizer=optimizer,
            scheduler=scheduler,
            include_action=(arm == "D"),
            arm=arm,
            allow_partial=False,
        )
        micro = []
        anchors_in_batch = 0
        current = trainer.cumulative_updates
        history.append({"cumulative_update": current, "loss": report["losses"][-1]})
        value = (
            validation_at(current) if current in set(int(v) for v in checkpoint_updates) else None
        )
        if value is not None:
            first5 = value.get("first5_mae7")
            if first5 is not None and math.isfinite(float(first5)):
                key = (float(first5), current)
                if best is None or key < best[0]:
                    best = (key, {"cumulative_update": current, "validation": value})
                    _save_training_checkpoint(
                        root / "checkpoint_best.pt",
                        trainer=trainer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        iterator=iterator,
                        arm=arm,
                        config=trainer.config,
                        validation=value,
                        best_validation=value,
                    )
                    saved.append(str(root / "checkpoint_best.pt"))
        if current in set(int(v) for v in checkpoint_updates):
            saved.append(
                str(
                    _save_training_checkpoint(
                        root / f"checkpoint_{current}.pt",
                        trainer=trainer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        iterator=iterator,
                        arm=arm,
                        config=trainer.config,
                        validation=value,
                        best_validation=None if best is None else best[1]["validation"],
                    )
                )
            )
    if micro and stop_reason == "completed":
        stop_reason = "incomplete_effective_batch"
    final_path = root / f"checkpoint_{trainer.cumulative_updates}.pt"
    if str(final_path) not in saved:
        saved.append(
            str(
                _save_training_checkpoint(
                    final_path,
                    trainer=trainer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    iterator=iterator,
                    arm=arm,
                    config=trainer.config,
                    validation=validation_cache.get(trainer.cumulative_updates),
                    best_validation=None if best is None else best[1]["validation"],
                )
            )
        )
    return {
        "status": "completed" if trainer.cumulative_updates >= requested else "stopped",
        "actual_updates": trainer.cumulative_updates - start,
        "cumulative_updates": trainer.cumulative_updates,
        "anchor_presentations": trainer.anchor_presentations,
        "checkpoints": saved,
        "best_validation": None if best is None else best[1],
        "stop_reason": stop_reason,
        "history": history,
    }


def make_validation_callback(
    context: Any,
    trainer: RawDistillationTrainer,
    arm: str,
    *,
    seed: int = 101,
) -> Callable[[], dict[str, Any]]:
    """Build a pure validation engine over ALL validation anchors.

    C/D use student-only RAW views; B uses actual prepared teacher RGB.  Test
    splits and teacher caches are never used.  Inference mode is entered without
    permanently changing requires_grad/training state.
    """
    import dataclasses

    import torch

    from owac.evaluation.raw_distillation import (
        FrozenBackend,
        PureStudentRuntime,
        aggregate_metrics,
        denormalize_actions,
        evaluate_windows,
        load_labels_without_teacher,
    )
    from owac.raw.distillation_pairs import RawFrameData

    if context.dataset is None or context.student is None or context.bridge is None:
        raise TrainingUnavailable("validation requires student/bridge/dataset")
    flags = {name: p.requires_grad for name, p in context.student.named_parameters()}
    modes = {name: module.training for name, module in context.student.named_modules()}
    backend = FrozenBackend(
        bridge=context.bridge,
        tokenizer=context.tokenizer,
        stats=context.stats,
        identity=context.backend_identity,
        tokenizer_sha256=str(context.tokenizer_identity.get("sha256", "")),
        stats_sha256=str(context.stats_identity.get("sha256", "")),
    )
    try:
        runtime = PureStudentRuntime(
            student=context.student, backend=backend, history=context.history
        )
    finally:
        for name, parameter in context.student.named_parameters():
            parameter.requires_grad_(flags[name])
        for name, module in context.student.named_modules():
            module.train(modes[name])

    def evaluate() -> dict[str, Any]:
        physical_records = []
        normalized_records = []
        step = int(trainer.cumulative_updates)
        anchors = [
            anchor
            for anchor in context.camera_anchors()
            if context.dataset.manifest.source_split(anchor.source_id) == "validation"
            and anchor.variant == "rawvla_rgb3"
        ]
        if not anchors:
            raise TrainingUnavailable("no validation anchors")
        flags_now = {name: p.requires_grad for name, p in context.student.named_parameters()}
        modes_now = {name: module.training for name, module in context.student.named_modules()}
        try:
            with torch.inference_mode():
                for anchor in anchors:
                    if arm == "B":
                        views = []
                        for frame_key in anchor.frame_keys:
                            sample = context.dataset.sample(frame_key)
                            context.teacher_adapter.reset()
                            prepared = context.teacher_adapter.prepare(
                                sample.raw.as_stem_input(
                                    camera_id=sample.camera_id
                                ).values.unsqueeze(1),
                                history=context.history,
                            )
                            rgb = prepared.prepared_rgb
                            if rgb.ndim == 5:
                                rgb = rgb[:, 0]
                            views.append(
                                dataclasses.replace(
                                    context.dataset.student_view(frame_key),
                                    raw=RawFrameData(
                                        values=rgb[0],
                                        units="teacher_rgb_minus1_1",
                                        raw_dtype="float32",
                                        physical_origin=sample.raw.physical_origin,
                                    ),
                                )
                            )
                    else:
                        views = [
                            context.dataset.student_view(frame_key)
                            for frame_key in anchor.frame_keys
                        ]
                    runtime.reset()
                    generator = torch.Generator(device=runtime._device()).manual_seed(seed)
                    noise = torch.randn(
                        (1, backend.action_horizon, backend.padded_dim),
                        generator=generator,
                        device=runtime._device(),
                        dtype=torch.float32,
                    )
                    if arm == "B":
                        from types import SimpleNamespace

                        from owac.backends.rawvla_language import prepare_language_state

                        rgb_input = RawStemInput(
                            values=torch.stack([view.raw.values for view in views], dim=0)
                            .unsqueeze(0)
                            .to(runtime._device()),
                            input_mode="rgb3_control",
                            units="teacher_rgb_minus1_1",
                            camera_ids=tuple(view.camera_id for view in views),
                        )
                        language, language_mask, state = prepare_language_state(
                            context.bridge,
                            [views[0].language],
                            views[0].state.reshape(1, -1),
                            device=runtime._device(),
                        )
                        visual = context.student(rgb_input).tokens
                        result = context.bridge.predict_action_from_visual_tokens(
                            visual,
                            language,
                            language_mask,
                            state,
                            noise=noise,
                            num_steps=backend.flow_steps,
                        )
                        normalized_actions = torch.as_tensor(
                            result["normalized_actions"], device=runtime._device()
                        )
                        prediction = SimpleNamespace(
                            normalized_actions=normalized_actions,
                            physical_actions=denormalize_actions(
                                normalized_actions[..., :7], backend.stats
                            ),
                        )
                    else:
                        prediction = runtime.infer_anchor(views, noise=noise)
                    labels = load_labels_without_teacher(context.dataset, anchor.frame_keys[0])
                    actions = labels["actions"]
                    future = labels["future_mask"].unsqueeze(0)
                    target_device = prediction.normalized_actions.device
                    target_normalized = torch.zeros(
                        1, backend.action_horizon, backend.padded_dim, device=target_device
                    )
                    target_normalized[:, :, : backend.control_dim] = (
                        actions[: backend.action_horizon].unsqueeze(0).to(target_device)
                    )
                    target_physical = denormalize_actions(
                        actions[: backend.action_horizon, : backend.control_dim]
                        .unsqueeze(0)
                        .to(target_device),
                        backend.stats,
                    )
                    normalized_records.append(
                        evaluate_windows(
                            prediction.normalized_actions,
                            target_normalized,
                            future,
                            split="validation",
                            step=step,
                            control_dim=backend.control_dim,
                            normalized=True,
                        )
                    )
                    physical_records.append(
                        evaluate_windows(
                            prediction.physical_actions,
                            target_physical,
                            future,
                            split="validation",
                            step=step,
                            control_dim=backend.control_dim,
                            normalized=False,
                        )
                    )
        finally:
            for name, parameter in context.student.named_parameters():
                parameter.requires_grad_(flags_now[name])
            for name, module in context.student.named_modules():
                module.train(modes_now[name])
        aggregate = aggregate_metrics(physical_records, split="validation", step=step)
        normalized = aggregate_metrics(normalized_records, split="validation", step=step)
        return {
            "step": step,
            "first5_mae7": aggregate.first5_mae7,
            "full10_mae7": aggregate.full10_mae7,
            "step1_mae7": aggregate.step1_mae7,
            "valid_first5": aggregate.valid_first5,
            "valid_full10": aggregate.valid_full10,
            "normalized": normalized.as_dict()
            if hasattr(normalized, "as_dict")
            else {
                "step1_mae7": normalized.step1_mae7,
                "first5_mae7": normalized.first5_mae7,
                "full10_mae7": normalized.full10_mae7,
            },
            "windows": len(physical_records),
        }

    return evaluate

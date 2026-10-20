"""Author RawVLA teacher: model construction, authority gates, preflight and training.

Only the author's RAW frontend is trainable. The original pi05/SigLIP backend is
loaded from the recovered checkpoint, frozen, and retained in eval mode with
autograd through its image input. No student or learned-frontend asset is
required.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from owac.raw.teacher_data import (
    CAMERA_ORDER,
    PADDED_ACTION_DIM,
    TeacherData,
    TeacherWindow,
)

PI05_CHECKPOINT_SHA256 = "68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8"
PI05_TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
PI05_STATS_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
AUTHOR_REVISION = "ba9c56b04da2c4931b70d8844710f5928d43775b"
AUTHOR_ARCHIVE_SHA256 = "57bc54e8a695ae07c2ffee9ebb2549e9efbb48536a97df14c5c8f63c52f3e012"
STARVLA_REVISION = "57d508cb180d0510ac9060358009c0f6ffaac420"
MAX_UPDATES = 2000
PILOT_UPDATES = 100
EFFECTIVE_BATCH = 8
MAX_ANCHORS = 16000


class TeacherTrainingBlocked(RuntimeError):
    """A required asset, identity, authority, or preflight gate is absent."""


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherAuthority:
    path: str
    sha256: str
    max_updates: int
    max_anchors: int
    max_gpu_hours: float
    max_new_disk_bytes: int
    max_replays: int
    lambda_action: float
    lambda_brightness: float
    lambda_chroma: float

    @classmethod
    def from_file(cls, path: str | Path) -> TeacherAuthority:
        source = Path(path)
        if not source.is_file():
            raise TeacherTrainingBlocked(f"authority receipt absent: {source}")
        payload = json.loads(source.read_text())
        if payload.get("schema") != "owac.raw-teacher.authorization.v1":
            raise TeacherTrainingBlocked("authority schema differs")
        auth = payload.get("authorization", {})
        if not auth.get("teacher_optimizer_updates"):
            raise TeacherTrainingBlocked("authority does not authorize teacher updates")
        if auth.get("student_training"):
            raise TeacherTrainingBlocked("student training must remain disabled")
        if auth.get("physical_actions"):
            raise TeacherTrainingBlocked("physical actions must remain disabled")
        limits = payload.get("limits", {})
        training = payload.get("training", {})
        max_updates = int(training.get("max_cumulative_updates", 0))
        if not 1 <= max_updates <= MAX_UPDATES:
            raise TeacherTrainingBlocked("teacher update budget exceeds 2000 or is zero")
        if int(training.get("effective_batch", 0)) != EFFECTIVE_BATCH:
            raise TeacherTrainingBlocked("authority effective batch must be 8")
        if (
            int(training.get("microbatch", 0)) != 1
            or int(training.get("gradient_accumulation", 0)) != 8
        ):
            raise TeacherTrainingBlocked("authority requires microbatch1 x accumulation8")
        if int(training.get("max_anchor_presentations", 0)) > MAX_ANCHORS:
            raise TeacherTrainingBlocked("authority anchor cap exceeds 16000")
        if float(limits.get("local_gpu_hours", 0)) > 12:
            raise TeacherTrainingBlocked("authority GPU-hour cap exceeds 12")
        if int(limits.get("additional_disk_bytes", 0)) > 50 * 1024**3:
            raise TeacherTrainingBlocked("authority disk cap exceeds 50 GiB")
        if int(limits.get("demonstrations_replayed_max", 0)) > 50:
            raise TeacherTrainingBlocked("authority replay cap exceeds 50")
        return cls(
            path=str(source),
            sha256=_file_sha(source),
            max_updates=max_updates,
            max_anchors=int(training.get("max_anchor_presentations", 0)),
            max_gpu_hours=float(limits["local_gpu_hours"]),
            max_new_disk_bytes=int(limits["additional_disk_bytes"]),
            max_replays=int(limits.get("demonstrations_replayed_max", 0)),
            lambda_action=float(training.get("lambda_native_action", 0.15)),
            lambda_brightness=float(training.get("lambda_brightness", 0.01)),
            lambda_chroma=float(training.get("lambda_chroma", 0.001)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "max_updates": self.max_updates,
            "max_anchors": self.max_anchors,
            "max_gpu_hours": self.max_gpu_hours,
            "max_new_disk_bytes": self.max_new_disk_bytes,
            "max_replays": self.max_replays,
            "lambda_action": self.lambda_action,
            "lambda_brightness": self.lambda_brightness,
            "lambda_chroma": self.lambda_chroma,
        }


@dataclass(slots=True)
class TeacherAssets:
    author_root: Path
    starvla_root: Path
    checkpoint: Path
    tokenizer: Path
    statistics: Path
    config_paths: tuple[Path, ...]


@dataclass(slots=True)
class AuthorTeacher:
    policy: Any
    frontend: nn.Module
    assets: TeacherAssets
    config: Mapping[str, Any]
    backend_hashes: Mapping[str, str]
    frontend_param_names: tuple[str, ...]
    load_audit: Mapping[str, Any] = field(default_factory=dict)

    def frontend_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for _, parameter in self.frontend.named_parameters()
            if parameter.requires_grad
        ]


def _unresolved_strings(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        result = []
        for key, item in value.items():
            result.extend(_unresolved_strings(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_unresolved_strings(item, f"{prefix}[{index}]"))
        return result
    if isinstance(value, str) and "${" in value:
        return [prefix]
    return []


def _effective_config(cfg: Any) -> dict[str, Any]:
    """Resolve only used config sections; record unresolved unused keys."""
    from omegaconf import OmegaConf

    framework = OmegaConf.to_container(cfg.framework, resolve=True)
    trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
    unresolved = _unresolved_strings(OmegaConf.to_container(cfg, resolve=False))
    return {
        "framework": framework,
        "trainer": trainer,
        "unresolved_unused": sorted(unresolved),
    }


def _sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_author_teacher(
    assets: TeacherAssets, *, device: str | torch.device = "cpu"
) -> AuthorTeacher:
    """Build exact PI05 + author RAWVLA; backend checkpoint loaded before attach.

    The frontend is created after the frozen-backend audit, so missing random
    frontend keys cannot be mistaken for checkpoint omissions.
    """
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    for path in (
        assets.author_root,
        assets.starvla_root,
        assets.author_root / "experiments" / "simulation" / "libero",
    ):
        _sys_path(path)
    try:
        from omegaconf import OmegaConf
        from starVLA.config_loader import load_config
        from starVLA.model.framework.VLM4A.PI05 import PI05
        from starVLA.model.modules.raw_frontend import build_raw_frontend

        from owac.backends.rawvla import load_audited
        from owac.backends.rawvla_features_n02 import load_checkpoint_state
    except Exception as exc:
        raise TeacherTrainingBlocked(f"teacher imports/environment unavailable: {exc}") from exc
    if not assets.config_paths:
        raise TeacherTrainingBlocked("a child author config is required")
    child_config = assets.config_paths[-1]
    if not child_config.is_file():
        raise TeacherTrainingBlocked(f"config path absent: {child_config}")
    try:
        # Official loader honors extends relative to this child.
        cfg = load_config(child_config)
    except Exception as exc:
        raise TeacherTrainingBlocked(f"author config merge failed: {exc}") from exc
    # Resolve every asset/env field used by model construction explicitly.
    cfg.framework.tokenizer.model_path = str(assets.tokenizer)
    cfg.trainer.pretrained_checkpoint = None
    raw_cfg = OmegaConf.to_container(cfg.framework.raw_frontend, resolve=True)
    if not isinstance(raw_cfg, dict):
        raise TeacherTrainingBlocked("raw_frontend config is absent")
    expected_raw = {
        "burst_frames": 6,
        "rnn_frames": 6,
        "state_dim": 128,
        "luma_state_dim": 64,
        "fft_patch_size": 32,
        "fft_stride": 16,
        "split_luma_chroma_condition": True,
    }
    for key, value in expected_raw.items():
        if raw_cfg.get(key) != value:
            raise TeacherTrainingBlocked(
                f"raw_frontend.{key}={raw_cfg.get(key)!r} differs from verified geometry {value!r}"
            )
    # Build without a frontend to keep the checkpoint audit exactly backend-only.
    cfg.framework.raw_frontend = None
    try:
        policy = PI05(cfg)
        state = load_checkpoint_state(assets.checkpoint)
        audit = load_audited(policy, state)
    except Exception as exc:
        raise TeacherTrainingBlocked(f"PI05/checkpoint load failed: {exc}") from exc
    if audit.get("active_missing") or audit.get("unexpected") or audit.get("shape_mismatch"):
        raise TeacherTrainingBlocked(f"backend load audit failed: {audit}")
    del state
    gc.collect()
    frontend = build_raw_frontend(OmegaConf.create(raw_cfg))
    if frontend is None:
        raise TeacherTrainingBlocked("author RAW frontend was not constructed")
    policy.raw_frontend_cfg = OmegaConf.create(raw_cfg)
    policy.raw_frontend = frontend
    policy.to(device)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    frontend.to(device=device, dtype=torch.float32)
    frontend.train(True)
    for parameter in frontend.parameters():
        parameter.requires_grad_(True)
    names = tuple(
        name for name, parameter in frontend.named_parameters() if parameter.requires_grad
    )
    count = sum(parameter.numel() for parameter in frontend.parameters())
    if count != 829172:
        raise TeacherTrainingBlocked(f"author frontend parameter count {count} != 829172")
    audit = dict(audit)
    audit["effective_policy_contract"] = effective_policy_contract(policy)
    hashes = {
        name: hashlib.sha256(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, tensor in policy.state_dict().items()
        if not name.startswith("raw_frontend.")
    }
    effective_config = _effective_config(cfg)
    effective_config["framework"]["raw_frontend"] = raw_cfg
    return AuthorTeacher(
        policy=policy,
        frontend=frontend,
        assets=assets,
        config=effective_config,
        backend_hashes=hashes,
        frontend_param_names=names,
        load_audit=audit,
    )


def effective_policy_contract(policy: Any) -> dict[str, Any]:
    """Record the loaded policy's effective state/tokenizer contract.

    The instance flag is authoritative; class defaults are not inspected.
    """
    framework = getattr(getattr(policy, "config", None), "framework", None)
    if framework is None:
        raise TeacherTrainingBlocked("policy effective config is unavailable")
    configured_flag = bool(getattr(framework, "discrete_state_input", False))
    instance_flag = bool(getattr(policy, "discrete_state_input", configured_flag))
    tokenizer_cfg = getattr(framework, "tokenizer", None)
    configured_pad = float(getattr(tokenizer_cfg, "pad_state_value", 0.0))
    effective_pad = 0.0 if not instance_flag else configured_pad

    def dtype_of(module: Any) -> str | None:
        if module is None:
            return None
        try:
            return str(next(module.parameters()).dtype)
        except (StopIteration, AttributeError, TypeError):
            return None

    return {
        "configured_discrete_state_input": configured_flag,
        "instance_discrete_state_input": instance_flag,
        "configured_pad_state_value": configured_pad,
        "effective_pad_state_value": effective_pad,
        "tokenizer_state_mode": "text_state" if instance_flag else "none",
        "image_keys": list(getattr(framework, "image_keys", ()) or ()),
        "image_resolution": list(getattr(framework, "image_resolution", ()) or ()),
        "action_dim": int(getattr(framework, "action_dim", 0)),
        "action_horizon": int(getattr(framework, "action_horizon", 0)),
        "flow_steps": int(getattr(framework, "num_inference_steps", 0)),
        "max_state_dim": int(getattr(framework, "max_state_dim", 0)),
        "language_module_dtype": dtype_of(
            getattr(getattr(policy, "vlm", None), "language_model", None)
        ),
        "action_module_dtype": dtype_of(
            getattr(getattr(policy, "action_head", None), "model", None)
        ),
    }


def _window_example(
    window: TeacherWindow,
    *,
    instruction: str = "put the cream cheese in the bowl",
    stats: Any = None,
) -> dict[str, Any]:
    # raw shape [T,K,C,H,W,3] -> per-camera [T,K,3,H,W] float [0,1]
    raw_images = [
        torch.from_numpy(window.raw_uint8[:, :, camera].transpose(0, 1, 4, 2, 3)).float() / 255.0
        for camera in range(window.raw_uint8.shape[2])
    ]
    normalized = window.normalized(stats) if stats is not None else None
    return {
        "raw_burst_float32": raw_images,
        "lang": instruction,
        # Native PI05 pads state8 itself according to the effective config.
        # Actual LIBERO config has discrete_state_input=false (0.0 pad, no
        # state token); never pass a pre-padded 32-vector here.
        "state": np.asarray(
            normalized["state"] if normalized is not None else window.state,
            dtype=np.float32,
        ).reshape(8),
        "action": np.asarray(
            normalized["padded_actions"] if normalized is not None else window.padded_actions,
            dtype=np.float32,
        ),
        "rawvla_light": {"lighting_domain": "normal"},
        "window": window,
    }


def native_masked_action_loss(output: Mapping[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    """Native flow-velocity squared loss masked by valid future actions.

    Native ``forward_from_observation`` returns its own unmasked mean; this
    function recomputes the same velocity objective with the valid future mask
    and reports real-7 and padded-32 diagnostics separately.
    """
    if "u_t" not in output or "v_t" not in output:
        raise TeacherTrainingBlocked("native forward did not return u_t/v_t")
    u_t = output["u_t"]
    v_t = output["v_t"]
    if u_t.shape != v_t.shape or u_t.ndim != 3 or u_t.shape[-1] != PADDED_ACTION_DIM:
        raise TeacherTrainingBlocked("native velocity tensors have unexpected shape")
    valid = mask.to(device=u_t.device, dtype=torch.bool)
    if valid.shape != u_t.shape[:2] or not bool(valid.any()):
        raise TeacherTrainingBlocked("future/action mask has no valid entries")
    squared = (u_t.float() - v_t.float()).square()
    action_loss = squared[valid].mean()
    real7 = squared[valid][:, :7] if squared[valid].shape[-1] >= 7 else None
    return {
        "action_loss": action_loss,
        "action_loss_sum": squared[valid].mean(dim=-1).sum(),
        "real7_mse": None if real7 is None else real7.mean(),
        "padded32_mse": squared[valid].mean(),
        "valid_actions": int(valid.sum()),
    }


class TeacherTrainer:
    """Exact author frontend training with a frozen backend and full resume."""

    def __init__(
        self,
        *,
        teacher: AuthorTeacher,
        data: TeacherData,
        authority: TeacherAuthority,
        run_root: str | Path,
        stats: Any = None,
        budget: Any = None,
        activation_memory_mode: str = "none",
    ) -> None:
        self.teacher = teacher
        self.data = data
        self.authority = authority
        self.run_root = Path(run_root)
        self.stats = stats
        self.budget = budget
        if activation_memory_mode not in {"none", "save_on_cpu"}:
            raise TeacherTrainingBlocked("activation_memory_mode must be none or save_on_cpu")
        self.activation_memory_mode = activation_memory_mode
        self.cumulative_updates = 0
        self.anchor_presentations = 0
        self._started = time.monotonic()
        self._step_times: list[float] = []
        self.best_validation: dict[str, Any] | None = None

    def _assert_frozen_backend(self) -> None:
        actual = {
            name: hashlib.sha256(
                value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, value in self.teacher.policy.state_dict().items()
            if name in self.teacher.backend_hashes
        }
        if actual != self.teacher.backend_hashes:
            raise TeacherTrainingBlocked("frozen backend differs from initial loaded weights")

    def _validate(self, callback: Any) -> dict[str, Any]:
        if self.budget is not None:
            self.budget.check_deadline()
        rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )
        try:
            value = dict(callback())
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
            self.teacher.policy.eval()
        score = value.get("first5_mae7")
        if score is None or not math.isfinite(float(score)):
            raise TeacherTrainingBlocked("validation first5_mae7 is missing/non-finite")
        self._assert_frozen_backend()
        value["step"] = self.cumulative_updates
        self.run_root.mkdir(parents=True, exist_ok=True)
        with (self.run_root / "validation.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(value, sort_keys=True) + "\n")
        return value

    def _policy_examples(self, windows: Sequence[TeacherWindow]) -> list[dict[str, Any]]:
        return [
            _window_example(window, instruction="put the cream cheese in the bowl")
            for window in windows
        ]

    def _activation_context(self):
        import contextlib

        if self.activation_memory_mode == "save_on_cpu":
            return torch.autograd.graph.save_on_cpu(pin_memory=True)
        return contextlib.nullcontext()

    def _micro_forward(
        self,
        window: TeacherWindow,
        *,
        retain_graph: bool = False,
        action_denominator: int | None = None,
        aux_scale: float = 1.0,
    ) -> dict[str, Any]:
        example = _window_example(window, stats=self.stats)
        self.teacher.frontend.train(True)
        output = self.teacher.policy.forward([example], return_debug=True)
        loss = native_masked_action_loss(output, torch.from_numpy(window.action_mask).unsqueeze(0))
        exposure = output.get("exposure_prior_loss")
        chroma = output.get("chroma_prior_loss")
        action_term = (
            loss["action_loss"]
            if action_denominator is None
            else loss["action_loss_sum"] / float(action_denominator)
        )
        total = self.authority.lambda_action * action_term
        if exposure is not None:
            total = total + float(aux_scale) * self.authority.lambda_brightness * exposure
        if chroma is not None:
            total = total + float(aux_scale) * self.authority.lambda_chroma * chroma
        if not torch.isfinite(total):
            raise TeacherTrainingBlocked("non-finite teacher loss")
        loss.update(
            {
                "total_loss": total,
                "exposure_prior_loss": exposure,
                "chroma_prior_loss": chroma,
                "output": output,
            }
        )
        return loss

    def preflight(self) -> dict[str, Any]:
        if self.authority.max_gpu_hours <= 0 or self.authority.max_new_disk_bytes <= 0:
            raise TeacherTrainingBlocked("positive resource caps are required")
        validation_anchors = list(self.data.iter_anchors("validation"))
        if not validation_anchors:
            raise TeacherTrainingBlocked("no validation anchors")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        before = dict(self.teacher.backend_hashes)
        self.teacher.policy.train(False)
        self.teacher.frontend.train(True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with self._activation_context():
                result = self._micro_forward(self.data.window(validation_anchors[0]))
                result["total_loss"].backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            grad_names = [
                name
                for name, parameter in self.teacher.frontend.named_parameters()
                if parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            ]
            after = {
                name: hashlib.sha256(
                    parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for name, parameter in self.teacher.policy.state_dict().items()
                if name in before
            }
            grad_tensors = [
                parameter.grad.detach().float().pow(2).sum()
                for parameter in self.teacher.frontend.parameters()
                if parameter.grad is not None
            ]
            frontend_grad_norm = float(
                torch.stack(grad_tensors).sum().sqrt() if grad_tensors else torch.tensor(0.0)
            )
            if not grad_names:
                raise TeacherTrainingBlocked("preflight produced no frontend gradients")
            if not math.isfinite(frontend_grad_norm) or frontend_grad_norm <= 0.0:
                raise TeacherTrainingBlocked("preflight frontend gradient is zero or non-finite")
            if before != after:
                raise TeacherTrainingBlocked("frozen backend changed during preflight")
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            return {
                "schema": "owac.raw-teacher.preflight.v1",
                "status": "ready",
                "forward_backward_seconds": elapsed,
                "peak_allocated_bytes": peak,
                "frontend_grad_norm": frontend_grad_norm,
                "frontend_grad_params": len(grad_names),
                "backend_stable": before == after,
                "backend_hash_count": len(before),
                "updates": 0,
                "optimizer_updates": 0,
                "window_source": validation_anchors[0].source_demo_id,
                "cameras": list(CAMERA_ORDER),
                "raw_shape": list(self.data.window(validation_anchors[0]).raw_uint8.shape),
            }
        finally:
            for parameter in self.teacher.frontend.parameters():
                parameter.grad = None
            self.teacher.policy.eval()
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if rng_state["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])

    def _save(
        self,
        path: Path,
        *,
        validation: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        if path.exists() and not overwrite:
            raise TeacherTrainingBlocked(f"refusing to overwrite existing checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        payload = {
            "schema": "owac.rawvla-teacher.checkpoint.v1",
            "frontend": self.teacher.frontend.state_dict(),
            "optimizer": None if not hasattr(self, "optimizer") else self.optimizer.state_dict(),
            "scheduler": None if not hasattr(self, "scheduler") else self.scheduler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": {
                "name": np.random.get_state()[0],
                "keys": np.random.get_state()[1].tolist(),
                "pos": int(np.random.get_state()[2]),
                "has_gauss": int(np.random.get_state()[3]),
                "cached_gaussian": float(np.random.get_state()[4]),
            },
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "cumulative_updates": int(self.cumulative_updates),
            "anchor_presentations": int(self.anchor_presentations),
            "sampler_position": int(getattr(self, "cursor", 0)),
            "sampler": None if not hasattr(self, "sampler") else self.sampler.state_dict(),
            "trainable_names": self.teacher.frontend_param_names,
            "backend_scope": "frozen_all_non_raw_frontend",
            "activation_memory_mode": self.activation_memory_mode,
            "effective_policy_contract": getattr(self.teacher, "load_audit", {}).get(
                "effective_policy_contract"
            ),
            "frontend_state_keys": sorted(self.teacher.frontend.state_dict()),
            "source": {
                "author_revision": AUTHOR_REVISION,
                "author_archive_sha256": AUTHOR_ARCHIVE_SHA256,
                "checkpoint_sha256": PI05_CHECKPOINT_SHA256,
                "tokenizer_sha256": PI05_TOKENIZER_SHA256,
                "statistics_sha256": PI05_STATS_SHA256,
            },
            "gpu_wall_seconds": time.monotonic() - self._started,
            "config_sha256": _identity(self.teacher.config),
            "data_sha256": _identity(self.data.manifest.as_dict()),
            "authority_sha256": self.authority.sha256,
            "validation": None if validation is None else dict(validation),
            "best_validation": self.best_validation,
            "backend_hashes": dict(self.teacher.backend_hashes),
        }
        torch.save(payload, tmp)
        tmp.replace(path)

    def _make_optimizer(self) -> None:
        if hasattr(self, "optimizer"):
            return
        self.optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.teacher.frontend.parameters()
                if parameter.requires_grad
            ],
            lr=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: (
                step / 100.0
                if step < 100
                else 0.01
                + 0.99
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi * min(1.0, (step - 100) / max(1, self.authority.max_updates - 100))
                    )
                )
            ),
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "owac.rawvla-teacher.checkpoint.v1":
            raise TeacherTrainingBlocked("teacher checkpoint schema differs")
        if payload.get("config_sha256") != _identity(self.teacher.config):
            raise TeacherTrainingBlocked("teacher checkpoint config identity differs")
        if payload.get("data_sha256") != _identity(self.data.manifest.as_dict()):
            raise TeacherTrainingBlocked("teacher checkpoint data identity differs")
        if payload.get("authority_sha256") != self.authority.sha256:
            raise TeacherTrainingBlocked("teacher checkpoint authority identity differs")
        expected_source = {
            "author_revision": AUTHOR_REVISION,
            "author_archive_sha256": AUTHOR_ARCHIVE_SHA256,
            "checkpoint_sha256": PI05_CHECKPOINT_SHA256,
            "tokenizer_sha256": PI05_TOKENIZER_SHA256,
            "statistics_sha256": PI05_STATS_SHA256,
        }
        if payload.get("source") != expected_source:
            raise TeacherTrainingBlocked("teacher checkpoint source/asset identity differs")
        self.teacher.frontend.load_state_dict(payload["frontend"], strict=True)
        self._make_optimizer()
        if payload.get("optimizer") is None or payload.get("scheduler") is None:
            raise TeacherTrainingBlocked("teacher checkpoint lacks optimizer/scheduler state")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        random.setstate(tuple(payload["python_rng"]))
        numpy_rng = payload["numpy_rng"]
        np.random.set_state(
            (
                str(numpy_rng["name"]),
                np.asarray(numpy_rng["keys"], dtype=np.uint32),
                int(numpy_rng["pos"]),
                int(numpy_rng["has_gauss"]),
                float(numpy_rng["cached_gaussian"]),
            )
        )
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        self.cumulative_updates = int(payload["cumulative_updates"])
        self.anchor_presentations = int(payload["anchor_presentations"])
        self.cursor = int(payload.get("sampler_position", 0))
        if payload.get("sampler") is not None:
            self.sampler = self.data.sampler(split="train", seed=101)
            self.sampler.load_state_dict(payload["sampler"])
        self.validation = payload.get("validation")
        self.best_validation = payload.get("best_validation")
        return {
            "resumed": True,
            "cumulative_updates": self.cumulative_updates,
            "anchor_presentations": self.anchor_presentations,
        }

    def train(
        self,
        *,
        max_updates: int | None = None,
        validation: Any = None,
    ) -> dict[str, Any]:
        # Real training is intentionally impossible without the explicit authority
        # object and a validation callback. This method is not called by import.
        if validation is None:
            raise TeacherTrainingBlocked("teacher training requires a validation callback")
        if not isinstance(self.teacher.frontend, nn.Module):
            raise TeacherTrainingBlocked("author frontend is not a torch module")
        target = (
            self.authority.max_updates
            if max_updates is None
            else min(int(max_updates), self.authority.max_updates)
        )
        if target <= 0 or target > MAX_UPDATES:
            raise TeacherTrainingBlocked("invalid cumulative update target")
        self.preflight()
        if not list(self.data.iter_anchors("train")):
            raise TeacherTrainingBlocked("no train anchors")
        self._make_optimizer()
        if not hasattr(self, "sampler"):
            self.sampler = self.data.sampler(split="train", seed=101)
        self.teacher.policy.eval()
        initial_validation = self._validate(validation)
        if initial_validation.get("first5_mae7") is None or not math.isfinite(
            float(initial_validation["first5_mae7"])
        ):
            raise TeacherTrainingBlocked("initial validation first5_mae7 is missing/non-finite")
        if self.best_validation is None:
            self.best_validation = {
                "first5_mae7": float(initial_validation["first5_mae7"]),
                "step": self.cumulative_updates,
            }
        if self.cumulative_updates == 0:
            self._save(
                self.run_root / "checkpoint_0000.pt",
                validation=initial_validation,
            )
            self._save(self.run_root / "checkpoint_best.pt", validation=initial_validation)
        best = (
            float(self.best_validation["first5_mae7"]),
            int(self.best_validation["step"]),
        )
        history = []
        stop_reason = "completed"
        while self.cumulative_updates < target:
            if self.budget is not None:
                self.budget.check_deadline()
            if (time.monotonic() - self._started) / 3600.0 >= self.authority.max_gpu_hours:
                stop_reason = "gpu_hour_cap"
                break
            estimated = (
                sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in self.teacher.frontend.state_dict().values()
                )
                * 12  # frontend + AdamW moments + atomic resume/candidate copies
                + 4 * 1024 * 1024
            )
            disk_root = self.budget.run_root if self.budget is not None else self.run_root
            current_bytes = sum(
                item.stat().st_size for item in disk_root.rglob("*") if item.is_file()
            )
            if current_bytes + estimated > self.authority.max_new_disk_bytes:
                stop_reason = "training_disk_cap"
                break
            anchors = [self.sampler.next_anchor() for _ in range(8)]
            valid_counts = [int(self.data.window(anchor).action_mask.sum()) for anchor in anchors]
            global_valid = sum(valid_counts)
            if global_valid <= 0:
                raise TeacherTrainingBlocked("microbatch has no valid future actions")
            self.optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            for anchor in anchors:
                if self.budget is not None:
                    self.budget.check_deadline()
                window = self.data.window(anchor)
                with self._activation_context():
                    loss = self._micro_forward(
                        window,
                        retain_graph=False,
                        action_denominator=global_valid,
                        aux_scale=1.0 / 8.0,
                    )
                    loss["total_loss"].backward()
                self.anchor_presentations += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.teacher.frontend.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            if self.budget is not None:
                self.budget.charge_update(anchors=8)
                self.budget.check_deadline()
            self.optimizer.step()
            self.scheduler.step()
            self.cumulative_updates += 1
            self._step_times.append(time.perf_counter() - start)
            self._save(self.run_root / "checkpoint_resume.pt", overwrite=True)
            if self.budget is not None:
                self.budget.commit_update()
            row = {
                "update": self.cumulative_updates,
                "anchors": self.anchor_presentations,
                "last_micro_total_loss": float(loss["total_loss"].detach()),
                "last_micro_native_action_loss": float(loss["action_loss"].detach()),
                "last_micro_real7_mse": float(loss["real7_mse"].detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "update_compute_seconds": self._step_times[-1],
                "sampler_epoch": self.sampler.epoch,
            }
            with (self.run_root / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
            if self.cumulative_updates == 1 or self.cumulative_updates % 20 == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            history.append(
                {
                    "update": self.cumulative_updates,
                    "loss": float(loss["total_loss"].detach()),
                }
            )
            if self.cumulative_updates in {100, 500, 1000, 2000}:
                value = self._validate(validation)
                key = (float(value["first5_mae7"]), self.cumulative_updates)
                if key < best:
                    best = key
                    self.best_validation = {"first5_mae7": best[0], "step": best[1]}
                self._save(
                    self.run_root / f"checkpoint_{self.cumulative_updates:04d}.pt",
                    validation=value,
                )
                first5 = value.get("first5_mae7")
                if first5 is None or not math.isfinite(float(first5)):
                    raise TeacherTrainingBlocked("validation returned invalid first5_mae7")
                if key == best:
                    self._save(
                        self.run_root / "checkpoint_best.pt",
                        validation=value,
                        overwrite=True,
                    )
            if self.anchor_presentations >= self.authority.max_anchors:
                stop_reason = "anchor_cap"
                break
        final_path = self.run_root / f"checkpoint_{self.cumulative_updates:04d}.pt"
        if not final_path.exists():
            self._save(
                final_path,
                validation=self._validate(validation)
                if self.cumulative_updates
                else initial_validation,
            )
        return {
            "schema": "owac.raw-teacher.train.v1",
            "actual_updates": self.cumulative_updates,
            "anchor_presentations": self.anchor_presentations,
            "history": history,
            "checkpoint_dir": str(self.run_root),
            "stop_reason": stop_reason,
            "best_validation": {"first5_mae7": best[0], "step": best[1]},
        }


__all__ = [
    "AuthorTeacher",
    "EFFECTIVE_BATCH",
    "MAX_ANCHORS",
    "MAX_UPDATES",
    "PILOT_UPDATES",
    "TeacherAssets",
    "TeacherAuthority",
    "TeacherTrainer",
    "TeacherTrainingBlocked",
    "build_author_teacher",
    "native_masked_action_loss",
]

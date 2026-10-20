"""Zero-update cost profiling for RAW distillation.

Profiling never constructs/calls an optimizer update.  Timing is measured per
anchor and normalized by the actual number of anchors, not by microbatch count.
Teacher/backend hashing is audited separately from model timing.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from owac.training.raw_distillation import (
    DistillationBatch,
    RawDistillationTrainer,
    _frozen_hashes,
    native_action_loss,
    token_distillation_loss,
)

__all__ = ["CostProfile", "ProfileUnavailable", "profile_inference", "profile_train_step"]


class ProfileUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class CostProfile:
    evidence_level: str
    device: str
    dtype: str
    batch_size: int
    context: Mapping[str, Any]
    parameter_count: int
    trainable_names: list[str]
    gradient_norms: Mapping[str, float | None]
    frozen_hashes_before: Mapping[str, str]
    frozen_hashes_after: Mapping[str, str]
    frozen_stable: bool
    forward_backward_ms: list[float]
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    audit_ms: float
    inference_p50_ms: float | None = None
    inference_p95_ms: float | None = None
    inference_samples: int = 0
    teacher_prep_ms: float | None = None
    assumptions: Mapping[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    @property
    def update_cap(self) -> int:
        return int(self.assumptions.get("update_cap", 0))

    @property
    def effective_batch(self) -> int:
        return int(self.assumptions.get("effective_batch", 0))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gpu_hours"] = None
        return data


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ProfileUnavailable("no timing samples")
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _gradient_backup(module: torch.nn.Module):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _restore_grads(module: torch.nn.Module, backup: Mapping[str, torch.Tensor | None]) -> None:
    for name, parameter in module.named_parameters():
        saved = backup.get(name)
        parameter.grad = None if saved is None else saved.detach().clone()


def profile_train_step(
    trainer: RawDistillationTrainer,
    microbatches: Sequence[DistillationBatch],
    *,
    evidence_level: str = "fixture_contract",
    update_cap: int = 5000,
    effective_batch: int = 16,
    repetitions: int = 1,
    include_action: bool = False,
) -> CostProfile:
    """Forward/backward only; optimizer.step is never called."""
    if not microbatches:
        raise ProfileUnavailable("real/fixture paired microbatches are unavailable")
    if evidence_level not in {"fixture_contract", "real_policy"}:
        raise ProfileUnavailable("profile evidence level must be declared")
    if effective_batch < 1 or update_cap < 0:
        raise ProfileUnavailable("invalid effective batch/update cap")
    first = microbatches[0]
    device = first.raw.values.device
    dtype = str(first.raw.values.dtype).replace("torch.", "")
    total_anchors = sum(int(batch.raw.values.shape[0]) for batch in microbatches)
    if total_anchors < 1:
        raise ProfileUnavailable("profile has no anchors")
    rng = _rng_state()
    grad_backup = _gradient_backup(trainer.student)
    trainer.student.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    audit_start = time.perf_counter()
    before = _frozen_hashes(trainer.bridge, trainer.teacher)
    audit_ms = (time.perf_counter() - audit_start) * 1000.0
    times: list[float] = []
    anchors_measured = 0
    try:
        for _ in range(max(1, int(repetitions))):
            trainer.student.zero_grad(set_to_none=True)
            _sync(device)
            start = time.perf_counter()
            for batch in microbatches:
                output = trainer.student(batch.raw)
                loss = token_distillation_loss(
                    output.projected_tokens, batch.teacher_tokens, output.tokens.mask
                ).total
                if include_action:
                    action = native_action_loss(
                        trainer.bridge,
                        output.tokens,
                        lang_tokens=batch.lang_tokens,
                        lang_masks=batch.lang_masks,
                        state=batch.state,
                        actions=batch.actions,
                        noise=batch.noise,
                        time=batch.time,
                        action_mask=batch.future_mask,
                    )
                    loss = loss + action.action_loss
                loss.backward()
                anchors_measured += int(batch.raw.values.shape[0])
            _sync(device)
            times.append((time.perf_counter() - start) * 1000.0)
        after = _frozen_hashes(trainer.bridge, trainer.teacher)
        gradient_norms = {
            name: (None if p.grad is None else float(p.grad.detach().float().norm()))
            for name, p in trainer.student.named_parameters()
            if p.requires_grad
        }
        trainable = [name for name, p in trainer.student.named_parameters() if p.requires_grad]
        student_params = sum(p.numel() for p in trainer.student.parameters())
        backend = getattr(trainer.bridge, "base", None)
        backend_params = (
            sum(p.numel() for p in backend.parameters())
            if isinstance(backend, torch.nn.Module)
            else 0
        )
        total_params = student_params + backend_params
        limitations = []
        if evidence_level == "fixture_contract":
            limitations.append("fixture_contract only; no real model/GPU-hours claim")
        if device.type != "cuda":
            limitations.append("GPU memory/latency not measured on CPU fixture")
        peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        peak_res = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        per_anchor = [value / max(1, total_anchors) for value in times]
        return CostProfile(
            evidence_level=evidence_level,
            device=str(device),
            dtype=dtype,
            batch_size=int(first.raw.values.shape[0]),
            context={
                "history_temporal": bool(trainer.history.is_temporal),
                "history_frames": trainer.history.temporal_frames,
                "phase": getattr(trainer.student, "_scope", None),
                "include_action": bool(include_action),
                "backend_parameters": backend_params,
                "student_parameters": student_params,
                "total_parameters": total_params,
            },
            parameter_count=total_params,
            trainable_names=trainable,
            gradient_norms=gradient_norms,
            frozen_hashes_before=before,
            frozen_hashes_after=after,
            frozen_stable=before == after,
            forward_backward_ms=times,
            peak_allocated_bytes=peak_alloc,
            peak_reserved_bytes=peak_res,
            audit_ms=audit_ms,
            assumptions={
                "effective_batch": int(effective_batch),
                "update_cap": int(update_cap),
                "measured_microbatches": len(microbatches),
                "measured_anchors_per_repetition": total_anchors,
                "measured_repetitions": max(1, int(repetitions)),
                "anchors_total_measured": anchors_measured,
                "one_update_estimate_ms": statistics.mean(per_anchor) * int(effective_batch),
                "cumulative_update_estimate_ms": statistics.mean(per_anchor)
                * int(effective_batch)
                * int(update_cap),
            },
            limitations=limitations,
        )
    finally:
        _restore_grads(trainer.student, grad_backup)
        _restore_rng(rng)


def profile_inference(
    runtime: Any,
    samples: Sequence[Any],
    *,
    warmup: int = 1,
    repetitions: int = 5,
    evidence_level: str = "fixture_contract",
) -> dict[str, Any]:
    if not samples:
        raise ProfileUnavailable("pure-student inference samples are unavailable")
    if int(warmup) < 0 or int(repetitions) < 1:
        raise ProfileUnavailable("warmup/repetitions must be nonnegative/positive")
    for sample in samples[:warmup]:
        runtime.infer(sample)
    times = []
    for _ in range(int(repetitions)):
        start = time.perf_counter()
        for sample in samples:
            runtime.infer(sample)
        _sync(runtime._device())
        times.append((time.perf_counter() - start) * 1000.0 / len(samples))
    return {
        "evidence_level": evidence_level,
        "samples": len(samples),
        "warmup": int(warmup),
        "repetitions": int(repetitions),
        "p50_ms": _percentile(times, 0.50),
        "p95_ms": _percentile(times, 0.95),
        "teacher_prep_ms": None,
        "limitations": ["teacher/cache preparation is measured separately"],
    }

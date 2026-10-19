"""Fresh P2 forward/backward profiling. Never performs an optimizer update."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager

import numpy as np
import torch

from owac.evaluation.n02_full.v12 import anchor_indices
from owac.frameworks.n02_full.sequence_v12 import EXECUTION_REVISION
from owac.training.n02_v12 import training_contract


@contextmanager
def forbid_optimizer_steps(optimizer):
    original = optimizer.step

    def forbidden(*args, **kwargs):
        raise RuntimeError("optimizer.step is forbidden in a zero-update profile")

    optimizer.step = forbidden
    try:
        yield
    finally:
        optimizer.step = original


def parameter_digest(parameters):
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_summary(groups, gradients):
    report = {}
    for role, parameters in groups.items():
        square, nonzero, connected, finite = 0.0, 0, 0, True
        for parameter in parameters:
            grad = gradients.get(id(parameter))
            if grad is None:
                continue
            connected += 1
            norm = float(grad.detach().float().norm())
            finite = finite and math.isfinite(norm)
            nonzero += int(norm > 0)
            square += norm * norm
        report[role] = {
            "l2": math.sqrt(square),
            "finite": finite,
            "connected_tensors": connected,
            "nonzero_tensors": nonzero,
            "total_tensors": len(parameters),
        }
    return report


def profile_fresh(trainer, *, batches=3):
    """Real official train windows and full frozen backend, with immutable trainable weights."""
    if batches < 1 or trainer.plan.resume is not None:
        raise ValueError("fresh profile needs positive batches and no resume checkpoint")
    if trainer.optimizer.state:
        raise ValueError("fresh profile requires an optimizer with no update state")
    parameters = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("optimizer contains duplicate parameters")
    expected = {id(p) for p in trainer.model.parameters() if p.requires_grad}
    expected.update(id(p) for p in trainer.masked.parameters() if p.requires_grad)
    expected.update(
        id(p) for m in trainer.bridge.bridge_modules() for p in m.parameters() if p.requires_grad
    )
    expected.update(id(p) for p in trainer.bridge.base.parameters() if p.requires_grad)
    if expected != {id(p) for p in parameters}:
        raise ValueError("optimizer does not cover every trainable frontend parameter exactly once")
    before = parameter_digest(parameters)
    groups = dict(trainer.model.parameter_groups())
    groups["masked_branch"] = trainer.masked.own_parameters()
    groups["bridge_projection"] = trainer.parameter_groups["bridge_projection"]
    assigned = {id(p) for values in groups.values() for p in values}
    groups["other_inference"] = [p for p in parameters if id(p) not in assigned]
    generator = torch.Generator(device=trainer.device).manual_seed(trainer.plan.seed + 101)
    records = []

    def sync():
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)

    trainer.model.train()
    trainer.masked.train()
    with forbid_optimizer_steps(trainer.optimizer):
        for number, index in enumerate(anchor_indices(trainer.sources["train"], batches)):
            trainer.optimizer.zero_grad(set_to_none=True)
            if trainer.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(trainer.device)
            trainer.log(f"ZERO-update profile window {number + 1}/{batches}, train anchor {index}")
            sync()
            started = time.perf_counter()
            result = trainer.microbatch(
                index,
                min(index + trainer.plan.batch_size, trainer.sources["train"].anchor_count),
                generator=generator,
            )
            sync()
            forward_seconds = time.perf_counter() - started
            if result is None:
                records.append({"anchor_index": index, "skipped": "all action targets padded"})
                continue
            if not bool(torch.isfinite(result["loss"])):
                raise FloatingPointError("nonfinite fresh loss")
            per_loss = {}
            if number == 0:
                for name, term in result["loss_tensors"].items():
                    if term is None or not term.requires_grad:
                        continue
                    grads = torch.autograd.grad(
                        term, parameters, allow_unused=True, retain_graph=True
                    )
                    per_loss[name] = gradient_summary(
                        groups, dict(zip(map(id, parameters), grads, strict=True))
                    )
                    del grads
            sync()
            started = time.perf_counter()
            result["loss"].backward()
            sync()
            backward_seconds = time.perf_counter() - started
            total_gradients = gradient_summary(groups, {id(p): p.grad for p in parameters})
            if not all(value["finite"] for value in total_gradients.values()):
                raise FloatingPointError("nonfinite fresh gradient")
            frozen_grads = [
                name for name, p in trainer.bridge.base.named_parameters() if p.grad is not None
            ]
            if frozen_grads:
                raise RuntimeError(f"frozen backend received gradients: {frozen_grads[:3]}")
            output = trainer.token_source.last_output
            current_input = result["batch"]
            frame_shapes = {
                camera: list(frame.values.shape)
                for camera, frame in current_input.observation.frames.items()
            }
            record = {
                "anchor_index": index,
                "loss": float(result["loss"].detach()),
                "terms": result["terms"],
                "info": result["info"],
                "forward_seconds_including_io": forward_seconds,
                "total_backward_seconds": backward_seconds,
                "per_loss_gradients": per_loss,
                "total_gradients": total_gradients,
                "q": output.q.detach().cpu().tolist(),
                "scored_scalars_per_candidate": output.diagnostics.get(
                    "scored_scalars_per_candidate"
                ),
                "saturation_policy": output.diagnostics.get("saturation_policy"),
                "frame_shapes_including_history": frame_shapes,
                "camera_ids": list(output.diagnostics.get("camera_ids", ())),
                "query_coverage_per_camera": list(output.diagnostics.get("coverage", ())),
                "query_count_per_camera": list(
                    output.diagnostics.get("per_camera_query_count", ())
                ),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device)
                if trainer.device.type == "cuda"
                else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device)
                if trainer.device.type == "cuda"
                else None,
            }
            records.append(record)
            trainer.log(
                f"profile anchor {index}: loss={record['loss']:.6g}, "
                f"F={forward_seconds:.2f}s B={backward_seconds:.2f}s, optimizer updates=0"
            )
            del result
    trainer.optimizer.zero_grad(set_to_none=True)
    after = parameter_digest(parameters)
    if before != after or trainer.optimizer.state:
        raise RuntimeError("profile changed weights or optimizer update state")
    valid = [r for r in records if "loss" in r]
    if not valid:
        raise RuntimeError("no supervised profile windows")
    elapsed = [r["forward_seconds_including_io"] + r["total_backward_seconds"] for r in valid]
    report = {
        "format": "n02-v12-fresh-zero-update-profile-v1",
        "execution_revision": EXECUTION_REVISION,
        "run_id": trainer.plan.run_id,
        "arm": trainer.plan.arm,
        "platform": trainer.plan.platform,
        "seed": trainer.plan.seed,
        "initialisation": "fresh frontend, official frozen pi0.5 checkpoint",
        "optimizer_updates": 0,
        "optimizer_state_entries": len(trainer.optimizer.state),
        "weights_unchanged": before == after,
        "trainable_sha256_before": before,
        "trainable_sha256_after": after,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "parameters_by_optimizer_role": {
            role: sum(p.numel() for p in values)
            for role, values in trainer.parameter_groups.items()
        },
        "frozen_backend_parameters": trainer.bridge.frozen_parameter_count,
        "adam_moment_bytes_estimate_unallocated": sum(
            2 * p.numel() * p.element_size() for p in parameters
        ),
        "bptt_steps": trainer.token_source.bptt_steps,
        "history": trainer.plan.history,
        "batch_size": trainer.plan.batch_size,
        "grad_accum_configured_not_executed": trainer.plan.grad_accum,
        "fwbw_seconds_p50": float(np.percentile(elapsed, 50)),
        "fwbw_seconds_p95": float(np.percentile(elapsed, 95)),
        "timing_scope": (
            "forward includes data IO, total backward; per-loss probes excluded from timing "
            "but included in peak memory; no optimizer step"
        ),
        "training_contract": training_contract(trainer.plan, trainer.identity),
        "assets": trainer.assets,
        "windows": records,
        "claim_boundary": (
            "implementation/gradient/resource profile only; "
            "no optimisation stability, training performance or closed-loop evidence"
        ),
    }
    trainer._write_json("fresh_profile.json", report)
    trainer.status(
        state="profile_complete", activity="从头画像完成；优化更新 0，参数校验一致；未启动训练"
    )
    trainer.log(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "optimizer_updates",
                    "weights_unchanged",
                    "fwbw_seconds_p50",
                    "fwbw_seconds_p95",
                )
            },
            ensure_ascii=False,
        )
    )
    return report

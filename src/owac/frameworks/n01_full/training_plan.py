"""Matched warmup/backend adaptation and checkpointed optimizer ownership.

All four arms use the same scopes. The original RGB tower is never trainable or
called. A parameter is present in the optimizer from the start, so unfreezing does
not silently discard Adam state or make a resumed optimizer a different shape.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from owac.backends.rawvla_features import VISION_KEY_PREFIXES


@dataclass(frozen=True)
class TrainingPlan:
    warmup_steps: int = 2000
    total_steps: int = 50000
    encoder_lr: float = 1e-5
    backend_lr: float = 1e-6
    lr_warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    backend_prefixes: tuple[str, ...] = ("action_head.",)
    gradient_clip: float = 1.0
    motion_aux_weight: float = 0.05
    version: str = "matched-encoder-warmup-action-expert-v1"

    def __post_init__(self):
        if not 0 <= self.warmup_steps < self.total_steps or self.lr_warmup_steps < 0:
            raise ValueError("invalid finite training/warmup budget")
        if min(self.encoder_lr, self.backend_lr, self.gradient_clip) <= 0:
            raise ValueError("learning rates and clip must be positive")
        if not 0 < self.min_lr_ratio <= 1 or self.motion_aux_weight < 0:
            raise ValueError("invalid LR ratio or optional motion supervision weight")
        if not self.backend_prefixes or any(not x for x in self.backend_prefixes):
            raise ValueError("backend adaptation needs explicit nonempty parameter prefixes")
        if any(
            v.startswith(p) or p.startswith(v)
            for p in self.backend_prefixes
            for v in VISION_KEY_PREFIXES
        ):
            raise ValueError("backend adaptation cannot include the replaced RGB tower")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if "backend_prefixes" in data:
            data["backend_prefixes"] = tuple(data["backend_prefixes"])
        return cls(**data)


class BackendSchedule:
    def __init__(self, model, policy, plan: TrainingPlan, *, master_weights=False):
        self.model, self.policy, self.plan = model, policy, plan
        self.backend = []
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(False)
            if name.startswith(plan.backend_prefixes):
                if name.startswith(VISION_KEY_PREFIXES):
                    raise ValueError("RGB tower reached the adaptation scope")
                if master_weights and parameter.dtype != torch.float32:
                    # Small LR updates must accumulate in FP32 rather than round
                    # away in a BF16 parameter at every Adam step.
                    parameter.data = parameter.data.float()
                self.backend.append((name, parameter))
        if not self.backend:
            raise ValueError(f"no backend parameters match {plan.backend_prefixes}")
        self.phase = "encoder_warmup"

    def parameter_groups(self):
        return [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": self.plan.encoder_lr,
                "name": "encoder_motion_bridge",
            },
            {
                "params": [p for _, p in self.backend],
                "lr": self.plan.backend_lr,
                "name": "matched_backend",
            },
        ]

    def apply(self, step, optimizer=None):
        adapting = int(step) >= self.plan.warmup_steps
        for _, parameter in self.backend:
            parameter.requires_grad_(adapting)
        self.phase = "backend_adaptation" if adapting else "encoder_warmup"
        warmup = min((step + 1) / max(self.plan.lr_warmup_steps, 1), 1.0)
        progress = min(
            max(step - self.plan.lr_warmup_steps, 0)
            / max(self.plan.total_steps - self.plan.lr_warmup_steps, 1),
            1.0,
        )
        ratio = warmup * (
            self.plan.min_lr_ratio
            + (1 - self.plan.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        )
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = ratio * (
                    self.plan.backend_lr
                    if group["name"] == "matched_backend"
                    else self.plan.encoder_lr
                )
        return self.report(step)

    def report(self, step):
        return {
            "phase": self.phase,
            "step": int(step),
            "plan": self.plan.to_dict(),
            "backend_tensors": len(self.backend),
            "backend_numel": sum(p.numel() for _, p in self.backend),
            "backend_parameter_names": [name for name, _ in self.backend],
        }

    def checkpoint(self):
        return {name: p.detach().cpu() for name, p in self.backend}

    def restore(self, payload):
        if set(payload) != {name for name, _ in self.backend}:
            raise ValueError("checkpoint backend adaptation scope differs")
        with torch.no_grad():
            for name, parameter in self.backend:
                if payload[name].shape != parameter.shape:
                    raise ValueError(f"backend shape mismatch for {name}")
                parameter.copy_(payload[name].to(parameter))

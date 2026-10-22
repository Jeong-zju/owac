"""Temporal distillation training engine with explicit full-prefix burn-in.

The engine is concrete but unauthorized by default.  It never calls an optimizer
during diagnostics.  A future authorized run must supply an explicit authority
receipt, config bit, accepted teacher, source split, parity and resource receipts.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import random
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from owac.baselines.raw_temporal_siglip import TemporalRawStemStudent, TemporalState
from owac.raw.temporal_distillation import TemporalAnchorSample
from owac.training.raw_distillation import native_action_loss, token_distillation_loss

__all__ = [
    "TemporalDistillationTrainer",
    "TemporalTrainingAuthority",
    "TemporalTrainingBlocked",
    "TemporalAnchorSampler",
    "TemporalTrainingConfig",
    "assert_split_allowed",
]


class TemporalTrainingBlocked(RuntimeError):
    """A real temporal training gate or resource is absent."""


@dataclass(frozen=True, slots=True)
class TemporalTrainingAuthority:
    authorized: bool = False
    max_updates: int = 0
    max_anchors: int = 0
    max_wall_seconds: int = 0
    max_output_bytes: int = 0
    scope: str = "raw-stem-temporal"
    receipt_sha256: str = ""
    authority_path: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TemporalTrainingAuthority:
        return cls(
            authorized=bool(value.get("authorized", False)),
            max_updates=int(value.get("max_updates", 0)),
            max_anchors=int(value.get("max_anchors", 0)),
            max_wall_seconds=int(value.get("max_wall_seconds", 0)),
            max_output_bytes=int(value.get("max_output_bytes", 0)),
            scope=str(value.get("scope", "raw-stem-temporal")),
            receipt_sha256=str(value.get("receipt_sha256", "")),
            authority_path=str(value.get("authority_path", "")),
        )


@dataclass(frozen=True, slots=True)
class TemporalTrainingConfig:
    seed: int = 101
    effective_batch: int = 16
    microbatch: int = 1
    max_cumulative_updates_per_arm: int = 5000
    pilot_max_updates_per_arm: int = 100
    max_anchor_presentations_per_arm: int = 80000
    checkpoint_updates: tuple[int, ...] = (0, 500, 2000, 5000)
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_updates: int = 100
    lambda_f: float = 1.0
    cosine_weight: float = 0.1
    lambda_a_D: float = 1.0
    optimizer: str = "AdamW"
    frozen_audit_interval: int = 500
    earlystop_patience: int = 0
    activation_memory_mode: str | None = None
    authorization: Mapping[str, Any] = field(default_factory=lambda: {"optimizer_updates": False})
    stage2: Mapping[str, Any] = field(default_factory=lambda: {"enabled": False})

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TemporalTrainingConfig:
        training = value.get("training", value)
        if not isinstance(training, Mapping):
            raise TemporalTrainingBlocked("training config is not a mapping")
        return cls(
            seed=int(training.get("seed", 101)),
            effective_batch=int(training.get("effective_batch", 16)),
            microbatch=int(training.get("microbatch", 1)),
            max_cumulative_updates_per_arm=int(
                training.get("max_cumulative_updates_per_arm", 5000)
            ),
            pilot_max_updates_per_arm=int(training.get("pilot_max_updates_per_arm", 100)),
            max_anchor_presentations_per_arm=int(
                training.get("max_anchor_presentations_per_arm", 80000)
            ),
            checkpoint_updates=tuple(int(item) for item in training.get("checkpoint_updates", ())),
            learning_rate=float(training.get("learning_rate", 1.0e-4)),
            weight_decay=float(training.get("weight_decay", 0.01)),
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            warmup_updates=int(training.get("warmup_updates", 100)),
            lambda_f=float(training.get("lambda_f", 1.0)),
            cosine_weight=float(training.get("cosine_weight", 0.1)),
            lambda_a_D=float(training.get("lambda_a_D", 1.0)),
            optimizer=str(training.get("optimizer", "AdamW")),
            frozen_audit_interval=int(training.get("frozen_audit_interval", 500)),
            earlystop_patience=int(training.get("earlystop_patience", 0)),
            activation_memory_mode=(
                None
                if training.get("activation_memory_mode") in (None, "none", "")
                else str(training.get("activation_memory_mode"))
            ),
            authorization=dict(value.get("authorization", {})),
            stage2=dict(training.get("stage2", {})),
        )


class TemporalAnchorSampler:
    """Seeded full-permutation anchor sampler with exact resumable cursor."""

    SCHEMA = "owac.raw-stem-temporal.anchor-sampler.v1"

    def __init__(
        self,
        candidates: Sequence[tuple[str, int]],
        *,
        seed: int = 101,
        batch_size: int = 16,
    ) -> None:
        self.candidates = [(str(source), int(endpoint)) for source, endpoint in candidates]
        if not self.candidates:
            raise TemporalTrainingBlocked("sampler requires candidates")
        if len(set(self.candidates)) != len(self.candidates):
            raise TemporalTrainingBlocked("sampler candidates contain duplicates")
        if int(batch_size) < 1:
            raise TemporalTrainingBlocked("sampler batch_size must be positive")
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(self.seed)
        self.epoch = 0
        self.permutation = [int(value) for value in self.rng.permutation(len(self.candidates))]
        self.cursor = 0

    def next_batch(self) -> list[tuple[str, int]]:
        selected: list[tuple[str, int]] = []
        while len(selected) < self.batch_size:
            if self.cursor >= len(self.permutation):
                self.epoch += 1
                self.permutation = [
                    int(value) for value in self.rng.permutation(len(self.candidates))
                ]
                self.cursor = 0
            take = min(self.batch_size - len(selected), len(self.permutation) - self.cursor)
            selected.extend(self.permutation[self.cursor : self.cursor + take])
            self.cursor += take
        return [self.candidates[index] for index in selected]

    def state(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "candidates": self.candidates,
            "epoch": self.epoch,
            "permutation": list(self.permutation),
            "cursor": self.cursor,
            "generator_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or payload.get("schema") != self.SCHEMA:
            raise TemporalTrainingBlocked("anchor sampler state schema differs")
        if int(payload.get("seed", -1)) != self.seed:
            raise TemporalTrainingBlocked("anchor sampler seed differs")
        if int(payload.get("batch_size", -1)) != self.batch_size:
            raise TemporalTrainingBlocked("anchor sampler batch size differs")
        saved_candidates = payload.get("candidates")
        if saved_candidates is not None and tuple(
            (str(source), int(endpoint)) for source, endpoint in saved_candidates
        ) != tuple(self.candidates):
            raise TemporalTrainingBlocked("anchor sampler candidates differ")
        permutation = [int(value) for value in payload["permutation"]]
        if sorted(permutation) != list(range(len(self.candidates))):
            raise TemporalTrainingBlocked("anchor sampler permutation differs")
        self.epoch = int(payload["epoch"])
        self.permutation = permutation
        self.cursor = int(payload["cursor"])
        state = copy.deepcopy(payload["generator_state"])
        generator = np.random.Generator(np.random.PCG64())
        generator.bit_generator.state = state
        self.rng = generator


def assert_split_allowed(split: str, *, purpose: str) -> None:
    if split not in {"train", "validation", "test"}:
        raise TemporalTrainingBlocked(f"unknown split {split!r}")
    if purpose in {"select", "train"} and split == "test":
        raise TemporalTrainingBlocked("test split is never selectable or trainable")
    if purpose == "train" and split != "train":
        raise TemporalTrainingBlocked("training may consume only the train split")


class TemporalDistillationTrainer:
    """Full-prefix burn-in, six-endpoint unroll, exact token/action denominators."""

    def __init__(
        self,
        *,
        student: TemporalRawStemStudent,
        bridge: Any,
        config: TemporalTrainingConfig | Mapping[str, Any],
        authority: TemporalTrainingAuthority | None = None,
        teacher_ready: bool = False,
        source_split_ready: bool = False,
        parity_ready: bool = False,
        resource_ready: bool = False,
        phase1_diagnosis: Mapping[str, Any] | None = None,
        matched_cd_declaration: Mapping[str, Any] | None = None,
        teacher: Any = None,
        arm: str = "C",
        run_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.student = student
        self.bridge = bridge
        self.config = (
            config
            if isinstance(config, TemporalTrainingConfig)
            else TemporalTrainingConfig.from_mapping(config)
        )
        self.authority = authority
        self.teacher_ready = bool(teacher_ready)
        self.source_split_ready = bool(source_split_ready)
        self.parity_ready = bool(parity_ready)
        self.resource_ready = bool(resource_ready)
        self.phase1_diagnosis = dict(phase1_diagnosis or {})
        self.matched_cd_declaration = dict(matched_cd_declaration or {})
        self.teacher = teacher
        self.arm = arm
        self.run_identity = dict(run_identity or {})
        if self.run_identity.get("scope") not in (None, "stem_plus_gate"):
            raise TemporalTrainingBlocked("run identity scope must be stem_plus_gate")
        self.scope = self.run_identity.get("scope", getattr(student, "_scope", "unknown"))
        self.cumulative_updates = 0
        self.anchor_presentations = 0
        self.sampler_state: dict[str, Any] = {}
        self.run_elapsed_seconds: float = 0.0
        self.selector_state: dict[str, Any] = {
            "best_value": None,
            "best_step": None,
            "best_checkpoint_path": None,
            "best_checkpoint_sha256": None,
        }
        if self.config.effective_batch < 1 or self.config.microbatch != 1:
            raise TemporalTrainingBlocked(
                "temporal v1 requires effective_batch>=1 and microbatch=1"
            )
        if self.config.activation_memory_mode not in (None, "save_on_cpu"):
            raise TemporalTrainingBlocked("unknown activation_memory_mode")

    def _authorized_for_update(self, *, include_action: bool) -> tuple[bool, str]:
        if self.authority is None or not self.authority.authorized:
            return False, "explicit actual temporal training authority receipt is required"
        if self.config.authorization.get("optimizer_updates") is not True:
            return False, "config.authorization.optimizer_updates is false"
        for name, value in (
            ("teacher_ready", self.teacher_ready),
            ("source_split_ready", self.source_split_ready),
            ("parity_ready", self.parity_ready),
            ("resource_ready", self.resource_ready),
        ):
            if not value:
                return False, f"{name} is not satisfied"
        if self.authority.max_updates <= 0 or self.authority.max_anchors <= 0:
            return False, "authority update/anchor caps must be positive"
        hard_updates = min(
            5000,
            self.authority.max_updates,
            self.config.max_cumulative_updates_per_arm,
        )
        hard_anchors = min(
            80000,
            self.authority.max_anchors,
            self.config.max_anchor_presentations_per_arm,
        )
        if self.cumulative_updates >= hard_updates:
            return False, "update budget exhausted"
        if self.anchor_presentations >= hard_anchors:
            return False, "anchor budget exhausted"
        if include_action and not (
            self.phase1_diagnosis.get("passed") is True
            and self.matched_cd_declaration.get("declared") is True
        ):
            return False, "D action loss requires phase-1 diagnosis and matched C/D"
        return True, "authorized"

    def _frozen_modules(self) -> list[tuple[str, nn.Module]]:
        modules: list[tuple[str, nn.Module]] = [("student_base", self.student.base)]
        teacher_module = self.teacher
        if teacher_module is not None and not isinstance(teacher_module, nn.Module):
            teacher_module = getattr(teacher_module, "policy", None)
        if isinstance(teacher_module, nn.Module):
            modules.append(("teacher", teacher_module))
        bridge_base = getattr(self.bridge, "base", None)
        if isinstance(bridge_base, nn.Module):
            modules.append(("bridge_base", bridge_base))
        seen: set[int] = set()
        unique: list[tuple[str, nn.Module]] = []
        for label, module in modules:
            if id(module) in seen:
                continue
            seen.add(id(module))
            unique.append((label, module))
        return unique

    def validate_optimizer_groups(self, optimizer: Any) -> None:
        if hasattr(self.student, "stem") and hasattr(self.student, "gate_logit"):
            expected = {
                id(parameter): parameter
                for parameter in self.student.stem.parameters()
                if parameter.requires_grad
            }
            if self.student.gate_logit.requires_grad:
                expected[id(self.student.gate_logit)] = self.student.gate_logit
        else:
            expected = {
                id(parameter): parameter
                for parameter in self.student.parameters()
                if parameter.requires_grad
            }
        actual: list[nn.Parameter] = []
        for group in optimizer.param_groups:
            for parameter in group.get("params", ()):
                if not isinstance(parameter, nn.Parameter):
                    raise TemporalTrainingBlocked("optimizer group contains a non-Parameter")
                actual.append(parameter)
        actual_ids = [id(parameter) for parameter in actual]
        if len(actual_ids) != len(set(actual_ids)):
            raise TemporalTrainingBlocked("optimizer parameter groups contain duplicates")
        if set(actual_ids) != set(expected):
            raise TemporalTrainingBlocked(
                "optimizer parameter groups differ from declared student trainables"
            )
        if any(not parameter.requires_grad for parameter in actual):
            raise TemporalTrainingBlocked("optimizer group contains a frozen parameter")

    def _burn_prefix(self, sample: TemporalAnchorSample) -> TemporalState | None:
        state: TemporalState | None = None
        with torch.no_grad():
            for packet in sample.prefix_packets:
                state = self.student.update_state(packet, state)
            if state is not None:
                state = self.student.detach_state(state)
        if state is not None and state.last_endpoint != sample.unroll_packets[0].endpoint - 5:
            raise TemporalTrainingBlocked("prefix burn-in endpoint does not precede the unroll")
        return state

    def forward_anchor(
        self,
        sample: TemporalAnchorSample,
        *,
        include_action: bool = False,
        use_cached_pairs: bool = False,
    ) -> dict[str, Any]:
        """Differentiable six-endpoint unroll from a current-student burn-in state."""
        if sample.target.source_id != sample.source_id:
            raise TemporalTrainingBlocked("anchor target/source mismatch")
        for row in sample.unroll_rows:
            if row.source_id != sample.source_id:
                raise TemporalTrainingBlocked("unroll row source mismatch")
        state = self._burn_prefix(sample)
        # Only the final endpoint needs the inherited SigLIP tower.  The preceding
        # five endpoints advance the recurrent patch state without building five
        # unused vision graphs.
        for packet in sample.unroll_packets[:-1]:
            state = self.student.update_state(packet, state)
        final, state = self.student(sample.unroll_packets[-1], state)
        target_device = final.projected_tokens.device
        teacher = sample.target.teacher_tokens.to(target_device)
        teacher_mask = (
            sample.target.teacher_token_mask[:, None]
            .expand(sample.target.teacher_tokens.shape[0], sample.target.teacher_tokens.shape[1])
            .to(target_device)
        )
        teacher_flat = teacher.reshape(1, -1, teacher.shape[-1])
        teacher_mask_flat = teacher_mask.reshape(1, -1)
        combined_mask = final.tokens.mask.to(dtype=torch.bool) & teacher_mask_flat
        feature = token_distillation_loss(
            final.projected_tokens,
            teacher_flat,
            combined_mask,
            cosine_weight=self.config.cosine_weight,
        )
        action = None
        action_count = 0
        if include_action:
            actions = sample.target.actions32.unsqueeze(0).to(final.projected_tokens.device)
            mask = sample.target.future_mask.unsqueeze(0).to(final.projected_tokens.device)
            device = final.projected_tokens.device
            # Cached noise/time are diagnostic pairs only.  Formal training uses
            # the native action-head sampler, not a fixed t=0.5 pair.
            noise = sample.target.noise.to(device) if use_cached_pairs else None
            flow_time = sample.target.flow_time.to(device) if use_cached_pairs else None
            action = native_action_loss(
                self.bridge,
                final.tokens,
                lang_tokens=sample.target.lang_tokens.to(device),
                lang_masks=sample.target.lang_masks.to(device),
                state=sample.target.state32.to(device),
                actions=actions,
                noise=noise,
                time=flow_time,
                action_mask=mask,
            )
            action_count = int(mask.sum())
        total = self.config.lambda_f * feature.total
        if action is not None:
            total = total + self.config.lambda_a_D * action.action_loss
        return {
            "total": total,
            "feature": feature,
            "action_loss": None if action is None else action.action_loss,
            "valid_tokens": feature.valid_tokens,
            "valid_actions": action_count,
            "outputs": [final],
            "final_output": final,
            "state": state,
            "sample": sample,
        }

    def _activation_context(self) -> Any:
        if self.config.activation_memory_mode == "save_on_cpu":
            return torch.autograd.graph.save_on_cpu(pin_memory=True)
        return contextlib.nullcontext()

    def _count_feature_tokens(self, sample: TemporalAnchorSample) -> int:
        """Count the exact final token mask without stem or vision execution."""
        from owac.baselines.raw_stem_siglip import _patch_pool, _prepare_measurement

        packet = sample.unroll_packets[-1]
        frame = packet.frames[-1]
        prepared = _prepare_measurement(frame, self.student.geometry)
        patch_stride = self.student.geometry.packed_patch_stride(frame.input_mode)
        usable = prepared["usable"].flatten(0, 1).to(torch.float32).unsqueeze(1)
        support = _patch_pool(usable, patch_stride, "avg")[:, 0].reshape(
            1, len(frame.camera_ids), int(self.student.geometry.num_patches)
        )
        current_mask = (support >= 1.0 - 1e-6) & prepared["present"][:, :, None]
        if packet.padding_mask[-1]:
            current_mask = torch.zeros_like(current_mask)
        teacher_mask = sample.target.teacher_token_mask[:, None].expand(
            sample.target.teacher_token_mask.shape[0],
            int(self.student.geometry.num_patches),
        )
        combined = current_mask.reshape(1, -1) & teacher_mask.reshape(1, -1).to(current_mask.device)
        return int(combined.sum())

    def zero_update_gradient_diagnostic(
        self,
        sample: TemporalAnchorSample,
        *,
        include_action: bool = False,
        mode: str = "total",
        use_cached_pairs: bool = True,
    ) -> dict[str, Any]:
        """Run a real forward/backward but never call an optimizer."""
        self.student.train(True)
        self.student.base.vision_tower.eval()
        self.student.base.projector.eval()
        result = self.forward_anchor(
            sample, include_action=include_action, use_cached_pairs=use_cached_pairs
        )
        if mode == "feature":
            loss = result["feature"].total
        elif mode == "action":
            if result["action_loss"] is None:
                raise TemporalTrainingBlocked("action-only diagnostic requires action loss")
            loss = result["action_loss"]
        elif mode == "total":
            loss = result["total"]
        else:
            raise TemporalTrainingBlocked("unknown gradient diagnostic mode")
        loss.backward()
        report = self.student.gradient_report()
        return {
            "total": float(loss.detach()),
            "feature_mse": float(result["feature"].mse.detach()),
            "feature_cosine": float(result["feature"].cosine.detach()),
            "action_loss": None
            if result["action_loss"] is None
            else float(result["action_loss"].detach()),
            "valid_tokens": result["valid_tokens"],
            "valid_actions": result["valid_actions"],
            "gradient_report": report,
            "optimizer_updates": 0,
        }

    def run_accumulated_update(
        self,
        samples: Sequence[TemporalAnchorSample],
        *,
        optimizer: Any,
        scheduler: Any,
        include_action: bool = False,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        if include_action and self.arm != "D":
            raise TemporalTrainingBlocked("formal action objective is D-arm only")
        allowed, reason = self._authorized_for_update(include_action=include_action)
        if not allowed:
            raise TemporalTrainingBlocked(reason)
        if not samples:
            raise TemporalTrainingBlocked("at least one anchor is required")
        if len(samples) > self.config.effective_batch:
            raise TemporalTrainingBlocked("microbatch exceeds effective_batch")
        if len(samples) != self.config.effective_batch and not allow_partial:
            raise TemporalTrainingBlocked(
                f"effective_batch requires {self.config.effective_batch} anchors, "
                f"got {len(samples)}"
            )
        hard_anchors = min(
            80000,
            self.authority.max_anchors,
            self.config.max_anchor_presentations_per_arm,
        )
        if self.anchor_presentations + len(samples) > hard_anchors:
            raise TemporalTrainingBlocked("anchor budget would be exceeded")
        started = time.perf_counter()
        self.validate_optimizer_groups(optimizer)
        # Parameter-independent mask denominators: no stem/vision forward pass.
        token_counts = [self._count_feature_tokens(sample) for sample in samples]
        action_counts = [int(sample.target.future_mask.sum()) for sample in samples]
        if min(token_counts) <= 0 or (include_action and min(action_counts) <= 0):
            raise TemporalTrainingBlocked("invalid token/action denominator")
        optimizer.zero_grad(set_to_none=True)
        total_token = sum(token_counts)
        total_action = sum(action_counts)
        losses = []
        feature_mse = 0.0
        feature_cosine = 0.0
        feature_total = 0.0
        with self._activation_context():
            for sample, token_count, action_count in zip(
                samples, token_counts, action_counts, strict=True
            ):
                result = self.forward_anchor(sample, include_action=include_action)
                if int(result["valid_tokens"]) != token_count:
                    raise TemporalTrainingBlocked(
                        "actual feature mask differs from parameter-independent count"
                    )
                if include_action and int(result["valid_actions"]) != action_count:
                    raise TemporalTrainingBlocked("actual action mask differs from future mask")
                loss = self.config.lambda_f * result["feature"].total * token_count / total_token
                if include_action:
                    loss = loss + self.config.lambda_a_D * result["action_loss"] * (
                        action_count / total_action
                    )
                if not bool(torch.isfinite(loss)):
                    raise TemporalTrainingBlocked("non-finite temporal loss")
                weight = token_count / total_token
                feature_mse += float(result["feature"].mse.detach()) * weight
                feature_cosine += float(result["feature"].cosine.detach()) * weight
                feature_total += float(result["feature"].total.detach()) * weight
                loss.backward()
                losses.append(float(loss.detach()))
            trainable = [
                parameter for parameter in self.student.parameters() if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable, self.config.gradient_clip)
            optimizer.step()
            scheduler.step()
        self.cumulative_updates += 1
        self.anchor_presentations += len(samples)
        return {
            "status": "updated",
            "updates": self.cumulative_updates,
            "anchors": self.anchor_presentations,
            "losses": losses,
            "token_denominator": total_token,
            "action_denominator": total_action,
            "feature_mse": feature_mse,
            "feature_cosine": feature_cosine,
            "feature_total": feature_total,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
            "actual_updates": 1,
        }

    def set_selector_state(
        self,
        *,
        best_value: float,
        best_step: int,
        best_checkpoint_path: str,
        best_checkpoint_sha256: str | None,
        best_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        self.selector_state = {
            "best_value": float(best_value),
            "best_step": int(best_step),
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_checkpoint_sha256": (
                None if best_checkpoint_sha256 is None else str(best_checkpoint_sha256)
            ),
            "best_metrics": dict(best_metrics) if best_metrics is not None else None,
        }

    @staticmethod
    def capture_rng_state() -> dict[str, Any]:
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": {
                "kind": str(numpy_state[0]),
                "keys": torch.from_numpy(np.asarray(numpy_state[1], dtype=np.uint32).copy()),
                "pos": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def restore_rng_state(value: Mapping[str, Any]) -> None:
        if "python" in value:
            random.setstate(value["python"])
        if "numpy" in value:
            numpy_state = value["numpy"]
            keys = numpy_state["keys"]
            if not torch.is_tensor(keys):
                raise TemporalTrainingBlocked("numpy RNG keys are not a tensor")
            np.random.set_state(
                (
                    str(numpy_state["kind"]),
                    keys.detach().cpu().numpy().astype(np.uint32),
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        if "torch" in value:
            torch.set_rng_state(value["torch"])
        if value.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(value["cuda"]))

    def checkpoint_payload(
        self,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        sampler_state: Mapping[str, Any] | None = None,
        rng_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": "owac.raw-stem-temporal.checkpoint.v1",
            "model": self.student.state_dict(),
            "model_identity": self.student.model_identity,
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "sampler": dict(self.sampler_state if sampler_state is None else sampler_state),
            "selector": dict(self.selector_state),
            "run_elapsed_seconds": float(self.run_elapsed_seconds),
            "rng": dict(rng_state or self.capture_rng_state()),
            "arm": self.arm,
            "scope": self.scope,
            "run_identity": dict(self.run_identity),
            "training_config": asdict(self.config),
            "activation_memory_mode": self.config.activation_memory_mode,
            "cumulative_updates": self.cumulative_updates,
            "anchor_presentations": self.anchor_presentations,
            "config": {
                "effective_batch": self.config.effective_batch,
                "max_cumulative_updates_per_arm": self.config.max_cumulative_updates_per_arm,
                "max_anchor_presentations_per_arm": self.config.max_anchor_presentations_per_arm,
            },
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        sampler_state: Mapping[str, Any] | None = None,
        rng_state: Mapping[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_payload(
            optimizer=optimizer,
            scheduler=scheduler,
            sampler_state=sampler_state,
            rng_state=rng_state,
        )
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        torch.save(payload, temp)
        try:
            os.link(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("schema") != (
            "owac.raw-stem-temporal.checkpoint.v1"
        ):
            raise TemporalTrainingBlocked("temporal checkpoint schema differs")
        self.student.load_state_dict(payload["model"], strict=True)
        if self.student.model_identity != payload.get("model_identity"):
            raise TemporalTrainingBlocked("temporal checkpoint model identity differs")
        if str(payload.get("arm", self.arm)) != self.arm:
            raise TemporalTrainingBlocked("temporal checkpoint arm differs")
        if "scope" in payload and payload.get("scope") != self.scope:
            raise TemporalTrainingBlocked("temporal checkpoint scope differs")
        saved_identity = payload.get("run_identity")
        if isinstance(saved_identity, Mapping):
            if self.run_identity and dict(saved_identity) != self.run_identity:
                raise TemporalTrainingBlocked("temporal checkpoint run identity differs")
            if not self.run_identity:
                self.run_identity = dict(saved_identity)
        saved_training = payload.get("training_config")
        if isinstance(saved_training, Mapping):
            actual_training = asdict(self.config)
            saved_training = dict(saved_training)
            # Export may revoke execution permission; the numerical recipe must match.
            actual_training.pop("authorization", None)
            saved_training.pop("authorization", None)
            if json.dumps(saved_training, sort_keys=True) != json.dumps(
                actual_training, sort_keys=True
            ):
                raise TemporalTrainingBlocked("checkpoint full training recipe differs")
        elif self.run_identity:
            raise TemporalTrainingBlocked("identified checkpoint lacks full training recipe")
        saved_config = payload.get("config", {})
        for key in (
            "effective_batch",
            "max_cumulative_updates_per_arm",
            "max_anchor_presentations_per_arm",
        ):
            expected = getattr(self.config, key)
            if saved_config.get(key, expected) != expected:
                raise TemporalTrainingBlocked(f"temporal checkpoint config differs: {key}")
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        self.sampler_state = dict(payload.get("sampler", {}))
        self.run_elapsed_seconds = float(payload.get("run_elapsed_seconds", 0.0))
        stored_selector = payload.get("selector")
        if isinstance(stored_selector, Mapping):
            self.selector_state = dict(stored_selector)
        self.cumulative_updates = int(payload["cumulative_updates"])
        self.anchor_presentations = int(payload["anchor_presentations"])
        if isinstance(payload.get("rng"), Mapping):
            self.restore_rng_state(payload["rng"])
        return dict(payload)

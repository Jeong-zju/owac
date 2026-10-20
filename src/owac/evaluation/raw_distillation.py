"""Pure-student inference, offline metrics, visual controls and stage-0 parity.

The runtime here never instantiates a teacher, ISP or paired cache.  A caller
must supply an exported student, a loaded frozen backend, tokenizer/statistics
and a pure ``StudentSample``.  Missing assets raise unavailable instead of
falling back.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.baselines.raw_stem_siglip import HistoryContract, assert_backend_frozen
from owac.core.encoder_contract_n02 import VisualTokens
from owac.raw.distillation_pairs import NormalisationStats, StudentSample

__all__ = [
    "FrozenBackend",
    "HistorySupport",
    "MetricRecord",
    "OfflineMetrics",
    "OutcomeLedger",
    "ParityFailure",
    "ParityTrace",
    "PureStudentRuntime",
    "Stage0ParityRunner",
    "StudentInferenceUnavailable",
    "VisualControlError",
    "assert_history_supported",
    "denormalize_actions",
    "aggregate_metrics",
    "evaluate_windows",
    "load_labels_without_teacher",
    "register_history_support",
    "select_validation",
    "wilson_interval",
    "zero_visual",
    "shuffle_visual_sources",
    "target_position_intervention",
]


class StudentInferenceUnavailable(RuntimeError):
    """The pure student or one of its required frozen assets is unavailable."""


class VisualControlError(ValueError):
    pass


class ParityFailure(AssertionError):
    pass


@dataclass(frozen=True, slots=True)
class FrozenBackend:
    bridge: Any
    tokenizer: Any
    stats: NormalisationStats
    identity: Mapping[str, Any]
    tokenizer_sha256: str = ""
    stats_sha256: str = ""
    action_horizon: int = 10
    flow_steps: int = 10
    control_dim: int = 7
    padded_dim: int = 32

    def __post_init__(self) -> None:
        if not self.identity.get("sha256"):
            raise StudentInferenceUnavailable("frozen backend identity/hash is required")
        if self.action_horizon < 1 or self.flow_steps < 1:
            raise ValueError("positive action horizon/flow steps required")
        if self.control_dim < 1 or self.padded_dim < self.control_dim:
            raise ValueError("invalid control/padded dimensions")


@dataclass(frozen=True, slots=True)
class HistorySupport:
    name: str
    temporal: bool
    supported: bool
    reason: str = ""


_HISTORY_SUPPORT: dict[str, HistorySupport] = {
    "per_frame": HistorySupport("per_frame", False, True),
    "causal_burst": HistorySupport(
        "causal_burst", True, False, "no temporal student history encoder is registered"
    ),
    "recurrent_state": HistorySupport(
        "recurrent_state", True, False, "no recurrent student state module is registered"
    ),
}


def register_history_support(support: HistorySupport) -> None:
    _HISTORY_SUPPORT[support.name] = support


def assert_history_supported(history: HistoryContract, name: str = "per_frame") -> HistorySupport:
    support = _HISTORY_SUPPORT.get(name)
    if support is None or not support.supported:
        raise StudentInferenceUnavailable(
            f"student history mode {name!r} is unavailable: "
            f"{'unknown' if support is None else support.reason}"
        )
    if history.is_temporal != support.temporal:
        raise StudentInferenceUnavailable(
            f"history {history.temporal_frames} frames is incompatible with {name!r}"
        )
    return support


def _tokenizer_ids(tokenizer: Any, language: str, device: torch.device) -> torch.Tensor:
    """Compatibility helper for a tokenize-only tokenizer.

    Pure inference itself uses ``prepare_language_state`` below; this helper is
    retained for callers that inspect tokenization in isolation.
    """
    tokenize = getattr(tokenizer, "tokenize", None)
    if callable(tokenize):
        try:
            result = tokenize(language, None)
        except TypeError:
            result = tokenize(language)
        ids = result[0] if isinstance(result, tuple) else result
    else:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            ids = encode(language)
        elif callable(tokenizer):
            ids = tokenizer(language)
        else:
            raise StudentInferenceUnavailable("tokenizer lacks tokenize/encode")
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device, dtype=torch.long).reshape(1, -1)
    return torch.as_tensor(list(ids), dtype=torch.long, device=device).reshape(1, -1)


def _pad_state(
    state: torch.Tensor, max_state_dim: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    if state is None:
        raise StudentInferenceUnavailable("current proprio/state is unavailable; no zero fallback")
    state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    if state.shape[1] > max_state_dim:
        raise StudentInferenceUnavailable("state is wider than backend max_state_dim")
    padding = max_state_dim - int(state.shape[1])
    if padding:
        state = torch.cat([state, torch.zeros((1, padding), device=device)], dim=1)
    return state, padding


def denormalize_actions(actions: torch.Tensor, stats: NormalisationStats) -> torch.Tensor:
    """Inverse quantile normalisation for the real control dimensions."""
    if stats.action is None:
        raise StudentInferenceUnavailable("action statistics are unavailable")
    width = int(actions.shape[-1])
    q01 = np.asarray(stats.action["q01"], dtype=np.float32).ravel()[:width]
    q99 = np.asarray(stats.action["q99"], dtype=np.float32).ravel()[:width]
    if q01.size < width or q99.size < width:
        raise StudentInferenceUnavailable("statistics are narrower than the action vector")
    low = torch.as_tensor(q01, device=actions.device, dtype=torch.float32)
    high = torch.as_tensor(q99, device=actions.device, dtype=torch.float32)
    return (actions + 1.0) * 0.5 * (high - low + 1e-6) + low


@dataclass(frozen=True, slots=True)
class InferenceResult:
    normalized_actions: torch.Tensor
    physical_actions: torch.Tensor
    first5_physical: torch.Tensor
    state_padding: int
    flow_steps: int
    cache_key: str | None
    run_stats: Mapping[str, Any]


class PureStudentRuntime:
    """Exported student + exact frozen backend; no teacher/ISP/cache is reachable."""

    def __init__(
        self,
        *,
        student: torch.nn.Module,
        backend: FrozenBackend,
        history: HistoryContract | None = None,
    ) -> None:
        self.student = student
        self.backend = backend
        self.history = history or HistoryContract()
        self.history_support = assert_history_supported(self.history, "per_frame")
        if getattr(student, "bridge_projection", None) is not None:
            raise StudentInferenceUnavailable("student bridge projection must be None")
        if getattr(backend.bridge, "projection", None) is not None:
            raise StudentInferenceUnavailable("backend bridge projection must be None")
        assert_backend_frozen(backend.bridge.base)
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        self._last_cache_key = None

    def _language(
        self, sample: StudentSample, language: str | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from owac.backends.rawvla_language import prepare_language_state

        text = sample.language if language is None else language
        if sample.state is None:
            raise StudentInferenceUnavailable(
                "current proprio/state is unavailable for canonical tokenization"
            )
        if not hasattr(self.backend.bridge.base, "tokenizer"):
            raise StudentInferenceUnavailable(
                "frozen bridge.base.tokenizer is required for prepare_language_state"
            )
        return prepare_language_state(
            self.backend.bridge,
            [text],
            sample.state.detach().cpu().numpy().reshape(1, -1),
            device=self._device(),
        )

    def _device(self) -> torch.device:
        parameter = next(self.student.parameters(), None)
        return torch.device("cpu") if parameter is None else parameter.device

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        frontend_factory: Callable[[Any, str], tuple[Any, Any]] | None = None,
        backend: FrozenBackend,
        history: HistoryContract | None = None,
        expected_checkpoint_sha256: str | None = None,
    ) -> PureStudentRuntime:
        from owac.baselines.raw_stem_siglip import load_student_artifact

        student = load_student_artifact(
            path,
            frontend_factory=frontend_factory,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        metadata = getattr(student, "metadata", {})
        recorded_backend = metadata.get("backend", {}) or {}
        if recorded_backend.get("sha256") != backend.identity.get("sha256"):
            raise StudentInferenceUnavailable(
                "frozen backend identity differs from the exported student artifact"
            )
        for label, recorded in (
            ("tokenizer", metadata.get("tokenizer")),
            ("normalizer", metadata.get("normalizer")),
        ):
            if isinstance(recorded, Mapping) and recorded.get("sha256"):
                actual = backend.tokenizer_sha256 if label == "tokenizer" else backend.stats_sha256
                if actual != recorded["sha256"]:
                    raise StudentInferenceUnavailable(
                        f"frozen {label} identity differs from the exported student artifact"
                    )
        return cls(student=student, backend=backend, history=history)

    def _anchor_raw(self, samples: Sequence[StudentSample]):
        from owac.baselines.raw_stem_siglip import RawStemInput

        if not samples:
            raise StudentInferenceUnavailable("camera anchor is empty")
        prepared = [sample.raw.as_stem_input(camera_id=sample.camera_id) for sample in samples]
        base = prepared[0]
        for item in prepared[1:]:
            if (
                item.input_mode,
                item.units,
                item.cfa_pattern,
                item.black_level,
                item.white_level,
            ) != (
                base.input_mode,
                base.units,
                base.cfa_pattern,
                base.black_level,
                base.white_level,
            ):
                raise StudentInferenceUnavailable("camera anchor metadata differs")
        return RawStemInput(
            values=torch.cat([item.values for item in prepared], dim=1),
            input_mode=base.input_mode,
            camera_ids=tuple(sample.camera_id for sample in samples),
            units=base.units,
            valid_mask=(
                None
                if any(item.valid_mask is None for item in prepared)
                else torch.cat([item.valid_mask for item in prepared], dim=1)
            ),
            bad_mask=(
                None
                if any(item.bad_mask is None for item in prepared)
                else torch.cat([item.bad_mask for item in prepared], dim=1)
            ),
            saturated_mask=(
                None
                if any(item.saturated_mask is None for item in prepared)
                else torch.cat([item.saturated_mask for item in prepared], dim=1)
            ),
            missing_mask=(
                None
                if any(item.missing_mask is None for item in prepared)
                else torch.cat([item.missing_mask for item in prepared], dim=1)
            ),
            black_level=base.black_level,
            white_level=base.white_level,
            exposure=base.exposure,
            noise=base.noise,
            cfa_pattern=base.cfa_pattern,
            physical_origins=tuple(sample.raw.physical_origin for sample in samples),
        )

    def infer_anchor(
        self,
        samples: Sequence[StudentSample],
        *,
        language: str | None = None,
        noise: torch.Tensor | None = None,
        flow_steps: int | None = None,
    ) -> InferenceResult:
        """One control anchor: all declared cameras are one input window."""
        samples = tuple(samples)
        if not samples:
            raise StudentInferenceUnavailable("camera anchor is empty")
        first = samples[0]
        if first.variant == "rgb3_control":
            raise StudentInferenceUnavailable("RGB3 control is not a pure student input")
        for sample in samples[1:]:
            if (sample.source_id, sample.frame_id, sample.timestamp) != (
                first.source_id,
                first.frame_id,
                first.timestamp,
            ):
                raise StudentInferenceUnavailable("camera anchor frames are not aligned")
            if (
                sample.state is None
                or first.state is None
                or not torch.equal(sample.state, first.state)
            ):
                raise StudentInferenceUnavailable("camera anchor states differ")
        if first.state is None:
            raise StudentInferenceUnavailable("state is unavailable")
        raw = self._anchor_raw(samples)
        lang_tokens, lang_masks, state = self._language(first, language)
        padding = int(state.shape[1] - first.state.numel())
        steps = self.backend.flow_steps if flow_steps is None else int(flow_steps)
        if noise is not None:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self._device())
        with torch.inference_mode():
            tokens = self.student(raw)
            prediction = self.backend.bridge.predict_action_from_visual_tokens(
                tokens.tokens,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                num_steps=steps,
            )
        normalized = torch.as_tensor(
            prediction["normalized_actions"], dtype=torch.float32, device=self._device()
        )
        physical = denormalize_actions(
            normalized[..., : self.backend.control_dim], self.backend.stats
        )
        self._last_cache_key = first.frame_key
        return InferenceResult(
            normalized_actions=normalized,
            physical_actions=physical,
            first5_physical=physical[:, :5],
            state_padding=padding,
            flow_steps=steps,
            cache_key=None,
            run_stats=prediction.get("run_stats", {}),
        )

    def infer(
        self,
        sample: StudentSample,
        *,
        language: str | None = None,
        noise: torch.Tensor | None = None,
        flow_steps: int | None = None,
    ) -> InferenceResult:
        return self.infer_anchor((sample,), language=language, noise=noise, flow_steps=flow_steps)

    def reset(self) -> dict[str, Any]:
        self._last_cache_key = None
        reset = getattr(self.backend.bridge, "reset_episodes", None)
        if callable(reset):
            return reset()
        return {"reset": "no persistent student state"}

    def dependency_audit(self) -> dict[str, Any]:
        forbidden = []
        for name, _value in vars(self).items():
            if name in {"student", "backend"}:
                continue
            if name.lower() in {"teacher", "isp", "cache"}:
                forbidden.append(name)
        return {"forbidden_dependencies": forbidden, "pure": not forbidden}


@dataclass(frozen=True, slots=True)
class MetricRecord:
    split: str
    step: int
    valid_tokens: int
    step1_mae7: float | None
    first5_mae7: float | None
    full10_mae7: float | None
    groups: Mapping[str, float]
    padding_mae: float | None = None
    valid_step1: int = 0
    valid_first5: int = 0
    valid_full10: int = 0
    normalized: bool = False


class OfflineMetrics:
    def __init__(self, *, control_dim: int = 7, groups: Mapping[str, Sequence[int]] | None = None):
        self.control_dim = int(control_dim)
        self.groups = dict(
            groups
            or {
                "translation": (0, 1, 2),
                "rotation": (3, 4, 5),
                "gripper": (6,),
            }
        )

    def window(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> MetricRecord:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction/target must share [B,H,D]")
        if int(prediction.shape[1]) != 10:
            raise ValueError("LIBERO metrics require the full 10-step horizon")
        if int(prediction.shape[-1]) < self.control_dim:
            raise ValueError(
                f"metrics require at least {self.control_dim} real dims, got {prediction.shape[-1]}"
            )
        mask = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
        if tuple(mask.shape) != tuple(prediction.shape[:2]):
            raise ValueError("mask must be [B,H]")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("metrics require finite predictions/targets")
        width = self.control_dim
        error = (prediction[..., :width] - target[..., :width]).abs().mean(dim=-1)
        counts = {
            "step1": int(mask[:, :1].sum()),
            "first5": int(mask[:, :5].sum()),
            "full10": int(mask[:, :10].sum()),
        }
        if counts["full10"] <= 0:
            raise ValueError("metrics require at least one valid full-horizon row")
        values = {
            name: None if count == 0 else error[:, :slice_end][mask[:, :slice_end]].mean().item()
            for name, count, slice_end in (
                ("step1", counts["step1"], 1),
                ("first5", counts["first5"], 5),
                ("full10", counts["full10"], 10),
            )
        }
        groups = {}
        for name, dims in self.groups.items():
            selected = list(dims)
            if max(selected) >= width:
                raise ValueError(f"group {name} exceeds real control width {width}")
            difference = (prediction[..., selected] - target[..., selected]).abs()
            groups[name] = difference[:, :10][mask[:, :10]].mean().item()
        padding = None
        if prediction.shape[-1] > width:
            difference = (prediction[..., width:] - target[..., width:]).abs()
            padding = difference[:, :10][mask[:, :10]].mean().item()
        return MetricRecord(
            split="",
            step=0,
            valid_tokens=int(mask.sum()),
            step1_mae7=values["step1"],
            first5_mae7=values["first5"],
            full10_mae7=values["full10"],
            groups=groups,
            padding_mae=padding,
            valid_step1=counts["step1"],
            valid_first5=counts["first5"],
            valid_full10=counts["full10"],
        )


def evaluate_windows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    split: str,
    step: int,
    control_dim: int = 7,
    normalized: bool = False,
) -> MetricRecord:
    record = OfflineMetrics(control_dim=control_dim).window(prediction, target, mask)
    return MetricRecord(
        split=split,
        step=int(step),
        valid_tokens=record.valid_tokens,
        step1_mae7=record.step1_mae7,
        first5_mae7=record.first5_mae7,
        full10_mae7=record.full10_mae7,
        groups=record.groups,
        padding_mae=record.padding_mae,
        valid_step1=record.valid_step1,
        valid_first5=record.valid_first5,
        valid_full10=record.valid_full10,
        normalized=normalized,
    )


def _weighted(values: Sequence[float | None], weights: Sequence[int]) -> float | None:
    pairs = [
        (float(value), int(weight))
        for value, weight in zip(values, weights, strict=True)
        if value is not None and weight > 0 and math.isfinite(float(value))
    ]
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / total


def aggregate_metrics(records: Sequence[MetricRecord], *, split: str, step: int) -> MetricRecord:
    """Denominator-weighted aggregate over every observed window.

    A window with no valid step1 entries contributes to step1 with zero weight but
    can still contribute to first5/full10.  Missing slices are reported as None,
    never silently replaced or dropped.
    """
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"no {split} records to aggregate")
    if any(record.step != step for record in selected):
        raise ValueError("aggregate requires one checkpoint step")
    step1 = _weighted(
        [record.step1_mae7 for record in selected], [record.valid_step1 for record in selected]
    )
    first5 = _weighted(
        [record.first5_mae7 for record in selected], [record.valid_first5 for record in selected]
    )
    full10 = _weighted(
        [record.full10_mae7 for record in selected], [record.valid_full10 for record in selected]
    )
    padding = _weighted(
        [record.padding_mae for record in selected], [record.valid_full10 for record in selected]
    )
    group_names = sorted({name for record in selected for name in record.groups})
    groups = {
        name: _weighted(
            [record.groups.get(name) for record in selected],
            [record.valid_full10 for record in selected],
        )
        for name in group_names
    }
    return MetricRecord(
        split=split,
        step=int(step),
        valid_tokens=sum(record.valid_tokens for record in selected),
        step1_mae7=step1,
        first5_mae7=first5,
        full10_mae7=full10,
        groups=groups,
        padding_mae=padding,
        valid_step1=sum(record.valid_step1 for record in selected),
        valid_first5=sum(record.valid_first5 for record in selected),
        valid_full10=sum(record.valid_full10 for record in selected),
        normalized=any(record.normalized for record in selected),
    )


def select_validation(records: Sequence[MetricRecord]) -> MetricRecord:
    if any(record.split == "test" for record in records):
        raise ValueError("test records are never selectable")
    records = [record for record in records if record.split == "validation"]
    if not records:
        raise ValueError("validation selection requires validation records")
    for record in records:
        if (
            record.first5_mae7 is None
            or not math.isfinite(float(record.first5_mae7))
            or record.valid_step1 <= 0
            or record.valid_first5 <= 0
            or record.valid_full10 <= 0
        ):
            raise ValueError("validation selection rejects NaN/invalid denominators")
    return min(records, key=lambda record: (float(record.first5_mae7), record.step))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial denominator")
    p = successes / total
    denom = 1 + z * z / total
    middle = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return ((middle - spread) / denom, (middle + spread) / denom)


@dataclass(slots=True)
class OutcomeLedger:
    planned: int = 0
    successes: int = 0
    failures: int = 0
    missing: int = 0
    regressions: int = 0
    recoveries: int = 0
    videos: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    config_sha256: str | None = None
    seed: int | None = None
    initial_state_ids: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    first_deviations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.successes + self.failures

    @property
    def total(self) -> int:
        return self.attempted

    @property
    def success_rate(self) -> float | None:
        return None if self.attempted == 0 else self.successes / self.attempted

    @property
    def fully_observed(self) -> bool:
        return self.missing == 0 and self.planned == self.attempted

    def add(
        self,
        *,
        previous_success: bool | None,
        current_success: bool | None,
        video: str | None = None,
        run_id: str | None = None,
        initial_state_id: str | None = None,
        first_deviation: Mapping[str, Any] | None = None,
    ) -> None:
        if run_id is not None:
            if run_id in self.run_ids:
                raise ValueError(f"duplicate run identity {run_id!r}")
            self.run_ids.append(run_id)
        if initial_state_id is not None:
            if initial_state_id in self.initial_state_ids:
                raise ValueError(f"duplicate initial-state identity {initial_state_id!r}")
            self.initial_state_ids.append(initial_state_id)
        self.planned += 1
        if current_success is None:
            self.missing += 1
        elif current_success:
            self.successes += 1
        else:
            self.failures += 1
        if previous_success is False and current_success is True:
            self.recoveries += 1
        if previous_success is True and current_success is False:
            self.regressions += 1
        if video is not None:
            self.videos.append(str(video))
        if first_deviation is not None:
            self.first_deviations.append(dict(first_deviation))

    def report(self) -> dict[str, Any]:
        low, high = (
            (None, None) if self.attempted == 0 else wilson_interval(self.successes, self.attempted)
        )
        return {
            "planned": self.planned,
            "attempted": self.attempted,
            "completed": self.attempted,
            "successes": self.successes,
            "failures": self.failures,
            "missing": self.missing,
            "total": self.attempted,
            "success_rate": self.success_rate,
            "conditional_completed_rate": self.success_rate,
            "wilson_low": low,
            "wilson_high": high,
            "regressions": self.regressions,
            "recoveries": self.recoveries,
            "videos": list(self.videos),
            "checkpoint": self.checkpoint,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "initial_state_ids": list(self.initial_state_ids),
            "run_ids": list(self.run_ids),
            "first_deviations": list(self.first_deviations),
            "complete_denominator": self.planned if self.planned else self.attempted,
            "fully_observed": self.fully_observed,
            "full_screen_claim": self.fully_observed,
        }


def zero_visual(tokens: VisualTokens, *, mask_visual: bool = False) -> VisualTokens:
    """Zero visual values while preserving prefix layout by default."""
    values = torch.zeros_like(tokens.values)
    mask = tokens.mask
    if mask_visual and mask is not None:
        mask = torch.zeros_like(mask)
    return VisualTokens(
        values=values,
        mask=mask,
        camera_ids=tokens.camera_ids,
        positions=tokens.positions,
        scale=tokens.scale,
        validity=tokens.validity,
    )


def shuffle_visual_sources(
    tokens: VisualTokens,
    permutation: Sequence[int],
    *,
    source_ids: Sequence[str] | None = None,
) -> VisualTokens:
    order = torch.as_tensor(list(permutation), dtype=torch.long, device=tokens.values.device)
    if order.numel() != tokens.values.shape[0]:
        raise VisualControlError("source permutation must cover the batch")
    if source_ids is None or len(source_ids) != tokens.values.shape[0]:
        raise VisualControlError("source ids are required for cross-source shuffle")
    if torch.equal(order, torch.arange(order.numel(), device=order.device)):
        raise VisualControlError("identity shuffle is not a cross-source control")
    for target, donor in enumerate(order.tolist()):
        if source_ids[target] == source_ids[donor]:
            raise VisualControlError("cross-source shuffle requires different source ids")
    return VisualTokens(
        values=tokens.values.index_select(0, order),
        mask=None if tokens.mask is None else tokens.mask.index_select(0, order),
        camera_ids=tokens.camera_ids,
        positions=None if tokens.positions is None else tokens.positions.index_select(0, order),
        scale=tokens.scale,
        validity=tokens.validity,
    )


def target_position_intervention(
    tokens: VisualTokens,
    *,
    target_mask: torch.Tensor,
    changed_actions: torch.Tensor | None,
    label_provenance: Mapping[str, Any] | str | None = None,
) -> tuple[VisualTokens, torch.Tensor]:
    if changed_actions is None:
        raise VisualControlError(
            "target intervention requires new action labels; old action labels are invalid"
        )
    if not label_provenance:
        raise VisualControlError(
            "target intervention requires label provenance for the changed actions"
        )
    if tokens.positions is None:
        raise VisualControlError("target intervention needs physical token positions")
    mask = torch.as_tensor(target_mask, dtype=torch.bool)
    if tuple(mask.shape) != tuple(tokens.values.shape[:2]):
        raise VisualControlError("target mask must be [B,N]")
    positions = tokens.positions.clone()
    shift = torch.zeros_like(positions)
    shift[..., 0] = 1.0
    positions = positions + shift * mask.unsqueeze(-1).to(positions.dtype)
    changed = VisualTokens(
        values=tokens.values,
        mask=tokens.mask,
        camera_ids=tokens.camera_ids,
        positions=positions,
        scale=tokens.scale,
        validity=tokens.validity,
    )
    return changed, changed_actions


# --------------------------------------------------------------------------- #
# Stage 0 parity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ParityTolerance:
    rtol: float = 1e-5
    atol: float = 1e-6

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParityTrace:
    name: str
    tensors: Mapping[str, torch.Tensor]
    history: HistoryContract
    reset: bool
    provenance: Mapping[str, Any]

    def digest(self) -> str:
        digest = hashlib.sha256()
        for name in sorted(self.tensors):
            tensor = self.tensors[name].detach().cpu().float().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ParityReport:
    passed: bool
    native_repeat_digest: str
    tolerance: ParityTolerance
    first_mismatch: str | None
    max_errors: Mapping[str, float]
    native_provenance: Mapping[str, Any]
    bridge_provenance: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "native_repeat_digest": self.native_repeat_digest,
            "tolerance": self.tolerance.as_dict(),
            "first_mismatch": self.first_mismatch,
            "max_errors": dict(self.max_errors),
            "native_provenance": dict(self.native_provenance),
            "bridge_provenance": dict(self.bridge_provenance),
        }


class Stage0ParityRunner:
    """Compare repeated native traces first, then a distinct bridge trace.

    ``native_call`` must be the real policy path (not a prebuilt bridge trace).
    ``bridge_call`` must be the projected teacher-token bridge.  The tolerance is
    frozen on the native repeat before the bridge callable is invoked.
    """

    def __init__(
        self,
        *,
        native_call: Callable[[Any], ParityTrace],
        bridge_call: Callable[[Any], ParityTrace],
        tolerance: ParityTolerance | None = None,
        fixed_seed: int = 101,
    ) -> None:
        if native_call is bridge_call:
            raise ValueError("native and bridge traces must be distinct callables")
        self.native_call = native_call
        self.bridge_call = bridge_call
        self.tolerance = tolerance or ParityTolerance()
        self.fixed_seed = int(fixed_seed)

    def run(self, batch: Any) -> ParityReport:
        torch.manual_seed(self.fixed_seed)
        np.random.seed(self.fixed_seed)
        first = self.native_call(batch)
        torch.manual_seed(self.fixed_seed)
        np.random.seed(self.fixed_seed)
        second = self.native_call(batch)
        if first.digest() != second.digest():
            raise ParityFailure("native policy is not repeatable at the fixed boundary")
        tolerance = self.tolerance  # frozen before bridge invocation
        bridge = self.bridge_call(copy.deepcopy(batch))
        expected = set(first.tensors)
        observed = set(bridge.tensors)
        if not expected & observed:
            raise ParityFailure("native and bridge traces share no tensor names")
        first_bad = None
        errors = {}
        for name in sorted(expected - observed):
            first_bad = first_bad or f"{name}:missing_bridge"
            errors[name] = float("inf")
        for name in sorted(observed - expected):
            first_bad = first_bad or f"{name}:unexpected_bridge"
            errors[name] = float("inf")
        for name in sorted(expected & observed):
            a = first.tensors[name].float()
            b = bridge.tensors[name].float()
            if a.shape != b.shape:
                first_bad = first_bad or f"{name}:shape"
                errors[name] = float("inf")
                continue
            error = float((a - b).abs().max())
            errors[name] = error
            close = torch.allclose(a, b, rtol=tolerance.rtol, atol=tolerance.atol)
            if first_bad is None and not close:
                first_bad = name
        return ParityReport(
            passed=first_bad is None,
            native_repeat_digest=first.digest(),
            tolerance=tolerance,
            first_mismatch=first_bad,
            max_errors=errors,
            native_provenance=dict(first.provenance),
            bridge_provenance=dict(bridge.provenance),
        )


def load_labels_without_teacher(dataset: Any, frame_key: str) -> dict[str, Any]:
    """Load expert actions/future mask without touching teacher RGB or cache.

    The paired NPZ may contain teacher arrays; only the label keys are selected.
    """
    from owac.raw.distillation_pairs import _npz, normalise_actions

    frame = dataset._frames[frame_key]
    data = _npz(
        frame.array_path,
        expected_sha=frame.array_sha256,
        keys=("actions_raw", "future_mask", "state"),
    )
    if "actions_raw" not in data or "future_mask" not in data:
        raise ValueError(f"{frame_key}: labels are absent")
    state = None
    if frame.state_dim is not None:
        if "state" not in data:
            raise ValueError(f"{frame_key}: declared state is absent")
        state = np.asarray(data["state"], dtype=np.float32)
    actions, _state_meta, _normal_meta = normalise_actions(
        np.asarray(data["actions_raw"], dtype=np.float32),
        state,
        dataset.manifest.normalization.stats,
        dataset_format=dataset.manifest.normalization.dataset_format,
        delta_action_indices=dataset.manifest.normalization.delta_action_indices,
    )
    future = np.asarray(data["future_mask"])
    if future.shape != (dataset.manifest.action_horizon,) or future.dtype != np.bool_:
        raise ValueError(f"{frame_key}: future mask shape/dtype differs")
    return {
        "actions": torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32)).clone(),
        "future_mask": torch.from_numpy(future.astype(np.bool_, copy=True)).clone(),
    }

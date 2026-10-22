"""Held-out projected-token validation and feature-based checkpoint selection.

The objective is the actual post-projection visual tokens, not RGB, actions or a
normalized proxy.  Both student and teacher tensors are reduced in FP32 with the
same masked reduction as training:

    MSE     = sum(mask * (student - teacher)^2) / (valid_tokens * width)
    cosine  = sum(mask * (1 - cos(student, teacher))) / valid_tokens
    total   = MSE + cosine_weight * cosine

No per-token or per-sample magnitude normalization is applied.  Selection is the
minimum ``total``; ties resolve to the earlier optimizer step.  Action metrics are
stored only as an explicitly separate diagnostic and can never select, stop or
switch a checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "FeatureSelectionError",
    "FeatureSelector",
    "FeatureSourceSummary",
    "FeatureValidationRecord",
    "FeatureWindowRecord",
    "TokenFeatureAccumulator",
    "evaluate_feature_validation",
    "feature_total",
    "masked_feature_components",
    "select_feature_candidate",
    "validation_endpoints",
]


class FeatureSelectionError(RuntimeError):
    """A held-out feature record or selection transition is not admissible."""


def feature_total(mse: float, cosine: float, *, cosine_weight: float = 0.1) -> float:
    """The exact scalar objective, exposed for tests and sidecar serialization."""
    if not math.isfinite(float(mse)) or not math.isfinite(float(cosine)):
        raise FeatureSelectionError("feature components must be finite")
    if not math.isfinite(float(cosine_weight)) or float(cosine_weight) < 0.0:
        raise FeatureSelectionError("cosine_weight must be finite and non-negative")
    return float(mse) + float(cosine_weight) * float(cosine)


def masked_feature_components(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    cosine_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return (mse, cosine, total, valid_tokens, width) in FP32.

    ``student``/``teacher`` share ``[B, N, D]``; ``mask`` is ``[B, N]``.  A mask is
    required for held-out validation because inactive cameras and padded endpoints
    must not contribute.  The denominator is the number of valid tokens for cosine
    and ``valid_tokens * width`` for MSE.
    """
    if not torch.is_tensor(student) or not torch.is_tensor(teacher):
        raise FeatureSelectionError("student/teacher tokens must be tensors")
    if student.shape != teacher.shape or student.ndim != 3:
        raise FeatureSelectionError("student/teacher tokens must share [B,N,D]")
    if student.numel() == 0:
        raise FeatureSelectionError("token tensors are empty")
    if not student.is_floating_point() or not teacher.is_floating_point():
        student = student.float()
        teacher = teacher.float()
    student = student.detach().float()
    teacher = teacher.detach().float().to(student.device)
    bsize, length, width = student.shape
    if mask is None:
        mask_tensor = torch.ones((bsize, length), dtype=torch.bool, device=student.device)
    else:
        if not torch.is_tensor(mask):
            raise FeatureSelectionError("token mask must be a tensor")
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=student.device)
        if tuple(mask_tensor.shape) != (bsize, length):
            raise FeatureSelectionError("token mask must be [B,N]")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise FeatureSelectionError("token tensors contain non-finite values")
    count = int(mask_tensor.sum())
    if count <= 0:
        raise FeatureSelectionError("token reduction requires at least one valid token")
    if width <= 0:
        raise FeatureSelectionError("token width must be positive")
    difference = student - teacher
    mse = (difference.square() * mask_tensor.unsqueeze(-1)).sum() / (count * width)
    cosine = F.cosine_similarity(student, teacher, dim=-1)
    cosine_loss = ((1.0 - cosine) * mask_tensor).sum() / count
    total = mse + float(cosine_weight) * cosine_loss
    if not bool(torch.isfinite(mse)) or not bool(torch.isfinite(cosine_loss)):
        raise FeatureSelectionError("non-finite feature reduction")
    return mse, cosine_loss, total, count, int(width)


def validation_endpoints(num_frames: int) -> tuple[int, ...]:
    """The fixed causal validation endpoints for one source (0/5/... < frames)."""
    frames = int(num_frames)
    if frames < 1:
        raise FeatureSelectionError("source num_frames must be positive")
    return tuple(range(0, frames, 5))


@dataclass(frozen=True, slots=True)
class FeatureWindowRecord:
    source_id: str
    endpoint: int
    valid_tokens: int
    width: int
    scalar_denominator: float
    feature_mse: float
    feature_cosine: float
    feature_total: float
    masked_tokens: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "endpoint": int(self.endpoint),
            "valid_tokens": int(self.valid_tokens),
            "width": int(self.width),
            "scalar_denominator": float(self.scalar_denominator),
            "feature_mse": float(self.feature_mse),
            "feature_cosine": float(self.feature_cosine),
            "feature_total": float(self.feature_total),
            "masked_tokens": int(self.masked_tokens),
        }


@dataclass(frozen=True, slots=True)
class FeatureSourceSummary:
    source_id: str
    windows: int
    valid_tokens: int
    scalar_denominator: float
    feature_mse: float
    feature_cosine: float
    feature_total: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "windows": int(self.windows),
            "valid_tokens": int(self.valid_tokens),
            "scalar_denominator": float(self.scalar_denominator),
            "feature_mse": float(self.feature_mse),
            "feature_cosine": float(self.feature_cosine),
            "feature_total": float(self.feature_total),
        }


@dataclass(frozen=True, slots=True)
class FeatureValidationRecord:
    split: str
    sources: int
    trajectories: int
    windows: int
    episode_resets: int
    valid_tokens: int
    scalar_denominator: float
    feature_mse: float
    feature_cosine: float
    feature_total: float
    cosine_weight: float
    per_source: Mapping[str, FeatureSourceSummary]
    source_order: tuple[str, ...] = field(default_factory=tuple)
    test_used: bool = False
    selection_eligible: bool = True

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test"}:
            raise FeatureSelectionError("feature record split must be validation/test")
        if self.split == "test" and self.selection_eligible:
            raise FeatureSelectionError("test split is never selection-eligible")
        if self.test_used:
            raise FeatureSelectionError("test split must never be consumed for selection")
        if self.sources < 1 or self.windows < 1:
            raise FeatureSelectionError("feature record needs at least one source/window")
        if self.episode_resets != self.sources:
            raise FeatureSelectionError("each validation source must reset episode history once")
        if self.valid_tokens < 1 or self.scalar_denominator <= 0.0:
            raise FeatureSelectionError("feature record denominator is empty")
        for value in (self.feature_mse, self.feature_cosine, self.feature_total):
            if not math.isfinite(float(value)):
                raise FeatureSelectionError("feature record contains non-finite metrics")
        expected = feature_total(
            self.feature_mse, self.feature_cosine, cosine_weight=self.cosine_weight
        )
        if not math.isclose(expected, float(self.feature_total), rel_tol=0.0, abs_tol=1e-12):
            raise FeatureSelectionError("feature record total does not equal MSE+w*cosine")

    @property
    def selection_value(self) -> float:
        return float(self.feature_total)

    def to_payload(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "sources": int(self.sources),
            "trajectories": int(self.trajectories),
            "windows": int(self.windows),
            "anchors": int(self.windows),
            "episode_resets": int(self.episode_resets),
            "valid_tokens": int(self.valid_tokens),
            "scalar_denominator": float(self.scalar_denominator),
            "feature_mse": float(self.feature_mse),
            "feature_cosine": float(self.feature_cosine),
            "feature_total": float(self.feature_total),
            "cosine_weight": float(self.cosine_weight),
            "per_source": {name: summary.to_payload() for name, summary in self.per_source.items()},
            "source_order": list(self.source_order),
            "test_used": bool(self.test_used),
            "selection_eligible": bool(self.selection_eligible),
        }


class TokenFeatureAccumulator:
    """Weighted held-out token accumulation with explicit episode-reset checks.

    A source must begin at endpoint 0 with no carried student state; endpoints must
    strictly increase.  Starting a second source without finishing the first is a
    programming error, which is exactly the cross-episode leakage this guard blocks.
    """

    def __init__(
        self,
        *,
        cosine_weight: float = 0.1,
        split: str = "validation",
        expected_sources: int | None = None,
        expected_windows: int | None = None,
    ) -> None:
        if not math.isfinite(float(cosine_weight)) or float(cosine_weight) < 0.0:
            raise FeatureSelectionError("cosine_weight must be finite and non-negative")
        if split not in {"validation", "test"}:
            raise FeatureSelectionError("accumulator split must be validation/test")
        self.cosine_weight = float(cosine_weight)
        self.split = split
        self.expected_sources = expected_sources
        self.expected_windows = expected_windows
        self._source_order: list[str] = []
        self._per_source: dict[str, dict[str, float]] = {}
        self._current: str | None = None
        self._last_endpoint: int | None = None
        self._windows = 0
        self._episode_resets = 0
        self._valid_tokens = 0
        self._scalar_denominator = 0.0
        self._weighted_mse = 0.0
        self._weighted_cosine = 0.0
        self._closed: FeatureValidationRecord | None = None

    def begin_source(self, source_id: str) -> None:
        if self._closed is not None:
            raise FeatureSelectionError("accumulator is already finalized")
        if not source_id:
            raise FeatureSelectionError("source id must be non-empty")
        if self._current is not None:
            raise FeatureSelectionError("cannot begin a source before finishing the current one")
        if source_id in self._per_source:
            raise FeatureSelectionError("source appears twice in one validation pass")
        self._current = str(source_id)
        self._last_endpoint = None
        self._source_order.append(str(source_id))
        self._per_source[str(source_id)] = {
            "windows": 0,
            "valid_tokens": 0,
            "scalar_denominator": 0.0,
            "weighted_mse": 0.0,
            "weighted_cosine": 0.0,
        }
        # A source begins with no inherited state; count exactly one reset.
        self._episode_resets += 1

    def add_window(
        self,
        source_id: str,
        endpoint: int,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> FeatureWindowRecord:
        if self._closed is not None:
            raise FeatureSelectionError("accumulator is already finalized")
        if self._current is None or str(source_id) != self._current:
            raise FeatureSelectionError("window source does not match the open source")
        endpoint = int(endpoint)
        if endpoint < 0 or endpoint % 5:
            raise FeatureSelectionError("validation endpoints must be non-negative multiples of 5")
        if self._last_endpoint is None:
            if endpoint != 0:
                raise FeatureSelectionError("first window of a source must be endpoint 0")
        elif endpoint <= self._last_endpoint:
            raise FeatureSelectionError(
                "validation endpoints must strictly increase within a source"
            )
        mse, cosine, total, valid, width = masked_feature_components(
            student_tokens,
            teacher_tokens,
            mask,
            cosine_weight=self.cosine_weight,
        )
        scalar = float(valid * width)
        bucket = self._per_source[self._current]
        bucket["windows"] += 1
        bucket["valid_tokens"] += valid
        bucket["scalar_denominator"] += scalar
        bucket["weighted_mse"] += float(mse) * scalar
        bucket["weighted_cosine"] += float(cosine) * valid
        self._windows += 1
        self._valid_tokens += valid
        self._scalar_denominator += scalar
        self._weighted_mse += float(mse) * scalar
        self._weighted_cosine += float(cosine) * valid
        self._last_endpoint = endpoint
        masked_tokens = int(sum(int(value) for value in mask.shape[:-1])) * int(mask.shape[-1])
        record = FeatureWindowRecord(
            source_id=str(source_id),
            endpoint=endpoint,
            valid_tokens=valid,
            width=width,
            scalar_denominator=scalar,
            feature_mse=float(mse),
            feature_cosine=float(cosine),
            feature_total=float(total),
            masked_tokens=masked_tokens,
        )
        return record

    def end_source(self) -> FeatureSourceSummary:
        if self._current is None:
            raise FeatureSelectionError("no open source to end")
        bucket = self._per_source[self._current]
        if int(bucket["windows"]) < 1 or float(bucket["scalar_denominator"]) <= 0.0:
            raise FeatureSelectionError("a source must contribute at least one non-empty window")
        mse = float(bucket["weighted_mse"]) / float(bucket["scalar_denominator"])
        cosine = float(bucket["weighted_cosine"]) / float(bucket["valid_tokens"])
        summary = FeatureSourceSummary(
            source_id=str(self._current),
            windows=int(bucket["windows"]),
            valid_tokens=int(bucket["valid_tokens"]),
            scalar_denominator=float(bucket["scalar_denominator"]),
            feature_mse=mse,
            feature_cosine=cosine,
            feature_total=feature_total(mse, cosine, cosine_weight=self.cosine_weight),
        )
        self._current = None
        self._last_endpoint = None
        return summary

    def finish(self) -> FeatureValidationRecord:
        if self._closed is not None:
            return self._closed
        if self._current is not None:
            self.end_source()
        if not self._source_order:
            raise FeatureSelectionError("no validation sources were accumulated")
        if self.expected_sources is not None and len(self._source_order) != int(
            self.expected_sources
        ):
            raise FeatureSelectionError(
                "expected "
                f"{self.expected_sources} validation sources, got {len(self._source_order)}"
            )
        if self.expected_windows is not None and self._windows != int(self.expected_windows):
            raise FeatureSelectionError(
                f"expected {self.expected_windows} validation windows, got {self._windows}"
            )
        if self._scalar_denominator <= 0.0 or self._valid_tokens <= 0:
            raise FeatureSelectionError("global feature denominator is empty")
        summaries: dict[str, FeatureSourceSummary] = {}
        for source_id in self._source_order:
            bucket = self._per_source[source_id]
            mse = float(bucket["weighted_mse"]) / float(bucket["scalar_denominator"])
            cosine = float(bucket["weighted_cosine"]) / float(bucket["valid_tokens"])
            summaries[source_id] = FeatureSourceSummary(
                source_id=source_id,
                windows=int(bucket["windows"]),
                valid_tokens=int(bucket["valid_tokens"]),
                scalar_denominator=float(bucket["scalar_denominator"]),
                feature_mse=mse,
                feature_cosine=cosine,
                feature_total=feature_total(mse, cosine, cosine_weight=self.cosine_weight),
            )
        mse = self._weighted_mse / self._scalar_denominator
        cosine = self._weighted_cosine / self._valid_tokens
        record = FeatureValidationRecord(
            split=self.split,
            sources=len(self._source_order),
            trajectories=len(self._source_order),
            windows=self._windows,
            episode_resets=self._episode_resets,
            valid_tokens=self._valid_tokens,
            scalar_denominator=self._scalar_denominator,
            feature_mse=mse,
            feature_cosine=cosine,
            feature_total=feature_total(mse, cosine, cosine_weight=self.cosine_weight),
            cosine_weight=self.cosine_weight,
            per_source=summaries,
            source_order=tuple(self._source_order),
            test_used=False,
            selection_eligible=self.split == "validation",
        )
        self._closed = record
        return record


def evaluate_feature_validation(
    student: Any,
    dataset: Any,
    cache: Any,
    data: Any,
    *,
    cosine_weight: float = 0.1,
    split: str = "validation",
    expected_sources: int = 5,
    expected_windows: int = 102,
) -> FeatureValidationRecord:
    """Evaluate every fixed window with explicit per-source history reset.

    Only the causal endpoint packet is consumed; no future teacher tokens, actions,
    or cached teacher RGB enter the objective.  The caller owns the loaded student;
    this function never runs an optimizer.
    """
    if split not in {"validation", "test"}:
        raise FeatureSelectionError("split must be validation or test")
    manifest = getattr(data, "manifest", None)
    if manifest is None or not callable(getattr(manifest, "sources_for_split", None)):
        raise FeatureSelectionError("data manifest cannot enumerate splits")
    sources = tuple(manifest.sources_for_split(split))
    if len(sources) != int(expected_sources):
        raise FeatureSelectionError(
            f"expected {expected_sources} {split} sources, got {len(sources)}"
        )
    accumulator = TokenFeatureAccumulator(
        cosine_weight=cosine_weight,
        split=split,
        expected_sources=expected_sources,
        expected_windows=expected_windows,
    )
    # Actual held-out validation must never run dropout or training-mode
    # numerical kernels, and the caller's mode must be restored afterwards.
    original_training = getattr(student, "training", None)
    eval_method = getattr(student, "eval", None)
    train_method = getattr(student, "train", None)
    if callable(eval_method):
        eval_method()
    try:
        return _evaluate_feature_validation_body(
            student,
            dataset,
            cache,
            data,
            sources=sources,
            accumulator=accumulator,
            split=split,
        )
    finally:
        if original_training is not None and callable(train_method):
            train_method(bool(original_training))


def _evaluate_feature_validation_body(
    student: Any,
    dataset: Any,
    cache: Any,
    data: Any,
    *,
    sources: Sequence[Any],
    accumulator: TokenFeatureAccumulator,
    split: str,
) -> FeatureValidationRecord:
    with torch.inference_mode():
        for source in sources:
            source_id = str(getattr(source, "source_demo_id", ""))
            if not source_id:
                raise FeatureSelectionError("validation source has no source_demo_id")
            num_frames = int(getattr(source, "num_frames", 0))
            endpoints = validation_endpoints(num_frames)
            accumulator.begin_source(source_id)
            state = None
            for endpoint in endpoints:
                packet = dataset.packet_for_endpoint(source_id, endpoint)
                if bool(getattr(packet, "reset", False)) != (endpoint == 0):
                    raise FeatureSelectionError("packet reset flag does not match source start")
                if endpoint == 0 and state is not None:
                    raise FeatureSelectionError("first endpoint must not inherit state")
                if endpoint > 0 and state is None:
                    raise FeatureSelectionError("later endpoint lost the causal student state")
                output, state = student(packet, state)
                teacher_row = cache.read(source_id, endpoint)
                teacher = teacher_row.teacher_tokens
                teacher_mask = teacher_row.teacher_token_mask
                if teacher.ndim != 3 or teacher_mask.ndim != 1:
                    raise FeatureSelectionError("teacher row token shapes differ")
                teacher = teacher.reshape(1, -1, teacher.shape[-1])
                teacher_mask = (
                    teacher_mask[:, None]
                    .expand(teacher_mask.shape[0], teacher_row.teacher_tokens.shape[1])
                    .reshape(1, -1)
                )
                values = getattr(output, "projected_tokens", None)
                if values is None:
                    values = getattr(getattr(output, "tokens", None), "values", None)
                mask = getattr(getattr(output, "tokens", None), "mask", None)
                if values is None or mask is None:
                    raise FeatureSelectionError("student output lacks tokens/mask")
                combined = mask.to(torch.bool).to(values.device) & teacher_mask.to(values.device)
                accumulator.add_window(source_id, endpoint, values, teacher, combined)
            if state is None:
                raise FeatureSelectionError("source produced no temporal state")
            if int(getattr(state, "last_endpoint", -1)) != endpoints[-1]:
                raise FeatureSelectionError("student state endpoint differs from the last window")
            accumulator.end_source()
    return accumulator.finish()


def select_feature_candidate(
    candidates: Iterable[tuple[int, float]],
) -> tuple[int, float] | None:
    """Minimum total; equal totals resolve to the earlier step."""
    best: tuple[int, float] | None = None
    for step, value in candidates:
        step = int(step)
        value = float(value)
        if step < 0 or not math.isfinite(value):
            raise FeatureSelectionError("selection candidates need step>=0 and finite value")
        if best is None or (value, step) < (best[1], best[0]):
            best = (step, value)
    return best


class FeatureSelector:
    """Token-total selector; action diagnostics are structurally inert."""

    SCHEMA = "owac.raw-stem-temporal.feature-selection.v1"
    METRIC = "validation_token_total"

    def __init__(self) -> None:
        self.best_step: int | None = None
        self.best_value: float | None = None
        self.best_record: Mapping[str, Any] | None = None
        self._offered: dict[int, float] = {}
        self.action_diagnostics: list[dict[str, Any]] = []

    def consider(
        self,
        step: int,
        record: FeatureValidationRecord | Mapping[str, Any],
        *,
        checkpoint_path: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> bool:
        if isinstance(record, FeatureValidationRecord):
            if record.split != "validation" or not record.selection_eligible or record.test_used:
                raise FeatureSelectionError("only eligible validation records may select")
            value = float(record.feature_total)
            payload = record.to_payload()
        elif isinstance(record, Mapping):
            if record.get("split") != "validation":
                raise FeatureSelectionError("only validation records may select")
            if record.get("selection_eligible") is not True:
                raise FeatureSelectionError("feature record is not selection-eligible")
            if record.get("test_used") not in (False, None):
                raise FeatureSelectionError("test split must never be selected on")
            value = float(record["feature_total"])
            payload = {
                key: record[key]
                for key in (
                    "split",
                    "sources",
                    "windows",
                    "anchors",
                    "episode_resets",
                    "valid_tokens",
                    "scalar_denominator",
                    "feature_mse",
                    "feature_cosine",
                    "feature_total",
                    "cosine_weight",
                )
                if key in record
            }
        else:
            raise FeatureSelectionError("unsupported feature record type")
        step = int(step)
        if step < 0 or not math.isfinite(value):
            raise FeatureSelectionError("selection needs step>=0 and finite total")
        if self._offered.get(step, value) != value:
            raise FeatureSelectionError("the same selection step was offered a different value")
        self._offered[step] = value
        improved = self.best_value is None or (value, step) < (self.best_value, self.best_step)
        if improved:
            enriched = dict(payload)
            enriched.update(
                {
                    "step": step,
                    "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
                    "checkpoint_sha256": (
                        None if checkpoint_sha256 is None else str(checkpoint_sha256)
                    ),
                }
            )
            self.best_step = step
            self.best_value = value
            self.best_record = enriched
        return improved

    def record_action_diagnostic(
        self,
        *,
        step: int,
        metric: str,
        value: float,
        checkpoint_sha256: str,
        timestamp: str,
        source: str,
    ) -> dict[str, Any]:
        """Store a diagnostic without any path into selection state."""
        if metric not in {"physical_first5_mae7", "physical_first1_mae7", "physical_full10_mae7"}:
            raise FeatureSelectionError("unknown action diagnostic metric")
        payload = {
            "step": int(step),
            "metric": str(metric),
            "value": float(value),
            "checkpoint_sha256": str(checkpoint_sha256),
            "timestamp": str(timestamp),
            "source": str(source),
            "used_for_selection": False,
            "used_for_stop": False,
        }
        if not math.isfinite(payload["value"]):
            raise FeatureSelectionError("action diagnostic value must be finite")
        self.action_diagnostics.append(payload)
        return payload

    def sidecar(
        self,
        *,
        checkpoint_path: str | None = None,
        checkpoint_sha256: str | None = None,
        parent_baseline: Mapping[str, Any] | None = None,
        legacy_action_selector: Mapping[str, Any] | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        record = dict(self.best_record or {})
        if checkpoint_path is not None:
            record["checkpoint_path"] = str(checkpoint_path)
        if checkpoint_sha256 is not None:
            record["checkpoint_sha256"] = str(checkpoint_sha256)
        return {
            "schema": self.SCHEMA,
            "selection_metric": self.METRIC,
            "direction": "min",
            "tie_break": "earlier_step",
            "best_step": self.best_step,
            "best_value": self.best_value,
            "checkpoint_path": record.get("checkpoint_path"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "valid_tokens": record.get("valid_tokens"),
            "scalar_denominator": record.get("scalar_denominator"),
            "feature_mse": record.get("feature_mse"),
            "feature_cosine": record.get("feature_cosine"),
            "feature_total": record.get("feature_total"),
            "parent_c50_baseline": None if parent_baseline is None else dict(parent_baseline),
            "legacy_action_selector": (
                None if legacy_action_selector is None else dict(legacy_action_selector)
            ),
            "action_diagnostics": list(self.action_diagnostics),
            "updated_at": updated_at,
            "test_used": False,
            "test_selection": False,
        }

"""One real three-state return type and the real calibrator wiring (N02 v1.2 P2).

P1 stored bare :class:`~owac.frameworks.n02_full.sensor.MeasurementScore` objects in
``locked_checks`` and then evaluated ``.verdict``, a field that never existed on
:class:`~owac.frameworks.n02_full.checks.CheckVerdict` (the real field is ``.status``).  Every
check therefore read as the *string* ``"None"`` and was counted as "0 false positives / 0
detections" - a wiring artefact, not a measurement result.

This module fixes the wiring on three axes:

* **one return type** - :class:`AbsoluteCheckResult` always carries a real three-state
  ``status`` together with the complete statistics of the check that produced it.  A check
  that produced no usable verdict is marked ``valid=False`` with a declared reason and is
  counted separately; it is never converted into the string ``"None"`` and never counted as
  a zero.
* **the real calibrator** - :meth:`N02V12Measurement.absolute_check` calls the existing
  :class:`~owac.frameworks.n02_full.checks.WindowedCalibrator` (calibration statistics ->
  frozen window threshold -> test verdict).  :func:`calibrate_windowed` is the only way to
  freeze that threshold, it refuses any partition other than the independent ``calibration``
  split, and it refuses to reuse the ids of the measurements it will later judge.
* **separated semantics** - ``insufficient_information`` (the available measurements cannot
  separate the explanations) and ``model_inadequate`` (the explanation family cannot account
  for the observation) are distinct statuses with distinct handling, and
  :data:`STATUS_INTERPRETATION` states what each one obliges the caller to do.

There is deliberately no threshold on the number of statuses and none on a calibration
error.  An uncalibrated or undecidable statistic is *declared* uncalibrated/invalid and the
energy and difference maps keep flowing to action training.

The residuals this module scores must come from the **pre-locked causal prediction** (the
prediction locked before the current frame was read).  Residuals of a fit that repeatedly
looked at the current frame are *not* an independent check, so :meth:`absolute_check`
requires the caller to declare which prediction version produced them and refuses a
version that claims to be a current-frame fit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from owac.frameworks.n02_full.checks import (
    STATUSES,
    CheckVerdict,
    ResidualView,
    ViewingRule,
    WindowedCalibrator,
    sequence_statistics,
)
from owac.frameworks.n02_full.sensor import (
    deduplicate_samples,
    gaussian_nll,
    sanitise_variance,
)

__all__ = [
    "ALLOWED_PREDICTION_VERSIONS",
    "AbsoluteCheckResult",
    "CalibrationIsolationError",
    "N02V12Measurement",
    "STATUS_INTERPRETATION",
    "WindowedCalibration",
    "calibrate_windowed",
    "summarise_results",
]

#: What each real status means and what the caller must do with it.
STATUS_INTERPRETATION: dict[str, str] = {
    "compatible": (
        "the declared candidate family and measurement model still explain the observed "
        "residual pattern under the frozen window threshold: keep the usable explanations and "
        "continue the action readout"
    ),
    "insufficient_information": (
        "the usable measurements cannot separate the explanations under the declared viewing "
        "rule (no usable valid samples / declared views): keep the ambiguity and do not force a "
        "confident choice"
    ),
    "model_inadequate": (
        "the residual sequence exceeds the frozen threshold and the declared interference does "
        "not explain it: the current explanation family cannot account for the observation, so "
        "propose an alternative explanation or rebuild the affected state"
    ),
    "invalid": (
        "no usable verdict was produced (missing or unfitted calibrator, missing verdict, or a "
        "status outside the three declared states): counted separately, never as a false "
        "positive and never as a detection, and never rendered as the string 'None'"
    ),
}

#: Prediction versions that are allowed to feed an absolute check.
ALLOWED_PREDICTION_VERSIONS = ("causal_pre_locked", "declared_reference")


class CalibrationIsolationError(ValueError):
    """Raised when a calibration would reuse the measurements it is supposed to judge."""


@dataclass(slots=True)
class AbsoluteCheckResult:
    """One real three-state absolute check result.

    ``status`` is exactly one of :data:`owac.frameworks.n02_full.checks.STATUSES`.
    ``valid`` is ``False`` only when no usable verdict exists; such a row is counted on its
    own and is excluded from every false-positive / detection count.
    """

    status: str
    valid: bool
    statistics: dict
    calibration_version: str
    #: Aggregate views that exceeded the frozen threshold, strongest first.
    local_residual_positions: tuple = ()
    viewing_rule_hash: str = ""
    n_sequences: int = 0
    per_sequence: list = field(default_factory=list)
    explanation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in STATUSES:
            raise ValueError(
                f"status must be one of {STATUSES} (never a missing-value string), "
                f"got {self.status!r}"
            )
        self.valid = bool(self.valid)
        self.statistics = {
            str(name): (float(value) if isinstance(value, bool) else value)
            for name, value in dict(self.statistics).items()
        }
        for name in ("calibration_version", "viewing_rule_hash", "explanation"):
            if not isinstance(getattr(self, name), str):
                object.__setattr__(self, name, str(getattr(self, name)))
        if not self.explanation:
            self.explanation = STATUS_INTERPRETATION[self.status]
        self.local_residual_positions = tuple(
            (int(row), int(column)) for row, column in self.local_residual_positions
        )
        self.n_sequences = int(self.n_sequences)
        self.per_sequence = [dict(item) for item in self.per_sequence]

    @classmethod
    def invalid(
        cls,
        *,
        reason: str,
        calibration_version: str = "uncalibrated",
        statistics: dict | None = None,
        status: str = "insufficient_information",
    ) -> AbsoluteCheckResult:
        """A declared non-verdict: no usable check ran, so nothing may be counted as a 0."""
        payload = {
            "invalid": 1.0,
            "reason_code": float(_reason_code(reason)),
            **dict(statistics or {}),
        }
        return cls(
            status=status,
            valid=False,
            statistics={**payload, "reason": reason},
            calibration_version=calibration_version,
            explanation=f"invalid result: {STATUS_INTERPRETATION['invalid']} ({reason})",
        )

    @property
    def counted_as_false_positive(self) -> bool:
        return False

    @property
    def counted_as_detection(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "valid": self.valid,
            "calibration_version": self.calibration_version,
            "viewing_rule_hash": self.viewing_rule_hash,
            "n_sequences": self.n_sequences,
            "local_residual_positions": [list(item) for item in self.local_residual_positions],
            "statistics": dict(self.statistics),
            "per_sequence": [dict(item) for item in self.per_sequence],
            "explanation": self.explanation,
        }


def _reason_code(reason: str) -> int:
    """Stable small integer so the reason also survives in a numeric statistics dict."""
    codes = {
        "no_fitted_calibrator": 1,
        "unfitted_calibrator": 2,
        "missing_verdict": 3,
        "unknown_status": 4,
        "invalid_prediction_version": 5,
        "invalid_measurement_set": 6,
        "non_finite_calibration_statistics": 7,
    }
    return codes.get(str(reason), 99)


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fitted(calibrator: Any) -> bool:
    return (
        calibrator is not None
        and hasattr(calibrator, "assess_views")
        and hasattr(calibrator, "threshold")
    )


def _resolve_calibrator(calibrator: Any, version: str) -> tuple[Any, str]:
    """Accept a bare frozen calibrator or the :class:`WindowedCalibration` wrapper."""
    if isinstance(calibrator, WindowedCalibration):
        return calibrator.calibrator, calibrator.version
    return calibrator, str(getattr(calibrator, "version", version))


def _as_interference_groups(interference_views: Any, n_groups: int) -> list[Any]:
    if interference_views is None:
        return [None] * n_groups
    items = list(interference_views)
    if items and all(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in items
    ):
        if len(items) != n_groups:
            raise ValueError(
                f"got {len(items)} interference view groups for {n_groups} residual sequences"
            )
        return [list(item) for item in items]
    return [items] * n_groups


@dataclass(slots=True)
class WindowedCalibration:
    """A frozen window threshold plus the identity of the split it was frozen on.

    The wrapper keeps the version, the partition and the viewing rule next to the frozen
    ``WindowedCalibrator``, so an absolute check can report *which* calibration produced its
    verdict instead of an anonymous number.
    """

    calibrator: WindowedCalibrator
    version: str
    partition: str
    rule: ViewingRule
    quantile: float
    n_sequences: int
    finite_sequences: int

    @property
    def threshold(self) -> float:
        return float(self.calibrator.threshold())

    @property
    def viewing_rule_hash(self) -> str:
        return str(self.calibrator.viewing_rule_hash)

    def assess_views(
        self,
        views: Sequence[ResidualView],
        *,
        version: str | None = None,
        rule: ViewingRule | None = None,
        interference_views: Sequence[ResidualView] | None = None,
    ) -> CheckVerdict:
        return self.calibrator.assess_views(
            views,
            version=version or self.version,
            rule=rule or self.rule,
            interference_views=interference_views,
        )

    def assess(self, statistics: Sequence[float], *, version: str | None = None) -> CheckVerdict:
        return self.calibrator.assess(statistics, version=version or self.version)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "partition": self.partition,
            "threshold": self.threshold,
            "quantile": float(self.quantile),
            "n_sequences": int(self.n_sequences),
            "finite_sequences": int(self.finite_sequences),
            "viewing_rule_hash": self.viewing_rule_hash,
        }


def calibrate_windowed(
    residual_groups: Sequence[Sequence[ResidualView]],
    *,
    partition: str,
    rule: ViewingRule,
    quantile: float = 0.95,
    calibration_ids: Iterable[Any] | None = None,
    test_ids: Iterable[Any] | None = None,
    version: str = "n02-v12-windowed-check-v1",
) -> WindowedCalibrator:
    """Freeze the window threshold on the independent ``calibration`` partition.

    ``residual_groups`` is one :class:`ResidualView` group per calibration sequence, and
    every group must carry the declared ``(scale, frame, region)`` views of ``rule``.
    Reusing the test measurements here would break the independence of the check, so an
    overlap between ``calibration_ids`` and ``test_ids`` raises instead of fitting.
    """
    if str(partition) != "calibration":
        raise CalibrationIsolationError(
            "the compatible threshold must be frozen on the independent calibration split; "
            f"got partition={partition!r}"
        )
    if calibration_ids is not None and test_ids is not None:
        overlap = set(map(str, calibration_ids)) & set(map(str, test_ids))
        if overlap:
            raise CalibrationIsolationError(
                f"calibration reuses {len(overlap)} measurement ids it must judge, "
                f"e.g. {sorted(overlap)[0]!r}"
            )
    statistics = sequence_statistics(residual_groups, rule=rule)
    finite = [value for value in statistics if math.isfinite(value)]
    if not finite:
        raise ValueError(
            "no finite calibration statistic under the declared viewing rule; refusing to "
            f"freeze the meaningless threshold {math.inf}"
        )
    calibrator = WindowedCalibrator.fit(
        statistics, quantile=float(quantile), rule_hash=rule.rule_hash()
    )
    return WindowedCalibration(
        calibrator=calibrator,
        version=str(version),
        partition=str(partition),
        rule=rule,
        quantile=float(quantile),
        n_sequences=len(statistics),
        finite_sequences=len(finite),
    )


class N02V12Measurement:
    """The complete measurement objective and the three-state absolute check.

    ``energy`` is the full declared objective over one measurement set (all valid samples,
    no fixed-stride subsample and no averaging that would hide a local structure).
    ``absolute_check`` calls the frozen :class:`WindowedCalibrator` and returns the real
    status.  Neither method sets a pass/fail threshold.
    """

    def __init__(
        self,
        *,
        min_variance: float | None = None,
        max_variance: float | None = None,
    ) -> None:
        if min_variance is not None:
            if min_variance <= 0.0:
                raise ValueError("min_variance must be positive")
        if max_variance is not None and min_variance is not None and max_variance < min_variance:
            raise ValueError("max_variance must be >= min_variance")
        self.min_variance = min_variance
        self.max_variance = max_variance

    # -- complete objective -------------------------------------------------------------

    def objective(
        self,
        *,
        mean: Any,
        variance: Any,
        masks: Any,
        target: Any,
        weights: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """``(energy (B, K), info)`` of the full declared objective.

        ``mean`` is ``(B, K, ...)``; ``variance``, ``target`` and ``masks`` broadcast to it.
        With a finite positive ``variance`` every valid scalar contributes
        ``0.5*((y-mu)^2/v + log v + log 2pi)``; without one the objective falls back to the
        empirical energy ``0.5*(y-mu)^2`` and says so in ``info['kind']`` (an empirical
        energy is a support weight, never a calibrated posterior).  ``weights`` lets the
        caller declare a disjoint weighting for derived representations of one measurement;
        overlapping multi-scale views must be weighted, not summed twice.
        """
        mean_tensor, target_tensor, variance_tensor, mask_tensor, weight_tensor = self._align(
            mean=mean, variance=variance, masks=masks, target=target, weights=weights
        )
        if weight_tensor is not None and (
            not bool(torch.isfinite(weight_tensor).all()) or bool((weight_tensor < 0).any())
        ):
            raise ValueError("measurement weights must be finite and nonnegative")
        weights_used = mask_tensor.to(mean_tensor.dtype)
        if weight_tensor is not None:
            weights_used = weights_used * weight_tensor
        scored = weights_used > 0
        if not bool((torch.isfinite(mean_tensor) | ~scored).all()) or not bool(
            (torch.isfinite(target_tensor) | ~scored).all()
        ):
            raise ValueError("nonfinite value in a scored measurement")
        # Mask before arithmetic: NaN * 0 still produces NaN (including in backward).
        mean_tensor = torch.where(scored, mean_tensor, 0.0)
        target_tensor = torch.where(scored, target_tensor, 0.0)
        if variance_tensor is None:
            residual = target_tensor - mean_tensor
            term = 0.5 * residual * residual
            kind = "empirical_energy"
            expression = "0.5*(y-mu)^2"
        else:
            if not bool(
                ((torch.isfinite(variance_tensor) & (variance_tensor > 0)) | ~scored).all()
            ):
                raise ValueError("scored measurement variance must be finite and positive")
            variance_tensor = torch.where(scored, variance_tensor, 1.0)
            variance_kwargs: dict[str, float] = {}
            if self.min_variance is not None:
                variance_kwargs["min_variance"] = float(self.min_variance)
            if self.max_variance is not None:
                variance_kwargs["max_variance"] = float(self.max_variance)
            variance_tensor = sanitise_variance(variance_tensor, **variance_kwargs)
            term = gaussian_nll(target_tensor, mean_tensor, variance_tensor, **variance_kwargs)
            kind = "gaussian_nll"
            expression = "0.5*((y-mu)^2/v + log v + log 2pi)"
        contributions = term * weights_used
        reduction = tuple(range(2, contributions.ndim))
        energy = contributions.sum(dim=reduction) if reduction else contributions
        valid_per_candidate = (
            mask_tensor.to(torch.float64).sum(dim=reduction)
            if reduction
            else mask_tensor.to(torch.float64)
        )
        info = {
            "kind": kind,
            "expression": expression,
            "shape": list(energy.shape),
            "n_valid_per_candidate": valid_per_candidate.reshape(-1).tolist(),
            "sampled_all_valid_cells": weight_tensor is None,
            "effective_weight_per_candidate": weights_used.sum(dim=reduction).reshape(-1).tolist()
            if reduction
            else weights_used.reshape(-1).tolist(),
            "variance_finite": variance_tensor is not None,
        }
        return energy, info

    def energy(
        self,
        *,
        mean: Any,
        variance: Any,
        masks: Any,
        target: Any,
        weights: Any | None = None,
    ) -> torch.Tensor:
        """``(B, K)`` complete objective over the same measurement set for every candidate."""
        tensor, _ = self.objective(
            mean=mean, variance=variance, masks=masks, target=target, weights=weights
        )
        return tensor

    @staticmethod
    def _align(
        *,
        mean: Any,
        variance: Any,
        masks: Any,
        target: Any,
        weights: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        mean_tensor = _tensor(mean)
        # RAW likelihood accumulation stays at least FP32 under mixed precision.
        if mean_tensor.dtype != torch.float64:
            mean_tensor = mean_tensor.float()
        if mean_tensor.ndim < 2:
            raise ValueError(
                f"mean must have shape (B, K, ...) so (B, K) energies are defined; got "
                f"{tuple(mean_tensor.shape)}"
            )

        def on_device(value, dtype=None):
            return torch.as_tensor(value, device=mean_tensor.device, dtype=dtype)

        target_tensor = _insert_candidate_axis(
            on_device(target, mean_tensor.dtype), mean_tensor.ndim
        )
        mask_tensor = _insert_candidate_axis(on_device(masks, torch.bool), mean_tensor.ndim)
        tensors: list[torch.Tensor] = [
            mean_tensor,
            target_tensor,
            mask_tensor.to(mean_tensor.dtype),
        ]
        variance_tensor: torch.Tensor | None = None
        if variance is not None:
            variance_tensor = _insert_candidate_axis(
                on_device(variance, mean_tensor.dtype), mean_tensor.ndim
            )
            tensors.append(variance_tensor)
        weight_tensor: torch.Tensor | None = None
        if weights is not None:
            weight_tensor = _insert_candidate_axis(
                on_device(weights, mean_tensor.dtype), mean_tensor.ndim
            )
            tensors.append(weight_tensor)
        aligned = torch.broadcast_tensors(*tensors)
        mean_out, target_out, mask_float = aligned[0], aligned[1], aligned[2]
        cursor = 3
        if variance_tensor is not None:
            variance_tensor = aligned[cursor]
            cursor += 1
        if weight_tensor is not None:
            weight_tensor = aligned[cursor]
        return mean_out, target_out, variance_tensor, mask_float >= 0.5, weight_tensor

    # -- sampling coverage: no fixed-stride blind spot ----------------------------------

    @staticmethod
    def coverage(
        *, masks: Any, target: Any | None = None, weights: Any | None = None
    ) -> dict[str, Any]:
        """How many real sensor cells the objective used, and how they relate to all of them.

        The full valid region is the point: when the objective uses every valid cell,
        ``fraction_used`` is ``1.0`` and ``used_mean_minus_full_sensor_mean`` is exactly
        ``0.0``.  A sparse fixed-stride query would show a smaller fraction and a non-zero
        difference, which is how the P1 sampling blind spot becomes visible instead of
        hidden.
        """
        mask_tensor = _tensor(masks, dtype=torch.bool)
        used = mask_tensor if weights is None else (mask_tensor & (_tensor(weights) != 0))
        n_full = int(mask_tensor.sum())
        n_used = int(used.sum())
        report: dict[str, Any] = {
            "n_sensor_cells": int(mask_tensor.numel()),
            "n_valid_cells": n_full,
            "n_used_cells": n_used,
            "fraction_used_of_valid": (float(n_used) / float(n_full)) if n_full else float("nan"),
            "uses_every_valid_cell": bool(n_full > 0 and n_used == n_full),
        }
        if target is not None:
            target_tensor = _tensor(target)
            full_values = (
                target_tensor[mask_tensor]
                if mask_tensor.shape == target_tensor.shape
                else target_tensor.reshape(-1)
            )
            used_values = target_tensor[used] if used.shape == target_tensor.shape else full_values
            full_mean = (
                float(full_values.to(torch.float64).mean()) if full_values.numel() else float("nan")
            )
            used_mean = (
                float(used_values.to(torch.float64).mean()) if used_values.numel() else float("nan")
            )
            report.update(
                {
                    "used_mean": used_mean,
                    "full_sensor_mean": full_mean,
                    "used_mean_minus_full_sensor_mean": used_mean - full_mean,
                }
            )
        return report

    @staticmethod
    def deduplicate(
        *, sample_ids: Any | None = None, coordinates: Any | None = None, decimals: int = 4
    ):
        """First-occurrence indices of unique measurements (delegated to the frozen helper).

        Inner-loop rounds, prefix replays and multi-scale features of one frame keep one
        identity here, so they cannot be counted as several independent pieces of evidence.
        """
        return deduplicate_samples(
            sample_ids=sample_ids, coordinates=coordinates, decimals=decimals
        )

    # -- real three-state absolute check ------------------------------------------------

    def absolute_check(
        self,
        *,
        residuals: Any,
        calibrator: WindowedCalibrator | None,
        version: str = "n02-v12-windowed-check-v1",
        rule: ViewingRule | None = None,
        interference_views: Any | None = None,
        prediction_version: str = "causal_pre_locked",
        measurement_ids: Iterable[Any] | None = None,
        calibration_ids: Iterable[Any] | None = None,
    ) -> AbsoluteCheckResult:
        """Assess one or more residual sequences against the frozen threshold.

        ``residuals`` is either one sequence of :class:`ResidualView` (one decision moment) or
        a sequence of such sequences.  A missing / unfitted calibrator, a missing verdict, or
        an unknown status produce ``valid=False`` with a real reason; they never produce the
        string ``"None"`` and are never counted as a zero.
        """
        if str(prediction_version) not in ALLOWED_PREDICTION_VERSIONS:
            return AbsoluteCheckResult.invalid(
                reason="invalid_prediction_version",
                calibration_version=version,
                statistics={"prediction_version": float(_reason_code(str(prediction_version)))},
            )
        if measurement_ids is not None and calibration_ids is not None:
            overlap = set(map(str, measurement_ids)) & set(map(str, calibration_ids))
            if overlap:
                raise CalibrationIsolationError(
                    f"the check reuses {len(overlap)} calibration measurements, "
                    f"e.g. {sorted(overlap)[0]!r}"
                )
        if not _fitted(calibrator):
            return AbsoluteCheckResult.invalid(
                reason="no_fitted_calibrator" if calibrator is None else "unfitted_calibrator",
                calibration_version=version,
                statistics={
                    "residual_views": float(len(residuals))
                    if isinstance(residuals, Sequence)
                    else float("nan")
                },
            )
        groups = _as_view_groups(residuals)
        if not groups:
            return AbsoluteCheckResult.invalid(
                reason="invalid_measurement_set",
                calibration_version=version,
                statistics={"residual_views": 0.0},
            )
        calibrator, calibration_version = _resolve_calibrator(calibrator, version)
        interference_groups = _as_interference_groups(interference_views, len(groups))
        verdicts: list[CheckVerdict] = []
        rule_hashes: set[str] = set()
        for group, interference_group in zip(groups, interference_groups, strict=True):
            if not group:
                return AbsoluteCheckResult.invalid(
                    reason="invalid_measurement_set", calibration_version=calibration_version
                )
            if all(isinstance(item, ResidualView) for item in group):
                if rule is None:
                    return AbsoluteCheckResult.invalid(
                        reason="missing_verdict",
                        calibration_version=calibration_version,
                        statistics={"residual_views": float(len(group))},
                    )
                verdict = calibrator.assess_views(
                    group,
                    version=calibration_version,
                    rule=rule,
                    interference_views=interference_group,
                )
            else:
                verdict = calibrator.assess(
                    [float(item) for item in group], version=calibration_version
                )
            if not isinstance(verdict, CheckVerdict):
                return AbsoluteCheckResult.invalid(
                    reason="missing_verdict", calibration_version=calibration_version
                )
            if verdict.status not in STATUSES:
                return AbsoluteCheckResult.invalid(
                    reason="unknown_status", calibration_version=calibration_version
                )
            verdicts.append(verdict)
            rule_hashes.add(str(getattr(verdict, "viewing_rule_hash", "")))

        status = _combine_statuses([verdict.status for verdict in verdicts])
        combined = _combine_statistics(verdicts)
        positions: list[tuple[int, int]] = []
        for verdict in verdicts:
            for position in getattr(verdict, "local_residual_positions", ()):
                if position not in positions:
                    positions.append((int(position[0]), int(position[1])))
        return AbsoluteCheckResult(
            status=status,
            valid=True,
            statistics=combined,
            calibration_version=calibration_version,
            local_residual_positions=tuple(positions[:16]),
            viewing_rule_hash=sorted(rule_hashes)[0] if len(rule_hashes) == 1 else "",
            n_sequences=len(verdicts),
            per_sequence=[
                {
                    "status": verdict.status,
                    "statistics": dict(verdict.statistics),
                    "viewing_rule_hash": str(getattr(verdict, "viewing_rule_hash", "")),
                }
                for verdict in verdicts
            ],
        )


def _tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    elif dtype is None and not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def _insert_candidate_axis(tensor: torch.Tensor, mean_ndim: int) -> torch.Tensor:
    """Accept a per-measurement ``(B, ...)`` tensor next to a per-candidate ``(B, K, ...)``."""
    if tensor.ndim == mean_ndim - 1:
        return tensor.unsqueeze(1)
    return tensor


def _as_view_groups(residuals: Any) -> list[list[Any]]:
    if isinstance(residuals, ResidualView):
        return [[residuals]]
    if not isinstance(residuals, Sequence):
        return []
    items = list(residuals)
    if not items:
        return []
    if all(isinstance(item, ResidualView) for item in items):
        return [items]
    if all(isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in items):
        return [list(item) for item in items]
    return []


def _combine_statuses(statuses: Sequence[str]) -> str:
    if any(status == "model_inadequate" for status in statuses):
        return "model_inadequate"
    if any(status == "insufficient_information" for status in statuses):
        return "insufficient_information"
    return "compatible"


def _combine_statistics(verdicts: Sequence[CheckVerdict]) -> dict[str, float]:
    payload: dict[str, float] = {
        "n_sequences": float(len(verdicts)),
        "n_compatible": float(sum(item.status == "compatible" for item in verdicts)),
        "n_insufficient_information": float(
            sum(item.status == "insufficient_information" for item in verdicts)
        ),
        "n_model_inadequate": float(sum(item.status == "model_inadequate" for item in verdicts)),
        "invalid": 0.0,
    }
    if len(verdicts) == 1:
        # one sequence: keep the real statistic names the frozen calibrator reported
        payload.update(
            {
                name: value
                for name, value in verdicts[0].statistics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        return payload
    numeric_names: set[str] = set()
    for verdict in verdicts:
        numeric_names.update(
            name
            for name, value in verdict.statistics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
    for name in sorted(numeric_names):
        values = [
            float(verdict.statistics[name])
            for verdict in verdicts
            if name in verdict.statistics
            and isinstance(verdict.statistics[name], (int, float))
            and not isinstance(verdict.statistics[name], bool)
        ]
        if values:
            payload[f"min__{name}"] = min(values)
            payload[f"max__{name}"] = max(values)
    return payload


def summarise_results(results: Sequence[AbsoluteCheckResult]) -> dict[str, Any]:
    """Real counts per status plus the separate invalid count.

    No threshold lives here: the counts report what happened.  ``invalid`` rows are excluded
    from every status count and from any downstream false-positive / detection rate, because
    a missing verdict is not a zero.
    """
    counts = {status: 0 for status in STATUSES}
    invalid = 0
    for result in results:
        if result.valid:
            counts[result.status] = counts.get(result.status, 0) + 1
        else:
            invalid += 1
    return {
        "total": len(results),
        "valid": len(results) - invalid,
        "invalid": invalid,
        "compatible": counts["compatible"],
        "insufficient_information": counts["insufficient_information"],
        "model_inadequate": counts["model_inadequate"],
        "status_interpretation": dict(STATUS_INTERPRETATION),
        "counting_contract": (
            "invalid rows carry no verdict: they are counted here and excluded from every "
            "false-positive and detection count; no status count is compared with a threshold"
        ),
    }


def v12_viewing_rule(query_stride: int, camera_ids: Sequence[str], window: int = 4) -> ViewingRule:
    """One shared rule for flattened per-camera query residuals.

    Local image windows require a spatial raster, which these residual vectors do not
    claim to provide. Keep that statistic disabled on both sides; never silently use
    a local maximum for calibration and a global mean for evaluation.
    """
    return ViewingRule(
        scales=(f"sensor_stride{query_stride}",),
        frames=(0,),
        regions=tuple(camera_ids),
        window=window,
        use_local_maximum=False,
    )

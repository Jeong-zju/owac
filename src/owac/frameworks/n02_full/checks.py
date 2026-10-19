"""Fixed-window absolute checks for the N02 P1 measurement model (T2).

A check answers one question: is the observed residual pattern explicable by the declared
candidate family and measurement model, or is the model inadequate?  ``status`` is exactly
one of three values and never names a specific relation:

* ``compatible`` - nothing above the frozen threshold;
* ``insufficient_information`` - no usable valid samples / view under the declared rule;
* ``model_inadequate`` - the sequence statistic is above the frozen threshold and the
  declared interference does not explain it.

The threshold is an empirical quantile of one statistic per calibration sequence, computed
under **one declared viewing rule** (scales, frames, regions, window, statistic).  Test and
calibration must use the identical rule: :meth:`WindowedCalibrator.assess_views` refuses a
mismatching rule hash.  This is a fixed-window empirical test, **not** an anytime e-process
guarantee, and an anomaly means the candidate family or measurement model is insufficient -
it does not identify which relation is wrong.

The check reports the whole-image mean standardised residual **and** the local window
maximum, so a local structure (for example the old "two separated bright spots" failure)
cannot be averaged away by an otherwise normal background.  Only fully valid windows count.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "STATUSES",
    "CheckVerdict",
    "ResidualView",
    "ViewingRule",
    "ViewingRuleMismatchError",
    "WindowedCalibrator",
    "select_views",
    "sequence_statistic",
    "sequence_statistics",
    "windowed_statistics",
]

#: The only three statuses an absolute check may report.
STATUSES = ("compatible", "insufficient_information", "model_inadequate")


class ViewingRuleMismatchError(ValueError):
    """Raised when calibration and test do not use the identical viewing rule."""


@dataclass(frozen=True, slots=True)
class ViewingRule:
    """Exactly which scales, frames and regions are inspected, and with which window.

    The rule hash is the identity of the viewing rule: calibration and test must present
    the same hash, and every declared ``(scale, frame, region)`` view must be present.
    """

    scales: tuple[str, ...]
    frames: tuple[int, ...]
    regions: tuple[str, ...] = ("full",)
    window: int = 4
    statistic: str = "max_local_or_global_abs_standardised_residual"
    use_global_mean: bool = True
    use_local_maximum: bool = True
    min_valid_samples: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "scales", tuple(str(item) for item in self.scales))
        object.__setattr__(self, "frames", tuple(int(item) for item in self.frames))
        object.__setattr__(self, "regions", tuple(str(item) for item in self.regions))
        if not self.scales:
            raise ValueError("a viewing rule needs at least one scale")
        if not self.frames:
            raise ValueError("a viewing rule needs at least one frame")
        if not self.regions:
            raise ValueError("a viewing rule needs at least one region")
        if self.window < 1:
            raise ValueError("window must be at least one sample")
        if self.min_valid_samples < 0:
            raise ValueError("min_valid_samples must be non-negative")
        if not (self.use_global_mean or self.use_local_maximum):
            raise ValueError("a viewing rule must use the global mean and/or the local maximum")
        if not isinstance(self.statistic, str) or not self.statistic:
            raise ValueError("statistic must be a non-empty name")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "scales": list(self.scales),
            "frames": list(self.frames),
            "regions": list(self.regions),
            "window": int(self.window),
            "statistic": self.statistic,
            "use_global_mean": bool(self.use_global_mean),
            "use_local_maximum": bool(self.use_local_maximum),
            "min_valid_samples": int(self.min_valid_samples),
        }

    def rule_hash(self) -> str:
        """sha256 of the canonical rule JSON; equal rules hash equal."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ResidualView:
    """One inspected view: a signed standardised-residual map plus its validity mask.

    ``origin`` is the top-left corner of the view in its sensor/frame coordinates, so a
    local position reported by the check is ``origin + (window centre)``.
    """

    standardised_residual: torch.Tensor
    valid: torch.Tensor | None = None
    scale: str = "stride8"
    frame_index: int = 0
    region_id: str = "full"
    origin: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        residual = torch.as_tensor(self.standardised_residual)
        if residual.ndim != 2:
            raise ValueError(
                f"standardised_residual must be a 2-D map, got shape {tuple(residual.shape)}"
            )
        self.standardised_residual = residual
        if self.valid is None:
            self.valid = torch.ones_like(residual, dtype=torch.bool)
        else:
            mask = torch.as_tensor(self.valid, dtype=torch.bool)
            if tuple(mask.shape) != tuple(residual.shape):
                raise ValueError(
                    f"valid shape {tuple(mask.shape)} != residual shape {tuple(residual.shape)}"
                )
            self.valid = mask
        self.scale = str(self.scale)
        self.frame_index = int(self.frame_index)
        self.region_id = str(self.region_id)
        self.origin = (int(self.origin[0]), int(self.origin[1]))


@dataclass(slots=True)
class CheckVerdict:
    """The absolute-check output; it never names a specific relation.

    ``interference_explainable`` is True only when the raw sequence was above threshold and
    the declared interference-corrected sequence fell back to or below it.  An anomaly is a
    statement about the candidate family / measurement model, not about one relation.
    """

    status: str
    interference_explainable: bool
    local_residual_positions: tuple[tuple[int, int], ...]
    calibration_version: str
    statistics: dict[str, float]
    viewing_rule_hash: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        self.interference_explainable = bool(self.interference_explainable)
        self.local_residual_positions = tuple(
            (int(row), int(col)) for row, col in self.local_residual_positions
        )
        self.calibration_version = str(self.calibration_version)
        self.statistics = {str(name): float(value) for name, value in dict(self.statistics).items()}
        self.viewing_rule_hash = str(self.viewing_rule_hash)


def select_views(views: Sequence[ResidualView], *, rule: ViewingRule) -> tuple[ResidualView, ...]:
    """Select the views the rule declares, in rule order.

    Every declared ``(scale, frame, region)`` combination must be present exactly once;
    otherwise :class:`ViewingRuleMismatchError` is raised, so calibration and test cannot
    silently inspect different view sets.  Views outside the rule are not inspected.
    """
    buckets: dict[tuple[str, int, str], list[ResidualView]] = {}
    for view in views:
        key = (str(view.scale), int(view.frame_index), str(view.region_id))
        buckets.setdefault(key, []).append(view)
    selected: list[ResidualView] = []
    missing: list[tuple[str, int, str]] = []
    for scale in rule.scales:
        for frame in rule.frames:
            for region in rule.regions:
                key = (scale, frame, region)
                found = buckets.pop(key, [])
                if not found:
                    missing.append(key)
                    continue
                if len(found) > 1:
                    raise ViewingRuleMismatchError(f"viewing rule expects one view for {key}")
                selected.append(found[0])
    if missing:
        raise ViewingRuleMismatchError(f"viewing rule requires missing views: {sorted(missing)}")
    return tuple(selected)


def _window_maps(
    view: ResidualView, rule: ViewingRule
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """``(abs_residual, local_mean, full_window_mask, n_valid)`` in float64.

    ``local_mean`` is ``-inf`` wherever a window is not fully valid, so partial border
    windows never inflate the local maximum.
    """
    residual = torch.as_tensor(view.standardised_residual, dtype=torch.float64)
    valid = torch.as_tensor(view.valid, dtype=torch.bool)
    n_valid = int(valid.sum())
    abs_residual = torch.where(valid, residual.abs(), torch.zeros_like(residual))
    window = int(rule.window)
    if n_valid == 0 or window > residual.shape[0] or window > residual.shape[1]:
        return (
            abs_residual,
            torch.full((0, 0), float("-inf"), dtype=torch.float64),
            torch.zeros((0, 0), dtype=torch.bool),
            n_valid,
        )
    kernel_size = window * window
    sum_pool = (
        F.avg_pool2d(
            (abs_residual * valid).unsqueeze(0).unsqueeze(0),
            window,
            stride=1,
            count_include_pad=False,
        )
        * kernel_size
    )
    count_pool = (
        F.avg_pool2d(
            valid.to(torch.float64).unsqueeze(0).unsqueeze(0),
            window,
            stride=1,
            count_include_pad=False,
        )
        * kernel_size
    )
    full_window = count_pool >= float(kernel_size)
    local_mean = torch.where(
        full_window,
        sum_pool / count_pool.clamp_min(1.0),
        torch.full_like(sum_pool, float("-inf")),
    )
    return abs_residual, local_mean[0, 0], full_window[0, 0], n_valid


def windowed_statistics(view: ResidualView, *, rule: ViewingRule) -> dict[str, float]:
    """Whole-image mean and local window maximum of one view, plus the rule statistic."""
    abs_residual, local_mean, full_window, n_valid = _window_maps(view, rule)
    if n_valid < max(1, int(rule.min_valid_samples)):
        return {
            "global_mean_abs_standardised_residual": float("nan"),
            "local_max_abs_standardised_residual": float("-inf"),
            "statistic": float("-inf"),
            "n_valid_samples": float(n_valid),
            "n_full_windows": float(int(full_window.sum())),
        }
    global_mean = float(abs_residual.sum()) / float(n_valid)
    local_max = float(local_mean.max()) if bool(full_window.any()) else float("-inf")
    components: list[float] = []
    if rule.use_global_mean:
        components.append(global_mean)
    if rule.use_local_maximum and math.isfinite(local_max):
        components.append(local_max)
    statistic = max(components) if components else float("-inf")
    return {
        "global_mean_abs_standardised_residual": global_mean,
        "local_max_abs_standardised_residual": local_max,
        "statistic": statistic,
        "n_valid_samples": float(n_valid),
        "n_full_windows": float(int(full_window.sum())),
    }


def _local_positions(
    views: Sequence[ResidualView],
    *,
    rule: ViewingRule,
    threshold: float,
    limit: int = 16,
) -> tuple[tuple[int, int], ...]:
    """Window centres whose local mean exceeds ``threshold``, strongest first.

    Positions are in the view's declared coordinate frame (``origin`` included).
    """
    ranked: list[tuple[float, tuple[int, int]]] = []
    half = int(rule.window) // 2
    for view in views:
        _, local_mean, full_window, n_valid = _window_maps(view, rule)
        if n_valid < max(1, int(rule.min_valid_samples)) or not bool(full_window.any()):
            continue
        rows, columns = torch.nonzero(full_window & (local_mean > threshold), as_tuple=True)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            value = float(local_mean[row, column])
            position = (
                int(view.origin[0]) + int(row) + half,
                int(view.origin[1]) + int(column) + half,
            )
            ranked.append((value, position))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    unique: list[tuple[int, int]] = []
    for _, position in ranked:
        if position not in unique:
            unique.append(position)
        if len(unique) >= limit:
            break
    return tuple(unique)


def sequence_statistic(views: Sequence[ResidualView], *, rule: ViewingRule) -> float:
    """One sequence's statistic: the maximum over the views the rule declares."""
    selected = select_views(views, rule=rule)
    values = [windowed_statistics(view, rule=rule)["statistic"] for view in selected]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("-inf")


def sequence_statistics(
    view_groups: Sequence[Sequence[ResidualView]], *, rule: ViewingRule
) -> list[float]:
    """One statistic per sequence, computed under the identical viewing rule."""
    return [sequence_statistic(group, rule=rule) for group in view_groups]


class WindowedCalibrator:
    """A frozen empirical 95% threshold over one statistic per calibration sequence.

    ``fit`` freezes the threshold and the identity (hash) of the viewing rule it was
    computed under.  ``assess`` compares a test sequence against that frozen threshold and
    never re-estimates it.  This is a fixed-window empirical test; it carries no anytime
    e-process guarantee.
    """

    __slots__ = ("_threshold", "_quantile", "_viewing_rule_hash", "_n_samples")

    def __init__(
        self, *, threshold: float, quantile: float, rule_hash: str, n_samples: int
    ) -> None:
        if not math.isfinite(float(threshold)):
            raise ValueError("threshold must be finite")
        if not 0.0 < float(quantile) <= 1.0:
            raise ValueError("quantile must lie in (0, 1]")
        if n_samples < 1:
            raise ValueError("at least one calibration sample is required")
        self._threshold = float(threshold)
        self._quantile = float(quantile)
        self._viewing_rule_hash = str(rule_hash)
        self._n_samples = int(n_samples)

    @classmethod
    def fit(
        cls,
        statistics: Sequence[float],
        *,
        quantile: float = 0.95,
        rule_hash: str,
    ) -> WindowedCalibrator:
        values = [float(item) for item in statistics]
        finite = [item for item in values if math.isfinite(item)]
        if not finite:
            raise ValueError("at least one finite calibration statistic is required")
        tensor = torch.tensor(finite, dtype=torch.float64)
        threshold = float(torch.quantile(tensor, float(quantile), interpolation="higher"))
        return cls(
            threshold=threshold,
            quantile=float(quantile),
            rule_hash=str(rule_hash),
            n_samples=len(finite),
        )

    @property
    def viewing_rule_hash(self) -> str:
        return self._viewing_rule_hash

    def threshold(self) -> float:
        return self._threshold

    @property
    def quantile(self) -> float:
        return self._quantile

    @property
    def calibration_sample_count(self) -> int:
        return self._n_samples

    def assess(
        self,
        statistics: Sequence[float],
        *,
        version: str,
        views: Sequence[ResidualView] | None = None,
        rule: ViewingRule | None = None,
        interference_views: Sequence[ResidualView] | None = None,
    ) -> CheckVerdict:
        """Assess one test sequence against the frozen threshold.

        With ``views``/``rule`` the whole-image mean and local maximum are recomputed under
        the rule and reported; without them the sequence statistics are compared directly
        (the scalar form of the frozen interface).
        """
        if views is not None:
            if rule is None:
                raise ValueError("rule is required when views are supplied")
            return self.assess_views(
                views,
                version=version,
                rule=rule,
                interference_views=interference_views,
            )
        values = [float(item) for item in statistics]
        finite = [item for item in values if math.isfinite(item)]
        base: dict[str, float] = {
            "threshold": self._threshold,
            "quantile": self._quantile,
            "calibration_samples": float(self._n_samples),
            "n_statistics": float(len(finite)),
        }
        if not finite:
            return CheckVerdict(
                status="insufficient_information",
                interference_explainable=False,
                local_residual_positions=(),
                calibration_version=version,
                statistics={**base, "max_sequence_statistic": float("nan")},
                viewing_rule_hash=self._viewing_rule_hash,
            )
        maximum = max(finite)
        status = "model_inadequate" if maximum > self._threshold else "compatible"
        return CheckVerdict(
            status=status,
            interference_explainable=False,
            local_residual_positions=(),
            calibration_version=version,
            statistics={**base, "max_sequence_statistic": maximum},
            viewing_rule_hash=self._viewing_rule_hash,
        )

    def assess_views(
        self,
        views: Sequence[ResidualView],
        *,
        version: str,
        rule: ViewingRule,
        interference_views: Sequence[ResidualView] | None = None,
    ) -> CheckVerdict:
        """Assess a test sequence's residual views under the frozen viewing rule.

        Both aggregates are always reported: ``global_mean_abs_standardised_residual`` and
        ``local_max_abs_standardised_residual``.  A raw statistic above the frozen threshold
        is ``model_inadequate`` unless the declared interference renders the corrected
        views compatible, in which case the verdict is ``compatible`` with
        ``interference_explainable=True``.
        """
        if rule.rule_hash() != self._viewing_rule_hash:
            raise ViewingRuleMismatchError(
                "calibration and test must use the identical viewing rule: "
                f"{rule.rule_hash()} != {self._viewing_rule_hash}"
            )
        selected = select_views(views, rule=rule)
        per_view = [windowed_statistics(view, rule=rule) for view in selected]
        required = float(max(1, int(rule.min_valid_samples)))
        # every declared view must carry enough valid samples; otherwise the sequence
        # simply has no information under this rule instead of a silent partial view set
        if any(item["n_valid_samples"] < required for item in per_view):
            total_valid = sum(item["n_valid_samples"] for item in per_view)
            return CheckVerdict(
                status="insufficient_information",
                interference_explainable=False,
                local_residual_positions=(),
                calibration_version=version,
                statistics={
                    "threshold": self._threshold,
                    "quantile": self._quantile,
                    "calibration_samples": float(self._n_samples),
                    "n_valid_samples": float(total_valid),
                    "global_mean_abs_standardised_residual": float("nan"),
                    "local_max_abs_standardised_residual": float("-inf"),
                    "sequence_statistic": float("-inf"),
                },
                viewing_rule_hash=self._viewing_rule_hash,
            )
        finite_stats = [item for item in per_view if math.isfinite(item["statistic"])]
        total_valid = sum(item["n_valid_samples"] for item in per_view)
        if total_valid > 0.0:
            global_mean = (
                sum(
                    item["global_mean_abs_standardised_residual"] * item["n_valid_samples"]
                    for item in per_view
                )
                / total_valid
            )
        else:
            global_mean = float("nan")
        local_values = [
            item["local_max_abs_standardised_residual"]
            for item in per_view
            if math.isfinite(item["local_max_abs_standardised_residual"])
        ]
        local_max = max(local_values) if local_values else float("-inf")
        raw_statistic = (
            max(item["statistic"] for item in finite_stats) if finite_stats else float("-inf")
        )
        statistics: dict[str, float] = {
            "threshold": self._threshold,
            "quantile": self._quantile,
            "calibration_samples": float(self._n_samples),
            "n_valid_samples": float(total_valid),
            "global_mean_abs_standardised_residual": global_mean,
            "local_max_abs_standardised_residual": local_max,
            "sequence_statistic": raw_statistic,
        }

        if not finite_stats:
            return CheckVerdict(
                status="insufficient_information",
                interference_explainable=False,
                local_residual_positions=(),
                calibration_version=version,
                statistics=statistics,
                viewing_rule_hash=self._viewing_rule_hash,
            )

        if raw_statistic <= self._threshold:
            return CheckVerdict(
                status="compatible",
                interference_explainable=False,
                local_residual_positions=(),
                calibration_version=version,
                statistics=statistics,
                viewing_rule_hash=self._viewing_rule_hash,
            )

        positions = _local_positions(selected, rule=rule, threshold=self._threshold)
        if interference_views is not None:
            corrected = select_views(interference_views, rule=rule)
            corrected_values = [
                windowed_statistics(view, rule=rule)["statistic"] for view in corrected
            ]
            corrected_finite = [value for value in corrected_values if math.isfinite(value)]
            corrected_statistic = max(corrected_finite) if corrected_finite else float("-inf")
            statistics["interference_statistic"] = corrected_statistic
            if math.isfinite(corrected_statistic) and corrected_statistic <= self._threshold:
                return CheckVerdict(
                    status="compatible",
                    interference_explainable=True,
                    local_residual_positions=positions,
                    calibration_version=version,
                    statistics=statistics,
                    viewing_rule_hash=self._viewing_rule_hash,
                )
        return CheckVerdict(
            status="model_inadequate",
            interference_explainable=False,
            local_residual_positions=positions,
            calibration_version=version,
            statistics=statistics,
            viewing_rule_hash=self._viewing_rule_hash,
        )

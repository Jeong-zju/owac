"""P2 RAW motion evaluation: metrics, calibrated risk table and the four comparison arms.

Metrics (``docs/prompts/n01_v1_0/P2_RAW_MOTION.md`` section 5)
--------------------------------------------------------------
End-point error (EPE) with quantiles, occlusion/mismatch recognition, risk-bucket
coverage, static false rejection, CFA phase consistency and end-to-end p50/p95
latency plus device memory.  Errors are computed in packed cells and reported in
both packed cells and sensor pixels (``sensor_per_packed``: 2 for ``cfa4``, 1 for
``rgb3``); the conversion lives in the training module and is reused here.

The four comparison arms
------------------------
* ``no_alignment`` -- zero displacement, sampled through the same RAW adapter.
* ``old_simplified`` -- the old simplified global alignment: one integer packed
  shift per pair estimated by normalised correlation on the real RAW planes.
  It sees only the two frames and never the analytic flow.  The old
  ``align_planes``/``align_features`` path applies a *given* offset field (and
  its ``variance`` shortcut is explicitly not reused, P2 section 3); this arm
  supplies the offsets that path would have needed.
* ``sea_raft`` -- the actual pinned SEA-RAFT adapter.
* ``oracle`` -- the analytic pull truth as an upper bound.  It is reported in its
  own section, is never constructed from an inference call and never enters an
  estimator: the oracle arm computes metrics directly from the supervision.

``fixed_support`` (the evaluated arm's own validity) and ``oracle_support`` (the
supervision's valid mask) are diagnostic labels on the same numbers, not
inference inputs.  Risk calibration is fitted on the ``calibration`` region only,
validated on ``validation`` and reported on ``test`` (four-way split contract).

Device contract
---------------
The supervision tensors are built on CPU by the generator and define the metric
device of a pair.  Model outputs may live on ``cuda``; every combination of a
model output with the supervision goes through :func:`_align_to` (moved to the
reference tensor's device), never through a dtype-only ``.to(torch.float32)``,
which keeps the source device and fails with a cross-device error.  The
calibration sampling path aligns the supervision to the model's own output
device instead, so no call site assumes ``torch.cuda.current_device()``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from owac.backends.raw_correspondence import (
    SENSOR_PER_PACKED,
    CorrespondenceEstimator,
    RawNormalization,
    RawSensorFrame,
    RiskCalibrationTable,
)
from owac.raw.n01_motion_data import (
    MotionDataGenerator,
    P2MotionNoiseConfig,
)
from owac.training.n01_motion import (
    MotionPairProvider,
    MotionSampleSpec,
    MotionTrainingConfig,
    MotionValueCalibration,
    build_estimator,
    flow_target_from_supervision,
    load_motion_training_config,
    packed_to_sensor,
)

__all__ = [
    "BUCKET_LABELS",
    "FITTING_SPLIT",
    "GROUP_LABELS",
    "SCHEMA",
    "ErrorCoverageTable",
    "EvaluationConfigError",
    "roc_auc",
    "run_evaluation",
]

SCHEMA = "owac.n01_full.p2_motion_eval.v1"
FITTING_SPLIT = "calibration"
SELECTION_SPLIT = "validation"
EVALUATION_SPLIT = "test"
REGIONS = (FITTING_SPLIT, SELECTION_SPLIT, EVALUATION_SPLIT)

BUCKET_LABELS = {
    "fast_alignable": "快且可对齐 (fast and alignable)",
    "fast_hard": "快且困难 (fast but hard)",
    "slow_structural_change": "慢但结构改变 (slow, structural change)",
    "current_frame_anomaly": "当前单帧异常 (current single-frame anomaly)",
}
GROUP_LABELS = {
    "no_alignment": "无对齐 (no alignment)",
    "old_simplified": "旧简化对齐 (old simplified global alignment)",
    "sea_raft": "实际 SEA-RAFT (actual SEA-RAFT)",
    "oracle": "oracle 上界 (upper bound, listed separately, never an inference input)",
}
GROUP_ORDER = ("no_alignment", "old_simplified", "sea_raft", "oracle")

#: Cells sampled per metric accumulator; the per-pair records keep the full counts.
MAX_CELL_SAMPLES = 200_000
QUANTILES = (0.5, 0.9, 0.95)


class EvaluationConfigError(ValueError):
    """The evaluation configuration or data regions are not usable as declared."""


# ---------------------------------------------------------------------------
# pure metric helpers
# ---------------------------------------------------------------------------


def _align_to(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Move ``tensor`` onto ``reference``'s device, keeping its dtype.

    ``.to(torch.float32)`` alone keeps the source device; combining a CUDA model
    output with the CPU supervision therefore raises a cross-device error.  Every
    such combination goes through this helper, and the device is taken from a
    tensor that is already in the expression (the supervision or the model
    output), never from ``torch.cuda.current_device()``.
    """
    if tensor.device != reference.device:
        return tensor.to(device=reference.device)
    return tensor


def roc_auc(scores: Any, labels: Any) -> float:
    """Rank AUC of ``scores`` against a boolean ``labels`` (no sklearn dependency).

    Returns ``float('nan')`` when only one class is present: an AUC without both
    classes is undefined, not 1.0.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("roc_auc needs aligned scores and labels")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks over ties so tied scores contribute the chance value
    sorted_scores = scores[order]
    start = 0
    for index in range(1, scores.size + 1):
        if index == scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def error_quantiles(
    errors_sensor_px: Any, quantiles: Sequence[float] = QUANTILES
) -> dict[str, float]:
    """Quantiles of the per-cell EPE; an empty sample has no quantile."""
    values = np.asarray(errors_sensor_px, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {f"p{int(q * 100)}": float("nan") for q in quantiles}
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in quantiles}


def _sample_cells(values: torch.Tensor, *, limit: int = MAX_CELL_SAMPLES) -> np.ndarray:
    flat = values.detach().to(torch.float64).reshape(-1).cpu().numpy()
    if flat.size <= limit:
        return flat
    stride = int(math.ceil(flat.size / limit))
    return flat[::stride]


# ---------------------------------------------------------------------------
# calibration table: proxy -> calibrated risk + coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorCoverageTable:
    """Independent proxy -> (calibrated risk, coverage) table.

    Fitted on the ``calibration`` region only.  ``flow_risk`` (the author mixture
    scale, sensor pixels) and ``visibility`` (the mixture weight on the small
    scale) are binned; each cell's calibrated risk is the bin's
    ``nominal_coverage`` quantile of the observed EPE and its coverage is the
    observed fraction of cells within that radius.  The table implements the
    :class:`~owac.backends.raw_correspondence.RiskCalibrationTable` protocol and
    is the only thing that turns the proxies into a calibrated risk.
    """

    version: str
    risk_edges: tuple[float, ...]
    visibility_edges: tuple[float, ...]
    risk: torch.Tensor
    coverage: torch.Tensor
    margin_sensor_px: float
    nominal_coverage: float
    fitted_split: str
    cells: int
    empty_bins: int

    def __post_init__(self) -> None:
        if self.fitted_split != FITTING_SPLIT:
            raise EvaluationConfigError(
                f"the error/coverage table is fitted on {FITTING_SPLIT!r} only, got "
                f"{self.fitted_split!r}"
            )
        if self.risk.ndim != 2 or self.risk.shape != self.coverage.shape:
            raise ValueError("risk and coverage tables must be aligned 2-D tensors")
        if not (0.0 < float(self.nominal_coverage) < 1.0):
            raise ValueError("nominal_coverage must be in (0, 1)")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.risk.shape[0]), int(self.risk.shape[1])

    @classmethod
    def fit(
        cls,
        samples: Sequence[Mapping[str, Any]],
        *,
        split: str,
        bins: int = 4,
        nominal_coverage: float = 0.9,
        version: str = "p2-motion-error-coverage-v1",
    ) -> ErrorCoverageTable:
        """Fit from ``{'flow_risk','visibility','error','valid'}`` per-pair tensors.

        ``split`` is an explicit argument so a caller cannot fit on validation or
        test by accident: anything but ``calibration`` is refused.
        """
        if split != FITTING_SPLIT:
            raise EvaluationConfigError(
                f"the error/coverage table fits on the {FITTING_SPLIT!r} region only; got {split!r}"
            )
        if int(bins) < 1:
            raise ValueError("bins must be positive")
        if not (0.0 < float(nominal_coverage) < 1.0):
            raise ValueError("nominal_coverage must be in (0, 1)")
        risks: list[np.ndarray] = []
        visibilities: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        for sample in samples:
            valid = sample["valid"].detach().to(torch.bool).reshape(-1).cpu().numpy()
            if not bool(valid.any()):
                continue
            risks.append(
                _sample_cells(sample["flow_risk"])[valid[:: _stride_of(sample["flow_risk"])]]
            )
            visibilities.append(
                _sample_cells(sample["visibility"])[valid[:: _stride_of(sample["visibility"])]]
            )
            errors.append(_sample_cells(sample["error"])[valid[:: _stride_of(sample["error"])]])
        if not risks:
            raise EvaluationConfigError(
                "no calibration cell is valid; the error/coverage table cannot be fitted"
            )
        risk_values = np.concatenate(risks)
        visibility_values = np.concatenate(visibilities)
        error_values = np.concatenate(errors)
        quantile_grid = np.linspace(0.0, 1.0, int(bins) + 2)[1:-1]
        risk_edges = tuple(
            float(value) for value in np.unique(np.quantile(risk_values, quantile_grid))
        )
        visibility_edges = tuple(
            float(value) for value in np.unique(np.quantile(visibility_values, quantile_grid))
        )
        risk_index = _bin_index(risk_values, risk_edges)
        visibility_index = _bin_index(visibility_values, visibility_edges)
        risk_table = torch.full((int(bins), int(bins)), float("nan"), dtype=torch.float64)
        coverage_table = torch.full((int(bins), int(bins)), float("nan"), dtype=torch.float64)
        empty = 0
        flat = risk_index * int(bins) + visibility_index
        for cell in range(int(bins) * int(bins)):
            selected = error_values[flat == cell]
            if selected.size == 0:
                empty += 1
                continue
            radius = float(np.quantile(selected, float(nominal_coverage)))
            risk_table.reshape(-1)[cell] = radius
            coverage_table.reshape(-1)[cell] = float(np.mean(selected <= radius))
        if empty:
            fallback_radius = float(np.quantile(error_values, float(nominal_coverage)))
            fallback_coverage = float(np.mean(error_values <= fallback_radius))
            risk_table = torch.nan_to_num(risk_table, nan=fallback_radius)
            coverage_table = torch.nan_to_num(coverage_table, nan=fallback_coverage)
        return cls(
            version=str(version),
            risk_edges=risk_edges,
            visibility_edges=visibility_edges,
            risk=risk_table,
            coverage=coverage_table,
            margin_sensor_px=float(np.quantile(error_values, float(nominal_coverage))),
            nominal_coverage=float(nominal_coverage),
            fitted_split=split,
            cells=int(error_values.size),
            empty_bins=int(empty),
        )

    def map_proxy(
        self, *, flow_risk: torch.Tensor, visibility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Protocol entry point: per-cell calibrated risk and coverage, same shape."""
        if flow_risk.shape != visibility.shape:
            raise ValueError("flow_risk and visibility must share one packed grid")
        rows, cols = self.shape
        device = flow_risk.device
        risk_index = torch.bucketize(
            flow_risk.detach().to(torch.float64),
            torch.tensor(self.risk_edges, device=device),
        )
        visibility_index = torch.bucketize(
            visibility.detach().to(torch.float64),
            torch.tensor(self.visibility_edges, device=device),
        )
        risk_index = risk_index.clamp(0, rows - 1)
        visibility_index = visibility_index.clamp(0, cols - 1)
        risk = self.risk.to(flow_risk.device)[risk_index, visibility_index]
        coverage = self.coverage.to(flow_risk.device)[risk_index, visibility_index]
        return risk.to(flow_risk.dtype), coverage.to(flow_risk.dtype)

    def coverage_at(self, *, flow_risk: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`map_proxy` for callers that only want the coverage map."""
        return self.map_proxy(flow_risk=flow_risk, visibility=visibility)[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "risk_edges": list(self.risk_edges),
            "visibility_edges": list(self.visibility_edges),
            "risk": self.risk.tolist(),
            "coverage": self.coverage.tolist(),
            "margin_sensor_px": float(self.margin_sensor_px),
            "nominal_coverage": float(self.nominal_coverage),
            "fitted_split": self.fitted_split,
            "cells": int(self.cells),
            "empty_bins": int(self.empty_bins),
        }


def _stride_of(tensor: torch.Tensor) -> int:
    size = int(tensor.numel())
    return max(1, int(math.ceil(size / MAX_CELL_SAMPLES)))


def _bin_index(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=np.float64), values).astype(np.int64)


# ---------------------------------------------------------------------------
# alignment arms
# ---------------------------------------------------------------------------


class _ZeroFlow(nn.Module):
    """Test-double with the vendored RAFT signature: the ``no_alignment`` arm."""

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        flow = torch.zeros(
            image1.shape[0], 2, image1.shape[-2], image1.shape[-1], device=image1.device
        )
        info = torch.zeros(
            image1.shape[0], 4, image1.shape[-2], image1.shape[-1], device=image1.device
        )
        return {
            "final": flow,
            "flow": [flow],
            "info": [info],
            "nf": None if test_mode else [torch.zeros_like(flow[:, :1])],
        }


@dataclass(frozen=True)
class GroupMap:
    """One arm's alignment of an already-arrived pair.

    ``flow`` is the shared packed-cell displacement field; ``channel_flow`` is
    the per-channel field when the arm has one (the oracle upper bound, or a
    per-phase simplified estimate).  ``risk``/``coverage`` carry calibrated
    values when a table is supplied.  The oracle arm is marked ``oracle`` and is
    only ever compared, never fed to an estimator.
    """

    flow: torch.Tensor
    valid: torch.Tensor
    channel_flow: torch.Tensor | None = None
    risk: torch.Tensor | None = None
    coverage: torch.Tensor | None = None
    risk_kind: str = "unavailable"
    latency_ms: float | None = None
    support_label: str = "fixed_support"
    oracle: bool = False


def _simplified_global_shift(
    reference_values: torch.Tensor, source_values: torch.Tensor, *, search_cells: int
) -> tuple[int, int]:
    """Integer packed *pull* displacement minimising the mean square difference.

    The returned ``(pull_y, pull_x)`` is used exactly like the adapter's packed
    flow: reference cell ``(r, c)`` is compared with the source sample at
    ``(r + pull_y, c + pull_x)``.  Only the two frames enter; no analytic flow and
    no per-image min/max scaling.
    """
    if reference_values.shape != source_values.shape:
        raise ValueError("the simplified alignment needs two equal-shape packed frames")
    channels, rows, cols = reference_values.shape
    reference = reference_values.to(torch.float64)
    source = source_values.to(torch.float64)
    best = (0, 0)
    best_score = float("inf")
    for pull_y in range(-int(search_cells), int(search_cells) + 1):
        for pull_x in range(-int(search_cells), int(search_cells) + 1):
            y0, y1 = max(0, -pull_y), min(rows, rows - pull_y)
            x0, x1 = max(0, -pull_x), min(cols, cols - pull_x)
            if y1 - y0 < rows // 2 or x1 - x0 < cols // 2:
                continue
            reference_crop = reference[:, y0:y1, x0:x1]
            source_crop = source[:, y0 + pull_y : y1 + pull_y, x0 + pull_x : x1 + pull_x]
            score = float(((reference_crop - source_crop) ** 2).mean())
            if score < best_score - 1e-12:
                best_score = score
                best = (pull_y, pull_x)
    return best


def _inside_mask(rows: int, cols: int, dy: int, dx: int, *, device: torch.device) -> torch.Tensor:
    row = torch.arange(rows, device=device)
    col = torch.arange(cols, device=device)
    return ((row + dy).clamp(0, rows - 1) == row + dy).view(-1, 1) & (
        (col + dx).clamp(0, cols - 1) == col + dx
    ).view(1, -1)


# ---------------------------------------------------------------------------
# evaluation session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Arm:
    name: str
    label: str
    oracle: bool = False


class MotionEvaluator:
    """Runs the four arms on the declared regions and writes the P2 receipt."""

    def __init__(
        self,
        config: MotionTrainingConfig,
        *,
        device: str | torch.device = "cpu",
        checkpoint: str | Path | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.data_config: P2MotionNoiseConfig = config.data.resolved_data_config()
        self.input_mode = config.data.input_mode or self.data_config.geometry.input_mode
        self.geometry = self.data_config.geometry_for(self.input_mode)
        self.mode = "cfa4" if self.geometry.is_cfa else "rgb3"
        self.sensor_per_packed = int(SENSOR_PER_PACKED[self.mode])
        if self.mode != config.model.mode:
            raise EvaluationConfigError(
                f"the data geometry is {self.mode} but model.mode is {config.model.mode!r}"
            )
        self.calibration = MotionValueCalibration.from_data_config(
            self.data_config, exposure_correction=config.data.exposure_correction
        )
        self.normalization = self.calibration.normalization(
            self.data_config.noise.calibration_version
        )
        self.checkpoint_report = _load_checkpoint(
            config,
            self.data_config,
            checkpoint,
            device=self.device,
            normalization=self.normalization,
        )
        self.estimator: CorrespondenceEstimator | None = None
        self.estimators: dict[str, CorrespondenceEstimator] = {}
        self.generator = MotionDataGenerator(config=self.data_config)
        self.provider = None

    # -- construction --------------------------------------------------------------------

    def _build_estimators(self) -> None:
        self.estimator = build_estimator(
            self.config,
            self.data_config,
            device=self.device,
            normalization=self.normalization,
        )
        if self.checkpoint_report["state"] is not None:
            self.estimator.model.load_state_dict(self.checkpoint_report["state"])
        zero = _ZeroFlow()
        self.estimators = {
            "sea_raft": self.estimator,
            "no_alignment": CorrespondenceEstimator(
                mode=self.mode,
                model=zero,
                iters=self.config.model.iters,
                device=str(self.device),
                normalization=self.normalization,
            ),
        }
        self.provider = MotionPairProvider(
            self.generator,
            self.calibration,
            self.estimator,
            input_mode=self.input_mode,
            mode=self.mode,
            cache_entries=self.config.data.cache_entries,
        )

    def specs(self, region: str, *, limit: int) -> tuple[MotionSampleSpec, ...]:
        """The region's pairs under ``limit``, case-balanced so no case disappears.

        A plain prefix cut would drop whole cases (and therefore whole evaluation
        buckets) when the cap is below the region size; every declared case keeps
        at least one pair, so a hard condition can never be silently deleted from
        the report.
        """
        if region not in REGIONS:
            raise EvaluationConfigError(f"unknown region {region!r}; expected {REGIONS}")
        region_spec = self.config.data.region(region)
        all_specs = region_spec.specs(region)
        limit = int(limit)
        if limit < len(all_specs):
            cases = list(region_spec.cases)
            per_case = max(1, limit // len(cases))
            selected: list[MotionSampleSpec] = []
            for case in cases:
                members = [spec for spec in all_specs if spec.case == case]
                selected.extend(members[:per_case])
            all_specs = tuple(selected)
        for spec in all_specs:
            if spec.split != region:
                raise EvaluationConfigError(
                    f"a {region!r} spec declares split {spec.split!r}; the four-way split must "
                    "stay explicit"
                )
        return all_specs

    def _corrected_pair(self, spec: MotionSampleSpec):
        pair = self.generator.sample_pair(
            split=spec.split,
            case=spec.case,
            sequence=spec.sequence,
            history_offset=int(spec.history_offset),
            input_mode=self.input_mode,
        )
        reference, source, exposure_known = self.calibration.corrected_frames(
            pair.visible.reference, pair.visible.source, pair.visible.metadata
        )
        return pair, reference, source, exposure_known

    # -- arms ------------------------------------------------------------------------------

    def arm_map(
        self,
        arm: str,
        reference: RawSensorFrame,
        source: RawSensorFrame,
        target,
        *,
        table: ErrorCoverageTable | None,
    ) -> GroupMap:
        # the supervision defines the metric device of this pair; every arm's
        # tensors are moved onto it before they are combined with the truth
        metric = target.channel_flow
        if arm == "sea_raft":
            estimator = self.estimators["sea_raft"]
            self._maybe_calibrate(estimator, table)
            if self.estimator is None:  # pragma: no cover - _build_estimators always runs first
                raise EvaluationConfigError("the evaluation estimators were not built")
            started = time.perf_counter()
            result = estimator.estimate(
                reference, source, iters=self.config.model.iters, taps=False
            )
            latency = (time.perf_counter() - started) * 1000.0
            risk = (
                result.calibrated_risk if result.calibrated_risk is not None else result.flow_risk
            )
            return GroupMap(
                flow=_align_to(result.packed_flow, metric).to(torch.float32),
                valid=_align_to(result.valid.all(dim=0), metric),
                risk=None if risk is None else _align_to(risk.detach(), metric).to(torch.float32),
                coverage=None
                if result.coverage is None
                else _align_to(result.coverage, metric).to(torch.float32),
                risk_kind=result.risk_kind,
                latency_ms=latency,
            )
        if arm == "no_alignment":
            estimator = self.estimators["no_alignment"]
            result = estimator.estimate(reference, source, iters=0, taps=False)
            return GroupMap(
                flow=_align_to(result.packed_flow, metric).to(torch.float32),
                valid=_align_to(result.valid.all(dim=0), metric),
                risk_kind="unavailable",
            )
        if arm == "old_simplified":
            search_cells = int(
                math.ceil(self.config.evaluation.simplified_search_px / self.sensor_per_packed)
            )
            reference_values = reference.values
            source_values = source.values
            dy, dx = _simplified_global_shift(
                reference_values, source_values, search_cells=search_cells
            )
            rows, cols = self.geometry.packed_shape
            flow = torch.zeros(2, rows, cols, dtype=torch.float32)
            flow[0] = float(dy)
            flow[1] = float(dx)
            valid = _inside_mask(rows, cols, dy, dx, device=metric.device)
            return GroupMap(flow=_align_to(flow, metric), valid=valid, risk_kind="unavailable")
        if arm == "oracle":
            channel_flow = target.channel_flow.to(torch.float32)
            return GroupMap(
                flow=target.flow.to(torch.float32),
                valid=target.valid,
                channel_flow=channel_flow,
                risk_kind="unavailable",
                support_label="oracle_support_diagnostic",
                oracle=True,
            )
        raise EvaluationConfigError(f"unknown comparison arm {arm!r}")

    def _maybe_calibrate(
        self, estimator: CorrespondenceEstimator, table: ErrorCoverageTable | None
    ) -> None:
        """Install the fitted table on the adapter's declared calibration slot.

        The adapter takes its table at construction time; the table is fitted from
        the adapter's own uncalibrated proxies first, so it is attached here.  The
        protocol instance is validated before it is installed.
        """
        if table is not None and not isinstance(table, RiskCalibrationTable):
            raise EvaluationConfigError(
                "the fitted error/coverage table must implement RiskCalibrationTable"
            )
        estimator._risk_calibration = table


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@dataclass
class _Accumulator:
    errors_sensor: list[np.ndarray] = field(default_factory=list)
    errors_packed: list[np.ndarray] = field(default_factory=list)
    risks: list[np.ndarray] = field(default_factory=list)
    coverages: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    cells: int = 0
    predicted_valid: int = 0
    supervision_valid: int = 0
    threshold: float = 1.0

    def add(
        self,
        error_sensor: torch.Tensor,
        error_packed: torch.Tensor,
        *,
        mask: torch.Tensor,
        risk: torch.Tensor | None,
        coverage: torch.Tensor | None,
        invalid_label: torch.Tensor,
        supervision_valid: torch.Tensor,
        predicted_valid: torch.Tensor,
    ) -> None:
        self.cells += int(mask.numel())
        self.supervision_valid += int(supervision_valid.sum())
        self.predicted_valid += int(predicted_valid.sum())
        if not bool(mask.any()):
            return
        self.errors_sensor.append(_sample_cells(error_sensor[mask]))
        self.errors_packed.append(_sample_cells(error_packed[mask]))
        if risk is not None:
            self.risks.append(_sample_cells(risk[mask]))
        if coverage is not None:
            self.coverages.append(_sample_cells(coverage[mask]))
        self.labels.append(_sample_cells(invalid_label[mask].to(torch.float64)))

    def summary(self) -> dict[str, Any]:
        if not self.errors_sensor:
            return {"cells": self.cells, "valid_cells": 0, "epe_sensor_px": None}
        errors = np.concatenate(self.errors_sensor)
        packed = np.concatenate(self.errors_packed)
        labels = np.concatenate(self.labels).astype(bool)
        summary = {
            "cells": int(self.cells),
            "valid_cells": int(errors.size),
            "supervision_valid_cells": int(self.supervision_valid),
            "predicted_valid_cells": int(self.predicted_valid),
            "epe_sensor_px": float(errors.mean()),
            "epe_packed": float(packed.mean()),
            "error_quantiles_sensor_px": error_quantiles(errors),
        }
        if self.risks:
            risks = np.concatenate(self.risks)
            summary["risk_mean_sensor_px"] = float(risks.mean())
            summary["occlusion_auc"] = roc_auc(risks, labels)
            summary["invalid_label_fraction"] = float(labels.mean())
            summary["occlusion_precision_recall"] = _precision_recall(
                risks, labels, threshold=float(self.threshold)
            )
        if self.coverages:
            coverages = np.concatenate(self.coverages)
            summary["coverage_mean"] = float(coverages.mean())
            summary["coverage_standard_deviation"] = float(coverages.std())
        return summary

    def with_threshold(self, threshold: float) -> _Accumulator:
        self.threshold = float(threshold)
        return self


def _precision_recall(
    scores: np.ndarray, labels: np.ndarray, *, threshold: float
) -> dict[str, float]:
    predicted = scores > float(threshold)
    true_positive = int((predicted & labels).sum())
    false_positive = int((predicted & ~labels).sum())
    false_negative = int((~predicted & labels).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold_sensor_px": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_cells": int(predicted.sum()),
    }


def _pair_error(
    predicted_flow: torch.Tensor,
    target,
    *,
    predicted_valid: torch.Tensor,
    group_map: GroupMap,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel EPE (sensor px and packed), the combined mask and the invalid label."""
    channel_flow = target.channel_flow.to(torch.float32)
    channel_valid = target.channel_valid
    predicted_valid = _align_to(predicted_valid, channel_flow)
    if group_map.channel_flow is not None:
        predicted = _align_to(group_map.channel_flow, channel_flow).to(torch.float32)
        error_packed = torch.linalg.vector_norm(predicted - channel_flow, dim=1)
    else:
        predicted = _align_to(predicted_flow, channel_flow).to(torch.float32)
        error_packed = torch.linalg.vector_norm(predicted.unsqueeze(0) - channel_flow, dim=1)
    mask = channel_valid & predicted_valid.unsqueeze(0)
    error_sensor = packed_to_sensor(error_packed, int(target.sensor_per_packed))
    invalid_label = ~channel_valid
    return error_sensor, error_packed, mask, invalid_label


def _cfa_phase_consistency(
    predicted_flow: torch.Tensor, target, *, sensor_per_packed: int
) -> float | None:
    """Spread of per-channel mean (prediction - truth) in sensor pixels, or None for rgb3."""
    if target.channel_valid.shape[0] != 4:
        return None
    channel_flow = target.channel_flow.to(torch.float32)
    differences = (
        _align_to(predicted_flow, channel_flow).to(torch.float32).unsqueeze(0) - channel_flow
    )
    result = 0.0
    for channel in range(4):
        valid = target.channel_valid[channel]
        if not bool(valid.any()):
            continue
        mean = differences[channel][:, valid].mean(dim=1) * float(sensor_per_packed)
        result = max(result, float(mean.abs().max()))
    return result


def _load_checkpoint(
    config: MotionTrainingConfig,
    data_config: P2MotionNoiseConfig,
    checkpoint: str | Path | None,
    *,
    device: str | torch.device,
    normalization: Mapping[str, RawNormalization],
) -> dict[str, Any]:
    if checkpoint is None:
        return {
            "path": None,
            "state": None,
            "update": None,
            "sha256": None,
            "parameter_hash": None,
        }
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise EvaluationConfigError(f"{path} is not a P2 motion training checkpoint")
    if payload.get("schema") != "owac.n01_full.p2_motion_checkpoint.v1":
        raise EvaluationConfigError(f"{path} carries schema {payload.get('schema')!r}")
    if payload.get("core_fingerprint") != config.core_fingerprint:
        raise EvaluationConfigError(
            f"{path} was written under a different run definition; its validation-selected "
            "checkpoint cannot be evaluated under this configuration"
        )
    return {
        "path": str(path),
        "state": payload["model"],
        "update": int(payload.get("update", -1)),
        "sha256": _sha256_file(path),
        "parameter_hash": payload.get("parameter_hash"),
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# top-level evaluation
# ---------------------------------------------------------------------------


def _validate_buckets(config: MotionTrainingConfig, cases: Sequence[str]) -> None:
    buckets = config.evaluation.buckets
    missing = sorted(set(cases) - set(buckets))
    if missing:
        raise EvaluationConfigError(
            f"every declared case needs an evaluation bucket; missing {missing}"
        )
    used = {buckets[case] for case in cases}
    absent = sorted(set(BUCKET_LABELS) - used)
    if absent:
        raise EvaluationConfigError(
            "the four declared buckets must all appear; no case maps to " + ", ".join(absent)
        )


def run_evaluation(
    config: MotionTrainingConfig,
    *,
    output: str | Path,
    device: str | torch.device = "cpu",
    checkpoint: str | Path | None = None,
    config_path: str | Path | None = None,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Evaluate the four arms on calibration/validation/test and write the receipt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    evaluator = MotionEvaluator(config, device=device, checkpoint=checkpoint)
    evaluator._build_estimators()
    cases = list(config.data.train.cases)
    _validate_buckets(config, cases)
    evaluation = config.evaluation
    limits = {
        FITTING_SPLIT: int(evaluation.max_calibration_pairs),
        SELECTION_SPLIT: int(evaluation.max_validation_pairs),
        EVALUATION_SPLIT: int(evaluation.max_test_pairs),
    }
    if max_pairs is not None:
        limits = {region: int(max_pairs) for region in REGIONS}
    pairs = {region: evaluator.specs(region, limit=limits[region]) for region in REGIONS}

    # 1) fit the error/coverage table on calibration only
    table_samples = _collect_calibration_samples(evaluator, pairs[FITTING_SPLIT])
    table = ErrorCoverageTable.fit(
        table_samples,
        split=FITTING_SPLIT,
        bins=int(evaluation.risk_bins),
        nominal_coverage=float(evaluation.risk_quantile),
        version=f"p2-motion-error-coverage-{evaluator.checkpoint_report.get('update') or 'init'}",
    )
    (output / "risk_calibration.json").write_text(
        json.dumps(table.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    # 2) evaluate every arm on validation (coverage validation) and test (final report)
    records: list[dict[str, Any]] = []
    per_region: dict[str, Any] = {}
    for region in (SELECTION_SPLIT, EVALUATION_SPLIT):
        region_records = _evaluate_region(evaluator, region, pairs[region], table)
        records.extend(region_records)
        per_region[region] = _aggregate_region(
            region_records, config, region=region, table=table, evaluator=evaluator
        )
    static = {
        region: _static_false_rejection(evaluator, region, pairs[region])
        for region in (SELECTION_SPLIT, EVALUATION_SPLIT)
    }
    latency = _latency_report(evaluator, pairs[EVALUATION_SPLIT] or pairs[SELECTION_SPLIT])
    buckets = _bucket_report(records, config, evaluator.sensor_per_packed)
    (output / "pairs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    receipt = {
        "run_mode": "evaluation",
        "schema": SCHEMA,
        "git_sha": _git_sha(),
        "config_fingerprint": config.core_fingerprint,
        "config_path": str(config_path or "configs/encoders/n01_full/p2_motion_train.json"),
        "config": config.as_dict(),
        "device": str(evaluator.device),
        "torch": torch.__version__,
        "python": __import__("sys").version,
        "numpy": np.__version__,
        "mode": evaluator.mode,
        "depth": config.model.depth,
        "iters": config.model.iters,
        "sensor_per_packed": evaluator.sensor_per_packed,
        "checkpoint": evaluator.checkpoint_report["path"],
        "checkpoint_sha256": evaluator.checkpoint_report["sha256"],
        "checkpoint_update": evaluator.checkpoint_report["update"],
        "parameter_hash": evaluator.checkpoint_report["parameter_hash"],
        "risk_calibration": {
            "version": table.version,
            "fitted_split": table.fitted_split,
            "cells": table.cells,
            "empty_bins": table.empty_bins,
            "nominal_coverage": table.nominal_coverage,
            "margin_sensor_px": table.margin_sensor_px,
            "path": str(output / "risk_calibration.json"),
        },
        "regions": {
            "fitted_on": FITTING_SPLIT,
            "validated_on": SELECTION_SPLIT,
            "reported_on": EVALUATION_SPLIT,
            "pairs": {region: len(pairs[region]) for region in REGIONS},
            "test_used_only_for_reporting": True,
        },
        "group_labels": GROUP_LABELS,
        "bucket_labels": BUCKET_LABELS,
        "oracle": {
            "role": "upper bound only",
            "listed_separately": True,
            "used_as_inference_input": False,
        },
        "support_labels": {
            "fixed_support": "the evaluated arm's own validity",
            "oracle_support": "the supervision's valid mask; diagnostic label only",
        },
        "regions_report": per_region,
        "buckets": buckets,
        "static_false_rejection": static,
        "latency": latency,
        "provider": evaluator.provider.receipt(),
        "wall_seconds": time.perf_counter() - started,
        "notes": config.notes,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    return receipt


def _collect_calibration_samples(
    evaluator: MotionEvaluator, specs: Sequence[MotionSampleSpec]
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not specs:
        raise EvaluationConfigError(
            "the calibration region is empty; the risk table cannot be fitted"
        )
    if evaluator.estimator is None:
        raise EvaluationConfigError("the evaluation estimators were not built")
    for spec in specs:
        pair, reference, source, _ = evaluator._corrected_pair(spec)
        target = flow_target_from_supervision(
            pair.supervision, evaluator.geometry, mode=evaluator.mode
        )
        result = evaluator.estimator.estimate(
            reference, source, iters=evaluator.config.model.iters, taps=False
        )
        predicted_valid = result.valid.all(dim=0)
        flow = result.packed_flow.to(torch.float32)
        # the model output owns the device here; the CPU supervision is moved onto
        # it rather than assuming the output is on the CPU (or on cuda:current)
        channel_flow = _align_to(target.channel_flow, flow).to(torch.float32)
        channel_valid = _align_to(target.channel_valid, flow)
        error_packed = torch.linalg.vector_norm(flow.unsqueeze(0) - channel_flow, dim=1)
        error_sensor = error_packed * float(evaluator.sensor_per_packed)
        valid = channel_valid & predicted_valid.unsqueeze(0)
        samples.append(
            {
                "flow_risk": result.flow_risk,
                "visibility": result.visibility,
                "error": error_sensor.mean(dim=0),
                "valid": valid.any(dim=0),
            }
        )
    return samples


def _evaluate_region(
    evaluator: MotionEvaluator,
    region: str,
    specs: Sequence[MotionSampleSpec],
    table: ErrorCoverageTable,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not specs:
        raise EvaluationConfigError(f"the {region} region is empty")
    for spec in specs:
        pair, reference, source, exposure_known = evaluator._corrected_pair(spec)
        target = flow_target_from_supervision(
            pair.supervision, evaluator.geometry, mode=evaluator.mode
        )
        for arm in GROUP_ORDER:
            group_map = evaluator.arm_map(arm, reference, source, target, table=table)
            error_sensor, error_packed, mask, invalid_label = _pair_error(
                group_map.flow,
                target,
                predicted_valid=group_map.valid,
                group_map=group_map,
            )
            valid_map = mask.any(dim=0)
            risk = None if group_map.risk is None else group_map.risk
            coverage = None if group_map.coverage is None else group_map.coverage
            accumulator = _Accumulator().with_threshold(
                float(evaluator.config.evaluation.risk_tolerance_sensor_px)
            )
            accumulator.add(
                error_sensor.mean(dim=0),
                error_packed.mean(dim=0),
                mask=valid_map,
                risk=risk,
                coverage=coverage,
                invalid_label=invalid_label.any(dim=0),
                supervision_valid=target.valid,
                predicted_valid=group_map.valid,
            )
            summary = accumulator.summary()
            if risk is not None:
                summary.update(
                    _risk_radius_coverage(
                        error_sensor.mean(dim=0),
                        risk,
                        valid_map,
                        fitted_coverage=coverage,
                    )
                )
            records.append(
                {
                    "region": region,
                    "case": spec.case,
                    "sequence": spec.sequence,
                    "history_offset": int(spec.history_offset),
                    "bucket": evaluator.config.evaluation.bucket_of(spec.case),
                    "arm": arm,
                    "support_label": group_map.support_label,
                    "risk_kind": group_map.risk_kind,
                    "exposure_known": bool(exposure_known),
                    "oracle_input": False,
                    "predicted_mean_flow_packed": group_map.flow.mean(dim=(1, 2)).tolist(),
                    "cfa_phase_consistency_sensor_px": (
                        None
                        if group_map.oracle
                        else _cfa_phase_consistency(
                            group_map.flow, target, sensor_per_packed=evaluator.sensor_per_packed
                        )
                    ),
                    "latency_ms": group_map.latency_ms,
                    **summary,
                }
            )
    return records


def _risk_radius_coverage(
    error_sensor: torch.Tensor,
    risk: torch.Tensor,
    mask: torch.Tensor,
    *,
    fitted_coverage: torch.Tensor | None,
) -> dict[str, float]:
    """Observed vs predicted coverage of the calibrated risk radius on this pair."""
    if not bool(mask.any()):
        return {"coverage_observed": float("nan")}
    risk = _align_to(risk, error_sensor)
    observed = (error_sensor[mask] <= risk[mask]).to(torch.float64)
    result = {"coverage_observed": float(observed.mean())}
    if fitted_coverage is not None:
        fitted_coverage = _align_to(fitted_coverage, error_sensor)
        result["coverage_predicted"] = float(fitted_coverage[mask].to(torch.float64).mean())
    return result


def _aggregate_region(
    records: Sequence[Mapping[str, Any]],
    config: MotionTrainingConfig,
    *,
    region: str,
    table: ErrorCoverageTable,
    evaluator: MotionEvaluator,
) -> dict[str, Any]:
    return {
        "pairs": len(
            {(record["case"], record["sequence"], record["history_offset"]) for record in records}
        ),
        "arms": {
            arm: _aggregate_arm([record for record in records if record["arm"] == arm])
            for arm in GROUP_ORDER
        },
        "calibration_coverage_on_validation": (
            _coverage_check(records, table) if region == SELECTION_SPLIT else None
        ),
    }


def _aggregate_arm(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"pairs": 0}
    quantiles = ("p50", "p90", "p95")
    result: dict[str, Any] = {
        "pairs": len(records),
        "epe_sensor_px": float(np.mean([record["epe_sensor_px"] for record in records])),
        "epe_packed": float(np.mean([record["epe_packed"] for record in records])),
        "valid_cells": int(sum(record["valid_cells"] for record in records)),
        "cells": int(sum(record["cells"] for record in records)),
        "error_quantiles_sensor_px": {
            quantile: float(
                np.mean([record["error_quantiles_sensor_px"][quantile] for record in records])
            )
            for quantile in quantiles
        },
    }
    with_auc = [record for record in records if record.get("occlusion_auc") is not None]
    finite_auc = [
        record["occlusion_auc"] for record in with_auc if not math.isnan(record["occlusion_auc"])
    ]
    if finite_auc:
        result["occlusion_auc_mean"] = float(np.mean(finite_auc))
    with_coverage = [record for record in records if record.get("coverage_mean") is not None]
    if with_coverage:
        result["coverage_mean"] = float(
            np.mean([record["coverage_mean"] for record in with_coverage])
        )
    with_phase = [
        record["cfa_phase_consistency_sensor_px"]
        for record in records
        if record.get("cfa_phase_consistency_sensor_px") is not None
    ]
    if with_phase:
        result["cfa_phase_consistency_sensor_px_max"] = float(np.max(with_phase))
    latencies = [record["latency_ms"] for record in records if record.get("latency_ms") is not None]
    if latencies:
        result["latency_ms_p50"] = float(np.percentile(latencies, 50))
        result["latency_ms_p95"] = float(np.percentile(latencies, 95))
    return result


def _coverage_check(
    records: Sequence[Mapping[str, Any]], table: ErrorCoverageTable
) -> dict[str, Any]:
    """Validate the calibration table's coverage on the validation region.

    The table itself is fitted on calibration only; here the observed fraction of
    valid cells inside the calibrated risk radius is compared with the nominal
    coverage and with the fitted coverage map.  Nothing is refitted.
    """
    by_arm: dict[str, Any] = {}
    for arm in GROUP_ORDER:
        members = [record for record in records if record["arm"] == arm]
        arm_summary = _aggregate_arm(members)
        observed = [
            record["coverage_observed"]
            for record in members
            if record.get("coverage_observed") is not None
            and not math.isnan(record["coverage_observed"])
        ]
        predicted = [
            record["coverage_predicted"]
            for record in members
            if record.get("coverage_predicted") is not None
        ]
        if observed:
            arm_summary["coverage_observed_mean"] = float(np.mean(observed))
            arm_summary["coverage_nominal"] = float(table.nominal_coverage)
            arm_summary["coverage_abs_error"] = abs(
                float(np.mean(observed)) - float(table.nominal_coverage)
            )
        if predicted:
            arm_summary["coverage_fitted_map_mean"] = float(np.mean(predicted))
        by_arm[arm] = arm_summary
    return {
        "nominal_coverage": float(table.nominal_coverage),
        "by_arm": by_arm,
        "note": (
            "coverage is validated on the validation region with the table fitted on "
            "calibration; no refit happens here"
        ),
    }


def _static_false_rejection(
    evaluator: MotionEvaluator, region: str, specs: Sequence[MotionSampleSpec]
) -> dict[str, Any]:
    """Static (zero-displacement) pairs must not be rejected as unreliable."""
    if not specs:
        return {"pairs": 0}
    tolerance = float(evaluator.config.evaluation.static_false_rejection_sensor_px)
    flags: list[float] = []
    magnitudes: list[float] = []
    for spec in specs[: max(1, min(2, len(specs)))]:
        frames = evaluator.generator.frames(
            split=spec.split,
            case=spec.case,
            sequence=spec.sequence,
            length=2,
            input_mode=evaluator.input_mode,
        )
        scene = frames[-1]
        # rendering one scene state twice carries the same declared gain and the
        # same noise realisation, so the pair is exactly static
        reference = evaluator.generator.render(
            scene,
            split=spec.split,
            case=spec.case,
            sequence=spec.sequence,
            input_mode=evaluator.input_mode,
        )
        source = evaluator.generator.render(
            scene,
            split=spec.split,
            case=spec.case,
            sequence=spec.sequence,
            input_mode=evaluator.input_mode,
        )
        result = evaluator.estimators["sea_raft"].estimate(
            reference, source, iters=evaluator.config.model.iters, taps=False
        )
        magnitude = torch.linalg.vector_norm(result.packed_flow, dim=0) * float(
            evaluator.sensor_per_packed
        )
        valid = result.valid.all(dim=0)
        if not bool(valid.any()):
            continue
        rejected = (magnitude > tolerance) | ~valid
        flags.append(float(rejected[valid].to(torch.float64).mean()))
        magnitudes.append(float(magnitude[valid].mean()))
    if not flags:
        return {"pairs": 0, "note": "no static cell was valid"}
    return {
        "pairs": len(flags),
        "tolerance_sensor_px": tolerance,
        "false_rejection_rate": float(np.mean(flags)),
        "mean_predicted_displacement_sensor_px": float(np.mean(magnitudes)),
        "note": (
            "a static pair is rendered twice from the same scene state; the declared truth is zero"
        ),
    }


def _latency_report(
    evaluator: MotionEvaluator, specs: Sequence[MotionSampleSpec]
) -> dict[str, Any]:
    if not specs:
        return {"measured": False}
    spec = specs[0]
    _, reference, source, _ = evaluator._corrected_pair(spec)
    cuda = evaluator.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for _ in range(int(evaluator.config.evaluation.latency_warmup)):
        evaluator.estimator.estimate(
            reference, source, iters=evaluator.config.model.iters, taps=False
        )
    timings: list[float] = []
    for _ in range(int(evaluator.config.evaluation.latency_repeats)):
        started = time.perf_counter()
        evaluator.estimator.estimate(
            reference, source, iters=evaluator.config.model.iters, taps=False
        )
        if cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    parameters = sum(parameter.numel() for parameter in evaluator.estimator.model.parameters())
    return {
        "measured": True,
        "device": str(evaluator.device),
        "iters": evaluator.config.model.iters,
        "repeats": len(timings),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "parameter_count": int(parameters),
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None,
        "note": (
            "end-to-end estimate() wall time on one already-arrived pair, excluding data rendering"
        ),
    }


def _bucket_report(
    records: Sequence[Mapping[str, Any]], config: MotionTrainingConfig, sensor_per_packed: int
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for bucket, label in BUCKET_LABELS.items():
        members = [record for record in records if record["bucket"] == bucket]
        arms = {
            arm: _aggregate_arm([record for record in members if record["arm"] == arm])
            for arm in GROUP_ORDER
        }
        cases = sorted({record["case"] for record in members})
        if not members:
            raise EvaluationConfigError(
                f"the {bucket!r} bucket ({label}) has no evaluated pair; hard conditions must "
                "not be dropped from the report"
            )
        report[bucket] = {
            "label": label,
            "cases": cases,
            "pairs": len(
                {(record["region"], record["case"], record["sequence"]) for record in members}
            ),
            "arms": arms,
        }
    return report


def _git_sha() -> str:
    from owac.training.n01 import git_sha

    return git_sha()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N01 v1.0 P2 RAW motion evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-pairs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_motion_training_config(args.config)
    receipt = run_evaluation(
        config,
        output=args.output,
        device=args.device,
        checkpoint=args.checkpoint,
        config_path=args.config,
        max_pairs=args.max_pairs,
    )
    print(
        json.dumps(
            {
                "run_mode": "evaluation",
                "buckets": sorted(receipt["buckets"]),
                "arms": sorted(receipt["regions_report"][EVALUATION_SPLIT]["arms"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

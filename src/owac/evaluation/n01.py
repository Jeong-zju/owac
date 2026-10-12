"""Evaluation CLI for N01: mechanism matrix, controlled baseline comparison, falsification.

Step 5 of ``docs/prompts/DeepSeek_N01_implementation.md``.  This module loads a
training checkpoint, re-creates the shared probe head from
:mod:`owac.training.n01`, and evaluates the *same* readout on every method:

* ``--suite mechanism``: low/medium/high SNR x still/slow/fast x wide/narrow
  tolerance, plus ``sudden_appear``, ``direction_reversal``, ``occlusion_release``
  and ``crossing``.  Every condition has a fixed seed and reports position error
  in pixels, over-tolerance rate, first stable detection frame, recovery frames
  after a turn, the *sample age* (algorithmic lag) and the processing time
  separately, support-domain width, history age, merge/split counts, memory and
  state bytes.
* ``--suite comparison``: :func:`owac.baselines.n01.mechanism_baselines` plus the
  candidate, all trained with the identical probe head, optimiser, step count and
  byte-identical pre-generated batches (a controlled-budget comparison).  Tuning
  budget and the actual settings are recorded per method; untuned is written as
  untuned.
* Falsification criteria are written into ``summary.md`` *before* the run, then
  evaluated numerically and reported whether or not they support the candidate.

Outputs: ``metrics.json``, ``per_condition.csv``, ``predictions.npz`` (with
``values``/``target_xy``/``prediction``/``valid``), ``summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - CLI path
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "owac.evaluation.n01 needs PyTorch, which is the optional 'encoder' extra.\n"
        "Install it with: uv sync --locked --extra encoder\n"
        f"Original import error: {exc}"
    ) from exc

from owac.training.n01 import (
    N01ProbeHeads,
    _percentile,
    _require,
    build_probe_heads,
    git_sha,
    load_checkpoint,
    peak_gpu_bytes,
    peak_rss_bytes,
    probe_parameter_count,
    save_checkpoint,
    sequence_step,
    tensor_bytes,
    units_note,
    write_json,
)

__all__ = [
    "Condition",
    "FALSIFICATION_CRITERIA",
    "build_conditions",
    "condition_metrics",
    "evaluate_probe",
    "probe_heads_for",
    "main",
    "mechanism_matrix",
    "predictions_payload",
    "write_summary",
]

#: Declared before any run.  The candidate is only supported if none of these
#: conditions is met; every one of them is computed from measured numbers.
FALSIFICATION_CRITERIA: tuple[dict[str, Any], ...] = (
    {
        "id": "F1_no_static_average_tradeoff",
        "statement": (
            "On the narrow-tolerance, high-SNR slice the candidate must not trade fast thin-edge "
            "accuracy for static averaging: mean position error on 'fast' must exceed the mean on "
            "'still' by more than 1.5 px."
        ),
        "measure": "position_error_px (pixels)",
        "threshold": "delta(fast - still) > 1.5 px on the high/narrow slice",
    },
    {
        "id": "F2_not_max_window_average",
        "statement": (
            "The candidate must not degenerate into a maximum-window average: the effective "
            "history length must not sit within 5% of max_history, and the mean weighted support "
            "count must stay below the full (2r+1)^2 window mass."
        ),
        "measure": "effective_history_frames (frames), support_count_mean (weighted samples)",
        "threshold": "history within 5% of max_history AND support_count >= 0.9 * window area",
    },
    {
        "id": "F3_fast_edge_not_worse_than_static_baseline",
        "statement": (
            "The candidate must not be worse than the strongest static-history baseline "
            "(fixed average over the same history) on the fast+motion slice by more than 2 px."
        ),
        "measure": "position_error_px on motion=fast",
        "threshold": "candidate_error - best_static_error > 2.0 px",
    },
    {
        "id": "F4_support_actually_reorganises",
        "statement": (
            "The support graph must actually reorganise: either merge_count or split_count must be "
            "non-zero in the mechanism matrix, otherwise the hysteresis band is inert."
        ),
        "measure": "merge/split counts from the encoder diagnostics",
        "threshold": "merge_count == 0 AND split_count == 0 on every condition",
    },
)

_CSV_FIELDS = (
    "condition",
    "seed",
    "snr",
    "motion",
    "tolerance",
    "augment",
    "frames_scored",
    "position_error_px",
    "position_error_norm",
    "over_tolerance_rate",
    "tolerance_px",
    "first_stable_frame",
    "recovery_frames",
    "processing_p50_ms",
    "processing_p95_ms",
    "processing_mean_ms",
    "sample_age_mean_frames",
    "sample_age_max_frames",
    "support_radius_mean_px",
    "support_width_mean_px",
    "support_count_mean",
    "history_age_mean_frames",
    "observed_frames_mean",
    "merge_count",
    "split_count",
    "state_bytes",
    "peak_rss_bytes",
)


@dataclass(frozen=True)
class Condition:
    """One fixed-seed mechanism condition."""

    name: str
    snr: str
    motion: str
    tolerance: str
    augment: str = "none"
    seed: int = 0
    tolerance_px: float = 2.0
    tolerance_source: str = "declared_default"

    def task_kwargs(self) -> dict[str, Any]:
        return {
            "snr": self.snr,
            "motion": self.motion,
            "tolerance": self.tolerance,
            "augment": self.augment,
            "seed": self.seed,
        }


def build_conditions(base_seed: int = 0, tolerance_px: float = 2.0) -> list[Condition]:
    """SNR x motion x tolerance grid plus the four event conditions."""
    conditions: list[Condition] = []
    index = 0
    for snr in ("low", "medium", "high"):
        for motion in ("still", "slow", "fast"):
            for tolerance in ("wide", "narrow"):
                index += 1
                conditions.append(
                    Condition(
                        name=f"{snr}_{motion}_{tolerance}",
                        snr=snr,
                        motion=motion,
                        tolerance=tolerance,
                        seed=base_seed + index,
                        tolerance_px=tolerance_px,
                    )
                )
    for augment in ("sudden_appear", "direction_reversal", "occlusion_release", "crossing"):
        index += 1
        conditions.append(
            Condition(
                name=f"event_{augment}",
                snr="medium",
                motion="slow",
                tolerance="wide",
                augment=augment,
                seed=base_seed + index,
                tolerance_px=tolerance_px,
            )
        )
    return conditions


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _soft_argmax(logits: Any) -> Any:
    """Expected (row, col) of a logit map.

    Accepts ``[B,H,W]`` (one frame) or ``[B,T,H,W]`` (a window) and always returns
    ``[B,T,2]`` / ``[B,2]`` with the time axis preserved, so the caller never has
    to guess which layout it received.
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.ndim != 4:
        raise ValueError(f"expected [B,H,W] or [B,T,H,W] logits, got {tuple(logits.shape)}")
    batch, frames, height, width = logits.shape
    probability = torch.softmax(logits.reshape(batch, frames, -1).float(), dim=-1)
    rows = torch.arange(height, device=logits.device, dtype=probability.dtype)
    cols = torch.arange(width, device=logits.device, dtype=probability.dtype)
    grid_rows = rows.view(1, 1, height, 1).expand(batch, frames, height, width)
    grid_cols = cols.view(1, 1, 1, width).expand(batch, frames, height, width)
    grid_rows = grid_rows.reshape(batch, frames, -1)
    grid_cols = grid_cols.reshape(batch, frames, -1)
    return torch.stack(
        [(probability * grid_rows).sum(-1), (probability * grid_cols).sum(-1)], dim=-1
    )


def _first_stable_frame(inside: list[bool]) -> int | None:
    """First frame from which every later frame is inside tolerance."""
    for index in range(len(inside)):
        if all(inside[index:]):
            return index
    return None


def _recovery_frames(inside: list[bool], reversal_index: int) -> int | None:
    """Frames needed after a direction reversal to reach a stable in-tolerance run."""
    for offset in range(reversal_index + 1, len(inside) + 1):
        tail = inside[offset:]
        if tail and all(tail):
            return offset - reversal_index
    return None


def condition_metrics(
    result: Any,
    condition: Condition,
    *,
    max_history: int,
    window_area: int,
    state_bytes: int,
    diagnostics: dict[str, Any],
    observed_frames: float,
) -> dict[str, Any]:
    """Per-condition metrics in explicit units, plus the arrays for reuse."""
    logits = result.logits  # [B,T,H,W]
    batch, frames, height, width = logits.shape
    target_xy = result.target_xy
    visible = result.target_visible.bool()
    expected = _soft_argmax(logits)
    centre = torch.stack(
        [
            target_xy[:, :, 0].clamp(0.0, 1.0) * (width - 1),
            target_xy[:, :, 1].clamp(0.0, 1.0) * (height - 1),
        ],
        dim=-1,
    )
    if centre.shape[1] != frames:  # a single target per window
        centre = centre.expand(-1, frames, -1)
    # Unnormalised logits can put mass anywhere; the report uses both the argmax
    # (mode) and the soft-argmax (mean) and keeps them distinguishable.
    error_px = (expected - centre).norm(dim=-1)  # pixels
    diagonal = math.hypot(height - 1, width - 1)
    error_norm = error_px / diagonal
    valid_frames = visible & torch.isfinite(error_px)
    inside = (error_px <= condition.tolerance_px) & valid_frames
    per_batch_error = [
        float(error_px[row][valid_frames[row]].mean())
        if bool(valid_frames[row].any())
        else float("nan")
        for row in range(batch)
    ]
    finite = [value for value in per_batch_error if math.isfinite(value)]
    scores = [bool(value) for value in inside.any(dim=0).tolist()]
    first_stable = _first_stable_frame(scores)
    reversal = None
    if condition.augment == "direction_reversal":
        # The reversal is where the target trajectory turns hardest; it is derived
        # from the synthetic target, never read from a hidden ground-truth field.
        if frames >= 3:
            velocity = centre[:, 1:] - centre[:, :-1]
            turn = (velocity[:, 1:] - velocity[:, :-1]).norm(dim=-1).mean(dim=0)
            reversal = int(torch.argmax(turn).item()) + 1
    recovery = None if reversal is None else _recovery_frames(scores, reversal)
    timings = result.step_times_ms
    sample_age = result.sample_age
    age_mean = (
        float(sample_age[visible].mean())
        if sample_age is not None and bool(visible.any())
        else None
    )
    age_max = (
        float(sample_age[visible].max()) if sample_age is not None and bool(visible.any()) else None
    )
    support_radius = diagnostics.get("selected_radius_mean")
    metrics = {
        "condition": condition.name,
        "seed": condition.seed,
        "snr": condition.snr,
        "motion": condition.motion,
        "tolerance": condition.tolerance,
        "augment": condition.augment,
        "tolerance_px": condition.tolerance_px,
        "tolerance_source": condition.tolerance_source,
        "frames_scored": int(valid_frames.sum()),
        "position_error_px": float(np.nanmean(per_batch_error)) if finite else float("nan"),
        "position_error_norm": float(error_norm[valid_frames].mean())
        if bool(valid_frames.any())
        else float("nan"),
        "over_tolerance_rate": (
            float((~inside[valid_frames]).float().mean())
            if bool(valid_frames.any())
            else float("nan")
        ),
        "first_stable_frame": first_stable,
        "direction_reversal_frame": reversal,
        "recovery_frames": recovery,
        "processing_p50_ms": statistics.median(timings) if timings else float("nan"),
        "processing_p95_ms": _percentile(sorted(timings), 0.95) if timings else float("nan"),
        "processing_mean_ms": float(sum(timings) / len(timings)) if timings else float("nan"),
        "sample_age_mean_frames": age_mean,
        "sample_age_max_frames": age_max,
        "support_radius_mean_px": support_radius,
        "support_width_mean_px": (
            2.0 * float(support_radius) + 1.0 if support_radius is not None else None
        ),
        "support_count_mean": diagnostics.get("mean_support_samples", 0.0)
        / max(diagnostics.get("mean_unique_frames", 1.0), 1e-6),
        "history_age_mean_frames": diagnostics.get("mean_age"),
        "observed_frames_mean": float(observed_frames),
        "merge_count": diagnostics.get("merge_count"),
        "split_count": diagnostics.get("split_count"),
        "state_bytes": int(state_bytes),
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_gpu_bytes": peak_gpu_bytes(),
        "max_history": int(max_history),
        "window_area": int(window_area),
    }
    arrays = {
        "error_px": error_px.detach().cpu().numpy(),
        "expected_xy": expected.detach().cpu().numpy(),
        "valid_frames": valid_frames.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
    }
    return {"metrics": metrics, "arrays": arrays}


# ---------------------------------------------------------------------------
# running a method over the matrix
# ---------------------------------------------------------------------------
def mechanism_matrix(
    encoder: Any,
    heads: N01ProbeHeads,
    config: Any,
    conditions: Sequence[Condition],
    *,
    batch_size: int,
    length: int,
    image_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Run the candidate over every condition with a fixed per-condition seed."""
    synthetic = _require("owac.frameworks.n01.synthetic", "the mechanism suite")
    per_condition: list[dict[str, Any]] = []
    archive: dict[str, Any] = {}
    diagnostics_rows: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        task = synthetic.SyntheticWeakEdgeTask(
            config, image_size=image_size, **condition.task_kwargs()
        )
        batch = task.sample(batch_size, length, seed=condition.seed)
        result = sequence_step(
            encoder,
            heads,
            batch,
            step=index,
            tbptt=0,
            grad_enabled=False,
            device=device,
            holdout=False,
        )
        state = result.state
        diagnostics = dict(result.last_output.diagnostics) if result.last_output is not None else {}
        observed = diagnostics.get("mean_unique_frames", float("nan"))
        diagnostics_rows.append({"condition": condition.name, **diagnostics})
        evaluated = condition_metrics(
            result,
            condition,
            max_history=int(config.max_history),
            window_area=(2 * int(config.neighbourhood_radius) + 1) ** 2,
            state_bytes=_state_bytes(state),
            diagnostics=diagnostics,
            observed_frames=observed,
        )
        per_condition.append(evaluated["metrics"])
        archive[condition.name] = {
            "values": batch.values.detach().cpu().numpy(),
            "target_xy": batch.target_xy.detach().cpu().numpy(),
            "edge_target": batch.edge_target.detach().cpu().numpy(),
            "prediction": evaluated["arrays"]["expected_xy"].astype(np.float32),
            "valid": evaluated["arrays"]["valid_frames"].astype(np.bool_),
            "error_px": evaluated["arrays"]["error_px"].astype(np.float32),
            "logits": evaluated["arrays"]["logits"].astype(np.float32),
            "seed": np.int64(condition.seed),
            "snr": condition.snr,
            "motion": condition.motion,
            "tolerance": condition.tolerance,
            "augment": condition.augment,
        }
    aggregate = _aggregate(per_condition)
    return {
        "conditions": per_condition,
        "aggregate": aggregate,
        "diagnostics": diagnostics_rows,
        "archive": archive,
    }


def _state_bytes(state: Any) -> int:
    if state is None:
        return 0
    from owac.frameworks.n01.state import state_bytes as _bytes

    try:
        return int(_bytes(state))
    except Exception:  # pragma: no cover - defensive
        total = 0
        for name in ("values", "variance", "valid", "observed", "frame_ids", "timestamps"):
            total += tensor_bytes(getattr(state, name, None))
        return total


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str) -> float | None:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else None

    return {
        "position_error_px_mean": mean_of("position_error_px"),
        "over_tolerance_rate_mean": mean_of("over_tolerance_rate"),
        "processing_p50_ms_mean": mean_of("processing_p50_ms"),
        "processing_p95_ms_mean": mean_of("processing_p95_ms"),
        "sample_age_mean_frames": mean_of("sample_age_mean_frames"),
        "support_width_mean_px": mean_of("support_width_mean_px"),
        "support_count_mean": mean_of("support_count_mean"),
        "history_age_mean_frames": mean_of("history_age_mean_frames"),
        "observed_frames_mean": mean_of("observed_frames_mean"),
        "merge_count_mean": mean_of("merge_count"),
        "split_count_mean": mean_of("split_count"),
        "state_bytes_mean": mean_of("state_bytes"),
        # A probe without a persistent measurement state reports None for state
        # bytes; unmeasured stays unmeasured instead of becoming zero.
        "state_bytes_max": max(
            (int(row["state_bytes"]) for row in rows if row.get("state_bytes") is not None),
            default=None,
        ),
        "conditions": len(rows),
    }


def _slice(rows: Sequence[dict[str, Any]], **fields: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if all(row.get(key) == value for key, value in fields.items()):
            selected.append(row)
    return selected


def falsification_report(
    rows: Sequence[dict[str, Any]],
    comparison: dict[str, Any] | None,
    config: Any,
) -> list[dict[str, Any]]:
    """Evaluate the pre-declared criteria against measured numbers."""

    def mean(values: Sequence[float]) -> float | None:
        finite = [
            float(value) for value in values if value is not None and math.isfinite(float(value))
        ]
        return float(np.mean(finite)) if finite else None

    fast = mean(
        [
            row["position_error_px"]
            for row in _slice(rows, motion="fast", tolerance="narrow", snr="high")
        ]
    )
    still = mean(
        [
            row["position_error_px"]
            for row in _slice(rows, motion="still", tolerance="narrow", snr="high")
        ]
    )
    history = mean([row["observed_frames_mean"] for row in rows])
    support_count = mean([row["support_count_mean"] for row in rows])
    window_area = (2 * int(config.neighbourhood_radius) + 1) ** 2
    merges = sum(float(row["merge_count"] or 0.0) for row in rows)
    splits = sum(float(row["split_count"] or 0.0) for row in rows)
    max_history = int(config.max_history)
    candidate_rows = (comparison or {}).get("methods", {}).get("n01", {}).get("conditions", rows)
    candidate_fast = mean(
        [row["position_error_px"] for row in _slice(candidate_rows, motion="fast")]
    )
    static_names = ("fixed_average", "ema", "single_frame_raw_cnn")
    best_static = None
    if comparison:
        for name in static_names:
            entry = comparison.get("methods", comparison).get(name)
            if not entry:
                continue
            value = mean(
                [
                    row["position_error_px"]
                    for row in _slice(entry.get("conditions", []), motion="fast")
                ]
            )
            if value is None:
                continue
            best_static = value if best_static is None else min(best_static, value)

    f1_delta = None if fast is None or still is None else fast - still
    f2_history = None if history is None else history >= 0.95 * max_history
    f2_support = None if support_count is None else support_count >= 0.9 * window_area
    f3_delta = (
        None if candidate_fast is None or best_static is None else candidate_fast - best_static
    )
    results = [
        {
            "id": FALSIFICATION_CRITERIA[0]["id"],
            "statement": FALSIFICATION_CRITERIA[0]["statement"],
            "measured": {
                "fast_narrow_error_px": fast,
                "still_narrow_error_px": still,
                "delta_px": f1_delta,
            },
            "threshold": FALSIFICATION_CRITERIA[0]["threshold"],
            "triggered": None if f1_delta is None else bool(f1_delta > 1.5),
        },
        {
            "id": FALSIFICATION_CRITERIA[1]["id"],
            "statement": FALSIFICATION_CRITERIA[1]["statement"],
            "measured": {
                "effective_history_frames": history,
                "max_history": max_history,
                "support_count_mean": support_count,
                "window_area": window_area,
            },
            "threshold": FALSIFICATION_CRITERIA[1]["threshold"],
            "triggered": (
                None
                if f2_history is None or f2_support is None
                else bool(f2_history and f2_support)
            ),
        },
        {
            "id": FALSIFICATION_CRITERIA[2]["id"],
            "statement": FALSIFICATION_CRITERIA[2]["statement"],
            "measured": {
                "candidate_fast_error_px": candidate_fast,
                "best_static_fast_error_px": best_static,
                "delta_px": f3_delta,
            },
            "threshold": FALSIFICATION_CRITERIA[2]["threshold"],
            "triggered": None if f3_delta is None else bool(f3_delta > 2.0),
        },
        {
            "id": FALSIFICATION_CRITERIA[3]["id"],
            "statement": FALSIFICATION_CRITERIA[3]["statement"],
            "measured": {"merge_count_sum": merges, "split_count_sum": splits},
            "threshold": FALSIFICATION_CRITERIA[3]["threshold"],
            "triggered": bool(merges == 0.0 and splits == 0.0),
        },
    ]
    return results


# ---------------------------------------------------------------------------
# controlled-budget comparison
# ---------------------------------------------------------------------------
def pregenerate_batches(
    config: Any,
    *,
    steps: int,
    batch_size: int,
    length: int,
    image_size: int,
    seed: int,
) -> list[Any]:
    """Byte-identical training batches shared by every compared method."""
    synthetic = _require("owac.frameworks.n01.synthetic", "the comparison suite")
    task = synthetic.SyntheticWeakEdgeTask(config, image_size=image_size, seed=seed)
    batches = []
    for index in range(steps):
        batches.append(task.sample(batch_size, length, seed=seed * 1_000_003 + index))
    return batches


def train_baseline(
    probe: Any,
    heads: N01ProbeHeads,
    batches: Sequence[Any],
    *,
    lr: float,
    weight_decay: float,
    grad_clip: float | None,
    location_mode: str,
    sigma: float,
    device: str,
) -> dict[str, Any]:
    """Train one baseline probe with the identical head, optimiser and steps."""
    if hasattr(probe, "encoder"):
        probe.encoder.calibrating = True
        with torch.no_grad():
            sample = batches[0]
            probe.encode(
                sample.values.to(device),
                sample.valid.to(device),
                sample.variance.to(device),
                sample.timestamps.to(device),
            )
        probe.encoder.calibrating = False
        probe.reset_state()
    parameters = [p for p in probe.parameters() if p.requires_grad] + [
        p for p in heads.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    times: list[float] = []
    for _step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        probe.reset_state()
        started = time.perf_counter()
        features = probe.encode(
            batch.values.to(device),
            batch.valid.to(device),
            batch.variance.to(device),
            batch.timestamps.to(device),
        )
        heads.build(int(features.shape[1]), int(batch.values.shape[2]))
        logits, _ = heads(features)
        # One map per window: the probe readout is the newest frame's position, so
        # the training target is that frame rather than the whole window.
        loss, _, _ = heads.location.location_loss(
            logits,
            batch.target_xy[:, -1].to(device),
            batch.target_visible[:, -1].to(device),
            mode=location_mode,
            sigma=sigma,
        )
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
        optimizer.step()
        times.append((time.perf_counter() - started) * 1000.0)
        losses.append(float(loss.detach()))
    return {
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last10": float(np.mean(losses[-10:])) if losses else None,
        "step_p50_ms": statistics.median(times) if times else None,
        "step_p95_ms": _percentile(sorted(times), 0.95) if times else None,
    }


def comparison_suite(
    config: Any,
    heads_template: N01ProbeHeads,
    conditions: Sequence[Condition],
    *,
    steps: int,
    batch_size: int,
    length: int,
    image_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float | None,
    location_mode: str,
    sigma: float,
    device: str,
    seed: int,
    tuned: bool = False,
) -> dict[str, Any]:
    """Every mechanism with the same head, optimiser, steps and data."""
    baselines = _require("owac.baselines.n01", "the comparison suite")
    registry = baselines.baseline_registry()
    batches = pregenerate_batches(
        config,
        steps=steps,
        batch_size=batch_size,
        length=length,
        image_size=image_size,
        seed=seed,
    )
    results: dict[str, Any] = {}
    for name, factory in registry.items():
        torch.manual_seed(seed)
        started = time.perf_counter()
        try:
            method = factory(config) if callable(factory) else factory
        except Exception as exc:  # pragma: no cover - one method may fail to build
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        method = method.to(device)
        torch.manual_seed(seed + 100_003)
        heads = probe_heads_for(method, config, image_size=image_size, device=device)
        training = train_baseline(
            method,
            heads,
            batches,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            location_mode=location_mode,
            sigma=sigma,
            device=device,
        )
        evaluated = evaluate_probe(method, heads, config, conditions, batch_size, length, device)
        results[name] = {
            "name": name,
            "input_spec": method.input_spec(),
            "parameters": int(method.parameter_count()),
            "probe_parameters": probe_parameter_count(heads),
            "training": training,
            "conditions": evaluated["conditions"],
            "aggregate": evaluated["aggregate"],
            "wall_s": time.perf_counter() - started,
        }
    budget = {
        "training_steps": int(steps),
        "batch_size": int(batch_size),
        "length": int(length),
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "location_mode": location_mode,
        "location_sigma_px": sigma,
        "aux_measurement_prediction": False,
        "identical_batches": True,
        "measured": {
            name: {
                "step_p50_ms": entry.get("training", {}).get("step_p50_ms"),
                "step_p95_ms": entry.get("training", {}).get("step_p95_ms"),
            }
            for name, entry in results.items()
            if "error" not in entry
        },
    }
    if not tuned:
        budget["tuning_status"] = (
            "UNTUNED: these are the starting parameters named in the implementation prompt "
            "(AdamW, lr=3e-4, identical step count). No per-method hyper-parameter search was run."
        )
    else:
        budget["tuning_status"] = "tuned per method; see per-method settings in this record"
    return {"budget": budget, "methods": results}


def probe_heads_for(
    probe: Any, config: Any, *, image_size: int, device: str, hidden: int = 32
) -> N01ProbeHeads:
    """Shared probe head sized from one real ``MechanismProbe.encode`` output.

    The head is created from the probe's own feature width, so every compared
    method is read out by the same architecture, and the auxiliary measurement
    head is disabled so the comparison budget is location-only for all methods.
    """
    with torch.no_grad():
        probe.reset_state()
        features = probe.encode(
            torch.zeros(1, 1, int(config.channels), image_size, image_size, device=device),
            torch.ones(1, 1, int(config.channels), image_size, image_size, device=device),
            torch.zeros(1, 1, int(config.channels), image_size, image_size, device=device),
            torch.zeros(1, 1, device=device),
        )
    # The comparison is location-only for every method, so the auxiliary
    # measurement head is disabled for this configuration only (a copy; the
    # checkpoint's own config is untouched).
    location_only = config.replace(aux_measurement_prediction=False)
    heads = N01ProbeHeads(location_only, (image_size, image_size), location_hidden=hidden).to(
        device
    )
    heads.build(int(features.shape[1]), int(config.channels))
    if heads.measurement is not None:  # pragma: no cover - defensive
        raise RuntimeError("the comparison head must be location-only")
    probe.reset_state()
    return heads


def evaluate_probe(
    probe: Any,
    heads: N01ProbeHeads,
    config: Any,
    conditions: Sequence[Condition],
    batch_size: int,
    length: int,
    device: str,
) -> dict[str, Any]:
    """Run one mechanism probe over the same matrix with a fixed seed per condition."""
    synthetic = _require("owac.frameworks.n01.synthetic", "the mechanism suite")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        task = synthetic.SyntheticWeakEdgeTask(config, **condition.task_kwargs())
        batch = task.sample(batch_size, length, seed=condition.seed)
        with torch.no_grad():
            probe.reset_state()
            features = probe.encode(
                batch.values.to(device),
                batch.valid.to(device),
                batch.variance.to(device),
                batch.timestamps.to(device),
            )
            heads.build(int(features.shape[1]), int(batch.values.shape[2]))
            logits, _ = heads(features)
        # One map per window: the probe readout describes the newest frame, so the
        # comparison uses that frame's target rather than the whole window.
        logits = logits.unsqueeze(1)
        target_xy = batch.target_xy[:, -1:].to(device)
        visible = batch.target_visible[:, -1:].to(device)
        expected = _soft_argmax(logits)
        height, width = logits.shape[-2:]
        centre = torch.stack(
            [
                target_xy[:, :, 0].clamp(0.0, 1.0) * (width - 1),
                target_xy[:, :, 1].clamp(0.0, 1.0) * (height - 1),
            ],
            dim=-1,
        )
        error = (expected - centre).norm(dim=-1)
        valid = visible & torch.isfinite(error)
        diagnostics = dict(probe.diagnostics())
        statistics_payload = dict(probe.statistics()) if hasattr(probe, "statistics") else {}
        rows.append(
            {
                "condition": condition.name,
                "seed": condition.seed,
                "snr": condition.snr,
                "motion": condition.motion,
                "tolerance": condition.tolerance,
                "augment": condition.augment,
                "tolerance_px": condition.tolerance_px,
                "position_error_px": float(error[valid].mean())
                if bool(valid.any())
                else float("nan"),
                "over_tolerance_rate": (
                    float((error[valid] > condition.tolerance_px).float().mean())
                    if bool(valid.any())
                    else float("nan")
                ),
                "first_stable_frame": _first_stable_frame(
                    [bool(value) for value in (error <= condition.tolerance_px)[valid].tolist()]
                ),
                "recovery_frames": None,
                "support_count_mean": statistics_payload.get("support_count")
                or diagnostics.get("mean_support_count"),
                "history_age_mean_frames": statistics_payload.get("age")
                or diagnostics.get("mean_age"),
                "merge_count": diagnostics.get("merge_count"),
                "split_count": diagnostics.get("split_count"),
                "state_bytes": diagnostics.get("state_bytes"),
            }
        )
    return {"conditions": rows, "aggregate": _aggregate(rows)}


# ---------------------------------------------------------------------------
# writing outputs
# ---------------------------------------------------------------------------
def predictions_payload(archive: dict[str, Any]) -> dict[str, Any]:
    """Flatten the per-condition records into the reusable ``predictions.npz`` arrays.

    ``values`` is the channel-mean measurement field ``[N,H,W]`` (measurement
    units), ``target_xy``/``prediction`` are ``[N,2]`` (row, col) in pixels, and
    ``valid`` is ``[N]`` so a trajectory plot can mask padding frames.
    """
    values, targets, predictions, valid, errors, labels, rows = [], [], [], [], [], [], []
    for row, (name, entry) in enumerate(archive.items()):
        value = np.asarray(entry["values"], dtype=np.float32)  # [B,T,C,H,W]
        value = value.mean(axis=2)  # channel mean, measurement units preserved
        batch, frames = value.shape[0], value.shape[1]
        target = np.asarray(entry["target_xy"], dtype=np.float32)
        if target.ndim == 2:  # one target per window
            target = np.repeat(target[:, None, :], frames, axis=1)
        validity = np.asarray(entry["valid"], dtype=np.bool_)
        if validity.ndim == 1:
            validity = np.repeat(validity[:, None], frames, axis=1)
        prediction = np.asarray(entry["prediction"], dtype=np.float32)
        if prediction.ndim == 2:  # one prediction per window
            prediction = np.repeat(prediction[:, None, :], frames, axis=1)
        error = np.asarray(entry["error_px"], dtype=np.float32)
        if error.ndim == 1:
            error = np.repeat(error[:, None], frames, axis=1)
        values.append(value.reshape(-1, *value.shape[2:]))
        targets.append(target.reshape(-1, 2))
        predictions.append(prediction.reshape(-1, 2))
        valid.append(validity.reshape(-1))
        errors.append(error.reshape(-1))
        labels.extend([name] * (batch * frames))
        rows.extend([row] * (batch * frames))
    return {
        "values": np.concatenate(values, axis=0),
        "target_xy": np.concatenate(targets, axis=0),
        "prediction": np.concatenate(predictions, axis=0),
        "valid": np.concatenate(valid, axis=0),
        "error_px": np.concatenate(errors, axis=0),
        "condition": np.asarray(labels),
        "condition_index": np.asarray(rows, dtype=np.int64),
    }


def write_per_condition_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _CSV_FIELDS})


def write_summary(
    path: Path,
    *,
    header: dict[str, Any],
    matrix: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    falsification: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# N01 evaluation summary")
    lines.append("")
    lines.append(f"- checkpoint: `{header.get('checkpoint')}`")
    lines.append(f"- config fingerprint: `{header.get('config_fingerprint')}`")
    lines.append(f"- git sha (evaluation run): `{header.get('git_sha')}`")
    lines.append(f"- checkpoint git sha: `{header.get('checkpoint_git_sha')}`")
    lines.append(f"- git sha matches: `{header.get('git_sha_matches')}`")
    lines.append(f"- config fingerprint matches: `{header.get('fingerprint_matches')}`")
    lines.append(f"- suites: `{', '.join(header.get('suites', []))}`")
    lines.append(f"- device: `{header.get('device')}`, torch `{header.get('torch_version')}`")
    lines.append("")
    if header.get("warnings"):
        lines.append("## Warnings")
        for warning in header["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append("## Falsification criteria (declared before the run)")
    lines.append("")
    for item in FALSIFICATION_CRITERIA:
        lines.append(f"- **{item['id']}** — {item['statement']}")
        lines.append(f"  - measure: {item['measure']}")
        lines.append(f"  - threshold: {item['threshold']}")
    lines.append("")
    lines.append("## Falsification results (measured)")
    lines.append("")
    lines.append("| id | triggered | measured |")
    lines.append("| --- | --- | --- |")
    for item in falsification:
        measured = json.dumps(item["measured"], sort_keys=True)
        lines.append(f"| {item['id']} | {item['triggered']} | `{measured}` |")
    lines.append("")
    lines.append(
        "`triggered = true` means the candidate's claim is *not* supported by that measurement; "
        "`null` means the comparison could not be computed from this run and is reported as such."
    )
    lines.append("")
    if matrix is not None:
        lines.append("## Mechanism matrix (candidate)")
        lines.append("")
        lines.append(
            "| condition | seed | pos err (px) | over tol | first stable | recovery | "
            "proc p50/p95 (ms) | sample age | support width | history age | merge/split |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in matrix["conditions"]:
            lines.append(
                "| {condition} | {seed} | {err:.3f} | {over:.3f} | {first} | {recovery} | "
                "{p50:.2f}/{p95:.2f} | {age} | {width} | {history} | {merge}/{split} |".format(
                    condition=row["condition"],
                    seed=row["seed"],
                    err=float(row["position_error_px"]),
                    over=float(row["over_tolerance_rate"]),
                    first=row["first_stable_frame"],
                    recovery=row["recovery_frames"],
                    p50=float(row["processing_p50_ms"]),
                    p95=float(row["processing_p95_ms"]),
                    age=_fmt(row["sample_age_mean_frames"]),
                    width=_fmt(row["support_width_mean_px"]),
                    history=_fmt(row["history_age_mean_frames"]),
                    merge=_fmt(row["merge_count"]),
                    split=_fmt(row["split_count"]),
                )
            )
        lines.append("")
        lines.append("### Candidate aggregate")
        lines.append("")
        lines.append(f"```json\n{json.dumps(matrix['aggregate'], indent=2, sort_keys=True)}\n```")
        lines.append("")
    if comparison is not None:
        lines.append("## Controlled-budget comparison")
        lines.append("")
        lines.append(f"- tuning status: {comparison['budget'].get('tuning_status')}")
        declared = {
            key: value
            for key, value in comparison["budget"].items()
            if key not in ("measured", "tuning_status")
        }
        lines.append(f"- identical budget: {json.dumps(declared, sort_keys=True)}")
        lines.append("")
        lines.append(
            "| method | params | pos err (px) | over tol | first stable | "
            "step p50/p95 (ms) | train loss (first -> last) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, entry in comparison["methods"].items():
            if "error" in entry:
                lines.append(f"| {name} | — | — | — | — | — | build failed: {entry['error']} |")
                continue
            aggregate = entry["aggregate"]
            training = entry["training"]
            lines.append(
                "| {name} | {params} | {err} | {over} | {first} | {p50}/{p95} | "
                "{first_loss} -> {last_loss} |".format(
                    name=name,
                    params=entry["parameters"],
                    err=_fmt(aggregate.get("position_error_px_mean"), 3),
                    over=_fmt(aggregate.get("over_tolerance_rate_mean"), 3),
                    first="per-condition",
                    p50=_fmt(training.get("step_p50_ms"), 2),
                    p95=_fmt(training.get("step_p95_ms"), 2),
                    first_loss=_fmt(training.get("loss_first"), 5),
                    last_loss=_fmt(training.get("loss_last"), 5),
                )
            )
        lines.append("")
    lines.append("## Units")
    lines.append("")
    for key, value in units_note().items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Not measured in this run")
    lines.append("")
    for item in header.get("not_measured", []):
        lines.append(f"- {item}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m owac.evaluation.n01",
        description="Evaluate an N01 checkpoint on the mechanism matrix and controlled comparison.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=("mechanism", "comparison", "all"), default="mechanism")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument(
        "--seed", type=int, default=1234, help="suite seed (per-condition seeds derive from it)"
    )
    parser.add_argument("--tolerance-px", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--comparison-steps", type=int, default=50)
    parser.add_argument("--comparison-lr", type=float, default=3e-4)
    parser.add_argument("--comparison-weight-decay", type=float, default=1e-4)
    parser.add_argument("--comparison-grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--comparison-tuned", action="store_true", help="mark the comparison as tuned"
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="do not write the evaluated (possibly probe-updated) checkpoint back",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.comparison_tuned:
        raise SystemExit("--comparison-tuned requires actual per-method tuning; not implemented")
    config_module = _require("owac.frameworks.n01.config", "config loading")
    model_module = _require("owac.frameworks.n01.model", "the N01 encoder")
    config = config_module.load_config(args.config)
    fingerprint = config_module.config_fingerprint(config)
    payload = load_checkpoint(args.checkpoint, map_location=args.device)
    stored_config = dict(payload.get("config", {}))
    warnings: list[str] = []
    stored_fingerprint = stored_config.get("config_fingerprint")
    fingerprint_matches = stored_fingerprint == fingerprint
    if stored_fingerprint and not fingerprint_matches:
        warnings.append(
            f"config fingerprint mismatch: checkpoint {stored_fingerprint} != current {fingerprint}"
        )
    run_sha = git_sha(Path(__file__).resolve().parents[3])
    checkpoint_sha = str(payload.get("git_sha", "unknown"))
    sha_matches = checkpoint_sha == run_sha
    if not sha_matches:
        warnings.append(f"git sha mismatch: checkpoint {checkpoint_sha} != current {run_sha}")
    if str(payload.get("torch_version")) != str(torch.__version__):
        warnings.append(
            "torch version mismatch: checkpoint "
            f"{payload.get('torch_version')} != current {torch.__version__}"
        )
    for warning in warnings:
        print(json.dumps({"warning": warning}), file=sys.stderr, flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    encoder = model_module.N01Encoder(config).to(args.device)
    encoder.load_state_dict(payload["model"])
    heads = build_probe_heads(
        encoder,
        (args.image_size, args.image_size),
        channels=int(config.channels),
        device=args.device,
    )
    if "probe" in payload:
        heads.load_state_dict(payload["probe"])
    encoder.eval()
    heads.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(base_seed=args.seed, tolerance_px=args.tolerance_px)
    matrix = None
    comparison = None
    archive: dict[str, Any] = {}
    if args.suite in ("mechanism", "all"):
        started = time.perf_counter()
        matrix = mechanism_matrix(
            encoder,
            heads,
            config,
            conditions,
            batch_size=args.batch_size,
            length=args.length,
            image_size=args.image_size,
            device=args.device,
            seed=args.seed,
        )
        matrix["wall_s"] = time.perf_counter() - started
        archive = matrix.pop("archive")
    if args.suite in ("comparison", "all"):
        comparison = comparison_suite(
            config,
            heads,
            conditions,
            steps=args.comparison_steps,
            batch_size=args.batch_size,
            length=args.length,
            image_size=args.image_size,
            lr=args.comparison_lr,
            weight_decay=args.comparison_weight_decay,
            grad_clip=(None if args.comparison_grad_clip <= 0 else args.comparison_grad_clip),
            location_mode="huber",
            sigma=1.5,
            device=args.device,
            seed=args.seed,
            tuned=args.comparison_tuned,
        )
        if matrix is not None:
            comparison["candidate_reference"] = {
                "aggregate": matrix["aggregate"],
                "source": "mechanism suite (trained checkpoint, not retrained in this comparison)",
            }
    falsification = falsification_report(matrix["conditions"], comparison, config) if matrix else []
    header = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "checkpoint_seed": int(payload.get("seed", -1)),
        "checkpoint_git_sha": checkpoint_sha,
        "config_fingerprint": fingerprint,
        "checkpoint_fingerprint": stored_fingerprint,
        "fingerprint_matches": fingerprint_matches,
        "git_sha": run_sha,
        "git_sha_matches": sha_matches,
        "torch_version": str(torch.__version__),
        "device": args.device,
        "suites": [args.suite],
        "warnings": warnings,
        "parameter_count": {
            "encoder": int(sum(p.numel() for p in encoder.parameters())),
            "probe": probe_parameter_count(heads),
        },
        "resources": {
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_gpu_bytes": peak_gpu_bytes(),
        },
        "command": " ".join(
            [Path(sys.executable).name, "-m", "owac.evaluation.n01", *sys.argv[1:]]
        ),
        "argv": list(sys.argv),
        "not_measured": [
            "closed-loop robot success rate (no robot or simulator in this run)",
            "real camera RAW data (synthetic weak-edge task only)",
            "hard real-time guarantees: --deadline_ms is an optional soft budget, not polled",
        ],
    }
    metrics = {
        "header": header,
        # Serialisable description of every condition; the dataclass instances
        # themselves are not JSON values.
        "conditions": [
            {
                "name": item.name,
                "snr": item.snr,
                "motion": item.motion,
                "tolerance": item.tolerance,
                "augment": item.augment,
                "seed": item.seed,
                "tolerance_px": item.tolerance_px,
                "tolerance_source": item.tolerance_source,
            }
            for item in build_conditions(args.seed, args.tolerance_px)
        ],
        "mechanism": matrix,
        "comparison": comparison,
        "falsification": falsification,
        "units": units_note(),
        "condition_table": [
            {
                "name": item.name,
                "seed": item.seed,
                "snr": item.snr,
                "motion": item.motion,
                "tolerance": item.tolerance,
                "augment": item.augment,
            }
            for item in conditions
        ],
    }
    write_json(output / "metrics.json", metrics)
    if matrix is not None:
        write_per_condition_csv(output / "per_condition.csv", matrix["conditions"])
        np.savez_compressed(output / "predictions.npz", **predictions_payload(archive))
    write_summary(
        output / "summary.md",
        header=header,
        matrix=matrix,
        comparison=comparison,
        falsification=falsification,
    )
    if not args.no_save_checkpoint and matrix is not None:
        save_checkpoint(output / "checkpoint.pt", payload)
    print(
        json.dumps(
            {
                "metrics": str(output / "metrics.json"),
                "summary": str(output / "summary.md"),
                "conditions": len(conditions),
                "falsification_triggered": [
                    item["id"] for item in falsification if item["triggered"]
                ],
                "warnings": warnings,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

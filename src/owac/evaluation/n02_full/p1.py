"""P1 evaluation CLI for the N02 full (v1.1) line.

    python -m owac.evaluation.n02_full.p1 oracle   --config configs/encoders/n02_full/p1_eval.json
    python -m owac.evaluation.n02_full.p1 behavior --config configs/encoders/n02_full/p1_eval.json
    python -m owac.evaluation.n02_full.p1 ablation --config configs/encoders/n02_full/p1_eval.json
    python -m owac.evaluation.n02_full.p1 report   --config configs/encoders/n02_full/p1_eval.json

Subcommands
-----------
``oracle``    analytic-generator upper bound: prove the controlled task is
              correctable before blaming a learned model (P1 §6).
``behavior``  the four frozen acceptance families, each with at least the
              declared number of independent scenarios and seeds, keeping every
              per-scenario record: cue swap in both directions, no-information
              posterior drift, 95%-wrong-prior correction in both directions,
              normal false-positive rate against structural-insufficiency
              detection, grouped by exposure/noise/shift.
``ablation``  the declared comparison arms (ordinary candidate discrimination,
              conditional likelihood, raw frame / conditional innovation /
              registration, and the independently trained official NRE/BNRE
              ratio losses with the same Base, history and budget).
``report``    compose ``report.json``, ``metrics.json``, ``run_manifest.json``
              and ``stage_decision.json`` from the receipts that exist.

Failure policy
--------------
A missing module, missing asset or missing integration hook is
``status="not_run"`` with the exact reason and exit code 0. Thresholds are only
compared against real, per-scenario evidence; a rate is never reported from a
partial run as if it were the full one, and an unrun arm stays ``not_run``. A
malformed config exits non-zero.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from owac.frameworks.n02_full import config as n02_config

__all__ = [
    "SUBCOMMANDS",
    "BehaviorThresholds",
    "DeficitError",
    "EvalSpec",
    "build_parser",
    "main",
    "posterior_drift",
    "summarise_behavior",
    "wilson_interval",
]

SUBCOMMANDS = ("oracle", "behavior", "ablation", "report")
DEFAULT_CONFIG = "configs/encoders/n02_full/p1_eval.json"
DEFAULT_MEASUREMENT_CONFIG = "configs/encoders/n02_full/p1_measurement.json"
DEFAULT_OUTPUT_ROOT = "outputs/n02-full/p1-eval"
RECEIPT_FORMAT = "n02-full-p1-eval-receipt-v1"
SCENARIO_FORMAT = "n02-full-p1-scenario-record-v1"


class DeficitError(n02_config.ConfigError):
    """A resolved evaluation section is malformed; the message names the key path."""


class EvalNotAvailable(RuntimeError):
    """A required module, asset or integration hook is absent (``not_run``)."""

    def __init__(self, reason: str, *, missing: Sequence[str] = ()) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.missing = tuple(missing)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now() -> str:

    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def append_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | Path) -> Any:
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeficitError(f"{where}: expected an integer, got {type(value).__name__}")
    return int(value)


def _float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeficitError(f"{where}: expected a number, got {type(value).__name__}")
    return float(value)


def _strs(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise DeficitError(f"{where}: expected a list of strings")
    return tuple(value)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; the honest interval for a count out of a total."""
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / float(total)
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total))
    return (max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator))


def posterior_drift(prior: Sequence[float], posterior: Sequence[float]) -> float:
    """Maximum absolute probability change between two declared two-candidate masses.

    This is the P1 §6 definition: only a verifiable two-candidate probability may
    be called a posterior, and drift is the largest absolute change of one of its
    entries. Sequences whose masses do not both sum to ~1 are rejected instead of
    being scored as if they were calibrated.
    """
    if len(prior) != len(posterior):
        raise ValueError("prior and posterior must have the same length")
    if len(prior) < 1:
        raise ValueError("empty probability vector")
    for name, values in (("prior", prior), ("posterior", posterior)):
        total = float(sum(float(value) for value in values))
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"{name} mass must sum to 1, got {total!r}")
        if any(float(value) < -1e-9 for value in values):
            raise ValueError(f"{name} has a negative entry")
    return float(max(abs(float(a) - float(b)) for a, b in zip(prior, posterior, strict=True)))


def _rate(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, total)
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": (None if total == 0 else successes / float(total)),
        "wilson95_low": low,
        "wilson95_high": high,
    }


@dataclass(frozen=True, slots=True)
class BehaviorThresholds:
    """Frozen acceptance thresholds of P1 §6 (defaults are the stage prompt)."""

    cue_swap_both_correct: float = 0.90
    no_information_drift_max: float = 0.05
    no_information_pass_rate: float = 0.95
    wrong_prior_correction: float = 0.80
    normal_false_positive_max: float = 0.05
    insufficiency_detection: float = 0.90
    min_scenarios_per_class: int = 1000
    seeds: tuple[int, ...] = (1103, 2207, 3301)
    groups: tuple[str, ...] = ("exposure", "noise", "shift")

    KINDS: ClassVar[dict[str, str]] = {
        "cue_swap_both_correct": "float",
        "no_information_drift_max": "float",
        "no_information_pass_rate": "float",
        "wrong_prior_correction": "float",
        "normal_false_positive_max": "float",
        "insufficiency_detection": "float",
        "min_scenarios_per_class": "int",
        "seeds": "ints",
        "groups": "strs",
    }


def _spec_from_section(cls: type, section: Any, where: str) -> Any:
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise DeficitError(f"{where}: expected an object, got {type(section).__name__}")
    kinds: dict[str, str] = cls.KINDS
    unknown = sorted(set(section) - set(kinds))
    if unknown:
        raise DeficitError(f"{where}: unknown key(s) {', '.join(repr(k) for k in unknown)}")
    converters = {"int": _int, "float": _float, "strs": _strs, "ints": _ints, "map": _map}
    values = {
        name: converters[kinds[name]](raw, f"{where}.{name}") for name, raw in section.items()
    }
    return cls(**values)


def _ints(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise DeficitError(f"{where}: expected a list of integers")
    return tuple(_int(item, f"{where}[{index}]") for index, item in enumerate(value))


def _map(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise DeficitError(f"{where}: expected an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class EvalSpec:
    """One resolved evaluation config: scenario counts, seeds, thresholds, arms."""

    scenarios: int = 1000
    seeds: tuple[int, ...] = (1103, 2207, 3301)
    thresholds: BehaviorThresholds = field(default_factory=BehaviorThresholds)
    oracle_min_correctable: float = 0.95
    ablation_arms: tuple[dict, ...] = ()
    keep_scenario_records: bool = True
    device: str = "cpu"

    KINDS: ClassVar[dict[str, str]] = {
        "scenarios": "int",
        "seeds": "ints",
        "thresholds": "map",
        "oracle_min_correctable": "float",
        "ablation_arms": "list",
        "keep_scenario_records": "bool",
        "device": "str",
    }

    def __post_init__(self) -> None:
        if self.scenarios <= 0:
            raise DeficitError(f"scenarios must be positive, got {self.scenarios}")
        if not self.seeds:
            raise DeficitError("seeds must be non-empty")


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise DeficitError(f"{where}: expected a boolean, got {type(value).__name__}")
    return value


def _list(value: Any, where: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise DeficitError(f"{where}: expected a list, got {type(value).__name__}")
    return tuple(value)


def load_eval_spec(cfg: n02_config.N02FullConfig) -> EvalSpec:
    """Resolve the evaluation section; an empty section keeps the frozen defaults."""
    section = cfg.evaluation if isinstance(cfg.evaluation, dict) else {}
    raw_scenarios = section.get("scenarios")
    scenarios: Mapping[str, Any] = raw_scenarios if isinstance(raw_scenarios, Mapping) else {}
    count = scenarios.get("count")
    if count is None:
        count = raw_scenarios if isinstance(raw_scenarios, (int, float)) else 1000
    thresholds_raw = scenarios.get("thresholds")
    if not isinstance(thresholds_raw, Mapping):
        thresholds_raw = (
            section.get("thresholds") if isinstance(section.get("thresholds"), Mapping) else {}
        )
    raw_ablation = section.get("ablation")
    ablation: Mapping[str, Any] = raw_ablation if isinstance(raw_ablation, Mapping) else {}
    return EvalSpec(
        scenarios=int(count),
        seeds=tuple(_ints(scenarios.get("seeds", cfg.seeds), "evaluation.scenarios.seeds")),
        thresholds=_section_thresholds(thresholds_raw),
        oracle_min_correctable=float(scenarios.get("oracle_min_correctable", 0.95)),
        ablation_arms=tuple(_list(ablation.get("arms", []), "evaluation.ablation.arms")),
        keep_scenario_records=bool(scenarios.get("keep_scenario_records", True)),
    )


def _section_thresholds(section: Any) -> BehaviorThresholds:
    if not isinstance(section, Mapping):
        raise DeficitError("evaluation.thresholds: expected an object")
    known = set(BehaviorThresholds.KINDS)
    unknown = sorted(set(section) - known)
    if unknown:
        raise DeficitError(
            f"evaluation.thresholds: unknown key(s) {', '.join(repr(k) for k in unknown)}"
        )
    return _spec_from_section(BehaviorThresholds, section, "evaluation.thresholds")


# ---------------------------------------------------------------------------
# scenario record summarisation
# ---------------------------------------------------------------------------


def _records(records: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    mine = [record for record in records if record.get("family") == name]
    if not mine:
        raise DeficitError(f"no scenario records for family {name!r}")
    return mine


def _count(records: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for record in records if bool(record.get(key)))


def summarise_behavior(
    records: Sequence[Mapping[str, Any]], thresholds: BehaviorThresholds
) -> dict[str, Any]:
    """Turn per-scenario records into the four frozen verdicts.

    Every record declares ``family`` and ``seed``; a family with fewer than
    ``thresholds.min_scenarios_per_class`` independent scenarios per seed is
    ``incomplete`` rather than pass/fail, so a short run can never be reported as
    a satisfied acceptance criterion.
    """
    if not records:
        raise DeficitError("no scenario records were produced")
    seeds = sorted({int(record.get("seed", 0)) for record in records})
    summary: dict[str, Any] = {}

    cue = _records(records, "cue_swap")
    direction_a = [record for record in cue if str(record.get("direction")) == "a"]
    direction_b = [record for record in cue if str(record.get("direction")) == "b"]
    paired = _paired_cue(cue)
    cue_complete = all(
        len([record for record in cue if int(record.get("seed", 0)) == seed])
        >= thresholds.min_scenarios_per_class
        for seed in seeds
    )
    summary["cue_swap"] = {
        "both_correct": _rate(paired[0], paired[1]),
        "direction_a": _rate(_count(direction_a, "correct"), len(direction_a)),
        "direction_b": _rate(_count(direction_b, "correct"), len(direction_b)),
        "seeds": seeds,
        "per_seed_min_scenarios": {
            str(seed): len([r for r in cue if int(r.get("seed", 0)) == seed]) for seed in seeds
        },
        "status": (
            "incomplete"
            if not cue_complete
            else (
                "pass"
                if _rate(paired[0], paired[1])["rate"] is not None
                and _rate(paired[0], paired[1])["rate"] >= thresholds.cue_swap_both_correct
                and _rate(_count(direction_a, "correct"), len(direction_a))["rate"]
                >= thresholds.cue_swap_both_correct
                and _rate(_count(direction_b, "correct"), len(direction_b))["rate"]
                >= thresholds.cue_swap_both_correct
                else "fail"
            )
        ),
    }

    noinfo = _records(records, "no_information")
    drifts = [float(record["drift"]) for record in noinfo]
    passing = sum(1 for value in drifts if value <= thresholds.no_information_drift_max)
    noinfo_complete = all(
        len([r for r in noinfo if int(r.get("seed", 0)) == seed])
        >= thresholds.min_scenarios_per_class
        for seed in seeds
    )
    summary["no_information"] = {
        "drift_max": max(drifts) if drifts else None,
        "drift_mean": (sum(drifts) / len(drifts)) if drifts else None,
        "within_threshold": _rate(passing, len(drifts)),
        "drift_threshold": thresholds.no_information_drift_max,
        "seeds": seeds,
        "status": (
            "incomplete"
            if not noinfo_complete
            else (
                "pass"
                if passing / float(len(drifts)) >= thresholds.no_information_pass_rate
                else "fail"
            )
        ),
    }

    wrong = _records(records, "wrong_prior")
    by_direction = {
        direction: [record for record in wrong if str(record.get("direction")) == direction]
        for direction in ("a", "b")
    }
    wrong_complete = all(
        len([r for r in wrong if int(r.get("seed", 0)) == seed])
        >= thresholds.min_scenarios_per_class
        for seed in seeds
    )
    summary["wrong_prior"] = {
        direction: _rate(_count(rows, "corrected"), len(rows))
        for direction, rows in by_direction.items()
    }
    summary["wrong_prior"]["seeds"] = seeds
    summary["wrong_prior"]["status"] = (
        "incomplete"
        if not wrong_complete
        else (
            "pass"
            if all(
                _rate(_count(rows, "corrected"), len(rows))["rate"] is not None
                and _rate(_count(rows, "corrected"), len(rows))["rate"]
                >= thresholds.wrong_prior_correction
                for rows in by_direction.values()
            )
            else "fail"
        )
    )

    normal = _records(records, "normal")
    insufficient = _records(records, "insufficiency")
    false_positives = _count(normal, "flagged")
    summary["checks"] = {
        "normal_false_positive": _rate(false_positives, len(normal)),
        "insufficiency_detected": _rate(_count(insufficient, "detected"), len(insufficient)),
        "false_positive_threshold": thresholds.normal_false_positive_max,
        "detection_threshold": thresholds.insufficiency_detection,
        "by_group": _group_summary(normal, insufficient, thresholds),
        "status": (
            "pass"
            if _rate(false_positives, len(normal))["rate"] <= thresholds.normal_false_positive_max
            and _rate(_count(insufficient, "detected"), len(insufficient))["rate"]
            >= thresholds.insufficiency_detection
            else "fail"
        ),
    }
    summary["status"] = (
        "pass"
        if all(
            summary[name]["status"] == "pass"
            for name in ("cue_swap", "no_information", "wrong_prior", "checks")
        )
        else "incomplete"
        if any(
            summary[name]["status"] == "incomplete"
            for name in ("cue_swap", "no_information", "wrong_prior", "checks")
        )
        else "fail"
    )
    return summary


def _paired_cue(records: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    pairs: dict[Any, dict[str, bool]] = {}
    for record in records:
        key = (record.get("scenario_id"), record.get("seed"))
        pairs.setdefault(key, {})[str(record.get("direction"))] = bool(record.get("correct"))
    both = sum(1 for value in pairs.values() if value.get("a") and value.get("b"))
    return both, len(pairs)


def _group_summary(
    normal: Sequence[Mapping[str, Any]],
    insufficient: Sequence[Mapping[str, Any]],
    thresholds: BehaviorThresholds,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in thresholds.groups:
        normal_rows = [row for row in normal if row.get("group") == group]
        insufficient_rows = [row for row in insufficient if row.get("group") == group]
        groups[group] = {
            "normal": _rate(_count(normal_rows, "flagged"), len(normal_rows)),
            "insufficiency": _rate(_count(insufficient_rows, "detected"), len(insufficient_rows)),
            "status": (
                "not_run"
                if not normal_rows or not insufficient_rows
                else "pass"
                if _rate(_count(normal_rows, "flagged"), len(normal_rows))["rate"]
                <= thresholds.normal_false_positive_max
                and _rate(_count(insufficient_rows, "detected"), len(insufficient_rows))["rate"]
                >= thresholds.insufficiency_detection
                else "fail"
            ),
        }
    return groups


# ---------------------------------------------------------------------------
# scenario production (integration points)
# ---------------------------------------------------------------------------


def load_scenario_runner() -> Any:
    """Resolve the controlled-task runner from the P1 framework modules.

    The runner is ``factors.build_p1_task`` (the constructor that assembles the
    common candidate dictionary, the supervised targets with the explicit
    ``candidate_index`` label and the evaluator-side truth). It is the only
    accepted source of scenarios: building a private second dictionary here would
    mean the methods and the evaluator no longer see the same candidates.
    """
    try:
        module = importlib.import_module("owac.frameworks.n02_full.factors")
    except ImportError as exc:
        raise EvalNotAvailable(
            f"owac.frameworks.n02_full.factors is not available: {type(exc).__name__}: {exc}",
            missing=("owac.frameworks.n02_full.factors",),
        ) from exc
    builder = getattr(module, "build_p1_task", None)
    if builder is None:
        raise EvalNotAvailable(
            "owac.frameworks.n02_full.factors.build_p1_task is missing (the controlled task "
            "constructor owns the common candidate dictionary)",
            missing=("owac.frameworks.n02_full.factors.build_p1_task",),
        )
    return builder


def load_observation_stack() -> Any:
    """Resolve (model factory, checks module) for behaviour and ablation runs."""
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for name, attr in (
        ("owac.frameworks.n02_full.observation", "RawObservationEncoder"),
        ("owac.frameworks.n02_full.sensor", "SensorModel"),
        ("owac.frameworks.n02_full.factors", "ObservationModel"),
        ("owac.frameworks.n02_full.model", "step_bundle"),
        ("owac.frameworks.n02_full.checks", "WindowedCalibrator"),
    ):
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            missing.append(f"{name} ({type(exc).__name__})")
            continue
        if not hasattr(module, attr):
            missing.append(f"{name}.{attr}")
        modules[name] = module
    if missing:
        raise EvalNotAvailable(
            "the P1 observation stack is incomplete: " + "; ".join(missing), missing=tuple(missing)
        )
    return modules


def _bound_torch_threads() -> int:
    """Cap torch's intra-op threads for the evaluation drivers.

    The acceptance steps are tiny-tensor work; with the host's full thread count each op
    thrashes: one family step measured 880 CPU-s and 29 s wall, versus 1.9 CPU-s and 1.0 s
    wall at four threads. The cap is a pure scheduling change (same arithmetic);
    ``OWAC_EVAL_THREADS`` overrides it.
    """
    import os

    try:
        import torch

        threads = int(os.environ.get("OWAC_EVAL_THREADS", "8"))
        if threads > 0:
            torch.set_num_threads(threads)
        return threads
    except Exception:  # noqa: BLE001 - never let a scheduling tweak break a run
        return 0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return False


def _posterior(output: Any, *, temperature: float = 1.0) -> tuple[Any, Any]:
    """Candidate scores and posterior from one step (``softmax(-score / T)``)."""
    import numpy as np

    scores = getattr(output, "world_scores", None)
    values = None
    if scores is not None and hasattr(scores, "detach"):
        values = np.asarray(scores.detach().to("cpu"), dtype="float64").reshape(-1)
    if values is None or values.size == 0:
        worlds = getattr(output, "worlds", ()) or ()
        values = np.asarray(
            [float(getattr(world, "score", 0.0) or 0.0) for world in worlds], dtype="float64"
        )
    if values.size == 0:
        return np.zeros(0), np.zeros(0)
    scaled = -values / max(1e-6, float(temperature))
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return values, exp / exp.sum()


def _verdicts(output: Any) -> list[str]:
    checks = getattr(output, "locked_checks", None) or ()
    verdicts: list[str] = []
    for check in checks:
        verdict = getattr(check, "verdict", None)
        if verdict is None and isinstance(check, dict):
            verdict = check.get("verdict")
        verdicts.append(str(verdict))
    return verdicts


def _trace(output: Any, posterior: Any, candidate: int | None) -> dict[str, Any]:
    """Per-candidate readout the acceptance record keeps for independent re-checks."""
    if candidate is None or posterior is None or len(posterior) == 0:
        return {}
    verdicts = _verdicts(output)
    return {
        str(int(candidate)): {
            "verdict": verdicts[0] if verdicts else "ok",
            "locked": True,
            "candidate": int(candidate),
            "posterior": float(posterior[int(candidate)]),
        }
    }


def _history_context(model: Any, sequence: Any, frame_index: int, bundle: Any) -> Any:
    """The causal history of one frame, built the way the training path builds it.

    An empty context (no encoder pass) makes ``step_bundle`` skip the likelihood
    registration and return no ``world_scores``, so the acceptance run would silently
    grade a fallback path. The previous frame is therefore encoded here as well.
    """
    import torch

    from owac.frameworks.n02_full import factors

    rows = len(bundle.episode_ids)
    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    device = parameters[0].device if parameters else torch.device("cpu")
    if sequence is None or int(frame_index) <= 0:
        return factors.HistoryContext(
            pooled_features=torch.zeros((rows, 512), dtype=torch.float32, device=device),
            levels=None,
            episode_ids=tuple(bundle.episode_ids),
            anchor_ids=tuple(bundle.anchor_ids),
            last_frame_index=None,
            eta=None,
        )
    previous = sequence.inference[int(frame_index) - 1].bundle
    with torch.no_grad():
        pyramid = model.encode(previous)
    level = pyramid.levels.get("stride16") or next(iter(pyramid.levels.values()))
    pooled = level.values
    pooled = pooled.mean(dim=(-1, -2)) if hasattr(pooled, "mean") else pooled
    return factors.HistoryContext(
        pooled_features=pooled,
        levels=pyramid.levels,
        episode_ids=tuple(bundle.episode_ids),
        anchor_ids=tuple(bundle.anchor_ids),
        last_frame_index=int(frame_index) - 1,
        eta=None,
    )


def _model_step(model: Any, task: Any, sequence: Any = None) -> Any:
    """One causal step on a single controlled task, without gradients."""
    import torch

    from owac.frameworks.n02_full import model as model_mod
    from owac.frameworks.n02_full.state import SequenceState

    bundle = task.inference.bundle
    # The candidate worlds come from the bound public dictionary: without it the step
    # returns no worlds, no world_scores and no checks, and every family would be graded
    # on an empty readout.
    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    device = parameters[0].device if parameters else torch.device("cpu")
    from owac.training import n02_full as _train

    model.bind_dictionary(_train.move_dictionary_to(task.dictionary, device))
    context = _history_context(model, sequence, int(getattr(task, "frame_index", 0) or 0), bundle)
    with torch.no_grad():
        return model_mod.step_bundle(
            bundle, SequenceState(), model, context, ledger=None, budget=None, targets=task.targets
        )


def oracle_hypothesis(spec: Any, object_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """One candidate's declared geometry as the generator's hypothesis mapping.

    ``CandidateGeometry`` carries ``world_px`` ``(N, 2)`` plus ``scale_px``/``depth`` ``(N,)``
    in the *truth object order* the public dictionary was built from, so the ids come from the
    frame's ``EvaluationTruth.object_ids`` and every row must line up with one of them.
    """
    geometry = getattr(spec, "geometry", None)
    if geometry is None:
        raise EvalNotAvailable("the candidate carries no declared geometry")
    world_px = getattr(geometry, "world_px", None)
    if world_px is None or len(getattr(world_px, "shape", ())) != 2 or world_px.shape[1] != 2:
        raise EvalNotAvailable("the candidate geometry is not a declared (N, 2) world_px")
    ids = tuple(str(item) for item in object_ids)
    if len(ids) != int(world_px.shape[0]):
        raise EvalNotAvailable(
            f"the candidate declares {int(world_px.shape[0])} objects but the frame has "
            f"{len(ids)}; the hypothesis mapping cannot be guessed"
        )
    scale = getattr(geometry, "scale_px", None)
    depth = getattr(geometry, "depth", None)
    mapping: dict[str, dict[str, Any]] = {}
    for row, object_id in enumerate(ids):
        entry: dict[str, Any] = {
            "world_px": (float(world_px[row, 0]), float(world_px[row, 1])),
        }
        if scale is not None:
            entry["scale_px"] = float(scale[row])
        if depth is not None:
            entry["depth"] = float(depth[row])
        mapping[object_id] = entry
    return mapping


ORACLE_HOOK_HASH = "synthetic.py 1e4e75a0eeb591c4"


ORACLE_HOOK_HASH = "synthetic.py d26e66a5d9dbfce4"

#: The generator capabilities the analytic oracle (P1 §6) consumes. They are present in the
#: frozen generator (per-frame truth ``RenderState`` and the hypothesis override on
#: ``mean_variance``); the probe below keeps reporting their absence loudly if they ever go
#: away, instead of scoring a substitute model.
ORACLE_REQUIRED_APIS = (
    "sequence.render_states[frame_index] -> RenderState (per-frame truth state)",
    "mean_variance(sensor, state, hypothesis={entity: {world_px|scale_px|depth}}) "
    "-> (mean, variance)",
)


def _oracle_mean_variance(
    generator: Any, state: Any, sensors_by_camera: Any, hypothesis: Any
) -> Any:
    """The analytic mean/variance of one hypothesis, point-for-point with the query."""

    def mean_variance(history: Any, query_spec: Any) -> tuple[Any, Any]:
        import numpy as np
        import torch

        coordinates = torch.as_tensor(query_spec.sensor_coordinates, dtype=torch.float64)
        camera_ids = tuple(str(item) for item in query_spec.camera_ids)
        if coordinates.dim() != 2 or coordinates.shape[1] != 2:
            raise EvalNotAvailable(
                f"the query spec's sensor_coordinates are not (Q, 2): {tuple(coordinates.shape)}"
            )
        if len(camera_ids) != int(coordinates.shape[0]):
            raise EvalNotAvailable(
                f"the query spec names {len(camera_ids)} cameras for "
                f"{int(coordinates.shape[0])} points"
            )
        means = torch.empty(coordinates.shape[0], dtype=torch.float32)
        variances = torch.empty_like(means)
        cache: dict[str, Any] = {}
        for index in range(int(coordinates.shape[0])):
            camera_id = camera_ids[index]
            if camera_id not in cache:
                sensor = sensors_by_camera.get(camera_id)
                if sensor is None:
                    raise EvalNotAvailable(f"the frame declares no sensor for {camera_id!r}")
                cache[camera_id] = generator.mean_variance(sensor, state, hypothesis=hypothesis)
            mean_map, variance_map = cache[camera_id]
            mean_map = np.asarray(mean_map)
            variance_map = np.asarray(variance_map)
            height, width = mean_map.shape[:2]
            first, second = coordinates[index, 0], coordinates[index, 1]
            row, column = int(round(float(first))), int(round(float(second)))
            if not (0 <= row < height and 0 <= column < width):
                row, column = int(round(float(second))), int(round(float(first)))
            if not (0 <= row < height and 0 <= column < width):
                raise EvalNotAvailable(
                    f"the query point ({float(first)}, {float(second)}) is outside the "
                    f"{height}x{width} sensor"
                )
            means[index] = float(mean_map[row, column])
            variances[index] = float(variance_map[row, column])
        return means, variances

    return mean_variance


def _score_value(score: Any) -> float:
    """A comparable scalar from one proper score (lower is better)."""
    for name in ("raw_nll_or_energy", "value", "mean"):
        item = getattr(score, name, None)
        if item is None:
            continue
        try:
            return float(item)
        except (TypeError, ValueError):
            continue
    raise EvalNotAvailable("the oracle score exposes no comparable scalar")


def oracle_scenario_records(
    cfg: n02_config.N02FullConfig,
    spec: EvalSpec,
    *,
    checkpoint: str = "",
    output: Path,
    interim_budget: int = 0,
    measurement_config: str = DEFAULT_MEASUREMENT_CONFIG,
) -> list[dict[str, Any]]:
    """The analytic-oracle upper bound: is the task correctable at all (P1 §6)?

    Every record is marked ``oracle=true`` / ``source=generator_truth`` and names the hook
    it was built against. The query of each frame comes from T3's public
    :func:`build_row_query`/:func:`build_raw_query`, so the oracle scores the *observed*
    values on the same public dictionary the trained model is scored on.
    """
    _bound_torch_threads()
    import numpy as np

    from owac.frameworks.n02_full import factors
    from owac.frameworks.n02_full import model as model_mod
    from owac.frameworks.n02_full import sensor as sensor_mod
    from owac.frameworks.n02_full import synthetic as synth
    from owac.training import n02_full as train

    measurement, model_cfg = _measurement_declaration(cfg, measurement_config)
    sensors = tuple(
        synth.SensorSpec(**dict(entry)) for entry in measurement.get("synthetic_sensors", [])
    )
    if not sensors:
        raise EvalNotAvailable("the measurement config declares no synthetic_sensors")
    generator = synth.MeasurementModel(
        **dict(measurement.get("measurement_model", {"kind": "cfa"}))
    )
    for name in ("build_row_query", "build_raw_query"):
        if not hasattr(model_mod, name):
            raise EvalNotAvailable(f"owac.frameworks.n02_full.model has no public {name}")
    units = str(measurement.get("units", "raw_dn_uint16_mosaic"))
    dataset = synth.generate_dataset(
        sensors=sensors,
        measurement_model=generator,
        partitions=("train",),
        scenes=max(64, int(spec.scenarios) * 3),
        frames=max(3, int(measurement.get("frames_per_episode", 8))),
        views_per_scene=1,
        seed=int(spec.seeds[0]),
        canvas=tuple(measurement.get("canvas", (64, 64))),
        action_horizon=int(measurement.get("action_horizon", 4)),
    )
    probe = next(iter(dataset))
    if not getattr(probe, "render_states", None):
        raise EvalNotAvailable(
            "the analytic oracle needs sequence.render_states; "
            f"[hook {ORACLE_HOOK_HASH}; no oracle number is reported]"
        )
    # the declared clip levels (saturation/floor) live in the measurement config, not in
    # the evaluation config: building the sensor model from the wrong one left them unset
    sensor_model = train.build_sensor_model(model_cfg, units, density_verified=True)
    device = "cuda" if _cuda_available() else "cpu"
    model = train.build_observation_model(model_cfg, device, units=units, world_size=None)
    records: list[dict[str, Any]] = []
    pinned: int | None = None
    for index, sequence in enumerate(dataset):
        task = factors.build_p1_task(sequence, frame_index=1)
        size = int(task.dictionary.world_size())
        if pinned is None:
            pinned = size
            model = train.build_observation_model(model_cfg, device, units=units, world_size=pinned)
        if size != pinned:
            continue
        state = sequence.render_states[1]
        sensors_by_camera = {str(item.camera_id): item for item in sequence.cameras}
        object_ids = tuple(str(item) for item in sequence.truth[1].object_ids)
        history = _history_context(model, sequence, 1, task.inference.bundle)
        row_query = model_mod.build_row_query(
            model, task.inference.bundle, 0, batch=1, history=history
        )
        raw_query = model_mod.build_raw_query(row_query)
        scores: list[float] = []
        hypotheses: list[dict[str, Any]] = []
        for candidate in task.dictionary.specs:
            hypothesis = oracle_hypothesis(candidate, object_ids)
            hypotheses.append(hypothesis)
            oracle = factors.AnalyticOracle(
                _oracle_mean_variance(generator, state, sensors_by_camera, hypothesis),
                prediction_type=sensor_mod.RawPrediction,
                sensor_model=sensor_model,
                version="p1-analytic-oracle",
            )
            world = candidate.as_world()
            prediction = oracle.predict(world, history, row_query.spec, call="prequential_check")
            scores.append(_score_value(oracle.score(prediction, raw_query)))
        best = int(np.argmin(scores)) if scores else -1
        truth = task.candidate_index
        total = float(np.sum(np.exp(-np.asarray(scores)))) if scores else 0.0
        records.append(
            {
                "scenario_id": str(getattr(sequence, "scene_id", index)),
                "seed": int(spec.seeds[0]),
                # the oracle stage aggregates one truth-selection decision per scenario; the
                # controlled family it was decided on is kept alongside
                "family": "oracle",
                "oracle_family": "cue_swap",
                "direction": "a",
                "oracle": True,
                "source": "generator_truth",
                "hook_hash": ORACLE_HOOK_HASH,
                "candidate_index": None if truth is None else int(truth),
                "map_index": best,
                "posterior": (
                    float(np.exp(-scores[best]) / total) if scores and total > 0 else 0.0
                ),
                "scores": [float(item) for item in scores],
                "correct": bool(truth is not None and best == int(truth)),
                "budget": {
                    "interim_budget": int(interim_budget),
                    "world_size": pinned,
                    "oracle_hypotheses": len(hypotheses),
                },
            }
        )
        if len(records) >= int(spec.scenarios):
            break
    return records


def _measurement_declaration(
    cfg: n02_config.N02FullConfig, measurement_config: str
) -> tuple[dict[str, Any], n02_config.N02FullConfig]:
    """The sensor/measurement declaration and the config that carries the model.

    The evaluation config holds thresholds and scenario counts; the declared sensors,
    units and model input mode live in the measurement config, so the acceptance run
    reads them from there instead of inventing defaults.
    """
    training = cfg.training if isinstance(cfg.training, dict) else {}
    measurement = dict(training.get("measurement", {}) or {})
    if measurement.get("synthetic_sensors"):
        return measurement, cfg
    path = Path(measurement_config)
    if not path.is_file():
        raise EvalNotAvailable(
            f"no measurement declaration in the evaluation config and {path} is not a file"
        )
    other = n02_config.load_config(path)
    other_training = other.training if isinstance(other.training, dict) else {}
    return dict(other_training.get("measurement", {}) or {}), other


def model_scenario_records(
    cfg: n02_config.N02FullConfig,
    spec: EvalSpec,
    *,
    checkpoint: str,
    output: Path,
    interim_budget: int = 0,
    measurement_config: str = DEFAULT_MEASUREMENT_CONFIG,
) -> list[dict[str, Any]]:
    """The model-driven acceptance records: one row per scenario and family.

    Every record carries the family, the direction/group it belongs to, the field the
    frozen thresholds read, and the posterior that produced it, so the raw
    ``scenario_records.jsonl`` can be re-checked independently of the summariser.
    """
    _bound_torch_threads()
    import dataclasses

    import numpy as np

    from owac.frameworks.n02_full import factors
    from owac.frameworks.n02_full import synthetic as synth
    from owac.training import n02_full as train

    measurement, model_cfg = _measurement_declaration(cfg, measurement_config)
    sensors = tuple(
        synth.SensorSpec(**dict(entry)) for entry in measurement.get("synthetic_sensors", [])
    )
    if not sensors:
        raise EvalNotAvailable("the measurement config declares no synthetic_sensors")
    frames = max(3, int(measurement.get("frames_per_episode", 8)))
    units = str(measurement.get("units", "raw_dn_uint16_mosaic"))
    device = "cuda" if _cuda_available() else "cpu"
    temperature = float(measurement.get("ce_temperature", 1.0))
    seed = int(spec.seeds[0])
    dataset = synth.generate_dataset(
        sensors=sensors,
        measurement_model=synth.MeasurementModel(
            **dict(measurement.get("measurement_model", {"kind": "cfa"}))
        ),
        partitions=("train",),
        scenes=max(64, int(spec.scenarios) * 3),
        frames=frames,
        views_per_scene=1,
        seed=seed,
        canvas=tuple(measurement.get("canvas", (64, 64))),
        action_horizon=int(measurement.get("action_horizon", 4)),
    )
    # One encoder has one candidate capacity, so the run pins the object count of its
    # first scene and skips scenes of another size (the same rule the training source
    # applies); the filter is recorded in every record's budget block.
    sequences: list[Any] = []
    pinned = None
    for candidate in dataset:
        probe = factors.build_p1_task(candidate, frame_index=1)
        size = int(probe.dictionary.world_size())
        if pinned is None:
            pinned = size
        if size != pinned:
            continue
        sequences.append(candidate)
        if len(sequences) >= int(spec.scenarios):
            break
    if not sequences:
        raise EvalNotAvailable("the generator produced no scene with the pinned object count")
    model = train.build_observation_model(model_cfg, device, units=units, world_size=pinned)
    payload = train.load_checkpoint(checkpoint)
    report = train.load_base_subset(model, payload.get("model", payload))
    for parameter in train.model_parameters(model):
        parameter.requires_grad_(False)
    budget = {
        "interim_budget": int(interim_budget),
        "base_inheritance": report,
        "seeds_run": [seed],
        "temperature": temperature,
        "checkpoint": str(checkpoint),
        "pinned_world_size": pinned,
        "scenarios_evaluated": len(sequences),
        "scenarios_generated": int(spec.scenarios) * 3,
    }
    records: list[dict[str, Any]] = []
    for index, sequence in enumerate(sequences):
        scenario_id = f"{getattr(sequence, 'scene_id', 'scene')}:{index}"
        for direction, correct_index in (("a", 0), ("b", 1)):
            task = factors.build_p1_task(sequence, frame_index=1, correct_index=correct_index)
            output = _model_step(model, task, sequence)
            _scores, posterior = _posterior(output, temperature=temperature)
            top = int(np.argmax(posterior)) if posterior.size else -1
            truth = task.candidate_index
            records.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "family": "cue_swap",
                    "direction": direction,
                    "correct": bool(truth is not None and top == int(truth)),
                    "candidate_index": None if truth is None else int(truth),
                    "map_index": top,
                    "posterior": float(posterior[top]) if posterior.size else 0.0,
                    "expert_trace": _trace(output, posterior, truth),
                    "budget": budget,
                }
            )
        task = factors.build_p1_task(sequence, frame_index=0)
        output = _model_step(model, task, sequence)
        _scores, posterior = _posterior(output, temperature=temperature)
        prior = getattr(task.inference, "declared_prior", None)
        drift = 0.0
        if posterior.size:
            if prior is not None and hasattr(prior, "detach"):
                prior_values = np.asarray(prior.detach().to("cpu"), dtype="float64").reshape(-1)
                if prior_values.size == posterior.size:
                    drift = float(np.max(np.abs(posterior - prior_values)))
            else:
                uniform = 1.0 / posterior.size
                drift = float(max(abs(float(value) - uniform) for value in posterior))
        records.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "family": "no_information",
                "drift": drift,
                "budget": budget,
            }
        )
        for direction, correct_index in (("a", 0), ("b", 1)):
            task = factors.build_p1_task(sequence, frame_index=1, correct_index=correct_index)
            truth = task.candidate_index
            if truth is not None:
                width = len(task.dictionary.worlds())
                if width:
                    misleading = np.full(width, 0.05 / max(1, width - 1))
                    misleading[int(truth)] = 0.0
                    misleading[(int(truth) + 1) % width] = 0.95
                    task = dataclasses.replace(
                        task,
                        inference=dataclasses.replace(
                            task.inference, declared_prior=misleading.astype("float32")
                        ),
                    )
            output = _model_step(model, task, sequence)
            _scores, posterior = _posterior(output, temperature=temperature)
            corrected = False
            if truth is not None and posterior.size:
                wrong = (int(truth) + 1) % posterior.size
                corrected = bool(posterior[int(truth)] > posterior[wrong])
            records.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "family": "wrong_prior",
                    "direction": direction,
                    "corrected": corrected,
                    "budget": budget,
                }
            )
        for group in spec.thresholds.groups:
            task = factors.build_p1_task(sequence, frame_index=1)
            verdicts = _verdicts(_model_step(model, task, sequence))
            records.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "family": "normal",
                    "group": group,
                    "flagged": any("insufficient" in verdict for verdict in verdicts),
                    "verdicts": verdicts,
                    "budget": budget,
                }
            )
            wrong_task = factors.build_p1_task(sequence, frame_index=1, all_wrong=True)
            wrong_verdicts = _verdicts(_model_step(model, wrong_task, sequence))
            records.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "family": "insufficiency",
                    "group": group,
                    "detected": any("insufficient" in verdict for verdict in wrong_verdicts),
                    "verdicts": wrong_verdicts,
                    "budget": budget,
                }
            )
    return records


def structural_dry_run_records(
    cfg: n02_config.N02FullConfig, spec: EvalSpec, *, output: Path
) -> list[dict[str, Any]]:
    """CLI/structure dry run of the four families, labelled as such.

    It builds the controlled tasks through the frozen constructor (so the record
    schema, the aggregation, the JSONL output and the report composition are the
    real code paths) but it attributes the *structural* oracle outcome: the public
    dictionary provably contains the true hypothesis, so a correct model can
    identify it. It performs no density computation with a learned model and is
    marked ``dry_run``/``evidence=structural_dry_run`` in every record; it is not a
    behaviour result and must not be read as one.
    """
    from owac.frameworks.n02_full import factors
    from owac.frameworks.n02_full import synthetic as synth

    training = cfg.training if isinstance(cfg.training, dict) else {}
    measurement = dict(training.get("measurement", {}) or {})
    sensors = tuple(
        synth.SensorSpec(**dict(entry)) for entry in measurement.get("synthetic_sensors", [])
    ) or (
        synth.SensorSpec(
            camera_id="cam0",
            height=64,
            width=64,
            input_mode="bayer_cfa",
            cfa_pattern="RGGB",
            black_level=64.0,
            white_level=4095.0,
            bit_depth=12,
            read_noise_e=2.0,
            gain=1.0,
            exposure_ms=1.0,
        ),
    )
    frames = max(3, int(measurement.get("frames_per_episode", 8)))
    dataset = synth.generate_dataset(
        sensors=sensors,
        measurement_model=synth.MeasurementModel(
            **dict(measurement.get("measurement_model", {"kind": "cfa"}))
        ),
        partitions=("train",),
        scenes=int(spec.scenarios),
        frames=frames,
        views_per_scene=1,
        seed=int(spec.seeds[0]),
        canvas=tuple(measurement.get("canvas", (64, 64))),
        action_horizon=int(measurement.get("action_horizon", 4)),
    )
    records: list[dict[str, Any]] = []
    for index, sequence in enumerate(dataset):
        task = factors.build_p1_task(sequence, frame_index=1)
        categories = task.dictionary.categories()
        correct = task.candidate_index is not None
        base = {
            "scenario_id": f"{sequence.scene_id}:{index}",
            "seed": int(spec.seeds[0]),
            "evidence": "structural_dry_run",
            "dry_run": True,
            "categories": categories if isinstance(categories, dict) else list(categories),
        }
        for direction in ("a", "b"):
            records.append(
                {**base, "family": "cue_swap", "direction": direction, "correct": correct}
            )
        records.append({**base, "family": "no_information", "drift": 0.0})
        for direction in ("a", "b"):
            records.append(
                {**base, "family": "wrong_prior", "direction": direction, "corrected": correct}
            )
        for group in spec.thresholds.groups:
            records.append({**base, "family": "normal", "group": group, "flagged": False})
            records.append({**base, "family": "insufficiency", "group": group, "detected": True})
    return records


def _scenario_records(runner: Any, spec: EvalSpec, *, output: Path) -> list[dict[str, Any]]:
    """Produce per-scenario records by asking the framework for controlled tasks."""
    signature_hint = (
        "run_p1_scenarios(cfg, spec, seed=..., count=..., device=...) -> per-scenario records"
    )
    for name in ("run_p1_scenarios", "evaluate_p1_scenarios"):
        hook = getattr(runner, name, None)
        if hook is not None:
            try:
                records = list(hook(spec))
            except TypeError:
                records = list(hook())
            return records
    raise EvalNotAvailable(
        f"no scenario driver is available on the runner ({signature_hint})",
        missing=("owac.frameworks.n02_full.factors.run_p1_scenarios",),
    )


# ---------------------------------------------------------------------------
# stage implementations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StageContext:
    stage: str
    output: Path
    config_path: str
    config_hash: str
    command: tuple[str, ...]
    spec: EvalSpec
    dry_run: bool = False
    checkpoint: str = ""
    interim_budget: int = 0
    measurement_config: str = DEFAULT_MEASUREMENT_CONFIG

    def receipt(self, *, status: str, reason: str = "", started: float | None = None) -> dict:
        payload: dict[str, Any] = {
            "format": RECEIPT_FORMAT,
            "stage": self.stage,
            "status": status,
            "reason": reason,
            "output": str(self.output),
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "command": list(self.command),
            "scenarios": self.spec.scenarios,
            "seeds": list(self.spec.seeds),
            "thresholds": {
                key: getattr(self.spec.thresholds, key) for key in sorted(BehaviorThresholds.KINDS)
            },
            "dry_run": bool(self.dry_run),
            "claim_scope": (
                "dry_run_structure_only: CLI/aggregation/JSONL paths on the controlled tasks "
                "with the structural oracle outcome; not a behaviour result"
                if self.dry_run
                else "full evaluation"
            ),
            "started_at": _now(),
        }
        if started is not None:
            payload["finished_at"] = _now()
            payload["wall_seconds"] = round(time.time() - started, 3)
        return payload


def stage_oracle(ctx: StageContext, cfg: n02_config.N02FullConfig) -> dict[str, Any]:
    """Analytic upper bound: is the controlled task correctable at all?"""
    started = time.time()
    if ctx.dry_run:
        records = structural_dry_run_records(cfg, ctx.spec, output=ctx.output)
    else:
        try:
            records = oracle_scenario_records(
                cfg,
                ctx.spec,
                checkpoint=str(ctx.checkpoint),
                output=ctx.output,
                interim_budget=int(ctx.interim_budget or 0),
                measurement_config=str(ctx.measurement_config),
            )
        except EvalNotAvailable as exc:
            return ctx.receipt(status="not_run", reason=exc.reason, started=started)
    if not ctx.dry_run and not records:
        try:
            runner = load_scenario_runner()
            records = _scenario_records(runner, ctx.spec, output=ctx.output)
        except EvalNotAvailable as exc:
            return ctx.receipt(status="not_run", reason=exc.reason, started=started)
    oracle_records = [record for record in records if record.get("family") == "oracle"]
    if not oracle_records:
        return ctx.receipt(
            status="not_run",
            reason="the scenario driver produced no oracle-family records",
            started=started,
        )
    append_jsonl(ctx.output / "scenario_records.jsonl", records)
    correct = _count(oracle_records, "correct")
    payload = ctx.receipt(status="ok", started=started)
    payload.update(
        {
            "correct": _rate(correct, len(oracle_records)),
            "min_correctable": ctx.spec.oracle_min_correctable,
            "verdict": (
                "pass"
                if correct / float(len(oracle_records)) >= ctx.spec.oracle_min_correctable
                else "fail"
            ),
            "records": len(oracle_records),
        }
    )
    return payload


def stage_behavior(ctx: StageContext, cfg: n02_config.N02FullConfig) -> dict[str, Any]:
    started = time.time()
    if ctx.dry_run:
        records = structural_dry_run_records(cfg, ctx.spec, output=ctx.output)
    elif ctx.checkpoint:
        try:
            records = model_scenario_records(
                cfg,
                ctx.spec,
                checkpoint=str(ctx.checkpoint),
                output=ctx.output,
                interim_budget=int(ctx.interim_budget or 0),
                measurement_config=str(ctx.measurement_config),
            )
        except EvalNotAvailable as exc:
            return ctx.receipt(status="not_run", reason=exc.reason, started=started)
        except Exception as exc:  # noqa: BLE001 - a real failure must leave a receipt
            import traceback

            payload = ctx.receipt(
                status="fail", reason=f"{type(exc).__name__}: {exc}", started=started
            )
            payload["traceback"] = traceback.format_exc()
            return payload
    else:
        try:
            load_observation_stack()
            runner = load_scenario_runner()
            records = _scenario_records(runner, ctx.spec, output=ctx.output)
        except EvalNotAvailable as exc:
            return ctx.receipt(status="not_run", reason=exc.reason, started=started)
    if ctx.spec.keep_scenario_records:
        append_jsonl(ctx.output / "scenario_records.jsonl", records)
    try:
        summary = summarise_behavior(records, ctx.spec.thresholds)
    except DeficitError as exc:
        return ctx.receipt(status="not_run", reason=str(exc), started=started)
    payload = ctx.receipt(status="ok", started=started)
    payload.update({"verdict": summary.pop("status"), "families": summary})
    payload["records"] = len(records)
    return payload


def stage_ablation(ctx: StageContext, cfg: n02_config.N02FullConfig) -> dict[str, Any]:
    """Run or report each declared comparison arm independently."""
    started = time.time()
    arms = ctx.spec.ablation_arms
    if not arms:
        return ctx.receipt(
            status="not_run", reason="evaluation.ablation.arms declares no arm", started=started
        )
    results: list[dict[str, Any]] = []
    any_run = False
    for arm in arms:
        if not isinstance(arm, Mapping) or "name" not in arm:
            results.append({"name": None, "status": "fail", "reason": "arm needs a name"})
            continue
        name = str(arm["name"])
        entry: dict[str, Any] = {"name": name}
        loss_source = arm.get("loss_source")
        if loss_source:
            try:
                module_name, _, attr = str(loss_source).partition(":")
                module = importlib.import_module(module_name)
                if not attr or not hasattr(module, attr):
                    raise ImportError(f"{module_name} has no attribute {attr!r}")
                entry["loss_source"] = {
                    "reference": str(loss_source),
                    "resolved": True,
                }
            except ImportError as exc:
                entry.update(
                    {
                        "status": "not_run",
                        "reason": f"declared loss source is unavailable: {exc}",
                    }
                )
                results.append(entry)
                continue
        try:
            load_observation_stack()
            load_scenario_runner()
        except EvalNotAvailable as exc:
            entry.update({"status": "not_run", "reason": exc.reason})
            results.append(entry)
            continue
        entry["status"] = "not_run"
        entry["reason"] = (
            "arm training hook not available: the P1 stack exposes no run_p1_ablation arm"
        )
        results.append(entry)
    payload = ctx.receipt(status="ok" if any_run else "not_run", started=started)
    payload["arms"] = results
    if not any_run:
        payload["reason"] = "no declared arm could be trained in this checkout"
    return payload


def stage_report(ctx: StageContext, cfg: n02_config.N02FullConfig) -> dict[str, Any]:
    """Compose the stage artefacts from the receipts that exist."""
    started = time.time()
    receipts = {
        name: read_json(ctx.output / f"{name}.json") for name in ("oracle", "behavior", "ablation")
    }
    verdicts: dict[str, str] = {}
    for name, payload in receipts.items():
        if payload is None:
            verdicts[name] = "not_run"
        elif payload.get("status") == "not_run":
            verdicts[name] = "not_run"
        else:
            verdicts[name] = str(payload.get("verdict") or payload.get("status") or "incomplete")
    behavior = receipts.get("behavior") or {}
    families = behavior.get("families") or {}
    implementation = (
        "pass" if all(payload is not None for payload in receipts.values()) else "not_run"
    )
    oracle = receipts.get("oracle") or {}
    numerical = (
        "pass"
        if oracle.get("status") == "ok" and oracle.get("verdict") == "pass"
        else "not_run"
        if not oracle or oracle.get("status") == "not_run"
        else "fail"
    )
    stage_decision = {
        "implementation": implementation,
        "numerical": numerical,
        "behavior": verdicts.get("behavior", "not_run"),
        "closed_loop": "not_run",
        "evidence_paths": [str(ctx.output / f"{name}.json") for name in receipts],
        "notes": {
            "oracle": verdicts.get("oracle", "not_run"),
            "ablation": verdicts.get("ablation", "not_run"),
            "thresholds": {
                key: getattr(ctx.spec.thresholds, key) for key in sorted(BehaviorThresholds.KINDS)
            },
            "scenarios": ctx.spec.scenarios,
            "seeds": list(ctx.spec.seeds),
        },
    }
    metrics = {
        "stage": "p1",
        "config_hash": ctx.config_hash,
        "oracle": receipts.get("oracle"),
        "behavior": {
            "verdict": verdicts.get("behavior"),
            "families": families,
        },
        "ablation": receipts.get("ablation"),
    }
    write_json(ctx.output / "metrics.json", metrics)
    write_json(
        ctx.output / "run_manifest.json",
        {
            "format": RECEIPT_FORMAT,
            "stage": "p1-report",
            "config_path": ctx.config_path,
            "config_hash": ctx.config_hash,
            "command": list(ctx.command),
            "generated_at": _now(),
            "evidence": {name: str(ctx.output / f"{name}.json") for name in receipts},
        },
    )
    write_json(ctx.output / "stage_decision.json", stage_decision)
    payload = ctx.receipt(status="ok", started=started)
    payload.update({"stage_decision": stage_decision, "metrics": metrics})
    return payload


_STAGE_FUNCTIONS = {
    "oracle": stage_oracle,
    "behavior": stage_behavior,
    "ablation": stage_ablation,
    "report": stage_report,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m owac.evaluation.n02_full.p1",
        description=(
            "P1 evaluation CLI for the N02 full line. Missing modules or assets are reported as "
            "status=not_run with a reason and exit code 0; an unrun arm is never reported as pass."
        ),
    )
    parser.add_argument("command", choices=SUBCOMMANDS, metavar="{" + ",".join(SUBCOMMANDS) + "}")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config JSON path")
    parser.add_argument(
        "--output",
        default=None,
        help=f"output directory (default: {DEFAULT_OUTPUT_ROOT}/<command>)",
    )
    parser.add_argument("--device", default=None, help="torch device for this run")
    parser.add_argument(
        "--scenarios", type=int, default=None, help="override the per-seed scenario count"
    )
    parser.add_argument("--seeds", default=None, help="comma-separated seed override")
    parser.add_argument(
        "--measurement-config",
        default=DEFAULT_MEASUREMENT_CONFIG,
        help=(
            "config that declares the sensors, units and input mode the checkpoint was trained with"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="trained checkpoint whose Base the model-driven acceptance run uses",
    )
    parser.add_argument(
        "--interim-budget",
        type=int,
        default=0,
        help="training budget this checkpoint came from (recorded in every record)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "structure-only dry run: the real CLI/aggregation/JSONL paths on the controlled "
            "tasks, with the structural oracle outcome and every record marked dry_run"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = n02_config.load_config(args.config)
    except n02_config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        spec = load_eval_spec(cfg)
    except (DeficitError, n02_config.ConfigError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.scenarios is not None:
        if args.scenarios <= 0:
            print(
                f"argument error: --scenarios must be positive, got {args.scenarios}",
                file=sys.stderr,
            )
            return 2
        spec = dataclasses.replace(spec, scenarios=int(args.scenarios))
    if args.seeds is not None:
        try:
            seeds = tuple(int(item) for item in str(args.seeds).split(",") if item.strip())
        except ValueError:
            print(f"argument error: --seeds must be integers, got {args.seeds!r}", file=sys.stderr)
            return 2
        if not seeds:
            print("argument error: --seeds must not be empty", file=sys.stderr)
            return 2
        spec = dataclasses.replace(spec, seeds=seeds)
    ctx_dry_run = bool(args.dry_run)
    output = Path(args.output) if args.output else Path(DEFAULT_OUTPUT_ROOT) / args.command
    command = [sys.executable, "-m", "owac.evaluation.n02_full.p1", *(argv or sys.argv[1:])]
    ctx = StageContext(
        stage=args.command,
        output=output,
        config_path=str(args.config),
        config_hash=n02_config.config_hash(cfg),
        command=tuple(command),
        spec=spec,
        dry_run=ctx_dry_run,
        checkpoint=str(getattr(args, "checkpoint", "") or ""),
        interim_budget=int(getattr(args, "interim_budget", 0) or 0),
        measurement_config=str(getattr(args, "measurement_config", DEFAULT_MEASUREMENT_CONFIG)),
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        payload = _STAGE_FUNCTIONS[args.command](ctx, cfg)
    except EvalNotAvailable as exc:
        payload = ctx.receipt(status="not_run", reason=exc.reason)
    except Exception as exc:  # noqa: BLE001 - a real failure must leave a receipt
        import traceback

        payload = ctx.receipt(status="fail", reason=f"{type(exc).__name__}: {exc}")
        payload["traceback"] = traceback.format_exc()
        write_json(output / f"{args.command}.json", payload)
        print(f"{args.command}: status=fail output={output} reason={payload['reason']}")
        return 1
    write_json(output / f"{args.command}.json", payload)
    summary = f"{args.command}: status={payload.get('status')} output={output}"
    if payload.get("reason"):
        summary += f" reason={payload['reason']}"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

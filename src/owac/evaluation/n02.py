"""Evaluation entry point for the N02 measurement-constrained recurrent field.

Command (the target command from the implementation prompt)::

    uv run --locked --extra encoder python -m owac.evaluation.n02 \\
        --config configs/encoders/n02/tiny.json \\
        --checkpoint outputs/n02-smoke/checkpoint.pt \\
        --suite mechanism --output outputs/n02-eval

Suites
------
``mechanism``
    The required trajectories and controls on the explicit two-hypothesis
    reference (:mod:`owac.frameworks.n02.analytic`), which is an exact implementation
    of the same scoring rule the learned encoder uses:

    * ``correct_prior`` -- a correct prior is not destroyed by weak evidence;
    * ``misleading_prior_corrected`` -- a strong wrong prior is corrected by weak
      but valid contradicting evidence;
    * ``unidentifiable_ambiguity`` -- evidence that cannot separate the two
      explanations keeps the ambiguity instead of answering arbitrarily.

    Controls: (i) re-submitting one observation must not raise the unique-evidence
    count or the posterior; (ii) the static case must equal the closed-form Bayes
    reference; (iii) forking one old state and feeding supporting / refuting /
    uninformative measurements must show that the answer does not drift merely
    because cycles passed.

    When the checkpoint's candidate count matches the two-hypothesis reference, the
    *learned* encoder is additionally run over the same sequence and its own
    per-step weights, cycle counts, state bytes and latency are recorded; otherwise
    that part is reported as skipped with the exact reason rather than replaced by
    the reference numbers.

``baselines``
    Runs the same measurement budget over every entry of
    ``owac.baselines.n02.BASELINE_REGISTRY``. The module is imported lazily; when it
    does not exist the suite reports it as unavailable instead of failing.

``readout``
    The two-incompatible-candidate probe: MAP selection vs grouped
    ``hypothesis_tokens`` vs a direct weighted-mean readout, reporting the
    wrong/interpolated action fraction from
    :func:`owac.frameworks.n02.readout.two_candidate_probe`.

Metrics are only recorded when they are actually measured. A quantity that this
run cannot measure is reported as ``null`` together with a ``reason``; it is never
filled in with a zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.core.encoder_contract import (
    CameraSpec,
    ComputeBudget,
    EmbodimentContext,
    MeasurementFrame,
)
from owac.frameworks.n02.config import N02Config, load_config
from owac.frameworks.n02.hypotheses import diversity_penalty
from owac.frameworks.n02.model import build_encoder
from owac.frameworks.n02.readout import (
    StructuredRepresentation,
    TokenReadout,
    hypothesis_tokens,
    map_hypothesis,
    two_candidate_probe,
)

#: One-dimensional measurement grid for the analytic fallback. Must stay below 16
#: so the encoder's grid-stride rule keeps the field at the frame's own resolution.
ANALYTIC_GRID_WIDTH = 8

#: Declared interpolation tolerance for "the readout produced a wrong action".
INTERPOLATION_TOLERANCE = 0.1

#: Reliability bins for the calibration table; bins with too few samples are
#: reported as insufficient rather than as a calibrated probability.
CALIBRATION_BINS = 5

#: Declared variance of one controlled cue observation. The two hypotheses'
#: profiles differ by 0.2 raw units at the cue coordinate, so one noiseless cue
#: contributes ``0.5 * 0.2**2 / 0.02 = 1.0`` nat of evidence: below the
#: ``log(0.95/0.05) = 2.94`` nat log-odds of a strong wrong prior, so no single
#: observation is decisive while a short sequence of them is.
CONTROLLED_CUE_VARIANCE = 0.02

#: Number of controlled cue observations used by the required trajectories.
CONTROLLED_CUE_COUNT = 5

SCENARIOS: dict[str, dict[str, Any]] = {
    "correct_prior": {
        "prior_odds": (0.9, 0.1),
        "true_hypothesis": 0,
        "cue_coordinate_fraction": 0.25,
        "declares_contradiction": False,
        "variant": "controlled",
        "description": (
            "correct prior (0.9 on the true explanation); "
            f"{CONTROLLED_CUE_COUNT} weak (1 nat each) supporting cues"
        ),
    },
    "misleading_prior_corrected": {
        "prior_odds": (0.05, 0.95),
        "true_hypothesis": 0,
        "cue_coordinate_fraction": 0.25,
        "declares_contradiction": True,
        "variant": "controlled",
        "description": (
            "strong wrong prior (0.95 on the false explanation, 2.94 nats of prior "
            f"log-odds) corrected by {CONTROLLED_CUE_COUNT} weak but valid "
            "contradicting cues (1 nat each): no single cue can flip it"
        ),
    },
    "unidentifiable_ambiguity": {
        "prior_odds": (1.0, 1.0),
        "true_hypothesis": 0,
        "cue_coordinate_fraction": 0.5,
        "declares_contradiction": False,
        "variant": "controlled",
        "description": (
            "uninformative prior (0.5/0.5); the cue sits where both hypotheses "
            "predict the same value, so the evidence cannot separate them and the "
            "ambiguity must survive every observation"
        ),
    },
    "misleading_prior_generator_noise": {
        "prior_odds": (0.05, 0.95),
        "true_hypothesis": 0,
        "cue_coordinate_fraction": 0.25,
        "declares_contradiction": True,
        "variant": "generator_default_noise",
        "description": (
            "the generator's own noisy sequence (cue variance 0.08, one cue): kept "
            "as the honest default-noise reference, corrected or not"
        ),
    },
}

# ----------------------------------------------------------------------------- helpers


def _module(name: str) -> tuple[Any | None, str]:
    import importlib

    try:
        return importlib.import_module(name), ""
    except ImportError as exc:
        return None, f"{name} unavailable: {exc}"


def _git_sha() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"p50": None, "p95": None, "n": 0}
    array = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "n": int(array.size),
    }


def _unmeasured(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


@dataclass(slots=True)
class Checkpoint:
    """What was loaded, including whether it can drive the mechanism trajectories."""

    path: Path | None = None
    payload: dict[str, Any] | None = None
    sha256: str | None = None
    step: int | None = None
    config: N02Config | None = None
    model: torch.nn.Module | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        return self.payload is not None


def load_checkpoint(path: Path | None, device: str) -> Checkpoint:
    if path is None:
        return Checkpoint()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = N02Config.from_dict(payload["config"])
    model = build_encoder(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return Checkpoint(
        path=Path(path),
        payload=payload,
        sha256=file_sha256(Path(path)),
        step=int(payload.get("step", -1)),
        config=config,
        model=model,
    )


def analytic_frames(case: Any, width: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One frame per observation, value (and validity) only at its sampling bin."""
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for observation in case.observations:
        frame = torch.zeros(1, 1, 1, width)
        mask = torch.zeros(1, 1, 1, width, dtype=torch.bool)
        index = int(round(observation.coordinate / case.geometry.extent * (width - 1)))
        index = min(max(index, 0), width - 1)
        if not observation.counts:
            masks.append(mask)
            values.append(frame)
            continue
        frame[0, 0, 0, index] = float(observation.value)
        mask[0, 0, 0, index] = True
        values.append(frame)
        masks.append(mask)
    return values, masks


def implied_location(analytic: Any, geometry: Any, hypothesis: int, samples: int = 401) -> float:
    """Normalised coordinate where ``hypothesis`` disagrees most with its competitor.

    This is the declared "where does this explanation say the decisive evidence
    is" location of the explicit two-hypothesis problem; the wrong-explanation
    location offset is the distance between the selected and the true explanation's
    implied location, in units of the geometry extent.
    """
    coordinates = torch.linspace(0.0, geometry.extent, samples, dtype=torch.float64)
    difference = analytic.forward_operator(
        hypothesis, coordinates, geometry
    ) - analytic.forward_operator(1 - hypothesis, coordinates, geometry)
    return float(coordinates[int(torch.argmax(difference))].item()) / float(geometry.extent)


# ----------------------------------------------------------------------------- mechanism


def build_scenario_case(analytic: Any, case_config: N02Config, scenario: str, *, seed: int) -> Any:
    """Build one declared evaluation sequence.

    ``controlled`` scenarios keep the generator's three ambiguous prefix
    observations and replace the noisy cue with ``CONTROLLED_CUE_COUNT`` noiseless
    cue observations of a declared variance. The evidence is weak because of that
    declared variance (1 nat each), not because of a lucky noise draw; the
    generator's own noisy sequence is kept as a separate scenario.
    """
    specification = SCENARIOS[scenario]
    case = analytic.delayed_evidence_case(
        case_config,
        prior_odds=specification["prior_odds"],
        true_hypothesis=specification["true_hypothesis"],
        delay=3,
        seed=seed,
    )
    if specification["variant"] != "controlled":
        return case
    coordinate = specification["cue_coordinate_fraction"] * case.geometry.extent
    value = float(
        analytic.forward_operator(
            specification["true_hypothesis"],
            torch.tensor([coordinate], dtype=torch.float64),
            case.geometry,
        ).item()
    )
    observations = list(case.observations[:3])
    for index in range(CONTROLLED_CUE_COUNT):
        observations.append(
            analytic.AnalyticObservation(
                value=value,
                coordinate=coordinate,
                variance=CONTROLLED_CUE_VARIANCE,
                identity=("cam0", 3 + index, 0),
            )
        )
    return replace(case, observations=observations, evidence_step=3)


def float64_bayes_reference(case: Any, case_config: N02Config) -> list[float]:
    """Closed-form posterior recomputed in Python float64 from the declaration.

    ``analytic.AnalyticCase.exact_posterior`` is the framework's own reference but
    rounds each per-hypothesis NLL through ``torch.tensor(float)`` (float32
    default), which costs about 1e-8 in the weights; see the report's limitations.
    This function recomputes ``softmax(-(NLL_k - beta * log prior_k) / T)`` in
    float64 so the implementation can be checked at float64 precision as well. The
    per-hypothesis constant ``0.5 * log variance`` cancels in the softmax and is
    therefore omitted, exactly as in the framework's own closed form.
    """
    import math

    nll = case.per_hypothesis_nll()
    logits = [
        -(nll[index] - case_config.beta_prior * math.log(case.prior_probability[index]))
        / case_config.temperature
        for index in (0, 1)
    ]
    shift = max(logits)
    exponentials = [math.exp(value - shift) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _scenario_trajectory(
    analytic: Any, case: Any, case_config: N02Config, scenario: str
) -> dict[str, Any]:
    """Per-observation trace of the exact N02 scoring rule on a growing buffer.

    Each step adds exactly one *new* observation and re-scores the frozen set, so
    the trace shows both the required trajectory and the fact that cycles cannot
    add evidence. Two independent references are computed per step: the
    framework's own ``exact_posterior`` and a float64 recomputation of the closed
    form (see :func:`float64_bayes_reference`).
    """
    specification = SCENARIOS[scenario]
    steps: list[dict[str, Any]] = []
    for count in range(1, len(case.observations) + 1):
        prefix = replace(case, observations=case.observations[:count])
        buffer = analytic.build_buffer(prefix)
        weights, diagnostics = analytic.analytic_posterior(prefix, buffer, case_config)
        exact = torch.tensor(prefix.exact_posterior(), dtype=torch.float64)
        reference = torch.tensor(float64_bayes_reference(prefix, case_config), dtype=torch.float64)
        contradiction = prefix.contradiction_flags(case_config.revision.contradiction_threshold)
        measured = weights[0].to(torch.float64)
        steps.append(
            {
                "observation_index": count - 1,
                "weights": [round(float(value), 12) for value in measured.tolist()],
                "exact_posterior": [round(float(value), 12) for value in exact.tolist()],
                "float64_bayes_posterior": [
                    round(float(value), 12) for value in reference.tolist()
                ],
                "max_abs_deviation_from_exact": float((measured - exact).abs().max().item()),
                "max_abs_deviation_from_float64_bayes": float(
                    (measured - reference).abs().max().item()
                ),
                "measurement_energy": diagnostics["measurement_energy"],
                "prior_probability": diagnostics["prior_probability"],
                "prior_energy": diagnostics["prior_energy"],
                "logits": diagnostics["logits"],
                "unique_evidence_count": buffer.unique_evidence_count,
                "duplicates_skipped": diagnostics["duplicates_skipped"],
                "observations_used": diagnostics["observations_used"],
                "contradiction_flags": list(contradiction),
                "top_index": int(torch.argmax(measured).item()),
                "top_weight": float(measured.max().item()),
                "true_hypothesis": int(case.true_hypothesis),
                "is_correct_top": bool(int(torch.argmax(measured).item()) == case.true_hypothesis),
            }
        )
    final = steps[-1]
    truth = int(case.true_hypothesis)
    wrong = 1 - truth
    correction_index = next(
        (
            step["observation_index"]
            for step in steps
            if step["is_correct_top"] and step["top_weight"] > 0.5
        ),
        None,
    )
    evidence_step = case.evidence_step
    already_correct = bool(
        correction_index is None or evidence_step is None or correction_index < evidence_step
    )
    post = [
        step
        for step in steps
        if evidence_step is not None and step["observation_index"] >= evidence_step
    ]
    location = {
        name: implied_location(analytic, case.geometry, index)
        for name, index in (("true", truth), ("other", wrong))
    }
    top_location = location["true"] if final["is_correct_top"] else location["other"]
    return {
        "scenario": scenario,
        "name": case.name,
        "variant": specification["variant"],
        "declares_contradiction": bool(specification["declares_contradiction"]),
        "prior_odds": list(case.prior_odds),
        "prior_probability": list(case.prior_probability),
        "prior_is_correct": bool(int(np.argmax(case.prior_probability)) == truth),
        "true_hypothesis": truth,
        "evidence_step": evidence_step,
        "observations": len(case.observations),
        "steps": steps,
        "final": final,
        "final_is_true": bool(final["is_correct_top"]),
        "final_top_weight": final["top_weight"],
        "correction_observation_index": correction_index,
        "already_correct_before_contradiction": already_correct,
        "contradicting_observations_to_correct": (
            None
            if correction_index is None or already_correct
            else correction_index - int(evidence_step) + 1
        ),
        "wrong_explanation_duration_steps": int(
            sum(1 for step in steps if not step["is_correct_top"])
        ),
        "wrong_explanation_area": float(sum(step["weights"][wrong] for step in steps)),
        "post_contradiction_wrong_fraction": (
            None
            if not specification["declares_contradiction"] or not post
            else float(sum(1 for step in post if not step["is_correct_top"]) / len(post))
        ),
        "post_contradiction_steps": len(post) if specification["declares_contradiction"] else 0,
        "true_retained_final": bool(final["weights"][truth] >= 0.25),
        "two_hypotheses_retained_final": bool(
            min(final["weights"]) >= 0.25 and final["weights"][truth] >= 0.25
        ),
        "ambiguity_final_entropy": float(
            -sum(value * np.log(max(value, 1e-12)) for value in final["weights"])
        ),
        "implied_location_true": location["true"],
        "implied_location_other": location["other"],
        "wrong_location_offset_normalised": float(abs(top_location - location["true"])),
        "measurement_residual_nats_per_observation": float(
            final["measurement_energy"][final["top_index"]] / max(1, final["observations_used"])
        ),
    }


def _repeated_observation_control(
    analytic: Any, case: Any, case_config: N02Config
) -> dict[str, Any]:
    report = analytic.repeated_observation_check(case, case_config)
    identical = bool(report["posterior_identical"])
    grew = bool(report["unique_evidence_duplicated"] > report["unique_evidence_single"])
    return {
        "unique_evidence_single": report["unique_evidence_single"],
        "unique_evidence_duplicated": report["unique_evidence_duplicated"],
        "unique_evidence_grew": grew,
        "duplicates_recorded": report["duplicates_recorded"],
        "posterior_identical": identical,
        "posterior_single": report["posterior_single"],
        "posterior_duplicated": report["posterior_duplicated"],
        "measurement_energy_single": report["measurement_energy_single"],
        "measurement_energy_duplicated": report["measurement_energy_duplicated"],
        "passed": bool(identical and not grew),
    }


def _fork_control(
    analytic: Any, case_config: N02Config, *, seed: int, delay: int
) -> dict[str, Any]:
    """Fork one old state; supporting / refuting / uninformative measurements.

    All three variants start from the *same* ambiguous prefix, so any change is
    caused by the new measurement and not by cycles having passed.
    """
    case = analytic.delayed_evidence_case(case_config, delay=delay, seed=seed)
    prefix = case.observations[:delay]
    truth = int(case.true_hypothesis)
    cue_coordinate = 0.25 * case.geometry.extent
    shared_coordinate = 0.5 * case.geometry.extent
    true_cue = float(
        analytic.forward_operator(truth, torch.tensor([cue_coordinate]), case.geometry).item()
    )
    false_cue = float(
        analytic.forward_operator(1 - truth, torch.tensor([cue_coordinate]), case.geometry).item()
    )
    shared_value = float(
        analytic.forward_operator(truth, torch.tensor([shared_coordinate]), case.geometry).item()
    )

    def posterior(observations: list[Any], cycles: int) -> tuple[list[float], dict[str, Any]]:
        variant_config = replace(case_config, cycles=cycles)
        variant_config.measurement = replace(case_config.measurement)
        variant = replace(case, observations=list(observations), config=variant_config)
        buffer = analytic.build_buffer(variant)
        weights, diagnostics = analytic.analytic_posterior(variant, buffer, variant_config)
        return [float(value) for value in weights[0].tolist()], diagnostics

    def observation(value: float, variance: float, index: int) -> Any:
        return analytic.AnalyticObservation(
            value=value,
            coordinate=cue_coordinate if index != 99 else shared_coordinate,
            variance=variance,
            identity=("cam0", len(prefix) + index, 0),
        )

    base, base_diagnostics = posterior(prefix, case_config.cycles)
    supporting, _ = posterior([*prefix, observation(true_cue, 0.08, 0)], case_config.cycles)
    refuting, _ = posterior([*prefix, observation(false_cue, 0.08, 1)], case_config.cycles)
    uninformative, uninformative_diagnostics = posterior(
        [
            *prefix,
            analytic.AnalyticObservation(
                value=shared_value,
                coordinate=shared_coordinate,
                variance=1.0,
                identity=("cam0", len(prefix) + 2, 0),
            ),
        ],
        case_config.cycles,
    )
    one_cycle, one_cycle_diagnostics = posterior(prefix, 1)
    many_cycles, many_cycles_diagnostics = posterior(prefix, max(2, case_config.cycles * 2))
    uninformative_delta = float(max(abs(a - b) for a, b in zip(base, uninformative, strict=True)))
    cycles_delta = float(max(abs(a - b) for a, b in zip(base, one_cycle, strict=True)))
    cycles_delta_wide = float(max(abs(a - b) for a, b in zip(base, many_cycles, strict=True)))
    return {
        "fork_point_observations": len(prefix),
        "true_hypothesis": truth,
        "base_posterior": base,
        "supporting_posterior": supporting,
        "refuting_posterior": refuting,
        "uninformative_posterior": uninformative,
        "supporting_moved_towards_truth": bool(supporting[truth] > base[truth]),
        "refuting_moved_away_from_truth": bool(refuting[truth] < base[truth]),
        "uninformative_posterior_delta": uninformative_delta,
        "uninformative_left_answer_unchanged": bool(uninformative_delta <= 1e-12),
        "one_cycle_posterior_delta": cycles_delta,
        "double_cycles_posterior_delta": cycles_delta_wide,
        "cycles_change_answer": bool(max(cycles_delta, cycles_delta_wide) > 1e-12),
        "no_drift_from_cycles": bool(
            uninformative_delta <= 1e-12 and max(cycles_delta, cycles_delta_wide) <= 1e-12
        ),
        "measurement_energy_base": base_diagnostics["measurement_energy"],
        "measurement_energy_uninformative": uninformative_diagnostics["measurement_energy"],
        "measurement_energy_one_cycle": one_cycle_diagnostics["measurement_energy"],
        "measurement_energy_many_cycles": many_cycles_diagnostics["measurement_energy"],
        "note": (
            "the exact scoring rule has no iterative refinement, so its answer is "
            "cycle-count independent by construction; the learned encoder is checked "
            "separately when a matching checkpoint is available"
        ),
    }


def _probability_metrics(trajectories: dict[str, Any]) -> dict[str, Any]:
    """NLL/Brier/calibration of the weights, which are exact posteriors here."""
    log_losses: list[float] = []
    briers: list[float] = []
    confidences: list[float] = []
    outcomes: list[float] = []
    for trajectory in trajectories.values():
        for step in trajectory["steps"]:
            truth = trajectory["true_hypothesis"]
            probability = float(step["weights"][truth])
            log_losses.append(-float(np.log(max(probability, 1e-12))))
            briers.append((probability - 1.0) ** 2)
            confidences.append(float(step["top_weight"]))
            outcomes.append(1.0 if step["is_correct_top"] else 0.0)
    bins: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    confidence_array = np.asarray(confidences)
    outcome_array = np.asarray(outcomes)
    for index in range(CALIBRATION_BINS):
        low, high = float(edges[index]), float(edges[index + 1])
        selected = (confidence_array > low) & (confidence_array <= high)
        count = int(selected.sum())
        bins.append(
            {
                "bin": [low, high],
                "count": count,
                "mean_confidence": float(confidence_array[selected].mean()) if count else None,
                "empirical_accuracy": float(outcome_array[selected].mean()) if count else None,
                "sufficient": bool(count >= 5),
            }
        )
    return {
        "basis": (
            "analytic two-hypothesis posterior under the declared Gaussian measurement "
            "model; these weights are posterior probabilities by construction and are "
            "checked against the closed-form reference above"
        ),
        "n_samples": len(log_losses),
        "nll": float(np.mean(log_losses)) if log_losses else None,
        "brier": float(np.mean(briers)) if briers else None,
        "mean_top_confidence": float(np.mean(confidences)) if confidences else None,
        "top1_accuracy": float(np.mean(outcomes)) if outcomes else None,
        "calibration_bins": bins,
        "calibration_note": (
            "bins with fewer than 5 samples are marked insufficient and are not read "
            "as calibration evidence"
        ),
    }


def _learned_trace(
    checkpoint: Checkpoint, analytic: Any, case: Any, case_config: N02Config, device: str
) -> dict[str, Any]:
    """Run the learned encoder over the same sequence, when its shape allows it."""
    if not checkpoint.loaded or checkpoint.config is None or checkpoint.model is None:
        return {
            "status": "skipped",
            "reason": "no checkpoint was supplied, so no learned trajectory exists",
        }
    config = checkpoint.config
    if config.K != case.geometry.K:
        return {
            "status": "skipped",
            "reason": (
                f"checkpoint K={config.K} does not match the explicit two-hypothesis "
                f"reference K={case.geometry.K}; the reference trajectory above is the "
                "only measured trajectory for this run"
            ),
        }
    if config.measurement.input_mode not in {"bayer_cfa", "rawvla_rgb3"}:
        return {"status": "skipped", "reason": "unsupported measurement input mode"}
    width = ANALYTIC_GRID_WIDTH
    values, masks = analytic_frames(case, width)
    camera = CameraSpec(
        camera_id="analytic_measurement",
        height=1,
        width=width,
        channels=1,
        input_mode="bayer_cfa",
        cfa_pattern="RGGB",
    )
    if config.measurement.input_mode != "bayer_cfa":
        return {
            "status": "skipped",
            "reason": (
                "the analytic reference has one measurement channel but the checkpoint "
                f"declares input_mode={config.measurement.input_mode!r}"
            ),
        }
    model = checkpoint.model
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    state = model.init_state(1, camera, device)
    steps: list[dict[str, Any]] = []
    latencies: list[float] = []
    with torch.no_grad():
        for index in range(len(values)):
            frame = MeasurementFrame(
                values=values[index],
                valid_mask=masks[index],
                frame_ids=(index,),
                episode_ids=(0,),
                camera_id=camera.camera_id,
            )
            start = time.perf_counter()
            output = model.step(frame, EmbodimentContext(), state, budget)
            latencies.append((time.perf_counter() - start) * 1000.0)
            state = model.detach_state(output.state)
            representation = output.representation
            weights = representation.weights.mean(dim=1)[0]
            steps.append(
                {
                    "observation_index": index,
                    "weights": [round(float(value), 12) for value in weights.tolist()],
                    "top_index": int(torch.argmax(weights).item()),
                    "top_weight": float(weights.max().item()),
                    "is_correct_top": bool(
                        int(torch.argmax(weights).item()) == int(case.true_hypothesis)
                    ),
                    "cycles_run": int(output.diagnostics["cycles_run"]),
                    "observation_history": int(output.diagnostics["observation_history"]),
                    "state_bytes": int(output.cost.state_bytes or 0),
                    "stop_reason": str(representation.diagnostics["stop_reason"]),
                }
            )
    distance, collapsed = diversity_penalty(representation.h)
    return {
        "status": "ran",
        "checkpoint_step": checkpoint.step,
        "checkpoint_sha256": checkpoint.sha256,
        "weights_are_trained": True,
        "steps": steps,
        "final_is_true": steps[-1]["is_correct_top"],
        "correction_observation_index": next(
            (step["observation_index"] for step in steps if step["is_correct_top"]), None
        ),
        "internal_iterations_at_first_correct": next(
            (step["cycles_run"] for step in steps if step["is_correct_top"]), None
        ),
        "state_bytes": steps[-1]["state_bytes"],
        "candidate_mean_pairwise_distance": float(distance.item()),
        "candidate_collapse_fraction": float(collapsed.item()),
        "encoder_step_ms": _percentiles(latencies),
        "note": (
            "the learned trace is reported next to, not instead of, the reference "
            "trajectory; it is not evidence of a trained mechanism claim by itself"
        ),
    }


def _learned_synthetic_trace(
    checkpoint: Checkpoint, config: N02Config, *, seed: int, device: str
) -> dict[str, Any]:
    """Run the learned encoder over the three scenario classes on synthetic episodes.

    The analytic reference above is a two-hypothesis problem, so a checkpoint with
    ``K > 2`` cannot drive it. This trace covers that case with measurement-domain
    synthetic episodes of the same three classes (correct prior, misleading prior,
    and an episode whose cue amplitude is exactly zero). It reports *diagnostics*
    of the field -- per-step candidate weights, entropy, cycles, state bytes and the
    collapse fraction -- and deliberately makes no accuracy claim: which candidate
    slot corresponds to which latent value is a training-time convention, not
    something this trace establishes.
    """
    if not checkpoint.loaded or checkpoint.model is None:
        return {
            "status": "skipped",
            "reason": "no checkpoint was supplied, so no learned trajectory exists",
        }
    if config.measurement.input_mode != "rawvla_rgb3":
        return {
            "status": "skipped",
            "reason": (
                "the synthetic trace needs three-channel frames but the checkpoint "
                f"declares input_mode={config.measurement.input_mode!r}"
            ),
        }
    synthetic, reason = _module("owac.frameworks.n02.synthetic")
    if synthetic is None:
        return {"status": "skipped", "reason": reason}
    generator = synthetic.SyntheticGenerator(
        synthetic.SyntheticOptions(height=64, width=64, channels=3), seed=seed, split="validation"
    )
    scenarios = {
        "correct_prior": {"prior_condition": "correct", "identifiable": True},
        "misleading_prior": {"prior_condition": "misleading", "identifiable": True},
        "unidentifiable": {"prior_condition": "uninformative", "identifiable": False},
    }
    camera = CameraSpec(
        camera_id="synthetic_measurement",
        height=64,
        width=64,
        channels=3,
        input_mode="rawvla_rgb3",
    )
    model = checkpoint.model
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    traces: dict[str, Any] = {}
    from owac.training.n02 import episode_prior_context

    for name, keywords in scenarios.items():
        episode = generator.generate(num_frames=6, episode_index=1, seed=seed, **keywords)
        state = model.init_state(1, camera, device)
        steps: list[dict[str, Any]] = []
        latencies: list[float] = []
        with torch.no_grad():
            for index in range(episode.num_frames):
                frame = MeasurementFrame(
                    values=torch.from_numpy(episode.frames[index]).unsqueeze(0).to(torch.float32),
                    frame_ids=(index,),
                    episode_ids=(0,),
                    camera_id=camera.camera_id,
                    metadata={"variance": generator.options.noise_sigma**2},
                    metadata_available={"variance": True},
                )
                start = time.perf_counter()
                output = model.step(
                    frame, episode_prior_context([episode], config.K), state, budget
                )
                latencies.append((time.perf_counter() - start) * 1000.0)
                state = model.detach_state(output.state)
                weights = output.representation.weights.mean(dim=1)[0]
                safe = weights.clamp_min(1e-12)
                steps.append(
                    {
                        "frame_index": index,
                        "weights": [round(float(value), 8) for value in weights.tolist()],
                        "top_weight": float(weights.max().item()),
                        "entropy": float(-(safe * safe.log()).sum().item()),
                        "uniform_entropy": float(np.log(max(config.K, 1))),
                        "cycles_run": int(output.diagnostics["cycles_run"]),
                        "state_bytes": int(output.cost.state_bytes or 0),
                        "solved": bool(output.validity.solved),
                        "stop_reason": str(output.representation.diagnostics["stop_reason"]),
                    }
                )
        distance, collapsed = diversity_penalty(output.representation.h)
        traces[name] = {
            "episode_id": episode.episode_id,
            "identifiable": bool(episode.identifiable),
            "cue_strength_max": float(np.max(episode.cue_strength)),
            "evidence_arrival_step": int(episode.evidence_arrival_step),
            "steps": steps,
            "encoder_step_ms": _percentiles(latencies),
            "final_top_weight": steps[-1]["top_weight"],
            "final_entropy": steps[-1]["entropy"],
            "final_unresolved": not steps[-1]["solved"],
            "candidate_mean_pairwise_distance": float(distance.item()),
            "candidate_collapse_fraction": float(collapsed.item()),
        }
    finals = {name: trace["steps"][-1]["weights"] for name, trace in traces.items()}
    separations = {
        f"{left}|{right}": float(
            max(abs(a - b) for a, b in zip(finals[left], finals[right], strict=True))
        )
        for left in finals
        for right in finals
        if left < right
    }
    return {
        "status": "ran",
        "checkpoint_step": checkpoint.step,
        "checkpoint_sha256": checkpoint.sha256,
        "weights_are_trained": True,
        "scenarios": traces,
        "final_weight_separation": separations,
        "final_weight_separation_note": (
            "largest absolute weight difference between the scenario classes at the "
            "last frame; a small number means this checkpoint does not separate the "
            "classes on these episodes and is reported as such"
        ),
        "note": (
            "diagnostics only: candidate-to-latent-value anchoring is a training-time "
            "convention and is not established by this trace, so no accuracy metric is "
            "derived from it here. The analytic trajectory validates its own update "
            "law and cannot establish the learned encoder's mechanism claim."
        ),
    }


def run_mechanism(
    config: N02Config, checkpoint: Checkpoint, *, seed: int, device: str
) -> dict[str, Any]:
    analytic, reason = _module("owac.frameworks.n02.analytic")
    if analytic is None:  # pragma: no cover - ships with the framework
        return {"status": "unavailable", "reason": reason}
    case_config = analytic.make_config(
        K=2,
        cycles=config.cycles,
        history=config.history,
        beta_prior=config.beta_prior,
        temperature=config.temperature,
        robust="gaussian",
    )
    trajectories: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    for name, specification in SCENARIOS.items():
        case = build_scenario_case(analytic, case_config, name, seed=seed)
        cases[name] = case
        trajectories[name] = _scenario_trajectory(analytic, case, case_config, name)
        trajectories[name]["description"] = specification["description"]
    deviations = [
        step["max_abs_deviation_from_exact"]
        for trajectory in trajectories.values()
        for step in trajectory["steps"]
    ]
    float64_deviations = [
        step["max_abs_deviation_from_float64_bayes"]
        for trajectory in trajectories.values()
        for step in trajectory["steps"]
    ]
    repeated = _repeated_observation_control(analytic, cases["correct_prior"], case_config)
    fork = _fork_control(analytic, case_config, seed=seed, delay=3)
    learned = _learned_trace(
        checkpoint, analytic, cases["misleading_prior_corrected"], case_config, device
    )
    misleading = trajectories["misleading_prior_corrected"]
    unidentifiable = trajectories["unidentifiable_ambiguity"]
    correct = trajectories["correct_prior"]
    post_rates = [
        trajectory["post_contradiction_wrong_fraction"]
        for trajectory in trajectories.values()
        if trajectory["post_contradiction_wrong_fraction"] is not None
    ]
    metrics = {
        "wrong_completion_rate_post_contradiction": (
            None if not post_rates else float(sum(post_rates) / len(post_rates))
        ),
        "wrong_explanation_duration_steps": {
            name: trajectory["wrong_explanation_duration_steps"]
            for name, trajectory in trajectories.items()
        },
        "wrong_explanation_area": {
            name: trajectory["wrong_explanation_area"] for name, trajectory in trajectories.items()
        },
        "true_geometry_coverage": float(
            sum(1 for trajectory in trajectories.values() if trajectory["true_retained_final"])
            / len(trajectories)
        ),
        "true_geometry_coverage_definition": (
            "share of scenarios whose final posterior keeps the true explanation at "
            "weight >= 0.25 (retained, not necessarily top-1)"
        ),
        "contradicting_observations_to_correct": {
            name: trajectory["contradicting_observations_to_correct"]
            for name, trajectory in trajectories.items()
        },
        "contradicting_observations_to_correct_definition": (
            "number of post-contradiction observations consumed before the true "
            "explanation became the top answer with weight > 0.5; 0/None means it was "
            "already correct before the contradiction arrived"
        ),
        "correction_observation_index": {
            name: trajectory["correction_observation_index"]
            for name, trajectory in trajectories.items()
        },
        "wrong_location_offset_normalised": {
            name: trajectory["wrong_location_offset_normalised"]
            for name, trajectory in trajectories.items()
        },
        "wrong_location_offset_definition": (
            "distance between the implied decisive location of the final top "
            "explanation and of the true explanation, in units of the geometry extent"
        ),
        "measurement_residual_nats_per_observation": {
            name: trajectory["measurement_residual_nats_per_observation"]
            for name, trajectory in trajectories.items()
        },
        "ambiguity_retained_unidentifiable": {
            "final_weights": unidentifiable["final"]["weights"],
            "min_weight": min(unidentifiable["final"]["weights"]),
            "top_weight": unidentifiable["final_top_weight"],
            "entropy": unidentifiable["ambiguity_final_entropy"],
            "retained": bool(unidentifiable["two_hypotheses_retained_final"]),
        },
        "correct_prior_final": {
            "final_weights": correct["final"]["weights"],
            "top_weight": correct["final_top_weight"],
            "final_is_true": correct["final_is_true"],
        },
        "misleading_prior_corrected": {
            "final_weights": misleading["final"]["weights"],
            "final_is_true": misleading["final_is_true"],
            "correction_observation_index": misleading["correction_observation_index"],
            "contradicting_observations_to_correct": misleading[
                "contradicting_observations_to_correct"
            ],
        },
        "generator_noise_reference": {
            "final_weights": trajectories["misleading_prior_generator_noise"]["final"]["weights"],
            "final_is_true": trajectories["misleading_prior_generator_noise"]["final_is_true"],
            "keep_note": (
                "the generator's own noisy single-cue sequence is kept as a negative or "
                "positive result exactly as measured; it is not a corrected trajectory"
            ),
        },
        "exact_static_case_max_abs_deviation": {
            "max": max(deviations),
            "tolerance": 1e-6,
            "passed": bool(max(deviations) <= 1e-6),
            "reference": "owac.frameworks.n02.analytic.AnalyticCase.exact_posterior",
            "cause_of_loose_tolerance": (
                "the framework reference rounds each per-hypothesis NLL through "
                "torch.tensor(float) (float32 default), which costs about 1e-8 in the "
                "weights; see float64_bayes_max_abs_deviation for the tight check"
            ),
        },
        "float64_bayes_max_abs_deviation": {
            "max": max(float64_deviations),
            "tolerance": 1e-12,
            "passed": bool(max(float64_deviations) <= 1e-12),
            "reference": "softmax(-(NLL_k - beta*log prior_k)/T) recomputed in float64",
        },
        "unaffected_answer_drift": _unmeasured(
            "the analytic reference keeps no spatial state; the drift of the retained "
            "explanation is reported by the fork control instead"
        ),
        "candidate_collapse_fraction": (
            None if learned.get("status") != "ran" else learned["candidate_collapse_fraction"]
        ),
        "state_bytes": (None if learned.get("status") != "ran" else learned["state_bytes"]),
        "encoder_latency_ms": (
            None if learned.get("status") != "ran" else learned["encoder_step_ms"]
        ),
        "readout_latency_ms": _unmeasured("the readout suite reports readout latency"),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    metrics["probability_metrics"] = _probability_metrics(trajectories)
    controls = {
        "repeated_observation": repeated,
        "exact_static_case": metrics["exact_static_case_max_abs_deviation"],
        "fork_from_same_old_state": fork,
    }
    return {
        "status": "ran",
        "source": "owac.frameworks.n02.analytic (explicit two-hypothesis reference)",
        "case_config": case_config.as_dict(),
        "trajectories": trajectories,
        "controls": controls,
        "metrics": metrics,
        "learned_encoder_trace": learned,
        "learned_encoder_trace_synthetic": _learned_synthetic_trace(
            checkpoint, config, seed=seed, device=device
        ),
    }


# ----------------------------------------------------------------------------- baselines


def _invoke_baseline(entry: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one registry entry under the shared measurement budget.

    Two entry shapes are supported and stated here so the contract is explicit:
    an Encoder-like builder (``entry(config)`` -> object with ``init_state``/``step``),
    or a callable taking the declared payload dict. Anything else is reported as
    unsupported rather than silently skipped.
    """
    if isinstance(entry, type) or callable(entry):
        try:
            candidate = entry(payload["config"])
        except TypeError as exc:
            return {
                "status": "unsupported-interface",
                "reason": f"entry(config) raised TypeError: {exc}",
            }
        if hasattr(candidate, "init_state") and hasattr(candidate, "step"):
            return {"status": "encoder-protocol", "object": candidate}
        return {
            "status": "unsupported-interface",
            "reason": f"entry(config) returned {type(candidate).__name__} without step/init_state",
        }
    if hasattr(entry, "run"):
        try:
            return {"status": "ran", "result": entry.run(payload)}
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": "unsupported-interface", "reason": f"unsupported entry type {type(entry)}"}


def run_baselines(config: N02Config, *, seed: int, device: str) -> dict[str, Any]:
    module, reason = _module("owac.baselines.n02")
    if module is None:
        return {
            "status": "unavailable",
            "reason": reason,
            "registry_entries": [],
            "note": (
                "the baseline registry is written by another teammate; this suite "
                "reports it as unavailable instead of failing the run"
            ),
        }
    registry = getattr(module, "BASELINE_REGISTRY", None)
    if registry is None:
        return {
            "status": "unavailable",
            "reason": "owac.baselines.n02 has no BASELINE_REGISTRY",
            "registry_entries": [],
        }
    analytic, analytic_reason = _module("owac.frameworks.n02.analytic")
    if analytic is None:  # pragma: no cover
        return {"status": "unavailable", "reason": analytic_reason, "registry_entries": []}
    budget = {
        "config": replace(config, K=2),
        "seed": seed,
        "sequences": 3,
        "observations_per_sequence": 4,
        "measurement": "analytic two-hypothesis (explicit delayed evidence)",
        "device": device,
    }
    budget["config"].measurement = replace(config.measurement)
    budget["config"].measurement.input_mode = "bayer_cfa"
    case_config = analytic.make_config(K=2, cycles=config.cycles, history=config.history)
    case = analytic.delayed_evidence_case(case_config, delay=3, seed=seed)
    reference_buffer = analytic.build_buffer(case)
    reference_weights, _ = analytic.analytic_posterior(case, reference_buffer, case_config)
    outcomes: dict[str, Any] = {}
    for name, entry in registry.items():
        outcome = _invoke_baseline(entry, budget)
        if outcome.get("status") == "encoder-protocol":
            candidate = outcome.pop("object")
            try:
                spec = CameraSpec(
                    camera_id="analytic_measurement",
                    height=1,
                    width=ANALYTIC_GRID_WIDTH,
                    channels=1,
                    input_mode="bayer_cfa",
                    cfa_pattern="RGGB",
                )
                values, masks = analytic_frames(case, ANALYTIC_GRID_WIDTH)
                state = candidate.init_state(1, spec, device)
                compute = ComputeBudget(max_updates=config.cycles, max_history=config.history)
                latencies: list[float] = []
                weights = None
                with torch.no_grad():
                    for index, value in enumerate(values):
                        frame = MeasurementFrame(
                            values=value,
                            valid_mask=masks[index],
                            frame_ids=(index,),
                            episode_ids=(0,),
                            camera_id=spec.camera_id,
                        )
                        start = time.perf_counter()
                        step_output = candidate.step(frame, EmbodimentContext(), state, compute)
                        latencies.append((time.perf_counter() - start) * 1000.0)
                        state = candidate.detach_state(step_output.state)
                        weights = getattr(step_output.representation, "weights", None)
                outcome["status"] = "ran"
                outcome["latency_ms"] = _percentiles(latencies)
                outcome["same_measurement_budget"] = {
                    "observations": len(values),
                    "cycles": config.cycles,
                    "history": config.history,
                }
                if weights is not None:
                    mean_weights = weights.mean(dim=1)[0]
                    outcome["final_weights"] = [float(v) for v in mean_weights.tolist()]
                    outcome["reference_final_weights"] = [
                        float(v) for v in reference_weights[0].tolist()
                    ]
                    outcome["abs_deviation_from_reference"] = float(
                        (mean_weights.to(torch.float64) - reference_weights[0]).abs().max().item()
                    )
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
                outcome = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        outcomes[name] = outcome
    ran = [name for name, outcome in outcomes.items() if outcome.get("status") == "ran"]
    return {
        "status": "ran" if ran else "available-but-not-runnable",
        "registry_entries": sorted(registry),
        "entries_ran": sorted(ran),
        "entries": outcomes,
        "budget": {
            "case": case.name,
            "observations": len(case.observations),
            "cycles": config.cycles,
            "history": config.history,
            "reference": "analytic.analytic_posterior on the same frozen measurement set",
        },
    }


# ----------------------------------------------------------------------------- readout


def _declared_action_matrix(geometry_dim: int, action_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(action_dim, geometry_dim, generator=generator, dtype=torch.float64)


def _controlled_two_candidate_representation(
    config: N02Config, weights: list[float]
) -> StructuredRepresentation:
    """A declared two-incompatible-candidate input for the readout probe.

    Candidate 0 and candidate 1 carry opposite declared geometry, so a readout
    that averages them produces an action that matches neither. This is a probe
    input, not a measurement; it exists to make the readout failure mode visible.
    """
    candidates = config.K
    geometry = torch.zeros(1, 1, candidates, config.measurement.geometry_dim)
    geometry[0, 0, 0, 0] = 1.0
    geometry[0, 0, 1, 0] = -1.0
    for index in range(2, candidates):
        geometry[0, 0, index, 0] = 0.05 * (index + 1)
    field = torch.zeros(1, 1, candidates, config.D, 2, 2)
    for index in range(candidates):
        field[0, 0, index] = float(index + 1) * 0.1
    weight_tensor = torch.tensor(weights).view(1, 1, candidates)
    energy = -torch.log(weight_tensor.clamp_min(1e-12))
    return StructuredRepresentation(
        h=field,
        weights=weight_tensor,
        measurement_energy=energy,
        prior_energy=torch.zeros_like(energy),
        relation_energy=torch.zeros_like(energy),
        geometry=geometry,
        valid=torch.ones(1, 1, dtype=torch.bool),
        support=torch.ones(1, 1, candidates, 2, 2),
        step_index=1,
        diagnostics={"kind": "controlled_two_candidate_probe"},
    )


def _probe_block(
    representation: StructuredRepresentation,
    readout: TokenReadout,
    action_fn: Any,
    *,
    token_budget: int,
) -> dict[str, Any]:
    block: dict[str, Any] = {"token_budget": token_budget}
    outputs: dict[str, Any] = {}
    for mode, function in (("map", map_hypothesis), ("tokens", hypothesis_tokens)):
        result = function(representation, token_budget, readout=readout)
        outputs[mode] = result
    for mode in ("map", "tokens", "mean"):
        probe = two_candidate_probe(representation, action_fn=action_fn, mode=mode)
        interpolation = probe.interpolation_fraction.mean(dim=1)
        block[mode] = {
            "mean_interpolation_fraction": float(interpolation.mean().item()),
            "max_interpolation_fraction": float(interpolation.max().item()),
            "wrong_action_fraction": float(
                (interpolation.abs() > INTERPOLATION_TOLERANCE).to(torch.float64).mean().item()
            ),
            "mean_action": [float(value) for value in probe.action.mean(dim=(0, 1)).tolist()],
        }
    grouped = outputs["tokens"]
    if grouped.mean_collapse is not None:
        collapse_probe = two_candidate_probe(representation, action_fn=action_fn, mode="mean")
        del collapse_probe
        selected = representation.map_geometry()
        weights = representation.weights
        mean_geometry = grouped.mean_collapse
        top = weights.topk(2, dim=-1)
        other = representation.geometry.gather(
            2,
            top.indices[..., 1]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, representation.geometry.shape[-1]),
        ).squeeze(2)
        difference = (other - selected).norm(dim=-1).clamp_min(1e-8)
        deviation = (mean_geometry - selected).norm(dim=-1)
        block["tokens_mean_collapse"] = {
            "mean_interpolation_fraction": float((deviation / difference).mean().item()),
            "note": (
                "what a backend gets if it averages the grouped tokens instead of "
                "keeping them apart; the declared grouped readout itself is the "
                "'tokens' row"
            ),
        }
    return block


def run_readout(
    config: N02Config, checkpoint: Checkpoint, *, seed: int, device: str
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = checkpoint.model if checkpoint.loaded else build_encoder(config).to(device)
    model.eval()
    if config.K < 2:
        return {
            "status": "unavailable",
            "reason": "the two-candidate probe needs K >= 2",
        }
    # Deliberately not an exact tie: ``torch.topk`` orders an exact tie as
    # (1, 0) while ``argmax`` returns 0, so the probe's "other candidate" would be
    # the selected one and its interpolation ratio would divide by the 1e-8 clamp.
    # Untied controlled weights so the "how far towards the discarded candidate did
    # this readout drift" ratio is well posed; the tied case is covered by tests.
    controlled_weights = [0.6, 0.4] + [0.0] * (config.K - 2)
    controlled = _controlled_two_candidate_representation(config, controlled_weights)
    readout = TokenReadout(config.D, config.K, token_dim=config.token_dim)
    matrix = _declared_action_matrix(config.measurement.geometry_dim, 4, seed)

    def action_fn(geometry: torch.Tensor) -> torch.Tensor:
        """Declared linear action probe: a fixed seeded matrix, not a robot label."""
        return geometry.to(matrix.dtype) @ matrix.T

    budget = min(config.tokens_per_camera, config.K * 32)
    budget = max(budget - budget % config.K, config.K)
    controlled_block = _probe_block(controlled, readout, action_fn, token_budget=budget)

    synthetic, synthetic_reason = _module("owac.frameworks.n02.synthetic")
    episode = None
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    camera: CameraSpec | None = None
    representation = None
    source_reason = ""
    if synthetic is not None and config.measurement.input_mode == "rawvla_rgb3":
        options = synthetic.SyntheticOptions(height=64, width=64, channels=3)
        generator = synthetic.SyntheticGenerator(options, seed=seed, split="validation")
        episode = generator.generate(
            prior_condition="misleading", num_frames=6, episode_index=3, seed=seed
        )
        camera = CameraSpec(
            camera_id="synthetic_measurement",
            height=64,
            width=64,
            channels=3,
            input_mode="rawvla_rgb3",
        )
    else:
        source_reason = synthetic_reason or (
            f"config declares input_mode={config.measurement.input_mode!r}, which does not "
            "match the synthetic generator's three-channel frames"
        )
        analytic, analytic_reason = _module("owac.frameworks.n02.analytic")
        if analytic is not None and config.measurement.input_mode == "bayer_cfa":
            case_config = analytic.make_config(K=2, cycles=config.cycles, history=config.history)
            case = analytic.delayed_evidence_case(case_config, delay=2, seed=seed)
            values, masks = analytic_frames(case, ANALYTIC_GRID_WIDTH)
            camera = CameraSpec(
                camera_id="analytic_measurement",
                height=1,
                width=ANALYTIC_GRID_WIDTH,
                channels=1,
                input_mode="bayer_cfa",
                cfa_pattern="RGGB",
            )
        else:
            source_reason = source_reason or analytic_reason
    compute = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    latencies: list[float] = []
    latent_block: dict[str, Any]
    if camera is None:
        latent_block = {
            "status": "skipped",
            "reason": (
                "no measurement source matches this config's declared input mode: " + source_reason
            ),
        }
    else:
        state = model.init_state(1, camera, device)
        with torch.no_grad():
            if episode is not None:
                step_values: list[torch.Tensor] = [
                    torch.from_numpy(episode.frames[step]).unsqueeze(0).to(torch.float32)
                    for step in range(episode.num_frames)
                ]
                step_masks: list[torch.Tensor | None] = [None] * len(step_values)
            else:
                step_values = values
                step_masks = list(masks)
            for index, value in enumerate(step_values):
                frame = MeasurementFrame(
                    values=value,
                    valid_mask=step_masks[index],
                    frame_ids=(index,),
                    episode_ids=(0,),
                    camera_id=camera.camera_id,
                )
                start = time.perf_counter()
                output = model.step(frame, EmbodimentContext(), state, compute)
                latencies.append((time.perf_counter() - start) * 1000.0)
                state = model.detach_state(output.state)
                representation = output.representation
    latent_probe = None
    if representation is not None:
        weights = representation.weights.mean(dim=1)[0]
        latent_block = {
            "status": "ran",
            "source": "synthetic generator" if episode is not None else "analytic fallback",
            "weights": [float(value) for value in weights.tolist()],
            "top_weight": float(weights.max().item()),
            "encoder_step_ms": _percentiles(latencies),
            "candidate_pair_separation": float(
                (representation.geometry[0, 0, 0] - representation.geometry[0, 0, 1]).norm().item()
            ),
        }
        if episode is None:
            latent_block["fallback_note"] = (
                "the analytic fallback carries K=2 candidates while the config declares "
                f"K={config.K}; the grouped-token budget is still split over K groups"
            )
        if representation.K >= 2:
            latent_probe = _probe_block(representation, readout, action_fn, token_budget=budget)
    readout_latencies: dict[str, Any] = {}
    with torch.no_grad():
        target = representation if representation is not None else controlled
        for mode, function in (("map", map_hypothesis), ("tokens", hypothesis_tokens)):
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                function(target, budget, readout=readout)
                samples.append((time.perf_counter() - start) * 1000.0)
            readout_latencies[mode] = _percentiles(samples)
    return {
        "status": "ran",
        "checkpoint": None if not checkpoint.loaded else checkpoint.sha256,
        "weights_are_trained": bool(checkpoint.loaded),
        "interpolation_tolerance": INTERPOLATION_TOLERANCE,
        "declared_action_matrix_seed": seed,
        "action_dim": 4,
        "readout_ms": readout_latencies,
        "token_budget": budget,
        "controlled_two_candidate_probe": controlled_block,
        "encoder_representation_probe": latent_probe,
        "encoder_representation": latent_block,
        "unavailable_sources": source_reason,
        "note": (
            "the controlled probe is a declared input, not a measurement; it makes the "
            "interpolation failure mode visible. The encoder probe reports the numbers "
            "of this checkpoint and is only as meaningful as that checkpoint is trained. "
            "With two incompatible candidates and weights (0.6, 0.4) the mean readout "
            "lands w_other = 0.4 of the way to the discarded candidate."
        ),
    }


# ----------------------------------------------------------------------------- reporting


def _report_markdown(results: dict[str, Any]) -> str:
    lines: list[str] = ["# N02 evaluation report", ""]
    lines.append(f"- suite(s): `{results['suite']}`")
    lines.append(f"- config: `{results['config_path'] or results['preset']}`")
    lines.append(f"- seed: {results['seed']}")
    lines.append(f"- git SHA: `{results['git_sha']}`")
    checkpoint = results["checkpoint"]
    lines.append(
        f"- checkpoint: `{checkpoint['path']}` (sha256 `{checkpoint['sha256']}`, "
        f"step {checkpoint['step']})"
        if checkpoint["path"]
        else "- checkpoint: none (reference-only run)"
    )
    lines.append("")
    mechanism = results.get("mechanism")
    if mechanism and mechanism.get("status") == "ran":
        lines.append("## Required trajectories")
        lines.append("")
        lines.append(
            "Per-observation weights of the explicit two-hypothesis reference "
            "(`owac.frameworks.n02.analytic`), which applies the same scoring rule as "
            "the learned encoder on a frozen measurement set."
        )
        lines.append("")
        for name, trajectory in mechanism["trajectories"].items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"{trajectory['description']}")
            lines.append("")
            lines.append("| obs | weight (true / other) | energy (true / other) | unique | top |")
            lines.append("| --- | --- | --- | --- | --- |")
            for step in trajectory["steps"]:
                weights = " / ".join(f"{value:.6f}" for value in step["weights"])
                energies = " / ".join(f"{value:.6f}" for value in step["measurement_energy"])
                contradictions = "".join(
                    "T" if flag else "." for flag in step["contradiction_flags"]
                )
                lines.append(
                    f"| {step['observation_index']} | {weights} | {energies} | "
                    f"{step['unique_evidence_count']} | {step['top_index']} "
                    f"(contradictions {contradictions}) |"
                )
            lines.append("")
            duration = trajectory["wrong_explanation_duration_steps"]
            area = trajectory["wrong_explanation_area"]
            offset = trajectory["wrong_location_offset_normalised"]
            lines.append(
                f"final weights {trajectory['final']['weights']}; "
                f"correct top: {trajectory['final_is_true']}; "
                f"wrong-explanation duration {duration} steps, area {area:.4f}; "
                f"wrong location offset {offset:.3f}"
            )
            lines.append("")
        lines.append("## Metrics")
        lines.append("")
        for key, value in mechanism["metrics"].items():
            if key in {"probability_metrics", "exact_static_case_max_abs_deviation"}:
                continue
            lines.append(f"- `{key}`: {json.dumps(value)}")
        lines.append("")
        lines.append("## Controls")
        lines.append("")
        controls = mechanism["controls"]
        lines.append(
            "- repeated observation: unique evidence "
            f"{controls['repeated_observation']['unique_evidence_single']} -> "
            f"{controls['repeated_observation']['unique_evidence_duplicated']}, "
            f"posterior identical: {controls['repeated_observation']['posterior_identical']}"
        )
        lines.append(
            "- exact static case: max abs deviation "
            f"{controls['exact_static_case']['max']:.3e} "
            f"(tolerance {controls['exact_static_case']['tolerance']:.0e}), "
            f"passed: {controls['exact_static_case']['passed']}"
        )
        fork = controls["fork_from_same_old_state"]
        lines.append(
            "- fork from the same old state: base "
            f"{[round(v, 6) for v in fork['base_posterior']]}, supporting "
            f"{[round(v, 6) for v in fork['supporting_posterior']]}, refuting "
            f"{[round(v, 6) for v in fork['refuting_posterior']]}, uninformative "
            f"{[round(v, 6) for v in fork['uninformative_posterior']]}"
        )
        lines.append(
            "- no drift from cycles: "
            f"{fork['no_drift_from_cycles']} (uninformative delta "
            f"{fork['uninformative_posterior_delta']:.3e}, cycle-count delta "
            f"{max(fork['one_cycle_posterior_delta'], fork['double_cycles_posterior_delta']):.3e})"
        )
        learned = mechanism.get("learned_encoder_trace", {})
        lines.append("")
        lines.append("## Learned encoder trace")
        lines.append("")
        if learned.get("status") == "ran":
            lines.append(
                f"- final is true: {learned['final_is_true']}; correction observation "
                f"{learned['correction_observation_index']}; internal iterations at first "
                f"correct {learned['internal_iterations_at_first_correct']}; state bytes "
                f"{learned['state_bytes']}"
            )
            lines.append(
                f"- candidate collapse fraction {learned['candidate_collapse_fraction']:.4f}"
            )
        else:
            lines.append(f"- skipped: {learned.get('reason')}")
        lines.append("")
    baselines = results.get("baselines")
    if baselines is not None:
        lines.append("## Baselines")
        lines.append("")
        lines.append(f"- status: {baselines['status']}")
        if baselines.get("reason"):
            lines.append(f"- reason: {baselines['reason']}")
        if baselines.get("registry_entries"):
            lines.append(f"- registry entries: {', '.join(baselines['registry_entries'])}")
            lines.append(
                f"- entries that ran: {', '.join(baselines.get('entries_ran', [])) or 'none'}"
            )
        lines.append("")
    readout = results.get("readout")
    if readout is not None and readout.get("status") == "ran":
        lines.append("## Readout probe (two incompatible candidates)")
        lines.append("")
        lines.append("| readout | mean interpolation | wrong-action fraction |")
        lines.append("| --- | --- | --- |")
        controlled = readout["controlled_two_candidate_probe"]
        for mode in ("map", "tokens", "mean"):
            block = controlled[mode]
            lines.append(
                f"| {mode} | {block['mean_interpolation_fraction']:.4f} | "
                f"{block['wrong_action_fraction']:.4f} |"
            )
        if "tokens_mean_collapse" in controlled:
            lines.append(
                f"| tokens (naive mean fusion) | "
                f"{controlled['tokens_mean_collapse']['mean_interpolation_fraction']:.4f} | - |"
            )
        lines.append("")
        encoder_probe = readout.get("encoder_representation_probe")
        if encoder_probe:
            lines.append("Encoder representation, same probe:")
            lines.append("")
            for mode in ("map", "tokens", "mean"):
                block = encoder_probe[mode]
                lines.append(
                    f"- `{mode}`: mean interpolation "
                    f"{block['mean_interpolation_fraction']:.4f}, wrong-action fraction "
                    f"{block['wrong_action_fraction']:.4f}"
                )
            lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for limitation in results.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    output: Path, results: dict[str, Any], trajectories: dict[str, Any], report: str
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "trajectories.json").write_text(json.dumps(trajectories, indent=2) + "\n")
    (output / "report.md").write_text(report)


# ----------------------------------------------------------------------------- entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the N02 measurement-constrained recurrent completion field."
    )
    parser.add_argument("--config", type=Path, default=None, help="N02 JSON config")
    parser.add_argument(
        "--preset", choices=("tiny", "small", "medium"), default=None, help="named scale preset"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--suite",
        choices=("mechanism", "baselines", "readout", "all"),
        default="mechanism",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/n02-eval"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(args.threads)))
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    if checkpoint.config is not None:
        config = checkpoint.config
    else:
        config = load_config(args.config, args.preset)
    suites = {"mechanism", "baselines", "readout"} if args.suite == "all" else {args.suite}
    results: dict[str, Any] = {
        "suite": args.suite,
        "config": config.as_dict(),
        "config_path": None if args.config is None else str(Path(args.config)),
        "preset": args.preset,
        "seed": int(args.seed),
        "git_sha": _git_sha(),
        "device": args.device,
        "torch_threads": int(torch.get_num_threads()),
        "torch_version": torch.__version__,
        "checkpoint": {
            "path": None if checkpoint.path is None else str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "step": checkpoint.step,
            "trained_config_source": checkpoint.config is not None,
        },
        "wall_seconds": None,
        "limitations": [],
    }
    started = time.perf_counter()
    if "mechanism" in suites:
        mechanism = run_mechanism(config, checkpoint, seed=int(args.seed), device=args.device)
        results["mechanism"] = mechanism
        if mechanism.get("status") == "ran":
            if mechanism["learned_encoder_trace"].get("status") != "ran":
                results["limitations"].append(
                    "the learned encoder was not run over the required trajectories: "
                    + str(mechanism["learned_encoder_trace"].get("reason"))
                )
    if "baselines" in suites:
        results["baselines"] = run_baselines(config, seed=int(args.seed), device=args.device)
    if "readout" in suites:
        results["readout"] = run_readout(
            config, checkpoint, seed=int(args.seed), device=args.device
        )
    results["limitations"].append(
        "the readout probe compares the selected candidate against the best *different* "
        "candidate; when two candidates are geometrically identical the interpolation is "
        "undefined and is reported as 0 instead of dividing by a clamped constant. The "
        "controlled input below uses untied weights (0.6, 0.4) so the comparison is "
        "well posed, and the exactly-tied case is exercised by the unit tests."
    )
    results["wall_seconds"] = time.perf_counter() - started
    results["peak_rss_bytes"] = peak_rss_bytes()
    if torch.cuda.is_available():  # pragma: no cover - CPU-only prototype host
        results["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
    else:
        results["peak_gpu_bytes"] = {
            "value": None,
            "reason": "no CUDA device is available on this host; nothing was measured",
        }
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.config is None and args.preset is None and args.checkpoint is None:
        args.preset = "tiny"
    results = evaluate(args)
    output = Path(args.output)
    trajectories = results.pop("_trajectories", None)
    if trajectories is None:
        trajectories = {}
        mechanism = results.get("mechanism")
        if isinstance(mechanism, dict):
            trajectories = mechanism.get("trajectories", {})
    write_outputs(output, results, trajectories, _report_markdown(results))
    print(
        json.dumps(
            {
                "output": str(output),
                "suite": results["suite"],
                "wall_seconds": results["wall_seconds"],
                "mechanism_status": results.get("mechanism", {}).get("status"),
                "baselines_status": results.get("baselines", {}).get("status"),
                "readout_status": results.get("readout", {}).get("status"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

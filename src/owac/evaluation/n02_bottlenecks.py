"""Fixed-checkpoint diagnostics; hidden physical templates are evaluator-only.

The oracle-filter arm deliberately has extra information and is never a learned
baseline. Core parameters/configuration are restored after every intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from owac.core.encoder_contract_n02 import ComputeBudget, MeasurementFrame
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.measurement import gaussian_nll
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import SyntheticGenerator, SyntheticOptions, values_for_task
from owac.training.n02 import episode_prior_context

ARMS = ("full", "no_propagation", "no_withdrawal", "neither", "oracle_filter")
CONDITIONS = ("correct", "misleading", "missing", "unidentifiable", "stale")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def paired_episodes(seed):
    options = SyntheticOptions(height=64, width=64)
    gen = SyntheticGenerator(options, seed=seed, split="validation")
    observed = gen.generate(seed=seed, prior_condition="correct", identifiable=True)
    observed = replace(observed, prior=replace(observed.prior, log_odds=math.log(19)))
    stale = SyntheticGenerator(replace(options, staleness=1.0), seed=seed, split="validation")
    changed = stale.generate(seed=seed, prior_condition="stale", identifiable=True)
    return gen, {
        "correct": observed,
        "misleading": replace(
            observed,
            prior=replace(observed.prior, condition="misleading", log_odds=-math.log(19)),
        ),
        "missing": replace(observed, prior=replace(observed.prior, condition="missing")),
        "unidentifiable": gen.generate(
            seed=seed, prior_condition="uninformative", identifiable=False
        ),
        "stale": replace(changed, prior=replace(changed.prior, log_odds=math.log(19))),
    }


def pool(value, size=16):
    return F.adaptive_avg_pool2d(torch.as_tensor(value, dtype=torch.float64), (size, size))


def physical_frame(gen, episode, index):
    templates = pool(gen.measurement_templates(episode, index))
    observation = pool(episode.frames[index][None])
    sigma = gen.options.noise_sigma / 4
    residual = ((observation - templates) / sigma).abs().mean(1)
    # Conservative semantic reference: known wrong template, observable
    # separation >= log(19), and a good alternative within the noise floor.
    separation = ((templates[0] - templates[1]) / sigma).square().sum(0) / 2
    truth = values_for_task(episode.task).index(episode.value_over_time[index])
    mask = torch.zeros(4, 16, 16, dtype=torch.bool)
    justified = (
        (separation >= math.log(19))
        & (residual[truth] <= 2.5)
        & (residual[1 - truth] > torch.maximum(1.5 * residual[truth], residual.new_tensor(2.5)))
    )
    mask[torch.arange(4) % 2 != truth] = justified
    return observation, templates, mask[None, None], truth, separation


def score_templates(observation, templates, variance, *, robust, reduction):
    """Independent formula; no learned head, common log-variance cancels."""
    z = (observation - templates) / math.sqrt(variance)
    rho = 0.5 * z.square()
    if robust == "huber":
        rho = torch.where(z.abs() <= 3, rho, 3 * (z.abs() - 1.5))
    return rho.sum((1, 2, 3)) if reduction == "sum" else rho.mean((1, 2, 3))


def calibration(gen, episode, model):
    rows = []
    for grid in (64, 16):
        for robust in ("gaussian", "huber"):
            for reduction in ("sum", "mean"):
                for withdrawal in (False, True) if grid == 16 else (False,):
                    initial = 0.0 if not episode.identifiable else -math.log(19)
                    logodds, evidence, penalty, masses, expected = initial, 0.0, 0.0, [], []
                    for index, values in enumerate(episode.frames):
                        templates = pool(gen.measurement_templates(episode, index), grid)
                        observed = pool(values[None], grid)
                        variance = gen.options.noise_sigma**2 / (64 // grid) ** 2
                        costs = score_templates(
                            observed, templates, variance, robust=robust, reduction=reduction
                        )
                        truth = values_for_task(episode.task).index(episode.value_over_time[index])
                        delta = float(costs[1 - truth] - costs[truth])
                        extra = 0.0
                        if withdrawal and index > 0:
                            residual = ((observed - templates) / math.sqrt(variance)).abs().mean(1)
                            score = residual[torch.arange(4) % 2][None, None]
                            energies = model.reviser.withdrawal_energy(
                                score, torch.ones_like(score), temperature=1.0
                            ).flatten()
                            extra = float(energies[1 - truth] - energies[truth])
                        expected.append(
                            float((templates[0] - templates[1]).square().sum() / (2 * variance))
                        )
                        evidence += delta
                        penalty += extra
                        logodds += delta + extra
                        masses.append(
                            float(torch.sigmoid(torch.tensor(logodds, dtype=torch.float64)))
                        )
                    rows.append(
                        {
                            "mode": f"grid{grid}_{robust}_{reduction}"
                            + ("_withdrawal" if withdrawal else ""),
                            "truth_mass": masses,
                            "final_ge_point7": masses[-1] >= 0.7,
                            "max_uniform_drift": max(abs(p - 0.5) for p in masses),
                            "summed_log_evidence": evidence,
                            "summed_withdrawal_difference": penalty,
                            "expected_gaussian_logbf": sum(expected),
                        }
                    )
    return rows


def class_margin(energy, truth):
    slots = torch.arange(energy.numel()) % 2
    return float(
        torch.logsumexp(-energy[slots == truth], 0) - torch.logsumexp(-energy[slots != truth], 0)
    )


def prediction_energy(model, field, observed):
    mean, _, geometry = model.measurement.predict(field)
    mean = mean[0, 0].double()
    variance = observed.new_tensor(0.01**2 / 16)
    energy = gaussian_nll(observed - mean, variance.log(), robust="huber", huber_delta=3).mean(
        (1, 2, 3)
    )
    return energy, mean, geometry[0, 0, :, 0]


def behavior_gate(condition, steps, change):
    if condition == "unidentifiable":
        return all(abs(r["semantic_mass"][0] - 0.5) <= 0.15 for r in steps)
    if condition == "misleading":
        prefix = [r for r in steps if not r["cue_visible"]]
        return bool(prefix) and prefix[-1]["truth_mass"] <= 0.25 and steps[-1]["truth_mass"] >= 0.7
    if condition == "missing":
        return steps[-1]["truth_mass"] > 0.5
    if condition == "stale":
        return steps[max(0, change - 1)]["truth_mass"] >= 0.7 and steps[-1]["truth_mass"] >= 0.7
    return steps[-1]["truth_mass"] >= 0.7


def run_episode(model, gen, episode, *, condition, arm, forks=False):
    """Rule interventions share checkpoint; forks also share the input state."""
    original_cfg = model.reviser.revision
    cfg = replace(original_cfg)
    cfg.propagation_enabled = arm not in {"no_propagation", "neither"}
    if arm in {"no_withdrawal", "neither"}:
        cfg.downweight_contradicted = 1.0
    original_map, original_step = model.reviser.contradiction_map, model.reviser.step
    model.reviser.revision = cfg
    live = {}
    traces, fork_rows, steps, candidate_rows = [], [], [], []

    def filtered(score, **kwargs):
        return original_map(score, **kwargs) & live["mask"].to(score.device)

    def captured(state, residual, **kwargs):
        old_h = state.h.clone() if forks and kwargs["iteration"] == 0 else None
        result = original_step(state, residual, **kwargs)
        direct, propagated = result.contradiction, result.propagated
        semantic_refutation = live["mask"]
        traces.append(
            {
                "frame": live["index"],
                "cycle": kwargs["iteration"],
                "direct_fraction": float(direct.float().mean()),
                "propagated_fraction": float(propagated.float().mean()),
                "direct_count": int(direct.sum()),
                "oracle_supported_direct_count": int((direct & semantic_refutation).sum()),
                "oracle_refutation_count": int(semantic_refutation.sum()),
                "all_candidates_bad_fraction": float(
                    model.reviser.all_candidates_bad(residual).float().mean()
                ),
            }
        )
        if forks and kwargs["iteration"] == 0 and live["index"] > 0:
            assert torch.equal(old_h, state.h), "revision mutated the fork input"
            cfg.propagation_enabled = False
            try:
                alternate = original_step(state, residual, **kwargs)
            finally:
                cfg.propagation_enabled = True
            assert torch.equal(old_h, state.h), "alternate mutated the fork input"
            on_energy, _, _ = prediction_energy(model, result.h, live["observed"])
            off_energy, _, _ = prediction_energy(model, alternate.h, live["observed"])
            truth = live["truth"]
            fork_rows.append(
                {
                    "frame": live["index"],
                    "cue_visible": live["cue_visible"],
                    "extra_selected_fraction": float(
                        (result.propagated & ~alternate.propagated).float().mean()
                    ),
                    "on_minus_off_mean_nll": float((on_energy - off_energy).mean()),
                    "on_minus_off_true_class_margin": class_margin(on_energy, truth)
                    - class_margin(off_energy, truth),
                    "on_minus_off_abs_class_margin": abs(class_margin(on_energy, truth))
                    - abs(class_margin(off_energy, truth)),
                    "field_mean_abs_change": float((result.h - alternate.h).abs().mean()),
                }
            )
        return result

    model.reviser.step = captured
    if arm == "oracle_filter":
        model.reviser.contradiction_map = filtered
    context = episode_prior_context([episode], model.config.K)
    quality = gen.measurement_quality(episode)
    state = None
    try:
        for index, values in enumerate(episode.frames):
            observed, templates, mask, truth, separation = physical_frame(gen, episode, index)
            live.update(
                index=index,
                observed=observed,
                mask=mask,
                truth=truth,
                cue_visible=quality["identifiable_now"][index],
            )
            frame = MeasurementFrame(
                values=torch.from_numpy(values)[None],
                frame_ids=(index,),
                episode_ids=(episode.episode_id,),
                metadata={"variance": gen.options.noise_sigma**2},
                metadata_available={"variance": True},
                camera_id="synthetic_measurement",
            )
            output = model.step(frame, context, state, ComputeBudget(4, 4))
            state = model.detach_state(output.state)
            rep = output.representation
            weights = rep.weights[0, 0]
            semantic = [float(weights[::2].sum()), float(weights[1::2].sum())]
            steps.append(
                {
                    "semantic_mass": semantic,
                    "truth_mass": semantic[truth],
                    "truth": truth,
                    "weights": weights.tolist(),
                    "measurement_energy": rep.measurement_energy.flatten().tolist(),
                    "withdrawal": rep.revision_energy.flatten().tolist(),
                    "geometry": rep.geometry[0, 0, :, 0].tolist(),
                    "cue_visible": quality["identifiable_now"][index],
                }
            )
            if arm == "full":
                _, mean, geometry = prediction_energy(model, rep.h, observed)
                roi = separation >= math.log(19)
                distances = (mean[:, None] - templates[None]).square()
                own = torch.arange(4) % 2
                geometry_target = own * 2 - 1
                metric = {
                    "frame": index,
                    "roi_cells": int(roi.sum()),
                    "geometry_sign_correct_fraction": float(
                        ((geometry > 0) == own.bool()).float().mean()
                    ),
                    "geometry_mse": float((geometry - geometry_target).square().mean()),
                    "mean_prediction_gap_between_classes": float(
                        (mean[::2].mean(0) - mean[1::2].mean(0)).abs().mean()
                    ),
                    "physical_template_gap": float((templates[0] - templates[1]).abs().mean()),
                }
                if roi.any():
                    mse = distances[..., roi].mean((2, 3))
                    metric["slot_template_assignment_correct"] = (
                        mse[torch.arange(4), own] < mse[torch.arange(4), 1 - own]
                    ).tolist()
                    contrast = templates[0] - templates[1]
                    predicted = mean[::2].mean(0) - mean[1::2].mean(0)
                    metric["semantic_contrast_cosine"] = float(
                        F.cosine_similarity(predicted.flatten(), contrast.flatten(), dim=0)
                    )
                if index == episode.num_frames - 1:
                    # Intervene on the geometry output only, preserving shared features.
                    hook = model.measurement.head.geometry.register_forward_hook(
                        lambda _m, _i, out: -out
                    )
                    try:
                        swapped, _, swapped_geometry = model.measurement.predict(rep.h)
                    finally:
                        hook.remove()
                    metric["geometry_flip_raw_max_delta"] = float(
                        (swapped[0, 0].double() - mean).abs().max()
                    )
                    metric["geometry_flip_geometry_mean_delta"] = float(
                        (swapped_geometry[0, 0, :, 0] - geometry).abs().mean()
                    )
                candidate_rows.append(metric)
    finally:
        model.reviser.revision = original_cfg
        model.reviser.contradiction_map, model.reviser.step = original_map, original_step
    return {
        "condition": condition,
        "arm": arm,
        "measurement_sha256": hashlib.sha256(episode.frames.tobytes()).hexdigest(),
        "prior": context.hypothesis_prior.flatten().tolist(),
        "gate": behavior_gate(condition, steps, episode.geometry_change_step),
        "steps": steps,
        "traces": traces,
        "forks": fork_rows,
        "candidates": candidate_rows,
    }


def replay_scores(record, scale):
    """Frozen predictions, dynamic posterior only; does not rerun the network."""
    probability = torch.tensor(record["prior"], dtype=torch.float32)
    rows = []
    for step in record["steps"]:
        logits = (
            probability.clamp_min(1e-12).log()
            - scale * torch.tensor(step["measurement_energy"])
            - torch.tensor(step["withdrawal"])
        )
        probability = torch.softmax(logits, 0)
        rows.append(probability.tolist())
    return rows


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(20261013)
    means = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "sequence_count": len(values),
    }


def summarize(records, calibration_rows, pairs):
    report = {"B1": {}, "B2": {}, "B3": {}}
    for condition in ("misleading", "unidentifiable"):
        report["B1"][condition] = {}
        for mode in sorted({r["mode"] for r in calibration_rows}):
            rows = [
                r for r in calibration_rows if r["condition"] == condition and r["mode"] == mode
            ]
            report["B1"][condition][mode] = {
                "final_ge_point7": sum(r["final_ge_point7"] for r in rows),
                "total": len(rows),
                "mean_final_truth_mass": float(np.mean([r["truth_mass"][-1] for r in rows])),
                "max_uniform_drift": max(r["max_uniform_drift"] for r in rows),
                "mean_log_evidence": float(np.mean([r["summed_log_evidence"] for r in rows])),
                "mean_expected_gaussian_logbf": float(
                    np.mean([r["expected_gaussian_logbf"] for r in rows])
                ),
            }
    full = [r for r in records if r["arm"] == "full"]
    no_cue = [r for r in full if r["condition"] == "unidentifiable"]
    all_no_cue_steps = [s for r in no_cue for s in r["steps"]]
    candidates = [
        c for r in full if r["condition"] == "missing" for c in r["candidates"] if c["roi_cells"]
    ]
    report["B2"] = {
        "no_cue_energy_by_slot": np.mean(
            [s["measurement_energy"] for s in all_no_cue_steps], 0
        ).tolist(),
        "no_cue_geometry_sign_correct": float(
            np.mean([c["geometry_sign_correct_fraction"] for r in no_cue for c in r["candidates"]])
        ),
        "no_cue_raw_prediction_class_gap": float(
            np.mean(
                [c["mean_prediction_gap_between_classes"] for r in no_cue for c in r["candidates"]]
            )
        ),
        "no_cue_physical_template_gap_max": max(
            c["physical_template_gap"] for r in no_cue for c in r["candidates"]
        ),
        "informative_missing_slot_template_assignment_correct": np.mean(
            [c["slot_template_assignment_correct"] for c in candidates], 0
        ).tolist(),
        "informative_missing_semantic_contrast_cosine": float(
            np.mean([c["semantic_contrast_cosine"] for c in candidates])
        ),
        "geometry_flip_raw_max_delta": max(
            c.get("geometry_flip_raw_max_delta", 0) for r in full for c in r["candidates"]
        ),
        "counterfactual_both_correct": sum(p["both_correct"] for p in pairs),
        "counterfactual_total": len(pairs),
        "counterfactual_p1_change": bootstrap([p["p1_when_1"] - p["p1_when_0"] for p in pairs]),
        "counterfactual_records": pairs,
        "frozen_prediction_rescoring": {},
    }
    for scale in (1, 16, 768):
        modes = {}
        for condition in CONDITIONS:
            rows = [r for r in full if r["condition"] == condition]
            passed, drift, errors = 0, [], []
            for row in rows:
                predicted = replay_scores(row, scale)
                if scale == 1:
                    errors.append(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(predicted)
                                    - np.asarray([s["weights"] for s in row["steps"]])
                                )
                            )
                        )
                    )
                final = predicted[-1]
                truth = row["steps"][-1]["truth"]
                passed += sum(final[truth::2]) >= 0.7
                drift.append(float(np.mean([abs(sum(w[::2]) - 0.5) for w in predicted])))
            modes[condition] = {
                "final_ge_point7": passed,
                "total": len(rows),
                "mean_uniform_drift": float(np.mean(drift)),
            }
            if errors:
                modes[condition]["replay_max_abs_error"] = max(errors)
                assert max(errors) < 1e-5, "scale=1 replay does not reproduce actual weights"
        report["B2"]["frozen_prediction_rescoring"][str(scale)] = modes
    for arm in ARMS:
        report["B3"][arm] = {}
        for condition in CONDITIONS:
            rows = [r for r in records if r["arm"] == arm and r["condition"] == condition]
            reference = {r["seed"]: r for r in full if r["condition"] == condition}
            traces = [t for r in rows for t in r["traces"]]
            direct = sum(t["direct_count"] for t in traces)
            report["B3"][arm][condition] = {
                "gates_passed": sum(r["gate"] for r in rows),
                "total": len(rows),
                "mean_final_truth_mass": float(
                    np.mean([r["steps"][-1]["truth_mass"] for r in rows])
                ),
                "mean_uniform_drift": float(
                    np.mean([abs(s["semantic_mass"][0] - 0.5) for r in rows for s in r["steps"]])
                ),
                "paired_final_truth_mass_minus_full": bootstrap(
                    [
                        r["steps"][-1]["truth_mass"]
                        - reference[r["seed"]]["steps"][-1]["truth_mass"]
                        for r in rows
                    ]
                ),
                "paired_uniform_drift_minus_full": bootstrap(
                    [
                        np.mean([abs(s["semantic_mass"][0] - 0.5) for s in r["steps"]])
                        - np.mean(
                            [
                                abs(s["semantic_mass"][0] - 0.5)
                                for s in reference[r["seed"]]["steps"]
                            ]
                        )
                        for r in rows
                    ]
                ),
                "direct_fraction": float(np.mean([t["direct_fraction"] for t in traces])),
                "propagated_fraction": float(np.mean([t["propagated_fraction"] for t in traces])),
                "oracle_supported_direct_precision": sum(
                    t["oracle_supported_direct_count"] for t in traces
                )
                / max(direct, 1),
                "all_candidates_bad_fraction": float(
                    np.mean([t["all_candidates_bad_fraction"] for t in traces])
                ),
            }
    report["B3"]["same_state_forks"] = {}
    for condition in CONDITIONS:
        rows = [r for r in full if r["condition"] == condition]
        report["B3"]["same_state_forks"][condition] = {
            key: bootstrap([np.mean([f[key] for f in r["forks"]]) for r in rows])
            for key in (
                "extra_selected_fraction",
                "on_minus_off_mean_nll",
                "on_minus_off_true_class_margin",
                "on_minus_off_abs_class_margin",
                "field_mean_abs_change",
            )
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/n02-repair-20261013/repaired/checkpoint.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/n02-bottlenecks-20261013"))
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=32)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = N02Encoder(N02Config.from_dict(payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    assert model.config.K == 4 and model.config.beta_relation == 0
    assert model.config.temperature == 1 and model.config.beta_prior == 1
    parameters_before = tensor_digest(model)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    source_paths = [
        Path(__file__),
        *Path("src/owac/frameworks/n02").glob("*.py"),
        Path("src/owac/training/n02.py"),
    ]
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_step": payload["step"],
        "source_hashes": {str(p): digest(p) for p in source_paths},
        "config": model.config.as_dict(),
        "seeds": seeds,
        "arms": ARMS,
        "training_steps": 0,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": 1,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records, calibration_rows, pairs = [], [], []
    with torch.no_grad(), (args.output / "records.jsonl").open("w") as stream:
        for seed in seeds:
            gen, episodes = paired_episodes(seed)
            for condition in ("misleading", "unidentifiable"):
                for row in calibration(gen, episodes[condition], model):
                    calibration_rows.append({"seed": seed, "condition": condition, **row})
            for condition, episode in episodes.items():
                hashes = set()
                for arm in ARMS:
                    row = run_episode(
                        model, gen, episode, condition=condition, arm=arm, forks=arm == "full"
                    )
                    row["seed"] = seed
                    hashes.add(row["measurement_sha256"])
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
                assert len(hashes) == 1
            # Generate paired sensor inputs from a common physical episode,
            # including the same stored noise, transmission and cue schedule.
            original = episodes["missing"]
            all_templates = np.stack([gen.measurement_templates(original, i) for i in range(8)])
            pair = {"seed": seed}
            for truth, value in enumerate(values_for_task(original.task)):
                changed = replace(
                    original,
                    frames=(all_templates[:, truth] + original.noise).astype(np.float32),
                    value=value,
                    value_over_time=(value,) * 8,
                )
                row = run_episode(model, gen, changed, condition="missing", arm="full")
                row["seed"], row["counterfactual_truth"] = seed, truth
                stream.write(json.dumps(row) + "\n")
                pair[f"p1_when_{truth}"] = row["steps"][-1]["semantic_mass"][1]
                pair[f"raw_sha256_{truth}"] = row["measurement_sha256"]
            pair["both_correct"] = pair["p1_when_0"] < 0.5 and pair["p1_when_1"] > 0.5
            pairs.append(pair)
            stream.flush()
            print(
                json.dumps(
                    {
                        "seed_completed": seed,
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
    assert tensor_digest(model) == parameters_before, "diagnostics changed trained parameters"
    assert model.config.as_dict() == manifest["config"], "diagnostics changed core configuration"
    summary = summarize(records, calibration_rows, pairs)
    summary["scope"] = (
        "One fixed 200-step checkpoint; synthetic opening direction; "
        "oracle arms are diagnostics only"
    )
    summary["resources"] = {
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_steps": 0,
    }
    summary["integrity"] = {
        "weights_unchanged": True,
        "config_unchanged": True,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "unique_seeds": len(seeds),
        "encoder_frame_calls": len(seeds) * (len(ARMS) * len(CONDITIONS) + 2) * 8,
    }
    (args.output / "calibration.json").write_text(json.dumps(calibration_rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"completed": True, "output": str(args.output), "resources": summary["resources"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Predeclared tiny training ladder; decisions use behavior, never loss decrease.

Counterfactual templates are evaluation references only here. They do not enter
the encoder. The final seed holdout uses the existing validation geometry set;
the generator has no separate test geometry split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.evaluation.n02_acceptance import evaluate
from owac.evaluation.n02_bottlenecks import paired_episodes
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import values_for_task
from owac.training.n02 import episode_prior_context, parse_args, train


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def extension_decision(development, *, contracts_passed, finite):
    """Frozen before training in the accompanying experiment protocol."""
    qualified, progress = [], {}
    for seed in (0, 1, 2):
        earlier, later = (development[str(seed)][str(step)] for step in (100, 200))
        gates = later["gates"]
        qualified.append(
            all(
                gates[name]["passed"] >= threshold
                for name, threshold in (
                    ("correct", 12),
                    ("misleading", 4),
                    ("unidentifiable", 12),
                    ("stale", 4),
                )
            )
            and later["counterfactual"]["both_correct"] >= 8
        )
        for step, result in ((100, earlier), (200, later)):
            progress.setdefault(str(step), []).append(
                sum(
                    result["gates"][name]["passed"]
                    for name in ("misleading", "unidentifiable", "stale")
                )
            )
    improving = sum(progress["200"]) >= sum(progress["100"])
    return {
        "extend_to_800": bool(sum(qualified) >= 2 and improving and contracts_passed and finite),
        "seed_qualified": dict(zip(map(str, (0, 1, 2)), qualified, strict=True)),
        "behavior_count_sum": {step: sum(counts) for step, counts in progress.items()},
        "nondecreasing": improving,
        "contracts_passed": bool(contracts_passed),
        "finite": bool(finite),
    }


@torch.no_grad()
def pair_probe(checkpoint, seeds):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = N02Config.from_dict(payload["config"])
    model = N02Encoder(config).eval()
    model.load_state_dict(payload["model"])
    rows, details = [], []
    for seed in seeds:
        generator, episodes = paired_episodes(seed)
        base = episodes["missing"]
        templates = np.stack([generator.measurement_templates(base, i) for i in range(8)])
        row = {"seed": seed}
        for truth, value in enumerate(values_for_task(base.task)):
            episode = replace(
                base,
                frames=(templates[:, truth] + base.noise).astype(np.float32),
                value=value,
                value_over_time=(value,) * 8,
            )
            context = episode_prior_context([episode], config.K)
            quality = generator.measurement_quality(episode)
            state = None
            for index, raw in enumerate(episode.frames):
                frame = MeasurementFrame(
                    values=torch.from_numpy(raw).unsqueeze(0),
                    frame_ids=(index,),
                    episode_ids=(episode.episode_id,),
                    metadata={"variance": generator.options.noise_sigma**2},
                    metadata_available={"variance": True},
                    camera_id="synthetic_measurement",
                )
                output = model.step(
                    frame, context, state, ComputeBudget(config.cycles, config.history)
                )
                state = model.detach_state(output.state)
                rep = output.representation
                mean, logvar, geo = model.measurement.predict(rep.h)
                target = F.adaptive_avg_pool2d(torch.from_numpy(templates[index]), mean.shape[-2:])
                expected = target[torch.arange(config.K) % 2]
                sensor_var = state.diagnostics["_buffer"].frames[-1].variance
                total_var = logvar.exp()[0, 0] + sensor_var[0]
                error = mean[0, 0] - expected
                counterfactual_error = mean[0, 0] - target[1 - torch.arange(config.K) % 2]
                details.append(
                    {
                        "seed": seed,
                        "truth": truth,
                        "frame": index,
                        "cue_visible": bool(quality["identifiable_now"][index]),
                        "p1": float(rep.weights[0, 0, 1::2].sum()),
                        "geometry": geo[0, 0, :, 0].tolist(),
                        "geometry_sign_accuracy": float(
                            ((geo[0, 0, :, 0] > 0) == (torch.arange(config.K) % 2 == 1))
                            .float()
                            .mean()
                        ),
                        "physical_template_rmse": float(error.square().mean().sqrt()),
                        "own_template_closer_fraction": float(
                            (
                                error.square().mean((1, 2, 3))
                                < counterfactual_error.square().mean((1, 2, 3))
                            )
                            .float()
                            .mean()
                        ),
                        "model_std": float(logvar.exp().mean().sqrt()),
                        "standardized_template_mse": float((error.square() / total_var).mean()),
                        "prediction_class_rmse": float(
                            (mean[0, 0, ::2].mean(0) - mean[0, 0, 1::2].mean(0))
                            .square()
                            .mean()
                            .sqrt()
                        ),
                        "semantic_refutation_fraction": float(
                            state.revision.contradiction.float().mean()
                        ),
                        "last_cycle_updated_fraction": output.cost.updated_fraction,
                        "last_cycle_reset_fraction": float(
                            state.revision.last_reset.float().mean()
                        ),
                        "last_cycle_propagated_fraction": float(
                            state.revision.propagated.float().mean()
                        ),
                    }
                )
            row[f"p1_when_{truth}"] = float(rep.weights[0, 0, 1::2].sum())
            row[f"measurement_sha256_{truth}"] = hashlib.sha256(
                episode.frames.tobytes()
            ).hexdigest()
        row["both_correct"] = row["p1_when_0"] < 0.5 and row["p1_when_1"] > 0.5
        rows.append(row)
    grouped = {}
    for visible in (False, True):
        selected = [row for row in details if row["cue_visible"] == visible]
        grouped["cue" if visible else "no_cue"] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "geometry_sign_accuracy",
                "physical_template_rmse",
                "own_template_closer_fraction",
                "model_std",
                "standardized_template_mse",
                "prediction_class_rmse",
                "semantic_refutation_fraction",
            )
        }
        grouped["cue" if visible else "no_cue"]["mean_abs_p1_drift"] = float(
            np.mean([abs(row["p1"] - 0.5) for row in selected])
        )
    return {
        "both_correct": sum(row["both_correct"] for row in rows),
        "total": len(rows),
        "mean_p1_change": float(np.mean([row["p1_when_1"] - row["p1_when_0"] for row in rows])),
        "diagnostics": grouped,
        "diagnostic_scope": "prediction after solving and semantic map from last revision cycle",
        "records": rows,
        "steps": details,
    }


def assessment(checkpoint, seeds, output):
    report = evaluate(checkpoint, seeds=seeds)
    report["counterfactual"] = pair_probe(checkpoint, seeds)
    write(output, report)
    report.pop("records")
    report["counterfactual"].pop("records")
    report["counterfactual"].pop("steps")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = json.loads(args.checks.read_text())
    if not checks["passed"]:
        raise ValueError("contracts must pass before the training ladder")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    source_paths = sorted(
        set(root.glob("src/owac/**/*.py")) | set(root.glob("configs/encoders/n02/*.json"))
    )
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in source_paths},
        "protocol_sha256": digest(
            root / "research/experiments/2026-10-13_n02_contracts_training.md"
        ),
        "checks_sha256": digest(args.checks),
        "development_seeds": list(range(400, 416)),
        "final_seeds": list(range(600, 632)),
        "split": "validation geometry for dev and final; disjoint seeds, no third geometry set",
        "training_seeds": [0, 1, 2],
    }
    write(args.output / "manifest.json", manifest)
    (args.output / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    started = time.perf_counter()
    development = {str(seed): {} for seed in (0, 1, 2)}
    training, finite = [], True

    def stage(seed, steps):
        nonlocal finite
        destination = args.output / f"seed{seed}"
        argv = [
            "--config",
            "configs/encoders/n02/tiny_contracts.json",
            "--data-source",
            "synthetic",
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            "--output",
            str(destination),
            "--threads",
            "1",
        ]
        if steps > 100:
            argv.extend(["--resume", str(destination / "checkpoint.pt")])
        result = train(parse_args(argv))
        checkpoint = destination / f"checkpoint{steps}.pt"
        shutil.copyfile(destination / "checkpoint.pt", checkpoint)
        write(destination / f"training{steps}.json", result)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        logs = [
            json.loads(line) for line in (destination / "train_log.jsonl").read_text().splitlines()
        ]
        this_finite = all(
            torch.isfinite(t).all().item() for t in payload["model"].values()
        ) and all(math.isfinite(row[key]) for row in logs for key in ("loss", "grad_norm"))
        finite = finite and this_finite
        training.append(
            {
                "seed": seed,
                "step": steps,
                "checkpoint_sha256": digest(checkpoint),
                "finite": this_finite,
                "training": result,
            }
        )
        write(args.output / "training.json", training)
        return checkpoint

    for seed in (0, 1, 2):
        for steps in (100, 200):
            checkpoint = stage(seed, steps)
            result = assessment(
                checkpoint, range(400, 416), args.output / f"dev_seed{seed}_step{steps}.json"
            )
            development[str(seed)][str(steps)] = result
            write(args.output / "development.json", development)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "step": steps,
                        "gates": result["gates"],
                        "pairs": result["counterfactual"]["both_correct"],
                    }
                ),
                flush=True,
            )
    decision = extension_decision(development, contracts_passed=checks["passed"], finite=finite)
    write(args.output / "decision.json", decision)  # written BEFORE viewing final seeds
    print(json.dumps(decision), flush=True)
    final_step = 800 if decision["extend_to_800"] else 200
    final = {}
    for seed in (0, 1, 2):
        checkpoint = (
            stage(seed, 800)
            if decision["extend_to_800"]
            else args.output / f"seed{seed}/checkpoint200.pt"
        )
        final[str(seed)] = assessment(
            checkpoint, range(600, 632), args.output / f"final_seed{seed}.json"
        )
    assert all(digest(root / path) == sha for path, sha in manifest["source_sha256"].items()), (
        "source changed during the run"
    )
    summary = {
        "decision": decision,
        "final_step": final_step,
        "formal_optimizer_steps": 3 * final_step,
        "development": development,
        "final": final,
        "wall_seconds": time.perf_counter() - started,
    }
    write(args.output / "summary.json", summary)
    print(
        json.dumps({"final_step": final_step, "wall_seconds": summary["wall_seconds"]}), flush=True
    )


if __name__ == "__main__":
    main()

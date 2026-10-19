"""Single-factor common/contrast loss weighting on frozen N02 causal fields.

The paired-NLL control and joint arm share all pixels, cached inverse-variance
weights, common-fit supervision, model, optimizer and sample order. Only the
contrast coefficient changes. Fixed geometry and oracle-common readouts remain
diagnostic interventions; no full encoder training occurs here.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_measurement_isolation import (
    FixedGeometryHead,
    assess,
    build_evaluation_cache,
    cache_manifest,
    class_predictions,
    contrast_of,
    evidence_criteria,
    file_digest,
    state_digest,
    tensor_digest,
    write_json,
)
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder


def joint_components(head, cache, indexes=None):
    """Exact paired Gaussian decomposition using the same fixed variance per class."""
    indexes = slice(None) if indexes is None else indexes
    predicted = class_predictions(head(cache["fields"][indexes]))
    target = cache["templates"][indexes]
    variance = cache["total_variance"][indexes]
    common_error = predicted.mean(1) - target.mean(1)
    contrast_error = contrast_of(predicted) - contrast_of(target)
    return {
        "common": 0.5 * (common_error.square() / variance).mean(),
        "contrast": 0.5 * (contrast_error.square() / variance).mean(),
        "log_variance": 0.5 * variance.log().mean(),
    }


def weighted_objective(components, coefficient):
    return components["common"] + coefficient * components["contrast"] + components["log_variance"]


@torch.no_grad()
def fixed_coefficient(head, train):
    components = joint_components(head, train)
    common, contrast = (float(components[key]) for key in ("common", "contrast"))
    if not math.isfinite(common) or not math.isfinite(contrast) or min(common, contrast) <= 0:
        raise ValueError("initial common and contrast errors must be positive and finite")
    coefficient = common / contrast
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("initial common/contrast coefficient must be positive and finite")
    return coefficient, {key: float(value) for key, value in components.items()}


def gradient_profile(head, train, coefficient):
    """Full-fit-cache component gradients on shared trunk, before any projection."""
    components = joint_components(head, train)
    parameters = tuple(head.head.trunk.parameters())
    common_gradient = torch.autograd.grad(components["common"], parameters, retain_graph=True)
    contrast_gradient = torch.autograd.grad(components["contrast"], parameters)
    common = torch.cat([value.detach().double().flatten() for value in common_gradient])
    contrast = torch.cat([value.detach().double().flatten() for value in contrast_gradient])
    common_norm = float(common.norm())
    contrast_norm = float(contrast.norm())
    denominator = common_norm * contrast_norm
    return {
        "scope": "full training cache; shared measurement-head trunk parameters only",
        "common_loss": float(components["common"].detach()),
        "contrast_loss": float(components["contrast"].detach()),
        "coefficient": coefficient,
        "common_gradient_norm": common_norm,
        "contrast_gradient_norm": contrast_norm,
        "weighted_contrast_gradient_norm": coefficient * contrast_norm,
        "cosine": float(common.dot(contrast)) / denominator if denominator > 0 else None,
        "common_plus_contrast_gradient_norm": float((common + contrast).norm()),
        "common_plus_weighted_contrast_gradient_norm": float(
            (common + coefficient * contrast).norm()
        ),
    }


def false_score_summary(predicted, cache):
    """Readout-only false evidence on truly equal physical templates."""
    physical_difference = cache["templates"][:, 1] - cache["templates"][:, 0]
    no_cue = physical_difference.double().square().flatten(1).sum(1) <= 1e-20
    residual = predicted.double() - cache["observed"].double()[:, None]
    score = (0.5 * residual.square() / cache["total_variance"].double()[:, None]).sum((2, 3, 4))
    gap = score[:, 1] - score[:, 0]
    absolute = gap[no_cue].abs()
    sequences = {}
    for index, identity in enumerate(cache["identities"]):
        if identity.get("world") == "no_cue":
            if not no_cue[index]:
                raise AssertionError("declared no-cue sequence has unequal physical templates")
            sequences.setdefault(identity["seed"], 0.0)
            sequences[identity["seed"]] += float(gap[index])
    rows = []
    for seed, total in sorted(sequences.items()):
        p1 = float(torch.sigmoid(torch.tensor(-total, dtype=torch.float64)))
        rows.append({"seed": seed, "score1_minus_score0": total, "p1": p1})
    return {
        "no_cue_frames": int(no_cue.sum()),
        "frame_mean_abs_log_bayes": float(absolute.mean()) if len(absolute) else None,
        "frame_p95_abs_log_bayes": float(absolute.quantile(0.95)) if len(absolute) else None,
        "all_no_cue_sequences": len(rows),
        "sequence_mean_abs_p1_drift": float(np.mean([abs(row["p1"] - 0.5) for row in rows]))
        if rows
        else None,
        "sequence_false_confidence_ge_0_95": sum(
            max(row["p1"], 1 - row["p1"]) >= 0.95 for row in rows
        ),
        "records": rows,
    }


@torch.no_grad()
def joint_assessment(head, cache, signal_scale):
    metrics = assess(head, cache, signal_scale)
    predicted = torch.cat([class_predictions(head(chunk)) for chunk in cache["fields"].split(16)])
    common = predicted.mean(1)
    target_common = cache["templates"].mean(1)
    contrast = contrast_of(predicted)
    variance = cache["total_variance"].double()
    common_mse = ((common.double() - target_common.double()).square() / variance).mean()
    contrast_mse = (
        (contrast.double() - contrast_of(cache["templates"]).double()).square() / variance
    ).mean()
    metrics["components"] = {
        "common": float(0.5 * common_mse),
        "contrast": float(0.5 * contrast_mse),
        "log_variance": float(0.5 * variance.log().mean()),
    }
    metrics["common_weighted_rmse"] = float(common_mse.sqrt())
    metrics["actual_common_false_score"] = false_score_summary(predicted, cache)
    oracle_common = torch.stack([target_common - contrast, target_common + contrast], 1)
    metrics["oracle_common_false_score"] = false_score_summary(oracle_common, cache)
    return metrics


def joint_criteria(assessment, initial):
    criteria = evidence_criteria(assessment["evaluation"])
    for split in ("train", "evaluation"):
        criteria[f"{split}_common_error_nondegrading"] = (
            assessment[split]["common_weighted_rmse"]
            <= initial[split]["common_weighted_rmse"] + 1e-12
        )
    return criteria


def fit_arm(source, train, evaluation, *, coefficient, schedule, signal_scale, output, label):
    head = FixedGeometryHead(source).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad], lr=3e-4
    )
    protected_before = {
        name: tensor_digest(parameter)
        for name, parameter in head.named_parameters()
        if not parameter.requires_grad
    }
    snapshots, logs = {}, []
    stages = {0, len(schedule) // 2, len(schedule)}

    def snapshot(step):
        snapshots[str(step)] = {
            "train": joint_assessment(head, train, signal_scale),
            "evaluation": joint_assessment(head, evaluation, signal_scale),
            "shared_trunk_gradients": gradient_profile(head, train, coefficient),
        }
        if step:
            criteria = joint_criteria(snapshots[str(step)], snapshots["0"])
            snapshots[str(step)]["criteria"] = criteria
            snapshots[str(step)]["all_criteria_pass"] = all(criteria.values())
        torch.save(head.state_dict(), output / f"{label}_step{step}.pt")
        write_json(output / f"{label}_snapshots.json", snapshots)

    snapshot(0)
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    for step, frame in enumerate(schedule, 1):
        indexes = torch.tensor([2 * int(frame), 2 * int(frame) + 1])
        optimizer.zero_grad(set_to_none=True)
        components = joint_components(head, train, indexes)
        loss = weighted_objective(components, coefficient)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"{label} step{step} has nonfinite objective/gradient")
        optimizer.step()
        logs.append(
            {
                "step": step,
                "cache_batch_frame": int(frame),
                "objective": float(loss.detach()),
                "common": float(components["common"].detach()),
                "contrast": float(components["contrast"].detach()),
                "log_variance": float(components["log_variance"].detach()),
                "coefficient": coefficient,
                "unclipped_gradient_norm": float(gradient_norm),
            }
        )
        if step in stages:
            snapshot(step)
    protected_after = {
        name: tensor_digest(parameter)
        for name, parameter in head.named_parameters()
        if not parameter.requires_grad
    }
    if protected_after != protected_before:
        raise AssertionError("frozen geometry or variance parameters changed")
    write_json(output / f"{label}_optimization.json", logs)
    return snapshots


def _hash_cache(cache):
    return {
        key: tensor_digest(value) for key, value in cache.items() if isinstance(value, torch.Tensor)
    }


def protocol(*, smoke):
    return {
        "diagnostic_only": True,
        "smoke": smoke,
        "seeds": [0] if smoke else [0, 1, 2],
        "checkpoint_step": 200,
        "fit_cache": "previous fixed causal fields: generator seed910, rng seeds910..917",
        "evaluation_seeds": [739] if smoke else list(range(740, 756)),
        "evaluation_split": "validation; same geometry set as earlier round, new seeds",
        "updates_per_arm": 2 if smoke else 200,
        "snapshot_steps": [0, 1, 2] if smoke else [0, 100, 200],
        "objectives": {
            "paired_nll": "Lc + Ld + Lvar; exact decomposition of original paired Gaussian NLL",
            "joint_balanced": "Lc + lambda*Ld + Lvar",
        },
        "coefficient": "lambda=Lc_initial/Ld_initial on full fit cache at original fixed-g head",
        "components": {
            "Lc": ".5 * mean((c-c_target)^2 / cached_total_variance)",
            "Ld": ".5 * mean((d-d_target)^2 / cached_total_variance)",
            "Lvar": ".5 * mean(log(cached_total_variance))",
            "c": "(mu0+mu1)/2",
            "d": "(mu1-mu0)/2",
        },
        "single_factor": "only contrast coefficient; same common fit, inverse variance and pixels",
        "geometry": "production head pre-tanh geometry0 parity +/-20, returned +/-1 fp32",
        "variance": "original checkpoint-predicted VALUES plus cached actual sensor variance",
        "no_cue_mask_in_objective": True,
        "sampler_seed": 910,
        "batch_size": 2,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "threads": 1,
        "criteria": {
            "eval_contrast_nmse_max": 0.5,
            "eval_contrast_cosine_min": 0.5,
            "eval_actual_common_pair_both_min": 12,
            "eval_pair_count": 16,
            "eval_no_cue_false_contrast_ratio_max": 0.1,
            "common_weighted_rmse": "train AND evaluation <= own step0 +1e-12",
        },
        "no_cue_false_score": "diagnostic only; no post-hoc new threshold",
        "gradient_probe": "Lc/Ld on shared trunk with full fit cache at0/100/200; no projection",
        "scope": "frozen-field head joint learnability; no end-to-end or deployment claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("outputs/n02-isolation-20261013/measurement")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(910)
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = protocol(smoke=args.smoke)
    write_json(args.output / "protocol.json", declaration)
    previous_manifest = json.loads((args.previous / "manifest.json").read_text())
    sources = (
        sorted(Path("src/owac/frameworks/n02").glob("*.py"))
        + sorted(Path("src/owac/core").glob("*.py"))
        + [
            Path(__file__),
            Path("src/owac/evaluation/n02_measurement_isolation.py"),
            Path("src/owac/evaluation/n02_bottlenecks.py"),
            Path("src/owac/training/n02.py"),
        ]
    )
    source_before = {str(path): file_digest(path) for path in sources}
    for path, expected in previous_manifest["source_sha256"].items():
        if file_digest(path) != expected:
            raise AssertionError(f"previous production/cache source changed: {path}")
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_before,
        "protocol_sha256": file_digest(args.output / "protocol.json"),
        "previous_manifest_sha256": file_digest(args.previous / "manifest.json"),
        "seeds": {},
    }
    started = time.perf_counter()
    prepared = {}
    # Freeze every input and all three lambda values before either arm optimizes.
    for seed in declaration["seeds"]:
        previous = previous_manifest["seeds"][str(seed)]
        checkpoint = Path(previous["checkpoint"])
        if file_digest(checkpoint) != previous["checkpoint_sha256"]:
            raise AssertionError("source checkpoint changed")
        cache_path = args.previous / f"cache_seed{seed}.pt"
        if file_digest(cache_path) != previous["cache_file_sha256"]:
            raise AssertionError("previous cached fields changed")
        train = torch.load(cache_path, map_location="cpu", weights_only=False)["train"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = N02Encoder(N02Config.from_dict(payload["config"])).eval()
        model.load_state_dict(payload["model"])
        evaluation = build_evaluation_cache(model, seeds=declaration["evaluation_seeds"])
        if set(train["raw_sha256"]) & set(evaluation["raw_sha256"]):
            raise AssertionError("fit and heldout RAW overlap")
        evaluation_path = args.output / f"heldout_seed{seed}.pt"
        torch.save(evaluation, evaluation_path)
        source = model.measurement.head
        coefficient, initial_components = fixed_coefficient(FixedGeometryHead(source), train)
        signal_scale = float(contrast_of(train["templates"]).double().square().mean())
        prepared[seed] = (model, train, evaluation, coefficient, signal_scale)
        manifest["seeds"][str(seed)] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": previous["checkpoint_sha256"],
            "encoder_state_sha256": state_digest(model),
            "previous_cache": str(cache_path),
            "previous_cache_sha256": previous["cache_file_sha256"],
            "fit_tensor_sha256": _hash_cache(train),
            "evaluation_file_sha256": file_digest(evaluation_path),
            "evaluation": cache_manifest(evaluation),
            "joint_coefficient": coefficient,
            "initial_components": initial_components,
            "signal_scale": signal_scale,
        }
    updates = declaration["updates_per_arm"]
    schedule = np.random.default_rng(910).integers(0, 64, size=updates)
    write_json(args.output / "schedule.json", schedule.tolist())
    manifest["schedule_sha256"] = file_digest(args.output / "schedule.json")
    write_json(args.output / "manifest_before_optimization.json", manifest)
    results = {}
    for seed, (model, train, evaluation, coefficient, signal_scale) in prepared.items():
        source = model.measurement.head
        cache_before = {"train": _hash_cache(train), "evaluation": _hash_cache(evaluation)}
        result = {
            "coefficient": coefficient,
            "exact_templates": assess(None, evaluation, signal_scale, exact=True),
            "arms": {},
        }
        for arm, value in (("paired_nll", 1.0), ("joint_balanced", coefficient)):
            result["arms"][arm] = fit_arm(
                source,
                train,
                evaluation,
                coefficient=value,
                schedule=schedule,
                signal_scale=signal_scale,
                output=args.output,
                label=f"seed{seed}_{arm}",
            )
            final = result["arms"][arm][str(updates)]
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "coefficient": value,
                        "contrast_nmse": final["evaluation"]["contrast_nmse"],
                        "both_correct": final["evaluation"]["actual_head_choices"]["both_correct"],
                        "criteria": final["criteria"],
                    }
                ),
                flush=True,
            )
        original = manifest["seeds"][str(seed)]
        protected = {
            "checkpoint_unchanged": file_digest(original["checkpoint"])
            == original["checkpoint_sha256"],
            "previous_cache_file_unchanged": file_digest(original["previous_cache"])
            == original["previous_cache_sha256"],
            "encoder_unchanged": state_digest(model) == original["encoder_state_sha256"],
            "cache_values_unchanged": cache_before
            == {"train": _hash_cache(train), "evaluation": _hash_cache(evaluation)},
        }
        if not all(protected.values()):
            raise AssertionError(protected)
        result["protected"] = protected
        results[str(seed)] = result
        write_json(args.output / "results.json", results)
    if source_before != {str(path): file_digest(path) for path in sources}:
        raise AssertionError("source changed during joint-head control")
    write_json(
        args.output / "summary.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "head_only_updates": len(declaration["seeds"]) * 2 * updates,
            "protocol_sha256": manifest["protocol_sha256"],
            "manifest_before_optimization_sha256": file_digest(
                args.output / "manifest_before_optimization.json"
            ),
            "source_unchanged": True,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()

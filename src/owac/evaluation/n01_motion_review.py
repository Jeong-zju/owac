"""Read-only P2 acceptance: checkpoint identity, stop rules and frozen evaluation.

This entry point never invokes a trainer, optimizer, fit-check or selection run.
All outputs go into a new audit directory, preserving historical artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n01_motion import (
    ErrorCoverageTable,
    MotionEvaluator,
    _collect_calibration_samples,
    _evaluate_region,
    run_evaluation,
)
from owac.raw.n01_motion_data import MotionDataGenerator
from owac.raw.n01_noise import flow_first_order_variance
from owac.training.n01_motion import MotionSampleSpec, decide_platform, load_motion_training_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def linearization_coverage(samples: int = 100000, seed: int = 20261015) -> dict:
    """Known Gaussian flow perturbations; no learned posterior is assumed.

    The edge and layer-occlusion examples start 0.1 px from a discontinuity.
    Their local derivative is zero, but finite perturbations can cross it.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 0.5, (samples, 2))
    sigma = torch.eye(2, dtype=torch.float64) * 0.25
    gradients = {
        "smooth_affine": [2.0, -1.0],
        "strong_edge": [0.0, 0.0],
        "layer_occlusion": [0.0, 0.0],
    }
    errors = {
        "smooth_affine": offsets @ np.array([2.0, -1.0]),
        "strong_edge": (offsets[:, 0] > 0.1).astype(float),
        "layer_occlusion": (offsets[:, 1] > 0.1).astype(float) * 4.0,
    }
    report = {}
    for name, gradient in gradients.items():
        variance = float(flow_first_order_variance(gradient, sigma))
        radius = 1.959963984540054 * np.sqrt(variance)
        report[name] = {
            "predicted_variance": variance,
            "empirical_variance": float(np.var(errors[name])),
            "nominal_coverage": 0.95,
            "observed_coverage": float(np.mean(np.abs(errors[name]) <= radius)),
        }
    return {
        "samples": samples,
        "seed": seed,
        "flow_units": "sensor_px",
        "sigma_flow": sigma.tolist(),
        "cases": report,
        "scope": "analytic counterexamples only; not SEA-RAFT posterior covariance",
    }


def stress_evaluation(config, checkpoint: Path, output: Path, device: str) -> dict:
    """Frozen diagnostic conditions with measured motion and one isolated anomaly.

    These are new synthetic diagnostics, never checkpoint selection inputs.
    The original per-case labels are insufficient to establish actual speed.
    """
    evaluator = MotionEvaluator(config, device=device, checkpoint=checkpoint)
    evaluator._build_estimators()
    table = ErrorCoverageTable.fit(
        _collect_calibration_samples(evaluator, evaluator.specs("calibration", limit=32)),
        split="calibration",
        bins=config.evaluation.risk_bins,
        nominal_coverage=config.evaluation.risk_quantile,
    )
    original_generator = evaluator.generator
    recipes = original_generator.config.cases
    common = dict(
        gain_source=1400.0,
        gain_reference=1400.0,
        exposure_known=True,
        jitter_px=0.0,
        rotation_deg=0.0,
        warp_amplitude_px=0.0,
        blur_length_px=0.0,
        blur_samples=1,
    )
    conditions = {
        "fast_alignable": (
            "exposure_gain",
            replace(
                recipes["exposure_gain"],
                **common,
                background="blobs",
                background_step_px=(12.0, 0.0),
                occluder=False,
            ),
        ),
        "fast_hard": (
            "repeated_texture",
            replace(
                recipes["repeated_texture"],
                **common,
                background_step_px=(12.0, 0.0),
                occluder=True,
                object_step_px=(-12.0, 0.0),
            ),
        ),
        "slow_structural_change": (
            "local_nonrigid",
            replace(
                recipes["local_nonrigid"],
                gain_source=1400.0,
                gain_reference=1400.0,
                exposure_known=True,
                background_step_px=(0.25, 0.0),
                jitter_px=0.0,
                warp_amplitude_px=12.0,
            ),
        ),
        "current_frame_anomaly": (
            "exposure_gain",
            replace(
                recipes["exposure_gain"],
                background="blobs",
                occluder=False,
                background_step_px=(2.0, 0.0),
                jitter_px=0.0,
                gain_source=1400.0,
                gain_reference=2800.0,
                exposure_known=False,
            ),
        ),
    }
    result = {}
    for condition, (case, recipe) in conditions.items():
        evaluator.generator = MotionDataGenerator(
            replace(original_generator.config, cases={**recipes, case: recipe})
        )
        records = []
        motions = []
        for index in range(2):
            spec = MotionSampleSpec(
                split="test",
                case=case,
                sequence=f"p2-independent-review-{condition}-{index}",
                history_offset=1,
            )
            pair, *_ = evaluator._corrected_pair(spec)
            truth = pair.supervision
            magnitude = torch.linalg.vector_norm(
                truth.sensor_xy_source - truth.sensor_xy_reference, dim=-1
            )
            motions.append(float(magnitude[truth.valid].mean()))
            members = _evaluate_region(evaluator, "test", [spec], table)
            for member in members:
                member["bucket"] = condition
            records.extend(members)
        result[condition] = {
            "recipe": recipe.to_dict(),
            "records": records,
            "mean_true_displacement_sensor_px": motions,
            "role": "synthetic diagnostic only; no parameter updates",
        }
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def review(run_root: Path, output: Path, device: str) -> dict:
    if output.exists():
        raise FileExistsError(f"use a fresh audit directory: {output}")
    output.mkdir(parents=True)
    result = {"formal_training_started": False, "parameter_updates": 0, "runs": {}}
    for mode in ("cfa4", "rgb3"):
        config_path = Path("configs/encoders/n01_full") / (
            "p2_motion_train.json" if mode == "cfa4" else "p2_motion_train_rgb3_formal.json"
        )
        config = load_motion_training_config(config_path)
        for seed in (101, 211, 307):
            tag = f"seed-{seed}" + ("-rgb3" if mode == "rgb3" else "")
            old = json.loads((run_root / tag / "receipt.json").read_text())
            history = old["val_history"]
            stopped = old["platform"]["stopped_at_update"]
            final_tail_start = (
                stopped - config.budget.plateau_checks * config.budget.validation_interval
            )
            decision = decide_platform(
                val_history=history,
                budget=old["budget"],
                interval=config.budget.validation_interval,
                monitor=config.budget.monitor,
                delta_val=config.budget.delta_val,
                plateau_checks=config.budget.plateau_checks,
                lr_drops=old["lr_drops"],
                final_phase=config.schedule.is_final_phase(max(0, final_tail_start)),
                diagnostics_passed=old["diagnostics"]["passed"],
            )
            identity = {}
            for kind in ("best", "last"):
                path = run_root / tag / "checkpoints" / f"{kind}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                persisted = [
                    "model",
                    "optimizer",
                    "scheduler",
                    "scaler",
                    "rng",
                    "sampler",
                    "hashes",
                    "samples_consumed",
                    "diagnostics",
                ]
                assert all(key in payload for key in persisted)
                assert payload["seed"] == seed and payload["mode"] == mode
                identity[kind] = {
                    "path": str(path),
                    "sha256": sha256(path),
                    "update": payload["update"],
                    "core_fingerprint": payload["core_fingerprint"],
                    "data_config": payload["hashes"]["data_config"],
                    "persisted_fields": persisted,
                    "rng_fields": sorted(payload["rng"]),
                    "sampler_fields": sorted(payload["sampler"]),
                }
                assert payload["update"] == (stopped if kind == "last" else old["best"]["update"])
                del payload
            checkpoint = run_root / tag / "checkpoints/best.pt"
            evaluated = run_evaluation(
                config,
                output=output / tag,
                device=device,
                checkpoint=checkpoint,
                config_path=config_path,
            )
            assert sha256(checkpoint) == identity["best"]["sha256"]
            result["runs"][tag] = {
                "old_status": old["platform"]["status"],
                "stop_update": stopped,
                "reviewed_status": decision.status,
                "conditions": decision.details,
                "reasons": decision.reasons,
                "checkpoints": identity,
                "evaluation": str(output / tag / "receipt.json"),
                "test": evaluated["regions_report"]["test"],
                "static_false_rejection": evaluated["static_false_rejection"],
                "latency": evaluated["latency"],
            }
            if seed == 101:
                stress_evaluation(config, checkpoint, output / f"stress-{mode}.json", device)
            (output / "review.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
            print(json.dumps({"completed": tag, "reviewed_status": decision.status}), flush=True)
    result["first_order_coverage"] = linearization_coverage()
    (output / "review.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    review(args.run_root, args.output, args.device)


if __name__ == "__main__":
    main()

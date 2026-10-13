"""Fixed 2x2 diagnostic continuation, isolated from production N02 training.

Balanced counterfactual fitting and a one-sided geometry-gradient intervention
are experimental objectives. Neither changes the encoder's inference equations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from owac.core.encoder_contract import ComputeBudget, EmbodimentContext
from owac.evaluation.n02_contracts import assessment, extension_decision
from owac.evaluation.n02_gradients import batch_digest, clone_with_optimizer, sha256, tree_digest
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.readout import representation_from_output
from owac.training.n02 import _frame, counterfactual_measurement_loss, step_terms, synthetic_batch

ARMS = ("baseline", "balanced_cf", "projected_geometry", "both")
TERMS = ("measurement", "geometry_set", "revision", "diversity", "mixture", "counterfactual")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def paired_cf_components(model, representation, targets, *, sensor_variance):
    """Exact Gaussian decomposition using adjacent semantic pairs, not class means.

    Lc and Ld are nonnegative quadratic terms. Their sum plus one log-variance
    term equals the original per-slot NLL when the shared variance contract holds.
    """
    mean, log_variance, _ = model.measurement.predict(representation.h)
    B, V, K, C, H, W = mean.shape
    if V != 1 or K % 2 or targets.shape[:3] != (B, 2, C):
        raise ValueError("paired decomposition requires one view and even binary slot pairs")
    targets = F.adaptive_avg_pool2d(
        targets.to(mean).reshape(B * 2, C, *targets.shape[-2:]), (H, W)
    ).reshape(B, 2, C, H, W)
    sensor = torch.as_tensor(sensor_variance, device=mean.device, dtype=mean.dtype)
    if sensor.ndim == 4:
        sensor = sensor[:, None, None]
    variance = log_variance.exp() + sensor
    if not torch.equal(variance[:, :, ::2], variance[:, :, 1::2]):
        raise ValueError("paired NLL identity requires equal variance within every slot pair")
    pair_variance = variance[:, :, ::2]
    common = (mean[:, :, ::2] + mean[:, :, 1::2]) / 2
    difference = (mean[:, :, 1::2] - mean[:, :, ::2]) / 2
    target_common = ((targets[:, 0] + targets[:, 1]) / 2)[:, None, None]
    target_difference = ((targets[:, 1] - targets[:, 0]) / 2)[:, None, None]
    common_residual, difference_residual = common - target_common, difference - target_difference
    return {
        "common": (0.5 * common_residual.square() / pair_variance).mean(),
        "difference": (0.5 * difference_residual.square() / pair_variance).mean(),
        "log_variance": (0.5 * pair_variance.log()).mean(),
        "common_mse": common_residual.square().mean(),
        "difference_mse": difference_residual.square().mean(),
        "difference_target_mse": target_difference.square().mean(),
        "model_std": log_variance.exp().mean().sqrt(),
    }


def objective(model, batch, *, contrast_weight=1.0, balanced=False, diversity_margin=0.1):
    """Same active-frame means/detached frame state as production; optional CF replacement."""
    tensors = {name: [] for name in TERMS}
    counts = {name: 0 for name in TERMS}
    components = {}
    config = model.config
    state = model.init_state(batch.batch_size, batch.camera, "cpu")
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    for index in range(batch.num_steps):
        context = batch.contexts[index] if batch.contexts else EmbodimentContext()
        output = model.step(_frame(batch, index), context, state, budget)
        state = model.detach_state(output.state)
        rep = representation_from_output(output)
        terms = step_terms(
            rep,
            anchor=batch.anchor[index],
            wrong=batch.wrong[index],
            post_contradiction=batch.post_contradiction[index],
            observations=int(output.diagnostics.get("likelihood_observations", 1)),
            diversity_active=batch.multi_solution
            if batch.identifiable_now is None
            else not bool(batch.identifiable_now[index].all()),
            diversity_margin=diversity_margin,
            ambiguity_prior=context.hypothesis_prior,
            identifiable_now=None
            if batch.identifiable_now is None
            else batch.identifiable_now[index],
            measurement_normalizer=(
                rep.h.shape[-2] * rep.h.shape[-1] * model.measurement.out_channels
                if config.measurement.energy_reduction == "sum"
                else 1.0
            ),
        )
        if batch.counterfactual_templates is None:
            raise ValueError("this diagnostic requires physical template supervision")
        sensor = output.state.diagnostics["_buffer"].frames[-1].variance
        values = paired_cf_components(
            model, rep, batch.counterfactual_templates[index], sensor_variance=sensor
        )
        terms["counterfactual"] = (
            values["common"] + contrast_weight * values["difference"] + values["log_variance"]
            if balanced
            else counterfactual_measurement_loss(
                model, rep, batch.counterfactual_templates[index], sensor_variance=sensor
            )
        )
        terms["counts"]["counterfactual"] = batch.batch_size
        for name, value in values.items():
            components.setdefault(name, []).append(value)
        for name in TERMS:
            if terms[name] is not None:
                tensors[name].append(terms[name])
                counts[name] += terms["counts"][name]
    means = {name: torch.stack(values).mean() for name, values in tensors.items() if values}
    weighted = {
        name: float(getattr(config.loss_weights, name)) * value for name, value in means.items()
    }
    total = sum(weighted.values(), torch.zeros(()))
    return (
        total,
        weighted,
        {
            "unweighted": {name: float(value.detach()) for name, value in means.items()},
            "weighted": {name: float(value.detach()) for name, value in weighted.items()},
            "active_steps": {name: len(values) for name, values in tensors.items()},
            "effective_samples": counts,
            "components": {
                name: float(torch.stack(values).mean().detach())
                for name, values in components.items()
            },
        },
    )


def global_norm(gradients):
    nonnull = [grad.norm() for grad in gradients if grad is not None]
    return float(torch.stack(nonnull).norm()) if nonnull else 0.0


def project_geometry(total, mixture, geometry, indices):
    """Remove only weighted mixture's component opposing weighted geometry loss."""
    dot = sum(
        float((mixture[i].double() * geometry[i].double()).sum())
        for i in indices
        if mixture[i] is not None and geometry[i] is not None
    )
    norm2 = sum(
        float(geometry[i].double().square().sum()) for i in indices if geometry[i] is not None
    )
    mixture_norm2 = sum(
        float(mixture[i].double().square().sum()) for i in indices if mixture[i] is not None
    )
    coefficient = dot / norm2 if dot < 0 and norm2 > 0 else 0.0
    projected = list(total)
    projected_mixture = list(mixture)
    for i in indices:
        if coefficient and geometry[i] is not None:
            if total[i] is None or mixture[i] is None:
                raise ValueError("projection cannot create a previously disconnected gradient")
            projected[i] = total[i] - coefficient * geometry[i]
            projected_mixture[i] = mixture[i] - coefficient * geometry[i]
    after_dot = sum(
        float((projected_mixture[i].double() * geometry[i].double()).sum())
        for i in indices
        if projected_mixture[i] is not None and geometry[i] is not None
    )
    after_norm2 = sum(
        float(projected_mixture[i].double().square().sum())
        for i in indices
        if projected_mixture[i] is not None
    )
    return tuple(projected), {
        "applied": coefficient != 0,
        "mixture_geometry_dot_before": dot,
        "mixture_geometry_cos_before": dot / math.sqrt(norm2 * mixture_norm2)
        if norm2 and mixture_norm2
        else None,
        "mixture_geometry_dot_after": after_dot,
        "mixture_geometry_cos_after": after_dot / math.sqrt(norm2 * after_norm2)
        if norm2 and after_norm2
        else None,
        "projection_coefficient": coefficient,
        "geometry_gradient_norm": math.sqrt(norm2),
    }


def compute_gradients(model, total, weighted, *, project):
    named = list(model.named_parameters())
    parameters = [parameter for _, parameter in named]
    indices = [i for i, (name, _) in enumerate(named) if ".head.geometry." in name]
    # The intervention only needs this parameter block. Asking autograd for
    # every encoder parameter here would add two unnecessary full backwards.
    geometry_block = torch.autograd.grad(
        weighted["geometry_set"],
        [parameters[i] for i in indices],
        retain_graph=True,
        allow_unused=True,
    )
    mixture_block = torch.autograd.grad(
        weighted["mixture"],
        [parameters[i] for i in indices],
        retain_graph=True,
        allow_unused=True,
    )
    geometry, mixture = [None] * len(parameters), [None] * len(parameters)
    for i, gg, gm in zip(indices, geometry_block, mixture_block, strict=True):
        geometry[i], mixture[i] = gg, gm
    original = torch.autograd.grad(total, parameters, allow_unused=True)
    projected, report = project_geometry(original, mixture, geometry, indices)
    report["opposing_component_found"] = report["applied"]
    report["applied"] = project and report["applied"]
    if not project:
        report["mixture_geometry_dot_after"] = report["mixture_geometry_dot_before"]
        report["mixture_geometry_cos_after"] = report["mixture_geometry_cos_before"]
    report.update(
        intervention_enabled=project,
        baseline_global_norm=global_norm(original),
        intervened_global_norm=global_norm(projected if project else original),
    )
    return projected if project else original, geometry, report


def optimizer_step(model, optimizer, gradients, geometry, *, clip=1.0, fixed_factor=None):
    named = list(model.named_parameters())
    before = [parameter.detach().clone() for _, parameter in named]
    optimizer.zero_grad(set_to_none=True)
    for (_, parameter), gradient in zip(named, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    norm = global_norm(gradients)
    if fixed_factor is None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    else:
        for _, parameter in named:
            if parameter.grad is not None:
                parameter.grad.mul_(fixed_factor)
    optimizer.step()
    geom_ss, other_ss, geom_dot = 0.0, 0.0, 0.0
    for index, (name, parameter) in enumerate(named):
        delta = (parameter.detach() - before[index]).double()
        if ".head.geometry." in name:
            geom_ss += float(delta.square().sum())
            if geometry[index] is not None:
                geom_dot += float((geometry[index].double() * delta).sum())
        else:
            other_ss += float(delta.square().sum())
    return {
        "clip_factor": min(1.0, clip / (norm + 1e-6)) if fixed_factor is None else fixed_factor,
        "geometry_parameter_delta_norm": math.sqrt(geom_ss),
        "nongeometry_parameter_delta_norm": math.sqrt(other_ss),
        "geometry_gradient_dot_actual_delta": geom_dot,
    }


def make_batch(config, generator_seed, rng_seed):
    return synthetic_batch(
        config,
        np.random.default_rng(rng_seed),
        batch_size=2,
        num_frames=8,
        frame_size=64,
        seed=generator_seed,
        split="train",
    )


def checkpoint_path(root, seed):
    return Path(root) / f"seed{seed}" / "checkpoint200.pt"


def prepare(output, checkpoint_root, *, steps):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "protocol.json").exists():
        raise FileExistsError("refusing to replace an existing prepared experiment")
    protocol = {
        "arms": list(ARMS),
        "seeds": [0, 1, 2],
        "diagnostic_steps": steps,
        "original_step": 200,
        "total_optimizer_updates": 3 * len(ARMS) * steps,
        "train_generator_seed": 920,
        "train_rng": "920 + zero-based diagnostic step",
        "calibration_generator_seed": 930,
        "calibration_rng": list(range(930, 938)),
        "lambda": "one global sum(mean Lc)/sum(mean Ld) over 3 checkpoints x8 calibration batches",
        "batch_size": 2,
        "sequence_length": 8,
        "frame_size": 64,
        "development_seeds": list(range(740, 756)),
        "holdout_seeds": list(range(760, 776)),
        "gradient_intervention": (
            "After active-frame means, remove mixture component opposed to geometry_set, "
            "on geometry MLP block only."
        ),
        "clipping": (
            "Production global clip after intervention; "
            "nongeometry Adam updates may change indirectly."
        ),
        "local_controls": (
            "First step of 6 projection arms: disposable identical gradients "
            "with baseline clip factor, no continuation."
        ),
        "decision": (
            "Original dev behavior gates plus at least 2/3 seeds passing identical "
            "holdout count thresholds; no automatic extra training beyond fixed200."
        ),
        "scope": (
            "Diagnostic continuation with extra synthetic template supervision; "
            "no inference-core change."
        ),
    }
    write(output / "protocol.json", protocol)
    check_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_n02_joint_training.py",
        "tests/test_n02_gradients.py",
        "tests/test_n02_contracts.py",
    ]
    checked = subprocess.run(check_command, capture_output=True, text=True, check=False)
    (output / "preflight.log").write_text(checked.stdout + checked.stderr)
    write(
        output / "preflight.json",
        {
            "command": check_command,
            "passed": checked.returncode == 0,
            "exit_code": checked.returncode,
            "log_sha256": sha256(output / "preflight.log"),
        },
    )
    if checked.returncode:
        raise RuntimeError("fixed-contract preflight failed; no calibration or training executed")
    sources = sorted(Path("src/owac/frameworks/n02").glob("*.py")) + [
        Path("src/owac/training/n02.py"),
        Path("src/owac/evaluation/n02_gradients.py"),
        Path("src/owac/evaluation/n02_contracts.py"),
        Path("src/owac/evaluation/n02_acceptance.py"),
        Path(__file__),
    ]
    paths = [checkpoint_path(checkpoint_root, seed) for seed in (0, 1, 2)]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    config = N02Config.from_dict(payloads[0]["config"])
    assert all(payload["config"] == payloads[0]["config"] for payload in payloads)
    cache = output / "cache"
    cache.mkdir()
    inputs = {}
    for index in range(steps):
        batch = make_batch(config, 920, 920 + index)
        inputs[str(index)] = batch_digest(batch)
        torch.save(batch, cache / f"train{index}.pt")
    calibration = [make_batch(config, 930, seed) for seed in range(930, 938)]
    torch.save(calibration, cache / "calibration.pt")
    records = []
    for seed, payload in enumerate(payloads):
        model, _ = clone_with_optimizer(payload)
        before = tree_digest(payload)
        with torch.no_grad():
            for index, batch in enumerate(calibration):
                _, _, record = objective(model, batch)
                records.append({"seed": seed, "rng_seed": 930 + index, **record["components"]})
        assert tree_digest(payload) == before
    common = float(np.mean([row["common"] for row in records]))
    difference = float(np.mean([row["difference"] for row in records]))
    if not math.isfinite(common) or not math.isfinite(difference) or difference <= 0:
        raise ValueError("nonpositive/nonfinite calibration difference loss")
    weight = common / difference
    write(
        output / "calibration.json",
        {"common": common, "difference": difference, "lambda": weight, "records": records},
    )
    manifest = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path): sha256(path) for path in sources},
        "checkpoint_sha256": {str(path): sha256(path) for path in paths},
        "train_input_sha256": inputs,
        "calibration_input_sha256": [batch_digest(batch) for batch in calibration],
        "config": config.as_dict(),
        "config_sha256": tree_digest(config.as_dict()),
        "lambda": weight,
        "torch_version": torch.__version__,
        "threads": 1,
        "optimizer_groups": [
            {k: v for k, v in p["optimizer"]["param_groups"][0].items() if k != "params"}
            for p in payloads
        ],
        "checkpoint_root": str(checkpoint_root),
        "steps": steps,
        "contracts_passed": checked.returncode == 0,
        "preflight_sha256": sha256(output / "preflight.json"),
        "protocol_sha256": sha256(output / "protocol.json"),
    }
    write(output / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def fit_metrics(model, calibration):
    rows = [objective(model, batch)[2] for batch in calibration]
    return {
        "geometry_mse": float(np.mean([row["unweighted"]["geometry_set"] for row in rows])),
        **{
            name: float(np.mean([row["components"][name] for row in rows]))
            for name in rows[0]["components"]
        },
        "model_std_batch_median": float(
            np.median([row["components"]["model_std"] for row in rows])
        ),
        "model_std_batch_range": [
            min(row["components"]["model_std"] for row in rows),
            max(row["components"]["model_std"] for row in rows),
        ],
        "scope": "Fixed calibration batches; descriptive joint fit, not behavior selection.",
    }


def worker(output, seed):
    manifest = json.loads((output / "manifest.json").read_text())
    original_path = checkpoint_path(manifest["checkpoint_root"], seed)
    payload = torch.load(original_path, map_location="cpu", weights_only=False)
    payload_before = tree_digest(payload)
    calibration = torch.load(output / "cache/calibration.pt", weights_only=False)
    root = output / f"seed{seed}"
    root.mkdir()
    started, cpu_started, load_start = time.perf_counter(), time.process_time(), os.getloadavg()
    initial = assessment(original_path, range(740, 756), root / "dev_step0.json")
    model, _ = clone_with_optimizer(payload)
    initial_fit = fit_metrics(model, calibration)
    for arm in ARMS:
        destination = root / arm
        destination.mkdir()
        model, optimizer = clone_with_optimizer(payload)
        model.train()
        development = {"0": initial}
        fits = {"0": initial_fit}
        balanced, project = arm in ("balanced_cf", "both"), arm in ("projected_geometry", "both")
        with (destination / "train_log.jsonl").open("w") as handle:
            for step in range(manifest["steps"]):
                step_started, step_cpu_started = time.perf_counter(), time.process_time()
                batch = torch.load(output / "cache" / f"train{step}.pt", weights_only=False)
                assert batch_digest(batch) == manifest["train_input_sha256"][str(step)]
                total, weighted, record = objective(
                    model,
                    batch,
                    balanced=balanced,
                    contrast_weight=manifest["lambda"],
                    diversity_margin=payload["run"]["diversity_margin"],
                )
                if not torch.isfinite(total):
                    raise FloatingPointError("nonfinite diagnostic objective")
                gradients, geometry, projection = compute_gradients(
                    model, total, weighted, project=project
                )
                clip = float(payload["run"]["grad_clip"])
                baseline_factor = min(1.0, clip / (projection["baseline_global_norm"] + 1e-6))
                if project and step == 0:
                    temporary = {
                        **payload,
                        "model": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                    }
                    control_model, control_optimizer = clone_with_optimizer(temporary)
                    record["local_baseline_clip_control"] = optimizer_step(
                        control_model,
                        control_optimizer,
                        gradients,
                        geometry,
                        clip=clip,
                        fixed_factor=baseline_factor,
                    )
                update = optimizer_step(model, optimizer, gradients, geometry, clip=clip)
                record.update(
                    seed=seed,
                    arm=arm,
                    diagnostic_step=step + 1,
                    total_seen=201 + step,
                    loss=float(total.detach()),
                    projection=projection,
                    update=update,
                    baseline_clip_factor=baseline_factor,
                    input_sha256=manifest["train_input_sha256"][str(step)],
                    step_wall_seconds=time.perf_counter() - step_started,
                    step_cpu_seconds=time.process_time() - step_cpu_started,
                )
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                if (step + 1) % 20 == 0:
                    print(
                        f"[joint] seed={seed} arm={arm} step={step + 1} loss={record['loss']:.6f}",
                        flush=True,
                    )
                if step + 1 in (100, manifest["steps"]):
                    saved = destination / f"diagnostic_step{step + 1}.pt"
                    diagnostic_payload = {
                        **payload,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": 201 + step,
                        "diagnostic_step": step + 1,
                        "total_seen": 201 + step,
                        "diagnostic_arm": arm,
                        "diagnostic_lambda": manifest["lambda"],
                        "diagnostic_scope": (
                            "fixed-input continuation; original RNG state is not its data cursor"
                        ),
                    }
                    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
                    torch.save(diagnostic_payload, saved)
                    development[str(step + 1)] = assessment(
                        saved, range(740, 756), destination / f"dev_step{step + 1}.json"
                    )
                    fits[str(step + 1)] = fit_metrics(model, calibration)
                    write(destination / "development.json", development)
                    write(destination / "fit.json", fits)
        assert tree_digest(payload) == payload_before
        assert sha256(original_path) == manifest["checkpoint_sha256"][str(original_path)]
    write(
        root / "resources.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "load_start": load_start,
            "load_end": os.getloadavg(),
            "finite": True,
        },
    )


def final_worker(output, seed):
    for arm in ARMS:
        destination = output / f"seed{seed}" / arm
        report = assessment(
            destination / "diagnostic_step200.pt",
            range(760, 776),
            destination / "holdout_step200.json",
        )
        write(destination / "holdout_summary.json", report)
        print(f"[joint-final] seed={seed} arm={arm}", flush=True)


def launch_workers(output, phase):
    processes = []
    for seed in (0, 1, 2):
        handle = (output / f"{phase}_seed{seed}.log").open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "owac.evaluation.n02_joint_training",
                "--phase",
                phase,
                "--output",
                str(output),
                "--seed",
                str(seed),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        processes.append((process, handle))
    for process, handle in processes:
        code = process.wait()
        handle.close()
        if code:
            raise RuntimeError(f"diagnostic {phase} worker exited {code}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/n02-joint-controls-20261013/factorial")
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs/n02-contracts-20261013")
    )
    parser.add_argument("--phase", choices=("prepare", "worker", "final", "run"), default="run")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.phase == "prepare":
        prepare(args.output, args.checkpoint_root, steps=200)
        return 0
    if args.phase == "worker":
        worker(args.output, args.seed)
        return 0
    if args.phase == "final":
        final_worker(args.output, args.seed)
        return 0
    started, load_start = time.perf_counter(), os.getloadavg()
    manifest = json.loads((args.output / "manifest.json").read_text())
    launch_workers(args.output, "worker")
    decisions, development = {}, {}
    for arm in ARMS:
        results = {
            str(seed): json.loads(
                (args.output / f"seed{seed}" / arm / "development.json").read_text()
            )
            for seed in (0, 1, 2)
        }
        development[arm] = results
        finite = all(
            json.loads((args.output / f"seed{seed}/resources.json").read_text())["finite"]
            for seed in (0, 1, 2)
        )
        decisions[arm] = extension_decision(
            results, contracts_passed=manifest["contracts_passed"], finite=finite
        )
        decisions[arm]["development_gates_pass"] = decisions[arm].pop("extend_to_800")
    write(args.output / "decision_before_holdout.json", decisions)
    launch_workers(args.output, "final")
    final = {
        arm: {
            str(seed): json.loads(
                (args.output / f"seed{seed}" / arm / "holdout_summary.json").read_text()
            )
            for seed in (0, 1, 2)
        }
        for arm in ARMS
    }
    holdout_qualified = {
        arm: {
            seed: all(
                report["gates"][name]["passed"] >= threshold
                for name, threshold in (
                    ("correct", 12),
                    ("misleading", 4),
                    ("unidentifiable", 12),
                    ("stale", 4),
                )
            )
            and report["counterfactual"]["both_correct"] >= 8
            for seed, report in reports.items()
        }
        for arm, reports in final.items()
    }
    assert all(sha256(path) == value for path, value in manifest["checkpoint_sha256"].items())
    assert all(sha256(path) == value for path, value in manifest["source_sha256"].items())
    write(
        args.output / "summary.json",
        {
            "decision": decisions,
            "holdout_seed_qualified": holdout_qualified,
            "evidence_supports_further_training": {
                arm: decisions[arm]["development_gates_pass"]
                and sum(holdout_qualified[arm].values()) >= 2
                for arm in ARMS
            },
            "development": development,
            "final": final,
            "diagnostic_optimizer_updates": 2400,
            "disposable_clip_controls": 6,
            "original_checkpoints_unchanged": True,
            "source_unchanged": True,
            "automatically_extended": False,
            "wall_seconds": time.perf_counter() - started,
            "load_start": load_start,
            "load_end": os.getloadavg(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

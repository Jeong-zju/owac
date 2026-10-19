"""Read-only loss-gradient and disposable AdamW-step audit for N02.

This is a fixed-batch local diagnostic, not extra training or a behavior score.
The objective uses production loss functions and exactly their frame aggregation.
Original checkpoint bytes, parameters, and AdamW state are never updated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from owac.core.encoder_contract_n02 import ComputeBudget, EmbodimentContext
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import build_encoder
from owac.frameworks.n02.readout import representation_from_output
from owac.training.n02 import (
    DIVERSITY_MARGIN,
    StepBatch,
    _frame,
    counterfactual_measurement_loss,
    step_terms,
    synthetic_batch,
)

TERMS = ("measurement", "geometry_set", "revision", "diversity", "mixture", "counterfactual")
GROUPS = ("geometry", "contrast", "common_decoder_trunk", "variance", "remaining_encoder")
ARMS = (
    "full",
    "omit_mixture",
    "omit_geometry_set",
    "omit_counterfactual",
    "full_no_clip",
    "zero_current_gradient",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(tree):
    """Content digest, independent of serialization storage IDs."""
    digest = hashlib.sha256()

    def add(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode())

    add(tree)
    return digest.hexdigest()


def batch_digest(batch: StepBatch):
    return tree_digest(
        {
            "values": batch.values,
            "masks": batch.masks,
            "anchor": batch.anchor,
            "wrong": batch.wrong,
            "post": batch.post_contradiction,
            "identifiable": batch.identifiable_now,
            "templates": batch.counterfactual_templates,
            "sample_xy": batch.sample_xy,
            "priors": [context.hypothesis_prior for context in batch.contexts],
            "prior_available": [context.hypothesis_prior_available for context in batch.contexts],
        }
    )


def branch(name):
    if ".head.geometry." in name:
        return "geometry"
    if ".head.contrast." in name:
        return "contrast"
    if ".head.log_variance." in name:
        return "variance"
    if ".head.trunk." in name or ".head.value." in name:
        return "common_decoder_trunk"
    return "remaining_encoder"


def training_objective(model, batch, *, diversity_margin=DIVERSITY_MARGIN):
    """Production eight-frame objective, including active-step normalization.

    State is detached at every frame, exactly as in training.n02.train. Templates
    and class labels reach only the production loss functions and diagnostics.
    """
    config = model.config
    names = TERMS if config.loss_weights.counterfactual > 0 else TERMS[:-1]
    tensors = {name: [] for name in names}
    counts = {name: 0 for name in names}
    state = model.init_state(batch.batch_size, batch.camera, "cpu")
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    functional = []
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
        if config.loss_weights.counterfactual > 0:
            if batch.counterfactual_templates is None:
                raise ValueError("counterfactual targets missing")
            terms["counterfactual"] = counterfactual_measurement_loss(
                model,
                rep,
                batch.counterfactual_templates[index],
                sensor_variance=output.state.diagnostics["_buffer"].frames[-1].variance,
            )
            terms["counts"]["counterfactual"] = batch.batch_size
        for name in names:
            if terms[name] is not None:
                tensors[name].append(terms[name])
                counts[name] += terms["counts"][name]
        with torch.no_grad():
            mean, _, geometry = model.measurement.predict(rep.h)
            target_geometry = (torch.arange(config.K) % 2) * 2.0 - 1.0
            record = {"geometry_mse": float((geometry[..., 0] - target_geometry).square().mean())}
            if batch.counterfactual_templates is not None:
                targets = batch.counterfactual_templates[index]
                B, _, C, H, W = targets.shape
                target = F.adaptive_avg_pool2d(targets.reshape(B * 2, C, H, W), mean.shape[-2:])
                target = target.reshape(B, 2, C, *mean.shape[-2:])
                true_delta = target[:, 1] - target[:, 0]
                predicted_delta = mean[:, 0, 1::2].mean(1) - mean[:, 0, ::2].mean(1)
                record.update(
                    contrast_sse=float((predicted_delta - true_delta).square().sum()),
                    contrast_target_ss=float(true_delta.square().sum()),
                    contrast_prediction_ss=float(predicted_delta.square().sum()),
                    contrast_dot=float((true_delta * predicted_delta).sum()),
                    contrast_elements=true_delta.numel(),
                    contrast_signal_samples=int((true_delta.square().flatten(1).sum(1) > 0).sum()),
                )
            functional.append(record)
    means = {name: torch.stack(values).mean() for name, values in tensors.items() if values}
    weighted = {
        name: float(getattr(config.loss_weights, name)) * val for name, val in means.items()
    }
    total = sum(weighted.values(), torch.zeros(()))
    metrics = {"geometry_mse": float(np.mean([item["geometry_mse"] for item in functional]))}
    if "contrast_sse" in functional[0]:
        sums = {
            key: sum(item[key] for item in functional)
            for key in functional[0]
            if key != "geometry_mse"
        }
        count, target_ss, pred_ss = (
            sums["contrast_elements"],
            sums["contrast_target_ss"],
            sums["contrast_prediction_ss"],
        )
        metrics.update(
            contrast_mse=sums["contrast_sse"] / count,
            contrast_nrmse=(sums["contrast_sse"] / target_ss) ** 0.5 if target_ss else None,
            contrast_cosine=sums["contrast_dot"] / (target_ss * pred_ss) ** 0.5
            if target_ss and pred_ss
            else None,
            contrast_target_rms=(target_ss / count) ** 0.5,
            contrast_prediction_rms=(pred_ss / count) ** 0.5,
            contrast_signal_samples=sums["contrast_signal_samples"],
        )
    return (
        total,
        weighted,
        {
            "unweighted": {name: float(value.detach()) for name, value in means.items()},
            "weighted": {name: float(value.detach()) for name, value in weighted.items()},
            "active_steps": {name: len(values) for name, values in tensors.items()},
            "effective_samples": counts,
            "functional": metrics,
        },
    )


def vector(grads, parameters, indices):
    return torch.cat(
        [
            (torch.zeros_like(parameters[i]) if grads[i] is None else grads[i])
            .detach()
            .double()
            .flatten()
            for i in indices
        ]
    )


def decompose_gradients(weighted, total, named_parameters, *, clip=1.0):
    """Differentiate each term and independently check the actual total backward."""
    names, parameters = zip(*named_parameters, strict=True)
    term_grads = {
        name: torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        for name, value in weighted.items()
    }
    total_grads = torch.autograd.grad(total, parameters, allow_unused=True)
    all_indices = list(range(len(parameters)))
    total_vector = vector(total_grads, parameters, all_indices)
    reconstruction = sum(
        (vector(grads, parameters, all_indices) for grads in term_grads.values()),
        torch.zeros_like(total_vector),
    )
    discrepancy = (reconstruction - total_vector).norm()
    relative = float(discrepancy / total_vector.norm().clamp_min(1e-12))
    torch.testing.assert_close(reconstruction, total_vector, rtol=3e-4, atol=3e-4)
    if relative > 2e-5:
        raise AssertionError(f"gradient sum relative error {relative}")
    # clip_grad_norm_ computes each norm and the global norm in parameter dtype.
    clip_norm = torch.stack([g.norm() for g in total_grads if g is not None]).norm()
    clip_factor = min(1.0, float(clip / (clip_norm + 1e-6)))
    groups = {
        group: [i for i, name in enumerate(names) if branch(name) == group] for group in GROUPS
    }
    groups = {name: indices for name, indices in groups.items() if indices}
    reports = {}
    for group, indices in groups.items():
        vectors = {name: vector(grads, parameters, indices) for name, grads in term_grads.items()}
        reports[group] = {
            "parameter_elements": sum(parameters[i].numel() for i in indices),
            "parameter_names": [names[i] for i in indices],
            "total_grad_norm": float(vector(total_grads, parameters, indices).norm()),
            "terms": {
                name: {
                    "grad_norm": float(vectors[name].norm())
                    if any(term_grads[name][i] is not None for i in indices)
                    else None,
                    "same_total_clip_grad_norm": float(vectors[name].norm()) * clip_factor
                    if any(term_grads[name][i] is not None for i in indices)
                    else None,
                    "connected_tensors": sum(term_grads[name][i] is not None for i in indices),
                    "exact_zero_connected_tensors": sum(
                        term_grads[name][i] is not None and not bool(term_grads[name][i].any())
                        for i in indices
                    ),
                    "nonzero_elements": int(torch.count_nonzero(vectors[name])),
                }
                for name in weighted
            },
            "cosines": {
                f"{left}|{right}": float(F.cosine_similarity(vectors[left], vectors[right], dim=0))
                if vectors[left].norm() > 0 and vectors[right].norm() > 0
                else None
                for left, right in itertools.combinations(weighted, 2)
            },
        }
    return (
        {
            "total_global_norm": float(clip_norm),
            "total_clip_factor": clip_factor,
            "gradient_sum_relative_error": relative,
            "groups": reports,
        },
        term_grads,
        total_grads,
    )


def clone_with_optimizer(payload):
    model = build_encoder(N02Config.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(payload["run"]["lr"]))
    # load_state_dict may share tensor storage; a deep copy is mandatory here.
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    return model, optimizer


def disposable_update(payload, gradients, *, clip):
    model, optimizer = clone_with_optimizer(payload)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float("inf") if clip is None else clip, error_if_nonfinite=True
    )
    optimizer.step()
    updates = {group: [] for group in GROUPS}
    for name, parameter in model.named_parameters():
        updates[branch(name)].append((parameter.detach() - before[name]).double().flatten())
    return model, {
        "unclipped_global_norm": float(norm),
        "clip_factor": 1.0 if clip is None else min(1.0, float(clip / (norm + 1e-6))),
        "branch_parameter_update_norm": {
            group: float(torch.cat(values).norm()) for group, values in updates.items() if values
        },
    }


def audit_checkpoint(checkpoint, batches):
    file_before = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload_before = tree_digest(payload)
    model, _ = clone_with_optimizer(payload)
    model_before = tree_digest(model.state_dict())
    clip = float(payload["run"]["grad_clip"])
    margin = float(payload["run"]["diversity_margin"])
    rows = []
    for rng_seed, batch in batches.items():
        batch_before = batch_digest(batch)
        model.train()
        total, weighted, row = training_objective(model, batch, diversity_margin=margin)
        row.update(
            checkpoint=str(checkpoint),
            seed=int(payload["seed"]),
            step=int(payload["step"]),
            rng_seed=rng_seed,
            input_sha256=batch_before,
        )
        row["gradients"], term_grads, total_grads = decompose_gradients(
            weighted, total, list(model.named_parameters()), clip=clip
        )
        row["updates"] = {}
        for arm in ARMS:
            gradients = total_grads
            if arm == "zero_current_gradient":
                gradients = tuple(
                    None if grad is None else torch.zeros_like(grad) for grad in total_grads
                )
            if arm.startswith("omit_"):
                omitted = arm.removeprefix("omit_")
                gradients = tuple(
                    sum(parts[1:], parts[0]) if parts else None
                    for i in range(len(total_grads))
                    for parts in [
                        [
                            grads[i]
                            for name, grads in term_grads.items()
                            if name != omitted and grads[i] is not None
                        ]
                    ]
                )
            updated, update = disposable_update(
                payload, gradients, clip=None if arm == "full_no_clip" else clip
            )
            with torch.no_grad():
                _, _, after = training_objective(updated, batch, diversity_margin=margin)
            update["functional_after"] = after["functional"]
            update["functional_delta"] = {
                metric: None
                if value is None or row["functional"][metric] is None
                else value - row["functional"][metric]
                for metric, value in after["functional"].items()
            }
            row["updates"][arm] = update
        assert tree_digest(model.state_dict()) == model_before, "original model changed"
        assert batch_digest(batch) == batch_before, "input batch changed"
        rows.append(row)
        print(
            f"[n02-gradients] seed={payload['seed']} step={payload['step']} rng={rng_seed}",
            flush=True,
        )
    assert tree_digest(payload) == payload_before, "checkpoint payload/optimizer state changed"
    assert sha256(checkpoint) == file_before, "original checkpoint bytes changed"
    return rows


def summarize(rows):
    result = {}
    for seed, step in sorted({(row["seed"], row["step"]) for row in rows}):
        selected = [row for row in rows if row["seed"] == seed and row["step"] == step]
        result[f"seed{seed}_step{step}"] = {
            "batches": len(selected),
            "mean_clip_factor": float(
                np.mean([row["gradients"]["total_clip_factor"] for row in selected])
            ),
            "weighted_losses": {
                name: float(
                    np.mean([row["weighted"][name] for row in selected if name in row["weighted"]])
                )
                for name in TERMS
                if any(name in row["weighted"] for row in selected)
            },
            "functional": {
                key: float(
                    np.mean(
                        [
                            row["functional"][key]
                            for row in selected
                            if row["functional"][key] is not None
                        ]
                    )
                )
                for key in selected[0]["functional"]
                if any(row["functional"][key] is not None for row in selected)
            },
            "mean_functional_deltas": {
                arm: {
                    key: float(
                        np.mean([row["updates"][arm]["functional_delta"][key] for row in selected])
                    )
                    for key in ("geometry_mse", "contrast_mse")
                }
                for arm in ARMS
            },
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs/n02-contracts-20261013")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/n02-isolation-20261013/gradients")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--rng-seeds", type=int, nargs="+", default=[900, 901, 902, 903])
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoints = [
        args.checkpoint_root / f"seed{seed}" / f"checkpoint{step}.pt"
        for seed in args.seeds
        for step in args.steps
    ]
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = N02Config.from_dict(first["config"])
    batches = {
        rng_seed: synthetic_batch(
            config,
            np.random.default_rng(rng_seed),
            batch_size=2,
            num_frames=8,
            frame_size=64,
            seed=900,
            split="train",
        )
        for rng_seed in args.rng_seeds
    }
    args.output.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path("src/owac/frameworks/n02").glob("*.py")) + [
        Path("src/owac/training/n02.py"),
        Path(__file__),
    ]
    manifest = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path): sha256(path) for path in sources},
        "config": config.as_dict(),
        "config_sha256": tree_digest(config.as_dict()),
        "checkpoint_sha256": {str(path): sha256(path) for path in checkpoints},
        "input_sha256": {str(seed): batch_digest(batch) for seed, batch in batches.items()},
        "generator_seed": 900,
        "split": "train",
        "batch_size": 2,
        "sequence_length": 8,
        "frame_size": 64,
        "torch_version": torch.__version__,
        "threads": 1,
        "boundary": (
            "Local same-batch disposable AdamW steps, not training continuation or generalization."
        ),
        "optimizer_groups": {
            str(path): [
                {key: value for key, value in group.items() if key != "params"}
                for group in torch.load(path, map_location="cpu", weights_only=False)["optimizer"][
                    "param_groups"
                ]
            ]
            for path in checkpoints
        },
    }
    protocol = {
        "checkpoints": len(checkpoints),
        "batches_per_checkpoint": len(batches),
        "arms": list(ARMS),
        "disposable_optimizer_steps": len(checkpoints) * len(batches) * len(ARMS),
        "generator_seed": 900,
        "rng_seeds": args.rng_seeds,
        "split": "train",
        "batch_size": 2,
        "sequence_length": 8,
        "frame_size": 64,
        "loss_aggregation": (
            "Production step_terms and counterfactual loss; active-frame means; "
            "original weights; frame state detach."
        ),
        "gradient_clip": (
            "One global factor after full weighted total; "
            "component effective norms use that same factor."
        ),
        "inactive_terms": (
            "Absent from gradients; None means disconnected; "
            "connected exact zero recorded separately."
        ),
        "zero_gradient_control": (
            "Connected gradients zeroed, None remains None; "
            "retained AdamW momentum and weight decay."
        ),
        "functional_contrast": (
            "Predicted odd-minus-even class mean versus pooled physical template1-minus-template0."
        ),
        "semantic_predicates": (
            "Detached hard predicates intentionally have no derivative; "
            "location straight-through gradient is a surrogate."
        ),
        "decision_scope": (
            "Local gradient direction and disposable same-batch response only; "
            "no continuation or model selection."
        ),
        "frozen_before_formal": True,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (args.output / "records.jsonl").open("w") as handle:
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["config"] == first["config"], "checkpoint configs differ"
            for row in audit_checkpoint(checkpoint, batches):
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                rows.append(row)
    summary = {
        "checkpoints": summarize(rows),
        "wall_seconds": time.perf_counter() - started,
        "original_files_models_optimizer_and_inputs_unchanged": True,
        "gradient_sum_max_relative_error": max(
            row["gradients"]["gradient_sum_relative_error"] for row in rows
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

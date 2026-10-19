"""Frozen-field, fixed-geometry diagnostics of the production N02 measurement head.

This is deliberately not end-to-end N02 training. Targets never enter the head
or the causal field cache; fixed slot geometry and exact-template references
are diagnostic interventions, not deployable model capabilities.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from owac.core.encoder_contract_n02 import ComputeBudget, MeasurementFrame
from owac.evaluation.n02_bottlenecks import paired_episodes
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.measurement import gaussian_nll
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import values_for_task
from owac.training.n02 import _frame, episode_prior_context, synthetic_batch


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    identity = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(identity + value.numpy().tobytes()).hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def declared_geometry(_module, _inputs, output):
    """Production head applies tanh afterwards: tanh(+-20) is exactly +-1 in fp32."""
    result = output.clone()
    parity = torch.arange(output.shape[1], device=output.device) % 2
    result[..., 0] = (parity * 2 - 1).to(output) * 20
    return result


class FixedGeometryHead(nn.Module):
    """Copy a production head; only detached fields are accepted as input."""

    def __init__(self, source):
        super().__init__()
        if source.head_kind != "geometry_coupled":
            raise ValueError("this diagnostic requires the coupled production head")
        self.head = copy.deepcopy(source)
        self.head.geometry.register_forward_hook(declared_geometry)
        self.head.geometry.requires_grad_(False)
        if self.head.log_variance is not None:
            self.head.log_variance.requires_grad_(False)

    def forward(self, fields):
        if fields.requires_grad:
            raise ValueError("encoder fields must be detached for this isolation")
        return self.head(fields)[0]


def class_predictions(mean):
    return torch.stack([mean[:, ::2].mean(1), mean[:, 1::2].mean(1)], 1)


def contrast_of(pairs):
    return (pairs[:, 1] - pairs[:, 0]) / 2


def head_objective(head, cache, indexes, objective, signal_scale):
    predicted = class_predictions(head(cache["fields"][indexes]))
    target = cache["templates"][indexes]
    if objective == "paired_nll":
        variance = cache["total_variance"][indexes, None]
        return gaussian_nll(predicted - target, variance.log(), robust="gaussian").mean()
    if objective == "contrast_mse":
        # Every pixel contributes. This fixed scale changes objective weighting;
        # it is not an equal-loss-scale comparison with paired Gaussian NLL.
        return (contrast_of(predicted) - contrast_of(target)).square().mean() / signal_scale
    raise ValueError(objective)


@torch.no_grad()
def frame_cache(model, frame, context, state, templates, identities, raw_quality):
    """Only frame/context/state enter the production encoder's causal solve."""
    output = model.step(
        frame, context, state, ComputeBudget(model.config.cycles, model.config.history)
    )
    state = model.detach_state(output.state)
    fields = output.representation.h[:, 0].detach().clone()
    size = fields.shape[-2:]
    batch, _, channels = templates.shape[:3]
    pooled_templates = F.adaptive_avg_pool2d(
        templates.reshape(batch * 2, channels, *templates.shape[-2:]), size
    ).reshape(batch, 2, channels, *size)
    observed = F.adaptive_avg_pool2d(frame.values, size)
    buffered = state.diagnostics["_buffer"].frames[-1]
    torch.testing.assert_close(observed, buffered.values, rtol=0, atol=0)
    _, log_variance, _ = model.measurement.head(fields)
    model_variance = log_variance.exp()[:, 0].detach().clone()
    sensor_variance = buffered.variance.detach().clone()
    sensor_variance = sensor_variance.expand_as(observed).clone()
    prior = context.hypothesis_prior
    if prior is None:
        prior = fields.new_ones(batch, model.config.K)
    result = {
        "fields": fields,
        "observed": observed,
        "templates": pooled_templates,
        "model_variance": model_variance,
        "sensor_variance": sensor_variance,
        "total_variance": model_variance + sensor_variance,
        "raw_quality": torch.as_tensor(raw_quality, dtype=torch.bool),
        "identities": identities,
        "raw_sha256": [tensor_digest(raw) for raw in frame.values],
        "prior_sha256": [tensor_digest(value) for value in prior],
        "full_template_sha256": [tensor_digest(value) for value in templates],
    }
    return result, state


def concatenate_cache(pieces):
    return {
        key: torch.cat([piece[key] for piece in pieces])
        if isinstance(pieces[0][key], torch.Tensor)
        else [item for piece in pieces for item in piece[key]]
        for key in pieces[0]
    }


@torch.no_grad()
def build_train_cache(model, *, batches=8):
    pieces = []
    for index in range(batches):
        rng_seed = 910 + index
        batch = synthetic_batch(
            model.config,
            np.random.default_rng(rng_seed),
            batch_size=2,
            num_frames=8,
            frame_size=64,
            seed=910,
            split="train",
        )
        state = None
        for frame_index in range(8):
            identities = [
                {
                    "split": "train",
                    "generator_seed": 910,
                    "rng_seed": rng_seed,
                    "batch": index,
                    "element": element,
                    "frame": frame_index,
                    "truth": int(batch.anchor[frame_index, element]),
                    "prior_condition": batch.notes["prior_conditions"][element],
                }
                for element in range(2)
            ]
            piece, state = frame_cache(
                model,
                _frame(batch, frame_index),
                batch.contexts[frame_index],
                state,
                batch.counterfactual_templates[frame_index],
                identities,
                batch.identifiable_now[frame_index],
            )
            pieces.append(piece)
    return concatenate_cache(pieces)


@torch.no_grad()
def build_evaluation_cache(model, *, seeds=range(720, 736)):
    pieces = []
    for seed in seeds:
        generator, episodes = paired_episodes(seed)
        base = episodes["missing"]
        templates = np.stack(
            [generator.measurement_templates(base, frame_index) for frame_index in range(8)]
        )
        worlds = []
        for truth, value in enumerate(values_for_task(base.task)):
            episode = replace(
                base,
                frames=(templates[:, truth] + base.noise).astype(np.float32),
                value=value,
                value_over_time=(value,) * 8,
            )
            worlds.append((str(truth), episode, templates))
        no_cue = episodes["unidentifiable"]
        worlds.append(
            (
                "no_cue",
                no_cue,
                np.stack([generator.measurement_templates(no_cue, index) for index in range(8)]),
            )
        )
        for world, episode, world_templates in worlds:
            state = None
            quality = generator.measurement_quality(episode)["identifiable_now"]
            context = episode_prior_context([episode], model.config.K)
            for index, raw in enumerate(episode.frames):
                frame = MeasurementFrame(
                    values=torch.from_numpy(raw).unsqueeze(0),
                    frame_ids=(index,),
                    episode_ids=(episode.episode_id,),
                    metadata={"variance": generator.options.noise_sigma**2},
                    metadata_available={"variance": True},
                    camera_id="synthetic_measurement",
                )
                identity = {
                    "split": "validation",
                    "seed": seed,
                    "world": world,
                    "frame": index,
                    "episode_id": episode.episode_id,
                }
                piece, state = frame_cache(
                    model,
                    frame,
                    context,
                    state,
                    torch.from_numpy(world_templates[index]).unsqueeze(0),
                    [identity],
                    [quality[index]],
                )
                pieces.append(piece)
    return concatenate_cache(pieces)


def contrast_metrics(predicted, target, signal_scale):
    """Physical contrast metrics; frame strata are never supplied to the loss."""
    estimate, truth = contrast_of(predicted).double(), contrast_of(target).double()
    signal = truth.square().flatten(1).sum(1)
    cue = signal > 1e-20
    no_cue = ~cue
    actual = estimate[cue].flatten()
    expected = truth[cue].flatten()
    denominator = float(expected.square().sum())
    numerator = float((actual * expected).sum())
    cosine_denominator = math.sqrt(float(actual.square().sum()) * denominator)
    normal_error = float((actual - expected).square().sum()) / max(denominator, 1e-30)
    proper = (predicted - target).square().flatten(1).mean(1)
    swapped = (predicted - target.flip(1)).square().flatten(1).mean(1)
    return {
        "cue_frames": int(cue.sum()),
        "no_cue_frames": int(no_cue.sum()),
        "projection_gain": numerator / max(denominator, 1e-30),
        "cosine": numerator / max(cosine_denominator, 1e-30),
        "contrast_nmse": normal_error,
        "proper_vs_swapped_fraction": float((proper[cue] < swapped[cue]).float().mean())
        if cue.any()
        else None,
        "no_cue_false_contrast_ratio": float(estimate[no_cue].square().mean()) / signal_scale
        if no_cue.any()
        else None,
        "template_rmse": float((predicted - target).square().mean().sqrt()),
    }


def pair_choices(predicted, cache):
    residual = predicted.double() - cache["observed"].double()[:, None]
    scores = (0.5 * residual.square() / cache["total_variance"].double()[:, None]).sum((2, 3, 4))
    pairs = {}
    for index, identity in enumerate(cache["identities"]):
        if identity.get("world") not in {"0", "1"}:
            continue
        seed, world = identity["seed"], identity["world"]
        pairs.setdefault(seed, {}).setdefault(world, 0.0)
        pairs[seed][world] += float(scores[index, 1] - scores[index, 0])
    rows = [
        {
            "seed": seed,
            "score1_minus_score0_when0": worlds["0"],
            "score1_minus_score0_when1": worlds["1"],
            "both_correct": worlds["0"] > 0 and worlds["1"] < 0,
        }
        for seed, worlds in sorted(pairs.items())
    ]
    return {
        "both_correct": sum(row["both_correct"] for row in rows),
        "total": len(rows),
        "records": rows,
    }


@torch.no_grad()
def assess(head, cache, signal_scale, *, exact=False):
    if exact:
        predicted = cache["templates"]
    else:
        predicted = torch.cat(
            [class_predictions(head(chunk)) for chunk in cache["fields"].split(16)]
        )
    result = contrast_metrics(predicted, cache["templates"], signal_scale)
    result["actual_head_choices"] = pair_choices(predicted, cache)
    common = cache["templates"].mean(1)
    contrast = contrast_of(predicted)
    oracle_common = torch.stack([common - contrast, common + contrast], 1)
    result["oracle_common_choices"] = pair_choices(oracle_common, cache)
    return result


def cache_manifest(cache):
    difference = cache["templates"][:, 1] - cache["templates"][:, 0]
    pooled_kl = (0.5 * difference.double().square() / cache["sensor_variance"].double()).sum(
        (1, 2, 3)
    )
    raw_identifiable = cache["raw_quality"]
    pooled_identifiable = pooled_kl >= math.log(19)
    return {
        "count": len(cache["fields"]),
        "tensor_sha256": {
            key: tensor_digest(value)
            for key, value in cache.items()
            if isinstance(value, torch.Tensor)
        },
        "identities": cache["identities"],
        "raw_sha256": cache["raw_sha256"],
        "prior_sha256": cache["prior_sha256"],
        "full_template_sha256": cache["full_template_sha256"],
        "pooled_oracle_kl": pooled_kl.tolist(),
        "raw_identifiable_count": int(raw_identifiable.sum()),
        "pooled_identifiable_count": int(pooled_identifiable.sum()),
        "raw_identifiable_but_pooled_not": int((raw_identifiable & ~pooled_identifiable).sum()),
        "pooled_identifiable_but_raw_not": int((~raw_identifiable & pooled_identifiable).sum()),
        "sensor_variance_min_max": [
            float(cache["sensor_variance"].min()),
            float(cache["sensor_variance"].max()),
        ],
    }


def evidence_criteria(metrics, *, expected_pairs=16):
    return {
        "contrast_nmse_le_0_5": metrics["contrast_nmse"] <= 0.5,
        "cosine_ge_0_5": metrics["cosine"] >= 0.5,
        "both_ge_12_of_16": metrics["actual_head_choices"]["both_correct"] >= 12
        and metrics["actual_head_choices"]["total"] == expected_pairs,
        "no_cue_false_contrast_le_0_1": metrics["no_cue_false_contrast_ratio"] is not None
        and metrics["no_cue_false_contrast_ratio"] <= 0.1,
    }


def optimize_head(source, cache, *, objective, schedule, signal_scale):
    head = FixedGeometryHead(source).train()
    params = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    protected_before = {
        name: tensor_digest(parameter)
        for name, parameter in head.named_parameters()
        if not parameter.requires_grad
    }
    logs = []
    for step, batch_frame in enumerate(schedule, 1):
        indexes = torch.tensor([2 * int(batch_frame), 2 * int(batch_frame) + 1])
        optimizer.zero_grad(set_to_none=True)
        loss = head_objective(head, cache, indexes, objective, signal_scale)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f"{objective} step {step}: nonfinite loss or gradient")
        optimizer.step()
        logs.append(
            {
                "step": step,
                "cache_batch_frame": int(batch_frame),
                "loss": float(loss.detach()),
                "unclipped_gradient_norm": float(grad_norm),
            }
        )
    protected_after = {
        name: tensor_digest(parameter)
        for name, parameter in head.named_parameters()
        if not parameter.requires_grad
    }
    if protected_after != protected_before:
        raise AssertionError("frozen geometry/variance parameters changed")
    return head.eval(), logs


def protocol(*, updates, seeds, train_batches, evaluation_seeds, smoke):
    return {
        "diagnostic_only": True,
        "smoke": smoke,
        "checkpoint_steps": 200,
        "checkpoint_seeds": seeds,
        "fit_generator_seed": 910,
        "fit_rng_seeds": list(range(910, 910 + train_batches)),
        "fit_split": "train",
        "fit_batch": 2,
        "fit_frames": 8,
        "frame_size": 64,
        "evaluation_split": "validation",
        "evaluation_seeds": evaluation_seeds,
        "geometry": "production head pre-tanh geometry0 overridden to parity +/-20",
        "variance": "cached original predicted variance VALUES + actual buffered sensor variance",
        "objectives": ["paired_nll", "contrast_mse"],
        "contrast_normalization": "fixed all-frame all-pixel train contrast mean-square",
        "contrast_objective_changes_weighting": True,
        "contrast_objective_scope": (
            "removes common-fit loss and inverse-variance spatial weights; its improvement "
            "cannot uniquely distinguish sparse signal dilution from shared-trunk interference"
        ),
        "no_spatial_cue_mask_in_losses": True,
        "updates_per_arm": updates,
        "sampler_seed": 910,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 0.01,
        "clip_gradient_norm": 1.0,
        "threads": 1,
        "precision": "float32 training, float64 contrast reductions/scoring",
        "criteria": {
            "eval_contrast_nmse_max": 0.5,
            "eval_cosine_min": 0.5,
            "eval_actual_common_pair_both_min": 12,
            "eval_pair_denominator": 16,
            "eval_no_cue_false_contrast_ratio_max": 0.1,
        },
        "reference": "exact templates and oracle common are nondeployable controls",
        "scope": "fixed-field head learnability; failure cannot prove architecture incapacity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs/n02-contracts-20261013")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(910)
    updates = 2 if args.smoke else 200
    seeds = [0] if args.smoke else [0, 1, 2]
    train_batches = 1 if args.smoke else 8
    evaluation_seeds = [719] if args.smoke else list(range(720, 736))
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = protocol(
        updates=updates,
        seeds=seeds,
        train_batches=train_batches,
        evaluation_seeds=evaluation_seeds,
        smoke=args.smoke,
    )
    # Written before cache generation/optimization; never edited using results.
    write_json(args.output / "protocol.json", declaration)
    dependencies = (
        sorted(Path("src/owac/frameworks/n02").glob("*.py"))
        + sorted(Path("src/owac/core").glob("*.py"))
        + [
            Path(__file__),
            Path("src/owac/training/n02.py"),
            Path("src/owac/evaluation/n02_bottlenecks.py"),
        ]
    )
    source_before = {str(path): file_digest(path) for path in dependencies}
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_before,
        "protocol_sha256": file_digest(args.output / "protocol.json"),
        "seeds": {},
    }
    write_json(args.output / "manifest.json", manifest)
    reports = {}
    started = time.perf_counter()
    schedule = np.random.default_rng(910).integers(0, train_batches * 8, size=updates)
    write_json(args.output / "schedule.json", schedule.tolist())
    for seed in seeds:
        checkpoint = args.checkpoint_root / f"seed{seed}" / "checkpoint200.pt"
        checkpoint_before = file_digest(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = N02Encoder(N02Config.from_dict(payload["config"])).eval()
        model.load_state_dict(payload["model"])
        encoder_before = state_digest(model)
        train = build_train_cache(model, batches=train_batches)
        evaluation = build_evaluation_cache(model, seeds=evaluation_seeds)
        if set(train["raw_sha256"]) & set(evaluation["raw_sha256"]):
            raise AssertionError("fit and held-out measurement overlap")
        torch.save({"train": train, "evaluation": evaluation}, args.output / f"cache_seed{seed}.pt")
        cache_before = {
            split: {
                key: tensor_digest(value) for key, value in cache.items() if torch.is_tensor(value)
            }
            for split, cache in (("train", train), ("evaluation", evaluation))
        }
        signal_scale = float(contrast_of(train["templates"]).double().square().mean())
        if signal_scale <= 0:
            raise AssertionError("training cache has no physical contrast signal")
        source = model.measurement.head
        fixed = FixedGeometryHead(source).eval()
        report = {
            "signal_scale": signal_scale,
            "step0": {
                "train": assess(fixed, train, signal_scale),
                "evaluation": assess(fixed, evaluation, signal_scale),
            },
            "exact_templates": {
                "train": assess(None, train, signal_scale, exact=True),
                "evaluation": assess(None, evaluation, signal_scale, exact=True),
            },
            "arms": {},
        }
        for objective in declaration["objectives"]:
            fitted, logs = optimize_head(
                source, train, objective=objective, schedule=schedule, signal_scale=signal_scale
            )
            assessment = {
                "train": assess(fitted, train, signal_scale),
                "evaluation": assess(fitted, evaluation, signal_scale),
            }
            assessment["criteria"] = evidence_criteria(assessment["evaluation"])
            assessment["all_criteria_pass"] = all(assessment["criteria"].values())
            report["arms"][objective] = assessment
            torch.save(fitted.state_dict(), args.output / f"head_seed{seed}_{objective}.pt")
            write_json(args.output / f"optimization_seed{seed}_{objective}.json", logs)
            print(
                json.dumps(
                    {"seed": seed, "objective": objective, "eval": assessment["evaluation"]}
                ),
                flush=True,
            )
        cache_after = {
            split: {
                key: tensor_digest(value) for key, value in cache.items() if torch.is_tensor(value)
            }
            for split, cache in (("train", train), ("evaluation", evaluation))
        }
        protected = {
            "checkpoint_unchanged": checkpoint_before == file_digest(checkpoint),
            "encoder_unchanged": encoder_before == state_digest(model),
            "all_cache_tensors_unchanged": cache_before == cache_after,
        }
        if not all(protected.values()):
            raise AssertionError(protected)
        manifest["seeds"][str(seed)] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_before,
            "encoder_state_sha256": encoder_before,
            "cache_file_sha256": file_digest(args.output / f"cache_seed{seed}.pt"),
            "fit": cache_manifest(train),
            "evaluation": cache_manifest(evaluation),
            "protected": protected,
        }
        reports[str(seed)] = report
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "results.json", reports)
    source_after = {str(path): file_digest(path) for path in dependencies}
    if source_after != source_before:
        raise AssertionError("production or diagnostic source changed during execution")
    write_json(
        args.output / "summary.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "head_only_updates": len(seeds) * 2 * updates,
            "source_unchanged": True,
            "protocol_sha256": manifest["protocol_sha256"],
            "results": reports,
        },
    )


if __name__ == "__main__":
    main()

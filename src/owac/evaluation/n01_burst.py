"""Matched full-size DBSR feature baselines before N01 support-graph experiments.

All arms share RAW data, head and backbone initialization. Extra residual blocks
provide an active capacity control for the attention predictor. The labelled
exact-alignment fixture remains separate from the main score. Training encodes
independent frames once within each optimizer step; no learned cache survives it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import resource
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from owac.baselines.burst_features import BurstFeatureConfig, BurstFeatureEncoder
from owac.training.n01 import LocationProbeHead

from .n01_content import ORACLE, make_protocol_batch, summarize_sequence
from .n01_content_data import CONDITIONS

METHODS = ("current", "mean", "mean_capacity", "dbsr_reference", "dbsr_consensus")


class BurstProbe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BurstFeatureEncoder(config)
        self.head = LocationProbeHead(config.feature_width)


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_models(protocol, seed):
    torch.manual_seed(seed)
    config = BurstFeatureConfig(**protocol["encoder"])
    canonical = BurstProbe(config)
    base_count = sum(p.numel() for p in canonical.encoder.backbone.parameters())
    target = sum(p.numel() for p in canonical.encoder.parameters())
    # Each additional upstream ResBlock has two biased 3x3 convolutions.
    block_count = sum(
        p.numel() for p in canonical.encoder.backbone.upstream.res_layers[0].parameters()
    )
    extra = max(0, round((target - base_count) / block_count))
    models = {}
    for method in protocol["methods"]:
        conf = replace(
            config,
            fusion="mean" if method == "mean_capacity" else method,
            residual_blocks=config.residual_blocks + (extra if method == "mean_capacity" else 0),
        )
        torch.manual_seed(seed)
        model = BurstProbe(conf)
        source = dict(canonical.named_parameters())
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in source and source[name].shape == parameter.shape:
                    parameter.copy_(source[name])
        models[method] = model
    return models


def oracle_offsets(batch, now, start, *, device):
    if batch.condition != ORACLE:
        return None
    # Fixture-only observed displacement, never target coordinates or future rows.
    shift = batch.translations[:, start : now + 1].flip(1) - batch.translations[:, now : now + 1]
    xy = shift.flip(-1).to(device)
    return xy[:, :, :, None, None].expand(-1, -1, -1, *batch.values.shape[-2:])


def run_sequence(model, batch, protocol, *, training=False):
    device = next(model.parameters()).device
    values, valid = batch.values.to(device), batch.valid.to(device)
    b, t, _, h, w = values.shape
    features, centers = model.encoder.encode_frames(values, valid)
    errors, losses, histories, masses, effective = [], [], [], [], []
    for now in range(t):
        start = max(0, now + 1 - model.encoder.config.max_history)
        represented, _, weights = model.encoder.fuse_encoded(
            features[:, start : now + 1].flip(1),
            centers[:, start : now + 1].flip(1),
            oracle_offsets(batch, now, start, device=device),
        )
        logits = model.head(represented)
        prediction = model.head.soft_argmax(logits)
        target = batch.target_xy[:, now].to(device)
        errors.append((prediction - target).norm(dim=1).detach().cpu())
        visible = batch.visible[:, now].to(device)
        if training and now >= protocol["burn_in"] and bool(visible.any()):
            normalized = target / target.new_tensor([h - 1, w - 1])
            heat = model.head.heatmap_target(normalized, (h, w), protocol["heatmap_sigma"])
            loss = -(heat.flatten(1) * logits.flatten(1).log_softmax(1)).sum(1)
            losses.append(loss[visible].mean())
        # Report actual contributing source count, not a learned posterior.
        detached = weights.detach()
        histories.append(float((detached > 0).sum(1).float().mean()))
        masses.append(float(detached[:, 1:].sum(1).mean()))
        supported = detached.sum(1) > 0
        effective.append(
            float((detached.square().sum(1).clamp_min(1e-12).reciprocal() * supported).mean())
        )
    metrics = summarize_sequence(torch.stack(errors, 1), batch.visible, histories, batch, protocol)
    metrics["contributing_feature_source_count_per_frame"] = metrics.pop(
        "selected_unique_frames_per_frame"
    )
    metrics["historical_feature_weight_mass_per_frame"] = masses
    metrics["effective_feature_source_count_per_frame"] = effective
    metrics["source_weight_interpretation"] = (
        "attention over features, not calibrated measurement precision"
    )
    return torch.stack(losses).mean() if losses else None, metrics


def evaluate(model, protocol, seed, split):
    model.eval()
    cases = {}
    with torch.no_grad():
        for index, condition in enumerate(CONDITIONS):
            batch = make_protocol_batch(protocol, seed, split, index, condition)
            _, cases[condition] = run_sequence(model, batch, protocol)
    ordinary = [row for name, row in cases.items() if name != ORACLE]
    return {
        "cases": cases,
        "main_mean_error_px": sum(r["mean_error_px"] for r in ordinary) / len(ordinary),
        "main_early_error_px": sum(r["early_error_px"] for r in ordinary) / len(ordinary),
        "oracle_excluded_from_main_aggregate": True,
    }


def train_method(model, protocol, seed, method, output):
    device = torch.device(protocol.get("device", "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"]
    )
    initial = {
        "backbone": parameter_digest(model.encoder.backbone),
        "head": parameter_digest(model.head),
        "full": parameter_digest(model),
    }
    best = float("inf")
    best_step = 0
    best_state = None
    curves = []
    input_hashes = []
    fixture_hashes = []
    times = []
    active = set()
    started = time.perf_counter()
    for step in range(protocol["steps"]):
        model.train()
        batch = make_protocol_batch(protocol, seed, "train", step)
        input_hashes.append(batch.input_hash())
        fixture_hashes.append(batch.fixture_hash())
        optimizer.zero_grad(set_to_none=True)
        before = time.perf_counter()
        loss, metrics = run_sequence(model, batch, protocol, training=True)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("missing/non-finite loss")
        loss.backward()
        active.update(
            name
            for name, p in model.named_parameters()
            if p.grad is not None and bool((p.grad != 0).any())
        )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - before) * 1000)
        row = {
            "step": step + 1,
            "condition": batch.condition,
            "loss": float(loss.detach()),
            "train_error_px": metrics["mean_error_px"],
            "gradient_norm": float(norm),
        }
        if (step + 1) % protocol["validation_every"] == 0 or step + 1 == protocol["steps"]:
            validation = evaluate(model, protocol, seed, "validation")
            row["validation_mean_error_px"] = validation["main_mean_error_px"]
            row["validation_early_error_px"] = score = validation["main_early_error_px"]
            if score < best:
                best = score
                best_step = step + 1
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model": best_state,
                        "optimizer": optimizer.state_dict(),
                        "step": best_step,
                        "seed": seed,
                        "method": method,
                        "protocol": protocol,
                        "encoder_config": model.encoder.config.to_dict(),
                    },
                    output / f"{seed}-{method}-best.pt",
                )
            print(json.dumps({"seed": seed, "method": method, **row}), flush=True)
        curves.append(row)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": protocol["steps"],
            "seed": seed,
            "method": method,
            "protocol": protocol,
            "encoder_config": model.encoder.config.to_dict(),
        },
        output / f"{seed}-{method}-last.pt",
    )
    model.load_state_dict(best_state, strict=True)
    split = "validation" if protocol.get("development", False) else "test"
    evaluation = evaluate(model, protocol, seed, split)
    return {
        "seed": seed,
        "method": method,
        "encoder_config": model.encoder.config.to_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "active_parameters": sum(p.numel() for n, p in model.named_parameters() if n in active),
        "initial_parameter_hashes": initial,
        "best_parameter_hash": parameter_digest(model),
        "best_step": best_step,
        "curves": curves,
        "train_input_sha256": input_hashes,
        "train_fixture_sha256": fixture_hashes,
        "evaluation_split": split,
        "evaluation": evaluation,
        "train_step_p50_ms": float(torch.tensor(times).quantile(0.5)),
        "train_step_p95_ms": float(torch.tensor(times).quantile(0.95)),
        "total_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rss_scope": "process lifetime high water mark; includes preceding methods",
    }


def source_manifest():
    root = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__),
        Path(__file__).with_name("n01_content.py"),
        Path(__file__).with_name("n01_content_data.py"),
        root / "src/owac/baselines/burst_features.py",
        root / "src/owac/training/n01.py",
    ]
    files += list((root / "src/owac/baselines/_vendor/dbsr").glob("*.py"))
    files += list((root / "src/owac/frameworks/n01").glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run_experiment(protocol, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("experiment output must be empty")
    if protocol["steps"] < 1 or protocol["validation_every"] < 1:
        raise ValueError("positive training and validation intervals required")
    if not protocol["methods"] or len(set(protocol["methods"])) != len(protocol["methods"]):
        raise ValueError("nonempty unique method names required")
    if any(m not in METHODS for m in protocol["methods"]):
        raise ValueError("unknown method")
    if protocol["encoder"].get("fusion", "dbsr_reference") not in {
        "dbsr_reference",
        "dbsr_consensus",
    }:
        raise ValueError("canonical encoder must contain the DBSR predictor for capacity matching")
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    payload = {
        "protocol": protocol,
        "source_manifest": source_manifest(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": protocol.get("device", "cpu"),
        },
        "scope": (
            "Synthetic mechanism development with full-size DBSR features; no RGB or pretrained "
            "weights, no N01 support graph, no camera/VLA quality claim."
        ),
        "training_cache": (
            "Per-frame features shared only within one optimizer step; "
            "independent encoder has no temporal normalization."
        ),
        "runs": [],
    }
    (output / "run.json").write_text(json.dumps(payload, indent=2) + "\n")
    for seed in protocol["seeds"]:
        for method, model in make_models(protocol, seed).items():
            payload["runs"].append(train_method(model, protocol, seed, method, output))
            (output / "metrics.partial.json").write_text(json.dumps(payload, indent=2) + "\n")
    if source_manifest() != payload["source_manifest"]:
        raise RuntimeError("experiment sources changed during execution; keep partial evidence")
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_experiment(json.loads(Path(args.config).read_text()), args.output)


if __name__ == "__main__":
    main()

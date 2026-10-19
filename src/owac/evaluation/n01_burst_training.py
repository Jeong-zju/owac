"""Train the full DBSR baselines to a declared validation plateau, with resume.

This is a synthetic mechanism baseline, not a DBSR RGB reproduction. Independent
frames are encoded once per optimizer step; only arrived sources enter a window.
No new trainable network is introduced. Test data is used only after training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch

from .n01_burst import make_models, parameter_digest, source_manifest
from .n01_content import ORACLE, make_protocol_batch, summarize_sequence
from .n01_content_data import CONDITIONS


def causal_windows(features, centers, start, stop, history):
    """Return independent newest-first windows, zeroing slots before time zero."""
    device = features.device
    now = torch.arange(start, stop, device=device)
    indices = now[:, None] - torch.arange(history, device=device)[None]
    arrived = indices >= 0
    selected = features[:, indices.clamp_min(0)]
    valid = centers[:, indices.clamp_min(0)] & arrived[None, :, :, None, None, None]
    return selected.flatten(0, 1), valid.flatten(0, 1), indices


def sequence(model, batch, protocol, *, training=False):
    """Batch causal references without changing the original per-frame layers."""
    device = next(model.parameters()).device
    raw, valid = batch.values.to(device), batch.valid.to(device)
    features, centers = model.encoder.encode_frames(raw, valid)
    b, t, _, h, w = raw.shape
    errors, source_counts, masses, effective, deviations, losses = [], [], [], [], [], []
    first = protocol["burn_in"] if training else 0
    chunk = protocol["reference_chunk"]
    for start in range(first, t, chunk):
        stop = min(t, start + chunk)
        window, mask, indices = causal_windows(
            features, centers, start, stop, model.encoder.config.max_history
        )
        offsets = None
        if batch.condition == ORACLE:
            translations = batch.translations.to(device)
            shift = translations[:, indices.clamp_min(0)] - translations[:, start:stop, None]
            offsets = shift.flip(-1).flatten(0, 1)[:, :, :, None, None].expand(-1, -1, -1, h, w)
        represented, _, weights = model.encoder.fuse_encoded(window, mask, offsets)
        logits = model.head(represented)
        target = batch.target_xy[:, start:stop].to(device).flatten(0, 1)
        error = (model.head.soft_argmax(logits) - target).norm(dim=1).reshape(b, stop - start)
        errors.append(error.detach())
        visible = batch.visible[:, start:stop].to(device).flatten()
        if training and bool(visible.any()):
            normalized = target / target.new_tensor([h - 1, w - 1])
            heat = model.head.heatmap_target(normalized, (h, w), protocol["heatmap_sigma"])
            loss = -(heat.flatten(1) * logits.flatten(1).log_softmax(1)).sum(1)
            losses.append(loss[visible])
        if not training:
            weight = weights.detach()
            uniform = mask.float() / mask.sum(1, keepdim=True).clamp_min(1)
            source_counts.append((weight > 0).sum(1).float().mean((1, 2, 3)).reshape(b, -1))
            masses.append(weight[:, 1:].sum(1).mean((1, 2, 3)).reshape(b, -1))
            supported = weight.sum(1) > 0
            effective.append(
                (weight.square().sum(1).clamp_min(1e-12).reciprocal() * supported)
                .mean((1, 2, 3))
                .reshape(b, -1)
            )
            deviations.append((weight - uniform).abs().mean((1, 2, 3, 4)).reshape(b, -1))
    loss = torch.cat(losses).mean() if losses else None
    error = torch.cat(errors, 1)
    if training:
        measured = batch.visible[:, first:].to(device)
        return loss, {"mean_error_px": float(error[measured].mean())}
    history = torch.cat(source_counts, 1).mean(0).cpu().tolist()
    metrics = summarize_sequence(error.cpu(), batch.visible, history, batch, protocol)
    metrics["contributing_feature_source_count_per_frame"] = metrics.pop(
        "selected_unique_frames_per_frame"
    )
    metrics["historical_feature_weight_mass_per_frame"] = (
        torch.cat(masses, 1).mean(0).cpu().tolist()
    )
    metrics["effective_feature_source_count_per_frame"] = (
        torch.cat(effective, 1).mean(0).cpu().tolist()
    )
    metrics["mean_absolute_weight_difference_from_uniform_per_frame"] = (
        torch.cat(deviations, 1).mean(0).cpu().tolist()
    )
    return loss, metrics


def evaluate(model, protocol, seed, split):
    """Evaluation sequences are independent of the training stream."""
    model.eval()
    cases = {}
    with torch.no_grad():
        for index, condition in enumerate(CONDITIONS):
            batch = make_protocol_batch(protocol, seed, split, index, condition)
            _, cases[condition] = sequence(model, batch, protocol)
    ordinary = [row for name, row in cases.items() if name != ORACLE]
    return {
        "cases": cases,
        "main_mean_error_px": sum(r["mean_error_px"] for r in ordinary) / len(ordinary),
        "main_early_error_px": sum(r["early_error_px"] for r in ordinary) / len(ordinary),
        "oracle_excluded_from_main_aggregate": True,
    }


def plateau(curves, protocol):
    """Operational stopping rule, never a claim of global optimality."""
    rows = [r for r in curves if "validation_early_error_px" in r]
    width = protocol["plateau_checks"]
    if len(rows) < width + 1 or rows[-1]["step"] < protocol["min_steps"]:
        return False
    if rows[-1]["learning_rate"] > protocol["plateau_learning_rate"]:
        return False
    # A plateau requires the whole confirmation window to use the reduced LR.
    if any(r["learning_rate"] > protocol["plateau_learning_rate"] for r in rows[-width:]):
        return False
    old_best = min(r["validation_early_error_px"] for r in rows[:-width])
    new_best = min(r["validation_early_error_px"] for r in rows)
    tolerance = max(protocol["plateau_absolute_px"], old_best * protocol["plateau_relative"])
    return old_best - new_best <= tolerance


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_sources():
    manifest = source_manifest()
    root = Path(__file__).resolve().parents[3]
    manifest[str(Path(__file__).relative_to(root))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return manifest


def configure_runtime(protocol):
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if protocol["device"].startswith("cuda"):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise ValueError("set CUBLAS_WORKSPACE_CONFIG before launching CUDA")
        torch.cuda.set_per_process_memory_fraction(protocol["gpu_memory_fraction"])
        torch.cuda.reset_peak_memory_stats()


def train(protocol, seed, method, output, *, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = copy.deepcopy(protocol)
    protocol["methods"] = [method]
    protocol["seeds"] = [seed]
    if not 0 < protocol["min_steps"] <= protocol["max_steps"] < 100000:
        raise ValueError("positive ordered training limits below the data seed range required")
    if (
        min(protocol["reference_chunk"], protocol["validation_every"], protocol["plateau_checks"])
        < 1
    ):
        raise ValueError("positive chunk/validation/plateau intervals required")
    configure_runtime(protocol)
    sources = training_sources()
    run_path = output / "run.json"
    last_path, best_path = output / "last.pt", output / "best.pt"
    if (output / "result.json").exists():
        raise FileExistsError("completed training is immutable")
    if run_path.exists() and not resume:
        raise FileExistsError("existing training requires --resume")
    model = make_models(protocol, seed)[method].to(protocol["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=protocol["lr_factor"],
        patience=protocol["lr_patience"],
        threshold=protocol["plateau_absolute_px"],
        threshold_mode="abs",
        min_lr=protocol["minimum_learning_rate"],
    )
    initial = {
        "backbone": parameter_digest(model.encoder.backbone),
        "head": parameter_digest(model.head),
        "full": parameter_digest(model),
    }
    curves, input_hashes, fixture_hashes, times = [], [], [], []
    best, best_step, start_step = float("inf"), 0, 0
    if resume:
        run = json.loads(run_path.read_text())
        if run["protocol"] != protocol or run["sources"] != sources:
            raise ValueError("resume protocol/source mismatch")
        checkpoint = torch.load(last_path, map_location=protocol["device"], weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        curves = checkpoint["curves"]
        input_hashes, fixture_hashes = checkpoint["input_hashes"], checkpoint["fixture_hashes"]
        times = checkpoint["times"]
        best, best_step, start_step = (
            checkpoint["best"],
            checkpoint["best_step"],
            checkpoint["step"],
        )
        initial = checkpoint["initial"]
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if protocol["device"].startswith("cuda"):
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
    else:
        run = {
            "protocol": protocol,
            "sources": sources,
            "initial": initial,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
            "scope": "full-size synthetic RAW feature baseline; no N01 graph or RGB",
        }
        atomic_json(run_path, run)
    started = time.perf_counter()
    stop_reason = "budget_limit"
    for step in range(start_step, protocol["max_steps"]):
        model.train()
        batch = make_protocol_batch(protocol, seed, "train", step)
        input_hashes.append(batch.input_hash())
        fixture_hashes.append(batch.fixture_hash())
        optimizer.zero_grad(set_to_none=True)
        before = time.perf_counter()
        loss, metrics = sequence(model, batch, protocol, training=True)
        if loss is None or not bool(torch.isfinite(loss)):
            raise RuntimeError("missing/nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if protocol["device"].startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - before) * 1000)
        row = {
            "step": step + 1,
            "loss": float(loss.detach()),
            "train_error_px": metrics["mean_error_px"],
            "gradient_norm": float(norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        validate = (step + 1) % protocol["validation_every"] == 0 or step + 1 == protocol[
            "max_steps"
        ]
        if validate:
            validation = evaluate(model, protocol, seed, "validation")
            row["validation_mean_error_px"] = validation["main_mean_error_px"]
            row["validation_early_error_px"] = score = validation["main_early_error_px"]
            if score < best:
                best, best_step = score, step + 1
                atomic_checkpoint(
                    best_path,
                    {
                        "model": model.state_dict(),
                        "encoder_config": model.encoder.config.to_dict(),
                        "step": best_step,
                        "seed": seed,
                        "method": method,
                        "protocol": protocol,
                    },
                )
            scheduler.step(score)
        curves.append(row)
        if validate:
            settled = plateau(curves, protocol)
            snapshot = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "step": step + 1,
                "curves": curves,
                "input_hashes": input_hashes,
                "fixture_hashes": fixture_hashes,
                "times": times,
                "best": best,
                "best_step": best_step,
                "initial": initial,
            }
            atomic_checkpoint(last_path, snapshot)
            progress = {
                "seed": seed,
                "method": method,
                **row,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "best_step": best_step,
                "best_validation_early_error_px": best,
                "validation_plateau": settled,
                "elapsed_seconds_this_invocation": time.perf_counter() - started,
                "curves": curves,
            }
            atomic_json(output / "progress.json", progress)
            print(json.dumps({k: v for k, v in progress.items() if k != "curves"}), flush=True)
            if settled:
                stop_reason = "validation_plateau"
                break
    checkpoint = torch.load(best_path, map_location=protocol["device"], weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    result = {
        **run,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": len(curves),
        "best_step": best_step,
        "stop_reason": stop_reason,
        "operationally_settled": stop_reason == "validation_plateau",
        "best_parameter_digest": parameter_digest(model),
        "curves": curves,
        "train_input_sha256": input_hashes,
        "train_fixture_sha256": fixture_hashes,
        "validation": evaluate(model, protocol, seed, "validation"),
        "test": evaluate(model, protocol, seed, "test")
        if stop_reason == "validation_plateau"
        else None,
        "train_step_p50_ms": float(torch.tensor(times).quantile(0.5)),
        "train_step_p95_ms": float(torch.tensor(times).quantile(0.95)),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
    }
    if training_sources() != sources:
        raise RuntimeError("training sources changed during run")
    atomic_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(
        json.loads(Path(args.config).read_text()),
        args.seed,
        args.method,
        args.output,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

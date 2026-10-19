"""Matched N01 content/history factorial training and transient evaluation.

Run with ``python -m owac.evaluation.n01_content --config ... --output ...``.
Ordinary conditions use no hidden geometry in the encoder. Exact integer
translation is a separately reported oracle-correspondence positive control,
included in the declared training mixture but excluded from model selection and
the main test aggregate. This is a synthetic mechanism probe, not camera/VLA
acceptance or evidence that the architecture generalizes across tasks.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from ..core.encoder_contract import MeasurementFrame
from ..frameworks.n01.config import N01Config
from ..frameworks.n01.model import N01Encoder, calibrate_statistics
from ..frameworks.n01.motion import CausalCorrespondence, MotionField
from .n01_content_data import CFA_OFFSETS, CONDITIONS, ContentBatch, make_batch, split_seed

METHODS = {
    "A": ("gate_only", (8,)),
    "B": ("gate_only", (1, 4, 8)),
    "C": ("integrated", (8,)),
    "D": ("integrated", (1, 4, 8)),
    "E": ("current_control", (8,)),
    "F": ("current_control", (1, 4, 8)),
}
ORACLE = "translation_oracle_alignment"


class ContentProbe(nn.Module):
    """Identical local spatial readout for all arms, no truth/history input."""

    def __init__(self, config):
        super().__init__()
        self.encoder = N01Encoder(config)
        self.head = nn.Sequential(
            nn.Conv2d(config.width, 16, 3, padding=1),
            nn.GroupNorm(1, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )


def base_config(protocol):
    return N01Config.tiny(
        input_mode="bayer_cfa",
        channels=4,
        planes=4,
        width=int(protocol["width"]),
        num_blocks=1,
        aux_measurement_prediction=False,
        readout_tokens=int(protocol["size"]) ** 2,
        readout_width=int(protocol["width"]),
        alignment="none",
        selection="analytic",
        budget_conditioning=False,
        content_mode="integrated",
    )


def make_protocol_batch(protocol, seed, split, index, condition=None):
    condition = condition or CONDITIONS[index % len(CONDITIONS)]
    event = 11 + (index // len(CONDITIONS)) % 4 if split == "train" else 12
    if split == "test":
        event = 13
    return make_batch(
        condition,
        seed=split_seed(seed, split, index),
        batch_size=int(
            protocol["batch_size"]
            if split in {"train", "calibration"}
            else protocol["evaluation_batch_size"]
        ),
        size=int(protocol["size"]),
        length=int(protocol["length"]),
        event_frame=event,
        contrast=float(protocol["contrast"]),
        noise_sigma=float(protocol["noise_sigma"]),
    )


def make_matched_models(protocol, seed):
    """Copy parameter tensors by name; preserve each mode's semantic buffers."""
    torch.manual_seed(seed)
    canonical = ContentProbe(base_config(protocol))
    reference = dict(canonical.named_parameters())
    calibration = make_protocol_batch(protocol, seed, "calibration", 0, "static_weak")
    length = int(protocol["burn_in"])
    batch = SimpleNamespace(
        values=calibration.values[:, :length],
        valid=calibration.valid[:, :length],
        variance=calibration.variance[:, :length],
        timestamps=torch.arange(length).float()[None].expand(calibration.values.shape[0], -1),
        meta={"sample_xy": CFA_OFFSETS},
    )
    calibrate_statistics(canonical.encoder, batch)
    models = {}
    with torch.no_grad():
        for method in protocol["methods"]:
            mode, depths = METHODS[method]
            model = ContentProbe(
                base_config(protocol).replace(content_mode=mode, time_depths=depths)
            )
            for name, parameter in model.named_parameters():
                if name not in reference or parameter.shape != reference[name].shape:
                    raise ValueError(f"unmatched trainable parameter {method}:{name}")
                parameter.copy_(reference[name])
            model.encoder.normalization_scales.copy_(canonical.encoder.normalization_scales)
            model.encoder.calibration_count.copy_(canonical.encoder.calibration_count)
            models[method] = model
    return models, {
        "input_sha256": calibration.input_hash(),
        "seed": calibration.seed,
        "frames_used": length,
        "reference_mode": "integrated",
        "reference_time_depths": [1, 4, 8],
        "normalization_scales": canonical.encoder.normalization_scales.tolist(),
    }


def parameter_hashes(model):
    """Separate common and added parameter receipts, excluding semantic buffers."""
    digests = {"common": hashlib.sha256(), "content": hashlib.sha256()}
    for name, parameter in sorted(model.named_parameters()):
        digest = digests["content" if ".content_update." in name else "common"]
        digest.update(f"{name}:{tuple(parameter.shape)}".encode())
        digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in digests.items()}


@contextlib.contextmanager
def oracle_translation(encoder, batch):
    """Fixture-only exact translation, causally indexed by arrived timestamps.

    Only the dedicated positive control replaces correspondence. It cannot be
    selected by ordinary conditions and no target positions enter this helper.
    The standard alignment operator masks out-of-frame coordinates, no wrapping.
    """
    if batch.condition != ORACLE:
        yield
        return
    previous_method = encoder._estimate_motion
    previous_estimator = encoder.correspondence
    # Integrator.aligned_slots explicitly checks its own config.alignment.
    # Core modules usually share the encoder config, while auxiliary content
    # integrators may own a replacement config, so preserve each distinct one.
    configs = {
        id(encoder.config): encoder.config,
        id(encoder.integrator.config): encoder.integrator.config,
    }
    if encoder.content_integrator is not None:
        configs[id(encoder.content_integrator.config)] = encoder.content_integrator.config
    previous_alignment = {identity: config.alignment for identity, config in configs.items()}
    for config in configs.values():
        config.alignment = "fixed"
    encoder.correspondence = CausalCorrespondence(encoder.config.replace(alignment="fixed"))

    def exact_motion(self, previous, frame):
        del self
        now = torch.as_tensor(frame.timestamp).long()
        member = torch.arange(previous.batch_size)
        current = batch.translations[member, now]
        fields = [
            MotionField(
                shift=torch.zeros_like(current),
                residual=torch.zeros(previous.batch_size),
                overlap=torch.ones(previous.batch_size),
                valid=torch.ones(previous.batch_size, dtype=torch.bool),
            )
        ]
        for slot in range(previous.num_slots - 1):
            old_time = previous.timestamps[slot].long().clamp_min(0)
            if bool((old_time > now).any()):
                raise ValueError("oracle fixture attempted to use future geometry")
            fields.append(
                MotionField(
                    shift=batch.translations[member, old_time] - current,
                    residual=torch.zeros(previous.batch_size),
                    overlap=torch.ones(previous.batch_size),
                    valid=previous.observed[slot],
                )
            )
        return fields

    encoder._estimate_motion = MethodType(exact_motion, encoder)
    try:
        yield
    finally:
        encoder._estimate_motion = previous_method
        encoder.correspondence = previous_estimator
        for identity, config in configs.items():
            config.alignment = previous_alignment[identity]


def recovery_delays(errors, visible, *, event, tolerance, consecutive):
    """First run of good *consecutive visible frames*, per episode; None=censored."""
    results = []
    for member in range(errors.shape[0]):
        found = None
        for start in range(event, errors.shape[1] - consecutive + 1):
            window = slice(start, start + consecutive)
            if bool(visible[member, window].all()) and bool(
                (errors[member, window] <= tolerance).all()
            ):
                found = start - event
                break
        results.append(found)
    return results


def summarize_sequence(errors, visible, history, batch, protocol):
    burn = int(protocol["burn_in"])
    event = batch.event_frame if batch.condition not in {"static_weak", ORACLE} else burn
    event_end = min(event + int(protocol["event_window"]), errors.shape[1])
    measured = visible.clone()
    measured[:, :burn] = False
    valid_errors = errors[measured]
    transient = errors[:, event:event_end]
    transient_valid = visible[:, event:event_end]
    sums = (transient * transient_valid).sum(1)
    counts = transient_valid.sum(1).clamp_min(1)
    recovery = None
    if batch.condition in {"occlusion_release", "local_change_stale_history"}:
        delays = recovery_delays(
            errors,
            visible,
            event=event,
            tolerance=float(protocol["recovery_tolerance_px"]),
            consecutive=int(protocol["recovery_consecutive"]),
        )
        recovered = [value for value in delays if value is not None]
        recovery = {
            "delay_frames_per_member": delays,
            "censored_count": len(delays) - len(recovered),
            "denominator": len(delays),
            "censored_fraction": (len(delays) - len(recovered)) / len(delays),
            "mean_delay_recovered_only": sum(recovered) / len(recovered) if recovered else None,
            "capped_delay_score": sum(
                value if value is not None else errors.shape[1] - event for value in delays
            )
            / len(delays),
            "censor_at_frames": errors.shape[1] - event,
            "convention": (
                "onset requires three-frame confirmation; no confirmed onset gets T-event "
                "penalty, including onsets too late to confirm; not survival-analysis RMST"
            ),
        }
    return {
        "condition": batch.condition,
        "oracle_alignment": batch.condition == ORACLE,
        "input_sha256": batch.input_hash(),
        "fixture_sha256": batch.fixture_hash(),
        "data_seed": batch.seed,
        "error_unit": "packed CFA grid cells (one cell = two physical sensor pixels)",
        "event_frame": event,
        "mean_error_px": float(valid_errors.mean()),
        "mean_error_sensor_px": float(valid_errors.mean() * 2),
        "early_error_px": float((sums / counts).mean()),
        "early_error_per_member": (sums / counts).tolist(),
        "event_error_auc_px_frames": float(sums.mean()),
        "stable_error_px": float(errors[:, -4:].mean()),
        "error_per_member_per_frame": errors.tolist(),
        "error_per_frame": errors.mean(0).tolist(),
        "supervised_visible_per_member_per_frame": measured.tolist(),
        "selected_unique_frames_per_frame": history,
        "recovery": recovery,
    }


def run_sequence(model, batch: ContentBatch, protocol, *, training=False):
    encoder = model.encoder
    members, length, _, height, width = batch.values.shape
    state = encoder.init_state(members, {"camera_ids": ["camera"], "shape": (height, width)})
    rows = torch.arange(height)[None, :, None]
    cols = torch.arange(width)[None, None, :]
    errors, history, roi_history, losses, frame_ms = [], [], [], [], []
    state_peak = 0
    with oracle_translation(encoder, batch):
        for frame_index in range(length):
            grad = training and frame_index >= int(protocol["burn_in"])
            started = time.perf_counter()
            with torch.set_grad_enabled(grad):
                output = encoder.step(
                    MeasurementFrame(
                        values=batch.values[:, frame_index],
                        valid_mask=batch.valid[:, frame_index],
                        timestamp=torch.full((members,), float(frame_index)),
                        camera_id="camera",
                        time_unit="frame",
                        sample_xy=CFA_OFFSETS,
                        metadata={"variance": batch.variance[:, frame_index]},
                    ),
                    state=state,
                )
                logits = model.head(output.representation).squeeze(1)
                probability = logits.flatten(1).softmax(1).reshape(members, height, width)
                prediction = torch.stack(
                    ((probability * rows).sum((1, 2)), (probability * cols).sum((1, 2))), 1
                )
                target = batch.target_xy[:, frame_index]
                errors.append((prediction - target).norm(dim=1).detach())
                if grad and bool(batch.visible[:, frame_index].any()):
                    distance = (rows - target[:, 0, None, None]).square() + (
                        cols - target[:, 1, None, None]
                    ).square()
                    heat = (-distance / (2 * float(protocol["heatmap_sigma"]) ** 2)).exp()
                    heat = heat / heat.sum((1, 2), keepdim=True).clamp_min(1e-12)
                    loss = -(heat.flatten(1) * logits.flatten(1).log_softmax(1)).sum(1)
                    losses.append(loss[batch.visible[:, frame_index]].mean())
            # Explicit one-step truncated BPTT. Raw history remains available;
            # gradients do not run backwards through previous graph updates.
            state = encoder.detach_state(output.state)
            history.append(float(output.diagnostics["mean_unique_frames"]))
            # Truth enters reporting only, after step and feature readout.
            statistics = encoder.last_statistics
            actual = (
                (
                    encoder.last_selection_weights
                    * statistics.unique_frames[encoder.last_selection_slice]
                )
                .sum(0)
                .mean(1)
            )
            roi = (rows - target[:, 0, None, None]).square() + (
                cols - target[:, 1, None, None]
            ).square() <= 4
            roi_history.append(
                float(((actual.detach() * roi).sum((1, 2)) / roi.sum((1, 2)).clamp_min(1)).mean())
            )
            state_peak = max(state_peak, int(output.cost["state_bytes"]))
            frame_ms.append((time.perf_counter() - started) * 1000)
    loss = torch.stack(losses).mean() if losses else None
    metrics = summarize_sequence(torch.stack(errors, 1), batch.visible, history, batch, protocol)
    metrics["frame_wall_ms"] = frame_ms
    metrics["roi_selected_unique_frames_per_frame"] = roi_history
    metrics["peak_state_bytes"] = state_peak
    return loss, metrics


def evaluate(model, protocol, seed, split):
    model.eval()
    cases = {}
    with torch.no_grad():
        for index, condition in enumerate(CONDITIONS):
            batch = make_protocol_batch(protocol, seed, split, index, condition)
            _, cases[condition] = run_sequence(model, batch, protocol)
    ordinary = [row for condition, row in cases.items() if condition != ORACLE]
    return {
        "cases": cases,
        "main_mean_error_px": sum(row["mean_error_px"] for row in ordinary) / len(ordinary),
        "main_early_error_px": sum(row["early_error_px"] for row in ordinary) / len(ordinary),
        "main_conditions": [name for name in CONDITIONS if name != ORACLE],
        "oracle_excluded_from_main_aggregate": True,
    }


def train_method(model, protocol, seed, method, output):
    initial_hashes = parameter_hashes(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    curves, input_hashes, fixture_hashes, step_ms = [], [], [], []
    best_score, best_step, best_state = float("inf"), 0, None
    active = set()
    start = time.perf_counter()
    for step in range(int(protocol["steps"])):
        model.train()
        batch = make_protocol_batch(protocol, seed, "train", step)
        input_hashes.append(batch.input_hash())
        fixture_hashes.append(batch.fixture_hash())
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        loss, measured = run_sequence(model, batch, protocol, training=True)
        if loss is None or not bool(torch.isfinite(loss)):
            raise RuntimeError("training produced no finite supervised loss")
        loss.backward()
        active.update(
            name for name, parameter in model.named_parameters() if parameter.grad is not None
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_ms.append((time.perf_counter() - step_start) * 1000)
        row = {
            "step": step + 1,
            "condition": batch.condition,
            "event_frame": batch.event_frame,
            "loss": float(loss.detach()),
            "train_error_px": measured["mean_error_px"],
        }
        if (step + 1) % int(protocol["validation_every"]) == 0 or step + 1 == protocol["steps"]:
            validation = evaluate(model, protocol, seed, "validation")
            score = validation["main_early_error_px"]
            row["validation_early_error_px"] = score
            row["validation_cases"] = {
                name: {
                    key: value
                    for key, value in case.items()
                    if key in {"mean_error_px", "early_error_px", "stable_error_px", "input_sha256"}
                }
                for name, case in validation["cases"].items()
            }
            if score < best_score:
                best_score, best_step = score, step + 1
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
                        "selection_metric": "validation main_early_error_px, oracle excluded",
                    },
                    output / f"seed{seed}-{method}-best.pt",
                )
            print(json.dumps({"seed": seed, "method": method, **row}), flush=True)
        curves.append(row)
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    evaluation_split = "validation" if protocol.get("development", False) else "test"
    test = evaluate(model, protocol, seed, evaluation_split)
    params = dict(model.named_parameters())
    return {
        "method": method,
        "initial_parameter_hashes": initial_hashes,
        "seed": seed,
        "content_mode": METHODS[method][0],
        "time_depths": list(METHODS[method][1]),
        "encoder_config": model.encoder.config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in params.values()),
        "gradient_active_parameter_count": sum(params[name].numel() for name in active),
        "inactive_parameter_names": sorted(set(params) - active),
        "best_step": best_step,
        "best_validation_early_error_px": best_score,
        "learning_curve": curves,
        "train_input_hashes": input_hashes,
        "train_fixture_hashes": fixture_hashes,
        "step_wall_ms": step_ms,
        "step_p50_ms": float(torch.tensor(step_ms).quantile(0.5)),
        "step_p95_ms": float(torch.tensor(step_ms).quantile(0.95)),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rss_scope": "process lifetime maximum; may include previously evaluated arms",
        "total_wall_seconds": time.perf_counter() - start,
        "evaluation_split": evaluation_split,
        ("development_validation" if evaluation_split == "validation" else "test"): test,
    }


def source_manifest():
    root = Path(__file__).resolve().parents[3]
    files = sorted((root / "src/owac/frameworks/n01").glob("*.py"))
    files += [Path(__file__), Path(__file__).with_name("n01_content_data.py")]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def run_experiment(protocol, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("refusing to overwrite an existing experiment result")
    torch.set_num_threads(int(protocol["threads"]))
    torch.use_deterministic_algorithms(True)
    payload = {
        "protocol": protocol,
        "source_manifest": source_manifest(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "boundaries": {
            "input": (
                "native synthetic RGGB linear spectral field; independent Gaussian sample noise"
            ),
            "oracle": (
                "known integer translation used in labelled control only; included in training, "
                "excluded from validation selection and main scores"
            ),
            "gradient": (
                "one-step truncated BPTT; burn-in without gradient, then visible-frame supervision"
            ),
            "normalization": (
                "one canonical adaptive catalogue training calibration; copied to every arm"
            ),
            "capacity_controls": (
                "E/F inject current measurement into same additional branch; "
                "old historical gates and graph retained"
            ),
            "parameter_budget": (
                "C/D/E/F match exactly; A/B lack content branch; report active and total counts"
            ),
            "units": "packed cell = two sensor pixels; no RGB reconstruction or teacher",
            "checkpoint_selection": (
                "minimum ordinary-condition validation early error; final test used once"
            ),
            "generalization": (
                "synthetic mechanism probe; no camera, VLA, sensor-transfer or robot acceptance"
            ),
        },
        "runs": [],
        "calibration": {},
    }
    for seed in protocol["seeds"]:
        models, calibration = make_matched_models(protocol, seed)
        payload["calibration"][str(seed)] = calibration
        for method, model in models.items():
            result = train_method(model, protocol, seed, method, output)
            payload["runs"].append(result)
            (output / "metrics.partial.json").write_text(json.dumps(payload, indent=2) + "\n")
    payload["source_manifest_at_end"] = source_manifest()
    if payload["source_manifest_at_end"] != payload["source_manifest"]:
        raise RuntimeError("experiment source changed during run; partial receipts preserved")
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--methods", choices=tuple(METHODS), nargs="+")
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.config).read_text())
    if args.development:
        protocol["development"] = True
    for key in ("steps", "seeds", "methods", "evaluation_batch_size"):
        value = getattr(args, key)
        if value is not None:
            protocol[key] = value
    if protocol["steps"] < 1 or protocol["validation_every"] < 1:
        parser.error("steps and validation_every must be positive")
    run_experiment(protocol, args.output)


if __name__ == "__main__":
    main()

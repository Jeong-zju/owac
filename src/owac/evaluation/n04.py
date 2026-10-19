"""Independent synthetic tracking and paired interventions; no task-success claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from ..frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from ..training.n04 import (
    _to_device,
    build_context,
    build_encoder,
    config_from_json,
    load_checkpoint,
)

MECHANISM_CONDITIONS = (
    "camera_only",
    "object_only",
    "both",
    "illumination",
    "same_direction",
    "occlusion",
    "calibration_error",
    "action_delay",
    "exposure_step",
)


def _mean(values):
    return statistics.fmean(values) if values else float("nan")


def _percentile(values, q):
    return (
        sorted(values)[min(len(values) - 1, round(q * (len(values) - 1)))]
        if values
        else float("nan")
    )


def _first(value):
    from ..frameworks.n04.model import fuse_scales

    return fuse_scales(value)


def _normalised_contrast(at, off):
    a, o = _mean(at), _mean(off)
    return (a - o) / max(a + o, 1e-12) if at and off else float("nan")


def _ratio_or_nan(a, b):
    return a / b if a > 0 and b > 0 else float("nan")


def _centroids(mask):
    ys, xs = torch.meshgrid(
        torch.arange(mask.shape[-2], device=mask.device),
        torch.arange(mask.shape[-1], device=mask.device),
        indexing="ij",
    )
    mass = mask.sum((-2, -1)).clamp_min(1e-6)
    return torch.stack(((mask * xs).sum((-2, -1)) / mass, (mask * ys).sum((-2, -1)) / mass), -1)[
        :, 0
    ]


def probe_metrics(head, stable, target):
    """Per-sample target localization; never use arbitrary feature energy as a target."""
    logits = head(_first(stable))
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    predicted = (logits.sigmoid() >= 0.5).to(target)
    errors = (_centroids(predicted) - _centroids(target)).norm(dim=-1)
    missing = predicted.flatten(1).sum(-1) == 0
    errors = torch.where(
        missing, errors.new_full(errors.shape, math.hypot(*target.shape[-2:])), errors
    )
    truth = target >= 0.5
    intersection = (predicted.bool() & truth).flatten(1).sum(-1)
    union = (predicted.bool() | truth).flatten(1).sum(-1).clamp_min(1)
    visible = truth.flatten(1).any(-1)
    errors = errors.masked_fill(~visible, float("nan"))
    iou = (intersection / union).masked_fill(~visible, float("nan"))
    return float(errors.nanmean()), float(iou.nanmean()), _centroids(predicted)


@dataclass
class ConditionResult:
    condition: str
    location_error: float
    change_response: float
    change_at_object: float
    change_off_object: float
    warp_error: float
    prediction_residual: float
    mask_coverage: float
    recovery_frames: int | None
    latency_p50_ms: float
    latency_p95_ms: float
    state_bytes: int
    frames: int
    extra: dict = field(default_factory=dict)
    trajectory: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class MechanismReport:
    conditions: list = field(default_factory=list)
    counterfactuals: list = field(default_factory=list)
    contrast: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {
            "conditions": [c.as_dict() for c in self.conditions],
            "counterfactuals": self.counterfactuals,
            "contrast": self.contrast,
            "notes": self.notes,
        }


@torch.no_grad()
def evaluate_condition(
    encoder,
    *,
    condition,
    scene,
    length=16,
    batch=2,
    seed=10000,
    warmup=2,
    pixel_stride=1.0,
    sequence=None,
):
    sequence = sequence or SyntheticSequenceGenerator(scene).sequence(
        batch=batch, length=length, condition=condition, seed=seed
    )
    camera = getattr(encoder, "config", getattr(encoder, "cfg", None))
    if camera is None:
        camera = encoder.encoder.config
    camera = camera.camera_ids[0]
    device = next(encoder.parameters()).device
    state = encoder.init_state(batch, device=device)[camera]
    previous = None
    rows, at_values, off_values, latencies = [], [], [], []
    head = getattr(encoder, "location_probe", None)
    for t, frame in enumerate(sequence):
        frame = _to_device(frame, device)
        context = build_context(frame, previous)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = encoder.step(
            frame.measured,
            context,
            state,
            camera_id=camera,
            saturation=frame.saturated_mask,
            measurement_valid=frame.valid_mask,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - started) * 1000
        state, previous = output.state, frame
        stable = _first(state.stable)
        target = frame.object_mask
        error, iou, center = (
            probe_metrics(head, stable, target) if head else (float("nan"), float("nan"), None)
        )
        alignment = float((stable - encoder.encode_measurement(frame.measured)).abs().mean())
        change = _first(state.change) if state.change is not None else None
        at, off, response = float("nan"), float("nan"), float("nan")
        if change is not None and change.abs().max() > 0:
            magnitude = change.abs().mean(1, keepdim=True)
            mask = F.adaptive_avg_pool2d(target, magnitude.shape[-2:])
            at = float((magnitude * mask).sum() / mask.sum().clamp_min(1e-6))
            off = float((magnitude * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1e-6))
            response = float(magnitude.mean())
        flow_error = float("nan")
        if state.forward_flow is not None:
            flow = F.interpolate(
                state.forward_flow.permute(0, 3, 1, 2),
                size=frame.oracle_camera_flow.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            flow_error = float((flow - frame.oracle_camera_flow / pixel_stride).abs().mean())
        residual = (
            float(output.innovation.residual.abs().mean()) if output.innovation else float("nan")
        )
        fp_rate = fn_rate = float("nan")
        if output.innovation is not None and output.innovation.gate_change.max() > 0:
            if frame.condition != "same_direction":
                gate = output.innovation.gate_change
                valid = (
                    F.adaptive_avg_pool2d(
                        frame.valid_mask * (1 - frame.saturated_mask), gate.shape[-2:]
                    )
                    > 0.5
                )
                moving = torch.as_tensor(frame.metadata["object_moving"], device=device)
                truth = (F.adaptive_avg_pool2d(target, gate.shape[-2:]) > 0.5) & moving[
                    :, None, None, None
                ]
                predicted_change = gate >= 0.5  # fixed threshold, never tuned on this evaluation
                positive, negative = valid & truth, valid & ~truth
                if negative.any():
                    fp_rate = float((predicted_change & negative).sum() / negative.sum())
                if positive.any():
                    fn_rate = float((~predicted_change & positive).sum() / positive.sum())
        rows.append(
            {
                "frame": t,
                "target_xy": _centroids(target).tolist(),
                "target_visible": (target >= 0.5).flatten(1).any(-1).tolist(),
                "predicted_xy": center.tolist() if center is not None else None,
                "location_error_px": error,
                "external_false_positive_rate": fp_rate,
                "external_false_negative_rate": fn_rate,
                "target_iou": iou,
                "feature_alignment_l1": alignment,
                "change_response": response,
                "change_at_object": at,
                "change_off_object": off,
                "warp_error": flow_error,
                "prediction_residual": residual,
                "mask_coverage": float(output.validity.mean()),
                "measurement_coverage": float(frame.valid_mask.mean()),
                "unknown": float(output.innovation.unknown.mean()) if output.innovation else None,
                "latency_ms": latency,
            }
        )
        if t >= warmup:
            at_values.append(at)
            off_values.append(off)
            latencies.append(latency)
    measured = rows[warmup:]

    def avg(key):
        return _mean([r[key] for r in measured if math.isfinite(r[key])])

    # Recovery is measured on the SAME continuous sequence, with censoring.
    recovery = None
    event = (
        scene.stop_after if condition == "object_only" else 6 if condition == "occlusion" else None
    )
    if head and event is not None:
        for t in range(event, len(rows) - 1):
            if all(
                r["target_iou"] >= 0.5 and r["location_error_px"] <= 2.0 for r in rows[t : t + 2]
            ):
                recovery = t - event
                break
    return ConditionResult(
        condition=condition,
        location_error=avg("location_error_px"),
        change_response=avg("change_response"),
        change_at_object=avg("change_at_object"),
        change_off_object=avg("change_off_object"),
        warp_error=avg("warp_error"),
        prediction_residual=avg("prediction_residual"),
        mask_coverage=avg("mask_coverage"),
        recovery_frames=recovery,
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        state_bytes=state.state_bytes(),
        frames=len(rows),
        extra={
            "external_false_positive_rate": avg("external_false_positive_rate"),
            "external_false_negative_rate": avg("external_false_negative_rate"),
            "external_gate_threshold": 0.5,
            "feature_alignment_l1": avg("feature_alignment_l1"),
            "target_iou": avg("target_iou"),
            "location_probe_available": head is not None,
            "recovery_event": event,
            "recovery_censored": event is not None and recovery is None,
            "change_contrast": _mean(at_values) - _mean(off_values),
            "change_contrast_normalised": _normalised_contrast(at_values, off_values),
        },
        trajectory=rows,
    )


@torch.no_grad()
def run_mechanism_suite(
    encoder,
    *,
    base_scene,
    conditions=MECHANISM_CONDITIONS,
    length=16,
    batch=2,
    seed=10000,
    pixel_stride=1.0,
):
    report = MechanismReport()
    for condition in conditions:
        scene = replace(base_scene, stop_after=8 if condition == "object_only" else None)
        report.conditions.append(
            evaluate_condition(
                encoder,
                condition=condition,
                scene=scene,
                length=length,
                batch=batch,
                seed=seed,
                pixel_stride=pixel_stride,
            )
        )
    generator = SyntheticSequenceGenerator(base_scene)
    base = generator.sequence(batch=batch, length=length, condition="camera_only", seed=seed)
    external = generator.sequence(batch=batch, length=length, condition="both", seed=seed)
    equal_context = all(
        torch.equal(a.proprio, b.proprio) and torch.equal(a.action, b.action)
        for a, b in zip(base, external, strict=True)
    )
    alternate = [
        replace(f, proprio=torch.zeros_like(f.proprio), action=torch.zeros_like(f.action))
        for f in base
    ]
    for title, left, right, invariant in (
        ("same_context_different_object", base, external, equal_context),
        (
            "same_visual_different_ego",
            base,
            alternate,
            all(torch.equal(a.measured, b.measured) for a, b in zip(base, alternate, strict=True)),
        ),
    ):
        if not invariant:
            raise AssertionError(f"counterfactual invariant failed: {title}")
        a, b = [
            evaluate_condition(
                encoder,
                condition=title,
                scene=base_scene,
                sequence=seq,
                length=length,
                batch=batch,
                seed=seed,
                pixel_stride=pixel_stride,
            )
            for seq in (left, right)
        ]
        report.counterfactuals.append(
            {
                "pair": title,
                "invariant_verified": invariant,
                "change_left": a.change_response,
                "change_right": b.change_response,
                "location_left_px": a.location_error,
                "location_right_px": b.location_error,
            }
        )
    lookup = {c.condition: c for c in report.conditions}
    if {"camera_only", "object_only"} <= lookup.keys():
        report.contrast["camera_vs_object_change"] = _ratio_or_nan(
            lookup["camera_only"].change_response, lookup["object_only"].change_response
        )
    if {"illumination", "object_only"} <= lookup.keys():
        report.contrast["illumination_response_ratio"] = _ratio_or_nan(
            lookup["illumination"].change_response, lookup["object_only"].change_response
        )
    report.notes = [
        "合成独立种子；location_error 为训练定位探针的目标质心误差（像素）；无探针为空。",
        "变化响应比是幅值代理，不能称为误报率；空分支/无支持指标为空。",
        "恢复要求连续两帧 IoU>=0.5 且误差<=2px，未恢复记 censored。",
        "四条基本条件的逐帧轨迹随报告保存；仅提供机制证据。",
    ]
    return report


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--suite", choices=["mechanism"], default="mechanism")
    parser.add_argument("--output", default="outputs/n04-eval")
    parser.add_argument("--baseline", default="candidate")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg, _ = config_from_json(args.config)
    encoder = build_encoder(cfg, baseline=None if args.baseline == "candidate" else args.baseline)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, encoder=encoder)
    encoder.to(args.device)
    encoder.eval()
    scene = SceneConfig(
        height=cfg.measurement_size[0],
        width=cfg.measurement_size[1],
        input_mode=cfg.input_mode,
        cfa_pattern=cfg.cfa_pattern,
        camera_translation=(0.6, 2.0),
        camera_rotation=(0.0, 0.7),
        object_velocity=(0.4, 1.8),
        illumination=(0.7, 1.3),
        calibration_scale=1.08,
        action_delay=2,
        seed=args.seed,
    )
    report = run_mechanism_suite(
        encoder,
        base_scene=scene,
        length=args.length,
        batch=args.batch,
        seed=args.seed,
        pixel_stride=cfg.geometry.pixel_stride,
    )
    checkpoint_hash = (
        hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None
    )
    payload = json_safe(
        {
            "schema_version": 2,
            "baseline": args.baseline,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "model": cfg.model.as_dict(),
            "trained": args.checkpoint is not None,
            "report": report.as_dict(),
        }
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mechanism.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "contrast": payload["report"]["contrast"],
                "location_mean_px": _mean([r.location_error for r in report.conditions]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

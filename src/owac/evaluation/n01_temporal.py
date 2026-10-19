"""Paired, measurement-only N01 motion/start/stop mechanism diagnostic.

Run the identical module against an archived source using PYTHONPATH for a
before/after comparison. It fits scales on independent training measurements;
there is no gradient training, localisation score, or task-benefit verdict.
Geometry enters only the reporting ROI, never the encoder or its calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from ..core.encoder_contract import MeasurementFrame
from ..frameworks.n01.config import N01Config
from ..frameworks.n01.model import N01Encoder, calibrate_statistics
from ..frameworks.n01.synthetic import SyntheticWeakEdgeTask


def motion_measurements(*, batch_size=24, speed=1.0, seed=1100):
    """Same Gaussian sensor-noise draw, slit and variance in all motion cases."""
    size, channels, length = 32, 3, 24
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch_size, length, channels, size, size, generator=generator) * 0.03
    cols = torch.arange(size)[None, None, None, None, :]
    centres = torch.full((length,), 16.0)
    if speed:
        # Eight still frames, eight measured moves, then eight still frames.
        centres[8:16] = 16.0 - speed * torch.arange(1, 9)
        centres[16:] = centres[15]
    centre = centres[None, :, None, None, None]
    coverage = (
        torch.minimum(cols + 0.5, centre + 0.75) - torch.maximum(cols - 0.5, centre - 0.75)
    ).clamp(0, 1)
    values = 0.2 + 0.05 * coverage + noise
    variance = torch.full_like(values, 0.03**2)
    return values, variance, centres


def run_case(encoder, values, variance, centres, *, valid=None):
    batch, length, _, height, width = values.shape
    if centres.ndim == 1:
        centres = centres[None].expand(batch, -1)
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    digest = hashlib.sha256()
    for tensor in (values, variance, valid, centres):
        digest.update(tensor.contiguous().numpy().tobytes())
    state = encoder.init_state(batch, {"camera_ids": ["camera"], "shape": (height, width)})
    rows = []
    with torch.inference_mode():
        for step in range(length):
            output = encoder.step(
                MeasurementFrame(
                    values=values[:, step],
                    valid_mask=valid[:, step],
                    timestamp=torch.full((batch,), float(step)),
                    camera_id="camera",
                    time_unit="frame",
                    metadata={"variance": variance[:, step]},
                ),
                state=state,
            )
            state = output.state
            stats = encoder.last_statistics
            weights = encoder.last_selection_weights
            selected = encoder.last_selection_slice
            actual = (weights * stats.unique_frames[selected]).sum(0).mean(1)
            depths = stats.depth_index[selected].float()[:, None, None, None, None]
            configured = (weights * depths).sum(0).mean(1)
            # This ROI belongs to evaluation only and is computed after step().
            roi = (torch.arange(width)[None] - centres[:, step, None]).abs() <= 2
            roi_history = (actual * roi[:, None]).sum((1, 2)) / (roi.sum(1) * height)
            roi_depth = (configured * roi[:, None]).sum((1, 2)) / (roi.sum(1) * height)
            peak = centres[:, step].round().long().clamp(0, width - 1)
            peak_history = actual.gather(2, peak[:, None, None].expand(-1, height, 1)).mean((1, 2))
            rows.append(
                {
                    "frame": step,
                    "roi_history_per_member": roi_history.tolist(),
                    "roi_history": float(roi_history.mean()),
                    "roi_configured_depth": float(roi_depth.mean()),
                    "peak_history": float(peak_history.mean()),
                    "peak_history_per_member": peak_history.tolist(),
                    "global_history": float(actual.mean()),
                }
            )
    rows[0]["input_sha256"] = digest.hexdigest()
    return rows


def validate_weak_edge(encoder, *, batch_size, seed, config):
    """Existing Poisson-Gaussian task; same seeds/geometry, no new speed labels."""
    cases = {}
    for motion in ("still", "slow", "fast"):
        sample = SyntheticWeakEdgeTask(
            config, image_size=32, snr="low", motion=motion, seed=seed
        ).sample(batch_size, 8)
        cases[motion] = run_case(
            encoder,
            sample.values,
            sample.variance,
            sample.target_xy[:, :, 1] * 31,
            valid=sample.valid,
        )
    return {
        "protocol": "original_low_snr_poisson_gaussian_task_v2",
        "nominal_fast_speed_px_per_frame": 1.6 * 31 / 32 / 7,
        "cases": cases,
        "summary": {
            motion: {
                "roi_history": sum(row["roi_history"] for row in rows[4:]) / 4,
                "peak_history": sum(row["peak_history"] for row in rows[4:]) / 4,
                "global_history": sum(row["global_history"] for row in rows[4:]) / 4,
            }
            for motion, rows in cases.items()
        },
    }


def validate_motion(*, batch_size=24, seed=1100, alignment="none"):
    config = N01Config.tiny(
        width=8,
        num_blocks=1,
        readout_tokens=16,
        readout_width=32,
        alignment=alignment,
        fractional_alignment=alignment != "none",
    )
    torch.manual_seed(0)
    encoder = N01Encoder(config).eval()
    calibration = SyntheticWeakEdgeTask(config, seed=500000).sample(2, 8)
    calibrate_statistics(encoder, calibration)
    scales = encoder.normalization_scales.clone()
    cases = {}
    for name, speed in (("still", 0.0), ("subpixel", 0.25), ("fast", 1.0)):
        cases[name] = run_case(
            encoder, *motion_measurements(batch_size=batch_size, speed=speed, seed=seed)
        )
    weak_edge = validate_weak_edge(encoder, batch_size=batch_size, seed=seed, config=config)
    assert torch.equal(scales, encoder.normalization_scales)

    def phase(name, begin, end):
        rows = cases[name][begin:end]
        by_member = torch.tensor([row["roi_history_per_member"] for row in rows]).mean(0)
        return by_member

    still = phase("still", 8, 16)
    moving = phase("fast", 8, 16)
    paired = still - moving
    return {
        "protocol": "n01-temporal-paired-v1",
        "encoder_source": inspect.getfile(N01Encoder),
        "encoder_source_sha256": hashlib.sha256(
            Path(inspect.getfile(N01Encoder)).read_bytes()
        ).hexdigest(),
        "config": config.to_dict(),
        "batch_size": batch_size,
        "seed": seed,
        "training": "independent calibration only; no gradient updates",
        "sensor": "linear slit; Gaussian sigma=0.03; contrast=0.05; 32x32x3",
        "calibration_seed": 500000,
        "normalization_scales": scales.tolist(),
        "metric": "mixture-weighted actual unique frames at the current edge ROI",
        "still_history": float(still.mean()),
        "fast_history": float(moving.mean()),
        "subpixel_history": float(phase("subpixel", 8, 16).mean()),
        "fast_stop_history": float(phase("fast", 22, 24).mean()),
        "still_peak_history": sum(row["peak_history"] for row in cases["still"][8:16]) / 8,
        "fast_peak_history": sum(row["peak_history"] for row in cases["fast"][8:16]) / 8,
        "paired_history_reduction": float(paired.mean()),
        "paired_positive_members": int((paired > 0).sum()),
        "paired_history_reduction_per_member": paired.tolist(),
        "cases": cases,
        "original_weak_edge": weak_edge,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1100)
    parser.add_argument("--alignment", choices=("none", "fixed", "candidate"), default="none")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    result = validate_motion(batch_size=args.batch_size, seed=args.seed, alignment=args.alignment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"cases", "original_weak_edge"}
            }
        )
    )
    print(json.dumps(result["original_weak_edge"]["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent sensor-noise and truth-motion diagnostics after matched tracking.

These are RAW-domain diagnostic limits, not newly trained N04 task performance.
All score variants see the same noisy samples and support. Oracle radiance/noise
and foreground support are explicitly identified; none enters normal tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from ..frameworks.n04.geometry import GridSpec, backward_grid_from_translation
from ..frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from ..training.n04 import build_context, build_encoder, config_from_json, load_checkpoint
from .n04_tracking import digest, save_json


def bilinear_variance(variance: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Propagate independent sensor noise through bilinear sampling: sum(w^2*v)."""
    b, c, h, w = variance.shape
    x = (grid[..., 0] + 1) * w / 2 - 0.5
    y = (grid[..., 1] + 1) * h / 2 - 0.5
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = x - x0, y - y0
    result = variance.new_zeros(b, c, *grid.shape[1:3])
    flat = variance.flatten(2)
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        ix, iy = x0 + dx, y0 + dy
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        index = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).flatten(1)
        sample = flat.gather(2, index[:, None].expand(-1, c, -1)).reshape_as(result)
        result += sample * (weight.square() * inside)[:, None]
    return result


def residual_score(current, prediction, *, current_variance=None, prediction_variance=None):
    residual = current - prediction
    if current_variance is not None:
        residual = residual / (current_variance + prediction_variance).clamp_min(1e-10).sqrt()
    return residual.square().mean(1, keepdim=True).sqrt()


def shot_read(clean, alpha, beta, rng):
    noisy = alpha * torch.poisson(clean / alpha, generator=rng)
    noisy += torch.randn(clean.shape, generator=rng) * beta**0.5
    return noisy.clamp(0, 1)


def frame_pair(
    seed, *, camera_on=False, object_on=False, alpha=0.005, beta=0.0001, background=0.15
):
    config = SceneConfig(
        canvas_margin=32,
        noise_sigma=0,
        background_level=background,
        object_colors=((0.7, 0.3, 0.2),),
        seed=seed,
    )
    renderer = SyntheticSequenceGenerator(config)
    rng = torch.Generator().manual_seed(seed + 19)
    batch = 4
    base = 64 + (torch.rand(batch, 1, 2, generator=rng) - 0.5) * 8

    def displacement():
        direction = torch.randn(batch, 2, generator=rng)
        return direction / direction.norm(dim=-1, keepdim=True) * 1.5

    camera = displacement() if camera_on else torch.zeros(batch, 2)
    obj = displacement() if object_on else torch.zeros(batch, 2)
    frames = []
    for t in range(2):
        frame = renderer.frame_batch(
            camera_translation=t * camera,
            camera_rotation=torch.zeros(batch),
            object_translation=t * obj,
            illumination=torch.ones(batch),
            base_object_centers=base,
            base_camera_offset=torch.zeros(batch, 2),
            camera_step=t * camera,
            generator=rng,
        )
        measured = shot_read(frame.oracle_clean, alpha, beta, rng)
        frame = replace(
            frame,
            measured=measured,
            saturated_mask=(measured >= 0.999).any(1, keepdim=True).float(),
            action=t * frame.proprio,
            metadata={**frame.metadata, "t": t},
        )
        frames.append(frame)
    return frames


def detection_metrics(score, positive, negative, threshold):
    return dict(
        false_positive=int(((score > threshold) & negative).sum()),
        negatives=int(negative.sum()),
        true_positive=int(((score > threshold) & positive).sum()),
        positives=int(positive.sum()),
    )


def noise_split(seeds, thresholds=None):
    rows, negatives_by_method = (
        [],
        {name: [] for name in ("raw", "normalized_estimated", "normalized_truth")},
    )
    noise_levels = ((0.001, 1e-5), (0.005, 1e-4), (0.02, 1e-4), (0.04, 1e-3))
    backgrounds = (0.03, 0.15, 0.35, 0.65)
    data_hash = hashlib.sha256()
    for i, seed in enumerate(seeds):
        alpha, beta = noise_levels[i % 4]
        background = backgrounds[(i // 4) % 4]
        for moving in (False, True):
            previous, current = frame_pair(
                seed, object_on=moving, alpha=alpha, beta=beta, background=background
            )
            for f in (previous, current):
                data_hash.update(f.measured.numpy().tobytes())
            support = (previous.saturated_mask == 0) & (current.saturated_mask == 0)
            observable = (current.oracle_clean - previous.oracle_clean).abs().amax(1, keepdim=True)
            positive, negative = (observable > 0.05) & support, (observable < 1e-6) & support
            scores = dict(raw=residual_score(current.measured, previous.measured))
            for name, radiances in (
                ("normalized_estimated", (current.measured, previous.measured)),
                ("normalized_truth", (current.oracle_clean, previous.oracle_clean)),
            ):
                scores[name] = residual_score(
                    current.measured,
                    previous.measured,
                    current_variance=alpha * radiances[0] + beta,
                    prediction_variance=alpha * radiances[1] + beta,
                )
            for method, score in scores.items():
                if thresholds is None:
                    if not moving:
                        negatives_by_method[method].append(score[negative])
                else:
                    rows.append(
                        dict(
                            seed=seed,
                            alpha=alpha,
                            beta=beta,
                            background=background,
                            moving=moving,
                            method=method,
                            **detection_metrics(score, positive, negative, thresholds[method]),
                        )
                    )
    if thresholds is None:
        thresholds = {
            k: float(torch.quantile(torch.cat(v), 0.99)) for k, v in negatives_by_method.items()
        }
    return thresholds, rows, data_hash.hexdigest()


@torch.no_grad()
def motion_split(encoder, seeds):
    rows = []
    alpha, beta = 0.005, 0.0001
    spec = GridSpec(64, 64)
    for seed in seeds:
        for condition, camera_on, object_on in (
            ("static", False, False),
            ("camera_only", True, False),
            ("object_only", False, True),
            ("both", True, True),
        ):
            previous, current = frame_pair(seed, camera_on=camera_on, object_on=object_on)
            state = encoder.init_state(4, device="cpu")["head"]
            # Each pair is a new episode; explicitly clear the frame cache.
            encoder._cache.clear()
            state = encoder.step(
                previous.measured,
                build_context(previous, None),
                state,
                measurement_valid=previous.valid_mask,
                saturation=previous.saturated_mask,
            ).state
            _, _, _, result = encoder.predict_before_observe(
                state, build_context(current, previous)
            )
            ego = previous.true_camera_translation - current.true_camera_translation
            obj = current.true_object_translation - previous.true_object_translation
            ego_grid = backward_grid_from_translation(spec, ego[:, 0], ego[:, 1], batch=4)
            object_grid = backward_grid_from_translation(
                spec, (ego + obj)[:, 0], (ego + obj)[:, 1], batch=4
            )
            foreground = current.object_mask > 0.5
            complete_grid = torch.where(foreground.permute(0, 2, 3, 1), object_grid, ego_grid)
            grids = {
                "learned_ego": result.backward_grid,
                "truth_ego": ego_grid,
                "truth_ego_object": complete_grid,
            }
            # Shared truth support in ALL variants: excludes unseen background and
            # partial foreground boundaries. It is an oracle diagnostic assumption.
            old_foreground_ego = F.grid_sample(previous.object_mask, ego_grid, align_corners=False)
            old_foreground_object = F.grid_sample(
                previous.object_mask, object_grid, align_corners=False
            )
            support = ((~foreground) & (old_foreground_ego < 0.01)) | (
                foreground & (old_foreground_object > 0.99)
            )
            support &= current.saturated_mask == 0
            for grid in grids.values():
                support &= (
                    F.grid_sample(
                        torch.ones_like(support, dtype=torch.float32), grid, align_corners=False
                    )
                    > 0.999
                )
                support &= F.grid_sample(previous.saturated_mask, grid, align_corners=False) < 1e-6
            for method, grid in grids.items():
                predicted = F.grid_sample(previous.measured, grid, align_corners=False)
                variance = bilinear_variance(alpha * previous.oracle_clean + beta, grid)
                score = residual_score(
                    current.measured,
                    predicted,
                    current_variance=alpha * current.oracle_clean + beta,
                    prediction_variance=variance,
                )
                error = (predicted - current.oracle_clean).square().mean(1, keepdim=True)
                fg, bg = foreground & support, (~foreground) & support
                rows.append(
                    dict(
                        seed=seed,
                        condition=condition,
                        method=method,
                        foreground_rmse=float(error[fg].mean().sqrt()),
                        background_rmse=float(error[bg].mean().sqrt()),
                        normalized_residual=float(score[support].mean()),
                        foreground_pixels=int(fg.sum()),
                        background_pixels=int(bg.sum()),
                    )
                )
    return rows


def aggregate_noise(rows):
    summary = {}
    for method in {r["method"] for r in rows}:
        selected = [r for r in rows if r["method"] == method]

        def aggregate(part):
            fp = sum(r["false_positive"] for r in part)
            n = sum(r["negatives"] for r in part)
            tp = sum(r["true_positive"] for r in part)
            p = sum(r["positives"] for r in part)
            return dict(false_positive_rate=fp / n, recall=tp / p, positives=p, negatives=n)

        summary[method] = aggregate(selected)
        summary[method]["by_alpha"] = {
            str(a): aggregate([r for r in selected if r["alpha"] == a])
            for a in sorted({r["alpha"] for r in selected})
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tracking = json.loads((args.tracking_dir / "protocol.json").read_text())
    if not tracking.get("complete") or not tracking.get("dynamic_benefit"):
        raise RuntimeError(
            "Dynamic-state benefit gate must pass before independent RAW diagnostics"
        )
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    root = args.output_dir
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    protocol = dict(
        tracking_protocol_sha256=digest(args.tracking_dir / "protocol.json"),
        diagnostic_only=True,
        raw_input="synthetic three-plane linear samples; not native Bayer",
        noise_factor=(
            "identical RAW residual, change only division by predicted noise standard deviation"
        ),
        noise_model="independent Poisson shot noise + Gaussian read noise, clipped to [0,1]",
        noise_development_seeds=list(range(650000, 650032)),
        noise_final_seeds=list(range(660000, 660032)),
        motion_development_seeds=list(range(670000, 670006)),
        motion_final_seeds=list(range(680000, 680006)),
        oracle_fields=(
            "exact alpha/beta; normalized_truth uses clean radiance; "
            "motion uses true foreground support in all variants"
        ),
        motion_factor=(
            "learned ego -> truth ego -> truth independent-object motion, "
            "masks/support/weights fixed"
        ),
        gate=(
            "noise: recall rises at calibrated 1% development FPR; final FPR reported, not adjusted"
        ),
        source_sha256={str(Path(__file__)): digest(Path(__file__))},
        complete=False,
    )
    save_json(root / "protocol.json", protocol)
    thresholds, _, dev_hash = noise_split(protocol["noise_development_seeds"])
    save_json(
        root / "noise-freeze.json",
        dict(thresholds=thresholds, development_data_sha256=dev_hash, protocol=protocol),
    )
    _, noise_rows, test_hash = noise_split(protocol["noise_final_seeds"], thresholds)
    save_json(root / "noise-final.json", noise_rows)
    noise_summary = aggregate_noise(noise_rows)
    save_json(root / "noise-summary.json", noise_summary)
    cfg, _ = config_from_json(Path("configs/encoders/n04/tiny-staged-research.json"))
    encoder = build_encoder(cfg)
    checkpoint = Path(tracking["source_checkpoint"])
    load_checkpoint(checkpoint, encoder=encoder)
    encoder.eval().requires_grad_(False)
    # No motion-model tuning: development checks precede a fixed final diagnostic.
    save_json(
        root / "motion-development.json",
        motion_split(encoder, protocol["motion_development_seeds"]),
    )
    save_json(
        root / "motion-freeze.json", dict(protocol=protocol, checkpoint_sha256=digest(checkpoint))
    )
    motion_rows = motion_split(encoder, protocol["motion_final_seeds"])
    save_json(root / "motion-final.json", motion_rows)
    protocol.update(complete=True, noise_final_data_sha256=test_hash)
    assert digest(Path(__file__)) == protocol["source_sha256"][str(Path(__file__))]
    save_json(root / "protocol.json", protocol)
    print(json.dumps(noise_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

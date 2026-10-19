"""Read-only real-window diagnosis of objective coverage and latent/state scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from owac.frameworks.n04.encoder_v11 import build_encoder
from owac.raw.n04_v11_data import (
    PLATFORM_SPECS,
    V11DataIndex,
    load_window_batch,
    sample_windows,
    window_candidates,
)
from owac.training.n04_v11 import ForecastEngine, V11Trainer, load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--config", help="Explicit no-update model counterfactual; never a resume")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = load_config(args.config) if args.config else payload["config"]
    set_seed(cfg["seed"])
    device = torch.device("cuda:0")
    encoder = build_encoder(cfg["model"], platform=cfg["platform"], arm=cfg["arm"]).to(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    # Reuse the exact data/context/target helpers, without constructing a backend or optimizer.
    helper = V11Trainer.__new__(V11Trainer)
    helper.encoder, helper.device, helper.config = encoder, device, cfg
    helper.platform = cfg["platform"]
    helper.spec = PLATFORM_SPECS[cfg["platform"]]
    helper.forecast_horizons = tuple(cfg["loss"]["forecast_horizons"])
    index = V11DataIndex.from_config(cfg)
    candidates = window_candidates(
        index,
        platform=cfg["platform"],
        split="dev",
        **{k: cfg["data"]["window"][k] for k in ("burn_in", "gradient", "stride")},
    )
    plans = sample_windows(candidates, seed=cfg["seed"], step=0, batch=args.windows)
    report = {
        "platform": cfg["platform"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": payload["step"],
        "optimizer_updates": 0,
        "counterfactual_config": args.config,
        "windows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry, start in plans:
        window = load_window_batch(
            entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
        )
        encoder.reset()
        state = None
        forecast = ForecastEngine(
            encoder.state_stack,
            helper.forecast_horizons,
            total_frames=window.raw[window.camera_ids[0]].shape[0],
            max_innovation_variance=1000,
        )
        row = {"episode": window.episode_id, "start": start, "frames": []}
        report["windows"].append(row)
        try:
            with torch.no_grad():
                for position in range(forecast.total_frames):
                    frame, context = (
                        helper._frame(window, position),
                        helper._context(window, position),
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = encoder.step(
                            frame.values,
                            context,
                            state,
                            camera_id=frame.camera_id,
                            frame_key=(window.episode_id, position),
                            compute_auxiliary=position >= 16,
                        )
                    state = output.state
                    details = {
                        "position": position,
                        "losses": {
                            k: float(v)
                            for k, v in output.losses.items()
                            if isinstance(v, torch.Tensor)
                        },
                        "cameras": {},
                    }
                    row["frames"].append(details)
                    for cam in window.camera_ids:
                        s = state.per_camera[cam]
                        flow = output.debug["flow"][cam]
                        latent = output.target_features[cam]["latent"]
                        details["cameras"][cam] = {
                            "flow_max_abs": float(flow.abs().amax()),
                            "flow_mean": flow.mean((0, 2, 3)).tolist(),
                            "flow_valid": float(
                                output.debug["flow_valid"].get(cam, torch.ones(1)).float().mean()
                            ),
                            "visible_fraction": output.debug["correspondence_terms"]
                            .get(cam, {})
                            .get("visible_fraction"),
                            "latent_rms": float(latent.square().mean().sqrt()),
                            "latent_absmax": float(latent.abs().amax()),
                            "support_valid": float(s.valid.float().mean()),
                            "state_rms": float(s.mode_mean.square().mean().sqrt()),
                            "cov_max": max(float(v.amax()) for v in s.mode_cov[:2]),
                            "R_mean": output.debug["quality"][cam]["mean"],
                        }
                        if position in (0, 16):
                            transition = []
                            for cell in encoder.state_stack.cells:
                                blocks, _ = cell.get_transition_model(
                                    s.mode_mean[:, :1, 0].reshape(1, -1)
                                )
                                a = torch.cat(
                                    (torch.cat(blocks[:2], -1), torch.cat(blocks[2:], -1)), -2
                                )[0]
                                transition.append(
                                    {
                                        "norm_bound": float(
                                            (a.abs().sum(0).amax() * a.abs().sum(1).amax()).sqrt()
                                        ),
                                        "spectral_radius": float(
                                            torch.linalg.eigvals(a).abs().amax()
                                        ),
                                    }
                                )
                            details["cameras"][cam]["transition"] = transition
                    if position >= 16:
                        helper._transport_forecast(forecast, window, output, position)
                        helper._add_forecast_anchors(
                            forecast,
                            window,
                            position,
                            tuple(frame.values[0].shape[-2:]),
                            output,
                            context.proprio.reshape(1, -1),
                            forecast.total_frames - 1,
                        )
                        helper._settle_forecast(forecast, frame, position, output)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["forecast"] = forecast.report()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "episode": row["episode"],
                    "error": row.get("error"),
                    "last_frame": row["frames"][-1] if row["frames"] else None,
                    "forecast": row["forecast"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

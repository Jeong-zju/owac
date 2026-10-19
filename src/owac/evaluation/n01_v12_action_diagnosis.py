"""Full action sampling on expert observations, without optimizer construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def action_metrics(predicted, target, valid):
    """Report only genuine controls and valid horizons, separately by control group."""
    predicted, target = np.asarray(predicted)[..., :7], np.asarray(target)[..., :7]
    valid = np.asarray(valid, dtype=bool)
    error = predicted[valid] - target[valid]
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("finite real action targets required")
    first = np.zeros_like(valid)
    first[..., :5] = True
    initial = predicted[valid & first] - target[valid & first]
    return {
        "mse7": float(np.mean(error**2)),
        "mae7": float(np.mean(np.abs(error))),
        "per_dim_mse": np.mean(error**2, axis=0).tolist(),
        "translation_mae": float(np.abs(error[:, :3]).mean()),
        "rotation_mae": float(np.abs(error[:, 3:6]).mean()),
        "gripper_mae": float(np.abs(error[:, 6]).mean()),
        "gripper_sign_agreement": float(
            np.mean((predicted[valid, 6] >= 0) == (target[valid, 6] >= 0))
        ),
        "executed_first5_mae7": float(np.abs(initial).mean()),
        "valid_steps": int(valid.sum()),
    }


def main():
    import torch

    from owac.frameworks.n01_full.acceleration import configure_acceleration
    from owac.frameworks.n01_full.model import N01V12Config
    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.runtime.n01_v12_service import _load_arm, load_arm_checkpoint
    from owac.training.n01_action import observation_from_batch
    from owac.training.n01_v12 import (
        _build_datasets,
        _policy_forward,
        _protocol_path,
        encode_windows_v12,
    )
    from owac.training.n01_v12_sampling import validation_indices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--role", choices=["raw", "rgb", "author"], required=True)
    args = parser.parse_args()
    repo = args.root / "source"
    folder = args.root / f"expert-{args.role}"
    folder.mkdir(parents=True, exist_ok=True)
    config_path = (
        repo
        / "configs/encoders/n01_full/v1_2/complete"
        / f"libero-{'raw' if args.role == 'author' else args.role}_n01-seed101.json"
    )
    config = N01V12Config.from_dict(json.loads(config_path.read_text()))
    device = torch.device("cuda:0")
    weights = (
        args.deploy
        / "source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors"
    )
    tokenizer = args.deploy / "source/openpi_converted_protocol/paligemma_tokenizer.model"
    if args.role != "author":
        model, policy, bridge, _ = _load_arm(weights, tokenizer, config_path, "libero", str(device))
        checkpoint = args.previous / f"checkpoints/libero-{args.role}_n01-captured.pt"
        restored = load_arm_checkpoint(model, checkpoint, device, policy=policy)
        model.to(device).eval()
        configure_acceleration(model, device, "bf16", pair_chunk=8)
    else:
        from pi_libero_common import build_model

        policy = build_model(weights, device, "bfloat16", tokenizer=tokenizer)
        restored = {"step": 0, "source": "original pretrained image policy"}
    datasets = {
        region: _build_datasets(repo, _protocol_path(repo, None), "libero", region, 32, config)
        for region in ("validation", "train")
    }
    cases = [
        (region, index)
        for region, count in [("validation", 8), ("train", 4)]
        for index in validation_indices(len(datasets[region]), count, 101)
    ]
    rows = []
    with torch.no_grad():
        for ordinal, (region, index) in enumerate(cases):
            sample = datasets[region][index]
            batch = collate_sequence_samples([sample])
            target = batch["action"].numpy()[0]
            valid = batch["action_mask"].numpy()[0]
            observation = observation_from_batch(policy, batch, device)
            truth = torch.nn.functional.pad(batch["action"].to(device), (0, policy.action_dim - 7))
            flow = None
            if args.role != "author":
                tokens, _ = encode_windows_v12(model, batch, device)
                torch.manual_seed(config.seed * 7919 + index)
                result = _policy_forward(bridge, observation, tokens, truth, return_debug=True)
                e = (result["v_t"] - result["u_t"]).square()[0, batch["action_mask"][0].to(device)]
                flow = {
                    "mse32": e.mean().item(),
                    "mse7": e[:, :7].mean().item(),
                    "mse_padding25": e[:, 7:].mean().item(),
                    "time": result["time"].tolist(),
                }
            else:
                from owac.raw.n01_paired_views import raw_unit_to_srgb_u8

                images = [
                    raw_unit_to_srgb_u8(
                        batch["windows"][cam].values[0, -1].permute(1, 2, 0).numpy()
                    )
                    for cam in config.cameras
                ]
                example = {
                    "image": images,
                    "lang": batch["meta"][0]["task_description"],
                    "state": batch["state"].numpy(),
                }
            predictions = []
            for seed in (101, 102):
                noise = (
                    np.random.default_rng(seed + ordinal * 104729)
                    .standard_normal((1, 10, 32))
                    .astype(np.float32)
                )
                if args.role == "author":
                    sampled = policy.predict_action([example], noise=noise, num_steps=10)[
                        "normalized_actions"
                    ][0]
                else:
                    sampled = (
                        bridge.sample_actions_with_features(
                            observation, tokens, noise=noise, num_steps=10
                        )[0]
                        .float()
                        .cpu()
                        .numpy()
                    )
                predictions.append(sampled)
            parity = None
            if ordinal == 0 and args.role != "author":
                from owac.runtime.n01_v12_service import N01V12Service

                service = N01V12Service(model, policy, bridge, device=str(device))
                frames = {}
                for camera, window in batch["windows"].items():
                    frames[camera] = {
                        "values": window.values.numpy(),
                        "value_units": "normalized_raw_measurement",
                        "episode_ids": [batch["meta"][0]["episode_key"]],
                        "frame_ids": window.frame_ids.numpy(),
                        "timestamp": window.timestamp.numpy(),
                        "time_unit": window.time_unit,
                        "new_measurement": window.new_measurement.numpy(),
                        "valid_mask": window.valid_mask.numpy(),
                        "bad_mask": window.bad_mask.numpy(),
                        "saturated_mask": window.saturated_mask.numpy(),
                    }
                    if window.variance is not None:
                        frames[camera]["variance"] = window.variance.numpy()
                first_noise = (
                    np.random.default_rng(101).standard_normal((1, 10, 32)).astype(np.float32)
                )
                online = service.predict_action(
                    frames=frames,
                    normalized_state=batch["state"].numpy(),
                    language=[batch["meta"][0]["task_description"]],
                    noise=first_noise,
                )["normalized_actions"][0]
                parity = {
                    "fresh_service_vs_offline_max_abs": float(
                        np.abs(online - predictions[0]).max()
                    ),
                    "fresh_service_vs_offline_mae7": float(
                        np.abs(online[:, :7] - predictions[0][:, :7]).mean()
                    ),
                }
                service.reset()
                del service
            stats = datasets[region].statistics.action
            lo, hi = np.asarray(stats["q01"][:7]), np.asarray(stats["q99"][:7])

            def physical(a, lo=lo, hi=hi):
                return (a[..., :7] + 1) * 0.5 * np.maximum(hi - lo, 1e-6) + lo

            metrics = [action_metrics(p, target, valid) for p in predictions]
            physical_metrics = [
                action_metrics(physical(p), batch["action_raw"].numpy()[0], valid)
                for p in predictions
            ]
            row = {
                "ordinal": ordinal,
                "region": region,
                "index": index,
                "meta": batch["meta"][0],
                "flow": flow,
                "service_parity": parity,
                "sampled_normalized": metrics,
                "sampled_environment_commands": physical_metrics,
                "zero_normalized_baseline": action_metrics(np.zeros_like(target), target, valid),
                "first_action_hold_baseline": action_metrics(
                    np.repeat(target[:1], len(target), axis=0), target, valid
                ),
            }
            np.savez_compressed(
                folder / f"sample-{ordinal:02d}.npz",
                predictions=np.stack(predictions),
                target=target,
                target_raw=batch["action_raw"].numpy()[0],
                valid=valid,
                state=batch["state"].numpy(),
                state_raw=sample.state_raw.numpy(),
            )
            with (folder / "samples.jsonl").open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            rows.append(row)
            print(
                json.dumps(
                    {
                        "role": args.role,
                        "sample": ordinal,
                        "flow": flow,
                        "sampled_mae7": metrics[0]["mae7"],
                        "zero_mae7": row["zero_normalized_baseline"]["mae7"],
                    }
                ),
                flush=True,
            )
    report = {
        "role": args.role,
        "restored": restored,
        "samples": len(rows),
        "sampling_seeds_per_observation": 2,
        "inference_steps": 10,
        "training_started": False,
        "rows": rows,
    }
    (folder / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

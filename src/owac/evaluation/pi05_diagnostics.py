"""Development-only probes of a frozen pi05 baseline, without retraining or test reuse."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from owac.evaluation.baseline_protocol import action_chunk, file_sha256, write_json


def rotation_error_degrees(predicted, target):
    """SO(3) error, including equivalent opposite axis-angle vectors near pi."""
    import numpy as np

    def quaternion(vectors):
        angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
        xyz = vectors * np.divide(
            np.sin(angle / 2), angle, out=np.full_like(angle, 0.5), where=angle > 1e-8
        )
        return np.concatenate([np.cos(angle / 2), xyz], axis=-1)

    dot = np.sum(quaternion(predicted) * quaternion(target), axis=-1)
    return np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))


def action_metrics(predicted, target):
    import numpy as np

    xyz = np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1)
    rotation = rotation_error_degrees(predicted[..., 3:6], target[..., 3:6])
    return {
        "xyz_error_mean_m": float(xyz.mean()),
        "xyz_error_p95_m": float(np.percentile(xyz, 95)),
        "rotation_error_mean_deg": float(rotation.mean()),
        "rotation_error_p95_deg": float(np.percentile(rotation, 95)),
        "gripper_sign_accuracy": float(np.mean((predicted[..., 6] >= 0) == (target[..., 6] >= 0))),
        "predicted_rotation_norm_mean": float(np.linalg.norm(predicted[..., 3:6], axis=-1).mean()),
    }


def install_padding_bridge(model):
    """Diagnostic intervention: impose the known zero-target bridge on padding only.

    Training zero-pads actions but supervises only seven velocities. In free
    sampling the other 25 velocities can drift. Setting those velocities to the
    initial noise reproduces x_padding(t)=t*noise, the training support. This is
    an ablation of the frozen model, not a replacement baseline result.
    """
    original = model.denoise_step
    model._diagnostic_noise = None
    model._diagnostic_padding_trace = []
    model._diagnostic_bridge_enabled = False

    def denoise(state, masks, cache, x_t, timestep):
        velocity = original(state, masks, cache, x_t, timestep)
        noise = model._diagnostic_noise
        if noise is not None:
            residual = x_t[..., 7:] - timestep[:, None, None] * noise[..., 7:]
            model._diagnostic_padding_trace.append(float(residual.square().mean().sqrt()))
        if model._diagnostic_bridge_enabled:
            velocity = velocity.clone()
            velocity[..., 7:] = noise[..., 7:]
        return velocity

    model.denoise_step = denoise


def predict(model, observation, protocol, noise, bridge=False, flow_steps=None):
    import torch

    model._diagnostic_noise = noise
    model._diagnostic_padding_trace = []
    model._diagnostic_bridge_enabled = bridge
    with torch.inference_mode():
        output = model.sample_actions(
            "cuda", observation, noise=noise.clone(), num_steps=flow_steps or protocol["flow_steps"]
        )
    return output, list(model._diagnostic_padding_trace)


def offline(args, protocol):
    import numpy as np
    import torch
    from openpi.models.tokenizer import PaligemmaTokenizer

    from owac.training.pi05_sim import batch, make_model

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(8)
    stats = json.loads((args.weights / "normalization.json").read_text())
    manifest = json.loads((args.data / "dataset.json").read_text())
    tokenizer = PaligemmaTokenizer(200)
    items, metadata = [], []
    audit = []
    for split in ("train", "validation"):
        for episode in [e for e in manifest["episodes"] if e["split"] == split][:2]:
            source = args.data / episode["path"]
            assert file_sha256(source) == episode["sha256"]
            with np.load(source, allow_pickle=False) as arrays:
                phases = arrays["phase"]
                audit.append(
                    {
                        "seed": episode["seed"],
                        "split": split,
                        "hash_verified": True,
                        "frames": len(phases),
                        "image_std": {k: float(arrays[k].std()) for k in ("external", "wrist")},
                        "phase_counts": {
                            str(p): int(np.sum(phases == p)) for p in np.unique(phases)
                        },
                    }
                )
                for phase in np.unique(phases):
                    indices = np.flatnonzero(phases == phase)
                    for frame in (indices[0], indices[len(indices) // 2], indices[-1]):
                        items.append(
                            {k: arrays[k][frame].copy() for k in ("external", "wrist", "state")}
                            | {
                                "actions": action_chunk(
                                    arrays["actions"], frame, protocol["action_horizon"]
                                )
                            }
                        )
                        metadata.append(
                            {
                                "seed": episode["seed"],
                                "split": split,
                                "phase": str(phase),
                                "frame": int(frame),
                            }
                        )
    write_json(args.output / "data-audit.json", audit)
    model = make_model(protocol, args.weights / "model.safetensors", dtype="bfloat16").cuda().eval()
    install_padding_bridge(model)
    records = []
    with (args.output / "predictions.jsonl").open("w") as stream:
        for offset in range(0, len(items), 4):
            subset = items[offset : offset + 4]
            obs, target = batch(subset, stats, tokenizer, "cuda")
            generator = torch.Generator(device="cuda").manual_seed(1729 + offset)
            noise = torch.randn(target.shape, generator=generator, device="cuda")
            targets = np.stack([item["actions"] for item in subset])
            for variant, bridge in (("original", False), ("padding_bridge", True)):
                normalized, padding_trace = predict(model, obs, protocol, noise, bridge)
                normalized = normalized.cpu().numpy()
                predicted = (
                    normalized[..., :7] * stats["actions"]["scale"] + stats["actions"]["center"]
                )
                for j in range(len(subset)):
                    record = metadata[offset + j] | {
                        "variant": variant,
                        "predicted": predicted[j].tolist(),
                        "target": targets[j].tolist(),
                        "state": subset[j]["state"].tolist(),
                        "padding_final_rms": float(np.sqrt(np.mean(normalized[j, :, 7:] ** 2))),
                        "padding_training_support_residual_rms_by_step": padding_trace,
                    }
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    records.append(record)
            print("OFFLINE_BATCH", offset + len(subset), "/", len(items), flush=True)
    summary = {
        "status": "complete",
        "purpose": "development diagnosis, not final test or retraining",
        "model_sha256": file_sha256(args.weights / "model.safetensors"),
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_manifest_sha256": file_sha256(args.data / "dataset.json"),
        "seeds": sorted({r["seed"] for r in records}),
        "sample_selection": (
            "first/middle/last frame of each phase in first two episodes of each split"
        ),
        "wall_seconds": time.perf_counter() - started,
        "metrics": {},
    }
    for variant in ("original", "padding_bridge"):
        summary["metrics"][variant] = {}
        for split in ("train", "validation"):
            selected = [r for r in records if r["variant"] == variant and r["split"] == split]
            summary["metrics"][variant][split] = {}
            for phase in ["all"] + sorted({r["phase"] for r in selected}):
                group = selected if phase == "all" else [r for r in selected if r["phase"] == phase]
                summary["metrics"][variant][split][phase] = action_metrics(
                    np.asarray([r["predicted"] for r in group])[:, :5],
                    np.asarray([r["target"] for r in group])[:, :5],
                ) | {
                    "observations": len(group),
                    "padding_final_rms_mean": float(
                        np.mean([r["padding_final_rms"] for r in group])
                    ),
                }
    write_json(args.output / "summary.json", summary)
    print("OFFLINE_SUMMARY", json.dumps(summary), flush=True)


def serve(args, protocol):
    import numpy as np
    import torch
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    from owac.training.pi05_sim import batch, make_model

    torch.set_num_threads(8)
    stats = json.loads((args.weights / "normalization.json").read_text())
    model = make_model(protocol, args.weights / "model.safetensors", dtype="bfloat16").cuda().eval()
    install_padding_bridge(model)
    tokenizer = PaligemmaTokenizer(200)

    class Policy:
        def infer(self, item):
            obs, _ = batch([item], stats, tokenizer, "cuda")
            generator = torch.Generator(device="cuda").manual_seed(
                int(item["seed"]) * 10000 + int(item["inference_index"])
            )
            noise = torch.randn(
                (1, protocol["action_horizon"], 32), generator=generator, device="cuda"
            )
            normalized, trace = predict(
                model, obs, protocol, noise, bridge=bool(item.get("padding_bridge"))
            )
            actions = (
                normalized[0, :, :7].cpu().numpy() * stats["actions"]["scale"]
                + stats["actions"]["center"]
            )
            return {"actions": np.asarray(actions, np.float32), "padding_residual_rms": trace}

    WebsocketPolicyServer(
        Policy(),
        host="127.0.0.1",
        port=args.port,
        metadata={
            "diagnostic": True,
            "model_sha256": file_sha256(args.weights / "model.safetensors"),
        },
    ).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline", "serve"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.mode == "offline":
        if args.data is None or args.output is None:
            parser.error("offline requires data and output")
        offline(args, protocol)
    else:
        serve(args, protocol)


if __name__ == "__main__":
    main()

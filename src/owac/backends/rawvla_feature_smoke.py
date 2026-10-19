"""One real NPZ batch through the pinned pretrained PI05 feature path.

Run in the existing Python 3.10 StarVLA runtime via PYTHONPATH; no benchmark jobs
or original visual path are modified. Weights/arrays remain on the server.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["libero", "robotwin"], required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    missing = [
        p for p in (args.npz, args.weights, args.stats, args.tokenizer) if not Path(p).is_file()
    ]
    if missing:
        print(json.dumps({"missing_assets": missing}))
        return 2
    import numpy as np
    import torch
    from safetensors.torch import load_file

    from owac.backends.rawvla import new_model, sha256_file
    from owac.backends.rawvla_features_n04 import (
        FeatureBridgeConfig,
        RawVLAFeaturePolicy,
        load_feature_audited,
        tokenize_backend_task,
        verify_gradient_path,
    )
    from owac.core.encoder_contract_n04 import MeasurementFrame
    from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
    from owac.frameworks.n04.model import N04Encoder
    from owac.raw.rawvla_sequence_n04 import RawVLASequenceLoader
    from owac.training.n04 import RawVLAFrame, build_real_context

    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    sequence = RawVLASequenceLoader(suite=args.suite).load(args.npz)
    stats = json.loads(Path(args.stats).read_text())
    if args.suite == "libero":
        stats = next(iter(stats.values()))
    else:
        stats = stats["norm_stats"]

    def normalize(value, key):
        item = stats[key]
        value = np.asarray(value, dtype=np.float32)
        if args.suite == "libero":
            q01, q99 = np.asarray(item["q01"]), np.asarray(item["q99"])
            value = (
                2
                * (value - q01[: value.shape[-1]])
                / (q99[: value.shape[-1]] - q01[: value.shape[-1]] + 1e-6)
                - 1
            )
        else:
            value = (value - np.asarray(item["mean"])) / (np.asarray(item["std"]) + 1e-8)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    model = new_model(args.suite, Path(args.tokenizer))
    audit = load_feature_audited(model, load_file(args.weights))
    model = model.to(device).eval()
    model.requires_grad_(False)
    height, width = sequence.measurements[sequence.camera_ids[0]].shape[-2:]
    cfg = N04Config(
        measurement_size=(height, width),
        camera_ids=sequence.camera_ids,
        geometry=N04GeometryConfig(feature_size=(height // 4, width // 4), pixel_stride=4),
        model=N04ModelConfig(
            width=32, hidden=64, blocks=2, scales=2, readout_tokens=256, readout_dim=2048
        ),
    )
    encoder = N04Encoder(cfg).to(device)
    policy = RawVLAFeaturePolicy(
        encoder,
        model,
        FeatureBridgeConfig(
            control_dim=7 if args.suite == "libero" else 14,
            action_horizon=10 if args.suite == "libero" else 50,
        ),
    )
    states = encoder.init_state(1, device=device)
    previous = None
    tokens = []
    frames = []
    for t in range(2):
        current = RawVLAFrame(
            measured=sequence.measurements[sequence.camera_ids[0]][t : t + 1].to(device),
            valid_mask=torch.ones(1, 1, height, width, device=device),
            saturated_mask=torch.zeros(1, 1, height, width, device=device),
            proprio=sequence.proprio[t : t + 1].to(device),
            action=torch.zeros(1, 1, device=device),
            metadata={"suite": args.suite, "camera": sequence.camera_ids[0]},
        )
        for camera in sequence.camera_ids:
            values = sequence.measurements[camera][t : t + 1].to(device)
            white = float(
                sequence.metadata.get(
                    "raw_white_level", 255 if args.suite == "libero" else float("inf")
                )
            )
            saturated = (values >= white).any(1, keepdim=True)
            values = values / 255 if args.suite == "libero" else values
            frame = MeasurementFrame(
                values=values,
                valid_mask=torch.ones_like(values[:, :1]),
                saturated_mask=saturated.float(),
                camera_id=camera,
                frame_ids=(t,),
                episode_ids=(sequence.episode_id,),
            )
            ctx = build_real_context(
                current,
                previous,
                camera_index=torch.tensor([sequence.camera_ids.index(camera)], device=device),
            )
            ctx.camera_id = camera
            visual, states[camera] = policy.encode_observation(frame, ctx, states[camera])
            if t == 1:
                tokens.append(visual)
                frames.append((frame, ctx))
        previous = current
    from owac.core.encoder_contract_n04 import VisualTokens

    visual = VisualTokens(
        torch.cat([v.values for v in tokens], 1), torch.cat([v.mask for v in tokens], 1)
    )
    # Combine independently encoded cameras into the actual language-width prefix.
    normalized_state = normalize(sequence.proprio[1].numpy(), "state")[None]
    language = sequence.metadata.get("task_description", sequence.metadata.get("task_name", ""))
    task, mask = tokenize_backend_task(model, language, normalized_state[0])
    task = torch.as_tensor(task, device=device)[None]
    mask = torch.as_tensor(mask, device=device)[None]
    prefix = policy.embed_prefix_from_visual_tokens(visual, task, mask)
    from owac.backends.rawvla_features_n04 import _FeatureBackendView

    view = _FeatureBackendView(model, prefix)
    observation = policy._observation(frames[0][1], task, mask, normalized_state)
    action_key = "action" if args.suite == "libero" else "actions"
    actions = normalize(sequence.supervised_action[1].numpy(), action_key)[None]
    actions = torch.nn.functional.pad(actions, (0, 32 - actions.shape[-1]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = type(model).forward_from_observation(view, observation, actions, return_debug=True)
    control_dim = policy.config.control_dim
    errors = (result["v_t"][..., :control_dim] - result["u_t"][..., :control_dim]).square()
    action_mask = sequence.supervised_action_mask[1:2].to(device)
    loss = (errors * action_mask[..., None]).sum() / (action_mask.sum() * control_dim).clamp_min(1)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_backward_ms = (time.perf_counter() - started) * 1000
    gradients = verify_gradient_path(
        policy,
        encoder_trainable_parameters=list(encoder.named_parameters()),
        forbidden_parameters=list(model.named_parameters()),
    )
    groups = {
        name: sum(v for k, v in gradients["gradient_norms"].items() if k.startswith(prefix))
        for name, prefix in {
            "propagation": "propagator.",
            "update": "decomposer.",
            "readout": "readout.",
        }.items()
    }
    if not gradients["ok"] or any(v <= 0 for v in groups.values()):
        raise RuntimeError(f"gradient path failed: {groups}")
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        sampled = type(model).sample_actions(view, observation, num_steps=10)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
    report = {
        "suite": args.suite,
        "data": sequence.summary(),
        "pretrained_weights": True,
        "input_metadata": sequence.metadata,
        "loss": float(loss.detach()),
        "gradient_groups": groups,
        "frozen_parameter_violations": gradients["frozen_parameter_violations"],
        "sample_shape": list(sampled.shape),
        "sample_finite": bool(sampled.isfinite().all()),
        "visual_shape": list(visual.values.shape),
        "visual_mask_coverage": float(visual.mask.float().mean()),
        "encoder_calls_for_two_observations": policy.counter.calls,
        "denoise_steps": 10,
        "forward_backward_ms": forward_backward_ms,
        "inference_ms": inference_ms,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "state_bytes": sum(s.state_bytes() for s in states.values()),
        "audit": audit,
        "hashes": {
            name: sha256_file(Path(path))
            for name, path in {
                "npz": args.npz,
                "weights": args.weights,
                "stats": args.stats,
                "tokenizer": args.tokenizer,
            }.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

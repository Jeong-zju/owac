"""Real NPZ -> causal N01 -> frozen pi0.5 flow training and standalone service.

Use the pinned Python 3.10 StarVLA runtime through PYTHONPATH. This module
never modifies or imports the running baseline snapshot. Training reports
fixture provenance explicitly; geometric labels are never fabricated for NPZs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from owac.backends.rawvla import new_model, sha256_file
from owac.backends.rawvla_features import (
    ACTION_PROTOCOLS,
    RawVLAFeaturePolicy,
    VisualPrefixConfig,
    VisualTokenPostprocessor,
    load_audited_with_replaced_vision,
)
from owac.core.encoder_contract import ComputeBudget, MeasurementFrame, VisualTokens
from owac.frameworks.n01.config import load_config
from owac.frameworks.n01.model import N01Encoder
from owac.raw.rawvla_sequence import RawVLASequenceDataset, collate_sequence_samples
from owac.training.n01 import git_sha


def action_masked_loss(velocity, target, mask, control_dim):
    """Exclude padded timesteps; keep the native 32-dim flow objective otherwise."""
    del control_dim
    if mask.shape != velocity.shape[:2]:
        raise ValueError("action mask must match the batch and horizon")
    weight = mask[:, :, None].expand_as(velocity)
    return ((velocity - target).square() * weight).sum() / weight.sum().clamp_min(1)


def observation_from_batch(policy, batch, device):
    """Keep pinned state padding/tokenization without entering an image pipeline."""
    state = batch["state"]
    if state is None:
        raise ValueError("real action training requires the protocol's current state")
    state = state.to(device).float()
    fill = policy.state_pad_value if policy.config.framework.discrete_state_input else 0.0
    state = F.pad(state, (0, policy.max_state_dim - state.shape[-1]), value=fill)[:, None]
    tokens, masks = [], []
    for b, meta in enumerate(batch["meta"]):
        language = meta["task_description"]
        if not language:
            raise ValueError("task language is missing; do not fabricate a prompt")
        token, mask = policy.tokenizer.tokenize(
            language, state[b, 0].cpu() if policy.config.framework.discrete_state_input else None
        )
        tokens.append(token)
        masks.append(mask)
    return SimpleNamespace(
        state=state,
        tokenized_prompt=torch.as_tensor(np.asarray(tokens), device=device),
        tokenized_prompt_mask=torch.as_tensor(np.asarray(masks), device=device),
    )


def encode_windows(encoder, batch, device, *, auxiliary=False, tbptt=4):
    """Independent state per camera/window; the same step is used online."""
    tokens, aux_losses, diagnostics = [], [], {}
    for camera, window in batch["windows"].items():
        values = window.values.to(device)
        b, length, _, h, w = values.shape
        state = encoder.init_state(b, {"camera_ids": [camera], "shape": (h, w)}, device)
        state.time_unit = window.time_unit
        # Strict measurement holdout from every input frame, including burn-in.
        holdout = torch.zeros(h, w, dtype=torch.bool, device=device)
        if auxiliary:
            holdout[h // 3 : h // 3 + 2, w // 3 : w // 3 + 2] = True
        for t in range(length):
            valid = window.valid_mask[:, t].to(device) & ~holdout
            metadata = {}
            if window.variance is not None:
                metadata["variance"] = window.variance[:, t].to(device)
            frame = MeasurementFrame(
                values=values[:, t].masked_fill(holdout[None, None], 0.0),
                valid_mask=valid,
                bad_mask=window.bad_mask[:, t].to(device),
                saturated_mask=window.saturated_mask[:, t].to(device),
                frame_ids=window.frame_ids[:, t].to(device),
                camera_id=camera,
                timestamp=window.timestamp[:, t].to(device),
                time_unit=window.time_unit,
                new_measurement=window.new_measurement[:, t].to(device),
                sample_xy=window.sample_xy,
                metadata=metadata,
            )
            output = encoder.step(
                frame,
                None,
                state,
                ComputeBudget(
                    max_updates=encoder.config.num_blocks, max_history=encoder.config.max_history
                ),
            )
            state = output.state
            if tbptt and (t + 1) % tbptt == 0 and t + 1 < length:
                state = encoder.detach_state(state)
        tokens.append(encoder.readout(output))
        diagnostics[camera] = output.diagnostics
        if auxiliary:
            prediction = encoder.predict_measurement(output)
            usable = window.valid_mask[:, -1].to(device) & ~window.bad_mask[:, -1].to(device)
            usable &= ~window.saturated_mask[:, -1].to(device)
            usable &= holdout[None, None]
            usable &= window.new_measurement[:, -1].to(device)[:, None, None, None]
            aux_losses.append(
                ((prediction - values[:, -1]).abs() * usable).sum() / usable.sum().clamp_min(1)
            )
    combined = VisualTokens(
        torch.cat([t.values for t in tokens], 1),
        torch.cat([t.mask for t in tokens], 1),
        [camera for t in tokens for camera in t.camera_ids],
        torch.cat([t.positions for t in tokens], 1),
    )
    auxiliary_loss = torch.stack(aux_losses).mean() if aux_losses else combined.values.sum() * 0
    return combined, auxiliary_loss, diagnostics


def load_backend(suite, checkpoint, tokenizer, device):
    from safetensors.torch import load_file

    policy = new_model(suite, tokenizer)
    audit = load_audited_with_replaced_vision(policy, load_file(str(checkpoint)))
    policy.to_precision("bfloat16")
    policy.to(device)
    policy.eval()
    return policy, audit


def build_bridge(policy, config, camera_order, device, method="n01", suite="libero"):
    if method == "n01":
        encoder = N01Encoder(config).to(device)
    else:
        from owac.baselines.raw_controls import RAWControlEncoder

        encoder = RAWControlEncoder(config, method).to(device)
    prefix_config = VisualPrefixConfig.from_readout_config(config, camera_order=camera_order)
    post = VisualTokenPostprocessor(prefix_config).to(device)
    bridge = RawVLAFeaturePolicy(
        policy,
        encoder=encoder,
        postprocessor=post,
        action_protocol=ACTION_PROTOCOLS["libero" if suite == "libero" else "robotwin"],
    )
    bridge.freeze_backend_parameters()
    return encoder, bridge


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    config = load_config(args.config)
    config = config.replace(
        selection="learned" if args.learned_support else "analytic",
        learnable_gate=args.learned_support,
        trainable_support=args.learned_support,
        aux_measurement_prediction=args.aux_weight > 0,
        # Unknown sensor noise is explicitly an assumed scale for this smoke.
        noise_read=args.assumed_variance,
    )
    ds_kwargs = json.loads(args.dataset_config.read_text()) if args.dataset_config else {}
    dataset = RawVLASequenceDataset(
        root=args.root,
        dataset_format=args.suite,
        window=args.window,
        fixture_provenance="real",
        **ds_kwargs,
    )
    policy, audit = load_backend(args.suite, args.weights, args.tokenizer, args.device)
    first = dataset[0]
    encoder, bridge = build_bridge(
        policy, config, list(first.windows), args.device, args.method, args.suite
    )
    params = [p for p in encoder.parameters() if p.requires_grad]
    params += [p for p in bridge.postprocessor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        if checkpoint["config"] != config.to_dict():
            raise ValueError("checkpoint configuration does not match this run")
        encoder.load_state_dict(checkpoint["encoder"])
        bridge.postprocessor.load_state_dict(checkpoint["postprocessor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
        torch.set_rng_state(checkpoint["rng"].cpu())
    if int(encoder.calibration_count) == 0:
        calibration = collate_sequence_samples([first])
        if args.crop:
            for window in calibration["windows"].values():
                for key in ("values", "valid_mask", "bad_mask", "saturated_mask", "variance"):
                    value = getattr(window, key)
                    if value is not None:
                        setattr(window, key, value[..., : args.crop, : args.crop])
        encoder.calibrating = True
        with torch.no_grad():
            encode_windows(encoder, calibration, args.device)
        encoder.calibrating = False
    args.output.mkdir(parents=True, exist_ok=True)
    dataset.write_split_manifest(args.output / "split.json")
    rows = []
    torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None
    before = {name: p.detach().clone() for name, p in encoder.named_parameters() if "stem" in name}
    for offset in range(args.steps):
        step = start + offset
        indices = [(step * args.batch_size + i) % len(dataset) for i in range(args.batch_size)]
        batch = collate_sequence_samples([dataset[i] for i in indices])
        if args.crop:
            # Disclosed top-left spatial crop; no resize, RGB conversion or remosaic.
            for window in batch["windows"].values():
                for key in ("values", "valid_mask", "bad_mask", "saturated_mask", "variance"):
                    value = getattr(window, key)
                    if value is not None:
                        setattr(window, key, value[..., : args.crop, : args.crop])
        tick = time.perf_counter()
        encoder.train()
        optimizer.zero_grad(set_to_none=True)
        tokens, auxiliary, diagnostics = encode_windows(
            encoder, batch, args.device, auxiliary=args.aux_weight > 0
        )
        obs = observation_from_batch(policy, batch, args.device)
        actions = batch["action"].to(args.device)
        actions = F.pad(actions, (0, policy.action_dim - actions.shape[-1]))
        result = bridge.forward_from_observation_with_features(
            obs, tokens, actions, return_debug=True
        )
        action_loss = action_masked_loss(
            result["v_t"],
            result["u_t"],
            batch["action_mask"].to(args.device),
            dataset.protocol.action_dim,
        )
        loss = action_loss + args.aux_weight * auxiliary
        loss.backward()
        grads = {"stem": 0.0, "readout": 0.0, "gate": 0.0, "selector": 0.0}
        for name, p in encoder.named_parameters():
            if p.grad is not None:
                for group, match in (
                    ("stem", "stem"),
                    ("readout", "readout_layer"),
                    ("gate", "integrator.edge_head"),
                    ("selector", "selector"),
                ):
                    if match in name:
                        grads[group] += float(p.grad.detach().norm())
        if (
            not torch.isfinite(loss)
            or (args.method != "c4_fixed_patch" and grads["stem"] <= 0)
            or grads["readout"] <= 0
        ):
            raise RuntimeError(f"invalid loss or broken encoder gradient: {grads}")
        if any(p.grad is not None for p in policy.parameters()):
            raise RuntimeError("frozen backend accumulated a parameter gradient")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        row = {
            "step": step + 1,
            "action_loss": float(action_loss.detach()),
            "aux_loss": float(auxiliary.detach()),
            "valid_actions": int(batch["action_mask"].sum()),
            "gradient_norms": grads,
            "wall_ms": (time.perf_counter() - tick) * 1000,
            "frames": [m["window_frames"] for m in batch["meta"]],
            "episodes": [m["episode_key"] for m in batch["meta"]],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    encoder.eval()
    with torch.inference_mode():
        tokens, _, _ = encode_windows(encoder, batch, args.device)
        noise = torch.randn_like(actions)
        encoder_calls_before = getattr(encoder, "calls", 0)
        denoise_before = bridge.stats["denoise_steps"]
        sampled = bridge.sample_actions_with_features(obs, tokens, noise=noise, num_steps=10)
        repeated = bridge.sample_actions_with_features(
            obs, tokens, noise=noise.clone(), num_steps=10
        )
    inference = {
        "shape": list(sampled.shape),
        "finite": bool(torch.isfinite(sampled).all()),
        "repeat_max_abs": float((sampled - repeated).abs().max()),
        "encoder_calls_during_denoising": getattr(encoder, "calls", 0) - encoder_calls_before,
        "actual_denoise_steps_two_calls": bridge.stats["denoise_steps"] - denoise_before,
        "denoising_steps": 10,
        "physical_control_executed": False,
    }
    if not inference["finite"] or inference["repeat_max_abs"] > 1e-5:
        raise RuntimeError(f"inference acceptance failed: {inference}")
    payload = {
        "encoder": encoder.state_dict(),
        "postprocessor": bridge.postprocessor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": start + args.steps,
        "rng": torch.get_rng_state(),
        "camera_order": list(first.windows),
        "suite": args.suite,
        "method": args.method,
        "weights_sha256": sha256_file(args.weights),
    }
    torch.save(payload, args.output / "checkpoint.pt")
    report = {
        "suite": args.suite,
        "method": args.method,
        "config": config.to_dict(),
        "git_sha": git_sha(),
        "source_manifest": str(Path(__file__).resolve().parents[3] / "source-manifest.json"),
        "source_manifest_sha256": sha256_file(
            Path(__file__).resolve().parents[3] / "source-manifest.json"
        )
        if (Path(__file__).resolve().parents[3] / "source-manifest.json").is_file()
        else None,
        "input_hashes": {
            "npz": sha256_file(Path(dataset.paths[0])),
            "statistics": sha256_file(dataset.stats_path),
            "tokenizer": sha256_file(args.tokenizer),
        },
        "rows": rows,
        "inference": inference,
        "load_audit": audit,
        "input_provenance": "author three-channel pseudo-RAW NPZ, no Bayer claim",
        "noise_provenance": "assumed constant variance, not calibrated sensor noise",
        "crop": args.crop,
        "data_spec": dataset.spec(),
        "encoder_parameter_update": sum(
            float((p - before[name]).abs().sum())
            for name, p in encoder.named_parameters()
            if name in before
        ),
        "backend_gradients": 0,
        "torch": torch.__version__,
        "python_runtime": __import__("sys").version,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()
        if args.device.startswith("cuda")
        else None,
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("/workspace/owac/rawvla"))
    p.add_argument("--suite", choices=("libero", "robotwin2"), required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--dataset-config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("n01", "c3_raw_cnn", "c4_fixed_patch"), default="n01")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--crop", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux-weight", type=float, default=0.0)
    p.add_argument("--assumed-variance", type=float, default=0.001)
    p.add_argument("--learned-support", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

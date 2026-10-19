"""Finite native 32-frame throughput and worst-content memory probes, outside main runs."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from owac.frameworks.n01_full.acceleration import configure_acceleration
from owac.frameworks.n01_full.model import N01V12Config, build_model
from owac.frameworks.n01_full.training_plan import BackendSchedule
from owac.raw.rawvla_sequence import collate_sequence_samples
from owac.training.n01_action import action_masked_loss, load_backend, observation_from_batch
from owac.training.n01_v12 import (
    _build_datasets,
    _motion_auxiliary,
    _policy_forward,
    _protocol_path,
    _suite,
    _v12_bridge,
    _weights_and_tokenizer,
    encode_windows_v12,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--deploy-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--pair-chunk", type=int, default=8)
    args = parser.parse_args(argv)
    config = N01V12Config.from_dict(json.loads(Path(args.config).read_text()))
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    dataset = _build_datasets(
        Path(args.repo_root),
        _protocol_path(Path(args.repo_root), None),
        config.platform,
        "train",
        32,
        config,
    )
    path = dataset.paths[0]
    arrays = dataset._load_episode(path)
    sample = dataset.build_sample(
        path,
        arrays,
        max(len(arrays["actions"]) - dataset.action_offset - dataset.action_horizon, 0),
    )
    model = build_model(config, device=device).train()
    weights, tokenizer = _weights_and_tokenizer(
        Path(args.repo_root), Path(args.deploy_root), config.platform
    )
    policy, audit = load_backend(_suite(config.platform), weights, tokenizer, device)
    bridge = _v12_bridge(model, policy, config)
    scope = BackendSchedule(model, policy, config.training, master_weights=True)
    scope.apply(config.training.warmup_steps)  # memory includes future backend gradients
    rows = []

    def measure(precision, batch_size, all_content, update=False):
        configure_acceleration(model, device, precision, pair_chunk=args.pair_chunk)
        model.encoder.support = None if all_content else model.support
        batch = collate_sequence_samples([sample] * batch_size)
        actions = batch["action"].to(device)
        actions = torch.nn.functional.pad(actions, (0, policy.action_dim - actions.shape[-1]))
        observation = observation_from_batch(policy, batch, device)
        optimizer = torch.optim.AdamW(scope.parameter_groups(), fused=True) if update else None
        probe = scope.backend[0][1]
        before = probe.detach().clone() if update else None
        times, losses = [], []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(2 if update else 1 if all_content else args.iterations):
            model.zero_grad(set_to_none=True)
            policy.zero_grad(set_to_none=True)
            torch.manual_seed(config.seed + iteration)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            tokens, _ = encode_windows_v12(model, batch, device)
            result = _policy_forward(bridge, observation, tokens, actions, return_debug=True)
            action_loss = action_masked_loss(
                result["v_t"],
                result["u_t"],
                batch["action_mask"].to(device),
                dataset.protocol.action_dim,
            )
            auxiliary = _motion_auxiliary(model, batch, device, iteration)
            loss = action_loss + config.training.motion_aux_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in scope.parameter_groups() for p in g["params"] if p.grad is not None],
                1,
                error_if_nonfinite=True,
            )
            if optimizer:
                optimizer.step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - started)
            losses.append(float(action_loss.detach()))
        row = {
            "precision": precision,
            "batch_size": batch_size,
            "all_content_stress": all_content,
            "seconds": times,
            "samples_per_second": batch_size / (sum(times) / len(times)),
            "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "action_losses": losses,
            "optimizer_updates": len(times) if update else 0,
            "finite_gradients": True,
            "backend_probe_dtype": str(probe.dtype),
            "backend_parameter_delta_max": float((probe.detach() - before).abs().max())
            if update
            else None,
        }
        return row

    for precision, size, all_content, update in [
        ("fp32", 1, False, False),
        ("bf16", 1, False, False),
        ("bf16", 2, False, False),
        ("bf16", 2, True, False),
        ("bf16", 2, True, True),
    ]:
        try:
            row = measure(precision, size, all_content, update)
        except torch.OutOfMemoryError as error:
            row = {
                "precision": precision,
                "batch_size": size,
                "all_content_stress": all_content,
                "oom": True,
                "error": str(error)[:300],
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
        model.zero_grad(set_to_none=True)
        policy.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        Path(args.output).write_text(
            json.dumps(
                {
                    "platform": config.platform,
                    "rows": rows,
                    "scope": "isolated performance probe; optimizer updates are discarded",
                    "same_real_sample_repeated_for_batch_memory_probe": True,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

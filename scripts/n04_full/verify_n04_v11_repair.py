"""Real-window forward/backward receipt; never runs an optimizer or writes weights."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from owac.raw.n04_v11_data import load_window_batch
from owac.training.n04_v11 import V11Trainer, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("libero", "robotwin"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config or f"configs/n04/v1_1/s2_launch_{args.platform}_D.json")
    trainer = V11Trainer(cfg, seed=101, run_dir=args.output, device="cuda:0")

    # A hard sentinel makes accidental updates fail, even while the user-stop marker exists.
    def forbidden(*args, **kwargs):
        raise AssertionError("verification must never update research weights")

    trainer.optimizer.step = forbidden
    trainer.scheduler.step = forbidden
    trainer.save_checkpoint = forbidden
    entry, start = trainer.dev_windows[0]
    window = load_window_batch(
        entry,
        start=start,
        burn_in=16,
        gradient=32,
        supervision_points=4,
        proprio_history=trainer.proprio_history,
    )
    versions = {n: p._version for n, p in trainer._trainable_parameters()}
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    print(
        json.dumps(
            {
                "phase": "forward_backward_only",
                "platform": args.platform,
                "source": str(entry.path),
                "frames": len(window),
            }
        ),
        flush=True,
    )
    trainer.encoder.train()
    trainer.backend.train()
    original_step = trainer.encoder.step
    observed = 0

    def traced_step(*values, **kwargs):
        nonlocal observed
        output = original_step(*values, **kwargs)
        observed += 1
        if observed % 4 == 0:
            print(
                json.dumps(
                    {"observed_frames": observed, "live_gb": torch.cuda.memory_allocated() / 1e9}
                ),
                flush=True,
            )
        return output

    trainer.encoder.step = traced_step
    trainer.optimizer.zero_grad(set_to_none=True)
    for repetition in range(args.repeat):
        trainer.optimizer.zero_grad(set_to_none=True)
        observed = 0
        torch.cuda.reset_peak_memory_stats()
        begun = time.perf_counter()
        loss, info = trainer._forward_window(window, train=True)
        torch.cuda.synchronize()
        print(
            json.dumps({"repetition": repetition, "seconds": time.perf_counter() - begun}),
            flush=True,
        )
    torch.cuda.synchronize()
    gradients = {}
    for name, parameter in trainer._trainable_parameters():
        assert parameter._version == versions[name], name
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all()), name
            prefix = name.split(".")[0:2]
            group = ".".join(prefix)
            record = gradients.setdefault(group, {"elements": 0, "abs_sum": 0.0})
            record["elements"] += parameter.numel()
            record["abs_sum"] += float(parameter.grad.float().abs().sum())
    assert all(info["terms"].get(n) is not None for n in trainer.loss_weights), info
    assert info["supervision_points"] == 4 and trainer.step == 0
    receipt = {
        "platform": args.platform,
        "source": str(entry.path),
        "window": [start, len(window)],
        "loss": float(loss),
        "info": info,
        "gradients": gradients,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "seconds": time.perf_counter() - begun,
        "optimizer_updates": 0,
        "parameter_versions_unchanged": True,
        "checkpoint_written": False,
        "weights": "fresh N04 encoder and official pretrained pi05; no encoder migration",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "real-window-verification.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

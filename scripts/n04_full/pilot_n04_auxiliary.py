"""Bounded real-data auxiliary-repair pilot, isolated from formal batch8 training."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from owac.training.n04_v11 import V11Trainer, load_config, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("pilot output must be new; do not overwrite prior diagnostics")
    cfg = load_config(args.config)
    cfg["owner"] = "Codex-diagnostic"
    cfg["optim"].update(batch=2, micro_batch=1, steps=args.updates, warmup_steps=4)
    cfg["monitor"].update(
        eval_every=4,
        eval_every_early=4,
        early_until=args.updates,
        checkpoint_every=4,
        keep_checkpoints=2,
    )
    args.output.mkdir(parents=True)
    write_json_atomic(args.output / "config.used.json", cfg)
    write_json_atomic(
        args.output / "diagnostic-identity.json",
        {
            "formal_training": False,
            "batch": 2,
            "warmup": 4,
            "updates": args.updates,
            "note": "Full architecture/window/objectives; isolated optimization-stability probe.",
        },
    )
    trainer = V11Trainer(cfg, seed=101, run_dir=args.output, device="cuda:0")
    try:
        trainer.run()
    except Exception as error:
        trainer._heartbeat("error", error=f"{type(error).__name__}: {error}")
        raise
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed = V11Trainer(cfg, seed=101, run_dir=args.output / "reload", device="cuda:0")
    resumed.load_checkpoint(args.output / "checkpoint-final.pt")
    result = resumed.evaluate_dev_action_loss()
    write_json_atomic(args.output / "reload-verification.json", result)


if __name__ == "__main__":
    main()

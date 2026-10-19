"""Exact configured initial training segment, independent reload and next update.

Unlike the historical pilots this does not change batch, warmup, learning rates,
data windows, losses or the 2000-update scheduler. A successful segment can be
continued using its checkpoint and all original receipts/metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from owac.training.n04_v11 import V11Trainer, load_config, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg["loss"].get("require_all_objectives"):
        raise ValueError("continuous verification requires mandatory objective coverage")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.jsonl").exists():
        raise FileExistsError("verification output already contains updates")
    (args.output / "train.pid").write_text(str(os.getpid()) + "\n")
    write_json_atomic(args.output / "config.used.json", cfg)
    receipt = {
        "platform": cfg["platform"],
        "configured_budget": cfg["optim"]["steps"],
        "batch": cfg["optim"]["batch"],
        "warmup": cfg["optim"]["warmup_steps"],
        "loss_weights": cfg["loss"]["weights"],
        "complete": False,
    }
    write_json_atomic(args.output / "continuous-verification.json", receipt)
    trainer = None
    try:
        trainer = V11Trainer(cfg, seed=cfg["seed"], run_dir=args.output)
        trainer.run(steps=args.updates)
        # The bounded run always writes checkpoint-final, while its numbered
        # saves and dev evaluations keep the configured cadence. Evaluate the
        # actual terminal weights and save them explicitly before reloading.
        before_dev = trainer.evaluate_dev_action_loss()["dev_loss"]
        if before_dev is None:
            raise AssertionError("continuous verification requires a fixed dev result")
        checkpoint = args.output / f"checkpoint-step{trainer.step:07d}.pt"
        trainer.last_checkpoint_step = trainer.step
        trainer.save_checkpoint(checkpoint)
        assert checkpoint.is_file()
        before_lr = trainer.scheduler.get_last_lr()
        receipt["checkpoint_step"] = trainer.step
        receipt["checkpoint"] = str(checkpoint)
        del trainer
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        trainer = V11Trainer(cfg, seed=cfg["seed"], run_dir=args.output)
        payload = trainer.load_checkpoint(checkpoint)
        assert trainer.step == args.updates
        assert trainer.scheduler.get_last_lr() == before_lr
        del payload
        after_dev = trainer.evaluate_dev_action_loss()["dev_loss"]
        if after_dev is None:
            raise AssertionError("reloaded model did not produce a fixed dev result")
        if abs(after_dev - before_dev) > 1e-5 * max(1, abs(before_dev)):
            raise AssertionError(f"reload changed fixed dev: {before_dev} -> {after_dev}")
        receipt["reload"] = {
            "step": trainer.step,
            "dev_before": before_dev,
            "dev_after": after_dev,
            "scheduler_restored": True,
        }
        write_json_atomic(args.output / "reload-verification.json", receipt["reload"])
        info = trainer.train_step(trainer.step)
        trainer._write_monitor(
            args.output / "metrics.jsonl", args.output / "progress.json", trainer.budget_steps, info
        )
        trainer.last_checkpoint_step = trainer.step
        resumed = args.output / f"checkpoint-step{trainer.step:07d}.pt"
        trainer.save_checkpoint(resumed)
        receipt["next_update"] = {
            "step": trainer.step,
            "terms": info["applied_terms"],
            "coverage": info["objective_coverage"],
            "forecast_diag": info["forecast_diag"],
        }
        receipt["resume_checkpoint"] = str(resumed)
        receipt["complete"] = True
        trainer._heartbeat(
            "diagnosing",
            validation_complete=True,
            note="Exact-config segment and independent reload passed; ready to continue",
        )
        write_json_atomic(args.output / "continuous-verification.json", receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        write_json_atomic(args.output / "continuous-verification.json", receipt)
        if trainer is not None:
            trainer._heartbeat("error", error=receipt["error"])
        raise


if __name__ == "__main__":
    main()

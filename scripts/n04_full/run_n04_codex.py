"""Codex-owned supervised training entry; reports initialization and failures truthfully."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from owac.training.n04_v11 import V11Trainer, load_config, write_json_atomic, write_progress


def resolve_resume(output: Path, resume: str | None) -> str | None:
    if resume != "latest":
        return resume
    candidates = [
        p
        for p in output.glob("checkpoint-step*.pt")
        if p.stem.removeprefix("checkpoint-step").isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint to resume in {output}")
    return str(max(candidates, key=lambda p: int(p.stem.removeprefix("checkpoint-step"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.pid").write_text(str(os.getpid()) + "\n")
    write_json_atomic(args.output / "config.used.json", cfg)

    def report(phase, error=None):
        write_progress(
            args.output / "progress.json",
            config=cfg,
            phase=phase,
            step=0,
            budget_steps=cfg["optim"]["steps"],
            activity={"error": error} if error else {},
        )

    report("initializing")
    trainer = None
    try:
        trainer = V11Trainer(cfg, seed=cfg.get("seed", 101), run_dir=args.output)
        write_json_atomic(
            args.output / "initialization.json",
            {
                "data": trainer.data_index.report(),
                "optimizer": trainer.optimizer_report,
                "memory": trainer.memory_settings(),
                "owner": "Codex",
            },
        )
        trainer.run(resume=resolve_resume(args.output, args.resume))
    except Exception as error:
        if trainer is None:
            report("error", f"{type(error).__name__}: {error}")
        else:
            trainer._heartbeat("error", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

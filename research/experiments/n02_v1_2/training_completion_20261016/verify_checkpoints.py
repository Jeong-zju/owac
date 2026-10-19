"""Read-only checkpoint audit: restore state, repeat fixed dev, never update weights."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["raw", "rgb"], required=True)
    args = parser.parse_args()
    receipt_dir = Path("outputs/n02-v12/completion-audit-20261016") / args.arm
    receipt_dir.mkdir(parents=True, exist_ok=False)
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                f"configs/encoders/n02_v12/{args.arm}_n02_robotwin.json",
                "--seed",
                "1103",
                "--max-updates",
                "1000",
                "--run-id",
                f"audit-{args.arm}-r5",
                "--output",
                str(receipt_dir / "runtime"),
            ]
        )
    )
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    windows = trainer.dev_batch()
    generator = torch.Generator(device=trainer.device).manual_seed(plan.seed + 17)
    horizon = int(plan.config["model"].get("action_horizon", 50))
    noise = torch.randn((len(windows), horizon, 32), generator=generator, device=trainer.device)
    times = (
        torch.rand((len(windows),), generator=generator, device=trainer.device).pow(1 / 1.5) * 0.999
        + 0.001
    )
    run = Path(f"outputs/n02-v12/P2/{args.arm}-n02-robotwin-r5-s1103")
    selected = [900, 1000] if args.arm == "raw" else [500, 1000]
    result = {"arm": args.arm, "optimizer_updates": 0, "checkpoints": []}
    for requested in selected:
        path = run / f"checkpoint_{requested:07d}.pt"
        step, cursor, payload = trainer.load_for_resume(path)
        assert step == requested
        nonfinite = []
        for section in ("model", "masked_branch"):
            nonfinite.extend(
                f"{section}.{name}"
                for name, value in payload[section].items()
                if value.is_floating_point() and not torch.isfinite(value).all()
            )
        assert not nonfinite, nonfinite
        after = trainer.evaluate_dev(windows, noise=noise, time_values=times, keep_prediction=True)
        before = payload["dev_reference"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        record = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "step": step,
            "cursor": cursor,
            "execution_revision": payload["execution_revision"],
            "training_contract_sha256": payload["training_contract"]["sha256"],
            "saved_dev_error": before["action_error"],
            "reloaded_dev_error": after["action_error"],
            "prediction_max_abs_difference": float(
                (after["prediction"] - before["prediction"]).abs().max()
            ),
            "per_window": after["per_window"],
            "anchor_identities": after["anchor_identities"],
            "nonfinite_weights": nonfinite,
        }
        result["checkpoints"].append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "anchor_identities"}), flush=True)
        (receipt_dir / "reload_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        del payload, before, after
        gc.collect()
    print("READ_ONLY_AUDIT_COMPLETE optimizer_updates=0", flush=True)


if __name__ == "__main__":
    main()

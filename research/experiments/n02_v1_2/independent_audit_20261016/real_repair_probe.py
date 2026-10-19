"""Bounded real-data/real-pi0.5 repair check. Never a capability or P3 run."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from owac.frameworks.n02_full.sequence_v12 import (
    EXECUTION_REVISION,
    replay_inputs,
    training_anchor_order,
    window_inputs,
)
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan

parser = argparse.ArgumentParser()
parser.add_argument("platform", choices=("robotwin", "libero"))
args = parser.parse_args()
root = Path.cwd()
output = root / "outputs/n02-v12/P2/independent-audit-20261016" / args.platform
plan = resolve_plan(
    build_parser().parse_args(
        [
            "--config",
            f"configs/encoders/n02_v12/raw_n02_{args.platform}.json",
            "--max-updates",
            "2",
            "--profile-updates",
            "2",
            "--validate-every",
            "1",
            "--previews",
            "1",
            "--output",
            str(output),
        ]
    )
)
plan.config["loop"]["dev_windows"] = 2
plan = dataclasses.replace(
    plan, run_id=f"repair-probe-{args.platform}", dev_windows=2, calibration_windows=2, grad_accum=1
)
trainer = N02V12Trainer(plan, LogTail())
started = time.time()
trainer.build()
trainer.checks = trainer.correctness_checks()
receipt = {
    "execution_revision": EXECUTION_REVISION,
    "platform": args.platform,
    "purpose": "bounded P2 repair check, not P3, not capability evidence",
}
if args.platform == "robotwin":
    checkpoint = root / "outputs/n02-v12/P2/raw-n02-robotwin-s1103/checkpoint_0001000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(payload["model"], strict=True)
    masked = trainer.masked.state_dict()
    masked.update(payload["masked_branch"])
    trainer.masked.load_state_dict(masked)
    receipt["initialization"] = {
        "path": str(checkpoint),
        "legacy_step": payload["step"],
        "note": "weights only; fresh optimizer and corrected step count",
    }
    del payload
    receipt["training"] = trainer.train()
    # Verify one stochastic suffix, including data permutation, loss, gradients and optimizer.
    saved = trainer.checkpoint_path(2)
    payload = torch.load(saved, map_location="cpu", weights_only=False)
    cursor = payload["cursor"]
    generator_state = payload["training_generator"]
    del payload
    anchor = training_anchor_order(trainer.sources["train"].anchor_count, plan.seed, cursor)[0]

    def suffix():
        generator = torch.Generator(device=trainer.device)
        generator.set_state(generator_state)
        trainer.optimizer.zero_grad(set_to_none=True)
        result = trainer.microbatch(anchor, anchor + 1, generator=generator)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in trainer.optimizer.param_groups for p in group["params"]],
            trainer.grad_clip,
            error_if_nonfinite=True,
        )
        trainer.optimizer.step()
        return {
            "loss": float(result["loss"].detach()),
            "tokens": trainer.token_source.last_output.tokens.detach().cpu(),
            "parameter": trainer.model.inference.slot_latents.detach().cpu().clone(),
        }

    first = suffix()
    trainer.load_for_resume(saved)
    second = suffix()
    receipt["stochastic_resume"] = {
        "anchor": anchor,
        "loss_delta": abs(first["loss"] - second["loss"]),
        "token_max_abs_delta": float((first["tokens"] - second["tokens"]).abs().max()),
        "updated_parameter_max_abs_delta": float(
            (first["parameter"] - second["parameter"]).abs().max()
        ),
        "saved_corrected_updates": 2,
        "extra_suffix_updates": "two copies of one probe suffix; neither saved as formal training",
    }
    trainer.load_for_resume(saved)
else:
    # Use a genuine interior history window, not the left-padded episode start.
    index = trainer.sources["train"].anchor_count // 2
    result = trainer.microbatch(
        index,
        index + 1,
        generator=torch.Generator(device=trainer.device).manual_seed(plan.seed + 1),
    )
    result["loss"].backward()
    stats = trainer.role_gradient_l1()
    receipt["forward_backward"] = {
        "anchor": index,
        "loss": float(result["loss"].detach()),
        "terms": result["terms"],
        "gradient_roles": stats,
        "info": result["info"],
    }
    receipt["optimizer_updates"] = 0
    receipt["previews"] = trainer.write_previews(result["batch"], result["output"])

batch = trainer.window(
    trainer.sources["dev"],
    trainer.sources["dev"].anchor_count // 2,
    trainer.sources["dev"].anchor_count // 2 + 1,
)
with torch.no_grad():
    train_tokens = trainer.token_source(batch).values.clone()
    replay = replay_inputs(trainer.model, window_inputs(batch))
    final = replay["rows"][-1]
    receipt["train_eval_token_max_abs_delta"] = float((train_tokens - final.tokens).abs().max())
    receipt["query_coverage"] = list(final.diagnostics["coverage"])
    receipt["queries_per_camera"] = list(final.diagnostics["per_camera_query_count"])
    receipt["history_measurements"] = len(final.state.geometry[0].entries)
    receipt["action_token_shape"] = list(train_tokens.shape)
    receipt["causal_prediction_present"] = final.causal is not None
    receipt["source_frame"] = replay["inputs"][-1].frame_index
receipt["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
receipt["elapsed_seconds"] = time.time() - started
receipt["closed_loop_completed"] = 0
(output / "repair_receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
trainer._write_json("checks.json", trainer.checks)
print(
    json.dumps(
        {
            key: value
            for key, value in receipt.items()
            if key not in ("training", "forward_backward")
        },
        indent=2,
    ),
    flush=True,
)

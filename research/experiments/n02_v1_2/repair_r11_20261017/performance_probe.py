"""Warm real-data B8/B12 forward/backward and bounded RAW fault injection; no updates."""

import argparse
import copy
import gc
import time
from pathlib import Path

import numpy as np
import torch

from owac.runtime.n02_v12_robotwin import Policy
from owac.runtime.rawvla_full_eval import write_json


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    policy = Policy(Path(args.config), Path(args.checkpoint), root / "policy")
    trainer = policy.trainer
    calibration = trainer.calibration_payload

    def forbidden(*a, **kw):
        raise RuntimeError("optimizer updates forbidden in performance probe")

    trainer.optimizer.step = forbidden
    timings = []
    for size, warmup in ((8, True), (8, False), (12, False)):
        trainer.model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.microbatch(
            10,
            10 + size,
            generator=torch.Generator(device=trainer.device).manual_seed(1140),
        )
        result["loss"].backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        finite = all(
            bool(torch.isfinite(p.grad).all())
            for group in trainer.parameter_groups.values()
            for p in group
            if p.grad is not None
        )
        assert finite
        timings.append(
            dict(
                batch=size,
                warmup=warmup,
                seconds=seconds,
                anchors_per_second=size / seconds,
                peak_bytes=torch.cuda.max_memory_allocated(),
                finite_gradients=finite,
            )
        )
        write_json(root / "timings.json", timings)
        del result
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.token_source.last_output = None
        trainer.token_source.last_outputs = []
        trainer.model.inference._last_state = None
        gc.collect()
        torch.cuda.empty_cache()
    trainer.model.eval()
    trainer.restore_fixed_calibration(calibration)
    batch = trainer.dev_batch()[1][0]
    physical_state = trainer._anchor_proprio(batch)[0] * (
        np.asarray(policy.stats["state"]["std"]) + 1e-8
    ) + np.asarray(policy.stats["state"]["mean"])
    history = {}
    for row, step in enumerate(batch.frame_index):
        images = [
            np.asarray(batch.observation.frames[camera].values[row])
            .transpose(1, 2, 0)
            .astype(np.float32)
            for camera in policy.cameras
        ]
        history[int(step)] = dict(step=int(step), image=images)
    example = dict(
        state=physical_state.astype(np.float32),
        task_name="grab_roller",
        lang="grab roller",
        case_id="r11-fault-injection",
        history=list(history.values()),
        step=int(batch.frame_index[-1]),
        flow_seed=188390828,
    )
    example = copy.deepcopy(example)
    example["history"][-1]["image"] = [im * 1000 for im in example["history"][-1]["image"]]
    example["image"] = example["history"][-1]["image"]
    result = policy.predict_action([example])
    output = trainer.token_source.last_output
    revision = output.diagnostics["revision"]
    assert (
        output.check.valid and output.check.status == "model_inadequate" and revision["triggered"]
    )
    assert revision["causal_verdict_preserved"] and revision["budget_per_measurement"] == 1
    write_json(
        root / "result.json",
        dict(
            optimizer_updates=0,
            new_closed_loops=0,
            timings=timings,
            timing_scope="one warm-cache microbatch per size; no optimizer step",
            fault=dict(
                scope="deliberate current-frame x1000 corruption, not sensor-noise robustness",
                check_status=output.check.status,
                check_valid=output.check.valid,
                revision=revision,
                finite_actions=bool(np.isfinite(result["actions"]).all()),
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())

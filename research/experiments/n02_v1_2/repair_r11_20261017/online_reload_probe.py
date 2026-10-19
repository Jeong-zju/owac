"""Restore the r11 step-zero diagnostic checkpoint through the real serving class."""

import argparse
import json
import time
from collections import Counter
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

    def forbidden(*a, **kw):
        raise RuntimeError("optimizer updates forbidden in serving acceptance")

    trainer.optimizer.step = forbidden
    windows = trainer.dev_batch()
    records = []
    for index in (0, 1, 2, 3):
        batch = windows[index][0]
        state = trainer._anchor_proprio(batch)[0]
        state = state * (np.asarray(policy.stats["state"]["std"]) + 1e-8) + np.asarray(
            policy.stats["state"]["mean"]
        )
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
            state=state.astype(np.float32),
            task_name="grab_roller",
            lang="grab roller",
            case_id=f"r11-serving-{index}",
            history=list(history.values()),
            image=list(history.values())[-1]["image"],
            step=int(batch.frame_index[-1]),
            flow_seed=188390828,
        )
        result = policy.predict_action([example])
        output = trainer.token_source.last_output
        assert result["execution_contract"] == trainer.plan.config["execution"]
        has_prefix = len(history) > 1
        assert bool(output.check.valid) == has_prefix
        records.append(
            dict(
                index=index,
                has_prefix=has_prefix,
                check=output.check.as_dict(),
                revision=output.diagnostics.get("revision"),
                finite=bool(np.isfinite(result["actions"]).all()),
                language=trainer.plan.config["policy"]["task_instructions"]["grab_roller"],
            )
        )
    parity = []
    with torch.inference_mode():
        for fast in (False, True):
            trainer.model.inference.fast_prefix = fast
            calls = Counter()
            handles = [
                m.register_forward_hook(lambda m, a, o, n=n, c=calls: c.update([n]))
                for n, m in (
                    ("readout", trainer.model.inference.readout),
                    ("decoder", trainer.model.inference.decoder),
                )
            ]
            torch.cuda.synchronize()
            started = time.perf_counter()
            visual = trainer.bridge.visual_tokens(windows[1][0])
            torch.cuda.synchronize()
            for handle in handles:
                handle.remove()
            output = trainer.token_source.last_output
            parity.append(
                dict(
                    tokens=visual.values.detach().cpu().clone(),
                    q=output.q.detach().cpu().clone(),
                    calls=dict(calls),
                    seconds=time.perf_counter() - started,
                    check=output.check.as_dict(),
                    revision=output.diagnostics.get("revision"),
                )
            )
    fast_prefix = dict(
        scope="eight distinct source frames at t10",
        token_max_delta=float((parity[0]["tokens"] - parity[1]["tokens"]).abs().max()),
        q_max_delta=float((parity[0]["q"] - parity[1]["q"]).abs().max()),
        calls=[p["calls"] for p in parity],
        seconds=[p["seconds"] for p in parity],
        checks=[p["check"] for p in parity],
    )
    assert fast_prefix["token_max_delta"] == 0 and fast_prefix["q_max_delta"] == 0
    trainer.model.inference.fast_prefix = True
    # A failed restore must remove, rather than leave installed, a stale threshold.
    old = trainer.calibration_payload
    with torch.no_grad():
        trainer.model.inference.readout.global_gain.add_(0.01)
    try:
        trainer.restore_fixed_calibration(old)
    except ValueError:
        assert trainer.model.inference.calibrator is None and trainer.calibration_payload is None
    else:
        raise AssertionError("stale threshold survived restore")
    write_json(
        root / "result.json",
        dict(
            optimizer_updates=0,
            new_closed_loops=0,
            checkpoint=args.checkpoint,
            rows=records,
            fast_prefix=fast_prefix,
            stale_restore_clears_threshold=True,
        ),
    )
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())

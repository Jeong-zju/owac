"""Instrument the official Arena loop, without replacing actions or success predicates."""

from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from owac.runtime.arena_droid import write_json


def array(value):
    if hasattr(value, "torch"):
        value = value.torch
    return value.detach().cpu().numpy()


def instrument(env, policy, original, *, num_steps, num_episodes):
    import numpy as np

    output = Path(os.environ["OWAC_ARENA_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=False)
    capture = None
    counts = {"control_steps": 0, "inference_calls": 0}
    started = time.monotonic()
    original_action = policy.get_action
    original_infer = policy._call_server_with_retry
    inference_times = []
    image_hashes = {}

    def infer(request):
        tick = time.monotonic()
        response = original_infer(request)
        inference_times.append(time.monotonic() - tick)
        counts["inference_calls"] += 1
        return response

    policy._call_server_with_retry = infer

    def get_action(env, obs):
        nonlocal capture
        step = counts["control_steps"]
        if os.environ.get("OWAC_ARENA_RAW") == "1":
            # Preserve exactly the official observation values across an extra render.
            obs = {
                **obs,
                "camera_obs": {key: value.clone() for key, value in obs["camera_obs"].items()},
            }
        images = {key: array(value) for key, value in obs["camera_obs"].items()}
        if all(int(value.max()) == 0 for value in images.values()):
            raise RuntimeError("All official RGB views are black; no policy score is valid")
        hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in images.items()}
        for key, digest in hashes.items():
            image_hashes.setdefault(key, set()).add(digest)
        if step >= 30 and all(len(values) == 1 for values in image_hashes.values()):
            raise RuntimeError("All RGB views remained frozen for 30 control steps")
        record = {
            "step": step,
            "episode_time_seconds": step * float(env.unwrapped.step_dt),
            "rgb_sha256": hashes,
            "joint_position": array(obs["policy"]["joint_pos"]).tolist(),
            "gripper_position": array(obs["policy"]["gripper_pos"]).tolist(),
        }
        if step == 0:
            write_json(
                output / "initial_observation.json",
                {
                    **record,
                    "cameras": {
                        key: {
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "minimum": int(v.min()),
                            "maximum": int(v.max()),
                        }
                        for key, v in images.items()
                    },
                    "scene_sensor_names": list(env.unwrapped.scene.sensors),
                    "step_dt": float(env.unwrapped.step_dt),
                },
            )
        if os.environ.get("OWAC_ARENA_RAW") == "1":
            from owac.simulation.arena_raw import ArenaRawCapture

            if capture is None:
                capture = ArenaRawCapture(env, output)
            record["raw"] = capture.capture(step)
            for key, value in obs["camera_obs"].items():
                if hashlib.sha256(array(value).tobytes()).hexdigest() != hashes[key]:
                    raise RuntimeError("RAW capture mutated an official RGB policy observation")
        actions = original_action(env, obs)
        values = array(actions)
        if values.shape != (1, 8) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid joint-position action shape/values: {values.shape}")
        record["action"] = values.tolist()
        with (output / "control.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        counts["control_steps"] += 1
        if step % 100 == 0:
            print("OWAC_ARENA_CONTROL", step, flush=True)
        return actions

    policy.get_action = get_action
    status = "running"
    error = None
    try:
        result = original(env, policy, num_steps=num_steps, num_episodes=num_episodes)
        status = "complete"
        return result
    except BaseException as failure:
        status = "infrastructure_failed"
        error = repr(failure)
        raise
    finally:
        write_json(
            output / "summary.json",
            {
                "status": status,
                "error": error,
                **counts,
                "completed_episode_count": int(env.unwrapped.completed_episode_count),
                "wall_seconds": time.monotonic() - started,
                "inference_seconds_p50": float(np.percentile(inference_times, 50))
                if inference_times
                else None,
                "inference_seconds_p95": float(np.percentile(inference_times, 95))
                if inference_times
                else None,
                "raw": capture.summary() if capture is not None else None,
                "distinct_rgb_frames": {key: len(values) for key, values in image_hashes.items()},
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    from isaaclab_arena.evaluation import experiment_runner, run_execution

    original = run_execution.rollout_policy

    def wrapped(env, policy, num_steps, num_episodes):
        return instrument(env, policy, original, num_steps=num_steps, num_episodes=num_episodes)

    run_execution.rollout_policy = wrapped
    experiment_runner.main()


if __name__ == "__main__":
    main()

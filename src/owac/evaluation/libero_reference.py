"""Observe the upstream LIBERO evaluator without changing its control loop.

This entry point also runs in the upstream simulator's Python 3.8 environment.
Only numerical records are exported. The upstream evaluator, transforms, state
selection, controller, settling steps and success predicate remain in use.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - upstream uses Python 3.8


class EpisodeObserver(logging.Handler):
    def __init__(self, root, expected):
        super().__init__()
        self.root = root
        self.expected = expected
        self.current = None
        self.rows = []
        self.task = {}
        self.environment = None
        self.task_index = -1
        self.task_episode = 0

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("\nTask: "):
            self.current = {
                **self.task,
                "episode_index": self.task_episode,
                "started_at_utc": utc_now(),
                "environment_steps": 0,
                "inference_ms": [],
                "error": None,
            }
        elif message.startswith("Caught exception:") and self.current is not None:
            self.current["error"] = message
        elif message.startswith("Success: ") and self.current is not None:
            self.current.update(
                success=message == "Success: True" and self.current["error"] is None,
                status="error" if self.current["error"] else "completed",
                completed_at_utc=utc_now(),
                control_steps=max(0, self.current["environment_steps"] - 10),
            )
            self.rows.append(self.current)
            with (self.root / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(self.current, allow_nan=False) + "\n")
            self.task_episode += 1
            self.current = None
            self.save_summary("running")

    def save_summary(self, status, error=None):
        import numpy as np

        errors = sum(row["status"] == "error" for row in self.rows)
        successes = sum(row["success"] for row in self.rows)
        latencies = [value for row in self.rows for value in row["inference_ms"]]
        summary = {
            "status": status,
            "updated_at_utc": utc_now(),
            "episodes": len(self.rows),
            "expected_episodes": self.expected,
            "successes": successes,
            "infrastructure_errors": errors,
            "success_rate": successes / len(self.rows) if self.rows and not errors else None,
            "inference_roundtrip_p50_ms": float(np.percentile(latencies, 50))
            if latencies
            else None,
            "inference_roundtrip_p95_ms": float(np.percentile(latencies, 95))
            if latencies
            else None,
            "error": error,
            "per_task": [
                {
                    "task_id": task_id,
                    "task_name": next(
                        row["task_name"] for row in self.rows if row["task_id"] == task_id
                    ),
                    "episodes": sum(row["task_id"] == task_id for row in self.rows),
                    "successes": sum(
                        row["success"] for row in self.rows if row["task_id"] == task_id
                    ),
                }
                for task_id in sorted({row["task_id"] for row in self.rows})
            ],
        }
        write_json(self.root / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=18010)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    specification = importlib.util.spec_from_file_location(
        "official_libero_eval", args.upstream_script
    )
    upstream = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = upstream
    specification.loader.exec_module(upstream)
    observer = EpisodeObserver(args.output, 10 * args.trials)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(observer)
    original_environment = upstream._get_libero_env
    original_client = upstream._websocket_client_policy.WebsocketClientPolicy

    def get_environment(task, resolution, seed):
        if observer.environment is not None:
            observer.environment.close()
        observer.task_index += 1
        observer.task_episode = 0
        observer.task = {
            "task_id": observer.task_index,
            "task_name": task.name,
            "task_description": task.language,
            "seed": seed,
        }
        environment, description = original_environment(task, resolution, seed)
        observer.environment = environment
        original_step = environment.step
        original_init = environment.set_init_state

        def step(action):
            observer.current["environment_steps"] += 1
            return original_step(action)

        def set_init_state(state):
            observer.current["initial_state_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
            return original_init(state)

        environment.step = step
        environment.set_init_state = set_init_state
        return environment, description

    def get_client(host, port):
        client = original_client(host, port)
        original_infer = client.infer

        def infer(observation):
            started = time.perf_counter()
            result = original_infer(observation)
            observer.current["inference_ms"].append(1000 * (time.perf_counter() - started))
            return result

        client.infer = infer
        return client

    # Only instrumentation and resource cleanup: no replacement policy/environment.
    upstream._get_libero_env = get_environment
    upstream._websocket_client_policy.WebsocketClientPolicy = get_client
    upstream.imageio.mimwrite = lambda *positional, **keyword: None
    started = time.monotonic()
    observer.save_summary("running")
    try:
        upstream.eval_libero(
            upstream.Args(
                host="127.0.0.1",
                port=args.port,
                task_suite_name=args.suite,
                num_trials_per_task=args.trials,
                seed=args.seed,
                video_out_path=str(args.output / "video_disabled"),
            )
        )
        if len(observer.rows) != observer.expected or any(row["error"] for row in observer.rows):
            raise RuntimeError(
                "Incomplete evaluation or infrastructure exception; score not accepted"
            )
        observer.save_summary("complete")
    except BaseException as error:
        observer.save_summary("failed", str(error))
        raise
    finally:
        if observer.environment is not None:
            observer.environment.close()
        write_json(args.output / "timing.json", {"wall_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()

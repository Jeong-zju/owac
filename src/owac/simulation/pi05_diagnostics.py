"""Replay demonstrations and trace frozen-model development rollouts in Isaac."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from owac.evaluation.baseline_protocol import file_sha256, write_json
from owac.simulation.lw_libero import build_environment, launch_simulator
from owac.simulation.pi05_benchmark import Benchmark


def run(args):
    import numpy as np
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = launch_simulator()
    env = None
    reports = []
    try:
        env = build_environment(0, demonstration=True)
        env.reset()
        bench = Benchmark(env, protocol)
        client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
        metadata = client.get_server_metadata()
        plans = [
            (2001, "demonstration_replay", 1, False),
            (2001, "original_h5", 5, False),
            (2001, "padding_bridge_h5", 5, True),
            (2001, "original_h1", 1, False),
            (2001, "padding_bridge_h1", 1, True),
            (2002, "original_h5", 5, False),
            (2002, "padding_bridge_h5", 5, True),
        ]
        for seed, name, horizon, bridge in plans:
            print("DIAGNOSTIC_CASE", seed, name, flush=True)
            started = time.perf_counter()
            bench.reset(seed)
            source = args.data / f"validation-{seed}.npz"
            with np.load(source, allow_pickle=False) as arrays:
                teacher_actions = arrays["actions"].copy()
            traces = []
            success = False
            inference_index = 0
            while bench.control_steps < protocol["episode_control_steps"]:
                obs = bench.observe()
                if name == "demonstration_replay":
                    frame = min(bench.control_steps // 2, len(teacher_actions) - 1)
                    actions = teacher_actions[frame : frame + 1]
                    padding_trace = []
                else:
                    result = client.infer(
                        obs
                        | {
                            "seed": seed,
                            "inference_index": inference_index,
                            "padding_bridge": bridge,
                        }
                    )
                    actions = result["actions"][:horizon]
                    padding_trace = result["padding_residual_rms"]
                    inference_index += 1
                if not np.isfinite(actions).all():
                    raise ValueError("Nonfinite diagnostic actions")
                trace = {
                    "control_steps_before": bench.control_steps,
                    "state": obs["state"].tolist(),
                    "actions": actions.tolist(),
                    "external_std": float(obs["external"].std()),
                    "wrist_std": float(obs["wrist"].std()),
                    "padding_support_residual_rms": padding_trace,
                }
                for action in actions:
                    bench.step(action)
                    if bench.stable >= 25:
                        success = True
                        break
                trace.update(metrics=bench.metrics())
                traces.append(trace)
                if success:
                    break
            report = {
                "seed": seed,
                "variant": name,
                "execute_horizon": horizon,
                "padding_bridge": bridge,
                "success": success,
                "status": "completed",
                "wall_seconds": time.perf_counter() - started,
                "inferences": inference_index,
                "demonstration_sha256": file_sha256(source),
                "initial_objects": {k: v.tolist() for k, v in bench.initial.items()},
            } | bench.metrics()
            write_json(args.output / f"{seed}-{name}.json", report)
            with (args.output / f"{seed}-{name}-trace.jsonl").open("w") as stream:
                for row in traces:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            reports.append(report)
            print("DIAGNOSTIC_RESULT", json.dumps(report), flush=True)
        write_json(
            args.output / "summary.json",
            {
                "status": "complete",
                "purpose": "development-only interventions, no final test seeds, no retraining",
                "protocol_sha256": file_sha256(args.protocol),
                "policy_metadata": metadata,
                "cases": reports,
            },
        )
    except Exception as error:
        write_json(args.output / "infrastructure-error.json", {"error": repr(error)})
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Wait for one declared repaired checkpoint; run the fixed RAW development episode once."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from owac.runtime.n02_v12_closed_loop import wait_server
from owac.runtime.rawvla_full_eval import paired_seed, rows, run_process, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("warm", "fresh"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    root = Path("/workspace/owac/rawvla")
    source = root / "source"
    run = f"raw-repair-{args.kind}-r7-s1103-a3"
    checkpoint = checkout / f"outputs/n02-v12/P2/{run}/checkpoint_{args.step:07d}.pt"
    case_id = f"{args.kind}-step{args.step:04d}"
    if args.attempt > 1:
        case_id += f"-attempt{args.attempt}"
    output = checkout / "outputs/n02-v12/repair-r7-20261016"
    case = output / case_id
    case.mkdir(parents=True, exist_ok=False)
    (case / "task_config").symlink_to(source / "third_party/RoboTwin/task_config")
    if not (case / "task_config/_camera_config.yml").is_file():
        raise FileNotFoundError("RoboTwin camera configuration is not reachable")
    demo = checkout / "outputs/n02-v12/demos/r5-lighting-20261016"
    protocol = json.loads((demo / "protocol.json").read_text())
    entry = next(e for e in protocol["entries"] if e["lighting_domain"] == "Normal")
    write_json(case / "entry.json", entry)
    deadline = time.monotonic() + 7200
    while not checkpoint.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(checkpoint))
        time.sleep(5)
    config = checkout / f"configs/encoders/n02_v12/raw_repair_{args.kind}_r7_robotwin.json"
    server_env = os.environ.copy()
    server_env["PYTHONPATH"] = f"{checkout}/src:{source}/starVLA"
    server_command = [
        str(checkout / "outputs/n02-full/runtime/py310-cu128/bin/python"),
        "-m",
        "owac.runtime.n02_v12_robotwin",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--evidence",
        str(case / "policy"),
        "--port",
        str(args.port),
    ]
    print(f"loading repaired RAW checkpoint {checkpoint}", flush=True)
    with (case / "policy.log").open("w") as log:
        server = subprocess.Popen(
            server_command, cwd=checkout, env=server_env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            wait_server(args.port, timeout=360)
            runtime = case / "runtime"
            runtime.mkdir()
            (runtime / "robotwin_eval.py").write_text(
                (demo / "runtime/robotwin_eval.py").read_text()
            )
            yml = (
                (demo / "runtime/raw.yml").read_text().replace("port: 18100", f"port: {args.port}")
            )
            (runtime / "policy.yml").write_text(yml)
            seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
            env = os.environ.copy()
            env.update(
                PYTHONPATH=f"{checkout}/src:{source}/starVLA:{source}/third_party/RoboTwin:"
                f"{source}/third_party/RoboTwin/script:{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
                PATH=f"{root}/envs/robotwin/bin:" + env["PATH"],
                CUDA_HOME="/usr/local/cuda-12.8",
                RAWVLA_ROBOTWIN_DENOISER="optix",
                ROBOTWIN_TEST_NUM="1",
                ROBOTWIN_EVAL_SEEDS=str(entry["dataset_seed"]),
                ROBOTWIN_FORCE_EVAL_SEED="1",
                ROBOTWIN_IMAGE_OBS_KEY="raw_linear",
                ROBOTWIN_LIGHTING_DOMAIN=entry["lighting_domain"],
                ROBOTWIN_LIGHTING_STRATEGY=entry["lighting_strategy"],
                ROBOTWIN_LIGHTING_SAMPLE_UNIT="paired",
                ROBOTWIN_LIGHTING_BENCHMARK_SEED=str(protocol["benchmark_seed"]),
                ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
                ROBOTWIN_POLICY_AUDIT_DIR=str(case),
                ROBOTWIN_POLICY_CHECKPOINT=str(checkpoint),
                RAWVLA_STEP_AUDIT=str(case / "actions.jsonl"),
                START_EPISODE=str(entry["dataset_episode_index"]),
                RAWVLA_FLOW_SEED=str(seed),
                RAWVLA_LANGUAGE_SEED=str(seed),
                RAWVLA_CASE_ID=case_id,
                RAWVLA_EXPECTED_ENTRY=json.dumps(entry),
                N02_CASE_VIDEO_DIR=str(case),
                VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
            )
            command = [
                str(root / "envs/robotwin/bin/python"),
                str(runtime / "robotwin_eval.py"),
                "--config",
                str(runtime / "policy.yml"),
                "--policy_ckpt_path",
                str(checkpoint),
                "--overrides",
                "--task_name",
                entry["task_name"],
                "--task_config",
                "owac_rawvla_clean",
                "--ckpt_setting",
                f"raw-r7-{case_id}",
                "--seed",
                "0",
                "--policy_name",
                "owac.runtime.n02_v12_robotwin",
            ]
            rc, seconds = run_process(
                command, source / "third_party/RoboTwin", env, case / "run.log", 900
            )
            result = {
                "case_id": case_id,
                "kind": args.kind,
                "step": args.step,
                "checkpoint": str(checkpoint),
                "returncode": rc,
                "wall_seconds": seconds,
                "entry": entry,
                "results": rows(case / "episode_results.jsonl"),
                "optimizer_updates_during_rollout": 0,
                "claim": "fixed development initial seed; no generalization or P4 claim",
            }
            write_json(case / "result.json", result)
            print(json.dumps(result), flush=True)
            if rc or len(result["results"]) != 1:
                raise RuntimeError("rollout infrastructure failure")
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()

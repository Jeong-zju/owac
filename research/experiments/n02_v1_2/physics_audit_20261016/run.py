"""Replay actual saved policy predictions for physics diagnosis, never for capability credit."""

import argparse
import json
import os
import re
from pathlib import Path

from owac.runtime.rawvla_full_eval import paired_seed, run_process, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("raw", "native", "raw-live", "raw-time"), required=True)
    args = p.parse_args()
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    root = Path("/workspace/owac/rawvla")
    source = root / "source"
    original = checkout / "outputs/n02-v12/policy-audit-20261016"
    repaired = checkout / "outputs/n02-v12/repair-r7-20261016"
    parent = (
        repaired / "warm-step0100-attempt2"
        if args.mode != "native"
        else original / "original-normal"
    )
    predictions = (
        parent / "policy/inference.jsonl"
        if args.mode != "native"
        else original / "original-policy/inference.jsonl"
    )
    case = checkout / f"outputs/n02-v12/physics-audit-20261016/{args.mode}-continued"
    case.mkdir(parents=True, exist_ok=False)
    (case / "task_config").symlink_to(source / "third_party/RoboTwin/task_config")
    runtime = case / "runtime"
    runtime.mkdir()
    code = (repaired / "warm-step0100-attempt2/runtime/robotwin_eval.py").read_text()
    code = code.replace(
        '"action_source": "model_prediction"',
        '"action_source": "recorded_model_action_replay_diagnostic"',
    )
    (runtime / "robotwin_eval.py").write_text(code)
    (runtime / "policy.yml").write_text(
        (parent / "runtime/policy.yml").read_text()
        if (parent / "runtime/policy.yml").exists()
        else (repaired / "warm-step0100-attempt2/runtime/policy.yml").read_text()
    )
    entry = json.loads((parent / "entry.json").read_text())
    seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{checkout}/src:{source}/starVLA:{source}/third_party/RoboTwin:{source}/third_party/RoboTwin/script:{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
        PATH=f"{root}/envs/robotwin/bin:" + env["PATH"],
        CUDA_HOME="/usr/local/cuda-12.8",
        RAWVLA_ROBOTWIN_DENOISER="optix",
        ROBOTWIN_TEST_NUM="1",
        ROBOTWIN_EVAL_SEEDS="0",
        ROBOTWIN_FORCE_EVAL_SEED="1",
        ROBOTWIN_IMAGE_OBS_KEY="raw_linear",
        ROBOTWIN_LIGHTING_DOMAIN="Normal",
        ROBOTWIN_LIGHTING_STRATEGY=entry["lighting_strategy"],
        ROBOTWIN_LIGHTING_SAMPLE_UNIT="paired",
        ROBOTWIN_LIGHTING_BENCHMARK_SEED="20261010",
        ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
        ROBOTWIN_POLICY_AUDIT_DIR=str(case),
        RAWVLA_STEP_AUDIT=str(case / "actions.jsonl"),
        START_EPISODE="0",
        RAWVLA_FLOW_SEED=str(seed),
        RAWVLA_LANGUAGE_SEED=str(seed),
        RAWVLA_CASE_ID=f"{args.mode}-physics-diagnostic",
        RAWVLA_EXPECTED_ENTRY=json.dumps(entry),
        N02_CASE_VIDEO_DIR=str(case),
        VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
        RAWVLA_REPLAY_SOURCE=str(predictions),
        RAWVLA_REPLAY_STATES=str(parent / "actions.jsonl"),
        RAWVLA_PHYSICS_DIR=str(case),
    )
    if args.mode == "raw-time":
        env["RAWVLA_PHYSICS_VIDEO"] = "1"
    # Keep the frozen protocol's exact lighting benchmark seed.
    protocol = json.loads(
        (checkout / "outputs/n02-v12/demos/r5-lighting-20261016/protocol.json").read_text()
    )
    env["ROBOTWIN_LIGHTING_BENCHMARK_SEED"] = str(protocol["benchmark_seed"])
    cmd = [
        str(root / "envs/robotwin/bin/python"),
        str(runtime / "robotwin_eval.py"),
        "--config",
        str(runtime / "policy.yml"),
        "--policy_ckpt_path",
        str(predictions),
        "--overrides",
        "--task_name",
        "grab_roller",
        "--task_config",
        "owac_rawvla_clean",
        "--ckpt_setting",
        f"physics-{args.mode}",
        "--seed",
        "0",
        "--policy_name",
        "owac.runtime.robotwin_physics_audit",
    ]
    if args.mode == "raw-live":
        from owac.runtime.n02_v12_closed_loop import wait_server

        wait_server(18330, timeout=360)
        env.update(
            RAWVLA_PHYSICS_LIVE="1",
            N02_REPLAN_INTERVAL="10",
            N02_MAX_JOINT_DELTA="0.05",
            N02_CONTROL_AUDIT=str(case / "control.jsonl"),
        )
        yml = runtime / "policy.yml"
        yml.write_text(re.sub(r"port: \d+", "port: 18330", yml.read_text()))
        checkpoint = (
            checkout / "outputs/n02-v12/P2/raw-repair-warm-r7-s1103-a3/checkpoint_0000100.pt"
        )
        cmd[cmd.index("--policy_ckpt_path") + 1] = str(checkpoint)
        env["ROBOTWIN_POLICY_CHECKPOINT"] = str(checkpoint)
        runtime_file = runtime / "robotwin_eval.py"
        runtime_file.write_text(
            runtime_file.read_text().replace(
                "recorded_model_action_replay_diagnostic", "live_raw_policy_with_bounded_execution"
            )
        )
    rc, sec = run_process(cmd, source / "third_party/RoboTwin", env, case / "run.log", 900)
    write_json(
        case / "run_result.json",
        {
            "returncode": rc,
            "seconds": sec,
            "diagnostic_only": args.mode != "raw-live",
            "prediction_source": str(checkpoint) if args.mode == "raw-live" else str(predictions),
        },
    )
    if rc:
        raise RuntimeError(f"physics replay failed: {case}")


if __name__ == "__main__":
    main()

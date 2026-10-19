"""One live RAW development rollout with contact, height and sustained-hold scoring."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from owac.runtime.n02_v12_closed_loop import prepare, wait_server
from owac.runtime.rawvla_full_eval import paired_seed, rows, run_process, write_json


def main(args):
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    root = Path("/workspace/owac/rawvla")
    source = root / "source"
    case = Path(args.output).resolve()
    config, checkpoint = Path(args.config).resolve(), Path(args.checkpoint).resolve()
    native = getattr(args, "native_reference", False)
    if not checkpoint.is_file() and not (native and checkpoint.is_dir()):
        raise FileNotFoundError(checkpoint)
    prepare(root, case, checkout)
    protocol = json.loads((case / "protocol.json").read_text())
    domain = getattr(args, "domain", "Normal")
    entry = next(e for e in protocol["entries"] if e["lighting_domain"] == domain)
    seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
    runtime = case / "runtime"
    yml = (runtime / "raw.yml").read_text()
    (runtime / "policy.yml").write_text(re.sub(r"port: \d+", f"port: {args.port}", yml))
    write_json(case / "entry.json", entry)
    write_json(
        case / "protocol.json",
        {
            "scope": "native RGB execution witness"
            if native
            else "one live RAW development episode, not P4",
            "entry": entry,
            "checkpoint": str(checkpoint),
            "config": str(config),
            "benchmark_seed": protocol["benchmark_seed"],
            "flow_seed": seed,
            "replan_interval": args.replan,
            "joint_delta_limit": args.joint_delta,
            "max_steps": args.max_steps,
            "timeout_seconds": args.timeout,
            "checkpoint_step": None if native else int(checkpoint.stem.removeprefix("checkpoint_")),
            "lighting_domain": domain,
            "flow_noise_mode": getattr(args, "flow_noise_mode", "independent"),
            "acceptance": (
                "actual two-hand contact, no table contact, z>=0.83m, continuous1s then passive2s"
            ),
            "video": "native RGB reference"
            if native
            else "render_rgb display only, simulation10fps; model input float32RAW3",
        },
    )
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{checkout}/src:{source}/starVLA",
        N02_REPLAN_INTERVAL=str(args.replan),
        N02_FLOW_NOISE_MODE=getattr(args, "flow_noise_mode", "independent"),
    )
    run_config = json.loads(config.read_text())
    if run_config.get("execution"):
        from owac.runtime.n02_control import ControlProtocol

        contract = ControlProtocol.parse(run_config["execution"])
        if args.replan != contract.replan_interval or args.joint_delta != contract.max_joint_delta:
            raise ValueError("rollout arguments differ from saved execution protocol")
        env["N02_EXECUTION_PROTOCOL"] = json.dumps(contract.as_dict())
    env.pop("N02_MAX_JOINT_DELTA", None)
    if args.joint_delta is not None:
        env["N02_MAX_JOINT_DELTA"] = str(args.joint_delta)
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
    if native:
        server_command = [
            str(checkout / "outputs/n02-full/runtime/py310-cu128/bin/python"),
            "-m",
            "owac.runtime.rawvla_robotwin",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(source / "openpi_converted_protocol/paligemma_tokenizer.model"),
            "--evidence",
            str(case / "policy"),
            "--port",
            str(args.port),
        ]
    with (case / "policy.log").open("w") as handle:
        server = subprocess.Popen(
            server_command, cwd=checkout, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        try:
            wait_server(args.port, timeout=360)
            env.update(
                PYTHONPATH=f"{checkout}/src:{source}/starVLA:{source}/third_party/RoboTwin:"
                f"{source}/third_party/RoboTwin/script:{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
                PATH=f"{root}/envs/robotwin/bin:" + env["PATH"],
                CUDA_HOME="/usr/local/cuda-12.8",
                RAWVLA_ROBOTWIN_DENOISER="optix",
                ROBOTWIN_TEST_NUM="1",
                ROBOTWIN_EVAL_SEEDS=str(entry["dataset_seed"]),
                ROBOTWIN_FORCE_EVAL_SEED="1",
                ROBOTWIN_IMAGE_OBS_KEY="render_rgb" if native else "raw_linear",
                ROBOTWIN_LIGHTING_DOMAIN=domain,
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
                RAWVLA_CASE_ID=case.name,
                RAWVLA_EXPECTED_ENTRY=json.dumps(entry),
                N02_CASE_VIDEO_DIR=str(case),
                VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
                RAWVLA_PHYSICS_LIVE="1",
                RAWVLA_PHYSICS_POLICY_KIND="native_rgb_policy" if native else "live_raw_policy",
                RAWVLA_PHYSICS_VIDEO="1",
                RAWVLA_PHYSICS_DIR=str(case),
                RAWVLA_PHYSICS_MAX_STEPS=str(args.max_steps),
                N02_CONTROL_AUDIT=str(case / "control.jsonl"),
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
                f"r8-{case.name}",
                "--seed",
                "0",
                "--policy_name",
                "owac.runtime.robotwin_physics_audit",
            ]
            rc, seconds = run_process(
                command, source / "third_party/RoboTwin", env, case / "run.log", args.timeout
            )
            physical_path = case / "summary.json"
            physical = json.loads(physical_path.read_text()) if physical_path.exists() else None
            result = {
                "case_id": case.name,
                "policy_kind": "native_rgb_reference" if native else "raw_n02",
                "checkpoint": str(checkpoint),
                "returncode": rc,
                "wall_seconds": seconds,
                "results": rows(case / "episode_results.jsonl"),
                "physical": physical,
                "accepted_success": physical.get("physical_accepted") if physical else None,
                "optimizer_updates_during_rollout": 0,
                "scope": f"fixed-seed {domain} development rollout",
                "lighting_domain": domain,
            }
            write_json(case / "result.json", result)
            if rc or physical is None:
                raise RuntimeError(f"rollout infrastructure failure: {case}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--replan", type=int, default=50)
    parser.add_argument("--joint-delta", type=float)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--native-reference", action="store_true")
    parser.add_argument(
        "--flow-noise-mode", choices=("independent", "fixed"), default="independent"
    )
    parser.add_argument(
        "--domain", choices=("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"), default="Normal"
    )
    main(parser.parse_args())

"""One same-condition normal-light rollout per reference path, without training."""

import argparse
import json
import os
from pathlib import Path

from owac.runtime.n02_v12_closed_loop import wait_server
from owac.runtime.rawvla_full_eval import paired_seed, rows, run_process, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("original", "bridge", "bridge-unpadded", "n02-raw-native", "n02-rgb-native"),
        required=True,
    )
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/workspace/owac/rawvla")
    source = root / "source"
    demo = args.checkout / "outputs/n02-v12/demos/r5-lighting-20261016"
    output = args.checkout / "outputs/n02-v12/policy-audit-20261016"
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "robotwin_eval.py").write_text((demo / "runtime/robotwin_eval.py").read_text())
    if not (output / "task_config").exists():
        try:
            (output / "task_config").symlink_to(source / "third_party/RoboTwin/task_config")
        except FileExistsError:
            pass
    protocol = json.loads((demo / "protocol.json").read_text())
    entry = next(e for e in protocol["entries"] if e["lighting_domain"] == "Normal")
    mode = args.mode
    port = {
        "original": 18200,
        "bridge": 18201,
        "bridge-unpadded": 18202,
        "n02-raw-native": 18203,
        "n02-rgb-native": 18204,
    }[mode]
    n02 = mode.startswith("n02-")
    arm = "raw" if mode == "n02-raw-native" else "rgb"
    checkpoint = (
        protocol["checkpoints"][arm]
        if n02
        else str(root / "checkpoints/pi05_robotwin/model.safetensors")
    )
    config = (demo / "runtime/raw.yml").read_text().replace("port: 18100", f"port: {port}")
    (runtime / f"{mode}.yml").write_text(config)
    case = output / f"{mode}-normal"
    case.mkdir(exist_ok=False)
    write_json(case / "entry.json", entry)
    seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{args.checkout}/src:{source}/starVLA:{source}/third_party/RoboTwin:"
        f"{source}/third_party/RoboTwin/script:"
        f"{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
        PATH=f"{root}/envs/robotwin/bin:" + env["PATH"],
        CUDA_HOME="/usr/local/cuda-12.8",
        RAWVLA_ROBOTWIN_DENOISER="optix",
        ROBOTWIN_TEST_NUM="1",
        ROBOTWIN_EVAL_SEEDS=str(entry["dataset_seed"]),
        ROBOTWIN_FORCE_EVAL_SEED="1",
        ROBOTWIN_IMAGE_OBS_KEY="raw_linear" if n02 else "render_rgb",
        ROBOTWIN_LIGHTING_DOMAIN=entry["lighting_domain"],
        ROBOTWIN_LIGHTING_STRATEGY=entry["lighting_strategy"],
        ROBOTWIN_LIGHTING_SAMPLE_UNIT="paired",
        ROBOTWIN_LIGHTING_BENCHMARK_SEED=str(protocol["benchmark_seed"]),
        ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
        ROBOTWIN_POLICY_AUDIT_DIR=str(case),
        ROBOTWIN_POLICY_CHECKPOINT=checkpoint,
        RAWVLA_STEP_AUDIT=str(case / "actions.jsonl"),
        START_EPISODE=str(entry["dataset_episode_index"]),
        RAWVLA_FLOW_SEED=str(seed),
        RAWVLA_LANGUAGE_SEED=str(seed),
        RAWVLA_CASE_ID=f"audit-{mode}-normal",
        RAWVLA_EXPECTED_ENTRY=json.dumps(entry),
        N02_CASE_VIDEO_DIR=str(case),
        VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
    )
    wait_server(port)
    command = [
        str(root / "envs/robotwin/bin/python"),
        str(runtime / "robotwin_eval.py"),
        "--config",
        str(runtime / f"{mode}.yml"),
        "--policy_ckpt_path",
        checkpoint,
        "--overrides",
        "--task_name",
        entry["task_name"],
        "--task_config",
        "owac_rawvla_clean",
        "--ckpt_setting",
        f"n02-policy-audit-{mode}",
        "--seed",
        "0",
        "--policy_name",
        "owac.runtime.n02_v12_robotwin" if n02 else "owac.runtime.rawvla_policy_audit",
    ]
    rc, seconds = run_process(command, source / "third_party/RoboTwin", env, case / "run.log", 900)
    result = {
        "mode": mode,
        "returncode": rc,
        "wall_seconds": seconds,
        "entry": entry,
        "results": rows(case / "episode_results.jsonl"),
        "optimizer_updates": 0,
        "weights_only_cross_revision_probe": n02,
        "policy_input_contract": "pi05-native-language-state-v1" if n02 else "native/reference",
    }
    write_json(case / "result.json", result)
    print(json.dumps(result), flush=True)
    if rc != 0 or len(result["results"]) != 1:
        raise RuntimeError(f"reference infrastructure failure: {case}")


if __name__ == "__main__":
    main()

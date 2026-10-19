"""Three one-state sensor captures; diagnostic holds are excluded from policy results."""

import json
import os
from pathlib import Path

from owac.runtime.n02_v12_closed_loop import prepare
from owac.runtime.rawvla_full_eval import paired_seed, run_process, write_json

checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
root = Path("/workspace/owac/rawvla")
output = checkout / "outputs/n02-v12/task-r9-noise-capture-a1"
prepare(root, output, checkout)
protocol = json.loads((output / "protocol.json").read_text())
for domain in ("Normal", "Low", "ExtremeLow"):
    entry = next(e for e in protocol["entries"] if e["lighting_domain"] == domain)
    case = output / domain
    case.mkdir()
    seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{checkout}/src:{root}/source/starVLA:{root}/source/third_party/RoboTwin:{root}/source/third_party/RoboTwin/script:{root}/source/starVLA/examples/simBenchmarks/Robotwin/eval_files",
        PATH=f"{root}/envs/robotwin/bin:" + env["PATH"],
        CUDA_HOME="/usr/local/cuda-12.8",
        VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
        RAWVLA_ROBOTWIN_DENOISER="optix",
        ROBOTWIN_TEST_NUM="1",
        ROBOTWIN_EVAL_SEEDS="0",
        ROBOTWIN_FORCE_EVAL_SEED="1",
        ROBOTWIN_IMAGE_OBS_KEY="raw_linear",
        ROBOTWIN_LIGHTING_DOMAIN=domain,
        ROBOTWIN_LIGHTING_STRATEGY=entry["lighting_strategy"],
        ROBOTWIN_LIGHTING_SAMPLE_UNIT="paired",
        ROBOTWIN_LIGHTING_BENCHMARK_SEED=str(protocol["benchmark_seed"]),
        ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
        ROBOTWIN_POLICY_AUDIT_DIR=str(case),
        START_EPISODE="0",
        RAWVLA_FLOW_SEED=str(seed),
        RAWVLA_LANGUAGE_SEED=str(seed),
        RAWVLA_CASE_ID=f"diagnostic-sensor-{domain}",
        RAWVLA_EXPECTED_ENTRY=json.dumps(entry),
        N02_CASE_VIDEO_DIR=str(case),
        N02_SENSOR_CAPTURE_DIR=str(case),
    )
    command = [
        str(root / "envs/robotwin/bin/python"),
        str(output / "runtime/robotwin_eval.py"),
        "--config",
        str(output / "runtime/raw.yml"),
        "--policy_ckpt_path",
        "sensor-diagnostic-no-policy",
        "--overrides",
        "--task_name",
        entry["task_name"],
        "--task_config",
        "owac_rawvla_clean",
        "--ckpt_setting",
        f"sensor-{domain}",
        "--seed",
        "0",
        "--policy_name",
        "owac.runtime.n02_sensor_capture",
    ]
    rc, seconds = run_process(
        command, root / "source/third_party/RoboTwin", env, case / "run.log", 240
    )
    write_json(
        case / "capture_run.json",
        dict(returncode=rc, seconds=seconds, diagnostic_only=True, entry=entry),
    )
    if rc or not (case / "sensor_capture.json").is_file():
        raise RuntimeError(f"sensor capture failed: {domain}")

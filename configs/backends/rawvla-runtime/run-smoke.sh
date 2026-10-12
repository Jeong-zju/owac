#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${RAWVLA_DEPLOY_ROOT:-/workspace/owac/rawvla}"
CODE="${DEPLOY_ROOT}/source"
RESULT="${RAWVLA_RESULT_ROOT:-${DEPLOY_ROOT}/records/recheck-$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTHONPATH="${DEPLOY_ROOT}/owac/src:${CODE}/starVLA:${CODE}/experiments/simulation/libero:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${CODE}/.local/libero"
export OPENPI_CONVERTED_ROOT="${CODE}/openpi_converted_protocol"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p "${RESULT}"
cd "${CODE}/starVLA"

case "${1:-libero-rgb}" in
  libero-rgb)
    for suite in libero_spatial libero_object libero_goal libero_10; do
      "${DEPLOY_ROOT}/envs/starvla/bin/python" \
        examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
        --model PI05 --model-source openpi --host 127.0.0.1 --port 18000 \
        --task-suite "${suite}" --task-start 0 --max-tasks 1 --num-trials 1 \
        --image-mode rgb --result-json "${RESULT}/${suite}.json" \
        --resume-path "${RESULT}/${suite}.episodes.jsonl"
    done
    ;;
  libero-lighting)
    "${DEPLOY_ROOT}/envs/starvla/bin/python" \
      examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py \
      --model PI05 --model-source openpi --host 127.0.0.1 --port 18000 \
      --task-suite libero_spatial --task-start 0 --max-tasks 1 \
      --image-mode rawvla_bench --rawvla-bench-representation default_isp \
      --rawvla-bench-manifest "${DEPLOY_ROOT}/records/libero-smoke-manifest-35.json" \
      --result-json "${RESULT}/libero-lighting.json" \
      --resume-path "${RESULT}/libero-lighting.episodes.jsonl"
    ;;
  robotwin-normal)
    export PATH="${DEPLOY_ROOT}/envs/robotwin/bin:${PATH}"
    export CUDA_HOME=/usr/local/cuda-12.8
    export RAWVLA_ROBOTWIN_DENOISER=optix
    export ROBOTWIN_POLICY_NAME=owac.runtime.rawvla_robotwin
    export ROBOTWIN_TEST_NUM=1
    export ROBOTWIN_EVAL_SEEDS=0
    export ROBOTWIN_IMAGE_OBS_KEY=default_isp
    export ROBOTWIN_LIGHTING_DOMAIN=Normal
    export ROBOTWIN_LIGHTING_STRATEGY=environment_only
    export ROBOTWIN_LIGHTING_SAMPLE_UNIT=paired
    export ROBOTWIN_POLICY_RECORD_EPISODE_DATA=1
    export ROBOTWIN_POLICY_TRAJECTORY_ROOT="${RESULT}/robotwin"
    timeout 600 bash examples/simBenchmarks/Robotwin/eval_files/eval.sh \
      grab_roller owac_rawvla_clean pi05-rawvla-normal 0 0 \
      "${DEPLOY_ROOT}/checkpoints/pi05_robotwin/model.safetensors" 18005 127.0.0.1
    ;;
  *)
    echo "Usage: $0 [libero-rgb|libero-lighting|robotwin-normal]" >&2
    exit 2
    ;;
esac

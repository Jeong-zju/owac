#!/usr/bin/env bash
# Use as a foreground supervisor command: run-on-gpu.sh raw|rgb <actual command>.
set -euo pipefail
role="${1:?Usage: run-on-gpu.sh raw|rgb command [args...]}"
shift
case "$role" in
  raw) export CUDA_VISIBLE_DEVICES=0 ;;
  rgb) export CUDA_VISIBLE_DEVICES=1 ;;
  *) printf 'Expected raw or rgb\n' >&2; exit 2 ;;
esac
DEPLOY_ROOT=/workspace/owac/rawvla
export N01_SOURCE="${N01_SOURCE:-$DEPLOY_ROOT/owac}"
export N01_PYTHON="${N01_PYTHON:-$DEPLOY_ROOT/envs/starvla/bin/python}"
export PYTHONPATH="$N01_SOURCE/src:$DEPLOY_ROOT/source/starVLA:$DEPLOY_ROOT/source/experiments/simulation/libero:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$DEPLOY_ROOT/source/.local/libero"
export OPENPI_CONVERTED_ROOT="$DEPLOY_ROOT/source/openpi_converted_protocol"
export HF_HOME=/workspace/.hf_home
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
export RAWVLA_ROBOTWIN_DENOISER="${RAWVLA_ROBOTWIN_DENOISER:-optix}"
printf 'role=%s physical_gpu=%s (inside process use cuda:0)\n' "$role" "$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name,memory.free --format=csv,noheader
exec "$@"

#!/usr/bin/env bash
set -euo pipefail
cd /workspace/owac/rawvla/research/n02-acceptance-20261012
export PYTHONPATH="$PWD/src:/workspace/owac/rawvla/source/starVLA"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
N02_SMOKE_OUTPUT="${N02_SMOKE_OUTPUT:-/workspace/owac/rawvla/research/n02-acceptance-20261012/outputs/reproduce}"
if [ -e "$N02_SMOKE_OUTPUT" ]; then
  printf "%s\n" "Choose a new N02_SMOKE_OUTPUT; refusing to overwrite existing evidence." >&2
  exit 2
fi
mkdir -p "$N02_SMOKE_OUTPUT"
/workspace/owac/rawvla/envs/starvla/bin/python -m compileall -q src
/workspace/owac/rawvla/envs/starvla/bin/python -m owac.training.n02_raw --config configs/encoders/n02/tiny.json --suite libero --checkpoint /workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors --tokenizer /workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model --data-root /workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/libero/raw --stats /workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/dataset_statistics.json --output "$N02_SMOKE_OUTPUT/libero" --stats-key openpi_pi05_libero_franka --steps 20 --inference-repeats 10 --window 2 --lr 0.0001 --seed 1 --threads 1 --device cuda
/workspace/owac/rawvla/envs/starvla/bin/python -m owac.training.n02_raw --config configs/encoders/n02/tiny.json --suite libero --checkpoint /workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors --tokenizer /workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model --data-root /workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/libero/raw --stats /workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/dataset_statistics.json --output "$N02_SMOKE_OUTPUT/libero-resume" --stats-key openpi_pi05_libero_franka --steps 21 --inference-repeats 10 --window 2 --lr 0.0001 --seed 1 --threads 1 --device cuda --resume "$N02_SMOKE_OUTPUT/libero/checkpoint.pt"
/workspace/owac/rawvla/envs/starvla/bin/python -m owac.training.n02_raw --config configs/encoders/n02/tiny.json --suite robotwin --checkpoint /workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors --tokenizer /workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model --data-root /workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/robotwin2/raw --stats /workspace/owac/rawvla/checkpoints/pi05_robotwin/policy_norm_stats.json --output "$N02_SMOKE_OUTPUT/robotwin" --stats-key '' --steps 20 --window 2 --lr 0.0001 --seed 1 --threads 1 --device cuda --inference-repeats 10

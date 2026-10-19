#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT=/workspace/owac/rawvla
PY_POLICY="$DEPLOY_ROOT/envs/starvla/bin/python"
uv pip install --python "$PY_POLICY" jax==0.5.3 jaxlib==0.5.3 flax==0.10.4 orbax-checkpoint==0.11.13 optax==0.2.4 ml-dtypes==0.4.1
mkdir -p "$DEPLOY_ROOT/openpi/examples"
curl -fL --retry 2 https://raw.githubusercontent.com/Physical-Intelligence/openpi/15a9616a00943ada6c20a0f158e3adb39df2ccac/examples/convert_jax_model_to_pytorch.py -o "$DEPLOY_ROOT/openpi/examples/convert_jax_model_to_pytorch.py"
export PYTHONPATH="$DEPLOY_ROOT/owac/src:$DEPLOY_ROOT/source/starVLA"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
"$PY_POLICY" -m owac.backends.rawvla convert-libero \
 "$DEPLOY_ROOT/checkpoints/pi05_libero_jax" \
 "$DEPLOY_ROOT/openpi/examples/convert_jax_model_to_pytorch.py" \
 "$DEPLOY_ROOT/source/openpi_converted_protocol/paligemma_tokenizer.model" \
 "$DEPLOY_ROOT/source/openpi_converted_protocol/pi05_libero_starvla"
"$PY_POLICY" "$DEPLOY_ROOT/source/experiments/simulation/robotwin/backbones/export_pi05_policy_norm_stats.py" --checkpoint-dir "$DEPLOY_ROOT/checkpoints/pi05_robotwin"

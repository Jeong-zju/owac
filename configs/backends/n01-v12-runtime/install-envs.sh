#!/usr/bin/env bash
# Run on the GPU server after reading /etc/vast-agents-guide.md.
set -euo pipefail
DEPLOY_ROOT=/workspace/owac/rawvla
PY_ENCODER="$DEPLOY_ROOT/envs/encoder/bin/python"
PY_POLICY="$DEPLOY_ROOT/envs/starvla/bin/python"
uv venv --python 3.11 "$DEPLOY_ROOT/envs/encoder"
uv pip install --python "$PY_ENCODER" \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY_ENCODER" numpy==2.4.6 pytest==9.1.1 \
  scipy pillow einops easydict timm safetensors matplotlib
uv venv --python 3.10 "$DEPLOY_ROOT/envs/starvla"
uv pip install --python "$PY_POLICY" \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY_POLICY" \
  -r "$DEPLOY_ROOT/owac/configs/backends/n01-v12-runtime/requirements-policy.txt"
uv pip install --python "$PY_POLICY" --no-deps \
  -e "$DEPLOY_ROOT/source/starVLA" \
  -e "$DEPLOY_ROOT/source/third_party/LIBERO" \
  -e "$DEPLOY_ROOT/source/third_party/openvla-oft"
uv pip freeze --python "$PY_ENCODER" > "$DEPLOY_ROOT/records/encoder-new-server-freeze.txt"
uv pip freeze --python "$PY_POLICY" > "$DEPLOY_ROOT/records/policy-new-server-freeze.txt"

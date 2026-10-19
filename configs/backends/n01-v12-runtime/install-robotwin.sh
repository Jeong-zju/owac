#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT=/workspace/owac/rawvla
PY_SIM="$DEPLOY_ROOT/envs/robotwin/bin/python"
uv venv --python 3.10 "$DEPLOY_ROOT/envs/robotwin"
uv pip install --python "$PY_SIM" torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY_SIM" \
  -r "$DEPLOY_ROOT/records/baseline-archive/remote/records/robotwin-requirements.txt" \
  'zarr==2.18.3' 'numpy==1.26.4' ninja wheel pyyaml-include==1.4.1
git clone https://github.com/NVlabs/curobo.git "$DEPLOY_ROOT/curobo"
git -C "$DEPLOY_ROOT/curobo" checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
uv pip install --python "$PY_SIM" --no-build-isolation -e "$DEPLOY_ROOT/curobo"
uv pip install --python "$PY_SIM" scipy==1.10.1 scikit-image==0.21.0 numpy==1.26.4 warp-lang==1.12.0
"$PY_SIM" "$DEPLOY_ROOT/owac/configs/backends/n01-v12-runtime/restore-runtime-compatibility.py"
uv pip freeze --python "$PY_SIM" > "$DEPLOY_ROOT/records/robotwin-new-server-freeze.txt"

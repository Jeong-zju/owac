#!/usr/bin/env bash
# Reproducible commands for N02 v1.1 P0 (2026-10-14).
#
# Target runtime is the isolated Python 3.10 + CUDA 12.8 environment; the root
# .venv is CPU-only and is used for static checks and the full regression suite.
# Every command below was actually executed for the P0 report; outputs land under
# outputs/n02-full/p0/ (gitignored). Nothing here starts a training run.
set -euo pipefail

cd /home/jeong/zeno/owac/worktrees/n02-measurement-completion
PY310=outputs/n02-full/runtime/py310-cu128/bin/python

# --- 1. static checks and full regression on the root CPU venv ---------------
make check
.venv/bin/python -m pytest -q

# --- 2. target runtime (Python 3.10 + torch 2.11.0+cu128) --------------------
export PYTHONPATH=src
"$PY310" -V
"$PY310" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY310" -m compileall -q src/owac
"$PY310" -m pytest tests/encoders/n02_full -q

# --- 3. runtime probe (frozen path read by the inventory subcommand) ---------
"$PY310" -m owac.runtime.n02_full_probe --device cuda --output outputs/n02-full/p0/runtime/probe.json
.venv/bin/python -m owac.runtime.n02_full_probe --device cpu --output outputs/n02-full/p0/runtime/probe_cpu.json

# --- 4. P0 CLI: inventory -> parity -> profile -> report ---------------------
CFG=configs/encoders/n02_full/p0.json
"$PY310" -m owac.evaluation.n02_full.p0 inventory --config "$CFG" --output outputs/n02-full/p0/inventory
"$PY310" -m owac.evaluation.n02_full.p0 parity    --config "$CFG" --output outputs/n02-full/p0/parity --device cpu
"$PY310" -m owac.evaluation.n02_full.p0 profile   --config "$CFG" --output outputs/n02-full/p0/profile --device cuda --max-updates 200
"$PY310" -m owac.evaluation.n02_full.p0 report    --config "$CFG" --output outputs/n02-full/p0/report

# --- 5. real RawVLA-Bench cache audit on the data host (read-only) -----------
# The cache lives on the data host, not on this workstation:
#   ssh -p 44327 root@107.209.104.125
#   /workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench   (705 GB, HF rev 8586c59...)
# Code is shipped with
#   tar czf - src/owac scripts configs/encoders/n02_full | ssh -p 44327 root@107.209.104.125 \
#       'tar xzf - -C /workspace/owac/n02-p0-audit'
# and runs under the host's own /venv/main/bin/python (numpy only), writing solely into
# /workspace/owac/n02-p0-audit/. Nothing on the data host is modified outside that directory and
# the frozen baseline processes are never touched.
#
#   cd /workspace/owac/n02-p0-audit
#   export PYTHONPATH=src
#   R=/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench
#   /venv/main/bin/python scripts/n02_p0_real_cache_audit.py --root "$R" --output outputs \
#       --hf-metadata /workspace/owac/rawvla/records/dataset-source-pinned.json --max-files 8 --curated-files 10
#   /venv/main/bin/python scripts/n02_p0_cache_metadata_scan.py --root "$R" --output outputs
#   for p in libero robotwin; do
#     /venv/main/bin/python -m owac.evaluation.n02_full.p0 inventory \
#       --config configs/encoders/n02_full/p0-remote-$p.json --output outputs/inventory_$p \
#       --max-files 8 --evidence-kind real_cache
#   done
#
# Receipts were pulled back into outputs/n02-full/p0/inventory-remote/ and archived under
# results/2026-10-14_n02_p0_foundation/real-cache/. The final stage decision combines the real
# inventory with the local parity/profile receipts (see outputs/n02-full/p0-bundle/).

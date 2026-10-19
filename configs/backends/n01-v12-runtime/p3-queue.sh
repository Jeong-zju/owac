#!/usr/bin/env bash
# N01 v1.2 P3 main training queue for one physical GPU.
#
# Usage (supervisor sets ROLE):  ROLE=raw|rgb owac-n01-v12-p3-queue.sh
#
# GPU0 (role=raw) runs the RAW arms, GPU1 (role=rgb) the ISP-RGB arms, one run at
# a time.  The current model (raw_n01 / rgb_n01) is queued first for each platform
# so the development loop can start against a freshly trained checkpoint.
#
# Every run resumes from its own latest.pt when one exists, so a restart of this
# program continues nearest checkpoint instead of overwriting from step 0.
#
# Working directory is /tmp on purpose: `python -m` puts the CWD first on
# sys.path, and /workspace/owac/rawvla/owac contains a stray `owac/` copy that
# would otherwise shadow the real src/ tree.
set -euo pipefail
role="${ROLE:?ROLE must be raw or rgb}"

case "$role" in
  # User decision (2026-10-15): concentrate GPU time on the current model.
  # Only the N01 arm runs per platform; the Base arms are deferred, and their
  # configurations stay on disk so they can be queued later without rebuilding
  # anything.  Four runs instead of eight.
  raw) gpu=0; queue="libero:raw_n01 robotwin2:raw_n01" ;;
  rgb) gpu=1; queue="libero:rgb_n01 robotwin2:rgb_n01" ;;
  *) printf 'ROLE must be raw or rgb, got %s\n' "$role" >&2; exit 2 ;;
esac

REPO="${N01_REPO:-/workspace/owac/rawvla/owac}"
DEPLOY="${N01_DEPLOY:-/workspace/owac/rawvla}"
N01_RUN="${N01_RUN_ROOT:-$DEPLOY/research/n01-v1/runs/v1-2-p3-complete}"
PY="${N01_PYTHON:-$DEPLOY/envs/starvla/bin/python}"
WRAP="${N01_GPU_WRAPPER:-$REPO/configs/backends/n01-v12-runtime/run-on-gpu.sh}"

# Registered P3 training budget for one run.  The measured throughput on this
# machine is recorded in the receipt; the value below is what this queue was
# actually started with, not a target that was reached.
MAX_STEPS="${MAX_STEPS:-0}"
CROP="${CROP:-0}"
ACCUM="${ACCUM:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PRECISION="${PRECISION:-bf16}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-8}"
PREFETCH="${PREFETCH:-2}"
TBPTT="${TBPTT:-8}"

export N01_SOURCE="$REPO"
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /tmp

printf 'queue=%s physical_gpu=%s budget_steps=%s crop=%s batch=%s accum=%s precision=%s flow_batch=%s tbptt=%s arms=%s\n' \
  "$role" "$gpu" "$MAX_STEPS" "$CROP" "$BATCH_SIZE" "$ACCUM" "$PRECISION" "$FLOW_BATCH_SIZE" "$TBPTT" "$queue"

# An independent audit hold must survive supervisor restarts and host reboots.
if [[ -f "$REPO/configs/encoders/n01_full/v1_2/TRAINING_HOLD.json" || -f "$N01_RUN/P3_AUDIT_HOLD.json" ]]; then
  printf "training is held by the user; queue cannot launch optimizer updates\n" >&2
  exit 42
fi

# Validate EVERY queued configuration before spending GPU time on the first item.
for item in $queue; do
  platform="${item%%:*}"
  arm="${item##*:}"
  config="$REPO/configs/encoders/n01_full/v1_2/complete/${platform}-${arm}-seed101.json"
  if [[ ! -f "$config" ]]; then
    printf 'config missing: %s\n' "$config" >&2
    exit 2
  fi
  "$PY" -c '
import json,sys
from pathlib import Path
from owac.frameworks.n01_full.model import N01V12Config
from owac.training.n01_v12 import _build_datasets,_protocol_path,_weights_and_tokenizer
config=N01V12Config.from_dict(json.load(open(sys.argv[1])))
repo,deploy=map(Path,sys.argv[2:4])
for region in ("train","validation"):
    dataset=_build_datasets(repo,_protocol_path(repo,None),config.platform,region,32,config)
    if not len(dataset): raise ValueError("empty "+region+" dataset")
_weights_and_tokenizer(repo,deploy,config.platform)
print("preflight:",config.platform,config.arm,"data and backend assets present",flush=True)
' "$config" "$REPO" "$DEPLOY"
done

failures=0
for item in $queue; do
  platform="${item%%:*}"
  arm="${item##*:}"
  config="$REPO/configs/encoders/n01_full/v1_2/complete/${platform}-${arm}-seed101.json"
  output="$N01_RUN/outputs/${platform}-${arm}-seed101"
  receipt="$N01_RUN/receipts/train-${platform}-${arm}-seed101.json"
  # On host reboot, advance past a fully completed, matching run. A checkpoint
  # alone is insufficient evidence of completion; unfinished runs resume below.
  if [[ -f "$receipt" ]] && "$PY" -c '
import json,sys
receipt,config=map(lambda p: json.load(open(p)),sys.argv[1:3])
budget=int(sys.argv[3]) or int(config["training"]["total_steps"])
ok=(receipt.get("config")==config and receipt.get("optimizer_steps",0)>=budget)
sys.exit(0 if ok else 1)
' "$receipt" "$config" "$MAX_STEPS"; then
    printf '=== COMPLETED %s; advancing queue\n' "$item"
    continue
  fi
  mkdir -p "$output"
  args=(train
    --repo-root "$REPO" --deploy-root "$DEPLOY"
    --config "$config" --run-root "$N01_RUN" --output "$output"
    --device cuda:0 --gpu-index "$gpu"
    --accum "$ACCUM" --tbptt "$TBPTT"
    --batch-size "$BATCH_SIZE" --precision "$PRECISION"
    --flow-batch-size "$FLOW_BATCH_SIZE" --prefetch "$PREFETCH"
    --crop "$CROP" --max-steps "$MAX_STEPS"
    --log-every 5 --ckpt-every 100 --val-every 250 --val-samples 8 --random-val-samples 4
    --initial-validation 1 --early-val-step 10)
  if [[ -f "$output/checkpoints/latest.pt" ]]; then
    args+=(--resume "$output/checkpoints/latest.pt")
    printf '=== RESUMING %s from %s\n' "$item" "$output/checkpoints/latest.pt"
  else
    printf '=== STARTING %s (no checkpoint yet)\n' "$item"
  fi
  # A single run failing must not abandon the rest of the queue.
  if ! bash "$WRAP" "$role" "$PY" -m owac.training.n01_v12 "${args[@]}"; then
    failures=$((failures + 1))
    printf '!!! run %s exited non-zero; continuing with the next queue item\n' "$item" >&2
  fi
done

# The queue is finite: exit so supervisor shows a settled status instead of
# pretending the work is still running.  A reboot restarts the program, which then
# resumes each run from its own latest.pt.
printf 'queue %s finished: failed_items=%s\n' "$role" "$failures"
if (( failures )); then exit 1; fi

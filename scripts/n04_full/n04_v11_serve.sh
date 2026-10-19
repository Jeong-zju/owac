#!/usr/bin/env bash
# N04 v1.1 策略推理服务：一条命令起服务（N04 编码器 + 冻结 π0.5 后端，官方 websocket 协议）。
#
# 用法（远端执行；平台、checkpoint、端口、设备、GPU 都可配）：
#   bash scripts/n04_full/n04_v11_serve.sh \
#     --checkpoint <run>/checkpoint-stepNNNNNNN.pt --platform libero --port 18000 --device cuda:0
#
# 与 WS-F 执行器对接（模板占位符 {checkpoint} {port} {device} {platform}）：
#   N04V11_POLICY_COMMAND='bash <SOURCE>/scripts/n04_full/n04_v11_serve.sh \
#       --checkpoint {checkpoint} --port {port} --device {device} --platform {platform}'
#
# 环境：
#   N04V11_ROOT     远端 N04 根目录（默认 /workspace/owac/research/n04-v1）
#   N04V11_SOURCE   源码目录（默认最新快照 $N04V11_ROOT/source/n04-v11-*，否则脚本所在 checkout）
#   N04V11_PYTHON   解释器（默认 $N04V11_ROOT/envs/n04v11/bin/python）
#   N04V11_GPU      物理 GPU 序号 → CUDA_VISIBLE_DEVICES（默认 0；与 --device cuda:N 无关）
#   N04V11_STARVLA  starVLA 源码根（默认 /workspace/owac/rawvla/source/starVLA）
#   N04V11_LOG_DIR  日志目录（默认 $N04V11_ROOT/runs/v1.1/logs）
#   N04V11_LOG      日志文件（默认 $LOG_DIR/n04-v11-policy-<platform>.log，固定路径）
#   N04V11_GPU_MEMORY_FRACTION  可选显存上限（设备总量占比，如 0.2）；与训练共存时超预算先让自己 OOM
#   N04V11_CHECKPOINT / N04V11_PLATFORM / N04V11_PORT / N04V11_DEVICE  等价命令行开关
# 端口约定：LIBERO 18000、RoboTwin 18005，只绑 127.0.0.1，验证后可随时关闭。
set -euo pipefail

ARGS=()
PLATFORM=""
CHECKPOINT=""
PORT=""
DEVICE=""
HOST=""
GPU_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="${2:-}"; ARGS+=("$1" "$2"); shift 2 ;;
    --platform=*) PLATFORM="${1#*=}"; ARGS+=("$1"); shift ;;
    --checkpoint) CHECKPOINT="${2:-}"; ARGS+=("$1" "$2"); shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; ARGS+=("$1"); shift ;;
    --port) PORT="${2:-}"; ARGS+=("$1" "$2"); shift 2 ;;
    --port=*) PORT="${1#*=}"; ARGS+=("$1"); shift ;;
    --device) DEVICE="${2:-}"; ARGS+=("$1" "$2"); shift 2 ;;
    --device=*) DEVICE="${1#*=}"; ARGS+=("$1"); shift ;;
    --host) HOST="${2:-}"; ARGS+=("$1" "$2"); shift 2 ;;
    --gpu) GPU_OVERRIDE="${2:-}"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

ROOT=${N04V11_ROOT:-/workspace/owac/research/n04-v1}
SOURCE=${N04V11_SOURCE:-}
if [[ -z "$SOURCE" ]]; then
  SOURCE=$(ls -dt "$ROOT"/source/n04-v11-* 2>/dev/null | head -1 || true)
fi
if [[ -z "$SOURCE" ]]; then
  SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
PY=${N04V11_PYTHON:-$ROOT/envs/n04v11/bin/python}
STARVLA=${N04V11_STARVLA:-/workspace/owac/rawvla/source/starVLA}
LOG_DIR=${N04V11_LOG_DIR:-$ROOT/runs/v1.1/logs}

PLATFORM=${PLATFORM:-${N04V11_PLATFORM:-libero}}
CHECKPOINT=${CHECKPOINT:-${N04V11_CHECKPOINT:-}}
DEVICE=${DEVICE:-${N04V11_DEVICE:-cuda:0}}
HOST=${HOST:-${N04V11_HOST:-127.0.0.1}}
if [[ -z "$PORT" ]]; then
  PORT=${N04V11_PORT:-}
fi
if [[ -z "$PORT" ]]; then
  if [[ "$PLATFORM" == "robotwin" ]]; then PORT=18005; else PORT=18000; fi
fi
if [[ -z "$CHECKPOINT" ]]; then
  echo "需要 --checkpoint（或 N04V11_CHECKPOINT）：N04 v1.1 的 checkpoint-step*.pt 文件" >&2
  exit 2
fi
# 环境变量给的开关补进命令行（命令行显式给出的优先，不重复追加）。
case " ${ARGS[*]:-} " in *" --platform "*) ;; *) ARGS+=(--platform "$PLATFORM") ;; esac
case " ${ARGS[*]:-} " in *" --port "*) ;; *) ARGS+=(--port "$PORT") ;; esac
case " ${ARGS[*]:-} " in *" --device "*) ;; *) ARGS+=(--device "$DEVICE") ;; esac
case " ${ARGS[*]:-} " in *" --checkpoint "*) ;; *) ARGS=(--checkpoint "$CHECKPOINT" "${ARGS[@]:-}") ;; esac

GPU=${GPU_OVERRIDE:-${N04V11_GPU:-0}}
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SOURCE/src:$STARVLA:$STARVLA/examples/simBenchmarks/LIBERO/eval_files/openpi${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

LOG=${N04V11_LOG:-$LOG_DIR/n04-v11-policy-$PLATFORM.log}
mkdir -p "$LOG_DIR"
{
  echo "[$(date -Is)] start platform=$PLATFORM port=$PORT device=$DEVICE gpu=$CUDA_VISIBLE_DEVICES"
  echo "[$(date -Is)] source=$SOURCE python=$PY"
  echo "[$(date -Is)] command=$PY -m owac.backends.n04_v11_server ${ARGS[*]}"
} >>"$LOG"

cd "$SOURCE"
exec "$PY" -m owac.backends.n04_v11_server "${ARGS[@]}" >>"$LOG" 2>&1

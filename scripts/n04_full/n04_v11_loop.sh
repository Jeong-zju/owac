#!/usr/bin/env bash
# N04 v1.1 S2 每模型 100 次闭环：远端一条命令跑一个平台的 50 条（tmux + 日志 + 断点续跑）。
#
# 用法（在远端执行；脚本自带的仓根即源码，也可用 N04V11_SOURCE 指向冻结快照）：
#   N04V11_CHECKPOINT=<run>/checkpoint-stepNNNNNNN.pt \
#   N04V11_POLICY_COMMAND='env PYTHONPATH=... <python> -m <n04 v1.1 策略服务> \
#       --checkpoint {checkpoint} --port {port}' \
#   bash scripts/n04_full/n04_v11_loop.sh libero 0
#
# 参数： <platform: libero|robotwin|all> [gpu index] [额外参数…（透传给执行器）]
# 环境：
#   N04V11_ROOT        远端 N04 根目录（默认 /workspace/owac/research/n04-v1）
#   N04V11_SOURCE      源码目录（默认脚本所在 checkout 根；冻结批次用快照目录）
#   N04V11_RUN         run 目录（默认 $N04V11_ROOT/runs/v1.1/s2/<platform>-<arm>-seed<seed>）
#   N04V11_CHECKPOINT  待评 checkpoint（必需；写进结果行并传给策略服务）
#   N04V11_POLICY_COMMAND 策略服务启动模板（必需；{checkpoint} {port} {device} {platform} 占位）
#   N04V11_ARM / N04V11_SEED / N04V11_CONFIG  覆盖默认 arm=D、seed=101、config=runs/<platform>_<arm>.json
#   N04V11_EVAL_LIST / N04V11_PER_DOMAIN  覆盖默认 eval100.json 与每域期望条数（正式批次保持 10）
#   N04V11_PROPRIO_MODE（exact|approx，默认 exact）本体历史口径
#   N04V11_IMAGE_REPRESENTATION（默认 raw） / N04V11_RESIZE_SIZE（256） /
#   N04V11_ROBOTWIN_IMAGE_KEY（默认 raw_linear）  相机与缩放口径
#   N04V11_FORCE=1     重跑已结算身份（默认断点续跑；已完成的 identity 不重复执行）
#   N04V11_EXTRA       追加给执行器的参数（例如 --limit 2 --no-video）
#
# 会话： tmux 会话 n04v11-loop-<platform>-<arm><seed>
# 日志： <run>/logs/loop-<platform>.log（同时 tee 到 tmux 面板）
# 查看： tmux attach -t n04v11-loop-<platform>-<arm><seed>
#        tail -f <run>/logs/loop-<platform>.log
#        python3 -m json.tool <run>/progress.json        # 监控页同源计数
#        wc -l <run>/eval_results.jsonl                  # 逐条结果
# 停止： tmux kill-session -t <session>   （下次启动自动从 eval_results.jsonl 续跑）
set -euo pipefail

PLATFORM=${1:?platform: libero|robotwin|all}
GPU=${2:-0}
shift 2 2>/dev/null || true

ROOT=${N04V11_ROOT:-/workspace/owac/research/n04-v1}
SOURCE=${N04V11_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ARM=${N04V11_ARM:-D}
SEED=${N04V11_SEED:-101}
PY=${N04V11_PYTHON:-$ROOT/envs/n04v11/bin/python}

if [[ "$PLATFORM" == "all" ]]; then
  CONFIG=${N04V11_CONFIG:-$SOURCE/configs/n04/v1_1/runs/libero_${ARM}.json}
  RUN=${N04V11_RUN:-$ROOT/runs/v1.1/s2/${ARM}-seed${SEED}-loop100}
else
  CONFIG=${N04V11_CONFIG:-$SOURCE/configs/n04/v1_1/runs/${PLATFORM}_${ARM}.json}
  RUN=${N04V11_RUN:-$ROOT/runs/v1.1/s2/${PLATFORM}-${ARM}-seed${SEED}}
fi
SESSION=n04v11-loop-${PLATFORM}-${ARM}${SEED}
LOG=$RUN/logs/loop-${PLATFORM}.log

CKPT=${N04V11_CHECKPOINT:?请设置 N04V11_CHECKPOINT=<待评 checkpoint 路径>}
EVAL_LIST=${N04V11_EVAL_LIST:-$SOURCE/configs/n04/v1_1/eval100.json}
PER_DOMAIN=${N04V11_PER_DOMAIN:-10}
# N04 臂 D 要 RAW（服务端按平台缩放）；官方 π0.5 对照用 default_isp。
IMAGE_REPRESENTATION=${N04V11_IMAGE_REPRESENTATION:-raw}
# LIBERO 训练帧 256（作者默认 224 会与服务端 training_obs_image_size 不一致）。
RESIZE_SIZE=${N04V11_RESIZE_SIZE:-256}
# RoboTwin 相机模态：臂 D 用 raw_linear（float32 线性 RAW）；π0.5 对照用 render_rgb。
ROBOTWIN_IMAGE_KEY=${N04V11_ROBOTWIN_IMAGE_KEY:-raw_linear}
# 本体历史口径：exact=客户端带最近 3 帧真实观测；approx=不带，服务端退回策略调用近似。
PROPRIO_MODE=${N04V11_PROPRIO_MODE:-exact}
case "$PROPRIO_MODE" in
  exact) PROPRIO_EXACT=1 ;;
  approx) PROPRIO_EXACT=0 ;;
  *) echo "N04V11_PROPRIO_MODE 只能是 exact 或 approx" >&2; exit 2 ;;
esac
# 注意：${VAR:?...} 的提示语里不能出现花括号（会被当成参数展开的结束符）。
POLICY_COMMAND=${N04V11_POLICY_COMMAND:?请设置 N04V11_POLICY_COMMAND=策略服务启动模板，含 checkpoint 与 port 占位符}

mkdir -p "$RUN/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "会话 $SESSION 已存在；不重复启动。查看： tmux attach -t $SESSION" >&2
  exit 1
fi

FORCE_FLAG=()
[[ "${N04V11_FORCE:-0}" == "1" ]] && FORCE_FLAG+=(--force)
EXTRA=()
[[ -n "${N04V11_EXTRA:-}" ]] && read -r -a EXTRA <<<"${N04V11_EXTRA}"
EXTRA+=("$@")

LAUNCHER=$RUN/logs/loop-${PLATFORM}.sh
cat > "$LAUNCHER" <<SH
#!/usr/bin/env bash
set -euo pipefail
cd "$SOURCE"
export PYTHONPATH="$SOURCE/src\${PYTHONPATH:+:\$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$GPU
export RAWVLA_N04_PROPRIO_EXACT=$PROPRIO_EXACT
# 协议 B 客户端（owac.runtime.rawvla_robotwin）的精确本体历史开关，同口径联动。
export RAWVLA_PROPRIO_HISTORY=$PROPRIO_EXACT
echo "[\$(date -u +%FT%TZ)] N04 v1.1 闭环 platform=$PLATFORM gpu=$GPU run=$RUN"
"$PY" -m owac.evaluation.n04_v11_closed_loop \\
  --config "$CONFIG" \\
  --eval-list "$EVAL_LIST" \\
  --per-domain "$PER_DOMAIN" \\
  --output "$RUN" \\
  --checkpoint "$CKPT" \\
  --platform "$PLATFORM" \\
  --seed "$SEED" \\
  --device cuda:0 \\
  --sim-device $GPU \\
  --image-representation "$IMAGE_REPRESENTATION" \\
  --resize-size "$RESIZE_SIZE" \\
  --proprio-mode "$PROPRIO_MODE" \\
  --robotwin-image-key "$ROBOTWIN_IMAGE_KEY" \\
  --policy-command '$POLICY_COMMAND' \\
  ${FORCE_FLAG[@]:-} ${EXTRA[@]:-}
echo "[\$(date -u +%FT%TZ)] 闭环进程结束 rc=\$?"
SH
chmod +x "$LAUNCHER"

tmux new-session -d -s "$SESSION" "bash $LAUNCHER 2>&1 | tee -a $LOG"
echo "已启动： session=$SESSION  gpu=$GPU"
echo "  运行目录 : $RUN"
echo "  日志     : $LOG"
echo "  查看进度 : tail -f $LOG"
echo "             python3 -m json.tool $RUN/progress.json"
echo "             wc -l $RUN/eval_results.jsonl"
echo "  进入会话 : tmux attach -t $SESSION"
echo "  停止     : tmux kill-session -t $SESSION"

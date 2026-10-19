#!/usr/bin/env bash
# N04 v1.1 S2 正式训练启动器（远端 tmux，独立会话与运行目录）。
#
# 用法（在远端执行）：
#   N04V11_SOURCE=<frozen snapshot dir> bash n04_v11_launch.sh libero D 101 0
#   N04V11_SOURCE=<frozen snapshot dir> bash n04_v11_launch.sh robotwin D 101 1
#
# 参数： <platform: libero|robotwin> [arm: D] [seed: 101] [gpu index]
# 环境： N04V11_SOURCE 冻结源码快照；N04V11_PYTHON / N04V11_STARVLA / N04V11_DATA /
#        N04V11_ROOT / N04V11_STEPS 可覆盖默认值。
set -euo pipefail

PLATFORM=${1:?platform: libero|robotwin}
ARM=${2:-D}
SEED=${3:-101}
GPU=${4:-0}

ROOT=${N04V11_ROOT:-/workspace/owac/research/n04-v1}
if [[ -f "$ROOT/TRAINING_STOPPED_BY_USER.json" ]]; then
  echo "用户已终止训练；当前仅修复与验证代码。停止标记：$ROOT/TRAINING_STOPPED_BY_USER.json" >&2
  exit 2
fi
SOURCE=${N04V11_SOURCE:?请设置 N04V11_SOURCE=<冻结源码快照目录>}
PY=${N04V11_PYTHON:-$ROOT/envs/n04v11/bin/python}
STARVLA=${N04V11_STARVLA:-/workspace/owac/rawvla/source/starVLA}
DATA=${N04V11_DATA:-/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data}

STAGE=v1.1/s2
RUN=$ROOT/runs/$STAGE/${PLATFORM}-${ARM}-seed${SEED}
SESSION=n04v11-${PLATFORM}-${ARM}${SEED}
CONFIG=${N04V11_CONFIG:-$SOURCE/configs/n04/v1_1/runs/${PLATFORM}_${ARM}.json}

mkdir -p "$RUN/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "会话 $SESSION 已存在；不重复启动。停止： tmux kill-session -t $SESSION" >&2
  exit 1
fi

cat > "$RUN/launch.json" <<JSON
{
  "stage": "S2",
  "platform": "$PLATFORM",
  "arm": "$ARM",
  "seed": $SEED,
  "gpu": $GPU,
  "session": "$SESSION",
  "source_snapshot": "$SOURCE",
  "config": "$CONFIG",
  "data_root": "$DATA",
  "python": "$PY",
  "starVLA": "$STARVLA",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
cp "$CONFIG" "$RUN/config.used.json"

CMD="cd $SOURCE && CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$SOURCE/src:$STARVLA $PY -m owac.training.n04_v11 \
--config $CONFIG --data-root $DATA --output $RUN --seed $SEED --device cuda:0"

cat > "$RUN/stop.sh" <<SH
#!/usr/bin/env bash
tmux kill-session -t $SESSION
SH
cat > "$RUN/resume.sh" <<SH
#!/usr/bin/env bash
# 断点续训：在正确 episode 历史上重新 burn-in，不继承另一 episode 状态。
set -euo pipefail
if [[ -f "$ROOT/TRAINING_STOPPED_BY_USER.json" ]]; then
  echo "Training stopped by user; resume is disabled." >&2
  exit 2
fi
RESUME=\${1:?请给出 checkpoint 路径，例如 $RUN/checkpoint-step0005000.pt}
tmux new-session -d -s $SESSION "$CMD --resume \$RESUME > $RUN/logs/train.resume.log 2>&1"
echo "resumed: tmux attach -t $SESSION"
SH
chmod +x "$RUN/stop.sh" "$RUN/resume.sh"

tmux new-session -d -s "$SESSION" "$CMD > $RUN/logs/train.log 2>&1; echo EXIT=\$? >> $RUN/logs/train.log"
sleep 3
echo "session=$SESSION"
echo "run_dir=$RUN"
echo "log=$RUN/logs/train.log"
echo "stop=$RUN/stop.sh"
echo "resume=$RUN/resume.sh"

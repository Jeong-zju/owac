#!/usr/bin/env bash
# N04 v1.1 S2 固定片段可视化：远端一条命令生成 LIBERO 与 RoboTwin 的 dev 片段图组，
# 写进 <run>/artifacts/<platform>-<arm>-seed<seed>/（PNG + index.json + 每帧 meta json）。
#
# 用法（在远端执行）：
#   N04V11_SOURCE=<frozen snapshot dir> bash n04_v11_visuals.sh [platforms] [clips] [frames] [device]
# 例：
#   N04V11_SOURCE=/workspace/owac/research/n04-v1/source/n04-v11-XXXX \
#     bash n04_v11_visuals.sh "libero robotwin" 3 8 cpu
#
# 环境变量： N04V11_SOURCE（必需）冻结源码快照；N04V11_ROOT / N04V11_PYTHON / N04V11_STARVLA /
#            N04V11_DATA / N04V11_STAGE / N04V11_ARM / N04V11_SEED 可覆盖默认值；
#            N04V11_THREADS（默认 2）出图进程的 CPU 线程上限，训练同机时避免抢 CPU；
#            N04V11_TAG（如 step0500）给输出目录加后缀，不同 checkpoint 的图不互相覆盖；
#            N04V11_CHECKPOINT_<PLATFORM>（大写）指定单个 checkpoint，跳过自动发现；
#            N04V11_LIST_CLIPS=1 只打印固定挑选的片段，不加载模型、不需要 checkpoint。
#
# 默认 device=cpu（不占训练 GPU）。任何平台找不到 checkpoint 时打印“尚未生成”并以非零退出，
# 不生成占位图或示例图。
set -uo pipefail

PLATFORMS=${1:-"libero robotwin"}
CLIPS=${2:-3}
FRAMES=${3:-8}
DEVICE=${4:-cpu}

ROOT=${N04V11_ROOT:-/workspace/owac/research/n04-v1}
SOURCE=${N04V11_SOURCE:?请设置 N04V11_SOURCE=<冻结源码快照目录>}
PY=${N04V11_PYTHON:-$ROOT/envs/n04v11/bin/python}
STARVLA=${N04V11_STARVLA:-/workspace/owac/rawvla/source/starVLA}
DATA=${N04V11_DATA:-/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data}
STAGE=${N04V11_STAGE:-v1.1/s2}
ARM=${N04V11_ARM:-D}
SEED=${N04V11_SEED:-101}
THREADS=${N04V11_THREADS:-2}
TAG=${N04V11_TAG:-}
LIST_CLIPS=${N04V11_LIST_CLIPS:-}

[ -d "$SOURCE/src/owac" ] || { echo "找不到 $SOURCE/src/owac；N04V11_SOURCE 不是本仓库快照" >&2; exit 2; }
[ -x "$PY" ] || { echo "找不到解释器 $PY；用 N04V11_PYTHON 指定" >&2; exit 2; }

if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""   # 可视化只读，不占训练 GPU
fi
# 训练同机时把线程池压住：只靠 torch.set_num_threads 不够（实测仍会用 ~8 核），
# BLAS/OpenMP 必须在进程启动前用环境变量限住。
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

made=0
missing=0
for PLAT in $PLATFORMS; do
  RUN=$ROOT/runs/$STAGE/${PLAT}-${ARM}-seed${SEED}
  CONFIG=$SOURCE/configs/n04/v1_1/runs/${PLAT}_${ARM}.json
  if [ ! -f "$CONFIG" ]; then
    echo "缺少配置 $CONFIG（平台 $PLAT）" >&2
    missing=$((missing + 1))
    continue
  fi

  CKPT=""
  if [ -z "$LIST_CLIPS" ]; then
    var="N04V11_CHECKPOINT_$(printf '%s' "$PLAT" | tr '[:lower:]' '[:upper:]')"
    CKPT="${!var:-}"
    if [ -z "$CKPT" ]; then
      CKPT=$(find "$RUN" -maxdepth 1 -name 'checkpoint-step*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    if [ -z "$CKPT" ] && [ -f "$RUN/checkpoint-final.pt" ]; then
      CKPT="$RUN/checkpoint-final.pt"
    fi
    if [ -z "$CKPT" ]; then
      echo "尚未生成：$RUN 下没有 checkpoint-step*.pt / checkpoint-final.pt（等训练写出后再跑本命令）" >&2
      missing=$((missing + 1))
      continue
    fi
  fi

  EXTRA=""
  [ -n "$TAG" ] && EXTRA="--tag $TAG"
  if [ -n "$LIST_CLIPS" ]; then
    EXTRA="--list-clips"
    echo "[visuals] 只列片段（platform=$PLAT clips=$CLIPS frames=$FRAMES），不出图"
  else
    mkdir -p "$RUN/artifacts"
    echo "[visuals] platform=$PLAT clips=$CLIPS frames=$FRAMES device=$DEVICE threads=$THREADS tag=${TAG:-none} ckpt=$CKPT"
  fi
  if ( cd "$SOURCE" && PYTHONPATH="$SOURCE/src:$STARVLA" "$PY" -m owac.evaluation.n04_v11_visuals \
        --config "$CONFIG" \
        --data-root "$DATA" \
        --output "$RUN/artifacts" \
        --checkpoint "$CKPT" \
        --device "$DEVICE" \
        --platform "$PLAT" \
        --arm "$ARM" \
        --seed "$SEED" \
        --clips "$CLIPS" \
        --frames "$FRAMES" \
        --threads "$THREADS" $EXTRA ); then
    made=$((made + 1))
    if [ -n "$LIST_CLIPS" ]; then
      echo "[visuals] $PLAT 片段已列出（未出图）"
    else
      echo "[visuals] $PLAT 完成：$RUN/artifacts/${PLAT}-${ARM}-seed${SEED}/index.json"
    fi
  else
    echo "可视化失败：平台 $PLAT（检查上面的报错；不写占位图）" >&2
    missing=$((missing + 1))
  fi
done

if [ "$made" -eq 0 ]; then
  echo "没有任何平台出图（missing=$missing）" >&2
  exit 1
fi
if [ "$missing" -gt 0 ]; then
  echo "部分平台未出图（missing=$missing，成功 $made）；上面已写明原因" >&2
  exit 1
fi
echo "全部完成：成功 $made 个平台"

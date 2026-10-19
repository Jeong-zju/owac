#!/usr/bin/env bash
# 同步 N04 v1.1 工作树到远端固定源码快照目录（排除数据／权重／venv／outputs／密钥）。
# 用法： n04_v11_sync.sh [本地源码目录] [远端目标目录]
set -euo pipefail

SRC=${1:-/home/jeong/zeno/owac/worktrees/n04-egomotion-separation}
SSH_PORT=${N04V11_SSH_PORT:-16724}
HOST=${N04V11_HOST:-root@154.59.156.22}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REV=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo nogit)
DEST=${2:-/workspace/owac/research/n04-v1/source/n04-v11-$STAMP-$REV}

rsync -az --delete-excluded \
  --exclude '.git/' --exclude '.venv/' --exclude 'outputs/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
  --exclude 'build/' --exclude '*.tar' --exclude '*.tar.gz' \
  --exclude '*.safetensors' --exclude '*.pt' --exclude '*.bin' \
  --exclude '*.log' --exclude '.env' \
  -e "ssh -p $SSH_PORT" "$SRC/" "$HOST:$DEST/"

echo "$DEST"

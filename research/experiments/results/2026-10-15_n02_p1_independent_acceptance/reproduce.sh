#!/usr/bin/env bash
# Run from /home/jeong/zeno/owac/worktrees/n02-measurement-completion.
# Remote source/weights are read only; output goes to a fresh local audit directory.
set -euo pipefail
cd /home/jeong/zeno/owac/worktrees/n02-measurement-completion
audit_scripts=research/experiments/results/2026-10-15_n02_p1_independent_acceptance
audit_output=$(mktemp -d outputs/n02-full/p1-independent-audit-recheck-XXXXXX)
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 16724 root@154.59.156.22 \
  'cd /workspace/owac/worktrees/n02-measurement-completion && PYTHONPATH=$PWD/src outputs/n02-full/runtime/py310-cu128/bin/python -' \
  < "$audit_scripts/audit_remote.py" > "$audit_output/remote-audit.json"
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 16724 root@154.59.156.22 \
  'cd /workspace/owac/worktrees/n02-measurement-completion && PYTHONPATH=$PWD/src outputs/n02-full/runtime/py310-cu128/bin/python -' \
  < "$audit_scripts/probe_wiring.py" > "$audit_output/wiring-probe.json"
printf 'Read-only audit receipts: %s\n' "$audit_output"

#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
exec "$task_root/configs/backends/pi05-runtime/.venv/bin/python" \
  -u -m owac.runtime.cloud_baseline "$@"

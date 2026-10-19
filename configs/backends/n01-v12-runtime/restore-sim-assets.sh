#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/.hf_home
/venv/main/bin/python - <<'PY'
from huggingface_hub import HfApi,snapshot_download
from pathlib import Path
import json
revision=HfApi().repo_info('TianxingChen/RoboTwin2.0',repo_type='dataset').sha
Path('/workspace/owac/rawvla/records/robotwin-assets-source.json').write_text(json.dumps({'repo':'TianxingChen/RoboTwin2.0','revision':revision}))
snapshot_download('TianxingChen/RoboTwin2.0',repo_type='dataset',revision=revision,allow_patterns=['background_texture.zip','embodiments.zip','objects.zip'],local_dir='/workspace/owac/rawvla/source/third_party/RoboTwin/assets',max_workers=3)
PY
cd /workspace/owac/rawvla/source/third_party/RoboTwin/assets
for group in background_texture embodiments objects; do
 unzip -q -o "$group.zip"
done

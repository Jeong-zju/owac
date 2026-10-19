#!/usr/bin/env bash
# Replays only P0 probes and evidence; no P1 training or policy evaluation.
set -euo pipefail
task_root=/home/jeong/zeno/owac/worktrees/n02-measurement-completion
cd "$task_root"
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
probe_root=outputs/n02-full/p0-independent-audit
py_gpu="$task_root/outputs/n02-full/runtime/py310-cu128/bin/python"
py_ref="$task_root/outputs/n02-full/p0/parity-venv/bin/python"
py_solver="$task_root/outputs/n02-full/runtime/py310-theseus/bin/python"
# See configs/backends/n02-full-runtime/INDEPENDENT_ACCEPTANCE.md to reconstruct
# the pinned reference environments and the actual custom Minkowski CUDA engine.
make check > "$probe_root/check.log" 2>&1
.venv/bin/python -m pytest -q > "$probe_root/full-tests.log" 2>&1
"$py_gpu" -m pytest tests/encoders/n02_full -q > "$probe_root/target-tests.log" 2>&1
"$py_ref" -m owac.frameworks.n02_full.components.upstream_refs --check all \
  --out "$probe_root/fixtures" --report "$probe_root/upstream-all.json" \
  > "$probe_root/upstream-all.log" 2>&1
"$py_solver" -m owac.frameworks.n02_full.components.upstream_theseus \
  --source-root third_party/pinned/facebookresearch_theseus@c8583de418 \
  --output "$probe_root/fixtures/theseus_upstream.json" > "$probe_root/theseus-validation.log" 2>&1
"$py_solver" -m owac.frameworks.n02_full.components.upstream_sparse \
  --engine-root third_party/pinned/shwoo93_MinkowskiEngine@bbc30ef581 \
  --output "$probe_root/fixtures/fcmae_sparse_upstream.json" > "$probe_root/sparse-validation.log" 2>&1
# Same discrete model and enumerated marginals for Gen and particles.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from owac.frameworks.n02_full.components.smc_ref import default_model, fixed_observations, exact_filter_enumeration
m = default_model(); obs = fixed_observations(m)
x = dict(transition=m.trans.tolist(), emission=m.emit.tolist(), init=m.init.tolist(),
         observations=obs.tolist(), exact=exact_filter_enumeration(m, obs).tolist(), seed=1103, num_particles=4096)
Path('outputs/n02-full/p0-independent-audit/fixtures/gen_model.json').write_text(json.dumps(x, indent=2)+'\n')
PY
export JULIA_DEPOT_PATH="$task_root/outputs/n02-full/runtime/julia-depot"
"$task_root/outputs/n02-full/runtime/julia-1.10.10/bin/julia" \
  --project=configs/backends/n02-full-runtime/gen-reference \
  src/owac/frameworks/n02_full/components/upstream_gen.jl \
  "$probe_root/fixtures/gen_model.json" "$probe_root/fixtures/gen_upstream.json" \
  third_party/pinned/probcomp_Gen.jl@2a27298b39 > "$probe_root/gen-validation.log" 2>&1
"$py_ref" -m owac.frameworks.n02_full.components.parity_suite \
  --fixtures "$probe_root/fixtures" --output "$probe_root/parity" > "$probe_root/parity.log" 2>&1
"$py_gpu" -m owac.evaluation.n02_full.p0 profile \
  --config configs/encoders/n02_full/p0.json --device cuda --output "$probe_root/profile-accepted" \
  > "$probe_root/profile-accepted.log" 2>&1
"$py_gpu" -m owac.evaluation.n02_full.p0 profile \
  --config configs/encoders/n02_full/p0-forward-profile.json --device cuda \
  --output "$probe_root/profile-forward-accepted" > "$probe_root/profile-forward-accepted.log" 2>&1
# Real-cache audit was run on 107.209.104.125:44327, entirely under the isolated
# /workspace/owac/n02-p0-independent-audit-20261014 directory. Exact remote command:
# PYTHONPATH=src /venv/main/bin/python scripts/n02_p0_independent_cache_check.py \
#   --root /workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench \
#   --hf-metadata /workspace/owac/rawvla/records/dataset-source-pinned.json \
#   --checkpoint-root /workspace/owac/rawvla/checkpoints --output outputs
# Copy its summary/manifests to probe_root before bundling. For each platform:
# PYTHONPATH=src /venv/main/bin/python -m owac.evaluation.n02_full.p0 inventory \
#   --config configs/encoders/n02_full/p0-remote-{libero,robotwin}.json \
#   --output outputs/inventory_{libero,robotwin} --max-files 2 --evidence-kind real_cache
.venv/bin/python - <<'PY'
from pathlib import Path
import gzip, json, hashlib
from owac.frameworks.n02_full.data import PartitionManifest, build_anchor_table
root = Path('outputs/n02-full/p0-independent-audit'); out = root/'frozen-data'; out.mkdir(exist_ok=True)
summary = {}
for fmt in ['libero','robotwin']:
    m = PartitionManifest.from_json((root/f'manifest_{fmt}.json').read_text()); rows = {}
    for part in ['train','dev','calibration']:
        table = build_anchor_table(m,part); raw = table.to_json().encode()
        path = out/f'{fmt}-{part}-anchors.json.gz'; path.write_bytes(gzip.compress(raw,mtime=0))
        rows[part] = dict(anchors=len(table),anchor_hash=table.sha256(),json_sha256=hashlib.sha256(raw).hexdigest(),
                         gzip_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path),window=table.window)
    summary[fmt] = dict(manifest_hash=m.content_hash(),partitions=rows)
(root/'frozen-data.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
.venv/bin/python research/experiments/results/2026-10-14_n02_p0_independent_acceptance/bundle_receipts.py

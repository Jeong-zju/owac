# C-full continuation runbook (implementation handoff)

Status: CPU implementation complete; **no GPU prepare/training has been run by
the implementation round**. Sol owns the source freeze, first real `prepare`,
first update-51 audit, acceptance, and resume. Local run root:

```bash
cd /home/jeong/zeno/owac/repo
ROOT=/home/jeong/zeno/owac/repo/outputs/raw-distillation-c-full-20261021-123313
PY=/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/runtime/bin/python
CFG=/home/jeong/zeno/owac/repo/configs/experiments/raw_stem_siglip_c_full_v1.json
MIG=$ROOT/migration-authorized.json          # Sol-signed; template stays authorized=false
AUTH=$ROOT/authority-C.json                  # Sol-signed, bound to user-authorization sha256
```

The signed migration and authority are verified by the runtime. Do not edit the
now-frozen config. The canonical GPU interpreter is the only interpreter for
model work; CPU tests use `uv run --locked`.

## 0. Monitor (read-only, local only)

Sol already runs the monitor at `http://127.0.0.1:8878/` (PID owned by Sol). To
restart it on the new module after the freeze, use a **new** terminal (do not
kill any other job):

```bash
cd /home/jeong/zeno/owac/repo
uv run --locked python -m owac.runtime.raw_distillation_monitor \
  --config configs/experiments/raw_stem_siglip_c_full_v1.json \
  --run-root outputs/raw-distillation-c-full-20261021-123313 \
  --host 127.0.0.1 --port 8878
```

It exposes only `GET/HEAD /`, `/index.html`, `/api/snapshot`; no arbitrary path,
no start endpoint, no upload. Before training it shows honest `preparing` with
`parent 50` and `new stage 0/4950 = 0.0%`, not an invented curve.

## 1. Prepare the actual C50 (zero optimizer steps)

`prepare` loads the teacher/student, strict-loads C50 with the **parent** recipe,
restores optimizer/scheduler/RNG/sampler/counters, migrates only the whitelisted
operational fields, copies the parent checkpoint into lineage, and evaluates the
fixed C50 102-window token baseline. It compares that baseline with the frozen
`attached-assessment.json` C50 values at `1e-6`; a mismatch is a hard failure.

```bash
cd /home/jeong/zeno/owac/repo
PYTHONPATH=src "$PY" -m owac.runtime.raw_temporal_continuation prepare \
  --config "$CFG" \
  --migration "$MIG" \
  --authority "$AUTH" \
  --device cuda:0 \
  --run-root "$ROOT"
```

Expected success JSON includes `status=prepared`, `optimizer_updates=0`,
`next_optimizer_step=51`, `cumulative_updates=50`, `validation_windows=102`,
`validation_sources=5`, and the C50 MSE/cosine/total. It writes
`prepared.json` with the real `prepare_wall_seconds` (the supervisor seeds its
cumulative new-stage wall from this value), `parent-baseline.json`,
`parent-history.jsonl`, `lineage/parent-C50.pt`, `lineage/parent-selector.json`,
`run-meta.json`, `run-status.json`.

Readiness receipts are carried by the signed migration, not re-signed:
`parity-receipt.json` and `resource-receipt-C.json` are checked against the full
teacher/backend/tokenizer/stats/source/cache identity, `passed=true`,
`optimizer_updates=0`; their original `research_config_sha256` remains
provenance. No new "passed" measurement is fabricated. If a receipt path or SHA
is missing, prepare refuses before model construction.

## 2. Supervisor phase A: run to cumulative 51

The supervisor needs `prepared.json`. It launches exactly one child with a new
session/process group and accounts real wall (including prepare) and all new
run-root bytes.

```bash
cd /home/jeong/zeno/owac/repo
PYTHONPATH=src "$PY" -m owac.runtime.raw_temporal_c_supervisor start \
  --run-root "$ROOT" \
  --config "$CFG" \
  --migration "$MIG" \
  --authority "$AUTH" \
  --runner-python "$PY" \
  --device cuda:0 \
  --stop-after-update 51
```

This performs **one** actual optimized update from step 50 to 51. It then saves
`checkpoints/checkpoint_final_000051.pt`, validates the full fixed validation set,
updates the token selection sidecar, and exits with `status=waiting_acceptance`.
It does not continue to 5000 without the Sol acceptance in step 3. The declared
cumulative cap remains 5000; `--stop-after-update 51` is operational only.

Audit at least:

- `updates.jsonl` has one real `cumulative_updates=51` line with
  `anchor_presentations=816`, `finite_gradients=true`, `lr_used/lr_next`, and
  non-zero actual changed parameters;
- `checkpoint_final_000051.pt` exists, its SHA is the one in
  `run-status.json.last_checkpoint`, and reloading it (via a later `resume`)
  does not rebuild the optimizer or reset RNG/sampler;
- `validation.jsonl` has the 102-window/5-source/5-reset record with actual
  `valid_tokens`, `scalar_denominator=valid_tokens·2048`, `test_used=false`;
- `selection.json` uses `validation_token_total`, not action MAE;
- the monitor shows `new stage 1/4950`, not 50/5000.

## 3. Sol acceptance (required before resume)

Only after auditing the actual checkpoint SHA and first-step records:

```bash
cd /home/jeong/zeno/owac/repo
SHA=$(uv run --locked python - <<'PY'
import json
from pathlib import Path
root=Path('outputs/raw-distillation-c-full-20261021-123313')
print(json.loads((root/'run-status.json').read_text())['last_checkpoint']['sha256'])
PY
)
PYTHONPATH=src "$PY" -m owac.runtime.raw_temporal_c_supervisor accept \
  --run-root "$ROOT" \
  --step 51 \
  --checkpoint-sha "$SHA" \
  --reviewer Sol \
  --note "actual step-51 audit accepted; C50 state preserved"
```

The acceptance is written atomically to `acceptance/phase-0051.json`; the
implementation never creates a passing acceptance. If the SHA, step, reviewer,
or schema is wrong, resume refuses.

## 4. Supervisor phase B: resume the same stage to cumulative 5000

```bash
cd /home/jeong/zeno/owac/repo
PYTHONPATH=src "$PY" -m owac.runtime.raw_temporal_c_supervisor resume \
  --run-root "$ROOT" \
  --stop-after-update 5000 \
  --acceptance "$ROOT/acceptance/phase-0051.json"
```

The supervisor resumes from the latest continuation checkpoint with the same
recipe and parent provenance, keeps cumulative counters/optimizer/scheduler/
RNG/sampler state, and adds this phase's real wall to the previous prepare + 51
phase wall. It enforces:

- cumulative updates ≤ 5000 (4950 new) and anchors ≤ 80000 (79200 new);
- real new-stage wall ≤ 230400 s with a proactive TERM margin so the hard cap is
  not exceeded by grace/kill waits; wall is recorded untruncated;
- new run-root bytes ≤ 20 GiB, counting harness logs/receipts and hardlink-deduped
  files while excluding only symlinks/shared cache;
- checkpoints every 250 updates plus milestones 500/2000/5000/final, bounded
  latest-two recovery plus pinned best/milestones, and a durable final state.

No automatic retry occurs. A nonzero child exit, signal, cap, or error leaves a
durable error/native-exit receipt; resume must be explicit.

## 5. Optional offline action diagnostic (never selection)

The training loop does not run action MAE. If desired, run a separate read-only
diagnostic against an **actual** checkpoint (this does not update weights and is
not required for selection):

```bash
cd /home/jeong/zeno/owac/repo
PYTHONPATH=src "$PY" -m owac.runtime.raw_temporal_continuation action-diagnostic \
  --config "$CFG" \
  --run-root "$ROOT" \
  --checkpoint "$ROOT/checkpoints/checkpoint_005000.pt" \
  --device cuda:0
```

It writes `action-diagnostic/action-<step>.json` with the actual checkpoint SHA,
timestamp, first1/first5/full10 MAE, `used_for_selection=false`,
`used_for_stop=false`, and `closed_loop=not_run/0 episodes`. The monitor shows it
separately from the token curves. Prior pilot action regression is preserved; no
control-recovery claim is made.

## 6. Status / artifacts

```bash
cd /home/jeong/zeno/owac/repo
uv run --locked python -m owac.runtime.raw_temporal_continuation status --run-root "$ROOT"
uv run --locked python -m owac.runtime.raw_temporal_c_supervisor status --run-root "$ROOT"
```

Key owned files: `run-status.json`, `run-meta.json`, `updates.jsonl`,
`validation.jsonl`, `selection.json`, `parent-baseline.json`,
`parent-history.jsonl`, `checkpoints/`, `lineage/`, `acceptance/`,
`supervisor-state.json`, `supervisor-heartbeat.json`.

## Boundaries and unresolved items

- No GPU prepare/training/action diagnostic/closed-loop episode was run in the
  implementation round.
- Closed loop remains `not_run` (0 episodes); control is not restored; the task
  is single-task normal-light LIBERO goal task6.
- Input is pseudoRAW3 from RGB8 inverse processing, not native Bayer.
- Teacher is the local author-architecture init0 frontend + actual LIBERO π0.5.
- Reaching 5000 updates is a diagnostic budget end, not a convergence or
  capability claim.
- The checkpoint cannot embed its own final SHA; `selection.json` is the
  authoritative actual-file SHA and the checkpoint records `null` only for the
  self-referential-best case. `selector_self_sha_embedded=false` is explicit in
  the continuation provenance.
- The tracked migration template remains `authorized=false`; only Sol's
  `$ROOT/migration-authorized.json` authorizes a real run.

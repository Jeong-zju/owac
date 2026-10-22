# C-only RAW→RAW token distillation continuation (C50 → cumulative 5000)

Date: 2026-10-21 (Asia/Shanghai). Status: **implementation and CPU acceptance
complete; no GPU prepare, training, action diagnostic, or closed-loop episode has
been launched by this implementation round.** Sol owns the first real prepare,
first-step audit, source freeze, and launch.

## Authorization and scope

User authorization (verbatim): “可以开始充分蒸馏，并给我一个简单的监控页面来实时查看
进度和效果。” It continues only the actual C arm from the completed C50 pilot:

- cumulative optimizer updates ≤ 5000, of which 4950 are new (next actual step 51);
- cumulative anchor presentations ≤ 80000, of which 79200 are new;
- ≤ 64 new GPU-wall hours (230400 s) and ≤ 20 GiB of new run-root output;
- C only; no B/D arm, no inherited-vision/backend unfreeze, no action loss, no
  weak-light, no closed loop, no physical robot, no download;
- same optimizer (AdamW), learning rate 1e-4, weight decay 0.01, clip 1.0,
  warmup 100, seed 101, effective batch 16, microbatch 1, full-prefix burn-in +
  six endpoint unroll at stride 5, stem+gate 14,454,784 trainable parameters;
- teacher is the **local author-architecture init0 RAW frontend + actual LIBERO
  π0.5**, not a published trained author frontend; the source data is three-channel
  LIBERO RGB8-inverse pseudoRAW3, **not native Bayer CFA**.

## Parent evidence actually used

| Item | Value |
| --- | --- |
| Parent C50 checkpoint | `outputs/raw-distillation-pilot-20261021-101725/pilot-C/checkpoint_final_000050.pt` |
| Parent C50 SHA256 | `631b6ea1e21bbe01e4e08b9da56432a0b9a703229a14a59463ac01a009052e1a` |
| Parent config | `.../authorized-config.json`, file SHA256 `8f04c2d4…b040` |
| Parent recipe SHA256 | `ccd0ec412f85113e28f54f4dc50dab41ae5f6d39e858ef3e8988290d404b7477` |
| Cache identity | `ba146ed39acd112d02855618cbc3436d31a6f07992fda86aadefddbd4ddc4b60` |
| Parent C50/800 counters | 50 optimizer updates, 800 anchor presentations, optimizer step 50, scheduler last_epoch 50 |

Sol independently recorded `sol-parent-state-audit.json`, `sol-assets-audit.json`
and `parent-verification.json` in the run root. Those receipts are read-only
cross-checks; the implementation does not replace the required new `prepare`.

## New recipe and narrow migration

New config: `configs/experiments/raw_stem_siglip_c_full_v1.json`.
New recipe SHA256: `5aa06daf56e21254cf85e67b4b9d9a08e5c481232bc7d07b4f97d11872f31eba`.

The only allowed differences from the parent recipe are operational and are
whitelisted explicitly in the migration manifest:

- `assets.output_root` (new owned run root);
- `authorization` (new authority binding);
- the `continuation` provenance block;
- `training.pilot_max_updates_per_arm` 50→5000;
- `training.checkpoint_updates`, `checkpoint_interval_updates=250`,
  `recovery_checkpoints=2`;
- `training.earlystop_patience` 1000→0 (action early stop disabled);
- `training.max_wall_seconds` 14400→230400, `combined_gpu_hours` 4→64;
- `training.selection_metric` and `evaluation.selection_metric`
  → `validation_token_total`.

All numerical/model/data/history fields are identity-checked: seed, effective
batch, microbatch, cumulative update/anchor caps, learning rate, optimizer,
weight decay, clip, warmup, frozen audit interval, activation-memory mode,
loss weights, stage-1/stage-2 scope, student definition, task, teacher
requirement, asset paths/hashes, evaluation flow/horizon/execute steps,
validation split, test-selection=false, groups and controls.

The tracked migration template
`configs/experiments/raw_stem_siglip_c_full_migration_v1.json` intentionally keeps
`authorized=false`. Sol signed the actual run at
`outputs/raw-distillation-c-full-20261021-123313/migration-authorized.json`
(authorized, recipe `5aa06daf…31eba`). The runtime refuses an unauthorized or
drifting manifest.

## Preparation and baseline

`prepare` performs zero optimizer steps. It:

1. verifies the parent checkpoint/config/new-config/recipe file hashes and the
   signed migration;
2. copies the actual C50 checkpoint into the new run root as lineage;
3. builds the teacher, inherited SigLIP student and frozen bridge from the same
   audited assets and cache;
4. strict-loads C50 with the **parent** config and parent run identity, restoring
   optimizer, scheduler, sampler state, RNG and counters; the next actual update
   is 51;
5. migrates only the operational recipe and binds the new recipe + parent
   provenance into every later checkpoint;
6. records the old action-MAE selector with explicit provenance and clears it
   from selection;
7. evaluates the C50 held-out projected-token baseline over the fixed 102
   windows / 5 validation trajectories with per-source state reset.

The baseline is compared item by item against the already-frozen parent
`attached-assessment.json` values at absolute tolerance **1e-6**, without
changing the tolerance:

| metric | C50 reference |
| --- | --- |
| feature MSE | 2.7655930869719563 |
| feature cosine | 0.047423013905976336 |
| feature total (MSE + 0.1·cos) | 2.770335388362554 |
| valid tokens | 52224 |
| scalar denominator | 106954752 |
| windows / sources / resets | 102 / 5 / 5 |

Any difference is a hard failure with actual/reference values; Sol must locate
the numerical cause rather than loosen the tolerance.

## Selection

Only final projected visual tokens are supervised/evaluated:
`L = FP32 masked MSE + 0.1·masked cosine` with the original amplitudes. MSE
denominator is `valid_tokens·2048`; cosine denominator is `valid_tokens`.
Selection is the minimum total over the fixed validation set, ties to the
earlier optimizer step. Test split is never selection-eligible.

Action MAE is a separate, optional, offline diagnostic with its own actual
checkpoint SHA and timestamp. It cannot early-stop, switch arms, choose a
checkpoint, or enter the selection sidecar. Prior pilot observations (C50 token
alignment improved; first-5 action error regressed; closed loop untested) are
preserved.

## Checkpoints, status, and monitoring

- Atomic `os.replace` checkpoint saves; each checkpoint binds the new recipe,
  parent checkpoint/config SHA, migration SHA, authority SHA, selection policy,
  interval/retention, and evidence that the new-stage timing excludes the parent
  50-step wall.
- Retention: latest two recovery checkpoints plus pinned milestones
  500/2000/5000 and the pinned best; lineage parent checkpoint is never deleted.
  A file cannot embed its own final SHA: for a self-referential best the
  checkpoint records `null` and `selection.json` carries the actual SHA.
- Periodic runtime JSONL every actual optimizer update includes counters, loss
  components, used vs next LR, anchors, trainable/finite-gradient summary,
  timing and GPU/host resources. Validation JSONL includes masks/denominators,
  per-source metrics, resets, actual checkpoint SHA and `test_used=false`.
- The Chinese local-only monitor at `127.0.0.1:8878` is read-only, serves only
  its configured run root, polls every 5 s, distinguishes the parent 50-step
  history from the new stage, and treats dead/reused/zombie PIDs as not running.
  Unknown values are `null`, not 0.
- The supervisor is the only launcher of training phases. It uses a new session
  process group, records PID/PGID/starttime, holds a `flock`, accounts real
  cumulative new-stage wall (seeded by `prepare_wall_seconds`) and all new
  run-root bytes, sends signals only to its own recorded group, and never
  auto-retries. Phase 51 requires an explicit Sol acceptance before resume.

## Implementation files

- `src/owac/runtime/raw_temporal_continuation.py`
- `src/owac/evaluation/raw_temporal_features.py`
- `src/owac/runtime/raw_distillation_monitor.py` + `.html`
- `src/owac/runtime/raw_temporal_c_supervisor.py`
- `configs/experiments/raw_stem_siglip_c_full_v1.json`
- `configs/experiments/raw_stem_siglip_c_full_migration_v1.json` (template)
- `tests/test_raw_temporal_continuation.py`,
  `tests/test_raw_temporal_features.py`,
  `tests/test_raw_distillation_monitor.py`,
  `tests/test_raw_temporal_c_supervisor.py`

No existing temporal trainer/runtime semantics were rewritten; the new runtime
composes the existing strict `TemporalDistillationTrainer` state machinery.

## Verification status

CPU tests, lint/format, compile checks and exact commands are recorded in the
completion review under
`outputs/raw-distillation-c-full-20261021-123313/harness/round-01/implementation-review.md`.
Not run in this round: GPU prepare, first actual update 51, resume to 5000,
action diagnostic, closed-loop episodes, real-browser visual QA. Control is not
restored and no convergence claim is made.

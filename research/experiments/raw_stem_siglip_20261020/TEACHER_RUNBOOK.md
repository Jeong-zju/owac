# 本阶段最终状态（2026-10-20）

教师已验收：Normal task6共同20新初态，Default20/20、RAW教师20/20；选初始化0。实际155前端更新、1240 anchors未作为选定权重；学生0更新。验证9成功＋1人为中断，完整N10；50回放／50策略场次已全部消耗，不重跑。主报告见 [teacher-independent-review.md](teacher-independent-review.md)，[模型注册表](/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/teacher-registry.json)。以下100更新／continuation段是历史执行命令，不是要求继续启动。

最终实际控制命令（已完成，重复同stage会被拒绝）：

```bash
cd /home/jeong/zeno/owac/repo
LIBERO_CONFIG_PATH="$PWD/outputs/rawvla-teacher-20261020/libero-config" \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
NUMBA_CACHE_DIR="$PWD/outputs/rawvla-teacher-20261020/numba-cache" \
MPLCONFIGDIR="$PWD/outputs/rawvla-teacher-20261020/mpl-cache" \
outputs/rawvla-teacher-20261020/runtime/bin/python \
  outputs/rawvla-teacher-20261020/receipts/evaluate-teacher-authorized.py \
  --split test --stage candidate0-paired-test20 \
  --artifact "$PWD/outputs/rawvla-teacher-20261020/teacher-candidate-0000.pt" \
  --identity "$PWD/outputs/rawvla-teacher-20261020/teacher-candidate-0000-identity.json"
```

只读核对：

```bash
cd /home/jeong/zeno/owac/repo
outputs/rawvla-teacher-20261020/runtime/bin/python -m owac.runtime.rawvla_teacher status \
  --manifest "$PWD/outputs/rawvla-teacher-20261020/raw/sources-audited.json"
```

冻结artifact是 `teacher-validated-normal-task6-init0.pt`，与测试使用的candidate文件字节相同；独立模型推理不需要训练RGB／RAW NPZ。它明确包含NeuralISP，不能冒称RGB-free学生。GPU模型不变；本轮控制用OSMesa CPU渲染，未把CPU ISP数值失败变成通过。

---

# Teacher reproduction runbook — task6 Normal

Status: author-frontend teacher only; 100 optimizer updates completed and audited, full continuation was running through the reviewed launcher at the time of this runbook. This document does not authorize a new research arm, student, physical robot, or policy control claim.

## Authority and scope

- Authority: `outputs/rawvla-teacher-20261020/authority.json`
- Authority SHA256: `74369e71b27867f83660066ec81d377c971be1d4dc9592a1a90783948e8316fc`
- Scope: task6 `libero_goal`, zero-based task id 6, task order 0, instruction `put the cream cheese in the bowl`, Normal lighting.
- Caps: 2000 cumulative frontend updates, 16,000 anchors, 12 GPU hours, 50 GiB new disk, 50 demonstration replays, 50 policy episodes (10 validation/40 test).

## Frozen identities

| Item | Path / value |
|---|---|
| Audited sources | `outputs/rawvla-teacher-20261020/raw/sources-audited.json` |
| Sources SHA256 | `c84966687c649ca6782c345f53a9a16f907b048184cc583d60067c1a58e8ed23` |
| Validation catalog | `outputs/rawvla-teacher-20261020/validation-catalog.json` |
| Catalog SHA256 | `8ca6e0e73d0bb7377e523e0cfc44800b459dacfde66be190d470e1b7dc6d21b0` |
| Initial states manifest | `outputs/rawvla-teacher-20261020/initial-states/manifest.json` |
| Initial states SHA256 | `7187d931e585273d5408da0103d77f9947fb9c2da99d01c662120740c4f20061` |
| Initial states array SHA256 | `49673c083a0c656aba5ef28047ee241f7da6bdb4e0b0b869288a9c2e5822b9f2` |
| Author root | `outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main` |
| StarVLA root | `outputs/raw-distillation-20261020-sol/starvla-pinned-source` |
| PI05 checkpoint | `outputs/checkpoints/raw-distillation/pi05_libero_starvla/bfloat16/model.safetensors` |
| Checkpoint SHA256 | `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8` |
| Tokenizer | `outputs/checkpoints/raw-distillation/paligemma_tokenizer.model` |
| Tokenizer SHA256 | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |
| Statistics | `outputs/checkpoints/raw-distillation/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json` |
| Statistics SHA256 | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |

The audited source manifest must remain 50 sources, 5,349 frames, 40/5/5 train/validation/test. The validation catalog is immutable and contains 102 anchors on five held-out sources.

## Actual launch used by Sol

The actual pilot and continuation used the reviewed launcher, not an ad-hoc Python one-liner:

```bash
cd /home/jeong/zeno/owac/repo
outputs/rawvla-teacher-20261020/runtime/bin/python \
  outputs/rawvla-teacher-20261020/receipts/train-authorized.py \
  --target 100

outputs/rawvla-teacher-20261020/runtime/bin/python \
  outputs/rawvla-teacher-20261020/receipts/train-authorized.py \
  --target 2000 \
  --resume outputs/rawvla-teacher-20261020/training/checkpoint_0100.pt \
  --stage cpu-continue2000 --activation-memory-mode save_on_cpu
```

The reviewed launcher checks the authority, takes a phase lease before model construction, loads checkpoint100 exactly, verifies frontend/optimizer/scheduler/sampler/CPU-CUDA RNG, reserves evaluation time, and writes phase receipts. Do not signal, restart, or alter the running continuation from another process.

## Current actual evidence

- Pilot 100 passed; 74 frontend parameter keys changed; AdamW step count 100; backend audit 813/813 stable.
- Validation first5 MAE: step0 `0.04319213129404822`; step100 `0.04350191758032704` (no improvement yet).
- Resume audit: frontend83 exact, optimizer steps exact, nonzero moments 74, scheduler/sampler/CPU-CUDA RNG exact.
- Normal GPU activation mode OOM before step101 under 19.5 GiB shared services.
- `save_on_cpu` actual later profile was ~1.48–1.55 sec/micro; projected 6.5h remaining, within the 12h phase with 3h evaluation reserve.
- `checkpoint_blocks` alone OOMed; `checkpoint_blocks` plus CPU offload had gradient error `1.54e-5` versus frozen tolerance `1.06e-7`, so it is **rejected** and production code has been removed. Archived evidence: `outputs/rawvla-teacher-20261020/rejected-memory-mode/`.
- No policy episodes, closed-loop control, or student work; control recovery is unknown.

## Reusable bound CLI

Example (`train --help` is the normative parser surface):

```bash
cd /home/jeong/zeno/owac/repo
outputs/rawvla-teacher-20261020/runtime/bin/python \
  -m owac.runtime.rawvla_teacher train \
  --authority /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/authority.json \
  --source-manifest /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/raw/sources-audited.json \
  --catalog /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/validation-catalog.json \
  --author-root /home/jeong/zeno/owac/repo/outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main \
  --starvla-root /home/jeong/zeno/owac/repo/outputs/raw-distillation-20261020-sol/starvla-pinned-source \
  --checkpoint /home/jeong/zeno/owac/repo/outputs/checkpoints/raw-distillation/pi05_libero_starvla/bfloat16/model.safetensors \
  --tokenizer /home/jeong/zeno/owac/repo/outputs/checkpoints/raw-distillation/paligemma_tokenizer.model \
  --statistics /home/jeong/zeno/owac/repo/outputs/checkpoints/raw-distillation/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json \
  --config /home/jeong/zeno/owac/repo/outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main/configs/libero/pi05.yaml \
  --run-root /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020 \
  --checkpoint-dir /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/training \
  --activation-memory-mode save_on_cpu \
  --resume /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/training/checkpoint_0100.pt \
  --max-updates 2000 \
  --run-id reviewed-continuation-001
```

The CLI:

- acquires the canonical `PhaseBudget` lease before model construction;
- reserves three GPU hours and rejects a run that cannot preserve them;
- requires `--activation-memory-mode` explicitly (`none` or `save_on_cpu`);
- binds the actual `TeacherOfflineEvaluator` and writes unique validation NPZ/JSON directories;
- rejects generic validation callback imports and budget bypass flags;
- refuses a ledger/trainer update or anchor mismatch rather than resetting/charging twice;
- writes checkpoints under the phase training directory, not the phase asset root.

The CLI is available but real execution remains Sol-reviewed. Do not start it while the current continuation owns the lease.

## Training semantics

- Trainable scope: author RawVLA frontend only, 829,172 parameters.
- Backend: frozen original PI05/SigLIP/action expert; 813 backend keys must remain stable.
- Objective: native action velocity loss (7D real / padded32) plus brightness/chroma auxiliaries; no objective change.
- Batch: microbatch1 × gradient accumulation8, effective8; 8 anchors per update, 16,000 anchor cap.
- Temporal: burst6, recurrent endpoints6, endpoint stride5; training BPTT resets recurrent state per window.
- Normalization distinction:
  - Training actions/state use the author training quantile path (clamp/max-range behavior where applicable).
  - Deployment/offline client state uses the native client `(x-q01)/(q99-q01+1e-6)*2-1` without clipping.
  - Actions are native 7D, padded to 32; state is canonical 8D padded to 32.
- No clean RGB data or teacher cache is read during policy inference; RAW-only evaluation is required.


## State-padding correction / erratum

Sol's round12 alarm was based on the PI05 class default and tokenizer pad
attribute without reading the effective LIBERO flag. The actual config sets
`discrete_state_input: false`; `PI0._init_runtime_config` overwrites the class
default. In false mode `_prepare_examples` pads state with `0.0` and the
tokenizer receives `None` state, so the original explicit zero-padded32 was
bitwise equivalent to state8 for this LIBERO policy.

See `research/experiments/raw_stem_siglip_20261020/TEACHER_PADDING_ERRATUM.md`
and `outputs/rawvla-teacher-20261020/round-13/`. The 155 physical updates are
not invalidated by state padding. The state8 interface cleanup is retained;
diagnostic `padded_state` now defaults to native false-mode `0.0` and accepts an
explicit pad for true-mode tests. Model paths pass state8, not the diagnostic
32-vector.

Current actual policy status: 8/8 completed validation episodes succeeded under
the actual false config, then the 9th was interrupted by the false-alarm
operator stop and is retained as an invalid attempt in the 10-episode
denominator; the 10th was not run. No 20-state test result exists. The 10/10
gate cannot pass now and is not relaxed. No control recovery or final selection
is claimed.

## Evaluation and checkpoints

- Validation uses only the five held-out sources / 102 frozen anchors; no test selection or performance filtering.
- Validation metrics are physical real7 first1/first5/full10 MAE; checkpoint selection is minimum physical first5 MAE, earlier step on ties.
- A checkpoint is only a candidate until its validation metrics are produced. Step100 did not improve MAE over step0.
- Offline evaluation should use strict frontend artifacts and the same backend/tokenizer/stats identities; random frontend fallback is forbidden.

## Budget and stop conditions

- Global counters are phase-wide, not per-process: 2000 updates, 16,000 anchors, 12 GPU hours, 50 GiB, 50 replays, 50 policy episodes.
- Keep three GPU hours for validation/evaluation; stop on deadline, disk reserve, nonfinite/zero gradients, backend hash change, or ledger mismatch.
- Never merge/reset ledgers, delete failed attempts to improve denominators, or stop unrelated jobs.
- Actual videos and policy results, if run later, must include failures and truthful fixed denominators.

## Runbook close-out

This runbook records the current teacher data/training/evaluation setup and actual evidence. It does not claim control recovery, final teacher checkpoint acceptance, closed-loop success, or student/physical results.

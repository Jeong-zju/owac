# RawVLA teacher reproduction prerequisite inventory (Round 11 correction)

2026-10-20. Read-only engineering/reconnaissance. **No training, optimizer step, rollout, robot action, download, author contact, code/config change, or environment change occurred.** This supersedes Round10 conclusions based on the incomplete local handoff snapshot. The old snapshot is retained only as historical evidence; the complete official source is authoritative for source/config claims.

Workspace: `/home/jeong/zeno/owac/repo`.

## Sol independent review (2026-10-20)

Sol accepts the corrected complete-source inventory, not a runnable training claim. The actual official `load_config` merger independently reproduces the listed values; receipt `outputs/raw-distillation-20261020-sol/teacher-search-v2/config-merge-audit.json`. Source/distribution hashes and checkpoint writer, recurrent loop, and expected eval artifact were independently read. The paper does NOT establish horizon10 in D.3; horizon10/flow10/replan5 here follow the actual released pi05 config and OWAC baseline protocol. D.2 comparison runs mention horizon8.

The suggestions below to reduce n or choose a smaller split are NOT adopted: the current proposal retains 50 sources split40/5/5 and Normal; insufficient sources trigger a stop and revised authorization. No silent mixed-lighting substitution. The authoritative research choices and proposed caps are in `docs/plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md`: public-source weights .15/.01/.001 and AdamW wd1e-8, maximum2000 updates. This is a proposal, not authorized training. Round10 original output is retained under its round directory; its snapshot incompleteness is resolved by the newly downloaded full source.

## 0. Source correction and scope

Complete official RawVLA main archive:

- revision `ba9c56b04da2c4931b70d8844710f5928d43775b`
- bytes `2811790`
- SHA256 `57bc54e8a695ae07c2ffee9ebb2549e9efbb48536a97df14c5c8f63c52f3e012`
- extracted root: `outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main`
- authoritative `baselines/rawvla/rawvla.py` SHA256 `404daa204da7c81b6f508bd41f68980e29be98cccdcd0f367e453ad7ae11ebb0`

Round10's “base config absent / no RawVLA launcher” conclusions were true only of the old incomplete `outputs/encoder-handoff-references/rawvla` snapshot and are withdrawn for the complete archive. The complete archive contains:

- `configs/base/rawvla.yaml`
- `configs/base/rawvla_libero.yaml`
- `configs/libero/{qwen3_oft,pi0,pi05}.yaml`
- `baselines/rawvla/README.md`
- LIBERO training/evaluation/replay scripts.

The extracted tar does not contain the `starVLA` submodule tree; the official submodule is recorded as SHA `57d508cb180d0510ac9060358009c0f6ffaac420` (`.gitmodules` + `submodules-starVLA.json`). The local `outputs/raw-distillation-20261020-sol/starvla-pinned-source` matches that revision and supplies the integration/loader/trainer source below. This is source evidence, not an execution claim.

## 1. Public teacher checkpoint status

Public inspected locations still do **not** yield an author-trained RawVLA frontend checkpoint file:

- `README.md:101-113` documents checkpoint utilities, but `tools/checkpoints/download_starvla_libero.py:28-32` downloads **StarVLA LIBERO backbone** checkpoints only; no RawVLA frontend URL.
- `experiments/simulation/libero/scripts/run_libero_models.sh:162-176` defines the expected per-backbone frontend path pattern:
  `.../rawvla/checkpoints/libero/<backbone>/<step_k>/steps_<step>_raw_frontend_pytorch_model.pt`
  with default `RAWVLA_CHECKPOINT_STEP=2000` from the same runner. This proves the expected artifact name/location, not public availability.
- `baselines/rawvla/optimal_backbone_configs.yaml` records selected 2k PI0.5 action weight `.15` and five-domain paired RGB MAE `.173979`; `baselines/rawvla/README.md:155-168` says these are evaluation selections, not rollout results.
- `baselines/rawvla/README.md:3-7` says existing 1k/2k checkpoints were re-rendered, not retrained, but does not bundle them.

Conclusion scoped to inspected public release/indexes: **learned RawVLA frontend checkpoint not found**. This is not proof that no private, unindexed, or externally hosted file exists. The accepted teacher checkpoint is the **output** of reproduction, not an input prerequisite; Round10's circular gate requiring an already-trained teacher checkpoint before teacher training is removed.

## 2. Exact recipe composition

Config merger: each child uses relative `extends`; base is merged first, child second, dotlist CLI highest (`README.md:201-204`). For the selected Native pi05 LIBERO scenario the actual composition is:

| Field | Effective source | Value |
|---|---|---|
| framework | `configs/libero/pi05.yaml:12-20` | `PI05`, bfloat16 |
| frozen modules | `pi05.yaml:27-30` | `vlm,action_head` |
| pretrained backend | `pi05.yaml:29` | `${RAWVLA_ROOT}/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| action weight | `pi05.yaml:30` | `0.15` |
| dataset normalization | `pi05.yaml:7-10` | `pi05_quantile`, LIBERO norm stats |
| observation stride | `pi05.yaml:7` overrides `rawvla_libero.yaml:8` | `5` |
| action horizon | `pi05.yaml:8` overrides `rawvla_libero.yaml:9` | `10` |
| raw frontend trainable | `rawvla.yaml:15-20` | rawvla, true |
| burst frames | `rawvla.yaml:2-6,21-22` | `6` |
| recurrent frames | same | `6` |
| state / luma state | `rawvla.yaml:25-26` | `128` / `64` |
| FFT patch/stride | `rawvla.yaml:27-28` | `32` / `16` |
| denoise eta | `rawvla.yaml:39` | `0.5` |
| exposure objective | `rawvla.yaml:34-37` | two-sided L1, target `.48`, weight `.01`, STE clamp |
| chroma prior | `rawvla.yaml:44-52` | output envelope, weight `.001`, margin `.10` |
| effective batch | `rawvla.yaml:9,60` | per-device batch `1`, accumulation `8` = 8 sequences |
| updates | `rawvla.yaml:68` | `2000` |
| warmup | `rawvla.yaml:69` | `100` |
| optimizer LR | `rawvla.yaml:63-65` | `1e-4` base/frontend |
| AdamW | `rawvla.yaml:70-73` | betas `.9,.95`, eps `1e-8`, weight decay `1e-8` |
| scheduler | `rawvla.yaml:67,75-76` | cosine with min LR `1e-6` |
| clip | `rawvla.yaml:61,77` | `1.0` |
| checkpoint format | `rawvla.yaml:79-80` | `pt`, `save_trainable_only: true` |

Final selected scenario: **task6, Normal, Native pi05, 10-step action horizon, 10 flow steps, 5-step replan**. `pi05.yaml:20` sets `num_inference_steps: 10`; `pi05.yaml:8` action horizon 10; `pi05.yaml:7` observation stride 5. Burst 6/rnn 6 come from `rawvla.yaml`.

This is a runnable **config composition**, subject to data/tokenizer/backend availability. It is not evidence that a checkpoint was trained or that control succeeds.

## 3. Official training CLI

The official LIBERO training example is in `README.md:188-199`:

```bash
export RAWVLA_ROOT="$PWD"
export LIBERO_CONFIG_PATH="$RAWVLA_ROOT/.local/libero"
conda activate starvla-libero
cd starVLA
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/libero/qwen3_oft.yaml \
  --datasets.vla_data.cache_root "$RAWVLA_ROOT/benchmark_data/RawVLA-Bench/data/libero/raw"
```

The README table (`README.md:170-176`) explicitly lists `configs/libero/pi05.yaml` as the PI0.5 RawVLA recipe. The corresponding PI0.5 invocation, obtained by substituting that official config and the same cache override, is:

```bash
# derived by substituting the official pi05 recipe; not executed here
export RAWVLA_ROOT="$PWD"
export LIBERO_CONFIG_PATH="$RAWVLA_ROOT/.local/libero"
conda activate starvla-libero
cd starVLA
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/libero/pi05.yaml \
  --datasets.vla_data.cache_root "$RAWVLA_ROOT/benchmark_data/RawVLA-Bench/data/libero/raw"
```

No separate committed pi05 shell script was found; this is a minimal substitution of the official generic command and official table entry, not a separately tested launch. It is **not executed**.

The base config expects `cache_root` `${RAWVLA_ROOT}/benchmark_data/libero/rawvla_light_train_raw_npz` (`rawvla_libero.yaml:7`); the README example overrides it. The teacher run must choose and record the actual cache root; do not silently mix the public RAW tree with the expected converted NPZ layout.

## 4. Checkpoint schema and temporal semantics

Checkpoint writer (`starvla-pinned-source/starVLA/training/train_starvla.py:313-335`):

- `save_trainable_only: true` is set by `rawvla.yaml:80`.
- The writer saves `unwrapped.raw_frontend.state_dict()` only and appends `_raw_frontend` (`train_starvla.py:318-325`).
- With `save_format: pt`, the expected file is:
  `steps_<completed_steps>_raw_frontend_pytorch_model.pt`.
- Evaluation expects exactly `steps_2000_raw_frontend_pytorch_model.pt` in the per-backbone layout (`run_libero_models.sh:162-176`).
- If `save_trainable_only` were false, the trainer would save the full model state dict; the selected config does not do that.
- No optimizer state is saved by this path; README states StarVLA does not save optimizer state. Resume is `is_resume`/latest checkpoint plus model reload, not full optimizer/RNG resume (`train_starvla.py:266-293`).
- The checkpoint is an exported frontend state, not a full RawVLA+pi05 teacher container. It must be paired with the frozen pi05 backend/config/tokenizer/stats and the frontend construction YAML.

Runtime state:

- `RAWVLA.forward(raw_burst, state=None, theta_prev=None)` consumes `[B,K,3,H,W]` and returns `state` and `theta` (`baselines/rawvla/rawvla.py:1099-1128`).
- The StarVLA adapter loops all recurrent observations with
  `out = model(raw[:, t], state=state, theta_prev=theta); state, theta = out.state, out.theta`
  without detaching (`starVLA/model/modules/raw_frontend/registry.py:332-337`).
- Thus the six recurrent steps are carried through one differentiable graph by default. `baselines/rawvla/README.md:76-86` documents detach only for explicit truncated BPTT/inference, not as the training default.
- State/theta are not checkpoint tensors; `initial_state` is zeros and `initial_theta` is neutral/`wb_anchor` (`rawvla.py:812-823`). Per-source reset is required.
- Burst axis is consecutive camera frames; recurrent axis is policy observations separated by `observation_stride`. Loader clipping repeats the first frame at sequence start (`starVLA/dataloader/rawvla_npz_datasets.py:224-230`).

The public source therefore supports the paper's no-detach BPTT within the six-observation window. This does not establish a longer-horizon training claim or any control result.

## 5. Source vs paper target

Primary paper (Sol independently fetched): <https://arxiv.org/html/2609.37530v1>, Appendix D1/D3.

| Item | Paper target | Complete public source | Status |
|---|---|---|---|
| Updates | 2000 | `rawvla.yaml:68` 2000 | match |
| Effective batch | 8 (1×8) | `rawvla.yaml:9,60` | match |
| Optimizer | AdamW 1e-4, beta(.9,.95), eps1e-8 | `rawvla.yaml:63-73` | LR/AdamW match |
| Weight decay | 1e-10 | `rawvla.yaml:73` 1e-8 | **difference** |
| Warmup | 100 | `rawvla.yaml:69` 100 | match |
| Scheduler | cosine min 1e-6 | `rawvla.yaml:67,75-76` | match |
| Clip | 1 | `rawvla.yaml:61,77` | match |
| PI0.5 action weight | .10 | `pi05.yaml:30` .15; `optimal_backbone_configs.yaml` pi05 .15 | **difference** |
| Brightness weight | .001 per paper target | `rawvla.yaml:35` .01; README line 117 `.01 * brightness` | **difference** |
| Chroma weight | .001 | `rawvla.yaml:49` .001; README line 117 | match |
| Burst / recurrent | 6 / 6 | `rawvla.yaml:21-22` | match |
| Observation stride | 5 | `pi05.yaml:7`; base `rawvla_libero.yaml:8` is 8 before override | match after merge |
| Action horizon | D.3 does not fix10; D.2 comparisons use8 | `pi05.yaml:8`; base 8 before override | match after merge |
| Flow steps | OWAC protocol, not fixed by D.3 | `pi05.yaml:20` | match |
| BPTT | no state/theta detach | registry loop lines 332-337 no detach; README documents optional detach | match in source integration |

The public source is close to, but not identical to, the paper target. PI0.5 action weight `.15` is an explicit public selection, and the exposure/brightness weight `.01` conflicts with the supplied paper `.001`. A future run must declare which value it uses; it cannot be called exact paper reproduction without resolving these.

## 6. Task6 Normal data reality

Public dataset (`README.md:125-143`): 2,403 successful trajectories, 1,771 LIBERO and 632 RoboTwin 2.0. Paired RAW/RGB train trees are published on Hugging Face. The public light benchmark samples one lighting/domain configuration per source/cache entry; it does not mean every source has a Normal variant.

Local metadata:

- Old `n01-full-v1` manifest: 338 Normal files/338 groups across four LIBERO suites.
- 84 rows have prefix `libero_goal_no_noops_1.0.0_lerobot`; this is **not** a task6 count because the manifest has no task-ID mapping.
- No task6-specific source demonstration IDs, no local task6 Normal NPZ payload, and no 40/5/5 source split are present.
- Task6 initial states: 50 candidate `pruned_init` states, not demonstrations.
- Therefore selected task6 **confirmed available Normal teacher source count is 0** in local metadata/payload. The actual number in the public dataset is unknown until an authorized inventory/download; it is likely below 50 for task6 Normal because lighting is sampled per source.
- Do not silently use mixed lighting or all five domains as if they were 50 Normal task6 trajectories. If 50 Normal task6 source trajectories are required, use authorized replay/selection to Normal or report the smaller actual n; do not fabricate the count.

## 7. Minimal bounded task6 Normal teacher-development plan

This is a proposal for separate authorization. It is not a full paper reproduction and not an execution claim.

1. **Backend**: use the recovered exact pi05 backend SHA `68710f4e...7ba6e8`, tokenizer/stats, and the `pi05.yaml` composition. Freeze `vlm,action_head`.
2. **Source**: use the complete archive revision above and the official StarVLA submodule SHA `57d508...`; record root and source hashes. Do not use the incomplete old handoff snapshot as config authority.
3. **Config**: base `rawvla.yaml` + `rawvla_libero.yaml` + `pi05.yaml`. Resolve paper-vs-source differences explicitly: choose `.15` public action weight or `.10` paper target; choose `.01` public brightness or `.001` paper target; choose `1e-8` public wd or `1e-10` paper target. Do not mix silently.
4. **Data**: inventory task6 Normal source IDs from the authorized public dataset; use all actually available task6 Normal sources with a source-disjoint split. If n<50, report n and use a bounded development run; if the study requires 50 Normal, authorize replay to Normal rather than mixing light domains.
5. **Temporal**: burst 6, recurrent 6, stride 5, causal reset per source, no future leakage. Keep state/theta attached within the 6-step window; detach only at an explicit window boundary if a truncated variant is selected.
6. **CLI**: use the official `accelerate ... train_starvla.py --config_yaml ../configs/libero/pi05.yaml` form (§3) with a recorded cache root. No separate pi05 script exists; do not claim one.
7. **Checkpoint**: expect `steps_2000_raw_frontend_pytorch_model.pt` from `save_trainable_only`; record its SHA, config, data/source IDs and frontend construction config. Do not expect optimizer state.
8. **Evaluation**: teacher frontend image/action checks on held-out task6 Normal sources; no control claim from offline image MAE. Teacher accepted checkpoint is an output of this run.
9. **Resources**: no measurement in this inventory; freeze GPU/disk caps before any separately authorized run.

Stop conditions:

- task6 Normal source inventory cannot establish source IDs/count;
- data payload absent or only mixed-lighting samples;
- backend/tokenizer/stats identity mismatch;
- base/child config composition unresolved;
- paper-vs-public loss/weight choice not declared;
- no source-disjoint evaluation;
- no bounded resource caps or no explicit new authorization.

## 8. Runnable vs incomplete now

Runnable/verified as source evidence:

- complete source/config inspection and hashes;
- exact config composition and official CLI form;
- source BPTT/auxiliary implementation inspection;
- recovered pi05 backend identity/load audit (not teacher).

Still incomplete:

- no author-trained RawVLA frontend checkpoint found;
- no confirmed local task6 Normal paired demo payload/count/source IDs;
- no actual teacher training, checkpoint output, stage-0 expert acceptance, resource measurement, or control result;
- no proof that the public frontend checkpoint exists outside inspected locations.

## 9. Evidence and unresolved facts

Primary evidence:

- `outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main/README.md`
- `.../configs/base/rawvla.yaml`
- `.../configs/base/rawvla_libero.yaml`
- `.../configs/libero/pi05.yaml`
- `.../baselines/rawvla/README.md`
- `.../baselines/rawvla/rawvla.py`
- `.../baselines/rawvla/optimal_backbone_configs.yaml`
- `.../experiments/simulation/libero/scripts/run_libero_models.sh`
- `outputs/raw-distillation-20261020-sol/starvla-pinned-source/starVLA/model/modules/raw_frontend/registry.py`
- `outputs/raw-distillation-20261020-sol/starvla-pinned-source/starVLA/training/train_starvla.py`
- `outputs/raw-distillation-20261020-sol/starvla-pinned-source/starVLA/dataloader/rawvla_npz_datasets.py`
- `research/experiments/raw_stem_siglip_20261020/asset-inventory.json`
- `research/experiments/raw_stem_siglip_20261020/backend-recovery.json`
- archive hash: `57bc54e8a695ae07c2ffee9ebb2549e9efbb48536a97df14c5c8f63c52f3e012`

Unresolved:

1. Public or private location of trained RawVLA frontend weights; inspected release does not expose a download.
2. Exact task6 Normal source IDs/count in the public HF dataset; local metadata is not task-labeled.
3. Whether the public dataset contains ≥50 task6 Normal trajectories or requires authorized Normal replay.
4. Whether `.15` PI0.5 action weight and `.01` brightness weight are the intended final reproduction, given paper `.10`/`.001`.
5. Resource requirements/runtime; not measured.
6. Any teacher/control capability; not established.

No training, update, rollout, download, or control claim was made.

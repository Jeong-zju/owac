# Shared RAW → new teacher targets → full visual student：远程运行手册

Sol实测更新：新204教师targets、双卡零更新预检、真实首步及完整重载通过；连续阶段已从step1运行到后续更新。实际启动命令与PID见outputs/raw-distillation-full-vision-20261022/student-continuous-command.json及student-continuous-launch.json，实际配置为shared-stage/authorized-config.json。下文“本轮未启动”保留为Harness工程交付时状态，不是当前运行状态。独立结果见full-visual-shared-20261022-independent-review.md。

状态：**共享源适配器、教师缓存准备入口、共享配置、CPU 契约测试已完成；本轮未启动 GPU。**  
旧 50 源 RAW/teacher cache 已停止使用并保持历史。新阶段固定共享 task6 **Normal 12 源**，只对 **train+validation 10 源**生成 teacher targets；test 2 源只进入 writable manifest，不生成 target、不做选型，最终评估需另行授权。

## 1. 固定协议

- 共享只读根：`/root/autodl-pub/datasets/rawvla`
- 共享 release revision：`8586c594fff9ce38580e29745f074e82e96e74f7`
- task6 Normal 12 源，固定分组拆分 **train 8 / validation 2 / test 2**，seed 101。
- 分组键：`(original hdf5_path, demo_key)`；同一原 hdf5+demo 的所有 lighting variants 必须同 partition。**不加入任何其他 lighting 到本训练集。**
- 实际 Normal split（来自 inventory）：
  - train: `demo_1, demo_4, demo_10, demo_23, demo_27, demo_32, demo_35, demo_47`
  - validation: `demo_26, demo_44`
  - test: `demo_21, demo_28`（不生成 target）
- 计数：train anchors **118**；validation windows **46** / 2 sources；train+validation cache rows **204**。
- 教师链：**actual shared RAW → validated init0 author-architecture NeuralISP/frontend → same pi05 SigLIP/projector**。教师与 action backend 冻结。
- teacher cache 复用既有 `prepare_temporal_cache` 的真实 native target fields（actions/states/teacher tokens/state/theta 等），不伪造字段、不调用未计算字段。`teacher_rgb` 是 RAW 经教师内部 ISP 的输出，不是共享 paired RGB。
- RAW-only adapter **不读取、不登记、不校验 paired RGB**。`rgb_path=raw_path` 只是 `TeacherSource` 兼容字段；以 metadata 明确标注 `paired_rgb_used=false`、`paired_rgb_path_registered=false`、`paired_rgb_identity_verified=false`。
- 共享 NPZ 的真实 SHA 写入 manifest/raw-file map/cache identity/receipt。历史 `source_hdf5_sha256` 字段只能作为 aggregate compatibility digest；`actual_source_hdf5_sha256=null`，不得声称已验证 HDF5 SHA。
- 新阶段 optimizer step 仍从 **0** 开始；本轮不启动。旧学生 1872 checkpoint 不加载。
- 预算保持剩余 aggregate GPU wall **140,841.2631939833 s** / 输出 **15,396,584,412 B**；teacher preparation 的实测 GPU wall 由 receipt 注入 fullvisual budget，不扩预算。
- 模板配置：`configs/experiments/raw_stem_siglip_full_visual_shared_v1.json`；**active shared-stage 配置**使用 `shared-stage/authorized-config.json`。模板 `authorization.optimizer_updates=false`，本轮禁止训练；只有 Sol 审阅并在 active config 显式开启后才可训练。

## 2. 文件

- 共享源 adapter：`src/owac/raw/shared_rawvla_distillation.py`
- 全视觉训练运行时（已扩为可配置 split/count/receipt）：`src/owac/runtime/raw_full_visual_distillation.py`
- 模板配置：`configs/experiments/raw_stem_siglip_full_visual_shared_v1.json`；active 运行配置：`shared-stage/authorized-config.json`
- 测试：`tests/test_shared_rawvla_distillation.py`、`tests/test_raw_full_visual_distillation.py`、`tests/test_raw_full_visual_monitor.py`
- inventory：`outputs/raw-distillation-full-vision-20261022/shared-task6-inventory.json`
- 当前新建输出 root（远程）：`/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage`
- 旧的 `full-visual-20261022-runbook.md` 与旧 50 配置/缓存为历史，不再执行。

## 3. 共享数据安全

- `/root/autodl-pub/datasets/rawvla` 全程**只读**；不写 manifest、cache、target 或任何副本到该根目录。
- 只读共享路径的 absolute manifest 写到 stage root，不复制 raw NPZ。
- 不把旧 50 源、旧 cache、旧学生 checkpoint 读入新路径。
- 不执行全量公开集下载。
- test 源只做 inventory/metadata/文件身份读取；不生成 target、不做 validation/selection。

## 4. 远程环境

```bash
ssh -p 27786 root@101.71.76.104
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export PYTHONPATH=/root/owac/repo/src
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
STAGE=/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage
CFG="$STAGE/authorized-config.json"
mkdir -p "$STAGE"
test -L /home/jeong/zeno/owac/repo
```

Sol 已通过真实 teacher CPU load（full pi05 checkpoint + init0 frontend artifact strict load，teacher/frontend frozen/eval；约 56.69 CPU 秒）；这只证明依赖/权重可加载，不证明 shared RAW forward/target。

## 5. 构建 writable source manifest（CPU，只读共享）

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export PYTHONPATH=/root/owac/repo/src
STAGE=/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage
INV=/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-task6-inventory.json
python -m owac.raw.shared_rawvla_distillation build-manifest \
  --inventory "$INV" \
  --shared-root /root/autodl-pub/datasets/rawvla \
  --manifest "$STAGE/source-manifest.json" \
  --split-seed 101 \
  --release-revision 8586c594fff9ce38580e29745f074e82e96e74f7
```

扫描器语义：只读取 NPZ ZIP/NPY header 与 scalar `metadata_json`，不解压/物化两路像素数组，也不做模型/teacher forward。若 stage 已存在同 schema manifest，**不要加 `--force` 覆盖**；保留该证据。先前 Sol 构建曾为 geometry/counts 解码像素数组，不是 model/test target；不声称从未触碰。

必须检查 `source-manifest.report.json`：

- `source_count=12`、`split_counts={"train":8,"validation":2,"test":2}`
- `train_anchors=118`、`validation_windows=46`、`cache_rows=204`
- `test_targets_generated=false`
- 12 个 `actual_source_npz_sha256` 为真实 64-hex；`actual_source_hdf5_sha256=null`
- `paired_rgb_used=false`、`paired_rgb_path_registered=false`、`paired_rgb_identity_verified=false`
- `paired_rgb_dataset_note` 明确共享 release 可能有 paired RGB，但本 adapter 不读取/登记。

## 6. 两卡 disjoint 生成 teacher targets（GPU，train+validation only）

manifest 已写好后：

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export PYTHONPATH=/root/owac/repo/src
export CUDA_VISIBLE_DEVICES=0,1
STAGE=/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage
CFG="$STAGE/authorized-config.json"
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.raw.shared_rawvla_distillation prepare \
  --config "$CFG" \
  --manifest "$STAGE/source-manifest.json" \
  --stage-root "$STAGE" \
  --cache-root "$STAGE/teacher-cache" \
  --device cuda \
  2>&1 | tee "$STAGE/teacher-prepare.log"
```

- 两 rank 使用同一 manifest/cache identity；按 source 列表 `rank::world_size` 分配 disjoint trajectories；每个 source 的 teacher frontend state/theta 独立 reset。
- cache manifest 只由 rank0 初始化；两 rank 用同一 identity 写不相交 source rows。
- 只展开 train+validation 10 源；test 2 源不生成任何 target。
- 完成后必须存在：
  - `teacher-cache/manifest.json`（cache identity）
  - `teacher-preparation-receipt.json`
  - 204 个 token rows（实际计数）
- 检查 receipt：
  - `split_counts={"train":8,"validation":2,"test":2}`
  - `cache_rows=204`、`actual_rows=204`
  - `test_targets_generated=false`、`test_observations_consumed=false`
  - `source_raw_sha256` 12 个真实 SHA；`actual_source_hdf5_sha256=null`
  - `paired_rgb_used=false`、`paired_rgb_identity_verified=false`
  - `gpu_wall_sum_seconds` 是两 rank 实测和，必须计入后续预算
  - `output_bytes` 只计 stage 内新输出
- CPU verify：

```bash
python -m owac.raw.shared_rawvla_distillation verify \
  --receipt "$STAGE/teacher-preparation-receipt.json" \
  --cache-root "$STAGE/teacher-cache"
```

## 7. Zero-update training preflight（Sol 审阅后）

当前 shared config 的 `authorization.optimizer_updates=false`；preflight 可运行，train 会被拒绝，直到 Sol 审阅并显式修改。

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export PYTHONPATH=/root/owac/repo/src
export CUDA_VISIBLE_DEVICES=0,1
STAGE=/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage
CFG="$STAGE/authorized-config.json"
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.runtime.raw_full_visual_distillation preflight \
  --config "$CFG" --run-root "$STAGE" \
  2>&1 | tee "$STAGE/preflight.log"
```

必须满足：
- 学生 **428,579,952** 全视觉 trainable；437 个实际 pi05 inherited tensor 数值审计；无 teacher/backend 进入 optimizer。
- shared manifest 12 源、8/2/2；cache 204 rows；validation 2 sources/46 windows；test target 0。
- receipt 的 cache identity 与 cache manifest 一致；`actual_source_npz_sha256` 来自真实 NPZ；HDF5 SHA 为 null/不可用，不伪装。
- 两卡 profile/representative gradients first/mid/last + gate/stem/position/norm/projector 非零；`find_unused=false` 不遗漏。
- preflight receipt 的 config 字段是 `config_digest`（不是 `config_sha256`）；`cache_identity_key` 与 teacher receipt/cache 一致。
- `preflight_phase_gpu_wall_seconds` 只表示本次预检 phase；`teacher_preparation_gpu_wall_seconds` 是已载入的 teacher prep charge；`cumulative_gpu_wall_seconds` 与 `gpu_wall_sum_seconds` 都必须是两者之和，不能 reset teacher prep。
- `finalization_reserve_seconds` 由预检实测；no optimizer step。

## 8. 授权后的 1-step 与长跑

只有 Sol 审阅后，把 shared config 中 `authorization.optimizer_updates` 显式改为 `true`，并记录决定；本工程不自行修改授权。

1-step（计数 1/3116）：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.runtime.raw_full_visual_distillation train \
  --config "$CFG" --run-root "$STAGE" \
  --preflight-receipt "$STAGE/preflight.json" \
  --stop-after-new-update 1 \
  2>&1 | tee "$STAGE/step1.log"
```

长跑从 latest 继续，不重置计数/预算：

```bash
RESUME=$(python - <<'PY'
import json
stage='/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage'
print(stage+'/'+json.load(open(stage+'/latest.json'))['path'])
PY
)
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.runtime.raw_full_visual_distillation train \
  --config "$CFG" --run-root "$STAGE" \
  --resume "$RESUME" --max-new-updates 3116 \
  2>&1 | tee "$STAGE/long-run.log"
```

- 长跑不传 `--preflight-receipt`；checkpoint 内有 receipt/预算。
- validation 每 250 新更新一次，2 sources/46 windows；test 永不参与。
- 保存 step0/250 intervals/final；latest full + best model + step0 model-only；按唯一 inode 与预留空间计量，不扩输出。
- `selection.json.checkpoint_path` 必须指向实际保留的 `best/model.pt`；历史完整 checkpoint 目录只作为 `source_checkpoint_path` lineage，被 prune 后不能是 selected 消费者路径。best 仅模型、无 optimizer；latest full 保留 optimizer 供 resume。
- teacher preparation 的 GPU wall 已通过 receipt 计入 aggregate budget；训练 startup/preflight/validation/checkpoint wall 继续累加。
- SIGTERM 优雅停止；不自动重启。

## 9. 监控（只读 8879，旧 8878 不动）

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export PYTHONPATH=/root/owac/repo/src
STAGE=/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage
nohup python -m owac.runtime.raw_full_visual_monitor \
  --run-root "$STAGE" --host 127.0.0.1 --port 8879 \
  > "$STAGE/monitor.log" 2>&1 &
```

本地 SSH 隧道：

```bash
ssh -p 27786 -N -L 8879:127.0.0.1:8879 root@101.71.76.104
# http://127.0.0.1:8879/
```

页面只读、固定路由 `/`、`/api/status`、`/healthz`，不列目录、不暴露 checkpoint 文件/路径；显示动态 12 源/8-2-2、204 cache rows、2/46 validation、双卡显存、aggregate GPU 预算、checkpoint selection、真实 rank PID/start ticks/last update age；存活时 native exit unknown。旧 8878 页面保持停止历史。

## 10. 本轮停止条件

- 不启动 GPU：build-manifest 仅 CPU 读取；prepare 由 Sol 审核后运行；本工程当前不运行任何 prepare/preflight/train。
- 任何 shared path 写操作、paired RGB 读取、test target 生成、旧 50 cache 读取、非 Normal 源进入训练集 → 立即停止。
- 任何 manifest/cache identity、NPZ SHA、source count、split count、cache row count 不一致 → 停止，不重试、不自动重启。
- 未完成实际 RAW teacher forward、cache row 检查、GPU preflight 前不得声称 target 已生成或训练可行。

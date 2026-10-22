# 学生整视觉头双卡蒸馏：远程运行手册（2026-10-22）

状态：**工程代码、CPU 契约测试与 Sol CPU 资产构建已完成；尚未在 GPU 上启动任何优化更新。**  
本手册供 Sol 在源码冻结、独立复核和远程资产校验后执行。实际验收仍是 GPU 预检、1 次真实更新、严格重载和长跑；CPU 测试或构建通过不等于训练/收敛验收。

## 1. 冻结范围

- 学生：整个视觉头可训练，共 **428,579,952** 参数：
  - `gate` 1,152
  - RAW stem 14,453,632（seed 101 新初始化）
  - position 294,912（继承实际 pi05）
  - 27 个 SigLIP blocks 411,466,608（继承实际 pi05）
  - post-layernorm 2,304（继承实际 pi05）
  - 原 projector 2,361,344（继承实际 pi05）
- 继承 tensor 共 **437** 个；stem 替换原有 patch embedding，gate 零初始化。不得加载旧学生 `checkpoint_final_001872.pt`。
- 教师和 pi05 language/action backend **冻结且不构建到训练图**；`teacher_loaded=False`、`backend_loaded=False`。只读教师 token cache。
- 目标：FP32 masked MSE + `0.1 × cosine`；无动作损失、无 B/D、无闭环、无下载。
- 数据：现有 50 源、固定 train/validation/test = **40/5/5**，969 条 train+validation cache 行只读。共享发布 RAW（`/root/autodl-pub/datasets/rawvla`）不映射到旧教师 cache、不复制、不训练。
- 新阶段计数从 **0** 开始，最多 **3,116** 全局更新 / **49,856** anchors；旧 1872 学生仅作历史 provenance。
- 剩余远程 aggregate GPU wall：**140,841.2631939833 s**；旧本地 debit **89,558.73680601672 s**；原始 cap **230,400 s**。
- 新增输出剩余：**15,396,584,412 B**；旧输出 **6,078,252,068 B**；原始 20 GiB。
- 旧本地 8878 监控和停止阶段保持不动；本阶段使用独立 8879 只读监控。

## 2. 新增/冻结文件

新增（仅本阶段）：

- `src/owac/runtime/raw_full_visual_distillation.py`
- `src/owac/runtime/raw_full_visual_monitor.py`
- `tests/test_raw_full_visual_distillation.py`
- `tests/test_raw_full_visual_monitor.py`
- `configs/experiments/raw_stem_siglip_full_visual_v1.json`
- `research/experiments/raw_stem_siglip_20261020/full-visual-20261022-runbook.md`
- 本任务报告：`outputs/raw-distillation-full-vision-20261022/full-visual-engineering-report.md`

不得修改旧运行时、旧监控、旧本地 run/supervisor，也不得改动旧阶段文件。

## 3. 远程环境与路径

已知远程事实（Sol 已核验）：

- SSH：`ssh -p 27786 root@101.71.76.104`
- 2 × RTX 5090 32GB；503GiB RAM。
- 隔离 venv：`/root/owac/runtime-full-vision`，torch `2.12.1+cu132`，transformers `4.57.0`、omegaconf；不重装 CUDA torch。
- 远程仓库：`/root/owac/repo`；逻辑路径别名 `/home/jeong/zeno/owac/repo -> /root/owac/repo`。
- 训练输出根：`/root/owac/repo/outputs/raw-distillation-full-vision-20261022`
- 源清单：`/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/raw/sources-audited.json`
- 不可变教师 cache：`/home/jeong/zeno/owac/repo/outputs/raw-distillation-align-20261020-153948/sol-cli-final-cache`
- pi05 visual 来源：`/home/jeong/zeno/owac/repo/outputs/checkpoints/raw-distillation/pi05_libero_starvla/bfloat16/model.safetensors`
- 旧学生 checkpoint：仅 provenance，**不传输/不加载**。

运行前检查：

```bash
ssh -p 27786 root@101.71.76.104
test -L /home/jeong/zeno/owac/repo
readlink -f /home/jeong/zeno/owac/repo
test -x /root/owac/runtime-full-vision/bin/python
/root/owac/runtime-full-vision/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

不得把共享发布 RAW 写入 cache、manifest 或训练集；不得复制整个公开数据集。

## 4. 源码同步/冻结

Sol 在源码冻结后只同步本阶段文件，不覆盖其他修改：

```bash
# 在源 checkout 执行；Sol 按实际传输清单审阅后运行
rsync -aR --partial \
  src/owac/runtime/raw_full_visual_distillation.py \
  src/owac/runtime/raw_full_visual_monitor.py \
  tests/test_raw_full_visual_distillation.py \
  tests/test_raw_full_visual_monitor.py \
  configs/experiments/raw_stem_siglip_full_visual_v1.json \
  research/experiments/raw_stem_siglip_20261020/full-visual-20261022-runbook.md \
  root@101.71.76.104:/root/owac/repo/
```

（实际 rsync 目的端按 Sol 的既有映射处理；不要改写配置中逻辑资产路径与 manifest identity。）

远程启动 shell：

```bash
ssh -p 27786 root@101.71.76.104
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/root/owac/repo/src
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
RUN=/root/owac/repo/outputs/raw-distillation-full-vision-20261022
CFG=/home/jeong/zeno/owac/repo/configs/experiments/raw_stem_siglip_full_visual_v1.json
mkdir -p "$RUN"
```

## 5. 真实零更新预检（不计 optimizer 更新）

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/root/owac/repo/src
RUN=/root/owac/repo/outputs/raw-distillation-full-vision-20261022
CFG=/home/jeong/zeno/owac/repo/configs/experiments/raw_stem_siglip_full_visual_v1.json
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.runtime.raw_full_visual_distillation preflight \
  --config "$CFG" --run-root "$RUN" \
  2>&1 | tee "$RUN/preflight.log"
```

预检必须全部满足，否则停止，不得启动训练：

- `preflight.json.optimizer_updates == 0`，没有 optimizer.step。
- `trainable_parameters == 428579952`；参数分组精确等于本文第 1 节。
- `cache_report.rows` 为 train+validation 行数（目标 969；只读、不含 test），`test_rows_preloaded=false`。
- `source.inherited_tensor_count == 437`，`source.inherited_values_equal_after_bf16_fp32_cast == 437`，`source.backend_loaded=false`、`source.teacher_loaded=false`。
- `source.forward_geometry.attention_implementation == "eager"`、`source.forward_geometry.hidden_act == "gelu_pytorch_tanh"`、`source.forward_geometry.layer_norm_eps == 1e-6`、grid `16×16`、`source.forward_geometry.projected_token_order == "camera_major_then_patch_row_major"`。
- `gradient_audit.missing == []`、`nonfinite == []`；gate/stem/position/blocks0/13/26/norm/projector 代表性梯度均非零。
- 两卡 `gpu_by_rank`/`rank_processes` 有真实值；`finalization_reserve_seconds` 和 `gpu_wall_sum_seconds` 已写入 receipt。
- `preflight.json` 实际 SHA 记录到任务 receipt；预检的 GPU wall 必须在训练 1 步时通过 `--preflight-receipt` 计入。

只读审计命令：

```bash
python - <<'PY'
import json, os
run='/root/owac/repo/outputs/raw-distillation-full-vision-20261022'
p=json.load(open(run+'/preflight.json'))
print({k:p.get(k) for k in ('optimizer_updates','trainable_parameters','estimated_full_checkpoint_bytes','finalization_reserve_seconds','gpu_wall_sum_seconds','cache_report')})
print(p.get('gradient_audit'))
print(p.get('forward_geometry'))
print(p.get('source'))
PY
```

不通过则只报告失败，不重试、不自动重启、不启动长跑。

## 6. 唯一授权的 1 次真实更新（计入 3,116）

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/root/owac/repo/src
RUN=/root/owac/repo/outputs/raw-distillation-full-vision-20261022
CFG=/home/jeong/zeno/owac/repo/configs/experiments/raw_stem_siglip_full_visual_v1.json
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.runtime.raw_full_visual_distillation train \
  --config "$CFG" --run-root "$RUN" \
  --preflight-receipt "$RUN/preflight.json" \
  --stop-after-new-update 1 \
  2>&1 | tee "$RUN/step1.log"
```

这一步会：

1. 保存 `checkpoints/step0`（仅 model，审计；不是旧学生）。
2. 真实更新 1 次，anchor 计数 16，budget equivalent 1885。
3. 保存 `checkpoints/checkpoint_final_000001`（model + optimizer + scheduler + 每 rank RNG/sampler + selection/validation）。
4. 更新 `latest.json`、`best/model.pt`、`updates.jsonl`、`validation.jsonl`、`selection.json`、`run-status.json`。

必须核验：

```bash
python - <<'PY'
import json
run='/root/owac/repo/outputs/raw-distillation-full-vision-20261022'
latest=json.load(open(run+'/latest.json')); print('latest',latest)
meta=json.load(open(run+'/checkpoints/checkpoint_final_000001/meta.json'))
print({k:meta.get(k) for k in ('kind','new_global_updates','new_anchor_presentations','budget_update_equivalents','optimizer_path','model_sha256','optimizer_sha256')})
print('world',meta.get('world_size'),'rng ranks',len(meta.get('rng_by_rank') or []))
PY
```

要求：`new_global_updates==1`、`new_anchor_presentations==16`、`budget_update_equivalents==1885`、`kind=="final"`、`optimizer_path=="optimizer.pt"`、两 rank RNG 存在；`latest.json.checkpoint_sha256` 等于 meta model SHA。失败即停，不继续长跑。

然后做**严格重载检查**（1 次更新后、长跑前）：

```bash
# 仍用双卡 torchrun 启动，resume 该 final，目标先只做 0 个新更新（应完成并保留状态）
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.runtime.raw_full_visual_distillation train \
  --config "$CFG" --run-root "$RUN" \
  --resume "$RUN/checkpoints/checkpoint_final_000001" \
  --max-new-updates 1 \
  2>&1 | tee "$RUN/reload-check.log"
```

`--max-new-updates 1` 的 resume 在已达到 1 时执行 **0 次 optimizer step**，加载并核验 model/optimizer/scheduler/每 rank RNG/sampler，写出 `reload-check.json` 后退出；返回 `new_global_updates==1`、`zero_optimizer_updates==true`，最新 checkpoint 未重写。必须确认没有重置为 0、没有重新消费 anchor、没有新建/覆盖 checkpoint。若失败，停止长跑。

## 7. 长跑（从新步骤 1 继续，总计 3,116）

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/root/owac/repo/src
RUN=/root/owac/repo/outputs/raw-distillation-full-vision-20261022
CFG=/home/jeong/zeno/owac/repo/configs/experiments/raw_stem_siglip_full_visual_v1.json
RESUME=$(python - <<'PY'
import json
run='/root/owac/repo/outputs/raw-distillation-full-vision-20261022'
print(run+'/'+json.load(open(run+'/latest.json'))['path'])
PY
)
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m owac.runtime.raw_full_visual_distillation train \
  --config "$CFG" --run-root "$RUN" \
  --resume "$RESUME" \
  --max-new-updates 3116 \
  2>&1 | tee "$RUN/long-run.log"
```

- 不传 `--preflight-receipt`：checkpoint 已含 receipt/budget；重载会恢复。
- 目标 `3116` 是新阶段累计，不是再加 3116；实际已从 step1 开始。
- 不允许自动重启/自动重试；SIGTERM 是唯一优雅停止方式（运行时会在当前更新后保存 final 并退出）。
- 每 250 新更新验证 102 windows/5 sources；每 250 保存 interval；最后保存 final。best 选 validation token total 最小值，平局取更早 step。
- 每小时实际 aggregate GPU wall 必须计入两个 rank；不能只按单卡估算。

## 8. 监控（新 8879；旧 8878 不动）

远程启动：

```bash
cd /root/owac/repo
source /root/owac/runtime-full-vision/bin/activate
export PYTHONPATH=/root/owac/repo/src
RUN=/root/owac/repo/outputs/raw-distillation-full-vision-20261022
nohup python -m owac.runtime.raw_full_visual_monitor \
  --run-root "$RUN" --host 127.0.0.1 --port 8879 \
  > "$RUN/monitor.log" 2>&1 &
```

本地只通过 SSH 隧道查看：

```bash
ssh -p 27786 -N -L 8879:127.0.0.1:8879 root@101.71.76.104
# 然后在本地浏览器打开 http://127.0.0.1:8879/
```

监控只读、固定路由 `/`、`/api/status`、`/healthz`；不列目录、不暴露 checkpoint 内容或路径；不写入训练 run；不与旧 8878 页面交互。页面显示：新阶段 0–3116 更新、428,579,952 全程可训练视觉头、40/5/5 数据、教师/后端冻结、验证 MSE/cosine/total/relative MSE、训练曲线、两卡 allocated/reserved/peak、aggregate GPU 预算、checkpoint selection、真实 rank PID/start ticks/last update age，以及存活时 `native_exit=unknown`。

## 9. 预算、保留与停止

- 预检 0 次更新；1 次更新计入 3,116；长跑从 step1 继续，不能重置计数或预算。
- 远程 aggregate GPU wall cap **140,841.2631939833 s**；最终验证/写 checkpoint 预留由预检实测写入 `finalization_reserve_seconds`。长跑在接近 cap 时先停止并保存 final，不能等 `assert` 抛错后丢失 final。
- 输出 cap **15,396,584,412 B**。checkpoint 采用目录内 `model.pt` + `optimizer.pt`；约 5.15GB/full。保留：`step0` model-only（cap 不足且非 best 时可删除）、`latest` full、`best` model hardlink；最大 latest+best，唯一 inode 计费。保存前实测剩余空间并预剪枝；绝不删除旧本地 run。
- 监控看到 `native_exit=unknown` 是存活期正常值；停止后运行时写状态/退出。不要用历史 PID/页面时间当作实时状态。
- 闭环/动作执行/控制恢复均未授权、未测量。

## 10. 停止和回滚

- SIGTERM 到 torchrun/进程组；各 rank 在下一安全点停止，rank0 保存 final 后退出；不自动重启。
- 不删除旧本地 `raw-distillation-c-full-20261021-123313`，不重启旧 supervisor/monitor。
- 不修改旧学生 checkpoint、cache manifest、teacher artifacts 或共享只读数据。
- 任何 preflight/1-step/reload 失败：停止，保留日志和 receipt，交回工程 review；不要用 CPU 测试替代 GPU 验收。

# N02 v1.2 P2 训练入口与真实动作训练（实现者回报，2026-10-16 Asia/Shanghai）

> 后续更新（2026-10-16）：当前修补版本为 `n02-v12-p2-causal-r4`。r2 和首轮 r3 画像还存在测量查询行列/像素中心错误，其数值保留历史身份。现行代码修补、从头零更新画像见 [r4 报告](P2_fresh_profile_20261016.md)，训练由用户自行开启，使用 [r4 用户入口](P2_r4_user_training.md)。下文旧启动/恢复命令不可直接用于 r4；原 P2 未通过的结论保持。

本文件记录 P2「训练入口与真实动作训练」这一条关键路径的实际实现与实测证据。阶段级总报告由调度者合并到
[P2_report.md](P2_report.md)；本文件只写我负责的部分，命令与输出都是真实执行过的。

## 1 交付文件

| 文件 | 行数 | 作用 |
| --- | --- | --- |
| `src/owac/backends/n02_v12_tokens.py` | 293 | 桥适配器：把冻结的 v1.2 链（`RawEncoderV12`→`N02V12Inference`→`N02ReadoutV12`）作为一次 `TokenSource.__call__` 接到 π0.5 特征桥 |
| `src/owac/training/n02_v12.py` | 2 105 | 真实训练入口：官方 RAW 窗口、真实 π0.5 flow-matching 动作损失、RAW 掩码/因果辅助项、保存恢复、状态输出、正确性检查 |
| `configs/encoders/n02_v12/{raw_n02_robotwin,raw_n02_libero,rgb_n02_robotwin,rgb_n02_libero}.json` | 4 文件 | 真实参数配置（无占位值）；RGB 臂是同一 RAW 经固定已声明 ISP 的受控对照 |
| `tests/encoders/n02_full/test_v12_training.py` | 823 | 20 个 CPU 测试：行记账、单次调用契约、声明的 ISP、掩码分支共享权重与真实无泄漏检查、任务指令读取、运行计划、状态字段、以及**在微型真实模块 + stub π0.5 上端到端跑 `train()` 并验证保存/恢复** |
| `/etc/supervisor/conf.d/owac-n02-v12-raw.conf`、`owac-n02-v12-raw-profile.conf`（远端） | — | supervisor 常驻：主作业（GPU0）与 200 更新画像（GPU1），`autorestart=false` |

## 2 训练用的确切命令与路径

解释器：`/workspace/owac/worktrees/n02-measurement-completion/outputs/n02-full/runtime/py310-cu128/bin/python`
（Python 3.10.19 / torch 2.11.0+cu128）。

```bash
cd /workspace/owac/worktrees/n02-measurement-completion/src          # 注意：cwd 必须是 src
export PYTHONPATH=/workspace/owac/worktrees/n02-measurement-completion/src:/workspace/owac/rawvla/source/starVLA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../outputs/n02-full/runtime/py310-cu128/bin/python -m owac.training.n02_v12 \
  --config ../configs/encoders/n02_v12/raw_n02_robotwin.json \
  --seed 1103 --max-updates 1000 --profile-updates 200 \
  --validate-every 50 --save-every 1000 --previews 1 \
  --output ../outputs/n02-v12/P2/raw-n02-robotwin-s1103
```

* 数据：`/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench`（官方 RawVLA-Bench，4810 文件 / 713 689 392 570 B，下载状态 complete）。
  冻结 P0 分区 manifest 收据 `/workspace/.../outputs/n02-v12/P2/measurement-completion/manifest-robotwin.json` + 冻结锚点
  `outputs/n02-full/p0-independent-audit/frozen-data/robotwin-{train,dev,calibration}-anchors.json.gz`（train 116 117 / dev 15 414 / calibration 15 413 锚点）。
* π0.5：RoboTwin `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`，统计同目录
  `policy_norm_stats.json`；tokenizer `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model`。
  加载审计：`frozen=True`、冻结参数 3 616 757 520、`active_missing=0`、`replaced_vision_present=439`。
* 显存：主作业 `CUDA_VISIBLE_DEVICES=0`（进程内 `cuda:0`），画像作业 `CUDA_VISIBLE_DEVICES=1`。每进程实测峰值约 58 GiB，故一卡一进程。
* 单次调用契约：`run_stats = {prefix_builds: 1, encoder_calls: 1, token_source_calls: 1, denoise_steps: 0}`，即每个决策时刻只跑一次 v1.2 链。

## 3 哪些参数在训练、哪些冻结

* 冻结（`freeze_backend=True`）：整个 π0.5 后端（语言栈、动作专家、动作/时间嵌入）3.62 G 参数；训练中每步断言
  `frozen_parameters_with_grad == []`（实测为空），违反即报错。
* 训练：`RawEncoderV12`（ConvNeXt V2-Base，raw 从零初始化，无 RGB 权重/教师）+ 3 个 1×1 层级投影、
  `N02V12Inference`（K 个候选种子、evidence/prior/residual 投影、SAVi corrector/predictor）、
  `RawMeasurementDecoderV12`、`N02ReadoutV12`（Perceiver）、FCMAE 掩码分支自有参数（mask token + 解码器 + 像素头），
  模型合计 106 324 683 参数。
* 硬路径如实记录：`model.gradient_paths()` 里候选 argmax 与接受/拒绝是硬选择，不带梯度；不是一个投影有梯度就宣称搜索可训练。
* 掩码分支与动作分支共享 **同一批 backbone 张量**（实测绑定 376 个，未绑定 0 个），因此 `L_masked_raw` 训练的是动作分支的
  backbone，不是第二份拷贝；掩码语义用 P0 的 mask-aware 端口（support 记账 + 只对有效位置统计 LayerNorm/GRN），
  未改成零填充。

## 4 训练目标

```text
L = L_action(π0.5 原生 flow-matching, 官方 noise β(1,1) / time Beta(1.5,1), 逐动作 mask)
  + λ_raw  · L_masked_raw          (FCMAE 掩码 RAW 重建，只在被遮挡且有效、未饱和的像素上取目标)
  + λ_pred · L_causal_raw_prediction(看到目标帧之前锁定的前缀特征 → 当前帧查询位置的 Gaussian NLL)
  + λ_noise· L_noise_calibration   (λ=0：官方缓存未声明噪声模型，不虚构真值，配置与状态里写明原因)
```

`L_causal` 的输入是**前缀行（t-7…t-1）的特征**、当前帧的**查询位置**与因果帧号；当前帧特征绝不进入该预测，
当前帧像素只作标签。单位：动作/本体沿 loader 的官方统计，训练**不重复归一化**（`prepare_control_targets` 只补 32 维）。

## 5 正确性检查（只保留影响正确性的）

`checks.json` 实测（RoboTwin，train 分区）：

* 单位与 shape：三相机 `(8,3,240,320) float32`，值域与 loader 一致（未除 255、未逐帧 max），动作 `[50,14]`、
  `action_offset=1`、逐动作 mask 有效 50；全部 finite。
* 因果性：12 个窗口全部通过（历史行是早于或等于锚点的真实帧号，从不读未来）。
* 分区隔离：train∩dev、train∩calibration、dev∩calibration 的测量 ID 交集均为 **0**。
* episode reset：选中行清零、未选中行逐位不变、version 递增，全部为真。
* 先验接线：同一真实窗口、同一权重，中性先验 vs 偏置先验 `max|Δlog p0| = 2.015`、`max|Δq| = 0.646`，
  先验确实进入 `log_p0` 与 `q`。
* 梯度接线（首个真实反向）：encoder L1 = 1.107e6、readout 7.48e5、proposals/corrector 4.59e3、decoder 30.1、
  masked 分支 12.7，全部 finite；冻结 π0.5 参数无梯度。
* 一次真实训练步：loss 3.215144 = action 2.472532 + masked_raw 0.005272 + causal_raw 0.737340（noise 关闭），
  动作 shape `1×50×14`（padding 前）/ 内部 32 维，`valid_target_count = 414 720`、饱和样本 0。
* 校准：在独立 calibration 分区拟合窗口阈值（4 条序列，阈值 0.2774，rule hash `92c3be36…`），并在 calibration 与
  **dev 留出序列**上真实调用得到三态判决（calibration 4/4 compatible；dev 3–4/4 compatible、其余 model_inadequate）。
  **未把该 calibrator 装进每步检查**：冻结的 `ViewingRule` 要求序列里出现它声明的每个 `(scale, frame, region)` 视图，
  而 `inference_v12._residual_views` 用的是决策时刻的真实源帧号，一条冻结规则无法覆盖训练中不断变化的帧号（不匹配会抛错）。
  因此每步检查显式为未校准的 `AbsoluteCheckResult.invalid`，与 0 误报/0 检出分开计数，绝不伪装成通过。

## 6 画像与实跑（真实命令、真实输出）

### 6.1 25 更新试点（GPU0，microbatch 1 / grad_accum 1）

```
step_seconds_mean 2.728  p50 2.530  p95 3.256   wall 68.2 s / 25 updates
peak_gpu_bytes 59 531 499 520 (≈55.4 GiB)
dev_action_error 1.676 (step 5) → 1.124 (step 10) → 1.261 (step 15) → 0.517 (step 25)
```

据此选定 **microbatch 1 锚点 × grad_accum 2**（有效 batch 2 锚点，约 5 s/update），并写进四份配置的 `loop` 段；
一进程一卡（峰值 58 GiB/95 GiB）。

### 6.2 200 更新画像（GPU1，microbatch 1 / grad_accum 2，独立 run 目录）

```
run_id raw-n02-robotwin-s1103-profile   wall 906.27 s / 200 updates
step_seconds mean 4.531  p50 4.307  p95 6.145    peak_gpu_bytes 59 973 648 896 (≈55.9 GiB)
last-10 loss 7.031   terms {action 0.3379, masked_raw 0.0279, causal_raw 6.6648}
dev_action_error (200 更新末, 固定 4 条官方 dev 窗口) 0.2694
checkpoint …/raw-n02-robotwin-s1103-profile/checkpoint_0000200.pt
```

`L_masked_raw`/`L_causal_raw` 是 RAW 域的目标（Gaussian NLL 含 log 方差项，可为负），与 π0.5 的动作 MSE 单位不同，
因此页面上的"训练 loss"是加权总和；判读能力看 dev 动作误差（同一固定 dev 集合，固定 flow 噪声/时间）。

### 6.3 1000 更新主作业（GPU0，`raw-n02-robotwin-s1103`，**已完成**）

```
state completed   step 1000/1000   activity 训练完成 1000 updates
wall 4 663.18 s (77.7 min)   step_seconds mean 4.663  p50 4.325  p95 6.316   peak_gpu 59 973 648 896 B (55.9 GiB)
最后 10 步均值 loss 7.795   terms {action 0.112676, masked_raw 0.065712, causal_raw -0.075839}
固定 dev 动作误差 0.948(50) → 0.349(150) → 0.289(350) → 0.200(600) → 0.1216(750) → 0.0924(1000)
checkpoint …/raw-n02-robotwin-s1103/checkpoint_0001000.pt   1 327 326 203 B
「动作全为 padding」的锚点跳过 21 个（真实缓存属性，计入 skipped_all_padding_anchors）
```

训练参数量：encoder 88 153 088 + decoder 1 243 851 + readout 7 173 632 + proposals/corrector 9 754 112 +
masked 分支 4 231 680 = 110 556 363；冻结 π0.5 3 616 757 520、其中带梯度 0 个。

**如实记录**：step 567–568 出现过一次瞬时 loss 尖峰（42.39 / 30.01，来自 RAW 域 Gaussian NLL 的 log 方差项），
~20 步内自行回到 0.1 以下，dev 动作误差未受影响（600 步 0.2003 为当时最好值）；梯度裁剪 1.0 全程生效。
`L_masked_raw`/`L_causal_raw` 与动作 MSE 单位不同，页面上的"训练 loss"是加权总和，**不能**当作能力分数。

### 6.4 受控 RGB 对照臂（同一 RAW + 固定已声明 ISP）

`rgb_n02_robotwin.json` 的 `representation.kind="raw3_fixed_isp"`：对**同一次采集的同一 RAW** 施加写死的
`clip(x/1.0)→per-channel gain→gamma 2.2→8 bit 量化`，不含任何逐帧统计；不是官方正常光配对 RGB。
先做 5 更新冒烟（`rgb-n02-smoke-s1103`，真实数据 + 真实 π0.5，完成，dev_action_error 0.880），再用与 RAW 完全
相同的预算（1000 更新、seed 1103、microbatch 1、grad_accum 2）在 GPU1 上跑 `rgb-n02-robotwin-s1103` 作为对照臂
（supervisor 程序 `owac-n02-v12-rgb`）。两个臂都跑满 1000 更新（同预算、同 seed、同 microbatch/accum、同一批真实数据分区）：

| 指标 | RAW `raw-n02-robotwin-s1103` | RGB `rgb-n02-robotwin-s1103` |
| --- | --- | --- |
| 更新数 | 1000 / 1000 | 1000 / 1000 |
| 墙钟 | 4 663.18 s | 4 800.30 s |
| step 均值 / p50 / p95 | 4.663 / 4.325 / 6.316 s | 4.800 / 4.415 / 6.429 s |
| 峰值显存 | 59 973 648 896 B | 59 973 648 896 B |
| 训练参数 | 110 556 363 | 110 556 363 |
| 跳过全 padding 锚点 | 21 | 21 |
| 固定 dev 动作误差 @1000 | **0.0924**（末次固定测量 0.08049） | **0.0785** |
| checkpoint(1000) | 1 327 326 203 B | 1 327 326 587 B |

同步骤 dev 轨迹：RAW 0.948(50) → 0.349(150) → 0.289(350) → 0.200(600) → 0.0924(1000)；
RGB 0.750(50) → 0.463(150) → 0.236(200) → 0.178(300) → 0.115(550) → 0.0785(1000)。

**如实报告：在这个 1000 更新预算与这个固定 dev 动作误差诊断上，没有观察到 RAW 相对受控 RGB 的优势**
（1000 步时 RGB 反而略低 0.014）。这**不是**能力结论：dev 动作误差是接线/优化诊断，P2 的检查点本来就不是
合格线，两臂也都没有做独立确认或闭环；RAW 的潜在信息优势需要 P3 的充分训练与 P4 的 100 次闭环来判定，
本阶段只报告"两臂都真实跑通、数字在这里"。

## 7 run 目录、状态输出与恢复

* 主作业 run_id `raw-n02-robotwin-s1103` → `outputs/n02-v12/P2/raw-n02-robotwin-s1103/`
  （`status.json`、`history.jsonl`、`profile.json`、`checks.json`、`report.json`、`checkpoint_*.pt`、预览 PNG）。
* 画像作业 run_id `raw-n02-robotwin-s1103-profile` → 同级的 `…-profile/`。
* 状态写入走冻结的 `owac.runtime.n02_v12_status`：每 10 个 optimizer update 或 5 秒更新一次，长步骤由后台线程每 5 秒
  heartbeat，每次 dev/保存/异常立即更新；`history.jsonl` 追加 `step,wall_time,train_loss,dev_action_error`。
* `eval` 字段恒为 `{planned:100, completed:0, success:0, task_failure:0, infrastructure_error:0}`（100 次闭环是 P4，本阶段未跑）。
* supervisor：`owac-n02-v12-raw`（主作业，GPU0）、`owac-n02-v12-raw-profile`（画像，GPU1），两者 `autorestart=false`，
  出错时 supervisor 显示 FAILED 并在 `status.json` 里留下 `state=failed` 与原因。
* 恢复命令（从 checkpoint 接着跑）：

```bash
cd /workspace/owac/worktrees/n02-measurement-completion/src
PYTHONPATH=$PWD:/workspace/owac/rawvla/source/starVLA CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../outputs/n02-full/runtime/py310-cu128/bin/python -m owac.training.n02_v12 \
  --config ../configs/encoders/n02_v12/raw_n02_robotwin.json --seed 1103 --max-updates 1002 \
  --resume ../outputs/n02-v12/P2/raw-n02-robotwin-s1103/checkpoint_0001000.pt \
  --output ../outputs/n02-v12/P2/raw-n02-robotwin-s1103
```

## 7.1 保存/恢复实测（跨进程，真实数字）

```
$ … -m owac.training.n02_v12 --config …/raw_n02_robotwin.json --seed 1103 --max-updates 1002 \
    --resume …/raw-n02-robotwin-s1103/checkpoint_0001000.pt --output …/raw-n02-robotwin-s1103
[20:03:24] resumed …/checkpoint_0001000.pt at step 1000 (dataset cursor 2021)
[20:03:25] save/resume: dev action max |diff| = 0.000e+00; dev action error 0.092403 -> 0.092403
[20:03:34] saved checkpoint …/checkpoint_0001001.pt (1327.3 MB) at step 1001
[20:03:42] saved checkpoint …/checkpoint_0001002.pt (1327.3 MB) at step 1002
```

同一后缀动作**逐元素最大绝对差 0.000e+00**（同一固定 dev 窗口、同一固定 flow 噪声/时间、同精度），
恢复后继续跑完 1001/1002 两个更新并再次保存。`checks.json.save_resume` 里保存了同一组数字。
恢复过程同时修好并验证了预览写出：`preview_raw_robotwin_head.png`、`preview_attention_robotwin_head.png`
（真实 encoder 注意力叠加，非人造圆斑）。

## 8 启动目录守卫（远端陈旧 `owac/`）

远端 worktree 根有一个陈旧未跟踪的 `owac/` 目录，`python -m` 从该目录启动会导入它而不是 `src/owac`（实测
`import owac` → `<root>/owac/__init__.py`）。`main()` 第一件事就是检查 `owac.__file__` 是否位于 `<repo>/src/owac`，
不是就打印实际路径、期望路径与两个可用的替代启动方式（`cd <repo>/src`，或直接按路径运行源文件）并以退出码 3 失败；
没有移动/删除那个目录。实测：从 worktree 根运行 `--checks-only` → 退出码 3 并打印上述信息；`--help` 两种方式都能跑。

## 9 测试与检查

* `uv run --locked pytest tests/encoders/n02_full/test_v12_training.py -q` → **20 passed**（含端到端 `train()` + 保存/恢复）。
* `uv run --locked pytest tests/encoders/n02_full -q` → 582 passed / 6 skipped / 4 failed；4 个失败都在
  `tests/encoders/n02_full/test_v12_eval.py`（另一位实现者同期的文件，不属于我的改动），我新增的文件全绿。
* `make check`：我的三个源码/测试文件 `ruff check` 与 `ruff format --check` 全绿；整仓仍有其他人的未清 lint
  （`src/owac/evaluation/n02_full/v12_viz.py`、`tests/encoders/n02_full/test_v12_eval.py` 等 E501/E402），未去改动他们。

## 10 未做 / 失败项（如实）

* 100 次闭环评测未做（P4）。
* `λ_noise` 恒为 0：官方缓存未声明噪声模型；代码在有声明噪声模型时会直接报错要求实现censored 项，而不是造一个真值。
* 每步三态检查保持"未校准"（原因见 §5），只有校准/留出序列上的三态判决是真实的。
* RGB 臂已跑：5 更新冒烟 + 与 RAW 同预算的 1000 更新对照（`owac-n02-v12-rgb`），但**未做能力结论**。
* 每步 `preview_files` 由评测侧集成写入 `preview/*.svg`（RAW/同源 RGB 放大、注意力、预测残差、轮次/支持、动作曲线），
  我这一侧只写 `preview_raw_*.png` / `preview_attention_*.png`；两者都在 run 目录里，页面按 `preview_files` 显示。
* `status.json` 里的 `dev_action_error`/`activity`/`eval_*` 在评测侧集成跑过后会被它更新为评测自己的 dev 数字
  （我看到 0.17435，与我在 `report.json` 里记录的最终 0.08049 不是同一次测量）；以 `report.json` / `checks.json`
  的固定 dev 数字为准。
* 三个 supervisor 作业现在都是 `EXITED`（正常跑完，非 FAILED）：`owac-n02-v12-raw` 19:57Z、`owac-n02-v12-rgb` 20:25Z、
  `owac-n02-v12-raw-profile` 18:56Z；`autorestart=false` 保证它们不会自己再起。
* 数据读取成本：每个锚点窗口经公开入口读两次（`window()` + `targets()`，第二次命中页缓存），未使用对方模块的私有方法。
* 上游冻结接口的两个小坑已按不修改他人文件的方式绕过，并在此记录：`inference_v12._declared_ids` 的空元组默认值在
  batch>1 时会抛错（本入口显式传 `None`）；`V12WindowSource.targets()` 的 mask 对单锚点是 1-D（本入口按锚点数 reshape）。
* run 目录里保留了失败尝试留下的 `history.jsonl` 行（第一次因"锚点动作全为 padding"报错，已改为跳过并计数，见
  `skipped_all_padding_anchors`）。

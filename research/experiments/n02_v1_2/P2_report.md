# N02 v1.2 P2 交付报告：完整编码器与真实 VLA 链路

> 后续更新（2026-10-16）：当前修补版本为 `n02-v12-p2-causal-r4`。r2 和首轮 r3 画像还存在测量查询行列/像素中心错误，其数值保留历史身份。现行代码修补、从头零更新画像见 [r4 报告](P2_fresh_profile_20261016.md)，训练由用户自行开启，使用 [r4 用户入口](P2_r4_user_training.md)。下文旧启动/恢复命令不可直接用于 r4；原 P2 未通过的结论保持。

> **2026-10-16 独立验收覆盖原交付判断：P2 未通过。** 原动作训练未使用历史状态，测量目标漏采且平均通道，因果检查、重建和独立评测仍有缺陷。现已修补并完成有界真实复验，尚未完成修补版 1000 步。以下保留 DeepSeek 原始交付及当时的结论，不能用其中“完整”“六类全部修复”覆盖[独立验收报告](P2_independent_acceptance_20261016.md)。

日期：2026-10-16（Asia/Shanghai）。对象是本轮 P2 实现。P0 已验收范围不变；P1 独立验收仍为
**未通过**，其六类缺陷在本阶段修复。本轮**只执行 P2**：不启动 P3、不提交、不推送。

阅读时请先分清三件事：**完整代码**（已实现并在真实数据/真实 π0.5 上跑通）、
**已训练程度**（本轮预算与训练状态）、**实际能力**（有没有 RAW 增量或机制增量）。
本阶段没有做能力结论，也没有做 100 次闭环。

## 0 结论速览

| 项目 | 状态 |
| --- | --- |
| 完整 v1.2 网络（RAW3→ConvNeXt V2-Base→多候选→RAW 测量预测→循环修正→Perceiver 读出→π0.5） | 已实现，真实前向/反向跑通 |
| P1 六类缺陷 | 逐条修复，见 §3，均有实测证据 |
| 真实数据与 P0 冻结分区 | 重建 manifest 与 P0 冻结值**逐位一致**，独立核实 |
| 真实 π0.5 后端 | RoboTwin 与 LIBERO 双平台真实加载并反向 |
| 真实动作训练 | **`raw_n02` 已跑完 1000 updates（seed 1103）**，见 §6 |
| 受控同源 RGB 对照 | **`rgb_n02` 也已跑完 1000 updates**；等步数对比下**未看到 RAW 优势**（RAW 0.0924 vs RGB 0.0785），见 §6.6 |
| 监控网页 | 已上线 `127.0.0.1:18080`，独立复核通过，并已切换到真实 run |
| 诊断可视化 | 固定 dev 官方轨迹 **44 张真实图**（含真实 attention），已发布到真实 run |
| 100 次闭环 | **未运行**（属 P4） |
| 四臂归因（`raw_base` / `rgb_base`） | **未运行** |

## 1 交付物与代码规模

新增/改造文件（本地工作树 = 远端工作树，分支 `lzh/n02-measurement-completion`）：

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `src/owac/frameworks/n02_full/encoder_v12.py` | 431 | RAW3 多尺度空间编码（stride4/8/16） |
| `src/owac/frameworks/n02_full/decoder_v12.py` | 1036 | 相机/时间条件 RAW 测量解码器（SAVi 移植） |
| `src/owac/frameworks/n02_full/readout_v12.py` | 956 | Perceiver IO 读出（官方移植） |
| `src/owac/frameworks/n02_full/inference_v12.py` | 1495 | 候选提议、循环修正、接受规则、状态重建 |
| `src/owac/frameworks/n02_full/model_v12.py` | 517 | 主入口 `N02V12Model` 与统一检查返回类型 |
| `src/owac/frameworks/n02_full/data_v12.py` | 637 | 真实窗口、P0 冻结分区、因果历史 |
| `src/owac/frameworks/n02_full/checks_v12.py` | 772 | 完整测量目标与三态绝对检查 |
| `src/owac/frameworks/n02_full/interventions_v12.py` | 589 | 真实测量干预（像素级） |
| `src/owac/backends/n02_v12_tokens.py` | 280 | 特征桥适配器（联合多相机 → π0.5 tokens） |
| `src/owac/training/n02_v12.py` | 2123 | 训练入口 |
| `src/owac/evaluation/n02_full/v12.py` | 1823 | 评测入口 |
| `src/owac/evaluation/n02_full/v12_viz.py` | 1102 | 诊断可视化 |
| `src/owac/runtime/n02_v12_monitor.py` / `.html` | 719 / 484 | 实时监控页（前端专家） |
| `src/owac/runtime/n02_v12_status.py` | 144 | 训练端状态写入 helper |
| 测试 `tests/encoders/n02_full/test_v12_{data,eval,inference,training,vision}.py` | 564+533+663+513+666 | 真实数值/接线测试 |
| `configs/encoders/n02_v12/{raw,rgb}_n02_{robotwin,libero}.json` | 4 个 | 实际参数配置 |

组件来源与复用：ConvNeXt V2-Base 与 FCMAE 复用 P0 已验移植；SAVi `SlotAttention`/`Processor`/
`TransformerPredictor` 复用 P0 移植，**spatial broadcast decoder 为新增移植**；Perceiver IO
核心（`attend`/`SelfAttention`/`CrossAttention`/`PerceiverEncoder`/`BasicDecoder` 与
`position_encoding`）为**新增移植**，文件头保留 Apache-2.0 与上游 commit。
本仓旧 `n02/readout.py` 是 token readout、**不是** Perceiver，未用于正式读出。
逐文件去向见 [reuse_map_P2.md](reuse_map_P2.md)。

参数量（实测）：encoder **88,153,088**（骨干 **87,692,800** = 官方 Base 无分类头）、
decoder 1,243,915、readout 7,174,656；完整 N02 前端 **106,324,683**。

## 2 真实数据边界（实测，不是声明）

| 项目 | RoboTwin | LIBERO |
| --- | --- | --- |
| 相机 | 3（head/left/right），各 `[8,3,240,320]` float32 **原尺度**（不除 255、不逐帧 max） | 2（agentview/wrist），各 `[8,3,256,256]`，uint8→FP32/255 |
| 动作 | 14 维，horizon **50**，offset **1** | 7 维，horizon **10**，offset **0**，每 5 步重规划 |
| 本体 | 14 维在线关节 | 8 维；**92 维仿真 `states` 从未进入读数** |
| 归一化 | checkpoint 明确 **MEAN_STD**，沿用 loader 统计 | quantile；夹爪 `(a+1)/2` |
| 历史 | 8 帧因果 | 8 帧因果 |

单位边界（已核实）：`RawVLAWindowDataset` 用正式统计后 sample 的 state/action **已经归一化**；
`prepare_control_targets()` 只补齐 32 维、`control_actions()` 只切回 7/14 维，**都不做归一化或
逆归一化**。训练不重复归一化；逆归一化由在线执行端负责。名为 `control_actions` 的切片不是物理控制量。

两个平台的实际表示：LIBERO 是 RGB8 逆处理的三通道 pseudo-RAW，RoboTwin 是 HDR-derived 三通道
float32 测量，**都不是原生 Bayer**。

### 2.1 P0 冻结分区（独立核实）

远端 `outputs/n02-full/p0-independent-audit/frozen-data/` 只有 6 个 anchors，**没有** manifest。
用 P0 同一入口 + seed 1103 重建，再与冻结收据对照：

| 平台 | 重建 manifest hash | 与 P0 冻结值 | 分区规模 |
| --- | --- | --- | --- |
| RoboTwin | `84433c778b0f2c40c127614c36b189dd67396dfd4fe656cbb53e1e3895867602` | **逐位一致** | train/dev/cal = 500/65/67 集，116117/15414/15413 anchors |
| LIBERO | `ff61cef7ad75e6d71581089677b05a1cd0fdb662e89d2e6a91a0d5c6fee26b52` | **逐位一致** | 1392/189/190 集，204301/28394/28436 anchors |

本报告的核实方式：直接读本地 P0 收据 `2026-10-14_n02_p0_independent_acceptance/frozen-data.json`
与 `manifest-*.json.gz`，与远端重建收据 `partition-verification.json` 对照，三处一致；
三区 anchors 的 signature（`relative_path, current_frame_index, action_start, action_stop,
action_mask_hash`）逐条相等，`verify_partitions` violations 为空。

### 2.2 分区隔离（真实 RAW 内容，不是文件名）

每平台每区取 8 场景，跨区逐元素比较 + 每相机 SHA256：
RoboTwin train↔dev / train↔cal / dev↔cal 各 192 个相机比较，**identical_content_pairs = 0**，
最小 `mean|Δ|` 3.58e-4 / 5.05e-4 / 6.22e-4，digest 交集 0；LIBERO 各 128 个比较，identical = 0。
最接近的一对仍为 1840320/1843200 个元素不同、SHA256 不同。

## 3 P1 六类缺陷的实际修复

| # | P1 错误 | 修复 | 实测证据 |
| --- | --- | --- | --- |
| A | `locked_checks` 存无 verdict 的 `MeasurementScore`；评测读 `.verdict`（实际字段 `.status`）→ 全部变字符串 `None` → 被计成 0 误报/0 检出；`WindowedCalibrator` 正式链路从未调用 | 统一真实返回类型（`status` + `valid` + `statistics` + `calibration_version`，带 `.verdict` 属性，invalid 时返回显式 `"invalid"`）；`model.step_bundle` 改存该类型；缺失 verdict → `valid=False` 并**单独计数**；三态分开判定 | 真实 dev：`compatible 24 / insufficient 0 / model_inadequate 0 / invalid 0`；未校准场景真实给出 `valid=False, reason=no_fitted_calibrator`；全 JSON **无字面量 `"None"`** |
| B | `declared_prior` 只写进 `task.inference.declared_prior`，从未进入 `log_p0`；同一测量下两个合法先验逐位相同 | 先验真实进入 `log_p0` 并参与 `q=softmax(log_p0−E/tau)`；四候选先验归一化（总和 = 1） | 同一真实样本、两个合法归一化先验（候选0 / 候选1 各 0.95）：`q=[0.9508,0.0158,0.0165,0.0169]` vs `[0.0175,0.9473,0.0173,0.0178]`，**max\|Δ\| = 0.9333**（P1 为 0） |
| C | cue swap 只重排索引；曝光/噪声/shift 只是字符串标签；no_information 只是对普通序列首帧打分 | 干预在**真实 RAW 像素与历史行**上发生并记录位置/强度；构造真实无信息测量 | cue_swap eps=1.0/0.5 的 `raw_diff_norm` 为 206.35（像素级，历史行同改）；eps=1.0/0.5 线性 2:1；no_information 458.19；exposure gain=2.0 893.29；noise σ 23.99；translation (8,−10) 255.41 |
| D | 训练与评测 synthetic 同 seed 同 partition，实测前 16/16 场景 RAW SHA 相同 | 主实验切官方数据 + P0 冻结分区；train/dev/cal 实测零重叠 | §2.1、§2.2 的真实 RAW 内容比较 |
| E | 错误候选带 `log_exposure=0.2*(index+1)`、正确候选默认 0，该数值进入候选编码器 → 32/32 字典捷径 | v1.2 提议来自**可学习 latent + 观测证据 + 因果前态 + 残差反馈**，不读真值 | `inference_v12.py` / `model_v12.py` / `data_v12.py` 中 `truth` 引用计数 **1 次（仅在 docstring 说明不读）**；`build_p1_task` 降为历史命令 |
| F | stride8 的 64×64 稀疏查询只看 128/8192 标量且固定 CFA 奇偶位 | 目标对**全部有效单元**求和，报告实际覆盖；三态残差图为完整传感器分辨率 | 实测 coverage `n_used/n_valid` = 1.0；残差图 240×320（230400 有效样本）与 256×256（131072） |

### 3.1 一个必须写明的真实研究事实（不是接线 bug）

真实 RoboTwin 样本上，该测量集**总目标 ≈ 1.4e4**（24576 个 query，每标量 NLL ≈ 0.57），
而候选间能量差只有 **≈ 0.05**。`tau = 1` 时**先验在 `q` 中占主导**。
这是能量尺度/有效样本数的问题，不是接线错误：`energies` 与先验无关（同测量集上 max|Δ| = 0），
`q` 的变化确实只来自 `log_p0`。P3 需要显式调 `tau` 或按有效样本数缩放后再谈「测量驱动」。

### 3.2 接受规则与理论边界（实测）

4 轮内循环，判据是完整目标 `J = −tau·logsumexp(log_p0 − E/tau)`（即 `min_q E_q[E] + tau·KL(q‖p0)`）：
轮 0/1 接受（14032.68 → 14029.44 → 14023.16），轮 2/3 拒绝并**保留旧状态**，
`accepted_total = 2/4`，`all_accepted_non_increasing = True`。
本轮只承诺适用条件内的性质（固定有限候选与测量）；不声称全局收敛，不要求训练 loss 单调。

## 4 监控网页（本阶段交付的一部分）

- 地址：远端 `http://127.0.0.1:18080/`；用户本地经 SSH `-L 8081:localhost:18080` 打开 `http://127.0.0.1:8081/`。
- 常驻：supervisor `owac-n02-v12-monitor`（`autorestart=true`），在**同一入口**上升级替换了旧 bootstrap 页，
  旧页 `owac-n02-monitor` 已置 `autostart=false`（文件保留），8080 平台 Jupyter 未动。
- 接口：`GET /api/status`、`GET /api/history?run_id=...`、`GET /api/preview?...`；只读配置的 run 目录，
  路径穿越（`../../etc`、`/etc/passwd`、URL 编码变体、软链接逃逸）全部 400 且不泄漏内容。
- 页面结构：①阶段/在做的事/心跳/状态 ②RAW 与 RGB 两列卡片（GPU、真实 step/目标、ETA、checkpoint、loss、dev 误差）
  ③训练 loss 与 dev 误差分图曲线 ④固定官方轨迹的 RAW/同源 RGB/attention/残差预览 ⑤闭环 n/100 与最近 30 行日志。
- 未运行写未运行；分母只算已完成；停更 >30 秒显示「数据未更新，请查看日志」而**不判训练失败**。

## 5 诊断可视化

固定 dev 官方轨迹（`open_laptop/episode_000044` 等），run 目录 `preview/` 下 **33 张 SVG**：
RAW 固定显示映射 + **同源 RGB**（由该次**同一 RAW** 经声明 ISP `clip(x·gains/white_level,0,1)**(1/2.2)`，
常量首帧冻结后复用，**不使用配对 RGB**）+ 局部放大；**真实** Perceiver encoder cross-attention
叠加（`readout_aux["encoder_attention"]`，head 平均后对 128 latent 求均值，显示前按帧最大值归一，
token→像素用同一 forward 的 `SpatialV12.coords`）；候选 RAW 预测 + 噪声尺度残差；每轮 energy/support
与 accept 标记；π0.5 真实采样 vs 参考动作块。每张图标 `episode / camera / frame / iteration / model step`。
页面 `/api/status` 显示 9 个 tile 且 `/api/preview` 返回 200。

### 5.1 用真实 1000 步 checkpoint 重跑的评测与图

在 `checkpoint_0001000.pt`（独立 `sha256 = 690271b6c351fdd2a8fadd55abc16f6dc558724e107cc7294d4a456d78dfbb3c`，
与评测收据内部记录一致）上重跑，图写入**真实 run** 的 `preview/`：**44 张 SVG + `figures.json`**
（4 个固定 dev 锚 × 3 相机 × {RAW 固定映射/同源 RGB/局部放大、真实 attention、候选预测+残差}
= 36，加 4 张每轮 support + 4 张动作曲线）。页面该 run 的 `previews` = 12 tiles + 8 other，
`options = {cameras: 3, frames: [164,18,163,60], loops: [3]}`，
`/api/preview?run_id=…&file=preview%2F01_raw_rgb_zoom_12331_robotwin_head.svg`
→ **http=200, image/svg+xml, 482332 bytes**。

真实评测数字（固定 dev 锚 3082/6165/9248/12331，t 固定 0.5，噪声 seed 1103）：
dev 动作误差 **mean 0.17435**、median 0.181496、四个值 `[0.030881, 0.183132, 0.303528, 0.179859]`，
`not_run = []`；三态 **compatible 4 / insufficient_information 0 / model_inadequate 0 / invalid 0**
（全行口径 32/32 compatible）；`status.json`、`eval_v12.json` 与 44 张 SVG 中字符串 `"None"` 出现 **0 次**。
真实干预 `raw_diff_norm`（每类 2 个样本）：cue_swap `[0.2588, 191.87]`、no_information `[0.7343, 382.95]`、
exposure_shift `[0.9560, 1074.51]`、noise_shift `[0.4687, 34.667]`、translation_shift `[0.6936, 316.50]`。

**诚实边界**：本轮的 4/4 全 `compatible` 且 `invalid = 0`，因此**无效 verdict 路径没有被这次运行触发过**，
不能用它证明「非零 invalid 会被正确单列」——该路径只有代码与契约说明，见 §3 缺陷 A 与 W3 的探针证据。
本轮三态计数区分度低，原因见 §10 第 5 条。

## 6 真实动作训练（实际结果）

本轮跑完 **1000 optimizer updates（seed 1103，真实官方 RoboTwin 训练分区）**，
产物：`outputs/n02-v12/P2/raw-n02-robotwin-s1103/checkpoint_0001000.pt`
（1,327,326,203 bytes）。supervisor 程序 `owac-n02-v12-raw`（GPU0，`autorestart=false`），
RGB 受控对照 `owac-n02-v12-rgb`（GPU1）同步在跑。**这是可视化检查点，不是能力合格线，
也不是「充分训练」**；P3 从它恢复并扣掉已做更新。

### 6.1 实测资源与画像（先 200 updates 画像，再跑 1000）

| 项目 | 实测值 |
| --- | --- |
| 1000 updates 主作业 | wall **4663.18 s（77.7 min）**，step mean **4.663 s** |
| 200 updates 画像（主作业自身） | wall **906.3 s**，step mean **4.531 s** / p50 4.307 / p95 6.145 |
| 200 updates 独立画像 run（GPU1） | wall 985.24 s，step mean 4.926 s |
| microbatch × 累积 | **1 锚点 × grad_accum 2**（有效 2 锚点），一进程一卡 |
| 峰值显存 | **59,973,648,896 B ≈ 55.9 GiB**（单卡，GPU0） |
| 画像 loss 组成 | action 0.3379 + masked_raw 0.0279 + causal_raw 6.6648 |
| 训练参数 / 冻结参数 | 训练 **110,556,363**；冻结 π0.5 **3,616,757,520**（`freeze_backend=True`） |
| 梯度裁剪 | 1.0，全程生效 |
| 跳过全 padding 锚点 | 21 个（计数，不静默丢弃） |
| 瞬时 loss 尖峰（**Lead 复核原始 `history.jsonl` 后的真实值**） | `raw_n02`：最大 **3.00e7**（step 522），次大 5.26e5（step 523）；626 行中 **10 行 > 100、2 行 > 1e4**；尖峰前后邻步为 0.8 / −9 / 210.8 / 1.6，**数步内恢复**，**全部 finite（无 NaN/Inf）**，最终 dev 仍降到 0.0924。`rgb_n02` 同口径最大 **3.09e3**、**0 行 > 1e4**。<br>**注**：训练入口的交付回报曾把这写成「一次瞬时尖峰 42.39」，与原始记录不符；本报告以原始 `history.jsonl` 为准。 |
| λ_noise | **0**：官方缓存未声明噪声模型；代码遇到声明噪声会直接报错要求实现 censored 项，**不造真值** |
| save/resume | 从 `checkpoint_0001000.pt` 恢复 → 同一后缀动作逐元素 **max \|diff\| = 0.000e+00**；续训到 step 1002 |

首次真实训练步（`checks.json` → `gradient_wiring`）：
**loss 3.2151 = action 2.4725 + masked_raw 0.0053 + causal_raw 0.7373**；
梯度 L1：encoder **809,351**（382/384 张量有梯度）、readout **551,202**（72/72）、
proposals_corrector **3,381.6**（59/59）、decoder **29.6**、masked_branch **11.5**；
**`frozen_parameters_with_grad = []`** —— 冻结的 π0.5 后端一个梯度都没有。
`run_stats = {prefix_builds: 1, encoder_calls: 1, token_source_calls: 1}`
→ 每个决策时刻只跑一次链。掩码分支与动作分支**共享同一批 backbone 张量**（实测绑定 376、未绑定 0），
掩码语义用 P0 mask-aware 端口，**未改成零填充**。


### 6.2 真实 dev 动作误差（固定 dev 官方样本，随训练变化）

21 个真实 dev 点（`history.jsonl`，`validate-every 50`）：

```text
step   50    100    150    200    250    300    350    400    450    500
dev  0.948 0.3838 0.3489 0.5526 0.4072 0.3130 0.2888 0.3550 0.2882 0.3476

step   550    600    650    700    750    800    850    900    950   1000
dev  0.2739 0.2003 0.2664 0.2146 0.1216 0.1686 0.1547 0.1223 0.1073 0.0924
```

**如实说明**：这是**该训练入口自己定义的 dev 动作误差**，不是平台成功率，也不是闭环能力；
曲线**非单调**（200/250/400/500 处有回升），符合「不要求训练 loss 单调、不要求达到平台期」的约定。
**不得把它读成 RAW 增量或机制增量**——本轮没有四臂对照，也没有闭环。

### 6.3 先验真实进入推理（训练运行自己的真实数字）

同一样本、同一测量，两个合法归一化先验：

| 先验 | `log_p0` | `q` |
| --- | --- | --- |
| uniform（0.25×4） | `[-1.386, -1.386, -1.386, -1.386]` = log 0.25 | `[0.2579, 0.2561, 0.2435, 0.2425]` |
| biased（候选0 = 0.90） | `[-0.1054, -3.4012, -3.4012, -3.4012]` | `[0.9037, 0.0332, 0.0316, 0.0315]` |

P1 缺失的接线已接通：先验改变了 `log_p0`，并因此改变 `q`。

### 6.4 三态绝对检查（真实调用，含一条限制）

`checks.json → calibration`：在**独立 calibration 分区**上真实拟合，
`status = fitted`、`sequences = 4`、**threshold 0.2774**、`quantile = 0.95`、
`version = n02-v12-windowed-check-v1`、规则 hash `92c3be36…`，并在 calibration/dev 留出序列上
给出真实三态判决（calibration 4/4 `compatible`）。

**限制（如实）**：冻结的 `ViewingRule` 要求被检序列出现它声明的每个 `(scale, frame, region)`
视图，而 `inference_v12._residual_views` 用**真实源帧号**，一条冻结规则覆盖不了训练中变化的帧号。
因此**每一步训练内的三态检查显式保持「未校准 invalid」**，只有 calibration/dev 留出序列上的判决是
真实调用得到的。该说明同时写进了 `checks.json` 的 `frame_axis_note`。这不是通过，也不是隐藏。

### 6.5 接线与正确性检查（全部真实运行输出，见 `checks.json`）

`units_and_shapes` / `causality`（12 个窗口，历史行是 anchor 的**早于或等于**真实 frame 索引）/
`partition_isolation`（train-dev、train-cal、dev-cal 交集全 **0**）/ `episode_reset`
（被选中行清零、未选中行**逐位不变**、version 递增）/ `prior_wiring` / `gradient_wiring` /
`calibration` / `profile` / `final`。

### 6.6 受控同源 RGB 对照臂（两条都跑完 1000 updates）

`rgb_n02` 与 `raw_n02` 使用**同一份 loader RAW 值**、**同一网络结构**、**同预算（1000 updates）**、
**同一 seed 1103**、同一训练入口，唯一差别是输入表示：受控 RGB 由该次**同一 RAW** 经
**固定、已声明**的 ISP 得到 —— `clip(x/input_scale,0,1)` → 逐通道增益 `[1,1,1]` →
gamma 2.2 → 8-bit 量化 → `/255`，**不使用官方正常光配对 RGB**（`report.json` 原文：
`kind = raw3_fixed_isp`）。

训练端 fixed-dev 动作误差（**同一定义、可直接比**）：

| step | 50 | 100 | 200 | 400 | 600 | 800 | 900 | **1000** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `raw_n02`（RAW3） | 0.9480 | 0.3838 | 0.5526 | 0.3550 | 0.2003 | 0.1686 | 0.1223 | **0.0924** |
| `rgb_n02`（受控 RGB） | 0.7496 | 0.6256 | 0.2352 | 0.2068 | 0.2094 | 0.1563 | 0.0786 | **0.0785** |

（`raw_n02` 另做了 save/resume 续训验证，step 1002 处为 0.0805；**同预算比较一律取等步数 1000**，
所以上表用 0.0924。两条臂的实际训练耗时：RAW 4663.2 s / RGB 4800.3 s，均 1000 updates。）

**如实结论：在 1000 updates 这个可视化检查点预算上，RAW3 相对受控同源 RGB 没有优势**
（0.0924 vs 0.0785，受控 RGB 低约 15%），前期（step 50/100）RAW 反而更差。
这**不是**「RAW 无用」的结论，也不能反过来说是 RGB 更好：这里只跑了 ① 单种子 ② 1000 updates
③ 一个开发指标 ④ 没有闭环 ⑤ 没有超参搜索。按方案第 8.3 节，**本轮不产生输入收益或机制收益主张**；
`raw_base` / `rgb_base` 两臂未训练，四臂归因**未完成**。


### 6.7 本轮过程中修掉的真实缺陷（保留失败记录）

| 现象 | 处理 |
| --- | --- |
| CUDA OOM（GPU0 95.01 GiB 中 PyTorch 已分配 85.55 GiB；OOM 文本含另一进程 7.45 GiB） | 按「训练与验证按实际显存排队、不能同挤一张卡」重做画像；确定 microbatch 1 × accum 2，峰值 55.9 GiB |
| `ValueError: action_mask must be nonempty [B,H]=(1, 50); got shape=(50,)` | 真实接线缺陷，已修 |
| `N02V12InferenceError: measurement_ids declares 0 rows, bundle has 8` | 真实接线缺陷，已修 |
| `final.previews` = **failed**（`ValueError: operands could not be broadcast (240,320,3,3) vs (240,320,1)`） | **未修**：训练入口自带的 `--previews` 出图路径有广播 bug。改用已验收的评测/可视化入口在同一 run 目录 `preview/` 出图（见 §5）。 |
| `rawvla_features.load_audited_replaced_vision` 的 `loaded_keys` 返回 int 而非 list | 已记录；评测侧按两种形态兼容 |

### 6.8 关于负 loss（不是接线错误）

训练总 loss 在部分步为**负值**（如 step 923 为 −3.149），这是 RAW 目标尺度的正常后果，不是符号错误：
RAW 值量级约 1e-3，而 Gaussian NLL 为 `0.5·((y−μ)²/v + log v + log 2π)`，
其关于 `v` 的最优解是 **`v = (y−μ)²`**；当 `v > (y−μ)²` 时 loss **随 v 上升而变大**，
所以模型**无法靠放大方差逃避残差**——这正是方案第 5.1 节要求的性质。残差小则 NLL 可为负。
P3 应同时报告按每标量缩放的均值版本，便于横向比较。


## 7 LIBERO 真实对照（协议确认，不是主结果）

两相机 `libero_agentview/libero_wrist`；RAW 键 uint8 载入后 float32 ∈ [0,1]；本体 8 维；动作 `(10,7)`，
dim 7 / horizon 10 / offset 0；`needs_replan(0, 0/4/5) = [false,false,true]`（每 5 步重规划）。
真实 forward/backward（保留图、不 detach）：loss **0.4956688**，`tokens.requires_grad = true`，
梯度 L1 encoder **36370.63**（348 tensors）、readout **55737.64**（72）；反面控制（先 detach 再反传）
全部为 0。真实采样 `control_actions_shape [1,10,7]`，10 步 denoise，finite。

## 8 实际运行命令

```bash
# 远端
cd /workspace/owac/worktrees/n02-measurement-completion
export PYTHONPATH="$PWD/src:/workspace/owac/rawvla/source/starVLA"
export N02_PY="$PWD/outputs/n02-full/runtime/py310-cu128/bin/python"

# 训练（RAW，GPU0）
CUDA_VISIBLE_DEVICES=0 "$N02_PY" -m owac.training.n02_v12 \
  --config configs/encoders/n02_v12/raw_n02_robotwin.json \
  --seed 1103 --max-updates 1000 \
  --output outputs/n02-v12/P2/raw-n02-robotwin-s1103

# 评测
CUDA_VISIBLE_DEVICES=0 "$N02_PY" -m owac.evaluation.n02_full.v12 \
  --config configs/encoders/n02_v12/raw_n02_robotwin.json \
  --checkpoint outputs/n02-v12/P2/raw-n02-robotwin-s1103/checkpoints/<file>.pt \
  --output outputs/n02-v12/P2/raw-n02-robotwin-s1103/eval

# 监控
supervisorctl status owac-n02-v12-monitor
```

## 9 未运行 / 未做（不得读成通过）

- **100 次闭环**：属 P4，本轮**未运行**。页面里 `eval.planned = 100, completed = 0`。
- **四臂归因**：只跑了 `raw_n02` 与 `rgb_n02` 两臂；`raw_base` / `rgb_base` **未训练**，
  因此**不能**做输入收益 vs 机制收益的拆分。
- **P3 充分训练**：未启动。1000 updates 是**可视化检查点**，不是「充分训练」。
- **`λ_noise = 0`**：官方缓存未声明噪声模型，噪声校准项**未启用**（代码遇到声明噪声模型会直接报错，
  不造真值）。因此本轮的 `L_noise` 恒为 0，不是「噪声目标已验证」。
- **每步训练内三态检查保持「未校准 invalid」**（§6.4）：冻结 `ViewingRule` 覆盖不了变化的真实帧号；
  只有 calibration/dev 留出序列上的三态判决是真实调用得到的。校准阈值在两条链路上还差约 131×（§10.5）。
- **三态绝对检查的正式规模**：本轮真实计数来自有限样本（eval dev 4 锚点 / 32 因果行），不是正式验收规模；
  且 W3 的探针残差来自**声明的因果参考预测器**而非 v1.2 decoder，故其检出率**不是模型能力结论**。
- **无效 verdict 路径**：本轮 eval 为 4/4 valid、invalid = 0，**该路径没有被这次运行触发过**，
  不能用它证明「非零 invalid 会被正确单列」。
- **真实负结果（保留）**：同一预算下 **RAW 相对受控同源 RGB 没有优势**（§6.6，0.0924 vs 0.0785）；
  参考预测器下 `no_information` 与 `translation` 几乎不触发 `model_inadequate`，
  `exposure_shift gain=2.0` 触发 `compatible → model_inadequate`。未调参凑数。
- **原生 Bayer / RGB 信息损失结论**：本轮数据不提供，未宣称。
- **RAW 增量与机制增量、闭环收益**：**均未验证。**

本报告之外，训练入口的独立记录见
[P2_training_entry.md](P2_training_entry.md)、评测入口见
[results/p2_eval_entry_evidence/](results/p2_eval_entry_evidence/README.md)、
前端页见 [2026-10-16_n02_v1_2_progress_page.md](2026-10-16_n02_v1_2_progress_page.md)。

## 10 未解决问题

1. **`tau` / 能量尺度**（§3.1）：当前先验在 `q` 中占主导，P3 需显式处理。
2. **训练输入端修复 `policy.checkpoint` 语义**：config 里 `policy.checkpoint` 指 π0.5 后端权重，
   任何「递归找名为 checkpoint 的键」的代码会把它当 N02 权重丢给 `torch.load`。
3. **`load_audited_replaced_vision` 的 `loaded_keys` 返回 int**（冻结桥），调用方需自行兼容。
4. **远端 worktree 根的自造镜像包 `owac/`**：本会话早期 rsync 命令把 `src/` 内容铺到了
   worktree 根（带尾斜杠的 `src/`），从根目录 cwd 跑 `python -m owac...` 会静默加载该副本。
   已归档为 `.rsync-shadow-owac-removed-20261016/`（逐字节保留、可 `mv` 还原），
   并已实测文档里的启动命令现在解析到 `src/owac/__init__.py`。后续同步必须写 `src configs tests`（无尾斜杠）。
5. **校准阈值在两条链路上差约 131×**（**需要人判，P3 处理**）：同一条 calibration 分区上，
   训练端 `checks.json` 得到 threshold **0.2774**（4 序列、`version = n02-v12-windowed-check-v1`、
   rule hash `92c3be36…`），评测端自拟合 20 个校准锚得到 threshold **36.487774**（q=0.95，
   min 0.458 / median 36.188 / max 36.488）。20 个样本取 q=0.95 实际取到最大值，
   这正是评测侧 4/4 全 `compatible` 的原因，**区分度很低**。两条链路的统计量定义是否本应一致，
   本轮未深挖；在解决之前，**不得把 4/4 compatible 当成校准有效性证据**。
6. **训练入口自带的 `--previews` 出图路径有广播 bug**（§6.6），本轮改用已验收的评测/可视化入口出图。
7. **训练端与评测端的 dev 动作误差不是同一度量**：训练端 `dev_action_error = 0.0805`（loader dev
   前 4 个 window，flow time 由 `generator(seed+17)` 抽 Beta(1.5,1)，噪声 seed 1120）；
   评测端 `0.17435`（P0 frozen anchors 4 个固定锚，t 固定 0.5，噪声 seed 1103）。
   样本集、flow time、噪声 seed 都不同，**两者都真实、不可直接比较**；报告与页面必须区分口径。
9. **训练稳定性**：`raw_n02` 在 step 522 出现 **3.0e7** 量级的真实 loss 爆炸（次大 5.26e5），
   数步内自行恢复、无 NaN/Inf、未用随机模型回退，最终 dev 仍正常。梯度裁剪 1.0 全程生效，
   所以优化器步长被限住，但**这个不稳定来源没有定位**（最可能是 RAW 目标的 `log σ²` /
   `(y−μ)²/σ²` 在极小残差下的数值敏感），P3 需要在更长预算下观察并处理。
   训练入口交付回报里写的「最大尖峰 42.39」与原始 `history.jsonl` 不符，已按原始记录更正。
8. **cleanup**：`outputs/n02-v12/P2/ui-sample-*`（界面测试样例）与 `raw-n02-robotwin-uninit`
   （随机初始化权重的评测）保留在场，页面已把它们与真实作业分开显示；若不再需要可清理。


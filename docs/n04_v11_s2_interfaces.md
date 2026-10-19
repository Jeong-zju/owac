# N04 v1.1 S2 内部接口契约（v1）

2026-10-19 v5当前契约：`implementation_version=n04-v1.1-s2-stability-repair-v5`、`loss_definition=v1_1_s2_normalized_latent_bounded_transition_v5`。融合特征先做FP32通道RMS=1归一化，再共用到质量/状态/读出与停止梯度目标；原始RAW统计不改。RKN以identity初始化，包含恒等项的完整A用可微1/∞范数上界缩放至≤1，均值和协方差共用，banded与dense回退一致；不夹最终方差或损失，停止保护保留。对应加入真实source有效mask上的端点越界几何代价（按实际W/H归一化），越界匹配仍无效。训练整批汇总`objective_coverage`，任一正权重项无有效window/项就清梯度、报错，不更新optimizer/scheduler。v4.2故障权重不续入v5。固定dev采用原生确定性计算，启动前要求CUBLAS_WORKSPACE_CONFIG=:4096:8；退出时恢复训练编译/精度/随机状态。两平台完整真实零更新、连续8更新、独立GPU重载及第9次更新通过，正式续训并保存step10；见[修复报告](../research/experiments/2026-10-19_n04_stability_repair_v5.md)。下方v4.2初始化和未归一化量纲为历史。


2026-10-16 当前 v4.2 约定：`implementation_version=n04-v1.1-s2-aux-repair-v4.2`，`loss_definition=v1_1_s2_independent_window_vicreg_v4_1`。表征项改为每窗口第一个梯度时刻、按相机汇聚query后跨独立训练序列统计，内部方差25／协方差1、正式外权重0.1；批次无梯度统计与逐窗口VJP回放必须一致，不允许训练时静默省略cache。RKN采用明确配置的constant_velocity初值，测量补边replicate但新增区域support仍False，最终prefix投影FP32。未来目标停梯度、horizon和按实际有效项归约沿用v3。编译精确多次反传走整个窗口的原生路径；已知缓存错误仅允许一次恢复RNG/BN后的完整原生重算，保留错误，不更新部分梯度。旧checkpoint由旧冻结源码解释，禁止静默resume。见[完整修复与收据](../research/experiments/2026-10-16_n04_aux_repair_v4.md)。下方v3表征约定与原始并行派发所有权表为历史；本轮由Codex统一修复。

2026-10-16 修复版实现约定（旧训练已按用户指令终止）：`implementation_version=n04-v1.1-s2-repair-v3`，损失定义 `v1_1_s2_live_forecast_observation_vicreg_v3`。历史同形权重不静默恢复／服务／可视化。每帧只编码一次，桥返回 `encoder_output`；`StepOutput.target_features` 为当前融合特征的 detach 视图；`forecast_inputs.R_diag` 为活方差。源后验／R 不 detach，未来目标和定位停梯度；监督帧也计算辅助项；跨段 horizon 保留，源段最多延迟4帧反传；按实际有效项归约并记录 `valid_loss_counts`。VICReg 在固定 query 上跨观测统计。Perceiver K/V 包含规范化 media 和 latents，attention 列数为 media＋queries；稠密图只用前 dense 列，不归一化掉其实际 attention mass。RKN 仅重算纯 cell 预测，禁止整步有状态编码器重算；profile 不更新 optimizer。完整解释与验证见[修复报告](../research/experiments/2026-10-16_n04_v11_s2_code_repair.md)。

日期：2026-10-15。用途：S2 并行实现的唯一共享约定。只写跨文件使用的形状／单位／因果顺序，不写验收框架、不写多层 schema。任何需要变更的约定先回到派发者更新本文，再改代码。

2026-10-16 独立修补版补充：四臂入口统一接收平台原存储单位的同源 RAW。A/B 在编码器 `_as_frames` 边界使用 `fixed_clip_srgb_v1`（固定曝光、裁剪到 [0,1]、sRGB 传递函数，再还原存储单位），训练、服务和 forecast target 共用；不允许调用方先做 ISP 再重复转换。A/B 的质量／对应只读 RGB 与由 RGB 得到的饱和 mask；D/C 保持 RAW。对应 RawScale 白电平统一透传 RAW 配置，RGB 臂为 1.0。未请求热图的 Perceiver 使用 SDPA，独立可视化可显式请求实际 attention。新增日志 `applied_terms`／`applied_train_loss` 记录真正反传的项；旧 `terms` 保留历史含义。已有训练进程仍使用 10-15 冻结快照，不能认为这些修补已生效于旧进程。见[独立审计](../research/experiments/2026-10-16_n04_v11_s2_independent_audit.md)。

## 0 文件所有权（同一文件只由一名执行者修改）

| 文件 | 所有者 |
| --- | --- |
| `frameworks/n04/_vendor/raft/**`、`frameworks/n04/correspondence.py`、`tests/encoders/n04_v11/test_correspondence.py` | WS-A correspondence |
| `frameworks/n04/quality.py`、`state_estimator.py`、`evidence_update.py`、`tests/encoders/n04_v11/test_state_evidence.py` | WS-B state |
| `frameworks/n04/readout.py`、`frameworks/n04/encoder_v11.py`、`tests/encoders/n04_v11/test_readout_encoder.py` | WS-C integration |
| `training/n04_v11.py`、`evaluation/n04_v11.py`、`configs/n04/v1_1/**`、`tests/encoders/n04_v11/test_data_training.py` | WS-D trainer |
| `raw/rawvla_sequence.py`（仅追加新方法，不动既有语义）、`raw/n04_v11_data.py`（可新建） | WS-D trainer |
| `runtime/n04_v11_progress.py`、`runtime/n04_v11_progress.html` | WS-E monitor |
| `frameworks/n04/measurement.py`、`backends/rawvla_features.py`、`baselines/_vendor/rkn/**`、`frameworks/n04/_vendor/convnext/**` | 只读：禁止修改；需要新能力时在新文件里包装 |

## 1 因果与调用契约

- `N04V11Encoder.reset(batch_or_episode) -> None`；`step(current_raw, proprio_context, new_measurement=True) -> StepOutput`。
- 内部 `predict` 不接收当前图像；当前 RAW、当前特征、当前 flow 只进 update／质量／读出，不进当前 prior。
- 一次新观测只融合一次：burn-in 与 10 次去噪复用同一 prefix；重复 `step` 同一 frame 视为错误（用一个显式计数器让调用方看见）。
- 时间：无可靠时间戳时按记录步；不把帧差命名为秒。
- episode／窗口边界必须 reset；burn-in 用当前模型与真实历史生成状态，不跨 episode 继承。

## 2 坐标、flow、特征网格

- 2D 坐标统一用**原始像素**：`x∈[0,W-1]`，`y∈[0,H-1]`，整数值即像素中心；H/W 取实际帧尺寸（libero 与 robotwin 可能不同，测试里打印实际值，不写死）。
- 归一化互转只在 `correspondence.py` 定义：
  `px_to_grid(xy, size_hw) = (xy + 0.5) * 2 / size_hw - 1`；`grid_to_px(grid, size_hw) = (grid + 1) * size_hw / 2 - 0.5`。
  所有 `F.grid_sample` 一律 `align_corners=False`。其它模块只能引用这两个函数。
- flow：`flow_px [B,2,H,W]`，单位原始像素，**定义在上一帧坐标**（作者 RAFT 约定）：`dst_xy = src_xy + flow_at_points(flow_px, src_xy)`。支持推进即 `xy_t = xy_{t-1} + flow_at_points(flow_t, xy_{t-1})`。提供 `flow_at_points`、`sample_flow_at_grid`、`points_inside_frame`、`feature_scale`（RAFT 计算网格 vs 原生像素的等比尺度；原生=1.0，resize 才 >1 且必须等比）。
- 特征网格：`latent_mean_map [B,256,h,w]`，`h=H/stride`、`w=W/stride`（S1 `fusion_stride=4`）。支持点采样必须用 measurement 的 pad/size 元数据把像素坐标映射到特征网格（往返测试，误差 < 0.5 feature px）。
- 支持点位置（canonical）保存为**原始像素坐标**，flow 直接相加；特征与 R 在该位置重新采样。

## 3 测量与质量头

- 复用 S1 `N04FullMeasurement.backbone_levels/fuse/measure`、`preprocess_frames`、`MeasurementFrame`、`DenseMeasurement`，不改语义。质量头不读深度头输出。
- `QualityHead.forward(values, support, saturated, latent, xy_px) -> R_diag`
  - `values [B,C,H,W]` 预处理后 RAW（平台固定缩放，FP32 统计）；
  - `support` / `saturated [B,1,H,W]` bool（骨干输入平面，与 S1 同名）；
  - `latent [B,256,h,w]` 融合特征；`xy_px [B,N,2]` 原始像素；
  - 返回 `R_diag [B,N,256]` float32，`>= 1e-6`，与潜在测量语义一致；不读历史 residual、不读 S1 深度方差。

## 4 状态、支持与双模式

- 每相机 `N=128` 个持续支持，保存稳定 ID、上一 xy、年龄、有效性、双模式 RKN 状态。
- 观测 `y=[B,N,256]`（当前特征采样）；状态 `m=[B,N,512]`；观测矩阵 `H=[I_256 | 0]`（不训练 H 的稠密矩阵）。
- 两模式：0 延续（小 Q）、1 修订（额外非负 Q，作用于可影响观测的状态方向）。`mode_logits/prior/posterior [B,N,2]`；转移矩阵 `[2,2]` softmax 参数化、行归一化，初始化约 `[[0.95,0.05],[0.3,0.7]]`。模式先验只由旧权重与转移矩阵得到。
- RKN 复用 `baselines/_vendor/rkn/cell.py`（`_predict` / `_update` 或明确拆开的同等逻辑）。**当前 prior 由 t-1 posterior 经 `_predict` 得到**（本体条件按 Ac-RKN 加性控制结构注入），**再用 t 的 y/R 经 `_update`**；禁止把 update 后的下一时刻 prior 当当前 prior（作者 `forward` 是 update→predict 顺序）。
- 模式分数与混合（FP32）：`log_weight_j = log(mode_prior_j) - 0.5*(r_j^T S_j^{-1} r_j + logdet S_j + d*log(2π))`，`mode_posterior = softmax(log_weight)`；混合后各自 `_update`，保留两个分量与概率进入下一轮预测和读出，不先平均成一个状态。
- 支持生命周期：首帧覆盖网格出生；`xy_t = xy_{t-1} + flow(xy_{t-1})`；越界／遮挡只传播预测；新区域才出生；不按每帧置信度最大值重排重排。flow 是总图像运动，不输出相机／物体伪标签。
- 出生／ID 语义（2026-10-15 WS-C 核实）：**遮挡不触发出生**（空槽只由 `vacant = ~in_bounds_xy` 决定；遮挡只把 `valid` 置 False、位置仍在画幅内）；**ID 是累计出生计数器，不是槽位索引**（>128 属正常）；出生候选必须"经当前 flow 后下一帧仍在画幅内"才可递上去（否则出生后 0–1 帧即死、同一像点每帧重生）；`_trajectory`/`_prev_frames`/`SupportState` 形状恒定，不随帧无界增长。
- 覆盖判据（`radius_px` = 半个最小网格间距）**保持现状**：支撑漂移出该半径后其原始像点重新判为未覆盖并补种，属"维持当前可见区域覆盖"的可接受语义。必须作为可观测指标跟踪：每窗口 `births_per_frame`、`ids_max`、`median_age`。复查判据：真实 dev 数据上若 `median_age < 5` 帧，再改为"按当前（预测后）位置判覆盖"；当前取"保持现状"，改语义需要真实数据证据。
- forecast 目标：`-logsumexp_j(log(mode_prior_j) + log_prob(target | prior_j))`，二次项与 log det 都保留，禁止截断 residual²/variance 后继续压 log variance。

## 5 读出与后端

- `Readout.forward(...) -> per_camera [B,256,2048]`：OpenFlamingo 完整 Perceiver，6 blocks / 256 queries；输入含当前稠密特征直通路径、两个记忆分量及其权重、2D 位置、年龄；不能只读 128 点丢掉全图。
- `StepOutput` 至少含：`prefix_tokens`（按相机顺序拼接，形状与 `RawVLAFeaturePolicy.embed_prefix_from_visual_tokens` 期望一致，WS-C 必须读该函数确认并在测试中断言）、`losses`（至少 `correspondence` 标量，由编码器用上一帧/当前帧 RAW 与当前 flow 调 `RaftCorrespondenceLoss`）、`forecast_inputs`（t 时刻各支持的两模式均值/结构化协方差、`mode_weights [B,N,2]`、`xy [B,N,2]`、`ids`，供训练器构造 forecast target）、`debug`（flow、质量 R、模式权重、支持轨迹、readout attention，供可视化）。
- 后端：`RawVLAFeaturePolicy(encoder, backend)`；语言缩放、mask、位置编号、动作归一化与 flow-matching 目标沿用既有桥。
- 梯度：冻结参数用 `requires_grad=False`，不得用 `torch.no_grad()` 包住训练后端；动作 loss 必须经 prefix 回传到测量／状态／读出。

## 6 损失与训练配置

- 外层权重：action 1.0、correspondence 0.1、forecast 0.1、representation 0.01；各辅助项先按有效样本／维度取均值。
- forecast 预测未来 1、2、4 个记录步；由训练器用 `StepOutput.forecast_inputs` 与未来帧 target 调 `evidence_update.forecast_loss`。target 构造：用真实 flow 链把支持 xy 从 t 输运到 t+k（仅 `no_grad`，只决定"看哪里"），在 t+k 的**同一 RAW 编码器 stop-gradient／EMA** 特征上采样（决定"预测什么内容"）。发出预测时不得读未来 RAW／本体／动作；窗口不足或 target 无效时跳过该 horizon 并计数，不填 0。
- optimizer AdamW；新增模块 weight decay 0.05；LR：ConvNeXt 1e-4、RAFT／状态／读出 3e-4、π0.5 适配 1e-5；warmup ~2k 后 cosine。
- 窗口 16 burn-in + 32 梯度帧，每窗口均匀 4 个动作监督点；有效 batch 8 段序列（microbatch 可为 1，累积实现）；网络 BF16，RAW 统计／协方差／NLL FP32。
- 每平台 100k optimizer updates 为名义预算，不是效果门槛。
- 四臂开关（S2 优先训练 D）：`arm ∈ {D, C, B, A}`，`input_mode ∈ {rawvla_raw3, rawvla_rgb3}`。
  - D：RAW + 完整 N04；C：RAW 单帧（同骨干／同读出，无状态／无对应）；
  - A：同源固定 ISP RGB 单帧；B：同源 RGB 的完整 N04（质量头不得读 RAW）。
- CLI 只提供常用参数：`--config --data-root --output --seed --device --resume --eval-list`。

## 7 监控数据约定（WS-D 写、WS-E 读）

`<run>/progress.json`（临时文件后 `os.replace`）+ `metrics.jsonl` + `eval_results.jsonl` + `artifacts/`：

```text
stage, platform, arm, seed, phase, updated_at,
step, budget_steps, train_loss, dev_loss, step_seconds, gpu_memory_gb,
eval{completed, planned, successes, errors, libero, robotwin},
artifacts[], log_tail[]
```

未产生的数值用 `null`，不用 0 冒充。需要扩展直接加字段，不写多版本 schema 迁移。

## 8 远端运行环境（WS-ENV）

远端实例（vast.ai，2026-10-15 实测）：2×RTX PRO 6000 Blackwell 96G；官方数据 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/{libero,robotwin2}`；N04 根 `/workspace/owac/research/n04-v1`（`runs/{s0,s1}` 为历史）；encoder env `/workspace/owac/research/n04-v1/envs/encoder/bin/python`（py3.12 / torch 2.14.1+cu130，CUDA 可用）。**远端缺**：starVLA 源码、π0.5 权重、`starvla` env、仿真器。同步目录固定为 `/workspace/owac/research/n04-v1/source/<snapshot-id>`，运行目录 `/workspace/owac/research/n04-v1/runs/v1.1/<stage>/<platform>-<arm>-seed101/`。

# N01 v1.0 共同开发契约

> **历史材料（2026-10-15 更新）。** 当前正式路线见 [v1.2 完整方案](../../N01_full_development_plan_2026-10-15_v1.2.md)。本文件保留旧原型/v1.0 契约以解释已有实现，不再作为后续阶段的独立启动依据；P0—P2 既有证据保留，旧 P3—P8 已被取代。

2026-10-14。与当阶段提示词一起交给 DeepSeek。研究依据为[完整方案](../../N01_full_development_plan_2026-10-14_v1.0.md)，本文件将方案细化为首版实现选择；这些选择尚待实现和实测。

## 1. 角色与工作方式

你是本阶段的实现者。按已经给定的数学定义、文件职责和实验顺序实际修改代码、执行验证、保存收据。路线判断、是否宣称增量、是否采用下一种架构由研究负责人复核；不要把任务改写成文献综述或自行换一条更容易的算法。

工作树为 `/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`。先读 `AGENTS.md`、`README.md`、`docs/status.md`、[实现审计](01_CURRENT_IMPLEMENTATION.md)与[远程执行说明](03_REMOTE_EXECUTION.md)。沿用当前历史，新实现统一进 `n01_full`；旧 `n01`、DBSR 冻结干预和旧结果作为回归参照保留。只处理本阶段，完成后交接，不自动执行后续阶段。

每阶段按以下顺序推进：复核前阶段实际接口与收据 → 写最小数值参考 → 实现正式模块 → 运行针对性行为测试 → 远程执行本阶段规定的训练/测量 → 整理结果与阶段报告。命令中标为“拟新增”的入口需先实现、检查 `--help`，再运行。不要把文档中的示例当成现成 CLI。

## 2. 首版路线已经确定

1. 内容骨干默认使用仓库全尺寸 DBSR：64 通道、9 个残差块、512 维特征。时间来源控制先于空间组合；P4 主通路为 `source_features`。`domain_content` 作为额外受控实验。
2. P1 先做完整来源重算与独立稠密参考；P5 才尝试局部执行。先使语义正确，再测速度。
3. 运动首选固定版本 SEA-RAFT，分别适配 CFA 的 4/8 通道入口和 pseudo-RAW 的 3/6 通道入口；校准数值单位，保留作者相关体和迭代更新。
4. 时间规则默认用 BOCPD；DSM-bocd 是独立鲁棒对照。统计保证使用单独的合法顺序检验支路。空间规则采用局部仿射预测相容检验，连通图只提候选。
5. 神经持久状态到 P5 再接官方 Mamba；P6 再接第二骨干 BSRT 合规特征支路。若分支没有增量，保留结果和已有完整支持版本，不强制加入最终模型。
6. 首版读出沿用 `VisualTokenPostprocessor`。最终每有效相机 256 个、宽度 2048 的 token；DBSR 512 维输出到桥的投影显式计数。P7 统一比较冻结后端与相同范围解冻，P8 再做全量闭环。

DBSR 首版 `readout` 返回256个512维 `VisualTokens`，`config` 同时明确 `feature_width=512, readout_tokens=256, readout_width=2048`。构造桥时显式传 `VisualPrefixConfig(in_dim=512,width=2048,...)`，或在 `from_readout_config` 传 `in_dim=512`；不使用其默认in_dim=readout_width造成维度误接。BSRT用实际feature_width替换512，最终宽度/数量仍相同。

## 3. 跨阶段接口

以下为拟新增类型，放在 `src/owac/frameworks/n01_full/contracts.py`；复用已有 `core/encoder_contract.py` 的外部类型，不要求整个仓库迁移内部状态。

| 对象 | 首版字段和语义 |
| --- | --- |
| `SourceKey` | `dataset_id, episode_id, camera_id, frame_id, channel_or_phase, sensor_y, sensor_x`。标识原始采样，不能用像素值、batch slot 或插值结果作 ID |
| `MeasurementRecord` | `key, raw_value, value_units, timestamp, time_unit, validity_kind, calibration_version, noise_block_id`；原始值不可变，校准值为有版本派生对象 |
| `CorrespondenceMap` | `source_frame_id, reference_frame_id, source_xy, tap_source_ids, tap_weights, valid, visibility, flow_risk, mapping_version`；`source_xy` 是从当前参考位置去历史源取样的 pull 坐标 |
| `SupportDomain` | `domain_id, reference_chart, basis, phase, evidence_source_ids, context_source_ids, temporal_segment_id, decision_version, mapping_version`；证据与上下文角色分开 |
| `EvidenceEstimate` | `theta, covariance, uncertainty_kind, identifiable_mask, evidence_source_ids, prior_id, support_version` |
| `DependencyNode` | `node_id, kind, parent_ids, parameter_version, support_version, mapping_version, calibration_version` |
| `ReadoutSnapshot` | `representation, validity, source_time, commit_time, versions, exact_replay_range, cost`；所有字段对应一次已提交结果 |

`validity_kind` 至少区分 valid / missing / bad / saturated；来源被拒绝是支持关系的状态，不是改写原始传感器状态。一个 RAW 样本可以被多个候选域引用，在一个估计中只计一次。先验单列，不加进独立观测数。

外部适配器为拟新增 `N01FullEncoder`：

```python
init_state(batch_size, camera_spec, device) -> FullState
step(frame: MeasurementFrame, embodiment=None, state=None,
     budget: ComputeBudget | None = None) -> EncoderOutput
readout(output: EncoderOutput) -> VisualTokens
reset(state, batch_mask) -> FullState
detach_state(state) -> FullState
```

`step` 内只处理一次唯一到达事件；`readout`、重复传输、flow matching 的十步去噪不再 observe。camera 状态独立，batch 内可局部 reset；初次到达新 episode 时先 reset 对应成员。

数据窗口采用 `[B,T,C,H,W]`，时间从旧到新；DBSR 融合入口当前是 **newest first**，转换只在一个命名明确的适配函数完成。CFA 的 C=4，保留相位和真实采样坐标；`rawvla_rgb3` 的 C=3，禁止伪造第四测量或 CFA 相位。不同相机尚无已验收几何映射时只在后端拼接 token，不在 RAW 统计中跨相机合并。

解析求解输出结构系数；神经输出非图像特征。主链不产生 RGB/sRGB/XYZ，不使用 RGB 教师、RGB 重建或作者 loader 的 PIL 输出。允许 RAW 域校准、重排、掩码和具有明确来源的 RAW 结构节点。

## 4. 撤回和重放的精确定义

先实现两个不同测试接口，不让“撤回成功”只有一个含糊指标：

- `fixed_decision_replay`：固定 motion/support 决策，替换被撤回源的内容后，重新编码、计算融合权重与读出；结果应不变。依赖其内容的 motion 要标为冻结决策依赖。
- `counterfactual_replay`：从输入事件流删掉来源，重新计算运动、支持判断、归一化和全部状态，与独立批量参考比较。只有这一模式可以声明整个结果已不依赖该源。

非线性特征从保留 RAW 重算。P4 空间域首版在源 RAW 上按域支持置无效、重新编码整个受影响源，并重算融合权重；这样处理卷积感受野跨域问题。邻域上下文若保留，需要在 `context_source_ids` 声明；内容隔离测试覆盖所有声称被清除的角色。

P1/P4 首版精确窗口为 32 个唯一帧，算法比较可用 8/16/32；更旧信息只作为带来源摘要。撤回超出精确范围时，作废污染摘要并从合法边界重建，结果记录历史损失。禁止承诺无限历史精确撤回。模型参数更新后，旧参数产生的派生缓存失效。

## 5. 数据、训练和统计

数据拆分分四区：train / calibration / validation / test；按源轨迹、场景、会话成组，同轨迹照明变体保持同组。先用 metadata，再用已审计的 `source_group`；真实相机留出另做显式表，不能用随机帧划分代替跨相机验证。默认按组 70/10/10/10 作首版候选；P0 检查各条件非空、数据规模和来源覆盖后冻结，已有官方分区优先遵守。

旧 offset=1000/2000 已消费。新生成器使用哈希命名空间 `n01-full-v1/<split>/<case>/<sequence>`，训练种子默认 101/211/307，数据种子与模型种子分开。官方闭环 13,250 条身份独立于离线四分区；P7 开发闭环使用独立初态，已有基线 test 结果只作已披露的历史参照。

P0 profiling 后锁定完整分辨率、有效 batch、最大步数和验证阈值。默认规划：机制 20k 起、最多 100k；运动 100k–300k；动作每平台每种子 30k–100k。数字是资源候选，不是充分训练证据。各臂有相同的两档学习率开发搜索、相同初始化公共层、数据流和监督信息；同时报告同更新量比较与各自到平台比较。

正式训练每 1,000 updates 验证，至少两次有效 LR 下降；最终低 LR 连续 8 次改善低于锁定 `delta_val`，best 不持续落在预算末端，且小集拟合/输入/梯度诊断通过，才记 `validation_plateau`。预算耗尽另记 `budget_exhausted`，保留 optimizer/scheduler/RNG/sampler/scaler 续训，不能重启优化器后称原运行继续。

checkpoint 保存模型、桥、后端可训练参数、优化器、scheduler、scaler、Python/NumPy/CPU/CUDA RNG、sampler permutation/cursor、数据/校准/支持/源码 hash、已消耗样本数。恢复测试必须检查下一批身份、下一次 loss/更新和连续运行对应，不能只比较 reload 后的 forward。

主机制指标：静态误差、事件后 1/2/4/8/16 帧误差、前 8 帧 AUC、恢复时间、删失/未恢复比例、误撤回率、错误历史残留。首版规划门槛为事件 AUC 相对 B2/B3 改善 5%、静态与当前异常退化不超过 2%；P0/P3 开发方差决定可检测性并冻结。以 episode/场景为重采样单元，配对 bootstrap 纳入训练 seed；比较顺序和 Holm 校正写入协议。

## 6. 阶段收据和完成条件

每阶段新增 `research/experiments/n01_full/Px_report.md` 和小型 `research/experiments/results/n01_full/Px/receipt.json`；大日志、权重、张量放 ignored outputs 或服务器。报告固定包含：

1. 实际修改文件、入口、前后数据流，以及与原要求的对应关系。
2. 真实执行命令、源码 SHA + dirty diff + 文件 hash、环境和资源、数据/配置/初始权重 hash。
3. 数值/工程结果、机制效果、真实数据与闭环结果分别陈述；表中保留失败、未执行和缺失数据。
4. 接口交接：可加载 checkpoint、模型配置、支持/校准版本、下一阶段用的命令与已知差异。
5. `implementation_status`、`training_status`、`effect_status` 三个字段；代码完成、训练平台、研究有效不是同一个 PASS。

本地执行 `make check` 与相关测试；共享桥/loader 有修改时执行全部相关回归。新测试必须有独立 oracle 或会使旧错误版本失败的实际性质。三种子和服务器曲线是实验要求，不用大量重复空测试替代。

遇到接口实现差异，按本契约在新命名空间适配并记清。遇到研究假设不成立，完成既定对照、保存负结果与诊断后交接，不靠改测试/阈值制造通过。阶段报告完成后等待研究复核，再由下一条提示词推进。

# N01 v1.0 支持契约：T1–T6 理论义务文件

| 项 | 值 |
| --- | --- |
| 版本 | v1.0-draft |
| 日期 | 2026-10-14 |
| 工作流 | P0 workflow D（共享任务板 `task-4`，执行者 `wf-theory`） |
| 实现基线 | 分支 `lzh/n01-evidence-integration`，HEAD `429591e`（父研究结果 `85790cc`） |
| 依据 | 方案 `docs/N01_full_development_plan_2026-10-14_v1.0.md`（§4、§5、§7–§12）；`docs/prompts/n01_v1_0/00_SHARED_CONTRACT.md`（§3–§4）；P0/P1/P3/P5 阶段提示词；`docs/prompts/n01_v1_0/01_CURRENT_IMPLEMENTATION.md` |
| 状态 | P0 交付草案 + **P1 更新（2026-10-14）**：T1/T2/T5 的代码入口、反例测试与 `analytic` 评估入口已落地并实际执行（见各节「P1 实现状态」）。**本文不含任何已完成证明；T1–T6 的每一项命题状态仍为「未证明」**，P1 只新增实现与数值/行为反例证据。 |

**用途。** 把方案 §5 的六项理论义务写成可判定的定义、逐条假设、可检验命题（标明未证明）、反例/失效条件与代码测试映射，作为 P0–P8 各阶段验收与交接的共同依据。理论文件在 P0/P1 建立、随算法变更一起评审（方案 §5）。

**不承诺。** 本文不承诺：收敛速度或充分训练、有限帧内必然检测变化、GPU 硬实时、任意长历史的精确撤回、稳定性等价于证据正确、固定设计统计结论在自适应选择数据流下继续成立、任务效果或闭环收益。方案 §5.1–§5.6 的公式是待实现与待检验的命题，不是已完成结果。

**阅读约定（每条陈述必须可归入其一）。**

- `[事实]` 本仓库源码、实验收据或方案文档中已存在、可核对的内容；引用时给出位置。
- `[假设]` 义务成立所需、当前未验证的条件。
- `[待证明]` 写成可检验陈述的命题；状态一律「未证明」，通过条件与失败条件分开列出。
- `[反例]` 对某个具体声明的否证情形，或对当前实现的已知失效。
- `[文献]` 外部既有结论，仅按所列条件引用，不代表本项目已实现或已验证。
- `[映射]` 代码/测试落点；尚不存在的路径标「P? 待实现」，给出所属阶段与建议符号名。

阶段标记沿用方案 §7（P0–P8）。「P? 待实现」表示落点尚未创建，最终符号名以对应阶段报告为准。

---

## 1. 契约字段的不变量（六项硬约束的可判定形式）

T1–T6 建立在这些不变量之上。每项给出一条可判定的检查规则。

### 1.1 证据角色与上下文角色分离

**字段。** `SupportDomain.evidence_source_ids` 与 `SupportDomain.context_source_ids`（共同契约 §3）。

**规则 E1。**

1. 每个域版本中两个集合都只含 `SourceKey`，且必须都能在 `MeasurementLedger` 中查到对应的 `MeasurementRecord`。
2. `evidence_source_ids` 是该域结构估计的测量证据：进入设计矩阵 `Φ`、噪声协方差 `Σ` 与精度累计 `Λ/η`；只有它计入独立观测数。
3. `context_source_ids` 是邻域/上下文角色：**不得**进入该域的 `Λ/η`，不得作为该域"结构被共同支持"的精度项。它可以参与内容/特征通路，但必须声明用途。
4. 同一 `SourceKey` 在同一域版本内不得同时出现在两个集合中（角色互斥）；若要改变角色，产生新版本。
5. 内容隔离测试必须覆盖所有被声明清除的角色：撤回来源 `s` 时，若 `s ∈ evidence_source_ids` 而 `s ∈ context_source_ids` 未被处理，则"结果不依赖 `s`"的声明不成立（共同契约 §4 最后一段）。

**判定。** 对给定域版本 `D`，检查：两集合不相交、均为 ledger 中有效记录、`fit_structure` 的 `Λ/η` 只由 evidence 集合构造、隔离测试的 manifest 列出所有角色。

**失效。** 上下文来源被隐式当作证据（精度虚增）；来源出现在两个集合（重复计数或隐藏依赖）；保留的上下文未被隔离测试覆盖（`counterfactual_replay` 的声明变成假）。

**映射。**

| 机制 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 角色字段与互斥检查 | `src/owac/frameworks/n01_full/contracts.py::SupportDomain`（`evidence_source_ids` / `context_source_ids`） | P0 已实现（契约字段/行为，不表示理论已证明） |
| 只由 evidence 构造统计 | `src/owac/frameworks/n01_full/statistics.py::fit_structure` | P1 已实现（只读 `evidence_source_ids`；`diagnostics.context_used_as_evidence=False`；`tests/encoders/n01_full/test_statistics.py::test_missing_bad_saturated_never_enter_the_design_and_context_never_does_either`） |
| 角色覆盖隔离测试 | `tests/encoders/n01_full/test_replay.py::test_isolation_covers_declared_roles` | P1 已实现（撤回 evidence 与 context 两个角色后替换其内容，读出不变） |
| 旧实现中掩码/有效性分离的回归参照 | `tests/encoders/n01/test_acceptance_regressions.py::test_cfa_stem_consumes_actual_sampling_offsets_and_quality_masks` | 已存在（旧语义，不构成新契约验收） |

### 1.2 `validity_kind` 四态，与「拒绝是支持关系状态」

**字段。** `MeasurementRecord.validity_kind`（共同契约 §3）。四态语义：

| 取值 | 可判定语义 | 是否提供测量信息 |
| --- | --- | --- |
| `valid` | 采样存在且未越界/未损坏，可用作观测证据 | 是（在对应噪声模型成立时） |
| `missing` | 没有测量（未到达或原始采样缺失；metadata 缺失不等于测量缺失） | 否；**不是零值，也不是反证** |
| `bad` | 采样存在但已知不可用（坏点、已知污染、掩码标记） | 否；必须记录检测来源与版本 |
| `saturated` | 采样存在但被位深/满阱截断，数值只能解释为区间 | 不按普通高斯观测使用；首版排除或显式截断模型 |

**规则 E2。**

1. 每条 `MeasurementRecord` 恰好取四态之一（互斥且穷尽）；`raw_value` 不可变，校准值是有版本的派生对象（方案 §4.1）。
2. `missing`/`bad`/`saturated` 不得被当作 `0` 或普通数值进入 `Φ`、`Λ`、`η`，也不计入独立观测数。
3. **来源被拒绝（从域中撤回）改变的是支持关系**——`SupportDomain` 成员与 `support_version`、`EvidenceEstimate` 的证据集合——**不是 `MeasurementRecord.validity_kind`**。不得把"已撤回"改写成"该采样变坏/饱和"；原始传感器状态与拒绝决策必须分开可审计。
4. 缺失不等于反证；饱和/坏点不等于"支持失效"的证据。三者只能降低可用信息，不能自动触发撤回（方案 §4.3 `retain/forget` 行、§5.3）。
5. 一帧同时携带 valid/bad/saturated 位图时，优先级与覆盖关系必须显式声明并测试（现有外部契约 `MeasurementFrame` 同时提供 `valid_mask / bad_mask / saturated_mask`）。

**判定。** 给定原始事件流与域版本序列，可逐记录核对 `validity_kind`；抽查被撤回来源，确认其记录不变而域成员变化。

**失效。** `missing` 当 0（把无数据当坏数据/证据）；`saturated` 当普通高斯观测（截断偏差进入均值与方差）；撤回导致 `validity_kind` 被改写（历史不可审计，"当前异常"与"结构变化"被混淆）；掩码来源/版本缺失导致 `bad` 不可追溯。

**映射。**

| 机制 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 四态字段与不变式检查 | `contracts.py::MeasurementRecord.validity_kind` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 记录不可变 + 状态分离审计 | `ledger.py::MeasurementLedger.get_sources` / `observe` / `with_content` | P1 已实现（`test_ledger.py::test_content_replacement_is_a_view_and_never_rewrites_the_raw_record`） |
| 缺测/饱和不参与估计 | `statistics.py::fit_structure`、`tests/encoders/n01_full/test_ledger.py::test_validity_precedence_and_values`、`test_statistics.py::test_missing_bad_saturated_never_enter_the_design_and_context_never_does_either` | P1 已实现 |
| P0 契约字段行为（四态与身份守恒） | `tests/encoders/n01_full/test_foundation.py` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 旧实现把 bad/saturated 折进 `valid` 的现行行为 | `src/owac/frameworks/n01/state.py::push`（`valid = valid & ~bad_mask & ~saturated_mask`，代码注释自述 "bad/saturated removed"）与 `src/owac/frameworks/n01_dbsr_raw.py::RawDomainHistory.step`（其后只消费 `state.valid[0]`） | 已存在；作为"旧语义"反例参照 |

### 1.3 `identifiable_mask` 与先验单列

**字段。** `EvidenceEstimate.theta, covariance, uncertainty_kind, identifiable_mask, evidence_source_ids, prior_id, support_version`（共同契约 §3）。

**规则 E3。**

1. `identifiable_mask` 形状与参数坐标一致，取值 `True` 当且仅当该坐标在固定设计下可辨识（秩判定使用声明的容差；建议 SVD/QR，记录条件数与阈值）。
2. 不可辨识方向必须返回无效 mask 项；**伪逆给出的零值不得被当作零方差的确定方向**（P1_ANALYTIC_REFERENCE §2）。
3. 先验单列：`prior_id` 标识先验来源与版本；无先验时为空。先验项 `Λ_prior/η_prior` 只在估计中出现一次，且**不计入独立观测数**。
4. 独立观测数 = 去重后参与该估计的唯一 `SourceKey` 数（跨候选域引用同一 RAW 样本时，在单个估计中只计一次，共同契约 §3）。派生节点（校准值、插值、积分节点）不增加独立观测数。
5. `covariance` 只在可辨识子空间内解释；`uncertainty_kind` 必须区分：已知 Σ 的精确协方差 / 未知相关性的保守界 / 神经代理方差（方案 §5.2 限制段）。

**判定。** 对照独立 FP64 稠密参考（`reference.py`，不得调用生产实现）比较 `theta / covariance / 秩 / identifiable_mask`；核对观测计数是否等于去重来源数、先验是否单列。

**失效。** 先验被算成观测信息（方差低估）；秩亏方向获得虚假 confidence；同一来源在多个候选中被重复计入一次估计；插值/派生节点被当作新观测；`uncertainty_kind` 标签混用（把保守界当精确后验）。

**映射。**

| 机制 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 字段与标签 | `contracts.py::EvidenceEstimate` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 秩/可辨识子空间 + 先验处理 | `statistics.py::fit_structure` | P1 已实现（SVD 行空间 leverage；秩亏 → `uncertainty_kind="none"` 且无协方差；`test_statistics.py::test_rank_deficiency_reports_the_identifiable_subspace_without_fake_precision`） |
| 独立 dense oracle | `src/owac/frameworks/n01_full/reference.py::dense_gls`（eigh 白化 + 显式稠密逆） | P1 已实现（不调用 `statistics.py`） |
| 数值/秩亏测试 | `tests/encoders/n01_full/test_statistics.py` | P1 已实现（手算、随机异方差、FP32、病态条件数单报、先验、保守界） |
| 旧实现的唯一帧计数回归 | `tests/encoders/n01/test_acceptance_regressions.py::test_repeated_id_and_padding_preserve_all_eight_unique_frames`、`test_fractional_overlap_uses_each_original_source_once` | 已存在（历史语义，非新契约验收） |

### 1.4 `DependencyNode` 四版本与「参数更新后旧派生缓存失效」

**字段。** `DependencyNode.node_id, kind, parent_ids, parameter_version, support_version, mapping_version, calibration_version`（共同契约 §3）。

**规则 E4。**

1. 每个派生对象（来源特征、积分节点、融合权重预测器缓存、递归状态、读出）必须能定位全部父项；`parent_ids` 为空且 `kind` 不是原始测量时，视为依赖图不完整（T5 的失效条件）。
2. 四个版本各自有稳定 ID（建议内容 hash/版本串，能区分"改了什么"）：`parameter_version` 覆盖模型/优化参数，`support_version` 覆盖域成员与支持决策，`mapping_version` 覆盖对应/坐标映射，`calibration_version` 覆盖噪声、单位、黑白电平与增益。
3. 派生节点仅在**其全部父项与自身的四个版本都与当前版本一致**时有效。
4. **参数更新后，用旧 `parameter_version` 产生的所有派生缓存失效**（方案 §8.2、§9.1 `N-stale`）。旧缓存可以保留作诊断，但不得作为当前结果的一部分；缓存命中率不是等价性证据。
5. `N-stale` 组是刻意的违反者，只用于测量"撤回重建本身的增量与污染代价"，不得报告为满足本规则。

**判定。** 注入版本变更（参数、映射、校准、支持各一），检查受影响闭包是否全部失效、未受影响节点是否保持；未失效的旧派生节点出现即失败。

**失效。** 只更新一个版本字段而实际另一个也变了；跨优化步骤复用旧参数缓存；依赖图缺父项却仍宣称局部等价；把 N-stale 结果当作撤回语义的验收。

**映射。**

| 机制 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 版本字段与图结构 | `contracts.py::DependencyNode` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 失效传播 | `src/owac/frameworks/n01_full/invalidation.py` | P5 待实现 |
| 版本变更测试 | `tests/encoders/n01_full/test_state_replay.py` | P5 待实现 |
| 参数版本进入 checkpoint/续训 | `src/owac/training/n01_full.py`、`src/owac/runtime/n01_burst_budget.py`（现有续训机制） | P4 待实现 / 现有机制复用 |
| 现有缓存复用风险点 | `src/owac/baselines/burst_features.py::BurstFeatureEncoder.encode_frames`（源特征重算入口） | 已存在 |

### 1.5 `exact_replay_range` 与精确回放 32 个唯一帧的边界

**字段。** `ReadoutSnapshot.exact_replay_range`（共同契约 §3；方案 §4.6、§5.5、共同契约 §4）。

**规则 E5。**

1. **唯一帧计数。** 对每个 `(episode_id, camera_id)`：唯一帧 = 具有唯一到达事件的 `frame_id` 数。重复传输、`readout`、flow matching 的十步去噪、同一帧在不同窗口再次出现，都不新增唯一帧；数值相同的两次真实曝光仍是两个测量（方案 §4.1）。
2. **精确回放范围。** `exact_replay_range` 是提交该快照时声明的区间 `[t_lo, t_hi]`（按帧身份或来源时间表示），其内至多包含 **32 个唯一帧**；不足 32 帧时范围截断到已有唯一帧。算法比较可用 8/16/32，但"精确"的上限是 32，必须写入模型规格与协议（`configs/encoders/n01_full/protocol_v1.json`）。
3. **范围内判定。** 对范围内任意来源的撤回请求，`counterfactual_replay` 结果必须与独立批量参考在声明容差内一致。
4. **范围外判定。** 超出精确范围时，必须**作废被污染摘要并从合法边界重建**，结果显式记录历史损失（作废了哪些摘要、从哪个边界重建）。禁止承诺无限历史精确撤回。
5. **范围按相机独立。** 不同相机不回借范围；跨 episode 不得借用；`ReadoutSnapshot` 的所有字段必须对应同一支持版本下的一次已提交结果，读出不推进测量时钟（方案 §4.6）。

**失效。** 承诺任意长历史精确撤回；重复帧占用窗口槽位（窗口被重复包提前耗尽）或重复包不登记导致真实新曝光被丢弃；摘要内含无法单独移除的旧来源却仍声明精确；`source_time` 与 `commit_time` 混用（把新版本贴到旧来源时刻）；读出返回未提交/半新半旧状态。

**映射。**

| 机制 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 快照字段与提交语义 | `contracts.py::ReadoutSnapshot` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 精确窗口/摘要边界 | `replay.py::full_replay`（P1 已实现 32 帧精确窗口、`ReplayRange`、历史损失与 `ReadoutSnapshot`）、`state_field.py`（P5 待实现） | P1/P5 待实现 |
| 唯一帧与窗口语义的现有实现 | `src/owac/frameworks/n01/state.py::push`（"repeated frames do not create extra independent observations"）、`reset`、`detach` | 已存在（旧语义） |
| 旧 8 帧唯一性回归 | `tests/encoders/n01/test_acceptance_regressions.py::test_repeated_id_and_padding_preserve_all_eight_unique_frames` | 已存在 |
| 32 帧边界测试 | `tests/encoders/n01_full/test_replay.py::test_exact_window_boundary_records_history_loss_and_rebuild_claim`（8 帧声明窗口 + 10 次曝光） | P1 已实现 |

### 1.6 两个测试接口：`fixed_decision_replay` 与 `counterfactual_replay`

**接口。** 共同契约 §4（方案 §4.5 给出同样的两层验收）。

| 接口 | 固定什么 | 重算什么 | 可以声明的结论 | 不得声明的结论 |
| --- | --- | --- | --- | --- |
| `fixed_decision_replay` | motion/support 决策（**冻结决策**） | 被撤回源的内容替换后重新编码、融合权重与读出 | 内容通路已清除：给定同一组决策，读出不再读取该源内容 | 最终结果完全不依赖该源 |
| `counterfactual_replay` | 不固定 | 从输入事件流删除该来源，重算运动、支持判断、归一化与全部状态，与独立批量参考比较 | 整个结果已不依赖该源（在该事件流与版本下） | 超出精确回放范围的精确性（见 §1.5） |

**规则 E6。**

1. `fixed_decision_replay` 必须输出冻结决策 manifest（冻结了哪些决策、其版本）。**依赖被撤回源内容的 motion 必须标为「冻结决策依赖」**——此时"结果不变"是该冻结条件的条件性结论，不能升级为独立性声明。
2. 两个接口的比较对象、替换内容、容差、比较量必须写在 manifest 中；替换内容需覆盖至少一种会改变内容通路的置换（如置零/替换为另一合法源/随机值）。
3. `counterfactual_replay` 的批量参考必须来自独立实现（`reference.py` 独立 FP64 稠密参考，不能调用生产 `statistics.py`/`replay.py` 当 oracle，P1_ANALYTIC_REFERENCE §1）。
4. 只有 `counterfactual_replay` 后，才允许输出"结果不依赖该源"；且该声明必须同时覆盖 §1.1 中所有被声明清除的角色（evidence 与 context）。
5. 撤回超出精确范围时，`counterfactual_replay` 只能声明"从合法边界重建 + 历史损失"，不能声明精确等价。

**失效。** 用 `fixed_decision_replay` 的结果冒充完整独立性；冻结的 motion 实际由被撤回源估计而来却未标注；批量参考与流式实现共享代码（同源缺陷被同时掩盖）；只比较 mask 或单个中间量；非线性特征用"减去旧特征"假装撤回；上下文角色未被隔离测试覆盖。

**映射。**

| 机制 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 两个接口与 manifest | `replay.py::fixed_decision_replay` / `counterfactual_replay` | P1 已实现（manifest 记录替换内容、比较量、容差与 `not_claimed`） |
| 独立批量参考 | `reference.py`（独立 NumPy FP64 稠密实现） | P1 已实现（`counterfactual_replay(reference_fn=...)` 由调用方注入 oracle；`replay.py` 不导入它） |
| 冻结决策依赖标注 | `replay.py::fixed_decision_replay(frozen_decision_deps=...)` → `FixedDecisionReplayResult.conditional_on` | P1 已实现 |
| 接口测试 | `tests/encoders/n01_full/test_replay.py::test_fixed_decision_replay_is_falsifiable`（含泄漏负对照）、`test_counterfactual_replay_matches_independent_batch_reference` | P1 已实现 |
| 依赖闭包的玩具非线性算子 | `replay.py::ToyProgram` + `recompute_successor_closure` | P1 已实现（`test_replay.py::test_dependency_closure_recomputation_matches_full_retraction`） |

---

## 2. T1 — 唯一来源与合并/分裂/撤回的一致性

### 2.1 定义（精确到可判定）

固定版本三元组 `v = (parameter_version, mapping_version, calibration_version)` 与固定求解规则：

- 域 `D` 的证据集合 `S(D) = set(D.evidence_source_ids)`；规范化 = 按 `SourceKey` 去重后按声明字典序（`dataset_id, episode_id, camera_id, frame_id, channel_or_phase, sensor_y, sensor_x`）排序。
- 操作语义：`S(merge(A,B)) = S(A) ∪ S(B)`；`S(split(D)) = S(D)` 的划分（子域由子域成员重新拟合，不平均分配父均值）；`S(retract(D,R)) = S(D) \ R`，且 `S(A∪B)` 去重后决定同一个参考目标函数。
- 判定程序：对同一事件流做三种执行——(i) 直接对来源集合求解；(ii) 以不同顺序/分块执行事件后求解；(iii) 撤回集合 `R` 后对剩余来源直接求解——比较充分统计量 `(Λ_D, η_D)` 与 `θ̂_D`。
- 通过条件：在声明数值容差内一致；撤回路径必须是"取剩余观测的协方差子块重新求逆再求解"，不是截取旧精度矩阵。
- 失败条件：任何顺序/分块差异超容差；撤回路径复用被删观测的旧精度/旧先验；重复来源被计成独立观测。

### 2.2 假设

- `[假设]` A-T1-1 每个来源有唯一稳定 ID（`SourceKey`），不依赖像素值、batch slot 或插值结果（`[事实]` 共同契约 §3 已规定该字段）。
- `[假设]` A-T1-2 所有有效因子被保留、不被隐式剪枝；每个来源在单个已提交估计中只计一次。
- `[假设]` A-T1-3 集合运算期间 `v`、basis、坐标与先验固定；跨版本的因子不得混在同一个目标函数里。
- `[假设]` A-T1-4 相关性以联合因子（协方差块）表示；共享黑电平、读出相关、插值相关不得拆成独立的伪来源项。
- `[假设]` A-T1-5 浮点实现使用规范排序/归约并声明容差；不承诺任意 GPU 核逐位一致（方案 §5.1）。
- `[假设]` A-T1-6 若使用非线性求解，需固定初始化、迭代与确定性执行才能谈"解一致"；否则只谈目标函数一致。
- `[假设]` A-T1-7 一个 RAW 样本可被多个候选域引用，但在单个估计中只计一次。

### 2.3 待证明命题（当前全部未证明）

- **P-T1-a（集合代数与执行顺序）。** 对任意合法域与任意事件分块/顺序，合并后来源集合去重并规范排序所得的 `(Λ_D, η_D)` 与直接求和一致（容差内）。集合运算的幂等/交换/结合是集合定义性质；本项目的证明义务在于**实现是否真的按该运算执行**，以及测试是否存在。当前：无新契约实现、无测试。
- **P-T1-b（撤回等价）。** 对任意 `R ⊆ S(D)`，`retract` 后从剩余来源直接求解，与"取原观测协方差 `Σ` 的剩余子块重新求逆再求解"一致。
- **P-T1-c（相关性正确）。** 已知相关结构时，撤回后的再求解等于在联合模型上按声明语义（条件化/边缘化）的结果；不等价于截取旧精度矩阵。
- **P-T1-d（分裂隔离）。** `split` 的子域由子域成员重新拟合；父均值不被平均分配；子域来源集合是父集合的划分。
- **P-T1-e（重复/重叠语义）。** 同一帧的重复传输不增加独立观测数；重叠域合并后不产生额外独立信息；跨相机/跨 episode 的相同 `frame_id` 不冲突。

### 2.4 反例与失效条件

- `[反例]` X-T1-i **截取旧精度矩阵**：`Σ = [[1, 0.5],[0.5, 1]]`、`Φ = [1,1]ᵀ`、`y = [1,3]` 时 GLS 均值为 2、估计方差 0.75；撤回第二观测后正确方差为 1，而"直接截取旧精度矩阵求逆"仍给 0.75。P1 已要求该错误版本必须被测试识别（P1_ANALYTIC_REFERENCE §4.2，待实现）。
- `[反例]` X-T1-ii **重复包计两次**：`y=[1,3]`、`σ²=1` 的正确均值为 2、方差 0.5；把重复到达的同一 `SourceKey` 当第三个观测会得到方差 1/3，系统性低估不确定性。
- `[反例/事实]` X-T1-iii **当前实现不满足时间来源撤回**：`src/owac/frameworks/n01_dbsr_raw.py::RawDomainHistory` 只检验空间图，半径 2、历史 8 固定，同坐标旧时间来源不会因空间边关闭而撤回（方案 §2；01_CURRENT_IMPLEMENTATION.md）。这不是 T1 的反例，而是"现有实现未满足 T1"的证据。
- `[反例/事实]` X-T1-iv **旧空间撤回重算的实验增量极小**：动态旧缓存/动态重算为 1.55314/1.55214，原 DBSR 为 1.54597（方案 §2 记录的既有事实）。它说明微小的均值差不构成机制成立证据，不构成 T1 的否证。
- `[失效]` F-T1-1 顺序/分块差异来自浮点加法顺序；只能要求容差与规范归约，不能要求逐位一致。
- `[失效]` F-T1-2 版本混用：不同 parameter/support/mapping/calibration 版本的因子混算，集合代数不适用（见 §1.4）。
- `[失效]` F-T1-3 非线性求解：只保证目标函数一致；不固定初始化/迭代时解不一致。
- `[失效]` F-T1-4 相关噪声拆分：共享噪声被当独立来源 → 目标函数与正确联合模型不同。
- `[失效]` F-T1-5 先验重复计入：跨域合并时同一先验只计一次（方案 §4.3 `merge` 行）。

### 2.5 代码与测试映射

| 义务 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 来源身份/字段 | `contracts.py::SourceKey` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 账本与幂等登记 | `ledger.py::MeasurementLedger.observe / get_sources / reset / without` | P1 已实现（`test_ledger.py`） |
| 合并/分裂/撤回 | `domains.py::make_domain / merge_domains / split_domain / retract_sources / restore_sources` | P1 已实现（`evaluation/n01_full.py analytic --case set_algebra`） |
| GLS 求解与秩 | `statistics.py::fit_structure` | P1 已实现 |
| 独立 FP64 oracle | `reference.py` | P1 已实现 |
| 数值与性质测试 | `tests/encoders/n01_full/test_ledger.py`、`test_statistics.py`、`test_replay.py` | P1 待实现 |
| 评估入口 | `src/owac/evaluation/n01_full.py`（`analytic` 子命令，9 个 case） | P1 已实现 |
| 旧去重/重叠语义回归 | `tests/encoders/n01/test_acceptance_regressions.py::test_fractional_overlap_uses_each_original_source_once` | 已存在（历史参照） |

### 2.6 P1 实现状态与证据（2026-10-14）

- **代码入口。** `ledger.py::MeasurementLedger`（身份去重、部分 reset、`without` 反事实视图、`with_content` 派生内容视图）、`domains.py::make_domain / merge_domains / split_domain / retract_sources / restore_sources`（`content_digest` 与血缘版本 `support_version` 分离）、`statistics.py::fit_structure`、`reference.py::dense_gls`、`replay.py::full_replay / fixed_decision_replay / counterfactual_replay`。
- **已执行的反例。** X-T1-i（截取旧精度矩阵）由 `test_statistics.py::test_correlated_observations_and_precision_slice_counterexample` 与 `analytic` case `correlated_two_observation` 固定：正确撤回方差 1.0、错误版本 0.75，二者被判别；X-T1-ii（重复包计两次）由 `test_ledger.py::test_duplicate_packet_does_not_advance_counts_or_precision` 与 case `duplicate_then_new_id` 固定（重放不增加测量数；数值相同的新曝光仍然登记）。
- **集合代数。** 交换/结合与规范排序、split 再 merge、retract 再 restore、同一事件流的不同分块由 `analytic --case set_algebra` 固定；比较只在声明版本固定时进行（`content_digest` 相同，`support_version` 因血缘不同而不同）。
- **仍未证明。** P-T1-a…e 仍是命题；P1 只提供有限小问题上的数值与行为证据，不含任意域/任意事件图的一般证明。

---

## 3. T2 — 固定支持条件下的估计与不确定性

### 3.1 定义（精确到可判定）

固定域 `D`、固定设计矩阵 `Φ`（每行是映射后真实采样坐标的基函数，按 CFA 相位分列；相位不先求均值）、已知零均值协方差 `Σ ≻ 0`、模型正确且可辨识；无先验支路 `Λ = ΦᵀΣ⁻¹Φ`、`η = ΦᵀΣ⁻¹y`、`θ̂ = solve(Λ, η)`、`cov = Λ⁻¹`；有先验支路另行加 `Λ_prior/η_prior`。

- 判定程序：把生产 `fit_structure` 的输出与独立 FP64 稠密参考（`reference.py`，不调用生产代码）比较 `θ̂`、`covariance`、秩、`identifiable_mask`、观测计数。
- 通过条件：声明容差内一致；满秩无先验时 `cov = (ΦᵀΣ⁻¹Φ)⁻¹`；独立等方差退化为样本均值与 `σ²/n`；先验单列且不计入观测数。
- 失败条件：伪逆零方向被当作零方差；先验被算成观测信息；数据依赖选域后仍宣称固定设计协方差。

### 3.2 假设

- `[假设]` A-T2-1 域与设计矩阵在估计前固定，或由独立校准数据产生；不由待估数据选择。
- `[假设]` A-T2-2 噪声零均值；`Σ` 已知且正定；局部常数/仿射基模型正确。
- `[假设]` A-T2-3 观测可辨识（设计矩阵满秩）；否则只在可辨识子空间报告。
- `[假设]` A-T2-4 相位、坐标与单位一致；不同相位不先平均（方案 §4.2）。
- `[假设]` A-T2-5 相关噪声以块形式进入：共享黑电平误差、插值相关；插值 `z = A y` 的噪声为 `AΣAᵀ`，同一 RAW 帧的原始节点与积分节点不得当作两组独立观测。
- `[假设]` A-T2-6 缺失/坏点/饱和的处理与高斯模型一致或明确排除（见 §1.2）。
- `[假设]` A-T2-7 `uncertainty_kind` 标签不混用：已知 `Σ` 的精确协方差 ≠ 未知相关性的保守界 ≠ 神经代理方差。

### 3.3 待证明命题（当前全部未证明）

- **P-T2-a（`[文献]` Gauss–Markov，本项目未验证）。** 在 A-T2-1/2/3 下，无先验 GLS 是最佳线性无偏估计，协方差为 `(ΦᵀΣ⁻¹Φ)⁻¹`；独立等方差常数模型退化为样本均值及 `σ²/n`。该结论的适用条件由文献给出，本项目尚未写出证明，也未在实现中验证。
- **P-T2-b（生产=独立参考）。** 随机小问题（分数坐标、异方差、已知仿射场）上生产实现与独立 FP64 稠密参考一致，并记录条件数（P1_ANALYTIC_REFERENCE §4.3–4.4，待实现）。
- **P-T2-c（秩与可辨识）。** 秩亏时 `identifiable_mask` 精确标出可辨识子空间；不给出虚假 confidence；伪逆零值不被当作零方差方向。
- **P-T2-d（先验与观测计数）。** `prior_id` 单列且不计入独立观测数；无先验满秩时观测计数 = 去重后的唯一来源数；同一 RAW 样本被多个候选域引用时在单次估计中只计一次。
- **P-T2-e（保守界）。** 未知相关性下单列固定权重方差保守界 `Var(Σ aᵢεᵢ) ≤ (Σ |aᵢ|σᵢ)²`，其保守性有数值展示；不与已知 `Σ` 的精确协方差共用标签（方案 §5.2）。

### 3.4 反例与失效条件

- `[反例]` X-T2-i **数据依赖选域**：用当前数据选域、估计运动或预测权重后，固定设计的 BLUE/协方差不再自动成立；**把权重 `detach` 不消除统计选择偏差**（方案 §5.2 限制段）。
- `[反例]` X-T2-ii **先验当观测**：`Λ_prior/η_prior` 计入观测计数 → 方差低估、独立观测数虚增（违反 §1.3 规则 3）。
- `[失效]` F-T2-1 未知相关/共享黑电平 → `Σ` 误设；插值相关被忽略。
- `[失效]` F-T2-2 饱和截断 → 分布截断，均值/方差有偏；不得按普通高斯处理。
- `[失效]` F-T2-3 局部仿射模型失配 → 结构偏差不进入协方差，必须单独报告。
- `[失效]` F-T2-4 病态设计 → 条件数爆炸；不得靠放大测试容差掩盖，需单报条件数。
- `[失效]` F-T2-5 神经"方差"未通过覆盖检验 → 只能标代理（方案 §5.2、§6.2）。

### 3.5 代码与测试映射

| 义务 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| GLS 实现 | `statistics.py::fit_structure`（`solve_triangular` + `lstsq` + Cholesky 逆） | P1 已实现 |
| 独立参考 | `reference.py`（`eigh` 白化路线） | P1 已实现 |
| 先验/秩/标签字段 | `contracts.py::EvidenceEstimate` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 数值测试（手算 + 随机 + 病态） | `tests/encoders/n01_full/test_statistics.py` | P1 已实现（病态问题单报条件数，不靠放大全局容差） |
| 现有解析统计入口（旧语义） | `src/owac/frameworks/n01/support.py::SupportDomainIntegrator.domain_statistics` | 已存在（旧实现，非新契约验收） |
| 噪声模型与校准 | `src/owac/frameworks/n01_full/calibration.py` | P3 待实现 |
| 运动方差近似 | `src/owac/backends/raw_correspondence.py`（`∇fᵀΣ_flow∇f` 只作近似，需留出数据检验） | P2 待实现 |

### 3.6 P1 实现状态与证据（2026-10-14）

- **代码入口。** `statistics.py::fit_structure`（逐相位 `cholesky` → `solve_triangular` → `lstsq`；满秩协方差由 `Lambda` 的 Cholesky 求逆；先验以 `Lambda_prior/eta_prior` 单列一次）与 `reference.py::dense_gls`（独立 `eigh` 白化 + 显式稠密逆，**不导入** `statistics.py`）。
- **已执行的数值验证。** 手算（y=[1,3]、σ²=[1,1] → 均值 2、方差 0.5；Σ=[[1,0.5],[0.5,1]] → 0.75，撤回第二观测后 1.0）；随机异方差仿射问题与 oracle 的预测一致（FP64 `atol=1e-10, rtol=1e-8`）；FP32 良态问题（`atol=1e-5, rtol=1e-4`）；病态设计单报条件数（>1e6，且不比较没有有效数字的预测方差）；秩亏掩码与标签；先验单列不计入独立观测数；未知相关性下固定权重保守界 `(Σ|aᵢ|σᵢ)²` 的数值保守性（等相关 0.5 噪声下经验方差小于该界，且严格大于独立观测公式）。
- **标签分离。** `covariance`（已知 Σ 的精确协方差或后验）、`proxy`（未知相关性的保守界，PSD 形式 `(|C|σ)(|C|σ)ᵀ`）、`none`（秩亏且无先验）由 `EvidenceEstimate.uncertainty_kind` 区分并分别测试。
- **限制（必须带入后续阶段）。** 自适应选域、由数据估计 flow/对应或数据依赖权重之后，**固定设计 GLS 的 BLUE/协方差不再自动成立**（X-T2-i）；把权重 `detach` 不消除统计选择偏差。P2/P3 引入这些步骤时必须另建合法条件构造（独立留出或重新推导），不得直接复用本节结论。
- **仍未证明。** P-T2-a…e 仍是命题；"生产 = 独立参考"只在声明的小问题上成立。

---

## 4. T3 — 顺序决策的误撤回控制

### 4.1 定义（精确到可判定）

对统计参考支路定义非负过程 `(E_t)_{t≥0}`，`E_0 = 1`，关于合法信息流 `F_t` 可测，并满足 `E[E_t | F_{t-1}] ≤ E_{t-1}`（非负上鞅）。撤回规则：首次出现 `E_t ≥ 1/α` 时撤回。

- **判定/通过条件。** 在上述条件成立时，`P_{H0}(∃t: E_t ≥ 1/α) ≤ α`（`[文献]` Ville 型界；本项目未证明、未验证其实现）。
- **判定程序（两层）。** (a) 构造层：检查 `E_t` 只用 `F_{t-1}` 与 t 时刻合法数据；(b) 经验层：独立静态长序列上的误撤回率 ≤ α，重采样单元为 episode/场景，不把帧当独立实验（方案 §9.3）。
- **多域/多候选/重启。** 预先分配 `α_j`，使 `Σα_j ≤ α_total`；单个域的保证不自动扩展到整段视频。
- **合并（merge）判定。** 使用"差异区间落入预先声明的结构容差"的等效性判断；**不能用"没有拒绝不一致"代替"已经相容"**（方案 §5.3）。

### 4.2 假设

- `[假设]` A-T3-1 零假设 `H0`（支持仍有效）有明确的概率模型与参数；残差在 `H0` 下的分布已知或有合法构造。
- `[假设]` A-T3-2 运动估计、候选生成与下注参数不依赖待检验的同一测量；若依赖，需给出合法的条件构造、独立留出或重新推导。
- `[假设]` A-T3-3 信息流 `F_t` 忠实反映实际使用数据；不引入未来帧、隐藏真值或 oracle。
- `[假设]` A-T3-4 `α_j` 预先分配，不在看到数据后调整阈值。
- `[假设]` A-T3-5 数据无信息或变化过小时，不承诺有限帧内必然检测；必须允许"尚未确定"态。
- `[假设]` A-T3-6 BOCPD/DSM-bocd 只提供模型内的时间分段后验，不自动满足 A-T3-1–A-T3-4；两者必须分开标注（方案 §5.3）。
- `[假设]` A-T3-7 等效性容差以传感器测量或几何任务单位声明；不直接比较未归一化的斜率与截距。

### 4.3 待证明命题（当前全部未证明）

- **P-T3-a（合法过程构造）。** 存在满足上鞅条件的 `E_t` 构造（似然比混合或现成 confidence-sequence 方法），且其 `H0` 模型与本项目残差相容。当前：未构造、未移植、无测试。
- **P-T3-b（α 分配）。** 多域、多候选与反复重启下的 `α_j` 分配使全视频误撤回率 ≤ `α_total`；单域保证不被外推。
- **P-T3-c（自适应选择下的合法性）。** 选域、估运动、定权重都使用当前数据时，仍存在合法的条件构造或独立留出方案。**这是 T3 的最大风险点。**
- **P-T3-d（等效性检验）。** merge 的差异区间检验在声明容差与功效下工作，且"未拒绝"不被当作"已相容"。
- **P-T3-e（外部实现边界，`[文献]` confseq）。** confseq 提供 time-uniform、非参、非渐近边界与停止规则，条件是其过程与信息流合法；`f_{t-1}` 可用、无未来信息、`E_t` 为非负上鞅（或 e-process 构造）。本项目尚未编译/移植（`src/confseq/boundaries.cpp` 与 `uniform_boundaries.h` 是 C++ 入口，无 `boundaries.py`；02_REFERENCES），未做前缀逐点数值对比。

### 4.4 反例与失效条件

- `[反例]` X-T3-i **把残差 sigmoid 改名成概率**：没有上鞅构造、没有合法 `H0` 时，`1/α` 阈值不控制任何错误率（方案 §5.3 明确禁止）。
- `[反例]` X-T3-ii **用同一测量先选后用**：用待检验测量选择运动/候选/下注参数后再检验，条件合法性被破坏（可选停止/选择偏差）。
- `[反例]` X-T3-iii **多域共享一个 α 而不分配**：家族误报率上界失效。
- `[失效]` F-T3-1 BOCPD 后验被当作 p 值或 confidence sequence：DSM-bocd 的鲁棒性也只有其模型条件；两者都不自动给出频率学误报界。
- `[失效]` F-T3-2 变化过小/无信息：有限帧内保证检测不可能；应输出"尚未确定"。
- `[失效]` F-T3-3 非平稳、未知相关、缺失数据破坏 `H0` 模型。
- `[失效]` F-T3-4 等效性用"没有拒绝不一致"代替"已经相容"。

### 4.5 代码与测试映射

| 义务 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 时间支持与分段 | `src/owac/frameworks/n01_full/temporal_support.py` | P3 待实现 |
| 空间相容与合并判据 | `src/owac/frameworks/n01_full/spatial_support.py` | P3 待实现 |
| 校准与 α 分配 | `src/owac/frameworks/n01_full/calibration.py` | P3 待实现 |
| 评估入口 | `src/owac/evaluation/n01_support.py` | P3 待实现 |
| 条件表（假设→支路→标签） | 阶段报告 `research/experiments/n01_full/P3_report.md` 与协议 `configs/encoders/n01_full/protocol_v1.json` | P3 待实现 |
| 外部参考实现（阅读材料，未安装） | DSM-bocd `bocpd.py` / `models.py`（提交 `9d05117…`）；confseq `boundaries.cpp` / `uniform_boundaries.h`（提交 `5ffe733…`） | 已取回、未移植、未验证 |
| 现有时间统计回归（旧语义） | `tests/encoders/n01/test_temporal.py`、`tests/encoders/n01/test_acceptance_regressions.py::test_temporal_stability_is_computed_per_trajectory` | 已存在（历史参照） |

---

## 5. T4 — 可变时间常数与局部交换的稳定边界

### 5.1 定义（精确到可判定）

解析状态场（方案 §5.4）：

```text
s_(t+1) = D_t P_t s_t + (I − D_t) u_t
D_t = diag(exp(−Δt / τ_j,t))
P_t ≥ 0，且每一行和为 1
0 ≤ D_jj ≤ d_max < 1
```

- **判定 1（有界性）。** 给定 `‖u_t‖∞ ≤ U`，检查 `‖s_t‖∞ ≤ max(‖s_0‖∞, U)`。
- **判定 2（收缩）。** 两条状态轨迹接受相同输入与相同外生系数、初值不同时，差值 `∞` 范数按 `d_max` 单调不增。
- **通过条件。** 在 A-T4-1/2 下两个判定成立（方案 §5.4 设想，本项目尚未写出证明与测试）。
- **失败条件。** `P_t` 或 `D_t` 依赖各自隐藏状态时，上述两个判定不再由凸组合直接给出；需要额外的 Lipschitz/small-gain 条件。

### 5.2 假设

- `[假设]` A-T4-1 状态参数化与上述公式一致；`P_t ≥ 0` 行随机；`0 ≤ D_jj ≤ d_max < 1`；比较窗口内系数外生固定。
- `[假设]` A-T4-2 行和小于 1 时的缺失质量被显式解释为泄漏到已声明的零/基准状态；不能无意中改变测量单位（方案 §5.4）。
- `[假设]` A-T4-3 门控/非线性网络的稳定性需要另加 Lipschitz/small-gain 条件；Mamba 的负状态参数与正步长只支持其**线性子状态**的分析（`[文献]`，02_REFERENCES；方案 §5.4）。REN 的严格稳定性结论不能外推到修改后的任意结构。
- `[假设]` A-T4-4 稳定性 ≠ 证据正确性：稳定状态可以长期记着错误内容；撤回仍按 T1/T5 检验（方案 §5.4 末段）。
- `[假设]` A-T4-5 状态重建必须能从干净检查点重放，且受 §1.5 的精确回放范围约束。

### 5.3 待证明命题（当前全部未证明）

- **P-T4-a（解析支路的界）。** 在 A-T4-1/2 下，`‖s_t‖∞` 的上界与两条轨迹的收缩界成立。方案 §5.4 给出设想与证明思路（凸组合与诱导范数）；**本项目尚未写出完成证明，也未写测试**。
- **P-T4-b（数值实现）。** 数值实现（含 FP32/BF16）满足该界，或给出偏离量与条件（方案要求 FP32 优先、转 BF16 前验证误差）。
- **P-T4-c（门控支路条件）。** 加入门控、投影、残差与空间交换后，恢复类似界所需的额外条件（Lipschitz 常数、small-gain）是什么、能否实测。
- **P-T4-d（`[文献]` 边界，未验证）。** Mamba/S4 的分析适用范围、REN 的保证范围；本项目不接入 REN 仓库（Julia），不作移植承诺。

### 5.4 反例与失效条件

- `[反例]` X-T4-i **`d_max = 1`（无泄漏）**：两条轨迹的差异不收缩（保持初值差），收缩命题失效。
- `[反例]` X-T4-ii **`P_t` 含负元素或行和 > 1**：非凸组合，`∞` 范数上界不成立。
- `[失效]` F-T4-1 隐藏状态依赖的门控：§5.4 的证明不直接成立，需要 A-T4-3 的额外条件。
- `[失效]` F-T4-2 状态稳定但内容错误：稳定上界不提供证据正确性；必须另测撤回。
- `[失效]` F-T4-3 状态规模：方案 §6.3 的 4 层 `d_state=64`、`expand=2` 配置若每位置保留 FP32 SSM 状态，单位置约 `4 × 1024 × 64 × 4 ≈ 1 MiB`（未计短卷积、梯度与 RAW 缓冲）；不能宣称线性复杂度就适合全分辨率，需要较粗域网格/有限活跃域/状态共享。

### 5.5 代码与测试映射

| 义务 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 解析泄漏/局部交换参考 | `src/owac/frameworks/n01_full/state_field.py` | P5 待实现 |
| 神经状态候选（官方 Mamba 模块） | 在 `state_field.py`/`encoder.py` 内接入；两种 cache 一起重放 | P5 待实现 |
| 界与收缩测试 | `tests/encoders/n01_full/test_state_replay.py` | P5 待实现 |
| 状态规模/时延 Pareto 报告 | `research/experiments/n01_full/P5_report.md` | P5 待实现 |
| 旧有界更新参照 | `src/owac/frameworks/n01/content.py::ContentUpdate`、`tests/encoders/n01/test_source_repairs.py::test_gate_units_are_bounded_differentiable_and_invertible` | 已存在（旧实现的界性质，非 T4 验收） |

---

## 6. T5 — 局部重建与全量重放等价

### 6.1 定义（精确到可判定）

依赖图 `G = (V, E)`，每个派生节点记录 `parent_ids` 与四版本（§1.4）。给定失效集合 `I`（必须覆盖所有受影响后继的闭包）：

- **判定程序。** 按拓扑序重算 `I` 中节点、其余节点保持不变，与同规则下的全量计算比较；比较粒度到节点输出与最终读出快照。
- **递归网络判定。** 把有界时间窗展开成依赖图，从**未受污染的检查点**重放到当前时刻；局部执行与全量参考分别跑。
- **通过条件。** 在声明容差内一致（`[待证明]` 依赖闭包上的归纳，适用于非线性算子，不要求特征线性）。
- **失败条件。** 依赖图不完整、失效集合未覆盖闭包、版本不一致、检查点受污染、比较量不完整。

### 6.2 假设

- `[假设]` A-T5-1 依赖图完整：每个派生节点声明全部父项；**卷积感受野跨界、递归网络的短卷积与 SSM 两类缓存、融合权重预测器**都在图内（方案 §4.5、§5.5）。
- `[假设]` A-T5-2 算子版本固定；失效节点集合恰好覆盖所有受影响后继，不做"只改 mask"的近似。
- `[假设]` A-T5-3 四个版本一致；参数更新后旧参数产生的派生缓存全部失效（§1.4）。
- `[假设]` A-T5-4 检查点未受污染；精确回放范围 ≤ 32 个唯一帧；超出范围时作废污染摘要并从合法边界重建并记录历史损失（§1.5）。
- `[假设]` A-T5-5 等价性是"声明容差内一致"，不是逐位一致；容差与比较量写入 manifest。

### 6.3 待证明命题（当前全部未证明）

- **P-T5-a（局部=全量）。** 在 A-T5-1/2/3 下，按拓扑序重算失效闭包与全量计算一致（依赖闭包上的归纳；非线性算子不需要线性假设）。方案 §5.5 给出证明思路，本项目尚未写出证明与测试。
- **P-T5-b（递归重放）。** 把递归窗展开为依赖图并从未污染检查点重放，与独立批量参考一致；Mamba 类模块的短卷积状态与 SSM 状态必须一起重放。
- **P-T5-c（版本不变量）。** 参数/support/mapping/calibration 任一版本变化 ⇒ 受影响派生缓存失效；`N-stale` 组是刻意违反者，只作因果诊断。
- **P-T5-d（两接口语义）。** `fixed_decision_replay` 与 `counterfactual_replay` 的结论边界（§1.6）在实现中可判定并可审计。

### 6.4 反例与失效条件

- `[反例]` X-T5-i **卷积感受野跨域边界**：只重算域内像素、不重编码受影响区域时，局部重算 ≠ 全量重放；必须重编码局部区域（使用声明的边界填充）或保守撤回整个特征节点（方案 §4.5）。
- `[反例]` X-T5-ii **只把旧隐状态乘零 mask**：不等价于重建；02_REFERENCES 记录 Mamba `step` 同时更新 conv_state 与 ssm_state，两者都必须重放。
- `[反例]` X-T5-iii **参数更新后复用旧派生缓存**：违反版本不变量；"缓存命中率高"不是等价性验收（方案 §5.5 明确禁止）。
- `[失效]` F-T5-1 摘要超出精确撤回范围：单个来源无法从摘要中移除时，必须整体作废并重建，或宣布超出范围；禁止"只存均值却承诺任意来源精确撤回"（方案 §4.6）。
- `[失效]` F-T5-2 依赖图不完整时，"未发现差异"是假阴性，不是等价。
- `[失效]` F-T5-3 只比较 mask 或单个中间量，不能声明完整撤回。
- `[失效]` F-T5-4 非线性局部最优问题（`[文献]` iSAM2 语境）：非线性局部最优 ≠ 全局最优，已边缘化摘要不一定可任意撤回；本项目不作 GPU 主线实现，只借概念。

### 6.5 代码与测试映射

| 义务 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 全量/两种回放接口 | `replay.py::full_replay / fixed_decision_replay / counterfactual_replay` | P1 已实现 |
| 独立批量参考 | `reference.py` | P1 已实现（由 `reference_fn` 注入，不与生产共享代码路径） |
| 失效传播与依赖图 | `invalidation.py`、`contracts.py::DependencyNode` | P5 / P0 已实现（契约字段/行为，不表示理论已证明） |
| 依赖闭包玩具算子 | `replay.py::ToyProgram`（sources → normalization → centering → tanh → quadratic → readout） | P1 已实现 |
| 重放测试 | `tests/encoders/n01_full/test_replay.py`（P1 已实现）、`test_state_replay.py`（P5 待实现） | P1/P5 待实现 |
| 现有唯一帧/重复帧实现 | `src/owac/frameworks/n01/state.py::push / reset / detach` | 已存在（旧语义参照） |
| 现有内容重算基线 | `src/owac/baselines/burst_features.py::BurstFeatureEncoder.encode_frames`、`MaskedDbsrFusion.forward` | 已存在 |

### 6.6 P1 实现状态与证据（2026-10-14）

- **代码入口。** `replay.py::full_replay`（一次完整求解、`ReadoutSnapshot` 提交、32 帧精确窗口与历史损失）、`fixed_decision_replay`（冻结决策 + 内容替换，输出 `conditional_on`）、`counterfactual_replay`（删除来源、重算全部决策、可由 `reference_fn` 注入独立 oracle 比较）、`ToyProgram` / `recompute_successor_closure`（依赖闭包）。
- **已执行的性质测试。** 依赖闭包重算 = 从零重算（两层非线性算子，差 ≤1e-12）；"只把被撤回输入置零"的错误版本被固定为不等价（差 >1e-6）；固定决策回放的可满足性与泄漏负对照；越界撤回只声明"从合法边界重建 + 历史损失"。
- **依赖图与版本。** `DependencyNode.is_stale_against` 由 `test_replay.py::test_version_changes_trigger_recomputation` 固定：support/parameter 版本变化使派生节点失效；`ContentReplacement` 是版本化派生视图，不重写原始记录。
- **仍未证明。** P-T5-a…d 仍是命题；P1 的玩具算子只作**性质测试**，不是正式内容骨干；卷积感受野跨界、递归网络双缓存、从未污染检查点重放等仍属 P4/P5。

---

## 7. T6 — 可中断执行与不确定性读出

### 7.1 定义（精确到可判定）

三层，分别有不同强度：

1. **操作预算与版本一致性（可严格判定的不变量）。** 任意允许的中断点只读已提交快照（`ReadoutSnapshot` 的所有字段对应同一支持版本下的一次已提交结果）；读出操作不生成新证据、不推进测量时钟；重复读出不改变 `source_time`。
2. **deadline SLO（经验量，不是硬实时）。** 端到端超期比例，包含排队、传输、撤回、重算与 π0.5 推理；按实测分布报告。
3. **统计提前停止。** 仅在 T3 的条件下，允许停止时刻依赖已见数据而保持覆盖。

- **判定程序。** (a) 版本一致性：在事件流任意前缀中断，检查读出与提交边界对齐，不存在半旧半新；(b) deadline：超期时返回已完成的合法状态或当前帧成熟单帧回退，全无效时输出无效 mask 并由预定义策略处理；(c) 成本口径：等待、撤回、重建、读出与传输全部计入。
- **通过条件。** 三层各自独立成文：只有第 1 层可作为严格保证；第 2 层是实测分布；第 3 层继承 T3 条件。
- **失败条件。** 把 p50/p95 当硬实时；读未提交状态；在中断点执行 observe/重算；在没有 T3 条件时宣称覆盖。

### 7.2 假设

- `[假设]` A-T6-1 提交是原子的（版本事务）；未完成版本不可读、不可部分读出。
- `[假设]` A-T6-2 统计支路的停止规则满足 T3 条件；工程/神经分支的置信度不继承该保证。
- `[假设]` A-T6-3 实测延迟分布对应真实硬件与负载；`p50/p95` 不等于 WCET（方案 §5.6）。
- `[假设]` A-T6-4 状态估计误差界 ≠ 机器人动作误差界；最终动作后果必须实测。
- `[假设]` A-T6-5 研究范围是表征读出期限；改变重规划频率、提前中断动作块或主动增加曝光属于新的控制/采集协议，必须单独比较（方案 §4.6）。

### 7.3 待证明命题（当前全部未证明/未测量）

- **P-T6-a（快照不变量）。** 任意中断点只读已提交快照、不生成新证据、重复读出不推进时钟；实现与测试均不存在。
- **P-T6-b（覆盖性停止）。** 在 T3 条件下，提前停止的覆盖性质在实现中成立。
- **P-T6-c（SLO，未测量）。** 端到端超期比例与等待/撤回/重建成本；未测量项必须写"未测量"。
- **P-T6-d（WCET，未做）。** 若要求确定的最坏执行时长，需另做 WCET/资源隔离分析（本方案不做，方案 §5.6 明示）。
- **P-T6-e（动作后果，未验证）。** 状态误差界到动作误差界的外推不成立；必须实测。

### 7.4 反例与失效条件

- `[反例]` X-T6-i **把 GPU p95 时延当硬实时保证**：平均/p95 不构成 WCET。
- `[反例]` X-T6-ii **deadline 到达时读一半旧状态一半新状态**：违反版本事务（方案 §4.6 明确禁止）。
- `[反例]` X-T6-iii **中断点执行 observe/重算**：违反"只读快照"，使"中断不生成新证据"不成立。
- `[失效]` F-T6-1 T3 条件不成立时的提前停止：覆盖保证无效。
- `[失效]` F-T6-2 状态误差界 → 动作误差界的外推；不成立。
- `[失效]` F-T6-3 把读出期限研究混入控制/采集协议变化：收益不可归因。
- `[失效]` F-T6-4 精确回放范围（32 唯一帧，§1.5）已耗尽时仍宣称精确撤回并按时读出：不可能同时满足，必须退回历史损失声明或单帧通路。

### 7.5 代码与测试映射

| 义务 | 目标路径与符号 | 状态 |
| --- | --- | --- |
| 快照字段与提交语义 | `contracts.py::ReadoutSnapshot` | P0 已实现（契约字段/行为，不表示理论已证明） |
| 期限与回退策略 | `src/owac/frameworks/n01_full/deadline.py` | P5 待实现 |
| deadline/快照测试 | `tests/encoders/n01_full/test_deadline.py` | P5 待实现 |
| 统一读出接口（复用并重新验收） | `src/owac/backends/rawvla_features.py::VisualTokenPostprocessor`、`VisualPrefix`、`RawVLAFeaturePolicy` | 已存在；P7 重新验收 |
| 现有服务/episode reset/重复帧路径 | `src/owac/runtime/n01_feature_service.py` | 已存在（旧模型工厂，非新契约验收） |
| 动作端到端后果 | `src/owac/training/n01_action.py`、P7/P8 训练与闭环协议 | P7/P8 待实现 |

---

## 8. 外部保证的适用边界（引用纪律）

使用 DSbocd/confseq 一类外部保证时，必须写明其适用条件与本项目尚未验证的部分。下表是当前引用边界；**不是**对实现的验收。

| 来源 | 它提供的保证 | 成立所需条件 | 本项目尚未验证的部分 |
| --- | --- | --- | --- |
| `[文献]` R12 confseq | time-uniform、非参、非渐近的 confidence sequence 与停止规则 | `E_t` 为合法非负上鞅/e-process；信息流忠实反映实际使用数据；无未来信息；`α` 与多重性正确分配 | 未编译/未移植（C++ 扩展 `boundaries.cpp` / `uniform_boundaries.h`）；未与作者前缀逐点数值对比；未验证 `H0` 模型与残差相容；`α` 分配未实现 |
| `[文献]` R10 BOCPD / R11 DSM-bocd | 模型内的时间分段后验；鲁棒变点对照 | 各自的模型、先验与超参假设 | 不提供频率学误报界；未移植；未做前缀逐点对比；鲁棒性条件未验证 |
| `[文献]` Gauss–Markov（T2 语境） | 固定设计下的 BLUE 与协方差 | 固定设计、零均值、`Σ` 已知正定、模型正确、可辨识 | 自适应选域/估运动后不适用；本项目未验证实现 |
| `[文献]` R13 iSAM2/GTSAM | 因子增量更新、删除与重线性化的参考 | 图结构与重线性化条件 | 非线性局部最优 ≠ 全局最优；边缘化摘要不一定可任意撤回；不作 GPU 主线实现 |
| `[文献]` R20 REN | 严格稳定/鲁棒动态模型参考 | 作者实现为 Julia，其结构条件 | 不接入当前 PyTorch 管线；结论不外推 |
| `[文献]` R15 S4 / R16 Mamba | 线性子状态的稳定分析 | 负状态参数与正步长；仅线性子状态 | 门控、投影、残差、空间交换不自动获得同保证；两类 cache 的撤回需另证 |

---

## 9. T1–T6 总表

| 义务 | 一句话定义 | 当前状态 | 主要交付阶段 |
| --- | --- | --- | --- |
| T1 唯一来源与合并/分裂/撤回一致性 | 固定版本下，来源集合代数决定同一目标函数；撤回后从剩余因子重新求解与直接求解一致 | **未证明（有反例）**；新契约无实现、无测试；旧实现已知不满足时间撤回 | P1（主）、P5 |
| T2 固定支持条件下的估计与不确定性 | 固定设计 + 已知正定 `Σ` + 可辨识时，GLS 是最佳线性无偏估计；先验单列、秩亏走可辨识 mask | **未证明/待实现**；经典结论仅按条件引用，本项目未验证 | P1（主）、P2/P3 |
| T3 顺序决策的误撤回控制 | 合法非负上鞅 + `1/α` 阈值给出误撤回控制；多域分配 `α`；等效性用容差区间 | **未证明（有失效方向）**；BOCPD 后验不满足该界，属已知边界 | P3（主） |
| T4 可变时间常数与局部交换的稳定边界 | `d_max<1` 的泄漏 + 非负行随机交换下有界且收缩；门控支路需另加条件 | **未证明/待实现**；公式为方案设想，无证明无测试 | P5（主） |
| T5 局部重建与全量重放等价 | 完整依赖图 + 覆盖闭包的失效集合 ⇒ 拓扑重算 = 全量重放 | **未证明（有反例模式）**；感受野跨界/双 cache/旧缓存均为已知失效点 | P1（语义）、P5（执行） |
| T6 可中断执行与不确定性读出 | 中断点只读已提交快照；deadline 是实测 SLO；统计提前停止继承 T3 条件 | **未证明/未测量**；只有版本一致性可望严格保证，WCET 未做 | P5（主）、P7 |

---

## 10. 最薄弱假设（研究价值排序）

以下按"若不成立，多少义务同时失效"排序。每条给出最便宜的否证实验。

1. **噪声协方差 `Σ` 已知且来源间独立（T2 的承重墙）。** 共享黑电平与读出相关、插值相关、同一帧派生节点都会破坏；`noise_block_id` 字段存在但尚无验收标准；未知相关性的保守界可能松到无法区分机制。**否证实验：** 用同一传感器的多条静态序列估计残差协方差，检查块间相关性是否显著、保守界是否宽到失去判别力。
2. **统计保证与数据驱动选择的交界（T2/T3）。** 第一版全流程（选域、估运动、定权重）都使用当前数据，固定设计结论与可选停止合法性同时失效；这是最容易在实现中静默违反的一条。**否证实验：** 对同一决策规则，比较"选择规则预先固定/独立校准"与"用当前数据选择"两条支路的经验误撤回率与方差覆盖。
3. **`SourceKey` 唯一性 ≠ 统计独立（T1）。** ID 去重只解决重复登记，不解决相关噪声；数值相同的两次真实曝光仍各计一次，这可能高估独立信息。**否证实验：** 构造共享噪声的重复曝光，检查观测数增长时方差是否按独立假设缩小。
4. **依赖图完整性（T5）。** 卷积感受野、递归 cache、融合权重预测器的闭包是工程声明；没有自动 oracle 检查图是否完整，因此"局部=全量"可能因假阴性通过。**否证实验：** 对同一撤回，分别做局部重算与全量重放，比较逐节点差异；在图中故意删除一个父项，验证测试会失败。
5. **32 帧精确窗口之外的可声明性（§1.5）。** "摘要的不可撤回性"没有形式定义；当前只能声明"历史损失"。长序列上的错误历史残留率因此可能无法判定。**否证实验：** 在超过 32 唯一帧的序列上撤回旧来源，检查结果是否被正确标注为"非精确"、污染摘要是否被作废。
6. **对应关系误差模型（T1/T2 的输入）。** P2 的运动模块尚未适配/训练/校准；`∇fᵀΣ_flow∇f` 只是一阶近似；对齐误差如何进入 `Σ` 尚未定义。**否证实验：** 在已知几何的合成序列上比较实际流误差与声明的风险区间覆盖率。
7. **期限保证分层（T6）。** 只有版本一致性可望严格保证；SLO 与 WCET 缺口明显，容易被"p95 很快"掩盖。**否证实验：** 在真实负载下测端到端超期比例与最坏值，检查是否有任何硬实时声明被写出。
8. **先验来源审计（T2/T3）。** `prior_id` 只是标识；若先验来自同一数据/同一场景，独立性需要证明。**否证实验：** 用与估计数据同源的先验重复 T2 覆盖检查，观察覆盖率是否低于声明。

---

## 11. 未决问题与变更记录

**未决问题（留待 P0/P1 评审与后续阶段裁决）。**

1. `exact_replay_range` 的表示粒度：按 `(episode_id, camera_id)` 的帧身份集合，还是统一时间轴区间；跨相机是否允许共享边界。
2. T3 的 `H0` 具体模型形式（残差分布、参数来自校准集还是在线估计），以及"无信息"的显式判定规则。
3. `context_source_ids` 参与内容通路时，"结果不依赖该源"的声明是否要求把上下文也纳入 `counterfactual_replay`（本文按 §1.1 规则 5 要求覆盖）。
4. `uncertainty_kind` 的取值集合与标签字典，需与 `protocol_v1.json` 一起冻结。
5. 任务读出期限（T6）与 P7/P8 闭环协议中 replan/动作中断的边界，需在 P8 前单独成文。

**变更记录。**

| 版本 | 日期 | 内容 |
| --- | --- | --- |
| v1.0-draft | 2026-10-14 | P0 首版：T1–T6 定义、假设、未证明命题、反例与代码/测试映射；六项契约不变量；外部保证适用边界；薄弱假设清单 |

**本文未做的事（自检边界）。** 未编写任何证明；未修改任何源码/测试/配置；未运行训练或数值实验；未验证引用的文献结论；未声称任何命题成立。代码/测试映射中的"待实现"项均以对应阶段报告为准。

**并发写作说明。** 本文写作期间，同一工作树中出现了其他工作流的 P0 脚手架（`src/owac/frameworks/n01_full/`、`configs/encoders/n01_full/`、`tests/encoders/n01_full/` 等）。本文未读取、未验证、也未把这些文件当作验收证据；映射状态仍按方案的计划阶段标注，以 P0 独立验收为准。

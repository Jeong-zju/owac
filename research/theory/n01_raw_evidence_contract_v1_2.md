# N01 v1.2 RAW 证据契约：理论命题到具体算子

> **2026-10-17 实现更新（代码补全，未训练）**：以下按10-15实现写成的行号、`SpatialSupport.observe` 与旧干预限制保留为历史。当前实际算子以本节及 [代码补全记录](../experiments/n01_v1_2/P3_code_completion.md) 为准；旧数值不迁移到新模型。
>
> ②【拟新增规则】当前入口 `N01V12Model.step → prepare_frame → DomainWindowMotion → WindowGLSSupport.replay_window → conservative_source_mask → DBSR → bridge`。GLS实际求解局部常数/仿射法方程与有限多尺度候选；域方差是显式声明，未拟合的量仍标proxy。未知相关性使用Cauchy上界，不等同于已知完整协方差的最优GLS。
>
> ②【拟新增规则】G1完整来源删除由 `source_replay.delete_sources` 执行：先限定原32帧窗口，再删整源，从剩余观测和固定初值重建同域motion、声明噪声、支持、参考、内容、权重及读出；不补更老帧。外部查询时钟、episode及生命周期到达计数属于请求/簿记上下文，不作为独立证据。`fixed_decision_readout` 保留motion/支持，只重算直接内容，单独命名。卷积之前保守屏蔽双线性反向tap，避免退出源经感受野进入特征；motion网络也先清零无效测量。完整相机checkpoint保持同一融合算子和梯度，诊断仅保留固定通道均值，未删掉实际计算的来源或通道。
>
> ③【未证明命题】G2的固定权重偏差/方差分解及自适应选择偏差限制不变；动作监督并不消除统计选择偏差。G3当前是在当前参考图上对有限合法窗口重放的BOCPD模型后验，有因果读出但不是可预测e-process或长期误报保证。单帧异常先暂缓；旧源撤回需要连续新证据及源特定后验质量。`n01_v12_calibrate` 可在真实标注的calibration/validation上比较配置，未执行就不能称已校准。
>
> ④【实测结果】本次仅检查算子与真实网络前向/反向，不执行优化器更新。未训练自然对应、完整编码器或新后端；不从这些检查推导RAW收益、机制收益或闭环能力。


状态：P3 理论说明（对应 `docs/N01_full_development_plan_2026-10-15_v1.2.md` §6）。写作日期 2026-10-15。
适用分支 `lzh/n01-evidence-integration`。本文件**不修改**任何实现，只把 v1.2 的命题逐条落到实际算子、张量与配置字段上，并如实标出哪些命题没有证明。

## 0. 标注约定

全文每个论断前标注四类之一，不允许混合：

| 标记 | 含义 | 处理方式 |
| --- | --- | --- |
| ①【已有原理】 | 标准结论 | 给出处；不声称是本项目定理 |
| ②【拟新增规则】 | 本项目设计选择和工程干预 | 说明它改变哪个算子；不当作贡献已成立 |
| ③【未证明命题】 | 当前既无证明也无实测支持的断言 | 明确列为未证明，不包装成保证 |
| ④【实测结果】 | 已实际运行并留下记录的观测 | 给出文件名与数值；比值必须给出分母来源 |

本文件不设「通过线」，不引入综合评分、证据健康度或「理论 PASS」系统；不新增任何无来源指标。研究假设未获支持时交付负结果即可，不以自造定理替代任务证据（plan §6.3，`docs/N01_full_development_plan_2026-10-15_v1.2.md`:169）。

---

## 1. 信息边界：同源 `Y = ISP_fixed(X)`

### 1.1 基础命题

①【已有原理】取同一次合法 RAW 测量为 `X`，其同源 RGB 为确定性映射 `Y = T(X)`（本项目取作者冻结 ISP，见 §1.2 的算子）。因为 `Y` 是 `X` 的函数、`σ(Y) ⊆ σ(X)`：

- 只读 `Y` 的决策规则也必然只读 `X`，故按风险集合嵌套有 `inf_{rules over σ(X)} risk ≤ inf_{rules over σ(Y)} risk`；等价说法是「任何只依赖 Y 的决策在信息层面可由拥有 X 的一方复合 T 来实现」（plan §6.1，:151）。
- 信息量口径：确定性映射满足 `H(Y|X) = 0`，因此在有限熵（有限精度）口径下 `I(X;Y) = H(Y)`，随之有题目要求的 `I(X;Y) ≤ H(Y)`；被该映射丢弃的自由度是 `H(X|Y)`。需要强调的是：两个平台的测量都是有限精度存储（LIBERO `uint8`、RoboTwin `float32`），才使这一离散口径成立；连续无噪声口径下此式退化，不能直接引用（见 §1.3 的 `cache_dtype`）。
- 等号（即 `Y` 对任务充分）时不存在 RAW 优势；可逆且保留任务信息的 ISP 不保证出现优势（plan :151）。因此**信息层面允许** ≠ **本方案能取到**：③【未证明命题】受限 N01 架构与有限训练不保证达到该上界，也不要求任意正的互信息增量都过线。

### 1.2 本路线中真正不可恢复的一步

①【已有原理】+ ④【实测结果】函数 `isp_provenance(platform)`（`src/owac/raw/n01_paired_views.py`:96-141）逐项登记了冻结 ISP 的身份与损失：

| 字段 | LIBERO | RoboTwin |
| --- | --- | --- |
| `cache_dtype` | `"uint8"` | `"float32"` |
| `quantisation` | `"uint8 cache stores rint(raw10*255/1023): the ISP input is quantised to 1/255 before any P3 model sees it, so ~2 bits of the 10-bit measurement are already lost"` | `"float32 cache keeps raw10/1023 at full precision"` |
| `raw_unit` | `"linear_pseudo_raw_view_over_255"` | `"linear_pseudo_raw_float32"` |
| `not_a_cfa` | `True` | `True` |

写入这个量化的算子是 `raw10_to_view8`（`n01_paired_views.py`:144-147，`np.rint(raw10 * 255/1023).astype(np.uint8)`）；torch 侧对应 `raw_unit_to_srgb`（:174-192）与 NumPy oracle `raw_unit_to_srgb_u8`（:150-161），两者由测试断言一致，防止训练路径与 oracle 漂移。`N01V12Config.data_kind` 在 LIBERO 上返回 `"libero_dequantized"`（`src/owac/frameworks/n01_full/model.py`:97-106），即**名称层面已承认这是去量化后的视图**，不是原始传感器计数。

结论必须写死：LIBERO 的 1/255 量化损失**不可恢复**——它是缓存属性，任何 P3 模型都在它之后才看到数据（`n01_paired_views.py`:30-35）；不得靠插值、去噪或超分辨率「修回」这部分信息，只能如实报告。④【实测结果】该视图在真实数据上的数值域为 float32 `[0.0, 0.4274]`，是暗光视图（`research/experiments/n01_v1_2/RUN_STATE.md`:95），这决定了「弱信号」不是修辞而是当前输入的实际状态。

### 1.3 对应的主链干预，以及它不能替代实验

②【拟新增规则】v1.2 把上述边界做成可操作的干预，而不是口号：

- **输入信息限制干预**：`raw_base` / `raw_n01` 只读 `rawvla_rgb3`，`rgb_base` / `rgb_n01` 只读**同源**经冻结 ISP 得到的 `rendered_rgb3`；两者由 `paired_views()`（`n01_paired_views.py`:223-257）在同一张量元素上成对产生，四臂共享源帧、相机、采集时间、历史长度和动作标签（plan §3.2，:64）。RAW 臂不生成 RGB 中间图像、不使用 RGB 教师（`model.py`:19-29）。
- **读出信息限制干预**：RAW 保留细节但池化后全部丢失，不能算「RAW 信息已被利用」（plan §4.4，:110），所以读出侧必须检查窄边/小目标是否仍影响动作。

③【未证明命题】这些干预只是把信息差异**分离出来**，不能凭理论宣称 RAW 必胜：`I(X;Y) ≤ H(Y)` 是上界关系，不是收益预测；受限模型、有限训练与算力都会改变结果（plan §6.1，:153）。因此本文件不使用任何形式的「理论证明 RAW 更优」表述；主链干预的价值在于让实验可归因，不在于替代实验。

---

## 2. G1 删除与重算

### 2.1 实际依赖图（从真实数据流写出）

解析链（删除检查走这条）：

```text
MeasurementFrame{values,validity,sample_xy,episode_ids,timestamp}
  → MeasurementLedger.observe            ledger.py:228
  → LedgerFrame{values,validity,checksum} + 四态 validity code
        VALID/MISSING/BAD/SATURATED      ledger.py:57-60；contracts.py:51
  → NoiseModel.{variance_of,covariance_for}     statistics.py:250 / :259      # Σ
  → VersionedDomain{evidence_source_ids, context_source_ids}   domains.py:130-180
  → fit_structure(domain, ledger, correspondence, noise_model) statistics.py:439
  → StructureReadout（按 ReadoutGrid 逐查询点）  replay.py:196
```

神经链（支持改变内容走这条）：

```text
窗口 [B,T,C,H,W]（newest-first，T≤32）
  → DbsrFeatureAdapter._window            adapters.py:930
  → DbsrFeatureAdapter._resolve_motion    adapters.py:862   → offsets_xy / valid / risk_kind
  → SpatialSupport.observe                support.py:647
        region_residual → TruncatedBOCPD.update → decide → expand_to_cells
  → effective_mask[:,1:] &= source_usable  adapters.py:595-598
  → DbsrFeatureAdapter.encode_window      adapters.py:849
  → MaskedDbsrFusion.forward              baselines/burst_features.py:91   → weights
  → _masked_pool → CameraTokens           adapters.py:602 / :942
  → readout → VisualTokenPostprocessor    adapters.py:702 / :439
  → π0.5 语言/动作后端
```

两条链的依赖顺序可以逐项核对：帧 → 来源记录 → 解析状态/噪声 → correspondence → 支持决定 → 融合权重 → token → bridge → 动作；`effective_mask` 在 `encode_window` 之前被改写，是「支持确实改变内容」的**结构性**证据（`adapters.py`:388-400 的 docstring 与 :595-598 的实现必须成对阅读）。

### 2.2 边退出 ≠ 整源删除

- ②【拟新增规则】**边退出**：`SpatialSupport.decide`（`support.py`:513-544）在 `change_logodds <= retract_logodds` 时把 `(source, region)` 置为 `SUPPORT_RETRACTED`（:537）。这是结构支持边的退出，不是源被删除——`support.py`:41-43 与 :530 的注释明确写 `RETRACTED is an edge exit, not a source deletion`。被撤回的源仍可贡献「曾发生变化」的**上下文**：解析链里成员角色被保留（`VersionedDomain.context_source_ids`，`domains.py`:162；`retract_sources` 按原角色重分 `remaining`，`domains.py`:387-393），BOCPD 状态按区域、按到达时间维护且**故意不随窗口滑动**（`SpatialSupport.shift`，`support.py`:607-614），所以「刚发生过变化」这一事实在窗口内持续可见。
- 因此 ③【未证明命题】/硬性表述边界：**只要保留了上下文角色，就不能声称最终表征与该源完全无关**。可以说的是「该源自身的测量不再进入内容通路的融合」（内容链上它的 cell 被 `expand_to_cells` 判为不可用、随即从掩码中清除，`support.py`:546-579 + `adapters.py`:596-598）。
- ②【拟新增规则】**整源删除**：只有 `MeasurementLedger.without(ids)`（`ledger.py`:524-559）把被删样本**在物理上涂掉**——`values[index]=0`、`validity[index]=MISSING`，且发生在任何下游决策回调之前（代码注释 :539-540 说明了理由：仅用 `has_source` 过滤会让 `frames()`/`get_frame()` 成为泄漏通道）。但 `without()` 只产出一个视图，**不负责重建**；全链路重建由 `counterfactual_replay` 完成（§2.3）。
- ③【未证明命题】`fixed_decision_replay` 用的 `with_content`（`ledger.py`:561-569）只是挂一个命名、带版本的 overlay，原始记录不动——它换的是内容，不是集合成员。把 overlay 与 `without` 当成同一操作会在报告里制造假结论。

### 2.3 两个回放接口不能混报

两个接口同时从第一版存在，就是为了让「撤回成功」不能停留在一个含糊指标上（`src/owac/frameworks/n01_full/config.py`:242-250）。

| 接口 | 固定什么 | 衡量什么 | 明确不声称什么（代码原文） |
| --- | --- | --- | --- |
| `fixed_decision_replay`（`replay.py`:404-477） | 决策：域成员、基、chart、噪声版本、mapping，由调用方以 `frozen_domains` 传入；`conditional_on = frozen_decision_deps`（:460） | **直接路径**：内容替换后读出的变化（`max_abs_diff` 对比 `readout means, variances and validity`，:456） | `not_claimed: ["whole-result independence (that needs counterfactual_replay)"]`（:466） |
| `counterfactual_replay`（`replay.py`:498-662） | 不固定：`recomputed = [ledger view, domain membership and versions, per-phase designs and noise sub-blocks, readout]`（:630-635） | **完整删源**：`full.retain_frames(window_frames).without(retracted)`（:538）后在重算域/拟合/读出上重跑 | 窗口外源：`claim = "rebuild_from_legal_boundary"`、`not_claimed: ["exact equivalence for sources outside the exact replay window"]`（:636-647） |

②【拟新增规则】报告纪律：`fixed_decision_replay` 的差值只说明直接路径，`counterfactual_replay` 才说明完整删源；两者**不能混报**，也不能用前者的「unchanged=True」代替后者。可选的重算回调是显式的：`correspondence_builder` 与 `noise_model_builder`（:539-546），未提供时 manifest 里 `correspondence_recomputed`/`noise_recomputed` 为 `False`，说明 correspondence 与噪声**没有被重算**，这一状态必须随结论一起报。

③【**已补｜2026-10-15 更新**】两个解析链回放接口（`replay.py`:404-477 与 :498-662）之外，
**神经内容链的整源重算接口现已实现**：`evaluation/n01_v12.py::counterfactual_content_replay`
让被删槽位**根本不调用 `step`**，于是该来源不在唯一帧缓冲、不在 BOCPD 到达序列、不参与区域支持决定、
也不可能成为参考，窗口只变短而不从窗外补帧；`intervene --mode {delete,mask}` 把两者分开命名，
receipt 用不同措辞声明 `recompute_scope`，**不得混报**。

据此可以把神经链的证据强度提升一级，但仍要精确：
（a）结构性接线——`source_usable` 在 `encode_window` 之前被 AND 进掩码（`adapters.py`），
被撤回的 cell 不可能进入融合；
（b）行为性检查——同一份权重下持续偏移窗口的 `raw_n01` 与 `raw_base` token 不同、静止历史下两臂一致。
   **【2026-10-16 降级：这条证据比原先写的弱】** 把比较符单独翻回反向（阈值不动、配置仍合法）后重跑，
   **arm-vs-arm 的 token 断言并没有失败**：融合权重几乎全部集中在参考槽位上，
   因此「是否把历史算进去」对 token 的影响很小——`token 不同/相同` 对「历史是否被使用」是个**钝器**。
   同样地，`test_support_leaves_stable_history_alone` 也**抓不到**这个反向：新增的恢复分支
   让它在静止序列上自我抵消（赔率低 → 反向判为「已变更」→ 撤回 → 随即被恢复）。
   **真正抓到反向的是 `test_support.py::test_a_long_stationary_history_stays_usable`**（翻转比较符即失败）。
   所以 (b) 只能作为**弱旁证**；判断「支持是否起作用」应当看**支持决定本身**
   （四态码与可用比例），而不是两臂 token 是否相等。
（`tests/encoders/n01_full/test_model_v1_2.py`）；
（c）**新的删除检查**（`tests/integration/test_n01_v12_deletion.py`，7 项）：
支持接受历史时删除一个有权重的来源确实改变 token；独立随机帧下支持撤回整段历史、
所有来源权重为 0，此时删除**正确地**不改变内容；删除与掩码的**可测**差异在账本
（`arrival_index`、`unique_frames_lifetime`、缓冲占用）而非内容——因为 `MaskedDbsrFusion`
对无效槽位清零并对有效来源重新 softmax，被掩码的帧本来就不带权重。
因此本文件**不声称**"删除必然改变内容"，那是一个本架构不承诺的性质。
仍未做：真实 checkpoint 上的 GPU 实测，需与开发闭环同样的训练暂停。

### 2.4 32 帧有界精确回放与 `HistoryLoss`

- ②【拟新增规则】窗口是硬上限：`ReplayConfig.exact_window_unique_frames = 32`（`config.py`:254），`N01V12Config.window_frames = 32`（`model.py`:116），支持状态容量按 `max_history - 1` 分配（`adapters.py`:578-584，`SpatialSupport.initial_state` 的 `history_sources` 语义见 `support.py`:583-590）。
- `_exact_window`（`replay.py`:238-256）按 `(episode, camera)` 取 `frames[-exact_window:]`，其余逐个登记为 `HistoryLoss(frame=..., reason="beyond_exact_window_summary_only")`（:253）。
- 窗口内删除后**位置标无效、不从窗口外补更老帧**：`retain_frames` 先把窗口外的帧从视图里移除（`full_replay` :297），`without` 只把被删索引置 `0`/`MISSING`（`ledger.py`:550-552）。因此「偷偷补更老帧」在实现上不可达，而不是靠约定。
- 更早历史的丢失**必须显式报告**：`FullReplayResult.history_loss`（`replay.py`:230）、`snapshot.cost["history_loss_frames"]`（:342）、manifest `history_loss`（:368）；`counterfactual_replay` 另有 `beyond_exact_range` / `beyond_exact_sources` / `invalidated_summary`（:636-639）与 `history_policy = "discard expired raw and summaries; rebuild from retained boundary"`（:640）。
- ①【已有原理】删除一致性本身对 RGB 同样成立（plan §1，:11），②【拟新增规则】本项目在**完整神经算子**上说明适用粒度和感受野：区域级判断 + 保守感受野展开，而不是假称任意像素级独立（`support.py`:5-8 与 `expand_to_cells` :546-579；plan §4.3，:106）。③【未证明命题】这只保证「证据通路可信」。

### 2.5 容差与跨版本比较

②【拟新增规则】数值比较用合理容差，不要求跨版本 bitwise：

- `fixed_decision_replay(tolerance=1e-10)`（`replay.py`:414），判定 `unchanged = diff <= tolerance`（:459）。
- `counterfactual_replay(reference_tolerance=(1e-10, 1e-8))`（:511），按 `atol + rtol*|reference|` 比较（:617）。
- 比较本身对有效性敏感：`StructureReadout.max_abs_diff`（:179-190）在有效性模式不一致时直接返回 `inf`；标签集合不同也返回 `inf`。这是刻意的——「非有限 vs 有限」的差异不能被平均掉。

①【已有原理】删除检查用于保证证据通路可信，**不作为独立任务性能贡献**（plan §4.5，:118）。RAW 不因此获得任何专属功劳。

---

## 3. G2 保留历史的条件性收益

### 3.1 分解与所需假设

②【拟新增规则】记目标为当前真值 `z_t`，历史测量 `y_i = z_i + ε_i`，权重 `w` 满足 `Σ_i w_i = 1`，估计 `ẑ = Σ_i w_i y_i`。则

```text
ẑ − z_t = Σ_i w_i (z_i − z_t) + Σ_i w_i ε_i
MSE     = (Σ_i w_i (z_i − z_t))² + wᵀΣw
```

即**历史偏差²（陈旧）** + **加权噪声方差**（plan §6.2，:160）。

该恒等式成立所需的假设必须逐条列出，缺一条就不能引用：

1. `w` 相对噪声是**固定设计**：`w` 与 `ε` 的叉积项 `2·(Σ_i w_i (z_i − z_t))·E[Σ_i w_i ε_i]` 必须为零。最干净的形式是条件在 `(z, w)` 上的 MSE，且 `E[ε | w] = 0`；
2. `E[ε_i] = 0`；
3. `Cov(ε) = Σ` 有限且**已知**（否则只能给 proxy，见 §3.3）；
4. 偏差项以场景的 `z_i` 为条件理解；`z_i` 随机时只在条件意义上成立。

第二项 `wᵀΣw` 说明「多帧一定降噪」是错的：当 `Σ` 含跨帧相关（对应关系不确定、插值共享像素、光照共同漂移）时，权重增大带来的方差下降远低于独立假设的预期，第一项还被运动/结构不相容放大。这一项在代码里对应 `NoiseModel.covariance_for(keys, dtype)`（`statistics.py`:259-276）——`dense` 取联合协方差的相关子块，`diagonal`/`unknown_correlation` 只给对角；以及 `NoiseModel.__post_init__` 对 `dense` 要求正定、并明确拒绝奇异联合协方差（`statistics.py`:193-205，理由：奇异协方差编码的是重复观测，属于原始因子问题）。

③【未证明命题】把 `w` 换成支持决定 `source_usable` 后，上式**不自动成立**：`source_usable` 是数据相关的 0/1 权重，且 `MaskedDbsrFusion` 的权重来自 `logits.softmax(1)`（`burst_features.py`:113），既非和为 1 的凸组合（`masked_fill(~valid, -inf)` 后仍是 softmax 归一化，故在有效源上和为 1，但参考帧在缺失时退化为有效帧共识，`burst_features.py`:96-99），也不是本文写下 `w` 时的固定设计。因此 §3.1 只能作为**设计直觉与诊断分解**，不能作为该网络的可证收益界。

### 3.2 自适应权重的数据选择偏差

③【未证明命题】当 `w` 是用同一批数据选出来的（本项目确实如此：`hazard`、`accept_logodds`、`retract_logodds`、`robust_*` 都是开发集尺度，见 §4.4），假设 1 失效，叉积项不为零，`MSE = 偏差² + wᵀΣw` 会**低估**实际期望误差。**`detach()` 不消除这种统计选择偏差**：`detach` 只切断反向传播的梯度，不切断「用同一批数据挑权重」这一事实；实现里 `observation = (masked.sum(1) / counts).detach()`（`support.py`:696）切断的正是梯度通路，它使支持决定不参与训练、使四臂保持等参数（`model.py`:267-277 的 `support_parameters: 0`），但对选择偏差毫无作用。

②【拟新增规则】因此报告纪律是：任何「支持带来了降噪/降偏差」的说法，必须区分（a）在设计集上按固定 `w` 计算的诊断量、（b）在独立数据上按选定 `w` 评估的结果。两者不得互相替代。

### 3.3 未知相关噪声下只报 proxy 或保守界

①【已有原理】`statistics.py`:18-27 明确把三种不确定度标签分开，并把「混合它们」记为已记录的失效模式：

- `covariance` — 在声明的 `Σ` 下的精确 GLS / 后验协方差（求解按 `L = cholesky(Σ)`、`A = solve_triangular(L, Φ)`、`theta = lstsq(A, b)`，见 `statistics.py`:6-16）；
- `proxy` — 相关未知时的保守固定权重界 `(Σ_i |a_i| σ_i)²`，以对角 PSD 包络存储，查询界由实际原始源权重算出，**绝不**用可能在带符号查询下相消的外积；
- `none` — 秩亏且无先验：坐标标为不可识别，不返回协方差（用 `identifiable_mask` 而不是加小 ridge，`statistics.py`:14-16）。

②【拟新增规则】运动/对应侧沿用同一纪律：`RiskCalibrationTable`（`src/owac/backends/raw_correspondence.py`:487-498）是独立标定表的协议，`map_proxy(flow_risk, visibility) → (calibrated_risk, coverage)`；在此之前 `RawCorrespondenceMap.risk_kind` 保持 `"proxy"`（:526-529, :545），只有标定表提供后才写 `"calibrated:<table version>"`。

③【未证明命题】没有可识别的相关噪声时，只能报 proxy 或保守界；不得把 `(Σ|a_i|σ_i)²` 当成精确方差，也不得把 `risk_kind="proxy"` 的分数写成误报概率。

④【实测结果】当前真实链路上 correspondence 的 `risk_kind` 仍为未标定状态（`research/experiments/n01_v1_2/P3_reuse_map.md`:16 记录「风险为未校准 proxy（`risk_kind` 字段已有），P3 必须保留该标注」）。任何依赖风险值的结论在标定完成前都只能是 proxy 级。

---

## 4. G3 持续变化的顺序证据

### 4.1 当前实现到底是什么

②【拟新增规则】唯一在线的顺序判据是 `SpatialSupport.observe` 内的 `TruncatedBOCPD`（`support.py`:162-289）：

- 输入是**每个区域一个标量**、按到达时间排列的 innovation：`usable = valid & history_valid.unsqueeze(-1)`，`observation = (masked.sum(1)/counts).detach()`（`support.py`:693-696）——即把窗口内可用历史源的残差做鲁棒聚合，`.detach()` 使其不参与训练；
- `region_residual`（:400-445）把残差除以**参考帧自身**高频（局部 Laplacian RMS）得到的尺度（:481-509，下限 `noise_floor`），并且注释明确写了为什么不用被检验场自身的离散度：否则恒定偏移会被整除掉，正是要检测的东西反而看不见；
- `TruncatedBOCPD.update` 返回 `log_short − log_long`（:285-289）——在保留的 run-length 假设上，「最近发生过 changepoint」与「没有」的两组后验质量之比。文档串 :233-239 直接写明这是**模型后验比值，不是已标定的误报概率**；
- 证据不足时返回 0 而不是无限证据：`threshold = max(2, keep//2)`，`both_present` 不成立时 `odds = 0`（:280-289）；截断保留 `keep_run_lengths=16` 个假设（`SupportConfig`，:94），被丢弃的 run length 在 `run_length_log_probability`（:291-307）里显式为 `-inf`；
- 判据本身是区域级共享的：`change_logodds.unsqueeze(1)` 把同一个区域统计量广播到该区域的全部历史源（:530、:538），所以撤回是「该区域的结构证据被判失效」，不是「某源单独被证伪」；
- 滞回是存在的但不对称：`RETRACTED` 在窗口内**不能被任何规则提升回 `ACCEPTED`**（只有 `decide` :539-543 能把 `DEFERRED` 提升），退出状态只在 `shift` 老化（:607-637）后消失；解析链上对应的恢复算子是 `restore_sources`（`domains.py`:402-439），它要求显式声明 `role` 并拒绝重复成员。这一不对称是设计选择，需要在报告里写清楚，不能把「撤回后没有自动恢复」当成检测失败。

### 4.2 它不是 e-value / confidence sequence

③【未证明命题】（本文件最重要的边界之一）

- `change_logodds` 是**模型后验**，因此**没有**长期误报的频率学保证。`support.py`:24-31 的模块文档已写下这一点：*no e-value, no confidence sequence and no frequency guarantee here*。
- 后验质量比不是检验鞅：把 run-length 上的混合写成「证据」需要构造对每个累积量满足几乎超鞅性的 e-process 或 e-detector；本实现的 generalized-Bayes 有界影响更新（`new_prec = prec + 2ω`、`new_mean = (prec·mean − 2ωv)/new_prec`，:216-229）是**稳健化**，稳健后验不等于合法 e-value。因此不能借 R18/R19 的 time-uniform 结论。
- 只有在残差满足可预测性（对滤波 `F_{t-1}` 无前视）与条件矩条件（`E[innovation|F_{t-1}]=0`、条件方差与声明一致）**并且**使用合法 e-process / confidence sequence 时，才能引用平均误报间隔或 time-uniform 覆盖类保证；本实现**尚未建立**这些条件，也未编译新统计库作为训练前置（plan §6.3，:167）。
- 结论措辞：只能标为「**模型/经验结论**」。`hazard`、`obs_var`、`prior_var`、`robust_*`、
  `large_residual_threshold` 仍然是声明值而非从数据估计的校准量，所以连「模型拟合良好」
  也不能从 `change_logodds` 自身读出。

②【拟新增规则】这条边界**不**允许把整个任务永久卡在「无法证明 PASS」：按 plan §6.3/§7（:169、:177-181），研究假设未获支持可以交付负结果，不设连续硬门槛、不接受自造「定理 PASS」替代任务证据。正确做法是同时交付（a）模型层结论与其假设清单、（b）在真实 calibration/validation 数据上重新选定的尺度、（c）明确写出哪些频率学保证**未被引用**。

### 4.3 尺度是开发集尺度

②【拟新增规则】`SupportConfig`（`support.py`:81-129）声明了全部几何与阈值，`to_dict()`（:131-155）把来源写进 `source` 字段：

```text
"Truncated BOCPD and GaussianUnknownMean transcribed from DSM-bocd
 9d05117bc8c6e971c836ee1df0c49e2d2950862b (MIT); bounded-influence
 generalized-Bayes update and the thresholds/scales are this project's"
```

即：算法结构来自作者实现，**阈值与尺度是本项目的、且是开发集旋钮**（`SupportConfig` 文档串 :84-88：*development-set knobs … not a derived optimum and not a pass threshold*）。这些值随每个 run 的有效配置落盘（`N01V12Config.to_dict()["support"]`，`model.py`:188）。

②/④【拟新增规则】+ 代码事实：至少有两处判据尺度**没有**进入 `SupportConfig`，因此也不在 `to_dict()` 的声明里，重定尺度时必须一并处理，否则「配置里声明了全部尺度」这句话不成立：

- `decide` 的「大残差」门限硬编码为 `1.0`（`support.py`:531 的 `large = residual >= 1.0`，以及 :702-703 用 `residual >= 1.0` 控制 `defer_count`）；同一条硬编码在 `observe` 与 `decide` 两处出现，改动时必须同时改，否则两条路径会给出不一致的 suspect 判定。
- 「最近发生过变化」的分组门限硬编码为 `max(2, keep//2)`（`support.py`:282），它决定 `log_short`/`log_long` 如何划分保留的 run-length 假设；`keep_run_lengths` 是配置项而这条划分规则不是。

### 4.4 实测：尺度偏保守的观测（不是收益）

④【实测结果】LIBERO `raw_n01` 真实 batch 端到端复核（32 帧、双相机、真实 π0.5 权重、`receipts/verify-libero-raw_n01-seed101.json`）报告的支持状态为 `applied_v1_2`（该字符串由 `adapters.py`:679 产生，判据是 `self.support is not None`，即支持已挂在内容通路上），计数为：

| 相机 | accepted | deferred |
| --- | ---: | ---: |
| agentview | 521 | 759 |
| wrist | 267 | 1013 |

这是**观测，不是成绩**：`deferred` 数量级远大于 `accepted`，说明默认尺度在真实暗光数据上偏保守、需要用真实 calibration/validation 数据重定（`RUN_STATE.md`:103；`research/experiments/n01_v1_2/P3_report.md`:122-123）。本轮没有为了图好看而人工压低风险值。

**分母已核实，为 5 × 256 = 1280。** `SupportDecision.counts` 在 `state_codes` 上逐状态求和（`support.py`），而 `state_codes` 的形状是 `[B, T_active, R]`，`T_active` 由 `observe` 的 `active = int(history_valid.shape[1])` 决定，`R = region_grid_shape(H,W)` 的乘积（`region_grid=16`、256×256 时 `R = 256`）。

曾疑心分母应为 `T_active = 31`（31 × 256 = 7936），但实测推翻了这个假设：**真实 RawVLA-Bench 的 32 槽窗口里只有 6 个唯一帧**。抽样 `dataset_from_protocol(..., window=32)[0]` 的 `frame_ids` 为 `[0]*27 + [1, 6, 11, 16, 21]`，且 0..26 槽的 `values` 与槽 0 **逐元素相同**（`torch.equal` 全真），即左填充是**同一个重复的填充帧**，真实帧只有 5 帧（`observation_stride=5`）。适配器的唯一帧去重因此正确判定 6 个唯一帧（1 填充 + 5 真实），最后一次融合窗口 = 1 参考 + 5 历史源 → `T_active = 5` → 分母 5 × 256 = **1280**。两个读数因此是一致的，759 的正确分母是 1280，不是 7936。

**跨样本分布（必须在报告里如实写）**：唯一帧数随样本位置变化很大——LIBERO 在样本 0/50/500/5000/20000/39000 处为 6/32/32/26/15/3，RoboTwin 为 9/32/32/32/32/32。因此 `window=32` 是**容量**（最多 32 个唯一已到达帧），不是"每个样本都喂了 32 个唯一帧"。任何"32 帧证据"的表述都必须同时给出该样本的实际唯一帧数，否则就是把容量说成实测量。

【代码事实｜不属于上述四类】与此配套的一条实现事实（供核对时参考，不构成缺陷断言）：模块常量 `P3_SUPPORT_ENABLED = False`（`adapters.py`:90-96）及其文档串仍是 P2 时期措辞（「step always reports `p3_support_applied == False`」），而 `set_support` 之后 `step` 实际按 `p3_support_applied = bool(support_decision is not None)`（:678）与 `p3_support_status`（:679）报告。判断某臂是否启用支持应以 `N01V12Config.spec.support_enabled`（`model.py`:80-85、:231-233）与诊断字段 `p3_support_status` 为准，而不是这个常量。

---

## 5. 创新边界

①【已有原理】以下都**已经存在**，不是本项目的贡献：

- 成熟多帧 RAW 融合与对齐：`MaskedDbsrFusion` 保留上游 `WeightedSum` 模块，只新增 mask 适配；`masked_fill(~valid, -inf).softmax(1)` 与「参考帧缺失时退化为有效帧共识」是该外层的既有融合语义（`src/owac/baselines/burst_features.py`:73-114、:96-99；来源与许可证见该文件头 :1-8）。对应 R01（DBSR）、R06（Handheld MFSR）。
- 标准变化检测：BOCPD 及其鲁棒变体（R16、R17；`support.py`:9-22 记录了转写来源与许可证）。
- 标准检测 + reset：`reset` 语义已存在于编码器契约（`adapters.py`:724）与缓冲区状态（`ledger.py`:481 的 `MeasurementLedger.reset`）；原 B1 标准 reset **是补充诊断，不默认作为第五训练臂**（plan §5.1，:133）。
- 特征注意力**不是**噪声后验：`burst_features.py`:5 明写 `Feature attention is not a noise posterior.`，这一条也是 ③ 的边界——不能从 `weights` 反推物理精度或概率。

②【拟新增规则】本项目的待检验增量**只有两条**：

1. **来源支持修订怎样真正改变内容与行动**：支持通过 `set_support` → `effective_mask[:,1:] &= source_usable` → `encode_window` → `MaskedDbsrFusion` 的 softmax 归一化与参考帧构造，改变进入融合的有效源集合、参考帧、归一化和最终桥接特征（`adapters.py`:388-400、:595-598；`burst_features.py`:91-114）。可判定的自检是「同一份权重下，持续偏移窗口上 `raw_n01` 与 `raw_base` 的 token 不同；静止历史上两臂一致」（`tests/encoders/n01_full/test_model_v1_2.py`:95-131 的两个用例）。
2. **归因矩阵**：同输入下的机制收益（`raw_n01` vs `raw_base`）与同机制下的输入差异（`raw_n01` vs `rgb_n01`），四臂共享三通道 stem、DBSR 宽度/深度、历史长度、token 网格与同一 bridge，支持零可学习参数（`model.py`:19-25、:267-277）。四格结果分别保留，不合成一个数（plan §5.3，:143）。

③【未证明命题】这两条都还是**待检验假设**，不是已成立的贡献；不得表述为「首次 RAW 进 VLA」「首次动作监督」「首次多帧 RAW 融合」「首次可撤回状态」（plan §2.2，:41）。

---

## 6. 未证明命题清单

以下全部为 ③【未证明命题】，交付时不得表述为已证明或已支持：

1. **仅 CV 证据的机制不等于具身贡献**：代理任务/探针结果不能替代闭环任务证据（plan §2.2，:39；仓库规则亦要求具身集成候选说明新规则如何改变物理决策）。支持改变 token 只是必要条件。
2. **不承诺无条件任务性能**：不承诺任何平台、任何条件、任何臂上的提升；`I(X;Y) ≤ H(Y)` 不蕴含收益，可逆 ISP 不保证优势（§1.1）。不设必须提升某百分比的通过线（plan §2.3，:45）。
3. **不承诺有限帧必检**：32 帧窗口内的持续变化不被保证检出；`change_logodds` 无频率学控制（§4.2）；窗口外的源只能 `rebuild_from_legal_boundary`，不声称精确等价（§2.4）。
4. **不声称在自制数据集上验证过自然 RAW，也不声称跨相机迁移**：两个平台都**没有原生 CFA**（`n01_paired_views.py`:136 `not_a_cfa: True`）。
4e. **【实测性质｜2026-10-16】** 在修正判据方向后重录的真实闭环里，同一个 suite 的两个照明极端表现出方向相反的可用性：

   | episode 照明 | 支持可用单元比例（44 次策略查询） |
   | --- | --- |
   | ExtremeOver (ev +7.72) | **0.00 – 0.00** |
   | ExtremeLow (ev −5.49) | **0.09 – 1.00** |

   即**亮且机器人在动时历史被全部撤回，接近全黑时历史反而大量保留**。
   与已测事实一致的解释是：本文件选用的判据度量**帧间差异**，而 ExtremeLow 下 RAW 视图均值仅约
   1/255，相邻帧几乎相同，于是赔率低、被判为"历史稳定"而保留。**该解释是假设，未做证伪实验**
   （如暗环境下注入已知人工运动、观察撤回率是否随之上升）。

   **【假设已检验｜2026-10-16，方向成立但暴露一个照明相关的不对称】** 在该假设的证伪实验里
   （`evaluation/n01_v12_null.py::motion_response`，在这两个 episode 的**真实录制帧**上
   把历史相对参考人为平移固定像素数）：

   | 注入位移 | ExtremeOver（亮，均值≈0.70） | ExtremeLow（暗，均值≈0.004） |
   | --- | --- | --- |
   | 0 | accepted 1.00 | accepted 1.00 |
   | 2 | **retracted 0.63** | **deferred 0.55** / retracted 0.10 |
   | 8 | retracted 0.76 | deferred 0.53 / retracted 0.17 |
   | 16 | retracted 0.78 | deferred 0.50 / retracted 0.23 |

   两条结论：（一）注入位移使两侧"接受"都从 1.00 塌到 0.18–0.36，**判据确实响应帧间差异**，
   上面的解释不是巧合；（二）**同样的物理位移在亮帧上得 RETRACTED、在暗帧上只得 DEFERRED**
   ——`DEFERRED` 是可恢复的暂缓，`RETRACTED` 是更强的判决，因此
   **判决的严厉程度依赖照明而不只依赖运动**。这使"反向激励"更具体：暗输入不仅更容易被当作稳定，
   即使被当作异常也得到更轻的判决。**本文件不判定这是缺陷还是设计选择**——
   它取决于设计意图是"按帧间差异"还是"按对应性"，属机制设计者的决定。

   **【已追到机制层｜2026-10-16】** 同一实验还测出了链条上每一环的数值：

   | 位移 | ExtremeOver 归一化残差 / 赔率中位数 | ExtremeLow 归一化残差 / 赔率中位数 |
   | --- | --- | --- |
   | 0 | 0.00 / −3.146 | 0.00 / −3.146（**两者完全相同**） |
   | 2 | 1.96 / **+12.61** | 1.08 / **−2.95** |
   | 16 | 3.52 / +235.3 | 1.18 / −2.86 |

   即：照明**不直接进入判据**（位移为 0 时两帧给出完全相同的残差与赔率），
   它通过**归一化残差对位移的响应**进入——同样的 2 像素位移在亮帧给出残差 1.96、暗帧只有 1.08；
   而观测方差 `obs_var = 2e-2` 很小，把这点差异放大成赔率 +12.61 与 −2.95；
   撤回门槛 `retract_logodds = 1.0` **恰好卡在两者之间**，于是同一物理位移得到 RETRACTED 与 DEFERRED。
   因此若要改变这一性质，可指认的抓手是 `obs_var`（放大倍数）或 `retract_logodds`（门槛位置），
   **而不是「撤回率」这类结果量**。本文件不声称哪一个位置是「对」的。
   若假设成立，这是任何"基于帧间差异"判据的**反向激励**：在信息量最低的输入上保留历史、
   在信息量最高的输入上丢弃历史。**本文件据此不断言该判据适合承担"测量支持"，只记录其性质。**

4d. **【更正｜2026-10-15】** 4c 的探针把已知噪声注入 slot 0，而 `region_residual` 既以 slot 0 为参考、
   **又**用它估计噪声尺度（`support.py`:432、:449-454）：分子与分母同时随 σ 缩放，因此
   "两平台都饱和在 ≈1.36"是**探针构造的性质**，不是关于估计量的证据。
   4c 中"由此证明估计量没有平台相关的标定错误"**不成立，予以撤回**。
   仍然成立的部分只有两条：σ=0 时残差精确为 0（统计量是零基的）；
   同一个绝对 σ 在两平台给出不同数值（反映帧自身尺度/结构不同）。
   要判定估计量是否标定正确，需要一个**不把待测量同时放进分子与分母**的检验，本文件不声称已有。

4c. **【零假设检验｜2026-10-15】** 4b 记下 6.8 倍跨平台残差差异后，本轮把检验做了
   （`evaluation/n01_v12_null.py`，CPU-only）：向真实参考帧注入**已知**噪声水平。
   实测 σ=0 时两平台残差**精确为 0**、且两侧都在 **≈1.36 饱和**
   （LIBERO 1.37–1.39、RoboTwin 1.36），而同一个绝对 σ 给出完全不同的数
   （σ=0.005：LIBERO 0.27 vs RoboTwin 1.01）。结论是**估计量没有平台相关的标定错误**，
   差异来自成像与内容（RoboTwin float32 低噪声近静止，LIBERO uint8 反量化且真在动）。
   因此 `large_residual_threshold = 1.0` 仍然**不该**被分位数替换——它是"残差超过帧自身局部细节尺度"
   这一**内容相对**陈述，两个平台上都有意义，只是落在分布的不同位置。
   可检验后果：**任何用到它的结果必须同时报告它的触发率**，否则读者无法判断该阈值是否被触发过。

4b. **【实测结果｜2026-10-15，修正本条先前的措辞】§4.3 说各尺度"应当用真实 calibration
   数据选定"，此前只有声明，现已测量**（`owac.evaluation.n01_v12 support-scales`，CPU-only，
   不占用训练 GPU）。结论是 `change_logodds` 的分布**跨平台不可迁移**：
   LIBERO 中位数 **+32.5**、RoboTwin 中位数 **−3.11**，而判定式是 `logodds <= -1.0`，
   于是同一阈值让 RoboTwin 撤回 **56.6%** 区域、LIBERO 只撤回 **9.7%**。
   该统计量还是**无界**的（实测最大 1.1e6），所以"同一个数"在两个平台上并非同一个操作点。
   把阈值取为校准分布的分位数（q10 / q75）后，两平台撤回率都落到 ≈10%
   （RoboTwin accepted 4.6% → **51.0%**）。这修正了本文件"尺度已在配置里声明"的过度自信：
   **声明过，但当时并未按数据选定**。仍未重定：`hazard`、`obs_var`、`robust_omega`、
   `robust_scale`、`large_residual_threshold`；也仍未建立"分位数规则本身是最优操作点"的论证——
   它只是**有记录、可复核**的操作点。

5. **两平台的输入性质不同且都不是原生传感器数据**：LIBERO 是逆处理 + `uint8` 量化缓存（`cache_dtype="uint8"`，`data_kind="libero_dequantized"`）；RoboTwin 是 HDR 派生的 float32 视图（`hdr_derived`）。不得据此声称恢复了真实传感器的原始动态范围（plan §3.1，:51）。
6. **§3.1 的 MSE 分解不自动覆盖神经实现**：`w` 的固定设计假设在 `source_usable` 与 softmax 权重上不成立（§3.1、§3.2）。
7. **§3.2 的选择偏差未被消除**：`detach()` 不切断选择（§3.2）。
8. **顺序证据的频率学保证未建立**：无 e-value、无 confidence sequence、无 time-uniform 覆盖或平均误报间隔结论（§4.2）。
9. **无关成像变化的鲁棒性不蕴含有用性**：把成像干扰与动作相关物理/目标变化混为一谈是失效模式；不得用「更鲁棒」替代「更有用」。
10. **四臂等参数不蕴含机制价值**：支持零参数只保证四臂在等参数、等后端暴露下比较（`model.py`:218-225 的类说明与 :267-277 的 `parameter_report`），不保证机制有效。
11. **单训练种子**：每平台每臂 1 个 seed（默认 101，`model.py`:57、:288-306），不能用于估计跨训练种子方差，报告必须明写（`model.py`:295-300）。

---

## 7. 实测状态（截至本文件写作时，如实记录，不美化）

④【实测结果】

| 项 | 状态 |
| --- | --- |
| 主体 8 个训练 run（`raw_base`/`raw_n01`/`rgb_base`/`rgb_n01` × LIBERO/RoboTwin × seed 101） | **尚未开始**（`RUN_STATE.md`:84；:10「主体训练尚未开始」） |
| 本轮 100 次开发闭环 | **未使用**（`RUN_STATE.md`:85；`P3_report.md`:13「0 次闭环 rollout」） |
| 同步可视化案例 | **未生成**（`RUN_STATE.md`:86；`P3_report.md`:14） |
| 本文件（`research/theory/n01_raw_evidence_contract_v1_2.md`） | 本次产出，替代 `RUN_STATE.md`:120 记录的「未产出」状态 |

已完成的真实复核**只有一项**，即 LIBERO `raw_n01` 的真实 batch 端到端（远程 GPU0、Python 3.10、真实 π0.5 权重，`receipts/verify-libero-raw_n01-seed101.json`）：

| 项 | 实测值 | 出处 |
| --- | --- | --- |
| 窗口/形状 | `effective_window_frames = {"agentview": 32, "wrist": 32}`，字段 `[1,32,3,256,256]` | `RUN_STATE.md`:94；`P3_report.md`:95、:114 |
| RAW 数值域 | float32 `[0.0, 0.4274]`（暗光，uint8 线性 RAW10 视图的 /255） | `RUN_STATE.md`:95 |
| 同源配对 | `identical_source=true`、`rgb_is_not_identity=true`，provenance 指向真实作者模块 | `RUN_STATE.md`:96 |
| action flow matching loss | 0.40667 | `RUN_STATE.md`:97 |
| encoder / bridge 梯度绝对和 | 20680.8 / 14441.3（非零 → 真实更新） | `RUN_STATE.md`:98 |
| 后端梯度 | **0.0 → 冻结成立** | `RUN_STATE.md`:99 |
| 视觉 token | `[1,512,512]`（2 相机 × 256） | `RUN_STATE.md`:100 |
| 支持状态与计数 | `applied_v1_2`；agentview accepted 521 / deferred 759，wrist accepted 267 / deferred 1013，分母均为 1280（= 5 历史源 × 256 区域；该样本只有 6 个唯一帧，见 §4.4） | `RUN_STATE.md`:101 |
| 参数量（同参数量事实） | 3,112,354 = encoder 1,496,480 + bridge 1,615,874 | `P3_report.md`:101-102 |

这些数字**不构成任何收益证据**：loss 与梯度只证明通路连通、参数真的在更新、后端确实冻结；支持计数只说明支持在真实数据上运行并给出偏保守的判定。没有成功率的闭环、没有第二个训练种子、没有跨平台复核。

---

## 附录 A：依赖图一页版

```text
                        ┌─────────── 内容（神经）通路 ───────────┐
MeasurementFrame ──┬──▶ DbsrFeatureAdapter.step ──▶ _window ──▶ _resolve_motion ──▶ offsets/valid/risk_kind
                   │        │
                   │        └─▶ SpatialSupport.observe ──▶ SupportDecision
                   │                    │  state ∈ {invalid, accepted, deferred, retracted}
                   │                    │  source_usable = expand_to_cells(...)  ← 保守感受野
                   │                    ▼
                   │            effective_mask[:,1:] &= source_usable
                   │                    ▼
                   │            encode_window ─▶ MaskedDbsrFusion.forward ─▶ weights
                   │                    ▼
                   │            _masked_pool ─▶ CameraTokens ─▶ readout ─▶ VisualTokenPostprocessor ─▶ π0.5
                   │
                   └──▶ MeasurementLedger.observe ──▶ LedgerFrame + validity ∈ {valid, missing, bad, saturated}
                              │
                              ├─▶ NoiseModel.covariance_for ──▶ Σ
                              ├─▶ VersionedDomain{evidence, context} ──▶ fit_structure ──▶ StructureReadout
                              │                                            （删除检查走这条）
                              ├─▶ without(ids)：物理涂掉 0/MISSING（整源删除）
                              ├─▶ with_content(replacement)：命名 overlay（内容替换，不动原始记录）
                              └─▶ _exact_window(32)：窗口外逐个登记 HistoryLoss
```

## 附录 B：本文件实际读过的文件

本次写作直接读取并据此引用（括号内为实际读取范围；未通读的文件已注明）：

1. `src/owac/frameworks/n01_full/support.py`（1-724，全文）
2. `src/owac/frameworks/n01_full/adapters.py`（82-103、333-502、503-707、840-934；其余部分只用结构检索）
3. `src/owac/baselines/burst_features.py`（1-264，全文）
4. `src/owac/frameworks/n01_full/replay.py`（140-439、440-614、615-674；其余部分只用结构检索）
5. `src/owac/frameworks/n01_full/ledger.py`（500-614；函数结构用检索确认）
6. `src/owac/frameworks/n01_full/domains.py`（128-247、368-462；函数结构用检索确认）
7. `src/owac/frameworks/n01_full/statistics.py`（1-30、30-47、155-284；函数结构用检索确认）
8. `src/owac/frameworks/n01_full/model.py`（1-317，全文）
9. `src/owac/frameworks/n01_full/config.py`（仅 241-257 `ReplayConfig`）
10. `src/owac/frameworks/n01_full/contracts.py`（仅 :51 `VALIDITY_KINDS`、:66 `SUPPORT_ROLES`）
11. `src/owac/backends/raw_correspondence.py`（484-563）
12. `src/owac/raw/n01_paired_views.py`（1-257，全文）
13. `docs/N01_full_development_plan_2026-10-15_v1.2.md`（1-284，全文）
14. `research/experiments/n01_v1_2/RUN_STATE.md`（1-121，全文）
15. `research/experiments/n01_v1_2/P3_report.md`（56-125；标题与结论行用检索确认）
16. `research/experiments/n01_v1_2/P3_reuse_map.md`（仅 :16 一行，经检索确认）
17. `research/theory/n01_support_contract_v1.md`（**仅标题结构**，经检索获得；未通读，仅用于术语对齐）
18. `tests/encoders/n01_full/test_model_v1_2.py`、`tests/encoders/n01_full/test_support.py`（**仅用例名结构**，经检索获得；用于确认存在哪些可判定自检，未通读）

未读取：`P3_worktree_diff.txt`（他人并行工作的大 diff）、任何 `.json` 结果全文（只读取了被引用的字段）、远程服务器上的 `receipts/verify-*.json` 原文（正是 §4.4 待核对项的所需材料）。

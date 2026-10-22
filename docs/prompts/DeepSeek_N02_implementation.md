# DeepSeek N02 测量约束的循环补全实现提示词

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 2026-10-16 当前训练由 Codex 接管。用户明确不让 DeepSeek 接入训练；不要启动、重配、重启或续训现有作业。本文旧启动语仅为历史，实现与运行状态见 [r5 记录](../../research/experiments/n02_v1_2/P2_training_live_20261016.md)。后续以用户新的直接授权为准。

> 历史原型任务。当前采用 [N02 v1.2](../plans/N02_full_development_v1.2.md)，P2 待实施；下方对 v1.1/P1 的启动安排已失效，原文仅供回查。新版[P2—P4自包含包](n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环。

> 历史原型提示词。2026-10-13起，N02 v1.1正式开发使用[分阶段提示词包](n02_v1_1/README.md)，P0 已在2026-10-14验收，当前接续P1。全部N02任务优先连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，按[远程开发说明](n02_v1_1/03_REMOTE_DEVELOPMENT.md)接入 `/workspace/owac/worktrees/n02-measurement-completion`。本文件保留用于核对旧实现原始要求，其中旧网格、四邻域propagate及tiny配方不作为v1.1必选实现。

以下为历史原型任务正文，仅在明确要求复现旧原型时执行；执行位置统一为上述远端 worktree。原任务要求在分支 `lzh/n02-measurement-completion` 中独立实现 **N02 测量约束的循环补全**。正式 v1.1 开发应下发分阶段提示词及共同远程执行说明；下文仅保留历史算法、接口和参考来源。

先通读本文件，再读取列出的仓库材料。从最小可运行算法开始写代码，逐阶段运行验证并继续接入训练和真实数据。你不是只写研究计划；应交付可执行的实现。当前三个 worktree 用于独立研究，只修改本分支对应机制和它需要的公共基础，不将三种机制强行串成一个模型。

本轮明确的完成目标是：核心算法及最近基线可运行、合成机制训练/验证可复现、真实 NPZ loader 可用、读出与 π0.5 训练/推理特征接口完整。实际数据/权重可访问时执行真实 smoke；完整 benchmark 长跑另按已有资源与冻结协议安排。下面标为“建议”“v0”“起跑”的设计属于本轮研究实现方案，可以被实验否定，不代表 roadmap 已规定具体公式。

## N02 需要实现的核心算法

研究目标是：在不完整测量下保留多个与行动有关的解释，并在新反证到达时撤回错误解释及其关联状态。请实现一个**测量约束的局部多假设循环状态场**。首轮参考 F02/F05 的循环推断和 F10 的一致性组合，不需要同时实现其他五个机制方向。

普通 recurrent encoder、预测编码、粒子滤波、Hopfield 检索均是已有机制。候选要单独实现并检验“反证定位—依赖传播—受影响区域重组”的更新规则，比较其是否在相同假设数、循环预算和历史下比普通循环更快纠错。

### 第一步 定义测量与假设的两个存储对象

实现 `MeasurementBuffer`，只储存当前及有限历史的真实观测、唯一 ID、mask 和采样条件；实现 `HypothesisState`，存储 `h[B,V,K,D,h,w]`、假设分数、局部关联边、修订状态与年龄。tiny 取 K=4、D=32，循环次数 4，历史 4；K 是候选解释数量，不是额外观测数量。

每个假设可以预测当前采样位置上的 RAW 值 `mu_k=M_theta(h_k,metadata)`，也可以输出潜在几何/关系。RAW prediction head 用于约束/辅助训练，VLA 主读出消费 h；预测值不写回测量缓冲。原生 CFA 的 head 仅在实际采样坐标/滤色响应上计算 likelihood，三通道 pseudo-RAW 使用另一个已声明的测量 head。

首个可手算的版本用“开口朝左/朝右”两个显式假设，已知几何到测量的简单前向算子，固定先验 odds 和可控噪声。先把正确/错误先验、同帧重复迭代、新反证更新做对，再扩展到 learned latent hypotheses。

### 第二步 固定观测集合上的循环求解

一次 `step` 接收新帧后，将其作为唯一新观测加入当前 buffer，冻结本次循环使用的 measurement set。每次循环更新 h，但重新计算**相同集合上的完整分数**，不将同一 likelihood 多次累加到上一轮分数：

```text
measurement_energy_k = sum_{i in unique_observations}
    mask_i * robust_nll(y_i, M_theta(h_k, m_i), variance_i)
energy_k = measurement_energy_k
         + beta_prior * prior_energy(h_k, causal_context)
         + beta_relation * relation_energy(h_k, neighbor_hypotheses)
score_k = -energy_k / temperature
weights = softmax(score, dim=K)
```

Gaussian 近似使用 `0.5*(residual^2/variance + log(variance))`，可选 Huber residual 限制异常点影响；已饱和样本首版单独 mask，不把 clipped 值当普通精确观测。噪声未知则以验证集拟合的尺度计算 energy；此时 weights 是相对解释权重，未校准前不称真实概率。固定 buffer 的累计或平均能量必须与对应噪声模型一致；用于 scheduler 的归一化残差另设字段，不随意归一化 posterior 来“解决”不同测量数的问题。

历史测量约束跨时刻潜在状态时，必须保留对应 `h_{t-i}` 或显式传播/对齐到其采样坐标。tiny 的动态版本先只以当前帧 likelihood 约束当前 h，历史经一次状态转移形成 prior；精确手算静态例子可以对整个 buffer 累计。不要直接用当前场景状态拟合所有未经对齐的历史帧。

先验 `prior_energy` 在本次 step 的循环中固定其来源。循环可重新优化 h/关系，但不能把上一轮由同一观测产生的 posterior 再当作一份独立 prior。新帧到达才进行时间更新。bounded-history 方案和递归 prior 方案分别实现，首版选择一种，避免同时保留完整历史 likelihood 又把其 posterior 重计一次。

### 第三步 反证驱动的局部修订

`RevisionController` 从实际观测与预测的归一化残差、假设之间的分歧及支持 mask 形成 contradiction map。对局部节点 u 建立与邻域 v 的依赖边；首版采用 4 邻域图和可学习的 compatibility，后续可以扩展稀疏关系。边表示“u 的解释依赖 v”，不把它自动当作物理因果。

每轮执行：

1. 标出新测量显著反驳的候选和局部位置，使用连续 residual 分数与配置阈值。
2. 将修订优先级沿**有记录的依赖边**传播一跳，累计到本轮待更新集合；保留传播前后的 mask 便于审计。
3. 用邻域消息、测量 residual、prior context 和当前 h 输入共享 `HypothesisUpdateCell`；只更新选择区域。tiny 可先 dense 运算后 mask 合并，但要如实报告其计算成本。
4. 错误候选降权并允许重置其受影响局部；提议新候选从当前局部测量与剩余上下文产生。保留未被反驳区域，避免每次 reset 全部状态。
5. 重新计算完整 energy 与 readout，不把上轮 energy 加进来。达到最大迭代或验证过的 residual/变化容差后停止，返回未消除歧义及实际循环数。

不靠将 prior 权重设为零来实现纠错。保留正确先验的帮助，并让弱但有效的反证可以改写强但错误的初始解释。`beta_prior` 可学习但有明确范围，起始值做 validation 扫描；反证出现时的处理规则必须有可单独关闭的配置。

### 第四步 学习假设和读出

stem 生成低分辨率 RAW 特征。`HypothesisInitializer` 由当前局部测量、因果历史和 K 个不同的初始化 embedding 形成 K 组潜在状态。tiny 用共享权重的 ConvGRU/残差 cell 做具体更新器；这些 cell 本身是已有材料，新增部分是 measurement energy 和修订调度。

为防所有假设退化成相同状态，训练先用合成多解任务检验输出是否覆盖不同几何。可用 mixture negative log-likelihood 或集合匹配监督，让至少一个候选接近真值；加小权重的任务相关表示多样性项，并对过度分散造成的无效假设计入结果。不要仅靠 latent 欧氏距离就宣称有 K 个不同物理解释。实际部署先验来自可用语言/历史/学习参数，手工正确/错误先验仅用于明确标注的机制干预。

读出实现两种：

- `map_hypothesis`：选择当前最佳整体解释或一致局部组合，保留选择索引及不确定性诊断，是最直接基线。
- `hypothesis_tokens`：给不同候选分配独立 token 组，附 hypothesis ID、位置、支持和权重；每相机 256 token 的总预算不变，K=4 时每组 64。不要先把不同开口位置平均为中间几何。

第二种接口仍可能在后端被错误融合，因此必须做“两个不相容候选”的动作/定位探针，与 MAP 和直接均值读出对照，测错误中间动作比例。不能以 token 分组的代码存在声称歧义已被后端正确利用。

### 第五步 训练与反证数据

合成生成器直接在测量域产生以下事件序列：先给歧义片段，经过随机延迟后显示区分线索，再增加噪声/遮挡；另有从始至终无法辨识的序列。开口方向、把手方位、两段轮廓是否连接是三个小任务，优先做第一个完整跑通。

先验条件为 correct、misleading、uninformative、missing、stale；stale 表示先前真实但场景后来改变。证据到达时刻与强度独立采样，不让固定帧号成为答案。机制监督可以包含隐藏几何，但只进 loss/evaluator，encoder 输入只含测量及声明的上下文。训练/验证分离形状、位置和遮挡模板。

tiny 目标取 `L_measurement + L_geometry_set + 0.1*L_revision + 0.01*L_diversity` 为起点。`L_revision` 在独立可控序列中约束反证后的错误假设权重/局部几何误差；`L_diversity` 只在有多解条件启用，防止唯一答案也被强行分裂。权重在配置中，给出归一化定义和各项消融。真实 NPZ 训练以 π0.5 原动作损失为主，保留可关闭的测量预测；没有 N02 标签时不能凭空生成“真实反证”标注。

### 第六步 最近基线与指标

至少实现 K=1 单假设、同参数/历史/循环数的普通 ConvGRU、同 K 和同测量函数的粒子/多假设滤波、同 K 的 dense 全局循环、候选无依赖传播、候选无反证重置、候选正确/误导先验的成对运行。先在显式两假设例子中跑一个精确 Bayes update 作为数值参照，再对 learned model 做相同测量预算的比较。DPF 参考实现是旧 TensorFlow 代码，读取算法并以本项目 PyTorch 重写必要最小部分，不为参考强装整套旧环境。

记录：误补全率、真实几何覆盖、反证到达后的纠错帧数/内部迭代数、错误解释持续面积、错误相关动作/定位偏移、未影响区域漂移、测量 residual、候选坍缩比例、p50/p95 成本。若把 weights 当概率输出，另测 NLL/Brier/校准；无法辨识时检查是否保留歧义，不奖励武断答案。

重复同一观测的测试应检查 unique evidence count、likelihood 集合和精确静态版本的 posterior 不重复相乘。learned solver 的近似解可以随迭代改善，不能要求所有特征完全不变。新反证测试应从相同旧状态分叉，分别输入支持/反驳/无信息测量，以排除“过了几轮自动换答案”。

### 文件职责和交付顺序

| 新文件或目录 | 应实现的内容 |
| --- | --- |
| `src/owac/frameworks/n02/config.py`、`state.py` | 配置、假设场、依赖边、测量集合 |
| `src/owac/frameworks/n02/measurement.py` | 测量预测、支持 mask、likelihood/energy |
| `src/owac/frameworks/n02/hypotheses.py` | 初始化、候选竞争/提议、HypothesisUpdateCell |
| `src/owac/frameworks/n02/revision.py` | 反证定位、依赖传播、局部重组与预算 |
| `src/owac/frameworks/n02/model.py`、`readout.py` | N02Encoder、MAP/分组假设读出 |
| `src/owac/baselines/n02.py` | 单假设、普通循环、多假设直接基线 |
| `src/owac/training/n02.py`、`src/owac/evaluation/n02.py` | 训练、恢复、反证矩阵评测 |
| `configs/encoders/n02/`、`tests/encoders/n02/` | tiny/small/medium、语义和梯度测试 |
| `research/candidates/N02.md`、`research/experiments/2026-10-12_n02_encoder.md` | 设计与实测记录 |

实现顺序是：显式两假设的精确例子 → learned hypotheses + dense 循环 → 局部反证/依赖传播 → 错误先验矩阵 → hypothesis readout → 真实 NPZ 与 π0.5。small 取 D=64/K=4/6 次循环，medium 取 D=96/K=8/8 次循环；K、宽度与循环独立做消融，不把三项一起增长解释为某单项收益。

至少给出一次先验正确、一次强错误先验被弱有效反证纠正、一次无可辨识证据仍维持歧义的逐步轨迹。保留没有纠正成功的配置。只有正确先验上的提升不足以验收 N02 的机制主张。

## 共同工程起点

你是本分支的实现者。任务是把研究假设落实为可训练、可运行、可比较的编码器，交付实际代码、配置、机制实验和 π0.5 特征接入。先完成小模型的完整纵向链路，再增加规模。下面的 v0 结构、超参数和文件名是本轮给出的具体实现起点，不是论文已证明的结论；若实验暴露问题，保留对照并据实修订。

### 必读文件和实施范围

进入本提示词开头指定的 worktree，读取以下文件：

1. `AGENTS.md`、`README.md`、`docs/status.md`。
2. `docs/OWAC_academic_research_proposal_2026-10-09.md` 的第 3—8 节，以及与你对应的第 5 节。
3. `docs/OWAC_research_roadmap_2026-10-08.md` 的完整框架契约、5.6、7—9 节。
4. `docs/architecture.md`、`docs/decisions/0005-physical-intelligence-contribution.md`、`docs/decisions/0006-bio-inspired-inference.md`。
5. `docs/decisions/0011-rawvla-bench-main-route.md`、`docs/decisions/0012-rawvla-pi05-protocol-freeze.md`。
6. `research/experiments/2026-10-10_rawvla_pi05_deployment.md`、`research/experiments/2026-10-10_rawvla_pi05_full_baseline.md`。
7. `src/owac/backends/rawvla.py`、`src/owac/runtime/rawvla_robotwin.py`、`tests/test_rawvla_contract.py`。

本轮用户已明确启动 N01、N02、N04 编码研究。可以建设原型和独立特征入口，不需要等待旧 `owac-dev-v0.1.0` 标签；该标签也不应被假定已经存在。三个分支共同继承部署标签 `owac-rawvla-pi05-v0.1.0`，对应提交 `16075fca7c5e5cd126900bc62531a5733fe4e0be`，并补入相同的最新研究文档快照。快照清单在 `docs/prompts/research-source-snapshot.json`。

当前基准使用 RawVLA 固定的 StarVLA π0.5。官方 openpi 是辅助参考，不是本轮另起一套基准的理由。数据和大型权重已部署在记录中的服务器 `/workspace/owac/rawvla/`；通过配置指向已有数据。先核对实际可用路径和运行环境，不能把服务器路径当作本机已经存在。旧原型复现也在指定远端进行，使用该机 CPU 小模型、合成测量及真实离线数据；数据路径以远程开发说明中的已核实位置为准。

### 两种输入分别实现

| 输入模式 | 实际含义 | 首轮用途 |
| --- | --- | --- |
| `rawvla_rgb3` | LIBERO 为逆处理得到的三通道 uint8 pseudo-RAW；RoboTwin 为三通道 float32 HDR-derived RAW | 当前 benchmark 上的训练与闭环接入 |
| `bayer_cfa` | 一像素一个滤色测量，带 CFA 排列及坐标；首轮用明确标注的合成 CFA | 检验原生测量、采样相位及噪声机制 |

三通道数据不能改名为 Bayer。不要重新马赛克 RawVLA 图像来宣称得到原生 RAW。两种输入共用核心规则，输入 stem 和元数据适配分开。真实相机数据是后续证据；本轮不恢复 Arena/PolaRiS 的历史路线。

数据加载器建立在 `src/owac/raw/rawvla_sequence.py`。参考作者 `RAWVLALightNPZDataset` 的动作索引和归一化，但单独输出测量张量，不继承其 PIL 图像输出和 RGB chroma target：

- LIBERO 键：`agentview_raw_uint8`、`wrist_raw_uint8`；当前 `robot_states[t]` 为 8 维，监督 `actions[t:t+10]` 为 7 维。使用已固定 quantile 统计。
- RoboTwin 键：`head_camera_raw_float32`、`left_camera_raw_float32`、`right_camera_raw_float32`；作者缓存以 `actions[t]` 作为当前 14 维关节状态，监督从 `actions[t+1:t+51]` 开始。在线当前状态来自实际 `joint_action/vector`。16 维 `robot_states` 是不同的末端表示。
- RoboTwin 的缓存状态映射只用于复用作者数据协议；区分“当前关节状态字段”和“已执行控制输入”。若缓存没有可靠的实际执行动作记录，动作上下文标为缺失，不能把未来监督当成已执行动作。
- 按 episode/采集来源划分 train/validation；同一来源的 RGB、RAW、照明衍生版本分到同一侧。正式闭环 manifest 单独保留。先检查可用身份字段，再生成确定性拆分清单。
- 采样一个连续窗口，给每帧保留 `episode_id/camera_id/frame_id`。开头 padding 的重复帧用 `new_measurement=False` 标识；动作尾部的 padding 用独立监督 mask，损失不重复计入填充项。
- 时间戳存在则使用真实单位；只有索引时使用 `time_unit=frame`，不编造秒数。普通随机 batch 在窗口起点 reset；连续 rollout 只在同一 episode 内携带状态。训练窗口允许先用 burn-in 前缀构造状态，再计算后段损失。

原生 CFA 使用显式黑白电平归一化 `x=(raw-black)/(white-black)`；保持缺失、坏点、饱和三个 mask。可以将 2×2 CFA 重排为四个测量平面，但记录每平面的实际采样偏移及 RGGB/BGGR/GRBG/GBRG 语义。空间裁剪、位移和重排同步更新这些坐标。对四平面分别采样，再做特征混合；不要把四个通道当作四个共址颜色。

已知噪声模型时在同一数值单位下计算 `variance=shot*max(x,0)+read_var`。LIBERO uint8 在转 FP32 后除以 255；RoboTwin 的范围依照作者元数据和 RAW 生成链核对，不能无条件除以 255 或按帧最大值归一化。噪声、曝光、内参、位姿等未提供的字段用显式 availability mask；未知噪声可以使用拟合的尺度模型，但写清它是估计量。

### 外部接口的 v0 约定

这是三条独立实现的互操作约定，不规定候选内部状态。建议定义于 `src/owac/core/encoder_contract.py`，保持类型模块轻量；张量依赖放在实现模块或类型检查分支。

```python
# 字段语义优先于具体 dataclass 写法；实现时补齐类型。
MeasurementFrame(
    values,              # [B,C,H,W]，单相机，原始数值单位在 metadata 中
    valid_mask, bad_mask, saturated_mask,
    sample_xy,           # CFA 为每通道采样坐标/偏移，不同分辨率不混放
    episode_ids, frame_ids, camera_id,
    timestamp, time_unit, new_measurement,
    metadata, metadata_available,
)
EmbodimentContext(
    proprio, proprio_available,
    executed_action, executed_action_available,
    camera_transform, camera_transform_available,
    task_tokens, task_mask,
)
ComputeBudget(max_updates, max_history, deadline_ms=None)
EncoderOutput(representation, state, validity, diagnostics, cost)
VisualTokens(values, mask, camera_ids, positions)  # [B,L,2048], [B,L]

encoder.init_state(batch_size, camera_spec, device)
encoder.step(frame, context, state, budget) -> EncoderOutput
encoder.reset(state, batch_mask) -> state
encoder.detach_state(state) -> state
readout(output, token_budget) -> VisualTokens
```

`representation` 可以是状态场、假设集合或几何状态。训练时逐帧调用与在线完全相同的 `step`，在 TBPTT 边界 detach；同一帧的内部迭代不能再次累计观测。每个相机维护独立状态，跨相机融合若引入，单独记录信息及成本。`deadline_ms` 为可选实测软预算；首轮用确定性的更新次数控制计算，不把 Python 时钟轮询称为硬实时保证。

读出层再将各自表示变成 π0.5 视觉前缀。首个受控配置取每个有效相机 256 个、2048 维 token，作为对照统一的读出预算；内部不受该形状限制。小型 CPU 测试可用更小宽度和 token 数，单独命名，不冒充真实后端配置。相机/空间/必要的时间编码由新读出层承担，其学习参数计入编码器。无效相机和无支持位置进入 mask。

### 实际 π0.5 接入位置

固定 RawVLA commit 为 `ba9c56b04da2c4931b70d8844710f5928d43775b`；其 StarVLA 子模块 commit 为 `57d508cb180d0510ac9060358009c0f6ffaac420`，代码也托管在 `ShuhongLL/RawVLA` 的对应提交中。`PI05.py` 主要继承父类；真正的数据流在 `starVLA/model/framework/VLM4A/PI0.py`：

1. `_apply_raw_frontend`：现有方法把 RAW 前端输出变回 `image`。新方法另建特征分支。
2. `_prepare_examples`、`_image_to_tensor`、`_build_observation_from_batch`：现有图像处理入口。新分支保留语言 tokenization、状态 padding、统计与动作处理，但测量直接走 OWAC loader/encoder。
3. `embed_prefix`：原来逐相机调用 `self.vlm.embed_image(image)`，再与语言嵌入拼接。新增 `embed_prefix_from_visual_tokens`，使用 OWAC 的 tokens/mask，并按原代码保留语言嵌入缩放、prefix attention mask、位置编号的约定。
4. `forward_from_observation`：训练前向。增加特征版本，复用 `action_head.prepare_flow_targets`、`embed_action_suffix` 和 `forward_with_action_expert`，使用原动作 flow-matching 目标。
5. `sample_actions` / `predict_action`：推理前向。加入对应特征版本，复用原 flow 求解、缓存与后处理。一个观测只运行一次 encoder，其视觉前缀供动作去噪迭代重复读取；10 次去噪不能变成 10 次新测量。

建议独立模块 `src/owac/backends/rawvla_features.py` 实现 `RawVLAFeaturePolicy` 和上述入口，通过明确的 wrapper/subclass 接入。需要改第三方时保存小补丁、固定 commit 和应用说明，不直接改正在评测的冻结目录。保留原 C0 的图像分支用于回归。

`OpenPIPaliGemma.py` 的 `embed_image` 包含 SigLIP/视觉 projector；OWAC 从 projector 之后的语言宽度接口进入。主配置加载语言与动作参数后不保留原视觉塔执行路径；若加载过程暂时构造了它，说明并在可行时释放。新读出替代视觉投影，记录新增参数。原 `load_audited` 保持服务于完整基线；另做明确允许“被替换视觉权重”的加载映射，仍逐项检查语言/动作参数。

**冻结后端参数时仍需让动作损失梯度回传至新 encoder。** 使用 `requires_grad_(False)` 冻结语言/动作参数，训练前向保留 autograd 图，不用 `no_grad` 包住整个后端；推理才使用 inference mode。补一个真实的梯度贯通验证。

动作设置沿用冻结协议：LIBERO horizon=10、replan=5、7 维控制；RoboTwin horizon=50、14 维绝对关节目标，执行重规划节奏读取现有 client，不能从 horizon 推断。模型内部 action_dim=32 的 padding 与真实控制维度分开。去噪次数是另一参数。新增状态 reset 与原动作缓存 reset 同时发生，并支持 batch 中部分 episode 结束。

### 训练和对照的共同顺序

先完成本方向的小型机制任务，再接真实 RawVLA NPZ，再做 π0.5 适配。推荐配置是起跑参数，不是已经调优的训练协议：

| 阶段 | 起跑设置 | 交付 |
| --- | --- | --- |
| 单元与解析小例子 | CPU、FP32、固定种子、可手工计算的小尺寸 | 核心更新与参考解一致，非零梯度 |
| 机制训练 | 宽度 32/64，序列 8—16 帧，AdamW，lr=3e-4，先 200 步 smoke | 曲线、保存/恢复、独立生成条件的验证 |
| NPZ 试运行 | 1—2 条真实轨迹、同源划分、先 20 个 batch | 数值范围、时间/动作对齐、真实 loss/backward |
| 小规模适配 | 编码器+readout 学习，语言/动作冻结；lr=1e-4，梯度裁剪 1.0，先单卡可承受 batch | 相同数据与更新预算下的 C3/C4 对照 |
| 扩展适配 | 只有冻结设置分析完后，单列有限参数或更充分后端适配 | 明确 trainable 参数清单、相同 baseline 适配预算 |

各方向可增补 RAW 测量预测或机制监督，但主路径没有 RGB 重建、RGB 教师或原视觉塔特征蒸馏。实现 `loss_weights` 配置，分别输出各 loss 及有效样本数；辅助预测不能把自身生成值当成新观测。没有真实动作标签的合成试验用显式命名的几何/定位探针，不伪造机器人动作标签。

直接系统对照优先实现 C3 普通 RAW CNN + 同读出，以及 C4 固定 RAW patch 投影 + 同等后端适配。明确 C4 的固定随机投影种子与归一化，学习部分只出现在披露的 readout/backend 中。现成 π0.5 Default 是冻结参考，不能自动叫“充分调优 C0”；另保留 proposal 中充分调优 RGB（C0）、浮点图像（C1）和作者 RawVLA（C2）的扩展比较配置。候选与机制对照共享历史长度、元数据、动作上下文、训练划分、调参范围与读出预算。

正式比较使用已冻结的 LIBERO 10,000、RoboTwin 3,250 身份；632 原条件核查独立计分。先导机制、离线验证和 smoke 用自己的小清单，不改这些分母。当前分支文档中的基线进度是带时间戳的历史快照，最终比较应读取最新完整归档。

### 可直接使用的目录和命令目标

新增可复用代码放 `src/owac/`，命令模块薄封装；机制自己的核心、配置、测试、训练目标和指标放本方向命名空间。三分支的共享接口按上文实现同一版本，先各自在隔离分支验证，不互改其他 worktree。后续合并优先整合 N01 的公共输入/读出骨架，再逐项合并 N02/N04 对共享层的必要改动；不整份覆盖共享文件，也不依赖另一个分支先完成才能开发核心。

依赖规则：本仓库核心环境 Python>=3.11，而已部署 RawVLA 运行时是 Python 3.10。为纯编码器建立可选 `encoder` extra（PyTorch/必要数据依赖）并锁定；不把作者仿真器塞进根环境。特征桥及其传递依赖保持 Python 3.10 可执行语法，在已固定服务器运行时可用 `PYTHONPATH=<本分支>/src:<隔离的StarVLA>/` 导入，并做兼容 smoke；不要直接把 `requires-python>=3.11` 的根包 pip 安装到 3.10。真实训练通过独立 runtime 配方记录。核心 `import owac` 仍然不初始化模型或设备。

每条路线均应交付下列 CLI，按本方向替换 `n02`，参数名可以扩展但示例必须真实可执行：

```bash
uv sync --locked --extra encoder
uv run --locked --extra encoder python -m owac.training.n02 --config configs/encoders/n02/tiny.json --synthetic --steps 200 --output outputs/n02-smoke
uv run --locked --extra encoder python -m owac.evaluation.n02 --config configs/encoders/n02/tiny.json --checkpoint outputs/n02-smoke/checkpoint.pt --suite mechanism --output outputs/n02-eval
uv run --locked --extra encoder pytest tests/encoders/n02 -q
make check
```

以上为需要你实现的目标命令，当前尚不存在。另给出真实 NPZ 训练、恢复 checkpoint、π0.5 单 batch 前向/反向和独立策略服务的命令。数据/权重不可用时，完成可执行的合成训练与接入代码，精确报告缺失资产；不要把 mock 后端称作真实 π0.5 验收，也不要因此把整个任务停在计划阶段。

最终报告以中文说明：目的、实际实现、实测效果、结论；附核心文件、运行命令、输入来源、种子、代码版本、最佳与失败条件。记录 p50/p95 编码器时延、读出时延、完整系统时延（能测才填）、训练/推理峰值显存和状态字节数。机制曲线和闭环成功率分别报告。Git 仅保存代码、小配置、来源哈希、清单和数字摘要；大数据、checkpoint 与视频保留既有外部存储或 ignored outputs。

## 参考论文与参考实现

先读与你当前实施步骤直接相关的代码，再读论文补充其假设。无需一开始安装所有外部项目。代码链接固定到 2026-10-12 核对的提交，文件路径由仓库树验证；主要接入文件已读取正文。`reference_sources.json` 记录核对层级与读取文件的哈希。引用外部实现时保留其许可证与来源，第三方代码仅取必要小单元或锁定依赖。

### 共同后端和数据参考

| 参考来源 | 具体看什么和如何使用 |
| --- | --- |
| [RawVLA 固定版本说明](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/README.md) | 沿用作者 benchmark、训练数据与环境入口。 |
| [RawVLA-Bench 固定数据版本](https://huggingface.co/datasets/ToferFish/RawVLA-Bench/tree/8586c594fff9ce38580e29745f074e82e96e74f7) | 检查数据目录、来源与训练轨迹；它不提供预渲染的正式测试轨迹。 |
| [作者 NPZ loader](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py) | 重点读 __getitem__、_normalize：相机键、history/burst、state_source_key、action_offset；独立实现 RAW 输出。 |
| [PI05 配置与继承](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py) | 查 2048/1024 宽度、action_dim、tokenizer、image_keys；主要执行逻辑继续读 PI0。 |
| [PI0 实际训练和推理入口](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py) | 重点读 embed_prefix、forward_from_observation、sample_actions；这里接入外部视觉 tokens。 |
| [OpenPIPaliGemma 视觉和语言接口](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/vlm/OpenPIPaliGemma.py) | 区分 vision_tower、multi_modal_projector、language_model、forward_prefix。 |
| [动作专家和 flow-matching](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py) | 复用 prepare_flow_targets、动作 suffix、velocity 解码及原 PI05 conditioning。 |
| [作者 LIBERO π0.5 配置](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/libero/pi05.yaml) | 核对动作 horizon、observation_stride、quantile、discrete_state_input；本地冻结加载配置优先。 |
| [作者 RoboTwin π0.5 配置](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/robotwin/pi05.yaml) | 读取继承的 base 配置并复用 checkpoint/stats；结合本地十四维动作审计。 |
| [LIBERO pseudo-RAW 生成链](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/libero/rawvla_bench/raw.py) | 理解逆处理来源、噪声和数值范围；不会因直读此数组而变成原生 CFA。 |
| [作者流式 RAW 前端](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/baselines/rawvla/rawvla.py) | 作为 C2 和已有时序机制的参照；借鉴状态与采样组织，不将其 RGB 输出放入 OWAC 主链路。 |
| [官方 openpi PyTorch 固定实现](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py) | 作为语言前缀/动作后端语义的辅助参考，不绕开本轮 StarVLA 固定路线。 |

### N02 机制参考和最小复用范围

| 参考来源 | 具体看什么和如何使用 |
| --- | --- |
| [Tang 等 循环视觉补全](https://doi.org/10.1073/pnas.1719397115) | 生物/计算动机；遮挡补全与低光测量分别设计实验。 |
| [Tang 等作者代码入口](https://github.com/kreimanlab/occlusion-classification/blob/94514806fb90da6f91193e86d97de7653c4a0247/README.md) | 查看作者实验材料与模型；不复用其视觉分类分数冒充 RAW 机器人效果。 |
| [Rao 和 Ballard 预测编码](https://doi.org/10.1038/4580) | 区分预测、残差与反复求解；本候选新增规则另作对照。 |
| [PredNet 循环预测参考](https://github.com/coxlab/prednet/blob/843b452f52bb44ca79ba6f65365e9d60cebed489/prednet.py) | 读取预测误差与 recurrent state 的组织；旧 Keras 只作算法参照，最小更新单元改写为 PyTorch。 |
| [Differentiable Particle Filters 论文](https://arxiv.org/abs/1805.11122) | 理解运动模型、测量模型和粒子候选的学习分工。 |
| [DPF 作者实现](https://github.com/tu-rbo/differentiable-particle-filters/blob/488f233a07f275d7321562899cec7e63ec389069/methods/dpf.py) | 实现同 K、同观测的多假设直接对照，注意 proposal/weighting/resampling 的关系。 |
| [现代 Hopfield PyTorch 实现](https://github.com/ml-jku/hopfield-layers/blob/f56f929c95b77a070ae675ea4f56b6d54d36e730/hflayers/__init__.py) | 可选的关联读取基线；关联检索不能代替测量支持和反证撤回。 |
| [现代 Hopfield 论文](https://arxiv.org/abs/2008.02217) | 定位既有关联更新与 attention 的关系；不宣称引入循环/记忆就是新颖性。 |
| [紧凑 ConvGRU 参考](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/update.py) | 为 HypothesisUpdateCell 提供可读的小型 recurrent cell；加上本候选自己的 energy/revision。 |

## 现在开始执行

先连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，进入远端 `/workspace/owac/worktrees/n02-measurement-completion`，检查分支和已有修改。先完成候选卡的“最小状态与更新规则”，随即实现本方向第一步的解析例子并运行；然后按上述阶段推进。每次汇报给出已完成代码和实际结果，不用连续计划代替实现。最终更新 `docs/status.md` 与本方向实验记录，提供复现命令和未完成项。

# DeepSeek N01 自适应时空证据积分实现提示词

请在 `/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration` 中独立实现 **N01 自适应时空证据积分**。本文件可以整体交给 DeepSeek，包含研究算法、共同接口、数据/后端接入、实施顺序和参考来源，不需要再拼接其他提示词。

先通读本文件，再读取列出的仓库材料。从最小可运行算法开始写代码，逐阶段运行验证并继续接入训练和真实数据。你不是只写研究计划；应交付可执行的实现。当前三个 worktree 用于独立研究，只修改本分支对应机制和它需要的公共基础，不将三种机制强行串成一个模型。

本轮明确的完成目标是：核心算法及最近基线可运行、合成机制训练/验证可复现、真实 NPZ loader 可用、读出与 π0.5 训练/推理特征接口完整。实际数据/权重可访问时执行真实 smoke；完整 benchmark 长跑另按已有资源与冻结协议安排。下面标为“建议”“v0”“起跑”的设计属于本轮研究实现方案，可以被实验否定，不代表 roadmap 已规定具体公式。

## N01 需要实现的核心算法

研究目标是：在同样的因果观测、曝光和后端下，按测量可靠性、运动与动作精度重新组织支持域，改善弱信号利用、定位与反应速度之间的取舍。请实现一个**支持域可重组的多时间尺度测量状态场**。首轮选取 F07 的时序状态与 F11 的显式统计作为实现材料；其他 F 家族继续保留。

普通 EMA、Kalman、ConvGRU 和 motion-compensated averaging 都是直接基线。拟研究的差别是：支持哪些采样、何时解除支持关系、何时缩短历史与读出，共同形成可检查的更新规则。不是给 ConvGRU 加一个 attention 就结束。

### 第一步 先做可手算的支持域积分

实现 `SupportDomainIntegrator`。输入经过校准的测量、样本坐标、噪声尺度、mask 及已知的时间间隔；输出局部统计和支持关系。

1. 每个相机保留最多 8 个**唯一帧**的因果环形缓冲。首个参考实现保留测量及 validity，在新帧到达后重算有界支持域；这样分裂时可以取回原测量，而不必从已经混合的均值中猜测如何拆开。
2. 支持域候选使用空间半径 `{0,1,2}` 与时间长度 `{1,4,8}` 的笛卡尔积。以上为 tiny 默认值，在配置中表达；真实时间可用时再把时间长度写成秒。所有历史都在当前时刻之前，帧数上限一致。
3. 同一 CFA 平面先按其真实采样位置形成局部邻接图。连接权重来自校准测量差、噪声归一化残差、边缘方向/距离与运动对应误差。先给出解析 gate，再做带可学习残差的小型 gate。平面之间的语义混合放在 feature stem。
4. `merge_threshold < split_threshold` 形成迟滞区：持续相容的邻接边可以打开；新的显著反证、运动不一致或遮挡让边关闭。先实现每次更新的一跳局部支持图及连通路径 mask，不把任意 transitive connection 的远处像素无限并入。保持半径上限。
5. 支持域分裂后依据当前有效边，从唯一观测缓冲重新计算各子域统计。状态字段保留 `support_edges`、`support_age`、`last_seen`、`history_ids`；不能只切换一个输出 gate 而仍沿用已经跨边界混合的旧均值。

对一个目标位置 p 的某个候选域，将历史测量记为 `x_i`，有效性为 `m_i`，噪声方差为 `v_i`，由时空支持关系得到的非负系数为 `c_i`。参考更新为：

```text
w_i = m_i * c_i / (v_i + eps)
S = sum_i w_i*x_i
W = sum_i w_i
mu = S/(W+eps)
V_diag = sum_i w_i*w_i*v_i / (W*W+eps)
age = sum_i w_i*(t-t_i)/(W+eps)
```

每个原始采样在一个支持域内只有一个最终系数。域重叠是允许的，但读出多个域的均值不代表获得多个独立观测。`V_diag` 只在相应独立噪声假设下是方差估计；运动插值或相关噪声下，输出 `variance_proxy` 与相关性说明，不把它称为精确后验。`sum(w)^2/sum(w^2)` 如用于诊断，只命名为权重集中度/名义有效数，同时单独给出 unique frame/sample count。

先用离散整数平移实现对齐，便于与逐源采样系数的参考解对照；再增加双线性坐标采样。FP32 累计均值和方差。任何时刻的统计只来自真实测量；feature 更新不会回写测量缓冲。

### 第二步 加入运动对应和可学习调度

将 `CausalCorrespondence` 独立出来，以当前/上一帧的低分辨率 RAW 特征估计位移、匹配残差和有效重叠区，首版用小范围相关性窗口 + 小型卷积头。只比较已到达帧。参考 RAFT 的相关性与 recurrent update 思路；完整 RGB 预训练 RAFT 只可作独立诊断参照，不成为主 RAW 路径的必要教师。

N01 的运动估计只服务于积分，不承担 N04 的自运动/外部变化分解。首轮包含三种配置：无对齐、固定同算法对齐、候选支持域重组。候选与运动补偿基线共用同一个 correspondence 模块及训练数据。

`SupportSelector` 为每个位置选择支持域或输出混合权重。输入包括 `mu/variance_proxy/age`、测量边缘、运动不一致、有效支持数及当前预算。任务精度/反应期限仅在确实作为公开任务要求提供时使用；不能输入目标真值位置。先以可解释的代价选择：

```text
J(domain) = lambda_noise * uncertainty_proxy
          + lambda_edge  * boundary_mixing_proxy
          + lambda_age   * mean_measurement_age
          + lambda_motion * correspondence_error
```

各项尺度按训练集统计规范化，权重只在 validation 选择。学习版以该选择为初始化/对照，通过定位与动作损失学习 domain logits；保留解析版。加上预算条件化的局部更新选择：预算低时减少候选半径、时间档或更新块数，返回最后有效表示，记录实际更新数量。首版即使为了选择而计算了全部候选，也必须把选择开销计入实测，不能据 gate 稀疏度声称省算力。

### 第三步 形成可复用编码器

建议 tiny 的顺序：`MeasurementStem -> SupportDomainIntegrator -> EvidenceBlock×2 -> SpatialReadout`。stem 使用 3×3 小卷积和 GroupNorm，处理 RAW 数值/坐标/metadata，避免 BatchNorm 跨时间 batch 的统计依赖。`EvidenceBlock` 的基本操作是支持图上的局部交换、保留/衰减与特征更新；显式统计与 learned features 分开保存，避免把 feature 幅度解释为可信度。

配置建议：tiny 宽度 32、2 个 block、历史 8；small 宽度 64、3 个 block；medium 宽度 128、4 个 block。宽度、空间尺度、最长历史和 block 数独立可调。首轮只要求 tiny 完整跑通、small 有真实成本采样、medium 有可实例化配置；只有实际训练的数据才能画规模趋势。

跨尺度组合保留细尺度边缘读出，粗尺度只参与上下文和调度，不用全局平均替代位置。总读出 token 预算保持共同约定。额外做一维异方差序列定位的 adapter，复用同一支持域积分/分裂规则，以验证它不依赖二维卷积才能定义。

### 第四步 训练目标

先实现合成弱边缘/窄缝定位任务：直接生成线性测量场、CFA 采样、Poisson-Gaussian 噪声、坏点、饱和和运动；生成器的几何真值只用于监督及评估。训练样本在线随机生成，验证固定独立 seed 与未见速度/对比组合。

第一轮 `L = L_location + 0.1*L_measurement_prediction + lambda_budget*L_update_cost`。定位用 Huber/热图交叉熵之一并写清单位；测量预测使用独立噪声 realization 或严格留出的测量，不让输入像素通过旁路复制目标。`lambda_budget` 默认 0，先建立精度基线，再比较非零值的曲线，不能用低计算 loss 掩盖任务误差。所有 loss 的有效像素与时间 mask 同步应用。

真实数据阶段保留同一编码器，用 π0.5 原生动作 loss 适配；测量预测作为可关闭辅助目标。至少运行辅助 loss 开/关和冻结/学习支持域的消融。

### 第五步 对照和反证实验

逐项实现：单帧 RAW CNN、固定空间/时间平均、固定时间常数 EMA、按噪声更新的 Kalman/递归估计、相同 correspondence 的运动补偿积分、同历史 ConvGRU、候选去掉 merge/split、候选固定期限。合成定位任务上所有方法使用相同 probe/readout 训练预算；完整 VLA 的最近基线优先选表现最强的两个，不必第一轮把每个滤波器都接大模型。

机制矩阵至少含：低/中/高 SNR × 静止/慢速/快速 × 宽/窄定位容差，另外加入突然出现、方向反转、遮挡解除和相邻目标交叉。记录位置误差、超过容差比例、首次稳定检出时刻、转向后恢复帧数、真实处理延迟、支持域宽度、历史年龄、merge/split 次数及内存。采样年龄是算法滞后，处理耗时是计算延迟，两者分报。

最重要的反证是：如果候选只提高静态平均信噪比，却在快速细边缘上拖尾，或者完全退化为最大窗口平均，那么 N01 的联合调度假设没有被支持。完整曲线保留，不能只报告最佳噪声条件。

### 文件职责和交付顺序

| 新文件或目录 | 应实现的内容 |
| --- | --- |
| `src/owac/frameworks/n01/config.py` | N01Config 与 tiny/small/medium 的参数校验 |
| `src/owac/frameworks/n01/support.py` | 支持图、合并/分裂、逐源统计的清楚参考实现 |
| `src/owac/frameworks/n01/motion.py` | 因果对应与有效 warp |
| `src/owac/frameworks/n01/state.py` | N01State、buffer、reset/detach |
| `src/owac/frameworks/n01/model.py` | N01Encoder、EvidenceBlock、SupportSelector |
| `src/owac/baselines/n01.py` | 上述最近机制基线 |
| `src/owac/training/n01.py`、`src/owac/evaluation/n01.py` | 真实训练循环、恢复和矩阵评测 CLI |
| `configs/encoders/n01/`、`tests/encoders/n01/` | 配置、数值/行为/训练贯通测试 |
| `research/candidates/N01.md` | 参照模板补齐框架设计、已有规则与拟新增差别 |
| `research/experiments/2026-10-12_n01_encoder.md` | 实际命令、配置、测量结果和下一项实验 |

先参考积分与手算例子，再支持图合并/分裂，再学习版/机制矩阵，再真实 NPZ 和特征后端；每阶段都保留可运行代码。重点测试：两帧不同方差的加权结果、同帧重复请求、交叉边缘分裂后旧混合被解除、部分 batch reset、时间间隔变化、相机状态隔离、训练/推理逐帧一致、梯度贯通。CPU reference 与向量化实现比较；不要只测输出 shape。

交付时给出三个可以现场查看的具体轨迹：静态弱边缘积累、快速边缘缩短历史、两个目标分离后支持域拆开。每个轨迹附测量、支持域、定位及状态诊断。可视化用于解释机制，数值指标决定实验结论。

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

当前基准使用 RawVLA 固定的 StarVLA π0.5。官方 openpi 是辅助参考，不是本轮另起一套基准的理由。数据和大型权重已部署在记录中的服务器 `/workspace/owac/rawvla/`；通过配置指向已有数据。先核对实际可用路径和运行环境，不能把服务器路径当作本机已经存在。日常实现从 CPU 小模型、合成测量和离线数据读取开始。

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

每条路线均应交付下列 CLI，按本方向替换 `n01`，参数名可以扩展但示例必须真实可执行：

```bash
uv sync --locked --extra encoder
uv run --locked --extra encoder python -m owac.training.n01 --config configs/encoders/n01/tiny.json --synthetic --steps 200 --output outputs/n01-smoke
uv run --locked --extra encoder python -m owac.evaluation.n01 --config configs/encoders/n01/tiny.json --checkpoint outputs/n01-smoke/checkpoint.pt --suite mechanism --output outputs/n01-eval
uv run --locked --extra encoder pytest tests/encoders/n01 -q
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

### N01 机制参考和最小复用范围

| 参考来源 | 具体看什么和如何使用 |
| --- | --- |
| [Stöckl 等 天蛾弱光时空求和](https://pubmed.ncbi.nlm.nih.gov/26948877/) | 生物启发：有限信号的时空积累与分辨率/响应取舍；不提供本提示词的支持域算法。 |
| [Sponberg 等 低光下运动跟踪](https://pubmed.ncbi.nlm.nih.gov/26068850/) | 设计运动速度与积分延迟的交叉实验；只在该生物实验范围内理解结论。 |
| [FilterPy Kalman 参考](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/kalman_filter.py) | 学习 predict/update 与 innovation、协方差组织；实现同噪声信息的直接估计基线。 |
| [RAFT ConvGRU 和更新块](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/update.py) | 读取 ConvGRU、SmallMotionEncoder、SmallUpdateBlock，作为普通循环和局部对应的实现材料。 |
| [RAFT 坐标和双线性采样](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/utils/utils.py) | 核对 coords_grid、bilinear_sampler 的坐标约定；另写 CFA 相位语义。 |
| [PyTorch unfold](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html) | 向量化有限邻域收集；与逐源参考实现比较，报告 unfold 的内存成本。 |
| [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) | 检查网格归一化、padding 与 align_corners；使用实际锁定 PyTorch 版本支持的参数。 |

## 现在开始执行

进入 `/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，检查分支和已有修改。先完成候选卡的“最小状态与更新规则”，随即实现本方向第一步的解析例子并运行；然后按上述阶段推进。每次汇报给出已完成代码和实际结果，不用连续计划代替实现。最终更新 `docs/status.md` 与本方向实验记录，提供复现命令和未完成项。

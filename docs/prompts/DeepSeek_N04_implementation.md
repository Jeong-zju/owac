# DeepSeek N04 自运动预测与变化分离实现提示词

请在 `/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`，分支 `lzh/n04-egomotion-separation` 中独立实现 **N04 自运动预测与变化分离**。本文件可以整体交给 DeepSeek，包含研究算法、共同接口、数据/后端接入、实施顺序和参考来源，不需要再拼接其他提示词。

先通读本文件，再读取列出的仓库材料。从最小可运行算法开始写代码，逐阶段运行验证并继续接入训练和真实数据。你不是只写研究计划；应交付可执行的实现。当前三个 worktree 用于独立研究，只修改本分支对应机制和它需要的公共基础，不将三种机制强行串成一个模型。

本轮明确的完成目标是：核心算法及最近基线可运行、合成机制训练/验证可复现、真实 NPZ loader 可用、读出与 π0.5 训练/推理特征接口完整。实际数据/权重可访问时执行真实 smoke；完整 benchmark 长跑另按已有资源与冻结协议安排。下面标为“建议”“v0”“起跑”的设计属于本轮研究实现方案，可以被实验否定，不代表 roadmap 已规定具体公式。

## N04 需要实现的核心算法

研究目标是：利用实际可得的本体状态、已执行运动与测量坐标预测自身运动造成的观测变化，再以新测量修正状态，并保留对外部真实变化的敏感性。请实现一个**自运动条件化的预测与创新分解编码器**。首轮以 F09 的测量坐标、F07 的持续状态和 F05 的迭代修正组织实现。

新增动作向量或做普通 feature warp 本身不是研究增量。候选的重点是：把传播误差、外部变化和成像不可靠性分别参与状态更新；需要与同上下文的 ConvGRU、运动补偿及 predict-update filter 比较。

### 第一步 先区分实际可用的运动信息

RawVLA NPZ 的公开本体状态并不自动提供相机六自由度位姿、深度或完整相机标定。实现两种配置，并分别报告：

| 配置 | 输入与作用 | 实际地位 |
| --- | --- | --- |
| `learned_proprio` | 当前/过去本体、可靠的已执行动作和上一状态，预测每相机的二维 feature displacement/传播参数 | 默认可部署路线，适合当前 RawVLA 信息 |
| `calibrated_geometry` | 实际可用的相机内参、外参、本体运动和深度/明确的场景平面假设，做显式投影 | 校准可得时的扩展；合成 oracle 仅作上限诊断 |

只有当前状态向量可用时，用 `q_t-q_{t-1}` 或其语义正确的相对变换作为运动输入，并保留时间间隔及 availability。LIBERO 的 8 维状态应按已有数据定义区分位置、旋转和夹爪；旋转不能盲目相减。RoboTwin 的 14 维关节状态明确左右臂与夹爪。缺少实际执行命令时不要用未来动作标签补齐。

头部固定相机、腕部运动相机分别建模；共享网络可加入 camera ID，但状态及运动标定分开。静止机器人仍可能存在物体运动，模型不能把所有变化归于动作。

### 第二步 实现可检查的传播算子

`EgomotionPredictor` 只使用上一状态和截至当前已实际发生的运动上下文，输出 ego warp/transition、传播可靠性及必要的低维成像变化参数。它不使用当前图像来产生“纯预测”；当前图像进入独立 correction 分支。

默认版采用 `MLP(proprio_delta,executed_action,dt,camera_id) + low-resolution previous features` 预测 coarse backward flow，再上采样到 feature 网格。限制初始位移幅度，使用零位移初始化做可复现起点。固定相机必须允许学到接近 identity 的 ego propagation。保留一个只有本体、不读上一视觉的更简单传播对照，以检验复杂条件是否必要。

显式几何版约定 `T_world_from_camera`：从相机坐标到世界坐标。若上一像素 p 有合法深度 d，则：

```text
X_prev = d * inverse(K_prev) * [u,v,1]
T_current_from_prev = inverse(T_world_from_current) @ T_world_from_prev
X_current = T_current_from_prev @ X_prev
p_current = project(K_current @ X_current)
```

这是 forward projection。实现 `grid_sample` 的 backward warp 时，必须用对应的反向采样关系及需要的当前深度，或用带可见性处理的 forward splat；不能把 forward flow 直接塞进 backward grid。第一版的解析测试用已知平面/二维平移，随后才扩展一般 SE(3)。没有深度时可用纯旋转/明确平面的 homography，单列假设；不能拿仿真隐藏 depth 填进正常策略。

CFA 数值的 warp 需考虑每个滤色平面的采样偏移。最简单可靠的实现是先将带采样坐标的测量编码为特征场，再在特征坐标下传播；若辅助预测原始 CFA，另用前向测量算子在目标 CFA 坐标采样。`align_corners`、像素中心、归一化网格、图像/feature 尺度转换明确固定。越界、遮挡和新暴露区域产生 validity，不能靠 border padding 制造历史证据。

### 第三步 创新分解和分开的状态更新

维护 `N04State`：上一稳定特征、变化特征、传播可靠性、有效区域、时间/相机身份，以及必要的历史上下文。内部状态可为多尺度场；不要求它是物理空间完整地图。

每个 step 的概念顺序：

```text
f_t = MeasurementStem(raw_t, metadata_t)
pred_t, warp_valid, uncertainty = Propagate(state_{t-1}, context_t)
innovation = f_t - pred_t
ego_error, external_change, imaging_unreliability = InnovationDecomposer(
    innovation, f_t, pred_t, warp_valid, uncertainty, metadata_t)
stable_t, change_t = Update(pred_t, f_t, decomposition)
representation = [stable_t, change_t, measured_feature_t, validity]
```

三个分数初版是用于更新的学习权重，未校准前不写成三个真实事件的概率。增加 unknown/ambiguous 的剩余通道或独立不确定性，尤其处理没有标定/没有动作/物体与相机同步运动的不可识别情况。

具体实现给出三个可检查的操作：

1. 运动解释可靠且 residual 小的区域，保持 warped state 并用当前测量修正。
2. 持续空间连贯的外部变化区域，增大当前测量更新、保留 change features，避免稳定分支把运动物体抹平。
3. 饱和、失效、遮挡或明显传播失配区域，降低历史/当前不可靠一方的权重，记录空缺，等待下一帧修正；不能将缺少测量强行分类为物体运动。

可使用 `stable=(1-g_update)*pred + g_update*f_t` 与独立 change cell 作为首版。g_update 根据观测可靠性、传播误差和变化分数联合给出；注意外部变化时应更及时更新，不是因为 residual 大就一律拒绝新测量。没有历史的区域直接以有效当前测量初始化。

照明分支首版只估计可靠区域上的全局/低频增益变化，帮助判断几何 residual；保留未消除的测量特征供 action readout。不要硬编码颜色/亮度对任务永远无关。开启/关闭该分支分别比较，并加入颜色决定目标选择的反例。

### 第四步 训练数据和损失

先做二维可控测量场：独立控制相机平移/旋转、目标运动、全局照明、噪声、遮挡和标定误差，再通过 CFA 采样得到观测。合成生成器可提供各变化的标签与真值对应，用作训练/评测；encoder 只能读取配置明确声明的测量、本体及标定。

tiny 初始目标：`L_location + 0.5*L_future_measurement + 0.1*L_change + 0.1*L_ego_warp`。未来测量用于训练监督时，以 `t-1` 状态和已执行的 `t-1→t` 上下文预测 t，目标只有在 loss 计算时读取，不能把它放入 predict 输入。把 `predict_before_observe()` 与 `correct_after_observe()` 写成两个函数便于检查。

`L_change` 只在合成可识别标签上监督；`L_ego_warp` 只在允许的已知几何训练数据上用真值。真实 RawVLA 没有这些标签时，使用可关闭的测量预测、动作 loss 与稳健自监督，不读取 simulator object pose/depth 当在线输入。留出的测量预测需检查遮挡和 noise，不能只惩罚均方差从而鼓励抹去移动目标。

默认 causal loop 与在线一致，TBPTT=8 或 16。先 identity propagation、再训练 ego predictor、再联合 decomposition/update、最后接动作 loss。记录“冻结传播模块”和“联合训练”两种结果，以判断分解模块是否只是依赖更好的对应估计。

### 第五步 机制矩阵和最强反例

必须分别生成：相机动/物体不动；相机不动/物体动；二者都动；只有照明变化；二者同向近同速；遮挡后新暴露；错误外参/动作延迟；曝光突然变化与饱和。前三项独立控制速度，不只测试训练范围。运动方向与光照变化随机组合，避免根据亮度识别场景类别。

最近基线至少有：单帧 CNN、同上下文 ConvGRU、普通运动补偿加 EMA、同测量模型的标准 predict-update filter、action-conditioned recurrent state model；再做候选无 ego、无外部变化分支、把全部 residual 视为噪声、上下文时间错位的消融。凡候选获得本体/实际动作/标定，对照也获得；oracle 几何组独立显示，不与无深度模型混报。

主指标是持续定位误差、目标移动后的恢复帧数、误把自运动当外部变化的比率、误抑制真实物体运动的比率、照明突变误报、遮挡恢复、动作修正幅度与时延；另报 warp error、预测 residual、mask 覆盖、p50/p95 时延和状态成本。只在合成有真值的数据上计算分解准确率；真实数据无标签则报告可测代理和闭环，不输出凭空的分解准确率。

N04 的关键反证：如果降低 ego residual 的同时也降低了对外部移动的敏感性，或者学习版用 action 作为任务 ID 捷径而不跟踪实际运动，那么该机制没有被支持。增加“相同动作上下文、不同实际物体运动”和“相同视觉变化、不同 ego 上下文”的配对条件。

### 文件职责和交付顺序

| 新文件或目录 | 应实现的内容 |
| --- | --- |
| `src/owac/frameworks/n04/config.py`、`state.py` | 两种运行模式、状态和时间语义 |
| `src/owac/frameworks/n04/geometry.py` | 坐标、投影、warp/splat 与 validity |
| `src/owac/frameworks/n04/ego.py` | 本体/动作条件化传播、独立 predict 接口 |
| `src/owac/frameworks/n04/innovation.py` | residual、变化/成像/传播误差分解 |
| `src/owac/frameworks/n04/model.py` | N04Encoder、多尺度组合和 correction |
| `src/owac/baselines/n04.py` | 同上下文 recurrence/filter/warp 对照 |
| `src/owac/training/n04.py`、`src/owac/evaluation/n04.py` | 训练恢复、运动矩阵和成对反例评测 |
| `configs/encoders/n04/`、`tests/encoders/n04/` | tiny/small/medium、坐标和因果测试 |
| `research/candidates/N04.md`、`research/experiments/2026-10-12_n04_encoder.md` | 规则、参考、可用元数据和结果 |

按顺序完成：整数平移/identity 的解析几何 → learned_proprio predict/correct → 四类基本变化矩阵 → 模型误差/不可识别测试 → CFA 相位/多视角 → RawVLA 离线与特征后端。tiny 为宽度 32、2 层、多尺度 2；small 宽度 64、3 层；medium 宽度 128、4 层。核心传播/更新应能组合为多个 block，并用一维移动信号例子复用其 predict/correct 规则。

测试必须覆盖 identity、正负单像素移动、变换组合/逆、warp 方向、不同 CFA 相位、out-of-bounds、遮挡/新区域、部分 batch reset、动作/状态时序、相机独立状态以及当前图像不能影响纯 predict 输出。加入同一帧调用多个去噪步时状态仅更新一次的集成测试。端到端动作梯度要到达传播与更新模块，除明确冻结的实验设置外不无意 detach。

最终交付至少包含四条并排轨迹：纯自运动得到稳定定位；纯物体运动得到及时更新；照明变化不伪造物体运动；错误自运动预测能被测量纠正。保留失败场景和 unknown 输出，明确当前信息不足时的边界。

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

每条路线均应交付下列 CLI，按本方向替换 `n04`，参数名可以扩展但示例必须真实可执行：

```bash
uv sync --locked --extra encoder
uv run --locked --extra encoder python -m owac.training.n04 --config configs/encoders/n04/tiny.json --synthetic --steps 200 --output outputs/n04-smoke
uv run --locked --extra encoder python -m owac.evaluation.n04 --config configs/encoders/n04/tiny.json --checkpoint outputs/n04-smoke/checkpoint.pt --suite mechanism --output outputs/n04-eval
uv run --locked --extra encoder pytest tests/encoders/n04 -q
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

### N04 机制参考和最小复用范围

| 参考来源 | 具体看什么和如何使用 |
| --- | --- |
| [Keller 等 感觉运动失配](https://pubmed.ncbi.nlm.nih.gov/22681686/) | 启发预测与实际反馈的差异；不将生物现象直接等同于机器人的可识别分解。 |
| [Kornia 深度与 warp](https://github.com/kornia/kornia/blob/4fcfcf863e0a9e2b3e93179c3fa96bcf278ce3c6/kornia/geometry/depth.py) | 读 depth_to_3d、warp_frame_depth；复核变换方向、深度属于哪个时刻及坐标单位。 |
| [Kornia 图像坐标变换](https://github.com/kornia/kornia/blob/4fcfcf863e0a9e2b3e93179c3fa96bcf278ce3c6/kornia/geometry/transform/imgwarp.py) | 参考 warp_perspective、remap 的网格变换；主路径作用于 RAW 特征，不生成中间 RGB。 |
| [Kornia 相机模型](https://github.com/kornia/kornia/blob/4fcfcf863e0a9e2b3e93179c3fa96bcf278ce3c6/kornia/geometry/camera/pinhole.py) | 理解 K、外参与图像坐标组织；仅在这些量实际可得时使用。 |
| [LieTorch 群变换](https://github.com/princeton-vl/lietorch/blob/e7df86554156b36846008d8ddbcc4d8521a16554/lietorch/groups.py) | 参考 SE3 的 inverse、compose、act 与表示约定；首版可用小型解析 PyTorch 算子，不强装 CUDA 扩展。 |
| [RAFT 运动更新参考](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/update.py) | 用于普通对应/循环基线；ego 预测与当前测量 correction 在本实现中分开。 |
| [DPF 预测与测量更新](https://github.com/tu-rbo/differentiable-particle-filters/blob/488f233a07f275d7321562899cec7e63ec389069/methods/dpf.py) | 与标准 state transition + measurement correction 比较，候选需要证明变化分离的额外作用。 |
| [Kalman innovation 参考](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/kalman_filter.py) | 作为小型可手算 predict-update 基线，避免把创新残差直接等同物体运动。 |
| [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) | 核对 backward sampling、归一化坐标和边界行为。 |

## 现在开始执行

进入 `/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`，检查分支和已有修改。先完成候选卡的“最小状态与更新规则”，随即实现本方向第一步的解析例子并运行；然后按上述阶段推进。每次汇报给出已完成代码和实际结果，不用连续计划代替实现。最终更新 `docs/status.md` 与本方向实验记录，提供复现命令和未完成项。

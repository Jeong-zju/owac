# GPT-6.1 Sol 执行 OWAC-D1.1 RAW 教师蒸馏方案

你是本任务的负责人 GPT-6.1 Sol。请在 OWAC 中落实已经认可的研究路线：**RAW stem＋继承当前策略的 SigLIP 权重＋教师蒸馏，先恢复正常光单任务控制，再研究弱光条件下的收益。** 你负责研究判断、接口设计、任务拆分、独立验收和返工；机械性简单工作与主要编码工作实际委派给用户本机的 DeepSeek Harness，默认 **V41-Flash，推理强度 Max**。

请实际完成当前授权范围内的工作，不只回复计划。先完成工程实现与验收；正式训练和机器人闭环按照下文的授权范围执行。不要把写好启动命令、Harness 自报完成或单元测试通过，写成控制能力已经恢复。

内部版本：**OWAC-D1.1**。本次修订取消固定单帧限制，明确完整教师策略以 RAW 为输入；D1.0 的单帧优先安排转为历史选择。实施以实际闭环效果、稳定性和测得成本决定时序设计。

## 已认可的方案与当前范围

本方案来自用户与 Codex 的[蒸馏讨论](thread://01a154bc-14d6-7d21-8355-9a32d6da6f30?hostId=local)，用户于 2026-10-20 明确认可最后讨论的路线。核心决定如下：

- 学生直接生成视觉 tokens，不逐个删除 ISP 算子。部署主链路保持 RAW→非图像表征→动作；学生的帧数、因果时序聚合和状态设计允许按效果选择，不固定为单帧。
- 继承**当前实际使用的 RawVLA π0.5 策略 checkpoint**中的 SigLIP blocks、位置编码、视觉归一化及原多模态投影。不能用另一个通用 SigLIP 的权重冒充同一策略初始化。
- 先证明能力可以迁移，学生容量接近原视觉塔，首轮不追求压缩、小模型或速度提升。
- 先围绕正常光单任务恢复做受控验证；**不规定教师或学生只能单帧，也不要求先证明单帧失败才能使用 burst／递归／记忆**。优先保留有效教师的真实因果时序能力，并为学生选择有依据的时序实现。是否加入额外复杂机制由效果、归因与成本决定，旧 N01／N02／N04 不自动恢复。
- **主教师策略对外必须接收 RAW 测量或因果 RAW 序列**，输出动作。允许它内部采用 Neural ISP→RGB→SigLIP→VLA；这与把配对 RGB 图像直接喂给 RGB 策略的教师不同。学生部署保持 RAW→非图像表征→动作；披露教师内部 ISP、RGB 预训练和额外训练数据。
- 首轮冻结 π0.5 的语言／动作后端，不沿用旧方案的后端 LoRA。视觉塔有限适配与后端适配是不同实验。

当前发送本 prompt，授权你完成方案落库、Harness 派工、代码实现、针对性测试、真实资产核验，以及资产支持时的零优化更新前反向／接口检查。**方案获认可不自动等于正式训练、仿真闭环或实机动作获授权。** 若当前会话没有另行明确的新训练／闭环授权，完成实现和预检后交付具体配置、资源画像、资产缺口及可执行命令，停在启动前。若用户另行给出本阶段授权，则在该范围内继续到阶段结果与独立验收，不逐项重复询问工程细节。

旧 N01／N02／N04 继续暂停，不恢复历史作业，不使用历史远端地址作为本轮可用资源。额外数据下载的大额存储／费用、超过预算、完整后端解冻、四任务扩展和实机动作，依各自新增授权处理。

## 工作位置与必读材料

工作区容器是 `/home/jeong/zeno/owac/`，Git 仓库为 `/home/jeong/zeno/owac/repo/`。默认直接在此 checkout 的 `main` 工作，保留已有修改。只有用户另行要求隔离时，才使用 `/home/jeong/zeno/owac/worktrees/<task>/`。所有 Git、依赖安装和代码检查都在目标 checkout 内运行。

首先读取并核对：

1. `/home/jeong/zeno/owac/AGENTS.md`。
2. `/home/jeong/zeno/owac/repo/AGENTS.md`、`README.md`、`docs/status.md`，以及 `git status --short --branch`。
3. `/home/jeong/zeno/owac/repo/research/archives/2026-10-19_worktree_consolidation/README.md`。
4. `/home/jeong/zeno/owac/repo/research/experiments/2026-10-19_visual_replacement_recovery.md` 与 `/home/jeong/zeno/owac/repo/configs/experiments/visual_replacement_single_task_v1.json`。
5. 与复用接口直接相关的实现：`src/owac/backends/rawvla.py`、`rawvla_features_n02.py`、`rawvla_language.py`、`src/owac/core/encoder_contract_n02.py`、`src/owac/raw/rawvla_sequence_n02.py`。这些路径均相对于主仓库。
6. `/home/jeong/zeno/owac/repo/research/reviews/2026-10-20_rawvla_public_teacher_assets.md`：作者公开资源、缺失权重与教师构建顺序；执行时核对最新状态。
7. 当前已落库的 `/home/jeong/zeno/owac/repo/docs/plans/RAW_STEM_SIGLIP_DISTILLATION_2026-10-20.md`、`docs/decisions/0016-raw-stem-siglip-distillation.md` 及 `research/candidates/RAW_STEM_SIGLIP.md`；核对而不重复新建同一采用决议。研究统筹台已将本路线设为唯一当前入口，旧分支保留历史。

第 4 项是旧 ConvNeXt 视觉替换协议，保留其历史与诊断经验；**新建本方案的配置和记录，不把它改名后交给旧 trainer**。旧方案的 ConvNeXt 初始化、后端 LoRA、学习率及无教师约束不自动继承。N02 特征桥只是可复用接入口，不意味着恢复 N02 算法。

2026-10-20 采用决议 0016 已明确变更历史“训练禁止 RGB 教师蒸馏”的约束，当前规则与研究统筹台已同步。执行时核对 `AGENTS.md`、`README.md`、`docs/status.md` 和架构说明：训练使用 RAW 输入的完整教师，教师内部允许生成 RGB；学生推理不生成 RGB/sRGB/XYZ 图像。保留历史文档和负结果，不重写旧实验的事实；实施进展继续同步到当前记录。

2026-10-19 的资产核查未确认本地完整的 LIBERO 策略、统计量、tokenizer、真实配对示范和初态清单。执行时重新检查实际文件；历史路径和存在性清单不能替代可加载验证。DROID 权重或合成样本不能代替 LIBERO 任务资产。

## 教师与学生的结构

```text
教师
同次采集的因果 RAW burst／历史＋教师递归状态
            → 作者 RawVLA Neural ISP／经验证的 RAW 前端 → 原 RGB 预处理
            → 当前策略的 SigLIP → 原多模态投影 → 教师视觉 tokens
            → 冻结 π0.5 语言／动作后端

学生
同次采集的因果 RAW 观测／历史
            → RAW 校准／打包 → RAW stem＋按效果选择的时序聚合／状态
            → 继承权重的 SigLIP blocks
            → 继承的原多模态投影 → 学生视觉 tokens
            → 同一个冻结 π0.5 语言／动作后端
```

这里的“同一个后端”指相同权重和执行语义；实现可以采用独立模型副本，但必须核对身份，避免训练学生时改动教师。教师固定在 eval 模式并停止梯度。

教师优先选择作者 RawVLA 的完整 RAW 策略并核查真实控制效果，保留其有效的 burst／递归能力。接口必须从 RAW 观测实际走到动作；只有 RGB 策略权重、配对 RGB 数据或未训练 RAW 前端时，不算已有 RAW 教师。不能为了符合旧单帧约束删掉教师历史。

教师和学生各自声明 RAW 帧数、采样时刻／stride、历史覆盖、状态、burn-in、reset、训练展开长度及成本。默认比较获得可比因果信息的实现，学生可用简单因果聚合或状态模块；若比较不同历史范围，明确额外观测优势，不能把差异单独归因于蒸馏或架构。拒绝未来帧、跨 episode 状态污染和对状态 reset 的遗漏。

真实 Bayer 采用校准与四通道打包入口，当前 benchmark 三通道 pseudo-RAW 使用独立三通道入口。两者分开配置、统计和报告。LIBERO 的 RGB8 逆处理、RoboTwin 的 HDR 来源及真实传感器 CFA 不能混称原生 RAW。

RAW stem 首版采用小型局部卷积／patch embedding，直接输出 SigLIP 期望的 patch embeddings，不输出供视觉塔再次编码的 RGB 图像。显式保留可获得的曝光、噪声、饱和与坏点信息；不存在的元数据如实标记，不伪造。不要因 pseudo-RAW 恰为三通道，就把任何三通道测量张量判成重建图像；审查它的来源和实际数据流。

从当前 checkpoint 读取并登记输入分辨率、patch size、网格、hidden size、位置编码、归一化位置、投影尺度、图像 token 数、相机顺序和 mask。常见的每相机 256 token、投影后 2048 维只是现有接口线索，以真实 checkpoint 为准。逐位置蒸馏要求教师／学生对应同一物理视野、裁剪、翻转、相机和时刻；Bayer 打包后的尺寸变化必须映射回同一视野，不能直接照搬像素坐标或随意插值对齐。

提取学生初始化时，先复制并核对所需视觉权重，再处理旧桥中释放原视觉塔／投影的逻辑。学生继承的投影不可被桥接代码误删，也不能在学生投影后再叠加一个隐藏的可学习投影。

## 作者公开模型与权重的实际状态

截至 2026-10-20，已核对作者项目页、官方仓库提交 `ba9c56b04da2c4931b70d8844710f5928d43775b`、模型 README、配置、checkpoint 工具、GitHub 文件清单／releases 和作者 Hugging Face 账号。详细来源见本仓库公开资产核查记录。

- RawVLA 模型源码、RAW 前端、StarVLA 接入和训练配置已公开。默认配置支持 `burst_frames=6`、`rnn_frames=6` 及递归状态；两个 6 分别属于输入 burst 与训练展开，不能简单认定为 36 个独立观测。
- **本轮未找到训练后 RawVLA Neural ISP／完整 RAW 教师的公开下载 checkpoint**。作者账号当时显示公开 models 为 0，GitHub releases 为空，所查树中未发现权重文件；不能由有限核查断言所有渠道都不存在。
- 仓库的 checkpoint 下载脚本列出的是 StarVLA Qwen 系列策略／基础模型，不是 RawVLA 前端权重，也不是当前所需 π0.5 完整 RAW 教师；π0.5 配置中的本地路径不能当成远端可下载资产。RawVLA-Bench 是数据集，不是权重。
- 当前公开训练配方从头初始化 RAW 前端；作者说明部分历史论文 checkpoint 早于当前前端变更，不能把新配方运行称为论文权重的逐位复现。

先核对可用作者权重／用户已有资产及匹配配置，组成并验收完整 RAW 教师。若确实缺少训练后前端，明确交付“先构建／训练 RAW 教师，再蒸馏学生”的分阶段计划、数据和资源预算；训练教师同样需要具体新授权。不能静默换成 RGB 输入教师、随机前端，或只用固定 ISP 结果冒充已验证的 RawVLA 教师。固定 ISP＋VLA 可作为另列 RAW 基线，须显式标注而不替代主教师。

现有 `src/owac/baselines/raw_stem_siglip.py` 中的单帧模块可复用为逐帧组件，但本轮已见其中仍有拒绝 temporal／burst teacher 的检查；资产配置中也可能保留 `teacher_single_frame_isp` 字段。执行者须审计并迁移这些限制，补上完整 RAW 策略及状态契约。**修改文档或删除报错不等于已实现时序支持**，须验证真实 RAW 输入、状态传递与重载行为。保留其他执行者的在途修改。

## 训练阶段与对照

### 阶段零 验证教师与接入口

先在缓存的真实专家观测上，把原教师投影后的 tokens 送入学生接入口，对比原策略路径。固定语言、状态、图像与相机顺序、归一化、精度、flow 噪声和时间。覆盖首窗口、后续重规划窗口与 episode reset。

逐项比较视觉 tokens、语言 tokens、padding／attention mask、position IDs、前缀、各去噪步速度场、归一化动作、反归一化动作及实际执行 chunk。先用同一模型重复运行测量误差，再冻结容差。不能看见差异后放宽门槛。真实差异未解释前，不投入蒸馏训练。

教师闭环验收在获得相应授权后执行；离线接口通过不能替代教师闭环能力。若当前教师在选定初态上无法完成任务，先诊断教师／环境／执行协议。

### 阶段一 RAW 输入侧适配

冻结继承的 SigLIP blocks、位置编码、原投影及语言／动作后端，训练 RAW 输入侧模块。单帧版本只更新 stem；若选用含可训练时序聚合／状态的学生，显式登记该模块与 stem 的联合更新范围。正常光下拟合**原投影之后**的教师视觉 tokens，缓存／重放必须复现生成目标时的因果历史和教师状态。

冻结是设置参数不更新，学生路径仍保留 autograd。不能把学生 SigLIP 或动作后端整体放入 `no_grad`，否则梯度无法传回 stem。教师路径可使用 `no_grad`；若教师输出进入学生损失，使用可安全参与 autograd 的脱离梯度张量。

同时运行 RGB stem 对照，以原教师实际使用的 RGB 图像替换输入；保持 stem 容量、后续初始化、可用因果历史和训练流程尽量可比；B 的 RGB 是取自教师内部成像输出的诊断输入，不取代主教师的 RAW 输入契约。RGB 也失败时先检查 stem 替换、空间对应、预处理和优化，不将问题直接归因于 RAW。

定期检查完整动作采样与纯学生验证表现。stem-only 失败只能说明这一冻结范围／预算下未恢复，不能据此否定整条路线。

### 阶段二 有限视觉适配与动作监督

阶段一出现瓶颈时，先分析训练拟合、留出误差、梯度与动作偏差，再选择一个最小视觉适配：解冻靠近输入的少量 blocks，或在视觉塔中加入小规模 LoRA。不要同时改变多个因素。两种方案都是待检验选择，不预设早期层足够，也不一次启动全部搜索。

使用简单目标：

```text
L = λf · Ltoken + λa · Lnative_action
```

`Ltoken` 同时约束方向与幅值，可采用 FP32 masked MSE 加 cosine；明确归约、有效 token 分母、尺度和系数，不能靠逐样本归一化隐藏投影幅值差异。首轮以最终投影 tokens 为主要目标，不堆多层、重建、感知或额外辅助损失。

`Lnative_action` 使用固定 RawVLA π0.5 的原生 flow-matching 目标和真实专家动作，保留原生归一化、有效动作／未来 mask、时间和噪声采样。冻结语言／动作参数，验证动作梯度确实到达 stem 及本轮声明可训练的视觉层。完整记录真实 7 维和补齐维度，不能用低 32 维均值损失替代动作正确性。

若诊断教师／学生速度场一致性，必须共用 noisy action、时间、语言、状态和噪声；先作为诊断，不自动再加一项训练损失。

首轮主体保留四臂：

| 臂 | 配置 | 回答的问题 |
| --- | --- | --- |
| A | 已验证的完整 RAW 输入教师，含其因果历史与状态，冻结 | 当前能够继承什么能力 |
| B | RGB stem＋继承 SigLIP＋特征蒸馏 | stem 替换与训练流程能否成立 |
| C | RAW stem＋继承 SigLIP＋特征蒸馏 | 纯表征对齐能恢复多少控制 |
| D | RAW stem＋继承 SigLIP＋特征蒸馏＋动作监督 | 动作监督是否补足纯蒸馏 |

原教师 tokens 经过桥接的零更新路径另外作为接口检查，不重复计为训练臂。B／C／D 的后端均冻结。若 D 使用有限视觉适配，C 的对应对照应使用相同可训练范围、起点、数据顺序及总预算；否则只能报告整体方案差异，不能声称收益由动作损失单独造成。阶段切换和共同 warm-start 的更新数纳入每臂累计预算。

### 阶段三 恢复正常光后研究弱光

正常光纯学生闭环恢复后，才提出下一阶段的曝光／噪声／弱光实验和预算。优先考虑退化 RAW 学生对齐同场景、同状态的可靠正常光教师表征，并接受专家动作监督；减弱教师约束，检验学生是否超越教师。真实配对与合成退化分开报告。

严格配对状态、相机、几何、时间及动作语义；只改变成像时可以共享适当动作标签，改变物理状态或正确动作时必须重新标注／规划。正常光 RAW 教师的额外曝光／条件是训练特权信息，纯学生测试不可访问；正常光目标也应经 RAW 教师实际生成，配对 RGB 输入策略只作另列辅助对照。信息完全消失时，不能把学生猜测当成从测量恢复。

单帧、多帧和递归均为可选设计，以闭环效果、稳定性及实测成本选择；不把时序能力推迟为“单帧失败后才允许”的资格门槛。主对照保持可比的因果历史，额外历史的信息收益另行说明。额外数据和预算扩展不自动启动。

## 首轮数据与预算

首个任务沿用已有恢复协议的 RAWVLA-Bench LIBERO Normal `libero_goal/task6`，指令为 `put the cream cheese in the bowl`。执行前核对任务编号规则和实际名称；历史 episode2 只作见证重放，不能替代新的留出初态。

首轮诊断规模沿用 50 条真实专家示范，按源示范拆为 train40／validation5／test5；同源帧、光照变体、退化版本和重采样不能跨分区。首轮 seed101、有效 batch16，每个适配臂**累计最多 5,000 更新**，不将阶段一和阶段二各算一次 5,000。5,000 是诊断上限，不是收敛或能力保证。

checkpoint 记录 0／500／2,000／5,000 及实际提前终止点；若阶段切换，记录 optimizer、参数范围及累计更新。每臂最多 80,000 次 anchor 呈现不等于独立样本数。额外无动作视觉蒸馏数据必须单独登记来源、数量、拆分、呈现次数与预算，不能静默加入。

先实际测前反向显存与吞吐，再登记 GPU 小时和磁盘上限。新 stem、视觉有限适配的学习率及 λ 系数尚未由讨论确定，由你选择最简配方并在运行前冻结理由、数值和选择规则；不照抄旧 ConvNeXt／后端 LoRA 的配置。首轮不做无界超参搜索，预算不足时分析数据、优化范围与初始化，再提出下一步。

离线动作沿用实际确认的协议。已有 LIBERO 恢复协议是 `action_horizon=10`、`flow_steps=10`、`replan_steps=5`，三者含义不同。原生训练目标保持固定，报告完整去噪后的首步、前 5 步、完整 10 步真实 7 维误差，拆开平移、旋转、夹爪及补齐维度。

选 checkpoint 只用验证集：默认沿用前 5 步真实动作 MAE 最小、同分取更早一步；纯学生验证闭环用于诊断，若要参与选型，须在测试前明确规则。可沿用最多 10 个固定验证初态；最终每臂一个选定 checkpoint，在共同的 20 个新测试初态上各运行一次。闭环数量只有取得相应授权后才能执行。

若沿用旧协议的“教师与学生均至少 16/20，且学生少不超过 2 次成功”，明确标为开发筛查，不能据 20 场宣称统计非劣。始终保留完整分母、配对回归与恢复、置信区间、失败原因和视频。教师不满足筛查时先排查原系统，不删困难初态或降低分母。

## 必须独立验收的内容

你按“原始要求→实际代码→真实证据”亲自复核，不能只读取 DeepSeek 的总结。

- **初始化与冻结**：真实 checkpoint 加载审计、继承参数对应、trainable 参数全名／数量、stem 与获准视觉层梯度、实际更新、冻结教师／后端的权重稳定性。补充优化器一步与重载检查时，必须已有对应的更新授权。
- **观测和空间对应**：同源 RAW／教师内部 RGB、裁剪／打包／patch 视野、相机顺序、时间与 mask；教师缓存键覆盖 RAW 历史身份、状态／burn-in、预处理、checkpoint、相机／时刻，修改后不得误用旧缓存。序列窗口按源轨迹拆分，重复 burst 中的同一观测不重复计为独立数据。
- **信息使用**：真实动作标签与观测对齐；零视觉、跨示范打乱视觉等诊断用于判断策略是否只依赖状态。目标位置干预要改变正确动作，避免用固定轨迹的拟合声称视觉控制成立。
- **纯学生路径**：独立评估无需加载教师、ISP、教师缓存或特征混合；无教师回退。用实际调用记录和依赖检查证明，不能只删配置中的名称。
- **行为与成本**：特征误差、动作误差和闭环分开报告；记录同初态教师成功→学生失败、反向恢复与首次偏离阶段。闭环偏离后观测已不同，不能继续作逐步数值归因。实测学生推理 p50／p95 时延、峰值显存、参数量；教师制备／缓存和训练成本另计，不预称更快。

针对新实现运行数值、梯度、配对、空间语义及缓存边界测试；提交前运行 `make check`，必要时运行相关 `uv run --locked pytest` 子集。缺真实资产时使用小夹具只能验证工程契约，真实策略项写 `unavailable`／`not_run`，不要合成成功结果。已有测试中的真实策略跳过项不算通过。

## DeepSeek Harness 的真实派工方法

### 分工与模型

必须实际使用本机 Harness 承担主要编码，而不是列一份委派计划后全部自己实现。默认使用：

```yaml
provider: deepseek-official
model: deepseek-flash
reasoningEffort: max
```

该安装中的展示名为 DeepSeek V41-Flash；Max 是推理强度，不是模型 ID 后缀。不要自动切换 Pro、其他模型或较低 effort，也不要用 Codex 自己的子 agent 代替用户指定的 DeepSeek Harness。

你负责锁定研究边界、关键设计选择、检查点选型规则、数据拆分和独立验收。DeepSeek 负责依据明确接口完成资料／文件清点、stem／教师抽取／训练入口／评估／测试／配置实现、格式修复和授权范围内的命令执行。你可以直接修复少量阻塞问题；主要代码应来自真实派工，不再用 GPT 重写一遍相同实现。

优先一名 Harness 执行者，按可独立验收的模块分轮；仅确有不相交任务时并行。每份任务单明确文件／模块所有权，告知“你不是唯一在此代码库工作的人，保留他人修改，不回退不属于自己的变更”。共享接口由你先定，公共接线与集成顺序由你控制。

### 已核对的本机入口

2026-10-20 已只读核对 `/home/jeong/.local/bin/dsh`，版本为 `0.2.1-rc.2`，支持 headless 的 stdin、`--json`、`--session-id`。随包说明在 `/home/jeong/.local/share/dsh-0.2.1-rc.2/node_modules/@deepseek-ai/dsh-headless/README.md`；实际执行时先检查版本和 `--help`，若安装变化，以本机对应版本文档为准。帮助／配置核对不代表模型 API 已通过调用。

headless 与已有 Web 人工会话分开，不重启或接管 `dsh-web.service`，不使用旧机器人运行适配器冒充编码 Harness。headless 默认不读 GUI 托管凭据，按官方规则读取启动环境／调度目录 `.env`／`$DSH_HOME/.env`。使用现有认证，缺失时报告准确缺项；如需现有托管认证，只按本机文档配置本轮 overlay，不复制密钥进仓库、不打印完整环境或配置、不放宽全局权限。

在仓库创建任务专用覆盖 `configs/research/raw_distillation_harness.patch.yml`，内容如下；若当前工作在获准的其他 checkout，将其中三个工作区路径同步改为该 checkout：

```yaml
- id: agent-default-model
  config:
    provider: deepseek-official
    model: deepseek-flash
    reasoningEffort: max
- id: llm-deepseek
  config:
    reasoningEffort: max
- id: fs-sandbox
  config:
    cwd: /home/jeong/zeno/owac/repo
- id: sandbox-policy
  config:
    workspaceRoot: /home/jeong/zeno/owac/repo
- id: system-prompt
  config:
    personaSuffix: >-
      Your coding workspace is /home/jeong/zeno/owac/repo.
      Read its AGENTS.md before working. Set workdir explicitly for shell calls.
      The launcher directory holds dispatch receipts, not implementation files.
```

从独立的新调度目录启动，避免项目 `.env` 干扰；不要修改用户原 `.env`。每轮使用 `outputs/raw-distillation-<run-id>/harness/round-01/` 等新目录。只读预检命令为：

```bash
/home/jeong/.local/bin/dsh --profile headless \
  --patch /home/jeong/zeno/owac/repo/configs/research/raw_distillation_harness.patch.yml \
  --help
```

只输出非敏感的有效 provider／model／effort／workspace 字段进行核对。若使用 `--dump-config`，不要直接打印整个配置树；它可能含凭据和 `!!js` 标签，按数据解析，不能执行标签内容。实际运行／续接时还要核对会话记录的模型选择，不能仅凭 overlay 默认值断言已有会话已切换。

### 工作单与启动模板

每轮先写 `task.md`，包括：本轮目标、允许写的文件、必须读的规则／前序证据、已确定接口、真实资产绝对路径、可执行命令与环境、预算、交付文件、验收条件、停止条件。不要整段复制聊天和所有历史日志。DeepSeek 不能自行更改研究目标、拆分／测试集、扩预算、解冻后端或进入下一实验阶段。

建议按“资产／权重接口清点→配对 loader 与 stem／教师抽取→蒸馏及动作训练入口→独立评估与集成修复”分轮，具体文件职责由你读代码后确定。复用可用组件；原接口不适配时增加薄层，不把旧 N01／N02／N04 机制接进来。

以下模板由你通过本机终端工具执行。先写好任务单，将第一行参数替换为本轮已创建的绝对目录；返工续接时在目录后追加从事件中读到的准确 session ID：

```bash
python3 - '<本轮绝对目录>' <<'PY'
import datetime, json, os, pathlib, subprocess, sys

repo = pathlib.Path('/home/jeong/zeno/owac/repo')
round_dir = pathlib.Path(sys.argv[1]).resolve(strict=True)
round_dir.relative_to(repo / 'outputs')
os.chmod(round_dir, 0o700)
command = ['/home/jeong/.local/bin/dsh', '--profile', 'headless',
           '--patch', str(repo / 'configs/research/raw_distillation_harness.patch.yml'),
           '--json']
if len(sys.argv) > 2:
    command += ['--session-id', sys.argv[2]]
command += ['-']
receipt = {'command': command, 'launch_cwd': str(round_dir), 'workspace': str(repo),
           'started_at': datetime.datetime.now().astimezone().isoformat()}
with (round_dir / 'task.md').open('rb') as task, \
     (round_dir / 'events.jsonl').open('xb') as events, \
     (round_dir / 'stderr.log').open('xb') as errors:
    child = subprocess.Popen(command, cwd=round_dir, stdin=task, stdout=events,
                             stderr=errors, start_new_session=True)
    receipt.update(pid=child.pid, pgid=os.getpgid(child.pid))
    (round_dir / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    receipt['exit_code'] = child.wait()
receipt['ended_at'] = datetime.datetime.now().astimezone().isoformat()
(round_dir / 'exit.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Harness exit:', receipt['exit_code'], flush=True)
sys.exit(receipt['exit_code'])
PY
```

用终端工具的持续进程会话运行并分段读取进展；工具短时返回不等于 Harness 失败，不重复启动同一工作单。保存 opening `session` 事件的准确身份、有效模型／effort、原始事件、stderr、退出码和最终答复。若已有事件不足以确认选择，读取本轮会话的非敏感选择记录，不能虚构证据。

Harness 退出 0、`final` 或“任务完成”只是执行收据。你必须检查真实 diff，运行关键复核、读取原始输出并写 `review.md`。失败时在新的 `round-02` 写具体返工单，使用 `--session-id` 续接自己创建且空闲的 headless 会话；续接拒绝时保留错误，以完整交接新建同模型会话。不得接管人工 Web 会话，不因超时或无输出擅自重复启动。

本轮模型调用和训练进程分开记账。获准的长训练使用独立、可恢复进程，保存 optimizer／scheduler／RNG／sampler 与 checkpoint，登记本轮拥有的进程、日志、实际 GPU 和监控责任人。你跟进到授权阶段的结果，不把“已启动”当交付，也不停止其他作业。

## 交付与开始执行

工程交付至少包含：新的采用决议与实验配置、真实 teacher／student 与配对数据实现、特征蒸馏及原生动作训练入口、纯学生推理／配对评估入口、关键测试、资产清单及缺口、准确命令、资源画像、Harness 派工记录和你的独立验收报告。可复用代码进入 `src/owac/`，脚本只作入口；大权重／缓存／日志／视频进入忽略目录。配置和入口由你实现后再登记真实命令，不把本文拟议位置宣称为已有 CLI。

若已获准运行实验，补交实际源码版本／diff、完整配置、数据拆分、checkpoint 与实际更新／重载证据、逐 episode 配对账本、曲线、视频路径与失败分析。每个结论标清工程实现、数值对齐、学习、离线动作、闭环能力的证据层级。单任务恢复不能外推四任务、原生 Bayer、跨传感器或超越教师。

最终向用户用自然中文说明：**目标是什么，实际改了什么，训练了哪些参数／多少更新，独立检查了什么，控制能力是否恢复，负结果或缺口是什么，下一步最小行动是什么。** 如果未获训练授权，交付可审阅的实现、通过／未通过／不可测项目和具体启动方案，再一次性说明所需训练范围与资源。

现在开始：按 OWAC-D1.1 核对完整 RAW 教师资产与因果历史契约，读取当前仓库与实际资产，完成你负责的路线／接口判断，写第一轮具体工作单，并通过本机 DeepSeek Harness V41-Flash Max 实际派工。每轮交付由你验收；已授权且证据通过的工程后续继续推进。

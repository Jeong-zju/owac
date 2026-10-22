# DeepSeek N01 v1.2 P3 完整编码器 真实训练与开发闭环

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

本次仅执行 **N01 v1.2 P3**。这是一份完整交接包，共同要求、参考实现、远程训练、前端专家任务和阶段指令均已展开；请从头读完再执行，不需要用户手动拼接其他提示词。准备本包时P3尚未实施，后续阶段的前置交付必须以届时实际记录为准。

## 本次执行的共同要求

你是负责实际编程、训练和实验交付的 DeepSeek。请执行用户选择的一个 v1.2 阶段，持续完成该阶段已授权的工作；只写计划、空壳接口、模拟日志或 smoke 不能算交付。阶段正文在本交接包后半部分。完成后停在该阶段，等待独立验收，不自行启动下一阶段。

### 工作位置和有效依据

- 本地 checkout：`/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`。远程执行时用该 checkout 的当前源码快照；远程没有这个本地绝对路径是正常的。
- 先读 checkout 的 `AGENTS.md`、`README.md`、`docs/status.md`，检查 `git status --short --branch`，保留当前未提交修改。不要新建另一个原型仓库，不修改 N02/N04，不重置用户的文件。
- 正式路线：`docs/N01_full_development_plan_2026-10-15_v1.2.md`，决议：`docs/decisions/0015-n01-v1-2-raw-information-route.md`。P3是完整编码器、同源RAW/RGB四臂主体训练和每轮总100次开发闭环，开发闭环优先当前模型且不要求全矩阵，P4是四臂冻结全量验证，P5是有依据的迁移扩展。原B1标准reset为补充，不默认第五训练臂。来源撤回服务于RAW信息利用，不单独冒充核心任务贡献。旧v1.1三臂全RAW矩阵及v1.0 P3—P8已被取代，不机械沿用旧阶段门槛。
- 继承 P0/P1 已验收资产和 P2 已修补代码。读 `research/experiments/n01_full/P1_acceptance_report.md`、`P2_acceptance_report.md`。P2 修补提交 `fe8d534`、验收记录 `d3dec90`；用户已决定继续推进，原独立复核“未完全通过”的限制保留，**不要求先补齐旧 P2 的合成训练停止条件才能做 P3**。
- P2 的运动训练主要来自程序生成场，不是 RawVLA-Bench 场景训练；其大运动与风险校准反例尚未消除。把已有模型当初始化/对照，在实际完整任务中诊断和改进，不直接宣布 motion 可靠。

### 用户明确要求的工作方式

**新服务器：**`ssh -p 34302 root@154.59.156.21 -L 8080:localhost:8080`。两张96GiB级GPU，默认GPU0跑RAW、GPU1跑RGB，训练及评测都可并行；开发闭环仍每轮总100次。环境/代码由Codex重新部署，读取本包远程段和实际部署记录后复用，不再连接已停用旧服务器。

**开发验证补充：每轮闭环默认总100次，优先当前模型；不要求全部平台/任务/成像条件/四臂都评完。** 跑完报告成功数/分母和失败即可，不按低分自动加跑；其他臂未运行如实写明，完整正式清单属于P4独立安排。

**2026-10-15最新补充：当前项目初期只训练1个种子，默认seed 101。** 两平台raw_base/raw_n01/rgb_base/rgb_n01每个平台、每个方法各1次主体训练；P5的适配/扩展也默认单种子。核心共8个主体训练run；不要自动加入211/307，不以少于三种子判未完成；三种子只保留为后续可选验证，须用户另行安排。此要求覆盖正式方案/历史材料中已被替代的三种子条款。单种子比较仍保持方法间数据、初始化规则和训练机会公平，结论不外推训练随机种子的稳定性。

把精力用于完整模型、真实数据、训练、可视化和行动结果。实现普通必要的错误提示与恢复；不要把“防御性工程”发展成另一项研究。

1. **不新增无依据的硬性通过线。** 不规定“提高 5% 才能下一步”“损失必须下降到某个自创值”“连续八个验证窗口平台”“两次降 LR 才算收敛”，不要求每个参数都有非零梯度，不以测试个数评定研究成功。
2. **不建立严苛哈希准入系统。** run 记源码版本、工作区补丁、有效配置、数据清单、checkpoint 和恢复位置即可。不要对每个 Python AST、每个 tensor、每次 batch、每次页面轮询做哈希；不为补写历史 hash 停止训练。权重传输损坏排查可以做一次普通校验。既有冻结清单的身份记录继续使用，不重新设计复杂签名体系。
3. **不自创总分。** 主结果是机器人成功率、事件后恢复、真实端到端成本；训练 loss、flow 误差、融合权重、支持年龄用于定位原因。禁止把它们加权成“证据健康度”“机制合格分”等无来源指标。
4. **保留有实质意义的正确性检查。** 不跨 train/test 泄漏；不读未来帧/专家未来动作；训练确实更新目标参数；来源完全删除后不能通过 flow、参考帧、归一化或缓存继续影响；恢复不能覆盖旧结果；正式评测保留原分母。这些直接关系结论，不属于可删的装饰门槛。
5. 小网络仅可用于短小工程调试，不作为正式模型或所谓“同名简化复现”。默认复用完整 DBSR、完整 SEA-RAFT 和已接通 π0.5；新增模块说明论文、作者代码、适配处及实际规模。不要照名字手搓 tiny Transformer/Mamba/RAFT。
6. 网络/算法调整后回到固定真实 RawVLA-Bench 案例，输出原画面上的热图和完整动作/开发闭环影响。简单几何信号可以查数值错误，不能成为长期主实验。
7. 常规可逆实现、诊断、已规划训练和恢复可直接推进。资源超过本阶段登记总预算、需要覆盖旧资产、需要新的付费资源或实物执行时，说明具体缺项再请求用户决定。普通负结果、未获正收益、没有“收敛 PASS”不构成停止编程的理由。

### RAW 输入与后端边界

实际主体是作者的三通道 HDR-derived / pseudo-RAW，代码入口称 `rawvla_rgb3`；它不是原生 Bayer CFA。LIBERO 为 agent/wrist 两相机、256×256 三通道整数缓存，RoboTwin 为 head/left/right 三相机、240×320 三通道浮点缓存；以 loader 的真实字段和相机顺序为准。不要因扩展名/变量名含 raw 就宣布传感器原始性。

主链：`RAW 测量 → 成熟内容编码 + 来源支持/修订 → 非图像特征 → 视觉前缀桥 → π0.5 语言/动作后端`。主方法不经过 RGB/sRGB/XYZ 恢复图，不用 RGB 教师蒸馏或 RGB 重建补回原视觉塔能力。可视化预览和独立 RGB 强基线可使用 RGB，分别保存。

继承四区清单 `data/manifests/n01-full-v1/n01-full-v1-index.json`：LIBERO train/calibration/validation/test 为 1228/175/206/162；RoboTwin 为 450/48/58/76，共 2403 条源轨迹。同一源轨迹的所有照明/噪声变体、相机和重叠窗口属于同一区；不重新随机按帧划分。数据源轨迹数和闭环 rollout 数不是一个分母。任务模板跨区共享，因此不声称未见任务泛化。

动作归一化沿用已审计协议：RoboTwin `MEAN_STD` 与 14 维原顺序；LIBERO 使用自己的 7 维统计。缓存 joint target 不能冒称实测 proprioception。采集帧到达节奏、后端推理节奏、action chunk 执行节奏分别记录，不能只修改推理频率就声称历史更多。正确动作会随物理/目标改变的干预需要新标签或重新规划，不能沿用旧专家动作。

### 持续执行和交接

建立小型 `RUN_STATE.md`，记录当前活动、正在运行的 supervisor program、输出目录、恢复命令、已完成矩阵项和下一步。每完成一个有意义的动作或出现错误就更新；长训练由独立进程持续写进度，不依赖对话一直在线。上下文压缩或 SSH 重连后先读它、查现有进程和 checkpoint，再恢复，不能重复开同一训练。

小型报告建议放 `research/experiments/n01_v1_2/P3_report.md`、`P4_report.md`、`P5_report.md`，小型汇总放相应 `research/experiments/results/n01_v1_2/<stage>/`；这些是本轮拟新增位置。大权重、逐帧数组、视频和原始日志留服务器/ignored 输出。修改后运行 `make check` 和实际受影响的测试；不要因改一个网页反复跑全库 GPU 矩阵。既有无关检查失败要记录归属，不能顺手改别人工作。

结尾先给同步案例和任务结果，再给训练覆盖、机制解释、局限、复现命令及原始文件位置。分别说明实现完成、实际训练量、理论命题状态、经验/闭环增量，不用一个总 PASS 掩盖不同问题。保存 diff 供验收；未获用户要求不推送或自动进入下一阶段。

## v1.2 核心主张 同源输入与四臂协议

本段是新路线的核心，不能被历史N/B0/B1全RAW矩阵覆盖。N01要回答“RAW保留了哪项行动信息，怎样积累/保留，动作怎样改善”；可撤回状态、去重和测试通过仅支撑正确性。

### 四臂与真实成本

| 方法ID | 输入 | 主体 |
| --- | --- | --- |
| `raw_base` | 合法RAW | 完整DBSR/SEA-RAFT等成熟内容与融合，共享RAW校准/metadata机会 |
| `raw_n01` | 同样RAW | 完整N01测量支持、实际内容修订与任务读出，主方法 |
| `rgb_base` | 同源RAW经过冻结ISP的RGB | 同规模、历史和后端适配机会的成熟内容/融合 |
| `rgb_n01` | 同样RGB | 对RGB统计合理适配的支持修订，检验通用时序收益 |

两个仿真平台是LIBERO与RoboTwin；四行是方法。当前每平台/方法1个训练种子，默认101，主体共8个run。P4按原每方法13,250正式身份，核心共53,000条。原B1标准检测/reset是补充诊断，不默认第五训练臂；三种子是后续可选，不是当前门槛。不为每次普通修改重训8组，只补受影响项。

**开发期每轮闭环总预算100次，优先当前模型（默认raw_n01），不是每平台/每臂100次。** 两平台均就绪可50＋50，单平台开发可100次；如要本轮比较多个臂，在100次内分配同身份子集并逐臂报真实分母。跑完即给成功率，不要求全任务/全条件/全四臂覆盖，不因低分自动追加。未测臂写未运行，四臂完整闭环不是P3硬门槛；P4正式全量在后续独立安排。本轮预算按一次开发交付累计，不是每个checkpoint或每次验证点各100次。默认先用离线验证选当前checkpoint，再跑100次闭环；若在预算内测试多个checkpoint，按各自身份分别报成功数/分母，不能混成同一模型100次。

### 输入必须同源同条件

离线每个真实源窗口先读取实际RAW `X`，再用同一冻结Default ISP产生 `Y=ISP_fixed(X)`。同源raw/rgb分支共享物理状态、相机、曝光/噪声条件、时间、历史和合法动作标签，噪声不能两边另采成不同观测。记录真实ISP源码位置、参数、dtype/量化、resize/裁剪顺序。复用作者冻结链，不为了胜出选特别糟糕的RGB处理。

**禁止直接拿作者缓存中的配对RGB作为暗光RAW的RGB对照。** 它可能是正常光参考，输入条件不同；必须核对，通常应从本条件RAW重新渲染RGB。RGB构造是独立对照入口，RAW主模型仍不产生RGB中间图像，也不接RGB教师或RGB重建目标。

在线四方法独立闭环，使用相同任务/初态/仿真seed和控制机会；RGB从该方法自己当前状态的RAW接口渲染。动作分叉后后续帧不再同物理状态，不假称整段闭环视频逐像素配对。精确同帧机制比较在离线相同状态窗口上做，禁止偷读别的方法未来画面。

### 对照不得泄漏或故意削弱

每臂motion、噪声/残差、支持和内容只用该臂可见输入计算。RGB臂不读RAW计算的flow、逐像素风险、特征、裁剪前数值或隐藏真值；同源标识和合法公开camera metadata可按同等规则提供。RAW-Base与RAW-N01共享测量输入/合法metadata/校准机会，不能把N01独有额外信息全算成机制增量。

RGB噪声有ISP导致的非线性/相关性，用有来源的RGB域残差与校准估计，不能套RAW泊松公式让它必然失败。三个通道相同shape不代表物理单位相同；新增明确的 `rendered_rgb3` 入口或等价表示标记，不把RGB假标RAW。RAW的 `rawvla_rgb3` 也不是CFA。

受控四臂使用同规模成熟骨干、同token/历史、同π0.5后端开放范围和合理搜索机会。初始化尽量同规则，域专用预训练差异明确计入成本；原预训练RGB视觉塔Default/RawVLA是独立强系统参照，不与受控四臂混作同初始化对照。使用原RGB系统作参照不授权其输出或教师特征进入RAW主链。

### 测量信息必须走到动作

主方法不只学习attention。校准信号/噪声、实际采样位置、相关性、对应风险与变化证据应改变合法支持和成熟内容融合；逐帧最大值归一化、过早通道平均、读出池化若抹掉微弱细节，要在实际案例中查明。保留平均融合权重、物理可靠性估计与来源干预的不同含义，不将attention命名为后验概率。

任务监督训练内容/桥/允许开放的后端，并检查支持选择是否有实际学习/校准依据；不能只看末端loss下降就说“RAW信息已利用”。用同源RAW与ISP-RGB局部、支持/来源干预及固定随机性的动作差说明信息路径。冻结干预导致分布外退化只能说明敏感性；声称稳定训练收益时用相应匹配重训证据。

分别报告同机制RAW/RGB差、同输入N01/Base差及其条件变化，不建立“RAW增益总分”或正交互硬门槛。只赢RGB-Base不自动证明超过RGB-N01。RAW不优于RGB时如实报告，不修改ISP或挑条件消除负结果；通用机制收益和RAW输入收益可以分别成立。

### 数据范围与主链理论

RoboTwin HDR派生float输入优先研究显示处理前信息，LIBERO RGB8逆处理/uint8缓存不能恢复已丢失传感器信息。两平台均无原生CFA；曝光、gain等字段缺失时用有标记的估计，不凭空填物理真值。真实CFA输入适配保持真实相位/坐标；未用真实数据验证的能力不宣称成立，也不重开合成信号训练前置关卡。

理论主线为：同源固定ISP的信息可用性边界；固定设计下 `MSE=历史偏差²+wᵀΣw` 的噪声—陈旧取舍；实际内容/读出的删除与重建正确性。信息原理属于已有基础，受限N01模型不因RAW输入自动达到最优；自适应权重选择偏差也不因detach消失。BOCPD后验不是长期误报保证。标准原理、拟新增规则、未证明命题和实测结果分别说明，不构建新的“理论PASS”系统。

## 参考实现导航

先读与你正在实现的部分有关的作者源码，再移植/复用，保留许可证和适配说明。这里固定代码版本是为了找到同一实现，不是要求构建一套哈希验收系统。采用新权重时另记权重来源和训练域。论文整网常含 RGB 重建、未来帧或 RGB 初始化，不能直接把它接到主链。

### P3 首选直接复用

**DBSR 内容与融合。** [论文](https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.html)；固定 `16f158e0553b6702cc1693c43f733c2830973278`。读作者 [encoders.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/encoders.py)、[merging.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/merging.py)、[blocks.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/layers/blocks.py)、[warp.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/layers/warp.py)。本仓已 vendor，优先复用 `DbsrFeatureAdapter` 和 `MaskedDbsrFusion`，默认宽度64、9个残差块、输出512；不复制作者原始 RAW→RGB 对齐分支和 RGB 重建头。许可证 CC-BY-NC-SA-4.0，保留作者声明。

**SEA-RAFT 对应。** [论文](https://arxiv.org/html/2405.14793v1)；固定 `9137517ba24e628442aec097d3afe71d03503b75`。作者 [raft.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/raft.py)、[extractor.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/extractor.py)、[corr.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/corr.py)、[update.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/update.py)、[train.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/train.py)。本地 `backends/raw_correspondence.py` 已做三通道/四相位适配，关闭自动 ImageNet 加载和 `/255`；保留完整作者主体。proxy 风险需目标数据校准，不能因用了概率输出就宣布有频率学覆盖保证。

**时间变化与鲁棒更新。** [BOCPD 原论文](https://arxiv.org/abs/0710.3742)、[鲁棒 BOCPD 论文](https://proceedings.mlr.press/v202/altamirano23a.html)；DSM-bocd 固定 `9d05117bc8c6e971c836ee1df0c49e2d2950862b` 的 [bocpd.py](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/bocpd.py)、[models.py](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/models.py)。沿对应轨迹处理创新，不在固定像素上把物体正常移动当结构突变。移植核心数值更新即可，不为旧 NumPy/JAX 依赖重建巨大运行栈；用短真实前缀与作者算法核对。

**顺序统计的理论参考。** [confidence sequences 论文](https://arxiv.org/abs/1810.08240)；[confseq](https://github.com/gostevehoward/confseq/tree/5ffe733ca2447a2e28c2c91f3b00086173f2ab2c) 的 `src/confseq/boundaries.cpp`、`uniform_boundaries.h`。不存在可直接照用的 `boundaries.py`。仅当残差满足相应可预测/条件矩假设时使用；不强制编译此库才能开始完整模型训练。

**π0.5 与 RawVLA。** [RawVLA 论文](https://arxiv.org/html/2609.37530v1)、[作者项目](https://shuhongll.github.io/rawvla/)、[π0.5 论文](https://arxiv.org/abs/2504.16054)。固定 RawVLA 源码 `57d508cb180d0510ac9060358009c0f6ffaac420`：

- [NPZ 数据加载](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py)：只查数据字段，作者 PIL/RGB 输出不能代替本项目 RAW loader。
- [PI0.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py)、[PI05.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py)：视觉前缀、训练和推理接点。
- [OpenPIPaliGemma.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/vlm/OpenPIPaliGemma.py)、[OpenPI_ActionHead.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py)：原生 action flow matching，不另写弱化 action MLP。
- 固定配置版本 `ba9c56b04da2c4931b70d8844710f5928d43775b` 的 [LIBERO pi05.yaml](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/libero/pi05.yaml)、[RoboTwin pi05.yaml](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/robotwin/pi05.yaml)、[LIBERO raw.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/libero/rawvla_bench/raw.py)、[rawvla.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/baselines/rawvla/rawvla.py)：对照冻结控制协议。
- openpi 固定 `15a9616a00943ada6c20a0f158e3adb39df2ccac` 的 [pi0_pytorch.py](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py) 用于理解损失；执行优先现有 StarVLA 桥，不升级后端。

### P5 或确有证据需要的替换

**BSRT。** [论文](https://arxiv.org/abs/2204.08332)；固定 `da8d5154478a1214bccdf7a54c716dad3a015985` 的 [bsrt.py](https://github.com/Algolzw/BSRT/blob/da8d5154478a1214bccdf7a54c716dad3a015985/code/synthetic/bsrt/model/bsrt.py)。作者 L 规模参考 `embed_dim=180`、`depths=[6,8,8,8,8,8,8]`。原实现含 Debayer3x3、SpyNet RGB 归一化/权重下载和作者本机路径，必须拆除/替换这些依赖，保持成熟特征主体；不是原论文整网复现。

**Burstormer。** [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.html)、[作者代码](https://github.com/akshaydudhane16/Burstormer)，从 `Burst Super-resolution/Network.py` 读特征/对齐/融合。主分支代码采用前记录真正 commit。其通道 attention 不能直接画成像素显著性图；不要把之前记录的 Git tree SHA 冒充 commit。

**真实 RAW 视频。** RViDeformer [论文](https://arxiv.org/abs/2305.00767)与固定 `4b3f2744f5d19707e718e735c12f2e531fa64dda` 的 [网络](https://github.com/cao-cong/RViDeformer/blob/4b3f2744f5d19707e718e735c12f2e531fa64dda/models/RViDeformer_arch.py)、[训练](https://github.com/cao-cong/RViDeformer/blob/4b3f2744f5d19707e718e735c12f2e531fa64dda/train_ReCRVD_supervised.py)、[数据说明](https://github.com/cao-cong/RViDeformer/blob/4b3f2744f5d19707e718e735c12f2e531fa64dda/README.md)。ReCRVD 是显示屏重拍数据，官方 90/30 场景划分及 ISO 范围按数据卡核对，不声称自然机器人闭环。另见 [RViDeNet/CRVD](https://github.com/cao-cong/RViDeNet/tree/4615032b010e70eebe038be46a5d5820d43e2ef4)、[AIM 2025 多传感器 RAW 视频](https://arxiv.org/abs/2508.16830)。先确认数据可用、许可和存储，再下载；许可证未核实不复制发布作者代码。

**有需要才添加神经记忆。** Mamba [论文](https://arxiv.org/abs/2312.00752)，固定 `e9594ce1c732d97440f0332fdc43170a2294dbfa` 的 [mamba_simple.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py)、[generation.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/utils/generation.py)。卷积缓存和 SSM 缓存都要处理删除，不保留混入旧源的 hidden state。P3 默认有界精确回放，不强行加入 Mamba。TS-Mamba 的 [作者仓库](https://github.com/QZ1-boy/TS-Mamba)可作近邻；之前 README 所指 `options/train/TSA/train_TSA.yaml` 为404，不能承诺直接运行。

**跨任务读出与视觉解释。** [Perceiver IO](https://arxiv.org/abs/2107.14795)、[作者实现](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/perceiver)可作独立读出参考，不要求为小 probe 搬整个 JAX 栈；[Attention is not Explanation](https://aclanthology.org/N19-1357/)、[Attention is not not Explanation](https://aclanthology.org/D19-1002/)提示做真实来源干预，两文是 NLP 语境，不能直接当作本视觉模型的定理。

完整37项论文与相邻工作继续见正式 v1.2 方案第11节；它们不是33个都必须安装的依赖。优先完成本阶段的完整任务，按具体缺口阅读，避免把文献清单变成新串行关卡。

### v1.2 必须核对的同源成像与信息边界

[Unprocessing Images for Learned Raw Denoising](https://arxiv.org/abs/1811.11127)、[作者代码](https://github.com/timothybrooks/unprocessing)用于理解成像与噪声域，不把unprocessing当恢复真实丢失信息的工具。[Learning to See in the Dark](https://arxiv.org/abs/1805.01934)、[作者代码](https://github.com/cchen156/Learning-to-See-in-the-Dark)提供真实短曝光RAW的背景；其RGB恢复目标不进入主链。[RAW-Adapter](https://arxiv.org/abs/2408.14802)、[作者代码](https://github.com/cuiziteng/ECCV_RAW_Adapter)与[Dr. RAW](https://papers.neurips.cc/paper_files/paper/2025/hash/20eadb6338a53114a6371f102c5ded95-Abstract-Conference.html)界定RAW高层任务与适配的已有边界。

固定RawVLA的 `experiments/simulation/libero/rawvla_bench/raw.py` 和RoboTwin对应Default链是拟复用的ISP定位入口，先检索当前冻结作者源码中的 `default_isp`，读取真实调用、dtype、resize和量化顺序再封装，不能凭本文假造已存在的函数签名。作者配对RGB可能为正常光参考，不能直接当受损RAW的RGB分支。

## 远程训练和长期运行（新服务器，2026-10-15）

### 连接、端口与部署依据

旧服务器已停用。用户当前登录命令为：

```bash
ssh -p 34302 root@154.59.156.21 -L 8080:localhost:8080
```

这条命令将本机8080转发到远端Jupyter。新服务器实查远端8080仍被Jupyter占用，保留它；N01实时监控绑定远端`127.0.0.1:18080`。查看监控请另用：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -p 34302 root@154.59.156.21 -L 8080:127.0.0.1:18080
```

随后打开`http://127.0.0.1:8080/`。两条转发不能同时占同一本机8080；可保留Jupyter连接，监控改成`-L 18080:127.0.0.1:18080`，打开`http://127.0.0.1:18080/`。不要停止Jupyter、关闭现有认证或将监控公开绑定`0.0.0.0`。

进入服务器先完整阅读`/etc/vast-agents-guide.md`及`/opt/supervisor-scripts/README.md`，再检查：

```bash
supervisorctl status
nvidia-smi
df -h /workspace
ss -ltnp 'sport = :18080'
```

当前新服务器为**两张NVIDIA RTX PRO 6000 Blackwell Server Edition，每张97,887 MiB显存**；初查两卡均空闲，驱动580.95.05、CUDA toolkit12.8。根存储约1.5 TiB，初始为空。运行时重测，不把初查空闲当专用GPU保证。`/workspace`没有独立持久卷，回收/销毁会丢失数据；重要源码、配置和恢复点同步到用户已有存储，不自行购买服务。

本轮由Codex实际部署代码和环境，部署结果见`docs/N01_new_server_deployment_2026-10-15.md`和本目录`remote_snapshot.json`。原服务器只读快照另存`remote_snapshot_legacy_20261015.json`，只作历史，不能据此声称新机已有旧模型/作业。部署日志在`/workspace/owac/rawvla/logs/`，下载/安装作业名以`owac-n01-v12-restore-*`或`owac-n01-v12-install-*`开头。开始研究前读实际部署记录；尚未完成或未恢复的资产如实处理，不重复下载已完成文件。

### 双GPU调度：RAW/RGB训练和验证均可并行

默认**GPU0为RAW队列，GPU1为RGB队列**。每个平台内，GPU0依次跑`raw_base`、`raw_n01`，GPU1依次跑`rgb_base`、`rgb_n01`；空闲时可按已登记任务调整。两个独立单卡作业即可，不为了两卡并行引入DDP、跨机集群或复杂调度器。两平台共8个主体run，每run仍只有seed101。

```bash
# 这是设备分配示例；后接本阶段实际训练/评估命令。
CUDA_VISIBLE_DEVICES=0 "$N01_PYTHON" -m owac.training.n01_v12 <RAW作业参数>
CUDA_VISIBLE_DEVICES=1 "$N01_PYTHON" -m owac.training.n01_v12 <RGB作业参数>
```

`owac.training.n01_v12`是P3要求新增的入口，当前P0—P2快照还没有完整P3训练器，不能把上面模板当已运行命令。指定`CUDA_VISIBLE_DEVICES=1`后，进程内通常使用`cuda:0`；日志同时写物理GPU索引/UUID与进程内设备，避免两作业都挤到物理GPU0。各作业有独立输出、checkpoint和日志，读同一份只读数据，不共写缓存或checkpoint。只要单卡实际显存允许就并行；不要同时把同卡塞满多个完整π0.5，也不要求先占满两卡才允许工作。

**验证同样可以双卡并行。** 每个worker的策略和模拟器分配对应卡，服务端口分开，例如RAW18000、RGB18001；另一平台若同时开服务另选18005/18006，先查空闲。EGL/Vulkan/SAPIEN的渲染选卡可能不随`CUDA_VISIBLE_DEVICES`自动变化，检查实际渲染设备，不虚称已绑定。评测worker各写自己的episode记录，最后按episode身份合并，不能只相加可能重复的计数。 作者RoboTwin `eval.sh`会重新设置`CUDA_VISIBLE_DEVICES`，它的`gpu_id`位置参数必须传物理0/1；不要把进程内`cuda:0`误传成两份外层脚本都使用物理GPU0。新机已实查SAPIEN在`CUDA_VISIBLE_DEVICES=1`时选择PCI `0000:9a:00.0`（物理GPU1），但新建worker仍应确认实际设备。

**开发期仍是每轮总100次闭环，不是每张卡100次。** 调度前保存一份100条总清单，将不同episode身份分到两卡；若比较多个方法，每个方法实际rollout也计入这100次，各自报告成功数/分母。允许两卡评估同一当前模型的不同episode。训练/评测进程数不改变验证预算，不因并行自动跑完整四臂或启动P4。P4另按正式冻结清单拆分，原分母和正常失败均保留。

### 代码、环境和资产位置

| 用途 | 新服务器路径或约定 |
| --- | --- |
| 当前N01源码快照与提示词 | `/workspace/owac/rawvla/owac`，包含当前checkout未提交的新文件 |
| RawVLA作者源码 | `/workspace/owac/rawvla/source`，固定既有作者版本与子模块 |
| RAW/RGB数据根 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`；LIBERO `libero/raw`、RoboTwin `robotwin2/raw` |
| encoder Python | `/workspace/owac/rawvla/envs/encoder/bin/python`，独立Python3.11/CUDA torch2.9.1环境 |
| policy及同进程特征接入 | `/workspace/owac/rawvla/envs/starvla/bin/python`，独立Python3.10/CUDA torch2.7.1环境；实际兼容性以部署验证为准 |
| RoboTwin模拟器 | `/workspace/owac/rawvla/envs/robotwin`，独立环境，具体可用范围见部署记录 |
| LIBERO权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO统计 | 同上父目录`dataset_statistics.json`，`openpi_pi05_libero_franka` / `action` |
| RoboTwin权重与统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`、同目录`policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 原Default小型记录/有效参数 | `/workspace/owac/rawvla/records/baseline-archive/`，来自本机归档；不是重跑结果 |
| 新run | `/workspace/owac/rawvla/research/n01-v1/runs/v1-2-<stage>-<run-id>/` |

旧服务器P2正式运动checkpoint和大训练目录不包含在本机小型数字归档中。**新机上若没有可恢复权重，就明确写未恢复，不加载本地smoke权重冒充正式模型，也不为恢复旧三种子合成训练而阻塞P3。** 按P3已允许的成熟作者初始化与真实任务训练安排继续，记录实际初始化；不能编造旧checkpoint已迁移。

`RawVLASequenceDataset(root=...)`的默认推导与实际布局不同，显式传正确`raw_cache_root`、`stats_path`和manifest，不因默认路径找不到重下全部数据。新run建`source/`、`config/`、`outputs/`、`receipts/`，复制本任务实际源码、配置、vendor及许可证。仅`git archive HEAD`会漏未提交实现。不要打包`.venv`、700GB数据、密钥或其他方向输出。记录版本与diff，确认实际导入路径及有效配置即可，不建立逐文件哈希准入。

Hugging Face认证已通过服务器私有token存储配置；直接使用Hub客户端默认凭据。**不要把token写进提示词、源码、配置、网页、日志、命令行参数或提交记录，不运行会打印认证值的环境转储。** 需要认证时读取现有私有存储，下载正常完成就继续，不新增token检查门槛。

本地CPU torch锁不得覆盖远端CUDA环境。端到端训练在同一autograd进程导入编码器和固定π0.5，先验证真实forward/backward；可使用已兼容的policy环境或独立兼容环境。**不要通过HTTP特征服务训练端到端梯度**，HTTP只用于闭环推理。Blackwell用cu128或实测可用的wheel，不安装/升级宿主NVIDIA驱动。

### 作业、恢复和简单监控

训练、评测worker、监控分别用supervisor，命名`owac-n01-v12-<stage>-<raw|rgb|monitor>-<run>`。包装脚本前台执行、flush日志；只定向加载本任务program，不重启整个supervisor，不操作管理服务。引用镜像logging工具时按其实际参数调用，再启用严格shell选项，避免工具脚本读取缺失位置参数导致任务在启动前退出。仅SSH私有转发的监控不需要Caddy/portal条目，也不要套缺条目就跳过的guard。

训练保存model/optimizer/scheduler/scaler/RNG/sampler及配置，用临时文件完成后替换。有限队列正常完成后不无限重启；异常恢复接续最近checkpoint，不从step0覆盖。仅单种子101，不自动加211/307。一次短续跑验证即可，不要求跨机器浮点逐位一致。

监控页保持简单：一张两行GPU/作业表、当前阶段、真实step/rollout、曲线、案例和日志即可。RAW/RGBworker各写各自状态，轻量汇总器读它们，不能竞争写同一个JSON；全局显示`已完成/100`及各worker计数。网页关闭不停止训练，训练退出仍显示最后状态与原因，等待GPU如实显示等待；监控失败不杀训练。具体前端专家任务见下一段。

## 立即分派给前端专家的完整任务

主执行者：在主体长任务开始前，把以下任务完整交给可调用的前端专家/子 agent。你负责进度写入、HTTP 服务和训练/评测 hook；前端专家负责页面。双方先约定下面的少量 JSON 字段后可并行，不相互回退文件。若运行环境没有子 agent/专家工具，明确说明并按相同要求自行实现页面，不假称已经调用专家，也不因此搁置监控。

> 你是本次 N01 v1.2 的前端专家。请实现一个中文、单页、只读的实时研究进度页，让用户在 DeepSeek 长对话和远程训练时看见“现在做什么、做到哪里、是否还活着、结果如何”。页面保持简单，优先原生 HTML/CSS/JavaScript，复用仓库 `baseline_dashboard_template.html` 中有用的图表思想，但它目前是静态快照，不可直接宣称实时。不要引入 React 工程、数据库、WebSocket、账户管理、复杂路由、拖拽组件或多个管理后台。
>
> 双GPU只增加一张两行表：物理GPU、RAW/RGB作业、阶段、step/rollout、显存和最后心跳；显示全局开发闭环完成数/100。并行两worker独立记录，不能把每卡100相加成200。不要为两卡增加新页面、数据库或复杂调度视图。
>
> **文件所有权：**你拥有拟新增 `src/owac/runtime/n01_v12_monitor.html`；主执行者拥有 `n01_v12_monitor.py`、进度 JSON writer 及训练/评测 hook。你不是独自在代码库中工作，不得回退其他人的修改。需要接口变化先同步主执行者。
>
> **布局：**顶部显示 N01 / v1.2 / P3 或 P4 或 P5、当前活动的一句中文、运行状态、最后更新时间和数据是否陈旧。下面放四张小卡片：当前方法/平台/训练种子；更新步数或 rollout 完成数；用时/速度/有依据时的预计剩余时间；GPU 显存/利用率。主区域两列：左边真实曲线，右边最新真实案例；四臂固定为raw_base/raw_n01/rgb_base/rgb_n01，按RAW/RGB×Base/N01排成简单2×2表，并标1个训练种子、默认101。底部是简单任务矩阵与最近日志。手机窄屏变一列；文字和坐标单位清楚，默认浅色，无复杂动画。
>
> **状态语义：**支持“编程中、检查中、等待资源、训练中、评测中、阶段结束、失败、需要用户处理”。编码工作总量未知时显示当前步骤和最近活动，不编造百分比。训练百分比只能来自 `step / planned_steps`；评测来自 `completed_rollouts / expected_rollouts`，同时列成功、正常失败、基础设施待恢复。P3开发评测默认本轮总100次，按实际分配显示每组分母，不能自动乘平台/方法；P4正式全量、P5 的迁移实验明确标注。数据来源明确标LIBERO逆处理/量化、RoboTwin HDR派生或真实CFA，不一概写真实传感器RAW。未测量显示“尚未测量”，不填 0；曲线空时说明暂无数据，不能伪造示例分数。预计时间只依已完成速度估算并注明“估计”。
>
> **曲线：**显示实际 train action loss、固定验证 action loss；有开发闭环结果时另画成功率，不与 loss 共用不标注的坐标。P4四臂主矩阵期望总量53,000，附加632另列；P4主要显示覆盖进度和按方法/平台/种子的成功率，运行中的成功率标“暂时结果”，原正式分母始终可见。只用已有指标，不新增综合评分。进度页不判定“收敛通过”。
>
> **案例：**先显示同源RAW固定预览、固定ISP的RGB和局部细节，再显示同一时刻的输入预览、真实历史融合热图/支持区域图及可选视频链接；可切换有限的固定案例。DBSR 通道均值热图标“平均融合权重”，不称后验或因果贡献。尚未生成时显示说明；不在浏览器端读取 RAW 全集或 checkpoint。原始图、图例、相机、episode/frame 和单位可读。日志保留最后约 100 行，错误使用文本渲染，页面不提供任意 shell、启动/停止任务或删除数据按钮。
>
> **接口：**每 3 秒 GET `/api/status`，约每 10 秒 GET `/api/metrics`；请求未结束不叠加下一次，后台标签页适当降频。HTTP 服务另由主执行者提供。最小状态含 `stage`、`state`、`activity`、`updated_at`、`current_run`、`runs`、`resources`、`artifacts`、`recent_log`。字段未知可省略或 null，不设计严格 schema 认证系统。训练卡片使用current_run的method/input_domain/data_kind/platform/seed/step/planned_steps；评测使用 completed_rollouts/expected_rollouts/successes/failures/infra_pending。metrics 返回带实际 step/时间戳的有限点列和明确 metric 名称/单位。
>
> **断连：**请求失败保留最后成功数据，显示“连接中断”和最后成功时间，自动重试；恢复后清除断连提示。状态更新时间超过约 60 秒时提示“进度数据较久未更新”，不自动判训练失败；失败须来自进程/退出状态。提供一个手动刷新按钮即可。
>
> **交付：**给主执行者页面文件、字段约定和真实浏览器验证记录。用一份小样例检查布局可以，但标明仅 UI 调试，正式网页接真实 run 后移除样例。实际检查首次无数据、训练更新、评测更新、服务中断恢复、窄屏布局、图表和案例读取。无需另建复杂前端测试矩阵或性能门槛。

### 主执行者必须完成的接线

前端任务不是把用户留给另一个人的理由。主执行者负责把页面、真实进度和服务接通，并给用户可打开的地址：

1. 拟新增 `src/owac/runtime/n01_v12_monitor.py`，用已有轻量服务或 Python 标准库 HTTP 服务。只绑定 loopback，限制读取本阶段输出目录，提供 `/`、`/api/status`、`/api/metrics` 和图像/视频文件。不要暴露服务器任意文件或把远程 Jupyter 当 N01 页。
2. 在本阶段 run 内写 `monitor/status.json`、`metrics.jsonl` 和小型案例索引。一个轻量汇总进程整合多作业状态；各训练 worker 写自己的文件，避免多个进程争写同一个 JSON。状态文件用临时文件替换，记录指标的真实 optimizer step；梯度累积的 microstep 不能当更新步。
3. 编程阶段由执行者在有意义的进展时更新 activity；训练/评测由实际循环持续更新。约 5–10 秒的轻量心跳和低频 GPU 采样足够，不每个 step 调 `nvidia-smi`，不在 HTTP 请求里加载 torch/model/大日志。metrics 下采样为有限点数，日志读尾部，页面轮询不增加 GPU 负担。
4. 该接口的**建议形状**如下，按实际需要删减字段；不要把样例值作为运行结果：

```json
{
  "stage": "P3",
  "state": "implementing",
  "activity": "接入完整编码器的真实数据训练入口",
  "updated_at": "实际UTC时间",
  "current_run": {"method": "raw_n01", "input_domain": "RAW", "data_kind": "hdr_derived", "platform": "libero", "seed": 101, "step": 0, "planned_steps": null},
  "runs": [],
  "resources": {},
  "artifacts": [],
  "recent_log": []
}
```

5. 拟提供以下 CLI；这是**要求新增的接口，不是当前已经存在的命令**：

```bash
PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.runtime.n01_v12_monitor \
  --run-root "$N01_RUN" --host 127.0.0.1 --port 18080
```

其中 `N01_SOURCE`、`N01_PYTHON`、`N01_RUN` 在本次启动脚本中设成实际绝对路径。按远程执行段用独立 supervisor program 持续运行。先检查远端 `/api/status` 和页面，再经 SSH 转发验证用户入口。优先在第一轮长训练之前上线，并持续复用到 P4/P5；不是最后补一张静态报告截图。

6. 监控失败不应杀掉正确运行的训练；保留日志、修复监控。仅 status 文件存在也不表示训练在运行，需结合进程/队列状态展示。页面关闭或 SSH 断开不影响作业，重新连上可恢复查看。

## P3 本次任务与执行顺序

**现在实际实施 v1.2 P3：一次建立完整 N01，用两平台真实 RawVLA-Bench 充分训练，完成同源RAW/RGB四臂归因、开发闭环、理论说明和信息—动作同步可视化。** 不拆回旧 P3/P4/P5/P7 的局部研究阶段，不等待局部代理分数先过线。这里的步骤只是同一 P3 内部的依赖顺序。

### 第一步 复用已有实现并上线进度页

先检查已有接口，写一页简短的“已有/待新增”映射，随即编程。不要重复实现 P0/P1 的解析参考和 P2 的完整 SEA-RAFT。立即按前面的专家任务分派监控页，同时建立真实 run 状态；在第一轮长训练前让用户能访问网页。

| 已有位置 | 需要继承的内容与实际限制 |
| --- | --- |
| `src/owac/frameworks/n01_full/adapters.py` | `DbsrFeatureAdapter`、`DbsrAdapterConfig`、`FullState`；已有 RAW 内容/运动/桥接，当前 P3 支持逻辑硬关闭，不能只改一个开关就称完整 N01 |
| 同目录 `ledger.py`、`domains.py`、`statistics.py`、`replay.py` | 来源、GLS、域修订和反事实参考；可作为小规模独立数值参考，不把逐像素 Python 对象账本搬进整幅实时训练 |
| 同目录 `motion_builders.py` | 从清除后来源重建 motion/noise 的接点；参考其依赖语义 |
| `src/owac/baselines/burst_features.py`、`_vendor/dbsr/` | 成熟内容与 `MaskedDbsrFusion`，已有真实 `[B,T,C,H,W]` 权重；融合前后依赖都要检查 |
| `src/owac/backends/raw_correspondence.py`、`_vendor/sea_raft/` | 完整 SEA-RAFT RAW 适配；坐标 pull 方向、sensor/packed 单位和风险 proxy 不能混用 |
| `src/owac/raw/rawvla_sequence.py`、`n01_full_manifest.py` | `RawVLASequenceDataset`、`MeasurementWindow`、`collate_sequence_samples` 和四区清单；显式指定真实数据路径 |
| `src/owac/backends/rawvla_features.py` | `VisualTokenPostprocessor`、`RawVLAFeaturePolicy` 和 action chunk 协议；复用原生训练/推理接点 |
| `src/owac/runtime/n01_full_policy_preflight.py` | 已有 `contract/raw_adapter` 真实预检；说明连通性，不是正式完整训练器 |
| `src/owac/training/n01_action.py` | 原型训练器，默认20步/小裁剪/旧工厂，**不能把它换个输出文件名当 v1.2 正式训练** |
| `src/owac/runtime/n01_feature_service.py`、`rawvla_full_eval.py` | 服务与冻结平台客户端的参考；旧 factory/默认 policy 路径须真正替换成完整 N01，不仅修改标签 |

外部保留现有语义：`init_state(batch_size,camera_spec,device)`；`step(frame,embodiment=None,state=...,budget=...,offsets_xy=None)`；`readout(output,token_budget=None)`；`reset(state,batch_mask=None)`；`detach_state(state)`；`state_bytes(state)`。按实际协议类型接入，不另造不兼容平行接口。`encode_window(values,cell_mask,offsets_xy)` 已有的输出只是内容融合；新增支持后的数据必须真的进入训练/服务。

新配置使用 `configs/encoders/n01_full/v1_2/`。建议文件职责如下；可合并简单辅助代码，不为凑表格造空壳：

| 拟新增或扩展位置 | 职责 |
| --- | --- |
| `raw/n01_paired_views.py` | 从同条件RAW生成冻结ISP-RGB；同源配对与显式表示标记 |
| `frameworks/n01_full/support.py` | 张量化空间域候选、沿轨迹的时间相容/变化状态、接受/暂缓/撤回的解释 |
| `frameworks/n01_full/model.py` | 完整模型工厂、支持参与内容通路、32帧重放及基线开关；复用 adapters 而非另写弱骨干 |
| `frameworks/n01_full/adapters.py`、`replay.py` | 因果来源/缓存/重算接线及公共接口，保留历史参考可用 |
| `training/n01_v12.py` | 真实动作训练、匹配矩阵、checkpoint/resume、进度事件；不是 HTTP 服务训练 |
| `evaluation/n01_v12.py` | 固定动作验证、真实来源干预、开发案例和汇总 |
| `runtime/n01_v12_service.py` | 将完整模型接到既有 π0.5/平台服务；真实相机到达、reset和期限快照 |
| `runtime/n01_v12_monitor.py`、`n01_v12_monitor.html` | 前面规定的简单实时监控 |
| `research/theory/n01_raw_evidence_contract_v1_2.md` | 主链依赖、删除命题、噪声/陈旧偏差和顺序保证边界 |

表中的包内路径均以 `src/owac/` 为前缀；不存在的文件是本次要写的文件。

### 第二步 在同一个完整编码器中实现六项功能

**A. 实际测量身份。** 按 episode/camera/真实帧号或实际到达标识登记来源，并保留 channel/phase/实际采样坐标。训练窗口重叠、同一包重发、π0.5 多次 flow denoising 不增加独立观测。用紧凑索引和张量 mask 承担大图；重复项去重不等于不同曝光彼此统计独立。缺失、坏点、饱和分别保留。

**B. 成熟内容和对应。** RAW与RGB两种输入域分别设明确入口；以下物理噪声/采样语义适用于RAW，RGB按共同四臂协议适配，不读取RAW估计结果。 默认完整 DBSR 宽64、9残差块、512特征及成熟融合，不削成小 CNN。三通道输入用已有三通道 stem，不补假第四通道。先对每个来源独立编码再对齐，以便依赖可追踪。SEA-RAFT 估计“当前参考坐标→历史源坐标”的 pull map，明确单位；当前时刻及过去才是合法输入。以真实 RawVLA 轨迹检查运动覆盖与错误区域，保留可视化；P2 checkpoint 可初始化，但不假定它在真实场景已准确。

**C. 空间支持与时间修订。** 用实际映射后的采样位置构造局部常数/仿射 GLS、校准噪声和残差；空间候选采用有限局部区域与多尺度邻域，具体尺度在目标数据开发集上设定并记入配置，不把任意 A–B、B–C 边的连通闭包当整个域相容。实现域级残差/相容性判断。沿对应轨迹维护 BOCPD 或鲁棒 BOCPD 数值状态，使用作者算法，不手造一个“变化置信度网络”取代它。

明确区分：稳定且可对齐的历史、映射不确定、单帧异常、持续结构变化、缺失/坏点/饱和。暂缓可疑来源，积累后续真实反证再接纳或退出；不能把运动速度本身当撤回依据。hazard、噪声/风险尺度和阈值使用 calibration/validation 的实际数据选定；BOCPD 输出是模型后验，不是已证明的误报概率。风险较差时展示并诊断，不通过人工压低风险值让图好看。

**D. 支持必须改变实际内容。** 支持改变进入融合的有效来源、参考构造、logit/softmax 归一化和必要的重新编码，影响最终桥接特征与动作。不能只把 support 用在诊断字段、loss 附项或最后输出 mask。检查卷积感受野、参考帧和权重预测是否已经混入退出区域。精确逐查询像素依赖太昂贵时，以区域和保守感受野界重算/屏蔽，说明分辨率与范围；不声称做到了并未实现的任意像素级独立。

**E. 两种撤回与32帧有界精确回放。**

- 结构支持边退出：该源不再给某结构提供测量支持，但可以留下“曾发生变化”等上下文；保留这种上下文就不能说最终表征完全与该源无关。
- 来源完全删除：删掉整帧或选定来源后，凡依赖它的内容、motion/noise估计、支持决定、参考、归一化、状态和读出均从剩余合法测量重建；这才可以检查全删除后的独立性。
- 默认每 episode/camera 最多32个唯一已到达帧，整个有限窗口重算优先。BOCPD 后验、解析状态、神经缓存都由这个窗口和同一固定初值重建，不能带着窗口之前的 posterior 再宣称精确32帧记忆。更早历史丢失明确报告。
- 窗口内来源删除后将其位置标无效，不从窗口外偷偷补进更老帧；若另做会重选窗口的策略，单独命名并计其输入变化。相机缺帧和 episode reset 可真实执行，batch 中局部 reset 不污染其他成员。
- `fixed_decision_replay` 固定决策只衡量直接路径；`counterfactual_replay` 重新计算受影响决策才能衡量完整删源。不能混报。训练时 optimizer 更新后不复用旧参数算出的特征缓存；使用 checkpointing/相机分块仍要处理真实32帧，不能实际只看8帧却在配置写32。

先完成正确的窗口重算，性能优化留有依据再做。没有证据需要时不加 Mamba、全局依赖图数据库、局部缓存失效框架或无限历史。

**F. 期限读出与行动。** 保留现有512→2048桥，默认每相机256个16×16带mask token，替换原RGB图像塔，使用π0.5原生语言/动作后端。来源状态更新后原子提交完整快照。期限前未产生新合法快照时，可返回带来源时间的最近合法快照或单帧降级；全无有效输入则返回显式不可用，不伪造零动作成功。只提供合法输出的工程语义，不承诺 GPU 的硬实时最坏时延。记录队列、数据搬运、flow、支持、重算、后端、动作提交的总 p50/p95/p99；改变重规划或 action chunk 中断规则属于另一个控制实验。

### 第三步 接通两平台真实动作训练

首轮就包含完整A—F及同源四臂输入/读出，不把RAW信息归因推迟到P5，不能训练完旧CNN后再宣布将来接支持。内部短 fit/debug 是允许的，但正式 N01 始终是完整对象。

1. 使用已有四区真实 RawVLA 数据。先打印有限真实 batch 的字段/shape/相机顺序/动作单位，保存几张固定预览，核对实际相邻时间间隔、运动与暗光条件。从本条件RAW经固定作者ISP生成RGB对照，不用正常光配对RGB。RAW主模型不使用RGB中间图，独立RGB臂按其输入域训练。
2. 接通 `RAW encoder → VisualTokenPostprocessor → π0.5 native action flow matching`，确认 encoder/bridge 的真实参数差与 autograd、冻结后端范围均符合配置。取少量训练轨迹做短 fit；通过真实图像/历史替换干预查看预测是否受视觉影响。若只依赖语言或 state，需要修训练，而不是用 loss 下降当视觉已学会。
3. 优先用动作监督适配完整 SEA-RAFT 与编码器；如 flow 辅助监督确有必要，来自真实 RawVLA 帧的已知空间变换可给有限已知对应，明确这是变换监督，不是自然真实 flow。自然曝光变化、噪声/遮挡未知时，不把两帧差或普通照明配对当真值。无需另设10万步合成预训练前置任务。
4. 主损失沿 π0.5 原生 action flow matching。RAW辅助项仅在噪声独立/曝光校准/目标可识别等条件明确时采用，并保留开关及无辅助对照。禁止 RGB 重建、RGB教师或用未来专家动作作为观测。正常条件与扰动条件有相同标签合法性才共享监督。
5. 先冻结后端训练编码器/桥，再在raw_base/raw_n01/rgb_base/rgb_n01四臂开放相同后端参数范围。具体范围、初始化、optimizer、LR schedule、有效batch、precision、窗口和帧到达间隔写有效配置；不同方法享有相同解冻与合理搜索机会。后端的合法预训练权重可复用，不保留其RGB塔给N暗中提供视觉内容。

**训练预算按真实速度制定，不能用 smoke 冒充充分。** 先测完整分辨率/32帧/两平台实际吞吐和显存，给出一轮核心矩阵的GPU小时估算。初始规划每平台、每方法、每种子约30k—100k真实 action optimizer updates；这是容量匹配的规划范围，不是论文保证或固定合格线。可从50k安排、预留到100k的有记录延长；根据作者训练代码、短开发曲线与资源确定实际值，配置和网页显示其含义，不把2k ISP配方直接称N01训练充分。

用固定噪声/flow时间的验证子集比较 checkpoint，再用普通随机验证检查全分布；报告训练源轨迹覆盖、条件覆盖与采样重复率。验证间隔按成本设置，不能每十步做一次昂贵闭环。以匹配曲线、训练拟合、验证表现、视觉利用和开发闭环共同判断优化是否充分；仍改善时按登记预算延长，未改善但拟合很差时诊断容量/梯度/对接。既不设人为收敛PASS，也不因一次loss反弹或达到规划末步就强称充分。真实预算到期可交付明确不足/负结果，不自动无限续训。

### 第四步 主体训练与本轮100次开发验证

严格执行前面四臂协议：raw_base/raw_n01/rgb_base/rgb_n01。两个平台分别训练，每臂1个训练种子101，共8个主体run；不自动211/307。原B1标准检测/reset可作额外解释性诊断，不默认第五训练臂。各臂有完整成熟内容、同历史与桥/后端适配机会，公开metadata和同域校准对Base/N01一致，RGB不套RAW物理噪声，不偷RAW的flow或支持。

四臂分别回答输入信息、通用机制及其结合的效果。不能只拿raw_n01对rgb_base就宣称RAW专属能力。原RGB视觉塔Default/RawVLA保留为实用强系统参照，预训练差异单列。支持/重算关闭的冻结干预与匹配重训消融分开，不把输入扰动的分布外退化当已证明的训练收益。

先在真实batch确认冻结ISP、RAW/RGB数值域、同帧同噪声、mask/相机/动作标签、梯度与服务。检查RGB对应图像不是正常光参考；以RoboTwin HDR真实任务片段优先展示RAW到显示处理的细节/裁剪/噪声变化，LIBERO逆处理缓存只作实际可用信息范围内的研究。曝光/gain/噪声参数缺失不假造真值，记录估计/proxy及适用范围。保留真实CFA输入规格，合法数据可访问时做有限真实案例；未测的能力不声明完成。

开发闭环独立登记任务/初态/seed清单，与正式确认身份分开；本轮开发闭环默认共100次，优先当前模型raw_n01，不是每平台100次或四臂各100次。两平台均就绪可各50次，只开发一个平台可该平台100次；若本轮做对照，在100次内分配同身份子集并逐臂报实际分母。完成100次就交成功率/失败/少量案例，不要求把全任务、全条件或全部四臂跑完，不因低成功率/区间宽自动追加，不为100次另加复杂验收。未测臂标未运行，其他臂离线/案例证据可先交，四臂完整闭环不是P3硬门槛。

可比较部分共享初态与控制机会；独立动作分叉后不能称后续画面逐帧同状态。正常失败/策略超时是有效结果，不是“还没跑成”；infra缺口有限恢复后如实保留，不无限重跑求100个成功。P3不自动扩为P4正式清单。

在线客户端必须发送全部合法已到达观测，policy多次采样不能重复计帧；RAW臂真正替换RGB塔，RGB臂真正走登记ISP，不能只改日志method名。完整服务用HTTP推理可以复用，端到端训练仍保持同一autograd图。

报告分平台/照明/任务成功率、事件恢复、端到端成本和四臂配对差。固定单训练seed下以episode/任务分组估计区间，不估计跨训练seed方差；rollout seed仍覆盖开发清单。普通修改只补受影响代码/案例/训练臂，不自动8组重训。没有增益可交负结果，不能换ISP/挑样本掩盖。

### 第五步 直接可理解的同步可视化

先在同源RAW固定预览与冻结ISP-RGB上显示实际局部差异、裁剪/量化及图例，再在相同固定真实开发案例中同步输出：输入预览、多相机时间；逐历史来源融合热图；对应点轨迹/warp叠图和放大；支持区域接受/暂缓/退出及历史年龄；真实删源后的特征与动作差；同初态闭环视频和成功/失败。

DBSR权重 `[B,T,C,H,W]` 按固定通道均值聚合，overlay到同一原画面，标注来源时间和共享色标，名称“平均融合权重”。不要归一化每一帧让弱影响也显得火红。对齐残差与有GT的flow error分开，无几何标定不画假真值机器人轨迹。固定随机性进行删源/替换来源的动作干预；低权重高影响、删源不变、失败和随机案例都展示。

固定案例由分层随机、事件与失败样例构成，记录选择规则。任何局部改动重跑这组案例看完整行为，不只贴新loss表。网页引用小图与服务器视频，不每3秒重生成视频或加载完整特征。大数组/视频只留输出，小型索引保留 episode/camera/frame/checkpoint/图例含义。

### 第六步 RAW信息与主链理论说明

写 `n01_raw_evidence_contract_v1_2.md`，将每个命题落到具体算子，不只在解析小夹具证明：

- 信息边界：说明同源Y=ISP(X)的信息可用性基础；这是已有原理，受限N01和有限训练不保证达到最优，也不要求正交互量过线。对应主链图像/读出信息限制干预，不能用理论宣称RAW必胜。
- G1 删除/重算：列实际依赖图，说明完全删除后所有可达依赖如何重建及为何不再受该源影响；区分边撤回、固定决策、完整反事实。在真实图像/完整前缀与动作读出上做可检查反例；这是正确性保障，不是RAW专属贡献；FP误差用合理容差，不要求跨版本bitwise。
- G2 保留历史的条件性收益：固定设计和权重、`y_i=z_i+eps_i`、权重和为1时，写清 `MSE=(sum_i w_i(z_i-z_t))^2 + w^T Sigma w` 的偏差/方差分解及相关噪声假设。自适应权重的数据选择偏差另述；`detach`不消除统计选择偏差。没有可识别相关噪声时仅报告proxy/保守界。
- G3 持续变化的顺序证据：若使用合法e-process/CS，逐项列条件可预测性和条件矩等假设；满足时才引用长期误报保证。若仅BOCPD后验与经验校准，准确标注当前是模型/经验结论及尚缺理论，不包装为无条件保证，不因此把完整任务永久卡在“无法证明PASS”。

创新对照写清：成熟多帧融合、标准检测+reset已经存在；新增假设是来源支持修订怎样真正改变内容和行动，以及相同输入下的机制收益、相同机制下的RAW/RGB差异。负结果同样诚实交付；不写无条件任务性能/有限帧必检保证。

### 要新增并实际跑通的命令接口

以下是拟实现CLI的约定，不是已经可直接运行的旧命令。参数可以按现有项目风格调整，但交付时必须写出真实最终命令和完整配置；不得抄本节后声称已经运行。

```bash
PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.training.n01_v12 \
  train --config "$N01_RUN/config/libero-raw_n01-seed101.json" --output "$N01_RUN/outputs/libero-raw_n01-seed101"

PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.training.n01_v12 \
  train --config "$N01_RUN/config/libero-raw_n01-seed101.json" --output "$N01_RUN/outputs/libero-raw_n01-seed101" \
  --resume "$N01_RUN/outputs/libero-raw_n01-seed101/checkpoints/latest.pt"

PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.evaluation.n01_v12 \
  development --config "$N01_RUN/config/development.json" --output "$N01_RUN/outputs/development"
```

三条仅示例一个矩阵项；实际包含两平台raw_base/raw_n01/rgb_base/rgb_n01各1个训练种子的训练、验证、服务和开发闭环调度，共8个主体run。配置必须明确加载哪个完整工厂，不能默认退回原型。

### P3 交付后停止

给出完整源码与有效配置、可续跑脚本/矩阵、全部run的实际训练步数与数据覆盖、本轮100次开发闭环的成功数/实际分母与未运行项、单种子四臂已完成的离线/案例结果、同源ISP/输入说明和RAW信息到动作的案例、真实案例入口、删源/状态/期限关键验证、理论说明和成本。网页显示“P3阶段结束/尚缺什么”，保留查看。报告未完成项和负结果；训练未完成不能写P3全部完成，性能未提高不伪造PASS。供独立验收后再固定P4候选，本次不自行启动正式全量P4或P5。

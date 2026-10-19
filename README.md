# OWAC

2026-10-15 **N02 v1.2完整阶段提示词已备，代码环境已部署**：[P2—P4入口](../worktrees/n02-measurement-completion/docs/prompts/n02_v1_2/README.md)。当前P2一次建立完整模型/真实π0.5/实际训练与简洁实时网页；P3充分训练整体比较，P4每个实际选中模型共100次闭环。仅seed1103，新服务器`93.91.156.102:51584`双约96GiB GPU，RAW/RGB可并行；环境/权重/双平台原策略迁移检查通过，数据全集后台同步；不要求三种子/全部矩阵/13,250全集/严苛hash或收敛判定。[部署记录](../worktrees/n02-measurement-completion/research/experiments/2026-10-15_n02_v1_2_remote_deployment.md)区分实际完成项与缺口。P1独立验收未通过保留；本轮Codex写提示词并部署，模型P2待实施，未启动长训练。

2026-10-19 **闭环失败有限诊断完成，训练保持停止。** 同四个Normal场景作者预训练策略4/4成功、RAW/RGB均0/4；相同3个Normal专家观测上的完整采样MAE7分别为作者0.0618、RAW0.6149、RGB0.3837。模型在专家观测上已明显偏离，优先查视觉—动作适配；不能仅归因于闭环误差累积。见[诊断与证据](research/experiments/n01_v1_2/P3_action_diagnosis_20261019.md)。

2026-10-19 **按用户最新要求停止RAW/RGB训练与后续队列。** 已禁用自动启动并保留检查点、原始日志及0/12配对小测结果；只有用户重新授权后才可恢复。见[停止记录](research/experiments/n01_v1_2/P3_training_stopped_20261019.md)。下方“已恢复训练”为本次停止前的历史。

2026-10-19 **20k附近RAW/RGB检查点的配对闭环小测完成：两臂各0/12。** 实际权重20,800/21,100步，同一组12场、共24次执行；四个LIBERO suite，ExtremeLow/Low/Normal各4场，没有未完成。两路训练均已恢复并实际推进，监控页显示成功率及正常光同场景回放。见[小测记录](research/experiments/n01_v1_2/P3_20k_closed_loop_probe.md)与[实时监控](http://127.0.0.1:18080/)。这是开发小测，训练重合未知，不作留出泛化或机制优劣结论。

2026-10-17 **按用户最新授权启动完整 N01 训练。** 两张 PRO 6000 分别负责 RAW/RGB，当前 LIBERO、随后 RoboTwin；每 run 50k 步预算，Base 仍留至 P4。新训练使用独立冻结源码与数据清单，不续旧有缺陷的权重。[简洁实时监控](http://127.0.0.1:18080/)只看状态、速度、验证误差和 GPU；[启动、加速与实测记录](research/experiments/n01_v1_2/P3_training_launch.md)。下文“代码补全后暂停”是本次授权前的状态，旧闭环留出撤回仍有效。

2026-10-17 **N01 v1.2 P3代码已补全，训练按用户要求保持暂停。** 完整SEA-RAFT、局部GLS/有界BOCPD、来源重算、统一原生输入与匹配后端适配已接入；八份配置及两平台运行接点齐全。新模型未训练，旧闭环“留出成立”仍撤回，不能标记P3研究完成。当前入口：[代码补全与无更新验收](research/experiments/n01_v1_2/P3_code_completion.md)。下方旧实现缺口及数值保留为历史。


2026-10-16 **N01 v1.2 P3 独立验收未通过，主体队列已保全并暂停，禁止按原队列自动续训。** 已修补断点/RNG、重复测量证据、reset 后验、参考有效性与留出筛选身份错误；旧“0/96 留出成立”撤回。运动对应/真实支持拟合、训练与服务视野及后端适配仍缺；Base 按既有决定留至 P4。详见[独立验收与修补记录](research/experiments/n01_v1_2/P3_independent_acceptance.md)。下方 10-15 的“P3待实施”及 DeepSeek 交付记录为历史。

2026-10-15 N01更换至新服务器`154.59.156.21:34302`，两张RTX PRO 6000 Blackwell Server Edition可并行RAW/RGB训练及验证，每轮开发闭环仍总100次。v1.2三份完整提示词与控制台已同步；实际环境/源码/数据恢复及验证见[新机部署记录](docs/N01_new_server_deployment_2026-10-15.md)。本次迁移未启动P3主体训练。

2026-10-15 **N01更新为v1.2：面向VLA的RAW时空证据编码。** [完整Markdown](docs/N01_full_development_plan_2026-10-15_v1.2.md)、[HTML阅读版](docs/N01_full_development_plan_2026-10-15_v1.2.html)、[DeepSeek完整执行包](docs/prompts/n01_v1_2/README.md)。当前P3待实施；核心为同源RAW/RGB×Base/N01四臂，每平台/方法单种子101，共8个主体训练run；P4核心53,000条。旧v1.1三臂全RAW矩阵为历史，来源撤回服务于信息利用和行动结果。本次同步方案、控制台和提示词，未训练新模型。

2026-10-15 **N01 正式方案更新至 v1.1。** [完整方案](docs/N01_full_development_plan_2026-10-15_v1.1.md)（[HTML 阅读版](docs/N01_full_development_plan_2026-10-15_v1.1.html)）采用“P3 完整模型、真实 RawVLA-Bench 充分训练与开发闭环 → P4 冻结全量验证 → P5 迁移扩展”。来源撤回、时空支持、成熟融合、32 帧有界回放、期限读出及 π0.5 训练整体纳入 P3；真实案例可视化与主链理论为共同交付。v1.0/P2 历史和负结果保留，旧 P3—P8 提示词不再执行。本次只更新方案与入口，未进入 P3、未启动训练或评测。见[正式决定](docs/decisions/0014-n01-v1-1-full-model-route.md)。

2026-10-15 **N01 v1.0 P2 独立验收：未完全通过，代码已修补，未续训。** 新增 10 项反例在原版本全部失败、修补后全部通过；本地 721 passed / 2 CUDA skipped、服务器 375 passed / 0 skipped。修正实际停止点/最终低 LR 窗口、风险分档、遮挡与静态误拒分母、来源/噪声块身份及恢复校准 hash。现有六组仅 CFA seed 307 满足全部平台条件；12 px 新合成诊断误差约 3.3–11.6 px，风险覆盖明显失准。原训练及后端接入证据保留，暂不建议直接按原配方续到 300k；遵照用户要求，本轮正式训练更新为 0。见[独立验收报告](research/experiments/n01_full/P2_acceptance_report.md)。下方 2026-10-14 P2 完成断言为修补前历史，以本次复核为准。

2026-10-14 **N01 v1.0 P2（RAW 运动对应与噪声模型）交付并停在交接点**：新增固定 SEA-RAFT（vendor commit `9137517`，BSD-3-Clause；CFA 4/8 与 pseudo-RAW 3/6 入口，关闭 torchvision ImageNet 自动加载与 `/255`）、解析真值合成运动生成器（平移/旋转/分层遮挡/局部非刚性/重复纹理/曝光/运动模糊 + 亚像素真值）、`Var(y|mu)=a·max(mu,0)+b` + 共享黑电平块噪声与可追溯校准 JSON、`DbsrFeatureAdapter`（显式三通道 stem、16×16 带 mask 池化、共享 `512→2048` 读出、P3 支持硬关闭）与 `n01_full_policy_preflight --mode raw_adapter`、`training/evaluation.n01_motion`（作者迭代损失、fit-check/两档 LR 选择/平台判据/完整恢复、四臂评测与风险分档）。本地全仓 **711 passed / 2 skipped**、`make check` 通过、独立验收对手 **53 项反例 + 16 变异矩阵**；修复 D1（平台判据把恒定/恶化曲线判成平台）、D2（噪声对角分支漏共享块方差）、A（strict `weights_only` 拒绝含 NumPy RNG 的本项目 checkpoint）、B（预检窗口计数调用了不存在的方法）、C/D（estimator 与评测各有多处 CPU/CUDA 设备混用）与 E（预检 stem 配对规则），另有 3.10/3.11/3.12 跨版本 AST 摘要修复。远程 CFA 第一段三种子全部结束：101 `validation_plateau`@82k（val EPE 0.069852）、211 `budget_exhausted`@100k、307 `validation_plateau`@99k（经历 62 分钟实例失联，从 13k 精确续训）；**pseudo-RAW（rgb3）正式臂三种子**（各臂自己的两档 LR 搜索 winner `lr=5e-5/iters=4`）也全部 `validation_plateau`：85k/81k/85k，val EPE best 0.0394/0.0401/0.0405。GPU 评测：CFA test EPE 0.0869/0.0893/0.0950、rgb3 0.0455/0.0426/0.0419（`no_alignment` 1.9213 / 0.9845、`old_simplified` 2.9852 / 8.7080、`oracle` 0.0 单列；两臂输入体制不同，不直接比较），CFA p50 6.79–6.84 ms、rgb3 p50 6.76–7.80 ms，峰值 103.7–108.4 MB。两平台 `raw_adapter` 真实预检（**正式 rgb3 模型** + identity 对照 + 故意错配诊断）通过。P2 GPU 小时合计约 **9.9**。**未进入 P3/P4，无 N01 机制增量或闭环结论**。见 [P2 阶段报告](research/experiments/n01_full/P2_report.md)与[收据](research/experiments/results/n01_full/P2/receipt.json)。

2026-10-14 **N01 v1.0 P1 独立验收修补后通过**。原交付的 34 项测试通过，但新增首批 14 项反例全部失败；已修复保守方差界、秩亏读出、先验坐标、反事实帧级泄漏、实际 32 帧窗口及策略梯度/更新证据。最终本地 495 passed、服务器 147 passed、解析两端 9/9、verify 22/22、153/153 文件哈希一致；两平台真实来源计数与 encoder/bridge 各自更新、后端冻结及 validation 无更新已重验。实现提交 `2d82bbb`。仍停在 P1，未进入 P2、无长训练或机制/闭环结论。见[独立验收报告](research/experiments/n01_full/P1_acceptance_report.md)；下方原 P1 完成记录保留为修补前历史。

2026-10-14 N01 v1.0 **P1（可组合与可撤回的解析参考）完成**：交付身份账本、版本化支持域、逐相位 GLS（秩/可辨识掩码/先验单列/未知相关性保守界）、两套回放接口与 32 帧精确窗口、独立 NumPy FP64 oracle 和一维同语义例子；`analytic` 子命令 9/9、新增 34 项测试、本地全仓 473 passed、`make check` 通过。远程源包 152/152 hash 一致、`verify` 22/22、125 passed，真实两平台唯一来源计数与 `mode=contract` 桥接/冻结证据（后端梯度 0、validation 参数漂移 0.0）通过；检查中修复 7 个缺陷。**未开始 P2/长训练，无机制效果结论**。见 [P1 报告](research/experiments/n01_full/P1_report.md)与[收据](research/experiments/results/n01_full/P1/receipt.json)。

2026-10-14 P0 独立验收修补完成：修正 RoboTwin 协议归一化键、可执行 profiling、严格数据身份检查、GPU 环境路径/依赖、任务层统计与完整预算。两平台四区真实 loader、远程 55/55 checks、139/139 源包、91 项 P0 测试通过；本地全仓 439 passed。仅接受 P0 工程起点，未开始 P1/长训练。 见[独立验收报告](research/experiments/n01_full/P0_acceptance_report.md)。

2026-10-14 N01 v1.0 **P0（问题、数据与依赖）完成**：交付内部契约与阶段配置、四区数据 manifest、上游锁与冻结协议、`n01_full_preflight` 四子命令、T1–T6 理论义务目录；在用户服务器新建独立 encoder GPU 环境（Python 3.11.16 / torch 2.9.1+cu128 / RTX PRO 6000 Blackwell），完成全尺寸 DBSR 64/9/512 profiling（首选起点 128/8：step p50 46.1 ms、峰值 reserved 4.4 GiB）与两平台真实 batch 读取；远程源包 227/227 hash 一致、`inventory --strict` 36/36 checks 通过。本地 `make check` 通过、全仓 434 passed（历史基线 348 + 新增 86）。**未开始 P1 算法、未启动任何长训练、无机制效果结论**。见 [P0 报告](research/experiments/n01_full/P0_report.md)与[收据](research/experiments/results/n01_full/P0/receipt.json)。

2026-10-14：[N01 v1.0 DeepSeek 阶段交接包](docs/prompts/n01_v1_0/README.md)已准备，包含 P0–P8、早期策略预检、代码审计、固定参考实现和远程训练路径。正式开发从 P0 开始；本次仅准备提示词与只读核查，没有启动新训练。

2026-10-14：[N01 全量开发方案 v1.0](docs/N01_full_development_plan_2026-10-14_v1.0.md)（[HTML 阅读版](docs/N01_full_development_plan_2026-10-14_v1.0.html)）已形成；覆盖支持域规则、理论保证的适用条件、成熟模块复用、充分训练及最终 π0.5 / RawVLA-Bench 全量验证。方案尚未实施。

2026-10-13 完成 DBSR 三种子收敛基线（9 组，含 3 组保持训练状态的预算续训）和两层独立冻结干预（CNN 硬拓扑 / 原 RAW 支持域，各 21 组）。基线早期误差：DBSR 1.52998、同容量固定平均 1.75656；同权重强制平均退化 0.55570。原 RAW 域撤回与重算的来源隔离通过，但动态重算仅比旧缓存改善约 0.000995，恢复失败数不变，N01 增量未建立。全仓 348 passed；无新增网络、原核心未改、未改冻结服务器基线。见[完整记录与边界](research/experiments/2026-10-13_n01_dbsr_mechanism.md)。

2026-10-13 已接入作者规模 DBSR RAW 特征层与融合器，完成五组 × 128 步开发验证：普通条件早期误差，容量补齐固定平均 1.96523、DBSR 参考融合 1.99160；当前尚无学习融合的独立收益。保留全部负结果、逐帧恢复和容量控制；N01 核心源码未变，支持图尚未接入新基线。全仓 330 passed，五个全尺寸检查点恢复结果一致。单种子合成验证不代表充分训练或真实后端验收。见[实验与复用记录](research/experiments/2026-10-13_n01_dbsr_features.md)。

2026-10-13 N01 内容通路与六组配对实验完成：18 个正式运行核验通过，C 比同容量当前内容控制 E 改善 0.05575 packed px，D 比 C 差约 0.00686 px；改善较小，全部最优检查点在预算末端，尚未排除训练不足。工程检查 291 passed。逐来源可靠性探索已完成：同权重新留出实验平均退化 0.01446 px，未采纳为正式算法；全仓 314 passed，新规则关闭时 554 张量逐位一致，62 项核心源码检查通过。完整记录见[逐来源实验](research/experiments/2026-10-13_n01_source_reliability.md)，未更新冻结基线或机器人任务。见[本轮记录](research/experiments/2026-10-13_n01_content_integration.md)。

2026-10-12 编码器研究准备：N01、N02、N04 使用三个独立 worktree，完整 DeepSeek 实现提示词、分支路径和启动语见 [研究入口](docs/prompts/README.md)。三个分支共享部署基线及 roadmap v6 / research proposal 快照。N01 已完成七组源码修复及快速运动下历史响应修复，见[最新验收记录](research/experiments/2026-10-13_n01_motion_repair.md)；可观察运动时缩短历史、停止后恢复积累的行为已验证，自动调度相对固定历史的任务收益留待后续研究。N02/N04 进度以各自工作树为准。

2026-10-19 r17顺序实验完成且audit-a2独立重载审计通过：旧Normal接近24窗口，状态→RAW首步6.80→6.39°、前十步7.65→7.24°，完整50步11.14→11.62°；抬升仅2窗口，阶段是闭合命令/双末端升高代理。新Normal固定home、改变物体位置/朝向，RAW/RGB同物理帧，20场景19专家合格（train11/val4/test4），423窗口（244/89/90）；scene019规划/保持失败保留排除、不替换，训练前可行性修订已留档。两种子所选留出关节RMS：固定状态8.17°、状态残差7.90°、RAW冻结6.39°、RAW联合6.49°、原生视觉4.36°。相同初态新留出前十步状态6.46→RAW1.13°、完整50步10.24→7.26°，支持视觉区分动作；RAW提供总体增益，当前后六块/融合联合更新没有额外总体留出收益，不能据此断言骨干硬伤或噪声为主因。审计发现PI0加载污染全局矩阵精度，原联合组固定基线输出漂移；隔离后两组重跑，基线输出及初始化RAW缓存均逐位一致。有效矩阵6144更新＋保留的原联合1024＝实际7168；π更新0、策略闭环0，专家成功不计RAW成功。抓取/抬升无完整50步真实未来，home全部有50；只验证Normal同物体背景的小样本。70项相关测试、make check通过；未提交推送，不重复/扩展训练。监控`/joint`；完整协议、修补、失败记录和结果见`research/experiments/n02_v1_2/residual_joint_r17_20261019/`。

2026-10-19 r16固定状态基线＋有界动作残差对照完成：同r15五条正常光示范、冻结RAW/native tokens，状态/RAW/原生视觉残差各2种子×1024，共6144次更新；前端/π更新0、闭环0。正常光历史留出关节RMS：固定状态11.00°、状态残差10.60°、RAW残差11.39°（较基线差3.5%）、原生视觉残差10.21°（改善7.2%，仅比状态残差低0.39°）。RAW较r15直接头16.60°明显改善，但训练1.20°/验证5.93°未转为Normal留出增益，两种子均如此；不能称视觉泛化已解决。选型保留零修正候选，每64步只用验证MSE选检查点/门控；六组均选强度1，RAW所选256/384、native1024/704，未按测试改门控。单验证episode覆盖仍有限；native为预训练塔＋原光照RGB重放，Over结果不构成成像鲁棒性因果对照。全部保存权重/预测重载与指标逐位复算、缓存及基线哈希/采样隔离通过，64项相关测试和make check通过；作业EXITED0、GPU释放，未提交推送。监控`/residual`，记录`research/experiments/n02_v1_2/residual_r16_20261019/conclusions.json`及`summary.json`。

2026-10-19 r15步数/数据/同结构读出对照完成：固定RAW850空间表示，3/5条正常光示范、256/512/1024预算点，状态/RAW/原生视觉小头各2种子，π各1种子，共14组、14336次真实读出/适配更新（前端更新0、闭环0）。正常光留出在1024预算下：3→5示范，状态12.70→11.00°、RAW19.02→16.60°、原生视觉17.15→13.58°、π20.85→17.37°。π3示范最终训练11.34→4.02°，说明256不足；但视觉增益仍未超过同结构非线性状态对照，部分视觉小头后期拟合回退。原生视觉更好不能独立归因为RAW结构：其预训练及原光照RGB重放不同；本轮也未延长编码器本身训练。119窗口当前状态/未来动作及357 RAW帧逐位对齐；同种子头初值/容量/采样配对，全部预测复算通过。纠正r14“五光照留出”误标：实际3 Normal＋2 Over，历史证据保留。59项相关测试和make check通过，作业退出、GPU释放；未提交推送。监控`/learning`，结果见`research/experiments/n02_v1_2/learning_r15_20261019/summary.json`和`conclusions.json`。

2026-10-19 r14简洁任务主链配对实验完成：RAW850骨干＋多尺度空间融合输出3×16×16空间tokens，绕开候选/测量修订/辅助loss，分别接约200万参数小动作头与残差适配器＋π LoRA。两分支共用逐位相同冻结tokens及采样顺序，32训练/11验证/55历史开发留出窗口按示范隔离，完整50步、offset1、原生32维监督/14维控制、真实padding掩码；各冻结256＋联合256，共1024次更新。冻结所选训练关节RMS小头3.15°、π18.84°；小头联合后留出20.49→18.81°，但纯状态线性对照15.83°更好，视觉留出增益未建立。π联合后无候选改善验证，选择第0步；不能当联合训练收益。四种所选权重均未过严格拟合门槛，闭环0；没有RGB教师或隐藏位置输入。54项相关测试和make check通过，作业停止且GPU释放；未提交推送。监控`/task-chain`，结果见`research/experiments/n02_v1_2/task_chain_r14_20261019/summary.json`及`conclusions.json`。

2026-10-19 r13冻结信息诊断完成：RAW850与最新单示范256权重均冻结，策略更新0、闭环0。新Normal静态快照48场景384帧按32/8/8源episode隔离，另核查原缓存48段各4帧的真实8帧历史；训练226个隔离小诊断头（含3种子、线性/MLP及对照），保存预测独立复算通过。RAW850新快照物体/双夹爪/相对位移误差从浅层7.10/6.81/8.68cm，到融合5.96/6.15/8.42cm，送π前基本相同；开合平衡准确率97.7%。原缓存夹爪位置4.53→5.13→4.04→4.04cm，深层部分退化由融合补回。额外上下文分支新增变化很弱、未见几何读出增益，值得优先核查；不能将失败归为RAW信息全丢或融合压缩。独立原生RGB视觉参照夹爪3.45cm，仍优于RAW；小读出误差不是闭环门槛，也不能区分结构与训练不足。物体真值仅用于隔离诊断；缓存开合是命令代理，EE/TCP分报，不作为运行时输入。单个留出场景冷重采RAW/RGB/标签/状态逐位一致，两RAW模型前后哈希相同。47项相关测试及make check通过，作业均退出、GPU释放；未提交推送。结果见`research/experiments/n02_v1_2/information_r13_20261019/summary.json`及`conclusions.json`。

2026-10-19 r12有界顺序验证已停止于第二步：第一步RAW850固定采样噪声复验400条动作仍失败、持续抓持0秒，原生π参照91条动作通过但不计RAW成功。第二步1条/3条正常光训练示范各256次真实更新，共512次；全部保存点未过拟合门槛。最终最差窗口关节RMS9.0°/27.5°、首条最大偏差38.4°/65.1°、夹爪平均误差0.639/0.831。去噪20/50步与单条组FP32复核均未解决误差。第三步新权重闭环未启动，成败未知；RAW真实成功目标仍未实现。作业均exit0且已停止。修复历史帧缓冲区所有权与恢复验证步号；监控首页显示拟合误差、目标线及明确失败状态。41项相关测试、make check通过；原始更新/样本覆盖/拟合行与检查点的独立复核见`research/experiments/n02_v1_2/sequence_r12_20261019/stage2_completion_audit.json`及`completion.json`。不扩展P3/P4，未提交推送。

2026-10-19 r11本轮闭环完成：采用联合适配臂所选第850步权重，grab_roller/seed0的正常光、弱光、极弱光各1回合，均正常完成400条动作，真实持续抓持均0秒，结果0/3。三回合实际执行与策略预测逐位一致，但动作块交界的最大相邻关节目标跳变分别144.3°/70.6°/100.3°；离线筛选通过尚未转化为闭环抓取。三路float32 RAW、相同初态/指令/采样种子已核验；零训练更新，未扩为正式P4、不提交推送。三段视频见监控页`/closed-loop`顶部；[配置、结果与审计收据](research/experiments/n02_v1_2/closed_loop_r11_20261019/receipt.json)。此前两臂各1000步训练均已完成，第1000步夹爪前缀退化，仍采用第850步。

2026-10-17 用户授权重新训练r11：GPU0任务臂、GPU1冻结RAW/N02前端对照，各1000次更新，seed1103、B12上限，优化器及数据游标重置，沿用已验收RAW-only表征/融合初值，不接续旧失败训练。新运行 `raw-task-r11-task-s1103-a1` / `raw-task-r11-control-s1103-a1` 已由supervisor启动，监控首页已切换；训练前189个源码/配置/页面/探针文件与验收快照一致。启动验收已通过：RAW组10更新/对照15更新均有限，完整B12均已跑通，初值指纹及前10次实际采样窗口一致；作业继续后台训练，不能称新RAW抓持成功；P3/P4未启动、未提交推送。见[启动计划](research/experiments/n02_v1_2/training_r11_20261017/launch_plan.json)。

2026-10-17 r11代码修复及零更新验收完成：修复真实训练表231个有效点、33段开场不可达，统一指令，恢复原生32维动作监督，增加可学习全局/多尺度融合并开放状态模块适配；固定权重校准可恢复且反证触发一次重建。原生RGB对照：10步执行两组均400步物理失败，恢复原生50步执行后89步通过持续抓持及2秒保持；故最终配置恢复原生50步、无额外限幅，保留前10步/关节/夹爪门槛。B12真实前后向约78.34GiB；新RAW优化更新0、未训练/未闭环，不计原生RGB成功为RAW成功。P3/P4未启动，未提交推送。见[修复报告](research/experiments/n02_v1_2/P2_repair_r11_20261017.md)。

2026-10-17 r10代码与训练设置复核未通过：定位均衡采样永久遗漏231个有效anchor（38段中33段t0不可达）、训练/闭环任务文字不一致、完整50步离线选型与前10步限幅执行脱节，以及在线校准/反证修订未接通。三份权重×4窗口零更新干预确认提示和内部补齐动作维度影响输出；FP32缓存/联合前向两窗口逐位一致。未改生产模型/训练配置，优化更新0、无新策略闭环，未提交推送。见[详细审查与证据](research/experiments/n02_v1_2/P2_code_review_r10_20261017.md)。

2026-10-17 r9任务表征/适配/验证覆盖修补与有界验证完成：动作监督真实进入RAW后六块和融合读出，验证改为全部5段留出演示×4位置。两组各64步、实际各508次曝光；任务组最佳整体误差下降15.9%但正常光开场退化5.6%、前10步退化22%，全部保存点未晋级；冻结视觉对照所选32步400指令真实闭环仍失败、持续抓持0秒。传感器噪声隔离及新权重复核共108条动作块，确认暗光输出敏感但不等于闭环主因。未增加抗噪loss；训练与诊断已停，P3/P4未启动、未提交推送。见[实施与证据](research/experiments/n02_v1_2/P2_task_adaptation_r9_20261017.md)。

2026-10-16 RAW深层表征两组各400步、两轮多尺度融合对照已完成；全部开发20帧对应匹配82.5%，RAW置空/翻转已明显影响动作。C1后端对照/LoRA各100步后均未通过真实抓持；D1进一步冻结整个RAW/N02前端，仅比较两档后端LoRA，各100步，开发集选中第50/25步权重，各200指令闭环仍失败。四次完成的开发闭环均失败、持续有效抓持0秒，另一次超时不计成败。图像信息恢复不等于任务语义或抓取已学会；当前有界训练与复验均结束，真实RAW成功目标未达成。未提交推送。见[修复与实测](research/experiments/n02_v1_2/P2_representation_and_fusion_r8_20261016.md)。

2026-10-16 RAW与原始π0.5视觉的同观测诊断：r7空间读出被stride16近恒定特征主导（拼接能量99.97%），左右翻转图像最终token仅变0.174%；固定画面换采样噪声，RAW首条关节目标最大变化90.6°，原始π0.5为1.48°。两路语言/状态/种子一致。继承的r5深层异常在r7迁移100步后仍存在，当前真实抓取仍失败；本轮零更新、无新闭环、未改生产算法，训练保持停止。见[视觉差距与证据](research/experiments/n02_v1_2/P2_raw_visual_gap_20261016.md)。

2026-10-16 物理复核确认此前两次“RAW 205步成功”均为误判，正式撤回。原判据触发时滚筒只触桌、没有夹爪接触，继续原预测后掉落；原始success=true记录保留但不计通过。新增持续接触/离桌/高度验收与可选动作衔接约束；同一RAW100权重真实复验400步仍失败，持续抓持0秒。真实RAW抓取目标尚未达成，训练保持停止，本轮优化更新0；未提交推送。 见[误判与抖动复核](research/experiments/n02_v1_2/P2_false_success_and_jitter_20261016.md)。

2026-10-16 辅助loss专项诊断完成：已复算两臂共2000步日志、三份旧权重12组分项梯度及12组激活。RAW900/RGB500策略读出有明显表征坍缩；因果辅助项的大梯度主要在测量解码器，经全局裁剪与Adam历史状态影响动作更新，不能把此前全参数34/252倍理解为共享编码器受压制。RGB700动作路径自身梯度也明显增大。尚未证明哪个目标诱发坍缩，训练消融未运行；实际更新0、未改生产loss或配置、不提交推送。 见[诊断结论与证据](research/experiments/n02_v1_2/P2_aux_loss_diagnosis_20261016.md)。以下为此前快照。

2026-10-16 整个 policy 框架核查：同一正常光初态下原始 π0.5 与原视觉塔＋特征桥均84步成功，整段动作逐位一致。修复公共状态提示14→32维的顺序错误，版本升为 `n02-v12-p2-policy-r6`；旧RAW900/RGB500改用正确提示仍400步失败，不能把该bug当唯一原因。联合训练梯度量级与代理验证指标存在缺口，尚未证明能力；仅5回合诊断与真实零更新画像，不启动新训练/P3/P4、不提交推送。 见[完整核查与证据](research/experiments/n02_v1_2/P2_policy_framework_audit_20261016.md)；监控闭环页已加入成功参照。以下为历史快照。

2026-10-16 **N02 五档光照闭环演示完成**：RAW900/RGB500 各 5 回合、均 0/5，正常光也未抓取成功；未训练，不能据此评价光照鲁棒性或 RAW 优势。已开放[并排视频](http://127.0.0.1:8081/closed-loop)；见[协议与证据](research/experiments/n02_v1_2/P2_lighting_demo_20261016.md)。正式 P4 100 回合未运行，以下为此前快照。

2026-10-16 **N02 P2 r5 两臂各 1000 步已正常结束**，真实恢复验证完成；RGB 后半程退化，不能据最终 RAW 较低误差认定优势。训练执行完成，稳定性与能力仍待验证，P3/P4 未启动。见[完成核查](research/experiments/n02_v1_2/P2_training_completion_20261016.md)。以下为历史快照。

2026-10-16 **Codex 已按用户最新授权接管并启动 N02 P2 修补版训练**：GPU0 RAW、GPU1 同源 RGB，各1000步，r5从头；监控首页只显示状态、速度与固定验证误差。见[运行与调优记录](research/experiments/n02_v1_2/P2_training_live_20261016.md)。以下“训练由用户开启”是上一轮快照。

2026-10-16 **N02 P2 修补版从头画像，执行版本 r4**。本轮修代码与真实零更新前后向画像；训练由用户自行开启。原 P2 失败及旧训练记录保留，不能由画像推断训练稳定性或任务能力。见[修补与画像报告](research/experiments/n02_v1_2/P2_fresh_profile_20261016.md)、[用户训练入口](research/experiments/n02_v1_2/P2_r4_user_training.md)。以下 r2 和“待实施”条目是历史快照。

2026-10-16 **N02 v1.2 P2 独立验收未通过，已修补关键执行/评测缺陷**。完整网络与真实后端有界复验通过；修补版仅保存 2 步诊断点，原 1000 步保留为旧路径证据。修补版 P2 正式训练未完成，P3/P4 未启动。见[独立验收报告](research/experiments/n02_v1_2/P2_independent_acceptance_20261016.md)。

2026-10-15 **N02 v1.2 阶段提示词已备，代码环境已部署**：[P2—P4完整包](docs/prompts/n02_v1_2/README.md) · [正式方案](docs/plans/N02_full_development_v1.2.md) · [部署记录](research/experiments/2026-10-15_n02_v1_2_remote_deployment.md)。单种子1103、每个实际选中模型共100次闭环；新机`93.91.156.102:51584`双约96GiB GPU，RAW/RGB可分卡并行。环境/权重/双平台原策略迁移检查已通过，数据全集后台同步，进度见监控页。P2一次建立完整网络、修复P1、实际训练与简洁实时网页；P3充分训练/整体比较，P4独立确认100次。不设全矩阵、全树hash或收敛门槛。P1独立验收未通过保持原判，当前P2待实施；本轮Codex只编写提示词并部署，未启动新模型长训练。以下旧版条目保留历史语境。

2026-10-15 N02 v1.1 P1 **独立验收未通过**：服务器真实累计5000步检查点存在，但绝对检查未接入、错误先验被丢弃、训练与评测RAW重叠且候选nuisance泄露真伪；单seed/每类961场景也未达到正式口径。独立复算线索双侧正确11/961，原“0误报/0检出”均来自None检查字段，不能据此归因为纯信息不足。暂不具备E1/P2准入条件；未续训或修改生产实现。见[独立验收报告](research/experiments/2026-10-15_n02_p1_independent_acceptance.md)。

2026-10-14 N02 提示词更新：P0—P7 与 E1/E2 统一优先在 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080` 开发。已只读核实远端数据/权重路径；[提示词入口](docs/prompts/n02_v1_1/README.md)包含首次建立远端 worktree、独立环境及交付步骤。P0 已验收，接续 P1；本次仅修改交接文档，未部署远端代码或启动训练。

2026-10-14 N02 v1.1 P0 **经独立修补后通过**：原交付存在部分 parity 被误判通过、任务划分、冻结窗口、归一化、训练 profile 和 FCMAE 稀疏数学缺陷，已修复并实跑。现在17项上游检查全部通过，2403集真实元数据与10个实际窗口核验完成，正式宽度模块200次训练更新已测；全仓640 passed / 6 skipped，目标运行时257 passed / 1 skipped，make check通过。仅P0 implementation/numerical通过，行为和闭环仍not_run；未启动P1或100k训练。见[独立验收报告](research/experiments/2026-10-14_n02_p0_independent_acceptance.md)。

2026-10-13 N02 v1.1开发交接：已完成[DeepSeek分阶段提示词](docs/prompts/n02_v1_1/README.md)，包含当前代码审计、共同实现契约、P0—P7及两次早期π0.5预检、固定源码参考与验收要求。当前交付为提示词，尚未实现新架构或启动正式训练；建议从P0逐阶段执行。

2026-10-13 N02 路线 v1.1：按最终效果选择实现，保留多尺度 RAW 证据与对象级多假设；P2—P4先建立全局求解和正确撤回，P5单独检验局部传播是否值得启用。增加P1后/P2末π0.5早期适配预检，正式训练与全量评测仍按P6/P7执行。分块不是强制表示，局部提速失败不自动否定有独立行为收益的全局机制；完整机制仍须优于成熟强基线。见[更新方案](docs/plans/N02_full_development_plan_2026-10-13.md)。本轮为路线修订，未启动训练或提交。

2026-10-13 最新规划：按用户要求启动完整 N02 路线设计，目标为发现新解释、学习依赖并撤回错误推断链。已完成 30 组文献/实现来源调研与 P0—P7 开发方案，采用有来源的 Base 级观测网络，列明理论适用条件、充分训练预算及 π0.5 / RawVLA-Bench 全量验证。当前交付是方案，尚未启动新实现或全量训练；旧局部调度归档与负结果保留，此前“仅做边界诊断”的后续安排由新方案取代。见 [Markdown](docs/plans/N02_full_development_plan_2026-10-13.md) · [HTML](docs/plans/N02_full_development_plan_2026-10-13.html)。

2026-10-13 用户授权提交上一轮并启动下一轮边界诊断：两个提交为 `8c60a78` / `843be67`，未推送。新固定扰动实验发现双亮点结构不足未被绝对拟合拒绝，末帧两像素位移使方向正确降至8/11/8（各16）。当前调度保持归档，下一步优先校准/配准与局部残差对照，不扩训。见[边界诊断](research/experiments/2026-10-13_n02_refutation_boundary.md)。

2026-10-13 后续路线已执行：重新训练的官方CNN测量基线加RAW首帧差分，在三个种子新留出均通过配对/保先验/强错纠正门槛；条件调度比较中，N02与简单事件缓存的每帧状态逐位相同、纠错及漂移相同，端到端约慢4%，独有收益0/3。按用户退出条件归档当前局部调度，保留测量基线和全部资产；本轮由用户授权提交。见[测量结果](research/experiments/2026-10-13_n02_retrained.md)、[机制比较](research/experiments/2026-10-13_n02_revision_comparison.md)与[0014决定](docs/decisions/0014-n02-scheduler-archive.md)。

2026-10-13 NRE／BNRE 独立评分对照完成：复用原版损失，冻结 tiny RAW stem，三种子×两方法各训练新头 200 步。新留出配对均为 0/16、两方向强错先验纠正均为 0/16，无信息保持均为 16/16；后者在初始头和零证据参考中也通过，不能视为学习收益。两方法开发/留出均 0/3 种子合格，不进入共用评分器的滤波／N02 对比。原核心及绝对拟合/局部反证检查保留，修改已获用户提交授权。见[协议、结果与复现](research/experiments/2026-10-13_n02_ratio.md)。

2026-10-13 N02 外部方法调研：结合联合控制的负结果，优先考虑 NRE／BNRE 直接学习测量证据，CAGrad 作为监督冲突对照，再用外部滤波实现检验局部修订增益。已核对论文、官方代码和固定版本，并运行原版损失组件接口检查；尚未接入 N02 或获得性能提升，不恢复扩训。见[可复用方法、接入限制与验证顺序](research/reviews/2026-10-13_n02_external_methods.md)。

2026-10-13 N02 联合控制完成：保留共同背景监督的差异加权改善了部分头部拟合，但仍有无信息虚假证据。完整 tiny 从三个既有 200 步模型分叉四组，各追加 200 步；几何投影降低了末尾几何误差，四组却都在开发/新留出上为 0/3 种子合格，12 个末检查点配对均为 0/16。独立复算通过，不延长至 800，不扩大模型；生产推理核心与原检查点保持原样。见[联合拟合、监督冲突控制与训练规模决定](research/experiments/2026-10-13_n02_joint_controls.md)。

2026-10-13 N02 隔离诊断：固定正确几何、冻结编码器后，专门差异目标能学到部分方向，但三个种子实际配对正确仅 0/3/0（各 16 对）；保留同一差异并替换为准确共同背景后均为 16/16。第 200 步几何监督与权重监督在 12/12 批次梯度反向；修订损失在自然激活批次可正常反传。六个头部实验均未达到预定标准，不扩大正式训练。见[测量差异与有效梯度隔离](research/experiments/2026-10-13_n02_learning_isolation.md)。

2026-10-13 N02 三契约变体完成定点审计与 tiny 三种子各 200 步。留出纠错/变化恢复均为 0/32；保歧义为 0、9、0/32，线索配对为 3、0、0/32。三个种子均未达到冻结门槛，不延长到 800 步，机制仍未验收。见[契约修订、行为结果与阻塞诊断](research/experiments/2026-10-13_n02_contracts_training.md)。

此前[三瓶颈独立诊断](research/experiments/2026-10-13_n02_bottlenecks.md)在固定旧检查点上确认评分尺度不足、候选标签与测量解释不一致，以及传播降低预测误差却加剧无信息偏向；保留该版本的原始结果。

2026-10-13 N02：已修复源码 review 的六项问题并完成同预算重训，工程审计通过；强错误先验纠正、歧义保持与场景变化恢复仍未通过，机制不标完成。见[修补与对比结果](research/experiments/2026-10-13_n02_repairs.md)。此前真实 NPZ → π0.5 离线连通性证据见[2026-10-12 验收](research/experiments/2026-10-12_n02_encoder.md#独立验收与修补)，本修补版本尚未在服务器重验。N01/N02/N04 的 worktree 与提示词见[研究入口](docs/prompts/README.md)。

发布固定：`owac-rawvla-pi05-v0.1.0` 标记两平台部署跑通、RawVLA-Bench 与对应 π0.5 权重及比较协议固定。LIBERO 全量成功率 58.20%；RoboTwin 全量仍在运行。范围与复用规则见 [0012 冻结记录](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)。

[打开 N01 / N02 / N04 研究统筹台](outputs/research-hub/index.html)：开发树、当前路线、文档阅读与提示词复制的统一入口。日常使用[自动更新入口](http://127.0.0.1:8876/)：刷新页面即在主 repo 执行 `make research-hub`。本机已配置 systemd 常驻服务，开机启动、异常退出自动恢复；离线快照也可直接阅读。详见[使用与维护说明](docs/research-hub.md)。

N04 当前采用 **v1.1**：S2 一次建立完整 RAW 编码器并接入 VLA，新增网络只用官方数据；[正式方案](../worktrees/n04-egomotion-separation/docs/plans/N04_full_development_v1.1.md)与阶段交接从统筹台进入。S0／S1 保留为历史，S2 待实施。DeepSeek各阶段任务采用单种子、每个待评模型100次闭环，并内置简洁实时监控任务；已指定N04专用双GPU服务器，可双模型／验证并行，HF凭据从服务器环境读取。

N01当前采用v1.2：同源RAW/RGB四臂、单种子，完整方案与DeepSeek交接从统筹台进入；四臂卡片为计划规格，P3待实施。

发布固定：`owac-rawvla-pi05-v0.1.0` 标记两平台部署跑通、RawVLA-Bench 与对应 π0.5 权重及比较协议固定。2026-10-14 主清单已完整：LIBERO 58.20%、RoboTwin 15.63%；[完整评测报告](research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)单独留档，原标签不移动。冻结范围见 [0012](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)。

当前和后续实验统一使用 **RAWVLA-bench**，见 [0011 路线决定](docs/decisions/0011-rawvla-bench-main-route.md)。主基线 10,000 + 3,250 条全部完成，另有 632 条原采样条件独立核查。沿用冻结环境、数据、检查点与比较协议；这份本机完整基线可作为后续算法参照，论文数值尚未复现。训练/评测参数与全部数字结果已[本机归档](research/archives/2026-10-14_rawvla_pi05_baseline.md)，后续按同一条件指纹配对比较。

Arena DROID / PolaRiS 评测、场景重建和该衍生集的公开发布已按用户要求终止。旧源码、权重、数据与结果保留；[Arena 历史页面](outputs/arena-droid-baseline/progress.html)及[PolaRiS 历史页面](outputs/polaris-rgb-raw-reconstruction/progress.html)已标注终止，部分结果不作为完整基准或全集交付。定时任务保持关闭。

OWAC 探索可组合、可扩展的新计算框架，以 RAW 编码和机器人闭环作为首个完整实例。π0.5 PyTorch 后端已完成离线部署，RAWVLA-bench 的现成三通道 RAW 按作者来源与处理历史使用；不冒称原始 Bayer CFA。N01 原型与真实特征接入已验收；完整计算框架的机制收益及机器人闭环仍待验证。

OWAC 探索可组合、可扩展的新计算框架，以 RAW 编码和机器人闭环作为首个完整实例。π0.5 PyTorch 后端已完成离线部署，RAWVLA-bench 的现成三通道 RAW 按作者来源与处理历史使用；不冒称原始 Bayer CFA。N02 当前局部调度实现已归档；原学习原型及真实离线接入记录保留，独立测量基线仅完成声明合成条件下的验收，完整机制与机器人闭环能力尚未建立。

工作空间为 `/home/jeong/zeno/owac/`，主仓库位于 `/home/jeong/zeno/owac/repo/`，后续 worktree 统一放在 `/home/jeong/zeno/owac/worktrees/<task>/`。代码、配置、研究决策及实验记录在本仓库的版本历史下维护。

GitHub 远端为私有仓库 [Jeong-zju/owac](https://github.com/Jeong-zju/owac)，主分支为 `main`。

## 先读这些文件

- [OWAC 学术研究方案](docs/OWAC_academic_research_proposal_2026-10-09.md)：面向研究讨论的 Markdown 正文，包含问题定义、核心框架、机制假设、实验设计与参考文献。
- [当前 research roadmap v6](docs/OWAC_research_roadmap_2026-10-08.md)：研究目标、12 个候选家族及证据要求。
- [生物弱光与信息补全综述](research/reviews/2026-10-09_bio_lowlight_inference.md)、[机制重组决议](docs/decisions/0006-bio-inspired-inference.md)：六类机制问题、原始依据与反证实验。
- [Cui 的 RAW 与机器人论文综述](research/reviews/2026-10-08_ziteng_cui_raw_physical_intelligence.md)、[具身增量决议](docs/decisions/0005-physical-intelligence-contribution.md)：已有边界、可吸收经验与新增实验要求。

- [N01 v1.2 DeepSeek执行提示词](docs/prompts/n01_v1_2/README.md)：P3/P4/P5每阶段一份完整包，当前使用P3；包含远程训练与实时监控页的前端专家任务，尚未启动新训练。
- [N01 当前正式方案 v1.2](docs/N01_full_development_plan_2026-10-15_v1.2.md)：完整模型、真实数据、理论、可视化、充分训练和新阶段；[历史方案 v1.0](docs/N01_full_development_plan_2026-10-14_v1.0.md)保留供追溯。
- [学术研究方案](docs/OWAC_academic_research_proposal_2026-10-09.md)：三个核心机制、后端接入与实验设计。
- [当前 research roadmap v6](docs/OWAC_research_roadmap_2026-10-08.md)：研究目标、12 个候选家族及证据要求。
- [RawVLA-Bench 冻结记录](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)、[当前路线](docs/decisions/0011-rawvla-bench-main-route.md)：已跑通的部署、现成 π0.5 与固定评测协议。
- [AGENTS.md](AGENTS.md)：所有 agent 的开发与研究规则。
- [当前状态与下一阶段](docs/status.md)：已完成的基础设施和待开展的研究。
- [RAWVLA-bench 当前路线](docs/decisions/0011-rawvla-bench-main-route.md)、[全量基线](research/experiments/2026-10-10_rawvla_pi05_full_baseline.md)：当前执行入口、数据和固定比较协议。
- [历史仿真 benchmark 优先决议](docs/decisions/0004-simulation-benchmark-first.md)、[选型调研](docs/simulation-benchmark-selection.md)：最新建设顺序、首选任务套件及待验证的版本/数据条件。
- [可运行的 RAW 仿真 benchmark](docs/raw-sim-bench-deployment.md)：Lightwheel-LIBERO 单任务、独立 runtime、原生 CFA 数据及启动命令。
- [π0.5 后端优先部署决议](docs/decisions/0003-pi05-backend-first.md)：用户分工、已完成的后端部署及其决策来源。
- [π0.5 本地部署](docs/pi05-deployment.md)：独立 PyTorch 环境、权重转换、离线推理与 RAW 接入审计。
- [开发说明](docs/development.md)、[架构边界](docs/architecture.md)：环境、目录职责和待验证的接口约定。
- [候选索引与研究模板](research/README.md)：F01—F12 种子卡、实验和评审记录。

## 本地开始

核心开发环境需要 Python 3.11+、Git、uv 和 Make。仓库使用轻量 Python 包作为共享工程入口；具体候选可以采用不同计算形式。π0.5 使用独立的 Python 3.11 / CUDA 12.8 环境，安装与运行步骤见[部署说明](docs/pi05-deployment.md)。

```bash
cd /home/jeong/zeno/owac/repo
uv sync --locked
make check
uv run --locked python -c "import owac; print(owac.__file__)"
```

`make check` 检查格式、静态问题和源码语法；`uv run --locked pytest` 运行部署补丁边界测试。π0.5 的 GPU 与推理检查在独立 runtime 中显式执行。这些检查不代表候选研究假设已得到验证。

## 工作空间与仓库目录

```text
/home/jeong/zeno/owac/
├── AGENTS.md             # 容器入口指引，完整规则随仓库版本管理
├── repo/                 # 主 Git checkout，以下目录树以此为根
└── worktrees/            # 后续并行 checkout 的统一位置
```

从主仓库创建后续 worktree 的方法见[开发说明](docs/development.md)。现有研究 worktree 与分支见[研究入口](docs/prompts/README.md)。

```text
.
├── AGENTS.md             # agent 规则
├── docs/                 # 当前/历史路线图、设计与开发记录
├── research/             # 12 个家族的种子卡、模板、实验记录
├── src/owac/
│   ├── core/             # 最小共享契约与通用类型的预留边界
│   ├── frameworks/       # 候选核心计算，保留内部表示自由
│   ├── raw/              # RAW 采集、校准、数据与拆分
│   ├── backends/         # 读出及现有 VLM/VLA 接入
│   ├── baselines/        # 机制与系统强基线
│   ├── training/         # 共享训练/拟合配方
│   ├── runtime/          # 状态生命周期、预算与执行
│   └── evaluation/       # 机制、架构、闭环及成本评测
├── configs/              # 候选、基线、数据和实验配置
├── scripts/              # 后续命令入口
├── tests/                # 后续单元和集成测试
├── data/                 # 本地数据；仅说明和小型清单入库
├── outputs/              # 本地运行产物，默认不入库
├── third_party/          # 外部实现的来源登记与本地缓存
├── pyproject.toml        # 包和开发工具配置
└── uv.lock               # 可复现依赖解析
```

目录是职责边界，不表示对应功能已经实现。研究卡目前均为 `seed`；主方法不经过 RGB/sRGB/XYZ 中间图像，也不预设任何家族为主线。原始数据、模型权重和大运行产物不纳入 Git。

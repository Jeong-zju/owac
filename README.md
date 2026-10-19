# OWAC

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

发布固定：`owac-rawvla-pi05-v0.1.0` 标记两平台部署跑通、RawVLA-Bench 与对应 π0.5 权重及比较协议固定。LIBERO 全量成功率 58.20%；RoboTwin 全量仍在运行。范围与复用规则见 [0012 冻结记录](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)。

当前和后续实验统一使用 **RAWVLA-bench**，见 [0011 路线决定](docs/decisions/0011-rawvla-bench-main-route.md)。[π0.5 全量基线](research/experiments/2026-10-10_rawvla_pi05_full_baseline.md)已在用户新服务器运行：作者 LIBERO 10,000 + RoboTwin 3,250 条完整清单，附加 632 条原采样条件单独核查；尚无最终成绩。沿用作者环境、数据与现成权重，后续算法按同一冻结协议比较。

Arena DROID / PolaRiS 评测、场景重建和该衍生集的公开发布已按用户要求终止。旧源码、权重、数据与结果保留，部分结果不作为完整基准或全集交付。定时任务保持关闭。

OWAC 探索可组合、可扩展的新计算框架，以 RAW 编码和机器人闭环作为首个完整实例。π0.5 PyTorch 后端已完成离线部署，RAWVLA-bench 的现成三通道 RAW 按作者来源与处理历史使用；不冒称原始 Bayer CFA。N01 原型与真实特征接入已验收；完整计算框架的机制收益及机器人闭环仍待验证。

工作空间为 `/home/jeong/zeno/owac/`，主仓库位于 `/home/jeong/zeno/owac/repo/`，后续 worktree 统一放在 `/home/jeong/zeno/owac/worktrees/<task>/`。代码、配置、研究决策及实验记录在本仓库的版本历史下维护。

GitHub 远端为私有仓库 [Jeong-zju/owac](https://github.com/Jeong-zju/owac)，主分支为 `main`。

## 先读这些文件

- [N01 v1.2 DeepSeek执行提示词](docs/prompts/n01_v1_2/README.md)：P3/P4/P5每阶段一份完整包，当前使用P3；包含远程训练与实时监控页的前端专家任务，尚未启动新训练。
- [N01 当前正式方案 v1.2](docs/N01_full_development_plan_2026-10-15_v1.2.md)：完整模型、真实数据、理论、可视化、充分训练和新阶段；[历史方案 v1.0](docs/N01_full_development_plan_2026-10-14_v1.0.md)保留供追溯。
- [学术研究方案](docs/OWAC_academic_research_proposal_2026-10-09.md)：三个核心机制、后端接入与实验设计。
- [当前 research roadmap v6](docs/OWAC_research_roadmap_2026-10-08.md)：研究目标、12 个候选家族及证据要求。
- [RawVLA-Bench 冻结记录](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)、[当前路线](docs/decisions/0011-rawvla-bench-main-route.md)：已跑通的部署、现成 π0.5 与固定评测协议。
- [AGENTS.md](AGENTS.md)：所有 agent 的开发与研究规则。
- [当前状态与下一阶段](docs/status.md)：已完成的基础设施和待开展的研究。
- [仿真 benchmark 优先决议](docs/decisions/0004-simulation-benchmark-first.md)、[选型调研](docs/simulation-benchmark-selection.md)：最新建设顺序、首选任务套件及待验证的版本/数据条件。
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

从主仓库创建后续 worktree 的方法见[开发说明](docs/development.md)。当前只预留容器目录，尚未创建额外 worktree。

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

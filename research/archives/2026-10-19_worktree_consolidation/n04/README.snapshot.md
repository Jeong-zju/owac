# OWAC

2026-10-19 **N04 v5工程修复完成，双卡正式训练已恢复**：固定特征尺度、限制动态算子扩张、为越界对应保留梯度，整批目标缺失时停止更新；另修复固定dev的数值重现性。467项测试通过、1项跳过，双平台完整真实零更新、8次优化、独立GPU重载和第9次更新通过；正式服务从9继续并保存到10。查看[训练状态／速度／效果](http://127.0.0.1:8090/)与[修复、失败保留和实际回执](research/experiments/2026-10-19_n04_stability_repair_v5.md)。旧故障权重保留，新轮fresh seed101；长期效果、收敛和GPU充分利用尚未证明。下方v4.2运行中为历史。


2026-10-16 **N04 辅助目标修复 v4.2 已执行，双卡新训练已启动**：修正跨序列表征约束、预测初始化、RoboTwin 补边梯度放大和编译反传回退；两平台完整真实模型零更新验证通过，443项本地测试通过、1项跳过。新训练从fresh seed101开始，保留完整网络与batch8，使用200步预热；旧权重和失败原型分别保留。查看[训练状态／速度／效果](http://127.0.0.1:8090/)与[修复、实际更新及限制](research/experiments/2026-10-16_n04_aux_repair_v4.md)。辅助项长期改善、任务效果与GPU充分利用尚未验证。下方v3状态为历史。

2026-10-16 **Codex 已直接接管修补版 N04 训练**，最新用户授权取代此前停止约束。两张 RTX PRO 6000 分别运行 LIBERO／RoboTwin，使用独立修补版 v3、新初始化、官方数据与 seed101；首段预算每平台 2000 次优化更新。监控入口 [训练状态／速度／效果](http://127.0.0.1:8090/) 每5秒刷新，固定开发动作误差与闭环成功率分开显示。源码优化保留完整模型和有效 batch8；历史 checkpoint 与0/100开发闭环保留。见[接管、性能与运行记录](research/experiments/2026-10-16_n04_codex_training.md)。下方“训练停止”是接管前历史。

2026-10-16 **所有 N04 训练已按用户指令终止**（北京时间 10:36）；已保存 checkpoint：LIBERO200／RoboTwin100。当前优先修复双向对应／裁剪自监督、Perceiver K/V、跨观测防塌缩、forecast 活梯度与跨段归约、输入有效性、纯 RKN 重算、停止与重载门禁。修补版 v3 已通过220项测试及双平台真实48帧完整前后向，optimizer更新0；1项本地协议测试因缺依赖跳过。训练保持停止，能力收益仍未建立。见[本轮修复与回执](research/experiments/2026-10-16_n04_v11_s2_code_repair.md)。下方“训练继续”均为停止前历史。

2026-10-16 **N04 v1.1 S2 独立核查与修补**：100 次开发闭环完成、0 成功、最终基础设施错误 0 已按远端逐身份记录核实；完整实现仍有对应训练缺项，效果未建立。补齐 A/B 同源固定 ISP、读出内存优化、确定性 dev 验证与真实反传日志；191 项测试通过，双平台真实 dev 帧四臂前后向通过。现有双卡训练继续使用原冻结快照，修补版独立保存，未进入 S3。见[审计报告](research/experiments/2026-10-16_n04_v11_s2_independent_audit.md)。下方“待实施”为 10 月 15 日路线准备时的历史状态。

2026-10-15 **N04 v1.1 已正式采用**：[完整方案](docs/plans/N04_full_development_v1.1.md) / [HTML 阅读版](docs/plans/N04_full_development_v1.1.html) / [采用决议](docs/decisions/0015-n04-v1.1-route.md)。保留 ConvNeXt-B，以连续 RAW 证据支持 VLA；S2 一次建立完整架构并接入动作学习，新增网络仅用官方数据。当前为 **S2 待实施**，交接从 [v1.1 DeepSeek 完整提示词](docs/prompts/n04_v1_1/README.md)进入：单种子、每待评模型100次闭环，含简洁实时监控任务，已指定N04专用双GPU服务器，可双模型／验证并行，HF凭据从服务器环境读取。下方 v1.0 与 S0／S1 记录保留为历史；S1 局部学习有效，几何方差与完整 RAW／闭环收益仍未建立。

2026-10-14 N04 v1.0 **S0 远端补齐验收通过**：两平台 40 条成对扩展、960 时刻 / 2,880 相机帧，以及 16 因素 / 152 条真实三维渲染诊断；C0–C7 为 8 passed、0 partial，本地与远端各 361 tests passed。实际时序、执行控制、几何支持与输入隔离均已验收，按训练相机重新冻结米制门槛。S1 尚未启动。详见[本轮记录](research/experiments/2026-10-14_n04_v1_s0_completion.md)。


2026-10-14 较早的 N04 S0 **独立验收修补（历史，已由上述远端补齐取代）**：原交付存在在线控制绕过证据等级、时间域比较、D1 假对应/空采集成功、热启动和连续性问题，已修补并通过 349 项本地测试及远端合同复跑。当前严格状态为**核心工程合同通过、完整 S0 数据交付 partial**（C5 仍 partial）。新增 LIBERO 两任务四条独立扩展（32 时刻/64 相机帧），连续执行与独立几何射线核验通过；RoboTwin live geometry 已实测，但历史控制与完整成对训练集仍缺。未进入 S1。详见[独立验收记录](research/experiments/2026-10-14_n04_v1_s0_independent_acceptance.md)。

2026-10-14 N04 [全量开发方案 v1.0](docs/plans/N04_full_development_v1.0.md) / [HTML 阅读版](docs/plans/N04_full_development_v1.0.html)：基于已提交的作者 RKN，制定 S0–S7 数据、正式规模 RAW 测量、动作 / 几何传播、误差建模、动态修订与 π0.5 / RawVLA-Bench 全量训练验收路线。方案区分文献依据、条件理论性质和待验证创新；本轮只交付文档，未新增训练或提交。

2026-10-13 N04 新[同测量对照与独立 RAW 诊断](research/experiments/2026-10-13_n04_matched_tracking.md)：复用现成 FilterPy/作者RKN确认匀速缺失中的动态状态收益；停止/反向代价和噪声分组取舍明确保留。新增N04机制及闭环收益仍未建立，本轮改动未提交，由用户安排提交。

2026-10-12 编码器研究准备：N01、N02、N04 使用三个独立 worktree，完整 DeepSeek 实现提示词、分支路径和启动语见 [研究入口](docs/prompts/README.md)。三个分支共享部署基线及 roadmap v6 / research proposal 快照，本分支 N04 的历史独立修补见 [验收记录](research/experiments/2026-10-12_n04_acceptance.md)及[六项源码修复](research/experiments/2026-10-13_n04_six_fixes.md)，此前的事件分数统一和三初始化复验见 [2026-10-13 记录](research/experiments/2026-10-13_n04_event_semantics.md)；[机制诊断](research/reviews/2026-10-13_n04_mechanism_diagnosis/README.md)发现的批次标签缺陷已按[分阶段重建与留出验收](research/experiments/2026-10-13_n04_staged_rebuild.md)修复：36权重、248测试，定位及召回有局部改善，但匹配消融更好，动态记忆收益尚未建立；旧错误数据结论保留为历史。其他方向在各自 worktree 维护。

发布固定：`owac-rawvla-pi05-v0.1.0` 标记两平台部署跑通、RawVLA-Bench 与对应 π0.5 权重及比较协议固定。LIBERO 全量成功率 58.20%；RoboTwin 全量仍在运行。范围与复用规则见 [0012 冻结记录](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)。

当前和后续实验统一使用 **RAWVLA-bench**，见 [0011 路线决定](docs/decisions/0011-rawvla-bench-main-route.md)。[π0.5 全量基线](research/experiments/2026-10-10_rawvla_pi05_full_baseline.md)已在用户新服务器运行：作者 LIBERO 10,000 + RoboTwin 3,250 条完整清单，附加 632 条原采样条件单独核查；尚无最终成绩。沿用作者环境、数据与现成权重，后续算法按同一冻结协议比较。

Arena DROID / PolaRiS 评测、场景重建和该衍生集的公开发布已按用户要求终止。旧源码、权重、数据与结果保留，部分结果不作为完整基准或全集交付。定时任务保持关闭。

OWAC 探索可组合、可扩展的新计算框架，以 RAW 编码和机器人闭环作为首个完整实例。π0.5 PyTorch 后端已完成离线部署，RAWVLA-bench 的现成三通道 RAW 按作者来源与处理历史使用；不冒称原始 Bayer CFA。本分支N04原型已完成合成机制复验，完整架构及具身收益尚未建立。

工作空间为 `/home/jeong/zeno/owac/`，主仓库位于 `/home/jeong/zeno/owac/repo/`，后续 worktree 统一放在 `/home/jeong/zeno/owac/worktrees/<task>/`。代码、配置、研究决策及实验记录在本仓库的版本历史下维护。

GitHub 远端为私有仓库 [Jeong-zju/owac](https://github.com/Jeong-zju/owac)，主分支为 `main`。

## 先读这些文件

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

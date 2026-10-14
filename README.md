# OWAC

2026-10-14 N01 v1.0 **P0（问题、数据与依赖）完成**：交付内部契约与阶段配置、四区数据 manifest、上游锁与冻结协议、`n01_full_preflight` 四子命令、T1–T6 理论义务目录；在用户服务器新建独立 encoder GPU 环境（Python 3.11.16 / torch 2.9.1+cu128 / RTX PRO 6000 Blackwell），完成全尺寸 DBSR 64/9/512 profiling（首选起点 128/8：step p50 46.1 ms、峰值 reserved 4.4 GiB）与两平台真实 batch 读取；远程源包 227/227 hash 一致、`inventory --strict` 36/36 checks 通过。本地 `make check` 通过、全仓 433 passed（历史基线 348 + 新增 85）。**未开始 P1 算法、未启动任何长训练、无机制效果结论**。见 [P0 报告](research/experiments/n01_full/P0_report.md)与[收据](research/experiments/results/n01_full/P0/receipt.json)。

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

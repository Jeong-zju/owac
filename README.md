# OWAC

[打开 N01 / N02 / N04 研究统筹台](outputs/research-hub/index.html)：开发树、当前路线、文档阅读与提示词复制的统一入口。日常使用[自动更新入口](http://127.0.0.1:8876/)：刷新页面即在主 repo 执行 `make research-hub`。服务用 `make research-hub-serve` 启动；离线快照也可直接阅读。详见[使用与维护说明](docs/research-hub.md)。

发布固定：`owac-rawvla-pi05-v0.1.0` 标记两平台部署跑通、RawVLA-Bench 与对应 π0.5 权重及比较协议固定。2026-10-14 主清单已完整：LIBERO 58.20%、RoboTwin 15.63%；[完整评测报告](research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)单独留档，原标签不移动。冻结范围见 [0012](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)。

当前和后续实验统一使用 **RAWVLA-bench**，见 [0011 路线决定](docs/decisions/0011-rawvla-bench-main-route.md)。主基线 10,000 + 3,250 条全部完成，另有 632 条原采样条件独立核查。沿用冻结环境、数据、检查点与比较协议；这份本机完整基线可作为后续算法参照，论文数值尚未复现。训练/评测参数与全部数字结果已[本机归档](research/archives/2026-10-14_rawvla_pi05_baseline.md)，后续按同一条件指纹配对比较。

Arena DROID / PolaRiS 评测、场景重建和该衍生集的公开发布已按用户要求终止。旧源码、权重、数据与结果保留；[Arena 历史页面](outputs/arena-droid-baseline/progress.html)及[PolaRiS 历史页面](outputs/polaris-rgb-raw-reconstruction/progress.html)已标注终止，部分结果不作为完整基准或全集交付。定时任务保持关闭。

OWAC 探索可组合、可扩展的新计算框架，以 RAW 编码和机器人闭环作为首个完整实例。π0.5 PyTorch 后端已完成离线部署，RAWVLA-bench 的现成三通道 RAW 按作者来源与处理历史使用；不冒称原始 Bayer CFA。候选框架尚未实现或验证。

工作空间为 `/home/jeong/zeno/owac/`，主仓库位于 `/home/jeong/zeno/owac/repo/`，后续 worktree 统一放在 `/home/jeong/zeno/owac/worktrees/<task>/`。代码、配置、研究决策及实验记录在本仓库的版本历史下维护。

GitHub 远端为私有仓库 [Jeong-zju/owac](https://github.com/Jeong-zju/owac)，主分支为 `main`。

## 先读这些文件

- [OWAC 学术研究方案](docs/OWAC_academic_research_proposal_2026-10-09.md)：面向研究讨论的 Markdown 正文，包含问题定义、核心框架、机制假设、实验设计与参考文献。
- [当前 research roadmap v6](docs/OWAC_research_roadmap_2026-10-08.md)：研究目标、12 个候选家族及证据要求。
- [生物弱光与信息补全综述](research/reviews/2026-10-09_bio_lowlight_inference.md)、[机制重组决议](docs/decisions/0006-bio-inspired-inference.md)：六类机制问题、原始依据与反证实验。
- [Cui 的 RAW 与机器人论文综述](research/reviews/2026-10-08_ziteng_cui_raw_physical_intelligence.md)、[具身增量决议](docs/decisions/0005-physical-intelligence-contribution.md)：已有边界、可吸收经验与新增实验要求。
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

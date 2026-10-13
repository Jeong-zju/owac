# OWAC

2026-10-13 N02 外部方法调研：结合联合控制的负结果，优先考虑 NRE／BNRE 直接学习测量证据，CAGrad 作为监督冲突对照，再用外部滤波实现检验局部修订增益。已核对论文、官方代码和固定版本，并运行原版损失组件接口检查；尚未接入 N02 或获得性能提升，不恢复扩训。见[可复用方法、接入限制与验证顺序](research/reviews/2026-10-13_n02_external_methods.md)。

2026-10-13 N02 联合控制完成：保留共同背景监督的差异加权改善了部分头部拟合，但仍有无信息虚假证据。完整 tiny 从三个既有 200 步模型分叉四组，各追加 200 步；几何投影降低了末尾几何误差，四组却都在开发/新留出上为 0/3 种子合格，12 个末检查点配对均为 0/16。独立复算通过，不延长至 800，不扩大模型；生产推理核心与原检查点保持原样。见[联合拟合、监督冲突控制与训练规模决定](research/experiments/2026-10-13_n02_joint_controls.md)。

2026-10-13 N02 隔离诊断：固定正确几何、冻结编码器后，专门差异目标能学到部分方向，但三个种子实际配对正确仅 0/3/0（各 16 对）；保留同一差异并替换为准确共同背景后均为 16/16。第 200 步几何监督与权重监督在 12/12 批次梯度反向；修订损失在自然激活批次可正常反传。六个头部实验均未达到预定标准，不扩大正式训练。见[测量差异与有效梯度隔离](research/experiments/2026-10-13_n02_learning_isolation.md)。

2026-10-13 N02 三契约变体完成定点审计与 tiny 三种子各 200 步。留出纠错/变化恢复均为 0/32；保歧义为 0、9、0/32，线索配对为 3、0、0/32。三个种子均未达到冻结门槛，不延长到 800 步，机制仍未验收。见[契约修订、行为结果与阻塞诊断](research/experiments/2026-10-13_n02_contracts_training.md)。

此前[三瓶颈独立诊断](research/experiments/2026-10-13_n02_bottlenecks.md)在固定旧检查点上确认评分尺度不足、候选标签与测量解释不一致，以及传播降低预测误差却加剧无信息偏向；保留该版本的原始结果。

2026-10-13 N02：已修复源码 review 的六项问题并完成同预算重训，工程审计通过；强错误先验纠正、歧义保持与场景变化恢复仍未通过，机制不标完成。见[修补与对比结果](research/experiments/2026-10-13_n02_repairs.md)。此前真实 NPZ → π0.5 离线连通性证据见[2026-10-12 验收](research/experiments/2026-10-12_n02_encoder.md#独立验收与修补)，本修补版本尚未在服务器重验。N01/N02/N04 的 worktree 与提示词见[研究入口](docs/prompts/README.md)。

发布固定：`owac-rawvla-pi05-v0.1.0` 标记两平台部署跑通、RawVLA-Bench 与对应 π0.5 权重及比较协议固定。LIBERO 全量成功率 58.20%；RoboTwin 全量仍在运行。范围与复用规则见 [0012 冻结记录](docs/decisions/0012-rawvla-pi05-protocol-freeze.md)。

当前和后续实验统一使用 **RAWVLA-bench**，见 [0011 路线决定](docs/decisions/0011-rawvla-bench-main-route.md)。[π0.5 全量基线](research/experiments/2026-10-10_rawvla_pi05_full_baseline.md)已在用户新服务器运行：作者 LIBERO 10,000 + RoboTwin 3,250 条完整清单，附加 632 条原采样条件单独核查；尚无最终成绩。沿用作者环境、数据与现成权重，后续算法按同一冻结协议比较。

Arena DROID / PolaRiS 评测、场景重建和该衍生集的公开发布已按用户要求终止。旧源码、权重、数据与结果保留，部分结果不作为完整基准或全集交付。定时任务保持关闭。

OWAC 探索可组合、可扩展的新计算框架，以 RAW 编码和机器人闭环作为首个完整实例。π0.5 PyTorch 后端已完成离线部署，RAWVLA-bench 的现成三通道 RAW 按作者来源与处理历史使用；不冒称原始 Bayer CFA。N02 当前为可训练原型，其机制收益与机器人闭环能力尚未建立。

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

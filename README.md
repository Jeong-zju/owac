# OWAC

2026-10-22 全量实验实时修订：教师2163源/74129窗口已全部生成native0。首次学生作业至少10更新后被容器116GiB配额OOM杀死，证据/初始模型/日志完整归档，不重算教师、不声称恢复未保存优化器。已由同一DeepSeek修复RAW缓存配置接线并改用每卡16源工作缓存，保持global16及完整历史，无额外内存/显存/时间停止门槛；第1实际更新即完整保存。Sol独立复核，累计42相关功能测试及make check通过，修复后作业从相同初始化重新启动待实测。

2026-10-22 最新用户授权：在现有SSH服务器新开全量RawVLA-Bench蒸馏实验，不改学生/教师/损失算法，训练整个学生视觉头。只保留新增硬盘380GiB上限；用户明确撤销GPU时间、显存、利用率等人为门槛及强制停止要求。全量只读共享数据已清点：2403轨迹（LIBERO1771、RoboTwin632）、408075帧、5照明；源轨迹拆分1923/240/240，先训练3遍训练集，global16双卡各8。旧实验/预算/结果保留，新实验不受旧3116更新上限约束；无动作损失/后端训练/闭环/实机授权。当前全量数据/双三相机接口由同一DeepSeek V41-Flash Max实现并经Sol独立复核，40项功能测试及make check通过；双卡教师目标已真实生成中，学生仍0更新，已配置目标完成后自动衔接3遍训练，无终止失败自动重试。监控http://127.0.0.1:8880/，SwanLab新run owac-all-bench-20261022。不能宣称充分训练或控制恢复。实验根outputs/raw-distillation-all-bench-20261022，临时tokens缓存/root/autodl-tmp/owac-all-bench-20261022。


2026-10-22 当前阶段已完成：全视觉头428579952参数累计3116新更新/49856anchors，完整checkpoint_final_003116，native0；此前阶段合计达到原5000更新等价上限。global16=双卡各8真并行，accum1；每步中位0.454秒，吞吐约旧串行4.5倍。当前蒸馏训练/教师缓存/状态重载/导出/配对评估链路的hash与无意义扫描/冻结审计/数值门槛已清理并同步服务器，123功能测试+makecheck通过。SwanLab私密OWAC已完整同步并结束：https://swanlab.cn/@Jeong/OWAC/runs/owac-fullvisual-shared-stage 。最终train loss0.07365，最佳val1.00053@1000、最终val1.00837，未作test/动作诊断/闭环，不宣称泛化或控制恢复。无预算扩展或定时任务重建。详见research/experiments/raw_stem_siglip_20261020/batch8-cleanup-20261022.md。

2026-10-22 当前运行已切换真batch8：从858完整状态恢复859并持续超过1250步，全局16=2rank×8、accum1，FP32权重/Adam+BF16 autocast。移除每步全参数/梯度审计及RAW范围/finite全扫描，status/GPU gather每10步。稳定host每步中位0.451秒（此前约2.04秒），25秒GPU均值85%/84%，不是收敛/控制结论。SwanLab私密OWAC已上线且持续回填/更新：https://swanlab.cn/@Jeong/OWAC/runs/owac-fullvisual-shared-stage 。导出/teacher-cache冷路径hash残留继续清理，本轮无新GPU benchmark/准入门槛。原3116新阶段/累计5000更新等价和资源预算不变，旧定时任务不恢复。

2026-10-22 用户明确要求清理蒸馏整条代码链的无意义防御性编程：取消Sol附加的数值阻断门槛，停止反复GPU对比测试；移除哈希、重复张量/参数/梯度扫描与冗余审计，不另造预检门槛。保持训练数学、因果/reset、完整恢复和资源上限。当前训练安全暂停于858更新，下一步使用双卡每rank8真batch、全局16和原BF16恢复；SwanLab私密OWAC项目监控并行接入。未授权其他研究阶段或预算扩展。

2026-10-22 最新批处理授权：保持全局batch16，优先双卡每rank8样本真正并行。旧串行路径已在858更新完整保存并native0退出；优化器、采样器、累计预算保留。DeepSeek V41-Flash Max已实现并通过8项工程测试和make check；真实双卡零更新数值/梯度/显存/吞吐验收进行中，未通过前不宣称切换完成。无新闭环、动作损失或预算扩展。

2026-10-22 哈希清理及性能修复已实测验收：当前训练执行路径不再计算/校验哈希，保留形状/有限值/相机/因果/reset及恢复计数检查。55步完整暂停后，双卡55→56更新/保存和重载验收均通过，连续训练已恢复并超过94步；采样器/优化器/预算未重置。RAW10源RAM缓存、静态packet每rank≤1.5GB缓存启用。稳定每步中位18秒→约2秒，两卡35秒平均利用率约8%/24%→44%/47%；不宣称持续满载。监控http://127.0.0.1:8879/。详见research/experiments/raw_stem_siglip_20261020/throughput-nohash-20261022.md。


2026-10-22 用户最新明确要求彻底移除本轮蒸馏链路中的所有哈希计算及校验。该明确指令覆盖历史资产/权重/状态/缓存/检查点及源码启动哈希要求；保留直接张量形状/有限值、相机/时间/分区/reset和优化器/采样器恢复检查。历史记录不重写，旧hash字段仅兼容读取而不校验。训练为性能修复已在新55步/880anchors完整保存并正常退出；从55恢复，不重算、不重置预算。实现由同一DeepSeek V41-Flash Max负责，Sol独立验收；未完成清理前不宣称已移除。


2026-10-22 15:10 远端整视觉头连续蒸馏已实际运行：从独立验收的全新step1完整状态恢复，实测已到新阶段4步/64训练anchors；Sol核对torchrun PID55499/startticks17190994/PGID55499及两rank进程，native exit仍为unknown（存活中）。严格双卡重载native0且0额外更新通过。仅共享RAW→新教师targets→全学生视觉头428,579,952参数，固定8/2/2、test不参与；教师/语言动作后端冻结。新监控http://127.0.0.1:8879/。最多3116新更新及剩余资源上限，既有定时自动化未恢复；未证明收敛或控制恢复。


2026-10-22 共享RAW整视觉头最新进展：远端直接只读共享task6 Normal12源，固定8/2/2；全新204行RAW教师目标已逐行及状态链独立验收，test0行。双RTX5090真实零更新梯度预检、1次全局16anchors优化更新及完整检查点独立验收通过；442视觉状态张量均实际更新，可训练428,579,952参数，教师与语言/动作后端冻结。重载检查进行中，尚未宣称长跑或收敛；新监控http://127.0.0.1:8879/，旧8878为停止历史。见research/experiments/raw_stem_siglip_20261020/full-visual-shared-20261022-independent-review.md。


2026-10-22 数据来源最新更正：用户明确要求“共享盘直接读RAW，在远端重新生成匹配的教师目标，再训练整个学生视觉头”。旧50条RAW及旧token缓存传输已停止，不作为新阶段训练数据；共享根目录/root/autodl-pub/datasets/rawvla按只读处理。task6正常光实有12条，采用固定源示范8/2/2拆分，仅train+validation生成新的RAW教师targets，test不参与训练或选型；不加入其他照明条件。整视觉头新阶段仍从0步开始，继承实际π0.5视觉权重、重置stem/gate；教师及动作后端冻结，资源总预算不扩展。工程和真实GPU预检进行中，尚无新优化更新。

2026-10-22 最新用户授权：本机stem-only蒸馏已按用户要求停止于1872步，定时跟进已删除。新阶段迁移至用户指定SSH实例101.71.76.104:27786，使用两张RTX5090；用户明确不从旧学生检查点续训，重新初始化RAW stem/gate，仍继承当前π0.5策略的位置编码、SigLIP与原投影，并训练整个学生视觉头（428,579,952参数）。教师与语言／动作后端冻结，仅token蒸馏，无动作损失／闭环／弱光／实机扩展。新建阶段与优化器，旧实验保留；当前工程实现、传输与预检进行中，不能称已启动或充分收敛。暂沿用原总预算剩余额度：最多3116次新全局更新／49856 anchors、140841.2632 aggregate GPU秒，双卡按两卡合计，新增训练输出剩余15396584412字节。实际资产／派工／授权记录见outputs/raw-distillation-full-vision-20261022/。本条取代历史stem-only当前范围，不改写其事实。

2026-10-21 最新授权：用户明确“可以开始充分蒸馏，并给我一个简单的监控页面来实时查看进度和效果。”本阶段仅C单臂RAW→RAW投影后视觉token蒸馏，从真实C50续训，累计最多5000更新／80000 anchors；新阶段最多64 GPU小时／20GiB。优化器、scheduler、RNG、sampler及累计50步必须恢复；只更新现有stem＋gate，继承视觉与后端冻结。按固定验证集token MSE＋0.1cosine选型，动作误差仅诊断；无B/D扩展、视觉解冻、闭环或弱光授权。当前C51真实首步已独立验收，连续蒸馏已启动并真实恢复到后续更新；实时进度见[本地监控](http://127.0.0.1:8878/)，每5秒刷新。验证token损失每250步检查；后台每20分钟核查，正常无异常时保持安静。闭环0，控制恢复未证明。派工与运行收据：`/home/jeong/zeno/owac/repo/outputs/raw-distillation-c-full-20261021-123313`。此条取代此前“暂不启动”的当前状态，历史结果不变。

2026-10-21 后续方向：只继续C单臂RAW→RAW视觉token充分蒸馏，不再默认开展B/D多轮对照；用户当前明确暂不启动、先审框架。教师／学生实际结构及候选预算见[框架说明](docs/plans/RAW_TEACHER_STUDENT_FRAMEWORK_2026-10-21.md)。新训练authority未签发；投影后token选型尚待实现，既有pilot动作选型记录保留。


2026-10-21 **B/C50 pilot已完成并独立验收：token对齐改善，前5步动作误差回退，闭环未测**。两臂各50真实更新／800 anchors，只更新14,454,784个stem＋门控参数；继承视觉、教师与后端冻结。选型B1／C0，纯学生重载通过；模型作业约1.54 GPU小时、新输出约10.35GiB。C原监督终端143、原生子进程退出码不可得，真实50权重与日志已审计；未重训。当前全部本轮模型作业结束，停止于已用完的50上限；D／视觉解冻／闭环／弱光／实机未启动。见[本轮独立报告](research/experiments/raw_stem_siglip_20261020/pilot-20261021-independent-review.md)。

2026-10-20 **OWAC-D1.1 ALIGN工程与真实零更新预检通过，STEM待新授权**。因果6帧／每相机递归学生、同源SigLIP初始化、45源969条教师缓存、真实token/动作梯度、独立纯学生重载已验收；164项针对性测试及make check通过。真实学生0更新／0闭环，控制未恢复；教师仍为本地作者架构init0，非作者发布训练权重。下一步仅建议B/C各≤50更新、合计≤4 GPU小时／20GiB，不自动启动。见[独立报告](research/experiments/raw_stem_siglip_20261020/align-independent-review.md)。

2026-10-20 限定教师阶段已验收：[独立报告](research/experiments/raw_stem_siglip_20261020/teacher-independent-review.md)。Normal task6共同20新初态：参照20/20、RAW教师20/20，仅开发筛查。选定前端初始化0；实际155前端更新／1240 anchors未带来已证明收益，未作为选定权重。验证9成功＋1人为中断保留完整N10；50回放／50策略场次已用完。学生0更新／0闭环，尚未恢复；旧N01／N02／N04仍暂停，不授权学生／弱光／实机扩展。


2026-10-20 授权前工程交付记录：每帧 RAW stem／SigLIP 蒸馏基线与受授权门控的训练、纯 artifact 评估入口已可审阅；68 项针对性测试通过，实际同源后端恢复及合成输入数值／梯度核对通过。新增优化更新0、闭环0，控制未恢复。完整 RAW 教师权重、真实配对数据和时序训练端到端尚未验收；不固定最终策略为单帧。见 [独立验收](research/experiments/raw_stem_siglip_20261020/independent-review.md)。


当前唯一实施主线是[RAW stem 与 SigLIP 蒸馏](docs/plans/RAW_STEM_SIGLIP_DISTILLATION_2026-10-20.md)：继承当前策略的视觉权重，先只训练 RAW stem，再有限适配视觉层并加入动作监督；首轮冻结语言／动作后端，先恢复正常光单任务控制。OWAC-D1.1 明确教师必须接收 RAW，内部可经 Neural ISP；教师与学生的单帧／burst／递归按效果选择。学生部署直接 RAW→tokens→动作。用户已认可方案，[GPT-6.1 Sol 执行 prompt](docs/prompts/GPT61_SOL_RAW_STEM_SIGLIP_DISTILLATION.md)已准备；工程、训练与验收进度见上方2026-10-21记录；方案采用阶段未启动训练／闭环。

[研究统筹台](http://127.0.0.1:8876/)展示当前方案、完整提示词，以及历史分支如何暂停、合并并收敛到新主线。旧[ConvNeXt 视觉替换方案](research/experiments/2026-10-19_visual_replacement_recovery.md)作为诊断历史保留，后端 LoRA 配方不沿用。N01／N02／N04 继续暂停，候选池不删除。

2026-10-19：N01、N02、N04 的代码、研究结论和本地已有训练记录统一收敛到本地主线 `main`。三个方向全部暂停；远程服务器不可达。历史文档中的“运行中”“下一阶段”和训练授权均为当时快照，当前不恢复训练。

当前入口为 `/home/jeong/zeno/owac/repo/`。见[合并、证据保全与恢复说明](research/archives/2026-10-19_worktree_consolidation/README.md)、[当前状态](docs/status.md)和[本地研究统筹台](docs/research-hub.md)。

| 方向 | 最后已有结论与记录 | 证据边界 |
| --- | --- | --- |
| N01 | [P3 停止与动作诊断](research/experiments/n01_v1_2/P3_action_diagnosis_20261019.md)：RAW/RGB 各 0/12；同四个 Normal 场景原策略 4/4、两臂各 0/4 | 新表征与动作后端的适配尚未成立 |
| N02 | [r17 结论](research/experiments/n02_v1_2/residual_joint_r17_20261019/conclusions.json)：Normal 小样本冻结 RAW 残差有动作读出收益，局部联合更新无额外总体收益 | r17 的 π 更新 0、学习策略闭环 0；历史 r11 闭环 0/3 保留 |
| N04 | [v5 修复与训练记录](research/experiments/2026-10-19_n04_stability_repair_v5.md)：保留 v4.2 目标消失/数值退出与 v5 工程修复 | 历史开发闭环 0/100；v5 长期效果与闭环未验证 |

各分支原始 README、状态和执行规则已逐字保存到合并归档；实验记录与负结果保留原路径。大型本地输出在 `outputs/worktree-consolidation-20261019/` 归档并校验。三个原 worktree 已按用户要求移除，后续直接在主 checkout 的 `main` 开发；历史分支与恢复归档保留。仅存远端的权重和训练数据尚未恢复，不将路径清单当作文件已本地保全。

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
└── worktrees/            # 仅在后续需要独立 checkout 时创建
```

从主仓库创建后续 worktree 的方法见[开发说明](docs/development.md)。三个历史 worktree 已于 2026-10-19 归档并清理，当前只保留主 checkout；`worktrees/` 在后续确需隔离时再创建。

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

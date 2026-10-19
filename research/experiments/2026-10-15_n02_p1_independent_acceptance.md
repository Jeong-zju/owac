# N02 v1.1 P1 独立验收：未完成，不通过

验收日期：2026-10-15（Asia/Shanghai）。对象为 DeepSeek 的 P1 交付，不改变 P0 已验收范围。

**结论：P1 未完成，验收不通过。** 真实训练和检查点存在，但绝对检查没有接入运行链路，错误先验没有进入推理，评测与训练数据重叠，候选字段还存在可直接读出正确答案的捷径。不能接受原交付的 `implementation=pass / numerical=pass`，也不能将全部失败归因为“任务没有可辨识信息”。目前证据不足以判断充分训练后的 N02 能力。

本轮独立判定为 `implementation=fail`、`numerical=incomplete`、`behavior=fail`、`closed_loop=not_run`。其中 behavior=fail 表示 **P1 行为验收门未通过**；现有统计含无效测量项，不是对正确实现、充分训练后能力的否定。闭环不是 P1 的必做项。E1/P2 的交接条件未成立。

## 1. 原要求与验收范围

基准为 [P1 原提示词](../../docs/prompts/n02_v1_1/P1_OBSERVATION.md)、[共同契约](../../docs/prompts/n02_v1_1/00_SHARED_CONTRACT.md) 和 [完整方案](../../docs/plans/N02_full_development_plan_2026-10-13.md)，重点核对：

- Base RAW 预训练、条件测量模型、因果预测与 fit 分离、干扰建模、正常 calibration 上冻结的三态绝对检查。
- 100k 预训练 + 50k 测量训练，三个种子；训练覆盖、健康、学习率敏感性和开发行为趋稳，不能由步数或 loss 下降代替。
- 每类每种子至少 1000 独立场景，线索互换各方向 ≥90%，无信息时 ≥95% 的可验证概率漂移 ≤.05，95% 错先验各方向纠正 ≥80%，正常误报 ≤5% 时结构不足检出 ≥90%。
- NRE/BNRE 与其他对照的相同信息和训练预算，校准器、原始记录、资源与版本证据。

原交付记录称用户另行允许单 seed、有界预算及停止部分训练。本次不将预算缩减本身判成擅自操作；将实际产物作为开发诊断验收，不能据此改写正式门槛。用户本轮要求验收，本轮没有续训、修改候选/阈值、启动 E1/P2、提交或推送。

## 2. 实际服务器与检查点

- SSH：`ssh -p 16724 root@154.59.156.22`；主机 `d7f18aee4c88`，2× NVIDIA RTX PRO 6000 Blackwell Server Edition，单卡 97887 MiB。验收开始时两卡显存均为 0 MiB。
- 代码目录：`/workspace/owac/worktrees/n02-measurement-completion`；Python 3.10.19 / torch 2.11.0+cu128。
- 本地验收 checkout：`/home/jeong/zeno/owac/worktrees/n02-measurement-completion`，Git HEAD `861c46f67fff552c81fff89c5116c63b815224c5`。保留原有未提交变更。
- 当前远端目录没有 `.git`，`git rev-parse` 失败，是迁移后的源码副本，不能称为可核对 HEAD 的 Git worktree。当前两端 **15/15** 项源码/配置完整 SHA256 一致；该检查不追溯证明每次历史运行的源码一致。哈希和 checkpoint 哈希见 [remote_receipt.json](results/2026-10-15_n02_p1_independent_acceptance/remote_receipt.json)。
- 独立读取 `torch` 检查点，原“5k”目录内 `updates=4000`；`formal-cfa4-meas-5k-true/checkpoints/final.pt` 内确实为 `updates=5000`。本报告使用真实 5000 步结果，保留 1000/4000 步旧记录。
- 5000 步检查点在独立进程加载 **405/405** 张量，无形状冲突、无遗漏，排除本次探针只加载 Base、漏加载 head 的解释。

| 训练项目 | 实际可核实进度 | 验收边界 |
| --- | --- | --- |
| 真实 RawVLA RAW3 预训练 | 最后日志 6500，目标 25000；未见 final manifest | 原记录为用户停止；不是完成 25k，更不是 100k×3 seed |
| 合成 CFA4 stem | 2000，final manifest/检查点存在 | 与 RAW3 是不同模式的独立 Base，不能合并当成同一条预训练 |
| CFA4 measurement | 真实累计 5000，seed 1103 | 未完成原 50k×3 seed，正式充分训练判据未建立 |
| NRE / BNRE | 各 2000，冻结 Base | manifest 明示与主臂训练预算不一致 |
| 普通候选判别 | 实际 4000，声明预算 2000 | 预算接线缺陷已在旧记录披露；无法作同预算比较 |
| 条件似然 | 最后日志 850/2000，未见 final manifest | 原记录为用户停止 |
| raw_frame / conditional_innovation / registration | not_run | 必要对照未完成 |

## 3. 独立重数的 5000 步结果

直接读取远端 `outputs/n02-full/p1-acceptance-5k-true/scenario_records.jsonl`，共 **11532 行**，不是使用 PASS 字段或手工摘要。实际只有 seed 1103，每类 **961** 个独立 scene ID；三个组标签的重复行不计成独立场景。

| 验收项 | 独立重数 | 与原门槛的关系 |
| --- | --- | --- |
| 线索互换 paired both-correct | **11/961 = 1.14%**；各方向也均 11/961 | 数字远低于 90%；且下文证明该任务没有真正交换测量线索 |
| 无信息漂移 | 961/961 ≤.05，最大 .0457018769 | 样本 <1000；使用未校准 energy softmax，不能视作已通过概率保持 |
| 错先验纠正 A / B | **76/961 = 7.91% / 695/961 = 72.32%** | 均低于 80%；且先验没有进入模型，不能视为有效纠错测量 |
| 正常误报 / 结构不足检出 | 原记录均 0/961 | 全部 verdict 为 `None`，两个 0 都是接线造成的无效统计 |
| oracle 选择正确 | 797/961 = 82.93%；真候选唯一最低仅 **1/961** | 大量并列，不能把 82.93% 当作可辨识率 |

这组数不能同时被用来证明“行为真实失败且实现完全正确”。独立重数确认原始记录，以下复现进一步检查这些记录究竟测到了什么。

## 4. 阻断验收的实现与评测缺陷

### A. 三态绝对检查未接入，0 误报是假统计

[model.py 的锁定检查](../../src/owac/frameworks/n02_full/model.py) 将 `model.score()` 返回的 `MeasurementScore` 直接放入 `locked_checks`。该类型仅有 relative_terms / raw_nll_or_energy / local_stats / kind / per_valid_scalar，没有 verdict。评测 `_verdicts()` 随后读取不存在的 verdict 并转为字符串 `None`。

全部 **2883 条 normal** 记录是四个 `None`，全部 **2883 条 insufficiency** 记录是三个 `None`。独立加载真实检查点也精确复现，见 [wiring_probe.json](results/2026-10-15_n02_p1_independent_acceptance/wiring_probe.json)。`WindowedCalibrator` 虽有单元测试，正式 step/evaluation 没有调用它，因而没有完成“校准集→冻结窗口阈值→测试三态”的交付。

此外，即便接入校准器，其输出字段是 `CheckVerdict.status`；模型不足状态为 `model_inadequate`。现有评测既读错字段，又用 `"insufficient" in verdict` 作结构不足判定，混淆信息不足与模型不足。曝光/噪声/位移组的循环仅换字符串标签，运行同一个任务，没有施加对应组干扰。

### B. 95% 错先验没有进入推理

评测只把先验写入 `task.inference.declared_prior`，而 `_model_step()` 向 `step_bundle()` 传的是 bundle、空 SequenceState、history；bundle 不含该先验。当前 step/model 模块也没有读取 declared_prior 的逻辑。

在真实 5000 步检查点、同一测量上，分别给候选 0 与候选 1 **0.95 的合法归一化先验**，输出四个分数和“posterior”**逐位相同**。所以 7.91%/72.32% 是分数排序，不能证明与强错先验对抗后的纠正。原评测还把真候选先验设为 0，导致四候选总质量小于 1；即使未来接通也须先修正概率构造。

### C. 将经验能量 softmax 当后验，且没有构造无信息序列

独立模型输出的 score kind 为 **empirical_energy**，评测 `_posterior()` 直接计算 `softmax(-score/T)`。没有建立契约要求的可验证二候选概率或独立校准。no_information 分支只是对普通生成序列的首帧打分，再和均匀分布比较；没有生成专用无信息测量，也没有检查已声明先验的概率保持。961/961 不能作为此要求的通过证据。

### D. 正式评测复用训练数据，且“线索互换”仅为字典重排

训练 `TaskWindowSource` 和 oracle/behavior 都调用同一个生成器、同一 seed 1103、`partitions=("train",)`。区别只是 scene 数量。独立比较两种生成设置的前 **16/16** 场景，全部序列 RAW SHA256 相同。这是实测重叠，不仅是分区名可疑。

cue_swap 的两个方向复用同一个 sequence/frame，只把 `correct_index` 改为 0/1，使正确候选在字典中换位置。测量线索没有互换。该测试只能检查候选顺序影响，不能代替要求的线索交换两侧正确率。

### E. 候选的数值 nuisance 字段泄露正确答案

`build_p1_task()` 对所有错误候选加入 `log_exposure=.2*(index+1)`，正确候选默认 0；`ObservationModel._build_world_vector()` 又把这一数值作为候选 encoder 的输入。无需读取 RAW，也无需读取 truth，只选 `log_exposure==0` 的候选，独立探针 **32/32** 个字典正确（16 场景×两种排列）。

这违反“候选真伪由测量决定”；不仅是候选名称中出现 correct/wrong。解析 oracle 只接收几何覆盖，不使用该 nuisance 标记，和学习模型输入也不对等。修复时必须去掉与真伪确定相关的数值捷径，并按真实声明统一干扰模型。

### F. oracle 有稀疏采样盲区，但原信息不足归因过度

独立重数：961 场景中 **693** 场四候选分数完全相同，其余存在非零分数差；正确候选唯一最低仅 1 场。原交付“所有候选期望信号都相同”的统一说法不成立。

当前 query stride=8，两台 64×64 CFA 相机总共只看 **128/8192** 个标量，并固定 CFA 奇偶位置。8 个独立几何探针中，5 个在稀疏 query 上候选均值完全相同，但对应全传感器均值差达约 **1051—3469 DN**；另一些场景在现有 query 上已有显著差异。验证了采样造成的信息盲区，不能据此宣称原始 RAW 本身完全没有信息，也不能排除前述接线与任务设计缺陷。

这些探针没有修改正式字典/采样/阈值，也没有重跑或替换原正式结果；全传感器比较仅为诊断。新方案若调整采样/字典，应标为新版本，并使用新的独立确认集。

## 5. 逐项验收状态

| P1 交付要求 | 状态 | 依据 |
| --- | --- | --- |
| RAW Base、多尺度、条件 decoder、传感器数值组件 | 已实现并有测试，集成未通过 | 代码/测试存在；不能覆盖以下接口缺陷 |
| fit/prequential、测量去重、mask/数值回归 | 组件测试通过 | 本轮未重新证明所有完整科学契约 |
| 历史估计干扰与组间评价 | incomplete | 组件函数存在；评测 history 固定 eta=None，组标签重复 |
| 校准器和绝对检查进入实际预测链 | fail | 没有真实三态结果，校准器没有被调用 |
| 可验证后验、无信息保持与强错先验纠正 | fail | 能量冒称后验；先验丢失；无信息任务未构造 |
| 独立分区、真线索交换、无标签捷径 | fail | RAW 哈希重叠、字典重排、nuisance 捷径 |
| 三 seed / 每类≥1000 / 充分训练与完整曲线 | incomplete | 只有 seed1103/961，预算与对照未齐 |
| 同预算 NRE/BNRE 及消融 | incomplete | manifest 明示不等预算，三输入消融 not_run |
| 集成 p50/p95、持久状态成本 | incomplete | 不能用已有模块 profile 或训练吞吐代替集成决策成本 |
| P1 后 E1 / P2 准入 | 未成立 | 修复和新的有效行为验收应先完成 |

## 6. 本轮实际验证、证据与后续顺序

- 本地 `uv run --locked pytest tests/encoders/n02_full -q`：**495 passed / 6 skipped**（本地 CUDA 不可用及 numba 缺失）；远端同目录测试：**503 passed / 1 skipped**，84.27s，见 [远端测试原文](results/2026-10-15_n02_p1_independent_acceptance/remote_tests.txt)。两端测试集合不同，分别记录，不当作完全相同的覆盖。
- `make check` 通过（最终结果见本轮收据）。绿测不能覆盖本报告用真实检查点和原始记录发现的集成缺陷。
- 原始记录全量重数、检查点状态读取/加载、当前源码哈希、16 场景重叠、32 字典捷径、对立先验、8 场景采样诊断均已实际运行。脚本与紧凑 JSON 在 [验收结果目录](results/2026-10-15_n02_p1_independent_acceptance/)。原始完整审计输出在本地 ignored `outputs/n02-full/p1-independent-audit-20261015/`；原检查点/RAW/逐场景记录留服务器。
- 没有重新执行 1000 场景×多 seed 正式训练/封存评价，也没有更改生产实现；这是独立验收与小规模故障定位。

建议的下一次 P1 修复顺序：先接通三态校准检查与合法先验，并拒绝缺失 verdict 的统计；消除候选捷径，建立独立 train/dev/calibration/确认集和真正的线索交换/无信息/结构不足场景；在同采样、同输入的正确 oracle 上验证可辨识性，再做有界诊断、匹配预算对照与正式行为验收。保留本轮失败、旧阈值和旧结果，新版本不得覆盖旧封存结果。完成这些修复和有效验收之前，继续保持 P1 未完成。

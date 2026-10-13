# N02 源码复核：错误先验、歧义保持与局部修订

复核版本：`7c567cc33bb6abc382489c4f75c4a3ba810e6d43`，分支
`lzh/n02-measurement-completion`。本次检查源码与现有 200 步 checkpoint，并做固定权重的定点
干预；没有修改编码器、生成器或训练算法，没有新训练或服务器操作。

**首先需要更正原验收的解释：合成生成器把遮挡掩码当成透过率，抹掉了原本应保留的反证线索。**
因此旧矩阵中的强错误先验纠正、场景变化恢复和传播对照不能作为 N02 学习能力的有效验收。
旧数值与 checkpoint 保留，作为错误输入条件下的运行记录；模型的无信息权重漂移仍是实际观测到的
行为。当前状态仍为 `prototype / mechanism not accepted`，应先修数据再重训重测。

## 1. [P1] 遮挡语义反转，使训练和验收缺少有效反证

位置：[`synthetic.py:971`](../../src/owac/frameworks/n02/synthetic.py#L971)、
[`synthetic.py:1262`](../../src/owac/frameworks/n02/synthetic.py#L1262)。

`_occluder_mask` 明确约定 1 表示被遮挡，0 表示无遮挡，并将两种候选的线索中心周围置为 0。
渲染器却使用 `attenuation = occluder_mask * transmission`。这把无遮挡区域的信号乘成了 0，
也把原本被遮挡区域当作保留区域。噪声随后也被同一个系数缩放。

使用验收实际的 seeds 100–103、validation、64×64、8 帧，检查三倍标准差的线索中心区域：

| seed | 标记为可辨识 | 末帧未遮挡线索峰值 | 实际线索中心 RAW 最大绝对值 | 声明噪声下 oracle 的末步正确权重 |
| --- | --- | --- | --- | --- |
| 100 | true | 0.42290 | **0** | 0.050002 |
| 101 | true | 0.48011 | **0** | 0.049910 |
| 102 | true | 0.52662 | **0** | 0.047709 |
| 103 | true | 0.63601 | **0** | 0.049995 |

最后一列给独立质量检查器已知场景参数、两种测量模板、实际全部观测与声明方差，进行精确 Gaussian
累计，从 95% 错误先验开始。检查器没有学到的重建误差，仍无法从这些输入纠正先验。隐藏参数只用于
检查数据中的区分信息，不进入 encoder；该检查不作为同信息量的模型比较。实际空间噪声方差下的
oracle 正确权重也只有 0.0269–0.0602。中心之外有微弱 Gaussian 尾部残留，不能把它视为预定反证主体。

验收器 [`n02_acceptance.py:106`](../../src/owac/evaluation/n02_acceptance.py#L106) 与训练器
[`training/n02.py:323`](../../src/owac/training/n02.py#L323) 只用 `cue_strength > 0` 判断可辨识，
没有检查渲染后 RAW 中的线索。这导致训练会在实际上无充分线索的帧上强迫输出真值。

已有两个测试分别检查掩码避开线索、输入与生成公式一致；
[`test_synthetic.py:52`](../../tests/encoders/n02/test_synthetic.py#L52) 复制了同一个错误的衰减公式。
两个测试都通过，却没有检查最终观测保留了线索。本次实际重跑这两个测试及单位依赖传播测试，
结果为 **3 passed**。

修补应统一遮挡语义，例如无遮挡区域透过率为 1、遮挡区域透过率为 transmission，软掩码使用
`1 - mask * (1 - transmission)`；生成、CFA 采样、template fit、质量检查和测试须使用一致的约定。
同时独立检查最终 RAW 的 cue on/off 差异和信息量，不能只验证内部公式或标签。

另一个相关缺口是噪声声明：当前 `applied_noise = attenuation * noise_sigma * unit_noise`，
实际方差为 `attenuation² * noise_sigma²`；训练和验收却声明所有像素的方差为 `noise_sigma²`。
修补时需要明确噪声在成像链中加入的位置，并使提供给 encoder 的方差与该位置一致。

## 2. [P1] 保歧义损失没有约束实际累积输出

位置：[`training/n02.py:645`](../../src/owac/training/n02.py#L645)、
[`model.py:534`](../../src/owac/frameworks/n02/model.py#L534)。

训练的 mixture 损失使用 `log_softmax(-measurement_energy)`，无 cue 时目标为均匀；
实际输出使用本帧 energy 加上一帧输出形成的 prior。损失没有直接约束实际的 `representation.weights`。
训练还在每帧边界 detach 状态（[`training/n02.py:932`](../../src/owac/training/n02.py#L932)），
所以后续失败不会沿状态链直接纠正早期偏差。

独立梯度探针给相同、完全均匀的四槽 measurement energy，但实际两类输出分别为 2%/98% 和 50%/50%。
两者的 mixture 损失完全相同，均为 `log(4)`，对实际输出权重没有梯度，对均匀 energy 的梯度为 0。
现有 checkpoint 的无 cue seed 103，正确类权重由 0.5767 漂到 **0.0207**。

这说明局部 evidence 监督不足以防止实际时序输出自信漂移；不意味着每次递归 prior 本身是重复计数。
内部循环使用固定来源 prior 的修补仍成立。暖状态重新注入初始上下文的干预没有一致改善，不能将其
单独认定为本次三项失败的根因。

修补应对实际类权重加入时序歧义/纠错约束并检查校准。无辨识证据时应保持传入先验的类质量；
有信息先验不能统一强行改成 50%/50%。当前无信息验收的先验为均匀，才要求保持均匀。

## 3. [P1] 弱依赖被传播阈值挡住

位置：[`revision.py:220`](../../src/owac/frameworks/n02/revision.py#L220)、
[`revision.py:313`](../../src/owac/frameworks/n02/revision.py#L313)。

依赖边先按来源候选做行归一化，再将单个邻居的反证乘以 decay，最后要求 vote 至少为 decay。
对于孤立反证源，这等价于要求归一化边权达到 **1**。只要同一行还依赖另一个候选，该来源的边权就
小于 1，无法单独触发目标更新；学习出的非零跨候选边也因此通常不能触发局部扩散。

在同一 5×5 网格中放一个反证点，用单位依赖时各槽都新增四个邻居；替换为实际 checkpoint 的依赖：

| 反证来源槽 | 单位依赖新增同槽邻居 | 学到的依赖新增同槽邻居 |
| --- | --- | --- |
| 0 | 4 | 2 |
| 1 | 4 | 4 |
| 2 | 4 | **0** |
| 3 | 4 | 4 |

所有情况均没有新增跨候选邻居。槽 2 的一个邻居 priority 为 0.49239，被固定的 0.5 门槛排除。
两跳单向链探针还得到 priority `[1, .5, .125, 0, 0]`，第二跳没有入选；原因是递归 source 已带衰减，
又乘了 `decay**hop`，且仍使用第一跳的阈值。默认只用一跳，后者是配置扩展缺陷。

传播不是完全没有执行：无 cue 的 128 次内部循环中，平均直接反证候选单元占 43.62%，
传播新增 24.87%，实际更新占 69.26%。这些大范围重建残差引起的更新，不能
代替弱依赖路径确实传递反证的证据。应将边兼容权、优先级累计与实际选择门槛分开设计，并增加
非单位、混合依赖与多跳的定点回归。

`edge.clamp_min(0)` 还使负边参数的梯度为零；当前 checkpoint 多个跨候选边为负。复核的独立
梯度探针确认该截断使负边不能直接由损失梯度恢复。修补边参数化时须同时检验选择门槛，不能仅
替换成正值参数化而保留上述达到 1 才传播的条件。

## 4. [P2] K=4 的候选集合没有同等的测量监督

位置：[`training/n02.py:597`](../../src/owac/training/n02.py#L597)、
[`training/n02.py:608`](../../src/owac/training/n02.py#L608)。

anchor 是二值标签 0/1，measurement loss 只直接监督槽 0/1。geometry 与 mixture 却把槽 0/2
当作同一类、槽 1/3 当作另一类。额外槽只有间接 mixture 监督，与直接重建监督不等价。
独立梯度探针在四槽 energy 完全相同、无 cue 时，measurement+mixture 仍只推动被 anchor 选中的
槽降低 energy。geometry 的固定 ±1 标签也没有把这些槽的测量预测绑定到完整物理解释。

现有无 cue 回放中，槽 0/1/2/3 的平均 energy 分别为 -1.908/-2.281/**4.399**/-1.158；
槽 2 常被反证并被排除。数据错误修补前不能把该数值当成泛化结论，但监督不对称是可独立复现的
实现问题。应采用与语义集合一致的 measurement mixture / 集合监督，并报告每槽质量与无效候选。

## 5. [P2] 强先验训练覆盖不足

位置：[`synthetic.py:1196`](../../src/owac/frameworks/n02/synthetic.py#L1196)。

训练的先验 log odds 绝对值仅在 0.6–1.6，置信度约为 64.6%–83.2%；验收使用 95%，
对应 log odds 2.944。条件名称覆盖了 misleading，但没有覆盖验收要求的强度。
200 步共 3200 个帧样本，仅 357 个具有 revision 监督，83 个优化步中该项有效。
这不单独证明失败由样本量造成；它说明当前 smoke 配方没有完成所要求条件的训练与验证。
应在独立拆分下覆盖先验强度和反证强度，保留 95% 的验收要求。

## 6. [P2] 反证降权配置没有接入

位置：[`config.py:60`](../../src/owac/frameworks/n02/config.py#L60)、
[`revision.py:443`](../../src/owac/frameworks/n02/revision.py#L443)。

`downweight_contradicted` 只被定义和检查取值，实际修订与能量计算从不读取它。
当前路径会重置场并重新计算 measurement energy，但没有该参数声明的显式降权操作。
固定 checkpoint、seed 103 的错误先验序列，把参数分别设为 0 和 1，全部逐帧输出权重逐位相同，
最大差异为 0。`strong_contradiction_threshold`、`propagate_priority` 等配置也未参与相应路径。

不能把调这个参数作为纠错方案；应明确撤回规则如何改变受反驳候选的分数，或者移除死配置及相关
功能声明。任何降权实现都需要有效测量支持，避免在无 cue 时把重建误差当作语义反证。

## 复现与修补顺序

小型证据与两个复现脚本在
[source review results](../experiments/results/2026-10-12_n02_source_review/manifest.json)。
详细 74 条固定权重回放留在 ignored `outputs/n02-source-review/report.json`，Git 留摘要、文件哈希及
生成脚本。固定 checkpoint SHA256：
`64884c124fc517b0dc3033aa6f82582812669c582ee0a67404de592bb480c56f`。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python \
  research/experiments/results/2026-10-12_n02_source_review/probe.py
OMP_NUM_THREADS=1 uv run --locked python \
  research/experiments/results/2026-10-12_n02_source_review/dataset_probe.py
```

顺序应为：修补生成器的遮挡/噪声及独立信息量检查 → 修补实际输出的歧义监督和候选集合监督 →
修补弱依赖选择与死配置 → 覆盖强先验并重新训练候选及同预算基线 → 在有效反证序列上重做三项验收。
本次定点干预只是故障定位，不能视为修补后模型已通过。解析数值、RAW 接口与真实 NPZ/π0.5 离线
梯度及推理证据仍按各自范围保留；未获得新的机器人能力结论。

## 后续修补状态（2026-10-13）

六项源码问题已修补，同一依赖模块的定向测试还修正了消息汇总轴。工程回归通过；同预算 200 步重训仍未通过三项行为，不能把本 review 的原因列表视为机制成立的充分条件。保留原 review 与旧数据诊断，最新结论及对比见[修补实验](../experiments/2026-10-13_n02_repairs.md)。

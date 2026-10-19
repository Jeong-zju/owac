# P5 长期状态 重放与期限读出

> **已被 v1.1 取代（2026-10-15），不再执行本提示词。** 当前正式路线见 [v1.2 完整方案](../../N01_full_development_plan_2026-10-15_v1.2.md)。旧局部阶段正文仅保留供追溯；新 P3 必须整体交付编码器、真实任务训练与开发闭环，不能按旧步骤继续或只改编号。新执行包尚未编写。

只执行 P5。读取[共同契约](00_SHARED_CONTRACT.md)、P4 实际结论，以及方案第 4.5–4.6、5.4–5.6、6.3 节。目标是保存可靠的长历史，同时能把污染从神经状态中真正撤回；先有正确全量重放，再评估局部执行是否值得。

## 1. 文件职责

新增 `frameworks/n01_full/state_field.py / invalidation.py / deadline.py`，`tests/encoders/n01_full/test_state_replay.py / test_deadline.py`，扩展训练器与评估器。状态字段包括 source lineage、support/mapping/calibration/parameter version、所有网络 cache、clean checkpoint ID 和精确回放范围。

## 2. 先实现解析状态参考

```text
s_next = D_t P_t s + (I-D_t) u
D_jj = exp(-delta_t/tau_j), 0 <= D_jj <= d_max < 1
P_ij >= 0, row_sum(P)=1
```

P 只在当前相容域间交换，禁止越过已拒绝边；无合法邻居用 self-loop。参考先设固定外生 P/D，证明有界性与两轨迹收缩，加入负权、行和>1、各自状态驱动门控等反例。可学习/状态驱动规则标经验分支，不继承参考定理。缺测只执行明确的 aging/过程模型，不增加测量。

## 3. 官方 Mamba 的候选实现

按[固定 Mamba 实现](02_REFERENCES.md)读取 `Mamba.forward / step / allocate_inference_cache` 与 `InferenceParams`。首候选为 feature width=512、4 blocks、d_state=64、expand=2、作者 d_conv 配置；Mamba-1 是本轮明确目标，不悄悄换 Mamba-2。

状态沿因果运动轨迹或稳定 domain ID 维护，先用粗域网格。四层 FP32 SSM 每域约 1 MiB，P0/P4 资源基础上实测并锁定活跃域上限；不能在每个全分辨率像素无界分配。短卷积 cache 与 SSM cache 都保存、reset、replay。使用完整作者模块，CPU 小模型仅供性质测试，不作正式性能组。

默认局部交换只在 block 之间对已提交同版本域特征作显式 P_t 变换，交换节点进入依赖图。域 merge/split 后不平均 Mamba hidden state，先根据新域源序列重放。多时间尺度比较保留固定 tau 的解析参考和官方完整 S4 配置；S4D 教学文件仅用于理解，不是强基线。

## 4. 从完整重放到局部重放

实现 `rebuild_from_clean_checkpoint`：寻找不依赖被撤回源的最近合法 checkpoint，再按时间重放到当前；若没有干净检查点，清空相关摘要，从保留 RAW 窗起点重建，记录丢掉的长历史。一个“时间更早”的检查点未必干净，必须检查 lineage。

`fixed_decision` 模式保持决策；`counterfactual` 模式重做 motion、segment、门控与空间交换。参数更新也使 cache 失效。训练 TBPTT 只裁剪梯度，不伪造无污染推理状态；记录 burn-in、训练展开和推理历史差异。

先以全量事件日志构造批量参考，再实现 dependency graph 的后继闭包重算。多跳空间传播、当前 reference 对过去全部 flow 的影响、读出位置编码都纳入闭包。只有和全量参考等价且端到端成本更低时启用局部模式；否则保留全量为默认。

## 5. 快照与 deadline

采用 pending/committed 两套版本：完整重建结束才提交。任一操作边界中断，只能读已提交快照；deadline 到达时选择合法旧快照或当前有效测量的成熟单帧通路。旧快照年龄、当前单帧回退成本都记录；全无效返回无效 mask，交由预定义客户端规则处理。

CPU 操作预算与 GPU 时间期限分开。GPU kernel 不可随时抢占，调度只在已完成 chunk 间检查软期限；实测包括排队、H2D、motion、支持决策、重编码、融合、状态/读出，P7 再补策略动作时延。不能把 p95 描述为硬实时上界。

## 6. 对照与验收

比较 P4 固定窗口 DBSR、完整显式 N-TS、同容量持久状态无撤回、同容量检测后 reset、精确 replay 状态。全部同观测到达序列与正式训练预算，三种子；局部重算/全量重算先同权重比较。额外能力比较与受控计算预算比较分开。

必测：streaming 与 batch reference；每个允许中断点版本一致；短卷积/SSM 任一 cache 保留污染时测试能失败；跨 episode/camera/参数版本隔离；撤回超出 RAW 窗范围的实际行为；千帧序列状态/内存不增长；先稳定长积累再突变和再稳定的恢复曲线。

最终报告质量—延迟—状态大小表、p50/p95/p99、超期率、fallback 比例与耗时、全量/局部受影响节点数及原始数据。若 Mamba 无增量，交付“显式支持版本继续，神经状态暂不采用”的准确结果即可；不删分支负结果。

执行针对性和 P1/P4 回归、`make check`，远程运行完整状态实验。交付 `P5_report.md`、T4/T5/T6 的假设/证据/反例、checkpoint 和推荐部署配置，停在 P5。

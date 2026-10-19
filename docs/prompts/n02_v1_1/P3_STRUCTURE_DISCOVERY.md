# 给DeepSeek：P3，发现初始候选之外的解释

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

## 本阶段执行位置

优先连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，在远端 `/workspace/owac/worktrees/n02-measurement-completion` 开发、测试和运行。先完整阅读[远程开发执行说明](03_REMOTE_DEVELOPMENT.md)，按其步骤准备尚未建立的 worktree、解释器与 P0 资产；随后读取远端 AGENTS.md、README.md、docs/status.md。本文命令、相对路径及 `python` 均指远端 checkout 和对应阶段的独立运行时，GPU 命令显式使用 `"$N02_PY"`。

在远端加载 P2/E2 检查点和同分区序列；枚举/Gen/particles 数值参考与结构训练都在远端对应独立环境运行。粒子轨迹、完整提议账本和大 checkpoint 写入远端阶段输出。

读[共同契约](00_SHARED_CONTRACT.md)、P2/E2交付和原方案T5/T6、P3、8.2。实现世界候选的结构变化，先数值正确，再学习提议。主要修改`hypotheses.py/proposals.py/state.py/model.py`；新增训练stage p3、`evaluation/n02_full/p3.py`及`research/theory/n02_full/t5_t6.md`。

来源：[R04 SCALOR discovery/propagation](02_REFERENCES.md#r04)、[R28 particles](02_REFERENCES.md#r28)、[R13 Gen SMC/MH/trace translator](02_REFERENCES.md#r13)、[R14 RJMCMC](02_REFERENCES.md#r14)。不用“多几个固定embedding”替代结构发现。

## 1. 先实现可以独立验证的结构空间

数值参考用0–3个实体、离散位置/形状/身份、有限关系类型，完整枚举概率；所有参数/先验/观测分布直接来自显式配置。实体标签语义固定：参考先使用带标签槽，空槽可出生、占用槽可死亡，不一边按位置排序一边漏掉排列概率。真结构在**模型支持空间**内，但从初始活动粒子集合删除；基础提议仍能到达它。

实现`StructureEdit(kind, parent_world_id, selected_ids, auxiliaries, proposal_metadata)`，kind先为birth/death/reassociate，再加入split/merge，关系复杂变化交给P4。每个world保存完整实体集合、跨相机关联、nuisance、连续变量及祖先，结构hash不依赖临时内存地址。

从同一MeasurementSet比较竞争编辑：旧结构+成像干扰、新实体/身份、旧实体消失或重关联。解码和复杂度先验共享；增加自由度必须有存在性/结构先验代价。

## 2. 概率参考与优化搜索分成两个引擎

### A. 标准SMC/RJMCMC参考

目标先写为显式状态轨迹密度γ_t，使用正确重要性比：

```text
log w_t = log w_(t-1)
          + log p(y_t | z_t, declared_history)
          + log p(z_t | z_(t-1), executed_action_if_available)
          - log q(z_t | z_(t-1), y_t, declared_history)
```

初始先验单独计算一次。若用一般SMC sampler路径则记录相应backward kernel及完整γ比，不能继续套上述filter公式。重复measurement调用不新增y_t；同一时刻rejuvenation只用保持同一target的合法MCMC核。

logsumexp归一化，ESS=1/sum(w²)；首版systematic resampling，ESS<K/2触发，记录随机数、祖先、重采样前权重及log-normalizer。重采样后权重均匀，不能下一轮又乘一次旧posterior。退化/粒子灭绝如实记录，不能复制真值粒子补救。

先做birth/death：明确选空槽/存活槽概率、birth参数density、reverse death选择概率、存在性先验，identity transformation的Jacobian=1。然后在连续低维参考实现可逆split/merge：`x1=x+u, x2=x-u`，inverse `x=(x1+x2)/2,u=(x1-x2)/2`，d维x的abs Jacobian=2^d；父槽/空槽/有序子对选择概率、u密度、边界条件和反向操作全部计入。超出边界直接拒绝，不能clip后仍使用原Jacobian。

接受公式：`log_alpha=min(0,log_target_new-log_target_old+log_q_reverse-log_q_forward+log_abs_det_jacobian)`，reverse merge使用逆Jacobian。如复杂编辑没定义逆映射，它只能进入B引擎。采样允许上坡，不套P2单调规则。

**这些概率量只赋予已定义完整密度的参考模型。** 对神经decoder的经验能量、隐式proposal或MAP后连续变量，不继承精确posterior；不得对每个SMC样本做任意LM收缩后仍按原q算权重。LM最优解只更新单独best_seen，或者使用另外推导正确的随机提议核。

### B. 正式高维近似结构搜索

使用K=8完整世界、每新时刻最多8个结构提议为起点，P0/P3开发冻结预算。学习提议从P1局部预测不足、P2对象状态和可用历史产生；SCALOR发现/传播为明确来源的候选生成基线。混合0.1基础探索分布（在已定义允许编辑上均匀选择、连续参数用规定prior）防止学习proposal完全封死结构。

每个编辑调用同预算全局连续求解，以**相同测量集合**和完整复杂度惩罚目标比较。首版确定top-K保留但去重结构，保存不同结构模式；score_kind标`empirical_energy`/`approximate_search`。点估计仅采用下降≥δ的编辑，δ由归一化能量的开发噪声底确定并冻结；分布诊断不称精确后验。标准SMC+相同提议集合/同预算是必保留对照。

best_seen点估计与候选分布独立保存。未找到解释时输出unresolved/model_inadequate，不捏造“新概念”。P3正式实现可以有概率参考和近似部署两种收据，分别报告；T5只对A成立。

## 3. 提议学习与事件定义

参考训练标签来自可枚举后验或离线高预算搜索，记录额外计算与是否近似；对多个好结构使用分布/集合目标，不能只模仿单一MAP。提议损失包含编辑类型CE、实体选择CE、连续参数NLL/回归和合法性，归一化后起点权重均1，再与RAW/实体目标按开发实验冻结。

硬resample/出生/接受不宣称可微，动作loss暂不穿过它们。连续参数可重参数化；首版不使用未经核验的straight-through。训练100k×3seed，有效32条×16帧，前端lr1e-5、新proposal1e-4起点，其余优化沿共同契约。

数据必须包括：新物体从遮挡出现；初始两实体误并；一个实体误分；相似实体身份交换；噪声亮点；旧实体重新出现；真实结构超出容量。每条保存initial_structure_hash与truth_membership=false评价证据，truth不进proposal。出生定义为产生新实体ID且连续可解释支持；遮挡重现优先重关联，不每帧出生。

`discovery_latency`从第一帧足以区分结构的新证据开始计，另报未发现为右删失/失败，不只对成功样本取均值。分辨“曾提出真结构”“接受真结构”“维持真结构”和动作是否恢复。

## 4. 必须通过的数值/行为检查

小系统：枚举后验、Gen、particles与自实现SMC在预定Monte Carlo误差内一致；使用独立重复/多粒子数显示误差缩小，不能只比argmax。固定双向编辑测试detailed balance账目，wrong q/Jacobian的反例测试应被检出。接近0/1概率在log域稳定，祖先/恢复结果可重演。

真实结构初始缺席测试；未来帧/隐藏truth不能影响当前proposal；noise-only虚假出生；重现无重复ID；跨相机同一world；split/merge撤销后状态维和因子无残留；K=4/8/16的模式覆盖、ESS、灭绝率和成本。

正式3seed、每机制≥1000独立场景：发现率≥80%、虚假出生≤5%、正确拆分/合并≥80%。包含对标准SMC、SCALOR式发现、随机/固定候选/同预算Gen的比较，不只对旧tiny。写T6的条件支持下界说明，仅说明proposal触达，不写成保证识别。

交付数值参考与部署搜索分别的模型/源码/config、完整提议接受日志、失败结构、训练曲线、计数/区间、资源。P3交接后再做P4，不顺手把权重下降当错误链撤回完成。

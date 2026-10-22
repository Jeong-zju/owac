# 给DeepSeek：P4，学习依赖并真正撤回错误推断链

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

## 本阶段执行位置

优先连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，在远端 `/workspace/owac/worktrees/n02-measurement-completion` 开发、测试和运行。先完整阅读[远程开发执行说明](03_REMOTE_DEVELOPMENT.md)，按其步骤准备尚未建立的 worktree、解释器与 P0 资产；随后读取远端 AGENTS.md、README.md、docs/status.md。本文命令、相对路径及 `python` 均指远端 checkout 和对应阶段的独立运行时，GPU 命令显式使用 `"$N02_PY"`。

在远端续做 P3 的实际版本；关系训练、全局重放、缓存一致性、流式快照/恢复均在远端运行。重放所需 RAW、choice tape、状态检查点与历史测量必须能从远端路径解析。

阅读[共同契约](00_SHARED_CONTRACT.md)、P3交付、原方案3.2/3.5/T3/P4。主要实现`relations.py/provenance.py`，扩展state/factors/model/policy；新增`evaluation/n02_full/p4.py`和`research/theory/n02_full/t3.md`。

参考[R06 dNRI](02_REFERENCES.md#r06)、[R12 ATMS](02_REFERENCES.md#r12)、[R13 Gen trace](02_REFERENCES.md#r13)、[R11 GTSAM](02_REFERENCES.md#r11)。本轮先全局重放正确性，不要求局部提速。

## 1. 两张图和两种传播接口

`RelationModel.predict_edges(history, entities, executed_actions)`产生G：有向预测关系的分布和可用性，shape `[B,K,N,N,R]`，mask非存活/未知节点。首版关系枚举none、co_motion、contact、occludes；语义和方向在生成器及配置一致。无法识别真实图的RAW数据只评价预测效用。

`Provenance.record_call(op, inputs, outputs, choices)`产生J：实际输入/程序/模型/随机选择版本到输出版本的DAG。采用稳定NodeID+revision；循环按time/iteration展开，G可循环但J不能用未展开环互相自证。

`predict_messages(G,Z)`只用于预测；`invalidate_dependencies(J,changed_versions)`只沿实际计算依赖使缓存失效。G可以提议要检查的区域，但不能替代J。全局注意力、GRN、共享归一化和控制流选择的实际依赖记录完整；因此首版某个叶变化可能使全体slots失效，这是正确结果。

## 2. 关系模型训练与在线因果路径

准确移植DNRI_Encoder的forward_rnn/prior_fc_out及decoder；训练允许用反向RNN的posterior作变分学习目标，但posterior张量不写入在线state，不进入在线policy评测。部署一律forward prior/single_step_forward；预测用过去和当前观测，无未来帧。

有图监督的机制集先学习边CE，再联合多步状态预测及KL/稀疏先验；比较1/5/10步预测，默认5步起点。执行删边/置随机边/全连边和有记录干预，验证关系确实用于预测。动作没有真实已执行日志时availability=false，不拿未来专家动作代替。

训练100k×3seed，有效16条×32帧，验证64/128帧长历史。训练有标签比例先100%，再50%、10%作明确消融，监督相同给B3。新relation lr1e-4、Base1e-5起点；关系CE/预测/KL分别记录梯度与激活率。不得以attention图可视化替代F1或干预。

## 3. 支持的逻辑与数值失效分开

实现 `ConclusionVersion` 的多个 `SupportAlternative`：一条alternative内是AND依赖，多条alternative是OR支持。每条支持列完整叶假设/测量及派生父版本，保留共享祖先，不用路径条数充当独立证据。

示例必须实现：`c <- (a AND b) OR d`。撤回a后，若d仍有效，c的逻辑存在状态保持supported；但曾由a/b/d共同计算的c置信度和hidden仍须失效重算。若d也失效，c才被逻辑撤回。删除a时只删一条边然后留下旧confidence是错误实现。

独立支持的判定基于叶来源与生成依赖，不能因为经过两个不同神经层就称独立；共享祖先a的两路径同时受a失效影响。无测量支持的模型先验可保留为hypothesis，不能标observed fact。结构/控制流变化重新生成子trace，旧分支结论不继续消费。

## 4. 先建可信重放，再建增量失效

`rebuild_from_history(valid_leaves, fixed_model, choice_tape)`是独立可信参考：从未受撤回影响的检查点或episode起点重放保留测量与声明先验，依次重建实体、关系、因子、候选和读出。不要把受污染hidden作为initial来做“全重算”。

每8真实帧保存重放检查点为起点，检查点含全部输入支持摘要与model版本。窗口外祖先失效时回到其影响之前的有效检查点；若不存在，回到episode开始。若测量档案缺失，报告incomplete_rebuild并保持不确定，不伪称撤回完成。

滑窗首版仍保留完整可重放测量引用；如采用边际化摘要，保存被消元变量/因子/线性化点/来源版本。摘要输入一变整体失效，从原因子重建，不能在旧Schur complement上删一个标记继续使用。新观测重用同一MeasurementKey重放，不加第二份证据。

然后实现`recompute_invalidated_trace`：对J拓扑遍历，任何实际输入/支持集合/模型/随机choice变化都使缓存失效；有效的无变化节点可复用。T3比较前者与后者的实体hidden、weights/energy、关系、有效支持及tokens；容差按P0冻结的确定性基线。两者程序/输入/choice tape必须一致。

choice tape以稳定节点地址记录随机数；控制流新增节点用节点ID派生新随机流，不能靠全局RNG调用次数碰巧同步。神经权重推理时冻结，训练optimizer.step增加model版本并在下一段重新构建状态，不能跨参数版本保留旧hidden冒充可比较重放。

## 5. 状态污染和策略缓存

专门生成错误先验先持续≥8帧进入GRU、关系预测器和策略token，然后新证据到达的任务。比较四臂：只改候选权重、清空所有历史、来源一致全局重放、依赖失效重算。所有方法拥有相同测量档案和预算；全清空不能无成本获得历史重放。

对被撤回假设派生的对象hidden、候选祖先状态、LM warm start、边际摘要、MAP读出、prefix版本和未执行动作chunk一起处理。已经执行的动作属于历史事实，不“撤回”执行记录；只重算今后决策。保持固定cadence的主基准下在下一规划边界重建，事件式提前规划另列扩展。

## 6. 定点测试、验收与交接

测试覆盖：三跳链；AND/OR替代支持；共享祖先双路径；关系改向；遮挡重现与身份交换；窗口外撤回；摘要失效；同measurement校准更正；程序版本变化；全局attention依赖；新增控制流节点；部分batch reset；保存恢复后撤回；未来帧扰动不变过去在线G。

T3证明用拓扑归纳，明确完整依赖、固定算子与随机输入、有效检查点/可重放档案的假设。缺支持记录时不能给局部等价证书，使用全局重建并计成本。

正式受控集合：关系F1≥.85（只在真实图可识别数据）；多跳错误链撤回正确≥90%；有独立有效支持判断误撤回≤5%；确定性trace修订与从头执行数值一致。F1按关系类型、身份条件、遮挡、监督比例和长历史分别报告；原始TP/FP/FN和分母必须保存。

交付G/J实际事件日志、每次撤回前后状态hash、重放参考差异、四臂行为和资源、三seed曲线、T3文件。P4验收的是撤回正确性；局部速度和独有研究增益交P5，不因存在一张DAG就宣布新机制成立。

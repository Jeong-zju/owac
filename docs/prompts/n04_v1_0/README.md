# N04 v1.0 分阶段 DeepSeek 开发提示词

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 历史文档（2026-10-15）：后续开发已由 [N04 v1.1](../n04_v1_1/README.md)取代。原文与旧启动语仅供追溯，不作为当前执行指令；当前入口为 v1.1 S2，旧数据扩展与衍生权重不进入官方数据主实验。

从 S0 开始，每次只交付一个阶段。这里把[全量开发方案](../../plans/N04_full_development_v1.0.md)转成明确的实现任务；算法选型、接口、训练次序和验收条件已经给定。DeepSeek 负责实现和运行，阶段结束后由独立审阅者对照代码与原始结果决定下一步。

本包依据 2026-10-14 实读的主仓库、N04 worktree、远端服务器和参考源码准备。当前 N04 HEAD 为 `fd6255b`，工作树另有未提交文件；准确身份见 [准备核查](preparation_audit.md)。这是提示词交付，不是 S0–S7 已实施。

## 使用顺序

| 阶段 | 交给 DeepSeek 的完整提示词 | 本阶段的具体结果 |
| --- | --- | --- |
| S0 | [数据、时间、坐标与远端准备](S0_data_contracts.md) | 可追溯数据与拆分、telemetry、新三维机制集、正式训练可用性结论 |
| S1 | [正式 RAW 单帧测量](S1_raw_measurement.md) | ConvNeXt-B 测量、同骨干无记忆基线、冻结测量缓存 |
| S2 | [空间 RKN 与动态状态](S2_spatial_rkn.md) | 逐锚点 RKN、关联、缺失传播和匹配对照 |
| S3 | [已执行动作与三维自运动](S3_action_geometry.md) | 依次完成 S3a Ac-RKN、S3b 本体/FK、S3c 深度投影 |
| S4 | [测量误差与外部事件证据](S4_uncertainty.md) | 依次完成增益、噪声、几何/时间协方差、可辨识事件 |
| S5 | [外部变化修订动态记忆](S5_dynamic_revision.md) | 三解释更新、后续预测干预；运动网络按诊断条件启用 |
| S6 | [组合训练、消融与迁移](S6_joint_ablation.md) | 至少三完整种子、强对照、配对区间和成本 |
| S7 | [真实 π0.5 训练与完整闭环](S7_pi05_closed_loop.md) | 接入、充分训练、正式完整分母和独立挑战三层证据 |

可直接发送第一条启动语：

```text
请进入 /home/jeong/zeno/owac/worktrees/n04-egomotion-separation，完整读取 docs/prompts/n04_v1_0/S0_data_contracts.md 及其中要求的文件，实际完成 S0。正式数据核查和后续训练优先使用 ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080；按提示词使用远端隔离源码快照、数据路径和环境。不要只写计划，完成代码、数据审计、数值验证和 S0 交接报告；本次不开展 S1 网络训练，不自动提交或推送。
```

后续将上面的文件名与阶段替换为表中对应项即可。每份提示词都含执行约定与远端信息；若 DeepSeek 无法读取仓库文件，把该提示词全文和它所需的前阶段 `handoff.json` 一并提供。不要同时把八份提示词交给一个会话要求一次跑完。

## 准备包内容

- [源码与证据核查](preparation_audit.md)：当前能继承的实现、真实缺口、最新远端基线状态。
- [远端运行规约](remote_execution.md)：SSH、数据、环境、源码同步、进程管理与取回结果。
- [共享接口与设计决定](implementation_contract.md)：已确定的首版形状、状态时序、训练及交接格式。
- [参考实现导航](references.md)：每项参考具体看什么、复用什么，以及版本和授权状态。
- [机器可读准备核查](preparation_validation.json)：源码快照、链接与命令验证结果。

后续开发使用本 worktree 原有分支；主 checkout 只作为部署历史参考。旧 `DeepSeek_N04_implementation.md` 留作 2026-10-12 原型历史，正式 v1.0 使用本目录。原 v1.0 方案中的“RoboTwin 尚未齐全”属于旧快照：本轮远端报告已为 13,250/13,250，仍须在 S0/S7 读取届时的身份审计，而不是重新启动旧基线。

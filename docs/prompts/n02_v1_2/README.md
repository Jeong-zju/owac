# N02 v1.2 DeepSeek 阶段提示词

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 2026-10-16 最新：用户已授权 Codex 接管真实 P2 训练，r5 RAW/RGB 双卡1000步作业已启动，监控简化。不得另起 DeepSeek 或重复训练。见[当前运行](../../../research/experiments/n02_v1_2/P2_training_live_20261016.md)。P3/P4仍未授权。

> 2026-10-16 用户最新指令：**训练由用户自行开启**。当前为 `n02-v12-p2-causal-r4` 的代码修补与零优化更新画像；以下阶段包中的“实际短训”不是本轮自动执行授权。先读[本轮报告](../../../research/experiments/n02_v1_2/P2_fresh_profile_20261016.md)与[用户训练入口](../../../research/experiments/n02_v1_2/P2_r4_user_training.md)，不得启动训练、P3/P4，不得把旧检查点 resume 为 r4。

> 2026-10-16 独立验收：原 P2 未通过，当前仍是 P2 修复验证。执行路径已改为 `n02-v12-p2-causal-r2`，旧 1000 步 decoder 与新逐通道因果目标不兼容，不能直接 resume 为修补版。完成有界真实复验不等于完成修补版 1000 步。先读[独立验收报告](../../../research/experiments/n02_v1_2/P2_independent_acceptance_20261016.md)，不得据旧交付自行启动 P3/P4。

当前从 **P2** 开始。每份阶段文件已包含执行位置、单种子预算、数据/策略边界、前端专家任务和直接参考链接；整份复制即可，不拼接旧v1.1契约。

| 顺序 | 可直接复制的完整包 | 交付重点 |
| --- | --- | --- |
| P2 | [完整编码器与真实VLA链路](P2_FULL_ENCODER.md) | 修复P1，完整网络、实际短训、真实轨迹可视化与实时网页 |
| P3 | [单种子充分训练与整体比较](P3_TRAINING_COMPARISON.md) | 从P2继续训练、RAW/RGB分卡并行、整体动作与机制比较 |
| P4 | [独立重载与100次闭环](P4_CLOSED_LOOP_100.md) | 已选模型各100次，成功率、失败和真实同步视频 |

新版采用seed1103；100次指每个实际选中模型总100次，默认RoboTwin主平台，不是每任务/条件100，不要求三种子或全矩阵跑完。若P3预先安排双平台，则同一个总预算分为50+50、分平台报告。旧13,250+632只保留历史完整协议，不是当前阶段验收条件。充分训练预算100k预训练/50k适配是资源计划，不加loss单调、平台期、任意成功线或全量hash门槛。

新服务器`93.91.156.102:51584`双约96GiB GPU，GPU0 RAW、GPU1 RGB；代码在`/workspace/owac/worktrees/n02-measurement-completion`。新机部署由Codex执行，状态以[部署报告](../../../research/experiments/2026-10-15_n02_v1_2_remote_deployment.md)为准。远端8080被Jupyter占用，研究监控18080，转发到本机8081：

```bash
ssh -p 51584 root@93.91.156.102 -L 8080:localhost:8080 -L 8081:localhost:18080
```

浏览器打开<http://127.0.0.1:8081/>。部署页先显示部署和空闲GPU，P2调用前端专家升级为真实训练/可视化监控；“未训练”不能显示成正在训练。HF凭据只在远端私有认证位置，不写入任何提示词。

辅助阅读：[执行约定](00_EXECUTION_CONTEXT.md) · [论文与固定源码](01_REFERENCE_CODE.md) · [前端专家原文任务](02_FRONTEND_EXPERT.md) · [远端操作](03_REMOTE_EXECUTION.md) · [v1.2方案](../../plans/N02_full_development_v1.2.md)。辅助文件已嵌入阶段包，单独阅读无需重复复制。

一次只执行用户交付的那个阶段；结束给实际改动、命令、模型/产物位置、网页和简洁报告。P1独立验收未通过保持原判，P0已通过的基础复用；不以旧合成题、旧hash或旧服务器要求重新卡住P2。

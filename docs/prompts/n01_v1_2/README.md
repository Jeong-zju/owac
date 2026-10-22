# N01 v1.2 DeepSeek 完整阶段提示词

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

当前使用 **P3 完整编码器与RAW信息归因**。每阶段只复制一份完整包：公共要求、四臂同源协议、作者代码、远程训练和前端专家任务均已展开，无需拼接。旧v1.1三臂全RAW主矩阵与v1.0局部后续包已被取代。

| 阶段 | 直接复制的完整文件 | 交付 |
| --- | --- | --- |
| **P3 当前待启动** | [P3_FULL_ENCODER.md](P3_FULL_ENCODER.md) | 完整N01，同源RAW/RGB×Base/N01四臂，单种子，两平台8个主体run，每轮共100次开发闭环和信息—动作案例 |
| P4 后续 | [P4_FROZEN_FULL_EVALUATION.md](P4_FROZEN_FULL_EVALUATION.md) | 冻结四臂，每方法10000＋3250，单种子核心53,000正式rollout |
| P5 后续 | [P5_TRANSFER_AND_EXTENSIONS.md](P5_TRANSFER_AND_EXTENSIONS.md) | 按限制选择真实CFA/跨域、第二骨干/读出、ISP或长记忆扩展，保留对应归因 |

四方法为`raw_base/raw_n01/rgb_base/rgb_n01`，平台只有LIBERO和RoboTwin。每平台/方法仅1个训练种子，默认101；三种子后续可选。原B1标准reset是补充，不默认增加第五臂。普通修改只补受影响项，不重复整个8-run矩阵。开发期每轮总100次闭环，优先当前模型，不按每臂/每平台乘开，也不要求先把四臂闭环全部跑完；100次后报成功率即可，P4全量另行安排。

能访问本机checkout的DeepSeek可使用以下启动语；仅在服务器运行时，需传当前源码快照与完整P3包，本机绝对路径不自动出现在服务器。新机已有源码入口`/workspace/owac/rawvla/owac`，远端执行时从该目录读取同一份完整包：

```text
请进入 /home/jeong/zeno/owac/worktrees/n01-evidence-integration，完整读取 docs/prompts/n01_v1_2/P3_FULL_ENCODER.md 和其中要求的当前仓库材料，实际执行N01 v1.2 P3。核心为同源RAW/RGB×Base/N01四臂、每平台/方法单种子101。先调用前端专家接通简单实时监控，再完整实现、真实训练、开发闭环和信息—动作可视化；不执行旧v1.1全RAW三臂或v1.0局部阶段，不把正常光配对RGB当暗光RAW的RGB对照。保留现有修改，完成后停在P3交接点。
```

服务器上的DeepSeek可直接使用：

```text
请在新服务器 /workspace/owac/rawvla/owac 工作，完整读取 docs/prompts/n01_v1_2/P3_FULL_ENCODER.md 和 docs/N01_new_server_deployment_2026-10-15.md，实际完成N01 v1.2 P3。复用已部署环境、完整RawVLA-Bench与π0.5权重，GPU0排RAW、GPU1排RGB，训练和验证可并行。每个主体run只用seed101，每轮开发闭环总计100次。先按完整包调用前端专家上线简单实时监控；保留现有修改，完成后停在P3交接点。
```

正式依据：[v1.2方案](../../N01_full_development_plan_2026-10-15_v1.2.md)、[同源HTML](../../N01_full_development_plan_2026-10-15_v1.2.html)、[决议0015](../../decisions/0015-n01-v1-2-raw-information-route.md)。P0—P2资产及P2独立复核限制保留。用户要的是RAW信息如何改善VLA行动，来源撤回/哈希/测试数量不是任务增量。

每包要求DeepSeek在长训练前调用前端专家实现中文实时单页：显示四臂、RAW/RGB、数据来源、单种子、真实step/rollout、曲线、同源案例和日志。实时监控由后续执行者实现；本次更新的是现有本机研究控制台及实现指令，不假称远端训练监控已经上线。

**新服务器：**`ssh -p 34302 root@154.59.156.21 -L 8080:localhost:8080`。两张RTX PRO 6000 Blackwell Server Edition（每张约96GiB），GPU0排RAW、GPU1排RGB；训练/验证均可并行，开发闭环仍每轮总100次。新机环境与代码由Codex实际部署，见[部署记录](../../N01_new_server_deployment_2026-10-15.md)和[新机快照](remote_snapshot.json)。8080是Jupyter，监控用远端18080，转发命令详见完整包。旧服务器已经停用，[旧快照](remote_snapshot_legacy_20261015.json)仅作历史。HF认证仅保存在服务器私有凭据位置，不复制到提示词。

## 编辑与装配

编辑源为 [00_SHARED_CONTEXT.md](00_SHARED_CONTEXT.md)、[04_RAW_RGB_PROTOCOL.md](04_RAW_RGB_PROTOCOL.md)、[01_REFERENCE_CODE.md](01_REFERENCE_CODE.md)、[02_REMOTE_EXECUTION.md](02_REMOTE_EXECUTION.md)、[03_FRONTEND_EXPERT.md](03_FRONTEND_EXPERT.md)及`stages/`正文。修改后在本checkout运行：

```bash
python3 docs/prompts/n01_v1_2/assemble.py
```

装配器只拼接文档，不连接服务器或启动训练。当前每份完整包可独立复制，P4/P5未来前置记录以实际交付为准；不为准备时尚无报告创建假验收结果。

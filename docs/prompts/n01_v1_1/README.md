# N01 v1.1 DeepSeek 阶段提示词

> **历史提示词：当前执行请使用 [N01 v1.2完整包](../n01_v1_2/README.md)。本文件不再作为后续启动依据。**

当前执行入口是 **P3 完整编码器、真实训练与开发闭环**。按用户2026-10-15补充，当前每平台/方法只训练1个种子，默认101；不自动安排三种子。每次只复制下面一份完整文件的全文交给 DeepSeek；共同要求、参考链接、远程训练和前端专家任务已经展开，不需要拼接。P4/P5留待前阶段实际交付与独立验收后使用。

| 阶段 | 直接复制的完整提示词 | 本次交付对象 |
| --- | --- | --- |
| **P3 当前待启动** | [P3_FULL_ENCODER.md](P3_FULL_ENCODER.md) | 完整N01、两平台真实动作训练、N/B0/B1单种子、开发闭环、理论与同步可视化 |
| P4 后续 | [P4_FROZEN_FULL_EVALUATION.md](P4_FROZEN_FULL_EVALUATION.md) | 冻结模型；每checkpoint完整LIBERO10000＋RoboTwin3250，N/B0/B1单种子共39,750条，失败与缺口保留 |
| P5 后续 | [P5_TRANSFER_AND_EXTENSIONS.md](P5_TRANSFER_AND_EXTENSIONS.md) | 依据P3/P4限制选择真实CFA/跨域、第二成熟骨干、第二读出或长记忆/成本扩展 |

如 DeepSeek 能直接读本地仓库，可发送这段启动语；如果它只在远程服务器运行，应提供完整P3正文和当前源码快照，不假设服务器有本地路径：

```text
请进入 /home/jeong/zeno/owac/worktrees/n01-evidence-integration，完整读取 docs/prompts/n01_v1_1/P3_FULL_ENCODER.md 以及其中要求阅读的当前仓库材料，实际执行 N01 v1.1 P3。先接通简单实时监控页，再完成完整模型、真实训练、匹配对照、开发闭环与可视化；按包内指令调用前端专家。不要执行旧v1.0 P3—P8，不为补齐旧局部指标停留，不在本次自动进入P4。保留当前未提交修改。
```

提示词要求 DeepSeek 在第一轮长训练前调用前端专家实现中文单页监控，并接到真实编程活动、训练更新和评测队列。页面本次**尚未实现或部署**；这里交付的是明确的实现指令。实际监控与本机研究统筹台用途不同：统筹台提供路线/提示词，远程监控页显示实际作业状态。

2026-10-15只读核查发现远端8080已由Jupyter监听，18080为空。保留用户原SSH命令作为连接说明，监控默认使用远端18080，经本机8080转发；未来执行时重新检查。提示词中提供了完整命令、端口冲突处理和supervisor要求，避免覆盖现有服务。

用户要求已经落实为执行规则：不新增任意提升门槛、综合自定义评分、逐文件/AST/tensor哈希准入或固定平台期“收敛PASS”。保留与研究结论直接相关的数据泄漏、因果、实际参数更新、完整删源、恢复和正式分母检查。训练是否充分依据真实覆盖、匹配曲线和任务表现说明；负结果可以交付，缺失训练不能被改名为完成。

正式依据：[v1.1方案](../../N01_full_development_plan_2026-10-15_v1.1.md)、[同源HTML](../../N01_full_development_plan_2026-10-15_v1.1.html)、[决议0014](../../decisions/0014-n01-v1-1-full-model-route.md)。P0—P2资产继承，原P2独立复核限制保留；旧v1.0后续包仅供历史追溯。

## 维护完整包

共同正文来自 [00_SHARED_CONTEXT.md](00_SHARED_CONTEXT.md)、[01_REFERENCE_CODE.md](01_REFERENCE_CODE.md)、[02_REMOTE_EXECUTION.md](02_REMOTE_EXECUTION.md)、[03_FRONTEND_EXPERT.md](03_FRONTEND_EXPERT.md)，阶段正文在 `stages/`。这些是编辑源，**无需逐份发送给DeepSeek**。修改源后在本checkout执行：

```bash
python3 docs/prompts/n01_v1_1/assemble.py
```

三个完整包随仓库保存。装配器只拼接文档，不连接服务器、不启动训练、不生成验收分数。参考实现的既有固定版本用来定位代码，来源记录见正式方案；本次部分作者链接重新打开核对，并只读检查远程指南、资源、环境和端口，不把链接可访问写成新网络已运行。远程准备快照见 [remote_snapshot.json](remote_snapshot.json)。

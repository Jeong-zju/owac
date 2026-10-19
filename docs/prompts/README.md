# 三项编码器研究的 DeepSeek 使用入口

2026-10-15 N04 正式开发采用 [v1.1 阶段交接](n04_v1_1/README.md)，当前为 [S2 完整模型](n04_v1_1/S2_full_encoder.md)待实施，随后 S3 单种子对照／针对性改进和 S4 每待评模型100次闭环；完整任务含前端专家实时监控，远端登录指令留空。S0／S1 已有记录和 v1.0 提示词均保留为历史；不重启旧 S0，不再逐项执行旧 S2—S7。

三个 worktree 分别研究 N01、N02、N04，具有相同的部署基线、研究文档快照和提示词。每份提示词都可独立使用，包含核心更新规则、参考代码、文件职责、数据接口、训练顺序、π0.5 接入和验收实验。

| 研究方向 | worktree | 分支 | 完整提示词 |
| --- | --- | --- | --- |
| N01 自适应时空证据积分 | `/home/jeong/zeno/owac/worktrees/n01-evidence-integration` | `lzh/n01-evidence-integration` | [N01 提示词](DeepSeek_N01_implementation.md) |
| N02 测量约束的循环补全 | `/home/jeong/zeno/owac/worktrees/n02-measurement-completion` | `lzh/n02-measurement-completion` | [N02 提示词](DeepSeek_N02_implementation.md) |
| N04 RAW 证据预测与状态修订 | `/home/jeong/zeno/owac/worktrees/n04-egomotion-separation` | `lzh/n04-egomotion-separation` | [N04 v1.1 入口](n04_v1_1/README.md) |

在对应 worktree 中启动一个 DeepSeek 编程会话，让它读取对应提示词并开始实现；如果该会话不能读取文件，就把提示词全文粘贴进去。三个会话可分别推进，不必把三份提示词同时交给一个会话。

可直接发送以下启动语，按方向选择一条：

```text
请进入 /home/jeong/zeno/owac/worktrees/n01-evidence-integration，完整读取 docs/prompts/DeepSeek_N01_implementation.md 和它要求阅读的仓库文档，然后按顺序实际实现 N01。先做可手算的支持域积分，再做支持域重组、训练和后端接入。
```

```text
请进入 /home/jeong/zeno/owac/worktrees/n02-measurement-completion，完整读取 docs/prompts/DeepSeek_N02_implementation.md 和它要求阅读的仓库文档，然后按顺序实际实现 N02。先做两假设测量更新，再做局部反证修订、训练和后端接入。
```

```text
请进入 /home/jeong/zeno/owac/worktrees/n04-egomotion-separation，完整读取 docs/plans/N04_full_development_v1.1.md、docs/prompts/n04_v1_1/S2_full_encoder.md 和其中要求的当前证据，实际完成 v1.1 S2 完整模型、官方数据整体训练和开发闭环；完成后交接，不自动进入 S3。
```

提示词中的具体结构与超参数是首轮可检验方案；不是 roadmap 已确定的最终架构。N01、N02、N04 独立检验自己的假设，保留 F01—F12 的候选空间。

部署起点为 `owac-rawvla-pi05-v0.1.0` / `16075fca7c5e5cd126900bc62531a5733fe4e0be`。共同准备提交仅补充研究文档和实现提示，不新增编码器实现、不改变冻结基准。主 checkout 的未提交代码与历史任务保持原样。[准备决议](../decisions/0013-encoder-research-worktrees.md)说明文档快照与分支关系；[来源清单](research-source-snapshot.json)保存快照哈希；[参考实现清单](reference_sources.json)保存代码版本与核对层级。

RawVLA 的当前三通道输入与原生 CFA 分别建入口；后端使用固定 StarVLA π0.5。三个提示词采用同一外部接口语义，内部状态各自设计。后续整合先选定公共输入/特征桥实现，再按机制目录合并；本轮无需合并算法或启动重复全量基准。

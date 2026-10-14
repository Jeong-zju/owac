# 三项编码器研究的 DeepSeek 使用入口

2026-10-14 N04 正式开发改用 [v1.0 分阶段提示词](n04_v1_0/README.md)：S0–S7 八份独立执行任务，包含已核实的远端训练环境、正式模型接口、参考实现和阶段验收。下方 N04 原提示词保留为 2026-10-12 原型历史；新开发从 [S0](n04_v1_0/S0_data_contracts.md) 开始。

三个 worktree 分别研究 N01、N02、N04，具有相同的部署基线、研究文档快照和提示词。每份提示词都可独立使用，包含核心更新规则、参考代码、文件职责、数据接口、训练顺序、π0.5 接入和验收实验。

| 研究方向 | worktree | 分支 | 完整提示词 |
| --- | --- | --- | --- |
| N01 自适应时空证据积分 | `/home/jeong/zeno/owac/worktrees/n01-evidence-integration` | `lzh/n01-evidence-integration` | [N01 提示词](DeepSeek_N01_implementation.md) |
| N02 测量约束的循环补全 | `/home/jeong/zeno/owac/worktrees/n02-measurement-completion` | `lzh/n02-measurement-completion` | [N02 提示词](DeepSeek_N02_implementation.md) |
| N04 自运动预测与变化分离 | `/home/jeong/zeno/owac/worktrees/n04-egomotion-separation` | `lzh/n04-egomotion-separation` | [N04 提示词](DeepSeek_N04_implementation.md) |

在对应 worktree 中启动一个 DeepSeek 编程会话，让它读取对应提示词并开始实现；如果该会话不能读取文件，就把提示词全文粘贴进去。三个会话可分别推进，不必把三份提示词同时交给一个会话。

可直接发送以下启动语，按方向选择一条：

```text
请进入 /home/jeong/zeno/owac/worktrees/n01-evidence-integration，完整读取 docs/prompts/DeepSeek_N01_implementation.md 和它要求阅读的仓库文档，然后按顺序实际实现 N01。先做可手算的支持域积分，再做支持域重组、训练和后端接入。
```

```text
请进入 /home/jeong/zeno/owac/worktrees/n02-measurement-completion，完整读取 docs/prompts/DeepSeek_N02_implementation.md 和它要求阅读的仓库文档，然后按顺序实际实现 N02。先做两假设测量更新，再做局部反证修订、训练和后端接入。
```

```text
请进入 /home/jeong/zeno/owac/worktrees/n04-egomotion-separation，完整读取 docs/prompts/DeepSeek_N04_implementation.md 和它要求阅读的仓库文档，然后按顺序实际实现 N04。先做可检查的运动传播，再做变化分解、训练和后端接入。
```

提示词中的具体结构与超参数是首轮可检验方案；不是 roadmap 已确定的最终架构。N01、N02、N04 独立检验自己的假设，保留 F01—F12 的候选空间。

部署起点为 `owac-rawvla-pi05-v0.1.0` / `16075fca7c5e5cd126900bc62531a5733fe4e0be`。共同准备提交仅补充研究文档和实现提示，不新增编码器实现、不改变冻结基准。主 checkout 的未提交代码与历史任务保持原样。[准备决议](../decisions/0013-encoder-research-worktrees.md)说明文档快照与分支关系；[来源清单](research-source-snapshot.json)保存快照哈希；[参考实现清单](reference_sources.json)保存代码版本与核对层级。

RawVLA 的当前三通道输入与原生 CFA 分别建入口；后端使用固定 StarVLA π0.5。三个提示词采用同一外部接口语义，内部状态各自设计。后续整合先选定公共输入/特征桥实现，再按机制目录合并；本轮无需合并算法或启动重复全量基准。

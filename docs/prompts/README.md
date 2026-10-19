# 三项编码器研究的 DeepSeek 使用入口

2026-10-15：N01当前正式路线为[v1.2](../N01_full_development_plan_2026-10-15_v1.2.md)，[P3/P4/P5完整执行包](n01_v1_2/README.md)已准备。当前只复制P3一份：同源RAW/RGB×Base/N01四臂、每平台/方法1个训练种子、远程训练及简单实时监控前端专家任务均已展开。旧v1.1三臂全RAW主矩阵和v1.0后续局部包为历史，不从P0重启。本次更新文档/控制台/交接材料，未训练新模型。

2026-10-16 N02 最新：用户指定由 Codex 接管训练，r5 RAW/RGB 双卡正式 P2 作业已经运行，禁止再按旧 DeepSeek 启动语建立重复训练。见[当前运行](../../research/experiments/n02_v1_2/P2_training_live_20261016.md)。

2026-10-16 N02 最新边界：`n02-v12-p2-causal-r4` 完成代码修补与从头零优化更新画像；训练由用户自行开启，旧阶段启动语不构成自动训练授权。见[画像报告](../../research/experiments/n02_v1_2/P2_fresh_profile_20261016.md)和[用户训练入口](../../research/experiments/n02_v1_2/P2_r4_user_training.md)。

2026-10-15：N02 已采用 **[v1.2 正式方案](../plans/N02_full_development_v1.2.md)**，当前 P2 完整实现待实施；[采用决议](../decisions/0015-n02-v1-2-route.md)规定 P3 充分训练与 P4 独立确认。[新版P2—P4自包含包](n02_v1_2/README.md)已备，seed1103、每个实际选中模型共100次闭环，双GPU新服务器和前端专家实时网页任务已内置；[v1.1 阶段包](n02_v1_1/README.md)与旧原型仅供历史回查，不再复制其中启动语。P1 独立验收未通过的事实及缺陷修复要求保留。

三个 worktree 分别研究 N01、N02、N04，具有相同的部署基线、研究文档快照和提示词。每份提示词都可独立使用，包含核心更新规则、参考代码、文件职责、数据接口、训练顺序、π0.5 接入和验收实验。

| 研究方向 | worktree | 分支 | 完整提示词 |
| --- | --- | --- | --- |
| N01 自适应时空证据积分 | `/home/jeong/zeno/owac/worktrees/n01-evidence-integration` | `lzh/n01-evidence-integration` | [v1.2执行入口](n01_v1_2/README.md)；[当前P3完整包](n01_v1_2/P3_FULL_ENCODER.md) |
| N02 测量约束的循环补全 | `/home/jeong/zeno/owac/worktrees/n02-measurement-completion` | `lzh/n02-measurement-completion` | [N02 提示词](DeepSeek_N02_implementation.md) |

| N01 自适应时空证据积分 | `/home/jeong/zeno/owac/worktrees/n01-evidence-integration` | `lzh/n01-evidence-integration` | [N01 提示词](DeepSeek_N01_implementation.md) |
| N02 RAW 测量约束的动作表征 | `/home/jeong/zeno/owac/worktrees/n02-measurement-completion` | `lzh/n02-measurement-completion` | [v1.2 正式方案](../plans/N02_full_development_v1.2.md)；[P2完整包](n02_v1_2/P2_FULL_ENCODER.md)；[历史包](n02_v1_1/README.md) |
| N04 自运动预测与变化分离 | `/home/jeong/zeno/owac/worktrees/n04-egomotion-separation` | `lzh/n04-egomotion-separation` | [N04 提示词](DeepSeek_N04_implementation.md) |

N02/N04 在对应 worktree 使用其当前有效提示词；N01 使用新v1.2完整包，P4/P5执行前读取届时实际前阶段报告。三个方向独立，不能用本页历史快照覆盖各 worktree 更新后的路线。

可直接发送以下启动语，按方向选择一条：

```text
请进入 /home/jeong/zeno/owac/worktrees/n01-evidence-integration，完整读取 docs/prompts/n01_v1_2/P3_FULL_ENCODER.md 及其要求的当前仓库材料，实际执行N01 v1.2 P3。先接通简单实时监控页，调用前端专家，完成完整模型、真实训练、匹配对照、开发闭环与可视化；保留现有修改，不执行旧v1.0局部阶段，完成后停在P3交接点。
```

```text
请在新服务器93.91.156.102:51584的 /workspace/owac/worktrees/n02-measurement-completion 工作。
完整读取 docs/prompts/n02_v1_2/P2_FULL_ENCODER.md 并实际执行P2。
先修复P1已证实问题，一次建立完整N02，接真实RAW/π0.5、实际训练和前端专家实时网页。
只用seed1103；后续所选模型共100次闭环，不继承旧全量/hash/收敛门槛。完成P2后报告并停止。
```

```text
请进入 /home/jeong/zeno/owac/worktrees/n04-egomotion-separation，完整读取 docs/prompts/DeepSeek_N04_implementation.md 和它要求阅读的仓库文档，然后按顺序实际实现 N04。先做可检查的运动传播，再做变化分解、训练和后端接入。
```

提示词中的具体结构与超参数是首轮可检验方案；不是 roadmap 已确定的最终架构。N01、N02、N04 独立检验自己的假设，保留 F01—F12 的候选空间。

部署起点为 `owac-rawvla-pi05-v0.1.0` / `16075fca7c5e5cd126900bc62531a5733fe4e0be`。共同准备提交仅补充研究文档和实现提示，不新增编码器实现、不改变冻结基准。主 checkout 的未提交代码与历史任务保持原样。[准备决议](../decisions/0013-encoder-research-worktrees.md)说明文档快照与分支关系；[来源清单](research-source-snapshot.json)保存快照哈希；[参考实现清单](reference_sources.json)保存代码版本与核对层级。

RawVLA 的当前三通道输入与原生 CFA 分别建入口；后端使用固定 StarVLA π0.5。三个提示词采用同一外部接口语义，内部状态各自设计。后续整合先选定公共输入/特征桥实现，再按机制目录合并；本轮无需合并算法或启动重复全量基准。

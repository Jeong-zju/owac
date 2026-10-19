# N02 v1.1：DeepSeek 分阶段开发提示词

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

2026-10-14 更新。依据 [v1.1 完整方案](../../plans/N02_full_development_plan_2026-10-13.md)编写的逐阶段交接包。P0 已独立修补验收；当前接续 P1，P1—P7 尚未执行。全部提示词改为指定服务器优先开发，算法和正式训练预算沿用 v1.1。

连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，活跃开发目录使用远端 `/workspace/owac/worktrees/n02-measurement-completion`，分支 `lzh/n02-measurement-completion`。本次核查该目录尚未建立，先按[远程开发执行说明](03_REMOTE_DEVELOPMENT.md)从已验收本地提交创建，再直接使用远端 RAW/权重。`/home/jeong/zeno/owac/worktrees/n02-measurement-completion` 是本地交接和验收副本。新实现放 `n02_full`；旧 `n02` 保留为历史回归。

## 从这里开始

P0 已通过[独立验收](../../../research/experiments/2026-10-14_n02_p0_independent_acceptance.md)。先完成远端接入检查，再交给 DeepSeek **P1**，每次只下发一个阶段。P0 提示词保留为基础复现和远端迁移检查入口。每个阶段提示词要求读取同一份[共同契约](00_SHARED_CONTRACT.md)，已经规定接口、算法边界、训练路径和报告格式。完成后把阶段报告、diff 和原始结果交回本聊天进行独立验收，再发下一阶段。DeepSeek负责执行明确开发；路线改变、失败归因、是否扩大预算和研究结论由用户与本聊天判断。

```text
本轮优先在 ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080 上开发，数据都在远端。
先读取本地交接副本 /home/jeong/zeno/owac/worktrees/n02-measurement-completion/docs/prompts/n02_v1_1/03_REMOTE_DEVELOPMENT.md。
按该文件完成首次接入或复用已有远端 /workspace/owac/worktrees/n02-measurement-completion，之后所有开发、测试、训练和评测在远端进行。
进入远端 worktree，完整阅读 AGENTS.md、README.md、docs/status.md，随后阅读：
docs/prompts/n02_v1_1/00_SHARED_CONTRACT.md
docs/prompts/n02_v1_1/01_CURRENT_IMPLEMENTATION.md
research/experiments/2026-10-14_n02_p0_independent_acceptance.md
docs/prompts/n02_v1_1/P1_OBSERVATION.md
P0 已独立验收，复用已修补实现、v2 manifest、冻结 anchors 和环境锁；本次只做 P1。
直接修改远端代码并完成阶段要求，使用服务器已有真实 RAW/权重；不要因本机没有数据改做纯本地 smoke。
按交付清单给出远端报告、实际命令、源码 diff/哈希和产物绝对路径，完成后停在 P1 交接点。
```

后续使用同一启动语，将阶段文件、前置交付与“本次只做”的阶段一起换成表中目标。若 DeepSeek 无法访问文件，粘贴共同契约和当阶段正文；附上它要求读取的现有代码，不让它猜接口。不要一次性要求它完成全部阶段。

| 顺序 | 提示词 | 本轮真正要做成的事情 |
| --- | --- | --- |
| 1（已验收） | [P0 基础、数据、上游一致性](P0_FOUNDATION.md) | 复用基础与冻结分区；接入时验证远端运行时、真实数据与资源画像 |
| 2 | [P1 RAW观测与绝对拟合](P1_OBSERVATION.md) | Base RAW预训练、候选测量模型、独立预测检查及充分训练 |
| 3 | [E1 早期策略预检](EARLY_POLICY_PREFLIGHT.md) | 以 `mode=direct` 做P1后≤2k适配检查 |
| 4 | [P2 对象循环与全局求解](P2_ENTITIES_SOLVER.md) | 固定结构内的实体状态、LM提议、完整目标接受和全局参考 |
| 5 | [E2 同协议预检](EARLY_POLICY_PREFLIGHT.md) | 以 `mode=objects` 检查对象读出，和E1配对比较 |
| 6 | [P3 新解释发现](P3_STRUCTURE_DISCOVERY.md) | 真实结构不在初始候选中的出生、拆合、身份修订 |
| 7 | [P4 依赖与撤回](P4_PROVENANCE_REPLAY.md) | 因果时间的关系提议、实际来源、多支持与神经状态重建 |
| 8 | [P5 集成与局部执行比较](P5_MECHANISM_AUDIT.md) | 与成熟机制强基线比较，单独决定是否启用局部执行 |
| 9 | [P6 正式π0.5适配](P6_POLICY_TRAINING.md) | 冻结后端/同预算解冻两轨，完整动作训练及机制保留 |
| 10 | [P7 完整闭环](P7_CLOSED_LOOP.md) | 原协议全量评测、扩展机制任务、统计与主张边界 |

## 本次已解决的设计空缺

- 数据层：保留旧loader数值转换，新增显式分区消费接口；轨迹及其照明衍生版本成组，开发与校准分开。
- 测量层：求解拟合与独立预测检查分开；同一测量、重叠尺度、前缀重放不重复计数。
- 求解层：Theseus提供连续提议，完整能量决定是否接受；异方差、饱和概率和结构先验不会被误当普通平方残差。
- 世界层：先建立同一世界候选的所有相机视图，再选择整世界MAP；跨相机的各自MAP不能拼成一个假世界。
- 推断层：概率参考与近似搜索分开；支持撤回和数值缓存失效分别处理；旧神经状态通过重放重建。
- 系统层：共享真实训练/推理入口；旧短训入口不充当正式训练器；先做两次早期预检。

这些是本交接包对v1.1的首版实现细化，不是已经验证的算法性质。参数、精度容差和统计协议按共同契约在P0开发试验后冻结。

## 阅读与复核材料

- [远程开发执行说明](03_REMOTE_DEVELOPMENT.md)：SSH、首次接入、实际数据/权重、独立环境及远端交付。
- [当前实现审计](01_CURRENT_IMPLEMENTATION.md)：哪些可以复用，哪些缺失，哪些历史结果不能沿用。
- [共同实现契约](00_SHARED_CONTRACT.md)：阶段间接口和确定的首版选择。
- [论文与固定源码导航](02_REFERENCES.md)：逐文件链接、用途、移植差异和核查层级。
- [来源检索记录](source_manifest.json)：本轮实际取回文件的SHA256；不等于运行复现。
- [首版准备检查收据](preparation_validation.json)：2026-10-13 历史测试/链接/文件 hash，不是当前修订的重新验证。
- [远程优先修订收据](remote_development_revision_2026-10-14.json)：本次只读 SSH 与提示词链接/格式检查；不代表远端部署或模型训练完成。

后期提示词须在执行前读取前一阶段真实交付。若接口或测量证据变化，先在对应提示词更新实际文件/配置映射，研究主线仍按v1.1，不直接运行一份已经过时的命令。

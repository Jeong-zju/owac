# N01 v1.0 DeepSeek 分阶段开发提示词

> **2026-10-15 版本迁移：本目录为历史交接包。** 当前路线是 [v1.2](../../N01_full_development_plan_2026-10-15_v1.2.md)，[新P3/P4/P5完整执行包](../n01_v1_2/README.md)已经准备。P0—P2 用于追溯原实现和验收；P3—P8 的旧局部阶段已被取代，不再执行。下方启动语与阶段表保留为历史，不能从 P0 重启或仅改编号继续使用。

2026-10-14。依据 [N01 v1.0 方案](../../N01_full_development_plan_2026-10-14_v1.0.md)、当前 worktree 源码和实验记录编写，包含 P0–P8 九份阶段提示词、共同接口、实现审计、参考代码与远程运行说明。这是开发交接包；新算法与正式训练尚未开始。

## 原启动语（历史，不再执行）

将下面启动语发给能访问仓库的 DeepSeek。每次只发一个阶段，完成后把报告、diff 与实际收据交回本聊天独立核查，再决定下一阶段。各阶段已经给出默认算法、修改文件、接口、矩阵和出口，不需要 DeepSeek 自行规划研究路线。

```text
请进入 /home/jeong/zeno/owac/worktrees/n01-evidence-integration，继续分支 lzh/n01-evidence-integration。
完整阅读 AGENTS.md、README.md、docs/status.md，再依次阅读：
docs/prompts/n01_v1_0/00_SHARED_CONTRACT.md
docs/prompts/n01_v1_0/01_CURRENT_IMPLEMENTATION.md
docs/prompts/n01_v1_0/02_REFERENCES.md
docs/prompts/n01_v1_0/03_REMOTE_EXECUTION.md
docs/prompts/n01_v1_0/P0_FOUNDATION.md
本次只执行 P0，按提示词实际实现、检查、保存收据与阶段报告，不只给建议。
实际 GPU 训练优先使用 ssh -p 44327 root@107.209.104.125；需要界面时加 -L 8080:localhost:8080。
服务器数据和策略环境位置以远程说明与现场核查为准。
完成后停在 P0 交接点，给出修改清单、实际命令、测试/资源结果和下一阶段需要的文件。
```

后续替换最后一个文件和阶段编号即可。如果 DeepSeek 不能读取本机文件，可从[研究统筹台](../../../../../repo/outputs/research-hub/index.html)的 N01 阶段复制完整交接包；它包含共同契约、审计、参考导航、远程说明和当阶段正文。代码仓库本身仍需提供给执行者，提示词不能代替源码。

## 分阶段文件

| 顺序 | 提示词 | 明确要完成的工作 |
| --- | --- | --- |
| P0 | [基础与冻结](P0_FOUNDATION.md) | 四区数据、真实路径、来源锁、内部类型、独立 GPU 环境与资源画像 |
| P1 | [解析参考](P1_ANALYTIC_REFERENCE.md) | 唯一来源、局部 GLS、merge/split/retract、相关协方差、独立全量回放 |
| P2 | [运动与噪声](P2_RAW_MOTION.md) | 完整 SEA-RAFT RAW 适配、单位/相位、亚像素对应和风险校准 |
| P3 | [时空支持](P3_SPATIOTEMPORAL_SUPPORT.md) | BOCPD/鲁棒对照、时间来源退出、域级相容、局部分裂与 T3 支路 |
| P4 | [成熟特征与匹配训练](P4_MATCHED_FEATURE_TRAINING.md) | 来源控制实际内容、完整 DBSR、7组×3种子、B3直接对照与来源消融 |
| P5 | [状态和期限](P5_STATE_REPLAY_DEADLINE.md) | 官方 Mamba、干净检查点重放、完整依赖失效、可中断快照与实测成本 |
| P6 | [真实 RAW 与迁移](P6_REAL_RAW_TRANSFER.md) | 真实数据身份、两种读出任务、BSRT第二骨干、跨条件与规模 |
| P7 | [正式动作训练](P7_POLICY_TRAINING.md) | 新完整模型真实桥、观测时序、同范围后端解冻、三种子充分训练 |
| P8 | [全量闭环](P8_FULL_CLOSED_LOOP.md) | 逐checkpoint完整13250身份、扩展动态协议、配对统计与总成本 |

[早期策略预检](EARLY_POLICY_PREFLIGHT.md)分别在 P1/P2 末执行，及早检验真实数据/形状/梯度。它不替代 P7，也不改变 P0–P8 的阶段编号。

## 本轮已经具体解决的交接问题

- 旧 DBSR 仅支持四平面 CFA；给三通道 RawVLA 定义独立 stem 适配及公平重训。
- 旧空间图无法让同坐标旧时间源退出；P3 用时间分段和来源成员直接表达撤回。
- 撤回后 CNN 感受野和融合权重仍可能有旧内容；P4 先按域重新编码整个受影响源。
- 相关噪声删除必须重建 covariance 子块；不能截旧 precision 或线性插值 variance。
- SEA-RAFT 原输入、归一化与自动 ImageNet 初始化一起适配；Mamba 两种 cache 一起重放。
- 旧动作训练器绑定旧 N01、冻结后端且恢复状态不完整；P7 明确新工厂/训练器/服务职责。
- RoboTwin 训练观测间隔尚未核验；P7 在正式动作训练前固定真实观测到达规则。
- 主 checkout 已有完整 Default 基线，旧 N01 文档的 RoboTwin 未完成状态保留为历史；P0/P8读取新记录。

## 证据与使用范围

- [当前实现审计](01_CURRENT_IMPLEMENTATION.md)：可复用资产、已发现缺口、旧结果范围。
- [共同契约](00_SHARED_CONTRACT.md)：默认算法、类型、时间/来源语义、训练和交付。
- [参考导航](02_REFERENCES.md)：25项研究来源、固定代码逐文件入口与移植边界。
- [远程执行](03_REMOTE_EXECUTION.md)：用户服务器、真实数据/权重路径、环境、supervisor与恢复方式。
- [源码检索收据](source_manifest.json)、[远程只读快照](remote_snapshot.json)、[交付检查](preparation_validation.json)。

后期提示词以当前设计为依据。执行前读取前一阶段真实交付；若研究负责人根据实验改变接口/路线，先更新相关提示词与契约再下发。当前所有模型/数据超参均按其所在阶段注明开发候选或正式冻结状态，不把规划预算当训练已充分。

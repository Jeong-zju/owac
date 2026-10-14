# 研究记录

RawVLA 训练/评测的完整设置和数字证据见[本机归档](archives/2026-10-14_rawvla_pi05_baseline.md)，未来实验使用[比较模板](templates/rawvla-comparison.json)。

当前实验参照为 RawVLA-Bench / π0.5，部署与协议标签 `owac-rawvla-pi05-v0.1.0`；见 [0012](../docs/decisions/0012-rawvla-pi05-protocol-freeze.md)及[全量基线](experiments/2026-10-10_rawvla_pi05_full_baseline.md)。2026-10-14 两平台完整基线已取得：LIBERO 58.20%、RoboTwin 15.63%；主清单 13,250/13,250，原条件 632 条独立计分。见[完整报告](experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)，论文数值尚未复现。

研究依据：[OWAC roadmap v6](../docs/OWAC_research_roadmap_2026-10-08.md)。当前 12 张卡仅从路线图建立种子索引，尚未完成设计、实现或实验，所有负责人和评审安排待定；家族名称不代表已证明的创新。无默认主线，允许新增、拆分、合并和复活候选。

| 候选 | 家族 | 状态 | 负责人 |
| --- | --- | --- | --- |
| [F01](candidates/F01.md) | 隔室式证据计算 | `seed` | 未定 |
| [F02](candidates/F02.md) | 预测与修正执行 | `seed` | 未定 |
| [F03](candidates/F03.md) | 约束与平衡态表征 | `seed` | 未定 |
| [F04](candidates/F04.md) | 相位与同步关系计算 | `seed` | 未定 |
| [F05](candidates/F05.md) | 可编程局部场 | `seed` | 未定 |
| [F06](candidates/F06.md) | 变化触发的运行时 | `seed` | 未定 |
| [F07](candidates/F07.md) | 多时间尺度连续状态 | `seed` | 未定 |
| [F08](candidates/F08.md) | 快慢可塑记忆 | `seed` | 未定 |
| [F09](candidates/F09.md) | 测量几何驱动 | `seed` | 未定 |
| [F10](candidates/F10.md) | 一致性组合与路由 | `seed` | 未定 |
| [F11](candidates/F11.md) | 解析多尺度与统计表征 | `seed` | 未定 |
| [F12](candidates/F12.md) | 可逆状态与延迟压缩 | `seed` | 未定 |

[Cui 专题综述](reviews/2026-10-08_ziteng_cui_raw_physical_intelligence.md)梳理 12 条相关工作与具身增量要求；[来源清单](reviews/2026-10-08_ziteng_cui_raw_sources.json)区分论文版本、作者页录用信息和访问范围。候选卡 v0.3 在此前具身假设上增补生物机制问题，尚未完成新规则推导；不因新增研究题目改变 `seed` 状态。

[生物弱光与补全综述](reviews/2026-10-09_bio_lowlight_inference.md)将 N01—N06 用于机制探索、F01—F12 用于架构设计、PI1—PI5 用于具身评价；[来源记录](reviews/2026-10-09_bio_lowlight_sources.json)区分生物观察和工程假设。六条机制不是必须串联的模块。

## 使用顺序

1. 从 [candidate 模板](templates/candidate.md)补齐候选卡；同时记录完整框架草图和最小机制实验，未解决部分明确标为待定。按自身假设选神经、解析或混合实现，不强制 token、固定层数或 F01 状态。
2. 运行前复制 [experiment 模板](templates/experiment.md)到 `experiments/YYYY-MM-DD_<candidate>_<purpose>.md`，登记假设、反证与预算；运行后补齐版本、真实资源和输出引用。失败、中止与未测量项同样保留。
3. 每两周参考 [decision 模板](templates/decision.md)生成 `decisions/YYYY-MM-DD_portfolio.md`（首次使用时建目录）。按多维证据与诊断分配资源，不使用单一加权总分淘汰候选。
4. 实验记录关联候选卡，组合评审关联实验记录；同步此索引和卡片状态。使用明确、可解释的状态记录实际成熟度，不能以时间表代替完成证据。

模板中的“待定/未测量”须保留到有设计或实测证据；估算单列，不补写为结果。卡片可逐步扩展，但进入下一层的证据与未解决问题必须可追溯。RSS 投稿冻结仅约束所报告的代码、结果和协议，持续候选池独立维护。

原始数据、大型权重、日志与生成结果保存在外部存储或仓库忽略的产物目录中，研究记录只保存可定位的 URI/路径、版本与校验值。仅记录脱敏的运行信息，不提交凭据或个人敏感信息。

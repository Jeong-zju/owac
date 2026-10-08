# 研究记录

研究依据：[OWAC roadmap v4](../docs/OWAC_research_roadmap_2026-10-08.md)。当前 12 张卡仅从路线图建立种子索引，尚未完成设计、实现或实验，所有负责人和评审安排待定；家族名称不代表已证明的创新。无默认主线，允许新增、拆分、合并和复活候选。

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

## 使用顺序

1. 从 [candidate 模板](templates/candidate.md)补齐候选卡；同时记录完整框架草图和最小机制实验，未解决部分明确标为待定。按自身假设选神经、解析或混合实现，不强制 token、固定层数或 F01 状态。
2. 运行前复制 [experiment 模板](templates/experiment.md)到 `experiments/YYYY-MM-DD_<candidate>_<purpose>.md`，登记假设、反证与预算；运行后补齐版本、真实资源和输出引用。失败、中止与未测量项同样保留。
3. 每两周参考 [decision 模板](templates/decision.md)生成 `decisions/YYYY-MM-DD_portfolio.md`（首次使用时建目录）。按多维证据与诊断分配资源，不使用单一加权总分淘汰候选。
4. 实验记录关联候选卡，组合评审关联实验记录；同步此索引和卡片状态。使用明确、可解释的状态记录实际成熟度，不能以时间表代替完成证据。

模板中的“待定/未测量”须保留到有设计或实测证据；估算单列，不补写为结果。卡片可逐步扩展，但进入下一层的证据与未解决问题必须可追溯。RSS 投稿冻结仅约束所报告的代码、结果和协议，持续候选池独立维护。

原始数据、大型权重、日志与生成结果保存在外部存储或仓库忽略的产物目录中，研究记录只保存可定位的 URI/路径、版本与校验值。仅记录脱敏的运行信息，不提交凭据或个人敏感信息。

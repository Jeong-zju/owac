# OWAC 当前状态

2026-10-19：用户暂停 N01、N02、N04，要求三个 worktree 合并回主线。远程服务器已不可达；本轮只合并、保全本地证据并运行本地代码检查。当前入口为 `repo/` 的本地 `main`，不自动恢复训练、诊断训练或下一阶段。

[合并与证据索引](../research/archives/2026-10-19_worktree_consolidation/README.md)记录原分支/提交、接口冲突处理、本地归档与远端缺口。四处原始状态快照分别保存在该目录的 `baseline/`、`n01/`、`n02/`、`n04/`，历史状态不作为当前作业状态。

| 方向 | 最后可查本地证据 | 当前结论 |
| --- | --- | --- |
| N01 v1.2 P3 | [停止记录](../research/experiments/n01_v1_2/P3_training_stopped_20261019.md)、[动作诊断](../research/experiments/n01_v1_2/P3_action_diagnosis_20261019.md)、[代码同步](../research/experiments/n01_v1_2/P3_code_sync_20261019.md) | RAW/RGB 各 0/12；正常光同场景作者 4/4、两臂各 0/4。专家观测动作已明显偏离，优先定位视觉—动作适配。未通过完整研究验收。 |
| N02 v1.2 P2/r17 | [r11 闭环收据](../research/experiments/n02_v1_2/closed_loop_r11_20261019/receipt.json)、[r17 结论](../research/experiments/n02_v1_2/residual_joint_r17_20261019/conclusions.json)、[r17 审计](../research/experiments/n02_v1_2/residual_joint_r17_20261019/) | r11 0/3；r17 新 Normal 19 条专家合格示范、423 窗口的小样本动作读出有冻结 RAW 收益，局部联合更新没有额外总体收益。有效 6144＋保留失败 1024＝实际 7168 更新；π 更新 0、学习策略闭环 0，不转为机器人成功。 |
| N04 v1.1 S2 | [v4.2 修复](../research/experiments/2026-10-16_n04_aux_repair_v4.md)、[v5 故障与修复](../research/experiments/2026-10-19_n04_stability_repair_v5.md)、[代码同步](../research/experiments/2026-10-19_n04_server_code_sync.md) | 历史开发闭环 0/100；v4.2 对应越界/目标消失、潜在量纲增长与 RoboTwin 退出保留。v5 短程零更新/优化/重载通过，只支持工程修复，长期稳定与任务收益未验收。 |

N04 v5 本地最后运行收据保存到 step10；这不是本轮确认的最终停止步。N01/N02 的已完成或停止回执同样以其各自记录时间为准。现在不能查询远端最终进程、日志、checkpoint 或存储状态，不能宣称远端独有权重已恢复或已丢失。

完整 Default π0.5 基线仍为 [2026-10-14 全量报告](../research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)：LIBERO 5820/10000（58.20%）、RoboTwin 508/3250（15.63%）；附加原条件 632 条另计。完整分母已取得，论文绝对数字未复现。原协议标签和历史负结果保持不动。

研究统筹台现从主 checkout 读取三条路线，当前节点为“暂停与本地证据”；旧提示词可阅读，但不会作为当前执行包发出。未来是否统一架构或恢复实验由新的研究决定确定。

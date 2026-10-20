# OWAC 当前状态

2026-10-20 **PREP 已验收通过，进入 ALIGN**；当前执行版本 OWAC-D1.1。教师／学生历史、相机状态与缓存对接及真实零更新梯度检查仍待完成；学生未训练、控制未恢复。见[阶段验收记录](../research/experiments/raw_stem_siglip_20261020/teacher-independent-review.md)。

2026-10-20 限定教师阶段已验收：[独立报告](../research/experiments/raw_stem_siglip_20261020/teacher-independent-review.md)。Normal task6共同20新初态：参照20/20、RAW教师20/20，仅开发筛查。选定前端初始化0；实际155前端更新／1240 anchors未带来已证明收益，未作为选定权重。验证9成功＋1人为中断保留完整N10；50回放／50策略场次已用完。学生0更新／0闭环，尚未恢复；旧N01／N02／N04仍暂停，不授权学生／弱光／实机扩展。


2026-10-20 授权前教师取得补查记录：官方6公开分支／main32历史提交与公开主页未找到训练后的RawVLA前端；完整源码已取得并独立合并配置，后端已恢复但不等于完整教师。已写[限定单任务教师复现提案](plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md)，拟仅训前端、累计≤2000更新，执行待明确授权；0优化、0新闭环。保留因果burst／递归，不固定单帧。


2026-10-20 授权前工程交付记录：每帧 RAW stem／SigLIP 蒸馏基线与受授权门控的训练、纯 artifact 评估入口已可审阅；68 项针对性测试通过，实际同源后端恢复及合成输入数值／梯度核对通过。新增优化更新0、闭环0，控制未恢复。完整 RAW 教师权重、真实配对数据和时序训练端到端尚未验收；不固定最终策略为单帧。见 [独立验收](../research/experiments/raw_stem_siglip_20261020/independent-review.md)。


2026-10-20 当前实施路线收敛到[RAW stem＋继承 SigLIP＋蒸馏](plans/RAW_STEM_SIGLIP_DISTILLATION_2026-10-20.md)。用户认可方案，执行 prompt 与采用决议已落库；统筹台默认展示此主线、完整提示词和可点击的分支／收敛历史。当前版本 OWAC-D1.1：主教师对外接收 RAW，内部可经 Neural ISP 成像；教师和学生不固定单帧，以控制效果、稳定性和成本选择因果历史／状态。学生部署直接 RAW→tokens，首轮后端冻结。工程实现进行中，本次修订未训练教师／学生或运行新闭环。作者模型实现已公开，训练后 RawVLA 前端／完整教师公开权重尚未找到，见[资源核查](../research/reviews/2026-10-20_rawvla_public_teacher_assets.md)。

2026-10-19 的[ConvNeXt 单任务视觉替换方案](../research/experiments/2026-10-19_visual_replacement_recovery.md)及其单帧前端转为诊断历史；旧后端 LoRA 配方不沿用。50 条真实示范、每适配臂累计最多 5,000 更新作为新方案首轮诊断规模，不能视为已执行或充分收敛。配套 RawVLA runtime、LIBERO 策略／统计／tokenizer、真实配对示范及初态仍须核验；DROID 权重不能替代。

2026-10-19 本地 worktree 清理完成：三个候选 checkout 均已移除，当前只保留 `/home/jeong/zeno/owac/repo/` 的 `main`，后续直接在此开发。清理前重新核验提交祖先关系、工作区、91,270 个证据文件与归档哈希；另保全 N02 本地第三方源码。历史分支、代码快照和主仓库证据归档保留，见[清理收据](../research/archives/2026-10-19_worktree_consolidation/worktree-cleanup.json)。

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

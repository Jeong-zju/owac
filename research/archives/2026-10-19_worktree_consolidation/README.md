# N01、N02、N04 主线合并与证据保全

2026-10-19（Asia/Shanghai），用户要求三个独立 worktree 收敛到主线，并保留既有结论和训练记录；远程服务器已不可达。当前三个方向全部暂停。历史文档的训练授权、RUNNING、PID 和后续阶段要求仅为原时点记录，不作为恢复授权或当前远端状态。

## 代码与历史

主 checkout 为 `/home/jeong/zeno/owac/repo/`，交付分支为本地 `main`。先在四个原分支分别提交未提交代码和证据，再在 `lzh/consolidate-paused-research` 顺序合并三个候选，完成本地检查后将 `main` 快进到整合结果。原分支与 worktree 保留；未推送 GitHub。原部署标签没有移动。

| 来源 | 合并前 HEAD | 保全提交 | 原有未提交文件数 |
| --- | --- | --- | ---: |
| 主 checkout / `lzh/pi05-sim-baseline` | `febf131d6d77d4a616170f8966bbf79d77dbd6ce` | `6465a009e7fec073abbfc7512b6bc25592fc5d42` | 9 |
| N01 / `lzh/n01-evidence-integration` | `d3dec908d78d0a86a761abc985b83113d836c8cf` | `e3efa7bce33e31d8fe3f03d648007f67658e0f8d` | 151 |
| N02 / `lzh/n02-measurement-completion` | `fe89b7f008ff77c1859df36ea6dc2a6ae12a1b95` | `0a7ce303b42d27d715c7b2b0df580f84c677b0b6` | 161 |
| N04 / `lzh/n04-egomotion-separation` | `a81cf25148870fd83fbc06a4fabf374510493978` | `855ec09eb4ab2688a46feb2e011f9d64c9c56df3` | 256 |

[提交收据](snapshot-commits.json)记录这些提交及原 checkout。`baseline/`、`n01/`、`n02/`、`n04/` 中的 `AGENTS.snapshot.md`、`README.snapshot.md`、`status.snapshot.md` 保留各分支入口原文；`checkout-inventory.json` 记录合并前 tracked/untracked 文件的 SHA-256 和符号链接。原实验记录的提交 SHA、路径、预算、缺陷、撤回、失败和未验证项均不因合并改写。

## 保留的结论

- **N01 v1.2 P3**：[停止记录](../../experiments/n01_v1_2/P3_training_stopped_20261019.md)、[20k 开发闭环](../../experiments/n01_v1_2/P3_20k_closed_loop_probe.md)、[动作诊断](../../experiments/n01_v1_2/P3_action_diagnosis_20261019.md)。RAW/RGB 各 0/12；同四个正常光场景原作者策略 4/4、两臂各 0/4。小样本专家观测上动作已明显偏离；32 维速度场损失的补齐维度会稀释真实 7 维误差。不能将低损失解释为任务能力，也不能只归因于闭环累积误差。旧留出声明的撤回和 P0—P2 限制保留。
- **N02 v1.2 P2 / r13—r17**：[r11 闭环](../../experiments/n02_v1_2/closed_loop_r11_20261019/receipt.json)、[r17 结论](../../experiments/n02_v1_2/residual_joint_r17_20261019/conclusions.json)。r11 所选 RAW850 闭环 0/3。r17 在 Normal、同物体/背景、19 条合格专家示范、423 窗口的小样本上，留出关节 RMS 为固定状态 8.17°、状态残差 7.90°、冻结 RAW 6.39°、局部联合 RAW 6.49°、原生视觉 4.36°；冻结 RAW 有动作读出收益，当前联合范围无额外总体收益。有效矩阵 6144＋保留失败联合 1024＝实际 7168 次更新；π 更新 0、学习策略闭环 0。原生视觉的输入/预训练差异和动作时间域限制保留，不把专家成功当学习策略成功。
- **N04 v1.1 S2**：[独立审计](../../experiments/2026-10-16_n04_v11_s2_independent_audit.md)、[v4.2](../../experiments/2026-10-16_n04_aux_repair_v4.md)、[v5](../../experiments/2026-10-19_n04_stability_repair_v5.md)。历史开发闭环 0/100。v4.2 对应越界导致有效辅助目标消失、潜在尺度增长、RoboTwin 数值保护退出等故障保留；v5 短程完整模型优化/确定性重载只支持工程修复。最后本地 v5 收据保存到 step10，不作为现在的最终停止步或长期稳定证据。
- **完整 Default π0.5**：[全量报告](../../experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)的 LIBERO 58.20%、RoboTwin 15.63% 及 13,250 条主分母保留；论文数值未复现。

所有来源分支的 `research/experiments/` 和 `research/reviews/` 按文件逐一核对。主 checkout 中较新的共享基线记录保留原路径；候选分支中 14 份同路径旧版本按内容去重另存 `historical-evidence/`，逐份保留原字节并登记来源。结果见[实验文件保全审计](evidence-tree-audit.json)。这是本地文件完整性检查，没有重新运行这些实验或核实远端现状。

## 接口冲突的处理

三个分支在同名共享文件中形成了不同的数据结构、动作桥接和加载 API。合并按候选隔离，保留原语义；本次不重新设计算法或统一接口。

| 候选 | 编码契约 | 数据加载器 | 特征桥 |
| --- | --- | --- | --- |
| N01 | `owac.core.encoder_contract` | `owac.raw.rawvla_sequence` | `owac.backends.rawvla_features` |
| N02 | `owac.core.encoder_contract_n02` | `owac.raw.rawvla_sequence_n02` | `owac.backends.rawvla_features_n02` |
| N04 | `owac.core.encoder_contract_n04` | `owac.raw.rawvla_sequence_n04` | `owac.backends.rawvla_features_n04` |

相关候选的当前源码、测试与打包入口随模块名调整；历史实验目录中的冻结探针和原收据保持原字节，复现时应恢复原分支保全提交/冻结源码。当前集成代码的 SHA 不冒充旧训练 SHA。公共 RoboTwin 客户端同时保留 N02 可选执行协议及 N04 可选本体历史，默认原生执行节奏保持 50 步。依赖与 vendor 许可/包数据取并集，本地 encoder 固定 CPU torch 2.9.1，远端 GPU 环境继续由历史部署记录解释。

[接口字节审计](isolated-interface-audit.json)确认六份隔离模块相对原分支只含已声明的模块名替换，没有修改候选内部计算。合并后的本地复验另处理了三项明确问题：原生 50 步节奏保留可审计的真实默认分支，N04 CPU profiling 缺失 GPU 统计时返回空缺值，N02 参考夹具存在性检查使用真实导出文件名和 `is_file()`。N04 RAFT 实际导入的 SciPy 已加入 encoder 依赖并锁定。

从 N02 原 worktree 逐字恢复了 17 份已有参考夹具/收据到主 checkout 的 `outputs/n02-full/p0/parity/`，没有重新生成参考值，见[恢复文件校验](n02/restored-local-reference-fixtures.json)。历史原始源码和失败测试日志仍保留。

研究统筹台登记改为从 `repo/`、`main` 读取三个方向，当前节点为“暂停与本地证据”；历史阶段原文可读，旧执行包不作为当前复制入口。主 AGENTS、README 和 status 以暂停状态为准。

## 本地证据与远端缺口

本地恢复根为 `outputs/worktree-consolidation-20261019/`，不进入 Git 的大产物仍按原规则存储。每个候选包含：

- `checkout-before.tar.gz`、`checkout-before.json` 和 `worktree-before.patch`：合并前源码及未提交内容。
- `local-evidence.tar.zst`、`local-evidence-manifest.json`、`archive-receipt.json`：原 worktree 的本地实验输出，以及被忽略的 research 数组/日志；逐成员解压读取、核对大小与 SHA-256。符号链接保存原目标，不将它引用的外部文件冒充已归档。
- `namespace-changes.json`：N02/N04 活动代码的接口路径调整清单。

`premerge.bundle` 保存已有 Git refs 的完整历史，已通过 `git bundle verify`；四处原 checkout 的源码快照另行保存。原 worktree、outputs、权重和数据均未删除。主 checkout 既有基线输出继续留在原目录。

归档只覆盖本机实际已有文件。远端独有 checkpoint、数据集、原始长日志或视频若未同步到本地，就仍属于缺口；历史路径、来源清单和最后收据不能证明这些文件现在可访问、已完整保全或已经丢失。用户报告服务器不可达，本轮未尝试连接，也未启动任何训练/恢复队列。

本次已完成并解压逐成员验证的归档如下，字节数为十进制；合计 91,270 个文件/链接，28,288,067,665 字节原始内容，13,959,188,530 字节归档。各自的 `archive-receipt.json` 保存归档和清单哈希。

| 方向 | 文件/链接 | 原始字节 | 归档字节 | 校验 |
| --- | ---: | ---: | ---: | --- |
| N01 | 4,740 | 1,563,674,060 | 1,194,804,696 | 全部通过 |
| N02 | 83,656 | 25,941,937,148 | 12,172,252,073 | 全部通过 |
| N04 | 2,874 | 782,456,457 | 592,131,761 | 全部通过 |

恢复时先校验清单与归档 SHA-256，再在新的 `worktrees/<recovery-name>/` checkout 恢复相应保全提交；归档只解压到新目录，保持本地主线和原证据不被覆盖。环境与远端权重不足时应明确无法重跑，不用新实现替代历史实验。

## 本地验证

四个来源快照保存前的 `make check` 均通过。各次合并的静态检查也已通过；最终完整测试、归档计数和主线祖先关系将在完成后写入本目录的验证收据。检查仅支持代码整合与本地证据保全，不提升任何候选的科学结论或机器人能力结论。

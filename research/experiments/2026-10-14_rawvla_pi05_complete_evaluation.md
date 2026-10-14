# RawVLA-Bench π0.5 完整仿真评测报告

本报告为 OWAC 后续算法提供可逐条配对的 Default π0.5 参照。2026-10-14 已修复主 RoboTwin 的 33 个基础设施缺口，并沿用原身份补跑；LIBERO 10,000 条与 RoboTwin 3,250 条均完整，身份审计异常为 0。主结果分别为 **58.20%** 和 **15.63%**；另有 632 条原缓存条件，独立成绩 **17.09%**。额外训练 0 步。

这份结果完成了冻结清单的闭环覆盖，可以用于本机同协议算法比较。**论文数字尚未复现**：原条件 RoboTwin 的 17.09% 明显低于[论文 Default](https://arxiv.org/html/2609.37530v1)的 46.84%；原 LeRobot 数值等价、公开论文逐条评测记录和原 RNG 序列尚未建立。完整分母与论文数值复现分别判定。

报告生成时间：2026-10-14T02:44:24Z（UTC）。[完整数字汇总](results/2026-10-14_rawvla_pi05_complete_evaluation/complete-summary.json)、[身份审计](results/2026-10-14_rawvla_pi05_complete_evaluation/identity-audit.json)、[逐 episode 数字参照](results/2026-10-14_rawvla_pi05_complete_evaluation/per-episode-baseline.jsonl)、[全部 RoboTwin 实际动作审计](results/2026-10-14_rawvla_pi05_complete_evaluation/robotwin-action-audit.json)。原部署/协议标签 `owac-rawvla-pi05-v0.1.0` 保持不变。

## 范围与计分

本次使用作者两套主 manifest：LIBERO 为四套任务各 10 个、每任务 50 个初态、每初态五个照明域，共 10,000 条；RoboTwin 为 13 个任务、每任务 50 个作者 dataset seed、每 seed 五域，共 3,250 条。每个原身份有且只有一个有效结果，策略未完成任务仍计入分母。基础设施异常原记录保留，通过同身份补跑补齐，没有换种子或按成功结果选取重跑。

| 评测范围 | 有效完成 | 策略成功 | 正常策略失败 | 成功率 |
| --- | --- | --- | --- | --- |
| LIBERO 主清单 | 10000 | 5820 | 4180 | 58.20% |
| RoboTwin 主清单 | 3250 | 508 | 2742 | 15.63% |
| RoboTwin 原缓存条件（独立计分） | 632 | 108 | 524 | 17.09% |

两平台合计 **13,250/13,250** 条主评测完成；632 条附加核查独立计分，全部数字参照共 13,882 条。两个环境的任务、分母和控制协议不同，因此报告保留各平台成绩。范围是论文主仿真中的 **Default π0.5**；其他 backbone、学习 ISP 方法、全部干扰/消融矩阵和实体机器人实验不属于本报告。

## 照明域与任务成绩

每个主 LIBERO 照明域 2,000 条，每个主 RoboTwin 照明域 650 条。附加原条件分母为 127/127/127/126/125。

| 照明域 | LIBERO 成功/完成 | LIBERO 成功率 | RoboTwin 成功/完成 | RoboTwin 成功率 | 原条件成功/完成 | 原条件成功率 |
| --- | --- | --- | --- | --- | --- | --- |
| ExtremeLow | 251/2000 | 12.55% | 23/650 | 3.54% | 4/127 | 3.15% |
| Low | 1728/2000 | 86.40% | 103/650 | 15.85% | 28/127 | 22.05% |
| Normal | 1918/2000 | 95.90% | 170/650 | 26.15% | 39/127 | 30.71% |
| Over | 1510/2000 | 75.50% | 123/650 | 18.92% | 27/126 | 21.43% |
| ExtremeOver | 413/2000 | 20.65% | 89/650 | 13.69% | 10/125 | 8.00% |

### LIBERO 套件与论文布局

| LIBERO 套件 | 成功/完成 | 成功率 |
| --- | --- | --- |
| libero_10 | 1288/2500 | 51.52% |
| libero_goal | 1558/2500 | 62.32% |
| libero_object | 1488/2500 | 59.52% |
| libero_spatial | 1486/2500 | 59.44% |

SOG 为 Spatial、Object、Goal 合计，每域 1,500 条；Long 为 LIBERO-10，每域 500 条。Avg 按所有任务的等量初态汇总，等于 0.75×SOG + 0.25×Long；不是两个列的简单平均。

| 照明域 | SOG 成功/完成 | SOG | Long 成功/完成 | Long | Avg |
| --- | --- | --- | --- | --- | --- |
| ExtremeLow | 157/1500 | 10.47% | 94/500 | 18.80% | 12.55% |
| Low | 1334/1500 | 88.93% | 394/500 | 78.80% | 86.40% |
| Normal | 1469/1500 | 97.93% | 449/500 | 89.80% | 95.90% |
| Over | 1284/1500 | 85.60% | 226/500 | 45.20% | 75.50% |
| ExtremeOver | 288/1500 | 19.20% | 125/500 | 25.00% | 20.65% |
| Overall | 4532/7500 | 60.43% | 1288/2500 | 51.52% | 58.20% |

### LIBERO 四十个任务

每行 250 条，每照明域 50 条。任务 ID 沿用原 manifest，指令来自实际计分记录。

| 套件/任务 ID | 任务指令 | 成功/250 | 总成功率 | ExtremeLow | Low | Normal | Over | ExtremeOver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libero_10/0 | put both the alphabet soup and the tomato sauce in the basket | 130 | 52.00% | 2.00% | 72.00% | 94.00% | 36.00% | 56.00% |
| libero_10/1 | put both the cream cheese box and the butter in the basket | 130 | 52.00% | 0.00% | 70.00% | 98.00% | 62.00% | 30.00% |
| libero_10/2 | turn on the stove and put the moka pot on it | 156 | 62.40% | 66.00% | 98.00% | 92.00% | 56.00% | 0.00% |
| libero_10/3 | put the black bowl in the bottom drawer of the cabinet and close it | 147 | 58.80% | 50.00% | 100.00% | 94.00% | 40.00% | 10.00% |
| libero_10/4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 100 | 40.00% | 4.00% | 78.00% | 92.00% | 26.00% | 0.00% |
| libero_10/5 | pick up the book and place it in the back compartment of the caddy | 173 | 69.20% | 10.00% | 82.00% | 100.00% | 78.00% | 76.00% |
| libero_10/6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 89 | 35.60% | 8.00% | 62.00% | 86.00% | 22.00% | 0.00% |
| libero_10/7 | put both the alphabet soup and the cream cheese box in the basket | 183 | 73.20% | 4.00% | 96.00% | 98.00% | 90.00% | 78.00% |
| libero_10/8 | put both moka pots on the stove | 81 | 32.40% | 38.00% | 54.00% | 54.00% | 16.00% | 0.00% |
| libero_10/9 | put the yellow and white mug in the microwave and close it | 99 | 39.60% | 6.00% | 76.00% | 90.00% | 26.00% | 0.00% |
| libero_goal/0 | open the middle drawer of the cabinet | 153 | 61.20% | 0.00% | 74.00% | 96.00% | 90.00% | 46.00% |
| libero_goal/1 | put the bowl on the stove | 173 | 69.20% | 6.00% | 94.00% | 100.00% | 100.00% | 46.00% |
| libero_goal/2 | put the wine bottle on top of the cabinet | 132 | 52.80% | 0.00% | 80.00% | 98.00% | 76.00% | 10.00% |
| libero_goal/3 | open the top drawer and put the bowl inside | 131 | 52.40% | 0.00% | 78.00% | 96.00% | 64.00% | 24.00% |
| libero_goal/4 | put the bowl on top of the cabinet | 164 | 65.60% | 4.00% | 84.00% | 100.00% | 100.00% | 40.00% |
| libero_goal/5 | push the plate to the front of the stove | 137 | 54.80% | 16.00% | 88.00% | 96.00% | 66.00% | 8.00% |
| libero_goal/6 | put the cream cheese in the bowl | 167 | 66.80% | 8.00% | 96.00% | 100.00% | 100.00% | 30.00% |
| libero_goal/7 | turn on the stove | 169 | 67.60% | 54.00% | 98.00% | 100.00% | 76.00% | 10.00% |
| libero_goal/8 | put the bowl on the plate | 186 | 74.40% | 20.00% | 100.00% | 98.00% | 96.00% | 58.00% |
| libero_goal/9 | put the wine bottle on the rack | 146 | 58.40% | 2.00% | 74.00% | 96.00% | 76.00% | 44.00% |
| libero_object/0 | pick up the alphabet soup and place it in the basket | 124 | 49.60% | 0.00% | 74.00% | 96.00% | 78.00% | 0.00% |
| libero_object/1 | pick up the cream cheese and place it in the basket | 147 | 58.80% | 8.00% | 98.00% | 100.00% | 78.00% | 10.00% |
| libero_object/2 | pick up the salad dressing and place it in the basket | 150 | 60.00% | 12.00% | 94.00% | 96.00% | 94.00% | 4.00% |
| libero_object/3 | pick up the bbq sauce and place it in the basket | 157 | 62.80% | 14.00% | 100.00% | 96.00% | 100.00% | 4.00% |
| libero_object/4 | pick up the ketchup and place it in the basket | 146 | 58.40% | 0.00% | 88.00% | 100.00% | 100.00% | 4.00% |
| libero_object/5 | pick up the tomato sauce and place it in the basket | 156 | 62.40% | 10.00% | 96.00% | 94.00% | 98.00% | 14.00% |
| libero_object/6 | pick up the butter and place it in the basket | 153 | 61.20% | 14.00% | 100.00% | 98.00% | 94.00% | 0.00% |
| libero_object/7 | pick up the milk and place it in the basket | 144 | 57.60% | 4.00% | 88.00% | 100.00% | 96.00% | 0.00% |
| libero_object/8 | pick up the chocolate pudding and place it in the basket | 151 | 60.40% | 28.00% | 92.00% | 100.00% | 82.00% | 0.00% |
| libero_object/9 | pick up the orange juice and place it in the basket | 160 | 64.00% | 30.00% | 100.00% | 96.00% | 92.00% | 2.00% |
| libero_spatial/0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 159 | 63.60% | 6.00% | 100.00% | 100.00% | 86.00% | 26.00% |
| libero_spatial/1 | pick up the black bowl next to the ramekin and place it on the plate | 146 | 58.40% | 0.00% | 92.00% | 100.00% | 88.00% | 12.00% |
| libero_spatial/2 | pick up the black bowl from table center and place it on the plate | 191 | 76.40% | 28.00% | 100.00% | 100.00% | 98.00% | 56.00% |
| libero_spatial/3 | pick up the black bowl on the cookie box and place it on the plate | 167 | 66.80% | 10.00% | 100.00% | 100.00% | 94.00% | 30.00% |
| libero_spatial/4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 110 | 44.00% | 0.00% | 50.00% | 90.00% | 70.00% | 10.00% |
| libero_spatial/5 | pick up the black bowl on the ramekin and place it on the plate | 128 | 51.20% | 0.00% | 84.00% | 98.00% | 58.00% | 16.00% |
| libero_spatial/6 | pick up the black bowl next to the cookie box and place it on the plate | 156 | 62.40% | 10.00% | 96.00% | 100.00% | 86.00% | 20.00% |
| libero_spatial/7 | pick up the black bowl on the stove and place it on the plate | 160 | 64.00% | 14.00% | 88.00% | 100.00% | 92.00% | 26.00% |
| libero_spatial/8 | pick up the black bowl next to the plate and place it on the plate | 168 | 67.20% | 16.00% | 100.00% | 100.00% | 96.00% | 24.00% |
| libero_spatial/9 | pick up the black bowl on the wooden cabinet and place it on the plate | 101 | 40.40% | 0.00% | 62.00% | 94.00% | 44.00% | 2.00% |

### RoboTwin 十三个任务

每行 250 条，每照明域 50 条；使用作者成功判据与最大策略步数。

| 任务 | 成功/250 | 总成功率 | ExtremeLow | Low | Normal | Over | ExtremeOver |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grab_roller | 152 | 60.80% | 16.00% | 78.00% | 94.00% | 72.00% | 44.00% |
| handover_mic | 21 | 8.40% | 0.00% | 6.00% | 20.00% | 4.00% | 12.00% |
| lift_pot | 15 | 6.00% | 0.00% | 12.00% | 4.00% | 14.00% | 0.00% |
| move_can_pot | 23 | 9.20% | 2.00% | 12.00% | 14.00% | 10.00% | 8.00% |
| open_laptop | 108 | 43.20% | 18.00% | 36.00% | 58.00% | 56.00% | 48.00% |
| pick_dual_bottles | 9 | 3.60% | 0.00% | 4.00% | 8.00% | 4.00% | 2.00% |
| place_dual_shoes | 9 | 3.60% | 0.00% | 0.00% | 16.00% | 0.00% | 2.00% |
| place_object_basket | 36 | 14.40% | 2.00% | 8.00% | 32.00% | 20.00% | 10.00% |
| place_phone_stand | 24 | 9.60% | 4.00% | 6.00% | 18.00% | 10.00% | 10.00% |
| put_bottles_dustbin | 4 | 1.60% | 0.00% | 2.00% | 6.00% | 0.00% | 0.00% |
| put_object_cabinet | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| stack_blocks_two | 34 | 13.60% | 2.00% | 12.00% | 36.00% | 14.00% | 4.00% |
| stack_bowls_two | 73 | 29.20% | 2.00% | 30.00% | 34.00% | 42.00% | 38.00% |

## 基础设施修复与结果接纳

33 个缺口分别为 grab_roller 5 条、open_laptop 11 条和 put_bottles_dustbin 17 条。异常发生在 π0.5 控制前的作者专家检查：抓取规划为空时仍构造 None 目标动作，或直接索引空瓶子动作列表。原主清单保留 85 次基础设施异常尝试；此前附加条件的 18 次异常也保留。最新身份审计的未补齐列表为空。

冻结 v3 原本已设置 `ROBOTWIN_FORCE_EVAL_SEED=1`，专家规划失败不会删除 dataset seed。此次在独立修复目录中给专家检查的空分支增加保护，让任务元信息正常返回；所有原任务方法均在策略 setup 前恢复，再按原 seed 重建场景。没有向策略输入专家动作，也没有改变成功判据、照明/噪声、正常化或动作块。

三个 supervisor 队列各负责 11 个互不重叠身份，均为有限任务。33 条已全部实际执行策略，**10 条成功、23 条正常失败**，新增基础设施失败次数见[队列审计](results/2026-10-14_rawvla_pi05_complete_evaluation/supervisor-status.json)。原有 3,217 条结果字节不变，原 ledger 已另存；只向正式 ledger 追加经审计的 33 条。

本次另保留两次运行阶段的基础设施阻塞：robotwin-2394 在第 1003 步停滞，按原 900 秒上限超时后重试；robotwin-2420 在第 669 步停滞超过八分钟，核对所属模拟器进程组后只终止该子进程并重试。它们的部分轨迹均未作为正常策略失败计分。2394 的重试与第一次完整执行到卡点前的 1,002 个步骤，状态和动作逐值完全相同，并越过卡点完成 1700 步评测；2420 的卡点前 668 个完整步骤也逐值一致。这提供了运行阻塞与策略行为的区分证据。具体 native 阻塞函数尚未确认，并发模拟器可能相关，不能据此断言 GPU 驱动根因。诊断保护只写栈日志，不改变任务或 RNG。[停滞恢复记录](results/2026-10-14_rawvla_pi05_complete_evaluation/stall-recovery.json)、[重试前缀对照](results/2026-10-14_rawvla_pi05_complete_evaluation/retry-prefix-comparison.json)、[诊断覆盖哈希](results/2026-10-14_rawvla_pi05_complete_evaluation/diagnostic-overlay.json)保留处理过程。

此前控制身份 `paper-0037` 的首个十四维状态、三路 240×320 RGB、flow seed 和 50×14 动作块与原运行逐值完全相同；本次沿用该验证过的保护逻辑。[输入一致性证据](results/2026-10-14_rawvla_pi05_complete_evaluation/control-input-parity.json)的作用是确认保护分支未触发时入口不变，不能据此声称修复后所有轨迹与旧异常轨迹等价，因为旧异常没有策略轨迹。

[修复协议](results/2026-10-14_rawvla_pi05_complete_evaluation/repair-protocol.json)、[逐条接纳审计](results/2026-10-14_rawvla_pi05_complete_evaluation/repair-validation.json)、[原源码保护哈希](results/2026-10-14_rawvla_pi05_complete_evaluation/source-guards.json)、[33 条实际模型输入与首动作核验](results/2026-10-14_rawvla_pi05_complete_evaluation/repaired-input-audit.json)及[正式结果追加记录](results/2026-10-14_rawvla_pi05_complete_evaluation/adoption.json)共同保存原身份、元信息、原方法恢复、连续实际动作与哈希。全量动作审计覆盖 3,882 条 RoboTwin 主/附加结果，共 2,343,224 个实际策略步，异常 0；LIBERO 10,000 条完成身份、唯一性、配对语言及[逐条 flow seed 核验](results/2026-10-14_rawvla_pi05_complete_evaluation/libero-flow-seed-audit.json)。

| 原身份 | 任务 | 原 dataset seed | 照明域 | 真实策略步数 | 任务结果 |
| --- | --- | --- | --- | --- | --- |
| robotwin-0190 | grab_roller | 39 | ExtremeLow | 400 | 正常失败 |
| robotwin-0191 | grab_roller | 39 | Low | 89 | 成功 |
| robotwin-0192 | grab_roller | 39 | Normal | 92 | 成功 |
| robotwin-0193 | grab_roller | 39 | Over | 132 | 成功 |
| robotwin-0194 | grab_roller | 39 | ExtremeOver | 155 | 成功 |
| robotwin-1115 | open_laptop | 25 | ExtremeLow | 700 | 正常失败 |
| robotwin-1116 | open_laptop | 25 | Low | 700 | 正常失败 |
| robotwin-1117 | open_laptop | 25 | Normal | 122 | 成功 |
| robotwin-1118 | open_laptop | 25 | Over | 73 | 成功 |
| robotwin-1119 | open_laptop | 25 | ExtremeOver | 73 | 成功 |
| robotwin-1161 | open_laptop | 37 | Low | 700 | 正常失败 |
| robotwin-1180 | open_laptop | 42 | ExtremeLow | 71 | 成功 |
| robotwin-1181 | open_laptop | 42 | Low | 301 | 成功 |
| robotwin-1182 | open_laptop | 42 | Normal | 700 | 正常失败 |
| robotwin-1183 | open_laptop | 42 | Over | 700 | 正常失败 |
| robotwin-1184 | open_laptop | 42 | ExtremeOver | 51 | 成功 |
| robotwin-2330 | put_bottles_dustbin | 27 | ExtremeLow | 1700 | 正常失败 |
| robotwin-2331 | put_bottles_dustbin | 27 | Low | 1700 | 正常失败 |
| robotwin-2332 | put_bottles_dustbin | 27 | Normal | 1700 | 正常失败 |
| robotwin-2333 | put_bottles_dustbin | 27 | Over | 1700 | 正常失败 |
| robotwin-2334 | put_bottles_dustbin | 27 | ExtremeOver | 1700 | 正常失败 |
| robotwin-2390 | put_bottles_dustbin | 44 | ExtremeLow | 1700 | 正常失败 |
| robotwin-2392 | put_bottles_dustbin | 44 | Normal | 1700 | 正常失败 |
| robotwin-2394 | put_bottles_dustbin | 44 | ExtremeOver | 1700 | 正常失败 |
| robotwin-2402 | put_bottles_dustbin | 46 | Normal | 1700 | 正常失败 |
| robotwin-2403 | put_bottles_dustbin | 46 | Over | 1700 | 正常失败 |
| robotwin-2406 | put_bottles_dustbin | 47 | Low | 1700 | 正常失败 |
| robotwin-2409 | put_bottles_dustbin | 47 | ExtremeOver | 1700 | 正常失败 |
| robotwin-2420 | put_bottles_dustbin | 51 | ExtremeLow | 1700 | 正常失败 |
| robotwin-2421 | put_bottles_dustbin | 51 | Low | 1700 | 正常失败 |
| robotwin-2422 | put_bottles_dustbin | 51 | Normal | 1700 | 正常失败 |
| robotwin-2423 | put_bottles_dustbin | 51 | Over | 1700 | 正常失败 |
| robotwin-2424 | put_bottles_dustbin | 51 | ExtremeOver | 1700 | 正常失败 |

## 模型、环境与固定协议

LIBERO 使用官方 OpenPI `pi05_libero`，通过作者 StarVLA 路线转换加载；RoboTwin 使用作者指定的 `SidneyXie/pi05_robotwin` 及原正常化统计。没有微调 backbone 或训练 ISP。[部署记录](2026-10-10_rawvla_pi05_deployment.md)保存源权重、依赖与转换过程；本次重算权重哈希，与冻结协议一致。

| 文件 | SHA256 |
| --- | --- |
| libero | `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8` |
| robotwin | `9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3` |
| robotwin_norm | `229a4a2d6cfd1ba4eb2bc0b3aa57f049094032939130e4901485e9ea9910f5f7` |

| 项目 | LIBERO | RoboTwin |
| --- | --- | --- |
| 实际策略图像 | 作者 `default_isp` | SAPIEN Color `render_rgb` |
| 动作块 / 重新规划步数 | 10 / 5 | 50 / 50 |
| 归一化 | 原 quantile | 原 MEAN_STD，epsilon 1e-8 |
| 状态和动作 | 原本体状态、七维环境动作 | 左 6 关节/夹爪 + 右 6 关节/夹爪，十四维绝对关节目标 |
| 仿真渲染 | MuJoCo EGL | 原 RT 设置，32 spp、depth 8，OptiX 去噪 |
| 任务与成功规则 | 原作者 client/任务 | 原作者 eval loop/任务 |

每个初态使用独立配对 flow seed：SHA256(`owac-pi05-full-v1/<platform>/<task>/<initial_state>`) 前四字节小端；相同任务初态的五个照明域使用相同随机流。RoboTwin 的 Python 和 NumPy 语言随机数均重置到配对种子，第 i 个动作块 Torch seed 为 (base + 50i) mod 2³²。原 seed 身份取 manifest，不自行以 0–49 替代 dataset seed。

固定协议 SHA256：`3ac9626ef71eab4719be540393f51b66b767d7348dc4cac0ce5f5455f5b52e0e`。

- LIBERO manifest：`e3a054f43060912d668e57d35873b573a289c6d3c941fec9c029569c82971168`。
- RoboTwin manifest：`632b8360836595a868affffd24f9ecf1262d68a52a505ef5ffe801598d8317e5`。
- RawVLA source：`ba9c56b04da2c4931b70d8844710f5928d43775b`；本次核查工作目录干净，原作者源码未改。
- 当前设备：NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 595.84, 97887 MiB；RoboTwin Python Python 3.10.21，Torch 2.7.1+cu128，SAPIEN 3.0.0b1，mplib 0.2.1；StarVLA Transformers 4.57.0。

数据集来源为 [ToferFish/RawVLA-Bench](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)，固定 revision `8586c594fff9ce38580e29745f074e82e96e74f7`。已下载 713.69 GB、4,810 个文件、2,403 对 RAW/RGB 轨迹；它们是训练/缓存轨迹，主闭环评测分母来自上述测试清单。现成 RAW 为作者三通道 HDR-derived / pseudo-RAW，不能作为原生 Bayer CFA 验收。

## 计算成本

| 范围 | 实际策略步数 | 平均步数 | 步数 P50/P95 | episode 墙钟均值（秒） | 墙钟 P50/P95（秒） |
| --- | --- | --- | --- | --- | --- |
| LIBERO | 2330573 | 233.06 | 220/520 | 12.414 | 11.439/24.804 |
| RoboTwin 主清单 | 1966195 | 604.98 | 600/1700 | 83.860 | 80.968/176.452 |
| RoboTwin 原条件 | 377029 | 596.56 | 600/1700 | 105.088 | 99.867/257.827 |

以上 episode 墙钟包含初始化、专家检查、模拟器、RPC、存储及策略控制；多进程共享同一 GPU，不能把墙钟除以动作数解释为独占设备的网络速度。补跑并发策略与原顺序队列不同，因此此处的墙钟只能描述此次运行成本。

| 范围 | 审计 episode 数 | 50×14 预测块数 | 同步 model forward 均值（秒） | P50/P95（秒） |
| --- | --- | --- | --- | --- |
| robotwin_main | 3250 | 39574 | 0.1167 | 0.0961/0.2095 |
| robotwin_original_conditions | 632 | 7592 | 0.1742 | 0.1748/0.2345 |

模型时延来自服务的同步 CUDA model forward，排除 NPZ 写入、网络 RPC 和模拟器。RPC 按正式计分目录及 case ID 匹配，验证 flow seed 连续性、50×14 预测块、有限值及实际步数对应的块数；覆盖差异 0 个。[RPC 审计](results/2026-10-14_rawvla_pi05_complete_evaluation/rpc-summary.json)保存详细覆盖与统计。GPU peak 是服务生命周期的 Torch peak allocated，不能当作每条 episode 峰值或总设备占用。LIBERO 未保存同口径完整 model-forward 时延，因此这里只报告其 episode 成本。

## 与论文数字的关系

[RawVLA 论文 Table 1/2](https://arxiv.org/html/2609.37530v1)报告 Default π0.5 的 LIBERO Avg 51.52%、RoboTwin Overall 46.84%。本轮 LIBERO 58.20% 比该论文列高 6.68 个百分点，但没有新算法或训练，这个差异不能归因于算法改进。RoboTwin 主清单 15.63% 和原条件 17.09% 也没有复现论文 46.84%。

632 条原缓存条件与主清单使用相同物理身份集合的一个子集，但全部缓存 EV/noise seed 与对应主清单不同，两个成绩分别报告。论文 Table 2 比率与缓存的域分配相符仅为分母推断，公开逐条论文评测 ledger 尚未证实，因此不能把这 632 条断言为论文的全部正式 episode。

Blackwell 上使用 OptiX 替代无法可靠运行的 OIDN；RoboTwin 采用原检查点与作者 StarVLA 预处理，StarVLA/原 LeRobot 完整数值等价尚未建立；论文 RNG 序列未公开。本报告确认固定实现的完整评测，不排除上述实现差异影响成绩。后续若要对齐论文绝对数字，应另建协议版本，先建立原生后端数值对照，再重新评测各方法。

## 后续算法比较与复现入口

后续算法以本报告的两个主平台成绩作为同环境对照，固定检查点、原 manifest、初态、照明/噪声、语言与动作随机流、归一化、动作预算和成功判据。若算法改变视觉输入，应披露实际 RAW/RGB 来源、处理和训练预算；若改变推理后端或控制预算，则建立新版本并同时重跑基线。逐条参照文件以 platform/task/base_id/domain 定位身份，包含 entry、language 和 action 哈希及实际步数，便于配对比较。

云端正式结果位于 `/workspace/owac/rawvla/records/pi05-full-20261010-v3`；33 条恢复证据位于 `/workspace/owac/rawvla/records/pi05-main-repair-20261014-v1`；完整导出位于 `/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1`。本地只保留数字、配置、协议和哈希，图像、权重及大动作数组留在云端。

```bash
# 仓库中的实现及聚合检查
make check
uv run --locked pytest -q tests/test_rawvla_complete_report.py tests/test_rawvla_main_repair.py tests/test_rawvla_main_repair_audit.py tests/test_rawvla_repair.py tests/test_rawvla_repair_audit.py tests/test_rawvla_contract.py tests/test_rawvla_full_eval.py

# 云端重新生成身份审计快照，不重复评测
PYTHONPATH=/workspace/owac/rawvla/records/pi05-full-20261010-v3/frozen/src /workspace/owac/rawvla/envs/hub/bin/python -m owac.runtime.rawvla_eval_audit --output /workspace/owac/rawvla/records/pi05-full-20261010-v3
```

本次 `make check` 通过；51 项协议、异常保护、身份接纳、动作完整性与聚合测试通过。[本地验证记录](results/2026-10-14_rawvla_pi05_complete_evaluation/local-validation.json)保存模块哈希；[产物校验和](results/2026-10-14_rawvla_pi05_complete_evaluation/checksums.json)覆盖本次数字证据。历史发布快照、异常和旧标签保留，不以本报告覆盖当时未完成的历史结论。

# RoboTwin 原条件核查五条基础设施异常修复

目的：补齐 RawVLA-Bench π0.5 原条件核查中五个没有产生策略动作的身份，保留原 dataset seed、照明/噪声、检查点、动作协议与分母。此次只补跑这五条，不重复执行已有 627 条，不启动主 3,250 清单的异常补跑。

做了什么：定位作者专家规划阶段的空目标动作和空列表索引，建立独立 `pi05-paper-repair-20261013-v1` 修复版本。原 source、v3 冻结代码、协议、发布标签及失败尝试保留。补跑仍使用原作者任务和 π0.5 服务 `localhost:18015`。

效果：本地 33 项边界与结果审计测试通过。已完成一个已有效执行的控制身份 `paper-0037`：修复分支未触发，专家成功，原方法在策略初始化前恢复；首个实际输入的十四维状态、三路 240×320 RGB 图像、flow seed 与首个 50×14 动作块均与历史运行逐值完全相同。五条均按原 seed/照明完成真实策略执行并通过独立审计，均为正常策略失败；已追加回正式原条件结果。632/632 条完整成绩为 108 成功、524 正常失败，成功率 17.09%。

结论：修复保持“专家规划可以失败，固定物理 seed 的策略仍须接受评测”的既有协议，避免规划失败被异常中断。专家失败不是策略成功，也不注入替代专家动作。五条的原身份、metadata hash、连续动作、lighting/noise seed、方法恢复和原 627 条未改写均通过审计；原条件核查现在完整。主清单 3,250 的其他异常仍单独保留。

## 失败身份与原因

| 身份 | 任务 / 原 seed | 原照明域 | 原异常 |
| --- | --- | --- | --- |
| paper-0038 | grab_roller / 39 | Over | 空抓取 target pose 被传入 Action 构造器 |
| paper-0222 | open_laptop / 25 | ExtremeLow | 同上 |
| paper-0235 | open_laptop / 42 | Low | 同上 |
| paper-0454 | put_bottles_dustbin / 27 | Normal | 直接索引空 right_action 列表 |
| paper-0470 | put_bottles_dustbin / 51 | Normal | 同上 |

`Base_Task.choose_grasp_pose()` 在没有可用规划时可返回 `(None, None)`，`grasp_actor()` 却仍构造目标为 None 的 move。瓶子任务还直接访问 `right_action[1][0]` / `left_action[1][0]`，没有处理已失败规划的空动作列表。全部发生在作者 `play_once()` 专家检查，π0.5 的实际控制尚未开始。

冻结 v3 已设置 `ROBOTWIN_FORCE_EVAL_SEED=1`，专家成功判据不用于删除原清单中的 seed；专家过程只提供作者任务语言元信息。原错误导致这些身份无法到达既有的固定 seed 策略入口。

## 修复边界

[rawvla_repair.py](../../src/owac/runtime/rawvla_repair.py)对作者方法 AST 添加三个局部保护：空抓取目标将 `plan_success` 标记为 false 并返回空动作列表；瓶子任务遇空左/右抓取动作时结束专家循环，仍执行原作者的任务元信息赋值。没有伪造目标位姿、提高专家成功标志或修改任务成功判据。

保护只在专家 `play_once()` 调用范围生效；无论返回或异常都恢复原 `Base_Task.grasp_actor` 和任务 `play_once` 方法，随后原 eval 再按同一个 seed 重新初始化策略场景。未知异常继续抛出。每条记录原方法/保护 AST 哈希、专家规划失败事件、原任务元信息、初始化物体状态与方法恢复证明。

物理场景、资产、相机、OptiX、原条件曝光及传感器噪声、语言和 flow RNG、原 pi05 权重/统计、十四维状态及动作、50 步动作块、最大步数和成功判据沿用已冻结协议。现成模型没有训练；修复没有向策略提供专家动作、物体真值或未来状态。此处初始化物体状态仅用于独立审计。

独立 [rawvla_repair_audit.py](../../src/owac/runtime/rawvla_repair_audit.py)核查 627 条继承记录完全不变、五条原始 entry/metadata hash、实际动作与状态维度/有限值/连续步数、照明/noise seed、原方法恢复及成功/正常失败，拒绝重复身份或缺分母。新的 632 条完整汇总仍与主清单 3,250 分开计分，不称为论文内部未公开逐条评测结果的确认复现。

## 运行与证据

父记录：`/workspace/owac/rawvla/records/pi05-full-20261010-v3`，原协议 SHA256 `3ac9626ef71eab4719be540393f51b66b767d7348dc4cac0ce5f5455f5b52e0e`。

控制记录：`/workspace/owac/rawvla/records/pi05-paper-repair-control-20261013-v1`。原首输入 `input_00545.npz`，控制首输入 `input_41808.npz`，均位于原加载策略服务的 `pi05-full-20261010-v2/robotwin-policy`。控制案例重复执行仅用于诊断，不并入正式分母或选择结果。

修复记录：`/workspace/owac/rawvla/records/pi05-paper-repair-20261013-v1`。`repair-protocol.json` 保留父协议、632 条条件清单、627 条原 ledger、局部修复和原 eval 的哈希；`source-guards.json` 固定四个原作者 source 文件哈希。`robotwin-paper/paper-*/attempt*/` 分别保留动作、lighting、expert-repair、episode result 和日志。模型实际输入与预测动作仍由原策略服务保存，大数组不下载。

后台任务：`owac-rawvla-paper-repair-v1`；`autostart=false` / `autorestart=false`，有限执行，退出 SSH/聊天不会中断。原主评测继续运行，没有修改原 source、冻结策略服务或管理服务。

本地核验：`make check`（140 个 Python 文件）、`uv run --locked pytest tests/test_rawvla_repair.py tests/test_rawvla_repair_audit.py tests/test_rawvla_contract.py tests/test_rawvla_full_eval.py`（33 passed，3.28 s）。修复用的云端 Python 3.10 与父环境相同；本地边界测试使用 Python 3.12。没有在本次任务中提交或移动发布标签。

## 完整补跑结果与接纳

2026-10-13 16:42:30（北京时间）正式结果已接纳。原 627 条 ledger 逐字节保存在修复目录的 `inherited-parent/`；仅追加五条，并为每条附加修复协议及独立验证哈希。原 source、冻结协议、权重、统计、策略服务及标签未改动；历史失败尝试保留。

| 身份 | 实际策略步数 | 最终结果 |
| --- | --- | --- |
| paper-0038 | 400 | 正常策略失败 |
| paper-0222 | 700 | 正常策略失败 |
| paper-0235 | 700 | 正常策略失败 |
| paper-0454 | 1700 | 正常策略失败 |
| paper-0470 | 1700 | 正常策略失败 |

**原条件核查：632/632 完成，108 成功、524 正常失败，成功率 17.09%。** 五域完整分母为 ExtremeLow 127、Low 127、Normal 127、Over 126、ExtremeOver 125；完整成功率分别为 3.15%、22.05%、30.71%、21.43%、8.00%。这套结果独立于主清单 3,250 条，不混入主分母，不据此宣称复现论文数值。

[独立审计](results/2026-10-13_rawvla_robotwin_infrastructure_repair/repair-validation.json)、[接纳与旧新 ledger 哈希](results/2026-10-13_rawvla_robotwin_infrastructure_repair/adoption.json)、[控制输入一致性](results/2026-10-13_rawvla_robotwin_infrastructure_repair/control-input-parity.json)、[本地检查](results/2026-10-13_rawvla_robotwin_infrastructure_repair/local-validation.json)、[完整数字汇总](results/2026-10-13_rawvla_robotwin_infrastructure_repair/canonical-summary.json)已留档。新修复和控制 supervisor 均正常结束。

接纳后再次核验时，调用 `rawvla_repair_audit.validate(output / "inherited-parent", output)`，使用逐字节保留的原 627 条快照；原 canonical ledger 已合法追加五条，不能再当成旧快照。

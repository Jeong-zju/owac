# 最后学生权重闭环对比（2026-10-26）

用户明确授权仿真闭环成功率与视频比较。本轮没有训练或实机动作。

## 结果

| 策略 | 成功/总数 | 95% Wilson区间 | 平均完成步数 | 范围 |
| --- | --- | --- | --- | --- |
| RAW教师 | 20/20 | 83.89%–100% | 94.40 | 86–123 |
| 最后学生84768 | 20/20 | 83.89%–100% | 93.65 | 85–103 |

教师成功→学生失败0/20，反向恢复0/20。未排除任何场次、未重试。进程native exit=0，墙钟825.19秒。该任务这组初态中，学生已实际完成闭环控制；20场不能证明统计非劣或全任务泛化。

## 范围与执行

LIBERO goal/task6，put_the_cream_cheese_in_the_bowl，Normal。复用2026-10-20冻结目录的20个历史test初态test-101100至test-101119，并非20个新抽取初态。全量学生24遍/84768更新的最后权重，不按本轮结果挑选checkpoint。教师前端为项目已有init0选定资产，不是作者公开训练后ISP权重。

同初态、语言、初始机器人状态、成像seed和每次重规划采样noise；真实6帧因果RAW历史，独立递归状态。首决策RAW完全一致；动作分叉后各自环境观测不同，不作后续同观测逐步数值归因。10步稳定等待，action horizon10、flow10、replan5，最多300策略步；直接使用环境check_success。教师视觉FP32与当前目标制备一致，学生FP32权重/BF16前向，共享冻结原动作后端。采用当前成对算术seed规则，与历史教师hash噪声规则不同，两臂均重新执行。原生quantile动作反归一化，不添加夹爪符号转换。

学生独立token路径仅含学生模型及状态，没有教师token回退；评估进程为公平比较同时加载教师及共享动作后端。没有优化更新。

## 独立验收与证据

Sol复核实际代码与原始NPZ/视频：20对初态、首RAW历史、首状态和采样noise一致；执行动作等于各预测chunk前5步；40条成功/步数账本与NPZ一致；40个视频实际帧数等于策略步数+11。全部检查通过。查看全部40个末帧及代表性并排过程帧。原始40段和全部20段并排视频均保留，不只选最相似片段。test-101108教师86步、学生102步，体现执行速度差异。

实现由DeepSeek Harness Web session-8f2a5da4-8e59-4413-a719-3ff4c453e57c（deepseek-official/deepseek-flash/max）round14完成，Sol独立修复/验收。58项针对性测试通过、无跳过；make check通过1071文件。

运行根：`outputs/raw-distillation-all-bench-20261022/closed-loop-final84768/`。
- `launch.py`、`config.json`、`exit.json`：完整可复现命令、配置、退出收据；base5039e5b加本次源码，执行源码快照在evidence/source。
- `rollouts/attempts.jsonl`、`summary.json`、每场NPZ/MP4：逐场原始证据。
- `evidence/independent-audit.json`、`all-final-frames.jpg`：独立检查与全部末帧。
- `videos.html`、`comparison/`：全部20对视频，左教师右学生、10fps，短片停留末帧。

当前证据只覆盖正常光单任务、历史固定初态。尚未覆盖其他任务、RoboTwin、弱光或真实机器人，不能由本结果外推。离线动作MAE差异与当前全部成功可以同时成立。

## 2026-10-27 用户阶段验收

用户确认STEM（消息称STEP）、ADAPT、CONTROL可标记通过。通过范围为当前正常光单任务基本复刻教师控制功能；ADAPT依实际全视觉头token蒸馏完成，不代表执行了历史计划中的动作监督。STEM作为输入侧组件随最终系统验收，历史stem-only负结果保留。弱光阶段待开展。

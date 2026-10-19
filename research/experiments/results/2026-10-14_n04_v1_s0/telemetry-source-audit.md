# N04 S0 遥测来源核查：LIBERO / RoboTwin2 训练缓存

- 执行者 `ws-telemetry`（S0 工作流 A1）；方式**远端只读** `ssh -p 44327 root@107.209.104.125`，未写 `/workspace/owac/rawvla/**`（脚本放远端 `/tmp`）。
- 机读结果：[`telemetry-source-audit.json`](telemetry-source-audit.json)（含全部 metadata、shape、dtype、文件行号证据）。
- 数据根 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`；冻结源码 `/workspace/owac/rawvla/source`（HEAD `ba9c56b04da2c4931b70d8844710f5928d43775b`）。
- 方法：全部 NPZ 用 zip 成员名扫描键并集（不读数组体）＋ 每平台等距 40 文件 `.npy` header 抽样 ＋ 每平台 3 文件深读并解析 `metadata_json` ＋ 全量 metadata 分布统计。

## 1 规模与键并集

| 平台 | NPZ | 键（出现文件数） |
| --- | --- | --- |
| LIBERO | 1771 | `agentview_raw_uint8` 1771、`wrist_raw_uint8` 1771、`actions` 1771、`states` 1771、`robot_states` 1771、`rewards` 1771、`dones` 1771、`metadata_json` 1771 |
| RoboTwin2 | 632 | `actions` 632、`robot_states` 632、`left_endpose` 632、`right_endpose` 632、`dones` 632、`rewards` 632、`head/left/right_camera_raw_float32` 各 632、`metadata_json` 632 |

**两平台键并集里都没有任何时间戳或帧率键**；帧号不能当秒。

## 2 深读样本（shape / dtype）

| 样本 | 数组 |
| --- | --- |
| LIBERO `libero_10.../episode_000001/variant_00.npz`（112.8 MB） | `agentview_raw_uint8 (293,256,256,3) uint8`、`wrist_raw_uint8` 同形、`actions (293,7) float32`、`states (293,123) float64`、`robot_states (293,8) float32`、`rewards/dones (293,) uint8` |
| LIBERO `libero_object.../episode_000035/variant_00.npz`、`libero_spatial.../episode_000499` | 同键同 dtype；`states` 维度随场景变化（40 文件抽样共 47/51/71/79/84/92/110/123） |
| RoboTwin2 `grab_roller/episode_000000.npz`、`place_dual_shoes/episode_000017.npz`、`stack_bowls_two/episode_000049.npz`（907.3 MB） | `actions (T,14) float32`、`robot_states (T,16)`、`left/right_endpose (T,7)`、三路 `*_raw_float32 (T,240,320,3) float32`、`dones/rewards (T,) uint8`；40 文件抽样形状完全一致 |

帧数：LIBERO 63—478（合计 261,131），RoboTwin2 88—783（合计 146,944）；**两平台帧内数组长度不一致的文件数均为 0**。

## 3 语义证据（远端源码，只读；完整引用见 JSON `evidence`）

| 事实 | 位置 |
| --- | --- |
| LIBERO：先取 sim state 并渲染 frame t，之后才 `actions.append(action)`、才 `rgb_env.step(action)` | `scripts/replay/replay_libero_train_rawvla_light_npz.py:273,284,288,325,333,350` |
| LIBERO `robot_states` = `[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]`，不是关节角 | 同文件 `:81`（`_state_from_obs`） |
| LIBERO 全部 1771 文件 `replay_mode="actions"` | 全量 `metadata_json` 统计 |
| RoboTwin：`copy_physics_state(clean→raw)` 后渲染，帧与 clean 观测同物理状态；`actions` 取 `joint_action["vector"]` | `scripts/replay/replay_robotwin2_paired_lighting_training.py:131,201,203,213,245` |
| RoboTwin `joint_action["vector"]` 由 `get_left/right_arm_jointState()` 组成 → **实测关节状态**；`_take_picture` 在 `scene.step()` 之后调用 | `third_party/RoboTwin/envs/_base_task.py:440,490,497,511,848,885,889` |
| RoboTwin `K`/`fovy` 只在运行时可得（`camera.get_intrinsic_matrix()`），不写入缓存 | `third_party/RoboTwin/envs/camera/camera.py:139,322` |

## 4 结论：能否从现有缓存补齐（可否 + 依据 + 缺什么）

| 项 | LIBERO | RoboTwin2 |
| --- | --- | --- |
| **实际执行动作** | `partial`：存在逐帧 7 维 OSC 命令序列，但证据只有 `producer_code_order`（代码顺序），`verified=false`；控制器限幅后真正应用的数值未记录 | `unavailable`：`actions` 是实测 14 维关节状态；metadata 引用的 `clean_replay_record`/`source_trajectory` 文件未随数据交付（远端 `find` 无结果） |
| **时间戳** | `unavailable`：键并集与 metadata 均无时间/帧率字段 | `unavailable`：同上 |
| **标定 K/T/畸变/时间偏移** | `unavailable`：缓存无标定字段；相机由 MuJoCo 模型运行时定义 | `unavailable`：运行时可得但未写入交付缓存 |

依据：以上结论的字段证据是**全量键扫描 + 40/平台 shape 抽样 + 3/平台深读 + 生产者源码行号**，不是抽样猜测；`timestamp`/`calibration` 两条来自全量键并集，属排他性证据。

## 5 未验证项

1. 未做真实 episode 级同版本重放核对 ⇒ LIBERO「frame t ↔ `actions[t]`」仍是代码级推断。
2. 未对全部 2403 个文件的数组体深读（header 全量 + 40/平台 shape + 3/平台深读）。
3. RoboTwin gripper 归一化区间未逐值验证。
4. 本地无真实 NPZ，真实缓存未挂 sidecar；`context_at`/`executed_between` 仅用合成 sidecar 验证。

## 6 补齐方案（未执行，待 Lead 决定）

1. **LIBERO 重放核对**：远端新建独立 run 目录，用同一冻结环境按相同协议重放若干 episode，逐帧比较相机帧、q/末端、时序并保存误差/哈希/容忍度；通过才把控制证据等级升为 `replay_verified` 并据此生成 sidecar，不通过则新建独立命名的扩展数据集。
2. **RoboTwin 扩展数据**：重跑 `play_once` 并在执行器侧记录真正下发的关节目标与时间，产出独立命名的新数据集；标定从同版本运行时 `get_intrinsic_matrix()`/`fovy` 取得并连同场景版本与哈希写入 provenance。
3. **不要**从实测关节状态反推控制，也**不要**给原 NPZ 挂未核实的控制 sidecar。

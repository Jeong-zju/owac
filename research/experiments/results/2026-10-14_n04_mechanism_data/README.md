# 2026-10-14 · N04 S0 · 工作流 C：机制数据 D0（解析三维渲染诊断）+ D1（采集器与可行性核查，partial）

## 身份与可复现版本

- 实验 ID / 负责人 / 日期时区：`2026-10-14_n04_mechanism_data` / ws-mechanism / 2026-10-14 Asia/Shanghai
- 候选卡 / 证据层：N04（自运动预测与外部变化分离）/ S0 数据与因果合同，机制层前的数据合同
- 状态：**完成（代码与本地生成、审计通过）**；D1 为 partial，本轮不启动大规模采集
- Git SHA / 分支：`fd6255b769d661af568738bec194bb599601a644` / `lzh/n04-egomotion-separation`
- 工作区 dirty：是（本 worktree 同时存在其他 teammate 与用户的未提交修改，均未触碰）
- 本次新增/修改（仅这些）：
  - `src/owac/raw/n04_mechanism_data.py`（新增，D0 渲染器/因子/审计 + D1 采集器与可行性）
  - `tests/encoders/n04_full/test_mechanism_data.py`（新增，20 项）
  - `outputs/n04-v1/s0/mechanism-data/`（生成产物：16 个 factor npz + manifest + audit + factors.json + d1/，71 MB，不进 Git）
  - 本目录（小清单、远端 probe 原始输出、命令日志）
- 配置：`configs/n04/full/s0_data.json`，sha256 `ff1a7e6004169448cd6e3acddac5bded9714f7d146f95b40c1645656ef5a1e33`
- 随机种子：`split_seed=20261014`（factor/batch/审计派生种子均由它决定，见 manifest.config）
- 环境：本地 `uv run --locked --extra encoder`，Python 3.12，torch 2.14.1+cu130，numpy 2.5.1，CPU

## 运行前设计

- 研究问题：S0 是否具备“逐样本标签可信”的解析三维诊断数据，以及 D1 真实仿真侧能否补齐
  实际执行动作 / K / pose / depth / visibility / 接触状态。
- 可证伪判据：若逐样本标签与独立投影重算不一致，或批次置换/单样本扰动发生串位，或
  `[B,1,2]+[B,2]→[B,B,2]` 类历史缺陷无法被检出，则 D0 不通过。
- 方法：可解释针孔 `K` + 多深度面 + 独立刚体 + 逐像素解析平面 z-buffer；图像、深度、实例
  索引、可见性、事件标签全部由**同一逐样本世界状态**计算；审计用**独立的点投影路径**重算。
- 输入信息与 GT 隔离：`online_*`（测量/上报位姿/动作增量/关联）与 `gt_*`（辐射、深度、实例、
  位姿、可见像素、事件）分列；审计检查两者不相交，GT 不得进入在线特征。
- 学习量：无（纯解析，无训练参数）。

## 命令与产物

```bash
# 冻结 CLI（先 --help 再执行）
uv run --locked --extra encoder python -m owac.raw.n04_mechanism_data --help
uv run --locked --extra encoder python -m owac.raw.n04_mechanism_data \
  --config configs/n04/full/s0_data.json \
  --output outputs/n04-v1/s0/mechanism-data \
  --remote-evidence research/experiments/results/2026-10-14_n04_mechanism_data/remote-evidence.json

# 测试
uv run --locked --extra encoder pytest -q tests/encoders/n04_full/test_mechanism_data.py   # 20 passed
uv run --locked --extra encoder pytest -q tests/encoders/n04                                  # 198 passed
ruff format/check src/owac/raw/n04_mechanism_data.py tests/encoders/n04_full/test_mechanism_data.py

# 远端只读核查（ssh -p 44327 root@107.209.104.125），原始输出见 remote-probe-*.json
#   1) 直接用本模块 d1_npz_field_probe 读两个平台 NPZ 头（不整包解压）
#   2) LIBERO 仿真器：K/depth/segmentation/contact/state replay（starvla env + MUJOCO_GL=egl）
#   3) RoboTwin：SAPIEN/camera API、资产、task_config、源码调用点扫描（robotwin env）
#   4) 版本/清单/日志/源轨迹清点
.venv/bin/python research/experiments/results/2026-10-14_n04_mechanism_data/assemble_remote_evidence.py
```

产物：

- `outputs/n04-v1/s0/mechanism-data/d0/<factor>.npz`：16 个因子序列（12 单因素 + 4 组合），
  每帧含 `online_measurement_rgb3`(f32 线性 3 通道)、`online_measurement_cfa`(u16 DN)、
  `online_clip_mask`、上报位姿/动作增量/关联，以及 `gt_*` 监督量。
- `d0/manifest.json`：单位/坐标系/CFA 声明/因素表/审计结果/逐文件 sha256。
- `d0/audit.json`：逐样本审计全部数字。
- `factors.json`：单因素在前、组合在后，`coverage_complete=true`（配置 12 项全覆盖）。
- `d1/collect_mechanism_data.py` + `d1/feasibility.json`：可运行采集器与逐项可行性。
- `run.json`：本次运行身份与 hash。

## 结果（实测）

### D0 规模与覆盖

- 16 个因子 × 24 帧 × B=2 × 2 相机 × 96×128；`visible_pixels_total=1,179,648`/因子。
- 关键事件覆盖：`stop_reverse` 停止 16 帧样本、反向 2 帧样本；`occlusion_recovery` 遮挡事件 4；
  `object_only` 出生 1 / 出界 1；`robot_self_occlusion` 可见机器人体结构并产生 6 次遮挡事件。
- `unidentifiable_full_view_plane`：单平面占满 100% 像素（`gt_body_index` 只有一个 id）。
- `wrong_association`：1186 个像素（0.10%）在线关联与 GT 不同，且全部位于声明左半区、
  仅 1↔3 交换，右半区完全干净。
- `time_shift`：`online_camera_translation[t] == gt_camera_translation[t-3]` 实测成立。
- `gain`：增益 1.021→1.634，前后半均值 0.4544→0.6382，并出现饱和像素。

### D0 逐样本审计（`d0/audit.json`，adaptive 参考网格）

| 检查 | B=1 | B=2 | B=4 |
| --- | --- | --- | --- |
| 标签中“独立投影从未到达”的像素（unexplained） | 0 | 0 | 0 |
| 独立投影获胜像素中标签遗漏（missed） | 0 | 0 | 0 |
| 周长尺度对称差超限（sym_diff_excess） | 0.0 | 0.0 | 0.0 |
| 独立投影 vs 渲染深度最大误差 | 2.4e-8 | 2.4e-8 | 2.4e-8 |
| 跨样本污染 `claim[i]<-ref[j]`（i≠j） | — | 0.0 | 0.0 |
| 图像↔标签重算（辐射/深度）最大误差 | 2.9e-8 / 5.2e-8 | 同 | 同 |
| 重算像素数（两相机合计） | 6143 | 12281 | 24567 |
| 原始 IoU 最小（小目标受 0.5px 点采样边界支配） | 0.786 | 0.639 | 0.638 |

- 批次置换 `[2,0,3,1]`：图像/深度/实例索引**完全逐样本置换**（精确相等）。
- 单样本扰动：其余样本逐位不变，被扰动样本变化。
- 静态零残差：两帧一致性最大绝对差 **0.0**。
- CFA：四种相位/通道与手工表一致，12-bit DN 最大误差 **0**。
- 不可辨识性：相机 +0.02 m 与平面 −0.02 m 两世界状态渲染**完全一致**（最大差 0.0），
  两者都占满视野 → 测量本身无法区分。
- **历史缺陷反例**：复现 `[B,1,2]+[B,2]→[B,B,2]` 合并标签后，审计给出
  unexplained=492/480/240/168、图像-标签不一致 3740 像素、跨样本污染 1.0 → 必被检出。

### D1 可行性（`d1/feasibility.json`，verified=7 available / 4 partial）

| 项 | 结论 | 证据 |
| --- | --- | --- |
| LIBERO 仿真资产 | available（executed） | LIBERO `5a2d7a2…`，assets/bddl 就位；实测创建 env、K=`fx=fy=309.02,cx=cy=128`@256、depth、segmentation 52 个 id、37 个接触（geom1/geom2/dist/pos） |
| RoboTwin 仿真资产 | available（API+资产级） | RoboTwin `8e9e665…`，SAPIEN 3.0.0b1，objects/embodiments/task_config 就位；**未创建 live env**（等级标注） |
| 控制器日志 | available | LIBERO `task_0.episodes.actions.jsonl` 48,898 行逐 step 动作；RoboTwin `_policy_data/episode0.pkl` 含 `predicted_actions(81,14)`、`action_source=model_prediction` |
| 同版本回放 | available | 两仓库 head 固定；LIBERO 按 NPZ `states[0]` 回放 **max|Δ|=0.0**；LIBERO-datasets 目录为空、RoboTwin 原始 clean 轨迹 pkl/hdf5 **不在这台服务器上** |
| 实际执行动作 | available（字段级） | LIBERO `actions` 是 `env.step` 的输入；RoboTwin `actions` 是 14 维关节**状态**（生产代码 `joint_action/vector`），不是下发命令 |
| K / pose / depth / visibility / contact（RoboTwin 侧） | **partial** | 运行时 API 与部署源码调用点均在（`Camera.get_depth/get_segmentation/get_config→get_intrinsic_matrix`、`scene.get_contacts()`、`robot_states/endpose`），但本轮未跑 live task env |
| CFA 表示 | available | 两平台 NPZ 均为三通道（无 CFA 键、无 depth/K 键）；D0 提供声明的 `synthetic_cfa`，未把 RGB 重马赛克冒充 RAW |

## 失败与未验证项

- RoboTwin 未创建 live task env：K/depth/segmentation/contact 的**运行时提取**尚未端到端跑通
  （Vulkan/渲染路径未验证），因此四项判为 partial，缺 `robotwin_*_executed`。
- RoboTwin 原始 clean 源轨迹（`_traj_data/episode*.pkl`、`episode*.hdf5`）在本机缺失，
  只能从现有 NPZ 重放或按新命名扩展集重采；不得把 `actions` 当执行命令使用。
- 真实 CFA/传感器：两平台都是作者三通道 pseudo-RAW；真实 CFA 需另建独立命名数据集。
- 未做：大规模 D1 采集（按任务要求 partial）、任何网络训练、任何 π0.5 闭环。
- 本地只跑 CPU 渲染与审计；未测 GPU 时长/显存（本轮不需要）。

## 追加任务 A（Lead 裁决后）：仓库级 ruff 修复

`assemble_remote_evidence.py` 原有 2 处 E501（docstring 107 字符、`write_text` 行 104 字符）并需格式化，
已只改该文件（docstring 路径缩写为 `<this directory>/`，其余交给 `ruff format`）。实测：

- `uv run --locked ruff check .` → `All checks passed!`
- `uv run --locked ruff format --check .` → `230 files already formatted`

## 追加任务 B（Lead 裁决后）：LIBERO 动作相位验证（`action-phase-verification.json`）

同版本环境（LIBERO `5a2d7a27…`、robosuite 1.4.0、mujoco 3.2.3、`MUJOCO_GL=egl`），3 个真实 episode
（goal/object/spatial，共 180 步，全部只读、无既有服务改动）：

- **相位对齐被核实**：`step(states[t], actions[t])` 与 `states[t+1]` 的中位误差
  **7.06e-4 / 1.63e-4 / 3.24e-4**，而 k=0（零滞后）与 k=+2 的中位误差为 0.05–0.14 →
  正确相位比错误相位好 **70–300 倍**，逐帧胜率 94.8%–100%。
  即 `actions[t]` 是作用在 t→t+1 区间、观测 frame t 之后下发的控制。
- **但不是位精确重放**：单步 max|Δ| = **0.098 / 0.176 / 0.487**（阈值 1e-9），无任何一帧 ≤1e-4；
  尾部误差集中在接触/约束阶段。按 Lead 规则 **`replay_verified` 不成立，保持 provisional**，
  `executed_action` 等级维持 `producer_code_order`。
- **残差原因已定位**：`set_init_state` 的状态回放是精确的（round-trip **0.0**）；同一 env 重复同一
  (state, action) 也精确（**1e-16**）；但**新建 env vs 已用 env** 在同一 state+action 上相差
  0.088–0.213，清零 MuJoCo `qacc_warmstart` 会改变结果 0–0.012 → 记录的 `states` 不足以确定求解器
  热启动与控制器/观测缓冲，单步位精确重放从 NPZ 本身不可达。
- 诚实更正：第一轮探针未调用 `env.reset()`，出现 0.05–0.13 的“重复不一致”，加入 producer 的
  reset 顺序后同 env 重复为 1e-16；该现象是探针伪影，已在结论中标注。

## 下一阶段输入

- S1/S3 可直接使用：`d0/*.npz` 的 `gt_depth_m`/`gt_body_index`/`gt_silhouette` 作为监督，
  `online_*` 作为输入；单因素在前、组合在后，便于消融排序。
- 若 S1 需要真实几何监督：先补 RoboTwin live env 一次小规模探针（同目录 `d1/collect_mechanism_data.py`
  `--sim-probe`，已验证“未确认对应即拒绝 sidecar”的判定逻辑）。
- 待 Lead 决策：是否需要本轮把 RoboTwin live env 探针补成 executed 级；以及 D0 是否需要
  放大到更多帧/更多相机（当前 71 MB，扩到 48 帧约翻倍）。

## 追加任务 C（Lead 裁决后）：把「位精确可复算」写进采集器 schema 并实测

实现落在 `src/owac/raw/n04_mechanism_data.py`（schema `n04-d1-executed-trace-v1`）：

- **每步字段**：`step_index`、`pre_state`、`issued_command`、`post_state`；`issued_command` 是采集器
  在调用点真正交给 `env.step` 的命令（`issued_command_source=collector_step_call_argument`），不是
  `actions` 改名。
- **热启动三选一并写清语义**：`disabled_by_flag`（设置 `mjDSBL_WARMSTART=256`）、
  `zeroed_each_step`（每步清零 `qacc_warmstart`）、`saved_before_step`（保存 `qacc_warmstart_before`
  与 `_after` 数组）。
- **控制器/观测缓冲显式声明**：`recorded_initial_conditions`（轨迹由 fresh+reset 环境与有序 recipe
  决定）或 `declared_not_recomputable`（外部录像未知缓冲，只能相位级）；不得沉默。
- **规则**：具备 ①每步 pre/post/command ②受控热启动 ③显式缓冲策略 ④有序 recipe ⑤通过的
  验证回放，`bit_exact_recomputable=true`；缺任一 → `phase_and_state_recorded` 或 `insufficient`，
  只能给相位级证据。`source_fidelity` 单独记录「对原始录像的保真度」。

**远端有界 dry-run（3 步 × 3 种热启动模式，写在 `manifests/collector-dryrun/<mode>/`）**：

| 模式 | 落盘文件 | npz 键 | 验证回放 max\|Δ\| | 判定 |
| --- | --- | --- | --- | --- |
| `saved_before_step` | npz 5890 B + trace.json 4381 B | + `qacc_warmstart_before/after` (3×37) | **0.0** | bit_exact_recomputable |
| `disabled_by_flag` | npz 3925 B + trace.json 4249 B | 4 个基础键 | 1.89e-14 | bit_exact_recomputable |
| `zeroed_each_step` | npz 3925 B + trace.json 4182 B | 4 个基础键 | 4.68e-16 | bit_exact_recomputable |

- 验证回放 = 用第二个 fresh env 重放同一条 recipe，并重新施加记录的热启动策略后与 `post_state` 比较；
  容差 1e-12。三种模式都通过，即「执行动作 ↔ 状态转移」在记录完整时**位精确可复算**。
- `saved_before_step` 与 `zeroed_each_step` 的 npz sha256 不同（前者多两数组）；`disabled_by_flag`
  与 `zeroed_each_step` 的 npz 相同（`8c387ea2…`），说明两种冷启动语义在该轨迹上等价。
- 本地留档 `trace-dryrun-<mode>.npz` / `.trace.json`，与远端 sha256 一致。
- 该 dry-run 同时把 `source_fidelity.level=phase_only` 写入 trace（外部录像的缓冲不可复算），
  与任务 B 的结论一致，不混淆「自采轨迹可复算」与「原录像保真」。

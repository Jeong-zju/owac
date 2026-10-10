# PolaRiS 专家动作回放与原生 RAW 采集试验

目的：实际执行用户提出的“根据机器人状态与动作回放专家数据”，检查能否在 Isaac Sim 中取得成功抓放演示及配对 RGB/RAW。用户本轮接受以成功专家数据作为试验目标。公开原训练场景映射尚未恢复，因此本次使用原动作重建可检验的物体场景，不能当成原训练图像的精确补采。

结论：已完成一条原始专家动作的成功物理回放，取得 108 帧双视角配对 RGB/仿真器原生 RAW。成功包括实际夹起、搬运、松手和稳定入箱，初始化后物体状态覆盖次数为零。原训练场景和相机映射未恢复，结果属于重建场景的新配对专家数据。

状态：本试验完成，完整 Arena DROID π0.5 九配置 / 900 条基准已恢复运行；回放试验不计入其分数，不训练或调整 backbone。定时任务关闭。

## 版本与来源

- 本地分支 / HEAD：`lzh/pi05-sim-baseline` / `7f7e91f3ddc9dcc10763253ed8d9119cb7735b08`；新增试验实现运行时未提交。每次云端尝试保存独立源码快照及逐文件 SHA256，原尝试不覆盖。
- 完整已跟踪 dirty diff 保存于本地被忽略的 `outputs/polaris-expert-replay/worktree-before.diff`；校验值、相关新增源码哈希及运行源码定位见数字归档的 `source_provenance.json`。用户其它研究文档变更保留，不纳入本任务提交。
- 公开专家数据：[owhan/PolaRiS-datasets](https://huggingface.co/datasets/owhan/PolaRiS-datasets/tree/dca86b8e186df63323eb375344a1977d92f508f7)，修订 `dca86b8e186df63323eb375344a1977d92f508f7`，RLDS 1.0.0。只下载首分片到云端，实读十条演示；未下载其余 31 分片。
- 机器人：[owhan/DROID-sim-environments](https://huggingface.co/datasets/owhan/DROID-sim-environments/tree/5b815672a9a2e3ae596ae845188ae4be5c5d2f34)，修订 `5b815672a9a2e3ae596ae845188ae4be5c5d2f34`。DROID Franka + Robotiq USD SHA256：`c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c`。
- 原记录：首分片索引 9，`pi_scene_v5/episode_14.npz`，指令 `put the cube in the left bin`，108 帧。源记录 SHA256 `01202d964bbc343bf03ceaea2188b3906c2db5e9a944d14f7d59b90fbcd233e6`；提取轨迹 SHA256 `70008c25c9e66451aaf2ef2d8afb6f4a9575269f042f933d320ca5881b9beb96`。
- 源末帧 reward=1 / terminal=true 仅作为原数据标记，回放成功另外用物理状态判定。原三路 JPEG 保存在云端源目录；新渲染图像另存，不冒充原图。
- Arena / Lab / Sim：`2cc9e9766f1f62a500b85c92207192d8c60fcce5` / `91a0906f93802eb859192dcb0bca879f891e92d6` / 6.1.0，独立 Python 3.12。复用已部署 RTX PRO 6000 Blackwell Max-Q 96 GB；本试验不加载 π0.5 权重。

公开共同训练配方确实包含仿真数据，但该 episode 是否进入实际下载检查点的训练仍未知；源路径与公开 sample-range 清单不一致的问题见[训练数据审计](../reviews/2026-10-10_pi05_training_data_audit.json)。本试验没有将“公开共同训练数据”扩写为“已核验实际模型训练覆盖”。

## 试验设计

机器人以源第一帧关节状态初始化，随后只执行原始七关节绝对目标和夹爪动作，不逐帧强制写入源机器人姿态。使用公开控制参数：物理步长 1/120 s，八个子步对应一个原始动作，15 Hz；肩/前臂 PD 为 stiffness 400 / damping 80。夹爪按官方二值接口将原动作阈值 0.5 映射到开 0 rad / 关 π/4 rad。源动作数组和观测数组原值保存。

用公开 USD 的关节锚点计算 CPU 前向运动学，与 Isaac 初始末端位置比较，误差大于 5 mm 则拒绝继续。夹爪基座局部 +X 的 0.14 m 作为试验抓取点。源动作第 47 帧闭合、第 98 帧释放；对应源关节状态估计方块初态、桌面高度和目标箱位置。重建使用未来专家轨迹，属于离线数据构造，不作为策略输入或独立测试。

方块边长 4 cm / 质量 40 g。初态中心约 `(0.48256, -0.25772, 0.15207)` m；目标箱中心 XY 约 `(0.48247, 0.23516)` m，桌面 Z≈0.13107 m。物体初始化后不再写入物体状态，不使用附着、吸附、瞬移或替换专家动作。箱内半宽 0.11 m，壁高 0.055 m；新场景不宣称与原 `pi_scene_v5` 相同。

运行前冻结成功判据：方块抬升至少 5 cm、XY 搬运至少 10 cm、抬升时至少五个闭合动作帧中方块与抓取点距离小于 5.5 cm；最后方块完整进入目标箱，夹爪已开且末端分离至少 5 cm，末 15 帧速度小于 5 cm/s。原 108 帧结束后仅保持末动作稳定两秒，附加帧不写入训练样本。失败尝试全部留档，不能仅凭源 reward 判成功。

同步两路外部/腕部新 RGB 与 CameraCore 原生 HDR/CFA/noise，320×180，GRBG 24-bit ADC 存为无损 little-endian uint32。每次采集冻结物理状态，记录物理子步与控制计数、源帧索引、原动作、实际机器人/方块状态、相机姿态、原生帧号和 SHA256；核对 HDR→CFA 量化误差及文件完整性。新 RGB 来自同一原生相机 render product，独立 PNG 存储。未作真实相机曝光、噪声或响应标定。

这是单源 episode 的功能试验，不构成独立成功率或泛化证据，不进行相邻帧 train/test 随机拆分。若试验成功，新增配对集只证明原动作能在该重建场景成功执行；仍须另行恢复原训练场景和相机，才能声称精确补充原始训练 RGB 的 RAW。

## 云端运行与归档

新目录：`/workspace/owac/reference/polaris-replay-pilot-v1`，supervisor `owac-polaris-replay-pilot-v1`。程序暂时 SIGSTOP 完整基准的外层调度器，让其当前子 episode 正常结束并清理策略服务，确认 GPU 无其它进程后运行回放；finally 恢复调度器。没有同时运行两个模拟器，没有更改冻结基准源码、协议和已完成结果。

```bash
# 云端：独立 Arena Python；每次使用新 attempt 名称。
PYTHONPATH=/workspace/owac/reference/polaris-replay-pilot-v1/source \
  /workspace/owac/reference/pi05-arena-droid-v1/Arena/.venv/bin/python \
  -m owac.runtime.polaris_replay supervise \
  --root /workspace/owac/reference/polaris-replay-pilot-v1 \
  --attempt <新的尝试名> --benchmark-pid <实际调度器PID>

# 本地：仅同步小型数字、状态和哈希；更新固定离线页面。
uv run --locked python -m owac.runtime.polaris_replay_archive
uv run --locked python -m owac.runtime.arena_droid_benchmark archive
```

本地[数字归档](results/2026-10-10_polaris_expert_replay_pilot/)和[固定页面](../../outputs/arena-droid-baseline/progress.html)分别显示本试验与完整基准。所有原始 JPEG、PNG、CFA、HDR、noise 和 NPZ 数组留在云端；没有下载权重。归档程序验证连续源帧、原生帧号、控制时间、量化、腕部姿态与同帧本体的一致性；页面生成时间不冒充远端检查时间。

## 尝试与当前结果

1. `attempt-001`：当前固定 Lab 已删除 CLI 的 `--headless` / `--enable_cameras`；未启动物理回放，随后恢复基准。
2. `attempt-002`：场景加载后退出，未写初始或轨迹文件；Kit 关闭吞掉异常输出，不以 exit=0 判完成。
3. `attempt-003`：加入独立异常文件后定位为 Panda 第四关节的默认零位越界，发生在 `sim.reset()`。下一版以源第一帧作为合法初态；原动作与物体布局没有修改。
4. `attempt-004`：初态与首帧采集通过，CPU FK / Isaac 末端位置误差 0.000428 mm；首次开夹爪目标被建为 int64，Warp 要求 FP32，动作执行前退出。下一版显式指定 FP32；原动作值未改变。
5. `attempt-005`：完整执行 108 帧原动作，再保持末动作 30 帧。物理回放成功，配对轨迹及全部采集已保存；写最终结果时 NumPy bool 无法 JSON 序列化。使用独立 `postprocess-source-v1` 从完整留存轨迹重算原成功规则、重读 CFA/HDR 并核对全部哈希，修复数字留档，没有重跑物理、重新采图或更改成功阈值。原异常、源码、动作和帧文件保留。物理过程结束约 07:37:18 UTC，数字修复完成于 07:40:59 UTC，两种时间不混用。

只有 `attempt-005` 是完整物理回放；前四次为启动/接口失败，其中第 4 次只记录首帧，均不能计入独立专家成功率。不是“五个独立试验成功一条”，也不是 π0.5 的 1/1 评测分数。

| 指标 | 实测 |
| --- | --- |
| 成功原始 source episode | 1 条，方块放入左箱 |
| 原动作 / 训练控制帧 | 108；保存值逐元素等于公开源动作 |
| 附加稳定帧 | 30，不进入训练样本 |
| 实际抬升 / XY 搬运 | 24.1509 cm / 46.0431 cm |
| 满足抬升与抓取点邻近的闭合帧 | 45 |
| 最终箱中心 XY 误差 | 3.4410 cm，完整落在箱内 |
| 松手后 TCP 与方块距离 | 21.3121 cm |
| 末 15 帧最高线速度 | 0.1144 mm/s |
| 源机器人状态与实回放关节 MAE | 0.00190364 rad，约 0.1091° |
| 腕部相机位置一致性最大误差 | 0.0156733 mm，逐帧核对同帧实际夹爪姿态 |
| 视角 / 配对观测 | 外部、腕部各 108 帧，共 216 个视角帧 |
| Native frame IDs | 两路均 11–118，严格递增，无重复 |
| 采集文件 / 哈希不一致 | 864 / 0；216 RGB PNG、216 HDR、216 CFA、216 noise |
| Native HDR→CFA 最大量化误差 | 0 DN，后处理再次重读验证 |
| 原动作改写 / 初始化后物体状态覆盖 | 0 / 0 |
| 首末源帧采集 wall 时间跨度 | 19.0360 s；物理原动作 7.2 s + 稳定 2 s |
| 子进程完整 wall 时间 | 34.3120 s，从队列完成到退出恢复；含 Kit 启动及末尾异常记录 |
| 全系统显存峰值 / 能耗 | 未测量 |

云端训练数据入口为 `/workspace/owac/reference/polaris-replay-pilot-v1/successful-expert-index.json`，实际 episode 位于 `attempts/attempt-005/replay/`。`dataset.json` 连接原动作与源状态、实回放完整关节/物体状态 NPZ、新双视角 RGB、原生 CFA/HDR/noise、传感器元数据、逐帧时序和已导出的重建 USD；`files.json` 和 `result.json` 提供核验。未从原始 RGB 逆处理 RAW；新的 RGB/RAW 同状态配对，原始训练 JPEG 单独保留。当前一条源演示未划 train/validation/test，不能按相邻帧随机拆分后作为独立评测。

训练时新 RGB/CFA 应配对 `replay_joint_positions` 的实际七关节与 `finger_joint / (π/4)` 本体状态，原 `original_robot_states` 保留作参考偏差诊断。方块完整状态是回放/成功核验真值，不进入视觉策略输入。按相机姿态和内参投影检查，两路相机中方块中心都在 108/108 个源帧视锥内；这不是像素配准或无遮挡验证。

本地 `make check` 与 40 项 CPU 测试通过；包括关节锚点数学、截断输入拒绝、抓取/松手判据，以及相机冻结和额外物理步的拒绝。上述 GPU 物理与采集证据独立保留。

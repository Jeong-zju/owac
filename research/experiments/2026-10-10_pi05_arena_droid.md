# Isaac Lab Arena DROID / 官方 π0.5 jointpos 联合验收

## 目的与边界

2026-10-10 用户进一步明确两阶段交付，见 [0009](../../docs/decisions/0009-baseline-then-training-replay.md)：完整 RGB 基准与原始仿真专家训练演示的可验证回放。本记录只包含单任务部署验收；同步采得的 RAW 不是原始训练演示回放，完整基准和配对训练数据集均未完成。原始数字与核验记录保留。

用户要求直接部署现成基线，并且同一仿真任务支持原生 CFA/RAW 采集。不训练或微调 backbone。本轮先验证官方魔方入碗任务，不报告整套 Arena 成绩。准入依据为 [0008](../../docs/decisions/0008-pretrained-and-native-cfa-required.md)。

本地运行分支 `lzh/pi05-sim-baseline`，开始时 HEAD `4764b68`；运行时本任务代码尚未提交，云端各独立源码快照的逐文件 SHA256 保存在小型留档 `source_hashes.json`，包含每次实际 GPU 运行版本。用户原有研究文档变更不参与云端源码部署。

固定协议：[pi05-arena-droid-v1.json](../../configs/baselines/pi05-arena-droid-v1.json)。每个阶段使用环境种子 0、1、2，分别运行一个完整 episode，70 秒仿真时限；RGB 和 RAW 配对。每个 episode 重启官方策略服务器并固定推理种子 7。环境保持官方任务、摆放、机器人、相机、动作接口、终止条件及成功判定。官方 adapter 仅使用 external_camera 与 wrist_camera；额外 external_camera_2 也监测图像刷新，但不输入策略。RAW 阶段使用官方 RGB 策略观测并同时采集原生传感器数据。运行故障没有策略分数；不替换失败的种子。

## 部署与来源

- 云端：RTX PRO 6000 Blackwell Max-Q，96 GB；实例 `212.51.142.37:16406`。
- 独立目录：`/workspace/owac/reference/pi05-arena-droid-v1`。旧实验、检查点和冻结源码保留。
- Arena：`2cc9e9766f1f62a500b85c92207192d8c60fcce5`；IsaacLab：`91a0906f93802eb859192dcb0bca879f891e92d6`。
- OpenPI：`c23745b5ad24e98f66967ea795a07b2588ed6c79`；GR00T submodule：`e29d8fc50b0e4745120ae3fb72447986fe638aa6`，本轮不使用其策略。
- 官方配置：`pi05_droid_jointpos_polaris`；检查点 `gs://openpi-assets-simeval/pi05_droid_jointpos`，不是普通 DROID 或 LIBERO 检查点。
- Isaac Sim 6.1.0.0 / IsaacLab 包元数据 17.0.2 / Python 3.12，OpenPI / Python 3.11，使用各自固定提交的 `uv.lock`。OpenPI 推理环境额外切换同版 Torch 2.7.1 与 TorchVision 0.22.1 的 CUDA 12.8 构建，以支持 Blackwell；官方 Transformers 补丁只应用在独立环境。
- 使用官方 JAX → PyTorch 转换器，BF16，无微调。转换器的 assets 源目录错误被显式修正，复制该检查点自身的 `assets/droid/norm_stats.json`。
- 转换模型 SHA256：`123f0e0e2017c429f72dbf79bc8d3cc7a6dcb4de3054e21f51faaa45e4c155bb`。
- 原始与转换后归一化 SHA256 均为 `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7`。
- 官方 chunk 长度 15，动作是 7 个绝对关节位置及 gripper，沿用官方 adapter；未加入 delta 或自制轴变换。

## 运行状态与故障留档

部署与下载完成，耗时 124.47 秒。RGB 有效闭环 **3/3**；同步原生 RAW 有效闭环 **2/3**。三个预定种子全部有完整 outcome，失败种子未替换。运行使用 supervisor；没有创建定时任务。

| 运行版本 | 实际情况 | 计分 |
|---|---|---|
| rgb-v1 | 启动时协议文件复制尚未完成；已修正同步顺序 | 无 episode，不计分 |
| rgb-v2 | 固定 Lab CLI 已移除 `--headless`；改用受支持的 `HEADLESS=1` | 无 episode，不计分 |
| rgb-v3 | 强制 SPG / 多频率渲染产生全黑 RGB；393 控制步的三路图像哈希分别只有一个值，主动停止 | 渲染故障，不计策略失败 |
| rgb-v4 | 默认 Lab experience 在约 90 秒初始化等待时中断；发现缺少 libXt，补齐依赖；尚不能据此认定永久卡死 | 无 episode，不计分 |
| rgb-v5 | 首个官方 episode 超时失败，但 1050 步三路 RGB 各只有一个哈希；输入冻结，策略成绩无效。根因是 headless KitVisualizer 不泵 app，renderer 却按其声明跳过刷新 | 渲染故障，保留原始官方 false、不计有效策略分数 |
| rgb-v6 | 关闭 Kit 可视化，由相机驱动 app 刷新；三路 RGB 每个控制步均有新哈希，3/3 完成官方全部判据 | 有效 3/3，小样本 |
| raw-v1 | TCP TIME_WAIT 被空端口检测误判为运行中服务器，修正 SO_REUSEADDR 后另建版本 | 未启动 episode，不计分 |
| raw-v2 | compiled 相机路径含正则片段，改用 resolved USD prim handle | 无动作/episode，不计分 |
| raw-v3 | Gf clippingRange 无法 JSON 序列化，改为显式 float 列表 | 无动作/episode，不计分 |
| raw-v4 | 将 stage-3 加噪缓冲误按 stage-2 白电平限制；已移除该错误限制，原值保存并统计越界比例 | 首帧接入故障，不计分 |
| raw-v5 | 新源码目录父目录缺失导致同步失败；supervisor 在启动 episode 前退出 | 无 episode，不计分 |
| raw-v6 | 种子 0/1 完整成功，290/289 帧每视角；种子 2 在 209 控制步发生 SIGBUS，无完整 outcome；磁盘、shm、RAM 充足，OOM=0，底层原因未确定 | 已完成种子有效，崩溃试次不计策略失败 |
| raw-v7 | 保持原协议与采集/策略代码，硬链接保留 v6 的完整 seed 0/1 证据，仅新进程重试原 seed 2；最终 1050 步超时，已抬升但未放入碗 | 有效 2/3；不重复计算复用记录 |

每次修复保留原运行目录并创建独立源码快照。SIGBUS 仅做基础设施重试，未确定根因，也没有宣称已经根治；完成记录复用须显式写明 `resumed_from`、`carried_forward_seeds`，不能作为新的独立 episode 再计数。实际命令、源码哈希、状态、日志尾部及控制记录保存在 [小型留档](results/2026-10-10_pi05_arena_droid/)。

## RAW 采集协议

实际采集 external 与 wrist 两个策略视角，保持原相机内参，直接查询每步 Fabric world pose；不使用默认缓存的 CameraData pose。1280×720，无裁剪/缩放；原生 CameraCore 的 `0-texread` 线性 FP16 HDR、`2-cfa` GRBG uint32 和 `3-noise` 原生噪声版本留在云端。每个动作前额外渲染，不推进物理时间，记录物理步、host 时间、视角 pose、帧编号、校验和及 HDR → CFA 量化误差；检查原 RGB 观测未被采集修改。辅助 ISP 缓冲不作为模型输入。RAW 初始关节状态与相同种子的 RGB 完全一致；额外渲染及随机成像使像素和动作轨迹并非逐位相同，不能把两阶段的差异全部归因于采集成本。

这是合成传感器 CFA，名义 24 位容器受 renderer FP16 精度限制。曝光、物理增益和真实噪声尚未校准，不能宣称真实相机 RAW。stage-3 加噪缓冲在饱和 CFA 附近可超过 stage-2 白电平；单独保存原值，不做 host 裁剪或归一化。三个完整 episode 都已完成动作/状态与 RAW 同步记录；云端已重读全部 9,774 个选中缓冲，SHA256 与逐帧清单一致，全部 HDR→CFA 数值复算误差仍为 0 DN；`verification.json` 保存核验器 SHA256、开始/完成时间与逐种子结果。

## 本地查看与复核

历史离线页面：`outputs/arena-droid-baseline/history.html`。当前完整九配置基准见[新记录](2026-10-10_pi05_arena_droid_benchmark.md)与固定 `progress.html`；下列命令仅刷新本历史验收数字及页面：

```bash
uv run --locked python -m owac.runtime.arena_droid_archive
```

严格白名单只同步小型 JSON / JSONL / YAML / 文本及日志尾部；不下载检查点、权重、RAW 缓冲、图像或数组。检查成功时间与本地生成时间分别记录。

仅重新生成页面和本地哈希清单（保留实际云端成功检查时间）：

```bash
uv run --locked python -m owac.runtime.arena_droid_archive --offline
```

完整性复核入口（仅在云端使用 `--verify-binaries`；本地无大数组）：

```bash
uv run --locked python -m owac.runtime.arena_droid_verify \
  --directory research/experiments/results/2026-10-10_pi05_arena_droid \
  --raw-run raw-v7
```

当前代码检查 `make check` 通过；完整 CPU 测试 30 项通过（包含 RAW 数值与结果计数边界）。RGB 与 RAW GPU 闭环测试均已完成。小样本证明同一任务中可以直接运行官方权重并采集原生 CFA；尚未验证大规模稳定性，也未冻结正式算法比较协议。

## 最终数字与解释

| 阶段 | 种子 | 官方判据 | 控制步 / 帧每视角 | 策略请求 | rollout wall / s | 单次请求 p50 / p95 / ms |
|---|---|---|---|---|---|---|
| RGB | 0 | 成功 | 290 | 20 | 59.297 | 123.505 / 145.750 |
| RGB | 1 | 成功 | 200 | 14 | 39.744 | 123.929 / 224.574 |
| RGB | 2 | 成功 | 176 | 12 | 35.099 | 122.070 / 261.479 |
| RAW | 0 | 成功 | 290 | 20 | 102.021 | 124.699 / 144.109 |
| RAW | 1 | 成功 | 289 | 20 | 98.942 | 123.117 / 141.422 |
| RAW | 2 | 超时失败 | 1050 | 70 | 363.516 | 121.767 / 127.546 |

RGB 成功率 100%，Wilson 95% 区间 **43.8503%–100%**；RAW 同步采集成功率 66.6667%，区间 **20.7660%–93.8508%**。这是相同 3 个名义环境种子的配对阶段，不合并为 6 个独立样本；不报告整套 Arena 分数。RAW seed 2 在第 145 步达到官方抬升判据，但到第 1050 步仍未满足目标容器接触/速度判据，属于有效任务失败，与先前 SIGBUS 基础设施崩溃分开保存。

两路各 **1,629** 帧，共 **3,258** 帧视角样本。每帧保存 HDR / CFA / noise 三个原生缓冲，选中缓冲共 **9,774** 文件 / **48,041,164,800 bytes**（约 44.74 GiB，留在云端）。控制序列 0…N−1、逐视角新 native 帧、物理子步间隔 8、原 RGB 观测未被采集原地修改均通过检查；全部记录的 HDR→CFA 误差为 **0 DN**。不声称保存了额外渲染产生的每一个中间帧。

策略 PyTorch peak allocated 为 **7,644,988,928 bytes / 7.11995 GiB**；这是策略进程指标，不是整个模拟器的峰值。运行中观察到总 GPU 占用约 15.3 GiB，未采集全程总占用峰值。RGB 运行总 wall 286.595 s；RAW 初始运行含崩溃 438.346 s，恢复运行 417.506 s，共 855.852 s。各成功/失败 rollout 的 wall 单独见表，不将恢复记录或不同长度轨迹伪装为受控的纯采集开销。

同一成功 episode 中“现成 π0.5 策略闭环 + 原生 CFA 采集”的两项能力已得到实测证明。RAW 阶段存在一个真实任务失败及一次底层原因未确定的 SIGBUS；额外渲染改变了 RGB 观测序列，不能据此宣称 RGB 与 RAW 模式严格等价或长期运行稳定。后续扩展应验证渲染时序与稳定性，固定所有对照共用的采集管线；不通过追加 backbone 训练补救本次分数。

最终云端模拟器、策略服务器和核验服务均由 supervisor 管理；实验结束后不保留运行中的模拟器/策略进程，不停止租用实例、不删除权重，也不重建定时任务。本地页面以最后成功云端检查时间为准。

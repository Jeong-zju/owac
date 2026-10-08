# 2026-10-08 · Lightwheel-LIBERO 原生 RAW 部署

实验 ID：`lw-libero-native-cfa-deployment-20261008`。负责人：Codex；日期/时区：2026-10-08 Asia/Shanghai。状态：单任务部署与 RAW 采集完成。证据层：工程实现；不验证 F01—F12 候选或新的具身机制。

## 目的与判据

用户只要求一个可运行且支持 RAW data 的 sim benchmark。选择既有调研首选 Lightwheel-LIBERO 中一个原任务：`L90L5PutTheRedMugOnTheLeftPlate`，`libero-1-1`，Panda。验收要求资产加载、环境创建、reset、控制步、有限本体状态，以及连续无损 CFA 导出。零信号、相位错、非有限数、截断/重复帧或异常关闭均失败。

没有训练、调参、学习量、候选比较或 train/validation/test 拆分。本次 32 帧属于同一个采集 episode，不能当成 32 次独立机器人成功试验。保持初始 TCP 位姿并张开夹爪，未使用 π0.5、专家动作或隐藏真值生成策略。

## 固定环境与成像

- 工作目录 `/home/jeong/zeno/owac/repo`；分支 `codex/raw-sim-bench`。
- 上游提交、所用源码文件哈希与兼容补丁：[runtime 目录](../../configs/simulation/lw-libero-runtime/)。LW `b2bcb2d00edef691f9fcc49039cbf0bcc7464605`，Arena `c7b70779f103e10d690d1a13863e8d77da7fc782`，Lab `37ddf626871758333d6ed89cf64ad702aef127d0`。
- Ubuntu 24.04、RTX 5090 32 GB、驱动 595.91.07、Python 3.12.12、Isaac 6.1.0.0、Torch 2.11.0+cu128、NumPy 2.5.1。包导出见 `requirements.freeze.txt`。
- seed 0；context 与环境均设置种子。RTX 渲染及默认噪声没有保证逐位确定性。
- 物理 dt 10 ms，控制 decimation 2。采集期间额外 `sim.render()` 暂停物理更新。时间戳同时记录仿真时间、host Unix ns 与单调时钟 ns；尚未标定 native 内部曝光时间与物理传感器时钟。
- 320×240 GRBG 原生 CFA，ADC 最大 16777215，uint32 little-endian；CFA 原点 [0,0]。全帧无裁切/resize/归一化/去马赛克，NumPy 无损保存。
- 原始 HDR 是 FP16，经 identity correction 后由原生 ADC 采样；不把 24 位整数存储解释为 24 位物理精度。真实相机、曝光秒数、物理增益与坏点标定为空。
- CFA 和原生默认 noise 分开存储。缺失缓冲拒绝采集；饱和掩码独立保存，未知坏点不由黑值推断。RGB 只保存为独立预览。

## 已完成运行

`outputs/raw-sim-bench/acceptance-08` 首次通过，`acceptance-09` 增加饱和掩码与来源哈希，`acceptance-10` 使用最终视角并增加机器人根状态/TCP 记录。三次均真实 reset、32 个控制步、32 帧，native 帧号 48—79，仿真时间约 0.04—0.66 s，最大量化误差均 0 DN。记录在 `report.json` / `frames.jsonl`；运行时未提交源码的逐文件 SHA256 及已跟踪 diff 均保存于产物目录。

`acceptance-10` 的源码基点为 `12e9a49cb25618e1c9b1ee779abfc53acc1ca160` 加工作树中的本次实现；相关未跟踪代码已由 `source_sha256` 固定。另一个工作中的 roadmap v5 文档修改保持原状，运行时 dirty diff 包含它们，未纳入本次功能提交。

| 项目 | 实测结果及方法 | 边界 |
| --- | --- | --- |
| 连续帧 | 32，逐帧新 native ID 与递增时间戳 | 同一个 episode |
| CFA 量化误差 | 最大 0 DN，与同帧线性 HDR 对应 GRBG 通道对照 | FP32 期望 ADC 计算 |
| 有效场景信号 | 每帧至少 2103 个不同 CFA 值（acceptance-10） | 不等于物理传感器有效位深 |
| 采集步 p50/p95 | 38.50 / 115.62 ms（acceptance-10） | 包含步进、额外渲染、保存和哈希；不含初始化 |
| 原任务成功 | false | 保持位姿采集，不测策略成功率 |
| 文件检查 | 保存 CFA 与 native 二进制逐值相同；各文件 SHA256、饱和掩码、帧/时间戳连续性通过 | acceptance-09 已独立离线检查 |
| CPU 检查 | `make check` 与 8 个数值/行为测试通过 | 5 个 RAW 边界测试、3 个 π0.5 部署补丁测试 |
| 显存/资源 | report 保存 Torch allocator 数值 | 不包含原生 RTX/PhysX，不能作为完整 GPU 峰值 |

命令入口见[部署说明](../../docs/raw-sim-bench-deployment.md)。

## 固定代码的最终验收

代码提交 `66117701129eadd95c92d422a3c5a34ad3e7aabb` 后执行：

```bash
cd /home/jeong/zeno/owac/repo
/usr/bin/time -v -o outputs/sim-bench-deploy/acceptance-final-resource.txt \
  bash scripts/run_raw_sim_bench.sh --output outputs/raw-sim-bench/acceptance-final --steps 32 \
  > outputs/sim-bench-deploy/acceptance-final.log 2>&1
```

进程退出 0，`report.json` 为 passed。32 帧原生 ID 48—79，量化最大误差 0 DN，每帧至少 2109 个不同 CFA 值；采集步 p50/p95 为 49.84 / 128.15 ms。完整进程墙钟 17.35 s、CPU user/system 为 31.19 / 5.12 s，`time -v` 最大 RSS 8,361,740 KiB（约 7.97 GiB）。采集 p50/p95 不含初始化，完整墙钟包含它。

独立离线复核全部 32 帧：保存 RAW 与同帧 native CFA 逐值相同，各 RAW/noise/state/mask 哈希正确，饱和掩码匹配，所有保存的状态字段有限，native ID 连续且仿真间隔为 0.02 s，单调时间戳递增。代码和 runtime 配置 SHA256 与运行报告一致，外部源码清单逐文件吻合；LW/Arena 的未改 Python 文件均与固定上游 Git blob SHA 一致，改动仅为登记补丁。详见[小型结果清单](2026-10-08_lw_libero_raw_deployment.json)。原始数据、预览及 native HDR 留在 `outputs/raw-sim-bench/acceptance-final/`，没有提交大数组。

## 失败与限制

原推荐是调研选型，不是已证实的版本组合。GitHub 大归档、git 拉取多次中断；有效文件通过官方 blob API 补齐并校验 Git blob SHA，当前安装是源码子集。新版模块化 Lab 与旧 Arena 不兼容，切换到 Lab v2.3.2 并补齐旧 PhysX/USD、SDK 和 headless 接口；目标任务文件及成功判据未改。

旧 Lab headless experience 没有完整启用 Isaac 6.1 原生相机管线，导致环境能创建但没有 native 文件。改用 6.1 官方 Python experience 并明确启用 sensor processing graph 后通过。早期测试会被 `SimulationApp.close()` 的快速进程退出掩盖异常；最终入口显式保存 failed 状态并传入 `exit_code=1`。

原生示例 ISP 的 `zero size in H0` 位于 CFA/noise 后，不影响本次已对照验证的主数据；320×240 ISP 标定未验证。柜门细长 collision mesh 有 CPU 回退，关闭有 USD 属性诊断。没有全套官方 benchmark 兼容、演示重放、π0.5 闭环、真实相机或 RAW/特征网络验收，也没有冻结正式共同起点标签。

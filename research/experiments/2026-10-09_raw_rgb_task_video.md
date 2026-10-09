# 2026-10-09 · 完整搬杯任务 RAW / RGB 视频

实验 ID：`lw-libero-raw-rgb-task-video-20261009`。负责人：Codex；日期/时区：2026-10-09 Asia/Shanghai。状态：完成。证据层：工程实现与物理仿真演示。目的为展示同一任务过程中合成 RAW 和 RGB 的可视差异；不评价新架构或学习策略。

## 设计与可复现版本

使用已部署的 Lightwheel-LIBERO `L90L5PutTheRedMugOnTheLeftPlate`、`libero-1-1`、Panda 和 Isaac Sim 6.1。控制器通过当前物体真值构造绝对 TCP 位姿目标，以差分 IK 和夹爪动作完成操作；无学习参数、无训练/数据拆分，不接入 π0.5。515 帧属于一个 episode，不作为 515 个独立试验。

源机器人基座偏低，本次展示变体明确加 0.4 m 固定底座、基座 x 偏移 −0.24 m；初始 q2/q4/q6 为 −0.785/−1.57/0.785 rad。关闭机器人 reset 重采样、成功及超时自动 reset，保留原物体布局和成功函数。抓取期间未直接写物体位姿。成功还需验证实际抬升超过 0.10 m、最终杯盘 XY 距离小于 0.04 m、根高度差在 0.025—0.09 m。上游成功函数本身不充分检查接触和放置高度，因此补充物理高度核验。原配置入口另做回归。

工作目录 `/home/jeong/zeno/owac/repo`，分支 `codex/raw-sim-bench`。运行基点为 `a53967420320faca768ea974a6453f2d0c31e7c1`，加本次源码工作树差异。所有实际使用的 `src/owac/simulation/*.py` SHA256 和逐文件快照分别保存在 `report.json` 与 `source/`；离线检查与运行后源文件一致。已跟踪差异及未跟踪清单保存在 `working-tree.patch` / `git-status.txt`。运行时其他研究文档修订保留，并不归入本次功能。

seed 0；不保证 RTX/default noise 逐位确定性。环境为 Ubuntu 24.04、RTX 5090 32 GB、驱动 595.91.07、Python 3.12.12、Isaac Sim 6.1.0.0、Isaac Lab 0.54.2、Torch 2.11.0+cu128、NumPy 2.5.1、Pillow 12.3.0。上游及依赖来源沿用[固定 runtime](../../configs/simulation/lw-libero-runtime/)，运行报告记录其 sources.json 哈希。无显存峰值或能耗测量。

## 成像和显示

相机固定于 `[3.59368, −3.20806, 1.80100]` m，朝向 `[2.34368, −2.00806, 0.95100]` m，USD 焦距 18。640×480 GRBG、原点 [0,0]、24 位 ADC 编码、uint32；线性 HDR 来源 FP16，不能解释为 24 位真实物理测量精度。真实相机、物理曝光和增益留空。`2-cfa` 与 `3-noise` 原生缓冲分开无损保存，默认噪声未经实机标定。

每两个 50 Hz 控制步采一帧，间隔 0.04 s；物理 dt 0.01 s。采集前两次额外渲染不推进物理，RGB annotator 与 RAW 读取对应同一物理状态。native ID 和主机单调时钟递增；RGB annotator 不提供独立曝光时间，本次不声称硬件级同步。杯体位置真值也用于确定预览 ROI；此 ROI 不用于模型输入。

视频左侧为独立 RGB，右侧为 `3-noise` 的灰度显示：`round(255 × log1p(15 × clip(DN / 16777215, 0, 1)) / log(16))`。黑白点和 log 参数全程固定，无逐帧自动拉伸、无去马赛克。下方相同 48×40 区域按最近邻放大 4 倍；RAW 格点只在原生 R/G/B 通道着色，不做颜色插值。显示的 uint8 转换、裁切、放大和 H.264 压缩均不进入原始数据流。

原生 noise 阶段可在 ADC 之后轻微越过白电平：515 帧总计 454 个采样超过 16777215，单帧最多 4 个，最大值 16808975；数据保持原样，仅显示公式裁到白点。主 `2-cfa` 数据在 ADC 范围内，饱和掩码按它生成。未知坏点不由黑值推断。

## 命令与产物

```bash
cd /home/jeong/zeno/owac/repo
/usr/bin/time -v -o outputs/task-video/full-task-04-resource.txt \
  bash scripts/render_raw_task_video.sh --output outputs/task-video/full-task-04 \
  > outputs/task-video/full-task-04.log 2>&1
uv run --locked python outputs/task-video/audit_video.py outputs/task-video/full-task-04
bash scripts/run_raw_sim_bench.sh --output outputs/raw-sim-bench/video-regression --steps 4 \
  > outputs/task-video/video-regression.log 2>&1
make check
uv run --locked pytest
```

复现时须使用新的输出目录；录制入口拒绝覆盖。原始采样、native HDR、逐帧 RGB PNG、本体/物体/动作状态及原成功判据保存在 `outputs/task-video/full-task-04/`。完成后单独进行的离线 audit 脚本和 `audit.json` 保留在该输出集合；小型[结果清单](2026-10-09_raw_rgb_task_video.json)纳入 Git。

成片为 `raw-rgb-comparison.mp4`，SHA256 `71747141bee58a5489f5acfd3f3d5a55f937435916dc024d5735ca5aa8ab7428`。主画面和杯体格点在中间帧与编码后的最后一帧经过人工检查；ffprobe 实际解码计数为 515 帧、25 fps、1280×768、20.600 s。

## 实测结果

| 项目 | 实测结果 | 核验方式 |
| --- | --- | --- |
| 完整任务 | 实际夹持、抬升、搬运、放置、松手、退开及稳定 | 连续 RGB / RAW、动作及刚体状态 |
| 抬升高度 | 0.16747 m | 抬升阶段末与稳定初始杯体根坐标比较 |
| 最终杯盘 XY 距离 | 0.000279 m | 最终刚体状态 |
| 杯盘根高度差 | 0.051725 m | 最终刚体状态及额外高度约束 |
| 原任务成功 | 退开与最终稳定阶段为 true；最后 25/25 帧通过 | 原 `_check_success`，不重复调用其有副作用的 wrapper |
| 连续采集 | 515 帧；仿真 0.06—20.62 s，逐帧间隔 0.04 s | 时间、控制步、native ID 和单调时钟检查 |
| CFA 对照 | 全帧最大量化误差 0 DN | 同帧 FP16 HDR → GRBG ADC 独立核验 |
| 无损与状态 | CFA/noise 与对应原生二进制逐值一致，RAW/noise/PNG/state 哈希全通过，状态有限，饱和掩码匹配 | 全部 515 帧离线复核 |
| 录制墙钟 | 122.02 s，CPU user/system 155.01/21.21 s | `/usr/bin/time -v`；含启动、渲染、编码、保存和关闭 |
| 最大进程 RSS | 8,304,640 KiB，约 7.92 GiB | `/usr/bin/time -v`；不作为完整 GPU 峰值 |
| 数据存储 | audit 时约 4.10 GB（3.81 GiB） | 递归求文件大小；不含另行保存的运行日志 |
| 原入口回归 | 原配置 4 个控制步、4 帧通过 | `video-regression/report.json`，不是新一次任务成功试验 |
| CPU 检查 | `make check` 与 10 项测试通过 | 2 项独立显示变换边界测试，加原有 8 项 |

## 失败与解释边界

前期探针发现 floor-mounted 默认安装位置不适于这条桌面抓取轨迹，且 reset 的机器人重采样会覆盖显式基座偏移；在演示变体中调整安装并禁用该重采样。目标杯有突出把手，抓取中心使用沿杯体局部 y 方向 −0.014 m 的圆柱体中心补偿；完整视频过程中物体仍由物理接触移动。

`full-task-01` 因旧 Lab PreviewSurface API 与当前 USD 材质命令不兼容，`full-task-02` 因旧版 SLERP 只接受一维四元数而失败。修复后 `full-task-03` 连续录出 515 帧，但使用插值旋转和不同初始化朝向的 IK 轨迹在下降时推偏杯子，最终放置失败；报告为 failed。最终 `full-task-04` 沿用已通过的固定抓取朝向和控制轨迹，真实操作成功。失败产物及日志保留，不以它们报告成功。

此结果支持“能够渲染完整物理任务并同步保存原生 CFA/RGB”。它不支持原官方配置的整套 benchmark 验收、学习策略成功率、真实相机噪声标定或新的具身算法贡献。展示变体和真值控制信息不能隐入未来方法/基线比较。共同研究起点标签仍未创建。

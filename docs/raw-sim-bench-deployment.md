# 可运行的 RAW 仿真 benchmark

本地已部署 **Lightwheel-LIBERO 单任务子集 + Isaac Sim 6.1 原生 CFA 相机**。任务为 `L90L5PutTheRedMugOnTheLeftPlate`，场景 `libero-1-1`，机器人 Panda。已实际创建环境、reset、执行控制步并保存连续 RAW。范围是用户要求的一个可运行采集 benchmark；没有执行 π0.5 策略评测、专家演示重放或全套任务验收。

## 直接运行

```bash
cd /home/jeong/zeno/owac/repo
bash scripts/run_raw_sim_bench.sh --steps 32
```

无需启动 π0.5 服务。运行使用专用 Python 3.12 环境，默认 headless，输出目录自动生成于 `outputs/raw-sim-bench/<UTC时间>/`。最后必须出现 `RAW_BENCH_PASSED`，且 `report.json` 的 `status` 为 `passed`。程序异常会写失败报告并以非零状态退出，避免 Isaac 快速关闭掩盖异常。指定已有输出目录会拒绝覆盖。

```bash
bash scripts/run_raw_sim_bench.sh --steps 64 --seed 0 \
  --output outputs/raw-sim-bench/my-first-run
```

默认控制保持 reset 后 TCP 位姿、张开夹爪。每个控制步包含两个 10 ms 物理步；额外渲染不推进物理。上限 300 步，避免越过原任务的 8 秒 episode。`build_environment()` 保留原有 Gym 环境、绝对位姿差分 IK、任务语言和成功判据，可用于后续控制器接入。

## 完整任务 RAW / RGB 对比视频

```bash
cd /home/jeong/zeno/owac/repo
bash scripts/render_raw_task_video.sh --output outputs/task-video/my-task-video
```

该入口连续执行接近、夹持、抬升、移动、放置、松手、退开和最终稳定，输出 `raw-rgb-comparison.mp4`。成功必须出现 `TASK_VIDEO_PASSED`；检查实际抬升超过 10 cm、原任务成功判据为真，以及杯体相对于盘的位置和高度。没有对物体写入瞬移位姿。一次视频是一次物理 episode，不能由视频帧数计算策略成功率。

这是**使用物体真值的脚本演示**。为使源场景中原本较低的 Panda 安装位置可达台面，演示模式加 40 cm 固定底座，基座 x 偏移 −24 cm，并调整三个初始关节位置；关闭机器人重采样及成功/超时自动 reset。任务物体初始布局和原成功函数沿用上游。它不是原配置下的官方 benchmark 分数，也不是 π0.5 策略结果；默认 `run_raw_sim_bench.sh` 仍使用原配置。

对比相机为 640×480；控制 50 Hz，每两个控制步采一帧，视频 25 fps、1280×768。录制时两次额外渲染均不推进物理，RAW 和 RGB 对应同一静止的物理状态；未标定内部曝光时钟或实际硬件同步。左侧 RGB 来自独立 annotator，右侧使用原生 `3-noise` CFA 进行**仅供显示**的固定映射：

```text
L = log1p(15 × clip(DN / 16777215, 0, 1)) / log(16)
显示值 = round(255 × L)
```

主画面保留单通道采样结构，以灰度查看；下方在同一杯体区域放大 4 倍，用 GRBG 格点颜色标记原生通道，不插值、不去马赛克，也不逐帧自动调亮。RAW 是合成传感器测量，默认噪声未经真实相机标定；预览转换和 H.264 编码只作用于视频副本。

原始 `frames/*.raw.npy` / `*.noise.npy` 仍是全分辨率 uint32 无损数据；RGB 逐帧保存为 PNG。`*.state.npz` 保存本体/物体/动作及饱和掩码，`frames.jsonl` 保存时刻、阶段、成功判据、native ID、原始 ROI 坐标和各文件 SHA256。`native/` 保留所采帧的 FP16 HDR、CFA 和 noise，删除未选中的预热帧及下游 ISP 调试缓冲。`source/` 保留运行源码快照；`report.json` 登记改动、显示公式、源码哈希及版本。视频编码需要 `ffmpeg`，中文字体使用本机 Noto Sans CJK。

本轮运行结果与失败诊断见[视频实验记录](../research/experiments/2026-10-09_raw_rgb_task_video.md)。

## RAW 数据

主数据来自 NVIDIA CameraCore 的 **线性 HDR → 原生 CFA ADC**，在去马赛克与 ISP 前读取 `2-cfa*.bin`，不是 RGB8 逆处理。配置为 320×240、GRBG、24 位 ADC 编码、uint32 无损存储，黑/白电平 0 / 16777215。HDR 来源是 FP16；整数容器位数不意味着 24 位独立物理测量精度。

这是**合成传感器 RAW**。物理相机、曝光秒数、物理增益及坏点标定未知，保存为 null。`3-noise` 是原生默认、未经真实相机标定的独立噪声版本。原有 PWL 压扩输出留在 native 调试文件中，主数据采用压扩前 CFA。

| 产物 | 内容 |
| --- | --- |
| `frames/*.raw.npy` | 原始无损单通道 CFA，uint32 |
| `frames/*.noise.npy` | 单独保存的原生噪声版本 |
| `frames/*.masks.npz` | 饱和掩码；不将黑值当成坏点 |
| `frames/*.state.npz` | 关节、机器人根状态、TCP、实际动作、五个物体状态 |
| `frames.jsonl` | 帧号、仿真/主机时间戳、语言、原成功判据及各文件 SHA256 |
| `sensor.json` | CFA 原点、位深、电平、相机位置、处理历史及未知参数 |
| `native/` | 同帧 FP16 HDR、CFA、noise、companding、ISP 调试缓冲；包含预热帧 |
| `preview.png` | 独立 RGB 预览，不参与 RAW 编码链路 |
| `report.json` | 验收结果、源文件/配置哈希、依赖版本及采集耗时 |
| `working-tree.patch` / `git-status.txt` | 实验时的已跟踪差异和未跟踪文件清单 |

采集拒绝空场景、非有限 HDR、超范围 CFA、相位错误、截断缓冲及重复帧。每帧按对应线性 HDR 的 GRBG 通道计算 FP32 ADC 期望值，允许最大 1 DN 浮点舍入差。原始 CFA 没有 JPEG、uint8、resize、归一化或去马赛克。

读取一帧不需要启动仿真：

```python
import numpy as np

raw = np.load("outputs/raw-sim-bench/my-first-run/frames/000000.raw.npy", allow_pickle=False)
assert raw.dtype == np.uint32 and raw.shape == (240, 320)
```

## 固定来源与本机环境

专用环境位于 `configs/simulation/lw-libero-runtime/.venv/`；不复用 π0.5 的 Python 环境。

| 组件 | 实际版本/提交 |
| --- | --- |
| Isaac Sim | 6.1.0.0，Kit 110.3.0 |
| Python / Torch / NumPy | 3.12.12 / 2.11.0+cu128 / 2.5.1 |
| Isaac Lab | `37ddf626871758333d6ed89cf64ad702aef127d0`，v2.3.2，加本地补丁 |
| Arena | `c7b70779f103e10d690d1a13863e8d77da7fc782`，所需源码子集 |
| LW-BenchHub | `b2bcb2d00edef691f9fcc49039cbf0bcc7464605`，所需源码子集 |
| Lightwheel SDK | 1.0.3；目标场景和 Cup030/Cup012/Cup014/Plate012 已下载 |
| 本机 | Ubuntu 24.04、RTX 5090 32 GB、驱动 595.91.07 |

固定信息与全部所用 Python 源文件哈希见 [sources.json](../configs/simulation/lw-libero-runtime/sources.json)，资产哈希见 [assets.json](../configs/simulation/lw-libero-runtime/assets.json)，实际包清单见 [requirements.freeze.txt](../configs/simulation/lw-libero-runtime/requirements.freeze.txt)。完整安装由本机已可运行的 Isaac 6.1 包独立复制后，重新安装本仓库内的 editable 源码；没有依赖其他工作空间的 editable 路径。没有验证新的机器从零安装。

原 LW 安装脚本约束 Sim 5.0，故本部署是**兼容补丁后的单任务工程子集**，不能据此宣称官方整个套件兼容 6.1。补丁均保存于 runtime 配置目录：

- Isaac Lab：PhysX `.impl.api` 路径更新、延迟已移除的软体类型注解、刚体材质判断避开已移除软体 API、headless viewport 空值判断、保持局部变换的 Xform 标准化，以及 USD ChangeBlock 内创建操作的兼容处理。软体任务未验证。
- Arena：Franka-only 模式跳过 G1/GR1 可选包。
- LW：最小源码包发现、关闭本次不用的 teleop 注册、SDK ENDPOINT 导入迁移、跳过资产中 `None,None,None` 的非 fixture size。没有修改目标任务或成功函数。
- 启动：采用 Isaac 6.1 官方 `isaacsim.exp.base.python.kit`，启用 Sensor Processing Graph 和 multi-tick；原 Panda 继续引用官方 Isaac 5.1 资产路径。

GitHub 大归档下载在本机反复中断，当前 LW/Arena 通过已下载的完整文件及 GitHub blob API 补齐所需源码，API 文件核对 Git blob SHA。不是空壳模块，也没有把下载失败当作完整仓库。一般网络环境下可获取固定提交、应用三个补丁，再从仓库根目录用 `uv pip install --no-deps -r configs/simulation/lw-libero-runtime/requirements.freeze.txt` 恢复实际包集合；NVIDIA 和 Torch 索引见清单头部。该 freeze 保留实际安装组合，旧 Lab 元数据约束没有重新做全量依赖解析。

## 验收边界

实际测试已导出连续 32 帧，RAW 与对应 HDR 的量化误差为 0 DN；保存数组与原始 native CFA 二进制逐值相同，文件哈希、饱和掩码与时间戳检查通过。`make check` 与 8 项 CPU 行为/数值测试通过。详细结果见[实验记录](../research/experiments/2026-10-08_lw_libero_raw_deployment.md)。

保持位姿不会完成搬杯任务，原成功判据实测为 false；不报告任务成功率。当前没有 VLA 闭环、真实 RAW 标定或跨任务协议验收，因此不创建 `owac-dev-v0.1.0` 研究共同起点标签。

原生示例 ISP 程序在这个分辨率输出 `zero size in H0`，它位于本次读取的 CFA/noise 阶段之后；CFA 与 HDR 已逐帧验证，预览另走 RGB annotator。不声称这套 ISP 已在 320×240 完成成像标定。场景部分细长柜门碰撞网格使用 PhysX 的 CPU collision 回退，关闭阶段还有 USD 无效属性诊断；本次正常返回，刚体状态与 RAW 校验通过。未测量完整渲染/PhysX 显存峰值；Torch 分配量不能代替它。

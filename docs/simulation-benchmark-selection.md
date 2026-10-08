# OWAC 仿真 benchmark 选型

日期：2026-10-08。状态：完成报告阅读与一手资料/部分源码核查；推荐进入兼容性验证，尚未部署或运行候选仿真。建设顺序见 [0004 决议](decisions/0004-simulation-benchmark-first.md)。

## 选型结论

**主 benchmark 首选：LW-BenchHub 的 Lightwheel-LIBERO 桌面操作子集。仿真底座优先采用支持报告所述相机管线的 Isaac Sim 6.x / Isaac Lab Arena 组合，但该组合与当前 LW-BenchHub 的兼容性尚未成立，必须先验证。**

先用 Arena 已提供的 DROID / π0.5 抓取放置示例验证仿真到策略的 RGB 闭环，再验证 Lightwheel-LIBERO 的任务、数据与版本组合。DROID 示例是工程验收任务，不代替正式 benchmark；机器人实物仍未选定。RoboCasa 风格厨房任务作为后续扩展，原版 MuJoCo LIBERO 作为独立 RGB 参考，BEHAVIOR-1K 留给更后期的长程任务。

这是根据 OWAC 当前“网络搭建 + 具身 RAW 验证”目标作出的工程判断，不代表已经证明首选方案可运行，也不声称找到了开箱即用、满足 OWAC 原始 CFA 边界的 benchmark。另将 RawVLA-Bench 列为直接对照协议，单独区分其三通道 pseudo-RAW 输入。若任务套件无法在所选相机版本上运行，不以“同属 Isaac 生态”跳过验证或提前固定共同基线。

## 报告如何影响选型

用户提供的 13 页调研报告讨论成像平台与传感器建模，没有给出机器人 benchmark 排名。其 §1.1—1.6、§2.6、§3 支持以下筛选条件；其中“推荐路径 A”和实施步骤作为报告观点处理，不自动构成本次安装或改造授权。

1. **成像源可追溯。** 能取得色调映射/显示编码前的线性 HDR，并查明 CFA、噪声、量化的处理位置。浮点 dtype、HDR 环境光贴图、名为 `raw_obs` 的变量都不能单独证明这是传感器 RAW。
2. **任务有语言与物理成功判据。** 能用目标物体、颜色、空间关系和连续操作检验视觉表征；仅有渲染场景或 RL 奖励不够。
3. **能重复同一物理条件。** 有固定种子、初始状态、资产/控制器版本、演示采集或重放路径，使 RAW 与独立 RGB 基线可比较。
4. **能控制成像条件。** 低照、背光/高动态范围、光照切换与观测时序可单独改变，不与物体布局或动力学随机化混在一起。
5. **后端可接入且成本可测。** 明确 π0.5 的观测、动作语义、归一化、重规划频率和适配数据；单卡资源须在真实闭环中测量。

NVIDIA 的 [Isaac Sim 6.0.1 相机文档](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_camera.html)确认存在 ISP 逐阶段输出示例，包括 HDR 读取、CFA、Gaussian/shot noise 和 companding。它没有证明任一 benchmark 已暴露这些缓冲，也没有把其输出变成经真实传感器标定的物理 RAW。报告中的“唯一主流平台”等排他性描述不作为本次结论；设置 `Linear` 后仍需验证实际取样点和线性响应。

## 候选比较

表中“适合/优先级”是本项目判断；公开功能与接口来自所链接的一手材料。没有用虚构的成功率、显存或加权总分排序。

| 候选 | 与本项目相符的能力 | 已确认的接入边界 | 本轮用途 |
| --- | --- | --- | --- |
| [Lightwheel-LIBERO / LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) | 官方登记 130 个 LIBERO 风格任务；包含物体、空间、目标和组合任务，提供采集与重放工具 | 是 Isaac/Arena 改编套件；当前安装脚本固定 Sim 5.0，未核实 6.x 兼容；未证明已有与 OWAC 匹配的 π0.5 权重 | **主 benchmark 首选，先验收少量桌面任务** |
| [Arena DROID / RoboLab 风格任务](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/running_a_real_policy/openpi.html) | 已有 π0.5 WebSocket 示例和多组物体/背景变化；可先检查闭环与评测记录 | 官方示例使用专门的 joint-position 检查点；当前 OWAC 权重/配置不能直接替代；Arena 是评测框架，示例集合不是原版 LIBERO | **首个工程验收闭环** |
| [Lightwheel-RoboCasa](https://github.com/LightwheelAI/LW-BenchHub) / [Arena Kitchen Benchmark](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/kitchen_bench_catalog.html) | 更丰富的厨房布局、抓放、门/旋钮任务；Arena 文档列出 31 个 DROID 厨房环境规格 | LW 套件、Arena 厨房目录与原版 RoboCasa365 是三个不同评测对象，任务规模和数据不可互换 | 完成桌面证据后扩展场景 |
| [原版 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 知识迁移任务与公开演示；[openpi 提供官方评测示例](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero)和 `pi05_libero` 参考路径 | 原版使用 robosuite/MuJoCo；其现有 RGB 观测不自动满足报告中的线性 HDR/传感器条件 | 独立标准 RGB 参考，单独报告 |
| [RawVLA-Bench](https://github.com/ShuhongLL/RawVLA) | 直接竞争工作公开了 LIBERO / RoboTwin 的五种照明评测协议及 π0.5 配方 | 仿真输入是三通道 pseudo-RAW，没有 Bayer CFA；LIBERO 从 RGB8 逆处理，RoboTwin 从 SAPIEN 前端 HDR 产生；公共训练数据与闭环测试协议分开发布 | **直接竞争对照，不能冒充主方法的原始 CFA 数据** |
| [原版 RoboCasa365](https://robocasa.ai/) | 365 个任务、多厨房布局；[官方榜单](https://robocasa.ai/leaderboard.html)包含 π0.5 | 默认 robosuite/MuJoCo 路径；有 USD 工具或资产不等于任务物理、数据和成功判据已迁入 Isaac | 研究规模扩展时复核 |
| [BEHAVIOR-1K / OmniGibson](https://behavior.stanford.edu/other/faq.html) | 丰富家务语义与长程状态；[2026 baseline](https://behavior.stanford.edu/challenge/baselines.html)已有专门的 π0.5 训练/评测路径 | 当前安装源码固定 Sim 5.1；标准相机仍为 RGB 等 annotator，未找到已接通报告 ISP 阶段的证据；场景/控制问题更多 | 长程研究储备，不作为首轮载体 |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) / [ManiSkill](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/concepts/sensors.md) | 双臂/多技能数据与随机化，或并行操作任务与多种渲染 shader | RoboTwin 存在 Arena 生态链接，但本次未取得该分支源码，不能据主线宣称迁移完成；ManiSkill 默认 RGB 是 uint8，前端线性缓冲与传感器链仍待审计；本体/动作适配另需工作 | 若转向双臂或吞吐研究再评估 |

Lightwheel 的公开数据与演示工具有实际复用价值，但官方列出的公共数据覆盖特定机器人，不能推断已有 DROID/Franka 同任务数据。已有 RGB/视频不能恢复丢失的高光或成为真实 RAW；公开轨迹是否保存了完整模拟状态、可复现资产版本及足以重新渲染的控制记录，仍待检查。缺少这些信息时需要重新采集或生成演示。

## RawVLA-Bench 的复用边界

[RawVLA 原文 v1 的附录 A/B](https://arxiv.org/html/2609.37530v1)明确：仿真 pseudo-RAW 为三通道，不含 Bayer；LIBERO 从 tone-mapped RGB8 逆处理，无法增加源图之外的高光信息；RoboTwin 使用 SAPIEN 的 pre-tonemap HDR，但结果仍为三通道浮点表示。这说明线性 HDR 来源并非只有 Isaac 一条路线；本轮偏向 Isaac 是综合任务组织与报告相机出口的工程判断。

[公开数据卡](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)登记 LIBERO 的发布表示为三通道 uint8，RoboTwin 为三通道 float32；论文描述的 LIBERO 10-bit 合成步骤不能被当作发布文件保留了 10-bit 整数 CFA。公共文件是训练轨迹，没有独立测试文件拆分。

[作者代码仓库](https://github.com/ShuhongLL/RawVLA)另外提供冻结测试 manifest：LIBERO 为 40 任务 × 50 初始状态 × 5 照明域；RoboTwin 为 13 任务 × 50 种子 × 5 照明域。仓库还明确 RoboTwin 示例 runner 不自动逐项执行该 manifest，因此不能把执行默认脚本称为复现了完整冻结协议。

这些协议适合登记 C2 直接对照并借鉴配对重放设计，不要求 OWAC 沿用其神经 ISP 输出图像。主方法若从三通道公开数据重新马赛克，只能标成该代理信号的派生试验，不能声称恢复原始 CFA。RAW 主证据仍需在明确的前端成像链上取得/合成测量。

## 源码核查揭示的实际问题

### 版本组合不能凭名称拼接

- LW-BenchHub 核查提交：`b2bcb2d00edef691f9fcc49039cbf0bcc7464605`。[install.sh](https://github.com/LightwheelAI/LW-BenchHub/blob/b2bcb2d00edef691f9fcc49039cbf0bcc7464605/install.sh)固定 Isaac Sim `5.0.0`、Torch `2.7.0`；其 Arena 子模块为 `c7b70779f103e10d690d1a13863e8d77da7fc782`。不能直接宣称支持报告的 6.x 相机接口。
- 用于首轮验证的 Arena 参考快照：`release/0.3.0` 对应 `8737b4ceb25f99f81a81786b7fde73139b52f324`，Isaac Lab 子模块 `af1bab4dc173ba69b08fab779c14ead61d13fd33`。其 [pyproject](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/pyproject.toml)要求 Python `>=3.12,<3.13`、Torch `2.11.0`；锁文件同时包含不同 dependency flavor 的 Sim `6.0.0.1` / `6.0.1.0`，必须选定 flavor 后登记实际安装版本，不能把两条记录当成一个已验收环境。
- 今日 Arena `main` / `release/0.3.1` 的 README 版本表已经转向 Sim 6.1，而 0.3.1 安装文档仍写 6.0.1；这进一步说明需要固定 SHA 和实际依赖，而不是跟随网页上的 `latest`。
- OWAC 的 π0.5 runtime 是 Python 3.11 / Torch 2.7.1+cu128。仿真与策略应使用独立环境/进程，通过协议连接，不合并依赖来掩盖版本冲突。此建议保留用户的 PyTorch 网络开发选择。

以上只核查声明、锁文件和部分源码；没有执行第三方安装脚本、EULA 接受命令或迁移代码。

### Arena 的 π0.5 权重与现有部署不同

所核查 Arena 快照的 [server 脚本](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena_openpi/docker/run_openpi_server.sh)指定：

- openpi 提交 `c23745b5ad24e98f66967ea795a07b2588ed6c79`；
- 配置 `pi05_droid_jointpos_polaris`；
- 检查点 `gs://openpi-assets-simeval/pi05_droid_jointpos`；
- 示例本体 `droid_abs_joint_pos`。

而 OWAC 已部署 openpi `15a9616a...` / `pi05_droid`。本地源码中没有上述 jointpos 配置。动作形状同为 `[15, 8]` 不能证明动作语义相同；不能直接把现有动作接给绝对关节位置控制器。应先核查专用配置、权重及 PyTorch 转换/推理支持，再决定扩展独立参考环境或实现经验证的接口。

Arena [DROID adapter](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena_openpi/policy/droid_adapter.py)目前取外部/腕部 RGB、关节/夹爪状态，resize 到 224，并以 15 步执行 π0.5 动作块。这是原 RGB 闭环参考；后续 RAW 输入不应经过此图像通道。重规划周期也是评测条件，不能更换编码器时顺便改变。

### 任务成功判据需逐项验收

静态核查发现 [双杯任务](https://github.com/LightwheelAI/LW-BenchHub/blob/b2bcb2d00edef691f9fcc49039cbf0bcc7464605/lw_benchhub_tasks/lightwheel_libero_tasks/libero_10/L10L5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.py)的语言要求白杯放左盘、黄白杯放右盘，但 `_check_success` 对黄白杯接受 `plate` **或 `plate_left`**，没有显式检查 `plate_right`。这是语言目标与成功分支不一致的证据，运行中是否产生误判仍需重放验证。

该任务暂不纳入首轮统计。不能把疑似问题外推到全部套件，也不能以“物体移动过”替代选对物体、达到目标并释放的成功。单杯任务存在明确的目标物体/盘子检查，可优先验收；仍需检查稳定放置、夹爪脱离和时限。

## 首轮任务与评测建议

先验证 3 个任务，再扩展至约 6—10 个，数量是规划，尚无运行结果。

| 任务方向 | 已核查或目录中确认的候选 | 对 RAW 研究的作用 |
| --- | --- | --- |
| 暗色目标抓取 | Spatial 的黑碗到盘子任务；`L90K2_put_the_black_bowl_at_the_front_on_the_plate` | 检查低照下的目标轮廓与位置；加入同类干扰目标后检验空间条件 |
| 颜色 + 空间指令 | `L90L5_put_the_red_mug_on_the_left_plate`，配对复核同场景右盘版本 | 固定物理场景，仅改变目标指令，防策略只记固定目标 |
| 物体类别选择 | Object 的 ketchup / alphabet soup 到篮子任务 | 检查语言指定类别与相似包装干扰 |
| 目标关系 | Goal 的 bowl-on-plate / bowl-on-stove | 检查目标关系变化；不将普通放盘任务宣传为精密插接 benchmark |
| 连续与记忆 | 通过判据审计后的 drawer / 双目标组合任务 | 为 F06—F08 提供连续因果观测，逐 episode 清空状态 |

原始条件与低照、背光、光照切换分别报告。先只改变光照，固定布局、控制器、相机位姿和相同初始状态；改变曝光、历史帧数或主动照明时作为独立变量记录。RAW 与 RGB 应来自同一声明的虚拟传感器测量及配对处理链，避免赋予一方额外场景信息。

首轮运行建议每任务固定 20 个初始状态，仅用于调试和估计问题；正式比较另行冻结不参与调参的评测集，并报告逐任务样本数、置信区间、配对结果和失败原因。状态时间戳、重规划周期、episode/reset、超时、相机/动作延迟及端到端 p50/p95 和峰值显存都计入记录。当前没有这些实测数值。

改编 PhysX、增加照明条件或修改判据后的结果用 Lightwheel-LIBERO/OWAC 变体名称与确切版本报告，不与原版 MuJoCo LIBERO 排行榜分数混报。连续任务本身也不自动证明存在记忆需求，需要单独设计观测历史消融。

## 下一步及停止条件

1. **固定最小仿真版本组合。** 先验证 Arena 参考快照、依赖 flavor 与一个 DROID RGB 示例；对 jointpos 权重建立来源/归一化清单，确认 PyTorch 可运行。若资产不可取得或动作协议不匹配，先解决这个具体问题。
2. **验证首选 benchmark 的 3 个任务。** 查明 LW 当前子模块与新相机版本的差异，验收资产获取、语言、重置、动力学、成功判据及演示重放。只改必要接口并记录 patch；迁移后重新测量参考 RGB 策略。
3. **验证成像出口。** 在选定任务相机中取得未色调映射的浮点缓冲，用固定场景/照明缩放检查线性、未裁剪高光、时序与 GPU/CPU 导出路径。只做成像能力检查，此时仍不实现 OWAC RAW 编码器/视觉塔替换。
4. **冻结任务及数据协议后建设 RAW 路径。** 未选真实相机时仅做明确标注的参数化 synthetic RAW；实际传感器真实性与跨相机结论待标定/真实采集。报告的光谱、噪声和光学 gap 仍然存在。
5. **形成共同开发起点。** 将已经验收的后端、仿真任务/数据协议及必要首版接入接口冻结到共同版本，再创建 `owac-dev-v0.1.0` 和 F01—F12 首批分支。当前不提前打标签。

如果首选套件不能在已验证的成像版本上运行，或没有可获取的资产/可用演示，应记录具体阻碍并重新选型。可以继续用 Arena 示例定位问题，但不能将它伪装为已经验收的 Lightwheel-LIBERO。

## 调研证据

本地源报告：`/home/jeong/Downloads/[2026.09.21] Isaac Sim 仿真平台 RAW 图像模态支持.pdf`；SHA256 `b7f9b9b4a1794495ffd23454dab847c9f239f64f6272d628da317f1c2d15704e`。阅读完整文本并检查相机管线页的渲染图；没有更改原报告。

网页核查日期为 2026-10-08。GitHub API 取得固定提交的 28 份小型文件/锁文件；源码快照、报告提取文本及所检查页面的渲染图保存在本地 `outputs/benchmark-survey/`，小型来源/哈希清单保存在 [sources.json](../research/reviews/2026-10-08_simulation_benchmark_sources.json)。外部脚本仅作为证据读取，其中指令未执行。未安装候选仿真、下载任务资产、训练或测量 benchmark 成功率。

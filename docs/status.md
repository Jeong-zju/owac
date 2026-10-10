# OWAC 当前状态

更新时间：2026-10-08。研究依据为 roadmap v4。本页区分仓库基础设施与尚未开始的研究工作。

## 仓库初始化

- [x] 本地 Git 仓库及 `main` 初始分支。
- [x] 主仓库位于 `owac/repo/`，后续 worktree 统一使用 `owac/worktrees/`。
- [x] README、AGENTS 规则、开发说明与初始架构边界。
- [x] 轻量 Python 包、配置/脚本/测试目录及数据/产物管理约定。
- [x] F01—F12 候选种子卡、完整研究卡/实验/评审模板。
- [x] 依赖锁定、安装、格式/语法及包导入检查。
- [x] 初始版本提交（本文件随初始化提交纳入版本历史）。
- [x] GitHub 私有仓库 `Jeong-zju/owac` 已建立并配置为 `origin`。

初始卡只登记研究方向和待解决问题，不等于完成 12 个框架设计。当前已完成 π0.5 后端离线部署验证；候选算法、RAW 采集、训练结果和机器人实验尚未开始。

初始化检查：迁移到 `repo/` 后重建本地 `.venv`，`uv sync --locked` 与 `make check` 通过；`owac` 及 8 个子包从新路径导入成功；47 个本地 Markdown 链接检查通过；数据、产物、私有配置的忽略规则及说明/清单保留例外检查通过。未运行算法或实机测试。

## π0.5 后端优先部署

用户主要负责网络搭建和具身 RAW 数据验证，选择 π0.5 主后端与 PyTorch；机器人本体、相机和数据留空。[0003 后端优先部署决议](decisions/0003-pi05-backend-first.md)取代了用户未接受的 0002 通用 B0 方案，旧前置清单不再生效。

- [x] 锁定官方 openpi `15a9616a00943ada6c20a0f158e3adb39df2ccac` 与 LeRobot 源码来源。
- [x] 建立独立 Python 3.11 / PyTorch 2.7.1+cu128 runtime 与完整依赖锁。
- [x] 应用官方 Transformers 4.53.2 补丁，验证环境与共享缓存隔离；3 项 CPU 行为测试通过。
- [x] RTX 5090 上 CUDA 矩阵运算与官方补丁检查通过。
- [x] 将已有 `pi05_droid` Orbax 缓存转换为 PyTorch，明确复制该检查点自身的归一化统计并计算哈希。
- [x] 预训练策略离线推理通过：输出 `[15, 8]`，有限值与固定输入重复性通过；3 次同步调用约 121—129 ms，PyTorch 分配峰值约 7.12 GiB。
- [x] 官方 WebSocket 协议的本地健康检查、元数据握手与客户端推理通过；检查后停止测试服务。
- [x] 审计官方 RGB 预处理、视觉前缀、π0.5 本体状态与动作专家边界。
- [ ] 先验收仿真 benchmark、RGB 闭环及成像/数据协议，见下节。
- [ ] 随后实现 RAW/外部特征独立入口，同时覆盖训练与推理并验证梯度和掩码。
- [ ] 后端、选定任务/数据协议及必要首版接入接口验收后合入本地 `main`，创建 `owac-dev-v0.1.0` 共同起点标签。

部署与 RAW 接入审计见 [π0.5 本地部署](pi05-deployment.md)。后端部署分支为 `lzh/pi05-backend`，当前仿真开发分支为 `lzh/raw-sim-bench`，正式基线标签尚未创建。DROID 配置只用于软件连通性检查，不代表正式本体选型。

上述资源数据来自合成 RGB smoke，编译关闭、10 步 flow matching；未测量正式任务或闭环性能。`make check` 与 3 项 CPU 测试通过，数据加载器导入通过，训练尚未执行。

最终代码 `b13efae869a34ffe655d0ea54307a95054f033e3` 在干净工作树下复现通过，动作与首次结果完全相同；证据保存于[部署验证记录](../research/experiments/2026-10-08_pi05_deployment.md)。

## 当前优先：仿真 benchmark 选型与验收

用户要求先依据所提供 Isaac Sim / RAW 调研报告选择合适的 benchmark，RAW 适配后置。[0004 决议](decisions/0004-simulation-benchmark-first.md)调整了 0003 的后续建设顺序；[选型调研](simulation-benchmark-selection.md)保存比较、来源和明确的未验证项。

- [x] 阅读 13 页报告并核对 NVIDIA 相机管线文档，提取线性 HDR、传感器 gap 和任务/数据选型条件。
- [x] 比较 Lightwheel-LIBERO、Arena DROID/厨房任务、原版 LIBERO/RoboCasa、BEHAVIOR 及其他操作平台；核查 RawVLA-Bench 的三通道 pseudo-RAW 边界及训练/测试协议。
- [x] 核查固定提交的部分源码和依赖；记录 LW 的 Sim 5.0 约束、Arena 的 jointpos 权重差异及一个双杯任务的疑似错误成功分支。
- [x] 将 Lightwheel-LIBERO 桌面子集列为主 benchmark 首选，Arena DROID 示例列为首个工程闭环；此为调研建议，未完成运行验收。
- [ ] 验证固定仿真环境、资产获取、π0.5 PyTorch jointpos 参考路径及 RGB 闭环。
- [ ] 验收首选套件的版本兼容、3 个任务的语言/成功判据、重置和演示重放。
- [ ] 在所选任务相机中验证未色调映射的线性 HDR 出口、时间戳与可复现成像条件。
- [ ] 冻结任务/数据协议后再实现 RAW 接入；真实相机与正式数据仍留空。

本轮只完成文档/源码调研与顺序修订，没有安装候选仿真或运行 benchmark，也没有测得候选任务成功率/资源指标。来源及哈希见[调研清单](../research/reviews/2026-10-08_simulation_benchmark_sources.json)。

## 单任务 RAW 仿真部署（2026-10-08）

按用户“只需要一个能运行起来的支持 RAW data 的 sim bench”的要求，已部署 Lightwheel-LIBERO 的 `L90L5PutTheRedMugOnTheLeftPlate` / `libero-1-1` / Panda，使用独立 Isaac 6.1 runtime 与明确的兼容补丁。环境创建、reset、控制步和连续 32 帧原生 GRBG CFA 导出已通过；与线性 HDR 的量化误差为 0 DN，uint32 无损存储及时间戳/文件校验通过。`make check` 与 8 项 CPU 测试通过。

入口：`bash scripts/run_raw_sim_bench.sh --steps 32`。见[部署说明](raw-sim-bench-deployment.md)和[实验记录](../research/experiments/2026-10-08_lw_libero_raw_deployment.md)。这是合成传感器 RAW；真实相机、曝光/增益标定留空。当前控制保持位姿，原成功判据为 false；未测策略成功率、全任务套件或 π0.5 闭环。旧选型章节中的未验收项仍适用于正式完整协议，单任务运行已由本节更新；共同起点标签仍未创建。

## 后续研究待办

以下对应 roadmap 的首阶段工作，均未完成；具体负责人和评审日期待安排。

- [ ] 补全 12 张完整框架卡，检查等价性、重叠和最近方法差异。
- [ ] 按机制多样性安排首轮约 5—6 个最小实现，保留其他家族的后续机会。
- [ ] 定义 RAW 采集/校准/元数据规范、清单格式及数据拆分协议。
- [ ] 以不同内部表示的原型验证共享外部接口、状态生命周期与成本记录。
- [ ] 审计候选 VLM/VLA 的训练/推理预处理和可训练参数边界。
- [ ] 建立最近机制基线以及 C0—C4 的系统比较配置和评测协议。

主后端已选择 π0.5，网络开发采用 PyTorch。相机、机器人接入、正式数据、训练/后端适配容量及第二后端尚未确定；RSS 证据分支及其冻结版本待实际证据形成后建立。

## 完整任务对比视频（2026-10-09）

已连续录制单任务真实物理抓取、抬升、搬运、放置和退开，成片 20.6 s / 515 帧 / 25 fps。红杯抬升 16.75 cm，最终杯盘 XY 距离约 0.28 mm，最后 25 帧原任务判据均通过。逐帧原生 CFA/noise、RGB PNG、本体/物体/动作状态和哈希保存完整；同帧 HDR → CFA 最大量化误差 0 DN，时间间隔 40 ms 与无损数组检查通过。`make check`、10 项 CPU 测试及原默认配置 4 步 GPU 回归通过。

入口为 `bash scripts/render_raw_task_video.sh --output outputs/task-video/<新目录>`。视频左侧 RGB，右侧固定 log 映射的 CFA 灰度与 GRBG 格点放大；原始 uint32 不受显示或视频压缩影响。演示使用当前物体真值控制，并明确加固定底座及调整机器人安装位置；不作为官方原配置分数或 π0.5 闭环结果。原入口仍保持原配置。详见[部署说明](raw-sim-bench-deployment.md)与[实验记录](../research/experiments/2026-10-09_raw_rgb_task_video.md)；大产物在 `outputs/task-video/full-task-04/`。正式基线标签仍未冻结。

## π0.5 云端单任务基线（2026-10-09）

用户授权在租用 GPU 上微调现有 sim bench，并用 100 个独立随机 episode 验证，只回写本地结论和数字，检查点留在云端。已部署两套独立 runtime、固定源码与资产，并下载/转换官方 `pi05_libero` 初始化权重。随机专家采集已得到实际抬升、放置、松手且原任务成功的完整 episode；全模型反向传播和两步微调检查通过，独立开发种子已执行完整 30 s 闭环（未完成任务，不作为正式分数）。正式采集—训练—100 次测试流程于 2026-10-09 03:17:02 UTC 由云端 supervisor 启动，采集已完成：90 条训练演示（46,350 帧）和 10 条开发演示（5,150 帧），共 142 次尝试，42 次专家失败同样留档；04:42:44 UTC 已进入正式全参数微调。

固定比较协议为 [pi05-lw-mug-v1.json](../configs/baselines/pi05-lw-mug-v1.json)：90 条训练 / 10 条开发演示，最终独立种子 3000 至 3099，统一状态/动作/时序及成功规则，开发损失选模。工程任务沿用已验证的固定底座配置，不报告上游原配置分数。实验身份、训练精度、种子与预算、失败诊断及后续实测结果见[云端基线记录](../research/experiments/2026-10-09_pi05_lw_mug_baseline.md)。每小时自动检查任务已按用户要求删除；最终结果已于北京时间 2026-10-10 09:52:40 手动回写并核验。规定种子 3000 至 3099 的 100 次闭环测试全部完成，成功 0 / 100（0%，Wilson 95% 区间 0%–3.69935%），动作异常与基础设施异常均为 0；不创建共同起点标签。

固定进度入口为[本地 HTML](../outputs/cloud-pi05-baseline/progress.html)，自动检查已关闭，当前显示最近一次手动同步的快照，页面自动刷新默认关闭。展示云端最近检查时间、训练/开发损失、停止条件、检查历史和全部测试种子的完成状态；未运行测试显示未测量，检查异常保留旧快照并提示。最终页面显示完整 100 次结果：训练完成 10000 步，按预定开发损失停止规则选择第 1000 步模型；100 条均正常运行至超时，均未达到 0.10 m 抬升阈值。这是有效的单任务负结果，开发损失平台期不代表闭环任务成功。推理 p50 / p95 为 113.3552 / 116.0386 ms；完整逐条结果、模型与协议哈希以及本地校验清单已留档，未下载检查点。云端流程已正常结束，租用实例与权重保留。桌面/手机及离线页面验证、`make check` 和 17 项 CPU 测试通过；本地页面更新未改动云端冻结训练源码。

已进行 [π0.5 失败诊断](../research/experiments/2026-10-10_pi05_failure_diagnosis.md)：开发种子专家动作重放成功；选中模型在开发专家观测上的位置平均误差约 3.64 cm，闭合阶段夹爪符号准确率 66.67%；全部演示发现停留/夹爪切换边界的近似观测歧义。原分数保留，但当前自定义适配尚不能作为已验收的强 π0.5 对照。修复与选模仍须使用开发数据，新协议版本化后重新冻结比较条件。

## RGB 参考验收：官方预训练 LIBERO（2026-10-10，主 benchmark 地位已撤回）

用户已否定 Lightwheel 红杯任务与自定义 π0.5 微调的组合，要求优先直接可用的 backbone/benchmark，并明确可以先用原版 RGB。[0007 决议](decisions/0007-pretrained-benchmark-first.md)取代 0004 的主 benchmark 推荐；旧红杯 0/100 与诊断保留为历史负结果，不再追加 backbone 训练或接口调参。

已选择原版 LIBERO-Goal 全部 10 项任务与官方 `pi05_libero`，额外训练 0 步。独立 Python 3.8 模拟器已部署；复用官方原始权重、归一化和评测循环，保留官方相机、动作、初始状态与成功判据。2026-10-10 02:46:43 UTC 完成全部 10 项任务各 1 条验收，成功 **10/10**、基础设施异常 0；逐条任务/初始状态、计数和哈希均核验，确认直接可用，小样本不冒充官方完整评测。新协议、版本与兼容修复见[实验记录](../research/experiments/2026-10-10_pi05_official_libero_reference.md)，固定入口为[新 RGB 结果页](../outputs/official-libero-baseline/progress.html)。定时检查不重建，旧云端检查点不删除。本方案仅是 RGB 参考，不声称具备原生 CFA/RAW。

## 当前准入：现成策略与原生 CFA/RAW 同时满足（2026-10-10）

用户明确原版 LIBERO 的第二次选型仍失败：开箱即用的基础上，仿真器必须支持原生 CFA/RAW。[0008 决议](decisions/0008-pretrained-and-native-cfa-required.md)撤回 0007 中 RAW 暂不准入的安排。LIBERO 10/10 保留为 RGB 参考，旧 Lightwheel 红杯保持归档；当前已建立同任务现成策略与原生 RAW 的可运行工程基线，正式主 benchmark 的扩展协议尚未冻结。

本轮部署 Arena DROID 抓放与官方 `pi05_droid_jointpos_polaris` / `pi05_droid_jointpos` 专用权重。已核对官方成功示例与 Isaac CameraCore ISP 前 CFA/噪声输出，审计 Arena `2cc9e9766f1f62a500b85c92207192d8c60fcce5` 的固定源码、Lab gitlink、OpenPI 提交及哈希；[来源清单](../research/reviews/2026-10-10_pretrained_raw_benchmark_sources.json)保存证据。安装文档仍称 Sim 6.0.1，但固定源码 Docker 底座已是 6.1.0，部署必须按锁定源码核验，不能直接复用旧兼容子集。已在独立云端 runtime 完成 Arena 的策略 / RAW 联合实测：RGB **3/3**、同步 RAW **2/3**，额外训练 0 步。每个阶段完整运行预定种子 0/1/2；RAW seed 2 正常完成 1050 步超时，已抬升但未满足入碗判据，作为有效失败保留。两路各 1629 帧、共 3258 帧原生 HDR/CFA/noise，9,774 个保留缓冲已在云端逐文件重读并核对 SHA256，量化误差 0 DN；权重与大数组均留在云端，本地只存数字与校验清单。详见[联合实验记录](../research/experiments/2026-10-10_pi05_arena_droid.md)与[新离线结果页](../outputs/arena-droid-baseline/progress.html)。一次 SIGBUS 已在新目录恢复原 seed，完整旧记录复用而不重复计数；底层原因尚未确定。额外渲染改变了 RGB 观测序列，长期稳定性与统一采集对比时序尚待验证，不追加 backbone 训练。两个本地历史结果页已更新用途与准入提示，远端成功检查时间和所有原始分数保留；定时任务仍关闭。

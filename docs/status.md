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

部署与 RAW 接入审计见 [π0.5 本地部署](pi05-deployment.md)。当前在 `codex/pi05-backend` 分支进行，正式基线标签尚未创建。DROID 配置只用于软件连通性检查，不代表正式本体选型。

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

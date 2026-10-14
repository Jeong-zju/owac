# P8 RawVLA Bench 全量闭环验证

只执行 P8。读取[共同契约](00_SHARED_CONTRACT.md)、[远程执行](03_REMOTE_EXECUTION.md)、P7 的冻结模型/服务/观测时序以及[协议0012](../../decisions/0012-rawvla-pi05-protocol-freeze.md)。目标是形成完整可逐条配对的仿真结论；成功与失败都按预定清单交付。

## 1. 冻结实际运行输入

读取主 checkout [完整 Default 报告](../../../../../repo/research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)及服务器 `records/pi05-complete-report-20261014-v1/`，核对 complete-summary、identity-audit、逐episode记录、模型和manifest hash。已有 Default 主分母为13250，额外632单列；不要因 N01 旧文档说“不完整”重新启动 Default。

为每个平台×方法×训练seed建立明确 run table：checkpoint hash、源码快照、校准/支持版本、动作/观测节奏、manifest hash、输出目录和预算。S2/S3/S3+/S4 按 P7 冻结部署档各三个训练seed；如全部执行，主条目是 `4×3×13250=159000`，不包含复用 S0 或另行 S1。先按开发 episode 时长估算资源，建立分批可恢复队列，不同时占满 GPU。

每个 checkpoint 独立完整身份清单；三个种子不能各跑1/3清单后拼成“全量”。P7 未训练的源消融不能在此临时新训或用 test 选择阈值；对照缺失明确写缺失。

## 2. 新评测入口与客户端

新增 `runtime/n01_full_eval.py`、`evaluation/n01_full_closed_loop.py`，两平台客户端适配放独立 `runtime/n01_full_clients/`。复用 `runtime/rawvla_full_eval.py` 的身份、seed、循环、失败和审计逻辑，以及 `runtime/rawvla_robotwin.py` 的动作协议；新服务/RAW 请求接线独立，不改原冻结 snapshot。

RawVLA 原客户端多走 RGB/ISP payload；必须在实际 simulator 的作者 RAW 路径取本次观测，传给 P7 服务的 `frames`。LIBERO 作者uint8线性视图与 RoboTwin HDR-derived float32 分开，按 P7 已验收的单位校准。禁止从离线训练 NPZ 代替正在运行的当前观测。RGB 仅单独录像或 S0/S1 基线。

对 S3/S3+/S4 使用完全相同 RAW 到达时刻、相机、数据搬运、action horizon/replan 和随机噪声身份。先在独立开发初态做完整 rollout，核对 frame_id、state/action normalization、夹爪、动作维度和成功判据，再冻结 source bundle 并启动主清单。渲染/额外相机读取可能消耗 RNG，显式隔离并核查每个身份的初始化/成像随机流。

## 3. 主清单与扩展条件

主协议：LIBERO 四套40任务×50初态×5照明=10000；RoboTwin 13任务×50 dataset seed×5照明=3250。LIBERO horizon10/replan5，RoboTwin horizon50/replan50、14维绝对关节目标。保留原最大步数、task success、动作单位、语言seed、归一化、lighting和render兼容修复。

632条原缓存条件独立配置与分数，不能加到3250里。S0 是已有0额外训练 Default；S1 是实际作者 neural ISP 方法，按 P7 资产/训练状态处理；S3/S3+/S4 才是核心机制公平比较。

另建动态历史协议：遮挡解除、局部可观察变化、临时测量异常、错配历史。固定事件时刻随机流和可观察条件；隐藏事件标签只用于评价，模型不读取。人为错误历史是机制压力测试，单列；改变任务几何导致正确动作改变时使用有效新示范/规划或独立任务验证，不能机械复用旧动作标签。

## 4. 运行与恢复规则

为每条身份记录 pending/running/success/normal_failure/infrastructure_error。正常超时和策略失败进入原分母；基础设施错误保留原log、相同身份新attempt，按冻结规则恢复，不能换seed或挑最佳。选择一个有效attempt的规则在运行前写入（正常完成的首个有效attempt）；所有attempt仍留档。

调度器启动先查已完成ledger/source/checkpoint hash；仅调度本队列未完成身份，不重跑成功或失败。每批后审计身份重复、缺失、额外case、异常分类、动作实际执行。若渲染或planner异常，先独立复现开发case，修复版本与受影响范围留hash，原结果保持。

## 5. 报告与统计

逐平台、逐任务、逐照明、逐训练seed给成功/完成/原分母，clean/adverse 和原协议汇总分别列。用相同 task/initial-state 配对成功差值，分层/聚类 bootstrap 以任务和初态为单位，并包含训练seed不确定性；照明变体和相邻帧不能当独立重复。保存每个seed结果，禁止只报最优seed。

动态套件另报事件后动作误差/恢复、前8帧AUC、未恢复比例/删失、deadline超期率、错误历史影响，以及静态积累代价。总时延分解 RAW生成→传输→motion→support/rebuild→tokens→π0.5→action execution，给p50/p95/p99；累计GPU小时、峰值显存、状态与数据量完整计数，功耗未测则不编造。

闭环规划门槛是受损条件改善3个百分点、clean退化≤2个百分点，实际以 P0/P7 已冻结检验力/协议为准。完整执行不要求正结果；如果对B3没有额外收益，则准确限制 N01 主张。仿真pseudo-RAW结果不代表原生CFA或实体机器人成功。

## 6. 最终交付

`P8_report.md`、全量逐episode ledger、身份audit、配对统计脚本/配置、总成本、失败分类、动态独立表、每个checkpoint完整覆盖证明和代表性成功/失败回放。小型数字与说明回写本地，模型/数组/录像留服务器或指定产物目录。

运行 `make check`、原全量身份/协议测试和新客户端/统计回归。最终分别判定实现语义、表征任务、仿真闭环，保留不成立和未验证项。停止于本阶段交付，不自行公开发布、改动原标签或继续新路线。

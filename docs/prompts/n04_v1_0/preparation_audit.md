# N04 提示词准备核查

核查日期：2026-10-14。范围为提示词设计所需的源码、既有结果、远端资产和接口；没有训练新模型，也没有重新独立验收整个历史基准。

## 当前 checkout 与证据

主仓库 `/home/jeong/zeno/owac/repo` 当前分支 `lzh/pi05-sim-baseline`、HEAD `16075fc`；N04 worktree `/home/jeong/zeno/owac/worktrees/n04-egomotion-separation` 当前分支 `lzh/n04-egomotion-separation`、HEAD `fd6255b`。两者均有既有未提交修改。本包按 N04 的实际文件内容准备，不能只用 HEAD 代表全部内容。共享 stash 中还有 FilterPy 比较改动，本包没有恢复它。

| 实读入口 | 当前实际能力 | v1.0 必须补齐的部分 |
| --- | --- | --- |
| [MeasurementFrame / StreamingEncoder](../../../src/owac/core/encoder_contract.py) | 原始测量、可用性和最小协议 | 物理时间、执行区间、到达时间、标定来源及正式状态适配 |
| [RawVLASequenceLoader](../../../src/owac/raw/rawvla_sequence.py) | 原数值张量、平台字段、source_identity、动作监督 mask | `executed_action=None`；时间单位 frame；无可靠 K/pose；序列整段载入须按正式数据规模 profiling |
| [proprio.py](../../../src/owac/frameworks/n04/proprio.py) | wxyz 相对旋转实现 | `R_current R_previous^T` 与 body-frame 注释不一致；S0 先用非交换旋转确定坐标语义 |
| [geometry.py](../../../src/owac/frameworks/n04/geometry.py) | GridSpec、warp、forward_splat、SE(3) 投影函数 | GridSpec 仅 stride 不能代替完整裁剪/CFA/卷积中心仿射；需逆采样与可见性独立验收 |
| [N04Encoder](../../../src/owac/frameworks/n04/model.py) | tiny MeasurementStem、predict/correct、特征场、change、RAW 历史参考、读出 | 没有接入作者 RKN；正式 ConvNeXt、空间状态和三解释修订均未实现 |
| [RKNPositionTracker](../../../src/owac/baselines/n04_tracking.py) | 固定二维测量，作者 cell 当前 update 后生成 next prior | 只用于独立诊断；不能当成正式 N04；需逐锚点 mask、空间关联与 256/512 正式维度 |
| [vendor RKN](../../../src/owac/baselines/_vendor/rkn/cell.py) | `get_transition_model`、`_update`、`_predict` 与结构化协方差 | 原默认观测维度由调用者给定；mask 前应先去除无效 NaN，不能直接用乘零处理 |
| [旧训练器](../../../src/owac/training/n04.py) | 原型训练、真实 NPZ smoke、schema 2 checkpoint | 不是充分训练引擎；新模块需完整 RNG/scheduler/scaler/sampler/resume 身份，参数必须先注册后构建 optimizer |
| [RawVLAFeaturePolicy](../../../src/owac/backends/rawvla_features.py) | 实际 PI0 forward/sample 前缀入口、语言 mask、动作 mask | 仍调用旧 `step(values, context, state, **kwargs)` 和 `readout_tokens`；不自动满足通用 StreamingEncoder；需显式适配正式模型 |
| [真实权重 smoke](../../../src/owac/backends/rawvla_feature_smoke.py) | 创建 tiny N04 后通过真实权重前反向 | 当前没有正式 full checkpoint 入口；S7 必须新增而非直接重跑旧脚本报完成 |

历史同测量试验支持“动态状态对连续运动缺失有用”，同时显示停止/反向滞后；旧新增分解/change 的增量与闭环收益未建立。相关报告部分文字提及 FilterPy 旧实现，当前源码已拆成 RKN 保留版，复现旧数字需要原运行源码，不能以当前 `n04_tracking.py` 冒认全套历史对照。[同测量结果](../../../research/experiments/2026-10-13_n04_matched_tracking.md)、[RKN 保留交付](../../../research/experiments/2026-10-13_n04_rkn_delivery.md)应一起读。

## 当前远端资产

使用用户指定的 `ssh -p 44327 root@107.209.104.125` 完成只读核查。GPU 为 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，报告显存 97887 MiB，检查瞬间占用 32047 MiB；四个历史策略服务仍 RUNNING。资源是瞬时信息，开发时重新检查，不自行停止其他服务。

远端另有独立 StarVLA checkout `/workspace/owac/rawvla/source/starVLA`，HEAD 为 `57d508cb180d0510ac9060358009c0f6ffaac420`；实际包路径是 `/workspace/owac/rawvla/source/starVLA/starVLA/`。已用 `importlib.util.find_spec` 核对，不把顶层 RawVLA 根路径和 StarVLA Python 包路径混用。PI0 调用 action expert 的 q/v 投影路径已实读，S7 据此固定 LoRA 模块。

StarVLA 环境 `/workspace/owac/rawvla/envs/starvla/bin/python` 实报 Python 3.10.21、torch 2.7.1+cu128、NumPy 1.26.4。本地包要求 Python >=3.11、encoder extra 的 NumPy >=2.3，故不能把本地 `uv sync` 直接套在原远端环境。新纯编码器环境与冻结后端环境的处理见[运行规约](remote_execution.md)。

远端文件扫描得到 LIBERO RAW NPZ 1771 个、RoboTwin RAW NPZ 632 个；这是文件计数，S0 仍须从 metadata 建立独立源轨迹和变体分组，不把计数直接当独立训练样本数。两平台权重、归一化统计、tokenizer 路径均存在。

最新 `/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1/complete-summary.json` 生成于北京时间 2026-10-14 10:44:24，报告主清单 13250/13250、附加原条件 632/632。LIBERO 5820/10000=58.20%，RoboTwin 508/3250≈15.63%。报告保留 33 条修复记录及原 3217 条未变声明；配套 `identity-audit.json` 已读取，S0/S7 继续核对其 errors、身份及 hash。

配套身份审计实读为 `state=complete`、`errors=[]`、`expected=completed=13250`。本轮没有重新逐条重算全部 rollout，只核对当前小报告与身份审计。这更新了本地旧状态页的进度快照，不扩大证据范围：属于仿真 Default PI05 完整清单结果；OptiX 适配、StarVLA 与原 LeRobot 数值差异仍在报告限制中，不代表 learned ISP、原生 Bayer、真实机器人或 N04 能力。

## 本轮验证

实际执行 `uv run --locked --extra encoder pytest -q tests/encoders/n04`：198 passed，两个已有 warning；`make check` 通过。测试确认当前原型的工程起点可运行，不证明 v1.0 实现完成。新增提示词另做本地链接、引用、命令语法、阶段覆盖和源码未变检查，结果写入 [preparation_validation.json](preparation_validation.json)。

本轮网页确认 RKN/Ac-RKN 论文、ConvNeXt Base 定义、MTS3 acRKN 入口、OpenVINS nullspace 文档、固定 RAFT/Dynamo-Depth 代码页和数据卡。GitHub REST 固定新增 SHA 时遇到 403 rate limit；raw.githubusercontent.com 下载遇到连接重置。因此 ConvNeXt/MTS3 使用已读的分支链接，并明确要求正式集成时固定 SHA；没有编造 commit 或把源码页可读写成完成代码复现。其余既有固定 SHA 的来源层级见[参考导航](references.md)。

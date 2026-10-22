# P7 π0.5 接入与充分动作训练

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> **已被 v1.1 取代（2026-10-15），不再执行本提示词。** 当前正式路线见 [v1.2 完整方案](../../N01_full_development_plan_2026-10-15_v1.2.md)。旧局部阶段正文仅保留供追溯；新 P3 必须整体交付编码器、真实任务训练与开发闭环，不能按旧步骤继续或只改编号。新执行包尚未编写。

只执行 P7。读取[共同契约](00_SHARED_CONTRACT.md)、[远程执行](03_REMOTE_EXECUTION.md)、P4–P6 的实际配置/模型/结果与[早期预检](EARLY_POLICY_PREFLIGHT.md)收据。目标是让完整 N01 成为实际可训练、可部署的 π0.5 前端，并公平比较后端适配成本。

## 1. 先复用正确入口

阅读 `raw/rawvla_sequence.py` 的 `RawVLAProtocol / RawVLASequenceDataset / normalize_actions_and_state`；`backends/rawvla_features.py` 的 `VisualTokenPostprocessor / RawVLAFeaturePolicy / load_audited_with_replaced_vision`；`training/n01_action.py` 的 `observation_from_batch / action_masked_loss / load_backend`。读取服务器固定 StarVLA `PI0.py::embed_prefix`、flow forward 与 `PI05.py`，不要切到最新版 openpi 另接一套。

新增 `training/n01_full_action.py`、`runtime/n01_full_service.py`、`runtime/n01_full_policy_audit.py`，复用明确公共函数；旧 `build_bridge` 只构造旧 N01，不直接调用它加载新 checkpoint。模型工厂按配置创建 B0/B2/B3/N-TS/已采纳状态，保存 backbone/input/support/calibration/parameter 版本，不能靠 state_dict key 猜模型。

新增两平台 `configs/encoders/n01_full/p7_*`、独立部署配置与测试。沿用 P4/P6 的三通道输入适配与共同投影，所有主对照输出每有效相机 256×2048。token 位置、相机顺序、缺失相机 mask 与原视觉前缀缩放逐项核对。

## 2. 动作协议与时间对齐

| 字段 | LIBERO | RoboTwin |
| --- | --- | --- |
| RAW 相机 | agentview、wrist | head、left、right |
| 数值 | 作者 uint8 /255 | 作者 float32 passthrough，不再 /255 |
| 控制维度 / flow 内部维度 | 7 / 32 | 14 / 32 |
| horizon / replan | 10 / 5 | 50 / 50 |
| 离线 state | `robot_states` 8维实测本体 | 作者 `actions[t][:14]` 缓存关节目标 |
| action offset | 0 | 1 |

RoboTwin 的缓存关节目标不是已测执行状态；现有协议明确 `proprio_available=False`，不能当真实传感器反馈或未来专家动作输入 N01。在线状态遵循已冻结客户端。保留作者约定的动作归一化，另报训练/在线 state 语义差异，不自行拿16维末端状态替换14维关节。

**观测 stride 不等于 action horizon，也不必等于 replan。** LIBERO loader 当前 stride=5 已核验；RoboTwin 默认 stride=1 但 `observation_stride_verified=False`。本阶段必须读取实际客户端/缓存时间戳，冻结观测到达规则。首版所有 RAW 特征组只在统一 replan 请求时接收当前真实 RAW；离线历史按实际时间间距抽取（能验证一帧对应一步时 RoboTwin 为50步，否则依据 timestamp 映射），保持训练/在线可用信息一致。若要每控制步 observe，则另建所有组共享的采集协议并计成本，不能只给 N01 更多帧。

在模拟器产生 RAW 的时刻分配 `episode/camera/frame/timestamp`，多次 RPC 或去噪重读沿用同一 ID，禁止 RPC 次数作为帧 ID。动作缓存按 horizon/replan 正确执行，十步 flow denoising 不调用 observe。

## 3. 正式训练器

训练与服务共用 `encode_observations` 和模型工厂。loss 采用既有 flow velocity 目标和动作时间 mask；保留模型32维训练目标与最终7/14维控制边界，不能另写 joint MSE 替代 flow matching。

第一段训练 encoder+postprocessor，backend 参数冻结但 forward 保持梯度可回传；禁止包在 `no_grad` 里切断 encoder。第二段统一解冻同样的语言前缀相关模块与 action expert，视觉图像塔保持移除。后端列实际参数名与数量清单；不依据含糊字符串“视觉相关”猜模块。

第一段backend保持eval模式；第二段按实际参与训练的模块设置train模式并固定dropout/normalization配方，所有组相同。保存encoder、postprocessor和policy实际子模块的state_dict；`RawVLAFeaturePolicy`并非自动注册全部参数的标准模型容器，不调用一个不存在的整桥state_dict代替完整checkpoint。

开发 LR 候选：encoder/bridge `1e-4/3e-4`，backend `1e-5/3e-5`；warm-up 1k–2k 为初始开发候选。用两档受控配方选择后冻结，两阶段总预算 30k–100k/平台/种子，选定 P0 实测可行有效 batch 和梯度累计。LoRA 只有在固定后端已有正式实现且适用时用相同 rank/模块配置；否则分组微调，不另造适配网络。显存不足先梯度累计、作者支持的 checkpointing，再共同选择更窄解冻范围，不只缩 N01 的对照。

正式训练 S2 当前帧、S3 DBSR、S3+ DBSR+reset、S4 完整 N01；至少三个模型种子，两个平台分训。冻结后端与同范围解冻两档分别给开发曲线，最终每臂部署档按预先相同开发规则选定。原 S0 Default 直接复用固定结果，不把0额外训练当容量匹配。

S1 RawVLA 官方 neural ISP 是独立系统对照：核对作者实际配置/权重/训练说明，若资产可用按其方法跑独立支路，RGB/ISP 只存在这条基线。缺少可复用方法权重时明确记录训练与数据需求，不能拿 Default ISP 冒充 S1；核心 S3/S3+/S4 比较继续，S1 完成状态单列。

## 4. 验证的具体动作

1. 两平台真实 batch 分别做前向、backward、optimizer step，保存 encoder、bridge、选定 backend 参数的前后差值；冻结参数差为零，解冻参数确有更新。
2. 图像塔调用计数为零；语言、state、action expert 确实使用；token padding 不影响合法输出，反向 mask 正确。
3. 连续运行和恢复运行下一批 identity/随机 noise/time/参数一致到声明容差，保存所有训练状态，不能沿用旧 smoke 仅有 torch CPU RNG 的 checkpoint 格式。
4. 参数更新、episode、相机校准变化、时间不连续、partial reset 均正确处理 cache；服务重复请求不增加支持数。
5. source 固定决策内容消融、actual motion / oracle 诊断、单相机缺失与全无效路径；候选和基线相同观测信息。
6. 独立服务真实 RPC、实际归一化/反归一化动作、两个平台开发初态闭环。服务内返回 normalized actions，客户端按唯一位置反归一化，避免重复处理。

早期开发闭环只使用独立开发任务/初态；覆盖动作方向、夹爪、视角和事件恢复，先少量完整 episode 再扩大。正式模型选择依据预注册动作验证及开发闭环规则，不能只看 loss；正式 P8 test 不进入选择。

## 5. 交付

完整训练在服务器 supervisor 执行；本地保留数字和 hash。交付 `P7_report.md`、各平台/方法/种子冻结与解冻训练曲线、平台状态、开发闭环、完整参数组、可加载 checkpoint 清单和服务启动/客户端命令。无优势或预算未到平台均据实保留。

本地 `make check`、loader/桥/旧动作回归与新服务测试；远程真实训练/服务/开发闭环收据齐全。达到本阶段训练与接口要求后停在 P7，由下一条提示词执行全量评测。

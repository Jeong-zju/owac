# N01 v1.0 当前实现审计

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

核查日期 2026-10-14；N01 HEAD 为 `429591e`，其父研究结果 `85790cc`。本文件依据实际源码与已存实验收据；本次不重新宣称旧模型训练或闭环验收。

## 已有资产与具体缺口

| 位置 | 实际已有实现 | v1.0 应怎么使用 |
| --- | --- | --- |
| [外部契约](../../../src/owac/core/encoder_contract.py) | `MeasurementFrame / EncoderOutput / VisualTokens / ComputeBudget` | 保留，新增内部类型不破坏公共接口 |
| [旧状态](../../../src/owac/frameworks/n01/state.py)、[旧积分](../../../src/owac/frameworks/n01/support.py) | 唯一帧 buffer、逐相位空间积分、开闭与重算 | 作为回归和来源统计参考；不把固定历史改名为时间来源撤回 |
| [旧运动与对应层](../../../src/owac/frameworks/n01/motion.py) | `CausalCorrespondence / MotionField / MotionChain`，简化运动/采样对应 | 小型对照和坐标语义参考，正式运动由 P2 补齐 |
| [DBSR](../../../src/owac/baselines/burst_features.py) | `RawBurstBackbone / MaskedDbsrFusion / BurstFeatureEncoder`，64/9/512，重算源特征 | 保留完整成熟模块。`encode_frames` 硬编码 C=4；新三通道适配必须改新 stem/配置，不可直接传 C=3 |
| [DBSR 原始支持实验](../../../src/owac/frameworks/n01_dbsr_raw.py) | `RawDomainHistory`，半径 2、历史 8，空间旧缓存/重算对照 | 历史负结果固定；无同坐标旧时间来源逐源退出机制 |
| [匹配基线](../../../src/owac/evaluation/n01_burst.py) | `make_models`，同初始化与激活残差块的 `mean_capacity` | 复用容量控制方法；不能用未参加 forward 的参数凑数量 |
| [基线训练](../../../src/owac/evaluation/n01_burst_training.py)、[续训](../../../src/owac/runtime/n01_burst_budget.py) | 因果 batch、验证平台、优化器/scheduler/RNG 保存 | 可借训练骨架；生成器、窗口、指标和拆分需改为新协议 |
| [真实 loader](../../../src/owac/raw/rawvla_sequence.py) | `RawVLASequenceDataset`，真实数值、动作/状态统计、按相机窗口 | 默认只生成 train/validation；但 `split_manifest` 的 `sides` 已可承载四区，无需重写数值转换 |
| [特征桥](../../../src/owac/backends/rawvla_features.py) | `RawVLAFeaturePolicy`、`VisualTokenPostprocessor`、视觉塔替换审计 | 复用并给新模型重新验收。对象本身不是完整 `nn.Module` 容器，保存/优化参数时显式列各子模块 |
| [动作训练](../../../src/owac/training/n01_action.py) | 真实 flow loss、旧 N01/C3/C4 smoke、20 batch 级实际参数更新证据 | `build_bridge` 绑定旧 `N01Encoder`；训练总是冻结后端；checkpoint 不含完整随机流和 sampler。新正式训练器应另建 |
| [独立服务](../../../src/owac/runtime/n01_feature_service.py) | 相机状态、episode reset、normalized action、重复帧路径 | `main` 仍构建旧模型。新服务需模型工厂和独立快照，不只替换 checkpoint 文件 |
| [验收打包](../../../src/owac/runtime/n01_acceptance.py) | 源码逐文件 hash、tar、真实双平台 smoke | 只打包旧 `configs/encoders/n01/*.json`；新增全量配置、锁和依赖闭包到新的打包器 |

上述引用中的 `../../../` 均从本提示词目录回到当前 checkout。拟新增 `frameworks/n01_full`、`training/n01_full.py`、`evaluation/n01_full.py` 在核查时尚不存在。

## 现有结果的正确用法

[DBSR 完整实验](../../../research/experiments/2026-10-13_n01_dbsr_mechanism.md)记录三种子 9 组训练共 74,496 updates。成熟 DBSR 在原小合成任务有平均收益；空间 RAW 撤回重算相比旧缓存仅改善约 0.001 packed px，未建立 N01 独立增量。不能把这些 checkpoint 的平台状态转移给新训练任务。

[真实后端修复验收](../../../research/experiments/2026-10-13_n01_source_repair.md)提供旧 N01 的两平台 NPZ/动作梯度/服务证据。它证明已有桥可以复用，不能证明新的 DBSR + 支持域 + 状态版本也通过。历史 32×32 crop 和假定噪声方差只适用于当时 smoke。

## 已发现的时间差

主 checkout 当前已有[完整基线报告](../../../../../repo/research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)：LIBERO 5,820/10,000（58.20%），RoboTwin 508/3,250（15.63%），原缓存条件 108/632（17.09%，单列）。主清单覆盖完成与论文数值复现不同；原冻结标签不移动。

N01 的旧 README/AGENTS/status 和 v1.0 方案第 11.3 节仍含“RoboTwin 不完整”的历史状态。P0 引用新的完整报告与服务器 `records/pi05-complete-report-20261014-v1/` 的实际审计，保留旧条目的日期和历史身份，不重复启动 Default 全量。该主 checkout 报告可能尚未提交，应将选用的报告/小型清单按 hash 纳入 P0 快照，不能只凭 Git HEAD 识别。

## 当前工作树保护范围

开始本次提示词工作时，N01 已有未提交的 `docs/OWAC_academic_research_proposal_2026-10-09.md`；主 checkout 和其他 worktree 也有独立工作。提示词不授权覆盖、回退或全目录搬运这些修改。后续执行者以实时 `git status --short --branch` 为准。

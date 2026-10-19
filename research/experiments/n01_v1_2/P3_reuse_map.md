# N01 v1.2 P3 已有/待新增映射

日期：2026-10-15。checkout：`/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`，基点 `d3dec90`。
本文件是 P3 第一步要求的接口盘点：哪些接口真正可复用、限制是什么、哪些必须新增。它不是能力验收，也不表示 P3 已完成。

## 1. 已有位置与实际限制

| 已有位置 | 实际可继承内容 | 实测限制（必须处理） |
| --- | --- | --- |
| `frameworks/n01_full/adapters.py` | `DbsrAdapterConfig`（`rawvla_rgb3` 三通道 stem + 全尺寸 DBSR 64/9/512 + `MaskedDbsrFusion`）、`FullState`、`init_state/step/readout/reset/detach_state/state_bytes`、`encode_window`、`_masked_pool`、`build_postprocessor`（512→2048，256 token/相机） | ① `P3_SUPPORT_ENABLED=False`，`diagnostics["p3_support_applied"]` 恒为 False，`p3_support_status="disabled_pending_p4"`；② `DbsrAdapterConfig.__post_init__` 拒绝 `max_history>8`；③ `N01Config.validate()` 在 `require_unique_frames=True` 时拒绝 `max_history>8`；④ `encode_window` 输出只是内容融合，没有支持参与通路 |
| `frameworks/n01_full/ledger.py` | 来源身份 `SourceKey`/`FrameIdentity`、四态 validity、`without()` 删源、`with_content()` 替换、`retain_frames()`、`reset()`、`unique_frames()` | 逐帧 Python 对象账本；不能把整幅实时训练压到逐像素 Python 对象上。P3 用它做小规模独立数值参考与删源语义基准，训练路径用张量化等价物 |
| `frameworks/n01_full/domains.py` | `VersionedDomain`、`make/merge/split/retract/restore`、证据/上下文角色区分 | 同上：结构支持域的语义参考。P3 张量化实现必须保持"边退出≠源删除"的区分 |
| `frameworks/n01_full/statistics.py` | 局部常数/仿射 GLS、秩与可辨识掩码、插值系数核对、`NoiseModel`、`CoordinateTransform` | 单位换算（sensor/packed）与插值 taps 已审计；P3 支持构造必须复用它的单位约定，不能第二套 |
| `frameworks/n01_full/replay.py` | `EventStream`/`consume`、`full_replay`、`fixed_decision_replay`、`counterfactual_replay`、`_exact_window`、`ToyProgram`/`recompute_successor_closure` | 精确窗口当前按 8 帧校验；两条撤回语义（固定决策 vs 反事实）已分开，P3 必须沿用这个区分而不是混报 |
| `frameworks/n01_full/motion_builders.py` | `WindowMotionProvider` 与清除后 ledger 上的 correspondence/noise 重算 builder | 依赖 ledger 语义；P3 的 32 帧重算要接同一依赖顺序，不能只重算 flow |
| `backends/raw_correspondence.py` | 完整 SEA-RAFT RAW 适配、CFA 4/8 与三通道 3/6、`RawSensorFrame`、`RawNormalization`、`RawCorrespondenceMap.taps_for_query`、`CorrespondenceEstimator` | pull 方向/相位/packed 单位已审计；风险为未校准 proxy（`risk_kind` 字段已有），P3 必须保留该标注 |
| `baselines/burst_features.py` | `RawBurstBackbone`、`MaskedDbsrFusion`（真实 `[B,T,C,H,W]` 权重）、`align_features`、`BurstFeatureEncoder` | `forward` 返回融合结果与权重，但不接受支持输入；P3 需要把"支持改变有效来源"接进 `align_features`/`fusion` 之前的 valid 与 offsets |
| `raw/rawvla_sequence.py` | `RawVLASequenceDataset`、`MeasurementWindow`、`MeasurementWindowBatch`、`collate_sequence_samples`、`RawVLAProtocol`、`load_statistics`、`normalize_actions_and_state` | 默认路径推导与真实布局不同，必须显式传 `raw_cache_root`/`stats_path`/manifest；缓存 joint target 是缓存目标而非实测 proprioception |
| `raw/n01_full_manifest.py` | 四区清单（LIBERO 1228/175/206/162，RoboTwin 450/48/58/76，共 2403 源轨迹） | 任务模板跨区共享 → 不声称未见任务泛化 |
| `backends/rawvla_features.py` | `VisualTokenPostprocessor`（512→2048）、`VisualPrefixConfig`、`RawVLAFeaturePolicy`、`ActionChunkCache`、`ActionProtocol`、`read_frozen_protocol_record` | 是桥与 action chunk 协议的正规接口；`MiniVisualTokenPostprocessor` 只用于短小调试 |
| `runtime/n01_full_policy_preflight.py` | `contract` / `raw_adapter` 真实预检，已跑通两平台 forward/backward/update | 只是连通性预检，不是完整训练器；它硬读 `P3_SUPPORT_ENABLED`，P3 打开支持后该字段的语义要改 |
| `training/n01_action.py` | 原型训练器：小裁剪、20 步默认、旧工厂 | **不能**改个输出名当 v1.2 正式训练；只借鉴其 checkpoint/resume 写法 |
| `runtime/n01_feature_service.py`、`runtime/rawvla_full_eval.py` | HTTP 特征服务与冻结平台客户端参考 | 旧 factory/默认 policy 路径必须真正替换成完整 N01，不是改标签 |
| `runtime/baseline_dashboard_template.html` | 静态仪表盘模板，图表思想与配色 | 静态快照，不可宣称实时；前端专家只读参考、不修改 |

外部保留语义（不得另造平行接口）：`init_state(batch_size,camera_spec,device)`、`step(frame,embodiment=None,state=...,budget=...,offsets_xy=None)`、`readout(output,token_budget=None)`、`reset(state,batch_mask=None)`、`detach_state(state)`、`state_bytes(state)`。

## 2. 待新增（P3 实际要写的文件）

| 拟新增位置（前缀 `src/owac/`） | 职责 | 依赖 |
| --- | --- | --- |
| `raw/n01_paired_views.py` | 从同条件 RAW 经冻结作者 ISP 生成 `rendered_rgb3`；同源配对、显式表示标记、参数/dtype/resize/量化顺序记录 | 作者 Default ISP 真实源码入口 |
| `frameworks/n01_full/support.py` | 张量化空间域候选、局部 GLS/噪声/残差、沿轨迹 BOCPD 数值状态、接受/暂缓/退出四态 | `statistics.py` 单位约定、`ledger.py` 语义 |
| `frameworks/n01_full/model.py` | 完整模型工厂、支持参与内容通路、32 帧有界精确回放、四臂开关（Base/N01 × RAW/RGB） | `adapters.py`、`support.py`、`replay.py` |
| `frameworks/n01_full/adapters.py`（扩展） | 支持改变进入融合的有效来源/参考/归一化；32 帧窗口；`p3_support_applied` 真实反映 | `support.py` |
| `training/n01_v12.py` | 真实动作训练（π0.5 原生 action flow matching）、四臂匹配矩阵、checkpoint/resume、进度事件 | `model.py`、`rawvla_features.py`、`rawvla_sequence.py` |
| `evaluation/n01_v12.py` | 固定动作验证、真实来源干预、开发案例、汇总 | `model.py`、`replay.py` |
| `runtime/n01_v12_service.py` | 完整模型接既有 π0.5/平台服务；相机到达、reset、期限快照 | `rawvla_features.py`、`model.py` |
| `runtime/n01_v12_monitor.py` | 回环 HTTP 服务 `/`、`/api/status`、`/api/metrics`、artifacts | 标准库 HTTP |
| `runtime/n01_v12_monitor.html` | 前端专家所有（本执行者不写） | — |
| `research/theory/n01_raw_evidence_contract_v1_2.md` | 信息边界、G1/G2/G3 落到具体算子 | — |

配置新增 `configs/encoders/n01_full/v1_2/`。

## 3. 已确认必须修改的硬约束

1. `N01Config.validate()`：`require_unique_frames=True` 且 `max_history>8` 直接 raise。P3 要求每 episode/camera 最多 32 个唯一已到达帧，且"不能实际只看 8 帧却在配置写 32"。→ 必须把该上限提高到 32（保留"唯一帧"语义，不取消去重）。
2. `DbsrAdapterConfig.__post_init__`：`max_history>8` 直接 raise。同上。
3. `adapters.py` 的 `p3_support_status` 三处（`step` 诊断、preflight 两处）在支持打开后必须显示真实状态，而不是恒为 `disabled_pending_p4`。
4. `_window()` 返回的 `cell_mask` 是"packed cell 内所有共位样本都有效"的保守掩码；32 帧时该形状不变，但 `masked_fill` 的语义要保持"坏点值永不进入卷积特征，只由掩码记录"。

## 4. 不新增的东西（用户明确要求）

不新增无依据硬性通过线、不建严苛哈希准入、不自创总分、不加 Mamba、不做全局依赖图数据库/局部缓存失效框架/无限历史、不把文献清单变成串行关卡、不重跑旧 P2 三种子作为前置。

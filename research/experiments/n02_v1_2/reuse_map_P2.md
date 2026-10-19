# N02 v1.2 P2 复用 / 改造 / 停用表

日期：2026-10-15（Asia/Shanghai）。对象是 P0 已验收基础与 P1 实现，依据是
[P1 独立验收](../2026-10-15_n02_p1_independent_acceptance.md) 与对实际调用点的追踪。
本表是 P2 开始开发的闸门；只用它定去向，不另建审计框架。

## 1 去向总表

| 现有实现 | 位置 | 去向 | 依据 |
| --- | --- | --- | --- |
| ConvNeXt V2-Base 移植 `RawConvNeXtV2` + GRN/Block，`forward_multiscale` | `components/convnextv2.py` | **复用** | P0 逐字节 parity 已验收；是官方 Base，不重写 |
| FCMAE / mask-aware 稠密移植 | `components/fcmae.py` | **复用** | 掩码语义已核验；不得为省显存改成零填充 |
| SAVi `SlotAttention(+Masked)` / `Processor` / `TransformerPredictor` / `GRUCell` | `components/savi.py` | **复用** | P0 PyTorch 移植有逐参数对照 |
| SAVi spatial broadcast decoder | 上游 `savi/modules/decoders.py` | **新增移植**（`decoder_v12.py`） | 本仓**尚无**该模块，不是"复用已有" |
| Perceiver IO 核心注意力/latent/query 读出 | 上游 `perceiver/perceiver.py` | **新增移植**（`readout_v12.py`） | 本仓 `n02/readout.py` 是旧 token readout，**不是 Perceiver** |
| RAW 观测前端 `RawObservationEncoder`（RAW3 打包/矩形对齐/坐标膨胀/有效性） | `observation.py:800` | **复用 + 扩展** | 已有 stride8/16；v12 需补 stride4 级与统一 `d_model` 投影 |
| 条件 RAW 解码器 `ConditionalRawDecoder`（grid=8, ch=32, hidden=64） | `factors.py:1052` | **停用**（新主路径） | 小网格近似，被方案 3.1 要求替换为 SAVi 来源解码器 |
| 候选编码器 `CandidateEncoder`（4K+4 MLP） | `factors.py:1035` | **停用**（新主路径） | 输入含泄漏字段，见 §2 缺陷 E |
| 真值构造候选字典 `build_p1_task` | `factors.py:1863` | **仅留历史命令** | 从隐藏真值 `truth.world_px` 造候选，违反"不从真值生成候选列表" |
| 合成生成器 `synthetic.py`（2998 行） | `frameworks/n02_full/synthetic.py` | **仅留历史命令** | 禁止默认切回 synthetic；v1.2 主路径用真实官方数据 |
| `TaskWindowSource` 的 synthetic 分支 | `training/n02_full.py:3451` | **停用**（默认） | P1 训练与评测用同 generator/同 seed 1103/同 `partitions=("train",)`，实测 16/16 RAW SHA 相同 |
| `TaskWindowSource` 的 real 分支 | `training/n02_full.py:3451` | **复用**（经 `data_v12.py` 包装） | 真实 pseudo-RAW 窗口入口 |
| 锚点表 `AnchorTable` / `build_anchor_table`、分区 `build_partition_manifest` | `data.py:749/1075` | **复用（冻结）** | P0 已冻结 408075 锚点；不重新分区 |
| `RawVLAWindowDataset` / `RawVLASample` / 正式统计入口 | `raw/rawvla_sequence.py` | **复用（冻结）** | 单位边界见 §3 |
| `MeasurementScore` / `RawPrediction` / 传感器数值组件 | `sensor.py` | **复用** | 单位与 `log σ²` 归一化项语义保留 |
| 绝对检查 `CheckVerdict` + `WindowedCalibrator` | `checks.py:150/332` | **复用组件**，但**必须真实接线** | 组件有测试，P1 正式 step/eval 从未调用它 |
| `model.step_bundle` 的 `locked_checks` 路径 | `model.py:1077-1258` | **改造** | 存无 verdict 的 `MeasurementScore`，见 §2 缺陷 A |
| 旧训练 CLI `training/n02_full.py`（5118 行） | — | **仅留历史命令** | 不再往里塞新网络；新入口 `training/n02_v12.py` |
| 旧评测 CLI `evaluation/n02_full/p1.py` + `_model_step`/`_verdicts` | `p1.py:678/613` | **停用为默认**，保留原封存结果 | 读 `.verdict`（实为 `.status`）、`"insufficient" in verdict` 混淆三态、先验不进推理 |
| 旧 `n02/readout.py` | `frameworks/n02/readout.py` | **停用** | 不是 Perceiver，不作正式读出 |
| 策略桥 `RawVLAFeaturePolicy` / `EncoderReadoutTokenSource` / `ActionProtocol` / `ActionChunkCache` | `backends/rawvla_features.py` | **复用（冻结协议）** | 只做接线，不改动作协议 |
| dNRI / Theseus / SMC 参考实现 | `components/{dnri,theseus_ref,smc_ref}.py` | **保留参考** | 首版不强制进主链 |
| `rawvla_features.single_batch` 设备与求导修复 | `backends/rawvla_features.py` | **保留** | 部署期修复；该 CLI 仍是随机特征后端探针，不能替代真实 RAW 训练 |

## 2 P1 六类错误：实际调用点与 P2 修法

| # | P1 错误 | 追踪到的真实调用点 | P2 修法 | 负责文件 |
| --- | --- | --- | --- | --- |
| A | `locked_checks` 存无 verdict 的 `MeasurementScore`；评测读 `.verdict`（实际字段是 `.status`），全部变成字符串 `None`，被计成 0 误报 / 0 检出；`WindowedCalibrator` 正式链路从未被调用 | 写入 `model.py:1092`、`1229`；读取 `evaluation/n02_full/p1.py:613 _verdicts`、`1128/1141`（`"insufficient" in verdict` 混淆信息不足与模型不足） | 统一真实返回类型 `AbsoluteCheckResult(status, valid, statistics, calibration_version)`；缺失 verdict 显式失败或标无效，**单独计数**；三态分开判定；真实调用 `WindowedCalibrator` | `checks_v12.py`（新建）+ `model.py` 接线 |
| B | `declared_prior` 只写进 `task.inference.declared_prior`，从未进入 `log_p0`；同一测量下两个合法先验输出逐位相同；评测还把真候选先验设为 0 使总质量 < 1 | `evaluation/n02_full/p1.py:678 _model_step` 向 `step_bundle()` 传的是 `bundle`（不含先验）+ 空 `SequenceState()` | 先验真实进入 `log_p0` 并参与 `q=softmax(log_p0-E/tau)`；四候选先验归一化；用同一样本的中性/偏置先验观察输出差异 | `inference_v12.py`（新建） |
| C | cue swap 只重排 `correct_index`，没有交换测量；曝光/噪声/shift 组只换字符串标签；no_information 只是对普通序列首帧打分 | P1 评测循环按组名切换标签后跑同一任务 | 干预在**真实 RAW 像素/历史**上发生并记录位置与强度；构造真实无信息序列；每组必须真正改变测量 | `interventions_v12.py`（新建） |
| D | 训练与评测 synthetic 同 seed 同 partition，实测前 16/16 场景 RAW SHA 相同 | `training/n02_full.py:3451 TaskWindowSource`（`kind="synthetic"`）与评测共用生成器 | 主实验切官方数据 P0 冻结分区；train/dev/calibration 严格隔离并实测无重叠 | `data_v12.py`（新建） |
| E | 错误候选带 `log_exposure=0.2*(index+1)`、正确候选默认 0，该数值进入候选编码器 → 32/32 字典捷径 | 构造 `factors.py:1904`；消费 `ObservationModel._build_world_vector()` | 去掉与真伪确定相关的数值捷径；提议从观测特征/可学习 latent/因果前态产生 | `data_v12.py` + `inference_v12.py`；`build_p1_task` 降级为历史 |
| F | stride8 的 64×64 稀疏查询只看 128/8192 标量且固定 CFA 奇偶位，5/8 探针在全传感器上有 ~1051—3469 DN 差异而 query 上无差异 | `model.py:305 _query_indices` | 保留完整有效区域，不用单一采样相位；报告实际采样点数与全传感器均值关系 | `data_v12.py` / `decoder_v12.py` |

## 3 不可动摇的单位与协议边界

- `RawVLAWindowDataset` 使用正式统计后，sample 的 state/action **已经归一化**；
  `prepare_control_targets()` 只补齐 32 维，`control_actions()` 只切回 7/14 维。
  **两者都不做归一化或逆归一化**。训练不重复归一化；在线采样输出在送 `take_action`
  前按各平台原统计**逆归一化一次**，并沿作者夹爪规则执行。名为 `control_actions`
  的归一化切片不是物理控制量。
- RoboTwin：三相机，float32 原尺度（不除 255、不逐帧 max）；`actions[t]` 为当前状态；
  offset1/horizon50；在线 14 维关节（左6+左夹爪+右6+右夹爪）；checkpoint 明确
  **MEAN_STD**，不做默认 min/max 或重排；原 NPZ 16 维 `robot_states` 不是在线 14 维关节状态。
- LIBERO：两相机；RAW uint8→FP32/255；8 维本体；7 维动作；offset0/horizon10/每5步重规划；
  **禁止 92 维仿真 states 进入策略**。
- LIBERO 是 RGB8 逆处理的三通道 pseudo-RAW，RoboTwin 是 HDR-derived 三通道 float32 测量，
  **都不是原生 Bayer**。受控 RGB 必须由**该次同一 RAW 经固定 ISP** 得到；不得把正常光配对
  RGB 当作退化 RAW 的同源 RGB。
- 未来专家动作只作训练标签；模拟器隐藏状态、深度、真值框不进入推断，也不用于构造候选。

# N04 v1.0 首版实现合同

本合同是为 DeepSeek 固定的首版工程设计。它细化原[方案第 5–12 节](../../plans/N04_full_development_v1.0.md)，形状和默认值是拟实施选择，当前源码尚未具备。后续证据推翻选择时，由审阅者形成下一版任务；不要在一次实验内同时改观测、状态、更新与指标。

## 模块分工和状态

新正式实现保留在 `frameworks/n04/`，新增 `full_config.py`、`contracts.py`、`measurement.py`、`state_estimator.py`、`measurement_geometry.py`、`uncertainty.py`、`revision.py`、`full_model.py`，配置使用 `configs/n04/full/`。旧 `config.py/model.py` 及旧 checkpoint 继续表示原型；正式 model_kind 固定 `n04_full_v1`，checkpoint schema 使用独立字符串 `n04-full-v1`。不靠扩大旧 width 字段冒充新结构。

按相机处理一份 `[B,C,H,W]` 当前测量；RAW 两模式分别使用 3 通道与 CFA 四平面 stem。ConvNeXt-B 输出四级空间场，经 1×1 lateral + 双线性上采样形成 256 通道测量场；上采样只作用于特征。当前帧测量函数没有历史参数。几何头输出当前深度/三维位置及同单位协方差，潜在头输出 `z[ B,N,256 ]`、`R_z[ B,N,256 ]`，两种协方差不互相替代。最终稳定 ID 的锚点是状态层关联结果，测量层先输出稠密场与独立候选，不能把历史位置作为单帧测量网络输入。

单相机首版最多 N=128 个有效锚点；候选按空间 8×16 分区选有效支持，必要时每区取最大几何置信度点，置信度与 GT 标签无关。输入分辨率按原数据保持并 pad 到骨干步长，mask 清除 padding。出生锚点采用单调 ID；空位与被遮挡旧锚点不互换身份。关联先用投影位置门控与当前/历史描述子双向最近邻；开发集冻结距离、深度与比值阈值，歧义保留 unmatched。固定相机、正确关联的诊断版本先运行，再单独启用实际关联。

潜在状态 `m[B,N,512]`，结构化协方差 `P_u/P_l/P_s[B,N,256]`。RKN 参数跨锚点共享，锚点展平为 `B*N` 调用，独立近似明确披露。初版作者默认 15 个转移基、带宽 3、系数网络 [64,64]，不沿用二维诊断的 5 个基/带宽 0。物理位置由共享读出解码监督；潜在后半部分不命名为真实速度。

S3 增加显式几何载体 `x_g=[p_W(3),v_W(3)]` 与 `[6,6]` covariance，用于三维投影、误差解释与 S5 动态修订。潜在状态与几何载体是同一观测的两种表征，不能把它们的后验当独立传感器相乘。S5 的模式证据首版只在有单位的几何测量空间计算；RAW 支持和协方差限定可用方向。几何到潜在状态的桥接是学习模块，单独消融，不据此声称严格联合 Bayesian 滤波。

元数据至少包括 episode/camera/anchor ID、measurement ID、最后真实观测时间、当前状态时间、缺失时长、association confidence、支持、reset reason。首版单相机通过后，多个相机先独立维护状态并拼接视觉前缀，不直接融合为独立证据；跨视角联合估计为单独实验。

## 因果 API

```python
# 拟新增接口；并非当前代码已经存在。
def measure(frame: MeasurementFrame) -> DenseMeasurement: ...
def predict_before_observe(
    state, executed_interval, proprio_history, calibration, target_time
) -> Prior: ...
def associate(prior, dense_measurement) -> AnchorMeasurement: ...
def correct_after_observe(prior, measurement, error_model) -> Posterior: ...
def readout(posterior, token_budget=256) -> VisualTokens: ...
```

这些接口由同一个 `N04FullEncoder.step` 编排，训练和在线共享。`predict` 的上下文不包含当前图像、当前图像光流或未来目标。当前已到达的本体可用，但记录 availability_time。`measure` 不读状态；`associate` 可以读 prior，不能把关联选择的偏差隐去。重复 measurement ID 返回缓存且不再次更新；时间前进而没有新观测时仍应传播。reset 以 batch/episode/camera 为边界清除缓存、锚点与时间，而非仅清均值。

作者 cell.forward 是“t 时刻校正，生成 t+1 prior”。在新 wrapper 中分开调用 update 和 predict 的薄适配器，并用完整 forward 做数值对照；每个物理间隔只传播一次。所有无效观测先用 `where(valid, value, finite_placeholder)` 清理，再调用网络/更新，避免 NaN×0 和掩码分支梯度污染。

隐状态/物理 covariance 采用 FP32；数值参考和有限差分用 FP64。RKN 结构需满足每个 2×2 块的 `P_u>=0, P_l>=0, P_u*P_l-P_s²>=0`；不仅是对角正。几何更新用 Cholesky solve 与 Joseph，模式混合包括分量均值离散项。上述为数值/局部模型合同，不是任意数据上的精确不确定性保证。

## 正式训练设置

初始正式预算继承 v1.0：S1/S2/S3 100k 起、上限 300k；S4 每小项 30k 起、上限 100k；S6 100k–200k；S7 每平台/系统/seed 100k 起。先 profile 完整形状，再使用梯度累积。正式种子首版 `[17,29,43]`，用于完整网络初始化、数据顺序和 dropout；训练种子与场景随机种子分别记录。某阶段固定一个测量 cache 的多个 RKN seed，只能称状态模块重复，S6 才重做三个完整 pipeline。

优化器 AdamW，新模块开发学习率候选 `{1e-4,3e-4,1e-3}`，默认 3e-4、weight decay 0.05、grad clip 1、warmup 5%、cosine decay，no-decay 组含 norm/bias。方差和初值参数单列 no-decay 组。BF16 只在验证后用于骨干/读出，几何、噪声和协方差保持 FP32。默认不用 EMA，便于 checkpoint/消融身份一致。正式评价每 5k 步一次，连续五次主开发指标相对改善 <1% 才记趋稳，同时检查困难组与过拟合；预算耗尽仍改善则按预注册 ×2 扩展一档或记训练未充分。

S1 有效 batch 128 帧；S2/S3 有效 batch 16 序列，每段 16 帧 burn-in +32 帧损失，TBPTT=16。burn-in 用历史更新状态但不计训练损失，训练块间 detach，绝不跨 episode 携带状态。64/128 帧跨度单列实验。损失按有效样本/维度归一，物理单位用训练集尺度标准化；每项原单位误差同时报告。有限训练 loss 不是唯一验收标准。

默认学习率、种子、MLP宽度、IMM模式先验和具体读出器是本次提示词补充的首版工程选择，不是已测最优值，也不自动属于论文原实现。记录在配置中，开发集只在给定的小网格和预算内选择；不能让 DeepSeek 自行换整条架构。需要更改这些结构时，交付单因素证据和明确差异供下一轮设计使用。

## 阶段交接

每阶段新增 `research/experiments/<date>_n04_v1_<stage>.md` 和相应小结果目录，`docs/status.md` 只追加事实。`handoff.json` 至少包含：

```json
{
  "stage": "S0",
  "implementation_status": "complete|partial",
  "experiment_status": "complete|running|blocked_by_data|not_started",
  "claim_status": "supported|not_supported|inconclusive|not_applicable",
  "source_manifest": "path and sha256",
  "data_manifest": "path and sha256",
  "split_manifest": "path and sha256",
  "configs": [], "checkpoints": [], "commands": [],
  "metrics": {}, "failed_groups": [], "resources": {},
  "next_stage_inputs": {}, "remaining_work": []
}
```

上述 `a|b` 是允许值说明，实际文件选择一个值。报告明确源码、配置、数据、权重 hash，保留全部失败与困难组。未经过数据审计的批次不进入正式结果；不得把 fixture、接口、充分训练、机制、闭环五层证据合成一个 PASS。阶段完成后先交接，再按下一条阶段提示词执行。

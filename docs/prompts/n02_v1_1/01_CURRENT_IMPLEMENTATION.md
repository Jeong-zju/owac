# 提示词编写前的实现审计

核查日期2026-10-13；N02 HEAD `843be674c35e8e058a364154e1c27a4b26ded2e4`。核查包括当前未提交v1.1方案与边界诊断，实际代码、结果文件及定点本地回归。本文是本地源码/资产核查，没有连接远端确认进程状态，也没有重训历史模型。

## 1. 仓库实际位置与变更

主checkout `/home/jeong/zeno/owac/repo`，HEAD `16075fca7c5e5cd126900bc62531a5733fe4e0be`，分支`lzh/pi05-sim-baseline`；存在大量与基线、历史路线及研究文档相关的未提交修改。N01/N04也有各自历史，本次不合并其实现。

N02 checkout已有未提交的README/status/候选卡、`docs/plans/`、反证边界实验及其脚本/结果。这些是本次依据，不需要先清空或强制提交。继续此checkout，新增`n02_full`。v1.1约618行的方案不是已经实现的代码。

## 2. 代码复用与缺口

| 实际文件/符号 | 已有内容 | 新阶段处理 |
| --- | --- | --- |
| [encoder_contract.py](../../../src/owac/core/encoder_contract.py)：MeasurementFrame/EmbodimentContext/VisualTokens | RAW与可用上下文、外部token契约 | 复用类型；新增bundle层协调多相机 |
| [rawvla_sequence.py](../../../src/owac/raw/rawvla_sequence.py)：RawVLAWindowDataset/normalise_actions/normalise_state | 原始数组、因果窗口、动作mask、CFA坐标与转换 | 复用数值；增加显式分区/锚点消费，禁止重新随机划分 |
| 同文件compute_split/_acquisition_source | 旧版train/validation；分组外键为LIBERO suite或RoboTwin task | P0新四分区需要源轨迹/场景来源盘点，不能直接改名称假装完成 |
| 同文件FORMAT_SPECS | LIBERO offset0，RoboTwin offset1；后者state_source_key=actions | 固定并审计真实字段；RoboTwin默认normalisation并非π0.5正式zscore配置，须显式传入 |
| [rawvla_features.py](../../../src/owac/backends/rawvla_features.py)：RawVLAFeaturePolicy | 跳过图像塔、训练flow loss、一次prefix多步denoise、冻结后端 | 复用；新版本化state/cache由adapter补齐 |
| 同文件EncoderReadoutTokenSource | 遍历每个相机，独立encoder状态和readout，再拼接 | E1可以作对照；正式对象路径新增WorldTokenSource，避免各视角互斥MAP拼接 |
| 同文件ActionChunkCache | episode/step缓存与reset | 新revision/model版本尚未纳入缓存key，须扩展或在新adapter管理 |
| [training/n02_raw.py](../../../src/owac/training/n02_raw.py) | 为smoke挑选train/validation各少量NPZ，单样本时间循环与真实π0.5 | 只作调用参考；正式batch、数据遍历、DDP与训练续跑另建n02_full入口 |
| [frameworks/n02/state.py](../../../src/owac/frameworks/n02/state.py)：MeasurementBuffer | 去重与bounded history、部分batch reset | 借鉴身份和回归，不直接当完整来源DAG；新账本需校准版本、回放与多相机身份 |
| [frameworks/n02/model.py](../../../src/owac/frameworks/n02/model.py)及readout/revision | 固定网格假设、小卷积/共享cell、旧局部反证调度 | 历史对照；不是Base/SAVi/Theseus/结构发现/ATMS实现 |
| [baselines/n02_measurement.py](../../../src/owac/baselines/n02_measurement.py) | 重训sbi CNN加首帧RAW差分 | 留作旧测量回归；不能把差分当独立帧似然 |

代码核查时不存在`src/owac/frameworks/n02_full/`。本地根`pyproject.toml`要求Python≥3.11，encoder extra的torch来自CPU index；旧真实StarVLA运行时为Python3.10，历史通过PYTHONPATH接入。P0必须建立隔离的CUDA兼容环境，而不是将根CPU锁当GPU部署配方。

## 3. 真正已有的证据

旧[真实接入记录](../../../research/experiments/2026-10-12_n02_encoder.md)证明当时版本的NPZ→旧N02→π0.5离线训练/推理连通与梯度，不是新Base或结构机制验收。

[重训评分器](../../../research/experiments/2026-10-13_n02_retrained.md)在声明合成条件下修复了部分测量能力；[局部修订对比](../../../research/experiments/2026-10-13_n02_revision_comparison.md)中，旧N02与简单事件缓存逐帧相同且约慢4%。[0014](../../decisions/0014-n02-scheduler-archive.md)归档该调度；v1.1允许新完整路线，不撤销旧负结果。

[新反证边界](../../../research/experiments/2026-10-13_n02_refutation_boundary.md)显示：两个分离亮点未被绝对拟合拒绝，2像素平移后方向正确数为8/11/8（各16）。所以P1要做独立预测、局部统计与成像干扰，不能照抄旧全图均值残差。

主checkout比本worktree多一项新归档：2026-10-13 16:42:30北京时间，RoboTwin原条件632条核查补齐，108成功/524正常失败，17.09%；见主checkout的`research/experiments/2026-10-13_rawvla_robotwin_infrastructure_repair.md`。它独立于3250条主分母。LIBERO 5820/10000和远端RUNNING字样均来自归档快照；本轮不将其当实时状态。

本次本地运行5个相关测试文件：**87 passed、4 skipped**；4项因本机缺真实NPZ或StarVLA/权重跳过。它验证已有数据/桥/状态/历史对照回归，不验证v1.1。命令和完整收据见[preparation_validation.json](preparation_validation.json)。

## 4. 上游源码已查到的具体移植问题

1. **FCMAE硬编码。** `models/fcmae.py`的patchify/unpatchify虽接收in_chans配置，内部仍写死3通道，mask还假设方形网格。CFA4与240×320输入必须改这两类假设，做矩形和多通道往返测试。稀疏encoder依赖MinkowskiEngine，普通dense零填充不等价。
2. **SAVi的mask。** `SlotAttention.__call__`丢弃padding_mask，标准注意力分支也存在未实现mask的路径。RAW坏点/缺失相机不能仅往函数传mask；移植要真正屏蔽有效key，保留slot维softmax再按有效输入归一化。
3. **GRU不只换名字。** Flax/PyTorch的参数布局、gate顺序和reset位置要逐项对应；先原参数/输入前向及梯度一致，再扩宽为512。
4. **dNRI时间泄漏。** `dnri/models/dnri.py`的DNRI_Encoder训练posterior使用forward_rnn和reverse_rnn；在线必须使用forward prior/single_step_forward。替换未来帧的测试必须不改变过去在线输出。
5. **GRN的全局性。** Base的GRN具有空间归一化依赖，不能把编码器写成“每块独立”；P5要计整帧编码开销和真实依赖。
6. **LM适用目标。** Theseus优化的是残差平方和；完整RAW目标含log方差/尾概率等。求解器给提议，外层完整目标接受，不把NLL平方。

固定源码与SHA见[参考导航](02_REFERENCES.md)和[source_manifest.json](source_manifest.json)。本次取回源码不意味着已完成上游运行复现。旧来源JSON的ATMS DOI数组被括号截断，本包引用完整转义URL，不沿用截断URL。

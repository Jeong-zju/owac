# S6 组合训练与充分消融开发提示词

## 执行上下文

你负责实际编程与实验。这份提示词已经指定研究路线，请按顺序完成当前阶段的代码、验证、训练或数据审计和交接，不停留在计划、空接口或伪代码。当前阶段之外的模块只保留必要接口。遇到数据事实与假设不符，先完成不依赖该条件的工作，再写出具体缺口、已检查来源与可执行补齐方案，不替换成虚构输入。

本地 checkout：`/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`；分支 `lzh/n04-egomotion-separation`。开始读 `AGENTS.md`、`README.md`、`docs/status.md`、[全量方案](../../plans/N04_full_development_v1.0.md)、[首版实现合同](implementation_contract.md)、[源码核查](preparation_audit.md)。先检查工作树，保留既有修改和 FilterPy stash。下面标“拟新增”的文件/API/CLI 是你的开发目标，不能当成已经存在。

**实际数据核查、正式训练和真实 π0.5 验证优先在用户指定远端执行：**

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

先读[远端运行规约](remote_execution.md)。数据根为 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`，LIBERO 在 `libero/raw`，RoboTwin 在 `robotwin2/raw`。新源码和产物放拟创建的 `/workspace/owac/research/n04-v1/`，冻结 RawVLA 数据/源码作为只读参考。纯 encoder 使用 S0 固定的独立 Python 3.11+ 环境；后端继续验证 `/workspace/owac/rawvla/envs/starvla/bin/python`（实测 3.10.21 / torch 2.7.1+cu128 / numpy 1.26.4）。本地 Python >=3.11 / NumPy >=2.3 的 uv 环境不能直接覆盖它。正式代码经包含未提交文件的 hash 快照同步，远端复核 hash 后运行，不能只同步 HEAD。

实际 GPU 是单张 96 GB 级 RTX PRO 6000；上次约 32 GB 被已有服务占用。先查 `nvidia-smi`、`supervisorctl status` 和磁盘，按剩余资源 profile，不自行停止其他任务。正式长任务用 N04 专属 supervisor program、绝对解释器与不可变源码快照；日志/checkpoint 留远端，小报告/hash/曲线回本地。

模型主线固定为完整 ConvNeXt-B、作者 RKN/Ac-RKN、显式几何与误差传播。旧 tiny N04 留作历史。输入模式 `rawvla_rgb3` 与 `bayer_cfa` 分开；主链没有 RGB/sRGB/XYZ 中间图、RGB 教师或 RGB 重建。实际执行动作、当前本体与未来动作监督各有独立字段。正确实现和负实验结果同样交付；不预设新方法必须胜出。

## 目标与前置

把已独立验证的测量、动态状态、动作/几何、误差和修订组合训练，回答新增机制在复杂组合中是否仍有作用。S0–S5每项具体权重/config/数据身份是输入。进入本阶段先整理supported/not_supported/inconclusive表；工程存在但机制无收益的部件不自动进入主配置。

S6必须训练三个完整pipeline种子 `[17,29,43]`，包括单帧骨干及后续状态；不能复用同一个视觉checkpoint只换RKN种子后称“三个完整模型”。每个种子内各对照共享匹配初始化与相同可用信息。

## 实现文件和配置

扩展 `training/n04_full.py`、`evaluation/n04_full.py`；拟新增 `evaluation/n04_statistics.py`、`training/n04_matrix.py`，配置 `configs/n04/full/s6_matrix.json` 与各system配置。矩阵调度放包内，可用脚本启动，不为每个方法复制训练代码。

`MatrixEntry`记录system_id、seed、parent/init checkpoint、trainable参数组、输入模式、action/geometry/noise/revision开关、数据/split/cache hash、步数/batch/学习率选择预算、后端模式、run目录。runner首先生成完整矩阵和预计成本，再按资源顺序运行，状态由原始run结果聚合。

## 必须包含的比较

| 系统 | 与完整方法的明确差异 |
| --- | --- |
| 无记忆 | 同骨干当前帧读出，缺失hold-last |
| 原版RKN | 同测量和空间支持，不加入已执行控制 |
| Ac-RKN | 动作控制和同样状态预算，不含新增误差修订 |
| Ac-RKN+geometry+calibrated R | 保留成熟几何与可信误差更新，不含新增动态解释约束 |
| Ac-RKN+adaptive Q / robust update | 与S5相同的误差证据，使用成熟自适应Q或鲁棒更新 |
| 标准IMM | 相同三模式预算，去掉新增RAW支持/nuisance方向约束 |
| 完整N04 | 只组合已确认的部件 |

另做匹配重训消融：无动作、无几何、无uncertainty传播、无事件修订、无geometry→latent桥接。容量能匹配的地方用同容量替代，不强求hold baseline有虚假参数；总参数/FLOPs/GPU小时分列。所有方法使用相同最大历史、已执行动作可用性、传感器信息、读出容量、数据访问与调参次数。

如果S5启用正式motion网络，增加MCDS-VSS式ego+residual-flow+feature-fusion结构参照，和“所有残差都做motion compensation”的对照。MCDS公开仓库要求联系作者取代码，当前不具备可直接复制的完整实现；按论文实现时命名paper-inspired reimplementation，不能标官方复现，不擅自给作者发邮件。HiP-RSSM/MTS3只在跨动力学/时间尺度瓶颈被证实后做扩展，不默认加进主系统。

## 训练次序与预算

先用冻结S1和各模块重放S0单因素cases，确认组合没有改变单位、时序或支持；然后用S1对应种子的初始化/权重，逐模块开放state/geometry/revision，最后全encoder端到端。三种子的完整阶段历史记录在matrix，S1预训练成本与S6追加成本都计入，不把额外预训练免费化。

初始S6 100k，最大200k，effective batch16 sequences、16 burn-in+32训练，学习率和loss默认沿前阶段，端到端骨干学习率默认为新增头的0.1倍并给所有对照相同选择机会。loss项只有相应模块存在时启用，记录有效样本分母；geometry/predict/revision/event/calibration分别记曲线，不只总loss。

采样按task/source均衡，首版factor strata比例：40%单因素、40%两因素、20%全组合；无效/不可辨识例也按预注册比例保留。固定该比例后再训练，报告实际采到数量。留出未见组合、对象、场景、速度/加速度、相机位姿、噪声参数、错误/过期历史；测试源身份与训练无交集。首次到达预算上限仍改善，按共享合同扩档，不用欠训练结果草率否定机制。

资源profile覆盖正式单相机和多相机、三模式、长序列与完整梯度。单卡上顺序训练不同seed；microbatch/累积/checkpointing可以调整，effective batch、模型规模与公平预算记录不变。实际资源不足就报告剩余矩阵与已完成run，不填模拟结果。

## 两类实验分开

冻结干预使用同一checkpoint，切断控制、几何、误差或revision→next-predict，测实际依赖；匹配重训消融从对应种子的统一起点按同预算重训，测替代方案能力。两类结果不得混排成同一baseline表。

归因H1/H2分别用最强相关baseline。对照更好时保留，查观测/关联/优化/分布/预算，报告“未支持/证据不足”而不调最终阈值。新增模块若只增加算力，则补等成本对照与充分调优能力表，不能归因于规则。

## 统计和完整指标

保存每source trajectory、condition、train seed的raw rows，先按轨迹聚合，再按task分层配对bootstrap，首版10000重采样、固定statistics seed=20261014；双方使用相同source身份。光照变体、相邻帧、像素不是独立样本。训练seed不混同environment seed；分别给每种子结果、种子均值、配对差与95%区间。只有3训练种子时对训练随机性的区间本身可能宽，不能用海量帧掩盖不确定性。

主要指标沿S0：prior/posterior geometry、缺失面积与时长桶、停止/反向积分误差和censored修订延迟、固定FPR事件召回、unknown覆盖、NLL/coverage/width。物理状态NEES只在有真实物理标签/同单位covariance的分支使用，latent不硬套。所有困难组单列，平均值不掩盖关键组失败。

成本报encoder/association/predict/correct/readout分项、online batch1同步CUDA p50/p95、warmup、峰值/状态显存、参数、吞吐、总GPU小时、训练FLOPs估计及其估计方法。多相机共享encoder调用与每新观测更新次数按日志重算。

## 可执行出口

实现 `python -m owac.training.n04_matrix --config configs/n04/full/s6_matrix.json --mode plan|run|summarize`（实际命令选一个mode），plan生成待执行矩阵，run执行/从已记录checkpoint恢复，summarize只读结果不启动训练。其子任务仍走同一个n04_full runner。实现统计聚合CLI并核对手算小表、重采样分组和缺失结果状态。

交付三完整种子所有主矩阵、配对统计、成本、失败样本与公平性表；若未完成，明确缺项。S7主对照至少无记忆/Ac-RKN/完整N04，若H1/H2依赖强误差baseline或IMM差异，这些也进入S7主要任务比较。最终配置由开发集固定，final只用于冻结后评价。

## 重点参考

[既有分阶段实验](../../../research/experiments/2026-10-13_n04_staged_rebuild.md)、[同测量对照](../../../research/experiments/2026-10-13_n04_matched_tracking.md)、[MCDS-VSS论文](https://arxiv.org/abs/2405.19921)、[MCDS-VSS实际代码入口状态](https://github.com/angelvillar96/MCDS-VSS)、[HiP-RSSM](https://github.com/ALRhub/HiP-RSSM)、[MTS3](https://github.com/ALRhub/MTS3)。扩展方法的许可/代码完整度在正式复用前核查，默认主线不以获取这些扩展为前置。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

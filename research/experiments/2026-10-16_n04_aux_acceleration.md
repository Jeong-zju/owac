# N04 辅助损失展示与加速实测

2026-10-16 后续执行结果：[v4.2辅助目标修复](2026-10-16_n04_aux_repair_v4.md)已替代本页的暂停状态，双平台在独立fresh run启动；表征外权重现为0.1，不能按下方旧0.01解释。本页保留v3排查、性能试验及负结果的原始范围。

2026-10-16，用户要求显示预测、表征辅助项，检查加速、数据预加载和辅助项退化。**最新状态：两平台训练已暂停，LIBERO保留step70、RoboTwin保留step45；辅助目标尚未修复，不能宣称训练健康或恢复作业。** 监控与两类辅助曲线已上线。新增两种诊断融合方案没有显示足够收益，均未用于训练；保留原r3源码、loss、容量及有效batch8。以下“恢复／继续”是辅助项排查前的阶段记录。

## 监控交付

入口 <http://127.0.0.1:8090/>。两平台动作曲线下面新增“预测辅助损失”“表征辅助损失”，各自显示原始数值、独立刻度曲线、最新步数及总损失权重0.1 / 0.01。来源为 metrics.jsonl 的 terms.forecast / terms.representation；API原来已有两个series，原页面漏画，非训练漏算。训练状态旁增加从实际config.used.json读取的“编译加速已开启”。性能验证期间明确标出验证状态，不伪装成持续优化。

辅助项来自变化训练窗口，不是固定开发指标或任务成功率。这两项当前没有明显下降；动作误差变平也不能据此认定整体收敛。预测项为完整双模式NLL，不能把方差变化单独解释成更好的预测能力；表征项约束同一query跨观测差异，不以训练下降代替闭环验证。

## 受控性能实验

保持n04-v1.1-s2-repair-v3、同一loss定义、seed101、官方RawVLA-Bench、48帧（16burn-in+32gradient）、4动作点、forecast1/2/4、完整网络容量。RoboTwin在保存第40步后暂停，检查点和所有已完成更新保留；未完成下一更新的4/8窗口未写权重，恢复后从40重算。LIBERO全程继续。每个probe都硬禁optimizer、scheduler.step及权重写出。

原方案逐锚点将10个诊断标量取回CPU，且使用动态boolean indexing。r4尝试固定形状掩码归约，将诊断向量保留GPU，窗口末尾一次传回；r5再编译纯诊断数学。每次都保留所有预测loss、有限值/R正值检查、仪表采样及NaN空支持语义。r4/r5均为独立实验快照，不热改r3运行源码。

同一真实开发窗口、同一模型、每轮相同Torch随机种子，对照交替运行；排除初次编译预热。固定训练步40，避免对应损失排程漂移。窗口实际有效计数每轮均为：对应32、预测267、动作4、表征4。每轮四项loss逐值相同。

| 方案 | 原方案热运行 | 候选热运行 | 决定 |
| --- | --- | --- | --- |
| r4 延迟传回、eager归约 | 28.097 / 26.880 / 28.216 / 27.831 s，均值27.756 | 28.388 / 29.358 s，均值28.873 | 慢约4%，不启用 |
| r5 再编译诊断数学 | 27.898 / 27.699 s，均值27.798 | 27.502 / 27.083 s，均值27.293 | 用时低约1.82%，少量重复不足以确认稳定实用收益，不启用 |

这是完整48帧前反向的时间，不是8窗口optimizer更新速度；不将GPU瞬时利用率、显存占用或冷启动时间充作提速证据。原始comparison.json和日志在远端 runs/v1.1/codex-profile/diagnostics-r4、diagnostics-r5 及 control-history/；本地原始副本和候选源码在被忽略的 outputs/n04-aux-acceleration-20261016/。受版本管理的小回执见 results/2026-10-16_n04_aux_acceleration/。

## 数值检查与恢复

初版比较曾要求完整CUDA梯度抽样逐位相同而中止。追加原方案对原方案控制后发现，其69124个梯度抽样的相对L2波动约0.075%—0.101%，r4为0.083%—0.088%，r5为0.072%—0.096%；四项loss和有效计数仍逐值相同。没有用放宽阈值伪造“CUDA梯度逐位等价”。独立CPU/CUDA小型控制确认预测损失及其梯度逐值相同，编译诊断在部分/空mask下与eager参考对齐；这些验证并不代表性能收益成立。失败的严格比较日志保留为 benchmark-diagnostics-r4-warmup-reference-rejected.log / strict-gradient-rejected.log。

候选本地测试224 passed、1 skipped（缺websockets），make check通过。因为性能未获采用，候选代码/配置/专用测试全部从活动checkout撤回，仅保留实验快照与证据；撤回前逐项diff确认两核心文件只包含本轮改动，未回退既有修补。活动核心文件重新与原冻结r3逐字节一致；最终保留源测试219 passed、1 skipped，监控13项通过，make check通过。

RoboTwin恢复仍使用 n04-v11-codex-compiled-r3-20261016 和 codex_compiled_raft_robotwin.json，通过checkpoint-step0000040.pt恢复模型、optimizer、scheduler、RNG；未启用r4/r5或迁移历史v2权重。监督配置和恢复决定留在run下 diagnostic-acceleration-decision.json。最后运行状态以 runtime.json 和监控实时值为准。尚未实现充分利用PRO 6000，跨序列合批与外层调度融合仍需后续实现与验证；未进入S3或新闭环。

## 数据加载实测

同一官方训练计划，实际读取8个48帧窗口并同时持有内存。LIBERO总读取0.527秒、0.563 GiB；RoboTwin总读取1.517秒、0.990 GiB。即使完全没有预取，也仅约占170／220秒更新用时的0.3%／0.7%；现有下一窗口CPU预取已经覆盖其中多数等待。预加载试验未改正式训练路径，不能宣称已经部署新的缓存或解决算力利用问题。NPZ抽样为ZIP_STORED；实际Python 3.12实现支持stored成员直接seek，没有观察到每次从文件头解压的假设问题。

宿主约1TiB内存不等于本容器可用额度：cgroup限额519251689472 bytes（约484 GiB），当时占用约462 GiB，其中约432 GiB文件缓存。不能按宿主空闲内存盲目把全量数据再复制进Python堆。原始记录见[data-loading-audit.json](results/2026-10-16_n04_aux_acceleration/data-loading-audit.json)。

## 辅助项真实退化与暂停

LIBERO完成step71后暂停，最新持久检查点为70；未保存的71与下一步部分计算未进入可恢复权重，原始metrics保留。RoboTwin在step45检查点后暂停，下一步2/8窗口没有optimizer更新。停止收据分别在两个run的`auxiliary-loss-diagnosis-pause.json`。两平台Supervisor为STOPPED，独立监控继续RUNNING；GPU诊断完成后无计算进程。

完整真实网络、官方两个固定dev窗口、同一flow随机种子，零optimizer更新的对照确认：LIBERO fresh→35→60→70的前缀时间标准差持续缩小。窗口0约0.00899→0.00327→0.00152→0.00121，预测NLL约1.299→1.321→1.466→1.732。表征项0.9899接近当前sqrt(eps)=0.01对应的常数输出惩罚0.99，不能称为良好收敛。step70两个不同任务的前缀差异RMS仅为前缀RMS的约0.43%。原始固定窗口结果见[fixed-window-auxiliary.json](results/2026-10-16_n04_aux_acceleration/fixed-window-auxiliary.json)。

进一步对照仍为零更新：同一窗口关闭BF16后时间变化更小，约0.000275；BF16约0.00121，说明BF16噪声掩盖了一部分实际退化，不能把切FP32当作修复。全黑RAW与真实RAW的前缀相对差异约1.79%，尚非逐值恒定，但视觉敏感性很弱。step70预测均值MSE在h1/2/4分别约0.690/1.515/4.482，而保持源后验均值的基线约0.243/0.246/0.239；这是学习潜在特征单位下的诊断，不是物理预测或机器人能力指标。移除当前加性本体控制的h1精确反事实更差，尚不支持把失配单因归于控制分支。见[causal-controls.json](results/2026-10-16_n04_aux_acceleration/causal-controls.json)。

一个真实train窗口的独立分目标反向表明：按当前外层权重，动作／预测／表征传到ConvNeXt的梯度L2约1.270／5.677／0.0000583；传到to_prefix约4.076／无该路径／0.000000221。参数版本不变、optimizer更新0。表征约束对读出极弱；预测确实有活梯度。它们不是“辅助项没接入”或BF16单一问题。相邻帧作为方差样本、共同学习目标的尺度漂移及读出忽略视觉的相互作用仍需分离验证，**尚未确定并实施有效修复**。见[objective-gradients.json](results/2026-10-16_n04_aux_acceleration/objective-gradients.json)。

## 与N02 v1.2 P2的速度口径

只读核对另一个checkout的当前r5完成记录及训练代码。N02 RoboTwin RAW臂1000次更新平均6.613秒、p50 5.709秒；每次batch2、每样本8帧，共16个时刻、2个动作监督点，BPTT2、整个真实π0.5冻结。N04 RoboTwin每次8×48=384个时刻（24倍）、32个动作监督点（16倍），包含RAFT和1/2/4步预测，串行单窗口累积，π0.5部分参数可训练。两种“update”不是同等工作量；N02已实际合批，N04的跨窗口合批仍是缺口。不能据此把N04约30多倍的每更新耗时差全部归于加载或单一模块。

N02测量入口为`/home/jeong/zeno/owac/worktrees/n02-measurement-completion/research/experiments/n02_v1_2/P2_training_completion_20261016.md`，配置和实现解释见同目录`P2_training_live_20261016.md`及`src/owac/training/n02_v12.py`。N02旧选中权重少量闭环同样0/10，速度和训练误差不构成已学会任务的证据。本轮未修改N02，也未启动其新训练。

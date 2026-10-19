# N04 v1.1 S2：Codex 接管训练、计算优化与简洁监控

2026-10-16，最新用户明确授权由 Codex 直接接入训练，不再让 DeepSeek 操作；首要界面只关注状态、速度与效果，并要求充分利用 RTX PRO 6000。此授权取代此前停止约束。旧停止回执、v2 权重和历史开发闭环0/100保留，不能用本轮工程修复覆盖负结果。

## 运行与预算

- checkout：`/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`，基线提交 `a81cf25148870fd83fbc06a4fabf374510493978`，运行含未提交实现。
- 服务器：用户原 N04 双卡实例，2 × RTX PRO 6000 Blackwell Server Edition，单卡约95.6 GiB，功耗上限600 W。未修改驱动／功耗上限。
- 版本：`n04-v1.1-s2-repair-v3`；loss `v1_1_s2_live_forecast_observation_vicreg_v3`。N04 新初始化，π0.5 使用官方权重；不迁移历史 v2 编码器。
- 官方 RawVLA-Bench 数据，seed101、D；LIBERO GPU0、RoboTwin GPU1。保留原拆分、归一化、本体历史及因果48帧窗口（16 burn-in +32 gradient）、每窗口4动作监督点、有效batch8、forecast1/2/4。完整 ConvNeXt-B／12轮RAFT／128支持／256读出token每相机／6层读出不缩减。
- **首段预算每平台2000次优化更新，warmup200后cosine**；这是本次接管的有限训练预算，非名义100k的完整训练或收敛结论。不是将步数改成窗口数。
- 固定开发集先评未训练起点；前100步每5步复验，之后每25步。每次2个固定开发窗口，固定flow matching噪声／时间并恢复训练RNG。此小型dev曲线用于学习进展，不代表独立最终测试。LIBERO第0步为1.08186，第5步为1.6359（变差），第10步为1.0104（较起点下降约6.6%）。这是早期固定开发集学习信号，不以它替代闭环效果。
- 第1步保存checkpoint，此后每10步、dev改进时另存best；周期权重保留最近2份和best，最终另存final。历史其他run权重不受影响。
- 由supervisor管理，训练异常不自动重启，避免无证据反复跑；监控服务可自动恢复。训练与SSH会话、浏览器、聊天生命周期分离。

## 保持训练目标的计算优化

1. 作者RKN稀疏带宽3的矩阵运算改成7条带的精确表达，保留作者basis、系数、噪声和双模式规则；dense参考路径继续可用。随机扰动basis后逐值、输入梯度及参数梯度与dense对齐。
2. 同源forecast的1/2/4步预测共享因果中间状态，从重复7次变为4次；不跨源、相机或TBPTT生命周期复用，不输入未来目标。
3. 梯度归约在完整有效计数可证明时，每源段将四损失合并一次求导。先保存在局部缓冲，最终核对实际valid counts，匹配后才写参数梯度；若不匹配，丢弃快路径梯度、恢复RNG及BN缓冲，按原实际计数归约重算。测试覆盖正常与缺失项fallback，与参考梯度相等。
4. 全窗口一次搬到GPU、CPU异步预取下一窗口、fused AdamW、cuDNN benchmark；减少重复拷贝、数据等待与小操作开销。
5. 编译纯RKN预测、测量backbone/fusion及Perceiver数学前向；进一步profile作者RAFT的纯GRU/update block，按预热实测选择。不编译有状态encoder.step，不使用CUDA graph指针复用，不改变精度政策（网络BF16，统计／协方差FP32）。

完整真实窗口前后向验证（optimizer更新0）：

| 平台／方案 | 窗口时间 | 峰值分配显存 |
| --- | ---: | ---: |
| 修复后原参考 LIBERO | 42.68 s | 53.9 GB |
| 修复后原参考 RoboTwin | 64.84 s | 81.2 GB |
| 带状RKN＋联合反传 LIBERO | 31.90 s | 55.14 GB |
| 带状RKN＋联合反传 RoboTwin | 46.07 s | 85.33 GB |
| 上述＋三处编译 RoboTwin，冷首轮 | 128.44 s | 见原始profile日志 |
| 上述＋三处编译 RoboTwin，预热第2轮 | 29.76 s | 74.68 GB |
| 再编译作者RAFT update block，预热第2轮 | **26.72 s** | **74.26 GB** |

前三组旧单轮脚本时间还包含参数梯度审计；编译两轮日志单列纯窗口前后向时间，最终receipt 29.88 s包含梯度审计。因此这些数据用于工程方向与成本判断，不能把它们直接称为实际optimizer更新速度或严格同比基准。真实更新含8窗口、clip、optimizer和调度；页面更新速度还计入验证／保存成本，采用最近5个完整周期均值。首次编译／恢复后的首轮标为预热，不使用恢复步数跳变推断速度。GPU忙碌度和功耗是采样瞬时值，不把占满显存当作计算用满，不宣称100%利用率。

## 界面与运维入口

用户入口：<http://127.0.0.1:8090/>（本地已有SSH转发，每5秒自动读取）。两张卡分别显示：

- 状态：模型加载、正常训练、验证、保存、停止／异常；展示最新窗口心跳年龄，不能仅凭旧loss把死进程标为运行。
- 速度：已完成更新／预算、当前8窗口进度、秒／更新、更新／小时、粗略剩余时间；GPU忙碌度、功耗、显存直接可见。
- 效果：固定开发动作误差、相对未训练起点的变化、最近真实验证步数；重复携带旧dev值的训练行不新增验证点。训练动作误差曲线仅作辅助，闭环成绩独立显示尚未评测。

本地SSH转发：

```bash
ssh -p 16724 -N -L 127.0.0.1:8090:127.0.0.1:8090 root@154.59.156.22
```

远端根目录 `/workspace/owac/research/n04-v1`：

- 授权回执：`CODEX_TRAINING_AUTHORIZATION.json`；旧停止标记已移到`control-history/`留档。
- 新run：`runs/v1.1/codex-s2/{libero,robotwin}-D-seed101`。配置`config.used.json`，心跳`progress.json`，指标`metrics.jsonl`，日志`logs/train.log`，权重`checkpoint-*.pt`。
- 冻结源码：`source/n04-v11-codex-train-20261016`（初始LIBERO eager）；`source/n04-v11-codex-compiled-20261016`（编译profile）；`source/n04-v11-codex-compiled-r2-20261016`（设备编号修正的失败启动）；`source/n04-v11-codex-compiled-r3-20261016`（梯度布局修复＋RAFT update block编译，当前两平台）。每个含`CODEX_SOURCE_MANIFEST.json`记录src/configs/scripts逐文件SHA256；不热改运行中代码。
- 入口：`scripts/n04_full/run_n04_codex.py`；配置`configs/n04/v1_1/codex_{libero,robotwin}_D.json`、`codex_compiled_{libero,robotwin}.json`。
- 进程：`owac-n04-codex-libero`、`owac-n04-codex-robotwin`、`owac-n04-codex-monitor`，配置`/etc/supervisor/conf.d/owac-n04-codex.conf`。
- 监控应用独立于训练源码：`monitor/codex/app/`；状态`monitor/codex/state/`。只读run与进程，不加载权重／占训练GPU；绑定loopback，用现有SSH访问。
- 性能原始证据：`runs/v1.1/codex-profile/{joint-libero,joint-robotwin,compiled-all-robotwin,compiled-raft-robotwin}/`；各有完整配置、日志与真实窗口验证回执。

RoboTwin首次监督启动发生设备编号错误：`CUDA_VISIBLE_DEVICES=1`隔离后旧平台配置仍选`cuda:1`，在构造encoder搬设备时退出，optimizer更新0。新Codex配置明确`device=cuda:0`，对应物理GPU1；失败心跳／配置保存在run下`startup-failure-device-ordinal/`，日志保留，不伪造为未失败。

RoboTwin第二次启动在完成8窗口后、首个优化更新时暴露编译梯度布局与fused AdamW不匹配。真实CUDA小型复现确认：参数为NCHW、梯度为channels-last时触发相同异常；梯度按参数stride重排后更新成功，数值与普通AdamW一致。已在optimizer边界逐张量对齐布局，不改变梯度值。由于失败的optimizer调用可能部分改变内部状态，本次进程和其内存状态全部丢弃，从相同seed新建；没有将其当作可恢复训练。旧目录整体归档`control-history/robotwin-codex-startup-errors-20261016/`，新run重建，失败不从证据中删除。

2026-10-16 04:04 UTC 实际运行已确认：RoboTwin修复后从相同seed新建，完成第1次完整有效batch8优化更新，保存非空`checkpoint-step0000001.pt`，首个完整周期227.81 s（含保存），第2次实际train_step为204.94 s；不是将无更新profile当成训练。LIBERO持续训练到第8步，等待第10步可恢复检查点后切换加速实现。最终实时状态见小型运行回执；本段是接入时观察值，不作为未来实时状态。

2026-10-16 04:09 UTC，LIBERO在第10步dev和checkpoint／best全部保存、指标写入后，暂停原进程；监督配置改为冻结r3源码、`codex_compiled_raft_libero.json`，以`checkpoint-step0000010.pt`恢复，优化器、调度器、训练RNG和同一loss定义一起重载。04:14:47 UTC已完成恢复后的第11步真实optimizer更新，首轮254.50 s包含编译预热。不是回退到step1或历史v2；切换前后配置完整保存在run下`compiled-migration-boundary.json`。

2026-10-16 04:18 UTC，LIBERO恢复后第12步train_step为158.28 s（第11步含编译为254.50 s）；RoboTwin已连续完成5步，第5步train_step为215.31 s。LIBERO第10步dev=1.0104377，相对初始1.0818601下降6.6%；RoboTwin第5步dev=0.6794，相对初始2.4531313下降约72.3%。这些是各2个固定窗口、8个动作监督点上的flow-matching MSE，不是物理单位的动作偏差或任务成功率。稳定速度仍以页面近期完整周期均值为准（包含验证／保存），不把上述最新单次train_step与冷启动均值混称。

## 验证与结论边界

`make check`通过；N04 v1.1／RawVLA合同／监控定向测试228通过，监控新增resume计数不计吞吐回归后合计229通过（监控11项），1项本地因缺websockets跳过。新增CUDA用例实际执行fused AdamW更新并对齐普通AdamW，另有两平台隔离GPU配置合同。远端三处编译及RAFT update block编译后真实48帧完整前后向、四项有效loss计数与全部预期训练组梯度检查通过，无optimizer更新、无权重写入；实际训练更新与保存状态另由运行回执记录。

这是修补版接管和性能优化的工程交付。尚不能证明方差可靠、motion／revision机制有效、编码器充分训练或机器人闭环提高；新模型闭环尚未运行。此前0/100开发闭环属于原checkpoint、原版本和原协议，保持原分母，不混入本次新run。

## 尚未完成的算力利用目标

双卡已各自持续训练，但编译后30秒采样的GPU忙碌度均值仅GPU0约30.1%、GPU1约33.1%，功耗均值123.1／127.5 W。单次读数显示两卡均P0，SM时钟2370／2340 MHz（上限2430），显存时钟12481 MHz（上限12481），温度31／30℃，功耗上限仍600 W；未发现该读数下的降频／功耗上限阻塞。两训练主进程约各占一个CPU核，这与逐帧Python调度和小张量串行工作造成空隙一致，但尚未做完整CUDA trace确认占比。

因此，本轮**不能声称已充分用满PRO 6000**。当前已删减可证明冗余、启用并验证编译和真实更新；跨独立窗口的合批与更多纯数学融合尚未完成，仍是性能缺口。不能通过改变因果边界、缩模型、缩窗口、少算loss或把占显存当满载来掩盖它。该缺口与训练可运行、监控可用、当前dev信号分开记录。

# 双卡蒸馏吞吐修复与移除哈希（2026-10-22）

用户先要求检查GPU利用率波动及数据管线，随后明确要求彻底移除本轮链路中的哈希校验。负责人Sol，主要实现由同一DeepSeek Harness Web V41-Flash Max完成。未更改损失、学习率、有效batch16、数据分区、因果历史、可训练范围或总预算。

## 实测定位

旧连续阶段第6–29步共24个稳定更新间隔：中位18秒，均值18.54秒（14–26秒），期间无验证或保存。35.36秒的独立被动采样中，两卡平均利用率7.86%/23.77%；每rank逻辑读取约0.80/0.87GiB，物理读取均0。不能将其归因为共享物理磁盘带宽不足。

同一批16个真实训练anchor的CPU诊断：旧RAW轨迹缓存仅2条，热轮仍10次miss、重复解压384.5MB，样本构造1.550秒；缓存10条后热轮0次miss/解压，0.892秒。教师204行目标原本已预加载，读取耗时不足毫秒。更大的问题是每个递归时刻为model_identity重复复制约57.8MB stem至CPU计算哈希，并在视觉参数更新后复制约1.65GB继承视觉计算哈希。一个实际批次需228次packet构造，其中110次重复。

## 实际修改

- 当前训练执行路径移除RAW文件、目标payload/identity、模型权重/递归状态、配置、保存/重载以及拥有者启动器的哈希计算及校验。活动配置为`shared-stage/config-nohash.json`，已移除sha/digest/fingerprint/checksum字段。
- 递归状态使用可读的进程内模型实例/参数版本代数，保留状态形状、相机、因果端点、source和reset检查。不同模型实例不能混用状态。没有用假哈希、缓存哈希或不透明随机值替代逐状态完整哈希。
- 共享RAW仍直接只读；train+validation10源一次加载到RAM；静态packet使用每rank最多1.5GB缓存，不缓存学习到的stem输出、隐藏状态或跨更新学生记忆。
- 去除每步不使用的代表权重CPU复制，仅原首更新证据需要时执行。新增host-wall阶段耗时，明确包含等待/同步，不冒充GPU kernel计时。
- 历史记录和旧字段保留，不重写已有事实；历史通用教师制备/导出工具仍有旧哈希API，但不被本次训练执行路径调用。原缓存目录名保留为普通兼容路径，不作为哈希校验依据。

## 独立执行验收

先对已拥有的两rank发送SIGTERM，在55步/880anchors完整保存，torchrun native exit0。Sol不做哈希，直接读取Adam442状态step55、scheduler55、两rank RNG/采样器和连续1–55日志确认恢复点。原计算成本计入预算。

134项相关测试及make check通过；Sol补充删除row identity旧字段比较后9项针对性测试及make check通过。实际同策略资产的CPU构建通过，428,579,952可训练参数。临时验收入口禁止OWAC代码调用hashlib构造器：实际双卡55步完整重载native0、0额外更新，两rank哈希调用均0；实际55→56优化更新、验证、完整保存也native0，两rank均0。该临时阻断探针不在连续训练命令中。

Sol直接读取56步权重/Adam/scheduler，442个Adam状态均step56，scheduler56，896anchors；抽样次序逐项等于55步sampler恢复后的下一批。全部442状态张量FP32且有限；代表视觉层相对step0有非零累计变化。55步模型已被既有latest+best保留规则裁剪，故55→56逐权重直接差值不可再测，不把相对step0变化写成单步差值。

连续训练从完整56步检查点恢复；配置`shared-stage/config-nohash.json`，性能侧配置`configs/experiments/raw_full_visual_performance_v1.json`，拥有者日志前缀`nohash-continuous`。无优化器/随机状态/采样器/预算重置，没有重复55或56更新。启动前剩余137866.9734096253 aggregate GPU秒；外部拥有者预算守卫继续有效。

## 修复后实测

新阶段第60–94步：34个更新间隔中位2秒，均值2.059秒；rank0记录每步host-wall中位2.038秒。与旧18秒中位相比约8.8倍吞吐，样本/历史长度不完全相同，属于同协议连续训练观测，不是严格逐批A/B。

- 样本构造中位5.63毫秒；状态版本设置2.09毫秒；前反向1.842秒；梯度数值检查/优化器93.8毫秒。
- 35.70秒被动采样77→94步：两卡平均利用率43.8%/46.94%，均峰值100%；低于20%的采样由34/35、27/35降至2/35、0/35。
- 两rank物理读取均0；逻辑读取约10KiB/0。158个训练endpoint packet全部进入缓存，rank0缓存1,141,604,352字节、无淘汰，不存储学习状态。

已经消除了主要哈希热路径和重复数据准备开销。GPU并非持续满载；当前计时大部分在逐anchor因果展开与前反向，不能把host-wall当GPU独占计算时间。未进行微批合并/序列批处理架构改写。训练继续，闭环0，未证明控制恢复或充分收敛。

原始证据：`outputs/raw-distillation-full-vision-20261022/`中的`throughput-baseline.json`、`data-profile.json`、两组live-sample、`throughput-nohash-comparison.json`、nohash reload/update56日志、退出收据、独立验收、source文件清单revision7（无哈希）及Harness round-03-throughput事件。

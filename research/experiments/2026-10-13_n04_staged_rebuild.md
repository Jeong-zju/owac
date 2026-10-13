# N04 分阶段修复与可信基准重建

负责人 lzh；2026-10-13。按用户明确指定的顺序，每个阶段只改变一个因素，阶段源码/配置/数据及结果冻结后继续。原错误数据的训练和数字保留，参见[诊断记录](../reviews/2026-10-13_n04_mechanism_diagnosis/README.md)。

## 固定协议

- tiny，64×64 三通道生成器线性测量，FP32 / RTX 5090。当前为合成机制研究，不表示实采 CFA 或新真实 π0.5 闭环。
- 每次 200 updates、batch=4、序列12、burn-in2、TBPTT8、AdamW lr=3e-4，顶层 loss 权重1/.5/.1/.1。结构、内部 loss 或数据分布变化逐阶段单独记录。
- 三个配对运行分别使用独立训练数据种子41000/41001/41002及初始化0/1/2；同一运行中方法共享相同数据及顺序。变化数据的哈希在训练前保存，训练后复核未被修改。
- 开发评测42000/42001/42002用于阶段诊断；最终测试45000/45001/45002预先保留，阶段开发中不读取。oracle 诊断另用43000/43001场景，仅作固定权重干预。
- 检测阈值固定0.5；9条件、16帧、batch=2，object-only第8帧停止。location、检测、持续 change、传播误差、恢复/censored及实际成本分别报告。
- 200步与相同数据是受控起跑比较，不是容量/算力严格匹配，也不能证明充分收敛。最终测试前固定候选版本，不用测试结果反向修改规则。

## 阶段与验收

1. **S1 数据修正。** 批次位移显式扩展物体轴，阻止未渲染对象进入标签；图像与掩码共享画布、裁剪、旋转，修正亚像素采样错位。batch=1/2/4、样本隔离、批次置换、多对象及分数位移/旋转新增10项行为回归。全仓225 passed（2个已有warning）。可信基准为candidate、motion-compensation、filter、no-change、no-illumination，三个独立数据/初始化运行。
2. **S2 观测残差接口。** 用历史观测形成与当前测量同语义的预测，任务 stable 继续自由表征；逐项检验静态、真实运动、遮挡、饱和、因果性和梯度。先固定S1权重作干预确认残差来源，再在相同S1数据上单独重训该规则。
3. **S3 成像解释与组合训练。** 先只把增益纳入观测预测，用真实/估计增益分别诊断；再仅改变训练的因子组合，用同结构对照隔离数据效果。原测量保持任务读出；成像变化与可靠性分别说明。
4. **S4 动态记忆。** 分别关闭历史记忆、事件门或变化读出，使用停止/反向/遮挡重现检查；外部真值运输仅作诊断。只有诊断显示某估计模块是瓶颈时，再训练相应模块；负结果如实记录。

## 运行与产物

代码入口 [run.py](results/2026-10-13_n04_staged_rebuild/run.py) 与 [oracles.py](results/2026-10-13_n04_staged_rebuild/oracles.py)。分阶段小型协议/指标留在 [结果目录](results/2026-10-13_n04_staged_rebuild/)，权重、完整训练历史、原始逐帧评测及日志留在忽略的 `outputs/n04-staged-rebuild-20261013/`。

```bash
uv run --locked --extra encoder python research/experiments/results/2026-10-13_n04_staged_rebuild/run.py --stage s1-data
uv run --locked --extra encoder python research/experiments/results/2026-10-13_n04_staged_rebuild/oracles.py --stage s1-data
```

## 实测结果

开发训练与固定权重诊断分开留档；最终测试在源码与配置冻结后统一执行。以下结果表由配对运行产物填写。

### 已落实的源码边界

| 阶段 | 唯一改动因素 | 源码提交 | 训练数据 |
| --- | --- | --- | --- |
| S1 | 批次/物体轴及图像—标签的共同几何采样 | `fc632be` | 新版可靠数据 |
| S2 | 残差参考改为历史传感器观测，经同一 stem 编码 | `66db15a` | 与 S1 逐张量哈希一致 |
| S2b | 预测与当前观测在 stem 前使用同一当前支持掩码 | `3187002` | 与 S1 一致 |
| S3a | 增益实际乘到历史线性观测预测，再进 stem | `c292112` | 与 S1 一致 |
| S3b | 训练分布加入 static 和三种运动/照明共存条件 | `f281536` | 新组合分布 |
| S3c | 在 S3b 分布上关闭实际增益解释，仅保留增益提示 | `f281536` | 与 S3b 逐张量哈希一致 |

所有阶段保留纯运动预测、独立观测修正、独立事件分数、三原因加 unknown、stable 更新和有符号 change cell。

新观测接口在 correction 内使用 `reference = stem(current_support * estimated_gain * propagated_RAW_history)`，与 `stem(current_support * current_RAW)` 作差。任务状态的 ego 预测单独保留。增益估计会读取当前 RAW，因此这里是观测后的条件成像解释，不是观测前的盲预测；`predict_before_observe` 仍不读取当前图像。不同重训模型的 stem 尺度也可能不同，不能仅由跨模型的特征残差幅值证明机制；固定权重干预和下游配对结果分别报告。旧默认配置用于重现，新增研究配置为 `configs/encoders/n04/tiny-staged-research.json`。检查点对残差参考、支持掩码和增益解释三项设计配置进行兼容性检查，跨设计加载必须显式声明诊断覆盖。新主路径没有输入真值运动、真值增益、物体标签或未来动作。

S2b 是执行中追加的单因素接口修正：当前观测会把饱和/不可用像素置零，而历史观测预测原先没有同样置零，GroupNorm 将这种差异扩散到全图。它在固定权重诊断后独立训练，并未和增益解释合并。

### 增益与观测接口的诊断

固定 S1 权重，仅切换观测参考：camera-only 残差三运行均值 0.36242→0.09076，定位 0.15423→0.16095 px。这支持内部任务表征污染了观测残差，不支持当时的任务收益。

固定 S2 权重，illumination 条件先匹配当前支持掩码：残差 0.09279→0.05245；随后让真值增益实际参与预测：0.05245→0.03476。掩码未匹配时，直接用真值增益反而为 0.09463。对应定位为 0.10227、0.14174、0.14725 px，不能把残差下降等同于定位改善。

真值 ego / gain / event 仅用于诊断。早期 S1/S2 oracle 的 ego 模式同时将 uncertainty 固定为0.05，属于“运动加置信度”的复合干预，不能归因于运动一项；原产物保留。审计后新的 `motion-only` 诊断与动态测试只替换运动，保留原传播器输出的不确定度。上文掩码与增益的单因素结果均未使用 ego 模式，不受该问题影响。特别是饱和后历史可保留多帧，当前/上一帧的真值增益比不能保证是每个历史像素的严格最优解；这类干预是瓶颈探针，不是数学上的最优上界。另有未饱和、无噪声、支持完全匹配的单元场景验证增益解释本身；物体平移仍留下非零残差，四种 CFA 排列保持各自采样相位。

### 动态记忆的结构诊断

冻结 S3b 三份权重，固定同一 stable 状态与缺失观测，只把 change 从全零改为 0.5：三运行的下一帧 stable 最大差均为 **0**，change 最大差约 **0.370**。源码原因是完全缺失时 stable 直接采用 ego 预测，composition block 的结果被 availability 掩码排除；本轮定位探针只读 stable。change 名义半衰期为 **2.30—2.32 帧**。这是目前实现和探针的具体依赖关系，不证明所有后端都无法利用变化记忆。

动态诊断分别撤掉历史 change、撤掉 block 中的 change 输入，另用真值事件、自运动及物体平移进行干预。反向场景使用独立重新采样的噪声，避免回放完全相同的噪声图案；完全遮挡为第8—11帧无观测，目标真值仅用于评测。恢复使用原来的 IoU≥0.5、平均定位误差≤2 px 连续两帧判据；未恢复保留 censored。该恢复指标是 batch=2 的序列统计，不冒称逐物体事件成功率。

### 比较与成本的边界

- 组合训练只改变数据分布；固定 200 updates 后，各旧条件的出现频次也随之改变，因此它的收益属于这份训练混合分布，尚不能全部归因于组合可识别性。
- 独立统计单位为三份训练数据/初始化；先对每份权重的三个评测场景取均值，再报告运行间标准差。没有把九次评测当成九份独立训练。
- 最终留出场景统一使用 canvas_margin=32，覆盖 16 帧最大相机位移30像素，防止边缘裁剪截断运动；全部方法使用同一设置。开发/诊断场景的实际最大相机轴位移已检查小于原 margin=24。
- 参数独立复核：编码器及定位探针294,123，加 future-measurement 辅助头867，共294,990，与训练日志一致。step 时延含编码/状态更新，不含真实 π0.5、动作头或后端前缀适配。
- 本轮训练输入是三通道合成线性测量。CFA 是采样/数值接口回归，未做新 CFA 大规模训练；没有新增真实 RAW、真实 π0.5 适配或机器人闭环收益证据。


### 开发集分阶段结果

均值先在每次训练的三个场景内计算，再汇总三次独立训练；定位给出均值±运行间标准差。

| 阶段 | 定位误差 px | 相机残差 | 照明残差 | 物体运动漏报 | 两者运动漏报 | 相机运动误报 |
| --- | --- | --- | --- | --- | --- | --- |
| s1-data | 0.385±0.182 | 0.3583 | 0.3640 | 29.72% | 18.64% | 1.38% |
| s2-observation | 0.383±0.190 | 0.1019 | 0.0969 | 15.69% | 17.87% | 1.40% |
| s2b-support | 0.409±0.199 | 0.0249 | 0.0577 | 44.66% | 28.97% | 1.21% |
| s3a-imaging | 0.424±0.185 | 0.0229 | 0.0264 | 38.28% | 20.50% | 1.35% |
| s3b-combined | 0.377±0.129 | 0.0228 | 0.0258 | 25.74% | 5.42% | 1.77% |
| s3c-combined-hint | 0.393±0.129 | 0.0247 | 0.0583 | 25.83% | 4.91% | 1.81% |

S3b 同分布滤波和无变化分支定位分别为 0.436、0.394 px。候选为 0.377 px，但运行间波动不能忽略；关闭整个变化分支也改变了事件门和更新，因此该对照不能单独证明历史记忆有用。


### 共存条件的开发诊断

下表只取 camera+object+illumination 三种原因同时存在的条件；其他组合完整保留在结果文件。

| 模型/训练 | 定位误差 px | 外部事件漏报 | 残差 |
| --- | --- | --- | --- |
| s2b-support/candidate | 0.2647 | 37.27% | 0.0627 |
| s3a-imaging/candidate | 0.2497 | 23.03% | 0.0297 |
| s3b-combined/candidate | 0.2077 | 6.13% | 0.0297 |
| s3b-combined/predict_update_filter | 0.2814 | — | — |
| s3b-combined/n04_no_change_branch | 0.2375 | — | 0.0303 |
| s3c-combined-hint/candidate | 0.2027 | 5.88% | 0.0637 |

仅加入增益解释有局部改善；加入组合训练后漏报进一步下降。但同样组合数据上的“仅增益提示”对照定位/漏报略优于实际增益解释，增益解释的额外任务收益未被这组开发结果建立。全因子 oracle 也没有稳定领先；真值残差更小/标签更准与下游性能不能互相替代。

## 最终留出结果与结论

开发完成后以 `f2e0db6` 冻结源码、配置、目标列表与脚本；最终45000/45001/45002只运行一次，不据其结果改模型、阈值或训练。36份权重、108次开发 suite、216条组合开发条件、54次最终13条件 suite、开发/最终各297条动态干预均完成；完整身份、源码、权重、数据配对和反事实不变量审计通过。

| 模型 | 原9条件定位 px | 组合3条件定位 px | 三因素共存漏报 |
| --- | --- | --- | --- |
| s1-data/candidate | 0.332±0.093 | 0.379±0.081 | 20.63% |
| s1-data/predict_update_filter | 0.431±0.132 | 0.417±0.014 | — |
| s3b-combined/candidate | 0.310±0.063 | 0.332±0.055 | 6.60% |
| s3b-combined/predict_update_filter | 0.442±0.141 | 0.423±0.010 | — |
| s3b-combined/n04_no_change_branch | 0.303±0.052 | 0.328±0.070 | — |
| s3c-combined-hint/candidate | 0.302±0.067 | 0.320±0.065 | 6.04% |

同分布、同结构的提示对照更关键：S3b 相比 S3c 的定位配对差为 **+0.0074±0.0259 px**（原9条件）和 **+0.0120±0.0262 px**（组合3条件），正值表示实际增益解释更差；没有建立其额外定位收益。候选相比 S1 平均下降0.0221 px，但三运行有一份退化；三份训练还不足以形成强统计结论。

事件召回改善同时带来误报代价：S1→S3b 的纯相机运动误报 **1.40%→1.81%**，纯照明误报 **1.49%→1.80%**。无变化分支没有事件检测输出，表中的空值不能解释成零漏报。

### 最终动态记忆结果

下表为连续24帧任务的平均定位误差；只有“无变化分支”和滤波基线各自重新训练，其余是同一候选权重的单因素或明确命名的诊断干预。

| 对照 | 停止 px | 反向 px | 含观测缺失序列 px | 缺失4帧内 px |
| --- | --- | --- | --- | --- |
| candidate/base | 0.1442 | 0.3417 | 0.5963 | 1.3697 |
| candidate/no_history | 0.1457 | 0.3423 | 0.5982 | 1.3752 |
| candidate/no_change_use | 0.1482 | 0.3454 | 0.6017 | 1.3795 |
| candidate/transport | 0.1430 | 0.3393 | 0.5922 | 1.3648 |
| candidate/event | 0.1364 | 0.3194 | 0.5675 | 1.3355 |
| candidate/ego | 0.1412 | 0.3690 | 0.6218 | 1.3582 |
| candidate/ego_transport | 0.1400 | 0.3663 | 0.6181 | 1.3554 |
| candidate/ego_transport_event | 0.1298 | 0.3442 | 0.5864 | 1.3140 |
| predict_update_filter/base | 0.1333 | 0.3538 | 0.6406 | 1.4337 |
| n04_no_change_branch/base | 0.1198 | 0.2940 | 0.5537 | 1.3596 |

删除历史 change 只使停止/反向/缺失序列误差增加 **0.0015/0.0007/0.0019 px**；真值物体运输只改善约 **0.0012/0.0023/0.0042 px**。变化分支整体关闭的匹配模型在三个动态场景中均更好。真值事件仍存在局部改善空间，但不能解决前述“缺失观测时 change 不进入 task prediction”的依赖路径问题。

所有动态恢复判据均为0帧且censored=0，原2 px/0.5 IoU门槛在这批场景上没有区分力；原门槛保留，主要比较预先记录的连续定位误差。真值运动、gain、event是诊断干预，完整真值组合并不保证已训练下游在分布改变后达到最优。

### 成本与工程审计

| 模型 | 训练参数总数 | batch=2状态 bytes | 条件p50均值 ms | 条件p95均值 ms |
| --- | --- | --- | --- | --- |
| s1-data/candidate | 294,990 | 7,241,744 | 5.27 | 8.47 |
| s1-data/predict_update_filter | 49,768 | 3,244,048 | 2.14 | 3.77 |
| s3b-combined/candidate | 294,990 | 7,241,744 | 8.50 | 11.96 |
| s3b-combined/predict_update_filter | 49,768 | 3,244,048 | 2.64 | 4.31 |
| s3b-combined/n04_no_change_branch | 294,990 | 7,241,744 | 6.60 | 9.71 |
| s3c-combined-hint/candidate | 294,990 | 7,241,744 | 6.35 | 9.43 |

同200步不是同容量或同计算量；候选约29.5万参数，简化滤波约5.0万。实际增益解释/第二次stem增加计算，当前没有效率收益。优化器状态还核对了实际曾接收梯度的参数；本轮readout未作真实动作适配。

全仓 **248 passed**（2个既有warning），`make check`通过；新版配置与S3b冻结配置一致。`git ls-files -ci --exclude-standard`为空。大权重、逐帧轨迹、完整历史与日志均保留在忽略目录，没有覆盖旧错误数据实验或推送远端。评测结束后仅补加runner的提前复跑保护及文档：重复请求现有阶段会在写协议前退出，已验证原协议哈希不变；编码器、数据和评分规则保持冻结。

### 当前验收判断

- **数据/工程契约：本轮修正及验证完成。** 渲染—标签逐样本一致、观测空间一致、支持掩码一致、实际RAW域增益解释和组合条件生效；纯predict的信息边界保持。
- **H1：未建立稳定支持。** 最终相机/物体change幅值比0.880，但开发为1.586，且本轮没有完成新分布下的无ego/错位训练对照，不能据一个平均比值通过。该幅值比不是检测误报率。
- **H2：未通过预设支持条件。** 最终照明/物体change幅值比1.057；同分布仅提示对照定位/漏报更好。成像解释降低特征残差，但未建立额外任务收益。
- **动态记忆：存在很小的缓存作用，尚无竞争性收益。** 变化分支整体消融更好，真值运输没有显示值得立即训练新运输网络的收益；缺失帧中的依赖路径瓶颈已用精确干预确认。
- **具身/架构主张：未建立。** 本轮为合成线性测量上的tiny起跑实验，不代表真实RAW、π0.5闭环、跨相机迁移或scaling验证。

### 下一次应验证的因素

先将“外部动态状态的预测”明确放到无观测时也能运行的prediction路径，保持观测修正的有效性约束。先用真值外部位移做该新路径的诊断上限，并同固定状态/无动态预测比较；只有定位和不确定性确有收益，再训练从历史可用观测估计运动的模块。未观测到的突然反向应增加不确定性，不能用未来真值偷渡成在线能力。

同时保持事件识别与观测更新可靠性的职责可检查：真值事件改善部分动态检测，却未稳定改善混合场景的定位，说明当前gate与任务更新已共同适应，不能只将分类分数推向0/1就假设任务更好。下一轮每次仍只修改一个接口或监督项。

## 复现与证据入口

```bash
# 在新输出位置/无本轮既有权重的checkout内运行；脚本拒绝覆盖冻结权重。
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/run.py --stage s2-observation
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/run.py --stage s2b-support
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/run.py --stage s3a-imaging
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/run.py --stage s3b-combined --models candidate predict_update_filter n04_no_change_branch
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/run.py --stage s3c-combined-hint
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/combinations_eval.py
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/oracles.py --stage s3b-combined --conditions camera_only object_only both illumination object_illumination both_illumination
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/dynamics.py --split development
# 冻结以后只执行一次留出测试
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/final_test.py
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/dynamics.py --split final
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/outcome_summary.py
uv run --locked python research/experiments/results/2026-10-13_n04_staged_rebuild/integrity_audit.py
```

汇总：[开发阶段](results/2026-10-13_n04_staged_rebuild/development-summary.json)、[留出及动态](results/2026-10-13_n04_staged_rebuild/outcome-summary.json)、[完整性审计](results/2026-10-13_n04_staged_rebuild/integrity-audit.json)、[参数核对](results/2026-10-13_n04_staged_rebuild/resource-audit.json)、[缺失观测路径干预](results/2026-10-13_n04_staged_rebuild/memory-path-diagnostic.json)。

# Sol 独立验收：限定 RAW 教师复现

验收时间：2026-10-20T15:04:50.819853+08:00。本记录是用户“授权上述限定教师复现、必要示范回放和仿真验证范围”的阶段结果；不把学生实现、loss 或 Harness 完成收据写成学生控制恢复。

## PREP 阶段验收登记

2026-10-20，按用户要求登记 **PREP 已验收通过，当前阶段 ALIGN**。用户所称 D1.0 PREP 的工程实现、资产核验与有效教师取得要求已满足；当前方案仍按取消固定单帧限制的 OWAC-D1.1 执行。ALIGN 未完成项是教师／学生时序范围、每相机状态与缓存对接，以及真实数据零更新梯度检查。此登记不将学生控制标为恢复，不新增学生训练或闭环授权。

## 结论与选定权重

取得本地重建并通过正常光单任务开发筛查的 **RAW 输入教师**：`libero_goal/task6`，指令 `put the cream cheese in the bowl`，task_order_index=0。共同20个训练前冻结的新测试初态，Default参照20/20、RAW教师20/20；配对退化0、恢复0。每臂成功率的95% Wilson区间为83.89%—100%。符合“双方≥16/20且教师少不超过2次”的开发筛查；不是统计非劣结论。

选中的是**作者架构的第0步初始化前端＋当前实际使用的同源π0.5权重**。它不是作者发布的训练后Neural ISP；本轮依然未找到该公开权重。初始化已能完成本任务，不能宣称本轮155次优化带来学习收益。学生训练0次、学生闭环0场，学生控制恢复尚未证明。

- 可执行前端artifact：[teacher-validated-normal-task6-init0.pt](/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/teacher-validated-normal-task6-init0.pt)，SHA256 `9de297ef2e1deefbf961a773f81491e6c4ee8e1a22bfc34b611519481b5b6d9d`。
- 完整模型注册表：[teacher-registry.json](/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/teacher-registry.json)。前端artifact必须与登记的8.52GBπ0.5后端、tokenizer、统计量和固定源码共同加载，不能当成完整VLA单文件。
- 版本化小收据：[teacher-control-acceptance.json](teacher-control-acceptance.json)。

## 实际实现、参数和训练

主要编码由本机 DeepSeek Harness V41-Flash Max 完成，Sol负责边界、接口、逐轮返工和真实命令验收。新增 `raw/teacher_data.py`、`simulation/rawvla_teacher_replay.py`、`training/rawvla_teacher.py`、`evaluation/rawvla_teacher.py`、`simulation/rawvla_teacher_controller.py`／`rawvla_teacher_eval.py`、预算账本与CLI、针对性测试；研究路线决议0016未重复新建。工作在原main checkout，保留已有修改，未提交／推送，旧N01／N02／N04未恢复。

只训练作者RawVLA前端829,172参数、74个参数张量；继承的SigLIP、原投影、语言／动作后端全部冻结。实际155更新、有效batch8=1×8、1,240次训练anchor呈现；保留checkpoint0、100和完整optimizer／scheduler／RNG／sampler的155恢复点。100点74个前端参数张量确实改变，AdamW各状态step=100，全部813项后端状态哈希稳定；100点重载前端83项、非零动量74项、scheduler、采样器和CPU/CUDA RNG逐项一致。

配方是登记的公开源码版本：AdamW lr1e-4、betas(.9,.95)、eps1e-8、wd1e-8、warmup100、cosine至1e-6、clip1；原生flow动作.15、brightness.01、chroma.001。动作仍用真实7维＋原生32维补齐、有效未来mask；未用低32维平均误差冒充动作能力。有效微批损失分母按整批有效未来步数归约。

验证选型只比较已登记并测量的0／100点：前5步物理7维MAE分别0.04319213／0.04350192，同分取更早。155仅为恢复点，未参与选型。第0步在测试前锁定；没有根据20测试结果重选、调参或重跑。达到可用教师目标后停止剩余额度；不是声称完成2000更新或复现论文训练收益。

实际模型参数总数3,617,586,692；本轮后端更新参数0。关键运行配置是 **LIBERO `discrete_state_input=false`**：原生helper有效补0、tokenizer不接收state。它与历史成功基线一致，不能根据PI05类默认True或未生效的pad=-2推断。当前接口传规范state8，由原生helper按实际配置补齐。

## 真实资产、历史与数据依赖

官方单任务HDF5 50条、5,349帧，SHA256 `ee58a7d72c143a71bf1448674e70940ac7781c1d1bb10a1b0c7a5c69b26abbef`。50/50必要状态条件回放完成；100个RAW/RGB NPZ与原动作、状态、稠密帧索引逐项审计。按源示范40/5/5，seed101；没有同源光照变体泄漏或缩小N。源9维robot_states显式变为机器人8维 `[eef_xyz,axis_angle,gripper2]`；仿真79维state仅供重置／回放，不进策略。

本轮为 LIBERO **RGB8逆处理的三通道pseudo-RAW**，不是原生Bayer。NN教师仍显式经过Neural ISP→RGB→原SigLIP／投影→冻结π0.5。依赖既有RGB预训练和LIBERO适配的π0.5；40/5/5仅是当前前端适配拆分，不声称整个预训练策略未见过这些示范。没有静默加入额外视觉数据。

教师每相机收集6个相邻物理帧，独立GRU／theta跨重规划保存，每5个环境步更新；episode reset清空各相机状态。训练BPTT为6个stride5端点、窗口内不断梯度；部署为持续因果状态，两者如实区分。双相机agentview／wrist，Native第三相机明确mask=False。动作采样horizon10、flow10、实际执行5，三者不同。

30个新初态在训练前冻结（验证10／测试20），固定种子和状态哈希，不用历史episode2或旧50初态替代。全部目标位置与源示范／历史初态的预设重复检查通过；未按策略表现筛初态。测试两角色的首个RAW视野哈希、光照和共用flow噪声均逐对一致；第一步动作已不同，此后的观测会分叉，不作逐步数值因果归因。

## 独立证据与明确差异

| 项目 | 实际证据 |
|---|---|
| 真实加载与冻结 | 同源checkpoint813/813，零active missing；测试后后端813项与前端83项均稳定 |
| 实际更新／恢复 | 100点74个前端张量改变；optimizer、scheduler、sampler、CPU/CUDA RNG重载一致；155点与全局155／1240／pending0一致 |
| 实际配置复核 | 102窗口中原生state8与旧显式zero32的图像、mask、token、state和完整动作逐位一致 |
| 教师tokens接口 | 真实专家RAW首窗口、frame30后续窗口、另一episode reset三例；原生重复误差0，先冻结1e-6容差；有效前缀、全部语言、mask、position IDs、10步速度场、归一化／物理动作、执行前5步差0 |
| 缺失相机占位 | 原生黑图嵌入与桥接零token的256个未观测相机位置，原始prefix最大差158；两边mask均False、对应prefix pad均False。原始差异完整报告，不称全量raw-prefix逐位相同；有效位置／全部语言仍严格比较。桥接新增投影参数0 |
| 真实控制 | 全40个测试执行chunk逐项等于各重规划采样结果前5步，40个视频帧数等于真实执行步数；动作、视频、初态和配对完整分母逐项核对 |
| 独立推理依赖 | 20场教师实际调用RAW前端774次=387重规划×2相机；独立入口不加载训练RAW/RGB、HDF5或教师特征缓存；读拦截0次，无回退 |

原始证据：[测试逐场审计](/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/receipts/sol-paired-test20-independent-audit.json)、[首观测配对审计](/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/receipts/sol-first-observation-pairing-audit.json)、[接口三例](/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/receipts/sol-real-teacher-bridge-v2-result.json)、[重复与reset独立复核](/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/receipts/sol-bridge-repeat-and-reset-audit.json)。

离线视觉诊断保持state、语言、mask、目标动作、时间／噪声规则一致，102 anchors前5步真实7维MAE：正常 0.04319；RAW常量零输入 0.31101；跨示范打乱RAW 0.22746。显示模型依赖视觉；这是离线诊断，**受控单因素目标位置干预未运行**，不能由此宣称强因果视觉推理或超越教师。

## 中断、负结果与门槛变更

完整验证N=10：9成功、1次Sol人为中断，不删分母、不重跑该初态。Sol一度根据类默认值误判state padding，实际配置为False，随后用历史checkpoint配置、源码及真实102窗口逐位等价排除误判。中断时已完成8场、第9场进行中，先前“仅1场”口头计数也已更正。第9场部分视频保留，但旧入口未及时保存部分动作轨迹，故该项**unavailable**，不补造。未来中断路径已修复为保存已知部分证据并传播原中断，普通策略失败仍保留为有效负证据。

Sol曾在测试前加10/10成本门槛；它因该中断无法验收，未宣称通过。在看到任何新测试结果前，明确撤下该成本门槛、锁定原验证MAE选中的0点，改用原授权的独立20对开发筛查；原最终≥16／20且差≤2的阈值不变。这个执行偏差和原始收据都保留。

训练前修复过79维回放state误选、raw-only读RGB、微批归约缺32维均值等接线错误。资源探针中GPU共享引发OOM、CPU ISP动作差0.00622超过0.001、激活重计算梯度差1.54e-5超过预冻结1.06e-7，均未放宽容差放行。实际控制保持原GPU Neural ISP和GPU后端，改用**OSMesa CPU渲染**；双方一致，不能声称本轮逐位复现历史EGL闭环或论文数字。

## 成本与运行责任

本机一张RTX5090，共享另一项目常驻服务；未停止／接管其他作业。Python3.11.14、Torch2.7.1+cu128、Transformers4.57、MuJoCo3.2.3、robosuite1.4；原作者版本差异登记在runtime收据。累计GPU预留墙钟约0.867小时，新增约5.75GiB，均低于12小时／50GiB。预算保守包含模型装载、渲染、保存等待；模型调用token另见Harness账本。50回放和50策略场次额度已用完，没有超额重试。

| 角色 | 实际act样本 | p50 / p95 毫秒 |
|---|---:|---:|
| Default参照 | 399 | 130.63 / 261.99 |
| RAW教师 | 387 | 128.06 / 281.85 |

计时从burst打包到物理动作输出，包含前端／tokenization／完整10步采样／后处理；不剔除warmup，不含仿真、视频或RGB逆处理数据制备。共享GPU且未优化，不能据此宣称更快。真实GPU驻留训练pilot峰值12,820,275,712bytes；常规CPU激活卸载探针峰值7,865,185,280bytes。学生延迟／显存尚未测量。

命令和资产见 [TEACHER_RUNBOOK.md](TEACHER_RUNBOOK.md)，原始调度在 `/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/harness/round-01` 至 `round-15`，保存task／events／stderr／exit／真实provider-model-effort选择及Sol review。源版本HEAD `63cf8a41e4a3833d6df03221af08179b6d3e9a33` 加本轮diff；`source-pilot100`、`source-cpu-continuation`、`source-paired-test20`均与实际启动哈希核对。原始未跟踪源码也纳入最终清单，不只看git diff。

## 最终工程检查与 Harness 身份

Sol独立运行 `make check`：ruff check通过、1006文件格式检查通过、compileall通过；新教师与既有蒸馏核心针对性测试130 passed、0 skipped；统筹台13项测试通过。真实策略结果来自上述40场实际测试及原始视频，不以测试套件替代。统筹台只收录Markdown，JSON验收收据通过本报告链接访问；曾误登记JSON导致摘要过期，已修正并重建、复测。

本轮15轮真实headless派工均记录官方provider／`deepseek-flash`／`max`，未以Codex子agent替代。执行末尾再次核对 `/home/jeong/.local/bin/dsh --version` 仍为 `0.2.1-rc.2`；本机发现的版本目录为0.2.1-rc.2及0.2.0-rc.2，不能据此判断用户此前升级是否发生在另一个安装入口。本轮没有升级Harness或重启／接管人工Web服务。版本证据保存在 `receipts/harness-final-version.json`，完整派工小收据见 [teacher-harness-ledger.json](teacher-harness-ledger.json)。

## 下一步范围

当前只是RAW教师已取得；不是RAW stem学生恢复。下一步需把本轮规范artifact、实际Neural ISP输出RGB和每相机因果状态绑定到蒸馏数据／缓存，匹配学生历史范围并做真实动作梯度零更新检查。旧每帧stem实现不能无条件拟合依赖历史的教师；clean RGB配对树也不能冒充教师实际使用的RGB图像。B／C／D学生优化、低光、四任务、原生Bayer、跨传感器和实机均未执行；正式学生训练仍需具体新授权。

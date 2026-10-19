# 2026-10-19 · N04 S2 数值稳定性与目标缺失修复 v5

## 当前状态

用户要求修复已经确认的训练故障，继续由 Codex 直接运行训练和监控。v4.2 LIBERO 已主动暂停在 1288 次更新（最近持久检查点1280），RoboTwin 早已在363次更新后异常退出（持久360）；旧权重、冻结源码、曲线和失败记录均保留。v5两平台完整真实零更新、连续8次完整batch8优化、独立确定性GPU重载及第9次更新均已通过。最初编译dev重载失败记录保留；修正参考计算后保持原验收阈值。两平台正式监督作业已从本轮step9恢复，并真实更新、保存至step10，继续向各2000更新运行。工程恢复不代表长期稳定、收敛或机器人能力通过。

## 身份与复现

- 实验ID：`2026-10-19_n04_stability_repair_v5`；负责人 Codex / lzh；时区 Asia/Shanghai。候选[N04](../candidates/N04.md)，当前仍为S2实现与优化诊断，未进入S3，机器人能力未验收。
- checkout：`/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`；分支`lzh/n04-egomotion-separation`；基点`a81cf25148870fd83fbc06a4fabf374510493978`。工作区dirty，未提交或推送；完整tracked diff、107个相关untracked实现/配置/测试文件清单和快照已留档于`outputs/n04-stability-repair-v5-20261019/`，校验见结果目录`worktree-and-tests.json`，不能只靠此基点复现。
- 实现`n04-v1.1-s2-stability-repair-v5`，损失`v1_1_s2_normalized_latent_bounded_transition_v5`。不恢复v4/历史短程原型权重；v5 fresh seed101。
- 配置`configs/n04/v1_1/codex_v5_{libero,robotwin}.json`；最终解析内容保存于各run的`config.used.json`，保留include链。完整模型、官方数据、batch8/micro_batch1、200步预热和2000更新计划不变；动作/对应/预测/表征外权重1/0.1/0.1/0.1。
- 冻结模型及零更新源码：远端`/workspace/owac/research/n04-v1/source/n04-v11-codex-stability-v5-20261019`。最终连续恢复与正式运行源码`source/n04-v11-codex-stability-v5-deterministic-dev-20261019`。中间`stability-v5-verified-20261019`只修正验证脚本终点保存/dev返回值读取；最终源另修复固定dev的计算路径。AST核对训练器只有新增`_fixed_dev_math`和修改`evaluate_dev_action_loss`，训练step/前反向AST及所有模型文件与真实零更新验证一致。各冻结目录均保留，不能热改；765文件清单与本地源码核对另存。
- 数据：官方RawVLA-Bench，`/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`；清单`research/experiments/results/2026-10-14_n04_manifest/splits.json`，沿用既有source-disjoint train/dev/final成员与哈希，不新增或重新挑选dev。真实存储为作者三通道HDR-derived/pseudo-RAW `rawvla_rgb3`，不称原生CFA。未加入S0/S1扩展、RGB教师或未来专家动作。
- 时间/容量：16帧burn-in＋32梯度帧，每窗4个动作监督时刻，forecast horizon1/2/4；ConvNeXt-B、完整RAFT 12迭代、每相机128支持、6块Perceiver/256×2048读出和真实π0.5，后端动作专家及LoRA训练范围保留。窗口与episode边界reset，辅助目标不改变因果输入可用时刻。
- 环境：沿用`envs/n04v11/bin/python`与StarVLA固定部署；PyTorch2.14.1+cu130，两张RTX PRO6000 Blackwell各95.6GiB，物理GPU0/1对应LIBERO/RoboTwin；BF16骨干/后端，FP32状态/协方差/似然与最终prefix投影，OMP/MKL/Inductor编译线程各4。本轮环境回执`20261019-v5-environment.json`确认Python3.12.3、StarVLA HEAD `57d508cb180d0510ac9060358009c0f6ffaac420`（仅未跟踪PROVENANCE_N04.md），并记录split与两平台规范化统计文件SHA256；环境历史明细另见[v4记录](2026-10-16_n04_aux_repair_v4.md)。

## 真实故障与定位

旧训练原始进度、Supervisor状态和本轮同checkpoint固定dev窗口诊断共同支持以下结论：

| 故障 | 原始证据 | 对训练的影响 |
| --- | --- | --- |
| LIBERO对应彻底越界 | checkpoint1280两段48帧dev窗口，256²图上最大flow约814–856像素，匹配/支持有效覆盖0 | forecast有效目标0，178个成熟anchor均`no_valid_support`；14个窗口尾端目标正常因`window_too_short`跳过 |
| 目标静默消失 | 正式日志forecast从580开始无值；correspondence最后584、583也曾缺失 | 后续700余次更新仍继续动作/表征，不能解释为完整四目标训练 |
| 潜在量纲增长 | LIBERO latent RMS约50–55；RoboTwin360约102–108 | 目标/状态/方差共同漂移，原始NLL曲线不能单独判读预测改善 |
| 动态算子允许扩张 | 同一权重真实局部线性A的谱半径约1.02–1.06，保守范数上界约1.28–1.50 | A反复传播可放大状态/协方差；原constant_velocity初始A本身也不是收缩算子 |
| RoboTwin数值保护退出 | 363更新后实际触发`forecast innovation variance exceeds configured stop limit`；最近持久360 | 2026-10-17 19:25:38 CST退出，GPU1闲置。不是仍在训练或正常完成 |
| 状态显示依赖残留PID | 原进程已死，train.pid仍存在 | 可能将退出或复用PID误作活作业 |

本轮诊断读原冻结v4.2源码和原检查点，不更新参数。RoboTwin checkpoint360固定dev窗口没有直接触发1000阈值；这不是对实际失败训练批的逐步重放，实际崩溃事实来自原始训练日志。相关JSON与停止回执见[结果目录](results/2026-10-19_n04_stability_repair_v5/)。

## 修复规则与代价

1. **固定观测量纲**：每个特征网格位置沿256通道做FP32非仿射RMS归一化（L2 normalize后乘√256），在measurement融合后执行一次。状态、质量头、readout与停止梯度的未来目标使用同一结果。保留方向信息和梯度，移除通道共同幅度这一自由度（退化近零向量用epsilon保护），原始RAW/统计不改；这改变潜在表示的量纲和优化配方，不能直接拼接v4/v5损失。作者RKN支持特征归一化，见[原编码器](https://raw.githubusercontent.com/ALRhub/rkn_share/master/rkn/Encoder.py)；本项目RMS=1缩放是明确的适配，不冒充完全相同的作者设置。
2. **约束真正组装后的动态算子**：包括恒等项、观测/隐藏四块，使用`||A||₂ ≤ sqrt(||A||₁ ||A||∞)`的可微保守上界，将A整体缩放到上界≤1，同一A用于均值和协方差传播；banded和dense原生回退均实现。使用identity初始A，所有残差basis仍可学习。不是只约束某个basis或对最终方差/损失截断。该规则排除A本身的指数放大，不保证带控制/Q/测量修订的整个滤波器有全局有界状态或校准方差；原数值停止保护仍保留。
3. **对应越界不能成为零目标**：在原photometric/census/smoothness/selfsup之外增加端点超出真实图像边界的几何代价。x/y分别按实际W/H归一化，以真正原始source有效mask归约，不受预测visibility控制。越界匹配仍无效，不夹flow、不把padding当测量、不伪造遮挡监督。只有raw源有效时该项才有支持；轻微真实边缘越界代价小，完全逃出图像仍有修正梯度。这是本项目正则项，不能称SMURF官方损失。
4. **禁止缺项后继续更新**：每个完整batch汇总四项实际参与的window数和有效项数，写入`objective_coverage`。开启`loss.require_all_objectives`后，任何正权重目标在整个batch均无有效参与，就在optimizer/scheduler更新前清除累计梯度并报错。不是要求每个窗口都有相同有效数；窗口尾端不能形成horizon仍按原协议诚实跳过。原精确归约/原生回退保留。
5. **监控和恢复**：作业状态必须匹配真实活进程及精确输出目录参数；残留或复用PID、诊断路径子串不作正式进程。页面显示四目标参与情况，历史辅助项超过50更新未出现时明确提示缺项。新Supervisor从`--resume latest`选择最高已完成数字checkpoint，排除best/临时文件，没有保存时拒绝静默fresh启动。新旧run曲线分开，验证阶段另有进度条。

## 设计、反证和验证口径

本轮假设是上述工程规则能阻止已复现的目标消失/算子扩张并支持真实联合优化。支持判据：数值参照与梯度检查通过、真实完整模型四目标及强制原生精确反向均正常、原batch8连续更新四项不缺失、保存后完整GPU重载固定dev一致、下一更新继续有效。若仍缺项、数值保护退出、重载不一致，则保留失败并停止对应作业，不把部分通过当成功。

这不是新机制因果比较或机器人增益实验，没有更换动作/任务信息、增加数据或执行新的闭环。只用既定dev作修复诊断，未动final成员；单seed、少量dev序列和短程验证限制均保留。算子谱约束和几何边界正则可能带来建模偏置，最终需与原方案及直接基线在受控预算下比较；本轮不凭修复自动裁定N04机制成立。

本地检查：确定性dev修补后相关测试467 passed、1 skipped（本机缺websockets）；`make check`通过（314文件）；监控15项测试通过、JavaScript语法通过。涵盖全矩阵/带状协方差与参数梯度参照、256步扩张反例、缺项不得更新、真实坐标单位、越界raw有效性和非零梯度、RMS尺度不变性/方向梯度、目标共享路径/停止梯度、最新checkpoint恢复与PID误判等。日志保存在`outputs/n04-stability-repair-v5-20261019/`；小型校验值另存结果目录。

两平台完整真实零更新验证均`complete=true`，optimizer更新0，分别检查action/correspondence/forecast/representation/joint_exact的实际梯度和参数未改；联合目标明确`gradient_eager_fallback=true`。跨8窗口的表征统计、48帧真实完整模型和π0.5均参与，不是stub。LIBERO/Robotwin实际forecast有效项178/267，correspondence32，action4；具体收据另存。

同一故障权重的零更新反事实：RoboTwin四步S_max诊断均值约473.65→1.2358，latent RMS→1；新预测MSE0.6378仍高于保持源状态0.1099。LIBERO原损坏flow仍无有效匹配，新增几何代价约4.6687且有梯度，forecast仍0，mandatory guard会阻止续训。这说明修补改变数值行为和失败响应，不说明同一单位下的预测质量提升，更不支持继续使用故障权重。

## 固定验证与重载验收发现的第二个问题

原完整8次更新两平台均四目标有效，但最初的严格重载验收实际失败，原始`continuous-verification.json`保持`complete=false`并保留错误：LIBERO 0.9221061170→0.9234932959；RoboTwin 0.3115729243→0.3124991953。没有通过放宽1e-5阈值把它改写为成功。

LIBERO只读同一checkpoint的对照确认：653个encoder和662个backend状态项逐值、dtype一致，终点dev前后持久状态也无变化；同一实例同一固定随机种子重复计算，原编译路径仍在0.92089468—0.92291814之间波动，native但保留benchmark设置也有约4.4e-5变化。这说明旧“固定dev”不是严格可复现参考，存储/加载遗漏未复现。尚未逐算子隔离哪一项CUDA实现主导差异，不冒称已经确认某一个kernel的内部原因。

最终固定dev统一使用原生前向、确定性算法、关闭该参考路径的TF32和cuDNN benchmark，CUBLAS使用明确的workspace配置；训练前反向仍用原编译路径。评估结束或异常都恢复原编译方法、精度/确定性设置、训练模式及Python/NumPy/Torch/CUDA随机状态。新增异常恢复和三种随机源隔离测试；本地467 passed、1 skipped，make check通过。不是将训练改成FP32或全程关闭编译。

| 原权重step8，无优化更新 | 两次独立模型构建，各重复3次 | 最大差异 |
| --- | --- | --- |
| LIBERO | 全部0.9205259084701538 | 0 |
| RoboTwin | 全部0.3108634054660797 | 0 |

保持原1e-5验收阈值。集成新路径后已再次加载原step8，逐项核对两平台653个encoder/662个backend状态及scheduler，与上述独立参考完全一致，并完成第9次真实更新。原8次更新的配置/训练目标不变，不重新计数；旧编译验证值保留在初始段原始日志，新确定性dev参考及其fresh seed101初始基准单独记录。各曲线中的验证数值计算路径变化需随报告披露，不能把小差异解释为学习收益。

## 命令、保存和连续验证

工作目录必须是冻结源码根目录，split_file为相对路径；环境变量按`configs/n04/v1_1/codex_v5_validation.supervisor.conf`和`codex_v5_training.supervisor.conf`；固定CUDA验证要求启动前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`（Supervisor已包含）。`CUDA_VISIBLE_DEVICES`按物理0/1设置，模型内部均`cuda:0`。Supervisor进程独立于SSH/浏览器连接，失败不自动重新训练。

```bash
# 完整真实零更新验证（每平台独立输出，不覆盖历史）
/workspace/owac/research/n04-v1/envs/n04v11/bin/python -u scripts/n04_full/verify_n04_auxiliary.py --config configs/n04/v1_1/codex_v5_libero.json --output /workspace/owac/research/n04-v1/runs/v1.1/codex-diagnosis/v5-zero-libero-seed101

# 完整batch8、原200预热/2000调度，8次更新后重新构建模型、严格重载和第9次更新
/workspace/owac/research/n04-v1/envs/n04v11/bin/python -u scripts/n04_full/validate_n04_continuous.py --config configs/n04/v1_1/codex_v5_libero.json --output /workspace/owac/research/n04-v1/runs/v1.1/codex-diagnosis/v5-exact-libero-seed101 --updates 8
```

RoboTwin替换平台名，seed不变。连续段未改batch、预热、学习率、训练成员、窗口和损失；只有执行循环暂时止于8，scheduler仍以2000为预算。成功后完整保存目录及回执可以继续为同一v5训练的初始段，不能混入先前batch2/warmup4原型。

8次初始更新由`owac-n04-v5-validate-{libero,robotwin}`完成；其原严格验收失败记录保留。`owac-n04-v5-recover-{libero,robotwin}`已完成新的严格确定性参考验收及第9次更新并正常退出，回执为`continuous-recovery-verification.json`的complete/reload/next_update及checkpoint。已在通过后保留全部初始段证据，续入`runs/v1.1/codex-s2/{platform}-D-stable-v5-seed101`，正式服务名`owac-n04-v5-{platform}`，最高数字保存自动恢复。

## 资源、结果及限制

当前未测到v5稳态速度、完整长程损失或闭环任务成功率。原batch8串行窗口和表征统计前向仍有计算成本；没有声称GPU充分利用或本轮提速。参数/模型规模未缩小，数据加载沿用逐窗GPU预加载与下一窗口CPU线程预取，没有新建全量Python RAM缓存；不把低显存或单次GPU利用率样本当最终瓶颈结论。此前实际8窗读取仅LIBERO0.527秒、RoboTwin1.517秒；见[数据加载与N02工作量比较](2026-10-16_n04_aux_acceleration.md)，该数据为10月16日历史测量，不冒充本轮速度实测。正式运行后以每更新端到端秒数、活跃目标与固定dev值记录效果，不能以显存占用替代计算吞吐。

网页继续使用[训练状态／速度／效果](http://127.0.0.1:8090/)，本地systemd SSH转发与远端独立监控保留。尚未验证的长程稳定性、保持基线之外的预测增量、完整表征可辨识性和机器人闭环效果需后续实际证据；旧0/100开发闭环保留。

### 实际恢复与最新回执（2026-10-19 11:12 CST快照）

| 项目 | LIBERO | RoboTwin |
| --- | ---: | ---: |
| 原完整batch8更新 | 8/8，四项均有效 | 8/8，四项均有效 |
| 独立确定性模型/重复dev | 2个模型×3次，差异0 | 2个模型×3次，差异0 |
| 集成程序严格重载 | 与0.9205259085参考完全一致 | 与0.3108634055参考完全一致 |
| 重载后真实第9次更新 | 完成、保存 | 完成、保存 |
| 第9次四项有效window | 均8/8 | 均8/8 |
| 第9次action/corr/forecast/repr有效项 | 32 / 256 / 1424 / 8 | 32 / 256 / 2136 / 8 |
| 正式Supervisor | RUNNING，PID782451 | RUNNING，PID782631 |
| 正式继续优化/持久保存 | 完成11，持久10 | 完成10，持久10 |
| 确定性fixed dev，计划第10步 | 0.95582658 | 0.29078390 |
| native fresh初始参考 | 1.18850359 | 2.45333278 |
| 当时每次更新用时 | 185.69秒（续训2次cycle均值） | 291.25秒（首次续训cycle） |

秒数包含该cycle开发验证/保存，首次还有编译预热；样本少，不能称稳态提速。第8步原生dev仅用于重载一致性旁路检查，**不参与按预定cadence=5的正式选模**；前8步原始验证仍为旧计算路径。第10步按新固定参考正式验证：LIBERO相对其第8步旁路诊断反而变差，不能把恢复工程通过写成预测/动作持续改善。当前四步预测MSE/保持源状态MSE约1，尚未证明有实用预测增量；两平台固定跨序列表征std约0.0499/0.0564，未证明表征目标已学好。仍在200步预热早期，未执行新闭环。

两平台`continuous-recovery-verification.json`均complete=true，原`continuous-verification.json`的complete=false和数值差异未改写。正式`promotion.json`记录初始9次更新、配置/权重/调度器身份、checkpoint元数据和初始metrics SHA。整个初始段移至正式run，旧诊断路径保留兼容symlink；`initial-segment-metrics.jsonl`和`initial-segment-train-report.json`保留原记录，不能把其中循环上限8误作正式2000预算或额外9次新实验。验证用step8/9和早期best权重以`checkpoint-verified-initial-*`保留，滚动数字checkpoint的对应映射在promotion中；原checkpoint-final-step8改名保留。初始GPU数值控制和未通过验收也独立保留。

`running-receipt.json`另确认两份step10 checkpoint均在独立CPU进程mmap读取成功（每份653个encoder键、662个backend键、7组optimizer、scheduler和v5身份）；这项只证明容器/元数据可读，不冒充GPU重载step10。实际GPU严格重载针对step8，正式服务实际加载step9后完成step10。网页已实查两平台正常训练、最近保存10、四目标4/4，修复试验横幅已结束；本地systemd隧道active，HTTP200。监控状态不依赖浏览器保持连接。

当前每平台预算仍2000、seed101、batch8、warmup200，不自动进入S3或扩展100k。GPU忙碌度仍偏低，跨窗口合批/调度瓶颈未解决；本轮不声称充分利用PRO6000或得到机器人收益。

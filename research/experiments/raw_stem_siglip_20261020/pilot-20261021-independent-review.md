# OWAC-D1.1 B/C50 pilot：独立验收与负结果

2026-10-21，Sol。**B/C各完成50次真实优化更新、800次anchor呈现，工程与重载验收通过；第50步验证token误差改善，但前5步动作误差变差，未证明正常光控制恢复。学生闭环0场。** 所有本轮模型作业已结束，不继续D、有限视觉适配、弱光、闭环或实机。

本轮根目录：`/home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725`。完整机器证据见[本轮摘要](pilot-20261021-evidence.json)、[实际最终验收](/home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/sol-final-acceptance.json)、[独立权重审计](/home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/sol-final-checkpoints.json)、[离线分母重算](/home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/sol-offline-aggregate-audit.json)。此前进行中记录保存在`working-progress-history.md`，原始失败／stderr／事件未删除。

## 授权、实际资产与初始化

本轮用户明确说“授权启动训练。”，按前次最小提案执行：B/C各**累计**≤50更新、有效batch16、各≤800 anchors，合计≤4 GPU小时／新增≤20GiB。首步计入50，恢复后不重置计数。旧N01/N02/N04和教师阶段已耗尽预算不恢复；没有额外下载。`user-authorization.json`、`authorized-config.json`、两臂authority和启动收据落盘。已写`AUTHORIZED_SCOPE_COMPLETE.json`，本轮更新预算已用完。

实际任务为Normal `libero_goal/task6`，`put the cream cheese in the bowl`；50源示范5349帧，train40/validation5/test5。按源拆分保持不变，全部优化anchor仅来自train40；测试5源未用于训练、选型或评估。复用训练＋验证45源969条教师缓存（约4.894GiB），667个监督候选anchor，未生成新测试教师目标。每臂800呈现覆盖667独立anchor，其余为第二轮重放；上下文、重复burst不当作新独立样本。

数据为LIBERO RGB8逆处理的**三通道pseudo-RAW**，不是原生Bayer。当前教师仍为本地作者架构**初始化0前端＋当前实际π0.5**，从RAW进入动作；前阶段共同20初态教师／参照各20/20仅开发筛查。不是作者公开训练后前端，也不把旧155次教师更新称为收益。原教师内部RGB仅用于B诊断输入和教师目标，C部署不产生RGB。

继承的是同一实际策略checkpoint，SHA `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8`。实测几何：224输入、patch14、16×16网格、hidden1152、原投影2048；两个活动相机按base_0→left_wrist顺序各256 tokens，失活right_wrist保持原mask语义。位置映射回256全视野；没有追加隐藏可学习投影。

C使用6个因果RAW源索引、端点stride5、相机独立递归patch状态；源起点reset，当前权重重放完整前缀burn-in，训练展开6端点。输入burst与展开长度的两个6不表示36个独立观测。B每端点取同RAW教师内部的一张RGB，同时使用同容量stem／状态模块；其RGB已含ISP时序信息，披露中间信息压缩差异，不能将B/C差异只归因RAW。缺失的真实曝光／传感器时间元数据不虚构。

## 实际训练范围与可恢复性

B/C均为特征蒸馏，`Ltoken = FP32 masked MSE + 0.1 cosine`，**没有动作损失更新**。各更新5个张量，共**14,454,784参数**：stem两层卷积的weight/bias及1152维gate。27个SigLIP blocks、位置编码、post-norm、继承原投影、语言／动作后端及教师保持冻结；学生梯度穿过冻结视觉塔，未整体no_grad。教师eval／停止梯度。未压缩视觉塔容量。

配方为AdamW lr1e-4、wd0.01、clip1、warmup100、seed101、microbatch1累计至16。50仅走到100步warmup的一半，不代表收敛。首个实际使用lr1e-6；进度文件lr字段是scheduler之后的**下一步**lr，不冒充本步值。

Sol亲自CPU比较实际0/1/50文件：5张量均有真实差值，437项继承视觉状态逐项完全相同；两臂初始5张量SHA也完全相同。保存／重载时完整教师、后端、视觉冻结SHA均为`a5cd67134bf730b58e5bf71e6c2b96a890308dda86b8970185e5d117506940f2`；此全量SHA为运行时对实际冻结张量计算，直接CPU逐项比较覆盖学生继承状态。optimizer各5组step50、scheduler last_epoch50，全部50批sampler顺序逐项复算；B/C顺序完全一致。首步optimizer／scheduler／CPU RNG精确重载，真实CUDA训练1→2批次与独立恢复结果吻合，CUDA RNG随真实恢复路径还原。具体见`sol-first-step-B/C.json`、`sol-resume-B/C.json`与`sol-bc-common-initialization.json`。

B first1→审计→resume50；C同样执行。B原生退出0。**C原监督终端返回143，独立训练PID2749243继续运行；Sol核对命令、PGID与start tick后以watch-only进程接管，没有重训、计数重置或终止信号。C原生退出码不可得，不能补写为0。** 实际50权重、800anchors、最终CLI `status=trained/updates=50`和完整审计证明本轮更新完成，异常保留在`reattach-*.json`。没有发现Harness执行终止C的调用；143原因未确认。

## 固定验证：表征改善，动作选型未恢复

五条源示范，102个因果窗口；首步／前5／完整10分母分别**102／505／985**。每checkpoint token分母52,224，MSE标量分母106,954,752；按有效token及动作数加权，不按示范均值直接平均。Sol从每源记录独立重算所有全局值。源ID＋端点＋seed101固定噪声，B/C与各checkpoint相同；不与前次未固定噪声的ALIGN数值直接作学习比较。`action_horizon=10/flow_steps=10/replan_steps=5`分别保留含义。

下列动作值为反归一化原控制器命令MAE7（库内称physical），不冒称米／弧度位姿误差。

| 臂 | 累计更新 | 验证token MSE | 首步MAE7 | 前5步MAE7 | 完整10步MAE7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | 0 | 4.158711 | 0.340686 | 0.300235 | 0.301160 |
| B | 1 | 4.140290 | 0.339937 | 0.298629 | 0.300344 |
| B | 50 | 2.350094 | 0.315641 | 0.312195 | 0.312815 |
| C | 0 | 4.007122 | 0.346356 | 0.305607 | 0.303655 |
| C | 1 | 4.002835 | 0.348850 | 0.306938 | 0.304388 |
| C | 50 | 2.765593 | 0.336108 | 0.327335 | 0.320514 |

从0到50，B/C验证token MSE分别下降**43.49%／30.98%**；前5步MAE分别上升**3.98%／7.11%**。每臂五条验证示范均为1条改善、4条回退，仅为离线误差，不是闭环成功／失败数。首步误差单独有所改善，不能替代实际执行前5步的指标。

按预先规则“验证前5步MAE最小、同分取早”复核：**B选择第1步0.29862856，C选择第0步0.30560670**；路径、权重SHA、数值与selection sidecar匹配。不能把C50冒充选定最优checkpoint，也不能把B的首步小改善称为控制恢复。

完整10步组误差如下，padding为补齐25维的归一化误差：

| checkpoint | 平移3维 | 旋转3维 | 夹爪1维 | padding25 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.374961 | 0.029501 | 0.894736 | 0.00105242 |
| B50 | 0.358931 | 0.028650 | 1.026959 | 0.00101319 |
| C0 | 0.388432 | 0.029546 | 0.871654 | 0.00107117 |
| C50 | 0.406966 | 0.028245 | 0.937967 | 0.00096410 |

B50的夹爪偏差增加；C50平移和夹爪均增加。padding改善不等于真实动作改善，不能用32维平均掩盖它们。

## 纯学生路径、状态与信息诊断

全新进程仅载入学生artifact＋相同冻结后端／tokenizer／统计量，禁止教师注册表、前端权重、缓存以及实际ISP模块；原视觉塔／投影释放，原embed_image调用替换为报错。C0/C50完整五源与逐源指标对附教师路径最大差**0**，预先容差1e-6。状态25快照恢复后到30的动作最大差**0**，C0状态进入C50明确拒绝。无teacher fallback、无特征混合，禁止访问记录0。

原StarVLA PI0顶部导入RAW工厂**定义**，构造无前端后端调用一次`build_raw_frontend(None)`。纯进程明确披露4个辅助定义模块，None工厂返回None，所有非None构造／成像辅助调用报错；实际`baselines.rawvla/darkisp/dayisp`模块和ISP实例均未加载。不能把这个定义依赖写成“完全没有导入RAW工厂”。初版过严守卫导致的CPU仅定义预检失败保留，之后真实独立运行通过。

C0与C50各用2源、端点0/5共4anchors／40有效动作步做同state／语言／噪声的零视觉与跨示范替换，分母完整。C50跨示范动作变化0.07785，大于C0的0.00913，说明更敏感，但C50跨示范前5误差0.29746反而低于原输入0.31995；零视觉0.32161。这**不证明正确依赖视觉**。目标位置干预not_run，不能声称视觉控制成立。

## 实测成本与证据边界

RTX5090，32GiB。B/C登记训练作业墙钟分别2326.75/2401.40秒（含加载、验证；C探测结束保守计时），合计约1.313 GPU小时。附教师诊断635.26秒、纯学生157.50秒；含全部离线进程模型作业合计约**1.536 GPU小时**。从第一次模型启动到离线完成保守计**1.594小时**，低于4小时；新增约**10.352GiB**，低于20GiB。旧缓存和教师制备／旧教师训练成本单列，不摊成学生速度收益。

逐更新采样最大GPU allocated/reserved：B 8.289/8.729GiB，C 10.488/10.922GiB；这些不是算子内峰值。CPU RSS采样最大B8.224／C11.750GiB。

纯C50含当前RAW内存组包、resize/H2D、10步flow及反归一化，20次独立replan＋2次warmup，p50 **206.93ms**（上中位数）、p95 **211.91ms**；峰值allocated/reserved **7.258/7.393GiB**；实际唯一参数**3,630,533,776**。不含相机采集／磁盘读取／冷启动，不宣称比教师更快。

| 层级 | 本轮证据 |
| --- | --- |
| 工程／来源／恢复 | 通过；72针对性测试、最终make check、研究台13测试；真实资产与零更新ALIGN接口此前通过 |
| 真实学习 | B/C各50，权重确实改变、留出token误差下降 |
| 离线动作 | 第50步前5与完整10误差回退；选型B1/C0 |
| 独立纯学生 | 实际重载／状态／无教师缓存路径通过 |
| 正常光闭环 | **not_run，0场；控制恢复未证明** |
| D、弱光、原生Bayer、四任务、实机 | **not_run／不在本轮授权内** |

## Harness、版本与实际命令

实际使用本机DeepSeek V41-Flash Max，7轮工作单／编码／返工，准确headless session `session-20cc796d-d7cb-4338-b059-bf8f9e8b28f0`，所有实际选择记录均`deepseek-official/deepseek-flash/max`。主要训练监督／梯度记录／验证噪声代码及评估脚本由Harness编写，Sol审设计、读真实代码与事件、退回错误实现，直接修少量接线阻塞并执行独立验收。沙箱CUDA不可用、评估脚本API／身份检查返工、源码摘要哈希失败及C监督终端异常均保留；Harness exit0没有当能力证据。`harness-ledger.json`单独记模型调用／token收据，不冒充API费用账单；编码与GPU训练分开记账。并行新增的Web默认规则与其他文档未回退；本轮实际派工按本会话原始headless任务单记录。

源码HEAD `9b7ad89961ea36c7f4e5530d278a513223c043ac`，main已有修改保留，没有提交或推送。**不能只按HEAD复现本轮**：实际382个冻结文件保存在`frozen-source-snapshot.tar.gz`，SHA `8b9be32c916cc75ae146e01ebf8a66921c4b5ff68533d95459eb25a9f72b25e1`；训练与评估脚本各有冻结清单，末次复核0漂移，源码diff及初始状态也落盘。

实际模型解释器为`/home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/runtime/bin/python`。以下监督命令每次只推进一个已授权阶段，实际执行4次，首步后由Sol验收文件放行：

```bash
cd /home/jeong/zeno/owac/repo
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/home/jeong/zeno/owac/repo/src \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
outputs/rawvla-teacher-20261020/runtime/bin/python -m owac.runtime.raw_temporal_pilot \
  --run-root /home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725 \
  --config /home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/authorized-config.json \
  --python /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/runtime/bin/python
```

真实阶段命令／PID／PGID见`launch-B/C-*.json`；C接管是`reattach-watch.py`，没有再启动train。离线实际通过独立owner运行`bash /home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/postprocess/run-offline.sh`，模型子进程和exit0分别见`offline-*.json`及各phase receipt；没有optimizer更新。配置中D的phase1／matchedCD路径不代表相应收据已经存在或D已获授权。监督状态现为complete；本轮权威收据不能被复用成新的50或下一阶段预算。

曲线：[训练采样指标](/home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/postprocess/run-20261021T035315Z/curves.png)、[验证动作与组误差](/home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/postprocess/run-20261021T035315Z/curves_validation.png)、[留出token](/home/jeong/zeno/owac/repo/outputs/raw-distillation-pilot-20261021-101725/postprocess/run-20261021T035315Z/curves_feature.png)，同名SVG可编辑。验证连线仅连接0/1/50三个实测点，不是连续验证采样。原始附教师／纯学生JSON、逐源离线账本与全部实际checkpoint／artifact位于同轮输出。

## 下一步最小行动

**2026-10-21后续用户指令已替代下面原提案：只做C单臂充分蒸馏，不再先做B消融或铺开对照；当前暂不启动，先审[实际框架](../../../docs/plans/RAW_TEACHER_STUDENT_FRAMEWORK_2026-10-21.md)。以下保留为本次pilot结束时的历史建议，未执行。**

本轮是明确的**表征学习通过、动作恢复未通过选型指标、闭环未测**的pilot；50尚未完成warmup，不能否定整条路线，也不应盲目扩大RAW臂或自动转D。先针对B做一个有界零优化归因：同一102验证窗口比较当前EMA与只取当前端点的输入侧时序处理，并检查夹爪速度场／归一化偏差；保留相同教师RGB、后端、噪声、相机与位置，说明不同历史范围。此项尚未运行，不冒充已定位根因。建议最多0.5GPU小时／新增1GiB、0优化、0闭环；根据结果再冻结一个最小下一轮训练变更与预算。**任何增加B/C累计更新、D、视觉解冻或闭环均需新的具体授权**，不重复询问本轮已完成的工程细节。

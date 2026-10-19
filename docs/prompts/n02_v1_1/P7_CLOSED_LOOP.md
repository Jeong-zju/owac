# 给DeepSeek：P7，全量闭环与机制因果验证

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

## 本阶段执行位置

优先连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，在远端 `/workspace/owac/worktrees/n02-measurement-completion` 开发、测试和运行。先完整阅读[远程开发执行说明](03_REMOTE_DEVELOPMENT.md)，按其步骤准备尚未建立的 worktree、解释器与 P0 资产；随后读取远端 AGENTS.md、README.md、docs/status.md。本文命令、相对路径及 `python` 均指远端 checkout 和对应阶段的独立运行时，GPU 命令显式使用 `"$N02_PY"`。

候选策略服务和仿真闭环均在指定远端运行，复用已有资产及已核对依赖，为 N02 设置独立作业名、空闲端口和结果目录。保留冻结基线服务；全量视频/轨迹在远端归档，本地只接收报告及抽样证据。

读[共同契约](00_SHARED_CONTRACT.md)、P5/P6正式交付、原方案P7/9，以及原始冻结评测实验。你的任务是实现并执行指定闭环评测、核对分母和给出实际结果；不由进程PASS推出能力成功。

主要新增`evaluation/n02_full/closed_loop.py`和配置，复用作者仿真/成功判据与现有评测身份逻辑。先读`src/owac/runtime/rawvla_full_eval.py`、`rawvla_robotwin.py`、`rawvla_eval_audit.py`、`rawvla_job_guard.py`，以及`tests/test_rawvla_full_eval.py`和`test_baseline_protocol.py`。原冻结基线源码/结果不覆盖。

## 1. 原协议主表

LIBERO40任务×50初态×5照明=10000；RoboTwin13任务×50dataset seeds×5域=3250。每平台每个训练方法3个独立正式seed；原则上每方法39750个测试身份。B0冻结现成checkpoint没有“3个独立训练seed”，按相同配对推理随机流作参考，重复推理的标签如实区分。

必保留B0 Default+原π0.5、B1作者ISP+原视觉塔、B2同Base直接RAW、B3开发集选定的最强完整机制、N02。A/B训练轨道分表，使用同预算同轨道比较。已经严格匹配checkpoint、配置、照明、随机流和代码的历史B0可以复用，必须保存等价清单；不匹配者不能直接拼表。

协议包括原camera顺序/分辨率、illumination/noise、语言与flow随机流、norm、动作horizon/cadence/denoise、最大步数、warmup/reset/成功判据。LIBERO10/5，RoboTwin50/50；Default RGB和N02 HDR-derived RAW分别记录输入来源。RAW3不能称原生CFA。作者论文某处horizon8与本地10/5另列差异，不临时改协议。

## 2. 实际在线状态与观察机会

环境每个控制步的时间索引记录，推理只消费冻结协议允许的相机观测；N02/B2/B3可用历史帧/曝光次数对齐。动作在此前已执行才可进入executed_action context。不能让N02每控制步额外观察，而B2仅每50步观察后仍声称同信息。

原协议主表保持固定规划cadence。机制扩展的事件重规划可在收到真实反证后中断未执行chunk，但给各方法相同观察/触发机会，另记控制成本和成功率，不并入原主表。已执行动作不会回滚。

每episode reset全部对象/粒子/账本/J/相机/前缀/chunk缓存。相同旧状态分叉、同初态与推理随机流；每个动作附input/frame/world/revision/model版本，可追到机制事件。禁止通过隐藏物体ID/接触真值为在线假设初始化。

## 3. 基础设施缺口不改变统计分母

使用唯一`platform/task/init_seed/domain/train_seed/method/protocol_version`作为identity，多次attempt不新增分母。有效结果包括success和正常策略timeout/failure；初始化/渲染/服务崩溃列infrastructure_failure。只按预注册恢复规则重跑原identity，保留原日志；未补齐报告incomplete，不将其删除后提高成功率。

主checkout的原条件632条核查与3250主清单独立。若需复用2026-10-13修复，先读取实际修复和控制输入parity收据，按同协议单独适配，不把632条合入3250。所有引用的服务器状态以本轮实际查询时间为准，旧本地RUNNING字段不是现时事实。

正式运行前只在开发初态做每方法的真实adapter smoke，确认原成功判据、动作数值、RAW相机和reset；smoke不占正式测试身份，也不启动第二套重复基线supervisor。

## 4. 单列机制扩展任务

固定四类：错误初始身份/关系；遮挡后实体进入；动作相关关系变化；一条支持失效但另一条有效支持仍存。机制测试场景与训练/开发不重合，每类至少1000独立场景目标，预先固定可行的任务映射和统计单位。

物理状态/目标改变后重新生成专家与任务成功规则；成像扰动与物理变化分开。隐藏真值只给evaluator；误导先验是明确干预，所有方法同样接收。至少包含发现有无、只降权/全reset/trace重放、G学习/随机/全连/上界图三类因果对照。

每个失败案例保留RAW测量索引、提议/接受/撤回、重建前后状态、动作chunk及真实任务判据，区分没发现、误撤回、旧hidden污染、policy没有利用新表示、预算超限与仿真异常。视频为辅助，计分依赖任务judge与原始轨迹。

## 5. 统计和结论

先逐平台、域、任务、训练seed给原始成功/正常失败/未完成数。样本微平均与任务宏平均预先定义；主报告不在一个不完整平台上给合并优越结论。配对bootstrap以task/初态簇为单位，照明变体同簇；报告每训练seed以及跨seed不确定性，不能把高度相关帧当样本。

预注册系统目标：相对同预算强前端总体成功率提升≥3个百分点，配对95%区间下界>0；Normal下降不超过2个百分点；低光与强光分报。没有整体提升但特定机制有效时只报告限定结论。

完整N02成立需要P3发现、P4撤回、P5超越强机制基线、P7操作结果同时支持。单纯Base更强、训练更长或输出更平滑不能归因于结构修订。局部提速独立评价，使用全局版本也可有行为收益。

最终交付完整身份ledger、attempt历史、模型与协议哈希、原始任务结果、逐环节p50/p95与显存/状态字节、动作/机制日志、分组统计和中文“主张—证据—限制”表。归档研究结果而非修改旧分数；提交/发布由用户另外安排。

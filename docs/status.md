# OWAC 当前状态

2026-10-22 19:00 实际运行：全量RAW教师2163源/74129窗口已完成native0、测试目标0。学生在137完整状态重载后已实际推进210更新/3360呈现，双卡global16、全视觉头训练；只读RAW映射生效，无新增OOM。SwanLab实际同步210且无上传错误：https://swanlab.cn/@Jeong/OWAC/runs/owac-all-bench-20261022-attempt2 。训练仍在运行，首遍验证未完成，不宣称泛化/控制恢复。源码e658cd6，独立报告见research/experiments/raw_stem_siglip_20261020/all-bench-20261022-independent-review.md。仅380GiB磁盘上限，其他人为资源门槛移除；无定时任务重建。


2026-10-22 数据I/O进展：OOM修复后全视觉头已真实推进137更新并完整保存/native0暂停；同一DeepSeek实现未压缩NPZ内部RAW帧只读映射，避免整条episode匿名RAM复制，不改数据/模型/全局16/因果历史。Sol独立复核，累计43项相关测试及make check通过，按137优化器/采样器/RNG完整状态续训待实测；教师2163源/74129窗口复用，测试目标0。

2026-10-22 全量实验实时修订：教师2163源/74129窗口已全部生成native0。首次学生作业至少10更新后被容器116GiB配额OOM杀死，证据/初始模型/日志完整归档，不重算教师、不声称恢复未保存优化器。已由同一DeepSeek修复RAW缓存配置接线并改用每卡16源工作缓存，保持global16及完整历史，无额外内存/显存/时间停止门槛；第1实际更新即完整保存。Sol独立复核，累计42相关功能测试及make check通过，修复后作业从相同初始化重新启动待实测。

2026-10-22 最新用户授权：在现有SSH服务器新开全量RawVLA-Bench蒸馏实验，不改学生/教师/损失算法，训练整个学生视觉头。只保留新增硬盘380GiB上限；用户明确撤销GPU时间、显存、利用率等人为门槛及强制停止要求。全量只读共享数据已清点：2403轨迹（LIBERO1771、RoboTwin632）、408075帧、5照明；源轨迹拆分1923/240/240，先训练3遍训练集，global16双卡各8。旧实验/预算/结果保留，新实验不受旧3116更新上限约束；无动作损失/后端训练/闭环/实机授权。当前全量数据/双三相机接口由同一DeepSeek V41-Flash Max实现并经Sol独立复核，40项功能测试及make check通过；双卡教师目标已真实生成中，学生仍0更新，已配置目标完成后自动衔接3遍训练，无终止失败自动重试。监控http://127.0.0.1:8880/，SwanLab新run owac-all-bench-20261022。不能宣称充分训练或控制恢复。实验根outputs/raw-distillation-all-bench-20261022，临时tokens缓存/root/autodl-tmp/owac-all-bench-20261022。


2026-10-22 当前阶段已完成：全视觉头428579952参数累计3116新更新/49856anchors，完整checkpoint_final_003116，native0；此前阶段合计达到原5000更新等价上限。global16=双卡各8真并行，accum1；每步中位0.454秒，吞吐约旧串行4.5倍。当前蒸馏训练/教师缓存/状态重载/导出/配对评估链路的hash与无意义扫描/冻结审计/数值门槛已清理并同步服务器，123功能测试+makecheck通过。SwanLab私密OWAC已完整同步并结束：https://swanlab.cn/@Jeong/OWAC/runs/owac-fullvisual-shared-stage 。最终train loss0.07365，最佳val1.00053@1000、最终val1.00837，未作test/动作诊断/闭环，不宣称泛化或控制恢复。无预算扩展或定时任务重建。详见research/experiments/raw_stem_siglip_20261020/batch8-cleanup-20261022.md。

2026-10-22 当前运行已切换真batch8：从858完整状态恢复859并持续超过1250步，全局16=2rank×8、accum1，FP32权重/Adam+BF16 autocast。移除每步全参数/梯度审计及RAW范围/finite全扫描，status/GPU gather每10步。稳定host每步中位0.451秒（此前约2.04秒），25秒GPU均值85%/84%，不是收敛/控制结论。SwanLab私密OWAC已上线且持续回填/更新：https://swanlab.cn/@Jeong/OWAC/runs/owac-fullvisual-shared-stage 。导出/teacher-cache冷路径hash残留继续清理，本轮无新GPU benchmark/准入门槛。原3116新阶段/累计5000更新等价和资源预算不变，旧定时任务不恢复。

2026-10-22 用户明确要求清理蒸馏整条代码链的无意义防御性编程：取消Sol附加的数值阻断门槛，停止反复GPU对比测试；移除哈希、重复张量/参数/梯度扫描与冗余审计，不另造预检门槛。保持训练数学、因果/reset、完整恢复和资源上限。当前训练安全暂停于858更新，下一步使用双卡每rank8真batch、全局16和原BF16恢复；SwanLab私密OWAC项目监控并行接入。未授权其他研究阶段或预算扩展。

2026-10-22 最新批处理授权：保持全局batch16，优先双卡每rank8样本真正并行。旧串行路径已在858更新完整保存并native0退出；优化器、采样器、累计预算保留。DeepSeek V41-Flash Max已实现并通过8项工程测试和make check；真实双卡零更新数值/梯度/显存/吞吐验收进行中，未通过前不宣称切换完成。无新闭环、动作损失或预算扩展。

2026-10-22 哈希清理及性能修复已实测验收：当前训练执行路径不再计算/校验哈希，保留形状/有限值/相机/因果/reset及恢复计数检查。55步完整暂停后，双卡55→56更新/保存和重载验收均通过，连续训练已恢复并超过94步；采样器/优化器/预算未重置。RAW10源RAM缓存、静态packet每rank≤1.5GB缓存启用。稳定每步中位18秒→约2秒，两卡35秒平均利用率约8%/24%→44%/47%；不宣称持续满载。监控http://127.0.0.1:8879/。详见research/experiments/raw_stem_siglip_20261020/throughput-nohash-20261022.md。


2026-10-22 用户最新明确要求彻底移除本轮蒸馏链路中的所有哈希计算及校验。该明确指令覆盖历史资产/权重/状态/缓存/检查点及源码启动哈希要求；保留直接张量形状/有限值、相机/时间/分区/reset和优化器/采样器恢复检查。历史记录不重写，旧hash字段仅兼容读取而不校验。训练为性能修复已在新55步/880anchors完整保存并正常退出；从55恢复，不重算、不重置预算。实现由同一DeepSeek V41-Flash Max负责，Sol独立验收；未完成清理前不宣称已移除。


2026-10-22 15:10 远端整视觉头连续蒸馏已实际运行：从独立验收的全新step1完整状态恢复，实测已到新阶段4步/64训练anchors；Sol核对torchrun PID55499/startticks17190994/PGID55499及两rank进程，native exit仍为unknown（存活中）。严格双卡重载native0且0额外更新通过。仅共享RAW→新教师targets→全学生视觉头428,579,952参数，固定8/2/2、test不参与；教师/语言动作后端冻结。新监控http://127.0.0.1:8879/。最多3116新更新及剩余资源上限，既有定时自动化未恢复；未证明收敛或控制恢复。


2026-10-22 共享RAW整视觉头最新进展：远端直接只读共享task6 Normal12源，固定8/2/2；全新204行RAW教师目标已逐行及状态链独立验收，test0行。双RTX5090真实零更新梯度预检、1次全局16anchors优化更新及完整检查点独立验收通过；442视觉状态张量均实际更新，可训练428,579,952参数，教师与语言/动作后端冻结。重载检查进行中，尚未宣称长跑或收敛；新监控http://127.0.0.1:8879/，旧8878为停止历史。见research/experiments/raw_stem_siglip_20261020/full-visual-shared-20261022-independent-review.md。


2026-10-22 数据来源最新更正：用户明确要求“共享盘直接读RAW，在远端重新生成匹配的教师目标，再训练整个学生视觉头”。旧50条RAW及旧token缓存传输已停止，不作为新阶段训练数据；共享根目录/root/autodl-pub/datasets/rawvla按只读处理。task6正常光实有12条，采用固定源示范8/2/2拆分，仅train+validation生成新的RAW教师targets，test不参与训练或选型；不加入其他照明条件。整视觉头新阶段仍从0步开始，继承实际π0.5视觉权重、重置stem/gate；教师及动作后端冻结，资源总预算不扩展。工程和真实GPU预检进行中，尚无新优化更新。

2026-10-22 最新用户授权：本机stem-only蒸馏已按用户要求停止于1872步，定时跟进已删除。新阶段迁移至用户指定SSH实例101.71.76.104:27786，使用两张RTX5090；用户明确不从旧学生检查点续训，重新初始化RAW stem/gate，仍继承当前π0.5策略的位置编码、SigLIP与原投影，并训练整个学生视觉头（428,579,952参数）。教师与语言／动作后端冻结，仅token蒸馏，无动作损失／闭环／弱光／实机扩展。新建阶段与优化器，旧实验保留；当前工程实现、传输与预检进行中，不能称已启动或充分收敛。暂沿用原总预算剩余额度：最多3116次新全局更新／49856 anchors、140841.2632 aggregate GPU秒，双卡按两卡合计，新增训练输出剩余15396584412字节。实际资产／派工／授权记录见outputs/raw-distillation-full-vision-20261022/。本条取代历史stem-only当前范围，不改写其事实。

2026-10-21 最新授权：用户明确“可以开始充分蒸馏，并给我一个简单的监控页面来实时查看进度和效果。”本阶段仅C单臂RAW→RAW投影后视觉token蒸馏，从真实C50续训，累计最多5000更新／80000 anchors；新阶段最多64 GPU小时／20GiB。优化器、scheduler、RNG、sampler及累计50步必须恢复；只更新现有stem＋gate，继承视觉与后端冻结。按固定验证集token MSE＋0.1cosine选型，动作误差仅诊断；无B/D扩展、视觉解冻、闭环或弱光授权。当前C51真实首步已独立验收，连续蒸馏已启动并真实恢复到后续更新；实时进度见[本地监控](http://127.0.0.1:8878/)，每5秒刷新。验证token损失每250步检查；后台每20分钟核查，正常无异常时保持安静。闭环0，控制恢复未证明。派工与运行收据：`/home/jeong/zeno/owac/repo/outputs/raw-distillation-c-full-20261021-123313`。此条取代此前“暂不启动”的当前状态，历史结果不变。

2026-10-21 后续方向：只继续C单臂RAW→RAW视觉token充分蒸馏，不再默认开展B/D多轮对照；用户当前明确暂不启动、先审框架。教师／学生实际结构及候选预算见[框架说明](../docs/plans/RAW_TEACHER_STUDENT_FRAMEWORK_2026-10-21.md)。新训练authority未签发；投影后token选型尚待实现，既有pilot动作选型记录保留。



2026-10-21 研究统筹台全部 prompt 的 DeepSeek 编程默认改为完全权限（`danger-full-access`）与不询问审批（`never`）；可见 Web 会话、编程／闭环 key 分离继续沿用。新建／续接后由执行者核对实际权限并切换旧会话，不要求用户逐项批准；既定任务、预算与阶段状态保持。

2026-10-21 研究统筹台提示词统一改为可见 DeepSeek Harness Web 编程：工作单、回复、工具调用与返工均在用户可见专用对话中完成，不使用 headless 编程派工。编程／闭环 key 继续分离；任务专用 Web 凭据文件与共用人工会话隔离。此次只更新工作方式与提示词。
2026-10-21 **B/C50 pilot已完成并独立验收：token对齐改善，前5步动作误差回退，闭环未测**。两臂各50真实更新／800 anchors，只更新14,454,784个stem＋门控参数；继承视觉、教师与后端冻结。选型B1／C0，纯学生重载通过；模型作业约1.54 GPU小时、新输出约10.35GiB。C原监督终端143、原生子进程退出码不可得，真实50权重与日志已审计；未重训。当前全部本轮模型作业结束，停止于已用完的50上限；D／视觉解冻／闭环／弱光／实机未启动。见[本轮独立报告](../research/experiments/raw_stem_siglip_20261020/pilot-20261021-independent-review.md)。


2026-10-20 DeepSeek 凭据按用途分离：所有 prompt 明确编程使用 `DEEPSEEK_CODING_API_KEY`、在线闭环使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`；本机私有文件由按角色启动入口读取，编码 Harness 显式绑定 coding 并禁用托管凭据回退。只更新配置与提示词，当前阶段与实验预算不变。见[使用说明](deepseek-credentials.md)。
2026-10-20 **OWAC-D1.1 ALIGN工程与真实零更新预检通过；STEM待新授权**。因果6帧／相机独立状态、同源SigLIP继承、完整前缀缓存和原生动作梯度、纯学生checkpoint0独立重载均已实际检查；训练40源＋验证5源共969条教师缓存。164项针对性测试通过，2项需toy AdamW更新的旧测试排除、0跳过，make check通过。真实学生0更新／0闭环，控制未恢复。建议下一步B/C各≤50更新、合计≤4GPU小时／20GiB；当前未签发authority。见[独立报告](../research/experiments/raw_stem_siglip_20261020/align-independent-review.md)及[命令/资源](../research/experiments/raw_stem_siglip_20261020/ALIGN_RUNBOOK.md)。

2026-10-20 限定教师阶段已验收：[独立报告](../research/experiments/raw_stem_siglip_20261020/teacher-independent-review.md)。Normal task6共同20新初态：参照20/20、RAW教师20/20，仅开发筛查。选定前端初始化0；实际155前端更新／1240 anchors未带来已证明收益，未作为选定权重。验证9成功＋1人为中断保留完整N10；50回放／50策略场次已用完。学生0更新／0闭环，尚未恢复；旧N01／N02／N04仍暂停，不授权学生／弱光／实机扩展。


2026-10-20 授权前教师取得补查记录：官方6公开分支／main32历史提交与公开主页未找到训练后的RawVLA前端；完整源码已取得并独立合并配置，后端已恢复但不等于完整教师。已写[限定单任务教师复现提案](plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md)，拟仅训前端、累计≤2000更新，执行待明确授权；0优化、0新闭环。保留因果burst／递归，不固定单帧。


2026-10-20 授权前工程交付记录：每帧 RAW stem／SigLIP 蒸馏基线与受授权门控的训练、纯 artifact 评估入口已可审阅；68 项针对性测试通过，实际同源后端恢复及合成输入数值／梯度核对通过。新增优化更新0、闭环0，控制未恢复。完整 RAW 教师权重、真实配对数据和时序训练端到端尚未验收；不固定最终策略为单帧。见 [独立验收](../research/experiments/raw_stem_siglip_20261020/independent-review.md)。


2026-10-20 当前实施路线收敛到[RAW stem＋继承 SigLIP＋蒸馏](plans/RAW_STEM_SIGLIP_DISTILLATION_2026-10-20.md)。用户认可方案，执行 prompt 与采用决议已落库；统筹台默认展示此主线、完整提示词和可点击的分支／收敛历史。当前版本 OWAC-D1.1：主教师对外接收 RAW，内部可经 Neural ISP 成像；教师和学生不固定单帧，以控制效果、稳定性和成本选择因果历史／状态。学生部署直接 RAW→tokens，首轮后端冻结。工程实现进行中，本次修订未训练教师／学生或运行新闭环。作者模型实现已公开，训练后 RawVLA 前端／完整教师公开权重尚未找到，见[资源核查](../research/reviews/2026-10-20_rawvla_public_teacher_assets.md)。

2026-10-19 的[ConvNeXt 单任务视觉替换方案](../research/experiments/2026-10-19_visual_replacement_recovery.md)及其单帧前端转为诊断历史；旧后端 LoRA 配方不沿用。50 条真实示范、每适配臂累计最多 5,000 更新作为新方案首轮诊断规模，不能视为已执行或充分收敛。配套 RawVLA runtime、LIBERO 策略／统计／tokenizer、真实配对示范及初态仍须核验；DROID 权重不能替代。

2026-10-19 本地 worktree 清理完成：三个候选 checkout 均已移除，当前只保留 `/home/jeong/zeno/owac/repo/` 的 `main`，后续直接在此开发。清理前重新核验提交祖先关系、工作区、91,270 个证据文件与归档哈希；另保全 N02 本地第三方源码。历史分支、代码快照和主仓库证据归档保留，见[清理收据](../research/archives/2026-10-19_worktree_consolidation/worktree-cleanup.json)。

2026-10-19：用户暂停 N01、N02、N04，要求三个 worktree 合并回主线。远程服务器已不可达；本轮只合并、保全本地证据并运行本地代码检查。当前入口为 `repo/` 的本地 `main`，不自动恢复训练、诊断训练或下一阶段。

[合并与证据索引](../research/archives/2026-10-19_worktree_consolidation/README.md)记录原分支/提交、接口冲突处理、本地归档与远端缺口。四处原始状态快照分别保存在该目录的 `baseline/`、`n01/`、`n02/`、`n04/`，历史状态不作为当前作业状态。

| 方向 | 最后可查本地证据 | 当前结论 |
| --- | --- | --- |
| N01 v1.2 P3 | [停止记录](../research/experiments/n01_v1_2/P3_training_stopped_20261019.md)、[动作诊断](../research/experiments/n01_v1_2/P3_action_diagnosis_20261019.md)、[代码同步](../research/experiments/n01_v1_2/P3_code_sync_20261019.md) | RAW/RGB 各 0/12；正常光同场景作者 4/4、两臂各 0/4。专家观测动作已明显偏离，优先定位视觉—动作适配。未通过完整研究验收。 |
| N02 v1.2 P2/r17 | [r11 闭环收据](../research/experiments/n02_v1_2/closed_loop_r11_20261019/receipt.json)、[r17 结论](../research/experiments/n02_v1_2/residual_joint_r17_20261019/conclusions.json)、[r17 审计](../research/experiments/n02_v1_2/residual_joint_r17_20261019/) | r11 0/3；r17 新 Normal 19 条专家合格示范、423 窗口的小样本动作读出有冻结 RAW 收益，局部联合更新没有额外总体收益。有效 6144＋保留失败 1024＝实际 7168 更新；π 更新 0、学习策略闭环 0，不转为机器人成功。 |
| N04 v1.1 S2 | [v4.2 修复](../research/experiments/2026-10-16_n04_aux_repair_v4.md)、[v5 故障与修复](../research/experiments/2026-10-19_n04_stability_repair_v5.md)、[代码同步](../research/experiments/2026-10-19_n04_server_code_sync.md) | 历史开发闭环 0/100；v4.2 对应越界/目标消失、潜在量纲增长与 RoboTwin 退出保留。v5 短程零更新/优化/重载通过，只支持工程修复，长期稳定与任务收益未验收。 |

N04 v5 本地最后运行收据保存到 step10；这不是本轮确认的最终停止步。N01/N02 的已完成或停止回执同样以其各自记录时间为准。现在不能查询远端最终进程、日志、checkpoint 或存储状态，不能宣称远端独有权重已恢复或已丢失。

完整 Default π0.5 基线仍为 [2026-10-14 全量报告](../research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)：LIBERO 5820/10000（58.20%）、RoboTwin 508/3250（15.63%）；附加原条件 632 条另计。完整分母已取得，论文绝对数字未复现。原协议标签和历史负结果保持不动。

研究统筹台现从主 checkout 读取三条路线，当前节点为“暂停与本地证据”；旧提示词可阅读，但不会作为当前执行包发出。未来是否统一架构或恢复实验由新的研究决定确定。

# OWAC 当前状态

2026-10-15 **N02 v1.2完整阶段提示词已备，代码环境已部署**：[P2—P4入口](../../worktrees/n02-measurement-completion/docs/prompts/n02_v1_2/README.md)。当前P2一次建立完整模型/真实π0.5/实际训练与简洁实时网页；P3充分训练整体比较，P4每个实际选中模型共100次闭环。仅seed1103，新服务器`93.91.156.102:51584`双约96GiB GPU，RAW/RGB可并行；环境/权重/双平台原策略迁移检查通过，数据全集后台同步；不要求三种子/全部矩阵/13,250全集/严苛hash或收敛判定。[部署记录](../../worktrees/n02-measurement-completion/research/experiments/2026-10-15_n02_v1_2_remote_deployment.md)区分实际完成项与缺口。P1独立验收未通过保留；本轮Codex写提示词并部署，模型P2待实施，未启动长训练。

2026-10-15 **N04编程任务补充专用双GPU服务器**：S2／S3／S4已填入用户指定的SSH／8080转发；允许双模型训练、双worker验证或一训一验，独立绑定GPU与输出，验证分片仍每模型合计100次。旧阶段产物保留，HF认证从服务器HF_TOKEN／既有缓存读取，明文token未写入提示词或仓库。监控增加两卡简表，保持单页。当前阶段状态未变；本轮未连接服务器、设置远端认证或启动训练。下方早期“登录留空”记录为历史。

2026-10-15 **N04 v1.1 DeepSeek各阶段任务同步**：S2／S3／S4已扩写为可直接复制的自包含提示词，附作者实现链接、代码职责、远端训练路径和前端专家简洁实时网页指令；登录／转发指令留空。当前口径是每平台／方法seed101、每个待评模型100次（默认LIBERO50＋RoboTwin50），不要求三种子、全部方法／消融、官方全清单或严苛hash／收敛判定。完整交接包只复制本阶段原文，避免旧验收要求混入。方案、采用决议、状态与研究卡已同步；本轮只写提示词，未启动训练、实现模型或部署远端网页。下方旧N04三种子／完整评测安排由本条取代。

2026-10-15 研究统筹台同步 **N04 v1.1 正式路线**：用户已批准[完整方案](../../worktrees/n04-egomotion-separation/docs/plans/N04_full_development_v1.1.md)。当前节点为 S2 完整架构与整体训练待实施，随后 S3 充分训练／四臂三种子匹配复验、S4 冻结完整评测。S0／S1 为历史；S1 局部学习有效但几何方差未通过，扩展数据及衍生权重不进入官方数据主实验。当前方案、采用决议、独立验收和新版阶段任务已绑定到交接包；旧 S2—S7 不再作为执行入口。本轮仅更新文档及显示，没有启动 S2 或训练。

2026-10-15 研究控制台同步 **N01 v1.2**：新增RAW信息→编码→行动链路、同源RAW/RGB×Base/N01四臂与数据边界卡片；P3当前完整包原文复制，开发每轮共100次闭环，优先当前模型，不要求四臂/两平台全覆盖；P4四臂单种子正式核心53,000后续另行安排。更新全部当前文档/提示词入口，P0—P2及旧版本为历史；N02/N04路线保持。新四臂模型、远端训练监控和正式训练尚未实施。

2026-10-15 研究统筹台同步 **N01 v1.1完整执行包**：P3完整模型/真实训练/开发闭环为当前默认；用户补充每平台/方法仅1个训练种子（默认101），P4核心矩阵39,750条，不默认三种子。P4冻结全量、P5有依据扩展各一份自包含提示词；包内含成熟作者代码、远程训练、简单实时监控页的前端专家任务。旧局部P3—P8退出执行入口，P0—P2历史保留。P2独立复核限制不改判；本次只准备提示词并只读核查服务器，未开发新模型、部署远程监控或启动训练。

2026-10-15 研究统筹台同步 N01 v1.0：P2 已提交（修补 `fe8d534`，验收记录 `d3dec90`），用户明确决定“视为通过”；当前节点切到 P3 待启动。此为用户阶段决策，保留原独立复核的未完全通过结论及训练/风险限制；P3 交接包纳入 P1/P2 最新报告。

2026-10-15 研究统筹台改为用户级 systemd 常驻服务 `owac-research-hub.service`：固定本机 8876 端口，开机启动、异常退出自动重启，沿用刷新时在主 repo 执行 `make research-hub`。服务配置与维护命令已保存，研究进度仍以各工作树的验收记录为准。

2026-10-14 研究统筹台同步 N01 v1.0：P1 独立修补后验收并提交（实现 `2d82bbb`，验收归档 `6fb3807`），当前节点为 P2 待启动；P2 交接包纳入 P0/P1 独立验收报告与修补后语义。本次仅更新显示与交接入口，未启动 P2。

2026-10-14 研究统筹台同步 N04 v1.0：S0 数据与因果合同已验收并提交（`a81cf25`），当前节点切换至 S1 待启动；证据入口采用远端补齐验收报告，S1 交接包加入最新验收与专用提示词。本次仅更新统筹显示，未启动 S1。

2026-10-14 RawVLA 参数与结果已独立归档本机：原清单、全部有效配置、13,882 条正式 episode 数字、历史异常与修复、正常化、依赖/源码版本、上游训练来源均保存。独立解压校验 668 个文件哈希及完整身份/seed/语言/成绩通过；附完整压缩包、收据与未来比较模板。本轮额外训练 0 步，参考配方与未知项独立标记。[归档入口](../research/archives/2026-10-14_rawvla_pi05_baseline.md)。

2026-10-14 研究统筹台新增自动生成：本地服务页面打开、浏览器刷新和“重新读取”均执行主 repo 的固定 `make research-hub`，完成后刷新内容与离线文件；生成中、失败和执行收据明确记录。入口为 `http://127.0.0.1:8876/`，工作树仍只读。

2026-10-14 新增本地[研究统筹台](research-hub.md)，汇总 N01/N02/N04 工作树、路线阶段、说明原文及提示词。`make research-hub` 生成可离线打开的页面；`make research-hub-serve` 支持按需重读。当前入口采用 N01 v1.0（P0—P8）、N02 v1.1（P0 独立验收后，P1 待启动）与 N04 v1.0（S0—S7）；原文缺失、未跟踪和摘要来源变化明确标注。本条仅记录统筹工具，不改变各路线的研究判定或执行进度。

2026-10-14 10:44:24（北京时间）RawVLA Default π0.5 完整基线已取得：主清单 **13,250/13,250**，LIBERO **5,820/10,000（58.20%）**、RoboTwin **508/3,250（15.63%）**；独立原条件 **108/632（17.09%）**。33 条缺口按原身份全部补齐，原 3,217 条有效结果不变；身份审计异常及未补齐身份均为 0，三个有限队列已退出。全部 RoboTwin 3,882 条实际轨迹与模型调用记录完整，51 项测试及静态检查通过。报告保存逐任务/域成绩、成本、协议和结果指纹；论文数值仍未复现，旧标签不移动。[完整报告](../research/experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)。

2026-10-14 09:39:45（北京时间）RawVLA 核查：主队列和审计已结束，13,217/13,250 有效完成；LIBERO 10,000/10,000、58.20%，RoboTwin 3,217/3,250、498 成功、33 条基础设施缺口，状态 incomplete。原条件核查 632/632、108 成功、17.09%。目前只有模型服务加载，等待不会增加进度；33 条仍需独立修复补跑。本次未重复启动，冻结标签不变。见 [最新状态](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md#最新运行快照)。

2026-10-13 16:42:30（北京时间）原条件五条基础设施异常已修复并按原 seed/照明补跑，全部完成真实策略控制，均为正常失败；原 627 条未变。附加正式达到 **632/632，108 成功，成功率 17.09%**；旧失败、修复协议、连续动作与独立审计保留。控制首个状态/三路图像/动作逐值一致，33 项测试通过，策略方法在 setup 前恢复；主评测、冻结代码与标签不变。[修复及完整结果](../research/experiments/2026-10-13_rawvla_robotwin_infrastructure_repair.md)。

2026-10-13 16:01:42（北京时间）RawVLA 进度核查：主清单 12,887/13,250；LIBERO 全部 10,000 条完成，成功率 58.20%；RoboTwin 2,887/3,250，状态 running，33 个基础设施异常身份尚未完成。附加原条件 627/632、状态 incomplete，缺 5 条。完成记录身份审计异常 0 个；保留原分母，完整 RoboTwin 成绩仍待补齐。部署/协议标签不移动，详见 [最新运行快照](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md#最新运行快照)。

2026-10-19 **服务器代码已同步留档到本地。** 保留主源码、三个执行快照、配置及版本清单；同名差异保留双方版本，本地较新实现未被覆盖。检查点和数据集未下载，训练保持停止。见[代码同步记录](../research/experiments/n01_v1_2/P3_code_sync_20261019.md)。


2026-10-19 **闭环失败有限诊断完成，训练保持停止。** 同四个Normal场景作者预训练策略4/4成功、RAW/RGB均0/4；相同3个Normal专家观测上的完整采样MAE7分别为作者0.0618、RAW0.6149、RGB0.3837。模型在专家观测上已明显偏离，优先查视觉—动作适配；不能仅归因于闭环误差累积。见[诊断与证据](../research/experiments/n01_v1_2/P3_action_diagnosis_20261019.md)。

2026-10-19 **按用户最新要求停止RAW/RGB训练与后续队列。** 已禁用自动启动并保留检查点、原始日志及0/12配对小测结果；只有用户重新授权后才可恢复。见[停止记录](../research/experiments/n01_v1_2/P3_training_stopped_20261019.md)。下方“已恢复训练”为本次停止前的历史。

2026-10-19 **20k附近RAW/RGB检查点的配对闭环小测完成：两臂各0/12。** 实际权重20,800/21,100步，同一组12场、共24次执行；四个LIBERO suite，ExtremeLow/Low/Normal各4场，没有未完成。两路训练均已恢复并实际推进，监控页显示成功率及正常光同场景回放。见[小测记录](../research/experiments/n01_v1_2/P3_20k_closed_loop_probe.md)与[实时监控](http://127.0.0.1:18080/)。这是开发小测，训练重合未知，不作留出泛化或机制优劣结论。

2026-10-17 **按用户最新授权启动完整 N01 训练。** 两张 PRO 6000 分别负责 RAW/RGB，当前 LIBERO、随后 RoboTwin；每 run 50k 步预算，Base 仍留至 P4。新训练使用独立冻结源码与数据清单，不续旧有缺陷的权重。[简洁实时监控](http://127.0.0.1:18080/)只看状态、速度、验证误差和 GPU；[启动、加速与实测记录](../research/experiments/n01_v1_2/P3_training_launch.md)。下文“代码补全后暂停”是本次授权前的状态，旧闭环留出撤回仍有效。

2026-10-17 **N01 v1.2 P3代码已补全，训练按用户要求保持暂停。** 完整SEA-RAFT、局部GLS/有界BOCPD、来源重算、统一原生输入与匹配后端适配已接入；八份配置及两平台运行接点齐全。新模型未训练，旧闭环“留出成立”仍撤回，不能标记P3研究完成。当前入口：[代码补全与无更新验收](../research/experiments/n01_v1_2/P3_code_completion.md)。下方旧实现缺口及数值保留为历史。


2026-10-16 **N01 v1.2 P3 独立验收未通过，主体队列已保全并暂停，禁止按原队列自动续训。** 已修补断点/RNG、重复测量证据、reset 后验、参考有效性与留出筛选身份错误；旧“0/96 留出成立”撤回。运动对应/真实支持拟合、训练与服务视野及后端适配仍缺；Base 按既有决定留至 P4。详见[独立验收与修补记录](../research/experiments/n01_v1_2/P3_independent_acceptance.md)。下方 10-15 的“P3待实施”及 DeepSeek 交付记录为历史。

2026-10-15 N01更换至新服务器`154.59.156.21:34302`，两张RTX PRO 6000 Blackwell Server Edition可并行RAW/RGB训练及验证，每轮开发闭环仍总100次。v1.2三份完整提示词与控制台已同步；实际环境/源码/数据恢复及验证见[新机部署记录](N01_new_server_deployment_2026-10-15.md)。本次迁移未启动P3主体训练。

2026-10-15 **N01 v1.2已采用，P3待实施。** [Markdown](N01_full_development_plan_2026-10-15_v1.2.md)、[HTML](N01_full_development_plan_2026-10-15_v1.2.html)、[决议0015](decisions/0015-n01-v1-2-raw-information-route.md)、[完整阶段包](prompts/n01_v1_2/README.md)统一更新。核心为同源RAW/RGB×成熟Base/N01四臂，单种子101，两平台8个主体run，P4核心53,000条；开发期每轮只要求共100次闭环拿成功率，优先当前模型，不按平台/臂乘开，不要求全部覆盖；原B1为补充，不默认第五臂或三种子。RGB由同条件RAW经固定ISP产生，禁用正常光参考误配与RAW中间量泄漏。理论/可视化围绕信息保留、噪声—细节—响应与行动，删除一致性为支撑。控制台显示四臂计划、数据边界及P3完整复制入口；P0—P2历史不改判，未启动新训练、远端监控或正式评测。下方v1.1为历史记录。

2026-10-15 **N01 v1.1 正式采用，P3 尚未实施。** 用户补充当前每平台/方法仅训练1个种子，默认101；P3核心为两平台×N/B0/B1共6个主体训练run，P4主全量矩阵为39,750条，三种子不作为当前门槛。 [Markdown](N01_full_development_plan_2026-10-15_v1.1.md)与[同源 HTML](N01_full_development_plan_2026-10-15_v1.1.html)已更新；[决议 0014](decisions/0014-n01-v1-1-full-model-route.md)取代 v1.0 尚未执行的 P3—P8。P3 一次交付完整编码器、两平台真实 RawVLA-Bench 训练、匹配强对照、开发闭环、主链理论及同步可视化；P4 为冻结 13,250 身份/每 checkpoint 的全量验证，P5 为原生 RAW/跨域等扩展。P0—P2 资产继续复用，P2 未完全通过和运动/风险反例不改判；不为补齐旧字段机械续训。[P3/P4/P5完整执行包](prompts/n01_v1_1/README.md)已经准备，含逐文件参考、远程训练和简单实时监控页的前端专家任务；取消无意义的硬门槛、自定义评分、严苛哈希与固定收敛PASS，保留实质正确性检查。只读核查发现远端8080已被Jupyter占用，建议监控使用远端18080经SSH转发。本次无新模型开发、监控部署、服务器训练或正式评测。

2026-10-15 **N01 v1.0 P2 独立验收：未完全通过，代码已修补，未续训。** 新增 10 项反例在原版本全部失败、修补后全部通过；本地 721 passed / 2 CUDA skipped、服务器 375 passed / 0 skipped。修正实际停止点/最终低 LR 窗口、风险分档、遮挡与静态误拒分母、来源/噪声块身份及恢复校准 hash。现有六组仅 CFA seed 307 满足全部平台条件；12 px 新合成诊断误差约 3.3–11.6 px，风险覆盖明显失准。原训练及后端接入证据保留，暂不建议直接按原配方续到 300k；遵照用户要求，本轮正式训练更新为 0。见[独立验收报告](../research/experiments/n01_full/P2_acceptance_report.md)。下方 2026-10-14 P2 完成断言为修补前历史，以本次复核为准。

2026-10-14 **N01 v1.0 P2（RAW 运动对应与噪声模型）交付并停在交接点**。新增 `backends/raw_correspondence.py`（固定 SEA-RAFT vendor commit `9137517` 的 RAW 适配：CFA 4/8、pseudo-RAW 3/6，关闭 torchvision ImageNet 自动加载与 `/255`，`estimate()` 返回 pull 坐标 + 4-tap SourceKey + proxy 风险并预留独立校准表协议）、`raw/n01_motion_data.py`（解析真值合成运动生成器：平移/旋转/分层遮挡/局部非刚性/重复纹理/曝光/运动模糊，GT 与模型可见输入严格分离）、`raw/n01_noise.py`（`Var(y|mu)=a·max(mu,0)+b` + 共享黑电平块 + 曝光/增益同变换，输出 P1 `NoiseModel`）、`frameworks/n01_full/adapters.py`（`DbsrFeatureAdapter`：显式三通道 stem、16×16 带 mask 池化、共享 `VisualTokenPostprocessor(512→2048, 256/相机)`、P3 硬关闭）、`frameworks/n01_full/motion_builders.py`（窗口 motion provider 与清除后 ledger 上的 correspondence/noise 重算 builder）、`training/n01_motion.py`（作者迭代损失、fit-check/两档 LR 选择/平台判据/完整恢复/`stem-report`）、`evaluation/n01_motion.py`（四臂与风险分档）与 `n01_full_policy_preflight --mode raw_adapter`。本地全仓 **709 passed / 2 skipped**（两个 skip 为 CUDA-only 用例，在 GPU 机上 `-k gpu` → 3 passed / 0 skipped）、`make check` 通过；内部独立验收对手 53 项反例 + 16 个变异矩阵，抓到并修复 D1（平台判据把恒定/恶化曲线判成 `validation_plateau`）与 D2（噪声对角分支漏已审计共享块方差），并按 3.10/3.11/3.12 三解释器修复 AST 摘要的跨版本缺陷；远程集成阶段又修复 A（strict `weights_only` 拒绝含 NumPy RNG 的本项目 checkpoint）、B（预检窗口计数调用不存在的方法）、C/D（estimator 与评测各有多处 CPU/CUDA 设备混用）与 E（裁定 rgb3 窗口↔rgb3 estimator 的配对规则并新增加载前 stem 校验）。远程 `source-v1..v5` 逐文件 hash 全部一致；256 拟合、两档 LR 选择（winner `lr=2.5e-4, iters=4`）、rgb3 dev 臂（stem 参数差 −9,408 与解析一致）通过；CFA 第一段 3 seeds × 100k 全部结束：seed 101 `validation_plateau`@82k（val EPE 0.069852）、seed 211 `budget_exhausted`@100k（best 0.0759@91k，尾部漂移）、seed 307 `validation_plateau`@99k（经历 62 分钟实例失联，从 update 13,000 精确续训）；**pseudo-RAW（rgb3）正式臂三种子**（各臂自己的两档 LR 搜索 winner `lr=5e-5/iters=4`）也全部 `validation_plateau`：85k/81k/85k，val EPE best 0.0394/0.0401/0.0405；两臂 GPU 评测（test 32 pairs）EPE CFA 0.0869/0.0893/0.0950、rgb3 0.0455/0.0426/0.0419，`no_alignment` 1.9213、`old_simplified` 2.9852、`oracle` 0.0 单列，GPU p50 6.79–6.84 ms / p95 6.81–6.90 ms / 峰值 103.7 MiB，静态误拒 0.0；两平台 `raw_adapter` 真实预检（crop=128）、identity 对照与故意错配诊断均通过（motion 路径现用 **正式 rgb3 模型** checkpoint，256-update dev 臂保留为对照）。P2 远程 GPU 时间合计约 **9.9 小时**。**未进入 P3/P4，无 N01 机制增量或闭环结论**；`implementation_status=PASS_P2`、`training_status=P2_SEGMENT_COMPLETE`、`effect_status=NOT_APPLICABLE`。见 [P2 报告](../research/experiments/n01_full/P2_report.md)与[收据](../research/experiments/results/n01_full/P2/receipt.json)。

2026-10-14 **N01 v1.0 P1 独立验收修补后通过**。原交付的 34 项测试通过，但新增首批 14 项反例全部失败；已修复保守方差界、秩亏读出、先验坐标、反事实帧级泄漏、实际 32 帧窗口及策略梯度/更新证据。最终本地 495 passed、服务器 147 passed、解析两端 9/9、verify 22/22、153/153 文件哈希一致；两平台真实来源计数与 encoder/bridge 各自更新、后端冻结及 validation 无更新已重验。实现提交 `2d82bbb`。仍停在 P1，未进入 P2、无长训练或机制/闭环结论。见[独立验收报告](../research/experiments/n01_full/P1_acceptance_report.md)；下方原 P1 完成记录保留为修补前历史。

2026-10-14 **N01 v1.0 P1（可组合与可撤回的解析参考）完成并停在交接点**。新增 `src/owac/frameworks/n01_full/{ledger,domains,statistics,replay,reference,sequence1d}.py`（身份账本与四态 validity、版本化支持域 make/merge/split/retract/restore、逐相位 GLS 与秩/可辨识掩码、`fixed_decision_replay`/`counterfactual_replay` 与 32 帧精确窗口、独立 NumPy FP64 oracle、一维同语义例子）、`src/owac/evaluation/n01_full.py` 的 `analytic` 子命令（阶段 §4 七项 + 插值/先验/保守界 + 一维例，9/9 pass）、`src/owac/runtime/n01_full_ledger_check.py` 与 `src/owac/runtime/n01_full_policy_preflight.py`（`mode=contract`，`raw_adapter` 明确留给 P2）。新增 34 项测试，本地全仓 473 passed、`make check` 通过。远程 `source-v8` 152/152 hash 一致、`verify` 22/22、125 passed、`analytic` 9/9；真实两平台唯一来源计数（LIBERO 33 到达/28 登记、RoboTwin 49/42，重放新增 0）与策略预检（两平台不同真实 episode 的 forward/backward/update、后端梯度 0、validation 参数漂移 0.0、token `[1,512/768,2048]`）通过。检查过程发现并修复 7 个缺陷（含真实 tensor 身份与时间轴两处严重问题），另有 P0 `config.py` 的 Python 3.10 兼容性修改。**未开始 P2、无长训练、无机制效果结论**；`training_status=NOT_STARTED`、`effect_status=NOT_APPLICABLE`。见 [P1 报告](../research/experiments/n01_full/P1_report.md)与[收据](../research/experiments/results/n01_full/P1/receipt.json)。

2026-10-14 P0 独立验收修补完成：修正 RoboTwin 协议归一化键、可执行 profiling、严格数据身份检查、GPU 环境路径/依赖、任务层统计与完整预算。两平台四区真实 loader、远程 55/55 checks、139/139 源包、91 项 P0 测试通过；本地全仓 439 passed。仅接受 P0 工程起点，未开始 P1/长训练。 见[独立验收报告](../research/experiments/n01_full/P0_acceptance_report.md)。

2026-10-14 **N01 v1.0 P0（问题、数据与依赖）完成并停在交接点**。新增 `src/owac/frameworks/n01_full/{contracts,config}.py`（SourceKey / MeasurementRecord / CorrespondenceMap / SupportDomain / EvidenceEstimate / DependencyNode / ReadoutSnapshot）、`src/owac/raw/n01_full_manifest.py`（2,403 条源轨迹 → train/calibration/validation/test 四区，零重叠，真实 loader 可精确消费 `sides`）、`src/owac/runtime/n01_full_preflight.py`（inventory/profile/bundle/verify）、`configs/encoders/n01_full/{protocol_v1.json,upstream_lock.json}`、`research/theory/n01_support_contract_v1.md`（T1–T6）与 4 个测试文件（86 passed）。远程新建独立编码器环境 `envs/encoder` 并完成真实 GPU forward/backward 与全尺寸 DBSR profiling；远程源包 227/227 hash 一致、`inventory --strict` 36/36 通过、bundle/verify passed（含上游许可；DBSR 为 CC-BY-NC-SA-4.0）。全仓 434 passed、`make check` 通过。**未开始 P1 算法、未启动长训练、无机制效果结论**；`training_status=NOT_STARTED`、`effect_status=NOT_APPLICABLE`。已记录待研究负责人处置的前序异常（研究方案文档三个标题的既有未提交误编辑）与任务级共享披露义务（40 个 LIBERO 任务模板 / 13 个 RoboTwin 任务出现在全部四区，P3 配对重采样须含任务层）。见 [P0 报告](../research/experiments/n01_full/P0_report.md)与[收据](../research/experiments/results/n01_full/P0/receipt.json)。

2026-10-14 已准备 [N01 v1.0 DeepSeek 分阶段提示词](prompts/n01_v1_0/README.md)：P0–P8 九阶段、共同实现契约、当前代码审计、上游逐文件导航、远程执行说明及 P1/P2 早期策略预检。只读核查用户服务器的实际数据、权重、环境与服务；新模型/训练尚未执行。主 checkout 已有完整 Default 基线报告，旧条目的 RoboTwin 未完成状态按历史保留，后续以新完整报告和实际审计为准。

2026-10-14 已交付 [N01 全量开发方案 v1.0](N01_full_development_plan_2026-10-14_v1.0.md)（[HTML](N01_full_development_plan_2026-10-14_v1.0.html)）：基于当前 DBSR 与撤回实验，定义来源级时空支持、六项理论义务、P0–P8 实施与充分训练、π0.5 / RawVLA-Bench 最终验证。核查 25 项论文/官方来源；这是后续研究方案，未执行新模型训练，历史负结果保持。

2026-10-13 完成 DBSR 三种子收敛基线（9 组，含 3 组保持训练状态的预算续训）和两层独立冻结干预（CNN 硬拓扑 / 原 RAW 支持域，各 21 组）。基线早期误差：DBSR 1.52998、同容量固定平均 1.75656；同权重强制平均退化 0.55570。原 RAW 域撤回与重算的来源隔离通过，但动态重算仅比旧缓存改善约 0.000995，恢复失败数不变，N01 增量未建立。全仓 348 passed；无新增网络、原核心未改、未改冻结服务器基线。见[完整记录与边界](../research/experiments/2026-10-13_n01_dbsr_mechanism.md)。

2026-10-13 已接入作者规模 DBSR RAW 特征层与融合器，完成五组 × 128 步开发验证：普通条件早期误差，容量补齐固定平均 1.96523、DBSR 参考融合 1.99160；当前尚无学习融合的独立收益。保留全部负结果、逐帧恢复和容量控制；N01 核心源码未变，支持图尚未接入新基线。全仓 330 passed，五个全尺寸检查点恢复结果一致。单种子合成验证不代表充分训练或真实后端验收。见[实验与复用记录](../research/experiments/2026-10-13_n01_dbsr_features.md)。

2026-10-13 N01 外部方法复用审查完成，见[文献与源码审查](../research/reviews/2026-10-13_n01_prior_art_reuse.md)。检查 7 个固定版本仓库、33 个源文件；DBSR `WeightedSum` 与 EDVR `TSAFusion` 的独立 CPU 前向/反向通过。建议先建立逐帧 RAW 特征融合强对照，再检验支持关系撤回增量；解析可靠性参考 IPOL 的局部统计与噪声校准。记录了 RGB 对齐/监督、未来帧、许可与有效性适配边界。本轮未修改 N01 算法或执行新任务训练，接口通过不表示性能提升。

2026-10-13 N01 内容通路与六组配对实验完成：18 个正式运行核验通过，C 比同容量当前内容控制 E 改善 0.05575 packed px，D 比 C 差约 0.00686 px；改善较小，全部最优检查点在预算末端，尚未排除训练不足。工程检查 291 passed。逐来源可靠性探索已完成：同权重新留出实验平均退化 0.01446 px，未采纳为正式算法；全仓 314 passed，新规则关闭时 554 张量逐位一致，62 项核心源码检查通过。完整记录见[逐来源实验](../research/experiments/2026-10-13_n01_source_reliability.md)，未更新冻结基线或机器人任务。见[本轮记录](../research/experiments/2026-10-13_n01_content_integration.md)。

2026-10-13 N01 机制目标复审与补测完成，见[完整记录及改进方案](../research/experiments/2026-10-13_n01_mechanism_reassessment.md)。当前 `189b66d` 同数据、同初始化、同 40 步对照：N01 2.589865 px、固定历史 2.586656 px，仍没有独立调度收益证据。源码/结构夹具发现，历史主要参与门控且四平面统计平均会丢失部分历史颜色区别；建议先验证积分内容直接进入特征，再检验固定/自适应和逐源可靠性规则。本轮保留核心源码，新增诊断收据与实验设计；未实施新架构或执行新服务器/闭环任务。

2026-10-13 N01 快速运动历史响应修复完成，见[本轮验收](../research/experiments/2026-10-13_n01_motion_repair.md)。保留积分、唯一采样、重组与四项代价；改进局部历史预测创新、相关通道/共享插值噪声处理及 motion 校准单位下限，补充实际历史与目标区域指标。18 项受保护源码校验一致，全仓 265 passed、make check 通过。

三组独立种子×24 轨迹、两个对齐配置的前后匹配验证通过，36 对输入哈希相等。候选对齐下目标中心实际历史：静止 7.952、1 px/frame 运动 5.151、停止后 7.932；修复前运动为 7.222。原低光约0.221 px/frame 样例缩短较小。旧 motion 检查点须重新校准/适配；本轮未新增服务器或闭环执行。自动调度相对固定历史的任务收益按用户要求留待后续研究。

2026-10-13 N01 七组源码问题修复与配对验证完成，见[本轮记录](../research/experiments/2026-10-13_n01_source_repair.md)。15 项核心定义/源码不变；修复统计门控饱和、轨迹校准缺失、运动 identity 误读、实验事件/SNR/容差、CFA 邻接、平台参数和伪时序指标。全仓 257 passed，make check 通过；真实两平台解析/学习共 12 batch，服务共 6 次请求通过，冻结基线保留。旧 gate 权重须重新训练。

同一修正数据与 40 步预算，N01 3.041 → 2.590 px，去重组 2.621、固定历史 2.586；六个基线每条件误差逐位一致。定位改善约14.8%，自适应调度仍未通过研究验证：固定历史略优，低 SNR 未训练快速轨迹历史仍长于静止。新生成器与旧矩阵数字不能直接比较，完整机器人任务收益仍未建立。


2026-10-12 N01 独立验收与修复：原“全部完成”报告不能验收，原因包括错误的支持域/方差/计数、重复帧挤掉唯一观测、训练与在线历史不同、重复训练同一随机批次、辅助目标泄漏，以及不可执行的真实 NPZ 训练入口。现已直接修复并复跑，见 [完整验收记录](../research/experiments/2026-10-12_n01_encoder.md#2026-10-12-独立验收与修复)与[候选卡](../research/candidates/N01.md)。

- 研究实现与本轮工程验收通过：9 个半径×时间支持域，严格逐源与插值系数核对；迟滞显式开闭，唯一帧/独立相机/部分 reset/零预算正确；一维与 CFA 入口保留；新增可恢复原生动作训练、C3/C4 与独立服务。
- CPU 在线随机 200 步训练已复跑，末步定位误差 1.113 px；22 条独立条件平均 2.308 px、超容差 0.354。40 步候选 3.026 px，固定期限消融 3.027、去重组 3.026。**联合调度与定位收益仍未获支持**；三条未训练轨迹也未显示可靠的运动缩短历史。
- 真实 NPZ/π0.5 已在固定服务器 runtime 通过：LIBERO/RoboTwin 各 6 组×20 batch，共240 batch；包含支持域解析/学习×辅助 loss 开/关及 C3/C4。真实 encoder/action 梯度、实际参数更新、冻结后端、padding mask、重复 flow 推理均核验。数据为作者三通道 pseudo-RAW，32×32 明确裁剪；不是全量闭环或原生 Bayer 结果。
- 独立 WebSocket 服务真实请求、重复帧、新 episode reset 通过；测试后停止。特征桥被纳入独立源码复制/哈希清单，原 C0 冻结代码前后哈希相同、原 RoboTwin 全量/审计进程保留。
- 锁定 CPU torch 2.9.1+cpu，make check 与全仓 240 项测试通过。tiny CPU 单步/读出 p50=9.96/4.24 ms，状态377044 B；small/medium 有成本采样，未声明规模能力趋势。

上述是实现与真实接入验收。算法有效性、正式 N01 全量闭环、充分调优系统对照、真实 CFA/相机迁移仍待研究；旧实验数字保留为历史证据。

2026-10-12 编码器实现交接：按用户要求准备 N01 自适应时空证据积分、N02 测量约束的循环补全、N04 自运动预测与变化分离三个独立 worktree。共同起点为已固定 RawVLA 部署加最新研究文档快照；三份自包含 DeepSeek 提示词已展开到公式、文件职责、训练步骤、StarVLA 特征接口、参考实现及反证实验。见[工作树与使用入口](prompts/README.md)和[0013 决议](decisions/0013-encoder-research-worktrees.md)。本条表示研究开发环境与提示词准备，不表示编码器实现、训练或机制验收。

2026-10-12 发布固定：本地提交及标签 `owac-rawvla-pi05-v0.1.0` 固定已跑通的 RawVLA-Bench / π0.5 部署和后续比较协议，见 [0012](decisions/0012-rawvla-pi05-protocol-freeze.md)。LIBERO 10,000 条完整成绩为 58.20%；RoboTwin 仍在运行、有基础设施缺口。该标签不表示两平台全量已齐或论文数值复现；旧 `owac-dev-v0.1.0` 的 RAW/特征接口门槛仍待验收。

2026-10-12 10:21:11（北京时间）RawVLA 进度核查：LIBERO 10,000/10,000 完成，成功 5,820、完整成功率 58.20%，身份审计通过。主清单累计 11,745/13,250；RoboTwin 1,745/3,250，仍运行，16 个异常身份未完成；附加原条件 627/632、5 个异常未补齐，状态 incomplete。整套基线尚未完成；见 [最新运行快照](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md#最新运行快照)。

2026-10-11 20:16:17（北京时间）RawVLA 进度核查：主清单完成 9,053/13,250；LIBERO 7,950/10,000，RoboTwin 1,103/3,250，两套主评测与审计仍运行。附加原条件 627/632、状态 incomplete；主 RoboTwin 与附加各 5、5 个异常案例待补齐，保留原分母；尚无完整基线成绩。见 [最新运行快照](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md#最新运行快照)。

2026-10-11 15:15:48（北京时间）RawVLA 进度核查：主清单完成 7,299/13,250；LIBERO 6,461/10,000，RoboTwin 838/3,250，两套主评测与审计仍运行。附加原条件核查已结束调度，627/632 有效完成、5 条异常未补齐，状态 incomplete；主 RoboTwin 另有 5 条异常未完成。均保留原分母，尚无完整基线成绩。见 [最新运行快照](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md#最新运行快照)。

2026-10-11 13:37:33（北京时间）RawVLA 进度核查：主清单完成 6,707/13,250；LIBERO 5,934/10,000，RoboTwin 773/3,250，附加原条件 609/632。三套评测与审计仍运行；主 RoboTwin 5 个、附加 5 个案例异常未完成，保留原分母；尚无完整基线成绩。见 [最新运行快照](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md#最新运行快照)。

2026-10-11 用户决定：**后续全部使用 RAWVLA-bench**，见 [0011](decisions/0011-rawvla-bench-main-route.md)。Arena DROID 的 RGB 评测、PolaRiS 重建及其公开发布已终止，旧数据与权重保留，旧页面标注历史状态。RAWVLA 既有全量评测继续运行；不重复启动，不创建定时任务。 停止后最后归档：Arena 290/900（211 成功、79 正常失败）；PolaRiS 配对覆盖 35/316、物理检查通过 19 条，其中约束辅助 1 条，训练加载验证仍为 18 条。全部为未完成路线的历史结果，两个旧页面与文件哈希已核验。

2026-10-11 01:50:48（北京时间）RawVLA 进度核查：主清单完成 3,031/13,250；LIBERO 2,683/10,000，RoboTwin 348/3,250，附加原条件 316/632。三套评测与审计后台仍运行；主 RoboTwin 5 个、附加 3 个案例在作者专家规划阶段异常、无策略动作，未删分母，完整成绩仍待有效补齐。详见 [最新运行快照](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md#最新运行快照)。

2026-10-10 最新：用户要求 π0.5 论文全量评测作为后续算法参照；[RawVLA-Bench 全量基线](../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md)已在新服务器由独立 supervisor 启动，固定作者 LIBERO 10,000 与 RoboTwin 3,250 条原始测试清单，共 13,250 条。重新运行，不复用部署成绩；RoboTwin 按论文 Table 2 Default 改用 SAPIEN `render_rgb`，照明/噪声清单 3,250 条核对无差异。首批实际 π0.5 成功、达到步数上限失败与基础设施异常均保留，固定每初态配对随机流；独立审计检查身份、重复分母和配对语言。当前 v3 补齐 Python/NumPy 指令种子；旧 RoboTwin 结果转诊断，全套重新执行。LIBERO 13 个重复身份均移出计分后重跑，原始记录与恢复哈希保留；进程退出控制已修复。另追加 632 条原缓存照明条件，核查论文 Table 2 与 3,250 manifest 的分母/曝光差异，两种结果分别报告。**仍在运行，尚无全量成绩或论文表格复现结论**；OptiX/Blackwell 与 StarVLA/原 LeRobot 数值边界详见记录。旧 Arena/PolaRiS 任务已停止，历史证据保留，未创建定时任务。

2026-10-10 新增：[RawVLA 作者环境 / π0.5 独立部署验收](../research/experiments/2026-10-10_rawvla_pi05_deployment.md)已在用户新服务器完成。RawVLA-Bench 全集 713.69 GB、4,810 个文件下载完成，1,771 对 LIBERO 与 632 对 RoboTwin RAW/RGB 清单无缺失。官方 `pi05_libero` 与作者指定 `SidneyXie/pi05_robotwin` 已部署，额外训练 0 步；LIBERO 四套各一条 RGB 成功 4/4，Spatial 首任务五照明域七初态 Default ISP 成功 19/35，RoboTwin 作者 HDR / Normal / Default ISP 的 `grab_roller` seed 0 在 81 步成功。模型实际输入、完整正常失败、动作与观测对齐和哈希已留档；仅为部署验收，完整论文协议与原 LeRobot 数值对照未完成。Blackwell 上显式采用 OptiX 替代异常的 OIDN，兼容改动详见记录。现成三通道 RAW 可用于 encoder 原型，不冒充原始 Bayer CFA；原 Arena 冻结基准与其他任务记录保留。

更新时间：2026-10-09。研究依据为 roadmap v6。本页区分仓库基础设施与尚未开始的研究工作。

## 已终止的历史路线：PolaRiS 重建与公开发布（2026-10-10）

2026-10-11 已按最新用户要求终止。本节以下保留当时的实施快照；不再扩展场景、修补抓取或公开上传。最终停止记录见两个旧实验的终止快照。

2026-10-11 01:54:21（北京时间）手动核查：旧重建队列全部结束，配对覆盖 35/316、通过回放和真实训练加载 18 条 / 1,808 帧，10,848 次 RAW/CFA 精确整数检查；尚未公开上传。独立冻结 RGB 基准完成 279/900（201 成功、78 正常失败），继续运行。用户已接受抓取约束，queue-v5/v6 已完成两条历史失败源身份，约束误差不合格，均未计入成功；queue-v7 校正源接近段桌面/初始化高度和启用时机后复测。

用户已授权重建全部 PolaRiS / Arena DROID 仿真训练轨迹，接受逐任务近似恢复、要求尽量接近源场景/RGB，并授权以 public 发布到 Hugging Face `Jeong-zju`；见 [0010 决议](decisions/0010-polaris-full-native-raw-reconstruction.md)。完整公开源已实读并逐分片核验：**316 条、52,914 帧、14 个源场景、17 条指令**，源身份无重复。全部源帧的两个 exterior JPEG 字段字节一致，三字段结构明确映射到外部与腕部两个独立相机。未恢复原物体初态、相机标定和全部源资产，新 RAW 配对同帧重建 RGB，原 JPEG 独立保留为参考。

方块入箱第 2 版完整回放 5 条，4 条物理成功、1 条有效失败；旋转任务完成 29 条，14 条成功、15 条有效失败，另 1 条锚点未解析，未替换源身份。当前配对采集覆盖 35 条独立源身份，全集仍未完成。2026-10-10 10:33:02 UTC，合并的 **18 条 / 1,808 帧**通过真实 LeRobot 保存/重载、官方 π0.5 DROID 输入变换与 15 步原动作块核验；6 个 RAW/CFA 字段共 **10,848 次精确整数检查**通过。实际状态、相机和传感器设置亦保存为独立审计数据。固定 LeRobot 的 RGB 统计名称判定问题已在新导出版本兼容，原生 uint32 按行无损存储并记录 180×320 恢复形状，没有修改全局依赖。

源机械臂状态驱动的三条失败诊断将关节偏差降至 0，任务仍失败；该模式保留原动作标签，但不声称执行原机械臂目标命令。源 marker 轨迹实际涉及多个积木、胶棒形物体和螺丝刀，单物体原型并不完整，现正核查多物体/再次抓取关系。14 个场景中有 10 个外部机位候选通过留出源帧重投影检查，但尚未恢复原始标定或完成图像对应审查；2026-10-11 用户已接受原动作触发抓取约束，按实际接近与当前相对位姿启停 USD 固定关节，保存事件和误差，辅助重建结果单列；正在进行新版本 GPU 验证，尚未加入公开集。

**全集仍未完成、尚未公开上传。** 其他任务场景正在扩展；逐条源 RGB 对应审查尚未完成，少量成功样例不代表全集。大数组和图像留在云端，本地只保存数字。查看 [固定全集重建进度页](../outputs/polaris-rgb-raw-reconstruction/progress.html) 与 [实验记录](../research/experiments/2026-10-10_polaris_full_reconstruction.md)；手动同步 `uv run --locked python -m owac.runtime.polaris_reconstruction_archive`。900 条基准的冻结源码、协议和历史不变，定时任务保持关闭。

## 仓库初始化

- [x] 本地 Git 仓库及 `main` 初始分支。
- [x] 主仓库位于 `owac/repo/`，后续 worktree 统一使用 `owac/worktrees/`。
- [x] README、AGENTS 规则、开发说明与初始架构边界。
- [x] 轻量 Python 包、配置/脚本/测试目录及数据/产物管理约定。
- [x] F01—F12 候选种子卡、完整研究卡/实验/评审模板。
- [x] 依赖锁定、安装、格式/语法及包导入检查。
- [x] 初始版本提交（本文件随初始化提交纳入版本历史）。
- [x] GitHub 私有仓库 `Jeong-zju/owac` 已建立并配置为 `origin`。

初始卡只登记研究方向和待解决问题，不等于完成 12 个框架设计。当前已完成 π0.5 后端离线部署验证；候选算法、RAW 采集、训练结果和机器人实验尚未开始。

初始化检查：迁移到 `repo/` 后重建本地 `.venv`，`uv sync --locked` 与 `make check` 通过；`owac` 及 8 个子包从新路径导入成功；47 个本地 Markdown 链接检查通过；数据、产物、私有配置的忽略规则及说明/清单保留例外检查通过。未运行算法或实机测试。

## π0.5 后端优先部署

用户主要负责网络搭建和具身 RAW 数据验证，选择 π0.5 主后端与 PyTorch；机器人本体、相机和数据留空。[0003 后端优先部署决议](decisions/0003-pi05-backend-first.md)取代了用户未接受的 0002 通用 B0 方案，旧前置清单不再生效。

- [x] 锁定官方 openpi `15a9616a00943ada6c20a0f158e3adb39df2ccac` 与 LeRobot 源码来源。
- [x] 建立独立 Python 3.11 / PyTorch 2.7.1+cu128 runtime 与完整依赖锁。
- [x] 应用官方 Transformers 4.53.2 补丁，验证环境与共享缓存隔离；3 项 CPU 行为测试通过。
- [x] RTX 5090 上 CUDA 矩阵运算与官方补丁检查通过。
- [x] 将已有 `pi05_droid` Orbax 缓存转换为 PyTorch，明确复制该检查点自身的归一化统计并计算哈希。
- [x] 预训练策略离线推理通过：输出 `[15, 8]`，有限值与固定输入重复性通过；3 次同步调用约 121—129 ms，PyTorch 分配峰值约 7.12 GiB。
- [x] 官方 WebSocket 协议的本地健康检查、元数据握手与客户端推理通过；检查后停止测试服务。
- [x] 审计官方 RGB 预处理、视觉前缀、π0.5 本体状态与动作专家边界。
- [ ] 先验收仿真 benchmark、RGB 闭环及成像/数据协议，见下节。
- [ ] 随后实现 RAW/外部特征独立入口，同时覆盖训练与推理并验证梯度和掩码。
- [ ] 后端、选定任务/数据协议及必要首版接入接口验收后合入本地 `main`，创建 `owac-dev-v0.1.0` 共同起点标签。

部署与 RAW 接入审计见 [π0.5 本地部署](pi05-deployment.md)。后端部署分支为 `lzh/pi05-backend`，当前仿真开发分支为 `lzh/raw-sim-bench`，正式基线标签尚未创建。DROID 配置只用于软件连通性检查，不代表正式本体选型。

上述资源数据来自合成 RGB smoke，编译关闭、10 步 flow matching；未测量正式任务或闭环性能。`make check` 与 3 项 CPU 测试通过，数据加载器导入通过，训练尚未执行。

最终代码 `b13efae869a34ffe655d0ea54307a95054f033e3` 在干净工作树下复现通过，动作与首次结果完全相同；证据保存于[部署验证记录](../research/experiments/2026-10-08_pi05_deployment.md)。

## 历史：仿真 benchmark 选型与验收（由 RAWVLA-bench 路线取代）

用户要求先依据所提供 Isaac Sim / RAW 调研报告选择合适的 benchmark，RAW 适配后置。[0004 决议](decisions/0004-simulation-benchmark-first.md)调整了 0003 的后续建设顺序；[选型调研](simulation-benchmark-selection.md)保存比较、来源和明确的未验证项。

- [x] 阅读 13 页报告并核对 NVIDIA 相机管线文档，提取线性 HDR、传感器 gap 和任务/数据选型条件。
- [x] 比较 Lightwheel-LIBERO、Arena DROID/厨房任务、原版 LIBERO/RoboCasa、BEHAVIOR 及其他操作平台；核查 RawVLA-Bench 的三通道 pseudo-RAW 边界及训练/测试协议。
- [x] 核查固定提交的部分源码和依赖；记录 LW 的 Sim 5.0 约束、Arena 的 jointpos 权重差异及一个双杯任务的疑似错误成功分支。
- [x] 将 Lightwheel-LIBERO 桌面子集列为主 benchmark 首选，Arena DROID 示例列为首个工程闭环；此为调研建议，未完成运行验收。
- [ ] 验证固定仿真环境、资产获取、π0.5 PyTorch jointpos 参考路径及 RGB 闭环。
- [ ] 验收首选套件的版本兼容、3 个任务的语言/成功判据、重置和演示重放。
- [ ] 在所选任务相机中验证未色调映射的线性 HDR 出口、时间戳与可复现成像条件。
- [ ] 冻结任务/数据协议后再实现 RAW 接入；真实相机与正式数据仍留空。

本轮只完成文档/源码调研与顺序修订，没有安装候选仿真或运行 benchmark，也没有测得候选任务成功率/资源指标。来源及哈希见[调研清单](../research/reviews/2026-10-08_simulation_benchmark_sources.json)。

## 单任务 RAW 仿真部署（2026-10-08）

按用户“只需要一个能运行起来的支持 RAW data 的 sim bench”的要求，已部署 Lightwheel-LIBERO 的 `L90L5PutTheRedMugOnTheLeftPlate` / `libero-1-1` / Panda，使用独立 Isaac 6.1 runtime 与明确的兼容补丁。环境创建、reset、控制步和连续 32 帧原生 GRBG CFA 导出已通过；与线性 HDR 的量化误差为 0 DN，uint32 无损存储及时间戳/文件校验通过。`make check` 与 8 项 CPU 测试通过。

入口：`bash scripts/run_raw_sim_bench.sh --steps 32`。见[部署说明](raw-sim-bench-deployment.md)和[实验记录](../research/experiments/2026-10-08_lw_libero_raw_deployment.md)。这是合成传感器 RAW；真实相机、曝光/增益标定留空。当前控制保持位姿，原成功判据为 false；未测策略成功率、全任务套件或 π0.5 闭环。旧选型章节中的未验收项仍适用于正式完整协议，单任务运行已由本节更新；共同起点标签仍未创建。

## RAW 文献与具身增量修订

- [x] 定向核对 Ziteng Cui 的 12 条 RAW/视觉/机器人工作线索，补充相邻事件 VLA 与动作条件化模型边界，见[专题综述](../research/reviews/2026-10-08_ziteng_cui_raw_physical_intelligence.md)和[来源清单](../research/reviews/2026-10-08_ziteng_cui_raw_sources.json)。
- [x] 将物理决策增量写入 roadmap v5、[0005 决议](decisions/0005-physical-intelligence-contribution.md)及候选/实验/评审模板。
- [x] 为 F01—F12 各补一项具身增量假设、已有边界及首项反证，卡片为 v0.2；全部仍是 `seed`。
- [ ] 推导每个候选的具体新规则并完成机制近邻检索，证明不能由普通适配或额外信息解释。
- [ ] 运行成像干预与物理/目标干预实验；按实际证据建立 RSS 分支。

本次完成研究文档修订，没有复现这些论文或运行候选/机器人实验；文献可访问不代表代码已具备复现条件。0004 的仿真与数据协议优先顺序继续有效。

## 生物弱光与补全机制修订（2026-10-09）

- [x] 核对弱信号传递、时空求和、循环视觉补全、记忆、自运动、跨感官和主动搜索的一手研究，见[机制综述](../research/reviews/2026-10-09_bio_lowlight_inference.md)与[来源记录](../research/reviews/2026-10-09_bio_lowlight_sources.json)。
- [x] roadmap v6 与[0006 决议](decisions/0006-bio-inspired-inference.md)将 PI1—PI5 改作评价维度，新增六类机制问题 N01—N06，保留 F01—F12。
- [x] 12 张候选卡更新为 v0.3，补充生物证据边界、拟探索规则及反证；仍全部为 `seed`。
- [ ] 推导精确计算规则，并与递归估计、循环网络、联想记忆和主动感知等最近方法建立可检验差异。
- [ ] 建立测量质量、先验有效性、运动/反应期限和反证到达实验，分别检验积分与补全假设。

本轮仅修订研究设计，未实现或运行候选算法。单任务仿真部署事实沿用上节，完整噪声/动态任务/接触及 VLA 闭环协议尚需验证。

## 学术研究方案（2026-10-09）

- [x] 将用户认可的路线整理为独立的 [Markdown 学术研究方案](OWAC_academic_research_proposal_2026-10-09.md)，包含摘要、相关工作、问题定义、框架图、机制假设、实验协议、实施安排与 22 项参考文献。
- [x] 明确 N01/N02/N04 的核心编码位置、N03 的记忆支持位置及 N05/N06 的可选系统扩展位置；附录继续保留 F01—F12。
- [x] 区分生物与已有方法的证据、待研究的计算规则和当前工程状态；本文不报告尚未获得的算法或实机结果。

本次仅交付和同步研究文档，格式保持 Markdown，未执行候选算法、训练或机器人实验。

## 后续研究待办

以下对应 roadmap 的首阶段工作，均未完成；具体负责人和评审日期待安排。

- [ ] 补全 12 张完整框架卡，检查等价性、重叠和最近方法差异。
- [ ] 按机制多样性安排首轮约 5—6 个最小实现，保留其他家族的后续机会。
- [ ] 定义 RAW 采集/校准/元数据规范、清单格式及数据拆分协议。
- [ ] 以不同内部表示的原型验证共享外部接口、状态生命周期与成本记录。
- [ ] 审计候选 VLM/VLA 的训练/推理预处理和可训练参数边界。
- [ ] 建立最近机制基线以及 C0—C4 的系统比较配置和评测协议。

主后端已选择 π0.5，网络开发采用 PyTorch。相机、机器人接入、正式数据、训练/后端适配容量及第二后端尚未确定；RSS 证据分支及其冻结版本待实际证据形成后建立。

## 完整任务对比视频（2026-10-09）

已连续录制单任务真实物理抓取、抬升、搬运、放置和退开，成片 20.6 s / 515 帧 / 25 fps。红杯抬升 16.75 cm，最终杯盘 XY 距离约 0.28 mm，最后 25 帧原任务判据均通过。逐帧原生 CFA/noise、RGB PNG、本体/物体/动作状态和哈希保存完整；同帧 HDR → CFA 最大量化误差 0 DN，时间间隔 40 ms 与无损数组检查通过。`make check`、10 项 CPU 测试及原默认配置 4 步 GPU 回归通过。

入口为 `bash scripts/render_raw_task_video.sh --output outputs/task-video/<新目录>`。视频左侧 RGB，右侧固定 log 映射的 CFA 灰度与 GRBG 格点放大；原始 uint32 不受显示或视频压缩影响。演示使用当前物体真值控制，并明确加固定底座及调整机器人安装位置；不作为官方原配置分数或 π0.5 闭环结果。原入口仍保持原配置。详见[部署说明](raw-sim-bench-deployment.md)与[实验记录](../research/experiments/2026-10-09_raw_rgb_task_video.md)；大产物在 `outputs/task-video/full-task-04/`。正式基线标签仍未冻结。

## π0.5 云端单任务基线（2026-10-09）

用户授权在租用 GPU 上微调现有 sim bench，并用 100 个独立随机 episode 验证，只回写本地结论和数字，检查点留在云端。已部署两套独立 runtime、固定源码与资产，并下载/转换官方 `pi05_libero` 初始化权重。随机专家采集已得到实际抬升、放置、松手且原任务成功的完整 episode；全模型反向传播和两步微调检查通过，独立开发种子已执行完整 30 s 闭环（未完成任务，不作为正式分数）。正式采集—训练—100 次测试流程于 2026-10-09 03:17:02 UTC 由云端 supervisor 启动，采集已完成：90 条训练演示（46,350 帧）和 10 条开发演示（5,150 帧），共 142 次尝试，42 次专家失败同样留档；04:42:44 UTC 已进入正式全参数微调。

固定比较协议为 [pi05-lw-mug-v1.json](../configs/baselines/pi05-lw-mug-v1.json)：90 条训练 / 10 条开发演示，最终独立种子 3000 至 3099，统一状态/动作/时序及成功规则，开发损失选模。工程任务沿用已验证的固定底座配置，不报告上游原配置分数。实验身份、训练精度、种子与预算、失败诊断及后续实测结果见[云端基线记录](../research/experiments/2026-10-09_pi05_lw_mug_baseline.md)。每小时自动检查任务已按用户要求删除；最终结果已于北京时间 2026-10-10 09:52:40 手动回写并核验。规定种子 3000 至 3099 的 100 次闭环测试全部完成，成功 0 / 100（0%，Wilson 95% 区间 0%–3.69935%），动作异常与基础设施异常均为 0；不创建共同起点标签。

固定进度入口为[本地 HTML](../outputs/cloud-pi05-baseline/progress.html)，自动检查已关闭，当前显示最近一次手动同步的快照，页面自动刷新默认关闭。展示云端最近检查时间、训练/开发损失、停止条件、检查历史和全部测试种子的完成状态；未运行测试显示未测量，检查异常保留旧快照并提示。最终页面显示完整 100 次结果：训练完成 10000 步，按预定开发损失停止规则选择第 1000 步模型；100 条均正常运行至超时，均未达到 0.10 m 抬升阈值。这是有效的单任务负结果，开发损失平台期不代表闭环任务成功。推理 p50 / p95 为 113.3552 / 116.0386 ms；完整逐条结果、模型与协议哈希以及本地校验清单已留档，未下载检查点。云端流程已正常结束，租用实例与权重保留。桌面/手机及离线页面验证、`make check` 和 17 项 CPU 测试通过；本地页面更新未改动云端冻结训练源码。

已进行 [π0.5 失败诊断](../research/experiments/2026-10-10_pi05_failure_diagnosis.md)：开发种子专家动作重放成功；选中模型在开发专家观测上的位置平均误差约 3.64 cm，闭合阶段夹爪符号准确率 66.67%；全部演示发现停留/夹爪切换边界的近似观测歧义。原分数保留，但当前自定义适配尚不能作为已验收的强 π0.5 对照。修复与选模仍须使用开发数据，新协议版本化后重新冻结比较条件。

## RGB 参考验收：官方预训练 LIBERO（2026-10-10，主 benchmark 地位已撤回）

用户已否定 Lightwheel 红杯任务与自定义 π0.5 微调的组合，要求优先直接可用的 backbone/benchmark，并明确可以先用原版 RGB。[0007 决议](decisions/0007-pretrained-benchmark-first.md)取代 0004 的主 benchmark 推荐；旧红杯 0/100 与诊断保留为历史负结果，不再追加 backbone 训练或接口调参。

已选择原版 LIBERO-Goal 全部 10 项任务与官方 `pi05_libero`，额外训练 0 步。独立 Python 3.8 模拟器已部署；复用官方原始权重、归一化和评测循环，保留官方相机、动作、初始状态与成功判据。2026-10-10 02:46:43 UTC 完成全部 10 项任务各 1 条验收，成功 **10/10**、基础设施异常 0；逐条任务/初始状态、计数和哈希均核验，确认直接可用，小样本不冒充官方完整评测。新协议、版本与兼容修复见[实验记录](../research/experiments/2026-10-10_pi05_official_libero_reference.md)，固定入口为[新 RGB 结果页](../outputs/official-libero-baseline/progress.html)。定时检查不重建，旧云端检查点不删除。本方案仅是 RGB 参考，不声称具备原生 CFA/RAW。

## 当前准入：现成策略与原生 CFA/RAW 同时满足（2026-10-10）

用户明确原版 LIBERO 的第二次选型仍失败：开箱即用的基础上，仿真器必须支持原生 CFA/RAW。[0008 决议](decisions/0008-pretrained-and-native-cfa-required.md)撤回 0007 中 RAW 暂不准入的安排。LIBERO 10/10 保留为 RGB 参考，旧 Lightwheel 红杯保持归档；当前已建立同任务现成策略与原生 RAW 的可运行工程基线，正式主 benchmark 的扩展协议尚未冻结。

本轮部署 Arena DROID 抓放与官方 `pi05_droid_jointpos_polaris` / `pi05_droid_jointpos` 专用权重。已核对官方成功示例与 Isaac CameraCore ISP 前 CFA/噪声输出，审计 Arena `2cc9e9766f1f62a500b85c92207192d8c60fcce5` 的固定源码、Lab gitlink、OpenPI 提交及哈希；[来源清单](../research/reviews/2026-10-10_pretrained_raw_benchmark_sources.json)保存证据。安装文档仍称 Sim 6.0.1，但固定源码 Docker 底座已是 6.1.0，部署必须按锁定源码核验，不能直接复用旧兼容子集。已在独立云端 runtime 完成 Arena 的策略 / RAW 联合实测：RGB **3/3**、同步 RAW **2/3**，额外训练 0 步。每个阶段完整运行预定种子 0/1/2；RAW seed 2 正常完成 1050 步超时，已抬升但未满足入碗判据，作为有效失败保留。两路各 1629 帧、共 3258 帧原生 HDR/CFA/noise，9,774 个保留缓冲已在云端逐文件重读并核对 SHA256，量化误差 0 DN；权重与大数组均留在云端，本地只存数字与校验清单。详见[联合实验记录](../research/experiments/2026-10-10_pi05_arena_droid.md)与[新离线结果页](../outputs/arena-droid-baseline/progress.html)。一次 SIGBUS 已在新目录恢复原 seed，完整旧记录复用而不重复计数；底层原因尚未确定。额外渲染改变了 RGB 观测序列，长期稳定性与统一采集对比时序尚待验证，不追加 backbone 训练。两个本地历史结果页已更新用途与准入提示，远端成功检查时间和所有原始分数保留；定时任务仍关闭。

## 当前交付完整基准与训练演示回放（2026-10-10）

用户确认 [0009 两阶段交付](decisions/0009-baseline-then-training-replay.md)，并明确固定 Isaac Sim / Arena DROID、仿真器原生 RAW，不采用 MuJoCo。第一阶段先取得完整 RGB 基准，第二阶段恢复专家演示的场景和状态，回放采集配对 RGB/RAW。历史 3/3 与 2/3 仅为单任务部署验收，3,258 个视角帧来自策略评测同步采集，不属于原训练演示回放。

第一阶段已冻结[九组官方变化、每组种子 0–99、共 900 条](../configs/baselines/pi05-arena-droid-benchmark-v1.json)，复用官方 jointpos 权重，额外训练 0 步，官方相机/动作/成功判据不变，RAW 额外渲染不介入基准。2026-10-10 05:49:24 UTC 已由 `owac-arena-benchmark-v1` 启动；每条使用新策略和仿真进程，保留正常失败，基础设施故障只允许原身份新目录重试一次，再次故障停止。最初两条已完成成功，900 条尚未完成。实际最新检查时间与逐配置进度以[固定离线页面](../outputs/arena-droid-baseline/progress.html)和[小型数字留档](../research/experiments/results/2026-10-10_pi05_arena_droid_benchmark/)为准，历史页面移至 `history.html`。协议、冻结源码哈希、运行位置和成本口径见[完整基准记录](../research/experiments/2026-10-10_pi05_arena_droid_benchmark.md)。

第二阶段的[源码/数据核查](../research/reviews/2026-10-10_arena_droid_expert_replay.md)已完成：PolaRiS 确有公开仿真共同训练数据，固定 OpenPI 配置混合真实 DROID 与仿真数据，不能将来源全部描述为真实数据；配置不单独证明实际检查点完整训练历史。公开仿真 RLDS 缺少逐帧物体状态，未找到九组场景对应的原始专家状态文件。官方 `record_demos.py`、Mimic `generate_dataset.py`、`replay_demos.py` 和机器人/物体状态 recorder 已核查并留哈希，但当前 DROID `mimic_env=None`，默认 SE3 遥操作与八维 jointpos 动作未接通，没有核验到直接可用的自动专家生成器。GR1/G1 数据和 Franka 方块状态机不是当前 DROID 的现成替代。这些脚本仅完成静态适用性核查，尚未在当前 DROID 上进行专家采集 GPU 验收。

2026-10-10 06:23:45 UTC 已在云端实读一条公开 PolaRiS 演示：310 个语言帧，三路 RGB、关节/夹爪状态和专家动作，来源指向作者本机 `kanav_cloth_chairs/episode_14.npz`。公开包未找到该原始 NPZ，尚未核验训练场景映射与 episode 初态。修正回放门槛：完整逐帧物体状态不是必要条件；若能恢复原场景、初态、控制与相机时序，可物理重放原动作并测量参考偏差。公开元数据合计 316 条，论文 OOD 表为 326 条；过滤清单另有 406 个键，首条数据与相关清单的 `Secondary1` / `Secondary` 路径不匹配，不能声称实际训练样本覆盖已核验。详细来源、哈希和范围见[训练数据实读审计](../research/reviews/2026-10-10_pi05_training_data_audit.json)。

在上述来源核查阶段，尚未生成配对训练集；随后用户要求的实际回放试验与结果见下节。不将评测策略轨迹当原始专家，不追加训练。云端完整基准持续运行，聊天无需保持运行；定时任务关闭，实例、历史记录和权重保留。手动同步命令为 `uv run --locked python -m owac.runtime.arena_droid_benchmark archive`；只回写小型数字，页面生成时间不冒充云端成功检查时间，本地归档修改不覆盖冻结运行代码。

## 原始专家动作回放试验完成（2026-10-10）

用户最新要求实际尝试机器人状态/动作回放，以能采到成功专家数据为试验目标。已实读公开 PolaRiS 首分片十条演示，并选取 `pi_scene_v5/episode_14.npz` 的“把方块放入左箱”原动作，在 Isaac Sim 6.1 / Arena DROID Franka+Robotiq 上完成物理回放。结果为 **一条成功源专家演示、108 帧双视角 RGB/原生 CFA**：真实抬升 24.1509 cm、搬运 46.0431 cm、松手后稳定入箱；没有更改原动作，初始化后物体状态写入为零。这是依据抓取/释放位置重建的场景，原训练场景与相机映射未恢复，新配对 RGB/RAW 不冒充原训练 JPEG 的精确补采，也不是 π0.5 的评测成绩。

216 个视角帧的 864 个 RGB/HDR/CFA/noise 文件已在云端逐文件重读，哈希不一致为零，原生 HDR→CFA 最大误差 0 DN；每帧计数与源动作时序匹配，腕部相机与同帧实际夹爪位置最大偏差 0.015673 mm。成功数据集含源动作/状态、实际完整机器人和方块状态、两路新 RGB/RAW、传感器、时间戳、哈希和重建 USD；云端入口为 `/workspace/owac/reference/polaris-replay-pilot-v1/successful-expert-index.json`。数据数组与图像留在云端，本地仅存[数字留档](../research/experiments/results/2026-10-10_polaris_expert_replay_pilot/)和[试验记录](../research/experiments/2026-10-10_polaris_expert_replay_pilot.md)，固定[进度页](../outputs/arena-droid-baseline/progress.html)已加入回放结果。

前四次启动/接口异常完整保留；第 5 次物理回放与配对数据保存完成后发生 NumPy bool 的 JSON 序列化错误，已用独立 CPU 后处理源码修复数字结果，未重跑物理或重新采图。完整 π0.5 基准只暂停外层调度，当前子 episode 自然结束后才运行试验；调度器已恢复，冻结源码、种子和已完成结果没有修改。定时任务仍关闭，不追加训练。`make check` 与 40 项 CPU 测试通过。

2026-10-10 08:01:47 UTC 手动更新完整基准：计划 900 条，已完成 54 条（44 成功、10 正常策略失败），846 条尚待完成，云端服务仍在运行。三次基础设施异常均已按原身份重试取得完整结果；未完成条目不计为无效数据或失败。页面将“有效完成”改为“已完成 / 计划”，并显示待完成数，避免把计划总数误认为已经执行的数量；仅改本地展示，定时任务保持关闭。

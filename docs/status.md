# OWAC 当前状态

2026-10-14 N04 v1.0 **S0 远端补齐验收通过**：两平台 40 条成对扩展、960 时刻 / 2,880 相机帧，以及 16 因素 / 152 条真实三维渲染诊断；C0–C7 为 8 passed、0 partial，本地与远端各 361 tests passed。实际时序、执行控制、几何支持与输入隔离均已验收，按训练相机重新冻结米制门槛。S1 尚未启动。详见[本轮记录](../research/experiments/2026-10-14_n04_v1_s0_completion.md)。


2026-10-14 较早的 N04 S0 **独立验收修补（历史，已由上述远端补齐取代）**：原交付存在在线控制绕过证据等级、时间域比较、D1 假对应/空采集成功、热启动和连续性问题，已修补并通过 349 项本地测试及远端合同复跑。当前严格状态为**核心工程合同通过、完整 S0 数据交付 partial**（C5 仍 partial）。新增 LIBERO 两任务四条独立扩展（32 时刻/64 相机帧），连续执行与独立几何射线核验通过；RoboTwin live geometry 已实测，但历史控制与完整成对训练集仍缺。未进入 S1。详见[独立验收记录](../research/experiments/2026-10-14_n04_v1_s0_independent_acceptance.md)。

2026-10-14 N04 v1.0 **S0 原交付记录（完整通过结论已由上方独立验收取代）**（未提交，提交由用户安排）：[阶段记录](../research/experiments/2026-10-14_n04_v1_s0.md)、[handoff](../research/experiments/results/2026-10-14_n04_v1_s0/handoff.json)。新增 `raw/telemetry.py`、`raw/n04_manifest.py`、`raw/n04_mechanism_data.py`、`frameworks/n04/measurement_geometry.py`、`evaluation/n04_contracts.py`、`scripts/n04_full/*`、`configs/n04/full/{s0_data,protocol}.json`、`research/theory/n04_v1.md` 与 7 个测试文件。远端**只读**盘点 2403 个真实 NPZ（LIBERO 1771 + RoboTwin 632，474.2 GiB）→ 2403 条独立源轨迹、每源 1 变体；主拆分 train/dev/final = **1917/251/235**（source ID 三项交集全为 0，成员表 hash `0d3d4191…`，两次独立生成逐字节一致）。字段覆盖表与 8 条不可用条件清单已冻结：**两平台时间戳与相机标定不可用**（只能 `time_unit="frame"`）、**RoboTwin 已执行动作不可用**、LIBERO 已执行动作仅 `producer_code_order`（provisional）。LIBERO 动作相位经同版本仿真核实（k=+1 中位误差 7.06e-4/1.63e-4/3.24e-4，正确/错误相位好 70–300 倍），但**位精确重放不成立**（缺 `qacc_warmstart`/控制器缓冲，单步 max|Δ|=0.098/0.176/0.487）→ 不升级证据等级。D0 交付 16 因子解析三维机制数据（B=1/2/4 逐样本审计：unexplained=missed=0、跨样本污染 0.0、静态残差 0.0、历史 `[B,1,B]` 广播缺陷可被检出）；D1 按用户裁决 **partial**（7 available / 4 partial，RoboTwin live env 未建）。合同验证 C0–C7：0 failed，C5 因真实时间戳/标定缺失为 partial；审计发现并修复 `N04Encoder.step` 的 NaN×掩码缺陷（+9 行，独立复核正常路径 66 张量逐位不变）。本地 324 tests、`make check` 通过（修复了准备记录中 1 处文档排版导致的失败）；远端新 [encoder 环境](../research/experiments/results/2026-10-14_n04_s0_infra/environment.json) Python 3.12.3 / torch 2.14.1+cu130 与 `uv.lock` 一致。**官方测试清单不参与训练拆分**，但只读核查发现训练 NPZ 与官方 10000+3250 清单派生自同一源 episode/初始状态池（RoboTwin 字段级确认、LIBERO 高可能性推断）→ S1/S6 不得据此宣称「未见过的 episode」泛化。S0 无网络训练、无 GPU 训练、未启动重复评测；S1 启动前必须重新打包同步快照。

2026-10-14 N04 v1.0 正式开发 [DeepSeek 提示词包](prompts/n04_v1_0/README.md)已按 S0–S7 编制，含源码差距、明确接口/训练/验收、参考实现与用户指定远端训练规约。只读核实远端数据、权重及 StarVLA Python 3.10 环境；远端最新完整基线报告和身份审计为 13,250/13,250、errors=[]，附加原条件 632/632。当前 N04 198 项定向测试及 make check 通过；本次交付为提示词准备，没有实现新阶段或训练新模型，没有创建提交。历史基线进度与完整证据范围见[准备核查](prompts/n04_v1_0/preparation_audit.md)。

2026-10-14 N04 全量开发方案 v1.0 已编制：[Markdown](plans/N04_full_development_v1.0.md) / [HTML](plans/N04_full_development_v1.0.html)。当前 HEAD `fd6255b` 已保留作者 RKN；此前 FilterPy 修改继续留在 stash。新方案包含 34 项论文 / 原始技术资料，明确 ConvNeXt-B、RKN / Ac-RKN 的正式配置，按 S0–S7 单因素诊断推进至 π0.5 在 RawVLA-Bench 两平台的充分训练与完整闭环。实际执行动作、时间和标定字段缺口列为首阶段前置；理论只承诺有假设的局部性质。候选仍为 prototype，本轮未执行新训练、未更改核心代码、未创建提交。

2026-10-13 N04 同测量动态对照与独立 RAW 诊断完成（未提交，提交由用户安排）：冻结单帧 RAW stem/定位头，600条最终结果、120组输入逐位相同；匀速缺失误差无滤波1.203px、FilterPy0.203px、作者RKN0.252±0.012px，三个初始化均有局部动态收益，停止/反向仍有滞后。噪声归一化可观测RAW变化召回84.92%→88.38%，留出误报0.99%，噪声组间取舍另报。同KF更新的真值可见速度只再改善0.029px，未达到预定训练触发值，本轮未训练Dynamo-Depth。256 tests与make check通过；N04核心未改，其新增分解/change收益及真实闭环仍未建立。见[完整记录](../research/experiments/2026-10-13_n04_matched_tracking.md)。

2026-10-13 N04 已有方法与代码复用评审完成：核查 FilterPy、RKN、Dynamo-Depth、DSO、OC-SORT、FloRNN、CoTracker 与 Dreamer 的关键接口，固定8个仓库提交，并下载检查 EMVD 后续工作的 IPOL 源码。建议先用同测量的 FilterPy/RKN 验证动态状态，再独立验证 RAW 噪声尺度及几何解释；明确未来帧、RGB、采样坐标和许可证边界。见[文献与源码评审](../research/reviews/2026-10-13_n04_existing_methods/README.md)。本轮未改核心、未训练外部方法，没有新增性能或机制通过结论。

2026-10-13 N04 分阶段重建完成：按单因素顺序修复数据、观测参考与支持掩码，加入实际RAW域增益解释和组合训练，随后验证动态记忆。36权重、108开发suite、54最终13条件suite与两组各297动态干预身份/哈希审计通过；248 tests、make check通过。留出定位0.332→0.310px、三因素共存漏报20.6%→6.6%，但同分布无变化分支0.303px、仅提示0.302px更好，误报也上升。缺失观测时change不进入定位状态预测，动态记忆增量仍未建立；没有本轮真实RAW/π0.5闭环或scaling收益。见[完整阶段与验收记录](../research/experiments/2026-10-13_n04_staged_rebuild.md)。下列旧批次缺陷诊断及旧结果保留为历史。

2026-10-13 N04 最新机制诊断修正先前结论：发现 `synthetic.py` 将 `[B,1,2]+[B,2]` 广播为 `[B,B,2]`，图像默认只画一个物体，标签却合并多个样本的物体位置。无噪声复现中 batch=2 每张图有 113 个假标签像素，batch=4 有 339；当前训练/评测分别使用 4/2，物体定位、漏报和消融排序需在正确数据上重训复验。此前哈希/输入不变量审计漏掉了渲染与标签逐样本一致性；215 项工程测试不保证数据正确。另在 batch=1 完全静态控制下，真值零运动时残差均值 0.24795，仅旁路 block 表征修正后为 1.33e-8，确认内部表征改写混入 innovation。照明增益仅作为分类线索、变化写入仍使用完整残差，组合运动/照明训练缺席。**本轮完成诊断和解决方案，未修改核心或重训**；既有数字保留但标签相关机制归因撤回待验。详见[诊断与实施顺序](../research/reviews/2026-10-13_n04_mechanism_diagnosis/README.md)。

2026-10-13 N04 事件分数统一完成：四种原因支持改为独立分数，外部事件始终监督“运动存在”；归一化 mix 仅供诊断，旧占比权重禁止静默加载。新增语义回归，全仓 215 passed、make check 通过。固定协议完成 42 checkpoint、252 次 suite（三初始化、continuity/fresh 各三评测种子）；源码、权重、评测身份和配对输入不变量审计通过。fresh 移动物体像素漏报 66.221%→53.448%，三个初始化都改善；相机误报 1.725%→2.533%、照明误报 2.247%→2.666%，定位 2.010→2.019px，未稳定改善。自运动比 1.064、照明比 1.516 仍 >1；单独关照明对照更好。完整候选定位好于 no-change 2.220px，但运动补偿 1.948px、filter 1.999px 仍更低；遮挡恢复候选 3/9 censored，两简单基线 0/9。**语义修复通过，外部变化分支有内部定位增量，整体机制收益未通过**。候选保持 prototype；详见[当前复验](../research/experiments/2026-10-13_n04_event_semantics.md)与[候选卡](../research/candidates/N04.md)。只验证 tiny 合成数据，不转称新真实动作训练或闭环收益；冻结全量 benchmark 独立保留。

2026-10-13 N04 六项源码修复完成：可信度断梯度、原因分支对称/stability 未用、残差 cell 与幅度标签错位、历史 change 未传播、非线性特征伪曝光增益、局部增益忽略 mask 均已修补。保留核心预测/分解/两种记忆及原测量读出；新增 19 项行为回归，全仓 206 passed、make check 通过。修复版、十个基线/消融及冻结传播同数据各训 200 步、各测三个种子，源码/权重 hash 审计通过。定位 2.149→2.041px，自运动比 1.396→1.194、照明比 1.744→1.548；但移动目标像素漏报 49.476%→68.568%，运动补偿 1.889px、无 change 1.906px 仍更好。**源码缺陷修复通过，核心机制收益仍未通过**。详见[修复与对比](../research/experiments/2026-10-13_n04_six_fixes.md)及[候选卡](../research/candidates/N04.md)。本轮合成验证与接口回归不转称真实 π0.5 动作/闭环收益，冻结全量 benchmark 保持独立。

2026-10-12 N04 独立复核与修补：原交付未满足提示词。查出 optimizer 漏纳懒建层、错误位移限幅、无效定位指标、多尺度未进损失、CFA 运行缺陷、同源 variant 拆分泄漏、状态/mask 语义以及虚构 π0.5 API。已修补并重训 tiny、五最近基线、四消融、C4 与冻结传播；两套真实 NPZ 各 20 步、真实预训练权重动作前反向与独立策略服务完成。详见[独立验收](../research/experiments/2026-10-12_n04_acceptance.md)和[候选卡](../research/candidates/N04.md)。187 项全仓测试与 make check 通过。候选独立定位均值 2.149px、变化幅值比 1.396、照明比 1.744，移动目标像素漏报 49.476%；小模型实现/接口可交付，H1/H2 与机器人任务收益仍未建立；不恢复或改动冻结全量 benchmark。

2026-10-12 N04 原交付历史（下述数字被后续独立审计限制解释，不作为当前验收）：分支 `lzh/n04-egomotion-separation` 落地阶段 1—4 的**可执行实现**（解析几何与可检查传播 → 三分支创新分解与更新 → 合成可控测量场训练与 checkpoint → 真实 NPZ 训练入口与 π0.5 特征接入）。实际执行：合成 tiny（宽 32、2 层、多尺度 2）200 步训练，`location` 0.0756→0.0253、`ego_warp` 0.6025→0.0494；机制矩阵评测 p50 11.85 ms / p95 15.31 ms、状态 7.90 MB；`tests/encoders/n04` 104 项与 `make check` 通过。**关键负结果（同预算对照）**：候选、无 ego、上下文错位、无变化分支、全残差当噪声五个配置各训 200 步后，`camera_only/object_only` 变化响应比 1.200—1.531（候选 1.391，**方向与假设相反**），照明误报 0.949—0.969（候选 0.959），去掉变化与照明分支后定位误差/residual 几乎不变；identity 传播的定位误差反而更低（0.0232 vs 0.0352），但 warp error 0.614→0.346 说明传播本身有效。因此“分解把自运动与外部变化分开”的假设在本轮 tiny 200 步上**未被支持**，且变化分支幅度仅 ~1e-3（stable 场 O(1)），当前指标可能还测不到差异。运行中另修复 7 个真实缺陷（评测未注入标定/延迟参数导致条件退化、遮挡掩码没有进入编码器、合成 proprio 泄漏物体位移、变化幅值先压标量使目标处/目标外恒等、`change_branch=False` 崩溃、几何层不收逐样本平移、消融 checkpoint 前缀不匹配）并全部留下回归测试。真实 RawVLA 数据与权重部署在服务器 `/workspace/owac/rawvla/`，本机不可达，`--npz-dir` 以退出码 2 精确报缺，**未做真实数据与真实 π0.5 权重验收**。见[候选卡](../research/candidates/N04.md)与[实验记录](../research/experiments/2026-10-12_n04_encoder.md)。本条表示实现与合成机制证据，不表示机制成立或后端验收通过。

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

## 本次固定以前的历史状态

以下为旧版本状态快照；涉及 Arena/PolaRiS 的“继续运行”及旧 benchmark 选择已被 0011/0012 取代。

更新时间：2026-10-08。研究依据为 roadmap v4。本页区分仓库基础设施与尚未开始的研究工作。

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

## 当前优先：仿真 benchmark 选型与验收

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

按用户要求先报告缺口，再决定是否适配官方遥操作/Mimic，或新增 DROID 脚本/规划专家。上述来源核查阶段尚未生成新专家或配对训练集；本轮实际回放结果见下节。不将评测策略轨迹当原始专家，不追加训练。云端完整基准持续运行，聊天无需保持运行；定时任务关闭，实例、历史记录和权重保留。手动同步命令为 `uv run --locked python -m owac.runtime.arena_droid_benchmark archive`；只回写小型数字，页面生成时间不冒充云端成功检查时间，本地归档修改不覆盖冻结运行代码。

## 原始专家动作回放试验完成（2026-10-10）

用户最新要求实际尝试机器人状态/动作回放，以能采到成功专家数据为试验目标。已实读公开 PolaRiS 首分片十条演示，并选取 `pi_scene_v5/episode_14.npz` 的“把方块放入左箱”原动作，在 Isaac Sim 6.1 / Arena DROID Franka+Robotiq 上完成物理回放。结果为 **一条成功源专家演示、108 帧双视角 RGB/原生 CFA**：真实抬升 24.1509 cm、搬运 46.0431 cm、松手后稳定入箱；没有更改原动作，初始化后物体状态写入为零。这是依据抓取/释放位置重建的场景，原训练场景与相机映射未恢复，新配对 RGB/RAW 不冒充原训练 JPEG 的精确补采，也不是 π0.5 的评测成绩。

216 个视角帧的 864 个 RGB/HDR/CFA/noise 文件已在云端逐文件重读，哈希不一致为零，原生 HDR→CFA 最大误差 0 DN；每帧计数与源动作时序匹配，腕部相机与同帧实际夹爪位置最大偏差 0.015673 mm。成功数据集含源动作/状态、实际完整机器人和方块状态、两路新 RGB/RAW、传感器、时间戳、哈希和重建 USD；云端入口为 `/workspace/owac/reference/polaris-replay-pilot-v1/successful-expert-index.json`。数据数组与图像留在云端，本地仅存[数字留档](../research/experiments/results/2026-10-10_polaris_expert_replay_pilot/)和[试验记录](../research/experiments/2026-10-10_polaris_expert_replay_pilot.md)，固定[进度页](../outputs/arena-droid-baseline/progress.html)已加入回放结果。

前四次启动/接口异常完整保留；第 5 次物理回放与配对数据保存完成后发生 NumPy bool 的 JSON 序列化错误，已用独立 CPU 后处理源码修复数字结果，未重跑物理或重新采图。完整 π0.5 基准只暂停外层调度，当前子 episode 自然结束后才运行试验；调度器已恢复，冻结源码、种子和已完成结果没有修改。定时任务仍关闭，不追加训练。`make check` 与 40 项 CPU 测试通过。

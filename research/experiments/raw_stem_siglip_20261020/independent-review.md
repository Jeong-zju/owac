# Sol 独立验收：RAW stem／SigLIP 蒸馏工程

2026-10-20。**结论：每帧基线的工程契约及实际同源权重接口得到有边界的验证；完整 RAW 教师、时序训练链、学习和控制未验收。实际优化更新0，新闭环0。** 主仓库 main 上保留原有修改，未推送。采用决议沿用0016；本报告不另立采用决议。

## 用户决定与范围

目标是正常光单任务能力迁移，再研究弱光。教师必须是实际 RAW 输入的已训练作者 RawVLA，内部可 Neural ISP→RGB→SigLIP；历史 Default ISP＋π0.5仅为诊断参照。用户已撤销“所有策略固定单帧”，允许从首轮采用有依据的因果历史／递归；无须先证明单帧失败。当前实现每帧 stem 组件和基线，不将它冻结为最终架构。未选定有效教师前，不虚构已验收的时序学生。

本轮授权工程、资产下载核验和零更新前反向；没有正式教师／学生训练、仿真闭环或实机动作授权。旧N01/N02/N04暂停，未接触历史远端。报告的学习率和预算是冻结诊断配方，不是运行结果。

## 实际实现

主要代码由用户本机 DeepSeek Harness `deepseek-official/deepseek-flash/max` 承担；Sol 提接口、分轮验收和具体返工，最后直接补修少量阻塞问题。复用代码位于 `src/owac/`：

- `baselines/raw_stem_siglip.py`：局部卷积／patch stem，真实 Bayer4／pseudo-RAW3／RGB对照分离；复制同策略 SigLIP blocks、位置、postnorm、原投影；继承审计、冻结与输入侧有限适配接口；空架构 artifact 导出／重载。
- `raw/distillation_pairs.py`：来源／示范拆分、真实状态动作、裁剪／CFA／相机／时间／mask、统计量、文件内容和缓存身份边界；纯学生视图不读取教师数组。
- `training/raw_distillation.py`：作者 RAWVLAOutput 适配，真实 state_dict 严格加载、state＋theta reset/carry，完整张量与因果前缀摘要；FP32有效token MSE＋cosine、原生 flow action；有效batch、累计更新／anchor上限、checkpoint／RNG／sampler／scope重载、验证选型和受新授权门控的循环。
- `evaluation/raw_distillation.py` / `raw_distillation_trace.py`：纯学生推理、完整去噪动作分组误差与分母、验证聚合、结果配对账本、视觉对照；不同 native／bridge 实现逐边界追踪与重复误差先行的容差冻结。
- `backends/raw_distillation_context.py`、`training/raw_distillation_profile.py`、`runtime/raw_distillation.py`：实际资产上下文、单独纯部署上下文、零更新画像和CLI接线。

参数配方：首轮仅stem，语言／动作后端与继承视觉层冻结；原生学生路径保留 autograd。stem AdamW lr1e-4、wd.01、clip1、warmup100，特征 MSE＋.1cos，λf1／Dλa1。可选前2层lr1e-6需诊断及匹配C/D声明，未启动。50源示范40/5/5、seed101、有效batch16、每臂阶段累计≤5000／anchor≤80000；10 horizon／10 flow／5 replan保留各自含义。

## 按原始要求—代码—证据验收

| 项目 | 结果 | 证据和限制 |
|---|---|---|
| 本机真实 Harness 主编码、实际Max选择 | 已核对 | round01—09及auth修复收据、持久request/header选择；[账本](harness-ledger.json)。CLI／Web入口实测rc2，未擅自升级或重启服务。exit0不等于完成；多轮有明确request_changes。 |
| 原成功基线完整分母 | 已重新计数 | LIBERO5820/10000、Normal1918/2000、task6Normal50/50；RoboTwin508/3250，另列原条件108/632。全部历史Default基线，非神经RAW教师。 |
| 相同策略文件恢复／可加载性 | 通过 | 下载16个同generation对象12,439,085,481bytes；同转换器生成文件SHA68710f4e…7ba6e8完全一致；原转换812/813、序列化重载813/813，无未解释缺项。[恢复](backend-recovery.json)。 |
| 真实视觉权重继承与几何 | 通过 | 当前实际文件逐继承张量哈希；224、patch14、16×16、hidden1152、27blocks、原投影2048、每相机256tokens，混合FP32/BF16。[接口](real-checkpoint-interface.json)。 |
| stem／原生动作梯度、冻结稳定 | 合成输入的真实权重接口通过 | 4个stem张量，共14,453,632参数；token损失和原生flow动作损失分别反向到全部4项；后端完整权重哈希稳定。无优化器步骤、无真实动作误差结论。 |
| 原生／桥接完整去噪数值 | 合成三相机首窗口通过 | 同一真实权重，10去噪步、50浮点边界最大误差0、原客户端后处理。非专家数据阶段零，不覆盖真实历史／首后窗口reset。[收据](real-policy-synthetic-parity.json)。 |
| 真实checkpoint0 artifact独立重载 | 修复后通过 | 首次maxerror2.0（eager未序列化、重建自动SDPA），固定原执行方式后0；混合精度保留；教师模块／NPZ缓存读拦截0触发。[重载](real-artifact-reload.json)。随机stem，0更新。 |
| 数值／mask／几何／缓存／来源／冻结／CLI针对性测试 | 68通过，0跳过 | 宿主严格拦截所有torch optimizer.step，尝试会FAIL，不转skip；原始 `sol-targeted-final-tests.log`。含无更新spy的循环／resume／导出来源、不同有效token分母、真实教师RGB轴、递归dataclass完整字节摘要。 |
| 既有桥接／语言规范回归 | 39通过、2跳过 | `sol-existing-bridge-tests.log`；历史硬编码环境的真实策略项跳过，不算通过，独立真实权重探针另列。 |
| 仓库检查 | 通过 | 宿主 `make check`：Ruff check／format、compileall；Harness沙箱uv cache只读失败原样保留。 |
| 作者真实公开RAWVLA结构、burst6／state＋theta／reset | 结构夹具通过 | 实际作者类，随机初始化；真实左补帧声明修复后，状态接续和reset输出／前缀相等。不代表已有训练好的教师。[结构收据](author-structure-fixture.json)。 |
| 已训练作者RAW前端／完整教师 | unavailable | 官方源码公开；当前公开repo／releases／作者HF入口未找到所需训练后前端权重。不以π0.5骨干、随机前端或Default ISP替代。[公开核查](teacher-release-audit.md)。 |
| 50真实task6配对源示范与新留出初态 | unavailable | 无当前本地完整配对示范清单；有50标准初态候选，不等于独立的10验证／20新测试集合。[首轮盘点](asset-inventory.json)是时间快照，已恢复后端以新收据为准。 |
| 真实专家阶段零／时序训练端到端 | not_run／未验收 | 每帧学生明确拒绝未实现历史，不默默丢帧；实际教师history、burn-in、相机独立状态及首／后重规划／reset须在有效资产上完成。不能把适配器支持burst写成完整时序策略已实现。 |
| 实际优化器更新／checkpoint学习重载 | not_run | 本轮0更新；spy、人工改写夹具权重和未训练artifact都不是学习。 |
| 真实动作精度／视觉控制干预／闭环能力 | not_run | 未执行真实零视觉／跨示范／改变正确动作的干预实验；工程算子测试不支持视觉控制结论。未新增回放或机器人动作。 |
| p50／p95、真实GPU小时／训练磁盘预算 | unavailable | RTX5090宿主可用；B1C1合成接口峰值11,104,629,248bytes，只支持设备契约。真实数据／教师画像缺失，未用微型时延承诺成本或更快。[资源](resources.json)。 |

## 独立返工与负结果

Sol没有依据Harness总结放行。主要发现并返工：teacher state_path只验hash未加载；错误假设作者输出为多帧tensor；tokenize-only原生接口不适配；桥接缺动作仍pass；全边界比较遗漏视觉／语言／速度场且NaN放行；CLI只有stub；评估随机重建stem并依赖教师上下文；真实factory内部类名和导入模块错误、混合精度重载失败；转换／保存重载计数混淆；教师当前帧／遗留时间轴处理使B相机轴错位；microbatch有效分母、CUDA RNG／scope／sampler恢复问题；checkpoint0未参与best、同名权重覆盖及原子保存问题。逐轮review和失败探针保留。

最后真实artifact maxerror2.0是工程负结果，不能靠放宽容差处理。当前改为显式eager，真实同输入重载后0，前后收据都保留。首个真实数值探针的统计量驱动类型错误同样保留，未把失败覆盖成成功。

## 停止边界与最小行动

完整能力尚未恢复，也没有足以判断路线优劣的学习证据。当前交付是可审阅基线实现、冻结配置、真实后端恢复与有边界的数值／梯度验证；不能写成整条RawVLA时序策略工程已最终验收。

下一步先取得有效作者前端checkpoint与配置；无公开权重时，需另行授权教师复现。随后补50条真实来源配对数据与拆分，完成可比历史／时序接线，运行真实专家阶段零及零更新画像，再一次性明确教师训练／学生B/C/D／闭环各自预算和授权。当前不启动新训练、不恢复旧作业、不扩任务或弱光。

实际命令、未来入口和需要的资产见 [RUNBOOK](RUNBOOK.md)。当前配置指向已核验后端，但教师、配对数据和阶段零仍为空，训练授权仍false。数据依赖须披露原RGB预训练、教师内部成像与训练时特权信息；本轮未加入额外视觉数据。


## 用户要求先取得教师后的补充验收

已完成公开权重补查、完整作者源码下载和配置实际合并；仍未取得训练后的Neural ISP checkpoint。官方全部6分支、main32个历史提交／release／tag，以及作者项目页、HF主页／数据集顶层附件未发现可用前端。搜索边界与网络失败如实保留，非不存在证明。[公开搜索v2](teacher-public-search-v2.json)、[配方清点及Sol修订](teacher-recipe-inventory.md)。

Harness round10/11仅做资料清点，Sol读取真实源码独立复核，撤回旧不完整snapshot的配置缺失判断。独立使用作者真实config_loader合并通过；公开默认RawVLA CPU实例结构计数829172参数、83state_dict张量（3316700bytes），随机初始化，未前向优化或取得教师能力。追加源码／数据下载与正式训练计划见[教师限定复现提案](../../../docs/plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md)。提案明确公共实现和论文差异、单任务Normal数据缺口、因果历史、上限与验收。仍0优化／0新闭环，没有控制恢复结论。

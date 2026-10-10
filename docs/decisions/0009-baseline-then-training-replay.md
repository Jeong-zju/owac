# 完整基准与训练演示回放

2026-10-10，用户明确交付顺序：先在支持 RAW 采集的仿真器中取得现成 π0.5 的完整基准结果，再恢复原始仿真训练演示的场景与状态，回放时同时采集 RGB 和 RAW，写入配对数据集。用户确认采用可验证的原始仿真专家演示回放，不采用真实 DROID 场景的近似重建。后续明确固定使用 Isaac Sim / Arena DROID，RAW 必须来自仿真器原生管线；MuJoCo 路径已否定。若没有现成专家数据，先寻找现成采集/生成脚本；没有直接可用的方案则报告缺口，共同决定下一步。

本决议纠正 [0008](0008-pretrained-and-native-cfa-required.md) 中把评测阶段同步 RAW 采集当作主要交付的安排。原有部署结果与失败记录全部保留；RGB 3/3、采集开启时 RGB 策略 2/3 属于单任务部署验收，完整基准和训练数据回放尚未完成。

## 第一阶段完整 RGB 基准

使用与 benchmark 匹配的公开 π0.5 检查点，不追加 backbone 训练。先冻结任务集合、测试配置、所有 episode、种子、时限、策略输入与动作变换、官方成功判据和基础设施重试规则，再运行完整评测。报告逐任务和逐 episode 结果、总体汇总、失败、区间及实际推理成本。不能把少量成功任务或官方演示页面称为完整 benchmark。

这一阶段策略按官方 RGB 接口评测，RAW 采集能力在相同仿真器上单独验证；采集时的额外渲染不介入基准控制循环。原始测试 episode 不作为第二阶段的专家训练演示，也不为补充训练数据重新生成专家标签。

## 第二阶段原始训练演示回放

选择与现成策略任务适配相对应的公开仿真专家数据，记录训练数据版本和实际权重之间的可追溯关系；基础模型全部预训练语料并不等于该 benchmark 的专家演示。不能根据推理配置推断某个检查点的完整实际训练历史。

回放恢复源演示中的场景、机器人、物体、相机、照明和时间记录。优先逐帧恢复保存的仿真状态；只有动作序列时，必须有可核验的初始状态、场景和参考轨迹，并测量物理回放偏差。不能只播放关节位置却把未恢复的物体和接触轨迹称为原训练场景。

每个样本保留源 episode 与帧索引、源文件哈希、状态、原专家动作和语言指令，以及回放生成的同步 RGB、ISP 前无损 CFA、传感器配置、时间戳与处理历史。原始训练 RGB 与回放 RGB 分开命名；不能将新渲染 RGB 冒充原图。两种新图像必须来自同一恢复状态和相机，禁止从 RGB8 逆处理或重新马赛克生成 RAW。合成 RAW 的标定范围与位深精度须披露。

逐 episode 核验源数据覆盖率、帧数、动作与状态对齐、恢复误差、相机矩阵和图像方向、RGB 与 RAW 同步、原生帧刷新、量化及校验和。按源 episode 或场景划分训练与验证，不能随机拆分相邻帧。数据存云端，Git 和本地页面保留小型清单、数字和结论，不下载检查点或大数组。

## 当前 Arena 选择与专家来源的缺口

[Arena OpenPI 官方说明](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/running_a_real_policy/openpi.html)提供 DROID 策略在仿真抓放环境中的评测入口。第一阶段采用该入口全部九组对象/容器/HDR 配置，每组固定环境种子 0–99，共 900 个独立完整 episode，复用专用官方 jointpos 权重，额外训练 0 步。该扩展是 OWAC 固定比较协议，上游示例九组各三条与本次区分；不报告整个 Arena 平台总分。协议、基础设施重试和实际进度见[完整基准记录](../../research/experiments/2026-10-10_pi05_arena_droid_benchmark.md)。云端已由 supervisor 启动，定时任务不重建。

固定 OpenPI 提交中的 [PolaRiS 配置](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/training/misc/polaris_config.py)引用真实 `droid` 1.0.1 和仿真 `polaris_droid_cotrain_dataset` 1.0.0，配置的数据混合比例为 90% / 10%。PolaRiS 确实有公开仿真共同训练数据，不能将其全部说成真实 DROID；配置本身也不能证明下载检查点的完整实际训练历史。[DROID 数据说明](https://raw.githubusercontent.com/droid-dataset/droid/main/docs/the-droid-dataset.md)描述真实机器人记录；[公开 PolaRiS RLDS schema](https://huggingface.co/datasets/owhan/PolaRiS-datasets/blob/dca86b8e186df63323eb375344a1977d92f508f7/polaris_droid_cotrain_dataset/1.0.0/features.json)有 RGB、关节/夹爪状态、动作与路径标识，但没有逐帧完整物体状态。已公布 USD 场景和评测初始条件不能替代缺失的专家物体轨迹。

官方 Isaac Lab 有 `record_demos.py`、Mimic `generate_dataset.py` 和 `replay_demos.py`。其状态记录器可保存机器人及已注册物体状态，回放有 `--validate_states`。但本次固定版本的 DROID `mimic_env=None`，默认 SE3 遥操作与八维 jointpos 动作未接通；尚未找到当前九组任务开箱即用的自动专家生成器。GR1/G1 专家数据与 Franka 状态机示例不能直接替代。详见[脚本与状态记录审计](../../research/reviews/2026-10-10_arena_droid_expert_replay.md)。这些通用脚本在当前 DROID 上尚未做 GPU 采集验收。

当前第二阶段仍缺专家来源及 DROID 适配。新采集或生成的数据必须明确标成新数据，不能冒充官方检查点的原始训练轨迹，也不能把第一阶段策略测试轨迹换名当专家。未启动专家生成或追加训练。已取得的 3/3 与 2/3 保留为历史部署证据，3,258 个视角帧不属于原始训练数据回放；第一阶段 900 条基准与第二阶段来源缺口分别报告。

来源与文件哈希见[当前专家脚本清单](../../research/reviews/2026-10-10_arena_droid_expert_replay.json)及[早期来源核查](../../research/reviews/2026-10-10_training_replay_sources.json)。定时任务保持关闭，现有权重与原始记录保留。

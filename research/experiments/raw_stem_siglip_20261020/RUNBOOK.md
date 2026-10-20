# RAW stem 蒸馏：当前命令与启动门槛

2026-10-20，工作目录 `/home/jeong/zeno/owac/repo`。当前授权仅工程、资产核验、零优化更新前反向。**实际更新0，新增闭环0。** 不恢复旧作业。所有以下 CLI 已有真实处理入口；真实教师和示范缺失时退出2是预期阻断，不能当作已训练。主配置仍 `authorization.optimizer_updates=false`。

## 已恢复资产与环境

- 同一历史 π0.5 checkpoint：`outputs/checkpoints/raw-distillation/pi05_libero_starvla/bfloat16/model.safetensors`，SHA256 `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8`，与历史成功基线逐字节相同。转换812/813、序列化文件重载813/813是两个不同审计阶段。
- tokenizer：`outputs/checkpoints/raw-distillation/paligemma_tokenizer.model`，SHA256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。
- 原统计量：`outputs/checkpoints/raw-distillation/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json`；原生 quantile，不把 RoboTwin 的均值方差规范搬入 LIBERO。
- 原 StarVLA source：`outputs/raw-distillation-20261020-sol/starvla-pinned-source/`，原提交 `57d508cb180d0510ac9060358009c0f6ffaac420`，关键文件与历史归档哈希一致。
- 可运行的审计解释器：`outputs/raw-distillation-20261020-sol/audit-runtime/bin/python`，Python3.11，Transformers4.57.0、tokenizers0.22.2，复用仓库已有 CUDA torch2.7.1+cu128。单独环境覆盖不修改原 runtime；当前主 `.venv` 为 CPU 测试环境。依赖清单见本目录的环境收据。大资产均在忽略目录。

以上只有冻结后端，**不包含训练好的 RawVLA Neural ISP**。另一项目 OpenPI 权重文件只作候选对照，未替代该同源文件。

## 当前可执行的核查

```bash
cd /home/jeong/zeno/owac/repo
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation --help
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation assets
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation preflight
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation train
```

当前 `assets/preflight` 应为 unavailable，`train` 为 blocked；分别缺少学习型 RAW 教师、真实配对 manifest／阶段零收据和训练授权。不要把返回0的夹具命令或 Harness final 当成真实策略验收。

## 实际导出与零更新证据

`outputs/raw-distillation-20261020-sol/untrained-real-student-checkpoint0-v2/` 是实际同源 SigLIP＋随机 stem 的 **0更新** artifact；它不是训练结果。真实权重导出、默认空架构重载、混合精度、eager 注意力及同输入输出完全一致已核查；不需要教师／ISP／NPZ 缓存。旧 `...checkpoint0/` 保留一次错误重载的失败证据，不能部署为通过版本。

实际数值／梯度脚本及原始 JSON 均在 `outputs/raw-distillation-20261020-sol/`：

- `sol-real-checkpoint-interface-probe.py` / `...-result.json`：原 checkpoint 完整加载、继承逐张量核对、特征及原生动作梯度、冻结后端哈希。
- `sol-real-native-parity-probe-v2.py` / `...-result-v2.json`：真实原生实现与桥接10去噪步、客户端后处理，合成三相机输入，50个浮点边界最大误差0。首版驱动统计量类型错误保留，v2明确修正，不混淆。
- `sol-real-artifact-probe-v2.py` / `...-result-v2.json`：实际 artifact 重载；旧版误差2.0，补齐 eager 执行方式后为0。教师导入／缓存读拦截器0触发。

这些输入是合成接口探针；**不是正常光专家数据阶段零通过、教师控制成功、学习或恢复控制**。不要直接重复运行同一脚本覆盖已有证据文件；新复核创建新路径／run-id。

## 真实资产齐备后的入口（本轮未执行）

首先把新运行配置冻结到独立文件，提供真实路径及哈希：已训练作者前端 checkpoint/构造配置、因果历史、50源示范40/5/5的配对 manifest、相机物理对应／裁剪、统计量、受验证的教师／专家来源收据。`prepare-pairs` 不替代示范采集或数据清单审核。Bayer 与三通道 pseudo-RAW 分开；当前 LIBERO 是 RGB8 逆处理。

```bash
# $CONFIG / $RAW_BURST / $PAIR_OUT / $AUTHORITY / $RUN_OUT 是需提供的真实资产，非已有路径。
# 固定形状、时刻、相机及真实历史后，教师从 RAW 实际生成 RGB 和 tokens：
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation --config "$CONFIG" prepare-pairs --raw "$RAW_BURST" --output "$PAIR_OUT"
# 缓存专家观测的阶段零核验，独立冻结 native-repeat 容差再比较 bridge：
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation --config "$CONFIG" parity
# 真实数据零更新 F+B／推理画像；冻结 GPU小时及磁盘上限后才能申请正式运行：
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation --config "$CONFIG" preflight
# 只有取得具体新授权、更新配置授权位和通过所有前置审计后才执行：
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation --config "$CONFIG" train --arm C --authority "$AUTHORITY" --checkpoint-dir "$RUN_OUT" --max-updates 5000
# 必须导出实际训练 checkpoint，禁止给新随机 stem 换一个步数标签：
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation --config "$CONFIG" export-student --training-checkpoint "$RUN_OUT/checkpoint_best.pt" --arm C --output "$RUN_OUT/student"
# 独立纯学生路径读取 artifact，教师文件可不可用，不能有回退：
outputs/raw-distillation-20261020-sol/audit-runtime/bin/python -m owac.runtime.raw_distillation --config "$CONFIG" evaluate --student-artifact "$RUN_OUT/student"
```

示例 C 不自动授权，B 首先检查 stem 替换流程；D 还须有阶段一诊断与匹配 C/D 声明。每臂累计最多5000、有效 batch16、seed101，所有 warm-start 更新都累计。正常光成功后才提出弱光扩展。真实专家／checkpoint／来源审计由 Sol 独立签认；当前合成 parity 收据不能打开正式训练门。

## 已知界限与下一步

作者模型与 RAW 状态适配器支持因果 burst／GRU／theta；当前学生、数据迭代器与 artifact 实现为 **每帧基线**，时序学生及教师历史的端到端训练接线尚未验收。它们不是最终策略必须单帧的研究决定。拿到有效作者 checkpoint 后，先确认其历史与真实能力，再选／补齐可比时序模型，不能删掉教师历史后冒称同一教师。当前历史模式不受支持时明确阻断。

仍缺：训练好的 RawVLA 前端及相匹配配置；50条真实 task6 配对专家示范和来源拆分；新留出初态的独立性；真实专家阶段零、教师闭环、真实训练画像／推理p50/p95。现有50个标准初态只是候选池。不得用历史episode2、Default成功率、其他任务数据、合成输入补这些缺口。

最小行动是取得作者前端 checkpoint；公开渠道未找到时，另行制定并授权教师复现。新教师训练、学生B/C/D训练、验证／测试闭环是不同范围。目前不能给出可靠 GPU小时预算；真实输入前反向画像仍是授权范围内的下一项预检，不能由合成微型时延估算替代。


## 教师取得的本次后续

补查全部公开分支／main历史后仍未找到已训练前端；完整源码与实际配置合并已核验。已形成[教师限定复现提案](../../../docs/plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md)及 `configs/experiments/rawvla_teacher_reproduction_proposal_v1.json`。后者是授权／研究范围清单，**不是现成trainer配置或启动命令**，所有执行授权false。提议只训作者前端累计≤2000、保留时序；预算封顶不替代真实画像。当前不运行官方trainer示例，也不恢复旧训练。

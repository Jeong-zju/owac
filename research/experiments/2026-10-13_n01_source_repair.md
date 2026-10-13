# 2026-10-13 N01 七组源码问题修复与同条件验证

本轮按用户要求修复此前 review 的七组问题（实验矩阵一组含三项），保留 N01 的支持域积分、迟滞合并/分裂和选择代价定义。工程修复验收通过，定位有所改善；自适应调度的研究验证仍未通过，不能据此宣称机器人任务收益。

## 问题与修复

| 问题 | 修复 | 验证结果 |
| --- | --- | --- |
| 原始精度质量进入 sigmoid，淹没均值、方差、年龄等通道 | 仅在 learned gate 输入处，用训练尺度与 `x/(s+abs(x))` 编码六个显式统计量；原始积分、统计输出和选择成本保持原单位。一维 adapter 使用同一函数，尺度只在显式训练校准时拟合 | 同数据 40 步检查点，门控饱和比例 99.9824% → 0%；保留精度质量、擦除其他统计通道，feature 最大差异 0 → 1.88437，其他统计重新参与计算 |
| 轨迹使用未拟合的单位尺度 | 重用训练校准函数，独立 low/slow seed=500000；已有训练尺度原样保留，测试轨迹不参与拟合；JSON 记录来源、尺度与计数 | 相同权重、相同测量，静止轨迹历史 1.00366 → 4.24438 帧；只校准，没有梯度训练 |
| 运动门读取当前帧 identity | 使用最近历史 slot 1 的真实 residual 与 valid，缺少历史报告 unresolved | 历史残差从 0 变 100 时 gate 1 → 6.305e-16；无效历史维持中性 gate，逐 batch 缺失统计通过 |
| 事件没执行、SNR 顺序不对、容差相同 | 让 augment 实际控制出现/遮挡、连续反转与两个区域的交叉；按像素面积积分线性信号，使亚像素运动改变观测；交换低/高 photon scale，使更大 photon scale 对应更大散粒噪声；wide/narrow 默认 2.5/0.75 grid px，显式覆盖另记来源 | 四种事件测量均不再等于普通 slow；名义信噪比由 4.77/6.80/3.16 改为 1.51/6.80/30.77，采样噪声实验顺序也正确；容差实际生效 |
| RoboTwin 服务漏传 suite | main 显式传 suite；检查 checkpoint 平台与策略/动作协议维度一致；元数据记录 suite | CPU 验证实际 main 的两个平台参数；真实服务验收见下文，RoboTwin control_dim=14、replan_steps=50 |
| CFA 稀疏铺在完整网格，没有同平面一步邻居 | 无损重排成四个 H/2×W/2 平面，保留 phase offsets、质量掩码与全部原始采样；仅监督字段缩到 cell grid，不插值测量 | 8×8 传感器仍有 64 个有效采样；空间非零边 0 → 79；反向重排逐值相等，四相位/坏点/饱和边界测试通过 |
| 把 batch 顺序当作稳定时间 | endpoint 比较的 first_stable/recovery 记为未测；完整机制矩阵逐轨迹计算时间，再报告条件平均与稳定比例 | endpoint 不随 batch 排列伪造时间；两条轨迹分别 0/3 帧，平均 1.5，交换 batch 后不变 |

门控编码的语义发生变化，旧权重不能默默载入新门控。新增 `evidence_encoding_version` 状态标记，严格加载旧检查点会报缺失，要求重新训练；新检查点保存/加载、合成恢复训练测试均通过。没有增加可训练参数。

## 核心机制审计

[core-preservation-audit.json](results/2026-10-13_n01_source_repair/core-preservation-audit.json) 对修复前后做了 15 项源码/AST 比较：`domain_statistics`、`update_edges`、逐边系数、`EvidenceBlock`、`SupportSelector`、`_select`、`_statistics_scale` 相同；config、state、correspondence、真实 loader、特征桥及全部基线文件哈希相同。积分权重、逐源平方系数方差、唯一帧/唯一采样计数、分裂后重算、图交换和成本定义均未更改。运动门修复仅纠正它读取的历史对象。

全仓 **257 passed**；`make check` 的 ruff、format、compileall 全通过。新增 17 项行为回归，覆盖上述问题、批次排列、校准不更新参数/不读测试样本、旧检查点拒绝误解释，以及两平台真实 main 参数转发。原逐源数值、插值共享源、严格 holdout、唯一缓冲、reset、一维无 Conv2d、冻结 C4 测试保留并通过。

## 修复前后效果

为避免把数据生成器修正误报成算法收益，先用修正后的生成器生成一份固定张量档案，修复前源码 `d4bf84c5a73a4fa97bcb9319f2aa845836c38994` 与修复后源码分别运行独立进程，读取**同一档案**。所有九个方法共享训练/验证测量、种子、初始化、probe head、优化器、历史信息与预算：40 步、B=4、T=12、32×32、seed=1234、head_seed=101237、AdamW lr=3e-4、weight_decay=1e-4、clip=1、最后一帧 Huber。校准均来自第一份训练 batch；解析 selector、alignment=none，未调参。22 条条件使用真正事件及各自容差。

| 方法 | 修复前误差 px | 修复后误差 px |
| --- | ---: | ---: |
| N01 | 3.0406 | 2.5895 |
| 去合并/分裂 | 3.0922 | 2.6207 |
| 固定历史 | 3.0401 | 2.5864 |
| 单帧 CNN | 4.5870 | 4.5870 |
| 固定平均 | 4.7106 | 4.7106 |
| EMA | 4.7158 | 4.7158 |
| Kalman | 4.7596 | 4.7596 |
| 运动补偿 | 4.7010 | 4.7010 |
| ConvGRU | 4.7984 | 4.7984 |

N01 改善 **14.83%**，超容差率 0.6932 → 0.6364。六个未改动基线每一条件的误差逐位一致；这为同条件对比提供了控制检查。N01 对去重组消融只领先 0.0312 px，仍比固定历史消融差 0.0032 px；这里只有一个种子、短预算，没有统计显著性或充分调优结论。改善主要说明门控输入不再被精度质量淹没，尚不能归因于自适应调度的独有价值。

三条匹配测量、seed=0 初始化、无训练轨迹：静止 1.004 → 4.244 帧，快速边缘 1.005 → 5.166 帧，交叉区域 1.005 → 3.601 帧。修复前后都读取固定轨迹张量；修复后另读独立训练校准张量。**快速仍比静止保留更长历史**，低 SNR 下的可靠运动缩短历史假设未获支持。这些未训练轨迹不作为训练模型能力。

额外完成一轮在线随机 200 步 smoke：loss_location 2.7845 → 0.9125，末步训练误差 1.2343 px；22 条独立完整轨迹平均 2.7654 px、超容差 0.5999，有非零 merge/split。200 步只检查训练可执行性；生成器已版本化为 `linear_pixel_footprint_v2`，不能拿本轮 200 步/机制矩阵与旧生成器的历史数字直接宣称改善。速度强度仍保持原值：32×32/T12 的 fast 只有 0.141 sensor px/frame、总位移 1.55 px；尚不能称为强运动压力测试。SNR 条件同时改变名义 contrast，此处没有将其误说成纯噪声因素分离。

CPU 单次 smoke 200 步 p50/p95=743/1032 ms，RSS 峰值约 2.77 GB，GPU 未测。这是本次实际资源记录，不作优化速度收益结论；本轮运行期间有短时 CPU 核查并行。40 步配对实验按先前、后修顺序单独运行；没有充分重复以评估硬件时延波动。

## 真实后端与服务

独立快照位于服务器 `/workspace/owac/rawvla/research/n01-source-repair-20261013/`。复用固定 Python 3.10 / torch 2.7.1+cu128、现成两平台 NPZ 与 π0.5 权重，不修改已有 C0 冻结目录。

LIBERO/RoboTwin 各解析无辅助、学习有辅助两组，每组 3 个实际 batch，共 12 个 batch。动作损失和推理输出有限，stem/readout 梯度与实际参数更新非零，学习版 gate/selector 梯度非零，冻结后端梯度为零；重复相同 noise 的 flow 推理一致，padding mask 与无视觉塔路径维持。作者数据为三通道 pseudo-RAW，32×32 明确裁剪、方差为声明的假定值；不是原生 Bayer，也不是任务闭环收益。

服务验收的首个脚本在 RoboTwin 读取数据时漏传 `dataset_format`，误用 LIBERO quantile 归一化并报错。四组训练及 LIBERO 请求已完成，未重训；修正独立验收脚本后只补跑 RoboTwin 服务请求，失败日志留在服务器 `validation-first-attempt.log`。最终服务结果见数字收据。

两平台各 3 次实际 WebSocket 请求：新帧、重复帧、新 episode；元数据、有限动作、unique_frames=1 与状态复用/重置核验。RoboTwin horizon/control/replan=50/14/50，LIBERO=10/7/5。动作块保持内部 32 维，客户端执行维度单独声明；未执行物理控制。服务已退出，测试 job 不自动启动/重启。

操作前后冻结 `rawvla_full_eval.py` 与 `backends/rawvla.py` 哈希必须与收据相同；原全量/审计 supervisor 进程保持。源码清单逐文件验证，运行时 HEAD 仍为修复前提交，完整修复由 source hash 与 Git diff 记录，不能仅凭 HEAD 判断版本。

## 复现与证据

小型收据、完整各条件数字及运行脚本文本位于 [results/2026-10-13_n01_source_repair/](results/2026-10-13_n01_source_repair/)。张量档案、检查点、PNG 和日志位于 ignored `outputs/n01-source-repair-20261013/`；服务器权重与真实数据未下载。固定数据 SHA 与全部条件见 `shared-data-manifest.json`；配对源码、生成和轨迹脚本按原文件保存为 `.py.txt`，需在本工作树 outputs 中恢复原文件名运行。修复前源码通过 `git archive d4bf84c` 还原到独立目录，以独立进程的 PYTHONPATH 选择版本。

```bash
make check
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python outputs/n01-source-repair-20261013/generate_shared.py
PYTHONPATH=outputs/n01-source-repair-20261013/before/src .venv/bin/python outputs/n01-source-repair-20261013/paired_validation.py before
PYTHONPATH=src .venv/bin/python outputs/n01-source-repair-20261013/paired_validation.py after
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python -m owac.training.n01 --config configs/encoders/n01/tiny.json --synthetic --steps 200 --log-every 20 --output outputs/n01-source-repair-20261013/smoke200
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python -m owac.evaluation.n01 --config configs/encoders/n01/tiny.json --checkpoint outputs/n01-source-repair-20261013/smoke200/checkpoint.pt --suite mechanism --no-save-checkpoint --output outputs/n01-source-repair-20261013/mechanism
```

下一项研究应固定更充分的运动强度与多种训练种子，在验证集上评估候选/固定历史/去重组的可靠差异，再决定是否做独立闭环。当前代码修复与后端接线验收不能替代这一研究证据。

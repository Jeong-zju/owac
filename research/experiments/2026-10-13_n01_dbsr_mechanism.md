# 2026-10-13 · N01 · DBSR 收敛基线与空间撤回独立实验

## 研究问题与运行前设计

本轮先训练作者规模 DBSR 特征基线及同容量固定平均、单帧对照，再用同一组冻结权重检验 N01 的空间支持关系与撤回重算。没有新增可训练网络。完整 N01 的时间历史选择不在这次空间规则干预中；空间邻接断开不会删除同坐标的旧来源。

实验分两层：来源隔离测试检查“撤回是否真正撤除了影响”；新的序列留出实验检查“这个操作是否改善任务表征”。冻结权重可以直接隔离算子的效果，但改变空间计算后会改变特征分布，因此负结果不能代替图约束模型的匹配重训，正结果也不能代替真实 RAW 和下游任务验证。

- 身份：`n01-dbsr-convergence-v1` / `n01-dbsr-spatial-support-frozen-probe-v1`；负责人：lzh 指定的本次实现与验收；日期时区：2026-10-13，Asia/Shanghai。
- 基线源码提交 `861a8cf`；任务队列、配置排序和支持适配随本记录提交。支持层有效性审计在机制 holdout 前修复了非有限 RAW 的缺失处理，并重新锁定源码（保留首次锁定收据）；配置与参数没有改变。原始 N01 核心源码保持；运行快照均包含执行源码哈希，服务器使用独立不可混用的 baseline/support source 目录。
- 当前状态：基线训练中；机制 holdout 尚未消费。结果完成后在本记录追加实际验收，不能从计划推定完成。
- 工作树：`lzh/n01-evidence-integration`；保留用户未提交研究方案修改。与任务相关的配置、源码和测试由本次独立提交记录；大产物留服务器。
- 实现/参数规模：上游固定提交 `16f158e0553b6702cc1693c43f733c2830973278`，64 / 9 / 512 的全尺寸 RAW 特征层和原融合层。模型参数为 current 1,119,393、mean_capacity 1,636,385、dbsr_reference 1,653,953，包含共同原定位头。没有 RGB/PWC 执行、RGB 重建或上游预训练权重。
- 环境：服务器 `107.209.104.125:44327`，RTX PRO 6000 Blackwell 96 GB；专用 venv，Torch 2.9.1+cu128、CUDA 12.8、NumPy 2.3.4。原 frozen RawVLA supervisor、源码和 runtime 不修改。3 个种子进程，每进程显存 allocator 上限为 GPU 总量的 12%；预算续训时另有独立进程，4 个训练进程合计上限为 48%；FP32、关闭 TF32，确定性算法开启。

## 基线训练与验收规则

协议：[burst_features_convergence.json](../../configs/encoders/n01/burst_features_convergence.json)。种子 61/73/89，每个先训练 DBSR，再训练同容量固定平均和单帧。每步 2 条、24 帧、16×16 packed CFA；最多 8 个唯一来源。四条件和噪声生成器沿用上一轮，包括静态弱信号、准确对齐运动、遮挡解除、局部错误历史。oracle 使用已到达的整数位移，排除主指标和选型。数据以序列拆分，train/validation/test 的 seed 区间独立；没有真实相机或物体留出实验。

至少 2,048 步、最多 8,192 步；每 256 步验证，每条件 64 条验证序列。AdamW 3e-4、weight_decay=1e-4、clip=1.0；ReduceLROnPlateau factor=0.3、patience=2，最低 lr=1e-5。降到最低学习率后，连续 6 个验证检查相对此前最佳的改善≤max(0.01 px,1%)，才称“达到声明的验证平台”。它不是全局最优或任意数据上的充分训练保证。若触及预算还在改善，则记 budget_limit，并保留正式 test，禁止进入机制 holdout。实际有容量控制触及 8,192 步且仍改善，采用子运行将最大步数扩展至 16,384；原记录保留，只有步数上限改变，优化器、scheduler、RNG 和样本流连续恢复，不重设学习率或选择器。未消费 test 的预算不足运行才允许续训。

共同验证规则选 best；只有达到平台才执行预定正式 test。各方法充分训练所需步骤可能不同，因此这里比较同一训练配方下的稳定结果，而不是相同步数的性能。固定预算的上一轮证据另外保留。

批量因果窗口与原逐帧版本的输出、损失和梯度已核验；没有时间归一化、未来输入或跨 optimizer step 的旧特征缓存。整数对齐使用确定性 gather；分数 `grid_sample` 保留原路径，本轮 GPU 训练没有分数位移。CUDA 该操作的限制见 [PyTorch 官方确定性文档](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)。last 检查点含模型、优化器、scheduler、数据步骤、CPU/CUDA RNG 与曲线；中断恢复和不中断训练的数值已核对一致。

## 空间支持与重算干预

适配器 [n01_dbsr.py](../../src/owac/frameworks/n01_dbsr.py)调用原 N01 的测量缓冲、解析邻接 gate、迟滞 update 和局部支持路径函数。图只从已到达 RAW、声明方差与有效性建立，不读取任务目标、可见性标签或事件时刻。每条序列重置，各成员独立；源身份仍唯一。

原 DBSR 的权重、偏置、残差结构和激活全部复用。图约束作用于每层卷积的输入采样路径，以及融合权重预测器；全开放时输出和梯度与原层一致。潜在特征通道已经混合四个 CFA 平面，因此共享路径采用四平面交集；这是保守的图到特征映射，可能损失可用邻域，不能冒称与逐平面积分完全等价。这里只检验硬支持拓扑，没有把原软边强度重复乘到每个卷积层。多层传播仍沿 DBSR 的原感受野；没有复现原 N01 积分域的固定空间半径或多时间尺度选择，因此不能把本适配器称为完整 N01 编码器。

三个空间干预与同权重无图对照比较：

| 干预 | 支持图 | 历史特征 |
| --- | --- | --- |
| none | 不约束 DBSR | 正常由来源编码 |
| static | burn-in 第 8 帧之后冻结 | 使用固定图从 RAW 重算 |
| dynamic_stale | 原迟滞开闭继续更新 | 保留来源到达时编码的旧特征 |
| dynamic_recompute | 原迟滞开闭继续更新 | 每次按最新图从原始 RAW 重算 |

static 对 dynamic_recompute 检查动态支持的差别；dynamic_stale 对 dynamic_recompute 检查旧依赖的重算；none 对所有图模式检查约束带来的总收益和代价。重新开放也会引起重算，因此自然序列的两个动态模式之差不能全部归因为“关闭”的效果；单独 split 夹具检验真正的来源撤回。

准确运动条件中，所有干预臂统一先在 RAW 上做 oracle 对齐，再编码，保证图和被处理来源位于同一参考坐标系。这个正例与上一阶段“编码后对齐”的 oracle 顺序不同，只在干预臂内部比较，仍排除主均值。普通条件未改变对齐顺序；none 与原 DBSR 相同。

## 独立噪声诊断与标定

开发阶段的真实 GPU 接口样例显示，原解析图在低信号条件下几乎全关闭。为排除任务网络影响，另取 seed=5501301，4 条×24 帧×4 CFA×16×16 的均值 0.3、σ=0.08 独立 Gaussian 测量，不含目标或事件。原配置在 burn-in 后逐平面开放比例 0.000984，四平面共同开放比例为 0。

这不是训练权重未充分学习造成的：此图只调用无参数解析规则。原 `edge_temperature=0.1` 会把常见噪声差当成强不相容，迟滞机制因而很少建立连接。不能先把成熟卷积退化为几乎逐点计算，再将性能下降归结为所有支持机制无价值。

保留 stock 不变，另预先锁定 noise_calibrated 对照。只调原 gate 的尺度参数：`edge_beta=8`；以标准 Gaussian 两侧 99% 分位 q≈2.5758、radius=2 的距离系数 exp(-1/8)，求解原公式在 q 处等于 split_threshold=0.35，得到 `edge_temperature≈3.97404`。这个规则只约束平坦、独立 Gaussian 噪声下的**单次成对反证率约 1%**，不保证图级误撤回率、真实边界识别或非 Gaussian 噪声适用性。merge/split threshold、迟滞、测量身份和重算原则未修改，没有新增网络。

同一独立平坦噪声样本上，标定配置逐平面开放比例 0.913666，共同开放比例 0.846283；另一个独立数值测试核对成对反证率在 0.7%–1.3%。这仅说明标定避免噪声把图全部关闭，不能证明任务收益。stock 和 noise_calibrated 的三种图模式都保留，none 共用；共 7 臂×3 种子。

机制协议：[burst_support_probe.json](../../configs/encoders/n01/burst_support_probe.json)，源码/配置和时间锁定见 [protocol-lock.json](results/2026-10-13_n01_dbsr_mechanism/protocol-lock.json)，独立标定收据见 [flat-noise-calibration.json](results/2026-10-13_n01_dbsr_mechanism/flat-noise-calibration.json)。机制 holdout 数据种子取训练种子+1000，仍用独立 test seed 区间；每条件每种子 24 条、24 帧，共 288 条配对序列，每臂使用相同测量和监督。所有参数冻结，模型哈希运行前后核对，避免通过重训练同时改变融合器。

## 验收与执行产物

新增来源隔离测试验证：图全开放时保持全尺寸 DBSR 数值与梯度；关闭竖直边界后，另一侧 RAW 的变化和梯度不会进入该侧重算特征；到达时已编码的缓存仍可能含跨边界内容；权重预测器同样不能越过关闭路径。原 N01 迟滞在持续相容后打开、反证到达后关闭，未来和其他 batch 成员不影响此前图。CPU 与真实 GPU 前向、反向及来源隔离均已执行。

服务器目录：`/workspace/owac/research/n01-dbsr-20261013/`。baseline-source / support-source 隔离；env 独立；baseline-runs 保存全部曲线与 best/last，logs 保存三个 seed 的完整输出。Supervisor 进程名 `owac-n01-dbsr-baseline-61/73/89`，只控制本轮进程。GPU 探针中 DBSR 步骤预热后约 54–90 ms，验证四条件约 2.51 秒，峰值 allocated 9.797 GB；这是小规模探针，不是部署延迟，受原任务同机竞争影响。正式训练与机制代价待实际结果留档。显存之外的功耗、通信、部署流式优化和实机成本未测量。

复现入口：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python -m owac.runtime.n01_burst_job \
  --config configs/encoders/n01/burst_features_convergence.json --seed 61 \
  --root outputs/n01-dbsr-converged
# 另运行种子 73、89，确认全部达到声明平台后：
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python -m owac.evaluation.n01_burst_support \
  --config configs/encoders/n01/burst_support_probe.json \
  --baselines outputs/n01-dbsr-converged --output outputs/n01-dbsr-support-probe
```

正式数字、逐帧错误与恢复、资源、源校验及结论在任务完成后追加；在此之前保持运行中。实际候选图模型的匹配重训、真实传感器、跨任务表征及 VLA 闭环均尚未验收。

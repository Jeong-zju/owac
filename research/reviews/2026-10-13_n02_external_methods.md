# N02 可复用方法与代码选型

当前最值得引入的是**神经似然比估计 NRE 及其保守性改进 BNRE**：先直接学习两种解释
相对于测量的证据，再检验 N02 的局部撤回规则。监督冲突可以使用作者发布的 CAGrad
作为训练对照；完整多假设递推则应补充基于已有滤波实现的外部基线。继续只调当前重建
损失或增加训练步数，优先级低于这些方法对照。

可以直接复用的是损失、训练器、滤波与物理算子接口；这次没有找到经过 N02 数据和
行为协议验证、可以直接加载就提高性能的现成权重。优先级是适配成本和本地瓶颈匹配度
的判断，不是这些方法在 N02 上的性能排名。

这个选择来自[上一轮受控结果](../experiments/2026-10-13_n02_joint_controls.md)：四配方
在开发和新留出均为 0/3 种子合格，全部 12 个末检查点的线索互换配对为 0/16。冻结头部
的差异方向可以部分学到，准确共同背景能恢复配对；局部几何投影降低几何误差，却没有
建立测量判别。以下适配判断是针对这些证据的研究建议，不是外部方法已经提高 N02
性能的结论。

## 方法与当前问题的对应

| 当前问题 | 可借用的方法 | 可复用部分 | 采用顺序 |
| --- | --- | --- | --- |
| 共同背景误差压过微弱的解释差异 | NRE，ICML 2020；BNRE，NeurIPS 2022 | 条件分类器、似然比训练损失、后验校准工具 | 首选，先独立验证测量判别 |
| 多项监督在共享参数上尺度和方向不一致 | CAGrad，NeurIPS 2021；FAMO，NeurIPS 2023 | 作者的梯度组合或动态权重模块 | CAGrad 作正交对照；FAMO 需先处理损失定义 |
| 学习方差使均值拟合难点被低权重处理 | β-NLL，ICLR 2022 | 小型损失函数 | 低成本备选，当前尚未证明这就是主因 |
| 多假设、测量加权和时间递推都由自己实现 | DPF，RSS 2018；torchfilter | 学习测量得分、log-space 权重更新、重采样基线 | 补最近方法基线；不直接整体替换 |
| 成像测量与候选解释缺少可靠的正向关系 | DeepInverse | Physics、噪声、data fidelity、展开优化器 | 物理模型路线的工具箱，需要自定义 RAW 算子 |
| 何时应放弃旧状态，如何区分异常点与真实变化 | Robust BOCPD，ICML 2023 | 鲁棒预测分布、run-length 递推 | 测量证据可靠后，再作为变化检测对照 |

这些方法多数并非 RAW 专用。它们处理的是测量、候选解释和不确定性的关系，可以让
RAW 直接作为观测输入；采用它们不要求在主链路插入 RGB 重建。已有成像模型的 RGB
权重能否迁移到当前 CFA 或作者三通道 pseudo-RAW，需要另行验证。

## 首选直接学习测量证据

### NRE 为什么比继续加重差异损失更值得先试

NRE 用分类任务区分来自联合分布的参数与观测配对，以及来自边缘分布乘积的配对，
学习 `r(θ,y) ≈ p(y|θ)/p(y)`。同一观测下比较两个候选时，分母消去，得到两者的
似然比。它不需要先完整估计高维观测的概率密度。原论文和目前的实现都支持以神经
嵌入处理结构化观测。[Hermans 等，ICML 2020](https://proceedings.mlr.press/v119/hermans20a.html)，
[sbi 方法选择指南](https://sbi.readthedocs.io/en/stable/how_to_guide/06_choosing_inference_method.html)。

N02 现在的两个 Gaussian 能量相减，本来就对应一个似然比。区别在训练途径：当前先
学习两份测量预测，再从它们的重建误差取得相对证据；NRE 直接训练这个相对量。因此，
它有望绕开必须先极准确地拟合共同背景的负担。**这是一项基于本地诊断的适配推断，
不是论文已经证明了当前数据上的收益。** 直接分类同样可能利用背景捷径，数据构造仍重要。

对有限 K 个候选，可以逐个计算 `s_k = log r(θ_k,y)`，再用
`w_k ∝ prior_k * exp(s_k)` 更新解释权重。此处无需引入 MCMC 采样。两候选的分数差
是 log Bayes factor；95% 错误先验下，翻转 MAP 至少需要超过 `log(19)` 的反向证据，
达到真类质量 0.7 则需要超过 `log(19×7/3)`。只把两种线索的排序选对，还不是纠错验收。

### 可以复用哪些代码

长期依赖优先考虑 [sbi](https://github.com/sbi-dev/sbi)，其当前实现有 NRE 和 BNRE，
并支持自定义观测嵌入。已固定并阅读的核心入口为：

- `sbi/inference/trainers/nre/bnre.py`：BNRE 训练及平衡正则。
- `sbi/neural_nets/ratio_estimators.py`：`RatioEstimator.unnormalized_log_ratio`。
  当前接口要求候选与观测的 batch 前缀一致，不会自动广播；它也没有通用的 `.loss()`，
  损失由具体 NRE trainer 构造。接入时不能照搬其他密度估计器的调用方式。
- [LAMPE 的 NRELoss 与 BNRELoss](https://github.com/probabilists/lampe/tree/master/lampe/inference)
  更小，适合审阅和低层试验。其仓库已声明停止开发并转向 sbi；可按固定版本复用这些
  MIT 组件，不宜把整个 LAMPE 视为仍持续维护的新依赖。

上述文件及许可证声明的固定版本见文末清单；本次没有向 OWAC 的环境安装完整 sbi。

### BNRE 能改善什么

BNRE 在 NRE 分类目标上增加平衡约束，针对有限仿真预算下的过度自信，论文报告了更
保守的后验近似。这与当前无信息输入下仍形成虚假置信的问题相关。
[论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7e6288bfb68182db7d6e328b0aefa89a-Abstract-Conference.html)，
[作者实验代码](https://github.com/montefiore-institute/balanced-nre)。

但该约束作用于批次平均，不能保证每个无信息场景都保持正确歧义。本次用原版
`BNRELoss` 构造了一个接口反例：得分 `s(θ,y)=3θ` 完全忽略观测，在正负 θ 均衡的
批次上，平衡惩罚可以为零，但均匀先验下仍把同一候选推到 0.9975。基础 NRE 分类
损失会惩罚这种得分；这个反例不是 BNRE 的训练结果，只说明不能把平衡项单独当验收。

### 接入 N02 必须保留的语义

1. **候选有可检验的解释。** θ 应描述候选几何或其他可观测状态；只给槽号会把问题
   退化为学习槽偏好。输入只能包含当前真实测量、可用掩码、标定、曝光等允许上下文，
   不得输入合成隐藏真值或未来线索。
2. **训练配对符合算法。** NRE 的负样本来自独立的参数与观测配对，不是强制选择
   “错误类别”。当前连续帧来自同一 episode，不能直接当独立样本随意 shuffle；
   要跨独立 episode 构造配对，并保持所需条件上下文一致。若改用二元世界分类目标，
   必须按其训练类别先验校正 log-odds，不能再声称原样复现 BNRE。
3. **先验与新证据分开。** 测量评分器不直接读取声明先验；先验在权重合成时进入一次。
   循环重算同一测量不重复累加。相关帧不能未经验证当作 IID likelihood 相乘。
4. **相对可信不等于绝对正确。** 似然比无法识别“两候选都不解释观测”，因为它没有
   绝对密度。当前 `revision.py::semantic_map` 还检查最佳候选的归一化残差，不能拿
   ratio 分数直接代替该阈值。第一版宜只对照相对测量更新，保留独立的支持与拟合检查。
5. **全帧得分不自动成为局部反证图。** 不能把一个全局 log-ratio 复制到所有像素，
   或把重叠 patch 当独立证据求和；局部反证需要额外的局部建模与校准。

## 训练优化可以直接借用作者实现

### CAGrad 比当前局部投影覆盖更完整

CAGrad 在平均梯度附近寻找兼顾各任务局部改善的更新方向。作者提供了 PyTorch
实现，可以对全部共享参数上的多个目标做组合，而当前试验只修改了几何 MLP 上
mixture 对 geometry_set 的反向分量。
[论文](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)，
[作者仓库](https://github.com/Cranial-XIX/CAGrad)。

同一作者的 [FAMO 仓库 `methods/weight_methods.py`](https://github.com/Cranial-XIX/FAMO/blob/main/methods/weight_methods.py)
也保留了 `CAGrad`，本次检查了其梯度矩阵与约束优化实现。它依赖 SciPy 的 `minimize`；
当前 OWAC 环境未安装 SciPy，尚未做完整库兼容性运行。可复用该优化内核，适配当前
逐帧 active-loss 平均、任务专属参数和共享参数，保留真实 AdamW 状态及统一裁剪。
不要直接整段替换 production backward：作者实现对共享参数有自己的归零和覆盖规则。

它是值得加入的训练基线，但当前所有几何投影臂仍未通过行为门槛，说明优化冲突未必是
唯一障碍。论文的收敛分析也不自动覆盖当前带状态 detach、AdamW、裁剪的完整 solver。

### FAMO 不能直接接现有损失列表

FAMO 根据各任务损失的进展动态调整权重，避免每轮存储全部任务梯度，适合降低多任务
优化成本。[论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract.html)，
[官方独立模块](https://github.com/Cranial-XIX/FAMO/blob/main/famo.py)。

代码计算 `D = losses - min_losses + 1e-8`，随后取 `log(D)`，默认下界为零。本次实际
将 N02 已存日志中的加权损失传入原版函数：measurement=-2.9221、counterfactual=-2.9231，
结果为非有限值。Gaussian NLL 可以为负，这不是本地训练异常，而是目标接口不匹配。

若采用 FAMO，需要先给出有依据的 loss 下界，或明确只对非负拟合、几何、分类分项
做动态平衡；任意加常数会改变相对权重，不能称为等价变换。未激活的 revision 也不能
不加区分地当作零值任务。本轮原版 FAMO 在正值 toy 目标上的一步更新可运行，但没有
做 N02 更新。因此暂将 CAGrad 排在 FAMO 前面。

### β-NLL 是小改动备选

Seitzer 等研究了可学习异方差 Gaussian NLL 在梯度训练中出现稳定但较差均值估计的
情况，提出按停止梯度后的方差幂对训练损失重权。
[ICLR 2022 论文](https://arxiv.org/abs/2203.09168)，
[作者仓库及可直接使用的函数](https://github.com/martius-lab/beta-nll)。

它适合检验方差加权是否妨碍均值拟合，代码改动很小；但它不专门分离共同背景与微弱
差异。上一轮也没有看到整体方差膨胀，不能预先宣布这是主因。若做对照，只改变训练
损失，推理仍使用原 Gaussian 评分，并同时检查均值、方差校准和实际线索选择。
仓库的核心损失为 MIT；数据加载和深度估计子目录另有许可证来源，不宜整包照搬。

## 多假设推断与物理模型的复用

### DPF 应成为真正的外部算法基线

DPF 把运动预测、观测权重和粒子集合组织成可训练的递归估计器，直接优化状态估计
效果。它提供了一个重要参照：测量模型不必先完整重建图像，标准估计结构也可以和
神经观测模型联合训练。
[Jonschkowski 等，RSS 2018](https://m.roboticsproceedings.org/rss14/p01.html)，
[原作者代码](https://github.com/tu-rbo/differentiable-particle-filters)。

原仓库基于 TensorFlow/Sonnet，不是现有 PyTorch 环境的即插即用依赖。
[torchfilter](https://github.com/stanford-iprl-lab/torchfilter) 是另一个 PyTorch 滤波库，
并非该论文的原版复现。它的 `ParticleFilterMeasurementModel` 接受
`states[N,M,state_dim]` 与观测，返回 `[N,M]` 的 log-likelihood；滤波器在 log-space
累加并归一化，提供软重采样。这些接口适合复用为对照。

限制在于：库不会替我们学好测量模型；其默认加权平均读出不适合保留互斥几何解释，
应检查粒子集合或 MAP 读出；重采样还可能丢失弱候选。N02 的唯一观测 ID 去重也不能
省去。仓库较旧，应在独立环境先做接口和多峰行为验证，再决定引用范围。

当前候选卡把 `tu-rbo/differentiable-particle-filters` 归到 Karkus 名下，与 RSS 官方
页面不符；本次将对应条目改为 Jonschkowski 等。不同 DPF 工作不能混用作者和代码归属。

### DeepInverse 适合复用物理一致性工具

DeepInverse 将正向成像算子、噪声、数据保真项和求解器分开，提供可展开的优化循环。
它能减少 RAW 成像算子与数值求解的重复开发。
[官方物理算子说明](https://deepinv.github.io/deepinv/user_guide/physics/physics.html)，
[展开求解器接口](https://deepinv.org/api/stubs/deepinv.unfolded.unfolded_builder.html)，
[代码](https://github.com/deepinv/deepinv)。

它不能直接把 N02 的二元几何代理变成可靠的 RAW 预测。适配时仍需定义候选状态到
传感器测量的正向关系，以及可观测的曝光、CFA、噪声和遮挡条件。可优先借用 Physics、
noise 和 data-fidelity 组件；其默认 demosaicing/RGB 重建示例及 RGB 预训练去噪器
不能直接放进当前禁止 RGB 中间图像的主链路。该路线有价值，但接入成本高于先验证 NRE。

## 真实变化恢复应借鉴鲁棒在线变点检测

Robust BOCPD 用广义贝叶斯更新降低离群观测引起虚假变点的风险，直接对应“残差大
究竟是噪声，还是旧状态不再成立”的问题。
[Altamirano 等，ICML 2023](https://proceedings.mlr.press/v202/altamirano23a.html)，
[作者代码 DSM-bocd](https://github.com/maltamiranomontero/DSM-bocd)。

可先将可信的低维创新或测量证据流输入该方法，作为 stale 条件下的变化检测基线。
它不直接输出视觉局部撤回图，也不能判断变化对机器人动作是否重要。测量得分如果
始终偏向一个候选，增加变点检测器不会自动恢复缺失的判别信息。

官方参考 `bocpd.py` 虽限制保留的 run-length 数量，却仍分配 `(T+1)×(T+1)` 的历史
矩阵；严格流式接入需要将状态更新与完整历史保存分开。README 使用 Python 3.9、
NumPy/SciPy/JAX 的固定旧环境，本轮仅做源码/接口审阅。它适合第二阶段，不能作为
当前视觉编码器的直接替换。

## 推荐的下一轮复用顺序

具体接入位置如下，第一轮保持为可独立运行的对照，不改写已验收的状态和证据计数契约。

| 当前代码 | 复用方式 | 首轮边界 |
| --- | --- | --- |
| [RAWStem / MeasurementHead](../../src/owac/frameworks/n02/measurement.py) | 复用相同 RAW 输入与嵌入，另建 NRE／BNRE 候选评分头 | 输出每候选相对分数；不伪造 `EnergyResult` 中的局部残差、支持度与绝对拟合指标 |
| [step_terms / TermAccumulator](../../src/owac/training/n02.py) | 对既有多任务目标单独增加 CAGrad 训练臂 | 明确 active-loss、共享与专属参数，保持已有 AdamW、裁剪和更新预算 |
| [ParticleFilterEncoder](../../src/owac/baselines/n02.py) 与 [行为验收](../../src/owac/evaluation/n02_acceptance.py) | 增加 torchfilter 来源明确的外部滤波对照，共用通过验证的测量评分 | 保留多峰输出和唯一观测语义；与现有自写粒子基线分别命名和报告 |

先保留现有 tiny 测量、掩码、标定、先验条件和行为验收，增加**独立的 NRE、BNRE
测量评分基线**。使用相同可用信息，提供冻结同一 RAW stem 的公平对照，并记录各头
的参数、训练数据、梯度更新和计算预算。评分器训练只接触训练 episode；校准、选型
和新的最终留出分开，不能再把已经多次诊断的种子当未见留出。

先回答三个直接问题：交换真实线索能否改变两侧选择；无信息时 log Bayes factor 是否
接近零并保持声明先验；有信息时证据量是否足以纠正 95% 错误先验。BNRE 另检查每种
噪声/遮挡条件的校准，不能只看总体平均。以上成立后，把同一测量评分器同时接入标准
滤波基线和 N02，比较局部撤回是否带来额外收益；此时才有依据讨论 N02 的机制贡献。

CAGrad 可作为独立训练对照，但不宜与新评分器、β-NLL 和变点检测一次性叠加。若
NRE/BNRE 的单独测量判别仍失败，应检查观测嵌入与仿真分布覆盖，而非立即加大模型。
这轮调研不改变既有行为门槛，也没有恢复 800 步训练。

## 固定版本与本地检查

下表为 2026-10-13 查询的默认分支快照，短 SHA 仅用于阅读，完整 SHA、源文件哈希
和核查范围在[来源清单](../experiments/results/2026-10-13_n02_external_methods/source_manifest.json)。
许可证列指仓库声明；文件或第三方子目录的具体声明优先。

| 仓库 | 固定提交 | 声明许可证 | 本次采用范围 |
| --- | --- | --- | --- |
| sbi-dev/sbi | 29af13750100 | Apache-2.0 | BNRE / RatioEstimator / 依赖静态检查；优先长期依赖 |
| probabilists/lampe | cbf500c09c53 | MIT | 原版 NRELoss / BNRELoss 接口运行；开发已停止 |
| montefiore-institute/balanced-nre | 3439e9a0f8ef | BSD-3-Clause | 原论文实验入口及来源核对 |
| Cranial-XIX/CAGrad | dc3d48152b61 | MIT | 原方法来源；内核另查同作者 FAMO 仓库 |
| Cranial-XIX/FAMO | 0a6ad06dbdd5 | MIT | CAGrad 静态审阅；FAMO 原版函数运行 |
| martius-lab/beta-nll | 669c9f251eb4 | MIT，含另许可子目录 | 论文与核心训练函数审阅 |
| stanford-iprl-lab/torchfilter | 4a8a085f6406 | MIT，包元数据另写 BSD | 滤波/测量接口审阅；完整兼容性待验证 |
| deepinv/deepinv | fd3814d7aa84 | BSD-3-Clause | 官方 Physics / unfolded API 审阅 |
| maltamiranomontero/DSM-bocd | 9d05117bc8c6 | MIT | 官方 BOCPD 递推及输入接口审阅 |

本地环境为 Python 3.12.12、PyTorch 2.9.1+cpu、NumPy 2.5.1。通过 AST 提取已经阅读
的原版 NRELoss、BNRELoss、FAMO 类，不改类体，不安装整个外部库：两种 ratio loss
的前向和反传均有限且梯度非零；FAMO 正值 toy 目标一步更新可运行；原样输入 N02
负 NLL 得到非有限值。没有训练 N02，也没有把这些接口检查当作论文性能复现。

回执为 [source_smoke.json](../experiments/results/2026-10-13_n02_external_methods/source_smoke.json)，
脚本为 [source_smoke.py.txt](../experiments/results/2026-10-13_n02_external_methods/source_smoke.py.txt)。
本地复现命令为 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python outputs/n02-literature-20261013/source_smoke.py`，
从本 checkout 根目录运行。归档脚本需恢复到该输出目录，依赖清单指定的三个上游源码
文件及上一轮训练日志；完整下载地址、SHA256、原始日志位置和行号均在来源清单中。
下载的源码及抓取记录留在 ignored `outputs/n02-literature-20261013/`；部分原始文件
下载遇到连接失败/API 限流，清单区分本地文件核查与官方网页/API 审阅，没有将缺失文件
标为本地验证。当前生产源码、依赖锁与原训练检查点保持不变。

`make check` 通过，新增本地链接、JSON 和源码/回执哈希核验通过；检查也发现
`docs/status.md` 中 5 处旧输出页面链接在此 worktree 缺失，均已存在于本轮基线，
未把它们记成本轮通过项。详见[交付检查回执](../experiments/results/2026-10-13_n02_external_methods/validation.json)。

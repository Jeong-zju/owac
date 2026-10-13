# N01 文献与源码复用审查

日期：2026-10-13。对应 OWAC `61a5e3a`；研究范围是寻找可复用的多帧 RAW / 特征融合与历史可靠性方法。本轮检查了 7 个仓库的固定提交、33 个源码/说明/许可文件，并对两个独立融合模块做了 CPU 接口验证。未接入新算法、下载模型权重或开展新一轮任务训练。

**建议先借用 DBSR 的逐帧特征融合，辅以 EDVR 的轻量融合对照；可靠性规则参考 Google 多帧方法及 IPOL 的局部统计与噪声校准。暂不继续围绕窗口长度或单像素风险公式调参。** 原因是这些工作分别对应当前两个具体问题：历史在进入特征前被过早压缩，以及以一张含噪当前帧判断历史对错。

这不是对外部方法在 N01 上性能的保证。当前需要建立一个充分利用历史的已有方法基线，再判断支持关系的建立、撤回与重算是否提供独立收益。

## 1. 从现有实验定位要借用的能力

[内容通路实验](../experiments/2026-10-13_n01_content_integration.md)中，C 相对同容量控制 E 改善约 0.05575 packed px；D 相对 C 退化约 0.00686 px。改善小，训练也未证明收敛，因此不能推导出“自适应一般无效”。[逐来源可靠性实验](../experiments/2026-10-13_n01_source_reliability.md)进一步表明，这版权重在没有任务网络参与的静态纯噪声实验中也把 MSE 增至约 3.11 倍，故该失败不能全部归因于训练不足。

源码中的区别是：

- `content.py::content_fields` 先把历史压成每 CFA 平面的均值等字段，`ContentUpdate` 再投影进特征。这已证明有内容通路，但网络看不到每帧局部细节及不同帧间的分歧。多个历史集合可以产生相同的积分均值。
- `reliability.py::innovation_weights` 用当前单个含噪测量作参考，正部截断估计超额差异；`history_scale` 再压低整段历史。参考噪声同时影响全部旧测量，产生相关选择偏差。当前帧异常时，也可能排斥正确历史。
- 以上不等于旧核心积分、唯一来源计数或分裂后重算错误。本轮不修改这些机制。

文献里常见的做法是保留各帧特征，在局部空间和多帧上下文中判断什么有用，再融合。固定保存最近 8 帧仍可进行自适应融合；“存多少帧”和“每个位置用哪帧、用多少”是不同问题。准确对齐的高速运动仍可能提供有效历史，不能把“越快越短”本身作为算法成功标准。

## 2. 最值得复用的四项工作

| 工作 | 借用什么 | 对应 N01 问题 | 实际接入判断 |
| --- | --- | --- | --- |
| Deep Burst Super-Resolution，CVPR 2021 | `WeightedSum`：逐位置、逐特征通道预测跨帧融合权重；可用参考帧或多帧平均特征作比较基准 | 历史过早压成均值；单一含噪参考 | 最贴近当前目标，独立模块已在现有 Torch 上实跑；整网的 RGB 对齐/输出不能原样搬入 |
| EDVR，CVPR Workshops 2019 | `TSAFusion`：时间相关门控、融合卷积和空间注意力 | 需要容易接入、有空间上下文的成熟特征融合对照 | 纯 Torch 模块已实跑；不需要同时移植 PCD 可变形对齐 |
| Handheld Multi-Frame SR，SIGGRAPH 2019；IPOL 2023 实现 | 局部均值/方差、噪声校准的差异收缩、运动场一致性和可靠性掩码 | 把噪声误判成旧历史错误 | 解析可靠性的重要参照；当前代码依赖 CUDA，且统计经过 RGB guide，需改为 CFA 分平面并重新校准 |
| KPN，CVPR 2018 | 局部时空核预测；避免网络只用参考帧的退火辅助训练 | 历史有信息，但网络未学会利用 | 训练诊断高度相关；原代码为 TensorFlow 1 风格/Python 2 语法，不是当前 Torch 的即插即用模块 |

### 2.1 DBSR：首选的内容融合模块

[论文](https://arxiv.org/abs/2101.10997)和[作者仓库](https://github.com/goutamgmb/deep-burst-sr)。已检查提交 `16f158e0553b6702cc1693c43f733c2830973278`。

可以直接复用的范围是 [`models/dbsr/merging.py::WeightedSum`](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/merging.py)，依赖同仓库 `models/layers/blocks.py`。输入为参考特征 `[B,1,C,H,W]`、已对齐的其他帧特征 `[B,T-1,C,H,W]` 和偏移 `[B,T-1,2,H,W]`；输出为融合特征 `[B,C,H,W]` 和跨帧权重 `[B,T,C,H,W]`。默认 softmax 沿时间轴归一化，参数不随 T 增长。

其权重预测器读取比较基准、各帧相对基准的特征差，以及可选采样偏移。类默认 `use_base_frame=False`，使用多帧平均特征；但 [`dbsrnet_cvpr2021`](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/dbsrnet.py) 整网工厂默认传入 `use_base_frame=True`。**必须显式配置两种模式，不能把多帧平均说成作者整网的默认策略。** 多帧平均也会被占多数的陈旧帧影响，它只是可检验的替代参考，不是已解决遮挡变化。

与 N01 的最小结合方式是：唯一 RAW 缓冲中的各帧分别通过共享编码器，再对齐特征，最后调用这个融合器，输出直接进入任务读出。RAW 均值、方差、年龄和有效性另存，不把注意力权重解释为独立测量的逆方差权重。这是采用已有融合模块的基线；在这条基线上加 N01 支持约束才是后续研究。

不能照搬的部分：[`encoders.py`](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/encoders.py) 把两个绿平面平均，构造三通道 `x_rgb` 供 PWC-Net 对齐；整网加载 PWC 权重，并输出高分辨率 RGB。N01 应先使用已有 RAW 对应或机制实验的准确对齐，保留四平面及采样坐标，去掉 RGB 输出头，重新进行任务训练。不能用原论文的 RGB 重建权重冒充 RAW-only 主方法的训练结果。

许可是 CC BY-NC-SA 4.0，文件还明确限学术研究使用；适合当前研究复现，但应保留来源和许可，后续产品化不能默认按宽松许可处理。

### 2.2 EDVR：最省事的工程对照

[论文](https://arxiv.org/abs/1905.02716)；[固定版 `TSAFusion`](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/edvr_arch.py)。BasicSR 提交 `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`，该模块适用项目 Apache-2.0 许可。

输入 `[B,T,C,H,W]` 已对齐特征，输出 `[B,C,H,W]`。时间权重来自特征相关性经 sigmoid，再经拼接卷积和空间注意力；这里不是跨帧 softmax，也没有显式噪声后验。模块本身只需要普通 Torch。导入整份 EDVR 文件会引入 PCD/DCN 依赖，独立抽取 TSA 即可避免安装整套复原框架。

`num_frame` 决定融合卷积通道数，因此同一个实例固定 T；`center_frame_idx` 默认 2。按从旧到新排列 8 帧时应显式置为 7，只提供已经采集的帧。N01 当前缓冲最新帧在前，适配时必须重排或显式选择索引 0，不能照抄中间帧配置。

两层空间下采样对尺寸有要求，本轮测试的是 16×16。启动不足 8 帧、空间无效区需要单独适配；补零不等于被正确忽略。全零输入在本轮随机初始化时仍输出非零特征，这是卷积偏置/空间残差的正常结果，但不能据此宣称缺失位置已有测量支持。

### 2.3 Google / IPOL：最贴近错误历史识别的解析方案

[Google 原始论文入口](https://research.google/pubs/handheld-multi-frame-super-resolution/)、[IPOL 论文与源码](https://www.ipol.im/pub/art/2023/460/)、[IPOL 作者维护仓库](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution)。这是 Google 方法的公开复现，不是 Google 发布的生产源码。

有价值的区别是：先用局部统计减少单点噪声干扰，再用噪声模型估计静态场景本来会产生多大差异，收缩可由噪声解释的部分，并联合局部运动场变化得到可靠性。可参考 [`robustness.py`](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution/blob/27c4655ae4752898e0ae6e9737697c4d89c95226/handheld_super_resolution/robustness.py) 的 `compute_local_stats`、`compute_d_sigma`、`compute_s`、`robustness_threshold`。这直接对应 N01 目前以单个像素差异排斥历史的缺陷，但仍有参考依赖和弱变化不可辨识问题。

已检查的最新提交是 `27c4655ae4752898e0ae6e9737697c4d89c95226`，MIT 许可。需要区分三个版本：Google 2019 论文、IPOL 2023 论文/发布源码、维护仓库 2026 版。维护仓库 README 明确说明已明显偏离 IPOL 在线 demo；当前代码包含开平方的三通道 guide 和对应的相机噪声 Monte Carlo LUT。这些新增处理不是本轮确认的 Google 原始方法细节。

最新代码的 `compute_guide_image` 会合并两个绿平面；[`noise_lut.py`](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution/blob/27c4655ae4752898e0ae6e9737697c4d89c95226/handheld_super_resolution/noise_lut.py) 则校验 LUT 所对应的变换、噪声 profile 和统计窗口。因此，不能把现成 RGB-guide LUT 用于 N01 的四 CFA 线性测量。应借用统计和校准方法，对四平面、实际采样、插值、增益及缺失规则重新标定。当前实现使用 Numba CUDA，尚未在本机运行。

如果继续解析分支，先要求它通过纯静态噪声、弱变化、强变化、当前帧异常四种积分诊断，并画误拒率与漏拒率，再进入任务训练。固定验证阈值和留出噪声参数，避免继续围绕同一异常样例手调。

### 2.4 KPN：训练为何可能一直忽略历史

[论文/作者项目页](https://bmild.github.io/kpn/)、[作者发布代码](https://github.com/google/burst-denoising)。检查提交 `7ff07e009568778ff337c259e192d1d81aedbea6`，Apache-2.0。

论文讨论了一个与当前现象相符的优化陷阱：单帧处理较容易，联合训练可能主要使用参考帧，其他帧分支难以形成有用能力。其退火策略在训练早期约束每帧经过预测核后能对目标作出贡献，再逐渐放松辅助项，让联合融合接管。实现对应 [`kpn_train.py`](https://github.com/google/burst-denoising/blob/7ff07e009568778ff337c259e192d1d81aedbea6/kpn_train.py) 的 `anneal_coeff` 和 [`kpn_arch.py`](https://github.com/google/burst-denoising/blob/7ff07e009568778ff337c259e192d1d81aedbea6/kpn_arch.py) 的 `convolve` / `convolve_per_layer`。

可借鉴“先让来源分支学会提供信号”的训练安排，但原始核不是 N01 已校准的正归一化积分系数；原训练还带图像域处理，不能照搬监督链。对 N01 更合理的候选是 RAW 域预测辅助任务，或在确实相容且可观测的片段上做共享任务读出辅助训练。不能强迫遮挡前旧帧预测它根本未观测到的新状态，否则会鼓励错误补全。

它支持优化困难这种可能解释，不证明当前只需增加步数。N01 的无网络纯噪声负结果仍然需要改估计规则；768 步是否足够则要看学习曲线和对照收敛情况。

## 3. 查过源码但暂不优先接入的方案

| 工作与来源 | 核查结果 | 本轮判断 |
| --- | --- | --- |
| [BIPNet，CVPR 2022](https://github.com/akshaydudhane16/BIPNet) | `Burst_Super_Resolution/Network.py` 含 4 通道 RAW stem、可变形对齐和 pseudo-burst 融合；`forward` 直接 `burst = burst[0]`，按单个 burst 实现，默认 14 帧、RGB 输出；实际使用 torchvision `DeformConv2d`。固定树没有 LICENSE 文件，检查的网络及 README 未给出明确复用许可 | 可以作为更完整的复现候选，但比拆出 DBSR/TSA 更重；需处理 batch、帧数与许可边界，尚未运行 |
| [FloRNN，ECCV 2022](https://github.com/nagejacob/FloRNN) | `models/flornn_raw.py::FloRNNRaw` 接收 RAW 与噪声参数，但调用 `demosaic` 后用 PWC 求光流；`forward_count=2`，在读到 i 时输出 i−2。依赖 softsplat/PWC；GPL-3.0 | 单向执行不等于零未来帧。适合研究带明确延迟预算的递归方案，不适合作为当前因果 RAW-only 编码器的直接替换 |
| [RViDeformer，TCSVT 2025](https://github.com/cao-cong/RViDeformer) | `models/RViDeformer_arch.py` 默认 4 通道进/出，可研究中间特征；实际 denoising 分支为 `upscale=1`。序列窗口注意力未施加因果约束，代码有 `.cuda()`；`train_CRVD_supervised.py` 明确优化 `raw_l1 + 0.5 * srgb_l1`，加载 ISP 权重。固定树未找到 LICENSE | 较新的 RAW 视频 backbone 参考，已有权重入口，但权重不等于纯 RAW 训练；接入成本高于两个融合模块，尚未运行 |

没有把近期但未核实可运行代码的方法放到首选：[FSN，CVPRW 2024](https://postech-cvlab.github.io/Burst_FSN/)研究参考帧质量选择，其[同名仓库](https://github.com/POSTECH-CVLab/Burst_FSN)目前核查到的是项目网页；[QMambaBSR，CVPR 2025](https://arxiv.org/abs/2408.08665)研究跨帧查询与噪声分离，本轮未找到可核实的作者模型实现；[BurstMamba，SIGGRAPH Asia 2026 作者项目页](https://ouenal.github.io/burstmamba/)提供关键帧/历史分支计算分工与对应关系传递思路，核查页面有论文和数据入口，未发现模型代码入口。未发现不等于确认没有代码。

[NTIRE 2026 RawFusion starting kit](https://github.com/Eve-ctr/RawFusion)虽然也输入 RAW burst，公开 baseline 是拼接 U-Net，输出 RGB；没有理由仅因年份新就排在已有独立融合模块之前。本轮不改变项目主数据路线，外部成像数据只作为方法背景。

## 4. 这次实际跑通了什么

只从固定源码提取原样的类/函数 AST，注入其普通 Torch 依赖；没有修改类内算法。运行前将文件内容与相应 Git tree 的 blob SHA-1 比对，33 个归档源文件均一致。完整清单、源码 SHA256 与测试数字见[收据](2026-10-13_n01_prior_art_sources.json)。

| 模块 | 本轮小尺寸配置 | 结果 |
| --- | --- | --- |
| EDVR `TSAFusion` | B=2，T=8，C=8，H=W=16，参考索引 7；5,544 参数 | 输出 `[2,8,16,16]`，前向/反向有限，每帧输入梯度非零 |
| DBSR `WeightedSum` | B=2，C=8，H=W=16；project=8，offset=4；9,140 参数 | 同一模块分别处理 T=1/4/8；平均基准和参考基准共 6 次均通过；跨帧权重和最大误差 2.39e-7 以下 |

环境为本仓库锁定的 Torch 2.9.1+cpu，单线程，随机种子 20261013、随机初始化/特征，未下载权重。这只是模块接口和可微路径验证，**不是完整仓库可安装证明、因果性验收、RAW 编码验收或性能改善证据**。参数量仅为本轮缩小配置的融合器，不是原论文整网参数量。

交付检查：`make check` 通过（ruff、format、compileall），探针复跑与归档数字一致，文档本地链接和 JSON 清单核验通过。本轮没有修改产品源码，因此没有重复执行全仓任务测试。

[探针脚本](probes/n01_upstream_fusion.py)可用现有本地缓存复跑：

```bash
uv run --locked python research/reviews/probes/n01_upstream_fusion.py outputs/n01-literature-20261013
```

缓存中保留固定提交 Git 元数据、源码 ZIP、提取文件和日志，均在被忽略的 `outputs/n01-literature-20261013/`；不把第三方源码并入 OWAC 源码树。跨机器复跑时，依据 JSON 中的 repo/commit 从作者仓库取回版本，把所需文件放在 `sources/<仓库名>/<原路径>`，并提供对应提交的 `<仓库名>/.git`；不依赖浮动 master/main。

## 5. 建议的下一轮最小对照

先做独立的、同任务监督的 RAW→特征基线：

```text
各个唯一 RAW 帧及 CFA 相位、噪声/有效性
    → 共享逐帧 RAW 编码器
    → 已有因果对齐（机制对照另报准确对齐）
    → DBSR WeightedSum 或 EDVR TSAFusion
    → 当前任务特征读出
```

全部只取当前及过去观测，不接 RGB 光流、不加载 RGB 重建权重，不输出 RGB。优先考虑 DBSR 的变帧数接口；TSA 是独立的结构和工程对照。建议的比较顺序如下，暂不把所有文献堆进一个新模型：

1. 同一共享编码器、同一读出下：当前帧、固定 8 帧特征平均、DBSR 融合。另列现有 C 为旧实现参照，不能把两条路径的容量差混入融合收益。
2. DBSR 中只改变比较基准：参考帧 vs 多帧平均。沿用静态弱信号、准确对齐运动、遮挡解除、局部错误历史，并加入当前帧异常反证。
3. 若成熟融合已形成可靠收益，再在同一 backbone 上比较支持图约束开/关、撤回后重算开/关。只有这一步回答 N01 的支持机制增量。
4. 解析路线单独验证 CFA 局部统计与噪声校准；通过无网络积分诊断后才接入上述任务模型。KPN 式辅助训练作为独立因素，避免一次改变融合、可靠性和训练目标后无法归因。

参数匹配与充分调优性能分别报告；匹配窗口、训练样本、优化步数、可用对齐信息及读出容量，实际计入各帧编码、对应、缓存、图维护和读出的延迟/内存。若缓存特征，应明确训练时重算/截断梯度规则，避免参数更新后继续使用旧权重生成的缓存特征。

接入时仍需解决的具体边界：

- 上游模块没有 N01 的 source ID、有效性和重复帧语义。调用前按唯一来源去重；缺失/越界不能作为零值观测参加多帧均值和 softmax，需显式有效性适配，并覆盖全无效情况。
- DBSR 偏移按 `(x,y)`；N01 的平移采样按 `(row,column)`。方向、正负号、packed/full-sensor 单位、CFA 相位和 `grid_sample` 的 `align_corners` 要用手算位移核对。先证明接口一致再更换光流。
- 学习特征可能已通过卷积跨越空间闭边。仅在最终 attention 上加图 mask，不能自动证明“所有特征都不跨闭边混合”；原 N01 的测量支持不变量与特征依赖应分别验收。撤回后若仍使用递归污染特征，重新算权重也不等于撤回证据。
- 记录事件后逐帧错误、恢复时间、错误历史质量、静态降噪和当前异常鲁棒性。不能只看最后一帧或有效历史长度；参考帧正确、错误两种方向都要测。

研究问题因此可以更清楚地表述为：**在成熟的多帧 RAW 特征融合已经充分利用历史之后，显式支持关系的建立与撤回，能否在错误历史出现时提高修正速度，同时保留静态弱信号的收益？** 学习融合本身、RAW 输入本身和动态权重本身都已有直接先例。

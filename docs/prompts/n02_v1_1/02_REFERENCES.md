# 论文、官方实现与固定文件导航

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

2026-10-13 提示词编写时按固定版本取回52个文件（包括许可证），见[source_manifest.json](source_manifest.json)。源码缓存位于ignored `outputs/n02-v11-prompt-audit-20261013/sources/`。清单记录文件内容hash；本次只做指定语义审阅，没有运行这些上游网络、Julia或CUDA求解器。论文核对以摘要/项目和原方案来源为主，正式理论推导仍需阅读全文。

2026-10-14 更新：P0 已实跑参考与 parity，见[独立验收](../../../research/experiments/2026-10-14_n02_p0_independent_acceptance.md)和[参考环境锁](../../../configs/backends/n02-full-runtime/INDEPENDENT_ACCEPTANCE.md)。后续在[指定远端](03_REMOTE_DEVELOPMENT.md)阅读/运行参考；所需缓存可按本清单固定 SHA 在远端获取，或选择性同步既有源码，旧缓存路径相对于接入后的远端 worktree。下载/编译结果与实际数值验证仍分别记录。

引用编号与v1.1方案一致，不要求实现全部30种方法。下面直接指明各阶段需要阅读的文件；其他相关工作见[原方案参考文献](../../plans/N02_full_development_plan_2026-10-13.md#12-参考文献与可复用实现)。

## R01

**ConvNeXt V2 / FCMAE** · [论文/官方入口](https://arxiv.org/abs/2301.00808)。P0/P1：正式Base、GRN与掩码预训练。读convnextv2_base、GRN、FCMAE.patchify/unpatchify/gen_random_mask；适配RAW通道与矩形，保留稀疏语义。

固定提交：`2553895753323c6fe0b2bf390683f5ea358a42b9`。

| 固定源码 | 读取任务 |
| --- | --- |
| [models/convnextv2.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/convnextv2.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [models/convnextv2_sparse.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/convnextv2_sparse.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [models/fcmae.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/fcmae.py) | 修复通道数/矩形假设前的原版掩码encoder-decoder与loss |
| [utils.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/utils.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [TRAINING.md](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/TRAINING.md) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [INSTALL.md](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/INSTALL.md) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [engine_pretrain.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/engine_pretrain.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/LICENSE) | 代码许可与保留声明 |
| [models/utils.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py) | GRN、SparseGRN、LayerNorm及全空间依赖 |

## R02

**SAVi / SAVi++** · [论文/官方入口](https://arxiv.org/abs/2111.12594)。P0/P2：Processor预测校正、SlotAttention、GRU与对象decoder。原始slot attention忽略padding_mask，RAW有效性为显式新增差异；训练监督条件按原论文披露。

固定提交：`ba8f15ee19472c6f9425c9647daf87910f17b605`。

| 固定源码 | 读取任务 |
| --- | --- |
| [savi/modules/attention.py](https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/attention.py) | inverted attention归一化轴、GRU更新和未使用的padding_mask |
| [savi/modules/video.py](https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/video.py) | Processor.corrected/predicted状态及时间scan |
| [savi/modules/misc.py](https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/misc.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [savi/modules/decoders.py](https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/decoders.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [savi/modules/initializers.py](https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/initializers.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [savi/configs/movi/savi++_conditional.py](https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/configs/movi/savi++_conditional.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/LICENSE) | 代码许可与保留声明 |

补充：[SAVi++论文](https://arxiv.org/abs/2206.07764)。源码的`initializers.py`含条件初始化；正式无隐藏truth推断不能直接使用真实首帧框。

## R04

**SCALOR** · [论文/官方入口](https://arxiv.org/abs/1910.02384)。P3/P5：discovery与propagation、存在性与实体生命周期的独立强对照。学习对象发现不等于N02独有机制。

固定提交：`8d116c0459d3fb379b204cdc58e7a76a707b365f`。

| 固定源码 | 读取任务 |
| --- | --- |
| [discovery.py](https://github.com/JindongJiang/SCALOR/blob/8d116c0459d3fb379b204cdc58e7a76a707b365f/discovery.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [propagation.py](https://github.com/JindongJiang/SCALOR/blob/8d116c0459d3fb379b204cdc58e7a76a707b365f/propagation.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [scalor.py](https://github.com/JindongJiang/SCALOR/blob/8d116c0459d3fb379b204cdc58e7a76a707b365f/scalor.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [modules.py](https://github.com/JindongJiang/SCALOR/blob/8d116c0459d3fb379b204cdc58e7a76a707b365f/modules.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/JindongJiang/SCALOR/blob/8d116c0459d3fb379b204cdc58e7a76a707b365f/LICENSE) | 代码许可与保留声明 |

## R06

**dNRI** · [论文/官方入口](https://openaccess.thecvf.com/content_CVPR_2020/html/Graber_Dynamic_Neural_Relational_Inference_CVPR_2020_paper.html)。P0/P4：核心DNRI、DNRI_Encoder、DNRI_Decoder都在dnri/models/dnri.py，不能只复制另一个encoders.py。训练reverse_rnn的posterior与在线single_step_forward严格分开。

固定提交：`96c94694608fcf2a2b372699a74e5c57be3c89f9`。

| 固定源码 | 读取任务 |
| --- | --- |
| [dnri/models/dnri.py](https://github.com/cgraber/cvpr_dNRI/blob/96c94694608fcf2a2b372699a74e5c57be3c89f9/dnri/models/dnri.py) | 训练双向posterior、在线forward prior、single_step_forward、关系decoder |
| [dnri/models/encoders.py](https://github.com/cgraber/cvpr_dNRI/blob/96c94694608fcf2a2b372699a74e5c57be3c89f9/dnri/models/encoders.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [dnri/models/decoders.py](https://github.com/cgraber/cvpr_dNRI/blob/96c94694608fcf2a2b372699a74e5c57be3c89f9/dnri/models/decoders.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [dnri/training/train.py](https://github.com/cgraber/cvpr_dNRI/blob/96c94694608fcf2a2b372699a74e5c57be3c89f9/dnri/training/train.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/cgraber/cvpr_dNRI/blob/96c94694608fcf2a2b372699a74e5c57be3c89f9/LICENSE) | 代码许可与保留声明 |

## R09

**Theseus** · [论文/官方入口](https://arxiv.org/abs/2207.09442)。P0/P2：AutoDiffCostFunction、Objective、LevenbergMarquardt、TheseusLayer。原simple_example用GaussNewton展示外层梯度；本项目另建LM与完整能量接受测试。

固定提交：`c8583de41824613fb135ee9bad0e930ded6be404`。

| 固定源码 | 读取任务 |
| --- | --- |
| [theseus/optimizer/nonlinear/levenberg_marquardt.py](https://github.com/facebookresearch/theseus/blob/c8583de41824613fb135ee9bad0e930ded6be404/theseus/optimizer/nonlinear/levenberg_marquardt.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [theseus/theseus_layer.py](https://github.com/facebookresearch/theseus/blob/c8583de41824613fb135ee9bad0e930ded6be404/theseus/theseus_layer.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/facebookresearch/theseus/blob/c8583de41824613fb135ee9bad0e930ded6be404/LICENSE) | 代码许可与保留声明 |
| [examples/simple_example.py](https://github.com/facebookresearch/theseus/blob/c8583de41824613fb135ee9bad0e930ded6be404/examples/simple_example.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [theseus/core/cost_function.py](https://github.com/facebookresearch/theseus/blob/c8583de41824613fb135ee9bad0e930ded6be404/theseus/core/cost_function.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |

## R13

**Gen** · [论文/官方入口](https://github.com/probcomp/Gen.jl)。P0/P3/P4/P5：SMC、MH、变维trace translator与增量重执行的独立参考。Julia独立运行，通过相同输入/输出协议核对。

固定提交：`2a27298b392e89fa67837f8871b624a41d901d19`。

| 固定源码 | 读取任务 |
| --- | --- |
| [src/inference/particle_filter.jl](https://github.com/probcomp/Gen.jl/blob/2a27298b392e89fa67837f8871b624a41d901d19/src/inference/particle_filter.jl) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [src/inference/mh.jl](https://github.com/probcomp/Gen.jl/blob/2a27298b392e89fa67837f8871b624a41d901d19/src/inference/mh.jl) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [src/inference/trace_translators.jl](https://github.com/probcomp/Gen.jl/blob/2a27298b392e89fa67837f8871b624a41d901d19/src/inference/trace_translators.jl) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [docs/src/ref/inference/trace_translators.md](https://github.com/probcomp/Gen.jl/blob/2a27298b392e89fa67837f8871b624a41d901d19/docs/src/ref/inference/trace_translators.md) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/probcomp/Gen.jl/blob/2a27298b392e89fa67837f8871b624a41d901d19/LICENSE) | 代码许可与保留声明 |

官方文档：[SMC](https://www.gen.dev/docs/stable/ref/inference/pf/) · [Trace Translators](https://www.gen.dev/docs/stable/ref/inference/trace_translators/)。网页本轮可读；API仍以固定源码/参考环境为准。

## R20

**Perceiver IO** · [论文/官方入口](https://arxiv.org/abs/2107.14795)。E1/E2/P6：perceiver.py里的注意力、encoder/decoder和位置编码。512宽、128 latents、每相机256查询属于本包配置，不是上游默认。

固定提交：`f5de0ede8430809180254ee957abf36ed62579ef`。

| 固定源码 | 读取任务 |
| --- | --- |
| [perceiver/perceiver.py](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/perceiver/perceiver.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [perceiver/position_encoding.py](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/perceiver/position_encoding.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/LICENSE) | 代码许可与保留声明 |

## R25

**固定RawVLA / StarVLA π0.5** · [论文/官方入口](https://arxiv.org/abs/2609.37530)。P0/E1/P6/P7：原NPZ索引、PI0/PI05 prefix与动作专家。固定57d508...用于后端阅读，主RawVLA benchmark/ISP另有ba9c56...版本，不混为同一快照。

固定提交：`57d508cb180d0510ac9060358009c0f6ffaac420`。

| 固定源码 | 读取任务 |
| --- | --- |
| [starVLA/dataloader/rawvla_npz_datasets.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py) | 真实NPZ相机/状态/动作索引、原归一化与PIL边界 |
| [starVLA/model/framework/VLM4A/PI0.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [starVLA/model/framework/VLM4A/PI05.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [starVLA/model/modules/action_model/OpenPI_ActionHead.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/LICENSE) | 代码许可与保留声明 |

benchmark/ISP与两平台配置的固定文件另见[既有参考清单](../reference_sources.json)；本地已运行桥为[rawvla_features.py](../../../src/owac/backends/rawvla_features.py)。官方[OpenPI论文](https://arxiv.org/abs/2504.16054)及[固定PyTorch实现](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py)用于核对概念，不更换实际后端。

## R27

**NRE / BNRE** · [论文/官方入口](https://proceedings.mlr.press/v119/hermans20a.html)。P1：同Base相对评分对照；joint/product-of-marginals的batch使用独立轨迹，不沿时间轴伪造负样本。不能用ratio当绝对拟合检验。

固定提交：`29af13750100b61ad528c3f8a49daa0ef8c96881`。

| 固定源码 | 读取任务 |
| --- | --- |
| [sbi/inference/trainers/nre/bnre.py](https://github.com/sbi-dev/sbi/blob/29af13750100b61ad528c3f8a49daa0ef8c96881/sbi/inference/trainers/nre/bnre.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [sbi/neural_nets/ratio_estimators.py](https://github.com/sbi-dev/sbi/blob/29af13750100b61ad528c3f8a49daa0ef8c96881/sbi/neural_nets/ratio_estimators.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |

较小的原版损失实现（本轮重新取回）：
- [lampe/inference/nre.py](https://github.com/probabilists/lampe/blob/cbf500c09c538a014e72477d166acc62a6d8409a/lampe/inference/nre.py)
- [lampe/inference/bnre.py](https://github.com/probabilists/lampe/blob/cbf500c09c538a014e72477d166acc62a6d8409a/lampe/inference/bnre.py)

[BNRE论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7e6288bfb68182db7d6e328b0aefa89a-Abstract-Conference.html)；[作者实验](https://github.com/montefiore-institute/balanced-nre)；[本地前轮原版loss运行及边界](../../../research/reviews/2026-10-13_n02_external_methods.md)。本地旧PrefixRatioHead强制冻结stem，是旧诊断；P1新同Base可训练对照需要单独适配，不能原封不动继承冻结断言。

## R28

**particles SMC** · [论文/官方入口](https://github.com/nchopin/particles)。P0/P3/P5：SMC状态机、log weights、ESS、重采样和state-space目标的NumPy参考。CPU参考性能与GPU主实现分报。

固定提交：`3388c4bd30ab9acc5a27c970089230f93610b627`。

| 固定源码 | 读取任务 |
| --- | --- |
| [particles/core.py](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/core.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [particles/resampling.py](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/resampling.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [particles/state_space_models.py](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/state_space_models.py) | 对应模块的原始实现/训练配置；按本阶段指定函数阅读 |
| [LICENSE](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/LICENSE) | 代码许可与保留声明 |

## R10

**DeepInverse**：[官方Physics与noise说明](https://deepinv.org/user_guide/physics/physics.html)（本轮可读）与[代码](https://github.com/deepinv/deepinv)。仅取传感器算子/噪声/数据保真接口；原RGB denoiser/demosaicing不进入主链。固定版本`fd3814d7aa84`的完整SHA及前轮文件核查见[来源清单](../../../research/experiments/results/2026-10-13_n02_external_methods/source_manifest.json)；本轮未重跑该库。

## R11

**iSAM2 / GTSAM**：[官方ISAM2说明](https://borglab.github.io/gtsam/isam2/)（本轮可读）及[作者论文入口](https://publications.ri.cmu.edu/isam2-incremental-smoothing-and-mapping-using-the-bayes-tree)。P5使用已知连续因子子问题；本包未固定GTSAM二进制，P0/P5按实际运行版本补锁，不将网页版本冒充已装版本。

## R12

**ATMS**：[de Kleer出版方DOI](https://doi.org/10.1016/0004-3702%2886%2990080-9) · [作者文献目录](https://www.dekleer.org/bib.htm)。用于AND/OR支持、共享祖先和替代支持语义。原方案来源可定位；本轮DOI网页读取失败，不能称本轮全文复核。没有原作者可运行代码收据；T3的实际神经计算等价性另证明。

## R14

**RJMCMC**：[Green 1995 DOI](https://doi.org/10.1093/biomet/82.4.711) · [作者软件页](https://people.maths.bris.ac.uk/~mapjg/Cpt/)。用于变维接受概率、逆映射和Jacobian。DOI本轮网页读取失败；Gen固定源码作为可实现参考，不能只凭本文简化split例宣称一般变维网络正确。

## R16

**PALM**：[作者机构论文页](https://www.tse-fr.eu/publications/journal-article/proximal-alternating-linearized-method-nonconvex-and-nonsmooth)。沿用原方案来源，本文未重新精读全文。P2只先证明充分下降/有界能量性质，驻点结论需另外核查正则性与KL假设。

## R18

**序贯检验**：[Game-theoretic statistics论文](https://arxiv.org/abs/2210.01948) · [作者课程](https://www.stat.cmu.edu/~aramdas/SAVI/savi20.html)。沿用原方案来源；P1固定窗口校准，P5只在已证明条件期望约束的合成子集实现e-process，不把BNRE logit转成伪e-value。

## 参考材料的使用顺序

先读取当前阶段指定的固定文件，再按P0运行原算子和parity；先拿到相同输入/参数下的输出与梯度收据，再改RAW通道、mask、时间或状态接口。每项移植差异在源码头和报告中列明。网页可访问、下载成功、参数量相近都不代替一致性验证。

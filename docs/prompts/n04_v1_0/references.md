# N04 开发参考实现导航

优先读与当前阶段直接相关的文件和函数。参考代码用于复用成熟部件、明确算法边界，不要求把全部方法串联进 N04。2026-10-14 核查层级：**本轮实读**指当前源码或网页选段；**既有固定来源**指仓库已有 commit/path/hash 清单，本轮没有重新执行外部实现；**条件参考**只有需要该分支时才集成。

## 核心复用

| 编号 | 论文或实现 | 看什么及在本项目怎样使用 | 核查与复用状态 |
| --- | --- | --- | --- |
| R01 | [RKN 论文](https://proceedings.mlr.press/v97/becker19a.html)、[作者 cell 固定版本](https://github.com/ALRhub/rkn_share/blob/7cb29e1c57307a87729fd5e20ca60a22f5798da5/rkn_cell/RKNCell.py) | locally linear mixture、upper/lower/side covariance、update→next predict；S2 的正式状态核心 | 本轮实读本地 vendor 和网页；MIT，固定 `7cb29e1c...`；本地 [provenance](../../../src/owac/baselines/_vendor/rkn/provenance.json) |
| R02 | [Ac-RKN 论文](https://proceedings.mlr.press/v155/shaj21a.html)、[MTS3 acRKN.py](https://github.com/ALRhub/MTS3/blob/master/agent/worldModels/acRKN.py) | 保留 RKN 转移，新增控制的加性影响；沿 acRKN 文件继续查其导入的转移实现 | 论文与代码入口本轮实读；MTS3 文件级授权/固定 SHA 待正式集成核定。默认由 MIT RKN 按论文实现 |
| R03 | [ConvNeXt 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)、[models/convnext.py](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)、[LICENSE](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE) | Block、downsample_layers、stages、convnext_base；完整 [3,3,27,3]/[128,256,512,1024]，导出空间特征 | Base 定义本轮实读；MIT 来源沿用方案。REST 403，新增固定 SHA 未获取，S1 下载时固定；随机初始化主线 |
| R04 | [本地原型训练器](../../../src/owac/training/n04.py)、[本地作者 RKN wrapper](../../../src/owac/baselines/n04_tracking.py) | 原型加载/损失入口；posterior 与 prior 时间索引；不要照抄不完整 checkpoint 状态 | 本轮实读；为历史参照，不冒充 v1.0 正式模型 |
| R05 | [RawVLA NPZ 作者 loader](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py)、[数据卡](https://huggingface.co/datasets/ToferFish/RawVLA-Bench) | 平台 state/action 时序、原始字段；作者 PIL/RGB 输出不用于 RAW 主链 | loader 固定源码页与数据卡本轮打开；本地 [独立测量 loader](../../../src/owac/raw/rawvla_sequence.py) 本轮实读 |
| R06 | [RawVLA LIBERO raw.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/libero/rawvla_bench/raw.py)、[rawvla policy](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/baselines/rawvla/rawvla.py)、[LIBERO config](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/libero/pi05.yaml)、[RoboTwin config](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/robotwin/pi05.yaml) | 追踪传感器生成→观测→动作块→执行器；S0 查控制与标定来源，S7 复用冻结协议 | 既有固定来源；远端实际部署 HEAD 本轮核实 `ba9c56b...`，部署补丁/数值边界由本地报告另记 |

## 坐标与误差建模

| 编号 | 来源 | 具体用途与边界 |
| --- | --- | --- |
| R07 | [Kornia depth.py](https://github.com/kornia/kornia/blob/4fcfcf863e0a9e2b3e93179c3fa96bcf278ce3c6/kornia/geometry/depth.py)、[imgwarp.py](https://github.com/kornia/kornia/blob/4fcfcf863e0a9e2b3e93179c3fa96bcf278ce3c6/kornia/geometry/transform/imgwarp.py) | 反投影/投影、采样网格与内参；既有固定来源。本地 geometry 已可作独立数值参照，不能忽略 center/align_corners 差异 |
| R08 | [OpenVINS nullspace](https://docs.openvins.com/update-null.html)、[update](https://docs.openvins.com/update.html)、[实现入口](https://github.com/rpng/open_vins) | 本轮读 nullspace 原理；用来写局部推导和反例，主线仍先 marginalization；GPL 系统不直接复制整套 |
| R09 | [Online Self-Calibration / degeneracy](https://arxiv.org/abs/2201.09170)、[Online Temporal Calibration](https://arxiv.org/abs/1808.00692) | 标定与时间可观测性、退化；方案论文来源，S0/S4 使用。N04 没有 IMU 时不凭空添加惯性观测 |
| R10 | [DSO 论文](https://arxiv.org/abs/1607.02565)、[AffLight / NumType.h](https://github.com/JakobEngel/dso/blob/7b0c99f01d238f801c625beaff90240bcb007198/src/util/NumType.h)、[Residuals.cpp](https://github.com/JakobEngel/dso/blob/7b0c99f01d238f801c625beaff90240bcb007198/src/FullSystem/Residuals.cpp) | 曝光/仿射光度解释的既有固定来源；GPL，只作原理/独立参考，不能把 RAW gain 直接乘非线性 feature |
| R11 | [Foi 原始传感器噪声资料](https://webpages.tuni.fi/foi/sensornoise.html) | Poisson–Gaussian、信号相关方差和裁剪；公式应用在传感器单位，软件授权另核；本轮未运行标定软件 |
| R12 | [本地 bilinear_variance](../../../src/owac/evaluation/n04_raw_diagnostics.py)、[对应测试](../../../tests/encoders/n04/test_raw_noise_diagnostics.py) | 本轮实读/测试；独立输入的插值方差平方权重、gain²。不能当作完整递归状态协方差 |
| R13 | [FilterPy KF](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/kalman_filter.py)、[IMM.py](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/IMM.py) | 既有固定 MIT 来源；Joseph、模式 interaction/mixing 和矩匹配的数值参照。隔离实验环境复现，不恢复当前共享 stash |
| R14 | [IEKF 局部理论](https://arxiv.org/abs/1410.1465)、[Bayesian online changepoint detection](https://arxiv.org/abs/0710.3742) | 稳定性条件与变化检测已有边界；论文条件不能直接转移到任意神经转移 |

## 条件运动与最近对照

| 编号 | 来源 | 在本项目中做什么 |
| --- | --- | --- |
| R15 | [RAFT core/raft.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/raft.py)、[update.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/update.py)、[utils.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/utils/utils.py)、[论文](https://arxiv.org/abs/2003.12039) | 既有固定版本，utils 网页本轮确认。完整对应网络仅 S5 诊断门通过后接入；检查 padding、flow像素单位、upsampling、align_corners=True 与本项目转换；BSD-3-Clause |
| R16 | [Dynamo-Depth 论文](https://arxiv.org/abs/2310.18887)、[Trainer.py](https://github.com/YihongSun/Dynamo-Depth/blob/227a5d938ddf9a3aba5fcf7a03c74384284b31d0/Trainer.py)、[tools.py](https://github.com/YihongSun/Dynamo-Depth/blob/227a5d938ddf9a3aba5fcf7a03c74384284b31d0/tools.py) | 固定源码 Trainer 页本轮打开；完整 depth/ego/motion 分阶段训练。顶层 MIT 与继承组件分别核许可，不能只保留两层小网络仍称复现 |
| R17 | [Monodepth2](https://github.com/nianticlabs/monodepth2)、[论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html) | 条件深度参考；检查 scale、projection、automasking。单目没有真实尺度时不在测试拿 GT 标尺；动态物体不能被默认静态mask全部删掉 |
| R18 | [MCDS-VSS 论文](https://arxiv.org/abs/2405.19921)、[作者仓库](https://github.com/angelvillar96/MCDS-VSS) | 本轮确认 README 要求联系作者获取代码；S6 只能明确标注论文结构复现，不能声称完整官方代码可用。ego+残余流+融合的顺序已有先例 |
| R19 | [SE3-Nets](https://arxiv.org/abs/1606.02378)、[SE3-Pose-Nets](https://arxiv.org/abs/1710.00489)、[SfM-Net](https://arxiv.org/abs/1704.07804) | 动作/相机/对象结构化运动的论文边界；它们的深度或点云输入不等于 N04 当前已知输入 |
| R20 | [HiP-RSSM](https://github.com/ALRhub/HiP-RSSM)、[MTS3](https://github.com/ALRhub/MTS3) | 变化动力学和时间尺度瓶颈证实后才增设对照；文件级许可证未完整确认，默认不复制代码，不因公开可读推定许可 |
| R21 | [EMVD 论文](https://arxiv.org/abs/2103.05407)、[IPOL 后续实现](https://www.ipol.im/pub/pre/464/) | RAW 时序融合和噪声处理参照；IPOL仍按预印本/源码授权待核，本地已有下载审计但本轮未训练，不作为必须依赖 |

## π0.5 真实接口

| 编号 | 固定实现 | 具体函数与作用 |
| --- | --- | --- |
| R22 | [PI0.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py)、[PI05.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py) | `embed_prefix/forward_from_observation/sample_actions` 的真实训练/采样；PI0 固定源码页及远端选段本轮实读，当前本地 bridge 实读。不要虚构 `backend.forward_from_features` |
| R23 | [OpenPIPaliGemma.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/vlm/OpenPIPaliGemma.py)、[OpenPI_ActionHead.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py) | 语言 embedding scale、前缀mask、action suffix/adaRMS、flow targets/velocity decoder，参数冻结但梯度仍回传 |
| R24 | [openpi 固定 PyTorch 模型](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py)、[π0.5 论文](https://arxiv.org/abs/2504.16054) | 官方架构交叉参照；执行后端保持已部署 StarVLA，不自动切换最新官方实现 |
| R25 | [本地真实 bridge](../../../src/owac/backends/rawvla_features.py)、[smoke](../../../src/owac/backends/rawvla_feature_smoke.py)、[service](../../../src/owac/backends/rawvla_feature_server.py)、[旧真实接入实验](../../../research/experiments/2026-10-12_n04_acceptance.md) | 本轮实读；现有tiny smoke硬编码旧encoder，新full checkpoint需显式适配；先真实权重梯度、再充分训练、最后完整rollout |

已有固定文件的完整 SHA256 清单在 [2026-10-12 来源清单](../reference_sources.json) 和 [2026-10-13 既有方法审计](../../../research/reviews/2026-10-13_n04_existing_methods/sources.json)。本轮网络下载失败不重写这些历史证据；新集成版本的来源/许可证/源码 hash 在实际开发时补入独立 manifest。

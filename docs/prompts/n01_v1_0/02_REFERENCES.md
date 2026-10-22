# N01 v1.0 论文与参考实现导航

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

2026-10-14。先读与当阶段对应的实现，再按下列边界移植。精确提交和取回文件 SHA256 见 [source_manifest.json](source_manifest.json)。本次固定了九个上游仓库的提交并取回主要源码；没有安装或运行这些新模型。原始 raw.githubusercontent 下载发生连接重置，随后通过 GitHub contents API 取回；失败路径也保留在来源记录中。

## 阶段与代码的对应关系

| 阶段 | 首先阅读 | 具体复用边界 |
| --- | --- | --- |
| P0/P4 | DBSR encoders、merging；本地 burst_features | 完整64/9/512内容支路和MaskedDbsrFusion，保持来源归属 |
| P1 | 本地source ledger/GLS参考；torch.linalg；iSAM2资料 | 精确来源集合、相关协方差子块、完整重建，不要求把GTSAM搬上GPU |
| P2 | SEA-RAFT raft/extractor/corr/update/train | 4/8或3/6通道，去除/255假设与自动ImageNet加载；保留作者迭代运动模块 |
| P3 | DSM-bocd bocpd/models；confseq | 前者是变点模型，后者提供条件适用的顺序边界；两者保证不互换 |
| P5 | Mamba mamba_simple/generation；完整S4 | 两种cache一同重放；固定系数解析支路才有当前T4证明 |
| P6 | BSRT bsrt；CRVD/ReCRVD README | RAW合法特征子网和真实数据身份；不沿用RGB/ISP目标 |
| P7/P8 | 固定RawVLA PI0/PI05/NPZ、OWAC桥与客户端 | 视觉前缀、动作归一化、真实RAW请求；不换最新版后端 |

## DBSR 已在仓库复用的固定实现

- [完整编码器](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/encoders.py)。
- [WeightedSum融合](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/merging.py)。
- [残差块](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/layers/blocks.py)。
- [原坐标warp参考](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/layers/warp.py)。

本地 [provenance](../../../src/owac/baselines/_vendor/dbsr/provenance.json) 与 [adapter](../../../src/owac/baselines/burst_features.py) 优先作为可运行参考。原 RGB/PWC 对齐和图像 decoder 不进入主链。

## 本次固定并取回的官方文件

### princeton-vl/SEA-RAFT

固定提交：`9137517ba24e628442aec097d3afe71d03503b75`。P2：重点检查 ResNetFPN._init_weights 的自动权重加载；原 3/6 通道与 /255 需要一起改。

- [core/raft.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/raft.py)。
- [core/extractor.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/extractor.py)。
- [core/update.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/update.py)。
- [core/corr.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/corr.py)。
- [train.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/train.py)。
- [LICENSE](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/LICENSE)。

### maltamiranomontero/DSM-bocd

固定提交：`9d05117bc8c6e971c836ee1df0c49e2d2950862b`。P3：GaussianUnknownMean、DSMGaussianUnknownMean、bocpd 是数值参考；作者依赖包括旧版 NumPy/SciPy/JAX，移植后与原前缀输出逐项比较。

- [bocpd.py](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/bocpd.py)。
- [models.py](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/models.py)。
- [README.md](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/README.md)。
- [LICENSE](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/LICENSE)。

### state-spaces/mamba

固定提交：`e9594ce1c732d97440f0332fdc43170a2294dbfa`。P5：Mamba.step 同时更新 conv_state 和 ssm_state；选作者完整配置并验收实际GPU兼容。

- [mamba_ssm/modules/mamba_simple.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py)。
- [mamba_ssm/utils/generation.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/utils/generation.py)。
- [LICENSE](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/LICENSE)。

### Algolzw/BSRT

固定提交：`da8d5154478a1214bccdf7a54c716dad3a015985`。P6：make_model 提供作者 S/L 规模；构造器和 forward 中的 SpyNet、debayer、RGB mean、checkpoint 路径均要审查，不能只移除最终输出头。

- [code/synthetic/bsrt/model/bsrt.py](https://github.com/Algolzw/BSRT/blob/da8d5154478a1214bccdf7a54c716dad3a015985/code/synthetic/bsrt/model/bsrt.py)。
- [README.md](https://github.com/Algolzw/BSRT/blob/da8d5154478a1214bccdf7a54c716dad3a015985/README.md)。
- [LICENSE](https://github.com/Algolzw/BSRT/blob/da8d5154478a1214bccdf7a54c716dad3a015985/LICENSE)。

### gostevehoward/confseq

固定提交：`5ffe733ca2447a2e28c2c91f3b00086173f2ab2c`。P1/P3：Python boundaries 是编译扩展，实际入口为 boundaries.cpp 和 uniform_boundaries.h，不存在本次最初猜测的 boundaries.py。

- [README.md](https://github.com/gostevehoward/confseq/blob/5ffe733ca2447a2e28c2c91f3b00086173f2ab2c/README.md)。
- [src/confseq/boundaries.cpp](https://github.com/gostevehoward/confseq/blob/5ffe733ca2447a2e28c2c91f3b00086173f2ab2c/src/confseq/boundaries.cpp)。
- [src/confseq/uniform_boundaries.h](https://github.com/gostevehoward/confseq/blob/5ffe733ca2447a2e28c2c91f3b00086173f2ab2c/src/confseq/uniform_boundaries.h)。

### state-spaces/s4

固定提交：`e757cef57d89e448c413de7325ed5601aceaac13`。P5：完整模型位于 models/s4/s4.py；s4d.py 是教学精简实现，不能作为正式强基线。

- [README.md](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/README.md)。
- [models/s4/s4.py](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4.py)。

### cao-cong/RViDeNet

固定提交：`4615032b010e70eebe038be46a5d5820d43e2ef4`。P6：使用官方真实RAW来源、相位/黑白电平和分区；作者的RGB/ISP、逆处理合成训练不是本项目主链。

- [README.md](https://github.com/cao-cong/RViDeNet/blob/4615032b010e70eebe038be46a5d5820d43e2ef4/README.md)。

### cao-cong/RViDeformer

固定提交：`4b3f2744f5d19707e718e735c12f2e531fa64dda`。P6：ReCRVD 是显示屏内容重拍的传感器数据，保留来源范围与90/30场景分区，不自动声称自然动态或跨传感器。

- [README.md](https://github.com/cao-cong/RViDeformer/blob/4b3f2744f5d19707e718e735c12f2e531fa64dda/README.md)。

### google-deepmind/deepmind-research

固定提交：`f5de0ede8430809180254ee957abf36ed62579ef`。后期可选：Perceiver是JAX/Haiku实现；CrossAttention、make_cross_attention_mask、BasicDecoder可参考。默认仍用已有VisualTokenPostprocessor，不额外引入未验收query网络。

- [perceiver/perceiver.py](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/perceiver/perceiver.py)。

## 固定 RawVLA 与策略实现

下列链接沿用已有 [reference_sources.json](../reference_sources.json) 的锁定版本；本次重新核对本地桥与服务器路径，未把它们升级为浮动HEAD。P7还需按实际安装源码核验哈希。

- [ShuhongLL/RawVLA README.md](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/README.md)。
- [ShuhongLL/RawVLA starVLA/dataloader/rawvla_npz_datasets.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py)。
- [ShuhongLL/RawVLA starVLA/model/framework/VLM4A/PI05.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py)。
- [ShuhongLL/RawVLA starVLA/model/framework/VLM4A/PI0.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py)。
- [ShuhongLL/RawVLA starVLA/model/modules/vlm/OpenPIPaliGemma.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/vlm/OpenPIPaliGemma.py)。
- [ShuhongLL/RawVLA starVLA/model/modules/action_model/OpenPI_ActionHead.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py)。
- [ShuhongLL/RawVLA configs/libero/pi05.yaml](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/libero/pi05.yaml)。
- [ShuhongLL/RawVLA configs/robotwin/pi05.yaml](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/robotwin/pi05.yaml)。
- [ShuhongLL/RawVLA experiments/simulation/libero/rawvla_bench/raw.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/libero/rawvla_bench/raw.py)。
- [ShuhongLL/RawVLA baselines/rawvla/rawvla.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/baselines/rawvla/rawvla.py)。
- [Physical-Intelligence/openpi src/openpi/models_pytorch/pi0_pytorch.py](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py)。

本地重点：[真实loader](../../../src/owac/raw/rawvla_sequence.py)、[特征桥](../../../src/owac/backends/rawvla_features.py)、[旧动作工具](../../../src/owac/training/n01_action.py)、[冻结评测](../../../src/owac/runtime/rawvla_full_eval.py)、[RoboTwin客户端](../../../src/owac/runtime/rawvla_robotwin.py)。

## 原方案的25项研究来源

以下为原方案已列论文/作者资料的导航；本轮新增核查集中在上面的代码文件。它们是阅读材料和创新边界，不都要求成为运行依赖。

### R01

Bhat et al. *Deep Burst Super-Resolution*. CVPR 2021。[论文](https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.html)；[已固定官方代码](https://github.com/goutamgmb/deep-burst-sr/tree/16f158e0553b6702cc1693c43f733c2830973278)。当前成熟 RAW 内容基线。

### R02

Luo et al. *BSRT: Improving Burst Super-Resolution With Swin Transformer and Flow-Guided Deformable Alignment*. CVPR Workshops 2022。[论文](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Luo_BSRT_Improving_Burst_Super-Resolution_With_Swin_Transformer_and_Flow-Guided_Deformable_CVPRW_2022_paper.html)；[官方代码](https://github.com/Algolzw/BSRT)。第二成熟骨干候选，须拆除不合规的图像链。

### R03

Wronski et al. *Handheld Multi-Frame Super-Resolution*. ACM TOG 2019。[作者机构页面与论文](https://research.google/pubs/handheld-multi-frame-super-resolution/)。CFA 跨帧采样、结构与鲁棒融合依据；不声称本项目已有其官方可运行代码。

### R04

Foi et al. *Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data*. IEEE TIP 2008。[作者论文/软件/数据页面](https://webpages.tuni.fi/foi/sensornoise.html)。传感器噪声模型与校准；软件有单独限制许可，不默认 vendoring。

### R05

Mildenhall et al. *Burst Denoising with Kernel Prediction Networks*. CVPR 2018。[论文与作者页面](https://bmild.github.io/kpn/)；[作者代码](https://github.com/google/burst-denoising)。来源内容利用与优化退火参考。

### R06

Teed and Deng. *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*. ECCV 2020。[论文](https://arxiv.org/abs/2003.12039)；[官方代码](https://github.com/princeton-vl/RAFT)。成熟对应模块背景。

### R07

Wang, Lipson and Deng. *SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow*. ECCV 2024。[论文](https://arxiv.org/abs/2405.14793)；[官方代码](https://github.com/princeton-vl/SEA-RAFT)。优先 RAW 适配候选；原输入为 RGB 风格接口。

### R08

Yue et al. *Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes*. CVPR 2020。[官方代码与 CRVD 数据说明](https://github.com/cao-cong/RViDeNet)。真实 RAW 视频诊断，数据与动作任务分开。

### R09

Cao et al. *RViDeformer: Efficient Raw Video Denoising Transformer with a Larger Benchmark Dataset*. arXiv 2023，后续期刊版本见作者仓库。[论文](https://arxiv.org/abs/2305.00767)；[官方代码与 ReCRVD](https://github.com/cao-cong/RViDeformer)。较复杂 RAW 数据与邻近架构，原 RGB/ISP 监督不能照搬。

### R10

Adams and MacKay. *Bayesian Online Changepoint Detection*. 2007。[论文](https://arxiv.org/abs/0710.3742)。时间支持分段的标准解析对照；不把社区实现冒称作者官方库。

### R11

Altamirano, Briol and Knoblauch. *Robust and Scalable Bayesian Online Changepoint Detection*. ICML 2023。[论文](https://arxiv.org/abs/2302.04759)；[作者页面](https://jeremiasknoblauch.github.io/publication/rccp/)；[官方代码 DSM-bocd](https://github.com/maltamiranomontero/DSM-bocd)。异常点与变化的鲁棒区分；`bocpd.py` / `models.py` 是参考入口。

### R12

Howard et al. *Time-uniform, Nonparametric, Nonasymptotic Confidence Sequences*. Annals of Statistics 2021。[论文](https://arxiv.org/abs/1810.08240)；[作者实现 confseq](https://github.com/gostevehoward/confseq)。有条件的长期错误控制和停止规则，不直接赋予神经置信度保证。

### R13

Kaess et al. *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. IJRR 2012。[作者页面与论文](https://people.csail.mit.edu/kaess/pub/Kaess12ijrr.html)；[GTSAM](https://github.com/borglab/gtsam)；[ISAM2 文档](https://gtsam.org/doxygen/4.0.0/a03643.html)。因子、依赖、删除与重线性化参考，不作为实时 GPU 主线的既定实现。

### R14

Felzenszwalb and Huttenlocher. *Efficient Graph-Based Image Segmentation*. IJCV 2004。[作者论文与 C++ 实现](https://cs.brown.edu/people/pfelzens/segment/)。空间区域组织对照，非时空证据有效性的保证。

### R15

Gu, Goel and Ré. *Efficiently Modeling Long Sequences with Structured State Spaces*. ICLR 2022。[论文](https://arxiv.org/abs/2111.00396)；[官方 S4](https://github.com/state-spaces/s4)。多时间尺度状态模型；教学精简代码不充当强性能基线。

### R16

Gu and Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 2023 预印本 / COLM 2024。[论文](https://arxiv.org/abs/2312.00752)；[官方实现](https://github.com/state-spaces/mamba)。可选择保留/遗忘的成熟神经模块，完整网络保证需另证。

### R17

Zhu et al. *Trajectory-aware Shifted State Space Models for Online Video Super-Resolution*. ICLR 2026，仓库所列论文版本 arXiv:2508.10453。[论文](https://arxiv.org/abs/2508.10453)；[官方 TS-Mamba](https://github.com/QZ1-boy/TS-Mamba)。运动轨迹与状态结合的直接近邻，尚未做 RAW/因果源码验收。

### R18

Di et al. *QMambaBSR: Burst Image Super-Resolution with Query State Space Model*. CVPR 2025。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Di_QMambaBSR_Burst_Image_Super-Resolution_with_Query_State_Space_Model_CVPR_2025_paper.html)。查询状态空间 burst 融合的创新边界；本轮未验收可运行模型代码。

### R19

Unal, Marty and Dai. *Keyframe-Centric State-Space Modeling for Burst Image Super-Resolution*，作者项目页标注 SIGGRAPH Asia 2026。[作者页面](https://ouenal.github.io/burstmamba/)；[相关预印本 Burst Image Super-Resolution with Mamba](https://arxiv.org/abs/2503.19634)。近期思路跟踪；会议日期晚于本文日期，按公开项目资料引用，不声称已完成会议发表或代码验收。

### R20

Revay, Wang and Manchester. *Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness*. IEEE TAC 2023。[论文](https://arxiv.org/abs/2104.05942)；[作者 REN 仓库](https://github.com/imanchester/REN)。严格稳定模型参考；作者实现为 Julia。

### R21

Jaegle et al. *Perceiver IO: A General Architecture for Structured Inputs & Outputs*. ICLR 2022。[论文](https://arxiv.org/abs/2107.14795)；[官方实现](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)。条件化查询读出；跨框架移植需单独验收。

### R22

Lehtinen et al. *Noise2Noise: Learning Image Restoration without Clean Data*. ICML 2018。[论文](https://proceedings.mlr.press/v80/lehtinen18a.html)。满足条件的独立噪声监督。

### R23

Batson and Royer. *Noise2Self: Blind Denoising by Self-Supervision*. ICML 2019。[论文](https://proceedings.mlr.press/v97/batson19a.html)。自监督的独立性与不变性要求，不能忽略对齐/共享来源泄漏。

### R24

Liu et al. *RawVLA: Embodied Neural Image Signal Processor for Robotic Manipulation*. arXiv 2026。[论文](https://arxiv.org/abs/2609.37530)；[作者项目/代码/数据入口](https://shuhongll.github.io/rawvla/)。直接具身对照和 RawVLA-Bench；实际输入身份以本地固定数据审计为准。

### R25

Physical Intelligence et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization*. 2025。[论文](https://arxiv.org/abs/2504.16054)；[官方 openpi](https://github.com/Physical-Intelligence/openpi)。最终后端；本项目执行版本沿用已冻结 StarVLA/RawVLA 适配，不切换到浮动最新版本。

## 数值接口

- [torch.linalg.cholesky](https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky.html)：固定正定协方差的分解。
- [torch.linalg.lstsq](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html)：参考最小二乘；按实际PyTorch版本确认driver/device支持。
- [GTSAM ISAM2](https://gtsam.org/doxygen/4.0.0/a03643.html)：因子删除与更新概念参照。

源码内容缓存位于 ignored `outputs/n01-v1-prompt-preparation/upstream/`；这不是安装目录，迁移工作区时按上游锁重新获取。模型、权重和数据许可分别登记，源码下载成功不代表训练复现。

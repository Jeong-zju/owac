## 参考实现导航

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

先读与你正在实现的部分有关的作者源码，再移植/复用，保留许可证和适配说明。这里固定代码版本是为了找到同一实现，不是要求构建一套哈希验收系统。采用新权重时另记权重来源和训练域。论文整网常含 RGB 重建、未来帧或 RGB 初始化，不能直接把它接到主链。

### P3 首选直接复用

**DBSR 内容与融合。** [论文](https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.html)；固定 `16f158e0553b6702cc1693c43f733c2830973278`。读作者 [encoders.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/encoders.py)、[merging.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/dbsr/merging.py)、[blocks.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/layers/blocks.py)、[warp.py](https://github.com/goutamgmb/deep-burst-sr/blob/16f158e0553b6702cc1693c43f733c2830973278/models/layers/warp.py)。本仓已 vendor，优先复用 `DbsrFeatureAdapter` 和 `MaskedDbsrFusion`，默认宽度64、9个残差块、输出512；不复制作者原始 RAW→RGB 对齐分支和 RGB 重建头。许可证 CC-BY-NC-SA-4.0，保留作者声明。

**SEA-RAFT 对应。** [论文](https://arxiv.org/html/2405.14793v1)；固定 `9137517ba24e628442aec097d3afe71d03503b75`。作者 [raft.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/raft.py)、[extractor.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/extractor.py)、[corr.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/corr.py)、[update.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/core/update.py)、[train.py](https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/train.py)。本地 `backends/raw_correspondence.py` 已做三通道/四相位适配，关闭自动 ImageNet 加载和 `/255`；保留完整作者主体。proxy 风险需目标数据校准，不能因用了概率输出就宣布有频率学覆盖保证。

**时间变化与鲁棒更新。** [BOCPD 原论文](https://arxiv.org/abs/0710.3742)、[鲁棒 BOCPD 论文](https://proceedings.mlr.press/v202/altamirano23a.html)；DSM-bocd 固定 `9d05117bc8c6e971c836ee1df0c49e2d2950862b` 的 [bocpd.py](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/bocpd.py)、[models.py](https://github.com/maltamiranomontero/DSM-bocd/blob/9d05117bc8c6e971c836ee1df0c49e2d2950862b/models.py)。沿对应轨迹处理创新，不在固定像素上把物体正常移动当结构突变。移植核心数值更新即可，不为旧 NumPy/JAX 依赖重建巨大运行栈；用短真实前缀与作者算法核对。

**顺序统计的理论参考。** [confidence sequences 论文](https://arxiv.org/abs/1810.08240)；[confseq](https://github.com/gostevehoward/confseq/tree/5ffe733ca2447a2e28c2c91f3b00086173f2ab2c) 的 `src/confseq/boundaries.cpp`、`uniform_boundaries.h`。不存在可直接照用的 `boundaries.py`。仅当残差满足相应可预测/条件矩假设时使用；不强制编译此库才能开始完整模型训练。

**π0.5 与 RawVLA。** [RawVLA 论文](https://arxiv.org/html/2609.37530v1)、[作者项目](https://shuhongll.github.io/rawvla/)、[π0.5 论文](https://arxiv.org/abs/2504.16054)。固定 RawVLA 源码 `57d508cb180d0510ac9060358009c0f6ffaac420`：

- [NPZ 数据加载](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py)：只查数据字段，作者 PIL/RGB 输出不能代替本项目 RAW loader。
- [PI0.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py)、[PI05.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py)：视觉前缀、训练和推理接点。
- [OpenPIPaliGemma.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/vlm/OpenPIPaliGemma.py)、[OpenPI_ActionHead.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py)：原生 action flow matching，不另写弱化 action MLP。
- 固定配置版本 `ba9c56b04da2c4931b70d8844710f5928d43775b` 的 [LIBERO pi05.yaml](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/libero/pi05.yaml)、[RoboTwin pi05.yaml](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/robotwin/pi05.yaml)、[LIBERO raw.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/libero/rawvla_bench/raw.py)、[rawvla.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/baselines/rawvla/rawvla.py)：对照冻结控制协议。
- openpi 固定 `15a9616a00943ada6c20a0f158e3adb39df2ccac` 的 [pi0_pytorch.py](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py) 用于理解损失；执行优先现有 StarVLA 桥，不升级后端。

### P5 或确有证据需要的替换

**BSRT。** [论文](https://arxiv.org/abs/2204.08332)；固定 `da8d5154478a1214bccdf7a54c716dad3a015985` 的 [bsrt.py](https://github.com/Algolzw/BSRT/blob/da8d5154478a1214bccdf7a54c716dad3a015985/code/synthetic/bsrt/model/bsrt.py)。作者 L 规模参考 `embed_dim=180`、`depths=[6,8,8,8,8,8,8]`。原实现含 Debayer3x3、SpyNet RGB 归一化/权重下载和作者本机路径，必须拆除/替换这些依赖，保持成熟特征主体；不是原论文整网复现。

**Burstormer。** [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.html)、[作者代码](https://github.com/akshaydudhane16/Burstormer)，从 `Burst Super-resolution/Network.py` 读特征/对齐/融合。主分支代码采用前记录真正 commit。其通道 attention 不能直接画成像素显著性图；不要把之前记录的 Git tree SHA 冒充 commit。

**真实 RAW 视频。** RViDeformer [论文](https://arxiv.org/abs/2305.00767)与固定 `4b3f2744f5d19707e718e735c12f2e531fa64dda` 的 [网络](https://github.com/cao-cong/RViDeformer/blob/4b3f2744f5d19707e718e735c12f2e531fa64dda/models/RViDeformer_arch.py)、[训练](https://github.com/cao-cong/RViDeformer/blob/4b3f2744f5d19707e718e735c12f2e531fa64dda/train_ReCRVD_supervised.py)、[数据说明](https://github.com/cao-cong/RViDeformer/blob/4b3f2744f5d19707e718e735c12f2e531fa64dda/README.md)。ReCRVD 是显示屏重拍数据，官方 90/30 场景划分及 ISO 范围按数据卡核对，不声称自然机器人闭环。另见 [RViDeNet/CRVD](https://github.com/cao-cong/RViDeNet/tree/4615032b010e70eebe038be46a5d5820d43e2ef4)、[AIM 2025 多传感器 RAW 视频](https://arxiv.org/abs/2508.16830)。先确认数据可用、许可和存储，再下载；许可证未核实不复制发布作者代码。

**有需要才添加神经记忆。** Mamba [论文](https://arxiv.org/abs/2312.00752)，固定 `e9594ce1c732d97440f0332fdc43170a2294dbfa` 的 [mamba_simple.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py)、[generation.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/utils/generation.py)。卷积缓存和 SSM 缓存都要处理删除，不保留混入旧源的 hidden state。P3 默认有界精确回放，不强行加入 Mamba。TS-Mamba 的 [作者仓库](https://github.com/QZ1-boy/TS-Mamba)可作近邻；之前 README 所指 `options/train/TSA/train_TSA.yaml` 为404，不能承诺直接运行。

**跨任务读出与视觉解释。** [Perceiver IO](https://arxiv.org/abs/2107.14795)、[作者实现](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/perceiver)可作独立读出参考，不要求为小 probe 搬整个 JAX 栈；[Attention is not Explanation](https://aclanthology.org/N19-1357/)、[Attention is not not Explanation](https://aclanthology.org/D19-1002/)提示做真实来源干预，两文是 NLP 语境，不能直接当作本视觉模型的定理。

完整37项论文与相邻工作继续见正式 v1.2 方案第11节；它们不是33个都必须安装的依赖。优先完成本阶段的完整任务，按具体缺口阅读，避免把文献清单变成新串行关卡。

### v1.2 必须核对的同源成像与信息边界

[Unprocessing Images for Learned Raw Denoising](https://arxiv.org/abs/1811.11127)、[作者代码](https://github.com/timothybrooks/unprocessing)用于理解成像与噪声域，不把unprocessing当恢复真实丢失信息的工具。[Learning to See in the Dark](https://arxiv.org/abs/1805.01934)、[作者代码](https://github.com/cchen156/Learning-to-See-in-the-Dark)提供真实短曝光RAW的背景；其RGB恢复目标不进入主链。[RAW-Adapter](https://arxiv.org/abs/2408.14802)、[作者代码](https://github.com/cuiziteng/ECCV_RAW_Adapter)与[Dr. RAW](https://papers.neurips.cc/paper_files/paper/2025/hash/20eadb6338a53114a6371f102c5ded95-Abstract-Conference.html)界定RAW高层任务与适配的已有边界。

固定RawVLA的 `experiments/simulation/libero/rawvla_bench/raw.py` 和RoboTwin对应Default链是拟复用的ISP定位入口，先检索当前冻结作者源码中的 `default_isp`，读取真实调用、dtype、resize和量化顺序再封装，不能凭本文假造已存在的函数签名。作者配对RGB可能为正常光参考，不能直接当受损RAW的RGB分支。

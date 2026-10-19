# DeepSeek N01 v1.1 P3 完整编码器 真实训练与开发闭环

> **历史提示词：当前执行请使用 [N01 v1.2完整包](../n01_v1_2/README.md)。本文件不再作为后续启动依据。**

本次仅执行 **N01 v1.1 P3**。这是一份完整交接包，共同要求、参考实现、远程训练、前端专家任务和阶段指令均已展开；请从头读完再执行，不需要用户手动拼接其他提示词。准备本包时P3尚未实施，后续阶段的前置交付必须以届时实际记录为准。

## 本次执行的共同要求

你是负责实际编程、训练和实验交付的 DeepSeek。请执行用户选择的一个 v1.1 阶段，持续完成该阶段已授权的工作；只写计划、空壳接口、模拟日志或 smoke 不能算交付。阶段正文在本交接包后半部分。完成后停在该阶段，等待独立验收，不自行启动下一阶段。

### 工作位置和有效依据

- 本地 checkout：`/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`。远程执行时用该 checkout 的当前源码快照；远程没有这个本地绝对路径是正常的。
- 先读 checkout 的 `AGENTS.md`、`README.md`、`docs/status.md`，检查 `git status --short --branch`，保留当前未提交修改。不要新建另一个原型仓库，不修改 N02/N04，不重置用户的文件。
- 正式路线：`docs/N01_full_development_plan_2026-10-15_v1.1.md`，决议：`docs/decisions/0014-n01-v1-1-full-model-route.md`。P3 是完整编码器、真实数据充分训练和开发闭环；P4 是冻结全量验证；P5 是有选择的迁移扩展。旧 v1.0 P3—P8 已失效，不机械沿用旧阶段门槛。
- 继承 P0/P1 已验收资产和 P2 已修补代码。读 `research/experiments/n01_full/P1_acceptance_report.md`、`P2_acceptance_report.md`。P2 修补提交 `fe8d534`、验收记录 `d3dec90`；用户已决定继续推进，原独立复核“未完全通过”的限制保留，**不要求先补齐旧 P2 的合成训练停止条件才能做 P3**。
- P2 的运动训练主要来自程序生成场，不是 RawVLA-Bench 场景训练；其大运动与风险校准反例尚未消除。把已有模型当初始化/对照，在实际完整任务中诊断和改进，不直接宣布 motion 可靠。

### 用户明确要求的工作方式

**2026-10-15最新补充：当前项目初期只训练1个种子，默认seed 101。** 两平台N/B0/B1每个平台、每个方法各1次主体训练；P5的适配/扩展也默认单种子。不要自动加入211/307，不以少于三种子判未完成；三种子只保留为后续可选验证，须用户另行安排。此要求覆盖正式方案/历史材料中已被替代的三种子条款。单种子比较仍保持方法间数据、初始化规则和训练机会公平，结论不外推训练随机种子的稳定性。

把精力用于完整模型、真实数据、训练、可视化和行动结果。实现普通必要的错误提示与恢复；不要把“防御性工程”发展成另一项研究。

1. **不新增无依据的硬性通过线。** 不规定“提高 5% 才能下一步”“损失必须下降到某个自创值”“连续八个验证窗口平台”“两次降 LR 才算收敛”，不要求每个参数都有非零梯度，不以测试个数评定研究成功。
2. **不建立严苛哈希准入系统。** run 记源码版本、工作区补丁、有效配置、数据清单、checkpoint 和恢复位置即可。不要对每个 Python AST、每个 tensor、每次 batch、每次页面轮询做哈希；不为补写历史 hash 停止训练。权重传输损坏排查可以做一次普通校验。既有冻结清单的身份记录继续使用，不重新设计复杂签名体系。
3. **不自创总分。** 主结果是机器人成功率、事件后恢复、真实端到端成本；训练 loss、flow 误差、融合权重、支持年龄用于定位原因。禁止把它们加权成“证据健康度”“机制合格分”等无来源指标。
4. **保留有实质意义的正确性检查。** 不跨 train/test 泄漏；不读未来帧/专家未来动作；训练确实更新目标参数；来源完全删除后不能通过 flow、参考帧、归一化或缓存继续影响；恢复不能覆盖旧结果；正式评测保留原分母。这些直接关系结论，不属于可删的装饰门槛。
5. 小网络仅可用于短小工程调试，不作为正式模型或所谓“同名简化复现”。默认复用完整 DBSR、完整 SEA-RAFT 和已接通 π0.5；新增模块说明论文、作者代码、适配处及实际规模。不要照名字手搓 tiny Transformer/Mamba/RAFT。
6. 网络/算法调整后回到固定真实 RawVLA-Bench 案例，输出原画面上的热图和完整动作/开发闭环影响。简单几何信号可以查数值错误，不能成为长期主实验。
7. 常规可逆实现、诊断、已规划训练和恢复可直接推进。资源超过本阶段登记总预算、需要覆盖旧资产、需要新的付费资源或实物执行时，说明具体缺项再请求用户决定。普通负结果、未获正收益、没有“收敛 PASS”不构成停止编程的理由。

### RAW 输入与后端边界

实际主体是作者的三通道 HDR-derived / pseudo-RAW，代码入口称 `rawvla_rgb3`；它不是原生 Bayer CFA。LIBERO 为 agent/wrist 两相机、256×256 三通道整数缓存，RoboTwin 为 head/left/right 三相机、240×320 三通道浮点缓存；以 loader 的真实字段和相机顺序为准。不要因扩展名/变量名含 raw 就宣布传感器原始性。

主链：`RAW 测量 → 成熟内容编码 + 来源支持/修订 → 非图像特征 → 视觉前缀桥 → π0.5 语言/动作后端`。主方法不经过 RGB/sRGB/XYZ 恢复图，不用 RGB 教师蒸馏或 RGB 重建补回原视觉塔能力。可视化预览和独立 RGB 强基线可使用 RGB，分别保存。

继承四区清单 `data/manifests/n01-full-v1/n01-full-v1-index.json`：LIBERO train/calibration/validation/test 为 1228/175/206/162；RoboTwin 为 450/48/58/76，共 2403 条源轨迹。同一源轨迹的所有照明/噪声变体、相机和重叠窗口属于同一区；不重新随机按帧划分。数据源轨迹数和闭环 rollout 数不是一个分母。任务模板跨区共享，因此不声称未见任务泛化。

动作归一化沿用已审计协议：RoboTwin `MEAN_STD` 与 14 维原顺序；LIBERO 使用自己的 7 维统计。缓存 joint target 不能冒称实测 proprioception。采集帧到达节奏、后端推理节奏、action chunk 执行节奏分别记录，不能只修改推理频率就声称历史更多。正确动作会随物理/目标改变的干预需要新标签或重新规划，不能沿用旧专家动作。

### 持续执行和交接

建立小型 `RUN_STATE.md`，记录当前活动、正在运行的 supervisor program、输出目录、恢复命令、已完成矩阵项和下一步。每完成一个有意义的动作或出现错误就更新；长训练由独立进程持续写进度，不依赖对话一直在线。上下文压缩或 SSH 重连后先读它、查现有进程和 checkpoint，再恢复，不能重复开同一训练。

小型报告建议放 `research/experiments/n01_v1_1/P3_report.md`、`P4_report.md`、`P5_report.md`，小型汇总放相应 `research/experiments/results/n01_v1_1/<stage>/`；这些是本轮拟新增位置。大权重、逐帧数组、视频和原始日志留服务器/ignored 输出。修改后运行 `make check` 和实际受影响的测试；不要因改一个网页反复跑全库 GPU 矩阵。既有无关检查失败要记录归属，不能顺手改别人工作。

结尾先给同步案例和任务结果，再给训练覆盖、机制解释、局限、复现命令及原始文件位置。分别说明实现完成、实际训练量、理论命题状态、经验/闭环增量，不用一个总 PASS 掩盖不同问题。保存 diff 供验收；未获用户要求不推送或自动进入下一阶段。

## 参考实现导航

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

完整 33 项论文与相邻工作继续见正式 v1.1 方案第13节；它们不是33个都必须安装的依赖。优先完成本阶段的完整任务，按具体缺口阅读，避免把文献清单变成新串行关卡。

## 远程训练和长期运行

### 连接和端口

用户提供的连接命令为：

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

**2026-10-15 本次只读核查：远端 8080 已被 Jupyter 占用，18080 未监听。** 不要停止 Jupyter 或抢占其端口。监控页默认绑定远端 `127.0.0.1:18080`；用户在自己的电脑另开终端运行：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -p 44327 root@107.209.104.125 -L 8080:127.0.0.1:18080
```

随后打开 `http://127.0.0.1:8080/`。若本机 8080 已被上一条 SSH 占用，先结束**用户自己的那条转发连接**，或改 `-L 18080:127.0.0.1:18080` 并打开 `http://127.0.0.1:18080/`。执行前重新查端口；若远端 18080 也被占用，选空闲端口并明确给出新的转发命令。不要为了监控改成公开 `0.0.0.0`、关闭现有认证或另建复杂登录系统。

进入服务器完整阅读 `/etc/vast-agents-guide.md` 和 supervisor 包装脚本说明 `/opt/supervisor-scripts/README.md`。再检查：

```bash
supervisorctl status
nvidia-smi
df -h /workspace
ss -ltnp 'sport = :18080'
```

本次核查为 RTX PRO 6000 Blackwell，约 96 GiB 显存，瞬时占用约 15.1 GiB；磁盘约剩 1.2 TiB。旧 RoboTwin policy 服务仍在运行。上述仅是准备时快照，正式运行前重测，不停止其他任务/管理服务。服务器指南说明容器回收/销毁可能丢失 `/workspace`，应查是否有持久卷，重要源码、配置和恢复点及时保存到用户已有存储，不自行购买服务。

### 已有资产和新输出位置

| 用途 | 路径或约定 |
| --- | --- |
| RawVLA 作者源码 | `/workspace/owac/rawvla/source` |
| RAW 数据根 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data` |
| LIBERO / RoboTwin 缓存 | 上述目录下 `libero/raw`、`robotwin2/raw` |
| encoder Python | `/workspace/owac/rawvla/envs/encoder/bin/python`；本次 Python 3.11.16 / torch 2.9.1+cu128，CUDA 可用 |
| policy Python | `/workspace/owac/rawvla/envs/starvla/bin/python`；本次 Python 3.10.21 / torch 2.7.1+cu128，CUDA 可用 |
| RoboTwin runtime | `/workspace/owac/rawvla/envs/robotwin` |
| LIBERO 权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO 统计 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/dataset_statistics.json`，既有 `openpi_pi05_libero_franka` / `action` |
| RoboTwin 权重与统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`、同目录 `policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 原完整 Default 记录 | `/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1/`，只读 |
| P2 训练/审计 | `/workspace/owac/rawvla/research/n01-v1/runs/p2-motion-20261014/`、`p2-audit-20261015/`，只读 |
| 本阶段新 run | `/workspace/owac/rawvla/research/n01-v1/runs/v1-1-<stage>-<run-id>/` |

`RawVLASequenceDataset(root=...)` 的默认路径推导与实际服务器布局有差异。显式传正确 `raw_cache_root`、`stats_path` 及清单侧别，不因默认路径找不到就重下数据。权重和统计在本阶段启动时逐项检查；本表不是“所有未来文件必然存在”的保证。

新 run 建 `source/`、`config/`、`outputs/`、`receipts/`。从当前 checkout 复制本任务实际源码、配置、需要的 vendor 和许可证，包含未跟踪的新代码；仅 `git archive HEAD` 会丢掉未提交实现。不要打包 `.venv`、700 GB 数据、密钥、其他方向输出。简单记录版本与 diff，确认远端加载的 `owac.__file__`、入口和有效配置正确即可，不新建逐文件哈希硬门槛。

编码器环境与固定策略环境有不同 Python/torch 版本。先复用已通的桥接方式；若必须在同一进程做端到端 autograd，建立能同时导入完整编码器和固定 π0.5 的独立兼容环境，验证真实 forward/backward。**不通过 HTTP 特征服务训练端到端梯度**，HTTP 仅用于独立闭环推理。新依赖不升级旧基线环境；本地 CPU torch 锁不覆盖远端 CUDA wheel。Blackwell 使用兼容 cu128 或经实测可用的 wheel，不装/升级宿主 NVIDIA 驱动。

### 作业和恢复

训练、评测队列、监控分别由 supervisor 管理，命名 `owac-n01-v11-<stage>-<role>-<run>`。包装脚本前台启动、实时 flush 日志。只添加自己的配置并用 `supervisorctl reread`、`supervisorctl update <本任务program或group>` 定向加载；不重启整个 supervisor，不操作已有管理/基线服务。仅 SSH 私有转发的服务不需要修改 Caddy/portal，也不要套会因没有 portal 条目而跳过启动的 guard。

训练保存 model、optimizer、scheduler、混合精度 scaler、RNG、sampler/数据进度及配置；checkpoint 用临时文件完成后替换。正常完成的有限队列不被 supervisor 无限重启；崩溃重启从最近有效 checkpoint 接续，同 run 不从 step 0 覆盖。当前仅调度每平台/方法1个训练种子，默认101；不自动排211/307。提前估算速度与总成本；不要在剩余显存不足时同时塞进多个 π0.5。

对相同恢复点做一次短续跑验证即可，普通浮点差异不要求跨机器/版本逐位一致。训练/评测队列与网页解耦：网页关掉不停止训练；训练退出后网页仍显示最后数据和真实退出原因。资源暂时被占用时网页显示“等待 GPU”，不假装正在训练。

## 立即分派给前端专家的完整任务

主执行者：在主体长任务开始前，把以下任务完整交给可调用的前端专家/子 agent。你负责进度写入、HTTP 服务和训练/评测 hook；前端专家负责页面。双方先约定下面的少量 JSON 字段后可并行，不相互回退文件。若运行环境没有子 agent/专家工具，明确说明并按相同要求自行实现页面，不假称已经调用专家，也不因此搁置监控。

> 你是本次 N01 v1.1 的前端专家。请实现一个中文、单页、只读的实时研究进度页，让用户在 DeepSeek 长对话和远程训练时看见“现在做什么、做到哪里、是否还活着、结果如何”。页面保持简单，优先原生 HTML/CSS/JavaScript，复用仓库 `baseline_dashboard_template.html` 中有用的图表思想，但它目前是静态快照，不可直接宣称实时。不要引入 React 工程、数据库、WebSocket、账户管理、复杂路由、拖拽组件或多个管理后台。
>
> **文件所有权：**你拥有拟新增 `src/owac/runtime/n01_v11_monitor.html`；主执行者拥有 `n01_v11_monitor.py`、进度 JSON writer 及训练/评测 hook。你不是独自在代码库中工作，不得回退其他人的修改。需要接口变化先同步主执行者。
>
> **布局：**顶部显示 N01 / v1.1 / P3 或 P4 或 P5、当前活动的一句中文、运行状态、最后更新时间和数据是否陈旧。下面放四张小卡片：当前方法/平台/训练种子；更新步数或 rollout 完成数；用时/速度/有依据时的预计剩余时间；GPU 显存/利用率。主区域两列：左边真实曲线，右边最新真实案例；底部是简单任务矩阵与最近日志。手机窄屏变一列；文字和坐标单位清楚，默认浅色，无复杂动画。
>
> **状态语义：**支持“编程中、检查中、等待资源、训练中、评测中、阶段结束、失败、需要用户处理”。编码工作总量未知时显示当前步骤和最近活动，不编造百分比。训练百分比只能来自 `step / planned_steps`；评测来自 `completed_rollouts / expected_rollouts`，同时列成功、正常失败、基础设施待恢复。P3 的开发评测、P4 的正式全量、P5 的迁移实验明确标注。未测量显示“尚未测量”，不填 0；曲线空时说明暂无数据，不能伪造示例分数。预计时间只依已完成速度估算并注明“估计”。
>
> **曲线：**显示实际 train action loss、固定验证 action loss；有开发闭环结果时另画成功率，不与 loss 共用不标注的坐标。P4 主要显示覆盖进度和按方法/平台/种子的成功率，运行中的成功率标“暂时结果”，原正式分母始终可见。只用已有指标，不新增综合评分。进度页不判定“收敛通过”。
>
> **案例：**显示同一时刻的输入预览、真实历史融合热图/支持区域图及可选视频链接；可切换有限的固定案例。DBSR 通道均值热图标“平均融合权重”，不称后验或因果贡献。尚未生成时显示说明；不在浏览器端读取 RAW 全集或 checkpoint。原始图、图例、相机、episode/frame 和单位可读。日志保留最后约 100 行，错误使用文本渲染，页面不提供任意 shell、启动/停止任务或删除数据按钮。
>
> **接口：**每 3 秒 GET `/api/status`，约每 10 秒 GET `/api/metrics`；请求未结束不叠加下一次，后台标签页适当降频。HTTP 服务另由主执行者提供。最小状态含 `stage`、`state`、`activity`、`updated_at`、`current_run`、`runs`、`resources`、`artifacts`、`recent_log`。字段未知可省略或 null，不设计严格 schema 认证系统。训练卡片使用 current_run 的 method/platform/seed/step/planned_steps；评测使用 completed_rollouts/expected_rollouts/successes/failures/infra_pending。metrics 返回带实际 step/时间戳的有限点列和明确 metric 名称/单位。
>
> **断连：**请求失败保留最后成功数据，显示“连接中断”和最后成功时间，自动重试；恢复后清除断连提示。状态更新时间超过约 60 秒时提示“进度数据较久未更新”，不自动判训练失败；失败须来自进程/退出状态。提供一个手动刷新按钮即可。
>
> **交付：**给主执行者页面文件、字段约定和真实浏览器验证记录。用一份小样例检查布局可以，但标明仅 UI 调试，正式网页接真实 run 后移除样例。实际检查首次无数据、训练更新、评测更新、服务中断恢复、窄屏布局、图表和案例读取。无需另建复杂前端测试矩阵或性能门槛。

### 主执行者必须完成的接线

前端任务不是把用户留给另一个人的理由。主执行者负责把页面、真实进度和服务接通，并给用户可打开的地址：

1. 拟新增 `src/owac/runtime/n01_v11_monitor.py`，用已有轻量服务或 Python 标准库 HTTP 服务。只绑定 loopback，限制读取本阶段输出目录，提供 `/`、`/api/status`、`/api/metrics` 和图像/视频文件。不要暴露服务器任意文件或把远程 Jupyter 当 N01 页。
2. 在本阶段 run 内写 `monitor/status.json`、`metrics.jsonl` 和小型案例索引。一个轻量汇总进程整合多作业状态；各训练 worker 写自己的文件，避免多个进程争写同一个 JSON。状态文件用临时文件替换，记录指标的真实 optimizer step；梯度累积的 microstep 不能当更新步。
3. 编程阶段由执行者在有意义的进展时更新 activity；训练/评测由实际循环持续更新。约 5–10 秒的轻量心跳和低频 GPU 采样足够，不每个 step 调 `nvidia-smi`，不在 HTTP 请求里加载 torch/model/大日志。metrics 下采样为有限点数，日志读尾部，页面轮询不增加 GPU 负担。
4. 该接口的**建议形状**如下，按实际需要删减字段；不要把样例值作为运行结果：

```json
{
  "stage": "P3",
  "state": "implementing",
  "activity": "接入完整编码器的真实数据训练入口",
  "updated_at": "实际UTC时间",
  "current_run": {"method": "N", "platform": "libero", "seed": 101, "step": 0, "planned_steps": null},
  "runs": [],
  "resources": {},
  "artifacts": [],
  "recent_log": []
}
```

5. 拟提供以下 CLI；这是**要求新增的接口，不是当前已经存在的命令**：

```bash
PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.runtime.n01_v11_monitor \
  --run-root "$N01_RUN" --host 127.0.0.1 --port 18080
```

其中 `N01_SOURCE`、`N01_PYTHON`、`N01_RUN` 在本次启动脚本中设成实际绝对路径。按远程执行段用独立 supervisor program 持续运行。先检查远端 `/api/status` 和页面，再经 SSH 转发验证用户入口。优先在第一轮长训练之前上线，并持续复用到 P4/P5；不是最后补一张静态报告截图。

6. 监控失败不应杀掉正确运行的训练；保留日志、修复监控。仅 status 文件存在也不表示训练在运行，需结合进程/队列状态展示。页面关闭或 SSH 断开不影响作业，重新连上可恢复查看。

## P3 本次任务与执行顺序

**现在实际实施 v1.1 P3：一次建立完整 N01，用两平台真实 RawVLA-Bench 充分训练，完成匹配强对照、开发闭环、理论说明和同步可视化。** 不拆回旧 P3/P4/P5/P7 的局部研究阶段，不等待局部代理分数先过线。这里的步骤只是同一 P3 内部的依赖顺序。

### 第一步 复用已有实现并上线进度页

先检查已有接口，写一页简短的“已有/待新增”映射，随即编程。不要重复实现 P0/P1 的解析参考和 P2 的完整 SEA-RAFT。立即按前面的专家任务分派监控页，同时建立真实 run 状态；在第一轮长训练前让用户能访问网页。

| 已有位置 | 需要继承的内容与实际限制 |
| --- | --- |
| `src/owac/frameworks/n01_full/adapters.py` | `DbsrFeatureAdapter`、`DbsrAdapterConfig`、`FullState`；已有 RAW 内容/运动/桥接，当前 P3 支持逻辑硬关闭，不能只改一个开关就称完整 N01 |
| 同目录 `ledger.py`、`domains.py`、`statistics.py`、`replay.py` | 来源、GLS、域修订和反事实参考；可作为小规模独立数值参考，不把逐像素 Python 对象账本搬进整幅实时训练 |
| 同目录 `motion_builders.py` | 从清除后来源重建 motion/noise 的接点；参考其依赖语义 |
| `src/owac/baselines/burst_features.py`、`_vendor/dbsr/` | 成熟内容与 `MaskedDbsrFusion`，已有真实 `[B,T,C,H,W]` 权重；融合前后依赖都要检查 |
| `src/owac/backends/raw_correspondence.py`、`_vendor/sea_raft/` | 完整 SEA-RAFT RAW 适配；坐标 pull 方向、sensor/packed 单位和风险 proxy 不能混用 |
| `src/owac/raw/rawvla_sequence.py`、`n01_full_manifest.py` | `RawVLASequenceDataset`、`MeasurementWindow`、`collate_sequence_samples` 和四区清单；显式指定真实数据路径 |
| `src/owac/backends/rawvla_features.py` | `VisualTokenPostprocessor`、`RawVLAFeaturePolicy` 和 action chunk 协议；复用原生训练/推理接点 |
| `src/owac/runtime/n01_full_policy_preflight.py` | 已有 `contract/raw_adapter` 真实预检；说明连通性，不是正式完整训练器 |
| `src/owac/training/n01_action.py` | 原型训练器，默认20步/小裁剪/旧工厂，**不能把它换个输出文件名当 v1.1 正式训练** |
| `src/owac/runtime/n01_feature_service.py`、`rawvla_full_eval.py` | 服务与冻结平台客户端的参考；旧 factory/默认 policy 路径须真正替换成完整 N01，不仅修改标签 |

外部保留现有语义：`init_state(batch_size,camera_spec,device)`；`step(frame,embodiment=None,state=...,budget=...,offsets_xy=None)`；`readout(output,token_budget=None)`；`reset(state,batch_mask=None)`；`detach_state(state)`；`state_bytes(state)`。按实际协议类型接入，不另造不兼容平行接口。`encode_window(values,cell_mask,offsets_xy)` 已有的输出只是内容融合；新增支持后的数据必须真的进入训练/服务。

新配置使用 `configs/encoders/n01_full/v1_1/`。建议文件职责如下；可合并简单辅助代码，不为凑表格造空壳：

| 拟新增或扩展位置 | 职责 |
| --- | --- |
| `frameworks/n01_full/support.py` | 张量化空间域候选、沿轨迹的时间相容/变化状态、接受/暂缓/撤回的解释 |
| `frameworks/n01_full/model.py` | 完整模型工厂、支持参与内容通路、32帧重放及基线开关；复用 adapters 而非另写弱骨干 |
| `frameworks/n01_full/adapters.py`、`replay.py` | 因果来源/缓存/重算接线及公共接口，保留历史参考可用 |
| `training/n01_v11.py` | 真实动作训练、匹配矩阵、checkpoint/resume、进度事件；不是 HTTP 服务训练 |
| `evaluation/n01_v11.py` | 固定动作验证、真实来源干预、开发案例和汇总 |
| `runtime/n01_v11_service.py` | 将完整模型接到既有 π0.5/平台服务；真实相机到达、reset和期限快照 |
| `runtime/n01_v11_monitor.py`、`n01_v11_monitor.html` | 前面规定的简单实时监控 |
| `research/theory/n01_support_contract_v1_1.md` | 主链依赖、删除命题、噪声/陈旧偏差和顺序保证边界 |

表中的包内路径均以 `src/owac/` 为前缀；不存在的文件是本次要写的文件。

### 第二步 在同一个完整编码器中实现六项功能

**A. 实际测量身份。** 按 episode/camera/真实帧号或实际到达标识登记来源，并保留 channel/phase/实际采样坐标。训练窗口重叠、同一包重发、π0.5 多次 flow denoising 不增加独立观测。用紧凑索引和张量 mask 承担大图；重复项去重不等于不同曝光彼此统计独立。缺失、坏点、饱和分别保留。

**B. 成熟内容和对应。** 默认完整 DBSR 宽64、9残差块、512特征及成熟融合，不削成小 CNN。三通道输入用已有三通道 stem，不补假第四通道。先对每个来源独立编码再对齐，以便依赖可追踪。SEA-RAFT 估计“当前参考坐标→历史源坐标”的 pull map，明确单位；当前时刻及过去才是合法输入。以真实 RawVLA 轨迹检查运动覆盖与错误区域，保留可视化；P2 checkpoint 可初始化，但不假定它在真实场景已准确。

**C. 空间支持与时间修订。** 用实际映射后的采样位置构造局部常数/仿射 GLS、校准噪声和残差；空间候选采用有限局部区域与多尺度邻域，具体尺度在目标数据开发集上设定并记入配置，不把任意 A–B、B–C 边的连通闭包当整个域相容。实现域级残差/相容性判断。沿对应轨迹维护 BOCPD 或鲁棒 BOCPD 数值状态，使用作者算法，不手造一个“变化置信度网络”取代它。

明确区分：稳定且可对齐的历史、映射不确定、单帧异常、持续结构变化、缺失/坏点/饱和。暂缓可疑来源，积累后续真实反证再接纳或退出；不能把运动速度本身当撤回依据。hazard、噪声/风险尺度和阈值使用 calibration/validation 的实际数据选定；BOCPD 输出是模型后验，不是已证明的误报概率。风险较差时展示并诊断，不通过人工压低风险值让图好看。

**D. 支持必须改变实际内容。** 支持改变进入融合的有效来源、参考构造、logit/softmax 归一化和必要的重新编码，影响最终桥接特征与动作。不能只把 support 用在诊断字段、loss 附项或最后输出 mask。检查卷积感受野、参考帧和权重预测是否已经混入退出区域。精确逐查询像素依赖太昂贵时，以区域和保守感受野界重算/屏蔽，说明分辨率与范围；不声称做到了并未实现的任意像素级独立。

**E. 两种撤回与32帧有界精确回放。**

- 结构支持边退出：该源不再给某结构提供测量支持，但可以留下“曾发生变化”等上下文；保留这种上下文就不能说最终表征完全与该源无关。
- 来源完全删除：删掉整帧或选定来源后，凡依赖它的内容、motion/noise估计、支持决定、参考、归一化、状态和读出均从剩余合法测量重建；这才可以检查全删除后的独立性。
- 默认每 episode/camera 最多32个唯一已到达帧，整个有限窗口重算优先。BOCPD 后验、解析状态、神经缓存都由这个窗口和同一固定初值重建，不能带着窗口之前的 posterior 再宣称精确32帧记忆。更早历史丢失明确报告。
- 窗口内来源删除后将其位置标无效，不从窗口外偷偷补进更老帧；若另做会重选窗口的策略，单独命名并计其输入变化。相机缺帧和 episode reset 可真实执行，batch 中局部 reset 不污染其他成员。
- `fixed_decision_replay` 固定决策只衡量直接路径；`counterfactual_replay` 重新计算受影响决策才能衡量完整删源。不能混报。训练时 optimizer 更新后不复用旧参数算出的特征缓存；使用 checkpointing/相机分块仍要处理真实32帧，不能实际只看8帧却在配置写32。

先完成正确的窗口重算，性能优化留有依据再做。没有证据需要时不加 Mamba、全局依赖图数据库、局部缓存失效框架或无限历史。

**F. 期限读出与行动。** 保留现有512→2048桥，默认每相机256个16×16带mask token，替换原RGB图像塔，使用π0.5原生语言/动作后端。来源状态更新后原子提交完整快照。期限前未产生新合法快照时，可返回带来源时间的最近合法快照或单帧降级；全无有效输入则返回显式不可用，不伪造零动作成功。只提供合法输出的工程语义，不承诺 GPU 的硬实时最坏时延。记录队列、数据搬运、flow、支持、重算、后端、动作提交的总 p50/p95/p99；改变重规划或 action chunk 中断规则属于另一个控制实验。

### 第三步 接通两平台真实动作训练

首轮就包含完整 A—F，不能训练完旧 CNN 后再宣布将来接支持。内部短 fit/debug 是允许的，但正式 N01 始终是完整对象。

1. 使用已有四区真实 RawVLA 数据。先打印有限真实 batch 的字段/shape/相机顺序/动作单位，保存几张固定预览，核对实际相邻时间间隔、运动与暗光条件。RGB 只在预览使用，不进入主模型。
2. 接通 `RAW encoder → VisualTokenPostprocessor → π0.5 native action flow matching`，确认 encoder/bridge 的真实参数差与 autograd、冻结后端范围均符合配置。取少量训练轨迹做短 fit；通过真实图像/历史替换干预查看预测是否受视觉影响。若只依赖语言或 state，需要修训练，而不是用 loss 下降当视觉已学会。
3. 优先用动作监督适配完整 SEA-RAFT 与编码器；如 flow 辅助监督确有必要，来自真实 RawVLA 帧的已知空间变换可给有限已知对应，明确这是变换监督，不是自然真实 flow。自然曝光变化、噪声/遮挡未知时，不把两帧差或普通照明配对当真值。无需另设10万步合成预训练前置任务。
4. 主损失沿 π0.5 原生 action flow matching。RAW辅助项仅在噪声独立/曝光校准/目标可识别等条件明确时采用，并保留开关及无辅助对照。禁止 RGB 重建、RGB教师或用未来专家动作作为观测。正常条件与扰动条件有相同标签合法性才共享监督。
5. 先冻结后端训练编码器/桥，再在 N/B0/B1 上开放相同后端参数范围。具体范围、初始化、optimizer、LR schedule、有效batch、precision、窗口和帧到达间隔写有效配置；不同方法享有相同解冻与合理搜索机会。后端的合法预训练权重可复用，不保留其RGB塔给N暗中提供视觉内容。

**训练预算按真实速度制定，不能用 smoke 冒充充分。** 先测完整分辨率/32帧/两平台实际吞吐和显存，给出一轮核心矩阵的GPU小时估算。初始规划每平台、每方法、每种子约30k—100k真实 action optimizer updates；这是容量匹配的规划范围，不是论文保证或固定合格线。可从50k安排、预留到100k的有记录延长；根据作者训练代码、短开发曲线与资源确定实际值，配置和网页显示其含义，不把2k ISP配方直接称N01训练充分。

用固定噪声/flow时间的验证子集比较 checkpoint，再用普通随机验证检查全分布；报告训练源轨迹覆盖、条件覆盖与采样重复率。验证间隔按成本设置，不能每十步做一次昂贵闭环。以匹配曲线、训练拟合、验证表现、视觉利用和开发闭环共同判断优化是否充分；仍改善时按登记预算延长，未改善但拟合很差时诊断容量/梯度/对接。既不设人为收敛PASS，也不因一次loss反弹或达到规划末步就强称充分。真实预算到期可交付明确不足/负结果，不自动无限续训。

### 第四步 匹配对照和开发闭环必须在 P3 完成

核心方法固定为：

| 方法 | 与N共享什么 | 主要差异 |
| --- | --- | --- |
| N | 完整N01 | 空间支持、沿轨迹修订与受影响内容重算 |
| B0 | 成熟DBSR、SEA-RAFT、输入历史、桥、后端、训练/搜索机会 | 成熟融合，不加入N01支持修订 |
| B1 | 同B0；标准BOCPD/鲁棒检测参考 | 标准变化检测触发统一/区域reset，检验N是否超出成熟融合+常规reset |

两平台分别训练，每臂仅1个训练种子，默认101；N/B0/B1共6个“平台×方法”主体训练run，允许顺序执行。不得自动追加211/307或把三种子作为P3门槛。B1 detector 用相同可见输入和 calibration 数据；辅助参数差异明列，必要时容量匹配，不能把 B1 削弱成“速度一大就清空”。独立 Default/RawVLA 原视觉塔系统是实用强参照，其预训练差异不能冒充N01机制的单变量对照。

额外解释性比较：同权重关闭时间修订/重算的冻结干预；若据此要主张训练后机制增量，再做匹配重训消融。两种证据分开。辅助loss开关仅在真的采用辅助项时安排，不机械扩张所有组合。

开发闭环使用独立登记的任务/初态/seed清单，三方法同身份；与 P4 正式确认身份尽可能隔离，作者协议固有重叠要明确披露。建议起跑每平台固定约100条分层开发rollout，按任务/照明/失败覆盖调整并登记；这是有成本的开发采样，不是成功数量门槛。三个方法执行同一开发清单，不能给候选单独选容易任务；单种子不能估计训练随机性，rollout seed仍按开发清单覆盖，二者不要混淆。小样本可先调试，最终P3报告保留完整开发清单结果和未运行项，不用5条成功视频代替。

检查在线客户端确实发送全部已到达的合法观测，encoder确实维护历史，action chunk多次采样不会重复计帧，真实模型权重/相机顺序被服务加载。完整模型通过WebSocket/HTTP等既有服务进行**推理**可复用，训练图仍在同一autograd进程。训练完才发现实际服务还走旧视觉塔是不合格的接线。

报告成功率、按平台/照明/任务的差异、事件后恢复（真实时间或到达帧数）和端到端成本。在这个固定训练seed下，以episode/任务为分组计算区间/配对差异；不估计跨训练seed方差，不把相邻帧当独立样本。无真实事件标注时给人工核查案例，不捏造“恢复真值”。P3不得冒充已完成P4的13,250全量确认。

### 第五步 直接可理解的同步可视化

在相同固定真实开发案例中同步输出：输入预览、多相机时间；逐历史来源融合热图；对应点轨迹/warp叠图和放大；支持区域接受/暂缓/退出及历史年龄；真实删源后的特征与动作差；同初态闭环视频和成功/失败。

DBSR权重 `[B,T,C,H,W]` 按固定通道均值聚合，overlay到同一原画面，标注来源时间和共享色标，名称“平均融合权重”。不要归一化每一帧让弱影响也显得火红。对齐残差与有GT的flow error分开，无几何标定不画假真值机器人轨迹。固定随机性进行删源/替换来源的动作干预；低权重高影响、删源不变、失败和随机案例都展示。

固定案例由分层随机、事件与失败样例构成，记录选择规则。任何局部改动重跑这组案例看完整行为，不只贴新loss表。网页引用小图与服务器视频，不每3秒重生成视频或加载完整特征。大数组/视频只留输出，小型索引保留 episode/camera/frame/checkpoint/图例含义。

### 第六步 理论说明贴着实际主链完成

写 `n01_support_contract_v1_1.md`，将每个命题落到具体算子，不只在解析小夹具证明：

- G1 删除/重算：列实际依赖图，说明完全删除后所有可达依赖如何重建及为何不再受该源影响；区分边撤回、固定决策、完整反事实。在真实图像/完整前缀与动作读出上做可检查反例；FP误差用合理容差，不要求跨版本bitwise。
- G2 保留历史的条件性收益：固定设计和权重、`y_i=z_i+eps_i`、权重和为1时，写清 `MSE=(sum_i w_i(z_i-z_t))^2 + w^T Sigma w` 的偏差/方差分解及相关噪声假设。自适应权重的数据选择偏差另述；`detach`不消除统计选择偏差。没有可识别相关噪声时仅报告proxy/保守界。
- G3 持续变化的顺序证据：若使用合法e-process/CS，逐项列条件可预测性和条件矩等假设；满足时才引用长期误报保证。若仅BOCPD后验与经验校准，准确标注当前是模型/经验结论及尚缺理论，不包装为无条件保证，不因此把完整任务永久卡在“无法证明PASS”。

创新对照写清：成熟多帧融合、标准检测+reset已经存在；新增假设是来源支持修订怎样真正改变内容和行动，以及何种条件下优于B0/B1。负结果同样诚实交付；不写无条件任务性能/有限帧必检保证。

### 要新增并实际跑通的命令接口

以下是拟实现CLI的约定，不是已经可直接运行的旧命令。参数可以按现有项目风格调整，但交付时必须写出真实最终命令和完整配置；不得抄本节后声称已经运行。

```bash
PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.training.n01_v11 \
  train --config "$N01_RUN/config/libero-N-seed101.json" --output "$N01_RUN/outputs/libero-N-seed101"

PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.training.n01_v11 \
  train --config "$N01_RUN/config/libero-N-seed101.json" --output "$N01_RUN/outputs/libero-N-seed101" \
  --resume "$N01_RUN/outputs/libero-N-seed101/checkpoints/latest.pt"

PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.evaluation.n01_v11 \
  development --config "$N01_RUN/config/development.json" --output "$N01_RUN/outputs/development"
```

三条仅示例一个矩阵项；实际包含两平台N/B0/B1各1个训练种子的训练、验证、服务和开发闭环调度。配置必须明确加载哪个完整工厂，不能默认退回原型。

### P3 交付后停止

给出完整源码与有效配置、可续跑脚本/矩阵、全部run的实际训练步数与数据覆盖、单种子N/B0/B1开发结果、真实案例入口、删源/状态/期限关键验证、理论说明和成本。网页显示“P3阶段结束/尚缺什么”，保留查看。报告未完成项和负结果；训练未完成不能写P3全部完成，性能未提高不伪造PASS。供独立验收后再固定P4候选，本次不自行启动正式全量P4或P5。

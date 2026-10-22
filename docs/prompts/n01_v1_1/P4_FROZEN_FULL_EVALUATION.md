# DeepSeek N01 v1.1 P4 冻结后的完整基准验证

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> **历史提示词：当前执行请使用 [N01 v1.2完整包](../n01_v1_2/README.md)。本文件不再作为后续启动依据。**

本次仅执行 **N01 v1.1 P4**。这是一份完整交接包，共同要求、参考实现、远程训练、前端专家任务和阶段指令均已展开；请从头读完再执行，不需要用户手动拼接其他提示词。准备本包时P3尚未实施，后续阶段的前置交付必须以届时实际记录为准。

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

## P4 本次任务与执行顺序

**现在实际执行 v1.1 P4：将 P3 交付的完整模型与匹配强对照冻结，完成原完整 RawVLA-Bench 的正式验证。** 这是完整结果验证，不再添加网络、调参训练或用正式测试追逐最佳checkpoint。

### 第一步 从实际 P3 交付确定评测对象

阅读 `research/experiments/n01_v1_1/P3_report.md`、P3独立验收记录、有效配置和服务器 `RUN_STATE.md`。这些属于前阶段未来交付，准备本提示词时尚不存在；执行时按真实位置读取，不因准备时找不到就创建假的已通过报告。

核对完整模型、N/B0/B1单种子权重和两平台服务都确实存在。P3负结果不妨碍如实做确认；如果主体训练/桥接/开发闭环实际未交付，则先修其必要缺口并如实标P3未完成，不拿旧prototype或未训练模型替代。修复会改变研究方法/选模时返回开发迭代，不能在正式测试中悄悄替换。

保留P3监控页，按前端专家任务让专家复用页面、切换P4内容；没有监控实现则先按同一简单规范补齐。不要为P4重写第二个复杂前端。

冻结一份可读配置，至少列出：

- 每个平台N、B0、B1各1个训练种子（默认101）的checkpoint和**根据开发/验证数据已确定**的选择规则。
- 完整模型和policy服务代码版本、有效配置与必要未提交补丁；输入规格、32帧、校准、相机顺序、bridge、后端解冻范围、动作统计和控制节奏。
- 原正式rollout身份、照明/噪声条件、语言生成seed、初态/任务seed、超时、成功判定、失败分类、允许的基础设施重试规则。
- 运行目录、恢复位置和总成本估算。保留已有manifest身份，不新增每帧/每文件签名仪式；真正需要的是固定“哪个模型在什么条件下跑了哪条轨迹”。

若P3使用的单种子不是101，以实际已登记种子为准，不为凑编号重训。当前只要求1个训练种子，不补211/307，不将单种子报告为矩阵缺口；若存在其他种子结果可另列，但不利用正式测试挑最佳seed。

### 第二步 接入已经冻结的平台而不修改原基线

重点阅读本仓：

| 位置 | 用途 |
| --- | --- |
| `src/owac/runtime/rawvla_full_eval.py` | 原13,250主清单生成、固定seed、分平台runner与恢复/汇总；其旧prepare含固定policy准备，不能直接执行后以为加载了N01 |
| `src/owac/runtime/rawvla_robotwin.py` | RoboTwin运行协议与已有基础设施修复；保留原身份和动作统计 |
| `src/owac/backends/rawvla_features.py` | `RawVLAFeaturePolicy`、action chunk和RAW前缀，核查实际服务载入模型 |
| P3实际新增完整模型/服务/调度入口 | 按P3实际文件接续，不假设计划中的CLI已经实现 |
| `docs/decisions/0012-rawvla-pi05-protocol-freeze.md` | 原协议固定范围 |
| `research/archives/2026-10-14_rawvla_pi05_baseline.md` | 如本worktree没有此文件，查主checkout同路径或服务器record，不重跑Default来补一份文档 |

远程原完整记录 `/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1/` 只读参考。Default为LIBERO5820/10000、RoboTwin508/3250；它保留原RGB视觉塔和不同预训练条件，是实用强系统参照。不能把其结果当N/B0/B1新权重的评测结果，也不能为了复用分数而忽略协议差异。

建议新建 `src/owac/runtime/n01_v11_benchmark.py` 作为明确载入P3完整模型的调度器，复用已冻结的客户端/manifest/统计。**不要重写仿真器/成功函数，也不要直接修改原 Default 冻结输出。** 新源码、新服务端口、新结果目录独立于 `pi05-full-20261010-v3`。检查首条真实请求的RAW相机、模型工厂、checkpoint和返回动作，确认原RGB塔不在N主链。这个短接线检查不承担性能结论。

### 第三步 完整执行核心矩阵

每一个“方法 × 训练种子”的平台checkpoint，分别覆盖：

| 主正式清单 | 每个checkpoint的原分母 |
| --- | --- |
| LIBERO | 10,000 |
| RoboTwin | 3,250 |
| 两平台合计 | 13,250；同时分平台报告，不用简单均值替代 |
| 附加原条件核查 | 632，单独目录/分母；如执行则另记成本 |

N/B0/B1 × 1个训练种子的主矩阵总计 `3×1×13,250=39,750` rollouts；N+B0为26,500，仅用于说明成本，不授权静默删掉B1。每个方法的单种子checkpoint按平台完成完整清单。rollout的任务/初态seed照原协议覆盖，不因只训练一个seed而只评一条轨迹。附加632不混入主分母，不默认重复跑旧模型已有且协议一致的独立核查。

先按真实吞吐估算总GPU/仿真小时，在网页展示矩阵与预计剩余。按已有资源串行或安全并行，不同时启动GPU内存容纳不下的多个policy服务。监督器与调度器支持断点，从已完成的身份继续；不能每次重启都从第0项重评。

**失败处理固定且诚实：**正常任务失败、碰撞、策略超时照原协议计入失败和原分母。基础设施错误标具体原因，按冻结规则同身份恢复；保留每次尝试及最终有效尝试的选择规则。不能换seed、删困难条件、无限重跑到成功、以最大尝试分数合并。未恢复的infra缺口始终可见，阶段标incomplete，不从分母消失。已成功的有效记录不重跑挑更好结果。

遇到Blackwell的OptiX/渲染问题参考既有修复并单独披露；不能通过更改任务物理、相机、光照或成功函数让某方法“跑通”。确定仅是日志/监控bug可修；会改变输出的方法/控制bug需要保留已消费测试与新版本边界，不能称原冻结确认持续有效。

### 第四步 实时页面显示正式覆盖而不是假完成

每行显示方法、平台、训练seed、实际checkpoint、完成数/原分母、成功、正常失败、infra待恢复、用时、队列状态。当前矩阵项的进度按真实已完成身份更新，另显示全部矩阵项的总进度；不要把多种子合并成功率画成一个checkpoint的成绩。

页面显示“进行中暂时成功率”和“最终完整成功率”的区别。评测程序退出但仍有缺口时显示未完成；退出码0不能覆盖真实缺口。显示最近异常、恢复活动、当前任务和少量真实视频。页面保留P3训练曲线链接，不重复读大checkpoint。监控页故障单独修，正确运行的rollout不为网页重启。

### 第五步 汇总任务结论和动态扩展

主要给N相对B0/B1在固定训练seed下的按平台、照明/任务结果。区间/配对比较以任务/episode分组；不计算不存在的跨训练seed方差，区间不代表训练随机性，不用视频帧制造巨大样本数。保留正常条件是否退化、事件后恢复和完整p50/p95/p99成本；没有统一总分，不设“必须提升X个百分点”的验收线，负结果完整报告。

冻结原协议的全量成绩与额外动态恢复扩展分开：自然动态、错误历史来源干预、成像异常、改变正确动作的物理/目标变化分别版本化。只有具备合法观测/真值/标签时安排扩展；不凭空造隐藏状态真值。N/B0/B1获得同样观测与控制机会。扩展结果不能替代原13,250覆盖，也不能通过改变原协议伪装确认收益。

冻结删源干预与匹配重训消融分别解释。原协议允许任务模板/训练初态重叠时如实披露；新增严格holdout单独报告，不修改原主分母。全量跑完只支持所测pseudo-RAW仿真任务，不扩张为原生CFA或实体机器人结论。

### 本阶段拟新增命令与交付

以下为需要依据P3实现落地的接口示意，不是声称当前已经存在：

```bash
PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.runtime.n01_v11_benchmark \
  run --config "$N01_RUN/config/frozen-benchmark.json" --output "$N01_RUN/outputs/benchmark"

PYTHONPATH="$N01_SOURCE/src" "$N01_PYTHON" -m owac.runtime.n01_v11_benchmark \
  status --output "$N01_RUN/outputs/benchmark"
```

同一 `run` 命令应按已有有效身份恢复，不必为恢复另造一套工作流。交付真实最终命令、冻结配置、矩阵与每checkpoint完整覆盖、失败/重试记录、统计和同步案例、成本与局限。大视频/轨迹留服务器，小型结果索引和 `research/experiments/n01_v1_1/P4_report.md` 入库。相关代码做必要检查，**不因页面改文案重新执行39,750次rollout**。

完成后停在P4；监控页保持只读结果。没有正收益可形成完整负结果；矩阵未跑完则明确未完成。不得自行用正式测试继续选模或启动P5。

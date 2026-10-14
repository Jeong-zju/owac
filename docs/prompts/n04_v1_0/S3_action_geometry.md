# S3 已执行动作与三维自运动开发提示词

## 执行上下文

你负责实际编程与实验。这份提示词已经指定研究路线，请按顺序完成当前阶段的代码、验证、训练或数据审计和交接，不停留在计划、空接口或伪代码。当前阶段之外的模块只保留必要接口。遇到数据事实与假设不符，先完成不依赖该条件的工作，再写出具体缺口、已检查来源与可执行补齐方案，不替换成虚构输入。

本地 checkout：`/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`；分支 `lzh/n04-egomotion-separation`。开始读 `AGENTS.md`、`README.md`、`docs/status.md`、[全量方案](../../plans/N04_full_development_v1.0.md)、[首版实现合同](implementation_contract.md)、[源码核查](preparation_audit.md)。先检查工作树，保留既有修改和 FilterPy stash。下面标“拟新增”的文件/API/CLI 是你的开发目标，不能当成已经存在。

**实际数据核查、正式训练和真实 π0.5 验证优先在用户指定远端执行：**

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

先读[远端运行规约](remote_execution.md)。数据根为 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`，LIBERO 在 `libero/raw`，RoboTwin 在 `robotwin2/raw`。新源码和产物放拟创建的 `/workspace/owac/research/n04-v1/`，冻结 RawVLA 数据/源码作为只读参考。纯 encoder 使用 S0 固定的独立 Python 3.11+ 环境；后端继续验证 `/workspace/owac/rawvla/envs/starvla/bin/python`（实测 3.10.21 / torch 2.7.1+cu128 / numpy 1.26.4）。本地 Python >=3.11 / NumPy >=2.3 的 uv 环境不能直接覆盖它。正式代码经包含未提交文件的 hash 快照同步，远端复核 hash 后运行，不能只同步 HEAD。

实际 GPU 是单张 96 GB 级 RTX PRO 6000；上次约 32 GB 被已有服务占用。先查 `nvidia-smi`、`supervisorctl status` 和磁盘，按剩余资源 profile，不自行停止其他任务。正式长任务用 N04 专属 supervisor program、绝对解释器与不可变源码快照；日志/checkpoint 留远端，小报告/hash/曲线回本地。

模型主线固定为完整 ConvNeXt-B、作者 RKN/Ac-RKN、显式几何与误差传播。旧 tiny N04 留作历史。输入模式 `rawvla_rgb3` 与 `bayer_cfa` 分开；主链没有 RGB/sRGB/XYZ 中间图、RGB 教师或 RGB 重建。实际执行动作、当前本体与未来动作监督各有独立字段。正确实现和负实验结果同样交付；不预设新方法必须胜出。

## 目标和阶段拆分

在 S2 相同测量和状态规模上，依次增加三个因素：S3a 已执行动作；S3b 本体/FK 给出的相机位姿；S3c 深度三维投影。每个子阶段单独配置、训练和评测。固定相机下机械臂运动是可见机器人结构运动；腕部相机移动才产生相机 ego motion。动作推动杯子时杯子仍是外部对象，不能用“是否由动作造成”区分 ego/external。

读取 S0 telemetry coverage 与单位、S1 cache/权重、S2 state/association/checkpoint。关键动作/标定缺失时先补 S0 对应新数据，不把旧 supervised action 当过去控制。

## 源码与文件落点

读现有 [ego.py](../../../src/owac/frameworks/n04/ego.py)、[geometry.py](../../../src/owac/frameworks/n04/geometry.py)、[proprio.py](../../../src/owac/frameworks/n04/proprio.py) 与前阶段正式 `state_estimator.py/telemetry.py/measurement_geometry.py`。旧 `LearnedProprioPropagator` 是特征warp预测器，不是Ac-RKN控制模型。

拟新增 `frameworks/n04/action_transition.py`、`kinematics.py`；扩展几何与正式state，配置 `s3a_acrkn.json/s3b_fk.json/s3c_geometry.json`。新增控制流与解析投影测试。沿用训练/评测引擎，禁止为每种对照另写行为不同的数据loader。

## S3a 普通 RKN 到 Ac-RKN

使用既有RKN locally-linear转移与结构化协方差，增加已执行控制的加性影响。参考Ac-RKN论文与MTS3入口；MTS3授权未核定时，采用已引入MIT RKN按论文实现控制项，不复制未获授权代码。方法标为OWAC论文实现，不能标官方复现。

首版控制模型固定：`m_minus = A(m_plus) @ m_plus + b(u_exec, mode, dt)`；b用两层128宽SiLU MLP再投影到512，最后层零初始化；零动作偏置用 `b(u)-b(0)` 约束为零，避免掩盖无动作基线。继续保留完整RKN A和Q，不以MLP代替全转移。动作mode和units由S0定，真实控制Δt作为输入；这是本文工程配置，Ac-RKN主对照也采用同一控制容量。

时间基准先在S0固定controller tick上训练，按实际控制片段依序推进，在未观测的多个tick上允许多次predict、只有真实观测处correct。变Δt扩展单列：控制tick不足一整步的区间用声明的连续时间/分段近似，需dense toy对照；不简单把学得A乘Δt。普通RKN与Ac-RKN使用同一时刻栅格，报告frame-based旧数据只能作为独立诊断。

保留action coverage/mode/mask，不把缺失写成零控制。未执行计划、裁剪前控制、未来专家动作均不进入context。将控制编码模块预先注册进optimizer；报告控制项/转移项的梯度与实际输出量级。

训练对照：无动作RKN vs正确执行Ac-RKN vs时间错位/序列置乱/未执行计划替换的冻结干预。动作置乱损害不是目标本身，需动作确实有用时预测改善；若无增量保留负结果，不能靠action token存在声明成功。

## S3b 本体和相机位姿

只新增真实可用本体与FK路径。明确 `T_W_from_C=T_W_from_B * T_B_from_EE * T_EE_from_C`，固定相机单独使用常量外参。本体在目标曝光/预测时刻的插值只使用当时已经到达样本；双边插值所需未来样本只能用于有额外延迟的单独估计，不可冒充零等待在线预测。

S0已恢复可信joint/robot model时用FK；只有末端位姿时可用可信hand-eye但不可声称拥有完整关节链。首版机器人可见结构用其几何支持标为robot-self，和外部对象分报，不能简单清掉整只手臂掩盖接触物体。

避免重复计同一运动：S3a控制补尚未由本体确认的动力学；S3b相机坐标变换来自可用pose，不能再把动作估计的相机位移叠加一次。通过纯camera/纯arm/arm推动物体三个场景验收。对action motion与FK motion的贡献写入单独诊断字段。

## S3c 深度与三维运输

先只用同一測量，分别将估计ego替换为GT ego、再加入GT深度，mask/关联/更新/读出不变。这是diagnostic，不混入普通分数。证明状态/任务存在空间后，用S1几何头的学习深度；若瓶颈仍明确，再按完整Monodepth2/已定深度模块训练，正式配置和来源独立。

引入显式几何载体x_g=[p_W,v_W]与P_g(6×6)：常速度物理传播 `p'=p+v*dt`，Q按连续白加速度离散化（dt³/3,dt²/2,dt）并记录速度单位；外部动态学习/修订在S5。测量位置由当前深度与K还原；与潜在RKN共同读取同一当前图像，不当成两个独立传感器合并置信度。

相机ego只改变投影坐标：`p_Ct=T_Ct_from_W * p_W`，不凭相机运动给p_W增加对象速度。若旧点以Ct-1保存，先变到W再到Ct。对新出生点用当前geometry初始化，对越界/负深度/遮挡保留不同support原因。投影得到forward correspondence，不能直接喂grid_sample；选择已有forward_splat+z-buffer或显式构建反向网格，分别数值验证多对一与空洞。

几何carrier用于运输锚点ID/坐标、潜在状态与其协方差一起随ID迁移，不对covariance直接做任意线性插值却忽略相关性。首版跟踪锚点一对一搬运；新关联重初始化或明确合并规则。常值latent field恒等传播与单位坐标测试必做。

## 训练、指标与退出

每子阶段默认100k–300k、effective batch16、16 burn-in+32训练，先冻结S1。S3b/c先固定S3a转移做解析单因素检查，再匹配重训；固定权重与重训结果分表。L_prior、多步prediction、posterior geometry与NLL同S2，GT在目标侧。

逐任务输出：预测前位置/深度/姿态误差，固定与腕部相机分开；控制时间偏移曲线；纯camera误报；p_W外部速度；轨迹缺失与恢复；FK/投影/控制完整成本。未启用事件头时用固定基础诊断分数，不临时引入S4模型。

关键测试包括相机绕静态目标旋转后对象世界速度为零；手臂移动但外部相机不动时相机T恒等；动作经过限幅后历史值一致；current图像和future动作替换不影响prior；已发生动作/pose改变产生可追踪响应；迟到观测重新传播只回放已发生控制。

复用 `owac.training.n04_full --mode profile/train` 与 `owac.evaluation.n04_full`，每子阶段一个独立run和handoff子记录。S3出口要有可靠的执行动作/FK/三维投影链与同因素对照；无增量的控制项可以从主配置去除，但论文主张同时收缩，不能隐藏结果。

## 重点参考

[Ac-RKN论文](https://proceedings.mlr.press/v155/shaj21a.html)、[MTS3 acRKN入口](https://github.com/ALRhub/MTS3/blob/master/agent/worldModels/acRKN.py)、[SE3-Pose-Nets](https://arxiv.org/abs/1710.00489)、[Kornia深度源码](https://github.com/kornia/kornia/blob/4fcfcf863e0a9e2b3e93179c3fa96bcf278ce3c6/kornia/geometry/depth.py)、[Monodepth2](https://github.com/nianticlabs/monodepth2)。后两者的采样/内参约定必须转换为本项目align_corners=False、zeros和像素中心合同。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

# S4 测量误差与外部事件证据开发提示词

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

## 目标与固定项

固定S3测量/转移，区分“相机/时间/增益误差可以解释的残差”与“仍支持物体动力学变化的残差”。例子：曝光翻倍时均值/方差按比例变换，不据亮度差给杯子赋速度；相机时间错了几毫秒导致残差时，应增加对应方向的不确定性，但真实杯子反向仍要能被检测。

四个子阶段独立推进：S4a增益→S4b噪声→S4c几何/时间误差→S4d事件证据。正式主更新先采用有先验的边缘化，nullspace硬投影只作为单独证据对照，不在一次更新上重复计算同一残差。

## 源码与交付位置

读 [现有 gain helper](../../../src/owac/frameworks/n04/innovation.py)、[bilinear_variance/residual_score](../../../src/owac/evaluation/n04_raw_diagnostics.py)、[噪声诊断测试](../../../tests/encoders/n04/test_raw_noise_diagnostics.py)、S0的坐标与S3的几何载体。

拟新增 `frameworks/n04/uncertainty.py`、`event_evidence.py`；扩展 `measurement_geometry.py`、`research/theory/n04_v1.md`；配置 `s4a_gain.json/s4b_sensor_noise.json/s4c_geometry_uncertainty.json/s4d_event_evidence.json`。旧innovation分数语义保留，正式事件schema独立。

## S4a 成像解释

在黑电平校正、线性RAW参考上使用metadata给出的已知曝光/增益变换，不能直接对任意非线性ConvNeXt特征乘g。metadata不充分时先用全局正增益与偏置、跨帧缓慢变化的先验，默认仅全局参数；低频4×4残余gain作为第二个配置。空间支持来自已验证几何/可见性并记录选择，不能用GT物体mask挑背景。

同一残差拟合gain再检验时存在选择/自由度变化：用独立背景支持拟合、在其余支持验证，或在开发集校准整个流程；不要直接套未拟合的chi-square阈值。大局部阴影/材质变化保留unknown；不能让自由gain field消除任意对象位移。

先固定真值gain诊断，再估计gain，其他模型/噪声不变。输出gain误差、静态残差、外部事件召回和下游位置误差。

## S4b 传感器噪声

RAW Var=alpha*signal+beta，alpha/beta来自训练标定或明确合成参数。若归一化x'=(x-black)/s，则variance'=variance/s²；乘性gain后variance乘g²。历史和当前都带噪，差分含两者方差，重用同一历史还需相关性声明。

对线性采样W，用full reference `WΣWᵀ`；生产近似仅单点方差采用 `sum(w_i²*var_i)`，与直接warp方差不同。不要漏掉边缘权重/invalid邻居/饱和。跨像素相关性不保存时记录diag近似并用聚合统计测误差。先Monte Carlo独立传感器噪声验证，再估计alpha/beta；固定全部阈值和测试组。

RAW variance只给RAW残差，不塞进latent R_z。潜在/几何测量各在自己的单位下校准；若使用传播Jacobian，给出具体映射和近似。报告每噪声组的FPR/recall/coverage/width，保留旧高噪声组召回下降这个已知风险。

## S4c 几何和时间协方差

首版在6维几何状态上实现完整dense reference，再优化。定义 e=x-mu_minus、eta为小位姿/时间/标定/gain误差，r=y-h(mu_minus,eta_hat)：

```text
S = H P H^T + J Sigma_eta J^T + H C J^T + J C^T H^T + R
K = (P H^T + C J^T) solve(S, I)
```

代码用Cholesky solve，不显式inverse。固定离线标定且独立的首版C=0需在配置声明；若eta也由共享历史在线更新，使用joint [x,eta] 和cross covariance。首版主线固定校准先验/边缘化；online joint calibration只作为有跨锚点共享相关性记录的单独对照，不能给每个锚点独立校准后当独立证据累积。

对joint state用H_joint=[H,J]和Joseph更新；写理论时Joseph中的P/H指joint系统。不要将非零C的情形仍套只包含x的独立噪声公式。几何非线性Jacobian用FP64中心差分验证旋转/平移/时间；增益量纲、dt(s)/rotation(rad)/position(m)分别固定。哪个误差已进P就不再同项加R。

至少四基线：固定R、相同容量learned R、显式传播、可用条件下的joint calibration。learned R也包含logdet训练/独立校准，不能用一个故意失调的固定阈值充当最强基线。

## S4d 可辨识事件

分别输出motion_presence和prediction_mismatch，不以softmax四类强迫所有原因互斥。事件为对已有预测的反证；匀速移动且被预测准确可以presence=1,mismatch=0。标签从生成过程/人工规则产生，不由自己的预测阈值全部伪造。

边缘化分数采用与上述S一致的NLL/NIS，同时保留support。硬投影对照：R=L L^T，Jw=L^-1 J；用SVD/QR求U使U^T Jw=0，e_perp=U^T L^-1 r。投影残差covariance还应包含投影后的HPH^T与适当相关项，不能白化R后自动认作单位总方差。rank tolerance写入配置，空补空间不是“检测到零变化”。

外部运动Jacobian Jd的可检维数由rank(U^T L^-1 Jd)描述。物体运动与nuisance同向、全画面移动平面、无尺度、关联错误等输出identifiability/unknown reason，不给高置信原因。unknown单独计覆盖，不能移出分母后虚增准确率；事件ROC、固定FPR召回、unknown频率一并报告。

## 训练与必做数值实验

每小项先冻结S1/S3参数，只校准对应noise/error head或拟合统计，初始30k上限100k；解析项无需为了凑步数训练。每项成立后才匹配重训一组，固定与重训表分开。开发集目标误报1%确定阈值，测试不能重选。

理论交付四项：PSD条件；局部一阶nuisance消除与重叠方向丢失；有限时域误差积累（L可大于1）；gain与采样variance尺度。每项写假设、推导、小数值对照、反例与不适用条件。Joseph保持代数PSD不等于经验后验正确，rank证明不等于任意场景可分离。

测试：随机PSD joint covariance含非零C时dense Kalman参照；错误忽略C的反例；SVD退化与空nullspace；finite-difference Jacobian；bilinear+gain噪声Monte Carlo；所有masked项NaN毒化；残差依赖gain/association选择后的经验覆盖；same measurement两次调用不能减小covariance。

H1门槛使用S0协议：标定/时间负例相对最强误差基线FPR降至少20%，外部事件recall下降≤2个百分点且配对区间支持方向；同时检查位置和修订代价，不能通过把R无限增大压掉所有更新。未达门槛仍保留误差建模工程结果与失败组，候选H1不通过。

复用正式runner逐项profile/train/evaluate；新增theory/Monte-Carlo诊断入口到evaluation包，交付S5使用的残差、covariance、identifiability与事件API，字段/单位明确。

## 重点参考

[OpenVINS nullspace原理](https://docs.openvins.com/update-null.html)、[测量更新](https://docs.openvins.com/update.html)、[时空标定可观测性](https://arxiv.org/abs/2201.09170)、[Poisson–Gaussian噪声](https://webpages.tuni.fi/foi/sensornoise.html)、[DSO AffLight源码](https://github.com/JakobEngel/dso/blob/7b0c99f01d238f801c625beaff90240bcb007198/src/util/NumType.h)、[FilterPy Joseph参照](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/kalman_filter.py)。DSO/OpenVINS作为估计原理与离线参照，不直接复制GPL系统到主链。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

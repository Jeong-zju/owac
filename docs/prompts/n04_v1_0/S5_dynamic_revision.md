# S5 外部变化修订动态记忆开发提示词

> 历史文档（2026-10-15）：后续开发已由 [N04 v1.1](../n04_v1_1/README.md)取代。原文与旧启动语仅供追溯，不作为当前执行指令；当前入口为 v1.1 S2，旧数据扩展与衍生权重不进入官方数据主实验。

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

## 目标与前置

杯子原来向右，最新有效测量显示它停止或反向后，下一次看不见时仍应按修订后的状态预测。只提高事件分类分数、只给token拼一个change特征都不完成S5。

读取S0–S4 handoff，固定S1测量/S3转移/S4误差参数与阈值。先实现有物理单位的几何版本，再单独加几何→潜在桥接；本阶段不要同时重训视觉主干。

## 文件与现有参照

拟新增 `frameworks/n04/revision.py`、`evaluation/n04_revision.py`，扩展正式state/transition/full_model；配置 `s5a_oracles.json/s5b_modes.json/s5b_bridge.json/s5c_motion_gate.json`，条件满足时再创建正式motion配置。旧 [model.py](../../../src/owac/frameworks/n04/model.py) 的change路径与 [同测量诊断](../../../src/owac/evaluation/n04_tracking.py) 用来理解历史问题，不把旧change copy进新系统。

## S5a 单环节诊断

同一测量、mask、Q/R和更新算法，分开替换：当前可见时的GT mismatch事件；GT当前位置；最近可见GT速度；GT关联。GT事件只提供“现有动力学失配”一位，不带新速度/方向。隐藏期间所有GT速度/位置/事件输入置NaN，只允许evaluator读取目标。记录诊断替换位置与普通路径逐值一致部分。

默认Ac-RKN、Ac-RKN+自适应Q、标准三模式切换先完成。自适应Q baseline采用同一过程噪声方向，开发集选择有限倍率 `{1,2,4,8}`；触发由S4相同创新分数，不能给N04更好的测量或更大的选择预算。

## S5b 三解释更新的确定公式

模式：continuation、dynamic_change、nuisance_error。continuation允许物体按已学速度持续运动，绝不等同static。对每个锚点保留三分量(μ_j,P_j,w_j)，一次当前测量在每个互斥解释下评价一次，然后按混合模型合成，不将三份似然当三条独立证据相乘。

```text
Q_cont = Q0
Q_dyn  = Q0 + Bdyn Lambda_dyn Bdyn^T
Sigma_eta,error = Sigma_eta,base + Delta_eta
logw_j = log(mode_prior_j) - 0.5*(r_j^T S_j^-1 r_j + logdet(S_j) + d*log(2*pi))
w = softmax(logw)
mu_mix = sum_j w_j mu_j
P_mix  = sum_j w_j [P_j + (mu_j-mu_mix)(mu_j-mu_mix)^T]
```

Bdyn首版只作用于物理速度3维；过程噪声通过位置-速度传播形成对应交叉块。Lambda正值、有开发冻结范围。nuisance_delta只在S4所建位姿/时间/gain方向扩展，不能用任意像素位移基解释局部对象运动。invalid测量不做模式似然更新；不可辨识时保留unknown与模式分歧。

模式预测使用3×3 Markov转移矩阵，初始对角0.95、另两项各0.025；三分量是持久filter bank，按标准IMM interaction分别混合条件先验（包含分量均值差的covariance），然后预测/校正/模式更新。先与标准IMM low-dimensional数值对照，避免每步把所有模式重置为完全同一分布导致没有模式记忆。参数是否学习、lambda范围与先验敏感性是开发配置，普通IMM baseline共享可用信息与同样3模式预算。

三个模式的几何prior都在看到当前测量前产生；当前观测只用于correct/模式权重更新。Qdyn在对应模式的predict中生效，不能看到大残差后重写先前记录的prior再作为预测成绩。初版状态增加 `mode_x[B,N,3,6]`、`mode_P[B,N,3,6,6]`、`mode_weights[B,N,3]`；每锚点共享一个512维RKN，几何三模式与潜在单状态分别标明，不宣称二者是精确joint多模式滤波。

修订按对应物理变量写回：dynamic分量的posterior速度/位置影响下次传播；nuisance分量修订误差量或其不确定性，不能虚构对象速度。小几何例子中先验证速度更新与下一次位置差；没有显式接触传感器/标签时首版不硬造接触模式，只在D1可观测接触扩展中单独加入。

保留最多3模式，而非直接扔掉分量只存mean。readout传递每模式值/权重/不确定性给固定容量resampler；不能把双峰中间位置当唯一确定目标。构造左/右运动等概率时均值落在不可行位置的反例，比较mean collapse与有限模式保留的状态/动作相关误差和成本。

## 几何与潜在状态桥接

几何版本确认后新增一个独立因素：`delta_m_l = w_dyn * MLP([innovation_g, corrected_velocity-prior_velocity, support, logdiag(Pg)])`，MLP两层128宽SiLU→256、最后层零初始化；输出修订潜在后半均值，并由更新后的m参与下一次RKN转移。坐标和速度先按训练统计标准化。几何与latent来自同一测量，不再相乘似然；上述为学习桥接而非精确joint filter。

这里的速度差与Pg取dynamic_change分量校正前后对应量，w_dyn取当前后验模式权重；continuation/nuisance的几何分量仍各自保存。损失同时检查latent解码与几何分量输出，不能用只改变几何旁路的结果证明latent路径已修订。

桥接前后RKN covariance需明确：首版将桥接视为额外状态控制，配置独立正值Q_bridge并经验校准；不把确定性均值修改后的旧covariance称精确后验。对照无桥接、同容量无模式残差MLP、完整支持/误差约束桥接。报告几何结果与latent读出结果，不能只在一条旁路上改数字而π0.5仍读旧state。

L_revision主要监督首次新证据后的未来多步geometry与stop/reverse积分误差；L_event为辅助，初始权重0.1。桥接MLP/模式lambda训练30k–100k起，几何/latent修订损失权重1，多步prediction权重0.5；先只训新参数，再单独匹配重训状态转移。threshold继续S4冻结。

## 必做机制干预和条件运动网络

冻结同一权重，在首个有效反证出现后：①只禁用写回下一状态，保留同样事件输出/读出；②保留动态均值修订但禁Q变化；③保留Q变化但禁均值修订；④将事件score置乱且其余相同。比较下一步及多步prior，而不只比较当前posterior；日志保存修订前state、写回量、下次predict实际读取值。对未来prediction loss到revision参数求有限非零梯度，独立有限差分/关闭路径核验。

H2门槛：停止/反向积分误差较Ac-RKN降低≥15%，missing段退化≤5%，revision latency不退化且配对区间支持方向。只改善presence分类不通过。

S5c只有GT运动在新的困难三维/操作任务通过S0预注册工程门才启用。旧二维GT速度额外0.02855px未达0.05px，不能据此启动大网络。条件不满足，交付 `motion_training_decision.json`、诊断与未启动原因，即该条件分支完成，不伪造Dynamo实现。

条件满足时首选完整RAFT作当前帧对应，若缺深度/动态分解再按完整Dynamo-Depth分阶段配置；不要两个大网络一次同时加。核查原作者任务的最小忠实运行、固定commit/组件license，然后分步RAW输入适配、units/grid转换、深度→ego→外部运动训练。current图像依赖的flow只进correct，隐藏期间只用最后确认动态状态。匹配对照包含“全部残余流用于补偿”与“按支持/误差解释后修订”，总算力分开计。

复用正式runner profile/train/evaluate，revision evaluator增加模式/路径干预矩阵。出口包含真实下一步state响应、同信息强baseline、负例/unknown、多峰反例和运动网络决定。

## 重点参考

[IMM数值实现](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/IMM.py)、[Ac-RKN](https://proceedings.mlr.press/v155/shaj21a.html)、[在线变化点原理](https://arxiv.org/abs/0710.3742)、[RAFT](https://github.com/princeton-vl/RAFT/tree/2888e15a51fa41140771d3f498ed8023cff098d1)、[Dynamo-Depth Trainer](https://github.com/YihongSun/Dynamo-Depth/blob/227a5d938ddf9a3aba5fcf7a03c74384284b31d0/Trainer.py)、[Dynamo采样函数](https://github.com/YihongSun/Dynamo-Depth/blob/227a5d938ddf9a3aba5fcf7a03c74384284b31d0/tools.py)。模式切换本身为已有方法，新增主张限定于受测量支持与nuisance方向约束的修订规则及实验增量。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

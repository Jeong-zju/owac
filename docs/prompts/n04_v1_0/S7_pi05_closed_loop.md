# S7 真实 π0.5 训练与闭环开发提示词

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

## 三个独立出口

S7a验收真实模型数据/梯度/状态合同；S7b完成正式训练；S7c完整闭环。顺序执行，每项单独报告。已有真实tiny smoke不等于正式N04训练完成，动作loss下降不等于机器人任务改善。

输入S6固定主配置、三个完整训练seed、候选/对照checkpoint、数据拆分和预注册协议。阶段启动时读取远端最新基线报告：`/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1/complete-summary.json`、`identity-audit.json`、`checksums.json`。准备轮读到13,250/13,250及632/632已完成，LIBERO58.20%、RoboTwin15.63%；不根据N04旧文档重新跑旧基线。审计仍需确认manifest、配置、权重及修复范围一致。

## 现有真实代码必须先读

- [RawVLAFeaturePolicy](../../../src/owac/backends/rawvla_features.py)：`encode_observation/_view/embed_prefix_from_visual_tokens/forward_from_observation/sample_actions`。
- [真实权重smoke](../../../src/owac/backends/rawvla_feature_smoke.py)、[服务](../../../src/owac/backends/rawvla_feature_server.py)、[后端构建/归一化](../../../src/owac/backends/rawvla.py)。
- [fixture桥接测试](../../../tests/encoders/n04/test_feature_bridge.py) 只支持CPU接口检查，不能取代真实权重。
- 远端独立StarVLA checkout `/workspace/owac/rawvla/source/starVLA` 内的 `starVLA/model/framework/VLM4A/PI0.py`（主要数据流）、`PI05.py`、`OpenPIPaliGemma.py`、`OpenPI_ActionHead.py`与两平台配置。参见[固定链接导航](references.md)，核对远端实际已修补文件而非追随上游main。

拟新增 `backends/n04_full_adapter.py`、`training/n04_policy.py`、`evaluation/n04_policy.py`、`runtime/n04_policy_server.py` 与`configs/n04/full/s7_*`。现有桥接可复用，但明确适配正式模型，不靠反射伪造后端不存在的方法。

## S7a 前缀、梯度与在线状态

1. 定义正式adapter，把现有bridge所调用`step(values, context, state, **kwargs)`和`readout_tokens`映射到N04FullEncoder的MeasurementFrame/telemetry API。明确旧`EgoMotionContext`与新ExecutedInterval转换，必要时扩展bridge接受已编码VisualTokens；不要在去噪循环内再次step。
2. 每相机256×2048视觉前缀。完整当前空间特征与最多128个动态锚点组成读出输入，用256个query的单层cross-attention resampler（8 heads，输入先投影2048，输出MLP）作为首版。无记忆/RKN/Ac-RKN/完整系统共享读出结构与容量；没有历史的对照只用当前场，缺失支持显式mask。模式分量/权重作为状态输入，不能只传三模式均值。
3. 多相机稳定顺序，LIBERO agentview/wrist，RoboTwin head/left/right。每相机维护独立state与measurement ID；prefix按固定相机顺序拼接。原language embedding sqrt(dim)缩放、language padding、attention block、position cumsum语义沿用真实PI0。不提供假全True mask。没有任何视觉支持时的输入和缺失标记在各系统一致。
4. 冻结π0.5参数用requires_grad_(False)，train forward保留梯度经VLM/action expert回到encoder/readout；不能把整个后端no_grad包住。inference才no_grad。报告每个可训练模块非零有限梯度、参数更新，后端冻结参数hash不变，model视觉塔确实未调用。
5. 实际LIBERO外部动作7维/horizon10，当前8维state；RoboTwin14维/horizon50，当前14维joint state。后端内部pad32维只属模型尺寸。LIBERO t开始动作监督、RoboTwin t+1开始，尾部padding mask与实际control_dim维度都从loss中排除；规范化用各自权重指定stats，不能混用DROID。官方10次denoise与执行长度/重规划间隔不同。
6. 语言tokenizer/state归一化、backend_state与运动delta区分。固定输入noise/time做真实forward，分别核对velocity target、suffix位置、mask、action unnormalization和输出形状；不能只检查loss finite。
7. 每条新相机观测一次measure/correct；10次denoise读取同一prefix。动作块只把执行器实际下发且执行确认的前缀写回telemetry，剩余计划不进history。episode/reset/环境异常/部分batch reset都清理对应状态与cache。同一frame ID但新episode不能复用旧token。
8. 迟到图像按曝光时刻校正历史后传播到控制决策时刻，记录等待、计算与输出时间；fixed-lag的等待不伪装成实时即时预测。读出无当前帧时仍可以输出有效历史预测，但带age与uncertainty。

两平台真实NPZ、真实权重各至少一个完整梯度更新/保存重载/实际采样；再做连续多次policy调用与reset、重复ID、部分缺失、延迟，逐步重算encoder_calls。旧smoke脚本硬编码tiny，你必须加正式full config/checkpoint入口或写新的full smoke，明确输出model_kind与实际参数量。

远端权重/统计/tokenizer路径按remote_execution.md；后端先以原Python3.10环境执行，代码需兼容torch2.7.1。新增纯encoder依赖若与原后端冲突，建N04独立后端环境并做数值一致性对照，不能直接升级冻结服务。

## S7b 正式训练

两平台分别训练；训练轨迹使用S0固定split。官方训练数1771/632是盘上文件计数，按源身份采样，不从测试manifest拿示范。先统一冻结π0.5，训练encoder/state/readout；再为全部主对照开启同一后端适配档。首版第二档固定使用action expert注意力q/v投影的LoRA：目标模块 `action_head.gemma_expert.model.layers.*.self_attn.q_proj` 与 `.v_proj`，rank=16、alpha=32、dropout=0；A按常规随机初始化、B零初始化，基础权重和语言侧保持冻结。先按实际named_modules展开精确清单、维度和参数量，加载原权重后再注入，校验零初始化前后forward一致、LoRA梯度/更新与保存重载。所有主对照使用同一清单和训练预算。PI0源码本轮已确认经这些q/v层调用action expert；运行时仍核对具体模块数。全量微调仅作为有开发证据后对所有主对照同等开放的第三档，不由N04独享。

系统最少无记忆、Ac-RKN、完整N04，各seed17/29/43；每个平台每系统每seed100k更新起，必要时200k+。学习率encoder默认3e-4开发网格1e-4/3e-4/1e-3，后端适配候选1e-5/3e-5，选择机会相同；正式容量、累计实际数据访问和GPU小时分列。effective batch从真实profile决定并在比较前冻结，序列训练保留burn-in和状态、动作监督不穿越episode。

主目标保持真实π0.5 flow-matching；首版总loss为action 1.0 + S6已归一化geometry/predict/revision辅助各0.1，辅助权重在开发集均等选择，记录去辅助损失对照。action loss的输入使用当前posterior，future action只作target。评开发action loss之外，保留S6机制probe和小规模独立开发闭环，防止动作训练破坏动态分离；最终官方测试不参与checkpoint选择。

保存完整RNG/optimizer/scheduler/scaler/sampler和module trainability；远端run表显示当前step/epoch等效覆盖/近期曲线/资源/下一次评价，作业管理沿S0。训练到100k仍未趋稳就延长或标未充分，不能用20步NPZ smoke或200步toy代替。

## S7c 最终完整闭环

用冻结LIBERO10000和RoboTwin3250主清单，每个系统每个训练seed都跑完整。三个系统×三个seed×13250=119250次主rollout；632原条件核查另计，若所有九模型都跑则另5688次，不能混进13250分母。新增强baseline会增加总次数，开跑前plan列清并profile估算，不把最低数量当全部研究成本。

每个run_id绑定系统/seed/权重hash、平台manifest、task/initial_state/condition/环境seed/语言seed/action_noise RNG。相同初态和外部随机条件配对，策略不同产生物理轨迹不同，不能要求逐像素相等。基础设施异常保留身份与原分母，修复补跑前incomplete；策略达到max steps是正常失败，不能重抽幸运seed覆盖。主清单与632原条件各自完整审计，复用冻结基础设施兼容策略但不修改其原记录。

独立机制闭环添加遮挡、延迟、错/过期历史、对象停止/反向；单列新manifest与成功判据。如果这些扰动改变正确动作，由新的执行策略/示范评价，不拿原专家动作当唯一正确答案。官方成绩与扩展挑战分别报告。

H3预定目标：相对同骨干Ac-RKN主要成功率总体至少+3个百分点、配对95%区间下界>0，关键控制组在S0容忍度内；两平台分别给分母与效果，不用随意合并加权掩盖失败。任务分层/初态分组/训练seed统计沿S6，间隔过宽写证据不足。只在扩展挑战有收益就限定结论。

完整成本：曝光→接收→编码→关联/传播/校正→读出→π0.5→通信→动作实际下发p50/p95，CUDA同步，warmup独列，在线batch1；状态/峰值显存、控制频率、执行长度、重规划间隔、迟到帧、discarded计划、复用prefix次数、训练GPU小时。若延迟不同影响任务，增加等延迟对照。

## 要实现并执行的运行接口

以下拟新增CLI与配置路径需落地，并在远端按阶段实际运行：

```bash
$N04_BACKEND_PYTHON -m owac.training.n04_policy --config configs/n04/full/s7_libero.json --mode contract --run-dir "$N04_RUN_DIR/contract-libero"
$N04_BACKEND_PYTHON -m owac.training.n04_policy --config configs/n04/full/s7_robotwin.json --mode contract --run-dir "$N04_RUN_DIR/contract-robotwin"
$N04_BACKEND_PYTHON -m owac.training.n04_policy --config configs/n04/full/s7_libero.json --mode profile --run-dir "$N04_RUN_DIR/profile"
$N04_BACKEND_PYTHON -m owac.training.n04_policy --config configs/n04/full/s7_libero.json --mode train --run-dir "$N04_RUN_DIR/train"
$N04_BACKEND_PYTHON -m owac.evaluation.n04_policy --config configs/n04/full/s7_eval_matrix.json --mode plan --output "$N04_RUN_DIR/eval-plan.json"
```

evaluate的run/summarize模式、任务恢复/分母审计也要实现，但必须在正式训练/冻结后使用。仿真进程使用已有对应suite环境，策略服务使用已验证backend解释器；RPC记录同一run身份与observations/actions时间，不让仿真环境和训练环境互相覆盖依赖。

最终交付系统及完整结果、独立机制反证、公平性/成本、三种子统计和未通过主张。全量开发完成可以包含负结果；若机制或任务未获益，交付完整可复现系统并明确结论，不把工程成功写成科学成功。

## 重点参考

[真实PI0入口](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py)、[PI05](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py)、[动作专家](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py)、[语言/视觉前缀](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/vlm/OpenPIPaliGemma.py)、[官方openpi固定实现](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py)、[数据卡](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)。固定部署源码是执行依据，上述资料帮助理解接口，不自动授权切换后端版本。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

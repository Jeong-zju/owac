# S2 空间 RKN 开发提示词

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

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

## 阶段目标与前置

固定 S1 测量，只增加动态状态。例子：杯子连续移动，连续三帧不可见时，模型利用过去状态继续预测；重新看见杯子已经停止时，必须同时报告旧预测造成的滞后。S2 证明动态路径的作用，不声称已经解决 H1/H2。

读取 S0/S1 handoff，固定具体测量 checkpoint/cache hash、训练/开发/最终拆分。没有正式 S1 cache 时可用诊断 fixture 开发数值代码，但正式验收保持未完成。

## 复用入口与拟新增文件

- [作者 cell](../../../src/owac/baselines/_vendor/rkn/cell.py)：`forward/_masked_update/_update/_predict/get_transition_model`，[来源](../../../src/owac/baselines/_vendor/rkn/provenance.json)。
- [RKNPositionTracker](../../../src/owac/baselines/n04_tracking.py)：只参考当前 posterior 与 next prior 的时序。
- [同测量 evaluator](../../../src/owac/evaluation/n04_tracking.py)、[状态干预测试](../../../tests/encoders/n04/test_tracking_baselines.py)。
- 拟新增 `frameworks/n04/state_estimator.py`、`association.py`、`full_model.py`，扩展 S1 `contracts.py/full_config.py`；`baselines/n04_full.py` 放同信息 GRU、hold-last，训练/评测共用 S1 runner；配置 `s2_rkn.json/s2_gru.json/s2_hold.json`。

## 首版确定实现

1. 每相机128锚点、obs256/state512。共享作者转移参数，展平 `[B,N]` 为样本轴；作者默认 basis15、bandwidth3、系数网络[64,64]，never_invalid=False。vendor 文件保留；外围薄 wrapper 提供单独 `predict`、`correct`，证明其合成与原 cell.forward 数值等价。
2. `m=[m_u,m_l]`，cov 三块 `[P_u,P_l,P_s]`；H=[I,0]。初始化于第一条有效观测：m_u=z、m_l=0，P_u=测量方差，P_l 使用开发配置的正值初始记忆方差、P_s=0；不能在未出生/无观测锚点上产生有效位置。首个 posterior 的 prior 指标单独标初始化，不能把当前测量初始化后称为预测。
3. 先固定相机和明确对应坐标，验证实际普通 RKN，动作、本体、gain、事件都关闭。给定同一 measurement 和 mask，方法只能在历史状态规则上不同。
   首版使用 S0 可核实的等物理间隔时间栅格；缺失 k 个 tick 就逐 tick predict k 次，不把整个 gap 当一个常量步。普通 RKN 和 GRU 共享相同时序，Δt 在此档为同一常数。变间隔的转移扩展留 S3 单独验收，不声称作者原 cell 已经支持任意 Δt。
4. 然后加入局部缺失与多目标关联。`AssociationResult` 包含 persistent anchor ID、当前候选索引、gate distance、descriptor score、matched/unmatched、出生/消失。门控与双向最近邻按共享合同；GT 关联只在 oracle 分支。分配新 ID 时新建全部 covariance/time，不能继承旧 occupant 的记忆。
5. 无当前观测时只 predict，协方差随传播演化，last_observed_time 不变。部分锚点无效与整个 batch 无效都支持；先 where 清理 NaN 再进 cell。readout 不能把有历史支持的遮挡状态全部 mask 掉，也不能把未初始化位置当有效。
6. 独立物理读出 head 将潜在 posterior 解码为三维位置/深度；prior 使用同一 head。潜在 variance 不直接叫米²，单独训练/校准几何输出方差。S2 不增加显式 action/geometric transport 分支。

## 对照和训练

三个对照共享冻结 cache：hold-last、单层 GRU(hidden512、相同输入/有效性/Δt/读出头)、正式 RKN。GRU 没有当前测量时使用显式 missing embedding 和同样 Δt，不偷偷读缺失值。hold-last 不拥有学习动态参数，但读出/当前测量相同，完整参数/成本另报。二维子任务可新增隔离数值 KF；不恢复共享 FilterPy stash，不将诊断 KF 冒称复杂空间模型。

默认100k更新、有效batch16序列、16 burn-in+32 loss、TBPTT16。cache 中 z 固定。训练损失初版：1.0*posterior geometry +1.0*prior geometry +0.5*多步 open-loop geometry +0.1*测量预测NLL。多步 horizon `[1,2,4,8]` 按物理时间记录；分支从同一 posterior 独立 rollout，未来帧只在 loss 处读取。R_z 可先继承固定值再在 train/dev residual 校准，做成单独配置因素。不能只训练可见段 posterior。

缺失片段按真实时长采样；连续运动、停止/反向、遮挡中的随机变速都覆盖。看不见且无额外线索时的随机变化不可提前知晓，报告不确定性和恢复，不把无法预测的未来 GT 当输入。正式RKN/GRU三个状态seed，固定S1缓存身份清楚标出。

## 必做实验与出口

- low-dimensional dense Kalman 参照校验均值/协方差，包括有侧协方差、缺失、batch/anchor mask；高维检查块行列式和梯度。
- `cell.forward` 对照 wrapper：输出 current posterior 和 next prior 分别相等；调用日志证明每间隔一次 predict。
- NaN毒化所有隐藏测量及隐藏GT；预测/梯度不依赖它们。只从 evaluator 给出的目标计算 loss。
- 在缺失开始时清零后半latent、清空全部状态、换其他episode状态，冻结参数；下一步 prior 应有可测影响。干预元数据完整，不能用重新训练代替干预。
- 只替换当前测量、最后可见速度或关联，各自独立oracle，不同时换mask/Q/R/读出。隐藏阶段不给真值速度。
- 报完整可见段误差、缺失时长曲线/面积、停止/反向积分误差、首次重现修订延迟、coverage/宽度、anchor ID切换、参数/显存/在线p50/p95。

S2动态门：相对缺失误差至少降10%，二维绝对≥0.05px；三维用S0冻结绝对阈值，并检查可见段容忍度。未达门槛先诊断 measurement/association/capacity/训练充分性；完成系统和负结果，不靠更改门槛。

复用 S1 CLI，以 `s2_rkn.json` 等执行 profile/train/evaluate，并新增 `--mode intervention` 读取冻结 checkpoint 与 intervention config，单独保存结果。交付 S3 所需稳定state schema、权重、完整参数规模、cache/split hash、因果与数值验证及动态收益/滞后表。

## 重点参考

[RKN论文](https://proceedings.mlr.press/v97/becker19a.html)、[固定作者RKNCell](https://github.com/ALRhub/rkn_share/blob/7cb29e1c57307a87729fd5e20ca60a22f5798da5/rkn_cell/RKNCell.py)、[FilterPy数值参照](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/kalman_filter.py)。作者RKN的结构化近似不保存全部空间相关性，报告这一实际建模选择，不扩写稳定性保证。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

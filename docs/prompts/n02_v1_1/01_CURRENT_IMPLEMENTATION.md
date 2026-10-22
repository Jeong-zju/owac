# 实现状态更新与提示词编写前审计

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

## 2026-10-14 执行入口更新

P0 已经独立修补验收：实现提交 `5ddfdb569528ff38075fb976cf8c3ae087fb2af0`，归档提交 `861c46f67fff552c81fff89c5116c63b815224c5`。现有 `n02_full` 包、显式数据分区、参考算子与环境锁均已实现；17 项 parity 通过，真实数据检查见[独立验收报告](../../../research/experiments/2026-10-14_n02_p0_independent_acceptance.md)。这些是此前验收收据，本次提示词修订未重跑模型测试。

用户指定全部阶段优先在 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080` 开发。当前已只读核实远端 RAW/权重存在，但正式远端 Git worktree 尚未创建；按[远程开发说明](03_REMOTE_DEVELOPMENT.md)完成接入后续做 P1。本地 P0 GPU 测量不代表远端性能。

**下文全部为 2026-10-13 历史快照。** “n02_full 不存在”“本机缺 RAW”和“未运行上游参考”只描述当时，不覆盖上面的最新状态，也不作为重建 P0 或跳过远端验证的理由。

核查日期2026-10-13；N02 HEAD `843be674c35e8e058a364154e1c27a4b26ded2e4`。核查包括当前未提交v1.1方案与边界诊断，实际代码、结果文件及定点本地回归。本文是本地源码/资产核查，没有连接远端确认进程状态，也没有重训历史模型。

## 1. 仓库实际位置与变更

主checkout `/home/jeong/zeno/owac/repo`，HEAD `16075fca7c5e5cd126900bc62531a5733fe4e0be`，分支`lzh/pi05-sim-baseline`；存在大量与基线、历史路线及研究文档相关的未提交修改。N01/N04也有各自历史，本次不合并其实现。

N02 checkout已有未提交的README/status/候选卡、`docs/plans/`、反证边界实验及其脚本/结果。这些是本次依据，不需要先清空或强制提交。继续此checkout，新增`n02_full`。v1.1约618行的方案不是已经实现的代码。

## 2. 代码复用与缺口

| 实际文件/符号 | 已有内容 | 新阶段处理 |
| --- | --- | --- |
| [encoder_contract.py](../../../src/owac/core/encoder_contract.py)：MeasurementFrame/EmbodimentContext/VisualTokens | RAW与可用上下文、外部token契约 | 复用类型；新增bundle层协调多相机 |
| [rawvla_sequence.py](../../../src/owac/raw/rawvla_sequence.py)：RawVLAWindowDataset/normalise_actions/normalise_state | 原始数组、因果窗口、动作mask、CFA坐标与转换 | 复用数值；增加显式分区/锚点消费，禁止重新随机划分 |
| 同文件compute_split/_acquisition_source | 旧版train/validation；分组外键为LIBERO suite或RoboTwin task | P0新四分区需要源轨迹/场景来源盘点，不能直接改名称假装完成 |
| 同文件FORMAT_SPECS | LIBERO offset0，RoboTwin offset1；后者state_source_key=actions | 固定并审计真实字段；RoboTwin默认normalisation并非π0.5正式zscore配置，须显式传入 |
| [rawvla_features.py](../../../src/owac/backends/rawvla_features.py)：RawVLAFeaturePolicy | 跳过图像塔、训练flow loss、一次prefix多步denoise、冻结后端 | 复用；新版本化state/cache由adapter补齐 |
| 同文件EncoderReadoutTokenSource | 遍历每个相机，独立encoder状态和readout，再拼接 | E1可以作对照；正式对象路径新增WorldTokenSource，避免各视角互斥MAP拼接 |
| 同文件ActionChunkCache | episode/step缓存与reset | 新revision/model版本尚未纳入缓存key，须扩展或在新adapter管理 |
| [training/n02_raw.py](../../../src/owac/training/n02_raw.py) | 为smoke挑选train/validation各少量NPZ，单样本时间循环与真实π0.5 | 只作调用参考；正式batch、数据遍历、DDP与训练续跑另建n02_full入口 |
| [frameworks/n02/state.py](../../../src/owac/frameworks/n02/state.py)：MeasurementBuffer | 去重与bounded history、部分batch reset | 借鉴身份和回归，不直接当完整来源DAG；新账本需校准版本、回放与多相机身份 |
| [frameworks/n02/model.py](../../../src/owac/frameworks/n02/model.py)及readout/revision | 固定网格假设、小卷积/共享cell、旧局部反证调度 | 历史对照；不是Base/SAVi/Theseus/结构发现/ATMS实现 |
| [baselines/n02_measurement.py](../../../src/owac/baselines/n02_measurement.py) | 重训sbi CNN加首帧RAW差分 | 留作旧测量回归；不能把差分当独立帧似然 |

代码核查时不存在`src/owac/frameworks/n02_full/`。本地根`pyproject.toml`要求Python≥3.11，encoder extra的torch来自CPU index；旧真实StarVLA运行时为Python3.10，历史通过PYTHONPATH接入。P0必须建立隔离的CUDA兼容环境，而不是将根CPU锁当GPU部署配方。

## 3. 真正已有的证据

旧[真实接入记录](../../../research/experiments/2026-10-12_n02_encoder.md)证明当时版本的NPZ→旧N02→π0.5离线训练/推理连通与梯度，不是新Base或结构机制验收。

[重训评分器](../../../research/experiments/2026-10-13_n02_retrained.md)在声明合成条件下修复了部分测量能力；[局部修订对比](../../../research/experiments/2026-10-13_n02_revision_comparison.md)中，旧N02与简单事件缓存逐帧相同且约慢4%。[0014](../../decisions/0014-n02-scheduler-archive.md)归档该调度；v1.1允许新完整路线，不撤销旧负结果。

[新反证边界](../../../research/experiments/2026-10-13_n02_refutation_boundary.md)显示：两个分离亮点未被绝对拟合拒绝，2像素平移后方向正确数为8/11/8（各16）。所以P1要做独立预测、局部统计与成像干扰，不能照抄旧全图均值残差。

主checkout比本worktree多一项新归档：2026-10-13 16:42:30北京时间，RoboTwin原条件632条核查补齐，108成功/524正常失败，17.09%；见主checkout的`research/experiments/2026-10-13_rawvla_robotwin_infrastructure_repair.md`。它独立于3250条主分母。LIBERO 5820/10000和远端RUNNING字样均来自归档快照；本轮不将其当实时状态。

本次本地运行5个相关测试文件：**87 passed、4 skipped**；4项因本机缺真实NPZ或StarVLA/权重跳过。它验证已有数据/桥/状态/历史对照回归，不验证v1.1。命令和完整收据见[preparation_validation.json](preparation_validation.json)。

## 4. 上游源码已查到的具体移植问题

1. **FCMAE硬编码。** `models/fcmae.py`的patchify/unpatchify虽接收in_chans配置，内部仍写死3通道，mask还假设方形网格。CFA4与240×320输入必须改这两类假设，做矩形和多通道往返测试。稀疏encoder依赖MinkowskiEngine，普通dense零填充不等价。
2. **SAVi的mask。** `SlotAttention.__call__`丢弃padding_mask，标准注意力分支也存在未实现mask的路径。RAW坏点/缺失相机不能仅往函数传mask；移植要真正屏蔽有效key，保留slot维softmax再按有效输入归一化。
3. **GRU不只换名字。** Flax/PyTorch的参数布局、gate顺序和reset位置要逐项对应；先原参数/输入前向及梯度一致，再扩宽为512。
4. **dNRI时间泄漏。** `dnri/models/dnri.py`的DNRI_Encoder训练posterior使用forward_rnn和reverse_rnn；在线必须使用forward prior/single_step_forward。替换未来帧的测试必须不改变过去在线输出。
5. **GRN的全局性。** Base的GRN具有空间归一化依赖，不能把编码器写成“每块独立”；P5要计整帧编码开销和真实依赖。
6. **LM适用目标。** Theseus优化的是残差平方和；完整RAW目标含log方差/尾概率等。求解器给提议，外层完整目标接受，不把NLL平方。

固定源码与SHA见[参考导航](02_REFERENCES.md)和[source_manifest.json](source_manifest.json)。本次取回源码不意味着已完成上游运行复现。旧来源JSON的ATMS DOI数组被括号截断，本包引用完整转义URL，不沿用截断URL。

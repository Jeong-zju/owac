# 给DeepSeek：P6，π0.5正式离线适配

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

## 本阶段执行位置

优先连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，在远端 `/workspace/owac/worktrees/n02-measurement-completion` 开发、测试和运行。先完整阅读[远程开发执行说明](03_REMOTE_DEVELOPMENT.md)，按其步骤准备尚未建立的 worktree、解释器与 P0 资产；随后读取远端 AGENTS.md、README.md、docs/status.md。本文命令、相对路径及 `python` 均指远端 checkout 和对应阶段的独立运行时，GPU 命令显式使用 `"$N02_PY"`。

在远端 N02 独立同进程运行时完成两平台全部离线适配；直接读取服务器的 RAW、π0.5 权重、tokenizer 与正式 stats。数据 loader、模型、优化器及 checkpoint 都在远端，不通过 SSH 流式拉 RAW 到本地训练。

读[共同契约](00_SHARED_CONTRACT.md)、[E1/E2接口规范](EARLY_POLICY_PREFLIGHT.md)、P5通过的实际版本和原方案P6/8/9。复用早期适配代码，扩大真实训练覆盖与公平预算；不重写第二套token/flow入口。

主要修改`training/n02_full.py`、`n02_full/policy.py/readout.py`、`evaluation/n02_full/policy.py`和两平台配置。参考[固定RawVLA源码](02_REFERENCES.md#r25)，现有`training/n02_raw.py`只作历史连通性参考。

## 1. 固定正式输入和动作协议

从P0分区manifest消费真实全部训练数据，轨迹/照明同组；训练/开发/calibration与正式闭环初态分开。检查随机anchor覆盖、DataLoader worker/DDP shard和恢复位置，不能继续旧smoke的“两文件”逻辑。

LIBERO两相机，robot_states[t]8维，actions[t:t+10]7维，quantile及原夹爪转换，执行前5步；RoboTwin三相机，actions[t]14维作者关节代理，监督t+1开始50步，显式MEAN_STD，执行50步。内部32维padding和10次flow denoise不变。记录实际stats/checkpoint/tokenizer hash。

NPZ当前观测与动作对齐不等于真实连续rollout状态。随机window使用burn-in重建；P0已验收窗口为rnn_frames=1；本阶段从16帧burn-in起点建立显式窗口配置，在原分区和锚点身份上生成版本化表，经开发验证后冻结，只在锚点算动作loss。连续训练块可carry state，在TBPTT边界detach，episode/参数版本变更时按契约重建。

专家动作只用于TrainingTargets，历史动作context必须有真正的执行来源；无记录时保持不可用，不因动作训练启动而填入未来标签。

## 2. 统一读出与机制梯度

N02/B3使用一个完整世界MAP，相关所有相机view一致；B2相同Base/Perceiver结构和256×2048每相机预算。读出包含实体与测量细节两种token source，按配置标注；附object-only消融。多个世界若各调用策略则分开输出、单独计费，不混平均动作。

训练和推理共用`encode_bundle_to_tokens`。冻结后端仍保留对tokens输入的autograd。每次实际flow loss单独检查梯度链：Base、实体校正、可微几何/solver、readout；离散结构proposal继续使用P3参考监督，不能把非零代理梯度当真实动作梯度。固定不可微决策时对连续路径做有限差分/消融验证。

保留RAW/结构/关系辅助目标防action loss破坏测量与撤回。先固定原权重，再开发比较辅助权重整体 `[0.1,1.0]` 两档，按动作开发与P1/P3/P4机制保留共同选型，不按最终测试。每项loss激活样本和梯度冲突继续记录。

prefix在一次动作chunk采样内只建一次，修改revision在下一次规划必须使旧prefix/action缓存失效。**原基准主轨道保持固定5/50规划节奏**；提前事件重规划作为单独扩展，对全部方法提供同信息/同观测机会、计完整控制成本，不能混进原协议主分数。

## 3. A/B两条正式适配轨道

### A：冻结语言/动作后端

所有方法仅训练各自前端/readout。作者2k参考曲线保留；正式20k起点，有效batch8动作锚点，每平台3独立训练seed。前端lr1e-5/readout1e-4起点，AdamW、warmup5%、cosine、clip1，充分训练与扩展规则按共同契约。不能让B2只训2k却把N02 20k写成同预算胜出。

### B：同预算共同适配

首版选择**action expert及其输入/输出投影、时间/条件投影一起解冻，语言主干仍冻结**；N02/B2/B3完全相同parameter name白名单。依据实际checkpoint/module树生成并保存白名单和参数数，断言同后端组一致；避免凭“最后几层”这种模糊名词实现。

后端lr1e-5、前端1e-5、readout1e-4起点，50k更新、有效batch64锚点，通过累积/DDP实现，预注册上限100k。显存不足优先减microbatch加累积/activation checkpointing，不单独给N02换更大容量。若该解冻方案因E1/P0资源不可行，在开发阶段提出统一替代配置再冻结，不能测试后单独改N02。

A/B分别报告，不把A的B2和B的N02交叉比较。B1作者ISP保留原2k复现，充分预算系统比较提供相同动作曝光/更新曲线；ISP有原视觉塔，与替换塔的容量差额披露。后端原始权重、原norm和被冻结模型不覆盖，所有新checkpoint独立保存。

## 4. 正式训练器必须实际支持的行为

按effective batch计optimizer更新和曝光，scheduler跟optimizer.step，累积不多走scheduler。AMP溢出步、梯度裁剪、有效监督数量与吞吐记录。DDP按source/anchor可追踪shard，验证epoch采样覆盖；不靠反复同一batch的loss下降交付。

恢复含所有状态：weights/optimizer/scheduler/scaler/RNG/sampler/step/model_version。执行同起点“连续N步”与“N1保存+恢复N2”的下一批身份、下一步loss与参数比较；推断state恢复另测。训练复现允许已记录数值非确定性，不能不核验批次次序。

每5000更新做固定开发动作和机制评测；训练epoch的unique anchors、轨迹、照明曝光和GPU小时单列。行为持续改善可按预注册上限扩展；最终测试不参加选模。每平台至少3正式seed，同seed沿上游续训，不把三个head初始化说成三个独立完整模型。

## 5. 离线验收与P7交接

用真实NPZ批次，验证input→实际flow loss→声明连续分支梯度；图像塔调用0；动作tail mask生效；训练/采样prefix一致；flow10次不重复编码；batch部分reset；同episode跨chunk/不同episode不串；revision后旧缓存不复用；保存恢复。

动作报告分为归一化空间和反归一化实际控制空间：全horizon与执行前缀的每维MAE/RMSE，平移/旋转/关节/夹爪按实际单位分组，夹爪切换准确率/延迟，低光/强光/遮挡组、输出异常、开发闭环和端到端时延。维度混合均值不替代物理可解释指标。

同时重测P1信息保持、P3发现和P4撤回，防动作微调把机制损坏。P6 loss下降只作为P7准入，不写最终任务成功。

交付每平台/方法/轨道3seed模型、白名单、统计/分区/源码/runtime锁、完整曲线、成本、动作与机制原始指标；明确通过的正式checkpoint、P7配置和闭环adapter入口。P7运行前冻结这些版本。

# 给DeepSeek：E1/E2，π0.5早期适配预检

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

## 本阶段执行位置

优先连接 `ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080`，在远端 `/workspace/owac/worktrees/n02-measurement-completion` 开发、测试和运行。先完整阅读[远程开发执行说明](03_REMOTE_DEVELOPMENT.md)，按其步骤准备尚未建立的 worktree、解释器与 P0 资产；随后读取远端 AGENTS.md、README.md、docs/status.md。本文命令、相对路径及 `python` 均指远端 checkout 和对应阶段的独立运行时，GPU 命令显式使用 `"$N02_PY"`。

E1/E2 都在远端用真实 NPZ、既有 π0.5 权重/tokenizer/stats，验证同进程前端梯度、采样和缓存；随后执行每平台不超过 2000 次更新的预检。使用 N02 独立环境与候选服务，直接读取服务器资产。

读[共同契约](00_SHARED_CONTRACT.md)、P1或P2实际报告、[正式系统协议](P6_POLICY_TRAINING.md)。本提示词执行两次：P1后`mode=direct`为E1；P2末`mode=objects`为E2。每次1个工程seed、每平台最多2000次optimizer更新；属于排查接口风险，不作为正式三seed能力证据。

## 1. 文件与固定后端

实现`n02_full/readout.py`、`policy.py`，训练入口增加`--stage preflight --mode direct|objects`，评测`evaluation/n02_full/policy.py`。必须阅读旧`RawVLAFeaturePolicy`的`forward_from_visual_tokens`、`embed_prefix_from_visual_tokens`、`_sample_impl`、`denoise_step`、`prepare_control_targets`和`reset_episodes`。上游[R25](02_REFERENCES.md#r25)的PI0/PI05/ActionHead只用于核对固定版本语义，不替换为最新OpenPI。

Base及其P1检查点保持来源；读出由[R20 Perceiver](02_REFERENCES.md#r20)的cross-attention/query结构移植，先做算子parity。首版512内部宽度、8头、128 latent queries、2个latent self-attention block，再用每相机16×16坐标查询解码为256个2048维token。该宽度/层数为本方案实现选择，不冒称官方预设。

direct：每相机stride8/16测量tokens输入读出；objects：同一个选中世界的实体+该相机多尺度测量tokens输入读出，附source_type/camera/位置编码。保留测量分支帮助弱局部细节，不只用对象池化；额外object-only消融测是否丢失信息。两种读出尽量共用块、输出预算和初始化种子，参数差额单列。

## 2. 实际调用路径

`WorldTokenSource`继承旧TokenSource约定（calls/reset/trainable_records等），持有**一个bundle级状态**。每次接到全部相机frames调用一次`model.step_bundle`；先在完整世界集合选择一个world_id，再生成各相机tokens并按冻结相机顺序拼接。direct路径可以没有世界结构，接口仍相同。旧按相机循环的EncoderReadoutTokenSource继续保留给旧模型。

训练与推理共用 `encode_bundle_to_tokens`。时间window逐时刻构造历史，不把T当batch；最后真实锚点产生对应动作监督。一次flow动作生成期间tokens、MAP world_id及prefix固定；10次denoise不再次编码RAW。冻结后端仅`requires_grad_(False)`，训练计算仍保留对输入tokens的梯度，不能用no_grad包住整后端。

缓存key至少包含episode、anchor、model/readout版本、selected world及revision版本、输入/校准版本。旧bridge每次采样新建prefix KV，不需要虚构跨调用缓存；如新增复用，必须完整版本匹配。动作chunk被同一因果状态的新revision推翻时由adapter清空未执行部分并重新规划；该事件式提前重规划在开发扩展轨道单独报告。正式基准默认固定5/50 cadence，见P6/P7。

reset清空指定batch的模型、token、prefix及action chunk，其他槽不变；随机训练窗口起点reset，连续同episode rollout才保留。语言/本体规范沿用原tokenizer：RoboTwin有状态序列化时使用原入口，不改成自创prompt。

## 3. 两平台验收矩阵

| 平台 | cameras | 环境动作维/内部padding | horizon / replan / denoise |
| --- | --- | --- | --- |
| LIBERO | agentview、wrist，按真实camera_ids | 7 / 32 | 10 / 5 / 10 |
| RoboTwin | head、left、right | 14 / 32 | 50 / 50 / 10 |

RoboTwin真实NPZ当前state读取actions[t]而监督从t+1起；显式MEAN_STD及stats_key；LIBERO quantile/夹爪变换用旧已审计函数。全部action_padding_mask传入bridge，不把无效尾帧重复训练。原loss对32维的约定沿用桥与上游；环境动作误差只统计有效7/14维。

先用真实训练分区1个batch完成forward/backward/save/reload/sample；核查Base、readout、对象路径（E2）有效梯度，冻结后端梯度为零，原图像塔调用为0。记录token RMS/方差、prefix dtype/masks、输出动作分布、实际动作loss。E2的连续求解器若有不可微硬选择，只报告实际可微分支，不能承诺所有模块都有动作梯度。

随后工程seed7101，预检有效batch8个动作锚点、lr前端1e-5/readout1e-4起点，AdamW、100步warmup、2000更新上限；保持RAW/结构loss以P1/P2冻结权重继续约束。E1/E2用同样样本/flow噪声清单、backend初值和readout初始化；E2多出的前端训练量明确披露，额外保留冻结前端仅训读出的对照。

只在P0指定少量开发任务/初态跑连通性闭环，任务/照明/cadence相同，预算执行前冻结，不能使用P7测试身份。在指定远端先验证候选仿真入口和可用资源；具体仿真依赖或资源确实阻塞时列出实际错误及闭环未运行，离线真实数据检查继续完成。

测试：同tokens/noise/time的桥训练结果一致；改变无效相机值不改输出；batch部分reset；恢复后下一个更新；连续两个锚点与重放一致；整世界MAP不跨相机混合；每次预测encoder1次、prefix1次、denoise10次；动作维、夹爪及尾padding。对象化前后比较全horizon和执行前缀误差，分单位与夹爪报告。

## 4. 结论与交接

成功标准是接口/梯度/状态/动作分布合理、真实训练可持续；2000步不改善不能直接判表示无效。明显退化先给定位证据：归一化→mask与位置→token尺度→读出→对象细节。若接口正常，保留负的短训成绩，正式能力仍由P6预算决定。E2若充分表示训练后仍明显损伤局部/动作信息，按P2交接规则暂停P3，不自行换成旧网格模型。

交付两平台实际数据/模型/runtime哈希、更新曲线、梯度分支表、保存恢复、开发闭环结果及E1/E2配对表，费用单列。不得把E1/E2工程seed并入P6正式3seed。

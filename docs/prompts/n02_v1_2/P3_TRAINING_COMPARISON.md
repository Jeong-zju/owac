# DeepSeek N02 v1.2 P3 单种子充分训练与整体比较

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

只执行 **P3**。输入是已交付的P2完整N02、真实动作链路与监控页。本阶段重点是把完整模型认真训练并得到可解释的整体比较，不再扩建一串局部机制。若P2有确定接线错误，先修并定向检查，不重做P0、不恢复旧人工CFA题。

## 执行位置与本轮预算

这是 N02 v1.2 当前阶段的完整执行包。你要直接修改代码并执行本阶段要求，不能只输出设计建议。遵从用户最新指令：单训练种子 **1103**；当前阶段确认评测 **每个实际选中的模型共100次闭环**，不是每任务100次，不是每照明域100次，不是三种子，不是13,250条全集。四臂是归因设计，不是每次修改必须全部训练、全部评测的门槛。未跑项目写未运行，按已完成比较限定结论。

登录命令已更新，旧服务器停用：

```bash
ssh -p 51584 root@93.91.156.102 -L 8080:localhost:8080
```

新机远端8080由平台Jupyter占用，保持它运行。监控应用监听 **127.0.0.1:18080**；用户本地使用下面的附加转发，打开 **http://127.0.0.1:8081/**：

```bash
ssh -p 51584 root@93.91.156.102 -L 8080:localhost:8080 -L 8081:localhost:18080
```

本地提示词/验收镜像为`/home/jeong/zeno/owac/worktrees/n02-measurement-completion`；真实开发和训练使用下表远端路径。已在远端的会话不要再SSH自己。自动化SSH不需要`-L`。启动前读取`/etc/vast-agents-guide.md`、当前checkout的`AGENTS.md`、`README.md`、`docs/status.md`、`docs/plans/N02_full_development_v1.2.md`、`research/experiments/2026-10-15_n02_v1_2_remote_deployment.md`；后者区分实际部署结果与待做研究。旧v1.1提示词仅查源码来源，不继承旧阶段门槛。

| 用途 | 远端绝对路径 |
| --- | --- |
| 活跃Git worktree | `/workspace/owac/worktrees/n02-measurement-completion` |
| 分支 | `lzh/n02-measurement-completion` |
| 开发检查 | checkout内`.venv/bin/python`，`uv run --locked`，`make check` |
| N02 GPU解释器 | `outputs/n02-full/runtime/py310-cu128/bin/python`，相对上述worktree |
| 作者π0.5环境 | `/workspace/owac/rawvla/envs/starvla/bin/python` |
| RoboTwin仿真环境 | `/workspace/owac/rawvla/envs/robotwin/bin/python` |
| RawVLA源码 | `/workspace/owac/rawvla/source`，StarVLA子目录`starVLA` |
| 官方数据根 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench` |
| LIBERO转换后权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO统计 | `/workspace/owac/rawvla/checkpoints/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json` |
| RoboTwin权重与统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`、同目录`policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 本阶段产物 | `outputs/n02-v12/<stage>/<run_id>/`，权重/视频/原始日志留在远端 |
| 阶段报告 | `research/experiments/n02_v1_2/`，仅源码、配置、小型结果供本地验收 |

Codex负责本次新机初始部署；先看部署报告和实际路径，不重新下载已有714GB数据，不复制虚拟环境，不重新做P0完整parity。旧`p0-remote-*.json`只借数据路径/统计，不复制其中三种子、worlds8、旧scales或时延门槛到v1.2配置。远端Git带本地未提交覆盖，`git status`有修改是正常状态。保留它们，不执行reset/clean、不擅自提交推送。HF认证已在远端私有位置配置，需要下载时读取既有认证；不把token写进配置、报告、网页或命令日志。

服务器有两张约96GiB的RTX PRO 6000 Blackwell。先看`nvidia-smi`和现有supervisor任务。默认**物理GPU0跑RAW、GPU1跑RGB**，每个作业用`CUDA_VISIBLE_DEVICES`只暴露一张卡，进程内设备是`cuda:0`；不同run目录、日志、checkpoint、策略端口（如RAW18100/RGB18101）、仿真输出和临时文件。首版不用DDP把两卡绑在同一训练上。验证也可RAW/RGB各一张卡并行，训练和验证按实际显存排队，不能不加区分地同时挤在一张卡。两平台后端分别适配，单种子不等于两个平台共享同一动作头。

长期训练和监控用本机supervisor管理，分别命名`owac-n02-v12-*`，保存完整启动命令与恢复命令。训练作业`autorestart=false`，出错保存状态并显示原因；监控可自动重启。只管理自己创建的进程，不停止Jupyter/Caddy或其他项目任务。

## 研究与实现边界

目标是把RAW相对固定ISP RGB保留的动作相关线索转化为VLA能力。主链：**RAW3→ConvNeXt V2-Base多尺度空间特征→多候选潜在解释→RAW测量预测及可靠性比较→循环修正/状态重建→直接空间特征与潜在状态联合读出→π0.5**。RAW主链不生成RGB/sRGB/XYZ供策略输入，不用RGB教师。RGB只用于独立对照及可视化。

LIBERO是RGB8逆处理的三通道pseudo-RAW；RoboTwin是HDR-derived三通道float32测量；都不是原生Bayer。LIBERO不能证明恢复源RGB8已丢失的信息。RoboTwin是本轮RAW信息研究主平台，LIBERO用于兼容和补充。不要把正常光配对RGB当作退化RAW的同源RGB：受控RGB必须由该次**同一RAW经固定ISP**得到。

复用`src/owac/raw/rawvla_sequence.py`、`src/owac/frameworks/n02_full/data.py`与P0分区。LIBERO两相机，RAW uint8转FP32/255，8维本体，7维动作，offset0/horizon10/每5步重规划；禁止92维仿真states进入策略。RoboTwin三相机，float32原尺度，不除255、不逐帧max；原NPZ16维robot_states不是在线14维关节状态；沿用`actions[t]`为当前状态、offset1/horizon50，在线真实joint_action/vector，左6关节/左夹爪/右6关节/右夹爪，checkpoint明确**MEAN_STD**，不做默认min/max或重排。逐动作mask沿用loader。

注意现有特征桥的单位边界：`RawVLAWindowDataset`使用正式统计后，sample的state/action已经归一化；`prepare_control_targets()`只补齐32维，`control_actions()`只切回7/14维，**都不会替你做归一化或逆归一化**。训练不重复归一化；在线采样输出在送`take_action`前按各平台原统计逆归一化一次，并沿作者夹爪规则执行。不能把名为control_actions的归一化切片当物理控制量。

用P0 manifest保持源轨迹/相邻帧/照明变体隔离；8帧历史在原分区身份上重新生成窗口，前期不足用mask，不读取未来。测量ID包含episode/frame/camera；同一帧内部4轮、前缀重放、多尺度特征不变成多份独立证据。缺少时间戳时沿用frame索引，不伪造秒；未知成像参数保留未知标记，文件名/光照域标签/候选编号不能变成答案线索。模拟器隐藏状态和未来专家动作只能做标签或报告。唯一测量计数用简单集合，不建立大型哈希账本。

只保留会影响正确性的检查：真实输入/动作单位和shape、finite、因果性、分区隔离、先验/校准/梯度接线、episode reset、保存恢复。禁止新增90%纠错率、任意自定义综合分数、逐文件全量SHA门槛、全套算法parity重跑、必须连续N次loss下降或必须达到某个平台期。代码版本、运行配置、checkpoint路径、实际更新数和小型样本身份足以追踪本轮；下载器原生完整性检查保留，不循环扫描714GB算hash。NaN或接线错误要修，正常失败或没有性能提升要如实报告，不能回退随机模型、伪造数据或跳过失败。

理论只承诺适用条件内性质：固定有限候选与测量时，`q=softmax(log(p0)-E/tau)`解`E_q[E]+tau*KL(q||p0)`；学习能量的q称支持权重，不冒充校准后验。连续更新在同一完整目标上接受非增步骤，失败可保留旧状态并记录拒绝；不要求训练loss单调，不声称全局收敛。允许全局重建真实神经状态与缓存；首版不强制关系图、RJMCMC、局部传播或额外Julia/Theseus运行时。

## 起步先确认真实交付

读`research/experiments/n02_v1_2/P2_report.md`、实际checkpoint/config、训练源与forward/sample主循环。核实已有实际optimizer更新数、恢复路径、train/dev/calibration分区与8帧窗口；P2短训从同一run续训，不重复把1000步再跑一遍。缺少关键文件先沿P2补完整可运行版本，不写“应已通过”跳过去。读取监控页的真实activity并更新为P3。

核心网络保持ConvNeXt V2-Base、多尺度直接支路、K4/slots16×512、内循环4、因果8帧、Perceiver到每相机256×2048的起点。只因明确资源画像或真实失败改配置并说明，不随意缩成tiny，不把大量局部代理头做成主线。

## 训练安排和双卡使用

所有训练seed=1103。不启动2207/3301，不为统计好看训练三份。首轮主平台RoboTwin，因为其HDR-derived RAW更适合研究显示处理前信息；LIBERO依原分区与动作协议做补充，预算按实际排期，不强制两平台所有方法一起跑完才能报告。

1. RAW预训练预算起点100k optimizer updates/effective batch256帧；RGB对照用同源固定ISP输入、同骨干/训练池/曝光量/初始化规则。GPU0 RAW、GPU1 RGB，可并行。已有兼容RAW3预训练可续训并扣累计更新；旧CFA4测量权重不得无说明混入。
2. 完整N02—π0.5适配预算起点50k optimizer updates/effective batch32决策锚点、每锚点8帧。microbatch以实际显存决定，用累积达到目标；报告每更新实际样本数，不把microsteps算optimizer updates。预训练与适配计数分开。
3. 先建立`raw_n02`与`rgb_n02`成对主模型；随后从同模态预训练起点运行`raw_base`/`rgb_base`普通融合，优先回答机制收益。普通融合也看相同8帧，用成熟SAVi来源predict/correct或普通Transformer时序融合，不能让baseline只看单帧。复用预训练要披露，不把共享训练算成四份独立复现。
4. 适配先训练新前端与读出，默认前2k适配更新冻结后端，随后使用同一后端解冻规则完成剩余预算；若P2实际起点不同，以配对共同规则记录。encoder/readout初始LR可用1e-4、动作后端1e-5、AdamW/warmup/cosine、BF16主计算+FP32测量作为工程起点，dev选少量必要调整；不是理论最优值，不做大规模网格搜索。参考官方训练配置，报告实际参数组和可训练数量。
5. 每1000更新保存可恢复checkpoint，每1000更新在固定dev样本计算原生动作loss与位置/旋转/夹爪误差，噪声/flow时间固定以便纵向比较；间隔可因耗时调整。不要每个保存点都启动100次闭环。保留last和dev选中best，optimizer/scheduler/RNG/sampler进度可恢复，定向试一次恢复即可。

100k/50k是明确资源计划，完成后报告计划训练结束及实际结果，不要求证明数学收敛或等到loss不再变化。正常loss波动、dev平台期或没有优于基线都不是无限续训理由；只有NaN、无梯度、错单位、数据错位等实现问题才停下修。预算内继续有收益可记为后续建议，不擅自翻倍/三种子；硬件中断保存并恢复，未跑完写累计数量与当前运行状态。

## 受控比较要回答的问题

四臂定义必须沿同一原测量与固定ISP：`rgb_base`、`raw_base`、`rgb_n02`、`raw_n02`。RGB N02使用实际RGB域观测目标/残差条件，不能套RAW Poisson-Gaussian方程。Base臂可比骨干、历史和后端适配，保留官方RGBπ0.5作独立强参照。RawVLA学习ISP是强近邻；如果当前尚未重训，不将论文数字当成本机baseline，也不因它尚未完成阻塞主对照交付。

本阶段先交付已完成的主模型与配对比较；四臂/学习ISP/普通循环强对照的缺项列待做并限制主张。不能为了“全部矩阵合格”反复重训，也不能在只有RAW/RGB N02两臂时声称独有机制优越。必要消融按具体问题选1—2项（如关闭测量反馈或单轮），先做固定dev诊断；不把所有消融都变成50k+100次的默认任务。

主读数使用现成任务成功/失败、原生flow-matching损失和有单位的动作误差。不要新造综合分数、用能量下降率选最强模型、规定成功率必须超过某线才能完成阶段。训练/开发误差与闭环不同，固定dev改善并不保证闭环成功。

## 必须用真实轨迹解释行为

固定来自dev的几条官方episode，同步RAW、同源RGB、attention叠加、局部RAW预测/残差、支持权重、修改前后状态和动作。对同一checkpoint做高attention区域遮挡与等面积随机区域遮挡，保持同一噪声/历史/动作采样条件；局部遮挡标为人工诊断，不能伪装成自然测试集。再做一次关闭测量反馈/置换过去历史观察完整动作输出；不注入隐藏真值正确候选。

给出“哪些线索RGB被削弱、RAW是否还保留、编码器是否真使用、动作是否改变”的具体样例和失败样例。只画好看热力图不足以解释利用；但也不为可视化发明复杂硬性打分。校准结果未成熟就保留未校准标记、看真实残差/候选差异，不让统计术语挡住动作训练。

开发sim只用于排错与选模，使用与正式100条不相同的初态。不要把P4测试失败反复拿来训练和挑模型。本阶段不要自动开始13,250全集或632附加核查。

## CLI与作业交接

优先延用P2已经实现的CLI；若需要补命令，只加pretrain/fit/evaluate/visualize中的必要选项。以下为接口意图，核对实际`--help`后填完整命令，不把占位命令标成已运行：

```bash
cd /workspace/owac/worktrees/n02-measurement-completion
export N02_PY="$PWD/outputs/n02-full/runtime/py310-cu128/bin/python"
export PYTHONPATH="$PWD/src:/workspace/owac/rawvla/source/starVLA"
CUDA_VISIBLE_DEVICES=0 "$N02_PY" -m owac.training.n02_v12 --config configs/encoders/n02_v12/raw_n02_robotwin.json --seed 1103 --resume outputs/n02-v12/P2/raw-n02-s1103/last.pt --max-updates 50000 --output outputs/n02-v12/P3/raw-n02-s1103
```

用supervisor分别管理RAW/RGB，不能在对话里输出“正在训练”后丢失真实进度。监控更新phase/target真实值，记录下载/加载/dev/保存等长停顿的activity。结束时若训练仍在跑，应明确这是进行中交接而非P3完成，并给PID/日志/恢复；真正报告训练完成必须有实际checkpoint和更新计数。

## 立即调用前端专家的明确任务

在开始长训练前，调用前端专家子任务并把以下任务原文交给它。你负责训练代码、状态输出与集成；专家只负责监控后端/页面。双方不独占整个仓库，不回退对方修改。若当前DeepSeek客户端没有子agent接口，由你按同一任务完成，不能因无法委派停止。

> 请为N02 v1.2实现一个简单、可实时查看的中文进度页。你拥有`src/owac/runtime/n02_v12_monitor.py`与`src/owac/runtime/n02_v12_monitor.html`；主agent拥有模型/训练/评测，先约定下面JSON字段。已有部署临时页在18080时，在同一个监控入口上升级，保留其下载/部署状态，不重复绑定端口。只用Python标准库ThreadingHTTPServer+单HTML/CSS/原生JS与本地SVG/canvas折线，不用React、Node构建、数据库、账号系统、WebSocket、外部CDN、复杂路由或控制作业按钮。
>
> 页面从上到下只有：①当前阶段、正在做的事、最后心跳、运行/完成/异常文字；②RAW与RGB两列卡片，显示GPU、真实optimizer step/目标、已用时间、最近step速度估计的ETA（暂无数据则留空）、checkpoint、当前loss与固定dev动作误差；③共享横轴两条训练/开发曲线，训练loss和dev误差单位不同分图，不画假平滑；④固定官方轨迹的RAW显示、同源RGB、attention叠加与残差预览，可选相机/帧/内循环，不存在的图写“尚未生成”；⑤闭环已完成n/100、成功s、失败f、基础设施错误e、当前成功率s/n，以及最近30行日志。未运行写未运行，分母不能把待跑项算进去。
>
> 数据接口`GET /api/status`与`GET /api/history?run_id=...`；只读配置好的run目录，禁止任意路径浏览。训练端每10 optimizer updates或5秒更新run下`status.json`，长步骤由轻量心跳每5秒更新时间；每次dev/保存/异常立即更新。写临时文件后原子rename。JSON约定：`run_id,stage,arm,platform,seed,state,activity,gpu,step,target_steps,updated_at,started_at,step_seconds,loss,dev_action_error,checkpoint,eval{planned,completed,success,task_failure,infrastructure_error},preview_files,log_tail`，不适用项null；history.jsonl追加`step,wall_time,train_loss,dev_action_error`。代码开发阶段也更新activity，不能训练开始前一片空白。可用现有bootstrap `/workspace/owac/deployment/downloads.json`显示部署状态。
>
> 浏览器每3秒轮询。超过30秒无心跳显示“数据未更新，请查看日志”，不把它自动判断训练失败；网络异常保留最后有效快照并显示更新时间。显存/利用率每10秒采样一次`nvidia-smi`，CPU读取即可。页面手机可读、无横向溢出、没有十几张仪表盘。localhost18080绑定，supervisor常驻，用户通过SSH的本地8081访问。凭据/完整环境变量不得入网页。
>
> 验证页面能显示一个明确标“界面测试”的样例，之后切换真实作业；检查真实step变化、训练错误显示、SSH退出后服务仍在、两GPU卡片独立、手机宽度和一张真实attention图可打开。样例不能混入训练结果。给主agent返回变更文件、启动命令、实际URL及截图位置。不要修改模型、损失、成功判据或训练预算。

你要把专家交付接入真实作业。网页是本阶段交付的一部分；无需追求设计奖，不允许花大量时间做权限、告警、调度等附加系统。DeepSeek长对话结束前确认网页仍显示当前真实状态，不能只交一个离线HTML截图。

## 直接参考论文与实现

先读当前实现再读对应原模块。已有P0移植和许可证保留，直接复用；需要适配时解释差异，不另写无来源的小网络，也不因上游主分支更新重写已验收组件。

- ConvNeXt V2：<https://arxiv.org/abs/2301.00808>；官方Base骨干、GRN与FCMAE：<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/convnextv2.py>、<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py>、<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/fcmae.py>；训练循环<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/engine_pretrain.py>。读取Base depths/dims、GRN、mask与loss，不用RGB权重做隐形教师。
- SAVi / SAVi++：<https://arxiv.org/abs/2111.12594>、<https://arxiv.org/abs/2206.07764>；官方predict/correct与slot注意力：<https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/video.py>、<https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/attention.py>；decoder <https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/decoders.py>。复用P0 PyTorch移植，不为了运行SAVi改成JAX训练全系统；对象槽不保证自然获得语义分割。
- Perceiver IO：<https://arxiv.org/abs/2107.14795>；官方attention/latent/decoder <https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/perceiver/perceiver.py>，位置编码同目录`position_encoding.py`；本仓`src/owac/backends/rawvla_features.py`是已存在策略桥；`src/owac/frameworks/n02/readout.py`是旧token readout，并非Perceiver。请在`n02_full/readout_v12.py`中移植官方Perceiver核心注意力/latent/query读出，默认512宽、128 latents、每相机256查询，再投影2048维；只对新增移植做一次必要数值/梯度对照，保留许可证，不重做P0全套。
- RawVLA论文、benchmark和官方固定实现：<https://arxiv.org/html/2609.37530v1>、<https://github.com/ShuhongLL/RawVLA/tree/ba9c56b04da2c4931b70d8844710f5928d43775b>、<https://huggingface.co/datasets/ToferFish/RawVLA-Bench/tree/8586c594fff9ce38580e29745f074e82e96e74f7>。主源码`experiments/simulation/{libero,robotwin}`，同源ISP查该提交实际实现。
- StarVLA NPZ loader和π0.5：<https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py>、<https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py>、<https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py>。关注prefix、flow-matching、padding、动作和normalizer，不经过PIL把RAW量化成RGB。
- 官方OpenPI固定PyTorch：<https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py>；π0.5论文<https://arxiv.org/abs/2504.16054>；RoboTwin原权重<https://huggingface.co/SidneyXie/pi05_robotwin/tree/e49e2ab6c11f07511573b67261bd129e88d0a416>。
- RAW噪声/处理：Unprocessing <https://arxiv.org/abs/1811.11127>；RAW-Adapter <https://arxiv.org/abs/2408.14802>；Dr.RAW <https://papers.neurips.cc/paper_files/paper/2025/hash/20eadb6338a53114a6371f102c5ded95-Abstract-Conference.html>。它们界定已有工作，不证明当前模块有RAW能力；Dr.RAW当前来源不能冒称可直接执行的完整代码。
- 多假设强近邻SWB <https://proceedings.mlr.press/v139/singh21a.html>；广义更新理论<https://arxiv.org/abs/1306.6430>；校准条件<https://arxiv.org/abs/2202.13415>；attention解释边界<https://aclanthology.org/N19-1357/>。
- 可选连续优化参考Theseus：<https://github.com/facebookresearch/theseus/blob/c8583de41824613fb135ee9bad0e930ded6be404/examples/simple_example.py>；状态重执行Gen <https://www.gen.dev/docs/stable/>。只参考必要机制，不把安装两套求解框架当P2/P3/P4前置。
- 本地组件来源：`src/owac/frameworks/n02_full/components/README.md`、`configs/encoders/n02_full/third_party.lock.json`。更多固定文件：`docs/prompts/n02_v1_1/02_REFERENCES.md`；只读源码导航，不执行该历史包的阶段要求。

## P3交付与停止

更新`research/experiments/n02_v1_2/P3_report.md`、`docs/status.md`和小型`runs.json`：每个run输入/骨干/历史/后端参数量、实际数据曝光与optimizer步数、seed、起点和best/last路径、时间显存、固定dev曲线、可视化、已有比较与失败。结论直接说当前模型训练到什么程度、RAW/RGB动作差异是什么、哪些机制证据尚缺。`make check`及只覆盖新增修改的必要检查即可，不要求重跑所有P0上游。提供给P4的固定待评模型列表和推荐100条清单生成规则，停止在P3，不自动换测试集调参、不开始P4、不提交推送。

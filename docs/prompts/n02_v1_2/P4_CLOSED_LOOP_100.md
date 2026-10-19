# DeepSeek N02 v1.2 P4 独立重载与100次闭环确认

只执行 **P4**。本阶段拿已训练并选定的模型，以独立进程重载，运行真实闭环并报告成功率。主要工作是把完整系统真实跑100次，不是再做长训练、全量benchmark或复杂验收框架。

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

## 选定模型与100次的明确含义

读`research/experiments/n02_v1_2/{P2_report,P3_report}.md`，列出本次实际待评checkpoint/config。默认优先`raw_n02`与`rgb_n02`，每个模型**总100条**同一任务/初态/光照清单，GPU0 RAW、GPU1 RGB并行；其余Base/学习ISP只有已训练且本次确有比较必要才纳入，不能等待四臂全部就绪才跑。四臂都评时是4×100而不是每任务/条件100；未选模型不自动训练。

当前默认主平台为RoboTwin，100条来自作者原3250条主manifest，按task与照明域分层、选取前先固定seed1103，尽量均衡任务和五域，样本身份无重复。选择只看清单字段，不看已有成败，保留原动作/相机/噪声/成功规则。LIBERO在本阶段作为补充不自动再加100；若P3已明确选择双平台方案，则**总100按LIBERO50+RoboTwin50**，两个平台各报s/50，不混成具有统一意义的跨平台成功率，也不再额外各跑100。两种排期只能择一，开跑前在run配置写清。没有明确P3安排时使用RoboTwin100的默认值。

生成`configs/evaluation/n02_v12/closed_loop_100.json`，保存父manifest来源、固定模型清单、每条task/init-or-seed/lighting/噪声随机种子与实际相机协议。正常光参考和退化条件均来自已定义协议，不临时挑简单任务。不必给每个文件生成hash；记录源版本、路径和case_id即可。这个100条子集结果不是官方全量分数，不与旧58.20%/15.63%直接做同分母显著性比较。

## 正式开跑前只做必要的独立确认

在新进程从实际checkpoint+config建立模型，读一个真实RAW3样本，走加载→编码→修正→桥→采样；检查参数确实恢复、动作单位/shape/finite正确。部署已修复特征桥single-batch CLI的CPU语言token与冻结后端输入不可求导问题，沿用修复版本；测试用随机特征只算桥接检查，不算模型能力。已有加载器的必要键/shape检查保留，不用人为全树SHA扫描。缺少有效模型参数时直接修复或报错，不能strict=False悄悄用随机参数跑100条。

确认以下四件事即可，不扩成一批新局部任务：①reset不跨episode带状态；②同一帧内循环没有累计成4次新证据；③三态status真实可读，未校准不冒称保证；④实际attention/残差来自选定checkpoint。保存恢复用一段真实后缀定向确认，正常数值容差即可。关键错误修复后重新标明模型版本，不能混合修复前后同一100条成绩。

调用作者现成仿真入口和既有`owac.runtime.rawvla_robotwin`/`rawvla_full_eval`中的协议实现，新增一个小的100条调度入口即可；**禁止直接默认调用旧full-eval的3250/10000硬编码循环**。需要接N02 RAW特征时让sim发送实际当帧RAW3、本体、语言与已执行动作，绕开RGB图像塔；RGB对照同一RAW经固定ISP再送RGB臂。不能沿用旧Default ISP服务但将结果改名为raw_n02。

RoboTwin保持checkpoint的MEAN_STD和14维顺序、50步action chunk、场景/光照/RAW噪声及原成功判据。Blackwell仿真按已验证部署使用OptiX降噪，报告该后端。不能用正常RGB观察替换弱光RAW，也不能注入专家规划动作“修复”策略。

若用LIBERO，保持作者初态/两相机/7维动作/10步块5步重规划及原成功判据；隐藏92维sim states不能进策略。平台已知专家检查/随机状态问题沿已部署修复，实际执行的eval seed固定并保存；不能专家检查失败后悄悄跳到更容易的seed。

## 并行运行与真实统计

两卡各一个模型及其sim/策略进程，共享只读模型资产；策略端口RAW18100/RGB18101，不共用动作缓存/临时NPZ/渲染目录。每条完成追加`episodes.jsonl`：case_id、模型、平台、task、init/seed、lighting、success布尔、终止原因、控制步数、耗时、视频路径；异常尝试另存attempts。resume按case_id跳过已完成项，不按成功跳过失败。

成功率=`成功s / 正常完成闭环n`。正式报告完成到100/100时给`s/100`和百分比；可以附标准Wilson95%区间，说明单训练种子的episode区间不代表跨训练稳定性，同一初态多照明有相关性。无需自己定义综合性能分、收敛阈值或强制置信度线。域/任务样本很小时只给计数，不渲染成精确统计结论。

超时/掉落/抓取失败属于有效任务失败，计入100。加载/通信/渲染崩溃属于基础设施错误，保留原attempt并对同一身份修复重试，不从正常分母删除任务失败、不换身份凑数；显示错误数量和原因。若暂时无法完成100，明确显示s/n、剩余数量与阻塞项，不能宣称100完成。失败多仍继续到100，不因失败率高提前停或挑新checkpoint；实现异常则先保存、修复并判断受影响结果需否重跑。

监控页每episode立即更新n/100与s/n，同时保留RAW/RGB各自GPU/耗时/当前task及最新视频。验证期间不进行影响选模的训练。SSH断开后supervisor继续；不要启动新定时自动化或无限重试循环。

## 让用户看懂成功和失败

从预先固定的case序列抽取若干视频，覆盖成功与失败；原帧上叠加真实attention，旁边同步RAW固定显示、同源RGB、候选残差、状态修订与实际动作。成功/失败画面同格式，注明相机/帧/模型。没有状态修订则显示无修订，不能画手工“纠错箭头”冒充模型行为。

少量机制干预在固定dev轨迹离线做，复用P3已经有的结果：关反馈、遮高支持区与同面积随机区、置换因果历史，观察动作是否改变。这些不默认各加100次闭环。报告区分输入收益、机制增量和当前缺少的四臂证据：只有两臂结果就只支持那组比较；RAW并未优于RGB或反馈无效都如实写。

成本测量取实际完整决策路径，包括读取传输、候选/内循环、重建、读出与π0.5，报告p50/p95与显存；不要只计GPU前向冒充端到端。视频导出开销可单列，不为赶时延换更短历史而不标记。

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

## P4交付与停止

在`research/experiments/n02_v1_2/P4_report.md`首段给每个模型实际`s/100`，或如实`s/n，未完成`；同时给标准分平台/照明域计数、异常/重试数、实际配置/模型路径、逐条小型ledger、视频/网页入口、完整时延显存、主张与证据边界。更新`docs/status.md`与研究卡，不把部署成功、绿测或loss下降当能力通过。只回传源码/配置/小型结果，RAW/大权重/视频留远端。`make check`和受影响的必要定向检查后停止，不追加三种子、13,250全量、632附加、长训、提交或推送。

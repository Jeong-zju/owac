# DeepSeek N02 v1.2 P2 完整编码器与真实VLA链路

> 2026-10-16 最新授权：Codex 已接管并启动 r5 P2 RAW/RGB 训练。下文“仅画像、用户自行启动”是上一轮边界，不再适用于本轮；不要再按本文旧命令启动重复作业。见[当前运行](../../../research/experiments/n02_v1_2/P2_training_live_20261016.md)。

> 2026-10-16 最新执行限制：用户要求先修代码、从头完成修补版 P2 画像，**训练由用户自行开启**。本轮只允许有界 `--profile-only` 前后向，优化更新 0；本文后续的训练预算是历史阶段要求，不授权代理启动。当前版本 `n02-v12-p2-causal-r4`，见[修补与画像报告](../../../research/experiments/n02_v1_2/P2_fresh_profile_20261016.md)及[用户启动入口](../../../research/experiments/n02_v1_2/P2_r4_user_training.md)。

只执行 **P2**。P0基础已通过；P1独立验收未通过，缺陷在本阶段修复。你要一次建立完整N02，接真实RawVLA-Bench、真实π0.5、实际训练和同步可视化；不要再按P2一个小组件/P3一个小组件的旧路线执行。

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

## 先理解代码和已证明的错误

阅读`research/experiments/2026-10-15_n02_p1_independent_acceptance.md`，再追踪以下文件实际调用，列出一张简短“复用/改造/停用”表即开始开发，不花数天搭审计框架：

- `src/owac/frameworks/n02_full/{model,observation,sensor,factors,checks,state,data,config}.py`。
- `src/owac/frameworks/n02_full/components/{convnextv2,fcmae,savi}.py`、旧`src/owac/frameworks/n02/readout.py`（后者不是Perceiver，正式读出按官方源码移植）。
- `src/owac/training/n02_full.py`，特别`_model_step`、真实RAW source、save/resume；`src/owac/evaluation/n02_full/p1.py`。
- `src/owac/backends/rawvla_features.py`的`RawVLAFeaturePolicy`、`EncoderReadoutTokenSource`、forward/sample/reset及动作协议。

本次部署还修复`rawvla_features.single_batch`的GPU设备与视觉输入求导接线；该CLI仍是随机特征的后端探针，不能替代下面真实RAW完整训练。P1现有5k权重来自人工CFA4；RAW3预训练曾中断。它们不是完整N02初始能力，也不能把5k历史步数算成本轮训练。需实修：

1. `task.inference.declared_prior`没有传入实际更新。新的先验必须改变`log_p0`并参与`q`，用同一真实样本的中性/偏置先验观察输出；不要只保存字段。
2. `locked_checks`存的是无verdict的MeasurementScore；评测读`.verdict`而CheckVerdict实际是`.status`。统一一个真实返回类型，校准器真实调用；缺失检查不得计为0误报/0检出。
3. 旧训练和评测synthetic同seed同partition，且“正确”候选nuisance=0、其他=.2*(i+1)，泄露答案。主实验切官方数据分区和学习提议，不能给正确候选特权。
4. cue swap只重排索引，没有交换测量；64×64 CFA stride8查询漏采信号。新干预在实际RAW像素/历史上发生，并用图展示位置；不继续围绕这个人工题刷门槛。
5. 曝光/噪声/shift组只是标签，不能写成真实鲁棒性实验。改变实际测量才算干预；没做不写通过。

## 一次实现的完整网络

首版配置`rawvla_rgb3`，256×256仅作为起点；骨干**ConvNeXt V2-Base** depths[3,3,27,3]/dims[128,256,512,1024]，stride4/8/16直接空间支路保留；候选K=4、16个slots、slot宽512、每决策8帧因果历史、内循环4轮。它们是初始参数，可因实测显存/速度明确调整；不能悄悄换成tiny CNN并叫Base。当前读出每相机256×2048 visual tokens，原始空间特征与修正状态均可进入Perceiver，不先合成RGB。

建议在现有目录按职责扩展`encoder_v12.py`、`inference_v12.py`、`decoder_v12.py`（名称可为少量合理等价文件）；`model.py`保留一条可追踪主入口；不要把新完整网络继续塞进5千行旧训练CLI。新训练入口建议`src/owac/training/n02_v12.py`，评测`src/owac/evaluation/n02_full/v12.py`；复用公共loader和桥。旧合成器留历史命令，禁止默认切回synthetic。完整主循环应能逐行读懂：

```text
读取当前与因果历史、唯一测量ID和已有状态
→ Base多尺度特征，缓存真实空间坐标/有效性
→ SAVi来源的predictor/corrector提出K个联合多相机潜在解释
→ camera/time条件RAW decoder预测同一测量空间
→ 同一测量集合的噪声/有效性代价，q=softmax(log_p0-E/tau)
→ 学习校正器提出状态更新，比较同一完整目标后接受或保留
→ 处理替代解释并重建受影响真实状态/attention缓存
→ 直接空间特征+K状态+支持权重进入Perceiver读出
→ 一次特征桥生成π0.5动作块，执行后更新历史
```

K是联合解释，不能每相机各挑MAP再拼出互相冲突的“世界”。初始提议从观测特征/可学习latent/因果前态产生，不从隐藏真值生成正确候选列表。RAW decoder优先复用SAVi空间broadcast decoder的已移植模块并按相机条件改输出通道；这只是测量预测，不声称完整3D物理渲染。披露新增层的来源/初始化，允许必要线性投影和输出层，不允许无来源小网络替换成熟骨干。

RAW目标FP32，若已知噪声参数采用信号相关方差及归一化项，学习方差不能无限变大逃避残差；未知噪声不伪造真值。对饱和使用实际裁剪事件的censored目标或显式不评分并报告可用覆盖，不把饱和值当精确强度。坏点/缺失mask独立。暗区低信噪不是一律屏蔽暗区，保留空间一致的弱线索。在线当前帧拟合残差与看到当前帧之前锁定的causal预测分开。

独立calibration轨迹产生兼容/测量不足/模型不足的诊断阈值；没有满足校准条件就标“未校准”，保留能量与差异图继续动作训练，不能因统计诊断未定卡住主模型。不得对反复拟合当前帧后的残差宣称独立检验误报保证。保留三态真实接线和明确的诊断解释；不设三态数量或校准误差必须过某线。

状态包含slots、候选支持、因果历史、缓存和随机状态；reset清除episode间内容。撤回/替换一个观测后，首版从该episode有效历史全局重算即可；同时更新真实张量和读出缓存，不只改provenance字符串。保存恢复同一后缀的动作应在同精度正常数值容差内一致，不要求跨GPU逐bit相等。

动作链复用既有`RawVLAFeaturePolicy`，不能在loss前detach新编码器；在固定官方样本实际反向后查看骨干/校正器/读出各自梯度与更新。硬选择不可微路径如实记录，不以一个投影有梯度冒充整个搜索可训练。只做轻量必要检查，不为每个函数写mock测试。

## 训练、页面与本阶段实跑

先交付完整可运行网络，再在真实训练池做一段实际动作训练，默认首个保存点1000 optimizer updates，seed1103；这是可视化检查点，不是能力合格线或“充分训练”。P3从它恢复，累计预算扣掉已做更新，避免重复训练。先用约200更新画像确定microbatch与累积，也计入实际更新记录；不要把画像变成新的独立开发阶段。首版可冻结后端先练编码器/读出，下一阶段按受控规则适配后端。

训练目标用π0.5原flow-matching动作损失，加RAW masked/causal辅助项。FCMAE目标像素不能经未mask当前帧特征泄漏；该预训练分支与未遮挡动作分支分开，noise监督仅在真实可得时启用。先用现有mask-aware移植，不为节省几个GB临时替换成错误零填充语义。

在`configs/encoders/n02_v12/`提供`raw_n02_robotwin.json`、`raw_n02_libero.json`及同源`rgb_n02`配置，训练参数写实际值。主平台RoboTwin先做完整真实链路，LIBERO做一次真实forward/backward和采样确认协议；不要求两个平台在P2各长训一次。若GPU1空闲，可并行跑同预算RGB N02对照，没准备好先完成RAW，不能等待全四臂才交付。

新CLI建议下面形式，**这是你要实现的接口，不是声称仓库已有命令**。实现后用`--help`核对，再写入实际报告：

```bash
cd /workspace/owac/worktrees/n02-measurement-completion
export PYTHONPATH="$PWD/src:/workspace/owac/rawvla/source/starVLA"
export N02_PY="$PWD/outputs/n02-full/runtime/py310-cu128/bin/python"
CUDA_VISIBLE_DEVICES=0 "$N02_PY" -m owac.training.n02_v12 --config configs/encoders/n02_v12/raw_n02_robotwin.json --seed 1103 --max-updates 1000 --output outputs/n02-v12/P2/raw-n02-s1103
```

按下节调用前端专家并接真实日志。用固定dev官方轨迹生成：RAW固定显示映射/同源RGB/局部放大、attention叠加原图、候选RAW预测与噪声尺度残差、每轮支持变化、实际动作曲线。图必须标episode/camera/frame/iteration/model step；attention的真实层与归一化说明一句即可。没有attention就先把实际slot/Perceiver权重取出，不能画人造圆斑冒充。固定几条成功/失败都保留，不只挑漂亮样本。

允许少量真实sim开发rollout确认模型能控制和reset；P2不要求100次正式评测，100次在P4对已选模型执行一次。开发失败正常保留，不因为失败自动改成功判据。P2交付训练/评测命令、真实checkpoint、可视化、简洁页面、必要修复测试；停止在P2，交给用户/独立验收继续P3。

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

## P2交付格式

更新`research/experiments/n02_v1_2/P2_report.md`和`docs/status.md`：用人能理解的语言说明“图里哪条RAW线索→模型如何改状态→动作有什么变化”，实际没有变化就直说。给出改动文件、旧P1六类错误如何修、实际运行命令/解释器/数据/模型路径、实际更新数与时间、网页URL与恢复命令、未解决问题。附小型配置与结果JSON，不要求成百上千行hash表。报告明确区分完整代码、已训练程度和实际能力；负结果同样可以完成工程阶段。最后运行`make check`与受影响的少量真实数值/数据/桥接测试，保留失败原因；不启动P3、不提交推送。

# DeepSeek N04 v1.1 S4 百次闭环评估与阶段结论

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

本次目标是把当前可用的 N04 和必要对照整理成可看懂、可继续研究的阶段结果：每个待评模型 100 次闭环、真实成功率、代表性成败与清楚的剩余问题。本阶段不要求官方 13,250 条、所有方法／消融或三训练种子，也不以这些未跑项目阻止交付。

## 先执行这些约定

你负责实际编程、远端训练和本阶段结果交接。不要只交计划，也不要把工作改写为大量预检查、验收框架或 mock 演示。用户当前要求优先于旧文档：**每个平台／方法只训练 1 个种子，默认 seed=101；每个待评模型版本 100 次闭环，默认 LIBERO 50 次、RoboTwin 50 次。** 两平台分别训练，对应两个平台权重共同构成该方法版本的这轮 100 次评估。不是所有模型共享 100 次，也不是每个平台各跑 100 次。

- 新增网络仅用官方 RawVLA-Bench 训练池。保留完整 ConvNeXt-B，不用 tiny 网络替代；主链直接 RAW→特征→VLA。S0 补采、受控渲染、S1 在扩展数据上训练的权重不进入主结果；不补采新训练轨迹，不用 RGB、外部深度／光流教师或隐藏几何标签。固定预训练 π0.5 是共同后端起点。
- 旧 v1.0 的局部几何出口、三种子、全清单评测、逐文件 hash、独立解包全量比对和“证明已收敛”要求均不作为本阶段门槛。不要继承旧 S1 的三种子调度队列，也不要把旧 handoff 中 `entry_ready` 当成事实裁决。
- 保留简单必要的数值保护：正方差、有效 mask、发现 NaN／Inf 时定位原因、checkpoint 能加载并继续运行。不要 `nan_to_num`、宽泛 `except`、返回零动作或假成功来隐藏模型错误；不要建立多层 schema／验证器／证据等级系统。关键数值或因果性错误需要修复，正常训练曲线波动和不显著的结果直接报告。
- 不设自定义综合分、成功率提升最低线、方差覆盖率硬线、连续若干次 plateau／斜率判定、显著性 p 值出口、固定单元测试数量或 bitwise 一致性要求。不要求网络先证明物理原因可辨识才开始整体训练。
- 代码版本、配置、数据拆分、checkpoint 路径、训练步数和实际成功／尝试数足以做本轮普通追溯。作者仓库 commit 是方便找代码的版本线索，不是运行前必须完成的哈希工程。
- 原则上每次只改本阶段所需部分，保留别人和用户的未提交工作；不自动提交、推送、升级共享环境或停止其他项目进程。完成当前阶段后写清已做／未做，停止在交接点。

本地对应 checkout 是 `/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`，分支 `lzh/n04-egomotion-separation`。如果你已经在远端同一源码 checkout，就在当前 checkout 工作，不因为本地绝对路径不存在而停工，也不要在服务器上照抄创建本地目录。先简短阅读 `AGENTS.md`、`README.md`、`docs/status.md` 和 [v1.1 方案](../../plans/N04_full_development_v1.1.md)的相关章节；发生预算冲突时以上本轮约定优先。

## 本轮先选什么模型

从 S3 已有结果选择一个当前 D checkpoint 和至少一个已经可用、最有信息量的对照，通常 C；若本轮只具备 D，就先交 D 的 100 次和比较缺项，不现场扩展成大量训练任务。A／B 或标准 RKN 已有可比结果可一起整理，没有就留空。

用已有训练／开发记录选一个权重并写明路径、训练步数、输入、seed101 和平台；不需要证明已收敛，也不用重新搜索所有 checkpoint。开始这轮评估后固定该权重和现有执行配置。不要一边根据同一批失败不断换权重一边称独立评测；如果它实际上是反复开发使用的 100 条，就直接叫开发评估。

沿现有 `eval100.json` 逐模型运行；已有完全相同版本和配置的有效 episode 结果直接复用，只补缺失身份，不为了 S4 仪式性再跑整套。模型／动作语义改动后只重评受影响模型，不重复其他方法。若 S3 尚未完成对照，只报告能比较的部分，不回到重写完整架构。

## 怎样执行和展示

1. 用现有仿真执行器、真实策略、固定成功判据跑闭环。不能拿离线动作 loss、mock 环境或预录视频代替真实 rollout。LIBERO 用原 7 维动作，RoboTwin 用 MEAN_STD／既有 14 维关节排列，重规划和相机沿已跑通的协议。
2. 每次 episode 结束立刻追加普通结果行：平台、模型／checkpoint 路径、task／初态、光照域、环境 seed、success、状态／错误原因、视频路径（有则写）。不写逐帧 hash，不要求完整仿真资产校验，不把正常失败移出分母。
3. 监控同步显示该模型总计 n/100、s/n、错误数和平台各 n/50；某方法没评估就“未评估”。已有结果不要显示为运行中，不用进度动画猜执行状态。用户无需等待长对话输出才能知道成功率。
4. 结果表最少列模型、输入、平台、seed、训练 step、successes／completed、成功率；合计 50／50 数字可附。只有一个训练种子，不报告跨种子标准差；统计区间可选，不是必须通过的显著性检验。用表格和任务视频讲结果，不创建综合评分。
5. 从已有录像／叠图挑代表性成功与失败。优先一个“弱 RAW 线索被使用”的例子和一个“旧记忆／对应错误导致动作错误”的例子；没有相应现象就如实展示实际行为，不能为了故事挑换任务或制造标签。读出 attention、质量／模式图与原始帧同步，区分模型估计与真实动作。

核心研究问题依可用证据逐个回答：D 相对 C 有无变化；RAW 相对同源 RGB 有无变化；是否能观察到条件预测／修订影响实际动作；开销和失败在哪里。少了 A／B 就限制 RAW 特定归因，不要求先补全所有方法才能写结果。相同架构在两平台表现不同直接报告，不为合计好看隐藏负例。

## 远端训练和长对话的运行方式

用户已指定这台服务器后续专门用于 N04 开发，连接和监控转发命令为：

```bash
ssh -p 16724 root@154.59.156.22 -L 8080:localhost:8080
```

监控服务默认绑定远端 `127.0.0.1:8080`；保持上面的 SSH 隧道后，本机打开 `http://127.0.0.1:8080/`。若会话已经在该服务器，直接使用当前源码目录，不再嵌套 SSH。8080 若被历史服务占用，保留原服务，给 N04 选择可用端口并相应更新转发说明，不直接杀旧进程。

服务器混合保留了其他开发阶段的目录、环境、权重和日志。后续新作业只用于 N04，但不因此删除或覆盖历史产物；通过下面明确的 N04 根目录、stage／platform／arm／run 名称选择输入和 checkpoint，不递归扫描全服务器后自动拿“最新权重”。恢复时先认清已有 N04 作业，避免重复启动。

以下是项目已有路径线索，实际存在就复用；迁移后允许在一个配置中改路径，不把旧路径缺失升级为算法失败：

| 用途 | 已有位置线索 |
| --- | --- |
| N04 项目／历史运行根 | `/workspace/owac/research/n04-v1`；新版 run 放 `runs/v1.1/<stage>/<platform>-<arm>-seed101/` |
| 官方数据 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/libero/raw` 和 `data/robotwin2/raw` |
| StarVLA 源码 | `/workspace/owac/rawvla/source/starVLA`，内部 Python 包 `starVLA/` |
| StarVLA 解释器线索 | `/workspace/owac/rawvla/envs/starvla/bin/python`；历史为 Python 3.10，先实际导入相关模块 |
| LIBERO 后端权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO 归一化 | 同权重目录的 `dataset_statistics.json` |
| RoboTwin 后端权重／归一化 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`、`policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 现有仿真环境／协议 | `/workspace/owac/rawvla/envs/robotwin`；`records/pi05-full-20261010-v3/` 的协议和清单 |

只需看 GPU 空闲、相关作业、磁盘和实际 Python 是否可用；不要做全盘扫描或每次从零安装。新 joint-training 模块必须和真实后端在同一进程反传，不能用两个 Python 进程传数组假装动作梯度贯通。优先沿用能加载后端的解释器；需要新依赖时用 N04 专属环境，不升级共享 StarVLA／RoboTwin 环境。不要强行把本地 `uv.lock` 的新 torch 环境灌进旧后端。桥接及其传递导入兼容实际 Python 版本；确有必要才做兼容性修改。

用简单 `rsync`／复制目录或已有打包工具把实际工作树同步到远端，包含必要未提交源码；排除数据、权重、venv、outputs 和密钥。不强制新建逐文件校验清单。为正在训练的代码留一个固定工作目录和配置副本，避免后续编辑无意改变运行中的实现。

### 双 GPU 调度和并行验证

服务器有两张 GPU。默认每张卡一个重负载 worker，最多同时训练两个模型：例如 GPU0 训练 LIBERO-D、GPU1 训练 RoboTwin-D；S3 也可并行两个独立平台／对照任务。每个进程明确设置 `CUDA_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=1`，各自使用独立 run 目录、日志、checkpoint、supervisor／tmux 名称和所需端口。进程内的 `cuda:0` 是被映射后的设备；在启动记录中写物理 GPU 编号，避免两进程实际挤在同一卡。先快速看实际空闲资源，不停其他历史作业，也不把“双卡”当作增加训练种子的理由。

验证同样允许两个 worker 并行，可一张卡跑 LIBERO50、另一张卡跑 RoboTwin50，或把同一固定列表拆成互不重叠的子列表。每个模型仍总计100次，各 worker 的分片相加为100，不是每个 worker 各跑100。各验证进程有自己的策略实例、仿真进程、端口和结果文件，仿真器另有渲染 GPU 参数时也绑定相应物理卡；结束后按 episode 身份合并，避免重复计数或同时写坏一个结果文件。不是要求先建设分布式任务平台，两个普通进程和简单结果合并即可。

可一张卡训练、另一张卡验证已保存的固定 checkpoint；两张卡都在训练时，新验证等空闲卡，不强行在同一卡叠加完整训练与仿真导致 OOM。若一个完整模型确实要占用两卡，就为该作业使用双卡并暂时不并行第二个模型，不通过缩成 tiny 来维持“两模型同时训练”。固定源码／配置后再开并行作业，避免另一个进程读到正在修改的代码。

### Hugging Face 下载与缓存

用户允许使用其 Hugging Face 凭据下载本任务需要的官方资源。凭据通过服务器的 `HF_TOKEN` 环境变量或已登录的 Hugging Face 缓存读取，在启动下载／训练进程之前配置；不把明文 token 写进源码、配置、提示词、命令日志、异常回显或监控网页，也不打印完整环境。环境中尚无 token 时，优先复用已下载资源；确需认证下载再提示补充服务器环境配置，不将其作为纯本地开发的前置。

两作业复用已有模型／数据缓存，同一大文件由一个进程下载后共享，避免重复拉取。认证不保证取消所有限流；遇429正常退避，不用多个进程不停重试。只下载实际需要且主训练允许的资源，不因具备 token 就下载外部预训练光流教师或上传任何项目产物。参考 [Hugging Face 环境变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)。

训练使用已有 supervisor 或 tmux 中的一种管理，给出用户可查看的会话／作业名、PID、日志、停止与 resume 命令。训练、监控采样器和网页服务独立运行，SSH 断开后继续。不要重复启动同一个训练，不做无限崩溃自动重启。checkpoint 至少包含模型、optimizer、scheduler、混合精度状态（适用时）、global step、seed 和配置；方便时保存 RNG／采样位置，不要求断点续训逐位重现。恢复时在正确 episode 历史上重新 burn-in，不继承另一 episode 的状态。

每次开始长运行前先把监控页启动并给出实际可访问地址。训练循环定期写进度；长对话的阶段切换和关键修改用简短文字说明，但用户无需靠对话滚动查运行状态。产物大文件留远端；本地只回传报告、小图、结果表和访问入口。

## 每个待评模型 100 次闭环的具体规则

本轮为项目初期开发评估。每个选定方法／模型版本一轮总计 **100 次闭环：LIBERO 50 次、RoboTwin 50 次**。两个平台分别显示成功数／已完成数、成功率；总计也可以显示，但标为当前 50／50 采样下的合计，不能冒充官方两平台全量成绩。

首次从已有官方任务和初态里写一份普通 `eval100.json`：每平台覆盖现有 5 个光照域，每域 10 个 episode；在既有任务池中尽量均匀选任务和初态。直接保存 task、initial_state／episode、环境 seed、光照域和平台，不设计新难度分数，不补采训练集，不必为这份表建立哈希契约。如果已有合适的固定 100 条列表就复用，避免反复改样本。

对照与候选使用同一份列表、相同相机、归一化、执行长度、重规划和成功判据。RoboTwin 用已有 `MEAN_STD` 统计和 14 维关节排列；不要按通用 YAML 猜成 min-max。LIBERO／RoboTwin 的 action horizon 和去噪次数不同于实际重规划节奏，沿用现有执行器。任务失败和超时照常算失败；基础设施错误单独保留并修复后补同一身份，没补齐时显示“完成 n/100，成功 s/n，基础设施错误 e”，不要删除失败或把错误偷偷算成功。

S2 优先验证当前 D；S3／S4 只评已经训练好且本轮选定的对照，每个各 100 次，不自动把全部强系统、消融和官方全清单排队。旧结果只有 checkpoint、输入／执行协议和这份 episode 身份确实匹配时才复用；已有 13,250 条 Default π0.5 汇总只能作背景，不能充作同源 C 或 A 的 100 次结果。

训练中常规验证用固定少量官方开发窗口的动作 loss，暂停训练看一次 100 次闭环即可；不默认每 5k 步都跑 100 次，也不要求所有 checkpoint 都跑闭环。最终交接报告成功率、动作 loss 曲线和代表性成败视频。100 次是本轮样本预算，**不设最低成功率或必须胜过对照的出口**；单种子结果不提供跨训练种子的稳定性结论。若已经用这份 100 条反复选模，称开发结果，不伪装为独立最终测试。

## 现在调用前端专家完成实时监控页

如果环境支持前端专家／子代理，把下方任务原样委派给前端专家，并由你负责接通真实数据和验收。该专家只拥有监控页面及监控读取服务，不修改训练、模型、数据加载和仿真。若没有可调用的专家角色，明确说明并由你按相同任务完成，不停工等待用户另配工具。S3／S4 优先复用 S2 页面，不各建一套看板。

**给前端专家的完整任务：**

> 为 N04 v1.1 做一个用户一眼能看懂的远程训练实时网页。用户主要想知道：现在在做什么，训练到哪一步，100 次闭环成功多少，模型在原图上看哪里，失败在哪里。只做一个中文单页，不要路由、登录系统、管理后台、复杂导航、3D 场景、拖拽布局或多层展开卡片。
>
> 先看 `src/owac/runtime/n04_s1_progress.py` 和 `n04_s1_progress.html`。可复用日志／GPU／静态服务思路，但删除三种子队列、S1 深度验收和全量评测假设，不影响旧页面。建议新文件 `src/owac/runtime/n04_v11_progress.py`、`n04_v11_progress.html`。采用现有轻量技术即可，优先 Python 标准库只读服务＋单 HTML／CSS／JS；不为简单曲线引入 React／数据库／大型图表构建。普通 Canvas／SVG 或已有小依赖都可以。
>
> 页面从上到下只放五块：①当前 stage／platform／arm／seed、真实作业状态、最后更新时间；②训练 step／名义 budget、最近 loss、步速／粗略剩余时间和 GPU 显存；③一张训练／固定开发 loss 曲线，辅助 loss 可勾选显示，标签明确是动作 loss 还是对应／预测损失；④当前模型闭环 `完成 n/100、成功 s/n` 与两平台分项，简单对照结果表；⑤最多 4 个最新可视化缩略图／视频链接，以及默认最后 20 行日志／最近一条错误。只保留一个 run 选择框和一个手动刷新按钮。手机上按顺序纵向排列即可。
>
> 双卡并行时，在页面顶部用两行简表显示 GPU0／GPU1 当前模型、stage、训练／验证、step 或 n/100；已有 run 选择框切换详情即可，不增加复杂导航。汇总各 worker 的验证分片，展示当前模型合计100次的真实进度；token 和环境变量值不能进入页面。
>
> 浏览器每 5 秒读取一次轻量 JSON；远端采样器约每 10—15 秒从训练写出的状态、JSONL 日志、评估结果和实际进程读取数据。不要让网页请求扫描大数据目录、加载 checkpoint 或占用训练 GPU。训练正常时写心跳；读不到新数据只显示“多久前更新／连接中断”，不要仅凭 loss 不降或心跳延迟判训练失败。ETA 根据已测速度估算并标“估计”；启动时不足以估算就显示“—”。训练 step 百分比仅表示当前名义预算完成度，不表示方法效果或收敛程度。
>
> 数据约定是轻量接口说明，不是复杂校验系统：`stage, platform, arm, seed, phase, updated_at, step, budget_steps, train_loss, dev_loss, step_seconds, gpu_memory_gb, eval{completed, planned, successes, errors, libero, robotwin}, artifacts[], log_tail[]`。未产生的数值用 null／“—”；别用 0 假装真实结果。需要扩展直接加字段，不写多版本 schema 迁移。例子只用于开发布局，交付页面必须读真实 run 数据，不能留下随机曲线、演示成功率或假 heartbeat。
>
> 建议提供 `GET /api/progress` 和当前 run 下的图／视频只读 URL，或者现有服务直接读取 `progress.json`；同源请求避免额外 CORS 工程。状态文件可以临时文件后 `os.replace`，JSONL 偶遇未写完的末行下次再读。网页不能写训练状态、启动／停止进程或触发自动补跑。采样／网页失败不能拖死训练，错误显示在页面即可，不吞掉训练自身的异常。
>
> 默认绑定远端 `127.0.0.1:8080`，沿用本任务上方已给出的 SSH 8080 转发，本机访问 `http://127.0.0.1:8080/`，不自行开公网入口。实际启动后交付可访问 URL、服务命令和日志路径。用真实训练让 step 和 loss 至少发生一次实际更新，评估开始后看到计数变化；刷新能恢复同一 run。若已有成功／失败样例，点开真实视频，并在原始图像上查看实际 attention／质量图叠加。没有生成的项目显示“尚未生成”，不为凑齐页面伪造数据。给出一张真实页面截图或简短浏览器验证记录即可，不做多轮像素完美验收。

你作为主编程者负责让训练和评估器主动写数据，前端专家负责展示。可在 run 目录使用 `progress.json`、`metrics.jsonl`、`eval_results.jsonl`、`artifacts/`；复用已有格式时做一层简单适配，不重写训练日志体系。模型可视化必须和输入来自同一帧，注意力图标为模型读取分布；它不是因果真值，也不能拿 RGB 重建假装 RAW 特征。

## 本轮完成标准和交付

完成标准是实际执行和清楚交付，不是性能过线：当前选定模型的每模型 100 次（或明确尚未补齐的分母）、真实结果表、可打开的成败视频／叠图、正常工作的实时页面、可重新执行的简单命令，以及可继续训练的 checkpoint。基础设施阻塞尚未修好就保留 n/100 与原因，不伪称完成；成功率低也属于有效研究结果。

报告放 `research/experiments/<日期>_n04_v11_s4.md`，小型逐 episode 结果放对应结果目录；大视频／权重留远端并写路径。检查结果表与逐条 success 能相加一致，实际加载选定模型采样一次，运行受影响检查和 `make check`。不要求多台机器重载逐位一致，不再做额外验收框架。

报告最后用普通文字写当前结论、最主要的一个失败原因和下一次值得做的实验。可以建议后续扩样／更多种子／官方全清单，但明确是后续选择，当前任务不自动启动。更新候选卡和状态，停止；不自动发布、提交或推送。

## 论文和可直接查阅的参考实现

按当前文件职责阅读，不需要把全部论文重新实现。优先复用作者模块；主训练随机初始化新增网络，**链接到作者代码不等于允许下载其外部数据预训练权重**。保留许可证和简短来源／修改说明即可。以下固定版本是查代码的定位，不需要验证每个字节哈希。

| 组件 | 论文与实现入口 | 具体复用点 |
| --- | --- | --- |
| 官方数据 | [RawVLA-Bench 数据卡](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)、[作者 NPZ loader](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py) | 字段、相机、目标时序；不要继承 PIL／uint8 RGB 转换作为 RAW 主链 |
| ConvNeXt-B | [论文](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)、[作者 convnext.py](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py) | 完整四级 Base；本地已有 vendor，不重写小网络 |
| RKN | [论文](https://proceedings.mlr.press/v97/becker19a.html)、[RKNCell.py](https://github.com/ALRhub/rkn_share/blob/7cb29e1c57307a87729fd5e20ca60a22f5798da5/rkn_cell/RKNCell.py) | 局部线性转移、upper／lower／side 协方差；本地 `src/owac/baselines/_vendor/rkn/cell.py` |
| 条件预测 | [Ac-RKN 论文](https://proceedings.mlr.press/v155/shaj21a.html)、[MTS3 acRKN.py 参考](https://github.com/ALRhub/MTS3/blob/ab971c0ec43df7664c4b727ca69d5b8dd0cab5da/agent/worldModels/acRKN.py) | MIT RKN 上按论文加控制／本体条件；MTS3 许可未明确，不整段自动复制 |
| RAFT 主体 | [论文](https://arxiv.org/abs/2003.12039)、[raft.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/raft.py)、[update.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/update.py) | `small=False`、相关体、迭代更新，初始 12 次迭代 |
| RAFT 支撑代码 | [extractor.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/extractor.py)、[corr.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/corr.py)、[utils.py](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/utils/utils.py) | 编码、像素坐标、padding、upsampling；RAW 缩放接入替换 RGB 归一化假设 |
| 自监督对应 | [SMURF 论文](https://arxiv.org/abs/2105.07014)、[smurf_trainer.py](https://github.com/google-research/google-research/blob/3ef52111c1fabc301880818f5a547570e926a76b/smurf/smurf_trainer.py)、[smurf_utils.py](https://github.com/google-research/google-research/blob/3ef52111c1fabc301880818f5a547570e926a76b/smurf/smurf_utils.py) | TensorFlow 训练逻辑需迁移到现有 PyTorch；对齐重投影、可见性和自监督目标，不只是借名字 |
| 对应损失细节 | [UFlow 论文](https://arxiv.org/abs/2006.04902)、[uflow_utils.py](https://github.com/google-research/google-research/blob/3ef52111c1fabc301880818f5a547570e926a76b/uflow/uflow_utils.py) | robust matching、遮挡、smoothness；结合 RAW 强度／噪声，不假定所有像素恒定亮度 |
| 双模式更新 | [FilterPy IMM.py](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/IMM.py)、[Kalman update](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/kalman_filter.py) | 模式先验、interaction、分量间分歧；神经状态中用 RKN 结构近似，不直接把 NumPy FilterPy 塞进反传 |
| Perceiver 读出 | [Flamingo 论文](https://arxiv.org/abs/2204.14198)、[OpenFlamingo helpers.py](https://github.com/mlfoundations/open_flamingo/blob/655f693fbfa04cd6e9a987d960654624d48917cf/open_flamingo/src/helpers.py) | 完整 `PerceiverResampler`／attention／feed-forward 结构；项目配置 6 层、256 queries |
| 防塌缩 | [VICReg 论文](https://arxiv.org/abs/2105.04906)、[作者 main_vicreg.py](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py) | 方差／协方差正则；只作为已有训练依据，不承诺物理表征定理 |
| 统一后端适配 | [LoRA 论文](https://arxiv.org/abs/2106.09685)、[作者 loralib](https://github.com/microsoft/LoRA/tree/main/loralib) | 现有库可用就复用；相同 VLM 注意力投影 rank=16，各臂一致 |
| π0.5 实际方法 | [PI0.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI0.py)、[PI05.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py) | `embed_prefix`、训练前向、动作采样；不要发明 `forward_from_features` 等不存在 API |
| π0.5 细节 | [OpenPIPaliGemma.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/vlm/OpenPIPaliGemma.py)、[OpenPI_ActionHead.py](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py) | 前缀 mask／位置／语言缩放、动作目标、flow matching；复用本地特征桥 |
| 执行器与输入 | [LIBERO raw.py](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/libero/rawvla_bench/raw.py)、[RawVLA policy](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/baselines/rawvla/rawvla.py) | 现有观测到动作执行路径；同源 RGB 对照的 ISP 参考入口，主 RAW 链不能借用 RGB 图像 |
| RAW 时序背景 | [EMVD](https://arxiv.org/abs/2103.05407)、[Seeing Motion in the Dark](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Seeing_Motion_in_the_Dark_ICCV_2019_paper.html)、[Matching in the Dark](https://arxiv.org/abs/2109.03585) | 弱信号／对应的依据，都是参考，不新增必须训练的依赖 |
| 轻量监控 | [Python ThreadingHTTPServer](https://docs.python.org/3/library/http.server.html)、[浏览器 Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)、[Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) | 本地只读服务、轮询、简单曲线，不搭大型前端系统 |

直接近邻与理论边界已在 [v1.1 方案第 7、14 节](../../plans/N04_full_development_v1.1.md)整理。不需要在编程任务里重做论文综述；新增公式写清假设和实际代码对应，不编造全局收敛保证或“首次提出”。

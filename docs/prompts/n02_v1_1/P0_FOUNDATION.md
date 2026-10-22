# 给DeepSeek：P0，建立可用于正式开发的基础

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

P0 已在 2026-10-14 经独立修补验收；当前阶段默认只做远端环境迁移、真实 inventory 与必要回归，不重写已通过基础。以下完整 P0 要求保留用于复现或补修，下一开发阶段为 P1。

先完整阅读[共同契约](00_SHARED_CONTRACT.md)、[当前实现](01_CURRENT_IMPLEMENTATION.md)和原方案P0/8/9/10节。本轮只完成P0。目标是让下一轮直接训练正式RAW观测模型，数据、上游算子和环境均有实际收据。

## 1. 先读这些实现

项目内参考：`src/owac/raw/rawvla_sequence.py`的FORMAT_SPECS、WindowOptions、RawVLASample、compute_split和`__getitem__`；`src/owac/core/encoder_contract.py`；`src/owac/backends/rawvla_features.py`的ActionProtocol；`src/owac/training/n02_raw.py`；旧数据和桥测试。

上游精确文件见[参考导航](02_REFERENCES.md)：R01的convnextv2/fcmae/sparse/utils，R02的attention/video/misc/initializers，R06的DNRI_Encoder，R09的TheseusLayer和LM，R13的particle_filter/trace_translators。采用表中SHA，不自动升级到latest。P0运行参考组件，不提前实现完整结构机制。

## 2. 建立目录、来源和运行时

1. 复用已验收的P0模块、配置目录和`evaluation/n02_full/p0.py`；仅补足本轮实际缺项。核对`--help`可列出`inventory`、`parity`、`profile`、`report`子命令；不要把这些名字写入报告就当作已有CLI。
2. `third_party.lock.json`每项保存repo/commit/具体文件hash/论文/许可证/本地获取位置/移植差异/参考运行命令。代码与权重许可证分开；首版不下载RGB权重。参考代码在ignored目录，必要移植源码和许可证跟随项目版本管理。
3. 远端根环境保留CPU检查，按`configs/backends/n02-full-runtime/INDEPENDENT_ACCEPTANCE.md`及既有锁重建远端独立环境。先以已部署StarVLA的Python3.10+GPU栈为集成兼容目标，用PYTHONPATH加载本项目兼容模块，不把要求≥3.11的根包强装进去。实际torch/CUDA版本以目标GPU和编译依赖探针冻结。SAVi JAX与Gen Julia各用隔离参考环境。
4. 若Theseus/MinkowskiEngine与该环境不能同时工作，先让原环境参考可运行；GPU前端可用单独环境训练，再对合并环境做实测。禁止无证据声称跨环境进程通信能传回动作梯度；E1前必须有同进程可微训练环境。真正需要改变集成目标时，在P0报告列明冲突、已测矩阵和建议，交接后定案。
5. 实际连接指定服务器，核对当时GPU/进程/磁盘/数据路径；使用`p0-remote-libero.json`和`p0-remote-robotwin.json`执行真实inventory，按独立验收收据复用v2 manifest与冻结anchor。200步画像在远端可用GPU上实跑，使用独立输出和作业；本地旧画像不代替远端结果。

## 3. 四分区和真实NPZ审计

新增`data.py`中的`PartitionManifest`和显式分区dataset封装，复用原loader转换。manifest包含source_file哈希、task/suite、源轨迹、照明衍生组、可用的场景/对象/采集会话、camera、partition、实际字段与缺失字段。

先从实际NPZ及来源清单恢复源轨迹身份。LIBERO的variant文件合并到同一源trajectory组；RoboTwin按任务+源episode组。可识别的同场景/采集session共享依赖时用连通分组扩大group；缺来源字段写unknown，不能宣称实现了object-disjoint。主任务适配使用task内按组约80/10/10的train/dev/calibration；另列可行的跨场景/物体留出切片。P7正式初态测试清单独立封存。

实现 `ManifestWindowDataset(..., manifest, partition, anchor_table)`，将已经选定的episode_paths直接交给数值加载层并关闭旧二次随机分区。锚点表明确episode/t/current/历史索引/action起止/masks，使多worker、DDP和恢复不会重置到相同少量样本。禁止只是改`split='train'`然后让旧compute_split再次分割。

每个平台至少读取多个任务、不同长度和照明的真实文件，实际输出：字段、dtype/range、分辨率、相机次序、时间来源、噪声可用性、语言、本体、动作单位及归一化统计hash。以下为必须验证的已知协议：

| 项目 | LIBERO | RoboTwin |
| --- | --- | --- |
| RAW键 | agentview_raw_uint8 / wrist_raw_uint8 | head/left/right_camera_raw_float32 |
| 转换 | FP32后除255，不能按帧max拉伸 | 按原浮点尺度，不能除255 |
| 当前本体 | robot_states[t]，8维 | actions[t]，14维作者代理；robot_states的16维末端量不用作关节 |
| 动作监督 | actions[t:t+10] | actions[t+1:t+51] |
| 正式π0.5统计 | quantile及原夹爪规则 | 显式checkpoint MEAN_STD/zscore和epsilon，不能沿用loader的通用默认minmax |
| 实际执行窗口 | 5 | 50 |

尾部padding不进loss；历史padding可参与张量形状但不增加证据。真值、未来动作、RGB图像不进入InferenceInput。现有loader把executed_action标为不可用，保持该事实；受控模拟若有真实已执行动作日志，另行声明。

## 4. 建立新机制数据而非放大旧16场景

`synthetic.py`生成器先实现可复现的2D/2.5D分层场景、实体姿态/速度/遮挡、相机位移、曝光/响应和噪声。输出直接为声明的传感器测量：CFA模式在采样点应用滤色响应，RAW3为另一个合成观测模型；不能从sRGB反马赛克宣称真实CFA。简单几何是数值与受控机制集，正式具身价值仍需P7。

每条序列返回互不混入的数据对象：InferenceInput序列、TrainingTargets、EvaluationTruth。场景包含多形状/尺度/位置/背景；train与封存test留出形状组合、数量、遮挡模式和交互结构。事件时间独立随机：曝光/噪声/±1–4像素相机位移、真实物体运动、遮挡、进入/离开、误并/误分、身份交换和关系变化。为P1先使用固定候选数，P3再激活变结构输入。

输出seed→场景/测量可复现hash、独立场景数、帧数、序列数、增强视图数；增广不冒充独立场景。建设目标沿用方案：P1至少100万训练帧，P2以后10万条16–64帧序列；P0先给可运行生成和成本画像，小样本验证不称全量数据已建成。dev/calibration各至少5000条目标，阶段测试每机制1000场景；生成清单先于正式训练冻结。

## 5. 原算子一致性，按此顺序完成

1. ConvNeXt Base block、GRN、多尺度输出：同参数FP32，逐层对照上游；先3通道/方形原输入，再验证4通道和矩形适配。测试GRN的全空间支持。FCMAE修复所有硬编码3和方形patch/mask；patchify→unpatchify逐值一致。
2. FCMAE稀疏encoder参考：固定mask与权重，检查掩码位置不泄漏、有效位置数和稀疏GRN统计。优先复现官方稀疏算子；若实现mask-aware dense移植，逐层验证有效坐标、归一化和梯度一致，并标注计算开销不同。普通零填充dense仅能另命名对照。
3. SAVi：分别导出JAX SlotAttention/GRU/predictor的输入、参数、输出和梯度；PyTorch对应参数布局，禁用随机性。原无mask路径作parity，新有效mask路径作为明确扩展做独立测试。用原width先对齐，再用512/16槽/4 predictor块profile。
4. dNRI：复现固定小系统的encoder prior/posterior和decoder一步；验证在线只forward prior，未来帧扰动不改变已产生输出。
5. Theseus：运行官方simple_example及有解析解的线性二次例，做FP64解/梯度有限差分；Gen/particles跑同一小离散模型的枚举与SMC，保存实际分布误差。

起始容差：FP32前向atol1e-5/rtol1e-4，输入/参数梯度atol1e-5/rtol1e-3；FP64解析解误差1e-8，数值梯度按步长扫描。先用原环境重复性与条件数验证这些值是否合理，必要修订仅在开发fixtures上记录原因并冻结。不能临时放宽容差来吞掉错误gate或时间轴。

## 6. 资源与完整性测试

在实际可执行的正式宽度模块上warmup后测200更新：参数、每模块耗时、吞吐、峰值allocated/reserved显存、有效batch、序列长、精度、GPU型号和软件版本。未实现的P3—P5不能编一个“全模型实际profile”；写模块实测、组合估计、未测三栏，各后续阶段再实测其200步。由实测提出并冻结决策p95与状态内存预算。

P0至少完成：分区无交叉/variant同组/读回manifest不重分区；两平台时间边界；未来动作扰动不改输入；部分batch reset；CFA相位crop一致；多相机不同形状；真实NPZ对作者索引；上游算子parity与小批次短训趋势；200步profile。

以下CLI已存在，在远端设置好`N02_PY`与`PYTHONPATH`后执行。完整parity使用按已验收锁重建的SAVi/数值参考解释器`N02_REF_PY`，显式指定本次带入的fixtures；这检验远端移植对同一参考结果的一致性，不冒充远端重新执行全部上游导出：

```bash
"$N02_PY" -m owac.evaluation.n02_full.p0 inventory --config configs/encoders/n02_full/p0-remote-libero.json --output outputs/n02-full/p0-remote/inventory-libero --max-files 5 --evidence-kind real_cache
"$N02_PY" -m owac.evaluation.n02_full.p0 inventory --config configs/encoders/n02_full/p0-remote-robotwin.json --output outputs/n02-full/p0-remote/inventory-robotwin --max-files 5 --evidence-kind real_cache
export N02_REF_PY="$PWD/outputs/n02-full/p0/parity-venv/bin/python"
"$N02_REF_PY" -m owac.frameworks.n02_full.components.parity_suite --fixtures outputs/n02-full/p0-independent-audit/fixtures --output outputs/n02-full/p0-remote/parity
"$N02_PY" -m owac.evaluation.n02_full.p0 profile --config configs/encoders/n02_full/p0.json --device cuda --output outputs/n02-full/p0-remote/profile
```

用已锁定对应runtime的python执行，不假设根uv启动CUDA。P0报告给出实际命令和所用绝对解释器路径；交付manifest、source lock、环境锁、parity、资源、测试、P1可直接使用的dataset/训练配置。P0未满足的真实资产或移植项写清楚；到此结束，不自动启动100k训练。

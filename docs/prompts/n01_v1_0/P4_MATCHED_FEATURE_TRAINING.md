# P4 成熟特征接入与匹配充分训练

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> **已被 v1.1 取代（2026-10-15），不再执行本提示词。** 当前正式路线见 [v1.2 完整方案](../../N01_full_development_plan_2026-10-15_v1.2.md)。旧局部阶段正文仅保留供追溯；新 P3 必须整体交付编码器、真实任务训练与开发闭环，不能按旧步骤继续或只改编号。新执行包尚未编写。

只执行 P4。读取[共同契约](00_SHARED_CONTRACT.md)、P1–P3 实际交付、[DBSR 源码](../../../src/owac/baselines/burst_features.py)和[旧匹配训练](../../../src/owac/evaluation/n01_burst_training.py)。本阶段回答：在充分训练的成熟融合之上，来源规则是否减少错误历史拖尾。冻结干预与匹配重训分别做，主要结论来自匹配重训。

## 1. 实现文件

新增 `frameworks/n01_full/encoder.py / content.py / adapters.py`、`training/n01_full.py`，扩展 `evaluation/n01_full.py`；新增 `runtime/n01_full_job.py` 负责可恢复训练矩阵和 source bundle。配置为 `configs/encoders/n01_full/p4_{development,matrix}.json`。复用 P2 早期预检已创建的 adapters，按实际接口补全，避免另造一条数值不同的输入路径。

`N01FullEncoder` 实现共同契约外部 API；内部明确分为 ledger → correspondence → support decisions → content encoder → fusion → readout。支持/对应由 P3/P2 提供；P4 首版冻结 motion 和解析规则参数，训练 DBSR/任务头/共同桥。所有方法使用同一 motion checkpoint 和校准信息；联合 motion 微调作为后续独立实验，不混入主矩阵。

## 2. 内容通路这样接

默认 `source_features`：对每个结构域，先得到 accepted 源采样和角色；对被拒 RAW 置无效，再运行共享 DBSR 完整骨干；映射到当前参考系后调用成熟融合器。重新计算 base/consensus、logits、softmax 和最终读出，不能仅将旧权重乘新 mask。

空间撤回首版采用 **按域掩码后整源重编码** 的全量参考。若一个源帧只有局部被撤回，也必须消除卷积感受野中该区域的内容影响；采用声明的无效零填充，只有合法源进入共享网络。源特征复用仅在源 mask/参数/映射版本完全相同且依赖合法时进行。不要把支持图硬塞进每一层卷积重写 DBSR。

`domain_content` 为第二通路：P1 系数在声明 CFA 相位和坐标上生成 RAW 域描述，使用同一个骨干编码作为额外节点。它是原源派生的内容，不计为额外独立测量。必须与同节点数的 `duplicate_current` 控制比较。先完成 source_features 主矩阵，再由开发集决定是否启动额外通路，不一次性扩大所有组合。

DBSR 四通道/三通道 stem 分开：在新适配器替换 `upstream.init_layer` 的输入通道，保留 64/9/512 和融合层；C=3 不复制第四通道。新 stem 从相同 seed 初始化，所有三通道基线同样适配。CFA Phase/validity 逻辑不能无条件施加到 pseudo-RAW。公开 vendor 原定义保留，差异写在适配器和 provenance 中。

## 3. 按顺序落实主矩阵

| ID | 内容/来源规则 | 必须共享的因素 |
| --- | --- | --- |
| B0 | 当前帧完整 DBSR 特征 | backbone 初始化、任务头、监督和有效 batch |
| B1 | 固定历史等权融合，激活残差块补容量 | 复用 `make_models` 的真实 active capacity 思路，记录参数差 |
| B2 | 成熟 DBSR 学习融合 | 全部可用合法历史，与 N01 相同 motion |
| B3 | 同 B2 + P3 标准检测/整域 reset | 同 detector/calibration、同样调参预算 |
| N-T | 来源级时间支持，全量重算，空间关闭 | 同 backbone/fusion/头，首先验证这一组 |
| N-S | 固定时间窗口，空间域规则 | 与 N-T 正交的来源规则 |
| N-TS | 时间和空间共同支持 | 完整支持主组 |

三种子 101/211/307，共 21 个主训练运行。先在开发集跑 B2/B3/N-T 小矩阵确认数据流、拟合和来源利用，再开完整矩阵。`N-stale` 首先是 N-TS 同权重同决策的污染诊断，故意保留旧依赖；不要作为可部署主组混入平均。若使用 domain_content，再加同节点数/同参数当前内容控制。

新模型工厂的method映射固定为：`B0=current`、`B1=mean_capacity`、`B2=dbsr_reference`、`B3=dbsr_reset`、`N-T=temporal`、`N-S=spatial`、`N-TS=spatiotemporal`；所有字符串在同一枚举/配置表定义，不在训练器和服务里分别猜别名。后三个模式共用一套encoder，只显式切换支持规则；`N-stale` 是独立intervention字段。

B1 使用真正参与 forward 的参数补容量，不增加闲置参数。参数、FLOPs、内存、历史、读出容量分别报告，B0 少历史这一事实保留。主矩阵共用 data stream identity 和公共参数初值；不同支持数量带来的计算差是待测结果。

## 4. 训练器必须完整实现

参考旧训练器的因果 windows、atomic checkpoint 和 plateau，新增实际 dataset/四分区/可恢复 sampler。DataLoader 的每个抽样包含 episode、结束帧、相机与噪声 identity；跨 step 不重复固定批次。使用窗口时从 burn-in 启动并按在线同样的 step 处理，optimizer 更新后清理所有旧参数派生缓存。

训练目标先沿用可见目标定位的 heatmap loss + softargmax 评估，使用新的复杂、随机事件生成数据；看不见的目标 mask 不作为零坐标监督。辅助 RAW loss 默认关闭。需要开启时目标源从输入、motion、统计和共享特征的完整依赖中排除，另设相同辅助损失控制组。

AdamW 起始开发 LR 候选 `1e-4/3e-4`，weight decay `1e-4`，梯度裁剪 1.0；这是本项目开发候选，不是 DBSR 官方配方。以相同 2k updates 开发预算选择每臂配方，记录全部选择成本，再从正式初值运行。完整分辨率/有效 batch 继承 P0；20k 初始预算，最多 100k 的续训上限与平台规则写入 matrix config。

checkpoint 与恢复按共同契约；新增“连续 12 步”和“7 步保存+恢复5步”比较，核对所有下一批身份及参数。GPU 非确定 kernel 若有差异，记录基于重复性实测的容差与原因；不要宣称逐位相同。

## 5. 重点验收

- 各臂共享数据 hash/初始化公共层；正式推理不读隐藏事件标签/flow/目标坐标。
- 替换被撤回内容，fixed-decision 读出与有效源梯度保持；被撤回源梯度为零（固定决策路径），完整反事实另重算 motion/support。
- accepted 源的内容变化能影响特征和任务输出，不能只有元数据 gate 改了而内容没进入。
- 同一个 source 在窗口重叠/重复服务/额外节点出现时，证据计数一致；三通道真实 batch 能运行。
- 全 invalid、参考帧局部缺失、当前异常、episode partial reset，输出/梯度有限且 mask 正确。
- 三种子平台、事件曲线、B2/B3 对照、静态/异常代价、删失恢复和总成本齐全。

本地跑新测试与旧 `tests/baselines/`、RAW loader/桥回归，`make check`；实际训练依[远程说明](03_REMOTE_EXECUTION.md)在服务器跑。新增 CLI 目标：`python -m owac.training.n01_full --config ... --run-id ... [--resume ...]`，先实现并在报告中写出实际命令。

交付 `P4_report.md`、21 行训练状态表、curve/paired metrics、checkpoint/源 hash 和真实来源消融。结果可能是无独立增量，完整保存。停止在 P4 交接点，不自动把 Mamba 当作补救。

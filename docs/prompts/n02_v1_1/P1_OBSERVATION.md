# 给DeepSeek：P1，可靠RAW观测和绝对拟合

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

在远端消费 P0 v2 分区、六份冻结 anchor 与正式归一化统计；Base 预训练、measurement 训练、真实 NPZ 检查、三 seed 评价和 GPU profile 均在该服务器执行。P0 本地 profile 仅作历史参考。

先读[共同契约](00_SHARED_CONTRACT.md)、P0真实交付、原方案3.4/7-P1/8节。P0的数据分区、GPU运行时和核心算子parity通过后开始。本轮完成P1的实现、训练和评价；E1另发提示词。固定实体候选，不加入出生、拆分或学习局部调度。

## 1. 指定修改与来源

实现`observation.py`、`sensor.py`、`factors.py`、`checks.py`、最小`model.py`；训练入口`training/n02_full.py`支持`--stage p1-pretrain|p1-measurement`；评测`evaluation/n02_full/p1.py`；配置`p1_pretrain.json`、`p1_measurement.json`、`p1_eval.json`。

参考[R01 ConvNeXt/FCMAE](02_REFERENCES.md#r01)、[R10 DeepInverse](02_REFERENCES.md#r10)、[R27 NRE/BNRE](02_REFERENCES.md#r27)。项目内`baselines/n02_measurement.py`和`evaluation/n02_refutation_boundary.py`只用于重现旧反例；旧mean-residual和首帧差分不作为主概率模型。

## 2. RAW Base与掩码预训练

采用已parity的Base，返回stride8/16特征，投影为512维并附原始采样坐标、相机与有效性。模式单独配置RAW3或CFA4。CFA4以512×512 sensor crop对应256×256 packed网格为常用配置；坐标始终是sensor坐标，不能把packed的256当传感器256。矩形正式输入采用P0的padding/crop政策。

FCMAE起点mask_ratio=.6、patch32、decoder width512/depth1、`norm_pix_loss=false`。预测目标是RAW测量，缺失/坏点不进loss；饱和另报，后续密度模型处理其尾概率。掩码不得从未来帧或真值生成。输出RAW重建仅用于训练损失，不作为策略视觉输入。

AdamW起点lr1.5e-4、weight_decay.05、beta(.9,.95)，前5% warmup、cosine；有效batch256帧。开发短试只比较lr `[7.5e-5,1.5e-4,3e-4]`，其余相同，按固定dev测量误差与局部细节选择后冻结。100k正式更新×3seed，记录独立数据覆盖和等价曝光。

## 3. 固定候选条件测量模型

定义 `ObservationModel.encode(bundle)->FeaturePyramid`，`predict(world, history, query_spec)->RawPrediction(mean, variance, masks, units)`；`score(prediction, raw_query)->MeasurementScore(relative_terms, raw_nll_or_energy, local_stats)`。world描述候选的几何/结构，不是直接表示答案的±1标签。

P1受控任务使用生成器给定的**公共候选字典**，构造几何参数不同的解释；所有方法看到同一字典，候选真伪由测量决定。包含正确、错误和全部错误字典三类。候选字段只含允许的结构/几何假设，truth只进loss/evaluator。

条件RAW decoder：由历史Base特征池化context与候选embedding生成低分辨率网格，逐级上采样到query采样点，预测均值与正方差；各候选共享decoder参数、固定相同容量，禁止候选各自使用能抄当前query的任意skip。受控几何还保留解析生成器oracle作为信息上界。P2再用对象组合decoder替换条件表示，保持RawPrediction接口。

`predict`区分两条调用：

1. `fit`：给定明确measurement集合优化候选，用于状态估计，可使用该集合特征。
2. `prequential_check`：时刻t的query在观测校正前由≤t-1历史生成预测；先打分、锁定检查结果，再允许y_t校正状态。无历史首帧不产生伪独立检查；必要的masked测量检查另报经验校准，不冒充条件独立。

成像干扰η首版包含全相机共享的log曝光、每帧二维相机平移和噪声尺度，带开发集冻结的范围与先验。相机运动作用所有背景/对象投影，实体移动只作用其实体状态。参考oracle干扰与从历史估计η分列；不能从EvaluationTruth直接读η。若当前query被用于优化η，检查统计须在校准集重演同一拟合过程，标记经验检验而非精确预测检验。

## 4. 密度、相关性和绝对检查

已知高斯/Poisson-Gaussian合成模型记录其假设与数值算法。高斯连续项 `0.5*((y-mu)^2/v + log(v) + log(2π))`；量化项为上下边界CDF差的负log，顶端/底端clip为对应尾概率；有效mask不含缺失/坏点。在稳定log空间算CDF差，做极小方差和饱和数值检查。真实伪RAW不确定的噪声结构单独拟合，无法验证joint density就标empirical_energy。

多尺度网格只用于表征/统计；每个真实采样位置在目标中只出现一次。共享首帧差分仅作对照：若y0噪声方差v0，差分协方差为diag(v_t)+v0·11ᵀ；使用完整协方差或条件创新白化，不独立相乘所有差分。P1主线直接建模原始帧。

绝对检查输出 `compatible / insufficient_information / model_inadequate`，以及干扰可解释性、局部残差位置和校准版本。首版固定窗口经验检验：对正常calibration每条序列完整运行全部尺度/区域/查看时刻，统计最大标准化残差或负预测分数，冻结其95%分位阈值；测试使用完全相同的查看规则。这个阈值针对整条序列的经验误报，不称anytime e-process保证。

同时报告全图均值和局部最大值，确保旧“双亮点”不会被背景平均。异常不能直接指认某条关系错；只表示候选族/测量模型不足，P3才比较具体结构替代。

## 5. 联合训练和对照

测量阶段有效batch32条×8帧，50k更新×3seed。Base lr起点3e-5、新head3e-4、weight_decay.01，warmup5%；开发只做0.5×/1×/2×整体lr短试，冻结后正式训练。loss组成：RAW proper prediction为1；有监督候选CE为1；几何aux为.1作为首轮开发起点。数值密度求和和训练按有效标量取均值分开记录，不用一个随分辨率变动的总和压制其他loss。

先单目标RAW/相对评分各短训，再联合；记录每个loss激活数、Base/heads梯度及夹角，复查旧背景与差异监督冲突。相对评分NRE/BNRE用同Base、同历史/标签与同训练预算，各自独立训练；官方损失来源和BNRE超参写入配置。另有普通候选判别、条件似然、原帧/条件创新/配准消融；oracle只作上界。

测试包含：预测query改变但历史相同应不改变先前预测分布；未来帧/标签替换不改过去输出；双亮点检出；±2像素相机位移与物体真实移动区分；饱和/坏点/不同CFA相位；新测量与重复调用目标；mask后的梯度；方差无限增大不能无代价改善proper score。

## 6. 正式行为验收与交付

按P0封存清单，每类至少1000独立场景、三个完整seed，保留逐场景结果：

- 线索互换两侧正确≥90%；报告paired both-correct和各方向计数。
- 无信息序列≥95%满足后验漂移≤.05；这里对可验证的二候选概率以最大绝对概率变化定义，不能将任意energy softmax当已校准后验。
- 95%错先验在冻结可辨识信息量条件下，两方向纠正各≥80%；准确模型oracle先证明任务可纠正。
- 正常序列误报≤5%时结构不足检出≥90%，按曝光/噪声/位移分组。

若头部单目标通过、联合失败，按实际梯度/预测分解报告；若oracle也失败，报告信息不足；不由DeepSeek自行删目标/改阈值/恢复旧调度。足够训练判据通过后仍未达行为门槛，保留失败并停止进入P2。

交付Base和head检查点、训练/验证曲线、校准器、候选字典与数据hash、三seed原始计数、资源及阶段报告。实现CLI `python -m owac.training.n02_full --stage p1-pretrain --config ...` 和对应measurement/evaluation命令；所有`...`在最终交接中替换成真实路径。P1通过后交接E1，不自行完成后续阶段。

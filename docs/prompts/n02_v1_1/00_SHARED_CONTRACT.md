# N02 v1.1共同实现契约

这是所有阶段提示词的一部分。你是实现者，按本文件和本轮阶段提示词开发；不自行重选研究路线，不把实现通过解释为机制或机器人能力成立。本文新文件名和API是**待实现契约**，现有文件见[源码审计](01_CURRENT_IMPLEMENTATION.md)。

## 1. 工作范围与默认决策

进入 `/home/jeong/zeno/owac/worktrees/n02-measurement-completion`，读该checkout的AGENTS、README、status及[v1.1方案](../../plans/N02_full_development_plan_2026-10-13.md)。先保存当前 `git status --short --branch`、HEAD和本轮涉及文件哈希到新输出目录。已有未提交规划、边界诊断和其他任务文件保留。按本轮明确任务开发、验证、运行阶段内实验；本提示词不要求提交、推送、创建定时任务或修改冻结基线服务。

正式主线固定：ConvNeXt V2-Base RAW观测 → SAVi式循环实体 → 完整世界候选 → Theseus连续提议与完整目标接受 → 结构搜索 → 实际来源撤回和重建 → Perceiver式读出 → 固定RawVLA/StarVLA π0.5。P2—P4用全局求解/重放，P5才比较局部执行。DeepSeek不需要再为这一串组件进行选型。

同Base直接RAW读出B2、同对象模型+标准SMC/全局求解+精确来源缓存B3贯穿比较。原作者Default B0和ISP B1有独立输入边界。旧tiny只用于回归。默认RAW从头预训练，不加载RGB权重，不输入RGB教师/重建目标；原生CFA与RawVLA三通道伪RAW分模式建模。

## 2. 目录和模块责任

在 `src/owac/frameworks/n02_full/` 实现以下模块；文件内容随阶段增长，不提前堆空壳。

| 模块 | 唯一责任 | 首次阶段 |
| --- | --- | --- |
| `config.py`、`state.py` | 版本化配置与状态、ID、reset/序列化 | P0 |
| `data.py`、`synthetic.py` | 显式分区、batch与受控RAW序列；复用现有转换 | P0 |
| `observation.py`、`sensor.py` | Base多尺度、测量分布、已声明成像干扰 | P1 |
| `factors.py`、`checks.py` | 唯一目标、拟合项、独立预测检查 | P1 |
| `entities.py`、`solver.py` | 预测校正、几何变量、完整目标接受/回退 | P2 |
| `hypotheses.py`、`proposals.py` | 世界粒子、SMC、结构编辑、近似搜索 | P3 |
| `relations.py`、`provenance.py` | dNRI因果提议、G/J、支持集、重放 | P4 |
| `model.py` | 同一因果步进，组织各模块与预算 | P1开始 |
| `readout.py`、`policy.py` | 整世界读出、版本化token source/动作缓存 | E1开始 |

正式训练入口 `src/owac/training/n02_full.py`；评测包 `src/owac/evaluation/n02_full/`；配置 `configs/encoders/n02_full/`；测试 `tests/encoders/n02_full/`；理论 `research/theory/n02_full/`。上游移植的必要算子可放 `n02_full/components/`，每个文件注明固定URL、SHA、许可证、差异和一致性测试。第三方完整参考checkout放 ignored `third_party/`；版本锁放可追踪的 `configs/encoders/n02_full/third_party.lock.json`。

复用 `owac.core.encoder_contract` 的MeasurementFrame/EmbodimentContext/VisualTokens/ComputeBudget；不复制旧RAW数值转换或动作协议。公共loader需要新增显式manifest消费入口时保持原调用行为；新四分区逻辑不伪装成旧SplitManifest。

## 3. 张量、身份和多视角

记 `B` 为episode batch，`T`为真实时间，`V`为相机，`K`为世界候选，`N`为实体槽，`D`为实体宽度。不能将T或V折进B后把相机/帧当成独立episode。

```python
# 接口语义固定，具体dataclass可增补类型；不要求一次性实现全部字段。
ObservationBundle(frames: dict[str, MeasurementFrame], context: EmbodimentContext,
                  episode_ids, anchor_ids, new_time_mask)  # 一个决策时刻全部可用相机
InferenceInput(bundle, declared_prior=None)              # 无隐藏真值/未来动作
TrainingTargets(actions, action_mask, optional_geometry, optional_relations)
MeasurementKey(episode_id, camera_id, source_frame_id, acquisition_id)
MeasurementRecord(key, raw_ref, calibration_version, masks, noise_metadata)
FeatureLevel(values, valid, sensor_coordinates, support_ids, level_id)
EntityState(entity_ids, slot_hidden, existence, geometry, geometry_frame, version)
WorldHypothesis(world_id, entities, associations, relations, nuisance,
                score, score_kind, ancestor_id, version)
ObjectiveKey(measurement_set_hash, model_version, structure_version,
             calibration_versions, initial_prior_version)
StageResult(implementation, numerical, behavior, closed_loop, evidence_paths)
```

`source_frame_id`含实际源身份；照明变体属于相同split组，但不同测量记录仍有不同acquisition ID。重传同一帧只保留一份；校准变更替换该测量的派生因子版本，不新增统计证据。batch索引不是持久身份。reset支持按batch mask清空指定episode，其他episode不变。

各相机RAW分辨率可以不同，输入用字典，编码后token序列拼接并携带相机ID/时间/原始坐标/有效mask。所有相机视图属于同一世界候选，MAP选择一次，不能每个相机各选一套不相容假设。P2先用联合多视角feature序列校正共享slots；没有外参时只定义视角条件的2D几何与学习关联，`metric_3d_available=false`；有真实标定/监督的机制集才能声明米制3D。未知外参不填单位矩阵冒充测量。

首版主体不添加持久网格GRU；保存测量、必要多尺度特征与对象状态。世界数起点K=8，活动实体槽N=16，D=512；背景解码分支不计为物体，空槽与暂时不可见不同。容量上限在配置中，超过容量报告暂缓/模型不足，不以假对象吸收。

## 4. 因果步进和不重复证据

```text
step_bundle(bundle, state, budget):
  按episode reset；登记本时刻所有新MeasurementKey
  用历史状态预测本时刻（仅new_time_mask为真者执行一次）
  在读取本时刻目标测量前形成prequential预测/检查用分布
  编码本时刻RAW，形成固定测量集合的ObjectiveKey
  校正实体，比较成像干扰解释与结构不足
  在同一ObjectiveKey下迭代连续求解/结构提议，不累计第二份likelihood
  结构采用后更新版本/来源；必要时重放重建
  输出世界集合、最好已见解、不确定性/剩余残差/成本
```

初始阶段保留完整短序列窗口并整体重算；窗口长度、可用历史与对照一致。切窗不能同时把旧posterior当prior又保留其原因子。P4增加窗口外检查点与重放。最初先实现支持完整序列的正确参考，再做有明确摘要语义的压缩。

优化模式 `best_seen` 与概率采样模式 `sampling` 分开：前者保持完整目标下降；后者允许合法的上坡MCMC步。输出 `score_kind=normalized_density / empirical_energy / importance_weight` 等真实类型。低能量softmax不是自动正确的模型后验。

## 5. 首版网络、测量与求解决定

Base采用官方depths `[3,3,27,3]`、dims `[128,256,512,1024]`；多尺度至少保留stride 8/16，输入常用256×256。RAW3直接3通道，CFA重排4平面并保留实际相位。输入padding/crop有坐标映射；RoboTwin的240×320不能无记录地拉伸。先使用stride对齐padding，256方形crop仅作为预注册训练增广，不抹掉全视野的正式评测。

校准/噪声/方差、能量累计与LM从FP32开始；数值参考FP64。混合精度仅在误差检查后启用。ConvNeXt GRN具有空间全局依赖，SAVi注意力也可能全局混合，J必须如实登记。P5的局部优化首版不承诺只编码局部RAW。

物理可计算合成模型使用归一化测量密度；真实伪RAW若相关噪声未知，模型明确为经验预测能量。原像素/采样单元只进一个目标集合，多尺度用作表示和检查，不把重叠尺度似然相乘。训练loss可取每有效标量均值，推断log-density按已声明概率模型求和，两者名称和温度不得混用。

方差采用正参数化并单列传感器方差与模型预测不确定性；合法饱和以尾概率、量化以区间概率处理，缺失/坏点不贡献测量项。高方差逃避拟合通过独立校准、方差统计和均值误差共同检验。

Theseus LM只提议固定结构的连续步。其二次子问题可固定当前方差/鲁棒权重；步的接受重新评估完整E（含log方差、饱和、初始与结构先验）。未满足下降则回溯完整E的梯度步，或保持最好解并记未收敛。不得把任意标量NLL直接塞进残差再平方。

## 6. 训练、状态与统计

三个正式种子先用 `[1103,2207,3301]`，P5新增 `[4409,5501]` 补齐全部上游训练；这些是本交接建议，P0固定后不因结果换种子。数据生成种子由partition+scene_id派生，和模型种子分离。同seed跨阶段续训，新增/累计更新数分别记录。

所有训练调用与在线相同的 `step_bundle`；监督通过TrainingTargets传入loss。真实NPZ仅能把审计确认的字段作为context：RoboTwin `actions[t]`作为作者约定当前关节代理，不扩大为已执行动作历史；`actions[t+1:]`只做监督。状态未知/动作未记录时availability为false。

检查点保存模型/优化器/scheduler/AMP/RNG/数据采样器位置/版本/分区哈希；流式快照另保存实体、粒子祖先、支持、因子、重放锚点和策略缓存版本。先用关闭dropout、固定选择的确定性重放测一致性，再测随机选择有记录的重放。跨TBPTT仅detach梯度，不reset证据。

优化起点AdamW，梯度裁剪1.0；具体学习率由阶段列出的有限开发网格选定，不能把未执行的数值写成“已调优”。每5000更新做固定开发行为评估。正式预算沿用方案：P1 100k+50k；P2/P3/P4各100k；P5 50k；E1/E2各≤2k；P6-A 20k起点，P6-B 50k起点/100k上限。达到步数不自动判定充分训练，按方案8.3检查覆盖、三seed行为、梯度、学习率敏感性和趋稳。

选择配置用dev；阈值用calibration；阶段封存测试只在配置冻结后运行一次。P1/P2等封存机制子集与P7最终闭环测试独立；看到封存结果后修复算法需要新版本和新的独立确认集，旧结果保留。成功计数、分母、缺失项、簇级bootstrap区间和每seed结果都保留。P0开发实验可以确定合理容差和预算，正式测试后不降低门槛。

## 7. 每阶段统一交付

每阶段在 `research/experiments/` 保存中文报告和小型 `results/<run>/` 清单；大数组/权重/日志放 `outputs/n02-full/<stage>/<run_id>/`。报告依次写：本轮原要求、实际实现与文件、复用来源及差异、实际命令、数据/种子/资源、原始计数与区间、失败与下一交接条件。至少有 `run_manifest.json`、`metrics.json`、`stage_decision.json` 和可运行复现命令。

`stage_decision.json`分别记录 `implementation`、`numerical`、`behavior`、`closed_loop`，值用pass/fail/not_run/incomplete；列出证据路径与代码/config/data SHA。DeepSeek可报告测得事实，不自行把“pass”写成总体研究验收。

每轮按改动运行目标测试及 `make check`；跨数据/后端公共接口时加旧回归。不安装整套无关环境或无限加通用异常包装。重点是上面具体数值、因果、概率、状态和行为契约。缺真实数据/运行时可继续完成独立本地实现与测试，真实部分报告未运行，不用stub替代收据。

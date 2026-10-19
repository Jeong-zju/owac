# N02 v1.2 P2 独立验收与修补

> 后续更新（2026-10-16）：当前修补版本为 `n02-v12-p2-causal-r4`。r2 和首轮 r3 画像还存在测量查询行列/像素中心错误，其数值保留历史身份。现行代码修补、从头零更新画像见 [r4 报告](P2_fresh_profile_20261016.md)，训练由用户自行开启，使用 [r4 用户入口](P2_r4_user_training.md)。下文旧启动/恢复命令不可直接用于 r4；原 P2 未通过的结论保持。

2026-10-16，Asia/Shanghai。结论：**DeepSeek 原交付未通过 P2 独立验收。已修复发现的执行和评测缺陷，并完成真实数据、完整 Base 网络与真实 π0.5 的有界复验；修复版尚未完成 1000 步 P2 训练，不能进入 P3。**

原来的 1000 步训练与检查点真实存在。问题是训练时的动作分支只看当前帧，评测时却把历史逐帧传入；测量约束还漏掉了大部分像素，并把三个通道平均。因而“完整历史推断网络已按方案训练”和“P1 六类缺陷全部修复”不成立。这是实现与证据问题，不能据此否定 RAW 或 N02 的研究价值。

本次没有启动 P3、P4、100 次闭环，也没有提交或推送。原日志、检查点和失败结果均保留。

## 1. 按原始要求逐项核查

依据为 `docs/prompts/n02_v1_2/P2_FULL_ENCODER.md` 和 v1.2 方案；代码基线 HEAD 为 `861c46f67fff552c81fff89c5116c63b815224c5`，DeepSeek 交付是未提交工作树。修补路径标记为 **`n02-v12-p2-causal-r2`**。

| 原始要求 | 原交付实际行为 | 独立判断与修补 |
| --- | --- | --- |
| 8 帧因果历史影响动作 | `microbatch/evaluate_dev` 只把 `anchor_bundle` 的当前行交给 token source，后者每次建空状态。其余行只服务辅助损失；评测则按时间传递状态 | 不通过。动作训练与评测共用 `sequence_v12.replay_inputs`，逐条处理不同 acquisition；padding 重复不计为新观测 |
| 测量约束不能重复 P1 固定查询盲点 | `arange(H*W)[::8][:8192]`。240×320 的每相机目标只有 8192 个像素，覆盖 10.67%；列均为 `col % 8 = 0`，最后采样行为 204，底部 35 行漏掉；三个通道取均值 | 不通过。正式四配置改为 stride=1、无数量截断、逐通道预测/评分；区分 sensor coverage 和抽样集合内部 coverage |
| 看到当前图像之前锁定因果预测，再做检查 | round-0 已经过当前 encoder 和 corrector，却标为 `causal_pre_locked`；选哪个候选也用当前帧修正后的 q | 不通过。预测只使用前一时刻保存的空间特征和状态；候选选择锁定在前态支持上。没有历史则显式 invalid |
| 校准和评测使用相同统计定义 | 训练从空状态打分，评测用历史状态；评测把源帧号改为 0，影响 decoder 的时间条件；训练启用 local maximum，评测关闭 | 不通过。真实源帧号不改，只给诊断使用相对轴 0；共用 residual 计算、ViewingRule 和样本选取 |
| 撤回后从有效历史重新建立状态 | `_scan_history` 从已受旧观测影响的 `state.slot_hidden` 开始，又用当前 evidence 扫旧 token，没有完整重放原先的接受/拒绝步骤 | 不通过。保存可重放输入，撤回后从空状态逐步重算真实张量和缓存；与不包含被撤回观测的干净运行逐元素比较 |
| 同源 RGB 对照在训练和评测保持同一输入定义 | 训练应用固定 ISP；评测入口忽略 representation，直接使用 RAW。干预也不能在 RGB 转换之后冒称同一 RAW 干预 | 不通过。共用 ISP；评测 source 明确应用表示转换，干预先作用在原 RAW，再进入各臂转换 |
| 真实 VLA 任务指令与 tokenizer 一致 | 训练读 NPZ metadata，评测用数据目录名拼任务文字、自写 tokenizer 格式，失败时返回零 token | 不通过。共用官方样本任务指令，调用策略自身 tokenizer；缺少指令或 tokenizer 错误保留失败，不用占位 token |
| 真实图像可显示 | `_image` 将 ndarray 原始像素字节直接 base64，声明为 PNG 却未编码；SVG/HTTP 200 及含 `<image>` 的测试均漏检 | 不通过。统一输出真正 PNG；对诊断目录已保存像素做无损格式修复，新增解码后逐像素一致性测试；旧原始图不覆盖 |
| 保存恢复可复现 | 原验证只比较固定 dev 输出；训练的局部 noise/time generator 没有保存 | 不完整。保存局部 generator；复验同一个随机训练后缀的 loss、token 和实际更新参数 |
| 正式训练池与更新记录可追踪 | cursor 顺序走 anchor 表，首轮预算只读前部；恢复会重置局部 flow RNG。全 padding 跳过数又被当作连续失败数 | 修补。按 seed/epoch 随机排列 anchor block，包含最后部分 block；保存 cursor 与 RNG，连续 padding 计数独立于累计计数 |
| 先验与权重公式一致 | 正先验确实进入 log_p0；但合法零先验被改成 1e-12，能量足够强时仍可能产生非零支持 | 保留已修的正先验接线；补上零质量支持限制，零先验保持零支持 |

原始执行探针使用实际移植组件的缩小配置，目的是隔离调用语义，**不是 Base 网络能力实验**。它在修改历史、保持当前行不变时得到 token 最大差 **0**；采样覆盖与坐标直接由原 `build_query_sets` 返回。见 [原始探针收据](independent_audit_20261016/original_execution_probe.json)。

## 2. 修补后的实际执行链

一条真实轨迹的不同历史帧依次进入同一套网络，前一时刻状态传给后一时刻。到当前帧之前，模型用历史预测当前 RAW；随后才读取当前 RAW，计算所有有效通道的残差，比较候选、尝试状态修正，最后把当前空间特征与修正状态交给 Perceiver 和 π0.5。训练与评测执行同一个过程。

为控制显存，历史前缀在当前帧边界 detach，即 **one-step truncated BPTT**。历史真实影响当前动作表示，但本轮不声称动作损失跨 8 帧完整反传。当前帧 encoder/corrector/readout、因果预测器和 decoder 都保留实际梯度。

变化集中在 `inference_v12.py`、新增 `sequence_v12.py`、`model_v12.py`、`checks_v12.py`、`data_v12.py`、`n02_v12_tokens.py`、训练/评测/可视化入口、四个配置和对应测试。骨干仍为官方来源 ConvNeXt V2-Base，未改为 tiny。原四配置另存 [original_configs](independent_audit_20261016/original_configs/)。

新增每次更新的 `optimization.jsonl`，保存 action/masked/causal 各损失、裁剪前梯度范数和样本身份。非有限 loss/梯度在 optimizer 写入之前报错。此改动增加定位证据，**不代表已经解决有限但巨大的 loss 尖峰**。

## 3. 真实权重、数据与数值复验

服务器仍为 `93.91.156.102:51584`，checkout 为 `/workspace/owac/worktrees/n02-measurement-completion`，GPU0/GPU1 分开运行 RoboTwin/LIBERO 诊断。两边均使用实际官方窗口、正式统计和真实 π0.5 权重；源数据与 P0 manifest 不改。

| 检查 | RoboTwin | LIBERO |
| --- | --- | --- |
| 前端 | 完整 106,324,683 参数；从原 1000 步前端/辅助分支取权重，优化器重新建立 | 完整 Base 前端；随机初始化，真实后端冻结 |
| 修补后动作执行 | 2 次诊断更新，保存 step=2 | 实际 forward/backward，0 次 optimizer 更新 |
| 历史 | 8 条不同 acquisition | 8 条不同 acquisition |
| 每相机逐通道 query | 230,400 = 240×320×3 | 196,608 = 256×256×3 |
| sensor coverage | 1 / 1 / 1 | 1 / 1 |
| token shape | 1×768×2048 | 1×512×2048 |
| 训练/评测 token 最大差 | **0** | **0** |
| 训练/评测 RGB 输入最大差 | **0** | **0** |
| 峰值显存（实测探针） | 48,262,237,184 B，约 44.95 GiB | 28,007,692,288 B，约 26.08 GiB |
| 闭环 | 0 | 0 |

RoboTwin 保存点为：

```text
outputs/n02-v12/P2/independent-audit-20261016/robotwin/checkpoint_0000002.pt
```

**step=2 是修补后的诊断步数，不能把原来的 1000 步改名为修补版 1002 步。** 另执行了同一个一步随机训练后缀的两份副本用于恢复检查，两份 loss 差、token 最大差、实际更新参数最大差均为 **0**；这些副本没有另存为正式训练进展。旧路径 checkpoint 禁止静默作为修补版 optimizer resume。

校准在修补后的固定权重、同一批真实 calibration 锚点、同一规则上复算：训练阈值和评测阈值均为 **56.66917913880113**，差 **0**，rule hash 相同。相同单样本统计量差也为 0。这确认两条链路的定义一致；仅 2 个校准序列，**不证明诊断有效性或误报保证**。局部窗口统计在扁平 query 向量上明确关闭，未声称它是二维局部结构检测器。

LIBERO 额外执行真实 10-step denoising，归一化动作块为 **1×10×7**，全部 finite；训练与评测的官方语言 token/mask 完全相同，冻结后端梯度列表为空。该动作没有交给模拟器执行，也不能当作闭环收益。

收据：

- [RoboTwin 训练与随机恢复](independent_audit_20261016/robotwin_repair_receipt.json)
- [LIBERO 前向/反向](independent_audit_20261016/libero_repair_receipt.json)
- [RoboTwin 校准一致性与真实预览](independent_audit_20261016/robotwin_final_probe.json)
- [LIBERO 实际采样、官方语言与 RGB 输入一致性](independent_audit_20261016/libero_final_probe.json)
- [独立评测原始小型报告](independent_audit_20261016/repaired_eval.json)

独立评测生成 **22 张真实图**，保存在远端诊断 run 的 `preview/`，包括实际 attention、逐候选 RAW 预测/残差与真实采样动作。训练自带预览另生成 3 张 SVG，已修复原广播错误和新增接入中的缺参错误。初次诊断收据里的 `previews.failed` 原样保留，修正后结果见 `*_final_probe.json`，没有覆盖原始失败。随后实际提取图像像素时发现 SVG 的 PNG 编码还缺失，已修正源码，并将本次诊断图中的原始像素无损编码为 PNG；修复清单见 [svg_encoding_repair.json](independent_audit_20261016/svg_encoding_repair.json)。原始 DeepSeek run 的旧 SVG 留档，不能以它们的 HTTP 200 证明图像有效。还发现 attention 浮点幅值在 resize 时被提前转换为 uint8，热图被清零；现保留浮点值直到颜色映射，并从相同真实权重/样本重画 attention，见 [attention_render_repair.json](independent_audit_20261016/attention_render_repair.json)。可查看[修复后实际 attention](independent_audit_20261016/attention_robotwin_head.svg)或[实际叠加像素](independent_audit_20261016/attention_robotwin_head_embedded.png)。亮色仅是该层注意力的显示归一化，本轮尚未证明某条 RAW 线索导致特定动作变化。

## 4. 原负结果与剩余问题

1. 原始 `history.jsonl` 的 step 522 loss 为 **30,037,980.55**，step 523 为 **525,629.14**。复算与原报告更正值一致；本次没有将这些尖峰删除或解释成普通小波动。
2. 原 RAW 0.0924 / RGB 0.0785 是旧单帧动作路径的同预算开发记录。它们可以保留，但不能代表修补后的历史推断网络；原评测 0.17435 还存在指令、时间条件和校准定义偏差，不再用于当前模型验收。
3. 修补后从旧权重做的 2 步诊断仍有很大的 causal loss：末步约 **2609.57**。它与 masked/action 项量级严重不同，有限梯度不等于优化稳定。重新建立的逐通道因果目标与旧权重的均值/非因果训练目标不同；应先用现有短画像定位方差、目标尺度与梯度贡献，不能直接加长训练掩盖问题。
   对同一个真实训练锚（`place_dual_shoes/episode_000041`，frame 78）做零更新诊断：全新前端 causal NLL/标量为 **0.7371**，旧 1000 步完整前端为 **4524.65**；仅重置旧前端的 decoder 后降为 **0.7412**。旧 decoder 的中位预测约 0.331，而真实 RAW 中位值约 0.000323，并伴随较小方差。该受控替换把此样本的大损失定位到了旧 decoder 与新逐通道因果目标的不兼容，**不是修补版从头训练已爆炸的证据**；也未追溯证明原 step 522 尖峰的唯一成因。见[真实尺度诊断](independent_audit_20261016/causal_scale_diagnosis.json)。因此后续 P2 从头初始化正式四臂中的所需前端，不延续旧 decoder/optimizer；旧权重只保留为诊断对照。
4. dense energy 的有效样本数和候选差异尺度已改变。tau=1 未经重新画像，不能沿用原“先验主导”的判断，更不能以有/没有 q 变化充当机制能力证据。未为凑先验探针结果而调整 tau。
5. 撤回目前保守地移除包含指定 ID 的整个联合相机时刻，再从保留历史重放。已验证完整联合时刻撤回；单相机局部撤回的额外信息保留未实现。状态只保留配置的 8 个时刻，重建从这段保留窗口的空状态开始。
6. 正式修补版 1000 步、受控 RGB 修补版重训、raw_base/rgb_base、充分训练、RAW/机制增量及 P4 闭环全部待做。LIBERO/RoboTwin 仍分别为 pseudo-RAW / HDR-derived RAW3，不产生 Bayer 结论。

下一阶段仍是 **P2 修复验证**：采用从头初始化的修补网络做短画像，记录新目标的损失/方差/梯度，再完成 P2 预算及可比 RGB 记录；不得把本报告视为 P3 长训授权。

## 5. 复现与检查

实际使用的有界脚本在 [real_repair_probe.py](independent_audit_20261016/real_repair_probe.py) 和 [final_real_probe.py](independent_audit_20261016/final_real_probe.py)。解释器、数据与权重路径沿正式配置；远端命令形式：

```bash
cd /workspace/owac/worktrees/n02-measurement-completion
export PYTHONPATH="$PWD/src:/workspace/owac/rawvla/source/starVLA"
export N02_PY="$PWD/outputs/n02-full/runtime/py310-cu128/bin/python"
CUDA_VISIBLE_DEVICES=0 "$N02_PY" research/experiments/n02_v1_2/independent_audit_20261016/real_repair_probe.py robotwin
CUDA_VISIBLE_DEVICES=1 "$N02_PY" research/experiments/n02_v1_2/independent_audit_20261016/real_repair_probe.py libero
CUDA_VISIBLE_DEVICES=0 "$N02_PY" research/experiments/n02_v1_2/independent_audit_20261016/final_real_probe.py robotwin
CUDA_VISIBLE_DEVICES=1 "$N02_PY" research/experiments/n02_v1_2/independent_audit_20261016/final_real_probe.py libero
```

脚本默认写诊断目录，重放时请另设输出目录以保留本轮收据，不能覆盖原 1000 步 run。原交付对应模块归档为 `independent_audit_20261016/pre-repair-source.tar.gz`；其余原始资产仍留远端。

本地 `make check` 通过，相关目录 **602 passed / 6 skipped**（23.24 s）；skip 是缺 numba 的上游依赖和本地 CUDA 用例，未当作通过。远端真实双平台复验另有上文收据。`git diff --check` 通过，22 个选定源码/配置/状态文件本地与远端一致。监控页保持在线并在旧卡片 activity 标注“独立验收未通过”；其卡片和曲线仍属于原 1000 步路径，修复诊断图留在独立目录，不混作旧模型结果。检查结果见 [validation.json](independent_audit_20261016/validation.json)。关键回归检查真正覆盖历史影响、当前图像不能泄漏到因果预测、逐通道完整 query、干净重放一致性、真实 attention 预览、RGB 表示和语言失败处理；旧版 P0/P1 回归仍保留。测试通过只支持这些接线与数值性质，不替代 RAW 或具身能力验收。

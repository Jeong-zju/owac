# OWAC-D1.1 ALIGN 运行与交付

当前只完成工程、真实资产零更新前反向／接口与独立部署检查。真实策略优化0次，学生闭环0场。采用决议0016不重复新建；旧N01/N02/N04及其历史预算继续暂停。验收结论以[Sol独立报告](align-independent-review.md)为准，不能用本文件的命令或Harness回执代替。

## 实际文件与环境

配置：`configs/experiments/raw_stem_siglip_align_v1.json`，所有执行授权位false。资产绝对路径／哈希见该配置及[证据快照](align-evidence.json)。选定教师是本地作者架构init0前端＋当前实际π0.5，已有前阶段20/20开发筛查，不是作者发布的训练后权重。50源5349帧，train40/validation5/test5；pseudo-RAW3，非原生Bayer。

```bash
cd /home/jeong/zeno/owac/repo
PY=outputs/rawvla-teacher-20261020/runtime/bin/python
CFG=configs/experiments/raw_stem_siglip_align_v1.json
RUN=outputs/raw-distillation-align-20261020-153948
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

实现入口：`src/owac/baselines/raw_temporal_siglip.py`；`raw/temporal_distillation.py`／`raw/temporal_packets.py`；`training/raw_temporal_distillation.py`；`evaluation/raw_temporal_distillation.py`；`runtime/raw_temporal_distillation.py`。脚本只调这些模块。主要代码由真实DeepSeek V41-Flash Max完成；10轮选择和独立review见[派工账本](align-harness-ledger.json)。没有接管人工Web会话。

## 本轮实际运行的入口

资产核验不建模型：

```bash
$PY -m owac.runtime.raw_temporal_distillation assets --config "$CFG"
```

已生成train40＋validation5的969条缓存，位于`$RUN/sol-cli-final-cache`，5.25GB。实际使用45个明确`--source-id`，没有为test5生成教师目标；精确命令保存在`$RUN/sol-full-cache-receipt.json`，调度脚本`$RUN/prepare-approved-cache.py`。单源命令如下，已存在行按校验状态链重载：

```bash
$PY -m owac.runtime.raw_temporal_distillation prepare-cache \
  --config "$CFG" --device cuda:0 \
  --source-id libero_goal/put_the_cream_cheese_in_the_bowl/demo_23 \
  --cache-dir "$RUN/sol-cli-final-cache"
```

最长训练源端点150，25前缀端点＋6展开，真实特征/动作反向0更新：

```bash
$PY -m owac.runtime.raw_temporal_distillation preflight \
  --config "$CFG" --device cuda:0 --arm C \
  --source-id libero_goal/put_the_cream_cheese_in_the_bowl/demo_23 \
  --anchor-endpoint 150 --action-diagnostic --save-on-cpu \
  --cache-dir "$RUN/sol-cli-final-cache" \
  --output-dir "$RUN/sol-ready-C"
```

B实际运行相同源/端点，改为`--arm B`，去掉`--action-diagnostic`，输出`sol-ready-B`。B使用教师实际内部RGB、同容量stem与门控，是诊断输入；不能作为纯RAW部署。

首／后窗口和跨源reset的最终CLI阶段零：

```bash
$PY -m owac.runtime.raw_temporal_distillation parity \
  --config "$CFG" --device cuda:0 \
  --source-id libero_goal/put_the_cream_cheese_in_the_bowl/demo_0 \
  --source-id libero_goal/put_the_cream_cheese_in_the_bowl/demo_23 \
  --endpoint 0 --endpoint 30 \
  --cache-dir "$RUN/sol-cli-final-cache" \
  --output-dir "$RUN/sol-ready-parity"
```

独立检查脚本及实际收据在`$RUN`：`sol-real-align-probe.py → sol-real-06/result.json`；`sol-pure-v2-probe.py → sol-pure-05/result.json`。后者新进程禁止教师/ISP/缓存导入和文件访问，检查checkpoint0重载、时序续接、reset、验证5源102 anchors、4锚点视觉干预和20调用时延。输出为未训练工程基线。

纯评估CLI（不加载教师/ISP/缓存；可选信息干预保留相同目标状态/语言/噪声）：

```bash
$PY -m owac.runtime.raw_temporal_distillation evaluate \
  --config "$CFG" --device cuda:0 --arm C \
  --student-artifact "$RUN/sol-real-06/student-checkpoint0" \
  --split validation --flow-steps 10 --visual-controls \
  --output-dir "$RUN/pure-cli-validation"
```

本轮独立脚本实际调用了与此CLI相同的纯运行/评估API；没有把上面尚未整条执行的CLI命令写成额外通过证据。跨示范干预只取同split、不同源、完全相同端点，保留目标相机身份/位置/mask；原始供给、配对和排除分母均报告。零视觉values/mask同时置零；目标位置干预not_run。干预不参与选checkpoint。

## 待新授权的最小运行

当前`train --config "$CFG" --arm C --max-updates 50`实际被缺authority阻止，退出2，在昂贵模型构建前停止。没有签发任何真实训练authority，也没有打开optimizer位。

建议新范围：B/C各≤50累计更新、有效batch16，各≤800 anchors；seed101、所有视觉继承层和后端冻结；正常光task6、保持40/5/5。合计≤4 GPU小时、新输出≤20GiB；建议B墙钟≤5400秒、C≤7200秒，另1800秒作缓存/验证/终止余量。最终配置pilot上限已收窄到50。第一实际更新必须检查参数变化、冻结稳定与真实optimizer/checkpoint恢复；未获得授权前不执行。

授权文件需要这些字段，**当前例子authorized=false且不构成授权**：

```json
{"authorized":false,"max_updates":50,"max_anchors":800,
 "max_wall_seconds":5400,"max_output_bytes":10737418240,
 "scope":"raw-stem-temporal","receipt_sha256":"<new-user-authorization-receipt-sha256>"}
```

C的墙钟字段改7200；两个10GiB上限分别计各自run与共用缓存，合计保守受20GiB监督。墙钟在更新边界检查，严格合计上限还需独立监督，为单次更新/验证留余量。旧教师budget ledger不参与。授权后另复制配置只打开optimizer授权，research配置digest不因授权位或输出路径变化；其他配方变更必须重新冻结相关收据。

待授权的准确命令结构（文件当前不存在，不可直接启动）：

```bash
$PY -m owac.runtime.raw_temporal_distillation train \
  --config "$RUN/authorized-config.json" --authority "$RUN/authority-B.json" \
  --arm B --max-updates 50 --device cuda:0 \
  --cache-dir "$RUN/sol-cli-final-cache" \
  --parity-receipt "$RUN/sol-ready-parity/parity.json" \
  --resource-receipt "$RUN/sol-ready-B/preflight.json" \
  --output-dir "$RUN/pilot-B"
```

C同样运行，替换为`authority-C.json`、`--arm C`、资源`sol-ready-C/preflight.json`及`pilot-C`。必须串行，由Sol监督总预算。首轮不自动启动D、stage2或任何闭环。`--resume`指向实际最后checkpoint，恢复全排列sampler、RNG、optimizer/scheduler、累计更新与已消耗墙钟，不把预算重算一次。0/500/2000/5000及实际终止点记录，50处也保存/验证；最终和历史最佳通过sidecar实际SHA绑定，前5步physical MAE7最小、同分较早。

导出真实checkpoint（当前无训练checkpoint；不能将新随机学生冒充训练后artifact）：

```bash
$PY -m owac.runtime.raw_temporal_distillation export-student \
  --config "$RUN/authorized-config.json" --arm C \
  --checkpoint "$RUN/pilot-C/checkpoint_final_000050.pt" --step 50 \
  --device cuda:0 --output-dir "$RUN/pilot-C-export"
```

该路径严格核对完整配方、源/缓存/后端身份、scope、冻结教师前端与后端。若预算提前停止，使用实际返回路径/累计step，不能假设50已完成。真实学习optimizer更新/恢复及训练checkpoint导出仍not_run，现有测试只证明工程边界及checkpoint0纯artifact。

## 实际验收和成本

Sol最终：164项通过、2项旧toy AdamW测试排除、0跳过，拦截所有torch.optim.step且尝试0；`make check`通过。个别Harness兼容测试曾执行CPU toy AdamW，不作用真实模型，不能用来填真实更新/重载项；详细限制与原始失败见独立报告。源代码、配置、Harness事件、失败日志和检查输出均保留。

最终配置最长前缀B/C特征前反向5.21/6.33秒；C动作单独7.34秒，C峰值8.36GiB，启用CPU activation offload。纯学生20调用p50/p95=204.9/214.3ms、峰值7.21GiB、3.63B独立参数；不是教师匹配速度比较。单checkpoint预计约1GiB，optimizer矩为估计110.3MiB，实际optimizer一步额外峰值未测。资源/外推及预算见[align-resources.json](align-resources.json)。没有正常光控制恢复、弱光收益、原生Bayer或跨传感器结论。

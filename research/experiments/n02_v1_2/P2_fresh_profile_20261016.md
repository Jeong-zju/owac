# N02 v1.2 修补版 P2：从头画像与实现修复

> 后续更新：用户另行授权 Codex 接管训练，r5 双卡真实训练已启动。本文保留 r4 零更新画像的历史结论，当前运行见[训练记录](P2_training_live_20261016.md)。

2026-10-16，当前执行版本 **`n02-v12-p2-causal-r4`**。按用户最新要求，本轮完成代码修补、从随机初始化的完整前端接真实冻结 π0.5 的有界前后向画像，**真实数据优化更新 0 次**。训练由用户自行开启。这里确认的是实现接线与资源开销，不判定训练稳定、RAW 优势或闭环能力；原 DeepSeek P2 未通过的结论保持。

本地 `/home/jeong/zeno/owac/worktrees/n02-measurement-completion`；远端 `/workspace/owac/worktrees/n02-measurement-completion`，`93.91.156.102:51584`。继续原 worktree，保留所有旧权重、日志、失败和未提交修改；未提交、未推送，未启动 P3/P4。用户启动说明与未安装的 supervisor 配置见 [训练入口](P2_r4_user_training.md)。

## 1. 先修目标实现，再测资源

这次发现了两个会让“前后向通过”仍然偏离目标的问题：因果损失训练不到上一时刻生成状态的校正器；完整查询虽生成了所有目标，坐标轴接反仍使非正方形图像丢掉一部分有效评分。绿色测试没有覆盖后者，直到真实画像核对实际评分数才发现。

| 已证明的缺陷 | 修补后的实际行为 | 验证 |
| --- | --- | --- |
| r2 在 anchor 前截断整个历史，causal RAW loss 对 corrector 的梯度为 0 | 默认保留最后两个不同时间步的 recurrent graph；更早历史保留数值并截断梯度 | 修补前探针为 0；r4 完整真实网络的 causal loss 对 corrector 为非零，见第 3 节 |
| 查询是 `(列,行)` 的整数索引，decoder/encoder 是 `(行,列)` 的传感器像素中心 | 共用传感器坐标转换，包含 +0.5、crop offset 和 packed scale；可视化 raster 同步按行列处理 | 非正方形 32×64、有裁剪偏移的坐标/采样斜坡测试；RoboTwin 实际评分从 518,400 恢复为 691,200 标量/候选 |
| objective 提供 weights 后绕过 valid mask；无效 NaN 仍经 `NaN*0` 污染结果 | 权重与有效掩码相乘；先屏蔽再做算术；有效值/方差/权重非法时拒绝；至少 FP32 累积 | 修补前带权能量 5001.8379，正确应为 0.91894；masked NaN 非 finite；修补后能量与 backward 均 finite |
| causal loss 声称使用看到目标前的支持，实际对候选等权平均 | 使用上一时刻已锁定并 detach 的 support，按有效标量归一化；无有效样本不启用此项 | 梯度和候选聚合记录写入画像；当前像素不进入因果预测的测试保持 |
| 声明 saturation_level 后，主目标反而把饱和值当精确强度 | 主评分一律显式排除饱和，并报告实际可评分数；没有接入的 censored helper 不再被当成主链能力 | 已知裁剪级别的全饱和测试：能量 0、q 保留声明先验、slots 不更新、内循环不接受空证据 |
| `init_variance` 经 log 再 softplus，实际初始值不等于声明值 | 用 inverse-softplus 初始化；拒绝非正、非 finite 初值 | 固定输出头权重后，声明 2.0 得到实际方差 2.0 |
| q=0 的候选只加有限 log bias，仍可泄漏 attention | 零支持候选在输入 attention mask 中彻底排除；非法支持被拒绝 | 零支持候选 attention 精确为 0 |
| 裸 ObservationBundle 的行数碰巧能被 history 整除，就被误当窗口并丢行；在线 state 默认清空 | 明确区分 WindowBatch 与当前观测；裸 bundle 每行均为当前决策，在线 state 保留，显式窗口独立重放 | 4 行当前 bundle 保留 4 行；窗口 train/eval 数值一致、两步截断梯度测试 |
| resume 只看执行版本，RAW/RGB 同形参数可能被误接；新 run 可能覆盖旧证据 | 保存并验证训练契约和真实数据身份，装载权重前拒绝错臂/配置/数据；新输出目录独占；checkpoint 原子替换 | 同形 RGB、改数据身份、已有 history 均被拒绝；tiny 模块保存/恢复测试通过 |
| fixed dev 取相邻最前几帧，容易全部来自 episode 开头 | 固定 dev 与校准诊断 dev 均按同一确定性规则分散抽取并记录 anchor；全 padding dev 不能算 0 error | 共用 anchor_indices；dev 记录实际身份 |

`freeze_backend=false` 现在被明确拒绝，因为当前 P2 optimizer 只实现冻结后端协议。没有以“配置接受了”冒称后端参与训练。四个正式配置均保持冻结 π0.5、完整 Base、原辅助权重和预算，并更换 r4 run_id，避免混入旧目录。

修补前代码归档 [pre-r3-source.tar.gz](fresh_profile_20261016/pre-r3-source.tar.gz)，最初探针 [pre_r3_probe.json](fresh_profile_20261016/pre_r3_probe.json)。首轮 r3 的两个真实画像保留在 `fresh_profile_20261016/superseded_r3_robotwin_{raw,rgb}/`，因查询坐标错误被 r4 替代，不混入下表。旧 r2 诊断和旧 1000 步也不能作为 r4 的训练记录。

## 2. 从头、真实数据、0 次更新

四次运行均使用 seed 1103、正式 ConvNeXt V2-Base（depths 3/3/27/3、dims 128/256/512/1024）、K=4、16 slots×512、8 帧、4 轮、每相机 256×2048 读出；模型前端随机初始化，未加载旧 N02 检查点。真实官方 π0.5 权重冻结，3,616,757,520 参数，active missing=0。RAW/RGB 在同平台具有完全相同的初始训练参数 SHA256，并使用相同 3 个 train anchors；RGB 来自同一次 RAW 的固定 ISP，辅助项也针对该臂的输入表示，不额外提供 RAW 标签。

| 平台 / 臂 | 训练参数（含 masked 分支） | 峰值 allocated GiB | 前向+总反向 p50 / p95 秒 | 实际优化更新 |
| --- | ---: | ---: | ---: | ---: |
| RoboTwin RAW | 110,556,363 | 73.414 | 4.319 / 4.528 | 0 |
| RoboTwin RGB | 110,556,363 | 73.414 | 4.415 / 4.531 | 0 |
| LIBERO RAW | 109,769,931 | 41.809 | 3.978 / 4.295 | 0 |
| LIBERO RGB | 109,769,931 | 41.809 | 3.795 / 4.323 | 0 |

每臂仅 3 个分散窗口，以上分位数只是这三个样本的描述。前向包含数据 I/O；反向为 total loss backward；首窗口额外的逐损失梯度探针不计入时延、但计入峰值显存。这里没有 AdamW step，也没有执行 grad_accum=2 的完整更新周期，不能把这些数字称为正式训练吞吐或 1000 步耗时保证。Adam 两组 moment 的未分配理论存储约 0.82 GiB/臂，step 临时量和累计梯度的额外峰值未实测。

三组窗口损失都是在**同一未更新模型**上测得，不是学习曲线。RoboTwin RAW total 1.289–9.492、RGB 1.425–9.545；LIBERO RAW 1.042–1.370、RGB 1.078–1.742。全部 loss 和实际 total gradients finite；不据此判定优化过程不会出现原 step 522 的尖峰。

画像代码 [n02_v12_profile.py](../../../src/owac/training/n02_v12_profile.py) 在入口中封锁 `optimizer.step()`，断言 optimizer state 为空，并对全部训练参数比较前后 SHA256；四份均相同。没有生成训练 checkpoint 或训练 history。CPU 单元测试会执行 tiny/fixture 优化来检验训练器，不计为真实数据训练。

实际回执：

- [RoboTwin RAW](fresh_profile_20261016/robotwin_raw/fresh_profile.json)、[RoboTwin RGB](fresh_profile_20261016/robotwin_rgb/fresh_profile.json)
- [LIBERO RAW](fresh_profile_20261016/libero_raw/fresh_profile.json)、[LIBERO RGB](fresh_profile_20261016/libero_rgb/fresh_profile.json)

RoboTwin anchors 为 29,029 / 58,058 / 87,087；LIBERO 为 51,075 / 102,150 / 153,225，具体轨迹、frame index、固定配置与分区身份在每份回执内。远端目录为 `outputs/n02-v12/P2/fresh-profile-r4-{raw,rgb}-{robotwin,libero}-20261016/`；这四个是画像目录，不是下面用户训练入口的正式目录。

## 3. 哪个目标实际训练哪些模块

以下是各臂首个真实窗口的未加权 loss 梯度 L2。当前三个辅助系数为 1/1/0；没有凭单窗口梯度比例改变权重。

| 平台/臂、loss | encoder | decoder | corrector | predictor | readout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Robo RAW action | 855.588 | 0.318 | 0.809 | 0.906 | 371.731 |
| Robo RAW masked | 2.915 | — | — | — | — |
| Robo RAW causal | 1.135 | 0.587 | **0.0292** | 0.0543 | — |
| Robo RGB action | 969.965 | 0.546 | 0.892 | 1.032 | 416.344 |
| Robo RGB causal | 1.408 | 0.642 | 0.0374 | 0.0670 | — |
| LIBERO RAW action | 73.664 | 0.0295 | 0.265 | 0.279 | 59.834 |
| LIBERO RAW causal | 0.0547 | 0.262 | 0.0153 | 0.0227 | — |
| LIBERO RGB action | 60.061 | 0.0627 | 0.223 | 0.230 | 54.103 |
| LIBERO RGB causal | 0.0623 | 0.377 | 0.0171 | 0.0290 | — |

“—”是该目标没有接到该模块。masked loss 通过共享的全部 376 个 backbone 参数张量反传，并训练自己的重建头；action/causal 只使用 stride4/8/16 支路，其 encoder connected 张量为 348，stride32 由 masked 分支训练。proposal 与相机 embedding 的梯度也在原始回执中。measurement 是解析模块；2048→2048 无额外 bridge projection，因此这两组参数数目为 0。冻结后端所有参数均无梯度。

动作损失在这些初始窗口对 encoder/readout 的梯度较大。这提示后续用户训练应同时看各损失、各模块梯度和 dev，不证明辅助项无用，也不是自动改 λ 的充分证据。两步截断使因果损失能学习上一时刻状态生成，**没有宣称完整 8 步 BPTT**；历史数值仍使用全部 8 帧。

## 4. 实际覆盖、训练/评测一致性与校准边界

使用两个平台相同 seed 的随机完整前端进行额外有界复验；没有构建 optimizer，前后前端参数哈希一致。该项不加载 π0.5，不把它重复算成真实后端画像。复验脚本见 [verify_fresh_frontend.py](fresh_profile_20261016/verify_fresh_frontend.py)。

| 平台 | 训练/评测 token 最大差 | 训练/评测校准阈值 | 阈值差 | 中性/偏置先验 q 最大差 |
| --- | ---: | ---: | ---: | ---: |
| robotwin | 0 | 0.406795487 | 0.0 | 0.580920 |
| libero | 0 | 0.578405165 | 0.0 | 0.679451 |

两个平台的三组完整历史窗口中，训练 token adapter 与评测重放均逐位相同。校准每条链各使用独立 calibration 分区中的同一组 4 个 anchors，不是原先不同采样/预测定义得到的两套阈值。这个小样本结果只排除接线差异，**不证明校准阈值有效、不声明误报率保证，也不将 compatible 当能力通过**。训练更新以后，随机初始权重的阈值不会继续安装为在线有效阈值。

RoboTwin 每个窗口 3×3×240×320=691,200 个目标标量，全部有效并实际评分，4 个候选使用同一集合。LIBERO 每个窗口 2×3×256×256=393,216 个查询，三个窗口实际可评分数为：

- anchor 51,075：315,273 个实际评分，显式排除饱和 77,943 个。
- anchor 102,150：393,216 个实际评分，显式排除饱和 0 个。
- anchor 153,225：343,449 个实际评分，显式排除饱和 49,767 个。

RAW/RGB 保持相同有效/饱和掩码；实际 decoder 评分数与 data-side 可评分数完全相等，没有把未进入 decoder 支持的点算作已覆盖。LIBERO 不满全图的部分来自显式饱和排除；查询空间本身覆盖全部像素/通道。

真实单位/shape/finite、12 个窗口的时间因果性、抽查测量身份的 train/dev/calibration 零交叉和 row-wise reset 均通过。完整 source 构建仍执行既有 P0 分区/anchor 身份核验；这里没有重复扫描数据全集算哈希。

复验回执：[RoboTwin](fresh_profile_20261016/fresh-acceptance-r4-robotwin.json)、[LIBERO](fresh_profile_20261016/fresh-acceptance-r4-libero.json)。

## 5. 检查、复现与尚未验证

`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --locked pytest tests/encoders/n02_full -q`：**611 passed / 6 skipped / 0 failed**。跳过项为旧 P1 的本地 CUDA/numba 依赖，不将其视为通过。`make check` exit 0，ruff/format/compileall 通过。日志：[pytest](fresh_profile_20261016/local_pytest.txt)、[make check](fresh_profile_20261016/local_make_check.txt)。

完整 profiling 命令与手动训练/恢复方式见 [用户入口](P2_r4_user_training.md)。画像运行时唯一 Git HEAD 是 `861c46f67fff552c81fff89c5116c63b815224c5` 加未提交修改，不能只用此 HEAD 当代码身份。现行相关源码/配置文件的小型哈希清单在 [source_manifest.json](fresh_profile_20261016/source_manifest.json)，用于本地/远端核对，未扫描数据或权重全集。最后一次校准诊断 dev 抽样改为同源分散 anchors 发生于四份资源画像之后；它不在资源画像调用路径中，另由上述真实前端复验和最终测试验证。

本轮没有验证真实 AdamW 更新的显存/时延、r4 GPU 更新后的 save/resume 后缀一致性、有限 loss 尖峰是否消失、1000 步 dev 改善、四臂归因或 100 次闭环。全部仍未运行，不因画像完成自动进入 P3/P4。两步 BPTT、hard accept/reject、全局窗口重建及未经验证的在线长序列行为也是当前边界。原训练的 step 522/523 爆炸与所有负结果保留。

当前可交付结论是：这次已定位的目标/掩码/坐标/梯度/恢复漏洞有实际修复与回执，完整网络可在两个平台完成从头真实前后向；正式训练需由用户启动并产生新的证据。

本地与远端现行相关源码/配置/测试 **37 个文件逐个哈希一致**。最终只读部署回执见 [deployment_receipt.json](fresh_profile_20261016/deployment_receipt.json)：新 r4 supervisor 配置未安装，两个正式训练目录不存在，没有存活的 N02 画像/训练进程；旧 RAW/RGB 作业保持 EXITED，监控保持 RUNNING，两卡回到空闲。`supervisorctl status` 因旧作业 EXITED 返回 3，这是该状态命令的退出码，不是本轮画像失败。

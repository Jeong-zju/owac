# N02 P2：Codex 接管训练与简洁监控

后续核查补注：发现 r5 公共状态提示先tokenize14维再补32维，偏离原生后端格式；本记录保持原始测量身份。原始策略同初态能成功；修正提示后旧N02仍失败，不能单因归责。见[整个policy框架核查](P2_policy_framework_audit_20261016.md)。

完成状态：两臂各 1000 updates 均已 exit 0。RGB 后半程退化，训练执行结束不等于稳定性或 RAW 优势通过；检查点重载存在小幅非零数值差。见[2026-10-16 完成核查](P2_training_completion_20261016.md)。以下保留启动时记录与调优证据。

2026-10-16。用户最新授权由 Codex 接入并启动训练，不再交给 DeepSeek；这取代上一轮“训练由用户开启”的限制。已经从随机初始化启动 RoboTwin RAW/RGB 两臂，各 1000 次优化更新，seed 1103。仍为 P2，不自动启动 P3/P4；没有把旧 1000 步、诊断训练或吞吐试跑算入本轮进度。

监控页 **http://127.0.0.1:8081/** 已打开在 Codex 浏览器，4 秒自动刷新。它只固定显示当前两臂：

- **状态**：准备/训练/完成/异常，真实步数、剩余时间。
- **速度**：样本/分钟、秒/更新，物理 GPU 利用率与显存。
- **效果**：训练前的固定验证误差、最新误差与变化比例；每 50 步验证。

每 100 步保存检查点。异常停止不会自动重启；错误显示在页面与原始日志中。默认页面不混入旧路径、吞吐试跑、零更新画像或样例；历史与注意力详情保留在 `/details`。离线验证误差不是闭环任务成功率，页面已直接说明。

## 正式作业

| 臂 | 物理 GPU | Supervisor | 远端 run |
| --- | --- | --- | --- |
| RAW | 0 | `owac-n02-v12-r5-raw` | `outputs/n02-v12/P2/raw-n02-robotwin-r5-s1103/` |
| 同源 RGB 对照 | 1 | `owac-n02-v12-r5-rgb` | `outputs/n02-v12/P2/rgb-n02-robotwin-r5-s1103/` |

远端 checkout 为 `/workspace/owac/worktrees/n02-measurement-completion`，SSH 为 `root@93.91.156.102:51584`；进程内 CUDA 均为 `cuda:0`。执行版本 `n02-v12-p2-train-r5`，独立新目录、无 resume。完整 ConvNeXt V2-Base / SAVi / Perceiver 网络与冻结 π0.5 保留；输入 8 帧、BPTT 2、K=4、16×512 slots、4 轮、全相机逐通道测量评分保留。

两臂使用同样的 **microbatch 2、grad_accum 1**，每次更新仍是 2 个 anchors；更改后训练块的抽取顺序随 batch 配置重新确定，两臂保持一致，不声称与旧 microbatch1 的逐更新轨迹一致。损失权重仍为 masked=1 / causal=1 / noise=0，学习率、warmup 与梯度裁剪均保留。RoboTwin 采用原 RAW 单位、MEAN_STD、14 维动作、horizon50/offset1；RGB 是同一次 RAW 经固定 ISP，不使用正常照明配对 RGB。

## 为吞吐做的实际改动

1. 同尺寸相机合并执行共享 ConvNeXt backbone，输出仍按相机保留坐标与身份；没有跨样本归一化或混合状态。
2. 允许 TF32 的矩阵运算，保留 FP32 参数、RAW 目标、方差和完整测量能量；没有把弱 RAW 目标改为 BF16，也没有降低分辨率、删历史、删候选或删内循环。
3. decoder 仅采样本相机的查询，再写回原查询位置，避免对其他相机查询做重复计算；输出/梯度数值对照通过。
4. decoder 激活使用 non-reentrant checkpoint 重算，换取更低显存。损失与参数梯度对照通过；真实完整前后向也通过。
5. 单 CPU worker、队列深度 2，读取下一批数据与 GPU 计算重叠。worker 独占训练 source 的可变 episode cache，顺序消费原 seed/cursor 决定的 batch；测试核对实际图像、标签、提示词与同步加载相同。恢复时按消费游标重新预取，不把尚未消费的数据计为训练。
6. fused AdamW；分角色的详细梯度统计从每步改为每 50 步，每步仍做 global norm 的非 finite 检查与裁剪。更新时间包含 CUDA synchronize，避免把未完成的 GPU 工作遗漏在计时外。
7. 增加真实 step0 固定 dev 基线，并修复普通状态更新把最新 dev 误差覆盖为 null 的问题。监控端也从真实 history 恢复最近验证点，保存/完成的重复点按 step 去重。

训练和独立评测共用模型的批处理开关与精度配置，避免恢复评测时静默换回另一条计算路径。运行契约包含这些性能配置；r4 不可 resume 为 r5。

## 已运行的调优证据

三个 10-update 的真实训练试跑均从头开始，在独立目录保存；这些不是正式进度：

| 试跑 | 平均 / p50 秒每更新 | 每次更新 anchors | 结论 |
| --- | ---: | ---: | --- |
| 原执行方式 | 8.265 / 8.026 | 2 | 实际 optimizer 基准 |
| 相机合批 + TF32 + fused AdamW + 降低统计频率 | 8.233 / 7.691 | 2 | 单靠这些改动提速有限 |
| 再加 CPU 预取 | 7.931 / 6.992 | 2 | 中位数改善，仍有等待与首次 autotune 成本 |

未使用激活 checkpoint 时，batch2 在真实全窗口前后向中 **OOM**（已分配 93.84 GiB，额外 320 MiB 失败）；保留完整失败日志，不能当成通过。启用 checkpoint 后，同样的真实完整网络：

| 零更新资源检查 | 峰值 allocated GiB | 第二个窗口前向+总反向秒 | anchors/窗口 |
| --- | ---: | ---: | ---: |
| batch1、checkpoint | 23.727 | 4.494 | 1 |
| batch2、checkpoint | 34.436 | 6.808 | 2 |

每组只检查 2 个窗口；首个窗口包含 cuDNN 搜索，不能当稳定吞吐。按第二个窗口折算，batch2 比两次 batch1 的前后向吞吐约高 32%；这不是 optimizer 吞吐的最终数值。正式训练继续用实测更新计时和 GPU 遥测核对。batch2 保持原有效 batch2 的预算，不把单纯显存占用当训练完成度。

证据目录 [training_start_20261016](training_start_20261016/) 包含三份 pilot 配置与实际 profile/report/history、batch2 OOM 日志、两个 checkpoint 资源回执、最终源码快照和启动回执。

## 检查与运行方式

本地检查 **617 passed / 6 skipped / 0 failed**，`make check` exit0。新增检查覆盖相机合批的坐标/特征/梯度、query 分组的值与梯度、checkpoint 的因果损失和参数梯度、预取的顺序/标签一致性，以及监控不引用历史试跑。跳过项仍为旧 P1 的 CUDA/numba 依赖，不计通过。

已安装 `configs/runtime/n02_v12_p2_r5.supervisor.conf` 到 `/etc/supervisor/conf.d/owac-n02-v12-r5.conf`，并已执行 start。`autostart=false`、`autorestart=false`，重开实例或异常后不会悄悄另起训练；现有平台服务和旧训练记录保持。主要日志：`outputs/n02-v12/logs/r5-{raw,rgb}-robotwin{,.err}.log`。

查看实际作业：

```bash
ssh -p 51584 root@93.91.156.102
supervisorctl status owac-n02-v12-r5-raw owac-n02-v12-r5-rgb
```

监控依旧监听远端 `127.0.0.1:18080`。本机已经有转发；需要重新建立时：

```bash
ssh -N -p 51584 root@93.91.156.102 -L 8081:localhost:18080
```

本轮不宣称原有限尖峰已解决，不因短期 dev 改善判定 RAW 或 N02 有能力收益。正式作业会按现有 1000 次更新预算结束；后续充分训练、四臂归因与闭环仍需新的阶段工作。未提交、未推送。

## 启动后实际核对

以下为 2026-10-16T03:55:20Z 的即时记录，训练还在继续：

| 臂 | 进度 | 秒/更新 | 样本/分钟 | 初始验证误差 | step50验证误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW 模型 | 80 / 1000 | 5.291 | 22.7 | 1.448695 | 0.520874 |
| RGB 对照 | 78 / 1000 | 5.101 | 23.5 | 1.549180 | 0.161177 |

两臂均已产生真实 optimizer 更新、有效固定 dev 点与有限损失。当前早期验证误差 RGB 更低；不据此判断最终 RAW 优势。当前吞吐相对原 10-update 试跑约为 1.5–1.6 倍，属于短窗口实测，长期速度仍由监控页实时刷新。GPU 遥测的均值/95分位/峰值单独保留，不把瞬时高利用率说成全程满载；因果执行与数据准备仍有串行开销。

[实时状态快照](training_start_20261016/live_status.json) · [GPU遥测](training_start_20261016/gpu_telemetry.csv) · [统计](training_start_20261016/gpu_utilization_summary.json) · [监控截图](training_start_20261016/monitor_live.png) · [完整相关源码快照](training_start_20261016/r5_source.tar.gz)。

本地与远端源码/配置/测试共80个文件哈希核对一致，运行期间未再改训练源码。

随后确认两臂第100步检查点均已完成原子保存（RAW 1,327,367,889 bytes；RGB 1,327,368,209 bytes），进程继续运行。第100步固定验证误差：RAW **0.391822**，RGB **0.093697**。回执：[checkpoint100_receipt.json](training_start_20261016/checkpoint100_receipt.json)。这仍是早期离线验证，不是任务成功率。

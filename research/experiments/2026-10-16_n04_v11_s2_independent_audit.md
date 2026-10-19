# N04 v1.1 S2 独立核查与修补（2026-10-16）

后续状态：用户已于北京时间 10:36 终止所有 N04 训练；本报告为停止前审计记录。后续代码修复与验证见[当前修复报告](2026-10-16_n04_v11_s2_code_repair.md)。以下“训练继续”均为历史。

目的：核对原始 S2 要求、工作树源码、远端 checkpoint、逐 episode 记录和实际视频，处理可复现的缺项；不把训练 loss 下降当作方法有效。

**结论：100 次开发闭环的最终 0/100 属实；完整架构的工程交接与训练充分性、机制收益必须分开。原交付的“全部实现／训练健康”措辞过强。当前保留 S2，未进入 S3。** 本轮补齐 RGB 输入、修正方差阈值接线和日志／验证问题，完成真实输入前后向检查；未改写正在运行的冻结源码，未重启已有训练，未追加闭环或种子。

## 1 实际核实的闭环

证据：[独立汇总](results/2026-10-16_n04_v11_s2_independent_audit/summary.json)。原始回执、JSONL、进度、checkpoint 元数据、核查脚本和抽帧保存在本地忽略目录 `outputs/n04-v11-s2-independent-20261016/`；远端小型审计目录为 `/workspace/owac/research/n04-v1/runs/v1.1/audit-20261016/`。

| 平台 | 预定／完成 | 成功／失败 | 最后结算基础设施错误 | 非空视频文件 | 步数 |
| --- | --- | --- | --- | --- | --- |
| LIBERO | 50／50 | 0／50 | 0 | 50 | 16,900 |
| RoboTwin | 50／50 | 0／50 | 0 | 50 | 30,000 |
| 合计 | 100／100 | 0／100 | 0 | 100 | 46,900 |

- 独立按 identity 取最后一行，与仓库 `eval100.json` 的 100 个身份完全相同；每平台每域 10 条。**历史错误不为零**：RoboTwin 原始 63 行中有 7 条基础设施错误行；最后结算已补齐为 0。LIBERO 保留 152 行历史，不能将行数直接当 episode 数。
- 两平台使用**各自**的 post-fix `checkpoint-step0000100.pt`，不是共享一个权重。CPU mmap 读取确认：step=100、seed101、D、`v1_1_forecast_perdim_v2`，optimizer 内 step 全为 100，scheduler epoch=100。
- LIBERO 步数为 520×15／300×10／280×10／220×15，timeout=true 共 50。RoboTwin 为 400×20／600×5／700×15／800×5／900×5，作者未提供 timeout 字段；维持行内 false，另报预算耗尽，不改造原字段。
- 视频文件 100 个均非空；两平台各抽查 ExtremeLow 文件可解码、帧数分别为 520／400，并另抽 Normal 帧查看。未声称逐帧验收全部视频。ExtremeLow 原视频显示极暗，文件存在不等于人能看清任务；后续观看应提供明确标注的显示增强，保留原视频和模型输入。
- 这些是开发评估；0/100 不能单独证明 N04 路线失败，也不能将原因唯一归为训练步数少。没有同预算对照、视觉输入干预和固定 dev 曲线时，不能排除编码／优化／后端匹配问题。

## 2 发现与本轮修改

| 可复现问题 | 处理与验证 |
| --- | --- |
| S2 要求 A/B/C/D 开关及同源 RGB 入口，但训练器直接拒绝 A/B，ISP 只存在配置名 | 补齐 `fixed_clip_srgb_v1`：同源 RAW、固定曝光、[0,1] 裁剪、sRGB 传递函数。公共编码输入边界同时覆盖训练、在线服务和 forecast target；输出仍保持平台存储单位，只施加一次 InputScale。A/B 的质量与对应只读 ISP 后图像及其饱和标记；D/C 不经过 ISP。不是作者 ISP 复现或独立 RGB 数据集 |
| RoboTwin 配置白电平 3.5 未传入对应损失 RawScale，实际默认 1.0 | 统一透传；RGB 两臂采用 ISP 后的 1.0 白电平。旧 checkpoint/结果的输入行为保留，不追溯改分 |
| Perceiver 各层显式物化 attention，训练还收集最后层热图 | 无热图时用 PyTorch SDPA；训练关闭热图采集，独立可视化仍显式请求真实 attention。参数键和网络尺寸不变，数值与梯度对照通过 |
| 固定 dev 窗口仍随机采 flow-matching 的时间／噪声，并消耗后续训练 RNG；backend 未切 eval；只评动作仍计算辅助项 | 固定验证 RNG、用 fork_rng 恢复训练随机状态，两模块 eval 后在 finally 恢复；跳过与动作 loss 无关的 VICReg／forecast，状态和动作路径保留 |
| 每帧 VICReg 显式生成 2048×2048 协方差，prefix 行数通常只有 512／768 | 使用较小的 Gram 矩阵等价计算同一协方差惩罚，输出及梯度数值对照通过；不改变其防塌缩目标，也不保证视觉表征有效 |
| 旧 `terms` 中 correspondence／representation 含 burn-in 统计，实际 backward 不含这些帧；forecast 报告均值与计划分母也不同 | 保留历史字段与原优化目标，新增 `applied_terms`／`applied_train_loss`，逐项记录实际反传尺度；不能将旧 total loss 当作准确的优化目标轨迹 |
| clip 默认可将非有限梯度写入 optimizer；闭环全完成仍永久显示 running | clip 打开 `error_if_nonfinite`；训练／闭环两个摘要入口正确给出 done；未靠置零掩盖异常 |
| handoff 顶层、not_done、next_steps 仍写 0/100、等待首个权重和旧 forecast≈300 | 更新当前权威字段；已有历史阶段快照和 pre-fix 数字继续保留 |

修改位置：`raw/n04_v11_data.py`、`frameworks/n04/{encoder_v11,readout}.py`、`training/n04_v11.py`、`evaluation/n04_v11_closed_loop.py`、A/B 配置及定向回归测试。

## 3 验证及其边界

- 定向测试：**191 passed、1 skipped**；跳过的是本地缺 `websockets` 的协议集成测试。`make check` 通过。测试只证明所覆盖的软件行为。
- 从远端官方 **dev** 各取同源连续两帧：LIBERO 2 相机 256×256、RoboTwin 3 相机 240×320。用完整 Base／RAFT／6 层读出在本地 RTX5090 分别运行 D/C/A/B。八项前后向全部通过：prefix 为 LIBERO `[1,512,2048]`、RoboTwin `[1,768,2048]`；D/B 的测量、RAFT、质量、状态、读出均有有限非零梯度。使用随机初始化编码器，**不是 π0.5 动作训练或能力验证**。[小型回执](results/2026-10-16_n04_v11_s2_independent_audit/real-input-check.json)
- SDPA 与显式 attention 的 FP32 输出及输入梯度一致性通过。单层真实尺寸微测（256 queries、4928 media、1024D、BF16、RTX5090、前反向）中位 1.503→1.099 ms，峰值 allocated 0.197→0.117 GB；仅说明这个层的节省，**不能外推整轮训练加速比**。[微测回执](results/2026-10-16_n04_v11_s2_independent_audit/readout-benchmark.json)
- 独立读取 LIBERO step100→200 的抽样参数值：测量、RAFT、质量、状态、读出各组均存在变化；这只能证明参数/统计发生更新，不能证明学到有用对应或控制。
- 本轮另用 step100 的真实 π0.5 权重在 CPU 上核查固定 dev 动作 loss，避免挤占两张训练 GPU。结果和确切窗口另存 `dev-action-audit.json`，见下方追加记录。

## 4 尚未解决的实现与研究问题

1. **对应训练有实质缩水。** `RaftCorrespondenceLoss` 提供 Wang 遮挡和双向接口，但 encoder 从未传 `backward_flow_px`；所谓 occlusion 实际来自前向终点是否越界，不能检测画幅内真实遮挡。模块自己也明确列出 teacher/student stop-gradient 自监督分支未实现。这与 S2 的至少实现项有差距，应补齐并重新固定训练版本，不能继续称“完整 SMURF/UFlow 训练”。本轮不在已运行权重上静默改变对应目标。
2. **forecast 切断了源端表征梯度。** 不仅未来 target stop-gradient，anchor 的 posterior、R 也 detach，forecast 因而只训练预测侧。target stop-gradient 测试通过不等于 forecast 在训练视觉测量表示。动作／prefix VICReg 仍有表征梯度；需要如实描述其实际分工。
3. **实际辅助项权重低于均值报告暗示。** 监督帧不结算对应／forecast；跨 TBPTT 段的 horizon 显式跳过，但 forecast 分母仍包含它们。这不构成未来泄漏，却削弱了有效预测目标；新 applied 字段使它可见。若改分母或监督帧接线，需要新损失定义与版本，不能接着旧曲线宣称同目标改善。
4. **平滑排程未接 set_step。** 当前仍靠进程内 `_pairs_seen`，含 burn-in、相机次数、dev 调用，并随 resume 清零；不能等同于 optimizer-step warmup。不要把“延后了”当作确定的训练日程；应在下次正式版本统一接线、记录变更。
5. **表示与能力未建立。** 模式、R 和 attention 尚无明确区分度，已有固定对应诊断接近 fresh；动作随 episode 变化还可能来自语言／本体／噪声，不能据此排除视觉通道无效。“训练数值有限”与“训练有效”是两个结论。
6. **prefix VICReg 不能排除跨观测塌缩。** 当前把可学习 query 的 token 轴也当样本。独立构造四个观测具有完全相同的 64×32 prefix（query 为正负正交基、跨观测标准差严格为 0），得到 variance penalty=0、covariance penalty=0。这证明即使正则项为零也可能没有观测区分度，并非证明当前真实模型已经塌缩。若更改为跨观测／时间的表征约束，需要新目标版本。[可复现反例](results/2026-10-16_n04_v11_s2_independent_audit/prefix-regularizer-counterexample.json)

## 5 运行处置与继续入口

审计最初快照（北京时间 10:05 左右）LIBERO step264、RoboTwin step136，均在跑，固定 dev loss 尚为 null。近 20 步中位约 129.1／152.0 秒；按当时剩余步数线性外推 100k 更新约 **149／176 天**，并行意味着等待较慢平台，不相加成总等待。估计未包含故障、额外验证或后续提速，不是推荐继续照跑数月的安排。

- 两个现有 tmux 训练使用 `/workspace/owac/research/n04-v1/source/n04-v11-20261015T165752Z-a81cf25`，未被本轮重启或覆盖。
- 修补版独立放在 `/workspace/owac/research/n04-v1/source/n04-v11-s2-audit-20261016`，并用于本轮 CPU 核查；**已有训练进程尚未采用修补版**。保留这个区别，不能称线上训练已修好。
- 原训练／resume 命令仍见运行目录 `launch.json`／`resume.sh`；另外已生成各 run 的 `resume-audited.sh <checkpoint>`，指向独立修补快照，现有 tmux 未结束时不会创建同名重复会话。未执行这些脚本。不要为套用工程修补丢弃尚未存盘的进度；下一次切换必须在已保存权重处恢复，并写明源版本、白电平修正和数值内核变化。
- 监控仍为远端 `127.0.0.1:8090`，通过 `ssh -p 16724 root@154.59.156.22 -L 8090:localhost:8090` 访问。看板显示的 step 百分比不是能力进度。

下一项实质工作是把 S2 的对应训练缺项、有效目标权重与可承担的训练速度处理清楚，再用固定 dev 动作／视觉干预判断是否值得继续当前路线。已有 100 次失败保留；当前没有依据追加早期 checkpoint 的同类百次评估来代替这些诊断。

## 6 固定 dev 动作 loss 补充（本轮实际完成）

在修补版代码、step100 的真实 π0.5／编码器权重上，以 CPU BF16、固定随机时间／噪声、每平台配置原定 4 个 dev 窗口／16 个动作监督点完成：

| 平台 | step100 dev action loss | 窗口／点 | 纯验证耗时 |
| --- | --- | --- | --- |
| LIBERO | 0.365399 | 4／16 | 186.8 s |
| RoboTwin | 0.362945 | 4／16 | 208.4 s |

[完整窗口与数值回执](results/2026-10-16_n04_v11_s2_independent_audit/dev-action-audit.json)。未做 optimizer update，未占 GPU，CPU 审计已退出。这些是 flow-matching 监督损失的单点锚，不是成功率，两个平台的数值也不能直接比较能力；尚无相同固定噪声下 fresh／后续 checkpoint 对照，不能据此称“dev 已改善”。原训练进程的 `dev_loss` 仍是 null，审计结果单独保存，未伪造写入其训练曲线。验证使用独立修补代码（含正确白电平与 SDPA），后续比较须沿用这个实现版本。

最终只读核查（UTC 2026-10-16T02:32:41Z／北京时间 +8）：LIBERO step276、RoboTwin step146，两训练 tmux 会话均保留；CPU 审计进程已结束，两份 `resume-audited.sh` 的 shell 语法检查通过。[最后运行快照](results/2026-10-16_n04_v11_s2_independent_audit/final-running-state.json)。

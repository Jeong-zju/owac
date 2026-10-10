# 0007 直接可用的预训练策略与 benchmark 优先

**2026-10-10 后续纠正：用户明确现成策略与仿真器原生 CFA/RAW 必须同时满足。[0008](0008-pretrained-and-native-cfa-required.md)取代本文允许 RAW 后置的准入安排；LIBERO 仅保留为 RGB 参考，不再作为主 benchmark。下文保存本次 RGB 实验的历史依据与真实结果。**

日期：2026-10-10。状态：用户已决定优先采用现成基线，可以先用原版 RGB；新组合已通过全部 10 项任务的零额外训练闭环验收。

## 用户要求与选型纠正

用户否定了当前 Lightwheel 红杯任务与自定义 π0.5 微调基线的选取，不希望继续为 backbone 的任务成功率投入开发和训练。用户明确选择“优先现成基线，可先用原版 RGB”。这取代 0004 对 Lightwheel-LIBERO 主 benchmark 的推荐；RAW/CFA 能力不再是当前基线选型的前置条件。

此前方案将可运行的任务/成像采集与已经可用的学习策略混在同一建设路线中。当前任务、机器人安装、观测和绝对 TCP 动作接口缺少直接配套的官方 π0.5 检查点；自定义微调的 0/100 结果以及开发诊断已留档。此次不再继续该 backbone 的训练、动作接口调参或阶段信息补救，旧记录保留为历史负结果。

## 新组合

- Benchmark：原版 LIBERO / robosuite / MuJoCo，首先采用 LIBERO-Goal 的全部 10 项任务。
- Backbone：Physical Intelligence 官方 `gs://openpi-assets/checkpoints/pi05_libero`，复用已经在云端转换并校验的 PyTorch 权重和该检查点原有归一化统计，额外训练 0 步。
- 来源：openpi `15a9616a00943ada6c20a0f158e3adb39df2ccac`，LIBERO 使用其 gitlink `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。
- 执行：直接调用官方 `examples/libero/main.py` 的评测循环；沿用原相机、图像翻转/缩放、本体状态、动作、稳定等待、初始状态文件和 `done` 成功判据。
- 验收：先对全部 10 项任务各执行 1 条 episode，初始状态为官方索引 0，种子 7；各项均保留成功、失败与基础设施异常。不通过重新训练来获得准入分数。
- 数字归档：本地只保存配置、逐 episode 结果、统计、运行资源、版本和哈希；检查点不下载。云端新目录与旧实验隔离，所有持续进程由 supervisor 管理，不重建定时任务。

官方[评测说明](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)报告 Spatial / Object / Goal / 10 为 98.8% / 98.2% / 98.0% / 92.4%，四套平均 96.85%。本次云端已实测全部 10 项 Goal 任务各 1 条，成功 **10/10**，基础设施异常 0，额外训练 0 步。少量验收只确认直接可用，不冒充官方完整评测；逐条结果与 SHA256 见[验收记录](../../research/experiments/2026-10-10_pi05_official_libero_reference.md)。

## 后续边界

新的固定协议为 [pi05-libero-rgb-v1.json](../../configs/baselines/pi05-libero-rgb-v1.json)。旧红杯协议的 90/10 专家数据、全参数微调和开发损失早停不再作为新主 benchmark 的统一前提；也不与新组合直接合并统计。

原版 RGB 入口不提供本项目已验收的原生 CFA/RAW。后续 RAW 或特征替换仍需独立确定测量来源及候选适配协议，不能将逆处理 RGB 或重新马赛克称为真实 RAW。现阶段优先建立无需 backbone 训练的 RGB 强参考；是否接入 RAW 是之后的明确任务。

Isaac / Arena 的[官方 DROID joint-position π0.5 抓放示例](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/running_a_real_policy/openpi.html)保留为需要原生 CFA 时的备选；已核验其专用检查点对象可匿名列出，但未在本轮部署或实测该闭环，不再以有文档示例代替验收。

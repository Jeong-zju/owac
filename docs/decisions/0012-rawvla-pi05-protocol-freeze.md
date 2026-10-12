# RawVLA-Bench 与 π0.5 部署及比较协议固定

日期：2026-10-12。状态：用户要求提交部署成果并创建带说明的标签。

标签 `owac-rawvla-pi05-v0.1.0` 固定已跑通的 RawVLA-Bench / π0.5 运行入口、现成权重选择、完整测试清单和比较协议。后续 OWAC 算法使用这套基线作为参照；候选的训练、预算、输入信息、逐任务和逐照明域指标应与该协议对应。变更环境、成像、检查点或协议时另建版本并重新建立相应基线，不修改正在运行的冻结源码。

## 固定的基线

| 项目 | LIBERO | RoboTwin |
| --- | --- | --- |
| 作者环境 | RawVLA 中固定的 LIBERO / StarVLA | RawVLA 中固定的 RoboTwin / StarVLA |
| π0.5 检查点 | OpenPI 官方 `pi05_libero`，使用固定官方转换函数转为 PyTorch | 作者指定 `SidneyXie/pi05_robotwin` 原 PyTorch 权重 |
| 权重 SHA256 | `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8` | `9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3` |
| 任务规模 | 四套 40 任务 × 50 初态 × 五照明域，共 10,000 条 | 13 任务 × 50 原 dataset seed × 五照明域，共 3,250 条 |
| Default 图像输入 | 作者 `default_isp` | SAPIEN `render_rgb` / Color |
| 动作协议 | horizon 10，replan 5，原 quantile 统计 | horizon 50，原十四维绝对关节目标，MEAN_STD |
| 额外 backbone 训练 | 0 步 | 0 步 |

作者 RawVLA 提交固定为 `ba9c56b04da2c4931b70d8844710f5928d43775b`；数据仓库为 `ToferFish/RawVLA-Bench`，revision `8586c594fff9ce38580e29745f074e82e96e74f7`。RoboTwin 权重 revision 为 `e49e2ab6c11f07511573b67261bd129e88d0a416`。原统计、tokenizer、submodule、依赖和实际服务代码哈希见[部署记录](../../research/experiments/2026-10-10_rawvla_pi05_deployment.md)及[部署审计](../../research/experiments/results/2026-10-10_rawvla_pi05_full_baseline/deployment-audit.json)。

冻结[协议文件](../../research/experiments/results/2026-10-10_rawvla_pi05_full_baseline/protocol.json) SHA256 为 `3ac9626ef71eab4719be540393f51b66b767d7348dc4cac0ce5f5455f5b52e0e`。[LIBERO 原清单](../../research/experiments/results/2026-10-10_rawvla_pi05_full_baseline/libero-manifest.json)与[RoboTwin 原清单](../../research/experiments/results/2026-10-10_rawvla_pi05_full_baseline/robotwin-manifest.json)以原始字节入库，分别校验为 `e3a054f43060912d668e57d35873b573a289c6d3c941fec9c029569c82971168`、`632b8360836595a868affffd24f9ecf1262d68a52a505ef5ffe801598d8317e5`。

每初态的语言和 action-noise 随机流独立重置，五照明域配对；成功判据和步数上限沿用作者环境。正常失败保留计分，基础设施失败保留原身份与分母，禁止换种子、删除失败或挑选重跑结果。另有 632 条原缓存照明条件核查，独立报告，不能混入主清单 13,250 的分母。逐任务、逐照明域、SOG/Long/Avg 与成本的定义沿用[全量基线记录](../../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md)。

## 本标签已经确认的范围

两平台均已完成真实 π0.5 推理和闭环成功样例，数据全集、检查点和资产已经部署。LIBERO 全量 10,000 条完成，成功 5,820，成功率 **58.20%**，唯一身份和分母审计通过。RoboTwin 全量仍运行，并有基础设施异常缺口；632 条附加核查为 627 条有效完成、5 条异常未补齐。带时间戳的数字见[审计快照](../../research/experiments/results/2026-10-10_rawvla_pi05_full_baseline/audit-status.json)和[发布快照](../../research/experiments/results/2026-10-10_rawvla_pi05_full_baseline/release.json)。

本标签表示部署跑通及基线/协议固定，不表示两平台全量成绩已齐、不表示已复现论文全部数值、不表示 RAW encoder 或实体机器人已经验收。RawVLA-Bench 的现成 RAW 为三通道 HDR-derived / pseudo-RAW，不能描述为原始 Bayer CFA。Blackwell 上 RoboTwin 显式使用 OptiX；StarVLA 与原 LeRobot 的完整数值等价尚未建立。这些边界在后续比较中保持一致并披露。

这是独立的 RawVLA-Bench 部署和协议标签；旧规则中包含 RAW/特征接口验收条件的 `owac-dev-v0.1.0` 共同开发起点仍未创建。后续完整成绩单独追加留档，不移动本标签。

## 验证与复用

本次提交前在独立工作树核验拟提交内容：`uv sync --locked`、`make check`、RawVLA 数值/状态/动作/清单/审计/进程边界测试，以及 `run-smoke.sh` 的 shell 语法检查。实际结果与源码哈希记录在发布快照中。GPU 部署及闭环证据来自既有云端运行，提交检查不额外启动重复全量评测。

运行配置见[固定 supervisor 配置](../../configs/backends/rawvla-runtime/full-eval.supervisor.conf)；只适用于记录中的已部署服务器环境。大数据、检查点、模型输入数组和动作日志保留云端；Git 保存实现、小型清单、协议、校验和及数字结果。本次操作仅创建本地提交及 annotated tag。

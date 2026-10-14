# Arena DROID / 官方 π0.5 完整 RGB 基准 v1

目的：在用户指定的 Isaac Sim / Arena DROID 与原生 RAW 能力基础上，取得覆盖官方九组抓放变化的完整现成 π0.5 基准。随后专家演示回放生成配对 RGB/RAW；两阶段分别验收。历史 RGB 3/3 与同步 RAW 2/3 不作为本次完整基准。

状态：2026-10-11 用户终止，后续统一使用 RAWVLA-bench；云端 supervisor 已停止。2026-10-10 05:49:24.429372 UTC 启动的 900 条计划未全部完成，只保留历史部分结果，不报告最终基准成绩。专家数据和脚本的核查结论见[专题记录](../reviews/2026-10-10_arena_droid_expert_replay.md)。

## 冻结比较协议

协议为 [pi05-arena-droid-benchmark-v1.json](../../configs/baselines/pi05-arena-droid-benchmark-v1.json)，覆盖上游 SRL OpenPI YAML 全部九组对象 / 容器 / HDR 变化：billiard hall、Rubik home office、soup can、orange、lemon、tomato sauce can、mustard bottle、sugar box、mug。每组环境种子 0–99，每个配置/种子独立运行一个完整 episode，共 **900 条**。种子外循环、上游变化顺序内循环；每条重新启动策略服务器和仿真进程。

上游示例每组三条；本次每组 100 条是 OWAC 冻结的比较协议，不冒充官方标准评测数量，也不代表整个 Arena 平台的全部任务。官方配置文件 SHA256 为 `bcf83708d360065d05cf5f4b071e851a60fb2ee8473e0b60ee31b9587dc2463a`，部署开始前已核验。

复用官方外部 / 腕部 RGB、本体状态、jointpos adapter、初始随机化和 `PickAndPlaceTask` 成功/超时判据，时间上限 70 s 仿真时间，即最多 1050 控制步。原样使用官方专用权重，额外训练 **0 步**；BF16，chunk 15，推理种子 7，Torch compile 关闭。RAW 额外渲染不介入这一阶段；原生 CameraCore 能力在相同任务历史运行中已验收。

保留所有正常策略失败，不替换种子。基础设施异常不计为策略失败；同一配置和种子最多自动重试一次，使用新 attempt 目录并保存原因。再次异常则停止流程，保留所有证据，后续修复须版本化。不重启已完成结果，不启动并发模拟器，不使用裸后台进程。

## 实际部署身份

| 项目 | 固定版本 / 路径 |
| --- | --- |
| 本地分支 / 启动时 HEAD | `lzh/pi05-sim-baseline` / `61631c9`；本任务新增运行代码当时未提交 |
| Arena | `2cc9e9766f1f62a500b85c92207192d8c60fcce5` |
| Isaac Lab | `91a0906f93802eb859192dcb0bca879f891e92d6` |
| OpenPI | `c23745b5ad24e98f66967ea795a07b2588ed6c79` |
| Isaac Sim | 6.1.0，独立 Python 3.12；策略为独立 Python 3.11 / Torch CUDA 12.8 |
| 策略配置 | `pi05_droid_jointpos_polaris` |
| 权重来源 | `gs://openpi-assets-simeval/pi05_droid_jointpos` |
| 转换 BF16 模型 SHA256 | `123f0e0e2017c429f72dbf79bc8d3cc7a6dcb4de3054e21f51faaa45e4c155bb` |
| 自身归一化 SHA256 | `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7` |
| GPU | RTX PRO 6000 Blackwell Max-Q，96 GB |
| 复用 runtime | `/workspace/owac/reference/pi05-arena-droid-v1` |
| 新冻结部署 | `/workspace/owac/reference/pi05-arena-droid-benchmark-deployment-v1` |
| 新运行目录 | `/workspace/owac/reference/pi05-arena-droid-benchmark-v1` |
| supervisor | `owac-arena-benchmark-v1`，autorestart=false，端口 18030 仅 localhost |

运行前确认旧 Arena 服务已退出、没有并发模拟器；新目录不覆盖旧验收。冻结运行源码逐文件 SHA256 保存在 `source_hashes.json`，权重/归一化/YAML 核验保存于 `deployment_validation.json`。只增加数值推理时延记录，不修改策略输入、动作、物理过程与成功判据。启动前 supervisor 状态命令因其他服务停止返回 3，修正状态读取后才创建服务，当时没有 episode；不计为策略结果。

## 结果与实际成本留档

云端每个 episode 完成后，核验原官方 outcome、seed、控制步序列、有限动作、本体维度、RGB 帧刷新和推理计数，再计入 `episodes.jsonl`。每条保存原 outcome / control / inference 文件 SHA256、有效成功失败、控制步、请求数、wall 时间、请求 p50/p95 和策略显存峰值。基础设施失败与重试另存 `status.json`。

总体与逐配置报告有效样本成功数、样本数、Wilson 95% 区间；完全完成须包含全部 900 个唯一配置/种子身份。总体时延从全部实际请求汇总，含每次冷启动，不用每 episode 分位数再取平均。策略 PyTorch peak allocated 不等于整个模拟器 GPU 峰值；全程总 GPU 峰值尚未测量。阶段进度不是最终性能结论。

截至本记录首次核查，billiard hall / seed 0 与 Rubik home office / seed 0 已完成并成功；这是 **2/900 的启动进度**，不能当成完整基准成绩。后续以本地[数字快照](results/2026-10-10_pi05_arena_droid_benchmark/)和[固定离线页面](../../outputs/arena-droid-baseline/progress.html)的实际成功检查时间为准。历史小样本与 RAW 验收页面移至[history.html](../../outputs/arena-droid-baseline/history.html)。

06:01:06 UTC 快照已有 6 条有效完成，均成功。tomato sauce can / seed 0 首次尝试在动作请求前检测到全部 RGB 为黑，基础设施尝试不计分；按冻结协议在新 attempt-02 重试原身份，完整成功并通过帧刷新核验。原始尝试与日志保留，未修改运行代码；首次渲染就绪的底层原因尚未隔离，不能宣称已根治。小型诊断与实际日志 SHA256 见 `infrastructure_diagnosis.json`。

## 手动同步与核验

2026-10-10 08:01:47 UTC 手动快照已核验 **54/900 条完成：44 成功、10 正常策略失败，846 条尚待完成**；supervisor 仍在运行。累计三次基础设施异常均已在原配置/种子的第二次尝试中取得完整结果（两成功、一正常失败），不丢弃失败种子。900 是计划总数，早先页面的 49 是当时已完成数量；尚未完成的条目不是被排除的无效样本。本地页面已明确区分计划、完成、待完成及成功/正常失败，不修改冻结云端源码。

```bash
uv run --locked python -m owac.runtime.arena_droid_benchmark archive
```

同步仅白名单小型 protocol/status/summary/episodes、部署身份、源码哈希和专家来源审计。禁止下载检查点、权重、RGB/RAW 图像或数组。云端读取要求快照稳定；本地从逐条身份重算数量、成败与区间，并对照云端汇总。连接或验证失败时保留旧数字与成功时间，页面显示检查错误。离线重建用同一命令加 `--offline`，不冒充新的云端检查。

本地生成后刷新逐文件 SHA256 与 HTML SHA256；实际远端检查时间和本地页面生成时间分别记录。生成器和归档核验的本地修改不回写正在运行的云端冻结源码。任务由 supervisor 持续运行，聊天无需保持运行，不重建定时检查。实验完成后仍保留租用实例和云端权重。

第二阶段没有生成数据：公开 PolaRiS 仿真共同训练数据没有完整物体状态；通用 Isaac Lab 采集/扩增/回放脚本已找到，但当前 DROID 接口需适配并决定专家来源。新生成专家集不能冒充官方权重原训练集，评测策略轨迹也不当成原始专家演示。

本地 `make check` 与 34 项 CPU 测试通过，覆盖冻结身份、失败保留、重复记录拒绝和云端虚假完成拒绝。离线页面只做静态内容与哈希检查，未宣称浏览器验收；这些检查不作为完整 GPU benchmark 或专家回放验收。

## 用户终止后的留档（2026-10-11）

2026-10-10 18:25:57 UTC 最后成功同步：**290/900 条完成，211 成功、79 正常策略失败**，其余 610 条没有完整计分记录。用户决定后续统一使用 RAWVLA-bench，`owac-arena-benchmark-v1` 已停止；停止时未完成的 episode 不计策略失败。原协议、计分 ledger、基础设施记录及权重保留。

[停止记录](results/2026-10-10_pi05_arena_droid_benchmark/route_retirement.json)与[最后数字](results/2026-10-10_pi05_arena_droid_benchmark/summary.json)均已本地归档；11 个小型结果文件及离线 HTML 的 SHA256 逐项核对通过。页面明确标注路线终止，不把部分结果作为完整基准。未停止租用实例或删除云端检查点。

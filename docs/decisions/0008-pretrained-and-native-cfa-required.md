# 0008 现成策略与原生 CFA/RAW 同时作为准入条件

日期：2026-10-10。状态：用户已明确两项硬条件；正式主 benchmark 尚未选定。本文取代 [0007](0007-pretrained-benchmark-first.md) 中允许暂不满足 RAW/CFA 的准入安排。

## 要求与纠正

用户要求在开箱即用的基础上，仿真器必须支持 CFA/RAW 采集。“可先用原版 RGB”允许现成策略使用 RGB，不表示可以放弃仿真器的原生成像能力。此前把这两个条件先后处理导致第二次选型仍不满足整体目标。

主 benchmark 必须同时提供：

1. 对应任务、本体、观测和动作接口的公开预训练策略，不需要重新训练 backbone 才能建立可用基线。
2. 从渲染器的显示处理前线性 HDR 进入传感器采样，取得 ISP 前 CFA 和噪声版本；保存无损数值、CFA 相位、电平、时间戳和处理记录。
3. 在同一固定环境、同一完整策略 episode 中，同时证明任务成功和连续 RAW 采集，能够同步记录动作、本体状态及成功判据。

基线可沿用官方 RGB 输入；后续研究候选使用同步采得的 CFA 替换视觉入口。由 RGB8 逆处理或重新马赛克得到的 pseudo-RAW 不满足第 2 项。合成传感器 RAW 须如实标记，未经真实相机标定不能称为真实相机测量。

## 已有组合的地位

| 组合 | 现成策略闭环 | 原生 CFA/RAW | 当前结论 |
| --- | --- | --- | --- |
| 修改后的 Lightwheel 红杯 / 自定义 π0.5 微调 | 0/100，有效负结果 | 已在 Isaac Sim 6.1 工程任务验收 | 退出主基线，停止 backbone 补救训练 |
| 原版 LIBERO-Goal / 官方 `pi05_libero` | 10/10 小样本验收，额外训练 0 步 | 当前 MuJoCo 管线不提供 | 仅保留为 RGB 参考，未通过主 benchmark 准入 |
| Isaac Lab Arena DROID 抓放 / 专用官方 π0.5 | 上游有对应权重、接口和成功示例；OWAC 未运行 | Isaac CameraCore 有原生管线；与该任务联合采集未验收 | 下一项验收候选，尚非已选定 benchmark |

保留全部原始数字、逐 episode 记录和权重；原实验运行状态不改成失败，选型不合格与已测策略分数分别记录。定时任务保持关闭。

## 下一项候选的可追溯依据

- 官方任务：`pick_and_place_maple_table`，DROID 本体 `droid_abs_joint_pos`，魔方放入碗；沿用官方场景、相机、动作和判据。
- 策略配置：`pi05_droid_jointpos_polaris`，权重 `gs://openpi-assets-simeval/pi05_droid_jointpos`，执行 horizon 15。不能用普通 `pi05_droid`、`pi05_libero` 或旧红杯微调权重代替。
- [Arena 官方运行说明](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/running_a_real_policy/openpi.html)提供零额外训练闭环。上游示例 9 组变化、每组 3 条；8 组全成功，sugar box 为 0/3。此为上游小样本，未在 OWAC 复现，不据此保证所有任务可用。
- [Isaac Sim 6.1 相机说明](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_camera.html#exposing-the-isp-camera-pipeline)提供 `camera_isp_pipeline.py`，可导出 HDR、Bayer CFA、噪声、压扩与 ISP 各阶段。现有 OWAC CameraCore 采集器在旧任务上已读取并核验原生 CFA；其任务迁移与同帧时序仍需实测。
- 本次审计 Arena 提交 `2cc9e9766f1f62a500b85c92207192d8c60fcce5`，IsaacLab gitlink `91a0906f93802eb859192dcb0bca879f891e92d6`，官方 OpenPI 固定为 `c23745b5ad24e98f66967ea795a07b2588ed6c79`。源码 URL、Git blob 与 SHA256 见[来源清单](../../research/reviews/2026-10-10_pretrained_raw_benchmark_sources.json)。

发现版本文档有漂移：安装页仍称 Sim 6.0.1 / Lab 3.0.0，而上述固定源码的 Docker 底座已是 Sim 6.1.0。此前的 Lightwheel 兼容补丁和旧 Arena 子集不能直接当作新官方任务的部署。需要按固定源码与锁文件组成独立 runtime，核验 CameraCore 能力后运行；不把同属 Isaac 生态当作兼容证据。

## 验收与停止补救的边界

先复现官方小规模抓放，再在相同任务中启用原生采集：逐帧核对 HDR → CFA 的相位、量化、尺寸、完整性与时间戳，保存 RAW 与动作/状态对齐证据，核查采集没有额外推进物理或改变策略输入。用固定版本和预定种子报告闭环结果、RAW 完整率及资源开销。通过后才冻结后续 100 次评测与算法比较协议。

如果官方权重在匹配环境仍不能直接完成任务，记录失败并撤回该候选；不再通过自定义 backbone 训练、改成功规则或修改机器人安装位置来取得准入成绩。少量采集接口接入与基础设施修复须披露，不能称作官方已有的一键 RAW benchmark。

本次完成要求纠正、现有组合撤回与候选源码审计。尚未启动 Arena 部署、策略推理或联合 RAW 采集；当前没有同时满足两项要求的 OWAC 实测准入结果。

# S0 可信数据与因果合同开发提示词

> 历史文档（2026-10-15）：后续开发已由 [N04 v1.1](../n04_v1_1/README.md)取代。原文与旧启动语仅供追溯，不作为当前执行指令；当前入口为 v1.1 S2，旧数据扩展与衍生权重不进入官方数据主实验。

## 执行上下文

你负责实际编程与实验。这份提示词已经指定研究路线，请按顺序完成当前阶段的代码、验证、训练或数据审计和交接，不停留在计划、空接口或伪代码。当前阶段之外的模块只保留必要接口。遇到数据事实与假设不符，先完成不依赖该条件的工作，再写出具体缺口、已检查来源与可执行补齐方案，不替换成虚构输入。

本地 checkout：`/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`；分支 `lzh/n04-egomotion-separation`。开始读 `AGENTS.md`、`README.md`、`docs/status.md`、[全量方案](../../plans/N04_full_development_v1.0.md)、[首版实现合同](implementation_contract.md)、[源码核查](preparation_audit.md)。先检查工作树，保留既有修改和 FilterPy stash。下面标“拟新增”的文件/API/CLI 是你的开发目标，不能当成已经存在。

**实际数据核查、正式训练和真实 π0.5 验证优先在用户指定远端执行：**

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

先读[远端运行规约](remote_execution.md)。数据根为 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`，LIBERO 在 `libero/raw`，RoboTwin 在 `robotwin2/raw`。新源码和产物放拟创建的 `/workspace/owac/research/n04-v1/`，冻结 RawVLA 数据/源码作为只读参考。纯 encoder 使用 S0 固定的独立 Python 3.11+ 环境；后端继续验证 `/workspace/owac/rawvla/envs/starvla/bin/python`（实测 3.10.21 / torch 2.7.1+cu128 / numpy 1.26.4）。本地 Python >=3.11 / NumPy >=2.3 的 uv 环境不能直接覆盖它。正式代码经包含未提交文件的 hash 快照同步，远端复核 hash 后运行，不能只同步 HEAD。

实际 GPU 是单张 96 GB 级 RTX PRO 6000；上次约 32 GB 被已有服务占用。先查 `nvidia-smi`、`supervisorctl status` 和磁盘，按剩余资源 profile，不自行停止其他任务。正式长任务用 N04 专属 supervisor program、绝对解释器与不可变源码快照；日志/checkpoint 留远端，小报告/hash/曲线回本地。

模型主线固定为完整 ConvNeXt-B、作者 RKN/Ac-RKN、显式几何与误差传播。旧 tiny N04 留作历史。输入模式 `rawvla_rgb3` 与 `bayer_cfa` 分开；主链没有 RGB/sRGB/XYZ 中间图、RGB 教师或 RGB 重建。实际执行动作、当前本体与未来动作监督各有独立字段。正确实现和负实验结果同样交付；不预设新方法必须胜出。

## 本阶段目标和完成边界

先把每一帧“实际测到了什么、什么时候到达、上一帧后真正执行了哪些控制”写清楚。例如：腕部相机图像到达时，机器人已执行两条限幅后控制；预测只能读取这两条记录和此前到达的本体，不能读取刚生成但尚未执行的动作块。S0 无神经网络训练，负责数据、坐标、诊断任务与后续远端运行设施。

当前 loader 确实把 `executed_action` 设为 None，时间为 frame；不要只把 `supervised_action` 改名来填空。远端有 1771 个 LIBERO RAW NPZ、632 个 RoboTwin RAW NPZ，但源轨迹/变体独立性仍须按 metadata 审计。

## 先读的现有实现

- [RawVLASequenceLoader / source_identity](../../../src/owac/raw/rawvla_sequence.py)：平台字段、metadata、拆分身份与 supervision offset。
- [MeasurementFrame / EmbodimentContext](../../../src/owac/core/encoder_contract.py)：保留旧用户，扩展物理时间通过专门类型。
- [build_real_context](../../../src/owac/training/n04.py)、[proprio_delta](../../../src/owac/frameworks/n04/proprio.py)、[EgoMotionContext](../../../src/owac/frameworks/n04/ego.py)。
- [geometry.py](../../../src/owac/frameworks/n04/geometry.py)：`GridSpec`、`forward_project_points`、`forward_splat`、`warp_feature_field`。
- [合成器](../../../src/owac/frameworks/n04/synthetic.py)、[逐样本标签回归](../../../tests/encoders/n04/test_render_labels.py)、[真实数据测试](../../../tests/encoders/n04/test_realdata.py)。
- 远端 RawVLA 作者 NPZ loader 与仿真 rollout/controller 实现，入口见[参考导航](references.md)的 R05/R06；从实际字段生产者追溯，不能仅根据名称猜时间。

## 文件与具体开发顺序

1. 拟新增 `src/owac/raw/telemetry.py`：定义 `FrameTiming`、`ExecutedControl`、`CalibrationRecord`、`TelemetrySidecar`，只存数据与校验语义。`FrameTiming` 含 exposure_start/mid/end、arrival_time、clock_id、单位、availability；`ExecutedControl` 含 command_id、command_time、start/end、下发后的数值、mode、units、frame、执行状态和来源日志位置；`CalibrationRecord` 含 K、T、畸变模型、坐标约定、时间偏移、标定版本、可用性与 provenance。可选滚动快门行时间独立标识。
2. 拟新增 `src/owac/raw/n04_manifest.py`：远端逐 NPZ 读 metadata 与数组 header，输出平台/字段/shape/dtype/范围/源身份/变体/有效帧数/缺失率；再对分层样本真正读取数组做有限值、对齐和数值检查。避免每个 worker 反复解压整条巨型轨迹，记录 I/O 和内存，准备只读分片测量缓存格式。
3. 扩展 `RawVLASequenceLoader` 为可选 sidecar 加载，默认旧路径行为保持；把普通测量、在线可用上下文、训练目标放入不同结构。新增 `context_at(exposure_time, decision_time)` 仅选择当时已到达的信息和已经执行的控制片段；`executed_between(t0,t1)` 保持命令顺序，并检查覆盖与间隔。
4. 拟新增 `src/owac/frameworks/n04/measurement_geometry.py`：先实现命名明确的 `T_A_from_B` 变换、像素/CFA/特征网格仿射、内参变换与有限差分辅助。保留 wxyz 边界；对 `relative_rotation_vector`：已知 `R_W_from_B` 时，world increment=`R_t @ R_prev.T`，body increment=`R_prev.T @ R_t`，分别测试。首版保留旧函数数值行为并修正注释/命名，新增显式 frame 参数的正式接口，不悄悄换旧 checkpoint 输入语义。
5. 拟新增 `src/owac/raw/n04_mechanism_data.py` 及 `src/owac/evaluation/n04_contracts.py`：构建下述 D0/D1 数据和审计，不将 GT 混入在线特征。
6. 拟新增 `scripts/n04_full/package_source.py`、N04 supervisor 模板、`configs/n04/full/s0_data.json`、`research/theory/n04_v1.md`，完成远端 snapshot/环境/配置固定。

## 数据补齐的确定分支

先核查原始缓存、作者生成脚本、仿真版本与控制日志。LIBERO：`robot_states` 是 8 维末端/夹爪，target `actions[t:t+10]` 是 7 维；RoboTwin：当前缓存 `actions[t]` 作 14 维 joint state，监督从 t+1 起，16 维 robot_states 是末端状态。只有找到生产代码和执行记录，才能决定旧动作值是否可转成过去控制。

有严格同版本回放条件时，选每平台至少两个任务、每任务两条训练源轨迹试重放，比较相机帧、q/末端、时序和实际控制。逐项保存误差、哈希和容忍度；只有可确认逐帧对应时才挂 sidecar。不能确认对应就创建独立命名的扩展训练数据，记录新初态/演示/控制日志，不挂在原 NPZ 名下；物理变化改变正确动作时重新产生动作监督。

D0 实现无需额外仿真器的解析三维渲染诊断：可解释针孔 K、多深度面与独立刚体、相机三轴旋转/平移、z-buffer、出生/出界/遮挡、物体停止/反向；标签由同一逐样本世界状态计算。输入为线性合成测量，CFA 分支直接采样所声明的辐射通道与相位，明确 synthetic CFA。不要把旧二维裁剪平移当新三维数据。

D1 正式机制数据优先用已部署 RawVLA 的 LIBERO/RoboTwin 仿真资产重新记录训练轨迹，保存实际执行、K/pose/depth/visibility、接触状态和来源。D0 可完成合同测试，D1 才覆盖真实渲染/遮挡/接触复杂度。若资产或控制记录不足，完整交付采集器、已可验证数据与缺口；S1 只在有真实几何标签/可靠对应监督的数据上训练，不能把无标签 RAW 重命名为深度监督集。

新数据至少覆盖静态、纯相机运动、纯对象运动、同时运动、停止/反向、遮挡恢复、机器人自身可见结构、增益、噪声、时移、错误关联、单移动平面占满视野的不可辨识样本。给出 factor table，先单因素再组合。

## 拆分与预注册

按 `source_trajectory_id` 分组，所有 RAW/RGB、光照、噪声、裁剪变体在一侧。首版分层 train/dev/final 比例 80/10/10，split seed=20261014；不同平台分别分层，少量源组的任务列出分配，不能用相邻帧补数量。另建 object/scene/camera holdout 和未见因素组合，开发集不能与 final 共享源轨迹。官方 10000+3250 的仿真测试清单是独立任务清单，不参与训练拆分。输出显式成员表和 SHA256，拆分选择不能依赖模型指标。

冻结指标：prior/posterior 区分，米/原图 px/特征 px 区分，motion_presence/prediction_mismatch/RAW observable change 分开；missing error 按物理缺失时长分桶；revision latency 从第一条可用有效新测量计时，未恢复 censored。

协议继承旧二维门槛相对 10% 且绝对 0.05 px；三维绝对门槛依据标定分辨率与任务容忍度，先算并冻结再跑模型。H1 开发误报预算 1%，干扰负例误报相对降低 20%、事件召回下降不超过 2 个百分点；H2 停止/反向积分误差降 15%、缺失期退化不超过 5%。不能测完再改阈值。新三维任务的具体绝对单位和可见段容忍度必须有数字与来源，写入 protocol.json。

## 必做验证

- B=1/2/4、多物体、多相机，随机逐样本渲染标签用独立投影重算；置换 batch 后各样本不串位。
- K 已知的 90 度旋转、非交换旋转复合/逆、正负深度、像素中心、crop/resize/CFA 四相位，手算与 FP64 参照。
- 未观测值替换 NaN、极大值：无观测更新与梯度均不受影响；不能只比 finite output。
- 未来动作和当前图像置乱：当前 prior 不变；改变已执行控制：未来实际使用控制的模型应变化（S0 测上下文选择，模型效果留 S3）。
- 迟到帧按曝光时刻回到缓存更新，再重放已执行控制；尚未到达的本体不能泄漏。arrival 输出延迟另记。
- 静态无噪声、恒等几何、同曝光时 sensor residual 接近数值精度；正曝光增益同时改变均值/方差；饱和/遮挡支持独立。
- 生成两次相同 manifest，成员与 hash 稳定；拆分源 ID 交集为零。

## 要实现并运行的 CLI

以下是拟新增 CLI。实现后先 `--help`，再在远端实际执行 audit/generate/validate，不只复制命令到报告：

```bash
$N04_PYTHON -m owac.raw.n04_manifest --config configs/n04/full/s0_data.json --output "$N04_RUN_DIR/inventory.json"
$N04_PYTHON -m owac.raw.n04_mechanism_data --config configs/n04/full/s0_data.json --output "$N04_RUN_DIR/mechanism-data"
$N04_PYTHON -m owac.evaluation.n04_contracts --config configs/n04/full/s0_data.json --output "$N04_RUN_DIR/contracts.json"
```

`N04_PYTHON` 和 `N04_RUN_DIR` 从 S0 实际环境/新 run 设置，配置保存绝对数据路径。输出 inventory、coverage、splits、telemetry provenance、D0/D1 manifest、protocol、environment、contract results 与 source package。远端新报告 `pi05-complete-report-20261014-v1` 已显示基线补齐，核对身份/分母后引用，不重新跑旧基线。

S0 结束给出 S1 可用的真实训练样本与标签覆盖、S3 可用的动作/标定覆盖，以及具体缺口。全量数据未就绪时可交付工程完成与数据部分完成两个状态；不要用 fixture 宣称正式数据合同全部通过。

## 重点参考

[RawVLA 作者 loader](https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py)、[数据卡](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)、[Kornia 深度坐标](https://github.com/kornia/kornia/blob/4fcfcf863e0a9e2b3e93179c3fa96bcf278ce3c6/kornia/geometry/depth.py)、[时空标定可观测性](https://arxiv.org/abs/2201.09170)、[时间标定](https://arxiv.org/abs/1808.00692)。它们提供字段和估计原理；本文给出的 sidecar 与阶段协议是 OWAC 实现选择。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

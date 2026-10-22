# P6 真实 RAW 规模与可复用表征

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> **已被 v1.1 取代（2026-10-15），不再执行本提示词。** 当前正式路线见 [v1.2 完整方案](../../N01_full_development_plan_2026-10-15_v1.2.md)。旧局部阶段正文仅保留供追溯；新 P3 必须整体交付编码器、真实任务训练与开发闭环，不能按旧步骤继续或只改编号。新执行包尚未编写。

只执行 P6。读取[共同契约](00_SHARED_CONTRACT.md)、P4/P5 的冻结模型与结论、完整方案第 9–10 节。本阶段把小合成定位推进到真实 RAW 与第二骨干，但不把无动作 RAW 数据当机器人示范。

## 1. 数据与任务先具体化

沿用 P0 inventory，在服务器数据根以外建立本阶段版本化 RAW 数据入口。优先使用[CRVD/RViDeformer 官方资料](02_REFERENCES.md)，执行许可和文件来源登记。CRVD 官方说明包含真实 GBRG、black=240、white=4095；这些值只用于确认来源的对应数据，不能套给其他传感器。作者原训练包含 ISP/RGB 或逆处理合成部分，本项目只取符合身份的 RAW 数据与允许任务。

ReCRVD 官方说明是显示视频后相机重拍的 120 scenes、90/30 划分；它是传感器 RAW，但场景/运动来源与自然真实动态不同。保留官方 test，训练区再划 calibration/validation。多个数据集或多个 ISO 不等于多传感器，逐条登记 sensor_id；没有第二传感器就把跨传感器结果留未完成。

首版两个非图像读出任务固定为：局部结构/边界参数估计；跨帧几何点跟踪/位移估计。标签优先来自有效真实标注；否则在独立 clean RAW 上形成已披露的参考边界与对应，参考帧不可作为推理输入。对受控 RAW 几何变换的标签单列为半合成任务，不冒充自然动态真值。没有合法标签时先交付数据/可运行读出与缺口，不能自行造物体检测标签或从隐藏状态取答案。

新增 `raw/n01_real_raw.py`、`training/n01_transfer.py`、`evaluation/n01_transfer.py`、对应 configs 和 tests。复用 RAW packing、校准与四区 manifest，不在 loader 内转 RGB。读出任务头复用项目已有定位/标准线性头；复杂几何用同样训练容量的成熟头，两个方法公平共享。

## 2. 迁移实验

对同一 P4/P5 checkpoint 分别做 frozen-feature probe 与同预算微调；先冻结核心只训两种任务头，再统一解冻 backbone/规则允许参数。按场景、会话、传感器留出，zero-shot 和 adaptation 分报。校准与超参只用指定区域；输出逐条件、逐场景与种子，而非随机像素平均。

选三组核心 B2/B3/N-TS（若 P5 采纳状态，另外保留显式 N-TS 以解释增量）。所有组共享 RAW/动作外的可用监督，当前帧基线保留作参照。真实 RAW 预测辅助损失需要明确独立来源；不能读取 test clean RAW 调噪声模型。

## 3. 第二骨干 BSRT

读取[BSRT 固定 `bsrt.py`](02_REFERENCES.md)并在新 `backends/bsrt_raw_features.py` 做适配，保留作者 S/L 正式配置。使用 P0 锁定的资源档，先完整基线再接相同支持契约。源码里 `debayer`、SpyNet、RGB mean、预训练路径与图像重建头逐项核对：采用 packed RAW feature stem、Swin/融合部分，运动统一使用 P2 RAW 模块，去除 RGB 路径和输出图像头。

对原 RAW 合法子模块做同参数数值对齐；改变光流/输入后声明为 BSRT-derived RAW feature baseline，不声称整网复现论文 PSNR。B2/B3/N-TS 的三个 BSRT 组都使用同一适配骨干、对应、初始化和任务头。骨干变强带来的收益与支持规则差值分别报告。

不要在 P6 同时启用未验收 Perceiver 与新状态模型。默认仍用已有共同读出；任务查询必须有必要性和同容量控制才增加，放独立实验身份。

## 4. 规模和实际成本

逐轴比较历史 8/16/32、packed 尺度 64/128/224、可用数据量与活跃域数。每次固定其他轴，记录相机数、batch、有效样本量、吞吐、GPU 显存/状态、I/O、重放率和误差。耗尽显存的点保留为不支持配置，不缩成小模型后沿用正式名称。

核心结果三训练种子；不要每个规模点都重新扫所有超参。先开发区选合法配置，再完整比较。规模曲线仅描述这些测试点，不宣称 scaling law。

## 5. 交付与完成定义

交付 `P6_report.md`、真实 RAW 数据/许可/标注来源表、跨场景/相机拆分、两个任务×两骨干关键矩阵、frozen/tuned 差异、尺度成本和实例失败。CFA 与 `rawvla_rgb3` 分表；真实 RAW 没拿到则此阶段数据项未完成，不能用重马赛克 RGB 顶替。

运行 loader/标签不泄漏/相位/未来帧边界测试、P1/P4/P5 回归和 `make check`；服务器完成实际迁移训练与测量。形成推荐进入 P7 的明确模型/输入/配置/支持规则版本，然后停在 P6 交接点。

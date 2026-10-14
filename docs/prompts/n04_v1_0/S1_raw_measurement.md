# S1 正式 RAW 测量开发提示词

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

## 阶段目标与输入

只增加正式当前帧测量网络。相同一张 RAW 帧，无论之前经历什么轨迹，`measure(frame)` 都应输出同样的测量；滤波、关联与历史尚不进入这个函数。先读 S0 的 handoff、source/split/data manifests、几何标签覆盖与标定合同。若 S0 无真实尺度标签，先使用明确拥有标签的新机制集；无尺度数据只能训练已定义的对应目标，不能测试时借真值补尺度。

## 现有入口与拟新增文件

现有 [MeasurementStem](../../../src/owac/frameworks/n04/model.py) 仅约万参数视觉诊断来源，不扩它的 width 冒充 Base；[RAW loader](../../../src/owac/raw/rawvla_sequence.py) 可复用读取与源身份；[旧训练器](../../../src/owac/training/n04.py) 仅参考数据接口与历史问题。

拟新增：`frameworks/n04/full_config.py`、`contracts.py`、`measurement.py`、`training/n04_full.py`、`evaluation/n04_full.py`、`configs/n04/full/s1_measurement.json`、`tests/encoders/n04_full/test_measurement.py`。训练基础设施放 training 包，scripts 只启动。

## 固定网络与输出

1. 引入作者 ConvNeXt-B 完整 stages：depths [3,3,27,3]，dims [128,256,512,1024]；保留 depthwise 7×7、LayerNorm、4× channel MLP、残差与 layer scale 的原结构。移除分类全局池化/分类头，导出四级空间场。集成时固定源码 SHA、license、实际修改 diff 和参数总量；初版随机初始化，任何 ImageNet 权重另开公平实验。
2. `rawvla_rgb3` 使用三测量通道。LIBERO 按已有 uint8/255 显式缩放；RoboTwin 保留 float32 测量单位，按训练集确定统一尺度，不逐帧 max 归一化。`bayer_cfa` 从单平面无损拆成四相位，在各相位噪声/掩码和采样坐标上处理后以四通道输入，无 demosaic。四相位真实坐标不可合成同一中心。传感器归一化先 FP32，metadata 记录尺度与方差尺度。
3. 统一输出四级特征，经 1×1 投影到 256 维，在 stage1 网格汇聚；空间融合为上采样相加 + 3×3/256 输出层，采样约定 align_corners=False。输入 pad 到适配步长并同步 support；CFA 先四平面再 pad。原图 pixel center→packed→各 stage 中心映射由 S0 的仿射链给出，不写死 `xy/stride`。
4. `DenseMeasurement` 拟字段：feature_pyramid、latent_mean_map(256)、latent_variance_map(256)、depth_mean、depth_variance、valid/support、sample_coordinates、measurement_ids、unit/scale。方差用稳定 softplus + config floor；深度正值头与真实尺度监督匹配。几何位置由已知 K/unprojection 得到，协方差用 Jacobian 从像素/深度单位传过去；K 不可用时 position unavailable，不造默认内参。
5. 单帧候选最多128个，按共享合同 8×16 分区选有效点；这个候选 ID 仅代表本帧位置，持久 ID 由 S2 关联层分配。有效性不借 GT 物体 mask；GT 只在损失与独立 oracle 诊断使用。坏点/饱和/传感器缺失与物体被遮挡是不同现象，不能把所有遮挡者图像清零。

## 训练目标与默认配方

几何主要 loss：有效支持上归一化三维位置/深度 SmoothL1 + 同单位 Gaussian NLL（含 logdet）；匹配图像的已知投影点做带负例的描述子对应学习。两帧各自独立编码，第二帧只通过训练目标参与，不能把两帧堆输入单帧测量网络。初版 `L=1.0*L_geometry+0.1*L_geometry_nll+0.1*L_correspondence`，所有项按有效点平均并用训练集固定尺度；latent 方差若未有可信 residual 监督，先配置固定初值，S2 在冻结 z 后用预测创新校准，不能凭自监督特征距离宣称校准完成。

RAW 域辅助预测是可选对照，不是默认主损失。严禁只最小化相邻特征差导致常数表征。输出 feature variance、空间方差、几何误差、正/负对应区分、方差覆盖/宽度，检查是否塌缩。GT/current measurement diagnostic 只替换测量，更新和指标保持一致。

正式默认 100k 步、最多300k，effective batch128，AdamW 配方见共享合同。先正式形状500–1000步 profiling，随机初始化 Base 确实完成反向，测峰值显存/吞吐/GPU小时；再执行学习率开发网格，均等选择预算，三 seed。数据顺序按 source group 采样，不让长轨迹或变体过多的场景占据全部 batch。每5k评开发集、最后只评冻结测试。

## 无记忆基线与缓存

建立相同骨干、相同测量/读出头的无记忆模型。可见帧直接当前测量；缺失期间保留最后有效测量且明确 age。这里固定 baseline 逻辑，S2 不另换更弱测量。

选定 S1 开发 checkpoint 后，用 eval 模式、关闭随机增强导出逐帧测量缓存：源文件/帧/相机/measurement ID、权重/预处理 hash、所有输出、GT 独立文件、mask、坐标与单位。hash 不同的 cache 不混用。缓存导出只读取本帧；随机抽样重新在线计算核对。S2 状态种子共享同一缓存，S6 三完整种子则各自生成对应独立 S1 模型，公平对照共享各自种子内的测量。

## 关键验证与命令

- 替换全部历史/轨迹 metadata 后 measurement 不变；交换 batch 样本仅置换对应输出。
- Bayer 四种 pattern、相位 crop、mask、pad 的精确采样位置与人工独立例子一致。
- NaN 无效 RAW 在网络前被 where 清理；有效值不受隐式 PIL/uint8/JPEG 路径影响。
- 每个 stage 参与几何 loss，参数集合等于 optimizer 分组并有非零有限梯度；所有层在 optimizer 创建前注册。
- 同一 source/camera/frame 的重复导出一致，缓存和在线误差在预定容忍内。
- 一个短训练连续执行与中断恢复执行比较 sampler ID、LR、step、参数和 loss；不是只检查 checkpoint 能 load。

你要实现下列 CLI 合同，后续阶段复用：

```bash
$N04_PYTHON -m owac.training.n04_full --config configs/n04/full/s1_measurement.json --mode profile --run-dir "$N04_RUN_DIR/profile"
$N04_PYTHON -m owac.training.n04_full --config configs/n04/full/s1_measurement.json --mode train --run-dir "$N04_RUN_DIR/train"
$N04_PYTHON -m owac.evaluation.n04_full --config configs/n04/full/s1_measurement.json --checkpoint "$N04_CHECKPOINT" --split dev --output "$N04_RUN_DIR/dev.json"
$N04_PYTHON -m owac.training.n04_full --config configs/n04/full/s1_measurement.json --mode export-measurements --checkpoint "$N04_CHECKPOINT" --run-dir "$N04_RUN_DIR/cache"
```

`--mode profile` 与正式 run 分开，profile 权重不默认为正式 seed 初始权重。`--resume <checkpoint>` 恢复完整状态；stage/config/source/split 身份写报告。出口为正式训练曲线、几何/对应误差和校准、三种子或明确未完成状态、无记忆基线、可复现缓存；不要求 Base 在简单 toy 上一定胜小模型，也不以 toy 成败宣布路线成立。

## 重点参考

[ConvNeXt 完整源码](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py) 读 `Block/ConvNeXt/downsample_layers/stages/convnext_base`；[原论文](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)；[传感器噪声作者资料](https://webpages.tuni.fi/foi/sensornoise.html)；[Monodepth2](https://github.com/nianticlabs/monodepth2) 只作为独立正式深度模块的候选，初版不用它替代本文已定几何头。参考的完整版本和复用边界见[导航](references.md)。

## 检查和交接要求

本地运行 `make check`，以及 `uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；后者的新目录由当前正式开发建立。远端按真实环境执行本阶段对应的数值/数据/模型检查。新增测试围绕本阶段关键行为和独立数值参照，不用重复 assert 形状堆数量。新训练模块需要实际反向、参数变化和 resume 验证，不只 `--help`。

交付本阶段实验 Markdown、`handoff.json`、配置、源码与数据 manifest、实际命令、原始结果索引、失败组和资源信息，格式见[共享合同](implementation_contract.md)。交接报告分开写“代码完成”“运行完成”“机制结果”“下一阶段输入”。训练未完成时写 supervisor/PID、已完成步数、最新 checkpoint、剩余工作；不把启动作业当完成。更新状态页和 N04 候选卡的对应事实，保留历史结论。不自动提交、推送、发布或跨入下一阶段。

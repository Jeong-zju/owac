# P2 RAW 运动对应与噪声模型

本次只执行 P2。读取[共同契约](00_SHARED_CONTRACT.md)、P0/P1 报告与 P1 实际来源/坐标接口。正式训练优先在 `ssh -p 44327 root@107.209.104.125` 的独立 GPU 环境执行。目标是同一杯子快速移动但仍能准确对应时保留历史；对齐不确定或遮挡时明确降低支持，不能把速度阈值作为撤回理由。

## 1. 复用与新增边界

按 [SEA-RAFT 固定源码导航](02_REFERENCES.md)读取 `core/raft.py / extractor.py / corr.py / update.py / train.py`。沿用完整作者特征提取、相关金字塔和迭代更新；新增 `src/owac/backends/raw_correspondence.py` 包装，厂商代码放版本化 vendor 或固定安装源，并保留许可和差异说明。

新增 `raw/n01_motion_data.py`、`training/n01_motion.py`、`evaluation/n01_motion.py`；配置放 `configs/encoders/n01_full/p2_motion_*.json`，测试放 `tests/encoders/n01_full/test_motion.py`。噪声校准放 `raw/n01_noise.py`，保存可追溯 calibration JSON。

## 2. 输入层要做完整适配

SEA-RAFT 当前 fnet 输入 3、cnet 输入 6，内部有 `/255`；`ResNetFPN` 默认还可能下载 ImageNet。CFA 模式改为 4/8；三通道模式为 3/6，但按实际 RAW 单位归一化。关闭自动 RGB/ImageNet 权重加载；`args.pretrain` 中模型深度选择与实际加载预训练权重要拆开，不能只换名字。

保留 RAW 分相位数值、坏点/饱和/越界 mask。归一化用固定传感器/曝光校准或训练区间统计，不按每张图 min/max。曝光、增益、黑白电平对值和方差同时变换；未知参数按未知记录。CFA 与 pseudo-RAW 两种 stem 单独配置、分别训练，参数差计入报告。

P1 的坐标单位是完整传感器 xy；网络可在 packed 网格预测。输出适配要做单位换算：packed 位移一格等于传感器两格，四相位有各自偏移。明确 `source_xy` 为参考帧→历史源的 pull mapping；为当前 t 和过去 s 估计双向一致性可以调用两次同一已到达帧对，不能读取 t+1。

## 3. 对应结果和噪声

实现 `estimate(reference_frame, source_frame) -> CorrespondenceMap`，输出每个参考查询的历史源坐标、可见/有效 mask、双向误差及 flow 风险。双线性插值返回实际四 taps 的 SourceKey 与权重；边界使用零填充/无效，无环绕。合成数据包括 0.25/0.5/0.75 像素位移、不同 CFA 相位和亚像素真值。

已知协方差时插值方差为 `a.T @ Sigma @ a`，不是对 variance 图按一阶插值。不要复制旧 `n01_dbsr_raw.align_planes(variance)` 作为新协方差算法。对应风险从独立 calibration 集拟合 error/coverage 表；SEA-RAFT 的 info 输出首先标为 proxy，验证前不当作后验 covariance。

噪声首版为 RAW 单位下 shot/read 与共享黑电平块：`Var(y|mu)=a*max(mu,0)+b`，a/b 来自训练/校准或明确合成参数。相关黑电平误差在 P1 联合因子表示。`grad(f).T @ Sigma_flow @ grad(f)` 为一阶近似，分别评估强边缘/遮挡下的覆盖误差；不塞成无条件正确的传感器噪声。

## 4. 训练任务与配方

先从已知连续辐射场采样 CFA 和噪声，生成平移、旋转、分层遮挡、局部非刚性、重复纹理、亮度/曝光变化及运动模糊；几何标签在生成器 supervision 中提供，模型只见 RAW 与可用 metadata。真实 RAW 数据能用时加入同传感器序列；来自 RGB 逆处理的素材单列，不混称真实 CFA。

训练保留作者迭代预测损失/权重结构，按照合法 valid flow mask 聚合，单位与输出一致。先 256 updates 的小集可拟合检查；再在开发集对两档 LR 和作者合理迭代档作同预算选择，固定后跑共同契约的 100k–300k 候选预算，实际到平台才结束。训练/校准/验证/测试四区独立；正式 checkpoint 根据 motion validation 选择，不根据 N01 test 收益选择。

旧简化对齐、无对齐、实际 SEA-RAFT、oracle 四组用同样 RAW 序列。oracle 只作上界，单列指标；旧简化版本也记录真实输入信息，不能获得隐藏 flow。

## 5. 验证与交付

必须交付：EPE/分位误差、遮挡/错配识别、风险分档覆盖、静态误拒、CFA 相位一致性、端到端 p50/p95 与显存。分别列快且可对齐、快且困难、慢但结构改变、当前单帧异常；比较时 fixed support/oracle support 是诊断标签，不作为实际推理输入。

测试覆盖 identity、整数/分数平移、正负 flow 方向、边界、相位、曝光单位、插值协方差、未来帧不可访问。用梯度及下一步更新确认模型真实训练，保存完整恢复收据。若实际模型与 oracle 差距仍大，报告对齐限制并提供可运行实际版本与 oracle 诊断配置，不删困难条件。

P2 末执行[早期策略预检](EARLY_POLICY_PREFLIGHT.md)的 `mode=raw_adapter`，检验真实三通道与已有桥，不能把它算作 P7。写 `P2_report.md`，交付可加载 motion checkpoint、校准 JSON、实际 config/lock、坐标说明和测量报告，然后停在 P2。

参考：[SEA-RAFT 论文](https://arxiv.org/abs/2405.14793)、[传感器噪声原始资料](https://webpages.tuni.fi/foi/sensornoise.html)、[官方代码逐文件导航](02_REFERENCES.md)。

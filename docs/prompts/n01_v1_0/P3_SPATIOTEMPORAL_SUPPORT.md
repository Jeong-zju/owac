# P3 空间与时间支持共同组织

本次只执行 P3。读取[共同契约](00_SHARED_CONTRACT.md)、P1/P2 的报告和实际模块、方案第 4.4–4.7、5.3、7、9 节。目标是让同坐标的旧遮挡物来源退出，同时保留未变化区域和单帧异常之前的可靠历史。正式实现不接收事件时刻、真值标签或 oracle flow。

## 1. 文件职责与默认选择

新增 `frameworks/n01_full/temporal_support.py / spatial_support.py / calibration.py`，扩展 P1 的域和回放接口。新增 `evaluation/n01_support.py` 与对应测试、配置。首版采用显式解析规则，不再加小型“可靠性预测网络”。

实现统一 `TemporalSupport.update(measurement, predictive_state) -> SupportDecision`。返回 accepted/pending/rejected 的源 ID、变化候选后验、原因、版本和被截断质量。三种策略独立配置：普通 BOCPD；作者 DSM-bocd 鲁棒模型；顺序检验参考。B3 选用同样检测器但整域 reset，N-T/N-TS 则按来源时间和空间成员重建，以隔离规则增量。

## 2. BOCPD 和鲁棒支路

复用[作者 DSM-bocd 的 `bocpd.py` 与 `models.py`](02_REFERENCES.md)作为批量数值参考；重构为因果流式状态并与每个序列前缀逐点比较。普通已知方差支路用作者 `GaussianUnknownMean` 的共轭更新；仿射结构先在 P1 给出的固定坐标设计下推广为局部线性模型。异方差支路显式使用逐源 variance，不能把所有残差标准化后仍宣称原模型完全等价。

用 log 概率、logsumexp；保留最多 K 个 run-length 候选。K 首版 32，另与完整小序列参考比较 K=8/16/32 的误差和每步剪掉的后验质量。run length 明确定义为 **当前 segment 已接纳的测量数**，用 source→segment 映射判断某历史源是否属于当前结构，不靠帧数组下标猜偏移。

默认 hazard 候选对应平均 segment 16/32/64 个有效测量，只在开发区锁定。接受阈值初始 0.95、拒绝阈值 0.05；中间为 pending。missing/bad/saturated 不作为零残差，也不触发反证；物理时间 aging 单独记。重复包既不更新 likelihood，也不推进 run length。

单帧异常：鲁棒支路沿作者 DSM 模型和影响函数运行，并在一次脉冲/持续阶跃之间校准。若添加显式 outlier mixture，命名为独立 `bocpd_outlier_mixture`，旧历史在 outlier 假设下保留、异常当前源为 pending；不能把这个工程混合冒称作者 DSM 实现。根据 calibration 的静态误拒与持续变化恢复联合选择用于后续的检测器，B3 与 N01 同享搜索预算。

## 3. 空间相容与局部分裂

先以同相位的四邻域提出候选；对两个域在相同查询坐标 q 评估 `d_q=phi_q.T(theta_A-theta_B)`。独立固定域时求差值方差，重叠来源时通过来源联合 covariance 计算交叉项。只有预声明置信区间整体位于 `[-epsilon_structure,+epsilon_structure]` 且合并后的域级残差检验通过才合并。区间不够窄记 pending，不是相容。

`epsilon_structure` 用校准 RAW/几何单位，不能比较不同中心坐标下的截距/斜率。合并后对整个域重新拟合与检查，覆盖 A≈B、B≈C、A不≈C 的链式反例。时间 segment 不相容时不合并。证据不足时保持细域，允许缺失，不用多数投票强行归物体。

持续变化先定位时间候选，再对可观察残差形成连通子域候选；受影响部分生成新 segment 与域版本，稳定部分继续累积。子域从源集合重建；被撤回源内容不经过父特征或旧权重返回结果。

## 4. 有限适用域的统计保证

工程 BOCPD 后验与频率学 T3 分开。统计参考先采用满足条件的零均值高斯/sub-Gaussian 创新 X_t、已知条件 variance bound v_t，lambda 在观察 X_t 前固定：

```text
E_t(lambda) = exp(lambda * sum(X_i) - lambda^2 * sum(v_i) / 2)
E_t = sum_k pi_k * E_t(lambda_k)   # 固定双侧 lambda 网格，sum pi=1
retract when E_t >= 1 / alpha_j
```

使用已知固定参考均值/独立合法预测模型的合成支路证明；如果实际 flow、域或残差均值从当前测量选择，标经验支路，不直接套用超鞅保证。所有新域、候选和重启分配全局编号 j≥1，`alpha_j=alpha_total/(j*(j+1))`，首版 `alpha_total=0.05`；merge/reopen 不重置统计预算。空间等效性使用合法区间或独立校准，普通“未拒绝差异”不是等效证明。

静态重复 Monte Carlo 的误撤回估计与置信区间支持数值检查，理论结论来自假设和推导，不能要求有限样本经验率每次必然≤alpha。加入未知均值/同源选择等反例，显示保证失效条件。

## 5. 实验与出口

矩阵：固定历史 GLS、鲁棒固定历史、检测后整域 reset、N-T、N-S、N-TS。先 oracle 对应隔离支持机制，再用 P2 实际对应。事件位置随机；无事件长序列、长静态后阶跃、脉冲、遮挡解除、快而可对齐、慢且局部变化、曝光改变、重复/缺失/异步、相似物交叉都要覆盖。

除了平均误差，保存逐源接纳/撤回、错误历史质量、静态误拒、事件后 AUC、恢复时长与删失、实际唯一测量数。为标注数据中的“错误历史质量”定义权重口径；attention mass 和解析信息量分开。变化方向反转、一次性异常后恢复必须可重现。

性能出口按共同契约预注册标准；没有增量时交付正确规则、完整负结果和与 B3 的差异，不把 P3 工程通过写成性能通过。运行 P1 回放/数值回归与 `make check`；服务器执行长序列和实际 motion 成本测量。交付 `P3_report.md`、冻结 detector/calibration/config、T3 适用表、逐事件数字，停在 P3。

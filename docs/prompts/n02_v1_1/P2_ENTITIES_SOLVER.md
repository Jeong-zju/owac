# 给DeepSeek：P2，循环实体与固定结构全局求解

阅读[共同契约](00_SHARED_CONTRACT.md)、P1和E1真实交付。实现固定实体数/固定关系内的实体循环与连续求解，先获得可信全局重算结果。本轮不启用出生/拆分、关系学习或学习局部区域选择。

## 1. 文件和原始实现

实现`entities.py`、`solver.py`，扩展`state.py/factors.py/model.py/readout.py`；新增训练stage p2与`evaluation/n02_full/p2.py`。来源为[R02 SAVi](02_REFERENCES.md#r02)的Processor、SlotAttention、GRU、predictor/decoder和[R09 Theseus](02_REFERENCES.md#r09)的LM、AutoDiffCostFunction、TheseusLayer。逐层parity沿用P0收据；不要写一个小ConvGRU后称SAVi移植。

## 2. 对象状态与预测—校正

16活动槽、宽512、4个标准self-attention predictor块，初始SlotAttention corrector迭代4次，稳定后8次。对象ID由identity管理器维护，不永久等于slot下标；遮挡保留ID与存在性分布，未观测不等于死亡。固定结构阶段N按task配置固定，unused槽有明确mask。

采用`predict_time(z_prev, executed_action_if_available)`预测t；同t进行`correct_measurement(predicted, X_t, iterations)`。重新求解同一t时不再次应用时间转移。对象表示包含hidden、存在性、2D视角几何或有标定时3D位姿；米制量单位和坐标系随geometry_frame保存。

初始化采用学习的slot均值/正标准差加已记录的高斯随机数，不用EvaluationTruth的首帧框。独立原版SAVi复现可使用原论文条件框，但该收据与正式无隐藏标签初始化分开。固定结构ID关联先用Hungarian：预测位置的按噪声尺度归一化距离加appearance余弦距离，初始权重1:1，在dev冻结；不可见位置项用availability屏蔽，只能用已有历史。匹配仅决定ID/槽排列，不把GT ID塞回模型；有监督loss的匹配另行计算。P3的birth/split/merge生成新ID并记录祖先，避免复用已退休ID。

多视角feature带camera和坐标编码拼接，使用共享slots校正；同一世界保存所有视角关联。没有外参时不施加伪3D一致性因子。几何/深度监督仅在训练可用的数据上启用，正式推断只用RAW/context；提供同监督标准SAVi式基线。

将SAVi inverted attention的slot维softmax和有效输入归一化准确移植；缺失输入权重为0，全缺失时保持预测状态。实例mask可由模型推断，不能从truth代填。学习属性与成像η分开，使全局相机位移与单物体移动可区分。

## 3. 对象组合的测量模型

以官方spatial-broadcast decoder结构生成每实体RAW参数及alpha/depth，背景独立分支；按声明遮挡组合规则形成测量均值/方差。首版可以使用可微soft depth排序；记录其近似与训练监督，比较无深度/有深度条件。输出仍是传感器采样上的预测，只用于factors与检查。

主要可优化变量限定为几何、速度、成像η；对象appearance/hidden由观测/历史提议后在一次内求解中固定，防无限自由decoder拟合所有候选。所有候选共享解码参数和固定复杂度先验。对几何变量的小扰动必须确实改变对应RAW预测与残差；每变量做敏感度和梯度表，不能用未连接的几何head交差。

## 4. 一个完整目标与两种接受

`FactorSet`由唯一测量、初始先验、每个时间转移、固定关系约束组成；每个因子有ID、输入变量版本和measurement支持。固定ObjectiveKey期间不可改变因子集合、温度或先验。先验不重复包含动态因子已经计算的转移。

`GlobalSolver.solve(problem, initial, budget)->SolveResult`返回best_state、初末E、每步候选/实际E、残差/梯度范数、接受与拒绝数、耗时、converged/budget_exhausted。实现以下确定流程：

1. 计算完整E(z)；用Theseus LM对固定方差的平方残差子问题提议d。P2起点max_iterations8、FP32；参考小问题FP64。
2. 用完整E重新评价z+d，包括log方差、饱和项、先验及跨对象因子。接受阈值 `E(z+d) <= E(z)-c*||d||²`，初始c=1e-4，在P0/P2开发数值例冻结。变量先按物理尺度无量纲化，否则该norm无意义。
3. 不通过则试完整E的梯度方向，初始步长1、每次乘.5、最多12次；仍不满足则保留当前best，记录失败/预算。小步或梯度到容差是stationary/未证明收敛的分类，不伪记接受。
4. 学习网络另外提出连续delta，经过完全同样的完整E接受规则；与纯LM、纯梯度、固定迭代同预算比较。

每批/每候选独立接受，不能一个batch平均下降掩盖部分样本能量上升。新测量/结构/模型版本切换另起ObjectiveKey，T2不跨这些变化比较。

训练先对固定、明确的unroll（4后8次）反传，Theseus用已数值验证的unroll模式；部署硬接受的梯度只沿被接受路径，不给拒绝步骤编造straight-through。隐式梯度作为后续单独实验，未达到驻点时不用其性质。测试torch gradcheck或FP64有限差分时选远离离散接受边界的固定路径。

## 5. T1/T2与必做反例

在`research/theory/n02_full/t1_t2.md`写出目标、假设、证明、失效范围、代码对应函数和测试ID：T1来自集合幂等；T2来自充分下降/下界，不能推出全局最优或任意GRU收敛。对接受序列求和给出步长平方和有界，驻点结论须另列额外条件。

测试至少包括：同帧1/4/8次内循环固定候选E不变；新帧正常新增因子；重叠views/scales不重复likelihood；单个batch候选上升被拒；LM二次下降但完整异方差NLL上升的构造例；拒绝后回溯成功/耗尽；相机移动与对象移动；slot排列变化不改变实体身份；部分reset/遮挡重现；窗口重放输出与从头一致。

动作读出保留测量特征与对象信息，并另跑object-only、direct-RAW。P2的全局attention依赖如实登记为全局；不要借旧propagate掩码来声称局部求解。

## 6. 训练、指标与交接

100k更新×3seed，有效32条×16帧。Base lr1e-5、新实体/decoder3e-4起点；AdamW、warmup5%、clip1；与P1同样有限0.5/1/2×开发网格后冻结。loss起点：RAW1、geometry/depth1（只有可用样本）、temporal_prediction1、existence .1；各分支按实际标量数归一化，单目标和联合短试后固定权重。模型实例/关系真值只在loss/evaluator。

按冻结机制集报告：IDF1≥.85、遮挡后重识别≥80%；全部接受步E上升计数0；重复观测不增证据；预算耗尽带剩余残差。相同Base/信息/监督比较direct、标准SAVi、纯优化、学习提议+接受，计算预算和充分调优两张表。学习循环必须有可重复行为收益，不能只比较训练loss。

保存全局求解轨迹、每对象来源/身份事件和不同表示的局部细节误差。实现完成后执行E2提示词的objects模式预检；E2也在P2交付中链接。若对象化在充分训练后仍损害关键测量/动作细节，报告具体证据并停在P2交接点，由用户与本聊天决定表示重选，不自动进入P3。

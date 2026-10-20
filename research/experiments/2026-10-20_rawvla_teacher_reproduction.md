# RawVLA 教师限定复现：授权执行记录

2026-10-20 用户明确授权：“授权上述限定教师复现、必要示范回放和仿真验证范围”。依据[提案](../../docs/plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md)，本记录追加执行事实，不改写提案原先未授权的历史。机器授权收据：[teacher-authority.json](raw_stem_siglip_20261020/teacher-authority.json)。

范围：task6 Normal，50源40/5/5，seed101；只更新作者RawVLA前端；有效batch8，累计≤2000更新。保留burst6／rec6／stride5；horizon10／flow10／replan5；采用公开实现动作.15、brightness.01、chroma.001及AdamWwd1e-8。最多50条必要示范回放、最多50场策略闭环（10验证＋20参照／20教师测试）；一张本机RTX5090教师阶段累计≤12GPU小时，新增磁盘≤50GiB。不授权学生训练／弱光扩展／其他任务／实机或历史作业恢复。

## 实际资产与判断

官方task6 HDF5已下载到`outputs/rawvla-teacher-20261020/assets/put_the_cream_cheese_in_the_bowl_demo.hdf5`，535715512bytes；SHA256 `ee58a7d72c143a71bf1448674e70940ac7781c1d1bb10a1b0c7a5c69b26abbef`与官方HF LFS元数据一致。独立读取确认50个demo、5349帧、真实7维动作、79维物理状态和每示范model XML；所有数值有限，末帧原始done=1。这是原始示范资产，不等于本轮策略成功。

HDF5的9维robot_states实际为[gripper2,eef_xyz3,quat_xyzw4]；策略需要[eef_xyz3,axis_angle3,gripper2]8维。Sol逐帧独立核对全部5349帧与obs/ee_pos、obs/ee_ori、obs/gripper_states：位置／夹爪误差0，轴角最大4.44e-16。必须显式转换，原9维另保留，不能直接截取／填充。

实际作者LIBERO子模块锁定`5a2d7a27d557089747032d7a3f599568e1796844`。1117个文件全部按Git blob身份核验：1110个与本地既有资产一致，经只读复制复用；7个小文件从官方pinned GitHub下载。旧项目不修改；完整源在本轮忽略目录。慢速完整archive下载由Sol停止自己的进程并保留partial，转为逐blob核验恢复；没有停止其他作业。

明确的时序工程修正：作者回放器会在生成历史之前删noop帧，不能把删后的相邻索引声称为相邻物理帧。本轮保留所有源camera帧，按作者原noop规则仅标记可用动作anchor；未来action chunk取连续物理步。保留source_frame_indices及原频率元数据，不伪造测量时间。这个修正保持50同源示范及预算，需单独披露，不声称逐字复现全部数据采样细节。状态条件渲染的done由原示范记录提供，不能称为本轮闭环成功。

## 工程与责任

Sol负责路线／状态语义／预算／真实资产与独立验收。主要数据、回放、训练／恢复及验证入口由本机DeepSeek V41-Flash Max分轮实现；保留task、raw events、session选择、stderr、退出码、diff和独立review。运行根：`outputs/rawvla-teacher-20261020/`。

13:31 执行快照：50/50示范状态条件回放完成，全部5349帧、100对NPZ身份／动作／canonical state逐项核验；这不是策略闭环。审计manifest为`outputs/rawvla-teacher-20261020/raw/sources-audited.json`，SHA256 `c84966687c649ca6782c345f53a9a16f907b048184cc583d60067c1a58e8ed23`。30个新初态在训练之前冻结（10验证／20测试），不复用历史见证初态。相机全帧时序保留，训练40／验证5／测试5按源示范隔离。

真实同源后端813/813参数加载成功；双相机burst6／rec6前反向通路可运行。GPU驻留激活后续窗口前反向0.1717秒、峰值12811280384bytes，只是工程资源探测。验证实际部署路径时进一步发现并在任何优化更新前修复：window误选仿真79维state而非机器人8维观测；raw-only窗口误读clean RGB；微批累积动作归约缺少32维均值；训练与部署quantile clipping规则不可混用。作者训练数据端有clip，原部署客户端不clip，各按原始规则执行。此前错误state的零更新结果仅保留诊断，不作为策略证据。

修复后的真实102窗口验证完成：首步／前5步／完整10步真实7维MAE分别0.04609108／0.04319213／0.04228370；Sol从原始NPZ另行归约一致（最大2.1e-9）。其中前5步有效动作505、完整10步985。当前仅有离线评分与工程证据，控制能力未知。100更新试跑已按获准配方启动，完成前不计作学习结果；累计上限仍2000，先审查100更新真实变化和冻结稳定性，再继续。运行配置为`configs/experiments/rawvla_teacher_execution_v1.json`，原提案保持历史状态。

## 阶段终验

取得可用RAW教师：新20共同测试初态，Default20/20、教师20/20，开发筛查通过。选定作者架构初始化0；实际训练155步／1240anchors，不宣称这些更新带来收益。验证完整10场为9成功＋1次Sol错误警报引起的人为中断；不删或重跑。实际配置False，所以padding疑点已被真实102窗口逐位等价排除。模型仍原GPU执行，渲染改OSMesa；不宣称历史EGL或论文复现。详见[终验报告](raw_stem_siglip_20261020/teacher-independent-review.md)与[模型注册表](raw_stem_siglip_20261020/teacher-control-acceptance.json)。学生0更新／0闭环，正式学生训练未授权。

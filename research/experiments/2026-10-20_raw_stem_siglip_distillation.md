# RAW stem＋继承 SigLIP＋教师蒸馏：工程实施与预检

2026-10-20，主checkout `main`；沿用采用决议[0016](../../docs/decisions/0016-raw-stem-siglip-distillation.md)。**新优化更新0，新仿真／实机闭环0，控制未恢复。**

用户执行中澄清：主教师须接收RAW，采用作者已训练RawVLA前端＋冻结VLA；历史Default基线只作诊断。教师与学生不固定单帧，可从首轮依据效果、稳定性和实测成本选择因果burst／递归。现有每帧stem是基线组件，时序训练端到端未验收，不将单帧写成最终策略约束。

本机Harness V41-Flash Max实际负责主要编码，Sol分轮审查真实代码和原始输出并返工。新的配对loader、RAW/RGB/Bayer stem、作者teacher adapter、蒸馏／原生动作目标、受授权门控的训练循环、纯artifact推理和配对评估入口已落入src/owac。历史同一π0.5后端已完整恢复，文件SHA与旧成功基线一致；实际权重下的合成接口、梯度、原生10步采样和修复后artifact重载得到有限验证。

仍缺已训练作者神经ISP和50条真实task6配对专家示范。真实专家阶段零、真实成本画像、教师／学生学习和控制均未验收。68个新针对性测试通过；既有回归39通过2跳过，跳过不算真实策略通过；宿主make check通过。

- [Sol独立验收、失败与逐项边界](raw_stem_siglip_20261020/independent-review.md)
- [实际命令与未来启动门槛](raw_stem_siglip_20261020/RUNBOOK.md)
- [新配置](../../configs/experiments/raw_stem_siglip_distillation_v1.json)
- [真实后端恢复](raw_stem_siglip_20261020/backend-recovery.json)、[实际权重接口](raw_stem_siglip_20261020/real-checkpoint-interface.json)、[合成数值桥接](raw_stem_siglip_20261020/real-policy-synthetic-parity.json)、[artifact重载](raw_stem_siglip_20261020/real-artifact-reload.json)
- [Harness分轮账本](raw_stem_siglip_20261020/harness-ledger.json)、[资源画像与不可测项](raw_stem_siglip_20261020/resources.json)、[环境](raw_stem_siglip_20261020/environment.json)
- [官方模型／权重发布核查](raw_stem_siglip_20261020/teacher-release-audit.md)、[首次资产快照](raw_stem_siglip_20261020/asset-inventory.json)

固定诊断设计仍为Normal libero_goal/task6、50源示范40/5/5、seed101、有效batch16、每臂阶段累计最多5000更新；主要token目标FP32 masked MSE＋.1cos，stem lr1e-4，首轮视觉／语言／动作后端冻结。动作采用原生flow目标；验证全去噪前5步真实7维MAE选型，早步打破平分。10horizon／10flow／5replan各自保留。GPU小时和训练磁盘上限待真实画像，未以合成输入伪造预算。N01/N02/N04继续暂停。

# 双卡真实batch8及蒸馏代码清理

用户要求：全局batch16不变，双卡各8样本真正并行，清除无意义hash、重复检查、审计门槛；监控使用用户提供凭据对应的SwanLab私密OWAC项目。

## 已运行结果

原串行训练在858更新/13728 anchors完整保存，native0退出。直接从完整状态恢复859，首批第一个anchor为demo32/endpoint70，与858 sampler下一个位置相同。优化器、scheduler、RNG、采样顺序及剩余预算保留。无新增研究训练臂，无闭环。

新路径真batch8每rank，global16，accumulation1；因果6帧和6 endpoint展开保留，各样本prefix按当前权重独立烧入，当前时间步合并处理，SigLIP/原投影有批维。整视觉头428579952参数更新，教师与动作后端不进训练优化器。FP32参数/Adam与原BF16 autocast不变。

实际持续更新859起，61次更新时记录：去前5次后host每步中位0.4513秒、前反向0.3117秒；旧hash-free serial约2.04秒。25秒两卡利用率均值84.96%/84.44%，范围71–100%/68–100%；峰值allocated约17.71GB，reserved约18.92/18.42GB。该采样说明运行吞吐，不证明学习收敛或控制。

## 删除/降频

已部署：删除strictFP32数值策略和benchmark阻断模块，逐步全梯度/参数missing/finite扫描、first-update参数克隆/审计、preflight梯度层级审计；RAW calibration/TemporalPacket全tensor有限值及范围扫描移除。GPU gather与status/输出目录递归扫描降到每10步，真实loss/update日志每步保留。预算停止和完整checkpoint保留。冷路径hash/export/teacher-cache残留继续清理，完成项另记。

## SwanLab

https://swanlab.cn/@Jeong/OWAC/runs/owac-fullvisual-shared-stage

私密项目OWAC，SDK0.10.2在独立CPU环境；独立进程读真实updates/validation/status，不加入DDP热路径。历史858训练步及8个验证点已回填并继续更新。SDK默认git、requirements、runtime等探测及console捕获关闭；不上传RAW、权重或缓存。凭据不入仓库。

## 门槛测试历史

早期GPU零更新测试保留原始失败：首次teacher mask维度错误已修；BF16 batch8损失/输出基本一致但梯度相对L2约3.95%，未通过Sol自加2%门槛；FP32和strictFP32又分别因6.72e-5/1.55e-5梯度相对差未过1e-5。用户随后明确取消这些冗余阻断，已停止测试和对应代码。不能将失败改记通过；也不据小数差异阻塞数学等价的实际批处理。CPU全梯度/掩码/变长prefix及真实DDP工程测试通过。所有GPU测试费用计入同一总账，无优化更新。

## 资源与边界

启动本次连续训练前远端已计6883.657775392秒，剩余133957.60541859124 aggregate GPU秒，外部双卡wall guard负责扣费；总230400秒上限不重置。新阶段更新上限3116，含先前1884更新等价总5000；当前仍在持续训练，不能把“运行中”写成完成/充分收敛。原始收据位于outputs/raw-distillation-full-vision-20261022/。已删除的自动化不重建。

## 最终交付

本阶段已完成3116真实全视觉头更新/49856anchors，native0退出，最终完整检查点shared-stage/checkpoints/checkpoint_final_003116，teacher_backend_optimizer_members=false，world_size2。batch8连续阶段859–3116共2258更新，无丢失/重算；全程batch8每步中位0.45355秒。达到原更新上限而非数值门槛停止。

最终训练token总损失0.07364796；初始验证4.004474，最低验证1.00053068（1000步，保留best/model.pt），最终验证1.00837483（46 windows/23552有效tokens/2 sources）。训练与验证差距明显，未执行测试集/动作诊断/闭环，不能宣称控制恢复或充分泛化。

原GPU总账包含此前阶段：98770.62497 aggregate GPU秒；远端本阶段累计保守计9211.88817秒，仍余131629.37503秒，但原累计5000更新等价/80000anchors上限已用尽。stage输出9694192534字节，低于15396584412上限。未扩大预算。

冷路径清理已完成并同步服务器：raw_stem/temporal student、两套训练器、teacher_data、temporal/shared teacher cache、paired loader、普通及temporal评估入口、CLI导出/重载中的实际hash调用和比较已删除，连无调用SHA helper定义移除。RAW/状态/patch有限值与范围全扫描、teacher/frontend全参数指纹、冻结快照审计和重复准入链已删；保留真实数学、相机/时间对应、reset、数据分区和完整恢复。旧记录中的字段兼容读取而不验算，缓存使用明确短名称/示范时刻；未删除其他N01/N02/N04历史代码。过时benchmark源文件也从服务器删除。

Sol独立最终组合功能测试123通过（无skip），make check通过；测试只验证本次工程功能，不是能力/新训练准入关卡。独立核对真实代码、实际更新与原始退出日志，未追加GPU审计。主要编码来自原Web DeepSeek V41-Flash Max会话；派工及原始事件在本轮harness目录。SwanLab已上传41119条记录、训练3116步和18个验证点，独立CPU进程native0结束。

冷路径最终同步在训练正常结束后，不重启训练、不重算教师目标；最终源码与必要测试包outputs/raw-distillation-full-vision-20261022/batch8-cleanup-source.tar.gz，无hash启动门禁。

## Git同步收尾

用户要求提交全部修改并使本地/服务器一致。同步前两端同HEAD；服务器两处额外差异是在旧continuation/temporal CLI重新调用audit_full/audit_termination及frozen_audit_interval，按用户最新移除审计指令采用本地清理版本。服务器原始差异已保存在忽略目录outputs/raw-distillation-full-vision-20261022/remote-before-git-sync.tar.gz；训练产物、缓存、日志及私密凭据不清除、不加入Git。

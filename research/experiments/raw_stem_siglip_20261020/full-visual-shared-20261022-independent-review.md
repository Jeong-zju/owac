# 共享 RAW、远端重新生成教师目标、整视觉头蒸馏：独立验收

2026-10-22，负责人 Sol。当前状态以运行收据为准；此记录不表示收敛或控制恢复。

用户最新要求覆盖早先旧50条迁移安排：从共享盘直接读取 RAW，在远端生成匹配的完整 RAW 教师投影后 tokens，重新训练整个学生视觉头。不使用旧学生1872步，也不使用旧50条教师缓存；旧中断传输产物保留且不进入 active shared-stage。

## 数据与教师

- 远端：101.71.76.104:27786，2×RTX5090；运行根 `/root/owac/repo/outputs/raw-distillation-full-vision-20261022/shared-stage`。
- 共享只读来源 `/root/autodl-pub/datasets/rawvla`；LIBERO RGB8 逆处理 pseudo-RAW3，不是原生 Bayer。仅 cream-cheese-in-bowl task6 Normal，共12源示范，固定seed101源分组8/2/2。118训练候选anchors、46验证窗口；test不生成目标、不进入模型选择。
- 新manifest SHA af3d8c5d077cf8283dd7628fe09ea15b07149abed87dfb19863e064e276e1797。
- 新缓存 identity ac8b8dcb399e898544efa2a31be4fb9169f8e9658d0a80da1afc2893595fb8f2；204行（158 train含burn-in、46 validation、0 test），每行2相机×256tokens×2048维。
- Sol实际逐行读取校验全部204行，数值有限、真实RAW SHA一致、来源处于共享根；194条相邻state/theta链一致、10个episode reset通过。缓存不是配对RGB策略的输出：完整教师实际从6帧因果RAW及相机递归状态经过Neural ISP、原RGB预处理、当前策略SigLIP及投影生成。
- 教师为本地作者架构init0前端，非作者发布的训练后前端；继承实际LIBERO π0.5策略，不能称论文权重复现。原教师历史20/20仅开发筛查，不是本次学生控制证据。
- 公开NPZ没有可验证原HDF5文件SHA，明确标记unknown；legacy字段仅aggregate NPZ identity兼容摘要。配对RGB不读取、不登记、不声称验证。

## 结构和更新范围

学生：只读共享RAW → 固定校准/尺寸映射 → 新初始化局部卷积stem → 6帧因果聚合/递归gate → 同策略位置编码 → 27层SigLIP → 原post-norm → 原线性多模态投影 → 教师目标所在的tokens。部署不生成RGB，不需要教师。

可训练总数428,579,952：stem14,453,632；gate1,152；位置编码294,912；blocks411,466,608；norm2,304；投影2,361,344。教师全部冻结；语言/动作后端冻结且训练进程不加载。437个继承的非patch视觉张量与真实checkpoint BF16→FP32值一致，CPU实际构建本地及远端均验证。stem/gate seed101重新初始化，优化器从零开始。

FP32参数/Adam状态，BF16 autocast，FP32 masked MSE +0.1 cosine；全视觉统一AdamW lr1e-5、warmup100、wd0.01、clip1。双卡DDP全局batch16、各8anchors，按有效token分母累积，no_sync覆盖forward/backward。CPU双rank梯度等价及不等mask测试通过。更新日志经独立审查修为全局16anchors的token加权均值，而非最后一个microbatch；记录实际使用LR和下一步LR。

## 实际检查与负结果

教师准备首次在路径映射slots序列化处退出，0目标、0更新；原失败日志/退出码保留，计费10.002852454 aggregate GPU秒。DeepSeek同一V41-Flash Max Web会话修复，Sol检查源码并验证真实映射加载后才重新准备。第二次准备native exit0，实际204目标；外部双卡保守墙时188.017113088秒。

真实双卡学生预检native exit0，0优化更新；全部参数有有限梯度，stem/gate/位置/27层blocks/norm/投影有效；teacher/backend不在优化器。46窗口验证total4.0044735574、MSE3.9982035627、cosine0.06269994775；只表示新初始化误差。预检单rank峰值allocated约5.09GiB；不等于真实训练完整峰值。外部双卡保守墙时110.01098023秒。

Sol在真实首次更新前发现并返工：预检config_digest与读取字段不一致；预检累计成本遗漏教师制备；监控共享来源标记硬编码false。收据原件保留，修复后按外部双卡墙时保守记账，不重置预算。33项相关测试、make check实际通过（原始Harness事件4120）；不以测试代替GPU更新/重载验收。

## 预算和运行证据

原总预算不扩：此前89558.736806秒，远端最多140841.263194 aggregate GPU秒；最多3116次新更新/49856训练anchor呈现；新输出15396584412字节。两卡墙时合计计费，含准备、失败、预检、重载、训练。外部拥有者包装器核验冻结源码并记录PID/startticks/PGID/native exit，独立剩余预算守卫预留保存时间，不自动重试。独立外部ledger为保守总账，内部计时不替代总账。

当前已完成新目标与零更新预检；真实首步/重载/长跑结果将在下方追加，以原始收据和权重为准。闭环0；控制恢复未证明。

派工原始记录：`outputs/raw-distillation-full-vision-20261022/harness/`；同一Web session-8f2a5da4-8e59-4413-a719-3ff4c453e57c，实际deepseek-official/deepseek-flash/max。主要实现来自真实Harness任务，Sol负责接口、独立代码复核及远端验收。

## 真实首步独立验收（15:06）

首次尝试在0更新初态保存时遇到远端Torch2.12的UntypedStorage object-pickle异常；0更新、无checkpoint，外部双卡98.009553284秒计入预算。Sol用远端双rank CPU Gloo独立复现同一traceback；原tensor对象collective失败，primitive JSON gather/restore两rank通过。DeepSeek修复为primitive RNG状态，并修复rank1只存一个CUDA RNG却按全设备索引读取的问题；36项针对性测试及make check通过。此修复不改变训练数值或初始化。

修复后首步native exit0。Sol直接读取step0与step1权重，442个状态张量全部FP32、有限且实际改变，共428,579,952参数；stem/gate/位置/第0、13、26层/norm/投影均实际更新。442个Adam状态step均为1，scheduler last_epoch=1；两rank RNG分别绑定设备0/1；sampler/RNG/优化器完整保存。模型SHA15d91a396ec932fc0017363cadc214d71e7786e525f39999e6b643270fffe6b3，优化器SHA63b1dc1e417b0ee7b923dcd3981df11cb50af234a7080bb454c9e0ba71ba1f8e。首步全局16anchors/8192有效tokens，训练total3.6008315086（与验证集分开），实际lr1e-7处于100步warmup；不是后续最大LR。两rank峰值allocated分别9,445,023,744及9,337,180,160字节；均实际参与训练。该首步含初始化/两次验证/完整保存外部双卡216.025634956秒，不能当稳定吞吐。

完整重载检查进行中；长跑须从新step1继续，不再初始化或重复消费这16个训练anchors。先前失败不计优化更新、成本照计。所有检查点、配置与失败原件保持可追溯。

## 完整重载与连续阶段启动验收

真实双卡严格重载native exit0，0额外更新，仍step1/16anchors；模型SHA未变，优化器完整恢复，未重新消费训练anchor。重载外部双卡72.008499302秒。连续阶段前累计远端保守扣账724.074633314秒，剩余140117.188560669 aggregate GPU秒。外部guard使用此真实剩余值；内部rank计时与外部保守总账不同，外部总账及guard优先，不用较低内部计时扩预算。

15:10连续阶段实际到4步/64anchors，非只启动命令。拥有者55498，torchrun55499/startticks17190994/PGID55499独立核对，存活且无native exit；rank0/1均实际后续更新，run-status无error。连续阶段日志与收据前缀student-continuous；完整配置shared-stage/authorized-config.json，冻结源码v6。两rank后续峰值allocated约11.56/11.54GB（十进制），完整资源以原始gpu_by_rank记录为准。新只读监控8879通过HTTP及真实/proc核验；旧8878保留历史，定时任务不恢复。

当前是已验证更新与重载后正在训练；不是训练完成。预算3116步上限不等于收敛；未运行学生闭环，控制恢复未证明。

## 用户最新哈希清理与吞吐修复

用户明确要求移除哈希校验，覆盖本报告前文历史哈希验收要求；旧证据不重写。已从55完整保存点恢复至56并连续运行，当前执行路径零哈希调用验收通过，稳定每步约2秒，详见[实测修复报告](throughput-nohash-20261022.md)。当前活动配置改为shared-stage/config-nohash.json，性能侧配置单独记录；无数据、损失、更新计数或总预算重置。

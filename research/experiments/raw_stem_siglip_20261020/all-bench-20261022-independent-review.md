# 全量 RawVLA-Bench 新蒸馏实验：Sol 独立复核

用户授权全量共享 RAW、新实验、整个学生视觉头；只保留新增磁盘380GiB上限，撤销GPU时间/显存/利用率等人为门槛。原实验完整保留。无新下载、动作损失、语言/动作后端训练、闭环或实机。

## 数据与配方

实际2403源轨迹：LIBERO1771（40任务）+RoboTwin632（13任务），总408075帧，5照明。按物理源轨迹seed101拆分1923训练/240验证/240测试，旧task6验证和测试源保留；每个源当前只有一个照明版本，未来同源变体应继承分区。训练56501个合格anchor（endpoint≥25），3532全局batch/遍，3遍10596更新，169536次呈现（每遍56501唯一anchor+11尾部补齐），不将重复呈现作为独立样本。验证8013窗口含启动阶段，另报≥25误差；测试不生成目标或参与选型。

同一作者架构init0 RAW前端+实际π0.5原视觉权重/原投影作为冻结教师；原始RGB patch stem保留。此教师不是作者发布训练权重，也没有全任务/照明控制验收。生成目标直接读取共享RAW，不读配对RGB；语言动作后端没有实例化。学生重新初始化stem/gate、继承相同策略视觉权重，整个428579952参数视觉头训练；不续用旧学生。损失仍MSE+0.1cosine，AdamW lr1e-5/weight_decay0.01/warmup100/gradient_clip1，BF16前向+FP32权重。

LIBERO两相机uint8/255；RoboTwin三相机线性float32不除255，全视野240×320→256方形映射，教师ISP与学生共享该映射，再各自224视野；不舍弃右相机。学生沿用相机token拼接注意力，位置表每相机重复，2/3相机分别组成全局16=2×8批次。6连续RAW帧，endpoint步长5，6endpoint训练展开及因果prefix不变。

## 实际实现与复核

主要代码由用户指定的同一DeepSeek Web会话session-8f2a5da4-8e59-4413-a719-3ff4c453e57c实现，实际模型deepseek-official/deepseek-flash/max；工作单和原始收据在outputs/raw-distillation-all-bench-20261022/harness及原会话。Sol逐段读实际实现，退回并修复了burst内误用stride5、CPU/GPU搬运、DDP绕过、目标stack API、双rank同时写checkpoint、best选型先更新后比较等接线错误，未以Harness总结代验收。Sol只直接修改少量启动/监控会话重载阻塞问题和研究记录。

新路径不调用旧3116更新/GPU预算门槛，不哈希、不做继承权重全张量重复比较或GPU准入benchmark。CPU RAW预取4线程/每rank256源工作缓存；mmap教师目标按源读取，历史留CPU，每次仅搬当前burst到GPU。FP32目标仅存tokens与元数据，不存RGB/动作/语言/递归状态快照，train+val tokens预计340.914GiB。完整源提交；部分源重启从头回放教师状态；测试目标0。

独立检查：新增全量接入8测试、共享batched/temporal24测试、SwanLab7测试，合计39项功能测试通过，make check通过。监控8880和新SwanLab运行已实际连通；额外修复CPU监控SDK重启后初始化/finish状态。工程功能测试不代表学生已经学习或控制恢复。

## 启动事实

教师目标实际从2026-10-22 17:51:32+08双卡运行，torchrun PID65046/PGID65046/startticks18168220，两个实际rank65059/65060。Sol核验原始/proc及实际FP32目标写入。首次启动器误用了不存在的venv torchrun可执行文件（没有GPU子进程）；错误原样保留attempts/launcher-path，修正为python -m torch.distributed.run后启动上述作业。无自动重试。

衔接进程PID65928/PGID65928/startticks18243804，只在上述教师原生退出0后启动一次学生，不自动重试失败。配置与命令在输出目录，训练按每250更新及每遍完整保存，保存Adam/scheduler/各rank RNG/epoch与批次游标，SIGUSR1在共同更新边界保存暂停。

当前事实：教师目标仍生成中，学生0更新；训练入口已复核并排队，尚无真实学生优化、重载或验证结果，不宣称算法或控制提升。后续原生收据/updates/validation/checkpoints为实际阶段事实。没有定时任务重建。

监控：http://127.0.0.1:8880/
SwanLab：https://swanlab.cn/@Jeong/OWAC/runs/owac-all-bench-20261022

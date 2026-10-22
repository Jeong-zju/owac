# 全量 RawVLA-Bench 新蒸馏实验：Sol 独立复核

用户授权全量共享 RAW、新实验、整个学生视觉头；只保留新增磁盘380GiB上限，撤销GPU时间/显存/利用率等人为门槛。原实验完整保留。无新下载、动作损失、语言/动作后端训练、闭环或实机。

## 数据与配方

实际2403源轨迹：LIBERO1771（40任务）+RoboTwin632（13任务），总408075帧，5照明。按物理源轨迹seed101拆分1923训练/240验证/240测试，旧task6验证和测试源保留；每个源当前只有一个照明版本，未来同源变体应继承分区。训练56501个合格anchor（endpoint≥25），3532全局batch/遍，3遍10596更新，169536次呈现（每遍56501唯一anchor+11尾部补齐），不将重复呈现作为独立样本。验证8013窗口含启动阶段，另报≥25误差；测试不生成目标或参与选型。

同一作者架构init0 RAW前端+实际π0.5原视觉权重/原投影作为冻结教师；原始RGB patch stem保留。此教师不是作者发布训练权重，也没有全任务/照明控制验收。生成目标直接读取共享RAW，不读配对RGB；语言动作后端没有实例化。学生重新初始化stem/gate、继承相同策略视觉权重，整个428579952参数视觉头训练；不续用旧学生。损失仍MSE+0.1cosine，AdamW lr1e-5/weight_decay0.01/warmup100/gradient_clip1，BF16前向+FP32权重。

LIBERO两相机uint8/255；RoboTwin三相机线性float32不除255，全视野240×320→256方形映射，教师ISP与学生共享该映射，再各自224视野；不舍弃右相机。学生沿用相机token拼接注意力，位置表每相机重复，2/3相机分别组成全局16=2×8批次。6连续RAW帧，endpoint步长5，6endpoint训练展开及因果prefix不变。

## 实际实现与复核

主要代码由用户指定的同一DeepSeek Web会话session-8f2a5da4-8e59-4413-a719-3ff4c453e57c实现，实际模型deepseek-official/deepseek-flash/max；工作单和原始收据在outputs/raw-distillation-all-bench-20261022/harness及原会话。Sol逐段读实际实现，退回并修复了burst内误用stride5、CPU/GPU搬运、DDP绕过、目标stack API、双rank同时写checkpoint、best选型先更新后比较等接线错误，未以Harness总结代验收。Sol只直接修改少量启动/监控会话重载阻塞问题和研究记录。

新路径不调用旧3116更新/GPU预算门槛，不哈希、不做继承权重全张量重复比较或GPU准入benchmark。CPU RAW预取4线程/每rank256源工作缓存；mmap教师目标按源读取，历史留CPU，每次仅搬当前burst到GPU。FP32目标仅存tokens与元数据，不存RGB/动作/语言/递归状态快照，train+val tokens预计340.914GiB。完整源提交；部分源重启从头回放教师状态；测试目标0。

独立检查：新增全量接入9测试、共享batched/temporal24测试、SwanLab7测试，合计40项功能测试通过，make check通过。监控8880和新SwanLab运行已实际连通；额外修复CPU监控SDK重启后初始化/finish状态。工程功能测试不代表学生已经学习或控制恢复。

## 启动事实

教师目标实际从2026-10-22 17:51:32+08双卡运行，torchrun PID65046/PGID65046/startticks18168220，两个实际rank65059/65060。Sol核验原始/proc及实际FP32目标写入。首次启动器误用了不存在的venv torchrun可执行文件（没有GPU子进程）；错误原样保留attempts/launcher-path，修正为python -m torch.distributed.run后启动上述作业。无自动重试。

衔接进程PID65928/PGID65928/startticks18243804，只在上述教师原生退出0后启动一次学生，不自动重试失败。配置与命令在输出目录，训练按每250更新及每遍完整保存，保存Adam/scheduler/各rank RNG/epoch与批次游标，SIGUSR1在共同更新边界保存暂停。

当前事实：教师目标仍生成中，学生0更新；训练入口已复核并排队，尚无真实学生优化、重载或验证结果，不宣称算法或控制提升。后续原生收据/updates/validation/checkpoints为实际阶段事实。没有定时任务重建。

监控：http://127.0.0.1:8880/
SwanLab：https://swanlab.cn/@Jeong/OWAC/runs/owac-all-bench-20261022

## 教师目标数值补充

Sol只读取已生成目标，在同一真实Normal task6 demo1、endpoint0、形状[2,256,2048]上比较旧/新教师路径：MSE0.00103437854，最大绝对差2.59603882（单分量峰值，不等同平均误差）。新独立视觉教师以FP32执行；旧完整策略按作者precision:bfloat16初始化，部分参数留FP32，并有原生autocast回退。权重、模块和目标公式沿用，浮点执行精度已单列披露；不能称与旧目标逐位一致，或将新旧实验差别全部归因于数据。此为读取现成结果的诊断，不是新GPU作业或数值准入阈值，不据此重跑缓存。

训练数据入口补充：每卡以4个CPU线程预取下一批8个完整样本，包括解码、归一化、缩放和历史窗口组装，按原采样顺序消费；不在worker中创建CUDA张量。验证直接复用训练token_distillation_loss以避免另一入口的全张量finite扫描，公式不变。40项相关功能测试和make check通过。

## 首次真实训练失败及修复

教师目标2163源/74129窗口全部完成，native0；耗时2475.400590658188秒（双GPU），不再生成教师目标。首个学生作业PID67855，从18:32:48到18:36:05，native1，wall197.02065682411194秒；日志记录至少10次全局更新/160呈现，MSE3.562329。最终未保存更新数未知，只有初始模型，没有优化器检查点，不声称从10步恢复。

原始cgroup证据：memory.max124554051584字节=116GiB，memory.peak124554067968，oom1/oom_kill1；rank0被SIGKILL，Torchrun随后终止rank1。宿主free-h的503GiB不能当作容器配额。首次数据工作缓存保留256源/卡，连同预取CPU窗口超过实际配额。GPU显存/时间的人为门槛没有触发。未修改提供商cgroup限制。

同一DeepSeek会话修复为每卡16源RAW工作缓存，并让配置实际传入Dataset；这是数据保留策略，不是增加内存停止阈值。global16/每卡8、所有源和完整因果历史、损失/参数范围不变。首次真实更新1即保存完整状态，此后每250更新/每遍继续保存。11项新入口CPU测试+24项共享路径+7项监控测试累计42项及make check通过。

失败日志、native exit、cgroup证据、初始模型、SwanLab及已记录更新全部保存在outputs/raw-distillation-all-bench-20261022/attempts/train-oom-20261022-183605；未重试教师。修复经Sol独立读diff及检查后，下一次学生作业从相同初始化/seed101、优化器和采样器0开始；另用SwanLab attempt2避免曲线混接。重新启动尚待实际日志确认，不把测试通过称为运行恢复。

## 只读RAW映射切换

修复后的训练已真实推进137更新，SIGUSR1在共同更新边界完整保存模型/Adam/scheduler/各rank RNG/采样epoch+cursor后native0暂停，未重置。读取器现支持ZIP_STORED NPZ内部.npy直接只读np.memmap；按实际NPY头定位payload，压缩成员保留正常np.load后备。既不解包复制RAW，也不增加缓存磁盘。小夹具验证存储成员、压缩后备及帧数值一致，未增加GPU微基准或准入门槛。恢复加载后释放CPUcheckpoint payload引用，避免两卡长期各留一份完整CPU权重/Adam副本。累计43相关功能测试及make check通过。下一作业从137完整状态续训，实际重载推进尚待日志确认。

## 当前持续运行验收（2026-10-22 19:00:54+08）

从137完整检查点实际重载后已推进至210更新/3360呈现，native torchrun PID69449/PGID69449，/proc startticks18565539与启动记录一致，尚无退出收据。执行源码e658cd61dab38d6a33deb28e13a23975bec82c29；全局batch16。记录批次210的MSE1.56097269、cos0.02681960、total1.56365466；不同批次不据此声称学习趋势或泛化改善。SwanLab实际同步至210，error:null；系统oom/oom_kill保持1/1，没有新增。最新137完整检查点5143552127字节，实际续训已经穿过137，不从0开始。

全量教师native0，实际2163训练+验证源/74129目标窗口，test目标0；与1923/240/240源划分一致。首次失败≥10已记录更新、最终未保存更新数未知，失败成本和原始产物完整保留。当前无验证结果、测试、动作诊断或闭环证据；不能宣称充分收敛、正常光控制恢复或弱光收益。

只读监控8880及SwanLab attempt2均已实连；普通作业监督进程仅观察本轮native exit并在失败时更新页面，不自动重试、不重建Codex定时任务。

## 文件句柄失败与恢复点

19:04:49，train-r3原生退出1，最后记录300更新。rank1实际OSError24 Too many open files，发生在TargetStore.read；目标mmap句柄一直按源保留，达到默认RLIMIT_NOFILE soft1024（hard65535）。OOM计数未增加，区别于第一次RAM故障。原生423.3115568161011秒及日志保留，不自动重试。

只读元数据检查确认最近完整checkpoint为global_step250/epoch0/step250、4000呈现，包含模型、442个参数的Adam状态、scheduler、2个rank的RNG及采样游标。已记录300之后的未保存步数未知，至少50个已记录更新需重算，不能称从300恢复。此元数据读取不加载权重数值、不做哈希或GPU数值门槛。

同一DeepSeek将目标读取改为每次只读映射、复制所需行、立即关闭；不再保留全部源的mmap句柄，不新增句柄停止阈值，复用OS文件页缓存。并发读取和已返回tensor生命周期的CPU回归通过，新增入口13测试+共享24+SwanLab8累计45项，make check通过。SwanLab亦修复native失败必须标crashed而非假成功。之后从250完整状态恢复，保留原始300步失败记录及其曲线，当前曲线仅沿250祖先重放，不接入废弃分支260～300的曲线。

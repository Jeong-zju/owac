# C单臂充分蒸馏：Sol独立验收与实时状态

最新状态（2026-10-21T13:15:25.208300+08:00）：**连续C蒸馏运行中**。实际累计56更新，本轮新增6次；上限累计5000／新阶段64 GPU小时／20GiB。第51步完整独立验收已通过，继承视觉和后端保持冻结。监控：http://127.0.0.1:8878/。以下保留按时间累积的工程过程及失败事实；本轮闭环0，控制恢复未证明。

2026-10-21。用户新授权“可以开始充分蒸馏，并给我一个简单的监控页面来实时查看进度和效果。”本记录持续追加真实证据；当前处于工程实施，尚未新增更新。

## 已冻结范围

从实际C50续训至累计最多5000更新／80000 anchors；新增4950更新／79200 anchors，上限64 GPU-wall小时／20GiB新输出。仅RAW stem＋gate 14,454,784参数。教师、继承SigLIP、位置／norm／原投影、π0.5语言／动作后端全冻结。固定train40／validation5／test5，seed101，effective batch16。

仅拟合教师最终投影后tokens，L=FP32 maskedMSE+.1cosine；按验证集全部102窗口的同一目标选型，同分取更早。动作MAE是诊断，不选型／提前终止／触发换方案。训练损失、验证token指标、离线动作、闭环各层分开。无B/D、视觉解冻、弱光、闭环或实机扩展。

## 独立资产和父状态审计

原始收据根：`outputs/raw-distillation-c-full-20261021-123313/`。

- Sol直接读取实际C50，SHA `631b6ea1e21bbe01e4e08b9da56432a0b9a703229a14a59463ac01a009052e1a`吻合；不是重训或从动作选型C0启动。
- `sol-parent-state-audit.json`：累计50／anchors800，5个优化器参数状态step均50；scheduler、sampler、python／numpy／torch／cuda RNG完整。各状态摘要已存；本次审计0更新。
- `sol-assets-audit.json`：真实8,522,039,320字节LIBERO权重、教师前端、tokenizer、统计量、源示范manifest全部重新计算SHA并吻合；cache存在。存在性与文件身份已核验，续训运行时可加载仍须实测。
- 教师仍是本地作者架构init0前端＋实际LIBERO策略；原共同20/20是开发筛查，非公开训练前端或本轮学习收益。数据为RGB8 inverse pseudoRAW3，非原生Bayer。

## 实际Harness派工

独立Web实例3081，保持人工3080不变。会话`session-8f2a5da4-8e59-4413-a719-3ff4c453e57c`，展示名“OWAC C充分蒸馏与实时监控 · Sol派工”。任务见runroot的`harness/round-01/task.md`；官方Web RPC创建／发prompt，真实模型请求projection确认deepseek-official／deepseek-flash／max，permissions danger-full-access。主要代码由该会话承担；Sol定边界、独立读代码／原始状态、运行复核。浏览器控制通道不可用，不能宣称已做可视化浏览器QA；官方Web会话历史能读取任务／工具／回复。

启动时遇旧credential layout及共享task-board锁：只为本轮私有配置使用version1空store＋coding角色环境，禁用本实例taskboard。未修改既有凭据、人工Web服务或全局权限。

## 当前验收项目

- 资产身份／父状态：通过。
- feature weighted reduction与selector：Sol运行14项CPU测试通过；不代表真实策略续训或控制通过。
- strict continuation migration／真实C50数值回放／51更新／恢复重载：待实现及独立检查。
- 监控HTTP/API／进程死亡及stale判定／页面：待实现及独立检查。
- 新阶段训练：尚未开始。
- 本轮闭环：not_run，0 episodes；控制能力恢复：未证明。


## 真实预检与首步结果（13:06）

- 首次prepare在优化器构造前因teacher_artifact_sha256字段映射错误失败。保留原始prepare.log与failure receipt；没有任何新增更新，保守计费114.335秒。修复严格字段映射，未放宽身份或1e-6容差。
- 第二次prepare真实native exit0，95.974秒；全5验证源／102窗口／52224有效tokens／106954752标量分母。MSE2.765593086972、cosine .047423013906、total2.770335388363，与历史C50三项abs_delta全部0。
- 第一次51 launcher因错误resolve虚拟环境Python符号链接，裸解释器缺numpy，实际nativeexit2／0新增更新；实际.2006秒已入账并保留state快照。Sol小修保留venv词法路径，真实runtime NumPy／Torch2.7.1导入通过。
- 第二次51真实native exit0，累计51／816 anchors。独立sol-phase51-audit.json全部15项通过：5个trainable均有有限非零梯度且真实值变化；437继承视觉张量逐值不变；完整冻结SHA仍a5cd67134bf730b58e5bf71e6c2b96a890308dda86b8970185e5d117506940f2。optimizer5组step51、scheduler51、sampler精确衔接父状态及真实batch。
- 51验证MSE2.746400003340、total2.751110732453，小幅改善不代表充分收敛。真实checkpoint SHA c1bf3ad38826b923d1e63cafc54b79167f5c9b29aaaa0bfdd72f1f18f347a02e。当前本轮实际只新增1优化更新，闭环0；未证明控制恢复。
- 继续前修复监控与真实supervisor字段接线，并核对phase2暂停后恢复最新checkpoint（不得退回51）及批准锚的保留。训练数值配方不变，当前GPU作业已退出；待修复验收后继续同一阶段。


## 连续阶段启动与跟进

- 实际C51原始权重、训练状态和原生退出0已独立核对，并经正式accept入口批准。批准锚硬链保留在lineage/accepted-C51.pt；不是另一个模型、不是重训。
- full-continuation-command.json启动同一阶段到累计5000，真实从C51恢复到52及后续；优化器/scheduler/sampler不重置。进程由独立supervisor和owned-job.py持有，终端断连不代表子进程终止；所有退出码由持有wait句柄的进程记录。
- 源版本仍main/9b7ad89961ea36c7f4e5530d278a513223c043ac及已有dirty diff。各轮源码清单与主要节点快照保留，最新为v6；v4后变化仅监控接线和活动进程退出码显示，训练/守护数学实现未改变。训练前后未覆盖其他人的修改。
- clean paused phase2恢复现选用实际最新owned checkpoint，独立fixture验证250→后续不会退回51；51授权锚不被训练retention删除。terminal error不自动重试，须独立审查和显式operator恢复；不无界重启。
- 监控HTTP路径限制/只读、部分JSONL尾、真实PID/复用/zombie/心跳失效、实际supervisor字段、跨阶段loading、live child exit=null已检查。HTML JS语法通过；浏览器控制不可用，未伪称截图视觉验收。监控服务是用户请求保留的常驻只读服务，未上传到外部站点。
- 原始测试：新增模块首轮64pass；旧接口/hub回归68pass；映射/预检debit修复45pass；最新supervisor＋monitor34pass；最终monitor12pass。各轮并非独立测试数总和，含重复。最终make check通过（ruff/1037文件format/compileall）。
- 主代码来自实际Harness Web V41-Flash Max派工，lead和两个互不相交worker的真实lastUsed projection均核验；Sol直接补丁仅小型接口/状态/launcher阻断，独立读取原始代码/权重/数据/日志，不以Harness总结替代验收。完整Web事件与模型选择记录在harness/round-01/。
- 后台跟进automation owac-c每20分钟在本线程核查；运行无异常保持安静，只在完成、失败、资源临界或需用户行动时通知，预算不扩展。

## 指标与当前边界

主训练/选型仅最终投影tokens的FP32 masked MSE+.1cosine。MSE归约按有效token×2048，cosine按有效token，保留幅值；验证固定5源/102窗口/5reset/52224有效tokens，test不参与。训练batch曲线与留出验证曲线分开。动作MAE只作诊断，不参与选择、早停、切换或本轮损失；当前新阶段动作诊断尚未运行，页面不能将父结果当当前checkpoint。

资源：supervisor记录真实跨phase累计wall，包含成功prepare、失败预检debit及51/后续作业；disk按本runroot真实文件、含Harness日志/收据、symlink排除、hardlink inode去重。更新日志提供本训练进程CUDA allocated/reserved和host RSS；这是每次更新结束时的采样，不冒称本阶段精确训练峰值。原50步资源画像是明确带入的前阶段证据，非本轮新测量。当前无压缩或速度提升结论。

本轮仍在运行，未宣布充分收敛。达到5000或资源上限后须审核学习曲线与验证结果，再在剩余同一预算内做只读离线动作和纯学生重载验收；预算不足写not_run，不加训练或闭环。

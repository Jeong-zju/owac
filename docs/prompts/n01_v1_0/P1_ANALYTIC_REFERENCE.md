# P1 可组合和可撤回的解析参考

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

本次只实现 P1。完整读取[共同契约](00_SHARED_CONTRACT.md)、P0 实际报告、配置和类型，以及方案第 4、5.1、5.2、5.5、6.1 节。目标例子：两个有重叠来源的区域合并不能增加虚假信息；其中一帧属于旧遮挡物时，删除它后必须等于只用剩余 RAW 重新求解。

## 1. 文件和 API

在 `src/owac/frameworks/n01_full/` 新增：

- `ledger.py`：`MeasurementLedger.observe(frame)`、`get_sources(ids)`、部分 episode reset；原始数据用按帧 tensor + 紧凑索引存储，避免给百万像素创建 Python 对象。
- `domains.py`：`make_domain / merge_domains / split_domain / retract_sources`；返回新版本，保持来源角色与坐标版本。
- `statistics.py`：`fit_structure(domain, ledger, correspondence, noise_model)`；常数/二维仿射 GLS，返回秩、估计与不确定性种类。
- `replay.py`：`full_replay(events)`、`fixed_decision_replay`、`counterfactual_replay`；先完整求解，后续阶段共享。
- `reference.py`：小问题的 NumPy FP64 独立稠密实现，不能调用生产 `statistics.py` 来充当 oracle。
- `tests/encoders/n01_full/test_ledger.py / test_statistics.py / test_replay.py`；`evaluation/n01_full.py` 增加 `analytic` 子命令。

先写 reference、再写生产实现。保留旧 `frameworks/n01/sequence1d.py` 入口与测试，新实现补一个同语义的一维适配例子。

## 2. 数学实现固定如下

每相位分别拟合 `phi=[1]` 或 `phi=[1,x,y]`。x/y 用域参考坐标中心化并按显式尺度归一化，保存变换，读出比较时回到相同查询坐标和单位。不同相位不能先求均值。固定设计下计算：

```text
L = cholesky(Sigma)
A = solve_triangular(L, Phi)
b = solve_triangular(L, y)
theta = lstsq(A, b)                      # 无先验支路
Lambda = A.T @ A
eta = A.T @ b
```

满秩时 covariance 用成熟 solve/Cholesky 得到 `Lambda^-1`；不用手写逆矩阵迭代。使用先验时另外加明确的 `Lambda_prior/eta_prior`，先验只计一次。秩不足用 SVD/QR 给出可辨识子空间；伪逆的零值不能当作零方差的确定方向，`identifiable_mask` 标清，必要时给 unavailable covariance。

相关因子以原观测协方差块 Sigma 保存。删掉来源后取 **Sigma 的剩余子块并重新求解**；不能截取旧 precision。插值 `z=A_interp y` 记录 `A_interp Sigma A_interp.T`；重复或重叠导致奇异时保留独立原始观测作为因子，或在显式可辨识子空间计算。派生积分节点可作后续内容描述，不增加独立精度项。

未知相关性时单列固定权重方差保守界 `(sum(abs(a_i)*sigma_i))^2`。通过数值测试展示它的保守性；不能与已知 Sigma 的精确 covariance 混用标签。

## 3. 来源操作的实际语义

observe 的重复判断以 SourceKey 为准。同一包重读不推进测量计数/精度/预算；新的曝光即使数值完全相同也登记。重新校准同一 RAW 只增加派生版本。

merge 对固定 basis/mapping/noise/prior 版本取来源并集，规范排序再重建。P1 由测试显式给出合法域；自主判断相容性留 P3。split 由子域成员 ID 重新拟合，不给子域拷贝父均值。时间撤回按源帧/采样 ID 删除当前域成员，不删磁盘原始记录，也不影响仍合法引用它的其他结构。

每次变更更新 support_version，派生节点记录完整父项。P1 无神经缓存，但加入一个两层非线性玩具算子作为 **依赖撤回性质测试**，检查完整后继闭包重算；它不是正式内容骨干。

## 4. 必做的数值和行为验证

1. `y=[1,3], variance=[1,1]`：均值 2、方差 1/2；重放第一个 ID 后仍相同；改为真正第三个新 ID 则按三观测计算。
2. `Sigma=[[1,0.5],[0.5,1]], Phi=ones`：均值 2、估计方差 0.75；撤回第二观测后方差为 1。直接截 precision 得 0.75 的错误版本必须被测试识别。
3. 已知仿射场、分数坐标、同相位值和异方差随机小问题：生产结果与独立 whitened least-squares 比較；记录条件数。
4. 重叠 merge、不同顺序、split 再 merge、retract 再恢复、相同事件不同 batch 分块；只在版本固定时检查集合代数。
5. missing/bad/saturated 不作为零值观测；全无测量返回无效 mask；无先验秩亏不获得虚假 confidence。
6. 两个 camera 同 frame ID、两个 episode 同 frame ID 不冲突；reset batch 第一个成员不改变第二个成员。
7. 内容替换撤回和删除来源全重放分别做；source/value/mapping/参数版本变化触发正确重算。

FP64 小问题初始容差 `atol=1e-10, rtol=1e-8`；FP32 良态问题 `atol=1e-5, rtol=1e-4`。病态问题单报数值条件，不靠扩大所有测试容差处理。条件数与随机种子固定在收据。

## 5. 交付

补齐 T1/T2/T5 的假设、推导、代码入口和反例；明确自适应选域/估计 flow 后不自动继承固定设计 GLS 协方差。运行新测试、旧一维/支持/状态相关回归与 `make check`。在服务器做一组真实数值流读取与唯一来源计数检查，不需要本阶段长训练。

P1 末执行 [EARLY_POLICY_PREFLIGHT](EARLY_POLICY_PREFLIGHT.md) 的 `mode=contract`，仅验证已有桥的类型/shape/mask，尚无完整神经 N01 时使用真实现有基线并标明。保存 `P1_report.md`、数值明细和回放参考路径，然后停在交接点。

参考：[GLS/Cholesky PyTorch 官方接口](https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky.html)、[lstsq](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html)、[iSAM2 原文与实现导航](02_REFERENCES.md)。

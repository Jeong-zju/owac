# N01 v1.2 P3 独立验收与修补记录

> **2026-10-17 后续**：[代码补全与无更新验收](P3_code_completion.md) 已接入完整主链并补齐本页列出的代码缺口。训练按用户新指令持续暂停；原始数据内容证明及训练/校准/闭环效果不因此成立。本页的诊断更新次数属于上一轮，不能计入本轮0更新验收。

日期：2026-10-16（Asia/Shanghai）。对象：`lzh/n01-evidence-integration`、HEAD `d3dec908d78d0a86a761abc985b83113d836c8cf` 及 DeepSeek 未提交工作树。主服务器 `154.59.156.21:34302`。本记录优先于历史 `P3_HANDOFF.md`、`P3_report.md` 的完成/留出口径，不改写历史原始记录。

**结论：P3 未通过独立验收。不能把剩余工作解释为只等训练结束。** 已修补可复现的状态/训练/划分缺陷，并停止主体队列，保留所有现有输出与 checkpoint。完整运动对应、真正的支持估计、匹配的训练/服务观测与完整训练方案仍未完成。没有进入 P4，没有启动 Base 臂或额外闭环。

## 1. 原要求与实际系统

原要求为完整 RAW 时空编码器→非图像前缀→π0.5 行动，并在 RawVLA-Bench 两平台训练、做开发闭环和同步可视化。用户先前已将 Base 臂推迟至 P4，本次沿用该决定，不要求重新启动八臂矩阵。

实际已训练的是 DBSR 内容编码器与 512→2048 bridge；`_v12_bridge()` 始终冻结 π0.5，训练器没有后续解冻阶段。已有 RAW/RGB 输入域修复、真实数据加载和动作损失证据仍保留。以下缺口不能用测试数量或训练步数补成完整模型：

| 原要求 | 当前生产路径 | 本次结论 |
| --- | --- | --- |
| 成熟运动对应参与融合与支持 | 工厂不设置 `motion_provider`；训练/服务使用 `identity_pending_p2` | 实质未实现项；不能自行称为获准的“范围限定” |
| 局部常量/仿射 GLS、空间候选、多尺度支持 | `region_residual()` 是逐像素绝对差的区域均值，除以参考图局部细节 RMS；没有 GLS/仿射拟合，`affine` 未参与计算，多尺度候选未进入生产判决 | 已纠正文档，保留其“代理残差”身份；没有实现完整估计器或完成物理噪声校准 |
| 训练/服务对应相同观测协议 | 训练 `--crop 128` 调用 `_crop_batch()` 截左上角；闭环把整幅画面 resize 至 128 | 同尺寸不等于同视野；当前闭环不能解释为训练模型的匹配性能验证 |
| 编码器/bridge 预热后按相同协议适配后端 | 训练器只有全程冻结后端这一阶段 | 需明确并实现正式训练方案，不能把当前 20k 预算直接称为充分训练 |
| 留出开发闭环 | 用照明 trial 编号连接 demonstration 编号 | 筛选身份错误；“留出成立”撤回 |

旧 P2 正式运动 checkpoint 未在新机恢复；本次只读重试旧机 `107.209.104.125:44327`，返回 `Connection refused`。未用随机权重、smoke 权重或 RAW 模型冒充有效 RGB 对应模型。即使恢复 P2 RAW 权重，RGB 臂仍需在自身 ISP 输入域建立对应，不得偷用 RAW 中间量；支持残差也必须使用实际对应后的合法观测。仅把 provider 挂上并不等于这些要求已满足。

## 2. 新增反例与修补

| 缺陷 | 独立证据 | 修补 |
| --- | --- | --- |
| 断点恢复重复读开头数据 | 将真实 AdamW 的四步连续训练与二步后恢复比较；原训练器在第 7 个 microbatch 出现采样分叉，随后参数不同 | 保存累计消费样本数，恢复 epoch/cursor；保存 Python/NumPy/CPU/CUDA RNG；验证数据/配置/累积与优化器契约 |
| 验证改写训练随机流且不切 eval | 原 `_validate()` 每样本 `torch.manual_seed()` 后不恢复，且不调用 eval | 验证隔离全部 RNG，切 eval，正常/异常都恢复原模式；有/无验证的恢复训练都与连续训练一致 |
| 重复帧制造新支持证据 | 原代码三次重发同一 `frame_id` 后 BOCPD 后验、age 改变；新增反例失败 | 只有真实新测量更新检测器及 age/defer；逐 batch member 移位，缺少合法观测的区域不更新后验 |
| episode reset 复制先验质量 | 原 reset 后 `logsumexp(log_message)=2.3979`，不是 0 | 全 reset 使用单个新先验；部分 reset 额外槽为 `-inf`，保留其他成员；稠密后验重建避免零概率占位覆盖真实假设 |
| 无效参考也能建立有效支持 | 把参考帧 mask 全清，原残差 validity 仍全真；新增反例失败 | 残差使用历史与参考 validity 的交集，尺度使用参考自身 validity |
| 留出筛选用错身份 | 真实日志 `episode=17 init_state=3`；反例中原函数保留了应排除的 trial 17 | 用 `init_state_index` 连接缓存 `demo_index`；缺字段拒收，不猜测；同一 demo 跨 split 冲突报错；计划注明内容身份尚未核验 |
| 队列“预检”在数天后才执行、失败最终 exit 0 | 读实际生产 shell；旧脚本在每个 item 启动前才检该项，末尾 printf 成功覆盖失败退出码 | 所有项在首个训练前预检；单项失败仍交接下一平台，最终 exit 1；持久审计 hold 阻止启动 |
| 不完整诊断被当主体训练 | 旧正式命令默认允许 identity 对应和左上角裁块 | 正式 train 拒绝这些配置；`--diagnostic` 必须显式传入，回执和恢复契约标记诊断身份 |

前三条支持反例在原代码上 **3/3 失败**；断点训练反例也在从远端保存的原训练器上失败。不是先写“通过”再解释。GPU 夹具使用真实 CUDA 随机流、AdamW 与 checkpoint IO，但替换昂贵的后端/数据 IO；它证明训练器续跑语义，不代表 VLA 能力。真实两平台通路另验，见下节。

旧 checkpoint 缺失 CUDA RNG 和采样位置，无法追溯补全。本次没有伪造迁移；新训练器明确拒绝把旧格式称为精确恢复。已有旧权重仍保留，可用于有明确身份的历史诊断，不能在修后机制下继续同一主体曲线。

## 3. 0/96 的身份重核

作者 `rawvla_bench/manifest.py` 定义 `episode_idx = base_seed_id * len(LIGHTING_DOMAINS) + domain_idx`，`init_state_index = base_seed_id`；实际评测读取 `initial_states[init_state_index]`。照明 trial 编号不能作为 demonstration 编号。

在原 `devloop-heldout/worker0-identities.json` 的 100 条上重算：

| 集合 | 原错误连接 | 改用真实初态字段连接缓存元数据 |
| --- | --- | --- |
| 选中 100 条 | test 36 / calibration 26 / validation 38 | train **68** / test 5 / calibration 7 / validation 9 / unknown 11 |
| 完成 96 条 | 原报告“留出成立” | train **65** / test 5 / calibration 7 / validation 9 / unknown 10 |

右列仍有一项待证假设：缓存 `demo_index` 与 benchmark 初态顺序的内容身份相同。新机 `source/benchmark_data/libero/LIBERO-datasets/` 是空目录，HDF5 文件数 0，本次没有拿不存在的原始 demonstration 做“内容检查通过”。因此 **65 是正确字段连接后的元数据计数，不是内容比对确认的训练重叠数**；不能把剩余 31 条自动升级成已验证留出结果。

可以保留的事实是：选中 100、完成 96、成功 0、4 条未完成。这是早期、不完整模型的开发诊断，既不是留出能力成绩，也不是机制增益结论。原 §1.46 的 69/100 和 §1.48 的“留出成立”均不再引用。日志在 `libero_10/task8/episode17` 首次动作后终止，无完整结束记录；终止原因仍未知，不把缺失 4 条记成策略失败，不追加 rollout 求满分母。

原始报告/plan/ledger 保留；新增审计注释与 `independent-audit-identity.json`，监控页说明同步降级。

## 4. 实际停止点与防止误续跑

2026-10-16 18:16:45 CST 的停止回执（实际停止发生在此前数分钟，回执时间不冒充最后一步的精确时间）：

| run | 最后日志步 | 磁盘 checkpoint 步 |
| --- | --- | --- |
| LIBERO raw_n01 | 2380 | **2000** |
| LIBERO rgb_n01 | 2500 | **2500** |

RAW 日志到 checkpoint 间 380 步不在可恢复快照中；没有声称保留未保存的内存状态。旧错误前缀、重启记录、全部主输出均未删除，未覆盖为新实验。

两个主体 supervisor 均 STOPPED，已取消 autostart；队列同时读取运行根的 `P3_AUDIT_HOLD.json`，退出码 42。停止/禁自动启动的原配置与原队列备份在 `receipts/pre-audit-*`。监控服务保留在线，状态为独立检查并说明暂停原因。少量诊断使用独立根 `v1-2-p3-independent-audit-20261016`，不向主体曲线或主体 checkpoint 写入。

预算算术也应更正：若按旧观测 **5.5 s/optimizer step**，20k 步为 30.6 小时，剩余约 17.5k 步为 26.7 小时，不是“四天”；RoboTwin 25.3 s/step × 20k 约 5.86 天。它们只是旧代码/旧观测协议的成本估计，不能作为修后模型的交付 ETA。

## 5. 验证与原始证据

本地 `make check` 通过，全仓 **847 passed / 6 skipped**；远端定向 **62 passed / 0 skipped**，其中包含 CUDA 精确续训测试及 ISP 实际交叉检查。LIBERO RAW/RGB 各完成 2 步后恢复到第 3 步，RoboTwin RAW/RGB 各 2 步，共 **10 次真实诊断 optimizer update + 10 次固定验证**，四份最终 checkpoint 的 schema、采样位置和 CUDA RNG 字段均核验。它们使用 `--diagnostic --crop 128 --accum 1`，仍是 identity 对应、π0.5 冻结，不能称为完整模型或主体训练。详细回执见相邻 `P3_independent_acceptance_receipt.json`。五份核心运行文件本地/远端 SHA-256 一致，8 份配置均可加载。原始本地日志放在被忽略的 `outputs/n01-v12-p3-audit/`，不把大日志/权重纳入 Git。

- 原训练器快照：`outputs/n01-v12-p3-audit/original/n01_v12.py`。
- 本地最终测试：`outputs/n01-v12-p3-audit/local-tests-final.log`。
- 远端定向测试：`outputs/n01-v12-p3-audit/remote-tests-final.log`。
- 可重放审计脚本：同目录 `n01-audit-{identity,real-training,final,freeze,annotate}.py`（保留实际执行版本）。
- 真实 RAW/RGB 通路日志：`outputs/n01-v12-p3-audit/real-{raw,rgb}.log`。
- 主运行根：`/workspace/owac/rawvla/research/n01-v1/runs/v1-2-p3-main`。
- 远端停止回执：主运行根下 `receipts/independent-audit-hold.json`。
- 远端身份重核：主运行根下 `receipts/independent-audit-identity.json`。
- 有限真实通路：`/workspace/owac/rawvla/research/n01-v1/runs/v1-2-p3-independent-audit-20261016/receipts/`。

本地复核（在本 checkout）：

```bash
make check
uv run --locked pytest tests/ -q
uv run --locked pytest \
  tests/integration/test_n01_v12_training_acceptance.py \
  tests/integration/test_n01_v12_support_acceptance.py \
  tests/integration/test_n01_v12_queue_acceptance.py \
  tests/integration/test_n01_v12_libero_dev.py -q
```

远端 GPU 复核仍需 source checkout 和 GPU，ISP 测试设置 `OWAC_RAWVLA_SOURCE=/workspace/owac/rawvla/source`；本地需要 CUDA 或作者源码的跳过不能称为通过。队列交接测试执行实际生产 shell，只用假子命令替代昂贵训练，覆盖首项成功、首项失败继续、最终非零退出和持久 hold；它不冒充几天后真实训练完成的交接实测。

## 6. 恢复主体训练前必须完成的具体工作

1. 完成 B/C：恢复并核验可用对应权重，分别限制 RAW/RGB 输入信息；让运动对应实际进入支持估计和内容融合；实现已登记的局部拟合/空间候选，或先由研究负责人明确修订方法定义。当前代理残差不得改名为 GLS。
2. 冻结一份真实可执行的观测协议：全视野/分辨率、RAW 噪声/resize/ISP 顺序、相机与合法历史到达；训练与服务逐字段/逐张量核对。不能仅把 `--crop 128` 改称 resize，也不能靠注释豁免视野差异。
3. 明确预热及后端适配阶段、实际梯度/更新和成本，验证停止/恢复；新代码与机制使用新 run 身份，先重测吞吐，再登记预算，不把当前 20k 直接升级为充分训练。
4. 补 demo→真实初态的内容身份审计，再固定开发清单；旧 0/96 留档，不通过重新解释它获得留出结论。修后模型的开发验证仍沿用每轮总 100 次，不自动扩大。
5. 完成上述可运行证据后才解除持久 hold、启用主体队列；Base 臂仍按用户决定留在 P4。未启动 P4 冻结全量，尚无机制或 RAW 优势证据。

本次接受范围仅为修补后的状态/训练器/筛选/调度正确性，以及回执中明确实跑的有限真实通路。**完整 N01 P3、充分训练、留出性能和机制收益均未验收通过。**

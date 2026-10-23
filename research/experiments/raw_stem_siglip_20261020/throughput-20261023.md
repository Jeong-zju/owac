# 全量蒸馏吞吐优化：2026-10-23

用户要求尽可能提高GPU利用率，保持global batch16。当前仅380GiB新增磁盘上限，无GPU时间/显存/利用率人为门槛。

Sol实测训练期间GPU占用0–100%波动，非仅验证；12410→12420仍推进。数据预取已经覆盖本次观测的加载等待。主要实现由原DeepSeek Web会话V41-Flash Max完成，Sol逐段审diff并退回有数值与指标错误的验证批处理版本，未部署该版本。当前验证保持原串行完整8013窗口，每500步及遍末一次，指标和分母未改。

部署4d6c126：前缀stem两个历史时刻合批，最多16个prefix packets，之后原时间顺序更新门控，不保留跨更新陈旧状态，不重播完整视觉塔。新增正常训练日志sample_wait_seconds与compute_seconds，无额外GPU同步。12617完整检查点native0保存后实际续训12620以上。

部署3a75a93：门控沿相同时刻对活动样本向量化，减少逐样本小GPU算子。12655完整检查点native0保存后从相同模型/Adam/scheduler/RNG/采样游标继续。global16、整个视觉头428579952参数、原生损失和BF16不变。23相关CPU测试及make check通过；含前缀长短/mask/reset数值回归。没有额外GPU准入benchmark。

首段20秒GPU均值49.15%/32.7%，显存峰值31472/30986MiB；12630记录wait0.00369秒、compute1.23217秒。第二段15秒GPU均值38.67%/39.53%，显存峰值30924/30066MiB；12660记录wait0.00393秒、compute1.49997秒。此前12100–12500普通日志步中位1.76379秒；不同批次/历史长度和短采样窗口不可直接作严格加速比。数据等待不是这些观测的主耗时，GPU仍未持续满载，不能声称优化已经解决所有瓶颈或已有显著加速。进一步性能工作应围绕前缀stem/递归小算子、CPU发射和通信的实际trace，不以减少历史/验证样本或扩大globalbatch替代。

实际运行train-r8，12660更新/202560次呈现，原生退出收据尚无。当前最新验证12500的feature_total0.426146，非本性能优化的验证收益。没有新动作/闭环证据。运行收据在outputs/raw-distillation-all-bench-20261022/{throughput-20261023-receipt,vector-throughput-receipt}.json。无定时任务恢复。


## 17:10 memory/data-pipeline repair (Sol independent review)

User reported periodic RAM/CPU/GPU stalls. Live cgroup limit is 116 GiB, not host RAM. Before repair, anonymous memory was 37.5–38.4 GiB and file cache 76.6–77.4 GiB. In 20.04 seconds direct reclaim scanned 2,149,693 pages (8.20 GiB scanned, not unique bytes freed), with 7,968 major faults and 453 memory.max events; OOM/kill remained the historical 1/1. CPU used 480 CPU-seconds and 117/200 periods throttled. These establish reclaim and scheduling pressure, not a new OOM or proven memory leak.

Same DeepSeek Web session official/flash/max implemented physical-frame sharing across overlapping bursts, 16-frame temporary normalization/resize chunks, and explicit release of completed batch/graph references before the next step or validation. Sol rejected the first whole-history temporary allocation and insufficient tests, then reviewed the revised diff. Complete causal history, both resize stages for RoboTwin, padding, camera order, global16, optimizer, loss and validation frequency are unchanged. No global cache flush, hashes, resource stop gate or model-output cache was introduced.

Independent validation: 26 related tests and make check passed. Tests cover nonconstant two-domain measurements, prefix/unroll equivalence, shared storage and bounded resize batch. Readonly NumPy warnings are expected on input views; inspected paths perform no in-place write to the mapped source.

Two simultaneous CPU processes on the real server, each preparing the same eight RoboTwin anchors at endpoint150, four loader workers/rank, OMP/MKL8: warm median 1.5301 s before -> 0.9738 s after; maximum process RSS 7.773 -> 5.966 GiB. This is a narrow warm loader benchmark, not end-to-end GPU throughput. OMP1 and OMP2 did not improve this benchmark, so thread configuration remains unchanged.

Owned train-r8 paused natively with exit0 at complete update19294. train-r9 resumed optimizer/scheduler/RNG/sampler from that checkpoint using d1d66c3, PID117308/startticks26538147; observed actual update19360, batch16. An owned exit observer was started; no new periodic automation. SwanLab sidecar and monitor remain alive. Recent logged ordinary-step median 1.13 s (7 points19300–19360), versus 2.00 s (19 points19100–19290); different anchors and short windows mean this is provisional, not a controlled speedup estimate.

After-resume 31.76 s sample: anonymous memory 23.43–23.88 GiB, file cache 91.02–91.46 GiB; no new OOM. Direct reclaim still scanned 4,257,784 pages with38,787 major faults. Restart/cache warmup and different batches confound comparison; memory pressure and GPU fluctuations are NOT fully resolved. OS file cache still fills available RAM, which by itself is not an allocation leak. Variable full-prefix lengths, three-camera versus two-camera work, full serial validation every500/epoch end, and checkpoint saves remain distinct sources of variation.

Raw evidence: outputs/raw-distillation-all-bench-20261022/harness/round-12/{dispatch.json,task.md,loader-bench-*.jsonl,memory-pressure*.json}; remote train-r8-exit.json and train-r9-launch.json. No test-set/closed-loop/control claim.

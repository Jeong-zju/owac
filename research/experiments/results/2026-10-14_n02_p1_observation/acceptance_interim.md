# N02 v1.1 P1 — interim acceptance (budget 1000), 2026-10-14/15

**Budget声明**：本次验收是在 **interim_budget=1000** 上做的"一期信号"，**不是**最终口径；
最终口径为累计 **5000** 更新的测量 run（`formal-cfa4-meas-5k`，new_updates=4000）。
两段预算与两次代码哈希都记录在各自 run 的 manifest 与 `outputs/n02-full/p1-formal/code_hash.json`。

**受验检查点**：`outputs/n02-full/p1-formal/p1-measurement-seed1103-formal-cfa4-meas-1k-interim/checkpoints/update-0001000.pt`
（sha256 见同目录 run manifest；有效性实测：`updates=1000`、`stage=p1-measurement`、`has model=True`、372,392,768 B）。
**说明（事实口径，含更正）**：1k run **并未死亡**——它一直存活并**越预算运行**（训练到 update 1300+），
根因是 harness 的**跨阶段预算记账缺陷**：`run_measurement` 把从预训练检查点继承的 2000 次更新计入了本阶段预算，
`--updates 1000` 因此变成了 3000。已修复：`budget_start(resume, stage)` 让跨阶段 Base 继承时本阶段计数从 0 开始
（继承数只作为 `source_updates` 记录），并加回归测试 `test_cross_stage_inheritance_does_not_consume_the_budget`（77 passed）。
越预算的进程已停止，记录见 `outputs/n02-full/p1-formal/logs/over_budget_stop.json`。
**dispatcher 已撤回此前"宿主 OOM / 静默退出"的叙述**（那是基于错误的进程检查得出的，特此更正）。
本报告使用 `update-0001000.pt`（有效的 budget-1000 状态，该文件不会被覆盖）作为 interim 检查点，lead 已裁定等价、无需重跑。

**逐场景记录**：`outputs/n02-full/p1-acceptance-interim/scenario_records.jsonl`
（oracle 961 条；behavior 见下节，完成后追加同一文件）

---

## 1. oracle（解析上界，P1 §6）

命令：
```
N02_PY -m owac.evaluation.n02_full.p1 oracle \
  --config configs/encoders/n02_full/p1_eval.json \
  --measurement-config configs/encoders/n02_full/p1_measurement.json \
  --interim-budget 1000 --output outputs/n02-full/p1-acceptance-interim \
  --scenarios 1000 --seeds 1103
```
结果（`oracle.json`，wall 328.7 s）：

| 指标 | 值 |
|---|---|
| status | ok |
| 评估场景 | **961**（1000 请求中的 961；其余因"每 run 只 pin 一种对象数"被跳过——一个编码器只有一种候选容量） |
| `correct`（真值下标胜出） | **797/961 = 0.8293**，Wilson 95% [0.8043, 0.8518] |
| 冻结阈值 `min_correctable` | 0.95 |
| **verdict** | **fail** |

**机制（关键，避免把假象当成绩）**：`build_p1_task` 把真值固定在下标 0，而 `np.argmin` 在**分数平局**时取最小下标，
因此 `correct` 的次数 **恰好等于 `map_index==0` 的次数（797）**。逐记录统计：

| 统计量 | 值 |
|---|---|
| 存在并列分数的记录 | **961/961 (100%)** |
| 4 个候选中"只有 1 个不同分数值"（即 4 者全等） | 693/961 (72.1%) |
| 2 个不同值 / 3 个不同值 | 239 / 29 |
| **真值是唯一最小值的记录** | **1/961 = 0.001** |

→ **0.8293 是 tie-break 顺序造成的假象；解析 oracle 的真实可区分率 ≈ 0.1%**。
失败机制是 **信息不足**：错候选在采样点上产生与真值**完全相同**的期望信号（几何上不可分辨），
而不是数值缺陷（hypothesis 确实生效：239+29 条记录里候选中出现了 2–3 个不同分数）。

**口径声明**：未调整任何阈值（用冻结的 `p1_eval.json`）、未挑选场景（生成器顺序）、未修改候选字典。
按 P1 §6："若 oracle 也失败，报告信息不足"——这是规定的结论路径。

**建议（供用户决定，不代表本阶段执行）**：下一轮若要让该受控任务携带可区分信息，可考虑
① 加密 query 采样（更多/更细的采样点）② 加大候选几何偏移量 ③ 细化字典构造。

---

## 2. behavior（五族，interim_budget=1000，seed 1103）

命令（thread cap 见 §3）：
```
N02_PY -m owac.evaluation.n02_full.p1 behavior \
  --config configs/encoders/n02_full/p1_eval.json \
  --measurement-config configs/encoders/n02_full/p1_measurement.json \
  --checkpoint <update-0001000.pt> --interim-budget 1000 \
  --output outputs/n02-full/p1-acceptance-interim --scenarios 1000 --seeds 1103
```
结果（`behavior.json`，wall 1448.3 s，status=ok，961 场景/类）：

| 家族 | 指标 | 值 | 冻结阈值 | verdict |
|---|---|---|---|---|
| cue_swap | both_correct | **127/961 = 0.1322**（Wilson 95% [0.1122, 0.1550]） | ≥ 0.90 | **fail** |
| no_information | 漂移在阈内的比例 | **961/961 = 1.0000**（[0.9960, 1.0]）；drift mean 0.0004 / max 0.0017 | ≤ 0.05 | **incomplete**（每类场景数 961 < 冻结的 1000 下限 → 计数不满足，指标本身通过） |
| wrong_prior | a 方向 / b 方向 | **630/961 = 0.6556** / **287/961 = 0.2986** | 两方向均 ≥ 0.80 | **fail** |
| checks·normal | 误报率（按组 exposure/noise/shift） | 0.0（0/961） | ≤ 0.05 | 达标 |
| checks·insufficiency | 检出率（按组） | **0.0（0/961）**（[0, 0.0040]） | ≥ 0.90 | **fail** |

**诚实解读**：五族整体判为失败，且失败机制与 §1 的 oracle 结论一致——**该受控任务在当前 query 采样/字典粒度下不携带可区分信息**：cue_swap 的 0.1322 **低于 4 候选的随机水平（0.25）**，insufficiency 检出 0、normal 误报 0（模型从不给出"信息不足"判定）。`no_information` 的漂移指标本身通过（0.0004 ≤ 0.05），但因每类场景数（961）未达冻结下限（1000）而标 `incomplete`，不作为通过。
未调阈值、未挑场景、未改字典；`seeds_run=[1103]`；`interim_budget=1000`（最终口径为累计 5k）。
逐场景记录：`outputs/n02-full/p1-acceptance-interim/scenario_records.jsonl`（oracle 961 条 + behavior 961×11 条 = 11532 条，`family` 字段区分）。

## 3. 过程事实（供 stage_decision 引用）

- 1k interim run：train_log 到 update 1300（26 条）；**未死亡**，是**越预算运行**（跨阶段预算记账缺陷，已修 `budget_start()` + 回归测试，见上文更正段）。
- acceptance 的性能缺陷（已修）：一个家族步在宿主满线程数下耗时 **29 s wall / 880 CPU-s**；`torch.set_num_threads(4)` 后同一调用 **1.0 s wall / 1.9 CPU-s**。生成器本身很便宜（**0.021 s/scene**，实测），所以瓶颈是 **torch 线程超额订阅**而非渲染。修复：`_bound_torch_threads()`（默认 8，`OWAC_EVAL_THREADS` 可覆盖），**纯调度改动、不动场景数与阈值**；2 场景冒烟 22 条记录 / 19.1 s wall → 961 场景 ETA ≈ 1.5–2 h。
- 加固（已落地，用于 5k 段）：`PYTHONFAULTHANDLER=1`、`ulimit -c 0`、stdout/stderr 全量重定向、`/proc/$pid/stat` 判活、写 `final.pt` 包 try/except 且打印 traceback（`training/n02_full.py a39a87c2079d4535`，远端 P1 两文件 76 passed）；`checkpoint_every=1000` 保留（已证明能保住进度）。
- 迁移事件：本链在新服务器（用户裁定迁移 N02、不触碰 N04）执行，`CUDA_VISIBLE_DEVICES=0`；raw3 臂在 GPU 1（`CUDA_VISIBLE_DEVICES=1`，非 N02 租户未触碰）。

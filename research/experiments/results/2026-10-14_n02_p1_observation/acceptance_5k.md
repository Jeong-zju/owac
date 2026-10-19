# N02 v1.1 P1 — FINAL-scope acceptance (budget 5000), 2026-10-15

**Budget声明（已更正）**：本文件验收的检查点实际处于**累计 4000 更新**状态，**不是** 5000——原因是 harness 的
`budget_start` 接线缺陷（`run_pretrain` 与 `run_measurement` 的 stage 字符串被互换），使 5k 续段只跑了
3000 新更新就停在累计 4000（run manifest 实测：`updates=4000`、`new_updates=3000`、`cumulative_updates=4000`）。
**本意是 5000**；缺陷已修复（换回两个 stage 字符串 + 新增接线守卫测试
`test_each_stage_budgets_its_own_budget`，远端 79 passed，`training/n02_full.py 4f20d98b4d2bfb70`）。
验收收据里的 `interim_budget=5000` 字段是当时传入的**请求值**，与检查点真实状态（4000）不一致——以本说明为准。
**因此本文件的正确标签是 `budget=4000`**；若之后补跑真正的 5000，会另写一节并与本文件的数字并列。
interim（budget=1000）的结果另见 `acceptance_interim.md`，两份都保留、互不覆盖。

**受验检查点**：`outputs/n02-full/p1-formal/p1-measurement-seed1103-formal-cfa4-meas-5k/checkpoints/final.pt`
**逐场景记录**：`outputs/n02-full/p1-acceptance-5k/scenario_records.jsonl`
**输出目录（独立，未覆盖 interim）**：`outputs/n02-full/p1-acceptance-5k/`

---

## 1. oracle（解析上界，P1 §6）

| 指标 | 值 |
|---|---|
| status / wall | ok / 99.3 s |
| 评估场景 | 961（1000 请求中 961；其余因"每 run 只 pin 一种对象数"跳过） |
| `correct`（真值下标胜出） | **797/961 = 0.8293**（Wilson 95% [0.8043, 0.8518]） |
| 阈值 `min_correctable` | 0.95 |
| **verdict** | **fail** |

**机制（与 interim 完全一致，且已用更强口径证实）**：`build_p1_task` 把真值固定在下标 0，
`np.argmin` 平局取最小下标 ⇒ `correct` 恰等于 `map_index==0` 的计数。
interim 段逐记录统计为：**961/961 条存在并列分数、693 条四个候选全等、真值唯一最小仅 1/961 (0.001)** ⇒
**解析 oracle 的真实可区分率 ≈0.1%，0.8293 是 tie-break 假象**；机制是**信息不足**
（错候选在采样点上产生与真值相同的期望信号），非数值缺陷。

## 2. behavior（五族，budget=5000，seed 1103，961 场景/类，wall 1426.7 s，status=ok）

| 家族 | 指标 | 实测 | 冻结阈值 | verdict |
|---|---|---|---|---|
| cue_swap | both_correct | **17/961 = 0.0177**（[0.0111, 0.0281]） | ≥0.90 | **fail** |
| no_information | 漂移在阈内比例 | **961/961 = 1.0000**（[0.9960, 1.0]）；drift mean **0.0081** / max **0.0397** | ≤0.05 | **incomplete**（961 < 冻结下限 1000；指标本身通过，但 max 漂移已接近阈值） |
| wrong_prior | a / b 方向 | **66/961 = 0.0687**（[0.0543, 0.0864]） / **718/961 = 0.7471**（[0.7187, 0.7736]） | 两向 ≥0.80 | **fail** |
| checks·normal | 误报率（exposure/noise/shift） | 0.0（0/961，三组） | ≤0.05 | 达标（无信息可报 ⇒ 也不误报） |
| checks·insufficiency | 检出率（三组） | **0.0（0/961）**（[0, 0.0040]） | ≥0.90 | **fail** |

**诚实解读**：最终口径同样整体 fail，且比 interim 更差/更极端——
cue_swap 从 0.1322 降到 **0.0177**（远低于 4 候选随机水平 0.25），wrong_prior 出现**强方向不对称**
（a 0.0687 vs b 0.7471，只在一个方向"纠正"），insufficiency 从不触发、normal 也从不误报。
这与 oracle 的结论一致：**该受控任务在当前 query 采样/字典粒度下不携带可判别信息**，
因此"测量训练 4000 步"并未带来可用能力（interim 与 final 均为 fail，机制相同）。
**未调阈值、未挑场景、未改字典**；`seeds_run=[1103]`。

## 3. 过程与口径

- **更正记录**：本节原先写"cumulative_updates=5000"；实测为 **4000**（见上"Budget声明（已更正）"）。数字未变，标签更正。

- 两段 code_hash：interim 段与 5k 段分别记在 `outputs/n02-full/p1-formal/code_hash.json`；
  1k→5k 的续段使用修复后的预算记账（`budget_start()`：跨阶段 Base 继承不计入本阶段预算），
  manifest 记 `new_updates=4000` / `cumulative_updates=5000`。
- 验收路径性能缺陷已修（纯调度）：一个家族步在宿主满线程数下 29.0 s / 880 CPU-s，
  限线程（`_bound_torch_threads()`，默认 8）后 1.0 s / 1.9 CPU-s；生成器 0.021 s/scene。
- 本文件由 T4（harness owner）写入；`stage_decision.json` 与阶段报告由 dispatcher 维护。

# N02 v1.1 P1 — true-5000 acceptance (seed 1103), 2026-10-15

**Budget声明**：本文件是**真正的累计 5000 更新**口径验收。来源：4000 状态检查点经
`--resume <…5k/checkpoints/final.pt> --updates 1000` 严格新增 1000 次更新（修复后的
`budget_start` 记账；源 manifest 实测 `updates=4000 / new_updates=3000`，本次续段
`new_updates=1000 → cumulative_updates=5000`，检查点 `update-0005000.pt` + `final.pt`）。
与 4000 口径（`acceptance_5k.md`）**并列保留、各自标注**。
两段 code_hash 与接线缺陷记录见 `outputs/n02-full/p1-formal/code_hash.json` 与
`restart_provenance.json` 的 `budget_start_wiring_defect`。
续段与验收所用代码：`training/n02_full.py 4f20d98b4d2bfb70`（修复后）。

**受验检查点**：`outputs/n02-full/p1-formal/p1-measurement-seed1103-formal-cfa4-meas-5k-true/checkpoints/final.pt`
**逐场景记录**：`outputs/n02-full/p1-acceptance-5k-true/scenario_records.jsonl`
**输出目录（独立）**：`outputs/n02-full/p1-acceptance-5k-true/`

---

## 1. oracle（解析上界，P1 §6）

| 指标 | 值 |
|---|---|
| status / wall | ok / 115 s |
| 评估场景 | 961（1000 请求；其余因"每 run 只 pin 一种对象数"跳过） |
| `correct` | **797/961 = 0.8293**（Wilson 95% [0.8043, 0.8518]） |
| 阈值 / verdict | 0.95 / **fail** |

与 4000 口径**逐位一致**（oracle 不含训练模型，是解析上界）——两次都说明：
0.8293 == `map_index==0` 计数、961/961 存在并列、真值唯一最小 1/961 ⇒
**真实可区分率 ≈0.1%，信息不足**，不是训练步数不够。

## 2. behavior（五族，961 场景/类，wall 1585.0 s，status=ok）

| 家族 | 指标 | 实测 | 冻结阈值 | verdict |
|---|---|---|---|---|
| cue_swap | both_correct | **11/961 = 0.0114**（[0.0064, 0.0204]） | ≥0.90 | **fail** |
| no_information | 漂移在阈内比例 | **961/961 = 1.0000**（[0.9960, 1.0]）；drift mean **0.0089** / max **0.0457** | ≤0.05 | **incomplete**（961 < 冻结下限 1000；指标本身通过，max 已贴近阈值） |
| wrong_prior | a / b 方向 | **76/961 = 0.0791** / **695/961 = 0.7232** | 两向 ≥0.80 | **fail** |
| checks·normal | 误报率（三组） | 0.0 | ≤0.05 | 达标 |
| checks·insufficiency | 检出率（三组） | **0.0** | ≥0.90 | **fail** |

## 3. 两个口径并列（同一 seed、同一 961 场景集、同一冻结配置）

| 指标 | budget=4000（`acceptance_5k.md`） | budget=5000（本文件） |
|---|---|---|
| oracle correct | 0.8293（fail，tie-break 假象） | 0.8293（fail，**逐位一致**） |
| cue_swap | 17/961 = 0.0177 | 11/961 = **0.0114** |
| wrong_prior a / b | 0.0687 / 0.7471 | 0.0791 / 0.7232 |
| insufficient 检出 / normal 误报 | 0.0 / 0.0 | 0.0 / 0.0 |
| no_information（在阈内比例 / max 漂移） | 1.0 / 0.0397 | 1.0 / 0.0457 |

**诚实结论**：多训练 1000 次更新**没有**带来任何可用能力；两口径都是 **fail**，
机制同为 **信息不足**（oracle 已证明该受控任务在当前 query 采样/字典粒度下不可区分）。
**未调阈值、未挑场景、未改字典**；`seeds_run=[1103]`。

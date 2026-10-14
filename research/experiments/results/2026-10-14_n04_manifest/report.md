# N04 S0 远端数据盘点与拆分（ws-manifest）

执行时间 2026-10-14（UTC 03:49–04:15）／分支 `lzh/n04-egomotion-separation`（HEAD `fd6255b`）。
全部数据读取为只读；`/workspace/owac/rawvla/**` 未写入；`splits.json` / `coverage.json` /
`inventory.json` 在首轮复核后未再改动（哈希保持有效）。

## 1. 交付物

| 位置 | 文件 | sha256 |
| --- | --- | --- |
| 本地 | `src/owac/raw/n04_manifest.py` | `cffd1f36d99fcfa03c8c4e38d8b2a2db31b2f7bebc7e0d485e4bf47d6f28b197` |
| 本地 | `tests/encoders/n04_full/test_manifest.py` | `85419dcfb37452f279ed702203ab7528034070c129c6ceb20d425ac07b020852` |
| 本地 | `research/experiments/results/2026-10-14_n04_manifest/official_test_disjointness.py`（追加交付脚本） | `c0f28d9537d78fc1ee760de82e92814bb876d96682e237997ab63cd1bb1c89c0` |
| 本地+远端 | `official-test-disjointness.json`（追加交付） | `7bc23e9b82be84cf8ace8c0733d04f97cf45deb17a3ad7f0b4fb89a0d2b617d0` |
| 本地+远端 | `splits.json` / `coverage.json` / `telemetry-provenance.json` / `inventory-hashes.json` / `inventory-summary.json` / `stdout.json` | `74653094…` / `3dbf18ca…` / `8415ed52…` / `1c413bf4…` / `64cc1f2b…` / `01a5734f…` |
| 远端 | `manifests/s0-manifest/POINTER.json`（含 8 个 run 目录状态与理由） | `925d4c1ea43a02be7642d90a37bff4227fa0fedef9701e22a9ac6774d1613225` |
| 远端 | run `data/s0-manifest/run-20261014T040654Z/`（`-rerun` 二次生成） | `inventory.json` 62,057,281 B `e0154ab7815e2019201bfde9cf391e23b2805d6f19aba2b849b66ae3a77b3c69` |
| 远端 | 模块级快照 `data/s0-manifest/code/n04-s0-ws-manifest-20261014T040622Z-5e275d95c691/` | 逐文件哈希见其 `SNAPSHOT.json` |

## 2. 实际命令（远端解释器 `/usr/bin/python3.12`，numpy 2.5.3）

```bash
SNAP=/workspace/owac/research/n04-v1/data/s0-manifest/code/n04-s0-ws-manifest-20261014T040622Z-5e275d95c691
cd "$SNAP" && PYTHONPATH=$SNAP/src /usr/bin/python3.12 -m owac.raw.n04_manifest --help
PYTHONPATH=$SNAP/src /usr/bin/python3.12 -m owac.raw.n04_manifest \
  --config "$SNAP/configs/n04/full/s0_data.json" \
  --output <run>/inventory.json --summary-out <run>/inventory-summary.json \
  --reference-root "$SNAP/references"
# 追加交付（只读官方测试清单）
/usr/bin/python3.12 /workspace/owac/research/n04-v1/data/s0-manifest/official_test_disjointness.py \
  --run-dir /workspace/owac/research/n04-v1/data/s0-manifest/run-20261014T040654Z \
  --output /workspace/owac/research/n04-v1/manifests/s0-manifest/official-test-disjointness.json
```

本地：`uv run --locked --extra encoder pytest -q tests/encoders/n04_full/test_manifest.py`（23 passed）、
`ruff format/check` 仅作用于本任务自有文件。

## 3. 关键结果

- 真实 NPZ：**2403**（LIBERO 1771 + RoboTwin 632），509,194,817,528 B（474.2 GiB）。
- 独立源轨迹：**2403**（LIBERO `dataset_name:task_name:demo_key`；RoboTwin `task_name:source_trajectory`），
  每源仅 **1** 个变体文件（`variant_index=0`），本份数据没有多照明/噪声变体文件。
- 帧对齐 2403/2403、`declared` 与实测 0 处不一致、`missing_rate` 全 0、有效帧 408,075。
- 深读 48 例（每平台 24，分层轮转，seed 20261014）：finite/alignment/missing_rate/camera_range/
  重复帧/proprio-action 帧数全部通过；LIBERO `states` 首列 24/24 等间隔严格单调（0.05 s），
  其余小数组前导列 `not_applicable`（非时间列不算失败）；RoboTwin 布局 24/24 数值核对通过（wxyz 模长 1）。
- 拆分（seed 20261014）：train/dev/final = **1917/251/235**（LIBERO 1413/186/172；RoboTwin 504/65/63），
  源 ID 交集三项全空；`member_table_sha256 = 0d3d41911816361b5b5e1093e07515f87ded913739d685e056cdeb9fe5f12247`，
  `main_split_sha256 = 8d2879385c7feb18e9b76420d4692fbe30d793442186ca1d5af8bfdb71ac5111`。
- 二次生成：两次独立全量运行 `content_sha256 = fce3bd48fdc288ffd7b96eee64c06cd8724eb29cef6dda22eba618bb14e31921`
  相同，`splits/coverage/telemetry-provenance/inventory-hashes` 逐字节一致；summary 仅计时段不同。
- 成本：扫描 9.44 s + 深读 0.39 s = 9.97 s，读 10.0 GB（≈1012 MiB/s），峰值 RSS 163 MiB，单进程。

## 4. 不可用项（与 Lead 裁决一致）

- 时间戳：两平台**不可用**（键并集无时间/帧率键、metadata 无 fps；`time_unit="frame"`）；
  LIBERO `states` 首列单调性只登记为未决观测。
- 相机标定：两平台**不可用**（RoboTwin 内参仅运行时计算，LIBERO 由 MuJoCo 定义）。
- 已执行动作：LIBERO `partial`/**provisional**（`producer_code_order`、`verified=false`）；
  RoboTwin **不可用**（渲染不执行动作 + 重放记录未交付）。
- 曝光/关节/位姿 `partial`；噪声模型 RoboTwin `available`、LIBERO `partial`；RGB 配对与源身份 `available`。
- 8 条不可用条件清单见 `coverage.json.unavailable_conditions`。

## 5. Lead 裁决记录（2026-10-14，6 项）

1. **分层口径接受**：`stratify_by=["platform","task_key"]` 予以接受（小任务也有非空 dev/final），
   记录为**对配置字段 `manifest.splits.stratify_by="platform"` 的已披露偏差 + 理由**，不重跑。
2. **scene holdout 窄覆盖接受**：27 源（可解析 `<SCENE><n>` 的仅 libero_10 的 10/40 任务）。
   **明确记录：scene holdout 覆盖窄，S1/S6 的泛化结论不得仅依赖它。**
3. **camera holdout 视角级语义接受**：**它是视角级 holdout，不是源组级相机 holdout**；
   当前数据无法按相机分组源轨迹，留出视角只做零样本/适配分报，源轨迹不排除。
4. **`manifests/s0-manifest/` + POINTER.json 命名接受**；与 ws-infra 的 `manifests/environment.json`、
   `manifests/frozen-assets.json` 不冲突（本任务不写这两个文件）。
5. **保留 superseded run 目录**；`POINTER.json.runs` 已逐目录标注
   `current` / `stability-rerun` / `superseded + superseded_reason`（含各自 `content_sha256`、
   `code_sha256`、`main_split_sha256`），避免后续误用旧 run。
6. **追加交付**：`official-test-disjointness.json`（见下节）。`splits.json`/`coverage.json`/`inventory.json`
   未做任何修改。

另确认：`rawvla_sequence.source_identity` 的 RoboTwin 折叠问题保持旧函数不改（保守但过粗、不泄漏），
仅在 `splits.json.identity_note` 记录；`rawvla_sequence.py` 归 ws-telemetry。

## 6. 官方测试清单与训练源轨迹的身份交集（⚠️ 发现重叠，未自行修补）

输入：`records/pi05-full-20261010-v3/libero-manifest.json`（10,000 条，
sha256 `e3a054f43060912d668e57d35873b573a289c6d3c941fec9c029569c82971168`）与
`robotwin-manifest.json`（3,250 条，sha256 `632b8360836595a868affffd24f9ecf1262d68a52a505ef5ffe801598d8317e5`）；
训练侧为冻结 `splits.json`（`74653094…`，2403 个 `source_trajectory_id`）与 `inventory.json`（`e0154ab7…`）。

| 层级 | 训练侧 | 测试侧 | 交集 |
| --- | --- | --- | --- |
| 任务（LIBERO `(suite,task_id)` / RoboTwin `task_name`） | 40 / 13 | 40 / 13 | **40 / 13** |
| 源 episode（LIBERO `+demo_index` vs `+init_state_index`；RoboTwin `+episode` vs `+dataset_episode_index`） | 1771 / 632 | 2000 / 650 | **1771 / 632** |
| 测试条目（每个源 episode 身份出现 5 次照明域） | — | 10000 / 3250 | **8855（88.55%）/ 3160（97.23%）** |
| `source_trajectory_id` 字符串 | 2403 | 0（清单不携带该字段） | 0 |

判读（**结构级 vs 语义级**分开写）：

- **结构级**：测试清单覆盖 `(suite, task_id, init_state_index)` 的完整 4×10×50 网格与
  `(task_name, dataset_episode_index)` 的 13×50 网格，训练身份必然落在网格内 → 该层交集等于
  训练侧全部条目。此结论与两侧编号是否同序无关。
- **语义级（RoboTwin）：字段级确认**。测试清单顶层 `dataset.trajectory_template =
  "_traj_data/episode{dataset_episode_index}.pkl"`、`episode_data_template =
  "data/episode{dataset_episode_index}.hdf5"`，与训练 metadata 的 `source_trajectory="episode{N}.pkl"`、
  `source_hdf5="episode{N}.hdf5"` 同名同构；并且 632/632 条训练记录的 `dataset_seed` 与同
  `(task, dataset_episode_index)` 测试条目的 `dataset_seed` **完全一致**（第二个独立字段佐证）。
- **语义级（LIBERO）：高可能性推断，未经生产者代码确认**。测试清单无 `task_name`、无 episode 模板；
  依据是两侧每任务索引基数均为 50、`protocol.json` 的清单来源名为
  `libero_manifest_50_init_states_10000_rollouts.json`、以及 LIBERO 标准评测使用数据集初始状态。
- **不主张**：测试 rollout 的观测/动作不在训练集中；训练拆分代码从未读取测试清单
  （`n04_manifest.py` 只读配置的两个训练数据根，`grep records_frozen_protocol|*-manifest.json` 命中 0），
  故这不是拆分代码泄漏，而是数据协议事实：**官方测试 episode 的初始状态/源 episode 与训练 render
  同源**。是否可接受、以及 S1/S6 泛化表述如何限定，请 Lead 裁决记录。

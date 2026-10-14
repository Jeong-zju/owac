# N04 S0 合同审计结果（工作流 D）

- 分支：`lzh/n04-egomotion-separation`，HEAD `fd6255b`（含未提交的并行工作流产物）
- 审计工具：`src/owac/evaluation/n04_contracts.py`
- 预注册门槛：`configs/n04/full/protocol.json`（S0 先算后冻结；诊断参照下的 provisional 冻结）
- C3 修复：`src/owac/frameworks/n04/model.py` 在 `N04Encoder.step` 进 stem 前加
  `where(observed, values, 0)` 清洗（+9 行，无其它语义改动；
  旧 sha256 `768c8203967c8dcf54af208833caeab61bfb9e80b08b88077656981729492c60` →
  新 sha256 `93af3a3994221050c6ad9e84395c1cfca636c2b6d203fbb4369b7020f3f6f599`）。

## 实际执行的命令

```bash
uv run --locked --extra encoder python -m owac.evaluation.n04_contracts --help

# 运行 1（冻结 CLI）
uv run --locked --extra encoder python -m owac.evaluation.n04_contracts \
  --config configs/n04/full/s0_data.json \
  --output research/experiments/results/2026-10-14_n04_s0_contracts/contracts.json

# 运行 2（同一 CLI 接入 ws-manifest 真实产物）
uv run --locked --extra encoder python -m owac.evaluation.n04_contracts \
  --config configs/n04/full/s0_data.json \
  --manifest research/experiments/results/2026-10-14_n04_manifest/splits.json \
  --output research/experiments/results/2026-10-14_n04_s0_contracts/contracts-with-manifest.json

uv run --locked --extra encoder pytest -q tests/encoders/n04_full/test_contracts.py   # 11 passed
uv run --locked --extra encoder pytest -q tests/encoders/n04                          # 198 passed
uv run --locked ruff format/check src/owac/evaluation/n04_contracts.py \
  tests/encoders/n04_full/test_contracts.py src/owac/frameworks/n04/model.py
# 正常路径逐位不变对照：用 git show HEAD 的旧 model.py 建临时同构包，与新实现同输入对比
```

## 结果总表（C0 为附加一致性检查，C1–C7 为 S0 必做验证）

| id | 检查 | 运行 1 | 运行 2（接入 manifest） |
| --- | --- | --- | --- |
| C0 | protocol.json 冻结与换算一致性 | passed | passed |
| C1 | D0 逐样本图像—标签对应 | passed | passed |
| C2 | 90°/复合逆/深度/像素中心/CFA | passed | passed |
| C3 | 未观测值 NaN/极大值不影响更新与梯度 | passed（修复后） | passed |
| C4 | 置乱不变性；已执行控制改变上下文 | passed | passed |
| C5 | 迟到帧回补与重放；未到达本体不泄漏 | partial（代码语义 passed；真实数据 skipped） | partial |
| C6 | sensor residual / 曝光增益 / 饱和遮挡 | passed | passed |
| C7 | manifest 一致性与拆分互斥 | partial（C7.3 未接产物） | passed |

汇总：运行 1 = 6 passed / 0 failed / 0 skipped / 2 partial；运行 2 = 7 passed / 1 partial。

## 关键数值证据

- C1：独立 FP64 重投影 vs `frame_labels`，B=1/2/4 最大误差 0.0 px（4 物体 × 2 相机）；
  batch 置换后 `body_index` 错位 0 像素、标签误差 0.0；`audit_d0` failures 为空。
- C2：90° yaw/roll/pitch 误差 0.0 px；`|AB−BA|=1.0`、逆序恒等式 0.0；像素中心精确；
  CFA 四相位全对、打包往返 0.0；crop→resize 内参链 1.42e-14 px；stride=32 单位链 8.88e-16 px；
  有限差分 6.99e-12；负深度 support=0。
- C3（修复后）：NaN 与 1e30 两种毒化下 stable/change/uncertainty/representation 全部有限，
  18/18 参数梯度有限，且与「干净掩码 + 零填充」运行逐位一致（max grad diff 0.0）；
  正常（全有效）路径相对修复前逐位不变（两帧、含显式饱和掩码与不传掩码两种调用）。
- C4：预测签名无测量参数；batch 置换先验差 0.0；已执行控制改变先验（结构扰动下 0.919）。
- C5（分层）：`code_semantics` passed —— 14/14 合成 sidecar 语义检查（迟到帧到达前不可见、
  到达后按曝光序可见、重放仅已执行且保序、planned→pending、`arrival_delay=0.1s` 记录、
  `target_actions_included=False`；`RawVLASequence.context_at` 与
  `RawVLASequenceLoader.context_at(sequence, …)` 真实调用一致）；
  `real_data` skipped —— 时间戳不可用、标定不可用、LIBERO 已执行动作 partial/provisional
  （`evidence_level=producer_code_order`, `verified=false`）、RoboTwin 不可用。
- C6：静态无噪声同曝光有历史后 residual=0.0；曝光增益 2× → 均值比 2.000048、
  方差比 1.915574（Poisson–Gaussian 预测 1.918421，400 采样）；饱和/遮挡各自把区域
  `g_update` 压到 0。
- C7：真实 splits 1917/251/235，源 ID 交集全 0；成员表 hash 重算 = 记录值
  `0d3d41911816361b5b5e1093e07515f87ded913739d685e056cdeb9fe5f12247`
  = 用同记录二次生成值 = `inventory-hashes.json` / `stdout.json` 记录值。

## 三维门槛（`research/theory/n04_v1.md` §7，诊断参照）

- 公式：`X = (u − cx)·Z/fx`；`s_px = Z/fx`；`s_feat = stride·Z/fx`
- f=320 px、Z=0.6 m、256×256、stride=32：
  1 原图 px = 1.875e-3 m = 1.875 mm；0.05 px = 9.375e-5 m = 0.09375 mm；
  0.05 px 折合 1.5625e-3 特征 px；1 特征 px = 60 mm
- Lead 裁决：门槛以原图像素空间为准 = max(0.05 px, 1 原图 px) = **1.875 mm**（诊断参照）；
  特征空间计分指标必须先换算回米/原图像素再判门槛，并同时报告特征等效值；
  不设 1 特征 px = 60 mm 的独立门槛。
- H1：误报预算 0.01 固定；干扰负例误报相对降 ≥20%；事件召回降 ≤2pp；配对区间支持方向。
- H2：停止/反向积分误差降 ≥15%；缺失期退化 ≤5%；修订延迟不退化。

## 依赖版本（本次运行的 sibling 文件 sha256 前 12 位）

`n04_mechanism_data 0613376f278b`、`n04_manifest cffd1f36d99f`、`telemetry 786cea46f032`、
`rawvla_sequence a56b052ab4c4`、`measurement_geometry 5012a499d5ec`、`geometry 35c2596ec815`、
`model 93af3a399422`、`n04_raw_diagnostics 2616b5830de0`。

## 未通过 / 未验证

- 无 failed。C5 的 `real_data` 分支仍是 skipped（依赖真实时间戳/标定/执行日志），
  合成 sidecar 通过不代表真实数据合同通过。
- C7.3 需要显式 `--manifest`；运行 1 因此为 partial，运行 2 已复核通过。

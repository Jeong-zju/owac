# N01 v1.0 · P1 阶段报告（可组合与可撤回的解析参考）

2026-10-14；工作树 `worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`。起点为 P0 独立验收后的 `b7f36a7`（P0 验收报告：`P0_acceptance_report.md`）；本阶段只执行 P1，结束后停在交接点。

## 0. 三个状态字段

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `implementation_status` | `PASS_P1` | P1 规定的五个模块 + 评估入口 + 三个测试文件全部落地；本地 473 passed、`make check` 通过；远程源包 152/152 hash 一致、`verify` 22/22 通过、远程 125 passed、`analytic` 9/9 |
| `training_status` | `NOT_STARTED` | 本阶段无长训练；仅远程 `mode=contract` 的 ≤2 batch/平台 开发性更新（不是训练预算，不产生模型能力结论） |
| `effect_status` | `NOT_APPLICABLE` | P1 是解析参考实现与数值/行为验证，不测机制效果；不比较 N01 增量 |

P1 的完成条件按阶段原文逐条满足（见 §1.3、§3）；数值结论只覆盖声明的小问题与远程真实数据读取，**不构成**理论证明、性能或闭环结论。

## 1. 实际修改文件、入口、数据流与要求对应

### 1.1 文件（本阶段新增/修改）

| 文件 | 内容 |
| --- | --- |
| `src/owac/frameworks/n01_full/ledger.py` | `MeasurementLedger.observe / get_sources / reset / without / with_content`；按帧 tensor + 紧凑索引；身份去重、四态 validity、部分 episode reset、反事实视图与派生内容视图 |
| `src/owac/frameworks/n01_full/domains.py` | `make_domain / merge_domains / split_domain / retract_sources / restore_sources`；`support_version`（血缘）与 `content_digest`（内容）分离；合并拒绝版本不一致域 |
| `src/owac/frameworks/n01_full/statistics.py` | `fit_structure`（逐相位 `cholesky → solve_triangular → lstsq`，`Lambda^-1` 用 Cholesky；秩/`identifiable_mask`；先验单列；未知相关性固定权重保守界；`InterpolationSpec` 与奇异回退） |
| `src/owac/frameworks/n01_full/replay.py` | `full_replay / fixed_decision_replay / counterfactual_replay`、`ReadoutSnapshot` 提交、32 帧精确窗口与历史损失、`ToyProgram` + `recompute_successor_closure`（依赖闭包性质测试） |
| `src/owac/frameworks/n01_full/reference.py` | 独立 NumPy FP64 稠密 oracle（`dense_gls` / `reference_fit`，`eigh` 白化 + 显式稠密逆；不调用 `statistics.py`） |
| `src/owac/frameworks/n01_full/sequence1d.py` | 新契约下的一维同语义适配例子（顺序保留旧 `frameworks/n01/sequence1d.py` 与旧测试不动） |
| `src/owac/frameworks/n01_full/config.py` | **P0 文件兼容性修改**：`typing.Self`（3.11+）改为具体类名前向引用，使 `import owac.frameworks.n01_full` 在固定的 Python 3.10 策略运行时可用（行为不变） |
| `src/owac/evaluation/n01_full.py` | 新增 `analytic` 子命令：阶段 §4 的 7 项检查 + 插值/保守界 + 一维例子，输出 `analytic.json` / `analytic.txt` |
| `src/owac/runtime/n01_full_ledger_check.py` | 真实数据唯一来源计数入口：真实 NPZ 窗口逐到达事件进入新 ledger |
| `src/owac/runtime/n01_full_policy_preflight.py` | `mode=contract` 策略预检（复用旧参考编码器 + 真实特征桥 + 冻结 π0.5）；`mode=raw_adapter` 明确拒绝（P2） |
| `configs/encoders/n01_full/policy_contract_encoder.json` | 与已验收 `configs/encoders/n01/small.json` **逐字节相同**（sha256 `9dd07bae…`），放入 P1 打包目录以保证快照自包含 |
| `tests/encoders/n01_full/test_ledger.py / test_statistics.py / test_replay.py` | 34 项新测试（见 §2.2） |
| `research/theory/n01_support_contract_v1.md` | T1/T2/T5 的假设、推导、代码入口、反例与「仍未证明」边界；映射表由「P1 待实现」更新为已实现 |

### 1.2 入口与数据流

```text
真实 NPZ 窗口 [B,T,C,H,W]
  └─ runtime/n01_full_ledger_check.iter_window_frames   # 逐时刻到达事件（轴 1）
      └─ MeasurementLedger.observe                      # SourceKey 去重；missing/bad/saturated 不入估计
          └─ domains.make/merge/split/retract/restore   # 版本化支持域（证据/上下文角色分离）
              └─ statistics.fit_structure               # 逐相位 GLS + 秩 + 不确定性标签
                  └─ replay.full_replay                 # 提交快照 + 精确窗口/历史损失
                      ├─ fixed_decision_replay          # 冻结决策 + 内容替换（内容隔离）
                      └─ counterfactual_replay          # 删除来源 + 全量重算 + 独立参考比较
                          └─ reference.py               # 独立 NumPy FP64 oracle（不与生产共享代码路径）
```

### 1.3 与阶段要求的对应

| 阶段要求 | 落地 |
| --- | --- |
| `ledger.py`：observe/get_sources/部分 reset，按帧 tensor + 紧凑索引 | ✅（每帧一个 tensor + 一个索引项；`get_sources` 按需物化记录） |
| `domains.py`：make/merge/split/retract，返回新版本 | ✅（+`restore_sources` 作为 retract 的逆，供 §4.4 的「retract 再恢复」） |
| `statistics.py`：常数/二维仿射 GLS，返回秩/估计/不确定性种类 | ✅（+先验、插值协方差、保守界） |
| `replay.py`：`full_replay` / 两种回放 | ✅ |
| `reference.py`：独立 NumPy FP64 稠密实现，不调用生产 `statistics.py` | ✅ |
| 三个测试文件 + `evaluation/n01_full.py` 的 `analytic` | ✅ |
| 保留旧 `frameworks/n01/sequence1d.py` 入口与测试，新实现补一维同语义例子 | ✅（旧文件未改；新增 `n01_full/sequence1d.py` + 测试 + `analytic` case） |
| 补齐 T1/T2/T5 假设、推导、代码入口、反例 | ✅（理论文件 §2.6/§3.6/§6.6 与映射表） |
| 明确自适应选域/估计 flow 后不继承固定设计 GLS 协方差 | ✅（理论文件 §3.6 限制段 + `X-T2-i`） |

## 2. 执行证据：命令、hash、环境、资源

### 2.1 起点与提交

| 项 | 值 |
| --- | --- |
| 起点 | `b7f36a7`（P0 修补提交）；工作树仅有的未提交修改是用户既有 `docs/OWAC_academic_research_proposal_2026-10-09.md`，未被本阶段触碰 |
| P1 提交 | `18d01e1`（解析核心 + P0 config 兼容）、`bf0ca4f`（测试 + `analytic`）、`e42ad1c`（真实数据入口 + 策略预检 + 参考编码器配置） |
| 远程对应 | 远程快照 `source-v8/` 的 152 个文件与提交后工作树**逐一 hash 比对一致**（0 不一致、0 缺失），即远程证据绑定 `e42ad1c` 的文件内容 |

### 2.2 本地命令（实际执行）

```bash
uv run --locked pytest -q                     # 473 passed, 1 warning
make check                                    # ruff check / ruff format --check / compileall 全部通过
uv run --locked python -m owac.evaluation.n01_full analytic --output <dir>   # 9/9 cases passed
```

新增测试分布：`test_ledger.py` 12、`test_statistics.py` 12、`test_replay.py` 10（含真实 loader 形状与 tensor 身份的回归测试）。本地环境：Python 3.12.12 / torch 2.9.1+cpu / numpy 2.5.1。

### 2.3 远程源包与验收

| 项 | 值 |
| --- | --- |
| 快照 | `/workspace/owac/rawvla/research/n01-v1/runs/p1-20261014/source-v8/`（v1–v7 保留在案，见 §4.1 的迭代日志） |
| 源包 | `source.tar.gz` sha256 `4d54972cf145578b96627405c65cae135c2cb34649b54b28c6ef788589c1c529`，152 文件；本地与远程重算一致 |
| `verify` | `status=passed`，22/22 checks，`failures=[]`、`pending=[]` |
| 远程测试 | `pytest -q tests/encoders/n01_full` → **125 passed** |
| 远程 `analytic` | **9/9 cases passed** |
| 远程路径 | `outputs/{verify,analytic,ledger-check,policy-contract}/`（本报告只回写小型 JSON） |

### 2.4 远程环境与资源

| 项 | 值 |
| --- | --- |
| 编码器环境 | `/workspace/owac/rawvla/envs/encoder/bin/python`：Python 3.11.16 / torch 2.9.1+cu128 / numpy 2.4.6 |
| 策略环境 | `/workspace/owac/rawvla/envs/starvla/bin/python`：Python 3.10.21 / torch 2.7.1+cu128 / CUDA 可用 |
| GPU | RTX PRO 6000 Blackwell，策略预检开始前占用 51,218 MiB / 97,887 MiB（含其他方向服务，未停止任何他人进程） |
| 数据/权重 hash | LIBERO 权重 `68710f4e…`、RoboTwin 权重 `9a5381c3…`、tokenizer `8986bb4f…`（与冻结协议逐条一致，运行时重算） |
| 裁剪 | 策略预检与 ledger 检查均显式 `--crop 32`（左上裁剪，如实披露；非全分辨率） |

## 3. 数值与行为结果

### 3.1 阶段 §4 的 7 项要求（`analytic` 全部 pass）

| # | 要求 | 观测（本地与远程一致） |
| --- | --- | --- |
| 1 | y=[1,3]、方差 1：均值 2、方差 1/2；重放第一个 ID 后不变；真正新 ID 按三观测 | 均值 2.0、方差 0.5；重放后方差仍 0.5 且 `duplicate_new_samples=0`；新曝光（数值相同）登记为第 3 个观测 → 均值 5/3、方差 1/3 |
| 2 | Σ=[[1,0.5],[0.5,1]]、Φ=ones：均值 2、方差 0.75；撤回第二观测后方差 1；截取旧 precision 的错误版本必须被识别 | 2.0 / 0.75；撤回后 1.0；错误版本给出 0.75，被判别（`wrong_version_detected=true`） |
| 3 | 已知仿射场、分数坐标、同相位值、异方差小问题：生产 vs 独立 whitened LS，记录条件数 | 2 相位 × 3×4 网格、分数采样坐标（+0.4）、逐样本方差：`max|Δmean| ≤ 1.8e-15`、`max|Δvar| = 0`（FP64 `atol=1e-10, rtol=1e-8`；差异来自求值顺序，远低于容差）；条件数 2.47/2.47；另有一维仿射例（cond 2.78） |
| 4 | 重叠 merge、不同顺序、split 再 merge、retract 再恢复、同事件不同分块 | `merge(A,B)` 与 `merge(B,A)` 版本相同；合并后证据 3 条（去重）；split 再 merge 的 `content_digest` 相同；retract 再 restore 的 `content_digest` 相同而 `support_version` 不同（血缘）；分块与单次事件流一致；声明版本变化改变摘要 |
| 5 | missing/bad/saturated 不作为零观测；全无效返回无效 mask；无先验秩亏不获得虚假 confidence | 5 个样本中 3 个无效 → 只用 2 个，均值 2.0，原因分别为 missing/bad/saturated；全无效：`rank=0`、掩码全 False、`uncertainty_kind=none`、`theta=None`；秩亏（仿射、x 退化）：`rank=2<3`、掩码 `[True,False,True]`、无协方差 |
| 6 | 两个 camera 同 frame ID、两个 episode 同 frame ID 不冲突；reset 第一个成员不影响第二个 | 三组同 frame_id 各自独立取值；`c0` 2 帧 / `c1` 1 帧；部分 reset 后另一成员 raw 值不变 |
| 7 | 内容替换撤回 vs 删除来源全重放；版本变化触发重算 | 冻结决策下替换内容：干净路径 `max|Δ|=0`（unchanged），**泄漏负对照** `max|Δ|=205.17`（被识别）；反事实重放与独立 oracle 一致；越界撤回声明 `rebuild_from_legal_boundary` 且历史损失 2 帧；parameter/support 版本变化使派生节点 stale |

### 3.2 插值、先验与保守界

- 插值行协方差按 `A_interp Σ A_interpᵀ` 记录：单行 `w=[0.5,0.5]`、`Σ=[[1,0.5],[0.5,1]]` 的常值拟合方差 = 0.75（手算一致）。
- 重复/重叠插值导致行协方差奇异（最小特征值低于声明容差）→ **回退到独立原始观测作为因子**，`singular_interpolation_fallback=true`，观测数仍为原始唯一来源数（Cholesky 单独判据不可靠，见 §4.1-④）。
- 先验单列：`prior_id=prior-v1`，独立观测数仍 3（先验不计入观测量），后验精度 = 2 观测 + 1 先验。
- 未知相关性：固定权重保守界 `(Σ|aᵢ|σᵢ)²` = 0.5；等相关 0.5 噪声 4,000 次抽样的经验方差 0.3389 < 0.5，且严格大于独立观测公式；标签为 `proxy`，不与精确 `covariance` 混用。

### 3.3 真实数值流与唯一来源计数（远程，`ledger-check-remote.json`）

| 平台 | 到达事件 | 登记帧 | 重复 | 唯一帧/相机 | 说明 |
| --- | --- | --- | --- | --- | --- |
| LIBERO（train） | 33（32 窗口帧 + 1 次重放） | 28 | 5 | agentview 14 / wrist 14 | 首帧 padding 与重复 frame_id 由身份去重，重放新增样本 0 |
| RoboTwin（train） | 49（48 + 1） | 42 | 7 | head 14 / left 14 / right 14 | 同上 |

- `sample_xy` 为加载器声明的 **逐通道 (row,col) 偏移** `[3,2]`（三通道 pseudo-RAW 为零），由命名适配函数展开为逐样本 `(x,y) = 像素 + 偏移`；未按猜测改写其它形状。
- validity 全为 `valid`（crop 32 内无掩码）；四区路径来自 `dataset_from_protocol`，与 P0 冻结清单同源。

### 3.4 `mode=contract` 策略预检（远程，`policy-contract-remote.json`）

复用**已验收的旧三通道参考编码器**与真实特征桥（P1 的解析系数不是神经前端，如实标注），冻结 π0.5 后端：

| 平台 | 训练 batch（不同 episode） | action loss | encoder 梯度范数 | 后端梯度 | token 形状 | 单步 wall |
| --- | --- | --- | --- | --- | --- | --- |
| LIBERO | `libero_10…episode_000002` / `libero_object…episode_000040` | 0.88432 / 4.72427 | 532.92 / 2483.39（有限） | 0 | `[1,512,2048]`（2 相机 ×256） | 2612.7 / 2083.4 ms |
| RoboTwin | `grab_roller…episode_000001` / `place_dual_shoes…episode_000023` | 1.07568 / 1.58268 | 90.45 / 195.19（有限） | 0 | `[1,768,2048]`（3 相机 ×256） | 3805.7 / 4335.1 ms |

- 实际参数更新：encoder 权重变化 L1 42.56（LIBERO）/ 44.11（RoboTwin）。
- validation：只做无更新前向（loss 0.5269 / 2.0567），**参数漂移 0.0**。
- 冻结后端 375 个参数组中梯度累计 0；token 宽度 2048 与 `VisualTokenPostprocessor` 约定一致。
- 该预检只证明接口/梯度连通与后端冻结，**不**证明动作质量或闭环收益（P7 仍须完成完整模型与三种子动作训练）。

## 4. 失败、未执行与缺失数据

### 4.1 P1 检查发现并修复的缺陷（保留迭代日志，源包 v1→v8）

| # | 缺陷 | 发现方式 | 修复 |
| --- | --- | --- | --- |
| ① | 0 维 tensor 身份：真实 loader 传 `frame_ids[:, t]`（0 维 tensor），其 `hash` 按对象，导致同一 `SourceKey` 被重复登记 | 远程真实数据唯一来源计数（第二版快照） | `ledger._scalar` 归一化为 Python 标量 + 回归测试 `test_tensor_identity_slices_deduplicate_like_python_ids` |
| ② | 时间轴错位：窗口是 `[B,T,C,H,W]`，最初按轴 0 迭代，一个窗口只读 1 个时刻 | 远程计数（第一版）远小于 8 帧 | 抽出 `iter_window_frames`（轴 1）+ 回归测试 |
| ③ | `sample_xy` 语义：加载器声明逐通道 `(row,col)`，不是逐样本坐标 | 远程 ledger 检查报形状错误 | 命名适配函数 `_sample_grid` 展开 + 回归测试；未猜测未知形状 |
| ④ | 奇异插值判据：`torch.linalg.cholesky` 对数值奇异矩阵可能返回极小正主元而不抛错 | 本地 `analytic` case `interpolation_and_bounds` 失败 | 改为特征值—容差判据触发文档化回退；测试固定两分支 |
| ⑤ | 校验前向用了 `torch.inference_mode()`，被桥的训练前向守卫拒绝 | 远程策略预检（v6） | 改用 `torch.no_grad()`，并以参数漂移 0.0 证明无更新 |
| ⑥ | `config.py` 导入 `typing.Self`（3.11+），在固定的 Python 3.10 策略环境无法 import | 远程策略预检（v6） | 改为具体类名前向引用（P0 文件的兼容性修改，行为不变，本地 125/473 测试仍通过） |
| ⑦ | 打包目录不含旧参考编码器配置，快照不自包含 | 远程策略预检（v5 之前） | `configs/encoders/n01_full/policy_contract_encoder.json`（与 `n01/small.json` 逐字节相同） |

远程快照 v1–v7 与 `outputs/verify/p1-verify-v{2..7}.json` 全部保留；只有 v8 作为最终证据。

### 4.2 未执行、未测量与明确不声称

| 项 | 状态 |
| --- | --- |
| 长训练 / 学习融合 / 机制效果 | **未执行**（P1 不含）；`effect_status=NOT_APPLICABLE` |
| `mode=raw_adapter`（三通道 DBSR 适配、因果窗口） | **明确拒绝**，属 P2；不伪造入口 |
| 全分辨率策略前向 | 未测（`--crop 32` 开发性裁剪，如实披露） |
| 真实 CFA/Bayer 数据 | 未用于本阶段；远程为作者三通道 pseudo-RAW |
| 32 帧精确窗口以外的撤回 | 只声明「从合法边界重建 + 历史损失」，不声称精确 |
| 理论命题 P-T1-a…e / P-T2-a…e / P-T5-a…d | 仍未证明；P1 只提供实现与有限反例证据 |
| GPU 独占 / 成本 | 未承诺；他项目服务占用未动 |

## 5. 接口交接（P2 可用路径）

```bash
# 解析参考（本地或任一 CPU 环境）
uv run --locked python -m owac.evaluation.n01_full analytic --output <dir>

# 远程（cwd = 快照，PYTHONPATH=<snapshot>/src）
python -m owac.runtime.n01_full_ledger_check --platform libero --platform robotwin2 --batches 2 --crop 32 --output <dir>
python -m owac.runtime.n01_full_policy_preflight --mode contract --platform libero --platform robotwin2 --output <dir>
# P2 交付后：--mode raw_adapter（当前显式拒绝）

# 源包与验收
python -m owac.runtime.n01_full_preflight bundle --output <empty-dir>
python -m owac.runtime.n01_full_preflight verify --bundle <dir> --source-root <dir> --protocol configs/encoders/n01_full/protocol_v1.json --report <json>
```

可复用对象：`MeasurementLedger`（身份/四态/reset/反事实视图）、`VersionedDomain`（版本化域操作）、`StructureFit`（逐相位 GLS + 预测回声明坐标）、`reference.py`（独立 oracle）、`replay`（两接口 + 玩具闭包）、`ToyProgram`（依赖闭包性质测试）。

已知差异与必须带进后续阶段的事项：

1. `config.py` 已为 Python 3.10 兼容改动（`Self` → 具体类名）；后续阶段不要再引入 3.11+ 专属 typing 特性到该模块。
2. 精确窗口默认 32（`ReplayConfig`），算法比较可用 8/16/32；越界只重建不精确。
3. `fit_structure` 的插值支持单 tap/行（`InterpolationSpec` 可显式多 tap）；多 tap 的 `CorrespondenceMap` 分组留给 P2 明确构造，不做隐式猜测。
4. 未测项：全分辨率、真实 CFA、卷积感受野跨界重编码（P4）、Mamba/递归重放（P5）、策略闭环（P7/P8）。

## 6. 结论

P1 的解析参考实现、独立 oracle、两套回放接口与真实数据入口已按阶段原文落地并通过本地 + 远程验收：本地 473 passed、`make check` 通过；远程源包 152/152 一致、`verify` 22/22 通过、125 passed、`analytic` 9/9；真实两平台窗口的唯一来源计数与 `mode=contract` 桥接/冻结证据齐备。检查过程中发现并修复了 7 个实现缺陷（含 2 个会破坏身份去重的严重问题），全部有回归测试。

本阶段没有机制效果或闭环结论；理论命题仍未证明。已停在 P1 交接点，等待研究复核后再由下一条提示词推进 P2。

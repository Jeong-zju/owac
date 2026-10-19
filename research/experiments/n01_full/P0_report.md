# N01 v1.0 · P0 阶段报告（问题、数据与依赖）

> 后续独立复核发现协议、可复跑入口、成本及版本一致性缺口；已修补。当前结论见 [P0_acceptance_report.md](P0_acceptance_report.md)。下文保留为原始交付记录。

日期 2026-10-14。工作树 `/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`。
本阶段只执行 P0；未开始 P1 算法、未启动任何正式长训练。

---

## 0. 三个状态字段

| 字段 | 取值 | 依据 |
| --- | --- | --- |
| `implementation_status` | **PASS（限 P0 范围）** | P0 文件职责表的 10 个路径全部交付并通过独立复核；`make check` 干净；全仓 434 passed（历史基线 348 + 新增 86）。不代表 P1+ 已实现。 |
| `training_status` | **NOT_STARTED** | P0 不含收敛训练。只做了真实尺寸 profiling（吞吐/显存）与两平台真实 batch 读取。 |
| `effect_status` | **NOT_APPLICABLE** | 本阶段不产生机制效果结论，也不得从此报告推导任何 N01 增量。所有 profiling 数字是成本，不是性能。 |

---

## 1. 实际修改文件、入口、数据流与要求对应

### 1.1 拟新增文件全部落地（10/10）

| P0 要求路径 | 实际交付 | sha256 |
| --- | --- | --- |
| `src/owac/frameworks/n01_full/contracts.py` | 契约类型 `SourceKey / MeasurementRecord / CorrespondenceMap / SupportDomain / EvidenceEstimate / DependencyNode / ReadoutSnapshot`（+ 支撑类型 `CalibratedMeasurement`、`ReplayRange`） | `e67ced38f0bb869392ee7cd7940c37fc9b2dbef8f7daa32bb58f492d1c454503` |
| `src/owac/frameworks/n01_full/config.py` | 输入身份/basis/history/replay/对应/内容/训练预算七段配置 + JSON round-trip | `9a8f7b0268680c153c7fea9397a4b1980ca35757461c93ae3f01b247582d53b2` |
| `src/owac/frameworks/n01_full/__init__.py` | 包出口 | `1e1a4a20c411427dd67113cda7ee7d9b9b163588e7d3120b3a2166f49d2806e8` |
| `src/owac/raw/n01_full_manifest.py` | inventory → source_group → 四区 → loader 可读 `sides` | `16c103293cb9b7685edd37bf0bc585b639856175e066bd460c924d673f035a66` |
| `src/owac/runtime/n01_full_preflight.py` | `inventory / profile / bundle / verify` 四子命令 | `96858a07daedff74fcb6765aaaee48256a697b1daaa010f99149bb9eb83c9b8f` |
| `configs/encoders/n01_full/protocol_v1.json` | 冻结研究矩阵、数据、随机流、预算、统计规则 | `7d0fa9e7cf992c2d24aca2585f93c82062fcf821e2e25d9f9e5f2ac68b82bfb2` |
| `configs/encoders/n01_full/upstream_lock.json` | 9 个上游仓库 + vendored DBSR 的提交/许可/文件 hash/依赖/配方/权重来源 | `7879e9c6b7da550b7f52f4955193061c9ce17c1b5a78d9a77d689f6d729338bf` |
| `configs/backends/n01-full-runtime/` | `requirements-gpu.txt` `70849ec9…`、`README.md` `bbcf6896…`、`owac-n01-v1.supervisor.conf` `57da091c…` | — |
| `data/manifests/n01-full-v1/` | 7 个小型 JSON（见 §3.3） | 见 §3.3 |
| `research/theory/n01_support_contract_v1.md` | T1–T6 定义/假设/待证明/反例/映射 | `d75f0cfeeaaea6f5bdb76623e6c6b0c999850ecf3f45a8a3ca285d855eba99ee` |
| `tests/encoders/n01_full/test_foundation.py` | 跨模块行为 + 独立 oracle | `8848b30dd63a3aa766cf30aa6697787ab9998ceff8ef19df599e1e9be45ad1b8` |

成员各自的定向测试：`test_contracts_config.py` `6c641f6a…`（20 passed）、`test_manifest.py` `c8d766b7…`（16 passed）、`test_preflight.py` `e53530fe…`（33 passed）、`test_foundation.py` `8848b30d…`（17 passed，含自包含的 `git archive HEAD` 判别测试），合计 **86 passed**。

### 1.2 数据流

```
服务器真实 NPZ 池（只读）
  ↓ n01_full_manifest：inventory（路径+大小+mtime 库存 hash）→ source_group → 四区
data/manifests/n01-full-v1/*.json（sides + rows + regions）
  ↓ protocol_v1.json 绑定 manifest/inventory/loader-evidence 的 sha256
  ↓ RawVLASequenceDataset(split_manifest=..., raw_cache_root=..., stats_path=...)
MeasurementWindow [T,C,H,W] → (P1+) SourceKey / MeasurementRecord / SupportDomain / EvidenceEstimate
  ↓ (P4+) N01FullEncoder.step → readout → VisualTokens → 冻结 π0.5 桥
```

- 本阶段新增的全部模块是**数据与契约起点**，不产生网络、不训练、不接后端。`N01FullEncoder` 外部适配器属 P4，P0 只在 protocol 里留 `pending_P4` 占位，不写空壳。
- 复用而未改写的既有资产：`core/encoder_contract.py`、`raw/rawvla_sequence.py`（`source_group` / `assign_side` / `build_split_manifest` / `RawVLASequenceDataset`）、`baselines/burst_features.py`、`backends/rawvla_features.py`、`runtime/n01_acceptance.py`。

---

## 2. 执行证据：命令、hash、环境、资源

### 2.1 起点冻结

```
HEAD   429591eb1308454b019eebcac6501f84cdb27fcd   （docs(n01): define full evidence integration development plan v1.0）
branch lzh/n01-evidence-integration
dirty  M README.md / M docs/OWAC_academic_research_proposal_2026-10-09.md /
       M docs/prompts/README.md / M docs/status.md ；?? docs/prompts/n01_v1_0/（19 文件）
```

以上 4 个未提交文件与未跟踪提示词集都是**上一轮「提示词准备」产物**，按 AGENTS.md 原样保留，本阶段未覆盖、未回退。

**前序交付身份复核（未重跑 74k 步矩阵）**

| 复核项 | 结果 |
| --- | --- |
| `docs/prompts/n01_v1_0/local_source_audit.json` 的 10 个文件 hash | **10/10 复算一致** |
| `research/experiments/results/2026-10-13_n01_dbsr_mechanism/protocol-lock.json` 锁定文件 | **4/4 一致**；`verification.json` 记录 348 passed 保留为历史证据 |
| 三种子结果 | 保留原收据，不重新宣称；checkpoint 平台状态不转移给新训练任务 |

**发现的前序异常（记录，未修改）**：`docs/OWAC_academic_research_proposal_2026-10-09.md` 的未提交改动把三个标题改成了 `### 5.1 自适应时空证据积分N01` / `5.2 …N02` / `5.3 …N04`，疑似误编辑。属他人未提交变更，本阶段不回退，交研究负责人处置。

### 2.2 本地命令（实际执行）

```
make check                                     → exit 0（ruff check / ruff format --check 234 files / compileall src）
uv run --locked --extra encoder pytest -q tests/            → 434 passed, 1 warning
uv run --locked --extra encoder pytest -q tests/encoders/n01_full → 86 passed
uv run --locked python -m owac.runtime.n01_full_preflight {inventory,profile,bundle,verify} --help → 5/5 exit 0
```

以上为 **P0 提交后（HEAD `dab6800` + 测试修正提交）** 在干净工作树上的数字。提交前工作树是「未提交」状态，其中 `test_foundation.py` 曾因断言「工作树是否脏」在提交后失败（432 passed / 1 failed）；该测试已改为**自包含**（在 `tmp_path` 自建 git 仓库与 `git archive HEAD` 对照），现在两种状态下均通过，判别力反而更强。

### 2.3 远程环境与资源

| 项 | 实测 |
| --- | --- |
| 连接 | `ssh -p 44327 root@107.209.104.125`（只读核查 + 独立新目录写入） |
| 环境 | `/workspace/owac/rawvla/envs/encoder` — Python 3.11.16 / torch 2.9.1+cu128 / CUDA 12.8 / cuDNN 9.10.2.21 / triton 3.5.1 / numpy 2.4.6 / pytest 9.1.1 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition，compute capability 12.0，97,249.5 MiB 总量 |
| GPU 验收 | `[1,2,4,64,64]` fp32 → `[1,512,64,64]` fp32 全 finite；1,497,056 参数、56 个梯度张量全 finite/非零；**真实 forward+backward**，非 import 检查 |
| 磁盘 | `/workspace` overlay 2.0T，本阶段结束时 799G 已用 / 1.3T 可用（起点 786G，+13G 为本阶段环境、快照与 profiling 产物） |
| 他项目服务 | 4 个 policy 服务全程 RUNNING、pid 未变；GPU 跑完回到基线 32,047 MiB，无残留进程 |
| 本地锁的关系 | 本地 `uv.lock` 的 encoder extra 固定 **CPU** torch 2.9.1，与远程 CUDA 构建分离，已在 `configs/backends/n01-full-runtime/README.md` 显式声明，未互相覆盖 |

### 2.4 数据、配置与权重 hash（本机重新实测，非转录）

| 资产 | sha256 | 与既有留档 |
| --- | --- | --- |
| LIBERO `dataset_statistics.json`（1,929 B） | `7e596e50c00c5407daed5809f6cc55b57877903bbadcd39ac30f13bb821411a9` | 与 `remote_snapshot.json` 转录值一致 |
| RoboTwin `policy_norm_stats.json`（3,291 B） | `229a4a2d6cfd1ba4eb2bc0b3aa57f049094032939130e4901485e9ea9910f5f7` | 一致 |
| `paligemma_tokenizer.model`（4,264,023 B） | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` | 一致 |
| LIBERO `model.safetensors`（8,522,039,320 B） | `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8` | 一致 |
| RoboTwin `model.safetensors`（9,354,050,752 B） | `9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3` | 一致 |
| 冻结 NPZ 计数 | LIBERO **1,771** / RoboTwin **632**（两次独立清点） | 一致 |

五个资产 hash 由本报告作者在服务器 `sha256sum` 实测（含 18 GB 权重），与本地留档转录值逐条一致 —— 转录值由此获得独立复验。

### 2.5 源码快照与源包（远程）

| 项 | 结果 |
| --- | --- |
| 快照 **v5（最终）** | `runs/p0-20261014/source-v5-20261014T1217/`，227 文件，tar sha256 `8a378c935eaa805d2942d70383ad1ed8402cf570d1649800e709e73ea51cdbad` |
| 远程 hash 复算 | **227/227 OK，0 不一致，exit 0**（本报告作者另行 `sha256sum` 复核 tar 与三个关键文件，逐条相符）；v1→v5 血缘记在 `CURRENT_SNAPSHOT.txt` |
| 包含内容 | 新增未跟踪源码（`git ls-files` + `--others` 显式列表，非 `git archive HEAD`）、`docs/prompts/n01_v1_0/` 19 文件、上游材料 `outputs/n01-v1-prompt-preparation/upstream/**`（24 文件含 LICENSE） |
| 排除内容 | 运行产物（`outputs/n01-*`、大日志、handoffs）、`.venv`、`__pycache__`、权重与密钥 |
| 上游许可归属 | SEA-RAFT BSD-3-Clause、DSM-bocd MIT、mamba Apache-2.0、BSRT MIT、**DBSR CC-BY-NC-SA-4.0**（`src/owac/baselines/_vendor/dbsr/LICENSE` + provenance 一并入包） |
| 源包 `bundle`（v5） | **135 文件**，`upstream included=true (24)`，`data/manifests/n01-full-v1` 闭包 **7 文件**（`closure:data/manifests/n01-full-v1 ok=True`），`verify` **passed**（failures 0 / pending 0，22/22 checks），archive sha256 `1b687a2628890bc1b01433fd0110c4b7efe522b3443cac316bb98a8cf64aa8b2` |

---

## 3. 工程与数据结果

### 3.1 真实远程 `inventory --strict`：36/36 checks ok

```
python -m owac.runtime.n01_full_preflight inventory \
  --config configs/encoders/n01_full/protocol_v1.json \
  --output outputs/p0-inventory-strict --strict
→ exit 0, status=ok, failures=[], pending=[]
```

这是 P0 预检第一次在**真实数据根 + 真实两份平台 manifest** 上真正通过；此前在快照 v2 上的失败结果（6/12 checks 不 ok）完整保留在 `outputs/p0-inventory-strict-v2-era` 供对照。协议在快照 v5 上又重跑一次，仍为 `status=ok / failures=[] / pending=[] / 36 checks 全 ok`（v4 时代结果保留在 `outputs/p0-inventory-strict-v4-era`）。

### 3.2 四区划分

| 平台 | train | calibration | validation | test | 池 |
| --- | --- | --- | --- | --- | --- |
| LIBERO | 1,228 | 175 | 206 | 162 | **1,771** |
| RoboTwin | 450 | 48 | 58 | 76 | **632** |
| 跨平台合计 | 1,678 | 223 | 264 | 238 | **2,403** |

- 规则：`source_group`（相对 raw root 的路径）→ `rawvla_sequence.assign_side` 在累计阈值 0.70/0.80/0.90 上判定；一组一文件（池中无同轨迹多变体）。
- 零重叠四路独立确认：本地 `verify_manifest`、服务器真实 loader 四区实例 `paths` 两两交集、集成波独立 oracle、远程 `inventory --strict` 的 `disjoint:*` 与 `union:*`。
- **无官方分区**（已核实并留证）：RawVLA-Bench README 原文 *"This release contains training trajectories only. It does not define a held-out test split or a leaderboard metric."*，目录下无任何 split/manifest JSON。四区同池划分；官方闭环 13,250 条评测身份不混入本四区。

### 3.3 manifest 落盘

| 文件 | sha256 |
| --- | --- |
| `n01-full-v1-index.json` | `33de7d02c86f993a562686037271f540ac5d850ddebed0c9e8641475d2e4e30d` |
| `n01-full-v1-libero-manifest.json` | `3462b7321409d054bf74a0414ce35f6a593d4c9a4154d3646e51cba2595019c9` |
| `n01-full-v1-libero-inventory.json` | `a3cbefe59421c811d04f8451940a010c8d08e37bb51570fe339ea4451af01129` |
| `n01-full-v1-loader-evidence-libero.json` | `24a7f1aeb4fc896bb73caac86fecfb39a32867e9374d875e5a70b5133a23543b` |
| `n01-full-v1-robotwin2-manifest.json` | `f04ba21530c4b9b47e4d3c70c834396e65c050b3752d44af7fa704d393bcdd97` |
| `n01-full-v1-robotwin2-inventory.json` | `1e176c61d2b36ec17211ee81837ab079f9b9063e57844998f39486476374a98a` |
| `n01-full-v1-loader-evidence-robotwin2.json` | `034a476be16be38c3533dad46c8433e713ce0994527ec40bbc4270de3d11cf33` |

两次独立复算（交付成员、集成波）与协议内记录值三方一致。

### 3.4 数据完整性调查（本阶段新增，**重要**）

因为 `benchmark = rawvla-bench-light-libero-v1-train-rerender-npz` 与 `replay_mode=actions`，按 P0「检查照明衍生是否确属同一基础轨迹」逐项实测：

| 问题 | 实测 | 结论 |
| --- | --- | --- |
| LIBERO 1,771 文件来自多少源演示？ | `hdf5_path` 只有 **40 个**不同值，每组 27–50 个文件 | 1,771 是约 44 个 rollout × 40 个任务模板 |
| 同组文件是同一轨迹的重渲染吗？ | 同组内 `states[0]`（123 维仿真初态）**互不相同**（最大差 0.043–0.123），`actions` 形状与数值也不同（247/232/235 帧） | **不是**。每组是同一任务下的不同 rollout |
| `demo_key` / `demo_index` 能否当身份？ | `demo_key='demo_1'` 跨 suite 对应**完全不同的 `hdf5_path`** | 是 per-suite/per-task 编号，**不可**当全局身份 |
| RoboTwin 是否同构？ | 632 文件、13 任务、每任务 44–50 episode；`source_trajectory` 是 per-task 名（`episodeN.pkl`），全局去重后 50 个名，与 632 个 `(task, source_trajectory)` 对基本 1:1 | 每个文件即一个源轨迹 |
| 照明如何分布？ | 两平台各 5 个照明域、样本数大致均衡（LIBERO 338–384，RoboTwin 125–127）；照明是**每个 rollout 的属性**，不作为独立文件变体出现 | 池中**不存在**同轨迹照明变体文件 |

**结论与披露义务**：
1. 按文件路径分组等于按源轨迹分组 —— B 的四区规则在**轨迹身份**上正确，不需要改。
2. 「同轨迹照明变体保持同组」在本池中是**空条件**（没有这种文件），不是已验证性质。相关分组逻辑由单元测试覆盖，但真实数据未施加该分支。
3. **任务级共享**：40 个 LIBERO 任务模板与 13 个 RoboTwin 任务会出现在全部四区（同任务、不同初态与不同 rollout）。这与 LIBERO 训练/评测惯例一致，但必须在 P3 统计里显式处理：配对 bootstrap 的**重采样单元不能只用 episode，还应包含任务/模板层**，否则置信区间会低估不确定性。
4. 库内 `inventory_hash` 只覆盖 `relative_path/size_bytes/mtime_ns`（未覆盖内容），内容级锚点是 6 个深读 NPZ probe + 7 个 JSON 的 sha256。这是已记录的已知边界，不冒称内容校验和。

### 3.5 真实尺寸 profiling（成本，不是性能）

完整 DBSR 64 通道 / 9 残差块 / 512 维，batch=1、单相机、fp32、packed RGGB `[1,T,4,H,W]`；每点 3 次预热 + 20 次同步 forward + 20 次同步 forward+backward。

| cells | history | entry | fwd p50 (s) | fwd p95 (s) | fwd+bwd p50 (s) | fwd+bwd p95 (s) | peak alloc (MiB) | peak reserved (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 8 | encoder | 0.00298 | 0.00300 | 0.00997 | 0.01007 | 828.9 | 1132.0 |
| 64 | 16 | direct | 0.01157 | 0.01169 | 0.02767 | 0.02774 | 1635.1 | 2146.0 |
| 64 | 32 | direct | 0.01849 | 0.01860 | 0.05477 | 0.05489 | 3247.4 | 4668.0 |
| 128 | 8 | encoder | 0.01596 | 0.01609 | **0.04607** | 0.04626 | 3298.5 | 4378.0 |
| 128 | 16 | direct | 0.04170 | 0.04175 | 0.10625 | 0.10638 | 6523.1 | 7168.0 |
| 128 | 32 | direct | 0.08170 | 0.08177 | 0.21788 | 0.21834 | 12972.4 | 15108.0 |
| 224 | 8 | encoder | 0.07101 | 0.07105 | 0.17460 | 0.17480 | 10092.9 | 13294.0 |
| 224 | 16 | direct | 0.12892 | 0.12902 | 0.34491 | 0.34527 | 19965.3 | 23134.0 |
| 224 | 32 | direct | 0.26088 | 0.26116 | 0.69522 | 0.69722 | 39720.1 | 46190.0 |
| 128 | 8 | encoder | 0.01079 | 0.01090 | 0.03010 | 0.03013 | 2279.6 | 2870.0 |（bf16 autocast，0.653×） |

- 原始文件：`runs/p0-20261014/outputs/profile_dbsr_p0-20261014.{json,csv}`；本表由报告作者从 JSON 逐点复算，与成员汇报一致。
- **`entry` 列的含义**：`history > 8` 被既有 `BurstFeatureEncoder` 入口的 unique-frame 上限拒绝（`"the first reference implementation keeps at most 8 unique frames"`），16/32 点改用**同一 RawBurstBackbone + MaskedDbsrFusion 组合**直接计时，参数量同为 1,497,056。这是既有实现约束，未放宽，已在 CSV 与收据中区分标注。
- I/O（真实 loader，单进程 page-cache 热，window=8）：LIBERO p50 **0.0812 s**、RoboTwin p50 **0.0344 s**（冷首触 0.111/0.093）。**MiB/s 未实测**，协议里记为 `throughput: derived_pending` 并写明推导式，不冒充实测。
- 未测且不得假装测过：SEA-RAFT 改造、支持重放、三通道（C=3）入口、Mamba、policy/action head、optimizer、评测、checkpoint。

### 3.6 成本估算（P0 要求：20k/100k/300k）

基点 = 128/8 fp32 encoder 入口实测 `step p50 = 0.046073 s`（单相机视角）。
每次 update = 相机数 × 0.046073 s（LIBERO 2 相机 → 0.09215 s；RoboTwin 3 相机 → 0.13822 s）。
30 runs = 3 种子 × 5 方法 × 2 平台。

| 更新数 | 30 runs 总 GPU-h | 6 runs 最小起点 | 有效 batch 8（×8） |
| --- | --- | --- | --- |
| 20k | **19.20** | 3.84 | 153.6 |
| 100k | **95.99** | 19.20 | 767.9 |
| 300k | **287.96** | 57.59 | 2,303.7 |

`value_status = estimated_from_measured_step_time`（**估算，非实测**）。不含 policy/action head、optimizer、评测、checkpoint、loader I/O、SEA-RAFT/Mamba。服务器瞬时空闲不等于后续专用 GPU；已有他项目服务占用须计入预算。

### 3.7 两平台真实 batch（只读）

| 平台 | 池 | split 路径数 | 相机 | window | 单位 | source 例 |
| --- | --- | --- | --- | --- | --- | --- |
| LIBERO | 1,771 | train 1,447 | agentview + wrist | `[8,3,256,256]` fp32 | uint8/255（max 0.0235–1.0），`pi05_quantile` | `libero_10_no_noops_1.0.0_lerobot/episode_000001/variant_00.npz`，ep_id 0，frame_ids `[0,0,1,6,11,16,21,26]`，首帧 padding 且 `new_measurement=False` |
| RoboTwin | 632 | train 508 | head + left + right | `[8,3,240,320]` fp32 | 作者 HDR float32 直通（head max 0.00486），`zscore` | `grab_roller/episode_000000.npz`，ep_id 0，frame_ids `[1..8]`，action `[50,14]` 且 `action_mask=0.64` |

两平台样本均 `all_finite=true`，`valid_fraction=1.0`；LIBERO 有真实首帧 padding、RoboTwin 有真实动作尾部 padding —— 都是正例，不是伪造。

---

## 4. 失败、未执行与缺失数据（保留，不掩饰）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 远程快照上 `pytest tests/encoders/n01_full` | **3 failed / 80 passed** | 三个失败都是**测试对宿主环境的假设**，非交付缺陷：① `git ls-tree` 在解包快照（非 git checkout）上 exit 128；② `test_inventory_against_the_frozen_platform_manifests` 假设服务器数据根**不存在**（本地为 pending），真实服务器上它存在故为 ok；③ 假 factory 用 `sha256_file` 签名，本机有 CUDA 因而走到按协议调用的分支。本地同提交 85 passed。未改测试、未跳过。 |
| `tests/baselines/test_burst_features.py` 的 provenance AST 摘要测试 | 在 Python 3.11 失败，3.12 通过 | 3.12 新增 `type_params` 进入 AST dump；文件 hash 已证一致，属版本敏感性，非快照缺陷。 |
| `history > 8` 经公开 encoder 入口 | **被拒绝** | 既有实现约束；P4 若需要须在正式实现中放开，P0 不放宽。 |
| 三通道（C=3）入口 | **不存在** | `ResEncoderWarpAlignnet` 硬编码 4 通道；未把四通道模型冒充三通道 profile。 |
| SEA-RAFT / 支持重放 / Mamba 成本 | **未知** | 未安装、未测、未估。 |
| `io_mib_per_second` | `derived_pending` | 只有 wall time 实测，无逐样本字节数，拒绝填假吞吐。 |
| `thresholds.*`（6）、`planned_gates.*.value`（2）、`statistics.{alpha,min_paired_units}` | `pending_calibration` | P3 才能实测确定；正式矩阵启动前必须冻结为具体值。 |
| `profiling.encoder_factory` | `pending_P4` | P0 职责表不含 `frameworks/n01_full/encoder.py`，未写空壳。 |
| 全量闭环 / 机制效果 | **未执行** | P0 不含；不得从本报告推导任何增量。 |
| 44 个上游文件读取 / 9 个仓库运行 | 只锁源码与许可 | 未安装、未运行任何新上游模型；`runtime_verified=false` 全条目。 |

---

## 5. 接口交接（下一阶段可用路径）

### 5.1 可加载配置与命令

```bash
# 远程（cwd = 源快照，PYTHONPATH = <snapshot>/src）
python -m owac.runtime.n01_full_preflight inventory \
  --config configs/encoders/n01_full/protocol_v1.json \
  --output outputs/p0-inventory-strict --strict          # 期望 status=ok, failures=[], pending=[]
python -m owac.runtime.n01_full_preflight bundle --output outputs/p0-bundle
python -m owac.runtime.n01_full_preflight verify --bundle outputs/p0-bundle \
  --source-root <snapshot> --protocol configs/encoders/n01_full/protocol_v1.json \
  --report outputs/p0-bundle-verify.json
python -m owac.runtime.n01_full_preflight profile --config ... --mode collect   # 本地 CPU 显式拒绝；须在 CUDA 宿主
```

### 5.2 已冻结可复用的版本

| 类别 | 版本 |
| --- | --- |
| 数据四区 | `split_version = n01-full-v1`，四区 hash 见 §3.3；`source_group` 不含照明后缀 |
| 校准/单位 | LIBERO `pseudo_raw_linear_view_uint8_over_255`；RoboTwin `hdr_derived_raw_float32_author_units`（直通） |
| 统计/归一化 | LIBERO `pi05_quantile`（`stats_key=openpi_pi05_libero_franka`, `action_stats_key=action`）；RoboTwin `zscore` |
| 模型/上游 | DBSR 64/9/512 + readout 256×2048（`bridge_in_dim=feature_width`）；上游提交与文件 hash 见 `upstream_lock.json` |
| 随机流 | 模型种子 101/211/307；数据种子 1101/1211/1307（命名空间 `n01-full-v1`，与模型种子强制不相交） |
| 后端 | 冻结 RawVLA PI0/PI05（StarVLA 适配）；权重/tokenizer hash 见 §2.4 |
| 精确回放 | 32 个唯一帧（算法比较 8/16/32）；超出即作废污染摘要并从合法边界重建 |

### 5.3 已知差异与须带进 P1 的事项

1. **任务级共享**（§3.4 结论 3）：四区共享 40 个 LIBERO 任务模板 / 13 个 RoboTwin 任务。P1/P3 的配对统计重采样单元须包含任务层，不能只用 episode，否则置信区间偏窄。
2. **`input_mode` 两层语义**已消歧：协议平台级为 `model_input_mode`（契约层 C=3），manifest row 级 `input_mode` 为缓存存储编码（`raw_uint8_pseudo_raw` / `raw_float32_hdr_derived`）。两者不可互换。
3. **`history > 8`** 需在 P4 正式实现中处理；P0 未放宽既有模块。
4. **环境位置偏离**：独立环境实际在 `/workspace/owac/rawvla/envs/encoder`（与实例既有 `envs/{hub,robotwin,starvla}` 并列），不是 `research/n01-v1/envs/encoder`（venv 不可重定位）；已在该目录留指针 README，并在 `configs/backends/n01-full-runtime/README.md` 记录。
5. **远程源包版本偏斜已闭合**：v4 快照曾早于协议收尾改动（三个文件 hash 不一致）；v5 快照（tar `8a378c93…`）与本地最终状态逐条 MATCH，远程 227/227 hash 复算一致。历史 v1–v4 保留在案，`CURRENT_SNAPSHOT.txt` 记录 v1→v5 血缘。
6. **上游许可**：DBSR 为 **CC-BY-NC-SA-4.0**（非商用），源包已含其 LICENSE 与 provenance；未取回许可的仓库（confseq/s4/RViDeNet/RViDeformer/deepmind-research）在锁中标 `not_retrieved`，**未核对前不得 vendor 或分发**。

---

## 6. 结论

P0 的工程与数据起点已交付并通过独立复核：契约与配置落地、四区 manifest 在真实两平台数据上可被原 loader 精确消费且零重叠、远程独立 GPU 环境实测可用、真实尺寸 profiling 与两平台真实 batch 有原始证据留存、理论义务目录与冻结协议就位。

本阶段**没有**产生机制效果结论，也**没有**完成完整训练或闭环验证；`training_status = NOT_STARTED`、`effect_status = NOT_APPLICABLE`。

已停在 P0 交接点，等待研究复核后再由下一条提示词推进 P1。未自动执行后续阶段。

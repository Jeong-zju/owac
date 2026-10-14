# N01 v1.0 P2 RAW 运动对应与噪声模型 — 阶段报告

2026-10-14；工作树 `/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`。
起点为 P1 修补提交 `2d82bbb` 与独立验收归档 `6fb3807`（本轮 HEAD 为 `01e0316`，其上是本阶段未提交改动）。
本阶段只执行 P2 及其规定的 `mode=raw_adapter` 早期预检，完成后停在 P2 交接点；不进入 P3/P4，不执行 300k 续训。

> 状态：本地实现、测试与内部独立验收已完成；远程 3 seeds × 100k 第一段全部结束（101 plateau@82k、211 budget_exhausted@100k、307 plateau@99k，经历一次 62 分钟实例失联后精确续训），三 seed GPU 评测与两平台 `raw_adapter` 真实预检均已通过并同步收据。P2 已停在交接点，未进入 P3/P4。

## 1. 实际修改文件、入口、数据流与要求对应

### 1.1 新增/修改文件（sha256 前缀 16 位；完整值见 `research/experiments/results/n01_full/P2/receipt.json`）

| 文件 | sha256 | 角色 |
| --- | --- | --- |
| `src/owac/baselines/_vendor/sea_raft/**`（10 文件，含 LICENSE、provenance.json） | 见 §2.1 | 固定上游 SEA-RAFT（BSD-3-Clause，commit `9137517`） |
| `src/owac/backends/raw_correspondence.py` | `1a39133b02d3b4f7` | `RawSensorFrame`、`CorrespondenceEstimator.estimate`、风险/校准协议 |
| `src/owac/raw/n01_motion_data.py` | `27e44636087fd03c` | 已知连续辐射场 → CFA/pseudo-RAW 运动对生成器 + 解析几何真值 |
| `src/owac/raw/n01_noise.py` | `44dd76f2bace2f09` | shot/read + 共享黑电平块噪声、曝光/增益变换、P1 `NoiseModel` 构造 |
| `src/owac/training/n01_motion.py` | `603c8521deae6704` | 作者式迭代损失、fit-check/select/train、平台判据、checkpoint/恢复、`stem-report` |
| `src/owac/evaluation/n01_motion.py` | `8c1f5f9b2dee7408` | EPE/分位、遮挡、风险分档、静态误拒、相位一致性、时延/显存、四臂对照 |
| `src/owac/frameworks/n01_full/adapters.py` | `ff460ee397c6e94a` | `DbsrFeatureAdapter`：三通道 stem + 因果窗口 + 256 token/相机读出 |
| `src/owac/frameworks/n01_full/motion_builders.py` | `fefbfd3c8ffd3ad1` | `WindowMotionProvider`、`LedgerCorrespondenceBuilder`、`LedgerNoiseModelBuilder` |
| `src/owac/frameworks/n01_full/statistics.py` | `d24107dbdf42b3c6` | 最小追加：`InterpolationSpec.source_ids` + `align_interpolation_sources`（P1 默认路径不变） |
| `src/owac/runtime/n01_full_policy_preflight.py` | `6e5c85f094072fc9` | `--mode raw_adapter` 真实路径 + `--motion-checkpoint` 接入 |
| `configs/encoders/n01_full/p2_motion_noise.json` | `658f51f0f28fbebc` | 合成/校准参数、四区 policy、种子规则 |
| `configs/encoders/n01_full/p2_motion_train.json` | `c37786f03af6dcce` | CFA 正式训练配方（100k/300k、delta_val=0.002、种子 101/211/307） |
| `configs/encoders/n01_full/p2_motion_train_rgb3.json` | `15a3703b65a1fc78` | pseudo-RAW 3 通道 dev 臂（单独配置、单独训练） |
| `tests/encoders/n01_full/test_motion.py` 等 7 个测试文件 | 见 §3.1 | 实现者测试 |
| `tests/encoders/n01_full/test_p2_acceptance.py` | `6761a5a0cc84185c` | 独立验收对手的 53 项反例测试（与实现者隔离） |
| `pyproject.toml` | `ea2e887fd1a7a8d8` | 仅 package-data 与 ruff per-file-ignore 的最小行 |

### 1.2 数据流（与本阶段要求的对应）

```text
已知连续辐射场 (SinusoidField/BlobField/StepField)
  → 解析相似变换 + 分层遮挡 + 局部非刚性 + 运动模糊       [P2 §4 合成数据]
  → CFA(4 相位) / pseudo-RAW(3 通道, HDR 派生) 采样 + shot/read + 共享黑电平块噪声
  → RawSensorFrame（完整传感器 xy、相位、有效性、来源身份）  [契约 §3 SourceKey/MeasurementRecord]
  → CorrespondenceEstimator（SEA-RAFT RAW 适配：fnet 4/3、cnet 8/6、无 /255、无 ImageNet 自动权重）
  → CorrespondenceMap（source_xy 为参考→历史源的 pull 坐标；4-tap SourceKey + 权重；无环绕）
  → WindowMotionProvider → DbsrFeatureAdapter.step(offsets_xy=…) → 因果窗口融合 → 256 token/相机
  → VisualTokenPostprocessor(in_dim=512,width=2048,tokens_per_camera=256) → 视觉前缀

撤回/重算路径（P1 验收硬要求）：
  MeasurementLedger（清除后的视图）
  → LedgerCorrespondenceBuilder / LedgerNoiseModelBuilder（只读 view，从保留 RAW 重算）
  → full_replay(correspondence_builder=…, noise_model_builder=…) → fit_structure → ReadoutSnapshot
```

与原要求的对应：SEA-RAFT 完整特征提取/相关金字塔/迭代更新保留（仅声明式改动）；CFA 与 pseudo-RAW 两套 stem 单独配置并各自真实训练；坐标单位换算与 pull 语义在 `raw_correspondence.py` docstring 与测试中固定；噪声用 `Var(y|mu)=a·max(mu,0)+b` + 共享黑电平块，相关黑电平误差走 P1 联合因子（dense），拒绝未审计结构；插值方差为 `aᵀΣa`。

## 2. 真实执行命令、来源身份、环境与资源

### 2.1 上游固定与 vendor 差异

固定 commit `9137517ba24e628442aec097d3afe71d03503b75`，BSD-3-Clause。核心文件上游 sha256 与 P0 固定锁（`configs/encoders/n01_full/upstream_lock.json`）逐一相等；本轮新增取回的 `core/layer.py`、`core/utils/utils.py` 在固定 commit 下重新拉取后与 provenance 声明逐字节一致（`580f4cced2b5f355…`、`04d88de0f161db56…`）。取回路径：`raw.githubusercontent.com` 连接重置（curl 35），降级用 GitHub contents API（与 P0 记录一致）。

声明式改动（全部写入 `provenance.json.changes`）：相对导入、去 `huggingface_hub`/`PyTorchModelHubMixin`、输入通道参数化、删除 torchvision ImageNet 自动加载分支、删除 `2*(x/255)-1`、`scipy` 局部化、格式空白。`corr.py`/`update.py`/`layer.py` 的可执行 AST 与上游等价（3.10/3.11/3.12 三解释器一致的规范化摘要；见 §3.3 跨版本修复）。

### 2.2 本地与远程环境

| | 环境 |
| --- | --- |
| 本地 | Python 3.12.3，torch 2.9.1+cpu，NumPy 2.5.1；全部检查 CPU/FP64 |
| 远程 encoder | `/workspace/owac/rawvla/envs/encoder/bin/python` = 3.11.16 / torch 2.9.1+cu128 / CUDA 可用（RTX PRO 6000 Blackwell 97,887 MiB） |
| 远程 policy | `/workspace/owac/rawvla/envs/starvla/bin/python` = 3.10.21 / torch 2.7.1+cu128 |

远程 run 根：`/workspace/owac/rawvla/research/n01-v1/runs/p2-motion-20261014/`（`source/`、`source-v2/`、`source-v3/`、`fit-check-v2/`、`select-v2/`、`seed-101/`、`receipts/`）。`/workspace` 非持久卷，关键小文件同步回本地 `research/experiments/results/n01_full/P2/`。

### 2.3 源包与远程 hash 校验

| 快照 | 文件数 | 远程逐文件比对 | verify |
| --- | --- | --- | --- |
| `source`（phase A，HEAD 01e0316 + 波次 1） | 172 | 172/172 matched | passed |
| `source-v2`（+ 波次 2，含训练/评测/motion_builders） | 179 | 179/179 matched | passed |
| `source-v3`（+ D1/D2 修复 + rgb3 配置） | 见 §3.4 | **PENDING** | **PENDING** |

本地：`make check` 通过；`uv run --locked pytest -q` **682 passed, 2 warnings**（P1 基线 495）。

## 3. 结果

### 3.1 本地工程结果

| 检查 | 结果 |
| --- | --- |
| `make check`（ruff check / format --check / compileall） | passed（275 files formatted） |
| `uv run --locked pytest -q`（全仓） | 682 passed（P1 基线 495 + P2 新增 187） |
| `tests/encoders/n01_full` | 338 passed |
| 独立验收入口 `test_p2_acceptance.py` | 53 passed，xfail 计数 0 |
| P1 回归（`test_p1_acceptance.py`+`test_replay.py`+`test_foundation.py`+`test_statistics.py`） | 61 passed |
| 本地 `--mode raw_adapter --smoke`（CPU stub 路径） | receipt `acceptance=false`、39 checks、failures=[]、图像塔 0 调用 |

真实数值证据（本地，CPU）：
- 生成器解析真值：`translation_subpixel` 的 `x+0.25`/`y-0.75`/`x-0.5` 与 estimator 输出逐格一致（pull 方向、packed↔sensor ×2、四相位偏移）。
- 噪声：`Var(y|mu)=a·max(mu,0)+b`；单样本块边际 = `cal.variance`（D2 修复后）；曝光增益 `v'=black+g(v−black)`、`var'=g²var`。
- 训练：最小配置全链路 `fit-check`/`select`/`train`/`--resume` 真实跑通；`fit-check`（shipped 配置，256 updates，CPU）13.0 → 2.4，六项诊断全过、六组梯度非零。
- rgb3 臂：独立配置 + 独立 256-update fit-check（18.4 → 2.4）；`stem-report` 实测参数差 **−9,408**（cnet −6,272、fnet −3,136，其余 0）与解析预期逐项相等，唯一形状差异为 `fnet.conv1.weight`/`cnet.conv1.weight`。

### 3.2 独立验收（与实现者隔离）

`tests/encoders/n01_full/test_p2_acceptance.py` 53 项，覆盖 9 个面：坐标/单位、插值方差、风险语义、噪声/曝光、数据与 GT 隔离、训练语义（mask/单位/判据/恢复）、adapter 与预检、ledger 重算与撤回、预检收据诚实性。16 个变异实现（在 `/tmp` 副本上打变异，从未进入工作树）全部被至少一条断言击败；每项断言都给出"会被什么错误实现击败"。

发现并修复的真实缺陷（详见 §3.3）：
- **D1 major**：`decide_platform` 旧实现把"恒定/持续恶化"的 validation 曲线判为 `validation_plateau`。
- **D2 medium**：`build_noise_model` 对角分支漏掉已审计共享块方差（单样本块系统性低报 σ_b²）。

### 3.3 缺陷、修复与回测

| 缺陷 | 证据（最小复现） | 修复 | 回测 |
| --- | --- | --- | --- |
| D1 平台判据 | 12 点恶化曲线与恒定曲线均判 `validation_plateau`；根因 `max(0, best−value)` | 新增 `real_improvement_over_run`（`first−best > delta_val`）与 `tail_within_delta_val_of_best`（`|final−best| ≤ delta_val`），逐条写入 reasons | 恒定/恶化/先改善后恶化 → `budget_exhausted`；慢速改善（每步 0.5×delta_val）→ 平台；R1/R2 变异证明断言非恒真 |
| D2 噪声边际 | 单样本块 `variance_of(k)=7.0` vs `cal.variance=7.25`；有兄弟样本时 7.25 | 边际统一为 `base_i + σ_b²(有键)`，对角/稠密两分支逐样本一致 | 7.0 → 7.25；`dense` 对角与 `diagonal` 一致；R3 变异证明断言非恒真 |
| 跨版本 | 远程 3.11 下 AST provenance 断言失败（3.12 `ast.dump` 多 `type_params`） | 自建规范化 AST 序列化（跳过 `type_params`，其余字段全输出），重算 provenance 摘要 | 本地 3.10/3.11/3.12 三解释器摘要一致；变异函数体/加方法可被抓住、改写 docstring 不触发；远程 3.11 下 277/277 通过 |

修复不改变训练轨迹：`decide_platform` 只读 validation 历史、不参与前向/反向/LR/RNG；`build_noise_model` 在训练路径无调用（训练只 `apply_exposure_gain`）；`scale` 标签不进 `core_fingerprint`。

### 3.3b 远程集成阶段发现的缺陷（A–E）

全部由真实环境/真实数据暴露，非本地可预知；修复后均在服务器上重验。

| 缺陷 | 症状与最小复现 | 修复 | 回测 |
| --- | --- | --- | --- |
| A `_load_pretrained` | torch 2.7.1 `weights_only=True` 拒绝含 NumPy RNG 的训练 checkpoint：`UnpicklingError … numpy._core.multiarray._reconstruct` | `_load_pretrained_payload`：先 strict；失败时**仅在文件字节含本项目 schema 标记**时重试 `weights_only=False`，随后校验 schema/`model`/`core_fingerprint`；非本 schema 一律拒绝；`pretrained_report` 记录 `weights_only_fallback/sha256/schema` | 同形 payload 修复前 `UnpicklingError` → 修复后 `fallback=True, schema=owac.n01_full.p2_motion_checkpoint.v1, loaded_tensors=276`；第三方 schema 被拒 |
| B preflight 窗口计数 | `MeasurementWindowBatch` 无 `unique_frame_count()`（该 helper 只在单窗类型上）：`AttributeError`，identity 预检 exit 1 | `_window_unique_counts()`：按 identity=(episode_id, frame_id) 逐成员计数，`unique_matches_window` 改为逐成员 bool 列表并记录 `unique_frame_identity_rule` | 负控换回旧表达式立刻以同一 `AttributeError` 失败；真实 collate 批次端到端通过 |
| C estimator 设备混用 | CUDA 上 `validity[:, ii, jj]`（CPU mask × CUDA 索引）：`indices … not on the same device`，GPU 评测 exit 1 | 后处理统一到 `flow.device`（`validity.to(device)`、origins/arange/coords 同设备），返回 map 全同设备并写入 docstring | 新增 CUDA 用例在 GPU 机真跑（`-k gpu` → 3 passed / 0 skipped）；spy 断言旧表达式 `.to()` 调用数为 0 |
| D evaluation 设备混用 | `_collect_calibration_samples`：`flow`(CUDA) − `target.channel_flow`(CPU) → `found at least two devices`，三 seed GPU 评测 exit 1 | `_align_to` 以模型输出 device 为准；全模块扫描后一并修掉 `_pair_error`/相位一致性/`map_proxy` 边界/`arm_map`/风险覆盖共 5 处 | `meta` 设备等价复现证明删掉对齐即报错；CUDA 端到端用例通过 |
| E 预检 stem 配对 | cfa4 checkpoint 配 rgb3 窗口：torch `size mismatch for cnet.conv1.weight [64,8,7,7] vs [64,6,7,7]` | 裁定**配对规则**：真实 3 通道窗口 ⇒ rgb3/3/6 estimator（禁止给 3 通道帧伪造第四测量）；并新增加载前 stem 推断校验，报出两侧通道、协议规则与建议 | 故意错配 exit 1、消息含两侧通道与规则、输出目录 0 文件；正确配对 exit 0 |

### 3.4 远程正式训练（第一段：3 seeds × 100k）

短任务（已完成，`source-v2`）：

| 项 | 结果 |
| --- | --- |
| `fit-check`（256 updates，GPU） | passed；loss 13.039 → 2.414；512 samples；96.6 s；六项诊断全 True；六组首步梯度非零；`parameter_hash_after` 记录 |
| `select`（4 候选同预算 256 updates，validation 选择） | winner `lr=2.5e-4, iters=4`，val EPE **1.2079** sensor px；其余：2.5e-4/8 → 1.4641、5e-5/4 → 2.7207、5e-5/8 → 1.8965；全部候选参数更新量 > 0；test 未使用；总 wall 476.4 s |
| 显存探针（batch 1/2/4，256×256 传感器） | peak reserved 566/650/762 MiB → 配置默认 batch=4 安全 |
| 吞吐分解（20 步） | 数据/渲染 p50 75.4 ms（15.6%）+ 前向反向 407.1 ms（84%）→ 0.47–0.48 s/update；设备被其他项目共享（util 91–99%，free 12.4–15.6 GiB） |
| rgb3 dev 臂（256 updates，`source-v3`） | fit-check passed（18.746 → 2.429，88.7 s，六项诊断全 True，六组梯度非零）；train 256 updates：loss 8.462 → 4.438，peak 723.9 MiB，`platform=budget_exhausted`（reasons: fewer than two validation points，符合 dev 规模预期）；`stem-report` 参数差 −9,408 与解析逐项相等 |

正式段（待回填）：

| seed | 状态（修复后判据） | wall / GPU 小时 | 平台判定与理由 | val EPE best / final（sensor px） | best checkpoint sha256 |
| --- | --- | --- | --- | --- | --- |
| 101 | **完成** | 13,990.8 s / **3.886** | `validation_plateau` @ 82,000（reasons: all satisfied；run_improvement 0.70487，best_update 82,000） | 0.069852 / 0.069852 | `012cad03a79df7fc42e93c17a8e7545e51a3bcdb2f414e80bab8b30bf44a7141` |
| 211 | **完成** | 4,749.3 s / **1.319** | `budget_exhausted` @ 100,000（reason: `tail_within_delta_val_of_best`） | 0.0758537 @91k / 0.089877 @100k | `8729c97bf9899a69e03fe48f6b382a55d836d332443de3897bf4df61e36729b4` |
| 307 | **完成**（断点续训） | ~4,686 s / **1.302** | `validation_plateau` @ 99,000（reasons: all satisfied） | 0.082897 @83k / 0.083346 @99k | `3d6c27c3778c9cbca0e22967c72db6635073a93504e76160d4356fbe4ccd2e38` |

seed 101 是 D1 修复价值的直接证据：source-v2 旧判据在 **81,000** 判 `validation_plateau`；source-v3 重算判 `budget_exhausted`（尾部漂离 best）→ 按授权 `--resume last.pt --max-updates 100000` 续训 1,000 updates 到 82,000，修复后判据方才接受 plateau。旧 receipt 保留为 `seed-101-receipt-pre-continuation.json`。

三种子均以 `--max-updates 100000` 上限运行；101/211 全程无 OOM、无进程被杀。吞吐：清理竞争前 0.41–0.70 s/update，清理后 **0.0475 s/update（21.06 updates/s）**；seed 101 总均 0.171 s/update（含被争用时段）。

### 3.5 评测与两平台 `mode=raw_adapter` 预检

| 项 | 结果 |
| --- | --- |
| 三 seed **GPU** 评测（`--device cuda`，test 32 pairs） | sea_raft EPE **0.08689 / 0.08930 / 0.09497**；`no_alignment` 1.92132、`old_simplified` 2.98518（三 seed 逐位一致，不依赖权重）、`oracle` 0.0（单列 `used_as_inference_input=false`） |
| GPU 时延与显存（end-to-end `estimate()` 单对已到达帧，iters=4，19,673,284 参数，repeats=5） | p50 **6.790 / 6.840 / 6.803 ms**；p95 **6.811 / 6.895 / 6.816 ms**；peak device **108,712,960 B = 103.7 MiB** |
| 设备自洽 | CPU 与 CUDA 的 sea_raft EPE 只差 3e-5–7e-5；CPU p50 41.9 ms 仅作历史对照，不作 GPU 指标 |
| 风险分档 | 四个 bucket 齐全（fast_hard 最好 0.027–0.032；current_frame_anomaly 0.067–0.074；slow_structural_change 0.051–0.099；fast_alignable 0.111–0.125） |
| 静态误拒 | validation、容差 0.5 px：三 seed 均 rate 0.0，mean predicted displacement 0.188/0.196/0.208 px |
| 风险校准表分区 | `fitted_split=calibration`；`regions={fitted_on: calibration, validated_on: validation, reported_on: test, test_used_only_for_reporting: true}` |
| motion 路径预检（两平台，`--crop 128`，rgb3 dev checkpoint） | exit 0；`p2_motion_status="motion_provider"`；`offsets_shape=[1,8,2,128,128]`；图像塔 0 调用；`p3_support_applied=false`；`unique_frames_window`/`unique_new_measurements_window`/`adapter_unique_lifetime` 为按成员列表、`unique_matches_window` 全 True、`unique_frame_identity_rule="(episode_id, frame_id)"`；`weights_only_fallback=true`、schema 正确、checkpoint sha256 `133eb47d…` |
| identity 对照 | crop-128 exit 0；crop-32 历史版保留（两版并列） |
| 故意错配诊断 | cfa4 checkpoint + rgb3 窗口 → 加载前明确报错（含两侧通道、协议规则与建议），**输出目录 0 文件** |

披露：预检用的 rgb3 motion 模型是 **256-update dev 臂**（`run_scale=dev`、`budget_exhausted`、update=256），只证明接口/梯度/单次到达语义连通，**不构成对齐质量结论**；正式质量证据是上表 CFA 三 seed 评测，两者分别陈述。motion 路径需要 `crop≥128`（rgb3 下 128×128 packed），这是 vendored 4 层金字塔的**几何下限**（训练配置 notes 早已声明、rgb3 臂训练几何即 128×128），不是为通过检查而放宽；`crop=32` 在 rgb3 下几何上不可能满足。是否用正式预算补一条 rgb3 臂（当前吞吐约 4 GPU 小时）留待研究负责人决定。

### 3.6 基础设施事件

| 时间（UTC） | 事件 | 处置 |
| --- | --- | --- |
| 15:20–16:32 | 实例 `107.209.104.125:44327` 失联（SSH/TCP 22/44327 均超时；本地网络对照正常：github.com/api.github.com:443、8.8.8.8:53、ICMP 均通） | 判定实例侧故障；seed 307（15:06:31Z 启动、13,000 updates 处被切断）被中断；守护脚本每 10 分钟只读重试并留痕 `remote-runner-outage-20261014T1520Z.json`（62 min、67 条探测）；16:32Z 恢复后先只读勘察（run 根完整、`/workspace` 未 recycle、307 `last.pt` 完整），再 `--resume last.pt` 精确续跑至 99,000，未从 step 0 重来、未重跑 101/211 |

已同步回本地的证据（46 个文件 / 964 KB，不依赖实例存活）：三 seed 训练 receipt 与平台重算、307 断点续训证据、三份 CUDA eval receipt 与风险校准、两份 identity 对照、motion 预检 receipt、断网留痕、source-v3/v4/v5 整树比对、失败日志原文与最终汇总 `remote-runner-p2-final-handoff.json`。

## 4. 接口交接

- **可加载 checkpoint**：`<run>/seed-<s>/checkpoints/{best,last}.pt`（内嵌 `core_fingerprint`、seed、iters、base_lr、优化器/scheduler/scaler、Python/NumPy/CPU/CUDA RNG、sampler permutation+cursor、已消耗样本数、`val_history`、`lr_drops`）。`core_fingerprint` 只覆盖 config 的 model/data/schedule/optimizer/selection，不含源码 hash；config 未变时换修复后的源码可精确续训。
- **模型配置**：`configs/encoders/n01_full/p2_motion_train.json`（CFA 正式）、`p2_motion_train_rgb3.json`（3 通道 dev）；`CorrespondenceEstimator(mode=cfa4|rgb3, depth, iters, normalization, risk_calibration)`。
- **支持/校准版本**：`noise_block_id` 按帧；`calibration_version=p2-synth-cfa-v1` ⇒ `noise_version=n01-noise-p2-synth-cfa-v1`（域绑定值）；`mapping_version` 由 provider/builder 声明（`p2-motion-window-v1` / `p2-ledger-correspondence-v1`）。
- **下一阶段命令**：`python -m owac.training.n01_motion {fit-check,select,train,stem-report}`、`python -m owac.evaluation.n01_motion`、`python -m owac.runtime.n01_full_policy_preflight --mode raw_adapter --platform libero --platform robotwin2 --motion-checkpoint <best.pt>`。
- **已知差异**：`RawSensorFrame.validity` 是单一布尔掩码，饱和/低于黑电平/出场景未分型（P1 `MeasurementRecord.validity_kind` 才区分）；CFA 的 ledger 几何是声明式（`raw_frame_from_ledger` 用调用者给的 pattern/phase/origin），要把 `CfaPlaneStack.global_sample_xy` 完全接进 ledger 构建侧留给 P4；预检的 motion 路径需要 `crop≥128`（vendored 金字塔几何下限）；rgb3 **正式预算**臂未训练（只有 256-update dev 臂；当前吞吐下约 4 GPU 小时可补）。
- **未验证项**：真实 SEA-RAFT 官方权重（本轮按用户决定"从零训练"不使用）；远程序列上的覆盖统计；P4 域支持/重编码；P3 时间规则；闭环。

## 5. 状态字段

- `implementation_status`：**PASS_P2**（全部模块、配置、测试、内部独立反例验收（53 项 + 16 变异）、D1/D2 与远程 A–E 修复回测，以及两平台 `raw_adapter` 真实预检与故意错配诊断均通过；本地全仓 709 passed / 2 skipped，`make check` 通过）
- `training_status`：**P2_FIRST_SEGMENT_COMPLETE**（远程 256 拟合 + 两档 LR 选择 winner `lr=2.5e-4/iters=4`；第一段三种子全部结束：101 plateau@82k、211 budget_exhausted@100k（尾部漂移）、307 plateau@99k；**非**充分训练结论，续训与 rgb3 正式臂未授权）
- `effect_status`：**NOT_APPLICABLE**（本阶段不产生 N01 机制增量或闭环结论；对齐质量只在合成 CFA test 分区报告，真实数据只证明接口连通）
- 资源：P2 远程 GPU 时间合计 **约 6.88 小时**（训练 6.507 = 101 3.886 + 211 1.319 + 307 1.302；select+fit-check 0.159；rgb3 dev 臂 0.057；显存探针 0.064；评测/预检/回归验证约 0.09）。设备为 RTX PRO 6000 Blackwell 单卡（共享）。

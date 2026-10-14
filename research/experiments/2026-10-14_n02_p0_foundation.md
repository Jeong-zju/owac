# N02 v1.1 P0：建立可用于正式开发的基础（2026-10-14）

> 历史交付记录：本文原先的“P0 完成”判定经独立复查不成立。保留全部原始数字和收据供追溯；修补后的当前结论见[独立验收记录](2026-10-14_n02_p0_independent_acceptance.md)。尤其不能继续使用旧 manifest 或把原 forward profile 当成训练更新画像。

- 分支 / 代码 SHA：`lzh/n02-measurement-completion`，`843be674c35e8e058a364154e1c27a4b26ded2e4`（本轮全部改动**未提交**，工作树保留；开工快照见 `outputs/n02-full/p0/preflight/`）。
- 目标运行时：`outputs/n02-full/runtime/py310-cu128/bin/python` = Python 3.10.19 + torch 2.11.0+cu128 + CUDA 12.8 + cuDNN 9.19 + numpy 2.2.6。
- CPU / 静态检查：根 `.venv` = Python 3.12.12 + torch 2.9.1+cpu。
- 本地硬件：NVIDIA RTX 5090 32 GB（sm_120，驱动 595.99.02），磁盘可用 458 GB。
- 数据主机（本轮新接入）：`ssh -p 44327 root@107.209.104.125`（vast.ai 容器 `2566cc983a37`），NVIDIA RTX PRO 6000 Blackwell Workstation Edition 97,887 MiB（驱动 595.84）；该机另有 10 个 compute 进程在跑**冻结基线评测**，本轮全部只读查询，未触碰。
- 收据目录：本地 `outputs/n02-full/p0/`（gitignored）；小型清单归档 `research/experiments/results/2026-10-14_n02_p0_foundation/`。
- **原交付机器判定（已被独立复查撤销）**（`stage_decision.json`，三路证据齐备）：`implementation=pass`、`numerical=pass`、`behavior=not_run`、`closed_loop=not_run`，`report.status=ok`。
- 执行方式：调度 Agent 冻结接口 + 5 个并行执行者（core / data / synthetic / parity / runtime）；调度 Agent 负责复审、集成、真实缓存审计与本文档。

## 1. 本轮原要求

按 `docs/prompts/n02_v1_1/P0_FOUNDATION.md` 完成 P0：建立 `n02_full` 命名空间、来源锁与 GPU 运行时；冻结四分区与数据约定；建设受控合成生成器；按顺序复现 ConvNeXt V2 / FCMAE / SAVi / dNRI / Theseus&Gen 参考算子；在真实可执行模块上做 200 步资源画像；交付可运行 CLI、清单与报告。**本轮只做 P0，停在交接点，不自动启动后续训练。**

追加要求（同日）：用户给出数据主机后，核实 RawVLA NPZ 是否在该机，并对真实缓存执行 P0 §3 的审计。

## 2. 实际实现与文件

| 文件 | 行数 | 内容 |
| --- | --- | --- |
| `src/owac/frameworks/n02_full/{__init__,config,state}.py` | 9 / 337 / 690 | 冻结契约：版本化配置、测量账本（去重/校准版本/部分 reset）、ObjectiveKey、ObservationBundle/InferenceInput/TrainingTargets、EntityState/WorldHypothesis、StageResult；顶层不 import torch/numpy |
| `src/owac/frameworks/n02_full/data.py` | 2099+ | `SourceIdentity`/`PartitionManifest`/`build_partition_manifest`/`verify_partitions`/`AnchorTable`/`ManifestWindowDataset`/`audit_dataset_root` |
| `src/owac/frameworks/n02_full/synthetic.py` | 2863 | `SensorSpec`/`ShapeSpec`/`SceneSpec`/`MeasurementModel`/`EventSchedule`/`EvaluationTruth`/`SyntheticSequence`/计数与成本画像 |
| `src/owac/frameworks/n02_full/components/*.py` | 335+516+562+381+379+284+1189+239+813+6 | ConvNeXt V2 RAW 适配、FCMAE（含 mask-aware dense 对照）、SAVi、dNRI、LM 参考、SMC 参考、`parity_suite.run_all`、`profile_targets.build_profile_targets`、上游加载器 |
| `src/owac/frameworks/n02_full/components/upstream/**` | 28 个 `.py`/`.jl` | 22 个审计文件 **逐字节** 拷贝 + 6 个同 commit 补充依赖（2 个 SAVi 文件为 lock 标注的缩减版）；`components/licenses/` 9 份 LICENSE |
| `src/owac/evaluation/n02_full/p0.py` | 1482+ | `inventory`（含 `--max-files`/`--evidence-kind` 透传与 `audit_limits`）/`parity`/`profile`/`report`；缺失上游一律 `not_run` + reason + exit 0 |
| `src/owac/runtime/n02_full_probe.py` | — | 真实环境探针（版本、显存、matmul/conv2d 精度），torch 延迟 import |
| `configs/encoders/n02_full/{p0.json,README.md,third_party.lock.json}` | 84 KB lock | 冻结配置（K=8/N=16/D=512、depths/dims、种子 1103/2207/3301、四分区 80/10/10、profile 200 步）；lock = 52 项 + 6 项补充 + `weights_downloaded:false` |
| `configs/backends/n02-full-runtime/{pyproject.toml,requirements.lock.txt,README.md}` | — | 独立 py3.10+cu128 运行时定义与锁（锁文件已用全新 venv 逐字节重建复验） |
| `scripts/n02_p0_real_cache_audit.py`、`scripts/n02_p0_cache_metadata_scan.py` | — | 真实缓存审计/元数据扫描的薄驱动（只读，数据主机上运行） |
| `tests/encoders/n02_full/`（12 文件 + conftest）、`tests/unit/test_n02_full_probe.py` | — | contract fixture 测试 + 真实探针测试 |
| `pyproject.toml` | +3 行 | 调度者唯一直接改动：`[tool.ruff] extend-exclude` 排除 vendored `components/upstream`，使 `make check` 覆盖本项目代码且不 lint 上游原件 |

## 3. 复用来源及差异

- **复用不复制**：`owac.core.encoder_contract`（`MeasurementFrame`/`EmbodimentContext`/`VisualTokens`/`CameraSpec`/`ComputeBudget`/`state_tensor_bytes`）与 `owac.raw.rawvla_sequence`（`FORMAT_SPECS`、`WindowOptions`、`RawVLASample`、`cfa_plane_positions`/`cfa_plane_values`/`apply_cfa_calibration`、`sensor_variance`、`normalise_*`、`discover_npz_files`）。`data.py` 只负责分区/锚点/审计，**不复制** RAW 数值转换与动作协议。
- **禁止二次分区**：`ManifestWindowDataset` 直接消费 manifest 选定的 `episode_paths`；测试用 `monkeypatch` 断言 `compute_split` 在该路径零调用，并以旧 loader 作阳性对照。
- **上游锁定**：52/52 缓存文件 sha256 = 审计清单 = 固定 commit 的 codeload 归档内同路径文件，三方一致；22/22 vendored 与缓存逐字节一致（复审抽查 `convnextv2/models/convnextv2.py`、`savi/modules/attention.py`、`dnri/models/dnri.py` 的 sha256 相同）。差异逐项写在 lock 的 `port_diff`。
- **旧资产未动**：旧 `frameworks/n02`、`SplitManifest`、`MeasurementBuffer` 与旧 16 场景生成器只作回归；`tests/encoders/n02` 265 passed / 4 skipped。

## 4. 实际命令（可复现）

本地脚本：`research/experiments/results/2026-10-14_n02_p0_foundation/reproduce.sh`。关键命令（全部真实执行）：

```bash
cd /home/jeong/zeno/owac/worktrees/n02-measurement-completion
export PATH="/home/jeong/.local/bin:$PATH"
make check                                     # exit 0
.venv/bin/python -m pytest -q                  # 631 passed, 6 skipped
PY=outputs/n02-full/runtime/py310-cu128/bin/python; export PYTHONPATH=src
$PY -m compileall -q src/owac                  # exit 0（py3.10 语法/标准库证据）
$PY -m pytest tests/encoders/n02_full -q       # 248 passed, 1 skipped
$PY -m owac.runtime.n02_full_probe --device cuda --output outputs/n02-full/p0/runtime/probe.json
$PY -m owac.evaluation.n02_full.p0 {inventory,parity,profile,report} ...   # 四子命令
```

数据主机（真实缓存，只读；只写 `/workspace/owac/n02-p0-audit/` 这个独立目录）：

```bash
ssh -p 44327 root@107.209.104.125
cd /workspace/owac/n02-p0-audit            # 代码通过 tar-over-ssh 送入，11 MB
export PYTHONPATH=src
R=/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench
/venv/main/bin/python scripts/n02_p0_real_cache_audit.py --root $R --output outputs \
    --hf-metadata /workspace/owac/rawvla/records/dataset-source-pinned.json --max-files 8 --curated-files 10
/venv/main/bin/python scripts/n02_p0_cache_metadata_scan.py --root $R --output outputs
for p in libero robotwin; do
  /venv/main/bin/python -m owac.evaluation.n02_full.p0 inventory \
    --config configs/encoders/n02_full/p0-remote-$p.json --output outputs/inventory_$p \
    --max-files 8 --evidence-kind real_cache
done
```

## 5. 数据、种子与资源

- **真实数据位置（本轮确认）**：RawVLA-Bench 在数据主机 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench`，HuggingFace `ToferFish/RawVLA-Bench` revision `8586c594fff9ce38580e29745f074e82e96e74f7`，4810 个文件下载完整（`records/dataset-download-complete.json`）。**旧记录里的 `/workspace/owac/rawvla/data/...` 路径在该机上不存在**（`ls` 报 No such file），实际布局多了一层 `source/benchmark_data/RawVLA-Bench/`。
- 计数（实测）：`data/libero/raw` 1771 个 npz / 96 GB、`data/libero/rgb` 1771 / 96 GB、`data/robotwin2/raw` 632 / 379 GB、`data/robotwin2/rgb` 632 / 95 GB（合计 705 GB）。
- **本地工作站仍无 NPZ**（12 条候选路径 + 全树检查）；本地 `p0.py inventory` 保持 `not_run`（cache_root 未配置），机器可读的真实 inventory 由数据主机产出。
- **合成数据**：种子由 `partition + scene_id` 派生（与模型种子 1103/2207/3301 分离）；本轮只做小样本生成与成本画像，**不声称** P1 的 100 万帧或 P2 的 10 万条序列已建成。
- **资源实测**：本地 RTX 5090（sm_120，32,088.5 MiB）；matmul fp32 vs CPU-FP64 误差 2.12472e-05、bf16 0.224387；conv2d fp32 5.72251e-06、bf16 0.148582；`matmul.allow_tf32=False`、`cudnn.allow_tf32=True`。数据主机 GPU 为 RTX PRO 6000 Blackwell（97,887 MiB），其 10 个 compute 进程属冻结基线，本轮未打扰。

## 6. 原始计数与区间

**测试**（真实计数）

| 范围 | 结果 |
| --- | --- |
| 根 `.venv` 全仓 `pytest -q` | **631 passed / 6 skipped / 0 failed**（skip 全为真实资产缺失：RawVLA NPZ 本地副本、StarVLA/π0.5、numba、无 CUDA 解释器） |
| 根 `.venv` `tests/encoders/n02_full` | **248 passed / 1 skipped** |
| py3.10+cu128 `tests/encoders/n02_full` | **248 passed / 1 skipped** |
| `make check` | **exit 0**（ruff check / format --check 256 files / compileall） |
| 旧回归 `tests/encoders/n02` | 265 passed / 4 skipped |

**真实缓存协议审计**（数据主机上运行；canonical 收据 `inventory_libero.json` / `inventory_robotwin.json`，`real_cache_evidence=true`，`max_files=8`，`truncated=true`）

| 检查 | LIBERO | RoboTwin |
| --- | --- | --- |
| 结论 | status=ok，**9 match / 1 declared_only / 1 not_declared**，problems=[] | status=ok，**8 match / 1 mismatch / 1 declared_only / 1 not_declared**，problems=[] |
| 测量键与相机次序 | `agentview_raw_uint8`+`wrist_raw_uint8`，uint8，**[T,256,256,3]**，键序一致 | `head/left/right_camera_raw_float32`，float32，**[T,240,320,3]**，键序一致 |
| 转换规则 | `float32/255.0`，无逐帧拉伸 ✓ | 原浮点尺度保留 ✓ |
| 当前本体 | `robot_states[t]` 8 维 ✓ | `actions[t]` 14 维作者代理 ✓（16 维 `robot_states` 仅末端量） |
| 动作监督窗口 | `actions[t:t+10]` ✓ | `actions[t+1:t+51]` ✓ |
| 正式归一化统计 | 加载器默认 `quantile` = 协议要求 ✓ | **mismatch**：加载器默认 `starvla_minmax`，协议要求显式 checkpoint MEAN_STD/zscore + epsilon |
| 实际执行动作窗口 | 仅声明（NPZ 无该字段） | 仅声明（NPZ 无该字段） |
| 时间戳 | **无 timestamp 键、单位未声明**（8/8 文件） | 同样缺失（8/8） |
| 噪声可用性 | 有（`metadata_json` 声明 sensor noise） | 有 |
| 语言/动作单位 | 通过 | 通过 |

**真实缓存元数据扫描**（全部 1771 + 632 个 episode，`metadata_json` 全部可解析、0 解析错误）

| 项目 | LIBERO | RoboTwin |
| --- | --- | --- |
| benchmark 标识 | `rawvla-bench-light-libero-v1-train-rerender-npz` | `rawvla-bench-light-robotwin2-v1-state-copy` |
| 照明域分布 | ExtremeLow 353 / Low 352 / **Normal 338** / Over 384 / ExtremeOver 344 | ExtremeLow 127 / Low 127 / **Normal 127** / Over 126 / ExtremeOver 125 |
| `lighting_ev` | min −5.49 / median 0.024 / max **+8.00** | min **−9.00** / median −0.008 / max +2.50 |
| 声明的传感器/噪声字段 | `lighting_scale`、`lighting_sample_unit`、`noise_seed`、`sensor_saturation_ev`、`agentview_stats` | `raw_white_level`(3.5)、`raw_sensor_read_noise`、`raw_sensor_shot_noise`、`lighting_applied` |
| 实测值域（各域抽样） | ExtremeLow p50=1 DN（max 6）、Low p50=2（max 13）、Normal p50=14（max 255）、Over p50≈36、ExtremeOver p50=50 | ExtremeLow p50=0.0、p99≈0.001、零值占比 12.5%；Normal 域 p50≈0.15 |

**真实四分区 manifest**（`build_partition_manifest` + `verify_partitions`，种子 1103，来源哈希取固定 revision 的 HF LFS sha256，未重算 705 GB）

| 平台 | 条目/组 | train | dev | calibration | 总帧 | 校验 |
| --- | --- | --- | --- | --- | --- | --- |
| LIBERO | 1771 / 1771 组 | 1411 文件 / 208,907 帧 | 178 / 26,120 | 182 / 26,104 | 261,131 | `disjoint=true`、`groups_single_partition=true`、violations=[]、读回逐项相同 |
| RoboTwin | 632 / 632 组 | 500 / 117,215 | 67 / 14,939 | 65 / 14,790 | 146,944 | 同上 |

manifest 条目实测样例：`illumination_group` 已按真实 `lighting_domain` 填充（Over 384 / ExtremeLow 353 / Low 352 / ExtremeOver 344 / Normal 338），`source_trajectory_id` 为“套件/集”级（variant 合并规则生效），`scene_id`/`acquisition_session_id`/`object_ids` 记 `unknown` 并在 `missing_fields` 声明（**不宣称 object-disjoint**）。本缓存 LIBERO 每个 episode 目录只有 1 个 variant 文件（实测 `variants per dir = 1 × 1771`）。

**上游算子 parity**（`outputs/n02-full/p0/parity/parity.json`；容差：前向 atol 1e-5/rtol 1e-4，梯度 atol 1e-5/rtol 1e-3，FP64 ≤1e-8）

| 检查 | canonical（py3.10, cpu） | 参考 venv（jax/flax/numba） |
| --- | --- | --- |
| 总计数 | **11 pass / 0 fail / 5 not_run** | **12 pass / 0 fail / 4 not_run** |
| ConvNeXt Base block / GRN 全空间 / 多尺度 | pass，max_abs **0.0**；远处扰动效应 6.396（两岸同值） | pass，0.0 |
| FCMAE patchify↔unpatchify | pass，**0.0**（3/4 通道 × 方形/240×320@p16/256×320@p32） | pass，0.0 |
| FCMAE mask-aware dense（命名对照） | pass，泄漏 0，有效位置统计误差 5.59e-09；零填充对照差 9.28e-01 | pass |
| SAVi SlotAttention / GRU | pass，前向 8.34e-07 / 2.98e-08；23 个参数梯度 worst abs 1.53e-05、worst rel 2.29e-04，违规 0 | pass |
| SAVi 有效 mask 扩展 | pass，无效 key 泄漏 0.0，全有效时与上游差 0.0 | pass |
| dNRI 因果一步 | pass，5.96e-08；**未来帧替换后已产生输出逐位不变** | pass |
| Theseus LM（自写参考，非上游代码） | pass，FP64 解析解误差 4.44e-15；完整目标否决 23 个"降残差"提议 | pass，3.97e-14 |
| particles SMC vs 枚举 | not_run（根 venv 无 numba） | pass，TV 均值 0.01215 / 最大 0.02329 |
| 20 步短训趋势 | pass，两侧 loss 序列逐步差 **0.0**（0.977544 → 0.002121） | pass |
| 上游稀疏 FCMAE / theseus-ai / Gen.jl / SAVi Processor | not_run（原因见 §7） | not_run（同） |

**200 步资源画像**（`profile.json`，本地 RTX 5090，fp32，200 updates + 20 warmup，每目标独立峰值口径）

| 目标 | 参数量 | mean / p50 / p95 秒每更新 | 吞吐 | 峰值 allocated / reserved |
| --- | --- | --- | --- | --- |
| `convnext_raw4`（Base，4 通道 CFA） | 87,694,848 | 0.004396 / 0.004366 / 0.004605 | 227.5 samples/s | 847.4 MB / 872.4 MB |
| `convnext_raw3`（Base，3 通道伪 RAW） | 87,692,800 | 0.004386 / — / 0.004570 | 228.0 samples/s | 847.2 MB / 872.4 MB |
| `fcmae_patchify_unpatchify` | 0 | 1.79e-05 | 55,763 /s | 12.7 MB / 25.2 MB |
| `fcmae_mask_decoder` | 4,756,992 | 2.10e-04 | 4,770 /s | 32.7 MB / 48.2 MB |
| `fcmae_full` | 92,449,792 | 0.016597 / — / 0.017049 | 60.3 /s | 967.3 MB / 985.7 MB |

**合成数据成本画像**（真实运行，`outputs/n02-full/p0/synthetic/sample_cost.json`）：CFA 8 序列 × 32 帧 × 2 相机 = 256 帧 / 512 测量，0.782 s（3.05 ms/帧，tracemalloc 峰值 33.7 MB）；RAW3 8 × 32 × 1 = 256 帧 / 256 测量，0.530 s（2.08 ms/帧，峰值 54.1 MB）。

**阶段判定**：`report.status=ok`；`stage_decision = implementation:pass / numerical:pass / behavior:not_run / closed_loop:not_run`（`data_hash=null`，P0 无数据哈希汇总）。初次本地运行（真实缓存接入前）为 `implementation:incomplete`，原件保留在 `results/.../local-first-pass/`。

## 7. 失败、未验证项与遗留问题

1. **RoboTwin 正式归一化统计 mismatch（P1 前必须解决）**：审计实测加载器默认 `starvla_minmax`，而 P0 §3 协议要求使用显式 checkpoint 的 MEAN_STD/zscore 与 epsilon。当前 loader 的通用默认**不能**直接用于正式训练配置。
2. **两类数据都没有时间戳键**（8/8 文件 `files_with_timestamp_key=0`，单位未声明）：时间来源只能是帧索引；跨设备/丢帧场景的时间语义仍不可验证。
3. **低照度域的表示损失（新发现，P1 需决策）**：LIBERO 的 `uint8` 伪 RAW 在 ExtremeLow/Low 域中位数只有 1–2 DN（`lighting_ev` 低至 −5.49），Normal/Over/ExtremeOver 域中位数 14–50 DN。若按协议 `FP32/255` 使用，低照度样本的有效强度信息极少；是否改用 `metadata_json` 声明的 `lighting_scale`/`lighting_ev` 作为已知标定（属允许的校准，不生成 RGB）必须由 P1 明确决策并单列对照。
4. **上游稀疏 FCMAE not_run**：`models/convnextv2_sparse.py` 依赖 MinkowskiEngine（PyPI 仅 sdist，隔离构建失败）。本轮以 **mask-aware dense** 命名对照替代并逐层验证有效坐标/归一化/梯度，明确标注计算开销不同于稀疏实现。
5. **theseus-ai not_run**：cp310 wheel 链接 CUDA 11（`libcublas.so.11` 等），cu128 下 `ImportError`；完整安装另被 `cholmod.h` 挡住。LM 收据来自自写 numpy 参考实现，不是上游代码运行结果。
6. **Gen.jl not_run**：Julia 未安装，只留源码语义笔记。
7. **SAVi Processor not_run**：flax 0.10.4 的 `nn.scan` 调用参考 jax 已删除的 `jax.api_util.debug_info`；SlotAttention/GRU/MLP 导出成功。SAVi 512 宽 / 16 槽 / 4 块 profile **未测**。
8. **集成目标结论**：无需变更，保持 Python 3.10 + CUDA 12.8；3.10 原生 venv 通过 PYTHONPATH 同进程加载本项目模块（E1 同进程可微要求满足）。**未声称**跨环境进程通信能传回动作梯度。
9. **遗留小项**：`components/parity_suite.py` 的 mask 扩展检查用 `float(tensor)` 处理 `requires_grad` 张量，触发 PyTorch UserWarning（应 `.detach()`）；不影响判定，留给 P1 顺手修。
10. **抽样边界**：真实审计为抽样（`max_files=8` + 10 个跨套件/任务/照明域的 curated 文件，`truncated=true`），字段/范围结论覆盖这些真实文件，**不是**对 2403 个文件的数值普查；manifest 覆盖全部 2403 个文件，来源哈希取固定 revision 的 HF LFS sha256（未重算 705 GB 内容）。
11. **复审中发现并已修复的 5 处缺陷**（均有复跑证据）：① `_runtime_info` 在 CUDA 可用时对 `device="cpu"` 调用 CUDA API 抛错；② `p0.py` 状态词汇表把 parity 的真实 pass 写成 `incomplete`；③ `--device cuda` 时模块从未搬到 GPU；④ 每目标峰值显存未隔离；⑤ `p0.py inventory` 无抽样上限，对真实缓存会读约 475 GB（已加 `--max-files`/`--evidence-kind` 并透传进 `audit_limits`）。

## 8. P0 交付清单逐项结论

| P0 要求 | 结论 | 证据 |
| --- | --- | --- |
| §2.1 P0 模块/配置目录/`p0.py`，`--help` 列四子命令 | **pass** | 四子命令真实执行 |
| §2.2 `third_party.lock.json`（含许可、差异、参考命令；代码与权重分开） | **pass** | 52 entries + 6 supplementary + `weights_downloaded:false`；52/52 三方 sha256 一致 |
| §2.3 独立运行时定义+锁+README；py3.10+GPU 集成目标；PYTHONPATH 加载；探针冻结 | **pass** | 锁文件逐字节重建复验；探针 4 行矩阵 |
| §2.4 Theseus/MinkowskiEngine 冲突与集成目标建议 | **pass**（结论=不变更） | `runtime_matrix.json` + 真实失败输出 |
| §2.5 资源盘点；远端只有连上才记录 | **pass** | 本地 `resources.json`；数据主机实测（GPU/磁盘/进程/数据路径），冻结基线未动 |
| §3 `PartitionManifest` + 显式分区 dataset；variant 同组；80/10/10；锚点表；禁二次分区 | **pass** | 真实 2403 条目 manifest：分区不相交、组不跨区、读回逐项相同；fixture 27 项覆盖锚点/时间边界 |
| §3 真实 NPZ 字段/协议审计 | **pass** | 两平台 `inventory.json`（`real_cache_evidence=true`）：LIBERO 9 match、RoboTwin 8 match + 1 mismatch（见 §7.1） |
| §4 合成生成器（2.5D、CFA/RAW3、事件、truth 隔离、留出、计数、成本） | **pass**（小样本） | 71 passed；成本画像真实 |
| §4 P1/P2 全量数据目标 | **not_run** | 仅声明参数与目标，未建成 |
| §5.1 ConvNeXt block/GRN/多尺度/FCMAE 往返（含 4 通道与矩形） | **pass** | max_abs 0.0；GRN 全空间效应 6.396 |
| §5.2 FCMAE 稀疏参考 | **not_run**（dense mask-aware 对照 pass） | MinkowskiEngine 安装失败原文 |
| §5.3 SAVi JAX↔PyTorch（含 mask 扩展） | **pass**（Processor 与 512 宽 profile not_run） | 前向 8.34e-07、参数梯度全过 |
| §5.4 dNRI prior/posterior/decoder 一步与在线因果性 | **pass** | 5.96e-08；未来帧扰动不改变已产生输出 |
| §5.5 Theseus 解/梯度 + Gen/particles 枚举与 SMC | **partial**：LM 参考 pass、SMC pass（参考 venv）；上游 theseus 与 Gen.jl **not_run** | FP64 4e-15；SMC TV 0.01215 |
| §6 容差先验证后冻结 | **pass** | `TOLERANCES` 冻结并落 receipts |
| §6 200 步 profile（参数/耗时/吞吐/峰值显存/有效batch/精度/GPU/软件） | **pass**（5 个已实现目标） | `profile.json`/`metrics.json`，三栏口径 |
| §6 九项完整性测试 | **pass**（真实 NPZ 一项已补做） | 见 §6 表 |
| §6 CLI 示例真实可运行 + 报告 + 可复现命令 | **pass** | `reproduce.sh`、`run_manifest.json`（含 sha256 与解释器路径） |
| P0 不自动启动 100k 训练 | **遵守** | 无训练进程、无定时任务、无提交；数据主机新增目录 11 MB，冻结基线未触碰 |

## 9. P1 可直接使用与交接条件

**可直接使用**：冻结配置 `configs/encoders/n02_full/p0.json`；`state.py` 契约；`data.py` 的 manifest/anchor/dataset 机制与审计器（含真实缓存收据与 `--max-files` 抽样能力）；`synthetic.py` 生成器；ConvNeXt V2 Base RAW 适配与 FCMAE（含 mask-aware dense 对照）；`parity_suite`/`profile_targets` 接口；py3.10+cu128 运行时与锁；`p0.py` 四子命令；**真实数据主机与路径、HF revision、LFS 哈希清单**。

**P1 开工前需要处理的条件（按优先级）**：
1. **数据接入**：在数据主机（或本地挂载同一缓存）建立 P1 训练运行目录，使用 `configs/encoders/n02_full/p0-remote-*.json` 形式指向 `data/{libero,robotwin2}/raw`；本地工作站仍无数据副本，跨机传输策略（只传清单/哈希，还是传 96–379 GB）需先定。
2. **RoboTwin 正式统计**：用显式 checkpoint MEAN_STD/zscore + epsilon 替换加载器默认 `starvla_minmax`，并把该决策写进冻结配置与测试。
3. **低照度表示决策**：明确是否使用 `metadata_json` 的 `lighting_scale`/`lighting_ev` 作为已知标定（分域报告），并保留 FP32/255 作为对照；不得生成 RGB 中间图。
4. **分区与照明域**：当前真实 manifest 已带 `illumination_group`；P1 需决定 dev/calibration 是否按照明域分层，并保留跨场景/物体留出的 unknown 声明。
5. 稀疏 FCMAE 路线决策（真编译 or 明确采用 dense mask-aware）；Theseus 若要用于 P2 需隔离 CUDA 11 环境或源码编译；SAVi 512 宽 profile 与 Processor 导出留到 P2 接入前。
6. 本文档 §7.9 的遗留小项（`float(requires_grad)` → `.detach()`）。

**边界声明**：P0 的 pass 仅表示工程与数值契约成立（parity、画像、分区机制、合成测量模型、真实缓存协议审计），**不表示**任何研究机制、具身能力或 π0.5 闭环结论成立；`behavior` 与 `closed_loop` 均未运行，P0 未启动训练。

# N01 v1.0 P0 独立验收与修补

2026-10-14；工作树 `worktrees/n01-evidence-integration`，复核提交 `57e0551`（及其父提交 `dab6800`）。

**裁定：原交付尚未完全满足 P0；本轮修补后，P0 工程验收通过，可以交接 P1。** `implementation_status=PASS_P0_AFTER_REPAIR`，`training_status=NOT_STARTED`，`effect_status=NOT_APPLICABLE`。本轮停在 P0，未启动 P1 或长训练。

依据是 [P0 原始要求](../../../docs/prompts/n01_v1_0/P0_FOUNDATION.md)、[共同契约](../../../docs/prompts/n01_v1_0/00_SHARED_CONTRACT.md)与完整方案。旧 [P0_report.md](P0_report.md) 和 [receipt.json](../results/n01_full/P0/receipt.json) 保留为原交付记录；本报告及 [acceptance.json](../results/n01_full/P0/acceptance.json) 是新的验收结论。

## 原交付缺口与实际修补

| 缺口 | 影响与修补 |
| --- | --- |
| RoboTwin 协议填 `stats_key=policy_norm_stats`、`action_stats_key=action`，实际文件是 `norm_stats/actions` | 按协议直接构造 loader 会报错；原 batch 脚本用了正确默认值，未发现此差异。修正协议，新增 `dataset_from_protocol`，让 split、统计路径及键名从同一协议进入真实 loader。 |
| `profile` 默认 factory 为 `pending_P4` | P0 的 GPU 数字来自服务器临时脚本，交付 CLI 不能复跑。新增 `owac.runtime.n01_full_profile:dbsr_factory`，只测已有完整 DBSR 64/9/512；8 帧走原 encoder，16/32 帧走同一 backbone/fusion 直接入口。未放宽旧缓冲区限制，未假装实现 P4。 |
| `inventory --strict` 只比较数量，未检查来源组或当前文件身份 | 增加来源组跨区、rows/sides 精确一致、RAW 相对路径集合、RGB 镜像路径集合、实时 size/mtime、inventory/loader-evidence hash、任务映射覆盖检查。新增反例测试保证同数量路径替换、来源跨区会失败。 |
| requirements 为 torch 2.7.1，supervisor 指向不存在的草图环境 | 修正为实际验证的 `torch==2.9.1+cu128` 与 `/workspace/owac/rawvla/envs/encoder`；记录完整 installed distributions。该 uv venv 无 pip，说明改用 `uv pip --python`。后续阶段依赖仍按上游锁另行部署。 |
| 任务层统计仅写在交接 prose | 重新读取全部 2,403 个 NPZ 的 metadata/actions，生成逐轨迹任务表，写入协议 hash。规定平台内按任务、轨迹分层重采样，训练 seed 作为交叉因素；各臂共用配对抽样索引。 |
| 自动成本汇总选最快的小尺寸点，未乘相机数/梯度累计；原估算又只列 5 个方法 | 改用明确的 128/8 测量点，7 臂 × 3 seed × 2 平台，LIBERO/RoboTwin 各 2/3 相机、累计 8。以原实测 46.073 ms 推算，20k/100k/300k 为约 215.0/1075.0/3225.1 GPU 小时；仅为 backbone 成本外推，300k 不代表授权增加 P4 上限。原 30-run 数字保留在 historical 字段。 |
| I/O 吞吐仍 pending | 从真实四区 loader 每区读取 20 个窗口，记录实际输出字节、耗时与 MiB/s。明确是预热后返回 RAW 张量的吞吐，不是物理磁盘带宽。 |
| “v5 227/227 一致”仅证明快照自洽 | 与 `57e0551` 比较时，201 个受版本控制文件一致，README 和 foundation 测试两处不同，另有 24 个 ignored 上游文件。新源包以当前修补文件为准，139/139 远程验证通过。 |

`SourceKey` 的非空身份、输入模型模式与存储编码分层、独立数据 seed 均可接受。环境位置不同本身不构成失败，调用配置指向错误才是缺口。未知许可继续明确未知；本次没有推送或公开分发。

## 原始要求与实测证据

| 要求 | 本轮独立证据 |
| --- | --- |
| 旧结果可追溯、保持原基线 | 旧源码审计 10/10、DBSR protocol-lock 4/4 hash 一致；未重跑旧 74k 矩阵。冻结 Default 的 summary 与 identity-audit 实读可用。 |
| 身份、配置、理论义务 | 契约/配置与针对性行为测试通过；T1–T6 的假设和反例存在，仍不宣称已证明。修正理论表中把 metadata 缺失等同测量缺失的措辞及陈旧的 P0 待实现标签。 |
| 四区真实 loader | 两平台所有四区从修正协议构造，paths 与 manifest 逐项相等，实际窗口及归一化动作有限。原分区不变：LIBERO 1228/175/206/162；RoboTwin 450/48/58/76。 |
| 数据身份与统计范围 | 全量扫描得到 40 个 LIBERO 模板、13 个 RoboTwin 任务；同任务内完全相同动作序列为 0。任务共享已披露，结论限同任务池的独立轨迹，不支持新任务、场景或相机泛化。 |
| 环境与资产 | Python 3.11.16 / torch 2.9.1+cu128；5 项资产 SHA256 重新流式计算均一致，包含策略大权重。 |
| 短 GPU profiling | 五个协议点均执行 3 次预热、20 次同步 forward 与 forward+backward；1,497,056 参数，梯度全有限且总范数非零。详情见下表。 |
| 源包与远程验收 | 最终 `source-v3` 139 文件与本地 bundle 一致；`inventory --strict --check-loader` 55/55，通过且 pending 为空；远程 P0 91 passed。 |
| 本地回归 | `make check` 通过，全仓 **439 passed**（原交付先复跑为 434 passed，本轮新增 5 个缺陷回归测试）。 |

| packed cells / history | 入口 | forward+backward p50 (ms) | peak allocated (MiB) |
| --- | --- | ---: | ---: |
| 64 / 8 | encoder | 10.113 | 757.4 |
| 128 / 8 | encoder | 46.305 | 3012.5 |
| 128 / 16 | direct | 106.924 | 5983.1 |
| 128 / 32 | direct | 218.969 | 11924.4 |
| 224 / 8 | encoder | 176.317 | 9216.8 |

以上均为单相机、microbatch=1、FP32、四通道合成 CFA 输入。原九点加 BF16 的历史测量保留，本轮通过交付 CLI 复跑五个声明点。真实三通道模型、SEA-RAFT、支持重放、Mamba、策略和读出桥成本尚未测量。

零位移输入使 `offset_feat_extractor.0.0.weight` 梯度为零，这是该卷积权重输入为零的结果。原 GPU 收据也只检查 `grad_any_nonzero`；不能将其转述成“每个参数梯度都非零”。本轮记录全部有限、总梯度非零及零梯度参数名称。一版过严的每参数非零断言失败已留档，经逐参数诊断修正；没有改模型来制造通过。

## 可复跑入口与交接

本地：`make check`、`uv run --locked pytest -q`。

远程新收据根为 `/workspace/owac/rawvla/research/n01-v1/runs/p0-audit-20261014/`，最终执行源码 `source-v3/`，独立 Python 为 `/workspace/owac/rawvla/envs/encoder/bin/python`。在该源码目录设 `PYTHONPATH=src`、`OMP_NUM_THREADS=2` 后运行：

```bash
python -m owac.runtime.n01_full_preflight inventory --strict --check-loader --output <new-inventory-directory>
python -m owac.runtime.n01_full_preflight profile --mode collect --output <new-profile-directory>
python -m owac.runtime.n01_full_preflight verify --bundle . --source-root . --report <new-verify-file>
python -m pytest -q tests/encoders/n01_full
```

机器可读结果：[总收据](../results/n01_full/P0/acceptance.json)、[四区与 I/O](../results/n01_full/P0/acceptance-inventory.json)、[GPU 原始点与成本](../results/n01_full/P0/acceptance-profile.json)、[139 文件 hash 与包验收](../results/n01_full/P0/acceptance-bundle.json)、[前序身份核查](../results/n01_full/P0/acceptance-prior-audit.json)、[数据身份](../results/n01_full/P0/acceptance-data-identity.json)。完整扫描、原失败日志及历史输出在 ignored `outputs/n01-full-p0-audit/`。

四个既有策略服务 PID 均未改变，本任务 GPU 进程已退出。最后一次 GPU 占用约 51,198 MiB，其中包括其他方向的 N02 进程；不能把全卡内存变化归因于本任务，也不能承诺独占 GPU。

P1 后续必须沿用任务映射、唯一来源与 replay 边界；P3 才冻结校准阈值，正式矩阵禁止携带 placeholder。P4 才建设真正的三通道与长历史 encoder。用户已有学术方案的三处标题改动保持原样，不纳入本次修补提交。

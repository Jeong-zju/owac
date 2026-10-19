# P0 冻结问题 数据和依赖

你现在执行 N01 v1.0 的 P0。读取[共同契约](00_SHARED_CONTRACT.md)、[实现审计](01_CURRENT_IMPLEMENTATION.md)、[远程说明](03_REMOTE_EXECUTION.md)和完整方案第 2、4、5、7–12 节。目标是交付后续能直接运行的工程/数据起点，不是再写一份抽象计划。只做 P0；不开始 P1 算法或正式长训练。

## 1. 按此顺序执行

1. 在 N01 worktree 记录 `git status --short --branch`、HEAD、dirty diff。核对旧 DBSR 与三种子结果的身份、配置和源 hash；用现有收据复核，不重新跑旧 74k 步矩阵。
2. 通过 `ssh -p 44327 root@107.209.104.125` 只读核查资源、数据和环境；遵照远程说明建立新的 `n01-v1` 目录和独立 GPU 环境。远程依赖、数据、路径真实可用后才写 `ready`。
3. 新建内部契约、四区 manifest、上游锁和协议；测试真实 loader 能消费四个 `sides`，无需另造 RAW 数值转换。
4. 用真实尺寸做小型 profiling 与序列读取：本阶段不是收敛训练，只采集吞吐、显存、帧数和 GPU 小时估计。
5. 保存 P0 报告、理论义务目录和下一阶段可用路径；更新 README/status 的当前入口，保留历史记录。

## 2. 明确的文件职责

下面均为拟新增文件：

| 路径 | 本阶段实现 |
| --- | --- |
| `src/owac/frameworks/n01_full/contracts.py` | 共同契约中的 dataclass、版本与来源角色；不写空 encoder 假装完整模型 |
| `src/owac/frameworks/n01_full/config.py` | 明确输入身份、basis、history、replay、对应、内容、训练预算的配置与 JSON round-trip |
| `src/owac/raw/n01_full_manifest.py` | inventory → 源轨迹分组 → train/calibration/validation/test；导出旧 loader 可读 `sides` |
| `src/owac/runtime/n01_full_preflight.py` | `inventory / profile / bundle / verify` 子命令，分开只读核查和打包 |
| `configs/encoders/n01_full/protocol_v1.json` | 冻结研究矩阵、数据、随机流、预算和统计规则 |
| `configs/encoders/n01_full/upstream_lock.json` | 精确提交、许可、所用文件 hash、依赖、配方、权重及预训练来源 |
| `configs/backends/n01-full-runtime/` | 独立 GPU requirements、运行环境说明和 supervisor 模板 |
| `data/manifests/n01-full-v1/` | 小型来源/分组/分区表；大数组不入 Git |
| `research/theory/n01_support_contract_v1.md` | T1–T6 的定义、假设、待证明命题、反例与后续代码/测试映射 |
| `tests/encoders/n01_full/test_foundation.py` | 身份、分区、manifest 消费、配置与打包闭包的实际行为 |

## 3. 数据决策

先读取两平台真实 NPZ 的 keys、dtype、shape、单位与 metadata（每平台至少覆盖不同任务/照明的少量文件），复用旧已验收的黑白电平和 action/state 归一化。检查 RGB/RAW 和照明衍生是否确属同一基础轨迹；路径规则不够时用 metadata 与人工可审阅映射表，不把文件 stem 相同自动当同源。

manifest 行至少有：相对路径、文件 hash 或可追溯库存 hash、dataset/input_mode、source_group、episode、camera、scene/session/sensor（缺失为 null）、照明、split、单位/校准版本。`source_group` 是基础轨迹身份，不含照明变体后缀。默认组比例 70/10/10/10；已有官方 train/test 优先保持，在 train 内划 calibration/validation。明确区分“无传感器 metadata”与“单一相机”。

给 `RawVLASequenceDataset` 传入实际 `raw_cache_root`、`stats_path`、`split_manifest`，并分别请求四区。核对实例 `paths` 恰等于 manifest 的那一区。现有 `_paths_for_split` 已按 `sides` 取值，不先删重写 loader。manifest 保存相对路径，执行时在新适配器解析为服务器绝对路径；四区都要有真实来源且零重叠。

生成数据新建 `n01-full-v1` 随机命名空间，源身份由命名空间和轨迹/帧/采样确定；旧 offset 不复用。把新正式 test 身份与已消费旧合成 holdout 对照留档。

## 4. 上游和资源选择

使用[参考导航](02_REFERENCES.md)和 [source_manifest.json](source_manifest.json)中的精确提交作为候选起点；本包的源码读取不代替你在服务器完成安装/数值验收。P0 固定 DBSR、SEA-RAFT、DSM-bocd；Mamba/BSRT 先锁源码与依赖可获取性，正式移植分别留到 P5/P6。保留 DBSR CC-BY-NC-SA 归属与 license 文件。

SEA-RAFT `ResNetFPN._init_weights` 默认可能加载 ImageNet；主线选择从随机/RAW 任务初始化，显式关闭该下载与权重加载。P2 再改输入层。记录准备用的作者配置名称和完整参数；不能只写“使用 RAFT”。

profiling 用完整 DBSR 64/9/512，空间候选 64/128/224 packed cells、历史 8/16/32、batch=1 起。每点预热后至少 20 次同步 forward/backward，报告 p50/p95、allocated/reserved/峰值显存、I/O 和有效 batch，记录相机数。只计已有可执行模块，SEA-RAFT 改造/支持重放尚未完成的成本单列未知。不能把原四通道模型直接 profile 为三通道。

P0 选择最小可用正式起点（首选 128、history=8），训练的有效 batch 由实测与梯度累计决定；CPU tiny 只作数值测试。20k/100k/300k 更新成本由 step time 估算，显式乘种子/方法/平台数量。服务器瞬时空闲不等于后续专用 GPU。记录仍在运行的服务和本任务分配显存。

## 5. 协议字段和验收

`protocol_v1.json` 至少包含：输入身份、坐标单位、四区 manifest/hash、数据/模型 seed、唯一来源策略、精确回放范围、模型与上游版本、B0–B3/N-T/N-S/N-TS、主指标、阈值来源、验证间隔、平台规则、预算上限、配对统计单元、多比较顺序、后端协议和阶段状态。P3 才能实测确定的阈值以 `pending_calibration` 表示，正式矩阵启动前必须冻结为具体值。

验收必须包括：同值不同曝光 ID 不同；重复包 ID 相同；相机/episode/相位不碰撞；同轨迹变体同组；四区真实 loader 消费一致；源包包括新 Python/JSON/license 且远程 hash 一致；新环境的 GPU tensor forward/backward 成功；冻结 Default 路径仍可读。

拟新增命令的目标形式如下，先实现再运行：

```bash
python -m owac.runtime.n01_full_preflight inventory --config configs/encoders/n01_full/protocol_v1.json --output outputs/n01-full/p0-inventory
python -m owac.runtime.n01_full_preflight profile --config configs/encoders/n01_full/protocol_v1.json --output outputs/n01-full/p0-profile
```

本地 `make check` 和 foundation/loader 针对性测试；远程真实两平台各取一个 batch，输出输入统计和 source IDs。本阶段没有新机制效果结论。交付 `P0_report.md`，附实际环境/manifest/lock/profiling/理论文件路径，然后停在 P0 交接点。

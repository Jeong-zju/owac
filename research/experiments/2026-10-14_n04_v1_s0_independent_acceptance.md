# N04 v1.0 S0 独立验收与修补（2026-10-14）

> 后续状态：用户授权的[远端数据补齐与验收](2026-10-14_n04_v1_s0_completion.md)已完成，S0 现已通过。以下保留本轮补齐前的 partial 结论和证据，不代表当前状态。


## 裁决

**原交付不能直接判为完整 S0 通过。修补后的核心工程合同验收通过；真实数据交付仍是 partial。**

依据是 [S0 提示词](../../docs/prompts/n04_v1_0/S0_data_contracts.md) 与[方案 §8-S0](../../docs/plans/N04_full_development_v1.0.md)。提示词允许“工程完成与数据部分完成”分开交付，但不能用 fixture 或 API 可调用来宣布正式数据合同全部通过。该边界本次不放宽，不启动 S1，不提交或推送。

原交付的 334 项测试本机确实全部通过，拆分数量和源组互斥也可复算。但是实际线上上下文绕过证据等级，D1 原对应检查也不足以支持 sidecar。修补前独立反例为 **13 failed**，见[遥测反例](results/2026-10-14_n04_s0_independent_audit/before-telemetry.txt)和[采集反例](results/2026-10-14_n04_s0_independent_audit/before-collector.txt)。原 handoff 已[原样备份](results/2026-10-14_n04_s0_independent_audit/handoff-before-independent-audit.json)。

## 原始要求—实际问题—修补

| 要求 | 原实际行为 | 修补与验收 |
| --- | --- | --- |
| 只有已核实、已发生控制进入在线上下文 | `executed_action` 挡住 provisional，但 `interval_report/context_for` 只看 `status=executed`；`producer_code_order`、`unknown`、`inferred` 均能进入 replay | 在线区间统一要求 verified 且有数值；三种未核实等级反例均被排除，C5.4 常驻 CLI |
| 只重放已发生的控制片段 | 与窗口相交就返回原完整区间，例如决策到 0.5 s，却返回结束于 0.9 s 的控制 | 只返回与查询窗口的交集，保留原命令时间和顺序，源记录不变；缺失时间不再用关联帧曝光时间猜测 |
| 不同时钟/单位不能比较，延迟含义独立 | 秒/毫秒和混合 clock 被直接比较；NaN/Inf 时间能进入记录；context 的 arrival_delay 实际是 decision−arrival | 拒绝不兼容时间域和非有限时间；arrival_delay=arrival−exposure_end，新增 processing_delay=decision−arrival |
| sidecar 必须验证原观测/动作/状态逐帧对应 | 将 `set_init_state` 后 `get_state` 相同当成 correspondence_verified，实际上没比较原图像和执行转移 | 该检查只证明状态 roundtrip，始终不授予原数据对应；新采集使用独立 extension 名称 |
| 采集器必须实际写出数据 | 不指定 trace 时，只生成 plan 也能成功返回 | 无实际采集的非 dry-run 返回非零；实际采集支持成对 pseudo-RAW、K、深度、位姿、分割、接触与控制轨迹 |
| 热启动策略必须在采集和复算两端一致 | `zeroed_each_step` 采集时没清零，只在验证回放清零 | 两端每步均清零；三种模式远端实测 |
| 序列必须是真正连续的执行 | 每步重置到外部 `states[t]`，不能直接作为连续轨迹 | 只在开头初始化一次，后续连续 step；逐数组核对 `pre[t+1]==post[t]` |
| 可复算结论必须来自数值证据 | 单凭 `verification.bit_exact=true` 就可授予证据等级 | 要求有限误差、正步数及 `0 <= error <= tolerance <= 1e-12` |
| 变体必须按可信源身份分组 | metadata 缺失时退回完整文件路径，变体可能被当成不同源 | 未核实 path 身份不能进入正式 split；当前 2403 条 metadata 身份与原拆分不变 |
| 合同失败必须阻止自动验收 | CLI 默认即使 failed 也返回 0，需另加参数才报错 | failed 始终返回非零；partial 仍单独报告，不冒充全部通过 |

代码集中于 [telemetry.py](../../src/owac/raw/telemetry.py)、[n04_mechanism_data.py](../../src/owac/raw/n04_mechanism_data.py)、[n04_d1_capture.py](../../src/owac/raw/n04_d1_capture.py)、[n04_manifest.py](../../src/owac/raw/n04_manifest.py) 与 [n04_contracts.py](../../src/owac/evaluation/n04_contracts.py)。未回退原有未提交工作，旧 loader golden 与旧 N04 回归仍通过。

## 真实数据与独立复算

- 远端再次执行两轮完整盘点：LIBERO 1771 + RoboTwin 632 = **2403 文件、2403 metadata 源身份**，总计 509,194,817,528 B。两轮 `inventory-hashes.json` 全部相同，新的 `content_sha256=e8a3df227eed7328c0f0b9a56426380b0230fba08faa1a7d0ae63ef36fc1107f`。代码身份变化使该整体 hash 不再等于旧版本，不能跨版本要求它相等。
- 主拆分仍为 **1917/251/235**；成员表 hash 保持 `0d3d41911816361b5b5e1093e07515f87ded913739d685e056cdeb9fe5f12247`，源 ID 两两交集为零。实际 CLI C7 重新复算，而非引用旧 PASS。
- D0 16 因子重新生成并运行 B=1/2/4 逐样本审计，见[新审计](results/2026-10-14_n04_s0_independent_audit/remote-a/mechanism/d0/audit.json)。这是解析合成机制数据，不能作为真实传感器或 H1/H2/H3 的证据。
- 冻结协议、两份评测清单、两份权重和三份完整结果文件共 **8 个文件**重新读字节计算 SHA256，全部一致，见[远端运行记录](results/2026-10-14_n04_s0_independent_audit/remote-a/audit.json)。本轮未操作任何 supervisor 服务，也未重跑基线。

## LIBERO 新扩展数据：确实生成，范围有限

从主拆分 train 中确定两项任务，每项两条源轨迹：stove/moka pot 的 demo_1、demo_11；black bowl/drawer 的 demo_0、demo_10。具体完整 source ID 在远端 A 的 audit.json 中。只复用源初态和控制命令，重新采集后续连续观测；**新数据不挂回原 NPZ，不继承原 episode 的成功标签**。

最终数据在 `/workspace/owac/research/n04-v1/runs/s0/independent-20261014-b/capture-{0,1,2,3}/`：4 条 × 8 时刻 × 2 相机 = **32 时刻、64 相机帧**。各目录保存 `.npz` 执行轨迹、`.trace.json`、`.measurements.npz`、`.measurements.json` 和 collection plan。本地只留小报告及哈希，大数组留远端。

逐帧字段含三通道 pseudo-RAW、参考 RGB、K、相机位姿、米制深度、实例/几何分割、可见支持、物体位姿、接触、本体、仿真秒，以及另存的实际 `env.step` 归一化限幅命令。`gt_*` 仅监督/诊断；不将参考 RGB 送入编码器。不声称原生 Bayer、真实曝光、硬件延迟或电机力矩。

三种 warm-start 方式覆盖四条采集；独立新建环境回放的状态最大差为 **0 / 0 / 1.7302e-14 / 0**。所有 pseudo-RAW、RGB、深度、分割和相机位姿逐元素重放一致。连续性 `max|pre[t+1]−post[t]|=0`。

仅有重放一致仍不足。独立 CPU `mj_ray` 对照 GPU 深度发现：Robosuite 返回的 viewport 中心 128 与像素索引中心 127.5 不能混用，而且 MSAA 深度可能对应子像素。首轮 A 保留为历史，不作为最终几何数据。新增采集器固定 `offsamples=0`，转换主点为 127.5 后重新采集 B。

最终独立验证直接重读数组与哈希，不接受报告内的 PASS：[独立数据结果](results/2026-10-14_n04_s0_independent_audit/remote-b/independent-data-validation.json)。1600 条射线中 1519 条与渲染几何 ID 对应；81 条几何 ID 不一致样本显式排除（未逐一归因），不记为零误差通过。对应射线的最大深度误差 **2.1532e-5 m（0.0216 mm）**，最大误差/数值界为 0.723。数值界由 float32 depth-buffer 精度给定 `8*eps*z²/near+1e-6 m`，不是调宽研究性能门槛。原 H1/H2/三维模型门槛未改变。

## RoboTwin 重新核查

本轮资源检查约 46 GiB 空闲，已成功创建 `grab_roller` 的新 seed=20261014 场景；读取 4 个相机的内外参、三通道 RAW、深度和分割，数组有限；取得 **33 个接触**。见[实际结果](results/2026-10-14_n04_s0_independent_audit/robotwin-live-probe.json)。Vulkan ICD / OIDN 日志告警保留在 txt，未修改共享依赖或驱动。

这解除“未建 live env、只能 API 级核查”的旧限制；**不证明旧 632 条缓存的逐帧对应、不恢复缺失的历史控制，不是 RoboTwin 新完整训练集**。RoboTwin 成对执行轨迹采集仍是后续数据补齐项，不能沿用旧缓存 actions 的名称把关节状态变成实际控制。

## 验收状态和 S1 准入

| 层级 | 本轮结论 |
| --- | --- |
| 原交付完整通过 | 否，存在已复现的因果和采集缺陷 |
| 核心数据/因果工程合同 | 修补后通过，本地和最终远端快照均为 349 tests passed；make check 通过 |
| C0–C7 CLI | 7 passed、0 failed、**C5 partial**；C5.1/2/4 代码语义通过，C5.3 原缓存实数据仍 skipped |
| D0 | 合同与逐样本审计通过 |
| D1 | partial；LIBERO 4 条真实渲染短扩展已核验，RoboTwin 仅新增 live geometry 可行性 |
| S0 完整真实数据合同 | 未完成，不合并成一个 PASS |
| S1 正式训练 | 未启动；只有 D0 和上述有标签短扩展可用于有明确范围的开发，不足以直接宣称正式规模训练集就绪 |
| H1/H2/H3、S7 闭环 | 未测量/未建立 |

剩余工作是正式 D1 因子覆盖与规模、RoboTwin 带执行日志的新数据、真实数据时移/到达合同，以及按真实标定在训练前重新冻结三维门槛。小规模的四条扩展不能代替充分训练数据。官方测试清单的源池重叠、scene holdout 窄和 camera 仅视角级的限制继续有效。

## 复跑与来源

本机：`uv run --locked --extra encoder pytest -q tests/encoders/n04 tests/encoders/n04_full`；`make check`；合同 CLI 传 `--manifest research/experiments/results/2026-10-14_n04_manifest/splits.json`。日志见本结果目录。

远端源码快照 B：`n04-s0-20261014T050411Z-357e5289dbbb`，**850 文件 / differences=0**，含当时未提交源码；A 保留盘点与早期失败诊断。快照 B 的具体命令见 `remote-b/audit.json`。新增 [audit_s0_remote.py](../../scripts/n04_full/audit_s0_remote.py) 可复跑完整有界流程，[validate_d1_capture.py](../../scripts/n04_full/validate_d1_capture.py) 独立重读数组。后续训练应重新打包最终工作树并核对运行代码，不读取可变 checkout。

按方案补做 S0 文献核查：原文摘要确认 [Ac-RKN](https://arxiv.org/abs/2010.10201) 已有动作条件转移，[MCDS-VSS](https://arxiv.org/abs/2405.19921) 已有相机/物体运动补偿与特征融合；新检索的 [Dyn-3D](https://arxiv.org/abs/2609.01059) 讨论用反事实三维渲染区分视觉变化与真实运动。此次为来源/摘要核查，未复现实验，不据此声称 N04 新颖性或机制成立。

最终归档快照：`n04-s0-20261014T051153Z-8bd493ab4460`，879 文件校验 differences=0。本轮已逐文件比较最终快照与采集 B 的 `src/` 和 `configs/`，运行代码无差异；最终远端重新运行 349 项测试全部通过。最新 [acceptance.json](results/2026-10-14_n04_s0_independent_audit/acceptance.json) 与 handoff 在打包后回填本次验收回执，运行源码未变。

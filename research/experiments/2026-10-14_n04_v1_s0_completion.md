# N04 v1.0 S0 远端数据补齐与验收

本记录续接[独立验收](2026-10-14_n04_v1_s0_independent_acceptance.md)。用户授权在远端直接补齐数据。S0 的原始出口是可信的数据、因果合同、分组拆分和逐样本诊断；S1 的充分训练、收敛曲线和正式模型规模不应提前成为 S0 的否决条件。

## 本轮解决的问题

| 原缺口 | 处理 | 证据范围 |
| --- | --- | --- |
| 仅四条短 LIBERO 扩展 | 固定两个任务各十条源轨迹，重新连续采集；同一源始终在同一拆分 | 新扩展，不挂回原 NPZ，不继承专家成功标签 |
| RoboTwin 无真实控制日志 | 两任务各十条新场景，保存控制器限幅后的关节驱动目标、实际本体与同步多相机几何 | 有执行证据的仿真数据，不是旧 632 条缓存的历史恢复 |
| 真实渲染因素缺少系统覆盖 | 独立 MuJoCo 三维渲染套件，16 因素、10 源种子、152 序列；组合 `both_gain_noise` 仅开发/留出 | 1,824 时刻、3,648 相机帧；已通过独立 NumPy 射线/包围盒复算 |
| 到达时刻只有合成 fixture | 采集时真实扣留每三帧中的一帧，经过后续执行再释放，保存实际 monotonic 时钟；逐次决策检查未到达帧及未来控制 | 仿真状态采样与主机队列到达合同；不冒称硬件曝光/真实传感器延迟 |

两平台数据在新根 `/workspace/owac/research/n04-v1/data/s0-paired-v1-20261014/`；受控渲染在 `/workspace/owac/research/n04-v1/runs/s0/completion-factors-20261014/`。数组留远端，本地保留命令、配置、计数、哈希、日志及逐序列验收结果。

## 实际发现并修复的采集问题

1. RoboTwin 原分割函数会把对象 ID 转成 uint8 调色板图。本轮直接保存 `Segmentation` 中的整数 ID，避免身份信息丢失。
2. RoboTwin RT 渲染器的主射线使用 Halton 子像素位置。新扩展在独立目录复制着色器，仅将主射线固定到像素中心，保留 32 spp 光照采样和 HDR 出口；关闭不可用的去噪器。未修改安装目录或冻结基准。原试验约 0.47 px 的偏差降至约 0.00038 px。
3. 两平台统一整数像素中心坐标。SAPIEN 内参的边缘坐标减去 0.5；已采数据通过可追溯转换修正，原字节仍保留。
4. LIBERO 的完整 `qpos/qvel` 包含场景对象，不是纯机器人本体。现在按模型中的机器人/夹爪关节索引提取在线输入，完整状态放入 `gt_sim_*`。新增输入/监督分离加载器，旧 schema 不可直接当新数据使用。
5. 物理积分后的派生相机/物体变换可能滞后于当前 `qpos`。LIBERO 新采集在状态边界执行 `forward`，随后记录同一状态的测量与控制。本轮重采固定的二十条 LIBERO 源，不修改原录制。
6. 几何审计区分 GL 深度缓冲精度、光栅子像素精度和三角面朝向。数值界从 `GL_SUBPIXEL_BITS` 与同一条射线邻域推导；未改 H1/H2 或模型的性能门槛。
7. 酒架等透明材质会混合多个表面。透明度 0.1 的实际网格证据解释了若干“最近射线 ≠ 深度缓冲”的大误差。保留 RAW、分割和可见性；新增 `gt_geometry_support` 排除透明像素的唯一深度监督。不是把残差大的像素事后删去，而是按渲染器声明的材质透明度统一定义支持。

8. RoboTwin 含旋转关节和直线关节，不能将所有控制量统一标为 rad。现逐关节记录名称、articulation 顺序、类型、位置/速度单位与展平索引；76 个驱动槽共 152 个读回量包含持续有效目标，不等同于 VLA 的 14 维动作。原控制数组不变，二十份元数据的前后哈希与转换回执留档。

早期错误解释“只有背面剔除”不足以解释全部差异，已由材质与具体三角交点核查纠正。透明表面的唯一几何不可观测性继续保留；不能从当前数据声称解决透明物体深度。

## 因素与用途

受控数据包含静态、纯相机运动、纯物体运动、同时运动、物体停止/反向、遮挡恢复、出生/消失、增益、噪声、帧时移、错误关联、标定偏移、阴影/高光、占满画面的移动平面，以及组合因素。对象位置是生成器干预的监督标签，不伪装成机器人执行控制。真实机器人可见结构、控制动态与接触记录由 LIBERO/RoboTwin 扩展补充。

受控场景直接渲染三维资产，和 D0 解析渲染是两层证据。其 RAW 是明确标注的 RGB-derived radiance proxy；LIBERO 沿冻结 RawVLA producer；RoboTwin 保存 HDR-derived 三通道 RAW。都不宣称原生 Bayer 或实测传感器。

静态图像不变、停止/反向的正/零/负速度、目标在遮挡段消失后重新可见，均由独立脚本读取数组检验。时移与错误标定会由 `rendered_training_frame` 真正作用到训练输入，GT 仍留在目标区。

## 失败记录与恢复

原始批量采集保留在 `runs/s0/completion-live-20261014/`。RoboTwin `adjust_bottle` 的三个初始种子被上游 `UnStableError` 拒绝；只对该初始化失败按 `seed+1000*k` 的固定规则重采，拆分位置不变，实际种子和源 ID 进入 resolved manifest。重复诊断与原失败日志保留，未按任务成功或模型结果筛选。

早期 RT 抖动、无 HDR 的 raster 尝试、解析球面与多边形球面差异、未完整考虑透明性的几何审计均保留为历史。两次启动顺序/配置错误发生在采集前，没有静默重复计入完成轨迹。最终分母以显式配置与逐文件复算为准。

## 验收回执

**S0 按原定数据与因果合同通过。** 充分训练及机制收益仍属于后续阶段，未启动 S1。

- 新成对扩展：40 条、960 时刻、2,880 相机帧；LIBERO/RoboTwin 各两任务、每任务十条；源级拆分 32/4/4，交集为零。全部逐文件哈希、连续状态、单位、像素对齐、在线输入隔离均通过。
- 受控渲染：152 序列、10 个源种子、16 类因素，1,824 时刻 / 3,648 相机帧；源拆分 8/1/1。独立 NumPy 几何复算最大误差/数值界为 0.0778；不以生成器的 PASS 字段代替复算。
- 真实交付队列：1,920 个决策时点、320 对乱序到达、2,120 次已执行控制条目重放，以及 1,920 次未来控制毒化检查全部通过。相机、物理秒、host 到达时间分别记账。
- LIBERO 不透明支持上的最大误差/数值界为 0.6454；RoboTwin 独立投影误差最大 0.000455 px。接触累计计数 30,164 是逐时刻接触条目的和，不是独立接触事件数。
- 最终远端 CLI：**C0–C7 = 8 passed / 0 failed / 0 skipped / 0 partial**。不提供 `--live-data-root` 的默认运行仍如实报告原缓存缺口，不能把新数据验收转借给旧缓存。
- 本地与最终远端均 **361 tests passed**；`make check` 通过，246 个 Python 文件格式检查通过；`git diff --check` 通过。
- 最终运行快照 `n04-s0-20261014T062313Z-cc61897ab1cb`，936 文件、differences=0，包 SHA256 已在解包前校验。采集源码 C/F/H 与最终 J 的用途分别记录；最终 J 实际重跑合同与测试。该记录和 handoff 的回执在打包后回填，`src/`、`configs/`、`scripts/`、`tests/` 共 170 文件与快照逐字节一致。
- 8 项原协议/清单/权重/结果资产重新读字节校验一致。只新增本任务 supervisor program，当前采集全部结束，没有停止或重启旧任务；未提交、推送或训练。

[机器可读验收](results/2026-10-14_n04_s0_completion/acceptance.json)、[完整合同报告](results/2026-10-14_n04_s0_completion/completion-final-contracts.json)、[新数据成员与哈希](results/2026-10-14_n04_s0_completion/dataset-manifest.json)、[逐序列数据审计](results/2026-10-14_n04_s0_completion/paired-final-audit.json)、[受控因素独立审计](results/2026-10-14_n04_s0_completion/factors-independent-audit.json)。

## 已冻结的标定门槛

[标定协议](../../configs/n04/full/protocol_calibrated_sim.json) 只使用训练源的相机标定与几何，未使用模型结果或 dev/final 调整。继续要求相对收益至少 10%，绝对收益严格超过对应深度 `Z/fx` 的一个原图像素；原 0.05 px 下限、H1 的 1% 误报预算、H2 的 15%/5% 保持不变。

训练数据中位深度处的一个原图像素分别为：LIBERO 固定相机 3.194 mm、腕部相机 2.124 mm；RoboTwin 前/头/左/右相机 2.313 / 2.126 / 1.741 / 1.731 mm。实际评分按明确的深度和像素单位换算，不能跨相机直接套旧名义 1.875 mm。

## 交接与保留限制

[当前 handoff](results/2026-10-14_n04_v1_s0/handoff.json) 已更新。S1 可以使用新扩展的 `online_rawvla_rgb3`、真实几何支持与独立 targets 开始测量模块阶段；加载入口为 `LiveExtensionSequence` / `rendered_training_frame`。普通编码输入不含 `gt_*`、控制后的状态或未来动作。Robotwin 历史版本的 `reference_rgb` 实际是作者 RAW 显示预览，此字段不是编码输入，也不用于 RGB 教师。

S1 仍需证明训练规模充分和测量收敛；当前通过的是 S0，不是 ConvNeXt-B 已训好。原 2403 缓存缺字段、真实硬件时序未测、透明材质不可唯一恢复深度、原官方源池重叠及 scene/camera holdout 狭窄，均是后续适用范围限制，不能据此宣称未见 episode 泛化或 H1/H2/H3 成立。新脚本控制轨迹也不冒充成功专家监督。

复跑：在最终快照使用 encoder Python 运行 `python -m owac.evaluation.n04_contracts --config configs/n04/full/s0_data.json --manifest research/experiments/results/2026-10-14_n04_manifest/splits.json --live-data-root /workspace/owac/research/n04-v1/data/s0-paired-v1-20261014 --output <新报告>`。数据校验入口为 `owac.evaluation.n04_live_contracts` 与 `owac.evaluation.n04_factor_contracts`；无 GPU/训练依赖的合同复验使用独立 encoder 环境。

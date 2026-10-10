# Arena DROID 专家采集、生成与回放脚本核查

目的：在用户指定的 Isaac Sim / Arena DROID 中，寻找可恢复场景和物体状态的专家数据来源，以及可复用的采集、生成、回放脚本，后续用原生 CameraCore 输出配对 RGB/RAW。用户已否定 MuJoCo 路径。

结论：找到了官方通用采集、Mimic 扩增和状态回放工具，但没有核验到当前 DROID 九组抓放任务可直接运行的自动专家生成器。公开 PolaRiS 仿真共同训练数据缺少逐帧物体状态，当前不能交付其原始训练轨迹的完整场景回放。不能将这一结论扩大成 Isaac 没有专家数据工具。

## 官方已有工具与当前适用性

| 官方工具 | 已具备的能力 | 当前 Arena DROID 的缺口 |
| --- | --- | --- |
| [record_demos.py](https://github.com/isaac-sim/IsaacLab/blob/91a0906f93802eb859192dcb0bca879f891e92d6/scripts/tools/record_demos.py) | 人工遥操作，成功演示导出 HDF5；通用 recorder 保存初始状态、动作、观测和每步物体/机器人状态 | 默认 keyboard / SpaceMouse 是 SE3 末端位姿命令，未接通 DROID 的 7 关节位置 + 1 夹爪动作；需要控制适配和人工种子演示 |
| [Mimic generate_dataset.py](https://github.com/isaac-sim/IsaacLab/blob/91a0906f93802eb859192dcb0bca879f891e92d6/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py) | 从已标注的成功种子演示扩增，不是从零凭空生成专家 | 当前 DROID 没有 Mimic embodiment 入口、末端位姿/动作转换和已标注种子集 |
| [replay_demos.py](https://github.com/isaac-sim/IsaacLab/blob/91a0906f93802eb859192dcb0bca879f891e92d6/scripts/tools/replay_demos.py) | 恢复初始状态后重放动作；`--validate_states` 核对机器人和刚体状态；可验证回放成功率 | 尚无匹配的 DROID 专家状态 HDF5；脚本没有原生 RAW 写入，需接入已验收的 CameraCore 采集器 |
| [lift_cube_sm.py](https://github.com/isaac-sim/IsaacLab/blob/91a0906f93802eb859192dcb0bca879f891e92d6/scripts/environments/state_machine/lift_cube_sm.py) | 自动状态机抓取、抬升 Franka 方块 | 本体、夹爪和任务与 DROID 不同；没有完整搬运放入不同容器的九配置专家，不能直接使用 |

源码证据：[DROID base 将 `mimic_env` 设为 `None`](https://github.com/isaac-sim/IsaacLab-Arena/blob/2cc9e9766f1f62a500b85c92207192d8c60fcce5/isaaclab_arena/embodiments/droid/droid.py#L160)，其 jointpos 子类没有补上实现。[Arena 编译器从 embodiment 取得 Mimic 入口](https://github.com/isaac-sim/IsaacLab-Arena/blob/2cc9e9766f1f62a500b85c92207192d8c60fcce5/isaaclab_arena/environments/arena_env_builder.py#L548)。因此给当前命令添加 `--mimic` 不能补齐专家生成能力。Arena 的 cuRobo 扩展用于环境构建时的可达性检查，本次没有核验到它提供这些任务的完整专家策略。

官方 [GR1 微波炉工作流](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/static_manipulation/step_3_data_generation.html)已有录制、标注、扩增 HDF5；G1 工作流也有数据。这些证明工具链可用，但对应不同本体、任务与策略，未作为 DROID 的替代数据。

## 状态与原生 RAW 回放边界

[状态记录器](https://github.com/isaac-sim/IsaacLab/blob/91a0906f93802eb859192dcb0bca879f891e92d6/source/isaaclab/isaaclab/envs/mdp/recorders/recorders.py)调用 `scene.get_state(is_relative=True)`，记录已注册 articulation / rigid object 的根状态和关节状态，包含物体而非仅机器人。相机、灯光、资产版本与 RAW 传感器配置仍需额外冻结和记录；HDF5 本身不是完整场景归档。

官方 replay 是“恢复初态 + 物理动作回放”，不强制逐帧状态恢复。`--validate_states` 默认对机器人/刚体字段采用绝对误差阈值 0.01；不能把这个笼统阈值称为精确 RGB/RAW 配准验收。应分别定义位置、姿态、速度误差与单位，并保存参考/回放状态差异。PhysX 回放可能有非确定性，成功演示也可能在回放时失败，需逐条报告，不能替换失败样本。

记录时 pre-step RGB / action 与 post-step state 属于不同相位：第 0 帧对应 initial state，后续动作前图像对应上一步的 post-step state。接入 CameraCore 时需保证同一恢复状态、视角、曝光和物理时间的 RGB / ISP 前 CFA，记录 native frame ID、传感器配置及哈希，禁止简单按同名帧索引对齐。

目前已在历史 Arena DROID 闭环中验证原生 CameraCore CFA 输出；本次没有运行这些采集/生成/回放脚本的 DROID GPU 验收，也没有生成新的专家数据或配对训练数据集。

## 原始数据核查

PolaRiS 确实公开仿真共同训练数据。固定 [OpenPI 配置](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/training/misc/polaris_config.py)包含真实 DROID 与仿真数据 90% / 10% 的混合，不能将策略来源全部归为真实数据；配置也不能单独证明下载权重完整的实际训练历史。

`owhan/PolaRiS-datasets` revision `dca86b8e186df63323eb375344a1977d92f508f7` 的[公开 schema](https://huggingface.co/datasets/owhan/PolaRiS-datasets/blob/dca86b8e186df63323eb375344a1977d92f508f7/polaris_droid_cotrain_dataset/1.0.0/features.json)包含三路 RGB、关节/夹爪状态、对应动作、`recording_folderpath` 和 `file_path`，没有完整物体状态。`owhan/DROID-sim-environments` revision `5b815672a9a2e3ae596ae845188ae4be5c5d2f34` 发布 USD 资产和 HDR，未核验到原训练逐帧状态记录；评测初态不能替代完整轨迹。在所核查的 [PolaRiS 公共实现](https://github.com/arhanjain/polaris/tree/129abc45f680559a7da62b9280fdb127b02b7275)中也没有找到现成录制/专家生成脚本。

## 下一步需要决定的工作

如果原始状态文件不能获得，两个可实现但尚未验收的方向是：

1. 适配官方遥操作 / IK 到 DROID，采少量成功种子演示，再补 DROID Mimic 接口并扩增。能复用最多官方代码，但需要人工种子和控制适配。
2. 实现 DROID 抓放的脚本或规划专家，成功轨迹通过官方 recorder 保存完整状态，再回放采集原生 RGB/RAW。可减少人工采集，但专家生成本身是新增工程，不是已找到的现成脚本。

这两种方式产生的是**新专家数据**，不能称为官方 π0.5 的原始训练数据。当前只报告选项，没有启动这两项实现，也没有用评测策略轨迹替代专家来源。第一阶段完整 RGB 基准独立运行，见[基准记录](../experiments/2026-10-10_pi05_arena_droid_benchmark.md)。

核查范围为固定 Arena `2cc9e9766f1f62a500b85c92207192d8c60fcce5` / Isaac Lab `91a0906f93802eb859192dcb0bca879f891e92d6` 源码、官方公开工作流和数据 schema；云端八个相关源码文件逐项 SHA256、实际检查时间与结论保存于[小型审计清单](2026-10-10_arena_droid_expert_replay.json)。这是源码适用性核查，不能替代实测验收。

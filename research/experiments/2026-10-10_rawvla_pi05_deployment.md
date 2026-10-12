# 2026-10-10 · RawVLA 作者环境与 π0.5 部署验收

目的：在用户新租用的独立服务器下载 RawVLA-Bench 和作者指定检查点，验证 LIBERO、RoboTwin 的真实 π0.5 推理与闭环，为 encoder 原型提供现成数据和策略参考。

做了什么：锁定作者源码和数据版本，部署两套 Python 3.10 环境，转换官方 LIBERO JAX 权重，直接加载 RoboTwin 的原始 PyTorch 权重及 normalizer，运行作者仿真器和成功判据，保存模型实际输入、动作、HDF5、视频与小型结果清单。额外训练 **0 步**。

效果：LIBERO 四套任务各一条 RGB 验收成功 **4/4**；Spatial 首任务五照明域、七个初态的 Default ISP 验收 **19/35**，全部失败保留。RoboTwin 按作者 HDR/RAW 配置执行 Normal 条件 `grab_roller`，seed 0 在 **81 步成功**。数据全集下载及文件大小、配对清单检查完成。

结论：两套 π0.5 的数据、权重、推理服务与仿真闭环均已跑通，可以开始 RawVLA 三通道输入的 encoder 原型与离线测试。本记录是小规模部署验收；不报告完整论文复现、强泛化或 OWAC 原生 CFA 准入通过。

## 身份与环境

- 用户明确授权新服务器上的作者 LIBERO/MuJoCo 与 RoboTwin 部署；本试验与已有 Arena DROID 冻结基准使用不同服务器和目录。
- 本地 checkout：`/home/jeong/zeno/owac/repo`，分支 `lzh/pi05-sim-baseline`，基准提交 `0fc31025f2effeaa1b1fd49d6b175b84ef624265`。原 checkout 已有大量未提交变更，本轮新增两个 Python 模块、八项协议测试、复现配置和此记录，仅追加状态说明；未提交或推送。
- 原有 tracked diff 快照：被忽略的 `outputs/rawvla-deployment/preexisting-worktree.diff`；未跟踪文件清单同目录保存。本轮部署模块及 SHA256 另存云端 `records/module-snapshot/`，见 [服务清单](results/2026-10-10_rawvla_pi05_deployment/service-check.json)。物理验收中的核心加载/推理函数与最终服务相同，随后增加 OptiX 选择及输入文件计数；不将最终文件哈希冒充每次早期启动的源码快照。
- 服务器工作目录：`/workspace/owac/rawvla`；源码 `source/`，隔离环境 `envs/starvla`、`envs/robotwin`，下载环境 `envs/hub`，权重 `checkpoints/`，证据 `records/`，日志 `logs/`。
- GPU：RTX PRO 6000 Blackwell Workstation Edition，97,887 MiB，驱动 595.84；Python 3.10.21、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128。主环境 `/venv/main` 未改动。完整 [StarVLA](results/2026-10-10_rawvla_pi05_deployment/starvla-freeze.txt) 与 [RoboTwin](results/2026-10-10_rawvla_pi05_deployment/robotwin-freeze.txt) 依赖导出已保存。
- 开始部署约 08:35 UTC / 16:35 北京时间；数据下载于 09:04:39 UTC 完成。最终服务检查时间见服务清单；日志、视频与大数组留在云端。

## 上游版本与检查点

| 对象 | 固定版本 / 来源 |
| --- | --- |
| [RawVLA](https://github.com/ShuhongLL/RawVLA) | `ba9c56b04da2c4931b70d8844710f5928d43775b` |
| 作者 StarVLA / LIBERO / RoboTwin 子模块 | `57d508cb180d0510ac9060358009c0f6ffaac420` / `5a2d7a27d557089747032d7a3f599568e1796844` / `8e9e66544e2e9efc5910721b8d97d6dee806fa20` |
| 作者 OpenVLA-OFT 子模块（环境依赖） | `d345e2734a1b56918fb68f3d3e5f470cc45059d8` |
| [RawVLA-Bench](https://huggingface.co/datasets/ToferFish/RawVLA-Bench) | `8586c594fff9ce38580e29745f074e82e96e74f7` |
| LIBERO π0.5 | PI 官方 `gs://openpi-assets/checkpoints/pi05_libero`；原始对象 generation / CRC32C / SHA256 见 [GCS 校验](results/2026-10-10_rawvla_pi05_deployment/libero-gcs-check.json) |
| RoboTwin π0.5 | [SidneyXie/pi05_robotwin](https://huggingface.co/SidneyXie/pi05_robotwin)，作者指定 revision `e49e2ab6c11f07511573b67261bd129e88d0a416` |

下载入口依据 [作者 RoboTwin backbone 指南](https://github.com/ShuhongLL/RawVLA/blob/main/experiments/simulation/robotwin/backbones/README.md)。另外下载 `TianxingChen/RoboTwin2.0` 的三个资产 ZIP，并执行作者 `setup_robotwin_assets.sh` 与 `update_embodiment_config_path.py`。没有安装无关模型或训练 RAW 前端。

LIBERO 转换使用固定 OpenPI `15a9616a00943ada6c20a0f158e3adb39df2ccac` 的两个官方 tensor slicing 函数，沿用原始统计。转换后 812 个权重匹配 813 个 StarVLA 键；唯一缺项是动作专家从不调用的语言词表 `lm_head`，明确置零保存。所有参与动作推理的参数缺失、意外键、形状不匹配均为零；最终通过作者严格加载器。见 [加载审计](results/2026-10-10_rawvla_pi05_deployment/libero-load-audit.json)。

RoboTwin 没有转换或重训原始 `model.safetensors`：按作者 PI05 loader 的名称映射载入，**813/813** 键匹配。作者 exporter 生成的 `policy_norm_stats.json` SHA256 为 `229a4a2d6cfd1ba4eb2bc0b3aa57f049094032939130e4901485e9ea9910f5f7`，与作者公布值一致。原权重 SHA256 为 `9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3`。见 [加载审计](results/2026-10-10_rawvla_pi05_deployment/robotwin-load-audit.json)。

## 数据与策略协议

[下载检查](results/2026-10-10_rawvla_pi05_deployment/dataset-check.json)：4,810 个文件、**713,689,392,570 bytes**（713.69 GB / 664.67 GiB）；1,771 对 LIBERO 与 632 对 RoboTwin NPZ。缺失文件、大小不匹配、缺少 RAW/RGB 配对均为零。HF pinned snapshot 下载完成后，逐文件核对大小与配对路径，并实际重读四个配对样本、计算 SHA256 和数组范围；没有二次完整重读 714 GB 计算全量 SHA256。

- LIBERO：两路三通道 uint8 pseudo-RAW，来源为 RGB 逆处理；`robot_states` 为八维本体，监督动作七维。NPZ 的 92 维 `states` 是仿真状态，不能作为正常策略输入。
- RoboTwin：三路三通道 float32、HDR 派生 RAW；NPZ 的 16 维 `robot_states` 是末端位姿/夹爪表示，不能当作十四维关节状态。作者训练配置使用 `state_source_key: actions` 与 `action_offset: 1`，将当前关节向量与后续动作监督分开。在线策略输入为真实观测的 `joint_action/vector`。
- 公共 NPZ 是训练轨迹缓存，没有独立测试文件拆分。正式闭环测试使用作者 manifest；本轮仅取明确的部署子集。
- LIBERO 原策略：两路图像，quantile 归一化，十步动作块、五步重新规划，原七维动作及成功判据。
- RoboTwin 原检查点：头部/左腕/右腕相机，左六关节/左夹爪/右六关节/右夹爪，MEAN_STD 归一化，32 维模型内部零填充、50 步块、输出十四维绝对关节位置。不使用通用 Qwen client 的关节重排、min/max 或夹爪二值化。
- RoboTwin 图像预处理使用作者 StarVLA PI05 实现；原 LeRobot 的逐像素预处理/数值等价性未全面对照，不能声称 native LeRobot 完整等价复现。
- 两个数据子集均无 Bayer CFA；可用于三通道 RAW 机制/encoder 原型，不能当作 OWAC 原始 CFA 训练数据。

## 实测闭环

| 协议 | 结果 | 范围 |
| --- | --- | --- |
| LIBERO RGB | 4/4 成功 | Spatial/Object/Goal/Long 各 task 0、init 0，作者 client seed 7；[逐任务](results/2026-10-10_rawvla_pi05_deployment/libero-rgb-summary.json) |
| LIBERO Default ISP | 19/35 成功 | Spatial task 0，init 0–6，每初态五域；[域汇总](results/2026-10-10_rawvla_pi05_deployment/libero-lighting-summary.json) |
| RoboTwin 作者 HDR + Normal + Default ISP | 1/1 成功、81 步 | `grab_roller`，seed 0，OptiX；动作/观测 81×14，有限值与数量对齐检查通过 |

LIBERO 五域分别为 ExtremeLow **0/7**、Low **7/7**、Normal **7/7**、Over **5/7**、ExtremeOver **0/7**。原 runner 的 PI05 路径按 manifest 展开，忽略 `NUM_TRIALS=1`，而 `MAX_NEW_EPISODES_PER_RUN` 仅传给另一条模型路径。本轮两次有意中断调度后，取原 manifest 的首七初态共 35 条完整前缀，逐条断言 entry 与原 manifest 相同，通过原 ledger 复用完成项并补齐剩余一条；没有重复计数、改光照或删除失败。[部署 manifest](results/2026-10-10_rawvla_pi05_deployment/libero-smoke-manifest-35.json) 保留父清单哈希。它是部署前缀，不是预注册研究比较或 10,000 条完整成绩。

RoboTwin 成功入口为云端 `records/robotwin-rawvla-normal/grab_roller/rollout/`，包含原生 HDF5、每控制步缓存、模型实际动作、视频及 `_policy_data/episode0.pkl`。HDF5 SHA256：`5c1a4fb11fb3b49c8128b9de789cb20ebfb8cc2f9544c3f7cd9be8409ab1e93a`。成功按原仿真器 `check_success()` 判定，输入只含当前图像、本体状态与语言；没有专家动作注入策略。

诊断结果全部保留：旧 OIDN 配置下 seed 100000 曾成功但渲染器持续报错，不纳入可靠渲染验收；seed 100001 超时。OptiX 下旧普通配置的 `grab_roller` seed 100000 和 legacy Default ISP 的 `lift_pot` seed 0 均 400 步超时。切换作者 HDR white level 3.5、Normal/paired/environment_only 配置后，取得上述成功。它们改变了配置和种子，不能据此单独归因于渲染、成像、随机性或策略泛化；未做选模、训练或统计性能比较。[所有带完整 HDF5 的结果](results/2026-10-10_rawvla_pi05_deployment/robotwin-closed-loop.json) 保存正常失败与校验。

## 兼容修改与验证

1. Blackwell 要求支持 sm_120 的 wheel：两环境换用 torch 2.7.1/cu128，保留作者其余模拟器版本。CuRobo 0.7.8 用独立 CUDA 12.8 nvcc 编译，MAX_JOBS=2，执行作者兼容补丁；没有安装或替换驱动。
2. 补齐作者环境缺少的 `future`、`easydict`。上游 diffusers 实际解析为 0.39.0，与作者 `peft==0.11.1` 冲突，改为 `peft==0.17.1`；依赖导出锁定实际环境。
3. LIBERO RAW 模式额外加入 `experiments/simulation/libero` 到 PYTHONPATH，避免作者 runner 的 `rawvla_bench` 导入缺失。
4. SAPIEN 3.0.0b1 的 OIDN 在 Blackwell 上报 invalid handle，客户端通过显式 `RAWVLA_ROBOTWIN_DENOISER=optix` 选择 OptiX。原场景、物理、RT shader、32 spp、path depth 8 和成功判据保留；降噪后端与作者默认不同，正式分数须披露。
5. RoboTwin 使用单独的检查点协议插件，避免通用 client 的关节顺序与统计类型错误。支持重置动作缓存，模型实际输入 NPZ 和 GPU forward 日志在 `records/robotwin-policy-v2/`。

两环境的 CUDA、依赖导入与无界面渲染检查通过。下载 RGB 样本的 GPU 前向输出均有限：LIBERO `[1,10,32]` normalized actions，RoboTwin `[50,14]` 原关节动作；这是前向检查，不是离线准确率。`make check`、八项数值/关节顺序/动作块重置/加载拒绝测试与 shell 语法检查通过。

原始日志记录每次启动与正常失败。时延日志中的 `inference_seconds` 仅含同步后的 GPU forward，不含输入 NPZ 保存、网络或模拟器；本轮不报告端到端性能、能耗或独立泛化统计。

## 复现与服务

服务由 supervisor 保持：LIBERO `owac-rawvla-libero-policy` / localhost:18000，RoboTwin `owac-rawvla-robotwin-policy` / localhost:18005；仅监听服务器回环地址。下载任务已经退出完成。最终服务重新加载权重后检查握手、ping 和 RoboTwin 前向；时间与模块哈希见服务清单。

登录用户提供的服务器后：

```bash
cd /workspace/owac/rawvla
supervisorctl status owac-rawvla-libero-policy owac-rawvla-robotwin-policy
bash run-smoke.sh libero-rgb
bash run-smoke.sh libero-lighting
bash run-smoke.sh robotwin-normal
```

复现入口的本地版本：[run-smoke.sh](../../configs/backends/rawvla-runtime/run-smoke.sh)。每次默认创建新的 `records/recheck-<UTC>/`，`RAWVLA_RESULT_ROOT` 可指定复现输出目录；不会覆盖既有验收。运行需要现有 supervisor 服务和已部署环境。作者 HDR 任务配置的本地副本：[owac_rawvla_clean.yml](../../configs/backends/rawvla-runtime/owac_rawvla_clean.yml)。

LIBERO 转换入口为 `owac.backends.rawvla convert-libero <JAX目录> <固定OpenPI转换脚本> <tokenizer> <输出目录>`；拒绝覆盖输出，严格审计有效参数。RoboTwin 插件入口为 `owac.runtime.rawvla_robotwin`。本地仅同步小型结果、哈希、依赖与配置；714 GB 数据、检查点、观测数组和视频留在服务器。该实例 `/workspace` 是容器 overlay，不是独立持久卷；保留实例时可继续使用，销毁前须另行备份大资产。

下一步可以基于现成 NPZ 建立 encoder 数据加载和离线机制实验；本轮没有修改候选架构、训练 encoder 或冻结共同起点标签。RoboTwin 原 LeRobot 数值对照与完整作者 manifest 评测仍未完成。

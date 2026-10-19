# N02 v1.1 P0 独立复查与修补验收（2026-10-14）

本次按原提示词逐项比较要求、实现和实际运行证据。DeepSeek 原交付不能直接判为 P0 完成：其 `implementation=pass / numerical=pass` 掩盖了未运行的必要参考，且数据划分、训练画像和 FCMAE 数学实现存在实际缺陷。原始交付及失败日志保留，本记录与新收据为修补后的验收入口。

工作树：`/home/jeong/zeno/owac/worktrees/n02-measurement-completion`；分支 `lzh/n02-measurement-completion`；起点 `843be674c35e8e058a364154e1c27a4b26ded2e4`。所有修改未提交。原工作树已有大量未提交实现，未回退或混入其他候选工作。源文件哈希见新收据 `source_manifest.json`，不能只凭 HEAD 复现本次实现。

**修补后验收结论：P0 通过。** `implementation=pass / numerical=pass / behavior=not_run / closed_loop=not_run`；完整 parity **17 pass / 0 fail / 0 not_run**。这里通过的是 P0 基础、数据与数值参考，不是 P1—P7 的行为或机器人能力。

## 1. 原要求与发现的缺陷

依据：[P0 提示词](../../docs/prompts/n02_v1_1/P0_FOUNDATION.md)、[共同契约](../../docs/prompts/n02_v1_1/00_SHARED_CONTRACT.md)、[完整 v1.1 方案](../../docs/plans/N02_full_development_plan_2026-10-13.md)。

| 原要求 | 原交付的实际问题 | 修补及验收方法 |
| --- | --- | --- |
| 必要上游算子必须实际运行，缺项不能完成 | parity 有 5 项 `not_run` 仍返回 pass；report 继承并接受不完整证据 | 聚合判定改为 incomplete；失败计数覆盖声明 pass；缺失阶段、必要训练 profile 缺失不能过关 |
| 任务内按源轨迹分组约 80/10/10 | LIBERO task 使用 suite；分配在全体 group 上进行；未优先恢复 NPZ 内源身份 | 恢复 metadata 的 task、源 HDF5/demo 或源 episode；按任务或连接任务组分配；2403 集重新生成 v2 manifest |
| 实际消费冻结 anchor 的窗口 | anchor 声明与数值 loader 默认窗口可不一致 | loader 默认读取表内完整 WindowOptions；显式冲突立即报错；回归测试验证历史与 action horizon |
| 正式 checkpoint 统计与 epsilon | RoboTwin 默认通用 minmax；inventory 对正式统计无实际文件验真 | 显式加载两平台 checkpoint 统计并记录 SHA；RoboTwin 使用 zscore 的 `std + 1e-8`；无统计不能判数据就绪 |
| 200 次训练更新与真实 batch/GPU 计时 | 只调用 forward；没有同步 GPU；dataclass 配置被忽略，实际 batch=1 | AdamW 前向/反向/step，单目标显存复位、同步计时；正确读取 batch=2；另做 forward 画像，二者分报 |
| FCMAE 稀疏数学及梯度一致 | stride=1 扩张有效坐标；downsample LN 沿空间归一化；GRN 用 RMS 替代 L2；缺失坐标的偏置泄漏 | 保持 stride=1 坐标；每坐标 channel LN；上游 L2 GRN；逐层重新施加 support；独立稀疏行参考和官方 CUDA 参考 |
| SAVi 原宽度前向、输入及参数梯度；512/16槽/4块画像 | Processor 未运行；predictor 未完整交付；512 配置未画像 | 修复 Flax/JAX 兼容组合与 mask 参数调用；新增原 postnorm Transformer predictor；导出 GRU/predictor 梯度并测正式宽度 |
| Theseus 官方示例及 FP64 解/梯度 | 手写 LM 不能代替官方示例；“CUDA 11 阻塞”尚未穷尽可用 dense 路线 | 在相同 Python3.10/torch2.11 栈运行 pinned 官方 example 与 TheseusLayer；解、隐式梯度及有限差分扫描 |
| Gen 与 particles 同一可枚举模型 | Gen 未安装，只有源码阅读 | 独立 Julia1.10.10 环境实际跑 pinned Gen；与 particles 共用同一 HMM、观测及枚举真值 |

## 2. 真实数据复核

实际连接 `ssh -p 44327 root@107.209.104.125`，容器 `2566cc983a37`。数据根为 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench`，其下实际文件位于 `data/libero/raw/` 和 `data/robotwin2/raw/`。因此简写的 `/workspace/owac/rawvla/data/...` 不是这台机器的真实位置。

重新扫描 1771 + 632 集元数据，校对固定 HF revision `8586c594fff9ce38580e29745f074e82e96e74f7` 的 LFS 文件大小。完整文件 SHA 来自该 revision 的 LFS 清单，未重新读取约 475GB RAW 文件计算 hash。新 manifest 与旧报告不同：

| 平台 | 任务数 | train 集/帧 | dev 集/帧 | calibration 集/帧 |
| --- | ---: | ---: | ---: | ---: |
| LIBERO | 40 | 1392 / 204301 | 189 / 28394 | 190 / 28436 |
| RoboTwin | 13 | 500 / 116117 | 65 / 15414 | 67 / 15413 |

所有任务均有三个分区；源 group 不跨区、读回 hash 不变。可用源身份已恢复，场景/物体/会话缺失字段保持 unknown，未宣称 object-disjoint。P7 正式初态测试清单仍需在 P7 独立封存。

每个平台另选 5 个不同任务，覆盖 5 个照明域、不同序列长度与首/中/尾 anchor，实际读取 `ManifestWindowDataset`；逐相机 RAW 转换、本体和 checkpoint 归一化动作与直接 NPZ 索引比较，全部最大误差为 0。RoboTwin 最后一帧对应未来动作全 padding，有效监督数为 0；LIBERO 最后一帧仍有 1 个有效动作。未将尾部填充值当成监督。

正式统计文件：

- LIBERO：`/workspace/owac/rawvla/checkpoints/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json`，quantile，保留既定夹爪处理。
- RoboTwin：`/workspace/owac/rawvla/checkpoints/pi05_robotwin/policy_norm_stats.json`，MEAN_STD/zscore，加性 epsilon `1e-8`。

两个新 remote config 均固定实际路径；已运行 CLI inventory 检查。曝光元数据只用于分组和审计，不自动注入推理或校正低照度输入；若 P1 使用已知曝光标定，必须另列信息条件和对照。数据缺 timestamp，时间依据仍为帧索引；RoboTwin 当前本体仍是作者 `actions[t]` 代理。

只在 `/workspace/owac/n02-p0-independent-audit-20261014/` 写入本次代码与收据。未改冻结基线目录、未启停评测服务、未使用远端 GPU 训练。

## 3. 数值参考与运行时

SAVi 使用 Flax0.10.2/JAX0.4.38 CPU 参考环境，pinned 原算子不改动。Theseus 使用 pinned 源码与 Python3.10.19/torch2.11.0+cu128，官方 20 步示例和 FP64 dense LM 均实跑；不声称 CHOLMOD/稀疏 CUDA solver 已验收。

Gen 使用 Julia1.10.10 与固定 `2a27298b392e89fa67837f8871b624a41d901d19`。4096 粒子、种子1103、12步的平均 TV 为 0.0144559，最大 TV 为 0.0268004；边界0.05预先写入运行脚本。该检查仅验证有限 HMM 参考，未宣称变维提议或 P3 能力。

官方 ConvNeXt-V2 的 MinkowskiEngine gitlink 实为 `shwoo93/MinkowskiEngine@bbc30ef581ea6deb505976b663f5fc2358a83749`，普通 PyPI 包不能代替其定制 depthwise kernel。CPU 构建虽可完成，该定制算子明确不支持 CPU。CUDA12.8 编译使用已归档的显式 include、关闭 NVTX 编译选项及 C++ 智能指针所有权兼容补丁；补丁不改卷积/归一化数学。失败尝试和最终证据分别留档。最终官方稀疏 CUDA 对照：四个 stage 有效坐标完全一致，前向最大绝对误差 `2.3841858e-7`，输入梯度及全部参数梯度通过原容差，参数梯度最大误差 `3.7252903e-8`。SAVi 原无 mask 路径的 SlotAttention/GRU/Processor/predictor 均通过；Theseus 官方 FP64 参考最大误差 `2.0205539e-13`。

## 4. 资源画像与预算

实测使用本地 RTX5090、FP32、batch2；每个可训练组件 warmup20次后测200次真实 AdamW 更新，总220个 step。FCMAE 用自身重建 loss；其他组件用明确的 output-square 探针 loss。此处更新验证运行成本，不是 P1 正式训练，也不证明任务能力。forward 另行 no_grad 测量，不能与训练更新混报。

最终数值见同名新收据目录的 `profile.json` 与 `profile-forward.json`，同次测量如下（batch2，FP32）：

| 模块 | 训练更新 p50 / p95（ms） | 峰值 allocated（MiB） | forward p95（ms） |
| --- | ---: | ---: | ---: |
| ConvNeXt RAW4 Base | 24.468 / 25.580 | 2035.0 | 6.451 |
| ConvNeXt RAW3 Base | 24.520 / 26.634 | 2034.8 | 6.462 |
| FCMAE full（dense 稀疏数学移植） | 27.213 / 31.288 | 2295.2 | 7.363 |
| SAVi 512 / 16槽 / 4 predictor块，T=8 | 37.412 / 39.883 | 529.9 | 9.971 |

这些是整批次模块耗时，不能直接当成逐 episode 闭环时延。模型初始化及输入使用固定种子1103。P3—P5 完整结构尚未实现，未提供虚构全模型画像。冻结开发预算为单 episode 决策 p95≤200ms、持久状态≤256MiB；这是后续实现要满足的边界，P1/P2 必须实测完整决策和持久状态，当前单模块 profile 不表示预算已通过。

## 5. 验收与复现

新收据目录：`research/experiments/results/2026-10-14_n02_p0_independent_acceptance/`。原始日志、完整 fixtures、manifest 与 anchor 文件：`outputs/n02-full/p0-independent-audit/`。脚本 `reproduce.sh` 和环境 README 记录绝对解释器、固定源码及实际命令。

实际最终检查：全仓 **640 passed / 6 skipped**；目标 Python3.10/CUDA 运行时新模块 **257 passed / 1 skipped**；`make check` **exit 0**（264 文件）。与入场复跑的 631 passed / 6 skipped 比较，新增9项针对真实缺陷的回归。6个 skip 分别来自本地缺远端 NPZ（2）、缺 StarVLA/策略资产（2）、根环境缺 numba（1）、根解释器无 CUDA（1）；远端 NPZ、参考 numba 和目标 CUDA 另有本次实际运行证据，未把这些 skip 改成假通过。保留的1条 warning 来自旧 N02 测试，新的 parity 已修复 requires-grad 标量转换。

P1 直接使用本次 v2 manifest、`frozen-data.json` 指定的全部 408075 个锚点与新 checkpoint 归一化入口；不能沿用旧 report 的 split。实际数据在已核实的远端，本地没有复制全量 RAW；正式训练应使用该主机的实际数据路径或先完成可访问的数据接入。曝光标定是否作为已知输入仍是 P1 显式对照条件；P7 初态测试和物体留出不能从本次数据审计推导。基础代码、协议、数值一致性与小批次画像的验收不能替代行为或闭环验收。未启动100k训练，未提交、推送或创建自动任务。

## 6. 提交归档

按用户后续指令提交本轮修改：契约文档 `c3c0d2503142e9ba1532f14fefaa924a18dfa8ed`，已验收实现 `5ddfdb569528ff38075fb976cf8c3ae087fb2af0`。上述“未提交”和起点 SHA 描述的是验收发生时的历史状态，原收据保持不变；代码内容由 `source_manifest.json` 的90个文件哈希核对。上游逐字节源码、许可证及兼容补丁上下文保留原空白；本项目代码的 `make check` 通过。大型产物、运行时、完整锚点和原始日志留在本地，Git 仅保存代码、锁、紧凑清单及验收结论。

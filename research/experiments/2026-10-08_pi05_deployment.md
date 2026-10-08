# 2026 年 10 月 8 日 π0.5 后端部署验证

实验 ID：`pi05-deployment-2026-10-08`。状态：完成。日期时区：Asia/Shanghai。证据层：实现与部署。执行：Codex 本地部署任务。机器人、相机与正式数据均未选择。

## 验证目的

验证固定版本的官方 π0.5 可以在本机 PyTorch 环境加载预训练权重、产生有限且形状正确的动作，并通过官方 WebSocket 协议调用。结果用于为 RAW 网络开发建立实际后端起点，不检验候选架构创新、RAW 感知或机器人任务成功率。

## 代码与环境版本

- 最终复现的 OWAC 代码 SHA：`b13efae869a34ffe655d0ea54307a95054f033e3`，分支 `codex/pi05-backend`；执行该次复现前工作树干净。
- 首次三次计时验证的 HEAD 为 `d2a90c3813eed1e718d95ce561fa489ea8d9348b`，执行代码当时未提交；源码快照为 `outputs/pi05-smoke/pi05_source.py`，SHA256 为 `45daa5fe060d03ac79f85c9d3cc5d819af92ed55f9b08f495c136a00d8d9193d`，相关差异保存于同目录 `source.diff`。
- openpi 固定为 `15a9616a00943ada6c20a0f158e3adb39df2ccac`，模型实现无本地修改；环境补丁与来源记录在 `configs/backends/pi05-runtime/`。
- LeRobot 固定为 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，使用官方源码归档；两个源码归档的 SHA256 验证通过。
- runtime 锁文件 SHA256：`6e1dfbbe9507c57f61d2e217b184e29942eb06550d3a9b87c72651f2e2031aa5`。
- 环境：Ubuntu 24.04.5、Python 3.11.14、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Transformers 4.53.2 加固定官方补丁、JAX 0.5.3（CPU）、Orbax 0.11.13。完整依赖以 runtime 锁文件为准。
- 硬件：RTX 5090 32 GB、驱动 595.91.07；CPU/系统内存具体型号与峰值未测量。GPU 与 CUDA 的实际运算验证通过。

## 权重与输入配置

本次复用 `/home/jeong/.cache/openpi/openpi-assets/checkpoints/pi05_droid` 的已有 Orbax 缓存，登记的官方来源为 `gs://openpi-assets/checkpoints/pi05_droid`。没有重新比对远端对象，转换器使用固定 openpi 提交的官方实现。

转换后的 `model.safetensors` SHA256：`07094e2d4d85c81444d341443fdcaf863c05232835ab7376805193b749b17e41`。DROID 归一化统计 SHA256：`403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b`；与源检查点相同。转换输出采用 BF16，并复制当前检查点自身的 assets。

推理配置为 `pi05_droid`：`pi05=True`，`gemma_2b` 语言模型与 `gemma_300m` 动作专家，15 步动作 horizon，32 维内部动作，最终 8 维 DROID 输出，语言最大长度 200，`discrete_state_input=True`。关闭 `torch.compile`，flow matching 10 步；其余变换由固定上游配置与本检查点统计确定。没有训练或调参。

Smoke 的生成器为该版本 `owac.backends.pi05.smoke`，种子 0。输入为两路 224×224×3 的随机 uint8 RGB、7 维零关节状态、1 维零夹爪状态及指令 `pick up the cup`。固定 float32 流噪声形状 `[15, 32]`；DROID 适配器另添加无效的第三相机占位。无真实数据清单或 train/validation/test 拆分，原因是纯软件部署检查。

## 实际命令与产物

工作目录：`/home/jeong/zeno/owac/repo/`。

```bash
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 doctor
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 convert --checkpoint /home/jeong/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output outputs/checkpoints/pi05_droid_pytorch
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 smoke --checkpoint outputs/checkpoints/pi05_droid_pytorch --output outputs/pi05-smoke --seed 0 --repeats 3
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 serve --checkpoint outputs/checkpoints/pi05_droid_pytorch --port 36891
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 smoke --checkpoint outputs/checkpoints/pi05_droid_pytorch --output outputs/pi05-smoke-final --seed 0 --repeats 1
```

首次 smoke 报告、源码快照与动作分别位于 `outputs/pi05-smoke/report.json`、`pi05_source.py`、`actions.json`；转换记录位于 `outputs/checkpoints/pi05_droid_pytorch/owac_conversion.json`。最终提交代码的单次复现记录为 `outputs/pi05-smoke-final/report.json` 和 `actions.json`。输出目录已存在时不会覆盖，后续复现使用新 run 目录。

WebSocket 客户端调用 `/healthz`，以 `WebsocketClientPolicy(host="127.0.0.1", port=36891)` 完成元数据握手，再使用固定上游 `make_droid_example()`（NumPy 种子 0）调用 `infer`。检查动作形状/有限值及平台元数据为空；结果位于 `outputs/pi05-websocket-report.json`。测试服务验收后以 Ctrl+C 关闭，未发送机器人指令。

## 实测结果

| 检查或资源 | 实际结果 | 范围 |
| --- | --- | --- |
| 权重加载与动作输出 | `[15, 8]`，所有值有限 | 合成 RGB 参考策略 |
| 首次运行内重复性 | 固定输入/噪声的 3 次输出完全相同 | 预热后重复调用 |
| 提交后跨进程复现 | 最终单次输出与首次结果完全相同 | 固定代码/锁文件/权重/输入/噪声；单次报告本身将运行内重复性标为 `not_tested` |
| 同步调用耗时 | 125.84、120.64、128.87 ms | 首次 smoke 的 3 次调用，包括变换与 CPU 回传，排除权重加载/I/O |
| 首次 smoke 预热 | 533.21 ms | 1 次；不外推为稳定启动时延 |
| 最终提交代码单次调用 | 133.80 ms | 独立进程的单次复现，不作统计比较 |
| PyTorch 分配/保留显存峰值 | 7,645,805,568 / 7,776,239,616 bytes | 含模型与推理；约 7.12 / 7.24 GiB，不等于整机全部显存 |
| WebSocket 健康检查/握手/推理 | 通过；客户端首调 334.67 ms | 含客户端与本地网络路径，输入与 smoke 不同 |
| 核心工程检查 | `make check` 通过，3 项 CPU 测试通过 | 环境补丁隔离、缓存原文件和路径边界 |
| 来源与重建检查 | 源码归档校验、环境锁检查及从干净源码归档重建环境补丁通过 | 重建得到相同 openpi 环境配置 |
| 数据加载器 | 导入通过 | 未加载正式数据、未训练 |

CPU 峰值、全流程安装/转换总资源、系统能耗、完整机器人通信和闭环成本均未测量。三次计时的分位数只描述本次 smoke，不作为正式部署时延或架构效率主张。

## 失败记录与下一步

GitHub Git 传输在本机超时，改为复用已核对提交的本地 Git 对象与固定官方源码归档；不使用旧 checkout 的未提交模型修改。首次策略服务尝试的端口 8765 被占用，改用本机空闲端口 36891 完成验收，未干预原服务。

结果说明 π0.5 软件后端在当前本机环境可用。RAW/外部特征入口尚未实现，真实 RAW、语义对齐、训练梯度和机器人动作正确性仍待各自验证。下一步实现绕开 RGB 图像预处理及视觉塔的特征入口，同时明确保留语言、所选本体状态和动作 flow 的协议；本体、相机和数据选型继续留空。

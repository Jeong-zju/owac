# π0.5 本地 PyTorch 部署

本地部署先运行官方 π0.5 的 RGB 参考推理，作为后续 RAW 网络接入的已知后端。用户选择 PyTorch；机器人、相机和正式数据配置留空。部署流程不发送机器人控制命令。

## 代码与独立环境

| 项目 | 固定来源或版本 |
| --- | --- |
| openpi | [官方 15a9616a00943ada6c20a0f158e3adb39df2ccac](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac)，Apache 2.0；随源代码保留 `LICENSE_GEMMA.txt` |
| LeRobot | [官方 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5](https://github.com/huggingface/lerobot/tree/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5)，Apache 2.0 |
| Python | 3.11，独立于 OWAC 核心开发环境 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| Transformers | 4.53.2，加同一 openpi 提交提供的 5 个替换文件 |
| JAX / Orbax | 0.5.3 / 0.11.13，使用 CPU 完成兼容代码与检查点转换 |
| 完整解析环境 | `configs/backends/pi05-runtime/pyproject.toml` 与同目录 `uv.lock` |

本机为 Ubuntu 24.04、RTX 5090 32 GB、驱动 595.91.07。官方文档的已测试系统为 Ubuntu 22.04；本机兼容性以实际部署检查结果为准。CUDA 12.8 的 PyTorch 2.7 支持 Blackwell。[官方 PyTorch 说明](https://pytorch.org/blog/pytorch-2-7/)

第三方代码放在被忽略的 `third_party/openpi` 与 `third_party/lerobot`；主仓库保存来源、版本、获取校验值和环境补丁。当前 openpi 从本机已有源码仓库的已提交版本克隆，未复制该旧 checkout 的未提交修改，并已核对官方提交身份。LeRobot 使用官方固定提交的源码归档。

openpi 的环境补丁只将 JAX CUDA 依赖改为 CPU 依赖，并让 LeRobot 指向固定的本地源码；原模型实现保持上游版本。补丁保存为 `configs/backends/pi05-runtime/openpi-environment.patch`。根项目显式选择 PyTorch CUDA 12.8 和本地来源，其余环境由 runtime 锁文件记录。

## 首次获取与安装

以下命令适用于尚未获取第三方代码的干净 checkout；已有目录应先核对来源和修改，不覆盖现有工作。源码归档的固定 URL 与 SHA256 见 `configs/backends/pi05-runtime/sources.json`。

```bash
cd /home/jeong/zeno/owac/repo
curl --fail --location https://codeload.github.com/Physical-Intelligence/openpi/tar.gz/15a9616a00943ada6c20a0f158e3adb39df2ccac --output third_party/openpi-15a9616.tar.gz
curl --fail --location https://codeload.github.com/huggingface/lerobot/tar.gz/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 --output third_party/lerobot-0cf8648.tar.gz
```

核对两个归档的 SHA256，解压并应用环境补丁：

```bash
sha256sum --check configs/backends/pi05-runtime/sources.sha256
mkdir third_party/openpi third_party/lerobot
tar -xzf third_party/openpi-15a9616.tar.gz --strip-components=1 -C third_party/openpi
tar -xzf third_party/lerobot-0cf8648.tar.gz --strip-components=1 -C third_party/lerobot
git apply --directory=third_party/openpi configs/backends/pi05-runtime/openpi-environment.patch
```

随后执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --locked --no-dev --project configs/backends/pi05-runtime
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 patch
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 doctor
```

补丁命令只允许修改当前虚拟环境中的 Transformers 文件，并用原子替换保留共享缓存的原文件。安装采用 copy 模式，避免官方 README 中直接覆盖硬链接文件可能污染其他环境的问题。再次 sync 或重装 Transformers 后重新运行 patch。[官方安装说明](https://github.com/Physical-Intelligence/openpi#pytorch-support)

## 当前权重与离线推理

本机已有 `pi05_droid` 和 `pi05_libero` 的 Orbax 缓存，本次离线检查使用前者。登记的官方来源为 `gs://openpi-assets/checkpoints/pi05_droid`；本次复用本地缓存，没有重新比对远端存储对象。转换产物与其归一化统计计算 SHA256，保存在输出目录中。

```bash
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 convert \
  --checkpoint /home/jeong/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
  --output outputs/checkpoints/pi05_droid_pytorch

uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 smoke \
  --checkpoint outputs/checkpoints/pi05_droid_pytorch \
  --output outputs/pi05-smoke \
  --seed 0 --repeats 3
```

转换调用固定版本的官方脚本。该版本从 `checkpoint.parent/assets` 查找统计文件，OWAC 包装入口改为明确复制当前 `checkpoint/assets`，避免缺失或误用邻近检查点统计。原缓存保持只读，转换输出已存在时明确拒绝覆盖。

Smoke 使用合成 RGB、零本体状态与固定流噪声，保留 DROID 配置的 15 步 horizon 和 32 维内部动作，最终输出 8 维动作。检查输出为 `[15, 8]`、所有值有限，固定输入的多次动作输出一致。`torch.compile` 在此检查中关闭，flow matching 使用 10 步；性能优化另行测量。

`report.json` 保存环境、源码 SHA/差异快照、实际执行模块、运行锁与检查点哈希、动作形状、同步计时和 PyTorch 分配/保留显存峰值。动作保存在 `actions.json`。计时包括 `policy.infer` 的输入/输出变换和 CPU 回传，排除权重加载与磁盘 I/O；3 次样本的分位数只描述该 smoke，不能代表正式机器人时延。失败同样保存报告。

## 本地策略服务

已验证的同一 RGB 参考策略也可以通过官方 openpi WebSocket 协议提供给本地客户端。服务默认监听 `127.0.0.1:8000`，前台运行，按 Ctrl+C 退出：

```bash
uv run --locked --project configs/backends/pi05-runtime python -m owac.backends.pi05 serve \
  --checkpoint outputs/checkpoints/pi05_droid_pytorch
```

客户端使用 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 的 `get_server_metadata()` 与 `infer(observation)`；观测格式见固定版本的 `droid_policy.py`。服务元数据将本体、相机、数据保持为空，并明确标记 RGB 参考路径。

## RAW 网络接入审计

以下位置均基于上述固定 openpi 提交，适用于 PyTorch π0.5：

| 位置 | 当前行为 | RAW 接入要求 |
| --- | --- | --- |
| `policies/droid_policy.py` 的 `_parse_image` 与 `DroidInputs` | RGB 浮点图像可能转换为 uint8；组装 RGB 图像字典和相机掩码 | 主 RAW 路径使用独立的观测适配器，保留无损 CFA 采样与元数据 |
| `training/config.py` 的 `ModelTransformFactory` 与 `transforms.py` | 调用图像 resize、语言 tokenization、状态 padding 和归一化 | 保留适用的语言/本体处理；RAW 输入避免进入 RGB resize/图像变换 |
| `models/model.py` 的 `Observation.from_dict` | 图像归一化到 `[-1, 1]`，PyTorch uint8 图像可能转换布局 | 新 RAW/特征观测不能通过伪装 RGB 图像复用该路径 |
| `models_pytorch/preprocessing_pytorch.py` | 固定图像键与 224×224，训练时执行图像增强 | 训练和推理都要提供独立路径，不只修改推理端 |
| `PI0Pytorch.embed_prefix` 与 `PaliGemmaWithExpertModel.embed_image` | SigLIP 和视觉投影输出 2048 维特征，与语言嵌入拼接成前缀 | 在视觉塔输出进入语言前缀的位置接入候选读出；保留有效性与 attention mask，分别处理空间/相机/时间信息 |
| `PaligemmaTokenizer` | 当前 `pi05_droid` 配置将归一化本体状态离散化后写入语言前缀 | 保留所选检查点的状态协议，复核 `discrete_state_input`；不能套用 π0 的连续 state token 假设 |
| `sample_actions`、`forward` 与 `policy_config.create_trained_policy` | 推理采用前缀 KV cache 和动作 flow matching；训练/推理均经过上述观测处理 | 特征入口同时支持动作推理与训练；登记编码器、读出、语言模型和动作专家的参数边界 |

接口位置的依据是 [官方 PyTorch 模型](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py)及 [官方策略加载](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/policies/policy_config.py)。本次完成源码审计，RAW/外部特征入口尚未实现。

官方固定版本的 PyTorch 训练尚未提供 LoRA/FSDP 和混合精度训练。后续优先验证编码器/读出训练与明确的局部解冻；需要 LoRA 或更大训练范围时单独实现/选型并验证。本次未运行训练。[官方 PyTorch 功能边界](https://github.com/Physical-Intelligence/openpi#pytorch-support)

## 当前验收记录

代码版本、完整复现条件、实测资源和失败记录见 [部署验证记录](../research/experiments/2026-10-08_pi05_deployment.md)。

- 独立 runtime 安装完成，固定 Transformers 补丁和 RTX 5090 上 CUDA 矩阵运算通过。
- 官方权重转换完成；推理输出 `[15, 8]`，有限值及固定输入/噪声的 3 次重复性检查通过。
- 同步 `policy.infer` 调用为 125.84、120.64、128.87 ms；单次预热 533.21 ms；PyTorch 分配/保留显存峰值约 7.12 / 7.24 GiB。结果保存在 `outputs/pi05-smoke/report.json`。
- 官方 WebSocket 服务的健康检查、空平台元数据及客户端推理通过；检查后已停止测试服务。默认启动端口可通过 `--port` 修改。
- `make check` 与 3 项部署补丁 CPU 测试通过；训练数据加载器导入通过，训练未执行。
- 机器人、相机、真实 RAW 数据和训练均未验证；相关平台配置留空。

# N04 远端开发与训练规约

用户指定优先在数据所在服务器执行实际训练。使用：

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

`-L` 只提供本地浏览器访问远端 localhost:8080 的通道，不是训练端口或数据目录；没有浏览器服务时普通 SSH 即可。每个阶段先读远端 `/etc/vast-agents-guide.md` 与目标目录适用 AGENTS。下列资产已在 2026-10-14 核对；每次启动重新验证存在性和实际版本。

## 路径与环境

| 用途 | 远端路径 |
| --- | --- |
| 冻结 RawVLA 源码参考 | `/workspace/owac/rawvla/source`，HEAD `ba9c56b04da2c4931b70d8844710f5928d43775b` |
| StarVLA 独立 checkout | `/workspace/owac/rawvla/source/starVLA`，HEAD `57d508cb180d0510ac9060358009c0f6ffaac420`；Python 包实际在其下 `starVLA/` |
| LIBERO RAW | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/libero/raw` |
| RoboTwin RAW | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/robotwin2/raw` |
| StarVLA 后端解释器 | `/workspace/owac/rawvla/envs/starvla/bin/python`，Python 3.10.21 / torch 2.7.1+cu128 / numpy 1.26.4 |
| RoboTwin 仿真环境 | `/workspace/owac/rawvla/envs/robotwin`，运行时核对解释器与依赖 |
| LIBERO 权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO 统计 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/dataset_statistics.json` |
| RoboTwin 权重 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors` |
| RoboTwin 统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 原冻结协议/测试清单 | `/workspace/owac/rawvla/records/pi05-full-20261010-v3/` 下的 `protocol.json`、`libero-manifest.json`、`robotwin-manifest.json` |
| 最新完整结果/审计 | `/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1/` 下的 `complete-summary.json`、`identity-audit.json`、`checksums.json` |
| 旧 N04 接入证据 | `/workspace/owac/rawvla/research/n04-acceptance-20261012/` |
| 本次正式 N04 根目录，拟创建 | `/workspace/owac/research/n04-v1/` |

新根目录内按 `source/<snapshot_id>/`、`runs/<stage>/<run_id>/`、`data/`、`envs/`、`manifests/` 分开。原数据只读复用；源码先形成不可变快照，再启动训练。训练期间本地继续编辑不会改变正在运行的模型。不要在冻结 RawVLA 源码上直接开发。

S0 为纯 encoder 创建独立 Python 3.11+ 环境并固定依赖，与本地 pyproject/uv.lock 一致；先检查服务器已有相容解释器与 wheel 缓存。后端连接继续用原 Python 3.10 runtime，通过 `PYTHONPATH=<snapshot>/src:/workspace/owac/rawvla/source/starVLA` 导入必要 OWAC 模块。新后端可达模块避免 Python 3.11 专有 API；在这两个解释器下分别 import/compile/执行小测试。若确需新增库，为 N04 建独立可复现的后端环境并验证与旧环境数值一致；不升级共享 starvla/robotwin 环境。

纯 encoder 环境在成功建立后写入 `manifests/environment.json` 的 `encoder_python`；后续提示词的 `$N04_PYTHON` 从该字段读取。不能把 shell 默认 `python` 当指定环境，也不因本地有 RTX 5090 就把正式数据搬回本地。

## 同步实际工作树

不要只 `git archive HEAD`：当前已有未提交但必要的实现与文档。S0 实现并测试 `scripts/n04_full/package_source.py`，收集 tracked 文件的当前字节，以及显式纳入的未跟踪 `src/`、`tests/`、配置和文档；排除 .git、.venv、outputs、数据、大权重、密钥和外部缓存。输出 tar.gz 与逐文件 SHA256、HEAD、tracked diff hash、纳入的 untracked 清单。用命名为 `n04-s0-<UTC时间>-<包hash前12位>` 的 snapshot_id。后续每阶段重复这一打包流程。

已有同步工具可以复用，但必须证明它包含实际未提交源码。以下是人工复制一个已生成包的可用方式，变量按真实输出填写：

```bash
scp -P 44327 "$N04_ARCHIVE" root@107.209.104.125:/workspace/owac/research/n04-v1/manifests/
ssh -p 44327 root@107.209.104.125
# 在远端：先核对包 sha256，再解到一个新的 source/<snapshot_id>/，核对逐文件 hash。
```

数据无需 rsync。小报告、曲线、配置与清单复制回本地 `research/experiments/results/<run>/`；大 checkpoint、测量缓存、日志保留远端并写 hash 和恢复路径。传输脚本使用结构化参数或正确 shell quoting，不把 JSON 字符串当 shell escaping。

## 启动与续训

先执行 `nvidia-smi`、`supervisorctl status`、相关 PID/cmd 检查、磁盘检查。本轮看到一张 96 GB 级 GPU 和约 32 GB 已占用；不假设全部显存空闲，不停其他项目服务。每个正式模型先以完整结构 profiling 500–1000 步；通过 microbatch、梯度累积、activation checkpointing 达到有效 batch，不把 Base 缩成 tiny。

S0 准备 N04 专属 supervisor 配置模板：`program=owac-n04-<stage>-<run_id>`，固定源码目录、绝对解释器、CUDA device、完整配置与 output，训练进程前台运行。使用 `autostart=false`、`autorestart=false`；中断由 resume checkpoint 明确恢复，避免崩溃后反复重跑或重复采样。输出重定向到该 run 的日志；记录 supervisor 名称、PID、命令和启动时间。只加载/启动本任务的 program，不执行影响全部服务的 restart/update。大批任务顺序排队，不因多个 seed 同时启动挤爆单卡。

首次短 run 成功后执行实际阶段训练。每 5k 更新评价开发集并保存状态；长期作业要留下可查 PID、已完成步数、checkpoint 与报告。断开 SSH 后继续由 supervisor 管理。资源不足时按已测吞吐记录剩余 GPU 小时与待完成任务，不把短 smoke 重命名为正式验收。

每个 checkpoint 保存 model、optimizer、scheduler、AMP scaler、Python/NumPy/Torch CPU 与全部 CUDA RNG、sampler 顺序/游标、global step、epoch、训练配置和数据身份。训练默认在完整片段边界保存；恢复时从已记录位置重新做 burn-in，不需要把随机别的片段状态当作前一历史。若支持片段中间保存，就同时恢复该批对应的动态状态和 sampler 位置。

正式训练、推理与数据包同步是收到相应阶段开发指令后的工作。本提示词准备轮没有启动这些作业。S7 的完整 rollout 数量很大，按该阶段预注册矩阵执行，不能用本机 toy 速度估算排期。

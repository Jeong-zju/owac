# N01 v1.0 远程训练与执行说明

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

真实训练优先放在用户指定服务器，数据与现成 π0.5 权重已经在那里。2026-10-14 本次只读 SSH 核查成功；没有启动新训练、安装依赖或停止服务。

## 连接与已核实位置

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

8080 仅用于可选的本地训练界面转发。命令行核查不需要 `-L`；若本地端口占用，可省略，或用 `-L 18080:localhost:8080`。进入服务器先完整读 `/etc/vast-agents-guide.md`，再运行 `supervisorctl status`、`nvidia-smi`、`df -h /workspace`。

| 资产 | 已核实路径 |
| --- | --- |
| RawVLA 总目录 | `/workspace/owac/rawvla` |
| 作者源码 | `/workspace/owac/rawvla/source` |
| 数据根 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data` |
| LIBERO RAW | 上述数据根的 `libero/raw`，本次清点 1,771 个 NPZ；运行前核对冻结清单 |
| RoboTwin RAW | 上述数据根的 `robotwin2/raw`，本次清点 632 个 NPZ；运行前核对冻结清单 |
| StarVLA Python | `/workspace/owac/rawvla/envs/starvla/bin/python` |
| RoboTwin runtime | `/workspace/owac/rawvla/envs/robotwin` |
| LIBERO 权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| RoboTwin 权重 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| LIBERO 统计 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/dataset_statistics.json`，`stats_key=openpi_pi05_libero_franka`、`action_stats_key=action`，沿用已验收配置 |
| RoboTwin 统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/policy_norm_stats.json` |
| 完整 Default 数字 | `/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1/` |

`RawVLASequenceDataset(root=...)` 默认推导的是 `root/data/...`，与服务器实际数据根不同。必须像现有 `configs/encoders/n01/rawvla-*-smoke.json` 一样显式传 `raw_cache_root` 与 `stats_path`。不要因为默认路径不存在重新下载 700 GB 数据或改旧目录。

当前实测 GPU 为 RTX PRO 6000 Blackwell，总显存 97,887 MiB，核查时已用约 32,047 MiB；这是瞬时余量。旧策略服务仍为 RUNNING，旧 N01 DBSR 训练已 EXITED。实际训练前重新测余量，把已有服务占用计入预算；不要自行停止其他项目服务。

StarVLA 实测 Python 3.10.21 / torch 2.7.1+cu128 / CUDA 可用。现有本地 `uv.lock` 的 encoder extra 固定 CPU torch 2.9.1，与此不同。

只读快照见 [remote_snapshot.json](remote_snapshot.json)。本次服务器 complete-summary 与 identity-audit 的文件 SHA256 均与主 checkout 留档相同；这里确认文件一致与清单状态，不重新运行或重新证明策略成绩。

## 独立执行布局

P0 建立以下新路径；这些是拟新增目录，不是本次已部署资产：

```text
/workspace/owac/rawvla/research/n01-v1/
  envs/encoder/                 # 独立 GPU 编码器/SEA-RAFT/Mamba 环境
  envs/policy/                  # 有新依赖时创建的独立兼容策略环境
  runs/<run_id>/source/         # 本次源码、配置与上游锁快照
  runs/<run_id>/outputs/        # 日志、曲线、checkpoint、逐事件记录
  runs/<run_id>/receipts/       # 小型来源、成本和验证记录
```

从当前 worktree 显式打包 `src/owac`、本阶段 configs、需要的 tests、上游许可/源码和锁；记录 HEAD、dirty diff 与所有实际文件 hash。包含新增未跟踪源码，不能只 `git archive HEAD` 丢掉实现，也不复制 `.venv`、outputs、其他方向文件或密钥。远程验证解包 hash 后启动。

本地用于代码/FP64 数值与回归检查；正式 GPU 运行在独立环境。复用现成 StarVLA 环境进行只读兼容检查；若要安装 SEA-RAFT/Mamba 等新包，放新环境。P7 若需新依赖，依据服务器已固定 requirements 建独立策略环境并保留 Python 3.10 兼容；通过 `PYTHONPATH=<新快照>/src:<作者源码>/starVLA` 加载本项目。项目常规安装要求 Python 3.11，不要把这个约束强行灌进冻结策略环境。

Blackwell 需要适用 GPU 架构的 PyTorch/CUDA 构建；P0 先验证 CUDA forward/backward 和真实 shape。选定兼容版本后保存单独 GPU requirements/lock，不让本地 CPU index 覆盖远程 CUDA torch。记录 Mamba/causal-conv1d 编译参数；不在原全量基线环境上升级。

## 长任务启动与恢复

每个长任务由独立 supervisor program 管理，名称 `owac-n01-v1-<stage>-<run>`。包装脚本调用明确的 Python、源码根和配置，以前台进程运行；日志同时写本次 outputs 与 supervisor 日志。任务配置保存后按服务器 guide 注册。命名、源快照和输出目录三者能互相对应。

一次只启动资源核验允许的作业；三种子可以串行。保存一个可续跑矩阵调度器，由配置里的已完成/进行中身份决定下个 run；退出后不得自动从 step 0 覆盖旧任务。先用短开发运行验证 checkpoint 恢复，再启动正式预算。长期训练的数据和权重留服务器，本地同步数字、源码清单和曲线摘要。

恢复读取同一源码/数据/校准版本和优化状态；更改配置或数据建立新 run，登记 parent。服务端读出绑定 loopback，已有 `-L` 足够查看界面，不需为训练新开公开无认证端口。

P8 的新模型评测另建源码快照、清单与端口；原 `pi05-full-20261010-v3` 和完整 Default 数字只读复用。不要把新模型写进原冻结输出目录。

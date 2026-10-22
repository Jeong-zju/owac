## 远程训练和长期运行

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

### 连接和端口

用户提供的连接命令为：

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

**2026-10-15 本次只读核查：远端 8080 已被 Jupyter 占用，18080 未监听。** 不要停止 Jupyter 或抢占其端口。监控页默认绑定远端 `127.0.0.1:18080`；用户在自己的电脑另开终端运行：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -p 44327 root@107.209.104.125 -L 8080:127.0.0.1:18080
```

随后打开 `http://127.0.0.1:8080/`。若本机 8080 已被上一条 SSH 占用，先结束**用户自己的那条转发连接**，或改 `-L 18080:127.0.0.1:18080` 并打开 `http://127.0.0.1:18080/`。执行前重新查端口；若远端 18080 也被占用，选空闲端口并明确给出新的转发命令。不要为了监控改成公开 `0.0.0.0`、关闭现有认证或另建复杂登录系统。

进入服务器完整阅读 `/etc/vast-agents-guide.md` 和 supervisor 包装脚本说明 `/opt/supervisor-scripts/README.md`。再检查：

```bash
supervisorctl status
nvidia-smi
df -h /workspace
ss -ltnp 'sport = :18080'
```

本次核查为 RTX PRO 6000 Blackwell，约 96 GiB 显存，瞬时占用约 15.1 GiB；磁盘约剩 1.2 TiB。旧 RoboTwin policy 服务仍在运行。上述仅是准备时快照，正式运行前重测，不停止其他任务/管理服务。服务器指南说明容器回收/销毁可能丢失 `/workspace`，应查是否有持久卷，重要源码、配置和恢复点及时保存到用户已有存储，不自行购买服务。

### 已有资产和新输出位置

| 用途 | 路径或约定 |
| --- | --- |
| RawVLA 作者源码 | `/workspace/owac/rawvla/source` |
| RAW 数据根 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data` |
| LIBERO / RoboTwin 缓存 | 上述目录下 `libero/raw`、`robotwin2/raw` |
| encoder Python | `/workspace/owac/rawvla/envs/encoder/bin/python`；本次 Python 3.11.16 / torch 2.9.1+cu128，CUDA 可用 |
| policy Python | `/workspace/owac/rawvla/envs/starvla/bin/python`；本次 Python 3.10.21 / torch 2.7.1+cu128，CUDA 可用 |
| RoboTwin runtime | `/workspace/owac/rawvla/envs/robotwin` |
| LIBERO 权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO 统计 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/dataset_statistics.json`，既有 `openpi_pi05_libero_franka` / `action` |
| RoboTwin 权重与统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`、同目录 `policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 原完整 Default 记录 | `/workspace/owac/rawvla/records/pi05-complete-report-20261014-v1/`，只读 |
| P2 训练/审计 | `/workspace/owac/rawvla/research/n01-v1/runs/p2-motion-20261014/`、`p2-audit-20261015/`，只读 |
| 本阶段新 run | `/workspace/owac/rawvla/research/n01-v1/runs/v1-1-<stage>-<run-id>/` |

`RawVLASequenceDataset(root=...)` 的默认路径推导与实际服务器布局有差异。显式传正确 `raw_cache_root`、`stats_path` 及清单侧别，不因默认路径找不到就重下数据。权重和统计在本阶段启动时逐项检查；本表不是“所有未来文件必然存在”的保证。

新 run 建 `source/`、`config/`、`outputs/`、`receipts/`。从当前 checkout 复制本任务实际源码、配置、需要的 vendor 和许可证，包含未跟踪的新代码；仅 `git archive HEAD` 会丢掉未提交实现。不要打包 `.venv`、700 GB 数据、密钥、其他方向输出。简单记录版本与 diff，确认远端加载的 `owac.__file__`、入口和有效配置正确即可，不新建逐文件哈希硬门槛。

编码器环境与固定策略环境有不同 Python/torch 版本。先复用已通的桥接方式；若必须在同一进程做端到端 autograd，建立能同时导入完整编码器和固定 π0.5 的独立兼容环境，验证真实 forward/backward。**不通过 HTTP 特征服务训练端到端梯度**，HTTP 仅用于独立闭环推理。新依赖不升级旧基线环境；本地 CPU torch 锁不覆盖远端 CUDA wheel。Blackwell 使用兼容 cu128 或经实测可用的 wheel，不装/升级宿主 NVIDIA 驱动。

### 作业和恢复

训练、评测队列、监控分别由 supervisor 管理，命名 `owac-n01-v11-<stage>-<role>-<run>`。包装脚本前台启动、实时 flush 日志。只添加自己的配置并用 `supervisorctl reread`、`supervisorctl update <本任务program或group>` 定向加载；不重启整个 supervisor，不操作已有管理/基线服务。仅 SSH 私有转发的服务不需要修改 Caddy/portal，也不要套会因没有 portal 条目而跳过启动的 guard。

训练保存 model、optimizer、scheduler、混合精度 scaler、RNG、sampler/数据进度及配置；checkpoint 用临时文件完成后替换。正常完成的有限队列不被 supervisor 无限重启；崩溃重启从最近有效 checkpoint 接续，同 run 不从 step 0 覆盖。当前仅调度每平台/方法1个训练种子，默认101；不自动排211/307。提前估算速度与总成本；不要在剩余显存不足时同时塞进多个 π0.5。

对相同恢复点做一次短续跑验证即可，普通浮点差异不要求跨机器/版本逐位一致。训练/评测队列与网页解耦：网页关掉不停止训练；训练退出后网页仍显示最后数据和真实退出原因。资源暂时被占用时网页显示“等待 GPU”，不假装正在训练。

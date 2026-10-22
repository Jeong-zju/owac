# N02 v1.1：远程开发执行说明

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

2026-10-14 用户指定：**所有阶段优先在数据所在服务器开发、测试、训练和评测**。本文件与共同契约一并执行。连接命令：

```bash
ssh -p 44327 root@107.209.104.125 -L 8080:localhost:8080
```

`-L` 只把本机 8080 转发到远端 8080，不挂载文件，也不意味着该端口已有应用。已在该主机的会话直接继续；自动化 SSH 命令可省略转发，避免重复占用本机 8080。需要网页时让本阶段应用监听远端 `127.0.0.1:8080`，先核对端口占用。

## 1. 代码位置与首次接入

| 用途 | 位置 |
| --- | --- |
| 远端仓库容器 | `/workspace/owac/` |
| 远端主 checkout，首次接入时创建 | `/workspace/owac/repo/` |
| **远端 N02 开发 worktree，首次接入时创建** | `/workspace/owac/worktrees/n02-measurement-completion/` |
| 开发分支 | `lzh/n02-measurement-completion` |
| 本地交接副本 | `/home/jeong/zeno/owac/worktrees/n02-measurement-completion/` |
| 已存在的 P0 远端审计快照 | `/workspace/owac/n02-p0-independent-audit-20261014/` |

本次只读 SSH 核实：主机 `2566cc983a37` 可连接，数据及下表权重路径存在；远端 `repo/`、`worktrees/` **尚未创建**。P0 审计目录仅有部分源码/配置/收据，不是完整 Git checkout。下述是交给 DeepSeek 的首次接入步骤，不表示本次修改提示词已完成部署。

P0 已验收代码提交 `5ddfdb569528ff38075fb976cf8c3ae087fb2af0`；包含验收归档的本地 HEAD 为 `861c46f67fff552c81fff89c5116c63b815224c5`。先读[独立验收](../../../research/experiments/2026-10-14_n02_p0_independent_acceptance.md)。P1 在此基础上续做；P0 的远端运行时迁移检查不重置已通过的算法/分区。

**首次接入，先在本地交接 worktree 执行以下打包命令。** Git bundle 带已提交历史；tar 另带本轮尚未提交的提示词、P0 冻结锚点和参考 fixtures。虚拟环境、RAW 数据和大权重均留在原机器，不复制本地 worktree 的 `.git` 指针。

```bash
cd /home/jeong/zeno/owac/worktrees/n02-measurement-completion
git status --short --branch
git rev-parse HEAD
git bundle create /tmp/n02-v11-source.bundle lzh/n02-measurement-completion
tar -czf /tmp/n02-v11-handoff.tar.gz \
  docs/prompts/n02_v1_1 \
  docs/prompts/README.md docs/prompts/DeepSeek_N02_implementation.md \
  outputs/n02-full/p0-independent-audit/frozen-data \
  outputs/n02-full/p0-independent-audit/fixtures
sha256sum /tmp/n02-v11-source.bundle /tmp/n02-v11-handoff.tar.gz
scp -P 44327 /tmp/n02-v11-source.bundle /tmp/n02-v11-handoff.tar.gz root@107.209.104.125:/tmp/
```

**再在远端执行。以下建仓命令只适用于上述目录不存在的首次接入。** 若已经存在正式 worktree，读取其 AGENTS/status、HEAD 和修改，直接续做，按文件比较后合入新提示词；不要再次 clone 或覆盖正在开发的文件。

```bash
cat /etc/vast-agents-guide.md
sha256sum /tmp/n02-v11-source.bundle /tmp/n02-v11-handoff.tar.gz
mkdir -p /workspace/owac/worktrees
git clone --no-checkout --branch lzh/n02-measurement-completion \
  /tmp/n02-v11-source.bundle /workspace/owac/repo
git -C /workspace/owac/repo checkout --detach lzh/n02-measurement-completion
git -C /workspace/owac/repo worktree add \
  /workspace/owac/worktrees/n02-measurement-completion lzh/n02-measurement-completion
cd /workspace/owac/worktrees/n02-measurement-completion
tar -xzf /tmp/n02-v11-handoff.tar.gz
git status --short --branch
git rev-parse HEAD
cat AGENTS.md README.md docs/status.md
```

两端 archive SHA 应相同。bundle 的 origin 是本地传输文件，不用它推送；本次阶段任务也不要求推送。首次接入后**远端 worktree 是活跃开发副本**，日常直接在那里修改；本地用于阅读、提示词修订及验收。需要本地历史未提交参考时只传本阶段引用的文件，例如 P1 的 `src/owac/evaluation/n02_refutation_boundary.py` 与对应实验记录，并记录 hash；不打包整个工作空间。

## 2. 直接使用已存在的远端数据和策略资产

| 资产 | 已核实绝对路径 |
| --- | --- |
| 数据集根 `data.cache_root` | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench` |
| LIBERO RAW | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/libero/raw` |
| RoboTwin RAW | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/robotwin2/raw` |
| LIBERO 正式归一化统计 | `/workspace/owac/rawvla/checkpoints/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json` |
| RoboTwin 正式归一化统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/policy_norm_stats.json` |
| LIBERO 策略权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| RoboTwin 策略权重 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 既有 StarVLA 源码 | `/workspace/owac/rawvla/source/starVLA` |
| 既有策略/仿真运行时，仅作为依赖参考 | `/workspace/owac/rawvla/envs/starvla`、`/workspace/owac/rawvla/envs/robotwin` |

数据根不是旧记录的 `/workspace/owac/rawvla/data/`。从仓库的 `configs/encoders/n02_full/p0-remote-libero.json` / `p0-remote-robotwin.json` 继承数据及归一化配置；其中 CPU 是 inventory 配置，不是正式训练设备。

P1 及后续消费 `research/experiments/results/2026-10-14_n02_p0_independent_acceptance/manifest-{libero,robotwin}.json.gz` 和 `frozen-data.json` 指向的六个 anchor 压缩文件，总计 408075 个锚点。先比对 manifest/解压 JSON/gzip SHA，再读取。P2/P6 若需更长历史，在同一分区和锚点身份上显式生成新版本 WindowOptions/anchor 表，记录新旧 hash；P0 当前 `rnn_frames=1`，不能把旧收据说成已经冻结 16 帧 burn-in。完整 RAW 不传回本地，不重新下载已有约数百 GB 缓存。

## 3. 在远端建立 N02 专用运行时

参考[已验收环境说明](../../../configs/backends/n02-full-runtime/INDEPENDENT_ACCEPTANCE.md)及同目录锁文件。那些 `/home/jeong/...` 解释器是**本地历史收据**，不能在远端直接调用；重新建立环境，不复制 venv 或本地编译的 `.so`。

在远端 N02 worktree 首次创建 GPU 组件环境的起点：

```bash
cd /workspace/owac/worktrees/n02-measurement-completion
uv venv --python 3.10.19 outputs/n02-full/runtime/py310-cu128
uv pip install --python outputs/n02-full/runtime/py310-cu128/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r configs/backends/n02-full-runtime/requirements.lock.txt
export N02_ROOT=/workspace/owac/worktrees/n02-measurement-completion
export N02_PY="$N02_ROOT/outputs/n02-full/runtime/py310-cu128/bin/python"
export PYTHONPATH="$N02_ROOT/src"
"$N02_PY" -m compileall -q src/owac
"$N02_PY" -m owac.runtime.n02_full_probe --device cuda \
  --output outputs/n02-full/remote-bootstrap/probe.json
"$N02_PY" -m pytest tests/encoders/n02_full -q
make check
```

环境已经存在且锁/探针相符时复用。`make check` 使用远端根 uv 环境（Python≥3.11，CPU）；模型训练和 GPU 测试显式使用 `N02_PY`，不靠当前 shell 默认 `python`。P0 已测 Python3.10.19 / torch2.11.0+cu128；远端硬件重新记录，**本地 RTX5090 的耗时不是远端测量**。本次只读核实 GPU 为 RTX PRO 6000 Blackwell（97887 MiB），既有 StarVLA 解释器为 Python3.10.21，源码 HEAD 为 `57d508cb180d0510ac9060358009c0f6ffaac420`。既有策略运行时与 P0 Python3.10.19 环境分别记录，不合称同一个环境。SAVi/Theseus/Gen 的参考环境按已验收锁分别重建，原始 P0 失败记录不再当作未解决结论。

E1/E2/P6 在 N02 独立运行时补齐固定 StarVLA 依赖并锁定；源代码版本核对 `02_REFERENCES.md` R25，不能只看目录名。先跑同进程 RAW前端→readout→π0.5 loss→前端梯度及采样，确认桥可用后开始阶段训练。需要适配 StarVLA 源码时，在 N02 的 ignored `third_party/` 建固定版本副本。直接复用数据和权重；环境包、候选源码及候选服务各有独立位置，不修改冻结基线。

## 4. 远端运行与交接

所有阶段相对路径均以远端 N02 worktree 为根。输出为 `/workspace/owac/worktrees/n02-measurement-completion/outputs/n02-full/<stage>/<run_id>/`；报告、小型清单放该 checkout 的 `research/experiments/`。参考源码缓存也放该 checkout 的 ignored `third_party/` 或既定 `outputs/` 缓存，固定 SHA 不变。

长任务使用远端独立的 N02 作业配置，保存启动命令、解释器、工作目录、PID/作业名、日志、checkpoint 和恢复命令；按主机作业管理方式运行，SSH 断线后可恢复。先查询当时 GPU/内存/磁盘与既有作业，选可用资源；资源不足时继续实现及 CPU/数据检查，训练单列待运行，不停他人或冻结评测进程。8080 用于可选可视化隧道，策略服务另选空闲端口。

交接提供远端 HEAD、工作区 diff、改动文件哈希、完整命令、实际数据/权重/配置 hash、绝对产物路径与阶段指标；只回传必要源码补丁和紧凑收据到本地供验收。生成权重、全量数组/视频和原始日志保留远端。连接失败先报告实际 SSH 错误并继续可做的代码工作；本地没有 RAW 副本不再作为跳过远端数据验证的理由。远端未跑的测试明确记未运行。

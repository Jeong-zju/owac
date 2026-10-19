# N04 v1.0 S0 · 工作流 E 交付记录（ws-infra）

状态：**交付完成，等待 Lead 验收**（不自行宣布阶段通过）。
checkout：`/home/jeong/zeno/owac/worktrees/n04-egomotion-separation`，分支 `lzh/n04-egomotion-separation`，HEAD `fd6255b769d661af568738bec194bb599601a644`。
远端根：`/workspace/owac/research/n04-v1/{source,runs,data,envs,manifests}/`（本轮新建）。
全部远端操作只读他人资产；未注册/启动任何 supervisor program；未停止他人 GPU 作业；未 `git commit/push/stash/checkout/restore`；未运行 `make check` 或 `ruff format .`。
实际命令全文见同目录 `commands.txt`；原始记录见同目录 `frozen-assets.json`、`environment.json`、`source-snapshots.json`、`<snapshot_id>.sync-record.json`、`<snapshot_id>.verify.json`。

## 1. 交付件

| 文件 | sha256 | 状态 |
| --- | --- | --- |
| `scripts/n04_full/package_source.py` | `7d7e852950703685f9a309614f39f2a86744fea0a905675f791a1d450ea9d441` | 新增，已 lint/测试 |
| `scripts/n04_full/n04_supervisor.conf.template` | `25b03f44f5e17b8909406f667321ac0bdfee10e82faa0c08ff936f0f8274f61e` | 新增，未注册 |
| `tests/encoders/n04_full/test_packaging.py` | `198a342a56f0b964c31fe691cf8fce802948adf8c5272e03d23b85a30400090f` | 新增，8 项测试通过 |
| `research/experiments/results/2026-10-14_n04_s0_infra/*` | 见 `checksums.sha256` | 本轮小结果与回传清单 |

未改动任何非本人文件（`src/owac/**`、`research/theory/*`、`configs/n04/full/*`、`docs/plans/**`、`docs/prompts/**` 全部未动）。

## 2. 源码快照（工作流 E 项 1、3）

工具能力：收集 tracked 文件的**当前字节** + 显式纳入的未跟踪 `src/`、`tests/`、`configs/`、`scripts/`、`docs/`、`research/`；排除 `.git`、`.venv`、`outputs`、`build`、缓存、数据与大产物、密钥；输出 `tar.gz` + 逐文件 SHA256 清单 + `manifest.json`，其中记录 `HEAD`、`tracked_diff_sha256`、纳入的 untracked 清单与排除原因；`snapshot_id = n04-s0-<UTC>-<包 hash 前 12 位>`；同一 `--timestamp` 下字节确定（两次构建 sha256 完全一致，测试覆盖）。

未提交源码的显式证明（manifest `uncommitted_evidence`，`verified=true`）：

- `modified_tracked = 6`，`untracked_not_in_head = 58`；
- 样本：`configs/n04/full/protocol.json`，sha256 `f8e9ce9c109d86626a9c9e733d87aff3cdbdbc114ab467f124a78e2cd2da4061`，`present_in_head=false`，`present_in_manifest=true`。

权威快照：

| 项 | 值 |
| --- | --- |
| snapshot_id | `n04-s0-20261014T040846Z-30640b9274ed` |
| 包 sha256 | `30640b9274ed209ab25728dfb7119cba57cae41eb6692505a067ca7c21bf0f53` |
| 包大小 | 8,117,011 B |
| packed | 781（tracked 723 + untracked 58） |
| 排除 | 1（`outputs/README.md`，`deny-dir:outputs`）；另有 1 条 warning：tracked 的 `research/.../four-trajectories.png` 命中媒体规则仍被打包 |
| HEAD / diff | `fd6255b…` / `tracked_diff_sha256=5f3c6d017c2d951206e8364206d3791c7b77bbd989fe7270dc97f0c8c8401a43` |
| 构建期稳定性 | `stability.stable=true`，`unstable_files=[]` |
| 远端位置 | `source/n04-s0-20261014T040846Z-30640b9274ed/`，包在 `source/<id>.tar.gz`，清单在 `manifests/` |

远端逐文件复核（`n04-s0-20261014T040846Z-30640b9274ed.sync-record.json`）：`PACKAGE_SHA_MATCH=yes`（远端 `sha256sum` 重算等于声明值）；`extracted_regular_files=781` = `counts.packed`；`sha256sum -c` → **781 OK / 0 FAILED**；用 `manifests/environment.json` 的 `encoder_python` 跑 `package_source.py verify` → `checked=781, mismatched=0, missing=0, extra=0, differences=0, package_sha256_match=true`。

被取代的两个快照（保留在远端作历史，**不得用于启动阶段**，见 `manifests/source-snapshots.json`）：

- `n04-s0-20261014T035852Z-906f1376bda2`：缺少 supervisor 模板契约测试（复核仍为 766/766 OK）。
- `n04-s0-20261014T040701Z-bd7782632fc6`：复核 **failed_lines=1** —— `tests/encoders/n04_full/test_mechanism_data.py` 在「先逐文件 hash、后写 tar」两趟之间被其他 agent 重写，导致 tar 内容与 manifest 不一致（真实竞态，非传输错误）。据此把工具改为**同一次读取中计算 hash 并写入归档**，并新增构建后稳定性复查与对应回归测试（`test_stability_flags_a_file_rewritten_during_packing`）；重建后差异归零。

## 3. 远端设施与纯 encoder 环境（工作流 E 项 2）

- 目录已建：`source/`、`runs/`、`data/`、`envs/`、`manifests/`（另注：`data/meta-scan-*.json`、`manifests/s0-manifest/` 来自其他工作流，本轮未动）。
- 探测事实：服务器仅有 `/usr/bin/python3.12`（3.12.3）；无 `~/.cache/uv`；`/root/.cache/pip` 约 44 MB；`uv 0.12.10`；磁盘 1.3 TB 可用。
- 独立环境：`uv venv --python /usr/bin/python3.12 /workspace/owac/research/n04-v1/envs/encoder`，按本地 `uv.lock`（`encoder` extra + `dev`）锁版本安装 `numpy==2.5.1`、`torch==2.14.1`、`pytest==9.1.1`，索引 `https://pypi.tuna.tsinghua.edu.cn/simple`（与 lock 一致），共 35 包，`envs/encoder.freeze.txt` 固定，占用 5.3 GB。
- `manifests/environment.json` 字段 **`encoder_python = /workspace/owac/research/n04-v1/envs/encoder/bin/python`**（`realpath /usr/bin/python3.12`）。
- 实测：`-V` → `Python 3.12.3`；`import numpy` → `2.5.1`；`torch 2.14.1+cu130 / cuda 13.0 / available True`，arch list 含 `sm_120`；`pytest 9.1.1`；小脚本输出 `array_sha256=ee6adb5c714cff4f22526ac6f637221a07343b8077e13d533427f46ac9ebd388`、`eig_min_max=[0.25859175550658114, 2.7973249503841604]`；GPU 计算实测 `Conv2d` forward+backward（`weight_grad_norm=0.20188182592391968`）与 512×512 矩阵乘，`capability=12.0`，峰值分配 37.13 MB。
- 共享环境未改动，且复核版本未变：`starvla` = Python 3.10.21 / numpy 1.26.4 / torch 2.7.1+cu128；`robotwin` 环境未触碰。

## 4. 冻结资产核对（工作流 E 项 4，只读）

`manifests/frozen-assets.json`（本地副本同名），声明值 vs 重算值 **7/7 全部一致**：

| 文件 | bytes | 重算 sha256 |
| --- | --- | --- |
| `pi05-complete-report-20261014-v1/complete-summary.json` | 253,420 | `df95081e6063aac017a4c12938f67e702c260123be1ef37b190042725b8beb7e` |
| `…/identity-audit.json` | 28,265 | `770e5494dc26f293e975aa40156b2d2b5e784a1e97995705507916f4d9265e85` |
| `…/checksums.json` | 481 | `af9ecb1dbf69b1e7d176cee5ebbab35711c80430ac4575fd31eba9caa210ff76` |
| `pi05-full-20261010-v3/protocol.json` | 3,650 | `3ac9626ef71eab4719be540393f51b66b767d7348dc4cac0ce5f5455f5b52e0e` |
| `…/libero-manifest.json` | 5,500,078 | `e3a054f43060912d668e57d35873b573a289c6d3c941fec9c029569c82971168` |
| `…/robotwin-manifest.json` | 2,627,420 | `632b8360836595a868affffd24f9ecf1262d68a52a505ef5ffe801598d8317e5` |
| 权重 LIBERO（bfloat16/model.safetensors） | 8,522,039,320 | `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8` |
| 权重 RoboTwin（checkpoints/pi05_robotwin） | 9,354,050,752 | `9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3` |

身份/分母结论（引用，不重跑）：

- `identity-audit.json`：`state=complete`、`errors=[]`、`expected=completed=13250`；更新于 `2026-10-14T02:44:00Z`。
- `complete-summary.json`：`version=owac-rawvla-pi05-complete-20261014-v1`、`created_at=2026-10-14T02:44:24Z`、`main_completed=13250`、`original_conditions_completed=632`、`extra_training_steps=0`、`limitations=5` 条。
- 分平台：LIBERO 10000/10000（success 5820）、RoboTwin 3250/3250（success 508）；附加原条件 632/632（success 108）。
- 源码身份：`/workspace/owac/rawvla/source` HEAD `ba9c56b04da2c4931b70d8844710f5928d43775b`、`starVLA` HEAD `57d508cb180d0510ac9060358009c0f6ffaac420`，与配置/文档预期一致。
- 结论：报告与冻结协议、manifest、两平台权重在文件与身份层面自洽（声明的 sha256 全部被重算验证）；本轮**没有**重跑任何基线，因此不扩大证据范围——仍属仿真 Default PI05 完整清单结果，不代表 N04 能力、learned ISP、原生 Bayer 或实机。

当前 `supervisorctl status` 快照：10 个 RUNNING，其中既有 policy 服务 4 个（`owac-rawvla-full-libero-policy` pid 48102、`owac-rawvla-full-robotwin-policy` pid 47920、`owac-rawvla-libero-policy` pid 47080、`owac-rawvla-robotwin-policy` pid 47002），其余为 `caddy`、`cron`、`instance_portal`、`syncthing`、`tensorboard`、`tunnel_manager`。
`nvidia-smi` 快照：`0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 86882 MiB used, 10368 MiB free, 100 %, 595.84, 46`。
注意：显存占用在核对瞬间已达 ~86.9 GB，仅剩 ~10.1 GB 空闲（preparation_audit 记录为 ~32 GB 已占用）。本轮未停止任何作业，S1 起排资源必须重新测量。

## 5. supervisor 配置模板（工作流 E 项 5）

`scripts/n04_full/n04_supervisor.conf.template`：`[program:owac-n04-__STAGE__-__RUN_ID__]`；`command` 绝对解释器 + 前台 `-u -m <entry> --config <snapshot>/configs/n04/full/<stage>.json --output <run_dir>`；`directory` 固定为不可变快照目录；`environment` 含 `PYTHONPATH=<snapshot>/src`、`CUDA_VISIBLE_DEVICES`、`N04_PYTHON/N04_SOURCE_DIR/N04_SNAPSHOT_ID/N04_RUN_DIR`；`autostart=false`、`autorestart=false`、`startretries=0`；`stopasgroup=true`；`redirect_stderr=true` 且日志重定向到 `<run_dir>/logs/`。文件头写明「只对该 program 做 `reread`/`update`/`start`，禁止全局 restart/update」与 `sed` 安装步骤。
本轮**未**把模板复制进 `/etc/supervisor/conf.d/`，**未**执行 `reread/update/start`。模板经 `configparser` 渲染解析测试（program 名、生命周期、绝对解释器、CUDA device、日志路径、无残留占位符）。

## 6. 未完成 / 未验证项

1. **快照是时间点记录**：构建后其他 agent 继续修改工作树，当前 tracked diff hash 已变为 `e465d467d70f749fcd4d35da52099a7800c6730b1ba9379458c5edd7180b2523`，与快照记录的 `5f3c6d01…` 不同。S1 启动前必须用同一命令重新打包 + 同步（工具与流程已验证，一次构建约 1 分钟）。
2. supervisor 模板**只做过本地渲染解析**，未由 supervisor 自身校验、未注册、未启动——按 S0 约束有意为之；`startsecs=30`、`stopwaitsecs=600` 等取值未在真实训练进程中验证。
3. `torch` 的 GPU 路径仅做了 `Conv2d`/matmul 级 smoke；正式 ConvNeXt-B 单步显存/吞吐 profiling 属 S1，本轮未做（也不应做）。
4. 环境安装用了清华镜像（与 `uv.lock` 的 registry 一致），未逐 wheel 比对 hash；版本与 lock 一致已记录，但「与 lock 字节级同一制品」未验证。
5. 未验证 `data/` 目录内容（`meta-scan-*.json` 属其他工作流）；未对 NPZ 数据本身做任何核对。
6. 冻结资产只做 hash/身份核对，未重跑任何 rollout；`limitations` 的 5 条限制未被本轮证据推翻或加强。

## 7. 需要 Lead 决策的点

1. **权威快照的时点**：是接受 `n04-s0-20261014T040846Z-30640b9274ed`（S0 证据），还是要求 S1 启动前统一重打包一次以纳入其他工作流后续代码（推荐后者；两者不冲突，本快照作为 S0 证据留档）。
2. **是否清理由本轮产生的两个被取代快照**（`source/n04-s0-20261014T035852Z-906f1376bda2/`、`source/n04-s0-20261014T040701Z-bd7782632fc6/` 及其包/清单）。我按「不做破坏性操作」保留并在 `manifests/source-snapshots.json` 标注，但其中一个按设计会复核失败，若下游脚本用 glob 取快照可能误用。
3. **显存余量**：核对瞬间仅约 10.1 GB 空闲且 GPU 利用率 100%（4 个既有 policy 服务在跑）。S1 的资源计划需要决定是否等待、降 batch 或与这些服务错峰——超出我的权限范围。
4. `manifests/environment.json` 的 `encoder_python` 已被 S0 固定；若 S1 需要额外依赖（例如 ConvNeXt 相关），按规约应新增独立环境而非就地升级，请确认这条约束继续有效。

## 8. 追加：被取代快照的防误用标记（Lead 裁决 2026-10-14）

按 Lead 裁决不删除被取代快照，改为就地标记（权威快照树**未触碰**，仍为 781 文件、verify 0 差异）：

| 远端绝对路径 | bytes | sha256 |
| --- | --- | --- |
| `/workspace/owac/research/n04-v1/source/n04-s0-20261014T035852Z-906f1376bda2/SUPERSEDED` | 767 | `d52da266e0afc00e4499f91fb98610527f1c7e6d0d3cda0b40f407f921b4061c` |
| `/workspace/owac/research/n04-v1/source/n04-s0-20261014T040701Z-bd7782632fc6/SUPERSEDED` | 428 | `69691a75e5881537f4dc90176cc2daaa278ab69c0f8e3c7854a84acad4476e72` |
| `/workspace/owac/research/n04-v1/source/n04-s0-20261014T040701Z-bd7782632fc6/DO_NOT_USE` | 928 | `a308fd685fe1f8d4440befdb6bec1a1cb00d99131f6a0d302a9306c52e18f585` |

标记时间 `2026-10-14T04:12:44Z`；`manifests/source-snapshots.json` 已追加 `markers` 字段与各快照的 `markers`/`verify_note`（两个被标记的树今后 `verify` 会多出 1 个 `extra_count`，属预期）。
`manifests/source-snapshots.json` sha256 = `3609cd8a69b4c117e68d4736e04d75acef8cab9fd359c5ad4de6a12128ac168a`（远端与本地副本一致）。
Lead 另确认：S0 不动 GPU 资源，显存余量写入 S1 资源约束；S1 新依赖继续新建独立环境，绝不就地升级共享 starvla/robotwin 环境（`uv.lock` 固定 torch 2.14.1，与远端 encoder env 2.14.1+cu130 一致）。

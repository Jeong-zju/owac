# N01 v1.2 P3 RUN_STATE

最后更新：2026-10-15（本地 `Asia/Shanghai`）。本文件是 P3 的恢复入口：上下文压缩或 SSH 重连后**先读这里**，再查进程与 checkpoint，不要重复开同一训练。

## 1. 当前活动

| 项 | 值 |
| --- | --- |
| 阶段 | P3（完整编码器 + 同源四臂 + 真实训练 + 开发闭环） |
| 状态 | 实现完成、两平台真实数据验证通过、**主体训练已在两个 GPU 队列中运行** |
| 本地 checkout | `/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`，基点 `d3dec90` |
| 远程部署根 | `/workspace/owac/rawvla` |
| 远程源码 | `/workspace/owac/rawvla/owac/src/owac`（**正式导入路径**） |
| P3 run 根 | `/workspace/owac/rawvla/research/n01-v1/runs/v1-2-p3-main/` |
| 监控端口 | 远端 `127.0.0.1:18080`（回环，SSH 转发查看） |

### 必须知道的两件事

1. **`/workspace/owac/rawvla/owac/owac/` 是一个多余副本，会遮蔽正式代码。**
   它不在 git 索引内，是同步时 `rsync src/ <repo>/` 造成的。因为 `python -m` 会把 CWD
   放在 `sys.path` 最前，只要 CWD 是 `/workspace/owac/rawvla/owac`，导入就会命中这个
   旧副本而不是 `src/owac`，症状是报出与当前源码不符的行号和 `AttributeError`。
   **规避（已完成，未删除任何文件）**：从 `/tmp` 之类的目录启动，用绝对路径调用包装脚本，
   并显式设置 `PYTHONPATH=<repo>/src`。是否清理该副本请由部署负责人决定。

2. **两个 Python 环境不要混**：编码器/训练用 `envs/starvla/bin/python`（3.10，带
   π0.5、safetensors、torch 2.7.1+cu128）；`envs/encoder`（3.11，torch 2.9.1+cu128）
   用于 DBSR/SEA-RAFT 研究。端到端训练必须与 π0.5 同进程。

## 2. 复现命令

```bash
# 环境（每次新 shell）
export REPO=/workspace/owac/rawvla/owac
export DEPLOY=/workspace/owac/rawvla
export N01_RUN=$DEPLOY/research/n01-v1/runs/v1-2-p3-main
export PYTHONPATH="$REPO/src"
cd /tmp            # 关键：不要在 $REPO 里启动，避免 owac/owac 遮蔽
wrap="$REPO/configs/backends/n01-v12-runtime/run-on-gpu.sh"   # raw=GPU0, rgb=GPU1

# 1) 写八个主体 run 的配置（已执行，幂等）
$DEPLOY/envs/starvla/bin/python -m owac.training.n01_v12 matrix --run-root "$N01_RUN"

# 2) 真实数据验证（不改参数、不训练）
bash "$wrap" raw $DEPLOY/envs/starvla/bin/python -m owac.training.n01_v12 verify \
  --repo-root "$REPO" --deploy-root "$DEPLOY" \
  --config "$N01_RUN/config/libero-raw_n01-seed101.json" \
  --device cuda:0 --output "$N01_RUN/receipts/verify-libero-raw_n01-seed101.json"

# 3) 主体训练（每个平台/方法/种子一次；此命令是模板，参数以实跑为准）
bash "$wrap" raw $DEPLOY/envs/starvla/bin/python -m owac.training.n01_v12 train \
  --repo-root "$REPO" --deploy-root "$DEPLOY" \
  --config "$N01_RUN/config/libero-raw_n01-seed101.json" \
  --run-root "$N01_RUN" --output "$N01_RUN/outputs/libero-raw_n01-seed101" \
  --device cuda:0 --lr 1e-5 --accum 4 --tbptt 8 --log-every 10 \
  --ckpt-every 500 --max-steps 0 --gpu-index 0

# 4) 续训：同一条命令加 --resume
  --resume "$N01_RUN/outputs/libero-raw_n01-seed101/checkpoints/latest.pt"
```

`--max-steps 0` 表示不设上限，按登记的 GPU 预算停止。续训接 `latest.pt`，不从 step 0 覆盖。

## 3. 输出目录约定（每个 run）

```
$N01_RUN/
  config/      八个 arm 的有效配置（含 model_factory 字段）
  outputs/<platform>-<arm>-seed101/checkpoints/latest.pt     model/optimizer/RNG/config
  receipts/    matrix.json、verify-*.json、train-*.json
  monitor/     workers/<role>.json、status.json、metrics.jsonl、logs/、cases/、artifacts/
```

## 4. 已完成矩阵项

| 项 | 状态 |
| --- | --- |
| 本地 `tests/encoders/n01_full` + `tests/integration` | 417 passed / 5 skipped（2 个 CUDA-only、3 个需 `OWAC_RAWVLA_SOURCE`） |
| 冻结 ISP 与作者源码核对 | 通过（12 passed，远程实跑） |
| BOCPD 与作者算法核对 | 通过 |
| 四臂工厂/同参数量/支持改变内容 | 通过（10 passed） |
| 远程 run 骨架 | 已建，8 个 arm 配置落盘 |
| 远程真实数据验证 | **已完成**（`receipts/verify-libero-raw_n01-seed101.json`，见下） |
| 8 个主体训练 run | **已启动，未完成**：supervisor `owac-n01-v12-p3-raw` / `-rgb`，每 run 登记 20,000 step |
| 实测吞吐 | 128² 裁剪 × 32 帧：**4.77 s/optimizer step**（accum=4）；完整 256² **OOM**（图 77 GiB） |
| 单 run / 单卡估计 | 20,000 step ≈25 h；单卡 4 run ≈100 h；两卡并行 ≈4.2 天跑完 8 个 run |
| 本轮 100 次开发闭环 | **未开始** |
| 同步可视化案例 | **已生成 30 图**（4 case × 2 相机 × 配对/局部/融合权重/支持区域），`monitor/cases/index.json` 已写入；已逐张目视核对 3 张 |
| 删源干预 | **已跑通**：删重复填充帧=无变化（正确），删最新真实帧 token 变但幅度小（1.4e-5 / flow v 0.0074） |
| `evaluation/n01_v12.py` | **已交付**（validate / intervene / cases） |
| `runtime/n01_v12_service.py` | **已交付并接通**：要求 F 实现 + 10 项契约测试 + **作者 websocket 客户端实测**（8/8 fresh、actions `[1,10,32]`、p50 205.6 ms）；**未接平台闭环** |
| 服务入口 | `serve`（含 `--checkpoint` 恢复训练臂）/ `smoke`（用作者客户端实测） |

### 远程验证实测结果（真实数据、GPU0、Python 3.10、真实 π0.5 权重）

| 项 | 实测值 |
| --- | --- |
| LIBERO train 区样本 | 39,296 |
| 相机序（loader 实读） | `["agentview","wrist"]`，与配置一致 |
| 有效窗口 | `{"agentview": 32, "wrist": 32}`，字段 `[1,32,3,256,256]` |
| RAW 数值域 | float32 `[0.0, 0.4274]`（暗光，uint8 线性 RAW10 的 /255） |
| 同源配对 | `identical_source=true`、`rgb_is_not_identity=true`，provenance 指向真实作者模块 |
| action flow matching loss | 0.40667 |
| encoder / bridge 梯度绝对和 | 20680.8 / 14441.3（真实更新） |
| 后端梯度 | **0.0 → 冻结成立** |
| 视觉 token | `[1,512,512]`（2 相机 × 256） |
| 支持状态 | `applied_v1_2`；agentview accepted 521 / deferred 759，wrist 267 / 1013，**分母 1280**（= 5 历史源 × 256 区域） |
| RoboTwin 复核 | 14,400 样本，3 相机 32 帧，loss 8.32649，梯度 143826/101624，后端 0.0 冻结，出现真实 retracted 600/376/464 |
| 训练中支持确认 | `raw_n01` 日志 `support_applied: {agentview: true, wrist: true}`、`applied_v1_2`；step 20 时 agentview 480 accepted / 3634 deferred / 1518 retracted |

大型 deferred 数是**观测**，不是成绩；默认支持尺度需要用真实 calibration/validation 权重定，
本轮没有为让图好看而人工压低风险值。理论说明 §4.4 曾疑心分母应为 7936（31×256），
已用实测更正：真实窗口只有 6 个唯一帧（27 槽是同一个重复填充帧，`values` 逐元素相同），
故 `T_active = 5`、分母 1280 正确。跨样本唯一帧数为 3–32，`window=32` 是**容量**不是每样本实测量。

**已知未实现缺口（理论复核发现）**：`fixed_decision_replay`/`counterfactual_replay` 只作用于
**解析链**；神经内容链（DBSR/fusion/bridge）**没有**反事实整源重算，"整源删除后神经链被完整重建"
当前**不成立**。这是 P3 要求 E 的未完部分。

本地 `make check` 通过，全仓 **765 passed / 5 skipped**（2 个 CUDA-only、3 个需
`OWAC_RAWVLA_SOURCE`）。验收 diff：`research/experiments/n01_v1_2/P3_worktree_diff.txt`。

## 4b. 训练队列与查看方式

```bash
supervisorctl status | grep p3                      # 两个队列的运行状态
tail -f /workspace/owac/rawvla/logs/n01-v12-p3-raw.log
tail -f /workspace/owac/rawvla/logs/n01-v12-p3-rgb.log
# 队列顺序（当前模型在最前）
#   GPU0: libero:raw_n01 -> libero:raw_base -> robotwin2:raw_n01 -> robotwin2:raw_base
#   GPU1: libero:rgb_n01 -> libero:rgb_base -> robotwin2:rgb_n01 -> robotwin2:rgb_base
# 每个 run 从自己的 checkpoints/latest.pt 续训；队列有限，跑完退出（不无限重启）。
```

监控服务：supervisor program `owac-n01-v12-p3-monitor`（`autostart=true`、`autorestart=true`，
日志 `/workspace/owac/rawvla/logs/n01-v12-p3-monitor.log`），绑定远端回环 `127.0.0.1:18080`。
实测 `/api/status` 返回真实在跑数据（`state=training`、`activity="raw_n01（RAW）正在训练中，step 60"`、
`planned_steps=20000`、`heartbeat_age_s≈5`），`/` 返回页面 200 / 79,520 B。
它**只读**、只写自己的 `monitor/status.json`，监控挂掉不会影响训练。

查看方式（保留 Jupyter 的 8080 不动）：

```bash
ssh -N -p 34302 root@154.59.156.21 -L 18080:127.0.0.1:18080
# 本机打开 http://127.0.0.1:18080/
```

## 4c. 可视化与干预的复现命令

```bash
# 生成同步案例图（30 张）并写 monitor/cases/index.json
bash "$wrap" raw $DEPLOY/envs/starvla/bin/python -m owac.evaluation.n01_v12 cases \
  --repo-root "$REPO" --deploy-root "$DEPLOY" \
  --config "$N01_RUN/config/libero-raw_n01-seed101.json" \
  --device cuda:0 --region validation --scan 4 \
  --output-dir "$N01_RUN/monitor" --output "$N01_RUN/receipts/cases-libero-raw_n01.json"

# 删源干预（slot 1 = 重复填充帧 -> 无变化；slot 31 = 最新真实帧 -> 有变化）
bash "$wrap" raw $DEPLOY/envs/starvla/bin/python -m owac.evaluation.n01_v12 intervene \
  --repo-root "$REPO" --deploy-root "$DEPLOY" \
  --config "$N01_RUN/config/libero-raw_n01-seed101.json" \
  --device cuda:0 --region validation --index 0 --camera agentview --slot 31 \
  --output "$N01_RUN/receipts/intervene-libero-raw_n01-slot31.json"
```

CJK 字体：服务器原本没有（`fc-list :lang=zh` 为 0），中文会渲染成豆腐块。
已取一份 Noto Sans SC 到 `$DEPLOY/assets/fonts/NotoSansSC-Regular.otf`（8.3 MB）；
找不到时 `cases` 自动改用 ASCII 标签而不是画乱码。

## 4d. 两个需要用户决定的缺项（详见 P3_report §1.17，已按实查更正）

1. **8 个 run 需要约 4 天挂钟**（≈5 s/step × 20,000 step × 4 run/卡）。队列已托管、可续训、
   不依赖本会话在线。要更快就需要调低每 run 的 step 预算——**这是登记预算变更，等用户决定**。
   当前进度约 raw 300 / rgb 240（≈1.3%）。
2. **100 次闭环**：原先我把"作者 server 注入口"记成阻塞项，**实查后更正为不是阻塞项**——
   作者 `WebsocketPolicyServer` 类可导入，`runtime/n01_v12_service.py serve` 已按既有
   `n01_feature_service` 的同一模式注入自定义 service，作者树不动，实测已通。
   真正缺的是**一个服务端 payload 适配器**（不是新客户端——作者客户端已内建 RAW 通路）：
   作者 `eval_starvla_openpi_client.py` 支持 `--image-mode rawvla_bench --rawvla-bench-representation raw`
   并构造 `raw_burst_float32`；它发送 `{"examples":[{image,lang,state}], "noise":...}`、
   读 `response["data"]["normalized_actions"][0]`。而 `N01V12Service.infer` 的契约是
   `frames`+`value_units`。需要写一个适配器把前者映射到后者并按 `{"ok":True,"data":{...}}` 回包。
   下一轮先定两点：优先走 `raw_burst_float32`（避免 `preprocess_env_obs` 先 resize 到 224 破坏
   RAW 采样坐标），以及 100 条身份清单 + 两卡分配（`rawvla_full_eval` 只有 10,000 次 P4 清单）。

   **checkpoint 缺口已消除**：`libero-raw_n01-seed101` step≈3,840、`libero-rgb_n01-seed101` step≈3,240
   均已有 `latest.pt`。

   **用户决定（2026-10-15）**：① 训练收窄为每平台只跑 `raw_n01`/`rgb_n01`（4 run），Base 臂推迟
   —— 队列已重启生效；② 授权新写 RAW 感知的 LIBERO 开发闭环通路。

## 4e. 100 次开发闭环的当前进度（本轮推进）

| 环节 | 状态 |
| --- | --- |
| 服务端 openpi 适配器 | **已完成并实测**（10 次到达、`unique_frames={agentview:10,wrist:10}`、p50 416.8 ms） |
| 可用 checkpoint | **已有**（`libero-raw_n01-seed101` step 3500 的 `latest.pt`，服务 metadata 已确认） |
| 评测脚手架 | **已 prepare**：`$DEPLOY/research/n01-v1/runs/v1-2-p3-devloop/`，13,250 条身份 |
| RAW 通路 | **已确认存在**：`run_libero` 已传 `--image-mode rawvla_bench`，只需把 representation 从 `default_isp` 换成 `raw`、端口指向本服务 |
| 开发闭环启动器 | **已完成并正在跑**：`runtime/n01_v12_libero_dev.py`（100 条分层身份、按身份交错分两卡、逐 episode 账本按身份合并去重） |
| 开发闭环主结果（无偏） | **100/100，成功 0，分母 100**（`devloop-balanced/`，5 个照明域各 20 条、4 suite × 10 task，全 0）；均值 29.4 s/episode |
| 开发闭环第一轮（偏暗，保留） | 100/100，成功 0（`devloop/`，旧规则；证据不覆盖） |
| 删源 GPU 实测（两个 regime 都有） | 删已撤回源（slot3）→ 全 0.0（正确的 no-op）；删参考帧（slot31）→ token 变、flow v 差 8.9e-03（**非零但小**） |
| RoboTwin 真实训练 | **已跑通 100/100 step**（loss 0.5554、**27.1 s/step**、checkpoint 已落盘；3 相机支持全部 `applied_v1_2`）→ 两平台训练不再缺一项 |
| 参数量口径 | 同平台内 Base/N01 完全相同（支持零参数）；LIBERO 3,112,354 vs RoboTwin 3,114,403 的差来自**相机数**，不是臂 |
| 预算后果 | RoboTwin 20k step ≈ **6.3 天/run**；LIBERO 先跑完 ~4 天后再加 ~6.3 天，4 个 run 合计 ≈ **10 天**挂钟 |
| RoboTwin 支持尺度 | **已重定并应用**：阈值改为校准分位数（q10/q75 → −3.2117 / 651.3786），accepted 4.6%→**51.0%**、retracted 56.6%→**9.9%**。config 内 `support.calibration` 记录规则与分位数 |
| LIBERO 阈值 | **刻意未改**（默认 q10 差 15%、撤回率 9.7% vs 10.0% 影响可忽略；两臂正在训练，改文件会造成中途换协议）→ provenance 记 `applied: false` |
| RoboTwin smoke checkpoint | 已移到 `outputs/robotwin2-raw_n01-smoke-precalibration/`，**不留给正式 run 续训**（它是校准前训练的，会带混合协议） |
| 新工具 | `owac.evaluation.n01_v12 support-scales`（CPU-only，不占训练 GPU） |
| 分位数规则的适用范围 | **只对零点任意的量成立**（两个 log-odds 阈值）。`large_residual_threshold` 是"残差 > 1σ"的物理陈述，取 q75 会无条件标记最高 25% → **已撤回**，改记为诊断 |
| `region_residual` 6.8 倍差异 | **已用零假设检验查清**（`evaluation/n01_v12_null.py`，CPU-only）：σ=0 两侧精确为 0、两侧都在 **≈1.36 饱和** → **估计量无平台标定错误**；差异来自成像/内容（RoboTwin 低噪声近静止 0.29 < 1.01；LIBERO 反量化+真实运动 1.97 > 1.37） |
| 对 `large_residual_threshold` 的结论 | **不重标定**，但**触发率必须与结果一起报告**；撤回分位数替换的理由更硬 |
| 仍未按数据选定 | `hazard`、`obs_var`、`robust_omega`、`robust_scale`、`large_residual_threshold`（明确记为未选定，未被悄悄替换） |
| 要求 E 神经链整源删除 | **已实现**：`evaluation/n01_v12.py::counterfactual_content_replay` + `intervene --mode {delete,mask}`，7 项测试；真实 checkpoint GPU 实测待下次训练暂停 |
| 选择规则 | **已修正**：任务优先 + 照明域轮转；真实清单上 5 个照明域各 20 条、mean_ev≈0.1–0.4（旧规则偏暗 40/40/20、mean_ev −3.22） |
| 按照明域拆解旧 100 条 | ExtremeLow 0/40、Low 0/40、**Normal 0/20** → **0% 不是照明造成的** |
| 闭环关键限定 | 100 条中 ExtremeLow 40 / Low 40 / Normal 20，`lighting_ev` 均值 −3.22 → 条件偏暗；checkpoint 仅 step 3500/20000 |
| 动作合理性 | 已由客户端 action stats 证实（env ∈ [-1.04,0.78]，夹爪 ±1.0）→ 0% 不是缩放/契约缺陷 |
| 训练恢复 | 编排脚本 `trap` 已自动恢复两个队列，当前 RUNNING 并从 `latest.pt` 续训 |

客户端约束（实测源码）：rawvla_bench 模式按每任务 250 条跑，`--num-trials` 被忽略，
**没有** per-episode 上限；要在 100 条停住需加一个有记录的 `--max-episodes`，或跑完如实报实际条数。

## 4f. 本轮新增：要求 E 神经链缺口已补

被删槽位**根本不进入 `step`** → 不在唯一帧缓冲、不在 BOCPD 到达序列、不参与支持决定、
不可能成为参考；窗口只变短，不从窗外补帧。`intervene --mode delete|mask` 分开命名，
receipt 分别声明 `recompute_scope`，不得混报。

三条实测结论（不是形状断言）：稳定场景下删除有权重的来源确实改变 token；
独立随机帧下支持撤回整段历史、所有权重为 0，删除**正确地**不改变内容；
删除与掩码的**可测**差异在账本（`arrival_index` 5 vs 6、`unique_frames_lifetime` 5 vs 6、
缓冲占用）而不是内容——硬要求内容不同等于断言本架构不承诺的性质。

仍未做：真实 checkpoint 上的 `--mode delete` GPU 实测（需与闭环同样的训练暂停，
32 帧 × 256² 在训练占用下放不下）。

## 4g. 监控页已显示阶段状态与闭环结果（本轮接线）

`/api/status` 实测：`stage=P3`、`state=training`、`dev_loop 100/100`（raw_n01 成功 0/失败 100；
rgb_n01 未运行显示 None）、`tasks=11`（区分 done/running/waiting_resource/partial/missing）、
`artifacts=30`、`runs` 四条且 `suspicious` 全 false。

- 闭环结果经 `RunStateWriter(...).rollouts(...)` 写入（不手改聚合结果，保持单写入者）；
- 阶段状态与任务矩阵经 `write_overrides(tasks=[...])`；
- RoboTwin smoke 的 worker 记录已从正式角色名改名为 `...-smoke-precalibration`，
  避免页面显示"正式 run 已在 step 100 结束"。

查看：`ssh -N -p 34302 root@154.59.156.21 -L 18080:127.0.0.1:18080` → `http://127.0.0.1:18080/`

## 4h. RAW→行动同步时间轴（已交付）

- 服务端 `Snapshot.diagnostics` 记录支持与融合（有界降采样）；
- 客户端注入第三个锚点，录制 RAW 全分辨率 PNG + 诊断 + 动作块；
- `evaluation/n01_v12_timeline.py` 施加**同一个冻结 ISP** 并出图；
- 产出 `monitor/artifacts/timeline_libero_spatial.png`（第 31 张 artifact，GET 200）；
- 该 episode 支持可用比例 0.016–0.128、256/4096 槽位带权重；
- 照明实测：ExtremeLow RAW 均值 1.0/255（近全黑）→ 与 0/100 一起读。

## 4i. 交付前数字复核（本轮）

直接读产物对照报告数字：100/100、成功 0、分母 100、5 域各 20、`representation=raw`、
时间轴 44 查询 / 256-4096 —— **全部一致**。

同时发现并撤下一处**贴错标签的产物**：第一版时间轴图（PNG 命名冲突时生成、帧混了不同 rollout）
在被取代后仍留在磁盘并注册在索引里。已从索引移除、删除旧图（GET 404）、删除陈旧 receipt；
正确的两张图与两段视频保留。当前 artifacts = **34**。

## 4j. 【阻断】独立复核发现两处严重缺陷（我已自行复现）

**缺陷 A：支持判据方向反了。** `support.py:289-300` 定义 `odds = log_short - log_long`
（正 = 最近发生变更），但 `support.py:541` 的 `changed = odds <= retract_logodds`（默认 −1.0）
在 odds **低**（= 稳定）时为真。实测：**静止场景第 9 次到达起全部 RETRACTED、usable 0**；
而第 17 次到达 +5 真变化时反而 **usable 23–45%**。`RETRACTED` 还是单向的，只能从 `DEFERRED` 恢复。
→ `raw_n01` 退化为**单帧编码器**。

**缺陷 B：RGB 臂从未拿到 ISP 输入。** `paired_views` 只被出图/报告路径调用三次，
训练与服务输入路径**零调用**；`THREE_CHANNEL_INPUT_MODES` 里**没有 `rendered_rgb3`**；
`dataset_from_protocol` 不接受 arm/domain；`adapters.py` 里 grep `config.arm` **零命中**。
→ `rgb_n01` 与 `raw_n01` 输入完全相同，**"同机制换输入"这一问从未执行**。
而 `model.py:23-25` 却写着 RGB 臂读 `rendered_rgb3` 张量——**该声明与实现不符**。

**测试为何没拦**：`tests/encoders/n01_full/test_support.py:197` 名字是对的，但静止半边只
`observe` **一次**（首观测 odds 被置 0 故不撤回），缺陷在第 9 次才发作；
位移半边断言"有非 ACCEPTED 状态"，**在反向实现下同样成立**，无法区分。

**对已有结果的影响**：
- **100 次闭环的 0/100 是在缺陷 A 生效时测的 → 不能用来评价支持机制**；
- 正在训练的两个 LIBERO run 带缺陷 A，rgb 臂还带缺陷 B → **不能作为 N01 臂引用**，
  rgb run **无法回答它存在的那个问题**；
- §1.26 把 RoboTwin 的 `retract_logodds` "校准"到 q10，**是在给方向反了的判据调阈值**；
  §1.30 的"估计量无标定错误"仍成立，但**不能**证明判据方向正确。

## 4k. 两处阻断缺陷已修 + 训练已重启（本轮）

用户决定：立刻停两个 run、先修 A+B 再重启。

**停止位置**：raw_n01 **6600/20000**、rgb_n01 **6420/20000**。旧 output 目录移到
`outputs/*-prefix-inverted-support[-and-raw-input]/`（保留证据），
`receipts/restart-after-fix-a-b.json` 记录移动、原因与重启代码的 sha256 前缀。

**缺陷 A（支持方向）**：`changed` 改为 `odds >= retract_logodds`；
`retract_logodds` 默认 **−1.0 → +1.0**（odds 在证据不足时恰为 0，门槛必须严格高于 0）；
校验反转为 **`accept < retract`**；**`RETRACTED` 改为可修订**。
实测：静止 32 帧从"第 9 次到达起 usable 0"变为**全程 usable 100%**；
真位移后 usable 0，新帧自洽后又可恢复。

**缺陷 B（RGB 输入）**：`input_mode` 由**臂声明的输入域**派生（RAW→`rawvla_rgb3`、RGB→`rendered_rgb3`），
变换落在 `DbsrFeatureAdapter.step` 里 `_window()` 之后；`from_dict` 丢弃存储值以免旧配置把 RGB 臂按回 RAW。
实测四臂解析正确，新测试断言两臂输入与 token 均不同。

**一并修掉**：episode reset 漏 `FullState.support`（跨集泄漏）、`shift` 与缓冲不同步、
支持图错切、视频硬编码失败串、伪造零动作、`usable` 口径、融合轴误标、三份配置的旧阈值序。
**更正我自己 §1.30**：零假设探针把噪声注入参考帧，分子分母同缩放，故"饱和⇒估计量无标定错误"**不成立，已撤回**。

**重启状态**：两个 run 均 `STARTING ... (no checkpoint yet)` → **从 step 0 开始**，
GPU 利用率 70–72%。**尚未提交**。

**未修的复核项**已逐条列在 P3_report §1.34 末（`_degrade_to_single_frame` 无调用者、
deadline 从不与耗时比较、`decide(age)` 未使用、`rollouts()` 无生产调用者、
`source_usable` 载荷未限界、出图未按 camera 校验、ISP 与 resize 顺序、弱测试等）。

## 4l. 复核的"惰性代码"已逐条处置（本轮）

**最重要的一条：期限读出（A–F 之一）此前只有声明没有实现。**
`deadline_ms` 从不与耗时比较、`max_snapshot_age_ms` 无人使用、`_degrade_to_single_frame` 无调用者
且其 `[:, -1:]` 切错轴。现已：计算后实测耗时并与 `deadline_ms` 比较；超时按
**未过期旧快照 → 单帧降级 → 明说没赶上** 的顺序处置；年龄界真正生效；回包新增
`deadline_met` / `deadline_elapsed_ms`；`guarantee` 改为如实描述。新增 5 项测试。

其余：`source_usable` 载荷改为精确计数 + 4096 点样本；出图按相机校验并修掉 `fusion` 未清空；
录制改为显式 `--record-timeline`；丢记录不再静默（计数 + `.errors` 边车）；去掉无人读的缩略图；
更正 `P3_SUPPORT_ENABLED` 文档；`decide(age)` 明确标注"透传但不被读取"（不发明用途）；
启动器跑完闭环后自动 `_publish_to_monitor`（失败打印事件）。

**仍未处理**：§3 的弱测试质量问题（恒真断言、只查形状等）大多仍在。

**重启中的训练**：raw_n01 / rgb_n01 均 step 80、RUNNING，从 0 开始（正确代码）。

## 4m. 【重要限定】修正方向后的实测撤回率（用户决定：继续跑、如实报告）

校准集实测（`receipts/support-scales-fixed-*.json`）：

**按相机拆开**（聚合值不能描述两个不同的总体）：

| 平台 / 相机 | accepted | **retracted** |
| --- | --- | --- |
| libero / agentview | 16.2% | **83.0%** |
| libero / wrist | 6.2% | **92.3%** |
| robotwin2 / head_camera | 58.3% | **40.9%** |
| robotwin2 / left_camera | 54.8% | **45.0%** |
| robotwin2 / right_camera | 57.0% | **42.7%** |

**比例随窗口变化**：step 80 的 agentview ≈96% retracted，step 380 ≈48% retracted。

**后果（已更正我先前过强的措辞）**：LIBERO 上历史**大部分但非全部**被撤回
（agentview 保留 16.2%、wrist 仅 6.2%）→ "支持改变内容"在 LIBERO 上**被大幅削弱但未完全关闭**，
wrist 接近关闭；RoboTwin 保留 ~55–58%，机制充分工作。
先前写的"退化为单帧臂"是**把聚合撤回率当成全臂失效**，按相机数据已推翻。

**这是未解决的歧义，不声称任一解释**：可能是统计量正确地反映了 LIBERO 窗口内频繁发生变更，
也可能是它过于敏感（区域级赔率广播给所有历史源 + "短跑"边界 8 对 32 帧窗口偏宽）。
分辨二者所需实验**未做**。

**用户决定**：保留 `retract=1.0`（零基含义）继续训练，如实报告，不调阈值让数字好看。

**对四臂比较的后果**：LIBERO 上 `raw_n01` vs `raw_base` 的差异**不得**解释为测量支持的收益/损失；
只有 RoboTwin 上该比较有机制层面意义；跨平台合并的"RAW 增益"说法不成立。

## 4n. 弱测试已修（本轮）

- `assert same in (True, False)`（恒真）→ 改为断言**删除与掩码在账本上必须可区分**；
- 名为"shift is retracted"实则只查"非 ACCEPTED"且停早一步的测试 → **改名 + 标注为烟测**，
  真性质由生产路径回归测试覆盖；
- `age_ms >= 0.0`（恒真）→ 改为断言实际经过时间与 `expired_by_age` 的两种取值；
- 冗余形状断言删除；
- 零假设探针测试 → 改为 `test_what_this_probe_can_and_cannot_establish`，
  只断言该探针**能**确立的两件事，并写明它**不能**判定估计量标定（与 §1.34 的撤回一致）。

**顺带修掉我自己的性能问题**：测试原先建全宽全深 DBSR，使全套从 64 s 涨到 275 s；
改窄改浅后受影响测试 110 s → **2.4 s**，全套 **43.6 s**。

## 4o. 有效开发闭环已完成（本轮）

`devloop-fixed/`：**100/100，成功 0，分母 100**（`not_completed=0`），4 suite 各 25，
5 个照明域皆有（Low 15 / ExtremeOver 15 / ExtremeLow 14 / Normal 13 / Over 13），
`representation=raw`，均值 **76.7 s**/episode，checkpoint 为修正后从零训到 **step 500** 的。

**这是 P3 的闭环主结果**（§1.23 那轮是缺陷 A 生效时测的，已降级为历史证据）。
但：仍只跑 `raw_n01` 一条臂（无对照）→ **不构成机制结论**；
checkpoint 仅 step 500/20000 → **不构成性能证据**，只证明协议有效与分母真实。

监控页已切到该结果（`dev_loop 100/100`，rgb_n01 显示未测量）。

**未定因的观察**：本轮 76.7 s/episode vs §1.23 的 29.4 s（≈2.6×）；未做区分实验，不写归因。

## 4p. 我验证了自己回归测试的检出能力（本轮），推翻两条自认为成立的断言

把支持判据的**比较符单独翻转**（阈值不动、配置仍合法，避免失败来自配置校验）后重跑：

| 测试 | 翻转后 |
| --- | --- |
| `test_support.py::test_a_long_stationary_history_stays_usable` | **失败** ✓ 真护栏 |
| `test_support.py::test_a_real_change_withdraws_the_history_then_recovers_as_it_settles` | **失败** ✓ |
| `test_support.py::test_support_masks_the_content_path_input` | **失败** ✓ |
| `test_model_v1_2.py` 两臂 token 比较 | **通过** ✗ 抓不到 |

**结论 1**：arm-vs-arm 的 token 比较对"历史是否被使用"是**钝器**——融合权重几乎全在参考槽位，
算进/丢掉历史对 token 的影响低于 `atol=1e-5`。判断支持是否起作用应看**支持决定本身**。
理论说明的 (b) 已**降级为弱旁证**。

**结论 2**：新增的恢复分支让**反向比较在静止序列上自我抵消**（赔率低→反向判"已变更"→撤回→立即恢复），
所以"静止场景正确"**不能**用来证明比较方向正确。

**并改正我自己的过度说法**：那个测试的 docstring 起初写"决定性量是支持决定本身，故该断言敏感"，
实测表明**它同样抓不到反向**，已改为如实陈述并指明真正的护栏是哪个测试。

## 4q. 时间轴/视频已按修正机制重录（本轮）

旧产物（缺陷 A 生效时录制）已从索引移除并删除文件（GET 404），
新产物 `devloop-timeline-fixed`（440 记录 ↔ 440 PNG）注册并可访问（GET 200）。
录制改用显式 `--record-timeline`，不再依赖外部导出的环境变量。

**重录暴露的对照**（44 次查询的可用单元比例）：

| episode | 照明 | usable-cell share |
| --- | --- | --- |
| libero_spatial t1 e4 | ExtremeOver (ev +7.72) | **0.00 – 0.00** |
| libero_spatial t0 e0 | ExtremeLow (ev −5.49) | **0.09 – 1.00** |

即**亮且真在动 → 历史全丢；近全黑 → 历史大量保留**。
**一致性假设（未证明）**：判据基于帧间差异，而 ExtremeLow 的 RAW 均值仅约 1/255，
相邻帧几乎相同 → 赔率低 → 判为"稳定"而保留。**未做证伪实验**。
若成立，这是"基于帧间差异"判据的**反向激励**（信息量低的输入反而保留历史），
属设计性质而非实现错误。

## 4r. "反向激励"假设已做证伪实验（本轮）

在**真实录制帧**上人为平移历史相对参考的像素数（`n01_v12_null.motion_response`）：

| 位移 | ExtremeOver（亮 0.70） | ExtremeLow（暗 0.004） |
| --- | --- | --- |
| 0 | accepted 1.00 | accepted 1.00 |
| 2 | **retracted 0.63** | **deferred 0.55** / retracted 0.10 |
| 8 | retracted 0.76 | deferred 0.53 / retracted 0.17 |
| 16 | retracted 0.78 | deferred 0.50 / retracted 0.23 |

**结论一**：判据确实响应帧间差异（接受率从 1.00 塌到 0.18–0.36）→ 原假设方向成立。
**结论二（未预料）**：**同样位移在亮帧得 RETRACTED、暗帧只得 DEFERRED**——
判决严厉程度**依赖照明**，不只依赖运动。使"反向激励"更具体：暗输入即使被判异常也得到更轻判决。
**不判定这是缺陷还是设计选择**（取决于设计意图是"帧间差异"还是"对应性"）。

**方法学更正**：探针字段原名 `raw_mean_over_255` 实为 [0,1] 均值（0.704 ≈ 180/255），已改名。

## 4s. 照明相关判决已追到机制层（本轮）

探针现在同时输出归一化残差与赔率：

| 位移 | ExtremeOver 残差/赔率 | ExtremeLow 残差/赔率 |
| --- | --- | --- |
| 0 | 0.00 / −3.146 | 0.00 / −3.146（**完全相同**） |
| 2 | 1.96 / **+12.61** | 1.08 / **−2.95** |
| 16 | 3.52 / +235.3 | 1.18 / −2.86 |

链条：照明**不直接进入判据**（位移 0 时两者完全一致）→ 通过**归一化残差对位移的响应**进入
（同样 2 px，亮帧残差 1.96、暗帧 1.08）→ 很小的 `obs_var=2e-2` 把差异放大成 +12.61 vs −2.95
→ 门槛 `retract_logodds=1.0` 恰好卡在中间 → 同一位移得到 RETRACTED 与 DEFERRED。

**若要改变该性质，抓手是 `obs_var` 或 `retract_logodds`，不是"撤回率"这类结果量**；
**本轮不动**（无依据说哪个位置对，且训练正在用当前配置跑）。
赔率 p90 在亮帧达 **50687**，与该统计量无界（§1.28）一致。

## 4t. 同源 ISP 说明已独立成文（本轮）

新增 `research/experiments/n01_v1_2/P3_isp_note.md`：把原先散在报告里的 ISP 内容收敛成可自足复核的一份，
含上游入口与 commit、固定增益/音调曲线、已验证项（逐元素相等、`identical_source`）、
已记录的信息损失（LIBERO uint8 的 ~2 bit）、**服务顺序偏差**（客户端先 resize，ISP 看到的是缩放后的帧）、
**未验证项**（逐元素交叉检查需要 `OWAC_RAWVLA_SOURCE`，未设置时只是历史结论）与复核命令。

**写作过程中发现自己引错了 receipt 的键路径**：初稿写 `pair[agentview]`，
实际是 `same_condition_pair.agentview`。已按 receipt 实际内容更正，报告同处一并更正
——交付文档里的引用必须能在产物里找到。

## 4u. "四臂工厂"审计：`rgb_base` 此前从未被构造（本轮补齐）

直接构造矩阵里每个配置并跑前向（不是看测试名字）后发现：
`raw_base`/`raw_n01`/`rgb_n01` 都真正跑过，**`rgb_base` 只在一处元数据断言里出现，从未被构造**——
"四臂工厂产出四条可用臂"当时只对三条成立。

本轮实测 **8/8 配置**（4 臂 × 2 平台）全部构造成功并跑通前向，且输入域跟随臂
（RAW 臂 `rawvla_rgb3`、RGB 臂 `rendered_rgb3`）、只有 N01 臂挂支持。
已固化为参数化测试（8 个用例），并**验证其检出能力**：
把输入域派生改回缺陷 B 之前的硬编码后，**失败 5 个用例**。

**边界**：这是接线证据，**不是**四臂比较结果；Base 臂从未训练（已推迟到 P4），
报告仍**没有**任何 RAW 增益或支持增益结论。

## 4v. "32 帧精确回放"审计（本轮）

推 37 次到达（cap=32）直接看窗口内容，而非看测试名。**已确认并固化为两个测试**：

| 性质 | 实测 |
| --- | --- |
| `filled` == `min(到达数, cap)` | 每次到达都成立 |
| 保留的是**最新的 32 帧**且 newest-first | 最终 ids `[36,35,…,7,6,5]` |
| 仍在窗口内的帧重送**不重复计数** | 重送 id=1（时间戳更晚）→ filled 不变 |
| 真正的新 id 仍被接受 | id=99 → filled +1 |

**又一次误读自己的探针**：初查读了 `frame_ids[0]`（只 slot 0），
以为"id 历史全丢、只剩最新一帧"，差点报一个不存在的 bug；正确切片是 `frame_ids[:, 0]`，
改对后实测完全正确。该坑（轴混用）在本项目已出现多次，已写进测试 docstring。

**边界**：这是单测量缓冲的**契约**证据，**不是**"回放能复现真实 rollout 动作"的证据。

## 4w. "对应"在 P3 运行中未启用（本轮审计）

查调用点（非测试名）后确认：`set_motion_provider` 只被 `runtime/n01_full_policy_preflight.py`
（预检工具）与测试调用；`training/n01_v12.py` 与 `runtime/n01_v12_service.py` **零命中**（grep `motion`）。
运行时实测：`p2_motion_status = "identity_pending_p2"`、
`p2_motion = {mapping_version: identity, decision_version: identity, risk_kind: unavailable}`。

即 **P3 训练与服务都用恒等偏移融合历史**，对应子系统已实现且有单测但**不活跃**。
**不是缺陷**（适配器显式命名 `identity_pending_p2`，设计上留待后续），
**是范围限定**："成熟内容与对应"在 P3 里**内容是活跃的、对应不是**；
任何关于跨帧对应/运动补偿的结论在 P3 **都没有依据**。

**补一处记录缺失**：诊断里本有 `p2_motion_status` 但训练日志没打印 → 已在 `train_step`
日志加入 `motion_status`（按相机）。**注意**：正在跑的两个 run 日志早于该改动，
不能据"日志无该字段"反推当时启用了对应。

## 4x. 试图核查"闭环是否落在训练集"：**该核查无效**（本轮）

两侧同源（都在 RawVLA-Bench data 下）。两种候选映射下闭环身份的 split 分布：

| 映射 | train | test | val | cal | 缺失 |
| --- | --- | --- | --- | --- | --- |
| `task*50 + episode` | **69/100（69%）** | 10 | 6 | 5 | 10 |
| `task + episode*10` | 52/100（52%） | 5 | 13 | 12 | 18 |

**随机身份落 train 的概率本身就是 69%**（1228/1771），而 task 优先映射给出的**恰好是 69%**
→ 结果**与随机无法区分**，该检查**没有分辨力**。

**我差点报成"69% 闭环身份属训练集"——那是把基率当发现**，按同一纪律不能作为结论。

**但对验收仍是真问题**：若闭环复用训练 demonstration，则 0/100 非留出估计。
判定它需要**经验证的身份映射**（按帧内容比对，或用 benchmark 的 provenance），**我没做**，
故**既不确认也不否认**。

## 4y. 闭环数据来源已查清（本轮），重叠问题仍未解决

从作者源码查清闭环读什么（不再猜映射）：`eval_libero.py:431-449` 的 `rawvla_bench` 分支
用 `raw_sources = [img, wrist_img]`（**env 实时渲染**）调用 `make_rawvla_observation(rgb8,...)`。
即：

| | 来源 | 读什么 |
| --- | --- | --- |
| P3 训练 | RawVLA-Bench 预建缓存 | `data/libero/raw/<suite>/episode_0000NN/variant_00.npz` |
| 闭环评测 | **LIBERO 模拟器实时渲染** | env 的 `img`/`wrist_img`，按 init state 重置 + 即时转 RAW |

**闭环不读 `variant_00.npz`**；共同上游是 LIBERO 数据集（README 写明 RawVLA-Bench
"derived from successful expert trajectories in LIBERO"）。

**重叠问题仍开放**：不是"读同一批文件"，而是"是否选中同一批 LIBERO episode"。
判定需**经验证的 (task,episode)→group 映射**；§1.44 的统计检查无分辨力（train 占 69% = 随机基率），
**内容比对未做** → **既不确认也不否认**。0/100 应描述为**开发诊断**，"留出"属性**未确立**。

**副产品（有价值）**：同段代码注释给出作者的权威顺序表述
`# Correct deployment order: simulator/render resolution -> RAW + frame noise -> downstream policy resize.`
→ 为服务顺序偏差提供**作者侧书面依据**（ISP 属 RAW 阶段、应在缩放前），已补进 `P3_isp_note.md`。

## 4z. 【重要】闭环 69/100 落在训练集：开发闭环不是留出评测（本轮已用数据确立）

映射不再靠推断：每个缓存 episode 的 `metadata_json` 带 `suite`/`task_id`/`demo_index`。
用其建索引得 **1771** 个 episode（40 个 suite-task 组合），查 100 个闭环身份得到的分布
与上一轮**推断**的映射**完全一致**；残余也吻合（全集 2000、缓存 1771 → 过滤 11.4%，
观测"不在清单"10.0%）。→ 映射在测量精度内成立。

| split | 闭环身份数 |
| --- | --- |
| **train** | **69** |
| test | 10 |
| validation | 6 |
| calibration | 5 |
| 不在清单 | 10 |

**69/100 落在训练集，且恰等于基率（69%）** → benchmark 从整套件每 task 的 init state 取 episode，
**不理会我的 train/test 划分**，重叠是构造必然。

**因此 0/100 是开发闭环诊断**（证明闭环接通、分母真实、协议可复现），
**不是泛化估计**。此前多次标为"开发诊断"但无证据，现在有实测依据。
**撤回上一轮的"既不确认也不否认"**——元数据映射有分辨力。

**P4 前置**：正式评测必须使用**尊重划分**的 episode 列表（只取 test/validation 侧）。

## 5a. 【当前引用的闭环结果】留出开发闭环 0 / 96

`devloop-heldout/`（`held_out_only: true`）：选中 100、**完成 96**、**成功 0**、**分母 96**、
未完成 4（`libero_10/8/17`、`/8/18`、`/9/10`、`/9/23`，如实列出未补跑）、均值 77.6 s/episode。
每 suite：libero_10 21、goal 25、object 25、spatial 25。**留出属性成立**。

**对照**：`devloop-fixed`（已降级）100/100 成功 0，但 69/100 属训练集。
两轮成功数相同，**只有留出这轮可引用**。

**监控页**：`dev_loop 96/100`、raw_n01 完成 96 / 成功 0 / 待完成 4、rgb_n01 未测量。
两轮旧展示记录已撤（`receipts/superseded-devloop-records.json` 记录原因），
四轮 `plan.json`/`report.json`/账本**全部保留**（撤的只是展示记录）。

**未变的限制**：仍只有 `raw_n01` 一条臂（无对照 → 非机制结论）；
checkpoint 约 step 900/20000（4.5%）→ 非性能证据，只证明协议有效/分母真实/留出成立。

## 5b. 监控页误报与暂停成本（本轮）

**监控页曾把正在训练的 run 报成"异常退出"**：`libero-rgb_n01-seed101` 显示
`alive=False, exit=心跳过期（103 秒未更新…）`，但实测 `supervisor RUNNING`、两条训练进程都在、
GPU0 57% / GPU1 86%、worker 文件 mtime 07:18:20（比 raw 更新）→ **一直在训练**。

**错配可算**：worker 每记录一个 step 才写一次，`--log-every 20`；LIBERO ≈5.5 s/step → 间隔 ≈110 s，
RoboTwin ≈27 s/step → ≈540 s；而阈值默认只有 **90 s** → **每个间隔都被判过期**。

**已修**：默认 90 → **1800 s**，算式写进注释；加回归护栏（默认 ≥900 s）；
既有测试的固定"5 分钟"夹具改为基于默认值推导（阈值一改就不再过期，正是本次暴露的脆弱点）。
修后两 run 均 `alive=True, suspicious=False`，`dev_loop 96/100`。

**顺带记录运行成本**：resume 实测 step `…840 → 520 → 540…`（日志有 `RESUMING …latest.pt`），
即正确续训，但 `--ckpt-every 500` 使**每次暂停最多重算 500 步 ≈ 46 分钟**。
不是缺陷，是 checkpoint 间隔与评测协议的组合副作用，已写入报告以便排期计入。

## 5c. 页面复核：产物全可达；任务矩阵曾低报交付（本轮已修）

- **产物可达性逐条实测**：`/api/artifacts` 的 **34/34 URL 返回 200**，无死链；
  `/`、`/api/status`、`/api/artifacts` 均 200。
- **任务矩阵过期**：仍把"闭环视频与失败样例图"标为 `missing`（实际 §1.31 交付、§1.40 重录），
  且**没有**留出闭环一行。与 §1.49 同类但方向相反——§1.49 是把活的 run 说成死的，
  这次是把**已交付的说成没做**；验收者按矩阵读会以为视频还没做。
- **已修为 13 行**：视频 → `done`；新增"留出开发闭环（100 次）" → `done`（96/100，成功 0，分母 96）；
  "支持尺度"保持 `partial`；**"Base 臂对照"保持 `missing`**（真实缺口，不掩饰）。

## 5d. 【高后果】队列里的 robotwin2-rgb_n01 无法启动（本轮已修）

八份配置逐份加载后发现：**4 条 Base 臂** 与 **robotwin2-rgb_n01** 的
`accept_logodds(0.5) >= retract_logodds(-1.0)`，被修 A 的新校验拒绝 → **无法加载**。

**根因是我自己的 `fix_thresholds.py`**：`CONFIGS` 手写三个名字，**漏掉第四个**；
我没断言覆盖数，只信了字面量。**后果**：该 run 排在 rgb 队列第二位，
LIBERO rgb_n01 结束后启动时**立刻死掉**——在**数天之后**。

**已修**：新脚本从目录取配置 + `assert len(paths)==8`；修正 5 份、3 份本已正确，
**8/8 全部加载成功**且输入域正确。**队列加前置校验**（加载不了即 SKIP 并报错，
已验证会 SKIP 坏配置/ACCEPT 好配置）。加测试把 `{platform}-{arm}-seed101` 命名钉在四臂矩阵上。
正在跑的两进程启动时已读完配置，**不受影响**。

**顺带**：四条 Base 配置此前也全部无法加载 → **P4 对照臂当时不可运行**；现已可加载，
"Base 臂对照" 的 `missing` 因此是纯粹的"本阶段不跑"，不再是"配置坏了"。

## 5e. 监控页把 raw/rgb 的 loss 曲线合并成一条（本轮已修）

查"还没执行但迟早会执行"的路径：`--val-every 1000` 在修正后训练里尚未触发（step≈760）。
**验证阶段本身正常**：`metrics.jsonl` 有 12 个 `val_action_loss` 点（step 1000–6000）——
它走 `progress.metric(...)` 不写 stdout，**"日志里没有"不等于"没执行"**。

**但同一步出现两个值**（step=1000 有 0.072577 / 0.071939）：两个 run 把同名指标写进同一文件，
而每行 `role` **全为 None**；图表只有 **1 条 series**（9 点）。
→ raw 与 rgb 的曲线被**交替合并成一条**，页面没有任何说明。
这正是本报告反复拒绝做的合并（把 raw 与 rgb 折成一条"综合线"），这次是页面自己干的。

**根因**：`RunStateWriter.metric()` 写 entry 时**漏写 `self.role`**（role 就在同一对象上）。

**已修**：① 指标行写入 role；② `read_metrics` 按 **(name, role)** 分组，旧行保留但标 `[未标注 arm]`（不猜）；
③ 图例带 `[role]` 后缀——**我第一版又错了**：后续 `if row.get("label")` 把后缀覆盖掉，
两条 series 名字仍相同；已删该覆盖。实测：`series: 2`，
`训练 action loss [libero-rgb_n01-seed101] points=6` / `[libero-raw_n01-seed101] points=5`。

旧指标已归档 `metrics-unattributed-archive.jsonl`（941 行，无归属且多为修 A 前机制所产生）。
**代价**：两 run 重启以让写入端生效，从 step 500 续训，**重算约 260 步 ≈ 24 分钟**。

## 5f. RoboTwin 训练通路已用修正后代码实测（本轮）

此前 RoboTwin 唯一证据是 **10-15 18:12 的 100 步 smoke**，早于修 A/B（10-16 01:53）；
上一轮只修好了**配置可加载**，"配置能加载" ≠ "训练循环能跑"。

本轮用当前代码 + 当前配置实跑 4 步（独立输出目录，不动主 run）：
`loss` 0.4720 / 1.2538 / 0.8356 / 0.3573，三相机 `support_status=applied_v1_2` ✓。

**顺带证实 §1.43 的改动生效**：日志出现新增的
`"motion_status": {"head_camera": "identity_pending_p2", ...}`，
即每个 run **自己声明**对应子系统未启用。

**时间预算确认**：76.6→152.6 s 跨 3 步 = **≈25.3 s/step** → 20000 步 ≈ **5.9 天**。

**仍未验证**：队列 LIBERO→RoboTwin 的**交接**（需等 LIBERO 跑完，约 4 天后）。
缓解：队列前置校验 + "单 item 失败不中断队列"。

## 5g. 交付与验收指引已成文（本轮）

新增 **`P3_HANDOFF.md`**：验收入口。含 ①逐项对照表（目标 13 项 → 状态 → 证据位置）；
②可引用数字**及其限制**（每个数字配"必须同时说的限制"）；③明确**没做**的 7 件事；
④本轮发现并修复的 **14 处缺陷**表（含三条"交付物在说假话"）；
⑤**主动撤回/降级**的 5 条结论；⑥复现与验收命令；⑦**我自己认为最脆弱的三处**。

**写完后做了引用核验**（文档的全部用途就是指向证据，指向错了就没用）：
- 引用的 24 个 `§1.x` **全部存在**（报告共 54 个编号小节）；
- 撤回率 83.0 / 92.3 / 40.9 与报告一致；
- 行数 1959 / 168、产物 1.70 MB 全部核对；
- **发现自己写错一处**：把 skip 写成"1 条需要 `OWAC_RAWVLA_SOURCE`"，
  实测是 **5 条**——**3 条**需要该环境变量（`frozen_isp` 的 74 与 108 行）、**2 条**需要 CUDA。
  已按实际分类更正，并写明这 2 条在本 worktree 内**跑不了**、不是"通过"。

## 5h. ISP 交叉检查实跑通过；我给的变量值一直是错的（本轮）

去把 §7 的 ISP 交叉检查真的跑起来，发现自己写错变量值：测试按
`<root>/starVLA/...`（LIBERO）与 `<root>/third_party/FastWAM/...`（RoboTwin）拼路径，
所以要给**两者的父目录**：

| 变量值 | 结果 |
| --- | --- |
| `.../source/starVLA`（原写法） | **3 skip**（"author module not found"） |
| `.../source`（**父目录**） | **4 passed，0 skip** |

**原写法看起来像"检查过了"**（命令成功退出），实际三条交叉检查**全都没执行**——
与 §1.49/1.51/1.53 同类（交付物状态与真实状态不一致），只是这次作者是我的验收文档。

→ `P3_isp_note.md` 的"逐元素相等"由"历史结论"**升级为当前可复现**；
`P3_HANDOFF.md` §7 改为实测可跑的命令，并写明"给错路径会 skip"。

## 4e. 100 次开发闭环的当前进度（本轮推进）

| 环节 | 状态 |
| --- | --- |
| 服务端 openpi 适配器 | **已完成并实测**（10 次到达、`unique_frames={agentview:10,wrist:10}`、p50 416.8 ms） |
| 可用 checkpoint | **已有**（`libero-raw_n01-seed101` step 3500 的 `latest.pt`，服务 metadata 已确认） |
| 评测脚手架 | **已 prepare**：`$DEPLOY/research/n01-v1/runs/v1-2-p3-devloop/`，13,250 条身份 |
| RAW 通路 | **已确认存在**：`run_libero` 已传 `--image-mode rawvla_bench`，只需把 representation 从 `default_isp` 换成 `raw`、端口指向本服务 |
| 开发闭环启动器 | **已完成并正在跑**：`runtime/n01_v12_libero_dev.py`（100 条分层身份、按身份交错分两卡、逐 episode 账本按身份合并去重） |
| 开发闭环主结果（无偏） | **100/100，成功 0，分母 100**（`devloop-balanced/`，5 个照明域各 20 条、4 suite × 10 task，全 0）；均值 29.4 s/episode |
| 开发闭环第一轮（偏暗，保留） | 100/100，成功 0（`devloop/`，旧规则；证据不覆盖） |
| 删源 GPU 实测（两个 regime 都有） | 删已撤回源（slot3）→ 全 0.0（正确的 no-op）；删参考帧（slot31）→ token 变、flow v 差 8.9e-03（**非零但小**） |
| RoboTwin 真实训练 | **已跑通 100/100 step**（loss 0.5554、**27.1 s/step**、checkpoint 已落盘；3 相机支持全部 `applied_v1_2`）→ 两平台训练不再缺一项 |
| 参数量口径 | 同平台内 Base/N01 完全相同（支持零参数）；LIBERO 3,112,354 vs RoboTwin 3,114,403 的差来自**相机数**，不是臂 |
| 预算后果 | RoboTwin 20k step ≈ **6.3 天/run**；LIBERO 先跑完 ~4 天后再加 ~6.3 天，4 个 run 合计 ≈ **10 天**挂钟 |
| RoboTwin 支持尺度 | **已重定并应用**：阈值改为校准分位数（q10/q75 → −3.2117 / 651.3786），accepted 4.6%→**51.0%**、retracted 56.6%→**9.9%**。config 内 `support.calibration` 记录规则与分位数 |
| LIBERO 阈值 | **刻意未改**（默认 q10 差 15%、撤回率 9.7% vs 10.0% 影响可忽略；两臂正在训练，改文件会造成中途换协议）→ provenance 记 `applied: false` |
| RoboTwin smoke checkpoint | 已移到 `outputs/robotwin2-raw_n01-smoke-precalibration/`，**不留给正式 run 续训**（它是校准前训练的，会带混合协议） |
| 新工具 | `owac.evaluation.n01_v12 support-scales`（CPU-only，不占训练 GPU） |
| 分位数规则的适用范围 | **只对零点任意的量成立**（两个 log-odds 阈值）。`large_residual_threshold` 是"残差 > 1σ"的物理陈述，取 q75 会无条件标记最高 25% → **已撤回**，改记为诊断 |
| `region_residual` 6.8 倍差异 | **已用零假设检验查清**（`evaluation/n01_v12_null.py`，CPU-only）：σ=0 两侧精确为 0、两侧都在 **≈1.36 饱和** → **估计量无平台标定错误**；差异来自成像/内容（RoboTwin 低噪声近静止 0.29 < 1.01；LIBERO 反量化+真实运动 1.97 > 1.37） |
| 对 `large_residual_threshold` 的结论 | **不重标定**，但**触发率必须与结果一起报告**；撤回分位数替换的理由更硬 |
| 仍未按数据选定 | `hazard`、`obs_var`、`robust_omega`、`robust_scale`、`large_residual_threshold`（明确记为未选定，未被悄悄替换） |
| 要求 E 神经链整源删除 | **已实现**：`evaluation/n01_v12.py::counterfactual_content_replay` + `intervene --mode {delete,mask}`，7 项测试；真实 checkpoint GPU 实测待下次训练暂停 |
| 选择规则 | **已修正**：任务优先 + 照明域轮转；真实清单上 5 个照明域各 20 条、mean_ev≈0.1–0.4（旧规则偏暗 40/40/20、mean_ev −3.22） |
| 按照明域拆解旧 100 条 | ExtremeLow 0/40、Low 0/40、**Normal 0/20** → **0% 不是照明造成的** |
| 闭环关键限定 | 100 条中 ExtremeLow 40 / Low 40 / Normal 20，`lighting_ev` 均值 −3.22 → 条件偏暗；checkpoint 仅 step 3500/20000 |
| 动作合理性 | 已由客户端 action stats 证实（env ∈ [-1.04,0.78]，夹爪 ±1.0）→ 0% 不是缩放/契约缺陷 |
| 训练恢复 | 编排脚本 `trap` 已自动恢复两个队列，当前 RUNNING 并从 `latest.pt` 续训 |

客户端约束（实测源码）：rawvla_bench 模式按每任务 250 条跑，`--num-trials` 被忽略，
**没有** per-episode 上限；要在 100 条停住需加一个有记录的 `--max-episodes`，或跑完如实报实际条数。

## 4f. 本轮新增：要求 E 神经链缺口已补

被删槽位**根本不进入 `step`** → 不在唯一帧缓冲、不在 BOCPD 到达序列、不参与支持决定、
不可能成为参考；窗口只变短，不从窗外补帧。`intervene --mode delete|mask` 分开命名，
receipt 分别声明 `recompute_scope`，不得混报。

三条实测结论（不是形状断言）：稳定场景下删除有权重的来源确实改变 token；
独立随机帧下支持撤回整段历史、所有权重为 0，删除**正确地**不改变内容；
删除与掩码的**可测**差异在账本（`arrival_index` 5 vs 6、`unique_frames_lifetime` 5 vs 6、
缓冲占用）而不是内容——硬要求内容不同等于断言本架构不承诺的性质。

仍未做：真实 checkpoint 上的 `--mode delete` GPU 实测（需与闭环同样的训练暂停，
32 帧 × 256² 在训练占用下放不下）。

## 4g. 监控页已显示阶段状态与闭环结果（本轮接线）

`/api/status` 实测：`stage=P3`、`state=training`、`dev_loop 100/100`（raw_n01 成功 0/失败 100；
rgb_n01 未运行显示 None）、`tasks=11`（区分 done/running/waiting_resource/partial/missing）、
`artifacts=30`、`runs` 四条且 `suspicious` 全 false。

- 闭环结果经 `RunStateWriter(...).rollouts(...)` 写入（不手改聚合结果，保持单写入者）；
- 阶段状态与任务矩阵经 `write_overrides(tasks=[...])`；
- RoboTwin smoke 的 worker 记录已从正式角色名改名为 `...-smoke-precalibration`，
  避免页面显示"正式 run 已在 step 100 结束"。

查看：`ssh -N -p 34302 root@154.59.156.21 -L 18080:127.0.0.1:18080` → `http://127.0.0.1:18080/`

## 4h. RAW→行动同步时间轴（已交付）

- 服务端 `Snapshot.diagnostics` 记录支持与融合（有界降采样）；
- 客户端注入第三个锚点，录制 RAW 全分辨率 PNG + 诊断 + 动作块；
- `evaluation/n01_v12_timeline.py` 施加**同一个冻结 ISP** 并出图；
- 产出 `monitor/artifacts/timeline_libero_spatial.png`（第 31 张 artifact，GET 200）；
- 该 episode 支持可用比例 0.016–0.128、256/4096 槽位带权重；
- 照明实测：ExtremeLow RAW 均值 1.0/255（近全黑）→ 与 0/100 一起读。

## 4i. 交付前数字复核（本轮）

直接读产物对照报告数字：100/100、成功 0、分母 100、5 域各 20、`representation=raw`、
时间轴 44 查询 / 256-4096 —— **全部一致**。

同时发现并撤下一处**贴错标签的产物**：第一版时间轴图（PNG 命名冲突时生成、帧混了不同 rollout）
在被取代后仍留在磁盘并注册在索引里。已从索引移除、删除旧图（GET 404）、删除陈旧 receipt；
正确的两张图与两段视频保留。当前 artifacts = **34**。

## 4j. 【阻断】独立复核发现两处严重缺陷（我已自行复现）

**缺陷 A：支持判据方向反了。** `support.py:289-300` 定义 `odds = log_short - log_long`
（正 = 最近发生变更），但 `support.py:541` 的 `changed = odds <= retract_logodds`（默认 −1.0）
在 odds **低**（= 稳定）时为真。实测：**静止场景第 9 次到达起全部 RETRACTED、usable 0**；
而第 17 次到达 +5 真变化时反而 **usable 23–45%**。`RETRACTED` 还是单向的，只能从 `DEFERRED` 恢复。
→ `raw_n01` 退化为**单帧编码器**。

**缺陷 B：RGB 臂从未拿到 ISP 输入。** `paired_views` 只被出图/报告路径调用三次，
训练与服务输入路径**零调用**；`THREE_CHANNEL_INPUT_MODES` 里**没有 `rendered_rgb3`**；
`dataset_from_protocol` 不接受 arm/domain；`adapters.py` 里 grep `config.arm` **零命中**。
→ `rgb_n01` 与 `raw_n01` 输入完全相同，**"同机制换输入"这一问从未执行**。
而 `model.py:23-25` 却写着 RGB 臂读 `rendered_rgb3` 张量——**该声明与实现不符**。

**测试为何没拦**：`tests/encoders/n01_full/test_support.py:197` 名字是对的，但静止半边只
`observe` **一次**（首观测 odds 被置 0 故不撤回），缺陷在第 9 次才发作；
位移半边断言"有非 ACCEPTED 状态"，**在反向实现下同样成立**，无法区分。

**对已有结果的影响**：
- **100 次闭环的 0/100 是在缺陷 A 生效时测的 → 不能用来评价支持机制**；
- 正在训练的两个 LIBERO run 带缺陷 A，rgb 臂还带缺陷 B → **不能作为 N01 臂引用**，
  rgb run **无法回答它存在的那个问题**；
- §1.26 把 RoboTwin 的 `retract_logodds` "校准"到 q10，**是在给方向反了的判据调阈值**；
  §1.30 的"估计量无标定错误"仍成立，但**不能**证明判据方向正确。

## 4k. 两处阻断缺陷已修 + 训练已重启（本轮）

用户决定：立刻停两个 run、先修 A+B 再重启。

**停止位置**：raw_n01 **6600/20000**、rgb_n01 **6420/20000**。旧 output 目录移到
`outputs/*-prefix-inverted-support[-and-raw-input]/`（保留证据），
`receipts/restart-after-fix-a-b.json` 记录移动、原因与重启代码的 sha256 前缀。

**缺陷 A（支持方向）**：`changed` 改为 `odds >= retract_logodds`；
`retract_logodds` 默认 **−1.0 → +1.0**（odds 在证据不足时恰为 0，门槛必须严格高于 0）；
校验反转为 **`accept < retract`**；**`RETRACTED` 改为可修订**。
实测：静止 32 帧从"第 9 次到达起 usable 0"变为**全程 usable 100%**；
真位移后 usable 0，新帧自洽后又可恢复。

**缺陷 B（RGB 输入）**：`input_mode` 由**臂声明的输入域**派生（RAW→`rawvla_rgb3`、RGB→`rendered_rgb3`），
变换落在 `DbsrFeatureAdapter.step` 里 `_window()` 之后；`from_dict` 丢弃存储值以免旧配置把 RGB 臂按回 RAW。
实测四臂解析正确，新测试断言两臂输入与 token 均不同。

**一并修掉**：episode reset 漏 `FullState.support`（跨集泄漏）、`shift` 与缓冲不同步、
支持图错切、视频硬编码失败串、伪造零动作、`usable` 口径、融合轴误标、三份配置的旧阈值序。
**更正我自己 §1.30**：零假设探针把噪声注入参考帧，分子分母同缩放，故"饱和⇒估计量无标定错误"**不成立，已撤回**。

**重启状态**：两个 run 均 `STARTING ... (no checkpoint yet)` → **从 step 0 开始**，
GPU 利用率 70–72%。**尚未提交**。

**未修的复核项**已逐条列在 P3_report §1.34 末（`_degrade_to_single_frame` 无调用者、
deadline 从不与耗时比较、`decide(age)` 未使用、`rollouts()` 无生产调用者、
`source_usable` 载荷未限界、出图未按 camera 校验、ISP 与 resize 顺序、弱测试等）。

## 4l. 复核的"惰性代码"已逐条处置（本轮）

**最重要的一条：期限读出（A–F 之一）此前只有声明没有实现。**
`deadline_ms` 从不与耗时比较、`max_snapshot_age_ms` 无人使用、`_degrade_to_single_frame` 无调用者
且其 `[:, -1:]` 切错轴。现已：计算后实测耗时并与 `deadline_ms` 比较；超时按
**未过期旧快照 → 单帧降级 → 明说没赶上** 的顺序处置；年龄界真正生效；回包新增
`deadline_met` / `deadline_elapsed_ms`；`guarantee` 改为如实描述。新增 5 项测试。

其余：`source_usable` 载荷改为精确计数 + 4096 点样本；出图按相机校验并修掉 `fusion` 未清空；
录制改为显式 `--record-timeline`；丢记录不再静默（计数 + `.errors` 边车）；去掉无人读的缩略图；
更正 `P3_SUPPORT_ENABLED` 文档；`decide(age)` 明确标注"透传但不被读取"（不发明用途）；
启动器跑完闭环后自动 `_publish_to_monitor`（失败打印事件）。

**仍未处理**：§3 的弱测试质量问题（恒真断言、只查形状等）大多仍在。

**重启中的训练**：raw_n01 / rgb_n01 均 step 80、RUNNING，从 0 开始（正确代码）。


## 4n. 弱测试已修（本轮）

- `assert same in (True, False)`（恒真）→ 改为断言**删除与掩码在账本上必须可区分**；
- 名为"shift is retracted"实则只查"非 ACCEPTED"且停早一步的测试 → **改名 + 标注为烟测**，
  真性质由生产路径回归测试覆盖；
- `age_ms >= 0.0`（恒真）→ 改为断言实际经过时间与 `expired_by_age` 的两种取值；
- 冗余形状断言删除；
- 零假设探针测试 → 改为 `test_what_this_probe_can_and_cannot_establish`，
  只断言该探针**能**确立的两件事，并写明它**不能**判定估计量标定（与 §1.34 的撤回一致）。

**顺带修掉我自己的性能问题**：测试原先建全宽全深 DBSR，使全套从 64 s 涨到 275 s；
改窄改浅后受影响测试 110 s → **2.4 s**，全套 **43.6 s**。

## 4o. 有效开发闭环已完成（本轮）

`devloop-fixed/`：**100/100，成功 0，分母 100**（`not_completed=0`），4 suite 各 25，
5 个照明域皆有（Low 15 / ExtremeOver 15 / ExtremeLow 14 / Normal 13 / Over 13），
`representation=raw`，均值 **76.7 s**/episode，checkpoint 为修正后从零训到 **step 500** 的。

**这是 P3 的闭环主结果**（§1.23 那轮是缺陷 A 生效时测的，已降级为历史证据）。
但：仍只跑 `raw_n01` 一条臂（无对照）→ **不构成机制结论**；
checkpoint 仅 step 500/20000 → **不构成性能证据**，只证明协议有效与分母真实。

监控页已切到该结果（`dev_loop 100/100`，rgb_n01 显示未测量）。

**未定因的观察**：本轮 76.7 s/episode vs §1.23 的 29.4 s（≈2.6×）；未做区分实验，不写归因。

## 4p. 我验证了自己回归测试的检出能力（本轮），推翻两条自认为成立的断言

把支持判据的**比较符单独翻转**（阈值不动、配置仍合法，避免失败来自配置校验）后重跑：

| 测试 | 翻转后 |
| --- | --- |
| `test_support.py::test_a_long_stationary_history_stays_usable` | **失败** ✓ 真护栏 |
| `test_support.py::test_a_real_change_withdraws_the_history_then_recovers_as_it_settles` | **失败** ✓ |
| `test_support.py::test_support_masks_the_content_path_input` | **失败** ✓ |
| `test_model_v1_2.py` 两臂 token 比较 | **通过** ✗ 抓不到 |

**结论 1**：arm-vs-arm 的 token 比较对"历史是否被使用"是**钝器**——融合权重几乎全在参考槽位，
算进/丢掉历史对 token 的影响低于 `atol=1e-5`。判断支持是否起作用应看**支持决定本身**。
理论说明的 (b) 已**降级为弱旁证**。

**结论 2**：新增的恢复分支让**反向比较在静止序列上自我抵消**（赔率低→反向判"已变更"→撤回→立即恢复），
所以"静止场景正确"**不能**用来证明比较方向正确。

**并改正我自己的过度说法**：那个测试的 docstring 起初写"决定性量是支持决定本身，故该断言敏感"，
实测表明**它同样抓不到反向**，已改为如实陈述并指明真正的护栏是哪个测试。

## 4q. 时间轴/视频已按修正机制重录（本轮）

旧产物（缺陷 A 生效时录制）已从索引移除并删除文件（GET 404），
新产物 `devloop-timeline-fixed`（440 记录 ↔ 440 PNG）注册并可访问（GET 200）。
录制改用显式 `--record-timeline`，不再依赖外部导出的环境变量。

**重录暴露的对照**（44 次查询的可用单元比例）：

| episode | 照明 | usable-cell share |
| --- | --- | --- |
| libero_spatial t1 e4 | ExtremeOver (ev +7.72) | **0.00 – 0.00** |
| libero_spatial t0 e0 | ExtremeLow (ev −5.49) | **0.09 – 1.00** |

即**亮且真在动 → 历史全丢；近全黑 → 历史大量保留**。
**一致性假设（未证明）**：判据基于帧间差异，而 ExtremeLow 的 RAW 均值仅约 1/255，
相邻帧几乎相同 → 赔率低 → 判为"稳定"而保留。**未做证伪实验**。
若成立，这是"基于帧间差异"判据的**反向激励**（信息量低的输入反而保留历史），
属设计性质而非实现错误。

## 4r. "反向激励"假设已做证伪实验（本轮）

在**真实录制帧**上人为平移历史相对参考的像素数（`n01_v12_null.motion_response`）：

| 位移 | ExtremeOver（亮 0.70） | ExtremeLow（暗 0.004） |
| --- | --- | --- |
| 0 | accepted 1.00 | accepted 1.00 |
| 2 | **retracted 0.63** | **deferred 0.55** / retracted 0.10 |
| 8 | retracted 0.76 | deferred 0.53 / retracted 0.17 |
| 16 | retracted 0.78 | deferred 0.50 / retracted 0.23 |

**结论一**：判据确实响应帧间差异（接受率从 1.00 塌到 0.18–0.36）→ 原假设方向成立。
**结论二（未预料）**：**同样位移在亮帧得 RETRACTED、暗帧只得 DEFERRED**——
判决严厉程度**依赖照明**，不只依赖运动。使"反向激励"更具体：暗输入即使被判异常也得到更轻判决。
**不判定这是缺陷还是设计选择**（取决于设计意图是"帧间差异"还是"对应性"）。

**方法学更正**：探针字段原名 `raw_mean_over_255` 实为 [0,1] 均值（0.704 ≈ 180/255），已改名。

## 4s. 照明相关判决已追到机制层（本轮）

探针现在同时输出归一化残差与赔率：

| 位移 | ExtremeOver 残差/赔率 | ExtremeLow 残差/赔率 |
| --- | --- | --- |
| 0 | 0.00 / −3.146 | 0.00 / −3.146（**完全相同**） |
| 2 | 1.96 / **+12.61** | 1.08 / **−2.95** |
| 16 | 3.52 / +235.3 | 1.18 / −2.86 |

链条：照明**不直接进入判据**（位移 0 时两者完全一致）→ 通过**归一化残差对位移的响应**进入
（同样 2 px，亮帧残差 1.96、暗帧 1.08）→ 很小的 `obs_var=2e-2` 把差异放大成 +12.61 vs −2.95
→ 门槛 `retract_logodds=1.0` 恰好卡在中间 → 同一位移得到 RETRACTED 与 DEFERRED。

**若要改变该性质，抓手是 `obs_var` 或 `retract_logodds`，不是"撤回率"这类结果量**；
**本轮不动**（无依据说哪个位置对，且训练正在用当前配置跑）。
赔率 p90 在亮帧达 **50687**，与该统计量无界（§1.28）一致。

## 4t. 同源 ISP 说明已独立成文（本轮）

新增 `research/experiments/n01_v1_2/P3_isp_note.md`：把原先散在报告里的 ISP 内容收敛成可自足复核的一份，
含上游入口与 commit、固定增益/音调曲线、已验证项（逐元素相等、`identical_source`）、
已记录的信息损失（LIBERO uint8 的 ~2 bit）、**服务顺序偏差**（客户端先 resize，ISP 看到的是缩放后的帧）、
**未验证项**（逐元素交叉检查需要 `OWAC_RAWVLA_SOURCE`，未设置时只是历史结论）与复核命令。

**写作过程中发现自己引错了 receipt 的键路径**：初稿写 `pair[agentview]`，
实际是 `same_condition_pair.agentview`。已按 receipt 实际内容更正，报告同处一并更正
——交付文档里的引用必须能在产物里找到。

## 4u. "四臂工厂"审计：`rgb_base` 此前从未被构造（本轮补齐）

直接构造矩阵里每个配置并跑前向（不是看测试名字）后发现：
`raw_base`/`raw_n01`/`rgb_n01` 都真正跑过，**`rgb_base` 只在一处元数据断言里出现，从未被构造**——
"四臂工厂产出四条可用臂"当时只对三条成立。

本轮实测 **8/8 配置**（4 臂 × 2 平台）全部构造成功并跑通前向，且输入域跟随臂
（RAW 臂 `rawvla_rgb3`、RGB 臂 `rendered_rgb3`）、只有 N01 臂挂支持。
已固化为参数化测试（8 个用例），并**验证其检出能力**：
把输入域派生改回缺陷 B 之前的硬编码后，**失败 5 个用例**。

**边界**：这是接线证据，**不是**四臂比较结果；Base 臂从未训练（已推迟到 P4），
报告仍**没有**任何 RAW 增益或支持增益结论。

## 4v. "32 帧精确回放"审计（本轮）

推 37 次到达（cap=32）直接看窗口内容，而非看测试名。**已确认并固化为两个测试**：

| 性质 | 实测 |
| --- | --- |
| `filled` == `min(到达数, cap)` | 每次到达都成立 |
| 保留的是**最新的 32 帧**且 newest-first | 最终 ids `[36,35,…,7,6,5]` |
| 仍在窗口内的帧重送**不重复计数** | 重送 id=1（时间戳更晚）→ filled 不变 |
| 真正的新 id 仍被接受 | id=99 → filled +1 |

**又一次误读自己的探针**：初查读了 `frame_ids[0]`（只 slot 0），
以为"id 历史全丢、只剩最新一帧"，差点报一个不存在的 bug；正确切片是 `frame_ids[:, 0]`，
改对后实测完全正确。该坑（轴混用）在本项目已出现多次，已写进测试 docstring。

**边界**：这是单测量缓冲的**契约**证据，**不是**"回放能复现真实 rollout 动作"的证据。

## 4w. "对应"在 P3 运行中未启用（本轮审计）

查调用点（非测试名）后确认：`set_motion_provider` 只被 `runtime/n01_full_policy_preflight.py`
（预检工具）与测试调用；`training/n01_v12.py` 与 `runtime/n01_v12_service.py` **零命中**（grep `motion`）。
运行时实测：`p2_motion_status = "identity_pending_p2"`、
`p2_motion = {mapping_version: identity, decision_version: identity, risk_kind: unavailable}`。

即 **P3 训练与服务都用恒等偏移融合历史**，对应子系统已实现且有单测但**不活跃**。
**不是缺陷**（适配器显式命名 `identity_pending_p2`，设计上留待后续），
**是范围限定**："成熟内容与对应"在 P3 里**内容是活跃的、对应不是**；
任何关于跨帧对应/运动补偿的结论在 P3 **都没有依据**。

**补一处记录缺失**：诊断里本有 `p2_motion_status` 但训练日志没打印 → 已在 `train_step`
日志加入 `motion_status`（按相机）。**注意**：正在跑的两个 run 日志早于该改动，
不能据"日志无该字段"反推当时启用了对应。

## 4x. 试图核查"闭环是否落在训练集"：**该核查无效**（本轮）

两侧同源（都在 RawVLA-Bench data 下）。两种候选映射下闭环身份的 split 分布：

| 映射 | train | test | val | cal | 缺失 |
| --- | --- | --- | --- | --- | --- |
| `task*50 + episode` | **69/100（69%）** | 10 | 6 | 5 | 10 |
| `task + episode*10` | 52/100（52%） | 5 | 13 | 12 | 18 |

**随机身份落 train 的概率本身就是 69%**（1228/1771），而 task 优先映射给出的**恰好是 69%**
→ 结果**与随机无法区分**，该检查**没有分辨力**。

**我差点报成"69% 闭环身份属训练集"——那是把基率当发现**，按同一纪律不能作为结论。

**但对验收仍是真问题**：若闭环复用训练 demonstration，则 0/100 非留出估计。
判定它需要**经验证的身份映射**（按帧内容比对，或用 benchmark 的 provenance），**我没做**，
故**既不确认也不否认**。

## 4y. 闭环数据来源已查清（本轮），重叠问题仍未解决

从作者源码查清闭环读什么（不再猜映射）：`eval_libero.py:431-449` 的 `rawvla_bench` 分支
用 `raw_sources = [img, wrist_img]`（**env 实时渲染**）调用 `make_rawvla_observation(rgb8,...)`。
即：

| | 来源 | 读什么 |
| --- | --- | --- |
| P3 训练 | RawVLA-Bench 预建缓存 | `data/libero/raw/<suite>/episode_0000NN/variant_00.npz` |
| 闭环评测 | **LIBERO 模拟器实时渲染** | env 的 `img`/`wrist_img`，按 init state 重置 + 即时转 RAW |

**闭环不读 `variant_00.npz`**；共同上游是 LIBERO 数据集（README 写明 RawVLA-Bench
"derived from successful expert trajectories in LIBERO"）。

**重叠问题仍开放**：不是"读同一批文件"，而是"是否选中同一批 LIBERO episode"。
判定需**经验证的 (task,episode)→group 映射**；§1.44 的统计检查无分辨力（train 占 69% = 随机基率），
**内容比对未做** → **既不确认也不否认**。0/100 应描述为**开发诊断**，"留出"属性**未确立**。

**副产品（有价值）**：同段代码注释给出作者的权威顺序表述
`# Correct deployment order: simulator/render resolution -> RAW + frame noise -> downstream policy resize.`
→ 为服务顺序偏差提供**作者侧书面依据**（ISP 属 RAW 阶段、应在缩放前），已补进 `P3_isp_note.md`。

## 4z. 【重要】闭环 69/100 落在训练集：开发闭环不是留出评测（本轮已用数据确立）

映射不再靠推断：每个缓存 episode 的 `metadata_json` 带 `suite`/`task_id`/`demo_index`。
用其建索引得 **1771** 个 episode（40 个 suite-task 组合），查 100 个闭环身份得到的分布
与上一轮**推断**的映射**完全一致**；残余也吻合（全集 2000、缓存 1771 → 过滤 11.4%，
观测"不在清单"10.0%）。→ 映射在测量精度内成立。

| split | 闭环身份数 |
| --- | --- |
| **train** | **69** |
| test | 10 |
| validation | 6 |
| calibration | 5 |
| 不在清单 | 10 |

**69/100 落在训练集，且恰等于基率（69%）** → benchmark 从整套件每 task 的 init state 取 episode，
**不理会我的 train/test 划分**，重叠是构造必然。

**因此 0/100 是开发闭环诊断**（证明闭环接通、分母真实、协议可复现），
**不是泛化估计**。此前多次标为"开发诊断"但无证据，现在有实测依据。
**撤回上一轮的"既不确认也不否认"**——元数据映射有分辨力。

**P4 前置**：正式评测必须使用**尊重划分**的 episode 列表（只取 test/validation 侧）。

## 5a. 【当前引用的闭环结果】留出开发闭环 0 / 96

`devloop-heldout/`（`held_out_only: true`）：选中 100、**完成 96**、**成功 0**、**分母 96**、
未完成 4（`libero_10/8/17`、`/8/18`、`/9/10`、`/9/23`，如实列出未补跑）、均值 77.6 s/episode。
每 suite：libero_10 21、goal 25、object 25、spatial 25。**留出属性成立**。

**对照**：`devloop-fixed`（已降级）100/100 成功 0，但 69/100 属训练集。
两轮成功数相同，**只有留出这轮可引用**。

**监控页**：`dev_loop 96/100`、raw_n01 完成 96 / 成功 0 / 待完成 4、rgb_n01 未测量。
两轮旧展示记录已撤（`receipts/superseded-devloop-records.json` 记录原因），
四轮 `plan.json`/`report.json`/账本**全部保留**（撤的只是展示记录）。

**未变的限制**：仍只有 `raw_n01` 一条臂（无对照 → 非机制结论）；
checkpoint 约 step 900/20000（4.5%）→ 非性能证据，只证明协议有效/分母真实/留出成立。

## 5b. 监控页误报与暂停成本（本轮）

**监控页曾把正在训练的 run 报成"异常退出"**：`libero-rgb_n01-seed101` 显示
`alive=False, exit=心跳过期（103 秒未更新…）`，但实测 `supervisor RUNNING`、两条训练进程都在、
GPU0 57% / GPU1 86%、worker 文件 mtime 07:18:20（比 raw 更新）→ **一直在训练**。

**错配可算**：worker 每记录一个 step 才写一次，`--log-every 20`；LIBERO ≈5.5 s/step → 间隔 ≈110 s，
RoboTwin ≈27 s/step → ≈540 s；而阈值默认只有 **90 s** → **每个间隔都被判过期**。

**已修**：默认 90 → **1800 s**，算式写进注释；加回归护栏（默认 ≥900 s）；
既有测试的固定"5 分钟"夹具改为基于默认值推导（阈值一改就不再过期，正是本次暴露的脆弱点）。
修后两 run 均 `alive=True, suspicious=False`，`dev_loop 96/100`。

**顺带记录运行成本**：resume 实测 step `…840 → 520 → 540…`（日志有 `RESUMING …latest.pt`），
即正确续训，但 `--ckpt-every 500` 使**每次暂停最多重算 500 步 ≈ 46 分钟**。
不是缺陷，是 checkpoint 间隔与评测协议的组合副作用，已写入报告以便排期计入。

## 5c. 页面复核：产物全可达；任务矩阵曾低报交付（本轮已修）

- **产物可达性逐条实测**：`/api/artifacts` 的 **34/34 URL 返回 200**，无死链；
  `/`、`/api/status`、`/api/artifacts` 均 200。
- **任务矩阵过期**：仍把"闭环视频与失败样例图"标为 `missing`（实际 §1.31 交付、§1.40 重录），
  且**没有**留出闭环一行。与 §1.49 同类但方向相反——§1.49 是把活的 run 说成死的，
  这次是把**已交付的说成没做**；验收者按矩阵读会以为视频还没做。
- **已修为 13 行**：视频 → `done`；新增"留出开发闭环（100 次）" → `done`（96/100，成功 0，分母 96）；
  "支持尺度"保持 `partial`；**"Base 臂对照"保持 `missing`**（真实缺口，不掩饰）。

## 5d. 【高后果】队列里的 robotwin2-rgb_n01 无法启动（本轮已修）

八份配置逐份加载后发现：**4 条 Base 臂** 与 **robotwin2-rgb_n01** 的
`accept_logodds(0.5) >= retract_logodds(-1.0)`，被修 A 的新校验拒绝 → **无法加载**。

**根因是我自己的 `fix_thresholds.py`**：`CONFIGS` 手写三个名字，**漏掉第四个**；
我没断言覆盖数，只信了字面量。**后果**：该 run 排在 rgb 队列第二位，
LIBERO rgb_n01 结束后启动时**立刻死掉**——在**数天之后**。

**已修**：新脚本从目录取配置 + `assert len(paths)==8`；修正 5 份、3 份本已正确，
**8/8 全部加载成功**且输入域正确。**队列加前置校验**（加载不了即 SKIP 并报错，
已验证会 SKIP 坏配置/ACCEPT 好配置）。加测试把 `{platform}-{arm}-seed101` 命名钉在四臂矩阵上。
正在跑的两进程启动时已读完配置，**不受影响**。

**顺带**：四条 Base 配置此前也全部无法加载 → **P4 对照臂当时不可运行**；现已可加载，
"Base 臂对照" 的 `missing` 因此是纯粹的"本阶段不跑"，不再是"配置坏了"。

## 5e. 监控页把 raw/rgb 的 loss 曲线合并成一条（本轮已修）

查"还没执行但迟早会执行"的路径：`--val-every 1000` 在修正后训练里尚未触发（step≈760）。
**验证阶段本身正常**：`metrics.jsonl` 有 12 个 `val_action_loss` 点（step 1000–6000）——
它走 `progress.metric(...)` 不写 stdout，**"日志里没有"不等于"没执行"**。

**但同一步出现两个值**（step=1000 有 0.072577 / 0.071939）：两个 run 把同名指标写进同一文件，
而每行 `role` **全为 None**；图表只有 **1 条 series**（9 点）。
→ raw 与 rgb 的曲线被**交替合并成一条**，页面没有任何说明。
这正是本报告反复拒绝做的合并（把 raw 与 rgb 折成一条"综合线"），这次是页面自己干的。

**根因**：`RunStateWriter.metric()` 写 entry 时**漏写 `self.role`**（role 就在同一对象上）。

**已修**：① 指标行写入 role；② `read_metrics` 按 **(name, role)** 分组，旧行保留但标 `[未标注 arm]`（不猜）；
③ 图例带 `[role]` 后缀——**我第一版又错了**：后续 `if row.get("label")` 把后缀覆盖掉，
两条 series 名字仍相同；已删该覆盖。实测：`series: 2`，
`训练 action loss [libero-rgb_n01-seed101] points=6` / `[libero-raw_n01-seed101] points=5`。

旧指标已归档 `metrics-unattributed-archive.jsonl`（941 行，无归属且多为修 A 前机制所产生）。
**代价**：两 run 重启以让写入端生效，从 step 500 续训，**重算约 260 步 ≈ 24 分钟**。

## 5f. RoboTwin 训练通路已用修正后代码实测（本轮）

此前 RoboTwin 唯一证据是 **10-15 18:12 的 100 步 smoke**，早于修 A/B（10-16 01:53）；
上一轮只修好了**配置可加载**，"配置能加载" ≠ "训练循环能跑"。

本轮用当前代码 + 当前配置实跑 4 步（独立输出目录，不动主 run）：
`loss` 0.4720 / 1.2538 / 0.8356 / 0.3573，三相机 `support_status=applied_v1_2` ✓。

**顺带证实 §1.43 的改动生效**：日志出现新增的
`"motion_status": {"head_camera": "identity_pending_p2", ...}`，
即每个 run **自己声明**对应子系统未启用。

**时间预算确认**：76.6→152.6 s 跨 3 步 = **≈25.3 s/step** → 20000 步 ≈ **5.9 天**。

**仍未验证**：队列 LIBERO→RoboTwin 的**交接**（需等 LIBERO 跑完，约 4 天后）。
缓解：队列前置校验 + "单 item 失败不中断队列"。

## 5g. 交付与验收指引已成文（本轮）

新增 **`P3_HANDOFF.md`**：验收入口。含 ①逐项对照表（目标 13 项 → 状态 → 证据位置）；
②可引用数字**及其限制**（每个数字配"必须同时说的限制"）；③明确**没做**的 7 件事；
④本轮发现并修复的 **14 处缺陷**表（含三条"交付物在说假话"）；
⑤**主动撤回/降级**的 5 条结论；⑥复现与验收命令；⑦**我自己认为最脆弱的三处**。

**写完后做了引用核验**（文档的全部用途就是指向证据，指向错了就没用）：
- 引用的 24 个 `§1.x` **全部存在**（报告共 54 个编号小节）；
- 撤回率 83.0 / 92.3 / 40.9 与报告一致；
- 行数 1959 / 168、产物 1.70 MB 全部核对；
- **发现自己写错一处**：把 skip 写成"1 条需要 `OWAC_RAWVLA_SOURCE`"，
  实测是 **5 条**——**3 条**需要该环境变量（`frozen_isp` 的 74 与 108 行）、**2 条**需要 CUDA。
  已按实际分类更正，并写明这 2 条在本 worktree 内**跑不了**、不是"通过"。

## 5h. ISP 交叉检查实跑通过；我给的变量值一直是错的（本轮）

去把 §7 的 ISP 交叉检查真的跑起来，发现自己写错变量值：测试按
`<root>/starVLA/...`（LIBERO）与 `<root>/third_party/FastWAM/...`（RoboTwin）拼路径，
所以要给**两者的父目录**：

| 变量值 | 结果 |
| --- | --- |
| `.../source/starVLA`（原写法） | **3 skip**（"author module not found"） |
| `.../source`（**父目录**） | **4 passed，0 skip** |

**原写法看起来像"检查过了"**（命令成功退出），实际三条交叉检查**全都没执行**——
与 §1.49/1.51/1.53 同类（交付物状态与真实状态不一致），只是这次作者是我的验收文档。

→ `P3_isp_note.md` 的"逐元素相等"由"历史结论"**升级为当前可复现**；
`P3_HANDOFF.md` §7 改为实测可跑的命令，并写明"给错路径会 skip"。

## 5. 下一步

1. 读 `receipts/verify-*.json` 确认真实 ISP 配对、相机序、梯度与后端冻结；
2. 用 1 个 arm 的短测（`--max-steps` 小值）量真实 step/s 与显存，据此写训练预算；
3. 用 supervisor 起 GPU0 RAW / GPU1 RGB 两个队列，命名 `owac-n01-v12-p3-<raw|rgb>-<run>`；
4. 上监控（远端 18080，回环），经 SSH 转发给用户；
5. 开发闭环：先落一份 100 条总清单再分到两卡，按各自分母报成功数；
6. 写 `research/experiments/n01_v1_2/P3_report.md`，停在 P3。

## 7. 终态记录（第 40 轮，本会话最后一轮）

**P3 的全部具名交付物已完成，处于"停在 P3 等待独立验收"状态。**
验收入口：`research/experiments/n01_v1_2/P3_HANDOFF.md`。

**未完成的只有训练本身**，且是墙钟约束：LIBERO raw/rgb 约 **780 / 800 步**（预算 20000，约 3.9%），
≈5.5 s/step → 还需约 4 天；RoboTwin 排队，≈25.3 s/step → 约 5.9 天；
Base 臂**未训练**（用户决定推迟到 P4，配置已修至可运行）。

**因此不声称 P3 完成。** 本阶段可引用的结论只有一条：
**留出开发闭环 0 / 96（完成 96 / 选中 100，未完成 4）**，
其价值在于"闭环接通、分母真实、留出成立"，**不是**机制或性能结论（仍只有 `raw_n01` 一条臂）。

**本会话共发现并修复 15 处缺陷**，其中 4 处属于"交付物表现出的状态与真实状态不一致"：
§1.49（把活的 run 报成死的）、§1.51（把已交付的报成没做）、
§1.53（把两个模型画成一条曲线）、§1.55（我自己的验收命令把"未执行"表现得像"已检查"）。
全部有实测记录。

**建议的后续动作**（不属本阶段授权范围）：
1. 等 LIBERO 跑完，核对队列是否正确交接给 RoboTwin（§1.54 已列为未验证项）；
2. P4 启动前先做 §1.46 提到的**内容比对实验**，确认评测 episode 与训练集的真实关系；
3. P4 正式评测**必须使用尊重划分的 episode 列表**（`--held-out-only` 已可用，见 §1.47）。

## 6. 未完成即未完成（不得在报告里写成已完成）

> **本节写于早期，内容已过期，保留原文以示记录，并附当前实况。**
> 过期的部分已按实测更正——否则本节会**把已交付的说成没做**，
> 与 §5c 记录的任务矩阵问题是同一类错误。

**原文（早期）**：主体训练、100 次开发闭环、同步可视化案例、
`research/theory/n01_raw_evidence_contract_v1_2.md` 与 `P3_report.md` 在本次记录时仍未产出。

**当前实况（第 40 轮）**：

| 项 | 早期状态 | 当前 |
| --- | --- | --- |
| 主体训练 | 未产出 | LIBERO raw_n01 / rgb_n01 **运行中**，约 780 / 800 步（预算 20000）；RoboTwin 排队 |
| 100 次开发闭环 | 未产出 | **已完成**：留出协议 96/100 完成、成功 0、真实分母 96 |
| 同步可视化案例 | 未产出 | **已完成**：时间轴图 + 两段 mp4（按修正后机制重录） |
| 理论说明 | 未产出 | **已完成**：`research/theory/n01_raw_evidence_contract_v1_2.md` |
| `P3_report.md` | 未产出 | **已完成**：1959 行 / 55 个编号小节 |

**唯一仍未完成的仍是主体训练本身**（墙钟约束），故**不得**写成 P3 完成。

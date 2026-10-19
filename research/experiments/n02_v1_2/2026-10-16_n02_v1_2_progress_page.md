# N02 v1.2 实时进度页（18080 监控入口升级）证据记录

日期：2026-10-16（Asia/Shanghai）｜执行：n02-frontend（teammate）
范围：只做监控端与写入端 helper。**未修改**模型、损失、成功判据、训练预算、评测逻辑。

## 1. 变更文件（本地 worktree `n02-measurement-completion`）

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `src/owac/runtime/n02_v12_monitor.py` | 686 | 只读 HTTP 监控端（stdlib，ThreadingHTTPServer），监听 `127.0.0.1:18080` |
| `src/owac/runtime/n02_v12_monitor.html` | 484 | 单文件页面，原生 JS + canvas 折线，无 CDN/框架/WebSocket |
| `src/owac/runtime/n02_v12_status.py` | 144 | 训练端写入 helper（纯标准库，原子写） |
| `scripts/n02_v12_seed_sample.py` | 166 | 「界面测试」样例写入端（只写样例 run 目录） |
| `research/experiments/n02_v1_2/owac-n02-v12-monitor.conf` | 16 | supervisor 配置（与远端 `/etc/supervisor/conf.d/` 内容一致） |

远端新增文件：`/etc/supervisor/conf.d/owac-n02-v12-monitor.conf`、
`/workspace/owac/deployment/owac-n02-v12-monitor.conf`、`/workspace/owac/deployment/n02_v12_seed_sample.py`。
同步命令（无 `--delete`，只传自己的文件，未覆盖他人文件）：

```bash
rsync -az -e "ssh -p 51584 -o BatchMode=yes" src/owac/runtime/n02_v12_monitor.py \
  src/owac/runtime/n02_v12_monitor.html src/owac/runtime/n02_v12_status.py \
  root@93.91.156.102:/workspace/owac/worktrees/n02-measurement-completion/src/owac/runtime/
rsync -az -e "ssh -p 51584 -o BatchMode=yes" scripts/n02_v12_seed_sample.py \
  root@93.91.156.102:/workspace/owac/worktrees/n02-measurement-completion/scripts/
```

## 2. supervisor 配置与启停

```ini
[program:owac-n02-v12-monitor]
command=/usr/bin/python3 -u /workspace/owac/worktrees/n02-measurement-completion/src/owac/runtime/n02_v12_monitor.py --port 18080
directory=/workspace/owac/worktrees/n02-measurement-completion
autostart=true
autorestart=true
startsecs=3
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile=/workspace/owac/deployment/n02-v12-monitor.log
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=2
```

首次安装 / 恢复：

```bash
sudo cp /workspace/owac/deployment/owac-n02-v12-monitor.conf /etc/supervisor/conf.d/
sudo supervisorctl reread && sudo supervisorctl update
sudo supervisorctl restart owac-n02-v12-monitor     # 重启
sudo supervisorctl status owac-n02-v12-monitor      # 查看
sudo supervisorctl stop owac-n02-v12-monitor        # 停止
```

同一个 18080 入口的原 bootstrap 程序 `owac-n02-monitor` 已 `stop`（其下载/部署状态由本页
`①` 区块与 `/api/status.deployment` 保留显示）。**未重复绑定端口**，未改动 8080 Jupyter。

样例写入端（30 分钟一轮，可重复执行；`--clean` 删除样例）：

```bash
cd /workspace/owac/worktrees/n02-measurement-completion
PYTHONPATH=src nohup python3 /workspace/owac/deployment/n02_v12_seed_sample.py \
  --run-root outputs/n02-v12 --animate --ticks 600 --interval 3 > /tmp/n02-sample.log 2>&1 &
```

## 3. 实际 URL

- 用户侧（SSH 隧道）：`http://127.0.0.1:8081/`，隧道 `ssh -p 51584 root@93.91.156.102 -L 8081:localhost:18080`（8081 转发当前由先前已存在的隧道持有，实测可用）。
- 远端本机：`http://127.0.0.1:18080/`；接口 `/api/status`、`/api/history?run_id=<id>`、`/api/preview?run_id=<id>&file=<relpath>`。

## 4. 写入端接口（训练端请按此调用）

模块：`owac.runtime.n02_v12_status`

```python
from owac.runtime.n02_v12_status import update_status, append_history, heartbeat

update_status(run_dir, **fields) -> dict      # 读-改-写 status.json；自动刷新 updated_at，首次补 started_at
append_history(run_dir, step, wall_time, train_loss, dev_action_error) -> dict   # 追加一行 JSONL
heartbeat(run_dir, **fields) -> dict          # 轻量心跳：只刷 updated_at（可选带字段），不 fsync
read_status(run_dir) -> dict                  # 可选：读回当前 status
```

均为纯标准库；`update_status`/`heartbeat` 用「临时文件 + `os.replace`」原子替换；`append_history` 只追加。
建议：每 10 optimizer updates 或 5 秒调 `update_status`，长步骤用后台线程每 5 秒 `heartbeat`，
dev/保存/异常时立即 `update_status`。

## 5. 验收逐条结论

| # | 结论 | 证据 |
| --- | --- | --- |
| 1 | 通过 | `supervisorctl status` → `owac-n02-v12-monitor RUNNING`，`autorestart=true`；`kill -9` 后 supervisor 自动拉起新 pid（28641→30457）且页面仍 `http=200`；SSH 会话结束不影响（整个验证过程由多条独立 SSH 完成，进程 pid 稳定） |
| 2 | 通过 | `/api/status` 返回合法 JSON（含 `display/gpus/deployment/runs`）；穿越与软链接逃逸全部 400/404（见 §6） |
| 3 | 通过 | 样例 run `ui-sample-raw`/`ui-sample-rgb`（`ui_test: true`，页面上有「界面测试」徽标）：step 递增 24→36（10 秒两次采样）、两条曲线、两张 GPU 卡片各自绑定 GPU0/GPU1、预览 4 格、闭环 3/100、日志 30 行；样例在独立 run 目录，未写入真实作业目录 |
| 4 | 部分 | 停更 >30 秒实测：横幅「数据未更新，请查看日志：最后心跳 …（47 秒前），超过 30 秒未更新；此处不判定训练失败。」且状态行仍为「运行中」（不判失败）。**真实作业尚未出现**，切换规则已用本地合成真实 run 验证（非样例 run 一旦有 status.json 即接管 current/raw）；待主 agent 的 `raw-n02-robotwin-s1103` 出现后可直接复核 |
| 5 | 通过 | 390px：`documentElement.scrollWidth=390`、越界元素 0 个、无横向溢出；attention 叠加图可在页面内点开（`/api/preview` 200 `image/svg+xml`），真实 attention 图未生成时显示「尚未生成」（用只有 raw/rgb 槽位的 rgb 样例实测 2 处「尚未生成」） |
| 6 | 通过 | 截图见 §7 |

## 6. 真实现场输出摘录（远端）

`curl 127.0.0.1:18080/api/status`（字段摘录）：

```json
{"updated_at": "2026-10-15T16:12:19Z", "stale_seconds": 30.0,
 "server": {"pid": 30457, "started_at": "2026-10-15T16:10:45Z", "port": 18080},
 "gpus": [{"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "mem_total_mib": 97887}, {... index 1 ...}],
 "deployment": {"ToferFish/RawVLA-Bench": "complete", "SidneyXie/pi05_robotwin": "complete",
                "TianxingChen/RoboTwin2.0": "complete", "pi05_libero_jax": "complete"},
 "display": {"current": "ui-sample-raw", "raw": "ui-sample-raw", "rgb": "ui-sample-rgb"}}
```

`/api/history?run_id=ui-sample-raw` → `http=200 bytes=9314`，数组首行：

```json
[{"step": 4, "wall_time": 1.68, "train_loss": 0.95, "dev_action_error": 1.39}, ...]
```

路径穿越 / 逃逸（真实响应）：

```
history?run_id=../../etc           -> 400 {"error": "invalid run_id", "run_id": "../../etc"}
history?run_id=/etc/passwd         -> 400 {"error": "invalid run_id", "run_id": "/etc/passwd"}
history?stage=../../etc&run_id=x   -> 404 {"error": "unknown stage", "run_id": "x"}
preview file=../../../../etc/passwd        -> 400 {"error": "invalid file", "run_id": "ui-sample-raw"}
preview file=/etc/passwd                   -> 400 {"error": "invalid file", "run_id": "ui-sample-raw"}
preview file=../../../deployment/downloads.json -> 400 {"error": "invalid file", ...}
preview file=%2e%2e%2f%2e%2e%2fetc%2fpasswd    -> 400 {"error": "invalid file", ...}
preview file=preview/escape.svg  (软链接→/etc/hostname)      -> 400
preview file=preview/etcdir/hostname（目录软链接→/etc）      -> 400
```

响应只含固定字段白名单（`STATUS_FIELDS`），训练端写入 status.json 的其他字段不回显；页面不含任何凭据或环境变量。

## 7. 截图（本地绝对路径）

| 文件 | 内容 |
| --- | --- |
| `/home/jeong/zeno/owac/worktrees/n02-measurement-completion/research/experiments/n02_v1_2/n02_v12_ui_test_desktop_1400.png` | 1400px 全页：①~⑤ 全部区块 |
| `.../n02_v12_ui_test_mobile_390.png` | 390px 全页，无横向溢出 |
| `.../n02_v12_ui_test_missing_preview_390.png` | 390px，切到缺 attention/残差的样例 → 2 处「尚未生成」 |
| `.../n02_v12_ui_test_attention_open_1440.png` | attention 叠加图在浏览器中打开 |
| `.../n02_v12_stale_over_30s_mobile_390.png` | 停更 >30 秒：「数据未更新，请查看日志」，状态仍「运行中」 |
| `.../n02_v12_network_error_snapshot_390.png` | 断网：横幅「网络异常：保留最后有效快照（时间）」，卡片保留最后数值 |

截图命令（本地无头 Chrome + 最小 CDP 脚本，无额外依赖）：

```bash
ssh -f -N -p 51584 -L 8087:127.0.0.1:18080 root@93.91.156.102
node /tmp/cdp_probe.mjs --url=http://127.0.0.1:8087/ --width=390 --height=844 \
  --out=<截图路径>.png --wait=11000
```

## 8. 已知限制 / 待确认

1. **真实作业切换**：`raw-n02-robotwin-s1103` 尚未创建，页面上的 `①③④⑤` 与 RAW 卡片目前自动跟随「界面测试」样例。真实 run 一旦写入 `status.json`，页面按规则自动切到它（`display.current/raw`）；RGB 卡片同理等 `rgb-n02-robotwin-s1103`。
2. 样例 run 目录在 `outputs/n02-v12/P2/ui-sample-*`（`status.json` 内 `ui_test: true`）。**任何汇总/评测脚本请过滤它们**，不需要时执行：
   `python3 /workspace/owac/deployment/n02_v12_seed_sample.py --run-root outputs/n02-v12 --clean`。
3. 旧 bootstrap 程序 `owac-n02-monitor` 的配置文件仍在 `/etc/supervisor/conf.d/owac-n02-monitor.conf`（已被 stop）。若主机重启 supervisord，两个程序会争 18080：建议由该文件所有者删除或改为 `autostart=false`；我未改动他人文件。
4. `/api/history` 最多返回最后 20000 行；`log_tail` 截取最后 30 行、每行 ≤500 字符。
5. 页面首屏两条曲线依赖 history 拉取（经 SSH 隧道约 1.5 秒）；加载期间显示「历史数据加载中…」。

## 9. W1 复核后的 3 项小改（2026-10-16，Lead 指令）

1. **旧页重启竞争**：`/etc/supervisor/conf.d/owac-n02-monitor.conf` 的 `autostart=true` → `autostart=false`
   （文件保留可回查，未删除；副本 `/workspace/owac/deployment/owac-n02-monitor.conf.autostart-false.bak`）。
   `supervisorctl reread && supervisorctl update` → `owac-n02-monitor: changed / stopped / updated process group`，
   现状 `owac-n02-monitor STOPPED Not started`，18080 只由 `owac-n02-v12-monitor` 独占（supervisord 重启后也不会再抢端口）。
2. **runs 列表过滤**：`snapshot_runs()` 只列**确实含 `status.json` 的一层目录**（`<root>/<stage>/<run_id>/status.json`）。
   实测 `curl /api/status`：
   `runs: [('ui-sample-raw', 'P2', 'raw_n02', True), ('ui-sample-rgb', 'P2', 'rgb_n02', True)]`，
   磁盘上同层的 `acceptance/`、`measurement-completion/` 与散落的 `.py/.json/.log` 文件均不再被当成作业。
3. **样例保留**：`ui-sample-raw/rgb` 未清理；样例写入端续跑 `--ticks 1800 --interval 3`（约 90 分钟后自停，数据保留），
   保持样例「运行中」以便复核「样例 → 真实作业」切换。停止方式：`kill <seed_sample 的 python3 pid>`。

## 10. W2 复核：界面测试样例 → 真实作业（2026-10-16）

真实 run `outputs/n02-v12/P2/raw-n02-robotwin-s1103`（写入端即 `owac.runtime.n02_v12_status`）。

**切换与真实数值（`curl /api/status`）**
- `display: {current: raw-n02-robotwin-s1103, raw: raw-n02-robotwin-s1103, rgb: ui-sample-rgb}`
- 第一次采样：`step 172/1000, updated_at 18:52:06Z, state running, eval {planned:100, completed:0, success:0, task_failure:0, infrastructure_error:0}`
- 45 秒后：`step 181/1000, updated_at 18:52:50Z` → step 与 updated_at 均前进；页面 ⑤ 显示 `0 / 100`、成功率 `—（暂无已完成回合）`，100 次待跑不计入分母。
- 卡片值（页面渲染文本）：`GPU 0 · NVIDIA RTX PRO 6000 Blackwell Max-Q · 显存 58214 / 97887 MiB · step 188/1000 · 每 step 5.884 秒 · loss -1.277 · dev —`。

**曲线**：`/api/history?run_id=raw-n02-robotwin-s1103` 返回 127 条真实点（step 1..183，`dev_action_error` 仅 3 点非 null）；样例 history 1800 条、step 轴 4..1000。图表附注实测 `共 186 step · raw_n02（真实）（原始点，未平滑） · 已排除界面测试样例`。

**GPU 卡片**：RAW 绑 GPU0（58214 MiB，来自 status.json 的 `gpu.cuda_visible_devices='0'`）；RGB 卡显示 `未运行｜该 arm 暂无真实作业（界面测试样例 ui-sample-rgb 不计入真实数据…）`。

**截图**（`research/experiments/n02_v1_2/`）：
- `n02_v12_real_run_desktop_1400.png`
- `n02_v12_real_run_mobile_390.png`（390px，`scrollWidth=390`、越界元素 0）
- `n02_v12_real_run_sample_selected_390.png`（真实作业存在时显式查看样例：琥珀色样例卡 + `界面测试样例数据（非训练结果）` 徽标）

**复核过程中修掉的 4 个问题（都在我自己的文件里）**
1. `gpu` 字段是 dict（`{'cuda_visible_devices': '1', ...}`）时匹配失败 → 新增 `_gpu_candidates()`；实测 `-profile` run 现在正确绑 GPU1（此前错显 GPU0）。
2. 多真实 run 并存时「最近心跳」会把 `-profile` 侧跑顶成 current → 新增主作业规则：新鲜运行中的真实 run 里按「目标步数大 → 开始时间早 → 心跳新」选；主 run 消失后退回最近更新的真实 run。
3. 选择器切换 run 时横幅不跟随 → `onchange` 补 `renderBanner()`。
4. 显式选样例时曲线附注仍写「已排除样例」；且 `state=failed` 被「数据未更新」盖住 → 附注改为「当前为界面测试样例数据」，失败态横幅优先于停更提示（停更仍不判失败）。

**样例与真实数据分离**：真实作业存在时，②卡片与③曲线只显示真实 run（样例仅可在「查看作业」中显式查看，届时带琥珀色卡片与徽标）；`sample-card` 默认视图中不出现。

## 11. 收尾确认：两个真实作业跑完后的在线状态（2026-10-16）

**页面存活 + display**：`curl /` → `http=200`；`/api/status`：
`display: {current: raw-n02-robotwin-s1103, raw: raw-n02-robotwin-s1103, rgb: rgb-n02-robotwin-s1103}`
（RGB 卡已指向真实对照臂，不再是样例/未运行）。supervisor `owac-n02-v12-monitor RUNNING`（pid 77791），18080 独占。

**①②真实值**（页面渲染文本，均取自 status.json / history.jsonl，不读 eval/eval_v12.json）：
- RAW：`已完成 raw-n02-robotwin-s1103 · GPU 0 · step 1002/1000 · 已用 3 小时 15 分 · 每 step 6.219 秒 · checkpoint …/checkpoint_0001002.pt · loss -0.2004 · dev 动作误差 0.0805`
- RGB：`已完成 rgb-n02-robotwin-s1103 · GPU 1 · step 1000/1000 · 已用 1 小时 21 分 · 每 step 4.800 秒 · checkpoint …/checkpoint_0001000.pt · loss 0.2834 · dev 动作误差 0.0785`
- history：raw 626 条（末条 step 1002, dev 0.08049274422228336）/ rgb 623 条（末条 step 1000, dev 0.07845230028033257）；曲线附注 `共 1002 step · raw_n02（真实） / rgb_n02（真实）（原始点，未平滑）`。

**④ 预览**：`previews.tiles = 12`（3 相机 × 4 帧，四个槽位齐全）+ `other = 8`（每组 rounds/action 两张）；页面 ④ 渲染选中轨迹的 4 格 + 8 格其它 = 12 张图全部加载、`尚未生成` 计数 0；切换相机/帧（robotwin_left + 帧 60）会换成另一组真实图（`preview/02_attention_12331_robotwin_left.svg`，12/12 加载）。
`/api/preview?run_id=raw-n02-robotwin-s1103&file=preview%2F02_attention_3082_robotwin_head.svg` → `http=200 content-type=image/svg+xml bytes=322068`。

**⑤ 闭环**：`0 / 100`、`成功 0 / 失败 0 / 基础设施 0`、成功率 `—（暂无已完成回合）`；`planned=100` 只在进度条里。

**⑤ 语义回归**（本次改动后重测）：`state=failed` → 横幅 `训练报错：异常：failed（由训练端写入，非超时推断）`；`running` 但停更 50 秒 → `数据未更新，请查看日志…此处不判定训练失败`（状态行仍「运行中」）。远端两个 completed run（age 401 秒）同样显示「数据未更新…不判定训练失败」。

**样例隔离**：默认视图样例卡片数 0，曲线只含两个真实 run；样例仍需在「查看作业」里显式选择。

**本轮改动**（仅 `src/owac/runtime/n02_v12_monitor.py`）：`_primary_key()` 统一主作业排序键（真实优先 → 目标步数大 → 开始时间早 → 心跳新），跨 tier 使用 —— 修正「最后写盘的 RGB run 顶替 RAW 主 run 当 current」；预览归一新增非图片后缀过滤（`/api/preview` 只服务图片）。

**最终截图**：`n02_v12_final_desktop_1400.png`、`n02_v12_final_mobile_390.png`（390px：`scrollWidth=390`、越界元素 0、无 console error）。

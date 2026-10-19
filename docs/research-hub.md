# OWAC 研究统筹台

统一查看 N01、N02、N04 的开发树、当前技术路线、说明文档和提示词。2026-10-19 起代码、路线登记和说明统一从主 checkout 的本地 `main` 读取。三个方向全部暂停、远端不可达；旧阶段仅供回查，生成结果在 `outputs/research-hub/`。

## 打开与更新

```bash
cd /home/jeong/zeno/owac/repo
make research-hub
```

打开 [统筹台离线页面](../outputs/research-hub/index.html)。它包含文档全文，直接打开即可搜索、阅读、复制和下载，不依赖网络或额外前端环境。生成的页面是读取时的快照；源文档修改后重新运行命令并刷新页面。

日常使用建议打开[自动更新入口](http://127.0.0.1:8876/)。服务启动命令：

```bash
make research-hub-serve
# Local: http://127.0.0.1:8876
```

服务仅监听本机。打开、刷新浏览器页面或点击“重新读取”时，页面会请求固定的 `POST /api/rebuild`，由服务以主仓库为工作目录执行 `make research-hub`，成功后显示最新内容并更新离线文件。页面会显示生成中或失败状态；失败时保留已有内容。执行记录与日志在 `outputs/research-hub/last-refresh.json`、`last-refresh.log`。服务只接受固定构建目标，不接收任意命令；不修改研究工作树或启动训练。

本机已安装用户级 systemd 常驻服务 `owac-research-hub.service`，异常退出后 3 秒自动重启，开机自动启动；用户 `jeong` 已启用 linger，退出登录后也保持运行。它直接使用主仓库 `.venv/bin/python`，刷新构建所需的 uv 路径也已固定。旧 `server.pid` / `server.log` 只作为历史保留；当前 PID、状态和日志以 systemd 为准。

```bash
# Codex 终端若未继承桌面会话环境，先设置用户服务总线
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
systemctl --user status owac-research-hub.service
journalctl --user -u owac-research-hub.service -n 50 --no-pager
# 手动重启或停止
systemctl --user restart owac-research-hub.service
systemctl --user stop owac-research-hub.service
```

重装服务（默认 checkout 必须位于 `~/zeno/owac/repo`，且已完成 `uv sync --locked`）：

```bash
mkdir -p ~/.config/systemd/user
cp configs/research/owac-research-hub.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now owac-research-hub.service
```

服务运行时无需另开 `make research-hub-serve`。如需临时前台调试，先停止常驻服务，Ctrl+C 结束调试后再用 `systemctl --user start owac-research-hub.service` 恢复。离线快照中的“重新读取”默认进入 8876 端口。

## 当前暂停与历史内容

三个方向当前节点均为“暂停与本地证据”，分别显示 N01 的 P3 停止/动作诊断、N02 的 r17 读出结论及历史闭环、N04 的 S2 故障/修复边界。所有旧阶段改为历史节点，原文仍可阅读；当前不提供旧训练包作为执行入口。

[合并与保全索引](../research/archives/2026-10-19_worktree_consolidation/README.md)列出完整来源提交、接口处理和归档路径。服务器不可达时，不将页面生成时间、历史 PID 或 RUNNING 状态当作远端仍在运行；也不声称远端独有权重已经恢复。

“复制原文”保留未经重写的 Markdown 字符串。未来重新授权后，可以登记新的阶段执行包；来源哈希和提示词保真规则继续有效。

## 维护规则

路线登记：[configs/research/hub.json](../configs/research/hub.json)。

1. 当前三个方向的登记 checkout 为 `repo`、分支为 `main`；原 worktree 保留作恢复来源。统筹台只读取文档和 Git 状态，不启动训练。
2. 自包含提示词可设`prompt_kind=complete`，直接复制文件全文；其工作位置/阶段范围由完整包本身给出，不再叠加通用旧门槛或来源hash前缀。可选`research_focus`展示研究链路、比较方法、预算和数据边界，缺省时其他路线保持原界面。
3. 每条路线配置 `version`、`current_stage`、阶段说明与 `prompt_sources`。`current_stage` 是人工登记的当前核对点，不代表自动批准执行。
3. 文档自动收录各自 `docs/`、`research/candidates/`、`research/experiments/`、`research/reviews/` 内路径含本方向编号的 Markdown，另含 README、AGENTS、status 和 `documents` 明确列出的来源。不把工作树中其他方向的旧提示词副本混入。
4. 阶段变更后，先读原始证据，再更新登记表。`summary_sources` 是摘要所依据文件的 SHA256；原文变化后页面立即提示“摘要待复核”。核对并改好摘要后才更新对应 hash 和 `reviewed_at`，不能仅刷新 hash 消除提示。
5. 新路线可在登记表添加；checkout 只允许主仓库自身或同一容器的 `worktrees/` 下。原文缺失会显示原因并禁用依赖它的交接包，不自动选择旧版本。

日期统一显示为北京时间。文件修改时间表示本地文件时间，不是实验完成时间；Git HEAD 与未提交修改分报。摘要和阶段标签不会通过文档里出现“通过”二字自行升级。页面没有实验完成百分比，路线节点仅展示顺序。

## 文件与验证

- `src/owac/runtime/research_hub.py`：轻量读取、来源校验、离线生成及本机只读服务。
- `src/owac/runtime/research_hub_template.html`：无外部依赖的页面、阅读器及复制界面。
- `tests/test_research_hub.py`：原文保真、目录边界、阶段节选、来源变化、缺失来源与服务刷新。
- `outputs/research-hub/index.html`、`inventory.json`：本机生成页面及来源清单，沿用 outputs 忽略规则。

```bash
uv run --locked pytest -q tests/test_research_hub.py
make check
```

页面测试只验证统筹工具；不重新验收 N01/N02/N04 的模型、机制或机器人能力。

本次验证（2026-10-14）：8 项 Python 测试、`make check` 和 `git diff --check` 通过；浏览器核对全部 27 个交接包的实际剪贴板全文、原文复制、下载一致性、搜索筛选、只读刷新、离线复制回退及 320/375/768px 布局。浏览器收据保存在本地 `outputs/research-hub/browser-verification.json`。

自动刷新变更验证（2026-10-14）：9 项测试及本次文件 Ruff 检查通过。实际在应用浏览器分别点击“重新读取”和刷新标签页，均显示构建成功；收据确认 cwd 为主 repo、命令为 `make research-hub`、退出码 0。当时全库 `make check` 因 `research/archives/2026-10-14_rawvla_pi05_baseline/` 中的冻结源码及第三方代码格式问题失败，未修改这些历史资产。完整基线归档随后移至被忽略的 `outputs/rawvla-archives/`；本次全库提交前重新执行 `make check`（151 个 Python 文件）及 `uv run --locked pytest -q`，均通过，全部 122 项测试通过。

常驻服务验证（2026-10-15）：unit 校验通过，状态 enabled/active；向服务主进程发送 SIGKILL 后，systemd 自动创建新 PID，HTTP 页面和 snapshot 均恢复 200。恢复后实际 `POST /api/rebuild` 的收据确认主 repo 下 `make research-hub` 退出码为 0。用户 linger 已启用；开机启动配置已核对，未为测试而重启整机。回执见本地 `outputs/research-hub/service-verification.json`。

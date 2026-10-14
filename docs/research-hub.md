# OWAC 研究统筹台

统一查看 N01、N02、N04 的开发树、当前技术路线、说明文档和提示词。代码和路线登记留在主仓库；说明与提示词从各自工作树读取，生成结果在 `outputs/research-hub/`。

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

当前本机服务已作为独立后台进程启动，日志和 PID 在 `outputs/research-hub/server.log`、`server.pid`；它不是定时任务。系统重启或手动停止后用上述命令启动。前台启动时 Ctrl+C 停止；端口冲突时用 `make research-hub-serve HUB_PORT=8877`。离线快照中的“重新读取”默认进入 8876 端口，改端口后直接打开对应服务地址。

## 页面里的三种交接内容

- N01：v1.0 已有 P0—P8 独立阶段提示词，每阶段复制“共同契约 + 实现审计 + 参考导航 + 远程执行说明 + 阶段原文”。P1/P2 另引用早期策略预检。当前为 P0 待启动；旧 DBSR / 来源撤回证据保留，新模型尚未实施。
- N02：每阶段复制“工作树与阶段范围 + 共同契约 + 阶段原文”；E1/E2 额外绑定 `direct` / `objects` 模式。依据 P0 独立验收记录，当前节点为 P1 待启动；P0 通过的范围仅为基础、数据与数值参考。
- N04：v1.0 方案目前没有独立阶段 prompt。页面按 S0—S7 提供“完整方案 + 所选阶段启动语”，标明生成来源；预览展示对应章节，复制和下载包含完整方案。

“复制原文”复制未经重写的 Markdown 字符串。“复制完整交接包”包含工作树、目标分支、阶段与来源全文，逐份附 SHA256。浏览器拒绝剪贴板时会打开选中全文的手动复制窗口；下载也可直接使用。

## 维护规则

路线登记：[configs/research/hub.json](../configs/research/hub.json)。

1. 开发继续在各自 `worktrees/<task>/`。统筹台不会修改其文件或合并算法。
2. 每条路线配置 `version`、`current_stage`、阶段说明与 `prompt_sources`。`current_stage` 是人工登记的当前核对点，不代表自动批准执行。
3. 文档自动收录各自 `docs/`、`research/candidates/`、`research/experiments/`、`research/reviews/` 内路径含本方向编号的 Markdown，另含 README、AGENTS、status 和 `documents` 明确列出的来源。不把工作树中其他方向的旧提示词副本混入。
4. 阶段变更后，先读原始证据，再更新登记表。`summary_sources` 是摘要所依据文件的 SHA256；原文变化后页面立即提示“摘要待复核”。核对并改好摘要后才更新对应 hash 和 `reviewed_at`，不能仅刷新 hash 消除提示。
5. 新路线可在登记表添加；工作树必须位于同一容器的 `worktrees/` 下。原文缺失会显示原因并禁用依赖它的交接包，不自动选择旧版本。

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

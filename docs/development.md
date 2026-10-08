# OWAC 本地开发

所有开发从本仓库及其版本历史出发。主 checkout 为 `/home/jeong/zeno/owac/repo/`；工作空间父目录只容纳主仓库、入口指引与 `worktrees/`。当前为本地 Git 项目；远端和发布策略留待后续任务确定。

## 环境与检查

使用 Python 3.11+ 和 uv，在仓库根目录执行：

```bash
cd /home/jeong/zeno/owac/repo
uv sync --locked
make check
uv run --locked python -c "import owac; print(owac.__file__)"
```

`uv sync` 创建本地 `.venv` 并安装可编辑包及开发依赖。`uv.lock` 入库，`.venv` 不入库。`make check` 执行 Ruff 检查、格式检查与源码编译；`make format` 可格式化 Python 文件。

新增依赖时修改 `pyproject.toml`，执行 `uv lock`、`uv sync --locked` 并审查锁文件。当前不需要 GPU、数据集或硬件驱动。

有实际测试后使用 `uv run --locked pytest`，或运行受影响的具体测试。当前没有测试用例，直接调用 pytest 会报告无测试；不可把这一状态报告为测试通过。硬件/GPU 测试通过对应 marker 标注，同时在运行前显式检查资源和执行条件，marker 本身不会禁止运行。

## 工作流

1. 阅读根目录 `AGENTS.md`、当前状态及任务相关研究卡；检查工作树变更。
2. 新候选复制研究卡模板，新实验复制实验模板到 `research/experiments/`；先写可证伪问题和比较条件。
3. 日常功能可使用 `git switch -c <type>/<short-name>`。并行开发使用同一仓库的分支/worktree，明确文件所有权；worktree 一律放在工作空间的 `worktrees/` 下。
4. 逻辑放在 `src/owac/`，运行参数放在 `configs/`，命令入口放在 `scripts/`。输出路径由配置控制，避免硬编码个人目录。
5. 运行有意义的检查，记录命令和结果；更新受影响的研究卡与 `docs/status.md`。
6. 审查 `git diff --check`、`git diff` 及暂存内容后提交。提交消息建议 `type(scope): summary`。

## 创建与管理 worktree

在主仓库的初始提交完成后，可以按任务建立独立 checkout。例如以下命令会创建 `feat/raw-schema` 分支和对应工作目录（仅为使用示例，初始化时没有执行）：

```bash
cd /home/jeong/zeno/owac/repo
git worktree add -b feat/raw-schema ../worktrees/raw-schema main
cd ../worktrees/raw-schema
uv sync --locked
make check
```

各 worktree 有独立的 `.venv`、配置和输出目录；共享 Git 历史。大型数据通过配置引用统一存储，避免重复复制。用 `git worktree list` 检查已有 checkout，清理时先保存所需代码、未跟踪文件和被忽略的实验产物，再用 `git worktree remove`；不要直接删除仍有工作的目录。

路径约束同样适用于 agent 和工具创建的 worktree；若某工具不能指定本工作空间内的位置，使用支持显式路径的 Git 命令。不要在 `/home/jeong/zeno/` 下平铺新的 OWAC checkout。

## 记录与大文件

- `data/manifests/`：小型数据来源/校验和/拆分清单；原始数据可以位于 `data/` 的其他子目录或外部存储。
- `outputs/<run-id>/`：建议存放解析后的完整配置、环境、日志、指标、权重与图表；Git 中的实验记录引用该位置。
- `research/experiments/`：保存实验意图、完整复现命令、代码与数据版本、结果摘要和失败诊断。
- `docs/decisions/`：维护影响多个模块的工程决议；研究组合评审使用 `research/templates/decision.md`。
- `third_party/`：本地外部实现缓存；其来源、许可证、版本与本地修改需要记录，不把整套外部历史意外嵌入主仓库。

原始 RAW、权重和大日志默认被忽略；小型合成测试夹具可在 `tests/fixtures/` 按需建立并说明来源。不得使用 `git add -f` 绕过忽略规则提交数据或凭据。

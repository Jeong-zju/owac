# OWAC 本地开发

所有开发从本仓库及其版本历史出发。主 checkout 为 `/home/jeong/zeno/owac/repo/`；工作空间父目录只容纳主仓库、入口指引与 `worktrees/`。GitHub 远端为私有仓库 [Jeong-zju/owac](https://github.com/Jeong-zju/owac)，`origin` 地址为 `https://github.com/Jeong-zju/owac.git`，主分支为 `main`。推送和发布遵循当前任务授权。

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
3. 先完成 [B0 共有开发基线](decisions/0002-common-development-baseline.md)，再从冻结标签创建 F01—F12 首批开发分支。日常功能默认使用 `codex/<short-name>`；后续子任务可继续本家族已有分支。并行开发使用同一仓库的分支/worktree，明确文件所有权；worktree 一律放在工作空间的 `worktrees/` 下。
4. 逻辑放在 `src/owac/`，运行参数放在 `configs/`，命令入口放在 `scripts/`。输出路径由配置控制，避免硬编码个人目录。
5. 运行有意义的检查，记录命令和结果；更新受影响的研究卡与 `docs/status.md`。
6. 审查 `git diff --check`、`git diff` 及暂存内容后提交。提交消息建议 `type(scope): summary`。

## 创建与管理 worktree

先在 `codex/b0-common-baseline` 建设分支完成共有工具与验收。B0 合入本地 `main` 后，以注释标签 `owac-dev-v0.1.0` 固定起点并记录 SHA；标签当前尚未创建。各候选首批分支使用该固定标签，避免开始日期不同导致公共代码不同。

以下命令仅在 B0 冻结后执行，创建候选分支和对应 checkout：

```bash
cd /home/jeong/zeno/owac/repo
git worktree add -b codex/f01-mechanism ../worktrees/f01-mechanism owac-dev-v0.1.0
cd ../worktrees/f01-mechanism
uv sync --locked
make check
uv run --locked pytest
```

各 worktree 有独立的 `.venv`、配置和输出目录；共享 Git 历史。候选记录其基线标签与 SHA；公共修复经共享分支验证、合入 `main` 后另建版本标签，候选显式合并需要的更新并记录，不移动旧标签。大型数据通过配置引用统一存储，避免重复复制。用 `git worktree list` 检查已有 checkout，清理时先保存所需代码、未跟踪文件和被忽略的实验产物，再用 `git worktree remove`；不要直接删除仍有工作的目录。

路径约束同样适用于 agent 和工具创建的 worktree；若某工具不能指定本工作空间内的位置，使用支持显式路径的 Git 命令。不要在 `/home/jeong/zeno/` 下平铺新的 OWAC checkout。

## 记录与大文件

- `data/manifests/`：小型数据来源/校验和/拆分清单；原始数据可以位于 `data/` 的其他子目录或外部存储。
- `outputs/<run-id>/`：建议存放解析后的完整配置、环境、日志、指标、权重与图表；Git 中的实验记录引用该位置。
- `research/experiments/`：保存实验意图、完整复现命令、代码与数据版本、结果摘要和失败诊断。
- `docs/decisions/`：维护影响多个模块的工程决议；研究组合评审使用 `research/templates/decision.md`。
- `third_party/`：本地外部实现缓存；其来源、许可证、版本与本地修改需要记录，不把整套外部历史意外嵌入主仓库。

原始 RAW、权重和大日志默认被忽略；小型合成测试夹具可在 `tests/fixtures/` 按需建立并说明来源。不得使用 `git add -f` 绕过忽略规则提交数据或凭据。

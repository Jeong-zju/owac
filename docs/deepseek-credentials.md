# DeepSeek 编程与闭环凭据

2026-10-20 用户指定两把不同用途的 API key。本机分别保存在当前用户的私有目录，目录权限 700，文件权限 600；仓库、提示词、网页与运行收据只记录角色和变量名。

| 用途 | 变量 | 本机私有文件 | key 尾号（仅识别） |
| --- | --- | --- | --- |
| 编程、Harness 派工、代码与测试修改 | `DEEPSEEK_CODING_API_KEY` | `~/.config/owac/deepseek/coding.env` | `164fee` |
| 闭环期间的在线推理、规划与决策请求 | `DEEPSEEK_CLOSED_LOOP_API_KEY` | `~/.config/owac/deepseek/closed-loop.env` | `0b2545` |

编程代理启动实验进程时，编码会话仍使用 coding；实验子进程发出的在线 DeepSeek 请求使用 closed-loop。用途按请求本身区分。没有 DeepSeek 请求的 π0.5 推理不需要 API key。

## 启动入口

在 `/home/jeong/zeno/owac/repo` 使用统一入口。它只读取所选角色的文件，清除另一角色变量，并为当前子进程将 `DEEPSEEK_API_KEY` 映射到所选 key。缺失或权限错误直接报错，不回退到通用 key 或另一把 key。

```bash
# 仅核对配置，不调用 API
uv run --locked python -m owac.runtime.deepseek_credentials --role coding --check
uv run --locked python -m owac.runtime.deepseek_credentials --role closed-loop --check

# 编程 Harness：可见 Web 服务，启动后打开认证入口并在专用对话发送工作单
/home/jeong/zeno/owac/repo/.venv/bin/python -m owac.runtime.deepseek_credentials \
  --role coding -- /home/jeong/.local/bin/dsh --profile web \
  --patch /home/jeong/zeno/owac/repo/configs/research/raw_distillation_harness.patch.yml \
  --host 127.0.0.1 --port 3081

# 在线闭环：-- 后替换为当次已授权、已核对的真实命令；此处不提供虚构 CLI
# .venv/bin/python -m owac.runtime.deepseek_credentials --role closed-loop -- <真实闭环命令>
```

Python 调度器使用 `environment_for_role("coding")` 或 `environment_for_role("closed-loop")`，通过 `subprocess.Popen(..., env=...)` 传入；不要修改父进程的全局 `os.environ`。远端运行须在实际执行主机安全配置同名私有文件，再从该主机注入环境；源码快照不包含凭据。

2026-10-21 起，编程只使用用户可见的 Harness Web 专用对话，禁止 headless／一次性后台编程派工。任务、回复、工具调用和返工均应在该会话可见；已有符合本任务配置的可见服务优先复用，新的专用 Web 服务默认使用 3081，不重启或接管原 3080 人工会话。启动认证 token 只用于打开浏览器，不放进提示词或共享记录。

当前编码 overlay 的 `llm-deepseek.config.apiKeyEnv` 固定为 `DEEPSEEK_CODING_API_KEY`；Web 保留 credentials / authorization / deepseek-account 服务，凭据存储路径单独指向 `~/.config/owac/deepseek/coding-web.credentials.yaml`，不读取共用 GUI 保存的其他 key。该私有文件初始 `refs: {}`、`records: {}`，实际 key 来自角色启动环境；不要通过设置写入其他用途的 key。现有文件已有值时须先核对其用途，不能覆盖后伪称沿用了原认证。其他 DeepSeek 消费端也必须核对实际凭据来源；环境不会覆盖 SDK 显式传入的旧 key。

## 提示词与记录

`docs/prompts/` 下所有 Markdown 提示词均附用途规则，包含独立阶段、公共契约、完整交接包及组装源。历史阶段仍遵循当前暂停／授权状态。重新组装后也须保留用途规则；改 key 时只更新本机对应私有文件，不将明文放进 prompt、YAML、参数、日志、网页或 Git。

核对只能输出角色、变量、配置是否存在；需要请求回执时记录角色与 provider/model，不记录 key。配置检查不等于真实 API 可用，也不产生新的训练或闭环授权。

## 编程权限默认值（2026-10-21）

任务 Web overlay 固定 `sandbox-policy.mode: danger-full-access`、`permission.defaultPreset: danger-full-access` 和 `approval.policy: never`。编程任务默认完全权限、无需人工审批。新建后核对实际 Permissions；旧设置或续接会话保留旧模式时，由执行者在可见对话发送 `/permission danger-full-access` 或操作 Permissions 完成切换，不要求用户手动批准。设置默认值不回写其他会话；当前研究任务、预算与停止点继续沿用。

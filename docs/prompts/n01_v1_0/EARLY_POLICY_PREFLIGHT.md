# 早期 π0.5 接口预检

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

这份提示词由 P1/P2 明确引用，不能替代 P7。依[远程说明](03_REMOTE_EXECUTION.md)在用户服务器运行；只检查真实数据和后端契约，最多 200 updates 的开发适配，不启动完整闭环清单。

## mode=contract 在 P1 末执行

先沿用已验收旧三通道编码器/真实特征桥读取两平台 NPZ，核对 P0 四区路径、新 SourceKey 与实际相机/帧对应。P1 的解析系数不是完整新神经前端；明确报告实际用的是旧参考编码器。复用 `observation_from_batch`、`action_masked_loss`、`load_backend` 和 `RawVLAFeaturePolicy`，每平台至少两个不同真实 batch 的 forward/backward/update，固定 backend 参数但允许梯度回传桥与 encoder。

读取实际 action/state normalization，LIBERO 控制7维/模型32维，RoboTwin控制14维/模型32维；时间 padding 按 mask 排除，保留原生32维 flow target 的既有处理。四区仅使用 train 的训练 batch，validation 只作无更新检查。输出实际参数更新与模型/输入 hash。

## mode=raw_adapter 在 P2 末执行

在拟新增 `frameworks/n01_full/adapters.py` 建 `DbsrFeatureAdapter`，复用完整 DBSR，并显式替换输入 stem 为3通道（CFA4通道仍保留）。包装成外部 `init_state/step/readout/reset` 接口，暂时关闭 P3 支持；实际 P2 motion 接入因果窗口。P4 复用这份适配，不另建数值转换。

DBSR 512维空间特征用确定性带mask池化到每相机16×16 token，然后共同 `VisualTokenPostprocessor(in_dim=512,width=2048,tokens_per_camera=256)` 投影；全无效 cell 返回 false mask。保持相机顺序、空间位置和视觉前缀缩放约定，禁止额外隐含的2048→2048随机投影被漏计。

对两平台各做真实小批次训练/推理、重复同一帧、换episode、部分reset、缺失相机与重复10步去噪检查。量化 source observe 次数与实际唯一帧匹配；重复读出不减少解析方差。辅助RAW loss默认关闭。

原图像塔退出主链，保持语言/动作专家。用视觉塔调用计数/钩子证明该 batch 没执行图像塔；记录 prefix token shape 和 mask；不能只看 import 成功。

## 交付

新增 `runtime/n01_full_preflight` 的 `policy` 子命令或独立小入口，模式显式选择。收据写实际使用的旧/新模块、steps、input identity、grad/update、源包 hash、backend freeze、服务请求和未验证项。预检成功只说明接口/梯度连通；P7 仍要按完整新模型完成三种子动作训练与开发闭环。

# N02 v1.2 前端专家任务

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

## 立即调用前端专家的明确任务

在开始长训练前，调用前端专家子任务并把以下任务原文交给它。你负责训练代码、状态输出与集成；专家只负责监控后端/页面。双方不独占整个仓库，不回退对方修改。若当前DeepSeek客户端没有子agent接口，由你按同一任务完成，不能因无法委派停止。

> 请为N02 v1.2实现一个简单、可实时查看的中文进度页。你拥有`src/owac/runtime/n02_v12_monitor.py`与`src/owac/runtime/n02_v12_monitor.html`；主agent拥有模型/训练/评测，先约定下面JSON字段。已有部署临时页在18080时，在同一个监控入口上升级，保留其下载/部署状态，不重复绑定端口。只用Python标准库ThreadingHTTPServer+单HTML/CSS/原生JS与本地SVG/canvas折线，不用React、Node构建、数据库、账号系统、WebSocket、外部CDN、复杂路由或控制作业按钮。
>
> 页面从上到下只有：①当前阶段、正在做的事、最后心跳、运行/完成/异常文字；②RAW与RGB两列卡片，显示GPU、真实optimizer step/目标、已用时间、最近step速度估计的ETA（暂无数据则留空）、checkpoint、当前loss与固定dev动作误差；③共享横轴两条训练/开发曲线，训练loss和dev误差单位不同分图，不画假平滑；④固定官方轨迹的RAW显示、同源RGB、attention叠加与残差预览，可选相机/帧/内循环，不存在的图写“尚未生成”；⑤闭环已完成n/100、成功s、失败f、基础设施错误e、当前成功率s/n，以及最近30行日志。未运行写未运行，分母不能把待跑项算进去。
>
> 数据接口`GET /api/status`与`GET /api/history?run_id=...`；只读配置好的run目录，禁止任意路径浏览。训练端每10 optimizer updates或5秒更新run下`status.json`，长步骤由轻量心跳每5秒更新时间；每次dev/保存/异常立即更新。写临时文件后原子rename。JSON约定：`run_id,stage,arm,platform,seed,state,activity,gpu,step,target_steps,updated_at,started_at,step_seconds,loss,dev_action_error,checkpoint,eval{planned,completed,success,task_failure,infrastructure_error},preview_files,log_tail`，不适用项null；history.jsonl追加`step,wall_time,train_loss,dev_action_error`。代码开发阶段也更新activity，不能训练开始前一片空白。可用现有bootstrap `/workspace/owac/deployment/downloads.json`显示部署状态。
>
> 浏览器每3秒轮询。超过30秒无心跳显示“数据未更新，请查看日志”，不把它自动判断训练失败；网络异常保留最后有效快照并显示更新时间。显存/利用率每10秒采样一次`nvidia-smi`，CPU读取即可。页面手机可读、无横向溢出、没有十几张仪表盘。localhost18080绑定，supervisor常驻，用户通过SSH的本地8081访问。凭据/完整环境变量不得入网页。
>
> 验证页面能显示一个明确标“界面测试”的样例，之后切换真实作业；检查真实step变化、训练错误显示、SSH退出后服务仍在、两GPU卡片独立、手机宽度和一张真实attention图可打开。样例不能混入训练结果。给主agent返回变更文件、启动命令、实际URL及截图位置。不要修改模型、损失、成功判据或训练预算。

你要把专家交付接入真实作业。网页是本阶段交付的一部分；无需追求设计奖，不允许花大量时间做权限、告警、调度等附加系统。DeepSeek长对话结束前确认网页仍显示当前真实状态，不能只交一个离线HTML截图。


# N01 v1.2 新服务器运行材料

2026-10-17最新授权：完整模型的两个GPU队列已注册，`owac-n01-v12-complete.supervisor.conf` 的 `autostart=true`。源码、配置和数据拆分清单冻结在新运行根 `source/`，队列预检两平台数据和后端资产，独立断点续训，完成receipt匹配才跳过。全局暂停锁已归档并由 `TRAINING_AUTHORIZATION.json` 记录新授权；旧主运行的audit hold和停用服务保留。下文暂停/休眠状态为此前记录。实时入口 [训练总览](http://127.0.0.1:18080/)，完整设置与实测见 [P3_training_launch.md](../../../research/experiments/n01_v1_2/P3_training_launch.md)。

实际部署结果见[部署记录](../../../docs/N01_new_server_deployment_2026-10-15.md)，不要将脚本存在等同于部署成功。服务器为`154.59.156.21:34302`，根目录`/workspace/owac/rawvla`。HF凭据只从服务器私有token存储读取。

- `install-envs.sh`：独立encoder（Python3.11/torch2.9.1 cu128）与policy（Python3.10/torch2.7.1 cu128），不用本地CPU锁。
- `requirements-policy.txt`：原工作环境的重要直接依赖，安装后以实际freeze记录完整版本。
- `download-assets.py dataset|robotwin|libero`：恢复固定RawVLA-Bench、RoboTwin权重、原generation的LIBERO JAX权重及tokenizer；libero对象表来自`records/baseline-archive/`。
- `convert-libero.sh`：原固定OpenPI转换与作者RoboTwin统计导出，LIBERO输出目录存在时不要重跑覆盖。
- `install-robotwin.sh`：独立模拟器和固定CuRobo，CUDA12.8/userspace编译，不操作宿主驱动。
- `restore-runtime-compatibility.py`：恢复归档的CuRobo Python文件与作者模拟器安装修补。
- `restore-sim-assets.sh`：官方RoboTwin资产，记录实际HF revision并解压。
- `run-on-gpu.sh raw|rgb <实际命令>`：RAW用物理GPU0、RGB用GPU1，前台执行供supervisor包装；进程内CUDA编号变成0。这是运行包装器，不是尚未实现的P3训练器。

新环境安装脚本用于新机，不是必须反复执行的验收关卡。先读服务器完整指南，检查已有环境/进程/数据，复用完成项。作者源码、当前N01 Git快照及本机原基准小型归档要先传到相应路径；本次已实际完成。长下载/安装/训练由独立supervisor管理，日志在`logs/`。既有Jupyter8080保留，未来研究监控用loopback18080经SSH访问。

每卡只排能放得下的作业，Base/N01默认串行；两卡可训练或验证并行，不引入DDP。开发闭环先生成总100条身份清单再拆到两worker，独立端口/输出并去重汇总。脚本不会自动启动P3长训练、额外种子或P4全量评测。

2026-10-16 独立验收：`p3-queue.sh` 是纳入版本管理的生产队列。先检查每项配置，再执行 LIBERO→RoboTwin；单项失败仍继续，队列最终返回非零。运行根存在 `P3_AUDIT_HOLD.json` 时退出 42。当前主队列已 STOPPED、autostart=false，hold 保留；不得移除 hold 来绕过[独立验收未完成项](../../../research/experiments/n01_v1_2/P3_independent_acceptance.md)。`train --diagnostic` 只用于隔离的有限诊断，不是正式训练。


2026-10-17代码补全：当前有效配置为 `configs/encoders/n01_full/v1_2/complete/` 的8份JSON，源码验收见 [P3_code_completion.md](../../../research/experiments/n01_v1_2/P3_code_completion.md)。队列默认输出新根 `v1-2-p3-complete`，不续旧主运行；保留用户全局 `TRAINING_HOLD.json`。`owac-n01-v12-complete.supervisor.conf` 是休眠模板，未注册/未启动。`robotwin-native.yml` 接 `owac.runtime.n01_v12_robotwin`，使用已验证的 `policy_norm_stats.json`，在线读取作者 `raw_rgb10` 并除1023；它保持50步动作重规划与显式OptiX适配。LIBERO完整配置使用生成副本的native transport补丁，32帧及实际ID不能由任意递增查询计数替代。真实服务需新配置匹配的检查点，未训练诊断需显式 `--allow-untrained`。

# N02 v1.2 新服务器操作与交接

当前远端`93.91.156.102:51584`，活跃checkout `/workspace/owac/worktrees/n02-measurement-completion`。本轮Codex负责重部署，实际完成项和大数据下载进度见[部署记录](../../../research/experiments/2026-10-15_n02_v1_2_remote_deployment.md)；不要照旧v1.1文档重建另一份仓库或连接已停用服务器。

```bash
ssh -p 51584 root@93.91.156.102 -L 8080:localhost:8080 -L 8081:localhost:18080
cd /workspace/owac/worktrees/n02-measurement-completion
export N02_ROOT="$PWD"
export N02_PY="$N02_ROOT/outputs/n02-full/runtime/py310-cu128/bin/python"
export PYTHONPATH="$N02_ROOT/src:/workspace/owac/rawvla/source/starVLA"
export HF_HOME=/workspace/.hf_home
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

先读当前AGENTS/README/status和本阶段完整包，直接在远端改代码。Python3.10运行时通过`PYTHONPATH`使用源码，不将本项目的Python≥3.11开发包强装进该环境。`make check`使用checkout根`.venv`；GPU/模型检查显式用`N02_PY`。不复制旧venv，不修改全局`/venv/main`。

GPU0默认RAW、GPU1默认RGB；分别使用`CUDA_VISIBLE_DEVICES=0`或`1`，进程内均用`cuda:0`。每个作业独立输出、端口和sim临时目录。首版不用DDP两卡一起训练一个模型。两平台原策略分别有独立checkpoint和normalizer，尤其RoboTwin显式MEAN_STD；数据路径、动作协议与引用已嵌入每阶段包。

实际大数据由`owac-n02-download`通过HF固定revision下载到`/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench`，断线后supervisor继续。只有少量样本可读时可以开始完整模型编程/IO检查；正式训练在对应分区所需文件到齐后启动，不悄悄缩训练池或改成合成数据。下载器可恢复已有完成文件，不逐文件重复计算714GB hash。HF认证已在服务器私有缓存配置，任何报告/页面都不打印token。

```bash
supervisorctl status owac-n02-download owac-n02-monitor
cat /workspace/owac/deployment/downloads.json
curl -fsS http://127.0.0.1:18080/api/status
```

本机浏览器<http://127.0.0.1:8081/>。远端8080是平台Jupyter，研究监控独立监听127.0.0.1:18080。部署页实现位于`/workspace/owac/deployment/bootstrap_monitor.py/.html`，是可用的部署进度页；P2按[前端专家任务](02_FRONTEND_EXPERT.md)升级真正训练曲线/attention/闭环进度，复用`owac-n02-monitor`服务，不开第二个抢端口的服务。

训练使用命名`owac-n02-v12-*`的supervisor作业；异常不无限重启，保留last/日志和恢复命令。查看已有任务后再分配两卡，不停其他进程。`owac-n02-bootstrap-libero/robotwin`只是迁移验收的原RGB π0.5服务配置，后续不是N02已训练策略；迁移检查结束后停止以释放显存，需要原模型对照时再手动启动。

交接保存可运行配置、实际optimizer计数、模型/产物绝对路径、命令、简洁报告和必要小型结果。大权重/原始RAW/视频留远端；本地同步源码差异与报告供独立验收，不同步venv、不覆盖本地其他未提交工作。没有要求提交推送。默认只做当前阶段，完成后停止。

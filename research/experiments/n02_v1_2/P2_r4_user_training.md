# P2 r4：用户训练入口

> 后续授权更新：用户已要求由 Codex 接管并启动训练，当前 r5 两臂已运行；本文保留 r4 手动入口的历史身份，现行状态与控制方式见[运行记录](P2_training_live_20261016.md)。

2026-10-16。用户明确训练由自己开启。本轮没有安装下面的 supervisor 配置、没有注册新训练程序、没有执行 start；四份真实画像均使用 `--profile-only`，优化更新 0。画像和代码修补见 [报告](P2_fresh_profile_20261016.md)。

当前版本 `n02-v12-p2-causal-r4`。完整 ConvNeXt V2-Base 从随机初始化开始，π0.5 加载官方权重并冻结。8 帧按因果顺序执行，训练梯度保留最后 2 个不同帧；microbatch 1 anchor、梯度累积 2、seed 1103、1000 次更新的原预算保持。辅助权重仍为 masked RAW 1、causal RAW 1、noise 0；没有凭几个窗口的梯度比例自动调参。RoboTwin 3 相机均为 240×320，保持传感器单位；LIBERO 2 相机 256×256、loader /255，仍是 pseudo-RAW。均不代表原生 Bayer。

## 用户选择训练时执行

先登录新机，确认两张卡的实际占用。下面仅注册新名字的 r4 作业，旧 1000 步作业和监控配置保留；新作业 `autostart=false`、`autorestart=false`。注册后需用户明确执行 start。

```bash
ssh -p 51584 root@93.91.156.102
cd /workspace/owac/worktrees/n02-measurement-completion
nvidia-smi
mkdir -p outputs/n02-v12/logs
cp configs/runtime/n02_v12_p2_r4.supervisor.conf /etc/supervisor/conf.d/owac-n02-v12-r4.conf
supervisorctl reread
supervisorctl update owac-n02-v12-r4-raw owac-n02-v12-r4-rgb
supervisorctl status owac-n02-v12-r4-raw owac-n02-v12-r4-rgb
```

用户决定开哪一臂，就执行哪条（RAW 物理 GPU0，RGB 物理 GPU1；进程内均为 cuda:0）：

```bash
supervisorctl start owac-n02-v12-r4-raw
supervisorctl start owac-n02-v12-r4-rgb
```

训练输出分别是：

- `/workspace/owac/worktrees/n02-measurement-completion/outputs/n02-v12/P2/raw-n02-robotwin-r4-s1103/`
- `/workspace/owac/worktrees/n02-measurement-completion/outputs/n02-v12/P2/rgb-n02-robotwin-r4-s1103/`

正式目录现在不存在。CLI 会拒绝已有非空目录，抢占同一路径的第二个新进程也会失败，防止覆盖旧记录。不要先创建上面两个正式目录。日志在 `outputs/n02-v12/logs/r4-{raw,rgb}-robotwin{,.err}.log`；状态与每次更新的历史在各 run 目录。监控原有服务会扫描新 run；画像卡片和旧路径卡片各有独立 run_id，不能混算。

## 从 r4 检查点恢复

恢复必须是 r4，且表示/平台/模型、真实数据身份、seed、损失、优化器、batch/accum/BPTT 配置一致。旧 1000 步、r2 诊断点都不能 resume 为 r4。先停止对应已经退出或失败的训练程序，再在同一 GPU 运行如下命令；示例总目标 1500 必须大于检查点已完成步数。这里不是本轮已执行命令，也不授权代理继续训练。

```bash
cd /workspace/owac/worktrees/n02-measurement-completion
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=src:/workspace/owac/rawvla/source/starVLA \
outputs/n02-full/runtime/py310-cu128/bin/python -m owac.training.n02_v12 \
  --config configs/encoders/n02_v12/raw_n02_robotwin.json \
  --seed 1103 --max-updates 1500 \
  --output outputs/n02-v12/P2/raw-n02-robotwin-r4-s1103 \
  --resume outputs/n02-v12/P2/raw-n02-robotwin-r4-s1103/checkpoint_0001000.pt
```

保存采用临时文件完成后原子替换，包含全局 RNG、本地训练 generator、数据游标与 optimizer。恢复接线在 tiny CPU 真实模块测试中检查；本轮没有通过真实优化更新验证 r4 GPU 的续训后缀一致性。

## 仅复跑零更新画像

画像独立于训练。使用一个全新输出目录，`--profile-only` 与 `--resume` 互斥；该入口禁止 optimizer.step，结束后核对全部训练参数 SHA256，并断言 optimizer state 仍为空。以下复跑命令也未由本轮执行：

```bash
cd /workspace/owac/worktrees/n02-measurement-completion
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=src:/workspace/owac/rawvla/source/starVLA \
outputs/n02-full/runtime/py310-cu128/bin/python -m owac.training.n02_v12 \
  --config configs/encoders/n02_v12/raw_n02_robotwin.json \
  --profile-only --profile-batches 3 --previews 0 \
  --run-id fresh-profile-r4-raw-robotwin-user \
  --output outputs/n02-v12/P2/fresh-profile-r4-raw-robotwin-user
```

LIBERO 配置是 `configs/encoders/n02_v12/{raw,rgb}_n02_libero.json`，具有独立官方统计、7 维动作与 horizon 10；RoboTwin 为 MEAN_STD、14 维动作与 horizon 50。不要跨平台复制动作归一化或重复归一化。本轮准备的 supervisor 文件只包含 RoboTwin；LIBERO 的画像不等于已安排双平台训练。

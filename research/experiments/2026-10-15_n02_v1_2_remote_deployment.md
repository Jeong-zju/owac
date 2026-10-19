# N02 v1.2 新服务器部署记录

日期2026-10-15。用户要求Codex迁移代码、环境与数据资产，DeepSeek后续按P2—P4阶段包实现。服务器`93.91.156.102:51584`，hostname `15eadaf35577`；两张RTX PRO 6000 Blackwell Max-Q，各97887MiB，驱动610.43.02，系统CUDA12.8。代码、三套运行时、两套策略权重与双平台有限闭环迁移检查已完成；数据全集仍由supervisor后台同步。未启动N02 v1.2模型实现或长训练。

## 登录与实时查看

```bash
ssh -p 51584 root@93.91.156.102 -L 8080:localhost:8080 -L 8081:localhost:18080
```

监控页<http://127.0.0.1:8081/>。远端8080为平台Jupyter，保持运行；N02页只监听127.0.0.1:18080，由`owac-n02-monitor`常驻。页面显示真实下载文件/字节与两卡状态、明确标注尚未训练；P2前端专家将升级真实训练曲线、attention图和闭环计数。用户已有8080隧道时，另开`ssh -p 51584 -N -L 8081:localhost:18080 root@93.91.156.102`即可。

## 代码与运行时

- 远端主Git容器`/workspace/owac/repo`，活跃worktree`/workspace/owac/worktrees/n02-measurement-completion`，分支`lzh/n02-measurement-completion`。基于本地HEAD`861c46f67fff552c81fff89c5116c63b815224c5`带Git历史，并覆盖传送本地现有未提交源码/文档/配置；没有创建新提交或推送。
- P0冻结分区、anchors与小型fixtures已传送。开发环境`.venv`由`uv sync --locked`建立。
- N02 GPU环境`outputs/n02-full/runtime/py310-cu128`为Python3.10.19、torch2.11.0+cu128、torchvision0.26.0+cu128；补入π0.5实际需要的transformers4.57.0等依赖。CUDA FP32/BF16 matmul与conv探针实跑通过，探针`outputs/n02-full/remote-bootstrap/probe.json`。实际依赖清单另存本记录结果目录；不将原始锁中未补后端依赖的状态冒充最终环境。
- 作者策略环境`/workspace/owac/rawvla/envs/starvla`与仿真环境`envs/robotwin`均Python3.10.21、torch2.7.1+cu128/torchvision0.22.1+cu128，分别依既有部署freeze重建。没有修改全局`/venv/main`、驱动或其他服务。
- RawVLA源码`/workspace/owac/rawvla/source`固定`ba9c56b04da2c4931b70d8844710f5928d43775b`，StarVLA `57d508cb180d0510ac9060358009c0f6ffaac420`；LIBERO与RoboTwin使用该父版本的子模块。OpenPI转换函数固定`15a9616a00943ada6c20a0f158e3adb39df2ccac`。

## 资产与当前进度

| 项目 | 实际位置 | 当前结果 |
| --- | --- | --- |
| RawVLA-Bench全集 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench` | 正在下载；目标4810文件/713689392570 bytes，进度以实时页为准 |
| RoboTwin原始π0.5 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors` | 下载完成，原权重不转换；原processor导出`policy_norm_stats.json`完成 |
| LIBERO官方JAX π0.5 | `/workspace/owac/rawvla/checkpoints/pi05_libero_jax` | 下载完成；用固定官方函数转换到下行路径完成 |
| LIBERO转换后权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` | 812源键/813模型键，有效参数缺失、意外键与shape不符均0；唯一inactive lm_head按既有加载规则处理 |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` | 下载完成 |
| RoboTwin仿真资产 | `/workspace/owac/rawvla/source/third_party/RoboTwin/assets` | 三组资产全部解压并更新配置，CuRobo0.7.8已按CUDA12.8/sm120编译，SAPIEN真实闭环通过 |

HF认证只保存在服务器私有认证文件与HF缓存token文件，权限0600，不回传本地、不进入Git/提示词/网页。数据revision`8586c594fff9ce38580e29745f074e82e96e74f7`、RoboTwin权重revision`e49e2ab6c11f07511573b67261bd129e88d0a416`；只用下载器原生完整性/已有文件大小检查，不逐文件重算714GB hash。

## 继续查看与恢复

```bash
supervisorctl status owac-n02-download owac-n02-monitor
cat /workspace/owac/deployment/downloads.json
tail -f /workspace/owac/deployment/download.log
# 若下载异常，保留缓存，恢复同一任务
supervisorctl start owac-n02-download
```

下载脚本`/workspace/owac/deployment/download_assets.py`，监控临时实现`bootstrap_monitor.py/.html`。日志和大权重留远端。部署过程的有限检查用于验证环境/数据/权重/模拟器可用，不代表N02算法已实现、充分训练或成功率提升。两平台真实窗口、已加载模型特征桥反传/采样和模拟器闭环均已实跑，结果如下。

## 最终部署检查

[实际小型结果](results/2026-10-15_n02_v1_2_remote_deployment/summary.json)保存时间、源码身份、模型、真实窗口和部署试跑。实际结果如下：

| 检查 | LIBERO | RoboTwin |
| --- | --- | --- |
| 官方RAW窗口 | 两相机，各8×1×3×256×256，quantile | 三相机，各8×1×3×240×320，zscore/MEAN_STD |
| 真实π0.5特征桥反传/采样 | loss0.503585，视觉输入梯度L1 257.695，输出1×10×7 | loss0.852783，视觉输入梯度L1 398.336，输出1×50×14 |
| 原策略有限部署闭环 | Spatial task0，1/1成功 | grab_roller，seed0/Normal/DefaultISP/OptiX，1/1成功 |

特征桥检查输入是随机visual tokens和探针监督，只验证真实后端加载、可导性与采样；表中动作shape为去掉padding的归一化切片，不是已逆归一化的物理动作。RAW窗口是真实官方数据的8帧IO检查，未替代P0研究分区。两个1/1是原RGB策略迁移验收，不是N02成绩或100次确认评测。

部署时修复`src/owac/backends/rawvla_features.py`的真实CLI缺陷：语言token/mask留在CPU导致GPU前向失败；冻结后端又未给视觉输入开启求导，无法检查反传。现在探针输入统一设备，并报告视觉输入梯度。普通训练/推理主接口未改；P2仍需真实完整编码器梯度接线。修补后本地特征桥回归23 passed/2个缺本地资产的real-backend测试skipped；远端两平台真实权重命令另行通过，不能把skip算成真模型验证。两checkout的`make check`及统筹台10项测试通过。

运行时兼容修复包括：SAPIEN/robosuite保留`setuptools78.1.0`以提供`pkg_resources`；按作者安装脚本修正SAPIEN URDF UTF-8/.srdf读取与mplib条件，生成embodiment绝对资产路径，显式Vulkan ICD与OptiX。驱动未改。LIBERO服务用转换后**safetensors文件**参数，不误传父目录。

迁移试跑后已停止`owac-n02-bootstrap-libero/robotwin`释放两卡；配置可手动启动，默认不自动占用GPU。`owac-n02-download`与`owac-n02-monitor`继续运行。训练所需分区文件未齐时先编程/真实样本IO，不修改分区、不静默缩数据集；正式训练等对应文件到齐。当前数据完整性完成状态只认实时下载结果，不把少量窗口可读写成全集就绪。

当前`/workspace`是容器overlay，不是独立持久卷。代码/小型证据同步到本地，RAW与权重留实例；销毁/重建实例前应另行安排大资产备份。

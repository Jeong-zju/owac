## 远程训练和长期运行（新服务器，2026-10-15）

### 连接、端口与部署依据

旧服务器已停用。用户当前登录命令为：

```bash
ssh -p 34302 root@154.59.156.21 -L 8080:localhost:8080
```

这条命令将本机8080转发到远端Jupyter。新服务器实查远端8080仍被Jupyter占用，保留它；N01实时监控绑定远端`127.0.0.1:18080`。查看监控请另用：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -p 34302 root@154.59.156.21 -L 8080:127.0.0.1:18080
```

随后打开`http://127.0.0.1:8080/`。两条转发不能同时占同一本机8080；可保留Jupyter连接，监控改成`-L 18080:127.0.0.1:18080`，打开`http://127.0.0.1:18080/`。不要停止Jupyter、关闭现有认证或将监控公开绑定`0.0.0.0`。

进入服务器先完整阅读`/etc/vast-agents-guide.md`及`/opt/supervisor-scripts/README.md`，再检查：

```bash
supervisorctl status
nvidia-smi
df -h /workspace
ss -ltnp 'sport = :18080'
```

当前新服务器为**两张NVIDIA RTX PRO 6000 Blackwell Server Edition，每张97,887 MiB显存**；初查两卡均空闲，驱动580.95.05、CUDA toolkit12.8。根存储约1.5 TiB，初始为空。运行时重测，不把初查空闲当专用GPU保证。`/workspace`没有独立持久卷，回收/销毁会丢失数据；重要源码、配置和恢复点同步到用户已有存储，不自行购买服务。

本轮由Codex实际部署代码和环境，部署结果见`docs/N01_new_server_deployment_2026-10-15.md`和本目录`remote_snapshot.json`。原服务器只读快照另存`remote_snapshot_legacy_20261015.json`，只作历史，不能据此声称新机已有旧模型/作业。部署日志在`/workspace/owac/rawvla/logs/`，下载/安装作业名以`owac-n01-v12-restore-*`或`owac-n01-v12-install-*`开头。开始研究前读实际部署记录；尚未完成或未恢复的资产如实处理，不重复下载已完成文件。

### 双GPU调度：RAW/RGB训练和验证均可并行

默认**GPU0为RAW队列，GPU1为RGB队列**。每个平台内，GPU0依次跑`raw_base`、`raw_n01`，GPU1依次跑`rgb_base`、`rgb_n01`；空闲时可按已登记任务调整。两个独立单卡作业即可，不为了两卡并行引入DDP、跨机集群或复杂调度器。两平台共8个主体run，每run仍只有seed101。

```bash
# 这是设备分配示例；后接本阶段实际训练/评估命令。
CUDA_VISIBLE_DEVICES=0 "$N01_PYTHON" -m owac.training.n01_v12 <RAW作业参数>
CUDA_VISIBLE_DEVICES=1 "$N01_PYTHON" -m owac.training.n01_v12 <RGB作业参数>
```

`owac.training.n01_v12`是P3要求新增的入口，当前P0—P2快照还没有完整P3训练器，不能把上面模板当已运行命令。指定`CUDA_VISIBLE_DEVICES=1`后，进程内通常使用`cuda:0`；日志同时写物理GPU索引/UUID与进程内设备，避免两作业都挤到物理GPU0。各作业有独立输出、checkpoint和日志，读同一份只读数据，不共写缓存或checkpoint。只要单卡实际显存允许就并行；不要同时把同卡塞满多个完整π0.5，也不要求先占满两卡才允许工作。

**验证同样可以双卡并行。** 每个worker的策略和模拟器分配对应卡，服务端口分开，例如RAW18000、RGB18001；另一平台若同时开服务另选18005/18006，先查空闲。EGL/Vulkan/SAPIEN的渲染选卡可能不随`CUDA_VISIBLE_DEVICES`自动变化，检查实际渲染设备，不虚称已绑定。评测worker各写自己的episode记录，最后按episode身份合并，不能只相加可能重复的计数。 作者RoboTwin `eval.sh`会重新设置`CUDA_VISIBLE_DEVICES`，它的`gpu_id`位置参数必须传物理0/1；不要把进程内`cuda:0`误传成两份外层脚本都使用物理GPU0。新机已实查SAPIEN在`CUDA_VISIBLE_DEVICES=1`时选择PCI `0000:9a:00.0`（物理GPU1），但新建worker仍应确认实际设备。

**开发期仍是每轮总100次闭环，不是每张卡100次。** 调度前保存一份100条总清单，将不同episode身份分到两卡；若比较多个方法，每个方法实际rollout也计入这100次，各自报告成功数/分母。允许两卡评估同一当前模型的不同episode。训练/评测进程数不改变验证预算，不因并行自动跑完整四臂或启动P4。P4另按正式冻结清单拆分，原分母和正常失败均保留。

### 代码、环境和资产位置

| 用途 | 新服务器路径或约定 |
| --- | --- |
| 当前N01源码快照与提示词 | `/workspace/owac/rawvla/owac`，包含当前checkout未提交的新文件 |
| RawVLA作者源码 | `/workspace/owac/rawvla/source`，固定既有作者版本与子模块 |
| RAW/RGB数据根 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`；LIBERO `libero/raw`、RoboTwin `robotwin2/raw` |
| encoder Python | `/workspace/owac/rawvla/envs/encoder/bin/python`，独立Python3.11/CUDA torch2.9.1环境 |
| policy及同进程特征接入 | `/workspace/owac/rawvla/envs/starvla/bin/python`，独立Python3.10/CUDA torch2.7.1环境；实际兼容性以部署验证为准 |
| RoboTwin模拟器 | `/workspace/owac/rawvla/envs/robotwin`，独立环境，具体可用范围见部署记录 |
| LIBERO权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO统计 | 同上父目录`dataset_statistics.json`，`openpi_pi05_libero_franka` / `action` |
| RoboTwin权重与统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`、同目录`policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 原Default小型记录/有效参数 | `/workspace/owac/rawvla/records/baseline-archive/`，来自本机归档；不是重跑结果 |
| 新run | `/workspace/owac/rawvla/research/n01-v1/runs/v1-2-<stage>-<run-id>/` |

旧服务器P2正式运动checkpoint和大训练目录不包含在本机小型数字归档中。**新机上若没有可恢复权重，就明确写未恢复，不加载本地smoke权重冒充正式模型，也不为恢复旧三种子合成训练而阻塞P3。** 按P3已允许的成熟作者初始化与真实任务训练安排继续，记录实际初始化；不能编造旧checkpoint已迁移。

`RawVLASequenceDataset(root=...)`的默认推导与实际布局不同，显式传正确`raw_cache_root`、`stats_path`和manifest，不因默认路径找不到重下全部数据。新run建`source/`、`config/`、`outputs/`、`receipts/`，复制本任务实际源码、配置、vendor及许可证。仅`git archive HEAD`会漏未提交实现。不要打包`.venv`、700GB数据、密钥或其他方向输出。记录版本与diff，确认实际导入路径及有效配置即可，不建立逐文件哈希准入。

Hugging Face认证已通过服务器私有token存储配置；直接使用Hub客户端默认凭据。**不要把token写进提示词、源码、配置、网页、日志、命令行参数或提交记录，不运行会打印认证值的环境转储。** 需要认证时读取现有私有存储，下载正常完成就继续，不新增token检查门槛。

本地CPU torch锁不得覆盖远端CUDA环境。端到端训练在同一autograd进程导入编码器和固定π0.5，先验证真实forward/backward；可使用已兼容的policy环境或独立兼容环境。**不要通过HTTP特征服务训练端到端梯度**，HTTP只用于闭环推理。Blackwell用cu128或实测可用的wheel，不安装/升级宿主NVIDIA驱动。

### 作业、恢复和简单监控

训练、评测worker、监控分别用supervisor，命名`owac-n01-v12-<stage>-<raw|rgb|monitor>-<run>`。包装脚本前台执行、flush日志；只定向加载本任务program，不重启整个supervisor，不操作管理服务。引用镜像logging工具时按其实际参数调用，再启用严格shell选项，避免工具脚本读取缺失位置参数导致任务在启动前退出。仅SSH私有转发的监控不需要Caddy/portal条目，也不要套缺条目就跳过的guard。

训练保存model/optimizer/scheduler/scaler/RNG/sampler及配置，用临时文件完成后替换。有限队列正常完成后不无限重启；异常恢复接续最近checkpoint，不从step0覆盖。仅单种子101，不自动加211/307。一次短续跑验证即可，不要求跨机器浮点逐位一致。

监控页保持简单：一张两行GPU/作业表、当前阶段、真实step/rollout、曲线、案例和日志即可。RAW/RGBworker各写各自状态，轻量汇总器读它们，不能竞争写同一个JSON；全局显示`已完成/100`及各worker计数。网页关闭不停止训练，训练退出仍显示最后状态与原因，等待GPU如实显示等待；监控失败不杀训练。具体前端专家任务见下一段。

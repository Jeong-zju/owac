# N02 v1.2 执行约定

## 执行位置与本轮预算

这是 N02 v1.2 当前阶段的完整执行包。你要直接修改代码并执行本阶段要求，不能只输出设计建议。遵从用户最新指令：单训练种子 **1103**；当前阶段确认评测 **每个实际选中的模型共100次闭环**，不是每任务100次，不是每照明域100次，不是三种子，不是13,250条全集。四臂是归因设计，不是每次修改必须全部训练、全部评测的门槛。未跑项目写未运行，按已完成比较限定结论。

登录命令已更新，旧服务器停用：

```bash
ssh -p 51584 root@93.91.156.102 -L 8080:localhost:8080
```

新机远端8080由平台Jupyter占用，保持它运行。监控应用监听 **127.0.0.1:18080**；用户本地使用下面的附加转发，打开 **http://127.0.0.1:8081/**：

```bash
ssh -p 51584 root@93.91.156.102 -L 8080:localhost:8080 -L 8081:localhost:18080
```

本地提示词/验收镜像为`/home/jeong/zeno/owac/worktrees/n02-measurement-completion`；真实开发和训练使用下表远端路径。已在远端的会话不要再SSH自己。自动化SSH不需要`-L`。启动前读取`/etc/vast-agents-guide.md`、当前checkout的`AGENTS.md`、`README.md`、`docs/status.md`、`docs/plans/N02_full_development_v1.2.md`、`research/experiments/2026-10-15_n02_v1_2_remote_deployment.md`；后者区分实际部署结果与待做研究。旧v1.1提示词仅查源码来源，不继承旧阶段门槛。

| 用途 | 远端绝对路径 |
| --- | --- |
| 活跃Git worktree | `/workspace/owac/worktrees/n02-measurement-completion` |
| 分支 | `lzh/n02-measurement-completion` |
| 开发检查 | checkout内`.venv/bin/python`，`uv run --locked`，`make check` |
| N02 GPU解释器 | `outputs/n02-full/runtime/py310-cu128/bin/python`，相对上述worktree |
| 作者π0.5环境 | `/workspace/owac/rawvla/envs/starvla/bin/python` |
| RoboTwin仿真环境 | `/workspace/owac/rawvla/envs/robotwin/bin/python` |
| RawVLA源码 | `/workspace/owac/rawvla/source`，StarVLA子目录`starVLA` |
| 官方数据根 | `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench` |
| LIBERO转换后权重 | `/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors` |
| LIBERO统计 | `/workspace/owac/rawvla/checkpoints/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json` |
| RoboTwin权重与统计 | `/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors`、同目录`policy_norm_stats.json` |
| tokenizer | `/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model` |
| 本阶段产物 | `outputs/n02-v12/<stage>/<run_id>/`，权重/视频/原始日志留在远端 |
| 阶段报告 | `research/experiments/n02_v1_2/`，仅源码、配置、小型结果供本地验收 |

Codex负责本次新机初始部署；先看部署报告和实际路径，不重新下载已有714GB数据，不复制虚拟环境，不重新做P0完整parity。旧`p0-remote-*.json`只借数据路径/统计，不复制其中三种子、worlds8、旧scales或时延门槛到v1.2配置。远端Git带本地未提交覆盖，`git status`有修改是正常状态。保留它们，不执行reset/clean、不擅自提交推送。HF认证已在远端私有位置配置，需要下载时读取既有认证；不把token写进配置、报告、网页或命令日志。

服务器有两张约96GiB的RTX PRO 6000 Blackwell。先看`nvidia-smi`和现有supervisor任务。默认**物理GPU0跑RAW、GPU1跑RGB**，每个作业用`CUDA_VISIBLE_DEVICES`只暴露一张卡，进程内设备是`cuda:0`；不同run目录、日志、checkpoint、策略端口（如RAW18100/RGB18101）、仿真输出和临时文件。首版不用DDP把两卡绑在同一训练上。验证也可RAW/RGB各一张卡并行，训练和验证按实际显存排队，不能不加区分地同时挤在一张卡。两平台后端分别适配，单种子不等于两个平台共享同一动作头。

长期训练和监控用本机supervisor管理，分别命名`owac-n02-v12-*`，保存完整启动命令与恢复命令。训练作业`autorestart=false`，出错保存状态并显示原因；监控可自动重启。只管理自己创建的进程，不停止Jupyter/Caddy或其他项目任务。

## 研究与实现边界

目标是把RAW相对固定ISP RGB保留的动作相关线索转化为VLA能力。主链：**RAW3→ConvNeXt V2-Base多尺度空间特征→多候选潜在解释→RAW测量预测及可靠性比较→循环修正/状态重建→直接空间特征与潜在状态联合读出→π0.5**。RAW主链不生成RGB/sRGB/XYZ供策略输入，不用RGB教师。RGB只用于独立对照及可视化。

LIBERO是RGB8逆处理的三通道pseudo-RAW；RoboTwin是HDR-derived三通道float32测量；都不是原生Bayer。LIBERO不能证明恢复源RGB8已丢失的信息。RoboTwin是本轮RAW信息研究主平台，LIBERO用于兼容和补充。不要把正常光配对RGB当作退化RAW的同源RGB：受控RGB必须由该次**同一RAW经固定ISP**得到。

复用`src/owac/raw/rawvla_sequence.py`、`src/owac/frameworks/n02_full/data.py`与P0分区。LIBERO两相机，RAW uint8转FP32/255，8维本体，7维动作，offset0/horizon10/每5步重规划；禁止92维仿真states进入策略。RoboTwin三相机，float32原尺度，不除255、不逐帧max；原NPZ16维robot_states不是在线14维关节状态；沿用`actions[t]`为当前状态、offset1/horizon50，在线真实joint_action/vector，左6关节/左夹爪/右6关节/右夹爪，checkpoint明确**MEAN_STD**，不做默认min/max或重排。逐动作mask沿用loader。

注意现有特征桥的单位边界：`RawVLAWindowDataset`使用正式统计后，sample的state/action已经归一化；`prepare_control_targets()`只补齐32维，`control_actions()`只切回7/14维，**都不会替你做归一化或逆归一化**。训练不重复归一化；在线采样输出在送`take_action`前按各平台原统计逆归一化一次，并沿作者夹爪规则执行。不能把名为control_actions的归一化切片当物理控制量。

用P0 manifest保持源轨迹/相邻帧/照明变体隔离；8帧历史在原分区身份上重新生成窗口，前期不足用mask，不读取未来。测量ID包含episode/frame/camera；同一帧内部4轮、前缀重放、多尺度特征不变成多份独立证据。缺少时间戳时沿用frame索引，不伪造秒；未知成像参数保留未知标记，文件名/光照域标签/候选编号不能变成答案线索。模拟器隐藏状态和未来专家动作只能做标签或报告。唯一测量计数用简单集合，不建立大型哈希账本。

只保留会影响正确性的检查：真实输入/动作单位和shape、finite、因果性、分区隔离、先验/校准/梯度接线、episode reset、保存恢复。禁止新增90%纠错率、任意自定义综合分数、逐文件全量SHA门槛、全套算法parity重跑、必须连续N次loss下降或必须达到某个平台期。代码版本、运行配置、checkpoint路径、实际更新数和小型样本身份足以追踪本轮；下载器原生完整性检查保留，不循环扫描714GB算hash。NaN或接线错误要修，正常失败或没有性能提升要如实报告，不能回退随机模型、伪造数据或跳过失败。

理论只承诺适用条件内性质：固定有限候选与测量时，`q=softmax(log(p0)-E/tau)`解`E_q[E]+tau*KL(q||p0)`；学习能量的q称支持权重，不冒充校准后验。连续更新在同一完整目标上接受非增步骤，失败可保留旧状态并记录拒绝；不要求训练loss单调，不声称全局收敛。允许全局重建真实神经状态与缓存；首版不强制关系图、RJMCMC、局部传播或额外Julia/Theseus运行时。


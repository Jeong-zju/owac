# N01 v1.2 新服务器部署（2026-10-15）

用户已停用旧服务器。本次实际重新部署当前N01源码、作者RawVLA、CUDA环境、两平台数据和π0.5权重；这是开发环境迁移，不启动P3主体训练或100次开发闭环。

登录：`ssh -p 34302 root@154.59.156.21 -L 8080:localhost:8080`。新机GPU0/1均为RTX PRO 6000 Blackwell Server Edition，各97,887MiB；驱动580.95.05，CUDA toolkit12.8。初查两卡空闲；约1.5TiB容器存储，无独立持久卷。

当前代码在`/workspace/owac/rawvla/owac`，从本机当前checkout复制，包含未提交v1.2文件；Git基点`d3dec908d78d0a86a761abc985b83113d836c8cf`，分支`lzh/n01-evidence-integration`。RawVLA作者源码在同级`source`，主版本`ba9c56b04da2c4931b70d8844710f5928d43775b`及原固定子模块。原Default小型记录已恢复到`records/baseline-archive/`，数字不冒充新机重跑结果。

| 环境 | Python / torch | 用途 |
| --- | --- | --- |
| `envs/encoder` | 3.11.16 / 2.9.1+cu128 | 完整DBSR、SEA-RAFT及编码器研究 |
| `envs/starvla` | 3.10.21 / 2.7.1+cu128 | 原π0.5、同进程特征/梯度接入、LIBERO |
| `envs/robotwin` | 3.10.21 / 2.7.1+cu128 | RoboTwin模拟器和CuRobo |

全部路径以上述部署根`/workspace/owac/rawvla/`为前缀。本地主CPU锁没有安装到远端，原`/venv/main`未改动。实际依赖导出保存在`records/*-new-server-freeze.txt`；安装/恢复脚本见[部署目录](../configs/backends/n01-v12-runtime/)。

两张卡已同时执行完整DBSR（64/9/512）前向、反向及一次真实optimizer更新，以及完整ResNet34 SEA-RAFT的128×128前向/反向：GPU0用encoder环境、GPU1用policy环境。输入是随机张量，只证明CUDA内核和环境可工作，不是任务训练或算法收益。

LIBERO官方JAX权重已重新下载并用原固定OpenPI转换函数恢复为StarVLA格式，活跃参数无缺失/额外键/形状不匹配；RoboTwin原始权重已下载，统计由作者exporter导出，继续使用原14维顺序与MEAN_STD。tokenizer、统计、检查点均放在完整提示词列出的原约定路径。

完整RawVLA-Bench及模拟器已部署，最终核查如下；[机器快照](prompts/n01_v1_2/remote_snapshot.json)记录实际时间、环境和原始收据位置。

8080由Jupyter占用，保留。N01监控要求由P3执行者实现，拟用回环18080，当前不宣称研究实时页已上线。监控转发使用`ssh -N -p 34302 root@154.59.156.21 -L 8080:127.0.0.1:18080`；若本机8080已用于Jupyter，改本机18080。HF凭据仅保存到服务器私有token存储，未写入源码、提示词或本报告。

默认GPU0排RAW、GPU1排RGB；每卡Base/N01依次跑，训练与验证均可双卡。实际前台命令可用`bash configs/backends/n01-v12-runtime/run-on-gpu.sh raw|rgb <命令>`包装，然后由各自supervisor管理。验证先划分一份总100条开发清单，各worker不重复领取，独立端口/输出，合并后去重统计；不是每卡100次。完整P3训练/评估/实时监控入口尚待阶段实现，本次不制造空壳CLI冒充可运行训练器。

旧服务器P2正式运动checkpoint不在本机小型归档中，未恢复；现有源码、配置、审计限制和数字记录仍可用。不要拿本机smoke checkpoint替代，不重跑旧P2三种子作为P3前置。

## 最终核查

- 固定版本RawVLA-Bench：**4,810个文件、713,689,392,570 bytes（约713.7GB）**，按原文件清单核对存在和字节大小，缺失0、大小不匹配0；LIBERO RAW 1,771个NPZ，RoboTwin RAW 632个NPZ。未重复全量计算文件哈希；训练/验证样本已实际解压读取。
- 两套真实数据→完整DBSR→512→2048 bridge→固定π0.5链路均已执行一次optimizer更新。LIBERO编码器/bridge梯度范数约6.56/2.63，参数实际变化；RoboTwin约925.00/478.75，参数实际变化。后端梯度均为0，原图像塔未调用，原10步动作采样得到有限输出。这里只说明接口和梯度可工作，数值大小不作为性能分数。
- LIBERO创建真实场景并完成两步模拟器控制、两路128×128图像渲染；RoboTwin初始化`grab_roller`、CuRobo规划器与三路240×320 RAW/Default ISP观测，读取14维真实关节状态。RoboTwin采用既有OptiX设置（32 spp、path depth8）；未跑策略成功率或100次闭环。
- SAPIEN在`CUDA_VISIBLE_DEVICES=1`下选择进程CUDA0、PCI `0000:9a:00.0`，与物理GPU1一致；双卡调度注意作者`eval.sh`会按自己的`gpu_id`重新设置可见卡。
- 核查后两卡无常驻模型占用，存储约剩784GiB。下载/安装作业已正常结束，并关闭它们的自动启动，避免实例重启重复安装。Jupyter、管理服务保留。
- 当前源码已包含v1.2方案、三份完整提示词和实际未提交修改；新机可直接从`/workspace/owac/rawvla/owac`继续开发。完整P3模型、正式训练、实时研究监控及开发闭环仍由P3阶段实施；旧P2正式运动权重未恢复。

本机`make check`与部署shell语法检查通过；研究控制台测试10项通过，实际P3复制内容已核对为新服务器、双GPU、单种子及每轮总100次。完整Markdown与HTML内容同步。以上检查不构成N01 v1.2能力验收。

# RAW 输入教师取得与限定复现提案

2026-10-20，Sol 研究判断。**状态：提案，未获训练／回放／闭环授权。当前教师 checkpoint 未取得；本轮新增优化0、策略闭环0。** 本文不新建采用决议，也不启动学生训练。用户允许教师与学生使用有依据的因果时序，不固定单帧。

## 已完成的取得工作

已下载并可加载核验的，是此前 OWAC 成功率评测所用的同一个 π0.5 后端：
`outputs/checkpoints/raw-distillation/pi05_libero_starvla/bfloat16/model.safetensors`，SHA256 `68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8`。加载审计和统计量／tokenizer 见 [backend-recovery](../../research/experiments/raw_stem_siglip_20261020/backend-recovery.json)。历史 task6 Normal 50/50 是 **Default ISP＋π0.5**，不能认作 Neural ISP 教师能力。

本次补查作者仓库6个公开分支、main全部32个历史提交、release/tag、项目页和作者 Hugging Face 主页及数据集顶层附件，未找到训练好的 RawVLA 前端下载。源码完整 archive 已下载2811790 bytes，SHA256 `57bc54e8a695ae07c2ffee9ebb2549e9efbb48536a97df14c5c8f63c52f3e012`，main `ba9c56b04da2c4931b70d8844710f5928d43775b`；StarVLA 子模块 `57d508cb180d0510ac9060358009c0f6ffaac420`。公开 runner 期待 `steps_2000_raw_frontend_pytorch_model.pt`，路径字符串不代表文件已经公开。HF 页面访问成功，宿主 API／镜像访问失败，不能从网络错误推断不存在；未穷尽数据集中每个 NPZ 目录。结论仅为**已检查公开入口未找到**，不能证明私人或未索引渠道不存在。完整收据见 [公开搜索v2](../../research/experiments/raw_stem_siglip_20261020/teacher-public-search-v2.json)。未向作者发消息。

## 建议采用的复现范围

建议做“**作者公开实现的 task6 Normal 限定复现／教师开发**”，不声称重现论文全任务指标。目标链为 RAW→作者 RawVLA Neural ISP→原 RGB 预处理→同一 π0.5。训练只更新 RawVLA 前端，SigLIP、投影、语言及动作后端全部冻结并保留对图像的反向梯度。真实公开默认容量 CPU 实例有829172个参数、83个state_dict张量，FP32张量3316700 bytes；这是结构计数，非学习证据。

我选择**公开代码配方**，因为它与公开模型类、适配器、当前已登记的作者选型文件一致；不无界搜索论文／源码两套超参数。论文与源码的差异明确保留：

| 项目 | 拟采用的公开实现 | 论文D1/D3 |
| --- | --- | --- |
| AdamW weight decay | 1e-8 | π0.5 1e-10 |
| 原生动作损失系数 | 0.15 | 0.10 |
| brightness L1系数 | 0.01 | 0.001 |
| chroma envelope系数 | 0.001 | 0.001 |

其余为 seed101、有效batch8（microbatch1×累积8），LR1e-4、betas(.9,.95)、eps1e-8、warmup100、cosine至1e-6、clip1，**累计最多2000更新**。先检查前100更新，包含在2000内；出现冻结权重变化、非有限梯度、数据错位即停。无自动加训、无后端解冻。原生32维flow-matching目标沿用策略语义，不以7维离线指标替代训练目标；报告真实7维与补齐维度分开。

保持作者 burst6、recurrent6、endpoint stride5，窗口内state/theta不detach；每相机与episode独立状态，reset清空，左补真实首帧并明确标记。对应原始环境步最早为t−30，最后burst为t−5…t；这不是36次独立观测。首个episode窗口使用实际可用历史，无未来帧。训练六次递归展开，部署按实际重规划时刻更新，并核对训练／部署历史范围。action horizon10、flow steps10、replan steps5沿用现有 OWAC 已核查协议；这些值不能互相替代，也不把论文比较组的horizon8说成本轮horizon10的依据。

## 数据、选型和教师有效性

仍坚持原定任务 `libero_goal/6`（默认task_order_index0），指令 `put the cream cheese in the bowl`；LIBERO RGB8逆处理得到三通道pseudo-RAW，不能称原生CFA。先取得50条真实源示范，源级train40/validation5/test5，同源光照／重采样一并归组；本地现有50个初态不是50条示范。公共数据按来源抽样光照，不能假定task6有50条Normal，不能静默混用其他光照或缩小样本数。

若作者公开成品不足50条Normal，先下载该任务的原始专家示范，再在固定生成环境按作者协议回放到Normal，保存同源RAW／RGB／状态／动作及来源哈希；**这属于需明确授权的示范回放**。回放失败保留分母，不靠重复同一示范凑50；不足则停止并修订提案。公共整库页面显示约714 GB，本提案只取该任务，禁止整库下载。当前尚无该任务实际字节清单，因此不声称数据已备好或保证50 GiB足够。

checkpoint保存0／100／500／1000／2000及提前停止点；按验证集完整去噪前5步真实7维MAE最小、同分取更早更新选一个，报告首步／前5／全10步、平移／旋转／夹爪／补齐维度、特征与图像指标。成像MAE只是诊断；纯RAW输入教师的闭环才决定能否作为有效教师。测试集不参与选型。

建议另外授权最多50个**策略闭环episode**：最多10个固定验证初态诊断；选定checkpoint后，同20个新测试初态分别运行Default参照和RawVLA教师各一次（40场）。先冻结初态清单并核查与训练／验证示范的物理初态泄漏；没有20个合格初态则停，不降低分母。报告完整分母、视频、置信区间、配对退步与恢复、失败阶段。开发筛查要求参照和教师均≥16/20且教师少不超过2次；这不是统计非劣证明。教师失败则诊断，不启动学生蒸馏。示范回放50条与策略闭环50场分账，不混称100个测试episode。

## 拟申请的预算与启动门槛

| 范围 | 提议硬上限（尚未授权，不是耗时预测） |
| --- | --- |
| 数据 | 仅该任务，新增磁盘总量≤50 GiB；含源示范、配对数据、缓存、checkpoint、视频 |
| 教师训练 | 一条seed101运行，前100诊断计入最多2000更新／16000 anchor呈现 |
| GPU | 本机一张RTX5090，教师阶段预检、训练及验证累计≤12 GPU小时；不租云、不借历史远端 |
| 示范生成 | 如缺Normal，仅最多50条原始专家示范回放；不采新机器人动作 |
| 验证 | 最多10验证＋20参照测试＋20教师测试，共50策略episode |

上述为我提出的预算封顶，**不是已实测资源画像或完成保证**。既有单相机合成后端探针峰值约11.1 GB，不能外推六次递归、双相机真实训练。取得真实数据后，先在实际双相机／burst／递归设置测零更新前反向峰值、吞吐与输出磁盘；预测超出12 GPU小时／50 GiB或不适配显存时，保存证据并停，不削历史、不降数据标准、不自动扩预算。训练和回放各自受授权门控。

工程启动前还需：任务／Normal筛选manifest；真实数据检查；显式mask与相机顺序；原生动作标签归一化；冻结参数及六步BPTT梯度核验；一致的教师训练和独立RAW推理入口；记录完整optimizer／scheduler／RNG／sampler。作者trainer只保存前端state_dict，**不满足完整恢复要求**，需增加可审阅薄层，交Harness实现、Sol验收，再开始授权训练。正文公开CLI仅作来源参考，不能绕过这些门槛直接启动。

实际官方入口是 `starVLA/training/train_starvla.py --config_yaml .../configs/libero/pi05.yaml`（README给Qwen示例、映射表列PI05）。当前未运行；其默认cache_root含全部任务／光照且不自带本提案拆分、时间／磁盘门控与完整恢复，故不把替换参数后的命令伪称已验收的OWAC任务启动器。工程完成后登记准确命令和源码diff。

## 本轮交付与下一步

已取得同源后端和完整作者源码；**未取得已训练Neural ISP，未训练任何参数，控制能力未确认恢复**。Harness V41-Flash Max进行了两轮配方清点，Sol独立核对公开原始文件、实际配置合并、权重来源及结构参数量；见 [配方清点](../../research/experiments/raw_stem_siglip_20261020/teacher-recipe-inventory.md)。旧不完整snapshot产生的“base配置缺失”判断已经撤回。

下一步由用户审阅本提案，明确是否授权上述限定教师开发预算、必要示范回放与仿真验证。授权后仍先通过实际资产和零更新门槛；不自动授权学生B/C/D训练、弱光、四任务或实机。

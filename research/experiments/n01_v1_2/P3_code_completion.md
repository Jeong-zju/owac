# N01 v1.2 P3：代码补全，训练保持暂停

**历史阶段记录。** 用户随后明确授权接入训练；最新运行状态、加速实测及监控见 [P3_training_launch.md](P3_training_launch.md)。本文件和JSON收据保留授权前的0次更新事实，不作为当前暂停状态。

2026-10-17。工作树 `lzh/n01-evidence-integration`，基于 `d3dec90` 加既有未提交交付；未提交或推送。本次遵照“做完代码层面的所有工作，训练暂时不要开始”，优化器更新授权为 **0**。这是完整实现的代码验收，不是训练、闭环效果或 P3 研究目标完成。

本记录取代旧交接中“代码已完整，只差墙钟训练”的判断。旧训练检查点、日志、失败闭环和报告原样保留，不迁移进新实现的训练曲线。新版本改变运动、支持、输入几何及后端适配，不能拿旧检查点无条件续训。

## 1. 现在实际运行的链路

`原生 RAW + 实际身份 → 同域观测算子 → 32 个唯一来源 → 当前参考到历史的 SEA-RAFT pull → 沿对应位置的局部 GLS / 有界 BOCPD 重放 → 融合前的来源屏蔽 → 完整 DBSR → 每相机 256×512 token → 512→2048 桥 → π0.5 原生动作后端`。

RGB 臂在原生 RAW 分辨率上先执行同一冻结 ISP，再按相同几何规则进入自己的网络。它不复用 RAW 臂的 flow、方差、支持或特征。这里的 RAW 仍是作者三通道 pseudo-RAW / HDR 派生数据，不能改称原生 CFA。

| 原缺口 | 当前实现与边界 |
| --- | --- |
| 主链没有运动估计 | `learned_motion.py` 注册完整 SEA-RAFT，直接使用可微网络；旧 P2 推理包装的 `no_grad` 不再截断动作梯度。当前配置明确随机初始化；不冒充已训练对应。BN 统计固定。 |
| 空间统计只是池化残差 | `tensor_gls.py` 求解局部常数/仿射 GLS，选择有限多尺度候选；对零创新的兼容分数同时保留系数与拟合残差，不用拟合亮度漂移消除真实变化。 |
| 窗口外后验影响所谓32帧状态 | `window_support.py` 每次由固定初值重放保留窗口，沿当前参考图上的实际映射位置更新作者 BOCPD；不携带已淘汰来源后验。 |
| 支持在卷积后屏蔽会漏信息 | 反向展开所有正权双线性采样 tap，屏蔽源域输入后重新编码，再在参考域限制融合。保守区域屏蔽会损失可用内容，不声称任意查询像素级最优依赖。 |
| 删源命令只改内容、丢其他相机 | `source_replay.py` 在已限定窗口删完整来源、重算运动/支持/参考/内容；不补旧帧。`intervene --mode delete|fixed|mask` 分别记录，全相机 token 共同送入后端，固定噪声比较去噪动作预测。没有执行仿真动作。 |
| 训练裁左上角、服务先 resize | `observation.py` 统一训练/服务/重算，验证原生尺寸；默认不裁剪、不 resize。完整配置 history_stride=1，使用每个可用到达；这是对作者默认 LIBERO stride=5 的显式修改。离线去掉 no-op 后的时间稀疏与在线物理时钟仍需披露。 |
| 原生32帧反向超显存 | 完整相机级activation checkpoint覆盖编码、对齐及成熟融合，诊断只保留固定通道均值；不缩短历史、不降分辨率、不替换网络。独立对比原计算的输出、权重均值及参数梯度。首次RoboTwin约90 GiB后OOM的日志保留，不把先前32帧前向当反向证据。 |
| 后端永久冻结 | `training_plan.py`：编码器/桥预热2000步，之后四臂一致开放 `action_head.`，不含原RGB塔；后端适配参数随检查点保存、服务恢复，优化器分组和LR随阶段一致。计划50k是预算，不是充分性结论。 |
| 固定验证仅看首轨迹 | 固定样本跨数据集取等距子集，flow噪声固定；另有每检查点可复现的随机验证。采样轨迹、条件、独立窗口覆盖及重复率可恢复、可报告。 |
| 服务遗漏两次查询间到达 | LIBERO 在生成的作者客户端副本上应用精确锚点补丁，传原生32帧及真实 episode/frame IDs；RoboTwin 插件每控制步记录三相机，保持50步动作节奏。重发与padding按ID去重。 |
| 多相机半提交、缓存跨episode | 相机状态/episode/快照/动作缓存原子提交，错误回滚；reset清除旧快照；全无效输入显式不可用；删空某相机返回全false有效性，其他相机仍可读出。逐相机诊断保留，记录队列/搬运/运动/支持/内容/后端/提交/总耗时分位数。不是硬实时承诺。 |
| demo序号猜成初态序号 | `n01_v12_identity.py` 联合匹配动作和状态内容，再比对初态内容；重复动作、歧义、缺资产、split冲突均拒绝猜测。`--held-out-only` 强制内容证明。 |
| 阈值没有可执行校准入口 | `n01_v12_calibrate.py` 对真实标注的episode/camera/source/region比较支持、噪声及风险配置，在calibration与validation分别报状态计数；不从无标签分位数伪造误报率，不自动选择阈值或读取test。 |
| 图形没有真实motion、会混相机 | 每相机独立捕获融合/支持，输出真实pull箭头、warp及残差；图明确不是GT flow error。服务时间轴记录有坐标的稀疏pull点和所有相机诊断。 |

有效配置在 `configs/encoders/n01_full/v1_2/complete/`，两平台四臂共8份，seed101；Base臂仍遵照原决定推迟，未启动任何臂。LIBERO原生256×256，RoboTwin240×320。完整骨干保持DBSR宽64/9块/512特征、完整SEA-RAFT resnet34/4次迭代。

## 2. 尚未生成的研究证据

这些不是隐去的能力结论，也不是待写的训练代码：新模型未训练；RGB/RoboTwin域方差及风险/阈值仍是显式声明、尚未拟合的量；自然运动准确性、支持收益、四臂增量、吞吐分布及完整训练预算必须实测。随机对应可能把全部历史判为不可用，不能压低风险掩盖它。已知空间变换辅助项提供合法的可微适配入口，权重可在配置关掉；它不是自然flow真值或RGB教师。

原始LIBERO HDF5示范数据当前未恢复，因此旧闭环的内容级初态映射仍未实证。证明工具已实现；缺数据时不生成“留出成立”。RoboTwin作者在线 `raw_rgb10` 是uint16，除1023后进入服务；离线HDR float32缓存与在线RAW10量化的差异保留，不冒充完全相同采样噪声。

## 3. 禁止训练与运行隔离

- 全局锁：`configs/encoders/n01_full/v1_2/TRAINING_HOLD.json`。训练器和队列都检查它，不能用 `--diagnostic` 绕过。
- 旧根 `.../runs/v1-2-p3-main/P3_AUDIT_HOLD.json` 保留；旧两个supervisor程序保持STOPPED、autostart=false。
- 新队列只读取 `complete/`，默认输出新根 `.../runs/v1-2-p3-complete`，原生分辨率、计划50k、有效batch=4微批累积、每臂自己的断点；不会装载旧主运行的权重。
- `owac-n01-v12-complete.supervisor.conf` 是未注册的休眠模板，autostart=false；没有启动或安排定时训练。
- `verify` 入口内部禁止标准torch优化器step；只做前向或求梯度，不保存适配后的权重。

## 4. 可复核入口（不启动训练）

在本工作树：

```bash
uv sync --locked --extra encoder --extra visualization --extra identity
make check
uv run --locked pytest --no-optimizer-updates -q
```

服务器上（路径按真实部署；使用 `/tmp` 避免旧根目录副本遮蔽 `src`）：

```bash
N01_SOURCE=/workspace/owac/rawvla/owac
N01_DEPLOY=/workspace/owac/rawvla
N01_PYTHON="$N01_DEPLOY/envs/starvla/bin/python"
cd /tmp
bash "$N01_SOURCE/configs/backends/n01-v12-runtime/run-on-gpu.sh" raw \
  "$N01_PYTHON" -m owac.training.n01_v12 verify \
  --repo-root "$N01_SOURCE" --deploy-root "$N01_DEPLOY" \
  --config "$N01_SOURCE/configs/encoders/n01_full/v1_2/complete/libero-raw_n01-seed101.json" \
  --device cuda:0 --full-history --forward-only --output /tmp/n01-forward32.json
```

有限梯度检查可将 `--forward-only` 换成 `--history-frames 2 --backend-phase warmup --check-motion-auxiliary`；仍然是0次优化器更新。32帧反向另有独立命令与回执；两帧检查只用于分阶段梯度和辅助项拆解。真实服务器回执位于 `.../runs/v1-2-p3-code-completion-20261017/`，小型摘要见本目录JSON收据。

同源ISP检查必须指向两个作者checkout的共同父目录：

```bash
cd /tmp
PYTHONPATH=/workspace/owac/rawvla/owac/src OWAC_RAWVLA_SOURCE=/workspace/owac/rawvla/source \
  /workspace/owac/rawvla/envs/encoder/bin/python -m pytest \
  /workspace/owac/rawvla/owac/tests/integration/test_n01_v12_frozen_isp.py -q
```

校准工具 `python -m owac.evaluation.n01_v12_calibrate --help`、内容证明工具 `python -m owac.evaluation.n01_v12_identity --help`、三类干预 `python -m owac.evaluation.n01_v12 intervene --help` 均为实际命令。真实评估/案例需要新配置匹配的 `--checkpoint`；无权重时必须显式 `--allow-untrained`，输出只作诊断。RoboTwin使用 `robotwin-native.yml` 的插件；服务须使用规范frames接口，LIBERO使用 `--openpi-adapter`。

## 5. 验证收据与完整改动

本地/远端的最终计数、真实张量与梯度、暂停状态以及证据限制统一写入 `P3_code_completion_receipt.json`。测试中的小网络仅作数值/依赖反例，完整网络的证据以真实服务器回执为准。测试通过、真实前向或反向均不能替代训练结果。

最终 `P3_worktree_diff.txt` 包含当前tracked diff及未跟踪源码/配置/报告内容（排除自身），供完整审阅；原DeepSeek差异已保存在被忽略的本地归档，不覆盖原训练数据。

### 本次最终实测

- `make check` 通过；本地 **853 passed / 25 skipped**。25项中19项在第一次优化器step之前跳过，6项因本地CUDA/作者源码不可用跳过；不把跳过项记成行为已验证。
- 远端定向 **31 passed / 0 skipped**，包含作者ISP逐元素交叉检查4项。
- 实际完整网络：LIBERO两相机32个唯一帧、256×256，原生动作loss反向约8.21秒/峰值47.71 GiB；RoboTwin三相机32个唯一帧、240×320，开放动作后端后反向约13.72秒/56.11 GiB。一次有限检查，不含optimizer状态，不是训练吞吐分布。
- 编码器/桥实际动作梯度非零；RoboTwin匹配后端范围梯度非零。未训练对应全部被拒时，native action到motion梯度为0；真实同域帧的已知变换辅助梯度非零，分别记录，不把后者写成前者。
- 本轮优化器更新 **0**。最终两GPU空闲，旧两个supervisor程序STOPPED，autostart=false、autorestart=false，全局及原运行hold都在；未注册新训练模板。

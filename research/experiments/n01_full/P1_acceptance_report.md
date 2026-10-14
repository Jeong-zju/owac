# N01 v1.0 P1 独立验收与修补

2026-10-14；工作树 `/home/jeong/zeno/owac/worktrees/n01-evidence-integration`，分支 `lzh/n01-evidence-integration`。审查对象为 `fd92383385fd17103e27a68c345fc7af15822ffe`（DeepSeek 的 P1 交付），要求来自 `docs/prompts/n01_v1_0/P1_ANALYTIC_REFERENCE.md`、共同契约、早期策略预检及完整方案 §4/5.1/5.2/5.5/6.1。

## 结论

**原交付未满足 P1，必须修补。** 原有 34 项 P1 测试全部通过，但独立添加的首批 14 项反例在原实现上全部失败。错误涉及估计方差、反事实来源隔离和真实窗口行为，不能以原有 `473 passed` 或 `analytic 9/9` 接受实现。原报告与原收据保持原样，作为历史交付记录；本报告与 `P1_acceptance/receipt.json` 是修补后验收入口。

**修补后通过 P1 验收**：本地 495 项、服务器 147 项测试通过，解析检查两端各 9/9，远程 verify 22/22、源包 153/153 文件校验一致，两平台真实数据计数和策略预检通过。代码修补提交 `2d82bbb`。P1 只交付解析参考与真实桥接连通性；未进入 P2，未开始长训练，未建立 N01 机制增量或闭环能力。

## 要求、反例与修补

| P1 要求 | 原交付的可复现问题 | 修补与验收依据 |
| --- | --- | --- |
| 未知相关性下的保守方差界 | 两点仿射拟合在首个采样位置应等于原观测，原实现却给出方差 **0**；二观测平均在完全相关单位噪声下真实方差 1，插值支路只报 **0.5** | 保留查询到原始 RAW 的完整权重，使用 `(Σ|a_i|σ_i)²`；矩阵接口返回有推导的较松对角包络。含 30 个固定随机有符号仿射查询和插值反例 |
| 秩亏不能产生虚假 confidence | 未知相关分支给不可辨识方向零方差；已知噪声支路也在不可辨识查询返回有限均值，使快照 mask 为 true | 用 SVD 行空间检查实际查询，不可辨识查询输出 NaN；无先验秩亏不给参数协方差 |
| 噪声、先验及坐标版本一致 | 求解忽略域声明的 noise/prior 绑定；独立 oracle 在观测秩亏时丢掉先验；仿射先验没有指定物理坐标 | 检查域绑定；oracle 保留后验；仿射先验显式声明变换，撤回后对均值/协方差同时换基；与物理坐标 NumPy 参考比较 |
| 数值条件与不确定性准确 | 正规矩阵使条件数平方；极大/极小噪声单位导致绝对秩阈值错误；多相位 dense 噪声的跨相位协方差被丢弃 | 白化 least-squares + QR 三角求解协方差；相对特征值/奇异值阈值；保留跨相位误差块。跨相位相关加先验需要联合后验，当前明确拒绝 |
| 原始数据不可变、统一来源查询 | `frames/get_frame` 暴露内部 tensor，可直接改写 RAW；删除与内容 overlay 组合后批量查询可“复活”来源；无 batch 维的有效 mask 被拒绝 | 公共帧接口返回副本；删除先于 overlay；各接口一致处理缺失内容；修复 3D mask 适配。非法采样坐标形状明确报错 |
| 完整反事实删除后再做决策 | `without` 只过滤 SourceKey 查询，帧级 API 仍暴露删除内容；把被删值从 1 改为 100，支持决策使最终输出从 **9 变 5** | 在支持/映射/噪声回调之前物理清除视图中的删除内容，标 missing；原账本保持不变。内容敏感支持规则现输出始终为 9；新增 mapping/noise builder 重算入口 |
| 32 帧精确窗口和越界重建 | 34 帧输入仍返回 34 帧账本并拟合全部 68 个采样；越界只变更 claim/history_loss 标签 | 支持构造前裁掉过期 RAW；34 帧例只保留 32 帧/64 个采样，均值由 18 正确变为 19；8 帧边界重建只使用合法 16 个采样，均值为 7 |
| 内容替换必须有可证伪检验 | 空 replacement 或空查询可直接给 unchanged | 要求非空查询和每个目标的实际替换；比较均值、方差、有效性；保留原泄漏负对照 |
| 来源规范排序 | 大于 2^53 的相邻整数 frame ID 经 float 排序后可能依赖输入顺序 | 全整数排序，固定相邻大整数反例 |
| 真实策略预检证明 encoder/bridge 各自连通 | 原“encoder gradient”合并了桥梯度；validation 只看 stem；没有实际输入和完整参数哈希 | encoder/bridge 分别检查有限非零梯度与逐步参数更新；validation 全参数/状态哈希相等；冻结后端前后哈希一致；记录实际桥输出 shape/mask/dtype 和输入哈希 |

新测试文件 `tests/encoders/n01_full/test_p1_acceptance.py` 共 22 项（含参数化）。首批 14 项的原版失败日志在 `P1_acceptance/adversarial-before.txt`；修补后全部通过，既有 P1 回归也通过。额外实际重算记录见 `version-recomputation.json`：内容版本替换使均值 2→5，映射版本使读出 1→3，玩具参数版本变化后显式失效 `hidden` 的闭包重算与全算差为 0；这不声称自动神经缓存失效已经实现。原 `test_ledger.py` 的两处断言随语义修正：派生视图的帧 API 应读到替换内容；错误坐标声明应报错，不能退回索引坐标。未扩大任何数值容差。

## 实际验证与来源

本地执行：

```bash
make check
uv run --locked pytest -q
uv run --locked pytest -q tests/encoders/n01_full/test_p1_acceptance.py
uv run --locked python -m owac.evaluation.n01_full analytic --output outputs/n01-p1-acceptance/analytic-v5
uv run --locked python -m owac.runtime.n01_full_preflight bundle --output outputs/n01-p1-acceptance/bundle-v5
```

本地全仓 **495 passed**，新增反例 **22 passed**，`analytic` **9/9**，`make check` 通过。旧 N01 的一条 tensor→scalar warning 保留。Python 3.12.12 / torch 2.9.1+cpu / NumPy 2.5.1。旧一维/支持/状态与共享桥测试包含在全仓回归中。

远程最终源码为 `/workspace/owac/rawvla/research/n01-v1/runs/p1-independent-20261014/source-v5/`；打包清单 153 个文件，包含未跟踪的新测试。源码基点是 fd92383 加清单中的实际文件内容，**不能把基点 SHA 当作修补后的源码 SHA**。清单记录 dirty diff hash；最终 153/153 文件与本地工作树逐一核验一致，其中 129 个受版本管理文件与修补提交 `2d82bbbfe4e556c91c6a55c2b9c500ea5f3183ce` 的 Git blob 哈希一致，另 24 个是保留在 ignored outputs 的固定上游源码/许可。文档/数字收据不属于执行包，随后单独提交。

在源码目录设 `PYTHONPATH=$PWD/src:/workspace/owac/rawvla/source/starVLA`、`OMP_NUM_THREADS=4`，依次用 `/workspace/owac/rawvla/envs/encoder/bin/python` 执行 `n01_full_preflight verify`、阶段 pytest、`n01_full analytic`、两平台 ledger 检查；用固定 `/workspace/owac/rawvla/envs/starvla/bin/python` 执行两平台 `n01_full_policy_preflight --mode contract --steps 2`。窗口读取与策略预检均披露 `crop=32`；两次训练更新来自不同真实 episode，validation 仅前向。远程没有修改冻结基线、权重或已有服务。

保留一次本轮执行失误：最初使用共同提示词中的草图路径 `research/n01-v1/envs/encoder/bin/python`，退出 127；之后依据 P0 验收报告切换到实际 `rawvla/envs/encoder/bin/python`。原失败日志留在远程 run，未将其计作测试通过。中间快照与最终 source-v5 分开保留。source、source-v4、source-v5 各执行两平台各 2 次开发更新，本轮共 12 次短连通性更新；每次重新初始化，最终数字只引用 source-v5。

## 理论交付与交接边界

`research/theory/n01_support_contract_v1.md` §12 补充固定设计 T1 集合目标、T2 GLS/先验/保守界及 T5 完整依赖闭包的推导、条件和代码对应。修正原错误的 PSD 外积界。数值测试不是一般程序正确性的证明，经典固定设计结论不能直接用于自适应选域或学习 flow。

- `implementation_status`：`PASS_P1_AFTER_REPAIR`。
- `training_status=NOT_STARTED`：仅本阶段规定的短策略连通性更新。
- `effect_status=NOT_APPLICABLE`：无 N01 效果、训练充分性或闭环结论。
- 当前窗口采用“过期数值信息丢弃、保留身份/历史损失”策略，不承诺长期数值摘要。离线全重放仍会消费完整事件归档；返回状态有界不等于已测得流式峰值内存有界。
- 数据依赖 correspondence/noise 需要通过 builder 在清除后的 ledger 上重算；直接参数只适用于预先固定的设计。回调闭包不得捕获原始未清除的内容。当前未实现 P2 flow、P4 神经 RAW 域重编码或 P5 递归缓存。
- `raw_adapter`、全分辨率策略运行、真实 CFA 与 P7/P8 闭环仍未执行。本轮没有自动推进 P2。

用户原有 `docs/OWAC_academic_research_proposal_2026-10-09.md` 未触碰，推送未执行。

## 最终服务器数字

| 平台 | 唯一帧登记/总到达 | 重放新增采样 | encoder 梯度范数（两批） | bridge 梯度范数（两批） | 实际视觉前缀 | validation 漂移 / 后端哈希 |
| --- | --- | --- | --- | --- | --- | --- |
| libero | 28/33 | 0 | 368.8384 / 1596.5495 | 164.0846 / 833.1912 | `[1, 512, 2048]` | 0.0 / 前后一致 |
| robotwin2 | 42/49 | 0 | 57.9084 / 132.3801 | 32.5413 / 58.0548 | `[1, 768, 2048]` | 0.0 / 前后一致 |

每个 train batch 的 encoder 与 bridge 实际更新量均大于零；完整逐步更新量、真实输入哈希、参数哈希、不同 episode 身份、冻结权重/tokenizer 哈希均见 `policy-contract-remote.json`。服务器 pytest 的 warning 仅来自执行包未附带仓库 pytest mark 注册配置，不影响 147 项实际测试结果。

# N02 v1.1 P1 阶段报告：RAW 观测与绝对拟合（首轮，seed 1103）

状态：**进行中**（implementation=pass / numerical=pass / behavior=进行中 / closed_loop=not_run）。本文件在验收完成后由 dispatcher 补全第 6、8 节；未完成项一律保持 `not_run`/`pending`，不写成结论。

## 1. 本轮原要求

依据 [P1 提示词](../../docs/prompts/n02_v1_1/P1_OBSERVATION.md)、[共同契约](../../docs/prompts/n02_v1_1/00_SHARED_CONTRACT.md) 与 [v1.1 方案](../../docs/plans/N02_full_development_plan_2026-10-13.md)：

- 实现 `observation.py`、`sensor.py`、`factors.py`、`checks.py`、最小 `model.py`；训练入口 `training/n02_full.py` 支持 `--stage p1-pretrain|p1-measurement`；评测 `evaluation/n02_full/p1.py`；配置 `p1_{pretrain,measurement,eval}.json`。
- Base（depths `[3,3,27,3]`、dims `[128,256,512,1024]`）RAW 掩码预训练（FCMAE mask_ratio .6 / patch32 / decoder 512×1 / norm_pix_loss=false），输出 stride8/16 的 512 维特征并携带 sensor 坐标/相机/validity；RAW3 与 CFA4 分模式。
- 固定候选条件测量模型：`encode/predict/score`、公共候选字典（正确/错误/全错）、fit 与 prequential_check 分离、η 干扰（共享 log 曝光 + 每帧二维相机平移 + 噪声尺度）只从历史估计。
- 绝对检查三态（compatible / insufficient_information / model_inadequate）与固定窗口经验阈值；双亮点不得被全图均值抹平。
- 训练：预训练 AdamW lr1.5e-4 / wd.05 / betas(.9,.95) / 5% warmup / cosine / 有效 batch 256 帧；测量阶段有效 batch 32 条×8 帧、Base lr 3e-5 / 新头 3e-4 / wd .01、loss 1 : 1 : 0.1；dev 网格先冻结判据再选型。
- 行为验收（每类 ≥1000 独立场景 ×3 seed）：线索互换两方向 ≥90%、无信息漂移 ≤.05、95% 错先验两方向纠正 ≥80%、正常误报 ≤5% 且结构不足检出 ≥90%；oracle 先证明任务可纠正。
- 完成后停在 P1 交接点，交接 E1。

## 2. 实际实现与文件

全部改动**未提交**（用户未授权 commit）。本地与远端 sha256 逐字一致（`--exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache'` 同步后在远端运行）。

| 文件 | sha256（前 16） | 责任 |
| --- | --- | --- |
| `src/owac/frameworks/n02_full/observation.py` | `660998df54ac741f` | T1：Base 多尺度 RAW 观测、坐标映射、FCMAE 组件 |
| `src/owac/frameworks/n02_full/sensor.py` | `6d590d64973fd2f2` | T2：测量分布/正参数化方差/量化与饱和/η |
| `src/owac/frameworks/n02_full/checks.py` | `b4c99a4db7fa2738` | T2：绝对检查与经验阈值 |
| `src/owac/frameworks/n02_full/factors.py` | `85abb00fe2eaced4` | T3：`ObservationModel`、候选字典、条件 decoder、单位仿射 |
| `src/owac/frameworks/n02_full/model.py` | `0479edb469b70e74` | T3：最小 `step_bundle`（目标身份/一次 likelihood/锁定检查） |
| `src/owac/frameworks/n02_full/config.py` | `e790ea6a703a4760` | T4：P1 配置扩展（training/evaluation） |
| `src/owac/training/n02_full.py` | `75537edf3b00268d` | T4：两阶段训练入口、精度检查、消融臂、续训 |
| `src/owac/evaluation/n02_full/p1.py` | `0af7ba8690c955ca` | T4：oracle/behavior/ablation/report |
| `configs/encoders/n02_full/p1_pretrain.json` | `b185d2ea78f1…` | 预训练配置（lr 1.5e-4、mask .6、有效 batch 256 帧） |
| `configs/encoders/n02_full/p1_measurement.json` | `5dd201bf7502…` | 测量配置（32×8、loss 1:1:0.1、逐相机 sensor 表） |
| `configs/encoders/n02_full/p1_eval.json` | `c574ac26d87c…` | 验收阈值与 7 个消融臂定义 |
| 测试：`tests/encoders/n02_full/test_p1_{observation,sensor,checks,factors,model,training,eval}.py` | 见 `results/` 清单 | 各 owner 自带回归 |

接口冻结件：[`docs/prompts/n02_v1_1/p1_interface_freeze.md`](../../docs/prompts/n02_v1_1/p1_interface_freeze.md)（含 3.1+/3.2+/3.5+/3.7/3.8/3.9 六处 dispatcher 裁定）。

## 3. 复用来源及差异

- **复用**：`components/convnextv2.py`、`components/fcmae.py`（P0 经 17 项 parity 验收的 ConvNeXt V2 RAW 适配与 FCMAE）、`state.py` 契约 dataclass、`data.py` 的 `ManifestWindowDataset`（v2 四分区 + 冻结 anchor，禁止二次分区）、`synthetic.py` 受控生成器；P0 运行时锁 `configs/backends/n02-full-runtime/requirements.lock.txt`。
- **差异**：P1 只在 `config.py` 追加 `training`/`evaluation` 两段（P0 配置测试保持全绿）；`sensor.py` 新增 `unit_semantics()`/`score_diagnostics()`；`factors.py` 新增 `UnitAffine`/`world_scores`/逐点相机索引；`observation.py` 新增 `FrameTransform`/`_frame_layout`。均不回改 P0 语义（有"数值逐位不变"测试）。
- **未复用**：旧 `frameworks/n02`、`baselines/n02_measurement.py`、`evaluation/n02_refutation_boundary.py` 只作对照阅读。

## 4. 实际命令与环境

- 远端 worktree `/workspace/owac/worktrees/n02-measurement-completion`（HEAD `861c46f67fff552c81fff89c5116c63b815224c5`，分支 `lzh/n02-measurement-completion`）；GPU `NVIDIA RTX PRO 6000 Blackwell`（95.6 GiB），与冻结 π.05 基线（约 31 GiB）共存。
- 解释器 `outputs/n02-full/runtime/py310-cu128/bin/python`（Python 3.10.19 / torch 2.11.0+cu128），`PYTHONPATH=$PWD/src`。
- 目标测试：`"$N02_PY" -m pytest tests/encoders/n02_full -q`（最新 **461 passed / 1 skipped**，skip 为既有 `particles.resampling` 缺 numba）；`make check` exit 0（280 files formatted）。
- P0 迁移检查：parity **16 pass / 0 fail / 1 not_run**（numba）、目标测试 257 passed/1 skipped、`n02_full_probe --device cuda` 通过。
- 训练命令（正式，seed 1103）：
  - `--stage p1-pretrain --config configs/encoders/n02_full/p1_pretrain.json --seed 1103 --updates 25000 --amp bf16 --platforms libero,robotwin --dev-eval-every 5000 --log-every 500 --tag formal-raw3-25k`
  - `--stage p1-pretrain --config configs/encoders/n02_full/p1_measurement.json --seed 1103 --updates 2000 --amp bf16 --synthetic-frames --tag formal-cfa4-stem-2k`
  - measurement 链（暂停待 T3 修复）：`start_chain.sh` → Job 3（`--stage p1-measurement --updates 5000 --amp bf16 --resume <stem>/final.pt`）→ nre → bnre → candidate_discrimination → conditional_likelihood。
- 验收命令：`"$N02_PY" -m owac.evaluation.n02_full.p1 {oracle,behavior,ablation,report} --config configs/encoders/n02_full/p1_eval.json`（dry-run 结构验证已通过：四子命令 exit 0、`scenario_records.jsonl` 逐场景记录、阈值判定 `incomplete`）。

## 5. 数据、种子与资源

- 数据：远端 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench`（未重新下载）；消费 v2 manifest 与六份冻结 anchor，**408075 锚点**，gzip SHA 与 `frozen-data.json` 全部一致（LIBERO 204301/28394/28436；RoboTwin 116117/15414/15413）；窗口 `rnn_frames=1`、LIBERO `action_horizon=10/offset=0`、RoboTwin `50/offset=1`。
- 归一化：LIBERO `quantile + gripper (a+1)/2`、RoboTwin 显式 checkpoint `MEAN_STD/zscore`；本机统计文件 SHA 记录在 manifest。
- 种子：模型 `[1103]`（本轮按用户裁定只跑 1 seed；`[2207,3301]` 待定）；数据生成 seed 由 partition+scene_id 派生。
- 资源实测：raw3 混平台 fp32 **6.81 s/update**、bf16 **5.01 s/update**；合成 CFA4（micro=32）**12.25 s/update**；单进程峰值 17.3 GiB 分配；并行两进程 + 基线 49.1–52.3 GiB。
- 用户裁定的预算（登记为偏离）：预训练 **25k**（25k×256/320418 ≈ **19.97 次等价曝光**，刚好达方案 §8.1 的 ≥20）而非 100k；CFA4 stem 2k；measurement 两臂各 5k；消融臂各 2k。CFA4 臂不能复用 raw3 权重（模式不同，禁止隐式互转），故单独有界预训练。

## 6. 原始计数与区间（截至本文件时刻）

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 远端目标测试 | 461 passed / 1 skipped | 本文件第 4 节命令 |
| P1 分文件测试 | observation 25 / sensor 45（含 checks）/ factors+model 64 / training+eval 75 | 各 owner 回报 + dispatcher 独立复跑 |
| 精度检查（预训练 bf16） | pass：loss_rel 1.3e-5 / grad_norm_rel 9.7e-5 / grad_slice 2.3e-3 / step 8.2e-3（容差 5e-2） | `outputs/n02-full/p1-precision/` |
| 精度检查（测量臂 bf16，T2/T3 修复后） | pass、`non_finite=False` | `outputs/n02-full/p1-precision/meas-arm-b8/` |
| dev 网格（先冻结判据） | lr0.5 0.04091 / lr1 0.07638 / lr2 0.07061（终点 dev loss）→ 选 **lr-scale 0.5** | `outputs/n02-full/p1-dev-grid/SELECTION_{RULE,DECISION}.md` |
| 双亮点回归（T2） | `model_inadequate`；`global_mean 0.9433 < thr 1.3874 < local_max 3.7751`，命中两处坐标 | `test_p1_checks.py` |
| 尺度一致性（T2） | 正确 DN 域 `per_valid_scalar` **1.9489** vs 归一化域 **8.51e12** vs 反向混域 **5.24e5**；显式换单位 → `UnitsMismatchError` | `outputs/n02-full/p1-t2-tests/scale_evidence.json` |
| CFA4 单步（T3+T4 集成） | `raw_preq` 2.18e8 → **594.51**；`candidate_ce` 1.06e6(detached) → **88.71**；`grad_norm_base` nan → **0.004198**；heads → **0.99999** | T4 回报 |
| 经验误报率（T2） | 34/800 = **4.25%** ≤ 5%（阈值 1.355338） | `test_p1_checks.py` |
| Job 2 CFA4 stem | 完成 2000 更新，loss 0.5082、`final.pt` 落地 | `outputs/n02-full/p1-formal/` |
| Job 1 raw3 预训练 | 运行中；update 1000 loss **0.5255**、`grad_norm_base 0.1093`、heads 0.0172 | `p1-pretrain-seed1103-formal-raw3-25k/train_log.jsonl` |
| 行为验收 | **pending / not_run** | 待 measurement 臂 |

## 7. 失败、事故与处置（全部保留原始记录）

1. **T1 `_build_mask` CUDA 跨设备崩溃**（CPU 测试无法发现）→ 修复 + 2 项 CUDA 回归测试（旧代码精确复现同一 `RuntimeError`）。
2. **T1 batched CFA4 把 batch 行当通道**（`got 2`）→ `_frame_layout` 显式 batch-first（冻结件 §3.9）。
3. **T3 历史特征被 `detach`**（382 个 encoder 参数无梯度）→ 不 detach（数值逐位不变）+ 可微 `StepOutput.world_scores`。
4. **T3/T4 CFA4 密度 1e8 量级失配、`grad=nan`** → T3 `UnitAffine` 反变换（1.98e6 → 36.05）+ T2 单位语义与 `log_ndtr` float32 反传缺陷修复（远尾 inf/错值）。
5. **T4 模式门"只报告不执行"**（CFA4 run 会真的训练 raw3 帧）→ 门在打开数据前执行 + 回归测试。
6. **T4 跨阶段续训被拒**（`stage does not match`）与 **Base 子集加载被拒**（`unknown submodule fcmae.mask_token`）→ 允许 pretrain→measurement 继承（`loaded 376/391`、跳过 15 个 fcmae 张量、新头 unmatched 29、optimizer 不恢复、计数归零）。
7. **watcher 管道 `$?` 取错**致链空跑却写 `chain done` → 改为直捕退出码、非 0 即中止，并加 60 s 看门狗（空闲超时 kill + `watchdog.json` + exit 3）。
8. **T3 `_normalized_coordinates` 逐点 Python 循环**致测量路径纯 CPU 自旋（2-update smoke 39 分钟 100% CPU、读入 checkpoint 后 rchar 停滞；`faulthandler` 栈钉在 `factors.py:1400`）→ 三个卡死进程已 kill，修复进行中。
9. **dev 评价滚动子集**（同一 update 两个 dev loss）与 **精度检查 NaN 盲判**（`max(0.0, nan)`）→ T4 自查修复；第一轮 lr 网格作废重跑。
10. **基础设施事故**：远端根目录混入未跟踪残留（已隔离到 `/workspace/owac/quarantine-wt-root-strays-20261014/`，含 MANIFEST）；`src/`、`tests/` 混入携带本地路径的 477 个 `.pyc`（155 个含 `/home/jeong`）→ 全部清除，清除后测试仍 461 passed 且路径恢复远端。
11. **uv 安装索引策略偏差**（远程说明原件缺 `--index-strategy unsafe-best-match`）→ 记录并加参数安装成功。

## 8. 未运行项与下一交接条件

- **not_run**：行为验收四类阈值（待 Job 3）、测量 raw3 真实窗口臂（Job 4，待 Job 1 完成 + 正式规模精度检查）、2/3 seed、FP64 精度参考、三个输入消融（`raw_frame`/`conditional_innovation`/`registration`，缺输入域开关，按裁定不作近似实现）、完整 ≥1000 场景 ×3 seed 口径（1-seed 阶段先 seed 1103）。
- **限制如实记录**：dev 网格终点为单点估计（中间点排序不稳定）、单 seed、CFA4 与 raw3 两模式不能共享 Base。
- **下一交接条件**：T3 修复落地 → 2-update smoke 通过 → `start_chain.sh` 串行 Job 3 → nre → bnre → 两消融臂 → `p1.py behavior/oracle` 跑 ≥1000 场景/类（seed 1103，逐场景记录保留） → dispatcher 复核后写 `stage_decision.json` 并交接 E1。

## 9. 复现命令（可运行）

```bash
# 目标测试（远端）
cd /workspace/owac/worktrees/n02-measurement-completion
N02_PY=$PWD/outputs/n02-full/runtime/py310-cu128/bin/python
PYTHONPATH=$PWD/src "$N02_PY" -m pytest tests/encoders/n02_full -q

# P0 迁移 parity（可选，numba 缺则为 16/0/1）
PYTHONPATH=$PWD/src "$N02_PY" -m owac.frameworks.n02_full.components.parity_suite \
  --fixtures outputs/n02-full/p0/parity/fixtures --output outputs/n02-full/p0/parity

# dev 网格（已冻结判据）
ls outputs/n02-full/p1-dev-grid/SELECTION_RULE.md outputs/n02-full/p1-dev-grid/SELECTION_DECISION.md

# 正式训练（seed 1103；Job 1 在跑）
bash outputs/n02-full/p1-formal/start_raw3.sh
bash outputs/n02-full/p1-formal/start_chain.sh          # stem 之后的链（T3 修好后）

# 验收（结构 dry-run / 正式）
PYTHONPATH=$PWD/src "$N02_PY" -m owac.evaluation.n02_full.p1 report --config configs/encoders/n02_full/p1_eval.json
```

大数组、检查点与原始日志留在远端 `outputs/n02-full/`；本文件与 `results/` 清单可追踪。

## 10. 迁移与当前运行状态（2026-10-14 22:00+ 补记）

**用户裁定**：N02 迁到新服务器（`ssh -p 16724 root@154.59.156.22`，2× RTX PRO 6000 Blackwell 各 97887 MiB），**不得强夺另一条研究线（N04）的训练**。

**迁移与验证**（台账 §21/§22）：代码（5 个关键文件哈希与本地逐位一致）、运行时（py310-cu128 重建，torch 2.11.0+cu128 见到 2 张 GPU）、资产（6 份 anchor + 2 份 manifest）、数据（HF `ToferFish/RawVLA-Bench` rev `8586c594fff9ce38580e29745f074e82e96e74f7`，665 GB，HF CLI 报 `✓ Downloaded`）、正式归一化统计（libero `b3a44bb2810436fb`、robotwin `229a4a2d6cfd1ba4`，`load_formal_normalisation` 实测可加载）。新机验证：`compileall` OK、`n02_full_probe` OK、**目标测试 500 passed / 1 skipped**、**parity 16 pass / 0 fail / 1 not_run（numba）**、`make check` 两端全绿。

**旧机 aborted_attempts（写进新机 provenance）**：attempt1 = pkill watcher 连带子进程（code `876f2006`，→update 50）；attempt2 = 同步风暴（code `669037fd`，→update 250）；attempt3 = 迁移前重启（含 `d1c068a6`，→~update 50）；旧机 N02 进程已全部停止，frozen π0.5 基线 4 进程未受影响，partial 日志留在旧机。

**当前在跑（交付时）**：GPU0 = CFA4 stem 2k 重跑 → interim **1000** 更新 → 验收三步（`oracle`+`behavior`+`report`，`--interim-budget 1000`）→ `--resume` 续到累计 **5000** → 四臂（nre/bnre/候选判别/条件似然，各 2k）；GPU1 = raw3 25k 预训练（bf16，混平台，检查点每 5000）。GPU1 上的非 N02 租户未触碰。

**尚未完成（下一轮交接条件）**：一期验收（oracle 正确率 + behavior 五族与阈值）、累计 5k、四臂、seed 2207/3301、raw3 measurement 臂（Job 4，含正式规模精度检查）、`raw_frame`/`conditional_innovation`/`registration` 三个输入消融（缺输入域开关，按裁定不作近似实现）。恢复与复现命令见 `results/2026-10-14_n02_p1_observation/README.md`。

## 11. 一期（interim 1000 更新）结论：受控任务信息不足（2026-10-15 补记）

**oracle（P1 §6 的"任务可纠正"上界）**：解析 oracle（生成器真值几何 + T2 `mean_variance` 钩子 + T3 公开 query 构造器）在 961 个匹配场景上 **correct 797/961 = 0.8293**（Wilson [0.8043, 0.8518]），冻结阈值 `min_correctable=0.95` → **verdict=fail**。

**dispatcher 独立复算与机制**：`scenario_records.jsonl` 961 条中 **961/961 存在并列的候选分数**；其中 693 条（72.1%）四个候选**只有一个不同分数值**（即完全同分）；**真值是唯一最小值的记录仅 1/961 = 0.0010**；`map_index==0` 的次数 797 恰等于 correct 次数 → **0.8293 是"平局时 argmin 取下标 0"的裁决假象，解析 oracle 的真实可区分率 ≈ 0.1%**。

**判定**：这是**受控任务在冻结配置下的信息不足**（候选假设在采样点上的期望信号相同），不是数值/实现缺陷；按 P1 §6"若 oracle 也失败，报告信息不足"处理。**未调阈值、未挑场景、未改候选字典**；下一轮是否加密 query 采样/加大候选几何差/细化字典由用户决定。

**关联**：该现象与旧"双亮点未被绝对拟合拒绝"的边界诊断同源（几何差异在当前采样/测量模型下不可分辨），本轮的 oracle 给出了可量化的形式。

**过程更正（T4 自查）**：interim 1k 的"静默退出"判断已撤回；真实根因是 harness 把**跨阶段继承的预训练更新数**计入本阶段预算（`start = resume.updates + --updates`），导致越预算运行；已修（`budget_start()`：继承数只作 `source_updates`，`--updates` 严格表示本阶段更新数，附回归测试）。

**behavior（五族）**：交付时仍在计算（CPU 密集的场景渲染，etime >90 min），由 T4 完成后写入 `results/2026-10-14_n02_p1_observation/acceptance_interim.md` 与 `metrics.json` 的 `acceptance_interim` 字段；按同一诚实口径（不修饰）。预期 cue_swap 类因同一信息不足机制不达标，无信息漂移/误报类另计。

## 12. 阶段结论与下一步建议（2026-10-15）

**P1 判定**：`implementation = pass`、`numerical = pass`、`behavior = fail`、`closed_loop = not_run`。

**行为门槛的失败是"任务信息不足"型失败，不是实现/数值缺陷**，证据链：
1. **解析 oracle（§6 的上界要求）**：797/961 = 0.8293 < 冻结阈值 0.95 → fail；逐记录统计显示 **961/961 存在并列候选分数、真值唯一最小仅 1/961 = 0.001、correct 次数恰等于 `map_index==0` 次数** → 0.829 是平局裁决假象，**真实可区分率 ≈ 0.1%**。
2. **测量模型（interim 1000，961 场景/类）**：`cue_swap both_correct 0.1322`（Wilson [0.1122,0.1550]）**低于 4 候选随机水平 0.25**；`wrong_prior` 两向 0.6556 / 0.2986；`insufficiency` 检出 **0.0**；`normal` 误报 **0.0**；`no_information` 漂移 961/961 在阈内（max 0.0017）但场景数 961 < 冻结下限 1000 → `incomplete`。
3. 两层的失败集中在同一机制：**候选假设在当前 query 采样与字典粒度下产生相同的期望信号** → 模型无信息可用，且从不判定"信息不足"（检出 0）。

**未做的事（按契约）**：未调阈值、未挑场景、未改候选字典、未删目标；未进入 P2/E1。**下一轮由用户决定**：加密 query 采样 / 加大候选几何差 / 细化候选字典，使受控任务携带可区分信息；随后在**累计 5k 检查点**上重跑同一冻结口径验收（链已排队自动执行），并核对四臂（nre/bnre/候选判别/条件似然，各 2k 有界预算）的相对评分对照。

**仍未完成（继任交接项）**：累计 5k 的正式口径验收（`outputs/n02-full/p1-acceptance-5k/`，watcher 已就位）、四个消融臂、raw3 25k 预训练（GPU 1，ETA ≈ 3 天）、raw3 measurement 臂、seed 2207/3301、`raw_frame`/`conditional_innovation`/`registration` 三输入消融。

**本轮工程上仍值得保留的方法学结论**：跨阶段继承的更新数**不得**计入本阶段预算（`budget_start()`）；评测/训练步必须限 torch 线程（否则单步 880 CPU-s vs 1.9 CPU-s）；热路径**每步一度量同步**（183→2，争用下 85→16.4 s/update）；跨阶段续训要显式映射预训练独有模块并记录 `source_updates`；`log_ndtr` 在 float32 远尾必须自实现；单位/尺度必须在打分前显式反变换（否则密度差 10⁶ 量级）。

## 13. 最终口径验收（累计 5000 更新，seed 1103）：fail，机制不变

`outputs/n02-full/p1-acceptance-5k/`（`claim_scope=full evaluation`；961 个有效场景/类；6 类共 11532 条逐场景记录）：

| 家族 | 5k 实测 | interim 1000 | 冻结阈值 | verdict |
| --- | --- | --- | --- | --- |
| oracle（解析上界） | **797/961 = 0.8293**（Wilson [0.8043,0.8518]） | 0.8293 | ≥0.95 | **fail** |
| cue_swap both_correct | **17/961 = 0.0177**（[0.0111,0.0281]） | 0.1322 | ≥0.90 | **fail** |
| no_information 漂移 | **961/961 = 1.0**（mean 0.0081 / max 0.0397） | 1.0（0.0004/0.0017） | ≤0.05 | incomplete（961<1000；指标通过） |
| wrong_prior a / b | **0.0687 / 0.7471** | 0.6556 / 0.2986 | 两向 ≥0.80 | **fail** |
| checks normal 误报 / insufficiency 检出 | **0.0 / 0.0** | 0.0 / 0.0 | ≤0.05 / ≥0.90 | checks fail |

**读数**：oracle 与 interim 完全相同（信息不足不变），而 cue_swap 从 0.1322 **退化**到 0.0177（低于 4 候选随机 0.25）、wrong_prior 方向 a 从 0.6556 退到 0.0687（方向 b 从 0.2986 升到 0.7471）——**在不可区分的任务上继续训练会破坏判别**，这与"信息不足"的机制一致，不改变结论。

**判定**：P1 行为门槛在最终口径上仍为 **fail（信息不足）**；按 P1 §6 保留失败并**不进入 P2/E1**。未调阈值、未挑场景、未改字典。修复路径（加密 query 采样 / 加大候选几何差 / 细化字典）仍是用户下一轮的决策项。

### 13.1 口径更正（2026-10-15 12:0x）

T4 自查发现 `budget_start()` 的两个 stage 字符串在 `run_pretrain`/`run_measurement` 中**接反**，因此第 13 节的"累计 5000"实测状态实际是**累计 4000 更新**（`new_updates=3000`；检查点为 `update-000{2,3,4}000.pt` + `final.pt`）。**数字本身不变且有效**，只需把预算口径改为 **4000**。修复已落地并新增**接线守卫测试**（用 `inspect.getsource` 检查两个函数体内各用哪个 stage 字符串——此前的单测只验证了 helper，没验证接线，因此漏过）。真正的累计 5000 扩展与独立目录验收（`p1-acceptance-5k-true/`）已排队。另：`candidate_discrimination` 臂同样受接线缺陷影响（声明 2000、实际会跑到 4000），按"不中途杀"保留并如实标注预算不可直接比。

## 14. 对照臂（有界预算）结果

四个对照臂在**冻结 CFA4 stem Base**（`base_frozen=true`、`base_inheritance: loaded 376 / skipped 15 (fcmae) / source_stage=p1-pretrain`）上各跑有界预算，`same_budget_as_main=false`（2k vs 主臂 5k），seed 1103：

| 臂 | declared / actual | 损失（首→末） | 读数 |
| --- | --- | --- | --- |
| `nre` | 2000 / 2000 | 0.693366 → **0.693168** | 停在 ln2 ⇒ 该损失在 2k 预算下**没有学到判别** |
| `bnre` | 2000 / 2000 | 2.322817 → 0.762704 | 有下降但**不构成判别证据**（oracle 已证任务不含可判别信息） |
| `candidate_discrimination` | 2000 / **4000**（`budget_start` 接线缺陷；与其它臂预算不可直接比） | … → 21.6998（`candidate_ce`） | CE 下降但**不构成学会判别**（同 oracle 结论） |
| `conditional_likelihood` | 2000 / 2000 | 见 `run_manifest.json` | 同口径记录 |

**未运行**：`raw_frame`、`conditional_innovation`、`registration`（缺输入域开关，按裁定不作近似实现）。

**代码哈希差异说明**：`candidate_discrimination` 跑在修复前代码（无线程限幅，6.99 s/update），其余臂跑限幅代码（`OWAC_TRAIN_THREADS=8`）——**仅调度差异、语义一致**；另有 `budget_start` 接线缺陷修复前后两版哈希，均已记入各 run manifest 与 `code_hash.json`。

**结论**：四个臂在有界预算下均**未显示可用判别能力**（与 oracle 的"信息不足"结论一致）；臂的价值在于把"相对评分接口（NRE/BNRE/CE/条件似然）能跑通且口径可核对"这件事做实，而不是给出正证据。

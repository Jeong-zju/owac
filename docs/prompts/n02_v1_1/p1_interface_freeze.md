# P1 接口冻结与文件所有权（dispatcher 冻结，2026-10-14）

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

> 历史 v1.1 文档。2026-10-15 当前路线已切换到 [N02 v1.2](../../plans/N02_full_development_v1.2.md)，当前 P2 完整实现待实施。以下正文中的启动语、服务器、阶段门槛和预算保留原始版本语境，不得作为当前执行包。新版[P2—P4自包含包](../n02_v1_2/README.md)已备，当前执行单种子与所选模型100次闭环；P1 独立验收未通过的记录保留。

本文件是 N02 v1.1 **P1（RAW 观测与绝对拟合）** 的调度冻结件：所有实现者按此文件的接口、所有权和证据规则并行开发，不得自行改动接口名或语义。与 [共同契约](00_SHARED_CONTRACT.md)、[P1 阶段原文](P1_OBSERVATION.md)、[远程开发说明](03_REMOTE_DEVELOPMENT.md)冲突时，以上三者为准，并立即回报 dispatcher。

## 0. 事实基线（2026-10-14 实测）

- 远端 worktree `/workspace/owac/worktrees/n02-measurement-completion`，分支 `lzh/n02-measurement-completion`，HEAD `861c46f67fff552c81fff89c5116c63b815224c5`；本地交接副本同 HEAD。
- 远端运行时 `N02_PY=outputs/n02-full/runtime/py310-cu128/bin/python`（Python 3.10.19，torch 2.11.0+cu128，`torch.cuda.is_available()=True`，GPU=RTX PRO 6000 Blackwell 97887 MiB）。根环境 `uv sync --locked` 用于 `make check`。
- 远端 P0 回归已通过：`make check` 全绿；目标运行时 `tests/encoders/n02_full` 与 `n02_full_probe --device cuda` 见 `outputs/n02-full/remote-bootstrap/`。
- P0 资产：v2 四分区 manifest（`research/experiments/results/2026-10-14_n02_p0_independent_acceptance/manifest-{libero,robotwin}.json.gz`，本机 `outputs/n02-full/p0-independent-audit/` 原始 JSON 亦可用）、六份冻结 anchor（`outputs/n02-full/p0-independent-audit/frozen-data/`，408075 锚点）、`frozen-data.json`。
- GPU 共享：冻结 π0.5 基线进程占用约 32 GB；**不得停止或修改该进程**。P1 作业启动前先 `nvidia-smi`，不超可用显存，不使用 `--exclusive` 优先权。

## 1. 代码编写与运行位置（冻结）

- **代码在本地交接副本编写**（`/home/jeong/zeno/owac/worktrees/n02-measurement-completion`），用 `rsync` 同步到远端后**在远端运行全部测试/训练/评测**。
- 任何在报告里出现的数字，必须是远端同步后文件哈希对得上的那一次运行的原始输出；同时给出远端 `sha256sum`。
- 同步命令（src/tests/configs 各自执行）：
  ```bash
  rsync -az -e "ssh -p 44327" \
    /home/jeong/zeno/owac/worktrees/n02-measurement-completion/src/owac/ \
    root@107.209.104.125:/workspace/owac/worktrees/n02-measurement-completion/src/owac/
  ```
- 远端测试命令模板：
  ```bash
  ssh -p 44327 root@107.209.104.125 'cd /workspace/owac/worktrees/n02-measurement-completion \
    && N02_ROOT=$PWD N02_PY=$PWD/outputs/n02-full/runtime/py310-cu128/bin/python \
       PYTHONPATH=$PWD/src "$N02_PY" -m pytest tests/encoders/n02_full/test_p1_xxx.py -q'
  ```
- CPU/静态检查在本地或远端根环境执行：`make check`（ruff + format + compileall），必须 exit 0。

## 2. 文件所有权（唯一写入者；跨所有者改动先回报 dispatcher）

| 文件 | Owner | P1 内容 |
| --- | --- | --- |
| `src/owac/frameworks/n02_full/observation.py` | T1 | Base 多尺度 RAW 编码、坐标映射、RAW3/CFA4、FCMAE 掩码预训练组件 |
| `tests/encoders/n02_full/test_p1_observation.py` | T1 | 观测层测试 |
| `src/owac/frameworks/n02_full/sensor.py` | T2 | 测量分布、正参数化方差、量化/饱和概率、η 干扰模型、`RawPrediction`/`MeasurementScore` |
| `src/owac/frameworks/n02_full/checks.py` | T2 | 绝对检查、经验阈值校准、局部残差与全图统计 |
| `tests/encoders/n02_full/test_p1_sensor.py`、`test_p1_checks.py` | T2 | 传感器与检查测试 |
| `src/owac/frameworks/n02_full/factors.py` | T3 | `ObservationModel` 门面、候选字典、条件 RAW decoder、fit/prequential 两条调用 |
| `src/owac/frameworks/n02_full/model.py` | T3 | 最小因果步进 `step_bundle`、目标身份、无第二份 likelihood |
| `tests/encoders/n02_full/test_p1_factors.py`、`test_p1_model.py` | T3 | 拟合与因果测试 |
| `src/owac/training/n02_full.py`、`src/owac/evaluation/n02_full/p1.py` | T4 | 两阶段训练入口、p1 评测子命令 |
| `configs/encoders/n02_full/p1_{pretrain,measurement,eval}.json`、`config.py` 扩展 | T4 | 配置与校验 |
| `tests/encoders/n02_full/test_p1_training.py`、`test_p1_eval.py` | T4 | 训练/评测测试 |

P0 文件（`config.py` 除外的 `data.py`/`state.py`/`synthetic.py`/`components/*`）**默认只读**；确需扩展时由 dispatcher 指派单一写入者并保持 P0 测试全绿。P1 不复用旧 `n02` 的实现，不修改 `baselines/n02_measurement.py` 与 `evaluation/n02_refutation_boundary.py`（只作对照阅读）。

## 3. 冻结接口

新增类型定义在各自 owner 模块内；`state.py` 已有类型（`MeasurementKey`、`MeasurementRecord`、`MeasurementLedger`、`ObjectiveKey`、`ObservationBundle`、`InferenceInput`、`TrainingTargets`、`FeatureLevel`、`WorldHypothesis`、`SequenceState`、`StageResult`）直接复用，不改名。

### 3.1 observation.py（T1）

```python
@dataclass(slots=True)
class FeaturePyramid:
    levels: dict[str, FeatureLevel]  # "stride8" / "stride16"
    camera_ids: tuple[str, ...]
    input_mode: str  # "raw3" | "cfa4"
    sensor_shape: dict[str, tuple[int, int]]  # camera -> (H, W) 传感器分辨率
    support_hash: str


class RawObservationEncoder(nn.Module):
    def encode(self, bundle: ObservationBundle) -> FeaturePyramid: ...
    def encode_frame(
        self, frame: MeasurementFrame, *, camera_id: str
    ) -> dict[str, FeatureLevel]: ...


class RawFCMAEPretrainer(nn.Module):
    def forward(
        self, frames: dict[str, MeasurementFrame], *, camera_ids: tuple[str, ...]
    ) -> FCMAEOutput: ...


@dataclass(slots=True)
class FCMAEOutput:
    loss: torch.Tensor  # 仅有效、非饱和采样点
    reconstruction: torch.Tensor
    mask: torch.Tensor
    valid_target_count: torch.Tensor
    saturated_count: torch.Tensor
```

约定（必须成立并测试）：
- `FeatureLevel.values` 形状 `(B, 512, H_l, W_l)`，`valid` 同空间形状 bool；`sensor_coordinates` 形状 `(B, H_l, W_l, 2)` float32，单位是**传感器像素坐标**（CFA4 packed 网格也映射回 sensor 坐标，禁止把 256 当传感器 256）；`level_id` 为 `stride8`/`stride16`。
- 多相机分辨率可不同；编码后 token 序列拼接但携带 camera ID 与原始坐标，不允许把相机折进 batch。
- RAW3 是 3 通道伪 RAW 模式；CFA4 是 4 平面重排（保留实际相位）。两模式由配置显式选择，共享实现不得隐式互转。
- 输入 padding/crop 必须有坐标映射；矩形输入沿用 P0 padding/crop 政策（stride 对齐），256 方形 crop 只作预注册增广。
- FCMAE 掩码只由当前帧 support 生成，seed 由 `(episode_id, camera_id, source_frame_id)` 派生；不得读取未来帧、truth 或 evaluation 字段。缺失/坏点不进 loss；饱和按尾概率另报。
- 重建输出只进训练 loss，绝不作为策略/读出输入；不复用 RGB 权重。

### 3.1+ T1 交付后的裁定（2026-10-14 dispatcher，T1 已按此实现并独立复跑）

- **多相机键**：单相机时 `levels` 键严格为 `stride8`/`stride16`；多相机时为 `"<camera_id>::<level_id>"`（相机不折进 batch）。`FeatureLevel.level_id` 始终是裸名 `stride8`/`stride16`。消费方（T3/T4）必须用 `FeaturePyramid.per_camera()` 或 `.tokens()`，**不得按字典键字符串猜测**。
- **`valid` 语义**：cell 有效 = 该 stride 对齐窗口内**至少有一个**可用测量（复用 P0 `fcmae.dilate_support` 规则）；不声明全窗口覆盖；padding 区 `valid=False` 且值为 0。需要“全窗口覆盖”判据时另行加字段，不得改判现有 valid。
- **归一化**：`metadata["calibration"]` 声明 black/white level 时输入与重建目标按 `(v-black)/(white-black)` 处理且不 clip（饱和单独计数）；未声明校准时原样使用调用方数值。
- **`FeaturePyramid.transforms`**：记录每相机 padding/crop（`FrameTransform`），不删除或改名任何冻结字段；`FCMAEOutput.reconstruction/mask` 带前导相机轴 `(V,B,...)`（单相机 squeeze）。

### 3.2 sensor.py（T2）

```python
@dataclass(slots=True)
class RawPrediction:
    mean: torch.Tensor  # (*, ) 与 query 采样点对齐
    variance: torch.Tensor  # 正参数化，同形状
    masks: dict[str, torch.Tensor]  # valid/bad/missing/saturated
    units: str  # 与 query 测量单位字符串逐字相等


@dataclass(slots=True)
class MeasurementScore:
    relative_terms: dict[str, torch.Tensor]
    raw_nll_or_energy: torch.Tensor
    local_stats: dict[str, torch.Tensor]  # 局部标准化残差、局部极大值等
    kind: str  # "proper_nll" | "empirical_energy"
    per_valid_scalar: torch.Tensor


@dataclass(slots=True)
class InterferenceEstimate:
    log_exposure: torch.Tensor  # 全相机共享
    camera_translation: torch.Tensor  # 每帧二维，按相机
    noise_scale: torch.Tensor
    source: str  # "history" | "oracle"
```

约定：
- 高斯连续项 `0.5*((y-mu)^2/v + log(v) + log(2*pi))`；量化项为 `-log(CDF(upper)-CDF(lower))`，稳定 log 空间计算；顶端/底端 clip 记尾概率。合法 `valid = ~(missing|bad)`；缺失/坏点不贡献任何测量项。
- 方差单列传感器方差与模型预测不确定性，正参数化（softplus/exp），极小方差与饱和有数值检查。
- 合成模型用归一化测量密度；真实伪 RAW 若 joint density 不可验证，`kind="empirical_energy"` 并如实标注。
- η 范围与先验在开发集冻结；相机运动作用于全部背景/对象投影，实体移动只作用该实体状态；**不得从 EvaluationTruth 读 η**；oracle η 只存在于 evaluator。
- 重叠多尺度不重复计入目标；每个真实采样位置只出现一次。

### 3.2+ T2 交付后的裁定（2026-10-14 dispatcher，T2 已按此实现并经我独立复跑）

- `RawPrediction` 保持 `mean/variance/masks/units` 四字段顺序不变，允许增补可选 `variance_components: dict`（`sensor`/`model`/`total`），用于“传感器方差与模型预测不确定性单列”。
- `sensor.py` **不提供** oracle η 构造函数；history 路径恒为 `source="history"`，oracle 只能由 evaluator 显式构造 `InterferenceEstimate(source="oracle")`。
- `checks.py` 误报率证据（远端最终哈希那一次）：正常序列经验误报 **34/800 = 4.25%**（阈值 1.355338 / calibration 400 条 seed 20261014 / test seed 202）；双亮点回归 `model_inadequate`，`global_mean=0.9433 < threshold=1.3874 < local_max=3.7751`，命中两处亮点坐标。
- 记录更正：T2 回报曾称全仓 `make check` 失败项含 `test_p1_factors.py`/`config.py`；我实测失败项只在 `src/owac/training/n02_full.py`(18)、`src/owac/evaluation/n02_full/p1.py`(7)、`tests/encoders/n02_full/test_p1_training.py`(3)，全部属 T4 未完成文件。T3 的 lint 自述正确。

### 3.3 checks.py（T2）

```python
@dataclass(slots=True)
class CheckVerdict:
    status: str  # "compatible" | "insufficient_information" | "model_inadequate"
    interference_explainable: bool
    local_residual_positions: tuple[tuple[int, int], ...]
    calibration_version: str
    statistics: dict[str, float]
    viewing_rule_hash: str


class WindowedCalibrator:
    @classmethod
    def fit(
        cls, statistics: Sequence[float], *, quantile: float = 0.95, rule_hash: str
    ) -> "WindowedCalibrator": ...
    def threshold(self) -> float: ...
    def assess(self, statistics: Sequence[float], *, version: str) -> CheckVerdict: ...
```

约定：固定窗口经验检验；calibration 与 test 使用**完全相同**的查看规则（全部尺度/区域/查看时刻），冻结 95% 分位阈值；同时报告全图均值与局部最大值（旧“双亮点”不得被背景均值抹平）；只输出上述三态，不指认具体关系错误；不宣称 anytime e-process 保证。阈值针对整条序列的经验误报率。

### 3.4 factors.py（T3）

```python
@dataclass(slots=True)
class QuerySpec:
    keys: tuple[MeasurementKey, ...]
    sensor_coordinates: torch.Tensor   # (Q, 2)
    camera_ids: tuple[str, ...]
    frame_index: int
    units: str

@dataclass(slots=True)
class HistoryContext:
    pooled_features: torch.Tensor      # (B, 512)
    levels: dict[str, FeatureLevel] | None
    episode_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    last_frame_index: int | None       # None 表示无历史
    eta: InterferenceEstimate | None

class ObservationModel:
    def __init__(self, encoder, sensor_model, config, *, version: str): ...
    def encode(self, bundle: ObservationBundle) -> FeaturePyramid: ...
    def predict(self, world: WorldHypothesis, history: HistoryContext, query_spec: QuerySpec,
                *, call: str) -> RawPrediction: ...   # call ∈ {"fit","prequential_check"}
    def score(self, prediction: RawPrediction, raw_query) -> MeasurementScore: ...

class CandidateDictionary:
    def worlds(self) -> tuple[WorldHypothesis, ...]: ...   # 正确/错误/全错三类
    def oracle_world(self) -> WorldHypothesis: ...          # 只允许 evaluator 使用
    def label_index(self, truth_geometry) -> int | None: ...  # 由字典构造显式给出正确候选下标

def build_p1_task(...) -> P1Task: ...
    # 组装 InferenceInput / TrainingTargets / EvaluationTruth / CandidateDictionary；
    # TrainingTargets.optional_geometry 必须为 dict 且含
    # {"candidate_index": int, "world_px": <真值几何>}（真值只进 loss/evaluator）
```

`label_index` 是**构造性**映射（字典生成时即确定哪个候选正确），不是最近邻启发式；最近邻匹配只允许作为交叉核对并如实报告。

约定：
- 候选字段只含允许的结构/几何假设（几何参数、干扰参数），truth 只进 loss/evaluator；所有方法看到同一字典；各候选共享 decoder 参数、容量固定且相等，禁止候选用可抄当前 query 的任意 skip。
- `fit` 可用明确测量集合的特征；`prequential_check` 只能用 ≤t-1 的历史，先打分锁定、再允许 y_t 校正；无历史首帧不产生伪独立检查（标记为 calibration-only）。
- 干扰 η 从历史估计；若当前 query 参与了 η 优化，检查统计必须在校准集重演同一拟合过程并标 `empirical`。
- 保留解析生成器 oracle 作为信息上界（同接口、独立类，不进候选字典）。

### 3.5 model.py（T3，最小因果步进）

```python
@dataclass(slots=True)
class StepOutput:
    objective: ObjectiveKey
    worlds: tuple[WorldHypothesis, ...]
    best_world: WorldHypothesis | None
    locked_checks: tuple[MeasurementScore, ...]  # 校正前锁定（prequential）
    fit_scores: tuple[MeasurementScore, ...]  # 校正后 fit 模式，与 worlds 对齐
    state: SequenceState
    ledger_version: int
    evidence_count: int
    residuals: dict[str, float]
    cost: dict[str, float]


def step_bundle(
    bundle: ObservationBundle,
    state: SequenceState,
    model: ObservationModel,
    history: HistoryContext,
    *,
    ledger: MeasurementLedger | None = None,
    budget=None,
    targets: TrainingTargets | None = None,
) -> StepOutput: ...
```

约定（P1 只实现支持契约的最小版本，不得提前堆 P2 功能）：按 episode reset；登记本时刻全部新 `MeasurementKey`；仅 `new_time_mask` 为真者执行一次预测；在读取本时刻目标测量**之前**形成 `prequential_check` 并锁定；同一 `ObjectiveKey` 下不得累计第二份 likelihood；重传同一帧只保留一份证据、校准变更只替换派生版本；P1 实体结构固定，不实现出生/拆分/学习局部调度。输出 `score_kind` 必须是 `normalized_density`/`empirical_energy`/`importance_weight` 中真实的一种。

### 3.6 training/evaluation CLI（T4）

```text
python -m owac.training.n02_full --stage p1-pretrain    --config configs/encoders/n02_full/p1_pretrain.json ...
python -m owac.training.n02_full --stage p1-measurement --config configs/encoders/n02_full/p1_measurement.json ...
python -m owac.evaluation.n02_full.p1 <子命令> --config configs/encoders/n02_full/p1_eval.json ...
```

- 训练超参起点按 P1 原文：预训练 AdamW lr 1.5e-4 / wd .05 / betas(.9,.95) / 5% warmup / cosine / 有效 batch 256 帧；开发网格 lr ∈ [7.5e-5, 1.5e-4, 3e-4]。测量阶段有效 batch 32 条×8 帧，Base lr 3e-5、新头 3e-4、wd .01、warmup 5%，开发网格 0.5×/1×/2×。
- 每 5000 更新固定开发行为评估；每 loss 记录激活数、Base/heads 梯度与夹角。
- 检查点保存 model/optimizer/scheduler/AMP/RNG/数据采样位置/版本/分区哈希；同 seed 可续训并分别记录新增与累计更新数。
- 正式 seed `[1103, 2207, 3301]`；数据生成 seed 由 partition+scene_id 派生，与模型 seed 分离。P1 预算：预训练 100k 更新/seed、测量 50k 更新/seed。
- 训练与推断共用同一 `step_bundle`；监督只经 `TrainingTargets`。

### 3.7 损失归属与标签契约（2026-10-14 dispatcher 裁定，修订 §3.4/§3.5）

- **loss 计算在训练 harness（T4）内**，不放进 `StepOutput`：保持 `model.py` 是纯在线路径。`StepOutput` 只提供可判定的原始量（`locked_checks` 为 prequential、`fit_scores` 为 fit、`worlds[*].score`/`score_kind`、`best_world`）。
- **有监督候选 CE 的正确标签**由受控任务构造方（T3 的 `CandidateDictionary.label_index` / `build_p1_task`）显式写入 `TrainingTargets.optional_geometry["candidate_index"]`；T4 优先使用该字段，缺失时该 loss 项记 `unavailable` + 原因，不得用启发式标签冒充，也不得伪造 0。
- **RAW proper prediction**：训练同时记录 prequential（`locked_checks`）与 fit（`fit_scores`）两条 proper score，分开报告；不得把两者相加成一份 likelihood（同一目标集合只计一次）。
- 几何 aux 使用 `best_world` 的 `geometry_frame="view_2d"` 与 `targets.optional_geometry["world_px"]` 的 MSE；帧不一致或键缺失时记 `unavailable`。
- 测量阶段须同时消费 (a) 受控序列（候选 CE + 几何 aux）与 (b) 真实伪 RAW 窗口（proper prediction / 噪声结构未知时 `empirical_energy`），分别记录每个 loss 的激活数、Base/heads 梯度范数与夹角。

### 3.5+ T3 交付后的裁定（2026-10-14 dispatcher，T3 已按此实现并经我独立复跑 43 passed）

- **`StepOutput.locked_checks` / `fit_scores` 的形状约定**：每 world 一项（`len(...) == len(worlds)`）；每个 `MeasurementScore` 的张量在 batch 内只对 `new_time_mask` 为真的行按行拼接，前导维 `R`（新时间行数），query 维 `Q`；B=1 时 `R=1`，与 worlds 1:1。`worlds[*].score` 是 fit 均值 NLL，`score_kind` 由 sensor kind 映射（`proper_nll → normalized_density`）。
- `HistoryContext` 增加可选字段 `eta_fit_keys: tuple = ()`（默认空）；`pooled_features` 在 `last_frame_index is None` 时允许为 `None`（校准先验路径）。
- `RawPrediction.masks` 可额外携带 `calibration_only`/`empirical_check`/`oracle_eta` 布尔张量；`valid`/`bad`/`missing`/`saturated` 语义不变。
- P1 的 `budget` 只允许一个 update cycle；P2 的求解循环不得提前实现（守最小契约）。
- `build_p1_task` 在 `all_wrong` 字典下**不写** `optional_geometry["candidate_index"]`；loss 侧据此记 `unavailable`，不伪造 0。
- `ObservationModel.score` 先逐字校验 `units` 再委托 sensor 层；`ObservedModel` 允许 duck-typed sensor 回退，但正式路径必须走 `sensor.build_prediction(...)`（带 `variance_components`）。

### 3.5++ T3 修复轮 1 裁定（2026-10-14 dispatcher，已独立复跑 50 passed / 431 passed）

- **新增 `StepOutput.world_scores: torch.Tensor`**（`len == len(worlds)`，与 `fit_scores` 同一 fit 目标集合，`requires_grad=True`）。`worlds[k].score` 是同一量的 detach float 在线读出；二者关系写进 docstring。候选 CE 由训练 harness 用 `softmax(-world_scores / T)` 计算，温度 `T` 与常数项必须写进配置与 manifest。
- **梯度连续性**：`_context_tensor`（`pooled_features`）与 `pool_feature_pyramid`（`FeatureLevel.values`）只做 dtype/device 转换、**不 detach**；测量值、候选几何、真值几何保持默认 detach。数值路径与 detach 版本逐位相同（有测试）。
- **多相机逐点语义**：`QuerySpec`/`RawQuery` 携带逐点相机索引（`camera_ids`、`camera_index`），prediction 的 `masks["camera_index"]` 与之配对；`score` 在委托前对齐断言，违规抛 `QueryAlignmentError`；**禁止跨相机广播**；`_build_raw_query` 允许各相机 query 点数不等；提供 `camera_point_counts()/camera_positions()/select_camera()`。

### 3.8 单位与尺度契约（2026-10-14 dispatcher 裁定，回应 CFA4 的 1e8 量级失配）

T4 实测：一次 CFA4 测量步给出 `raw_proper_prediction_prequential ≈ 2.18e8`、`candidate_ce ≈ 1.06e6`，一步后 `grad_norm=nan`。这是单位/尺度不匹配，不是 loss 组合问题。裁定：

1. `RawPrediction.mean/variance` 与全部密度、量化区间、饱和尾概率计算**一律在 query 测量声明的 `units` 下进行**；`units` 不等必须拒绝（已有机制保持）。
2. 内部归一化（如 `(v-black)/(white-black)`）必须在打分前显式反变换回声明单位；仿射参数（black/white/scale）与 `calibration_version` 写入 manifest。**禁止**用归一化域预测直接对 DN 值打分。
3. `SensorModel` 的 `min_variance`/`max_variance`、量化步长（区间 ±0.5 step）、饱和阈值**按同一 unit 解释**；参数与解释写进 docstring 与配置。
4. 尺度一致性测试（T2 提供 helper、T3 集成）：对合成传感器用解析 oracle（mean=真实均值、variance=声明噪声方差，单位 DN²）打分，NLL 必须是 O(1) 且梯度有限；把预测换到归一化域必须触发尺度断言或产生可识别的巨大 NLL（作为缺陷检测）。
5. manifest 记录单位链（合成 DN / raw3 loader 单位）、典型 NLL 量级与方差分位数；正式运行前必须给出该量级证据。

### 3.9 batched CFA4 形状契约（2026-10-14 dispatcher 裁定）

`values` 张量恒为 `(B, C, H, W)`：CFA4 单行 mosaic 时 `C=1`，4 平面 packed 时 `C=4`；**batch 维永远在最前**，不得把 B≥2 的 `(B,H,W)` 当作 `(C,H,W)` 解释。T1 修复 `_frame_geometry` 的通道判定并加 B≥2 的 CFA4 测试；合成臂提速（`micro_batch_episodes>1`）依赖此契约。

## 4. 证据与回报规则（所有 owner）

1. 每个模块必须带能在真实缺陷上失败的测试（数值、因果、状态或数据边界），禁止只做 import/smoke。
2. 回报必须含：改动文件绝对路径 + `sha256sum`（本地与远端一致）、确切命令、解释器绝对路径、配置路径与哈希、数据/manifest/anchor 哈希、种子、输出目录绝对路径、原始计数、失败项与未运行项。
3. 未运行的一律写 `not_run`；不得把 forward/短试写成训练结论，不得把 energy softmax 当校准后验。
4. 触碰公共接口或数据边界时，附加 P0 回归：`make check` + 目标运行时 `tests/encoders/n02_full`（或明确子集）全绿。
5. 报告用中文，交回 dispatcher；由 dispatcher 汇总为阶段报告并做独立验收，子代理不直接向用户汇报。

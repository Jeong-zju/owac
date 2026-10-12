"""N04 训练与恢复。

训练目标（tiny 起跑配置，全部可配置）：:

    L = w_loc * L_location
      + w_fut * L_future_measurement
      + w_chg * L_change
      + w_ego * L_ego_warp

* ``L_location``：稳定表示与当前测量特征的对齐（定位/跟踪）。
* ``L_future_measurement``：**未来测量只在 loss 处读取**。用 ``t-1`` 的状态与截至
  ``t-1`` 已实际发生的运动上下文预测 ``t``，再与 ``t`` 的测量比较；比较量包含遮挡与
  饱和掩码。绝不把未来测量放进 predict 输入。
* ``L_change``：只在合成可识别标签上监督（目标掩码 × 目标是否真的在动）。真实数据
  没有该标签时关闭。
* ``L_ego_warp``：只在允许的已知几何合成数据上用真值相机 flow 监督传播。

默认 causal loop 与在线一致（逐帧 ``step``），TBPTT=8/16，边界 detach。梯度不经过任何
未声明的捷径：``freeze_propagation`` 只冻结传播模块参数，用于区分“分解是否只是依赖更好
的对应估计”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig, N04TrainConfig
from ..frameworks.n04.ego import EgoMotionContext
from ..frameworks.n04.model import N04Encoder
from ..frameworks.n04.proprio import layout_for_suite, proprio_delta
from ..frameworks.n04.synthetic import SceneConfig, SyntheticFrame, SyntheticSequenceGenerator
from ..raw.rawvla_sequence import RawVLASequence, RawVLASequenceLoader

__all__ = [
    "RawVLAFrame",
    "TrainResult",
    "build_context",
    "build_encoder",
    "build_real_context",
    "discover_npz",
    "future_measurement_loss",
    "load_checkpoint",
    "make_rawvla_batches",
    "make_synthetic_batches",
    "save_checkpoint",
    "split_npz_paths",
    "train",
]

#: 文档中记录的 RawVLA-Bench 数据部署位置（服务器）。本机不可达时用于报缺。
DOCUMENTED_RAW_DATA_ROOT = "/workspace/owac/rawvla/"


# --------------------------------------------------------------------------------------
# 上下文构造
# --------------------------------------------------------------------------------------


def build_context(
    frame: SyntheticFrame | RawVLAFrame,
    previous: SyntheticFrame | RawVLAFrame | None,
    *,
    camera_index: torch.Tensor | None = None,
    dt: float = 1.0,
) -> EgoMotionContext:
    """由**已发生**的帧构造运动上下文。

    ``proprio_delta = q_t - q_{t-1}``；第一帧没有前置状态，``*_available`` 标为缺失，
    而不是填零冒充“已发生的运动”。
    """
    if isinstance(frame, RawVLAFrame):
        return build_real_context(frame, previous, camera_index=camera_index)
    b = frame.proprio.shape[0]
    device = frame.proprio.device
    if previous is None:
        #: 第一帧没有前置本体：``proprio_delta`` 用零占位并标为不可用，
        #: 向量长度与后续帧保持一致（缺失 ≠ 测到的零运动）。
        return EgoMotionContext(
            proprio_delta=torch.zeros_like(frame.proprio),
            proprio_delta_available=False,
            executed_action=frame.action,
            executed_action_available=True,
            dt=torch.full((b,), dt, device=device),
            dt_available=False,
            time_unit="frame",
            camera_index=camera_index,
            camera_id="head",
            expected_width=frame.proprio.shape[-1] + frame.action.shape[-1] + 1,
            info={"first_frame": True},
        )
    return EgoMotionContext(
        proprio_delta=frame.proprio - previous.proprio,
        proprio_delta_available=True,
        executed_action=frame.action,
        executed_action_available=True,
        dt=torch.full((b,), dt, device=device),
        dt_available=False,  # 合成帧只有索引，没有真实时间单位，不编造秒数
        time_unit="frame",
        camera_index=camera_index,
        camera_id="head",
        expected_width=frame.proprio.shape[-1] + frame.action.shape[-1] + 1,
        info={"first_frame": False},
    )


def build_real_context(
    frame: RawVLAFrame,
    previous: RawVLAFrame | None,
    *,
    camera_index: torch.Tensor | None = None,
) -> EgoMotionContext:
    """真实 NPZ 的运动上下文：只用已发生的本体与确实记录到的已执行动作。

    * 本体相对量按通道语义计算（旋转走四元数相对旋转，见 ``proprio.py``），不是逐元素差。
    * 缓存没有可靠的已执行动作记录时，``executed_action_available=False``，零占位只用于
      保持向量宽度，绝不被解释成“测到的零动作”。
    * 只有帧索引时 ``time_unit="frame"``、``dt_available=False``，不编造秒数。
    """
    suite = str(frame.metadata.get("suite", "libero"))
    layout = layout_for_suite(suite)
    proprio_available = bool(frame.metadata.get("proprio_available", True))
    delta = proprio_delta(
        frame.proprio,
        None if previous is None else previous.proprio,
        layout=layout,
        available=proprio_available and previous is not None,
        time_unit=str(frame.metadata.get("time_unit", "frame")),
    )
    action_available = bool(frame.metadata.get("executed_action_available", False))
    return EgoMotionContext(
        proprio_delta=delta.delta,
        proprio_delta_available=bool(proprio_available and previous is not None),
        #: 宽度固定的零占位 + availability=False：缺失被显式表达，不冒充零运动。
        executed_action=frame.action,
        executed_action_available=action_available,
        dt=None,
        dt_available=False,
        time_unit=str(frame.metadata.get("time_unit", "frame")),
        camera_index=camera_index,
        camera_id=str(frame.metadata.get("camera", "")),
        expected_width=frame.proprio.shape[-1] + frame.action.shape[-1] + 1,
        info={
            "first_frame": previous is None,
            "suite": suite,
            "proprio_groups": None if delta.info is None else delta.info.get("groups"),
            "executed_action_available": action_available,
        },
    )


# --------------------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------------------


def future_measurement_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    charbonnier_eps: float = 1e-3,
) -> torch.Tensor:
    """留出的测量预测损失。

    使用 Charbonnier（平滑 L1）而不是纯 MSE：MSE 下“抹去移动目标”常常是最优解，
      而平滑 L1 对少数大幅误差（移动目标所在位置）保留梯度。遮挡/饱和位置通过
    ``target_valid`` 排除，只惩罚有真实测量支持的位置。
    """
    diff = prediction - target
    weight = target_valid
    numer = torch.sqrt(diff * diff + charbonnier_eps**2) * weight
    denom = weight.sum().clamp_min(1.0)
    return numer.sum() / denom


@dataclass
class TrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    seconds: float = 0.0
    trainable_parameters: int = 0
    total_parameters: int = 0


def _configure_threads(threads: int) -> None:
    """固定 CPU 线程数。共享机器上线程池互相抢占会让同一配置的耗时相差两个数量级，
    这既破坏可复现性也误导时延报告。"""
    if threads and threads > 0:
        torch.set_num_threads(int(threads))


def _count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


# --------------------------------------------------------------------------------------
# 合成 batch 生成
# --------------------------------------------------------------------------------------


def make_synthetic_batches(
    *,
    cfg: N04Config,
    train_cfg: N04TrainConfig,
    steps: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> list[list[SyntheticFrame]]:
    """为每个训练 step 生成一条独立条件的合成序列。"""
    conditions = (
        "camera_only",
        "object_only",
        "both",
        "illumination",
        "same_direction",
        "occlusion",
        "calibration_error",
        "action_delay",
        "exposure_step",
    )
    batches: list[list[SyntheticFrame]] = []
    scene = train_cfg.synthetic
    for i in range(steps):
        condition = conditions[i % len(conditions)]
        scene_cfg = SceneConfig(
            height=cfg.measurement_size[0],
            width=cfg.measurement_size[1],
            input_mode=cfg.input_mode,
            cfa_pattern=cfg.cfa_pattern,
            camera_translation=scene.get("camera_translation", (1.2, 1.2))
            if condition != "camera_only"
            else scene.get("camera_translation", (1.2, 1.2)),
            camera_rotation=scene.get("camera_rotation", (0.0, 0.4)),
            object_velocity=scene.get("object_velocity", (0.8, 1.2)),
            illumination=scene.get("illumination", (0.7, 1.3)),
            noise_sigma=scene.get("noise_sigma", 0.01),
            occluder=scene.get("occluder", False),
            action_delay=scene.get("action_delay", 2),
            calibration_scale=scene.get("calibration_scale", 1.08),
            seed=seed + i,
        )
        gen = SyntheticSequenceGenerator(scene_cfg)
        batches.append(
            gen.sequence(
                batch=train_cfg.batch_size,
                length=train_cfg.sequence_length,
                condition=condition,
                seed=seed + 1000 + i,
                device=device,
            )
        )
    return batches


# --------------------------------------------------------------------------------------
# 真实 RawVLA NPZ
# --------------------------------------------------------------------------------------


@dataclass
class RawVLAFrame:
    """真实 NPZ 的一帧。

    只有在线可用量有值；合成数据才有的 ``oracle_*`` / ``object_mask`` 等监督字段在这里
    显式缺席（``None``），对应 loss 分支返回 0，而不是用假标签监督。动作上下文是否可用
    由 ``metadata`` 记录，缺失时标 False，不拿未来监督动作冒充“已执行动作”。
    """

    measured: torch.Tensor  # [B,C,H,W]
    valid_mask: torch.Tensor  # [B,1,H,W]
    saturated_mask: torch.Tensor  # [B,1,H,W]
    proprio: torch.Tensor  # [B,D]
    action: torch.Tensor  # [B,A]（缺失时为占位零向量，可用性见 metadata）
    condition: str = "real"
    metadata: dict[str, Any] = field(default_factory=dict)
    true_camera_translation: torch.Tensor | None = None
    true_object_translation: torch.Tensor | None = None
    true_camera_rotation: torch.Tensor | None = None
    true_illumination: torch.Tensor | None = None
    oracle_clean: torch.Tensor | None = None
    object_mask: torch.Tensor | None = None
    oracle_camera_flow: torch.Tensor | None = None


def discover_npz(root: Path | str) -> list[Path]:
    """在目录下发现 NPZ 轨迹文件，排序保证确定性。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"找不到 NPZ 目录：{root}。文档记录的 RawVLA-Bench 数据部署在服务器 "
            f"{DOCUMENTED_RAW_DATA_ROOT}；本机不可达时如实报告缺失，不把合成数据当真实验收。"
        )
    paths = sorted(p for p in root.rglob("*.npz") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"{root} 下没有 *.npz 文件")
    return paths


def split_npz_paths(
    paths: list[Path],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    """按 episode 身份做确定性拆分。

    同一来源的 RGB / RAW / 照明衍生版本必须落在同一侧：调用方需要保证这些文件的 stem
    共享同一身份前缀。拆分只依赖身份哈希与种子，不依赖读取顺序。
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction 必须在 [0, 1) 内")
    train, val = [], []
    for path in paths:
        digest = hashlib.sha1(f"{seed}:{path.stem}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 1000
        (val if bucket < int(val_fraction * 1000) else train).append(path)
    if not val:
        #: 数据太少时不能造出空验证集：如实返回空列表，由调用方决定如何处理。
        return train, val
    if not train:
        return paths[: max(1, len(paths) - 1)], paths[max(1, len(paths) - 1) :]
    return train, val


def _normalise_measurement(values: torch.Tensor, suite: str) -> torch.Tensor:
    """按已记录的数值单位归一化测量。

    LIBERO 缓存是 uint8 pseudo-RAW，转 FP32 后除以 255；RoboTwin 是 float32
    HDR-derived RAW，范围依作者元数据，**不无条件除以 255、也不按帧最大值归一化**。
    """
    if suite == "libero":
        return values / 255.0
    if suite == "robotwin":
        return values
    raise ValueError(f"未知套件 {suite!r}")


def make_rawvla_batches(
    *,
    cfg: N04Config,
    train_cfg: N04TrainConfig,
    npz_dir: Path | str,
    suite: str,
    length: int | None = None,
    batch: int | None = None,
    steps: int | None = None,
    limit: int | None = None,
    split: str = "train",
    val_fraction: float = 0.2,
    device: torch.device | str = "cpu",
) -> list[list[RawVLAFrame]]:
    """把真实 NPZ 轨迹切成与合成路径同形的 batch 序列。

    每一步（``batches`` 的一项）是 ``batch`` 条**并行窗口**沿时间的展开，与
    `make_synthetic_batches` 完全同形，因此训练循环、TBPTT 与在线推理路径不需要分叉。
    """
    length = int(length or train_cfg.sequence_length)
    batch = int(batch or train_cfg.batch_size)
    steps = int(steps or train_cfg.steps)
    device = torch.device(device)

    paths = discover_npz(npz_dir)
    train_paths, val_paths = split_npz_paths(paths, val_fraction=val_fraction, seed=train_cfg.seed)
    chosen = train_paths if split == "train" else val_paths
    if limit is not None:
        chosen = chosen[: int(limit)]
    if not chosen:
        raise FileNotFoundError(
            f"{npz_dir} 的 {split} 侧没有轨迹（共 {len(paths)} 个 NPZ，"
            f"train={len(train_paths)} val={len(val_paths)}）"
        )

    loader = RawVLASequenceLoader(suite=suite)
    windows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in chosen:
        sequence: RawVLASequence = loader.load(path)
        #: 每个序列只用一路相机：状态按相机独立维护，跨相机融合不在本轮范围。
        if cfg.camera_ids and cfg.camera_ids[0] in sequence.measurements:
            camera = cfg.camera_ids[0]
        else:
            camera = sequence.camera_ids[0]
        summaries.append(sequence.summary() | {"split": split, "camera": camera})
        #: 分辨率必须与配置一致：stem 的 stride=1 特征网格与状态网格由配置固定，
        #: 静默缩放会改变采样相位（CFA 尤其不能这样处理）。
        measured_shape = tuple(sequence.measurements[camera].shape[-2:])
        if measured_shape != tuple(cfg.measurement_size):
            raise ValueError(
                f"{path.name} 的测量分辨率 {measured_shape} 与配置 measurement_size "
                f"{tuple(cfg.measurement_size)} 不一致；请用 --measurement-size H W 显式指定"
                f"真实分辨率（不会静默缩放测量）"
            )
        for window in RawVLASequenceLoader.windows(
            sequence, length=length, burn_in=train_cfg.burn_in
        ):
            window["camera"] = camera
            window["value_range"] = sequence.value_range
            window["source"] = str(path)
            windows.append(window)
    if len(windows) < batch:
        raise ValueError(
            f"{split} 侧只有 {len(windows)} 个窗口，少于 batch={batch}；"
            f"请减小 batch 或增加轨迹（文件数 {len(chosen)}）"
        )

    batches: list[list[RawVLAFrame]] = []
    for step in range(steps):
        picks = [windows[(step * batch + i) % len(windows)] for i in range(batch)]
        span = min(int(w["index"].numel()) for w in picks)
        frames: list[RawVLAFrame] = []
        for t in range(span):
            measured = torch.stack(
                [_normalise_measurement(w["measurements"][w["camera"]][t], suite) for w in picks]
            ).to(device)
            proprio = torch.stack([w["proprio"][t] for w in picks]).to(device)
            if all(w["executed_action"] is None for w in picks):
                action = torch.zeros(proprio.shape[0], 1, device=device)
                action_available = False
            else:
                action = torch.stack(
                    [
                        torch.zeros(w["proprio"].shape[-1])
                        if w["executed_action"] is None
                        else w["executed_action"][t]
                        for w in picks
                    ]
                ).to(device)
                action_available = all(w["executed_action"] is not None for w in picks)
            frames.append(
                RawVLAFrame(
                    measured=measured,
                    valid_mask=torch.ones(
                        measured.shape[0], 1, *measured.shape[-2:], device=device
                    ),
                    saturated_mask=_saturation_mask(measured, suite),
                    proprio=proprio,
                    action=action,
                    metadata={
                        "suite": suite,
                        "camera": picks[0]["camera"],
                        "episode_ids": [w["episode_id"] for w in picks],
                        "frame_ids": torch.stack([w["frame_ids"][t] for w in picks]).to(device),
                        "new_measurement": torch.tensor(
                            [bool(w["new_measurement"][t]) for w in picks], device=device
                        ),
                        "proprio_available": bool(all(w["availability"].proprio for w in picks)),
                        "executed_action_available": action_available,
                        "executed_action_note": (
                            "缓存没有可靠的已执行控制输入记录；在线应使用真实观测的 "
                            "joint_action/vector，这里不拿未来监督动作补齐"
                        ),
                        "time_unit": picks[0]["time_unit"],
                        "value_range": {w["episode_id"]: w.get("value_range") for w in picks},
                        "sequences": summaries[: len(chosen)],
                        "measurement_scale": "libero:uint8/255"
                        if suite == "libero"
                        else "robotwin:as-is",
                    },
                )
            )
        batches.append(frames)
    return batches


def _saturation_mask(measured: torch.Tensor, suite: str) -> torch.Tensor:
    """饱和掩码：只在白电平已知时给出，否则保持 0 并在元数据里标为未知。"""
    if suite == "libero":
        #: uint8 pseudo-RAW 的白电平是 255，归一化后 1.0；任一通道饱和即该像素饱和。
        return (measured >= 1.0).any(dim=1, keepdim=True).to(measured.dtype)
    return torch.zeros(measured.shape[0], 1, *measured.shape[-2:], device=measured.device)


# --------------------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------------------


def build_encoder(cfg: N04Config, *, baseline: str | None = None) -> nn.Module:
    """构造待训练模型。

    * ``baseline=None``：N04 候选本体。
    * ``baseline="n04_*"``：N04 的机制消融（无 ego / 无变化分支 / 全部残差当噪声 /
      上下文错位）。它们与候选共享 ``step`` 接口，因此在**同一训练预算**下可比；
      ``single_frame`` / ``convgru`` / 滤波器等外部最近基线不走这条入口（它们的接口与
      损失不同），其数字必须标注为未训练随机初始化，不能与候选直接并列。
    """
    if baseline is None:
        return N04Encoder(cfg)
    if not baseline.startswith("n04_"):
        raise ValueError(
            f"{baseline!r} 不是 N04 消融；只有共享 step 接口的 n04_* 消融能在本入口同预算训练。"
            " single_frame/convgru/motion_compensation/predict_update_filter/"
            "action_conditioned_recurrent 目前只在评测中作为未训练对照。"
        )
    from ..baselines.n04 import build_baseline

    return build_baseline(baseline, cfg)


def train(
    cfg: N04Config,
    train_cfg: N04TrainConfig,
    *,
    batches: list[list[SyntheticFrame]] | None = None,
    output_dir: Path | str | None = None,
    resume: Path | str | None = None,
    baseline: str | None = None,
) -> TrainResult:
    """跑一次训练（或从 checkpoint 恢复继续）。

    ``baseline`` 用于在同一预算下训练 N04 机制消融（``n04_*``），使“候选 vs 消融”的
    对照不受“只有候选训练过”的污染。
    """
    _configure_threads(train_cfg.threads)
    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    encoder = build_encoder(cfg, baseline=baseline).to(device)
    weights = dict(cfg.model.loss_weights)
    optimizer = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    #: 测量预测头：从状态与上下文预测下一帧测量。它属于编码器的可训练部分，
    #: 只在 loss 计算时使用（预测方向是 t-1 -> t，目标 t 不进入 predict 输入）。
    predictor_head = nn.Conv2d(cfg.model.width, cfg.channels, 3, padding=1).to(device)
    start_step = 0
    history: list[dict[str, float]] = []
    if resume is not None:
        payload = load_checkpoint(
            resume, encoder=encoder, optimizer=optimizer, predictor_head=predictor_head
        )
        start_step = int(payload.get("step", 0))
        history = list(payload.get("history", []))
    if train_cfg.freeze_propagation:
        propagator = getattr(encoder, "propagator", None)
        if propagator is None:
            raise ValueError(
                "freeze_propagation 只支持 N04 候选本体；消融的传播设置由它自己的配置决定"
            )
        for p in propagator.parameters():
            p.requires_grad_(False)
    trainable, total = _count_parameters(encoder)
    if batches is None:
        batches = make_synthetic_batches(
            cfg=cfg, train_cfg=train_cfg, steps=train_cfg.steps, seed=train_cfg.seed, device=device
        )
    for p in predictor_head.parameters():
        p.requires_grad_(True)
    head_optimizer = torch.optim.AdamW(predictor_head.parameters(), lr=train_cfg.lr)
    started = time.time()
    step = start_step
    iterator = batches[start_step:] if start_step < len(batches) else []
    #: 状态按相机名索引；真实 NPZ 的相机名来自配置（如 agentview/head），不是写死的 head。
    camera_key = cfg.camera_ids[0] if cfg.camera_ids else "head"
    for sequence in iterator:
        step += 1
        optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        #: batch 维以实际帧为准：真实 NPZ 的窗口数可能与配置名义值不同。
        states = encoder.init_state(sequence[0].measured.shape[0], device=device)
        previous: SyntheticFrame | RawVLAFrame | None = None
        totals = {k: 0.0 for k in weights}
        valid_samples = 0
        for t, frame in enumerate(sequence):
            frame = _to_device(frame, device)
            context = build_context(frame, previous, dt=1.0)
            state_before = states[camera_key]
            out = encoder.step(
                frame.measured,
                context,
                state_before,
                camera_id=camera_key,
                #: 测量侧的饱和与遮挡必须显式传入：不传等于把被遮挡像素当成新证据。
                saturation=frame.saturated_mask,
                measurement_valid=frame.valid_mask,
            )
            measured = encoder.encode_measurement(frame.measured)
            if t >= train_cfg.burn_in:
                #: 只在物理时间推进、且 burn-in 之后计算 loss，避免用状态初始化阶段
                #: 的瞬态主导梯度。
                totals["location"] += _location_loss(out.state.stable, measured)
                totals["change"] += _change_loss(out.state.change, frame)
                totals["ego_warp"] += _ego_warp_loss(out.state.forward_flow, frame, cfg)
                assert previous is not None
                totals["future_measurement"] += _future_loss(
                    predictor_head, state_before.stable, previous, frame
                )
                valid_samples += 1
            if (t + 1) % train_cfg.tbptt == 0:
                #: TBPTT 边界：切断图，保留状态值。
                states = encoder.detach_state({camera_key: out.state})
            else:
                states = {camera_key: out.state}
            previous = frame
        if valid_samples == 0:
            continue
        loss = (
            weights["location"] * totals["location"]
            + weights["future_measurement"] * totals["future_measurement"]
            + weights["change"] * totals["change"]
            + weights["ego_warp"] * totals["ego_warp"]
        ) / valid_samples
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor_head.parameters()),
                train_cfg.grad_clip,
            )
        optimizer.step()
        head_optimizer.step()
        #: 记录前 detach：带梯度的张量转标量会触发警告，且这里只需要数值。
        record = {
            k: float(v.detach() if torch.is_tensor(v) else v) / valid_samples
            for k, v in totals.items()
        }
        record["total"] = float(loss.detach())
        record["valid_samples"] = float(valid_samples)
        record["step"] = float(step)
        history.append(record)
        if train_cfg.log_every and step % train_cfg.log_every == 0:
            print(
                f"step {step:4d} total={record['total']:.5f} "
                f"loc={record['location']:.5f} fut={record['future_measurement']:.5f} "
                f"chg={record['change']:.5f} ego={record['ego_warp']:.5f}",
                flush=True,
            )
    result = TrainResult(
        history=history,
        final=history[-1] if history else {},
        steps=step,
        seconds=time.time() - started,
        trainable_parameters=trainable + sum(p.numel() for p in predictor_head.parameters()),
        total_parameters=total + sum(p.numel() for p in predictor_head.parameters()),
    )
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out_dir / "checkpoint.pt",
            encoder=encoder,
            optimizer=optimizer,
            predictor_head=predictor_head,
            cfg=cfg,
            train_cfg=train_cfg,
            step=step,
            history=history,
        )
        (out_dir / "history.json").write_text(
            json.dumps(
                {
                    "config": cfg.model.as_dict(),
                    "train": {
                        "steps": train_cfg.steps,
                        "tbptt": train_cfg.tbptt,
                        "sequence_length": train_cfg.sequence_length,
                        "batch_size": train_cfg.batch_size,
                        "lr": train_cfg.lr,
                        "seed": train_cfg.seed,
                        "freeze_propagation": train_cfg.freeze_propagation,
                    },
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return result


def _location_loss(
    stable: torch.Tensor | list[torch.Tensor], measured: torch.Tensor
) -> torch.Tensor:
    target = measured
    value = stable[0] if isinstance(stable, list) else stable
    if value.shape[1] != target.shape[1]:
        #: 宽度不同时用平均池化把测量降到特征通道数（仅用于对齐形状，不引入新信息）。
        repeat = max(1, value.shape[1] // max(1, target.shape[1]))
        target = target.repeat(1, repeat, 1, 1)[:, : value.shape[1]]
    return F.l1_loss(value, target)


def _change_loss(
    change: torch.Tensor | list[torch.Tensor] | None, frame: SyntheticFrame | RawVLAFrame
) -> torch.Tensor:
    """只在合成可识别标签上监督变化分支：目标掩码 × 目标是否真的在动。"""
    if change is None:
        return torch.zeros((), device=frame.measured.device)
    if frame.true_object_translation is None or frame.object_mask is None:
        #: 真实数据没有“目标是否在动”的标签：该分支不参与监督，而不是拿代理当真值。
        return torch.zeros((), device=frame.measured.device)
    value = change[0] if isinstance(change, list) else change
    moving = (frame.true_object_translation.abs().sum(dim=-1, keepdim=True) > 1e-6).to(value.dtype)
    mask = frame.object_mask
    if mask.shape[-2:] != value.shape[-2:]:
        mask = F.adaptive_avg_pool2d(mask, value.shape[-2:])
    target = mask * moving.reshape(-1, 1, 1, 1)
    #: 变化格只有相对幅值有意义（GroupNorm 会统一绝对尺度），因此比较**归一化后**的
    #: 空间分布：目标位置相对非目标位置应当有更高的变化能量。
    magnitude = value.abs().mean(dim=1, keepdim=True)
    scale = magnitude.mean().detach().clamp_min(1e-6)
    normalised = magnitude / scale
    target_scale = target.mean().detach().clamp_min(1e-6)
    if float(target_scale) < 1e-3:  # 该帧没有可识别标签（例如只有相机运动）
        return torch.zeros((), device=value.device, dtype=value.dtype)
    return F.l1_loss(normalised, target / target_scale)


def _ego_warp_loss(
    forward_flow: torch.Tensor | None,
    frame: SyntheticFrame | RawVLAFrame,
    cfg: N04Config,
) -> torch.Tensor:
    """只在已知几何的合成数据上，用真值相机 flow 监督传播。"""
    if forward_flow is None or frame.oracle_camera_flow is None:
        #: 真实 NPZ 没有真值 ego flow：该损失为 0，不构造假标签。
        return torch.zeros((), device=frame.measured.device)
    target = frame.oracle_camera_flow
    value = forward_flow
    if value.shape[1:3] != target.shape[1:3]:
        value = F.interpolate(
            value.permute(0, 3, 1, 2), size=target.shape[1:3], mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
    #: 真值 flow 以测量像素为单位，模型输出在特征网格像素上；按 pixel_stride 换算。
    scale = max(cfg.geometry.pixel_stride, 1e-6)
    return F.l1_loss(value, target / scale)


def _future_loss(
    head: nn.Module,
    stable: torch.Tensor | list[torch.Tensor],
    previous: SyntheticFrame,
    current: SyntheticFrame,
) -> torch.Tensor:
    """用 ``t-1`` 的状态与已发生的运动预测 ``t`` 的测量，再与 ``t`` 的真实测量比较。

    这是真正的“未来测量”监督：``previous`` 的状态只包含 ``t-1`` 及更早的信息，
    ``current.measured`` **只在 loss 处读取**，绝不进入 predict 输入。
    遮挡与饱和位置通过 ``target_valid`` 排除，不鼓励抹掉移动目标。
    """
    value = stable[0] if isinstance(stable, list) else stable
    prediction = head(value)
    target = current.measured
    if prediction.shape[-2:] != target.shape[-2:]:
        prediction = F.interpolate(
            prediction, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    if prediction.shape[1] != target.shape[1]:
        repeat = max(1, prediction.shape[1] // max(1, target.shape[1]))
        target = target.repeat(1, repeat, 1, 1)[:, : prediction.shape[1]]
    valid = (1.0 - current.saturated_mask) * current.valid_mask
    if valid.shape[-2:] != prediction.shape[-2:]:
        valid = F.adaptive_avg_pool2d(valid, prediction.shape[-2:])
    if valid.shape[1] != prediction.shape[1]:
        valid = valid.expand(-1, prediction.shape[1], -1, -1)
    return future_measurement_loss(prediction, target, valid)


def _align_state_dict(
    state_dict: dict[str, torch.Tensor], module: nn.Module
) -> dict[str, torch.Tensor]:
    """把 checkpoint 的 key 前缀对齐到目标模块。

    训练 N04 消融时被训练的模块是 ``N04Ablation``，它把真正的编码器放在 ``encoder.`` 子模块
    下；candidate 的 checkpoint 则没有这层前缀。两种 checkpoint 都要能装进两种目标，
    因此按**与目标 key 的重合数**选择变体，而不是靠 try/except 猜。
    ``load_state_dict(strict=True)`` 仍然负责最终校验：真正不匹配时照旧报错。
    """
    target = set(module.state_dict().keys())
    candidates: dict[str, dict[str, torch.Tensor]] = {"as-is": state_dict}
    stripped = {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    if stripped:
        candidates["stripped"] = stripped
    prefixed = {f"encoder.{k}": v for k, v in state_dict.items()}
    candidates["prefixed"] = prefixed
    name, best = max(candidates.items(), key=lambda item: len(target & set(item[1])))
    if len(target & set(best)) == 0 and name != "as-is":
        return state_dict
    return best


def save_checkpoint(
    path: Path,
    *,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    predictor_head: nn.Module,
    cfg: N04Config,
    train_cfg: N04TrainConfig,
    step: int,
    history: list[dict[str, float]],
) -> None:
    payload = {
        "encoder": encoder.state_dict(),
        "predictor_head": predictor_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": cfg.model.as_dict(),
        "input_mode": cfg.input_mode,
        "measurement_size": list(cfg.measurement_size),
        "cfa_pattern": cfg.cfa_pattern,
        "train_config": {
            "tbptt": train_cfg.tbptt,
            "sequence_length": train_cfg.sequence_length,
            "batch_size": train_cfg.batch_size,
            "lr": train_cfg.lr,
            "seed": train_cfg.seed,
            "freeze_propagation": train_cfg.freeze_propagation,
        },
        "step": step,
        "history": history,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    *,
    encoder: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    predictor_head: nn.Module | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if encoder is not None:
        state = _align_state_dict(payload["encoder"], encoder)
        #: 懒建的运动编码器要按 checkpoint 记录的实际宽度先建好，否则恢复会丢参数。
        #: 目标是消融 wrapper 时，真正持有编码器的是它的 ``encoder`` 子模块。
        prepare = getattr(encoder, "ensure_motion_dim_from_state_dict", None)
        if prepare is None:
            prepare = getattr(
                getattr(encoder, "encoder", None), "ensure_motion_dim_from_state_dict", None
            )
        if prepare is not None:
            #: 推理接口统一按不带前缀的 key 读取形状。
            prepare(
                {
                    (k[len("encoder.") :] if k.startswith("encoder.") else k): v
                    for k, v in state.items()
                }
            )
        encoder.load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if predictor_head is not None and "predictor_head" in payload:
        predictor_head.load_state_dict(payload["predictor_head"])
    return payload


def _to_device(
    frame: SyntheticFrame | RawVLAFrame, device: torch.device
) -> SyntheticFrame | RawVLAFrame:
    for name in (
        "measured",
        "valid_mask",
        "saturated_mask",
        "proprio",
        "action",
        "true_camera_translation",
        "true_object_translation",
        "true_camera_rotation",
        "true_illumination",
        "oracle_clean",
        "object_mask",
        "oracle_camera_flow",
    ):
        value = getattr(frame, name, None)
        #: 真实帧的 oracle 字段是 None：跳过而不是伪造一个空张量。
        if isinstance(value, torch.Tensor):
            setattr(frame, name, value.to(device))
    return frame


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def config_from_json(
    path: Path | str, train_cfg: N04TrainConfig | None = None
) -> tuple[N04Config, N04TrainConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = N04ModelConfig.from_dict(data["model"])
    geom = N04GeometryConfig(**data.get("geometry", {}))
    cfg = N04Config(
        input_mode=data.get("input_mode", "rawvla_rgb3"),
        measurement_size=tuple(data.get("measurement_size", (64, 64))),
        channels=int(data.get("channels", 3)),
        cfa_pattern=data.get("cfa_pattern", "RGGB"),
        camera_ids=tuple(data.get("camera_ids", ("head",))),
        geometry=geom,
        model=model,
        seed=int(data.get("seed", 0)),
    )
    train = train_cfg or N04TrainConfig(**data.get("train", {}))
    return cfg, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N04 编码器训练")
    parser.add_argument("--config", required=True, help="编码器配置 JSON")
    parser.add_argument("--synthetic", action="store_true", help="使用合成可控测量场")
    parser.add_argument("--steps", type=int, default=None, help="训练步数")
    parser.add_argument("--output", default="outputs/n04")
    parser.add_argument("--resume", default=None, help="恢复 checkpoint 路径")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--freeze-propagation", action="store_true")
    parser.add_argument("--tbptt", type=int, default=None)
    parser.add_argument(
        "--npz-dir", default=None, help="真实 RawVLA NPZ 目录（与 --synthetic 互斥）"
    )
    parser.add_argument("--suite", default="libero", choices=["libero", "robotwin"])
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="按 episode 身份的验证集比例"
    )
    parser.add_argument("--npz-limit", type=int, default=None, help="只取前 N 条轨迹（先导 smoke）")
    parser.add_argument(
        "--baseline",
        default=None,
        help="在同一预算下训练 N04 机制消融（n04_no_ego / n04_no_change_branch / "
        "n04_all_residual_noise / n04_time_shifted_context）",
    )
    parser.add_argument(
        "--measurement-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="真实数据的测量分辨率；同时覆盖 geometry.feature_size（stride=1）",
    )
    args = parser.parse_args(argv)

    cfg, train_cfg = config_from_json(args.config)
    overrides: dict[str, Any] = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.tbptt is not None:
        overrides["tbptt"] = args.tbptt
    if args.freeze_propagation:
        overrides["freeze_propagation"] = True
    if overrides:
        train_cfg = N04TrainConfig(**{**train_cfg.__dict__, **overrides})
    if args.measurement_size is not None:
        height, width = (int(v) for v in args.measurement_size)
        cfg = replace(
            cfg,
            measurement_size=(height, width),
            geometry=replace(cfg.geometry, feature_size=(height, width)),
        )
    if args.synthetic and args.npz_dir:
        parser.error("--synthetic 与 --npz-dir 互斥：合成数据不是真实验收")
    if not args.synthetic and not args.npz_dir:
        parser.error("必须指定 --synthetic 或 --npz-dir <真实 NPZ 目录>")

    batches = None
    data_summary: dict[str, Any] = {"source": "synthetic"}
    if args.npz_dir:
        try:
            batches = make_rawvla_batches(
                cfg=cfg,
                train_cfg=train_cfg,
                npz_dir=args.npz_dir,
                suite=args.suite,
                limit=args.npz_limit,
                split=args.split,
                val_fraction=args.val_fraction,
                device=train_cfg.device,
            )
        except FileNotFoundError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
        data_summary = {
            "source": "rawvla_npz",
            "npz_dir": str(args.npz_dir),
            "suite": args.suite,
            "split": args.split,
            "val_fraction": args.val_fraction,
            "limit": args.npz_limit,
            "sequences": batches[0][0].metadata.get("sequences", []),
            "measurement_scale": batches[0][0].metadata.get("measurement_scale"),
            "executed_action_available": batches[0][0].metadata.get("executed_action_available"),
            "note": "缓存没有已执行动作记录时动作上下文标为缺失；不改用未来监督动作",
        }
    result = train(
        cfg,
        train_cfg,
        batches=batches,
        output_dir=args.output,
        resume=args.resume,
        baseline=args.baseline,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(
        json.dumps(data_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "steps": result.steps,
                "seconds": round(result.seconds, 3),
                "trainable_parameters": result.trainable_parameters,
                "total_parameters": result.total_parameters,
                "final": result.final,
                "data": data_summary["source"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

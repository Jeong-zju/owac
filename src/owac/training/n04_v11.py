"""N04 v1.1 S2 训练入口：官方 RAW 窗口 + 真实 π0.5 动作流匹配。

CLI（只保留常用参数）::

    python -m owac.training.n04_v11 --config configs/n04/v1_1/runs/libero_D.json \
        --data-root /workspace/.../RawVLA-Bench/data --output runs/v1.1/s2/libero-D-seed101 \
        --seed 101 --device cuda:0 [--resume <ckpt>]

约定：

* 窗口 16 burn-in + 32 梯度帧，每窗口均匀 4 个动作监督点；有效 batch 8 段序列，
  ``optim.micro_batch`` 用梯度累积实现。
* 一次新观测只融合一次。非监督帧走 ``encoder.step + readout_tokens``（与
  `RawVLAFeaturePolicy.encode_observation` 同一调用，但保留契约 §5 的 ``StepOutput``
  供辅助损失读取）；监督帧走 ``policy.forward_from_observation``（真实 π0.5
  flow-matching，动作梯度经 prefix 回到编码器）。
* burn-in 用当前模型在真实历史上生成状态、不跨 episode 继承、不保留梯度。
* 网络 BF16，损失／协方差／NLL FP32；seed 在构造模型之前设置。
* 每 ``monitor.log_every`` 步写 ``metrics.jsonl`` 与 ``progress.json`` 心跳（``os.replace``
  原子写），每 ``monitor.checkpoint_every`` 步存 checkpoint，固定少量 dev 窗口评动作 loss。
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import random
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..backends.rawvla_features import (
    FeatureBridgeConfig,
    RawVLAFeaturePolicy,
    tokenize_backend_task,
)
from ..core.encoder_contract import EmbodimentContext
from ..frameworks.n04.compatibility import (
    IMPLEMENTATION_VERSION,
    LOSS_DEFINITION,
    require_current_implementation,
)
from ..raw.n04_v11_data import (
    PLATFORM_SPECS,
    ActionNormalizer,
    V11DataIndex,
    WindowBatch,
    load_window_batch,
    sample_windows,
    saturation_mask,
    window_candidates,
)

__all__ = [
    "DEFAULT_LOSS_WEIGHTS",
    "PROGRESS_FIELDS",
    "V11Trainer",
    "build_optimizer",
    "build_pi05_backend",
    "build_v11_encoder",
    "forecast_nll",
    "load_config",
    "main",
    "masked_mean",
    "summarize_eval_results",
    "vicreg_terms",
    "weighted_total",
    "write_json_atomic",
    "write_progress",
]

#: 契约 §6 的外层权重。
DEFAULT_LOSS_WEIGHTS = {
    "action": 1.0,
    "correspondence": 0.1,
    "forecast": 0.1,
    "representation": 0.01,
}

#: 契约 §7 的进度字段；未产生的值写 null，不用 0 冒充。
PROGRESS_FIELDS = (
    "stage",
    "platform",
    "arm",
    "seed",
    "phase",
    "updated_at",
    "step",
    "budget_steps",
    "train_loss",
    "dev_loss",
    "step_seconds",
    "gpu_memory_gb",
    "eval",
    "artifacts",
    "log_tail",
)

#: 预测目标使用同一 RAW 编码器的停止梯度分支，取后续 1、2、4 个记录步。
FORECAST_HORIZONS = (1, 2, 4)

#: 允许透传给 WS-C `EncoderV11Config` 的字段；其余键留在 JSON 里但不进构造函数。
_ENCODER_CONFIG_FIELDS = frozenset(
    {
        "arm",
        "cameras",
        "input_mode",
        "scale_name",
        "saturation_level",
        "in_channels",
        "supports_per_camera",
        "obs_dim",
        "proprio_dim",
        "control_hidden",
        "trajectory_len",
        "collect_readout_attention",
        "allow_duplicate_step",
        "state_var_init",
        "transition_init",
        "measurement_padding_mode",
        "revision_variance_init",
        "raft_iters",
        "raft_resize_hw",
    }
)

#: 结构指纹：这些变了必须拒绝 resume（模型／优化器／平台身份）
STRUCTURAL_FINGERPRINT_KEYS = ("stage", "platform", "arm", "model", "optim")
#: 损失定义指纹：允许有意变更（如 forecast 归约口径升级），但必须记录 from/to
LOSS_FINGERPRINT_KEYS = ("loss_definition", "loss")
#: 旧 checkpoint 里的一次性组合指纹（读取时兼容）
CHECKPOINT_FINGERPRINT_KEYS = ("stage", "platform", "arm", "model", "loss", "optim")


# --------------------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, *, data_root: str | None = None) -> dict[str, Any]:
    """读 JSON 配置并展开 ``include``（相对被包含文件所在目录）。"""

    def load_one(target: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
        target = target.resolve()
        if target in seen:
            raise ValueError(f"配置 include 成环：{target}")
        payload = json.loads(target.read_text())
        includes = payload.pop("include", [])
        merged: dict[str, Any] = {}
        for item in includes:
            merged = _deep_merge(merged, load_one(target.parent / item, seen + (target,)))
        return _deep_merge(merged, payload)

    config = load_one(Path(path), ())
    if data_root:
        config.setdefault("data", {})["root"] = str(data_root)
    for section in ("data", "model", "loss", "optim", "monitor"):
        if section not in config:
            raise ValueError(f"配置缺少 {section!r} 段：{path}")
    return config


def _fingerprint(config: dict[str, Any], keys: Sequence[str]) -> str:
    payload = json.dumps({key: config.get(key) for key in keys}, sort_keys=True, ensure_ascii=False)
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    """旧的一次性指纹（保留给历史 checkpoint 读取，不再作为 resume 判据）。"""
    return _fingerprint(config, CHECKPOINT_FINGERPRINT_KEYS)


def structural_fingerprint(config: dict[str, Any]) -> str:
    return _fingerprint(config, STRUCTURAL_FINGERPRINT_KEYS)


def loss_fingerprint(config: dict[str, Any]) -> str:
    return _fingerprint(config, LOSS_FINGERPRINT_KEYS)


# --------------------------------------------------------------------------------------
# 损失（纯函数，便于测试）
# --------------------------------------------------------------------------------------


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor | None = None, *, eps: float = 1e-6
) -> torch.Tensor:
    """先按有效样本取均值：``sum(values*mask) / max(sum(mask), eps)``。"""
    values = values.to(torch.float32)
    if mask is None:
        return values.mean()
    weights = mask.to(torch.float32)
    if weights.shape != values.shape:
        weights = weights.expand_as(values)
    total = weights.sum()
    if float(total) <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return (values * weights).sum() / total.clamp_min(eps)


def weighted_total(
    terms: dict[str, torch.Tensor | None],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor | None, dict[str, float | None]]:
    """按外层权重合成总损失；缺失项记 ``None``（不加 0 冒充）。"""
    weights = dict(weights or DEFAULT_LOSS_WEIGHTS)
    report: dict[str, float | None] = {name: None for name in weights}
    total: torch.Tensor | None = None
    for name, weight in weights.items():
        value = terms.get(name)
        if value is None:
            continue
        value = value.to(torch.float32).reshape(())
        report[name] = float(value.detach().cpu())
        total = value * float(weight) if total is None else total + value * float(weight)
    return total, report


def vicreg_terms(
    features: torch.Tensor, *, max_rows: int = 4096, eps: float = 1e-4
) -> dict[str, torch.Tensor]:
    """VICReg 方差／协方差项（只作为已有防塌缩依据，不承诺物理语义）。"""
    width = int(features.shape[-1])
    rows = features.to(torch.float32).reshape(-1, width)
    if rows.shape[0] > max_rows:
        index = torch.linspace(0, rows.shape[0] - 1, max_rows).long().to(rows.device)
        rows = rows[index]
    centered = rows - rows.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.square().mean(dim=0) + eps)
    variance = torch.relu(1.0 - std).mean()
    if centered.shape[0] < 2:
        return {"variance": variance, "covariance": torch.zeros_like(variance)}
    denominator = centered.shape[0] - 1
    if centered.shape[0] < width:
        # ||X^T X||_F² = ||X X^T||_F²：prefix 为 512/768×2048 时选更小的 Gram。
        # 减去同一协方差的对角平方和，保留 VICReg 原目标；clamp 仅消除舍入负零。
        gram = (centered @ centered.transpose(0, 1)) / denominator
        diagonal = centered.square().sum(dim=0) / denominator
        off_diagonal_sum = (gram.square().sum() - diagonal.square().sum()).clamp_min(0.0)
    else:
        covariance = (centered.transpose(0, 1) @ centered) / denominator
        off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
        off_diagonal_sum = off_diagonal.square().sum()
    return {"variance": variance, "covariance": off_diagonal_sum / width}


def observation_vicreg_terms(features: torch.Tensor, eps: float = 1e-4) -> dict[str, torch.Tensor]:
    """[observations, fixed query, feature]: learned queries are never observations.

    Per-query covariance is computed with the smaller observation Gram matrix. Identical
    prefixes across frames incur a variance penalty even if queries differ arbitrarily.
    """
    if features.ndim != 3 or features.shape[0] < 2:
        raise ValueError("observation VICReg requires [T>=2, queries, features]")
    x = features.float()
    centered = (x - x.mean(dim=0, keepdim=True)).permute(1, 0, 2)
    count, width = x.shape[0], x.shape[-1]
    std = torch.sqrt(centered.square().sum(dim=1) / (count - 1) + eps)
    variance = torch.relu(1 - std).mean()
    gram = centered @ centered.transpose(-1, -2) / (count - 1)
    diagonal = centered.square().sum(dim=1) / (count - 1)
    covariance = (gram.square().sum((-1, -2)) - diagonal.square().sum(-1)).clamp_min(
        0
    ).mean() / width
    return {"variance": variance, "covariance": covariance}


def summarize_prefix(values: torch.Tensor, cameras: int) -> torch.Tensor:
    """One scene descriptor per camera; learned query identities are not samples."""
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] % cameras:
        raise ValueError("representation requires one complete multi-camera prefix")
    return values.float().reshape(cameras, -1, values.shape[-1]).mean(dim=1)


@dataclass
class RepresentationGradientCache:
    features: torch.Tensor
    gradients: torch.Tensor
    loss: torch.Tensor
    diagnostics: dict[str, float]

    def replay(self, index: int, live: torch.Tensor) -> torch.Tensor:
        """Exact batch-objective VJP under serial 1/B window gradient accumulation.

        The reported value remains the actual batch loss. The linear term supplies
        its derivative, rather than treating detached scenes as extra live samples.
        """
        reference = self.features[index].to(live)
        error = (live.detach() - reference).square().mean().sqrt()
        tolerance = 0.002 + 0.005 * reference.square().mean().sqrt()
        if not bool(error <= tolerance):
            raise RuntimeError("representation replay differs from cached observations")
        linear = (live * self.gradients[index].to(live)).sum() * len(self.features)
        return self.loss.to(live) + (linear - linear.detach())


def cache_representation_gradients(
    observations: torch.Tensor, *, variance_weight: float = 25.0, covariance_weight: float = 1.0
) -> RepresentationGradientCache:
    """VICReg across independent windows, preserving each camera's identity.

    25/1 are the variance/covariance coefficients of the reference VICReg recipe;
    the declared external representation weight is applied once by the trainer.
    """
    with torch.enable_grad():
        features = observations.detach().float().requires_grad_(True)
        terms = observation_vicreg_terms(features)
        loss = variance_weight * terms["variance"] + covariance_weight * terms["covariance"]
        gradient = torch.autograd.grad(loss, features)[0]
    if not bool(torch.isfinite(loss) & torch.isfinite(gradient).all()):
        raise FloatingPointError("non-finite independent-window representation objective")
    return RepresentationGradientCache(
        features=features.detach(),
        gradients=gradient.detach(),
        loss=loss.detach(),
        diagnostics={
            "variance": float(terms["variance"].detach()),
            "covariance": float(terms["covariance"].detach()),
            "std": float(features.detach().std(dim=0).mean()),
            "windows": int(features.shape[0]),
        },
    )


def configure_compiled_autograd() -> dict[str, Any]:
    """AOTAutograd must support exact per-term fallback and our FP32 backward context.

    PyTorch documents donated_buffer=False for retain_graph=True and backward_pass_autocast
    ='off' for an autocast forward whose backward executes outside autocast.
    https://github.com/pytorch/pytorch/blob/main/torch/_functorch/config.py
    """
    from torch._functorch import config as aot_config

    aot_config.donated_buffer = False
    aot_config.backward_pass_autocast = "off"
    return {"donated_buffer": False, "backward_pass_autocast": "off"}


def forecast_nll(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    variance_floor: float = 1e-6,
) -> torch.Tensor:
    """Gaussian NLL：保留完整二次项与 log 方差，不做截断（契约 §5）。"""
    var = variance.to(torch.float32).clamp_min(variance_floor)
    residual = (target.to(torch.float32) - mean.to(torch.float32)).square()
    nll = 0.5 * (residual / var + var.log() + math.log(2 * math.pi))
    return masked_mean(nll, mask)


def _debug_dict(output: Any) -> dict[str, Any]:
    debug = getattr(output, "debug", None)
    return debug if isinstance(debug, dict) else {}


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _scalar_terms(output: Any) -> dict[str, torch.Tensor]:
    """编码器直接给出的辅助标量（已按有效样本／维度取均值）。"""
    candidates = [getattr(output, "losses", None), _debug_dict(output).get("losses")]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return {
                str(name): value
                for name, value in candidate.items()
                if isinstance(value, torch.Tensor)
            }
    return {}


def _posterior_state(output: Any, camera: str) -> Any | None:
    """t0 的**后验**支持状态（``StepOutput.state.per_camera[camera]``）；A/C 臂返回 ``None``。

    预测从后验起、按 horizon roll k 步（见 `ForecastEngine.add`），不再使用 ``forecast_inputs``
    里那个比 t0 早一帧的先验。
    """
    state = output.get("state") if isinstance(output, dict) else getattr(output, "state", None)
    per_camera = getattr(state, "per_camera", None)
    if not isinstance(per_camera, dict):
        return None
    return per_camera.get(camera)


def _quality_variance(output: Any, camera: str) -> torch.Tensor | None:
    """t 时刻的测量方差 R（质量头），只在预测侧使用；目标时刻的 R 不读。"""
    inputs = getattr(output, "forecast_inputs", {})
    live = inputs.get(camera, {}).get("R_diag")
    if isinstance(live, torch.Tensor):
        return live
    debug = _debug_dict(output)
    quality = debug.get("quality")
    if not isinstance(quality, dict):
        return None
    entry = quality.get(camera)
    if not isinstance(entry, dict):
        return None
    R = entry.get("R_diag")
    return R if isinstance(R, torch.Tensor) else None


@dataclass
class _ForecastAnchor:
    """t0 时刻发出的一个 k 步预测。

    因果边界：``prediction``（预测到 t0+k 的 prior）、``R``、``xy_px`` 的起点、roll 用的本体
    都只来自 t0；``xy_px`` 之后按**真实 flow** 输运（no_grad，只决定"看哪里"）；target 是
    t0+k 帧同一 RAW 编码器特征的 stop-gradient 采样（决定"预测什么内容"，只作损失目标）。
    """

    position: int
    horizon: int
    camera: str
    ids: torch.Tensor
    xy_px: torch.Tensor
    valid: torch.Tensor
    #: t0 的活后验：源段的图延迟至本段最后一个预测结算后释放
    posterior: Any
    #: t0 的本体条件（detach，hold 住逐 k 步推进；绝不读 t0 之后的本体）
    proprio: torch.Tensor
    R: torch.Tensor
    segment_last: int = 0
    proprio_valid: torch.Tensor | None = None
    prediction: Any = None
    reason: str | None = None


class ForecastEngine:
    """多步 forecast 状态机：预测侧不读未来，输运读真实 flow 但只用于定位。"""

    def __init__(
        self,
        estimator: Any,
        horizons: Sequence[int] = FORECAST_HORIZONS,
        *,
        total_frames: int,
        weight: float = 0.0,
        diagnostics_every: int = 1,
        max_innovation_variance: float | None = None,
    ) -> None:
        self.estimator = estimator
        #: Historical API field; v3 gradients are normalized by the trainer after valid counts.
        self.weight = float(weight)
        #: 仪表节流：每 N 个 settle 记一次（只影响诊断记录，不影响数值与梯度）
        self.diagnostics_every = max(1, int(diagnostics_every))
        self._settle_count = 0
        self.max_innovation_variance = max_innovation_variance
        if max_innovation_variance is not None and (
            not math.isfinite(max_innovation_variance) or max_innovation_variance <= 0
        ):
            raise ValueError("max_innovation_variance must be positive and finite")
        self.horizons = tuple(int(h) for h in horizons)
        if (
            not self.horizons
            or any(h <= 0 for h in self.horizons)
            or len(set(self.horizons)) != len(self.horizons)
        ):
            raise ValueError("forecast horizons must be positive and unique")
        self.total_frames = int(total_frames)
        self.anchors: list[_ForecastAnchor] = []
        #: 每个 horizon 的结算前仪表累计（quad/logdet/r_rms/S/R；除以 count 得到均值）
        self.diagnostics: dict[int, dict[str, float]] = {}
        self.created = 0
        self.settled = 0
        self.skipped = 0
        self.skip_reasons: dict[str, int] = defaultdict(int)
        self._missing_transport: set[tuple[int, str | None]] = set()
        self._roll_cache: dict[tuple[int, str], tuple[int, Any]] = {}

    # -- 预测侧（只用 ≤ t0 的信息） -------------------------------------------------

    def add(
        self,
        position: int,
        camera: str,
        posterior: Any,
        R: torch.Tensor,
        proprio_context: torch.Tensor,
        *,
        size_hw: tuple[int, int],
        segment_last: int,
        proprio_valid: torch.Tensor | None = None,
    ) -> int:
        """记录活的 t0 后验与 R；只对输入本体和定位坐标停梯度。

        只跳过窗口之外的目标。跨 TBPTT 段的源图由训练器保留至未来目标到达，
        目标特征停梯度，未来本体和专家动作不进入预测侧。
        """
        if posterior is None:
            return 0
        created = 0
        for horizon in self.horizons:
            if position + horizon >= self.total_frames:
                self._skip("window_too_short")
                continue
            self.anchors.append(
                _ForecastAnchor(
                    position=int(position),
                    horizon=int(horizon),
                    camera=str(camera),
                    ids=posterior.ids.detach().clone(),
                    xy_px=posterior.xy.detach().clone(),
                    valid=(posterior.valid & ~posterior.fresh).detach().clone(),
                    posterior=posterior,
                    proprio=proprio_context.detach().clone(),
                    R=R,
                    segment_last=int(segment_last),
                    proprio_valid=proprio_valid,
                )
            )
            created += 1
        self.created += created
        return created

    def skip_all(self, reason: str) -> None:
        """整批 horizon 都不可用（例如本帧没有预测侧输入）时显式计数。"""
        for _ in self.horizons:
            self._skip(reason)

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] += 1

    # -- 输运（真实 flow，no_grad，只决定看哪里） ------------------------------------

    def mark_missing_transport(self, position: int, camera: str | None = None) -> None:
        """本帧没有可用 flow（例如监督帧走的是桥路径）：受影响锚点显式跳过。"""
        self._missing_transport.add((int(position), camera))

    def advance(self, flow_px: torch.Tensor, xy_px: torch.Tensor) -> torch.Tensor:
        """把一组 xy 沿当前帧的 flow 推进一步（no_grad）。"""
        from ..frameworks.n04.correspondence import flow_at_points

        with torch.no_grad():
            step = flow_at_points(flow_px.detach(), xy_px.detach())
        return xy_px + step

    def transport(
        self,
        camera: str,
        flow_px: torch.Tensor,
        size_hw: tuple[int, int],
        visible: torch.Tensor | None = None,
    ) -> None:
        for anchor in self.anchors:
            if anchor.camera != camera or anchor.reason is not None:
                continue
            if visible is not None:
                from ..frameworks.n04.correspondence import flow_at_points

                support = flow_at_points(
                    visible[:, None].float().expand(-1, 2, -1, -1), anchor.xy_px
                )
                anchor.valid = anchor.valid & (support[..., 0] > 0.5)
            anchor.xy_px = self.advance(flow_px, anchor.xy_px)
            height, width = float(size_hw[0]), float(size_hw[1])
            inside = (
                (anchor.xy_px[..., 0] >= 0)
                & (anchor.xy_px[..., 0] <= width - 1)
                & (anchor.xy_px[..., 1] >= 0)
                & (anchor.xy_px[..., 1] <= height - 1)
            )
            anchor.valid = anchor.valid & inside

    # -- 结算（target 停止梯度） ----------------------------------------------------

    def due(self, position: int) -> list[_ForecastAnchor]:
        due = [
            anchor
            for anchor in self.anchors
            if anchor.reason is None and anchor.position + anchor.horizon == int(position)
        ]
        return due

    def _roll(self, anchor: _ForecastAnchor) -> Any:
        """Share the same causal 1→2→4 rollout across horizons from one source."""
        key = (anchor.position, anchor.camera)
        reached, state = self._roll_cache.get(key, (0, anchor.posterior))
        if reached > anchor.horizon:
            reached, state = 0, anchor.posterior
        for _ in range(reached, anchor.horizon):
            kwargs = {} if anchor.proprio_valid is None else {"proprio_valid": anchor.proprio_valid}
            state = self.estimator.predict(state, anchor.proprio, **kwargs)
        self._roll_cache[key] = (anchor.horizon, state)
        return state

    def settle(self, anchor: _ForecastAnchor, target: torch.Tensor | None) -> torch.Tensor | None:
        """在 t0+k 结算并返回活 loss，由源段统一反传；目标严格停梯度。"""
        from ..frameworks.n04.evidence_update import forecast_loss

        if any(
            (pos, None) in self._missing_transport
            or (pos, anchor.camera) in self._missing_transport
            for pos in range(anchor.position + 1, anchor.position + anchor.horizon + 1)
        ):
            anchor.reason = "missing_transport"
            self._skip("missing_transport")
            return None
        if target is None or not bool(anchor.valid.any()):
            anchor.reason = "no_valid_support"
            self._skip("no_valid_support")
            return None
        prediction = self._roll(anchor)
        anchor.prediction = prediction
        if self.max_innovation_variance is not None:
            variance = prediction.mode_cov[0] + anchor.R[:, :, None, :]
            mask = anchor.valid & prediction.valid
            if bool(mask.any()) and bool(variance[mask].amax() > self.max_innovation_variance):
                raise FloatingPointError(
                    "forecast innovation variance exceeds configured stop limit"
                )
        self._settle_count += 1
        if self._settle_count % self.diagnostics_every == 0:
            self._instrument(anchor, target)
        loss = forecast_loss(target, prediction, anchor.R, valid=anchor.valid)
        # 诊断只留值；活 loss 及源后验的图由训练器在源段结算后释放。
        anchor.prediction = _detach_state(prediction)
        anchor.reason = "settled"
        self.settled += 1
        return loss

    @torch.no_grad()
    def _instrument(self, anchor: _ForecastAnchor, target: torch.Tensor) -> None:
        """结算前用 WS-B 的只读仪表：quad/logdet/模式权重/S/R 谱（no_grad，不进图）。

        判据（与 WS-B 约定，d=256）：``quad≈d`` 残差增长、``quad≫d`` 方差跟不上、``quad≈0``
        且 logdet 大 = 方差膨胀。分母为空时 WS-B 返回 NaN，写盘时按约定转 null，不写 0。
        """
        from ..frameworks.n04.evidence_update import forecast_diagnostics

        report = forecast_diagnostics(target, anchor.prediction, anchor.R, valid=anchor.valid)
        dimension = anchor.prediction.obs_dim
        prediction = (
            anchor.prediction.mode_mean[..., :dimension] * anchor.prediction.mode_weights[..., None]
        ).sum(dim=2)
        persistence = (
            anchor.posterior.mode_mean[..., :dimension] * anchor.posterior.mode_weights[..., None]
        ).sum(dim=2)
        mask = anchor.valid & anchor.prediction.valid
        if bool(mask.any()):
            pred_error = (prediction - target).square().mean(-1)[mask].mean()
            hold_error = (persistence - target).square().mean(-1)[mask].mean()
            report["prediction_mse"] = float(pred_error)
            report["persistence_mse"] = float(hold_error)
            report["target_rms"] = float(target[mask].square().mean().sqrt())
        entries = self.diagnostics.setdefault(anchor.horizon, {"count": 0.0})
        entries["count"] += 1.0
        for key, value in report.items():
            entries[key] = entries.get(key, 0.0) + float(value)

    def report(self) -> dict[str, Any]:
        diagnostics: dict[str, dict[str, float]] = {}
        for horizon, entries in sorted(self.diagnostics.items()):
            count = max(1.0, entries.get("count", 0.0))
            diagnostics[str(horizon)] = {
                key: (float(value) / count if count else float(value))
                for key, value in entries.items()
                if key != "count"
            }
            diagnostics[str(horizon)]["count"] = float(entries.get("count", 0.0))
        return {
            "created": self.created,
            "settled": self.settled,
            "skipped": self.skipped,
            "skip_reasons": dict(self.skip_reasons),
            "diagnostics": diagnostics,
        }


# --------------------------------------------------------------------------------------
# 优化器：分组学习率 + LoRA
# --------------------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """rank=16 的低秩增量；基座权重冻结，只有 A/B 可训练。"""

    def __init__(self, base: nn.Linear, *, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scale

    # 冻结的后端代码会直接读 ``...q_proj.weight.dtype``；包装后保持同一属性面。
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    targets: Sequence[str] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
) -> dict[str, Any]:
    """把命中的 ``nn.Linear`` 换成 LoRA 包装；返回被替换的模块名。"""
    matches: list[tuple[nn.Module, str]] = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(
                target in full_name or target in child_name for target in targets
            ):
                matches.append((module, child_name))
    for parent, child_name in matches:
        child = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
    return {
        "targets": list(targets),
        "rank": int(rank),
        "alpha": float(alpha),
        "wrapped": [f"{parent.__class__.__name__}.{name}" for parent, name in matches],
    }


def freeze_parameters(
    model: nn.Module,
    *,
    trainable_prefixes: Sequence[str],
    keep_substrings: Sequence[str] = ("lora_",),
) -> dict[str, Any]:
    """冻结预训练参数（``requires_grad=False``）；只放开动作专家与 LoRA。"""
    model.requires_grad_(False)
    trainable, frozen = [], 0
    for name, parameter in model.named_parameters():
        wanted = name.startswith(tuple(trainable_prefixes)) or any(
            token in name for token in keep_substrings
        )
        parameter.requires_grad_(wanted)
        if wanted:
            trainable.append(name)
        else:
            frozen += 1
    return {"trainable": trainable, "trainable_tensors": len(trainable), "frozen_tensors": frozen}


def align_optimizer_gradients(optimizer: torch.optim.Optimizer) -> int:
    """Match parameter strides after manual accumulation of compiled gradients.

    Compiled convolution backward can return channels-last gradients for NCHW
    parameters. Fused AdamW requires parameter, gradient and moments to share
    layout; changing strides here leaves values and the objective unchanged.
    """
    aligned = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and parameter.grad.stride() != parameter.stride():
                gradient = torch.empty_like(parameter, memory_format=torch.preserve_format)
                gradient.copy_(parameter.grad)
                parameter.grad = gradient
                aligned += 1
    return aligned


def build_optimizer(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    groups: Sequence[dict[str, Any]],
    betas: Sequence[float] = (0.9, 0.95),
    decay_floor_ndim: int = 2,
    unmatched: str = "pi05_adapt",
    fused: bool = False,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """按名字前缀分组：``groups=[{name, lr, weight_decay, patterns}]``。

    新增模块 ``weight_decay=0.05``；bias／归一化（``ndim<=1``）不加权重衰减。未命中任何
    分组的可训练参数落到 ``unmatched`` 指定的分组（默认最低学习率的 π0.5 分组），并记入报告。
    """
    by_name: dict[str, list[nn.Parameter]] = {group["name"]: [] for group in groups}
    decay_settings: dict[str, float] = {
        group["name"]: float(group["weight_decay"]) for group in groups
    }
    if unmatched not in by_name:
        raise ValueError(f"unmatched 分组 {unmatched!r} 不在 optim.groups 里")
    unmatched_names: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        hit = None
        for group in groups:
            if any(pattern in name for pattern in group.get("patterns", [])):
                hit = group["name"]
                break
        if hit is None:
            hit = unmatched
            unmatched_names.append(name)
        by_name[hit].append(parameter)
    param_groups: list[dict[str, Any]] = []
    report: dict[str, Any] = {"groups": {}, "unmatched_parameters": unmatched_names[:20]}
    for group in groups:
        name = group["name"]
        parameters = by_name[name]
        if not parameters:
            report["groups"][name] = {"lr": float(group["lr"]), "tensors": 0, "params": 0}
            continue
        decay = [p for p in parameters if p.ndim >= decay_floor_ndim]
        no_decay = [p for p in parameters if p.ndim < decay_floor_ndim]
        if decay:
            param_groups.append(
                {
                    "name": f"{name}:decay",
                    "params": decay,
                    "lr": float(group["lr"]),
                    "weight_decay": float(decay_settings[name]),
                }
            )
        if no_decay:
            param_groups.append(
                {
                    "name": f"{name}:no_decay",
                    "params": no_decay,
                    "lr": float(group["lr"]),
                    "weight_decay": 0.0,
                }
            )
        report["groups"][name] = {
            "lr": float(group["lr"]),
            "weight_decay": float(decay_settings[name]),
            "tensors": len(parameters),
            "params": int(sum(p.numel() for p in parameters)),
        }
    if not param_groups:
        raise ValueError("没有可训练参数：检查 optim.groups 与模型冻结设置")
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(float(b) for b in betas), fused=fused)
    report["n_param_groups"] = len(param_groups)
    return optimizer, report


# --------------------------------------------------------------------------------------
# 监控输出（WS-E 只读）
# --------------------------------------------------------------------------------------


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """临时文件 + ``os.replace``：读者永远不会看到写了一半的 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_eval_results(path: str | Path, *, planned: int | None = None) -> dict[str, Any]:
    """按平台统计闭环结果：成功／失败（含超时）／基础设施错误分开计数。

    同一 ``identity`` 只按最后一行结算：基础设施错误补跑成功后不再重复计入错误数。
    """
    rows = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        latest[str(row.get("identity") or f"row:{index}")] = row
    rows = list(latest.values())
    summary: dict[str, Any] = {
        "planned": planned,
        "completed": 0,
        "successes": 0,
        "failures": 0,
        "errors": 0,
        "status": "not_started" if not rows else "running",
    }
    per_platform: dict[str, dict[str, int]] = {}
    for row in rows:
        platform = str(row.get("platform", "unknown"))
        bucket = per_platform.setdefault(
            platform, {"completed": 0, "successes": 0, "failures": 0, "errors": 0}
        )
        status = row.get("status")
        if status in ("success", "failure"):
            summary["completed"] += 1
            bucket["completed"] += 1
            if status == "success":
                summary["successes"] += 1
                bucket["successes"] += 1
            else:
                summary["failures"] += 1
                bucket["failures"] += 1
        elif status == "infrastructure_error":
            summary["errors"] += 1
            bucket["errors"] += 1
    for name in PLATFORM_SPECS:
        summary[name] = per_platform.get(
            name, {"completed": 0, "successes": 0, "failures": 0, "errors": 0}
        )
    if planned is not None and summary["completed"] >= planned and summary["errors"] == 0:
        summary["status"] = "done"
    return summary


def write_progress(
    path: str | Path,
    *,
    config: dict[str, Any],
    phase: str,
    step: int,
    budget_steps: int,
    train_loss: float | None = None,
    dev_loss: float | None = None,
    step_seconds: float | None = None,
    gpu_memory_gb: float | None = None,
    eval_summary: dict[str, Any] | None = None,
    artifacts: Sequence[str] | None = None,
    log_tail: Sequence[str] | None = None,
    activity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """写契约 §7 的 ``progress.json``（字段齐全，未产生的值写 null）。"""
    payload = {
        "stage": config.get("stage"),
        "platform": config.get("platform"),
        "arm": config.get("arm"),
        "seed": config.get("seed"),
        "phase": phase,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": int(step),
        "budget_steps": int(budget_steps),
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "step_seconds": step_seconds,
        "gpu_memory_gb": gpu_memory_gb,
        "eval": eval_summary
        or {
            "planned": None,
            "completed": 0,
            "successes": 0,
            "failures": 0,
            "errors": 0,
            "status": "not_started",
        },
        "artifacts": list(artifacts or []),
        "log_tail": list(log_tail or []),
        "loss_definition": config.get("loss_definition"),
        "owner": config.get("owner"),
        "gpu_index": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        if os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].isdigit()
        else None,
        "activity": dict(activity or {}),
    }
    missing = [field for field in PROGRESS_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"progress.json 缺少契约字段：{missing}")
    write_json_atomic(path, payload)
    return payload


# --------------------------------------------------------------------------------------
# 训练器
# --------------------------------------------------------------------------------------


def build_pi05_backend(config: dict[str, Any], *, platform: str) -> nn.Module:
    """加载冻结的 π0.5 StarVLA 后端（远端环境；本机缺 starVLA／权重时明确报错）。

    只做加载与审计：冻结、LoRA 注入和动作专家解冻由 `freeze_parameters` / `apply_lora`
    统一处理，保证四臂使用相同后端范围。
    """
    from safetensors.torch import load_file

    from ..backends.rawvla import new_model
    from ..backends.rawvla_features import load_feature_audited

    checkpoint = Path(config["checkpoints"][platform])
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"找不到 π0.5 权重 {checkpoint}；文档记录的位置见 S2 提示词（远端 rawvla/checkpoints）"
        )
    tokenizer = Path(config["tokenizer"])
    model = new_model("libero" if platform == "libero" else "robotwin", tokenizer)
    audit = load_feature_audited(model, load_file(str(checkpoint)))
    model.audit = {  # 供启动记录使用，不改模型行为
        **audit,
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer),
        "platform": platform,
        "normalization": "q01_q99_minus1_1" if platform == "libero" else "MEAN_STD",
    }
    return model


def _resolve_builder(spec: str, *, default: str) -> Any:
    """``module:function`` 形式的构造器解析（缺依赖时给出明确信息）。"""
    target = spec or default
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"构造器写法应为 'module:function'，实际 {target!r}")
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:  # pragma: no cover - 远端环境缺依赖时给出明确原因
        raise ImportError(
            f"找不到 {module_name}（{error}）；该模块由其他工作流负责，本机不可用时如实报告"
        ) from error
    if not hasattr(module, attribute):
        raise AttributeError(
            f"{module_name} 没有 {attribute}；检查 model.builder / backend.builder 配置"
        )
    return getattr(module, attribute)


class V11Trainer:
    """S2 单平台训练循环。``encoder`` / ``backend`` / ``data_index`` 可注入以便测试。"""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        seed: int,
        run_dir: str | Path,
        device: str | None = None,
        encoder: nn.Module | None = None,
        backend: nn.Module | None = None,
        data_index: V11DataIndex | None = None,
        normalizer: ActionNormalizer | None = None,
    ) -> None:
        self.config = config
        self.config.setdefault("loss_definition", LOSS_DEFINITION)
        self.platform = str(config["platform"])
        if self.platform not in PLATFORM_SPECS:
            raise ValueError(f"未知平台 {self.platform!r}；支持 {sorted(PLATFORM_SPECS)}")
        self.spec = PLATFORM_SPECS[self.platform]
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or config.get("device") or _default_device())
        self.lora_report: dict[str, Any] = {"enabled": False, "wrapped": []}
        self.freeze_report: dict[str, Any] = {}
        # seed 必须在构造模型之前设置（契约 §6）。
        set_seed(self.seed)
        window = config["data"]["window"]
        self.window = {
            "burn_in": int(window.get("burn_in", 16)),
            "gradient": int(window.get("gradient", 32)),
            "stride": int(window.get("stride", 32)),
            "supervision_points": int(window.get("supervision_points", 4)),
        }
        self.proprio_history = int(config["data"].get("proprio_history", 4))
        self.batch = int(config["optim"].get("batch", 8))
        self.micro_batch = int(config["optim"].get("micro_batch", self.batch))
        if self.batch % self.micro_batch:
            raise ValueError("optim.micro_batch 必须整除 optim.batch")
        self.budget_steps = int(config["optim"].get("steps", 100_000))
        self.loss_weights = dict(config["loss"].get("weights", DEFAULT_LOSS_WEIGHTS))
        self.gradient_checkpointing = bool(config["model"].get("gradient_checkpointing", False))
        if self.gradient_checkpointing:
            raise ValueError("stateful encoder.step cannot be recomputed by checkpointing")
        self.raft_resize_hw = config["model"].get("raft_resize_hw")
        self.forecast_enabled = bool(config["loss"].get("forecast", True))
        self.forecast_horizons = tuple(
            int(h) for h in config["loss"].get("forecast_horizons", FORECAST_HORIZONS)
        )
        self.representation_mode = str(config["loss"].get("representation_mode", "temporal"))
        if self.representation_mode not in ("temporal", "independent_windows"):
            raise ValueError("unknown representation_mode")
        self._check_arm(config)
        self.monitor = config.get("monitor", {})
        self.log_tail: deque[str] = deque(maxlen=20)

        self.data_index = data_index or V11DataIndex.from_config(config)
        self.train_candidates = window_candidates(
            self.data_index,
            platform=self.platform,
            split=str(config["data"].get("train_split", "train")),
            burn_in=self.window["burn_in"],
            gradient=self.window["gradient"],
            stride=self.window["stride"],
        )
        dev_entries = window_candidates(
            self.data_index,
            platform=self.platform,
            split=str(config["data"].get("dev_split", "dev")),
            burn_in=self.window["burn_in"],
            gradient=self.window["gradient"],
            stride=self.window["stride"],
        )
        dev_count = min(int(config["data"].get("dev_windows", 4)), len(dev_entries))
        self.dev_windows = (
            sample_windows(dev_entries, seed=self.seed, step=0, batch=dev_count)
            if dev_count
            else []
        )
        self.normalizer = normalizer or ActionNormalizer.from_file(
            self.platform, config["data"]["norm_stats"][self.platform]
        )

        # 注入的模块也要搬到同一设备：否则 CPU 权重 + CUDA 输入会以"dtype 不一致"的形式报错
        self.encoder = (encoder if encoder is not None else self._build_encoder()).to(self.device)
        readout_config = getattr(getattr(self.encoder, "readout", None), "config", None)
        if readout_config is not None:
            readout_config.collect_attention = False
        self.backend = (backend if backend is not None else self._build_backend()).to(self.device)
        self.policy = RawVLAFeaturePolicy(self.encoder, self.backend, self._bridge_config())
        self.optimizer, self.optimizer_report = build_optimizer(
            list(self._trainable_parameters()),
            groups=config["optim"]["groups"],
            betas=config["optim"].get("betas", (0.9, 0.95)),
            unmatched=str(config["optim"].get("unmatched", "pi05_adapt")),
            fused=bool(config["optim"].get("fused", False)) and self.device.type == "cuda",
        )
        self.scheduler = self._build_scheduler()
        self.amp_dtype = self._amp_dtype()
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(self.amp_dtype == torch.float16 and self.device.type == "cuda")
        )
        self.step = 0
        self.resumed_from: dict[str, Any] | None = None
        self.loss_definition_change: dict[str, Any] | None = None
        self.epoch_losses: list[float] = []
        self.loss_ema: float | None = None
        self.dev_loss: float | None = None
        self.performance = dict(config.get("performance", {}))
        self.compiled_autograd_settings = (
            configure_compiled_autograd()
            if any(k.startswith("compile_") and v for k, v in self.performance.items())
            else {}
        )
        self.best_dev_loss: float | None = None
        self.initial_dev_loss: float | None = None
        self.last_checkpoint_step = 0
        self.cycle_speeds: deque[float] = deque(maxlen=5)
        self.activity: dict[str, Any] = {}
        self._eager_methods: list[tuple[Any, str, Any]] = []

        def compile_method(module, name, **options):
            original = getattr(module, name)
            self._eager_methods.append((module, name, original))
            setattr(module, name, torch.compile(original, **options))

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(self.performance.get("cudnn_benchmark", False))
        stack = getattr(self.encoder, "state_stack", None)
        if stack is not None and hasattr(stack, "checkpoint_predictions"):
            stack.checkpoint_predictions = bool(self.performance.get("rkn_checkpoint", True))
        if self.performance.get("compile_banded") and stack is not None:
            compile_method(stack, "banded_kernel", fullgraph=True, dynamic=False)
        if self.performance.get("compile_prediction") and stack is not None:
            compile_method(stack, "_predict_impl", fullgraph=True, dynamic=False)
        if self.performance.get("compile_measurement"):
            compile_method(self.encoder.measurement, "backbone_levels", dynamic=False)
            compile_method(self.encoder.measurement, "fuse", dynamic=False)
        if self.performance.get("compile_readout"):
            compile_method(self.encoder.readout.resampler, "forward", dynamic=False)
        if self.performance.get("compile_raft_update"):
            # The author GRU/update block is pure tensor math; the wrapper still
            # owns bidirectional visibility, RAW validation and crop supervision.
            block = self.encoder.correspondence.raft.update_block
            compile_method(block, "forward", fullgraph=True, dynamic=False)

    @staticmethod
    def _check_arm(config: dict[str, Any]) -> None:
        """四臂都由同源 RAW 进入；A/B 的 ISP 在编码器的公共输入边界执行。"""
        arm = str(config.get("arm", "D"))
        if arm not in ("A", "B", "C", "D"):
            raise ValueError(f"未知 arm {arm!r}")
        if arm in ("A", "B") and config["model"].get("isp") != "fixed_clip_srgb_v1":
            raise ValueError("A/B 需要 model.isp=fixed_clip_srgb_v1")

    # -- 构造 ---------------------------------------------------------------------

    def _build_encoder(self) -> nn.Module:
        model_cfg = dict(self.config["model"])
        spec = model_cfg.pop("builder", "owac.training.n04_v11:build_v11_encoder")
        builder = _resolve_builder(spec, default=spec)
        module = builder(model_cfg, platform=self.platform, arm=self.config.get("arm", "D"))
        return module.to(self.device)

    def _build_backend(self) -> nn.Module:
        backend_cfg = dict(self.config["backend"])
        spec = backend_cfg.pop("builder", "owac.training.n04_v11:build_pi05_backend")
        builder = _resolve_builder(spec, default=spec)
        module = builder(backend_cfg, platform=self.platform)
        adapter = self.config["backend"].get("adapter", {})
        if adapter.get("lora", {}).get("enabled", False):
            self.lora_report = apply_lora(
                module,
                rank=int(adapter["lora"].get("rank", 16)),
                alpha=float(adapter["lora"].get("alpha", 16.0)),
                targets=tuple(
                    adapter["lora"].get(
                        "targets",
                        (
                            "self_attn.q_proj",
                            "self_attn.k_proj",
                            "self_attn.v_proj",
                            "self_attn.o_proj",
                        ),
                    )
                ),
            )
        else:
            self.lora_report = {"enabled": False, "wrapped": []}
        self.freeze_report = freeze_parameters(
            module,
            trainable_prefixes=tuple(adapter.get("trainable_prefixes", ("action_head.",))),
        )
        return module.to(self.device)

    def _bridge_config(self) -> FeatureBridgeConfig:
        return FeatureBridgeConfig(
            action_dim=int(self.config["backend"].get("action_dim", 32)),
            control_dim=int(self.spec.action_dim),
            action_horizon=int(self.spec.action_horizon),
            num_denoise_steps=int(self.config["backend"].get("num_denoise_steps", 10)),
            tokens_per_camera=int(self.config["model"].get("tokens_per_camera", 256)),
            token_dim=int(self.config["model"].get("token_dim", 2048)),
            name=str(self.config["model"].get("name", "owac-n04-v1.1")),
        )

    def _trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        named: list[tuple[str, nn.Parameter]] = []
        for prefix, module in (("encoder", self.encoder), ("backend", self.backend)):
            named.extend(
                (f"{prefix}.{name}", parameter) for name, parameter in module.named_parameters()
            )
        return named

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        warmup = max(1, int(self.config["optim"].get("warmup_steps", 2000)))
        total = max(warmup + 1, int(self.config["optim"].get("steps", self.budget_steps)))
        min_ratio = float(self.config["optim"].get("min_lr_ratio", 0.0))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)

    def _amp_dtype(self) -> torch.dtype | None:
        if not bool(self.config["optim"].get("amp", True)):
            return None
        if str(self.config["optim"].get("amp_dtype", "bfloat16")) != "bfloat16":
            raise ValueError("N04 supports BF16 or FP32; FP16 scaled autograd is not implemented")
        return torch.bfloat16 if self.device.type == "cuda" else None

    def _require_training_enabled(self) -> None:
        root = Path(os.environ.get("N04V11_ROOT", "/workspace/owac/research/n04-v1"))
        marker = root / "TRAINING_STOPPED_BY_USER.json"
        if marker.exists():
            raise RuntimeError(
                f"Training stopped by user: {marker}; optimizer updates are forbidden"
            )

    @contextlib.contextmanager
    def network_dtype(self):
        if self.amp_dtype is None:
            yield
        else:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                yield

    # -- 单帧调用构造 --------------------------------------------------------------
    # WS-C 的 ``N04V11Encoder.step`` 是**单帧单序列**在线接口：``values`` 是一帧的多相机
    # 列表（每项 ``[C,H,W]``），没有 batch 维。批量由本训练器在调用方循环，
    # ``optim.micro_batch`` 只决定多少段窗口共享一次 backward。

    def _reset_states(self, window: WindowBatch) -> Any:
        """窗口边界 reset：不跨 episode 继承状态，burn-in 用当前模型重新生成。"""
        return self.encoder.reset(f"{self.platform}:{window.episode_id}:{window.start}")

    def _saturation(self, window: WindowBatch, position: int) -> list[torch.Tensor] | None:
        white_level = window.metadata.get("raw_white_level")
        masks = [
            saturation_mask(
                window.raw[camera][position],
                self.platform,
                white_level=None if white_level is None else float(white_level),
            )
            for camera in window.camera_ids
        ]
        if any(mask is None for mask in masks):
            return None
        return [mask.to(self.device) for mask in masks if mask is not None]

    def _frame(self, window: WindowBatch, position: int) -> _ScheduleFrame:
        return _ScheduleFrame(
            values=[window.raw[camera][position].to(self.device) for camera in window.camera_ids],
            camera_id=list(window.camera_ids),
            frame_ids=(int(window.frame_ids[position]),),
            episode_ids=(window.episode_id,),
            saturated_mask=self._saturation(window, position),
        )

    def _context(self, window: WindowBatch, position: int) -> EmbodimentContext:
        """本体上下文：只含 ``≤ t`` 的真实帧，新的在前。

        WS-B 的加性控制输入是单个向量（``proprio_dim`` 维），所以历史按 新→旧 拼接后展平；
        ``model.proprio_dim`` 配成 ``(history+1) × 平台本体维度``，不会被静默截断。
        """
        history = window.proprio[position].flip(0).reshape(-1).to(self.device)
        return EmbodimentContext(
            proprio=history,
            proprio_available=True,
            executed_action=None,
            executed_action_available=False,
            time_unit="frame",
            metadata={
                "platform": self.platform,
                "proprio_expr": self.spec.proprio_expr,
                "history_frames": int(window.proprio.shape[1]),
                "ordering": "newest_first",
                "padded_frames": int(window.proprio_padded[position]),
                "future_action_excluded": True,
            },
        )

    def _encode(self, frame: _ScheduleFrame, context: EmbodimentContext, state: Any):
        """一次新观测只融合一次：与桥的 ``encode_observation`` 同参，但保留 ``StepOutput``。

        保留 StepOutput 是因为契约 §5 的 debug（flow／质量 R／模式权重／支持轨迹）与辅助
        损失项需要它；监督帧改走 ``policy.forward_from_observation``，同样只融合一次。
        """
        output = self.encoder.step(
            frame.values,
            context,
            state,
            camera_id=frame.camera_id,
            frame_key=(tuple(frame.episode_ids), tuple(frame.frame_ids)),
            measurement_valid=None,
            saturation=frame.saturated_mask,
            new_measurement=frame.new_measurement,
        )
        tokens = self.encoder.readout_tokens(output, self.policy.config.tokens_per_camera)
        return tokens, output

    def _token_values(self, tokens: Any) -> torch.Tensor | None:
        values = getattr(tokens, "values", tokens)
        return values if isinstance(values, torch.Tensor) else None

    def _contract_frames(self, frame: _ScheduleFrame) -> list[Any]:
        """把一帧的多相机张量写成合约帧（与 WS-C ``_as_frames`` 同字段），供目标前向使用。"""
        from ..frameworks.n04.contracts import (
            MeasurementFrame as ContractMeasurementFrame,
        )
        from ..frameworks.n04.contracts import build_measurement_id, input_scale_for

        encoder = self.encoder
        config = encoder.config
        frames = []
        for camera, values in zip(frame.camera_id, frame.values, strict=True):
            frames.append(
                ContractMeasurementFrame(
                    raw=values,
                    camera_id=camera,
                    measurement_id=build_measurement_id(
                        f"{self.platform}:target", 0, camera, config.input_mode
                    ),
                    input_mode=config.input_mode,
                    scale=input_scale_for(config.scale_name),
                    size_hw=(int(values.shape[-2]), int(values.shape[-1])),
                    cfa_pattern=None,
                    time_unit="frame",
                    saturation_level=config.saturation_level,
                    provenance={"source": "training.n04_v11 target forward"},
                )
            )
        # 与 step/在线服务使用同一个输入变换；目标分支也不能把 A/B 的 RAW 直接喂骨干。
        if getattr(config, "arm", self.config.get("arm", "D")) in ("A", "B"):
            frames, _ = encoder._as_frames(frame.values, frame.camera_id, None, None)
        return frames

    @torch.no_grad()
    def _forecast_engine(
        self, total: int, *, weight: float, n_forecast: int
    ) -> ForecastEngine | None:
        """多步 forecast 只在有状态模块、且配置打开时启用（A/C 臂保持 None）。"""
        if not self.forecast_enabled:
            return None
        estimator = getattr(self.encoder, "state_stack", None)
        if estimator is None:
            return None
        return ForecastEngine(
            estimator,
            self.forecast_horizons,
            total_frames=total,
            weight=weight
            * float(self.loss_weights.get("forecast", 0.0))
            / float(max(1, n_forecast)),
            diagnostics_every=int(self.config["loss"].get("forecast_diagnostics_every", 1)),
            max_innovation_variance=self.config["loss"].get("forecast_max_variance"),
        )

    def _forecast_anchor_due(self, position: int) -> bool:
        """锚点包含监督帧；只登记 burn-in 之后的实际后验。"""
        if not self.forecast_enabled:
            return False
        stride = int(self.config["loss"].get("forecast_anchor_stride", 1))
        return position >= self.window["burn_in"] and position % max(1, stride) == 0

    def _add_forecast_anchors(
        self,
        forecast: ForecastEngine,
        window: WindowBatch,
        position: int,
        size_hw: tuple[int, int],
        output: Any,
        proprio_context: torch.Tensor,
        segment_last: int,
    ) -> None:
        """t0 记录预测：先验来自 ``forecast_inputs``，R 来自本帧质量头。"""
        for camera in window.camera_ids:
            posterior = _posterior_state(output, camera)
            R = _quality_variance(output, camera)
            if posterior is None or R is None:
                forecast.skip_all("no_posterior_state")
                continue
            forecast.add(
                position,
                camera,
                posterior,
                R,
                proprio_context,
                size_hw=size_hw,
                segment_last=segment_last,
                proprio_valid=getattr(output, "forecast_inputs", {})
                .get(camera, {})
                .get("proprio_valid"),
            )

    def _transport_forecast(
        self, forecast: ForecastEngine, window: WindowBatch, output: Any, position: int
    ) -> None:
        """把锚点 xy 沿当前帧真实 flow 推进一步（no_grad，只决定看哪里）。"""
        debug = _debug_dict(output)
        flows = debug.get("flow")
        for camera in window.camera_ids:
            flow_px = None
            if isinstance(flows, dict):
                item = flows.get(camera)
                flow_px = item.get("flow_px") if isinstance(item, dict) else item
            if not isinstance(flow_px, torch.Tensor):
                forecast.mark_missing_transport(position, camera)
                continue
            size_hw = (int(window.raw[camera].shape[-2]), int(window.raw[camera].shape[-1]))
            visible = debug.get("flow_valid", {}).get(camera)
            forecast.transport(camera, flow_px, size_hw, visible)

    def _settle_forecast(
        self, forecast: ForecastEngine, frame: _ScheduleFrame, position: int, output: Any = None
    ) -> list[tuple[_ForecastAnchor, torch.Tensor]]:
        due = forecast.due(position)
        if not due:
            return []
        targets = getattr(output, "target_features", None) or self._target_features(frame)
        losses: list[tuple[_ForecastAnchor, torch.Tensor]] = []
        for anchor in due:
            spec = None if targets is None else targets.get(anchor.camera)
            target = None if spec is None else self._sample_latent(spec, anchor.xy_px)
            if spec is not None and isinstance(spec.get("support"), torch.Tensor):
                support_spec = dict(spec)
                support_spec["latent"] = spec["support"].float()
                support_spec["input_support"] = True
                supported = self._sample_latent(support_spec, anchor.xy_px)
                anchor.valid = anchor.valid & (supported[..., 0] > 0.999)
            loss = forecast.settle(anchor, target)
            if loss is not None:
                losses.append((anchor, loss))
        return losses

    def _target_features(self, frame: _ScheduleFrame) -> dict[str, dict[str, Any]] | None:
        """t+k 帧的 stop-gradient 目标特征：同一 RAW 编码器的融合特征 + 采样网格入口。

        只用于损失目标，不进入任何预测路径；没有真实测量网络（例如单测注入的替身）时返回
        ``None``，调用方按"跳过并计数"处理，不用 0 冒充。
        """
        measurement = getattr(self.encoder, "measurement", None)
        if measurement is None:
            return None
        from ..frameworks.n04.measurement import preprocess_frames

        frames = self._contract_frames(frame)
        batch = preprocess_frames(frames, measurement.config)
        # 契约 §6：多步预测目标是"同一 RAW 编码器的 stop-gradient 目标"。这里必须显式 no_grad：
        # 它在带梯度的帧上被调用（每个梯度帧最多 3 个 horizon），若留图则会为每帧多存数份
        # 全分辨率 ConvNeXt 激活，直接吃满显存（远端 96 GiB 实测 OOM 就是这一步）。
        with torch.no_grad():
            levels = [level.float() for level in measurement.backbone_levels(batch.values)]
            latent = measurement.fuse(levels)
            if hasattr(self.encoder, "normalize_observation"):
                latent = self.encoder.normalize_observation(latent)
        latent = latent.detach()
        targets: dict[str, dict[str, Any]] = {}
        for index, camera in enumerate(frame.camera_id):
            targets[camera] = {
                "latent": latent[index : index + 1],
                "values": batch.values[index : index + 1],
                "support": batch.support[index : index + 1],
                "saturated": batch.saturated[index : index + 1],
                "pad_xy": tuple(float(v) for v in batch.pad_xy),
                "pad_hw": (int(batch.values.shape[-2]), int(batch.values.shape[-1])),
                "size_hw": (int(frames[index].size_hw[0]), int(frames[index].size_hw[1])),
            }
        return targets

    def _sample_latent(self, target: dict[str, Any], xy_px: torch.Tensor) -> torch.Tensor | None:
        """在原始像素坐标上采样融合特征（pad/stride 走 SampleGrid 元数据，不写死）。"""
        from ..frameworks.n04.contracts import PixelPoints, build_sample_grid
        from ..frameworks.n04.correspondence import px_to_grid

        latent = target["latent"]
        grid = build_sample_grid(
            input_mode=self.encoder.config.input_mode,
            sensor_size_hw=target["size_hw"],
            pad_xy=target["pad_xy"],
            pad_hw=target["pad_hw"],
            stride=self.encoder.measurement.config.measurement.fusion_stride,
            cfa_pattern=None,
        ).with_device(latent.device)
        transform = (
            grid.sensor_to_backbone_input if target.get("input_support") else grid.sensor_to_feature
        )
        feature_xy = transform.apply(
            PixelPoints(xy=xy_px.to(latent.device, torch.float32), unit="original_px")
        ).xy
        height, width = latent.shape[-2:]
        sampled = torch.nn.functional.grid_sample(
            latent.float(),
            px_to_grid(feature_xy, (height, width)).unsqueeze(1),
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        )
        return sampled.squeeze(2).permute(0, 2, 1).contiguous()

    def _representation_term(self, features: torch.Tensor | None) -> torch.Tensor | None:
        if features is None or getattr(self, "_evaluating_actions_only", False):
            return None
        terms = observation_vicreg_terms(features.to(torch.float32))
        return terms["variance"] + terms["covariance"]

    def _task_text(self, window: WindowBatch) -> str:
        metadata = window.metadata
        for key in ("task_description", "task_name", "task", "instruction"):
            value = metadata.get(key)
            if value:
                return str(value)
        return str(window.episode_id)

    def _task_tokens(self, window: WindowBatch, backend_state: torch.Tensor):
        text = self._task_text(window)
        discrete = bool(getattr(self.backend, "discrete_state_input", False))
        cached = getattr(self, "_token_cache", None)
        if not discrete and cached is not None and cached[0] == text:
            return cached[1]
        tokens, mask = tokenize_backend_task(
            self.backend, text, backend_state[0].detach().to("cpu").numpy()
        )
        # 真后端 tokenizer 返回**一维** numpy（[L]）；桥要求 [B,L]（B=1）：
        # 这里显式补 batch 维并固定 dtype（token=long，mask=bool）
        result = (
            torch.as_tensor(tokens, device=self.device).to(torch.long).reshape(1, -1),
            torch.as_tensor(mask, device=self.device).to(torch.bool).reshape(1, -1),
        )
        if not discrete:
            self._token_cache = (text, result)
        return result

    def _backend_state(self, window: WindowBatch, position: int) -> torch.Tensor:
        current = self.normalizer.normalize_state(window.proprio[position, -1].to(self.device))
        return current.reshape(1, -1)

    # -- 前向 ---------------------------------------------------------------------

    def _supervision_segments(
        self, window: WindowBatch
    ) -> list[tuple[int | None, tuple[int, ...]]]:
        """把梯度帧切成"一个监督点 + 其后到下一个监督点之前"的段。

        段边界只由窗口计划决定（``supervision_points``），不按 loss 大小调整。第一段从
        ``burn_in`` 开始，最后一段包含最后一个监督点之后的帧。
        """
        burn_in = self.window["burn_in"]
        total = len(window)
        points = sorted(
            int(position) for position, _, _ in window.supervision_points() if position >= burn_in
        )
        if not points:
            return [(None, tuple(range(burn_in, total)))]
        segments: list[tuple[int | None, tuple[int, ...]]] = []
        for index, point in enumerate(points):
            start = burn_in if index == 0 else point
            end = points[index + 1] if index + 1 < len(points) else total
            segments.append((point, tuple(range(start, end))))
        return segments

    def _run_encoder_step(self, frame: _ScheduleFrame, context: EmbodimentContext, state: Any):
        """Stateful encoder must execute exactly once per observation."""
        if self.gradient_checkpointing:
            raise ValueError("stateful encoder.step cannot be recomputed by checkpointing")
        return self._encode(frame, context, state)

    def _forward_window(self, window: WindowBatch, *, train: bool, **kwargs):
        """Keep compiled single-backward execution; exact fallback uses native autograd.

        AOT's shared saved intermediates can be mutated by repeated backwards even with
        donation disabled. Replay the whole window with the same RNG, buffers and weights
        using the original pure routines, rather than dropping terms or changing counts.
        """
        exact = (
            train
            and not kwargs.get("single_backward", False)
            and not (
                self.performance.get("verified_joint_backward", False)
                if kwargs.get("_fast_counts") is None
                else kwargs["_fast_counts"]
            )
        )
        switched = []
        retry_allowed = train and not exact and not kwargs.get("single_backward", False)
        retry_allowed = retry_allowed and bool(self._eager_methods)
        if retry_allowed:
            saved_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            saved_cuda = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
            saved_buffers = [
                (buffer, buffer.clone())
                for module in (self.encoder, self.backend)
                for name, buffer in module.named_buffers()
                if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
            ]
            saved_grads = [
                (p, p.grad, None if p.grad is None else p.grad._version)
                for _, p in self._trainable_parameters()
            ]
        if exact:
            for module, name, eager in self._eager_methods:
                switched.append((module, name, getattr(module, name)))
                setattr(module, name, eager)
        try:
            try:
                result = self._forward_window_impl(window, train=train, **kwargs)
            except RuntimeError as error:
                known = any(
                    text in str(error)
                    for text in ("donated buffer", "modified by an inplace operation")
                )
                if not (retry_allowed and known):
                    raise
                message = str(error).splitlines()[0]
                receipt = {
                    "window": window.episode_id,
                    "start": window.start,
                    "time": time.time(),
                    "error": message,
                    "traceback": traceback.format_exc(),
                }
                with (self.run_dir / "compiled-backward-retries.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(receipt) + "\n")
                # A traceback retains the failed CUDA graphs. Release those frame
                # references before rebuilding the full native window graph.
                error.__traceback__ = None
                error.__context__ = None
                # autograd.grad accumulates into local maps; .grad is written only
                # after every backward succeeds. Never retry after partial writes.
                if any(
                    p.grad is not old or (old is not None and old._version != version)
                    for p, old, version in saved_grads
                ):
                    raise RuntimeError(
                        "compiled retry would overwrite partial accumulated gradients"
                    ) from error
                random.setstate(saved_rng[0])
                np.random.set_state(saved_rng[1])
                torch.set_rng_state(saved_rng[2])
                if saved_cuda:
                    torch.cuda.set_rng_state_all(saved_cuda)
                with torch.no_grad():
                    for buffer, value in saved_buffers:
                        buffer.copy_(value)
                for module, name, eager in self._eager_methods:
                    switched.append((module, name, getattr(module, name)))
                    setattr(module, name, eager)
                result = self._forward_window_impl(
                    window, train=train, **{**kwargs, "_fast_counts": False}
                )
                result[1]["compiled_backward_retry"] = message
            if switched:
                result[1]["gradient_eager_fallback"] = True
            return result
        finally:
            for module, name, compiled in reversed(switched):
                setattr(module, name, compiled)

    def _forward_window_impl(
        self,
        window: WindowBatch,
        *,
        train: bool,
        backward_weight: float = 1.0,
        single_backward: bool = False,
        _fast_counts: bool | None = None,
        representation_cache: RepresentationGradientCache | None = None,
        representation_index: int = 0,
        representation_observations: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """One fusion per frame; exact means over produced losses, with bounded TBPTT graphs.

        A forecast may cross a TBPTT boundary. Its source graph belongs to the old segment;
        only that segment's backward is delayed by at most max(horizons) frames. Per-term
        gradient sums are normalized after all valid samples are known. This preserves
        the declared weights without padding missing losses or retaining an entire window.
        """
        if self.device.type == "cuda" and self.performance.get("preload_window"):
            window = dataclasses.replace(
                window,
                raw={k: v.to(self.device) for k, v in window.raw.items()},
                proprio=window.proprio.to(self.device),
                targets=window.targets.to(self.device),
                target_mask=window.target_mask.to(self.device),
            )
        if (
            train
            and self.representation_mode == "independent_windows"
            and self.loss_weights.get("representation", 0)
            and representation_cache is None
        ):
            raise ValueError("independent-window representation requires the batch gradient cache")
        burn_in, total = self.window["burn_in"], len(window)
        segments = self._supervision_segments(window)
        points = {point for point, _ in segments if point is not None}
        position_segment = {p: i for i, (_, ps) in enumerate(segments) for p in ps}
        ends = {i: max(ps) for i, (_, ps) in enumerate(segments)}
        end_segment = {end: i for i, end in ends.items()}
        segment_terms = {i: defaultdict(list) for i in ends}
        observations = {i: [] for i in ends}
        reports: dict[str, list[torch.Tensor]] = defaultdict(list)
        counts: dict[str, int] = defaultdict(int)
        gradient_sums: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        parameters = [p for _, p in self._trainable_parameters() if p.requires_grad]
        all_terms: dict[str, list[torch.Tensor]] = defaultdict(list)
        finished: set[int] = set()
        state = self._reset_states(window)
        self.policy.counter.reset()
        forecast = self._forecast_engine(total, weight=0.0, n_forecast=1)
        max_horizon = max(self.forecast_horizons, default=0) if forecast is not None else 0
        sources = {}
        fast = (
            train
            and not single_backward
            and (
                self.performance.get("verified_joint_backward", False)
                if _fast_counts is None
                else _fast_counts
            )
        )
        expected = {
            "action": len(points),
            "correspondence": total - burn_in,
            "representation": (
                int(representation_cache is not None)
                if self.representation_mode == "independent_windows"
                else sum(len(ps) >= 2 for _, ps in segments)
            ),
            "forecast": sum(
                len(window.camera_ids)
                for pos in range(burn_in, total)
                if self._forecast_anchor_due(pos)
                for h in self.forecast_horizons
                if pos + h < total
            )
            if forecast is not None
            else 0,
        }
        if fast:
            rng = (
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            )
            batch_norm = [
                (b, b.clone())
                for module in (self.encoder, self.backend)
                for name, b in module.named_buffers()
                if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
            ]

        def record(segment, name, value):
            if value is None:
                return
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite {name} loss at window {window.episode_id}")
            value = value.float().reshape(())
            segment_terms[segment][name].append(value)
            reports[name].append(value.detach())

        def finish(segment):
            groups = segment_terms.pop(segment)
            active = []
            for name, values in groups.items():
                counts[name] += len(values)
                if train and self.loss_weights.get(name, 0) and values:
                    value = torch.stack(values).sum()
                    if value.requires_grad:
                        if single_backward:
                            all_terms[name].append(value)
                        else:
                            active.append((name, value))
            if fast and active:
                joint = [
                    v * self.loss_weights[n] / expected[n]
                    for n, v in active
                    if expected.get(n, 0) > 0
                ]
                active = [("joint", torch.stack(joint).sum())] if joint else []
            for index, (name, value) in enumerate(active):
                grads = torch.autograd.grad(
                    value, parameters, allow_unused=True, retain_graph=index < len(active) - 1
                )
                for j, grad in enumerate(grads):
                    if grad is not None:
                        if j in gradient_sums[name]:
                            gradient_sums[name][j].add_(grad.detach())
                        else:
                            gradient_sums[name][j] = grad.detach().clone()
            observations.pop(segment, None)
            finished.add(segment)
            if forecast is not None:
                for anchor in forecast.anchors:
                    if anchor.segment_last == ends[segment]:
                        forecast._roll_cache.pop((anchor.position, anchor.camera), None)
                forecast.anchors = [a for a in forecast.anchors if a.segment_last != ends[segment]]

        for position in range(total):
            segment = position_segment.get(position)
            frame, context = self._frame(window, position), self._context(window, position)
            grad = train and position >= burn_in
            with torch.enable_grad() if grad else torch.no_grad():
                if position >= burn_in and position in points:
                    result = self._supervised_step(window, position, frame, context, state)
                    tokens, output = result["visual_tokens"], result["encoder_output"]
                    record(segment, "action", result["loss"])
                else:
                    with self.network_dtype():
                        tokens, output = self._run_encoder_step(frame, context, state)
                state = getattr(output, "state", state)
                if position >= burn_in:
                    scalars = _scalar_terms(output)
                    for name, value in scalars.items():
                        if name != "forecast":
                            record(segment, name, value)
                            sources[name] = "encoder.losses(all gradient frames)"
                    values = self._token_values(tokens)
                    if (
                        values is not None
                        and representation_observations is not None
                        and position == burn_in
                    ):
                        representation_observations.append(
                            summarize_prefix(values, len(window.camera_ids)).detach()
                        )
                    if (
                        values is not None
                        and representation_cache is not None
                        and position == burn_in
                    ):
                        vector = summarize_prefix(values, len(window.camera_ids))
                        record(
                            segment,
                            "representation",
                            representation_cache.replay(representation_index, vector),
                        )
                        sources["representation"] = "VICReg across independent training windows"
                    if (
                        values is not None
                        and self.representation_mode == "temporal"
                        and not getattr(self, "_evaluating_actions_only", False)
                    ):
                        observations[segment].append(values)
                    if forecast is not None:
                        self._transport_forecast(forecast, window, output, position)
                        if self._forecast_anchor_due(position):
                            self._add_forecast_anchors(
                                forecast,
                                window,
                                position,
                                tuple(frame.values[0].shape[-2:]),
                                output,
                                context.proprio.reshape(1, -1),
                                ends[segment],
                            )
                        for anchor, loss in self._settle_forecast(
                            forecast, frame, position, output
                        ):
                            record(end_segment[anchor.segment_last], "forecast", loss)
                            sources["forecast"] = "causal forecast, live source / stop-grad target"
                    if position == ends[segment]:
                        frames = observations[segment]
                        if len(frames) >= 2:
                            features = torch.cat(frames, dim=0)
                            terms = observation_vicreg_terms(features)
                            record(
                                segment, "representation", terms["variance"] + terms["covariance"]
                            )
                            sources["representation"] = "VICReg across observations at fixed query"
                        observations[segment] = []
                        state = _detach_state(state)
            # Delayed graphs are independent after the state cut above; no cross-segment
            # backward is possible. The final frames also settle all in-window horizons.
            for ready in list(ends):
                if ready not in finished and (
                    position >= min(total - 1, ends[ready] + max_horizon)
                ):
                    finish(ready)

        mismatch = fast and any(
            self.loss_weights.get(n, 0) and counts.get(n, 0) != expected.get(n, 0)
            for n in self.loss_weights
        )
        if mismatch:
            # Nothing has touched .grad yet. Restore stochastic/buffer state and rerun
            # the exact per-term implementation if any actual valid count differs.
            gradient_sums.clear()
            with torch.no_grad():
                for buffer, saved in batch_norm:
                    buffer.copy_(saved)
            torch.set_rng_state(rng[0])
            if rng[1]:
                torch.cuda.set_rng_state_all(rng[1])
            result, fallback_info = self._forward_window(
                window,
                train=train,
                backward_weight=backward_weight,
                _fast_counts=False,
                representation_cache=representation_cache,
                representation_index=representation_index,
                representation_observations=representation_observations,
            )
            fallback_info["valid_count_fallback"] = True
            return result, fallback_info
        if train:
            if single_backward:
                objective = [
                    torch.stack(v).sum() * self.loss_weights[n] / counts[n]
                    for n, v in all_terms.items()
                    if v
                ]
                if objective:
                    (torch.stack(objective).sum() * backward_weight).backward()
            else:
                for name, gradients in gradient_sums.items():
                    scale = (
                        backward_weight
                        if fast
                        else backward_weight * self.loss_weights[name] / counts[name]
                    )
                    for j, grad in gradients.items():
                        value = grad * scale
                        if parameters[j].grad is None:
                            parameters[j].grad = value
                        else:
                            parameters[j].grad.add_(value)
        terms = {name: torch.stack(values).mean() for name, values in reports.items() if values}
        for name in self.loss_weights:
            terms.setdefault(name, None)
        total_loss, report = weighted_total(terms, self.loss_weights)
        info = {
            "terms": report,
            "applied_terms": dict(report),
            "applied_train_loss": None if total_loss is None else float(total_loss.detach()),
            "term_sources": sources,
            "supervision_points": counts.get("action", 0),
            "forecast": None if forecast is None else forecast.report(),
            "windows": 1,
            "segments": len(segments),
            "valid_loss_counts": dict(counts),
            "gradient_fast_path": bool(fast),
            "denominators": {
                "windows": 1.0 / max(float(backward_weight), 1e-12),
                "points": counts.get("action", 0),
                "gradient_frames": total - burn_in,
                "forecast_anchors": counts.get("forecast", 0),
            },
        }
        return (None if total_loss is None else total_loss.detach()), info

    def _forward_windows(
        self,
        windows: Sequence[WindowBatch],
        *,
        train: bool,
        backward_weight: float | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """逐窗口前向；返回按窗口平均的**报告用**损失（已 detach）。

        ``train=True`` 时每个窗口内部逐源段／损失项累计梯度（权重默认 ``1/窗口数``，与
        `train_step` 的尺度一致）；返回的标量只能用于打印／监控，不能再次 backward。
        """
        weight = (
            float(backward_weight)
            if backward_weight is not None
            else (1.0 / float(max(1, len(windows))) if train else 0.0)
        )
        results = []
        for window in windows:
            loss, info = self._forward_window(window, train=train, backward_weight=weight)
            if loss is None:
                raise RuntimeError(
                    f"{window.episode_id} 的窗口 [{window.start}, {window.start + len(window)}) "
                    "没有任何可用损失项：检查动作监督点与编码器输出"
                )
            results.append((loss, info))
        total_loss = torch.stack([loss for loss, _ in results]).mean()
        return total_loss, _merge_window_info([info for _, info in results])

    def _supervised_step(
        self,
        window: WindowBatch,
        position: int,
        frame: _ScheduleFrame,
        context: EmbodimentContext,
        state: Any,
    ) -> dict[str, Any]:
        """监督点：真实 π0.5 flow-matching，经 prefix 把动作梯度送回编码器。"""
        backend_state = self._backend_state(window, position)
        task_tokens, task_mask = self._task_tokens(window, backend_state)
        actions = self.normalizer.normalize_actions(
            window.targets[position].to(self.device)
        ).unsqueeze(0)
        mask = window.target_mask[position].to(self.device).unsqueeze(0)
        with self.network_dtype():
            result = self.policy.forward_from_observation(
                frame,
                context,
                state,
                actions,
                task_tokens,
                task_mask,
                backend_state=backend_state,
                action_mask=mask,
            )
        return result

    def _heartbeat(self, phase="train", **activity):
        self.activity.update(activity)
        self.activity.update(
            {
                "initial_dev_loss": self.initial_dev_loss,
                "best_dev_loss": self.best_dev_loss,
                "last_checkpoint_step": self.last_checkpoint_step,
            }
        )
        write_progress(
            self.run_dir / "progress.json",
            config=self.config,
            phase=phase,
            step=self.step,
            budget_steps=self.budget_steps,
            train_loss=self.loss_ema,
            dev_loss=self.dev_loss,
            step_seconds=float(np.mean(self.cycle_speeds)) if self.cycle_speeds else None,
            log_tail=list(self.log_tail),
            activity=self.activity,
        )

    def _training_windows(self, plan):
        def load(spec):
            entry, window_start = spec
            return load_window_batch(
                entry,
                start=window_start,
                burn_in=self.window["burn_in"],
                gradient=self.window["gradient"],
                supervision_points=self.window["supervision_points"],
                proprio_history=self.proprio_history,
            )

        if not self.performance.get("prefetch_windows"):
            for spec in plan:
                yield load(spec)
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="n04-data") as pool:
            pending = pool.submit(load, plan[0])
            for index in range(len(plan)):
                window = pending.result()
                if index + 1 < len(plan):
                    pending = pool.submit(load, plan[index + 1])
                yield window

    # -- 一步训练 -----------------------------------------------------------------

    def _training_plan(self, step: int):
        plan = sample_windows(
            self.train_candidates, seed=self.seed, step=int(step), batch=self.batch
        )
        if (
            self.representation_mode == "independent_windows"
            and self.loss_weights.get("representation", 0)
            and len({entry.source_id for entry, _ in plan}) < 2
        ):
            alternatives = [
                candidate
                for candidate in self.train_candidates
                if candidate[0].source_id != plan[0][0].source_id
            ]
            if len(plan) < 2 or not alternatives:
                raise ValueError(
                    "independent-window training needs batch>=2 and two source sequences"
                )
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(step), 411, 777]))
            plan[-1] = alternatives[int(rng.integers(len(alternatives)))]
        return plan

    def _representation_seed(self, step: int, index: int) -> int:
        return int(
            np.random.SeedSequence([self.seed, int(step), int(index), 411]).generate_state(1)[0]
        )

    def _seed_window(self, step: int, index: int) -> None:
        value = self._representation_seed(step, index)
        torch.manual_seed(value)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(value)

    @torch.no_grad()
    def _representation_observation(self, window: WindowBatch) -> torch.Tensor:
        """Same causal burn-in and first gradient observation; no policy or future frames."""
        state = self._reset_states(window)
        output = None
        for position in range(self.window["burn_in"] + 1):
            frame, context = self._frame(window, position), self._context(window, position)
            with self.network_dtype():
                output = self.encoder.step(
                    frame.values,
                    context,
                    state,
                    camera_id=frame.camera_id,
                    frame_key=(tuple(frame.episode_ids), tuple(frame.frame_ids)),
                    saturation=frame.saturated_mask,
                    compute_auxiliary=False,
                )
            state = getattr(output, "state", state)
        tokens = self.encoder.readout_tokens(output, self.policy.config.tokens_per_camera)
        return summarize_prefix(self._token_values(tokens), len(window.camera_ids))

    def _representation_batch(self, plan, step: int) -> RepresentationGradientCache | None:
        if self.representation_mode != "independent_windows" or not self.loss_weights.get(
            "representation", 0
        ):
            return None
        if len(plan) < 2:
            raise ValueError("independent-window representation requires batch >= 2")
        # Keep prepass stochastic state/buffers out of the actual training pass.
        buffers = [
            (b, b.clone())
            for name, b in self.encoder.named_buffers()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        ]
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        observations, identities = [], set()
        try:
            with torch.random.fork_rng(devices=devices):
                for index, window in enumerate(self._training_windows(plan)):
                    self._seed_window(step, index)
                    observations.append(self._representation_observation(window))
                    identities.add(plan[index][0].source_id)
                    self._heartbeat(
                        "representation_cache", completed_windows=index + 1, total_windows=len(plan)
                    )
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
        if len(identities) < 2:
            raise ValueError("representation batch needs at least two different source sequences")
        cfg = self.config["loss"].get("representation", {})
        return cache_representation_gradients(
            torch.stack(observations),
            variance_weight=float(cfg.get("variance_weight", 25.0)),
            covariance_weight=float(cfg.get("covariance_weight", 1.0)),
        )

    def train_step(self, step: int) -> dict[str, Any]:
        """一次 optimizer update：逐窗口／源段累计实际均值梯度，最后统一 clip + step。

        每个窗口的 backward 权重是 ``1/len(plan)``（``optim.batch`` 段窗口的均值），与旧实现
        ``(micro_batch 均值 / micro_batch 数)`` 的有效尺度逐值相同；监督段之间做 TBPTT detach。
        """
        self._require_training_enabled()
        started = time.perf_counter()
        correspondence_loss = getattr(self.encoder, "correspondence_loss", None)
        if correspondence_loss is not None and hasattr(correspondence_loss, "set_step"):
            correspondence_loss.set_step(int(step))
        plan = self._training_plan(int(step))
        self.optimizer.zero_grad(set_to_none=True)
        representation_cache = self._representation_batch(plan, int(step))
        if representation_cache is not None:
            self.activity["representation_diag"] = dict(representation_cache.diagnostics)
        window_weight = 1.0 / float(len(plan))
        losses: list[float] = []
        term_totals: dict[str, list[float]] = defaultdict(list)
        applied_totals: dict[str, list[float]] = defaultdict(list)
        applied_losses: list[float] = []
        supervision_points = 0
        segments = 0
        gradient_fallback_windows = 0
        valid_loss_counts: dict[str, int] = defaultdict(int)
        forecast_skip_reasons: dict[str, int] = defaultdict(int)
        forecast_reports: list[dict[str, Any]] = []
        forecast_diag: dict[str, float] = {}
        self._heartbeat(
            "train", completed_windows=0, total_windows=len(plan), update_started_at=time.time()
        )
        for index, window in enumerate(self._training_windows(plan)):
            if representation_cache is not None:
                self._seed_window(int(step), index)
            loss, info = self._forward_window(
                window,
                train=True,
                backward_weight=window_weight,
                representation_cache=representation_cache,
                representation_index=index,
            )
            if loss is None:
                raise RuntimeError("本步没有任何可用的损失项：检查窗口、动作监督点与编码器输出")
            losses.append(float(loss.detach().cpu()))
            self._heartbeat("train", completed_windows=index + 1, total_windows=len(plan))
            supervision_points += int(info["supervision_points"])
            segments += int(info["segments"])
            for name, count in info.get("valid_loss_counts", {}).items():
                valid_loss_counts[name] += int(count)
            gradient_fallback_windows += int(
                info.get("valid_count_fallback", False)
                or info.get("gradient_eager_fallback", False)
            )
            if info.get("forecast") is not None:
                forecast_reports.append(info["forecast"])
                for reason, count in info["forecast"].get("skip_reasons", {}).items():
                    forecast_skip_reasons[reason] += int(count)
                for horizon, entries in (info["forecast"].get("diagnostics") or {}).items():
                    prefix = f"h{horizon}_"
                    for key, value in entries.items():
                        if key == "count":
                            continue
                        bucket = forecast_diag.setdefault(prefix + key, 0.0)
                        forecast_diag[prefix + key] = bucket + float(value)
            for name, value in info["terms"].items():
                if value is not None:
                    term_totals[name].append(value)
            if info.get("applied_train_loss") is not None:
                applied_losses.append(info["applied_train_loss"])
            for name, value in info.get("applied_terms", {}).items():
                if value is not None:
                    applied_totals[name].append(value)
        coverage = {
            name: {
                "windows": len(applied_totals.get(name, [])),
                "valid_items": valid_loss_counts.get(name, 0),
            }
            for name, weight in self.loss_weights.items()
            if weight > 0
        }
        self.activity["objective_coverage"] = coverage
        self.activity["forecast_skip_reasons"] = dict(forecast_skip_reasons)
        if self.config["loss"].get("require_all_objectives", False):
            missing = [
                name
                for name, counts in coverage.items()
                if counts["windows"] == 0 or counts["valid_items"] == 0
            ]
            if missing:
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Missing required objectives before optimizer update: {missing}; "
                    f"coverage={coverage}; forecast_skips={dict(forecast_skip_reasons)}"
                )
        del forecast_reports
        n_windows = max(1, len(plan))
        for key in list(forecast_diag):
            forecast_diag[key] = forecast_diag[key] / n_windows
        grad_clip = float(self.config["optim"].get("grad_clip", 1.0))
        # Reject non-finite gradients even when clipping is disabled.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            grad_clip if grad_clip > 0 else float("inf"),
            error_if_nonfinite=True,
        )
        self._require_training_enabled()
        align_optimizer_gradients(self.optimizer)
        self.optimizer.step()
        self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step = int(step) + 1
        mean_loss = float(np.mean(losses)) if losses else None
        if mean_loss is not None and math.isfinite(mean_loss):
            self.loss_ema = (
                mean_loss if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * mean_loss
            )
        return {
            "step": int(step),
            "train_loss": mean_loss,
            "loss_ema": self.loss_ema,
            "terms": {
                name: (float(sum(values) / n_windows) if values else None)
                for name, values in term_totals.items()
            },
            "applied_train_loss": float(np.mean(applied_losses)) if applied_losses else None,
            "applied_terms": {
                name: float(sum(values) / n_windows) for name, values in applied_totals.items()
            },
            "supervision_points": supervision_points,
            "segments": segments,
            "gradient_fallback_windows": gradient_fallback_windows,
            "forecast_diag": forecast_diag,
            "objective_coverage": coverage,
            "forecast_skip_reasons": dict(forecast_skip_reasons),
            "representation_diag": (
                None if representation_cache is None else representation_cache.diagnostics
            ),
            "gradient_norm_before_clip": float(gradient_norm.detach()),
            "window_plan": [
                {"source_id": entry.source_id, "path": entry.relpath, "start": start}
                for entry, start in plan
            ],
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "step_seconds": time.perf_counter() - started,
            "gpu_memory": _gpu_memory(),
        }

    def profile_window(self, window: WindowBatch | None = None) -> dict[str, Any]:
        """Profile the actual full-window objective without changing weights or optimizer state."""
        self._require_training_enabled()
        entry, start = self.train_candidates[0] if window is None else (None, None)
        active = window or load_window_batch(
            entry,
            start=start,
            burn_in=self.window["burn_in"],
            gradient=self.window["gradient"],
            supervision_points=self.window["supervision_points"],
            proprio_history=self.proprio_history,
        )
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss, info = self._forward_window(active, train=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "platform": self.platform,
            "window": {
                "burn_in": self.window["burn_in"],
                "gradient": len(active) - self.window["burn_in"],
                "frames": len(active),
            },
            "segments": info["segments"],
            "terms": info["terms"],
            "memory_settings": self.memory_settings(),
            "stages": {
                "window_forward_backward": {
                    "peak_memory_gb": _gpu_memory().get("peak_gb"),
                    "seconds": elapsed,
                }
            },
            "optimizer_updates": 0,
            "note": "same full-window objective; no optimizer update; stub backend if injected",
        }

    # -- 开发集动作 loss -----------------------------------------------------------

    @contextlib.contextmanager
    def _fixed_dev_math(self):
        """Use one native deterministic reference for fixed-dev/reload comparisons.

        Compiled/eager and benchmark-selected kernels gave different dev values
        despite identical checkpoint tensors in the real-GPU controls. Keep
        this reference out of optimization and restore every training setting.
        """
        if self.device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
            ":4096:8",
            ":16:8",
        ):
            raise ValueError("fixed CUDA dev requires CUBLAS_WORKSPACE_CONFIG=:4096:8 at startup")
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        precision = torch.get_float32_matmul_precision()
        switched = []
        try:
            torch.use_deterministic_algorithms(True)
            torch.set_float32_matmul_precision("highest")
            for module, name, eager in self._eager_methods:
                switched.append((module, name, getattr(module, name)))
                setattr(module, name, eager)
            with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
                yield
        finally:
            for module, name, method in reversed(switched):
                setattr(module, name, method)
            torch.set_float32_matmul_precision(precision)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    def evaluate_dev_action_loss(self) -> dict[str, Any]:
        if not self.dev_windows:
            return {"dev_loss": None, "windows": 0, "points": 0}
        was_training = self.encoder.training
        backend_was_training = self.backend.training
        forecast_was_enabled = self.forecast_enabled
        previous_actions_only = getattr(self, "_evaluating_actions_only", False)
        auxiliary_dev = bool(self.monitor.get("auxiliary_dev", False))
        self.forecast_enabled = forecast_was_enabled if auxiliary_dev else False
        self._evaluating_actions_only = True
        self.encoder.eval()
        self.backend.eval()
        started = time.perf_counter()
        losses: list[float] = []
        points = 0
        auxiliary_forecasts, representation_observations = [], []
        # 固定 flow-matching 时间/噪声，并恢复训练 RNG；否则同一 checkpoint 的 dev 曲线
        # 也会随机变动，验证频率还会改变后续训练的随机序列。
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with self._fixed_dev_math(), torch.random.fork_rng(devices=devices), torch.no_grad():
                random.seed(self.seed)
                np.random.seed(self.seed % (2**32))
                torch.random.default_generator.manual_seed(self.seed)
                if devices:
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(self.seed)
                for entry, start in self.dev_windows:
                    window = load_window_batch(
                        entry,
                        start=start,
                        burn_in=self.window["burn_in"],
                        gradient=self.window["gradient"],
                        supervision_points=self.window["supervision_points"],
                        proprio_history=self.proprio_history,
                    )
                    if auxiliary_dev:
                        loss, info = self._forward_window(
                            window,
                            train=False,
                            representation_observations=representation_observations,
                        )
                    else:
                        loss, info = self._forward_windows([window], train=False)
                    if auxiliary_dev and info.get("forecast"):
                        auxiliary_forecasts.append(info["forecast"])
                    if loss is not None and info["supervision_points"]:
                        losses.append(float(info["terms"]["action"]))
                        points += int(info["supervision_points"])
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            self.encoder.train(was_training)
            self.backend.train(backend_was_training)
            self.forecast_enabled = forecast_was_enabled
            self._evaluating_actions_only = previous_actions_only
        dev = float(np.mean(losses)) if losses else None
        self.dev_loss = dev
        if auxiliary_dev:
            diagnostics = {}
            for horizon in self.forecast_horizons:
                entries = [
                    report["diagnostics"][str(horizon)]
                    for report in auxiliary_forecasts
                    if str(horizon) in report.get("diagnostics", {})
                ]
                count = sum(row["count"] for row in entries)
                if count:
                    diagnostics[str(horizon)] = {
                        key: sum(row[key] * row["count"] for row in entries) / count
                        for key in entries[0]
                        if key != "count"
                    }
            representation = (
                cache_representation_gradients(torch.stack(representation_observations)).diagnostics
                if len(representation_observations) >= 2
                else None
            )
            auxiliary = {
                "step": self.step,
                "windows": len(losses),
                "forecast": diagnostics,
                "representation": representation,
            }
            self.activity["auxiliary_validation"] = auxiliary
            write_json_atomic(self.run_dir / "auxiliary-validation.json", auxiliary)
            with (self.run_dir / "auxiliary-validation.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(auxiliary, ensure_ascii=False) + "\n")
        return {
            "dev_loss": dev,
            "windows": len(losses),
            "points": points,
            "seconds": time.perf_counter() - started,
        }

    # -- checkpoint ----------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "stage": self.config.get("stage", "S2"),
            "platform": self.platform,
            "arm": self.config.get("arm", "D"),
            "seed": self.seed,
            "step": self.step,
            "config": self.config,
            "config_fingerprint": config_fingerprint(self.config),
            "structural_fingerprint": structural_fingerprint(self.config),
            "loss_fingerprint": loss_fingerprint(self.config),
            "loss_definition": self.config.get("loss_definition"),
            "encoder": self.encoder.state_dict(),
            "backend": self.backend.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "loss_ema": self.loss_ema,
            "dev_loss": self.dev_loss,
            "window": dict(self.window),
            "extra": {
                "initial_dev_loss": self.initial_dev_loss,
                "best_dev_loss": self.best_dev_loss,
                "last_checkpoint_step": self.last_checkpoint_step,
                **dict(extra or {}),
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return str(path)

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require_current_implementation(payload)
        stored_config = payload.get("config") or {}
        stored_structural = payload.get("structural_fingerprint") or structural_fingerprint(
            stored_config
        )
        current_structural = structural_fingerprint(self.config)
        if stored_structural != current_structural:
            raise ValueError(
                "checkpoint 的模型／优化器结构（stage/platform/arm/model/optim）与当前配置不一致；"
                "结构与优化器必须一致才能 resume（损失定义可以有意变更）"
            )
        stored_loss = payload.get("loss_fingerprint") or loss_fingerprint(stored_config)
        current_loss = loss_fingerprint(self.config)
        if stored_loss != current_loss:
            # 允许的有意变更：记录 from/to，写进 resume-notes.md 与 train-report.json
            self.loss_definition_change = {
                "from_fingerprint": stored_loss,
                "to_fingerprint": current_loss,
                "from_definition": stored_config.get("loss_definition"),
                "to_definition": self.config.get("loss_definition"),
                "from_loss": stored_config.get("loss"),
                "to_loss": self.config.get("loss"),
            }
        self.encoder.load_state_dict(payload["encoder"], strict=strict)
        self.backend.load_state_dict(payload["backend"], strict=strict)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload.get("scaler", {}))
        self.step = int(payload["step"])
        self.loss_ema = payload.get("loss_ema")
        self.dev_loss = payload.get("dev_loss")
        saved = payload.get("extra", {})
        self.initial_dev_loss = saved.get("initial_dev_loss")
        self.best_dev_loss = saved.get("best_dev_loss")
        self.last_checkpoint_step = self.step
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        correspondence_loss = getattr(self.encoder, "correspondence_loss", None)
        if correspondence_loss is not None and hasattr(correspondence_loss, "set_step"):
            correspondence_loss.set_step(self.step)
        return payload

    # -- 主循环 -------------------------------------------------------------------

    def run(self, *, steps: int | None = None, resume: str | None = None) -> dict[str, Any]:
        self._require_training_enabled()
        steps = int(steps or self.budget_steps)
        if resume:
            payload = self.load_checkpoint(resume)
            self.resumed_from = {
                "checkpoint": str(resume),
                "step": int(payload.get("step", 0)),
            }
            self.log(f"resumed from {resume} at step {self.step}")
            self._write_resume_notes(resume, payload)
        log_every = int(self.monitor.get("log_every", 20))
        checkpoint_every = int(self.monitor.get("checkpoint_every", 5000))
        eval_every = int(self.monitor.get("eval_every", 1000))
        metrics_path = self.run_dir / "metrics.jsonl"
        progress_path = self.run_dir / "progress.json"
        started = time.perf_counter()
        self._maybe_ckpt: list[dict[str, Any]] = []
        if bool(self.monitor.get("profile", False)) and self.step == 0:
            profile = self.profile_window()
            write_json_atomic(self.run_dir / "profile.json", profile)
            with (self.run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"step": 0, "phase": "profile", **profile},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
            self.log(
                "profile peak_gb="
                f"{profile['stages']['window_forward_backward']['peak_memory_gb']} "
                f"step_s={profile['stages']['window_forward_backward']['seconds']}"
            )
        write_json_atomic(self.run_dir / "config.used.json", self.config)
        if self.monitor.get("initial_dev", False) and self.step == 0:
            self._heartbeat("validate")
            dev = self.evaluate_dev_action_loss()
            self.dev_loss = dev["dev_loss"]
            self.initial_dev_loss = self.best_dev_loss = self.dev_loss
            self._write_monitor(metrics_path, progress_path, steps, {}, phase="validate")
        for step in range(self.step, steps):
            cycle_started = time.perf_counter()
            info = self.train_step(step)
            if info["train_loss"] is not None:
                self.epoch_losses.append(info["train_loss"])
            self.log(
                f"step {self.step:7d} loss={_fmt(info['train_loss'])} "
                f"action={_fmt(info['terms'].get('action'))} "
                f"lr={info['lr']:.2e} {info['step_seconds']:.2f}s"
            )
            improved = False
            early = self.step <= int(self.monitor.get("early_until", 0))
            cadence = int(self.monitor.get("eval_every_early", eval_every)) if early else eval_every
            if self.step % max(1, cadence) == 0:
                self._heartbeat("validate")
                dev = self.evaluate_dev_action_loss()
                self.dev_loss = dev["dev_loss"]
                improved = self.dev_loss is not None and (
                    self.best_dev_loss is None or self.dev_loss < self.best_dev_loss
                )
                if improved:
                    self.best_dev_loss = self.dev_loss
                self.log(f"fixed dev action loss at step {self.step}: {_fmt(self.dev_loss)}")
            should_save = self.step % max(1, checkpoint_every) == 0 or (
                self.step == 1 and self.monitor.get("checkpoint_first", False)
            )
            if should_save or (improved and self.monitor.get("save_best", False)):
                self._heartbeat("saving")
                self.last_checkpoint_step = self.step
                if should_save:
                    path = self.save_checkpoint(self.run_dir / f"checkpoint-step{self.step:07d}.pt")
                    self._maybe_ckpt.append({"path": path, "step": self.step})
                    keep = int(self.monitor.get("keep_checkpoints", 0))
                    if keep > 0:
                        for old in sorted(self.run_dir.glob("checkpoint-step*.pt"))[:-keep]:
                            old.unlink()
                if improved and self.monitor.get("save_best", False):
                    self.save_checkpoint(self.run_dir / "checkpoint-best.pt")
            self.cycle_speeds.append(time.perf_counter() - cycle_started)
            info["step_seconds"] = float(np.mean(self.cycle_speeds))
            if self.step % max(1, log_every) == 0 or should_save or improved:
                self._write_monitor(metrics_path, progress_path, steps, info)
            self._heartbeat("train")
        final = self.save_checkpoint(self.run_dir / "checkpoint-final.pt")
        report = {
            "stage": self.config.get("stage", "S2"),
            "platform": self.platform,
            "arm": self.config.get("arm", "D"),
            "seed": self.seed,
            "steps": self.step,
            "budget_steps": steps,
            "elapsed_s": time.perf_counter() - started,
            "final_checkpoint": final,
            "checkpoints": self._maybe_ckpt,
            "loss_ema": self.loss_ema,
            "dev_loss": self.dev_loss,
            "optimizer": self.optimizer_report,
            "data": self.data_index.report(),
            "train_windows": len(self.train_candidates),
            "dev_windows": len(self.dev_windows),
            "loss_weights": self.loss_weights,
            "loss_definition": self.config.get("loss_definition"),
            "loss_definition_notes": (
                "v4 supports gradient-cached VICReg across independent window descriptors; "
                "variance/covariance coefficients are explicit. Source posterior/R retain "
                "forecast gradients and targets are stopped. Historical curves use different "
                "objectives and must not be joined. See config for initialization and precision."
            ),
            "resumed_from": getattr(self, "resumed_from", None),
            "loss_definition_change": getattr(self, "loss_definition_change", None),
            "memory": self.memory_settings(),
        }
        write_json_atomic(self.run_dir / "train-report.json", report)
        self._write_monitor(
            metrics_path, progress_path, steps, {"train_loss": self.loss_ema}, phase="done"
        )
        return report

    def _write_monitor(
        self,
        metrics_path: Path,
        progress_path: Path,
        budget: int,
        info: dict[str, Any],
        *,
        phase: str = "train",
    ) -> None:
        row = {
            "step": self.step,
            "phase": phase,
            "loss_definition": self.config.get("loss_definition"),
            "train_loss": info.get("train_loss", self.loss_ema),
            "dev_loss": self.dev_loss,
            "terms": info.get("terms"),
            "applied_train_loss": info.get("applied_train_loss"),
            "applied_terms": info.get("applied_terms"),
            "lr": info.get("lr"),
            "gradient_norm_before_clip": info.get("gradient_norm_before_clip"),
            "representation_diag": info.get("representation_diag"),
            "window_plan": info.get("window_plan"),
            "gradient_fallback_windows": info.get("gradient_fallback_windows"),
            "objective_coverage": info.get("objective_coverage"),
            "forecast_skip_reasons": info.get("forecast_skip_reasons"),
            "step_seconds": info.get("step_seconds"),
            "gpu_memory_gb": (info.get("gpu_memory") or {}).get("peak_gb"),
            "gpu_live_gb": (info.get("gpu_memory") or {}).get("live_gb"),
            "gpu_reserved_gb": (info.get("gpu_memory") or {}).get("reserved_gb"),
            "forecast_diag": {
                key: _finite_or_none(value)
                for key, value in (info.get("forecast_diag") or {}).items()
            }
            or None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        write_progress(
            progress_path,
            config=self.config,
            phase=phase,
            step=self.step,
            budget_steps=budget,
            train_loss=row["train_loss"],
            dev_loss=self.dev_loss,
            step_seconds=info.get("step_seconds"),
            gpu_memory_gb=(info.get("gpu_memory") or {}).get("peak_gb"),
            eval_summary=self._eval_summary(),
            artifacts=self._artifacts(),
            log_tail=list(self.log_tail),
            activity={
                **self.activity,
                "initial_dev_loss": self.initial_dev_loss,
                "best_dev_loss": self.best_dev_loss,
                "last_checkpoint_step": self.last_checkpoint_step,
            },
        )

    def _eval_summary(self) -> dict[str, Any] | None:
        results = self.run_dir / "eval_results.jsonl"
        if not results.exists():
            return None
        plan = self.config.get("eval", {}).get("planned")
        return summarize_eval_results(results, planned=plan)

    def _artifacts(self, limit: int = 20) -> list[str]:
        directory = self.run_dir / "artifacts"
        if not directory.exists():
            return []
        return sorted(
            str(path.relative_to(self.run_dir)) for path in directory.rglob("*") if path.is_file()
        )[:limit]

    def memory_settings(self) -> dict[str, Any]:
        """本运行实际生效的内存旋钮（写进 train-report.json，不再靠回忆）。"""
        return {
            "gradient_checkpointing": bool(self.gradient_checkpointing),
            "raft_resize_hw": (None if self.raft_resize_hw is None else list(self.raft_resize_hw)),
            "batch": self.batch,
            "window": dict(self.window),
            "forecast_anchor_stride": int(self.config["loss"].get("forecast_anchor_stride", 1)),
            "per_supervision_point_backward": False,
            "per_segment_term_gradients": True,
            "max_forecast_delay_frames": max(self.forecast_horizons, default=0),
            "micro_batch": self.micro_batch,
            "compiled_autograd": self.compiled_autograd_settings,
            "representation_mode": self.representation_mode,
        }

    def _write_resume_notes(self, checkpoint: str, payload: dict[str, Any]) -> None:
        """断点续跑说明：口径变更 + 断点身份，避免把两段曲线直接比较。"""
        notes = self.run_dir / "resume-notes.md"
        lines = [
            f"# resume notes ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})",
            "",
            f"- resumed from: `{checkpoint}` at step {payload.get('step')}",
            f"- loss_definition: `{self.config.get('loss_definition')}`",
            (
                "- loss 定义变更（有意，已放行）："
                f"`{change['from_definition']}` → `{change['to_definition']}`"
                if (change := self.loss_definition_change)
                else "- loss 定义未变"
            ),
            f"- implementation_version: `{IMPLEMENTATION_VERSION}`",
            "- forecast 源端活梯度、未来目标停梯度；辅助项包含所有梯度帧，按实际有效项归约。",
            f"- representation_mode: `{self.representation_mode}`；"
            "不同定义的曲线不可直接比较或拼接。",
            f"- memory: `{json.dumps(self.memory_settings(), ensure_ascii=False)}`",
            "",
        ]
        notes.write_text("\n".join(lines))

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}] {message}"
        self.log_tail.append(line)
        print(line, flush=True)


@dataclass
class _ScheduleFrame:
    """一帧、一段序列的多相机输入（WS-C ``step`` 的单帧调用面）。

    属性名与 `RawVLAFeaturePolicy.encode_observation` 读取的字段一致，因此同一对象既能喂
    ``encoder.step``，也能喂 ``policy.forward_from_observation``。
    """

    values: list[torch.Tensor]  # 每个相机一项 [C,H,W]
    camera_id: list[str]
    frame_ids: tuple[int, ...]
    episode_ids: tuple[str, ...]
    saturated_mask: list[torch.Tensor] | None = None
    valid_mask: None = None
    bad_mask: None = None
    new_measurement: bool = True


def _detach_tree(value: Any) -> Any:
    """递归切断张量图（容器保持类型），不改值。"""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _detach_state(state: Any) -> Any:
    """TBPTT 边界：把状态里**所有**张量字段切断，不只编码器 ``detach()`` 覆盖的那几个。

    WS-C 的 ``EncoderState.detach`` 只处理模式均值／协方差／权重；``SupportState.xy`` 是
    ``xy_prev + flow`` 的结果，仍挂着该段的 RAFT 图。如果只调用编码器的 ``detach()``，上一个
    监督段的激活会被 ``xy`` 一路引用，显存随段数线性增长（远端 96 GiB 实测即此现象）。
    """
    if state is None:
        return None
    if dataclasses.is_dataclass(state):
        per_camera = getattr(state, "per_camera", None)
        if isinstance(per_camera, dict):
            return dataclasses.replace(
                state, per_camera={cam: _detach_state(item) for cam, item in per_camera.items()}
            )
        return dataclasses.replace(
            state,
            **{
                field.name: _detach_tree(getattr(state, field.name))
                for field in dataclasses.fields(state)
            },
        )
    method = getattr(state, "detach", None)
    if callable(method):
        return method()
    return _detach_tree(state)


def _merge_window_info(infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把逐段窗口的信息合并成一个 micro-batch 报告（缺失项仍记 null）。"""
    if not infos:
        raise ValueError("没有窗口信息可合并")
    terms: dict[str, float | None] = {}
    for name in infos[0]["terms"]:
        values = [info["terms"].get(name) for info in infos]
        values = [value for value in values if value is not None]
        terms[name] = float(sum(values) / len(infos)) if values else None
    sources: dict[str, str] = {}
    for info in infos:
        sources.update(info["term_sources"])
    forecast_reports = [info.get("forecast") for info in infos if info.get("forecast") is not None]
    forecast: dict[str, Any] | None = None
    diagnostics: dict[str, dict[str, float]] = {}
    for report in forecast_reports:
        for horizon, entries in (report.get("diagnostics") or {}).items():
            bucket = diagnostics.setdefault(horizon, {})
            bucket["count"] = bucket.get("count", 0.0) + float(entries.get("count", 0.0))
            for key, value in entries.items():
                if key == "count":
                    continue
                bucket[key] = bucket.get(key, 0.0) + float(value) * float(entries.get("count", 0.0))
    for entries in diagnostics.values():
        count = max(1.0, entries.get("count", 0.0))
        for key in list(entries):
            if key != "count":
                entries[key] = entries[key] / count
    if forecast_reports:
        reasons: dict[str, int] = defaultdict(int)
        for report in forecast_reports:
            for reason, count in report["skip_reasons"].items():
                reasons[reason] += int(count)
        forecast = {
            "created": int(sum(report["created"] for report in forecast_reports)),
            "settled": int(sum(report["settled"] for report in forecast_reports)),
            "skipped": int(sum(report["skipped"] for report in forecast_reports)),
            "skip_reasons": dict(reasons),
            "diagnostics": diagnostics,
        }
    return {
        "terms": terms,
        "applied_terms": dict(terms),
        "applied_train_loss": float(np.mean([info["applied_train_loss"] for info in infos])),
        "valid_loss_counts": {
            n: sum(i.get("valid_loss_counts", {}).get(n, 0) for i in infos) for n in terms
        },
        "term_sources": sources,
        "supervision_points": int(sum(info["supervision_points"] for info in infos)),
        "forecast": forecast,
        "windows": len(infos),
    }


def build_v11_encoder(model_cfg: dict[str, Any], *, platform: str, arm: str = "D") -> nn.Module:
    """构造 WS-C 的 ``N04V11Encoder``：JSON ``model`` 段按 ``EncoderV11Config`` 字段过滤。

    未识别的键（骨干结构、token 预算、模块开关等）在这里忽略并返回在 ``ignored_keys`` 里，
    不传给编码器构造函数；四臂开关由 ``arm`` 走 WS-C 自己的 ``ARM_SPECS``。
    """
    from ..frameworks.n04.encoder_v11 import EncoderV11Config, build_encoder

    spec = PLATFORM_SPECS[platform]
    fields = {
        name: value
        for name, value in model_cfg.items()
        if name in _ENCODER_CONFIG_FIELDS and value is not None
    }
    fields.setdefault("cameras", tuple(spec.camera_ids))
    fields["arm"] = arm
    if "scale_name" not in fields:
        raise ValueError(f"{platform} 缺少 model.scale_name（平台固定缩放名）")
    ignored = sorted(set(model_cfg) - _ENCODER_CONFIG_FIELDS - {"builder"})
    encoder = build_encoder(EncoderV11Config(**fields))
    encoder.config_ignored_keys = ignored
    return encoder


def _finite_or_none(value: Any) -> float | None:
    """NaN/Inf 按约定转 null（仪表分母为空时 WS-B 会返回 NaN）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def _gpu_memory() -> dict[str, float] | None:
    """三种口径一起报：peak 是单调最大值（不能用来判断是否增长），live/reserved 才是当前占用。"""
    if not torch.cuda.is_available():
        return None
    return {
        "peak_gb": float(torch.cuda.max_memory_allocated() / 1e9),
        "live_gb": float(torch.cuda.memory_allocated() / 1e9),
        "reserved_gb": float(torch.cuda.memory_reserved() / 1e9),
    }


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """在构造模型之前设置；只要求可复现采样，不要求逐位确定性。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, data_root=args.data_root)
    seed = int(args.seed if args.seed is not None else config.get("seed", 101))
    config["seed"] = seed
    trainer = V11Trainer(config, seed=seed, run_dir=args.output, device=args.device)
    report = trainer.run(resume=args.resume)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "config"}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())

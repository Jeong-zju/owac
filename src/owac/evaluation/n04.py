"""N04 机制评测：运动矩阵、成对反例、时延与状态成本。

主指标（只在**合成有真值**的数据上计算分解准确率；真实数据无标签时只报可测代理）：

* 持续定位误差：稳定表示与当前测量特征之间的对齐误差，以及目标质心定位误差。
* 目标移动后的恢复帧数：目标停止后，表示重新落到正确位置所需的帧数。
* 误把自运动当外部变化的比率：纯相机运动条件下 change 分支的响应。
* 误抑制真实物体运动的比率：纯物体运动条件下 change 分支的响应。
* 照明突变误报：只有照明变化时 change 分支的响应。
* 遮挡恢复：遮挡解除后重新获得有效表示的帧数。
* 动作修正幅度与时延：候选相对基线的动作/表示修正量。
* warp error、预测 residual、mask 覆盖、p50/p95 时延与状态成本。

机制矩阵条件由 `SyntheticSequenceGenerator` 独立控制；成对反例包括
“相同动作上下文 / 不同物体运动”与“相同视觉变化 / 不同 ego 上下文”。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..baselines.n04 import build_baseline
from ..frameworks.n04.model import N04Encoder
from ..frameworks.n04.synthetic import SceneConfig, SyntheticFrame, SyntheticSequenceGenerator
from ..training.n04 import build_context, config_from_json, load_checkpoint

__all__ = ["ConditionResult", "MechanismReport", "evaluate_condition", "run_mechanism_suite"]

#: 机制矩阵：每个条件独立控制一组干预。
MECHANISM_CONDITIONS: tuple[str, ...] = (
    "camera_only",  # 相机动 / 物体不动
    "object_only",  # 相机不动 / 物体动
    "both",  # 二者都动
    "illumination",  # 只有照明变化
    "same_direction",  # 二者同向近同速（不可识别）
    "occlusion",  # 遮挡后重新暴露
    "calibration_error",  # 错误外参映射
    "action_delay",  # 动作延迟
    "exposure_step",  # 曝光突变与饱和
)

#: 成对反例：(同一对条件, 被固定的量, 被改变的量)。
COUNTERFACTUAL_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("camera_only", "object_only", "相同相机/动作上下文，物体运动不同"),
    ("object_only", "illumination", "相同测量变化幅度，ego 上下文不同"),
)


@dataclass
class ConditionResult:
    """单个机制条件的结果。"""

    condition: str
    location_error: float
    change_response: float
    change_at_object: float
    change_off_object: float
    warp_error: float
    prediction_residual: float
    mask_coverage: float
    recovery_frames: int | None
    latency_p50_ms: float
    latency_p95_ms: float
    state_bytes: int
    frames: int
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "location_error": self.location_error,
            "change_response": self.change_response,
            "change_at_object": self.change_at_object,
            "change_off_object": self.change_off_object,
            "warp_error": self.warp_error,
            "prediction_residual": self.prediction_residual,
            "mask_coverage": self.mask_coverage,
            "recovery_frames": self.recovery_frames,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "state_bytes": self.state_bytes,
            "frames": self.frames,
            "extra": self.extra,
        }


@dataclass
class MechanismReport:
    conditions: list[ConditionResult] = field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    contrast: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "conditions": [c.as_dict() for c in self.conditions],
            "counterfactuals": self.counterfactuals,
            "contrast": self.contrast,
            "notes": self.notes,
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def _measurement_centroid(mask: torch.Tensor) -> torch.Tensor | None:
    total = mask.sum()
    if float(total) <= 0:
        return None
    ys, xs = torch.meshgrid(
        torch.arange(mask.shape[-2], dtype=torch.float32),
        torch.arange(mask.shape[-1], dtype=torch.float32),
        indexing="ij",
    )
    cy = float((mask[0, 0] * ys).sum() / total)
    cx = float((mask[0, 0] * xs).sum() / total)
    return torch.tensor([cx, cy])


def evaluate_condition(
    encoder: N04Encoder,
    *,
    condition: str,
    scene: SceneConfig,
    length: int = 16,
    batch: int = 2,
    seed: int = 0,
    warmup: int = 2,
    pixel_stride: float = 1.0,
) -> ConditionResult:
    """在一个机制条件上跑完整序列并汇总指标。"""
    gen = SyntheticSequenceGenerator(scene)
    sequence = gen.sequence(batch=batch, length=length, condition=condition, seed=seed)
    states = encoder.init_state(batch)
    previous: SyntheticFrame | None = None
    location_errors: list[float] = []
    change_responses: list[float] = []
    change_at: list[float] = []
    change_off: list[float] = []
    warp_errors: list[float] = []
    residuals: list[float] = []
    coverages: list[float] = []
    latencies: list[float] = []
    centroid_errors: list[float] = []
    state_bytes = 0
    for t, frame in enumerate(sequence):
        context = build_context(frame, previous)
        started = time.perf_counter()
        out = encoder.step(frame.measured, context, states["head"], camera_id="head")
        latencies.append((time.perf_counter() - started) * 1000.0)
        states = {"head": out.state}
        if t < warmup:
            previous = frame
            continue
        measured = encoder.encode_measurement(frame.measured)
        stable = out.state.stable
        stable0 = stable[0] if isinstance(stable, list) else stable
        location_errors.append(float(F.l1_loss(stable0, measured)))
        #: 基线可能没有创新分解（例如单帧 CNN）：此时预测残差记为 0 并在记录中说明，
        #: 而不是给它编一个残差。
        if out.innovation is not None and out.innovation.residual is not None:
            residuals.append(float(out.innovation.residual.abs().mean()))
        coverages.append(float(out.validity.mean()))
        change = out.state.change
        if change is not None:
            change0 = change[0] if isinstance(change, list) else change
            #: **逐像素**幅值图：先压成标量再做掩码加权会让“目标处 / 目标外”恒等，
            #: 那样的对比没有任何分辨力（曾把 change_contrast 算成 4e-12）。
            magnitude = change0.abs().mean(dim=1, keepdim=True)  # [B,1,h,w]
            change_responses.append(float(magnitude.mean()))
            mask = frame.object_mask
            if mask.shape[-2:] != change0.shape[-2:]:
                mask = F.adaptive_avg_pool2d(mask, change0.shape[-2:])
            if mask.shape[1] != magnitude.shape[1]:
                mask = mask.expand(-1, magnitude.shape[1], -1, -1)
            if float(mask.mean()) > 1e-3:
                at = float((magnitude * mask).sum() / mask.sum().clamp_min(1e-6))
                inverse = (1.0 - mask).clamp(0, 1)
                off = float((magnitude * inverse).sum() / inverse.sum().clamp_min(1e-6))
                change_at.append(at)
                change_off.append(off)
        if out.state.forward_flow is not None:
            #: 真值 flow 在测量像素上，模型输出在特征像素上。
            target = frame.oracle_camera_flow / max(pixel_stride, 1e-6)
            flow = out.state.forward_flow
            if flow.shape[1:3] != target.shape[1:3]:
                flow = F.interpolate(
                    flow.permute(0, 3, 1, 2),
                    size=target.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            warp_errors.append(float(F.l1_loss(flow, target)))
        centroid = _measurement_centroid(frame.object_mask)
        if centroid is not None:
            ys, xs = torch.meshgrid(
                torch.arange(stable0.shape[-2], dtype=torch.float32),
                torch.arange(stable0.shape[-1], dtype=torch.float32),
                indexing="ij",
            )
            weight = stable0[0].abs().mean(dim=0)
            total = weight.sum().clamp_min(1e-6)
            cy = (
                float((weight * ys).sum() / total) / stable0.shape[-2] * frame.object_mask.shape[-2]
            )
            cx = (
                float((weight * xs).sum() / total) / stable0.shape[-1] * frame.object_mask.shape[-1]
            )
            centroid_errors.append(
                float(torch.tensor([cx - float(centroid[0]), cy - float(centroid[1])]).norm())
            )
        state_bytes = out.state.state_bytes()
        previous = frame
    recovery = _recovery_frames(
        encoder, scene, condition=condition, seed=seed, batch=batch, pixel_stride=pixel_stride
    )
    return ConditionResult(
        condition=condition,
        location_error=_mean(location_errors),
        change_response=_mean(change_responses),
        change_at_object=_mean(change_at),
        change_off_object=_mean(change_off),
        warp_error=_mean(warp_errors),
        prediction_residual=_mean(residuals),
        mask_coverage=_mean(coverages),
        recovery_frames=recovery,
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        state_bytes=state_bytes,
        frames=len(sequence),
        extra={
            "centroid_error_px": _mean(centroid_errors),
            "change_contrast": _mean(change_at) - _mean(change_off),
            #: 归一化对比：量纲无关，跨条件可比；“目标处能量占比”。
            "change_contrast_normalised": _normalised_contrast(change_at, change_off),
        },
    )


def _recovery_frames(
    encoder: N04Encoder,
    scene: SceneConfig,
    *,
    condition: str,
    seed: int,
    batch: int,
    pixel_stride: float = 1.0,
) -> int | None:
    """目标停止移动后，表示重新稳定到与测量一致所需的帧数。

    只在 ``object_only`` 这类“目标先动后停”的条件下有意义；其他条件返回 None。
    """
    if condition != "object_only":
        return None
    gen = SyntheticSequenceGenerator(scene)
    moving = gen.sequence(batch=batch, length=8, condition="object_only", seed=seed)
    stopped_scene = SceneConfig(**{**scene.__dict__, "object_velocity": 0.0})
    stopped = SyntheticSequenceGenerator(stopped_scene).sequence(
        batch=batch, length=8, condition="object_only", seed=seed
    )
    states = encoder.init_state(batch)
    previous = None
    for frame in moving:
        out = encoder.step(
            frame.measured, build_context(frame, previous), states["head"], camera_id="head"
        )
        states = {"head": out.state}
        previous = frame
    #: 停住之后，残差降到停止瞬间的一半所需帧数
    baseline = None
    for t, frame in enumerate(stopped):
        out = encoder.step(
            frame.measured, build_context(frame, previous), states["head"], camera_id="head"
        )
        states = {"head": out.state}
        previous = frame
        residual = _residual_of(encoder, out, frame)
        if baseline is None:
            baseline = residual
        elif residual <= 0.5 * baseline:
            return t
    return len(stopped)


def run_mechanism_suite(
    encoder: N04Encoder,
    *,
    base_scene: SceneConfig,
    conditions: tuple[str, ...] = MECHANISM_CONDITIONS,
    length: int = 16,
    batch: int = 2,
    seed: int = 0,
    pixel_stride: float = 1.0,
) -> MechanismReport:
    report = MechanismReport()
    for condition in conditions:
        report.conditions.append(
            evaluate_condition(
                encoder,
                condition=condition,
                scene=base_scene,
                length=length,
                batch=batch,
                seed=seed,
                pixel_stride=pixel_stride,
            )
        )
    lookup = {c.condition: c for c in report.conditions}
    for left, right, description in COUNTERFACTUAL_PAIRS:
        if left not in lookup or right not in lookup:
            continue
        report.counterfactuals.append(
            {
                "pair": [left, right],
                "description": description,
                "change_left": lookup[left].change_response,
                "change_right": lookup[right].change_response,
                "ratio": lookup[right].change_response / max(lookup[left].change_response, 1e-9),
                "location_left": lookup[left].location_error,
                "location_right": lookup[right].location_error,
            }
        )
    if "camera_only" in lookup and "object_only" in lookup:
        camera, object_motion = lookup["camera_only"], lookup["object_only"]
        #: 方向：**越低越好**。相机自运动应当被传播解释掉（响应低），物体运动应当保留
        #: （响应高）。没有变化分支的基线（响应恒为 0）在这里必须是 NaN，不能报成 0.000
        #: 而被误读为“完美分离”。
        report.contrast["camera_vs_object_change"] = _ratio_or_nan(
            camera.change_response, object_motion.change_response
        )
    if "illumination" in lookup and "object_only" in lookup:
        #: 方向：**越低越好**。照明变化不应被当成物体运动（误报率）。
        report.contrast["illumination_false_alarm"] = _ratio_or_nan(
            lookup["illumination"].change_response, lookup["object_only"].change_response
        )
    report.notes.append(
        "分解准确率只在合成有真值的数据上计算；真实 RawVLA NPZ 无这些标签，"
        "只能报告可测代理与闭环，不输出凭空的分解准确率。"
    )
    report.notes.append(
        "contrast 方向：camera_vs_object_change 与 illumination_false_alarm 都是越低越好；"
        "没有变化分支的模型记 NaN（不是 0）。"
    )
    return report


def _ratio_or_nan(numerator: float, denominator: float) -> float:
    """两项都必须>0 才有定义；否则返回 NaN，不把“没有该分支”记成 0。"""
    if not (numerator > 0.0 and denominator > 0.0):
        return float("nan")
    return float(numerator / denominator)


def _residual_of(encoder, output, frame: SyntheticFrame) -> float:
    """本步的预测残差。没有创新分解的基线退化为“稳定表示与当前测量特征的差异”。"""
    if output.innovation is not None and output.innovation.residual is not None:
        return float(output.innovation.residual.abs().mean())
    measured = encoder.encode_measurement(frame.measured)
    stable = output.state.stable
    stable0 = stable[0] if isinstance(stable, list) else stable
    return float((stable0 - measured).abs().mean())


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _normalised_contrast(at: list[float], off: list[float]) -> float:
    """``(at - off) / (at + off)``：目标处变化能量相对目标外的归一化对比。

    只用绝对幅值无法跨条件比较（不同条件的变化总能量本来就不一样）。
    """
    if not at or not off:
        return float("nan")
    a, o = _mean(at), _mean(off)
    return float((a - o) / max(a + o, 1e-12))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N04 机制矩阵评测")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--suite", default="mechanism")
    parser.add_argument("--output", default="outputs/n04-eval")
    parser.add_argument("--baseline", default="candidate")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads and args.threads > 0:
        torch.set_num_threads(int(args.threads))

    cfg, train_cfg = config_from_json(args.config)
    if args.baseline == "candidate":
        encoder = N04Encoder(cfg)
        if args.checkpoint:
            load_checkpoint(args.checkpoint, encoder=encoder)
    else:
        encoder = build_baseline(args.baseline, cfg)
        #: N04 消融内部包着同一个 N04Encoder：给了 checkpoint 就必须装上，否则
        #: “候选 vs 消融”会变成“训练过的 vs 随机初始化的”，对照不成立。
        if args.checkpoint and args.baseline.startswith("n04_"):
            load_checkpoint(args.checkpoint, encoder=encoder.encoder)
    encoder.eval()
    scene = SceneConfig(
        height=cfg.measurement_size[0],
        width=cfg.measurement_size[1],
        input_mode=cfg.input_mode,
        cfa_pattern=cfg.cfa_pattern,
        camera_translation=(1.2, 1.2),
        camera_rotation=(0.0, 0.4),
        object_velocity=(0.8, 1.2),
        illumination=(0.7, 1.3),
        noise_sigma=0.01,
        seed=args.seed,
    )
    with torch.no_grad():
        report = run_mechanism_suite(
            encoder,
            base_scene=scene,
            length=args.length,
            batch=args.batch,
            seed=args.seed,
            pixel_stride=cfg.geometry.pixel_stride,
        )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": args.suite,
        "baseline": args.baseline,
        "model": cfg.model.as_dict(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "measurement_size": list(cfg.measurement_size),
        "pixel_stride": cfg.geometry.pixel_stride,
        "report": report.as_dict(),
    }
    (out_dir / "mechanism.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["report"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

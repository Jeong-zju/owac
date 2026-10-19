"""S1 测量评测：开发/最终拆分上的几何误差、方差校准、对应区分与无记忆基线诊断。

用法::

    python -m owac.evaluation.n04_full --config configs/n04/full/s1_measurement.json \\
        --checkpoint <ckpt> --split dev --output <report.json>

指标语义（与方案 §11.1 一致）：

* ``depth_*``：校正前（本阶段即测量本身）的逐像素深度误差，米与训练尺度归一化分报；
* ``coverage_*``：|残差| ≤ kσ 的经验覆盖率，σ 来自预测方差（独立开发集校准检查）；
* ``protocol_ratio``：与[标定协议](../../../../configs/n04/full/protocol_calibrated_sim.json)
  冻结的“该相机一个原图像素对应的米制门槛”之比；<1 表示测量误差已低于一个原图像素；
* ``carry_*``：无记忆基线诊断——缺失时沿用最近一次测量，报告随 age 的测量漂移。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..frameworks.n04.full_config import N04FullConfig, load_config
from ..training.n04_full import (
    SceneRef,
    Trainer,
    _paired_refs,
    build_scene_views,
)

__all__ = ["carry_diagnostics", "evaluate_main", "main", "protocol_comparison"]

DEFAULT_PROTOCOL = "configs/n04/full/protocol_calibrated_sim.json"


def protocol_comparison(protocol_path: str | Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """把每相机实测深度误差与冻结协议的一个原图像素门槛比较。"""
    path = Path(protocol_path)
    if not path.exists():
        return {"available": False, "reason": f"protocol file not found: {path}"}
    protocol = json.loads(path.read_text())
    camera_references = protocol.get("camera_references", {})
    out: dict[str, Any] = {"available": True, "protocol_sha256": _sha256(path), "groups": {}}
    for label, group in (metrics.get("groups") or {}).items():
        reference = camera_references.get(label)
        if reference is None or group.get("depth_mae_m") is None:
            continue
        one_pixel = float(reference["one_original_pixel_m"])
        out["groups"][label] = {
            "depth_mae_m": group["depth_mae_m"],
            "one_original_pixel_m": one_pixel,
            "protocol_ratio": group["depth_mae_m"] / one_pixel if one_pixel else None,
            "reference_depth_m": reference.get("reference_depth_m"),
        }
    return out


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def carry_diagnostics(
    config: N04FullConfig, trainer: Trainer, split: str, *, max_age: int = 3
) -> dict[str, Any]:
    """无记忆基线：缺失时保留最近一次测量，报告随 age 的潜在 / 深度漂移。"""
    trainer.model.eval()
    store = trainer.store
    latent_drift: dict[int, list[float]] = defaultdict(list)
    depth_drift: dict[int, list[float]] = defaultdict(list)
    sequences = 0
    with torch.no_grad():
        for ref in _paired_refs(config, split, store):
            sequences += 1
            n_frames = len(store.paired_record(ref.path)["frames"])
            for camera_id, camera_index in (
                (ref.camera_a, ref.camera_index_a),
                (ref.camera_b, ref.camera_index_b),
            ):
                measurements = []
                for index in range(n_frames):
                    single = SceneRef(
                        kind=ref.kind,
                        path=ref.path,
                        frame_index=index,
                        camera_a=camera_id,
                        camera_b=camera_id,
                        camera_index_a=camera_index,
                        camera_index_b=camera_index,
                        source_trajectory_id=ref.source_trajectory_id,
                        platform=ref.platform,
                        split=ref.split,
                        shape_hw=ref.shape_hw,
                        scale_name=ref.scale_name,
                    )
                    frame, _, _, _ = build_scene_views(single, store)
                    measurements.append(
                        trainer.model.measure([frame.move_to(device=trainer.device)])
                    )
                for index in range(len(measurements)):
                    fresh = measurements[index]
                    for age in range(1, max_age + 1):
                        if index - age < 0:
                            continue
                        carried = measurements[index - age]
                        reference_norm = float(fresh.latent_mean_map.abs().mean()) + 1e-8
                        latent_drift[age].append(
                            float((fresh.latent_mean_map - carried.latent_mean_map).abs().mean())
                            / reference_norm
                        )
                        depth_drift[age].append(
                            float((fresh.depth_mean - carried.depth_mean).abs().mean())
                        )
    trainer.model.train()
    return {
        "sequences": sequences,
        "latent_relative_drift": {
            str(age): float(np.mean(values))
            for age, values in sorted(latent_drift.items())
            if values
        },
        "depth_drift_m": {
            str(age): float(np.mean(values))
            for age, values in sorted(depth_drift.items())
            if values
        },
        "note": "无记忆基线诊断：数值随 age 增长说明沿用旧测量会偏离当前帧测量，供 S2 状态模块对照",
    }


def evaluate_main(
    config: N04FullConfig,
    checkpoint: str,
    *,
    split: str,
    output: str | Path,
    protocol: str | None = DEFAULT_PROTOCOL,
    with_carry: bool = True,
    batch_frames: int | None = None,
) -> dict[str, Any]:
    trainer = Trainer(config, seed=config.training.seeds[0], run_dir=Path(output).parent)
    payload = trainer.load_checkpoint(checkpoint)
    metrics = trainer.evaluate(split, batch_frames=batch_frames)
    report: dict[str, Any] = {
        "stage": config.stage,
        "config_sha256": config.sha256(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(Path(checkpoint)),
        "checkpoint_step": payload.get("step"),
        "split": split,
        "metrics": metrics,
    }
    if protocol:
        report["protocol"] = protocol_comparison(protocol, metrics)
    if with_carry:
        report["memoryless_carry"] = carry_diagnostics(config, trainer, split)
    _write(output, report)
    return report


def _write(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--no-carry", action="store_true")
    parser.add_argument("--batch-frames", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    report = evaluate_main(
        config,
        args.checkpoint,
        split=args.split,
        output=args.output,
        protocol=args.protocol,
        with_carry=not args.no_carry,
        batch_frames=args.batch_frames,
    )
    print(
        json.dumps(
            {
                "split": report["split"],
                "metrics": {
                    key: value for key, value in report["metrics"].items() if key not in ("groups",)
                },
                "protocol": report.get("protocol", {}).get("groups", {}),
                "memoryless_carry": report.get("memoryless_carry", {}),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())

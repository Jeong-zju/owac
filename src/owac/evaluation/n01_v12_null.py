"""Null calibration for the support novelty statistic.

The residual is ``pooled |history - reference| / noise_scale``, where the scale comes from
the reference frame's local Laplacian magnitude.  Measured on real calibration data the
median residual is 1.97 on LIBERO and 0.29 on RoboTwin -- 6.8x apart -- so one fixed
threshold such as ``large_residual_threshold = 1.0`` cannot mean the same thing on both.

Two explanations fit that spread and they imply opposite fixes:

* the estimator is miscalibrated (the noise scale is wrong by a platform-dependent
  factor), in which case the estimator must be fixed; or
* the estimator is behaving correctly and the spread reflects the reference frames'
  structure-to-noise ratio, which differs by platform, in which case the statistic is
  simply content-dependent and no rescaling makes one threshold transferable.

This probe separates them by injecting a KNOWN noise level into real frames and watching
how the statistic responds.  If the response tracks the injected level on both platforms,
the estimator is not the problem.

It runs on CPU, because the support mechanism uses no image features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

DEFAULT_SIGMAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)


def probe(
    values: torch.Tensor,
    cell_mask: torch.Tensor,
    support: Any,
    *,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    seed: int = 20261015,
) -> dict[str, Any]:
    """Residual response to a known injected noise level.

    ``values`` is one camera's window ``[B, T, C, H, W]``; only slots 0 and 1 are used, so
    the statistic sees exactly one innovation against one reference.
    """

    generator = torch.Generator().manual_seed(seed)
    reference = values[:, :1].clone()
    mask = cell_mask[:, :1]
    curve: dict[str, float | None] = {}
    for sigma in sigmas:
        noise = torch.randn(reference.shape, generator=generator, dtype=reference.dtype) * sigma
        other = (reference + noise).clamp(0.0, 1.0)
        pair = torch.cat([other, reference], dim=1)  # newest first
        paired_mask = torch.cat([mask, mask], dim=1)
        residual, valid = support.region_residual(pair, paired_mask)
        finite = residual[valid]
        curve[f"{sigma:g}"] = None if finite.numel() == 0 else round(float(finite.median()), 4)
    flat = reference.reshape(reference.shape[0], -1)
    curve["_frame_std"] = round(float(flat.std()), 5)
    curve["_frame_mean"] = round(float(flat.mean()), 5)
    return curve


def motion_response(
    values: torch.Tensor,
    mask: torch.Tensor,
    support: Any,
    *,
    shifts: tuple[int, ...] = (0, 2, 4, 8, 16),
    arrivals: int | None = None,
) -> dict[str, Any]:
    """Decision fractions against a KNOWN artificial displacement of the history.

    This is the falsification test for the explanation offered for the measured contrast
    between a bright moving episode (no usable history) and a near-black one (history
    largely retained).  If the criterion really is reacting to frame-to-frame difference,
    then sliding the history against the reference must raise the withdrawal rate, and it
    must do so on the dark frames too -- which is what makes it a test rather than a
    restatement of the contrast.
    """

    reference = values[:, :1]
    length = int(arrivals or min(values.shape[1], 32))

    def fractions(offset: int) -> dict[str, float] | None:
        if offset:
            history = torch.roll(reference, shifts=offset, dims=-1).repeat(1, length - 1, 1, 1, 1)
        else:
            history = reference.repeat(1, length - 1, 1, 1, 1)
        window = torch.cat([reference, history], dim=1)
        state = support.initial_state(
            window.shape[0], length, window.shape[-2], window.shape[-1], window.device
        )
        source_valid = torch.ones(window.shape[0], length, dtype=torch.bool, device=window.device)
        decision = None
        for t in range(2, length + 1):
            state, decision = support.observe(
                state, window[:, :t], mask[:, :t], source_valid[:, :t]
            )
        if decision is None:
            return None
        codes = decision.state.reshape(-1)
        total = max(int(codes.numel()), 1)
        # The detector consumes the noise-normalised region residual, so the residual and
        # odds scales are what decide between a deferral and a withdrawal.  Reporting them
        # turns "bright episodes are judged more harshly" into something checkable.
        residual = decision.region_residual.reshape(-1).float()
        odds = decision.change_logodds.reshape(-1).float()
        return {
            "accepted": round(float((codes == 1).sum()) / total, 4),
            "deferred": round(float((codes == 2).sum()) / total, 4),
            "retracted": round(float((codes == 3).sum()) / total, 4),
            "residual_median": round(float(residual.median()), 4),
            "logodds_median": round(float(odds.median()), 4),
            "logodds_p90": round(float(odds.quantile(0.9)), 4),
        }

    return {
        "arrivals": length,
        "by_shift": {str(shift): fractions(shift) for shift in shifts},
        "reading": (
            "under the difference-based explanation the withdrawal rate must rise with the "
            "injected displacement, including on near-black frames"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="null calibration of the residual statistic")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--region", default="calibration")
    parser.add_argument(
        "--protocol",
        default=None,
        help="explicit protocol file; defaults to the repository protocol for the platform",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sigmas", default=",".join(str(s) for s in DEFAULT_SIGMAS))
    args = parser.parse_args(argv)

    import sys

    sys.path.insert(0, str(Path(args.repo_root) / "src"))
    from owac.evaluation.n01_v12 import _datasets, _load_config
    from owac.frameworks.n01_full.support import SpatialSupport

    config = _load_config(args.config)
    dataset = _datasets(args, config, args.region)
    support = SpatialSupport(config.support)
    sigmas = tuple(float(value) for value in args.sigmas.split(",") if value)

    per_camera: dict[str, list[dict[str, Any]]] = {}
    for index in range(min(int(args.samples), len(dataset))):
        sample = dataset[index]
        for camera, window in sorted(sample.windows.items()):
            if args.camera and camera != args.camera:
                continue
            values = torch.as_tensor(window.values).float()
            valid = torch.as_tensor(window.valid_mask)
            if values.ndim == 4:
                values = values[None]
                valid = valid[None]
            if values.shape[1] < 2:
                continue
            cell = valid.all(2, keepdim=True)
            per_camera.setdefault(camera, []).append(
                probe(values[:, :2], cell[:, :2], support, sigmas=sigmas)
            )

    summary: dict[str, Any] = {
        "stage": "P3",
        "command": "support-null",
        "platform": config.platform,
        "region": args.region,
        "sigmas": list(sigmas),
        "reading": (
            "If the median residual rises with the injected level on both platforms, the "
            "estimator tracks noise and the cross-platform spread reflects scene structure "
            "rather than a miscalibrated scale."
        ),
        "cameras": {},
    }
    for camera, curves in per_camera.items():
        medians = {}
        for key in [f"{sigma:g}" for sigma in sigmas]:
            values_seen = [curve[key] for curve in curves if curve.get(key) is not None]
            medians[key] = (
                None if not values_seen else round(float(torch.tensor(values_seen).median()), 4)
            )
        summary["cameras"][camera] = {
            "windows": len(curves),
            "residual_median_by_injected_sigma": medians,
            "frame_std": round(float(torch.tensor([c["_frame_std"] for c in curves]).median()), 5),
        }
    if args.output:
        Path(args.output).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

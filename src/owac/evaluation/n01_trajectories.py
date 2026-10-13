"""N01 support-domain trajectory explorer.

Produces the three trajectories the implementation prompt asks for, each with its
measurement, its support domain, its localisation view and its state
diagnostics:

1. **static weak edge accumulation** -- a still low-contrast slit is integrated
   over a longer time depth;
2. **fast edge shortens the history** -- the same slit moving quickly must not be
   integrated over the same window, so the chosen time depth and the support age
   drop;
3. **two targets separate and the support domain splits** -- a second target
   appears next to the first and the support graph loses the crossing edges.

The module writes dependency-free PNG panels (no matplotlib/PIL; a small zlib
PNG encoder is included) plus a JSON diagnostics summary, so a reader can look at
the mechanism directly while the numbers that decide conclusions come from the
evaluation suite.

Visualisation explains the mechanism; it never replaces a metric.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..core.encoder_contract import ComputeBudget, MeasurementFrame
from ..frameworks.n01.config import N01Config
from ..frameworks.n01.model import N01Encoder
from ..frameworks.n01.synthetic import SyntheticWeakEdgeTask

__all__ = ["TrajectoryResult", "run_trajectory", "write_trajectory_artifacts", "TRAJECTORIES"]

TRAJECTORIES: tuple[dict[str, Any], ...] = (
    {
        "name": "static_weak_edge",
        "title": "static weak edge: accumulate while the scene is still",
        "snr": "low",
        "motion": "still",
        "tolerance": "wide",
        "seed": 11,
        "length": 8,
        "image_size": 32,
    },
    {
        "name": "fast_edge_shortens_history",
        "title": "fast edge: the chosen history must shorten",
        "snr": "low",
        "motion": "fast",
        "tolerance": "narrow",
        "seed": 12,
        "length": 8,
        "image_size": 32,
    },
    {
        "name": "two_targets_split",
        "title": "two targets separate: the support domain splits",
        "snr": "medium",
        "motion": "crossing",
        "tolerance": "narrow",
        "seed": 13,
        "length": 8,
        "image_size": 32,
    },
)


@dataclass
class TrajectoryResult:
    """Per-frame measurements, support state and diagnostics of one trajectory."""

    name: str
    title: str
    config: dict[str, Any]
    images: dict[str, torch.Tensor]  # panels, [T,H,W]
    diagnostics: list[dict[str, Any]]
    scene: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        def column(key: str) -> list[float]:
            return [float(item[key]) for item in self.diagnostics]

        return {
            "name": self.name,
            "title": self.title,
            "config": self.config,
            "scene": self.scene,
            "frames": len(self.diagnostics),
            "mean_measurement_age_frames": _mean(column("mean_age")),
            "mean_support_width": _mean(column("mean_support_count")),
            "selected_radius_mean": _mean(column("selected_radius_mean")),
            "selected_history_frames_mean": _mean(column("selected_depth_mean")),
            "merge_count_total": float(self.diagnostics[-1]["merge_count"]),
            "split_count_total": float(self.diagnostics[-1]["split_count"]),
            "edge_on_fraction_mean": _mean(column("edge_mean")),
            "mixture_participation_mean": _mean(column("mixture_participation")),
            "per_frame": self.diagnostics,
        }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def build_encoder(config: N01Config, seed: int = 0) -> N01Encoder:
    """Deterministically initialised encoder (visualisation uses one fixed draw)."""
    torch.manual_seed(seed)
    return N01Encoder(config)


def run_trajectory(
    spec: dict[str, Any],
    *,
    config: N01Config | None = None,
    device: str = "cpu",
    encoder: N01Encoder | None = None,
) -> TrajectoryResult:
    """Step one synthetic trajectory through the real encoder and record panels."""
    config = config or N01Config.tiny(
        width=16,
        num_blocks=1,
        max_history=8,
        spatial_radii=(0, 1, 2),
        time_depths=(1, 4, 8),
        neighbourhood_radius=2,
        readout_tokens=64,
        readout_width=64,
        alignment="candidate",
        fractional_alignment=True,
    )
    task = SyntheticWeakEdgeTask(
        config,
        image_size=int(spec["image_size"]),
        snr=str(spec["snr"]),
        motion=str(spec["motion"]),
        tolerance=str(spec["tolerance"]),
        seed=int(spec["seed"]),
    )
    batch = task.sample(1, int(spec["length"]), seed=int(spec["seed"]))
    encoder = encoder or build_encoder(config, seed=int(spec["seed"]))
    encoder.to(device)
    encoder.eval()
    from ..frameworks.n01.model import calibrate_statistics

    calibration_source = "retained_encoder_training_scales"
    if not int(encoder.calibration_count):
        # Same declared training calibration for every evaluation trajectory.
        # It is independent of all three evaluation seeds and never optimises weights.
        calibration = SyntheticWeakEdgeTask(
            config, image_size=int(spec["image_size"]), snr="low", motion="slow", seed=500000
        ).sample(1, config.max_history)
        calibrate_statistics(encoder, calibration, device=device)
        calibration_source = "independent_training_low_slow_seed_500000"
    diagnostics: list[dict[str, Any]] = []
    panels = {
        "measurement": [],
        "support_weight": [],
        "integrated_mu": [],
        "connectivity": [],
    }
    state = encoder.init_state(
        1, {"camera_ids": ["camera"], "shape": tuple(batch.values.shape[-2:])}, device
    )
    with torch.inference_mode():
        for index in range(int(spec["length"])):
            frame = MeasurementFrame(
                values=batch.values[:, index].to(device),
                valid_mask=batch.valid[:, index].to(device),
                camera_id="camera",
                timestamp=batch.timestamps[:, index].to(device),
                time_unit="frame",
                new_measurement=batch.new_measurement[:, index].to(device),
                sample_xy=batch.meta.get("sample_xy"),
                metadata={"variance": batch.variance[:, index].to(device)},
            )
            output = encoder.step(frame, None, state, ComputeBudget(max_updates=1, max_history=8))
            state = output.state
            statistics = encoder.last_statistics
            measurement = batch.values[0, index].mean(dim=0)
            panels["measurement"].append(measurement.cpu())
            weights = encoder.last_selection_weights
            support = (weights * statistics.unique_samples[encoder.last_selection_slice]).sum(0)[
                0, 0
            ]
            panels["support_weight"].append(support.cpu())
            panels["integrated_mu"].append(
                (weights * statistics.mu[encoder.last_selection_slice]).sum(0)[0, 0].cpu()
            )
            # Radius-zero connectivity is always identity; display the real open edges.
            panels["connectivity"].append(
                state.plane_support("camera").open_edges[0].float().mean((0, 1)).cpu()
            )
            record = {key: value for key, value in output.diagnostics.items()}
            record.update(
                {
                    "frame": index,
                    "mean_support_count": output.diagnostics["mean_support_samples"],
                    "mean_variance_proxy": float(statistics.variance_proxy.mean()),
                    "measurement_mean": float(measurement.mean()),
                    "target_row": float(batch.target_xy[0, index, 0]),
                    "target_col": float(batch.target_xy[0, index, 1]),
                    "visible": bool(batch.target_visible[0, index]),
                }
            )
            diagnostics.append(_jsonable(record))
    return TrajectoryResult(
        name=str(spec["name"]),
        title=str(spec["title"]),
        config={
            "snr": spec["snr"],
            "motion": spec["motion"],
            "tolerance": spec["tolerance"],
            "seed": spec["seed"],
            "length": spec["length"],
            "image_size": spec["image_size"],
            "config_fingerprint": _fingerprint(config),
            "calibration_source": calibration_source,
            "calibration_count": int(encoder.calibration_count),
            "normalization_scales": encoder.normalization_scales.detach().cpu().tolist(),
        },
        images={key: torch.stack(value) for key, value in panels.items()},
        diagnostics=diagnostics,
        scene=dict(batch.meta.get("spec", {})),
    )


def _fingerprint(config: N01Config) -> str:
    from ..frameworks.n01.config import config_fingerprint

    return config_fingerprint(config)


def _jsonable(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


# ---------------------------------------------------------------------------
# Dependency-free PNG output
# ---------------------------------------------------------------------------
def normalize_panel(panel: torch.Tensor) -> torch.Tensor:
    """Map a panel to [0, 1] with a robust (2nd..98th percentile) stretch."""
    flat = panel.reshape(-1).float()
    if flat.numel() == 0:
        return panel
    low = torch.quantile(flat, 0.02)
    high = torch.quantile(flat, 0.98)
    if float(high - low) < 1e-12:
        return torch.zeros_like(panel)
    return ((panel - low) / (high - low)).clamp(0.0, 1.0)


def to_rgb(panel: torch.Tensor) -> torch.Tensor:
    """[H,W] in [0,1] -> [H,W,3] uint8, quantised deterministically."""
    values = (normalize_panel(panel) * 255.0).round().to(torch.uint8)
    return values.unsqueeze(-1).expand(-1, -1, 3).contiguous()


def write_png(path: Path, rgb: torch.Tensor, scale: int = 6) -> None:
    """Write an [H,W,3] uint8 tensor as a PNG, upscaled by ``scale`` (nearest)."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("expected an [H, W, 3] uint8 image")
    image = rgb.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
    height, width = int(image.shape[0]), int(image.shape[1])
    raw = bytearray()
    pixels = image.to(torch.uint8).numpy()
    for row in range(height):
        raw.append(0)  # filter type 0 (None)
        raw.extend(pixels[row].tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        stream.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        stream.write(_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        stream.write(_chunk(b"IEND", b""))


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def panel_view(key: str, frame: torch.Tensor) -> torch.Tensor:
    """Per-panel display transform.

    ``support_count`` spans four orders of magnitude, so it is shown on a log
    scale; the other panels use a robust percentile stretch.  The transforms are
    display-only and are recorded in the JSON summary, never in a metric.
    """
    if key == "support_weight":
        return torch.log1p(frame.clamp_min(0.0))
    return frame


def contact_sheet(
    result: TrajectoryResult,
    columns: tuple[str, ...] = (
        "measurement",
        "support_weight",
        "integrated_mu",
        "connectivity",
    ),
) -> torch.Tensor:
    """One image: rows = frames, columns = the requested panels."""
    rows = []
    for index in range(result.images[columns[0]].shape[0]):
        rows.append(
            torch.cat(
                [to_rgb(panel_view(key, result.images[key][index])).float() for key in columns],
                dim=1,
            )
        )
    return torch.cat(rows, dim=0).to(torch.uint8)


def write_trajectory_artifacts(result: TrajectoryResult, output: Path) -> dict[str, Any]:
    """Write the PNG contact sheet and the JSON diagnostics of one trajectory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sheet = contact_sheet(result)
    png_path = output / f"{result.name}.png"
    write_png(png_path, sheet)
    json_path = output / f"{result.name}.json"
    summary = result.summary()
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "name": result.name,
        "image": str(png_path),
        "panels": ["measurement", "log1p(support_count)", "integrated_mu", "connectivity"],
        "diagnostics": str(json_path),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/n01-trajectories", type=Path)
    parser.add_argument("--seed", default=0, type=int, help="encoder initialisation seed")
    args = parser.parse_args(argv)
    manifest = []
    for spec in TRAJECTORIES:
        encoder = build_encoder(_default_config(), seed=args.seed)
        result = run_trajectory(spec, encoder=encoder)
        manifest.append(write_trajectory_artifacts(result, args.output))
    index = Path(args.output) / "index.json"
    index.write_text(json.dumps({"trajectories": manifest}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index": str(index), "count": len(manifest)}))
    return 0


def _default_config() -> N01Config:
    return N01Config.tiny(
        width=16,
        num_blocks=1,
        max_history=8,
        spatial_radii=(0, 1, 2),
        time_depths=(1, 4, 8),
        neighbourhood_radius=2,
        readout_tokens=64,
        readout_width=64,
        alignment="candidate",
        fractional_alignment=True,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

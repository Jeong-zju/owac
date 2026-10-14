"""Real-data ledger check: observe raw RawVLA windows into the new ledger.

P1 owes one real numeric-stream read with a unique-source count
(``P1_ANALYTIC_REFERENCE.md`` section 5).  This entry point builds the frozen
four-region loader from the protocol, feeds real NPZ windows through
:class:`~owac.frameworks.n01_full.ledger.MeasurementLedger` one arrival event at
a time, and reports the identity accounting: registered exposures per camera,
samples and validity counts, and the proof that re-sending one packet adds no
measurement.

It runs in either the CPU encoder environment or the GPU one; it never imports
the policy backend, so the check is cheap and cannot be confused with the
``mode=contract`` policy preflight.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from owac.core.encoder_contract import MeasurementFrame

DEFAULT_PROTOCOL = "configs/encoders/n01_full/protocol_v1.json"


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="owac-n01-full-ledger-check", description=__doc__)
    top.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    top.add_argument("--repo-root", type=Path, default=Path("."))
    top.add_argument(
        "--platform",
        action="append",
        choices=("libero", "robotwin2"),
        help="platform(s) to check; default both",
    )
    top.add_argument("--batches", type=int, default=2, help="real batches per platform")
    top.add_argument("--region", default="train", help="four-region split to read")
    top.add_argument("--crop", type=int, default=32, help="top-left crop; 0 keeps the window")
    top.add_argument("--output", type=Path, default=Path("outputs/n01-full-ledger-check"))
    return top


def _crop_batch(batch: dict[str, Any], crop: int) -> None:
    if crop <= 0:
        return
    for window in batch["windows"].values():
        for key in ("values", "valid_mask", "bad_mask", "saturated_mask", "variance"):
            value = getattr(window, key, None)
            if value is not None:
                setattr(window, key, value[..., :crop, :crop])


def _sample_grid(values: Any, offsets: Any) -> Any:
    """Expand the loader's per-channel sampling offsets into per-sample coordinates.

    ``RawVLASequenceDataset`` declares ``sample_xy`` as one ``(row, col)`` offset
    per *channel*: zeros for the co-located three-channel pseudo-RAW and the real
    CFA plane phase for CFA plane stacks.  The ledger stores one ``(x, y)``
    coordinate per sample, so the expansion is explicit here: a sample's
    coordinate is its pixel index plus its channel offset.  An unexpected offset
    shape is not reinterpreted silently; it is reported as undeclared.
    """
    if offsets is None:
        return None
    tensor = offsets if hasattr(offsets, "shape") else None
    if tensor is None or tensor.ndim != 2 or tensor.shape[1] != 2:
        return None
    import torch

    channels, height, width = values.shape[1], values.shape[2], values.shape[3]
    if tensor.shape[0] != channels:
        return None
    rows = (
        torch.arange(height, dtype=torch.float64).view(1, height, 1).expand(channels, height, width)
    )
    columns = (
        torch.arange(width, dtype=torch.float64).view(1, 1, width).expand(channels, height, width)
    )
    grid = torch.stack(
        [
            columns + tensor[:, 1].to(torch.float64).view(-1, 1, 1),
            rows + tensor[:, 0].to(torch.float64).view(-1, 1, 1),
        ],
        dim=-1,
    )
    return grid.unsqueeze(0).expand(values.shape[0], -1, -1, -1, -1)


def iter_window_frames(window: Any, episode_keys: Sequence[str], camera: str) -> Any:
    """Yield ``(time_step, MeasurementFrame)`` for one collated loader window.

    The loader window is ``[B, T, C, H, W]`` with a leading *time* axis, so the
    arrival events are indexed on axis 1; indexing axis 0 would read one batch
    member once and silently collapse the window to a single exposure.
    """
    values = window.values
    for step in range(values.shape[1]):
        frame = MeasurementFrame(
            values=values[:, step],
            valid_mask=None if window.valid_mask is None else window.valid_mask[:, step],
            bad_mask=None if window.bad_mask is None else window.bad_mask[:, step],
            saturated_mask=(
                None if window.saturated_mask is None else window.saturated_mask[:, step]
            ),
            frame_ids=window.frame_ids[:, step],
            episode_ids=list(episode_keys),
            camera_id=camera,
            timestamp=None if window.timestamp is None else window.timestamp[:, step],
            time_unit=window.time_unit,
            new_measurement=(
                None if window.new_measurement is None else window.new_measurement[:, step]
            ),
            sample_xy=_sample_grid(values[:, step], window.sample_xy),
        )
        yield step, frame


def ledger_source_check(dataset: Any, platform: str, *, batches: int, crop: int) -> dict[str, Any]:
    """Observe real loader windows into the new ledger and count unique sources.

    This is the P1 real-data check of the identity contract: one exposure per
    ``(episode, camera, frame_id)``, repeats do not add measurements, and the
    ``SourceKey`` values correspond to the actual camera/frame ids of the batch.
    """
    from owac.frameworks.n01_full.ledger import MeasurementLedger
    from owac.raw.rawvla_sequence import collate_sequence_samples

    ledger = MeasurementLedger(f"rawvla_bench:{platform}", value_units="pseudo_raw")
    rows: list[dict[str, Any]] = []
    window_sample_xy: Any = None
    observed_windows: list[tuple[str, str, MeasurementFrame]] = []
    for index in range(max(batches, 1)):
        sample = dataset[index]
        batch = collate_sequence_samples([sample])
        _crop_batch(batch, crop)
        for camera, window in batch["windows"].items():
            window_sample_xy = window.sample_xy
            episode_keys = [meta["episode_key"] for meta in batch["meta"]]
            for _step, frame in iter_window_frames(window, episode_keys, camera):
                outcome = ledger.observe(frame)
                if not observed_windows:
                    observed_windows.append((camera, str(batch["meta"][0]["episode_key"]), frame))
                rows.append(
                    {
                        "camera": camera,
                        "registered": len(outcome.registered_frames),
                        "duplicates": len(outcome.duplicate_frames),
                        "skipped": len(outcome.skipped_frames),
                    }
                )
    repeat = None
    if observed_windows:
        camera, _, frame = observed_windows[0]
        outcome = ledger.observe(frame)
        repeat = {
            "camera": camera,
            "new_samples": outcome.new_samples,
            "duplicates": len(outcome.duplicate_frames),
        }
    stats = ledger.ledger_stats()
    return {
        "batches": max(batches, 1),
        "frames_observed": len(rows),
        "registered_frames": stats["frames"],
        "unique_frames_per_camera": {
            camera: ledger.unique_frame_count(camera_id=camera)
            for camera in sorted({row["camera"] for row in rows})
        },
        "samples": stats["samples"],
        "validity_counts": stats["validity_counts"],
        "duplicate_repeat_new_samples": None if repeat is None else repeat["new_samples"],
        "duplicate_repeat_frames": None if repeat is None else repeat["duplicates"],
        "arrivals": stats["arrivals"],
        "skipped_arrivals": stats["skipped_arrivals"],
        "sample_xy_declared": bool(observed_windows) and window_sample_xy is not None,
        "sample_xy_shape": None if window_sample_xy is None else list(window_sample_xy.shape),
        "sample_xy_rule": "per-channel (row,col) offset expanded to (x,y) = pixel + offset",
        "first_source_example": (
            None if not observed_windows else list(observed_windows[0][2].frame_ids.tolist())
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from owac.raw.n01_full_manifest import dataset_from_protocol

    protocol = json.loads(args.protocol.read_text())
    platforms = args.platform or ["libero", "robotwin2"]
    report: dict[str, Any] = {
        "kind": "n01-full-ledger-check",
        "protocol": str(args.protocol),
        "region": args.region,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "platforms": {},
    }
    for name in platforms:
        dataset = dataset_from_protocol(protocol, name, args.region, repo_root=args.repo_root)
        report["platforms"][name] = {
            "manifest_path": str(dataset.paths[0]),
            "samples": len(dataset),
            "statistics": str(dataset.stats_path),
            "check": ledger_source_check(dataset, name, batches=args.batches, crop=args.crop),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = run(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ledger-check.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    summary = {
        name: entry["check"]["unique_frames_per_camera"]
        for name, entry in report["platforms"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"receipt: {args.output / 'ledger-check.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

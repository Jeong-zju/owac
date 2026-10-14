"""Independently re-read a bounded D1 extension, not its PASS fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(record_path: Path) -> dict:
    record = json.loads(record_path.read_text())
    root = record_path.parent
    arrays_path = root / Path(record["npz"]).name
    trace_path = root / record["source_trace"]
    trace = json.loads(trace_path.read_text())
    state_path = root / trace["npz"]
    checks = dict(
        measurement_hash=hashlib.sha256(arrays_path.read_bytes()).hexdigest() == record["sha256"],
        trace_record_hash=hashlib.sha256(trace_path.read_bytes()).hexdigest()
        == record["trace_sha256"],
        trace_array_hash=hashlib.sha256(state_path.read_bytes()).hexdigest() == trace["npz_sha256"],
        extension_only=record["mode"] == "extension_set"
        and not record["source_correspondence_verified"],
    )
    with (
        np.load(arrays_path, allow_pickle=False) as data,
        np.load(state_path, allow_pickle=False) as states,
    ):
        checks["finite_arrays"] = all(np.isfinite(data[key]).all() for key in data.files)
        checks["finite_trace"] = all(np.isfinite(states[key]).all() for key in states.files)
        continuity = float(np.max(np.abs(states["pre_state"][1:] - states["post_state"][:-1])))
        checks["continuous_execution"] = continuity <= 1e-12
        checks["frame_trace_alignment"] = (
            len(states["step_index"]) == len(data["online_sim_time_s"]) == record["frames"]
        )
        checks["physical_time"] = bool(np.all(np.diff(data["online_sim_time_s"]) > 0))
        raw = data["online_rawvla_rgb3"]
        depth = data["gt_depth_m"]
        seg = data["gt_segmentation"]
        checks["aligned_pixel_grids"] = raw.shape[:-1] == depth.shape == seg.shape[:-1]
        checks["positive_depth"] = bool(np.all(depth > 0))
        pose = data["gt_T_W_from_C"]
        rotations = pose[..., :3, :3]
        orthogonal_error = float(
            np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3)))
        )
        checks["rigid_camera_pose"] = orthogonal_error < 1e-6 and bool(
            np.all(np.linalg.det(rotations) > 0)
        )
        matches = data["audit_ray_matching_geom"]
        errors = data["audit_ray_error_m"]
        bounds = data["audit_ray_bound_m"]
        checks["independent_depth_rays"] = bool(
            np.all(matches.mean(axis=1) >= 0.8) and np.all(errors <= bounds)
        )
        metrics = dict(
            frames=int(raw.shape[0]),
            camera_frames=int(raw.shape[0] * raw.shape[1]),
            continuity_max_abs=continuity,
            rotation_orthogonality_error=orthogonal_error,
            ray_samples=int(matches.size),
            matched_rays=int(matches.sum()),
            max_depth_error_m=float(errors.max()),
            max_error_over_bound=float((errors / bounds).max()),
        )
    return dict(
        record=str(record_path),
        checks={k: bool(v) for k, v in checks.items()},
        metrics=metrics,
        passed=bool(all(checks.values())),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [validate(path) for path in sorted(args.run.glob("capture-*/*.measurements.json"))]
    result = dict(records=records, passed=bool(records) and all(r["passed"] for r in records))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

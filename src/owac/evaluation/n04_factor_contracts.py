"""Independent array-level audits of physically rendered S0 factor data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from owac.raw.n04_live_collection import save_json, sha256
from owac.raw.n04_rendered_factors import FACTORS


def reference_depth(pose, k, positions, uv):
    """Independent NumPy ray/AABB intersections; no simulator or report metrics."""
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(k).T @ pose[:3, :3].T
    origin = pose[:3, 3]
    floor = -origin[2] / rays[:, 2]
    sizes = np.array([[0.18, 0.18, 0.18], [0.16, 0.16, 0.16], [0.65, 0.65, 0.02], [10, 10, 0.01]])
    direction = np.where(abs(rays) < 1e-14, 1e-14, rays)
    a = (positions[1:5, None, :] - sizes[:, None, :] - origin) / direction
    b = (positions[1:5, None, :] + sizes[:, None, :] - origin) / direction
    enter = np.minimum(a, b).max(axis=-1)
    leave = np.maximum(a, b).min(axis=-1)
    hits = np.where((leave >= enter) & (enter > 0), enter, np.inf)
    all_hits = np.concatenate((floor[None, :], hits), axis=0)
    return all_hits.min(axis=0), all_hits.argmin(axis=0)


def independent_geometry(data):
    uv = np.array([[u, v] for v in (16, 32, 64, 96, 112) for u in (16, 32, 64, 96, 112)])
    maximum = 0.0
    for frame in range(len(data["gt_depth_m"])):
        for view in range(2):
            reference, geom = reference_depth(
                data["gt_T_W_from_C"][frame, view],
                data["online_K"][frame, view],
                data["gt_body_position_W"][frame],
                uv,
            )
            gpu = data["gt_depth_m"][frame, view, uv[:, 1], uv[:, 0]]
            matched = data["audit_ray_matching_geom"][frame, view * 25 : (view + 1) * 25]
            labels = data["gt_segmentation"][frame, view, uv[:, 1], uv[:, 0], 0]
            if np.any(geom[matched] != labels[matched]):
                return False, float("inf")
            bounds = data["audit_ray_bound_m"][frame, view * 25 : (view + 1) * 25]
            error = abs(reference - gpu)
            maximum = max(maximum, float(np.max(error[matched] / bounds[matched], initial=0)))
            if np.any(error[matched] > bounds[matched]):
                return False, maximum
    return True, maximum


def validate(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for entry in manifest["sequences"]:
        path = root / entry["file"]
        factor = entry["factor"]
        with np.load(path, allow_pickle=False) as data:
            raw = data["online_rawvla_rgb3"]
            depth = data["gt_depth_m"]
            seg = data["gt_segmentation"]
            body = data["gt_body_position_W"]
            camera = data["gt_T_W_from_C"]
            k = data["online_K"]
            matches = data["audit_ray_matching_geom"]
            errors = data["audit_ray_error_m"]
            bounds = data["audit_ray_bound_m"]
            reference_passed, reference_ratio = independent_geometry(data)
            target = int(data["gt_target_geom_id"][0])
            visible = (seg[..., 0] == target).sum(axis=(2, 3))
            targetmotion = float(np.max(abs(body[:, 1] - body[0, 1])))
            cameramotion = float(np.max(abs(camera - camera[0])))
            checks = dict(
                independent_numpy_geometry=reference_passed,
                hash=sha256(path) == entry["sha256"],
                finite=all(np.isfinite(data[key]).all() for key in data.files),
                pixel_alignment=raw.shape[:-1] == depth.shape == seg.shape[:-1],
                ray_depth=bool(np.all(matches.mean(axis=1) >= 0.8) and np.all(errors <= bounds)),
                physics_time=bool(np.allclose(np.diff(data["gt_time_s"]), 0.05)),
            )
            if factor == "static":
                checks["intervention"] = np.array_equal(raw, np.broadcast_to(raw[0], raw.shape))
            elif factor == "camera":
                checks["intervention"] = cameramotion > 0.1 and targetmotion == 0
            elif factor == "object":
                checks["intervention"] = targetmotion > 0.1 and cameramotion == 0
            elif factor in ("both", "both_gain_noise"):
                checks["intervention"] = targetmotion > 0.1 and cameramotion > 0.1
            elif factor == "stop_reverse":
                velocity = np.diff(body[:, 1, 0])
                checks["intervention"] = bool(
                    np.any(velocity > 0.01)
                    and np.any(velocity < -0.01)
                    and np.count_nonzero(abs(velocity) < 1e-12) >= 3
                )
            elif factor in ("occlusion_recovery", "occlusion_reverse", "birth_disappearance"):
                checks["intervention"] = bool(
                    np.all(visible[:4] > 0)
                    and np.all(visible[4:8] == 0)
                    and np.all(visible[8:] > 0)
                )
            elif factor == "gain":
                corrected = raw / data["gt_gain"][:, None, None, None, None]
                checks["intervention"] = (
                    float(np.max(abs(corrected - corrected[0]))) < 1e-7 and targetmotion == 0
                )
            elif factor == "noise":
                checks["intervention"] = (
                    float(np.std(raw[1:] - raw[:-1])) > 0.001 and targetmotion == 0
                )
            elif factor == "time_shift":
                checks["intervention"] = np.array_equal(
                    data["input_frame_index"], np.maximum(0, np.arange(12) - 2)
                )
            elif factor == "wrong_association":
                checks["intervention"] = np.array_equal(data["input_association"], [1, 0])
            elif factor == "calibration_offset":
                checks["intervention"] = np.all(data["input_K"][..., 0, 2] - k[..., 0, 2] == 2)
            elif factor == "shadow_highlight":
                checks["intervention"] = (
                    np.max(abs(raw[-1] - raw[0])) > 0.001
                    and targetmotion == 0
                    and cameramotion == 0
                )
            elif factor == "moving_plane":
                # A homogeneous plane can move tangentially without giving a reliable
                # motion signal. Quantify actual residual, don't assert exact equality.
                checks["intervention"] = bool(
                    np.max(abs(body[:, 4] - body[0, 4])) > 0.1 and np.unique(seg[..., 0]).size == 1
                )
            rows.append(
                dict(
                    **entry,
                    checks={k: bool(v) for k, v in checks.items()},
                    passed=bool(all(checks.values())),
                    raw_change_max=float(np.max(abs(raw[-1] - raw[0]))),
                    target_visible_pixels=visible.tolist(),
                    independent_numpy_ray_ratio=reference_ratio,
                    ray_match_min=float(matches.mean(axis=1).min()),
                    ray_error_ratio_max=float((errors / bounds).max()),
                    frames=int(raw.shape[0]),
                    camera_frames=int(raw.shape[0] * raw.shape[1]),
                )
            )
    groups = {
        s: {e["source_trajectory_id"] for e in rows if e["split"] == s}
        for s in ("train", "dev", "final")
    }
    disjoint = all(
        not groups[a] & groups[b]
        for a, b in [("train", "dev"), ("train", "final"), ("dev", "final")]
    )
    coverage = {f: sum(e["factor"] == f for e in rows) for f in FACTORS}
    return dict(
        schema="n04-factor-audit-v1",
        source_disjoint=disjoint,
        full_split_matrix={s: len(g) for s, g in groups.items()}
        == {"train": 8, "dev": 1, "final": 1},
        source_split_counts={s: len(g) for s, g in groups.items()},
        coverage=coverage,
        records=rows,
        frames=sum(r["frames"] for r in rows),
        camera_frames=sum(r["camera_frames"] for r in rows),
        passed=bool(rows)
        and disjoint
        and {s: len(g) for s, g in groups.items()} == {"train": 8, "dev": 1, "final": 1}
        and all(coverage[f] == (2 if f == "both_gain_noise" else 10) for f in FACTORS)
        and all(r["passed"] for r in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

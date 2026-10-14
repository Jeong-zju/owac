"""Estimate external camera candidates from source RGB/robot-pose correspondences."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_dataset_export import load_source_record
from owac.runtime.polaris_replay import forward, utc


def moving_cyan_points(images, indices):
    from PIL import Image
    from scipy.ndimage import label

    decoded = [np.asarray(Image.open(io.BytesIO(images[i]))).astype(np.int16) for i in indices]
    tracks = []
    for j, image in enumerate(decoded):
        red, green, blue = np.moveaxis(image, -1, 0)
        mask = (
            (green > 65)
            & (blue > 85)
            & (red < 0.7 * np.minimum(green, blue))
            & (blue < 1.65 * green)
            & (green < 1.45 * blue)
        )
        change = np.zeros(mask.shape, dtype=bool)
        for neighbor in [j - 1, j + 1]:
            if 0 <= neighbor < len(decoded):
                change |= np.max(np.abs(image - decoded[neighbor]), axis=2) > 24
        components, count = label(mask)
        candidates = []
        for identity in range(1, count + 1):
            component = components == identity
            area = int(component.sum())
            if 8 <= area <= 450 and np.count_nonzero(change & component) / area >= 0.15:
                y, x = np.nonzero(component)
                if np.ptp(x) < 70 and np.ptp(y) < 50:
                    candidates.append(
                        {
                            "frame": int(indices[j]),
                            "pixel_xy": [float(x.mean()), float(y.mean())],
                            "area": area,
                        }
                    )
        tracks.extend(candidates)
    return tracks


def estimate(poses, tracks):
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    if len({t["frame"] for t in tracks}) < 24:
        raise ValueError("Fewer than 24 usable source RGB/robot correspondences")
    frames = np.asarray([t["frame"] for t in tracks])
    transforms = poses[frames]
    pixels = np.asarray([t["pixel_xy"] for t in tracks])
    rotations, translations = transforms[:, :3, :3], transforms[:, :3, 3]
    mount = np.asarray([-0.06, 0, 0])
    points = translations + rotations @ mount
    if np.linalg.svd(points - points.mean(0), compute_uv=False)[-1] < 0.008:
        raise ValueError("Source trajectory lacks enough three-dimensional camera constraints")
    unique_frames = np.unique(frames)
    training = np.isin(frames, unique_frames[np.arange(len(unique_frames)) % 3 != 0])
    held_out = ~training
    groups = [np.flatnonzero(frames == frame) for frame in unique_frames]
    cv2.setRNGSeed(42)
    best = None
    for focal in [100, 160, 240, 320, 450]:
        intrinsic = np.asarray([[focal, 0, 160], [0, focal, 90], [0, 0, 1.0]])
        ok, rotation, translation, inliers = cv2.solvePnPRansac(
            points[training],
            pixels[training],
            intrinsic,
            np.zeros(5),
            iterationsCount=500,
            reprojectionError=8,
            confidence=0.999,
        )
        if not ok or inliers is None or len(inliers) < 12:
            continue
        initial = np.r_[rotation.ravel(), translation.ravel(), np.log(focal), mount]

        def project(parameters):
            q = translations + rotations @ parameters[7:10]
            f = np.exp(parameters[6])
            matrix = np.asarray([[f, 0, 160], [0, f, 90], [0, 0, 1.0]])
            result, _ = cv2.projectPoints(q, parameters[:3], parameters[3:6], matrix, None)
            return result[:, 0]

        def residual(parameters):
            delta = project(parameters) - pixels
            # Competing cyan components remain candidates rather than fixed false matches.
            selected = [g[np.argmin(np.linalg.norm(delta[g], axis=1))] for g in groups]
            selected = np.asarray([i for i in selected if training[i]])
            return delta[selected].ravel()

        optimized = least_squares(
            residual,
            initial,
            loss="soft_l1",
            f_scale=4,
            max_nfev=1200,
            bounds=(
                np.r_[[-np.inf] * 6, np.log(100), -0.12, -0.05, -0.05],
                np.r_[[np.inf] * 6, np.log(1000), 0.03, 0.05, 0.05],
            ),
        )
        all_errors = np.linalg.norm(project(optimized.x) - pixels, axis=1)
        matched = np.asarray([g[np.argmin(all_errors[g])] for g in groups])
        errors = all_errors[matched]
        held_out_matches = held_out[matched]
        candidate = (
            float(np.median(errors[held_out_matches])),
            optimized.x,
            errors,
            matched,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("No robust source camera estimate from heuristic cyan tracking")
    score, parameters, errors, matched = best
    held_out_matches = held_out[matched]
    rotation = cv2.Rodrigues(parameters[:3])[0]
    points = translations + rotations @ parameters[7:10]
    depths = (points[matched] @ rotation.T + parameters[3:6])[:, 2]
    p90 = float(np.percentile(errors[held_out_matches], 90))
    if np.any(depths <= 0) or score > 8 or p90 > 16:
        raise ValueError(
            f"Camera candidate failed withheld-source-frame check: "
            f"median {score:.2f}px, p90 {p90:.2f}px"
        )
    eye = -rotation.T @ parameters[3:6]
    opengl = rotation.T @ np.diag([1, -1, -1])
    focal = float(np.exp(parameters[6]))
    return {
        "camera_eye": eye.tolist(),
        "camera_rotation_opengl_xyzw": Rotation.from_matrix(opengl).as_quat().tolist(),
        "focal_length_mm": focal * 5.376 / 320,
        "focal_length_pixels": focal,
        "estimated_cyan_mount_offset_in_gripper_base": parameters[7:10].tolist(),
        "correspondences": len(matched),
        "competing_component_candidates": len(tracks),
        "matched_tracks": [tracks[i] for i in matched],
        "withheld_correspondences": int(held_out_matches.sum()),
        "withheld_median_reprojection_error_pixels": score,
        "withheld_p90_reprojection_error_pixels": p90,
        "original_camera_calibration_recovered": False,
        "basis": (
            "moving cyan candidate matching, robot FK, pinhole fit; "
            "candidate requires visual review"
        ),
        "per_episode_rgb_correspondence_review_complete": False,
    }


def run(inventory, kinematics, output):
    output.mkdir(parents=True, exist_ok=False)
    kin = json.loads(kinematics.read_text())
    base = np.asarray(kin["base"])
    chain = [(np.asarray(a), np.asarray(b), axis) for a, b, axis in kin["chain"]]
    entries = [json.loads(x) for x in (inventory / "episodes.jsonl").read_text().splitlines()]
    scenes = {}
    for entry in entries:
        scenes.setdefault(entry["source_scene_name"], entry)
    results = []
    for scene, source in sorted(scenes.items()):
        record = load_source_record(source, inventory)
        with np.load(inventory / "episodes" / source["id"] / "source_numeric.npz") as data:
            poses = np.asarray([forward(base, chain, q[:7])[0] for q in data["states"]])
        indices = np.arange(0, source["frames"], max(1, source["frames"] // 90))
        tracks = moving_cyan_points(record["steps/observation/exterior_image_1_left"], indices)
        result = {
            "scene": scene,
            "source_id": source["id"],
            "tracks": tracks,
            "source_record_sha256": source["record_sha256"],
            "estimated_at_utc": utc(),
        }
        try:
            result.update(status="candidate", estimate=estimate(poses, tracks))
        except ValueError as error:
            result.update(status="insufficient_correspondences", reason=str(error))
        results.append(result)
        write_json(output / f"{scene}.json", result)
        write_json(
            output / "status.json",
            {
                "status": "running" if len(results) < len(scenes) else "complete",
                "source_scenes": len(scenes),
                "checked_scenes": len(results),
                "candidate_estimates": sum(r["status"] == "candidate" for r in results),
                "original_camera_calibration_recovered": False,
                "kinematics_sha256": sha256(kinematics),
                "checked_at_utc": utc(),
            },
        )
        print(scene, result["status"], len(tracks), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inventory, args.kinematics, args.output)


if __name__ == "__main__":
    main()

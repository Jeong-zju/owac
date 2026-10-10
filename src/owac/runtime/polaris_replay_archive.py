"""Archive only the expert replay pilot's small numerical metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_replay import transform, utc

REMOTE = "/workspace/owac/reference/polaris-replay-pilot-v1"
SMALL = (
    "coordination.json",
    "launch-config.json",
    "replay/config.json",
    "replay/initial.json",
    "replay/started.json",
    "replay/error.json",
    "replay/postprocess.json",
    "replay/result.json",
    "replay/dataset.json",
    "replay/frames.jsonl",
    "replay/files.json",
    "replay/sensor-external_camera.json",
    "replay/sensor-wrist_camera.json",
)


def verify_frames(rows, result, config):
    original = [r for r in rows if r["source_frame"] is not None]
    count = config["frames"]
    if [r["source_frame"] for r in original] != list(range(count)):
        raise ValueError("Missing or repeated original expert frame")
    if result["source_frames"] != count or result["native_capture"]["control_frames"] != count:
        raise ValueError("Capture count does not cover the original trajectory")
    if result["verified_file_count"] != count * 8:
        raise ValueError("Verified capture file count is incomplete")
    previous = {}
    maximum_wrist_pose_error = 0.0
    for frame, row in enumerate(original):
        capture = row["capture"]
        if capture["step"] != frame or capture["physical_step_counter"] != frame:
            raise ValueError("Capture and source action indices differ")
        if set(capture["views"]) != {"external_camera", "wrist_camera"}:
            raise ValueError("Paired capture is missing a view")
        if not np.isfinite(row["original_action"]).all() or len(row["original_action"]) != 8:
            raise ValueError("Invalid original action")
        if frame and (
            capture["physics_substep_counter"]
            - original[frame - 1]["capture"]["physics_substep_counter"]
            != config["decimation"]
        ):
            raise ValueError("Camera capture changed physical control timing")
        for name, view in capture["views"].items():
            if view["native_frame_id"] <= previous.get(name, -1):
                raise ValueError("Native CFA frame was reused or moved backwards")
            previous[name] = view["native_frame_id"]
            if view["cfa_quantization_maximum_error_dn"] > 1:
                raise ValueError("Native HDR/CFA quantization mismatch")
            if view["dtype"] != "uint32" or view["shape"] != list(
                reversed(config["camera_resolution"])
            ):
                raise ValueError("Native CFA is not a complete lossless integer frame")
            if view["paired_rgb"]["shape"] != [*reversed(config["camera_resolution"]), 3]:
                raise ValueError("Paired RGB shape differs from native CFA")
            if name == "wrist_camera":
                x, y, z, w = row["gripper_base_quaternion_xyzw"]
                expected = transform(row["gripper_base_position"], [w, x, y, z]) @ [
                    0.011,
                    -0.031,
                    -0.074,
                    1,
                ]
                error = float(np.linalg.norm(expected[:3] - view["position_world"]))
                maximum_wrist_pose_error = max(error, maximum_wrist_pose_error)
                if error > 0.002:
                    raise ValueError(
                        f"Native wrist camera pose is stale or inconsistent: {error} m"
                    )
                expected_rotation = (
                    transform(row["gripper_base_position"], [w, x, y, z])
                    @ transform((0.011, -0.031, -0.074), (-0.420, 0.570, 0.576, -0.409))
                )[:3, :3]
                rx, ry, rz, rw = view["quaternion_opengl_xyzw"]
                measured_rotation = transform((0, 0, 0), (rw, rx, ry, rz))[:3, :3]
                if np.max(np.abs(measured_rotation - expected_rotation)) > 0.005:
                    raise ValueError(
                        "Native wrist camera orientation does not match the source mount"
                    )
    if result["checksum_mismatches"] or result["object_state_overrides_after_initialization"]:
        raise ValueError("Replay altered object state or failed checksums")
    if result["original_action_changed"]:
        raise ValueError("Replay changed original expert actions")
    return {
        "original_frames": count,
        "view_frames": 2 * count,
        "valid": True,
        "maximum_wrist_position_error_m": maximum_wrist_pose_error,
    }


def archive(destination, benchmark_destination):
    script = f"""import json,pathlib,datetime,hashlib,subprocess
root=pathlib.Path({REMOTE!r})
files={{}}
index=root/'successful-expert-index.json'
if index.exists(): files[index.name]=index.read_text()
for attempt in sorted((root/'attempts').glob('attempt-*')):
 for name in {SMALL!r}:
  path=attempt/name
  if path.exists():
   if path.stat().st_size>8_000_000: raise ValueError('Metadata unexpectedly large')
   files[str(path.relative_to(root))]=path.read_text()
supervisor=subprocess.run(['supervisorctl','status','owac-arena-benchmark-v1'],
                         capture_output=True,text=True).stdout.strip()
print(json.dumps({{'checked_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),
                  'remote_root':str(root),'benchmark_supervisor':supervisor,'files':files}}))
"""
    response = subprocess.run(
        [
            "ssh",
            "-C",
            "-p",
            "16406",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "root@212.51.142.37",
            "python3 -",
        ],
        input=script,
        capture_output=True,
        text=True,
        check=True,
    )
    snapshot = json.loads(response.stdout)
    files = snapshot.pop("files")
    results = []
    attempts = []
    for relative, text in files.items():
        if relative.endswith("coordination.json"):
            attempts.append(json.loads(text))
        if relative.endswith("replay/result.json"):
            result = json.loads(text)
            prefix = relative.removesuffix("result.json")
            rows = [json.loads(x) for x in files[prefix + "frames.jsonl"].splitlines()]
            config = json.loads(files[prefix + "config.json"])
            result["local_frame_validation"] = verify_frames(rows, result, config)
            result["attempt"] = relative.split("/")[1]
            result["remote_directory"] = REMOTE + "/" + prefix
            results.append(result)
    successful = [r for r in results if r["success"]]
    summary = {
        **snapshot,
        "status": "successful_reconstructed_expert_replay"
        if successful
        else "no_successful_replay_yet",
        "scope": "original expert actions; reconstructed scene; original alignment unknown",
        "successful_unique_source_episodes": len(
            {r["source"]["record_sha256"] for r in successful}
        ),
        "complete_physical_replays": len(results),
        "attempts": attempts,
        "results": results,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for relative, text in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    write_json(destination / "summary.json", summary)
    manifest = {
        str(path.relative_to(destination)): sha256(path)
        for path in destination.rglob("*")
        if path.is_file() and path.name != "archive.json"
    }
    write_json(
        destination / "archive.json", {**snapshot, "sha256": manifest, "local_generated_utc": utc()}
    )
    if benchmark_destination is not None:
        write_json(benchmark_destination / "replay_pilot.json", summary)
    for name, digest in manifest.items():
        if sha256(destination / name) != digest:
            raise ValueError(f"Local replay archive checksum mismatch: {name}")
    print(
        json.dumps(
            {
                "destination": str(destination),
                "results": len(results),
                "successful_sources": len(successful),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("research/experiments/results/2026-10-10_polaris_expert_replay_pilot"),
    )
    parser.add_argument(
        "--benchmark-destination",
        type=Path,
        default=Path("research/experiments/results/2026-10-10_pi05_arena_droid_benchmark"),
    )
    args = parser.parse_args()
    archive(args.destination, args.benchmark_destination)


if __name__ == "__main__":
    main()

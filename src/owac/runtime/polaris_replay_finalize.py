"""Finish metadata after a fully recorded replay, without rerunning physics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_replay import utc
from owac.runtime.polaris_replay_archive import verify_frames
from owac.simulation.polaris_replay import evaluate
from owac.simulation.raw_camera import RawCameraProfile, audit_cfa, read_buffer


def finalize(directory):
    if (directory / "result.json").exists():
        raise FileExistsError("Completed replay result already exists")
    rows = [json.loads(x) for x in (directory / "frames.jsonl").read_text().splitlines()]
    config = json.loads((directory / "config.json").read_text())
    dataset = json.loads((directory / "dataset.json").read_text())
    source = Path(config["source_episode"]) / "trajectory.npz"
    with np.load(source) as original, np.load(directory / dataset["trajectory"]["path"]) as paired:
        actions = original["actions"]
        if not np.array_equal(actions, paired["original_actions"]):
            raise ValueError("Paired dataset changed the original actions")
        if not np.array_equal(
            actions, np.asarray([r["original_action"] for r in rows[: len(actions)]])
        ):
            raise ValueError("Recorded command values differ from the original expert actions")
    if len(rows) != len(actions) + round(config["settle_seconds"] * 15):
        raise ValueError("Replay did not finish all source and settling frames")
    width, height = config["camera_resolution"]
    profile = RawCameraProfile(width=width, height=height)
    files = []
    maximum = 0
    ids = {}
    for row in rows[: len(actions)]:
        for name, view in row["capture"]["views"].items():
            for item in [*view["files"].values(), view["paired_rgb"]]:
                path = directory / item["path"]
                if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                    raise ValueError(f"Replay capture checksum mismatch: {path}")
                files.append(item)
            hdr = read_buffer(
                directory / view["files"]["0-texread"]["path"], "<f2", (height, width, 4)
            )
            cfa = read_buffer(directory / view["files"]["2-cfa"]["path"], "<u4", (height, width))
            audit = audit_cfa(hdr, cfa, profile)
            maximum = max(maximum, audit["cfa_quantization_maximum_error_dn"])
            ids.setdefault(name, []).append(view["native_frame_id"])
    coordination = json.loads((directory.parent / "coordination.json").read_text())
    error_path = directory / "error.json"
    original_finish = datetime.fromtimestamp(error_path.stat().st_mtime, UTC).isoformat()
    result = {
        **evaluate(rows, config),
        "started_utc": json.loads((directory / "started.json").read_text())["started_utc"],
        "finished_utc": original_finish,
        "finish_time_source": "error.json mtime after trajectory and dataset serialization",
        "finalized_utc": utc(),
        "wall_seconds": None,
        "wall_timer_note": "original monotonic timer lost during JSON serialization error",
        "capture_span_seconds": (
            rows[len(actions) - 1]["capture"]["host_timestamp_ns"]
            - rows[0]["capture"]["host_timestamp_ns"]
        )
        / 1e9,
        "child_wall_seconds": (
            datetime.fromisoformat(coordination["resumed_dispatcher_utc"])
            - datetime.fromisoformat(coordination["episode_drained_utc"])
        ).total_seconds(),
        "source_frames": len(actions),
        "settle_frames": len(rows) - len(actions),
        "original_action_changed": False,
        "object_state_overrides_after_initialization": 0,
        "scope": config["scope"],
        "source": config["source"],
        "native_capture": {
            "control_frames": len(actions),
            "views": {
                name: {
                    "frames": len(values),
                    "first_native_frame_id": min(values),
                    "last_native_frame_id": max(values),
                }
                for name, values in ids.items()
            },
            "maximum_quantization_error_dn": maximum,
            "scope": "synthetic native scene HDR to CFA; physical calibration is not claimed",
        },
        "verified_file_count": len(files),
        "checksum_mismatches": [],
        "metadata_sha256": sha256(directory / "frames.jsonl"),
        "config_sha256": sha256(directory / "config.json"),
        "finalization_repair": "cast NumPy bool to JSON bool; no physics or capture rerun",
    }
    result["frame_validation"] = verify_frames(rows, result, config)
    for key in ("trajectory", "frame_manifest", "scene"):
        if sha256(directory / dataset[key]["path"]) != dataset[key]["sha256"]:
            raise ValueError(f"Paired dataset manifest mismatch: {key}")
    write_json(directory / "files.json", {"files": files, "verified_utc": utc()})
    write_json(directory / "result.json", result)
    write_json(
        directory / "postprocess.json",
        {
            "finalized_utc": utc(),
            "original_error_sha256": sha256(error_path),
            "finalizer_sha256": sha256(Path(__file__)),
            "evaluator_sha256": sha256(Path(__file__).parents[1] / "simulation/polaris_replay.py"),
            "physics_rerun": False,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.directory)))


if __name__ == "__main__":
    main()

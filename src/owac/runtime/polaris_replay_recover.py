"""Recover a completed native replay after a numerical JSON serialization failure."""

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


def recover(replay):
    error = json.loads((replay / "error.json").read_text())
    if "not JSON serializable" not in error["error"] or "write_json" not in error["traceback"]:
        raise ValueError("Only completed numerical serialization failures are recoverable here")
    if (replay / "result.json").exists() or (replay / "postprocess.json").exists():
        raise ValueError("Existing result/recovery must be inspected, not overwritten")
    config = json.loads((replay / "config.json").read_text())
    data = json.loads((replay / "dataset.json").read_text())
    rows = [json.loads(x) for x in (replay / "frames.jsonl").read_text().splitlines()]
    original = [r for r in rows if r["source_frame"] is not None]
    count = config["frames"]
    if len(rows) != count + data["settle_frames_are_auxiliary_and_excluded_from_training"]:
        raise ValueError("Incomplete physical settling interval")
    unchanged = {
        name: sha256(replay / name)
        for name in (
            "frames.jsonl",
            "paired-trajectory.npz",
            "reconstructed_scene.usda",
            "dataset.json",
        )
    }
    if unchanged["paired-trajectory.npz"] != data["trajectory"]["sha256"]:
        raise ValueError("Recorded numerical trajectory changed")
    if unchanged["frames.jsonl"] != data["frame_manifest"]["sha256"]:
        raise ValueError("Recorded frame manifest changed")
    with np.load(Path(config["source_episode"]) / "trajectory.npz") as source:
        actions = source["actions"].copy()
    with np.load(replay / "paired-trajectory.npz") as recorded:
        if not np.array_equal(actions, recorded["original_actions"]):
            raise ValueError("Original expert labels changed")
    files = []
    for row in original:
        for view in row["capture"]["views"].values():
            files += [*view["files"].values(), view["paired_rgb"]]
    for metadata in files:
        path = replay / metadata["path"]
        if sha256(path) != metadata["sha256"]:
            raise ValueError("Captured image or native buffer changed")
    finished = datetime.fromtimestamp((replay / "error.json").stat().st_mtime, UTC).isoformat()
    started = json.loads((replay / "started.json").read_text())["started_utc"]
    result = {
        **evaluate(rows, config),
        "started_utc": started,
        "finished_utc": finished,
        "physical_finish_time_basis": "error.json mtime after completed data save",
        "wall_seconds": (
            datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        ).total_seconds(),
        "wall_time_basis": "capture start to error-file mtime; pre-capture startup excluded",
        "source_frames": count,
        "settle_frames": len(rows) - count,
        "original_action_changed": False,
        "object_state_overrides_after_initialization": 0,
        "scope": config["scope"],
        "source": config["source"],
        "native_capture": {
            "control_frames": count,
            "views": {
                name: {
                    "frames": count,
                    "first_native_frame_id": original[0]["capture"]["views"][name][
                        "native_frame_id"
                    ],
                    "last_native_frame_id": original[-1]["capture"]["views"][name][
                        "native_frame_id"
                    ],
                }
                for name in data["views"]
            },
            "maximum_quantization_error_dn": max(
                view["cfa_quantization_maximum_error_dn"]
                for row in original
                for view in row["capture"]["views"].values()
            ),
            "scope": "synthetic native HDR to CFA; physical noise calibration not claimed",
        },
        "verified_file_count": len(files),
        "checksum_mismatches": [],
        "metadata_sha256": unchanged["frames.jsonl"],
        "config_sha256": sha256(replay / "config.json"),
        "postprocessed_utc": utc(),
        "physical_replay_repeated": False,
    }
    checked = verify_frames(rows, result, config)
    # Validate serialization before writing any final metadata.
    json.dumps(result)
    write_json(replay / "result.json", result)
    write_json(replay / "files.json", {"files": files, "verified_utc": utc()})
    write_json(
        replay / "postprocess.json",
        {
            "reason": error["error"],
            "original_error_preserved": True,
            "unchanged_captured_data_sha256": unchanged,
            "validation": checked,
            "physical_replay_repeated": False,
            "processed_at_utc": utc(),
        },
    )
    for name, digest in unchanged.items():
        if sha256(replay / name) != digest:
            raise ValueError("Recovery changed captured data")
    batch = replay.parents[2]
    status = json.loads((batch / "status.json").read_text())
    for row in status["completed"]:
        if row["id"] == replay.parent.name:
            row.update(
                result=result,
                infrastructure_error=False,
                original_exit_code=row["exit_code"],
                recovered_postprocessing=True,
            )
    status["original_status_sha256"] = sha256(batch / "status.json")
    status["recovered_at_utc"] = utc()
    write_json(batch / "recovered-status-v1.json", status)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.replay)
    print(json.dumps({k: result[k] for k in ("success", "checks", "source_frames")}))


if __name__ == "__main__":
    main()

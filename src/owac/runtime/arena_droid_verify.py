"""Validate complete paired Arena trials and optional cloud-only RAW binary hashes."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.arena_droid_archive import completed_episode


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    d = 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return [max(0.0, center - radius), min(1.0, center + radius)]


def verify_phase(run, seeds, *, raw=False, verify_binaries=False):
    status = json.loads((run / "status.json").read_text())
    require(status["status"] == "complete", "Run is incomplete")
    require(status["completed_seeds"] == seeds, "Predeclared seeds are incomplete or changed")
    trials = []
    total_bytes = 0
    total_files = 0
    for seed in seeds:
        directory = run / f"seed-{seed}"
        episode, summary = completed_episode(directory, seed)
        require(episode is not None, f"No valid episode for seed {seed}")
        evidence = directory / "evidence"
        controls = [
            json.loads(line) for line in (evidence / "control.jsonl").read_text().splitlines()
        ]
        count = len(controls)
        require(count == episode["episode_length"] == summary["control_steps"], "Step mismatch")
        require(count > 0, "Empty episode")
        ids = {view: [] for view in ("external_camera", "wrist_camera")}
        cfa_hashes = {view: set() for view in ids}
        previous_timestamp = -1
        previous_physics = None
        maximum_error = 0
        for step, record in enumerate(controls):
            require(record["step"] == step, "Control sequence has gaps or duplicates")
            for key, size in (("joint_position", 7), ("gripper_position", 1), ("action", 8)):
                value = record[key]
                require(len(value) == 1 and len(value[0]) == size, f"Invalid {key} shape")
                require(all(math.isfinite(x) for x in value[0]), f"Nonfinite {key}")
            if not raw:
                continue
            sample = record.get("raw")
            require(sample is not None, "Missing RAW control frame")
            require(
                sample["step"] == step and sample["phase"] == "pre_action", "RAW phase mismatch"
            )
            require(set(sample["views"]) == set(ids), "Missing RAW view")
            require(sample["host_timestamp_ns"] > previous_timestamp, "RAW timestamps not ordered")
            previous_timestamp = sample["host_timestamp_ns"]
            physics = (sample["physical_step_counter"], sample["physics_substep_counter"])
            if previous_physics is not None:
                require(physics[0] - previous_physics[0] == 1, "Extra or missing control advance")
                require(physics[1] - previous_physics[1] == 8, "Extra or missing physics advance")
            previous_physics = physics
            for view, frame in sample["views"].items():
                require(frame["shape"] == [720, 1280] and frame["dtype"] == "uint32", "CFA format")
                require(0 <= frame["minimum_dn"] <= frame["maximum_dn"] <= 16777215, "CFA range")
                require(frame["unique_values"] >= 16, "CFA contains no useful scene signal")
                require(frame["cfa_quantization_maximum_error_dn"] <= 1, "HDR/CFA mismatch")
                maximum_error = max(maximum_error, frame["cfa_quantization_maximum_error_dn"])
                frame_id = frame["native_frame_id"]
                require(not ids[view] or frame_id > ids[view][-1], "Repeated or old native frame")
                ids[view].append(frame_id)
                cfa_hashes[view].add(frame["files"]["2-cfa"]["sha256"])
                require(set(frame["files"]) == {"0-texread", "2-cfa", "3-noise"}, "Buffer set")
                for kind, file in frame["files"].items():
                    expected = 720 * 1280 * (8 if kind == "0-texread" else 4)
                    require(file["bytes"] == expected, "Native buffer size")
                    require(len(file["sha256"]) == 64, "Missing buffer hash")
                    total_bytes += expected
                    total_files += 1
                    if verify_binaries:
                        path = evidence / file["path"]
                        require(
                            path.resolve().is_relative_to(evidence.resolve()),
                            "Buffer escapes evidence",
                        )
                        require(path.stat().st_size == expected, "Cloud buffer incomplete")
                        require(sha256(path) == file["sha256"], "Cloud buffer SHA256 mismatch")
                if verify_binaries:
                    from owac.simulation.raw_camera import RawCameraProfile, audit_cfa, read_buffer

                    files = frame["files"]
                    hdr = read_buffer(evidence / files["0-texread"]["path"], "<f2", (720, 1280, 4))
                    cfa = read_buffer(evidence / files["2-cfa"]["path"], "<u4", (720, 1280))
                    audit = audit_cfa(hdr, cfa, RawCameraProfile(width=1280, height=720))
                    for key, value in audit.items():
                        require(frame[key] == value, f"Stored CFA audit disagrees: {key}")
                    noisy = read_buffer(evidence / files["3-noise"]["path"], "<u4", (720, 1280))
                    require(int(noisy.max()) == frame["noise"]["maximum_dn"], "Noise audit")
                    require(int(noisy.min()) == frame["noise"]["minimum_dn"], "Noise audit")
        for view, expected in summary["distinct_rgb_frames"].items():
            actual = len({record["rgb_sha256"][view] for record in controls})
            require(actual == expected and actual > 1, "Frozen or inconsistent RGB")
        if raw:
            require(summary["raw"]["control_frames"] == count, "RAW frame count mismatch")
            for view in ids:
                require(len(ids[view]) == summary["raw"]["views"][view]["frames"], "View count")
                require(len(cfa_hashes[view]) > 1, "Native CFA view is frozen")
        metrics = json.loads((directory / "policy_metrics.json").read_text())
        trials.append(
            {
                "seed": seed,
                "success": episode["success"],
                "control_steps": count,
                "inference_calls": summary["inference_calls"],
                "rollout_wall_seconds": summary["wall_seconds"],
                "inference_seconds_p50": summary["inference_seconds_p50"],
                "inference_seconds_p95": summary["inference_seconds_p95"],
                "policy_peak_allocated_bytes": metrics["peak_allocated_bytes"],
                "raw_frames_per_view": count if raw else None,
                "maximum_quantization_error_dn": maximum_error if raw else None,
            }
        )
    successes = sum(trial["success"] for trial in trials)
    prior_wall = 0.0
    parent = status.get("resumed_from")
    seen = {run.name}
    while parent:
        name = Path(parent).name
        require(name not in seen, "Resume lineage contains a cycle")
        seen.add(name)
        previous = json.loads((run.parent / name / "status.json").read_text())
        prior_wall += previous["wall_seconds"]
        parent = previous.get("resumed_from")
    return {
        "run": run.name,
        "successes": successes,
        "episodes": len(trials),
        "wilson_95": wilson(successes, len(trials)),
        "trials": trials,
        "run_wall_seconds": status["wall_seconds"],
        "prior_attempt_wall_seconds": prior_wall,
        "all_attempts_wall_seconds": status["wall_seconds"] + prior_wall,
        "resumed_from": status.get("resumed_from"),
        "carried_forward_seeds": status.get("carried_forward_seeds", []),
        "raw_binary_files": total_files,
        "raw_binary_bytes": total_bytes,
        "cloud_binary_hashes_verified": verify_binaries if raw else None,
        "cloud_hdr_cfa_reaudited": verify_binaries if raw else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rgb-run", default="rgb-v6")
    parser.add_argument("--raw-run", default="raw-v6")
    parser.add_argument("--verify-binaries", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    seeds = protocol["environment_seeds"]
    result = {
        "verification_started_at_utc": datetime.now(UTC).isoformat(),
        "verifier_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(args.directory / "protocol.json"),
        "scope": "single task paired seeds; phases are not independent pooled trials",
        "rgb": verify_phase(args.directory / "runs" / args.rgb_run, seeds),
        "raw": verify_phase(
            args.directory / "runs" / args.raw_run,
            seeds,
            raw=True,
            verify_binaries=args.verify_binaries,
        ),
    }
    result["verified_at_utc"] = datetime.now(UTC).isoformat()
    path = args.directory / (
        "verification.json" if args.verify_binaries else "local_verification.json"
    )
    write_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

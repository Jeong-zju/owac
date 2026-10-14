"""Versioned scene-reconstruction batches that retain all source episodes and failures."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_replay import descendants, forward, prepare, robot_chain, utc

RUNTIME = Path("/workspace/owac/reference/pi05-arena-droid-v1/Arena")
FIELD_MAPPING = {
    "exterior_image_1_left": "external_camera",
    "exterior_image_2_left": "external_camera",
    "wrist_image_left": "wrist_camera",
}


def effective_status(root):
    saved = json.loads((root / "status.json").read_text())
    recovery = root / "recovered-status-v1.json"
    if recovery.exists():
        recovered = json.loads(recovery.read_text())
        if recovered["original_status_sha256"] != sha256(root / "status.json"):
            raise ValueError("Recovery does not match original immutable batch status")
        for row in recovered["completed"]:
            if row.get("recovered_postprocessing"):
                path = root / "runs" / row["id"] / "replay/result.json"
                if json.loads(path.read_text()) != row["result"]:
                    raise ValueError("Recovered batch result differs from verified replay result")
        return recovered
    return saved


def carry_anchors(actions, tcp, minimum_displacement=0.1):
    close = np.flatnonzero((actions[:, 7] > 0.5) & (np.r_[0, actions[:-1, 7]] <= 0.5))
    opening = np.flatnonzero((actions[:, 7] <= 0.5) & (np.r_[0, actions[:-1, 7]] > 0.5))
    cycles = []
    for start in close:
        end = next((int(i) for i in opening if i > start), None)
        if end is not None:
            cycles.append(
                {
                    "close": int(start),
                    "open": end,
                    "xy_displacement_m": float(np.linalg.norm(tcp[end, :2] - tcp[start, :2])),
                    "maximum_lift_m": float(tcp[start : end + 1, 2].max() - tcp[start, 2]),
                }
            )
    carries = [c for c in cycles if c["xy_displacement_m"] >= minimum_displacement]
    if not carries:
        raise ValueError("No source carry segment; cannot infer cube scene from a dry close")
    return (carries[0]["close"], carries[-1]["open"]), cycles


def yaw_of_opening_axis(pose):
    axis = pose[:2, 1]
    if np.linalg.norm(axis) < 0.2:
        raise ValueError("Gripper opening axis is nearly vertical; floor yaw cannot be inferred")
    return float(np.arctan2(axis[1], axis[0]))


def settled_approach_layout(config, tcp):
    """Infer initial support height from the source approach, not the first close frame.

    This is offline scene inference. It changes initialization and attachment readiness,
    never source labels or post-initialization object state.
    """
    result = copy.deepcopy(config)
    cycles = result["grasp_assistance"]["cycles"]
    cycle = cycles[0]
    frames = range(cycle["close"], min(cycle["close"] + 6, cycle["open"]))
    ready = min(frames, key=lambda frame: tcp[frame, 2])
    height = result.get("object_size", [result["cube_size"]] * 3)[2]
    top = float(tcp[ready, 2] - height / 2)
    result["table_height_revision"] = {
        "old_top_m": result["table_top"],
        "new_top_m": top,
        "source_ready_frame": ready,
        "object_size_z_m": height,
        "object_state_overrides_after_initialization": 0,
        "basis": "minimum source TCP z in first six closed frames; approximate support plane",
    }
    result["table_top"] = top
    result["cube_initial_position"][2] = top + height / 2 + 0.001
    result["target_center"][2] = top
    cycle["not_before"] = ready
    result["grasp_assistance"]["readiness_rule"] = result["table_height_revision"]["basis"]
    result["grasp_assistance"]["gripper_object_collisions_filtered_only_while_attached"] = True
    return result


def rotate_anchors(actions, poses):
    close = np.flatnonzero((actions[:, 7] > 0.5) & (np.r_[0, actions[:-1, 7]] <= 0.5))
    opening = np.flatnonzero((actions[:, 7] <= 0.5) & (np.r_[0, actions[:-1, 7]] > 0.5))
    candidates = []
    for start in close:
        end = next((int(i) for i in opening if i > start), None)
        if end is None:
            continue
        angle = yaw_of_opening_axis(poses[end]) - yaw_of_opening_axis(poses[start])
        delta = float(np.arctan2(np.sin(angle), np.cos(angle)))
        candidates.append({"close": int(start), "open": end, "yaw_change_rad": delta})
    selected = [c for c in candidates if abs(c["yaw_change_rad"]) > np.pi / 6]
    if not selected:
        raise ValueError("No original released grasp with measurable horizontal rotation")
    best = max(selected, key=lambda c: abs(c["yaw_change_rad"]))
    return (best["close"], best["open"]), candidates


def plan(
    inventory,
    output,
    asset,
    profile="cube",
    limit=None,
    source_ids=None,
    initial_grasp="carry",
    replay_mode="joint_targets",
    camera_estimates=None,
    tcp_offset_m=0.14,
    grasp_assistance=False,
    settled_approach=False,
):
    if settled_approach and (not grasp_assistance or profile == "marker"):
        raise ValueError("Settled approach is validated only for assisted cube/rotate layouts")
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(x) for x in (inventory / "episodes.jsonl").read_text().splitlines()]
    summary = json.loads((inventory / "summary.json").read_text())
    if sha256(inventory / "episodes.jsonl") != summary["source_manifest_sha256"]:
        raise ValueError("Source inventory changed before reconstruction")
    (output / "assets").mkdir()
    shutil.copy2(asset, output / "assets" / asset.name)
    base, chain = robot_chain(asset)
    selected, pending = [], []
    instruction = {
        "cube": "put the cube in the left bin",
        "rotate": "turn the block 90 degrees",
        "marker": "put the marker in the mug",
    }[profile]
    for row in rows:
        if (
            row["instructions"] != [instruction]
            or (limit and len(selected) >= limit)
            or (source_ids and row["id"] not in source_ids)
        ):
            pending.append(
                {
                    "id": row["id"],
                    "scene": row["source_scene_name"],
                    "status": "scene_profile_pending",
                }
            )
            continue
        episode = output / "episodes" / row["id"]
        episode.mkdir(parents=True)
        with np.load(inventory / "episodes" / row["id"] / "source_numeric.npz") as saved:
            np.savez_compressed(
                episode / "trajectory.npz", actions=saved["actions"], states=saved["states"]
            )
            actions, states = saved["actions"].copy(), saved["states"].copy()
        source = {**row, "instruction": row["instructions"][0]}
        write_json(episode / "source.json", source)
        poses = [forward(base, chain, q[:7], (tcp_offset_m, 0, 0)) for q in states]
        tcp = np.asarray([pose[1] for pose in poses])
        try:
            if profile == "rotate":
                anchors, cycles = rotate_anchors(actions, [pose[0] for pose in poses])
            else:
                anchors, cycles = carry_anchors(actions, tcp, 0.05 if profile == "marker" else 0.1)
        except ValueError as error:
            pending.append(
                {"id": row["id"], "status": "anchor_review_required", "reason": str(error)}
            )
            continue
        config = prepare(output, episode, carry_cycle=anchors, tcp_offset=(tcp_offset_m, 0, 0))
        config.update(
            experiment="polaris-native-raw-reconstruction-v1-cube-profile-v2",
            scope="coarse reconstructed source task; original scene/camera calibration is unknown",
            source_scene_profile="pi_scene_v5/cube-left-bin-v2",
            source_inventory_episode=str(inventory / "episodes" / row["id"]),
            replay_mode=replay_mode,
            source_action_execution=(
                "original joint targets"
                if replay_mode == "joint_targets"
                else "reference-driven arm; original gripper command"
            ),
            source_grip_cycles=cycles,
            anchor_rule="first carry close, last carry release; exclude dry closes from layout",
            cube_color=[0.03, 0.05, 0.8],
            table_color=[0.04, 0.09, 0.07],
            left_bin_color=[0.8, 0.85, 0.85],
            right_bin_color=[0.8, 0.85, 0.85],
            source_camera_mapping=FIELD_MAPPING,
            source_rgb_correspondence={
                "profile_reference_visually_reviewed": True,
                "per_episode_scene_and_rgb_review_complete": False,
                "basis": "source pi_scene_v5 cube-left-bin keyframe visually inspected",
                "reference_rgb_keyframes": row["reference_rgb_keyframes"],
                "original_camera_calibration_recovered": False,
                "original_background_recovered": False,
                "original_rgb_pixel_alignment_verified": False,
            },
        )
        if profile == "rotate":
            yaw = yaw_of_opening_axis(poses[anchors[0]][0]) - np.pi / 2
            config.update(
                experiment="polaris-native-raw-reconstruction-v1-rotate-profile-v1",
                source_scene_profile="workbench_zed/wood-block-rotation-v1",
                task_kind="rotate",
                object_size=[0.10, 0.035, 0.03],
                object_initial_quaternion_xyzw=[
                    0,
                    0,
                    float(np.sin(yaw / 2)),
                    float(np.cos(yaw / 2)),
                ],
                cube_initial_position=[*tcp[anchors[0], :2], config["table_top"] + 0.016],
                cube_color=[0.65, 0.42, 0.20],
                table_color=[0.45, 0.13, 0.4],
                spawn_target_bins=False,
                anchor_rule="released source grasp with largest horizontal orientation change",
                success_rotation_target_deg=90,
                success_rotation_tolerance_deg=25,
                source_rgb_correspondence={
                    **config["source_rgb_correspondence"],
                    "basis": "workbench_zed source wood block on purple work mat visually reviewed",
                },
            )
        elif initial_grasp == "first_close":
            first = row["close_transition_frames"][0]
            config.update(
                experiment="polaris-native-raw-reconstruction-v1-cube-profile-v3",
                source_scene_profile="pi_scene_v5/cube-left-bin-v3",
                cube_initial_position=[*tcp[first, :2], config["table_top"] + 0.021],
                initial_anchor_frame=first,
                initial_anchor_rule="first source close XY; table height from later carry grasp",
                revision_reason="source includes failed initial grasps which may move the object",
            )
        if profile == "marker":
            yaw = yaw_of_opening_axis(poses[anchors[0]][0]) - np.pi / 2
            config.update(
                experiment="polaris-native-raw-reconstruction-v1-marker-profile-v1",
                source_scene_profile=f"{row['source_scene_name']}/marker-mug-v1",
                task_kind="insert",
                object_size=[0.10, 0.02, 0.025],
                object_initial_quaternion_xyzw=[
                    0,
                    0,
                    float(np.sin(yaw / 2)),
                    float(np.cos(yaw / 2)),
                ],
                cube_initial_position=[*tcp[anchors[0], :2], config["table_top"] + 0.0135],
                cube_color=[0.42, 0.02, 0.65],
                cube_mass=0.012,
                table_color=[0.8, 0.63, 0.03],
                table_shape="cylinder",
                target_kind="mug",
                mug_inner_radius=0.05,
                mug_wall_height=0.10,
                left_bin_color=[0.65, 0.015, 0.01],
                success={**config["success"], "lift_minimum_m": 0.03, "transport_minimum_m": 0.05},
                source_rgb_correspondence={
                    **config["source_rgb_correspondence"],
                    "basis": "kanav_cloth_chairs source purple grasped brick/red mug/yellow table",
                    "asset_approximation": "colored slender brick; source instruction retained",
                    "profile_reference_visually_reviewed": row["source_scene_name"]
                    == "kanav_cloth_chairs",
                },
            )
            if row["source_scene_name"] != "kanav_cloth_chairs":
                pending.append(
                    {"id": row["id"], "status": "scene_profile_reference_review_pending"}
                )
                continue
        if camera_estimates:
            camera_file = camera_estimates / f"{row['source_scene_name']}.json"
            camera = json.loads(camera_file.read_text())
            config["source_rgb_correspondence"].update(
                camera_estimate_status=camera["status"],
                camera_estimate_sha256=sha256(camera_file),
                camera_estimate_source_id=camera["source_id"],
            )
            if camera["status"] == "candidate":
                config["external_camera_estimate"] = camera["estimate"]
            else:
                config["source_rgb_correspondence"]["camera_estimate_failure"] = camera["reason"]
        if grasp_assistance:
            config["grasp_assistance"] = {
                "method": "source_action_fixed_joint",
                "authorized_by_user": True,
                "authorization_date_local": "2026-10-11",
                "maximum_tcp_distance_m": 0.075,
                "wait_frames": 8,
                "cycles": [
                    {"close": c["close"], "open": c["open"], "object": "cube"}
                    for c in cycles
                    if c["close"] >= anchors[0] and c["open"] <= anchors[1]
                ],
                "association_basis": "selected source close/release cycle; explicit scene object",
                "per_episode_rgb_association_verified": False,
                "scope": "offline reconstruction only; excluded from frozen policy benchmark",
            }
            if settled_approach:
                config = settled_approach_layout(config, tcp)
        config_path = output / "configs" / f"{row['id']}.json"
        write_json(config_path, config)
        selected.append({"id": row["id"], "config": str(config_path), "source": row})
    result = {
        "source_episodes": summary["episodes"],
        "source_frames": summary["frames"],
        "source_inventory_sha256": summary["source_manifest_sha256"],
        "ready_for_trial": selected,
        "scene_profiles_pending": pending,
        "full_dataset_complete": False,
        "publish_ready": False,
        "prepared_at_utc": utc(),
        "profile": profile,
    }
    write_json(output / "plan.json", result)
    return result


def reusable_replays(plan, previous):
    rows = {row["id"]: row for row in effective_status(previous)["completed"]}
    result = {}
    for trial in plan["ready_for_trial"]:
        old = rows.get(trial["id"])
        if not old or old["infrastructure_error"] or old["result"] is None:
            continue
        replay = previous / "runs" / trial["id"] / "replay"
        original = json.loads((replay / "config.json").read_text())
        requested = json.loads(Path(trial["config"]).read_text())
        ignore = {"source_episode", "asset"}
        if {k: v for k, v in original.items() if k not in ignore} != {
            k: v for k, v in requested.items() if k not in ignore
        }:
            continue
        with (
            np.load(Path(original["source_episode"]) / "trajectory.npz") as a,
            np.load(Path(requested["source_episode"]) / "trajectory.npz") as b,
        ):
            if not all(np.array_equal(a[key], b[key]) for key in ("actions", "states")):
                raise ValueError("Source arrays changed between supposedly identical replays")
        for metadata in json.loads((replay / "files.json").read_text())["files"]:
            if sha256(replay / metadata["path"]) != metadata["sha256"]:
                raise ValueError("Old native capture cannot be reused after checksum change")
        result[trial["id"]] = {
            "replay": str(replay),
            "result": old["result"],
            "config_sha256": sha256(replay / "config.json"),
        }
    return result


def run(root, source, benchmark_pid, reuse_batch=None):
    import psutil

    batch = json.loads((root / "plan.json").read_text())
    reuse = reusable_replays(batch, reuse_batch) if reuse_batch else {}
    write_json(
        root / "reuse-v1.json",
        {
            "previous_batch": str(reuse_batch) if reuse_batch else None,
            "basis": "identical config except source/asset paths; exact source arrays; file SHA",
            "replays": reuse,
        },
    )
    (root / "runs").mkdir(exist_ok=False)
    status = {
        "status": "running",
        "stage": "drain_benchmark_episode",
        "started_at_utc": utc(),
        "completed": [],
    }
    benchmark = psutil.Process(benchmark_pid)
    if "owac.runtime.arena_droid_benchmark" not in " ".join(benchmark.cmdline()):
        raise ValueError("Cannot pause unrelated process")
    created = benchmark.create_time()
    child, paused = None, False

    def save(**updates):
        status.update(updates, checked_at_utc=utc())
        write_json(root / "status.json", status)

    def interrupted(signum, frame):
        raise InterruptedError(f"Batch received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        os.kill(benchmark_pid, signal.SIGSTOP)
        paused = True
        save(benchmark_pid=benchmark_pid, paused_at_utc=utc())
        deadline = time.monotonic() + 600
        while descendants(benchmark_pid):
            if time.monotonic() >= deadline:
                raise TimeoutError("Current benchmark episode did not drain")
            time.sleep(2)
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if gpu:
            raise RuntimeError(f"GPU is not free after benchmark drain: {gpu}")
        for trial in batch["ready_for_trial"]:
            episode = root / "runs" / trial["id"]
            episode.mkdir()
            if trial["id"] in reuse:
                old = reuse[trial["id"]]
                (episode / "replay").symlink_to(old["replay"], target_is_directory=True)
                status["completed"].append(
                    {
                        "id": trial["id"],
                        "exit_code": 0,
                        "result": old["result"],
                        "infrastructure_error": False,
                        "reused_verified_replay": old["replay"],
                        "new_physical_replay_executed": False,
                    }
                )
                save()
                continue
            save(stage="native_capture_replay", active_source_id=trial["id"])
            command = [
                str(RUNTIME / ".venv/bin/python"),
                "-u",
                "-m",
                "owac.simulation.polaris_replay",
                "--config",
                trial["config"],
                "--output",
                str(episode / "replay"),
                "--viz",
                "none",
                "--device",
                "cuda:0",
                "--experience",
                str(
                    RUNTIME
                    / ".venv/lib/python3.12/site-packages/isaacsim/apps"
                    / "isaacsim.exp.base.python.kit"
                ),
            ]
            write_json(episode / "command.json", {"argv": command, "source_snapshot": str(source)})
            with (episode / "simulator.log").open("w") as log:
                child = subprocess.Popen(
                    command,
                    cwd=RUNTIME,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ
                    | {
                        "PYTHONPATH": str(source),
                        "OMNI_KIT_ALLOW_ROOT": "1",
                        "OMNI_KIT_ACCEPT_EULA": "YES",
                        "ACCEPT_EULA": "Y",
                        "PYTHONUNBUFFERED": "1",
                        "HEADLESS": "1",
                    },
                )
                code = child.wait(timeout=900)
                child = None
            result_path = episode / "replay/result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            status["completed"].append(
                {
                    "id": trial["id"],
                    "exit_code": code,
                    "result": result,
                    "infrastructure_error": code != 0 or result is None,
                }
            )
            save()
        save(
            status="complete",
            stage="first_scene_batch_complete",
            full_dataset_complete=False,
            scene_profiles_pending=len(batch["scene_profiles_pending"]),
        )
    except BaseException as error:
        save(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if paused:
            current = psutil.Process(benchmark_pid)
            if current.create_time() != created:
                raise RuntimeError("Benchmark PID was reused; manual resumption needed")
            os.kill(benchmark_pid, signal.SIGCONT)
            save(benchmark_resumed_at_utc=utc())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "queue"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--benchmark-pid", type=int)
    parser.add_argument("--profile", choices=["cube", "rotate", "marker"], default="cube")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reuse-batch", type=Path)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--initial-grasp", choices=["carry", "first_close"], default="carry")
    parser.add_argument("--queue-manifest", type=Path)
    parser.add_argument("--camera-estimates", type=Path)
    parser.add_argument("--tcp-offset-m", type=float, default=0.14)
    parser.add_argument("--grasp-assistance", action="store_true")
    parser.add_argument("--settled-approach-layout", action="store_true")
    parser.add_argument(
        "--replay-mode", choices=["joint_targets", "reference_arm_states"], default="joint_targets"
    )
    args = parser.parse_args()
    if args.command == "plan":
        plan(
            args.inventory,
            args.root,
            args.asset,
            args.profile,
            args.limit,
            set(args.source_id) if args.source_id else None,
            args.initial_grasp,
            args.replay_mode,
            args.camera_estimates,
            args.tcp_offset_m,
            args.grasp_assistance,
            args.settled_approach_layout,
        )
    elif args.command == "run":
        run(args.root, args.source, args.benchmark_pid, args.reuse_batch)
    else:
        queue = json.loads(args.queue_manifest.read_text())
        progress = args.queue_manifest.with_name("queue-status.json")
        state = {"status": "running", "started_at_utc": utc(), "completed_batches": []}
        try:
            for dependency in queue.get("dependencies", []):
                dependency = Path(dependency)
                while True:
                    prior = json.loads(dependency.read_text()) if dependency.exists() else {}
                    if prior.get("status") == "complete":
                        break
                    if prior.get("status") == "failed":
                        raise RuntimeError(f"Prior GPU queue failed: {dependency}")
                    state.update(
                        stage="waiting_for_prior_gpu_queue",
                        dependency=str(dependency),
                        updated_at_utc=utc(),
                    )
                    write_json(progress, state)
                    time.sleep(10)
            for entry in queue["batches"]:
                root = Path(entry["root"])
                state.update(stage="replaying_batch", active_batch=str(root), updated_at_utc=utc())
                write_json(progress, state)
                run(
                    root,
                    args.source,
                    args.benchmark_pid,
                    Path(entry["reuse_batch"]) if entry.get("reuse_batch") else None,
                )
                state["completed_batches"].append(str(root))
            state.update(status="complete", completed_at_utc=utc(), full_dataset_complete=False)
            write_json(progress, state)
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(progress, state)
            raise


if __name__ == "__main__":
    main()

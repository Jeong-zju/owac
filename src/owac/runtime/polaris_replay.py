"""Plan and supervise an explicitly reconstructed PolaRiS expert replay pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json


def transform(position, quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = position
    return matrix


def joint_transform(anchor0, anchor1, axis, angle):
    quaternion = [math.cos(angle / 2), 0, 0, 0]
    quaternion["XYZ".index(axis) + 1] = math.sin(angle / 2)
    return anchor0 @ transform([0, 0, 0], quaternion) @ np.linalg.inv(anchor1)


def robot_chain(asset):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(asset))
    parent = "/panda/panda_link0"
    root = np.asarray(
        UsdGeom.Xformable(stage.GetPrimAtPath(parent)).ComputeLocalToWorldTransform(0)
    ).T
    chain = []
    for name in [*(f"panda_joint{i}" for i in range(1, 9)), "panda_hand_joint"]:
        prim = next(p for p in stage.Traverse() if p.GetName() == name)
        joint = UsdPhysics.Joint(prim)
        if str(joint.GetBody0Rel().GetTargets()[0]) != parent:
            raise ValueError(f"Unexpected joint chain at {name}")
        parent = str(joint.GetBody1Rel().GetTargets()[0])
        anchors = []
        for side in (0, 1):
            pos = prim.GetAttribute(f"physics:localPos{side}").Get()
            quat = prim.GetAttribute(f"physics:localRot{side}").Get()
            anchors.append(transform(pos, [quat.GetReal(), *quat.GetImaginary()]))
        axis = prim.GetAttribute("physics:axis").Get() or "X"
        chain.append((anchors[0], anchors[1], axis))
    return root, chain


def forward(root, chain, q, tcp_offset=(0.14, 0, 0)):
    pose = root.copy()
    for index, (a0, a1, axis) in enumerate(chain):
        pose = pose @ joint_transform(a0, a1, axis, float(q[index]) if index < 7 else 0)
    tcp = pose @ np.r_[tcp_offset, 1]
    return pose, tcp[:3]


def prepare(root, episode=None, carry_cycle=None, tcp_offset=(0.14, 0, 0)):
    asset = root / "assets/franka_robotiq_2f_85_flattened.usd"
    episode = episode or root / "episodes/episode-009"
    with np.load(episode / "trajectory.npz") as data:
        actions, states = data["actions"], data["states"]
    initial, chain = robot_chain(asset)
    tcp = np.asarray([forward(initial, chain, q[:7], tcp_offset)[1] for q in states])
    closing = np.flatnonzero((actions[:, 7] > 0.5) & (np.r_[0, actions[:-1, 7]] <= 0.5))
    if not len(closing):
        raise ValueError("No original close transition")
    close = int(closing[0])
    opening = np.flatnonzero((actions[:, 7] <= 0.5) & (np.r_[0, actions[:-1, 7]] > 0.5))
    release = int(next(i for i in opening if i > close))
    if carry_cycle is not None:
        close, release = carry_cycle
        if close not in closing or release not in opening or release <= close:
            raise ValueError("Reconstruction anchors must be original close/open transitions")
    table_top = float(tcp[close, 2] - 0.02)
    config = {
        "experiment": "polaris-expert-replay-pilot-v1",
        "scope": "reconstructed scene; original scene and camera alignment are unknown",
        "source_episode": str(episode),
        "source": json.loads((episode / "source.json").read_text()),
        "asset": str(asset),
        "asset_sha256": sha256(asset),
        "physics_dt": 1 / 120,
        "decimation": 8,
        "frames": len(actions),
        "source_action_modified": False,
        "scene_uses_reference_future_for_offline_reconstruction": True,
        "source_close_frame": close,
        "source_release_frame": release,
        "tcp_offset": list(tcp_offset),
        "initial_base_pose": initial.tolist(),
        "cube_size": 0.04,
        "cube_mass": 0.04,
        "cube_initial_position": [*tcp[close, :2], table_top + 0.021],
        "target_center": [*tcp[release, :2], table_top],
        "bin_inner_half_extent": 0.11,
        "bin_wall_height": 0.055,
        "table_top": table_top,
        "camera_resolution": [320, 180],
        "settle_seconds": 2,
        "success": {
            "lift_minimum_m": 0.05,
            "transport_minimum_m": 0.10,
            "target_margin_m": 0.02,
            "released_tcp_separation_m": 0.05,
            "stable_last_frames": 15,
            "stable_maximum_speed_m_s": 0.05,
        },
        "cpu_fk": {
            "initial_tcp": tcp[0].tolist(),
            "close_tcp": tcp[close].tolist(),
            "release_tcp": tcp[release].tolist(),
            "minimum_tcp_z": float(tcp[:, 2].min()),
            "maximum_tcp_z": float(tcp[:, 2].max()),
            "zero_joint_base": forward(initial, chain, np.zeros(7), tcp_offset)[0].tolist(),
        },
    }
    write_json(root / "config.json", config)
    return config


def utc():
    return datetime.now(UTC).isoformat()


def descendants(pid):
    import psutil

    return [p for p in psutil.Process(pid).children(recursive=True) if p.status() != "zombie"]


def supervise(root, attempt, benchmark_pid):
    """Stop only the dispatcher; its current episode finishes before the pilot starts."""
    import psutil

    output = root / "attempts" / attempt
    output.mkdir(parents=True, exist_ok=False)
    record = {"attempt": attempt, "started_utc": utc(), "benchmark_pid": benchmark_pid}
    snapshot = root / f"deployment-{attempt}"
    shutil.copytree(root / "source", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(root / "config.json", output / "launch-config.json")
    record["source_snapshot"] = str(snapshot)
    record["source_files_sha256"] = {
        str(path.relative_to(snapshot)): sha256(path) for path in snapshot.rglob("*.py")
    }
    paused = False
    child = None

    def terminate(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        benchmark = psutil.Process(benchmark_pid)
        if "owac.runtime.arena_droid_benchmark" not in " ".join(benchmark.cmdline()):
            raise RuntimeError("Refusing to pause an unrelated process")
        record["benchmark_create_time"] = benchmark.create_time()
        os.kill(benchmark_pid, signal.SIGSTOP)
        paused = True
        record["paused_dispatcher_utc"] = utc()
        write_json(output / "coordination.json", record)
        deadline = time.monotonic() + 600
        while descendants(benchmark_pid):
            if time.monotonic() > deadline:
                raise TimeoutError("Current benchmark episode did not drain")
            time.sleep(2)
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if gpu:
            raise RuntimeError(f"GPU still has active processes: {gpu}")
        record["episode_drained_utc"] = utc()
        write_json(output / "coordination.json", record)
        runtime = Path("/workspace/owac/reference/pi05-arena-droid-v1/Arena")
        cmd = [
            str(runtime / ".venv/bin/python"),
            "-m",
            "owac.simulation.polaris_replay",
            "--config",
            str(output / "launch-config.json"),
            "--output",
            str(output / "replay"),
            "--viz",
            "none",
            "--device",
            "cuda:0",
            "--experience",
            str(
                runtime
                / ".venv/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
            ),
        ]
        record["command"] = cmd
        write_json(output / "coordination.json", record)
        with (output / "simulator.log").open("w") as log:
            environment = os.environ | {
                "PYTHONPATH": str(snapshot),
                "OMNI_KIT_ALLOW_ROOT": "1",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "ACCEPT_EULA": "Y",
                "PYTHONUNBUFFERED": "1",
                "HEADLESS": "1",
            }
            child = subprocess.Popen(
                cmd, cwd=runtime, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            record["replay_pid"] = child.pid
            write_json(output / "coordination.json", record)
            record["exit_code"] = child.wait(timeout=900)
            record["replay_result_written"] = (output / "replay/result.json").exists()
            if record["exit_code"] != 0 or not record["replay_result_written"]:
                raise RuntimeError("Simulator did not write a completed physical replay result")
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        if paused:
            current = psutil.Process(benchmark_pid)
            if current.create_time() != record["benchmark_create_time"]:
                raise RuntimeError("Dispatcher PID identity changed; cannot safely resume")
            os.kill(benchmark_pid, signal.SIGCONT)
            record["resumed_dispatcher_utc"] = utc()
        record["finished_utc"] = utc()
        write_json(output / "coordination.json", record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "supervise"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", default="attempt-001")
    parser.add_argument("--benchmark-pid", type=int)
    args = parser.parse_args()
    result = (
        prepare(args.root)
        if args.mode == "prepare"
        else supervise(args.root, args.attempt, args.benchmark_pid)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

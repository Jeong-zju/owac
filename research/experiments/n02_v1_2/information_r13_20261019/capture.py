"""Isolated, labeled static simulator snapshots; never a policy evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
SOURCE = Path("/workspace/owac/rawvla/source")
OUT = ROOT / "outputs/n02-v12/information-r13"
CAMS = ("head_camera", "left_camera", "right_camera")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def selection():
    files = sorted(
        (SOURCE / "benchmark_data/RawVLA-Bench/data/robotwin2/raw/grab_roller").glob("*.npz")
    )
    files = [p for p in files if p.name != "episode_000000.npz"]
    rng = np.random.default_rng(13019)
    order = rng.permutation(len(files))
    # Freeze diagnostic grouping independently of feature values and target performance.
    files = [files[i] for i in order]
    if len(files) < 48:
        raise ValueError("48 independent source episodes required")
    return [
        dict(
            scene=i,
            split="train" if i < 32 else "val" if i < 40 else "test",
            path=str(p),
            seed=713019 + i,
        )
        for i, p in enumerate(files[:48])
    ]


def set_snapshot(robot, action, openness):
    """Place actual articulation joints, including mimic fingers, without physics steps."""
    for side, start, opening in (("left", 0, openness[0]), ("right", 7, openness[1])):
        entity = getattr(robot, side + "_entity")
        active = entity.get_active_joints()
        q = entity.get_qpos().copy()
        for joint, value in zip(
            getattr(robot, side + "_arm_joints"), action[start : start + 6], strict=True
        ):
            q[active.index(joint)] = value
            joint.set_drive_target(float(value))
        low, high = getattr(robot, side + "_gripper_scale")
        g = low + float(opening) * (high - low)
        for joint, mult, bias in getattr(robot, side + "_gripper"):
            value = g * mult + bias
            q[active.index(joint)] = value
            joint.set_drive_target(float(value))
        entity.set_qpos(q)
        entity.set_qvel(np.zeros_like(q))
        setattr(robot, side + "_gripper_val", float(opening))
        if np.max(np.abs(entity.get_qpos() - q)) > 1e-6:
            raise ValueError("articulation snapshot mismatch")


def capture(args):
    import sapien
    from envs.lighting import apply_lighting
    from replay_robotwin2_clean_training import build_robotwin_args
    from script.collect_data import class_decorator

    original = sapien.render.set_ray_tracing_denoiser
    sapien.render.set_ray_tracing_denoiser = lambda name: original(
        "optix" if name == "oidn" else name
    )
    OUT.mkdir(parents=True, exist_ok=True)
    plan = selection()
    atomic_json(OUT / "selection.json", plan)
    for row in plan[: args.scenes]:
        destination = OUT / "snapshots" / f"scene_{row['scene']:03d}.npz"
        if destination.exists():
            continue
        rng = np.random.default_rng(row["seed"])
        with np.load(row["path"], allow_pickle=False) as a:
            actions = a["actions"]
        config = build_robotwin_args(
            SOURCE / "third_party/RoboTwin", OUT / "runtime", "grab_roller"
        )
        config.update(
            now_ep_num=row["scene"],
            seed=row["seed"],
            save_data=False,
            hdr_raw_white_level=3.5,
            raw_sensor_noise_enabled=True,
            raw_sensor_noise_seed=row["seed"],
            raw_sensor_shot_noise=2.5e-5,
            raw_sensor_read_noise=3.90625e-8,
        )
        task = class_decorator("grab_roller")
        task.setup_demo(**config)
        apply_lighting(task, "Normal", "environment_only", sample_unit=0.5)
        images, rgb, labels, states, records = [], [], [], [], []
        for frame in range(8):
            t = int(round(frame * (len(actions) - 1) / 7))
            # Balanced endpoints are independently randomized, not taken from the task phase.
            opening = rng.integers(0, 2, size=2).astype(float)
            set_snapshot(task.robot, actions[t], opening)
            pose = task.roller.get_pose()
            pose.p = np.array(
                [rng.uniform(-0.15, 0.15), rng.uniform(-0.25, -0.05), rng.uniform(0.77, 0.87)]
            )
            task.roller.actor.set_pose(pose)
            task._update_render()
            obs = task.get_obs()
            obj = np.asarray(task.roller.get_pose().p)
            # TCP coordinates, object coordinates and differences all use world metres.
            left = np.asarray(task.robot.get_left_tcp_pose()[:3])
            right = np.asarray(task.robot.get_right_tcp_pose()[:3])
            labels.append(np.concatenate([obj, left, right, obj - left, obj - right, opening]))
            states.append(np.asarray(obs["joint_action"]["vector"]))
            images.append(np.stack([obs["observation"][c]["raw_linear"] for c in CAMS]))
            rgb.append(np.stack([obs["observation"][c]["render_rgb"] for c in CAMS]))
            records.append(
                dict(
                    **row,
                    frame=frame,
                    source_t=t,
                    model_id=int(task.model_id),
                    roller_quaternion=np.asarray(pose.q).tolist(),
                )
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                raw=np.asarray(images, dtype=np.float32),
                rgb=np.asarray(rgb, dtype=np.uint8),
                y=np.asarray(labels, dtype=np.float32),
                state=np.asarray(states, dtype=np.float32),
                records=json.dumps(records),
            )
        temporary.replace(destination)
        task.close_env()
        print(
            json.dumps(dict(scene=row["scene"], split=row["split"], samples=len(labels))),
            flush=True,
        )
        atomic_json(OUT / "capture_progress.json", dict(completed_scenes=row["scene"] + 1))
    files = sorted((OUT / "snapshots").glob("scene_*.npz"))
    atomic_json(
        OUT / "capture_receipt.json",
        dict(
            diagnostic_only=True,
            policy_updates=0,
            action_execution=0,
            scenes=len(files),
            frames=8 * len(files),
            coordinate_frame="world, metres; gripper position = TCP, not cached EE convention",
            scope=(
                "Controlled static snapshots; object can float/intersect robot. "
                "No physical rollout claim."
            ),
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=48)
    args = parser.parse_args()
    os.chdir(SOURCE / "third_party/RoboTwin")
    capture(args)

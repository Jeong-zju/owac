"""S0 live paired extensions. Frozen benchmark inputs are only read.

Host sample/arrival times and physics seconds are separate clocks. Every third
frame is held by the actual delivery queue until after the following control;
the release timestamp is measured, not invented from an action horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def joint_control_layout(entities) -> list[dict]:
    """Flattening order is [position, velocity] for every read-back joint."""
    rows = []
    for slot, entity in enumerate(entities):
        for joint in entity.get_active_joints():
            kind = str(joint.type)
            if kind == "prismatic":
                position_unit, velocity_unit = "m", "m/s"
            elif kind in ("revolute", "revolute_unwrapped"):
                position_unit, velocity_unit = "rad", "rad/s"
            else:
                raise ValueError(f"unsupported drive joint type: {kind}")
            rows.append(
                dict(
                    articulation_slot=slot,
                    name=joint.name,
                    joint_type=kind,
                    position_unit=position_unit,
                    velocity_unit=velocity_unit,
                    value_indices=[2 * len(rows), 2 * len(rows) + 1],
                )
            )
    return rows


class LiberoCapture:
    def __init__(self, source: Path, seed: int):
        from owac.raw.n04_d1_capture import configure_renderer
        from owac.raw.n04_mechanism_data import _libero_env_from_npz

        self.env, metadata, bddl = _libero_env_from_npz(source, env_seed=seed)
        self.seed = seed
        with np.load(source, allow_pickle=False) as data:
            initial = data["states"][0].copy()
            self.actions = data["actions"].copy()
        self.env.reset()
        configure_renderer(self.env)
        self.env.set_init_state(initial)
        self.meta = dict(
            platform="libero",
            source_npz=str(source),
            source_sha256=sha256(source),
            task=metadata["task_name"],
            bddl=str(bddl),
            seed=seed,
            command_semantics="clipped normalized OSC env.step argument; not torque",
            camera_names=["agentview", "robot0_eye_in_hand"],
            pixel_coordinates="integer_pixel_centres_top_left",
            control_mode="osc_delta_eef",
            control_units="normalized",
            geometry_audit="independent CPU mj_ray against GL depth",
            representation="RGB-derived rawvla_rgb3; frozen author producer",
        )

    def state(self):
        # mj_step integrates qpos after its forward pass; refresh derived geometry
        # so camera/body transforms and the recorded current qpos share one state.
        self.env.sim.forward()
        return self.env.sim.get_state().flatten().copy()

    def capture(self, index):
        from owac.raw.n04_d1_capture import capture_frame

        result = capture_frame(self.env, frame_index=index, noise_seed=self.seed)
        result["gt_contact_count"] = np.asarray(self.env.sim.data.ncon)
        return result

    def step(self, index):
        command = np.clip(self.actions[index], -1.0, 1.0).astype(np.float64)
        start = time.monotonic()
        sim_start = float(self.env.sim.data.time)
        self.env.step(command.tolist())
        return dict(
            value=command.tolist(),
            start=start,
            end=time.monotonic(),
            sim_start=sim_start,
            sim_end=float(self.env.sim.data.time),
        )

    def close(self):
        self.env.close()


class RobotwinCapture:
    def __init__(self, task: str, seed: int, output: Path):
        root = Path("/workspace/owac/rawvla/source/third_party/RoboTwin")
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / "script"), str(root.parents[1] / "scripts/replay")]
        from replay_robotwin2_clean_training import build_robotwin_args
        from script.collect_data import class_decorator

        args = build_robotwin_args(root, output, task)
        args.update(
            now_ep_num=0,
            seed=seed,
            save_data=False,
            hdr_raw_white_level=1.0,
            raw_sensor_noise_enabled=False,
        )
        # Read integer IDs directly; upstream palette uint8 conversion loses identity.
        args["data_type"].update(depth=True, mesh_segmentation=False, actor_segmentation=False)
        self.env = class_decorator(task)
        original_load_camera = self.env.load_camera

        def load_calibrated_cameras(**kwargs):
            import sapien

            # Preserve HDR outputs while putting all primary rays at the declared
            # pixel centre. Copy shaders into this run; installed assets stay read-only.
            shader_source = Path(sapien.__file__).parent / "vulkan_shader"
            shaders = output.parent / "calibrated-rt-shaders"
            original = (shader_source / "rt/camera.rgen").read_text()
            target = "vec2 r = halton2d(frameCount * spp + i, 2, 3);"
            if original.count(target) != 1:
                raise ValueError("unexpected frozen SAPIEN primary-ray shader")
            patched = original.replace(target, "vec2 r = vec2(0.5, 0.5);")
            if not shaders.exists():
                shutil.copytree(shader_source, shaders)
                (shaders / "rt/camera.rgen").write_text(patched)
            if (shaders / "rt/camera.rgen").read_text() != patched:
                raise ValueError("run shader contents changed")
            sapien.render.set_camera_shader_dir(str(shaders / "rt"))
            sapien.render.set_ray_tracing_denoiser("none")
            self.shader_provenance = dict(
                original_sha256=sha256(shader_source / "rt/camera.rgen"),
                collected_sha256=sha256(shaders / "rt/camera.rgen"),
                path=str(shaders / "rt/camera.rgen"),
                change="primary rays at pixel centre; 32 spp shading; no denoiser",
            )
            original_load_camera(**kwargs)

        self.env.load_camera = load_calibrated_cameras
        self.env.setup_demo(**args)
        self.clock = 0.0
        self.cameras = [
            ("left_camera", self.env.cameras.left_camera),
            ("right_camera", self.env.cameras.right_camera),
        ] + list(
            zip(
                self.env.cameras.static_camera_name,
                self.env.cameras.static_camera_list,
                strict=True,
            )
        )
        robot = self.env.robot
        self.initial = [
            np.asarray(robot.get_left_arm_jointState()[:-1]),
            np.asarray(robot.get_right_arm_jointState()[:-1]),
        ]
        self.meta = dict(
            platform="robotwin2",
            task=task,
            seed=seed,
            source_npz=None,
            camera_names=[name for name, _ in self.cameras],
            pixel_coordinates="integer_pixel_centres_top_left",
            command_semantics=(
                "read back every active joint drive position and velocity AFTER "
                "setter; includes gripper limits"
            ),
            control_mode="joint_drive_target",
            control_units="per command_layout: revolute rad/rad_s; prismatic m/m_s",
            command_layout=joint_control_layout([robot.left_entity, robot.right_entity]),
            geometry_audit=(
                "Position buffer independently projected using K; extrinsic vs cam2world"
            ),
            representation="renderer HDR-derived raw_linear; three-channel pseudo-RAW",
            renderer=self.shader_provenance,
            intervention=(
                "small bounded joint motion, stop, reverse and gripper change; no "
                "task-success label"
            ),
        )

    def state(self):
        robot = self.env.robot
        return np.concatenate(
            [
                robot.left_entity.get_qpos(),
                robot.left_entity.get_qvel(),
                robot.right_entity.get_qpos(),
                robot.right_entity.get_qvel(),
            ]
        )

    def capture(self, index):
        obs = self.env.get_obs()
        results = []
        for name, camera in self.cameras:
            item = obs["observation"][name]
            depth = np.asarray(item["depth"]) / 1000.0
            position = np.asarray(camera.get_picture("Position")).copy()
            seg = np.asarray(camera.get_picture("Segmentation")).copy()
            k = np.array(item["intrinsic_cv"], copy=True)
            k[:2, 2] -= 0.5  # SAPIEN edge coordinates -> integer pixel centres
            t = np.eye(4)
            t[:3] = item["extrinsic_cv"]
            pose = np.linalg.inv(t)
            # OpenGL position uses x right, y up, z backward; CV uses y down,z forward.
            cv = position[..., :3] * np.array([1.0, -1.0, -1.0])
            valid = (depth > 1e-5) & (position[..., 3] < 1)
            vv, uu = np.indices(depth.shape)
            coords = cv @ k.T
            denom = np.where(valid, coords[..., 2], 1.0)
            errors = np.maximum(abs(coords[..., 0] / denom - uu), abs(coords[..., 1] / denom - vv))
            # Saved sampled XYZ enables a separate validator to repeat the projection.
            sample = valid & ((uu % 32) == 16) & ((vv % 32) == 16)
            results.append(
                dict(
                    online_rawvla_rgb3=item["raw_linear"],
                    reference_rgb=item["rgb"],
                    online_K=k,
                    gt_T_W_from_C=pose,
                    gt_depth_m=depth,
                    gt_segmentation=seg[..., :2].astype(np.int64),
                    gt_visible_support=valid,
                    audit_projection_max_px=np.asarray(np.max(errors[valid], initial=0)),
                    audit_extrinsic_error=np.asarray(
                        np.max(abs(pose - item["cam2world_gl"] @ np.diag([1, -1, -1, 1])))
                    ),
                    audit_xyz_C=cv[::32, ::32],
                    audit_depth_error_m=np.asarray(
                        np.max(abs(cv[..., 2][valid] - depth[valid]), initial=0)
                    ),
                    audit_valid_samples=np.asarray(sample.sum()),
                )
            )
        frame = {key: np.stack([r[key] for r in results]) for key in results[0]}
        frame["online_sim_time_s"] = np.asarray(self.clock)
        frame["online_proprio"] = self.state()
        entities = (
            self.env.scene.get_all_actors()
            if hasattr(self.env.scene, "get_all_actors")
            else self.env.scene.entities
        )
        frame["gt_entity_ids"] = np.asarray([e.per_scene_id for e in entities], dtype=np.int64)
        frame["gt_entity_pose"] = np.asarray(
            [np.r_[e.get_pose().p, e.get_pose().q] for e in entities]
        )
        contacts = self.env.scene.get_contacts()
        frame["gt_contact_count"] = np.asarray(len(contacts))
        frame["gt_contact_points"] = np.asarray(
            [
                np.r_[p.position, p.normal, p.impulse, p.separation]
                for c in contacts
                for p in c.points
            ]
        ).reshape(-1, 10)
        return frame

    def step(self, index):
        # Controlled action intervention, distinct from expert or policy supervision.
        amplitude = [
            0.0,
            0.025,
            0.05,
            0.075,
            0.075,
            0.075,
            0.05,
            0.025,
            0.0,
            -0.025,
            -0.05,
            -0.075,
        ][index % 12]
        start = time.monotonic()
        robot = self.env.robot
        for side, initial in zip(("left", "right"), self.initial, strict=True):
            target = initial.copy()
            target[0] += amplitude * (1 if side == "left" else -1)
            robot.set_arm_joints(target, np.zeros_like(target), side)
            robot.set_gripper(0.8 if index % 12 < 6 else 0.2, side)
        # Record actual drive readback, not the planned normalized gripper command.
        joints = robot.left_entity.get_active_joints() + robot.right_entity.get_active_joints()
        command = np.concatenate(
            [np.r_[j.get_drive_target(), j.get_drive_velocity_target()] for j in joints]
        )
        sim_start = self.clock
        for _ in range(10):
            self.env.scene.step()
            self.clock += float(self.env.scene.get_timestep())
        return dict(
            value=command.tolist(),
            start=start,
            end=time.monotonic(),
            sim_start=sim_start,
            sim_end=self.clock,
        )

    def close(self):
        self.env.close_env()


def collect(adapter, output: Path, *, frames: int, source_id: str, split: str):
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="n04-live-extension-v2",
        source_trajectory_id=source_id,
        split=split,
        mode="independent_extension",
        source_correspondence_verified=False,
        native_cfa=False,
        metadata=adapter.meta,
        frames=[],
        controls=[],
        host_clock="monotonic_process",
        simulation_clock="seconds_since_initialization",
        time_semantics=(
            "instantaneous simulator state sampled at host_sample_s; simulator "
            "paused during rendering; host_arrival_s is actual queue release, "
            "not hardware exposure"
        ),
        factor_scope=(
            "renderer geometry, robot motion/self-occlusion, contact; controlled queue delay"
        ),
        supervision_policy=(
            "gt_* forbidden online; no inherited expert success or action supervision"
        ),
    )
    pending = []
    try:
        for index in range(frames):
            sample_time = time.monotonic()
            pre = adapter.state()
            arrays = adapter.capture(index)
            arrays["trace_pre_state"] = pre
            render_done = time.monotonic()
            control = adapter.step(index)
            arrays["trace_post_state"] = adapter.state()
            arrays["trace_issued_command"] = np.asarray(control["value"])
            path = output / f"frame-{index:04d}.npz"
            np.savez_compressed(path, **arrays)
            entry = dict(
                index=index,
                file=path.name,
                sha256=sha256(path),
                host_sample_s=sample_time,
                host_render_done_s=render_done,
                host_arrival_s=None,
                sim_time_s=float(arrays["online_sim_time_s"]),
                queue_delayed=index % 3 == 0,
            )
            record["frames"].append(entry)
            control.update(
                command_id=str(index), frame=index, source=path.name, evidence_level="executed_log"
            )
            record["controls"].append(control)
            # Deliver ordinary frames immediately, then earlier held frames. This
            # creates real out-of-order arrival, while retaining actual release times.
            if not entry["queue_delayed"]:
                entry["host_arrival_s"] = time.monotonic()
            for held in pending:
                held["host_arrival_s"] = time.monotonic()
            pending = [entry] if entry["queue_delayed"] else []
            save_json(output / "record.json", record)
        for held in pending:
            held["host_arrival_s"] = time.monotonic()
        save_json(output / "record.json", record)
    finally:
        adapter.close()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["libero", "robotwin2"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--split", choices=["train", "dev", "final"], required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    adapter = (
        LiberoCapture(args.source, args.seed)
        if args.platform == "libero"
        else RobotwinCapture(args.task, args.seed, args.output)
    )
    collect(adapter, args.output, frames=args.frames, source_id=args.source_id, split=args.split)


if __name__ == "__main__":
    main()

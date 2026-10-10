"""Execute original PolaRiS joint targets with physical objects and native CFA cameras."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_replay import forward, robot_chain, transform, utc


def tensor(value):
    """The pinned Isaac Lab buffers expose Torch through ProxyArray.torch."""
    return getattr(value, "torch", value)


def evaluate(rows, config):
    positions = np.asarray([r["cube_position"] for r in rows])
    speeds = np.asarray([r["cube_speed"] for r in rows])
    target = np.asarray(config["target_center"])
    size = config["cube_size"]
    rules = config["success"]
    tail = rows[-rules["stable_last_frames"] :]
    extent = config["bin_inner_half_extent"] - size / 2 - rules["target_margin_m"]
    inside = np.all(np.abs(positions[-len(tail) :, :2] - target[:2]) < extent)
    height = np.all(np.abs(positions[-len(tail) :, 2] - (target[2] + 0.01 + size / 2)) < 0.025)
    lifted = float(positions[:, 2].max() - positions[0, 2])
    transported = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    separation = float(np.linalg.norm(positions[-1] - np.asarray(rows[-1]["tcp_position"])))
    carried = sum(
        r["original_action"][7] > 0.5
        and r["cube_position"][2] > positions[0, 2] + rules["lift_minimum_m"]
        and np.linalg.norm(np.asarray(r["cube_position"]) - r["tcp_position"]) < 0.055
        for r in rows
    )
    checks = {
        "lifted": lifted >= rules["lift_minimum_m"],
        "transported": transported >= rules["transport_minimum_m"],
        "sustained_grasp": bool(carried >= 5),
        "inside_target_bin": bool(inside and height),
        "released": rows[-1]["original_action"][7] <= 0.5
        and rows[-1]["actual_gripper_joint"] < 0.15
        and separation >= rules["released_tcp_separation_m"],
        "stable": bool(np.max(speeds[-len(tail) :]) < rules["stable_maximum_speed_m_s"]),
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "maximum_lift_m": lifted,
        "lifted_grasp_frames": int(carried),
        "transport_distance_m": transported,
        "final_cube_position": positions[-1].tolist(),
        "final_tcp_separation_m": separation,
        "final_bin_xy_error_m": float(np.linalg.norm(positions[-1, :2] - target[:2])),
        "tail_maximum_speed_m_s": float(speeds[-len(tail) :].max()),
        "source_tracking_joint_mae_rad": float(
            np.mean([r["source_joint_mae_rad"] for r in rows if r["source_frame"] is not None])
        ),
    }


def build_scene(config):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from pxr import Gf

    with np.load(Path(config["source_episode"]) / "trajectory.npz") as saved:
        initial_state = saved["states"][0]
    initial_joints = {
        f"panda_joint{i + 1}": float(value) for i, value in enumerate(initial_state[:7])
    } | {
        "finger_joint": float(np.clip(initial_state[7], 0, 1)) * np.pi / 4,
        "right_outer.*": 0,
        "left_inner.*": 0,
        "right_inner.*": 0,
    }
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=2)
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=config["asset"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=initial_joints, joint_vel={".*": 0}),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87,
                velocity_limit_sim=2.175,
                stiffness=400,
                damping=80,
            ),
            "forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12,
                velocity_limit_sim=2.61,
                stiffness=400,
                damping=80,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=200,
                velocity_limit_sim=5,
                stiffness=None,
                damping=None,
            ),
        },
    )
    size = config["cube_size"]
    scene.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32, solver_velocity_iteration_count=8
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=config["cube_mass"]),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1, dynamic_friction=1, restitution=0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.03)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(config["cube_initial_position"])),
    )

    def box(name, position, shape, color):
        setattr(
            scene,
            name,
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.CuboidCfg(
                    size=shape,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=position),
            ),
        )

    top = config["table_top"]
    box("table", (0.55, 0, top - 0.02), (0.8, 1.4, 0.04), (0.22, 0.23, 0.25))
    box("robot_base", (0, 0, -0.025), (0.25, 0.25, 0.05), (0.12, 0.12, 0.12))
    half = config["bin_inner_half_extent"]
    wall = config["bin_wall_height"]
    target = config["target_center"]
    for prefix, x, y, color in [
        ("left", target[0], target[1], (0.10, 0.25, 0.65)),
        ("right", target[0] + 0.29, target[1], (0.20, 0.55, 0.20)),
    ]:
        box(f"{prefix}_floor", (x, y, top + 0.005), (2 * half, 2 * half, 0.01), color)
        for index, sign in enumerate((-1, 1)):
            box(
                f"{prefix}_wall_x{index}",
                (x + sign * (half + 0.005), y, top + wall / 2),
                (0.01, 2 * half + 0.02, wall),
                color,
            )
            box(
                f"{prefix}_wall_y{index}",
                (x, y + sign * (half + 0.005), top + wall / 2),
                (2 * half, 0.01, wall),
                color,
            )
    scene.light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000)
    )
    width, height = config["camera_resolution"]
    optics = sim_utils.PinholeCameraCfg(
        focal_length=2.8, horizontal_aperture=5.376, vertical_aperture=3.024
    )
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(1.15, -1.15, 1), Gf.Vec3d(0.47, 0, top), Gf.Vec3d(0, 0, 1)
    )
    rotation = view.GetInverse().ExtractRotationQuat()
    scene.external_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_camera",
        spawn=optics,
        width=width,
        height=height,
        update_period=0,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(1.15, -1.15, 1),
            rot=(*rotation.GetImaginary(), rotation.GetReal()),
            convention="opengl",
        ),
    )
    scene.wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_camera",
        spawn=optics,
        width=width,
        height=height,
        update_period=0,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(0.570, 0.576, -0.409, -0.420), convention="opengl"
        ),
    )
    return scene


def run(config, output, app):
    import torch
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from PIL import Image

    from owac.simulation.arena_raw import ArenaRawCapture
    from owac.simulation.raw_camera import RawCameraProfile

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    start = time.monotonic()
    sim = SimulationContext(
        SimulationCfg(dt=config["physics_dt"], render_interval=8, device="cuda:0")
    )
    scene = InteractiveScene(build_scene(config))
    sim.reset()
    scene.reset()
    robot, cube = scene["robot"], scene["cube"]
    joint_ids, names = robot.find_joints(
        [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True
    )
    grip_ids, _ = robot.find_joints("finger_joint")
    body_ids, _ = robot.find_bodies("base_link")
    if len(body_ids) != 1 or len(joint_ids) != 7 or len(grip_ids) != 1:
        raise RuntimeError("DROID asset does not match original action interface")
    data = np.load(Path(config["source_episode"]) / "trajectory.npz")
    actions, states = data["actions"], data["states"]
    q = tensor(robot.data.default_joint_pos).clone()
    q[:, joint_ids] = torch.as_tensor(states[0, :7], device=sim.device)
    q[:, grip_ids] = float(np.clip(states[0, 7], 0, 1)) * np.pi / 4
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(config["physics_dt"])
    root, chain = robot_chain(config["asset"])
    base_expected = forward(root, chain, states[0, :7])[0]

    def observe(source_frame, command):
        actual = tensor(robot.data.joint_pos)[0, joint_ids].cpu().numpy()
        body = tensor(robot.data.body_state_w)[0, body_ids[0]].cpu().numpy()
        x, y, z, w = body[3:7]
        tcp = transform(body[:3], [w, x, y, z]) @ np.r_[config["tcp_offset"], 1]
        return {
            "source_frame": source_frame,
            "original_action": command.tolist(),
            "joint_position": actual.tolist(),
            "all_joint_positions": tensor(robot.data.joint_pos)[0].cpu().numpy().tolist(),
            "all_joint_velocities": tensor(robot.data.joint_vel)[0].cpu().numpy().tolist(),
            "robot_root_state_xyzw": tensor(robot.data.root_state_w)[0].cpu().numpy().tolist(),
            "actual_gripper_joint": float(tensor(robot.data.joint_pos)[0, grip_ids[0]]),
            "tcp_position": tcp[:3].tolist(),
            "gripper_base_position": body[:3].tolist(),
            "gripper_base_quaternion_xyzw": body[3:7].tolist(),
            "cube_position": tensor(cube.data.root_pos_w)[0].cpu().numpy().tolist(),
            "cube_quaternion_xyzw": tensor(cube.data.root_quat_w)[0].cpu().numpy().tolist(),
            "cube_velocity": tensor(cube.data.root_lin_vel_w)[0].cpu().numpy().tolist(),
            "cube_root_state_xyzw": tensor(cube.data.root_state_w)[0].cpu().numpy().tolist(),
            "cube_speed": float(torch.linalg.vector_norm(tensor(cube.data.root_lin_vel_w)[0])),
            "source_joint_mae_rad": float(np.abs(actual - states[source_frame, :7]).mean())
            if source_frame is not None
            else None,
            "simulation_time_seconds": sim.get_physics_step_count() * config["physics_dt"],
        }

    initial = observe(0, actions[0])
    error = float(np.linalg.norm(base_expected[:3, 3] - initial["gripper_base_position"]))
    write_json(
        output / "initial.json", {**initial, "fk_position_error_m": error, "joint_names": names}
    )
    if error > 0.005:
        raise RuntimeError(f"CPU FK does not agree with Isaac DROID: {error} m")
    shim = SimpleNamespace(scene=scene, sim=sim, common_step_counter=0)
    shim.unwrapped = shim
    width, height = config["camera_resolution"]
    capture = ArenaRawCapture(shim, output, profile=RawCameraProfile(width=width, height=height))
    import omni.usd

    scene_path = output / "reconstructed_scene.usda"
    omni.usd.get_context().get_stage().Export(str(scene_path))
    write_json(output / "started.json", {"started_utc": utc(), "source_frames": len(actions)})
    rows = []
    all_files = []
    total = len(actions) + round(
        config["settle_seconds"] / (config["physics_dt"] * config["decimation"])
    )
    with (output / "frames.jsonl").open("w") as stream:
        for step in range(total):
            if not app.is_running():
                raise RuntimeError("Simulator exited during original expert replay")
            sim.forward()
            source_frame = step if step < len(actions) else None
            command = actions[min(step, len(actions) - 1)]
            row = observe(source_frame, command)
            if source_frame is not None:
                row["capture"] = capture.capture(step)
                for name, (_, camera, _) in capture.cameras.items():
                    rgb, _ = camera.sensor.get_data("rgb")
                    if rgb is None:
                        raise RuntimeError("Native paired RGB is unavailable")
                    rgb = rgb.numpy()[..., :3]
                    if (
                        rgb.dtype != np.uint8
                        or rgb.shape != (height, width, 3)
                        or int(rgb.max()) == 0
                    ):
                        raise RuntimeError(f"Invalid paired RGB: {rgb.shape} {rgb.dtype}")
                    path = output / "rgb" / name / f"{step:05d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(path)
                    row["capture"]["views"][name]["paired_rgb"] = {
                        "path": str(path.relative_to(output)),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "source": "same native render product and held physics state",
                    }
                    all_files.extend(row["capture"]["views"][name]["files"].values())
                    all_files.append(row["capture"]["views"][name]["paired_rgb"])
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            robot.set_joint_position_target(
                torch.as_tensor(command[:7][None], device=sim.device), joint_ids
            )
            robot.set_joint_position_target(
                torch.tensor(
                    [[np.pi / 4 if command[7] > 0.5 else 0.0]],
                    dtype=torch.float32,
                    device=sim.device,
                ),
                grip_ids,
            )
            for _ in range(config["decimation"]):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(config["physics_dt"])
            shim.common_step_counter += 1
            if step % 10 == 0:
                print("REPLAY_FRAME", step, json.dumps(row["cube_position"]), flush=True)
    mismatches = [r["path"] for r in all_files if sha256(output / r["path"]) != r["sha256"]]
    if mismatches:
        raise RuntimeError(f"Capture checksum mismatch: {mismatches}")
    original_rows = rows[: len(actions)]
    trajectory = output / "paired-trajectory.npz"
    np.savez_compressed(
        trajectory,
        original_actions=actions,
        original_robot_states=states,
        replay_joint_positions=np.asarray([r["all_joint_positions"] for r in original_rows]),
        replay_joint_velocities=np.asarray([r["all_joint_velocities"] for r in original_rows]),
        robot_root_states_xyzw=np.asarray([r["robot_root_state_xyzw"] for r in original_rows]),
        cube_root_states_xyzw=np.asarray([r["cube_root_state_xyzw"] for r in original_rows]),
        source_frame_indices=np.arange(len(actions)),
        sim_timestamps_s=np.asarray([r["simulation_time_seconds"] for r in original_rows]),
    )
    with np.load(trajectory) as saved:
        if not np.array_equal(saved["original_actions"], actions):
            raise RuntimeError("Paired dataset modified original actions")
    write_json(
        output / "dataset.json",
        {
            "format": "owac-paired-native-cfa-v1",
            "source": config["source"],
            "scene_scope": config["scope"],
            "frames": len(actions),
            "views": ["external_camera", "wrist_camera"],
            "trajectory": {"path": trajectory.name, "sha256": sha256(trajectory)},
            "frame_manifest": {"path": "frames.jsonl", "sha256": sha256(output / "frames.jsonl")},
            "sensor_metadata": [f"sensor-{name}.json" for name in capture.cameras],
            "rgb_origin": "new Isaac native camera render in the reconstructed replay scene",
            "source_rgb_preserved_separately": str(Path(config["source_episode"]) / "original_rgb"),
            "original_rgb_pixel_alignment_verified": False,
            "cfa": {"pattern": "GRBG", "dtype": "<u4", "shape": [height, width], "bits": 24},
            "hdr": {"dtype": "<f2", "shape": [height, width, 4]},
            "settle_frames_are_auxiliary_and_excluded_from_training": total - len(actions),
            "privileged_object_state_is_replay_audit_only": True,
            "all_joint_names": robot.joint_names,
            "scene": {"path": scene_path.name, "sha256": sha256(scene_path)},
        },
    )
    result = {
        **evaluate(rows, config),
        "started_utc": json.loads((output / "started.json").read_text())["started_utc"],
        "finished_utc": utc(),
        "wall_seconds": time.monotonic() - start,
        "source_frames": len(actions),
        "settle_frames": total - len(actions),
        "original_action_changed": False,
        "object_state_overrides_after_initialization": 0,
        "scope": config["scope"],
        "source": config["source"],
        "native_capture": capture.summary(),
        "verified_file_count": len(all_files),
        "checksum_mismatches": mismatches,
        "metadata_sha256": sha256(output / "frames.jsonl"),
        "config_sha256": sha256(output / "config.json"),
    }
    write_json(output / "result.json", result)
    write_json(output / "files.json", {"files": all_files, "verified_utc": utc()})
    print(json.dumps(result), flush=True)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args, enable_cameras=True)
    try:
        run(json.loads(args.config.read_text()), args.output, launcher.app)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "error.json", {"error": repr(error), "traceback": traceback.format_exc()}
        )
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()

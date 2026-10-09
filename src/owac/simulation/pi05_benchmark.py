"""Collect expert episodes or run learned-policy episodes in the deployed Isaac task.

Object truth is confined to the expert and evaluation metrics. The policy receives
only two RGB views, current proprioception and the task prompt.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from owac.evaluation.baseline_protocol import file_sha256, physical_success, write_json
from owac.simulation.lw_libero import build_environment, launch_simulator

OBJECTS = ("red_coffee_mug", "porcelain_mug", "white_yellow_mug", "plate_left", "plate_right")


class Benchmark:
    def __init__(self, env, protocol):
        import isaaclab.sim as sim
        import numpy as np
        from isaacsim.core.experimental.utils.app import enable_extension
        from pxr import Gf

        enable_extension("isaacsim.sensors.experimental.rtx")
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        self.env, self.protocol = env, protocol
        self.robot = env.scene["robot"]
        self.arm = env.action_manager.get_term("arm_action")
        self.task = env.cfg.isaaclab_arena_env.task
        root = self.robot.data.root_pos_w[0].cpu().numpy()
        pedestal = sim.CuboidCfg(
            size=(0.24, 0.16, 0.4), collision_props=sim.CollisionPropertiesCfg()
        )
        pedestal.func("/World/OWACRobotPedestal", pedestal, translation=(*root[:2], 0.2))
        center = np.mean([env.scene[k].data.root_pos_w[0].cpu().numpy() for k in OBJECTS], axis=0)
        geometry = protocol["camera_geometry"]
        self.external_target = center + geometry["external_target_offset_from_initial_center"]
        self.external_eye = center + geometry["external_eye_offset_from_initial_center"]
        self.cameras, self.sensors = [], []
        for name in ("External", "Wrist"):
            camera = RtxCamera(
                "/World/OWACBaseline" + name,
                tick_rate=0,
                attributes={
                    "focalLength": geometry[name.lower() + "_focal_length_mm"],
                    # USD's default near plane is 1 stage unit (1 m here),
                    # which clips the entire wrist working area.
                    "clippingRange": Gf.Vec2f(*geometry["clipping_range_m"]),
                },
            )
            sensor = CameraSensor(camera, resolution=(224, 224), annotators=["rgb"])
            self.cameras.append(camera)
            self.sensors.append(sensor)
        self.update_cameras()
        for _ in range(40):
            env.sim.render()

    def tcp_world(self):
        import isaaclab.utils.math as math

        p, q = self.arm._compute_frame_pose()
        return math.combine_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, p, q
        )

    def update_cameras(self):
        import isaaclab.utils.math as math
        import numpy as np
        import torch
        from pxr import Gf

        tcp, quat = self.tcp_world()
        geometry = self.protocol["camera_geometry"]
        offsets = torch.tensor(
            [geometry["wrist_eye_tcp_local_m"]], device=self.env.device, dtype=torch.float32
        )
        wrist_eye = (tcp + math.quat_apply(quat, offsets))[0].cpu().numpy()
        wrist_target = (
            (
                tcp
                + math.quat_apply(
                    quat,
                    torch.tensor(
                        [geometry["wrist_target_tcp_local_m"]],
                        device=self.env.device,
                        dtype=torch.float32,
                    ),
                )
            )[0]
            .cpu()
            .numpy()
        )
        wrist_up = (
            math.quat_apply(
                quat,
                torch.tensor(
                    [geometry["wrist_up_tcp_local"]],
                    device=self.env.device,
                    dtype=torch.float32,
                ),
            )[0]
            .cpu()
            .numpy()
        )
        for camera, eye, target, up in zip(
            self.cameras,
            (self.external_eye, wrist_eye),
            (self.external_target, wrist_target),
            ((0, 0, 1), wrist_up),
            strict=True,
        ):
            rotation = (
                Gf.Matrix4d()
                .SetLookAt(
                    Gf.Vec3d(*map(float, eye)),
                    Gf.Vec3d(*map(float, target)),
                    Gf.Vec3d(*map(float, up)),
                )
                .GetInverse()
                .ExtractRotationQuat()
            )
            camera.set_world_poses(
                positions=np.asarray(eye),
                orientations=np.asarray([rotation.GetReal(), *rotation.GetImaginary()]),
            )

    def observe(self):
        import isaaclab.utils.math as math
        import numpy as np

        self.update_cameras()
        before = float(self.env.sim.current_time)
        self.env.sim.render()
        self.env.sim.render()
        if float(self.env.sim.current_time) != before:
            raise RuntimeError("Rendering advanced physical time")
        images = []
        for camera_index, sensor in enumerate(self.sensors):
            data, _ = sensor.get_data("rgb")
            image = data.numpy()[..., :3].copy()
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
            # A wrist view may be occluded by the cup/gripper. That is a physical
            # observation, not a reason to discard a test episode or demonstration.
            if (
                image.shape != (224, 224, 3)
                or image.dtype != np.uint8
                or (camera_index == 0 and image.std() < 1)
            ):
                if image.shape == (224, 224, 3) and image.dtype == np.uint8:
                    from PIL import Image

                    debug = Path("outputs/cloud-camera-debug")
                    debug.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(debug / f"camera-{camera_index}.png")
                raise RuntimeError(
                    f"Invalid RGB camera {camera_index}: shape={image.shape}, "
                    f"dtype={image.dtype}, std={image.std()}, "
                    f"range={image.min()}..{image.max()}"
                )
            images.append(image)
        p, q = self.arm._compute_frame_pose()
        state = np.concatenate(
            (
                p[0].cpu().numpy(),
                math.axis_angle_from_quat(q)[0].cpu().numpy(),
                self.robot.data.joint_pos[0, -2:].cpu().numpy(),
            )
        ).astype(np.float32)
        return {"external": images[0], "wrist": images[1], "state": state}

    def reset(self, seed):
        import numpy as np
        from lw_benchhub.utils.place_utils import env_utils

        self.task.context.seed = seed
        self.task.seed = seed
        self.task.rng = np.random.default_rng(seed)
        # Gym's seed does not reseed LW's cached object placement samplers.
        orchestrator = self.env.cfg.isaaclab_arena_env.orchestrator
        self.task.placement_initializer = env_utils._get_placement_initializer(
            orchestrator, self.task.object_cfgs, seed
        )
        self.env.reset(seed=seed)
        self.initial = {
            k: self.env.scene[k].data.root_state_w[0].cpu().numpy().copy() for k in OBJECTS
        }
        self.initial_cup_z = float(self.initial["red_coffee_mug"][2])
        self.maximum_lift = 0.0
        self.control_steps = 0
        self.last_action = None
        self.stable = 0

    def metrics(self):
        import numpy as np

        cup = self.env.scene["red_coffee_mug"].data.root_pos_w[0].cpu().numpy()
        plate = self.env.scene["plate_left"].data.root_pos_w[0].cpu().numpy()
        self.maximum_lift = max(self.maximum_lift, float(cup[2]) - self.initial_cup_z)
        m = {
            "original_predicate": bool(self.task._check_success(self.env).item()),
            "lift_m": self.maximum_lift,
            "xy_distance_m": float(np.linalg.norm(cup[:2] - plate[:2])),
            "height_m": float(cup[2] - plate[2]),
            "finger_positions": self.robot.data.joint_pos[0, -2:].tolist(),
            "cup_position": cup.tolist(),
            "plate_position": plate.tolist(),
            "control_steps": self.control_steps,
        }
        m["physical_success"] = physical_success(
            predicate=m["original_predicate"],
            lift=m["lift_m"],
            xy=m["xy_distance_m"],
            height=m["height_m"],
            fingers=m["finger_positions"],
            open_command=self.last_action is not None and self.last_action[6] >= 0,
        )
        return m

    def pose_action(self, position, quaternion, closed):
        import isaaclab.utils.math as math
        import numpy as np
        import torch

        p, q = math.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            torch.as_tensor(
                np.asarray(position)[None], device=self.env.device, dtype=torch.float32
            ),
            torch.as_tensor(
                np.asarray(quaternion)[None], device=self.env.device, dtype=torch.float32
            ),
        )
        return np.concatenate(
            (
                p[0].cpu().numpy(),
                math.axis_angle_from_quat(q)[0].cpu().numpy(),
                [-1.0 if closed else 1.0],
            )
        ).astype(np.float32)

    def step(self, action):
        import isaaclab.utils.math as math
        import numpy as np
        import torch

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Policy must produce seven finite action dimensions")
        angle = float(np.linalg.norm(action[3:6]))
        if angle < 1e-6:
            quaternion = torch.tensor([[1.0, 0, 0, 0]], device=self.env.device)
        else:
            quaternion = math.quat_from_angle_axis(
                torch.tensor([angle], device=self.env.device),
                torch.as_tensor((action[3:6] / angle)[None], device=self.env.device),
            )
        command = torch.cat(
            (
                torch.as_tensor(action[:3][None], device=self.env.device),
                quaternion,
                torch.tensor([[1.0 if action[6] >= 0 else -1.0]], device=self.env.device),
            ),
            dim=1,
        )
        self.last_action = action
        for _ in range(2):
            self.env.step(command)
            self.control_steps += 1
            self.metrics()
        self.stable = self.stable + 1 if self.metrics()["physical_success"] else 0

    def expert(self):
        import isaaclab.utils.math as math
        import numpy as np
        import torch

        observations, actions, phases = [], [], []
        cup_quat = self.env.scene["red_coffee_mug"].data.root_quat_w.clone()
        orient = (
            math.quat_mul(
                cup_quat, torch.tensor([[0, 2**-0.5, -(2**-0.5), 0]], device=self.env.device)
            )[0]
            .cpu()
            .numpy()
        )

        def drive(phase, end, quaternion, control_steps, closed=False, settle=40):
            start = self.tcp_world()[0][0].cpu().numpy().copy()
            for j in range(0, control_steps + settle, 2):
                alpha = min((j + 2) / control_steps, 1.0)
                smooth = alpha * alpha * (3 - 2 * alpha)
                action = self.pose_action(start + (end - start) * smooth, quaternion, closed)
                # Save the observation BEFORE the action that is its supervision.
                observations.append(self.observe())
                actions.append(action)
                phases.append(phase)
                self.step(action)
            print("EXPERT_PHASE", phase, json.dumps(self.metrics()), flush=True)

        initial_tcp = self.tcp_world()[0][0].cpu().numpy().copy()
        drive("stabilize", initial_tcp, orient, 40)
        cup = self.env.scene["red_coffee_mug"].data.root_pos_w[0].cpu().numpy().copy()
        plate = self.env.scene["plate_left"].data.root_pos_w[0].cpu().numpy().copy()
        offset = (
            math.quat_apply(cup_quat, torch.tensor([[0.0, -0.014, 0.0]], device=self.env.device))[0]
            .cpu()
            .numpy()
        )
        grasp = cup + offset
        high = grasp + [0, 0, 0.19]
        drive("approach", high, orient, 80)
        drive("descend", grasp, orient, 60)
        drive("close", grasp, orient, 50, True)
        drive("lift", high, orient, 70, True)
        if self.maximum_lift < 0.1:
            raise RuntimeError("Expert failed to lift mug")
        target_high = high.copy()
        target_high[:2] = plate[:2] + offset[:2]
        drive("transfer", target_high, orient, 100, True)
        target_low = target_high.copy()
        target_low[2] = plate[2] + 0.0205 + 0.0554 + 0.003
        drive("place", target_low, orient, 60, True)
        drive("release", target_low, orient, 40)
        drive("retreat", target_high + [0, 0, 0.1], orient, 70)
        hold_p, hold_q = self.tcp_world()
        drive("settle", hold_p[0].cpu().numpy(), hold_q[0].cpu().numpy(), 60)
        m = self.metrics()
        if not m["physical_success"] or self.stable < 25:
            raise RuntimeError("Expert failed final physical-success checks: " + json.dumps(m))
        return {
            "external": np.stack([o["external"] for o in observations]),
            "wrist": np.stack([o["wrist"] for o in observations]),
            "state": np.stack([o["state"] for o in observations]),
            "actions": np.stack(actions),
            "phase": np.asarray(phases),
        }, m


def collect(bench, output, protocol, limit=None):
    import numpy as np

    manifest_path = output / "dataset.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "protocol_sha256": file_sha256(Path(protocol["config_path"])),
            "episodes": [],
            "attempts": [],
            "status": "collecting",
        }
    )
    for split in ("train", "validation"):
        count = sum(x["split"] == split for x in manifest["episodes"])
        desired = min(protocol[split + "_episodes"], limit or 10**9)
        seen = [x["seed"] for x in manifest["attempts"] if x["split"] == split]
        seed = max(seen) + 1 if seen else protocol[split + "_seed_start"]
        while count < desired:
            if seed >= protocol[split + "_seed_start"] + 1000:
                raise RuntimeError("Exhausted predefined expert seed pool")
            started = time.perf_counter()
            record = {"seed": seed, "split": split}
            try:
                bench.reset(seed)
                record["initial_objects"] = {k: v.tolist() for k, v in bench.initial.items()}
                arrays, m = bench.expert()
                path = output / f"{split}-{seed}.npz"
                np.savez_compressed(path, **arrays)
                record.update(status="success", frames=len(arrays["actions"]), **m)
                manifest["episodes"].append(
                    {
                        "seed": seed,
                        "split": split,
                        "path": path.name,
                        "sha256": file_sha256(path),
                        "frames": len(arrays["actions"]),
                    }
                )
                count += 1
            except Exception as error:
                record.update(status="failed", error=repr(error))
                traceback.print_exc()
            record["wall_seconds"] = time.perf_counter() - started
            manifest["attempts"].append(record)
            write_json(manifest_path, manifest)
            print("EXPERT_EPISODE", json.dumps(record), flush=True)
            if record["status"] == "failed" and not record["error"].startswith(
                "RuntimeError('Expert failed"
            ):
                raise RuntimeError("Collection infrastructure failed: " + record["error"])
            seed += 1
    manifest["status"] = "smoke_complete" if limit else "complete"
    write_json(manifest_path, manifest)


def evaluate(bench, output, protocol, host, port, seeds):
    import numpy as np
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    from owac.evaluation.baseline_protocol import wilson_interval

    client = WebsocketClientPolicy(host=host, port=port)
    metadata = client.get_server_metadata()
    records_path = output / "episodes.jsonl"
    existing = (
        [json.loads(line) for line in records_path.read_text().splitlines()]
        if records_path.exists()
        else []
    )
    done = {r["seed"] for r in existing}
    with records_path.open("a") as stream:
        for seed in seeds:
            if seed in done:
                continue
            record = {"seed": seed, "success": False, "inference_ms": []}
            started = time.perf_counter()
            try:
                bench.reset(seed)
                record["initial_objects"] = {k: v.tolist() for k, v in bench.initial.items()}
                while bench.control_steps < protocol["episode_control_steps"]:
                    observation = bench.observe()
                    observation["seed"] = seed
                    observation["inference_index"] = len(record["inference_ms"])
                    tick = time.perf_counter()
                    result = client.infer(observation)
                    record["inference_ms"].append((time.perf_counter() - tick) * 1000)
                    actions = np.asarray(result["actions"])
                    if actions.shape != (protocol["action_horizon"], 7):
                        raise ValueError("Unexpected policy action shape")
                    for action in actions[: protocol["execute_horizon"]]:
                        bench.step(action)
                        if bench.stable >= 25:
                            record["success"] = True
                            break
                    if record["success"]:
                        break
                record.update(bench.metrics(), status="completed")
            except ValueError as error:
                # Malformed/nonfinite learned actions are policy failures.
                record.update(status="error", error=repr(error))
                traceback.print_exc()
            except Exception as error:
                # A disconnected server or broken simulator invalidates a run;
                # it cannot become a reported 0/100 policy score.
                write_json(
                    output / "infrastructure-error.json", {"seed": seed, "error": repr(error)}
                )
                traceback.print_exc()
                raise
            record["wall_seconds"] = time.perf_counter() - started
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            existing.append(record)
            print(
                "EVAL_EPISODE",
                json.dumps({k: v for k, v in record.items() if k != "inference_ms"}),
                flush=True,
            )
    successes = sum(r["success"] for r in existing)
    timing = [t for r in existing for t in r["inference_ms"]]
    summary = {
        "status": "complete" if len(existing) == len(seeds) else "incomplete",
        "policy_metadata": metadata,
        "episodes": len(existing),
        "successes": successes,
        "success_rate": successes / len(existing),
        "wilson_95": wilson_interval(successes, len(existing)),
        "errors": sum(r["status"] == "error" for r in existing),
        "inference_ms_p50": float(np.percentile(timing, 50)) if timing else None,
        "inference_ms_p95": float(np.percentile(timing, 95)) if timing else None,
        "total_episode_wall_seconds": sum(r["wall_seconds"] for r in existing),
        "protocol_sha256": file_sha256(Path(protocol["config_path"])),
        "episodes_sha256": file_sha256(records_path),
    }
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "evaluate"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed-start", type=int, help="Override seeds for development smoke only")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol["config_path"] = str(args.protocol.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = launch_simulator()
    env = None
    try:
        env = build_environment(0, demonstration=True)
        env.reset()
        bench = Benchmark(env, protocol)
        write_json(
            args.output / "camera.json",
            {
                "external_eye_world": bench.external_eye.tolist(),
                "external_target_world": bench.external_target.tolist(),
                "geometry": protocol["camera_geometry"],
            },
        )
        if args.mode == "collect":
            collect(bench, args.output, protocol, args.limit)
        else:
            seeds = protocol["test_seeds"][: args.limit] if args.limit else protocol["test_seeds"]
            if args.seed_start is not None:
                if args.limit is None:
                    parser.error("--seed-start requires an explicit --limit")
                seeds = list(range(args.seed_start, args.seed_start + args.limit))
            evaluate(bench, args.output, protocol, args.host, args.port, seeds)
    except Exception as error:
        # Kit shutdown can terminate Python before its outer exception handler.
        traceback.print_exc()
        write_json(args.output / "infrastructure-error.json", {"error": repr(error)})
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()

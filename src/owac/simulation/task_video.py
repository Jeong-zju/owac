"""Record a physical scripted mug-placement demonstration with native RAW/RGB views."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from owac.simulation.lw_libero import TASK, build_environment, launch_simulator, sha256
from owac.simulation.raw_camera import NativeRawCamera, RawCameraProfile
from owac.simulation.raw_preview import comparison_frame


class DemoRecorder:
    """Capture at 25 Hz after two 50 Hz control steps; render a frozen physical state."""

    def __init__(self, env, output):
        import isaaclab.sim as sim
        import numpy as np
        from pxr import UsdGeom

        self.env, self.output = env, output
        self.robot = env.scene["robot"]
        self.arm = env.action_manager.get_term("arm_action")
        self.task = env.cfg.isaaclab_arena_env.task
        root = self.robot.data.root_pos_w[0].cpu().numpy()
        pedestal = sim.CuboidCfg(
            size=(0.24, 0.16, 0.4),
            collision_props=sim.CollisionPropertiesCfg(),
        )
        pedestal.func("/World/OWACRobotPedestal", pedestal, translation=(root[0], root[1], 0.2))
        center = np.mean(
            [
                env.scene[k].data.root_pos_w[0].cpu().numpy()
                for k in ("red_coffee_mug", "plate_left", "plate_right")
            ],
            axis=0,
        )
        target = center + [0, 0.2, 0.15]
        self.camera = NativeRawCamera(
            output / "native",
            RawCameraProfile(width=640, height=480),
            target + [1.25, -1.2, 0.85],
            target,
        )
        self.projection = (
            UsdGeom.Camera(env.scene.stage.GetPrimAtPath("/World/OWACRawCamera"))
            .GetCamera()
            .frustum
        )
        self.video_path = output / "raw-rgb-comparison.mp4"
        self.video_log = (output / "ffmpeg.log").open("w")
        self.encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "warning",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1280x768",
                "-r",
                "25",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.video_path),
            ],
            stdin=subprocess.PIPE,
            stderr=self.video_log,
        )
        (output / "frames").mkdir()
        self.stream = (output / "frames.jsonl").open("w")
        self.records, self.phases, self.kept_native = [], [], set()
        self.steps = 0
        for _ in range(40):
            env.sim.render()

    def tcp_pose(self):
        import isaaclab.utils.math as math

        p, q = self.arm._compute_frame_pose()
        p, q = math.combine_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, p, q
        )
        return p[0].cpu().numpy(), q[0].cpu().numpy()

    def drive(self, phase, end, orient, steps, *, closed=False, settle_steps=40):
        import isaaclab.utils.math as math
        import numpy as np
        import torch

        start, _ = self.tcp_pose()
        first = len(self.records)
        for j in range(steps + settle_steps):
            alpha = min((j + 1) / steps, 1.0)
            smooth = alpha * alpha * (3 - 2 * alpha)
            goal = start + (end - start) * smooth
            p, q = math.subtract_frame_transforms(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                torch.as_tensor(goal[None], device=self.env.device, dtype=torch.float32),
                torch.tensor(orient[None], device=self.env.device, dtype=torch.float32),
            )
            action = torch.cat(
                (p, q, torch.full((1, 1), -1.0 if closed else 1.0, device=self.env.device)), dim=1
            )
            self.env.step(action)
            self.steps += 1
            if self.steps % 2 == 0:
                self.capture(phase, action)
        tcp, _ = self.tcp_pose()
        self.phases.append(
            {
                "phase": phase,
                "first_frame": first,
                "last_frame": len(self.records) - 1,
                "tcp_position": tcp.tolist(),
                "goal_position": np.asarray(end).tolist(),
                "tracking_error_m": float(np.linalg.norm(tcp - end)),
                "cup_position": self.env.scene["red_coffee_mug"].data.root_pos_w[0].tolist(),
                "gripper_joints": self.robot.data.joint_pos[0, -2:].tolist(),
                "original_task_success": bool(self.task._check_success(self.env).item()),
            }
        )
        print("DEMO_PHASE", json.dumps(self.phases[-1], ensure_ascii=False), flush=True)
        self.clean_native()

    def capture(self, phase, action):
        import time

        import numpy as np
        from PIL import Image
        from pxr import Gf

        before = float(self.env.sim.current_time)
        # Flush both native and RGB queues while physics remains frozen.
        self.env.sim.render()
        self.env.sim.render()
        if float(self.env.sim.current_time) != before:
            raise RuntimeError("Rendering changed the physical timestamp")
        cfa, noisy, audit = self.camera.capture()
        rgb, rgb_info = self.camera.sensor.get_data("rgb")
        rgb = rgb.numpy()[..., :3]
        frame = len(self.records)
        stem = f"{frame:06d}"
        paths = {
            key: self.output / "frames" / (stem + suffix)
            for key, suffix in {
                "raw": ".raw.npy",
                "noise": ".noise.npy",
                "rgb": ".rgb.png",
                "state": ".state.npz",
            }.items()
        }
        np.save(paths["raw"], cfa, allow_pickle=False)
        np.save(paths["noise"], noisy, allow_pickle=False)
        Image.fromarray(rgb).save(paths["rgb"])
        objects = {
            k: self.env.scene[k].data.root_state_w[0].cpu().numpy()
            for k in (
                "red_coffee_mug",
                "porcelain_mug",
                "white_yellow_mug",
                "plate_left",
                "plate_right",
            )
        }
        tcp, quat = self.tcp_pose()
        np.savez(
            paths["state"],
            **objects,
            joint_pos=self.robot.data.joint_pos[0].cpu().numpy(),
            joint_vel=self.robot.data.joint_vel[0].cpu().numpy(),
            robot_root_state=self.robot.data.root_state_w[0].cpu().numpy(),
            tcp_position=tcp,
            tcp_quaternion=quat,
            action=action[0].cpu().numpy(),
            saturated=cfa == self.camera.profile.white_level,
        )
        success = bool(self.task._check_success(self.env).item())
        world = objects["red_coffee_mug"][:3]
        matrix = self.projection.ComputeViewMatrix() * self.projection.ComputeProjectionMatrix()
        ndc = matrix.Transform(Gf.Vec3d(*[float(v) for v in world]))
        roi = ((ndc[0] + 1) * 320, (1 - ndc[1]) * 240)
        display = comparison_frame(
            rgb, noisy, time_seconds=before, phase=phase, success=success, roi_center=roi
        )
        self.encoder.stdin.write(display.tobytes())
        if frame % 50 == 0:
            Image.fromarray(display).save(self.output / f"preview-{frame:04d}.png")
        record = {
            "frame": frame,
            "control_step": self.steps,
            "simulation_time_seconds": before,
            "host_timestamp_ns": time.time_ns(),
            "host_monotonic_ns": time.monotonic_ns(),
            "phase": phase,
            "original_task_success": success,
            "cup_position": world.tolist(),
            "plate_position": objects["plate_left"][:3].tolist(),
            "rgb_info": rgb_info,
            "roi_center_pixels": list(roi),
            **audit,
            **{key: str(path.relative_to(self.output)) for key, path in paths.items()},
            **{key + "_sha256": sha256(path) for key, path in paths.items()},
        }
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()
        self.records.append(record)
        self.kept_native.add(audit["native_frame_id"])

    def clean_native(self):
        for path in self.camera.directory.glob("*.bin"):
            match = re.fullmatch(r"(\d+)-[a-z]+(\d+)\.bin", path.name)
            if match and (int(match[1]) not in (0, 2, 3) or int(match[2]) not in self.kept_native):
                path.unlink()

    def close(self):
        if self.encoder.stdin is not None and not self.encoder.stdin.closed:
            self.encoder.stdin.close()
        result = self.encoder.wait()
        self.stream.close()
        self.video_log.close()
        (self.output / "sensor.json").write_text(json.dumps(self.camera.metadata, indent=2) + "\n")
        if result:
            raise RuntimeError(f"Video encoding failed with exit status {result}")


def record_demo(env, output):
    import isaaclab.utils.math as math
    import numpy as np
    import torch

    env.reset()
    recorder = DemoRecorder(env, output)
    try:
        cup_quat = env.scene["red_coffee_mug"].data.root_quat_w
        orient = (
            math.quat_mul(cup_quat, torch.tensor([[0, 2**-0.5, -(2**-0.5), 0]], device=env.device))[
                0
            ]
            .cpu()
            .numpy()
        )
        initial_tcp, _ = recorder.tcp_pose()
        recorder.drive("初始化与稳定", initial_tcp, orient, 40)
        cup = env.scene["red_coffee_mug"].data.root_pos_w[0].cpu().numpy().copy()
        plate = env.scene["plate_left"].data.root_pos_w[0].cpu().numpy().copy()
        # Cup030's handle extends its bounding box; center the grip on the cylindrical body.
        body_offset = (
            math.quat_apply(cup_quat, torch.tensor([[0.0, -0.014, 0.0]], device=env.device))[0]
            .cpu()
            .numpy()
        )
        grasp = cup + body_offset
        high = grasp + [0, 0, 0.19]
        recorder.drive("接近红杯", high, orient, 80)
        recorder.drive("下降对准", grasp, orient, 60)
        recorder.drive("闭合夹爪", grasp, orient, 50, closed=True)
        recorder.drive("抬升红杯", high, orient, 70, closed=True)
        lift = env.scene["red_coffee_mug"].data.root_pos_w[0].cpu().numpy().copy()
        if lift[2] - cup[2] < 0.10:
            raise RuntimeError(
                "Mug was not physically lifted; recording cannot claim a completed task"
            )
        target_high = high.copy()
        target_high[:2] = plate[:2] + body_offset[:2]
        recorder.drive("移动至左盘", target_high, orient, 100, closed=True)
        target_low = target_high.copy()
        target_low[2] = plate[2] + 0.0205 + 0.0554 + 0.003
        recorder.drive("下降放置", target_low, orient, 60, closed=True)
        recorder.drive("松开夹爪", target_low, orient, 40)
        recorder.drive("机械臂退开", target_high + [0, 0, 0.10], orient, 70)
        hold_pos, hold_quat = recorder.tcp_pose()
        recorder.drive("任务完成与稳定", hold_pos, hold_quat, 60)
        final = env.scene["red_coffee_mug"].data.root_state_w[0].cpu().numpy()
        final_plate = env.scene["plate_left"].data.root_pos_w[0].cpu().numpy()
        success = bool(recorder.task._check_success(env).item())
        on_plate_height = 0.025 < final[2] - final_plate[2] < 0.09
        if not success or not on_plate_height or np.linalg.norm(final[:2] - final_plate[:2]) > 0.04:
            raise RuntimeError("Final mug placement did not pass task and physical-height checks")
        return {
            "task_success": success,
            "physical_lift_m": float(lift[2] - cup[2]),
            "final_cup_plate_xy_distance_m": float(np.linalg.norm(final[:2] - final_plate[:2])),
            "final_cup_plate_root_height_m": float(final[2] - final_plate[2]),
            "frames": len(recorder.records),
            "video_fps": 25,
            "video_duration_seconds": len(recorder.records) / 25,
            "phases": recorder.phases,
            "maximum_cfa_quantization_error_dn": max(
                x["cfa_quantization_maximum_error_dn"] for x in recorder.records
            ),
            "last_25_frame_successes": sum(
                x["original_task_success"] for x in recorder.records[-25:]
            ),
            "sensor": recorder.camera.metadata,
        }
    finally:
        recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "task": TASK,
        "seed": 0,
        "controller": "scripted privileged-state pose controller; no learned policy",
        "demonstration_changes": {
            "fixed_pedestal_height_m": 0.4,
            "base_x_offset_m": -0.24,
            "initial_joint_pos": {
                "panda_joint2": -0.785,
                "panda_joint4": -1.57,
                "panda_joint6": 0.785,
            },
            "robot_resampling": False,
            "automatic_success_timeout_reset": False,
        },
        "display_transform": {
            "source": "native noisy CFA, stage 3-noise",
            "black": 0,
            "white": 16777215,
            "mapping": "log1p(15*clip(DN/white,0,1))/log(16)",
            "video_only_uint8_conversion": True,
            "per_frame_auto_scale": False,
            "demosaicing": False,
            "zoom": "nearest neighbor; GRBG site colors only",
        },
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        "source_sha256": {
            str(p.relative_to(root)): sha256(p)
            for p in sorted((root / "src/owac/simulation").glob("*.py"))
        },
        "packages": {
            key: importlib.metadata.version(key)
            for key in ("isaacsim", "isaaclab", "torch", "numpy", "pillow")
        },
        "command": [sys.executable, *sys.argv],
        "output": str(output),
        "runtime_manifest_sha256": sha256(
            root / "configs/simulation/lw-libero-runtime/sources.json"
        ),
    }
    for path in (root / "src/owac/simulation").glob("*.py"):
        snapshot = output / "source" / path.relative_to(root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    (output / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=root))
    (output / "git-status.txt").write_bytes(
        subprocess.check_output(["git", "status", "--short"], cwd=root)
    )
    launcher = launch_simulator()
    env, exit_code = None, 0
    try:
        env = build_environment(0, demonstration=True)
        report.update(record_demo(env, output))
        report["status"] = "passed"
        report["video_sha256"] = sha256(output / "raw-rgb-comparison.mp4")
    except BaseException:
        exit_code = 1
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        traceback.print_exc(file=sys.__stderr__)
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("TASK_VIDEO_" + report["status"].upper(), output, flush=True)
        if env is not None:
            env.close()
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()

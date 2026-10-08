"""Run the deployed Lightwheel-LIBERO task and export native synthetic CFA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from owac.simulation.raw_camera import NativeRawCamera, RawCameraProfile

TASK = "L90L5PutTheRedMugOnTheLeftPlate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_environment(seed: int):
    """Use original scene, placement, robot, and success code with minimal registration."""
    import gymnasium as gym
    import lw_benchhub.utils.env as lw

    def register():
        import lw_benchhub.core.robots.franka  # noqa: F401
        import lw_benchhub.core.scenes.kitchen  # noqa: F401

        gym.register(
            "Robocasa-Task-" + TASK,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": (
                    "lw_benchhub_tasks.lightwheel_libero_tasks.libero_90."
                    "L90L5_put_the_red_mug_on_the_left_plate:" + TASK
                )
            },
            disable_env_checker=True,
        )

    lw.discover_and_import_lw_benchhub_modules = register
    cfg = lw.parse_env_cfg(
        scene_backend="robocasa",
        task_backend="robocasa",
        scene_name="libero-1-1",
        robot_name="Panda",
        task_name=TASK,
        robot_scale=1.0,
        execute_mode=lw.ExecuteMode.EVAL,
        device="cuda:0",
        num_envs=1,
        use_fabric=True,
        enable_cameras=False,
        headless_mode=True,
        seed=seed,
    )
    cfg.seed = seed
    return gym.make(cfg.isaaclab_arena_env.name, cfg=cfg).unwrapped


def collect(env, output: Path, steps: int) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    env.reset()
    task = env.cfg.isaaclab_arena_env.task
    # Frame placement follows the task's actual table and object positions.
    object_names = (
        "red_coffee_mug",
        "porcelain_mug",
        "white_yellow_mug",
        "plate_left",
        "plate_right",
    )
    center = np.mean(
        [env.scene[name].data.root_pos_w[0].cpu().numpy() for name in object_names], axis=0
    )
    target = (float(center[0]), float(center[1]) + 0.2, float(center[2]) + 0.15)
    eye = (target[0] + 0.9, target[1] - 2.0, target[2] + 1.55)
    camera = NativeRawCamera(output / "native", RawCameraProfile(), eye, target)
    (output / "frames").mkdir()
    # Capture retries render the same physical state; they do not step the robot.
    for _ in range(40):
        env.sim.render()
    arm = env.action_manager.get_term("arm_action")
    pos, quat = arm._compute_frame_pose()
    action = torch.cat((pos, quat, torch.ones((1, 1), device=env.device)), dim=-1)
    records = []
    latencies = []
    with (output / "frames.jsonl").open("w") as stream:
        for step in range(steps):
            started = time.perf_counter()
            _, reward, terminated, truncated, _ = env.step(action)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError(
                    "Episode ended during hold-pose collection; reset needs a new run"
                )
            env.sim.render()
            cfa, noisy, audit = camera.capture()
            stem = f"{step:06d}"
            raw_path = output / "frames" / f"{stem}.raw.npy"
            noise_path = output / "frames" / f"{stem}.noise.npy"
            np.save(raw_path, cfa, allow_pickle=False)
            np.save(noise_path, noisy, allow_pickle=False)
            mask_path = output / "frames" / f"{stem}.masks.npz"
            np.savez(mask_path, saturated=cfa == camera.profile.white_level)
            robot = env.scene["robot"]
            joint_pos = robot.data.joint_pos[0].cpu().numpy()
            joint_vel = robot.data.joint_vel[0].cpu().numpy()
            if not np.isfinite(joint_pos).all() or not np.isfinite(joint_vel).all():
                raise ValueError("Robot state contains non-finite values")
            state_path = output / "frames" / f"{stem}.state.npz"
            np.savez(
                state_path,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                robot_root_state=robot.data.root_state_w[0].cpu().numpy(),
                tcp_position=env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy(),
                tcp_quaternion=env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy(),
                action=action[0].cpu().numpy(),
                **{
                    name: env.scene[name].data.root_state_w[0].cpu().numpy()
                    for name in (
                        "red_coffee_mug",
                        "porcelain_mug",
                        "white_yellow_mug",
                        "plate_left",
                        "plate_right",
                    )
                },
            )
            success = bool(task._check_success(env).item())
            record = {
                "step": step,
                "simulation_time_seconds": float(env.sim.current_time),
                "host_timestamp_ns": time.time_ns(),
                "host_monotonic_ns": time.monotonic_ns(),
                "language": task.get_ep_meta()["lang"],
                "task_success_predicate": success,
                "reward": float(reward[0]),
                "raw": str(raw_path.relative_to(output)),
                "noise": str(noise_path.relative_to(output)),
                "state": str(state_path.relative_to(output)),
                "masks": str(mask_path.relative_to(output)),
                "raw_sha256": sha256(raw_path),
                "noise_sha256": sha256(noise_path),
                "state_sha256": sha256(state_path),
                "masks_sha256": sha256(mask_path),
                **audit,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            latencies.append((time.perf_counter() - started) * 1000)
            print(f"RAW_FRAME {step + 1}/{steps} native={audit['native_frame_id']}", flush=True)
    rgb, _ = camera.sensor.get_data("rgb")
    Image.fromarray(rgb.numpy()[..., :3]).save(output / "preview.png")
    (output / "sensor.json").write_text(json.dumps(camera.metadata, indent=2) + "\n")
    return {
        "task": TASK,
        "scene": "libero-1-1",
        "robot": "Panda",
        "controller": "hold initial TCP pose, absolute differential IK, gripper open",
        "frames": len(records),
        "simulation_time_first": records[0]["simulation_time_seconds"],
        "simulation_time_last": records[-1]["simulation_time_seconds"],
        "task_success_predicate": records[-1]["task_success_predicate"],
        "task_policy_evaluation": False,
        "maximum_quantization_error_dn": max(
            r["cfa_quantization_maximum_error_dn"] for r in records
        ),
        "unique_cfa_values_minimum": min(r["unique_values"] for r in records),
        "collection_step_ms_p50": float(np.percentile(latencies, 50)),
        "collection_step_ms_p95": float(np.percentile(latencies, 95)),
        "torch_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "sensor": camera.metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 300:
        parser.error("steps must be between 1 and 300 (within one 8-second episode)")
    root = Path(__file__).resolve().parents[3]
    output = (
        args.output
        or root / "outputs/raw-sim-bench" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["PYNPUT_BACKEND"] = "dummy"
    os.environ["OWAC_NO_TELEOP"] = "1"
    os.environ["OWAC_FRANKA_ONLY"] = "1"
    report = {
        "status": "starting",
        "seed": args.seed,
        "command": [sys.executable, *sys.argv],
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        "source_sha256": {
            str(p.relative_to(root)): sha256(p)
            for p in sorted((root / "src/owac/simulation").glob("*.py"))
        },
        "runtime_config_sha256": {
            str(p.relative_to(root)): sha256(p)
            for p in sorted((root / "configs/simulation/lw-libero-runtime").glob("*"))
            if p.is_file()
        },
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("isaacsim", "isaaclab", "isaaclab-arena", "lightwheel-sdk", "torch", "numpy")
        },
        "output": str(output),
    }
    (output / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=root))
    (output / "git-status.txt").write_bytes(
        subprocess.check_output(["git", "status", "--short"], cwd=root)
    )
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        device="cuda:0",
        experience=str(
            importlib.metadata.distribution("isaacsim-app").locate_file(
                "isaacsim/apps/isaacsim.exp.base.python.kit"
            )
        ),
        kit_args=(
            "--/rtx/spg/enabled=true --/rtx/hydra/supportMultiTickRate=true "
            "--/rtx/rendering/perSensorTickTlas=true"
        ),
    )
    env = None
    exit_code = 0
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        asset_root = (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
        )
        for name in ("default", "cloud", "nvidia"):
            settings.set(f"/persistent/isaac/asset_root/{name}", asset_root)
        env = build_environment(args.seed)
        report.update(collect(env, output, args.steps))
        report["status"] = "passed"
    except BaseException:
        exit_code = 1
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        traceback.print_exc(file=sys.__stderr__)
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"RAW_BENCH_{report['status'].upper()} {output}", flush=True)
        if env is not None:
            env.close()
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()

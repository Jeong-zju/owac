"""Recorded-policy replay for diagnosis only; logs hidden physics, never claims policy success."""

import json
import os
from pathlib import Path

import numpy as np


class PhysicsAudit:
    def __init__(self, task):
        self.task = task
        self.physics_video = os.environ.get("RAWVLA_PHYSICS_VIDEO") == "1"
        self.saved_video_path = task.eval_video_path
        self.root = Path(os.environ["RAWVLA_PHYSICS_DIR"])
        self.path = self.root / "physics.jsonl"
        from owac.runtime.robotwin_grasp_acceptance import GraspHold

        self.verifier = GraspHold()
        self.count = 0
        self.initial_z = float(task.roller.get_pose().p[2])
        self.original_check = task.check_success
        self.first_trigger = None
        self.dt = float(task.scene.get_timestep())
        task.check_success = self.check
        self.record("initial", bool(self.original_check()))
        if self.physics_video:
            task.eval_video_path = None
            self.capture_video()

    def capture_video(self):
        self.task._update_render()
        observation = self.task.get_obs()
        if self.saved_video_path is not None:
            self.task.eval_video_ffmpeg.stdin.write(
                observation["observation"]["head_camera"]["render_rgb"].tobytes()
            )

    def record(self, phase, official):
        task = self.task
        pose = task.roller.get_pose()
        actor_name = task.roller.get_name()
        contacts = []
        for contact in task.scene.get_contacts():
            names = [b.entity.name for b in contact.bodies]
            if actor_name not in names:
                continue
            points = []
            for point in contact.points:
                points.append(
                    {
                        "position": np.asarray(point.position).tolist(),
                        "impulse": np.asarray(point.impulse).tolist()
                        if hasattr(point, "impulse")
                        else None,
                    }
                )
            contacts.append({"bodies": names, "points": points})
        joints = {}
        for side in ("left", "right"):
            entity = getattr(task.robot, side + "_entity")
            joints[side] = {
                j.name: float(v)
                for j, v in zip(entity.get_active_joints(), entity.get_qpos(), strict=True)
            }
        row = {
            "phase": phase,
            "physics_tick": self.count,
            "dt": self.dt,
            "control_step": int(task.take_action_cnt),
            "official_success": official,
            "roller_p": np.asarray(pose.p).tolist(),
            "roller_q": np.asarray(pose.q).tolist(),
            "lift_from_initial": float(pose.p[2]) - self.initial_z,
            "gripper_command": [
                float(task.robot.get_left_gripper_val()),
                float(task.robot.get_right_gripper_val()),
            ],
            "closed_flags": [
                bool(task.is_left_gripper_close()),
                bool(task.is_right_gripper_close()),
            ],
            "joint_positions": joints,
            "roller_contacts": contacts,
            "gripper_link_names": list(task.robot.gripper_name),
        }
        row["physical_valid"] = self.verifier.update(row) or self.verifier.valid
        row["hold_seconds"] = self.verifier.hold_seconds
        if official and self.first_trigger is None:
            self.first_trigger = dict(row)
            (self.root / "first_trigger.json").write_text(json.dumps(row, indent=2) + "\n")
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return row

    def check(self):
        self.count += 1
        if self.physics_video and self.count % max(1, round(0.1 / self.dt)) == 0:
            self.capture_video()
        official = bool(self.original_check())
        if self.count % 10 == 0 or official:
            self.record("policy_trace", official)
        # Complete recorded predictions after the transient flag. Original benchmark untouched.
        return os.environ.get("RAWVLA_PHYSICS_LIVE") == "1" and self.verifier.hold_seconds >= 1.0

    def finish(self, model):
        task = self.task
        rows = []
        for i in range(round(2.0 / self.dt)):
            task.scene.step()
            self.count += 1
            if i % max(1, round(0.02 / self.dt)) == 0:
                rows.append(self.record("passive_hold", bool(self.original_check())))
            if self.physics_video and self.count % max(1, round(0.1 / self.dt)) == 0:
                self.capture_video()
            if not self.physics_video and i % max(1, round(0.1 / self.dt)) == 0:
                task._update_render()
                observation = task.get_obs()
                if task.eval_video_path is not None:
                    task.eval_video_ffmpeg.stdin.write(
                        observation["observation"]["head_camera"]["render_rgb"].tobytes()
                    )
        summary = {
            "diagnostic_only": os.environ.get("RAWVLA_PHYSICS_LIVE") != "1",
            "action_source": os.environ.get("RAWVLA_PHYSICS_POLICY_KIND", "live_raw_policy")
            if os.environ.get("RAWVLA_PHYSICS_LIVE") == "1"
            else "recorded_model_predictions",
            "new_policy_inferences": "live" if os.environ.get("RAWVLA_PHYSICS_LIVE") == "1" else 0,
            "optimizer_updates": 0,
            "video_timebase": "simulation_10fps" if self.physics_video else "control_command_10fps",
            "control_steps": int(task.take_action_cnt),
            "physical_accepted": self.verifier.max_hold_seconds >= 1
            and all(r["physical_valid"] for r in rows),
            "max_hold_seconds": self.verifier.max_hold_seconds,
            "first_official_trigger": self.first_trigger,
            "final_hold": rows[-1],
            "hold_min_z": min(r["roller_p"][2] for r in rows),
            "hold_max_z": max(r["roller_p"][2] for r in rows),
            "hold_official_fraction": float(np.mean([r["official_success"] for r in rows])),
            "max_state_difference_before_original_stop": getattr(
                model, "max_state_difference", None
            ),
        }
        task.eval_video_path = self.saved_video_path
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


class Replay:
    def __init__(self):
        rows = [
            json.loads(s) for s in Path(os.environ["RAWVLA_REPLAY_SOURCE"]).read_text().splitlines()
        ]
        self.commands = {}
        for index, row in enumerate(rows):
            for offset, action in enumerate(row["actions"]):
                self.commands[row.get("step", 50 * index) + offset] = np.asarray(
                    action, dtype=np.float32
                )
        self.original_states = {
            r["step"]: np.asarray(r["state"])
            for r in map(
                json.loads, Path(os.environ["RAWVLA_REPLAY_STATES"]).read_text().splitlines()
            )
        }
        self.max_state_difference = 0.0
        self.audit = None

    def step(self, example, step):
        if step in self.original_states:
            self.max_state_difference = max(
                self.max_state_difference,
                float(np.max(np.abs(self.original_states[step] - example["state"]))),
            )
        return self.commands[step].copy()


def get_model(arguments):
    if os.environ.get("RAWVLA_PHYSICS_LIVE") == "1":
        from owac.runtime.n02_v12_robotwin import get_model as live_model

        return live_model(arguments)
    import sapien

    original = sapien.render.set_ray_tracing_denoiser
    sapien.render.set_ray_tracing_denoiser = lambda name: original(
        "optix" if name == "oidn" else name
    )
    return Replay()


def reset_model(model):
    if os.environ.get("RAWVLA_PHYSICS_LIVE") == "1":
        from owac.runtime.n02_v12_robotwin import reset_model as live_reset

        live_reset(model)
    model.audit = None


def eval(task, model, observation):
    from owac.runtime.rawvla_robotwin import eval as base_eval

    if getattr(model, "audit", None) is None:
        model.audit = PhysicsAudit(task)
        task.step_lim = (
            len(model.commands)
            if hasattr(model, "commands")
            else int(os.environ.get("RAWVLA_PHYSICS_MAX_STEPS", "400"))
        )
        if not 1 <= task.step_lim <= 1000:
            raise ValueError("physical development rollout budget must be 1..1000 commands")
    base_eval(task, model, observation)
    if task.take_action_cnt == task.step_lim or task.eval_success:
        model.audit.finish(model)

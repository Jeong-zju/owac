"""Native three-camera RoboTwin plugin for the complete N01 service.

Uses the frozen pi05_robotwin MEAN_STD and 50-step chunk protocol. Collects
measurements on EVERY environment step, including between action replans. RAW
frames stay float32 at camera resolution; state/action normalization reuses the
accepted baseline functions. This module never creates a trainer.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import numpy as np

from owac.runtime.rawvla_robotwin import normalize_state, unnormalize_actions

CAMERAS = ("head_camera", "left_camera", "right_camera")


class NativeRoboTwinClient:
    def __init__(self, args, rpc=None):
        if rpc is None:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

            rpc = WebsocketClientPolicy(args.get("host", "127.0.0.1"), int(args.get("port", 18005)))
        self.client = rpc
        self.raw_key = args.get("raw_input_field", "raw_rgb10")
        if int(args.get("replan_steps", 50)) != 50:
            raise ValueError("changing RoboTwin's frozen 50-step cadence is a separate experiment")
        stats = args.get("norm_stats")
        if stats is None:
            path = args.get("norm_stats_path")
            if not path:
                raise ValueError("native RoboTwin client requires the frozen normalization stats")
            stats = json.loads(Path(path).read_text())
        self.stats = stats.get("norm_stats", stats)
        self.window = int(args.get("window_frames", 32))
        if not 1 <= self.window <= 32:
            raise ValueError("native transport history must fit the registered 32-frame bound")
        self.episode_base = str(
            args.get("episode_id") or os.environ.get("RAWVLA_CASE_ID") or "robotwin-session"
        )
        self.reset_index = -1
        self.reset()

    def reset(self):
        self.reset_index += 1
        self.episode_id = f"{self.episode_base}/reset-{self.reset_index}"
        self.history = {camera: collections.deque(maxlen=self.window) for camera in CAMERAS}
        self.actions = None
        self.chunk_start = 0

    def step(self, example, step):
        if len(example["image"]) != 3:
            raise ValueError("native RoboTwin requires head/left/right cameras")
        for camera, value in zip(CAMERAS, example["image"], strict=True):
            value = np.asarray(value)
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.float32:
                raise ValueError("native RoboTwin frames must be float32 HWC pseudo-RAW")
            history = self.history[camera]
            if history and step < history[-1][0]:
                raise ValueError("reset the native client at episode boundaries")
            if not history or step > history[-1][0]:
                history.append((int(step), value.copy()))
        if self.actions is None or step - self.chunk_start >= 50:
            frames = {}
            for camera, history in self.history.items():
                ids = np.asarray([entry[0] for entry in history], dtype=np.int64)[None]
                values = np.stack([entry[1] for entry in history]).transpose(0, 3, 1, 2)[None]
                frames[camera] = {
                    "values": values,
                    "frame_ids": ids,
                    "timestamp": ids.astype(np.float32),
                    "episode_ids": [self.episode_id],
                    "time_unit": "frame",
                    "new_measurement": np.ones(ids.shape, dtype=bool),
                    "value_units": "normalized_raw_measurement",
                }
            payload = {
                "frames": frames,
                "normalized_state": normalize_state(example["state"], self.stats["state"])[None],
                "language": [example["lang"]],
            }
            seed = os.environ.get("RAWVLA_FLOW_SEED")
            if seed is not None:
                payload["noise"] = np.random.default_rng(int(seed) + step).standard_normal(
                    (1, 50, 32), dtype=np.float32
                )
            response = self.client.predict_action(payload)
            if not response.get("ok"):
                raise RuntimeError(f"N01 RoboTwin inference failed: {response}")
            data = response["data"]
            if "normalized_actions" not in data:
                raise RuntimeError("N01 returned no legal normalized action chunk")
            self.actions = unnormalize_actions(
                np.asarray(data["normalized_actions"])[0], self.stats["actions"]
            )
            self.chunk_start = step
        return self.actions[step - self.chunk_start].copy()


def get_model(usr_args):
    # Preserve the accepted Blackwell simulator adaptation before creating tasks.
    denoiser = os.environ.get("RAWVLA_ROBOTWIN_DENOISER")
    if denoiser:
        if denoiser not in {"optix", "none", "oidn"}:
            raise ValueError("unknown explicit RoboTwin denoiser")
        import sapien

        original = sapien.render.set_ray_tracing_denoiser

        def set_denoiser(name):
            original(denoiser if name == "oidn" else name)

        sapien.render.set_ray_tracing_denoiser = set_denoiser
    return NativeRoboTwinClient(usr_args)


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    raw_key = os.environ.get("OWAC_ROBOTWIN_RAW_KEY", model.raw_key)
    images = []
    for camera in CAMERAS:
        camera_obs = observation["observation"][camera]
        if raw_key not in camera_obs:
            raise ValueError(f"camera {camera} has no native {raw_key}; never fall back to RGB")
        value = np.asarray(camera_obs[raw_key])
        if raw_key == "raw_rgb10":
            if value.dtype != np.uint16:
                raise ValueError("author raw_rgb10 transport must be uint16 RAW10")
            value = value.astype(np.float32) / 1023.0
        elif raw_key != "raw_float32":
            raise ValueError("unsupported RAW unit declaration")
        images.append(value)
    example = {
        "image": images,
        "state": observation["joint_action"]["vector"],
        "lang": str(TASK_ENV.get_instruction()),
    }
    action = model.step(example, int(TASK_ENV.take_action_cnt))
    if hasattr(TASK_ENV, "policy_predicted_actions"):
        TASK_ENV.policy_predicted_actions.append(action.copy())
    TASK_ENV.take_action(action)

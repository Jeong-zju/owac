"""Concrete task6 LIBERO controller/environment adapters for teacher evaluation.

Module import is headless: LIBERO/author modules are loaded only when an
environment or concrete controller is constructed.  The controller consumes
RAW frames only; paired clean-RGB data/caches are never consulted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.evaluation.rawvla_teacher import (
    TeacherOfflineEvaluator,
    load_pinned_postprocessor,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TASK_ID = 6
TASK_SUITE = "libero_goal"
INSTRUCTION = "put the cream cheese in the bowl"
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256


class TeacherControllerError(RuntimeError):
    pass


def normalize_state8(state: np.ndarray, stats: Any) -> np.ndarray:
    raw = np.asarray(state, dtype=np.float32).reshape(-1)
    if raw.size != 8:
        raise TeacherControllerError(f"expected raw canonical state8, got {raw.shape}")
    q01 = np.asarray(stats.state["q01"], dtype=np.float32)[:8]
    q99 = np.asarray(stats.state["q99"], dtype=np.float32)[:8]
    # Exact native deployment normalize_quantile: no clipping, +1e-6.
    return ((raw - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)


def raw_observation_frames(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """180-degree native camera convention, 256x256, no default ISP."""
    return {
        "agentview": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
    }


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = float(np.sqrt(max(0.0, 1.0 - q[3] * q[3])))
    if abs(den) < 1e-12:
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * np.arccos(q[3])) / den).astype(np.float32)


def _state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        )
    ).astype(np.float32)


@dataclass
class _AuthorEnv:
    env: Any
    initial_state: np.ndarray
    seed: int

    def reset(self) -> dict[str, Any]:
        # Exact ordering: seed -> reset -> lighting XML -> apply -> set_init_state.
        self.env.seed(int(self.seed))
        self.env.reset()
        if hasattr(self.env, "get_sim_state"):
            _ = self.env.get_sim_state()
        return self.env.set_init_state(self.initial_state)

    def _observe(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "raw": raw_observation_frames(observation),
            "robot_state": _state_from_obs(observation),
            "native": observation,
        }

    def step(self, action: Any):
        observation, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        return self._observe(observation), reward, done, info

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    def close(self) -> None:
        self.env.close()


def _load_author_replay_module(author_root: str | Path) -> Any:
    """Load the author replay script directly; no old replay/hash module."""
    root = Path(author_root)
    path = root / "scripts/replay/replay_libero_train_rawvla_light_npz.py"
    if not path.is_file():
        raise TeacherControllerError(f"author replay script absent: {path}")
    for extra in (root, root / "experiments/simulation/libero"):
        value = str(extra.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    name = "owac_author_rawvla_replay"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TeacherControllerError(f"cannot load author replay module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TeacherControllerError(f"author replay dependencies/import failed: {exc}") from exc
    return module


class LIBEROEnvAdapter:
    """Pinned task6 environment with deterministic lighting and dummy wait."""

    def __init__(
        self,
        *,
        author_root: str | Path,
        libero_config: str | Path,
        task_id: int = TASK_ID,
        env_factory: Any = None,
        setup_wait_steps: int = 10,
    ) -> None:
        self.author_root = Path(author_root)
        self.libero_config = Path(libero_config)
        self.task_id = int(task_id)
        if self.task_id != TASK_ID:
            raise TeacherControllerError("this authority is restricted to zero-based task6")
        self.setup_wait_steps = int(setup_wait_steps)
        self._env_factory = env_factory
        self.author: Any = None
        self.task: Any = None
        self.env: Any = None
        self.lighting: dict[str, Any] | None = None

    def _load(self) -> None:
        if self._env_factory is not None:
            return
        if not self.libero_config.is_file():
            raise TeacherControllerError(f"LIBERO config absent: {self.libero_config}")
        os.environ["LIBERO_CONFIG_PATH"] = str(self.libero_config.parent)
        from libero.libero import benchmark

        self.author = _load_author_replay_module(self.author_root)
        self.task = benchmark.get_benchmark_dict()[TASK_SUITE]().get_task(self.task_id)
        if self.task.name != "put_the_cream_cheese_in_the_bowl":
            raise TeacherControllerError(f"task numbering/name mismatch: {self.task.name}")

    def _make(self, seed: int):
        if self._env_factory is not None:
            return self._env_factory(self.task, seed)
        assert self.author is not None
        return self.author._make_env(self.task, seed)

    def validate(self) -> None:
        if self._env_factory is not None:
            return
        self._load()
        if self.task is None:
            raise TeacherControllerError("LIBERO task6 did not load")

    def bind(self, state_entry: dict[str, Any], states_array: Any) -> BoundLIBEROEnv:
        return BoundLIBEROEnv(self, state_entry, states_array)

    def reset(self, state_entry: dict[str, Any], states_array: np.ndarray) -> dict[str, Any]:
        self._load()
        index = int(state_entry["array_index"])
        state = np.asarray(states_array[index], dtype=np.float64)
        if state.shape != (79,) or not np.isfinite(state).all():
            raise TeacherControllerError("initial state must be finite exact79")
        seed = int(state_entry["seed"])
        self.env = self._make(seed)
        self.env.seed(seed)
        self.env.reset()
        if self.author is not None:
            from rawvla_bench.libero import apply_entry_lighting
            from rawvla_bench.lighting import rawvla_light_xml_from_string

            lighting = self._normal_lighting(seed)
            xml = rawvla_light_xml_from_string(self.env.sim.model.get_xml())
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            apply_entry_lighting(self.env, lighting)
            self.lighting = lighting
        observation = self.env.set_init_state(state)
        self._initialized = True
        return self._observe(observation)

    def _normal_lighting(self, seed: int) -> dict[str, Any]:
        from rawvla_bench.lighting import lighting_parameters_for_sample

        rng = np.random.default_rng(int(seed))
        ev = float(rng.uniform(-0.5, 0.5))
        sample_unit = float(rng.random())
        result = dict(lighting_parameters_for_sample("Normal", ev, sample_unit=sample_unit))
        result.update(
            {
                "noise_seed": int(seed) ^ 0x5A17,
                "lighting_domain": "Normal",
                "representation": "Normal",
            }
        )
        return result

    def _observe(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "raw": raw_observation_frames(observation),
            "robot_state": _state_from_obs(observation),
            "native": observation,
        }

    def step(self, action: Any):
        observation, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        return self._observe(observation), reward, done, info

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


class BoundLIBEROEnv:
    """Thin no-arg reset wrapper consumed by ClosedLoopDriver."""

    def __init__(
        self, adapter: LIBEROEnvAdapter, state_entry: dict[str, Any], states_array: Any
    ) -> None:
        self.adapter = adapter
        self.state_entry = state_entry
        self.states_array = states_array

    def reset(self):
        return self.adapter.reset(self.state_entry, self.states_array)

    def step(self, action):
        return self.adapter.step(action)

    def check_success(self) -> bool:
        return self.adapter.check_success()

    def close(self) -> None:
        self.adapter.close()

    @property
    def lighting(self):
        return self.adapter.lighting

    def validate(self) -> None:
        self.adapter.validate()


class TeacherPolicyController:
    """Default-ISP reference or learned-RAW author policy."""

    def __init__(
        self,
        *,
        teacher: Any,
        stats: Any,
        role: str,
        author_root: str | Path,
        seed: int = 101,
        lighting: dict[str, Any] | None = None,
    ) -> None:
        if role not in {"default_reference", "learned_raw_teacher"}:
            raise TeacherControllerError(f"unknown role: {role}")
        self.teacher = teacher
        self.stats = stats
        self.role = role
        self.seed = int(seed)
        self.author_root = Path(author_root)
        import sys

        for extra in (
            self.author_root / "experiments/simulation/libero",
            self.author_root / "starVLA",
        ):
            if extra.exists() and str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        self.lighting: dict[str, Any] = dict(lighting or {})
        self._evaluator = TeacherOfflineEvaluator(teacher=teacher, data=None, stats=stats)
        self._camera_state: dict[str, Any] = {}
        self._camera_theta: dict[str, Any] = {}
        self._ring: list[dict[str, Any]] = []
        self._last_frame_index: int | None = None
        self._device = next(teacher.frontend.model.parameters()).device

    def reset(self) -> None:
        self._camera_state = {}
        self._camera_theta = {}
        self._ring = []
        self._last_frame_index = None
        self.teacher.policy.eval()
        self.teacher.frontend.eval()

    def _raw_from_obs(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        native = observation.get("native", observation)
        return raw_observation_frames(native)

    def observe(self, observation: dict[str, Any], policy_step: int) -> None:
        """Collect unique physical frames; repeated current observations are skipped."""
        from rawvla_bench.raw import make_rawvla_observation

        frame_index = int(policy_step)
        if self._last_frame_index is not None:
            if frame_index < self._last_frame_index:
                raise TeacherControllerError(
                    f"decreasing physical frame index: {frame_index} < {self._last_frame_index}"
                )
            if frame_index == self._last_frame_index:
                return
        frames = self._raw_from_obs(observation)
        noisy_raw = {}
        for idx, key in enumerate(("agentview", "wrist")):
            noisy_raw[key] = make_rawvla_observation(
                frames[key],
                noise_seed=int(self.lighting.get("noise_seed", self.seed)),
                frame_index=int(policy_step),
                view_index=idx,
                representation="raw",
                sensor_saturation_ev=float(self.lighting.get("sensor_saturation_ev", 0.0)),
            )
        encoded_rgb = {
            key: hashlib.sha256(np.ascontiguousarray(frames[key]).tobytes()).hexdigest()
            for key in frames
        }
        encoded_raw = {
            key: hashlib.sha256(np.ascontiguousarray(noisy_raw[key]).tobytes()).hexdigest()
            for key in noisy_raw
        }
        self._ring.append(
            {
                "rgb": frames,
                "raw": noisy_raw,
                "frame_index": frame_index,
                "rgb_sha256": encoded_rgb,
                "raw_sha256": encoded_raw,
            }
        )
        self._ring = self._ring[-6:]
        self._last_frame_index = frame_index

    def act(
        self,
        observation: dict[str, Any],
        robot_state: np.ndarray,
        noise: np.ndarray | torch.Tensor,
        *,
        episode_id: str,
        policy_step: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._ring:
            self.observe(observation, policy_step)
        frames = self._ring[-1]["rgb"]
        state32 = normalize_state8(robot_state, self.stats)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        full_started = time.perf_counter()
        frontend_started = time.perf_counter()
        burst_images = []
        for key in ("agentview", "wrist"):
            hist = [item["raw"][key] for item in self._ring]
            hist = [hist[0]] * (6 - len(hist)) + hist
            burst = (
                torch.from_numpy(np.stack(hist)).permute(0, 3, 1, 2).float().to(self._device)
                / 255.0
            )
            burst_images.append(burst)
        from rawvla_bench.raw import make_rawvla_observation

        if self.role == "default_reference":
            rgb_for_policy = {}
            native_image_tools = load_pinned_postprocessor().__globals__["image_tools"]
            for idx, key in enumerate(("agentview", "wrist")):
                fixed_rgb = make_rawvla_observation(
                    frames[key],
                    noise_seed=int(self.lighting.get("noise_seed", self.seed)),
                    frame_index=int(policy_step),
                    view_index=idx,
                    representation="default_isp",
                    sensor_saturation_ev=float(self.lighting.get("sensor_saturation_ev", 0.0)),
                )
                # Native benchmark client resizes Default-ISP RGB8 with PIL
                # before the model helper; do not substitute tensor resizing.
                fixed_rgb = native_image_tools.convert_to_uint8(
                    native_image_tools.resize_image(fixed_rgb, 224, 224)
                )
                rgb_for_policy[key] = (
                    torch.from_numpy(np.array(fixed_rgb, copy=True))
                    .permute(2, 0, 1)
                    .float()
                    .to(self._device)
                    / 255.0
                )
        else:
            rgb_for_policy = {}
            for idx, key in enumerate(("agentview", "wrist")):
                with torch.no_grad():
                    out = self.teacher.frontend.model(
                        burst_images[idx].unsqueeze(0),
                        state=self._camera_state.get(key),
                        theta_prev=self._camera_theta.get(key),
                    )
                self._camera_state[key] = out.state.detach()
                self._camera_theta[key] = out.theta.detach()
                rgb_for_policy[key] = out.rgb.detach()[0].float().clamp(0.0, 1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        frontend_seconds = time.perf_counter() - frontend_started
        processed_rgb_sha256 = {
            key: hashlib.sha256(
                rgb_for_policy[key].detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            for key in rgb_for_policy
        }
        policy_obs = self._evaluator._policy_observation(
            rgb_for_policy,
            state32,
            INSTRUCTION,
        )
        noise_t = torch.as_tensor(noise, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            normalized = self.teacher.policy.sample_actions(policy_obs, noise=noise_t, num_steps=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        postprocess = load_pinned_postprocessor()
        physical = postprocess(
            np.asarray(normalized.detach().cpu(), dtype=np.float32)[0],
            np.asarray(robot_state, dtype=np.float32).reshape(8),
            {
                "actions": {
                    "q01": np.asarray(self.stats.action["q01"], dtype=np.float32),
                    "q99": np.asarray(self.stats.action["q99"], dtype=np.float32),
                }
            },
            model_name="PI05",
            model_source="openpi",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        full_seconds = time.perf_counter() - full_started
        peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        )
        metadata = {
            "role": self.role,
            "raw_input": True,
            "default_isp": self.role == "default_reference",
            "normalized_actions": np.asarray(normalized.detach().cpu(), dtype=np.float32),
            "physical_actions": np.asarray(physical, dtype=np.float32),
            "state8": np.asarray(robot_state, dtype=np.float32).reshape(8).copy(),
            "noise_sha256": hashlib.sha256(
                np.ascontiguousarray(noise_t.detach().cpu().numpy()).tobytes()
            ).hexdigest(),
            "camera_frame_indices": [item["frame_index"] for item in self._ring],
            "raw_sha256": [item["raw_sha256"] for item in self._ring],
            "processed_rgb_sha256": processed_rgb_sha256,
            "frontend_seconds": float(frontend_seconds),
            "full_seconds": float(full_seconds),
            "peak_allocated_bytes": peak_allocated,
            "lighting": dict(self.lighting),
            "policy_step": int(policy_step),
        }
        return np.asarray(physical, dtype=np.float32)[:5], metadata


__all__ = [
    "INSTRUCTION",
    "LIBERO_DUMMY_ACTION",
    "LIBEROEnvAdapter",
    "TeacherControllerError",
    "TeacherPolicyController",
    "normalize_state8",
    "raw_observation_frames",
]

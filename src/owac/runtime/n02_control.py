"""Shared, versioned action execution for N02, native witnesses and validation."""

import dataclasses
import json
import math

import numpy as np


def decision_flow_seed(base_seed: int, step: int, mode: str = "independent") -> int:
    """An explicit sampling intervention; independent preserves the prior schedule."""
    if mode not in ("independent", "fixed") or int(step) < 0:
        raise ValueError("unknown flow-noise mode or negative decision step")
    return (int(base_seed) + (int(step) if mode == "independent" else 0)) % 2**32


@dataclasses.dataclass(frozen=True)
class ControlProtocol:
    version: str = "n02-control-r11-v1"
    horizon: int = 50
    replan_interval: int = 10
    max_joint_delta: float | None = 0.05
    max_gripper_delta: float | None = 0.05
    clip_grippers: bool = True

    def __post_init__(self):
        if self.version != "n02-control-r11-v1" or self.horizon != 50:
            raise ValueError("unsupported action execution protocol")
        if not 1 <= self.replan_interval <= self.horizon:
            raise ValueError("replan interval must be 1..50")
        for value in (self.max_joint_delta, self.max_gripper_delta):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("action delta limits must be finite and positive")

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def parse(cls, value):
        return cls(**(json.loads(value) if isinstance(value, str) else value))


class ActionExecution:
    """One persistent command history, reset only at an episode boundary."""

    def __init__(self, protocol):
        self.protocol = protocol
        self.previous = None

    def reset(self):
        self.previous = None

    def step(self, requested, state):
        action = np.asarray(requested, dtype=np.float32).copy()
        measured = np.asarray(state, dtype=np.float32)
        if action.shape != (14,) or measured.shape != (14,):
            raise ValueError("execution requires 14-dimensional actions and measured state")
        if not np.isfinite(action).all() or not np.isfinite(measured).all():
            raise ValueError("nonfinite execution action/state")
        previous = measured if self.previous is None else self.previous
        limits = np.full(14, np.inf, dtype=np.float32)
        if self.protocol.max_joint_delta is not None:
            limits[:] = self.protocol.max_joint_delta
        limits[[6, 13]] = (
            np.inf if self.protocol.max_gripper_delta is None else self.protocol.max_gripper_delta
        )
        # Unlimited axes must preserve the requested float32 command exactly.
        action = np.where(
            np.isfinite(limits), previous + np.clip(action - previous, -limits, limits), action
        )
        if self.protocol.clip_grippers:
            action[[6, 13]] = action[[6, 13]].clip(0, 1)
        self.previous = action.copy()
        return action

    def prefix(self, requested, state):
        """A prefix from a declared starting state; this is not a physical rollout."""
        requested = np.asarray(requested)
        if requested.shape != (50, 14):
            raise ValueError("expected a 50x14 chunk")
        return np.stack(
            [self.step(action, state) for action in requested[: self.protocol.replan_interval]]
        )

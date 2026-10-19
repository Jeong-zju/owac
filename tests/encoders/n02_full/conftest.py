"""Contract fixtures for the N02 P0 data tests.

Every NPZ written here is a **contract fixture**, not real RawVLA data. The files follow the
documented author key schema (per-camera raw keys, ``actions``, ``robot_states``,
``metadata_json``) so that the partition/anchor/dataset/audit mechanism can be checked with
declared numbers. The real RawVLA cache is not on this workstation (2026-10-14: ``data/`` holds
only README files and the deployed server is unreachable), so no test in this directory is
evidence about the real cache.

Fixtures are generated into ``tmp_path`` on purpose: a checked-in binary would be one more
asset that can go stale, and the schema is short enough to state in code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

LIBERO_SUITE_A = "libero_spatial_no_noops_1.0.0_lerobot"
LIBERO_SUITE_B = "libero_object_no_noops_1.0.0_lerobot"
LIBERO_SUITES = (LIBERO_SUITE_A, LIBERO_SUITE_B)
ROBOTWIN_TASKS = ("grab_roller", "lift_pot")


def write_libero_episode(
    cache_root: str | Path,
    suite: str,
    episode: int,
    variant: int = 0,
    *,
    num_frames: int = 24,
    agentview_hw: tuple[int, int] = (8, 8),
    wrist_hw: tuple[int, int] = (6, 10),
    actions: np.ndarray | None = None,
    robot_states: np.ndarray | None = None,
    lighting_domain: str | None = None,
    task_description: str | None = None,
    extra: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Write one author-schema LIBERO NPZ; return the arrays that were stored.

    LIBERO raw frames are uint8 pseudo-RAW: the loader must divide by 255.0 and must never
    rescale per frame. ``robot_states[t]`` is the 8-dim proprio read at the same step.
    """
    directory = Path(cache_root) / "data/libero/raw" / suite / f"episode_{episode:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"variant_{variant:02d}.npz"
    rng = np.random.default_rng(episode * 101 + variant if seed is None else seed)
    agentview = rng.integers(0, 256, size=(num_frames, *agentview_hw, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(num_frames, *wrist_hw, 3), dtype=np.uint8)
    if actions is None:
        steps = np.arange(num_frames, dtype=np.float32)
        actions = np.zeros((num_frames, 7), dtype=np.float32)
        actions[:, 0] = steps / 100.0
        actions[:, 1] = np.sin(steps)
        actions[:, 6] = np.linspace(-1.0, 1.0, num_frames, dtype=np.float32)
    if robot_states is None:
        # Distinctive per row so a test can prove which source row was read.
        robot_states = (
            np.arange(num_frames, dtype=np.float32)[:, None]
            + np.arange(8, dtype=np.float32)[None, :] * 0.5
        )
    metadata = {
        "task_description": task_description or f"{suite} episode {episode}",
        "lighting_domain": lighting_domain if lighting_domain is not None else "normal",
    }
    payload = {
        "agentview_raw_uint8": agentview,
        "wrist_raw_uint8": wrist,
        "actions": actions,
        "robot_states": robot_states,
        "metadata_json": np.array(json.dumps(metadata)),
    }
    if extra:
        payload.update(extra)
    np.savez(path, **payload)
    return {
        "path": path,
        "relative_path": str(path.relative_to(Path(cache_root))),
        "actions": actions,
        "robot_states": robot_states,
        "agentview": agentview,
        "wrist": wrist,
        "metadata": metadata,
    }


def write_robotwin_episode(
    cache_root: str | Path,
    task: str,
    episode: int,
    *,
    num_frames: int = 60,
    height: int = 8,
    width: int = 8,
    actions: np.ndarray | None = None,
    robot_states: np.ndarray | None = None,
    extra: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Write one author-schema RoboTwin NPZ; return the arrays that were stored.

    RoboTwin raw frames are float32 HDR-derived RAW in their own float scale: the loader must
    neither divide by 255 nor per-frame normalise. ``actions[t]`` is the author's current 14-dim
    joint proxy; the cached 16-dim ``robot_states`` is an end-effector representation.
    """
    directory = Path(cache_root) / "data/robotwin2/raw" / task
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{episode:06d}.npz"
    rng = np.random.default_rng(episode * 7919 + len(task) if seed is None else seed)
    head = (rng.random((num_frames, height, width, 3)) * 0.4332).astype(np.float32)
    left = (rng.random((num_frames, height, width, 3)) * 0.3744).astype(np.float32)
    right = (0.0003 + rng.random((num_frames, height, width, 3)) * 0.3734).astype(np.float32)
    if actions is None:
        steps = np.arange(num_frames, dtype=np.float32)[:, None]
        columns = np.arange(14, dtype=np.float32)[None, :]
        actions = (np.sin(steps * 0.05 + columns * 0.1) + columns * 0.1).astype(np.float32)
    if robot_states is None:
        robot_states = (
            np.arange(num_frames, dtype=np.float32)[:, None]
            + np.arange(16, dtype=np.float32)[None, :] * 0.25
        )
    metadata = {"task_name": f"{task}_demo", "lighting_domain": "normal"}
    payload = {
        "head_camera_raw_float32": head,
        "left_camera_raw_float32": left,
        "right_camera_raw_float32": right,
        "actions": actions,
        "robot_states": robot_states,
        "metadata_json": np.array(json.dumps(metadata)),
    }
    if extra:
        payload.update(extra)
    np.savez(path, **payload)
    return {
        "path": path,
        "relative_path": str(path.relative_to(Path(cache_root))),
        "actions": actions,
        "robot_states": robot_states,
        "head": head,
        "left": left,
        "right": right,
        "metadata": metadata,
    }


@pytest.fixture
def libero_writer():
    return write_libero_episode


@pytest.fixture
def robotwin_writer():
    return write_robotwin_episode

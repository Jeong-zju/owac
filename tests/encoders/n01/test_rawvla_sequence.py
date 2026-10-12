"""Numerical tests for the RawVLA NPZ sequence loader, on synthetic fixtures.

No real RawVLA data exists on this machine (``/workspace/owac/rawvla`` is absent),
so every NPZ used here is generated inside ``tmp_path`` with the author key names
and declares ``metadata_json["synthetic_fixture"] = true``.  Nothing in this file
claims a value measured from the real dataset; the single piece of real range
evidence is the committed 2026-10-10 deployment record, which is read (not
re-measured) by one test to justify the RoboTwin "no /255" decision.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from owac.core.encoder_contract import MeasurementFrame
from owac.raw.cfa import (
    CFA_PATTERNS,
    CfaMosaic,
    crop_planes,
    estimate_noise_model,
    shift_planes,
    shot_read_variance,
)
from owac.raw.rawvla_sequence import (
    LIBERO_NOISE_MODEL,
    LIBERO_PROTOCOL,
    ROBOTWIN_PROTOCOL,
    MeasurementWindow,
    MissingRawVLAAssetError,
    RawVLASequenceDataset,
    assign_side,
    build_split_manifest,
    collate_measurement_windows,
    collate_sequence_samples,
    default_rawvla_root,
    load_split_manifest,
    measurement_window_from_cfa,
    rawvla_asset_requirements,
    recorded_range_evidence,
    source_group,
    write_split_manifest,
)

FRAMES = 12
HEIGHT = 4
WIDTH = 4
SUITE = "libero_spatial_no_noops_1.0.0_lerobot"
FIXTURE_NOTE = (
    "OWAC-generated synthetic fixture written with the author RawVLA key names; "
    "this is not real RawVLA data"
)
RECORDED_RANGE_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research"
    / "experiments"
    / "results"
    / "2026-10-10_rawvla_pi05_deployment"
    / "dataset-check.json"
)


# --------------------------------------------------------------------------------------
# Synthetic fixtures (author key names, explicit synthetic declaration)
# --------------------------------------------------------------------------------------


def libero_raw_value(t: int, i: int, j: int, c: int, offset: int = 0) -> int:
    return (offset + 16 * t + 4 * i + j + c) % 256


def libero_action(t: int, d: int) -> float:
    return 0.05 * t + 0.1 * d


def libero_state(t: int, d: int) -> float:
    return 0.25 * t + 0.5 * d


ROBOTWIN_BASE = 0.43


def robotwin_frame_scale(t: int) -> float:
    """Odd frames are 4x darker, so a per-frame max scaling would visibly rescale them."""
    return 1.0 if t % 2 == 0 else 0.25


def robotwin_raw_value(t: int, i: int, j: int, c: int, base: float = ROBOTWIN_BASE) -> float:
    spatial = 0.001 * (3 * i + 5 * j + 7 * c)
    return robotwin_frame_scale(t) * (base * 0.25 * (1.0 + 0.2 * t) + spatial)


def robotwin_action(t: int, d: int) -> float:
    return -1.0 + 0.15 * t + 0.05 * d


def _index_grid(frames: int, height: int, width: int) -> tuple[np.ndarray, ...]:
    t = np.arange(frames)[:, None, None, None]
    i = np.arange(height)[None, :, None, None]
    j = np.arange(width)[None, None, :, None]
    c = np.arange(3)[None, None, None, :]
    return t, i, j, c


def _libero_camera(frames: int, height: int, width: int, offset: int) -> np.ndarray:
    t, i, j, c = _index_grid(frames, height, width)
    return ((offset + 16 * t + 4 * i + j + c) % 256).astype(np.uint8)


def _libero_actions(frames: int) -> np.ndarray:
    t = np.arange(frames)[:, None]
    d = np.arange(7)[None, :]
    return (0.05 * t + 0.1 * d).astype(np.float32)


def _libero_states(frames: int) -> np.ndarray:
    t = np.arange(frames)[:, None]
    d = np.arange(8)[None, :]
    return (0.25 * t + 0.5 * d).astype(np.float32)


def _robotwin_camera(
    frames: int, height: int, width: int, base: float, frame_scale: bool = True
) -> np.ndarray:
    t, i, j, c = _index_grid(frames, height, width)
    spatial = 0.001 * (3 * i + 5 * j + 7 * c)
    values = base * 0.25 * (1.0 + 0.2 * t) + spatial
    if frame_scale:
        scales = np.array([robotwin_frame_scale(t) for t in range(frames)])
        values = values * scales[:, None, None, None]
    return values.astype(np.float32)


def _robotwin_actions(frames: int) -> np.ndarray:
    t = np.arange(frames)[:, None]
    d = np.arange(14)[None, :]
    return (-1.0 + 0.15 * t + 0.05 * d).astype(np.float32)


def write_libero_stats(path: Path) -> Path:
    """Quantile statistics with hand-checkable values (q01=0, q99=1 for actions)."""
    stats = {
        "norm_stats": {
            "state": {
                "q01": [-1.0] * 8,
                "q99": [3.0] * 8,
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
            },
            "actions": {
                "q01": [0.0] * 7,
                "q99": [1.0] * 7,
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
            },
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats), encoding="utf-8")
    return path


def write_robotwin_stats(path: Path) -> Path:
    """Mean/std statistics with distinguishable action and state constants."""
    stats = {
        "norm_stats": {
            "state": {"mean": [-0.5] * 14, "std": [2.0] * 14},
            "actions": {"mean": [0.25] * 14, "std": [0.5] * 14},
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats), encoding="utf-8")
    return path


def write_libero_episode(
    root: Path,
    *,
    suite: str = SUITE,
    episode: str = "000018",
    variant: str = "variant_00",
    offset: int = 0,
    frames: int = FRAMES,
    height: int = HEIGHT,
    width: int = WIDTH,
    lighting_domain: str = "Normal",
    session_id: str | None = None,
    timestamps: np.ndarray | None = None,
) -> Path:
    path = root / "data" / "libero" / "raw" / suite / f"episode_{episode}" / f"{variant}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "task_description": f"synthetic {suite}",
        "lighting_domain": lighting_domain,
        "synthetic_fixture": True,
        "fixture_note": FIXTURE_NOTE,
    }
    if session_id is not None:
        metadata["session_id"] = session_id
    arrays: dict[str, np.ndarray] = {
        "agentview_raw_uint8": _libero_camera(frames, height, width, offset),
        "wrist_raw_uint8": _libero_camera(frames, height, width, offset + 7),
        "actions": _libero_actions(frames),
        "robot_states": _libero_states(frames),
        "states": np.zeros((frames, 92), dtype=np.float64),
        "rewards": np.zeros(frames, dtype=np.uint8),
        "dones": np.zeros(frames, dtype=np.uint8),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if timestamps is not None:
        arrays["timestamps"] = np.asarray(timestamps, dtype=np.float64)
    np.savez(path, **arrays)
    return path


def write_libero_rgb_twin(
    root: Path,
    *,
    suite: str = SUITE,
    episode: str = "000018",
    variant: str = "variant_00",
    frames: int = FRAMES,
) -> Path:
    path = root / "data" / "libero" / "rgb" / suite / f"episode_{episode}" / f"{variant}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((frames, HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    np.savez(
        path,
        agentview_rgb_uint8=rgb,
        wrist_rgb_uint8=rgb,
        actions=_libero_actions(frames),
        metadata_json=np.asarray(
            json.dumps({"synthetic_fixture": True, "fixture_note": FIXTURE_NOTE})
        ),
    )
    return path


def write_robotwin_episode(
    root: Path,
    *,
    task: str = "grab_roller",
    episode: str = "000018",
    frames: int = FRAMES,
    height: int = HEIGHT,
    width: int = WIDTH,
    frame_scale: bool = True,
    session_id: str | None = None,
    timestamps: np.ndarray | None = None,
) -> Path:
    path = root / "data" / "robotwin2" / "raw" / task / f"episode_{episode}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "task_name": "grab_roller",
        "synthetic_fixture": True,
        "fixture_note": FIXTURE_NOTE,
    }
    if session_id is not None:
        metadata["session_id"] = session_id
    actions = _robotwin_actions(frames)
    arrays: dict[str, np.ndarray] = {
        "head_camera_raw_float32": _robotwin_camera(
            frames, height, width, ROBOTWIN_BASE, frame_scale
        ),
        "left_camera_raw_float32": _robotwin_camera(frames, height, width, 0.37, frame_scale),
        "right_camera_raw_float32": _robotwin_camera(frames, height, width, 0.37, frame_scale),
        # Deliberately unusable sentinels: this array must never be read as joint state.
        "robot_states": np.full((frames, 16), 999.0, dtype=np.float32),
        "left_endpose": np.zeros((frames, 7), dtype=np.float32),
        "right_endpose": np.zeros((frames, 7), dtype=np.float32),
        "actions": actions,
        "rewards": np.zeros(frames, dtype=np.uint8),
        "dones": np.zeros(frames, dtype=np.uint8),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if timestamps is not None:
        arrays["timestamps"] = np.asarray(timestamps, dtype=np.float64)
    np.savez(path, **arrays)
    return path


def libero_dataset(root: Path, stats: Path, **overrides) -> RawVLASequenceDataset:
    kwargs = {
        "root": root,
        "dataset_format": "libero",
        "split": "all",
        "stats_path": stats,
        "window": 4,
        "observation_stride": 1,
        "noise_model": LIBERO_NOISE_MODEL,
        "range_evidence_path": RECORDED_RANGE_EVIDENCE,
    }
    kwargs.update(overrides)
    return RawVLASequenceDataset(**kwargs)


def robotwin_dataset(root: Path, stats: Path, **overrides) -> RawVLASequenceDataset:
    kwargs = {
        "root": root,
        "dataset_format": "robotwin2",
        "split": "all",
        "stats_path": stats,
        "window": 4,
        "action_horizon": 5,
    }
    kwargs.update(overrides)
    return RawVLASequenceDataset(**kwargs)


@pytest.fixture
def libero_root(tmp_path: Path) -> Path:
    root = tmp_path / "rawvla"
    write_libero_stats(tmp_path / "libero_norm_stats.json")
    write_libero_episode(root, episode="000018", variant="variant_00", offset=0)
    return root


@pytest.fixture
def robotwin_root(tmp_path: Path) -> Path:
    root = tmp_path / "rawvla"
    write_robotwin_stats(tmp_path / "policy_norm_stats.json")
    write_robotwin_episode(root)
    return root


def libero_stats_path(tmp_path: Path) -> Path:
    return tmp_path / "libero_norm_stats.json"


def robotwin_stats_path(tmp_path: Path) -> Path:
    return tmp_path / "policy_norm_stats.json"


# --------------------------------------------------------------------------------------
# (1) keys, shapes, unit scaling and hand-computed values
# --------------------------------------------------------------------------------------


def test_libero_keys_shapes_and_uint8_over_255(tmp_path: Path, libero_root: Path) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path))
    sample = dataset.sample_at(0, 6)
    assert sample.camera_ids == ["agentview", "wrist"]
    agentview = sample.window("agentview")
    wrist = sample.window("wrist")
    assert agentview.values.shape == (4, 3, HEIGHT, WIDTH)
    assert agentview.values.dtype == torch.float32
    assert agentview.frame_ids.tolist() == [3, 4, 5, 6]
    assert agentview.new_measurement.tolist() == [True, True, True, True]
    assert agentview.metadata["padded_leading_frames"] == 0
    assert agentview.camera_id == "agentview"
    assert agentview.metadata["camera_key"] == "agentview_raw_uint8"
    assert agentview.metadata["value_units"] == LIBERO_PROTOCOL.raw_unit
    assert agentview.metadata_available["timestamp"] is False
    assert agentview.metadata_available["saturation_level"] is True
    # hand-computed: last window frame is t=6 -> (16*6 + 4i + j + c)/255
    for i in range(HEIGHT):
        for j in range(WIDTH):
            for c in range(3):
                expected = libero_raw_value(6, i, j, c) / 255.0
                assert agentview.values[-1, c, i, j].item() == pytest.approx(expected, abs=1e-7)
    assert wrist.metadata["camera_key"] == "wrist_raw_uint8"
    assert wrist.values[-1, 0, 0, 0].item() == pytest.approx(
        libero_raw_value(6, 0, 0, 0, offset=7) / 255.0
    )
    # three-channel pseudo-RAW is co-located, and must not be renamed to Bayer
    assert torch.equal(agentview.sample_xy, torch.zeros((3, 2), dtype=torch.int64))
    assert agentview.metadata["co_located_channels"] is True
    assert agentview.metadata["sampling_geometry"] == "per_pixel_trichromatic_collocated"
    assert agentview.metadata["synthetic_fixture"] is True
    assert agentview.metadata["fixture_note"] == FIXTURE_NOTE
    # final frame of the fixture never saturates below 255
    assert agentview.saturated_mask.sum().item() == 0
    assert agentview.valid_mask.all()
    assert agentview.bad_mask.sum().item() == 0


def test_camera_subset_and_state_source_override_are_explicit(
    tmp_path: Path, libero_root: Path
) -> None:
    single = libero_dataset(
        libero_root, libero_stats_path(tmp_path), camera_keys=("agentview_raw_uint8",)
    )
    sample = single.sample_at(0, 2)
    assert sample.camera_ids == ["agentview"]
    assert single.spec()["camera_keys"] == ["agentview_raw_uint8"]
    # reusing the 92-dim simulator state is allowed but must stay flagged as a cache channel
    simulator_state = libero_dataset(
        libero_root, libero_stats_path(tmp_path), state_source_key="states"
    ).sample_at(0, 2)
    assert simulator_state.state_kind == "cached_channel:states"
    assert simulator_state.state.shape == (8,)
    assert simulator_state.meta["state_kind"] == "cached_channel:states"
    # uint8 digital numbers may not be passed through as if they were calibrated
    with pytest.raises(ValueError, match="must be divided by 255"):
        libero_dataset(libero_root, libero_stats_path(tmp_path), normalization="none")
    # a missing state key is reported with the available keys
    with pytest.raises(KeyError, match="state_source_key='joint_positions' is missing"):
        libero_dataset(
            libero_root, libero_stats_path(tmp_path), state_source_key="joint_positions"
        ).sample_at(0, 2)


def test_libero_uint8_scale_is_enforced(tmp_path: Path) -> None:
    root = tmp_path / "rawvla"
    write_libero_stats(tmp_path / "libero_norm_stats.json")
    path = write_libero_episode(root)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays["agentview_raw_uint8"] = arrays["agentview_raw_uint8"].astype(np.float32)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="refusing to guess a scale"):
        libero_dataset(root, libero_stats_path(tmp_path)).sample_at(0, 3)


def test_robotwin_shapes_and_float32_passthrough(tmp_path: Path, robotwin_root: Path) -> None:
    dataset = robotwin_dataset(robotwin_root, robotwin_stats_path(tmp_path))
    sample = dataset.sample_at(0, 5)
    assert sample.camera_ids == ["head_camera", "left_camera", "right_camera"]
    window = sample.window("head_camera")
    assert window.values.shape == (4, 3, HEIGHT, WIDTH)
    assert window.metadata["value_units"] == ROBOTWIN_PROTOCOL.raw_unit
    assert window.metadata["raw_scaling"] == "author_float32_passthrough"
    assert window.metadata["declared_calibration"] is None

    path = Path(dataset.episodes[0])
    with np.load(path, allow_pickle=False) as archive:
        stored = np.asarray(archive["head_camera_raw_float32"])
    frames = window.frame_ids.tolist()
    expected = torch.from_numpy(np.ascontiguousarray(stored[frames].transpose(0, 3, 1, 2)))
    # exact identity: no /255, no per-frame max normalisation
    assert torch.equal(window.values, expected)
    assert window.values.max().item() == pytest.approx(robotwin_raw_value(4, 3, 3, 2), rel=1e-6)
    assert window.values.max().item() < 1.0
    assert window.values.max().item() / 255.0 < 0.002
    # the odd frame really is 4x darker: a per-frame max scaling would erase that
    odd = window.frame_ids.tolist().index(3)
    assert window.values[odd, 2, 3, 3].item() == pytest.approx(
        robotwin_raw_value(3, 3, 3, 2), rel=1e-6
    )
    assert window.values[odd, 2, 3, 3].item() == pytest.approx(
        0.25 * (ROBOTWIN_BASE * 0.25 * (1.0 + 0.2 * 3) + 0.001 * (9 + 15 + 14)), rel=1e-6
    )
    # saturation level is not declared for the float32 chain -> explicit all-False mask
    assert window.saturated_mask.sum().item() == 0
    assert window.metadata_available["saturation_level"] is False


def test_robotwin_declared_calibration_is_applied(tmp_path: Path) -> None:
    root = tmp_path / "rawvla"
    write_robotwin_stats(tmp_path / "policy_norm_stats.json")
    path = write_robotwin_episode(root)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["raw_black_level"] = 0.0
    metadata["raw_white_level"] = 1.0
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)
    dataset = robotwin_dataset(root, robotwin_stats_path(tmp_path))
    window = dataset.sample_at(0, 2).window("head_camera")
    assert window.metadata["declared_calibration"]["mode"] == "declared_affine"
    raw = robotwin_raw_value(2, 1, 2, 0)
    assert window.values[-1, 0, 1, 2].item() == pytest.approx(raw, rel=1e-6)

    metadata["raw_white_level"] = None
    arrays["metadata_json"] = np.asarray(
        json.dumps({k: v for k, v in metadata.items() if v is not None})
    )
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="calibration is incomplete"):
        robotwin_dataset(root, robotwin_stats_path(tmp_path)).sample_at(0, 2)


# --------------------------------------------------------------------------------------
# (2) action indexing, action_offset and normalisation arithmetic
# --------------------------------------------------------------------------------------


def test_libero_action_window_and_quantile_normalisation(tmp_path: Path, libero_root: Path) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path), action_horizon=10)
    sample = dataset.sample_at(0, 3)
    # indices are clipped at the episode end (author protocol), the tail is masked
    assert sample.action_indices.tolist() == list(range(3, 12)) + [11]
    assert sample.action_mask.tolist() == [True] * 9 + [False]
    assert sample.action_raw.shape == (10, 7)
    for step, t in enumerate(range(3, 12)):
        for d in range(7):
            assert sample.action_raw[step, d].item() == pytest.approx(libero_action(t, d), rel=1e-6)
    # last step repeats the final in-range action but is masked out of the loss
    assert sample.action_raw[-1, 0].item() == pytest.approx(libero_action(11, 0), rel=1e-6)
    # hand-computed quantile min-max: q01=0, q99=1 -> clip(2x-1)
    for step, t in enumerate(range(3, 12)):
        for d in range(7):
            expected = max(-1.0, min(1.0, 2.0 * libero_action(t, d) - 1.0))
            assert sample.action[step, d].item() == pytest.approx(expected, abs=1e-6)
    # action_offset must shift the supervision window
    shifted = libero_dataset(
        libero_root, libero_stats_path(tmp_path), action_horizon=10, action_offset=2
    ).sample_at(0, 3)
    assert shifted.action_indices.tolist() == list(range(5, 12)) + [11, 11, 11]
    assert shifted.action_raw[0, 0].item() == pytest.approx(libero_action(5, 0), rel=1e-6)


def test_libero_state_uses_robot_states_not_simulator_states(
    tmp_path: Path, libero_root: Path
) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path))
    sample = dataset.sample_at(0, 4)
    assert sample.state is not None and sample.state.shape == (8,)
    assert sample.state_kind == "measured_proprio"
    assert sample.state_available is True
    assert sample.proprio_available is True
    for d in range(8):
        raw = libero_state(4, d)
        expected = max(-1.0, min(1.0, 2.0 * (raw - (-1.0)) / (3.0 - (-1.0)) - 1.0))
        assert sample.state[d].item() == pytest.approx(expected, abs=1e-6)
        assert sample.state_raw[d].item() == pytest.approx(raw, rel=1e-6)
    # the 92-dim simulator 'states' array is never used
    assert "states" in dataset._load_episode(dataset.episodes[0])
    assert sample.meta["state_kind"] == "measured_proprio"


def test_robotwin_action_offset_and_zscore(tmp_path: Path, robotwin_root: Path) -> None:
    dataset = robotwin_dataset(robotwin_root, robotwin_stats_path(tmp_path), action_horizon=5)
    sample = dataset.sample_at(0, 4)
    # author protocol: supervision starts at actions[t+1]
    assert sample.action_indices.tolist() == [5, 6, 7, 8, 9]
    assert sample.action_mask.all()
    for step, t in enumerate(range(5, 10)):
        for d in range(14):
            raw = robotwin_action(t, d)
            assert sample.action_raw[step, d].item() == pytest.approx(raw, rel=1e-6)
            expected = (raw - 0.25) / 0.5
            assert sample.action[step, d].item() == pytest.approx(expected, abs=1e-6)
    # tail padding: t = N-3 with horizon 5 -> two supervised steps then a masked repeat
    tail = dataset.sample_at(0, FRAMES - 3)
    assert tail.action_indices.tolist() == [
        FRAMES - 2,
        FRAMES - 1,
        FRAMES - 1,
        FRAMES - 1,
        FRAMES - 1,
    ]
    assert tail.action_mask.tolist() == [True, True, False, False, False]
    # zero offset must move the window onto actions[t]
    zero = robotwin_dataset(
        robotwin_root, robotwin_stats_path(tmp_path), action_horizon=5, action_offset=0
    ).sample_at(0, 4)
    assert zero.action_indices.tolist() == [4, 5, 6, 7, 8]


def test_robotwin_state_is_the_action_channel_not_robot_states(
    tmp_path: Path, robotwin_root: Path
) -> None:
    dataset = robotwin_dataset(robotwin_root, robotwin_stats_path(tmp_path))
    sample = dataset.sample_at(0, 4)
    assert sample.state is not None and sample.state.shape == (14,)
    assert sample.state_kind == "cached_joint_target_channel"
    assert sample.proprio_available is False
    for d in range(14):
        raw = robotwin_action(4, d)
        assert sample.state_raw[d].item() == pytest.approx(raw, rel=1e-6)
        # state statistics are mean=-0.5, std=2.0, unlike the action statistics
        assert sample.state[d].item() == pytest.approx((raw + 0.5) / 2.0, abs=1e-6)
    # the 16-dim robot_states sentinels never appear anywhere in the output
    assert float(sample.state.abs().max()) < 100.0
    assert dataset.state_source_key == "actions"
    assert dataset.protocol.state_note.startswith("actions[t][:14]")


def test_delta_action_indices_subtract_the_current_state(
    tmp_path: Path, robotwin_root: Path
) -> None:
    dataset = robotwin_dataset(
        robotwin_root,
        robotwin_stats_path(tmp_path),
        action_horizon=5,
        delta_action_indices=(0, 1, 2),
    )
    sample = dataset.sample_at(0, 4)
    for step, t in enumerate(range(5, 10)):
        for d in range(3):
            delta = robotwin_action(t, d) - robotwin_action(4, d)
            expected = (delta - 0.25) / 0.5
            assert sample.action[step, d].item() == pytest.approx(expected, abs=1e-6)
        # untouched dims keep the plain z-score
        assert sample.action[step, 5].item() == pytest.approx(
            (robotwin_action(t, 5) - 0.25) / 0.5, abs=1e-6
        )


# --------------------------------------------------------------------------------------
# (3) repeated padding frames are not new measurements
# --------------------------------------------------------------------------------------


def test_leading_padding_frames_are_not_new_measurements(tmp_path: Path, libero_root: Path) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path))
    window = dataset.sample_at(0, 1).window("agentview")
    assert window.frame_ids.tolist() == [0, 0, 0, 1]
    assert window.new_measurement.tolist() == [True, False, False, True]
    assert window.unique_frame_count() == 2
    assert window.metadata["padded_leading_frames"] == 2
    # duplicated frames carry identical measurements of the same physical frame
    assert torch.equal(window.values[0], window.values[1])
    assert torch.equal(window.values[1], window.values[2])
    assert not torch.equal(window.values[0], window.values[3])
    # a mask-weighted mean equals the mean over unique frames, and differs from the
    # naive mean that would count the padded repeat three times
    weights = window.measurement_weights()
    assert weights.tolist() == [1.0, 0.0, 0.0, 1.0]
    weighted = (window.values * weights.view(-1, 1, 1, 1)).sum(0) / weights.sum()
    unique = window.values[[0, 3]].mean(0)
    naive = window.values.mean(0)
    assert torch.allclose(weighted, unique, atol=1e-7)
    assert not torch.allclose(naive, weighted, atol=1e-4)


def test_window_start_always_resets_and_action_tail_is_masked(
    tmp_path: Path, libero_root: Path
) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path), action_horizon=10)
    for frame in (0, 1, 4, FRAMES - 1):
        sample = dataset.sample_at(0, frame)
        assert sample.meta["new_measurement"][0] is True
        for window in sample.windows.values():
            assert window.new_measurement[0].item() is True
            assert window.metadata["reset_required_at_window_start"] is True
    tail = dataset.sample_at(0, FRAMES - 8)
    assert tail.action_mask.tolist() == [True] * 8 + [False, False]
    # masked action steps do not contribute to a masked loss
    squared = (tail.action - tail.action_raw).pow(2).sum(-1)
    masked = (squared * tail.action_mask).sum() / tail.action_mask.sum()
    unmasked = squared.mean()
    assert masked.item() != pytest.approx(unmasked.item())


def test_rollout_plan_carries_state_only_inside_one_episode(
    tmp_path: Path, libero_root: Path
) -> None:
    write_libero_episode(libero_root, episode="000019", variant="variant_00", offset=5)
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path))
    plans = dataset.plan_rollout(0, start_frame=0, windows=3, stride=1)
    assert [plan.current_frame for plan in plans] == [0, 1, 2]
    assert plans[0].reset_required is True
    assert plans[1].reset_required is False and plans[1].continues_within_episode is True
    assert all(plan.episode_index == 0 for plan in plans)
    assert all(max(plan.window_frames) < FRAMES for plan in plans)
    other = dataset.plan_rollout(1, start_frame=0, windows=2, stride=1)
    assert other[0].reset_required is True
    assert other[0].episode_key.endswith("episode_000019/variant_00.npz")
    # a plan never steps across the end of the episode
    edge = dataset.plan_rollout(0, start_frame=FRAMES - 1, windows=3, stride=1)
    assert len(edge) == 1
    assert max(edge[0].window_frames) == FRAMES - 1


# --------------------------------------------------------------------------------------
# (4) deterministic, source-aware splits
# --------------------------------------------------------------------------------------


def test_source_group_keeps_variants_and_rgb_twins_together() -> None:
    raw = source_group(
        f"data/libero/raw/{SUITE}/episode_000018/variant_00.npz", dataset_format="libero"
    )
    rgb = source_group(
        f"data/libero/rgb/{SUITE}/episode_000018/variant_00.npz", dataset_format="libero"
    )
    other_variant = source_group(
        f"data/libero/raw/{SUITE}/episode_000018/variant_01.npz", dataset_format="libero"
    )
    lighting = source_group(
        f"data/libero/raw/{SUITE}/episode_000018/variant_02_lighting_Low.npz",
        dataset_format="libero",
    )
    assert raw == rgb == other_variant == lighting == f"{SUITE}/episode_000018"
    robotwin = source_group(
        "data/robotwin2/raw/grab_roller/episode_000018.npz", dataset_format="robotwin2"
    )
    assert robotwin == "grab_roller/episode_000018"
    assert assign_side(raw, split_version="v1") == assign_side(rgb, split_version="v1")


def test_split_never_crosses_an_episode_or_its_variants(tmp_path: Path) -> None:
    root = tmp_path / "rawvla"
    write_libero_stats(tmp_path / "libero_norm_stats.json")
    paths = []
    for episode in range(6):
        for variant, offset in (("variant_00", 0), ("variant_01", 11)):
            paths.append(
                write_libero_episode(
                    root,
                    episode=f"{episode:06d}",
                    variant=variant,
                    offset=offset,
                    lighting_domain="Low" if variant.endswith("01") else "Normal",
                )
            )
    rgb_twin = write_libero_rgb_twin(root, episode="000000", variant="variant_00")
    dataset = RawVLASequenceDataset(
        root=root,
        dataset_format="libero",
        split="train",
        stats_path=libero_stats_path(tmp_path),
        window=4,
        observation_stride=1,
        train_fraction=0.5,
    )
    manifest = dataset.split
    sides = manifest["sides"]
    assert manifest["counts"]["files"] == len(paths)
    assert sorted(sides["train"] + sides["validation"]) == sorted(str(p) for p in paths)
    assert sides["train"] and sides["validation"]
    parents = set()
    for entry in manifest["source_groups"]:
        directories = {Path(file).parent for file in entry["files"]}
        assert len(directories) == 1
        assert len(entry["files"]) == 2
        assert entry["group"].endswith(Path(entry["files"][0]).parent.name)
        parents.add(Path(entry["files"][0]).parent.name)
    assert len(parents) == 6
    # the RGB twin shares the group key and the side of its RAW partner, so it cannot leak
    raw_group = source_group(Path(paths[0]).relative_to(root), dataset_format="libero")
    rgb_group = source_group(Path(rgb_twin).relative_to(root), dataset_format="libero")
    assert raw_group == rgb_group == f"{SUITE}/episode_000000"
    raw_side = dataset._split_of(str(paths[0]))
    assert raw_side in {"train", "validation"}
    assert (
        assign_side(raw_group, split_version=dataset.split_version, train_fraction=0.5) == raw_side
    )
    # deterministic across constructions
    again = RawVLASequenceDataset(
        root=root,
        dataset_format="libero",
        split="train",
        stats_path=libero_stats_path(tmp_path),
        window=4,
        observation_stride=1,
        train_fraction=0.5,
    )
    assert again.split["sides"] == sides
    assert again.paths == dataset.paths
    assert len(dataset) == len(dataset.paths) * 32
    written = dataset.write_split_manifest(tmp_path / "splits" / "n01.json")
    assert load_split_manifest(written)["sides"] == sides


def test_split_uses_declared_source_identity_when_available(tmp_path: Path) -> None:
    root = tmp_path / "rawvla"
    write_libero_stats(tmp_path / "libero_norm_stats.json")
    for episode in range(4):
        write_libero_episode(
            root,
            episode=f"{episode:06d}",
            variant="variant_00",
            offset=episode,
            session_id="session-A" if episode < 2 else f"session-{episode}",
        )
    dataset = RawVLASequenceDataset(
        root=root,
        dataset_format="libero",
        split="all",
        stats_path=libero_stats_path(tmp_path),
        window=4,
        observation_stride=1,
        train_fraction=0.5,
    )
    path_based = dataset.split["counts"]["groups"]
    dataset.rebuild_split(use_metadata=True)
    assert dataset.split["counts"]["groups"] == path_based - 1
    shared = [
        entry for entry in dataset.split["source_groups"] if entry["group"] == "libero:session-A"
    ]
    assert len(shared) == 1 and len(shared[0]["files"]) == 2
    assert dataset.split["counts"]["files"] == 4
    # sorted() is byte order, so the numeric session tags come before "session-A"
    assert dataset.observed_identity_fields()["session_id"] == [
        "session-2",
        "session-3",
        "session-A",
    ]


def test_manifest_hash_rule_is_reproducible(tmp_path: Path) -> None:
    manifest = build_split_manifest(
        ["a/episode_1.npz", "a/episode_2.npz"], dataset_format="libero", train_fraction=0.5
    )
    for entry in manifest["source_groups"]:
        assert entry["side"] == assign_side(entry["group"], train_fraction=0.5)
    written = write_split_manifest(tmp_path / "manifest.json", manifest)
    assert load_split_manifest(written) == manifest


# --------------------------------------------------------------------------------------
# (5) CFA rearrangement: coordinates, black/white levels and masks
# --------------------------------------------------------------------------------------


def _mosaic(values: np.ndarray) -> CfaMosaic:
    return CfaMosaic(values=torch.tensor(values, dtype=torch.float32), pattern="RGGB")


def _ramp_mosaic(rows: int = 6, cols: int = 6) -> np.ndarray:
    return np.array([[64 + 10 * r + c for c in range(cols)] for r in range(rows)], dtype=np.float32)


def test_cfa_planes_keep_real_offsets_and_calibration() -> None:
    mosaic = _mosaic(_ramp_mosaic())
    stack = mosaic.split(black_level=64.0, white_level=1023.0)
    span = 1023.0 - 64.0
    assert stack.plane_count == 4
    assert stack.colors == CFA_PATTERNS["RGGB"] == ("R", "G", "G", "B")
    assert stack.sample_xy.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert stack.global_sample_xy.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert stack.co_located is False
    assert stack.metadata["calibration"] == "x=(raw-black)/(white-black)"
    # every plane reads the mosaic pixel at its own (row, col) offset
    ramp = _ramp_mosaic()
    expected = {
        (0, 0): (64 - 64) / span,
        (0, 1): (65 - 64) / span,
        (1, 0): (74 - 64) / span,
        (1, 1): (75 - 64) / span,
    }
    for plane, (dr, dc) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        assert stack.values[plane, 0, 0].item() == pytest.approx(expected[(dr, dc)], abs=1e-7)
        assert stack.values[plane, 1, 1].item() == pytest.approx(
            (ramp[2 + dr, 2 + dc] - 64) / span, abs=1e-7
        )
    assert stack.white_level.tolist() == [1023.0] * 4
    assert stack.black_level.tolist() == [64.0] * 4


def test_cfa_masks_stay_separate_and_invalid_values_are_not_measurements() -> None:
    values = _ramp_mosaic()
    missing = np.zeros_like(values, dtype=bool)
    missing[0, 1] = True
    bad = np.zeros_like(values, dtype=bool)
    bad[2, 2] = True
    mosaic = CfaMosaic(
        values=torch.tensor(values),
        pattern="RGGB",
        missing=torch.tensor(missing),
        bad=torch.tensor(bad),
    )
    stack = mosaic.split(black_level=64.0, white_level=1023.0, saturation_level=104.0)
    # missing pixel (0,1) lands on plane 1 at (0,0); bad pixel (2,2) on plane 0 at (1,1)
    assert stack.missing[1, 0, 0].item() is True
    assert stack.bad[0, 1, 1].item() is True
    assert stack.valid[1, 0, 0].item() is False
    assert stack.valid[0, 1, 1].item() is False
    # raw >= 104 saturates: three pixels in each plane
    assert stack.saturated.sum().item() == 12
    assert torch.equal(stack.valid, ~(stack.missing | stack.bad | stack.saturated))
    # filled values are placeholders, never measurements
    assert stack.values[1, 0, 0].item() == 0.0
    assert stack.metadata["saturation_rule"] == "raw >= declared saturation_level"
    assert stack.metadata["saturation_level_available"] is True
    # an unknown full-well level must be declared, not assumed
    unknown = mosaic.split(black_level=64.0, white_level=1023.0, saturation_level=math.inf)
    assert unknown.saturated.sum().item() == 0
    assert unknown.metadata["saturation_level_available"] is False
    with pytest.raises(ValueError, match="must be finite"):
        mosaic.split(black_level=64.0, white_level=math.inf)


def test_cfa_crop_and_shift_move_coordinates_with_the_samples() -> None:
    mosaic = _mosaic(_ramp_mosaic())
    span = 1023.0 - 64.0
    # odd sensor crop: the phase flips, so plane 0 samples blue instead of red
    cropped = mosaic.crops(top=1, left=1, height=4, width=4)
    stack = cropped.split(black_level=64.0, white_level=1023.0)
    assert cropped.geometry().origin == (1, 1)
    assert stack.sample_xy.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert stack.global_sample_xy.tolist() == [[1, 1], [1, 2], [2, 1], [2, 2]]
    assert stack.colors == ("B", "G", "G", "R")
    assert stack.values[0, 0, 0].item() == pytest.approx((75 - 64) / span, abs=1e-7)
    assert stack.values[3, 0, 0].item() == pytest.approx((86 - 64) / span, abs=1e-7)
    # plane-space crop: two plane pixels per sensor step, so the phase is unchanged
    plane_crop = crop_planes(stack, top=1, left=1)
    assert plane_crop.geometry.origin == (3, 3)
    assert plane_crop.colors == ("B", "G", "G", "R")
    assert plane_crop.values[0, 0, 0].item() == pytest.approx((97 - 64) / span, abs=1e-7)
    assert plane_crop.plane_shape() == (1, 1)
    # sensor shift marks vacated samples missing instead of inventing measurements
    shifted = mosaic.shifted(dy=1, dx=0)
    assert shifted.geometry().origin == (1, 0)
    assert shifted.values[0, 0].item() == pytest.approx(74.0)
    assert shifted.missing[-1].all()
    shifted_stack = shifted.split(black_level=64.0, white_level=1023.0)
    assert shifted_stack.colors == ("G", "B", "R", "G")
    # planes 2 and 3 sample the vacated last mosaic row: masked out, never measurements
    assert shifted_stack.valid[2:, -1, :].sum().item() == 0
    assert shifted_stack.values[2:, -1, :].abs().sum().item() == 0.0
    assert shifted_stack.valid[:2, -1, :].all()
    # plane-space shift is the even-step equivalent and moves the origin by 2
    shifted_planes = shift_planes(stack, dy=1, dx=0)
    assert shifted_planes.geometry.origin == (3, 1)
    assert shifted_planes.missing[:, -1, :].all()
    assert shifted_planes.colors == ("B", "G", "G", "R")
    # content previously at plane row 1 is now at plane row 0
    assert torch.equal(shifted_planes.values[:, 0, :], stack.values[:, 1, :])


def test_cfa_odd_crop_cannot_be_split_without_an_even_frame() -> None:
    mosaic = _mosaic(_ramp_mosaic())
    odd = mosaic.crops(top=1, left=0, height=5, width=6)
    with pytest.raises(ValueError, match="even-sized frame"):
        odd.split(black_level=64.0, white_level=1023.0)


def test_cfa_window_keeps_four_planes_apart() -> None:
    stack = _mosaic(_ramp_mosaic()).split(black_level=64.0, white_level=1023.0)
    window = measurement_window_from_cfa(stack, camera_id="cfa0", frame_id=7)
    assert window.values.shape == (1, 4, 3, 3)
    assert window.camera_id == "cfa0"
    assert window.sample_xy.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert window.metadata["co_located"] is False
    assert window.metadata_available["co_located_channels"] is False
    assert window.metadata["plane_colors"] == ["R", "G", "G", "B"]
    assert window.frame_ids.tolist() == [7]
    assert window.time_unit == "frame"
    assert window.timestamp.tolist() == [7.0]
    second = measurement_window_from_cfa(
        stack, camera_id="cfa0", frame_id=8, timestamp=0.5, time_unit="second"
    )
    assert second.time_unit == "second" and second.timestamp.tolist() == [0.5]
    with pytest.raises(ValueError, match="do not invent seconds"):
        measurement_window_from_cfa(stack, time_unit="second")


def test_cfa_noise_variance_and_fitted_estimator() -> None:
    x = torch.tensor([0.0, 0.5, -0.25, 2.0], dtype=torch.float32)
    variance = shot_read_variance(x, shot=4.0e-4, read_var=1.0e-5)
    expected = [1.0e-5, 4.0e-4 * 0.5 + 1.0e-5, 1.0e-5, 4.0e-4 * 2.0 + 1.0e-5]
    assert variance.tolist() == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="non-negative"):
        shot_read_variance(x, shot=-1.0, read_var=0.0)

    def noisy_stack(shot: float, seed: int):
        rng = np.random.default_rng(seed)
        grid = np.array(
            [[64 + 8 * r + 4 * c for c in range(64)] for r in range(64)], dtype=np.float64
        )
        x_signal = (grid - 64.0) / 959.0
        variance_dn = (shot * x_signal + 1.0e-5) * 959.0**2
        noisy = grid + rng.normal(0.0, np.sqrt(variance_dn), size=grid.shape)
        return CfaMosaic(values=torch.tensor(noisy, dtype=torch.float32)).split(
            black_level=64.0, white_level=1023.0
        )

    low = estimate_noise_model(noisy_stack(1.0e-3, 0))
    high = estimate_noise_model(noisy_stack(1.0e-2, 0))
    assert low.estimator is True and low.fitted is True
    assert low.shot > 0.0 and low.read_var >= 0.0
    assert high.shot > low.shot
    evidence = low.as_dict()["evidence"]
    assert evidence["assumption"].startswith("independent noise")
    assert len(evidence["bins"]) >= 2
    with pytest.raises(ValueError, match="not enough valid"):
        estimate_noise_model(
            CfaMosaic(values=torch.zeros(4, 4)).split(black_level=0.0, white_level=1.0),
            min_samples=1000,
        )


# --------------------------------------------------------------------------------------
# (6) missing assets produce actionable errors, never fake data
# --------------------------------------------------------------------------------------


def test_missing_snapshot_reports_every_required_asset(tmp_path: Path) -> None:
    absent = tmp_path / "absent-rawvla-root"
    stats = tmp_path / "absent-norm-stats.json"
    with pytest.raises(MissingRawVLAAssetError) as excinfo:
        RawVLASequenceDataset(root=absent, dataset_format="libero", stats_path=stats)
    message = str(excinfo.value)
    for fragment in (
        str(absent),
        str(absent / "data" / "libero" / "raw"),
        str(stats),
        "agentview_raw_uint8",
        "wrist_raw_uint8",
        "raw_cache",
        "stats",
        "RAWVLA_ROOT",
        "synthetic_fixture",
    ):
        assert fragment in message, fragment
    assert {item.kind for item in excinfo.value.missing} == {"root", "raw_cache", "stats"}
    assert excinfo.value.args[0].startswith("RawVLA assets for dataset_format='libero'")

    requirements = rawvla_asset_requirements(root=absent, dataset_format="libero", stats_path=stats)
    by_kind = {item.kind: item for item in requirements}
    assert by_kind["root"].satisfied is False
    assert by_kind["raw_cache"].satisfied is False
    assert by_kind["stats"].required is True and by_kind["stats"].satisfied is False
    assert by_kind["rgb_cache"].required is False
    assert by_kind["root"].env_var is None  # the path was given explicitly
    default_root_assets = rawvla_asset_requirements(dataset_format="libero", stats_path=stats)
    assert {item.kind: item.env_var for item in default_root_assets}["root"] == "RAWVLA_ROOT"
    assert all(item.satisfied is False for item in default_root_assets if item.required)


def test_missing_robotwin_snapshot_names_its_own_keys(tmp_path: Path) -> None:
    absent = tmp_path / "absent-rawvla-root"
    with pytest.raises(MissingRawVLAAssetError) as excinfo:
        RawVLASequenceDataset(root=absent, dataset_format="robotwin2")
    message = str(excinfo.value)
    for fragment in (
        "head_camera_raw_float32",
        "left_camera_raw_float32",
        "right_camera_raw_float32",
        "data/robotwin2/raw",
    ):
        assert fragment in message, fragment
    assert "policy_norm_stats.json" in message
    assert "No RawVLA data was loaded" in message


def test_present_cache_with_absent_statistics_reports_only_statistics(
    tmp_path: Path, libero_root: Path
) -> None:
    stats = tmp_path / "missing-stats.json"
    with pytest.raises(MissingRawVLAAssetError) as excinfo:
        RawVLASequenceDataset(root=libero_root, dataset_format="libero", stats_path=stats)
    message = str(excinfo.value)
    assert {item.kind for item in excinfo.value.missing} == {"stats"}
    assert str(stats) in message
    assert "[raw_cache]" not in message


def test_unknown_dataset_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown RawVLA dataset_format"):
        RawVLASequenceDataset(root=tmp_path, dataset_format="libero_raw")


def test_default_root_points_at_the_absent_server_snapshot() -> None:
    assert default_rawvla_root() == Path("/workspace/owac/rawvla")
    assert not default_rawvla_root().exists()


# --------------------------------------------------------------------------------------
# provenance, statistics and reporting
# --------------------------------------------------------------------------------------


def test_fixture_provenance_is_declared_and_not_reported_as_real(
    tmp_path: Path, libero_root: Path
) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path))
    assert dataset.spec()["real_data_available"] is None  # not scanned yet, not claimed
    stats = dataset.stats(max_files=4)
    assert stats["provenance"]["synthetic_fixture_declared"] == 1
    assert stats["provenance"]["synthetic_fixture_all_scanned"] is True
    assert stats["provenance"]["episodes"][0]["synthetic_fixture"] is True
    assert stats["real_data_available"] is False
    assert stats["synthetic_fixture_episodes"] == [dataset.episodes[0]]
    assert stats["ranges"]["complete"] is True
    assert stats["ranges"]["keys"]["agentview_raw_uint8"]["max"] == 193.0
    assert stats["ranges"]["keys"]["agentview_raw_uint8"]["min"] == 0.0
    assert stats["ranges"]["keys"]["agentview_raw_uint8"]["dtype"] == "uint8"
    assert stats["ranges"]["keys"]["robot_states"]["max"] == pytest.approx(6.25, rel=1e-6)
    assert stats["data_present"] is True
    assert stats["executed_action_available"] is False
    assert stats["official_closed_loop_manifest"]["used"] is False
    assert stats["supported_input_modes"]["bayer_cfa"].startswith("owac.raw.cfa")
    assert stats["camera_dtype"] == "uint8"
    assert stats["raw_value_policy"]["mode"] == "uint8_to_float32_over_255"
    assert stats["state_source_key"] == "robot_states"
    assert stats["action_dim"] == 7 and stats["action_horizon"] == 10
    with pytest.raises(ValueError, match="synthetic fixture"):
        libero_dataset(
            libero_root, libero_stats_path(tmp_path), fixture_provenance="real"
        ).sample_at(0, 0)
    unlabelled = tmp_path / "unlabelled"
    write_libero_stats(tmp_path / "libero_norm_stats.json")
    path = write_libero_episode(unlabelled, episode="000018", variant="variant_00")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.pop("synthetic_fixture")
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="does not declare"):
        libero_dataset(
            unlabelled, libero_stats_path(tmp_path), fixture_provenance="synthetic"
        ).sample_at(0, 0)


def test_executed_action_is_never_fabricated(
    tmp_path: Path, libero_root: Path, robotwin_root: Path
) -> None:
    libero = libero_dataset(libero_root, libero_stats_path(tmp_path)).sample_at(0, 4)
    assert libero.executed_action is None
    assert libero.executed_action_available is False
    assert libero.window("agentview").executed_action is None
    assert libero.window("agentview").executed_action_available is False
    assert "not a measured executed action" in libero.meta["executed_action_note"]
    context = libero.as_embodiment_context()
    assert context.executed_action is None
    assert bool(context.executed_action_available.item()) is False
    assert bool(context.proprio_available.item()) is True
    robotwin = robotwin_dataset(robotwin_root, robotwin_stats_path(tmp_path)).sample_at(0, 4)
    assert robotwin.executed_action is None and robotwin.executed_action_available is False
    assert robotwin.proprio_available is False


def test_measurement_window_matches_the_frozen_contract(tmp_path: Path, libero_root: Path) -> None:
    window = (
        libero_dataset(libero_root, libero_stats_path(tmp_path)).sample_at(0, 2).window("agentview")
    )
    contract_fields = {field.name for field in dataclasses.fields(MeasurementFrame)}
    window_fields = {field.name for field in dataclasses.fields(MeasurementWindow)}
    assert contract_fields <= window_fields
    assert window.frame_ids.tolist() == [0, 0, 1, 2]
    frame = window.to_measurement_frame(3)
    assert isinstance(frame, MeasurementFrame)
    assert frame.values.shape == (1, 3, HEIGHT, WIDTH)
    assert frame.valid_mask.shape == (1, 3, HEIGHT, WIDTH)
    assert frame.episode_ids.tolist() == [0]
    assert frame.frame_ids.tolist() == [2]
    assert frame.camera_id == "agentview"
    assert frame.time_unit == "frame"
    assert frame.new_measurement.tolist() == [True]
    assert window.to_measurement_frame(1).new_measurement.tolist() == [False]
    assert frame.sample_xy.tolist() == [[0, 0]] * 3
    assert frame.metadata["episode_key"].endswith("variant_00.npz")
    with pytest.raises(IndexError, match="outside the window"):
        window.to_measurement_frame(99)


def test_time_unit_uses_declared_seconds_only(tmp_path: Path) -> None:
    root = tmp_path / "rawvla"
    write_libero_stats(tmp_path / "libero_norm_stats.json")
    write_libero_episode(root, episode="000018", variant="variant_00")
    write_libero_episode(
        root,
        episode="000019",
        variant="variant_00",
        offset=13,
        timestamps=np.arange(FRAMES) * 0.1,
    )
    dataset = libero_dataset(root, libero_stats_path(tmp_path))
    indexed = dataset.sample_at(0, 3).window("agentview")
    assert indexed.time_unit == "frame"
    assert indexed.timestamp.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert indexed.metadata_available["timestamp"] is False
    assert indexed.metadata_available["timestamp_source"] == "frame_index"
    timed = dataset.sample_at(1, 3).window("agentview")
    assert timed.time_unit == "second"
    assert timed.timestamp.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert timed.metadata_available["timestamp"] is True
    assert dataset.spec()["time_unit_observed"] == {"frame": 1, "second": 1}
    assert "never invented" in dataset.spec()["time_unit_policy"]


def test_noise_model_is_explicit_about_its_source(tmp_path: Path, libero_root: Path) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path))
    window = dataset.sample_at(0, 3).window("agentview")
    assert window.variance is not None
    expected = 4.0e-4 * window.values.clamp(min=0.0) + 1.0e-5
    assert torch.allclose(window.variance, expected, atol=1e-9)
    report = dataset.spec()
    assert report["noise_model"]["shot"] == 4.0e-4
    assert report["noise_model"]["estimator"] is False
    assert "DEFAULT_SENSOR_NOISE" in report["noise_model"]["evidence"]["source"]
    assert "quantisation variance is excluded" in report["noise_model"]["evidence"]["note"]


def test_robotwin_has_no_noise_model_until_one_is_supplied(
    tmp_path: Path, robotwin_root: Path
) -> None:
    dataset = robotwin_dataset(robotwin_root, robotwin_stats_path(tmp_path))
    window = dataset.sample_at(0, 2).window("head_camera")
    assert window.variance is None
    assert window.metadata_available["noise_model"] is False
    assert dataset.spec()["noise_model"] is None


# --------------------------------------------------------------------------------------
# real loss / backward and DataLoader integration
# --------------------------------------------------------------------------------------


def test_window_feeds_a_real_masked_loss_and_backward(tmp_path: Path, libero_root: Path) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path))
    window = dataset.sample_at(0, 2).window("agentview")
    values = window.values
    weights = window.measurement_weights()
    model = torch.nn.Conv2d(3, 3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    losses = []
    for _ in range(40):
        prediction = model(values[:-1])
        target = values[1:].detach()
        mask = weights[1:].view(-1, 1, 1, 1)
        residual = (prediction - target).pow(2)
        loss = (residual * mask).sum() / mask.sum().clamp(min=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]
    # the padded duplicate contributes no gradient: zeroing its weight changes nothing
    assert weights.tolist() == [1.0, 0.0, 1.0, 1.0]
    grads = [parameter.grad for parameter in model.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert any(float(grad.abs().sum()) > 0.0 for grad in grads if grad is not None)


def test_action_supervision_supports_a_real_masked_regression_loss(
    tmp_path: Path, robotwin_root: Path
) -> None:
    sample = robotwin_dataset(robotwin_root, robotwin_stats_path(tmp_path)).sample_at(0, FRAMES - 3)
    head = torch.nn.Linear(14, 14)
    target = sample.action.detach()
    mask = sample.action_mask
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    history = []
    for _ in range(30):
        prediction = head(sample.state.detach()).unsqueeze(0).expand(target.shape[0], -1)
        loss = ((prediction - target).pow(2).sum(-1) * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    assert math.isfinite(history[0]) and history[0] > 0.0
    assert history[-1] < history[0]
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    # masked-out tail steps carried the repeated action, so they must not be supervised
    assert mask.tolist() == [True, True, False, False, False]


def test_dataloader_collation_batches_windows(tmp_path: Path, libero_root: Path) -> None:
    dataset = libero_dataset(libero_root, libero_stats_path(tmp_path), samples_per_episode=4)
    assert len(dataset) == 4
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_sequence_samples)
    batch = next(iter(loader))
    assert set(batch["windows"]) == {"agentview", "wrist"}
    agentview = batch["windows"]["agentview"]
    assert agentview.values.shape == (2, 4, 3, HEIGHT, WIDTH)
    assert agentview.new_measurement.shape == (2, 4)
    assert bool(agentview.new_measurement[:, 0].all())
    assert agentview.frame_ids.shape == (2, 4)
    assert agentview.time_unit == "frame"
    assert agentview.action_indices.shape == (2, 10)  # one per supervised action step
    assert batch["action"].shape[:2] == (2, 10)
    assert batch["action_mask"].shape == (2, 10)
    assert batch["executed_action"] is None
    assert batch["executed_action_available"].tolist() == [False, False]
    assert len(batch["meta"]) == 2
    single = collate_measurement_windows([dataset.sample_at(0, 2).window("agentview")])
    assert single.values.shape == (1, 4, 3, HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="one batch holds one camera"):
        collate_measurement_windows(
            [dataset.sample_at(0, 2).window("agentview"), dataset.sample_at(0, 2).window("wrist")]
        )


# --------------------------------------------------------------------------------------
# recorded deployment evidence for the RoboTwin "no /255" decision
# --------------------------------------------------------------------------------------


def test_recorded_deployment_ranges_justify_the_scaling_decision() -> None:
    if not RECORDED_RANGE_EVIDENCE.is_file():
        pytest.skip(f"recorded range evidence is absent: {RECORDED_RANGE_EVIDENCE}")
    evidence = recorded_range_evidence(RECORDED_RANGE_EVIDENCE)
    assert evidence["revision"] == "8586c594fff9ce38580e29745f074e82e96e74f7"
    libero = next(
        sample
        for sample in evidence["samples"]
        if "libero/raw" in sample["path"] and sample["path"].endswith("variant_00.npz")
    )
    assert libero["features"]["agentview_raw_uint8"]["dtype"] == "uint8"
    assert libero["features"]["agentview_raw_uint8"]["max"] == 8.0
    assert libero["features"]["robot_states"]["shape"][-1] == 8
    assert libero["features"]["actions"]["shape"][-1] == 7
    robotwin = next(sample for sample in evidence["samples"] if "robotwin2/raw" in sample["path"])
    head = robotwin["features"]["head_camera_raw_float32"]
    assert head["dtype"] == "float32"
    assert head["shape"] == [88, 240, 320, 3]
    assert head["max"] == pytest.approx(0.4331582486629486, rel=1e-9)
    assert head["min"] == 0.0
    assert robotwin["features"]["robot_states"]["shape"][-1] == 16
    assert robotwin["features"]["actions"]["shape"][-1] == 14
    # dividing the recorded float32 RAW by 255 would push the whole scene below 2e-3
    assert head["max"] / 255.0 < 0.002
    assert "0.4332" in ROBOTWIN_PROTOCOL.raw_range_evidence
    assert "max 8" in LIBERO_PROTOCOL.raw_range_evidence
    assert len(evidence["samples"]) == 4  # the four audited paired samples

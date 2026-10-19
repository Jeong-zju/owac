"""S1 测试夹具：按 S0 冻结的 schema 构造一个极小的合成数据集。

夹具不是研究数据，只用于验证接口、坐标、因果与恢复合同；真实训练在远端 S0 扩展上
执行。相机几何是显式构造的：相机 A 在单位位姿，相机 B 水平平移 ``baseline_m``，
深度常量，因此已知对应点的像素位移可以被独立计算。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

CAMERA_NAMES_LIBERO = ["agentview", "robot0_eye_in_hand"]
CAMERA_NAMES_ROBOTWIN = ["left_camera", "right_camera", "head_camera", "front_camera"]
BASELINE_M = 0.1
DEPTH_M = 1.0
FOCAL_PX = 50.0


def _K(width: int, height: int) -> np.ndarray:
    return np.array(
        [[FOCAL_PX, 0.0, (width - 1) / 2.0], [0.0, FOCAL_PX, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pose(dx: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = dx
    return pose


def _depth_map(height: int, width: int, *, invalid_rows: int = 4) -> np.ndarray:
    depth = np.full((height, width), DEPTH_M, dtype=np.float32)
    depth[:invalid_rows, :] = 0.0  # 无效深度区域（遮挡 / 无测量）
    return depth


def _write_paired_episode(
    root: Path,
    name: str,
    *,
    platform: str,
    camera_names: list[str],
    frames: int,
    shape_hw: tuple[int, int],
    source_trajectory_id: str,
    split: str,
    rng: np.random.Generator,
) -> None:
    height, width = shape_hw
    episode = root / name
    episode.mkdir(parents=True)
    record = {
        "schema": "n04-live-extension-v2",
        "source_trajectory_id": source_trajectory_id,
        "split": split,
        "metadata": {
            "platform": platform,
            "camera_names": camera_names,
            "pixel_coordinates": "integer_pixel_centres_top_left",
        },
        "frames": [
            {"file": f"frame-{index:04d}.npz", "index": index, "sim_time_s": 0.1 * index}
            for index in range(frames)
        ],
    }
    (episode / "record.json").write_text(json.dumps(record))
    n_cameras = len(camera_names)
    for index in range(frames):
        if platform == "libero":
            raw = rng.integers(0, 256, size=(n_cameras, height, width, 3), dtype=np.uint8)
        else:
            raw = rng.random((n_cameras, height, width, 3)).astype(np.float32)
        depth = np.stack([_depth_map(height, width) for _ in range(n_cameras)])
        poses = np.stack([_pose(0.0), _pose(BASELINE_M)] + [_pose(0.0)] * (n_cameras - 2))
        np.savez(
            episode / f"frame-{index:04d}.npz",
            online_rawvla_rgb3=raw,
            online_K=np.stack([_K(width, height) for _ in range(n_cameras)]),
            online_sim_time_s=np.array(0.1 * index),
            gt_depth_m=depth,
            gt_visible_support=np.ones((n_cameras, height, width), dtype=bool),
            gt_geometry_support=np.ones((n_cameras, height, width), dtype=bool),
            gt_T_W_from_C=poses,
        )


def _write_factor_root(root: Path, *, shape_hw: tuple[int, int], frames: int) -> None:
    height, width = shape_hw
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    sequences = []
    # 同一 seed 下写两个 factor 文件：它们共享 source_trajectory_id，只有变体名不同，
    # 用于回归 measurement ID / 缓存命名冲突（S1 导出复核发现的缺陷）。
    for seed, split in ((0, "train"), (1, "dev")):
        for factor in ("static", "camera"):
            name = f"seed-{seed:02d}-{factor}.npz"
            raw = rng.random((frames, 2, height, width, 3)).astype(np.float32)
            depth = np.stack(
                [
                    [_depth_map(height, width, invalid_rows=2) for _ in range(2)]
                    for _ in range(frames)
                ]
            )
            poses = np.stack([np.stack([_pose(0.0), _pose(BASELINE_M)]) for _ in range(frames)])
            np.savez(
                root / name,
                online_rawvla_rgb3=raw,
                input_K=np.stack(
                    [np.stack([_K(width, height), _K(width, height)]) for _ in range(frames)]
                ),
                input_frame_index=np.arange(frames, dtype=np.int64),
                gt_depth_m=depth,
                gt_T_W_from_C=poses,
            )
            sequences.append(
                {
                    "file": name,
                    "split": split,
                    "factor": factor,
                    "seed": seed,
                    "source_trajectory_id": f"n04ext/rendered-rig/seed-{seed}",
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "n04-rendered-factors-v1",
                "sequences": sequences,
                "frames_per_sequence": frames,
            }
        )
    )


@pytest.fixture(scope="session")
def synthetic_roots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """合成数据根：一个 LIBERO 源、一个 RoboTwin 源、两个受控渲染源（train/dev）。"""
    root = tmp_path_factory.mktemp("n04_s1_data")
    paired = root / "paired"
    paired.mkdir()
    rng = np.random.default_rng(20261014)
    _write_paired_episode(
        paired,
        "episode-000",
        platform="libero",
        camera_names=CAMERA_NAMES_LIBERO,
        frames=3,
        shape_hw=(64, 48),
        source_trajectory_id="libero:task:demo_0",
        split="train",
        rng=rng,
    )
    _write_paired_episode(
        paired,
        "episode-001",
        platform="robotwin2",
        camera_names=CAMERA_NAMES_ROBOTWIN,
        frames=3,
        shape_hw=(48, 64),
        source_trajectory_id="robotwin2:task:demo_0",
        split="train",
        rng=rng,
    )
    _write_paired_episode(
        paired,
        "episode-002",
        platform="libero",
        camera_names=CAMERA_NAMES_LIBERO,
        frames=2,
        shape_hw=(64, 48),
        source_trajectory_id="libero:task:demo_dev",
        split="dev",
        rng=rng,
    )
    factor = root / "factor"
    _write_factor_root(factor, shape_hw=(32, 32), frames=3)
    return {"paired_root": str(paired), "factor_root": str(factor)}


@pytest.fixture(scope="session")
def s1_config_payload(synthetic_roots: dict[str, str]) -> dict:
    config_path = Path(__file__).resolve().parents[3] / "configs/n04/full/s1_measurement.json"
    payload = json.loads(config_path.read_text())
    payload["data"]["paired_root"] = synthetic_roots["paired_root"]
    payload["data"]["factor_root"] = synthetic_roots["factor_root"]
    payload["training"]["batch_frames"] = 4
    payload["training"]["eval_batch_frames"] = 4
    payload["training"]["steps"] = 4
    payload["training"]["lr_grid_steps"] = 2
    payload["training"]["profile_steps"] = 2
    payload["training"]["eval_every"] = 2
    payload["training"]["checkpoint_every"] = 2
    payload["training"]["log_every"] = 1
    payload["training"]["amp"] = False
    payload["training"]["device"] = "cpu"
    return payload


@pytest.fixture(scope="session")
def s1_config(s1_config_payload: dict, tmp_path_factory: pytest.TempPathFactory):
    """通过真实 JSON 装载路径构造配置，顺带验证装载器本身。"""
    from owac.frameworks.n04.full_config import load_config

    path = tmp_path_factory.mktemp("n04_s1_config") / "s1_measurement.json"
    path.write_text(json.dumps(s1_config_payload))
    return load_config(path)

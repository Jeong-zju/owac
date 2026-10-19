"""N04 S0 manifest/inventory 的行为与边界测试。

fixture 按远端真实 NPZ 的布局与 dtype 重建（ZIP stored、``metadata_json`` 成员、
相机大数组 + 小数组），因此测试覆盖的读取路径与远端一致：header-only 盘点、按帧
seek 采样、深读检查、源轨迹分组与固定种子拆分。

远端 1773/632 的真实数字由实际运行给出，不在测试里用固定数字冒充。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from owac.raw import n04_manifest as manifest

LIBERO_CAMERAS = ("agentview_raw_uint8", "wrist_raw_uint8")
ROBOTWIN_CAMERAS = (
    "head_camera_raw_float32",
    "left_camera_raw_float32",
    "right_camera_raw_float32",
)


# ---------------------------------------------------------------------------
# fixture 构造
# ---------------------------------------------------------------------------


def _save_npz(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict | None, *, compressed=False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in arrays.items()}
    if metadata is not None:
        payload["metadata_json"] = np.array(json.dumps(metadata), dtype=str)
    saver = np.savez_compressed if compressed else np.savez
    with path.open("wb") as handle:
        saver(handle, **payload)


def libero_arrays(frames: int, seed: int, camera_frames: int | None = None, value: int = 7):
    generator = np.random.default_rng(seed)
    camera_frames = frames if camera_frames is None else camera_frames
    small = 3 if camera_frames == frames else 2
    small = min(small, camera_frames)
    images = {
        key: (generator.integers(0, 256, size=(camera_frames, 8, 8, 3))).astype(np.uint8)
        for key in LIBERO_CAMERAS
    }
    images[LIBERO_CAMERAS[0]][0, 0, 0, 0] = value
    images[LIBERO_CAMERAS[0]][0, 0, 0, 1] = 255
    states = np.zeros((frames, 10), dtype=np.float64)
    states[:, 0] = 0.75 + 0.05 * np.arange(frames)
    states[:, 1] = generator.normal(size=frames)
    return {
        **images,
        "robot_states": generator.normal(size=(frames, 8)).astype(np.float32),
        "actions": generator.normal(size=(frames, 7)).astype(np.float32),
        "states": states,
        "dones": np.zeros(frames, dtype=np.uint8),
        "rewards": np.zeros(frames, dtype=np.uint8),
    }


def robotwin_arrays(frames: int, seed: int, camera_frames: int | None = None):
    generator = np.random.default_rng(seed)
    camera_frames = frames if camera_frames is None else camera_frames
    norm = np.sqrt(0.5)
    left = np.zeros((frames, 7), dtype=np.float32)
    right = np.zeros((frames, 7), dtype=np.float32)
    left[:, 3:] = np.array([norm, 0.0, 0.0, norm], dtype=np.float32)
    right[:, 3:] = np.array([norm, 0.0, 0.0, -norm], dtype=np.float32)
    left[:, :3] = generator.normal(size=(frames, 3)).astype(np.float32)
    right[:, :3] = 0.5 + generator.normal(size=(frames, 3)).astype(np.float32)
    proprio = np.concatenate(
        [
            left,
            np.full((frames, 1), 0.5, dtype=np.float32),
            right,
            np.full((frames, 1), 1.0, dtype=np.float32),
        ],
        axis=1,
    )
    return {
        **{
            key: generator.normal(size=(camera_frames, 4, 4, 3)).astype(np.float32)
            for key in ROBOTWIN_CAMERAS
        },
        "robot_states": proprio,
        "actions": generator.normal(size=(frames, 14)).astype(np.float32),
        "left_endpose": left,
        "right_endpose": right,
        "dones": np.ones(frames, dtype=np.uint8),
        "rewards": np.ones(frames, dtype=np.uint8),
    }


def libero_metadata(
    dataset: str, task: str, demo: int, *, lighting: str = "Normal", frames: int = 6
):
    return {
        "benchmark": "rawvla-bench-light-libero-v1-train-rerender-npz",
        "dataset_name": dataset,
        "suite": dataset.split("_no_noops")[0],
        "task_id": 0,
        "task_name": task,
        "demo_index": demo,
        "demo_key": f"demo_{demo}",
        "trajectory_id": demo,
        "noops_filtered": 0,
        "hdf5_path": f"{task}_demo.hdf5",
        "variant_index": 0,
        "representation": "raw",
        "render_order": "env_lighting_then_rgb256_then_fixed_unprocess_noise_then_policy_resize",
        "replay_mode": "actions",
        "lighting_domain": lighting,
        "lighting_ev": 1.25,
        "lighting_rig": "common_agentview_key",
        "lighting_scale": 2.0,
        "lighting_sample_unit": 0.5,
        "noise_seed": 12345,
        "sensor_saturation_ev": 0.5,
        "num_frames": frames,
    }


def robotwin_metadata(task: str, episode: int, *, lighting: str = "Low", frames: int = 6):
    return {
        "benchmark": "rawvla-bench-light-robotwin2-v1-train-state-copy",
        "task_name": task,
        "episode": episode,
        "entry_index": episode,
        "dataset_seed": 0,
        "source_hdf5": f"episode{episode}.hdf5",
        "source_trajectory": f"episode{episode}.pkl",
        "clean_protocol": "original_lighting_joint_action_replay",
        "clean_replay_record": f"episode{episode}.json",
        "raw_protocol": "copy_clean_physics_state_each_frame_then_render_without_action",
        "raw_npz": f"episode_{episode:06d}.npz",
        "rgb_npz": f"episode_{episode:06d}.npz",
        "views": ["head_camera", "left_camera", "right_camera"],
        "representation": "three_channel_linear_float32",
        "lighting_domain": lighting,
        "lighting_ev": -3.5,
        "lighting_strategy": "environment_only",
        "lighting_scale": 0.5,
        "lighting_sample_unit": 0.5,
        "noise_seed": 999,
        "raw_sensor_noise_enabled": True,
        "raw_sensor_read_noise": 3.90625e-08,
        "raw_sensor_shot_noise": 2.5e-05,
        "raw_white_level": 3.5,
        "num_frames": frames,
    }


def write_config(tmp_path: Path, *, deep_per_platform: int = 2, seed: int = 20261014) -> Path:
    config = {
        "schema": "n04-s0-data-v1",
        "split_seed": seed,
        "split_ratio": {"train": 0.8, "dev": 0.1, "final": 0.1},
        "remote": {
            "data": {
                "libero_raw": str(tmp_path / "libero" / "raw"),
                "robotwin_raw": str(tmp_path / "robotwin2" / "raw"),
            }
        },
        "manifest": {
            "npz_glob": "**/*.npz",
            "deep_sample": {"per_platform": deep_per_platform, "seed": seed, "max_frames_read": 2},
            "splits": {
                "group_key": "source_trajectory_id",
                "stratify_by": "platform",
                "variants_stay_together": True,
                "holdouts": ["object", "scene", "camera", "unseen_factor_combination"],
            },
        },
    }
    path = tmp_path / "s0_data.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def build_libero(
    tmp_path: Path,
    *,
    sources: int = 3,
    variants: int = 1,
    cameras_frames=None,
    dataset: str = "libero_object_no_noops_1.0.0_lerobot",
    task: str = "SCENE1_pick_up_the_bowl",
    frames: int = 6,
) -> list[Path]:
    paths = []
    for demo in range(sources):
        for variant in range(variants):
            lighting = ("Normal", "ExtremeLow", "Over")[(demo + variant) % 3]
            relpath = f"{dataset}/episode_{demo:06d}/variant_{variant:02d}.npz"
            path = tmp_path / "libero" / "raw" / relpath
            metadata = libero_metadata(dataset, task, demo, lighting=lighting, frames=frames)
            metadata["rgb_npz_relpath"] = relpath
            _save_npz(
                path,
                libero_arrays(frames, seed=demo * 10 + variant, camera_frames=cameras_frames),
                metadata,
            )
            rgb = tmp_path / "libero" / "rgb" / relpath
            rgb.parent.mkdir(parents=True, exist_ok=True)
            rgb.write_bytes(b"rgb")
            paths.append(path)
    return paths


def build_robotwin(
    tmp_path: Path,
    *,
    episodes: int = 3,
    camera_frames=None,
    task: str = "grab_roller",
    frames: int = 6,
    compressed: bool = False,
) -> list[Path]:
    paths = []
    for episode in range(episodes):
        path = tmp_path / "robotwin2" / "raw" / task / f"episode_{episode:06d}.npz"
        _save_npz(
            path,
            robotwin_arrays(frames, seed=episode, camera_frames=camera_frames),
            robotwin_metadata(task, episode, frames=frames),
            compressed=compressed,
        )
        rgb = tmp_path / "robotwin2" / "rgb" / task / f"episode_{episode:06d}.npz"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(b"rgb")
        paths.append(path)
    return paths


@pytest.fixture()
def fixture_run(tmp_path: Path):
    build_libero(
        tmp_path,
        sources=2,
        task="pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    )
    build_libero(
        tmp_path,
        sources=1,
        dataset="libero_10_no_noops_1.0.0_lerobot",
        task="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    )
    build_robotwin(tmp_path, episodes=3)
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    code = manifest.main(["--config", str(config), "--output", str(out)])
    assert code == 0
    inventory = json.loads(out.read_text(encoding="utf-8"))
    splits = json.loads((out.parent / "splits.json").read_text(encoding="utf-8"))
    coverage = json.loads((out.parent / "coverage.json").read_text(encoding="utf-8"))
    provenance = json.loads((out.parent / "telemetry-provenance.json").read_text(encoding="utf-8"))
    hashes = json.loads((out.parent / "inventory-hashes.json").read_text(encoding="utf-8"))
    return {
        "tmp_path": tmp_path,
        "config": config,
        "out": out,
        "inventory": inventory,
        "splits": splits,
        "coverage": coverage,
        "provenance": provenance,
        "hashes": hashes,
    }


# ---------------------------------------------------------------------------
# 逐 NPZ 记录
# ---------------------------------------------------------------------------


def test_inventory_records_match_real_npz_layout(fixture_run):
    inventory = fixture_run["inventory"]
    assert inventory["schema"] == manifest.SCHEMA
    assert inventory["counts"]["files"] == 6
    assert inventory["counts"]["files_by_platform"] == {"libero": 3, "robotwin": 3}
    records = {(r["platform"], r["relpath"]): r for r in inventory["records"]}
    libero = records[
        ("libero", "libero_object_no_noops_1.0.0_lerobot/episode_000000/variant_00.npz")
    ]
    assert libero["source_trajectory_id"] == (
        "libero:libero_object_no_noops_1.0.0_lerobot:"
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket:demo_0"
    )
    assert libero["identity_source"] == "metadata"
    assert libero["variant_index"] == 0
    assert libero["variant_label"] == "variant_00|lighting_domain=Normal|noise_seed=12345"
    assert libero["headers"]["agentview_raw_uint8"] == {"shape": [6, 8, 8, 3], "dtype": "uint8"}
    assert libero["headers"]["states"]["dtype"] == "float64"
    camera = libero["arrays"]["agentview_raw_uint8"]
    # 相机大数组按帧采样，小数组整读
    assert camera["read_mode"] == "sampled_frames"
    assert libero["arrays"]["robot_states"]["read_mode"] == "full"
    assert camera["min"] == 0.0
    assert camera["max"] == 255.0
    assert camera["saturation_fraction"] > 0.0
    assert camera["nonfinite"] == 0
    assert libero["frames"]["declared"] == 6
    assert libero["frames"]["effective"] == 6
    assert libero["frames"]["aligned"] is True
    assert libero["frames"]["missing_rate"] == 0.0
    assert libero["rgb_sibling"]["exists"] is True
    robotwin = records[("robotwin", "grab_roller/episode_000000.npz")]
    assert robotwin["source_trajectory_id"] == "robotwin:grab_roller:episode0.pkl"
    assert robotwin["arrays"]["head_camera_raw_float32"]["dtype"] == "float32"
    assert robotwin["arrays"]["head_camera_raw_float32"]["saturation_fraction"] is not None
    assert robotwin["metadata"]["raw_protocol"].endswith("render_without_action")
    assert inventory["run"]["io"]["bytes_read"] > 0
    assert inventory["run"]["io"]["files"] == inventory["counts"]["files"]


def test_missing_rate_reports_truncated_stream(tmp_path: Path):
    build_libero(tmp_path, sources=1, cameras_frames=4, frames=6)
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    assert (
        manifest.main(["--config", str(config), "--platform", "libero", "--output", str(out)]) == 0
    )
    inventory = json.loads(out.read_text(encoding="utf-8"))
    record = inventory["records"][0]
    assert record["frames"]["effective"] == 4
    assert record["frames"]["aligned"] is False
    assert record["frames"]["missing_rate"] == pytest.approx(1 - 4 / 6)
    deep = inventory["deep_sample"]["records"][0]
    alignment = next(check for check in deep["checks"] if check["name"] == "stream_alignment")
    assert alignment["status"] == "fail"
    assert inventory["deep_sample"]["summary"]["failures"]


def test_header_only_scan_does_not_read_whole_camera(tmp_path: Path):
    build_libero(tmp_path, sources=1, frames=6)
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    assert (
        manifest.main(["--config", str(config), "--platform", "libero", "--output", str(out)]) == 0
    )
    inventory = json.loads(out.read_text(encoding="utf-8"))
    record = inventory["records"][0]
    assert record["io"]["bytes_read"] < record["size_bytes"]
    report = record["arrays"]["agentview_raw_uint8"]
    assert report["frames_read"] == [0, 5]
    whole_array = int(np.prod(record["headers"]["agentview_raw_uint8"]["shape"]))
    assert report["elements"] < whole_array


def test_compressed_member_fallback(tmp_path: Path):
    build_robotwin(tmp_path, episodes=1, compressed=True)
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    assert (
        manifest.main(["--config", str(config), "--platform", "robotwin", "--output", str(out)])
        == 0
    )
    inventory = json.loads(out.read_text(encoding="utf-8"))
    record = inventory["records"][0]
    assert record["members"]["head_camera_raw_float32.npy"]["compress_type"] == zipfile.ZIP_DEFLATED
    assert record["arrays"]["head_camera_raw_float32"]["elements"] == 2 * 4 * 4 * 3
    assert record["frames"]["effective"] == 6


# ---------------------------------------------------------------------------
# 深读检查
# ---------------------------------------------------------------------------


def test_deep_sample_checks_pass_on_fixture(fixture_run):
    inventory = fixture_run["inventory"]
    deep = inventory["deep_sample"]
    assert len(deep["records"]) == 4  # per_platform=2
    assert deep["summary"]["failures"] == []
    for record in deep["records"]:
        checks = {check["name"]: check for check in record["checks"]}
        assert checks["finite_values"]["status"] == "pass"
        assert checks["stream_alignment"]["status"] == "pass"
        assert checks["sampled_frame_duplicates"]["status"] == "pass"
        time_checks = {
            name: check
            for name, check in checks.items()
            if name.startswith("time_column_monotonic:")
        }
        if record["platform"] == "libero":
            assert time_checks["time_column_monotonic:states"]["status"] == "pass"
            assert "严格单调=True" in time_checks["time_column_monotonic:states"]["detail"]
            assert (
                "列语义未在 metadata 声明" in time_checks["time_column_monotonic:states"]["detail"]
            )
        else:
            # RoboTwin 的小数组没有时间列：非单调列记为 not_applicable，不是失败
            assert time_checks
            assert all(check["status"] != "pass" for check in time_checks.values())
            assert checks["endpose_layout"]["status"] == "pass"
            assert "四元数模长" in checks["endpose_layout"]["detail"]
    counts = deep["summary"]["check_counts"]
    assert counts["libero.time_column_monotonic:states"]["pass"] == 2
    assert counts["robotwin.endpose_layout"]["pass"] == 2
    assert not any(
        key.startswith("robotwin.time_column_monotonic:") and "pass" in value
        for key, value in counts.items()
    )


def test_deep_sample_detects_nonfinite(tmp_path: Path):
    arrays = robotwin_arrays(4, seed=0)
    arrays["head_camera_raw_float32"][0, 0, 0, 0] = np.nan
    path = tmp_path / "robotwin2" / "raw" / "grab_roller" / "episode_000000.npz"
    _save_npz(path, arrays, robotwin_metadata("grab_roller", 0, frames=4))
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    assert (
        manifest.main(["--config", str(config), "--platform", "robotwin", "--output", str(out)])
        == 0
    )
    inventory = json.loads(out.read_text(encoding="utf-8"))
    assert inventory["records"][0]["arrays"]["head_camera_raw_float32"]["nonfinite"] >= 1
    failures = inventory["deep_sample"]["summary"]["failures"]
    assert any(item["check"] == "finite_values" for item in failures)


def test_deep_sample_is_stratified_and_deterministic(tmp_path: Path):
    build_libero(
        tmp_path,
        sources=2,
        dataset="libero_goal_no_noops_1.0.0_lerobot",
        task="STUDY_SCENE1_pick_up_the_book",
    )
    build_libero(
        tmp_path, sources=2, dataset="libero_object_no_noops_1.0.0_lerobot", task="pick_up_the_bowl"
    )
    config = write_config(tmp_path, deep_per_platform=2)
    selected = []
    for index in range(2):
        out = tmp_path / f"run{index}" / "inventory.json"
        assert (
            manifest.main(["--config", str(config), "--platform", "libero", "--output", str(out)])
            == 0
        )
        inventory = json.loads(out.read_text(encoding="utf-8"))
        selected.append(inventory["deep_sample"]["selected"])
    assert selected[0] == selected[1]
    tasks = {
        record["task_key"]
        for record in json.loads(
            (tmp_path / "run0" / "inventory.json").read_text(encoding="utf-8")
        )["records"]
        if record["relpath"] in selected[0]
    }
    assert len(tasks) == 2  # 每个任务至少抽到一条


# ---------------------------------------------------------------------------
# 拆分
# ---------------------------------------------------------------------------


def test_variants_stay_with_source_and_no_overlap(fixture_run):
    splits = fixture_run["splits"]
    main = splits["main"]
    assert main["group_key"] == "source_trajectory_id"
    assert main["source_overlap"] == {"train_dev": [], "train_final": [], "dev_final": []}
    assignment = {item["source_trajectory_id"]: item for item in main["assignments"]}
    assert assignment["robotwin:grab_roller:episode0.pkl"]["n_files"] == 1
    assert sum(len(item["relpaths"]) for item in main["assignments"]) == main["n_files"]
    assert main["n_files"] == 6
    assert splits["source_multiplicity"]["variants_per_source_max"] == 1


def test_variants_of_one_source_land_on_the_same_side(tmp_path: Path):
    build_libero(tmp_path, sources=1, variants=3)
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    assert (
        manifest.main(["--config", str(config), "--platform", "libero", "--output", str(out)]) == 0
    )
    splits = json.loads((out.parent / "splits.json").read_text(encoding="utf-8"))
    main = splits["main"]
    assert main["n_sources"] == 1
    assert main["n_files"] == 3
    assert len(main["assignments"]) == 1
    assert main["assignments"][0]["split"] == "train"
    assert len(main["assignments"][0]["relpaths"]) == 3
    multiplicity = splits["source_multiplicity"]
    assert multiplicity["files_per_source"] == {"3": 1}
    assert multiplicity["variants_per_source_max"] == 3


def test_small_group_allocations_are_listed(tmp_path: Path):
    build_libero(tmp_path, sources=5, task="STUDY_SCENE1_small_task")
    build_libero(
        tmp_path,
        sources=12,
        dataset="libero_10_no_noops_1.0.0_lerobot",
        task="STUDY_SCENE2_large_task",
    )
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    assert (
        manifest.main(["--config", str(config), "--platform", "libero", "--output", str(out)]) == 0
    )
    splits = json.loads((out.parent / "splits.json").read_text(encoding="utf-8"))
    main = splits["main"]
    by_task = {item["task_key"]: item for item in main["per_task"]}
    small = next(key for key in by_task if "small_task" in key)
    large = next(key for key in by_task if "large_task" in key)
    assert by_task[small]["counts"] == {"train": 4, "dev": 1, "final": 0}
    assert by_task[large]["counts"] == {"train": 10, "dev": 1, "final": 1}
    listed = {(item["task_key"], item["n_sources"]) for item in main["small_group_allocations"]}
    assert (small, 5) in listed
    assert all(item[1] < 10 for item in listed)
    small_entry = next(
        item for item in main["small_group_allocations"] if item["task_key"] == small
    )
    assert sorted(small_entry["members"]) == ["dev", "final", "train"]
    assert len(small_entry["members"]["train"]) == 4
    assert main["counts_by_split"]["train"] == 14
    assert main["n_sources"] == 17


def test_split_is_stable_across_runs(fixture_run, tmp_path: Path):
    config = fixture_run["config"]
    second = tmp_path / "second" / "inventory.json"
    assert manifest.main(["--config", str(config), "--output", str(second)]) == 0
    second_hash = json.loads((second.parent / "inventory-hashes.json").read_text(encoding="utf-8"))
    second_splits = json.loads((second.parent / "splits.json").read_text(encoding="utf-8"))
    assert second_hash["content_sha256"] == fixture_run["hashes"]["content_sha256"]
    assert second_hash["main_split_sha256"] == fixture_run["hashes"]["main_split_sha256"]
    assert (
        second_splits["main"]["member_table_sha256"]
        == fixture_run["splits"]["main"]["member_table_sha256"]
    )
    assert fixture_run["hashes"]["code_sha256"]


def test_split_seed_changes_assignment_only_within_stratum(tmp_path: Path):
    build_libero(tmp_path, sources=10, task="SCENE1_task")
    build_robotwin(tmp_path, episodes=10, task="grab_roller")
    config = write_config(tmp_path)
    tables = []
    for seed in (20261014, 7):
        out = tmp_path / f"run{seed}" / "inventory.json"
        assert (
            manifest.main(
                ["--config", str(config), "--split-seed", str(seed), "--output", str(out)]
            )
            == 0
        )
        splits = json.loads((out.parent / "splits.json").read_text(encoding="utf-8"))
        tables.append(
            {item["source_trajectory_id"]: item["split"] for item in splits["main"]["assignments"]}
        )
        assert splits["main"]["counts_by_split"] == {"train": 16, "dev": 2, "final": 2}
    assert tables[0] != tables[1]
    assert set(tables[0]) == set(tables[1])


def test_holdouts_cover_requested_names(fixture_run):
    holdouts = fixture_run["splits"]["holdouts"]
    assert set(holdouts) == {"object", "scene", "camera", "unseen_factor_combination"}
    object_holdout = holdouts["object"]
    assert object_holdout["level"] == "source_group"
    assert object_holdout["n_held_out_sources"] > 0
    assert object_holdout["checks"]["held_out_tasks_present_in_train"] == []
    train_ids = {
        item["source_trajectory_id"] for item in object_holdout["train_dev_split"]["assignments"]
    }
    assert not train_ids & set(object_holdout["held_out_source_ids"])
    camera = holdouts["camera"]
    assert camera["views"]["robotwin"]["held_out_view"] == "right"
    assert camera["views"]["libero"]["train_views"] == ["agentview"]
    assert camera["views"]["libero"]["shares_sources_with_train"] is True
    factor = holdouts["unseen_factor_combination"]["platforms"]["libero"]
    assert factor["held_out_factor"]["lighting_domain"] in {"Normal", "ExtremeLow", "Over"}
    factor_train = {
        item["source_trajectory_id"] for item in factor["train_dev_split"]["assignments"]
    }
    assert not factor_train & set(factor["held_out_source_ids"])
    scenes = holdouts["scene"]
    assert scenes["checks"]["held_out_scenes"] == ["STUDY_SCENE1"]
    assert scenes["checks"]["held_out_scenes_present_in_train"] == []


def test_factor_holdout_keeps_task_visible_in_train(tmp_path: Path):
    build_libero(tmp_path, sources=8)
    config = write_config(tmp_path)
    out = tmp_path / "run" / "inventory.json"
    assert (
        manifest.main(["--config", str(config), "--platform", "libero", "--output", str(out)]) == 0
    )
    splits = json.loads((out.parent / "splits.json").read_text(encoding="utf-8"))
    factor = splits["holdouts"]["unseen_factor_combination"]["platforms"]["libero"]
    assert factor["status"] == "built"
    assert factor["n_held_out_sources"] == 2
    assert factor["held_out_factor"] == {"lighting_domain": "Over", "n_files": 2}
    assert factor["held_out_tasks_absent_from_train"] == []
    assert factor["train_dev_split"]["n_sources"] == 6
    assert len(factor["train_dev_split"]["excluded_files"]) == 2


# ---------------------------------------------------------------------------
# 覆盖表与 provenance
# ---------------------------------------------------------------------------


def test_coverage_states_unavailable_conditions(fixture_run):
    coverage = fixture_run["coverage"]
    assert coverage["schema"] == "n04-s0-coverage-v1"
    assert {entry["platform"] for entry in coverage["fields"]} == {"libero", "robotwin"}
    libero = next(entry for entry in coverage["fields"] if entry["platform"] == "libero")
    robotwin = next(entry for entry in coverage["fields"] if entry["platform"] == "robotwin")
    rows = {row["field"]: row for row in libero["rows"]}
    robotwin_rows = {row["field"]: row for row in robotwin["rows"]}
    assert rows["camera_measurement"]["status"] == "available"
    assert rows["camera_calibration"]["status"] == "unavailable"
    assert "无" in rows["camera_calibration"]["evidence"]
    assert "replay_mode" in rows["executed_action"]["evidence"]
    assert robotwin_rows["executed_action"]["status"] == "unavailable"
    assert "without_action" in robotwin_rows["executed_action"]["evidence"]
    # Lead 裁决：两平台时间戳均不可用；单调首列只作未决观测
    assert rows["timestamp"]["status"] == "unavailable"
    assert robotwin_rows["timestamp"]["status"] == "unavailable"
    assert "Lead 裁决" in rows["timestamp"]["evidence"]
    assert "等间隔严格单调" in rows["timestamp"]["evidence"]
    assert "未决观测" in rows["timestamp"]["unavailable_conditions"][-1]
    assert "没有任何等间隔严格单调列" in robotwin_rows["timestamp"]["evidence"]
    assert rows["executed_action"]["status"] == "partial"
    assert rows["executed_action"]["verified"] is False
    assert rows["executed_action"]["provisional"] is True
    assert rows["executed_action"]["evidence_level"] == "producer_code_order"
    assert robotwin_rows["executed_action"]["evidence_level"] == "producer_code_order"
    assert rows["camera_calibration"]["evidence_level"] == "producer_code_order"
    assert (
        "MoJoCo" in rows["camera_calibration"]["evidence"]
        or "MuJoCo" in rows["camera_calibration"]["evidence"]
    )
    assert coverage["lead_rulings"]["rulings"]
    assert coverage["references"]["telemetry_source_audit"]["path"].endswith(
        "telemetry-source-audit.json"
    )
    for entry in coverage["fields"]:
        for row in entry["rows"]:
            assert row["evidence_level"] in {
                "measured",
                "metadata_only",
                "producer_code_order",
                "negative_scan",
            }
            if row["provisional"]:
                assert row["status"] != "available"
    assert robotwin_rows["exposure"]["status"] == "partial"
    assert robotwin_rows["noise_model"]["status"] == "available"
    assert rows["noise_model"]["status"] == "partial"
    assert rows["rgb_pair"]["status"] == "available"
    assert rows["source_identity"]["status"] == "available"
    for entry in coverage["fields"]:
        for row in entry["rows"]:
            assert row["evidence"], row["field"]
    verdicts = {item["condition"] for item in coverage["unavailable_conditions"]}
    assert any("executed_action" in item for item in verdicts)


def test_summary_out_is_small_and_not_duplicating_companions(fixture_run, tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    assert (
        manifest.main(
            [
                "--config",
                str(fixture_run["config"]),
                "--output",
                str(tmp_path / "run" / "inventory.json"),
                "--summary-out",
                str(summary_path),
            ]
        )
        == 0
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "records" not in summary
    assert set(summary) == {
        "schema",
        "generated_utc",
        "config",
        "roots",
        "counts",
        "definitions",
        "aggregates",
        "deep_sample",
        "run",
        "hashes",
        "companions",
    }
    assert summary["hashes"]["main_split_sha256"]
    assert summary["companions"]["coverage_sha256"] == summary["hashes"]["coverage_sha256"]


def test_reference_root_resolves_audit_path(fixture_run, tmp_path: Path):
    root = tmp_path / "reference-root"
    audit = root / manifest.TELEMETRY_SOURCE_AUDIT
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text('{"schema": "stub"}', encoding="utf-8")
    out = tmp_path / "ref" / "inventory.json"
    assert (
        manifest.main(
            [
                "--config",
                str(fixture_run["config"]),
                "--output",
                str(out),
                "--reference-root",
                str(root),
            ]
        )
        == 0
    )
    coverage = json.loads((out.parent / "coverage.json").read_text(encoding="utf-8"))
    reference = coverage["references"]["telemetry_source_audit"]
    assert reference["present_at_run"] is True
    assert reference["sha256"] == manifest.sha256_file(audit)
    assert reference["path"] == manifest.TELEMETRY_SOURCE_AUDIT


def test_telemetry_provenance_lists_sources_and_gaps(fixture_run):
    provenance = fixture_run["provenance"]
    items = {item["item"]: item for item in provenance["items"]}
    assert items["libero.action"]["presence"]["actions"] == 3
    assert items["libero.action"]["gaps"]
    assert items["robotwin.noise"]["presence"]["raw_sensor_shot_noise"] == 3
    assert items["robotwin.noise"]["measured_values"]["raw_sensor_shot_noise"]
    assert "不可用" in items["libero.calibration"]["semantics"]
    assert items["libero.episode_time"]["semantics"].startswith("不可用")
    assert items["robotwin.episode_time"]["semantics"].startswith("不可用")
    assert items["libero.action"]["semantics"].startswith("已执行动作候选")
    assert items["robotwin.action"]["semantics"].startswith("不可用")
    assert provenance["references"]["telemetry_source_audit"]["path"] == (
        manifest.TELEMETRY_SOURCE_AUDIT
    )
    assert items["global.sidecar"]["n_files"] == 6


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_allocate_counts_keeps_total_and_is_deterministic():
    ratios = {"train": 0.8, "dev": 0.1, "final": 0.1}
    assert manifest.allocate_counts(1, ratios) == {"train": 1, "dev": 0, "final": 0}
    assert manifest.allocate_counts(5, ratios) == {"train": 4, "dev": 1, "final": 0}
    assert manifest.allocate_counts(10, ratios) == {"train": 8, "dev": 1, "final": 1}
    assert manifest.allocate_counts(13, ratios) == {"train": 11, "dev": 1, "final": 1}
    for size in range(1, 60):
        counts = manifest.allocate_counts(size, ratios)
        assert sum(counts.values()) == size
        assert manifest.allocate_counts(size, ratios) == counts


def test_frame_indices_cover_endpoints():
    assert manifest.frame_indices(6, 2) == [0, 5]
    assert manifest.frame_indices(6, 4) == [0, 2, 3, 5]
    assert manifest.frame_indices(1, 4) == [0]
    assert manifest.frame_indices(0, 4) == []


def test_stable_order_ignores_input_order():
    items = ["b", "a", "c"]
    assert manifest.stable_order(items, 20261014) == manifest.stable_order(
        list(reversed(items)), 20261014
    )


def test_source_identity_prefers_metadata_and_falls_back_to_path():
    metadata = libero_metadata("d", "t", 3)
    assert manifest.source_trajectory_id(manifest.LIBERO_SPEC, metadata, "rel.npz") == (
        "libero:d:t:demo_3",
        "metadata",
    )
    assert manifest.source_trajectory_id(
        manifest.LIBERO_SPEC, {}, "x/episode_1/variant_00.npz"
    ) == (
        "libero:path:x/episode_1/variant_00.npz",
        "path",
    )
    robotwin = robotwin_metadata("grab_roller", 2)
    assert manifest.source_trajectory_id(manifest.ROBOTWIN_SPEC, robotwin, "rel.npz") == (
        "robotwin:grab_roller:episode2.pkl",
        "metadata",
    )


def test_strip_volatile_removes_timing_only():
    payload = {"run": {"wall_seconds": 1.0}, "a": 1, "io": {"read_seconds": 0.5, "bytes_read": 7}}
    assert manifest.strip_volatile(payload) == {"a": 1, "io": {"bytes_read": 7}}

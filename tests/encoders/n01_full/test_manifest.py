"""Behaviour tests for the N01 full four-region manifest (workflow B).

Everything here runs on synthetic NPZ fixtures written with the author RawVLA key names
and an explicit ``metadata_json["synthetic_fixture"] = true`` declaration. The real
server inventory and the real loader run are recorded in
``data/manifests/n01-full-v1/*.json`` and in the workflow report; these tests pin the
builder contract: source-group identity, the four-way rule, row identity fields, the
disjoint partition and the loader consuming raw ``sides``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from owac.raw.n01_full_manifest import (
    CAMERA_ID_MULTI,
    CAMERA_ID_NO_CAMERA,
    CAMERA_ID_UNKNOWN_METADATA,
    INVENTORY_SCHEMA,
    REGIONS,
    SCHEMA,
    SPLIT_VERSION,
    TIME_ORDER,
    build_manifest,
    camera_identity,
    camera_ids_for_keys,
    episode_id_of,
    four_way_side,
    inventory_hash,
    inventory_hash_recipe,
    loader_split_manifest,
    manifest_digest,
    probe_npz,
    region_paths,
    scan_npz_inventory,
    verify_manifest,
)
from owac.raw.rawvla_sequence import (
    LIBERO_PROTOCOL,
    ROBOTWIN_PROTOCOL,
    RawVLASequenceDataset,
    assign_side,
    build_split_manifest,
    load_split_manifest,
    source_group,
    write_split_manifest,
)

FRAMES = 12
HEIGHT = 4
WIDTH = 4
FIXTURE_NOTE = "OWAC synthetic fixture; not real RawVLA data"
DOMAINS = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")
LIBERO_SUITES = (
    "libero_10_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
)
ROBOTWIN_TASKS = ("grab_roller", "open_laptop", "lift_pot")


# --------------------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------------------


def _uint8_burst(frames: int, offset: int) -> np.ndarray:
    t = np.arange(frames)[:, None, None, None]
    i = np.arange(HEIGHT)[None, :, None, None]
    j = np.arange(WIDTH)[None, None, :, None]
    c = np.arange(3)[None, None, None, :]
    return ((offset + 16 * t + 4 * i + j + c) % 256).astype(np.uint8)


def _float32_burst(frames: int, base: float) -> np.ndarray:
    t = np.arange(frames)[:, None, None, None]
    i = np.arange(HEIGHT)[None, :, None, None]
    j = np.arange(WIDTH)[None, None, :, None]
    c = np.arange(3)[None, None, None, :]
    return (base * 0.25 * (1.0 + 0.2 * t) + 0.001 * (3 * i + 5 * j + 7 * c)).astype(np.float32)


def _metadata_json(domain: str, frames: int) -> np.ndarray:
    return np.asarray(
        json.dumps(
            {
                "synthetic_fixture": True,
                "fixture_note": FIXTURE_NOTE,
                "benchmark": "owac-synthetic-manifest-fixture",
                "lighting_domain": domain,
                "lighting_ev": 0.25,
                "lighting_scale": 1.5,
                "num_frames": frames,
            }
        )
    )


def write_libero_episode(
    raw_root: Path,
    *,
    suite: str = LIBERO_SUITES[3],
    episode: str = "000001",
    variant: str = "variant_00",
    cameras: tuple[str, ...] = ("agentview_raw_uint8", "wrist_raw_uint8"),
    metadata: bool = True,
    lighting_domain: str = "Normal",
    frames: int = FRAMES,
) -> Path:
    path = raw_root / suite / f"episode_{episode}" / f"{variant}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "actions": np.zeros((frames, 7), dtype=np.float32),
        "robot_states": np.zeros((frames, 8), dtype=np.float32),
        "states": np.zeros((frames, 92), dtype=np.float64),
        "rewards": np.zeros(frames, dtype=np.uint8),
        "dones": np.zeros(frames, dtype=np.uint8),
    }
    for index, key in enumerate(cameras):
        arrays[key] = _uint8_burst(frames, 7 * index)
    if metadata:
        arrays["metadata_json"] = _metadata_json(lighting_domain, frames)
    np.savez(path, **arrays)
    return path


def write_robotwin_episode(
    raw_root: Path,
    *,
    task: str = ROBOTWIN_TASKS[0],
    episode: str = "000001",
    cameras: tuple[str, ...] = (
        "head_camera_raw_float32",
        "left_camera_raw_float32",
        "right_camera_raw_float32",
    ),
    metadata: bool = True,
    lighting_domain: str = "Normal",
    frames: int = FRAMES,
) -> Path:
    path = raw_root / task / f"episode_{episode}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "actions": np.zeros((frames, 14), dtype=np.float32),
        "robot_states": np.zeros((frames, 16), dtype=np.float32),
        "left_endpose": np.zeros((frames, 7), dtype=np.float32),
        "right_endpose": np.zeros((frames, 7), dtype=np.float32),
        "rewards": np.zeros(frames, dtype=np.uint8),
        "dones": np.zeros(frames, dtype=np.uint8),
    }
    for index, key in enumerate(cameras):
        arrays[key] = _float32_burst(frames, 0.43 - 0.03 * index)
    if metadata:
        arrays["metadata_json"] = _metadata_json(lighting_domain, frames)
    np.savez(path, **arrays)
    return path


def write_libero_stats(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"q01": [-1.0] * 8, "q99": [3.0] * 8},
                    "actions": {"q01": [0.0] * 7, "q99": [1.0] * 7},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def write_robotwin_stats(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"mean": [-0.5] * 14, "std": [2.0] * 14},
                    "actions": {"mean": [0.25] * 14, "std": [0.5] * 14},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A real on-disk mini cache for both platforms plus both statistics files."""
    libero_root = tmp_path / "libero_raw"
    robotwin_root = tmp_path / "robotwin2_raw"
    for index in range(220):
        write_libero_episode(
            libero_root,
            suite=LIBERO_SUITES[index % len(LIBERO_SUITES)],
            episode=f"{index:06d}",
        )
    for index in range(140):
        write_robotwin_episode(
            robotwin_root,
            task=ROBOTWIN_TASKS[index % len(ROBOTWIN_TASKS)],
            episode=f"{index:06d}",
        )
    return (
        libero_root,
        robotwin_root,
        write_libero_stats(tmp_path / "libero_stats.json"),
        write_robotwin_stats(tmp_path / "robotwin_stats.json"),
    )


def make_dataset(
    raw_root: Path,
    dataset_format: str,
    stats_path: Path,
    manifest: dict,
    region: str,
) -> RawVLASequenceDataset:
    return RawVLASequenceDataset(
        root=raw_root.parent,
        dataset_format=dataset_format,
        split=region,
        raw_cache_root=raw_root,
        stats_path=stats_path,
        split_manifest=loader_split_manifest(manifest, raw_root),
        window=4,
        observation_stride=1,
        action_horizon=5,
    )


# --------------------------------------------------------------------------------------
# (1) source-group identity and the four-way rule
# --------------------------------------------------------------------------------------


def test_lighting_and_rgb_variants_never_split_one_trajectory() -> None:
    variants = (
        "data/libero/raw/suite/episode_000001/variant_00.npz",
        "data/libero/raw/suite/episode_000001/variant_01.npz",
        "data/libero/raw/suite/episode_000001_lighting_Low.npz",
        "data/libero/rgb/suite/episode_000001/variant_00.npz",
        "suite/episode_000001/variant_00.npz",
    )
    groups = {source_group(path, dataset_format="libero") for path in variants}
    assert groups == {"suite/episode_000001"}
    sides = {four_way_side(group, split_version=SPLIT_VERSION) for group in groups}
    assert len(sides) == 1


def test_four_way_side_is_the_cumulative_generalisation_of_assign_side() -> None:
    names = [f"suite_{index // 40:03d}/episode_{index:06d}" for index in range(400)]
    seen = set()
    for name in names:
        inside = {
            fraction: assign_side(name, split_version=SPLIT_VERSION, train_fraction=fraction)
            == "train"
            for fraction in (0.7, 0.8, 0.9)
        }
        expected = (
            "train"
            if inside[0.7]
            else "calibration"
            if inside[0.8]
            else "validation"
            if inside[0.9]
            else "test"
        )
        side = four_way_side(name, split_version=SPLIT_VERSION)
        assert side == expected
        assert side == four_way_side(name, split_version=SPLIT_VERSION)
        seen.add(side)
    assert seen == set(REGIONS)


def test_train_region_equals_legacy_build_split_manifest_train_side(tmp_path: Path) -> None:
    raw_root = tmp_path / "libero_raw"
    relatives = [f"suite/episode_{index:06d}/variant_00.npz" for index in range(160)]
    absolute = [raw_root / relative for relative in relatives]
    legacy = build_split_manifest(
        absolute,
        dataset_format="libero",
        relative_to=raw_root,
        train_fraction=0.7,
        split_version=SPLIT_VERSION,
    )
    ours = {
        str(raw_root / relative)
        for relative in relatives
        if four_way_side(
            source_group(relative, dataset_format="libero"), split_version=SPLIT_VERSION
        )
        == "train"
    }
    assert ours == set(legacy["sides"]["train"])


# --------------------------------------------------------------------------------------
# (2) row identity fields, including the no-metadata vs single-camera distinction
# --------------------------------------------------------------------------------------


def test_camera_identity_tokens_keep_no_metadata_and_single_camera_distinct() -> None:
    unknown = camera_identity([], LIBERO_PROTOCOL, metadata_present=False)
    no_camera = camera_identity([], LIBERO_PROTOCOL, metadata_present=True)
    single = camera_identity(["agentview_raw_uint8"], LIBERO_PROTOCOL, metadata_present=True)
    multi = camera_identity(
        ["agentview_raw_uint8", "wrist_raw_uint8"], LIBERO_PROTOCOL, metadata_present=True
    )
    assert unknown == {
        "camera_ids": [],
        "camera_id": CAMERA_ID_UNKNOWN_METADATA,
        "camera_count": 0,
    }
    assert no_camera["camera_id"] == CAMERA_ID_NO_CAMERA
    assert single == {"camera_ids": ["agentview"], "camera_id": "agentview", "camera_count": 1}
    assert multi["camera_id"] == CAMERA_ID_MULTI
    assert multi["camera_ids"] == ["agentview", "wrist"]
    assert len({unknown["camera_id"], no_camera["camera_id"], single["camera_id"]}) == 3
    assert CAMERA_ID_MULTI not in {unknown["camera_id"], no_camera["camera_id"], "agentview"}


def test_undeclared_camera_like_key_is_an_interface_error() -> None:
    with pytest.raises(ValueError, match="not declared"):
        camera_ids_for_keys(["side_camera_raw_uint8"], LIBERO_PROTOCOL)


def test_episode_id_is_namespaced_and_never_empty() -> None:
    assert (
        episode_id_of(
            "libero_spatial_no_noops_1.0.0_lerobot/episode_000018/variant_00.npz",
            fallback="fallback",
        )
        == "libero_spatial_no_noops_1.0.0_lerobot/episode_000018"
    )
    assert (
        episode_id_of("grab_roller/episode_000018.npz", fallback="fallback")
        == "grab_roller/episode_000018"
    )
    assert episode_id_of("odd/layout.npz", fallback="odd/layout") == "odd/layout"


def test_manifest_rows_and_identity_tokens_on_disk(tmp_path: Path) -> None:
    raw_root = tmp_path / "libero_raw"
    write_libero_episode(raw_root, episode="000001")  # two cameras, metadata present
    write_libero_episode(
        raw_root,
        episode="000002",
        cameras=("agentview_raw_uint8",),  # single camera
    )
    write_libero_episode(
        raw_root,
        episode="000003",
        cameras=("wrist_raw_uint8",),
        metadata=False,  # no metadata
    )
    inventory = scan_npz_inventory(raw_root, dataset="libero", created_utc="2026-10-14T00:00:00Z")
    manifest = build_manifest(inventory, created_utc="2026-10-14T00:00:00Z")
    rows = {row["episode"]: row for row in manifest["rows"]}
    assert rows["episode_000001"]["camera_id"] == CAMERA_ID_MULTI
    assert rows["episode_000002"]["camera_id"] == "agentview"
    assert rows["episode_000003"]["camera_id"] == CAMERA_ID_UNKNOWN_METADATA
    assert rows["episode_000003"]["sensor_metadata_present"] is False
    assert rows["episode_000003"]["lighting"] is None
    for row in manifest["rows"]:
        assert row["camera_id"] not in ("", None)
        assert row["camera_ids"] != [""]
        assert row["dataset_id"] == "libero"
        assert row["episode_id"] == f"libero_spatial_no_noops_1.0.0_lerobot/{row['episode']}"
        assert row["source_group"] == row["episode_id"]
        assert row["units"] == LIBERO_PROTOCOL.raw_unit
        assert row["input_mode"] == "raw_uint8_pseudo_raw"
        if row["sensor_metadata_present"]:
            assert row["calibration_version"] == "owac-synthetic-manifest-fixture"
        else:
            assert row["calibration_version"] is None
        assert row["inventory_hash"] == inventory_hash(
            row["relative_path"], row["size_bytes"], row["mtime_ns"]
        )
    required = {
        "relative_path",
        "inventory_hash",
        "dataset_id",
        "input_mode",
        "source_group",
        "episode",
        "episode_id",
        "camera_id",
        "camera_ids",
        "scene",
        "session",
        "sensor",
        "lighting",
        "split",
        "units",
        "calibration_version",
    }
    assert required <= set(manifest["rows"][0])
    assert manifest["schema"] == SCHEMA
    assert manifest["time_order"] == TIME_ORDER
    assert manifest["counts"]["files"] == 3


def test_robotwin_rows_use_protocol_camera_names(tmp_path: Path) -> None:
    raw_root = tmp_path / "robotwin2_raw"
    write_robotwin_episode(raw_root, task="open_laptop", episode="000007")
    inventory = scan_npz_inventory(raw_root, dataset="robotwin2")
    row = build_manifest(inventory)["rows"][0]
    assert row["dataset_id"] == "robotwin2"
    assert row["episode_id"] == "open_laptop/episode_000007"
    assert row["camera_ids"] == ["head_camera", "left_camera", "right_camera"]
    assert row["camera_id"] == CAMERA_ID_MULTI
    assert row["units"] == ROBOTWIN_PROTOCOL.raw_unit
    assert row["input_mode"] == "raw_float32_hdr_derived"


# --------------------------------------------------------------------------------------
# (3) inventory scan, partition and traceable inventory hash
# --------------------------------------------------------------------------------------


def test_scan_reports_metadata_lighting_and_probe(tmp_path: Path) -> None:
    raw_root = tmp_path / "libero_raw"
    path = write_libero_episode(raw_root, episode="000004")
    inventory = scan_npz_inventory(
        raw_root,
        dataset="libero",
        server="fixture",
        probe_paths=["libero_spatial_no_noops_1.0.0_lerobot/episode_000004/variant_00.npz"],
        created_utc="2026-10-14T00:00:00Z",
    )
    record = inventory["files"][0]
    assert inventory["schema"] == INVENTORY_SCHEMA
    assert record["metadata_present"] is True
    assert record["metadata_fields"]["lighting_domain"] == "Normal"
    assert record["metadata_fields"]["num_frames"] == FRAMES
    assert record["camera_keys"] == ["agentview_raw_uint8", "wrist_raw_uint8"]
    assert inventory["aggregate"]["lighting_domains"] == {"Normal": 1}
    assert inventory["aggregate"]["camera_key_sets"] == {"agentview_raw_uint8+wrist_raw_uint8": 1}
    probe = probe_npz(path, relative_to=raw_root)
    assert probe["relative_path"].endswith("episode_000004/variant_00.npz")
    keys = {entry["key"]: entry for entry in probe["keys"]}
    assert keys["agentview_raw_uint8"]["dtype"] == "uint8"
    assert keys["agentview_raw_uint8"]["shape"] == [FRAMES, HEIGHT, WIDTH, 3]
    assert keys["actions"]["dtype"] == "float32"
    assert keys["actions"]["shape"] == [FRAMES, 7]
    assert probe["metadata"]["lighting_domain"] == "Normal"
    assert inventory["probe"][0]["metadata"]["benchmark"] == "owac-synthetic-manifest-fixture"


def test_auto_probe_paths_covers_domains_and_prefixes(tmp_path: Path) -> None:
    raw_root = tmp_path / "libero_raw"
    domains = ("ExtremeLow", "Low", "Normal", "Over")
    for index in range(40):
        write_libero_episode(
            raw_root,
            suite=LIBERO_SUITES[index % len(LIBERO_SUITES)],
            episode=f"{index:06d}",
            lighting_domain=domains[index % len(domains)],
        )
    inventory = scan_npz_inventory(raw_root, dataset="libero", probe_max=4)
    assert len(inventory["probe"]) == 4
    assert {entry["metadata"]["lighting_domain"] for entry in inventory["probe"]} == set(domains)
    prefixes = {entry["relative_path"].split("/", 1)[0] for entry in inventory["probe"]}
    assert len(prefixes) == 4


def test_scan_rejects_a_root_without_npz(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no \\*\\.npz"):
        scan_npz_inventory(empty, dataset="libero")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        scan_npz_inventory(tmp_path / "missing", dataset="libero")


def test_inventory_hash_recipe_is_documented_and_detects_drift() -> None:
    recipe = inventory_hash_recipe()
    assert "size_bytes" in recipe and "mtime_ns" in recipe and "relative_path" in recipe
    base = inventory_hash("a/b.npz", 10, 1000)
    assert len(base) == 32
    assert inventory_hash("a/b.npz", 10, 1001) != base
    assert inventory_hash("a/b.npz", 11, 1000) != base
    assert base == inventory_hash("a/b.npz", 10, 1000)


def test_build_rejects_a_tampered_inventory_record(tmp_path: Path) -> None:
    raw_root = tmp_path / "libero_raw"
    write_libero_episode(raw_root, episode="000009")
    inventory = scan_npz_inventory(raw_root, dataset="libero")
    inventory["files"][0]["size_bytes"] += 1
    with pytest.raises(ValueError, match="inventory hash mismatch"):
        build_manifest(inventory)


def test_four_regions_are_disjoint_and_cover_every_row(tmp_path: Path) -> None:
    libero_root, _, _, _ = fixture_tree(tmp_path)
    inventory = scan_npz_inventory(libero_root, dataset="libero")
    manifest = build_manifest(inventory)
    report = verify_manifest(manifest)
    assert report["partition_ok"] is True
    assert all(count == 0 for count in report["pairwise_overlaps"].values())
    assert report["union_files"] == report["row_files"] == 220
    assert report["duplicate_rows"] == 0
    assert report["group_side_conflicts"] == {}
    assert set(report["region_files"]) == set(REGIONS)
    assert all(count > 0 for count in report["region_files"].values())
    total = sum(entry["files"] for entry in manifest["counts"]["regions"].values())
    assert total == 220
    for region in REGIONS:
        assert len(region_paths(manifest, region)) == report["region_files"][region]


def test_manifest_round_trips_through_legacy_split_io(tmp_path: Path) -> None:
    raw_root = tmp_path / "robotwin2_raw"
    write_robotwin_episode(raw_root, episode="000011")
    manifest = build_manifest(scan_npz_inventory(raw_root, dataset="robotwin2"))
    target = write_split_manifest(tmp_path / "manifest.json", manifest)
    reloaded = load_split_manifest(target)
    assert reloaded == manifest
    assert manifest_digest(reloaded) == manifest_digest(manifest)
    assert reloaded["sides"].keys() == {"train", "calibration", "validation", "test"}


# --------------------------------------------------------------------------------------
# (4) the old loader consumes all four sides
# --------------------------------------------------------------------------------------


def test_loader_consumes_all_four_regions_with_exact_paths(tmp_path: Path) -> None:
    libero_root, robotwin_root, libero_stats, robotwin_stats = fixture_tree(tmp_path)
    cases = (
        (libero_root, "libero", libero_stats),
        (robotwin_root, "robotwin2", robotwin_stats),
    )
    for raw_root, dataset_format, stats_path in cases:
        manifest = build_manifest(scan_npz_inventory(raw_root, dataset=dataset_format))
        sides = loader_split_manifest(manifest, raw_root)
        assert set(sides["sides"]) == set(REGIONS)
        seen: dict[str, set[str]] = {}
        for region in REGIONS:
            dataset = make_dataset(raw_root, dataset_format, stats_path, manifest, region)
            expected = sides["sides"][region]
            assert dataset.split_name == region
            assert list(dataset.paths) == expected
            assert set(dataset.paths) == set(expected)
            assert all(Path(path).is_file() for path in dataset.paths)
            seen[region] = set(dataset.paths)
            sample = dataset[0]
            assert sample.action.shape[0] == 5
            sampled_relative = Path(dataset.paths[0]).relative_to(raw_root).as_posix()
            row = next(
                item for item in manifest["rows"] if item["relative_path"] == sampled_relative
            )
            assert sorted(sample.camera_ids) == row["camera_ids"]
        for index, left in enumerate(REGIONS):
            for right in REGIONS[index + 1 :]:
                assert not (seen[left] & seen[right])
        on_disk = {path.as_posix() for path in raw_root.rglob("*.npz") if path.is_file()}
        assert on_disk == {Path(path).as_posix() for path in seen["train"]} | {
            Path(path).as_posix() for path in seen["calibration"]
        } | {Path(path).as_posix() for path in seen["validation"]} | {
            Path(path).as_posix() for path in seen["test"]
        }

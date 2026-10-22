"""CPU contracts for shared-RAWVLA task6 Normal 8/2/2 preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from owac.raw.shared_rawvla_distillation import (
    SHARED_PROTOCOL,
    SHARED_RECEIPT_SCHEMA,
    SHARED_SPLITS,
    SharedDataError,
    build_shared_manifest,
    load_shared_manifest,
    shared_cache_identity,
)


def _write_npz(path: Path, *, frames: int, demo: str, lighting: str = "Normal") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "hdf5_path": "put_the_cream_cheese_in_the_bowl_demo.hdf5",
        "demo_key": demo,
        "demo_index": int(demo.split("_")[-1]),
        "lighting_domain": lighting,
        "task_id": 6,
        "suite": "libero_goal",
    }
    np.savez(
        path,
        agentview_raw_uint8=np.zeros((frames, 4, 4, 3), dtype=np.uint8),
        wrist_raw_uint8=np.zeros((frames, 4, 4, 3), dtype=np.uint8),
        actions=np.zeros((frames, 7), dtype=np.float32),
        states=np.zeros((frames, 79), dtype=np.float64),
        robot_states=np.zeros((frames, 8), dtype=np.float32),
        rewards=np.zeros((frames,), dtype=np.uint8),
        dones=np.zeros((frames,), dtype=np.uint8),
        metadata_json=np.array(json.dumps(metadata)),
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    shared = tmp_path / "shared"
    matches = []
    frames = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    for index in range(12):
        demo = f"demo_{index + 1}"
        path = shared / "data" / "libero" / "raw" / "ds" / f"episode_{index:06d}" / "variant_00.npz"
        _write_npz(path, frames=frames[index], demo=demo)
        matches.append(
            {"path": str(path), "demo": demo, "lighting": "Normal", "frames": frames[index]}
        )
    # A non-Normal variant of a selected demo must never enter the Normal set.
    over_path = shared / "data" / "libero" / "raw" / "ds" / "episode_000000" / "variant_over.npz"
    _write_npz(over_path, frames=30, demo="demo_1", lighting="Over")
    matches.append({"path": str(over_path), "demo": "demo_1", "lighting": "Over", "frames": 30})
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"count": len(matches), "matches": matches}), encoding="utf-8")
    return shared, inventory, tmp_path / "source-manifest.json"


def test_shared_manifest_is_normal_only_8_2_2_and_absolute(tmp_path: Path) -> None:
    shared, inventory, output = _fixture(tmp_path)
    report = build_shared_manifest(
        inventory_path=inventory,
        shared_root=shared,
        manifest_path=output,
        split_seed=101,
    )
    manifest = load_shared_manifest(output)
    assert report.split_counts == SHARED_SPLITS
    assert len(manifest.sources) == 12
    assert all(source.lighting == "Normal" for source in manifest.sources)
    assert all(source.raw_path.startswith(str(shared.resolve())) for source in manifest.sources)
    assert manifest.provenance["actual_source_hdf5_sha256"] is None
    assert manifest.provenance["actual_source_hdf5_sha256_available"] is False
    assert manifest.provenance["paired_rgb_used"] is False
    assert manifest.provenance["paired_rgb_path_registered"] is False
    assert manifest.provenance["paired_rgb_identity_verified"] is False
    assert manifest.provenance["paired_rgb_identity_fabricated"] is False
    assert all(source.metadata["paired_rgb_used"] is False for source in manifest.sources)
    assert all(source.metadata["paired_rgb_not_read"] is True for source in manifest.sources)
    assert all(
        source.metadata["actual_source_npz_sha256"] == source.raw_sha256
        for source in manifest.sources
    )
    assert manifest.path_semantics == "absolute"
    assert not any(
        source.metadata.get("demo_key") == "demo_1" and source.raw_path.endswith("variant_over.npz")
        for source in manifest.sources
    )
    assert report.train_anchors == sum(
        len(range(25, source.num_frames, 5))
        for source in manifest.sources
        if source.split == "train"
    )
    assert report.validation_windows == sum(
        len(range(0, source.num_frames, 5))
        for source in manifest.sources
        if source.split == "validation"
    )
    assert report.cache_rows == sum(
        len(range(0, source.num_frames, 5))
        for source in manifest.sources
        if source.split in {"train", "validation"}
    )
    assert report.manifest_sha256 == ""


def test_shared_manifest_is_deterministic_and_rejects_non_normal_count(tmp_path: Path) -> None:
    shared, inventory, output = _fixture(tmp_path)
    first = build_shared_manifest(
        inventory_path=inventory, shared_root=shared, manifest_path=output
    )
    second = build_shared_manifest(
        inventory_path=inventory, shared_root=shared, manifest_path=output, force=True
    )
    assert first.manifest_sha256 == second.manifest_sha256 == ""
    assert first.split_counts == second.split_counts
    with pytest.raises(SharedDataError, match="already exists"):
        build_shared_manifest(inventory_path=inventory, shared_root=shared, manifest_path=output)
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    payload["matches"] = [item for item in payload["matches"] if item["lighting"] == "Normal"][:11]
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SharedDataError, match="Normal source count"):
        build_shared_manifest(
            inventory_path=inventory, shared_root=shared, manifest_path=tmp_path / "bad.json"
        )


def test_shared_cache_identity_binds_raw_teacher_and_causal_state(tmp_path: Path) -> None:
    shared, inventory, output = _fixture(tmp_path)
    report = build_shared_manifest(
        inventory_path=inventory, shared_root=shared, manifest_path=output
    )
    manifest = load_shared_manifest(output)
    identity = shared_cache_identity(
        assets_identity={
            "artifact_sha256": "a" * 64,
            "backend_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "stats_sha256": "d" * 64,
            "sources_sha256": report.manifest_sha256,
        },
        manifest=manifest,
        manifest_sha256=report.manifest_sha256,
    )
    assert identity["protocol"] == SHARED_PROTOCOL
    assert identity["namespace"] == "shared-default"
    assert identity["splits"] == SHARED_SPLITS
    assert identity["causal_history"] == 6
    assert identity["raw_burst_frames"] == 6
    assert identity["endpoint_stride"] == 5
    assert len(identity["source_raw_sha256"]) == 12
    assert identity["test_targets_generated"] is False
    assert identity["actual_source_hdf5_sha256"] is None
    assert identity["actual_source_hdf5_sha256_available"] is False
    assert identity["paired_rgb_used"] is False
    assert identity["paired_rgb_path_registered"] is False
    assert identity["paired_rgb_identity_verified"] is False
    assert identity["paired_rgb_not_read"] is True
    assert identity["paired_rgb_identity_fabricated"] is False
    assert len(identity["source_group_metadata"]) == 12


def test_shared_config_registers_protocol_counts_and_no_training() -> None:
    path = Path("configs/experiments/raw_stem_siglip_full_visual_shared_v1.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["protocol"] == SHARED_PROTOCOL
    assert config["data"]["expected_splits"] == SHARED_SPLITS
    assert config["data"]["expected_train_anchors"] == 118
    assert config["data"]["expected_validation_windows"] == 46
    assert config["data"]["expected_cache_rows"] == 204
    assert config["data"]["test_targets_generated"] is False
    assert config["authorization"]["optimizer_updates"] is False
    assert config["assets"]["source_manifest_sha256"] is None
    assert (
        config["assets"]["source_manifest_sha256_source"]
        == "teacher_preparation_receipt.manifest_sha256"
    )
    assert config["cache"]["identity_from_teacher_preparation_receipt"] is True
    assert config["remote"]["require_teacher_preparation_receipt"] is True
    assert config["evaluation"]["validation_sources"] == 2
    assert config["training"]["max_new_global_updates"] == 3116


def test_real_inventory_normal_split_is_exactly_8_2_2() -> None:
    from owac.raw.shared_rawvla_distillation import _normal_matches, _split_groups, load_inventory

    inventory = Path("outputs/raw-distillation-full-vision-20261022/shared-task6-inventory.json")
    normal = _normal_matches(load_inventory(inventory))
    groups = [(Path(item.path).parent.parent.name, item.demo) for item in normal]
    split = _split_groups(groups, seed=101)
    by_demo = {item.demo: split[(Path(item.path).parent.parent.name, item.demo)] for item in normal}
    assert {key for key, value in by_demo.items() if value == "train"} == {
        "demo_1",
        "demo_4",
        "demo_10",
        "demo_23",
        "demo_27",
        "demo_32",
        "demo_35",
        "demo_47",
    }
    assert {key for key, value in by_demo.items() if value == "validation"} == {
        "demo_26",
        "demo_44",
    }
    assert {key for key, value in by_demo.items() if value == "test"} == {"demo_21", "demo_28"}
    frames = {item.demo: item.frames for item in normal}
    train_anchors = sum(
        len(range(25, frames[demo], 5))
        for demo, split_name in by_demo.items()
        if split_name == "train"
    )
    validation_windows = sum(
        len(range(0, frames[demo], 5))
        for demo, split_name in by_demo.items()
        if split_name == "validation"
    )
    cache_rows = sum(
        len(range(0, frames[demo], 5))
        for demo, split_name in by_demo.items()
        if split_name in {"train", "validation"}
    )
    assert train_anchors == 118
    assert validation_windows == 46
    assert cache_rows == 204


def test_fullvisual_data_contract_accepts_shared_and_keeps_legacy_default() -> None:
    from owac.runtime.raw_full_visual_distillation import _data_contract

    shared = json.loads(
        Path("configs/experiments/raw_stem_siglip_full_visual_shared_v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract = _data_contract(shared)
    assert contract["protocol"] == SHARED_PROTOCOL
    assert contract["expected_source_count"] == 12
    assert contract["expected_splits"] == SHARED_SPLITS
    assert contract["expected_train_anchors"] == 118
    assert contract["expected_validation_windows"] == 46
    assert contract["expected_cache_rows"] == 204
    assert contract["test_targets_generated"] is False
    legacy = _data_contract({})
    assert legacy["protocol"] == "legacy-c-full-40-5-5.v1"
    assert legacy["expected_splits"] == {"train": 40, "validation": 5, "test": 5}


def test_teacher_receipt_is_required_and_binds_cache_identity(tmp_path: Path) -> None:
    from owac.runtime.raw_full_visual_distillation import PathMapper, _teacher_receipt

    receipt_path = tmp_path / "teacher-preparation-receipt.json"
    receipt = {
        "schema": SHARED_RECEIPT_SCHEMA,
        "protocol": SHARED_PROTOCOL,
        "test_targets_generated": False,
        "cache_identity_key": "a" * 64,
        "manifest_sha256": "b" * 64,
        "gpu_wall_sum_seconds": 12.5,
        "actual_source_hdf5_sha256": None,
        "actual_source_hdf5_sha256_available": False,
        "paired_rgb_used": False,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    config = {
        "remote": {
            "teacher_preparation_receipt": str(receipt_path),
            "require_teacher_preparation_receipt": True,
        }
    }
    result = _teacher_receipt(config, PathMapper())
    assert result is not None
    assert result["payload"]["cache_identity_key"] == "a" * 64
    assert result["sha256"] is None
    missing = {
        "remote": {
            "teacher_preparation_receipt": str(tmp_path / "absent.json"),
            "require_teacher_preparation_receipt": True,
        }
    }
    with pytest.raises(Exception, match="absent"):
        _teacher_receipt(missing, PathMapper())


def test_manifest_scanner_never_materializes_pixel_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared, inventory, output = _fixture(tmp_path)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("manifest scan must not call np.load on shared pixel arrays")

    monkeypatch.setattr(np, "load", forbidden_load)
    report = build_shared_manifest(
        inventory_path=inventory,
        shared_root=shared,
        manifest_path=output,
        split_seed=101,
    )
    assert report.split_counts == SHARED_SPLITS
    manifest = load_shared_manifest(output)
    expected_rows = sum(
        len(range(0, source.num_frames, 5))
        for source in manifest.sources
        if source.split in {"train", "validation"}
    )
    assert report.cache_rows == expected_rows
    assert expected_rows > 0


def test_preserved_first_build_manifest_without_new_honesty_keys_still_loads(
    tmp_path: Path,
) -> None:
    shared, inventory, output = _fixture(tmp_path)
    build_shared_manifest(inventory_path=inventory, shared_root=shared, manifest_path=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    for key in ("paired_rgb_used", "paired_rgb_path_registered", "paired_rgb_identity_verified"):
        payload["provenance"].pop(key, None)
    for source in payload["sources"]:
        for key in (
            "paired_rgb_used",
            "paired_rgb_path_registered",
            "paired_rgb_identity_verified",
            "actual_source_npz_sha256",
        ):
            source["metadata"].pop(key, None)
    output.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_shared_manifest(output)
    assert len(manifest.sources) == 12

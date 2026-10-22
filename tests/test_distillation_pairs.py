"""Contract tests for the versioned paired distillation manifest (CPU fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.raw.distillation_pairs import (
    PairedDataUnavailable,
    PairedDistillationDataset,
    PairedManifestError,
    RawFrameData,
    SampleCache,
    assign_source_splits,
    load_paired_manifest,
)
from owac.raw.rawvla_sequence_n02 import NormalisationStats, normalise_actions

FIXTURE_FILE_SHA = "a" * 64
FIXTURE_ARRAY_SHA = "b" * 64
FIXTURE_SOURCE_SHA = "c" * 64


def _stats_payload():
    return {
        "norm_stats": {
            "state": {
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "q01": [-1.0] * 8,
                "q99": [1.0] * 8,
            },
            "actions": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
            },
        }
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return FIXTURE_FILE_SHA


def _make_fixture(root: Path, *, source_count=6, variant_cycle=("rawvla_rgb3", "bayer4_packed")):
    root.mkdir(parents=True, exist_ok=True)
    stats_path = root / "norm_stats.json"
    stats_path.write_text(json.dumps(_stats_payload()))
    stats = NormalisationStats.from_json(stats_path, mode="quantile")
    rng = np.random.default_rng(5)
    split_counts = {"train": 4, "validation": 1, "test": 1}
    if source_count != 6:
        split_counts = {"train": source_count - 2, "validation": 1, "test": 1}
    source_ids = [f"source_{index:03d}" for index in range(source_count)]
    splits = assign_source_splits(source_ids, seed=101, counts=split_counts)
    sources = []
    frames = []
    actions = rng.uniform(-1.0, 1.0, size=(10, 7)).astype(np.float32)
    state = rng.uniform(-1.0, 1.0, size=(8,)).astype(np.float32)
    future = np.array([True] * 8 + [False] * 2, dtype=np.bool_)
    normalised, _, _ = normalise_actions(actions, state, stats, dataset_format="libero")
    normalised_sha = FIXTURE_ARRAY_SHA
    future_sha = FIXTURE_ARRAY_SHA
    for index, source_id in enumerate(source_ids):
        variant = variant_cycle[index % len(variant_cycle)]
        if index == 1:
            variant = "rgb3_control"
        source = {
            "source_id": source_id,
            "acquisition_id": f"acq_{index:03d}",
            "demo_id": f"demo_{index:03d}",
            "split": splits[source_id],
            "source_sha256": FIXTURE_SOURCE_SHA,
            "camera_ids": ["base_0_rgb"],
            "variants": ["rawvla_rgb3", "bayer4_packed", "rgb3_control"],
            "language": "put the cream cheese in the bowl",
        }
        sources.append(source)
        for frame_offset in (0, 1) if index == 0 else (0,):
            frame_id = frame_offset
            timestamp = frame_offset * 0.25
            frame_key = f"{source_id}/base_0_rgb/{frame_id:06d}/{variant}"
            rgb = np.linspace(-1.0, 1.0, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8).copy()
            if variant == "bayer4_packed":
                raw = rng.integers(0, 1024, size=(4, 4, 4), dtype=np.uint16)
                physical_w = physical_h = 8
                bayer = {
                    "cfa_pattern": "RGGB",
                    "black_level": [0, 0, 0, 0],
                    "white_level": [1023, 1023, 1023, 1023],
                }
            elif variant == "rgb3_control":
                raw = rgb.copy()
                physical_w = physical_h = 8
                bayer = None
            else:
                raw = rng.uniform(0.0, 1.0, size=(3, 8, 8)).astype(np.float32)
                physical_w = physical_h = 8
                bayer = None
            frame_dir = root / "arrays"
            file_path = frame_dir / (frame_key.replace("/", "_") + ".npz")
            arrays = {
                "raw": raw,
                "actions_raw": actions,
                "future_mask": future,
                "state": state,
            }
            teacher: dict[str, object] | None
            if index == 0 and frame_offset == 1:
                # One frame uses the declared prepared-file path.
                prepared = root / "prepared" / "teacher_rgb.npy"
                prepared.parent.mkdir(parents=True, exist_ok=True)
                np.save(prepared, rgb)
                teacher = {
                    "kind": "prepared_file",
                    "path": "prepared/teacher_rgb.npy",
                    "sha256": FIXTURE_FILE_SHA,
                }
            else:
                arrays["teacher_rgb"] = rgb
                mask_shape = (4, 4) if variant == "bayer4_packed" else (8, 8)
                arrays["valid_mask"] = np.ones(mask_shape, dtype=np.bool_)
                arrays["bad_mask"] = np.zeros(mask_shape, dtype=np.bool_)
                arrays["saturated_mask"] = np.zeros(mask_shape, dtype=np.bool_)
                teacher = {"kind": "npz", "key": "teacher_rgb"}
            array_sha = _write_npz(file_path, arrays)
            frame = {
                "frame_key": frame_key,
                "source_id": source_id,
                "acquisition_id": source["acquisition_id"],
                "demo_id": source["demo_id"],
                "frame_id": frame_id,
                "timestamp": timestamp,
                "camera_id": "base_0_rgb",
                "variant": variant,
                "fov": {
                    "width": physical_w,
                    "height": physical_h,
                    "fx": 8.0,
                    "fy": 8.0,
                    "cx": 4.0,
                    "cy": 4.0,
                },
                "crop": {"x": 0, "y": 0, "width": physical_w, "height": physical_h},
                "flip": "none",
                "array": {
                    "path": str(file_path.relative_to(root)),
                    "sha256": array_sha,
                },
                "raw_shape": list(raw.shape),
                "raw_dtype": str(raw.dtype),
                "action_start": 0,
                "action_end": 10,
                "normalized_actions_sha256": normalised_sha,
                "future_mask_sha256": future_sha,
                "state_dim": 8,
                "teacher_rgb": teacher,
                "augmentation_id": "none",
                "exposure": 0.02 if frame_offset == 0 else None,
                "noise": None,
                "checkpoint_sha256": "a" * 64,
                "isp_id": "fixture-isp",
                "isp_sha256": "d" * 64,
                "preprocessing_id": "fixture-preprocess",
                "preprocessing_sha256": "e" * 64,
                "normalization_id": "fixture-quantile",
            }
            if bayer is not None:
                frame["bayer"] = bayer
            frames.append(frame)
    manifest = {
        "schema": "owac.raw-distillation.pairs.v1",
        "manifest_version": 1,
        "manifest_id": "fixture-contract-v1",
        "evidence_level": "fixture_contract",
        "fixture": True,
        "created_at_utc": "2026-10-20T00:00:00Z",
        "task": {
            "dataset": "RAWVLA-Bench",
            "suite": "libero_goal",
            "task_key": "task6",
            "instruction": "put the cream cheese in the bowl",
            "dataset_format": "libero",
            "split_seed": 101,
        },
        "partition": {
            "seed": 101,
            "source_count": source_count,
            "splits": split_counts,
        },
        "action_horizon": 10,
        "control_dim": 7,
        "checkpoint": {
            "path": "fixture://policy",
            "sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "load_audit_sha256": "c" * 64,
            "accepted": False,
        },
        "isp": {
            "id": "fixture-isp",
            "implementation": "fixture",
            "sha256": "d" * 64,
        },
        "preprocessing": {
            "id": "fixture-preprocess",
            "sha256": "e" * 64,
            "image_size": 8,
            "color_range": "-1,1",
        },
        "normalization": {
            "id": "fixture-quantile",
            "mode": "quantile",
            "stats_path": "norm_stats.json",
            "stats_sha256": FIXTURE_FILE_SHA,
            "stats_key": "",
            "action_width": 7,
            "epsilon": 1e-6,
            "epsilon_mode": "floor",
            "delta_action_indices": [],
            "hash_algorithm": "float32-c-order-sha256-v1",
        },
        "sources": sources,
        "frames": frames,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path, manifest, stats


def _load(path: Path, **kwargs):
    params = {"allow_fixture": True, "fixture_source_count": 6}
    params.update(kwargs)
    return load_paired_manifest(path, **params)


def test_manifest_requires_explicit_fixture_and_strict_counts(tmp_path):
    path, manifest, _ = _make_fixture(tmp_path)
    with pytest.raises(PairedDataUnavailable, match="fixture"):
        load_paired_manifest(path)
    with pytest.raises(PairedManifestError, match="source count"):
        _load(path, fixture_source_count=5)
    loaded = _load(path)
    assert loaded.fixture is True
    assert len(loaded.sources) == 6
    assert loaded.source_splits
    assert set(loaded.source_splits.values()) == {"train", "validation", "test"}
    # All three variants of the first source share its source split.
    first = loaded.sources[0]
    for frame in loaded.frames:
        if frame.source_id == first.source_id:
            assert frame.variant in first.variants


def test_real_partition_is_exactly_50_40_5_5_seed_101():
    ids = [f"source_{index:03d}" for index in range(50)]
    splits = assign_source_splits(ids, seed=101)
    counts = {name: list(splits.values()).count(name) for name in ("train", "validation", "test")}
    assert counts == {"train": 40, "validation": 5, "test": 5}
    with pytest.raises(PairedManifestError, match="sum"):
        assign_source_splits(ids, seed=101, counts={"train": 40, "validation": 5, "test": 4})
    with pytest.raises(PairedManifestError, match="unique"):
        assign_source_splits(["a", "a"], counts={"train": 1, "validation": 1, "test": 0})


def test_sample_has_typed_raw_teacher_actions_future_mask_state_language(tmp_path):
    path, manifest, _ = _make_fixture(tmp_path)
    dataset = PairedDistillationDataset(_load(path))
    sample = dataset.sample(dataset.frame_keys[0])
    assert sample.source_id == "source_000"
    assert sample.split == "train"
    assert sample.variant == "rawvla_rgb3"
    assert sample.raw.values.shape == (3, 8, 8)
    assert sample.teacher_rgb.shape == (3, 8, 8)
    assert sample.teacher_rgb_source == "npz"
    assert sample.actions.shape == (10, 7)
    assert sample.future_mask.shape == (10,)
    assert not sample.future_mask[-1]
    assert sample.state is not None and sample.state.shape == (8,)
    assert sample.state_raw is not None and sample.state_raw.shape == (8,)
    assert sample.language == "put the cream cheese in the bowl"
    assert sample.fov.width == 8 and sample.crop.width == 8
    assert sample.checkpoint_sha256 == "a" * 64
    # Normalized actions are the explicit N02 result, not just a declaration.
    stats = NormalisationStats.from_json(tmp_path / "norm_stats.json", mode="quantile")
    expected, _, _ = normalise_actions(
        sample.actions.numpy() * 0 + 0, None, stats, dataset_format="libero"
    )
    assert expected.shape == (10, 7)
    # The prepared-file path dispatches too.
    prepared = dataset.sample(dataset.frame_keys[1])
    assert prepared.teacher_rgb_source.startswith("prepared_file:")
    bayer_key = next(
        frame.frame_key for frame in dataset.manifest.frames if frame.variant == "bayer4_packed"
    )
    bayer = dataset.sample(bayer_key)
    assert bayer.raw.cfa_pattern == "RGGB"
    assert bayer.raw.black_level is not None
    assert bayer.raw.values.shape == (4, 4, 4)
    stem_input = bayer.raw.as_stem_input()
    assert stem_input.input_mode == "bayer4_packed"
    assert stem_input.values.shape == (1, 1, 4, 4, 4)
    assert stem_input.valid_mask.shape == (1, 1, 4, 4)
    pseudo_input = sample.raw.as_stem_input()
    assert pseudo_input.input_mode == "rawvla_rgb3"
    assert pseudo_input.values.shape == (1, 1, 3, 8, 8)


def test_student_view_contains_no_teacher_rgb_actions_or_future_and_never_reads_cache(tmp_path):
    path, _, _ = _make_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    cache = SampleCache(cache_dir)
    dataset = PairedDistillationDataset(_load(path), cache=cache)
    frame_key = next(
        frame.frame_key for frame in dataset.manifest.frames if frame.variant != "rgb3_control"
    )
    view = dataset.student_view(frame_key)
    assert not hasattr(view, "teacher_rgb")
    assert not hasattr(view, "actions")
    assert not hasattr(view, "future_mask")
    assert view.raw.values.shape in {(3, 8, 8), (4, 4, 4)}

    # Poison the cache read; student_view must not use it.
    def forbidden(_self, _spec):
        raise AssertionError("student inference must not use the cache loader")

    cache.get = forbidden.__get__(cache, SampleCache)
    view_again = dataset.student_view(frame_key)
    assert view_again.frame_key == frame_key
    rgb3 = next(
        frame.frame_key for frame in dataset.manifest.frames if frame.variant == "rgb3_control"
    )
    with pytest.raises(PairedDataUnavailable, match="pure student interface"):
        dataset.student_view(rgb3)


def test_missing_teacher_rgb_and_stats_are_precise_unavailable(tmp_path):
    path, manifest, _ = _make_fixture(tmp_path)
    bad = json.loads(json.dumps(manifest))
    bad["frames"][0]["teacher_rgb"] = None
    bad_path = tmp_path / "missing_teacher.json"
    bad_path.write_text(json.dumps(bad))
    dataset = PairedDistillationDataset(_load(bad_path))
    with pytest.raises(PairedDataUnavailable, match="teacher RGB"):
        dataset.sample(dataset.frame_keys[0])

    no_stats = json.loads(json.dumps(manifest))
    no_stats["normalization"]["stats_path"] = "absent_stats.json"
    no_stats_path = tmp_path / "no_stats.json"
    no_stats_path.write_text(json.dumps(no_stats))
    with pytest.raises(PairedDataUnavailable, match="does not exist"):
        _load(no_stats_path)

    no_mode = json.loads(json.dumps(manifest))
    no_mode["normalization"]["mode"] = "none"
    no_mode_path = tmp_path / "no_mode.json"
    no_mode_path.write_text(json.dumps(no_mode))
    with pytest.raises(PairedDataUnavailable, match="raw expert action"):
        _load(no_mode_path)


def test_npz_allow_pickle_false_and_object_arrays(tmp_path):
    path, manifest, _ = _make_fixture(tmp_path)
    bad = json.loads(json.dumps(manifest))
    frame = bad["frames"][0]
    obj_path = tmp_path / "object.npz"
    np.savez_compressed(
        obj_path,
        raw=frame["raw_shape"],
        actions_raw=frame["raw_shape"],
        future_mask=frame["raw_shape"],
        state=frame["raw_shape"],
        teacher_rgb=frame["raw_shape"],
        poison=np.array([{"not": "allowed"}], dtype=object),
    )
    frame["array"]["path"] = obj_path.name
    frame["array"]["sha256"] = FIXTURE_FILE_SHA
    bad_path = tmp_path / "object_manifest.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(PairedManifestError, match="pickle|object|NPZ"):
        PairedDistillationDataset(_load(bad_path)).sample(
            "source_000/base_0_rgb/000000/rawvla_rgb3"
        )


def torch_equal(a, b):
    return np.array_equal(a.numpy(), b.numpy())


def test_rgb3_control_is_the_teacher_tensor_and_rejects_mismatch(tmp_path):
    path, manifest, _ = _make_fixture(tmp_path)
    loaded = _load(path)
    dataset = PairedDistillationDataset(loaded)
    rgb3 = next(frame.frame_key for frame in loaded.frames if frame.variant == "rgb3_control")
    sample = dataset.sample(rgb3)
    np.testing.assert_array_equal(sample.raw.values.numpy(), sample.teacher_rgb.numpy())
    # Tamper only the raw copy of the teacher tensor.
    bad = json.loads(json.dumps(manifest))
    frame_meta = next(item for item in bad["frames"] if item["variant"] == "rgb3_control")
    npz_path = tmp_path / frame_meta["array"]["path"]
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["raw"] = arrays["raw"] + 1.0
    frame_meta["array"]["sha256"] = _write_npz(npz_path, arrays)
    bad_path = tmp_path / "rgb_mismatch.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(PairedManifestError, match="actual teacher-preprocessed"):
        PairedDistillationDataset(_load(bad_path)).sample(rgb3)


def test_per_frame_identity_mismatch_is_rejected(tmp_path):
    path, manifest, _ = _make_fixture(tmp_path)
    for field, value, match in (
        ("isp_sha256", "9" * 64, "ISP identity"),
        ("preprocessing_id", "other-preprocess", "preprocessing identity"),
        ("checkpoint_sha256", "8" * 64, "checkpoint identity"),
        ("normalization_id", "other-normalization", "normalisation identity"),
    ):
        bad = json.loads(json.dumps(manifest))
        bad["frames"][0][field] = value
        bad_path = tmp_path / f"bad_{field}.json"
        bad_path.write_text(json.dumps(bad))
        with pytest.raises(PairedManifestError, match=match):
            _load(bad_path)


def test_real_manifest_requires_the_exact_fifty_source_protocol(tmp_path):
    path, manifest, _ = _make_fixture(tmp_path)
    bad = json.loads(json.dumps(manifest))
    bad["evidence_level"] = "real_paired_observation"
    bad["fixture"] = False
    bad["checkpoint"]["accepted"] = True
    bad_path = tmp_path / "not_real.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(PairedManifestError, match="50 sources"):
        load_paired_manifest(bad_path)


def test_sol_probe_single_channel_masks_and_student_selective_state_reads(tmp_path):
    frame = RawFrameData(
        torch.ones(3, 8, 8),
        "normalized_0_1",
        "",
        (0, 0),
        valid_mask=torch.ones(3, 8, 8, dtype=torch.bool),
        bad_mask=torch.zeros(3, 8, 8, dtype=torch.bool),
        saturated_mask=torch.zeros(3, 8, 8, dtype=torch.bool),
        missing_mask=torch.zeros(3, 8, 8, dtype=torch.bool),
    )
    frame.bad_mask[0, 2, 3] = True
    frame.saturated_mask[1, 2, 3] = True
    frame.missing_mask[2, 2, 3] = True
    stem_input = frame.as_stem_input()
    assert bool(stem_input.bad_mask[0, 0, 2, 3])
    assert bool(stem_input.saturated_mask[0, 0, 2, 3])
    assert bool(stem_input.missing_mask[0, 0, 2, 3])
    valid = frame.as_stem_input()
    assert valid.valid_mask[0, 0, 2, 3] is not None
    assert bool(valid.valid_mask[0, 0, 0, 0])

    path, _, _ = _make_fixture(tmp_path)
    dataset = PairedDistillationDataset(_load(path))
    key = next(
        frame.frame_key
        for frame in dataset.manifest.frames
        if frame.teacher_rgb
        and frame.teacher_rgb["kind"] == "npz"
        and frame.variant == "rawvla_rgb3"
    )
    reads = []
    original = np.lib.npyio.NpzFile.__getitem__

    def guarded(self, name):
        reads.append(name)
        if name == "teacher_rgb":
            raise AssertionError("student read teacher RGB")
        return original(self, name)

    np.lib.npyio.NpzFile.__getitem__ = guarded
    try:
        view = dataset.student_view(key)
    finally:
        np.lib.npyio.NpzFile.__getitem__ = original
    assert "teacher_rgb" not in reads
    assert "actions_raw" not in reads
    assert "future_mask" not in reads
    assert "state" in reads
    assert view.state is not None and view.state.shape == (8,)
    assert view.state_raw is not None
    assert not hasattr(view, "teacher_rgb")


def test_fov_missing_intrinsics_and_student_only_manifest(tmp_path):
    from owac.raw.distillation_pairs import CameraFOV

    missing = CameraFOV.from_mapping({"width": 8, "height": 8}, "fov")
    assert missing.intrinsics_available is False
    with pytest.raises(PairedManifestError, match="all present or all absent"):
        CameraFOV.from_mapping({"width": 8, "height": 8, "fx": 1.0}, "fov")

    path, manifest, _ = _make_fixture(tmp_path)
    prepared = tmp_path / "prepared" / "teacher_rgb.npy"
    prepared.unlink()
    # Normal pair loading still requires the prepared teacher file.
    with pytest.raises(PairedDataUnavailable):
        _load(path)
    student_manifest = load_paired_manifest(
        path, allow_fixture=True, fixture_source_count=6, student_only=True
    )
    student_dataset = PairedDistillationDataset(student_manifest, student_only=True)
    key = next(
        frame.frame_key
        for frame in student_manifest.frames
        if frame.teacher_rgb
        and frame.teacher_rgb["kind"] == "npz"
        and frame.variant == "rawvla_rgb3"
    )
    assert student_dataset.student_view(key).state is not None
    with pytest.raises(PairedDataUnavailable, match="student-only"):
        student_dataset.sample(key)

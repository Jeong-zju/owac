"""P0 data tests, part 1: partitions, two-platform time boundaries and the dataset wrapper.

Fixtures are small contract NPZ files written into ``tmp_path`` (see ``conftest.py``); they are
**not** real RawVLA data. The real cache is absent on this workstation, so nothing here is
evidence about the deployed data.

Every assertion checks a declared number, identity or mask: the source frame index the window
reads, which state array it uses, that tail padding is masked out of a masked loss, that history
padding adds no new measurement, that future actions never reach the inference input, and that
the manifest is read back without re-partitioning.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from owac.frameworks.n02_full import data as d
from owac.raw import rawvla_sequence_n02 as raw

from .conftest import LIBERO_SUITE_A, LIBERO_SUITE_B, LIBERO_SUITES, ROBOTWIN_TASKS

LIBERO_WINDOW = raw.WindowOptions(
    observation_stride=1, rnn_frames=1, burst_frames=1, action_horizon=10, action_offset=0
)
ROBOTWIN_WINDOW = raw.WindowOptions(
    observation_stride=1, rnn_frames=1, burst_frames=1, action_horizon=50, action_offset=1
)

LIBERO_EPISODES = (18, 19, 20, 21)


# --------------------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------------------


def write_libero_matrix(root, writer, *, num_frames: int = 24, suites=LIBERO_SUITES) -> dict:
    """Two suites x four episodes x two lighting variants: eight source-trajectory groups."""
    stored: dict[str, dict] = {}
    for suite in suites:
        for episode in LIBERO_EPISODES:
            for variant in (0, 1):
                record = writer(
                    root,
                    suite,
                    episode,
                    variant,
                    num_frames=num_frames,
                    lighting_domain=f"light_{variant}",
                )
                stored[record["relative_path"]] = record
    return stored


def write_robotwin_matrix(root, writer, *, num_frames: int = 60) -> dict:
    """One episode per RoboTwin task: one group per task."""
    stored: dict[str, dict] = {}
    for task in (*ROBOTWIN_TASKS, "open_laptop"):
        record = writer(root, task, 18, num_frames=num_frames)
        stored[record["relative_path"]] = record
    return stored


def build_manifest(root, dataset_format: str = "libero", *, seed: int = 7) -> d.PartitionManifest:
    loader = raw.RawVLAWindowDataset(root, dataset_format=dataset_format, hash_episodes=True)
    return d.build_partition_manifest(
        loader.episodes(), dataset_format=dataset_format, cache_root=root, seed=seed
    )


def anchor_index(table: d.AnchorTable, relative_path: str, current_frame_index: int) -> int:
    for index, anchor in enumerate(table):
        if (
            anchor.relative_path == relative_path
            and anchor.current_frame_index == current_frame_index
        ):
            return index
    raise AssertionError(f"no anchor at {relative_path} t={current_frame_index}")


def table_for(
    manifest: d.PartitionManifest,
    partition: str,
    window: raw.WindowOptions,
    *,
    stride: int = 1,
    rnn_frames: int | None = None,
    burst_frames: int | None = None,
    action_horizon: int | None = None,
    action_offset: int | None = None,
) -> d.AnchorTable:
    return d.build_anchor_table(
        manifest,
        partition,
        window=window,
        stride=stride,
        rnn_frames=window.rnn_frames if rnn_frames is None else rnn_frames,
        burst_frames=window.burst_frames if burst_frames is None else burst_frames,
        action_horizon=window.action_horizon if action_horizon is None else action_horizon,
        action_offset=window.action_offset if action_offset is None else action_offset,
    )


def libero_quantile_stats() -> raw.NormalisationStats:
    """Declared statistics for the fixture path; the values are test constants, not author data."""
    return raw.NormalisationStats(
        mode="quantile",
        state={"q01": [0.0] * 8, "q99": [40.0] * 8},
        action={"q01": [0.0] * 7, "q99": [1.0] * 7},
    )


def robotwin_zscore_stats() -> raw.NormalisationStats:
    """Mean 0 / std 1, so the normalised proprio equals the stored current-state proxy."""
    return raw.NormalisationStats(
        mode="zscore",
        state={"mean": [0.0] * 14, "std": [1.0] * 14},
        action={"mean": [0.0] * 14, "std": [1.0] * 14},
    )


def assert_measurement_frames_equal(left, right) -> None:
    assert set(left) == set(right)
    for camera_id in left:
        assert left[camera_id].values.shape == right[camera_id].values.shape
        assert np.array_equal(left[camera_id].values, right[camera_id].values)
        assert np.array_equal(left[camera_id].valid_mask, right[camera_id].valid_mask)
        assert np.array_equal(left[camera_id].bad_mask, right[camera_id].bad_mask)
        assert np.array_equal(left[camera_id].saturated_mask, right[camera_id].saturated_mask)
        assert np.array_equal(left[camera_id].new_measurement, right[camera_id].new_measurement)
        assert left[camera_id].frame_ids == right[camera_id].frame_ids
        assert left[camera_id].episode_ids == right[camera_id].episode_ids
        assert left[camera_id].metadata == right[camera_id].metadata
        assert left[camera_id].metadata_available == right[camera_id].metadata_available


def assert_inference_inputs_equal(left, right) -> None:
    assert_measurement_frames_equal(left.bundle.frames, right.bundle.frames)
    assert np.array_equal(left.bundle.context.proprio, right.bundle.context.proprio)
    assert left.bundle.context.proprio_available == right.bundle.context.proprio_available
    assert left.bundle.context.task_tokens == right.bundle.context.task_tokens
    assert left.bundle.context.task_mask == right.bundle.context.task_mask
    assert left.bundle.context.executed_action is None
    assert left.bundle.context.executed_action_available is False
    assert left.bundle.episode_ids == right.bundle.episode_ids
    assert left.bundle.anchor_ids == right.bundle.anchor_ids
    assert np.array_equal(left.bundle.new_time_mask, right.bundle.new_time_mask)
    assert left.declared_prior == right.declared_prior


def skip_without_state():
    return pytest.importorskip(
        "owac.frameworks.n02_full.state",
        reason="frozen P0 state contract (owned by the n02-core task) is not in this checkout",
    )


# --------------------------------------------------------------------------------------
# two-platform time boundaries
# --------------------------------------------------------------------------------------


def test_libero_window_reads_robot_states_at_t_and_actions_from_t(tmp_path, libero_writer) -> None:
    """LIBERO: state source ``robot_states[t]``, supervision ``actions[t:t+10]``."""
    root = tmp_path / "libero"
    stored = write_libero_matrix(root, libero_writer)
    manifest = build_manifest(root)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    table = table_for(manifest, partition, LIBERO_WINDOW)
    anchor = anchor_index(table, target, 5)
    stats = libero_quantile_stats()
    dataset = d.ManifestWindowDataset(
        root, manifest, partition, table, window_options=LIBERO_WINDOW, normalisation=stats
    )
    sample = dataset[anchor]

    assert sample.episode.relative_path == target
    assert sample.split == partition
    assert sample.observation_indices.tolist() == [5]
    assert sample.burst_indices.tolist() == [[5]]
    assert sample.supervision_indices.tolist() == list(range(5, 15))
    assert sample.supervision_mask.all()
    assert sample.padding_contribution == 0.0
    assert sample.provenance["window"]["state_source_key"] == "robot_states"

    expected_actions, _, _ = raw.normalise_actions(
        stored[target]["actions"][5:15],
        stored[target]["robot_states"][5],
        stats,
        dataset_format="libero",
    )
    assert np.array_equal(sample.supervision_actions, expected_actions)

    expected_state, _ = raw.normalise_state(stored[target]["robot_states"][5], stats)
    assert expected_state is not None
    assert sample.proprio is not None
    assert np.allclose(sample.proprio, expected_state, atol=1e-6, rtol=0.0)
    # The neighbouring rows differ, so this pins the read index to t and not t +/- 1.
    for other in (4, 6):
        neighbour, _ = raw.normalise_state(stored[target]["robot_states"][other], stats)
        assert not np.allclose(sample.proprio, neighbour, atol=1e-6, rtol=0.0)

    assert sample.executed_action is None
    assert sample.executed_action_available is False
    record = table[anchor]
    assert (record.t, record.current_frame_index) == (5, 5)
    assert (record.action_start, record.action_stop) == (5, 15)


def test_robotwin_window_reads_actions_as_current_state_and_offset_one(
    tmp_path, robotwin_writer
) -> None:
    """RoboTwin: ``actions[t]`` is the declared current 14-dim proxy, supervision from t+1."""
    root = tmp_path / "robotwin"
    stored = write_robotwin_matrix(root, robotwin_writer)
    manifest = build_manifest(root, "robotwin")
    target = next(path for path in stored if f"/{ROBOTWIN_TASKS[0]}/" in path)
    partition = manifest.partition_of(target)
    table = table_for(manifest, partition, ROBOTWIN_WINDOW)
    anchor = anchor_index(table, target, 4)
    stats = robotwin_zscore_stats()
    dataset = d.ManifestWindowDataset(
        root, manifest, partition, table, window_options=ROBOTWIN_WINDOW, normalisation=stats
    )
    sample = dataset[anchor]

    assert sample.supervision_indices.tolist() == list(range(5, 55))
    assert sample.supervision_mask.all()
    assert sample.observation_indices.tolist() == [4]
    assert sample.proprio is not None
    assert sample.proprio.shape == (14,)
    assert np.allclose(sample.proprio, stored[target]["actions"][4], atol=1e-6, rtol=0.0)
    assert not np.allclose(sample.proprio, stored[target]["actions"][3], atol=1e-6, rtol=0.0)
    assert stored[target]["robot_states"].shape[-1] == 16
    assert sample.provenance["window"]["state_source_key"] == "actions"
    assert sample.executed_action is None
    assert sample.executed_action_available is False

    # RoboTwin float32 keeps its own scale: only the declared current row is read, unchanged.
    head = sample.burst_values[0][0, 0]
    assert np.array_equal(head, np.moveaxis(stored[target]["head"][4], -1, 0))
    assert float(head.max()) < 1.0
    record = table[anchor]
    assert (record.t, record.current_frame_index) == (4, 4)
    assert (record.action_start, record.action_stop) == (5, 55)


def test_action_tail_padding_is_masked_out_of_a_masked_loss(tmp_path, libero_writer) -> None:
    """Padding repeats the last row, so only the mask keeps it out of the loss."""
    root = tmp_path / "libero"
    writer = libero_writer
    base: dict[str, dict] = {}
    for episode in (18, 19, 20):
        record = writer(root, LIBERO_SUITE_A, episode, 0, num_frames=12)
        base[record["relative_path"]] = record
    manifest = build_manifest(root)
    target = sorted(base)[0]
    partition = manifest.partition_of(target)
    table = table_for(manifest, partition, LIBERO_WINDOW)
    anchor = anchor_index(table, target, 10)
    dataset = d.ManifestWindowDataset(
        root, manifest, partition, table, window_options=LIBERO_WINDOW
    )
    sample = dataset[anchor]

    expected_mask = np.arange(10, 20) <= 11
    assert sample.supervision_mask.tolist() == expected_mask.tolist()
    assert sample.supervision_indices[~expected_mask].tolist() == [11] * int((~expected_mask).sum())
    assert sample.padding_contribution == pytest.approx(float((~expected_mask).mean()))

    # Every padded position repeats a source row that the mask already admits, so a masked
    # reduction counts each stored row exactly once while an unmasked one double-counts it.
    valid_rows = np.asarray(sample.supervision_indices)[expected_mask]
    valid_actions = np.asarray(sample.supervision_actions)[expected_mask]
    assert len(set(valid_rows.tolist())) == int(expected_mask.sum())
    lookup = {
        int(row): value for row, value in zip(valid_rows.tolist(), valid_actions, strict=True)
    }
    for row, value in zip(
        np.asarray(sample.supervision_indices)[~expected_mask].tolist(),
        np.asarray(sample.supervision_actions)[~expected_mask],
        strict=True,
    ):
        assert row in lookup, "a padded action row must repeat an already-counted row"
        assert np.array_equal(value, lookup[row])
    assert lookup[11].tolist() == base[target]["actions"][11].tolist()
    masked_sum = float(valid_actions.sum())
    unmasked_sum = float(np.asarray(sample.supervision_actions).sum())
    assert unmasked_sum != pytest.approx(masked_sum)

    state = skip_without_state()
    targets = d.sample_to_training_targets(sample)
    assert isinstance(targets, state.TrainingTargets)
    assert targets.action_mask.tolist() == expected_mask.tolist()
    assert targets.actions.shape == (10, 7)


def test_history_padding_keeps_shape_but_adds_no_evidence(tmp_path, libero_writer) -> None:
    root = tmp_path / "libero"
    stored = write_libero_matrix(root, libero_writer)
    manifest = build_manifest(root)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    window = raw.WindowOptions(
        observation_stride=2, rnn_frames=4, burst_frames=2, action_horizon=10, action_offset=0
    )
    table = table_for(manifest, partition, window, stride=2)
    anchor = anchor_index(table, target, 2)
    dataset = d.ManifestWindowDataset(root, manifest, partition, table, window_options=window)
    sample = dataset[anchor]

    assert sample.observation_indices.tolist() == [0, 0, 0, 2]
    assert sample.burst_indices.tolist() == [[0, 0], [0, 0], [0, 0], [1, 2]]
    assert sample.new_measurement.tolist() == [False, False, False, True]
    assert sample.burst_new_measurement.tolist() == [
        [True, False],
        [False, False],
        [False, False],
        [True, True],
    ]
    # Eight read positions carry only three distinct source frames.
    assert sample.unique_measurement_count == 3
    assert sample.unique_measurement_count == int(sample.burst_new_measurement.sum())
    assert sample.unique_measurement_count < sample.burst_new_measurement.size
    assert sample.burst_values[0].shape == (4, 2, 3, 8, 8)
    assert sample.burst_values[1].shape == (4, 2, 3, 6, 10)
    assert sample.observation_indices[0] == 0  # clipped history, not a negative index

    frames = d.sample_frames(sample)
    assert frames["libero_agentview"].new_measurement.tolist() == [False, False, False, True]
    assert frames["libero_agentview"].is_new_measurement is False


def test_future_action_perturbation_leaves_the_inference_input_unchanged(
    tmp_path, libero_writer
) -> None:
    """Only RAW, proprio, language and executed actions may enter the inference input."""
    root = tmp_path / "libero"
    records: dict[int, dict] = {}
    for episode in (18, 19, 20):
        records[episode] = libero_writer(root, LIBERO_SUITE_A, episode, 0, num_frames=24)
    manifest_a = build_manifest(root)
    target = sorted(record["relative_path"] for record in records.values())[0]
    partition = manifest_a.partition_of(target)
    window = raw.WindowOptions(
        observation_stride=2, rnn_frames=3, burst_frames=2, action_horizon=10, action_offset=0
    )
    stats = libero_quantile_stats()
    table_a = table_for(manifest_a, partition, window, stride=2)
    anchor = anchor_index(table_a, target, 4)
    sample_a = d.ManifestWindowDataset(
        root, manifest_a, partition, table_a, window_options=window, normalisation=stats
    )[anchor]

    # Rewrite the same files with different actions strictly after the anchor, then read the
    # same anchor again through a fresh dataset (the loader caches arrays per instance).
    for episode, record in records.items():
        changed = record["actions"].copy()
        changed[5:] += 0.5
        libero_writer(
            root,
            LIBERO_SUITE_A,
            episode,
            0,
            num_frames=24,
            actions=changed,
            robot_states=record["robot_states"],
        )
    manifest_b = build_manifest(root)
    assert manifest_b.partitions == manifest_a.partitions
    table_b = table_for(manifest_b, partition, window, stride=2)
    assert table_b.entries == table_a.entries
    assert table_b.sha256() == table_a.sha256()
    sample_b = d.ManifestWindowDataset(
        root, manifest_b, partition, table_b, window_options=window, normalisation=stats
    )[anchor]

    assert not np.array_equal(sample_a.supervision_actions, sample_b.supervision_actions)
    assert np.array_equal(sample_a.burst_values[0], sample_b.burst_values[0])
    assert np.array_equal(sample_a.burst_values[1], sample_b.burst_values[1])
    assert np.array_equal(sample_a.proprio, sample_b.proprio)
    assert sample_a.observation_indices.tolist() == sample_b.observation_indices.tolist()
    assert sample_a.burst_indices.tolist() == sample_b.burst_indices.tolist()
    assert sample_a.unique_measurement_count == sample_b.unique_measurement_count
    assert sample_a.new_measurement.tolist() == sample_b.new_measurement.tolist()
    assert sample_a.executed_action is None and sample_a.executed_action_available is False

    assert_measurement_frames_equal(d.sample_frames(sample_a), d.sample_frames(sample_b))

    state = skip_without_state()
    inference_a = d.sample_to_inference_input(sample_a)
    inference_b = d.sample_to_inference_input(sample_b)
    assert isinstance(inference_a, state.InferenceInput)
    assert_inference_inputs_equal(inference_a, inference_b)
    targets_a = d.sample_to_training_targets(sample_a)
    targets_b = d.sample_to_training_targets(sample_b)
    assert not np.array_equal(targets_a.actions, targets_b.actions)
    assert targets_a.action_mask.tolist() == targets_b.action_mask.tolist()


def test_multi_camera_shapes_are_loaded_separately(tmp_path, libero_writer) -> None:
    """Two cameras of one episode with different resolutions must not be mixed."""
    root = tmp_path / "libero"
    stored = write_libero_matrix(root, libero_writer)
    manifest = build_manifest(root)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    table = table_for(manifest, partition, LIBERO_WINDOW)
    anchor = anchor_index(table, target, 3)
    dataset = d.ManifestWindowDataset(
        root, manifest, partition, table, window_options=LIBERO_WINDOW
    )
    sample = dataset[anchor]

    assert sample.camera_ids == ("libero_agentview", "libero_wrist")
    assert sample.burst_values[0].shape == (1, 1, 3, 8, 8)
    assert sample.burst_values[1].shape == (1, 1, 3, 6, 10)
    assert sample.pixel_valid_mask[0].shape == (1, 1, 8, 8)
    assert sample.pixel_valid_mask[1].shape == (1, 1, 6, 10)
    assert np.allclose(
        sample.burst_values[0][0, 0],
        np.moveaxis(stored[target]["agentview"][3], -1, 0).astype(np.float32) / 255.0,
    )
    assert np.allclose(
        sample.burst_values[1][0, 0],
        np.moveaxis(stored[target]["wrist"][3], -1, 0).astype(np.float32) / 255.0,
    )
    available = sample.measurement_payload("libero_agentview")["metadata_available"]
    assert available["timestamp"] is False
    assert available["executed_action"] is False

    frames = d.sample_frames(sample)
    assert frames["libero_agentview"].values.shape == (1, 3, 8, 8)
    assert frames["libero_wrist"].values.shape == (1, 3, 6, 10)
    state = skip_without_state()
    inference = d.sample_to_inference_input(sample)
    assert isinstance(inference, state.InferenceInput)
    assert inference.bundle.frames["libero_agentview"].values.shape == (1, 3, 8, 8)
    assert inference.bundle.frames["libero_wrist"].values.shape == (1, 3, 6, 10)


# --------------------------------------------------------------------------------------
# the manifest path never recomputes a random split
# --------------------------------------------------------------------------------------


def test_manifest_dataset_never_calls_compute_split(tmp_path, libero_writer, monkeypatch) -> None:
    root = tmp_path / "libero"
    stored = write_libero_matrix(root, libero_writer)
    manifest = build_manifest(root)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    table = table_for(manifest, partition, LIBERO_WINDOW)

    calls: list[dict] = []

    def _forbidden(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("compute_split must not be called on the manifest path")

    monkeypatch.setattr(raw, "compute_split", _forbidden)
    # Control: the frozen two-sided loader really does reach the patched function.
    with pytest.raises(AssertionError):
        raw.RawVLAWindowDataset(root, dataset_format="libero").split_manifest()
    assert len(calls) == 1, "the monkeypatch must really intercept compute_split"
    calls.clear()

    dataset = d.ManifestWindowDataset(
        root, manifest, partition, table, window_options=LIBERO_WINDOW
    )
    for index in (0, 1, len(dataset) - 1):
        sample = dataset[index]
        assert sample.split == partition
        assert sample.provenance["split_source"] == "explicit_partition_manifest"
        assert sample.provenance["partition"] == partition
        assert sample.episode.relative_path in set(manifest.partitions[partition])
    assert calls == []

    # Two fresh dataset objects map the same index to the same sample: no hidden draw order.
    first = d.ManifestWindowDataset(root, manifest, partition, table, window_options=LIBERO_WINDOW)[
        17
    ]
    second = d.ManifestWindowDataset(
        root, manifest, partition, table, window_options=LIBERO_WINDOW
    )[17]
    assert first.episode.relative_path == second.episode.relative_path
    assert first.observation_indices.tolist() == second.observation_indices.tolist()
    assert np.array_equal(first.burst_values[0], second.burst_values[0])


# --------------------------------------------------------------------------------------
# partitions, groups and the manifest round-trip
# --------------------------------------------------------------------------------------


def test_partition_manifest_is_group_pure_and_read_back_verbatim(tmp_path, libero_writer) -> None:
    root = tmp_path / "libero"
    write_libero_matrix(root, libero_writer)
    manifest = build_manifest(root, seed=7)
    result = d.verify_partitions(manifest)

    assert result["violations"] == []
    assert result["disjoint"] is True
    assert result["groups_single_partition"] is True
    assert result["counts"]["files"] == 16
    assert result["counts"]["groups"] == 8
    assert sum(p["files"] for p in result["counts"]["partitions"].values()) == 16
    assert (
        sum(p["frames"] for p in result["counts"]["partitions"].values())
        == result["counts"]["frames"]
    )
    total_frames = result["counts"]["frames"]
    # Four groups per task: preserving all three partitions requires 2/1/1.
    assert result["counts"]["partitions"]["train"]["frames"] / total_frames == 0.5
    assert result["counts"]["partitions"]["dev"]["frames"] > 0
    assert result["counts"]["partitions"]["calibration"]["frames"] > 0

    for suite in LIBERO_SUITES:
        for episode in LIBERO_EPISODES:
            variant_0 = f"data/libero/raw/{suite}/episode_{episode:06d}/variant_00.npz"
            variant_1 = f"data/libero/raw/{suite}/episode_{episode:06d}/variant_01.npz"
            group_0 = manifest.group_of(variant_0)
            group_1 = manifest.group_of(variant_1)
            assert group_0 == group_1, "both variants must sit in one source-trajectory group"
            assert set(manifest.groups[group_0]) == {variant_0, variant_1}
            assert manifest.partition_of(variant_0) == manifest.partition_of(variant_1)
            assert manifest.entry_of(variant_0).source_trajectory_id.endswith(
                f"episode_{episode:06d}"
            )
            assert manifest.entry_of(variant_0).illumination_group == "light_0"
            assert manifest.entry_of(variant_1).illumination_group == "light_1"

    back = d.PartitionManifest.from_json(manifest.to_json())
    assert back == manifest
    assert back.content_hash() == manifest.content_hash()
    assert back.partitions == manifest.partitions
    assert back.groups == manifest.groups
    for entry in manifest.entries:
        assert back.partition_of(entry.relative_path) == manifest.partition_of(entry.relative_path)
        assert back.group_of(entry.relative_path) == manifest.group_of(entry.relative_path)

    with pytest.raises(KeyError):
        manifest.partition_of("data/libero/raw/none/variant_00.npz")
    with pytest.raises(KeyError):
        manifest.group_of("data/libero/raw/none/variant_00.npz")
    with pytest.raises(ValueError):
        d.build_partition_manifest([], dataset_format="libero", cache_root=root, seed=0)


def test_declared_scene_ids_merge_groups_through_connected_components(
    tmp_path, libero_writer
) -> None:
    root = tmp_path / "libero"
    first = libero_writer(root, LIBERO_SUITE_A, 18, 0, extra={"scene_id": np.array("scene_shared")})
    second = libero_writer(
        root, LIBERO_SUITE_B, 19, 0, extra={"scene_id": np.array("scene_shared")}
    )
    third = libero_writer(root, LIBERO_SUITE_A, 20, 0, extra={"scene_id": np.array("scene_other")})
    manifest = build_manifest(root)

    shared_group = manifest.group_of(first["relative_path"])
    assert shared_group == manifest.group_of(second["relative_path"])
    assert shared_group != manifest.group_of(third["relative_path"])
    assert set(manifest.groups[shared_group]) == {
        first["relative_path"],
        second["relative_path"],
    }
    assert manifest.partition_of(first["relative_path"]) == manifest.partition_of(
        second["relative_path"]
    )
    assert manifest.entry_of(first["relative_path"]).scene_id == "scene_shared"
    assert "connected components" in manifest.notes


def test_holdout_slices_are_feasible_and_group_complete(tmp_path, libero_writer) -> None:
    root = tmp_path / "libero"
    scenes = {
        18: ("scene_a", ("mug",)),
        19: ("scene_a", ("plate",)),
        20: ("scene_b", ("plate",)),
        21: ("scene_b", ("bowl",)),
    }
    for episode, (scene, objects) in scenes.items():
        libero_writer(
            root,
            LIBERO_SUITE_A,
            episode,
            0,
            extra={
                "scene_id": np.array(scene),
                "object_ids": np.array(list(objects)),
            },
        )
    manifest = build_manifest(root)

    assert set(manifest.holdout_slices) == {"cross_scene", "cross_object"}
    for name, paths in manifest.holdout_slices.items():
        assert paths, f"{name} slice should be feasible for this fixture"
        held = set(paths)
        for group_paths in manifest.groups.values():
            members = set(group_paths)
            assert members <= held or not (members & held), f"{name} splits a group"
        assert held < {entry.relative_path for entry in manifest.entries}
    # scene_a has fewer/equal frames and is chosen by the deterministic tie-break; both held
    # groups declare scene_a.
    for path in manifest.holdout_slices["cross_scene"]:
        assert manifest.entry_of(path).scene_id == "scene_a"
    assert "cross_scene slice holds out scene 'scene_a'" in manifest.notes
    assert "cross_object slice holds out object 'bowl'" in manifest.notes


def test_verify_partitions_detects_a_crossed_group_and_a_duplicate_path(
    tmp_path, libero_writer
) -> None:
    root = tmp_path / "libero"
    write_libero_matrix(root, libero_writer)
    manifest = build_manifest(root)

    moved = manifest.partitions["dev"][0]
    source = manifest.partition_of(moved)
    target_partition = "calibration" if source != "calibration" else "train"
    partitions = {
        name: tuple(path for path in paths if path != moved)
        for name, paths in manifest.partitions.items()
    }
    partitions[target_partition] = tuple(sorted((*partitions[target_partition], moved)))
    crossed = dataclasses.replace(manifest, partitions=partitions)
    crossed_result = d.verify_partitions(crossed)
    assert crossed_result["disjoint"] is True
    assert crossed_result["groups_single_partition"] is False
    assert any("spans partitions" in violation for violation in crossed_result["violations"])

    duplicated = dataclasses.replace(
        manifest,
        partitions={
            **manifest.partitions,
            "train": (*manifest.partitions["train"], manifest.partitions["dev"][0]),
        },
    )
    duplicate_result = d.verify_partitions(duplicated)
    assert duplicate_result["disjoint"] is False
    assert any(
        "more than one partition" in violation for violation in duplicate_result["violations"]
    )

"""P0 data tests, part 2: the anchor table.

The fixtures are contract NPZ files written into ``tmp_path`` (see ``conftest.py``), not real
RawVLA data. The tests check the declared anchor geometry (grid position vs source frame, history
indices, action bounds and mask hash), deterministic serialisation, and that strided shards are
disjoint with a complete union.
"""

from __future__ import annotations

import numpy as np
import pytest

from owac.frameworks.n02_full import data as d
from owac.raw import rawvla_sequence as raw

from .conftest import LIBERO_SUITE_A, ROBOTWIN_TASKS

LIBERO_WINDOW = raw.WindowOptions(
    observation_stride=1, rnn_frames=1, burst_frames=1, action_horizon=10, action_offset=0
)
ROBOTWIN_WINDOW = raw.WindowOptions(
    observation_stride=1, rnn_frames=1, burst_frames=1, action_horizon=50, action_offset=1
)
HEX_DIGITS = set("0123456789abcdef")


def libero_cache(tmp_path, writer, *, num_frames: int = 24) -> tuple:
    stored = {}
    for episode in (18, 19, 20, 21, 22, 23):
        record = writer(tmp_path / "libero", LIBERO_SUITE_A, episode, 0, num_frames=num_frames)
        stored[record["relative_path"]] = record
    root = tmp_path / "libero"
    loader = raw.RawVLAWindowDataset(root, dataset_format="libero", hash_episodes=True)
    manifest = d.build_partition_manifest(
        loader.episodes(), dataset_format="libero", cache_root=root, seed=7
    )
    return root, manifest, stored


def robotwin_cache(tmp_path, writer, *, num_frames: int = 60) -> tuple:
    tasks = (*ROBOTWIN_TASKS, "open_laptop")
    stored = {}
    for task in tasks:
        record = writer(tmp_path / "robotwin", task, 18, num_frames=num_frames)
        stored[record["relative_path"]] = record
    root = tmp_path / "robotwin"
    loader = raw.RawVLAWindowDataset(root, dataset_format="robotwin", hash_episodes=True)
    manifest = d.build_partition_manifest(
        loader.episodes(), dataset_format="robotwin", cache_root=root, seed=7
    )
    return root, manifest, stored


def find_anchor(table: d.AnchorTable, relative_path: str, current: int) -> d.AnchorRecord:
    for anchor in table:
        if anchor.relative_path == relative_path and anchor.current_frame_index == current:
            return anchor
    raise AssertionError(f"no anchor {relative_path} t={current}")


def test_anchor_rows_declare_the_declared_window_geometry(tmp_path, libero_writer) -> None:
    _, manifest, stored = libero_cache(tmp_path, libero_writer)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    table = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )

    record = find_anchor(table, target, 5)
    assert record.t == 5 and record.current_frame_index == 5
    assert record.history_indices == (5,)
    assert (record.action_start, record.action_stop) == (5, 15)
    assert record.partition == partition
    assert record.episode_id == manifest.entry_of(target).episode_id
    assert len(record.action_mask_hash) == 64
    assert set(record.action_mask_hash) <= HEX_DIGITS

    # The tail anchor's mask really is partial: t=20 with num_frames=24 masks four entries.
    tail = find_anchor(table, target, 20)
    expected = (np.arange(20, 30) >= 0) & (np.arange(20, 30) <= 23)
    assert expected.tolist() == [True, True, True, True, False, False, False, False, False, False]
    assert tail.action_mask_hash != record.action_mask_hash

    dataset = d.ManifestWindowDataset(
        tmp_path / "libero", manifest, partition, table, window_options=LIBERO_WINDOW
    )
    index = next(i for i, anchor in enumerate(table) if anchor == tail)
    sample = dataset[index]
    # The loader clips the tail: positions past the episode repeat the last stored row.
    assert sample.supervision_indices.tolist() == [20, 21, 22, 23, 23, 23, 23, 23, 23, 23]
    assert sample.supervision_mask.tolist() == expected.tolist()
    assert sample.padding_contribution == pytest.approx(0.6)
    assert set(np.asarray(sample.supervision_indices)[expected].tolist()) == {20, 21, 22, 23}
    assert set(np.asarray(sample.supervision_indices)[~expected].tolist()) == {23}


def test_anchor_rows_follow_the_robotwin_offset_and_horizon(tmp_path, robotwin_writer) -> None:
    _, manifest, stored = robotwin_cache(tmp_path, robotwin_writer)
    target = next(path for path in stored if f"/{ROBOTWIN_TASKS[0]}/" in path)
    partition = manifest.partition_of(target)
    table = d.build_anchor_table(
        manifest,
        partition,
        window=ROBOTWIN_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=50,
        action_offset=1,
    )
    record = find_anchor(table, target, 9)
    assert (record.action_start, record.action_stop) == (10, 60)
    assert record.history_indices == (9,)

    dataset = d.ManifestWindowDataset(
        tmp_path / "robotwin", manifest, partition, table, window_options=ROBOTWIN_WINDOW
    )
    index = next(i for i, anchor in enumerate(table) if anchor == record)
    sample = dataset[index]
    assert sample.supervision_indices.tolist() == list(range(10, 60))
    # The window ends exactly on the last stored row (index 59), so nothing is padded.
    assert sample.supervision_mask.all()
    assert sample.padding_contribution == 0.0


def test_anchor_stride_separates_grid_position_from_source_frame(tmp_path, libero_writer) -> None:
    _, manifest, stored = libero_cache(tmp_path, libero_writer)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    window = raw.WindowOptions(
        observation_stride=2, rnn_frames=3, burst_frames=2, action_horizon=10, action_offset=0
    )
    table = d.build_anchor_table(
        manifest,
        partition,
        window=window,
        stride=4,
        rnn_frames=3,
        burst_frames=2,
        action_horizon=10,
        action_offset=0,
    )
    entries = [entry for entry in table if entry.relative_path == target]
    assert [entry.t for entry in entries[:3]] == [0, 1, 2]
    assert [entry.current_frame_index for entry in entries[:3]] == [0, 4, 8]
    assert entries[0].history_indices == (0, 0, 0)
    assert entries[2].history_indices == (4, 6, 8)
    assert table.window["observation_stride"] == 2
    assert table.window["rnn_frames"] == 3
    assert table.window["burst_frames"] == 2
    assert table.notes.startswith("anchor grid stride=4; history spacing=2")


def test_anchor_table_serialisation_and_sha_are_reproducible(tmp_path, libero_writer) -> None:
    _, manifest, _ = libero_cache(tmp_path, libero_writer)
    partition = manifest.partition_of(manifest.partitions["train"][0])
    first = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )
    second = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )
    assert first.sha256() == second.sha256()
    assert list(first.iter_indices()) == list(range(len(first)))

    other_stride = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=2,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )
    assert other_stride.sha256() != first.sha256()

    back = d.AnchorTable.from_json(first.to_json())
    assert back == first
    assert back.sha256() == first.sha256()
    assert list(back.iter_indices()) == list(first.iter_indices())


def test_shards_are_disjoint_and_their_union_is_the_whole_table(tmp_path, libero_writer) -> None:
    _, manifest, stored = libero_cache(tmp_path, libero_writer)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    table = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )
    everything = set(table.iter_indices())
    assert len(everything) == len(table) > 0

    for world_size in (1, 2, 3, 5):
        index_sets = []
        for rank in range(world_size):
            shard = table.shard(world_size, rank)
            indices = set(shard.iter_indices())
            index_sets.append(indices)
            assert indices <= everything
            assert all(index % world_size == rank for index in indices)
            assert [entry for entry in shard] == [table[index] for index in shard.iter_indices()]
            assert shard.world_size == world_size and shard.rank == rank
        union: set[int] = set()
        for left in range(world_size):
            for right in range(left + 1, world_size):
                assert not (index_sets[left] & index_sets[right]), "shards must be disjoint"
            union |= index_sets[left]
        assert union == everything, "shards must cover every anchor"
        sizes = sorted(len(index_set) for index_set in index_sets)
        assert sizes[-1] - sizes[0] <= 1

    with pytest.raises(ValueError):
        table.shard(2, 2)
    with pytest.raises(ValueError):
        table.shard(0, 0)


def test_max_anchors_truncates_deterministically(tmp_path, libero_writer) -> None:
    _, manifest, stored = libero_cache(tmp_path, libero_writer)
    target = sorted(stored)[0]
    partition = manifest.partition_of(target)
    full = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )
    capped = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
        max_anchors=10,
    )
    again = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
        max_anchors=10,
    )
    assert len(capped) == 10
    assert capped.sha256() == again.sha256()
    assert list(capped.iter_indices()) == list(range(10))
    assert capped.entries == full.entries[:10]
    assert "truncated to 10 anchors" in capped.notes
    assert "truncated" not in full.notes
    with pytest.raises(ValueError):
        d.build_anchor_table(
            manifest,
            partition,
            window=LIBERO_WINDOW,
            stride=1,
            rnn_frames=1,
            burst_frames=1,
            action_horizon=10,
            action_offset=0,
            max_anchors=0,
        )


def test_action_mask_hash_tracks_identity_and_padding(tmp_path, libero_writer) -> None:
    _, manifest, _ = libero_cache(tmp_path, libero_writer)
    first = manifest.partitions["train"][0]
    partition = manifest.partition_of(first)
    table = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )
    siblings = [path for path in manifest.partitions[partition] if path != first]
    assert siblings, "the partition must contain more than one episode for this comparison"
    second = siblings[-1]
    first_anchor = find_anchor(table, first, 5)
    same_file_other_start = find_anchor(table, first, 6)
    other_file_same_start = find_anchor(table, second, 5)
    assert first_anchor.action_mask_hash != same_file_other_start.action_mask_hash
    assert first_anchor.action_mask_hash != other_file_same_start.action_mask_hash
    assert first_anchor.action_start == other_file_same_start.action_start
    # The same anchor recomputed from the same manifest is bit-identical.
    again = d.build_anchor_table(
        manifest,
        partition,
        window=LIBERO_WINDOW,
        stride=1,
        rnn_frames=1,
        burst_frames=1,
        action_horizon=10,
        action_offset=0,
    )
    assert find_anchor(again, first, 5) == first_anchor


def test_anchor_table_validates_inputs(tmp_path, libero_writer) -> None:
    _, manifest, _ = libero_cache(tmp_path, libero_writer)
    with pytest.raises(KeyError):
        d.build_anchor_table(
            manifest,
            "not_a_partition",
            window=LIBERO_WINDOW,
            stride=1,
            rnn_frames=1,
            burst_frames=1,
            action_horizon=10,
            action_offset=0,
        )
    with pytest.raises(ValueError):
        d.build_anchor_table(
            manifest,
            "train",
            window=LIBERO_WINDOW,
            stride=0,
            rnn_frames=1,
            burst_frames=1,
            action_horizon=10,
            action_offset=0,
        )
    inconsistent = raw.WindowOptions(
        observation_stride=1, rnn_frames=3, burst_frames=1, action_horizon=10, action_offset=0
    )
    with pytest.raises(ValueError):
        d.build_anchor_table(
            manifest,
            "train",
            window=inconsistent,
            stride=1,
            rnn_frames=1,
            burst_frames=1,
            action_horizon=10,
            action_offset=0,
        )
    with pytest.raises(ValueError):
        d.build_anchor_table(
            manifest,
            "train",
            window=LIBERO_WINDOW,
            stride=1,
            rnn_frames=1,
            burst_frames=1,
            action_horizon=4,
            action_offset=0,
        )

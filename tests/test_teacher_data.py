"""Fixture tests for strict source pairing and causal teacher windows."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from owac.raw.teacher_data import (
    CAMERA_ORDER,
    TeacherData,
    TeacherDataError,
    assign_source_splits,
    load_source_manifest,
    load_teacher_episode,
    make_window,
)


def _write_episode(root: Path, index: int, *, frames: int = 6) -> Path:
    rng = np.random.default_rng(index)
    path = root / f"episode_{index:02d}.npz"
    raw_agent = rng.integers(0, 256, size=(frames, 4, 4, 3), dtype=np.uint8)
    raw_wrist = rng.integers(0, 256, size=(frames, 4, 4, 3), dtype=np.uint8)
    rgb_agent = rng.integers(0, 256, size=(frames, 4, 4, 3), dtype=np.uint8)
    rgb_wrist = rng.integers(0, 256, size=(frames, 4, 4, 3), dtype=np.uint8)
    actions = rng.normal(size=(frames, 7)).astype(np.float32)
    states = rng.normal(size=(frames, 17)).astype(np.float32)
    robot = rng.normal(size=(frames, 8)).astype(np.float32)
    dones = np.zeros(frames, dtype=np.uint8)
    dones[-1] = 1
    np.savez(
        path,
        agentview_raw_uint8=raw_agent,
        wrist_raw_uint8=raw_wrist,
        agentview_rgb_uint8=rgb_agent,
        wrist_rgb_uint8=rgb_wrist,
        actions=actions,
        states=states,
        robot_states=robot,
        dones=dones,
        timestamps=np.arange(frames, dtype=np.float64) * 0.1,
        source_frame_indices=np.arange(frames, dtype=np.int64) * 2,
        native_anchor_mask=np.ones(frames, dtype=np.bool_),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "noops_filtered": 2,
                    "representation": "raw",
                    "control_freq_hz": 20,
                }
            )
        ),
    )
    return path


@pytest.fixture()
def teacher_manifest(tmp_path):
    ids = [f"libero_goal/{i:03d}" for i in range(50)]
    splits = assign_source_splits(ids, seed=101)
    sources = []
    for index, source_id in enumerate(ids):
        path = _write_episode(tmp_path, index)
        sources.append(
            {
                "source_demo_id": source_id,
                "split": splits[source_id],
                "raw_path": str(path),
                "rgb_path": str(path),
                "raw_sha256": "a" * 64,
                "rgb_sha256": "a" * 64,
                "num_frames": 6,
                "task_name": "put_the_cream_cheese_in_the_bowl",
                "lighting": "Normal",
                "camera_order": ["agentview", "wrist"],
                "metadata": {"demo_index": index},
            }
        )
    payload = {
        "schema": "owac.raw-teacher.sources.v1",
        "task": {
            "suite": "libero_goal",
            "task_id": 6,
            "task_name": "put_the_cream_cheese_in_the_bowl",
            "instruction": "put the cream cheese in the bowl",
            "lighting": "Normal",
        },
        "provenance": {"author_revision": "ba9c56b", "replay_domain": "Normal"},
        "source_hdf5_sha256": "a" * 64,
        "split_seed": 101,
        "path_semantics": "absolute",
        "sources": sources,
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload))
    return path


def test_manifest_is_source_level_40_5_5_and_unique(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path, strict=True)
    assert len(manifest.sources) == 50
    assert [len(manifest.sources_for_split(s)) for s in ("train", "validation", "test")] == [
        40,
        5,
        5,
    ]
    all_ids = [item.source_demo_id for item in manifest.sources]
    assert len(set(all_ids)) == 50
    splits = {item.source_demo_id: item.split for item in manifest.sources}
    validation = {item.source_demo_id for item in manifest.sources_for_split("validation")}
    train = {item.source_demo_id for item in manifest.sources_for_split("train")}
    assert not validation & train
    assert all(splits[item] == "validation" for item in validation)


def test_npz_pair_and_uint8_contract(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    source = manifest.sources[0]
    episode = load_teacher_episode(source, repo_root=tmp_path, path_semantics="absolute")
    assert episode.raw["agentview"].dtype == np.uint8
    assert episode.raw["agentview"].shape == (6, 4, 4, 3)
    assert episode.source_frame_indices.tolist() == [0, 2, 4, 6, 8, 10]
    window = make_window(episode, 3)
    assert window.raw_uint8.shape == (6, 6, 2, 4, 4, 3)
    assert window.rgb_uint8.shape == window.raw_uint8.shape
    assert window.camera_order == CAMERA_ORDER
    assert np.issubdtype(window.raw_uint8.dtype, np.uint8)
    assert np.all(window.source_frame_indices <= episode.source_frame_indices[3])
    assert np.all(window.source_frame_indices >= 0)


def test_causal_window_left_clipping_and_end_action_mask(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    episode = load_teacher_episode(
        manifest.sources[0], repo_root=tmp_path, path_semantics="absolute"
    )
    first = make_window(episode, 0)
    assert np.all(first.source_frame_indices == 0)
    assert first.action_mask.tolist() == [True] * 6 + [False] * 4
    last = make_window(episode, 5)
    assert last.action_mask.tolist() == [True] + [False] * 9
    assert np.all(last.source_frame_indices <= 10)
    assert last.padded_actions.shape == (10, 32)
    assert last.padded_state.shape == (32,)
    assert np.all(last.padded_actions[1:] == 0)
    assert np.all(last.padded_actions[:, :7][last.action_mask] == last.actions[last.action_mask])


def test_quantile_normalization_preserves_mask_not_padding(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    window = make_window(
        load_teacher_episode(manifest.sources[0], repo_root=tmp_path, path_semantics="absolute"), 2
    )
    stats = SimpleNamespace(
        action={"q01": np.full(7, -2.0), "q99": np.full(7, 4.0)},
        state={"q01": np.full(17, -2.0), "q99": np.full(17, 2.0)},
    )
    normalized = window.normalized(stats)
    assert normalized["padded_actions"].shape == (10, 32)
    assert normalized["action_mask"].tolist() == window.action_mask.tolist()
    expected = np.clip(2.0 * (window.actions - (-2.0)) / 6.0 - 1.0, -1.0, 1.0)
    assert np.allclose(normalized["actions"][window.action_mask], expected[window.action_mask])
    assert np.all(np.abs(normalized["actions"]) <= 1.0)
    assert np.any(normalized["padded_state"][:8] != 0)
    assert np.allclose(normalized["padded_state"][8:], 0.0)
    configured = window.normalized(stats, state_pad_value=-2.0)
    assert np.allclose(configured["padded_state"][8:], -2.0)


def test_cache_validation_and_legacy_hash_mismatch_is_ignored(
    teacher_manifest, tmp_path, monkeypatch
):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    source = manifest.sources[0]
    first = load_teacher_episode(source, repo_root=tmp_path, path_semantics="absolute")
    second = load_teacher_episode(source, repo_root=tmp_path, path_semantics="absolute")
    assert first.actions is not second.actions or np.array_equal(first.actions, second.actions)
    data = TeacherData(manifest, repo_root=tmp_path, cache_entries=1)
    assert data.episode(source.source_demo_id).source.source_demo_id == source.source_demo_id
    # Legacy hash mismatch fields are ignored; arrays are still structurally valid.
    from dataclasses import replace

    bad = replace(source, raw_sha256="0" * 64)
    ignored = load_teacher_episode(bad, repo_root=tmp_path, path_semantics="absolute")
    assert np.array_equal(ignored.raw["agentview"], first.raw["agentview"])


def test_split_windows_never_cross_sources(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    data = TeacherData(manifest, repo_root=tmp_path, cache_entries=2)
    windows = list(data.iter_windows("validation"))
    assert windows
    validation_ids = {item.source_demo_id for item in manifest.sources_for_split("validation")}
    assert {window.source_demo_id for window in windows} <= validation_ids
    assert all(window.split == "validation" for window in windows)


def test_policy_window_excludes_privileged_simulation_state(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    data = TeacherData(manifest, repo_root=tmp_path)
    source = manifest.sources[0]
    episode = data.episode(source.source_demo_id)
    window = make_window(episode, 2)
    assert episode.states.shape[1] == 17  # arbitrary simulator-only fixture width
    assert window.state.shape == (8,)
    np.testing.assert_array_equal(window.state, episode.robot_states[2])
    assert not np.array_equal(window.state, episode.states[2, :8])
    raw_only = data.episode_raw(source.source_demo_id)
    raw_window = make_window(raw_only, 2)
    assert raw_window.rgb_uint8.size == 0
    np.testing.assert_array_equal(raw_window.state, window.state)


def test_manifest_rejects_duplicate_mixed_lighting_and_silent_shrink(teacher_manifest, tmp_path):
    payload = json.loads(teacher_manifest.read_text())
    duplicate = json.loads(json.dumps(payload))
    duplicate["sources"][1]["source_demo_id"] = duplicate["sources"][0]["source_demo_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(duplicate))
    with pytest.raises(TeacherDataError, match="duplicate"):
        load_source_manifest(path, repo_root=tmp_path, strict=False)

    mixed = json.loads(json.dumps(payload))
    mixed["sources"][0]["lighting"] = "ExtremeLow"
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(mixed))
    with pytest.raises(TeacherDataError, match="Normal"):
        load_source_manifest(path, repo_root=tmp_path, strict=False)

    shrink = json.loads(json.dumps(payload))
    shrink["sources"] = shrink["sources"][:-1]
    path = tmp_path / "shrink.json"
    path.write_text(json.dumps(shrink))
    with pytest.raises(TeacherDataError, match="split counts|exactly 50"):
        load_source_manifest(path, repo_root=tmp_path, strict=True)


def test_episode_cache_returns_same_validated_episode(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    data = TeacherData(manifest, repo_root=tmp_path, cache_entries=1)
    source_id = manifest.sources[0].source_demo_id
    assert data.episode(source_id) is data.episode(source_id)


def test_canonical_state8_order_and_quaternion_conversion_nonzero():
    from owac.raw.teacher_data import canonical_policy_state, quaternion_to_axisangle

    robot9 = np.asarray(
        [
            [
                0.0362271,
                -0.03623069,
                -0.20543062,
                0.00756122,
                1.1849612,
                0.99967569,
                0.0167386,
                -0.01900751,
                0.00265439,
            ]
        ],
        dtype=np.float32,
    )
    state = canonical_policy_state(
        robot9[:, 2:5], quaternion_to_axisangle(robot9[:, 5:9]), robot9[:, 0:2]
    )
    assert state.shape == (1, 8)
    assert np.all(np.isfinite(state))
    assert not np.allclose(state, 0)
    assert np.allclose(state[0, :3], [-0.20543062, 0.00756122, 1.1849612])
    assert np.allclose(state[0, 6:8], [0.0362271, -0.03623069])
    assert np.linalg.norm(state[0, 3:6]) > 0


def test_native_anchor_and_nominal_dt_are_exposed(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    window = make_window(
        load_teacher_episode(manifest.sources[0], repo_root=tmp_path, path_semantics="absolute"),
        1,
    )
    assert window.anchor_eligible is True
    assert window.nominal_dt == pytest.approx(0.05)
    assert window.metadata["dense_observation_history"] is True


def test_anchor_index_is_lazy_and_does_not_load_pixels(teacher_manifest, tmp_path, monkeypatch):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    data = TeacherData(manifest, repo_root=tmp_path)
    monkeypatch.setattr(
        data,
        "episode",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pixels loaded")),
    )
    anchors = list(data.iter_anchors("validation"))
    assert anchors
    assert all(anchor.split == "validation" for anchor in anchors)


def test_source_block_sampler_uses_32_anchors_and_resumes(teacher_manifest, tmp_path):
    from owac.raw.teacher_data import TeacherWindowSampler

    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    data = TeacherData(manifest, repo_root=tmp_path)
    sampler = TeacherWindowSampler(data, split="train", seed=101)
    first = [sampler.next() for _ in range(8)]
    assert len({window.source_demo_id for window in first}) == 1
    state = sampler.state_dict()
    resumed = TeacherWindowSampler(data, split="train", seed=101)
    resumed.load_state_dict(state)
    assert resumed.next().source_demo_id == sampler.next().source_demo_id


def test_actual_pi05_normalization_stats_contract():
    from owac.raw.teacher_data import load_pi05_normalization_stats

    path = Path(
        "outputs/checkpoints/raw-distillation/pi05_libero_jax/"
        "assets/physical-intelligence/libero/norm_stats.json"
    )
    if not path.is_file():
        pytest.skip("actual pi05 norm_stats file is absent in this checkout")
    stats = load_pi05_normalization_stats(path)
    assert len(stats.action["q01"]) == 7
    assert len(stats.action["q99"]) == 7
    assert len(stats.state["q01"]) == 8
    assert len(stats.state["q99"]) == 8


def test_raw_only_view_never_opens_rgb_path(teacher_manifest, tmp_path):
    manifest = load_source_manifest(teacher_manifest, repo_root=tmp_path)
    source = manifest.sources[0]
    data = TeacherData(manifest, repo_root=tmp_path)
    episode = data.episode_raw(source.source_demo_id)
    assert episode.rgb == {}
    assert episode.raw["agentview"].shape[0] == source.num_frames

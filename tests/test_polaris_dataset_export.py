import copy
import json

import pytest

from owac.runtime.polaris_dataset_export import release_ready, select_latest_replays


def complete_fixture():
    protocol = {
        "source_episodes": 2,
        "source_control_frames": 5,
        "source_rgb_fields": ["external1", "external2", "wrist"],
    }
    coverage = {
        "episodes": 2,
        "source_ids": ["a", "b"],
        "source_frames": 5,
        "views": ["external1", "external2", "wrist"],
        "failed_or_unresolved_episodes": 0,
        "training_loader_validated": True,
        "per_episode_source_rgb_correspondence_review_complete": True,
    }
    return coverage, protocol


def test_one_successful_sample_cannot_be_published_as_complete_source_corpus():
    coverage, protocol = complete_fixture()
    coverage.update(episodes=1, source_ids=["a"], source_frames=3)
    with pytest.raises(ValueError, match="all original source episode"):
        release_ready(coverage, protocol)


def test_capture_coverage_does_not_replace_scene_correspondence_review():
    coverage, protocol = complete_fixture()
    assert release_ready(coverage, protocol)
    coverage["per_episode_source_rgb_correspondence_review_complete"] = False
    with pytest.raises(ValueError, match="correspondence review"):
        release_ready(coverage, protocol)


def test_raw_field_and_source_frame_gaps_block_full_release():
    coverage, protocol = complete_fixture()
    frames_missing = copy.deepcopy(coverage)
    frames_missing["source_frames"] = 4
    with pytest.raises(ValueError, match="control frames"):
        release_ready(frames_missing, protocol)
    coverage["views"].pop()
    with pytest.raises(ValueError, match="schema field"):
        release_ready(coverage, protocol)


def test_newer_failed_reconstruction_does_not_cherry_pick_older_success(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for root, start, rows in (
        (old, "2026-10-10T01:00:00Z", [("a", True), ("b", True)]),
        (new, "2026-10-10T02:00:00Z", [("a", False)]),
    ):
        root.mkdir()
        (root / "status.json").write_text(
            json.dumps(
                {
                    "started_at_utc": start,
                    "completed": [
                        {"id": identity, "infrastructure_error": False, "result": {"success": ok}}
                        for identity, ok in rows
                    ],
                }
            )
        )
    accepted, outcomes = select_latest_replays([new, old])
    assert [row["id"] for row in accepted] == ["b"]
    assert len(outcomes) == 3
    assert outcomes[-1]["physical_success"] is False

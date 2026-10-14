import json

import pytest

from owac.runtime.arena_droid import sha256
from owac.runtime.polaris_dataset_publish import validate_source_coverage


def test_matching_aggregate_counts_cannot_hide_wrong_original_episodes(tmp_path):
    dataset, inventory = tmp_path / "dataset", tmp_path / "inventory"
    dataset.mkdir()
    inventory.mkdir()
    (dataset / "coverage.json").write_text(
        json.dumps(
            {
                "episodes": 2,
                "source_ids": ["a", "wrong"],
                "source_frames": 5,
                "views": ["external1", "external2", "wrist"],
                "failed_or_unresolved_episodes": 0,
                "training_loader_validated": True,
                "per_episode_source_rgb_correspondence_review_complete": True,
            }
        )
    )
    manifest = inventory / "episodes.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in [{"id": "a", "frames": 2}, {"id": "b", "frames": 3}])
    )
    protocol = {
        "source_episodes": 2,
        "source_control_frames": 5,
        "source_rgb_fields": ["external1", "external2", "wrist"],
        "source_inventory_sha256": sha256(manifest),
    }
    with pytest.raises(ValueError, match="identities do not match"):
        validate_source_coverage(dataset, inventory, protocol)

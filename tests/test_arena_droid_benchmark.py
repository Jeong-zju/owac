import json
from pathlib import Path

import pytest

from owac.runtime.arena_droid_benchmark import identities, summarize, validate_snapshot


def protocol():
    return {
        "variations": [{"name": "a"}, {"name": "b"}],
        "environment_seeds": [0, 1],
        "expected_episodes": 4,
    }


def test_duplicate_and_undeclared_episode_cannot_complete_benchmark():
    rows = [{"variation": "a", "seed": 0, "success": True}]
    assert not summarize(protocol(), rows)["complete"]
    with pytest.raises(ValueError, match="Duplicated or undeclared"):
        summarize(protocol(), rows * 2)
    with pytest.raises(ValueError, match="Duplicated or undeclared"):
        summarize(protocol(), [{"variation": "a", "seed": 2, "success": False}])


def test_all_frozen_failures_remain_in_the_denominator():
    rows = [
        {"variation": name, "seed": seed, "success": seed == 0}
        for name, seed in identities(protocol())
    ]
    result = summarize(protocol(), rows)
    assert result["complete"]
    assert result["successes"] == 2
    assert result["episodes"] == 4
    assert result["success_rate"] == 0.5
    assert summarize(protocol(), [])["success_rate"] is None
    with pytest.raises(ValueError, match="boolean"):
        summarize(protocol(), [{"variation": "a", "seed": 0, "success": None}])


def test_full_protocol_has_all_nine_variations_and_100_unique_seeds():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/baselines/pi05-arena-droid-benchmark-v1.json").read_text())
    pairs = identities(config)
    assert len(pairs) == 900
    assert len(set(name for name, _ in pairs)) == 9
    assert set(seed for _, seed in pairs) == set(range(100))
    assert pairs[0][1] == pairs[8][1] == 0
    assert pairs[9][1] == 1


def test_false_cloud_completion_and_inconsistent_counts_are_rejected():
    rows = [{"variation": "a", "seed": 0, "success": False, "inference_calls": 3}]
    reported = {
        **summarize(protocol(), rows),
        "inference_calls": 3,
        "inference_roundtrip_p50_ms": 100,
        "inference_roundtrip_p95_ms": 110,
    }
    status = {"status": "running", "completed_episodes": 1}
    assert validate_snapshot(protocol(), status, reported, rows)["successes"] == 0
    with pytest.raises(ValueError, match="full frozen"):
        validate_snapshot(protocol(), {**status, "status": "complete"}, reported, rows)
    with pytest.raises(ValueError, match="episode records"):
        validate_snapshot(protocol(), status, {**reported, "successes": 1}, rows)
    with pytest.raises(ValueError, match="inference count"):
        validate_snapshot(protocol(), status, {**reported, "inference_calls": 4}, rows)

import numpy as np
import pytest

from owac.evaluation.baseline_protocol import (
    action_chunk,
    file_sha256,
    physical_success,
    wilson_interval,
    write_json,
)
from owac.runtime.baseline_archive import verify_completed
from owac.training.pi05_sim import normalization


def test_chunks_repeat_last_action_without_leaking_next_episode():
    actions = np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(action_chunk(actions, 1, 5), [[2], [3], [3], [3], [3]])
    with pytest.raises(ValueError):
        action_chunk(actions, 3, 1)


def test_wilson_uncertainty_for_zero_and_all_success():
    assert wilson_interval(0, 100) == pytest.approx([0, 0.03699349820698568])
    assert wilson_interval(100, 100) == pytest.approx([0.9630065017930143, 1])
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_physical_success_rejects_held_mug_and_predicate_only():
    values = dict(predicate=True, lift=0.15, xy=0.001, height=0.075, fingers=[0.04, 0.04])
    assert physical_success(**values)
    assert not physical_success(**(values | {"fingers": [0.003, 0.003]}))
    assert not physical_success(**(values | {"fingers": [0.03, 0.03]}))
    assert not physical_success(**(values | {"open_command": False}))
    assert not physical_success(**(values | {"lift": 0.0}))
    assert not physical_success(**(values | {"predicate": False}))


def test_normalization_handles_constant_orientation_without_division_by_zero():
    values = np.array([[0, 3.14], [1, 3.14], [2, 3.14]], dtype=np.float32)
    stats = normalization(values)
    normalized = (values - stats["center"]) / stats["scale"]
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[:, 1], 0)


def test_archive_rejects_duplicate_seed_even_if_episode_count_and_success_count_match(tmp_path):
    import json

    (tmp_path / "evaluation").mkdir()
    write_json(tmp_path / "protocol.json", {"test_seeds": list(range(3000, 3100))})
    write_json(
        tmp_path / "training.json",
        {
            "smoke_test": False,
            "convergence_observed": True,
            "protocol_sha256": file_sha256(tmp_path / "protocol.json"),
        },
    )
    records = [{"seed": seed, "success": True} for seed in range(3000, 3100)]
    records[-1]["seed"] = 3000
    path = tmp_path / "evaluation/episodes.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    write_json(
        tmp_path / "evaluation/summary.json",
        {
            "status": "complete",
            "episodes": 100,
            "successes": 100,
            "episodes_sha256": file_sha256(path),
        },
    )
    with pytest.raises(ValueError, match="prescribed test seeds"):
        verify_completed(tmp_path)

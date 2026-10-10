import json
from pathlib import Path

import pytest

from owac.runtime.arena_droid_archive import completed_episode
from owac.runtime.arena_droid_verify import verify_phase, wilson


def episode_fixture(directory: Path, *, status="complete", success=True, seed=0, count=1):
    (directory / "evidence").mkdir()
    (directory / "official").mkdir()
    (directory / "evidence/summary.json").write_text(
        json.dumps({"status": status, "completed_episode_count": count})
    )
    (directory / "official/episode_results_rebuild0.jsonl").write_text(
        json.dumps({"success": success, "seed": seed, "episode_length": 200}) + "\n"
    )


def test_infrastructure_failure_does_not_receive_a_policy_score(tmp_path):
    episode_fixture(tmp_path, status="infrastructure_failed", success=False)
    assert completed_episode(tmp_path, 0)[0] is None


def test_missing_boolean_success_is_not_counted_as_failure(tmp_path):
    episode_fixture(tmp_path, success=None)
    with pytest.raises(ValueError, match="explicit official outcome"):
        completed_episode(tmp_path, 0)


def test_wrong_seed_cannot_replace_a_predeclared_trial(tmp_path):
    episode_fixture(tmp_path, seed=3)
    with pytest.raises(ValueError, match="seed or completion"):
        completed_episode(tmp_path, 0)


def test_duplicate_episode_is_not_silently_averaged(tmp_path):
    episode_fixture(tmp_path)
    path = tmp_path / "official/episode_results_rebuild0.jsonl"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="exactly one"):
        completed_episode(tmp_path, 0)


def test_finished_episode_with_frozen_camera_is_excluded(tmp_path):
    episode_fixture(tmp_path, success=False)
    (tmp_path / "evidence/input_integrity.json").write_text(
        json.dumps({"valid": False, "reason": "camera frozen"})
    )
    episode, summary = completed_episode(tmp_path, 0)
    assert episode is None
    assert summary["input_integrity"] is False


def test_wilson_does_not_treat_three_successes_as_certain():
    low, high = wilson(3, 3)
    assert low == pytest.approx(0.4385029682449546)
    assert high == pytest.approx(1)


def raw_fixture(run):
    seed = run / "seed-0"
    seed.mkdir()
    episode_fixture(seed)
    (run / "status.json").write_text(
        json.dumps({"status": "complete", "completed_seeds": [0], "wall_seconds": 1})
    )
    controls = []
    for step in range(2):
        frame = {
            "shape": [720, 1280],
            "dtype": "uint32",
            "minimum_dn": 0,
            "maximum_dn": 100,
            "unique_values": 100,
            "cfa_quantization_maximum_error_dn": 0,
            "native_frame_id": step,
            "files": {
                kind: {"bytes": 720 * 1280 * size, "sha256": str(step) * 64}
                for kind, size in (("0-texread", 8), ("2-cfa", 4), ("3-noise", 4))
            },
        }
        controls.append(
            {
                "step": step,
                "joint_position": [[0] * 7],
                "gripper_position": [[0]],
                "action": [[0] * 8],
                "rgb_sha256": {"rgb": str(step)},
                "raw": {
                    "step": step,
                    "phase": "pre_action",
                    "host_timestamp_ns": step + 1,
                    "physical_step_counter": step,
                    "physics_substep_counter": step * 8,
                    "views": {view: frame for view in ("external_camera", "wrist_camera")},
                },
            }
        )
    path = seed / "evidence/control.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in controls))
    summary = {
        "status": "complete",
        "completed_episode_count": 1,
        "control_steps": 2,
        "distinct_rgb_frames": {"rgb": 2},
        "inference_calls": 1,
        "wall_seconds": 1,
        "inference_seconds_p50": 0.1,
        "inference_seconds_p95": 0.1,
        "raw": {
            "control_frames": 2,
            "views": {view: {"frames": 2} for view in ("external_camera", "wrist_camera")},
        },
    }
    (seed / "evidence/summary.json").write_text(json.dumps(summary))
    (seed / "official/episode_results_rebuild0.jsonl").write_text(
        json.dumps({"success": True, "seed": 0, "episode_length": 2}) + "\n"
    )
    (seed / "policy_metrics.json").write_text(json.dumps({"peak_allocated_bytes": 1}))
    return path, controls


def test_complete_policy_does_not_hide_missing_raw_view(tmp_path):
    path, controls = raw_fixture(tmp_path)
    del controls[1]["raw"]["views"]["wrist_camera"]
    path.write_text("".join(json.dumps(record) + "\n" for record in controls))
    with pytest.raises(ValueError, match="Missing RAW view"):
        verify_phase(tmp_path, [0], raw=True)


def test_reused_native_frame_invalidates_continuous_capture(tmp_path):
    path, controls = raw_fixture(tmp_path)
    controls[1]["raw"]["views"]["wrist_camera"]["native_frame_id"] = 0
    path.write_text("".join(json.dumps(record) + "\n" for record in controls))
    with pytest.raises(ValueError, match="Repeated or old native frame"):
        verify_phase(tmp_path, [0], raw=True)

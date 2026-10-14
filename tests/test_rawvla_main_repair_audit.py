"""A recovered run must prove original identity, restored methods, and actual finite actions."""

import json

import pytest

from owac.runtime.rawvla_full_eval import digest, paired_seed, write_json
from owac.runtime.rawvla_main_repair_audit import verify_policy_trajectory


@pytest.fixture
def trajectory(tmp_path):
    entry = {
        "task_name": "grab_roller",
        "dataset_seed": 39,
        "lighting_domain": "Normal",
        "lighting_ev": 0.0,
        "lighting_scale": 1.0,
        "noise_seed": 123,
        "raw_white_level": 3.5,
        "sensor_saturation_ev": 0.0,
        "raw_sensor_shot_noise": 2.5e-5,
        "raw_sensor_read_noise": 3.90625e-8,
    }
    write_json(tmp_path / "lighting.json", entry)
    write_json(
        tmp_path / "expert-repair.json",
        {
            "entry": entry,
            "exception": None,
            "methods_restored_before_policy_setup": True,
            "episode_info": {"info": {"{A}": "roller/base2"}},
            "events": ["empty_grasp_plan"],
            "expert_plan_success": False,
        },
    )
    actions = tmp_path / "actions.jsonl"
    actions.write_text(json.dumps({"step": 0, "state": [0] * 14, "action": [0] * 14}) + "\n")
    record = {
        "case_id": "robotwin-0192",
        "entry": entry,
        "eval_seed": 39,
        "policy_flow_seed": paired_seed("robotwin", "grab_roller", 39),
        "action_source": "model_prediction",
        "image_input_key": "render_rgb",
        "directory": str(tmp_path),
        "num_steps": 1,
        "actions_sha256": digest(actions),
        "language": "grab roller",
        "success": False,
    }
    return record, entry


def test_accepts_complete_normal_failure_without_fabricating_expert_success(trajectory):
    result = verify_policy_trajectory(*trajectory)
    assert result["success"] is False and result["expert_plan_success"] is False
    assert result["num_steps"] == 1 and result["eval_seed"] == 39


@pytest.mark.parametrize(
    "field,value",
    [
        ("eval_seed", 40),
        ("policy_flow_seed", 123),
        ("action_source", "expert"),
        ("image_input_key", "raw"),
        ("num_steps", 0),
    ],
)
def test_rejects_recovery_that_changes_policy_protocol(trajectory, field, value):
    record, entry = trajectory
    record[field] = value
    with pytest.raises(ValueError):
        verify_policy_trajectory(record, entry)


def test_rejects_audit_with_policy_methods_still_patched(trajectory):
    from pathlib import Path

    record, entry = trajectory
    path = Path(record["directory"]) / "expert-repair.json"
    expert = json.loads(path.read_text())
    expert["methods_restored_before_policy_setup"] = False
    write_json(path, expert)
    with pytest.raises(ValueError, match="restoration failed"):
        verify_policy_trajectory(record, entry)

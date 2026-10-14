import json

import pytest

from owac.runtime.rawvla_complete_report import audit_robotwin_actions, grouped_metrics, metrics
from owac.runtime.rawvla_full_eval import digest


def episode(suite, task, domain, success):
    return {
        "rawvla_bench_entry": {
            "suite": suite,
            "task_id": task,
            "lighting_domain": domain,
        },
        "task_description": f"task {task}",
        "success": success,
        "num_steps": 100,
        "wall_seconds": 10,
    }


def test_failures_remain_in_success_denominator():
    result = metrics([episode("libero_10", 0, "Normal", value) for value in (True, False)])
    assert result["completed"] == 2
    assert result["success"] == result["policy_failure"] == 1
    assert result["success_rate_percent"] == 50
    assert result["actual_policy_steps"] == 200


def test_libero_avg_preserves_three_to_one_sog_long_weighting():
    records = [
        episode(suite, task, domain, suite != "libero_10")
        for domain in ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        for task in range(10)
    ]
    result = grouped_metrics(records, "libero")
    assert len(result["tasks"]) == 40
    for row in result["paper_layout"].values():
        assert row["SOG"]["success_rate_percent"] == 100
        assert row["Long"]["success_rate_percent"] == 0
        assert row["Avg"]["success_rate_percent"] == 75


def test_missing_domain_is_not_reported_as_zero_percent():
    result = grouped_metrics([episode("libero_10", 0, "Normal", False)], "libero")
    assert result["domains"]["Low"]["success_rate_percent"] is None
    assert result["domains"]["Normal"]["success_rate_percent"] == pytest.approx(0)


@pytest.mark.parametrize("corruption", [None, "truncated", "changed", "nonfinite"])
def test_full_action_audit_rejects_corrupt_or_incomplete_trajectory(tmp_path, corruption):
    path = tmp_path / "actions.jsonl"
    action = {"step": 0, "state": [0] * 14, "action": [0] * 14}
    if corruption == "nonfinite":
        action["action"][0] = float("inf")
    path.write_text(json.dumps(action) + "\n")
    record = {
        "case_id": "robotwin-0000",
        "eval_seed": 39,
        "entry": {"dataset_seed": 39},
        "action_source": "model_prediction",
        "image_input_key": "render_rgb",
        "directory": str(tmp_path),
        "actions_sha256": digest(path),
        "num_steps": 2 if corruption == "truncated" else 1,
    }
    if corruption == "changed":
        path.write_text("{}\n")
    if corruption:
        with pytest.raises(ValueError):
            audit_robotwin_actions([record])
    else:
        assert audit_robotwin_actions([record]) == {
            "episodes": 1,
            "actual_policy_steps": 1,
            "errors": [],
        }

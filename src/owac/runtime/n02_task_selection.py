"""Stdlib-only checkpoint gating, shared by training and the read-only monitor."""

import json
from pathlib import Path

REPAIR_PROTOCOL = "n02-task-repair-r11-v1"


def select_repaired_checkpoint(output):
    """Gate the executed prefix, joints and grippers before any physical validation."""
    import math

    output = Path(output)
    baseline = json.loads((output / "dev_baseline.json").read_text())
    config = json.loads((output / "run_config.json").read_text())
    if config.get("task_adaptation", {}).get("protocol") != REPAIR_PROTOCOL:
        raise ValueError("r11 selection requires the repaired protocol")
    limit = float(config["task_adaptation"].get("prefix_max_ratio", 1.0))
    if not 0 < limit <= 1:
        raise ValueError("prefix guard cannot allow regression")

    def guards(record):
        control = record["task_control"]
        opening = control["strata"]["Normal/opening"]
        proxy = record["executed_prefix_proxy"]
        values = {
            "normal_opening_prefix": opening["first10_mse"],
            "normal_opening_joints": opening["prefix_joint_mse"],
            "normal_opening_grippers": opening["prefix_gripper_mse"],
            "all_prefix_joints": control["prefix_joint_mse"],
            "all_prefix_grippers": control["prefix_gripper_mse"],
            "executed_prefix_proxy": proxy["balanced_prefix_mse"],
        }
        if config["execution"]["replan_interval"] == 50:
            values.update(
                normal_opening_full=opening["mse"],
                normal_opening_full_joints=opening["joint_mse"],
                normal_opening_full_grippers=opening["gripper_mse"],
                all_full_joints=control["joint_mse"],
                all_full_grippers=control["gripper_mse"],
                executed_chunk_proxy=proxy["balanced_mse"],
            )
        return values

    initial = guards(baseline)
    if not all(math.isfinite(value) and value >= 0 for value in initial.values()):
        raise ValueError("nonfinite or negative baseline guard")
    choices = []
    for path in sorted(output.glob("checkpoint_*.metrics.json")):
        row = json.loads(path.read_text())
        checkpoint = output / f"checkpoint_{row['step']:07d}.pt"
        if not checkpoint.is_file():
            continue
        eligible = (
            row.get("execution_contract") == baseline["execution_contract"] == config["execution"]
            and row.get("task_dev_manifest_sha256") == baseline["task_dev_manifest_sha256"]
            and bool(row.get("calibration_weights_sha256"))
        )
        if config.get("online_checks", {}).get("enabled"):
            checks = row.get("online_checks", {})
            eligible = (
                eligible
                and checks.get("expected_causal", 0) > 0
                and (checks.get("valid", 0) >= checks["expected_causal"])
            )
        values = guards(row)
        finite = all(math.isfinite(v) for v in values.values())
        primary = row["sampled_control_mse"]
        failures = [k for k, v in values.items() if not math.isfinite(v) or v > initial[k] * limit]
        eligible = (
            eligible
            and finite
            and math.isfinite(primary)
            and primary < baseline["sampled_control_mse"]
            and not failures
        )
        choices.append(
            {
                "step": row["step"],
                "checkpoint": str(checkpoint),
                "score": primary,
                "eligible": bool(eligible),
                "failed_guards": failures,
                "guards": values,
            }
        )
    candidates = [row for row in choices if row["eligible"]]
    return {
        "protocol": REPAIR_PROTOCOL,
        "scope": "offline_candidate_requires_physical_validation",
        "candidates": choices,
        "selected": min(candidates, key=lambda row: (row["score"], row["step"]))
        if candidates
        else None,
        "physical_success": None,
    }

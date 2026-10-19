"""Summarize completed physical trials without promoting simulator flags to success."""

import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def main():
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    evidence = checkout / "research/experiments/n02_v1_2/closed_loop_r11_20261019"
    plan = json.loads((evidence / "plan.json").read_text())
    cases = []
    for spec in plan["cases"]:
        case = checkout / "outputs/n02-v12/representation-r8-rollouts" / spec["case"]
        if not (case / "result.json").exists():
            cases.append({"case_id": spec["case"], "domain": spec["domain"], "state": "pending"})
            continue
        result = json.loads((case / "result.json").read_text())
        physical = result.get("physical") or {}
        valid = (
            result["returncode"] == 0
            and physical.get("diagnostic_only") is False
            and physical.get("action_source") == "live_raw_policy"
            and isinstance(physical.get("physical_accepted"), bool)
        )
        if not valid:
            cases.append(
                {
                    "case_id": spec["case"],
                    "domain": spec["domain"],
                    "state": "infrastructure_error",
                    "accepted_success": None,
                    "returncode": result["returncode"],
                    "result": result,
                }
            )
            continue
        controls = read_rows(case / "control.jsonl")
        inference = read_rows(case / "policy/inference.jsonl")
        joints = [i for i in range(14) if i not in (6, 13)]
        deltas = [
            max(abs(b["executed"][j] - a["executed"][j]) for j in joints) * 180 / math.pi
            for a, b in zip(controls, controls[1:], strict=False)
        ]
        boundaries = [
            v for row, v in zip(controls[1:], deltas, strict=True) if row["step"] % 50 == 0
        ]
        predicted = {
            r["step"] + offset: action
            for r in inference
            for offset, action in enumerate(r["actions"])
        }
        difference = max(
            (
                abs(row["executed"][i] - predicted[row["step"]][i])
                for row in controls
                for i in range(14)
            ),
            default=0.0,
        )
        audit = json.loads((case / "policy/load_audit.json").read_text())
        motion = {
            "scope": "executed joint target changes, not measured physical velocity",
            "commands": len(controls),
            "inferences": len(inference),
            "joint_step_max_deg": max(deltas) if deltas else None,
            "joint_step_p95_deg": percentile(deltas, 0.95) if deltas else None,
            "chunk_boundary_max_deg": max(boundaries) if boundaries else None,
            "max_executed_vs_policy_prediction": difference,
            "inference_seconds_p50": percentile([r["inference_seconds"] for r in inference], 0.5)
            if inference
            else None,
            "all_predictions_finite": bool(inference) and all(r["finite"] for r in inference),
            "checkpoint_matches_plan": audit["checkpoint_sha256"] == plan["checkpoint_sha256"],
            "optimizer_updates": audit["optimizer_updates"],
        }
        if motion["optimizer_updates"] or not motion["checkpoint_matches_plan"]:
            raise ValueError("unexpected training or checkpoint mismatch")
        (case / "motion.json").write_text(json.dumps(motion, indent=2) + "\n")
        cases.append(
            {
                "case_id": spec["case"],
                "domain": spec["domain"],
                "state": "completed" if valid else "infrastructure_error",
                "accepted_success": physical["physical_accepted"] if valid else None,
                "steps": physical.get("control_steps"),
                "hold_seconds": physical.get("max_hold_seconds"),
                "official_triggered": physical.get("first_official_trigger") is not None,
                "motion": motion,
                "wall_seconds": result["wall_seconds"],
                "video": str(case / "episode0.mp4"),
                "video_sha256": hashlib.sha256((case / "episode0.mp4").read_bytes()).hexdigest(),
            }
        )
    completed = [r for r in cases if r["state"] == "completed"]
    receipt = {
        "scope": plan["scope"],
        "planned": 3,
        "completed": len(completed),
        "success": sum(r["accepted_success"] for r in completed),
        "optimizer_updates": 0,
        "cases": cases,
    }
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()

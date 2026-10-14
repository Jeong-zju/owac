"""Validate a five-case expert repair before accepting a complete 632-case result."""

import argparse
import json
import math
import shutil
from pathlib import Path

from owac.runtime.rawvla_full_eval import digest, rows, write_json
from owac.runtime.rawvla_repair import PAPER_CASES


def validate(parent, output):
    protocol = json.loads((output / "repair-protocol.json").read_text())
    if digest(parent / "protocol.json") != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent frozen protocol changed")
    if digest(parent / "robotwin-paper-conditions.json") != protocol["selection_sha256"]:
        raise ValueError("Original metadata selection changed")
    if digest(parent / "robotwin-paper/episodes.jsonl") != protocol["inherited_ledger_sha256"]:
        raise ValueError("Parent completed ledger changed before repair acceptance")
    for name, expected in json.loads((output / "source-guards.json").read_text()).items():
        if digest(name) != expected:
            raise ValueError(f"Original simulator source changed: {name}")
    selection = json.loads((parent / "robotwin-paper-conditions.json").read_text())
    cases = {case["case_id"]: case for case in selection["entries"]}
    original = {row["case_id"]: row for row in rows(parent / "robotwin-paper/episodes.jsonl")}
    if len(cases) != 632 or len(original) != 627:
        raise ValueError("Repair must retain exactly 627 original results out of 632 conditions")
    ledger = rows(output / "robotwin-paper/episodes.jsonl")
    complete = {row["case_id"]: row for row in ledger}
    if len(complete) != len(ledger) or set(complete) != set(cases):
        raise ValueError("Repaired ledger is duplicate, incomplete, or contains unknown identities")
    if set(complete) - set(original) != set(PAPER_CASES):
        raise ValueError("Repair changed a case outside the five missing identities")
    for case_id, row in original.items():
        if complete[case_id] != row:
            raise ValueError(f"Inherited completed result changed: {case_id}")
    parity = json.loads((output / "control-input-parity.json").read_text())
    if (
        not parity["flow_seed_equal"]
        or parity["first_action_max_abs_difference"] != 0
        or set(parity["arrays"]) != {"state", "camera_0", "camera_1", "camera_2"}
        or not all(item["exact"] for item in parity["arrays"].values())
    ):
        raise ValueError("Known-good control input/action parity failed")
    recovered = []
    for case_id in PAPER_CASES:
        row, case = complete[case_id], cases[case_id]
        if row["entry"] != case["entry"] or row["eval_seed"] != case["entry"]["dataset_seed"]:
            raise ValueError(f"Repaired physical identity changed: {case_id}")
        if row["source_metadata_sha256"] != case["metadata_sha256"]:
            raise ValueError(f"Repaired original metadata identity changed: {case_id}")
        if row["action_source"] != "model_prediction" or row["image_input_key"] != "render_rgb":
            raise ValueError(f"Repaired policy/input source changed: {case_id}")
        path = Path(row["directory"])
        expert = json.loads((path / "expert-repair.json").read_text())
        if (
            expert["entry"] != case["entry"]
            or expert["exception"]
            or not expert["methods_restored_before_policy_setup"]
            or not expert["episode_info"]["info"]
        ):
            raise ValueError(f"Expert metadata or method restoration failed: {case_id}")
        actions = rows(path / "actions.jsonl")
        if digest(path / "actions.jsonl") != row["actions_sha256"]:
            raise ValueError(f"Actual policy action ledger changed: {case_id}")
        if row["num_steps"] <= 0 or len(actions) != row["num_steps"]:
            raise ValueError(f"No complete actual policy trajectory: {case_id}")
        for step, action in enumerate(actions):
            if (
                action["step"] != step
                or len(action["state"]) != 14
                or len(action["action"]) != 14
                or not all(math.isfinite(x) for x in action["state"] + action["action"])
            ):
                raise ValueError(f"Policy state/action boundary failed: {case_id}/{step}")
        lighting = json.loads((path / "lighting.json").read_text())
        for field in ("lighting_ev", "lighting_scale", "noise_seed", "raw_white_level"):
            if not math.isclose(lighting[field], case["entry"][field], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Repaired original lighting changed: {case_id}/{field}")
        recovered.append(
            {
                "case_id": case_id,
                "task": row["task"],
                "eval_seed": row["eval_seed"],
                "domain": case["entry"]["lighting_domain"],
                "language": row["language"],
                "success": row["success"],
                "num_steps": row["num_steps"],
                "expert_events": expert["events"],
                "expert_plan_success": expert["expert_plan_success"],
                "expert_audit_sha256": digest(path / "expert-repair.json"),
                "actions_sha256": row["actions_sha256"],
                "directory": str(path),
            }
        )
    return {
        "state": "verified",
        "parent": str(parent),
        "repair_output": str(output),
        "repair_protocol_sha256": digest(output / "repair-protocol.json"),
        "original_completed_unchanged": len(original),
        "repaired": recovered,
        "completed": len(ledger),
        "success": sum(row["success"] for row in ledger),
        "success_rate_percent": 100 * sum(row["success"] for row in ledger) / len(ledger),
        "policy_methods_unchanged": True,
        "original_source_hashes_unchanged": True,
        "control_input_and_first_action_exact": True,
    }


def adopt(parent, output):
    """Archive the old ledger, then append only independently verified recovery records."""
    result = validate(parent, output)
    write_json(output / "repair-validation.json", result)
    archive = output / "inherited-parent"
    archive.mkdir(exist_ok=False)
    (archive / "robotwin-paper").mkdir()
    for name in (
        "protocol.json",
        "robotwin-paper-conditions.json",
        "robotwin-paper/episodes.jsonl",
        "robotwin-paper/summary.json",
    ):
        shutil.copy2(parent / name, archive / name)
    ledger_path = parent / "robotwin-paper/episodes.jsonl"
    old_bytes = ledger_path.read_bytes()
    if digest(archive / "robotwin-paper/episodes.jsonl") != digest(ledger_path):
        raise ValueError("Parent ledger changed while archiving")
    validation_sha = digest(output / "repair-validation.json")
    repaired = {row["case_id"]: row for row in rows(output / "robotwin-paper/episodes.jsonl")}
    additions = []
    for case_id in PAPER_CASES:
        row = dict(repaired[case_id])
        row["recovery"] = {
            "repair_protocol_sha256": result["repair_protocol_sha256"],
            "repair_validation_sha256": validation_sha,
            "repair_output": str(output),
            "scope": "expert-check-only guards; original methods restored before policy setup",
        }
        additions.append(row)
    new_bytes = old_bytes + b"".join((json.dumps(row) + "\n").encode() for row in additions)
    if not old_bytes.endswith(b"\n"):
        raise ValueError("Parent ledger has an unfinished tail")
    temporary = ledger_path.with_suffix(".expert-repair-v1.tmp")
    temporary.write_bytes(new_bytes)
    temporary.replace(ledger_path)
    from owac.runtime.rawvla_paper_eval import summarize

    summary = summarize(parent, 632, finished=True)
    if summary["state"] != "complete" or summary["success"] != result["success"]:
        raise ValueError("Adopted recovery summary differs from independently audited result")
    adoption = {
        "state": "complete",
        "parent": str(parent),
        "repair_output": str(output),
        "archived_parent": str(archive),
        "old_ledger_sha256": digest(archive / "robotwin-paper/episodes.jsonl"),
        "new_ledger_sha256": digest(ledger_path),
        "added_case_ids": list(PAPER_CASES),
        "validation_sha256": validation_sha,
        "repair_protocol_sha256": result["repair_protocol_sha256"],
        "summary": summary,
        "original_protocol_unchanged": True,
        "original_627_results_unchanged": True,
    }
    write_json(output / "adoption.json", adoption)
    (parent / "recovery").mkdir(exist_ok=True)
    write_json(parent / "recovery/paper-expert-repair-v1.json", adoption)
    # Post-adoption validation uses the preserved original 627-result snapshot.
    validate(archive, output)
    return adoption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adopt", action="store_true")
    args = parser.parse_args()
    result = adopt(args.parent, args.output) if args.adopt else validate(args.parent, args.output)
    if not args.adopt:
        write_json(args.output / "repair-validation.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

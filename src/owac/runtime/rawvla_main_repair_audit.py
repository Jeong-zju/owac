"""Validate and adopt full-manifest recovery without changing inherited outcomes."""

import argparse
import json
import math
import shutil
from pathlib import Path

from owac.runtime.rawvla_eval_audit import audit
from owac.runtime.rawvla_full_eval import digest, paired_seed, rows, summary, write_json


def verify_policy_trajectory(record, entry):
    case_id = record["case_id"]
    if record["entry"] != entry or record["eval_seed"] != entry["dataset_seed"]:
        raise ValueError(f"Recovered physical identity changed: {case_id}")
    seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
    if record["policy_flow_seed"] != seed:
        raise ValueError(f"Recovered flow seed changed: {case_id}")
    if record["action_source"] != "model_prediction" or record["image_input_key"] != "render_rgb":
        raise ValueError(f"Recovered policy/input source changed: {case_id}")
    directory = Path(record["directory"])
    expert = json.loads((directory / "expert-repair.json").read_text())
    if (
        expert["entry"] != entry
        or expert["exception"]
        or not expert["methods_restored_before_policy_setup"]
        or not expert["episode_info"]["info"]
    ):
        raise ValueError(f"Expert metadata or restoration failed: {case_id}")
    path = directory / "actions.jsonl"
    if digest(path) != record["actions_sha256"]:
        raise ValueError(f"Recovered actual actions changed: {case_id}")
    actions = rows(path)
    if len(actions) != record["num_steps"] or record["num_steps"] <= 0:
        raise ValueError(f"Missing actual policy steps: {case_id}")
    for index, action in enumerate(actions):
        if (
            action["step"] != index
            or len(action["state"]) != 14
            or len(action["action"]) != 14
            or not all(math.isfinite(x) for x in action["state"] + action["action"])
        ):
            raise ValueError(f"Invalid actual policy state/action: {case_id}/{index}")
    lighting = json.loads((directory / "lighting.json").read_text())
    for field in (
        "lighting_ev",
        "lighting_scale",
        "noise_seed",
        "raw_white_level",
        "sensor_saturation_ev",
        "raw_sensor_shot_noise",
        "raw_sensor_read_noise",
    ):
        if not math.isclose(lighting[field], entry[field], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Recovered lighting changed: {case_id}/{field}")
    return {
        "case_id": case_id,
        "task": entry["task_name"],
        "eval_seed": record["eval_seed"],
        "domain": entry["lighting_domain"],
        "success": record["success"],
        "num_steps": record["num_steps"],
        "language": record["language"],
        "expert_events": expert["events"],
        "expert_plan_success": expert["expert_plan_success"],
        "expert_audit_sha256": digest(directory / "expert-repair.json"),
        "actions_sha256": record["actions_sha256"],
        "directory": str(directory),
    }


def validate(parent, output):
    protocol = json.loads((output / "repair-protocol.json").read_text())
    for file, key in (
        ("protocol.json", "parent_protocol_sha256"),
        ("robotwin-manifest.json", "manifest_sha256"),
        ("robotwin/episodes.jsonl", "inherited_ledger_sha256"),
    ):
        if digest(parent / file) != protocol[key]:
            raise ValueError(f"Original parent evidence changed: {file}")
    for file, expected in json.loads((output / "source-guards.json").read_text()).items():
        if digest(file) != expected:
            raise ValueError(f"Original simulator source changed: {file}")
    parity = json.loads((output / "control-input-parity.json").read_text())
    if (
        not parity["flow_seed_equal"]
        or parity["first_action_max_abs_difference"] != 0
        or set(parity["arrays"]) != {"state", "camera_0", "camera_1", "camera_2"}
        or not all(item["exact"] for item in parity["arrays"].values())
    ):
        raise ValueError("Validated expert-guard control parity failed")
    inherited = rows(parent / "robotwin/episodes.jsonl")
    wanted = {case["case_id"]: case["entry"] for case in protocol["cases"]}
    manifest = json.loads((parent / "robotwin-manifest.json").read_text())
    expected = {f"robotwin-{index:04d}": entry for index, entry in enumerate(manifest["entries"])}
    missing = set(expected) - {row["case_id"] for row in inherited}
    if set(wanted) != missing or any(wanted[key] != expected[key] for key in wanted):
        raise ValueError("Recovery selection differs from original missing manifest identities")
    repaired = [
        record
        for index in range(protocol["workers"])
        for record in rows(output / "workers" / str(index) / "episodes.jsonl")
    ]
    lookup = {row["case_id"]: row for row in repaired}
    if len(inherited) != 3217 or len(wanted) != 33:
        raise ValueError("Original full-manifest recovery denominator changed")
    if len(lookup) != len(repaired) or set(lookup) != set(wanted):
        raise ValueError("Recovered cases are duplicate, incomplete, or unexpected")
    recovered = [verify_policy_trajectory(lookup[key], wanted[key]) for key in sorted(wanted)]
    candidate = inherited + [lookup[key] for key in sorted(wanted)]
    write_json(output / "recovered.json", recovered)
    ledger = output / "robotwin/episodes.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in candidate))
    report = audit(output)
    if report["state"] != "complete" or report["errors"] or report["completed"] != 13250:
        raise ValueError(f"Full identity/paired-language audit failed: {report['errors']}")
    write_json(output / "candidate-audit.json", report)
    return {
        "state": "verified",
        "inherited_completed_unchanged": 3217,
        "repaired": recovered,
        "robotwin_completed": 3250,
        "main_completed": 13250,
        "robotwin_success": report["platforms"]["robotwin"]["success"],
        "repair_protocol_sha256": digest(output / "repair-protocol.json"),
        "candidate_ledger_sha256": digest(ledger),
        "candidate_audit_sha256": digest(output / "candidate-audit.json"),
        "paired_languages_and_original_manifest_verified": True,
        "policy_methods_restored_before_setup": True,
        "original_simulator_source_unchanged": True,
    }


def adopt(parent, output):
    result = validate(parent, output)
    write_json(output / "repair-validation.json", result)
    archive = output / "inherited-parent"
    archive.mkdir(exist_ok=False)
    (archive / "robotwin").mkdir()
    for name in (
        "protocol.json",
        "libero-manifest.json",
        "robotwin-manifest.json",
        "robotwin/episodes.jsonl",
        "robotwin/attempts.jsonl",
        "audit-status.json",
        "robotwin-status.json",
    ):
        shutil.copy2(parent / name, archive / name)
    path = parent / "robotwin/episodes.jsonl"
    old_bytes = path.read_bytes()
    if digest(path) != digest(archive / "robotwin/episodes.jsonl"):
        raise ValueError("Original ledger changed during adoption")
    candidate = {row["case_id"]: row for row in rows(output / "robotwin/episodes.jsonl")}
    additional = []
    validation_sha = digest(output / "repair-validation.json")
    for row in result["repaired"]:
        entry = dict(candidate[row["case_id"]])
        entry["recovery"] = {
            "repair_output": str(output),
            "repair_protocol_sha256": result["repair_protocol_sha256"],
            "validation_sha256": validation_sha,
            "scope": "Expert-only guards; original policy-time methods restored",
        }
        additional.append(entry)
    if not old_bytes.endswith(b"\n"):
        raise ValueError("Original ledger has an unfinished tail")
    temporary = path.with_suffix(".main-repair-v1.tmp")
    temporary.write_bytes(
        old_bytes + b"".join((json.dumps(row) + "\n").encode() for row in additional)
    )
    temporary.replace(path)
    summary(parent, "robotwin", 3250, "complete")
    report = audit(parent)
    if report["state"] != "complete" or report["errors"]:
        raise ValueError("Adopted main ledger failed independent audit")
    write_json(parent / "audit-status.json", report)
    result = {
        "state": "complete",
        "parent": str(parent),
        "repair_output": str(output),
        "archived_parent": str(archive),
        "old_ledger_sha256": digest(archive / "robotwin/episodes.jsonl"),
        "new_ledger_sha256": digest(path),
        "validation_sha256": validation_sha,
        "added_case_ids": [row["case_id"] for row in additional],
        "main_completed": report["completed"],
        "platform_scores": {
            name: values["final_success_rate_percent"]
            for name, values in report["platforms"].items()
        },
        "original_protocol_unchanged": True,
        "original_3217_records_unchanged": True,
    }
    write_json(output / "adoption.json", result)
    write_json(parent / "recovery/main-expert-repair-v1.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adopt", action="store_true")
    args = parser.parse_args()
    result = adopt(args.parent, args.output) if args.adopt else validate(args.parent, args.output)
    if not args.adopt:
        write_json(args.output / "repair-validation.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

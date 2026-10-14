"""Accept recovered identities only with complete physical/action/source evidence."""

import json

import pytest

from owac.runtime.rawvla_full_eval import digest, write_json
from owac.runtime.rawvla_repair import PAPER_CASES
from owac.runtime.rawvla_repair_audit import adopt, validate


@pytest.fixture
def evidence(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "repair"
    for folder in (parent, output):
        (folder / "robotwin-paper").mkdir(parents=True)
    write_json(parent / "protocol.json", {"frozen": True})
    original, recovered, selection = [], [], []
    for index, case_id in enumerate([*(f"existing-{i}" for i in range(627)), *PAPER_CASES]):
        entry = {
            "dataset_seed": index,
            "lighting_domain": "Normal",
            "lighting_ev": 0.0,
            "lighting_scale": 1.0,
            "noise_seed": index,
            "raw_white_level": 3.5,
        }
        metadata = f"metadata-{index}"
        selection.append({"case_id": case_id, "entry": entry, "metadata_sha256": metadata})
        if index < 627:
            original.append({"case_id": case_id, "entry": entry, "success": False})
            continue
        folder = output / "robotwin-paper" / case_id / "attempt1"
        folder.mkdir(parents=True)
        actions = folder / "actions.jsonl"
        actions.write_text(json.dumps({"step": 0, "state": [0] * 14, "action": [0] * 14}) + "\n")
        write_json(folder / "lighting.json", entry)
        write_json(
            folder / "expert-repair.json",
            {
                "entry": entry,
                "exception": None,
                "methods_restored_before_policy_setup": True,
                "episode_info": {"info": {"{A}": "object"}},
                "events": ["empty_grasp_plan"],
                "expert_plan_success": False,
            },
        )
        recovered.append(
            {
                "case_id": case_id,
                "entry": entry,
                "eval_seed": index,
                "source_metadata_sha256": metadata,
                "action_source": "model_prediction",
                "image_input_key": "render_rgb",
                "directory": str(folder),
                "actions_sha256": digest(actions),
                "num_steps": 1,
                "task": "grab_roller",
                "language": "grab",
                "success": False,
            }
        )
    write_json(parent / "robotwin-paper-conditions.json", {"entries": selection})
    (parent / "robotwin-paper/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in original)
    )
    write_json(parent / "robotwin-paper/summary.json", {"state": "incomplete", "completed": 627})
    (output / "robotwin-paper/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in original + recovered)
    )
    write_json(
        output / "repair-protocol.json",
        {
            "parent_protocol_sha256": digest(parent / "protocol.json"),
            "selection_sha256": digest(parent / "robotwin-paper-conditions.json"),
            "inherited_ledger_sha256": digest(parent / "robotwin-paper/episodes.jsonl"),
        },
    )
    source = tmp_path / "original_simulator.py"
    source.write_text("original source")
    write_json(output / "source-guards.json", {str(source): digest(source)})
    write_json(
        output / "control-input-parity.json",
        {
            "flow_seed_equal": True,
            "first_action_max_abs_difference": 0,
            "arrays": {
                name: {"exact": True} for name in ["state", "camera_0", "camera_1", "camera_2"]
            },
        },
    )
    return parent, output


def test_complete_recovery_accepts_normal_failures_with_original_denominator(evidence):
    result = validate(*evidence)
    assert result["state"] == "verified"
    assert result["completed"] == 632 and result["original_completed_unchanged"] == 627
    assert result["success"] == 0 and result["success_rate_percent"] == 0
    assert {row["case_id"] for row in result["repaired"]} == set(PAPER_CASES)


@pytest.mark.parametrize("mutation", ["duplicate", "replace_inherited", "replace_seed"])
def test_recovery_rejects_identity_or_existing_result_changes(evidence, mutation):
    parent, output = evidence
    path = output / "robotwin-paper/episodes.jsonl"
    ledger = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "duplicate":
        ledger.append(ledger[-1])
    elif mutation == "replace_inherited":
        ledger[0]["success"] = True
    else:
        ledger[-1]["eval_seed"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
    with pytest.raises(ValueError):
        validate(parent, output)


def test_recovery_rejects_methods_left_patched_for_policy(evidence):
    parent, output = evidence
    path = output / "robotwin-paper" / PAPER_CASES[0] / "attempt1/expert-repair.json"
    expert = json.loads(path.read_text())
    expert["methods_restored_before_policy_setup"] = False
    write_json(path, expert)
    with pytest.raises(ValueError, match="restoration failed"):
        validate(parent, output)


def test_recovery_rejects_changed_original_simulator_source(evidence):
    parent, output = evidence
    source = next(iter(json.loads((output / "source-guards.json").read_text())))
    from pathlib import Path

    Path(source).write_text("changed simulator")
    with pytest.raises(ValueError, match="Original simulator source changed"):
        validate(parent, output)


def test_adoption_archives_original_bytes_and_appends_only_missing_cases(evidence):
    parent, output = evidence
    ledger = parent / "robotwin-paper/episodes.jsonl"
    original_bytes = ledger.read_bytes()
    protocol_sha = digest(parent / "protocol.json")
    result = adopt(parent, output)
    assert result["state"] == "complete"
    assert ledger.read_bytes().startswith(original_bytes)
    assert (
        output / "inherited-parent/robotwin-paper/episodes.jsonl"
    ).read_bytes() == original_bytes
    assert digest(parent / "protocol.json") == protocol_sha
    assert json.loads((parent / "robotwin-paper/summary.json").read_text())["completed"] == 632
    assert validate(output / "inherited-parent", output)["completed"] == 632


def test_adoption_refuses_incomplete_recovery_without_changing_originals(evidence):
    parent, output = evidence
    ledger = parent / "robotwin-paper/episodes.jsonl"
    original_bytes = ledger.read_bytes()
    repair_ledger = output / "robotwin-paper/episodes.jsonl"
    repair_ledger.write_text("\n".join(repair_ledger.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        adopt(parent, output)
    assert ledger.read_bytes() == original_bytes
    assert not (output / "inherited-parent").exists()

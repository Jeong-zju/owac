"""CPU-only supervisor/schema tests; no model or optimizer is constructed."""

from __future__ import annotations

import json
from pathlib import Path

from owac.runtime.raw_temporal_pilot import (
    ACCEPTANCE_SCHEMA,
    PilotSupervisor,
    _freeze_ok,
    _sha256,
    acceptance_example,
)


def test_supervisor_plan_is_sequential_and_acceptance_example_is_disabled(tmp_path):
    supervisor = PilotSupervisor(tmp_path, python=Path("/canonical/python"))
    plan = supervisor.plan()
    assert [item["phase"] for item in plan] == [
        "b_first1",
        "wait_for_b_acceptance",
        "b_resume50",
        "c_first1",
        "wait_for_c_acceptance",
        "c_resume50",
    ]
    command = plan[0]["command"]
    assert command[command.index("--arm") + 1] == "B"
    assert command[command.index("--max-updates") + 1] == "1"
    example = acceptance_example()
    assert example["schema"] == ACCEPTANCE_SCHEMA
    assert example["authorized"] is False
    assert example["accepted"] is False


def test_supervisor_requires_start_and_source_freeze(tmp_path):
    supervisor = PilotSupervisor(tmp_path, python=Path("/canonical/python"))
    result = supervisor.run()
    assert result["status"] == "waiting_for_sol_start"
    supervisor._lock_handle.close()
    (tmp_path / "START_ALLOWED").write_text("allowed")
    supervisor = PilotSupervisor(tmp_path, python=Path("/canonical/python"))
    result = supervisor.run()
    assert result["status"] == "waiting_for_source_freeze"


def test_supervisor_acceptance_binds_actual_checkpoint_sha(tmp_path):
    supervisor = PilotSupervisor(tmp_path)
    checkpoint = tmp_path / "checkpoint_final_000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "schema": ACCEPTANCE_SCHEMA,
        "authorized": True,
        "accepted": True,
        "arm": "B",
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (tmp_path / "acceptance-B.json").write_text(json.dumps(payload))
    assert supervisor._acceptance("B", checkpoint) is True
    payload["checkpoint_sha256"] = "0" * 64
    (tmp_path / "acceptance-B.json").write_text(json.dumps(payload))
    assert supervisor._acceptance("B", checkpoint) is False


def test_source_freeze_detects_file_change(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/owac/runtime/raw_temporal_pilot.py"
    freeze = tmp_path / "source-freeze.json"
    freeze.write_text(
        json.dumps({"files": {str(source.relative_to(source.parents[3])): _sha256(source)}})
    )
    assert _freeze_ok(tmp_path)
    freeze.write_text(json.dumps({"files": {str(source.relative_to(source.parents[3])): "0" * 64}}))
    assert not _freeze_ok(tmp_path)


def test_supervisor_fake_runner_first_step_state(tmp_path):
    (tmp_path / "user-authorization.json").write_text(
        json.dumps({"combined_wall_cap_seconds": 14400, "new_output_cap_bytes": 20 * 1024**3})
    )
    (tmp_path / "authority-B.json").write_text(
        json.dumps(
            {
                "authorized": True,
                "max_wall_seconds": 5400,
                "max_output_bytes": 10 * 1024**3,
                "receipt_sha256": _sha256(tmp_path / "user-authorization.json"),
                "max_updates": 50,
            }
        )
    )
    calls = []

    def runner(command):
        calls.append(command)
        target = tmp_path / "pilot-B" / "checkpoint_final_000001.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake checkpoint")
        return 0

    supervisor = PilotSupervisor(tmp_path, runner=runner)
    result = supervisor.run()
    assert result["status"] == "waiting_for_sol_start"
    (tmp_path / "START_ALLOWED").write_text("allowed")
    supervisor = PilotSupervisor(tmp_path, runner=runner)
    result = supervisor.run()
    # Freeze is absent, so still no child launch.
    assert result["status"] == "waiting_for_source_freeze"
    assert calls == []


def test_failed_phase_is_terminal_and_not_relaunched(tmp_path):
    supervisor = PilotSupervisor(tmp_path, python=Path("/canonical/python"))
    supervisor.state.phase = "b_failed"
    calls = []
    supervisor._runner = lambda command: calls.append(command) or 0
    result = supervisor.run()
    assert result["status"] == "failed"
    assert calls == []


def test_inflight_prepared_job_blocks_restart(tmp_path):
    supervisor = PilotSupervisor(tmp_path, python=Path("/canonical/python"))
    supervisor.state.active_job = {"status": "prepared", "phase": "b_first1"}
    calls = []
    supervisor._runner = lambda command: calls.append(command) or 0
    result = supervisor.run()
    assert result["status"] == "attention_active_job"
    assert calls == []


def test_global_disk_cap_blocks_before_child(tmp_path, monkeypatch):
    import json
    import time

    import owac.runtime.raw_temporal_pilot as pilot

    (tmp_path / "user-authorization.json").write_text(
        json.dumps({"combined_wall_cap_seconds": 14400, "new_output_cap_bytes": 20 * 1024**3})
    )
    (tmp_path / "authority-B.json").write_text(
        json.dumps(
            {
                "authorized": True,
                "max_wall_seconds": 5400,
                "max_output_bytes": 10 * 1024**3,
                "receipt_sha256": _sha256(tmp_path / "user-authorization.json"),
                "max_updates": 50,
            }
        )
    )
    supervisor = PilotSupervisor(tmp_path, python=Path("/canonical/python"))
    supervisor.state.started_at = time.time()
    monkeypatch.setattr(pilot, "_runroot_bytes", lambda root: 19 * 1024**3)
    result = supervisor._run_phase("B", 1, "b_first1", 1000)
    assert result["status"] == "disk_cap_waiting"


def test_combined_and_arm_deadlines_block(tmp_path):
    import json
    import time

    (tmp_path / "user-authorization.json").write_text(
        json.dumps({"combined_wall_cap_seconds": 14400, "new_output_cap_bytes": 20 * 1024**3})
    )
    (tmp_path / "authority-B.json").write_text(
        json.dumps(
            {
                "authorized": True,
                "max_wall_seconds": 5400,
                "max_output_bytes": 10 * 1024**3,
                "receipt_sha256": _sha256(tmp_path / "user-authorization.json"),
                "max_updates": 50,
            }
        )
    )
    supervisor = PilotSupervisor(tmp_path, python=Path("/canonical/python"))
    supervisor.state.started_at = time.time() - 15000
    result = supervisor._run_phase("B", 1, "b_first1", 100)
    assert result["status"] == "combined_wall_exhausted"
    supervisor.state.started_at = time.time()
    supervisor.state.arm_wall_seconds["B"] = 6000
    result = supervisor._run_phase("B", 1, "b_first1", 1000)
    assert result["status"] == "arm_wall_exhausted"

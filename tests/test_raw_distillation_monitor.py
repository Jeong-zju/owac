"""CPU tests for the read-only C-full distillation monitor.

These tests exercise real process liveness (spawned Python processes), torn JSONL
tails, PID reuse, stale heartbeats, honest preparing state, HTTP endpoint
restrictions, path/credential redaction, and stdlib-only importability.  Nothing
here needs a GPU, network access, or a real training run.
"""

import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from owac.runtime import raw_distillation_monitor as monitor
from owac.runtime.raw_temporal_c_supervisor import Supervisor

NOW = datetime.now().astimezone()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iso(moment=NOW):
    return moment.isoformat(timespec="seconds")


def spawn_sleeper():
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_exit(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - safety net
        proc.kill()
        proc.wait(timeout=5)


def start_ticks(pid):
    return monitor.read_proc_start_ticks(pid)


def update_record(step, **overrides):
    new_stage = max(0, step - 50)
    record = {
        "schema": "owac.raw-stem-temporal.c-full.update.v1",
        "event": "update",
        "timestamp": iso(),
        "cumulative_updates": step,
        "new_stage_updates": new_stage,
        "parent_cumulative_updates": 50,
        "anchor_presentations": 800 + new_stage * 16,
        "new_stage_anchor_presentations": new_stage * 16,
        "sampled_source_endpoints": [{"source_id": "source-a", "endpoint": 25}],
        "feature_mse": 2.5 - new_stage * 0.1,
        "feature_cosine": 0.05,
        "feature_total": 2.5 - new_stage * 0.1 + 0.005,
        "loss_total": 2.5 - new_stage * 0.1 + 0.005,
        "lr_used": 5e-05,
        "lr_next": 5e-05,
        "trainable_parameter_count": 14454784,
        "trainable_tensor_count": 5,
        "finite_gradients": True,
        "all_gradients_nonzero": True,
        "gradient_summary": {"secret_gradient": {"norm": 1.0}},
        "elapsed_seconds": 40.0 + new_stage,
        "new_stage_wall_seconds": 40.0 + new_stage,
        "gpu_allocated_bytes": 10765686784 + new_stage,
        "gpu_reserved_bytes": 11117002752 + new_stage,
        "host_rss_bytes": 12537995264,
        "output_bytes": 1234567 + new_stage,
        "checkpoint": None,
    }
    record.update(overrides)
    return record


def validation_record(step, **overrides):
    record = {
        "schema": "owac.raw-stem-temporal.c-full.validation.v1",
        "timestamp": iso(),
        "step": step,
        "new_stage_step": max(0, step - 50),
        "split": "validation",
        "sources": 5,
        "trajectories": 5,
        "windows": 102,
        "anchors": 102,
        "episode_resets": 5,
        "valid_tokens": 8192,
        "scalar_denominator": 16777216,
        "feature_mse": 1.0,
        "feature_cosine": 0.1,
        "feature_total": 1.01,
        "cosine_weight": 0.1,
        "per_source": {"source-a": {"feature_total": 1.01}},
        "selection_metric": "validation_token_total",
        "selection_eligible": True,
        "is_best": False,
        "checkpoint_path": f"checkpoints/checkpoint_{step:06d}.pt",
        "checkpoint_sha256": (f"{step:064x}")[-64:],
        "test_used": False,
        "action_diagnostic": None,
    }
    record.update(overrides)
    return record


def parent_record(step, **overrides):
    record = {
        "schema": "owac.raw-stem-temporal.c-full.parent-history.v1",
        "event": "parent_update",
        "cumulative_updates": step,
        "anchor_presentations": step * 16,
        "feature_mse": 3.6 - step * 0.01,
        "feature_cosine": 0.05,
        "feature_total": 3.65 - step * 0.01,
        "learning_rate": 2e-06,
        "timestamp": iso(),
        "gpu_allocated_bytes": 10765686784,
        "gpu_reserved_bytes": 11117002752,
    }
    record.update(overrides)
    return record


def run_status(pid, ticks, pgid=None, **overrides):
    record = {
        "schema": "owac.raw-stem-temporal.c-full.run-status.v1",
        "status": "running",
        "phase": "running",
        "pid": pid,
        "pid_start_ticks": ticks,
        "pgid": pgid if pgid is not None else pid,
        "native_exit": None,
        "started_at": iso(),
        "updated_at": iso(),
        "finished_at": None,
        "cumulative_updates": 51,
        "new_stage_updates": 1,
        "target_cumulative_updates": 5000,
        "stop_after_update": 51,
        "anchor_presentations": 816,
        "new_stage_anchor_presentations": 16,
        "new_stage_wall_seconds": 40.0,
        "new_stage_wall_cap_seconds": 230400,
        "output_bytes": 1234,
        "output_cap_bytes": 21474836480,
        "last_checkpoint": {
            "step": 51,
            "path": "checkpoints/latest.pt",
            "sha256": "a" * 64,
            "kind": "interval",
        },
        "last_validation": None,
        "best": None,
        "selection_metric": "validation_token_total",
        "error": None,
        "closed_loop": "not_run",
        "closed_loop_episodes": 0,
        "control_restored": False,
        "recipe_sha256": "b" * 64,
        "parent_checkpoint_sha256": "c" * 64,
        "migration_sha256": "d" * 64,
        "authority_sha256": "e" * 64,
    }
    record.update(overrides)
    return record


def heartbeat(pid, ticks, **overrides):
    record = {
        "schema": "owac.raw-stem-temporal.c-full.heartbeat.v1",
        "pid": pid,
        "pid_start_ticks": ticks,
        "pgid": pid,
        "seq": 17,
        "updated_at": iso(),
        "phase": "running",
        "cumulative_updates": 51,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Honest preparing state and JSONL tolerance.
# ---------------------------------------------------------------------------


def test_preparing_before_training_is_honest_and_null_not_zero(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "harness").mkdir()
    write_json(root / "run-meta.json", {"schema": "fixture", "parent_run": "pilot-C"})
    write_json(root / "prepared.json", {"schema": "fixture", "updated_at": iso()})
    config = tmp_path / "raw_stem_siglip_c_full_v1.json"
    write_json(config, {"schema": "fixture-config"})

    snapshot = monitor.build_snapshot(root, config)

    assert snapshot["status"] == "preparing"
    assert snapshot["status_declared"] == "preparing"
    assert snapshot["stale"] is False
    assert snapshot["process"]["pid"] is None
    assert snapshot["process"]["alive"] is None
    assert snapshot["process"]["state"] == "unknown"
    assert snapshot["heartbeat"] is None
    assert snapshot["train_series"] == []
    assert snapshot["validation_series"] == []
    assert snapshot["parent_history"]["available"] is False
    assert snapshot["parent_history"]["points"] == []
    assert snapshot["supervisor"] is None
    assert snapshot["acceptance"]["present"] is False
    assert snapshot["last_checkpoint"] is None
    assert snapshot["error"] is None
    assert snapshot["action_diagnostic"] is None
    assert snapshot["closed_loop"] == {
        "status": "not_run",
        "episodes": 0,
        "control_restored": False,
        "note": "本阶段未授权闭环；不代表控制已恢复",
    }
    for key in (
        "cumulative_updates",
        "new_stage_updates",
        "parent_cumulative_updates",
        "target_cumulative_updates",
        "anchor_presentations",
        "new_stage_wall_seconds",
        "output_bytes",
        "gpu",
    ):
        assert snapshot["progress"][key] is None, key
    assert snapshot["selection"]["available"] is False
    assert snapshot["selection"]["best_step"] is None
    assert snapshot["config"]["present"] is True
    assert snapshot["config"]["sha256"] is not None
    assert "raw_stem_siglip_c_full_v1.json" == snapshot["config"]["filename"]
    json.dumps(snapshot, ensure_ascii=False, allow_nan=False)


def test_jsonl_torn_tail_and_invalid_lines_are_handled(tmp_path):
    path = tmp_path / "updates.jsonl"
    warnings = []
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    parsed = monitor.read_jsonl(path, warnings)
    assert parsed["present"] is True
    assert [row["a"] for row in parsed["rows"]] == [1, 2]
    assert parsed["torn_tail"] is False

    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"a": 3, "torn": ')
    parsed = monitor.read_jsonl(path, warnings)
    assert [row["a"] for row in parsed["rows"]] == [1, 2]
    assert parsed["torn_tail"] is True

    # Even a syntactically complete last line is ignored without a trailing newline.
    path.write_text('{"a": 1}\n{"a": 2}', encoding="utf-8")
    parsed = monitor.read_jsonl(path, warnings)
    assert [row["a"] for row in parsed["rows"]] == [1]
    assert parsed["torn_tail"] is True

    path.write_text('{"a": 1}\nnot-json\n{"a": 3}\n', encoding="utf-8")
    parsed = monitor.read_jsonl(path, warnings)
    assert [row["a"] for row in parsed["rows"]] == [1, 3]
    assert parsed["invalid_lines"] == 1
    assert monitor.read_jsonl(tmp_path / "missing.jsonl", warnings)["rows"] == []
    assert warnings == []


def test_new_stage_fraction_uses_new_stage_counters_and_unknown_parent_is_null(tmp_path):
    known = tmp_path / "known-parent"
    known.mkdir()
    write_json(
        known / "run-status.json",
        {
            "status": "preparing",
            "phase": "preparing",
            "updated_at": iso(),
            "cumulative_updates": 50,
            "parent_cumulative_updates": 50,
            "new_stage_updates": 0,
            "target_cumulative_updates": 5000,
            "new_stage_wall_seconds": 0.0,
            "new_stage_wall_cap_seconds": 230400,
            "output_bytes": 0,
            "output_cap_bytes": 21474836480,
        },
    )
    known_snapshot = monitor.build_snapshot(known)
    progress = known_snapshot["progress"]
    assert progress["parent_cumulative_updates"] == 50
    assert progress["new_stage_updates"] == 0
    assert progress["new_stage_target_updates"] == 4950
    assert progress["new_stage_fraction"] == 0.0
    assert progress["progress_fraction"] == 0.01
    assert progress["parent_cumulative_updates"] != progress["new_stage_target_updates"]

    unknown = tmp_path / "unknown-parent"
    unknown.mkdir()
    write_json(
        unknown / "run-status.json",
        {
            "status": "preparing",
            "phase": "preparing",
            "updated_at": iso(),
            "cumulative_updates": 10,
            "new_stage_updates": 0,
            "target_cumulative_updates": 5000,
        },
    )
    unknown_snapshot = monitor.build_snapshot(unknown)
    progress = unknown_snapshot["progress"]
    assert progress["parent_cumulative_updates"] is None
    assert progress["new_stage_target_updates"] is None
    assert progress["new_stage_fraction"] is None
    assert progress["progress_fraction"] == 10 / 5000


def test_zombie_process_is_dead_even_when_starttime_matches(tmp_path):
    if not hasattr(os, "fork"):
        pytest.skip("fork is not available on this platform")
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child exits immediately without reaping
        os._exit(0)
    try:
        state_character, ticks = None, None
        for _ in range(200):
            state_character, ticks = monitor.read_proc_stat(pid)
            if state_character == "Z":
                break
            time.sleep(0.01)
        assert state_character == "Z"
        root = tmp_path / "zombie"
        root.mkdir()
        write_json(root / "run-status.json", run_status(pid, ticks))
        write_json(root / "heartbeat.json", heartbeat(pid, ticks))
        snapshot = monitor.build_snapshot(root)
        assert snapshot["process"]["alive"] is False
        assert snapshot["process"]["state"] == "dead"
        assert snapshot["process"]["zombie"] is True
        assert snapshot["process"]["observed_start_ticks"] == ticks
        assert snapshot["process"]["observed_process_state"] == "Z"
        assert snapshot["status"] == "stale"
        assert snapshot["stale_reason"] == "process_dead"
    finally:
        os.waitpid(pid, 0)


# ---------------------------------------------------------------------------
# Real process liveness.
# ---------------------------------------------------------------------------


def test_real_alive_process_full_snapshot_and_dead_after_exit(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    config = tmp_path / "config.json"
    write_json(config, {"schema": "fixture-config", "run_root": str(root)})
    proc = spawn_sleeper()
    try:
        ticks = start_ticks(proc.pid)
        write_json(root / "run-status.json", run_status(proc.pid, ticks))
        write_json(root / "heartbeat.json", heartbeat(proc.pid, ticks))
        append_jsonl(root / "updates.jsonl", update_record(51))
        append_jsonl(root / "updates.jsonl", update_record(52, feature_total=2.3))
        # Torn update line: must be ignored, earlier lines still served.
        with (root / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"cumulative_updates": 53, "torn": ')
        append_jsonl(root / "validation.jsonl", validation_record(50, feature_total=2.8))
        append_jsonl(
            root / "validation.jsonl",
            validation_record(
                250,
                is_best=True,
                action_diagnostic={
                    "metric": "physical_first5_mae7",
                    "value": 0.9,
                    "checkpoint_sha256": "f" * 64,
                    "timestamp": iso(),
                    "source": "offline",
                },
            ),
        )
        write_json(
            root / "selection.json",
            {
                "schema": "owac.raw-stem-temporal.feature-selection.v1",
                "selection_metric": "validation_token_total",
                "direction": "min",
                "tie_break": "earlier_step",
                "best_step": 250,
                "best_value": 1.01,
                "checkpoint_path": "checkpoints/checkpoint_000250.pt",
                "checkpoint_sha256": "9" * 64,
                "valid_tokens": 8192,
                "scalar_denominator": 16777216,
                "feature_mse": 1.0,
                "feature_cosine": 0.1,
                "parent_c50_baseline": {
                    "step": 50,
                    "checkpoint_sha256": "8" * 64,
                    "feature_total": 2.766,
                },
                "legacy_action_selector": {
                    "cleared": True,
                    "metric": "physical_first5_mae7",
                    "best_step": 0,
                    "best_value": 0.3056066959063605,
                    "checkpoint_sha256": "7" * 64,
                    "source": "pilot-C/selection.json",
                    "used_for_selection": True,
                },
                "updated_at": iso(),
                "test_used": False,
                "test_selection": False,
            },
        )
        write_json(
            root / "parent-baseline.json",
            {
                "schema": "owac.raw-stem-temporal.c-full.parent-baseline.v1",
                "step": 50,
                "checkpoint_sha256": "8" * 64,
                "feature_mse": 2.765593,
                "feature_cosine": 0.047423,
                "feature_total": 2.766,
                "valid_tokens": 8192,
                "scalar_denominator": 16777216,
                "windows": 102,
                "sources": 5,
                "episode_resets": 5,
            },
        )
        append_jsonl(root / "parent-history.jsonl", parent_record(1))
        append_jsonl(root / "parent-history.jsonl", parent_record(2))
        append_jsonl(root / "parent-history.jsonl", parent_record(50, feature_total=2.766))
        write_json(
            root / "supervisor-state.json",
            {
                "schema": "owac.raw-stem-temporal.c-full.supervisor-state.v1",
                "status": "running",
                "phase": "train",
                "pid": os.getpid(),
                "pid_start_ticks": start_ticks(os.getpid()),
                "child_pid": proc.pid,
                "child_pid_start_ticks": ticks,
                "child_pgid": proc.pid,
                "child_native_exit": None,
                "new_stage_wall_seconds": 60.0,
                "new_stage_wall_cap_seconds": 230400,
                "new_stage_output_bytes": 2048,
                "new_stage_output_cap_bytes": 21474836480,
                "last_error": None,
                "runner_command": [
                    "/usr/bin/python",
                    "-m",
                    "owac.runtime.raw_temporal_continuation",
                ],
                "secret": "never-expose",
                "updated_at": iso(),
            },
        )
        write_json(
            root / "supervisor-heartbeat.json",
            {
                "pid": os.getpid(),
                "pid_start_ticks": start_ticks(os.getpid()),
                "child_pid": proc.pid,
                "child_pid_start_ticks": ticks,
                "seq": 3,
                "updated_at": iso(),
                "status": "running",
            },
        )
        write_json(
            root / "acceptance" / "phase-0051.json",
            {
                "schema": "owac.sol.acceptance.v1",
                "status": "accepted",
                "step": 51,
                "checkpoint_sha256": "a" * 64,
                "reviewer": "sol",
                "note": "fixture acceptance",
                "accepted_at": iso(),
            },
        )

        snapshot = monitor.build_snapshot(root, config)
        assert snapshot["status"] == "running"
        assert snapshot["status_declared"] == "running"
        assert snapshot["stale"] is False
        assert snapshot["process"]["pid"] == proc.pid
        assert snapshot["process"]["pid_start_ticks"] == ticks
        assert snapshot["process"]["alive"] is True
        assert snapshot["process"]["state"] == "alive"
        assert snapshot["process"]["observed_start_ticks"] == ticks
        assert snapshot["process"]["heartbeat_age_seconds"] is not None
        assert snapshot["process"]["heartbeat_age_seconds"] <= 30
        assert snapshot["heartbeat"]["seq"] == 17

        assert len(snapshot["train_series"]) == 2
        assert snapshot["train_series"][0]["step"] == 51
        assert snapshot["train_series"][1]["feature_total"] == 2.3
        assert snapshot["series_read"]["updates"]["torn_tail"] is True
        assert len(snapshot["validation_series"]) == 2
        assert snapshot["validation_series"][1]["is_best"] is True

        progress = snapshot["progress"]
        assert progress["cumulative_updates"] == 51
        assert progress["new_stage_updates"] == 1
        assert progress["parent_cumulative_updates"] == 50
        assert progress["target_cumulative_updates"] == 5000
        assert progress["anchor_presentations"] == 816
        assert progress["new_stage_target_updates"] == 4950
        assert progress["new_stage_fraction"] == 1 / 4950
        assert progress["progress_fraction"] == 51 / 5000
        # The supervisor's cumulative wall/output accounts win over the runner's
        # per-phase record; both are real, with the wider accounting displayed.
        assert progress["new_stage_wall_seconds"] == 60.0
        assert progress["new_stage_wall_cap_seconds"] == 230400
        assert progress["wall_source"] == "supervisor-state.json"
        assert progress["output_bytes"] == 2048
        assert progress["output_cap_bytes"] == 21474836480
        assert progress["output_source"] == "supervisor-state.json"
        assert progress["gpu"]["allocated_bytes"] == 10765686784 + 2
        assert progress["gpu"]["reserved_bytes"] == 11117002752 + 2
        assert progress["gpu"]["record_step"] == 52
        # Ratios stay full precision in the API; the page does presentation rounding.
        assert progress["wall_fraction"] == 60.0 / 230400
        assert progress["disk_fraction"] == 2048 / 21474836480
        assert progress["output_scope"] == "all_new_run_root_bytes"
        assert "harness" in progress["output_scope_description"]
        assert snapshot["disk"]["new_stage_output_bytes"] == 2048
        assert snapshot["disk"]["new_stage_output_cap_bytes"] == 21474836480
        assert snapshot["disk"]["fraction"] == 2048 / 21474836480
        assert snapshot["disk"]["cap_covers_all_new_run_root_bytes"] is True
        assert "harness" in snapshot["disk"]["scope_description"]

        selection = snapshot["selection"]
        assert selection["available"] is True
        assert selection["metric"] == "validation_token_total"
        assert selection["best_step"] == 250
        assert selection["best_value"] == 1.01
        assert selection["checkpoint_sha256"] == "9" * 64
        assert selection["parent_c50_baseline"]["feature_total"] == 2.766
        assert selection["legacy_action_selector"]["used_for_selection"] is False
        assert selection["legacy_action_selector"]["declared_used_for_selection"] is True
        assert selection["test_used"] is False

        diagnostic = snapshot["action_diagnostic"]
        assert diagnostic["metric"] == "physical_first5_mae7"
        assert diagnostic["value"] == 0.9
        assert diagnostic["used_for_selection"] is False
        assert diagnostic["used_for_stopping"] is False
        assert diagnostic["record_step"] == 250

        assert snapshot["last_checkpoint"]["step"] == 51
        assert snapshot["last_checkpoint"]["path"] == "latest.pt"
        assert snapshot["last_checkpoint"]["sha256"] == "a" * 64
        assert snapshot["error"] is None
        assert snapshot["closed_loop"]["status"] == "not_run"
        assert snapshot["closed_loop"]["episodes"] == 0
        assert snapshot["closed_loop"]["control_restored"] is False

        parent = snapshot["parent_history"]
        assert parent["available"] is True
        assert [point["step"] for point in parent["points"]] == [1, 2, 50]
        assert parent["point_count"] == 3
        assert parent["last_step"] == 50
        assert parent["baseline"]["feature_total"] == 2.766
        assert parent["baseline"]["windows"] == 102
        assert parent["baseline_present"] is True

        supervisor = snapshot["supervisor"]
        assert supervisor["present"] is True
        assert supervisor["alive"] is True
        assert supervisor["child_pid"] == proc.pid
        assert supervisor["child_alive"] is True
        assert supervisor["runner_command_basenames"] == [
            "python",
            "-m",
            "owac.runtime.raw_temporal_continuation",
        ]
        assert "never-expose" not in json.dumps(snapshot, ensure_ascii=False)
        assert snapshot["acceptance"]["present"] is True
        assert snapshot["acceptance"]["step"] == 51
        assert snapshot["acceptance"]["matches_last_checkpoint"] is True
        assert snapshot["acceptance"]["matches_selection_checkpoint"] is False
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False)

        # The same recorded PID is now dead: it must never keep displaying running.
        wait_for_exit(proc)
        dead = monitor.build_snapshot(root, config)
        assert dead["status"] != "running"
        assert dead["status"] == "stale"
        assert dead["stale"] is True
        assert dead["stale_reason"] == "process_dead"
        assert dead["process"]["alive"] is False
        assert dead["process"]["state"] == "dead"
        assert dead["status_declared"] == "running"
        assert len(dead["train_series"]) == 2
    finally:
        if proc.poll() is None:
            wait_for_exit(proc)


def test_reused_pid_and_missing_starttime_never_show_running(tmp_path):
    proc = spawn_sleeper()
    try:
        ticks = start_ticks(proc.pid)
        reused_root = tmp_path / "reused"
        reused_root.mkdir()
        write_json(reused_root / "run-status.json", run_status(proc.pid, ticks + 1))
        write_json(reused_root / "heartbeat.json", heartbeat(proc.pid, ticks + 1))
        reused = monitor.build_snapshot(reused_root)
        assert reused["process"]["alive"] is False
        assert reused["process"]["state"] == "reused"
        assert reused["process"]["observed_start_ticks"] == ticks
        assert reused["status"] == "stale"
        assert reused["stale_reason"] == "process_reused"

        unknown_root = tmp_path / "unverified"
        unknown_root.mkdir()
        status = run_status(proc.pid, None)
        status.pop("pid_start_ticks")
        write_json(unknown_root / "run-status.json", status)
        unknown = monitor.build_snapshot(unknown_root)
        assert unknown["process"]["alive"] is None
        assert unknown["process"]["state"] == "unknown"
        assert unknown["status"] != "running"
        assert unknown["status"] == "unknown"

        dead_pid = proc.pid
        wait_for_exit(proc)
        dead_root = tmp_path / "dead"
        dead_root.mkdir()
        write_json(dead_root / "run-status.json", run_status(dead_pid, ticks))
        write_json(dead_root / "heartbeat.json", heartbeat(dead_pid, ticks))
        dead = monitor.build_snapshot(dead_root)
        assert dead["process"]["alive"] is False
        assert dead["process"]["state"] == "dead"
        assert dead["status"] == "stale"
        assert dead["stale_reason"] == "process_dead"
    finally:
        if proc.poll() is None:
            wait_for_exit(proc)


def test_stale_heartbeat_marks_live_process_stale_and_fresh_heartbeat_recovers(tmp_path):
    proc = spawn_sleeper()
    try:
        ticks = start_ticks(proc.pid)
        root = tmp_path / "run"
        root.mkdir()
        old = iso(NOW - timedelta(seconds=120))
        write_json(root / "run-status.json", run_status(proc.pid, ticks, updated_at=old))
        write_json(
            root / "heartbeat.json",
            heartbeat(proc.pid, ticks, updated_at=old, seq=1),
        )
        stale = monitor.build_snapshot(root)
        assert stale["process"]["alive"] is True
        assert stale["process"]["heartbeat_age_seconds"] >= 100
        assert stale["heartbeat"]["stale"] is True
        assert stale["status"] == "stale"
        assert stale["stale_reason"] == "heartbeat_stale"

        write_json(root / "run-status.json", run_status(proc.pid, ticks))
        write_json(root / "heartbeat.json", heartbeat(proc.pid, ticks))
        fresh = monitor.build_snapshot(root)
        assert fresh["process"]["alive"] is True
        assert fresh["status"] == "running"
        assert fresh["stale"] is False

        # A live, verified PID without any heartbeat is not "running" once the
        # run-status itself is old; missing liveness evidence stays honest.
        write_json(root / "run-status.json", run_status(proc.pid, ticks, updated_at=old))
        (root / "heartbeat.json").unlink()
        no_heartbeat = monitor.build_snapshot(root)
        assert no_heartbeat["process"]["alive"] is True
        assert no_heartbeat["heartbeat"] is None
        assert no_heartbeat["status"] == "stale"
        assert no_heartbeat["stale_reason"] == "heartbeat_missing"
    finally:
        if proc.poll() is None:
            wait_for_exit(proc)


# ---------------------------------------------------------------------------
# Read-only HTTP surface.
# ---------------------------------------------------------------------------


def request(server, method, path):
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


def test_http_routes_are_restricted_head_and_read_only(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    server = monitor.create_server("127.0.0.1", 0, root, None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response, body = request(server, "GET", "/")
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/html")
        text = body.decode("utf-8")
        assert "<title>RAW学生蒸馏训练进度</title>" in text
        assert "RAW学生蒸馏训练进度" in text
        assert "/api/snapshot" in text
        assert "setInterval(poll, 5000)" in text
        assert "未知" in text
        assert "http://" not in text and "https://" not in text
        assert "<script src" not in text and "<link" not in text
        assert response.getheader("Cache-Control").startswith("no-store")
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert "default-src 'none'" in response.getheader("Content-Security-Policy")

        with_index = request(server, "GET", "/index.html")
        assert with_index[0].status == 200 and with_index[1] == body

        response, body = request(server, "GET", "/api/snapshot")
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("application/json")
        snapshot = json.loads(body)
        assert snapshot["schema"] == monitor.SNAPSHOT_SCHEMA
        assert snapshot["status"] == "preparing"
        assert snapshot["train_series"] == []
        assert snapshot["process"]["alive"] is None
        assert snapshot["closed_loop"]["status"] == "not_run"

        response, body = request(server, "HEAD", "/api/snapshot")
        assert response.status == 200
        assert int(response.getheader("Content-Length")) > 0
        assert body == b""

        response, body = request(server, "HEAD", "/")
        assert response.status == 200
        assert body == b""

        for path in (
            "/favicon.ico",
            "/api/status",
            "/api/train",
            "/start",
            "/api/start",
            "/api/stop",
            "/api/snapshot/",
            "/api/snapshot?file=/etc/passwd",
            "/api/snapshot?path=/etc/passwd",
            "/../etc/passwd",
            "//etc/passwd",
            "/api/snapshot/../../etc/passwd",
        ):
            response, _ = request(server, "GET", path)
            assert response.status == 404, path

        for method in ("POST", "PUT", "DELETE", "PATCH", "TRACE", "OPTIONS"):
            response, body = request(server, method, "/api/snapshot")
            assert response.status == 405, method
            assert response.getheader("Allow") == "GET, HEAD"
            assert "read-only" in body.decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Selection honesty, redaction, and standalone stdlib-only entry.
# ---------------------------------------------------------------------------


def test_selection_is_token_metric_and_action_diagnostic_is_separate(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_json(
        root / "run-status.json",
        {"status": "preparing", "selection_metric": "validation_token_total", "best": None},
    )
    append_jsonl(
        root / "validation.jsonl",
        validation_record(
            250,
            feature_total=1.01,
            is_best=True,
            action_diagnostic={
                "metric": "physical_first5_mae7",
                "value": 0.9,
                "checkpoint_sha256": "f" * 64,
                "timestamp": iso(),
                "source": "offline",
            },
        ),
    )
    write_json(
        root / "selection.json",
        {
            "selection_metric": "validation_token_total",
            "direction": "min",
            "tie_break": "earlier_step",
            "best_step": 250,
            "best_value": 1.01,
            "checkpoint_path": "checkpoints/checkpoint_000250.pt",
            "checkpoint_sha256": "9" * 64,
            "legacy_action_selector": {
                "cleared": True,
                "metric": "physical_first5_mae7",
                "best_step": 0,
                "best_value": 0.3056,
                "checkpoint_sha256": "7" * 64,
                "source": "pilot-C/selection.json",
                "used_for_selection": False,
            },
        },
    )
    snapshot = monitor.build_snapshot(root)
    assert snapshot["selection"]["metric"] == "validation_token_total"
    assert snapshot["selection"]["best_step"] == 250
    assert snapshot["selection"]["best_value"] == 1.01
    assert snapshot["selection"]["legacy_action_selector"]["metric"] == "physical_first5_mae7"
    assert snapshot["selection"]["legacy_action_selector"]["used_for_selection"] is False
    assert snapshot["action_diagnostic"]["metric"] == "physical_first5_mae7"
    assert snapshot["action_diagnostic"]["used_for_selection"] is False


def test_snapshot_redacts_secrets_and_absolute_paths(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("OWAC_MONITOR_SECRET", "environment-secret-value")
    secret = "hunter2-do-not-expose"
    write_json(
        root / "run-status.json",
        {
            "status": "running",
            "pid": 123456789,
            "pid_start_ticks": 1,
            "updated_at": iso(),
            "cumulative_updates": 51,
            "last_checkpoint": {
                "step": 51,
                "path": "/home/private/run/checkpoints/latest.pt",
                "sha256": "a" * 64,
                "kind": "interval",
            },
            "api_token": secret,
            "environment": "SHOULD_NOT_APPEAR",
            "error": None,
        },
    )
    write_json(
        root / "heartbeat.json",
        {"pid": 123456789, "pid_start_ticks": 1, "updated_at": iso(), "seq": 1},
    )
    write_json(
        root / "selection.json",
        {
            "selection_metric": "validation_token_total",
            "best_step": 51,
            "best_value": 1.0,
            "secret_key": secret,
        },
    )
    write_json(
        root / "supervisor-state.json",
        {
            "status": "running",
            "pid": 123456789,
            "pid_start_ticks": 1,
            "child_pid": 123456789,
            "runner_command": ["/usr/bin/python", "-m", "owac.runtime.raw_temporal_continuation"],
            "secret_token": secret,
        },
    )
    snapshot = monitor.build_snapshot(root, tmp_path / "config.json")
    packed = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    assert secret not in packed
    assert "environment-secret-value" not in packed
    assert "SHOULD_NOT_APPEAR" not in packed
    assert str(tmp_path) not in packed
    assert "/home/private" not in packed
    assert snapshot["last_checkpoint"]["path"] == "latest.pt"
    assert snapshot["run_root"] == root.name
    assert snapshot["supervisor"]["runner_command_basenames"] == [
        "python",
        "-m",
        "owac.runtime.raw_temporal_continuation",
    ]


def test_module_is_standalone_stdlib_only_and_cli_help(tmp_path):
    module_path = Path(monitor.__file__).resolve()
    root = tmp_path / "run"
    root.mkdir()
    program = (
        "import json, runpy, sys\n"
        f"module = runpy.run_path({str(module_path)!r})\n"
        f"snapshot = module['build_snapshot']({str(root)!r})\n"
        "print(json.dumps({'snapshot': snapshot, 'owac_imported': 'owac' in sys.modules},"
        " ensure_ascii=False, allow_nan=False))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["owac_imported"] is False
    assert payload["snapshot"]["status"] == "preparing"
    assert payload["snapshot"]["closed_loop"]["status"] == "not_run"

    helped = subprocess.run(
        [sys.executable, "-m", "owac.runtime.raw_distillation_monitor", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert helped.returncode == 0
    for flag in ("--config", "--run-root", "--host", "--port", "--poll-seconds"):
        assert flag in helped.stdout
    assert monitor.DEFAULT_HOST == "127.0.0.1"
    assert monitor.DEFAULT_PORT == 8878
    assert monitor.DEFAULT_POLL_SECONDS == 5


def test_actual_supervisor_state_shape_loading_model_and_prepared_stopped(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    old = NOW - timedelta(minutes=5)
    write_json(
        root / "run-status.json",
        run_status(
            dead.pid,
            123456,
            status="prepared",
            phase="preparing",
            updated_at=iso(old),
            finished_at=iso(old),
            cumulative_updates=50,
            new_stage_updates=0,
            anchor_presentations=800,
            native_exit=0,
        ),
    )
    write_json(
        root / "heartbeat.json",
        heartbeat(
            dead.pid,
            123456,
            phase="preparing",
            updated_at=iso(old),
            cumulative_updates=50,
        ),
    )
    inputs = {
        "runner_python": sys.executable,
        "config": str(tmp_path / "config.json"),
        "migration": str(tmp_path / "migration.json"),
        "authority": str(tmp_path / "authority.json"),
        "device": "cuda:0",
        "parity_receipt": None,
        "resource_receipt": None,
    }
    prepared = {
        "path": "prepared.json",
        "sha256": "a" * 64,
        "prepare_wall_seconds": 95.97436231200118,
        "failed_preflight_charge_seconds": 114.53577269776724,
        "failed_preflight_receipts": [],
    }
    supervisor = Supervisor(root)
    state = supervisor._initial_state(inputs=inputs, stop_after_update=51, prepared=prepared)
    child = spawn_sleeper()
    try:
        child_ticks = start_ticks(child.pid)
        state.update(
            {
                "status": "running",
                "phase": "phase-1",
                "child_pid": child.pid,
                "child_pgid": child.pid,
                "child_start_ticks": child_ticks,
                "child_native_exit": 0,  # prior phase exit is not this live child's exit
            }
        )
        supervisor._save_state(state)
        supervisor._write_heartbeat(state, force=True)
        assert (root / "supervisor-heartbeat.json").is_file()

        first = monitor.build_snapshot(root, None, now=datetime.now().astimezone())
        supervisor_block = first["supervisor"]
        assert supervisor_block["pid"] == os.getpid()
        assert supervisor_block["pid_start_ticks"] == start_ticks(os.getpid())
        assert supervisor_block["child_pid"] == child.pid
        assert supervisor_block["child_pid_start_ticks"] == child_ticks
        assert supervisor_block["child_start_ticks"] == child_ticks
        assert supervisor_block["heartbeat"]["child_start_ticks"] == child_ticks
        assert supervisor_block["supervisor_fresh"] is True
        assert supervisor_block["child_alive"] is True
        assert supervisor_block["live_verified"] is True
        assert supervisor_block["child_native_exit"] is None
        assert first["status"] == "preparing"
        assert first["status_detail"] == "loading_model"
        assert first["stale"] is False
        assert first["process"]["active_source"] == "supervisor_child"
        assert first["process"]["pid"] == child.pid
        assert first["process"]["runner_pid"] == dead.pid
        assert first["process"]["alive"] is True
        # A new verified child also supersedes the prior phase's terminal state.
        old_status = json.loads((root / "run-status.json").read_text())
        for previous in ("stopped_at_update_limit", "stopped"):
            write_json(root / "run-status.json", {**old_status, "status": previous})
            resumed = monitor.build_snapshot(root, None, now=datetime.now().astimezone())
            assert resumed["status"] == "preparing"
            assert resumed["status_detail"] == "loading_model"
        write_json(root / "run-status.json", old_status)
    finally:
        wait_for_exit(child)

    # Old prepare PID dead, supervisor child gone: prepared is a real terminal
    # preflight state waiting for training, not a stale/error/completed run.
    after = monitor.build_snapshot(root, None, now=datetime.now().astimezone())
    assert after["supervisor"]["live_verified"] is False
    assert after["status"] == "prepared"
    assert after["status_detail"] == "preflight_complete_waiting_training"
    assert after["stale"] is False

    # stopped is its own status, not completed or an error.
    stopped = run_status(
        dead.pid,
        123456,
        status="stopped",
        phase="finished",
        updated_at=iso(old),
        finished_at=iso(old),
        cumulative_updates=51,
        native_exit=0,
    )
    write_json(root / "run-status.json", stopped)
    stopped_snapshot = monitor.build_snapshot(root, None, now=datetime.now().astimezone())
    assert stopped_snapshot["status"] == "stopped"
    assert stopped_snapshot["status_detail"] == "stopped"
    assert stopped_snapshot["stale"] is False

    # A legacy heartbeat phase "finished" alone must map to stopped, never to
    # successful completion.
    finished_root = tmp_path / "finished"
    finished_root.mkdir()
    write_json(
        finished_root / "heartbeat.json",
        heartbeat(
            dead.pid,
            123456,
            phase="finished",
            updated_at=iso(old),
            cumulative_updates=51,
        ),
    )
    finished = monitor.build_snapshot(finished_root, None, now=datetime.now().astimezone())
    assert finished["status"] == "stopped"
    assert finished["status"] != "completed"


def test_page_copy_is_plain_chinese_and_dom_contract_preserved():
    html_path = Path(monitor.__file__).with_name("raw_distillation_monitor.html")
    html = html_path.read_text(encoding="utf-8")
    assert "<title>RAW学生蒸馏训练进度</title>" in html
    for phrase in (
        "教师和学生接收同源RAW观测",
        "学生学习教师交给动作模型的视觉特征",
        "已训练50步的学生继续训练，累计最多5000步",
        "只训练RAW输入模块和历史融合门控，其他模型参数冻结",
        "一步＝用一批样本更新一次参数",
        "不是一次机器人执行，也不是把全部训练数据完整学习一遍",
        "每250个训练步进行一次留出验证",
        "特征数值误差（越低越接近教师）",
        "特征方向误差（越低越好）",
        "总损失＝特征数值误差＋0.1×特征方向误差",
        "做过参数更新的训练样本",
        "不用于参数更新的留出示范",
        "较低的特征误差不代表动作成功",
        "续训前的50步训练记录",
        "开始本轮训练时的模型（已训练50步）",
        "累计训练步数",
        "本轮续训",
        "选择最佳模型",
        "保存的模型",
        "技术详情（核查信息",
    ):
        assert phrase in html, phrase

    main = html.split("<main>", 1)[1].split("</main>", 1)[0]
    main = re.sub(r"<script\b.*?</script>", "", main, flags=re.S)
    main = re.sub(r"<style\b.*?</style>", "", main, flags=re.S)
    main = re.sub(r"<details\b.*?</details>", "", main, flags=re.S)
    for banned in (
        "父阶段",
        "新阶段",
        "C50",
        "C51",
        "C-full",
        "C-FULL",
        "累计更新",
        "选型",
        "phase-1",
        "phase-2",
        "stop_after_update",
        "validation_token_total",
        "sidecar",
        "provenance",
    ):
        assert banned not in main, banned

    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), "duplicate DOM ids"
    # Small interface-ID guard.  The full 119-ID comparison against the frozen
    # pre-change DOM is an independent source-freeze check and must not make
    # generic pytest depend on the ignored local outputs directory.
    required_ids = {
        "status-badge",
        "status-text",
        "status-sub",
        "run-root",
        "config-text",
        "generated-at",
        "poll-seconds",
        "stat-cumulative",
        "stat-validation",
        "stat-best",
        "stat-checkpoint",
        "bar-cumulative",
        "chart-total",
        "chart-mse",
        "chart-cosine",
        "parent-points",
        "validation-table",
        "sel-metric",
        "diag-body",
        "closed-status",
        "proc-pid",
        "sup-status",
        "acc-status",
        "updates-table",
        "warnings-panel",
        "last-refresh",
    }
    assert required_ids <= set(ids)

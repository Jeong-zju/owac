"""Phase budget ledger tests; no GPU or real training."""

from __future__ import annotations

import json
import os

import pytest

from owac.runtime.rawvla_teacher_budget import BudgetExceeded, PhaseBudget


def test_two_eleven_hour_phases_cannot_bypass_twelve_hour_cap(tmp_path):
    clock = {"now": 0.0}
    budget = PhaseBudget(tmp_path, clock=lambda: clock["now"])
    with budget.lease("first", gpu=True):
        clock["now"] = 11 * 3600
    with pytest.raises(BudgetExceeded, match="GPU"):
        with PhaseBudget(tmp_path, clock=lambda: clock["now"]).lease(
            "second", gpu=True, reserved_gpu_seconds=11 * 3600
        ):
            pass


def test_asset_new_bytes_and_counters_are_global(tmp_path):
    budget = PhaseBudget(tmp_path)
    with budget.lease("replay", gpu=False, replays=50, new_bytes=10):
        pass
    with pytest.raises(BudgetExceeded, match="replay"):
        with budget.lease("again", gpu=False, replays=1):
            pass
    with budget.lease("train", gpu=False, updates=2000, anchors=16000):
        pass
    with pytest.raises(BudgetExceeded, match="update"):
        with budget.lease("more", gpu=False, updates=1):
            pass


def test_live_lease_refused_and_stale_lease_not_stolen(tmp_path):
    budget = PhaseBudget(tmp_path)
    lease = tmp_path / "budget" / "lease.json"
    lease.parent.mkdir(parents=True)
    lease.write_text(json.dumps({"pid": os.getpid(), "scope": "live"}))
    with pytest.raises(BudgetExceeded, match="live pid"):
        with budget.lease("blocked", gpu=False):
            pass
    lease.write_text(json.dumps({"pid": 999999999, "scope": "stale"}))
    with pytest.raises(BudgetExceeded, match="stale"):
        with budget.lease("blocked", gpu=False):
            pass


def test_successful_lease_is_atomic_and_removed(tmp_path):
    budget = PhaseBudget(tmp_path)
    with budget.lease("status-check", gpu=False, new_bytes=0):
        assert json.loads((tmp_path / "budget" / "lease.json").read_text())["pid"] == os.getpid()
    ledger = json.loads((tmp_path / "budget" / "ledger.json").read_text())
    assert ledger["schema"] == "owac.raw-teacher.budget.v1"
    assert not (tmp_path / "budget" / "lease.json").exists()


def test_existing_legacy_ledger_is_preserved_not_reset():
    from pathlib import Path

    root = Path("outputs/rawvla-teacher-20261020")
    ledger = root / "budget" / "ledger.json"
    if not ledger.is_file():
        pytest.skip("real teacher ledger is absent in this checkout")
    before = ledger.read_bytes()
    lease = root / "budget" / "lease.json"
    lease_before = lease.read_bytes() if lease.is_file() else None
    budget = PhaseBudget(root)
    assert budget._state["replays"] == 50
    assert budget._state["updates"] == json.loads(before)["updates"]
    assert budget._state["gpu_seconds"] > 0
    # Reading must not rewrite/restart the ledger or alter another process's lease.
    assert ledger.read_bytes() == before
    assert (lease.read_bytes() if lease.is_file() else None) == lease_before


def test_pending_update_stops_resume_without_silent_redo(tmp_path):
    budget = PhaseBudget(tmp_path)
    with budget.lease("interrupted-update", gpu=False):
        budget.charge_update(anchors=8)
    with pytest.raises(BudgetExceeded, match="pending update"):
        with PhaseBudget(tmp_path).lease("resume", gpu=False):
            pass
    ledger = json.loads((tmp_path / "budget/ledger.json").read_text())
    assert (ledger["updates"], ledger["pending_updates"], ledger["pending_anchors"]) == (0, 1, 8)


def test_disk_includes_assets_outside_trainer_and_active_deadline(tmp_path):
    clock = {"now": 1.0}
    budget = PhaseBudget(tmp_path, authorized_new_bytes=5000, clock=lambda: clock["now"])
    (tmp_path / "dataset.bin").write_bytes(b"x" * 5000)
    with pytest.raises(BudgetExceeded, match="phase-wide disk"):
        budget.check_disk(reserve_bytes=1)
    with budget.lease("short-probe", reserved_gpu_seconds=5):
        clock["now"] = 6.0
        with pytest.raises(BudgetExceeded, match="deadline"):
            budget.check_deadline()


def test_authority_requires_canonical_ledger_and_mutation_requires_lease(tmp_path):
    (tmp_path / "authority.json").write_text(
        json.dumps({"schema": "owac.raw-teacher.authorization.v1", "run_root": str(tmp_path)})
    )
    with pytest.raises(BudgetExceeded, match="canonical"):
        PhaseBudget(tmp_path, ledger_path=tmp_path / "new-counts.json")
    budget = PhaseBudget(tmp_path)
    with pytest.raises(BudgetExceeded, match="owned active"):
        budget.charge_update(anchors=8)
    with budget.lease("committed", gpu=False):
        budget.charge_update(anchors=8)
        budget.commit_update()
        budget.charge_policy_episode("validation")
    ledger = json.loads((tmp_path / "budget/ledger.json").read_text())
    assert (ledger["updates"], ledger["anchors"], ledger["validation_episodes"]) == (1, 8, 1)
    assert len(ledger["authority_sha256"]) == 64

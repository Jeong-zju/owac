"""Full-baseline recovery must preserve all previous outcomes and fixed identities."""

import pytest

from owac.runtime.rawvla_main_repair import missing_cases


@pytest.fixture
def frozen_cases():
    domains = ["ExtremeLow", "Low", "Normal", "Over", "ExtremeOver"]
    entries = [
        {
            "task_name": "grab_roller" if task == 0 else f"task-{task}",
            "dataset_seed": seed,
            "lighting_domain": domain,
        }
        for task in range(13)
        for seed in range(50)
        for domain in domains
    ]
    ledger = [
        {"case_id": f"robotwin-{index:04d}", "entry": entries[index], "success": False}
        for index in range(33, 3250)
    ]
    attempts = [
        {"case_id": f"robotwin-{index:04d}", "entry": entries[index], "returncode": 1}
        for index in range(33)
    ]
    return {"entries": entries}, ledger, attempts


def test_selects_only_original_33_and_disjoint_worker_assignment(frozen_cases):
    cases = missing_cases(*frozen_cases)
    assert len(cases) == 33
    assert {row["case_id"] for row in cases} == {f"robotwin-{index:04d}" for index in range(33)}
    workers = [cases[index::3] for index in range(3)]
    assert all(len(worker) == 11 for worker in workers)
    assert len({row["case_id"] for worker in workers for row in worker}) == 33


def test_rejects_changed_inherited_identity(frozen_cases):
    manifest, ledger, attempts = frozen_cases
    ledger[0] = dict(ledger[0], entry=dict(ledger[0]["entry"], dataset_seed=999))
    with pytest.raises(ValueError, match="differs"):
        missing_cases(manifest, ledger, attempts)


def test_rejects_unknown_failure_without_replacing_seed(frozen_cases):
    manifest, ledger, attempts = frozen_cases
    attempts.pop()
    with pytest.raises(ValueError, match="outside diagnosed"):
        missing_cases(manifest, ledger, attempts)


def test_rejects_changed_completion_count(frozen_cases):
    manifest, ledger, attempts = frozen_cases
    ledger.pop()
    with pytest.raises(ValueError, match="3217"):
        missing_cases(manifest, ledger, attempts)

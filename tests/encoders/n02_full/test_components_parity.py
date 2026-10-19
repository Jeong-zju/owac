"""Contract tests for the parity suite interface and its recorded verdicts.

The suite is the interface ``evaluation/n02_full/p0.py parity`` calls, so its shape is frozen:
``PARITY_VERSION``, ``available_checks()``, ``run_all(config, device, dtype, checks)`` returning
status/counts/tolerances/versions, never raising, and reporting missing dependencies as
``not_run`` with a concrete reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from owac.frameworks.n02_full.components import parity_suite

OUTPUT_DIR = Path("outputs/n02-full/p0/parity")
FIXTURE_DIR = OUTPUT_DIR / "fixtures"


def test_parity_version_and_available_checks():
    assert parity_suite.PARITY_VERSION == "n02-full-components-v1"
    checks = parity_suite.available_checks()
    assert isinstance(checks, tuple)
    for expected in (
        "convnext_base_block",
        "convnext_base_multiscale",
        "fcmae_patchify_roundtrip",
        "fcmae_mask_aware_dense",
        "savi_slot_attention",
        "dnri_one_step_causal",
        "theseus_lm_quadratic",
        "smc_ref",
        "short_train_trend",
    ):
        assert expected in checks


def test_parity_tolerances_match_the_p0_contract():
    assert parity_suite.TOLERANCES["fp32_forward"] == {"atol": 1e-5, "rtol": 1e-4}
    assert parity_suite.TOLERANCES["fp32_grad"] == {"atol": 1e-5, "rtol": 1e-3}
    assert parity_suite.TOLERANCES["fp64_analytic"]["atol"] == 1e-8


def test_run_all_returns_the_frozen_payload_shape_and_never_raises():
    payload = parity_suite.run_all(
        {"fixture_dir": str(FIXTURE_DIR), "output_dir": str(OUTPUT_DIR)},
        checks=["convnext_base_block", "does_not_exist"],
    )
    assert payload["status"] in {"pass", "fail", "not_run", "incomplete"}
    for key in (
        "checks",
        "pass_count",
        "fail_count",
        "not_run_count",
        "tolerances",
        "versions",
    ):
        assert key in payload
    assert set(payload["versions"]) >= {"torch", "python", "jax", "theseus"}
    assert payload["fail_count"] == 0
    assert payload["checks"]["does_not_exist"]["status"] == "not_run"
    for entry in payload["checks"].values():
        assert set(entry) >= {
            "status",
            "max_abs_err",
            "max_rel_err",
            "atol",
            "rtol",
            "reason",
            "evidence_path",
        }


def test_every_check_reports_a_specific_reason():
    payload = parity_suite.run_all({"fixture_dir": str(FIXTURE_DIR), "output_dir": str(OUTPUT_DIR)})
    for name, entry in payload["checks"].items():
        assert entry["reason"], name
        if entry["status"] == "not_run":
            assert len(entry["reason"]) > 20, (name, entry["reason"])
        if entry["status"] == "pass":
            assert entry["max_abs_err"] is not None, name
            assert entry["atol"] is not None and entry["rtol"] is not None, name


def test_no_check_fails_with_the_exported_reference_fixtures():
    payload = parity_suite.run_all({"fixture_dir": str(FIXTURE_DIR), "output_dir": str(OUTPUT_DIR)})
    failures = {name: e["reason"] for name, e in payload["checks"].items() if e["status"] == "fail"}
    assert failures == {}
    # reference-backed checks must really be reference-backed, not silently skipped
    for name in (
        "convnext_base_block",
        "convnext_base_multiscale",
        "fcmae_patchify_roundtrip",
        "dnri_one_step_causal",
        "savi_slot_attention",
        "short_train_trend",
    ):
        if not (FIXTURE_DIR / f"{name}.pt").exists() and not any(
            (FIXTURE_DIR / f"{name}.npz",).__iter__()
        ):
            continue
        assert payload["checks"][name]["status"] in {"pass", "fail"}, name


def test_missing_fixture_directory_yields_not_run_not_crash(tmp_path):
    payload = parity_suite.run_all(
        {"fixture_dir": str(tmp_path / "empty"), "output_dir": str(tmp_path)},
        checks=["convnext_base_block", "dnri_one_step_causal", "short_train_trend"],
    )
    assert payload["fail_count"] == 0
    assert payload["not_run_count"] == 3
    for entry in payload["checks"].values():
        assert "not exported" in entry["reason"] or "fixture" in entry["reason"]


def test_smc_check_is_not_run_without_numba_and_pass_with_it():
    payload = parity_suite.run_all(
        {"fixture_dir": str(FIXTURE_DIR), "output_dir": str(OUTPUT_DIR)},
        checks=["smc_ref"],
    )
    entry = payload["checks"]["smc_ref"]
    try:
        import numba  # noqa: F401

        has_numba = True
    except Exception:  # noqa: BLE001
        has_numba = False
    if has_numba:
        assert entry["status"] == "pass", entry["reason"]
    else:
        assert entry["status"] == "not_run"
        assert "numba" in entry["reason"]


def test_recorded_receipts_agree_with_the_live_run():
    receipt = OUTPUT_DIR / "parity.json"
    if not receipt.exists():
        pytest.skip(f"canonical receipt {receipt} not written yet")
    recorded = json.loads(receipt.read_text())
    assert recorded["parity_version"] == parity_suite.PARITY_VERSION
    live = parity_suite.run_all({"fixture_dir": str(FIXTURE_DIR), "output_dir": str(OUTPUT_DIR)})
    recorded_checks = recorded["checks"]
    for name, entry in live["checks"].items():
        if name not in recorded_checks:
            continue
        # the recorded verdict may differ only by being stricter (e.g. a skipped dependency),
        # never by turning a failure into a pass
        if recorded_checks[name]["status"] == "fail":
            assert entry["status"] == "fail", name
        assert recorded_checks[name]["atol"] == entry["atol"], name


@pytest.mark.parametrize("dtype", ["float32"])
def test_run_all_on_explicit_device_argument(dtype):
    payload = parity_suite.run_all(
        {"fixture_dir": str(FIXTURE_DIR), "output_dir": str(OUTPUT_DIR)},
        device="cpu",
        dtype=dtype,
        checks=["convnext_base_block", "fcmae_mask_aware_dense"],
    )
    assert payload["device"] == "cpu"
    assert payload["dtype"] == dtype
    assert all(e["status"] != "fail" for e in payload["checks"].values())

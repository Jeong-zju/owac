"""P0 data tests, part 3: the real-NPZ audit mechanism.

The NPZ files used here are contract fixtures written into ``tmp_path`` (see ``conftest.py``):
they follow the documented key schema but they are **not** real RawVLA data. The real cache is
absent on this workstation (2026-10-14), so these tests verify the audit *mechanism* and its
honesty rules: a missing or empty root is ``not_run`` with a reason and expected fields, every
reported number comes from reading the file, the fixture run is marked as such and never claims
real-cache evidence, and the RoboTwin formal-statistics gap is flagged rather than smoothed over.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from owac.frameworks.n02_full import data as d

from .conftest import LIBERO_SUITE_A, ROBOTWIN_TASKS

HEX_DIGITS = set("0123456789abcdef")


def check_by_id(report: dict, check_id: str) -> dict:
    for check in report["protocol_checks"]:
        if check["id"] == check_id:
            return check
    raise AssertionError(f"missing protocol check {check_id!r}")


def audit_libero_cache(tmp_path, writer, *, num_frames: int = 24) -> tuple[Path, dict]:
    record = writer(tmp_path / "libero", LIBERO_SUITE_A, 18, 0, num_frames=num_frames)
    return tmp_path / "libero", record


def test_audit_missing_root_reports_not_run_with_reason_and_expected_fields(tmp_path) -> None:
    missing = tmp_path / "does_not_exist"
    report = d.audit_dataset_root(missing, "libero")
    assert report["status"] == "not_run"
    assert report["reason"].startswith("cache_root_missing")
    assert str(missing) in report["reason"]
    assert report["checked_path"] == str(missing)
    assert report["files_checked"] == 0
    assert report["expected_fields"]["measurement_keys"] == [
        "agentview_raw_uint8",
        "wrist_raw_uint8",
    ]
    assert report["expected_fields"]["state_source_key"] == "robot_states"
    assert report["expected_fields"]["action_supervision_expression"] == "actions[t:t+10]"
    assert report["expected_fields"]["executed_action_window"] == 5
    assert "files" not in report
    assert report["real_cache_evidence"] is False


def test_audit_empty_root_reports_not_run_without_inventing_files(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report = d.audit_dataset_root(empty, "robotwin")
    assert report["status"] == "not_run"
    assert report["reason"].startswith("no_npz_found")
    assert report["search_pattern"] == "no_npz_found"
    assert report["files_total"] == 0
    assert report["expected_fields"]["measurement_keys"] == [
        "head_camera_raw_float32",
        "left_camera_raw_float32",
        "right_camera_raw_float32",
    ]


def test_audit_libero_fixture_reports_measured_fields_and_verdicts(tmp_path, libero_writer) -> None:
    root, record = audit_libero_cache(tmp_path, libero_writer)
    report = d.audit_dataset_root(root, "libero", evidence_kind="contract_fixture")
    assert report["status"] == "incomplete"
    assert report["evidence_kind"] == "contract_fixture"
    assert report["real_cache_evidence"] is False
    assert report["files_checked"] == 1
    assert report["problems"] == []

    entry = report["files"][0]
    expected_relative = f"data/libero/raw/{LIBERO_SUITE_A}/episode_000018/variant_00.npz"
    assert entry["relative_path"] == expected_relative
    assert len(entry["sha256"]) == 64
    assert entry["num_frames"] == 24
    agentview = entry["measurements"]["agentview_raw_uint8"]
    assert agentview["present"] is True
    assert agentview["dtype"] == "uint8"
    assert agentview["shape"] == [24, 8, 8, 3]
    assert agentview["resolution_hw"] == [8, 8]
    assert agentview["channels"] == 3
    assert agentview["min"] == float(record["agentview"].min())
    assert agentview["max"] == float(record["agentview"].max())
    assert agentview["finite"] is True
    quantiles = agentview["quantiles"]
    assert quantiles["q01"] <= quantiles["q50"] <= quantiles["q99"]
    assert agentview["quantile_kind"] == "exact"
    wrist = entry["measurements"]["wrist_raw_uint8"]
    assert wrist["resolution_hw"] == [6, 10]
    assert entry["camera_order_matches_declared"] is True
    assert entry["file_measurement_key_order"] == [
        "agentview_raw_uint8",
        "wrist_raw_uint8",
    ]
    assert entry["state_source_key"] == "robot_states"
    assert entry["state_source_width"] == 8
    assert entry["language"]["task_text"]
    assert entry["timestamp"]["array_keys_found"] == []
    assert entry["noise"]["declared"] is False
    assert entry["actions"]["present"] is True
    assert entry["actions"]["columns"] == 7
    assert len(entry["actions"]["normalisation_stats_sha256"]) == 64
    assert set(entry["actions"]["normalisation_stats_sha256"]) <= HEX_DIGITS
    assert entry["actions"]["units"] == d.UNKNOWN

    assert check_by_id(report, "raw_measurement_keys")["verdict"] == "match"
    assert check_by_id(report, "measurement_dtype_conversion")["verdict"] == "match"
    assert check_by_id(report, "current_proprio_source")["observed"]["widths"] == [8]
    assert check_by_id(report, "current_proprio_source")["verdict"] == "match"
    assert check_by_id(report, "action_supervision_window")["verdict"] == "match"
    assert check_by_id(report, "formal_normalisation_statistics")["verdict"] == "not_run"
    assert check_by_id(report, "executed_action_window")["verdict"] == "declared_only"
    assert check_by_id(report, "executed_action_window")["observed"]["npz_field"] is None
    assert check_by_id(report, "camera_order")["verdict"] == "match"
    assert check_by_id(report, "measurement_timestamp")["verdict"] == "not_declared"
    assert check_by_id(report, "noise_availability")["verdict"] == "not_declared"
    assert check_by_id(report, "language_keys")["verdict"] == "match"
    assert check_by_id(report, "action_units")["verdict"] == "match"
    assert report["checks_summary"]["match"] >= 7
    assert "mismatch" not in report["checks_summary"]


def test_audit_robotwin_fixture_flags_the_formal_statistics_gap(tmp_path, robotwin_writer) -> None:
    root = tmp_path / "robotwin"
    record = robotwin_writer(root, ROBOTWIN_TASKS[0], 18, num_frames=60)
    report = d.audit_dataset_root(root, "robotwin", evidence_kind="contract_fixture")
    assert report["status"] == "incomplete"
    assert report["real_cache_evidence"] is False

    entry = report["files"][0]
    assert entry["measurements"]["head_camera_raw_float32"]["dtype"] == "float32"
    assert entry["measurements"]["head_camera_raw_float32"]["resolution_hw"] == [8, 8]
    assert entry["state_source_width"] == 14
    assert entry["actions"]["columns"] == 14
    assert entry["proprio"]["robot_states"]["shape"] == [60, 16]

    raw_keys = check_by_id(report, "raw_measurement_keys")
    assert raw_keys["verdict"] == "match"
    assert raw_keys["observed"]["present_files"] == {
        "head_camera_raw_float32": 1,
        "left_camera_raw_float32": 1,
        "right_camera_raw_float32": 1,
    }
    conversion = check_by_id(report, "measurement_dtype_conversion")
    assert conversion["verdict"] == "match"
    assert conversion["observed"]["dtypes"]["head_camera_raw_float32"] == ["float32"]
    stats_check = check_by_id(report, "formal_normalisation_statistics")
    assert stats_check["verdict"] == "not_run"
    assert stats_check["observed"]["loader_default"] == "starvla_minmax"
    assert "checkpoint" in stats_check["note"]
    supervision = check_by_id(report, "action_supervision_window")
    assert supervision["verdict"] == "match"
    assert supervision["observed"]["loader_default"] == {"action_offset": 1, "action_horizon": 50}
    assert check_by_id(report, "executed_action_window")["expected"]["actions_executed"] == 50
    assert record["actions"].shape[0] >= 51


def test_audit_reads_exactly_max_files_and_reports_the_limit(tmp_path, libero_writer) -> None:
    root = tmp_path / "libero"
    for episode in (18, 19, 20):
        libero_writer(root, LIBERO_SUITE_A, episode, 0, num_frames=12)
    report = d.audit_dataset_root(root, "libero", max_files=2, evidence_kind="contract_fixture")
    assert report["status"] == "incomplete"
    assert report["files_total"] == 3
    assert report["files_checked"] == 2
    assert report["truncated"] is True
    assert len(report["files"]) == 2
    assert any("max_files=2" in line for line in report["limits"])
    full = d.audit_dataset_root(root, "libero", evidence_kind="contract_fixture")
    assert full["truncated"] is False
    assert full["files_checked"] == 3
    with pytest.raises(ValueError):
        d.audit_dataset_root(root, "libero", max_files=0)


def test_audit_fixture_numbers_come_from_the_file_and_are_stable(tmp_path, libero_writer) -> None:
    root = tmp_path / "libero"
    record = libero_writer(root, LIBERO_SUITE_A, 18, 0, num_frames=16)
    first = d.audit_dataset_root(root, "libero", evidence_kind="contract_fixture")
    second = d.audit_dataset_root(root, "libero", evidence_kind="contract_fixture")
    assert first["files"][0]["sha256"] == second["files"][0]["sha256"]
    assert (
        first["files"][0]["actions"]["normalisation_stats_sha256"]
        == second["files"][0]["actions"]["normalisation_stats_sha256"]
    )
    # The hash is a fingerprint of the file's own per-column statistics.
    stats = first["files"][0]["actions"]["normalisation_stats"]
    assert stats["rows"] == 16
    assert stats["columns"] == 7
    assert np.allclose(stats["min"], record["actions"].min(axis=0), atol=0.0)
    assert np.allclose(stats["max"], record["actions"].max(axis=0), atol=0.0)
    assert stats["q50"] == pytest.approx(
        np.quantile(record["actions"].astype(np.float64), 0.5, axis=0).tolist()
    )
    # A different action block gives a different fingerprint.
    other = tmp_path / "other"
    changed = record["actions"].copy()
    changed[:, 0] += 10.0
    libero_writer(other, LIBERO_SUITE_A, 18, 0, num_frames=16, actions=changed)
    third = d.audit_dataset_root(other, "libero", evidence_kind="contract_fixture")
    assert (
        third["files"][0]["actions"]["normalisation_stats_sha256"]
        != first["files"][0]["actions"]["normalisation_stats_sha256"]
    )


def test_audit_marks_uncertainty_instead_of_guessing(tmp_path, libero_writer) -> None:
    """Fixture evidence must stay marked as fixture evidence, never as the real cache."""
    root = tmp_path / "libero"
    libero_writer(root, LIBERO_SUITE_A, 18, 0, num_frames=12)
    fixture = d.audit_dataset_root(root, "libero", evidence_kind="contract_fixture")
    assert fixture["real_cache_evidence"] is False
    assert fixture["evidence_kind"] == "contract_fixture"
    assert any("declared by the caller" in note for note in fixture["notes"])
    # Even when the caller declares a real cache, the recorded evidence kind is echoed verbatim
    # instead of being rewritten into a claim.
    declared = d.audit_dataset_root(root, "libero", evidence_kind="real_cache")
    assert declared["evidence_kind"] == "real_cache"
    assert declared["real_cache_evidence"] is True
    assert declared["files"] == fixture["files"]
    # Unknown formats are refused rather than audited against a guessed schema.
    with pytest.raises(ValueError):
        d.audit_dataset_root(root, "not_a_format")


def test_audit_reports_missing_keys_instead_of_filling_them(tmp_path, libero_writer) -> None:
    root = tmp_path / "libero"
    directory = root / "data/libero/raw" / LIBERO_SUITE_A / "episode_000018"
    directory.mkdir(parents=True)
    payload = {
        "agentview_raw_uint8": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "actions": np.zeros((4, 7), dtype=np.float32),
        "robot_states": np.zeros((4, 8), dtype=np.float32),
        "metadata_json": np.array(json.dumps({"task_description": "partial fixture"})),
    }
    np.savez(directory / "variant_00.npz", **payload)
    report = d.audit_dataset_root(root, "libero", evidence_kind="contract_fixture")
    assert report["status"] == "fail"
    entry = report["files"][0]
    assert entry["measurements"]["wrist_raw_uint8"] == {"present": False}
    assert "wrist_raw_uint8" in entry["missing_required_keys"]
    assert check_by_id(report, "raw_measurement_keys")["verdict"] == "mismatch"
    assert any("wrist_raw_uint8" in problem for problem in report["problems"])

"""S0 合同审计套件测试（工作流 D）。

测试覆盖审计工具的参照核、冻结门槛一致性与报告行为；合同本身的通过/失败结论由
``contracts.json`` 记录，不在测试里硬编码成「必须通过」。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from owac.evaluation import n04_contracts as contracts

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/n04/full/s0_data.json"
PROTOCOL = REPO / "configs/n04/full/protocol.json"


def context(manifest_path: str | None = None) -> contracts.RunContext:
    return contracts.RunContext(
        config=json.loads(CONFIG.read_text(encoding="utf-8")),
        config_path=CONFIG,
        protocol=json.loads(PROTOCOL.read_text(encoding="utf-8")),
        protocol_path=PROTOCOL,
        seed=20261014,
        manifest_path=manifest_path,
        threads=4,
    )


def test_reference_projection_kernel_matches_loop_implementation():
    result = contracts._c1_reference_selftest(context())
    assert result["status"] == "passed"
    assert result["evidence"]["max_abs_diff_vec_vs_loop"] <= 1e-12
    assert result["evidence"]["max_abs_diff_permutation"] <= 1e-12


def test_rotation_hand_cases_and_composition_identities():
    torch_module, geometry = contracts._c2_torch()
    result = contracts._c2_rotations(torch_module, geometry)
    assert result["status"] == "passed"
    evidence = result["evidence"]
    assert evidence["max_hand_error_px"] <= 1e-9
    assert evidence["noncommutativity_max_abs"] > 1e-6
    assert evidence["inverse_order_error"] == 0.0
    assert all(case["in_front"] for case in evidence["cases"])


def test_cfa_phase_law_matches_bayer_definition():
    result = contracts._c2_cfa(torch)
    assert result["status"] == "passed"
    assert result["evidence"]["roundtrip_max_err"] == 0.0
    assert all(row["match"] for row in result["evidence"]["phase_law"])


def test_measurement_geometry_module_checks_pass():
    results = {item["id"]: item for item in contracts._c2_geometry_module(torch)}
    assert set(results) == {"C2.5", "C2.6"}
    assert all(item["status"] == "passed" for item in results.values())
    assert results["C2.5"]["evidence"]["crop_then_resize_max_err_px"] <= 1e-9
    assert results["C2.6"]["evidence"]["finite_difference_max_err"] <= 1e-6
    assert results["C2.6"]["evidence"]["feature_intrinsics_expected"] == [10.0, 3.5]


def test_protocol_frozen_and_consistent_with_config():
    result = contracts._check_c0(context())
    assert result["status"] == "passed"
    by_id = {item["id"]: item for item in result["subchecks"]}
    assert by_id["C0.1"]["evidence"]["frozen"] is True
    assert by_id["C0.4"]["evidence"]["diagnostic_reference_labeled"] is True
    recomputed = by_id["C0.4"]["evidence"]["recomputed_m"]
    assert recomputed["inherited_original_m"] == pytest.approx(9.375e-5, rel=1e-12)
    assert recomputed["resolution_original_m"] == pytest.approx(1.875e-3, rel=1e-12)
    assert recomputed["feature_scale_m"] == pytest.approx(0.06, rel=1e-12)


def test_split_reference_kernel_is_deterministic_and_disjoint():
    result = contracts._c7_split_kernel(context())
    assert result["status"] == "passed"
    evidence = result["evidence"]
    assert evidence["pairwise_intersection_sizes"] == {
        "train_dev": 0,
        "train_final": 0,
        "dev_final": 0,
    }
    assert evidence["counts"] == {"train": 80, "dev": 10, "final": 10}


def test_telemetry_loader_binding_is_a_real_call():
    result = contracts._c5_telemetry_context({"context_at": True, "executed_between": True})
    assert result["status"] == "passed"
    binding = result["evidence"]["loader_binding"]
    assert binding["sequence_context_type"] == "OnlineContext"
    assert binding["loader_context_type"] == "OnlineContext"
    assert binding["context_matches_sidecar"] is True
    assert binding["sequence_interval_type"] == "ExecutedInterval"
    assert binding["interval_controls"] == ["c1", "c2"]


def test_nan_unobserved_status_matches_reported_evidence():
    result = contracts._check_c3(context())
    by_id = {item["id"]: item for item in result["subchecks"]}
    nan = by_id["C3.1"]["evidence"]
    expected = (
        nan["all_fields_finite"]
        and nan["equal_to_clean_masked_run"]
        and nan["param_grads_nonfinite"] == 0
    )
    assert by_id["C3.1"]["status"] == ("passed" if expected else "failed")
    assert by_id["C3.2"]["status"] == "passed"


def test_manifest_verifier_detects_overlap(tmp_path):
    overlapping = {
        "splits": {
            "train": ["a", "b", "c"],
            "dev": ["c", "d"],
            "final": ["e"],
        }
    }
    path = tmp_path / "manifest_overlap.json"
    path.write_text(json.dumps(overlapping), encoding="utf-8")
    result = contracts._verify_manifest_file(context(str(path)))
    assert result["status"] == "failed"
    assert result["evidence"]["overlap"]["dev|train"] == 1

    disjoint = {"train": ["a", "b"], "dev": ["c"], "final": ["d"]}
    path = tmp_path / "manifest_disjoint.json"
    path.write_text(json.dumps(disjoint), encoding="utf-8")
    result = contracts._verify_manifest_file(context(str(path)))
    assert result["status"] == "passed"
    assert result["evidence"]["members_total"] == 4

    unknown = {"members": [{"id": "a"}]}
    path = tmp_path / "manifest_unknown.json"
    path.write_text(json.dumps(unknown), encoding="utf-8")
    result = contracts._verify_manifest_file(context(str(path)))
    assert result["status"] == "skipped"


def _tiny_encoder():
    from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
    from owac.frameworks.n04.model import N04Encoder

    torch.manual_seed(99)
    return N04Encoder(
        N04Config(
            measurement_size=(16, 16),
            camera_ids=("head",),
            geometry=N04GeometryConfig(feature_size=(16, 16), pixel_stride=1.0),
            model=N04ModelConfig(
                width=8,
                hidden=8,
                blocks=1,
                scales=1,
                scale_strides=(1,),
                readout_tokens=4,
                readout_dim=16,
                predictor="identity",
            ),
        )
    )


def test_encoder_step_sanitizes_nan_and_huge_unobserved():
    """C3 回归：毒化未观测区后后验/协方差有限、梯度 18/18 有限且与干净运行逐位一致。"""
    from owac.frameworks.n04.ego import EgoMotionContext

    encoder = _tiny_encoder()
    context = EgoMotionContext(
        proprio_delta=torch.zeros(1, 4),
        proprio_delta_available=False,
        executed_action=torch.zeros(1, 3),
        executed_action_available=True,
        dt=torch.ones(1),
        dt_available=False,
        expected_width=8,
        camera_id="head",
    )
    mask = torch.ones(1, 1, 16, 16)
    mask[:, :, :8, :] = 0.0
    base = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(4))
    reference = contracts._step_with_grads(torch, encoder, base * mask, context, mask)
    for poison in (float("nan"), 1e30):
        poisoned = torch.where(mask > 0, base, torch.full_like(base, poison))
        outcome = contracts._step_with_grads(torch, encoder, poisoned, context, mask)
        for key in ("stable", "change", "uncertainty", "representation"):
            assert torch.isfinite(outcome[key]).all(), (poison, key)
            assert torch.equal(outcome[key], reference[key]), (poison, key)
        finite_grads = [name for name, grad in outcome["grads"].items() if grad is not None]
        assert len(finite_grads) == 18
        assert all(torch.isfinite(outcome["grads"][name]).all() for name in finite_grads)
        for name in finite_grads:
            assert torch.equal(outcome["grads"][name], reference["grads"][name]), (poison, name)
    sanitized = torch.where(
        mask > 0,
        torch.where(mask > 0, base, torch.full_like(base, float("nan"))),
        torch.zeros_like(base),
    )
    outcome = contracts._step_with_grads(torch, encoder, sanitized, context, mask)
    assert torch.equal(outcome["stable"], reference["stable"])


def test_cli_writes_report_with_all_checks(tmp_path):
    output = tmp_path / "contracts.json"
    exit_code = contracts.main(
        [
            "--config",
            str(CONFIG),
            "--protocol",
            str(PROTOCOL),
            "--output",
            str(output),
            "--threads",
            "2",
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == contracts.SCHEMA
    ids = [check["id"] for check in report["checks"]]
    assert ids == ["C0"] + [check_id for check_id, _ in contracts.REQUIRED_CHECKS]
    for check in report["checks"]:
        assert check["status"] in contracts.STATUSES
        for sub in check["subchecks"]:
            assert sub["status"] in contracts.STATUSES
            if sub["status"] == "skipped":
                assert sub["reason"], f"{sub['id']} skipped 但没有原因"
            assert sub["evidence"] is not None
    summary = report["summary"]
    assert summary["total"] == len(report["checks"])
    assert (
        summary["passed"] + summary["failed"] + summary["skipped"] + summary["partial"]
        == summary["total"]
    )
    assert summary["required_total"] == 7
    assert report["availability_policy"]["policy"].startswith("缺失项记 skipped")
    assert report["source"]["protocol"]["sha256"] == contracts._sha256_file(PROTOCOL)
    assert report["source"]["config"]["sha256"] == contracts._sha256_file(CONFIG)
    dependencies = report["source"]["dependencies"]
    assert dependencies["owac.frameworks.n04.measurement_geometry"]["sha256"]
    assert dependencies["owac.raw.telemetry"]["importable"] is True

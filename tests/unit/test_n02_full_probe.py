"""Behaviour tests for the N02 full-development runtime probe.

CPU cases run in the root ``.venv`` (CPU torch) and assert that a CUDA field that
was not measured is ``null`` with a stated reason and never ``0``, and that the
reported errors come from the declared measurement instead of from constants.
Cases that need a real GPU are marked ``gpu`` and skip when no device is present.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from owac.runtime import n02_full_probe

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
TOP_LEVEL_KEYS = {
    "schema_version",
    "probe",
    "generated_utc",
    "requested_device",
    "resolved_device",
    "python",
    "platform",
    "packages",
    "cuda",
    "tf32",
    "operators",
    "unavailable",
}
OPERATOR_KEYS = {"matmul_fp32", "matmul_bf16", "conv2d_fp32", "conv2d_bf16"}


def run_cli(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    output = tmp_path / "probe.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC_DIR)
    completed = subprocess.run(
        [sys.executable, "-m", "owac.runtime.n02_full_probe", "--output", str(output), *args],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
        check=False,
    )
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    return completed, report


def import_torch() -> Any:
    return pytest.importorskip("torch")


def require_cuda() -> Any:
    torch = import_torch()
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device in this interpreter")
    return torch


def test_module_import_stays_lightweight(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC_DIR)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, owac.runtime.n02_full_probe as p; "
            "print('torch' in sys.modules, p.PROBE_SCHEMA_VERSION)",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
        check=True,
    )
    assert completed.stdout.split() == ["False", str(n02_full_probe.PROBE_SCHEMA_VERSION)]


def test_cli_cpu_report_schema(tmp_path: Path) -> None:
    completed, report = run_cli(tmp_path, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr
    assert set(report) >= TOP_LEVEL_KEYS
    assert report["schema_version"] == n02_full_probe.PROBE_SCHEMA_VERSION
    assert report["probe"] == n02_full_probe.PROBE_NAME
    assert report["requested_device"] == "cpu"
    assert report["resolved_device"] == "cpu"
    assert report["python"]["executable"] == sys.executable
    assert report["python"]["version"] == ".".join(str(part) for part in sys.version_info[:3])
    assert isinstance(report["packages"]["torch"]["version"], str)
    assert report["packages"]["torch"]["version"]
    assert set(report["operators"]) == OPERATOR_KEYS
    assert "wrote" in completed.stdout


def test_cpu_cuda_fields_are_null_and_annotated(tmp_path: Path) -> None:
    torch = import_torch()
    if torch.cuda.is_available():
        pytest.skip("this interpreter sees a CUDA device")
    report = n02_full_probe.run_probe("cpu")
    assert report["cuda"] is None
    assert "cuda" in report["unavailable"]
    assert "cuda" in report["unavailable"]["cuda"].lower()
    assert report["tf32"] is None
    assert "tf32" in report["unavailable"]
    for label, case in report["operators"].items():
        assert case["device"] == "cpu", label
        assert not case["device"].startswith("cuda")
    assert report["packages"]["torch"]["cuda_version"] is None
    assert "packages.torch.cuda_version" in report["unavailable"]


def test_cli_refuses_cuda_without_a_device(tmp_path: Path) -> None:
    torch = import_torch()
    if torch.cuda.is_available():
        pytest.skip("this interpreter sees a CUDA device")
    completed, report = run_cli(tmp_path, "--device", "cuda")
    assert completed.returncode == 2
    assert "cuda" in completed.stderr.lower()
    assert report == {}


def test_measured_errors_come_from_the_declared_measurement(tmp_path: Path) -> None:
    torch = import_torch()
    report = n02_full_probe.run_probe("cpu", seed=n02_full_probe.DEFAULT_SEED)
    case = report["operators"]["matmul_fp32"]
    if case["status"] != "measured":
        pytest.fail(f"float32 matmul is not measurable on this CPU runtime: {case['reason']}")
    assert case["max_abs_error_vs_fp64"] is not None
    assert 0.0 < case["max_abs_error_vs_fp64"] < 1e-3
    assert case["mean_abs_error_vs_fp64"] is not None
    assert 0.0 < case["mean_abs_error_vs_fp64"] <= case["max_abs_error_vs_fp64"]
    assert case["elapsed_ms"] is not None and case["elapsed_ms"] > 0.0

    shapes = case["input_recipe"]["shapes"]
    seed = case["input_recipe"]["seed"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lhs = torch.randn(shapes["lhs"][0], shapes["lhs"][1], generator=generator, dtype=torch.float32)
    rhs = torch.randn(shapes["rhs"][0], shapes["rhs"][1], generator=generator, dtype=torch.float32)
    product = lhs @ rhs
    independent = float((product.double() - (lhs.double() @ rhs.double())).abs().max().item())
    assert case["max_abs_error_vs_fp64"] == pytest.approx(independent, rel=0.05)
    fingerprint = case["output_fingerprint"]
    payload = product.contiguous().numpy().tobytes()
    assert fingerprint["sha256"] == hashlib.sha256(payload).hexdigest()
    assert fingerprint["numel"] == product.numel()


def test_seed_changes_the_reported_numbers() -> None:
    import_torch()
    first = n02_full_probe.run_probe("cpu", seed=1103)
    second = n02_full_probe.run_probe("cpu", seed=2207)
    first_case = first["operators"]["matmul_fp32"]
    second_case = second["operators"]["matmul_fp32"]
    assert first_case["status"] == second_case["status"] == "measured"
    assert first_case["input_recipe"]["seed"] != second_case["input_recipe"]["seed"]
    assert first_case["output_fingerprint"]["sha256"] != second_case["output_fingerprint"]["sha256"]
    assert first_case["max_abs_error_vs_fp64"] != second_case["max_abs_error_vs_fp64"]


def test_bfloat16_is_less_accurate_than_float32() -> None:
    import_torch()
    report = n02_full_probe.run_probe("cpu")
    compared = 0
    for kind in ("matmul", "conv2d"):
        low = report["operators"][f"{kind}_bf16"]
        high = report["operators"][f"{kind}_fp32"]
        if low["status"] != "measured" or high["status"] != "measured":
            continue
        assert low["max_abs_error_vs_fp64"] > high["max_abs_error_vs_fp64"] * 10.0
        compared += 1
    assert compared >= 1, "no dtype pair was measurable on this CPU runtime"


@pytest.mark.gpu
def test_gpu_report_matches_torch_device_facts(tmp_path: Path) -> None:
    torch = require_cuda()
    completed, report = run_cli(tmp_path, "--device", "cuda")
    assert completed.returncode == 0, completed.stderr
    cuda = report["cuda"]
    assert cuda is not None
    assert "cuda" not in report["unavailable"]
    assert cuda["available"] is True
    assert cuda["device_count"] == torch.cuda.device_count()
    device = cuda["devices"][cuda["current_device"]]
    properties = torch.cuda.get_device_properties(cuda["current_device"])
    assert device["name"] == properties.name
    assert device["total_memory_bytes"] == properties.total_memory
    assert device["capability"] == list(torch.cuda.get_device_capability(cuda["current_device"]))
    assert device["free_memory_bytes_at_probe"] > 0
    assert device["free_memory_bytes_at_probe"] <= device["total_memory_bytes"]
    assert report["packages"]["torch"]["cuda_version"] is not None
    assert report["tf32"] is not None
    assert report["operators"]["matmul_fp32"]["status"] == "measured"
    assert report["operators"]["conv2d_fp32"]["status"] == "measured"
    assert report["operators"]["matmul_fp32"]["device"].startswith("cuda")

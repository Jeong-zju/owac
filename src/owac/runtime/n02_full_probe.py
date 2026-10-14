"""Measure the runtime that actually executes this file.

The probe reports only facts measured by the running interpreter: package
versions, CUDA device facts, and the accuracy of one matmul and one conv2d per
dtype against a float64 reference computed on the CPU.  A field that could not
be measured is ``null`` and appears in the ``unavailable`` mapping with a
reason; a failed measurement is never reported as ``0``.

Driver version and ``nvidia-smi`` process state are deliberately out of scope:
they belong to ``outputs/n02-full/p0/runtime/resources.json``.

``torch`` is imported inside :func:`run_probe`, so importing this module (or
``owac``) never loads a deep-learning stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROBE_SCHEMA_VERSION = 1
PROBE_NAME = "owac-n02-full-runtime-probe"
DEFAULT_SEED = 1103

# Deterministic and small enough to stay a probe, but large enough that a wrong
# kernel, a silent dtype fallback, or a missing GPU changes the measured error.
# Draw order for the shared CPU generator is documented in _build_operands.
MATMUL_RECIPE = {"m": 256, "k": 256, "n": 256}
CONV2D_RECIPE = {
    "batch": 2,
    "in_channels": 4,
    "height": 48,
    "width": 64,
    "out_channels": 8,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
}

DTYPES = (("float32", "float32", "fp32"), ("bfloat16", "bfloat16", "bf16"))


class ProbeError(RuntimeError):
    """The requested measurement cannot be taken in this interpreter."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _python_report() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "in_virtualenv": sys.prefix != sys.base_prefix,
    }


def _platform_report() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _torch_report(torch_module: Any, unavailable: dict[str, str]) -> dict[str, Any]:
    report = {
        "version": torch_module.__version__,
        "git_version": getattr(torch_module.version, "git_version", None),
        "debug_build": bool(torch_module.version.debug),
        "cuda_version": torch_module.version.cuda,
        "num_threads": int(torch_module.get_num_threads()),
        "num_interop_threads": int(torch_module.get_num_interop_threads()),
    }
    if report["git_version"] is None:
        unavailable["packages.torch.git_version"] = "torch.version.git_version is None"
    if report["cuda_version"] is None:
        unavailable["packages.torch.cuda_version"] = (
            "torch.version.cuda is None: this torch build carries no CUDA support"
        )
    return report


def _torchvision_report(unavailable: dict[str, str]) -> dict[str, Any] | None:
    try:
        import torchvision
    except Exception as exc:  # noqa: BLE001 - the reason is the measurement
        unavailable["packages.torchvision"] = f"import torchvision failed: {exc!r}"
        return None
    return {
        "version": torchvision.__version__,
        "cuda_version": getattr(torchvision.version, "cuda", None),
    }


def _resolve_device(torch_module: Any, requested: str) -> tuple[str, int | None]:
    if requested == "cpu":
        return "cpu", None
    available = bool(torch_module.cuda.is_available())
    if requested == "cuda" and not available:
        raise ProbeError(
            f"requested --device cuda but torch.cuda.is_available() is false in {sys.executable}"
        )
    if available:
        return "cuda", int(torch_module.cuda.current_device())
    return "cpu", None


def _cuda_report(torch_module: Any, unavailable: dict[str, str]) -> dict[str, Any] | None:
    """Measure CUDA facts, or record why the whole CUDA section is unmeasurable."""
    if not bool(torch_module.cuda.is_available()):
        unavailable["cuda"] = (
            "torch.cuda.is_available() is false in this interpreter: no CUDA device "
            "fields were measured (they are null, not 0)"
        )
        return None

    try:
        cudnn_version = torch_module.backends.cudnn.version()
    except Exception as exc:  # noqa: BLE001 - the reason is the measurement
        cudnn_version = None
        unavailable["cuda.cudnn_version"] = f"torch.backends.cudnn.version() failed: {exc!r}"

    devices = []
    for index in range(int(torch_module.cuda.device_count())):
        properties = torch_module.cuda.get_device_properties(index)
        free_bytes, total_bytes = (int(value) for value in torch_module.cuda.mem_get_info(index))
        major, minor = torch_module.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "capability": [int(major), int(minor)],
                "sm": f"sm_{int(major)}{int(minor)}",
                "total_memory_bytes": int(properties.total_memory),
                "total_memory_bytes_from_mem_get_info": total_bytes,
                "free_memory_bytes_at_probe": free_bytes,
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )

    return {
        "available": True,
        "device_count": int(torch_module.cuda.device_count()),
        "current_device": int(torch_module.cuda.current_device()),
        "cudnn_version": cudnn_version,
        "cudnn_enabled": bool(torch_module.backends.cudnn.enabled),
        "bf16_supported": bool(torch_module.cuda.is_bf16_supported()),
        "memory_allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "devices": devices,
    }


def _tf32_report(
    torch_module: Any, unavailable: dict[str, str], resolved_device: str
) -> dict[str, Any] | None:
    """Report the global TF32 flags once, before the operator smoke tests run."""
    if resolved_device != "cuda":
        unavailable["tf32"] = (
            "no CUDA device in this run: TF32 flags are not applicable to CPU kernels "
            "(they are null, not false)"
        )
        return None
    report = {
        "matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "matmul_fp32_precision": _fp32_precision(torch_module.backends.cuda.matmul),
        "cudnn_conv_fp32_precision": _fp32_precision(torch_module.backends.cudnn.conv),
        "note": "flags read before the smoke tests; the probe does not modify them",
    }
    for key in ("matmul_fp32_precision", "cudnn_conv_fp32_precision"):
        if report[key] is None:
            unavailable[f"tf32.{key}"] = "not exposed by this torch version"
    return report


def _to_device(tensor: Any, device: str, index: int | None, dtype: Any) -> Any:
    if device == "cuda":
        return tensor.to(device=f"cuda:{index}", dtype=dtype)
    return tensor.to(dtype=dtype)


def _build_operands(
    torch_module: Any, kind: str, requested: str, device: str, index: int | None, seed: int
) -> tuple[dict[str, Any], Any, Any]:
    """Rebuild one deterministic operand set, its float64 reference, and the runner.

    Draw order (one CPU generator seeded with ``seed``): matmul draws lhs then rhs;
    conv2d draws input, weight then bias.  Operands are always drawn as float32 and
    cast to the requested dtype, so the float64 reference is exact for the data and
    every reported error is end-to-end (cast + kernel + accumulator) rounding.
    """
    generator = torch_module.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch_module, requested)
    if kind == "matmul":
        m, k, n = MATMUL_RECIPE["m"], MATMUL_RECIPE["k"], MATMUL_RECIPE["n"]
        lhs = torch_module.randn(m, k, generator=generator, dtype=torch_module.float32)
        rhs = torch_module.randn(k, n, generator=generator, dtype=torch_module.float32)
        dev_lhs = _to_device(lhs, device, index, dtype)
        dev_rhs = _to_device(rhs, device, index, dtype)
        reference = lhs.to(torch_module.float64) @ rhs.to(torch_module.float64)
        shapes = {"lhs": [m, k], "rhs": [k, n]}
        return shapes, reference, (lambda: dev_lhs @ dev_rhs)
    recipe = CONV2D_RECIPE
    inputs = torch_module.randn(
        recipe["batch"],
        recipe["in_channels"],
        recipe["height"],
        recipe["width"],
        generator=generator,
        dtype=torch_module.float32,
    )
    weight = torch_module.randn(
        recipe["out_channels"],
        recipe["in_channels"],
        recipe["kernel_size"],
        recipe["kernel_size"],
        generator=generator,
        dtype=torch_module.float32,
    )
    bias = torch_module.randn(
        recipe["out_channels"], generator=generator, dtype=torch_module.float32
    )
    dev_inputs = _to_device(inputs, device, index, dtype)
    dev_weight = _to_device(weight, device, index, dtype)
    dev_bias = _to_device(bias, device, index, dtype)
    reference = torch_module.nn.functional.conv2d(
        inputs.to(torch_module.float64),
        weight.to(torch_module.float64),
        bias.to(torch_module.float64),
        stride=recipe["stride"],
        padding=recipe["padding"],
    )
    shapes = {
        "input": [recipe["batch"], recipe["in_channels"], recipe["height"], recipe["width"]],
        "weight": [
            recipe["out_channels"],
            recipe["in_channels"],
            recipe["kernel_size"],
            recipe["kernel_size"],
        ],
        "bias": [recipe["out_channels"]],
        "stride": recipe["stride"],
        "padding": recipe["padding"],
    }

    def run_conv2d() -> Any:
        return torch_module.nn.functional.conv2d(
            dev_inputs, dev_weight, dev_bias, stride=recipe["stride"], padding=recipe["padding"]
        )

    return shapes, reference, run_conv2d


def _fp32_precision(module: Any) -> Any:
    """Read the torch>=2.9 ``fp32_precision`` knob of one backend namespace."""
    return getattr(module, "fp32_precision", None)


def _fingerprint(torch_module: Any, tensor: Any) -> dict[str, Any]:
    flat = tensor.detach().to(device="cpu", dtype=torch_module.float32).contiguous()
    payload = flat.numpy().tobytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "numel": int(flat.numel()),
        "dtype": "float32",
        "sum": float(flat.to(torch_module.float64).sum().item()),
        "abs_max": float(flat.abs().max().item()),
    }


def _measure_case(
    torch_module: Any,
    kind: str,
    requested: str,
    label: str,
    device: str,
    index: int | None,
    seed: int,
    unavailable: dict[str, str],
) -> dict[str, Any]:
    device_label = device if index is None else f"{device}:{index}"
    case: dict[str, Any] = {
        "op": kind,
        "dtype": requested,
        "device": device_label,
        "status": "unavailable",
        "max_abs_error_vs_fp64": None,
        "mean_abs_error_vs_fp64": None,
        "reference": {
            "device": "cpu",
            "dtype": "float64",
            "kind": "float64 evaluation of the same float32 operands",
        },
        "input_recipe": {
            "seed": seed,
            "draw_dtype": "float32",
            "draw_order": "lhs,rhs" if kind == "matmul" else "input,weight,bias",
        },
        "output_fingerprint": None,
        "elapsed_ms": None,
        "reason": None,
    }
    try:
        shapes, reference, runner = _build_operands(
            torch_module, kind, requested, device, index, seed
        )
        case["input_recipe"]["shapes"] = shapes
        started = time.perf_counter()
        result = runner()
        if device == "cuda":
            torch_module.cuda.synchronize(index)
        case["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        # The result may live on the GPU while the reference is a CPU tensor.
        delta = (result.detach().to(device="cpu", dtype=torch_module.float64) - reference).abs()
        case["max_abs_error_vs_fp64"] = float(delta.max().item())
        case["mean_abs_error_vs_fp64"] = float(delta.mean().item())
        case["output_fingerprint"] = _fingerprint(torch_module, result)
        case["status"] = "measured"
    except Exception as exc:  # noqa: BLE001 - the reason is the measurement
        case["reason"] = f"{type(exc).__name__}: {exc}"
        unavailable[f"operators.{label}"] = case["reason"]
    return case


def _measure(
    torch_module: Any, device: str, index: int | None, seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    unavailable: dict[str, str] = {}
    operators: dict[str, Any] = {}
    for kind in ("matmul", "conv2d"):
        for requested, _, short in DTYPES:
            label = f"{kind}_{short}"
            operators[label] = _measure_case(
                torch_module, kind, requested, label, device, index, seed, unavailable
            )
    return operators, unavailable


def run_probe(device: str = "auto", *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Measure the current interpreter and return the report as plain JSON types."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - the reason is the measurement
        raise ProbeError(f"cannot import torch in {sys.executable}: {exc!r}") from exc

    resolved, index = _resolve_device(torch, device)
    unavailable: dict[str, str] = {}
    operators, operator_unavailable = _measure(torch, resolved, index, seed)
    unavailable.update(operator_unavailable)
    payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe": PROBE_NAME,
        "generated_utc": _utc_now(),
        "generated_epoch_s": time.time(),
        "requested_device": device,
        "resolved_device": resolved if index is None else f"{resolved}:{index}",
        "python": _python_report(),
        "platform": _platform_report(),
        "packages": {
            "torch": _torch_report(torch, unavailable),
            "torchvision": _torchvision_report(unavailable),
        },
        "cuda": _cuda_report(torch, unavailable),
        "tf32": _tf32_report(torch, unavailable, resolved),
        "operators": operators,
        "unavailable": unavailable,
    }
    return payload


def _summary(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"probe={payload['probe']} schema={payload['schema_version']}",
        f"python={payload['python']['version']} executable={payload['python']['executable']}",
        f"torch={payload['packages']['torch']['version']} "
        f"cuda={payload['packages']['torch']['cuda_version']} "
        f"torchvision={payload['packages']['torchvision']}",
        f"resolved_device={payload['resolved_device']}",
    ]
    cuda = payload["cuda"]
    if cuda is None:
        lines.append("cuda=unavailable (see unavailable['cuda'])")
    else:
        device = cuda["devices"][cuda["current_device"]]
        lines.append(
            f"device={device['name']} capability={device['capability']} "
            f"total_mib={device['total_memory_bytes'] / 2**20:.1f} "
            f"free_mib={device['free_memory_bytes_at_probe'] / 2**20:.1f} "
            f"cudnn={cuda['cudnn_version']}"
        )
    for label, case in payload["operators"].items():
        if case["status"] == "measured":
            lines.append(
                f"{label}: max_abs_err_vs_fp64={case['max_abs_error_vs_fp64']:.6g} "
                f"elapsed_ms={case['elapsed_ms']:.3f}"
            )
        else:
            lines.append(f"{label}: unavailable ({case['reason']})")
    lines.append(f"unavailable_entries={len(payload['unavailable'])}")
    return lines


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m owac.runtime.n02_full_probe",
        description=(
            "Measure the running interpreter (versions, CUDA device facts, matmul and "
            "conv2d accuracy) and write the report as JSON."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="operand generator seed")
    parser.add_argument("--print-json", action="store_true", help="also dump the report to stdout")
    args = parser.parse_args(argv)

    try:
        payload = run_probe(args.device, seed=args.seed)
    except ProbeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 2

    _write_json(args.output, payload)
    for line in _summary(payload):
        print(line)
    print(f"wrote {args.output}")
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

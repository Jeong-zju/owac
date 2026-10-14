"""Execute pinned official sparse FCMAE encoder with the custom Minkowski kernel.

The engine has a disclosed C++ ownership compatibility patch; numerical source
files and ConvNeXt sources are unchanged. Never use import placeholders here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from .fcmae import MaskAwareDenseConvNeXtV2Encoder


def run(engine_root: Path, output: Path) -> dict:
    engine_root = engine_root.resolve()
    sys.path.insert(0, str(engine_root))
    sys.path.insert(0, str(engine_root / "MinkowskiEngine"))
    upstream = Path(__file__).parent / "upstream" / "convnextv2"
    sys.path.insert(0, str(upstream.resolve()))
    import MinkowskiEngine as me
    from models.convnextv2_sparse import SparseConvNeXtV2

    assert Path(me.__file__).resolve().is_relative_to(engine_root)
    torch.manual_seed(2207)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dims, depths = (8, 16, 32, 64), (1, 1, 1, 1)
    port = MaskAwareDenseConvNeXtV2Encoder(in_chans=3, depths=depths, dims=dims).cuda().eval()
    ref = SparseConvNeXtV2(in_chans=3, depths=depths, dims=dims, D=2).cuda().eval()
    # Nonzero affine terms expose support leakage and GRN normalization errors.
    with torch.no_grad():
        for name, value in port.named_parameters():
            if "grn" in name:
                value.uniform_(-0.2, 0.2)
    port_params = dict(port.named_parameters())
    ref_params = dict(ref.named_parameters())
    mappings = {}
    for name, value in port_params.items():
        target = name
        layout = "identity"
        if name.startswith("stages"):
            target = target.replace(".norm.", ".norm.ln.")
            target = target.replace(".pwconv1.", ".pwconv1.linear.")
            target = target.replace(".pwconv2.", ".pwconv2.linear.")
            if name.endswith("dwconv.weight"):
                target = target.replace(".weight", ".kernel")
                layout = "depthwise"
        elif name.startswith("downsample_layers") and not name.startswith("downsample_layers.0"):
            if name.split(".")[2] == "0":
                target = target.replace(".0.", ".0.ln.")
            elif name.endswith("weight"):
                target = target.replace(".weight", ".kernel")
                layout = "convolution"
        mappings[name] = (target, layout)
        with torch.no_grad():
            ref_params[target].copy_(to_sparse_layout(value, layout).reshape_as(ref_params[target]))
    assert len(mappings) == len(ref_params)
    actual, expected, supports = {}, {}, {}
    hooks = []

    def record_ref(name, size):
        def hook(module, args, value):
            coords = value.C.long().to(value.F.device)
            indices = coords[:, 1:] // torch.tensor(value.tensor_stride, device=coords.device)
            dense = value.F.new_zeros(2, value.F.shape[1], size, size)
            valid = value.F.new_zeros(2, 1, size, size)
            dense[coords[:, 0], :, indices[:, 0], indices[:, 1]] = value.F
            valid[coords[:, 0], 0, indices[:, 0], indices[:, 1]] = 1
            expected[name] = dense
            supports[name] = valid

        return hook

    def record_port(name):
        def hook(module, args, value):
            actual[name] = value

        return hook

    for i in range(4):
        name = f"stage{i}"
        hooks.append(ref.stages[i].register_forward_hook(record_ref(name, 16 // 2**i)))
        hooks.append(port.stages[i][-1].register_forward_hook(record_port(name)))
    x = torch.randn(2, 3, 64, 64, device="cuda")
    mask = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], device="cuda", dtype=torch.float32)
    xr, xp = x.clone().requires_grad_(), x.clone().requires_grad_()
    ref(xr, mask)
    port(xp, mask)
    generator = torch.Generator(device="cuda").manual_seed(1103)
    projections = {
        name: torch.randn(value.shape, generator=generator, device="cuda")
        for name, value in expected.items()
    }
    sum((expected[name] * projections[name]).mean() for name in expected).backward()
    sum((actual[name][0] * projections[name]).mean() for name in actual).backward()
    layers = {}
    for name in expected:
        a, b = actual[name][0], expected[name]
        layers[name] = {
            "max_abs": float((a - b).abs().max().detach()),
            "forward_close": torch.allclose(a, b, atol=1e-5, rtol=1e-4),
            "support_identical": torch.equal(actual[name][1], supports[name]),
            "active_positions": int(supports[name].sum().item()),
        }
    gradients = {}
    for name, value in port_params.items():
        target, layout = mappings[name]
        a = to_sparse_layout(value.grad, layout).reshape_as(ref_params[target])
        b = ref_params[target].grad
        gradients[name] = {
            "max_abs": float((a - b).abs().max()),
            "close": torch.allclose(a, b, atol=1e-5, rtol=1e-3),
        }
    input_close = torch.allclose(xp.grad, xr.grad, atol=1e-5, rtol=1e-3)
    ok = (
        input_close
        and all(v["close"] for v in gradients.values())
        and all(v["forward_close"] and v["support_identical"] for v in layers.values())
    )
    hashes = {
        name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
        for name in ["models/convnextv2_sparse.py", "models/utils.py"]
    }
    report = {
        "status": "pass" if ok else "fail",
        "layers": layers,
        "parameter_gradients": gradients,
        "input_grad_max_abs": float((xp.grad - xr.grad).abs().max()),
        "input_grad_close": input_close,
        "source_hashes": hashes,
        "port_source_hashes": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("fcmae.py", "convnextv2.py")
        },
        "engine_source": me.__file__,
        "torch": torch.__version__,
        "engine_commit": "bbc30ef581ea6deb505976b663f5fc2358a83749",
        "engine_compatibility_patch": (
            "configs/backends/n02-full-runtime/minkowski-cuda128-ownership.patch"
        ),
        "device": torch.cuda.get_device_name(0),
        "dims": dims,
        "depths": depths,
        "mask": mask.cpu().tolist(),
        "scope": ("official sparse encoder versus mask-aware dense; distinct computational cost"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "layers": layers,
                "max_parameter_grad_error": max(v["max_abs"] for v in gradients.values()),
            }
        )
    )
    for hook in hooks:
        hook.remove()
    return report


def to_sparse_layout(value, layout):
    if layout == "depthwise":
        return value.permute(3, 2, 0, 1).reshape(49, -1)
    if layout == "convolution":
        return value.permute(3, 2, 1, 0).reshape(4, value.shape[1], value.shape[0])
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.engine_root, args.output)
    raise SystemExit(0 if result["status"] == "pass" else 1)

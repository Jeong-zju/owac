"""Run inspected upstream fusion classes in isolation; no OWAC integration or training."""

import argparse
import ast
import hashlib
import json
import subprocess
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("cache", type=Path, help="Directory with pinned Git trees and sources/")
root = parser.parse_args().cache.resolve()
sources = root / "sources"
torch.set_num_threads(1)
torch.manual_seed(20261013)


def verify(repo, relative):
    source = sources / repo / relative
    raw = source.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    expected = subprocess.check_output(
        ["git", "-C", str(root / repo), "ls-tree", "HEAD", relative], text=True
    ).split()[2]
    assert blob == expected, (repo, relative, blob, expected)
    return source


def selected(source, names, extra=None):
    tree = ast.parse(source.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in body} == set(names)
    ns = {"torch": torch, "nn": nn, "F": F, **(extra or {})}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), ns)
    return ns


basic = selected(verify("BasicSR", "basicsr/archs/edvr_arch.py"), ["TSAFusion"])
blocks = selected(
    verify("deep-burst-sr", "models/layers/blocks.py"),
    ["get_activation", "get_attention", "conv_block", "ResBlock"],
)
merge = selected(
    verify("deep-burst-sr", "models/dbsr/merging.py"),
    ["WeightedSum"],
    {"blocks": types.SimpleNamespace(**blocks)},
)
records = []
for method in ("TSAFusion", "WeightedSum_mean", "WeightedSum_reference"):
    if method == "TSAFusion":
        model = basic["TSAFusion"](num_feat=8, num_frame=8, center_frame_idx=7)
        lengths = (8,)
    else:
        model = merge["WeightedSum"](
            input_dim=8,
            project_dim=8,
            offset_feat_dim=4,
            use_base_frame=method.endswith("reference"),
        )
        lengths = (1, 4, 8)
    for length in lengths:
        model.zero_grad(set_to_none=True)
        x = torch.randn(2, length, 8, 16, 16, requires_grad=True)
        if method == "TSAFusion":
            out = model(x)
            weight_error = None
        else:
            offsets = torch.zeros(2, length - 1, 2, 16, 16)
            result = model(
                {
                    "ref_feat": x[:, :1].contiguous(),
                    "oth_feat": x[:, 1:].contiguous(),
                    "offsets": offsets,
                }
            )
            out = result["fused_enc"]
            weight_error = float((result["fusion_weights"].sum(1) - 1).abs().max().detach())
            assert weight_error < 1e-6
        assert out.shape == (2, 8, 16, 16) and torch.isfinite(out).all()
        out.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert all(float(g) > 0 for g in x.grad.abs().sum((0, 2, 3, 4)))
        with torch.no_grad():
            if method == "TSAFusion":
                zero_out = model(torch.zeros_like(x))
            else:
                zero_out = model(
                    {
                        "ref_feat": torch.zeros_like(x[:, :1]).contiguous(),
                        "oth_feat": torch.zeros_like(x[:, 1:]).contiguous(),
                        "offsets": offsets,
                    }
                )["fused_enc"]
        records.append(
            {
                "method": method,
                "frames": length,
                "input_shape": list(x.shape),
                "output_shape": list(out.shape),
                "parameters": sum(p.numel() for p in model.parameters()),
                "finite_forward_backward": True,
                "per_frame_input_gradient_l1": x.grad.abs().sum((0, 2, 3, 4)).tolist(),
                "weight_sum_max_error": weight_error,
                "zero_input_output_max_abs": float(zero_out.abs().max()),
            }
        )
report = {
    "scope": (
        "Unmodified selected upstream class/function ASTs, random weights and features, CPU only. "
        "Not full repository import, RAW integration, training, task quality, causality, "
        "or performance benchmark. Inputs are all valid; neither upstream interface accepts "
        "validity or source IDs."
    ),
    "seed": 20261013,
    "torch": torch.__version__,
    "device": "cpu",
    "source_git_blob_hashes_verified": True,
    "results": records,
}
(root / "fusion-probe.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

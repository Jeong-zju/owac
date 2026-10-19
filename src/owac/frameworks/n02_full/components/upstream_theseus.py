"""Run pinned Theseus dense solvers in its reference interpreter (P0 only)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import runpy
from pathlib import Path


def export(output: str, source_root: str) -> dict:
    import theseus as th
    import torch

    root = Path(source_root).resolve()
    if not Path(th.__file__).resolve().is_relative_to(root):
        raise ValueError("Theseus must be imported from the pinned source checkout")
    torch.manual_seed(1103)
    example = root / "examples/simple_example.py"
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        runpy.run_path(str(example), run_name="__main__")
    losses = [
        float(line.split(":")[-1])
        for line in stream.getvalue().splitlines()
        if line.startswith("Outer loss:")
    ]
    a = torch.randn(1, 8, 3, dtype=torch.float64)
    b = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    x = th.Vector(tensor=torch.zeros(1, 3, dtype=torch.float64), name="x")
    av, bv = th.Variable(a, name="a"), th.Variable(b.detach(), name="b")

    def error(optim_vars, aux_vars):
        matrix, rhs = aux_vars
        return (matrix.tensor @ optim_vars[0].tensor.unsqueeze(-1)).squeeze(-1) - rhs.tensor

    objective = th.Objective(dtype=torch.float64)
    objective.add(
        th.AutoDiffCostFunction(
            [x],
            error,
            8,
            aux_vars=[av, bv],
            cost_weight=th.ScaleCostWeight(torch.ones(1, 1, dtype=torch.float64)),
        )
    )
    layer = th.TheseusLayer(
        th.LevenbergMarquardt(
            objective, max_iterations=50, abs_err_tolerance=1e-14, rel_err_tolerance=1e-14
        )
    )

    def solve(rhs):
        return layer(
            {"b": rhs, "x": torch.zeros(1, 3, dtype=torch.float64)},
            optimizer_kwargs={"backward_mode": "implicit", "damping": 1e-6},
        )[0]["x"]

    solution = solve(b)
    reference = torch.linalg.lstsq(a[0], b[0]).solution
    loss = solution.square().sum()
    grad = torch.autograd.grad(loss, b)[0]
    analytic_grad = 2 * torch.linalg.pinv(a[0]).T @ reference
    sweep = []
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        numerical = torch.zeros_like(b)
        for i in range(b.shape[-1]):
            delta = torch.zeros_like(b)
            delta[0, i] = h
            numerical[0, i] = (
                solve(b.detach() + delta).square().sum() - solve(b.detach() - delta).square().sum()
            ) / (2 * h)
        sweep.append({"step": h, "max_abs_error": float((numerical - grad).detach().abs().max())})
    source_files = [
        "examples/simple_example.py",
        "theseus/theseus_layer.py",
        "theseus/optimizer/nonlinear/levenberg_marquardt.py",
    ]
    payload = {
        "source_root": str(root),
        "upstream_commit": "c8583de41824613fb135ee9bad0e930ded6be404",
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files
        },
        "simple_example_losses": losses,
        "solution_max_abs": float((solution[0] - reference).detach().abs().max()),
        "gradient_max_abs": float((grad[0] - analytic_grad).detach().abs().max()),
        "finite_difference_sweep": sweep,
        "torch": torch.__version__,
        "theseus": th.__version__,
        "device": "cpu",
        "dtype": "float64",
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.source_root), indent=2))

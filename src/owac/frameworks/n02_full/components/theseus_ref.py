"""Theseus / Levenberg-Marquardt reference (R09).

Upstream (verbatim copies kept next to this file)::

    components/upstream/theseus/theseus/optimizer/nonlinear/levenberg_marquardt.py
    components/upstream/theseus/theseus/theseus_layer.py
    components/upstream/theseus/theseus/core/cost_function.py
    components/upstream/theseus/examples/simple_example.py
    repository facebookresearch/theseus @ c8583de41824613fb135ee9bad0e930ded6be404

This module contains **a reference implementation written here, not upstream code**.  It encodes
the decision of the P0 contract section 5 and of audit item 6: Theseus LM is only allowed to
propose continuous steps at a *fixed* structure, and the step is accepted by re-evaluating the
complete outer objective -- including the terms LM cannot see (here: a log-variance / saturation
style penalty).  Upstream Theseus optimises a residual sum of squares, so a term such as a
negative log-likelihood with a log-variance must not be squared into that residual.

Two properties are asserted by the parity suite:

* on a linear least-squares problem, the reference LM converges to the analytic minimiser with
  FP64 error below 1e-8, and its gradient agrees with a finite-difference sweep;
* when the outer objective contains a term that the LM residual does not model, steps that would
  reduce the residual sum are rejected whenever they increase the full objective, the outer loop
  never increases the full objective, and it records whether it accepted a proposal or fell back.

The upstream Theseus install attempt and its result live in
``outputs/n02-full/p0/parity/logs/13_theseus_install_attempt.log``; when it fails, the upstream
``examples/simple_example.py`` run is reported as ``not_run`` with that log as evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_REPOSITORY = "facebookresearch/theseus"
UPSTREAM_COMMIT = "c8583de41824613fb135ee9bad0e930ded6be404"


# --------------------------------------------------------------------------------------
# reference LM (written here, not upstream)
# --------------------------------------------------------------------------------------


@dataclass
class LMConfig:
    max_iterations: int = 50
    initial_damping: float = 1.0
    damping_up: float = 10.0
    damping_down: float = 0.1
    step_tolerance: float = 1e-12
    use_marquardt_scaling: bool = True


@dataclass
class LMResult:
    x: np.ndarray
    cost: float
    iterations: int
    accepted_steps: int
    damping: float
    converged: bool
    history: list[float] = field(default_factory=list)


def numerical_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    """Central finite differences of a residual vector (used for validation and fallbacks)."""
    r0 = np.asarray(residual_fn(x), dtype=np.float64)
    jac = np.zeros((r0.size, x.size), dtype=np.float64)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = step
        jac[:, i] = (np.asarray(residual_fn(x + dx)) - np.asarray(residual_fn(x - dx))) / (2 * step)
    return jac


def lm_minimize(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    config: LMConfig | None = None,
    jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> LMResult:
    """Hand-written Levenberg-Marquardt on ``0.5 * ||r(x)||^2`` in float64.

    The damping is added to the (optionally Marquardt-scaled) Gauss-Newton normal equations; a
    step is accepted only if it lowers the residual cost, otherwise the damping is increased.
    This mirrors the upstream ``LevenbergMarquardt`` update rule closely enough for the analytic
    checks in this module and is explicitly *not* claimed to be the upstream implementation.
    """
    cfg = config or LMConfig()
    x = np.array(x0, dtype=np.float64).copy()
    damping = float(cfg.initial_damping)
    r = np.asarray(residual_fn(x), dtype=np.float64)
    cost = 0.5 * float(r @ r)
    accepted = 0
    converged = False
    history = [cost]
    for _ in range(cfg.max_iterations):
        jac = (
            np.asarray(jacobian_fn(x), dtype=np.float64)
            if jacobian_fn is not None
            else numerical_jacobian(residual_fn, x)
        )
        jtj = jac.T @ jac
        g = jac.T @ r
        if cfg.use_marquardt_scaling:
            diag = np.clip(np.diag(jtj), 1e-12, None)
            mat = jtj + damping * np.diag(diag)
        else:
            mat = jtj + damping * np.eye(x.size)
        try:
            step = np.linalg.solve(mat, -g)
        except np.linalg.LinAlgError:
            damping *= cfg.damping_up
            continue
        if float(np.linalg.norm(step)) <= cfg.step_tolerance:
            converged = True
            break
        candidate = x + step
        r_new = np.asarray(residual_fn(candidate), dtype=np.float64)
        cost_new = 0.5 * float(r_new @ r_new)
        if cost_new < cost:
            x, r, cost = candidate, r_new, cost_new
            damping = max(damping * cfg.damping_down, 1e-12)
            accepted += 1
        else:
            damping *= cfg.damping_up
        history.append(cost)
    return LMResult(
        x=x,
        cost=cost,
        iterations=len(history) - 1,
        accepted_steps=accepted,
        damping=damping,
        converged=converged,
        history=history,
    )


# --------------------------------------------------------------------------------------
# proposal + full-objective acceptance
# --------------------------------------------------------------------------------------


@dataclass
class OuterStepRecord:
    iteration: int
    full_objective_before: float
    full_objective_proposed: float
    residual_cost_before: float
    residual_cost_proposed: float
    accepted: bool
    used_fallback: bool
    damping: float


@dataclass
class OuterResult:
    x: np.ndarray
    full_objective: float
    records: list[OuterStepRecord]
    n_accepted: int
    n_rejected: int


def fixed_structure_outer_loop(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    full_objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n_iterations: int = 20,
    damping0: float = 1.0,
    fallback_step: float = 1e-3,
) -> OuterResult:
    """LM proposes a continuous step at fixed structure; the *full* objective accepts it.

    The LM is run for a single Gauss-Newton step to obtain a proposal (structure held fixed).
    The proposal is accepted only if ``full_objective`` strictly decreases.  A rejected proposal
    triggers one backtracking step along the negative *full* objective gradient (finite
    differences), and if that also fails the previous best solution is kept and the iteration is
    recorded as rejected -- so the outer objective never increases.
    """
    x = np.array(x0, dtype=np.float64).copy()
    damping = float(damping0)
    records: list[OuterStepRecord] = []
    n_accepted = n_rejected = 0
    full_before = float(full_objective(x))
    r_before = np.asarray(residual_fn(x), dtype=np.float64)
    residual_cost_before = 0.5 * float(r_before @ r_before)

    for it in range(n_iterations):
        proposal = lm_minimize(residual_fn, x, LMConfig(max_iterations=1, initial_damping=damping))
        damping = proposal.damping
        x_prop = proposal.x
        full_prop = float(full_objective(x_prop))
        r_prop = np.asarray(residual_fn(x_prop), dtype=np.float64)
        residual_cost_prop = 0.5 * float(r_prop @ r_prop)
        used_fallback = False
        accepted = full_prop < full_before
        if accepted:
            x, full_before, residual_cost_before = x_prop, full_prop, residual_cost_prop
            n_accepted += 1
        else:
            # Backtrack along the full objective's own gradient; if that fails, keep the best.
            grad = _full_objective_gradient(full_objective, x)
            step_size = fallback_step
            improved = False
            for _ in range(8):
                trial = x - step_size * grad
                if float(full_objective(trial)) < full_before:
                    x = trial
                    full_before = float(full_objective(trial))
                    residual_cost_before = 0.5 * float(
                        np.asarray(residual_fn(x), dtype=np.float64) @ np.asarray(residual_fn(x))
                    )
                    improved = True
                    break
                step_size *= 0.5
            used_fallback = improved
            full_prop = full_before
            n_rejected += 1
        records.append(
            OuterStepRecord(
                iteration=it,
                full_objective_before=float(records[-1].full_objective_proposed)
                if records
                else float(full_objective(x0)),
                full_objective_proposed=full_prop,
                residual_cost_before=residual_cost_before,
                residual_cost_proposed=residual_cost_prop,
                accepted=accepted,
                used_fallback=used_fallback,
                damping=damping,
            )
        )
    return OuterResult(
        x=x,
        full_objective=full_before,
        records=records,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
    )


def _full_objective_gradient(objective: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    step = 1e-6
    grad = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = step
        grad[i] = (objective(x + dx) - objective(x - dx)) / (2 * step)
    return grad


# --------------------------------------------------------------------------------------
# analytic checks
# --------------------------------------------------------------------------------------


def linear_least_squares_problem(seed: int = 1103) -> dict[str, Any]:
    """Ill-conditioned but well-posed linear system with an analytic minimiser."""
    rng = np.random.default_rng(seed)
    m, n = 20, 5
    a = rng.normal(size=(m, n))
    a[:, -1] = a[:, 0] + 1e-3 * rng.normal(size=m)  # mild conditioning stress
    x_star = rng.normal(size=n)
    b = a @ x_star
    return {"a": a, "b": b, "x_star": x_star, "x_generating": x_star}


def run_lm_reference_check(out_dir: str | Path | None = None, seed: int = 1103) -> dict[str, Any]:
    """Analytic LM solution, finite-difference gradient sweep, and outer-acceptance behaviour."""
    problem = linear_least_squares_problem(seed)
    a, b = problem["a"], problem["b"]

    def residual(x):
        return a @ x - b

    x0 = np.zeros(a.shape[1])
    result = lm_minimize(residual, x0, LMConfig(max_iterations=60))
    x_lstsq = np.linalg.lstsq(a, b, rcond=None)[0]
    err_vs_analytic = float(np.abs(result.x - x_lstsq).max())
    err_vs_generating = float(np.abs(a @ result.x - b).max())

    # finite-difference sweep of the residual cost gradient at the found solution
    fd_sweep = []
    for step in (1e-4, 1e-5, 1e-6, 1e-7):
        jac = numerical_jacobian(residual, result.x, step=step)
        grad = jac.T @ residual(result.x)
        fd_sweep.append({"step": step, "grad_norm": float(np.linalg.norm(grad))})

    # outer loop with a term LM cannot see and that *conflicts* with the residual: a prior pull
    # towards x_prior.  The LM step (residual-only) therefore proposes a step that reduces the
    # residual sum while increasing the complete objective, which the outer loop must reject.
    x_prior = 0.5 * np.ones_like(x0)
    prior_weight = 5.0

    def full_objective(x):
        resid = residual(x)
        prior_penalty = 0.5 * prior_weight * float(np.sum((x - x_prior) ** 2))
        return 0.5 * float(resid @ resid) + prior_penalty

    # control term that is aligned with the residual (log-variance style): every residual
    # decrease also decreases this objective, which is the case Theseus can solve directly.
    def aligned_objective(x):
        resid = residual(x)
        return 0.5 * float(resid @ resid) + 2.0 * float(np.sum(np.log1p(resid**2)))

    outer = fixed_structure_outer_loop(residual, full_objective, x0, n_iterations=25)
    aligned = fixed_structure_outer_loop(residual, aligned_objective, x0, n_iterations=25)
    full_values = [rec.full_objective_proposed for rec in outer.records]
    monotone = all(
        b_ <= a_ + 1e-12
        for a_, b_ in zip([full_objective(x0)] + full_values, full_values, strict=False)
    )
    rejected_with_residual_decrease = sum(
        1
        for rec in outer.records
        if (not rec.accepted) and rec.residual_cost_proposed < rec.residual_cost_before
    )

    report: dict[str, Any] = {
        "linear_least_squares": {
            "analytic_error_max_abs": err_vs_analytic,
            "residual_max_abs": err_vs_generating,
            "iterations": result.iterations,
            "accepted_steps": result.accepted_steps,
            "converged": result.converged,
            "final_cost": result.cost,
            "finite_difference_sweep": fd_sweep,
            "x_solution": result.x.tolist(),
            "x_analytic": x_lstsq.tolist(),
        },
        "outer_acceptance": {
            "n_accepted": outer.n_accepted,
            "n_rejected": outer.n_rejected,
            "full_objective_initial": full_objective(x0),
            "full_objective_final": outer.full_objective,
            "monotone_non_increasing": bool(monotone),
            "rejected_steps_that_lowered_residual": rejected_with_residual_decrease,
            "residual_cost_initial": 0.5 * float(residual(x0) @ residual(x0)),
            "prior_weight": prior_weight,
            "x_prior": x_prior.tolist(),
            "outer_x_final": outer.x.tolist(),
            "records": [rec.__dict__ for rec in outer.records[:10]],
        },
        "outer_acceptance_aligned_term": {
            "n_accepted": aligned.n_accepted,
            "n_rejected": aligned.n_rejected,
            "full_objective_initial": aligned_objective(x0),
            "full_objective_final": aligned.full_objective,
            "note": (
                "control case: the extra term is monotone in the residual, so LM steps are "
                "accepted whenever they lower the residual sum"
            ),
        },
    }
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "theseus_lm_reference.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="reference LM checks")
    parser.add_argument("--out", default="outputs/n02-full/p0/parity/theseus")
    args = parser.parse_args()
    payload = run_lm_reference_check(args.out)
    print(json.dumps(payload["linear_least_squares"], indent=2))
    print(json.dumps(payload["outer_acceptance"], indent=2))

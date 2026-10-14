"""SMC numerical reference (R28) on a fixed small discrete model.

Upstream (verbatim copies kept next to this file)::

    components/upstream/particles/particles/core.py
    components/upstream/particles/particles/resampling.py
    components/upstream/particles/particles/state_space_models.py
    repository nchopin/particles @ 3388c4bd30ab9acc5a27c970089230f93610b627

The reference run executes the *complete* pinned checkout in ``third_party/pinned/`` (same
commit, all modules), because the bootstrap filter needs ``particles.hmm`` and
``particles.distributions`` as well; the three audited files are byte-identical between the two
locations (see ``third_party.lock.json``).

What is compared
----------------
A fixed discrete hidden Markov model with ``K`` states and ``M`` observation symbols:

* upstream exact filtering (``particles.hmm.BaumWelch(...).forward().filt``) and an independent
  brute-force enumeration written here,
* upstream SMC (``particles.state_space_models.Bootstrap`` + ``particles.SMC``) with a fixed
  number of particles and a fixed NumPy seed.

The saved numbers are the total-variation distance and the per-state maximum absolute error of
the SMC filtering distribution against the exact one, per time step.  Tiny ``K`` keeps the
"exact" side genuinely exhaustive rather than a second approximation.

This module is pure NumPy and therefore runs in the project root environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_REPOSITORY = "nchopin/particles"
UPSTREAM_COMMIT = "3388c4bd30ab9acc5a27c970089230f93610b627"
PINNED_ROOT = Path("third_party/pinned/nchopin_particles@3388c4bd30")


def ensure_particles_on_path() -> Path:
    import sys

    root = PINNED_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"pinned particles checkout missing: {root} (fetch the commit tarball first)"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# --------------------------------------------------------------------------------------
# the fixed model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscreteModel:
    """Small stationary HMM: ``trans`` (K, K) rows sum to 1, ``emit`` (K, M) rows sum to 1."""

    trans: np.ndarray
    emit: np.ndarray
    init: np.ndarray

    @property
    def num_states(self) -> int:
        return int(self.trans.shape[0])

    @property
    def num_symbols(self) -> int:
        return int(self.emit.shape[1])


def default_model() -> DiscreteModel:
    """Deterministic 5-state / 3-symbol model used by every P0 SMC measurement."""
    trans = np.array(
        [
            [0.70, 0.20, 0.05, 0.04, 0.01],
            [0.15, 0.60, 0.15, 0.06, 0.04],
            [0.05, 0.15, 0.60, 0.15, 0.05],
            [0.04, 0.06, 0.15, 0.60, 0.15],
            [0.01, 0.04, 0.05, 0.20, 0.70],
        ]
    )
    emit = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.30, 0.55, 0.15],
            [0.10, 0.60, 0.30],
            [0.15, 0.35, 0.50],
            [0.05, 0.20, 0.75],
        ]
    )
    init = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    return DiscreteModel(trans=trans, emit=emit, init=init)


def fixed_observations(model: DiscreteModel, length: int = 12, seed: int = 1103) -> np.ndarray:
    """Observations drawn from the model itself with a fixed seed (realised once, then frozen)."""
    rng = np.random.default_rng(seed)
    state = int(rng.choice(model.num_states, p=model.init))
    obs = []
    for _ in range(length):
        obs.append(int(rng.choice(model.num_symbols, p=model.emit[state])))
        state = int(rng.choice(model.num_states, p=model.trans[state]))
    return np.array(obs, dtype=np.int64)


# --------------------------------------------------------------------------------------
# exact filtering
# --------------------------------------------------------------------------------------


def exact_filter_enumeration(model: DiscreteModel, obs: np.ndarray) -> np.ndarray:
    """Brute-force forward recursion, written independently of the upstream implementation."""
    prior = np.array(model.init, dtype=np.float64)
    rows = []
    for y in obs:
        post = prior * model.emit[:, int(y)]
        post = post / post.sum()
        rows.append(post)
        prior = post @ model.trans
    return np.array(rows)


def exact_filter_upstream(model: DiscreteModel, obs: np.ndarray) -> np.ndarray:
    """Upstream exact filter: ``particles.hmm.BaumWelch`` forward recursion."""
    ensure_particles_on_path()
    import particles.hmm as upstream_hmm

    class _HMM(upstream_hmm.HMM):
        def PY(self, t, xp, x):
            import particles.distributions as dists

            return dists.Categorical(p=model.emit[np.asarray(x, dtype=int)])

    hmm = _HMM(init_dist=np.array(model.init), trans_mat=np.array(model.trans))
    algo = upstream_hmm.BaumWelch(hmm=hmm, data=list(np.asarray(obs, dtype=int)))
    algo.forward()
    return np.array(algo.filt, dtype=np.float64)


# --------------------------------------------------------------------------------------
# upstream SMC
# --------------------------------------------------------------------------------------


def run_upstream_smc(
    model: DiscreteModel,
    obs: np.ndarray,
    num_particles: int = 4096,
    seed: int = 1103,
) -> dict[str, Any]:
    """Bootstrap particle filter from the pinned upstream checkout; returns weighted marginals."""
    ensure_particles_on_path()
    import particles
    import particles.distributions as dists
    import particles.state_space_models as ssms

    class _HMM(ssms.StateSpaceModel):
        def PX0(self):
            return dists.Categorical(p=np.array(model.init))

        def PX(self, t, xp):
            return dists.Categorical(p=model.trans[np.asarray(xp, dtype=int)])

        def PY(self, t, xp, x):
            return dists.Categorical(p=model.emit[np.asarray(x, dtype=int)])

    np.random.seed(seed)
    ssm = _HMM()
    fk = ssms.Bootstrap(ssm=ssm, data=list(np.asarray(obs, dtype=int)))
    algo = particles.SMC(fk=fk, N=num_particles, store_history=True)
    algo.run()
    hist = algo.hist
    marginals = []
    # ``particles`` stores per-step weights as ``Weights`` objects in ``hist.wgts`` (the
    # normalised vector is ``.W``) and particle positions in ``hist.X``; see
    # particles/smoothing.py:ParticleHistory and particles/resampling.py:Weights.
    for t in range(len(obs)):
        weights = np.asarray(hist.wgts[t].W, dtype=np.float64)
        states = np.asarray(hist.X[t], dtype=int).reshape(-1)
        marginal = np.zeros(model.num_states)
        for k in range(model.num_states):
            marginal[k] = weights[states == k].sum()
        marginals.append(marginal)
    return {
        "marginals": np.array(marginals),
        "log_likelihood": float(algo.logLt),
        "num_particles": int(num_particles),
        "seed": int(seed),
        # ``algo.W`` is the final normalised weight vector in this version; older/newer releases
        # wrap it in a ``Weights`` object, so both shapes are accepted.
        "final_ess": float(1.0 / np.sum(np.asarray(getattr(algo.W, "W", algo.W)) ** 2)),
    }


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------


def run_smc_reference_check(
    out_dir: str | Path | None = None,
    num_particles: int = 4096,
    seed: int = 1103,
) -> dict[str, Any]:
    """Run SMC vs exact enumeration and return the measured distribution errors."""
    model = default_model()
    obs = fixed_observations(model)
    exact_enum = exact_filter_enumeration(model, obs)
    exact_up = exact_filter_upstream(model, obs)
    smc = run_upstream_smc(model, obs, num_particles=num_particles, seed=seed)
    smc_marg = smc["marginals"]

    tv = 0.5 * np.abs(smc_marg - exact_enum).sum(axis=1)
    max_state_err = np.abs(smc_marg - exact_enum).max(axis=1)
    result: dict[str, Any] = {
        "model": {
            "num_states": model.num_states,
            "num_symbols": model.num_symbols,
            "observations": obs.tolist(),
            "transition_sha256": _array_sha256(model.trans),
            "emission_sha256": _array_sha256(model.emit),
            "init_sha256": _array_sha256(model.init),
        },
        "exact_upstream_vs_enumeration_max_abs": float(np.abs(exact_up - exact_enum).max()),
        "smc": {
            "num_particles": smc["num_particles"],
            "seed": smc["seed"],
            "final_ess": smc["final_ess"],
            "log_likelihood": smc["log_likelihood"],
            "tv_per_step": tv.tolist(),
            "max_state_err_per_step": max_state_err.tolist(),
            "tv_mean": float(tv.mean()),
            "tv_max": float(tv.max()),
            "max_state_err_max": float(max_state_err.max()),
        },
        "final_step": {
            "exact": exact_enum[-1].tolist(),
            "smc": smc_marg[-1].tolist(),
        },
    }
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "smc_reference.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(
            out_path / "smc_reference.npz",
            exact=exact_enum,
            exact_upstream=exact_up,
            smc=smc_marg,
            observations=obs,
        )
    return result


def _array_sha256(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="particles SMC reference check")
    parser.add_argument("--out", default="outputs/n02-full/p0/parity/smc")
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1103)
    args = parser.parse_args()
    report = run_smc_reference_check(args.out, num_particles=args.particles, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "smc"}, indent=2))
    smc = report["smc"]
    print(
        f"smc tv_mean={smc['tv_mean']:.5f} tv_max={smc['tv_max']:.5f} "
        f"max_state_err={smc['max_state_err_max']:.5f}"
    )

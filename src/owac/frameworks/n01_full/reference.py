"""Independent dense FP64 reference for the N01 v1.0 analytic estimates.

This module is the oracle of ``docs/prompts/n01_v1_0/P1_ANALYTIC_REFERENCE.md``
section 1: a small-problem, NumPy-only, dense FP64 implementation that never
calls the production ``statistics.py``.  The two implementations differ on
purpose, so an agreement is evidence about the mathematics rather than about a
shared code path:

* production whitens with ``torch.linalg.cholesky`` and solves by Cholesky /
  ``lstsq``; this module forms ``Sigma^-1/2`` from ``numpy.linalg.eigh`` and
  returns the explicit dense inverse of the normal matrix;
* production evaluates the per-phase basis inside the domain's declared
  centering/scaling transform; this module normalises independently unless the
  caller passes the transform it wants compared.

Predictions, not raw coefficients, are the comparison surface: the least-squares
fit is invariant to the invertible affine reparameterisation the two
implementations apply, so ``predict`` is meaningful across the two coordinate
normalisations (the parameter-space prior is the exception and requires a shared
``normalization``).

``[文献]`` the estimators here are ordinary GLS / Gauss-Markov results; this
module does not prove them and does not extend their validity to data-dependent
designs (see ``research/theory/n01_support_contract_v1.md`` T2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from owac.frameworks.n01_full.contracts import SourceKey

__all__ = [
    "DenseGLSResult",
    "ReferenceObservation",
    "BlockFit",
    "ReferenceFit",
    "dense_gls",
    "reference_fit",
    "reference_predict",
]

_FLOAT = np.float64


@dataclass(frozen=True)
class DenseGLSResult:
    """One dense whitened least-squares solve at FP64."""

    theta: np.ndarray
    covariance: np.ndarray | None
    rank: int
    condition_number: float
    singular_values: np.ndarray
    observation_count: int
    parameter_count: int


def _as_matrix(value: Any, *, name: str, square: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=_FLOAT)
    if square and matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")
    return matrix


def _symmetric_whitener(covariance: np.ndarray, *, name: str) -> np.ndarray:
    """``Sigma^-1/2`` via the eigen-decomposition of a symmetric PD matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0.0):
        raise np.linalg.LinAlgError(
            f"{name} is not positive definite (smallest eigenvalue {eigenvalues.min():.3e}); "
            "the dense reference refuses to regularise it silently"
        )
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def dense_gls(
    design: Any,
    values: Any,
    covariance: Any | None = None,
    *,
    prior_covariance: Any | None = None,
    prior_mean: Any | None = None,
    rank_rtol: float | None = None,
) -> DenseGLSResult:
    """Whitened least squares with an optional Gaussian prior, dense FP64.

    ``design`` is ``[n_observations, n_parameters]``, ``values`` is
    ``[n_observations]`` and ``covariance`` is the observation covariance
    ``[n, n]`` (identity when omitted).  The prior adds ``Lambda_prior`` /
    ``eta_prior`` once; it is not an observation and the caller counts it
    separately.
    """
    phi = _as_matrix(design, name="design")
    y = np.asarray(values, dtype=_FLOAT).reshape(-1)
    if phi.ndim != 2:
        raise ValueError("design must be two dimensional")
    if phi.shape[0] != y.shape[0]:
        raise ValueError(f"design has {phi.shape[0]} rows but values has {y.shape[0]}")
    if covariance is None:
        whitener = np.eye(phi.shape[0], dtype=_FLOAT)
    else:
        sigma = _as_matrix(covariance, name="covariance", square=True)
        if sigma.shape[0] != phi.shape[0]:
            raise ValueError(f"covariance {sigma.shape} does not match {phi.shape[0]} observations")
        whitener = _symmetric_whitener(sigma, name="covariance")

    whitened = whitener.T @ phi
    target = whitener.T @ y
    _, singular_values, _ = np.linalg.svd(whitened, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    if rank_rtol is None:
        rank_rtol = max(whitened.shape) * float(np.finfo(_FLOAT).eps)
    tolerance = rank_rtol * largest
    rank = int(np.sum(singular_values > tolerance))

    lambda_matrix = whitened.T @ whitened
    eta = whitened.T @ target
    if prior_covariance is not None:
        prior = _as_matrix(prior_covariance, name="prior_covariance", square=True)
        if prior.shape[0] != phi.shape[1]:
            raise ValueError(
                f"prior covariance {prior.shape} does not match {phi.shape[1]} parameters"
            )
        lambda_prior = np.linalg.inv(prior)
        mean = (
            np.zeros(phi.shape[1], dtype=_FLOAT)
            if prior_mean is None
            else np.asarray(prior_mean, dtype=_FLOAT).reshape(-1)
        )
        lambda_matrix = lambda_matrix + lambda_prior
        eta = eta + lambda_prior @ mean

    if rank == phi.shape[1] or prior_covariance is not None:
        # Cholesky is the positive-definiteness test; ``matrix_rank`` on the normal
        # matrix would square the condition number and wrongly declare a genuinely
        # full-rank but ill-conditioned design rank deficient.
        try:
            np.linalg.cholesky(lambda_matrix)
            theta = np.linalg.solve(lambda_matrix, eta)
            covariance = np.linalg.inv(lambda_matrix)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(whitened, target, rcond=None)[0]
            covariance = None
    else:
        theta = np.linalg.lstsq(whitened, target, rcond=None)[0]
        covariance = None
    condition = (
        float("inf")
        if singular_values.size == 0 or singular_values[-1] <= 0.0
        else float(singular_values[0] / singular_values[-1])
    )
    return DenseGLSResult(
        theta=theta,
        covariance=covariance,
        rank=rank,
        condition_number=condition,
        singular_values=singular_values,
        observation_count=int(phi.shape[0]),
        parameter_count=int(phi.shape[1]),
    )


@dataclass(frozen=True)
class ReferenceObservation:
    """One raw observation as the oracle sees it: identity, phase, coordinate, value."""

    source_id: SourceKey
    phase: int | str
    xy: tuple[float, float]
    value: float
    variance: float


@dataclass(frozen=True)
class BlockFit:
    """The oracle's fit for one phase/channel block."""

    phase: int | str
    theta: np.ndarray
    covariance: np.ndarray | None
    rank: int
    condition_number: float
    center: tuple[float, float]
    scale: tuple[float, float]
    source_ids: tuple[SourceKey, ...]
    observation_count: int
    parameter_names: tuple[str, ...]

    def predict(self, xy: tuple[float, float]) -> tuple[float, float]:
        phi = _basis_row(self.parameter_names, self.center, self.scale, xy)
        mean = float(phi @ self.theta)
        if self.covariance is None:
            return mean, float("nan")
        return mean, float(phi @ self.covariance @ phi)


@dataclass(frozen=True)
class ReferenceFit:
    """Per-phase dense FP64 reference fit."""

    blocks: Mapping[int | str, BlockFit]
    basis: str
    dimension: int

    def predict(self, phase: int | str, xy: tuple[float, float]) -> tuple[float, float]:
        block = self.blocks.get(phase)
        if block is None:
            return float("nan"), float("nan")
        return block.predict(xy)

    def theta(self, phase: int | str) -> np.ndarray:
        block = self.blocks.get(phase)
        if block is None:
            raise KeyError(f"unknown reference phase {phase!r}")
        return block.theta


def _parameter_names(basis: str, dimension: int) -> tuple[str, ...]:
    if basis == "constant":
        return ("a",)
    if basis == "affine":
        if dimension == 2:
            return ("a", "bx", "by")
        if dimension == 1:
            return ("a", "bx")
        raise ValueError(f"unsupported dimension {dimension!r} for the affine basis")
    raise ValueError(f"unknown reference basis {basis!r}; expected 'constant' or 'affine'")


def _basis_row(
    names: tuple[str, ...],
    center: tuple[float, float],
    scale: tuple[float, float],
    xy: tuple[float, float],
) -> np.ndarray:
    values = []
    for name in names:
        if name == "a":
            values.append(1.0)
        elif name == "bx":
            values.append((xy[0] - center[0]) / scale[0])
        elif name == "by":
            values.append((xy[1] - center[1]) / scale[1])
        else:  # pragma: no cover - guarded by _parameter_names
            raise ValueError(f"unknown reference basis term {name!r}")
    return np.asarray(values, dtype=_FLOAT)


def reference_fit(
    observations: Iterable[ReferenceObservation],
    *,
    basis: str = "affine",
    dimension: int = 2,
    normalization: Mapping[int | str, tuple[tuple[float, float], tuple[float, float]]]
    | None = None,
    prior_covariance: Any | None = None,
    prior_mean: Any | None = None,
) -> ReferenceFit:
    """Fit the per-phase dense reference on raw observations.

    Phases are fitted separately (their averages are never formed first).
    ``normalization`` optionally supplies the production transform per phase
    ``(center, scale)``; without it the oracle picks its own and comparisons must
    use predictions rather than coefficients.
    """
    names = _parameter_names(basis, dimension)
    grouped: dict[int | str, list[ReferenceObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.phase, []).append(observation)

    blocks: dict[int | str, BlockFit] = {}
    for phase in sorted(grouped, key=str):
        rows = grouped[phase]
        # Deduplicate by identity the same way the contract requires: one RAW
        # sample counts once even if a caller lists it twice.
        unique: list[ReferenceObservation] = []
        seen: set[tuple[Any, ...]] = set()
        for row in rows:
            identity = row.source_id.identity()
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(row)
        coordinates = np.asarray([row.xy for row in unique], dtype=_FLOAT)
        declared = None if normalization is None else normalization.get(phase)
        if declared is None:
            center = tuple(float(value) for value in coordinates.mean(axis=0))
            spread = coordinates.std(axis=0)
            scale = tuple(float(value) if value > 0 else 1.0 for value in spread)
        else:
            center = tuple(float(value) for value in declared[0])
            scale = tuple(float(value) for value in declared[1])
        design = np.stack(
            [_basis_row(names, center, scale, tuple(row.xy)) for row in unique], axis=0
        )
        variances = np.asarray([row.variance for row in unique], dtype=_FLOAT)
        if np.any(variances <= 0.0):
            raise ValueError("reference observations need positive declared variances")
        values = np.asarray([row.value for row in unique], dtype=_FLOAT)
        covariance = np.diag(variances)
        result = dense_gls(
            design,
            values,
            covariance,
            prior_covariance=prior_covariance,
            prior_mean=prior_mean,
        )
        blocks[phase] = BlockFit(
            phase=phase,
            theta=result.theta,
            covariance=result.covariance,
            rank=result.rank,
            condition_number=result.condition_number,
            center=center,
            scale=scale,
            source_ids=tuple(row.source_id for row in unique),
            observation_count=len(unique),
            parameter_names=names,
        )
    return ReferenceFit(blocks=blocks, basis=basis, dimension=dimension)


def reference_predict(
    fit: ReferenceFit, phase: int | str, xy: tuple[float, float]
) -> tuple[float, float]:
    """Convenience wrapper: ``(mean, variance)`` at a query coordinate."""
    return fit.predict(phase, xy)

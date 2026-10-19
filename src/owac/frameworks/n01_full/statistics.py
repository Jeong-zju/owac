"""Fixed-design structure fits: per-phase GLS with explicit rank and labels.

Implementation of ``docs/prompts/n01_v1_0/P1_ANALYTIC_REFERENCE.md`` section 2.
The observation model is ``y = Phi theta + eps`` with a declared covariance
``Sigma``; every CFA phase (or measurement channel) is fitted separately and the
phases are never averaged first.  The solve follows the prompt literally::

    L = cholesky(Sigma)
    A = solve_triangular(L, Phi)
    b = solve_triangular(L, y)
    theta = lstsq(A, b)
    Lambda = A.T @ A ; eta = A.T @ b

with ``Lambda^-1`` obtained by Cholesky when the design is full rank, a declared
prior added once as ``Lambda_prior/eta_prior``, and rank deficiency reported
through ``identifiable_mask`` instead of a small ridge.

Three uncertainty labels are kept apart, because mixing them is a documented
failure mode:

* ``covariance`` -- exact GLS/posterior covariance under the declared ``Sigma``;
* ``proxy`` -- the conservative fixed-weight bound
  ``(sum |a_i| sigma_i)^2`` used when the correlation is unknown; it is stored as
  a diagonal PSD envelope; query bounds are computed from the actual raw-source
  weights, never from an outer product that can cancel for signed queries;
* ``none`` -- rank-deficient without a prior: coordinates are marked
  unidentifiable and no covariance is returned.

The module never builds the design from anything but the domain's *evidence*
sources: context sources, missing/bad/saturated samples and absent identities are
reported in ``excluded_sources``/``diagnostics`` and never enter ``Phi`` or
``Sigma``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from owac.frameworks.n01_full.contracts import (
    CorrespondenceMap,
    EvidenceEstimate,
    SourceKey,
    SupportDomain,
)
from owac.frameworks.n01_full.domains import (
    VersionedDomain,
    canonical_sources,
)
from owac.frameworks.n01_full.ledger import MeasurementLedger

__all__ = [
    "BASIS_TERMS",
    "CoordinateTransform",
    "NoiseModel",
    "GaussianPrior",
    "InterpolationSpec",
    "BlockFit",
    "StructureFit",
    "align_interpolation_sources",
    "fit_structure",
    "interpolation_from_correspondence",
]

#: Basis terms per declared basis kind, as a function of the declared dimension.
BASIS_TERMS: Mapping[str, Mapping[int, tuple[str, ...]]] = MappingProxyType(
    {
        "phase_local_constant": MappingProxyType({1: ("a",), 2: ("a",)}),
        "phase_local_affine": MappingProxyType({1: ("a", "bx"), 2: ("a", "bx", "by")}),
    }
)

NOISE_KINDS = ("diagonal", "dense", "unknown_correlation")


@dataclass(frozen=True)
class CoordinateTransform:
    """Explicit centering/scaling of a domain's declared coordinate chart.

    The transform is saved with the fit so a later readout can be compared at the
    same query coordinates and units instead of in an undocumented normalised
    frame.  Coordinates are ``(x, y)`` in ``units``.
    """

    center: tuple[float, float]
    scale: tuple[float, float]
    chart: str
    units: str = "sensor_pixel"

    def __post_init__(self) -> None:
        if len(self.center) != 2 or len(self.scale) != 2:
            raise ValueError("CoordinateTransform center/scale must be two-dimensional")
        if self.scale[0] == 0.0 or self.scale[1] == 0.0:
            raise ValueError("CoordinateTransform scale must be non-zero")
        if not self.chart:
            raise ValueError("CoordinateTransform.chart must name the reference chart")

    def normalize(self, xy: tuple[float, float]) -> tuple[float, float]:
        return (
            (float(xy[0]) - self.center[0]) / self.scale[0],
            (float(xy[1]) - self.center[1]) / self.scale[1],
        )

    def denormalize(self, normalized: tuple[float, float]) -> tuple[float, float]:
        return (
            normalized[0] * self.scale[0] + self.center[0],
            normalized[1] * self.scale[1] + self.center[1],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "scale": list(self.scale),
            "chart": self.chart,
            "units": self.units,
        }


def _basis_terms(basis: str, dimension: int) -> tuple[str, ...]:
    if basis not in BASIS_TERMS:
        if basis == "directional_per_phase":
            raise ValueError(
                "directional_per_phase is declared by the configuration but not part of the "
                "P1 analytic reference; implement it in the phase that needs it"
            )
        raise ValueError(f"unknown basis kind {basis!r}")
    table = BASIS_TERMS[basis]
    if dimension not in table:
        raise ValueError(f"basis {basis!r} has no declared form for dimension {dimension}")
    return table[dimension]


def _basis_row(
    terms: Sequence[str],
    transform: CoordinateTransform,
    xy: tuple[float, float],
    *,
    dtype: torch.dtype,
) -> list[float]:
    normalized = transform.normalize(xy)
    row: list[float] = []
    for term in terms:
        if term == "a":
            row.append(1.0)
        elif term == "bx":
            row.append(normalized[0])
        elif term == "by":
            row.append(normalized[1])
        else:  # pragma: no cover - guarded by BASIS_TERMS
            raise ValueError(f"unknown basis term {term!r}")
    del dtype
    return row


@dataclass(frozen=True)
class NoiseModel:
    """Declared observation covariance for the evidence sources.

    ``diagonal`` and ``unknown_correlation`` declare per-source variances;
    ``dense`` declares the joint covariance over ``source_order``.  Retraction
    takes the remaining sub-block of that *observation* covariance, never a slice
    of a previously assembled precision matrix.
    """

    kind: str
    version: str
    source_order: tuple[SourceKey, ...] = ()
    variances: tuple[float, ...] = ()
    matrix: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.kind not in NOISE_KINDS:
            raise ValueError(f"unknown noise kind {self.kind!r}; expected {NOISE_KINDS}")
        if not self.version:
            raise ValueError("NoiseModel.version must be declared")
        if len(self.source_order) != len(self.variances):
            raise ValueError("NoiseModel source_order and variances must be aligned")
        if self.kind == "dense":
            if self.matrix is None:
                raise ValueError("a dense NoiseModel requires the joint covariance matrix")
            if tuple(self.matrix.shape) != (len(self.source_order),) * 2:
                raise ValueError(
                    f"dense covariance {tuple(self.matrix.shape)} does not match "
                    f"{len(self.source_order)} declared sources"
                )
        for variance in self.variances:
            if variance <= 0.0:
                raise ValueError("declared variances must be positive")
        if self.kind == "dense":
            assert self.matrix is not None
            eigenvalues = torch.linalg.eigvalsh(self.matrix)
            tolerance = (
                max(self.matrix.shape)
                * float(torch.finfo(torch.float64).eps)
                * float(eigenvalues.max())
            )
            if float(eigenvalues.min()) <= tolerance:
                raise ValueError(
                    "a dense noise covariance must be positive definite; a singular joint "
                    "covariance encodes duplicate observations and belongs in the raw factors"
                )

    @classmethod
    def diagonal(
        cls,
        variances: Mapping[SourceKey, float] | Sequence[float],
        *,
        source_order: Sequence[SourceKey] | None = None,
        version: str = "noise-diagonal-v1",
    ) -> NoiseModel:
        order, values = _ordered_variances(variances, source_order)
        return cls(kind="diagonal", version=version, source_order=order, variances=values)

    @classmethod
    def unknown_correlation(
        cls,
        variances: Mapping[SourceKey, float] | Sequence[float],
        *,
        source_order: Sequence[SourceKey] | None = None,
        version: str = "noise-unknown-correlation-v1",
    ) -> NoiseModel:
        order, values = _ordered_variances(variances, source_order)
        return cls(
            kind="unknown_correlation", version=version, source_order=order, variances=values
        )

    @classmethod
    def dense(
        cls,
        source_order: Sequence[SourceKey],
        matrix: Any,
        *,
        version: str = "noise-dense-v1",
    ) -> NoiseModel:
        covariance = torch.as_tensor(matrix, dtype=torch.float64)
        order = tuple(source_order)
        variances = tuple(float(covariance[i, i]) for i in range(len(order)))
        return cls(
            kind="dense",
            version=version,
            source_order=order,
            variances=variances,
            matrix=covariance,
        )

    def variance_of(self, key: SourceKey) -> float:
        for position, source in enumerate(self.source_order):
            if source.identity() == key.identity():
                return float(self.variances[position])
        raise KeyError(
            f"noise version {self.version!r} declares no variance for {key.identity()}; a "
            "missing declaration is not a default variance"
        )

    def covariance_for(self, keys: Sequence[SourceKey], *, dtype: torch.dtype) -> torch.Tensor:
        """Noise covariance of exactly these observations, in the given order."""
        if self.kind == "dense":
            positions = []
            for key in keys:
                for position, source in enumerate(self.source_order):
                    if source.identity() == key.identity():
                        positions.append(position)
                        break
                else:
                    raise KeyError(
                        f"dense noise version {self.version!r} does not cover {key.identity()}"
                    )
            index = torch.tensor(positions, dtype=torch.long)
            assert self.matrix is not None
            return self.matrix[index][:, index].to(dtype)
        variances = [self.variance_of(key) for key in keys]
        return torch.diag(torch.tensor(variances, dtype=dtype))


def _ordered_variances(
    variances: Mapping[SourceKey, float] | Sequence[float],
    source_order: Sequence[SourceKey] | None,
) -> tuple[tuple[SourceKey, ...], tuple[float, ...]]:
    if isinstance(variances, Mapping):
        if source_order is not None:
            raise ValueError("pass either a variance mapping or an explicit source order, not both")
        order = tuple(variances)
        return order, tuple(float(variances[key]) for key in order)
    if source_order is None:
        raise ValueError("a variance sequence needs the matching source_order")
    values = tuple(float(item) for item in variances)
    return tuple(source_order), values


@dataclass(frozen=True)
class GaussianPrior:
    """A declared Gaussian prior, added once in parameter space.

    It is never an observation: it does not enter
    ``EvidenceEstimate.independent_observation_count``.
    """

    prior_id: str
    mean: torch.Tensor
    covariance: torch.Tensor
    transform: CoordinateTransform | None = None

    def __post_init__(self) -> None:
        if self.mean.numel() > 1 and self.transform is None:
            raise ValueError("an affine prior requires an explicit coordinate transform")
        if not self.prior_id:
            raise ValueError("GaussianPrior.prior_id must be declared")
        if self.mean.ndim != 1:
            raise ValueError("GaussianPrior.mean must be one dimensional")
        if tuple(self.covariance.shape) != (self.mean.shape[0], self.mean.shape[0]):
            raise ValueError("GaussianPrior.covariance must match the prior mean")


@dataclass(frozen=True)
class InterpolationSpec:
    """Explicit ``z = A_interp y`` applied to the domain's canonical evidence set.

    ``weights`` has shape ``[n_rows, n_evidence]`` and is aligned with
    ``canonical_sources(domain.evidence_source_ids)``.  Overlapping taps are
    allowed: the row covariance is ``A_interp Sigma A_interp^T``, which is exactly
    where duplicate information shows up as a singular block.

    P2 addition (optional, default ``None`` keeps P1 behaviour): ``source_ids``
    names the identity of every column, so a builder that recomputed its weights
    from a redacted ledger view can declare *which* sources those columns are and
    the fit re-indexes them onto its own canonical evidence order by identity
    instead of trusting a positional convention.
    """

    row_positions: tuple[tuple[float, float], ...]
    weights: torch.Tensor
    version: str = "interp-v1"
    source_ids: tuple[SourceKey, ...] | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("InterpolationSpec.version must be declared")
        if self.weights.ndim != 2:
            raise ValueError("InterpolationSpec.weights must be [n_rows, n_evidence]")
        if self.weights.shape[0] != len(self.row_positions):
            raise ValueError("InterpolationSpec needs one declared position per row")
        if self.source_ids is not None and len(self.source_ids) != self.weights.shape[1]:
            raise ValueError(
                f"InterpolationSpec.source_ids has {len(self.source_ids)} entries but weights "
                f"has {self.weights.shape[1]} columns"
            )


@dataclass(frozen=True)
class BlockFit:
    """One phase/channel block of a structure fit."""

    phase: int | str
    parameter_names: tuple[str, ...]
    theta: torch.Tensor
    covariance: torch.Tensor | None
    rank: int
    condition_number: float
    identifiable: torch.Tensor
    observation_count: int
    source_ids: tuple[SourceKey, ...]
    transform: CoordinateTransform
    row_projector: torch.Tensor
    prior_supplied: bool
    raw_noise_weights: torch.Tensor | None = None
    raw_estimator_weights: torch.Tensor | None = None

    def predict(self, xy: tuple[float, float]) -> tuple[float, float]:
        phi = torch.tensor(
            _basis_row(self.parameter_names, self.transform, xy, dtype=self.theta.dtype),
            dtype=self.theta.dtype,
        )
        if not self.prior_supplied and not torch.allclose(
            phi @ self.row_projector, phi, atol=1e-8, rtol=1e-8
        ):
            return float("nan"), float("nan")
        mean = float(phi @ self.theta)
        if self.raw_noise_weights is not None:
            return mean, float((phi @ self.raw_noise_weights).abs().sum().square())
        if self.covariance is None:
            return mean, float("nan")
        return mean, float(phi @ self.covariance @ phi)


@dataclass(frozen=True)
class StructureFit:
    """The result of :func:`fit_structure`: estimate plus what it is based on."""

    estimate: EvidenceEstimate
    rank: int
    parameter_count: int
    parameter_names: tuple[str, ...]
    condition_number: float
    blocks: Mapping[int | str, BlockFit]
    excluded_sources: tuple[tuple[SourceKey, str], ...]
    diagnostics: Mapping[str, Any]
    domain_id: str
    support_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", MappingProxyType(dict(self.blocks)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def transform(self) -> CoordinateTransform | None:
        """The single block's transform, or ``None`` when several blocks exist."""
        if len(self.blocks) == 1:
            return next(iter(self.blocks.values())).transform
        return None

    def transform_for(self, phase: int | str) -> CoordinateTransform:
        return self.blocks[phase].transform

    def predict(self, phase: int | str, xy: tuple[float, float]) -> tuple[float, float]:
        """``(mean, variance)`` at a declared-chart coordinate; NaN when unknown."""
        block = self.blocks.get(phase)
        if block is None:
            return float("nan"), float("nan")
        return block.predict(xy)

    def predictions(
        self, queries: Sequence[tuple[int | str, float, float]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorised readout at declared query coordinates ``(phase, x, y)``."""
        means, variances = [], []
        for phase, x, y in queries:
            mean, variance = self.predict(phase, (x, y))
            means.append(mean)
            variances.append(variance)
        return torch.tensor(means, dtype=torch.float64), torch.tensor(
            variances, dtype=torch.float64
        )


def fit_structure(
    domain: VersionedDomain | SupportDomain,
    ledger: MeasurementLedger,
    correspondence: CorrespondenceMap | InterpolationSpec | None = None,
    noise_model: NoiseModel | None = None,
    *,
    basis: str | None = None,
    dimension: int = 2,
    prior: GaussianPrior | None = None,
    dtype: torch.dtype = torch.float64,
    rank_rtol: float | None = None,
    leverage_tol: float = 1e-8,
) -> StructureFit:
    """Fit the declared structure from the domain's evidence sources only.

    ``correspondence`` is either an explicit :class:`InterpolationSpec` or a
    :class:`CorrespondenceMap` with one tap per reference row; a plain
    ``None`` samples every evidence source directly.
    """
    if isinstance(domain, VersionedDomain):
        support = domain.domain
        support_version = domain.support_version
    elif isinstance(domain, SupportDomain):
        support = domain
        support_version = f"unversioned-{domain.domain_id}"
    else:
        raise TypeError("fit_structure expects a SupportDomain or VersionedDomain")

    if isinstance(domain, VersionedDomain):
        bound_noise = domain.binding("noise_version")
        if bound_noise is not None and (noise_model is None or noise_model.version != bound_noise):
            raise ValueError("noise version differs from the committed domain binding")
        bound_prior = domain.binding("prior_id")
        if bound_prior is not None and (prior is None or prior.prior_id != bound_prior):
            raise ValueError("prior differs from the committed domain binding")
    basis_kind = basis or support.basis
    terms = _basis_terms(basis_kind, dimension)
    evidence, duplicates = canonical_sources(support.evidence_source_ids)
    if duplicates:  # pragma: no cover - SupportDomain already refuses duplicates
        raise ValueError("the domain lists a duplicate evidence source")

    used: list[SourceKey] = []
    values: list[float] = []
    coordinates: list[tuple[float, float]] = []
    phases: list[int | str] = []
    excluded: list[tuple[SourceKey, str]] = []
    for key in evidence:
        if not ledger.has_source(key):
            excluded.append((key, "absent"))
            continue
        record = ledger.get_source(key)
        if record.validity_kind != "valid":
            excluded.append((key, record.validity_kind))
            continue
        if support.phase is not None and key.channel_or_phase != support.phase:
            raise ValueError(
                f"domain {support.domain_id!r} declares phase {support.phase!r} but evidence "
                f"source {key.identity()} belongs to phase {key.channel_or_phase!r}"
            )
        used.append(key)
        values.append(float(record.raw_value))
        coordinates.append(ledger.sample_coordinates(key))
        phases.append(key.channel_or_phase)

    if noise_model is None:
        noise = NoiseModel.diagonal({key: 1.0 for key in used}, version="assumed-unit-variance-v1")
        assumed_unit_variance = True
    else:
        noise = noise_model
        assumed_unit_variance = False

    diagnostics: dict[str, Any] = {
        "basis": basis_kind,
        "dimension": dimension,
        "evidence_count": len(evidence),
        "used_count": len(used),
        "context_count": len(support.context_source_ids),
        "context_used_as_evidence": False,
        "assumed_unit_variance": assumed_unit_variance,
        "noise_version": noise.version,
        "noise_kind": noise.kind,
        "interpolated": False,
        "singular_interpolation_fallback": False,
        "prior_id": None if prior is None else prior.prior_id,
        "excluded": [
            {"source": list(key.identity()), "reason": reason} for key, reason in excluded
        ],
    }

    if not used:
        parameter_count = len(terms)
        mask = torch.zeros(parameter_count, dtype=torch.bool)
        estimate = EvidenceEstimate(
            theta=None,
            uncertainty_kind="none",
            identifiable_mask=mask,
            evidence_source_ids=(),
            support_version=support_version,
            covariance=None,
            prior_id=None if prior is None else prior.prior_id,
        )
        diagnostics["invalid_reason"] = "no valid measurement in the domain"
        return StructureFit(
            estimate=estimate,
            rank=0,
            parameter_count=parameter_count,
            parameter_names=tuple(terms),
            condition_number=float("inf"),
            blocks={},
            excluded_sources=tuple(excluded),
            diagnostics=diagnostics,
            domain_id=support.domain_id,
            support_version=support_version,
        )

    rows = _build_rows(
        support=support,
        ledger=ledger,
        used=used,
        values=values,
        coordinates=coordinates,
        phases=phases,
        correspondence=correspondence,
        terms=terms,
        dimension=dimension,
        noise=noise,
        dtype=dtype,
        diagnostics=diagnostics,
    )

    blocks: dict[int | str, BlockFit] = {}
    for phase in sorted(rows, key=str):
        blocks[phase] = _solve_block(
            phase=phase,
            data=rows[phase],
            terms=terms,
            noise=noise,
            prior=prior,
            dtype=dtype,
            rank_rtol=rank_rtol,
            leverage_tol=leverage_tol,
            diagnostics=diagnostics,
        )

    parameter_names = tuple(
        f"{phase}:{name}" for phase in blocks for name in blocks[phase].parameter_names
    )
    thetas = [blocks[phase].theta for phase in blocks]
    masks = [blocks[phase].identifiable for phase in blocks]
    covariances = [blocks[phase].covariance for phase in blocks]
    covariance = (
        torch.block_diag(*covariances) if all(item is not None for item in covariances) else None
    )
    if covariance is not None and noise.kind == "dense" and len(blocks) > 1:
        # Separate phase fits still have correlated errors when raw noise crosses phases.
        sigma = noise.covariance_for(used, dtype=dtype)
        if prior is not None:
            cross_phase = torch.tensor([[a != b for b in phases] for a in phases])
            if bool((sigma[cross_phase] != 0).any()):
                raise ValueError("cross-phase correlated priors require a joint posterior solve")
        else:
            operators = []
            for block in blocks.values():
                operator = torch.zeros((len(terms), len(used)), dtype=dtype)
                for column, source in enumerate(block.source_ids):
                    operator[:, used.index(source)] = block.raw_estimator_weights[:, column]
                operators.append(operator)
            operator = torch.cat(operators)
            covariance = operator @ sigma @ operator.T
    if noise.kind == "unknown_correlation":
        if covariance is not None:
            # Arbitrary correlation also spans phase blocks; the global PSD envelope
            # needs the total parameter dimension, not each phase's dimension alone.
            covariance = covariance * len(blocks)
        uncertainty_kind = "proxy" if covariance is not None else "none"
    else:
        uncertainty_kind = "covariance" if covariance is not None else "none"
    rank = sum(blocks[phase].rank for phase in blocks)
    condition_number = max(
        (blocks[phase].condition_number for phase in blocks), default=float("inf")
    )
    estimate = EvidenceEstimate(
        theta=torch.cat(thetas) if thetas else None,
        uncertainty_kind=uncertainty_kind,
        identifiable_mask=torch.cat(masks),
        evidence_source_ids=tuple(used),
        support_version=support_version,
        covariance=covariance,
        prior_id=None if prior is None else prior.prior_id,
    )
    diagnostics["ranks"] = {str(phase): blocks[phase].rank for phase in blocks}
    diagnostics["condition_numbers"] = {
        str(phase): blocks[phase].condition_number for phase in blocks
    }
    diagnostics["identifiable"] = {
        str(phase): [bool(item) for item in blocks[phase].identifiable] for phase in blocks
    }
    return StructureFit(
        estimate=estimate,
        rank=rank,
        parameter_count=len(parameter_names),
        parameter_names=parameter_names,
        condition_number=condition_number,
        blocks=blocks,
        excluded_sources=tuple(excluded),
        diagnostics=diagnostics,
        domain_id=support.domain_id,
        support_version=support_version,
    )


@dataclass(frozen=True)
class _BlockRows:
    phase: int | str
    positions: tuple[tuple[float, float], ...]
    design: torch.Tensor  # [n, p] unwhitened
    values: torch.Tensor  # [n]
    covariance: torch.Tensor  # [n, n]
    row_variances: torch.Tensor  # [n], declared variances used by the proxy branch
    source_ids: tuple[SourceKey, ...]
    source_ids_per_row: tuple[tuple[SourceKey, ...], ...]
    transform: CoordinateTransform
    raw_operator: torch.Tensor  # rows from original independent-identity raw values
    raw_variances: torch.Tensor


def _build_rows(
    *,
    support: SupportDomain,
    ledger: MeasurementLedger,
    used: Sequence[SourceKey],
    values: Sequence[float],
    coordinates: Sequence[tuple[float, float]],
    phases: Sequence[int | str],
    correspondence: CorrespondenceMap | InterpolationSpec | None,
    terms: Sequence[str],
    dimension: int,
    noise: NoiseModel,
    dtype: torch.dtype,
    diagnostics: dict[str, Any],
) -> dict[int | str, _BlockRows]:
    if correspondence is None:
        return _direct_rows(
            support=support,
            used=used,
            values=values,
            coordinates=coordinates,
            phases=phases,
            terms=terms,
            dimension=dimension,
            noise=noise,
            dtype=dtype,
        )
    spec = correspondence
    if not isinstance(spec, InterpolationSpec):
        if not spec.valid:
            raise ValueError(
                "an invalid correspondence carries no sampling taps; it cannot build rows"
            )
        converted = interpolation_from_correspondence(spec, used, ledger)
        spec = converted
    describe = getattr(correspondence, "describe", None)
    if callable(describe):
        diagnostics["correspondence_report"] = describe()
    if spec.source_ids is not None:
        # P2: name the columns by identity and let the fit place them in its own order.
        spec = align_interpolation_sources(spec, used)
        diagnostics["interpolation_sources_aligned"] = len(used)
    weights = spec.weights.to(dtype)
    if weights.shape[1] != len(used):
        raise ValueError(
            f"interpolation weights cover {weights.shape[1]} sources but the domain has "
            f"{len(used)} usable evidence sources"
        )
    diagnostics["interpolated"] = True
    diagnostics["interpolation_version"] = spec.version
    try:
        return _interpolated_rows(
            spec=spec,
            reference_chart=support.reference_chart,
            used=used,
            values=values,
            coordinates=coordinates,
            phases=phases,
            terms=terms,
            dimension=dimension,
            noise=noise,
            dtype=dtype,
        )
    except torch.linalg.LinAlgError as exc:
        # Duplicate/overlapping taps make A_interp Sigma A_interp^T singular.  The
        # contract keeps the independent raw observations as the factors instead
        # of inverting a singular interpolated block.
        diagnostics["singular_interpolation_fallback"] = True
        diagnostics["singular_interpolation_error"] = str(exc)[:200]
        return _direct_rows(
            support=support,
            used=used,
            values=values,
            coordinates=coordinates,
            phases=phases,
            terms=terms,
            dimension=dimension,
            noise=noise,
            dtype=dtype,
        )


def _direct_rows(
    *,
    support: SupportDomain,
    used: Sequence[SourceKey],
    values: Sequence[float],
    coordinates: Sequence[tuple[float, float]],
    phases: Sequence[int | str],
    terms: Sequence[str],
    dimension: int,
    noise: NoiseModel,
    dtype: torch.dtype,
) -> dict[int | str, _BlockRows]:
    grouped: dict[int | str, list[int]] = {}
    for index, phase in enumerate(phases):
        grouped.setdefault(phase, []).append(index)
    rows: dict[int | str, _BlockRows] = {}
    for phase, indices in grouped.items():
        block_sources = [used[index] for index in indices]
        block_positions = [coordinates[index] for index in indices]
        transform = _make_transform(block_positions, support.reference_chart, dimension)
        design = torch.tensor(
            [_basis_row(terms, transform, xy, dtype=dtype) for xy in block_positions],
            dtype=dtype,
        )
        rows[phase] = _BlockRows(
            phase=phase,
            positions=tuple(block_positions),
            design=design,
            values=torch.tensor([values[index] for index in indices], dtype=dtype),
            covariance=noise.covariance_for(block_sources, dtype=dtype),
            row_variances=torch.tensor(
                [noise.variance_of(key) for key in block_sources], dtype=dtype
            ),
            source_ids=tuple(block_sources),
            source_ids_per_row=tuple((key,) for key in block_sources),
            transform=transform,
            raw_operator=torch.eye(len(indices), dtype=dtype),
            raw_variances=torch.tensor(
                [noise.variance_of(key) for key in block_sources], dtype=dtype
            ),
        )
    return rows


def _interpolated_rows(
    *,
    spec: InterpolationSpec,
    reference_chart: str,
    used: Sequence[SourceKey],
    values: Sequence[float],
    coordinates: Sequence[tuple[float, float]],
    phases: Sequence[int | str],
    terms: Sequence[str],
    dimension: int,
    noise: NoiseModel,
    dtype: torch.dtype,
) -> dict[int | str, _BlockRows]:
    evidence_covariance = noise.covariance_for(used, dtype=dtype)
    weights = spec.weights.to(dtype)
    row_covariance = weights @ evidence_covariance @ weights.T
    # A singular row covariance is a factual duplicate-information signal.  Cholesky
    # alone is not a reliable test here (a numerically singular block can still come
    # back with a tiny positive pivot), so the eigen-spectrum is checked against a
    # declared tolerance and the documented fallback keeps the raw observations.
    eigenvalues = torch.linalg.eigvalsh(row_covariance)
    largest = float(eigenvalues.max()) if eigenvalues.numel() else 0.0
    tolerance = max(row_covariance.shape) * float(torch.finfo(dtype).eps) * largest
    if float(eigenvalues.min()) <= tolerance:
        raise torch.linalg.LinAlgError(
            "interpolated row covariance is singular "
            f"(smallest eigenvalue {float(eigenvalues.min()):.3e} <= tolerance {tolerance:.3e}); "
            "the independent raw observations are kept as factors instead"
        )

    grouped: dict[int | str, list[int]] = {}
    for row in range(len(spec.row_positions)):
        support_rows = torch.nonzero(weights[row].abs() > 0, as_tuple=False).reshape(-1)
        if support_rows.numel() == 0:
            raise ValueError(f"interpolation row {row} has no taps")
        phases_in_row = {phases[int(index)] for index in support_rows}
        if len(phases_in_row) != 1:
            raise ValueError(
                "an interpolated row mixes phases; phases are fitted separately and are "
                "never averaged first"
            )
        grouped.setdefault(phases_in_row.pop(), []).append(row)

    rows: dict[int | str, _BlockRows] = {}
    value_tensor = torch.tensor(values, dtype=dtype)
    for phase, row_index in grouped.items():
        positions = tuple(spec.row_positions[row] for row in row_index)
        transform = _make_transform(positions, reference_chart, dimension)
        evidence_basis = torch.tensor(
            [_basis_row(terms, transform, xy, dtype=dtype) for xy in coordinates], dtype=dtype
        )
        block_weights = weights[row_index]
        design = block_weights @ evidence_basis
        row_values = block_weights @ value_tensor
        block_covariance = row_covariance[row_index][:, row_index]
        rows[phase] = _BlockRows(
            phase=phase,
            positions=positions,
            design=design,
            values=row_values,
            covariance=block_covariance,
            row_variances=torch.diagonal(block_covariance),
            source_ids=tuple(used),
            source_ids_per_row=tuple(_nonzero_sources(weights[row], used) for row in row_index),
            transform=transform,
            raw_operator=block_weights,
            raw_variances=torch.tensor([noise.variance_of(key) for key in used], dtype=dtype),
        )
    return rows


def _nonzero_sources(weights_row: torch.Tensor, used: Sequence[SourceKey]) -> tuple[SourceKey, ...]:
    return tuple(
        used[index]
        for index in torch.nonzero(weights_row.abs() > 0, as_tuple=False).reshape(-1).tolist()
    )


def _make_transform(
    positions: Sequence[tuple[float, float]], chart: str, dimension: int
) -> CoordinateTransform:
    tensor = torch.tensor(positions, dtype=torch.float64)
    center = tensor.mean(dim=0)
    spread = tensor.std(dim=0, unbiased=False)
    scale = torch.where(spread > 0, spread, torch.ones_like(spread))
    if dimension == 1:
        scale[1] = 1.0
        center[1] = tensor[:, 1].mean()
    return CoordinateTransform(
        center=(float(center[0]), float(center[1])),
        scale=(float(scale[0]), float(scale[1])),
        chart=chart,
    )


def _solve_block(
    *,
    phase: int | str,
    data: _BlockRows,
    terms: Sequence[str],
    noise: NoiseModel,
    prior: GaussianPrior | None,
    dtype: torch.dtype,
    rank_rtol: float | None,
    leverage_tol: float,
    diagnostics: dict[str, Any],
) -> BlockFit:
    design = data.design
    values = data.values
    covariance = data.covariance
    if prior is not None and len(terms) != prior.mean.shape[0]:
        raise ValueError(
            f"the declared prior has {prior.mean.shape[0]} parameters but the basis has "
            f"{len(terms)}; a prior cannot be silently reshaped"
        )
    if prior is not None and noise.kind == "unknown_correlation":
        raise ValueError(
            "the conservative fixed-weight bound has no declared parameter-space prior; "
            "adding one silently would mix a proxy label with a posterior covariance"
        )

    cholesky = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(cholesky, design, upper=False)
    target = torch.linalg.solve_triangular(cholesky, values.unsqueeze(1), upper=False).squeeze(1)

    singular_values = torch.linalg.svdvals(whitened)
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = (rank_rtol or max(design.shape) * float(torch.finfo(dtype).eps)) * largest
    rank = int((singular_values > tolerance).sum())
    condition_number = (
        float("inf")
        if singular_values.numel() == 0 or float(singular_values[-1]) <= 0.0
        else float(singular_values[0] / singular_values[-1])
    )

    full_rank = rank == len(terms)
    raw_noise_weights = None
    # Use the same rank tolerance for the coefficient estimate and identifiability.
    rcond = rank_rtol if rank_rtol is not None else max(design.shape) * torch.finfo(dtype).eps
    theta = torch.linalg.lstsq(whitened, target, rcond=rcond, driver="gelsd").solution
    if noise.kind == "unknown_correlation":
        # Compose weights all the way back to RAW before applying the scalar bound.
        inverse_cholesky = torch.linalg.solve_triangular(
            cholesky, torch.eye(len(values), dtype=dtype), upper=False
        )
        raw_weights = torch.linalg.pinv(whitened, rtol=rcond) @ inverse_cholesky @ data.raw_operator
        if full_rank:
            raw_noise_weights = raw_weights * data.raw_variances.sqrt()[None, :]
            scale = raw_noise_weights.abs().sum(dim=1)
            # Cauchy-Schwarz: (sum_j |q_j| s_j)^2 <= p sum_j q_j^2 s_j^2.
            covariance_matrix = len(terms) * torch.diag(scale.square())
        else:
            covariance_matrix = None
    elif prior is not None:
        prior_mean, prior_covariance = _prior_in_chart(prior, data.transform, terms, dtype)
        prior_factor = torch.linalg.cholesky(prior_covariance)
        prior_whitener = torch.linalg.solve_triangular(
            prior_factor, torch.eye(len(terms), dtype=dtype), upper=False
        )
        augmented = torch.cat([whitened, prior_whitener])
        augmented_target = torch.cat([target, prior_whitener @ prior_mean])
        theta = torch.linalg.lstsq(augmented, augmented_target, driver="gelsd").solution
        _, triangular = torch.linalg.qr(augmented, mode="reduced")
        inverse_triangular = torch.linalg.solve_triangular(
            triangular, torch.eye(len(terms), dtype=dtype), upper=True
        )
        covariance_matrix = inverse_triangular @ inverse_triangular.T
    elif full_rank:
        # QR triangular solves avoid squaring the design condition number in Lambda.
        _, triangular = torch.linalg.qr(whitened, mode="reduced")
        inverse_triangular = torch.linalg.solve_triangular(
            triangular, torch.eye(len(terms), dtype=dtype), upper=True
        )
        covariance_matrix = inverse_triangular @ inverse_triangular.T
    else:
        covariance_matrix = None

    _, _, right_vectors = torch.linalg.svd(whitened, full_matrices=False)
    # Coordinate j is identifiable iff the unit vector e_j lies in the row space of
    # the design: the squared norm of column j of V restricted to the rank rows.
    leverage = (right_vectors[:rank, :] ** 2).sum(dim=0)
    identifiable = leverage >= 1.0 - leverage_tol
    diagnostics.setdefault("leverage", {})[str(phase)] = [float(item) for item in leverage]

    return BlockFit(
        phase=phase,
        parameter_names=tuple(terms),
        theta=theta,
        covariance=covariance_matrix,
        rank=rank,
        condition_number=condition_number,
        identifiable=identifiable,
        observation_count=int(design.shape[0]),
        source_ids=tuple(dict.fromkeys(data.source_ids)),
        transform=data.transform,
        row_projector=right_vectors[:rank].T @ right_vectors[:rank],
        prior_supplied=prior is not None,
        raw_noise_weights=raw_noise_weights,
        raw_estimator_weights=(
            torch.linalg.pinv(whitened, rtol=rcond)
            @ torch.linalg.solve_triangular(cholesky, data.raw_operator, upper=False)
        ),
    )


def _prior_in_chart(prior, transform, terms, dtype):
    """Re-express a fixed physical prior when support centering/scaling changes."""
    change = torch.eye(len(terms), dtype=dtype)
    if len(terms) > 1:
        declared = prior.transform
        if declared.chart != transform.chart or declared.units != transform.units:
            raise ValueError("prior and observations must use the same physical coordinate chart")
        for column, name in enumerate(terms[1:], 1):
            axis = 0 if name == "bx" else 1
            change[0, column] = (transform.center[axis] - declared.center[axis]) / declared.scale[
                axis
            ]
            change[column, column] = transform.scale[axis] / declared.scale[axis]
    return change @ prior.mean.to(dtype), change @ prior.covariance.to(dtype) @ change.T


def align_interpolation_sources(
    spec: InterpolationSpec, used: Sequence[SourceKey]
) -> InterpolationSpec:
    """Re-index ``spec`` columns onto ``used`` by source identity.

    P2 builders name their columns explicitly (``source_ids``) because a redacted
    ledger view can differ from the domain's evidence set.  A nonzero weight on a
    column that the domain does not carry is refused rather than silently dropped
    or renormalised: that would change what the row measures.
    """
    if spec.source_ids is None:
        return spec
    index = {key.identity(): position for position, key in enumerate(used)}
    columns: list[int | None] = []
    for key in spec.source_ids:
        columns.append(index.get(key.identity()))
    dropped_nonzero = [
        key
        for column, key in enumerate(spec.source_ids)
        if columns[column] is None and bool(spec.weights[:, column].abs().max() > 0)
    ]
    if dropped_nonzero:
        raise ValueError(
            "interpolation rows carry nonzero weight on sources outside the domain's usable "
            f"evidence set: {[key.identity() for key in dropped_nonzero]}"
        )
    weights = torch.zeros(
        (spec.weights.shape[0], len(used)), dtype=spec.weights.dtype, device=spec.weights.device
    )
    for column, position in enumerate(columns):
        if position is not None:
            weights[:, position] = spec.weights[:, column]
    return InterpolationSpec(
        row_positions=spec.row_positions,
        weights=weights,
        version=spec.version,
        source_ids=tuple(used),
    )


def interpolation_from_correspondence(
    correspondence: CorrespondenceMap,
    evidence_ids: Sequence[SourceKey],
    ledger: MeasurementLedger | None = None,
) -> InterpolationSpec:
    """Convert a one-tap-per-row :class:`CorrespondenceMap` into an explicit spec.

    A correspondence with several taps per reference row is not convertible here:
    the contract type stores taps flat, so the row grouping would be a guess.  P2
    builds multi-tap maps through :class:`InterpolationSpec` explicitly instead.
    """
    del ledger
    positions = torch.as_tensor(correspondence.source_xy, dtype=torch.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"CorrespondenceMap.source_xy must be [n_rows, 2] pull coordinates, got "
            f"{tuple(positions.shape)}"
        )
    if len(correspondence.tap_source_ids) != positions.shape[0]:
        raise ValueError(
            "P1 converts only a one-tap-per-row correspondence; this map has "
            f"{len(correspondence.tap_source_ids)} taps for {positions.shape[0]} rows"
        )
    index = {key.identity(): position for position, key in enumerate(evidence_ids)}
    weights = torch.zeros((positions.shape[0], len(evidence_ids)), dtype=torch.float64)
    for row, (tap, weight) in enumerate(
        zip(correspondence.tap_source_ids, correspondence.tap_weights, strict=True)
    ):
        position = index.get(tap.identity())
        if position is None:
            raise ValueError(
                f"correspondence tap {tap.identity()} is not part of the domain's evidence set"
            )
        weights[row, position] = float(weight)
    return InterpolationSpec(
        row_positions=tuple((float(x), float(y)) for x, y in positions),
        weights=weights,
        version=f"correspondence-{correspondence.mapping_version}",
    )

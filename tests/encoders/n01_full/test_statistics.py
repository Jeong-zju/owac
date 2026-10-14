"""Numeric and behavioural properties of fixed-design structure fits.

The oracle used throughout is ``reference.py`` (independent NumPy FP64 dense
least squares), never the production solver.  The counterexample of
``P1_ANALYTIC_REFERENCE.md`` section 4.2 -- slicing the old precision matrix --
is reproduced here and must stay rejected.
"""

import math

import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full import reference as reference_module
from owac.frameworks.n01_full.contracts import SourceKey
from owac.frameworks.n01_full.domains import make_domain, retract_sources
from owac.frameworks.n01_full.ledger import MeasurementLedger
from owac.frameworks.n01_full.statistics import (
    GaussianPrior,
    InterpolationSpec,
    NoiseModel,
    fit_structure,
)

DATASET = "stats"

FP64_ATOL, FP64_RTOL = 1e-10, 1e-8
FP32_ATOL, FP32_RTOL = 1e-5, 1e-4


def make_ledger(
    values, *, height=1, width=None, offsets=None, valid=None, bad=None, saturated=None
):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.reshape(1, 1, *tensor.shape)
    ledger = MeasurementLedger(DATASET, value_units="unit")
    ledger.observe(
        MeasurementFrame(
            values=tensor,
            valid_mask=None if valid is None else torch.as_tensor(valid).reshape(tensor.shape),
            bad_mask=None if bad is None else torch.as_tensor(bad).reshape(tensor.shape),
            saturated_mask=(
                None if saturated is None else torch.as_tensor(saturated).reshape(tensor.shape)
            ),
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
            sample_xy=None if offsets is None else torch.as_tensor(offsets, dtype=torch.float64),
        )
    )
    return ledger


def line_keys(length, *, frame_id=0, channel=0):
    return [
        SourceKey(DATASET, "e0", "c0", frame_id, channel, 0, position) for position in range(length)
    ]


def constant_domain(keys, *, domain_id="D", phase=0):
    return make_domain(
        domain_id,
        reference_chart="ref",
        basis="phase_local_constant",
        phase=phase,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
    )


def affine_domain(keys, *, domain_id="D", phase=None):
    return make_domain(
        domain_id,
        reference_chart="ref",
        basis="phase_local_affine",
        phase=phase,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
    )


def oracle_for(ledger, domain, variances, *, basis="affine", dimension=2):
    observations = []
    for key in domain.evidence_source_ids:
        if not ledger.has_source(key):
            continue
        record = ledger.get_source(key)
        if record.validity_kind != "valid":
            continue
        observations.append(
            reference_module.ReferenceObservation(
                source_id=key,
                phase=key.channel_or_phase,
                xy=ledger.sample_coordinates(key),
                value=float(record.raw_value),
                variance=float(variances[key]),
            )
        )
    return reference_module.reference_fit(observations, basis=basis, dimension=dimension)


def test_two_observations_mean_and_variance():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    domain = constant_domain(keys)
    fit = fit_structure(domain, ledger, None, NoiseModel.diagonal(dict.fromkeys(keys, 1.0)))
    assert float(fit.estimate.theta[0]) == pytest.approx(2.0, abs=FP64_ATOL)
    assert float(fit.estimate.covariance[0, 0]) == pytest.approx(0.5, abs=FP64_ATOL)
    assert fit.estimate.independent_observation_count == 2
    assert fit.estimate.uncertainty_kind == "covariance"


def test_correlated_observations_and_precision_slice_counterexample():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    domain = constant_domain(keys)
    noise = NoiseModel.dense(keys, [[1.0, 0.5], [0.5, 1.0]])
    fit = fit_structure(domain, ledger, None, noise)
    assert float(fit.estimate.theta[0]) == pytest.approx(2.0, abs=FP64_ATOL)
    assert float(fit.estimate.covariance[0, 0]) == pytest.approx(0.75, abs=FP64_ATOL)

    remaining = retract_sources(domain, [keys[1]])
    after = fit_structure(remaining, ledger, None, noise)
    assert float(after.estimate.theta[0]) == pytest.approx(1.0, abs=FP64_ATOL)
    assert float(after.estimate.covariance[0, 0]) == pytest.approx(1.0, abs=FP64_ATOL)

    wrong = float(1.0 / torch.linalg.inv(noise.matrix)[0, 0])
    assert wrong == pytest.approx(0.75, abs=FP64_ATOL)
    assert abs(wrong - 1.0) > 1e-6, "the precision-slice version must not be accepted"


def test_production_matches_independent_whitened_least_squares():
    height, width = 3, 4
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(width, dtype=torch.float64),
            torch.arange(height, dtype=torch.float64),
            indexing="xy",
        ),
        dim=-1,
    )
    generator = torch.Generator().manual_seed(23)
    field = 1.5 + 0.75 * coordinates[..., 0] - 0.25 * coordinates[..., 1]
    spread = torch.linspace(0.05, 0.3, height * width, dtype=torch.float64).reshape(height, width)
    values = field + spread * torch.randn((height, width), generator=generator)
    offsets = (coordinates + 0.4).reshape(1, 1, height, width, 2)
    ledger = make_ledger(values, offsets=offsets)
    keys = [SourceKey(DATASET, "e0", "c0", 0, 0, y, x) for y in range(height) for x in range(width)]
    variances = {key: float(spread[key.sensor_y, key.sensor_x] ** 2) for key in keys}
    domain = affine_domain(keys)
    fit = fit_structure(domain, ledger, None, NoiseModel.diagonal(variances))
    oracle = oracle_for(ledger, domain, variances)
    assert fit.condition_number < 1e3
    for x, y in ((0.0, 0.0), (1.5, 1.5), (3.0, 2.0), (2.25, 0.75)):
        mean, variance = fit.predict(0, (x, y))
        oracle_mean, oracle_variance = oracle.predict(0, (x, y))
        assert mean == pytest.approx(oracle_mean, abs=FP64_ATOL, rel=FP64_RTOL)
        assert variance == pytest.approx(oracle_variance, abs=FP64_ATOL, rel=FP64_RTOL)


def test_fp32_well_conditioned_problem_meets_fp32_tolerance():
    ledger = make_ledger([1.0, 2.0, 3.0, 4.0])
    keys = line_keys(4)
    variances = {key: 1.0 for key in keys}
    domain = affine_domain(keys)
    fit64 = fit_structure(domain, ledger, None, NoiseModel.diagonal(variances), dimension=1)
    fit32 = fit_structure(
        domain, ledger, None, NoiseModel.diagonal(variances), dimension=1, dtype=torch.float32
    )
    assert fit32.estimate.theta.dtype == torch.float32
    for x in (0.0, 1.0, 2.5, 3.0):
        mean64, variance64 = fit64.predict(0, (x, 0.0))
        mean32, variance32 = fit32.predict(0, (x, 0.0))
        assert mean32 == pytest.approx(mean64, abs=FP32_ATOL, rel=FP32_RTOL)
        assert variance32 == pytest.approx(variance64, abs=FP32_ATOL, rel=FP32_RTOL)


def test_ill_conditioned_problem_reports_condition_number_instead_of_hiding_it():
    # Three nearly collinear points: the affine basis is full rank but the design
    # is ill-conditioned, so the condition number must be reported rather than
    # hidden behind a loosened global tolerance.
    coordinates = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0 + 1e-9)]
    # The storage grid is 1 x 3 (H x W); the declared sampling coordinates carry
    # the two-dimensional geometry.
    keys = [SourceKey(DATASET, "e0", "c0", 0, 0, 0, position) for position in range(3)]
    ledger = MeasurementLedger(DATASET, value_units="unit")
    ledger.observe(
        MeasurementFrame(
            values=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64).reshape(1, 1, 1, 3),
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
            sample_xy=torch.tensor(coordinates, dtype=torch.float64).reshape(1, 1, 1, 3, 2),
        )
    )
    domain = affine_domain(keys)
    variances = {key: 1.0 for key in keys}
    fit = fit_structure(domain, ledger, None, NoiseModel.diagonal(variances))
    assert fit.rank == fit.parameter_count
    assert math.isfinite(fit.condition_number)
    assert fit.condition_number > 1e6
    oracle = oracle_for(ledger, domain, variances)
    production_mean, production_variance = fit.predict(0, (1.0, 1.0))
    oracle_mean, _ = oracle.predict(0, (1.0, 1.0))
    # Declared loose tolerance for this ill-conditioned case only; the global
    # default stays at FP64 and the condition number is reported alongside.
    assert production_mean == pytest.approx(oracle_mean, abs=1e-6, rel=1e-6)
    # The predictive variance of an ill-conditioned design has no accurate digits;
    # it is reported together with the condition number instead of being compared
    # under a silently widened tolerance.
    assert math.isfinite(production_variance)
    assert production_variance > 0.0


def test_missing_bad_saturated_never_enter_the_design_and_context_never_does_either():
    values = [10.0, 1.0, 999.0, 3.0, 55.0]
    ledger = make_ledger(
        values,
        valid=[False, True, False, True, True],
        bad=[False, False, True, False, False],
        saturated=[False, False, False, False, True],
    )
    keys = line_keys(5)
    variances = {key: 1.0 for key in keys}
    everything = constant_domain(keys)
    fit = fit_structure(everything, ledger, None, NoiseModel.diagonal(variances))
    assert fit.estimate.independent_observation_count == 2
    assert {reason for _, reason in fit.excluded_sources} == {"missing", "bad", "saturated"}
    assert float(fit.estimate.theta[0]) == pytest.approx(2.0, abs=FP64_ATOL)

    usable = constant_domain([keys[1], keys[3]], domain_id="U")
    clean = fit_structure(usable, ledger, None, NoiseModel.diagonal(variances))
    assert float(clean.estimate.theta[0]) == pytest.approx(2.0, abs=FP64_ATOL)

    # A context source with an absurd value must not move the estimate.
    with_context = make_domain(
        "C",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=[keys[1], keys[3]],
        context_source_ids=[keys[4]],
        decision_version="dv1",
        mapping_version="mv1",
    )
    contextual = fit_structure(with_context, ledger, None, NoiseModel.diagonal(variances))
    assert float(contextual.estimate.theta[0]) == pytest.approx(2.0, abs=FP64_ATOL)
    assert contextual.estimate.independent_observation_count == 2
    assert contextual.diagnostics["context_used_as_evidence"] is False


def test_all_invalid_domain_returns_an_invalid_mask():
    ledger = make_ledger([1.0], valid=[False])
    keys = line_keys(1)
    domain = constant_domain(keys)
    fit = fit_structure(domain, ledger, None, NoiseModel.diagonal({keys[0]: 1.0}))
    assert fit.rank == 0
    assert fit.estimate.theta is None
    assert fit.estimate.uncertainty_kind == "none"
    assert fit.estimate.covariance is None
    assert not bool(fit.estimate.identifiable_mask.any())
    assert fit.estimate.independent_observation_count == 0


def test_rank_deficiency_reports_the_identifiable_subspace_without_fake_precision():
    ledger = make_ledger(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    keys = [SourceKey(DATASET, "e0", "c0", 0, 0, y, 0) for y in range(2)]
    domain = affine_domain(keys)
    fit = fit_structure(domain, ledger, None, NoiseModel.diagonal(dict.fromkeys(keys, 1.0)))
    assert fit.rank == 2 < fit.parameter_count
    assert fit.estimate.uncertainty_kind == "none"
    assert fit.estimate.covariance is None
    # e_0 and e_2 span the row space; the x slope is not determined.
    assert [bool(item) for item in fit.estimate.identifiable_mask] == [True, False, True]
    assert fit.diagnostics["leverage"]["0"][1] < 0.5


def test_prior_is_counted_once_and_not_as_an_observation():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    domain = constant_domain(keys)
    noise = NoiseModel.diagonal(dict.fromkeys(keys, 1.0))
    plain = fit_structure(domain, ledger, None, noise)
    prior = GaussianPrior(
        prior_id="prior-v1",
        mean=torch.tensor([5.0], dtype=torch.float64),
        covariance=torch.tensor([[1.0]], dtype=torch.float64),
    )
    with_prior = fit_structure(domain, ledger, None, noise, prior=prior)
    assert with_prior.estimate.prior_id == "prior-v1"
    assert with_prior.estimate.independent_observation_count == 2
    assert plain.estimate.prior_id is None
    # posterior precision = 2 observations + 1 prior; mean is pulled towards 5.
    assert float(with_prior.estimate.theta[0]) == pytest.approx(
        (2.0 * 2 + 5.0) / 3.0, abs=FP64_ATOL
    )
    assert float(with_prior.estimate.covariance[0, 0]) == pytest.approx(1.0 / 3.0, abs=FP64_ATOL)


def test_unknown_correlation_bound_is_conservative_and_separately_labelled():
    ledger = make_ledger([1.0, 3.0, 2.0])
    keys = line_keys(3)
    domain = constant_domain(keys)
    noise = NoiseModel.unknown_correlation(dict.fromkeys(keys, 0.5))
    fit = fit_structure(domain, ledger, None, noise)
    assert fit.estimate.uncertainty_kind == "proxy"
    bound = float(fit.estimate.covariance[0, 0])

    generator = torch.Generator().manual_seed(29)
    draws = 4000
    # Marginal variance 0.5 per sample with equicorrelation 0.5, matching the
    # declared variance exactly: shared and idiosyncratic each contribute 0.25.
    sigma = math.sqrt(0.25)
    shared = torch.randn(draws, generator=generator) * sigma
    idiosyncratic = torch.randn((draws, 3), generator=generator) * sigma
    samples = shared[:, None] + idiosyncratic + torch.tensor([1.0, 3.0, 2.0])
    empirical = float(samples.mean(dim=1).var(unbiased=True))
    independent = float((torch.tensor([1.0, 1.0, 1.0]) / 3).square().sum()) * 0.5
    assert empirical <= bound
    assert independent < bound, "the independent-observation formula is not the conservative bound"


def test_interpolation_row_covariance_is_A_sigma_A_transpose_and_singular_falls_back():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    domain = constant_domain(keys)
    noise = NoiseModel.dense(keys, [[1.0, 0.5], [0.5, 1.0]])
    single = InterpolationSpec(
        row_positions=((0.5, 0.0),), weights=torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    )
    fit = fit_structure(domain, ledger, single, noise)
    # w^T Sigma w = 0.25*1 + 0.25*1 + 2*0.25*0.5 = 0.75 for a constant basis.
    assert float(fit.estimate.covariance[0, 0]) == pytest.approx(0.75, abs=FP64_ATOL)
    assert fit.diagnostics["interpolated"] is True

    duplicate = InterpolationSpec(
        row_positions=((0.0, 0.0), (1.0, 0.0)),
        weights=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64),
    )
    singular = fit_structure(domain, ledger, duplicate, noise)
    assert singular.diagnostics["singular_interpolation_fallback"] is True
    # The fallback keeps the independent raw observations as the factors.
    assert singular.estimate.independent_observation_count == 2
    assert "singular" in singular.diagnostics["singular_interpolation_error"]


def test_phases_are_fitted_separately_and_never_averaged_first():
    tensor = torch.tensor([[[[1.0, 1.0, 1.0]], [[5.0, 5.0, 5.0]]]], dtype=torch.float64)
    ledger = MeasurementLedger(DATASET, value_units="unit")
    ledger.observe(
        MeasurementFrame(
            values=tensor,
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
        )
    )
    keys = [
        SourceKey(DATASET, "e0", "c0", 0, channel, 0, x) for channel in (0, 1) for x in range(3)
    ]
    domain = make_domain(
        "P",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=None,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
    )
    fit = fit_structure(domain, ledger, None, NoiseModel.diagonal(dict.fromkeys(keys, 1.0)))
    assert fit.predict(0, (0.0, 0.0))[0] == pytest.approx(1.0, abs=FP64_ATOL)
    assert fit.predict(1, (0.0, 0.0))[0] == pytest.approx(5.0, abs=FP64_ATOL)
    assert set(fit.blocks) == {0, 1}
    assert float(torch.cat([fit.blocks[0].theta, fit.blocks[1].theta]).mean()) == pytest.approx(
        3.0, abs=FP64_ATOL
    )

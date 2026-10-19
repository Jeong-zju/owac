"""Explicit two-hypothesis measurement update checked against exact Bayes.

These tests are the numerical floor of N02: the same scoring rule the learned
field uses is exercised on a problem whose posterior can be written down, so a
regression in the energy convention cannot hide behind a training curve.
"""

import math

import pytest
import torch

from owac.frameworks.n02.analytic import (
    AnalyticCase,
    AnalyticGeometry,
    AnalyticObservation,
    analytic_energy,
    analytic_posterior,
    build_buffer,
    delayed_evidence_case,
    forward_operator,
    make_config,
    repeated_observation_check,
)
from owac.frameworks.n02.hypotheses import neighbour_validity
from owac.frameworks.n02.revision import RevisionController
from owac.frameworks.n02.state import HypothesisState

TOL = 1e-9


def hand_computed_posterior(prior_odds, residuals, variance):
    """Closed-form posterior for two hypotheses, independent of the implementation."""
    log_prior = [math.log(p / sum(prior_odds)) for p in prior_odds]
    logits = [
        log_prior[index] - sum(0.5 * residual**2 / variance for residual in residuals[index])
        for index in (0, 1)
    ]
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def test_forward_operator_is_the_declared_mirror_and_bump():
    geometry = AnalyticGeometry(kind="opening_direction")
    coordinates = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    left = forward_operator(0, coordinates, geometry)
    right = forward_operator(1, coordinates, geometry)
    # The two hypotheses predict mirrored raw profiles.
    torch.testing.assert_close(left.flip(0), right)
    handle = AnalyticGeometry(kind="handle_side")
    bump_left = forward_operator(0, coordinates, handle)
    bump_right = forward_operator(1, coordinates, handle)
    assert bump_left[1] > bump_left[0] and bump_right[3] > bump_right[4]


@pytest.mark.parametrize("value", [0.4, 0.6, 0.8, 1.1])
@pytest.mark.parametrize("prior_odds", [(1.0, 1.0), (3.0, 1.0), (0.02, 0.98)])
def test_posterior_matches_closed_form_bayes(value, prior_odds):
    config = make_config()
    geometry = AnalyticGeometry(kind="opening_direction")
    observation = AnalyticObservation(
        value=value, coordinate=0.25, variance=0.01, identity=("cam0", 0, 0)
    )
    case = AnalyticCase(
        config=config,
        geometry=geometry,
        prior_odds=prior_odds,
        observations=[observation],
        true_hypothesis=1,
    )
    weights, diagnostics = analytic_posterior(case, build_buffer(case), config)
    coordinates = torch.tensor([0.25], dtype=torch.float64)
    residuals = [
        value - float(forward_operator(hypothesis, coordinates, geometry).to(torch.float64).item())
        for hypothesis in (0, 1)
    ]
    expected = hand_computed_posterior(prior_odds, [[residuals[0]], [residuals[1]]], 0.01)
    assert diagnostics["observations_used"] == 1
    for actual, want in zip(diagnostics["weights"], expected, strict=True):
        assert actual == pytest.approx(want, abs=1e-9)


def test_repeated_observation_does_not_double_the_likelihood():
    config = make_config()
    case = AnalyticCase(
        config=config,
        geometry=AnalyticGeometry(),
        prior_odds=(3.0, 1.0),
        observations=[
            AnalyticObservation(value=0.8, coordinate=0.25, variance=0.01, identity=("cam0", 0, 0))
        ],
    )
    report = repeated_observation_check(case, config)
    assert report["unique_evidence_single"] == 1
    assert report["unique_evidence_duplicated"] == 1
    assert report["duplicates_recorded"] == 1
    assert report["posterior_identical"] is True
    assert report["measurement_energy_single"] == report["measurement_energy_duplicated"]


def test_new_contradicting_measurement_overturns_a_strong_prior():
    config = make_config()
    geometry = AnalyticGeometry(kind="opening_direction")
    misleading = AnalyticCase(
        config=config,
        geometry=geometry,
        prior_odds=(0.02, 0.98),
        true_hypothesis=1,
        observations=[
            AnalyticObservation(value=0.8, coordinate=0.25, variance=0.01, identity=("cam0", 0, 0))
        ],
        name="misleading_prior",
    )
    # The strong but wrong prior favours "left" 49:1; the evidence says "right"
    # (z = |0.8 - 0.6| / 0.1 = 2 sigma), and 2 sigma must not be dismissed just
    # because the prior was confident.
    weights, diagnostics = analytic_posterior(misleading, build_buffer(misleading), config)
    assert diagnostics["weights"][1] > 0.99
    assert weights[0, 1] > 0.99
    assert weights[0, 1] == pytest.approx(misleading.exact_posterior()[1], abs=TOL)
    # A weak prior correction must not be produced by zeroing the prior: with the
    # prior switched off the same evidence is only a 2-sigma preference.
    uninformative, _ = analytic_posterior(
        misleading, build_buffer(misleading), config, beta_prior=0.0
    )
    assert uninformative[0, 1] < 0.99


def test_unidentifiable_sequence_keeps_the_ambiguity():
    config = make_config()
    ambiguous = delayed_evidence_case(config, delay=3, seed=7, misidentify_coordinate=True)
    weights, diagnostics = analytic_posterior(ambiguous, build_buffer(ambiguous), config)
    assert ambiguous.exact_evidence_count() == 4
    assert abs(diagnostics["weights"][0] - diagnostics["weights"][1]) < 0.05


def test_delayed_cue_arrival_is_not_a_fixed_frame_index():
    config = make_config()
    arrivals = set()
    for seed in range(8):
        case = delayed_evidence_case(config, delay=(seed % 5) + 1, seed=seed)
        assert case.evidence_step == (seed % 5) + 1
        arrivals.add(case.evidence_step)
    assert len(arrivals) >= 3, arrivals


def test_energy_matches_the_gaussian_nll_definition():
    config = make_config()
    geometry = AnalyticGeometry(kind="opening_direction")
    case = AnalyticCase(
        config=config,
        geometry=geometry,
        prior_odds=(1.0, 1.0),
        observations=[
            AnalyticObservation(value=0.7, coordinate=0.25, variance=0.04, identity=("cam0", 0, 0)),
            AnalyticObservation(value=0.5, coordinate=0.75, variance=0.04, identity=("cam0", 1, 0)),
        ],
    )
    energy, prior = analytic_energy(case, build_buffer(case), config)
    expected = []
    for hypothesis in (0, 1):
        total = 0.0
        for observation in case.observations:
            predicted = float(
                forward_operator(
                    hypothesis,
                    torch.tensor([observation.coordinate], dtype=torch.float64),
                    geometry,
                )
                .to(torch.float64)
                .item()
            )
            # Gaussian NLL: 0.5 * r^2 / v + 0.5 * log(v). The log term is constant
            # across candidates for a fixed measurement set, but it is part of the
            # declared energy and is checked here.
            total += 0.5 * (
                observation.value - predicted
            ) ** 2 / observation.variance + 0.5 * math.log(observation.variance)
        expected.append(total)
    assert energy[0, 0].item() == pytest.approx(expected[0], abs=TOL)
    assert energy[0, 1].item() == pytest.approx(expected[1], abs=TOL)
    assert prior[0, 0].item() == pytest.approx(0.5)


def test_saturated_observation_is_excluded_not_treated_as_exact():
    config = make_config()
    geometry = AnalyticGeometry(kind="opening_direction")
    case = AnalyticCase(
        config=config,
        geometry=geometry,
        prior_odds=(1.0, 1.0),
        observations=[
            AnalyticObservation(
                value=1.0, coordinate=0.25, variance=0.01, identity=("cam0", 0, 0), saturated=True
            )
        ],
    )
    weights, diagnostics = analytic_posterior(case, build_buffer(case), config)
    assert diagnostics["observations_used"] == 1  # the frame is retained for audit
    assert case.exact_evidence_count() == 0
    assert weights[0, 0].item() == pytest.approx(0.5)


def test_contradiction_and_revision_run_on_the_explicit_problem():
    """The real RevisionController must single out the refuted candidate."""
    config = make_config(K=2, cycles=4)
    revision = RevisionController(config, width=config.D)
    batch, views, K, width, height, width_px = 1, 1, 2, config.D, 1, 1
    valid = neighbour_validity(height, width_px, torch.device("cpu")).view(
        batch, views, height, width_px, 4
    )
    edge = torch.zeros(batch, views, height, width_px, 4, K, K) + torch.eye(K).view(
        1, 1, 1, 1, 1, K, K
    )
    geometry = AnalyticGeometry(kind="opening_direction")
    case = AnalyticCase(
        config=config,
        geometry=geometry,
        prior_odds=(3.0, 1.0),
        true_hypothesis=1,
        observations=[
            AnalyticObservation(value=1.1, coordinate=0.25, variance=0.01, identity=("cam0", 0, 0))
        ],
    )
    residuals = torch.tensor([[5.0], [3.0]]).view(batch, K, height, width_px)
    weights, diagnostics = analytic_posterior(case, build_buffer(case), config)
    state = HypothesisState(
        h=torch.zeros(batch, views, K, width, height, width_px),
        measurement_energy=torch.zeros(batch, views, K),
        weights=torch.tensor(diagnostics["weights"]).view(batch, views, K),
        prior_energy=torch.zeros(batch, views, K),
        relation_energy=torch.zeros(batch, views, K),
        total_energy=torch.zeros(batch, views, K),
        logits=torch.zeros(batch, views, K),
        geometry=torch.zeros(batch, views, K, config.measurement.geometry_dim),
        age=torch.zeros(batch, views, K),
        times_reset=torch.zeros(batch, views, K),
        dependency_valid=valid,
        dependency_edge=edge,
    )
    output = revision.step(
        state,
        residuals.unsqueeze(1),
        residual_feature=residuals.view(batch, views, K, height, width_px),
        prior_feature=torch.zeros(batch, views, K, height, width_px),
        location_code=torch.zeros(batch, views, 4, height, width_px),
        dependency_valid=valid,
        iteration=0,
        cold_start=False,
    )
    contradicted = output.contradiction.flatten().tolist()
    assert contradicted == [True, False]
    assert output.reset.flatten().tolist() == [True, False]
    assert float(output.h.abs().sum().item()) != 0.0  # the proposal rebuilt the cell


def test_no_contradiction_when_every_candidate_fits_the_noise_model():
    config = make_config(K=2)
    revision = RevisionController(config, width=config.D)
    score = torch.tensor([[[[[0.5]], [[1.0]]]]])
    from owac.frameworks.n02.state import RevisionState

    flags = revision.contradiction_map(score, revision=RevisionState())
    assert not bool(flags.any())
    suspect = revision.all_candidates_bad(torch.tensor([[[[[3.0]], [[3.5]]]]]))
    assert bool(suspect.all())


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        forward_operator(2, torch.tensor([0.5]), AnalyticGeometry())
    with pytest.raises(ValueError):
        AnalyticCase(
            config=make_config(),
            geometry=AnalyticGeometry(),
            prior_odds=(0.0, 0.0),
            observations=[],
        )
    with pytest.raises(ValueError):
        AnalyticGeometry(kind="not_a_geometry").names()
